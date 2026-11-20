//! CSTRACE v1 reader (format: docs/VERIFICATION_AND_BENCHMARKS.md, "Format"; JavaScript
//! reference: `TraceReader` in scripts/trace_lib.mjs). Everything is little-endian.

use crate::json::{self, Json};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"CSTRACE\0";
pub const VERSION_MAJOR: u16 = 1;

pub const TAG_WRITE: u8 = 0x57;
pub const TAG_UNMAPPED: u8 = 0x55;
pub const TAG_RESET: u8 = 0x52;
pub const TAG_CALL: u8 = 0x43;
pub const TAG_END: u8 = 0x45;

/// Function flag bit 0: the export returns a pointer (an address, compared only on the same wasm).
pub const FFLAG_POINTER_RESULT: u8 = 1;
/// Region flag bits.
pub const RFLAG_HOST_WRITABLE: u8 = 1;
pub const RFLAG_DIGESTED: u8 = 2;
pub const RFLAG_ALIAS: u8 = 4;

/// Wasm value type codes.
pub const VT_I32: u8 = 0x7f;
pub const VT_I64: u8 = 0x7e;
pub const VT_F32: u8 = 0x7d;
pub const VT_F64: u8 = 0x7c;

pub fn vt_size(t: u8) -> Option<usize> {
    match t {
        VT_I32 | VT_F32 => Some(4),
        VT_I64 | VT_F64 => Some(8),
        _ => None,
    }
}

pub fn vt_name(t: u8) -> &'static str {
    match t {
        VT_I32 => "i32",
        VT_I64 => "i64",
        VT_F32 => "f32",
        VT_F64 => "f64",
        _ => "?",
    }
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub params: Vec<u8>,
    pub results: Vec<u8>,
    pub flags: u8,
    pub arg_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct Region {
    pub name: String,
    pub getter_index: u16,
    pub has_arg: bool,
    pub arg: u32,
    pub elem: u8,
    pub byte_len: u32,
    pub flags: u8,
}

#[derive(Debug)]
pub struct Header {
    pub major: u16,
    pub minor: u16,
    pub header_len: u32,
    pub wasm_sha256: [u8; 32],
    pub mem_bytes: u32,
    pub flags: u32,
    pub meta: Json,
    pub funcs: Vec<Func>,
    pub regions: Vec<Region>,
    /// Region indices carrying a digest, in table order (the order of digests in a CALL record).
    pub digested: Vec<usize>,
}

impl Header {
    pub fn wasm_sha256_hex(&self) -> String {
        self.wasm_sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// One-paragraph description (the `--info` output).
    pub fn describe(&self) -> String {
        let s = |k: &str| {
            self.meta
                .get(k)
                .map(|v| format!("{v:?}"))
                .unwrap_or_else(|| "-".into())
        };
        format!(
            "CSTRACE v{}.{}, header {} B, wasm sha256 {}, wasm memory {} B, flags {} ({}{}), {} functions,              {} regions ({} digested), label {}, instance {}, digestMask {}",
            self.major,
            self.minor,
            self.header_len,
            self.wasm_sha256_hex(),
            self.mem_bytes,
            self.flags,
            if self.flags & 1 != 0 { "strict" } else { "default" },
            if self.flags & 2 != 0 { ", whole-memory digests" } else { "" },
            self.funcs.len(),
            self.regions.len(),
            self.digested.len(),
            s("label"),
            s("instance"),
            s("digestMask"),
        )
    }
}

/// One CALL record. Slices borrow the reader's buffer until the next record is read.
pub struct Call<'a> {
    pub seq: u32,
    pub func: u16,
    pub args: &'a [u8],
    /// 0 returned, 1 trapped / threw.
    pub status: u8,
    pub result: Option<&'a [u8]>,
    /// One little-endian `u32` XXH32 per digested region, in `Header::digested` order.
    pub digests: Option<&'a [u8]>,
    pub mem_digest: Option<u32>,
}

impl Call<'_> {
    pub fn digest(&self, i: usize) -> u32 {
        let d = self.digests.expect("call has digests");
        u32::from_le_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]])
    }
}

pub enum Record<'a> {
    Write {
        region: u16,
        offset: u32,
        data: &'a [u8],
    },
    Unmapped {
        addr: u32,
        data: &'a [u8],
    },
    Reset,
    Call(Call<'a>),
    End {
        reason: u8,
        calls: u32,
        writes: u32,
        unmapped: u32,
    },
    /// End of file. `clean`: an END record was seen and nothing follows it; `torn`: the last
    /// record was cut off (process killed mid-flush), so the prefix before it was replayed.
    Eof {
        clean: bool,
        torn: bool,
    },
}

pub const END_REASONS: [&str; 4] = ["exit", "instance-cap", "total-cap", "instance-collected"];

pub struct TraceReader {
    rd: BufReader<File>,
    size: u64,
    pos: u64,
    buf: Vec<u8>,
    ended: bool,
    /// Per function: (argument bytes, result bytes).
    func_sizes: Vec<(usize, usize)>,
    n_regions: usize,
    n_digested: usize,
}

/// Reading past the end of the file: a torn final record.
struct Truncated;

impl TraceReader {
    /// Opens a trace and parses its header; records are then read with `next_record`.
    pub fn open(path: &Path) -> Result<(TraceReader, Header), String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = f.metadata().map_err(|e| e.to_string())?.len();
        let mut r = TraceReader {
            rd: BufReader::with_capacity(8 << 20, f),
            size,
            pos: 0,
            buf: Vec::with_capacity(1 << 20),
            ended: false,
            func_sizes: Vec::new(),
            n_regions: 0,
            n_digested: 0,
        };
        let h = r
            .read_header()
            .map_err(|e| format!("{}: bad CSTRACE header: {e}", path.display()))?;
        r.func_sizes = h
            .funcs
            .iter()
            .map(|f| {
                (
                    f.arg_bytes,
                    f.results
                        .first()
                        .map(|&t| vt_size(t).unwrap_or(0))
                        .unwrap_or(0),
                )
            })
            .collect();
        r.n_regions = h.regions.len();
        r.n_digested = h.digested.len();
        Ok((r, h))
    }

    fn fill(&mut self, n: usize) -> io::Result<()> {
        self.buf.resize(n, 0);
        self.rd.read_exact(&mut self.buf[..n])?;
        self.pos += n as u64;
        Ok(())
    }
    fn u8(&mut self) -> io::Result<u8> {
        let mut b = [0u8; 1];
        self.rd.read_exact(&mut b)?;
        self.pos += 1;
        Ok(b[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        let mut b = [0u8; 2];
        self.rd.read_exact(&mut b)?;
        self.pos += 2;
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.rd.read_exact(&mut b)?;
        self.pos += 4;
        Ok(u32::from_le_bytes(b))
    }
    fn str16(&mut self) -> Result<String, String> {
        let n = self.u16().map_err(|e| e.to_string())? as usize;
        self.fill(n).map_err(|e| e.to_string())?;
        String::from_utf8(self.buf[..n].to_vec()).map_err(|e| e.to_string())
    }

    fn read_header(&mut self) -> Result<Header, String> {
        let e = |x: io::Error| x.to_string();
        self.fill(8).map_err(e)?;
        if &self.buf[..8] != MAGIC {
            return Err("not a CSTRACE file".into());
        }
        let major = self.u16().map_err(e)?;
        let minor = self.u16().map_err(e)?;
        if major != VERSION_MAJOR {
            return Err(format!("unsupported trace version {major}.{minor}"));
        }
        let header_len = self.u32().map_err(e)?;
        self.fill(32).map_err(e)?;
        let mut wasm_sha256 = [0u8; 32];
        wasm_sha256.copy_from_slice(&self.buf[..32]);
        let mem_bytes = self.u32().map_err(e)?;
        let flags = self.u32().map_err(e)?;
        let mlen = self.u32().map_err(e)? as usize;
        self.fill(mlen).map_err(e)?;
        let meta_text = std::str::from_utf8(&self.buf[..mlen])
            .map_err(|x| x.to_string())?
            .to_owned();
        let meta = json::parse(&meta_text)?;
        let nf = self.u16().map_err(e)? as usize;
        let mut funcs = Vec::with_capacity(nf);
        for _ in 0..nf {
            let name = self.str16()?;
            let np = self.u8().map_err(e)? as usize;
            self.fill(np).map_err(e)?;
            let params = self.buf[..np].to_vec();
            let nr = self.u8().map_err(e)? as usize;
            self.fill(nr).map_err(e)?;
            let results = self.buf[..nr].to_vec();
            let fl = self.u8().map_err(e)?;
            let mut arg_bytes = 0;
            for &t in &params {
                arg_bytes += vt_size(t)
                    .ok_or_else(|| format!("{name}: unsupported parameter type 0x{t:02x}"))?;
            }
            for &t in &results {
                vt_size(t).ok_or_else(|| format!("{name}: unsupported result type 0x{t:02x}"))?;
            }
            if results.len() > 1 {
                return Err(format!("{name}: multi-value results are not supported"));
            }
            funcs.push(Func {
                name,
                params,
                results,
                flags: fl,
                arg_bytes,
            });
        }
        let nr = self.u16().map_err(e)? as usize;
        let mut regions = Vec::with_capacity(nr);
        for _ in 0..nr {
            let name = self.str16()?;
            let getter_index = self.u16().map_err(e)?;
            let has_arg = self.u8().map_err(e)? == 1;
            let arg = self.u32().map_err(e)?;
            let elem = self.u8().map_err(e)?;
            let byte_len = self.u32().map_err(e)?;
            let rflags = self.u8().map_err(e)?;
            if getter_index as usize >= funcs.len() {
                return Err(format!(
                    "region {name}: getter index {getter_index} out of range"
                ));
            }
            regions.push(Region {
                name,
                getter_index,
                has_arg,
                arg,
                elem,
                byte_len,
                flags: rflags,
            });
        }
        if self.pos != header_len as u64 {
            return Err(format!(
                "header length mismatch {} != {header_len}",
                self.pos
            ));
        }
        let digested = (0..regions.len())
            .filter(|&i| regions[i].flags & RFLAG_DIGESTED != 0)
            .collect();
        Ok(Header {
            major,
            minor,
            header_len,
            wasm_sha256,
            mem_bytes,
            flags,
            meta,
            funcs,
            regions,
            digested,
        })
    }

    /// Reads the next record. After `Eof` it keeps returning `Eof`.
    pub fn next_record(&mut self) -> Result<Record<'_>, String> {
        if self.pos >= self.size {
            return Ok(Record::Eof {
                clean: self.ended,
                torn: false,
            });
        }
        let start = self.pos;
        match self.read_record() {
            Ok(Ok(r)) => {
                let r = r.resolve(&self.buf);
                if let Record::End { .. } = r {
                    self.ended = true;
                    if self.pos < self.size {
                        return Err(format!("data after END record at byte {}", self.pos));
                    }
                }
                Ok(r)
            }
            Ok(Err(Truncated)) => {
                // Stop at the last whole record.
                self.pos = self.size;
                Ok(Record::Eof {
                    clean: false,
                    torn: true,
                })
            }
            Err(e) => Err(format!("record at byte {start}: {e}")),
        }
    }

    /// Reads one record into `self.buf`, returning offsets into it.
    fn read_record(&mut self) -> Result<Result<RawRecord, Truncated>, String> {
        macro_rules! t {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(x) if x.kind() == io::ErrorKind::UnexpectedEof => return Ok(Err(Truncated)),
                    Err(x) => return Err(x.to_string()),
                }
            };
        }
        let tag = t!(self.u8());
        match tag {
            TAG_WRITE => {
                let region = t!(self.u16());
                let offset = t!(self.u32());
                let len = t!(self.u32()) as usize;
                if self.pos + len as u64 > self.size {
                    return Ok(Err(Truncated));
                }
                if region as usize >= self.n_regions {
                    return Err(format!("bad region index {region}"));
                }
                t!(self.fill(len));
                Ok(Ok(RawRecord::Write {
                    region,
                    offset,
                    len,
                }))
            }
            TAG_UNMAPPED => {
                let addr = t!(self.u32());
                let len = t!(self.u32()) as usize;
                if self.pos + len as u64 > self.size {
                    return Ok(Err(Truncated));
                }
                t!(self.fill(len));
                Ok(Ok(RawRecord::Unmapped { addr, len }))
            }
            TAG_RESET => Ok(Ok(RawRecord::Reset)),
            TAG_CALL => {
                let seq = t!(self.u32());
                let func = t!(self.u16());
                let &(arg_bytes, result_bytes) = self
                    .func_sizes
                    .get(func as usize)
                    .ok_or_else(|| format!("bad function index {func}"))?;
                let n_dig = self.n_digested;
                // Layout in self.buf: args | result | digests.
                self.buf.clear();
                let mut tmp = vec![0u8; arg_bytes];
                t!(self.rd.read_exact(&mut tmp));
                self.pos += arg_bytes as u64;
                self.buf.extend_from_slice(&tmp);
                let status = t!(self.u8());
                let mut has_result = false;
                if status == 0 && result_bytes > 0 {
                    tmp.resize(result_bytes, 0);
                    t!(self.rd.read_exact(&mut tmp));
                    self.pos += result_bytes as u64;
                    self.buf.extend_from_slice(&tmp);
                    has_result = true;
                }
                let dflags = t!(self.u8());
                let has_digests = dflags & 1 != 0;
                if has_digests {
                    tmp.resize(4 * n_dig, 0);
                    t!(self.rd.read_exact(&mut tmp));
                    self.pos += 4 * n_dig as u64;
                    self.buf.extend_from_slice(&tmp);
                }
                let mem_digest = if dflags & 2 != 0 {
                    Some(t!(self.u32()))
                } else {
                    None
                };
                Ok(Ok(RawRecord::Call {
                    seq,
                    func,
                    arg_bytes,
                    status,
                    result_bytes: if has_result { result_bytes } else { 0 },
                    has_result,
                    digest_bytes: if has_digests { 4 * n_dig } else { 0 },
                    has_digests,
                    mem_digest,
                }))
            }
            TAG_END => {
                let reason = t!(self.u8());
                let calls = t!(self.u32());
                let writes = t!(self.u32());
                let unmapped = t!(self.u32());
                Ok(Ok(RawRecord::End {
                    reason,
                    calls,
                    writes,
                    unmapped,
                }))
            }
            _ => Err(format!("bad record tag 0x{tag:02x}")),
        }
    }
}

enum RawRecord {
    Write {
        region: u16,
        offset: u32,
        len: usize,
    },
    Unmapped {
        addr: u32,
        len: usize,
    },
    Reset,
    Call {
        seq: u32,
        func: u16,
        arg_bytes: usize,
        status: u8,
        result_bytes: usize,
        has_result: bool,
        digest_bytes: usize,
        has_digests: bool,
        mem_digest: Option<u32>,
    },
    End {
        reason: u8,
        calls: u32,
        writes: u32,
        unmapped: u32,
    },
}

impl RawRecord {
    fn resolve(self, buf: &[u8]) -> Record<'_> {
        match self {
            RawRecord::Write {
                region,
                offset,
                len,
            } => Record::Write {
                region,
                offset,
                data: &buf[..len],
            },
            RawRecord::Unmapped { addr, len } => Record::Unmapped {
                addr,
                data: &buf[..len],
            },
            RawRecord::Reset => Record::Reset,
            RawRecord::Call {
                seq,
                func,
                arg_bytes,
                status,
                result_bytes,
                has_result,
                digest_bytes,
                has_digests,
                mem_digest,
            } => {
                let args = &buf[..arg_bytes];
                let result = if has_result {
                    Some(&buf[arg_bytes..arg_bytes + result_bytes])
                } else {
                    None
                };
                let d0 = arg_bytes + result_bytes;
                let digests = if has_digests {
                    Some(&buf[d0..d0 + digest_bytes])
                } else {
                    None
                };
                Record::Call(Call {
                    seq,
                    func,
                    args,
                    status,
                    result,
                    digests,
                    mem_digest,
                })
            }
            RawRecord::End {
                reason,
                calls,
                writes,
                unmapped,
            } => Record::End {
                reason,
                calls,
                writes,
                unmapped,
            },
        }
    }
}
