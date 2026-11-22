//! Replays one CSTRACE file against the native kernel.
//!
//! The three phases are separate methods so a timing mode can measure only `invoke`:
//! * apply: [`Replayer::apply_write`] / [`Replayer::reset`] (host writes, `RESET`);
//! * call: [`Replayer::invoke`] (one kernel export with the recorded argument bits);
//! * verify: [`Replayer::check_result`] and [`Replayer::check_digests`].

use crate::allowlist;
use crate::json::Json;
use crate::kernel::{Export, Ret};
use crate::trace::*;
use chronos_spectral::native::{self, NativeRegion, RegionKind};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Native exports, built once (the panic hook reads names from it).
pub fn exports() -> &'static [Export] {
    static E: OnceLock<Vec<Export>> = OnceLock::new();
    E.get_or_init(crate::kernel::exports)
}

/// What the kernel is doing right now, for the panic hook (a kernel panic aborts the process:
/// the exports are `extern "C"` and cannot unwind).
pub static CUR_CALL: AtomicU32 = AtomicU32::new(u32::MAX);
pub static CUR_EXPORT: AtomicUsize = AtomicUsize::new(usize::MAX);
pub static CUR_TRACE: Mutex<String> = Mutex::new(String::new());

pub struct Options {
    /// Divergent calls reported per trace before it stops (default 1: the first divergence).
    pub max_errors: usize,
    /// Directory for native region dumps.
    pub dump_dir: PathBuf,
    /// Extra dumps: (call index, region names) dumped after that call even when it matches.
    pub dump_at: Vec<(u32, Vec<String>)>,
    /// Hash regions unmasked and skip the mask check (traces recorded before the mask existed).
    pub no_mask: bool,
    /// Print each trace's header before replaying it.
    pub info: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Every record replayed, every check matched.
    Pass,
    /// Stopped at a raw-address (`UNMAPPED`) write: everything before call `at_call` matched.
    PortablePrefix { at_call: u32, addr: u32, len: usize },
    /// A recorded trap: the native kernel cannot replay it in-process (it would abort).
    RecordedTrap { at_call: u32 },
    /// At least one divergence (or a structural error).
    Fail,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub calls: u64,
    pub digest_calls: u64,
    pub region_digests: u64,
    pub writes: u64,
    pub write_bytes: u64,
    pub resets: u64,
    pub mismatches: u64,
    pub skipped_pointer_results: u64,
    pub nan_results: u64,
    /// Whole-memory digests (validation traces) are meaningful only against the same wasm.
    pub skipped_mem_digests: u64,
}

pub struct Report {
    pub name: String,
    pub outcome: Outcome,
    pub stats: Stats,
    pub end: String,
    pub secs: f64,
    pub file_bytes: u64,
    pub messages: Vec<String>,
}

struct RegionBind {
    name: String,
    getter: unsafe fn(&[u8]) -> Ret,
    arg: [u8; 4],
    byte_len: usize,
    native: NativeRegion,
}

struct DigestSlot {
    region: usize,
    mask: Vec<usize>,
    prev: Vec<u8>,
    prev_hash: u32,
    valid: bool,
}

pub struct Replayer<'h> {
    header: &'h Header,
    /// Trace function index -> native export index.
    func_map: Vec<usize>,
    regions: Vec<RegionBind>,
    digests: Vec<DigestSlot>,
    ring_slots: Vec<usize>,
    scratch: Vec<u8>,
}

impl<'h> Replayer<'h> {
    /// Binds the trace's function and region tables to this native build, checking every
    /// signature, region size, element type and flag before any memory is touched.
    pub fn new(header: &'h Header, opts: &Options) -> Result<Replayer<'h>, String> {
        let ex = exports();
        let by_name: BTreeMap<&str, usize> =
            ex.iter().enumerate().map(|(i, e)| (e.name, i)).collect();
        let mut errs = Vec::new();
        let mut func_map = Vec::with_capacity(header.funcs.len());
        for f in &header.funcs {
            match by_name.get(f.name.as_str()) {
                None => {
                    errs.push(format!("export {} is not in the native kernel", f.name));
                    func_map.push(usize::MAX);
                }
                Some(&i) => {
                    let e = &ex[i];
                    if e.params != f.params
                        || e.results != f.results.as_slice()
                        || e.arg_bytes != f.arg_bytes
                    {
                        errs.push(format!(
                            "export {}: trace signature ({}) -> ({}) != native ({}) -> ({})",
                            f.name,
                            types(&f.params),
                            types(&f.results),
                            types(&e.params),
                            types(e.results)
                        ));
                    }
                    if e.pointer != (f.flags & FFLAG_POINTER_RESULT != 0) {
                        errs.push(format!(
                            "export {}: pointer-result flag disagrees with the native return type",
                            f.name
                        ));
                    }
                    func_map.push(i);
                }
            }
        }
        let traced: BTreeSet<&str> = header.funcs.iter().map(|f| f.name.as_str()).collect();
        for e in ex {
            if !traced.contains(e.name) {
                errs.push(format!(
                    "native export {} is missing from the trace's function table",
                    e.name
                ));
            }
        }

        let natives = native::regions();
        let (arena_lo, arena_hi) = native::arena_range();
        let ring_slots: Vec<usize> = natives
            .iter()
            .filter_map(|r| matches!(r.kind, RegionKind::RingSlot(_)).then_some(r.field_addr))
            .collect();
        let mut regions = Vec::with_capacity(header.regions.len());
        for r in &header.regions {
            let Some(n) = natives.iter().find(|n| n.name == r.name) else {
                errs.push(format!(
                    "region {} is not known to the native build",
                    r.name
                ));
                continue;
            };
            let getter_name = &header.funcs[r.getter_index as usize].name;
            let mut bad = Vec::new();
            if getter_name != n.getter {
                bad.push(format!("getter {getter_name} != native {}", n.getter));
            }
            let want_arg = match n.kind {
                RegionKind::RingSlot(k) | RegionKind::Indexed(k) => Some(k),
                _ => None,
            };
            if (r.has_arg, r.arg) != (want_arg.is_some(), want_arg.unwrap_or(r.arg))
                || (!r.has_arg && r.arg != 0)
            {
                bad.push(format!(
                    "getter argument {:?} != native {:?}",
                    r.has_arg.then_some(r.arg),
                    want_arg
                ));
            }
            if r.elem != n.elem {
                bad.push(format!("element type {} != native {}", r.elem, n.elem));
            }
            if r.byte_len as usize != n.byte_len {
                bad.push(format!(
                    "byte length {} != native {} (layout differs)",
                    r.byte_len, n.byte_len
                ));
            }
            let alias = matches!(n.kind, RegionKind::RingAlias | RegionKind::FixedAlias);
            if (r.flags & RFLAG_ALIAS != 0) != alias || (r.flags & RFLAG_DIGESTED != 0) == alias {
                bad.push(format!(
                    "alias/digested flags 0x{:x} disagree with the native region kind",
                    r.flags
                ));
            }
            if (r.flags & RFLAG_HOST_WRITABLE != 0) != n.host_writable {
                bad.push(format!(
                    "host-writable flag disagrees with the native getter (*mut = {})",
                    n.host_writable
                ));
            }
            if n.field_addr < arena_lo || n.field_addr + n.byte_len > arena_hi {
                bad.push("native field lies outside the arena".into());
            }
            if !bad.is_empty() {
                errs.push(format!("region {}: {}", r.name, bad.join("; ")));
                continue;
            }
            let gi = func_map[r.getter_index as usize];
            if gi == usize::MAX {
                continue;
            }
            regions.push(RegionBind {
                name: r.name.clone(),
                getter: ex[gi].call,
                arg: r.arg.to_le_bytes(),
                byte_len: n.byte_len,
                native: *n,
            });
        }
        for n in &natives {
            if !header.regions.iter().any(|r| r.name == n.name) {
                errs.push(format!(
                    "native region {} is missing from the trace's region table",
                    n.name
                ));
            }
        }

        // Digest mask: must be exactly the allowlist (or absent with --no-mask).
        let mut digests = Vec::with_capacity(header.digested.len());
        let mask = header.meta.get("digestMask");
        if !opts.no_mask {
            match mask {
                None => errs.push(
                    "trace has no meta.digestMask (recorded before the platform-word mask); re-record it, or pass \
                     --no-mask to hash raw regions"
                        .into(),
                ),
                Some(m) => {
                    if let Err(e) = check_mask(m, header) {
                        errs.push(e);
                    }
                }
            }
        }
        for &ri in &header.digested {
            let r = &header.regions[ri];
            digests.push(DigestSlot {
                region: ri,
                mask: if opts.no_mask {
                    Vec::new()
                } else {
                    allowlist::words_for(&r.name)
                },
                prev: Vec::new(),
                prev_hash: 0,
                valid: false,
            });
        }
        // `regions` is indexed by trace region index: every region must have bound.
        if errs.is_empty() && regions.len() != header.regions.len() {
            errs.push(format!(
                "bound {} of {} regions",
                regions.len(),
                header.regions.len()
            ));
        }
        if !errs.is_empty() {
            return Err(format!(
                "trace does not match this native build:\n    {}",
                errs.join("\n    ")
            ));
        }
        Ok(Replayer {
            header,
            func_map,
            regions,
            digests,
            ring_slots,
            scratch: Vec::new(),
        })
    }

    /// Resolves region `ri` through its getter now, and checks the address against the native
    /// layout (a fixed region must be its field; an alias must be a ring slot).
    pub fn resolve(&self, ri: usize) -> Result<*mut u8, String> {
        let b = &self.regions[ri];
        // SAFETY: getters only compute an address.
        let p = match unsafe { (b.getter)(&b.arg) } {
            Ret::Ptr(p) => p,
            other => return Err(format!("getter for {} returned {other:?}", b.name)),
        };
        let addr = p.addr();
        let ok = match b.native.kind {
            RegionKind::Fixed
            | RegionKind::FixedAlias
            | RegionKind::Indexed(_)
            | RegionKind::RingSlot(_) => addr == b.native.field_addr,
            RegionKind::RingAlias => self.ring_slots.contains(&addr),
        };
        if !ok {
            return Err(format!(
                "getter for {} returned 0x{addr:x}, outside its native field",
                b.name
            ));
        }
        Ok(p)
    }

    /// `RESET`: the fresh-instance state.
    pub fn reset(&mut self) {
        // SAFETY: no kernel call is in progress and no reference into the arena is live.
        unsafe { native::reset_to_pristine() };
        for d in &mut self.digests {
            d.valid = false;
        }
    }

    /// `W`: copies `data` to region + offset, the region resolved through its getter now.
    pub fn apply_write(&mut self, region: u16, offset: u32, data: &[u8]) -> Result<(), String> {
        let ri = region as usize;
        let b = &self.regions[ri];
        let end = offset as u64 + data.len() as u64;
        if end > b.byte_len as u64 {
            return Err(format!(
                "host write [{offset}, {end}) overruns region {} ({} B)",
                b.name, b.byte_len
            ));
        }
        let p = self.resolve(ri)?;
        // SAFETY: [p + offset, p + end) lies inside the region's native field (checked above and
        // in `resolve`), and no kernel call is in progress.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                p.add(offset as usize),
                data.len(),
            )
        };
        Ok(())
    }

    /// Calls trace function `func` with the recorded argument bits.
    #[inline]
    pub fn invoke(&mut self, func: u16, args: &[u8]) -> Ret {
        let i = self.func_map[func as usize];
        CUR_EXPORT.store(i, Ordering::Relaxed);
        // SAFETY: the argument length was checked against the signature when the trace was read;
        // the kernel is driven from this thread only.
        unsafe { (exports()[i].call)(args) }
    }

    /// Compares the native return value with the recorded one. NaN matches any NaN (the JS API
    /// that recorded the value cannot observe NaN payloads); pointer results are addresses and
    /// are not compared across builds. Returns a description of the difference, if any.
    pub fn check_result(&self, call: &Call, ret: Ret, stats: &mut Stats) -> Option<String> {
        let f = &self.header.funcs[call.func as usize];
        let rec = call.result?;
        if f.flags & FFLAG_POINTER_RESULT != 0 {
            stats.skipped_pointer_results += 1;
            return None;
        }
        let rec32 = u32::from_le_bytes(rec[..4].try_into().unwrap());
        match (f.results[0], ret) {
            (VT_I32, Ret::I32(v)) => (v != rec32)
                .then(|| format!("return {v} (0x{v:08x}) != recorded {rec32} (0x{rec32:08x})")),
            (VT_F32, Ret::F32(v)) => {
                let (a, b) = (f32::from_bits(v), f32::from_bits(rec32));
                if a.is_nan() && b.is_nan() {
                    stats.nan_results += 1;
                    None
                } else {
                    (v != rec32).then(|| {
                        format!("return {a:e} (0x{v:08x}) != recorded {b:e} (0x{rec32:08x})")
                    })
                }
            }
            (t, r) => Some(format!(
                "return {r:?} cannot be compared with a recorded {}",
                vt_name(t)
            )),
        }
    }

    /// Hashes every digested region (masked words zeroed) and compares with the recorded digests.
    pub fn check_digests(
        &mut self,
        call: &Call,
        stats: &mut Stats,
    ) -> Result<Vec<(usize, u32, u32)>, String> {
        let mut out = Vec::new();
        stats.digest_calls += 1;
        for i in 0..self.digests.len() {
            let ri = self.digests[i].region;
            let len = self.regions[ri].byte_len;
            let p = self.resolve(ri)?;
            // SAFETY: [p, p + len) is the region's native field; no kernel call is in progress.
            let cur = unsafe { std::slice::from_raw_parts(p.cast_const(), len) };
            let d = &mut self.digests[i];
            let hash = if d.valid && d.prev.as_slice() == cur {
                d.prev_hash
            } else {
                let h = if d.mask.is_empty() {
                    crate::xxh32::xxh32(cur, 0)
                } else {
                    self.scratch.clear();
                    self.scratch.extend_from_slice(cur);
                    for &w in &d.mask {
                        self.scratch[4 * w..4 * w + 4].fill(0);
                    }
                    crate::xxh32::xxh32(&self.scratch, 0)
                };
                d.prev.clear();
                d.prev.extend_from_slice(cur);
                d.prev_hash = h;
                d.valid = true;
                h
            };
            stats.region_digests += 1;
            let want = call.digest(i);
            if hash != want {
                out.push((ri, hash, want));
            }
        }
        Ok(out)
    }

    /// Writes region `ri`'s current native bytes (unmasked) to `dir`.
    pub fn dump_region(
        &self,
        dir: &Path,
        stem: &str,
        call: u32,
        ri: usize,
    ) -> Result<PathBuf, String> {
        let b = &self.regions[ri];
        let p = self.resolve(ri)?;
        // SAFETY: as in `check_digests`.
        let cur = unsafe { std::slice::from_raw_parts(p.cast_const(), b.byte_len) };
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let file = dir.join(format!("{stem}.c{call}.{}.native.bin", file_safe(&b.name)));
        std::fs::write(&file, cur).map_err(|e| format!("{}: {e}", file.display()))?;
        Ok(file)
    }

    pub fn region_index(&self, name: &str) -> Option<usize> {
        self.regions.iter().position(|r| r.name == name)
    }

    pub fn region_name(&self, ri: usize) -> &str {
        &self.regions[ri].name
    }
}

fn types(t: &[u8]) -> String {
    t.iter().map(|&x| vt_name(x)).collect::<Vec<_>>().join(", ")
}

pub fn file_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Checks that `meta.digestMask` is exactly the allowlist.
fn check_mask(m: &Json, header: &Header) -> Result<(), String> {
    let Json::Obj(obj) = m else {
        return Err("meta.digestMask is not an object".into());
    };
    let mut got: BTreeSet<(String, usize)> = BTreeSet::new();
    for (region, words) in obj {
        let Json::Arr(a) = words else {
            return Err(format!("meta.digestMask.{region} is not an array"));
        };
        let r = header
            .regions
            .iter()
            .find(|r| &r.name == region)
            .ok_or_else(|| format!("meta.digestMask names unknown region {region}"))?;
        for w in a {
            let w = w
                .as_f64()
                .filter(|x| x.fract() == 0.0 && *x >= 0.0)
                .ok_or("meta.digestMask word is not an index")? as usize;
            if r.elem == 1 || 4 * w + 4 > r.byte_len as usize {
                return Err(format!(
                    "meta.digestMask word {region}[{w}] is not a 32-bit word of the region"
                ));
            }
            got.insert((region.clone(), w));
        }
    }
    let want: BTreeSet<(String, usize)> = allowlist::PLATFORM_WORDS
        .iter()
        .map(|w| (w.region.to_owned(), w.word))
        .collect();
    if got != want {
        return Err(format!(
            "meta.digestMask {:?} is not this driver's platform-word allowlist {:?} (native/replay/allowlist.rs)",
            got, want
        ));
    }
    Ok(())
}

fn fmt_args(f: &Func, args: &[u8]) -> String {
    let mut s = String::new();
    let mut o = 0;
    for (k, &t) in f.params.iter().enumerate() {
        if k > 0 {
            s.push_str(", ");
        }
        match t {
            VT_I32 => {
                let v = u32::from_le_bytes(args[o..o + 4].try_into().unwrap());
                let _ = write!(s, "{}", v as i32);
            }
            VT_F32 => {
                let v = f32::from_bits(u32::from_le_bytes(args[o..o + 4].try_into().unwrap()));
                let _ = write!(s, "{v:?}");
            }
            VT_F64 => {
                let v = f64::from_bits(u64::from_le_bytes(args[o..o + 8].try_into().unwrap()));
                let _ = write!(s, "{v:?}");
            }
            _ => s.push('?'),
        }
        o += vt_size(t).unwrap_or(4);
    }
    s
}

/// Replays one trace file.
pub fn replay_file(path: &Path, opts: &Options) -> Report {
    let t0 = Instant::now();
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = name.trim_end_matches(".cstrace").to_owned();
    let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut report = Report {
        name: name.clone(),
        outcome: Outcome::Fail,
        stats: Stats::default(),
        end: "not replayed".to_owned(),
        secs: 0.0,
        file_bytes,
        messages: Vec::new(),
    };
    if let Ok(mut t) = CUR_TRACE.lock() {
        t.clone_from(&name);
    }
    let (mut rd, header) = match TraceReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            report.messages.push(e);
            return report;
        }
    };
    if opts.info {
        println!("{name}: {}", header.describe());
    }
    let mut rp = match Replayer::new(&header, opts) {
        Ok(r) => r,
        Err(e) => {
            report.messages.push(e);
            return report;
        }
    };

    // A trace starts from a fresh instance.
    rp.reset();
    report.end.clear();
    let st = &mut report.stats;
    let mut errors = 0usize;
    let mut end_rec: Option<(u8, u32, u32, u32)> = None;
    let mut stop: Option<Outcome> = None;
    loop {
        let rec = match rd.next_record() {
            Ok(r) => r,
            Err(e) => {
                report.messages.push(e);
                stop = Some(Outcome::Fail);
                break;
            }
        };
        match rec {
            Record::Write {
                region,
                offset,
                data,
            } => {
                if let Err(e) = rp.apply_write(region, offset, data) {
                    report
                        .messages
                        .push(format!("before call #{}: {e}", st.calls));
                    stop = Some(Outcome::Fail);
                    break;
                }
                st.writes += 1;
                st.write_bytes += data.len() as u64;
            }
            Record::Reset => {
                rp.reset();
                st.resets += 1;
            }
            Record::Unmapped { addr, data } => {
                stop = Some(Outcome::PortablePrefix {
                    at_call: st.calls as u32,
                    addr,
                    len: data.len(),
                });
                break;
            }
            Record::Call(call) => {
                if call.seq as u64 != st.calls {
                    report.messages.push(format!(
                        "call sequence {} != expected {}",
                        call.seq, st.calls
                    ));
                    stop = Some(Outcome::Fail);
                    break;
                }
                let f = &header.funcs[call.func as usize];
                if call.status != 0 {
                    stop = Some(Outcome::RecordedTrap { at_call: call.seq });
                    break;
                }
                if call.mem_digest.is_some() {
                    st.skipped_mem_digests += 1;
                }
                CUR_CALL.store(call.seq, Ordering::Relaxed);
                let ret = rp.invoke(call.func, call.args);
                CUR_CALL.store(u32::MAX, Ordering::Relaxed);
                st.calls += 1;
                let where_ =
                    || format!("call #{} {}({})", call.seq, f.name, fmt_args(f, call.args));
                let mut diverged = false;
                if let Some(d) = rp.check_result(&call, ret, st) {
                    report.messages.push(format!("{}: {d}", where_()));
                    diverged = true;
                }
                if call.digests.is_some() {
                    match rp.check_digests(&call, st) {
                        Err(e) => {
                            report.messages.push(format!("{}: {e}", where_()));
                            stop = Some(Outcome::Fail);
                            break;
                        }
                        Ok(bad) if !bad.is_empty() => {
                            diverged = true;
                            let mut msg = format!(
                                "{}: {} of {} region digests differ:",
                                where_(),
                                bad.len(),
                                header.digested.len()
                            );
                            for (ri, got, want) in bad {
                                let dump = rp.dump_region(&opts.dump_dir, &stem, call.seq, ri);
                                let _ = write!(
                                    msg,
                                    "\n      region {}: native {got:08x} != recorded {want:08x}",
                                    rp.region_name(ri)
                                );
                                match dump {
                                    Ok(p) => {
                                        let _ = write!(
                                            msg,
                                            "\n        native dump: {}\n        diff vs wasm: node scripts/trace_region_diff.mjs {} --call={} --region={} --native={}",
                                            p.display(),
                                            path.display(),
                                            call.seq,
                                            rp.region_name(ri),
                                            p.display()
                                        );
                                    }
                                    Err(e) => {
                                        let _ = write!(msg, " (dump failed: {e})");
                                    }
                                }
                            }
                            report.messages.push(msg);
                        }
                        Ok(_) => {}
                    }
                }
                for (c, names) in &opts.dump_at {
                    if *c == call.seq {
                        for n in names {
                            match rp.region_index(n) {
                                Some(ri) => {
                                    match rp.dump_region(&opts.dump_dir, &stem, call.seq, ri) {
                                        Ok(p) => report.messages.push(format!(
                                            "dumped {n} after call #{}: {}",
                                            call.seq,
                                            p.display()
                                        )),
                                        Err(e) => {
                                            report.messages.push(format!("dump of {n} failed: {e}"))
                                        }
                                    }
                                }
                                None => report
                                    .messages
                                    .push(format!("--dump-at: unknown region {n}")),
                            }
                        }
                    }
                }
                if diverged {
                    st.mismatches += 1;
                    errors += 1;
                    if errors >= opts.max_errors {
                        stop = Some(Outcome::Fail);
                        break;
                    }
                }
            }
            Record::End {
                reason,
                calls,
                writes,
                unmapped,
            } => {
                end_rec = Some((reason, calls, writes, unmapped));
            }
            Record::Eof { clean, torn } => {
                report.end = match end_rec {
                    Some((r, ..)) if clean => END_REASONS
                        .get(r as usize)
                        .copied()
                        .unwrap_or("?")
                        .to_owned(),
                    _ if torn => "torn tail (prefix replayed)".to_owned(),
                    _ => "no END record (prefix replayed)".to_owned(),
                };
                break;
            }
        }
    }
    if report.end.is_empty() {
        report.end = "stopped".to_owned();
    }
    report.outcome = match stop {
        Some(o) => o,
        None if st.mismatches > 0 => Outcome::Fail,
        None => {
            if let Some((_, calls, writes, unmapped)) = end_rec {
                if (calls as u64, writes as u64, unmapped) != (st.calls, st.writes, 0) {
                    report.messages.push(format!(
                        "END record counts (calls={calls}, writes={writes}, unmapped={unmapped}) disagree with the replay ({}, {}, 0)",
                        st.calls, st.writes
                    ));
                    Outcome::Fail
                } else {
                    Outcome::Pass
                }
            } else {
                Outcome::Pass
            }
        }
    };
    report.secs = t0.elapsed().as_secs_f64();
    report
}
