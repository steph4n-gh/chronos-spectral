//! `--bench`: times the native kernel on a recorded trace, measuring the same thing as
//! `node scripts/trace_replay.mjs --bench` measures on the wasm (docs/VERIFICATION_AND_BENCHMARKS.md,
//! "Native vs wasm kernel timing").
//!
//! Per trace:
//! 1. **Load** (untimed): the records are read once into memory (`Op`s plus one byte store, so a
//!    pass does no I/O and no allocation). Loading stops where a native replay must stop: at the
//!    first `U` (raw-address) record or the first recorded trap. The wasm side stops at the same
//!    record, so both time the identical call sequence (the comparison script checks the counts).
//! 2. **Passes**: each pass starts from the fresh-instance state (`reset_to_pristine`) and replays
//!    every op. Host writes and resets are applied untimed; each kernel call is timed alone with
//!    the monotonic clock (`Instant`: `mach_absolute_time` on macOS, `CLOCK_MONOTONIC` on Linux).
//!    Return values are still compared with the recorded ones (outside the timed region), so a
//!    timing pass that drifts from the recording is reported; region digests are not computed.
//! 3. `warmup` passes are discarded; `passes` passes are measured.
//! 4. **Reconciliation** (`--bench-reconcile=K`): K passes with no per-call timer (one wall-clock
//!    reading around the pass) and K passes that apply every host write but make no kernel call.
//!    Their difference is the kernel's share of the wall-clock, measured without per-call timers;
//!    it must agree with the sum of the per-call times.
//!
//! Timer overhead: the mean of an empty timed interval (`t1 - t0` with nothing between) is
//! measured at start-up and subtracted from every per-call sample (clamped at zero).

use crate::kernel::Ret;
use crate::replayer::{Options, Replayer, Stats, CUR_CALL};
use crate::trace::*;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub struct BenchOptions {
    pub warmup: usize,
    pub passes: usize,
    pub reconcile: usize,
    pub json: Option<std::path::PathBuf>,
    /// Print the per-export table for every trace (default: totals only).
    pub verbose: bool,
}

enum Op {
    Write {
        region: u16,
        offset: u32,
        start: usize,
        len: usize,
    },
    Reset,
    Call {
        seq: u32,
        func: u16,
        args: (usize, usize),
        result: Option<(usize, usize)>,
    },
}

struct Loaded {
    header: Header,
    ops: Vec<Op>,
    data: Vec<u8>,
    calls: usize,
    writes: usize,
    write_bytes: usize,
    /// Why loading stopped: "end", or where the native-portable prefix ends.
    stop: String,
}

fn load(path: &Path) -> Result<Loaded, String> {
    let (mut rd, header) = TraceReader::open(path)?;
    let mut ops = Vec::new();
    let mut data = Vec::new();
    let (mut calls, mut writes, mut write_bytes) = (0usize, 0usize, 0usize);
    let stop;
    loop {
        match rd.next_record()? {
            Record::Write {
                region,
                offset,
                data: d,
            } => {
                ops.push(Op::Write {
                    region,
                    offset,
                    start: data.len(),
                    len: d.len(),
                });
                data.extend_from_slice(d);
                writes += 1;
                write_bytes += d.len();
            }
            Record::Reset => ops.push(Op::Reset),
            Record::Unmapped { .. } => {
                stop = format!("prefix@{calls} (UNMAPPED write)");
                break;
            }
            Record::Call(c) => {
                if c.status != 0 {
                    stop = format!("prefix@{calls} (recorded trap)");
                    break;
                }
                let a = (data.len(), c.args.len());
                data.extend_from_slice(c.args);
                let result = c.result.map(|r| {
                    let s = (data.len(), r.len());
                    data.extend_from_slice(r);
                    s
                });
                ops.push(Op::Call {
                    seq: c.seq,
                    func: c.func,
                    args: a,
                    result,
                });
                calls += 1;
            }
            Record::End { .. } => {}
            Record::Eof { .. } => {
                stop = "end".to_owned();
                break;
            }
        }
    }
    Ok(Loaded {
        header,
        ops,
        data,
        calls,
        writes,
        write_bytes,
        stop,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Every call timed on its own.
    Timed,
    /// Writes and calls, one wall-clock reading around the pass.
    Untimed,
    /// Writes only, no kernel call.
    ApplyOnly,
}

struct PassOut {
    wall_ns: u64,
    /// Sum of the overhead-corrected per-call times.
    kernel_ns: u64,
    mismatches: u64,
}

/// Mean of an empty timed interval, in ns (the per-sample correction), and the smallest non-zero
/// interval seen (the clock's effective resolution).
pub fn calibrate_timer() -> (f64, f64) {
    const N: usize = 200_000;
    let mut sum = 0u128;
    let mut min_nz = u128::MAX;
    for _ in 0..N {
        let t0 = Instant::now();
        let t1 = Instant::now();
        let d = t1.duration_since(t0).as_nanos();
        sum += d;
        if d > 0 && d < min_nz {
            min_nz = d;
        }
    }
    (sum as f64 / N as f64, min_nz as f64)
}

fn run_pass(
    rp: &mut Replayer,
    ld: &Loaded,
    mode: Mode,
    overhead_ns: f64,
    samples: Option<&mut Vec<Vec<u64>>>,
    stats: &mut Stats,
) -> Result<PassOut, String> {
    let mut samples = samples;
    let mut out = PassOut {
        wall_ns: 0,
        kernel_ns: 0,
        mismatches: 0,
    };
    rp.reset();
    let pass_t0 = Instant::now();
    for op in &ld.ops {
        match *op {
            Op::Write {
                region,
                offset,
                start,
                len,
            } => rp.apply_write(region, offset, &ld.data[start..start + len])?,
            Op::Reset => rp.reset(),
            Op::Call {
                seq,
                func,
                args,
                result,
            } => {
                if mode == Mode::ApplyOnly {
                    continue;
                }
                let a = &ld.data[args.0..args.0 + args.1];
                CUR_CALL.store(seq, Ordering::Relaxed);
                let ret: Ret = if mode == Mode::Timed {
                    let t0 = Instant::now();
                    let r = rp.invoke(func, a);
                    let t1 = Instant::now();
                    let ns = t1.duration_since(t0).as_nanos() as f64 - overhead_ns;
                    let ns = if ns > 0.0 { ns.round() as u64 } else { 0 };
                    out.kernel_ns += ns;
                    if let Some(s) = samples.as_deref_mut() {
                        s[func as usize].push(ns);
                    }
                    r
                } else {
                    rp.invoke(func, a)
                };
                CUR_CALL.store(u32::MAX, Ordering::Relaxed);
                let call = Call {
                    seq,
                    func,
                    args: a,
                    status: 0,
                    result: result.map(|(s, n)| &ld.data[s..s + n]),
                    digests: None,
                    mem_digest: None,
                };
                if rp.check_result(&call, ret, stats).is_some() {
                    out.mismatches += 1;
                }
            }
        }
    }
    out.wall_ns = pass_t0.elapsed().as_nanos() as u64;
    Ok(out)
}

fn quantile(sorted: &[u64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    // Linear interpolation between closest ranks (same definition as the wasm side).
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] as f64 + (sorted[hi] as f64 - sorted[lo] as f64) * (pos - lo as f64)
}

fn median_f(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        0.0
    } else if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
    }
}

fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn jnums(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.1}")).collect();
    format!("[{}]", parts.join(","))
}

/// Runs the timing mode over `files`; returns the process exit status.
pub fn run(files: &[std::path::PathBuf], opts: &Options, b: &BenchOptions, build: &str) -> i32 {
    let (overhead_ns, resolution_ns) = calibrate_timer();
    println!(
        "chronos_replay --bench: {build}; warmup {} + {} measured pass(es) per trace; timer: Instant, \
         empty-interval mean {overhead_ns:.1} ns (subtracted per call), resolution {resolution_ns:.1} ns",
        b.warmup, b.passes
    );
    let mut json_traces = Vec::new();
    let mut failed = false;
    for f in files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Ok(mut t) = crate::replayer::CUR_TRACE.lock() {
            t.clone_from(&name);
        }
        let t_load = Instant::now();
        let ld = match load(f) {
            Ok(l) => l,
            Err(e) => {
                println!("[FAIL] {name}: {e}");
                failed = true;
                continue;
            }
        };
        let load_s = t_load.elapsed().as_secs_f64();
        let mut rp = match Replayer::new(&ld.header, opts) {
            Ok(r) => r,
            Err(e) => {
                println!("[FAIL] {name}: {e}");
                failed = true;
                continue;
            }
        };
        let nf = ld.header.funcs.len();
        let mut samples: Vec<Vec<u64>> = vec![Vec::new(); nf];
        let mut stats = Stats::default();
        let mut pass_kernel = Vec::new();
        let mut pass_wall = Vec::new();
        let mut warm_kernel = Vec::new();
        let mut mismatches = 0u64;
        let mut err = None;
        for p in 0..b.warmup + b.passes {
            let measured = p >= b.warmup;
            if p == b.warmup {
                // Marker for scripts/bench_profile.mjs: attach the sampler from here on.
                eprintln!("[bench] {name}: measuring {} pass(es)", b.passes);
            }
            match run_pass(
                &mut rp,
                &ld,
                Mode::Timed,
                overhead_ns,
                if measured { Some(&mut samples) } else { None },
                &mut stats,
            ) {
                Ok(o) => {
                    mismatches += o.mismatches;
                    if measured {
                        pass_kernel.push(o.kernel_ns as f64);
                        pass_wall.push(o.wall_ns as f64);
                    } else {
                        warm_kernel.push(o.kernel_ns as f64);
                    }
                }
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        let mut rec_untimed = Vec::new();
        let mut rec_apply = Vec::new();
        if err.is_none() {
            for _ in 0..b.reconcile {
                for (mode, v) in [
                    (Mode::Untimed, &mut rec_untimed),
                    (Mode::ApplyOnly, &mut rec_apply),
                ] {
                    match run_pass(&mut rp, &ld, mode, overhead_ns, None, &mut stats) {
                        Ok(o) => {
                            if mode == Mode::Untimed {
                                mismatches += o.mismatches;
                            }
                            v.push(o.wall_ns as f64)
                        }
                        Err(e) => err = Some(e),
                    }
                }
            }
        }
        if let Some(e) = err {
            println!("[FAIL] {name}: {e}");
            failed = true;
            continue;
        }
        if mismatches > 0 {
            failed = true;
        }
        let km = median_f(&pass_kernel);
        let wm = median_f(&pass_wall);
        println!(
            "[{}] {name}: {} calls, {} host writes ({:.1} MB), stop={}, loaded in {load_s:.1} s; \
             kernel {:.3} ms/pass (median of {}; min {:.3}, max {:.3}), pass wall {:.3} ms, \
             return mismatches {mismatches}",
            if mismatches == 0 { "BENCH" } else { "FAIL " },
            ld.calls,
            ld.writes,
            ld.write_bytes as f64 / 1e6,
            ld.stop,
            km / 1e6,
            pass_kernel.len(),
            pass_kernel.iter().cloned().fold(f64::INFINITY, f64::min) / 1e6,
            pass_kernel.iter().cloned().fold(0.0, f64::max) / 1e6,
            wm / 1e6,
        );
        if b.reconcile > 0 {
            let u = median_f(&rec_untimed);
            let a = median_f(&rec_apply);
            println!(
                "    reconcile: untimed pass {:.3} ms - apply-only pass {:.3} ms = {:.3} ms kernel by wall-clock \
                 vs {:.3} ms summed per call ({:+.1}%); timed pass wall {:.3} ms",
                u / 1e6,
                a / 1e6,
                (u - a) / 1e6,
                km / 1e6,
                100.0 * (km - (u - a)) / (u - a).max(1.0),
                wm / 1e6
            );
        }
        let mut rows = Vec::new();
        let mut jexp = String::new();
        for (fi, s) in samples.iter_mut().enumerate() {
            if s.is_empty() {
                continue;
            }
            s.sort_unstable();
            let n = s.len();
            let sum: u128 = s.iter().map(|&x| x as u128).sum();
            let mean = sum as f64 / n as f64;
            let med = quantile(s, 0.5);
            let p95 = quantile(s, 0.95);
            let per_pass = n / b.passes.max(1);
            let fname = &ld.header.funcs[fi].name;
            rows.push((sum, fname.clone(), per_pass, med, p95, mean));
            let _ = write!(
                jexp,
                "{{\"name\":{},\"count\":{per_pass},\"median_ns\":{med:.1},\"p95_ns\":{p95:.1},\"mean_ns\":{mean:.1},\
                 \"total_ns_per_pass\":{:.1},\"min_ns\":{},\"max_ns\":{},\"samples_ns\":[",
                jstr(fname),
                sum as f64 / b.passes.max(1) as f64,
                s[0],
                s[n - 1]
            );
            // Every measured sample (sorted), so per-export statistics can be pooled exactly
            // across traces and runs.
            for (k, x) in s.iter().enumerate() {
                let _ = write!(jexp, "{}{x}", if k > 0 { "," } else { "" });
            }
            jexp.push_str("]}");
            jexp.push(',');
        }
        jexp.pop();
        rows.sort_by(|a, b| b.0.cmp(&a.0));
        if b.verbose {
            println!(
                "    {:<36} {:>8} {:>12} {:>12} {:>12} {:>9}",
                "export", "calls", "median us", "p95 us", "mean us", "share"
            );
            for (sum, n, c, med, p95, mean) in &rows {
                println!(
                    "    {:<36} {:>8} {:>12.3} {:>12.3} {:>12.3} {:>8.1}%",
                    n,
                    c,
                    med / 1e3,
                    p95 / 1e3,
                    mean / 1e3,
                    100.0 * (*sum as f64 / b.passes.max(1) as f64) / km.max(1.0)
                );
            }
        }
        json_traces.push(format!(
            "{{\"name\":{},\"calls\":{},\"writes\":{},\"write_bytes\":{},\"stop\":{},\"mismatches\":{mismatches},\
             \"warmup_kernel_ns\":{},\"pass_kernel_ns\":{},\"pass_wall_ns\":{},\"reconcile_untimed_ns\":{},\
             \"reconcile_apply_ns\":{},\"exports\":[{}]}}",
            jstr(&name),
            ld.calls,
            ld.writes,
            ld.write_bytes,
            jstr(&ld.stop),
            jnums(&warm_kernel),
            jnums(&pass_kernel),
            jnums(&pass_wall),
            jnums(&rec_untimed),
            jnums(&rec_apply),
            jexp
        ));
    }
    if let Some(p) = &b.json {
        let doc = format!(
            "{{\"side\":\"native\",\"backend\":{},\"arch\":{},\"build\":{},\"timer\":{{\"clock\":\"Instant\",\
             \"overhead_ns\":{overhead_ns:.2},\"resolution_ns\":{resolution_ns:.2}}},\"warmup\":{},\"passes\":{},\
             \"traces\":[{}]}}\n",
            jstr(chronos_spectral::simd::BACKEND_NAME),
            jstr(std::env::consts::ARCH),
            jstr(build),
            b.warmup,
            b.passes,
            json_traces.join(",")
        );
        if let Err(e) = std::fs::write(p, doc) {
            eprintln!("chronos_replay: cannot write {}: {e}", p.display());
            return 1;
        }
    }
    i32::from(failed)
}