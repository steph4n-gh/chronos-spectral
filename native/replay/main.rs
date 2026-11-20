//! `chronos_replay`: replays CSTRACE v1 kernel traces (recorded from the wasm kernel by
//! scripts/trace_hook.mjs) against the native build of the same kernel source, and checks every
//! recorded return value and per-region XXH32 digest bit for bit.
//!
//! ```text
//! cargo run --release --manifest-path native/Cargo.toml --target host-tuple --bin chronos_replay -- \
//!     [options] <trace.cstrace | dir>...
//! ```
//!
//! Options:
//!   --max-errors=N            divergent calls reported per trace before it stops (default 1)
//!   --dump-dir=DIR            native region dumps on a digest mismatch (default target/replay-dumps)
//!   --dump-at=CALL:R1[,R2]    also dump regions R1, R2 after call CALL (repeatable)
//!   --require-full            a trace that stops at an UNMAPPED write (portable prefix) fails
//!   --wasm-sha256=HEX         every trace must have been recorded against this wasm binary
//!   --no-mask                 hash regions raw (traces recorded before the platform-word mask)
//!   --info                    print each trace's header (version, wasm sha256, metadata) first
//!   --in-process              replay every trace in this process (default: one child per trace)
//!   --list-allowlist          print the platform-word allowlist and exit
//!
//! Result per trace: PASS (every record replayed, 0 mismatches), PREFIX (the trace reached a
//! raw-address write a native build cannot follow; everything before it matched), TRAP (a
//! recorded trap; native kernel functions abort on panic, so it cannot be replayed in-process),
//! FAIL (a divergence, a structural error, or a native panic). Exit status: 0 when no trace
//! failed (and, with --require-full, none stopped early), 1 otherwise, 2 on usage errors.

mod allowlist;
mod json;
mod kernel;
mod replayer;
mod trace;
mod xxh32;

use replayer::{Options, Outcome, Report};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::Ordering;

/// Child exit codes (one trace per child process).
const CHILD_PASS: i32 = 0;
const CHILD_FAIL: i32 = 1;
const CHILD_PREFIX: i32 = 10;
const CHILD_TRAP: i32 = 11;

struct Cli {
    opts: Options,
    inputs: Vec<PathBuf>,
    require_full: bool,
    wasm_sha256: Option<String>,
    in_process: bool,
    child: bool,
    passthrough: Vec<String>,
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: chronos_replay [--max-errors=N] [--dump-dir=DIR] [--dump-at=CALL:REGION[,REGION]] [--require-full]\n\
         \x20                     [--wasm-sha256=HEX] [--no-mask] [--info] [--in-process] [--list-allowlist] <trace|dir>..."
    );
    ExitCode::from(2)
}

fn parse_cli() -> Result<Cli, String> {
    let mut cli = Cli {
        opts: Options {
            max_errors: 1,
            dump_dir: PathBuf::from("target/replay-dumps"),
            dump_at: Vec::new(),
            no_mask: false,
            info: false,
        },
        inputs: Vec::new(),
        require_full: false,
        wasm_sha256: None,
        in_process: false,
        child: false,
        passthrough: Vec::new(),
    };
    for a in std::env::args().skip(1) {
        let (k, v) = match a.split_once('=') {
            Some((k, v)) => (k.to_owned(), Some(v.to_owned())),
            None => (a.clone(), None),
        };
        let pass = !matches!(k.as_str(), "--in-process" | "--child") && k.starts_with("--");
        match (k.as_str(), v) {
            ("--max-errors", Some(v)) => {
                cli.opts.max_errors = v.parse().map_err(|_| format!("bad {a}"))?
            }
            ("--dump-dir", Some(v)) => cli.opts.dump_dir = PathBuf::from(v),
            ("--dump-at", Some(v)) => {
                let (c, rs) = v.split_once(':').ok_or_else(|| format!("bad {a}"))?;
                let c: u32 = c.parse().map_err(|_| format!("bad {a}"))?;
                cli.opts
                    .dump_at
                    .push((c, rs.split(',').map(str::to_owned).collect()));
            }
            ("--require-full", None) => cli.require_full = true,
            ("--wasm-sha256", Some(v)) => cli.wasm_sha256 = Some(v.to_ascii_lowercase()),
            ("--no-mask", None) => cli.opts.no_mask = true,
            ("--info", None) => cli.opts.info = true,
            ("--in-process", None) => cli.in_process = true,
            ("--child", None) => cli.child = true,
            ("--list-allowlist", None) => {
                for w in allowlist::PLATFORM_WORDS {
                    println!(
                        "{}[{}]: {}",
                        w.region,
                        w.word,
                        w.why.split_whitespace().collect::<Vec<_>>().join(" ")
                    );
                }
                std::process::exit(0);
            }
            (k, _) if k.starts_with("--") => return Err(format!("unknown option {a}")),
            _ => cli.inputs.push(PathBuf::from(a.clone())),
        }
        if pass {
            cli.passthrough.push(a);
        }
    }
    Ok(cli)
}

fn expand(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .map_err(|e| format!("{}: {e}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.extension().is_some_and(|x| x == "cstrace"))
                .collect();
            v.sort();
            files.extend(v);
        } else if p.is_file() {
            files.push(p.clone());
        } else {
            return Err(format!("{}: no such file or directory", p.display()));
        }
    }
    Ok(files)
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn print_report(r: &Report) {
    let s = &r.stats;
    let tag = match &r.outcome {
        Outcome::Pass => "[PASS]  ",
        Outcome::PortablePrefix { .. } => "[PREFIX]",
        Outcome::RecordedTrap { .. } => "[TRAP]  ",
        Outcome::Fail => "[FAIL]  ",
    };
    let what = match &r.outcome {
        Outcome::PortablePrefix { at_call, addr, len } => format!(
            "portable prefix ended at call {at_call} (UNMAPPED raw-address write of {len} B at wasm address {addr}); "
        ),
        Outcome::RecordedTrap { at_call } => format!("recorded trap at call {at_call} cannot be replayed natively; "),
        _ => String::new(),
    };
    println!(
        "{tag} {}: {what}{} calls replayed, {} digest-checked ({} region digests), {} host writes ({:.2} MB), {} resets, \
         {:.1} MB, end={}, mismatches={}, {:.1} s",
        r.name,
        s.calls,
        s.digest_calls,
        s.region_digests,
        s.writes,
        mb(s.write_bytes),
        s.resets,
        mb(r.file_bytes),
        r.end,
        s.mismatches,
        r.secs
    );
    let mut notes = Vec::new();
    if s.skipped_pointer_results > 0 {
        notes.push(format!(
            "{} pointer results not compared (addresses)",
            s.skipped_pointer_results
        ));
    }
    if s.nan_results > 0 {
        notes.push(format!(
            "{} f32 results matched as NaN = NaN",
            s.nan_results
        ));
    }
    if s.skipped_mem_digests > 0 {
        notes.push(format!(
            "{} whole-memory digests skipped (wasm-only)",
            s.skipped_mem_digests
        ));
    }
    if !notes.is_empty() {
        println!("    ({})", notes.join("; "));
    }
    for m in &r.messages {
        println!("    {m}");
    }
}

fn exit_code_for(o: &Outcome) -> i32 {
    match o {
        Outcome::Pass => CHILD_PASS,
        Outcome::PortablePrefix { .. } => CHILD_PREFIX,
        Outcome::RecordedTrap { .. } => CHILD_TRAP,
        Outcome::Fail => CHILD_FAIL,
    }
}

fn check_sha(file: &Path, want: &Option<String>) -> Result<(), String> {
    let Some(want) = want else { return Ok(()) };
    let (_, h) = trace::TraceReader::open(file)?;
    let got = h.wasm_sha256_hex();
    if &got != want {
        return Err(format!(
            "{}: recorded against wasm {got}, expected {want}",
            file.display()
        ));
    }
    Ok(())
}

/// Whether this build traps integer overflow (profile `checked` or `dev`). Probed at run time:
/// the kernel and the driver are compiled with the same profile settings.
fn overflow_checks_enabled() -> bool {
    let quiet = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let trapped =
        std::panic::catch_unwind(|| std::hint::black_box(u8::MAX) + std::hint::black_box(1u8))
            .is_err();
    std::panic::set_hook(quiet);
    trapped
}

fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let call = replayer::CUR_CALL.load(Ordering::Relaxed);
        let ex = replayer::CUR_EXPORT.load(Ordering::Relaxed);
        let trace = replayer::CUR_TRACE
            .try_lock()
            .map(|t| t.clone())
            .unwrap_or_default();
        if call != u32::MAX {
            let name = replayer::exports().get(ex).map(|e| e.name).unwrap_or("?");
            eprintln!(
                "[chronos_replay] NATIVE KERNEL PANIC in {trace}, call #{call} {name}: {info}"
            );
        } else {
            eprintln!("[chronos_replay] panic in {trace} (outside a kernel call): {info}");
        }
        default(info);
    }));
}

fn main() -> ExitCode {
    let cli = match parse_cli() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("chronos_replay: {e}");
            return usage();
        }
    };
    if cli.inputs.is_empty() {
        return usage();
    }
    let files = match expand(&cli.inputs) {
        Ok(f) if !f.is_empty() => f,
        Ok(_) => {
            eprintln!(
                "FAIL: no .cstrace files found in: {}",
                cli.inputs
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            return ExitCode::from(2);
        }
    };
    let overflow_checks = overflow_checks_enabled();
    install_panic_hook();

    // Child: exactly one trace, outcome in the exit status.
    if cli.child {
        let r = match check_sha(&files[0], &cli.wasm_sha256) {
            Ok(()) => replayer::replay_file(&files[0], &cli.opts),
            Err(e) => {
                println!("[FAIL]   {e}");
                return ExitCode::from(CHILD_FAIL as u8);
            }
        };
        print_report(&r);
        return ExitCode::from(exit_code_for(&r.outcome) as u8);
    }

    // Digest coverage per digest-checked call, from the native layout (every non-alias region is
    // digested; the driver checks the trace's table against exactly this layout).
    let digested: Vec<_> = chronos_spectral::native::regions()
        .into_iter()
        .filter(|r| r.kind != chronos_spectral::native::RegionKind::RingAlias)
        .collect();
    let hashed: usize = digested.iter().map(|r| r.byte_len).sum();
    let exempt = 4 * allowlist::PLATFORM_WORDS.len();
    println!(
        "chronos_replay: {} trace(s), native kernel {} / simd backend {} ({} B arena, overflow checks {}, debug assertions {})",
        files.len(),
        std::env::consts::ARCH,
        chronos_spectral::simd::BACKEND_NAME,
        chronos_spectral::native::arena_bytes(),
        if overflow_checks { "on" } else { "off" },
        if cfg!(debug_assertions) { "on" } else { "off" },
    );
    println!(
        "  digests: {} regions, {} B hashed per digest-checked call; platform-word allowlist exempts {} B ({:.6}%): {}",
        digested.len(),
        hashed,
        exempt,
        100.0 * exempt as f64 / hashed as f64,
        allowlist::PLATFORM_WORDS
            .iter()
            .map(|w| format!("{}[{}]", w.region, w.word))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let (mut pass, mut prefix, mut trap, mut fail) = (0, 0, 0, 0);
    let isolate = !cli.in_process && files.len() > 1;
    let exe = std::env::current_exe().ok();
    for f in &files {
        let code = if isolate {
            let Some(exe) = &exe else {
                eprintln!("FAIL: cannot locate the driver executable for child processes");
                return ExitCode::from(1);
            };
            match Command::new(exe)
                .args(&cli.passthrough)
                .arg("--child")
                .arg(f)
                .status()
            {
                Ok(st) => match st.code() {
                    Some(c) => c,
                    None => {
                        println!("[FAIL]   {}: child terminated by a signal (native kernel abort; see stderr)", f.display());
                        CHILD_FAIL
                    }
                },
                Err(e) => {
                    println!("[FAIL]   {}: cannot spawn child: {e}", f.display());
                    CHILD_FAIL
                }
            }
        } else {
            match check_sha(f, &cli.wasm_sha256) {
                Ok(()) => {
                    let r = replayer::replay_file(f, &cli.opts);
                    print_report(&r);
                    exit_code_for(&r.outcome)
                }
                Err(e) => {
                    println!("[FAIL]   {e}");
                    CHILD_FAIL
                }
            }
        };
        match code {
            CHILD_PASS => pass += 1,
            CHILD_PREFIX => prefix += 1,
            CHILD_TRAP => trap += 1,
            _ => fail += 1,
        }
    }
    let ok = fail == 0 && (!cli.require_full || (prefix == 0 && trap == 0));
    println!(
        "{} {} trace(s): {pass} pass, {prefix} portable prefix, {trap} recorded trap, {fail} fail",
        if ok { "[PASS]" } else { "[FAIL]" },
        files.len()
    );
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
