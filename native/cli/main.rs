//! Native high-throughput `chronos` CLI for offline & live video analysis powered by `ChronosEngine`.
//!
//! Streams raw RGBA8 frames directly into `engine.ingest_rgba_mut()` (zero-copy ring ingest),
//! supports multi-file parallel batch execution (`--jobs N`), streaming `--ndjson`, `--hwaccel`,
//! `--crop`, `--720p`, `--raw-in WxH`, `rtsp://` low-latency feeds, and `--fps auto`.

use chronos_spectral::native::{ChronosEngine, TrackSnapshot};
use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Clone)]
struct CliOpts {
    inputs: Vec<String>,
    res_mode: u32, // 0=SD, 1=HD, 2=720p
    fps: Option<f64>,
    lab_mode: u32,
    swarm_mode: u32,
    bos_mode: u32,
    hodge_mode: u32,
    lookahead_k: u32,
    seed: Option<(f32, f32, f32, f32)>,
    crop: Option<(u32, u32, u32, u32)>,
    raw_in: Option<(usize, usize)>,
    hwaccel: bool,
    jobs: usize,
    out_json: Option<String>,
    out_csv: Option<String>,
    out_ndjson: Option<String>,
    max_frames: Option<usize>,
}

fn usage() -> ! {
    eprintln!(
        "Usage:\n  chronos analyze <input1> [input2 ...] [--sd|--hd|--720p] [--fps 30|auto]\n                  \
         [--jobs N] [--hwaccel] [--crop x,y,w,h] [--raw-in WxH]\n                  \
         [--lab-mode 0|1|2] [--swarm 0|1|2] [--bos 0|1|2] [--hodge 0|1|2] [--k 8]\n                  \
         [--seed x,y,w,h] [--out tracks.json] [--csv tracks.csv] [--ndjson stream.ndjson|-]\n                  \
         [--max-frames N]"
    );
    std::process::exit(2);
}

fn parse_args() -> CliOpts {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        usage();
    }
    let mut i = 0usize;
    if args[0] == "analyze" {
        i = 1;
    }

    let mut opts = CliOpts {
        inputs: Vec::new(),
        res_mode: 1,
        fps: Some(30.0),
        lab_mode: 0,
        swarm_mode: 0,
        bos_mode: 0,
        hodge_mode: 0,
        lookahead_k: 8,
        seed: None,
        crop: None,
        raw_in: None,
        hwaccel: false,
        jobs: 1,
        out_json: None,
        out_csv: None,
        out_ndjson: None,
        max_frames: None,
    };

    while i < args.len() {
        match args[i].as_str() {
            "--sd" => opts.res_mode = 0,
            "--hd" => opts.res_mode = 1,
            "--720p" => opts.res_mode = 2,
            "--hwaccel" => opts.hwaccel = true,
            "--fps" => {
                i += 1;
                let s = args.get(i).unwrap_or_else(|| usage());
                if s.eq_ignore_ascii_case("auto") {
                    opts.fps = None;
                } else {
                    opts.fps = Some(s.parse().unwrap_or_else(|_| usage()));
                }
            }
            "--jobs" | "-j" => {
                i += 1;
                opts.jobs = args.get(i).and_then(|s| s.parse::<usize>().ok()).unwrap_or_else(|| usage()).max(1);
            }
            "--lab-mode" => {
                i += 1;
                opts.lab_mode = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--swarm" => {
                i += 1;
                opts.swarm_mode = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--bos" => {
                i += 1;
                opts.bos_mode = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--hodge" => {
                i += 1;
                opts.hodge_mode = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--k" => {
                i += 1;
                opts.lookahead_k = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--seed" => {
                i += 1;
                let spec = args.get(i).unwrap_or_else(|| usage());
                let parts: Vec<f32> = spec.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                if parts.len() != 4 {
                    eprintln!("Invalid --seed format; expected x,y,w,h");
                    std::process::exit(2);
                }
                opts.seed = Some((parts[0], parts[1], parts[2], parts[3]));
            }
            "--crop" => {
                i += 1;
                let spec = args.get(i).unwrap_or_else(|| usage());
                let parts: Vec<u32> = spec.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                if parts.len() != 4 || parts[2] == 0 || parts[3] == 0 {
                    eprintln!("Invalid --crop format; expected x,y,w,h");
                    std::process::exit(2);
                }
                opts.crop = Some((parts[0], parts[1], parts[2], parts[3]));
            }
            "--raw-in" => {
                i += 1;
                let spec = args.get(i).unwrap_or_else(|| usage());
                let parts: Vec<usize> = spec
                    .split(|c| c == 'x' || c == 'X')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                if parts.len() != 2 || parts[0] == 0 || parts[1] == 0 {
                    eprintln!("Invalid --raw-in format; expected WxH");
                    std::process::exit(2);
                }
                opts.raw_in = Some((parts[0], parts[1]));
            }
            "--out" => {
                i += 1;
                opts.out_json = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--csv" => {
                i += 1;
                opts.out_csv = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--ndjson" => {
                i += 1;
                opts.out_ndjson = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--max-frames" => {
                i += 1;
                opts.max_frames = Some(args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage()));
            }
            other if other.starts_with("--") => {
                eprintln!("Unknown option: {other}");
                usage();
            }
            input => opts.inputs.push(input.to_string()),
        }
        i += 1;
    }

    if opts.inputs.is_empty() {
        usage();
    }
    opts
}

fn probe_fps(input: &str) -> f64 {
    if input == "-" {
        return 30.0;
    }
    let Ok(out) = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input,
        ])
        .output()
    else {
        return 30.0;
    };
    if !out.status.success() {
        return 30.0;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        let line = line.trim();
        if let Some((num, den)) = line.split_once('/') {
            if let (Ok(n), Ok(d)) = (num.parse::<f64>(), den.parse::<f64>()) {
                if n > 0.0 && d > 0.0 {
                    let fps = n / d;
                    if (1.0..=1000.0).contains(&fps) {
                        return fps;
                    }
                }
            }
        } else if let Ok(fps) = line.parse::<f64>() {
            if (1.0..=1000.0).contains(&fps) {
                return fps;
            }
        }
    }
    30.0
}

fn derive_batch_path(base: &str, input: &str, multi: bool) -> String {
    if !multi || base == "-" {
        return base.to_string();
    }
    let stem = Path::new(input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip");
    let p = Path::new(base);
    if p.is_dir() {
        return p.join(format!("{stem}.out")).to_string_lossy().into_owned();
    }
    let parent = p.parent().unwrap_or_else(|| Path::new(""));
    let base_stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("out");
    parent
        .join(format!("{base_stem}_{stem}.{ext}"))
        .to_string_lossy()
        .into_owned()
}

struct FrameSummary {
    frame: usize,
    t_sec: f64,
    clutter_db: f32,
    active: usize,
    tracks: Vec<TrackSnapshot>,
}

fn write_ndjson_line<W: Write>(
    wtr: &mut W,
    input: &str,
    frame_idx: usize,
    pts_sec: f64,
    clutter_db: f32,
    tracks: &[TrackSnapshot],
) -> io::Result<()> {
    write!(
        wtr,
        "{{\"input\":\"{}\",\"frame\":{},\"t\":{:.4},\"clutter_db\":{:.2},\"active\":{},\"tracks\":[",
        input.replace('\\', "\\\\").replace('"', "\\\""),
        frame_idx,
        pts_sec,
        clutter_db,
        tracks.len()
    )?;
    for (ti, tr) in tracks.iter().enumerate() {
        let tc = if ti + 1 < tracks.len() { "," } else { "" };
        write!(
            wtr,
            "{{\"id\":{},\"x\":{:.2},\"y\":{:.2},\"vx\":{:.2},\"vy\":{:.2},\"vx_s\":{:.2},\"vy_s\":{:.2},\"vis\":{:.1},\"regime\":{}}}{tc}",
            tr.track_id, tr.x, tr.y, tr.vx, tr.vy, tr.vx_px_s, tr.vy_px_s, tr.vis, tr.regime
        )?;
    }
    writeln!(wtr, "]}}")
}

fn run_one(input: &str, opts: &CliOpts, multi: bool) {
    let mut engine = ChronosEngine::new();
    engine.set_resolution_mode(opts.res_mode);
    if opts.res_mode == 2 && engine.resolution_mode() != 2 {
        eprintln!(
            "Error: --720p requires building chronos with `--features hires-720p` \
             (cargo build --release --manifest-path native/Cargo.toml --target host-tuple --features hires-720p --bin chronos)"
        );
        std::process::exit(2);
    }
    engine.set_params(
        opts.lookahead_k,
        0.035,
        1.45,
        25.0,
        -core::f32::consts::FRAC_PI_2,
        opts.lab_mode,
    );
    engine.set_swarm_mode(opts.swarm_mode);
    engine.set_bos_mode(opts.bos_mode);
    engine.set_hodge_mode(opts.hodge_mode);

    let (w, h) = engine.frame_dims();
    if let Some((rw, rh)) = opts.raw_in {
        if rw != w || rh != h {
            eprintln!("Error: --raw-in {rw}x{rh} does not match active engine dimensions {w}x{h}");
            std::process::exit(2);
        }
    }
    let fps = opts.fps.unwrap_or_else(|| probe_fps(input)).max(1.0);
    let dt_sec = (1.0 / fps) as f32;

    let mut ffmpeg_child = None;
    let mut reader: Box<dyn Read> = if opts.raw_in.is_some() {
        if input == "-" {
            Box::new(io::stdin())
        } else {
            Box::new(File::open(input).unwrap_or_else(|e| {
                eprintln!("Failed to open raw RGBA input {input}: {e}");
                std::process::exit(1);
            }))
        }
    } else {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error"]);
        if opts.hwaccel {
            cmd.args(["-hwaccel", "auto"]);
        }
        if input.starts_with("rtsp://") || input.starts_with("rtsps://") {
            cmd.args(["-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay"]);
        }
        let vf = match opts.crop {
            Some((cx, cy, cw, ch)) => format!("crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:flags=area"),
            None => format!("scale={w}:{h}:flags=area"),
        };
        cmd.args([
            "-i",
            input,
            "-vf",
            &vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
        let mut child = cmd.spawn().unwrap_or_else(|e| {
            eprintln!("Failed to launch ffmpeg for {input}: {e}");
            std::process::exit(1);
        });
        let stdout = child.stdout.take().expect("ffmpeg stdout");
        ffmpeg_child = Some(child);
        Box::new(stdout)
    };

    let csv_path = opts.out_csv.as_ref().map(|p| derive_batch_path(p, input, multi));
    let mut csv_writer = csv_path.as_ref().map(|p| {
        let mut wtr = BufWriter::new(File::create(p).expect("create csv"));
        writeln!(
            wtr,
            "frame,t_sec,slot,track_id,x,y,vx,vy,vx_px_s,vy_px_s,vis,bridged,regime,ode_r2,clutter_db"
        )
        .unwrap();
        wtr
    });

    let ndjson_path = opts.out_ndjson.as_ref().map(|p| derive_batch_path(p, input, multi));
    let mut ndjson_writer: Option<Box<dyn Write>> = ndjson_path.as_ref().map(|p| {
        if p == "-" {
            Box::new(BufWriter::new(io::stdout())) as Box<dyn Write>
        } else {
            Box::new(BufWriter::new(File::create(p).expect("create ndjson"))) as Box<dyn Write>
        }
    });

    let mut frames_out: Vec<FrameSummary> = Vec::new();
    let mut frame_idx = 0usize;
    let mut total_kernel_sec = 0.0f64;
    let mut peak_active = 0usize;
    let wall_t0 = Instant::now();

    loop {
        if let Some(limit) = opts.max_frames {
            if frame_idx >= limit {
                break;
            }
        }
        // Zero-copy read directly into the engine's next RGBA8 ring slot!
        let dst = engine.ingest_rgba_mut();
        let frame_bytes = dst.len();
        let mut read_so_far = 0usize;
        while read_so_far < frame_bytes {
            match reader.read(&mut dst[read_so_far..]) {
                Ok(0) => break,
                Ok(n) => read_so_far += n,
                Err(e) => {
                    eprintln!("Error reading frame stream for {input}: {e}");
                    std::process::exit(1);
                }
            }
        }
        if read_so_far < frame_bytes {
            break;
        }

        let pts_sec = (frame_idx as f64) / fps;
        let kt0 = Instant::now();
        let _disp = engine.step_with_pts(pts_sec, dt_sec);
        if frame_idx == 0 {
            if let Some((sx, sy, sw, sh)) = opts.seed {
                let tid = engine.seed_target(sx, sy, sw, sh);
                eprintln!("[chronos] seeded user target -> T{tid} at ({sx},{sy},{sw}x{sh})");
            }
        }
        total_kernel_sec += kt0.elapsed().as_secs_f64();

        let telem = engine.telemetry();
        let clutter_db = telem[4];
        let tracks = engine.active_tracks();
        if tracks.len() > peak_active {
            peak_active = tracks.len();
        }
        if let Some(wtr) = csv_writer.as_mut() {
            for tr in &tracks {
                writeln!(
                    wtr,
                    "{frame_idx},{pts_sec:.6},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{},{:.4},{:.2}",
                    tr.slot,
                    tr.track_id,
                    tr.x,
                    tr.y,
                    tr.vx,
                    tr.vy,
                    tr.vx_px_s,
                    tr.vy_px_s,
                    tr.vis,
                    tr.bridged,
                    tr.regime,
                    tr.ode_r2,
                    clutter_db
                )
                .unwrap();
            }
        }
        if let Some(wtr) = ndjson_writer.as_mut() {
            if let Err(err) = write_ndjson_line(wtr, input, frame_idx, pts_sec, clutter_db, &tracks) {
                if err.kind() == std::io::ErrorKind::BrokenPipe {
                    break;
                }
                panic!("ndjson write failed: {err}");
            }
        }
        if opts.out_json.is_some() {
            frames_out.push(FrameSummary {
                frame: frame_idx,
                t_sec: pts_sec,
                clutter_db,
                active: tracks.len(),
                tracks,
            });
        }
        frame_idx += 1;
    }

    if let Some(mut child) = ffmpeg_child {
        let _ = child.kill();
        let _ = child.wait();
    }
    if let Some(mut wtr) = csv_writer {
        wtr.flush().unwrap();
    }
    if let Some(mut wtr) = ndjson_writer {
        if let Err(err) = wtr.flush() {
            if err.kind() != std::io::ErrorKind::BrokenPipe {
                panic!("ndjson flush failed: {err}");
            }
        }
    }

    if let Some(json_base) = opts.out_json.as_ref() {
        let json_path = derive_batch_path(json_base, input, multi);
        let mut wtr = BufWriter::new(File::create(&json_path).expect("create json"));
        writeln!(wtr, "{{").unwrap();
        writeln!(wtr, "  \"version\": \"3.2.0\",").unwrap();
        writeln!(wtr, "  \"backend\": \"{}\",", chronos_spectral::simd::BACKEND_NAME).unwrap();
        writeln!(
            wtr,
            "  \"width\": {w}, \"height\": {h}, \"fps\": {fps:.3}, \"frames\": {frame_idx}, \"peak_active\": {peak_active},"
        )
        .unwrap();
        writeln!(
            wtr,
            "  \"kernel_fps\": {:.2},",
            (frame_idx as f64) / total_kernel_sec.max(1e-9)
        )
        .unwrap();
        writeln!(wtr, "  \"timeline\": [").unwrap();
        for (idx, fr) in frames_out.iter().enumerate() {
            let comma = if idx + 1 < frames_out.len() { "," } else { "" };
            write!(
                wtr,
                "    {{\"frame\":{},\"t\":{:.4},\"clutter_db\":{:.2},\"active\":{},\"tracks\":[",
                fr.frame, fr.t_sec, fr.clutter_db, fr.active
            )
            .unwrap();
            for (ti, tr) in fr.tracks.iter().enumerate() {
                let tc = if ti + 1 < fr.tracks.len() { "," } else { "" };
                write!(
                    wtr,
                    "{{\"id\":{},\"x\":{:.2},\"y\":{:.2},\"vx\":{:.2},\"vy\":{:.2},\"vx_s\":{:.2},\"vy_s\":{:.2},\"vis\":{:.1},\"regime\":{}}}{tc}",
                    tr.track_id, tr.x, tr.y, tr.vx, tr.vy, tr.vx_px_s, tr.vy_px_s, tr.vis, tr.regime
                )
                .unwrap();
            }
            writeln!(wtr, "]}}{comma}").unwrap();
        }
        writeln!(wtr, "  ]\n}}").unwrap();
        wtr.flush().unwrap();
    }

    let wall_sec = wall_t0.elapsed().as_secs_f64();
    let kernel_fps = (frame_idx as f64) / total_kernel_sec.max(1e-9);
    let e2e_fps = (frame_idx as f64) / wall_sec.max(1e-9);
    let mean_ms = (total_kernel_sec * 1e3) / (frame_idx.max(1) as f64);
    let to_stderr = opts.out_ndjson.as_deref() == Some("-");
    let summary = format!(
        "chronos ({}, {}x{} @ {:.2}fps, {}): {} frames | kernel {:.1} fps ({:.3} ms/f) | e2e {:.1} fps | peak_active={}",
        chronos_spectral::simd::BACKEND_NAME,
        w,
        h,
        fps,
        input,
        frame_idx,
        kernel_fps,
        mean_ms,
        e2e_fps,
        peak_active
    );
    if to_stderr {
        eprintln!("{summary}");
    } else {
        println!("{summary}");
    }
}

fn main() {
    let opts = parse_args();
    let multi = opts.inputs.len() > 1;
    let n_jobs = opts.jobs.min(opts.inputs.len()).max(1);
    if n_jobs <= 1 {
        for input in &opts.inputs {
            run_one(input, &opts, multi);
        }
    } else {
        let next_idx = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..n_jobs {
                s.spawn(|| loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= opts.inputs.len() {
                        break;
                    }
                    run_one(&opts.inputs[idx], &opts, multi);
                });
            }
        });
    }
}
