//! Multi-instance `ChronosEngine` concurrency and user-seed tests.
//!
//! Proves two completely independent `ChronosEngine` instances run concurrently on separate
//! threads (one 320x180 SD, one 640x360 HD) with zero shared global state, plus `seed_target()`
//! locking onto a stationary object at `t = 0`.

use chronos_spectral::native::ChronosEngine;

fn paint_disc(frame: &mut [u8], w: usize, h: usize, cx: f32, cy: f32, r: f32) {
    let r2 = r * r;
    for y in 0..h {
        for x in 0..w {
            let hash = (x as u32)
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add((y as u32).wrapping_mul(0x85EB_CA77));
            let bg = 50 + ((hash >> 27) as u8);
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let v = if dx * dx + dy * dy <= r2 { 245 } else { bg };
            let off = (y * w + x) * 4;
            frame[off] = v;
            frame[off + 1] = v;
            frame[off + 2] = v;
            frame[off + 3] = 255;
        }
    }
}

#[test]
fn two_independent_engines_run_concurrently_without_interference() {
    let t_sd = std::thread::spawn(|| {
        let mut eng = ChronosEngine::new();
        eng.set_resolution(false);
        assert_eq!(eng.frame_dims(), (320, 180));
        let mut frame = vec![0u8; 320 * 180 * 4];
        for f in 0..16u32 {
            paint_disc(&mut frame, 320, 180, 70.0 + 3.0 * f as f32, 90.0, 6.0);
            eng.write_rgba(&frame).unwrap();
            let _ = eng.step(1.0 / 30.0);
        }
        let telem = eng.telemetry();
        assert!(telem.iter().all(|x| x.is_finite()), "SD non-finite telemetry");
        assert_eq!(telem[30] as u32, 320);
        assert_eq!(telem[31] as u32, 180);
    });

    let t_hd = std::thread::spawn(|| {
        let mut eng = ChronosEngine::new();
        eng.set_resolution(true);
        assert_eq!(eng.frame_dims(), (640, 360));
        let mut frame = vec![0u8; 640 * 360 * 4];
        for f in 0..16u32 {
            paint_disc(&mut frame, 640, 360, 160.0 + 5.0 * f as f32, 180.0, 10.0);
            eng.write_rgba(&frame).unwrap();
            let _ = eng.step(1.0 / 30.0);
        }
        let telem = eng.telemetry();
        assert!(telem.iter().all(|x| x.is_finite()), "HD non-finite telemetry");
        assert_eq!(telem[30] as u32, 640);
        assert_eq!(telem[31] as u32, 360);
    });

    t_sd.join().expect("SD thread panicked");
    t_hd.join().expect("HD thread panicked");
}

#[test]
fn seed_target_locks_stationary_object_and_tracks_motion() {
    let mut eng = ChronosEngine::new();
    eng.set_resolution(false);
    let (w, h) = eng.frame_dims();
    let mut frame = vec![0u8; w * h * 4];

    // Stationary for 5 frames, then begins moving (like a rocket on a launch pad before ignition)
    paint_disc(&mut frame, w, h, 160.0, 100.0, 7.0);
    eng.write_rgba(&frame).unwrap();
    let _ = eng.step(1.0 / 30.0);

    let tid = eng.seed_target(148.0, 88.0, 24.0, 24.0);
    assert!(tid > 0, "seed_target should return a positive track_id");

    for f in 1..20u32 {
        let cx = if f < 5 { 160.0 } else { 160.0 + (f - 4) as f32 * 2.5 };
        paint_disc(&mut frame, w, h, cx, 100.0, 7.0);
        eng.write_rgba(&frame).unwrap();
        let _ = eng.step(1.0 / 30.0);
    }

    let tracks = eng.active_tracks();
    assert!(
        tracks.iter().any(|tr| tr.track_id == tid),
        "seeded track T{tid} should remain locked through static wait and motion; got {:?}",
        tracks
    );
}

#[test]
fn zero_copy_ingest_and_field_views_work() {
    let mut eng = ChronosEngine::new();
    eng.set_resolution(false);
    eng.set_bos_mode(1);
    eng.set_hodge_mode(1);
    let (w, h) = eng.frame_dims();
    for f in 0..24u32 {
        let buf = eng.ingest_rgba_mut();
        assert_eq!(buf.len(), w * h * 4);
        paint_disc(buf, w, h, 80.0 + 3.0 * f as f32, 90.0, 7.0);
        let _ = eng.step_with_pts(f as f64 / 30.0, 1.0 / 30.0);
    }
    assert_eq!(eng.eca_res().len(), w * h);
    assert!(eng.eca_res().iter().all(|v| v.is_finite()));
    for kind in 0..=19u32 {
        let d = eng.field_view_desc(kind).expect("field view kind must exist");
        assert!(!d.ptr.is_null() && d.len > 0);
    }
    let tracks = eng.active_tracks();
    assert!(!tracks.is_empty(), "expected at least one active track");
    assert!(tracks.iter().all(|t| t.vx_px_s.is_finite() && t.vy_px_s.is_finite()));
}

#[cfg(feature = "hires-720p")]
#[test]
fn hires_720p_engine_tracks_and_stays_finite() {
    let mut eng = ChronosEngine::new();
    eng.set_resolution_mode(2);
    assert_eq!(eng.frame_dims(), (1280, 720));
    let (w, h) = eng.frame_dims();
    for f in 0..12u32 {
        let buf = eng.ingest_rgba_mut();
        assert_eq!(buf.len(), w * h * 4);
        paint_disc(buf, w, h, 300.0 + 8.0 * f as f32, 360.0, 20.0);
        let _ = eng.step_with_pts(f as f64 / 30.0, 1.0 / 30.0);
    }
    let telem = eng.telemetry();
    assert!(telem.iter().all(|v| v.is_finite()), "720p non-finite telemetry");
    assert_eq!(telem[30] as u32, 1280);
    assert_eq!(telem[31] as u32, 720);
    assert_eq!(eng.eca_res().len(), 1280 * 720);
}

#[cfg(feature = "hires-720p")]
#[test]
fn hires_720p_sharp_texture_pan_stays_in_bounds() {
    // Regression test for sub-pixel ego-motion tile stride at scale_px=4 (win_span=34 > 18).
    let mut eng = ChronosEngine::new();
    eng.set_resolution_mode(2);
    let (w, h) = eng.frame_dims();
    let tex_w = w + 64;
    let mut rng = 0x9e37_79b9u32;
    let mut noise = vec![0u8; tex_w * h];
    for p in &mut noise {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        *p = ((rng >> 24) as u8).clamp(12, 243);
    }
    for (f, shift) in [0usize, 3, 6, 9, 12].iter().copied().enumerate() {
        let buf = eng.ingest_rgba_mut();
        for y in 0..h {
            let src_row = y * tex_w + shift;
            let dst_row = y * w * 4;
            for x in 0..w {
                let v = noise[src_row + x];
                let o = dst_row + x * 4;
                buf[o] = v;
                buf[o + 1] = v;
                buf[o + 2] = v;
                buf[o + 3] = 255;
            }
        }
        let _ = eng.step_with_pts(f as f64 / 30.0, 1.0 / 30.0);
    }
    let telem = eng.telemetry();
    assert!(telem.iter().all(|v| v.is_finite()), "720p pan telemetry non-finite");
}
