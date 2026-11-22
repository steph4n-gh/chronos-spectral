//! Miri & native smoke test for Depth Sensing (`src/depth.rs` & `BosState::step_stereo`):
//! - Option A: monocular parallax + fail-closed pure-rotation gate
//! - Option B: calibrated stereo disparity (`256x128` -> Right-image `64x36` + Hero-16)
//! - Option C: external depth-plane ingest (`128x64`), consume-once staleness, rejected-meta safety,
//!   and provenance firewall (`SURVEYED_LIDAR` vs `ASSUMED_NEURAL_PRIOR`)
//!
//! Strict Stacked Borrows hygiene: raw slices into the static `Arena` are scoped tightly and never
//! retained across any mutating `chronos_*` kernel call.

use chronos_spectral as k;
use std::sync::Mutex;

static KERNEL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn paint_textured_stereo_pair(left: &mut [u8], right: &mut [u8], w: usize, h: usize, disp_px: f32) {
    let sample_luma = |fx: f32, fy: f32| -> u8 {
        let v = 0.50
            + 0.22 * (fx * 0.37 + (fy * 0.19).cos() * 0.9).sin()
            + 0.18 * (fy * 0.43 - (fx * 0.23).sin() * 0.8).cos()
            + 0.08 * ((fx + fy) * 0.61).sin();
        (v.clamp(0.05, 0.95) * 255.0) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * 4;
            let vl = sample_luma(x as f32, y as f32);
            let vr = sample_luma(x as f32 + disp_px, y as f32);
            left[off] = vl;
            left[off + 1] = vl;
            left[off + 2] = vl;
            left[off + 3] = 255;
            right[off] = vr;
            right[off + 1] = vr;
            right[off + 2] = vr;
            right[off + 3] = 255;
        }
    }
}

fn paint_mono_dots(dst: &mut [u8], w: usize, h: usize, cam_x: f32, yaw: f32, f_px: f32) {
    dst.fill(20);
    for px in dst.chunks_exact_mut(4) {
        px[3] = 255;
    }
    let cx = (w as f32) * 0.5;
    let cy = (h as f32) * 0.5;
    let cyaw = yaw.cos();
    let syaw = yaw.sin();
    for idx in 0..32usize {
        let gx = (idx % 8) as f32;
        let gy = (idx / 8) as f32;
        let z = 4.5 + ((idx * 37) % 11) as f32;
        let u0 = 28.0 + gx * 36.0 + ((idx % 3) as f32) * 4.0;
        let v0 = 24.0 + gy * 40.0 + ((idx % 2) as f32) * 5.0;
        let wx = ((u0 - cx) * z) / f_px - cam_x;
        let wy = ((v0 - cy) * z) / f_px;
        let wz = z;
        let xc = cyaw * wx - syaw * wz;
        let zc = syaw * wx + cyaw * wz;
        if zc < 1.0 {
            continue;
        }
        let u = cx + (f_px * xc) / zc;
        let v = cy + (f_px * wy) / zc;
        if u < 4.0 || u > (w as f32) - 5.0 || v < 4.0 || v > (h as f32) - 5.0 {
            continue;
        }
        let x0 = (u - 3.0).floor().max(0.0) as usize;
        let x1 = (u + 3.0).ceil().min((w - 1) as f32) as usize;
        let y0 = (v - 3.0).floor().max(0.0) as usize;
        let y1 = (v + 3.0).ceil().min((h - 1) as f32) as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let d2 = (x as f32 - u) * (x as f32 - u) + (y as f32 - v) * (y as f32 - v);
                let g = (-d2 / 2.2).exp();
                if g < 0.05 {
                    continue;
                }
                let o = (y * w + x) * 4;
                let lum = (20.0 + g * 225.0).min(255.0) as u8;
                dst[o] = lum;
                dst[o + 1] = lum;
                dst[o + 2] = lum;
            }
        }
    }
}

#[inline]
unsafe fn read_depth_telem() -> [u32; 32] {
    let mut out = [0u32; 32];
    let s = std::slice::from_raw_parts(k::chronos_get_depth_telemetry_ptr(), 32);
    out.copy_from_slice(s);
    out
}

/// Fast Miri-sized ABI & Stacked Borrows test:
/// - Under Miri (`cfg!(miri)`): exercises `chronos_init()`, full depth FFI pointer / Stacked Borrows
///   re-borrow hygiene across valid + rejected `chronos_set_depth_external_meta` calls, ONE SD
///   `chronos_tick` with active external depth (`step_external` + `export_hero_depth` + `Arena`),
///   and `chronos_depth_capture_stereo_left()` BOS/Stereo pointer hygiene (skips the 32,768-node
///   IC-LK stereo sweep that would exceed CI's 75-min interpreter budget).
/// - Natively (`!cfg!(miri)`): also runs the rejected-meta stale tick and full textured stereo pair
///   tick (`READY_STEREO` / `SURVEYED_STEREO`).
#[test]
fn miri_depth_abi_smoke() {
    let _guard = KERNEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        k::chronos_init();
        k::chronos_set_resolution(0); // 320x180 SD

        // 1. External LiDAR staging + Stacked Borrows re-borrow hygiene across valid & rejected setters
        assert_eq!(k::chronos_set_depth_mode(3), 1);
        {
            let ext_slice = std::slice::from_raw_parts_mut(k::chronos_get_depth_ingest_ptr(), 128 * 64);
            for v in ext_slice[..64 * 32].iter_mut() {
                *v = 2.5;
            }
        }
        assert_eq!(k::chronos_set_depth_external_meta(64, 32, 300.0, 1, 7), 1);

        // Freshly re-acquire ingest pointer after mutating kernel call (Stacked Borrows check)
        {
            let ext_slice = std::slice::from_raw_parts_mut(k::chronos_get_depth_ingest_ptr(), 128 * 64);
            for v in ext_slice[..64 * 32].iter_mut() {
                *v = 9.0;
            }
        }
        assert_eq!(k::chronos_set_depth_external_meta(200, 32, 300.0, 3, 8), 0);

        if cfg!(miri) {
            // Re-stage valid external plane and execute ONE full SD chronos_tick under Miri (~5 min)
            {
                let ext_slice = std::slice::from_raw_parts_mut(k::chronos_get_depth_ingest_ptr(), 128 * 64);
                for v in ext_slice[..64 * 32].iter_mut() {
                    *v = 2.5;
                }
            }
            assert_eq!(k::chronos_set_depth_external_meta(64, 32, 300.0, 1, 9), 1);
            let _ = k::chronos_tick(1.0 / 30.0);
            let telem = read_depth_telem();
            assert_eq!(telem[2], 7, "expected DEPTH_STATE_READY_EXTERNAL");
            assert_eq!(telem[4], 5, "expected DEPTH_PROV_SURVEYED_LIDAR");
            assert_eq!(telem[12], 9, "expected seq 9");
            {
                let field = std::slice::from_raw_parts(k::chronos_get_depth_field_ptr(), 64 * 36 * 2);
                assert!((field[0] - 0.4).abs() < 0.02, "expected inv_z=0.4, got {}", field[0]);
                let _tracks = std::slice::from_raw_parts(k::chronos_get_depth_tracks_ptr(), 3072 * 2);
                let _flags = std::slice::from_raw_parts(k::chronos_get_depth_track_flags_ptr(), 3072);
                let _hero = std::slice::from_raw_parts(k::chronos_get_depth_hero_ptr(), 16 * 4);
            }
            // Exercise Stereo Left capture FFI & BOS isolation without sweeping 32,768 nodes in Miri
            assert_eq!(k::chronos_set_depth_mode(2), 1);
            assert_eq!(k::chronos_set_depth_calib(500.0, 0.0, 0.012, 2.0, -1), 1);
            let w = k::chronos_get_frame_w() as usize;
            let h = k::chronos_get_frame_h() as usize;
            {
                let dst_l = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
                dst_l[..64].fill(128);
            }
            assert_eq!(k::chronos_depth_capture_stereo_left(), 1);
            let bos_telem = std::slice::from_raw_parts(k::chronos_get_bos_telemetry_ptr(), 32);
            assert_eq!(bos_telem[2], 0, "stereo capture must keep BOS_STATE_OFF");
            return;
        }

        // Native full 3-tick path: rejected external -> valid external -> textured stereo pair
        let _ = k::chronos_tick(1.0 / 30.0);
        let telem = read_depth_telem();
        assert_ne!(telem[2], 7, "rejected external meta must fail closed");
        assert_eq!(telem[4], 0, "rejected external meta must publish PROV_NONE");

        {
            let ext_slice = std::slice::from_raw_parts_mut(k::chronos_get_depth_ingest_ptr(), 128 * 64);
            for v in ext_slice[..64 * 32].iter_mut() {
                *v = 2.5;
            }
        }
        assert_eq!(k::chronos_set_depth_external_meta(64, 32, 300.0, 1, 9), 1);
        let _ = k::chronos_tick(1.0 / 30.0);
        let telem = read_depth_telem();
        assert_eq!(telem[2], 7, "expected DEPTH_STATE_READY_EXTERNAL");
        assert_eq!(telem[4], 5, "expected DEPTH_PROV_SURVEYED_LIDAR");

        // 2. Calibrated Stereo Left capture + Right tick (exercises BosState stereo borrow & export)
        k::chronos_reset_tracks();
        assert_eq!(k::chronos_set_depth_mode(2), 1);
        assert_eq!(k::chronos_set_depth_calib(500.0, 0.0, 0.012, 2.0, -1), 1);
        let w = k::chronos_get_frame_w() as usize;
        let h = k::chronos_get_frame_h() as usize;
        let mut left = vec![0u8; w * h * 4];
        let mut right = vec![0u8; w * h * 4];
        paint_textured_stereo_pair(&mut left, &mut right, w, h, 1.2);
        {
            let dst_l = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
            dst_l.copy_from_slice(&left);
        }
        assert_eq!(k::chronos_depth_capture_stereo_left(), 1);
        {
            let dst_r = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
            dst_r.copy_from_slice(&right);
        }
        let _ = k::chronos_tick(1.0 / 30.0);
        let telem = read_depth_telem();
        assert_eq!(telem[2], 6, "expected DEPTH_STATE_READY_STEREO");
        assert_eq!(telem[4], 4, "expected DEPTH_PROV_SURVEYED_STEREO");
    }
}

#[test]
fn depth_external_provenance_and_stereo_smoke() {
    let _guard = KERNEL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        k::chronos_init();
        k::chronos_set_resolution(0); // 320x180 SD for fast native/Miri execution

        // 1. Option C: External depth plane with SURVEYED_LIDAR provenance
        assert_eq!(k::chronos_set_depth_mode(3), 1);
        assert_eq!(k::chronos_get_depth_mode(), 3);
        {
            let ext_slice = std::slice::from_raw_parts_mut(k::chronos_get_depth_ingest_ptr(), 128 * 64);
            for y in 0..32usize {
                for x in 0..64usize {
                    ext_slice[y * 64 + x] = if x < 32 { 2.5 } else { 8.0 };
                }
            }
        }
        assert_eq!(k::chronos_set_depth_external_meta(64, 32, 300.0, 1, 42), 1);
        let _ = k::chronos_tick(1.0 / 30.0);

        let telem = read_depth_telem();
        assert_eq!(telem[2], 7, "expected DEPTH_STATE_READY_EXTERNAL");
        assert_eq!(telem[3], 0, "SURVEYED_LIDAR must not set neural-prior reason bit");
        assert_eq!(telem[4], 5, "expected DEPTH_PROV_SURVEYED_LIDAR");
        assert_eq!(telem[9], 64 * 36, "all 2304 field cells should be valid");
        assert_eq!(telem[12], 42, "expected seq 42");

        // Next tick without fresh external metadata must fail closed (consume-once staleness)
        // Re-acquire ingest slice freshly for strict Stacked Borrows compliance
        {
            let ext_slice = std::slice::from_raw_parts_mut(k::chronos_get_depth_ingest_ptr(), 128 * 64);
            for v in ext_slice[..64 * 32].iter_mut() {
                *v = 9.0;
            }
        }
        assert_eq!(k::chronos_set_depth_external_meta(200, 32, 300.0, 3, 43), 0);
        let _ = k::chronos_tick(1.0 / 30.0);
        let telem = read_depth_telem();
        assert_ne!(telem[2], 7, "stale/rejected external plane must fail closed");
        assert_eq!(telem[4], 0, "stale/rejected external plane must publish PROV_NONE");
        assert_ne!(telem[3] & (1 << 4), 0, "expected DEPTH_REASON_EXT_STALE");

        // Switch provenance to ASSUMED_NEURAL_PRIOR -> provenance firewall bit (1<<5) must assert
        {
            let ext_slice = std::slice::from_raw_parts_mut(k::chronos_get_depth_ingest_ptr(), 128 * 64);
            for y in 0..32usize {
                for x in 0..64usize {
                    ext_slice[y * 64 + x] = if x < 32 { 2.5 } else { 8.0 };
                }
            }
        }
        assert_eq!(k::chronos_set_depth_external_meta(64, 32, 300.0, 3, 44), 1);
        let _ = k::chronos_tick(1.0 / 30.0);
        let telem = read_depth_telem();
        assert_eq!(telem[4], 6, "expected DEPTH_PROV_ASSUMED_NEURAL_PRIOR");
        assert_ne!(
            telem[3] & (1 << 5),
            0,
            "ASSUMED_NEURAL_PRIOR must assert DEPTH_REASON_ASSUMED_NEURAL_PRIOR"
        );

        {
            let field = std::slice::from_raw_parts(k::chronos_get_depth_field_ptr(), 64 * 36 * 2);
            assert!(field.iter().all(|v| v.is_finite()));
            let left_iz = field[(18 * 64 + 8) * 2];
            let right_iz = field[(18 * 64 + 54) * 2];
            assert!((left_iz - 0.4).abs() < 0.02, "left inv_z={left_iz}");
            assert!((right_iz - 0.125).abs() < 0.02, "right inv_z={right_iz}");
        }

        // 2. Option B: Calibrated Stereo BOS (d = 1.2 px, f = 500 px, B = 0.012 m -> Z = 5.0 m)
        k::chronos_reset_tracks();
        assert_eq!(k::chronos_set_depth_mode(2), 1);
        assert_eq!(k::chronos_set_depth_calib(500.0, 0.0, 0.012, 2.0, -1), 1);
        let w = k::chronos_get_frame_w() as usize;
        let h = k::chronos_get_frame_h() as usize;
        let mut left = vec![0u8; w * h * 4];
        let mut right = vec![0u8; w * h * 4];
        paint_textured_stereo_pair(&mut left, &mut right, w, h, 1.2);

        {
            let dst_l = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
            dst_l.copy_from_slice(&left);
        }
        assert_eq!(k::chronos_depth_capture_stereo_left(), 1);

        {
            let dst_r = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
            dst_r.copy_from_slice(&right);
        }
        let _ = k::chronos_tick(1.0 / 30.0);

        let telem = read_depth_telem();
        assert_eq!(telem[2], 6, "expected DEPTH_STATE_READY_STEREO");
        assert_eq!(telem[4], 4, "expected DEPTH_PROV_SURVEYED_STEREO");
        let med_z = f32::from_bits(telem[29]);
        let epi_rms = f32::from_bits(telem[28]);
        assert!(
            (med_z - 5.0).abs() < 0.25,
            "expected stereo med_z near 5.0 m, got {med_z}"
        );
        assert!(epi_rms < 0.08, "expected epi_rms < 0.08 px, got {epi_rms}");

        // BOS ABI must stay OFF while stereo owns the buffer (no BOS_STATE_READY leak)
        {
            let bos_telem = std::slice::from_raw_parts(k::chronos_get_bos_telemetry_ptr(), 32);
            assert_eq!(bos_telem[2], 0, "stereo must not leak BOS_STATE_READY");
        }

        // Second tick without fresh left capture must NOT synthesize stereo from single video stream
        {
            let dst_r2 = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
            dst_r2.copy_from_slice(&right);
        }
        let _ = k::chronos_tick(1.0 / 30.0);
        let telem = read_depth_telem();
        assert_ne!(telem[2], 6, "unpaired frame must not stay READY_STEREO");
        assert_eq!(telem[4], 0, "unpaired frame must publish PROV_NONE");

        // 3. Option A: Monocular Parallax (pure lateral -> READY_PARALLAX, pure yaw -> REFUSED_PURE_ROTATION)
        k::chronos_reset_tracks();
        k::chronos_set_swarm_mode(2);
        assert_eq!(k::chronos_set_depth_mode(1), 1);
        assert_eq!(k::chronos_set_depth_calib(250.0, 0.30, 0.0, 32.0, 0), 1);
        let dt = 1.0f32 / 30.0;
        for f in 0..24u32 {
            {
                let dst = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
                paint_mono_dots(dst, w, h, 0.30 * (f as f32) * dt, 0.0, 250.0);
            }
            let _ = k::chronos_tick(dt);
        }
        let telem = read_depth_telem();
        assert_eq!(telem[2], 2, "Option A lateral expected READY_PARALLAX(2), got {}", telem[2]);
        assert_eq!(telem[4], 2, "Option A lateral expected MONO_METRIC(2), got {}", telem[4]);
        assert!(telem[5] >= 12, "Option A expected >= 12 valid tracks, got {}", telem[5]);

        // Pure tripod yaw must fail closed with 0 depths
        k::chronos_reset_tracks();
        for f in 0..22u32 {
            {
                let dst = std::slice::from_raw_parts_mut(k::chronos_get_ingest_rgba_ptr(), w * h * 4);
                paint_mono_dots(dst, w, h, 0.0, 0.04 * (f as f32) * dt, 250.0);
            }
            let _ = k::chronos_tick(dt);
        }
        let telem = read_depth_telem();
        assert_eq!(telem[2], 4, "Option A pure yaw expected REFUSED_PURE_ROTATION(4), got {}", telem[2]);
        assert_eq!(telem[5], 0, "Option A pure yaw must emit 0 valid depths");

        // 4. Reject out-of-range API calls cleanly
        assert_eq!(k::chronos_set_depth_mode(99), 0);
        assert_eq!(k::chronos_set_depth_calib(-1.0, 0.1, 0.1, 10.0, 0), 0);
        assert_eq!(k::chronos_set_depth_calib(500.0, 0.1, 0.1, 80.0, 0), 0);
        assert_eq!(k::chronos_set_depth_external_meta(200, 64, 500.0, 1, 1), 0);
        assert_eq!(k::chronos_set_depth_external_meta(64, 32, 500.0, 9, 1), 0);
    }
}
