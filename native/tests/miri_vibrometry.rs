//! Miri & native smoke test for calibrated rolling-shutter vibrometry and audio staging
//! (`src/vibrometry.rs`).

use chronos_spectral as k;

const W: usize = k::SD_FRAME_W;
const H: usize = k::SD_FRAME_H;
const FRAMES: u32 = if cfg!(miri) { 2 } else { 8 };
const DT: f32 = 1.0 / 30.0;

#[test]
fn vibrometry_audio_and_rolling_shutter_ticks() {
    let mut frame = vec![0u8; W * H * 4];
    for y in 0..H {
        for x in 0..W {
            let v = (((x * 7 + y * 3) & 127) + 64) as u8;
            let p = &mut frame[(y * W + x) * 4..][..4];
            p.copy_from_slice(&[v, v, v, 255]);
        }
    }

    unsafe {
        k::chronos_init();
        k::chronos_set_resolution(0);
        assert_eq!(k::chronos_set_rs_calib(2, 27.7, 500.0, 48000.0, 5000.0), 1);

        let audio = std::slice::from_raw_parts_mut(k::chronos_get_audio_staging_ptr(), 1024);
        for (i, s) in audio.iter_mut().enumerate() {
            *s = ((i as f32) * 0.1).sin() * 0.5;
        }

        for f in 0..FRAMES {
            let t_us = (f as f64 * DT as f64 * 1e6).round() as u64;
            let _ = k::chronos_push_audio_block(
                1024,
                t_us as u32,
                (t_us >> 32) as u32,
                48000.0,
                0.0,
            );
            let dst = k::chronos_get_ingest_rgba_ptr();
            core::ptr::copy_nonoverlapping(frame.as_ptr(), dst, frame.len());
            let staged = k::chronos_set_ingest_meta(
                1,
                f + 1,
                t_us as u32,
                (t_us >> 32) as u32,
                -1.0,
                2 | 16,
                W as u32,
                H as u32,
                0,
                0,
                W as u32,
                H as u32,
                0,
                0, // scan_dir = top-to-bottom
                0,
                0,
                W as u32,
                H as u32,
                0,
                1,
            );
            assert_eq!(staged, 1);
            let _ = k::chronos_tick(DT);
        }

        let vtelem = std::slice::from_raw_parts(k::chronos_get_vibro_telemetry_ptr(), 32);
        assert_eq!(vtelem.len(), 32);
        let vplanes = std::slice::from_raw_parts(k::chronos_get_vibro_planes_ptr(), 64 * 4);
        assert!(vplanes.iter().all(|x| x.is_finite()), "non-finite vibro planes");
    }
}
