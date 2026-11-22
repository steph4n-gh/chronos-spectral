//! Miri & native smoke test for dense swarm, continuum, Rust Lanczos, field/track inspection,
//! focus pinning, reset, and `native::reset_to_pristine()`.

use chronos_spectral as k;

const W: usize = k::SD_FRAME_W;
const H: usize = k::SD_FRAME_H;
const FRAMES: u32 = if cfg!(miri) { 3 } else { 10 };
const DT: f32 = 1.0 / 30.0;

fn paint_swarm(frame: &mut [u8], f: u32) {
    frame.fill(20);
    for px in frame.chunks_exact_mut(4) {
        px[3] = 255;
    }
    for i in 0..24usize {
        let row = i / 6;
        let col = i % 6;
        let cx = (30 + col * 45 + (f as usize) * 2) as isize;
        let cy = (25 + row * 38 + (f as usize)) as isize;
        for dy in -3..=3isize {
            for dx in -3..=3isize {
                if dx * dx + dy * dy <= 9 {
                    let x = (cx + dx).clamp(0, (W - 1) as isize) as usize;
                    let y = (cy + dy).clamp(0, (H - 1) as isize) as usize;
                    let off = (y * W + x) * 4;
                    frame[off] = 240;
                    frame[off + 1] = 230;
                    frame[off + 2] = 210;
                }
            }
        }
    }
}

#[test]
fn swarm_continuum_lanczos_and_inspect_lifecycle() {
    let mut frame = vec![0u8; W * H * 4];
    unsafe {
        k::chronos_init();
        k::chronos_set_resolution(0);
        k::chronos_set_swarm_mode(2);
        assert_eq!(k::chronos_set_micro_metrology_mode(2), 1);

        for f in 0..FRAMES {
            paint_swarm(&mut frame, f);
            let dst = k::chronos_get_ingest_rgba_ptr();
            core::ptr::copy_nonoverlapping(frame.as_ptr(), dst, frame.len());
            let t_us = (f as f64 * DT as f64 * 1e6).round() as u64;
            assert_eq!(
                k::chronos_set_ingest_meta(
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
                    0xFFFF_FFFF,
                    0,
                    0,
                    W as u32,
                    H as u32,
                    0,
                    1,
                ),
                1
            );
            k::chronos_pre_tick(DT);
            let lam2 = k::chronos_solve_fiedler_lanczos();
            assert!(lam2.is_finite());
            let res = k::chronos_get_fiedler_res_norm();
            assert!(res.is_finite());
            let _ = k::chronos_post_tick(DT);
        }

        let _ = k::chronos_inspect_field_point(80.0, 60.0);
        let _ = k::chronos_inspect_field_point(-1.0, f32::NAN);
        let _ = k::chronos_inspect_micro_track(1);
        k::chronos_set_focus_target(1);
        k::chronos_reset_tracks();

        let stelem = std::slice::from_raw_parts(k::chronos_get_swarm_telemetry_ptr(), 64);
        assert!(stelem.iter().all(|x| x.is_finite()), "non-finite swarm telemetry");

        k::native::reset_to_pristine();
    }
}
