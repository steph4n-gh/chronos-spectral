//! Drives the real kernel through its host protocol on a small synthetic scene: `chronos_init`,
//! `chronos_set_params`, then a few SD frames (textured background, one bright disc moving 3 px per frame), each
//! written into the ingest ring slot, staged with `chronos_set_ingest_meta` and processed by
//! `chronos_tick` (pre-tick, the internal Fiedler solve, post-tick).
//!
//! Its main use is Miri: it is the smallest test that runs the whole per-frame pipeline over the
//! arena, the raw-pointer SIMD loads and stores and `arena_mut()`:
//!
//! ```text
//! cargo +nightly-2026-10-19 miri test --manifest-path native/Cargo.toml \
//!     --target <host triple> --features simd-force-scalar --test kernel_ticks
//! ```
//!
//! Natively (`cargo test --manifest-path native/Cargo.toml --target host-tuple`, Gate S) it is a
//! fast smoke test of the same sequence on the selected SIMD backend. The kernel is a global
//! singleton, so everything lives in this single test function (this file is its own process).

use chronos_spectral as k;

const W: usize = k::SD_FRAME_W;
const H: usize = k::SD_FRAME_H;
/// Frames per run. Miri interprets every operation (on Apple M4 Pro: `chronos_init` 33 s, the
/// first frame 5.7 min, a frame with the display stage 10 min), so it runs two: the second one
/// reaches the display stage (see `LOOKAHEAD_K`). Natively the extra frames cost nothing.
const FRAMES: u32 = if cfg!(miri) { 2 } else { 24 };
/// Look-ahead depth: the display stage runs on the frame `K` behind the newest one, so with the
/// default `K = 8` it starts only at the 10th frame. Miri uses `K = 1` so that its few frames
/// reach the display stage too; natively the default runs, over more than one ring cycle.
const LOOKAHEAD_K: u32 = if cfg!(miri) { 1 } else { 8 };
const DT: f32 = 1.0 / 30.0;
/// `chronos_set_ingest_meta` flags: presentation timestamp known, geometry known.
const FLAGS: u32 = 2 | 16;

/// Deterministic textured background with a bright disc at (`cx`, `cy`).
fn paint(frame: &mut [u8], cx: f32, cy: f32) {
    for y in 0..H {
        for x in 0..W {
            let h = (x as u32)
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add((y as u32).wrapping_mul(0x85EB_CA77));
            let tex = 60 + ((h >> 27) as u8) + ((x / 16 + y / 16) % 2) as u8 * 30;
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let v = if dx * dx + dy * dy <= 36.0 { 250 } else { tex };
            let p = &mut frame[(y * W + x) * 4..][..4];
            p.copy_from_slice(&[v, v.saturating_sub(10), v.saturating_sub(20), 255]);
        }
    }
}

#[test]
fn init_then_ticks_on_synthetic_frames() {
    let mut frame = vec![0u8; W * H * 4];
    let mut display_slots = Vec::new();
    unsafe {
        k::chronos_init();
        k::chronos_set_resolution(0);
        // The chronos_init defaults, except for the look-ahead depth.
        k::chronos_set_params(
            LOOKAHEAD_K,
            0.035,
            1.45,
            25.0,
            -core::f32::consts::FRAC_PI_2,
            0,
        );
        assert_eq!(k::chronos_get_frame_w() as usize, W);
        assert_eq!(k::chronos_get_frame_h() as usize, H);
        eprintln!("kernel_ticks: chronos_init done");

        for f in 0..FRAMES {
            paint(&mut frame, 60.0 + 3.0 * f as f32, 90.0 + 1.5 * f as f32);
            // The ingest pointer moves with the ring cursor: fetch it for every frame.
            let dst = k::chronos_get_ingest_rgba_ptr();
            core::ptr::copy_nonoverlapping(frame.as_ptr(), dst, frame.len());
            let t_us = (f as f64 * DT as f64 * 1e6).round() as u64;
            let staged = k::chronos_set_ingest_meta(
                1,
                f + 1,
                t_us as u32,
                (t_us >> 32) as u32,
                -1.0,
                FLAGS,
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
            );
            assert_eq!(
                staged, 1,
                "frame {f}: chronos_set_ingest_meta rejected the frame"
            );
            // `chronos_tick` returns the display slot; the ingest status and the commit count
            // are wave-1 telemetry words 8 and 4.
            display_slots.push(k::chronos_tick(DT));
            let w1 = k::chronos_get_wave1_telemetry_ptr();
            assert_eq!(*w1.add(8), k::INGEST_STATUS_OK, "frame {f}: ingest status");
            assert_eq!(*w1.add(4), f + 1, "frame {f}: committed frames");
            eprintln!("kernel_ticks: frame {f} ticked");
        }

        // The outputs a host reads are finite.
        let telem = std::slice::from_raw_parts(k::chronos_get_telemetry_ptr(), k::TELEMETRY_LEN);
        assert!(
            telem.iter().all(|x| x.is_finite()),
            "non-finite telemetry: {:?}",
            telem
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_finite())
                .collect::<Vec<_>>()
        );
        let lap = std::slice::from_raw_parts(k::chronos_get_laplacian_ptr(), 1024);
        assert!(lap.iter().all(|x| x.is_finite()));
    }
    println!(
        "kernel_ticks: {FRAMES} SD frames on simd backend {}, display slots {display_slots:?}",
        k::simd::BACKEND_NAME
    );
}
