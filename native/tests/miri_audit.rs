//! Miri Stacked-Borrows & native SDK verification for v3.4.0 Physics Audit extensions:
//! 1. Per-engine `audio_staging_mut`, `push_audio_block`, `copy_recent_audio`, `set_ingest_meta`, and `reset`.
//! 2. Dense optical-flow continuum fallback (`WIND_MODE_DENSE_AUDIT = 3`) & `dense_flow_planes()` views.
//! 3. Boundary rejection on invalid audio counts / rates / metadata without UB.

use chronos_spectral::native::{ChronosEngine, NativeIngestMeta};

#[test]
fn miri_v34_audio_ingest_reset_and_dense_audit() {
    let mut eng = ChronosEngine::new();
    eng.set_resolution_mode(0);
    assert_eq!(eng.frame_dims(), (320, 180));

    // Audio boundary checks + u64 timestamp crossing 2^32 us (> 4295s)
    assert_eq!(eng.push_audio_block(0, 10_000, 48_000.0, 0.0), 0);
    assert_eq!(eng.push_audio_block(4097, 10_000, 48_000.0, 0.0), 0);
    assert_eq!(eng.push_audio_block(64, 10_000, 100.0, 0.0), 0);
    {
        let stg = eng.audio_staging_mut();
        assert_eq!(stg.len(), 4096);
        for (i, s) in stg.iter_mut().take(128).enumerate() {
            *s = ((i as f32) * 0.05).sin() * 0.25;
        }
    }
    let hi_ts: u64 = 5_000_000_000;
    assert_eq!(eng.push_audio_block(128, hi_ts, 48_000.0, 0.0), 1);
    let mut recent = [0.0f32; 64];
    let (got, end_us, rate_hz) = eng.copy_recent_audio(&mut recent);
    assert_eq!(got, 64);
    assert_eq!(end_us, hi_ts);
    assert!((rate_hz - 48_000.0).abs() < 1.0);
    assert!(recent.iter().all(|v| v.is_finite()));

    // Explicit ingest metadata + dense audit mode (mode 3)
    eng.set_swarm_mode(0);
    assert_eq!(eng.set_hodge_mode(1), 1);
    assert_eq!(eng.set_wind_mode(3), 1);
    let n_frames = if cfg!(miri) { 2usize } else { 5usize };
    for f in 0..n_frames {
        let meta = NativeIngestMeta {
            epoch: 1,
            seq: (f as u32) + 1,
            pts_us: (f as u64) * 33_333,
            age_ms: -1.0,
            flags: (1 << 1) | (1 << 4), // INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_GEOM_KNOWN
            src_w: 320,
            src_h: 180,
            crop_x: 0,
            crop_y: 0,
            crop_w: 320,
            crop_h: 180,
            orientation: 0,
            scan_dir: u32::MAX,
            dst_x: 0,
            dst_y: 0,
            dst_w: 320,
            dst_h: 180,
            gap_mask: 0,
            native_1to1: 1,
        };
        assert_eq!(eng.set_ingest_meta(meta), 1);
        let rgba = eng.ingest_rgba_mut();
        // Compact deterministic pattern
        let shift = (f as u32) * 2;
        for y in 0..180usize {
            let row_off = y * 320 * 4;
            for x in 0..320usize {
                let v = ((x as u32)
                    .wrapping_sub(shift)
                    .wrapping_mul(13)
                    .wrapping_add((y as u32).wrapping_mul(7))
                    & 0xFF) as u8;
                let off = row_off + x * 4;
                rgba[off] = v;
                rgba[off + 1] = v;
                rgba[off + 2] = v;
                rgba[off + 3] = 255;
            }
        }
        eng.step(1.0 / 30.0);
    }

    let (vx, vy, mask) = eng.dense_flow_planes();
    assert_eq!(vx.len(), 2304);
    assert_eq!(vy.len(), 2304);
    assert_eq!(mask.len(), 2304);
    assert!(vx.iter().all(|v| v.is_finite()));
    assert!(vy.iter().all(|v| v.is_finite()));

    // Clean reset preserves no stale wind/audio state
    eng.reset();
    assert_eq!(eng.wind_telemetry()[0], 0);
    assert_eq!(eng.frame_dims(), (320, 180));
}
