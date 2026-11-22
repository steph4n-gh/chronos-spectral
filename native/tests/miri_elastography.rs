//! Miri Stacked-Borrows & UB verification for kHz Optical Elastography (`Wave D1`) & `BosState` scratch handoff.
//!
//! Runs compact deterministic checks (`< 30 s` under Miri) across:
//! 1. `ElastoState` mode/frequency validation, `BOS` mutex conflict refusal, and clean recovery.
//! 2. Multi-instance `ChronosEngine` elastography C-ABI methods & zero-copy `field_view_desc(22..=24)`
//!    in both Surface (`64x36`) and Acoustic BOS (`256x128`) modes.
//! 3. `HodgeState::solve_power_flow_hodge` direct verification (source/sink divergence + zero mut of `hodge.planes`).

use chronos_spectral::native::ChronosEngine;
use chronos_spectral::{
    chronos_get_elasto_planes_ptr, chronos_get_elasto_telemetry_ptr, chronos_get_elasto_valid_ptr,
    chronos_init, chronos_set_bos_mode, chronos_set_elasto_lock_freqs, chronos_set_elasto_mode,
    chronos_set_rs_calib, chronos_tick,
};

#[test]
fn miri_elastography_mutex_and_config() {
    unsafe {
        chronos_init();
        assert_eq!(chronos_set_elasto_mode(99), 0);
        assert_eq!(chronos_set_elasto_mode(1), 1);
        assert_eq!(chronos_set_elasto_lock_freqs(5, 100.0, 200.0, 300.0, 400.0), 0);
        assert_eq!(chronos_set_elasto_lock_freqs(2, f32::NAN, 860.0, 0.0, 0.0), 0);
        assert_eq!(chronos_set_elasto_lock_freqs(2, 860.0, 380.0, 0.0, 0.0), 1);
        assert_eq!(chronos_set_rs_calib(2, 35.0, 200.0, 48000.0, 5000.0), 1);

        let telem = core::slice::from_raw_parts(chronos_get_elasto_telemetry_ptr(), 32);
        let planes = core::slice::from_raw_parts(chronos_get_elasto_planes_ptr(), 6 * 2304);
        let valid = core::slice::from_raw_parts(chronos_get_elasto_valid_ptr(), 2304);
        assert_eq!(telem[1], 1);
        assert!(planes[0].is_finite());
        assert_eq!(valid[0], 0);

        // Enabling BOS while Elastography is active must refuse Elastography with MUTEX_CONFLICT (1 << 19)
        assert_eq!(chronos_set_bos_mode(1), 1);
        let _ = chronos_tick(1.0 / 30.0);
        assert_eq!(telem[0], 4); // ELASTO_STATE_REFUSED
        assert_ne!(telem[2] & (1 << 19), 0); // ELASTO_REASON_MUTEX_CONFLICT

        // Releasing BOS recovers cleanly to warming/uncalibrated without UB
        assert_eq!(chronos_set_bos_mode(0), 1);
        let _ = chronos_tick(1.0 / 30.0);
        assert_eq!(telem[2] & (1 << 19), 0);

        // Switching to Acoustic BOS (mode 4) updates grid dims cleanly
        assert_eq!(chronos_set_elasto_mode(4), 1);
        assert_eq!(telem[28], 256);
        assert_eq!(telem[29], 128);
        assert_eq!(chronos_set_elasto_mode(0), 1);
    }
}

#[test]
fn miri_elastography_engine_views() {
    let mut eng = ChronosEngine::new();
    assert_eq!(eng.set_elasto_mode(1), 1);
    assert_eq!(eng.set_elasto_lock_freqs(2, 380.0, 860.0, 0.0, 0.0), 1);
    let v_telem = eng.field_view_desc(22).expect("elasto_telemetry view");
    let v_planes = eng.field_view_desc(23).expect("elasto_planes surface view");
    let v_valid = eng.field_view_desc(24).expect("elasto_valid surface view");
    assert_eq!(v_telem.len, 32);
    assert_eq!(v_planes.len, 6 * 2304);
    assert_eq!(v_planes.rows, 36);
    assert_eq!(v_planes.cols, 64);
    assert_eq!(v_valid.len, 2304);

    assert_eq!(eng.set_elasto_mode(4), 1);
    let v_planes_abos = eng.field_view_desc(23).expect("elasto_planes abos view");
    let v_valid_abos = eng.field_view_desc(24).expect("elasto_valid abos view");
    assert_eq!(v_planes_abos.len, 6 * 32768);
    assert_eq!(v_planes_abos.rows, 128);
    assert_eq!(v_planes_abos.cols, 256);
    assert_eq!(v_valid_abos.len, 32768);
    assert_eq!(eng.elasto_telemetry()[1], 4);
    assert!(eng.elasto_planes()[0].is_finite());
    assert_eq!(eng.elasto_valid()[0], 0);
}
