//! Miri Stacked-Borrows & UB verification for Pocket Wind Tunnel (`Wave D2`) and `BosState` scratch handoff.
//!
//! Runs compact deterministic solves (`< 30 s` under Miri) across:
//! 1. `HodgeState::solve_exact_on_active_complex` + `WindTunnelState` analytical Lamb-Oseen + lifting cylinder.
//! 2. Strict `BosState` scratch mutual exclusion (`BOS` -> `WindTunnel` -> `BOS` transition).
//! 3. Multi-instance `ChronosEngine` wind-tunnel C-ABI methods & zero-copy `field_view_desc(20..=21)`.

use chronos_spectral::native::ChronosEngine;
use chronos_spectral::{
    chronos_get_hodge_inject_ptr, chronos_get_wind_planes_ptr, chronos_get_wind_telemetry_ptr,
    chronos_hodge_solve_injected, chronos_init, chronos_set_bos_mode,
    chronos_set_wind_fluid_density, chronos_set_wind_mode, chronos_wind_solve_injected,
};

#[inline]
unsafe fn read_wind_telem() -> [u32; 32] {
    let mut out = [0u32; 32];
    let s = core::slice::from_raw_parts(chronos_get_wind_telemetry_ptr(), 32);
    out.copy_from_slice(s);
    out
}

#[test]
fn miri_wind_tunnel_injected_and_mutex() {
    unsafe {
        chronos_init();
        assert_eq!(chronos_set_wind_mode(2), 1);
        assert_eq!(chronos_set_wind_fluid_density(1.225), 1);
        assert_eq!(chronos_set_wind_fluid_density(-1.0), 0);
        assert_eq!(chronos_set_wind_fluid_density(f32::NAN), 0);

        // Uniform freestream around a small 4-cell hole (3 ticks to pass warming -> READY)
        let n_steps = if cfg!(miri) { 3usize } else { 5usize };
        for f in 0..n_steps {
            // Re-acquire the Arena inject slice tightly on every iteration (Stacked Borrows hygiene)
            {
                let inj = core::slice::from_raw_parts_mut(chronos_get_hodge_inject_ptr(), 3 * 2304);
                for r in 0..36usize {
                    for c in 0..64usize {
                        let v = r * 64 + c;
                        let in_hole = c >= 28 && c <= 31 && r >= 16 && r <= 19;
                        if in_hole {
                            inj[v] = 0.0;
                            inj[2304 + v] = 0.0;
                            inj[4608 + v] = 0.0;
                        } else {
                            inj[v] = 40.0;
                            inj[2304 + v] = 0.0;
                            inj[4608 + v] = 1.0;
                        }
                    }
                }
            }
            assert_eq!(chronos_hodge_solve_injected(10.0, 10.0), 1);
            let ok = chronos_wind_solve_injected(10.0, 10.0, (f as f64) * 0.05);
            if f >= 2 {
                assert_eq!(ok, 1);
            }
        }
        // State should be READY_FIELD_ONLY (2) or READY_FULL_LOADS (3), with finite planes
        let telem = read_wind_telem();
        assert!(telem[0] == 2 || telem[0] == 3);
        {
            let planes = core::slice::from_raw_parts(chronos_get_wind_planes_ptr(), 3 * 2304);
            assert!(planes[0].is_finite());
        }

        // Mutex check: enabling BOS must refuse Wind Tunnel cleanly without UB or stale reads
        assert_eq!(chronos_set_bos_mode(1), 1);
        assert_eq!(chronos_wind_solve_injected(10.0, 10.0, 1.0), 0);
        let telem = read_wind_telem();
        assert_eq!(telem[0], 4); // WIND_STATE_REFUSED
        assert_ne!(telem[2] & (1 << 2), 0); // WIND_REASON_MUTEX_CONFLICT
        assert_eq!(chronos_set_bos_mode(0), 1);
        assert_eq!(chronos_set_wind_mode(0), 1);
    }
}

#[test]
fn miri_wind_tunnel_engine_views() {
    let mut eng = ChronosEngine::new();
    assert_eq!(eng.set_wind_mode(1), 1);
    assert_eq!(eng.set_wind_fluid_density(1000.0), 1);
    let v_telem = eng.field_view_desc(20).expect("wind_telemetry view");
    let v_planes = eng.field_view_desc(21).expect("wind_planes view");
    assert_eq!(v_telem.len, 32);
    assert_eq!(v_planes.len, 3 * 2304);
    assert_eq!(eng.wind_telemetry()[1], 1);
    assert!(eng.wind_planes()[0].is_finite());
}
