//! Miri & native smoke test for four-branch occluded collision conservation (`src/physics_id.rs`).

use chronos_spectral as k;

#[test]
fn occluded_conservation_staged_eval() {
    unsafe {
        k::chronos_init();
        k::chronos_set_params(8, 0.035, 1.45, 25.0, -core::f32::consts::FRAC_PI_2, 2);

        let inp = std::slice::from_raw_parts_mut(k::chronos_get_conservation_input_ptr(), 32);
        inp.fill(0.0);
        // Symmetric head-on equal-mass elastic encounter behind occluder
        // Track A pre: (100, 100), v=(+40, 0); Track B pre: (200, 100), v=(-40, 0)
        // Post after dt=0.5s: slot 0 at (120, 100), v=(-40, 0); slot 1 at (180, 100), v=(+40, 0)
        inp[0] = 1.0; // valid
        inp[1] = 100.0;
        inp[2] = 100.0;
        inp[3] = 40.0;
        inp[4] = 0.0;
        inp[5] = 200.0;
        inp[6] = 100.0;
        inp[7] = -40.0;
        inp[8] = 0.0;
        inp[9] = 120.0;
        inp[10] = 100.0;
        inp[11] = -40.0;
        inp[12] = 0.0;
        inp[13] = 180.0;
        inp[14] = 100.0;
        inp[15] = 40.0;
        inp[16] = 0.0;
        inp[17] = 0.5; // gap dt
        inp[18] = 1.0; // mass_ratio prior
        inp[19] = 1.0; // mass_known

        let _status = k::chronos_eval_occluded_pair();
        let out = std::slice::from_raw_parts(k::chronos_get_conservation_telemetry_ptr(), 32);
        assert!(out.iter().all(|x| x.is_finite()), "non-finite conservation telemetry");
    }
}
