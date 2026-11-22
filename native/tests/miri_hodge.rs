//! Miri & native smoke test for the 64x36 discrete Hodge 2-complex (`src/hodge.rs`):
//! injected exact + coexact vector field with an internal void hole (`beta_1`).

use chronos_spectral as k;

const GW: usize = 64;
const GH: usize = 36;
const CELLS: usize = GW * GH;

#[test]
fn hodge_injected_solve_with_hole() {
    unsafe {
        k::chronos_init();
        assert_eq!(k::chronos_set_hodge_mode(2), 1);

        let inj = std::slice::from_raw_parts_mut(k::chronos_get_hodge_inject_ptr(), CELLS * 3);
        for j in 0..GH {
            for i in 0..GW {
                let idx = j * GW + i;
                // Carve a 6x6 internal hole centered near (32, 18)
                let in_hole = (29..=34).contains(&i) && (15..=20).contains(&j);
                let dx = i as f32 - 31.5;
                let dy = j as f32 - 17.5;
                let r2 = (dx * dx + dy * dy).max(1.0);
                // Exact gradient + coexact circulation around hole
                inj[idx] = if in_hole { 0.0 } else { 0.25 * dx / r2 - 0.8 * dy / r2 };
                inj[CELLS + idx] = if in_hole { 0.0 } else { 0.25 * dy / r2 + 0.8 * dx / r2 };
                inj[2 * CELLS + idx] = if in_hole { 0.0 } else { 1.0 };
            }
        }

        let ok = k::chronos_hodge_solve_injected(10.0, 10.0);
        assert_eq!(ok, 1, "Hodge injected solve should converge");

        let telem = std::slice::from_raw_parts(k::chronos_get_hodge_telemetry_ptr(), 32);
        assert_eq!(telem[0], 2, "Hodge state should be READY");
        assert_eq!(telem[2], 1, "Hodge solve should be converged");
        let orth_ec_rel = f32::from_bits(telem[15]);
        assert!(orth_ec_rel.is_finite() && orth_ec_rel < 1e-3, "orth_ec_rel={orth_ec_rel}");

        let planes = std::slice::from_raw_parts(k::chronos_get_hodge_planes_ptr(), CELLS * 7);
        assert!(planes.iter().all(|x| x.is_finite()), "non-finite Hodge planes");
    }
}
