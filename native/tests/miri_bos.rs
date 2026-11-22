//! Miri & native smoke test for BOS Schlieren (`src/bos.rs`): manufactured inhomogeneous Neumann
//! Poisson solve (DCT-II/III + PCG + curl + RGBA overlay) and reference reset.

use chronos_spectral as k;

#[test]
fn bos_manufactured_solve_and_reset() {
    unsafe {
        k::chronos_init();
        k::chronos_set_resolution(0);
        assert_eq!(k::chronos_set_bos_mode(1), 1);
        assert_eq!(k::chronos_set_bos_gauge_roi(4, 4, 24, 24), 1);
        assert_eq!(k::chronos_set_bos_overlay_kind(0), 1);

        let ok = k::chronos_bos_solve_manufactured_test(0, 1.5, -0.75);
        assert_eq!(ok, 1, "manufactured BOS solve should succeed");

        let telem = std::slice::from_raw_parts(k::chronos_get_bos_telemetry_ptr(), 32);
        assert_eq!(telem[2], 2, "BOS state should be READY");
        let planes = std::slice::from_raw_parts(k::chronos_get_bos_planes_ptr(), 256 * 128 * 6);
        assert!(planes.iter().all(|x| x.is_finite()), "non-finite BOS planes");
        let valid = std::slice::from_raw_parts(k::chronos_get_bos_valid_ptr(), 256 * 128);
        assert!(valid.iter().any(|&v| v != 0), "expected valid BOS nodes");
        let rgba = std::slice::from_raw_parts(k::chronos_get_bos_rgba_ptr(), 256 * 128 * 4);
        assert_eq!(rgba.len(), 256 * 128 * 4);

        k::chronos_reset_bos_ref();
    }
}
