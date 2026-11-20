//! Proves the kernel links into a native `std` binary as a Rust library and runs on the
//! selected SIMD backend.
//!
//! The kernel is a global singleton (`static mut ARENA`), so everything that touches it lives in
//! this single test function; integration tests that drive the kernel must not run in parallel.

use chronos_spectral::simd::{f32x4, f32x4_add, f32x4_extract_lane, f32x4_sqrt};

#[test]
fn kernel_links_initialises_and_simd_layer_runs() {
    // The SIMD layer is reachable as a library API (here: the native backend selected by
    // `crate::simd`: NEON on aarch64, SSE on x86_64, the scalar reference elsewhere).
    let v = f32x4_sqrt(f32x4_add(
        f32x4(1.0, 3.0, 8.0, 15.0),
        f32x4(0.0, 1.0, 1.0, 1.0),
    ));
    assert_eq!(f32x4_extract_lane::<0>(v), 1.0);
    assert_eq!(f32x4_extract_lane::<1>(v), 2.0);
    assert_eq!(f32x4_extract_lane::<2>(v), 3.0);
    assert_eq!(f32x4_extract_lane::<3>(v), 4.0);

    // The C-ABI kernel entry points are plain Rust functions of the library.
    unsafe {
        chronos_spectral::chronos_init();
        assert_eq!(
            chronos_spectral::chronos_get_frame_w(),
            chronos_spectral::SD_FRAME_W as u32
        );
        assert!(!chronos_spectral::chronos_get_telemetry_ptr().is_null());
    }
}
