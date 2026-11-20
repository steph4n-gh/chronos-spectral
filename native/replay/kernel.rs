//! The native kernel's C ABI as a dispatch table: export name -> adapter that decodes recorded
//! argument bit patterns, calls the native `chronos_*` function, and returns the result bits.
//!
//! The Rust signature of every entry is checked by the compiler (each adapter calls the real
//! function with arguments of the listed types), the wasm value types are derived from those Rust
//! types, and `exports_cover_every_kernel_export` checks the list against `src/*.rs`.

use crate::trace::{VT_F32, VT_F64, VT_I32};

/// A native return value, as wasm would carry it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ret {
    Void,
    /// i32 result (Rust `u32`), raw bits.
    I32(u32),
    /// f32 result, raw bits.
    F32(u32),
    /// Pointer result (native address). Not comparable with a wasm address.
    Ptr(usize),
}

pub trait Arg: Sized {
    const CODE: u8;
    const SIZE: usize;
    fn read(b: &[u8], o: &mut usize) -> Self;
    fn dummy() -> Self;
}

impl Arg for u32 {
    const CODE: u8 = VT_I32;
    const SIZE: usize = 4;
    fn read(b: &[u8], o: &mut usize) -> u32 {
        let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    }
    fn dummy() -> u32 {
        0
    }
}

impl Arg for f32 {
    const CODE: u8 = VT_F32;
    const SIZE: usize = 4;
    fn read(b: &[u8], o: &mut usize) -> f32 {
        f32::from_bits(u32::read(b, o))
    }
    fn dummy() -> f32 {
        0.0
    }
}

impl Arg for f64 {
    const CODE: u8 = VT_F64;
    const SIZE: usize = 8;
    fn read(b: &[u8], o: &mut usize) -> f64 {
        let v = f64::from_bits(u64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()));
        *o += 8;
        v
    }
    fn dummy() -> f64 {
        0.0
    }
}

pub trait IntoRet {
    const RESULTS: &'static [u8];
    const POINTER: bool = false;
    fn into_ret(self) -> Ret;
}

impl IntoRet for () {
    const RESULTS: &'static [u8] = &[];
    fn into_ret(self) -> Ret {
        Ret::Void
    }
}

impl IntoRet for u32 {
    const RESULTS: &'static [u8] = &[VT_I32];
    fn into_ret(self) -> Ret {
        Ret::I32(self)
    }
}

impl IntoRet for f32 {
    const RESULTS: &'static [u8] = &[VT_F32];
    fn into_ret(self) -> Ret {
        Ret::F32(self.to_bits())
    }
}

impl<T> IntoRet for *const T {
    const RESULTS: &'static [u8] = &[VT_I32];
    const POINTER: bool = true;
    fn into_ret(self) -> Ret {
        Ret::Ptr(self as usize)
    }
}

impl<T> IntoRet for *mut T {
    const RESULTS: &'static [u8] = &[VT_I32];
    const POINTER: bool = true;
    fn into_ret(self) -> Ret {
        Ret::Ptr(self as usize)
    }
}

/// One native export.
pub struct Export {
    pub name: &'static str,
    /// Wasm parameter types implied by the Rust signature.
    pub params: Vec<u8>,
    /// Wasm result types implied by the Rust signature.
    pub results: &'static [u8],
    /// Whether the export returns a pointer.
    pub pointer: bool,
    /// Bytes of recorded argument bits this export consumes.
    pub arg_bytes: usize,
    /// Calls the native function with the recorded argument bits.
    ///
    /// # Safety
    /// `args` must hold `arg_bytes` bytes; the kernel is a global singleton (one caller at a
    /// time).
    pub call: unsafe fn(&[u8]) -> Ret,
}

/// Infers the result types of a function from a never-called closure that calls it.
fn ret_info<R: IntoRet>(_f: fn() -> R) -> (&'static [u8], bool) {
    (R::RESULTS, R::POINTER)
}

macro_rules! exports {
    ($($name:ident ( $($ty:ty),* );)*) => {
        /// Every export of the kernel, in the order of `src/lib.rs`.
        pub fn exports() -> Vec<Export> {
            vec![$(
                {
                    #[allow(unused_unsafe, unused_mut, unused_variables, clippy::unused_unit)]
                    unsafe fn adapter(args: &[u8]) -> Ret {
                        let mut o = 0usize;
                        let r = unsafe { chronos_spectral::$name($(<$ty as Arg>::read(args, &mut o)),*) };
                        r.into_ret()
                    }
                    #[allow(unused_unsafe)]
                    let (results, pointer) = ret_info(|| unsafe { chronos_spectral::$name($(<$ty as Arg>::dummy()),*) });
                    Export {
                        name: stringify!($name),
                        params: vec![$(<$ty as Arg>::CODE),*],
                        results,
                        pointer,
                        arg_bytes: 0 $(+ <$ty as Arg>::SIZE)*,
                        call: adapter,
                    }
                }
            ),*]
        }
    };
}

exports! {
    chronos_init();
    chronos_reset_tracks();
    chronos_set_ingest_pts(f64);
    chronos_set_focus_target(u32);
    chronos_get_focus_target();
    chronos_set_resolution(u32);
    chronos_get_resolution();
    chronos_get_frame_w();
    chronos_get_frame_h();
    chronos_get_pip_dim();
    chronos_get_ring_slot_ptr(u32);
    chronos_set_params(u32, f32, f32, f32, f32, u32);
    chronos_set_spectral_weight(f32);
    chronos_set_audio_latency(f32);
    chronos_trigger_stimulus();
    chronos_get_ingest_rgba_ptr();
    chronos_get_display_rgba_ptr();
    chronos_get_eca_res_ptr();
    chronos_get_pip_rgba_ptr();
    chronos_get_all_pip_rgba_ptr();
    chronos_get_slot_meta_ptr();
    chronos_get_laplacian_ptr();
    chronos_get_fiedler_ptr();
    chronos_set_fiedler_lambda2(f32);
    chronos_set_fiedler_spectrum(f32, f32);
    chronos_solve_fiedler_lanczos();
    chronos_get_fiedler_res_norm();
    chronos_get_telemetry_ptr();
    chronos_set_swarm_mode(u32);
    chronos_get_swarm_mode();
    chronos_get_micro_tracks_ptr();
    chronos_get_continuum_field_ptr();
    chronos_get_rigid_bodies_ptr();
    chronos_get_swarm_telemetry_ptr();
    chronos_get_modal_field_ptr();
    chronos_set_ingest_meta(u32, u32, u32, u32, f32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);
    chronos_get_ring_meta_ptr();
    chronos_get_row_luma_ptr();
    chronos_get_wave1_telemetry_ptr();
    chronos_set_bos_mode(u32);
    chronos_reset_bos_ref();
    chronos_set_bos_gauge_roi(u32, u32, u32, u32);
    chronos_set_bos_overlay_kind(u32);
    chronos_get_bos_telemetry_ptr();
    chronos_get_bos_planes_ptr();
    chronos_get_bos_valid_ptr();
    chronos_get_bos_rgba_ptr();
    chronos_bos_solve_manufactured_test(u32, f32, f32);
    chronos_set_micro_metrology_mode(u32);
    chronos_get_micro_cov_ptr();
    chronos_get_micro_residual_ptr();
    chronos_get_micro_loom_ptr();
    chronos_get_micro_quality_ptr();
    chronos_get_hero_candidates_ptr();
    chronos_get_wave2_telemetry_ptr();
    chronos_set_rs_calib(u32, f32, f32, f32, f32);
    chronos_get_audio_staging_ptr();
    chronos_push_audio_block(u32, u32, u32, f32, f32);
    chronos_get_vibro_telemetry_ptr();
    chronos_get_vibro_planes_ptr();
    chronos_set_hodge_mode(u32);
    chronos_get_hodge_telemetry_ptr();
    chronos_get_hodge_planes_ptr();
    chronos_get_hodge_inject_ptr();
    chronos_hodge_solve_injected(f32, f32);
    chronos_get_conservation_telemetry_ptr();
    chronos_get_conservation_input_ptr();
    chronos_eval_occluded_pair();
    chronos_inspect_field_point(f32, f32);
    chronos_inspect_micro_track(u32);
    chronos_pre_tick(f32);
    chronos_post_tick(f32);
    chronos_tick(f32);
    padic_ultrametric_val(u32, u32);
}

#[cfg(test)]
mod tests {
    use super::exports;

    /// The table lists exactly the `#[no_mangle] extern "C"` functions of the kernel sources.
    #[test]
    fn exports_cover_every_kernel_export() {
        let src = [
            include_str!("../../src/lib.rs"),
            include_str!("../../src/simd_ops.rs"),
        ];
        let mut want: Vec<String> = Vec::new();
        for s in src {
            let mut no_mangle = false;
            for line in s.lines() {
                let t = line.trim();
                if t == "#[no_mangle]" {
                    no_mangle = true;
                    continue;
                }
                if no_mangle && !t.starts_with("#[") {
                    let name = t
                        .split("extern \"C\" fn ")
                        .nth(1)
                        .and_then(|r| r.split('(').next())
                        .unwrap_or_else(|| {
                            panic!("#[no_mangle] item is not an extern \"C\" fn: {t}")
                        });
                    want.push(name.to_owned());
                    no_mangle = false;
                }
            }
        }
        // The native-only cdylib runtime's `rust_eh_personality` is not a kernel export.
        want.retain(|n| n != "rust_eh_personality");
        let mut have: Vec<String> = exports().iter().map(|e| e.name.to_owned()).collect();
        want.sort();
        have.sort();
        assert_eq!(have, want);
    }
}
