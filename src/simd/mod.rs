//! Portable SIMD layer: the WebAssembly SIMD128 API on every target.
//!
//! The kernel is written against `core::arch::wasm32`. This module exposes the same names and
//! signatures (`v128`, the lane ops, `v128_*`, the `f32x4(..)`-style constructors and the
//! const-generic `*_extract_lane` / `*_replace_lane` / `*_shuffle` / `v128_store*_lane`) for
//! every op the kernel uses, so kernel code imports `crate::simd::*` and is otherwise unchanged.
//! The WebAssembly SIMD specification is the single definition of what each op computes; every
//! backend must reproduce it bit for bit (NaN payloads excepted, which the spec leaves open).
//!
//! # Backends
//!
//! A backend is an ordinary module that defines every item in `layer_items!` with exactly the
//! `core::arch::wasm32` signature. Backends compile side by side; `crate::simd::*` re-exports the
//! selected one.
//!
//! | target                                      | selected backend           | status        |
//! |---------------------------------------------|----------------------------|---------------|
//! | `wasm32`                                    | `core::arch::wasm32`       | shipped       |
//! | `aarch64` (NEON)                            | `simd::neon`               | implemented   |
//! | `x86_64` with `sse4.1` + `ssse3` enabled    | `simd::sse`                | implemented   |
//! | any other native target, or the             | `simd::scalar`             | implemented   |
//! | `simd-force-scalar` feature                 |                            |               |
//!
//! On wasm32 the layer is a pure re-export of `core::arch::wasm32`, which is what keeps the
//! shipped `web/chronos_spectral.wasm` byte-identical. `scalar` is the portable reference
//! (oracle) implementation; it is always compiled natively and, behind the `simd-conformance`
//! feature, on wasm32 as well so it can be checked against real SIMD128 in V8.
//!
//! ## Adding the `neon` / `sse` backend
//!
//! 1. Create `src/simd/neon.rs` (or `sse.rs`) implementing every item in `layer_items!`.
//!    `v128` must be a 16-byte, 16-aligned `Copy` type: the kernel casts `*const f32` to
//!    `*const v128` and stores `v128` in structs.
//! 2. Declare it next to `pub mod scalar;` below, gated on its target, for example
//!    `#[cfg(all(target_arch = "aarch64", target_feature = "neon"))] pub mod neon;`, so it builds
//!    even when `simd-force-scalar` selects the scalar backend.
//! 3. Add its re-export line in the "Backend selection" block below and exclude its condition
//!    from the scalar line's `cfg`, keeping the conditions mutually exclusive.
//! 4. `cargo test --target host-tuple` then runs the conformance suite against the new backend
//!    with no new test code: `selected_backend_matches_scalar_reference` compares
//!    `crate::simd` (now the new backend) with `crate::simd::scalar`. To also compare it while
//!    `simd-force-scalar` is on, add one line to the tests in `conformance.rs`:
//!    `backend_table!(neon_table, crate::simd::neon);` plus a call to `assert_pair_matches`.
//!
//! # One SIMD path on every target
//!
//! Kernel code has no `cfg(target_arch = "wasm32")` SIMD branches and no hand-written non-wasm
//! alternatives: every backend runs the same sequence of layer ops, so results can match bit for
//! bit. The removed alternatives are marked in the kernel sources with
//! `// [simd-layer] non-wasm fallback removed; blank lines keep Wasm panic line numbers`,
//! followed by blank lines. Those blank lines are deliberate: panic `Location`s (file, line,
//! column of every bounds check, `unwrap`, ...) are stored in the shipped Wasm's data section,
//! so deleting lines would change `web/chronos_spectral.wasm`. Collapse the gaps in the first
//! change that rebuilds and commits the Wasm for another reason.
//!
//! # Native builds
//!
//! `.cargo/config.toml` keeps wasm32 as the default target, so native commands name the host:
//! `cargo check --target host-tuple`, `cargo test --target host-tuple`. Rust consumers (native
//! integration tests, host binaries) depend on `native/Cargo.toml`, which builds this same source
//! as an `rlib`; the root package stays a `cdylib` so the Wasm keeps fat LTO.

/// Invokes `$m!($($prefix)* ; <every layer item>)`.
///
/// This list is the layer: the re-export below, the scalar backend, every future backend and the
/// conformance coverage check are all driven by it. Adding an op means adding it here, to every
/// backend, and to the conformance table (the coverage test fails until all three agree).
macro_rules! layer_items {
    ($m:ident ! ( $($prefix:tt)* )) => {
        $m! { $($prefix)* ;
            // Type and constructors.
            v128, f32x4, f64x2, i16x8, i32x4, u32x4, u64x2,
            // f32x4.
            f32x4_abs, f32x4_add, f32x4_convert_i32x4, f32x4_convert_u32x4, f32x4_div,
            f32x4_extract_lane, f32x4_floor, f32x4_ge, f32x4_gt, f32x4_le, f32x4_lt, f32x4_max,
            f32x4_min, f32x4_mul, f32x4_nearest, f32x4_neg, f32x4_pmax, f32x4_pmin,
            f32x4_replace_lane, f32x4_splat, f32x4_sqrt, f32x4_sub,
            // f64x2.
            f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_splat, f64x2_sqrt, f64x2_sub,
            // i8x16 / u8x16.
            i8x16_add, i8x16_bitmask, i8x16_max, i8x16_min, i8x16_narrow_i16x8, i8x16_shuffle,
            i8x16_splat, i8x16_sub,
            u8x16_add_sat, u8x16_ge, u8x16_gt, u8x16_le, u8x16_lt, u8x16_max, u8x16_min,
            u8x16_narrow_i16x8, u8x16_splat, u8x16_sub_sat,
            // i16x8 / u16x8.
            i16x8_add, i16x8_extadd_pairwise_i8x16, i16x8_extend_high_i8x16,
            i16x8_extend_low_i8x16, i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i16x8_lt,
            i16x8_max, i16x8_min, i16x8_narrow_i32x4, i16x8_q15mulr_sat, i16x8_shl, i16x8_shr,
            i16x8_shuffle, i16x8_splat, i16x8_sub, i16x8_sub_sat,
            u16x8_add, u16x8_extadd_pairwise_u8x16, u16x8_extend_high_u8x16,
            u16x8_extend_low_u8x16, u16x8_extmul_high_u8x16, u16x8_extmul_low_u8x16, u16x8_mul,
            u16x8_shr, u16x8_splat,
            // i32x4 / u32x4.
            i32x4_add, i32x4_all_true, i32x4_bitmask, i32x4_dot_i16x8, i32x4_eq,
            i32x4_extadd_pairwise_i16x8, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8,
            i32x4_extmul_high_i16x8, i32x4_extmul_low_i16x8, i32x4_extract_lane, i32x4_ge,
            i32x4_lt, i32x4_max, i32x4_min, i32x4_ne, i32x4_replace_lane, i32x4_shl, i32x4_shr,
            i32x4_shuffle, i32x4_splat, i32x4_sub, i32x4_trunc_sat_f32x4,
            u32x4_eq, u32x4_extadd_pairwise_u16x8, u32x4_extend_high_u16x8,
            u32x4_extend_low_u16x8, u32x4_extract_lane, u32x4_ge, u32x4_lt, u32x4_min, u32x4_shr,
            u32x4_splat,
            // i64x2 / u64x2.
            i64x2_shuffle, u64x2_extract_lane,
            // Whole-register ops, loads and stores.
            v128_and, v128_andnot, v128_any_true, v128_bitselect, v128_load, v128_load32_zero,
            v128_load64_zero, v128_not, v128_or, v128_store, v128_store32_lane, v128_store64_lane,
        }
    };
}

macro_rules! reexport_layer {
    ($($seg:ident)::+ ; $($item:ident),* $(,)?) => {
        pub use $($seg)::+::{ $($item),* };
    };
}

#[cfg(any(test, feature = "simd-conformance"))]
macro_rules! layer_item_names {
    (; $($item:ident),* $(,)?) => {
        /// Every item the layer exposes (the type `v128` first), as written in `layer_items!`.
        pub const LAYER_ITEMS: &[&str] = &[$(stringify!($item)),*];
    };
}

#[cfg(any(test, feature = "simd-conformance"))]
layer_items!(layer_item_names!());

// ---------------------------------------------------------------------------------------------
// Backends (compiled side by side).
// ---------------------------------------------------------------------------------------------

#[cfg(any(not(target_arch = "wasm32"), feature = "simd-conformance"))]
pub mod scalar;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon;
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
pub mod sse;

#[cfg(any(test, feature = "simd-conformance"))]
pub mod conformance;

// ---------------------------------------------------------------------------------------------
// Backend selection (exactly one line applies on any target).
// ---------------------------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
layer_items!(reexport_layer!(core::arch::wasm32));

#[cfg(all(target_arch = "aarch64", target_feature = "neon", not(feature = "simd-force-scalar")))]
layer_items!(reexport_layer!(self::neon));

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3", not(feature = "simd-force-scalar")))]
layer_items!(reexport_layer!(self::sse));

// The scalar line excludes every condition above, so exactly one selection applies.
#[cfg(not(any(
    target_arch = "wasm32",
    all(target_arch = "aarch64", target_feature = "neon", not(feature = "simd-force-scalar")),
    all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3", not(feature = "simd-force-scalar")),
)))]
layer_items!(reexport_layer!(self::scalar));
