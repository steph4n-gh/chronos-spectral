//! SIMD layer conformance suite.
//!
//! One table, [`op_table!`], lists every op in the layer (every const-generic instantiation the
//! kernel uses included) with how to generate its inputs and how to compare its outputs. The
//! table is written once and instantiated for any backend module that has the
//! `core::arch::wasm32` names, via [`backend_table!`]:
//!
//! ```ignore
//! backend_table!(neon_table, crate::simd::neon);
//! let mut report = Report::new();
//! run_pair(neon_table::TABLE, scalar_table::TABLE, SEED, N_RANDOM, &mut report);
//! ```
//!
//! [`run_pair`] evaluates both backends on identical inputs, op by op: structured edge-value
//! lanes first (zeros, signed zeros, +-1, integer min/max and saturation boundaries, NaN, +-inf,
//! subnormals, rounding halfway cases, shift counts at and beyond the lane width, every pairing
//! of those for binary ops), then seeded pseudo-random lanes. Outputs are compared bitwise,
//! except that float lanes of arithmetic ops compare equal when both are NaN, because wasm
//! leaves NaN payload and sign unspecified there. Ops that wasm defines as bit operations or
//! selections (`abs`, `neg`, `pmin`, `pmax`, splats, lane moves, shuffles) compare bitwise even
//! for NaN.
//!
//! Instantiations live in this file:
//! - native `cargo test --target host-tuple`: the selected backend (`crate::simd`) against
//!   `simd::scalar`, plus the negative controls in [`negative_controls`];
//! - wasm32 with the `simd-conformance` feature: `core::arch::wasm32` (real SIMD128, executed
//!   by the host engine) against `simd::scalar`, exported as `chronos_simd_conformance_*` and
//!   driven by `scripts/gate_simd_conformance.mjs`.
//!
//! Everything here is `no_std`, allocation-free and deterministic for a given seed.

use core::hint::black_box;

/// One 128-bit value as bytes, lane order little-endian (as in wasm and in memory).
pub type Bytes = [u8; 16];

/// Inputs of one evaluation: up to three vector operands, plus `s`, which holds scalar operands
/// as lanes (splat or replace-lane value in lane 0, constructor arguments in lanes `0..n`, a
/// shift count in bytes `0..4`, a load/store offset in byte 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Input {
    pub v: [Bytes; 3],
    pub s: Bytes,
}

/// How two outputs of the same op are compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Check {
    /// All 16 bytes bitwise.
    Exact = 0,
    /// Per `f32` lane: identical bits, or both NaN.
    F32 = 1,
    /// Per `f64` lane: identical bits, or both NaN.
    F64 = 2,
}

/// Distribution the generated lanes are drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lanes {
    B8,
    B16,
    B32,
    B64,
    F32,
    F64,
}

/// Operand shape of an op, which decides how inputs are generated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `n` vector operands.
    Vec(u8),
    /// One vector operand and a shift count.
    Shift,
    /// Scalar operands only (splats and constructors).
    Scalars,
    /// One vector operand and a replacement lane value.
    Replace,
    /// Two vector operands holding distinct byte patterns (shuffles).
    Shuffle,
    /// 32 bytes of memory (`v[0] ++ v[1]`) read at offset `s[0] % 16`.
    Load,
    /// `v[0]` written into 32 bytes of memory (`v[1] ++ v[2]`) at offset `s[0] % 16`; the output
    /// is the 16 bytes starting at that offset.
    Store,
}

/// One row of the conformance table, bound to a backend.
#[derive(Clone, Copy)]
pub struct OpEntry {
    /// Layer item name, for example `"i32x4_shuffle"`.
    pub name: &'static str,
    /// Const-generic arguments, for example `"<1, 2, 3, 4>"`, or `""`.
    pub variant: &'static str,
    pub lanes: Lanes,
    pub shape: Shape,
    pub check: Check,
    pub f: fn(&Input) -> Bytes,
}

// ---------------------------------------------------------------------------------------------
// Operand plumbing used by the table (public so `backend_table!` expansions can reach it).
// ---------------------------------------------------------------------------------------------

/// Reinterprets 16 bytes as a backend's `v128` (opaque to the optimiser, so the op really runs).
#[inline(always)]
pub fn vec_in<V: Copy>(b: Bytes) -> V {
    const { assert!(core::mem::size_of::<V>() == 16) };
    // SAFETY: `V` is a 16-byte plain-data SIMD register type for which every bit pattern is
    // valid; `transmute_copy` reads unaligned when `V` is more aligned than `Bytes`.
    unsafe { core::mem::transmute_copy::<Bytes, V>(&black_box(b)) }
}

/// Reinterprets a backend's `v128` as 16 bytes.
#[inline(always)]
pub fn vec_out<V: Copy>(v: V) -> Bytes {
    const { assert!(core::mem::size_of::<V>() == 16) };
    // SAFETY: as in `vec_in`; `Bytes` has alignment 1.
    unsafe { core::mem::transmute_copy::<V, Bytes>(&black_box(v)) }
}

macro_rules! scalar_io {
    ($get:ident, $put:ident, $t:ty, $w:literal) => {
        /// Lane `i` of the scalar operands, read as this type.
        #[inline(always)]
        pub fn $get(x: &Input, i: usize) -> $t {
            let mut b = [0u8; $w];
            b.copy_from_slice(&x.s[i * $w..(i + 1) * $w]);
            black_box(<$t>::from_le_bytes(b))
        }

        /// A scalar result, stored in the low bytes of the output.
        #[inline(always)]
        pub fn $put(v: $t) -> Bytes {
            let mut out = [0u8; 16];
            out[..$w].copy_from_slice(&v.to_le_bytes());
            out
        }
    };
}

scalar_io!(s_i8, out_i8, i8, 1);
scalar_io!(s_u8, out_u8, u8, 1);
scalar_io!(s_i16, out_i16, i16, 2);
scalar_io!(s_u16, out_u16, u16, 2);
scalar_io!(s_i32, out_i32, i32, 4);
scalar_io!(s_u32, out_u32, u32, 4);
scalar_io!(s_f32, out_f32, f32, 4);
scalar_io!(s_u64, out_u64, u64, 8);
scalar_io!(s_f64, out_f64, f64, 8);

#[inline(always)]
pub fn out_bool(v: bool) -> Bytes {
    out_u8(v as u8)
}

/// The 32-byte memory image `a ++ b` used by load and store ops.
#[inline(always)]
pub fn memory(a: Bytes, b: Bytes) -> [u8; 32] {
    let mut m = [0u8; 32];
    m[..16].copy_from_slice(&a);
    m[16..].copy_from_slice(&b);
    m
}

/// Load/store offset, `0..16`, so every access stays inside the 32-byte image.
#[inline(always)]
pub fn offset(x: &Input) -> usize {
    black_box((x.s[0] & 15) as usize)
}

/// Pointer to byte `off` of `m`, opaque to the optimiser so the load really happens (and, on
/// wasm32, is selected as the dedicated `v128.load*` instruction rather than folded away).
#[inline(always)]
pub fn at(m: &[u8; 32], off: usize) -> *const u8 {
    black_box(m[off..].as_ptr())
}

/// Mutable counterpart of [`at`]: the store really reaches memory.
#[inline(always)]
pub fn at_mut(m: &mut [u8; 32], off: usize) -> *mut u8 {
    black_box(m[off..].as_mut_ptr())
}

/// The 16 bytes of `m` starting at `off`.
#[inline(always)]
pub fn window(m: &[u8; 32], off: usize) -> Bytes {
    let mut out = [0u8; 16];
    out.copy_from_slice(&m[off..off + 16]);
    out
}

// ---------------------------------------------------------------------------------------------
// The table.
// ---------------------------------------------------------------------------------------------

/// Expands to `&[OpEntry]` with every layer op bound to the backend module in scope as `be`.
///
/// Use [`backend_table!`] rather than invoking this directly. Every item of the layer must
/// appear here at least once (`every_layer_item_has_a_conformance_entry` checks it), and every
/// const-generic instantiation in the kernel must appear exactly as used
/// (`every_kernel_const_generic_instantiation_is_in_the_table` checks it).
macro_rules! op_table {
    () => {{
        macro_rules! entry {
            ($name:ident, $variant:expr, $lanes:ident, $shape:expr, $check:ident, $f:expr) => {
                OpEntry {
                    name: stringify!($name),
                    variant: $variant,
                    lanes: Lanes::$lanes,
                    shape: $shape,
                    check: Check::$check,
                    f: $f,
                }
            };
        }
        // Vector-only ops.
        macro_rules! v1 {
            ($name:ident, $lanes:ident, $check:ident) => {
                entry!($name, "", $lanes, Shape::Vec(1), $check, |x| vec_out(be::$name(vec_in(x.v[0]))))
            };
        }
        macro_rules! v2 {
            ($name:ident, $lanes:ident, $check:ident) => {
                entry!($name, "", $lanes, Shape::Vec(2), $check, |x| {
                    vec_out(be::$name(vec_in(x.v[0]), vec_in(x.v[1])))
                })
            };
        }
        macro_rules! v3 {
            ($name:ident, $lanes:ident, $check:ident) => {
                entry!($name, "", $lanes, Shape::Vec(3), $check, |x| {
                    vec_out(be::$name(vec_in(x.v[0]), vec_in(x.v[1]), vec_in(x.v[2])))
                })
            };
        }
        // Vector to scalar (reductions).
        macro_rules! reduce {
            ($name:ident, $lanes:ident, $out:ident) => {
                entry!($name, "", $lanes, Shape::Vec(1), Exact, |x| $out(be::$name(vec_in(x.v[0]))))
            };
        }
        macro_rules! shift {
            ($name:ident, $lanes:ident) => {
                entry!($name, "", $lanes, Shape::Shift, Exact, |x| {
                    vec_out(be::$name(vec_in(x.v[0]), s_u32(x, 0)))
                })
            };
        }
        macro_rules! splat {
            ($name:ident, $lanes:ident, $get:ident) => {
                entry!($name, "", $lanes, Shape::Scalars, Exact, |x| vec_out(be::$name($get(x, 0))))
            };
        }
        // (Fixed arities: a macro defined inside a macro cannot use `$(..)*` repetitions.)
        macro_rules! ctor2 {
            ($name:ident, $lanes:ident, $g:ident) => {
                entry!($name, "", $lanes, Shape::Scalars, Exact, |x| vec_out(be::$name($g(x, 0), $g(x, 1))))
            };
        }
        macro_rules! ctor4 {
            ($name:ident, $lanes:ident, $g:ident) => {
                entry!($name, "", $lanes, Shape::Scalars, Exact, |x| {
                    vec_out(be::$name($g(x, 0), $g(x, 1), $g(x, 2), $g(x, 3)))
                })
            };
        }
        macro_rules! ctor8 {
            ($name:ident, $lanes:ident, $g:ident) => {
                entry!($name, "", $lanes, Shape::Scalars, Exact, |x| {
                    vec_out(be::$name($g(x, 0), $g(x, 1), $g(x, 2), $g(x, 3), $g(x, 4), $g(x, 5), $g(x, 6), $g(x, 7)))
                })
            };
        }
        macro_rules! extract {
            ($name:ident, $n:literal, $lanes:ident, $out:ident) => {
                entry!($name, concat!("<", $n, ">"), $lanes, Shape::Vec(1), Exact, |x| {
                    $out(be::$name::<$n>(vec_in(x.v[0])))
                })
            };
        }
        macro_rules! replace {
            ($name:ident, $n:literal, $lanes:ident, $get:ident) => {
                entry!($name, concat!("<", $n, ">"), $lanes, Shape::Replace, Exact, |x| {
                    vec_out(be::$name::<$n>(vec_in(x.v[0]), $get(x, 0)))
                })
            };
        }
        macro_rules! sh2 {
            ($name:ident, $a:literal, $b:literal) => {
                entry!($name, concat!("<", stringify!($a, $b), ">"), B8, Shape::Shuffle, Exact, |x| {
                    vec_out(be::$name::<$a, $b>(vec_in(x.v[0]), vec_in(x.v[1])))
                })
            };
        }
        macro_rules! sh4 {
            ($name:ident, $a:literal, $b:literal, $c:literal, $d:literal) => {
                entry!($name, concat!("<", stringify!($a, $b, $c, $d), ">"), B8, Shape::Shuffle, Exact, |x| {
                    vec_out(be::$name::<$a, $b, $c, $d>(vec_in(x.v[0]), vec_in(x.v[1])))
                })
            };
        }
        macro_rules! sh8 {
            ($name:ident, $a:literal, $b:literal, $c:literal, $d:literal,
             $e:literal, $f:literal, $g:literal, $h:literal) => {
                entry!($name, concat!("<", stringify!($a, $b, $c, $d, $e, $f, $g, $h), ">"), B8, Shape::Shuffle, Exact, |x| {
                    vec_out(be::$name::<$a, $b, $c, $d, $e, $f, $g, $h>(vec_in(x.v[0]), vec_in(x.v[1])))
                })
            };
        }
        macro_rules! sh16 {
            ($name:ident, $a:literal, $b:literal, $c:literal, $d:literal,
             $e:literal, $f:literal, $g:literal, $h:literal,
             $i:literal, $j:literal, $k:literal, $l:literal,
             $m:literal, $n:literal, $o:literal, $p:literal) => {
                entry!(
                    $name,
                    concat!("<", stringify!($a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p), ">"),
                    B8,
                    Shape::Shuffle,
                    Exact,
                    |x| {
                        vec_out(be::$name::<$a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p>(
                            vec_in(x.v[0]),
                            vec_in(x.v[1]),
                        ))
                    }
                )
            };
        }
        macro_rules! store_lane {
            ($name:ident, $n:literal, $t:ty) => {
                entry!($name, concat!("<", $n, ">"), B32, Shape::Store, Exact, |x| {
                    let mut m = memory(x.v[1], x.v[2]);
                    let off = offset(x);
                    // SAFETY: `off + 16 <= 32`, inside `m`; the op allows unaligned pointers.
                    unsafe { be::$name::<$n>(vec_in(x.v[0]), at_mut(&mut m, off) as *mut $t) };
                    window(&m, off)
                })
            };
        }

        &[
            // ---- f32x4 arithmetic (NaN payload unspecified) ----
            v2!(f32x4_add, F32, F32),
            v2!(f32x4_sub, F32, F32),
            v2!(f32x4_mul, F32, F32),
            v2!(f32x4_div, F32, F32),
            v2!(f32x4_min, F32, F32),
            v2!(f32x4_max, F32, F32),
            v1!(f32x4_sqrt, F32, F32),
            v1!(f32x4_floor, F32, F32),
            v1!(f32x4_nearest, F32, F32),
            // ---- f32x4 bit operations, selections, comparisons, conversions (exact) ----
            v1!(f32x4_abs, F32, Exact),
            v1!(f32x4_neg, F32, Exact),
            v2!(f32x4_pmin, F32, Exact),
            v2!(f32x4_pmax, F32, Exact),
            v2!(f32x4_lt, F32, Exact),
            v2!(f32x4_le, F32, Exact),
            v2!(f32x4_gt, F32, Exact),
            v2!(f32x4_ge, F32, Exact),
            v1!(f32x4_convert_i32x4, B32, Exact),
            v1!(f32x4_convert_u32x4, B32, Exact),
            v1!(i32x4_trunc_sat_f32x4, F32, Exact),
            // ---- f64x2 ----
            v2!(f64x2_add, F64, F64),
            v2!(f64x2_sub, F64, F64),
            v2!(f64x2_mul, F64, F64),
            v1!(f64x2_sqrt, F64, F64),
            // ---- i8x16 / u8x16 ----
            v2!(i8x16_add, B8, Exact),
            v2!(i8x16_sub, B8, Exact),
            v2!(i8x16_min, B8, Exact),
            v2!(i8x16_max, B8, Exact),
            v2!(i8x16_narrow_i16x8, B16, Exact),
            v2!(u8x16_add_sat, B8, Exact),
            v2!(u8x16_sub_sat, B8, Exact),
            v2!(u8x16_min, B8, Exact),
            v2!(u8x16_max, B8, Exact),
            v2!(u8x16_lt, B8, Exact),
            v2!(u8x16_le, B8, Exact),
            v2!(u8x16_gt, B8, Exact),
            v2!(u8x16_ge, B8, Exact),
            v2!(u8x16_narrow_i16x8, B16, Exact),
            reduce!(i8x16_bitmask, B8, out_u16),
            // ---- i16x8 / u16x8 ----
            v2!(i16x8_add, B16, Exact),
            v2!(u16x8_add, B16, Exact),
            v2!(i16x8_sub, B16, Exact),
            v2!(i16x8_sub_sat, B16, Exact),
            v2!(u16x8_mul, B16, Exact),
            v2!(i16x8_min, B16, Exact),
            v2!(i16x8_max, B16, Exact),
            v2!(i16x8_lt, B16, Exact),
            v2!(i16x8_q15mulr_sat, B16, Exact),
            v2!(i16x8_narrow_i32x4, B32, Exact),
            v1!(i16x8_extend_low_i8x16, B8, Exact),
            v1!(i16x8_extend_high_i8x16, B8, Exact),
            v1!(u16x8_extend_low_u8x16, B8, Exact),
            v1!(u16x8_extend_high_u8x16, B8, Exact),
            v2!(i16x8_extmul_low_i8x16, B8, Exact),
            v2!(i16x8_extmul_high_i8x16, B8, Exact),
            v2!(u16x8_extmul_low_u8x16, B8, Exact),
            v2!(u16x8_extmul_high_u8x16, B8, Exact),
            v1!(i16x8_extadd_pairwise_i8x16, B8, Exact),
            v1!(u16x8_extadd_pairwise_u8x16, B8, Exact),
            shift!(i16x8_shl, B16),
            shift!(i16x8_shr, B16),
            shift!(u16x8_shr, B16),
            // ---- i32x4 / u32x4 ----
            v2!(i32x4_add, B32, Exact),
            v2!(i32x4_sub, B32, Exact),
            v2!(i32x4_min, B32, Exact),
            v2!(i32x4_max, B32, Exact),
            v2!(u32x4_min, B32, Exact),
            v2!(i32x4_eq, B32, Exact),
            v2!(u32x4_eq, B32, Exact),
            v2!(i32x4_ne, B32, Exact),
            v2!(i32x4_lt, B32, Exact),
            v2!(i32x4_ge, B32, Exact),
            v2!(u32x4_lt, B32, Exact),
            v2!(u32x4_ge, B32, Exact),
            v2!(i32x4_dot_i16x8, B16, Exact),
            v1!(i32x4_extend_low_i16x8, B16, Exact),
            v1!(i32x4_extend_high_i16x8, B16, Exact),
            v1!(u32x4_extend_low_u16x8, B16, Exact),
            v1!(u32x4_extend_high_u16x8, B16, Exact),
            v2!(i32x4_extmul_low_i16x8, B16, Exact),
            v2!(i32x4_extmul_high_i16x8, B16, Exact),
            v1!(i32x4_extadd_pairwise_i16x8, B16, Exact),
            v1!(u32x4_extadd_pairwise_u16x8, B16, Exact),
            shift!(i32x4_shl, B32),
            shift!(i32x4_shr, B32),
            shift!(u32x4_shr, B32),
            reduce!(i32x4_all_true, B32, out_bool),
            reduce!(i32x4_bitmask, B32, out_u8),
            // ---- whole register ----
            v2!(v128_and, B32, Exact),
            v2!(v128_andnot, B32, Exact),
            v2!(v128_or, B32, Exact),
            v1!(v128_not, B32, Exact),
            v3!(v128_bitselect, B32, Exact),
            reduce!(v128_any_true, B8, out_bool),
            // ---- splats and constructors ----
            splat!(f32x4_splat, F32, s_f32),
            splat!(f64x2_splat, F64, s_f64),
            splat!(i8x16_splat, B8, s_i8),
            splat!(u8x16_splat, B8, s_u8),
            splat!(i16x8_splat, B16, s_i16),
            splat!(u16x8_splat, B16, s_u16),
            splat!(i32x4_splat, B32, s_i32),
            splat!(u32x4_splat, B32, s_u32),
            ctor4!(f32x4, F32, s_f32),
            ctor2!(f64x2, F64, s_f64),
            ctor8!(i16x8, B16, s_i16),
            ctor4!(i32x4, B32, s_i32),
            ctor4!(u32x4, B32, s_u32),
            ctor2!(u64x2, B64, s_u64),
            // ---- lane extraction and replacement (every lane) ----
            extract!(f32x4_extract_lane, 0, F32, out_f32),
            extract!(f32x4_extract_lane, 1, F32, out_f32),
            extract!(f32x4_extract_lane, 2, F32, out_f32),
            extract!(f32x4_extract_lane, 3, F32, out_f32),
            extract!(f64x2_extract_lane, 0, F64, out_f64),
            extract!(f64x2_extract_lane, 1, F64, out_f64),
            extract!(i32x4_extract_lane, 0, B32, out_i32),
            extract!(i32x4_extract_lane, 1, B32, out_i32),
            extract!(i32x4_extract_lane, 2, B32, out_i32),
            extract!(i32x4_extract_lane, 3, B32, out_i32),
            extract!(u32x4_extract_lane, 0, B32, out_u32),
            extract!(u32x4_extract_lane, 1, B32, out_u32),
            extract!(u32x4_extract_lane, 2, B32, out_u32),
            extract!(u32x4_extract_lane, 3, B32, out_u32),
            extract!(u64x2_extract_lane, 0, B64, out_u64),
            extract!(u64x2_extract_lane, 1, B64, out_u64),
            replace!(f32x4_replace_lane, 0, F32, s_f32),
            replace!(f32x4_replace_lane, 1, F32, s_f32),
            replace!(f32x4_replace_lane, 2, F32, s_f32),
            replace!(f32x4_replace_lane, 3, F32, s_f32),
            replace!(i32x4_replace_lane, 0, B32, s_i32),
            replace!(i32x4_replace_lane, 1, B32, s_i32),
            replace!(i32x4_replace_lane, 2, B32, s_i32),
            replace!(i32x4_replace_lane, 3, B32, s_i32),
            // ---- shuffles: every pattern the kernel uses, plus identity / reverse / all-b ----
            sh16!(i8x16_shuffle, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            sh16!(i8x16_shuffle, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16),
            sh16!(i8x16_shuffle, 15, 16, 0, 31, 7, 24, 8, 23, 1, 30, 14, 17, 2, 29, 13, 18),
            sh16!(i8x16_shuffle, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
            sh16!(i8x16_shuffle, 0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            sh16!(i8x16_shuffle, 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29),
            sh16!(i8x16_shuffle, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
            sh16!(i8x16_shuffle, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17),
            sh16!(i8x16_shuffle, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18),
            sh16!(i8x16_shuffle, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31),
            sh16!(i8x16_shuffle, 2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18, 22, 26, 30),
            sh16!(i8x16_shuffle, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
            sh8!(i16x8_shuffle, 0, 1, 2, 3, 4, 5, 6, 7),
            sh8!(i16x8_shuffle, 15, 14, 13, 12, 11, 10, 9, 8),
            sh8!(i16x8_shuffle, 0, 0, 1, 1, 2, 2, 3, 3),
            sh8!(i16x8_shuffle, 0, 4, 8, 12, 1, 5, 9, 13),
            sh8!(i16x8_shuffle, 2, 6, 10, 14, 3, 7, 11, 15),
            sh8!(i16x8_shuffle, 4, 4, 5, 5, 6, 6, 7, 7),
            sh4!(i32x4_shuffle, 0, 1, 2, 3),
            sh4!(i32x4_shuffle, 7, 6, 5, 4),
            sh4!(i32x4_shuffle, 0, 0, 0, 2),
            sh4!(i32x4_shuffle, 0, 0, 1, 1),
            sh4!(i32x4_shuffle, 0, 0, 1, 2),
            sh4!(i32x4_shuffle, 0, 1, 4, 5),
            sh4!(i32x4_shuffle, 0, 1, 6, 3),
            sh4!(i32x4_shuffle, 0, 1, 6, 7),
            sh4!(i32x4_shuffle, 0, 2, 0, 1),
            sh4!(i32x4_shuffle, 0, 2, 1, 3),
            sh4!(i32x4_shuffle, 0, 2, 4, 6),
            sh4!(i32x4_shuffle, 0, 4, 0, 4),
            sh4!(i32x4_shuffle, 0, 4, 1, 5),
            sh4!(i32x4_shuffle, 0, 5, 0, 5),
            sh4!(i32x4_shuffle, 0, 5, 2, 7),
            sh4!(i32x4_shuffle, 0, 5, 6, 7),
            sh4!(i32x4_shuffle, 1, 0, 3, 2),
            sh4!(i32x4_shuffle, 1, 1, 1, 1),
            sh4!(i32x4_shuffle, 1, 1, 1, 3),
            sh4!(i32x4_shuffle, 1, 2, 3, 3),
            sh4!(i32x4_shuffle, 1, 2, 3, 4),
            sh4!(i32x4_shuffle, 1, 2, 3, 5),
            sh4!(i32x4_shuffle, 1, 2, 3, 6),
            sh4!(i32x4_shuffle, 1, 2, 5, 6),
            sh4!(i32x4_shuffle, 1, 3, 2, 3),
            sh4!(i32x4_shuffle, 1, 3, 5, 7),
            sh4!(i32x4_shuffle, 2, 2, 2, 2),
            sh4!(i32x4_shuffle, 2, 2, 3, 3),
            sh4!(i32x4_shuffle, 2, 3, 0, 1),
            sh4!(i32x4_shuffle, 2, 3, 2, 3),
            sh4!(i32x4_shuffle, 2, 3, 3, 3),
            sh4!(i32x4_shuffle, 2, 3, 4, 5),
            sh4!(i32x4_shuffle, 2, 3, 6, 7),
            sh4!(i32x4_shuffle, 2, 6, 2, 6),
            sh4!(i32x4_shuffle, 2, 6, 3, 7),
            sh4!(i32x4_shuffle, 2, 7, 2, 7),
            sh4!(i32x4_shuffle, 3, 0, 1, 2),
            sh4!(i32x4_shuffle, 3, 2, 1, 0),
            sh4!(i32x4_shuffle, 3, 3, 3, 3),
            sh4!(i32x4_shuffle, 3, 4, 5, 6),
            sh4!(i32x4_shuffle, 3, 5, 6, 7),
            sh2!(i64x2_shuffle, 0, 1),
            sh2!(i64x2_shuffle, 3, 2),
            sh2!(i64x2_shuffle, 0, 2),
            sh2!(i64x2_shuffle, 1, 0),
            sh2!(i64x2_shuffle, 1, 3),
            // ---- loads and stores (unaligned offsets 0..16) ----
            entry!(v128_load, "", B8, Shape::Load, Exact, |x| {
                let m = memory(x.v[0], x.v[1]);
                // SAFETY: `offset(x) + 16 <= 32`, inside `m`; the op allows unaligned pointers.
                vec_out(unsafe { be::v128_load(at(&m, offset(x)) as *const _) })
            }),
            entry!(v128_load32_zero, "", B8, Shape::Load, Exact, |x| {
                let m = memory(x.v[0], x.v[1]);
                // SAFETY: as above, reading 4 bytes.
                vec_out(unsafe { be::v128_load32_zero(at(&m, offset(x)) as *const u32) })
            }),
            entry!(v128_load64_zero, "", B8, Shape::Load, Exact, |x| {
                let m = memory(x.v[0], x.v[1]);
                // SAFETY: as above, reading 8 bytes.
                vec_out(unsafe { be::v128_load64_zero(at(&m, offset(x)) as *const u64) })
            }),
            entry!(v128_store, "", B32, Shape::Store, Exact, |x| {
                let mut m = memory(x.v[1], x.v[2]);
                let off = offset(x);
                // SAFETY: `off + 16 <= 32`, inside `m`; the op allows unaligned pointers.
                unsafe { be::v128_store(at_mut(&mut m, off) as *mut _, vec_in(x.v[0])) };
                window(&m, off)
            }),
            store_lane!(v128_store32_lane, 0, u32),
            store_lane!(v128_store32_lane, 1, u32),
            store_lane!(v128_store32_lane, 2, u32),
            store_lane!(v128_store32_lane, 3, u32),
            store_lane!(v128_store64_lane, 0, u64),
            store_lane!(v128_store64_lane, 1, u64),
        ]
    }};
}

/// `backend_table!(name, path::to::backend)` defines `mod name { pub static TABLE: &[OpEntry] }`
/// with every conformance entry bound to that backend module.
macro_rules! backend_table {
    ($modname:ident, $($path:ident)::+) => {
        #[allow(clippy::all)]
        pub mod $modname {
            #[allow(unused_imports)]
            use $($path)::+ as be;
            #[allow(unused_imports)]
            use $crate::simd::conformance::*;
            pub static TABLE: &[OpEntry] = $crate::simd::conformance::op_table!();
        }
    };
}

#[allow(unused_imports)]
pub(crate) use {backend_table, op_table};

// ---------------------------------------------------------------------------------------------
// Input generation.
// ---------------------------------------------------------------------------------------------

/// SplitMix64: tiny, deterministic, platform independent.
#[derive(Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const POOL_B8: &[u64] = &[
    0x00, 0x01, 0x02, 0x0F, 0x10, 0x40, 0x55, 0x7E, 0x7F, 0x80, 0x81, 0xAA, 0xC0, 0xF0, 0xFE, 0xFF,
];

const POOL_B16: &[u64] = &[
    0x0000, 0x0001, 0x0002, 0x007F, 0x0080, 0x00FE, 0x00FF, 0x0100, 0x3FFF, 0x4000, 0x4001, 0x5555,
    0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xAAAA, 0xBFFF, 0xC000, 0xFF00, 0xFF7F, 0xFF80, 0xFFFE, 0xFFFF,
];

const POOL_B32: &[u64] = &[
    0x0000_0000,
    0x0000_0001,
    0x0000_0002,
    0x0000_007F,
    0x0000_0080,
    0x0000_00FF,
    0x0000_0100,
    0x0000_7FFF,
    0x0000_8000,
    0x0000_FFFF,
    0x0001_0000,
    0x00FF_FFFF,
    0x0100_0000,
    0x0100_0001,
    0x0100_0003,
    0x5555_5555,
    0x7FFF_FF7F,
    0x7FFF_FFBF,
    0x7FFF_FFC0,
    0x7FFF_FFFE,
    0x7FFF_FFFF,
    0x8000_0000,
    0x8000_0001,
    0x8000_7FFF,
    0xAAAA_AAAA,
    0xFFFF_7FFF,
    0xFFFF_8000,
    0xFFFF_FF00,
    0xFFFF_FF7F,
    0xFFFF_FF80,
    0xFFFF_FFFE,
    0xFFFF_FFFF,
];

const POOL_B64: &[u64] = &[
    0,
    1,
    0x0000_0000_FFFF_FFFF,
    0x0123_4567_89AB_CDEF,
    0x7FFF_FFFF_FFFF_FFFF,
    0x8000_0000_0000_0000,
    0xFFFF_FFFF_0000_0000,
    u64::MAX,
];

/// `f32` bit patterns: signed zeros, +-1, rounding halfway cases, 2^23 and 2^31 / 2^32
/// boundaries for conversions, extremes, subnormals, infinities and quiet / signalling NaNs.
const POOL_F32: &[u64] = &[
    0x0000_0000,
    0x8000_0000, // +0, -0
    0x3F80_0000,
    0xBF80_0000, // +1, -1
    0x3F00_0000,
    0xBF00_0000, // +0.5, -0.5
    0x3FC0_0000,
    0xBFC0_0000, // +1.5, -1.5
    0x4020_0000,
    0xC020_0000, // +2.5, -2.5
    0x3EFF_FFFF,
    0x3F00_0001, // 0.5 - ulp, 0.5 + ulp
    0x3F7F_FFFF,
    0x4000_0000,
    0x4040_0000,
    0x4080_0000, // 1 - ulp, 2, 3, 4
    0x4A80_0001,
    0xCA80_0003, // 4194304.5, -4194305.5
    0x4AFF_FFFF,
    0xCAFF_FFFF, // +-8388607.5
    0x4B00_0000,
    0x4B00_0001, // 2^23, 2^23 + 1
    0x4EFF_FFFF,
    0x4F00_0000,
    0xCF00_0000,
    0xCF00_0001, // i32 range edges
    0x4F7F_FFFF,
    0x4F80_0000, // u32 range edges
    0x7F7F_FFFF,
    0xFF7F_FFFF, // +-MAX
    0x0080_0000,
    0x8080_0000, // +-MIN_POSITIVE
    0x0000_0001,
    0x8000_0001,
    0x0040_0000,
    0x007F_FFFF,
    0x807F_FFFF, // subnormals
    0x7F80_0000,
    0xFF80_0000, // +-inf
    0x7FC0_0000,
    0xFFC0_0000,
    0x7FC1_2345, // quiet NaNs
    0x7FA0_0001,
    0xFF80_0001, // signalling NaNs
    0x3DCC_CCCD,
    0x4049_0FDB,
    0x5F80_0000,
    0x2F80_0000, // 0.1, pi, 2^64, 2^-32
];

/// `f64` bit patterns, same idea as `POOL_F32`.
const POOL_F64: &[u64] = &[
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000, // +0, -0
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000, // +1, -1
    0x3FE0_0000_0000_0000,
    0xBFE0_0000_0000_0000, // +0.5, -0.5
    0x3FF8_0000_0000_0000,
    0x4000_0000_0000_0000,
    0x4008_0000_0000_0000,
    0x4010_0000_0000_0000,
    0x3FF0_0000_0000_0001,
    0x3FEF_FFFF_FFFF_FFFF, // 1 + ulp, 1 - ulp
    0x4340_0000_0000_0000,
    0x4330_0000_0000_0001, // 2^53, 2^52 + 1
    0x7FEF_FFFF_FFFF_FFFF,
    0xFFEF_FFFF_FFFF_FFFF, // +-MAX
    0x0010_0000_0000_0000,
    0x8010_0000_0000_0000, // +-MIN_POSITIVE
    0x0000_0000_0000_0001,
    0x8000_0000_0000_0001,
    0x000F_FFFF_FFFF_FFFF, // subnormals
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000, // +-inf
    0x7FF8_0000_0000_0000,
    0xFFF8_0000_0000_0000,
    0x7FF8_0000_0001_2345, // quiet NaNs
    0x7FF0_0000_0000_0001, // signalling NaN
    0x3FB9_9999_9999_999A,
    0x7E37_E43C_8800_759C,
    0x01A5_6E1F_C2F8_F359, // 0.1, 1e300, 1e-300
];

/// Shift counts: in range, exactly the lane widths, beyond them, and huge values (wasm takes
/// the count modulo the lane width).
const SHIFT_COUNTS: &[u32] = &[
    0,
    1,
    2,
    7,
    8,
    14,
    15,
    16,
    17,
    30,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    255,
    256,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
];

fn lane_width(l: Lanes) -> usize {
    match l {
        Lanes::B8 => 1,
        Lanes::B16 => 2,
        Lanes::B32 | Lanes::F32 => 4,
        Lanes::B64 | Lanes::F64 => 8,
    }
}

fn pool(l: Lanes) -> &'static [u64] {
    match l {
        Lanes::B8 => POOL_B8,
        Lanes::B16 => POOL_B16,
        Lanes::B32 => POOL_B32,
        Lanes::B64 => POOL_B64,
        Lanes::F32 => POOL_F32,
        Lanes::F64 => POOL_F64,
    }
}

fn put_lane(b: &mut Bytes, l: Lanes, i: usize, v: u64) {
    let w = lane_width(l);
    b[i * w..(i + 1) * w].copy_from_slice(&v.to_le_bytes()[..w]);
}

fn get_lane(b: &Bytes, l: Lanes, i: usize) -> u64 {
    let w = lane_width(l);
    let mut v = [0u8; 8];
    v[..w].copy_from_slice(&b[i * w..(i + 1) * w]);
    u64::from_le_bytes(v)
}

fn random_lane(l: Lanes, r: &mut Rng) -> u64 {
    let x = r.next();
    let p = pool(l);
    let pick = p[(x >> 40) as usize % p.len()];
    let sel = x >> 60; // 0..16
    match l {
        Lanes::F32 => {
            let lo = x & 0xFFFF_FFFF;
            let sign = (x >> 59) & 1;
            match sel {
                0..=4 => lo, // raw bits: every exponent, NaN payloads, subnormals
                5..=9 => {
                    // moderate magnitudes 2^-24..2^24, where rounding of results is exercised
                    let e = 103 + (x >> 32) % 49;
                    (lo & 0x807F_FFFF) | (e << 23)
                }
                10 | 11 => {
                    // exact halfway cases n + 0.5 with |n| < 2^22
                    let n = ((x >> 32) & 0x3F_FFFF) as f32 + 0.5;
                    (n.to_bits() as u64) | (sign << 31)
                }
                12 => (((x >> 32) % 201) as f32 - 100.0).to_bits() as u64,
                _ => pick,
            }
        }
        Lanes::F64 => match sel {
            0..=5 => x,
            6..=11 => {
                let e = 963 + (x >> 20) % 121; // 2^-60..2^60
                (x & 0x800F_FFFF_FFFF_FFFF) | (e << 52)
            }
            _ => pick,
        },
        _ => {
            if sel < 12 {
                x
            } else {
                pick
            }
        }
    }
}

/// Number of structured edge vectors generated for `e` before the random ones.
pub fn edge_count(e: &OpEntry) -> u32 {
    let p = pool(e.lanes).len();
    let n = 16 / lane_width(e.lanes);
    let c = match e.shape {
        Shape::Vec(1) | Shape::Vec(3) => 2 + p,
        Shape::Vec(_) => 2 + (p * p).div_ceil(n),
        Shape::Shift => SHIFT_COUNTS.len() * p.div_ceil(n),
        Shape::Scalars => p,
        Shape::Replace => 2 * p,
        Shape::Shuffle => 3,
        Shape::Load | Shape::Store => 32,
    };
    c as u32
}

fn name_hash(e: &OpEntry) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64; // FNV-1a
    for b in e.name.bytes().chain(e.variant.bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The `k`-th input vector for `e`. Edge vectors come first; the rest are pseudo-random,
/// seeded by `seed`, the entry's name and `k`, so they do not depend on table order.
pub fn gen_input(e: &OpEntry, k: u32, seed: u64) -> Input {
    let mut x = Input {
        v: [[0; 16]; 3],
        s: [0; 16],
    };
    let l = e.lanes;
    let p = pool(l);
    let np = p.len();
    let n = 16 / lane_width(l);
    let edges = edge_count(e);
    let k_us = k as usize;

    if k < edges {
        match e.shape {
            Shape::Vec(nv) => {
                if k == 0 {
                    // all zeros
                } else if k == 1 {
                    x.v = [[0xFF; 16]; 3];
                } else if nv == 2 {
                    // Every (a, b) pair of pool values appears in some lane.
                    let base = (k_us - 2) * n;
                    for i in 0..n {
                        let slot = base + i;
                        put_lane(&mut x.v[0], l, i, p[slot % np]);
                        put_lane(&mut x.v[1], l, i, p[(slot / np) % np]);
                    }
                } else {
                    // Every pool value appears in every lane position.
                    let r = k_us - 2;
                    for i in 0..n {
                        for (j, v) in x.v.iter_mut().enumerate() {
                            put_lane(v, l, i, p[(r + i + j * 7) % np]);
                        }
                    }
                }
            }
            Shape::Shift => {
                let q = k_us / SHIFT_COUNTS.len();
                for i in 0..n {
                    put_lane(&mut x.v[0], l, i, p[(q * n + i) % np]);
                }
                x.s[..4].copy_from_slice(&SHIFT_COUNTS[k_us % SHIFT_COUNTS.len()].to_le_bytes());
            }
            Shape::Scalars | Shape::Replace => {
                for i in 0..n {
                    put_lane(&mut x.v[0], l, i, p[(k_us + i) % np]);
                    put_lane(&mut x.s, l, i, p[(k_us + i * 5 + k_us / np) % np]);
                }
            }
            Shape::Shuffle => {
                for i in 0..16 {
                    let (a, b) = match k {
                        0 => (i as u8, 16 + i as u8),
                        1 => (0x80 | i as u8, 0xC0 | i as u8),
                        _ => (0xFF - i as u8, 0x7F - i as u8),
                    };
                    x.v[0][i] = a;
                    x.v[1][i] = b;
                }
            }
            Shape::Load | Shape::Store => {
                for i in 0..16 {
                    x.v[0][i] = (i as u8).wrapping_mul(17).wrapping_add(k as u8 >> 4);
                    x.v[1][i] = 0xA0 ^ i as u8;
                    x.v[2][i] = 0x50 ^ (i as u8).wrapping_mul(3);
                }
                if k >= 16 {
                    x.v[0] = [0xFF; 16];
                }
                x.s[0] = (k % 16) as u8;
            }
        }
        return x;
    }

    let mut r = Rng::new(seed ^ name_hash(e) ^ (k as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    for v in x.v.iter_mut() {
        for i in 0..n {
            put_lane(v, l, i, random_lane(l, &mut r));
        }
    }
    for i in 0..n {
        put_lane(&mut x.s, l, i, random_lane(l, &mut r));
    }
    match e.shape {
        Shape::Vec(2) | Shape::Vec(3) => {
            // Equal and sign-flipped operands exercise the min/max, compare and zero-sign paths.
            for i in 0..n {
                let roll = r.next() >> 61;
                let a = get_lane(&x.v[0], l, i);
                if roll == 0 {
                    put_lane(&mut x.v[1], l, i, a);
                } else if roll == 1 && matches!(l, Lanes::F32 | Lanes::F64) {
                    let sign = 1u64 << (lane_width(l) * 8 - 1);
                    put_lane(&mut x.v[1], l, i, a ^ sign);
                }
            }
        }
        Shape::Shift => {
            let c = r.next();
            let count = if c & 1 == 0 {
                (c >> 1) as u32 % 72
            } else {
                (c >> 32) as u32
            };
            x.s[..4].copy_from_slice(&count.to_le_bytes());
        }
        _ => {}
    }
    x
}

// ---------------------------------------------------------------------------------------------
// Comparison and reporting.
// ---------------------------------------------------------------------------------------------

/// Whether two outputs agree under `check`.
pub fn outputs_match(check: Check, a: &Bytes, b: &Bytes) -> bool {
    match check {
        Check::Exact => a == b,
        Check::F32 => (0..4).all(|i| {
            let (x, y) = (
                get_lane(a, Lanes::F32, i) as u32,
                get_lane(b, Lanes::F32, i) as u32,
            );
            x == y || (f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan())
        }),
        Check::F64 => (0..2).all(|i| {
            let (x, y) = (get_lane(a, Lanes::F64, i), get_lane(b, Lanes::F64, i));
            x == y || (f64::from_bits(x).is_nan() && f64::from_bits(y).is_nan())
        }),
    }
}

/// Maximum table size a [`Report`] can describe.
pub const MAX_ENTRIES: usize = 384;
/// Mismatch records kept in a [`Report`] (the count is always exact).
pub const MAX_RECORDS: usize = 32;
/// Records kept per entry, so one broken op cannot crowd out the others.
pub const MAX_RECORDS_PER_ENTRY: u32 = 4;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct EntryStat {
    pub vectors: u32,
    pub mismatches: u32,
}

/// One disagreement: entry index, vector index, the inputs and both outputs (104 bytes).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MismatchRecord {
    pub entry: u32,
    pub vector: u32,
    pub input: Input,
    pub out_a: Bytes,
    pub out_b: Bytes,
}

/// Result of [`run_pair`]. `#[repr(C)]` with only `u32` / byte fields so the wasm gate script can
/// read it straight from linear memory: `entries`, `recorded`, `mismatches`, `vectors` at byte
/// offsets 0, 4, 8, 12; `stats` at 16 (8 bytes each); `records` after that (104 bytes each).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Report {
    pub entries: u32,
    pub recorded: u32,
    pub mismatches: u32,
    pub vectors: u32,
    pub stats: [EntryStat; MAX_ENTRIES],
    pub records: [MismatchRecord; MAX_RECORDS],
}

impl Report {
    pub const fn new() -> Self {
        const EMPTY: MismatchRecord = MismatchRecord {
            entry: 0,
            vector: 0,
            input: Input {
                v: [[0; 16]; 3],
                s: [0; 16],
            },
            out_a: [0; 16],
            out_b: [0; 16],
        };
        Report {
            entries: 0,
            recorded: 0,
            mismatches: 0,
            vectors: 0,
            stats: [EntryStat {
                vectors: 0,
                mismatches: 0,
            }; MAX_ENTRIES],
            records: [EMPTY; MAX_RECORDS],
        }
    }
}

impl Default for Report {
    fn default() -> Self {
        Self::new()
    }
}

/// Why two tables cannot be compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairError {
    LengthDiffers,
    TooManyEntries,
    EntryDiffers(u32),
}

/// Runs every entry of `a` and `b` (two instantiations of [`op_table!`]) on the same
/// `edge_count + n_random` inputs per entry and records every disagreement in `report`.
pub fn run_pair(
    a: &[OpEntry],
    b: &[OpEntry],
    seed: u64,
    n_random: u32,
    report: &mut Report,
) -> Result<(), PairError> {
    *report = Report::new();
    if a.len() != b.len() {
        return Err(PairError::LengthDiffers);
    }
    if a.len() > MAX_ENTRIES {
        return Err(PairError::TooManyEntries);
    }
    for (i, (ea, eb)) in a.iter().zip(b).enumerate() {
        if ea.name != eb.name || ea.variant != eb.variant || ea.check != eb.check {
            return Err(PairError::EntryDiffers(i as u32));
        }
    }
    report.entries = a.len() as u32;
    for (i, (ea, eb)) in a.iter().zip(b).enumerate() {
        let n = edge_count(ea) + n_random;
        let mut bad = 0u32;
        for k in 0..n {
            let x = gen_input(ea, k, seed);
            let oa = (ea.f)(&x);
            let ob = (eb.f)(&x);
            if !outputs_match(ea.check, &oa, &ob) {
                if bad < MAX_RECORDS_PER_ENTRY && (report.recorded as usize) < MAX_RECORDS {
                    report.records[report.recorded as usize] = MismatchRecord {
                        entry: i as u32,
                        vector: k,
                        input: x,
                        out_a: oa,
                        out_b: ob,
                    };
                    report.recorded += 1;
                }
                bad += 1;
            }
        }
        report.stats[i] = EntryStat {
            vectors: n,
            mismatches: bad,
        };
        report.vectors = report.vectors.saturating_add(n);
        report.mismatches = report.mismatches.saturating_add(bad);
    }
    Ok(())
}

/// Number of layer items (excluding the type `v128`) with no entry in `table`.
pub fn uncovered_layer_items(table: &[OpEntry]) -> usize {
    crate::simd::LAYER_ITEMS
        .iter()
        .filter(|&&n| n != "v128" && !table.iter().any(|e| e.name == n))
        .count()
}

/// Number of distinct op names in `table`.
pub fn distinct_ops(table: &[OpEntry]) -> usize {
    (0..table.len())
        .filter(|&i| !table[..i].iter().any(|e| e.name == table[i].name))
        .count()
}

// ---------------------------------------------------------------------------------------------
// Negative controls: deliberately wrong backends. A green conformance run means something only
// if each of these is caught, and caught only in the op it breaks.
// ---------------------------------------------------------------------------------------------

pub mod negative_controls {
    //! Each module is `simd::scalar` with exactly one op replaced by a plausible but wrong
    //! implementation (the kind of mistake a hand-written NEON / SSE port makes). `TARGET` names
    //! the broken op.

    use crate::simd::scalar::{v128, v128_load, v128_store};

    fn to_array<T: Copy + Default, const N: usize>(v: v128) -> [T; N] {
        const { assert!(core::mem::size_of::<[T; N]>() == 16) };
        let mut out = [T::default(); N];
        // SAFETY: `out` is exactly 16 bytes and `v128_store` allows unaligned pointers.
        unsafe { v128_store(out.as_mut_ptr() as *mut v128, v) };
        out
    }

    fn from_array<T: Copy, const N: usize>(a: [T; N]) -> v128 {
        const { assert!(core::mem::size_of::<[T; N]>() == 16) };
        // SAFETY: `a` is exactly 16 bytes and `v128_load` allows unaligned pointers.
        unsafe { v128_load(a.as_ptr() as *const v128) }
    }

    /// `f32x4_min` that returns the first operand for `{+0, -0}` (ignores the signed-zero rule).
    pub mod min_ignores_signed_zero {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "f32x4_min";
        pub fn f32x4_min(a: v128, b: v128) -> v128 {
            let (a, b): ([f32; 4], [f32; 4]) = (super::to_array(a), super::to_array(b));
            super::from_array(core::array::from_fn::<f32, 4, _>(|i| {
                if a[i].is_nan() || b[i].is_nan() {
                    f32::NAN
                } else if b[i] < a[i] {
                    b[i]
                } else {
                    a[i]
                }
            }))
        }
    }

    /// `f32x4_nearest` that rounds halfway cases away from zero instead of to even.
    pub mod nearest_ties_away {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "f32x4_nearest";
        pub fn f32x4_nearest(a: v128) -> v128 {
            let away = f32x4_floor(f32x4_add(f32x4_abs(a), f32x4_splat(0.5)));
            let neg = f32x4_lt(a, f32x4_splat(0.0));
            let r = v128_bitselect(f32x4_neg(away), away, neg);
            // keep large / non-finite values as they are, like the real op
            let big = f32x4_ge(f32x4_abs(a), f32x4_splat(8_388_608.0));
            v128_bitselect(a, r, v128_or(big, v128_not(f32x4_ge(a, a))))
        }
    }

    /// `i16x8_shr` that reduces the count modulo 32 instead of modulo 16.
    pub mod shr_count_mod_32 {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "i16x8_shr";
        pub fn i16x8_shr(a: v128, amt: u32) -> v128 {
            let l: [i16; 8] = super::to_array(a);
            super::from_array(l.map(|x| ((x as i32) >> (amt & 31)) as i16))
        }
    }

    /// `i16x8_narrow_i32x4` that wraps instead of saturating.
    pub mod narrow_wraps {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "i16x8_narrow_i32x4";
        pub fn i16x8_narrow_i32x4(a: v128, b: v128) -> v128 {
            let (a, b): ([i32; 4], [i32; 4]) = (super::to_array(a), super::to_array(b));
            super::from_array(core::array::from_fn::<i16, 8, _>(|i| {
                if i < 4 {
                    a[i] as i16
                } else {
                    b[i - 4] as i16
                }
            }))
        }
    }

    /// `u8x16_narrow_i16x8` that treats its inputs as unsigned (negative lanes become 255).
    pub mod narrow_unsigned_input {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "u8x16_narrow_i16x8";
        pub fn u8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
            let (a, b): ([u16; 8], [u16; 8]) = (super::to_array(a), super::to_array(b));
            super::from_array(core::array::from_fn::<u8, 16, _>(|i| {
                let x = if i < 8 { a[i] } else { b[i - 8] };
                if x > 255 {
                    255
                } else {
                    x as u8
                }
            }))
        }
    }

    /// `f32x4_pmax` implemented as the NaN-propagating `f32x4_max`.
    pub mod pmax_propagates_nan {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "f32x4_pmax";
        pub fn f32x4_pmax(a: v128, b: v128) -> v128 {
            f32x4_max(a, b)
        }
    }

    /// `i32x4_trunc_sat_f32x4` with x86 `cvttps2dq` behaviour: NaN and out-of-range give `i32::MIN`.
    pub mod trunc_sat_x86_indefinite {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "i32x4_trunc_sat_f32x4";
        pub fn i32x4_trunc_sat_f32x4(a: v128) -> v128 {
            let l: [f32; 4] = super::to_array(a);
            super::from_array(l.map(|x| {
                if x.is_nan() || !(-2_147_483_648.0..2_147_483_648.0).contains(&x) {
                    i32::MIN
                } else {
                    x as i32
                }
            }))
        }
    }

    /// `i16x8_q15mulr_sat` without the saturation of `-32768 * -32768`.
    pub mod q15mulr_wraps {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "i16x8_q15mulr_sat";
        pub fn i16x8_q15mulr_sat(a: v128, b: v128) -> v128 {
            let (a, b): ([i16; 8], [i16; 8]) = (super::to_array(a), super::to_array(b));
            super::from_array(core::array::from_fn::<i16, 8, _>(|i| {
                ((a[i] as i32 * b[i] as i32 + 0x4000) >> 15) as i16
            }))
        }
    }

    /// `f32x4_sqrt` that rounds toward zero instead of to nearest.
    pub mod sqrt_truncates {
        pub use crate::simd::scalar::*;
        pub const TARGET: &str = "f32x4_sqrt";
        pub fn f32x4_sqrt(a: v128) -> v128 {
            let x: [f32; 4] = super::to_array(a);
            let r: [f32; 4] = super::to_array(crate::simd::scalar::f32x4_sqrt(a));
            super::from_array(core::array::from_fn::<f32, 4, _>(|i| {
                let (xi, ri) = (x[i] as f64, r[i] as f64);
                if ri.is_finite() && ri > 0.0 && ri * ri > xi {
                    f32::from_bits(r[i].to_bits() - 1) // the root was rounded up: step down
                } else {
                    r[i]
                }
            }))
        }
    }

    backend_table!(
        min_ignores_signed_zero_table,
        crate::simd::conformance::negative_controls::min_ignores_signed_zero
    );
    backend_table!(
        nearest_ties_away_table,
        crate::simd::conformance::negative_controls::nearest_ties_away
    );
    backend_table!(
        shr_count_mod_32_table,
        crate::simd::conformance::negative_controls::shr_count_mod_32
    );
    backend_table!(
        narrow_wraps_table,
        crate::simd::conformance::negative_controls::narrow_wraps
    );
    backend_table!(
        narrow_unsigned_input_table,
        crate::simd::conformance::negative_controls::narrow_unsigned_input
    );
    backend_table!(
        pmax_propagates_nan_table,
        crate::simd::conformance::negative_controls::pmax_propagates_nan
    );
    backend_table!(
        trunc_sat_x86_indefinite_table,
        crate::simd::conformance::negative_controls::trunc_sat_x86_indefinite
    );
    backend_table!(
        q15mulr_wraps_table,
        crate::simd::conformance::negative_controls::q15mulr_wraps
    );
    backend_table!(
        sqrt_truncates_table,
        crate::simd::conformance::negative_controls::sqrt_truncates
    );

    /// `(broken op, table)` for every negative control.
    pub static ALL: &[(&str, &[super::OpEntry])] = &[
        (
            min_ignores_signed_zero::TARGET,
            min_ignores_signed_zero_table::TABLE,
        ),
        (nearest_ties_away::TARGET, nearest_ties_away_table::TABLE),
        (shr_count_mod_32::TARGET, shr_count_mod_32_table::TABLE),
        (narrow_wraps::TARGET, narrow_wraps_table::TABLE),
        (
            narrow_unsigned_input::TARGET,
            narrow_unsigned_input_table::TABLE,
        ),
        (
            pmax_propagates_nan::TARGET,
            pmax_propagates_nan_table::TABLE,
        ),
        (
            trunc_sat_x86_indefinite::TARGET,
            trunc_sat_x86_indefinite_table::TABLE,
        ),
        (q15mulr_wraps::TARGET, q15mulr_wraps_table::TABLE),
        (sqrt_truncates::TARGET, sqrt_truncates_table::TABLE),
    ];
}

/// Checks a negative-control report: at least one mismatch, and every mismatching entry is the
/// broken op. Returns `(mismatches, mismatches outside the target)`.
pub fn negative_control_outcome(table: &[OpEntry], target: &str, report: &Report) -> (u32, u32) {
    let mut outside = 0u32;
    for (i, e) in table.iter().enumerate() {
        if e.name != target {
            outside = outside.saturating_add(report.stats[i].mismatches);
        }
    }
    (report.mismatches, outside)
}

// ---------------------------------------------------------------------------------------------
// wasm32 export (feature `simd-conformance`): real SIMD128 vs `simd::scalar` inside the engine.
// ---------------------------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", feature = "simd-conformance"))]
mod wasm_export {
    use super::*;
    use core::ptr::addr_of_mut;

    backend_table!(wasm_table, core::arch::wasm32);
    backend_table!(scalar_table, crate::simd::scalar);

    static mut REPORT: Report = Report::new();

    fn report() -> &'static mut Report {
        // SAFETY: wasm32 is single-threaded and the exports below never overlap.
        unsafe { &mut *addr_of_mut!(REPORT) }
    }

    fn seed_of(seed_lo: u32, seed_hi: u32) -> u64 {
        (seed_hi as u64) << 32 | seed_lo as u64
    }

    fn status(r: Result<(), PairError>) -> u32 {
        match r {
            Ok(()) => report().mismatches,
            Err(_) => u32::MAX,
        }
    }

    /// Runs `core::arch::wasm32` (as `a`) against `simd::scalar` (as `b`); returns the mismatch
    /// count, or `u32::MAX` if the tables disagree in shape. Details: `..._report_ptr`.
    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_run(
        seed_lo: u32,
        seed_hi: u32,
        n_random: u32,
    ) -> u32 {
        status(run_pair(
            wasm_table::TABLE,
            scalar_table::TABLE,
            seed_of(seed_lo, seed_hi),
            n_random,
            report(),
        ))
    }

    /// Runs `core::arch::wasm32` against negative control `which`; the gate expects mismatches,
    /// all in the control's target op.
    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_negative_control(
        which: u32,
        seed_lo: u32,
        seed_hi: u32,
        n_random: u32,
    ) -> u32 {
        match negative_controls::ALL.get(which as usize) {
            Some((_, table)) => status(run_pair(
                wasm_table::TABLE,
                table,
                seed_of(seed_lo, seed_hi),
                n_random,
                report(),
            )),
            None => u32::MAX,
        }
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_negative_control_count() -> u32 {
        negative_controls::ALL.len() as u32
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_negative_control_target_ptr(
        which: u32,
    ) -> *const u8 {
        negative_controls::ALL
            .get(which as usize)
            .map_or(core::ptr::null(), |(t, _)| t.as_ptr())
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_negative_control_target_len(which: u32) -> u32 {
        negative_controls::ALL
            .get(which as usize)
            .map_or(0, |(t, _)| t.len() as u32)
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_report_ptr() -> *const Report {
        addr_of_mut!(REPORT) as *const Report
    }

    /// Byte offset of `Report::stats` (8 bytes per entry: vectors, mismatches).
    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_stats_offset() -> u32 {
        core::mem::offset_of!(Report, stats) as u32
    }

    /// Byte offset of `Report::records`.
    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_records_offset() -> u32 {
        core::mem::offset_of!(Report, records) as u32
    }

    /// Size of one `MismatchRecord`: entry u32, vector u32, then `v[0..3]`, `s`, `out_a`, `out_b`
    /// (16 bytes each).
    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_record_size() -> u32 {
        core::mem::size_of::<MismatchRecord>() as u32
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_count() -> u32 {
        wasm_table::TABLE.len() as u32
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_name_ptr(i: u32) -> *const u8 {
        wasm_table::TABLE
            .get(i as usize)
            .map_or(core::ptr::null(), |e| e.name.as_ptr())
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_name_len(i: u32) -> u32 {
        wasm_table::TABLE
            .get(i as usize)
            .map_or(0, |e| e.name.len() as u32)
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_variant_ptr(i: u32) -> *const u8 {
        wasm_table::TABLE
            .get(i as usize)
            .map_or(core::ptr::null(), |e| e.variant.as_ptr())
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_variant_len(i: u32) -> u32 {
        wasm_table::TABLE
            .get(i as usize)
            .map_or(0, |e| e.variant.len() as u32)
    }

    /// `Check` of entry `i` (0 exact, 1 f32 lanes NaN-tolerant, 2 f64 lanes NaN-tolerant).
    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_check(i: u32) -> u32 {
        wasm_table::TABLE
            .get(i as usize)
            .map_or(u32::MAX, |e| e.check as u32)
    }

    /// Lane width in bytes used to generate (and display) entry `i`'s inputs.
    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_lane_bytes(i: u32) -> u32 {
        wasm_table::TABLE
            .get(i as usize)
            .map_or(0, |e| lane_width(e.lanes) as u32)
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_entry_is_float(i: u32) -> u32 {
        wasm_table::TABLE
            .get(i as usize)
            .map_or(0, |e| matches!(e.lanes, Lanes::F32 | Lanes::F64) as u32)
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_layer_item_count() -> u32 {
        crate::simd::LAYER_ITEMS.len() as u32
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_uncovered_layer_items() -> u32 {
        uncovered_layer_items(wasm_table::TABLE) as u32
    }

    #[no_mangle]
    pub extern "C" fn chronos_simd_conformance_distinct_ops() -> u32 {
        distinct_ops(wasm_table::TABLE) as u32
    }
}

// ---------------------------------------------------------------------------------------------
// Native tests: `cargo test --target host-tuple`.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::boxed::Box;
    use std::string::String;
    use std::vec::Vec;
    use std::{format, println};

    backend_table!(selected_table, crate::simd);
    backend_table!(scalar_table, crate::simd::scalar);
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3", target_feature = "sse4.1"))]
    backend_table!(sse_table, crate::simd::sse);

    const SEED: u64 = 0x5EED_C0DE_2026_1119;
    const N_RANDOM: u32 = 4096;

    fn describe(table: &[OpEntry], rec: &MismatchRecord) -> String {
        let e = &table[rec.entry as usize];
        format!(
            "{}{} vector {}: inputs {:02x?} scalars {:02x?}\n  a = {:02x?}\n  b = {:02x?}",
            e.name, e.variant, rec.vector, rec.input.v, rec.input.s, rec.out_a, rec.out_b
        )
    }

    /// Runs `a` against `b` and panics with every recorded mismatch if they disagree.
    pub fn assert_pair_matches(a: &[OpEntry], b: &[OpEntry]) {
        assert_pair_matches_n(a, b, N_RANDOM);
    }

    /// [`assert_pair_matches`] with `n_random` random vectors per entry.
    pub fn assert_pair_matches_n(a: &[OpEntry], b: &[OpEntry], n_random: u32) {
        let mut report = Box::new(Report::new());
        run_pair(a, b, SEED, n_random, &mut report).expect("tables must have the same shape");
        let details: Vec<String> = report.records[..report.recorded as usize]
            .iter()
            .map(|r| describe(a, r))
            .collect();
        assert_eq!(
            report.mismatches,
            0,
            "conformance mismatches:\n{}",
            details.join("\n")
        );
        let per_op = report.vectors as usize / a.len();
        println!(
            "conformance: {} ops, {} entries, {} vectors ({} per entry on average), 0 mismatches",
            distinct_ops(a),
            a.len(),
            report.vectors,
            per_op
        );
    }

    #[test]
    fn selected_backend_matches_scalar_reference() {
        assert_pair_matches(selected_table::TABLE, scalar_table::TABLE);
    }

    /// Heavy one-off run, a million random vectors per entry:
    /// `cargo test --release --target <triple> -- --ignored --nocapture heavy`.
    #[test]
    #[ignore]
    fn heavy_selected_backend_matches_scalar_reference() {
        assert_pair_matches_n(selected_table::TABLE, scalar_table::TABLE, 1_000_000);
    }

    /// `simd::sse` against the reference also when `simd-force-scalar` selects `simd::scalar`.
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3", target_feature = "sse4.1"))]
    #[test]
    fn sse_backend_matches_scalar_reference() {
        assert_pair_matches(sse_table::TABLE, scalar_table::TABLE);
    }

    /// `.cargo/config.toml` enables SSSE3 and SSE4.1 for every x86_64 target, so unless
    /// `simd-force-scalar` is on, an x86_64 build must run `simd::sse`. Landing on `simd::scalar`
    /// means the flags were lost, typically to a `RUSTFLAGS` variable, which replaces
    /// `target.*.rustflags` (see Gate S in `build.sh`).
    #[cfg(all(target_arch = "x86_64", not(feature = "simd-force-scalar")))]
    #[test]
    fn x86_64_builds_select_the_sse_backend() {
        assert!(
            cfg!(all(target_feature = "ssse3", target_feature = "sse4.1")),
            "x86_64 build without SSSE3/SSE4.1 selects the scalar backend; were the \
             `.cargo/config.toml` rustflags replaced by a RUSTFLAGS variable?"
        );
        #[cfg(all(target_feature = "ssse3", target_feature = "sse4.1"))]
        assert!(
            core::any::TypeId::of::<crate::simd::v128>()
                == core::any::TypeId::of::<crate::simd::sse::v128>(),
            "SSSE3/SSE4.1 are enabled but the layer did not select simd::sse"
        );
    }

    #[test]
    fn every_layer_item_has_a_conformance_entry() {
        let missing: Vec<&str> = crate::simd::LAYER_ITEMS
            .iter()
            .copied()
            .filter(|&n| n != "v128" && !scalar_table::TABLE.iter().any(|e| e.name == n))
            .collect();
        assert!(
            missing.is_empty(),
            "layer items without a conformance entry: {missing:?}"
        );
        assert_eq!(uncovered_layer_items(scalar_table::TABLE), 0);
        for e in scalar_table::TABLE {
            assert!(
                crate::simd::LAYER_ITEMS.contains(&e.name),
                "{} is not a layer item",
                e.name
            );
        }
    }

    /// Every const-generic instantiation written in the kernel (`*_shuffle::<..>`,
    /// `*_extract_lane::<..>`, `*_replace_lane::<..>`, `v128_store*_lane::<..>`) must have its own
    /// table entry, because a native backend may specialise per pattern.
    #[test]
    fn every_kernel_const_generic_instantiation_is_in_the_table() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut missing = Vec::new();
        let mut seen = 0usize;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|x| x.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let mut rest = text.as_str();
            while let Some(pos) = rest.find("::<") {
                let head = &rest[..pos];
                let name_start = head
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(0, |p| p + 1);
                let name = &head[name_start..];
                let tail = &rest[pos + 3..];
                let Some(end) = tail.find('>') else { break };
                let args: String = tail[..end].chars().filter(|c| !c.is_whitespace()).collect();
                rest = &tail[end..];
                let generic_op = name.ends_with("_shuffle")
                    || name.ends_with("_extract_lane")
                    || name.ends_with("_replace_lane")
                    || name.starts_with("v128_store") && name.ends_with("_lane");
                if !generic_op || !args.chars().all(|c| c.is_ascii_digit() || c == ',') {
                    continue; // not a layer op, or a macro metavariable such as `$lane`
                }
                seen += 1;
                let found = scalar_table::TABLE.iter().any(|e| {
                    e.name == name
                        && e.variant
                            .chars()
                            .filter(|c| !c.is_whitespace())
                            .collect::<String>()
                            == format!("<{args}>")
                });
                if !found {
                    missing.push(format!("{}: {name}::<{args}>", path.display()));
                }
            }
        }
        assert!(
            seen > 500,
            "scanner found only {seen} instantiations; is it still reading src/?"
        );
        missing.sort();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "add these to op_table!:\n{}",
            missing.join("\n")
        );
    }

    #[test]
    fn negative_controls_are_detected_only_in_their_target_op() {
        for (target, table) in negative_controls::ALL {
            let mut report = Box::new(Report::new());
            run_pair(scalar_table::TABLE, table, SEED, N_RANDOM, &mut report).unwrap();
            let (bad, outside) = negative_control_outcome(table, target, &report);
            assert!(bad > 0, "negative control for {target} was not detected");
            assert_eq!(
                outside, 0,
                "negative control for {target} produced mismatches in other ops"
            );
            println!("negative control {target}: {bad} mismatches, all in {target}");
        }
    }

    #[test]
    fn edge_vectors_cover_every_pool_value_and_pair() {
        // Binary f32 ops see every (a, b) pair of edge values in some lane.
        let e = scalar_table::TABLE
            .iter()
            .find(|e| e.name == "f32x4_min")
            .unwrap();
        let mut pairs = std::collections::HashSet::new();
        for k in 0..edge_count(e) {
            let x = gen_input(e, k, SEED);
            for i in 0..4 {
                pairs.insert((
                    get_lane(&x.v[0], Lanes::F32, i),
                    get_lane(&x.v[1], Lanes::F32, i),
                ));
            }
        }
        for &a in POOL_F32 {
            for &b in POOL_F32 {
                assert!(pairs.contains(&(a, b)), "pair {a:#x} {b:#x} missing");
            }
        }
        // Shifts see every count, including counts at and beyond the lane width.
        let e = scalar_table::TABLE
            .iter()
            .find(|e| e.name == "i16x8_shr")
            .unwrap();
        let counts: std::collections::HashSet<u32> = (0..edge_count(e))
            .map(|k| u32::from_le_bytes(gen_input(e, k, SEED).s[..4].try_into().unwrap()))
            .collect();
        assert!(SHIFT_COUNTS.iter().all(|c| counts.contains(c)));
    }

    #[test]
    fn runs_are_deterministic_for_a_seed() {
        for e in scalar_table::TABLE.iter().take(8) {
            for k in [0, 1, edge_count(e), edge_count(e) + 100] {
                assert_eq!(gen_input(e, k, SEED), gen_input(e, k, SEED));
            }
        }
    }
}
