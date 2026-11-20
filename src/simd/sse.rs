//! x86_64 SSE backend of the SIMD layer: SSE2, SSSE3 and SSE4.1 from `core::arch::x86_64`.
//!
//! Every function has the `core::arch::wasm32` name and signature and computes what
//! `simd::scalar` (the WebAssembly SIMD specification) computes, bit for bit; NaN payloads are
//! the only freedom, exactly as in `simd::conformance`. `cargo test --target <x86_64 triple>`
//! compares this module with `simd::scalar` op by op.
//!
//! # Instruction set
//!
//! SSE2 + SSSE3 + SSE4.1, fixed at compile time. This module compiles only when `ssse3` and
//! `sse4.1` are enabled for the whole crate (`.cargo/config.toml` enables them for every x86_64
//! target); there is no runtime CPU detection and no indirect call, and without those features
//! the layer selects `simd::scalar`. No AVX, AVX2 or FMA: a fused multiply-add rounds once where
//! wasm rounds twice, and results must match wasm exactly.
//!
//! Float results assume the default MXCSR (round to nearest even, FTZ and DAZ off), which Rust
//! never changes. A host that sets FTZ or DAZ breaks wasm semantics for subnormals here, and
//! equally in Rust's own scalar `f32` arithmetic on x86_64.
//!
//! # Representation
//!
//! `v128` is a `#[repr(transparent)]` newtype over `__m128i`: 16 bytes, 16-byte aligned, `Copy`,
//! kept in an XMM register, and a distinct type like `core::arch::wasm32::v128` (kernel code
//! cannot call raw x86 intrinsics on it, which would not port to the other backends). x86 is
//! little-endian, so lane `i` of every lane width occupies the same bytes as in wasm and in
//! memory. Float ops view the register as `__m128` / `__m128d`; those casts emit no code.
//!
//! # Where x86 differs from wasm
//!
//! Each op below that needs more than one instruction says why in its doc comment. In short:
//! `minps`/`maxps` return their second operand on NaN and do not order signed zeros
//! (`f32x4_min`/`max` need both operand orders); `cvttps2dq` returns `0x8000_0000` for NaN and
//! overflow (`i32x4_trunc_sat_f32x4`); there is no unsigned `u32 -> f32` conversion
//! (`f32x4_convert_u32x4`); `pmulhrsw` wraps `-32768 * -32768` (`i16x8_q15mulr_sat`); there are no
//! unsigned compares (`u8x16_*`, `u32x4_lt`/`ge`); x86 shifts by a count at or above the lane
//! width give 0 or sign-fill where wasm takes the count modulo the width; and there are no
//! single-instruction widening multiplies (`extmul`). Loads and stores use unaligned forms, as
//! in wasm (`movdqu`, `movd`, `movq`, `pextrd`/`pextrq` to memory, or the float-domain
//! equivalents `movups`, `movss`, `movsd`, `extractps`, `movlps` that LLVM may pick instead).
//!
//! Instruction counts in the doc comments leave out constants loaded from memory and the
//! all-ones (`pcmpeqd x, x`) or zero (`pxor x, x`) registers; inside a loop LLVM materialises
//! those once, outside it.
//!
//! # Shuffles
//!
//! `i8x16_shuffle`, `i16x8_shuffle`, `i32x4_shuffle` and `i64x2_shuffle` turn their const-generic
//! indices into a byte pattern and a `Plan` at compile time (`const` blocks); the `match`es on
//! those constants keep only the planned instruction sequence (rustc drops the other arms of a
//! `match` on a constant even in debug builds, and LLVM folds them in release). Patterns made of
//! whole 32-bit lanes always take at most two instructions (`pshufd`, `shufps`, `unpck*`,
//! `palignr`, `blendps`, `insertps`, or `shufps`/`insertps` after a `pshufd`). 16- and 8-bit
//! patterns try word and byte forms (`pshuflw`/`pshufhw`, `punpck*`, `pblendw`, `pblendvb`,
//! `palignr`, `packuswb`) before the generic `pshufb` (one source) or `pshufb | pshufb` (two).
//!
//! LLVM represents every x86 shuffle intrinsic as a generic `shufflevector`, merges chains of
//! them and picks the final instructions itself, so the plan fixes what LLVM is given, not
//! always what it emits. In release builds it emits the planned sequence, or an equivalent one
//! no longer, for every pattern the kernel uses, with one exception: the two
//! 16-bit 4x2 transposes (`<0, 4, 8, 12, 1, 5, 9, 13>`, `<2, 6, 10, 14, 3, 7, 11, 15>`), which
//! four `punpck*wd` could do as a pair, are merged into one mask each and lowered as
//! `pshufd`, `pshuflw` on each input and `punpckldq` (five each), whatever intrinsics are used.
//!
//! # Safety
//!
//! Rust makes every `core::arch::x86_64` intrinsic `unsafe` to call from a function that does not
//! itself carry `#[target_feature]`, SSE2 ones included and even when the feature is enabled for
//! the whole build, because the running CPU might lack the instruction; `#[target_feature]`
//! cannot be combined with `#[inline(always)]`. So intrinsic calls here sit in `unsafe` blocks,
//! all justified by one invariant: this module compiles only when SSSE3 and SSE4.1 (which imply
//! SSE2 and SSE3) are enabled for the whole crate, which the `compile_error!` below enforces, so
//! the compiler may already emit these instructions anywhere in the build and the build as a
//! whole requires a CPU that has them. The only other `unsafe` code is the plain-data
//! `transmute`s (commented where they occur) and the loads and stores, whose `core::arch::wasm32`
//! signatures are `unsafe fn` (raw pointers).

#![allow(non_camel_case_types)]

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "ssse3",
    target_feature = "sse4.1"
)))]
compile_error!("simd::sse needs x86_64 with the ssse3 and sse4.1 target features enabled");

use core::arch::x86_64::*;

/// 128-bit SIMD register: an SSE `__m128i` holding lanes in wasm (little-endian) order.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct v128(__m128i);

const _: () = assert!(core::mem::size_of::<v128>() == 16 && core::mem::align_of::<v128>() == 16);

#[inline(always)]
fn ps(a: v128) -> __m128 {
    unsafe { _mm_castsi128_ps(a.0) }
}

#[inline(always)]
fn pd(a: v128) -> __m128d {
    unsafe { _mm_castsi128_pd(a.0) }
}

#[inline(always)]
fn from_ps(a: __m128) -> v128 {
    unsafe { v128(_mm_castps_si128(a)) }
}

#[inline(always)]
fn from_pd(a: __m128d) -> v128 {
    unsafe { v128(_mm_castpd_si128(a)) }
}

#[inline(always)]
fn ones() -> __m128i {
    unsafe { _mm_set1_epi32(-1) }
}

// ---------------------------------------------------------------------------------------------
// SSSE3 and SSE4.1 intrinsics (and below, the SSE2 shuffles) as safe functions named after the
// instruction, so op bodies and the shuffle dispatch read like the assembly. Every `unsafe`
// block in this module that calls an intrinsic relies on "Safety" in the module docs.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
fn pshufb(a: __m128i, m: __m128i) -> __m128i {
    unsafe { _mm_shuffle_epi8(a, m) }
}

/// Bytes `N..N + 16` of `x ++ y` (`palignr y, x, N`).
#[inline(always)]
fn palignr<const N: i32>(x: __m128i, y: __m128i) -> __m128i {
    unsafe { _mm_alignr_epi8::<N>(y, x) }
}

#[inline(always)]
fn pmulhrsw(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_mulhrs_epi16(a, b) }
}

/// Unsigned bytes of `u` times signed bytes of `s`, adjacent pairs summed with `i16` saturation.
#[inline(always)]
fn pmaddubsw(u: __m128i, s: __m128i) -> __m128i {
    unsafe { _mm_maddubs_epi16(u, s) }
}

#[inline(always)]
fn pminsb(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_min_epi8(a, b) }
}

#[inline(always)]
fn pmaxsb(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_max_epi8(a, b) }
}

#[inline(always)]
fn pminsd(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_min_epi32(a, b) }
}

#[inline(always)]
fn pmaxsd(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_max_epi32(a, b) }
}

#[inline(always)]
fn pminud(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_min_epu32(a, b) }
}

#[inline(always)]
fn pmaxud(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_max_epu32(a, b) }
}

/// 16-bit lane `j` from `y` where bit `j` of `M` is set, else from `x`.
#[inline(always)]
fn pblendw<const M: i32>(x: __m128i, y: __m128i) -> __m128i {
    unsafe { _mm_blend_epi16::<M>(x, y) }
}

/// 32-bit lane `j` from `y` where bit `j` of `M` is set, else from `x`.
#[inline(always)]
fn blendps<const M: i32>(x: __m128i, y: __m128i) -> __m128i {
    unsafe { _mm_castps_si128(_mm_blend_ps::<M>(_mm_castsi128_ps(x), _mm_castsi128_ps(y))) }
}

/// Byte `j` from `y` where the top bit of byte `j` of `m` is set, else from `x`.
#[inline(always)]
fn pblendvb(x: __m128i, y: __m128i, m: __m128i) -> __m128i {
    unsafe { _mm_blendv_epi8(x, y, m) }
}

/// `x` with 32-bit lane `(M >> 4) & 3` replaced by lane `M >> 6` of `y` (zero mask unused).
#[inline(always)]
fn insertps<const M: i32>(x: __m128i, y: __m128i) -> __m128i {
    unsafe { _mm_castps_si128(_mm_insert_ps::<M>(_mm_castsi128_ps(x), _mm_castsi128_ps(y))) }
}

#[inline(always)]
fn pinsrd<const N: i32>(x: __m128i, v: i32) -> __m128i {
    unsafe { _mm_insert_epi32::<N>(x, v) }
}

#[inline(always)]
fn pextrd<const N: i32>(x: __m128i) -> i32 {
    unsafe { _mm_extract_epi32::<N>(x) }
}

#[inline(always)]
fn pextrq<const N: i32>(x: __m128i) -> i64 {
    unsafe { _mm_extract_epi64::<N>(x) }
}

#[inline(always)]
fn roundps<const MODE: i32>(x: __m128) -> __m128 {
    unsafe { _mm_round_ps::<MODE>(x) }
}

#[inline(always)]
fn pmovsxbw(a: __m128i) -> __m128i {
    unsafe { _mm_cvtepi8_epi16(a) }
}

#[inline(always)]
fn pmovzxbw(a: __m128i) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(a) }
}

#[inline(always)]
fn pmovsxwd(a: __m128i) -> __m128i {
    unsafe { _mm_cvtepi16_epi32(a) }
}

#[inline(always)]
fn pmovzxwd(a: __m128i) -> __m128i {
    unsafe { _mm_cvtepu16_epi32(a) }
}

/// `ptest`: whether `a & b` is all zeros.
#[inline(always)]
fn ptest_zero(a: __m128i, b: __m128i) -> bool {
    unsafe { _mm_testz_si128(a, b) != 0 }
}

// SSE2 shuffles under the same naming, so the shuffle dispatch can name every instruction alike.

#[inline(always)]
fn pshufd<const M: i32>(x: __m128i) -> __m128i {
    unsafe { _mm_shuffle_epi32::<M>(x) }
}

#[inline(always)]
fn pshuflw<const M: i32>(x: __m128i) -> __m128i {
    unsafe { _mm_shufflelo_epi16::<M>(x) }
}

#[inline(always)]
fn pshufhw<const M: i32>(x: __m128i) -> __m128i {
    unsafe { _mm_shufflehi_epi16::<M>(x) }
}

/// Lanes 0 and 1 from `x`, lanes 2 and 3 from `y`, each selected by two bits of `M`.
#[inline(always)]
fn shufps<const M: i32>(x: __m128i, y: __m128i) -> __m128i {
    unsafe {
        _mm_castps_si128(_mm_shuffle_ps::<M>(
            _mm_castsi128_ps(x),
            _mm_castsi128_ps(y),
        ))
    }
}

/// `x` shifted down by `N` bytes, zeros shifted in.
#[inline(always)]
fn psrldq<const N: i32>(x: __m128i) -> __m128i {
    unsafe { _mm_srli_si128::<N>(x) }
}

// ---------------------------------------------------------------------------------------------
// Constructors (`v128.const`).
// ---------------------------------------------------------------------------------------------

// SAFETY (every `transmute` in this section): the source is a 16-byte array of plain integers or
// floats and `__m128i` is 16 bytes of plain data for which every bit pattern is valid; x86 is
// little-endian, so array element `i` lands in lane `i`, as in wasm.

#[inline(always)]
pub const fn f32x4(a0: f32, a1: f32, a2: f32, a3: f32) -> v128 {
    v128(unsafe { core::mem::transmute::<[f32; 4], __m128i>([a0, a1, a2, a3]) })
}

#[inline(always)]
pub const fn f64x2(a0: f64, a1: f64) -> v128 {
    v128(unsafe { core::mem::transmute::<[f64; 2], __m128i>([a0, a1]) })
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub const fn i16x8(a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16) -> v128 {
    v128(unsafe { core::mem::transmute::<[i16; 8], __m128i>([a0, a1, a2, a3, a4, a5, a6, a7]) })
}

#[inline(always)]
pub const fn i32x4(a0: i32, a1: i32, a2: i32, a3: i32) -> v128 {
    v128(unsafe { core::mem::transmute::<[i32; 4], __m128i>([a0, a1, a2, a3]) })
}

#[inline(always)]
pub const fn u32x4(a0: u32, a1: u32, a2: u32, a3: u32) -> v128 {
    v128(unsafe { core::mem::transmute::<[u32; 4], __m128i>([a0, a1, a2, a3]) })
}

#[inline(always)]
pub const fn u64x2(a0: u64, a1: u64) -> v128 {
    v128(unsafe { core::mem::transmute::<[u64; 2], __m128i>([a0, a1]) })
}

// ---------------------------------------------------------------------------------------------
// Splats, lane extraction and replacement.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn f32x4_splat(a: f32) -> v128 {
    unsafe { from_ps(_mm_set1_ps(a)) }
}

#[inline(always)]
pub fn f64x2_splat(a: f64) -> v128 {
    unsafe { from_pd(_mm_set1_pd(a)) }
}

#[inline(always)]
pub fn i8x16_splat(a: i8) -> v128 {
    unsafe { v128(_mm_set1_epi8(a)) }
}

#[inline(always)]
pub fn u8x16_splat(a: u8) -> v128 {
    unsafe { v128(_mm_set1_epi8(a as i8)) }
}

#[inline(always)]
pub fn i16x8_splat(a: i16) -> v128 {
    unsafe { v128(_mm_set1_epi16(a)) }
}

#[inline(always)]
pub fn u16x8_splat(a: u16) -> v128 {
    unsafe { v128(_mm_set1_epi16(a as i16)) }
}

#[inline(always)]
pub fn i32x4_splat(a: i32) -> v128 {
    unsafe { v128(_mm_set1_epi32(a)) }
}

#[inline(always)]
pub fn u32x4_splat(a: u32) -> v128 {
    unsafe { v128(_mm_set1_epi32(a as i32)) }
}

#[inline(always)]
pub fn f32x4_extract_lane<const N: usize>(a: v128) -> f32 {
    const { assert!(N < 4) };
    unsafe {
        let x = ps(a);
        _mm_cvtss_f32(match N {
            0 => x,
            1 => _mm_shuffle_ps::<0b01_01_01_01>(x, x),
            2 => _mm_movehl_ps(x, x),
            _ => _mm_shuffle_ps::<0b11_11_11_11>(x, x),
        })
    }
}

#[inline(always)]
pub fn f64x2_extract_lane<const N: usize>(a: v128) -> f64 {
    const { assert!(N < 2) };
    unsafe {
        let x = pd(a);
        _mm_cvtsd_f64(match N {
            0 => x,
            _ => _mm_unpackhi_pd(x, x),
        })
    }
}

#[inline(always)]
pub fn i32x4_extract_lane<const N: usize>(a: v128) -> i32 {
    const { assert!(N < 4) };
    unsafe {
        match N {
            0 => _mm_cvtsi128_si32(a.0),
            1 => pextrd::<1>(a.0),
            2 => pextrd::<2>(a.0),
            _ => pextrd::<3>(a.0),
        }
    }
}

#[inline(always)]
pub fn u32x4_extract_lane<const N: usize>(a: v128) -> u32 {
    i32x4_extract_lane::<N>(a) as u32
}

#[inline(always)]
pub fn u64x2_extract_lane<const N: usize>(a: v128) -> u64 {
    const { assert!(N < 2) };
    unsafe {
        (match N {
            0 => _mm_cvtsi128_si64(a.0),
            _ => pextrq::<1>(a.0),
        }) as u64
    }
}

#[inline(always)]
pub fn f32x4_replace_lane<const N: usize>(a: v128, val: f32) -> v128 {
    const { assert!(N < 4) };
    unsafe {
        let v = _mm_castps_si128(_mm_set_ss(val));
        v128(match N {
            0 => insertps::<0x00>(a.0, v),
            1 => insertps::<0x10>(a.0, v),
            2 => insertps::<0x20>(a.0, v),
            _ => insertps::<0x30>(a.0, v),
        })
    }
}

#[inline(always)]
pub fn i32x4_replace_lane<const N: usize>(a: v128, val: i32) -> v128 {
    const { assert!(N < 4) };
    v128(match N {
        0 => pinsrd::<0>(a.0, val),
        1 => pinsrd::<1>(a.0, val),
        2 => pinsrd::<2>(a.0, val),
        _ => pinsrd::<3>(a.0, val),
    })
}

// ---------------------------------------------------------------------------------------------
// f32x4.
// ---------------------------------------------------------------------------------------------

/// Clears the sign bit (`andps` with a constant): NaN payloads are preserved, as in wasm.
#[inline(always)]
pub fn f32x4_abs(a: v128) -> v128 {
    unsafe { v128(_mm_and_si128(a.0, _mm_set1_epi32(0x7FFF_FFFF))) }
}

/// Flips the sign bit (`xorps` with a constant): NaN payloads are preserved, as in wasm.
#[inline(always)]
pub fn f32x4_neg(a: v128) -> v128 {
    unsafe { v128(_mm_xor_si128(a.0, _mm_set1_epi32(i32::MIN))) }
}

#[inline(always)]
pub fn f32x4_add(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_add_ps(ps(a), ps(b))) }
}

#[inline(always)]
pub fn f32x4_sub(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_sub_ps(ps(a), ps(b))) }
}

#[inline(always)]
pub fn f32x4_mul(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_mul_ps(ps(a), ps(b))) }
}

#[inline(always)]
pub fn f32x4_div(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_div_ps(ps(a), ps(b))) }
}

/// `sqrtps`: correctly rounded, as wasm requires.
#[inline(always)]
pub fn f32x4_sqrt(a: v128) -> v128 {
    unsafe { from_ps(_mm_sqrt_ps(ps(a))) }
}

/// `roundps` toward negative infinity, precision exception suppressed.
#[inline(always)]
pub fn f32x4_floor(a: v128) -> v128 {
    from_ps(roundps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(
        ps(a),
    ))
}

/// `roundps` to nearest, ties to even, precision exception suppressed; zero results keep the
/// input's sign, as in wasm.
#[inline(always)]
pub fn f32x4_nearest(a: v128) -> v128 {
    from_ps(roundps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(ps(a)))
}

/// Wasm `min`: NaN if either lane is NaN, and `-0 < +0`. Three instructions:
/// `minps(a, b) | minps(b, a)`.
///
/// `minps(x, y)` is `x < y ? x : y`, so it returns `y` when either lane is NaN and when the lanes
/// are equal (including `{+0, -0}`). Computing it both ways yields both inputs exactly in those
/// cases and the true minimum twice otherwise; OR-ing the two gives `-0` for `{+0, -0}`, a NaN
/// when either input is NaN (all exponent bits and a non-zero mantissa survive the OR), and the
/// minimum otherwise. No single SSE instruction orders signed zeros or propagates NaN from both
/// operands, so the two `minps` are needed, and one more instruction must merge them.
#[inline(always)]
pub fn f32x4_min(a: v128, b: v128) -> v128 {
    unsafe {
        let (a, b) = (ps(a), ps(b));
        from_ps(_mm_or_ps(_mm_min_ps(a, b), _mm_min_ps(b, a)))
    }
}

/// Wasm `max`: NaN if either lane is NaN, and `+0 > -0`. Five instructions:
/// `(maxps(a, b) & maxps(b, a)) | cmpunordps(a, b)`.
///
/// As in [`f32x4_min`], the two `maxps` agree except for NaN and `{+0, -0}`, where they return
/// both inputs. AND-ing gives `+0` for the zeros (and the maximum otherwise), but can turn a NaN
/// into a number, so the NaN lanes are forced to all-ones (a NaN) with the unordered-compare mask,
/// which runs in parallel with the `maxps`. Unlike `min`, the merge cannot be a single bitwise op:
/// OR would give `-0`, AND loses NaNs.
#[inline(always)]
pub fn f32x4_max(a: v128, b: v128) -> v128 {
    unsafe {
        let (a, b) = (ps(a), ps(b));
        let m = _mm_and_ps(_mm_max_ps(a, b), _mm_max_ps(b, a));
        from_ps(_mm_or_ps(m, _mm_cmpunord_ps(a, b)))
    }
}

/// Wasm `pmin` is `b < a ? b : a`, which is exactly `minps(b, a)` (bits of one input, never a
/// NaN from `b`).
#[inline(always)]
pub fn f32x4_pmin(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_min_ps(ps(b), ps(a))) }
}

/// Wasm `pmax` is `a < b ? b : a`, which is exactly `maxps(b, a)`.
#[inline(always)]
pub fn f32x4_pmax(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_max_ps(ps(b), ps(a))) }
}

#[inline(always)]
pub fn f32x4_lt(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_cmplt_ps(ps(a), ps(b))) }
}

#[inline(always)]
pub fn f32x4_le(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_cmple_ps(ps(a), ps(b))) }
}

/// `cmpltps` with the operands swapped (false for NaN, as in wasm).
#[inline(always)]
pub fn f32x4_gt(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_cmpgt_ps(ps(a), ps(b))) }
}

/// `cmpleps` with the operands swapped (false for NaN, as in wasm).
#[inline(always)]
pub fn f32x4_ge(a: v128, b: v128) -> v128 {
    unsafe { from_ps(_mm_cmpge_ps(ps(a), ps(b))) }
}

/// `cvtdq2ps`: rounds to nearest, ties to even (default MXCSR).
#[inline(always)]
pub fn f32x4_convert_i32x4(a: v128) -> v128 {
    unsafe { from_ps(_mm_cvtepi32_ps(a.0)) }
}

/// Unsigned `u32` to `f32`, correctly rounded. Five instructions; SSE has no unsigned convert.
///
/// Each lane `x` is split into its 16-bit halves, each turned into an exact float by bit
/// tricks: `lo = 2^23 + (x & 0xFFFF)` (`pblendw` of `x` with the bits of 2^23) and
/// `hi = 2^39 + (x >> 16) * 2^16` (`psrld`, then `pblendw` with the bits of 2^39).
/// `hi - (2^39 + 2^23)` is exact (`subps`: a multiple of 2^16 below 2^32 in magnitude), and
/// adding `lo` (`addps`) yields `x` with a single rounding. A shorter route through the signed
/// `cvtdq2ps` rounds twice for `x >= 2^31`; this is also what LLVM emits for `uitofp`.
#[inline(always)]
pub fn f32x4_convert_u32x4(a: v128) -> v128 {
    unsafe {
        let lo = pblendw::<0xAA>(a.0, _mm_set1_epi32(0x4B00_0000));
        let hi = pblendw::<0xAA>(_mm_srli_epi32::<16>(a.0), _mm_set1_epi32(0x5300_0000));
        let hi = _mm_sub_ps(
            _mm_castsi128_ps(hi),
            _mm_set1_ps(f32::from_bits(0x5300_0080)),
        );
        from_ps(_mm_add_ps(hi, _mm_castsi128_ps(lo)))
    }
}

/// `f32` to `i32` toward zero, saturating, NaN to 0. Five instructions:
/// `(cvttps2dq(a) ^ cmpleps(2^31, a)) & cmpordps(a, a)`.
///
/// `cvttps2dq` already truncates and gives `i32::MIN` for every negative overflow, but it also
/// gives `0x8000_0000` for positive overflow and for NaN. XOR-ing the `a >= 2^31` mask turns the
/// former into `0x7FFF_FFFF`; AND-ing the ordered mask zeroes NaN lanes. The two masks are
/// independent of the conversion, so the sequence is two instructions deep after it.
#[inline(always)]
pub fn i32x4_trunc_sat_f32x4(a: v128) -> v128 {
    unsafe {
        let x = ps(a);
        let r = _mm_cvttps_epi32(x);
        let over = _mm_castps_si128(_mm_cmpge_ps(x, _mm_set1_ps(2_147_483_648.0)));
        let ordered = _mm_castps_si128(_mm_cmpord_ps(x, x));
        v128(_mm_and_si128(_mm_xor_si128(r, over), ordered))
    }
}

// ---------------------------------------------------------------------------------------------
// f64x2.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn f64x2_add(a: v128, b: v128) -> v128 {
    unsafe { from_pd(_mm_add_pd(pd(a), pd(b))) }
}

#[inline(always)]
pub fn f64x2_sub(a: v128, b: v128) -> v128 {
    unsafe { from_pd(_mm_sub_pd(pd(a), pd(b))) }
}

#[inline(always)]
pub fn f64x2_mul(a: v128, b: v128) -> v128 {
    unsafe { from_pd(_mm_mul_pd(pd(a), pd(b))) }
}

#[inline(always)]
pub fn f64x2_sqrt(a: v128) -> v128 {
    unsafe { from_pd(_mm_sqrt_pd(pd(a))) }
}

// ---------------------------------------------------------------------------------------------
// i8x16 / u8x16.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn i8x16_add(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_add_epi8(a.0, b.0)) }
}

#[inline(always)]
pub fn i8x16_sub(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_sub_epi8(a.0, b.0)) }
}

#[inline(always)]
pub fn i8x16_min(a: v128, b: v128) -> v128 {
    v128(pminsb(a.0, b.0))
}

#[inline(always)]
pub fn i8x16_max(a: v128, b: v128) -> v128 {
    v128(pmaxsb(a.0, b.0))
}

/// `pmovmskb`: bit `i` is the sign bit of lane `i`.
#[inline(always)]
pub fn i8x16_bitmask(a: v128) -> u16 {
    unsafe { _mm_movemask_epi8(a.0) as u16 }
}

/// `packsswb`: signed saturation, lanes 0..8 from `a`, 8..16 from `b`, as in wasm.
#[inline(always)]
pub fn i8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_packs_epi16(a.0, b.0)) }
}

#[inline(always)]
pub fn u8x16_add_sat(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_adds_epu8(a.0, b.0)) }
}

#[inline(always)]
pub fn u8x16_sub_sat(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_subs_epu8(a.0, b.0)) }
}

#[inline(always)]
pub fn u8x16_min(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_min_epu8(a.0, b.0)) }
}

#[inline(always)]
pub fn u8x16_max(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_max_epu8(a.0, b.0)) }
}

/// `!(a >= b)`: `pmaxub`, `pcmpeqb`, then an XOR with all-ones (three instructions). SSE has only
/// signed byte compares; the alternative (flip both sign bits, `pcmpgtb`) is also three. Kept as
/// a NOT so LLVM can fold it into a following `andnot` / `bitselect`.
#[inline(always)]
pub fn u8x16_lt(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_xor_si128(u8x16_ge(a, b).0, ones())) }
}

/// `a <= b` as `pminub(a, b) == a` (two instructions; SSE has only signed byte compares).
#[inline(always)]
pub fn u8x16_le(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_cmpeq_epi8(_mm_min_epu8(a.0, b.0), a.0)) }
}

/// `!(a <= b)`: three instructions, as [`u8x16_lt`].
#[inline(always)]
pub fn u8x16_gt(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_xor_si128(u8x16_le(a, b).0, ones())) }
}

/// `a >= b` as `pmaxub(a, b) == a` (two instructions; SSE has only signed byte compares).
#[inline(always)]
pub fn u8x16_ge(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_cmpeq_epi8(_mm_max_epu8(a.0, b.0), a.0)) }
}

/// `packuswb`: signed `i16` saturated to `u8`, as in wasm.
#[inline(always)]
pub fn u8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_packus_epi16(a.0, b.0)) }
}

// ---------------------------------------------------------------------------------------------
// i16x8 / u16x8.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn i16x8_add(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_add_epi16(a.0, b.0)) }
}

/// Same op as `i16x8_add` (wasm `i16x8.add`); `core::arch::wasm32` names it both ways.
pub use self::i16x8_add as u16x8_add;

#[inline(always)]
pub fn i16x8_sub(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_sub_epi16(a.0, b.0)) }
}

#[inline(always)]
pub fn i16x8_sub_sat(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_subs_epi16(a.0, b.0)) }
}

/// `pmullw` (wrapping 16-bit multiply).
#[inline(always)]
pub fn u16x8_mul(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_mullo_epi16(a.0, b.0)) }
}

#[inline(always)]
pub fn i16x8_min(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_min_epi16(a.0, b.0)) }
}

#[inline(always)]
pub fn i16x8_max(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_max_epi16(a.0, b.0)) }
}

/// `pcmpgtw` with the operands swapped.
#[inline(always)]
pub fn i16x8_lt(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_cmplt_epi16(a.0, b.0)) }
}

/// Shift count for `psllw`/`psraw`/`psrlw` and friends: wasm takes it modulo the lane width,
/// x86 saturates it (0 or sign fill), so it is masked first. A constant count folds into the
/// immediate form of the shift; a variable one costs `and` + `movd` before the shift.
#[inline(always)]
fn count(amt: u32, lane_bits: u32) -> __m128i {
    unsafe { _mm_cvtsi32_si128((amt & (lane_bits - 1)) as i32) }
}

/// `psllw` by `amt mod 16`.
#[inline(always)]
pub fn i16x8_shl(a: v128, amt: u32) -> v128 {
    unsafe { v128(_mm_sll_epi16(a.0, count(amt, 16))) }
}

/// `psraw` by `amt mod 16`.
#[inline(always)]
pub fn i16x8_shr(a: v128, amt: u32) -> v128 {
    unsafe { v128(_mm_sra_epi16(a.0, count(amt, 16))) }
}

/// `psrlw` by `amt mod 16`.
#[inline(always)]
pub fn u16x8_shr(a: v128, amt: u32) -> v128 {
    unsafe { v128(_mm_srl_epi16(a.0, count(amt, 16))) }
}

/// Q15 rounding multiply, saturating. Three instructions: `pmulhrsw`, then `pcmpeqw` against
/// `0x8000` and `pxor`.
///
/// `pmulhrsw` computes `(a * b + 2^14) >> 15` exactly but wraps the one overflowing case,
/// `-32768 * -32768`, to `0x8000` instead of saturating to `0x7FFF`. No other product rounds to
/// `-32768` (the most negative is `-32768 * 32767`, which gives `-32767`), so XOR-ing the lanes
/// equal to `0x8000` with all-ones fixes exactly that case.
#[inline(always)]
pub fn i16x8_q15mulr_sat(a: v128, b: v128) -> v128 {
    unsafe {
        let r = pmulhrsw(a.0, b.0);
        v128(_mm_xor_si128(
            r,
            _mm_cmpeq_epi16(r, _mm_set1_epi16(i16::MIN)),
        ))
    }
}

/// `packssdw`: signed saturation, lanes 0..4 from `a`, 4..8 from `b`.
#[inline(always)]
pub fn i16x8_narrow_i32x4(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_packs_epi32(a.0, b.0)) }
}

#[inline(always)]
pub fn i16x8_extend_low_i8x16(a: v128) -> v128 {
    v128(pmovsxbw(a.0))
}

/// Two instructions: `punpckhbw a, a` puts byte `8 + i` in both halves of word `i`, `psraw 8`
/// sign-extends it. (`pmovsxbw` reads only the low half, so it would need a shuffle first too.)
#[inline(always)]
pub fn i16x8_extend_high_i8x16(a: v128) -> v128 {
    unsafe { v128(_mm_srai_epi16::<8>(_mm_unpackhi_epi8(a.0, a.0))) }
}

#[inline(always)]
pub fn u16x8_extend_low_u8x16(a: v128) -> v128 {
    v128(pmovzxbw(a.0))
}

/// `punpckhbw` with zero.
#[inline(always)]
pub fn u16x8_extend_high_u8x16(a: v128) -> v128 {
    unsafe { v128(_mm_unpackhi_epi8(a.0, _mm_setzero_si128())) }
}

/// Three instructions: `pmovsxbw` both operands, `pmullw`. x86 has no widening byte multiply
/// (`pmaddubsw` adds pairs and treats one operand as unsigned).
#[inline(always)]
pub fn i16x8_extmul_low_i8x16(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_mullo_epi16(pmovsxbw(a.0), pmovsxbw(b.0))) }
}

/// Three instructions: `punpckhbw` with zero below each operand turns byte `x` into the word
/// `x * 256`, and `pmulhw` returns the high half of `(x * 256) * (y * 256)`, which is `x * y`
/// exactly. (`punpckhbw` overwrites its zero operand, so LLVM zeroes two registers with `pxor`,
/// an idiom handled at register rename.) Sign-extending the high bytes first would take four
/// instructions before the multiply.
#[inline(always)]
pub fn i16x8_extmul_high_i8x16(a: v128, b: v128) -> v128 {
    unsafe {
        let z = _mm_setzero_si128();
        v128(_mm_mulhi_epi16(
            _mm_unpackhi_epi8(z, a.0),
            _mm_unpackhi_epi8(z, b.0),
        ))
    }
}

/// Three instructions: `pmovzxbw` both operands, `pmullw` (the product fits in 16 bits).
#[inline(always)]
pub fn u16x8_extmul_low_u8x16(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_mullo_epi16(pmovzxbw(a.0), pmovzxbw(b.0))) }
}

/// Three instructions: `punpckhbw` with zero for both operands, `pmullw`.
#[inline(always)]
pub fn u16x8_extmul_high_u8x16(a: v128, b: v128) -> v128 {
    unsafe {
        let z = _mm_setzero_si128();
        v128(_mm_mullo_epi16(
            _mm_unpackhi_epi8(a.0, z),
            _mm_unpackhi_epi8(b.0, z),
        ))
    }
}

/// `pmaddubsw(1, a)`: unsigned 1 times each signed byte, pairs summed (range -256..=254, so the
/// `i16` saturation never applies).
#[inline(always)]
pub fn i16x8_extadd_pairwise_i8x16(a: v128) -> v128 {
    unsafe { v128(pmaddubsw(_mm_set1_epi8(1), a.0)) }
}

/// `pmaddubsw(a, 1)`: each unsigned byte times signed 1, pairs summed (at most 510).
#[inline(always)]
pub fn u16x8_extadd_pairwise_u8x16(a: v128) -> v128 {
    unsafe { v128(pmaddubsw(a.0, _mm_set1_epi8(1))) }
}

// ---------------------------------------------------------------------------------------------
// i32x4 / u32x4.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn i32x4_add(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_add_epi32(a.0, b.0)) }
}

#[inline(always)]
pub fn i32x4_sub(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_sub_epi32(a.0, b.0)) }
}

#[inline(always)]
pub fn i32x4_min(a: v128, b: v128) -> v128 {
    v128(pminsd(a.0, b.0))
}

#[inline(always)]
pub fn i32x4_max(a: v128, b: v128) -> v128 {
    v128(pmaxsd(a.0, b.0))
}

#[inline(always)]
pub fn u32x4_min(a: v128, b: v128) -> v128 {
    v128(pminud(a.0, b.0))
}

#[inline(always)]
pub fn i32x4_eq(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_cmpeq_epi32(a.0, b.0)) }
}

/// Same op as `i32x4_eq` (wasm `i32x4.eq`); `core::arch::wasm32` names it both ways.
pub use self::i32x4_eq as u32x4_eq;

/// `!(a == b)`: `pcmpeqd`, then XOR with all-ones (SSE has no not-equal compare).
#[inline(always)]
pub fn i32x4_ne(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_xor_si128(_mm_cmpeq_epi32(a.0, b.0), ones())) }
}

/// `pcmpgtd` with the operands swapped.
#[inline(always)]
pub fn i32x4_lt(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_cmplt_epi32(a.0, b.0)) }
}

/// `a >= b`, written as `pmaxsd(a, b) == a`. LLVM canonicalises it to `!(b > a)`, which it emits
/// as `pcmpgtd` and an XOR with all-ones. Two instructions either way: SSE has only `>` and `==`.
#[inline(always)]
pub fn i32x4_ge(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_cmpeq_epi32(pmaxsd(a.0, b.0), a.0)) }
}

/// `!(a >= b)`: `pmaxud`, `pcmpeqd`, XOR with all-ones (three instructions; SSE has only signed
/// compares, and flipping both sign bits before `pcmpgtd` is also three).
#[inline(always)]
pub fn u32x4_lt(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_xor_si128(u32x4_ge(a, b).0, ones())) }
}

/// `a >= b` as `pmaxud(a, b) == a` (two instructions; SSE has only signed compares).
#[inline(always)]
pub fn u32x4_ge(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_cmpeq_epi32(pmaxud(a.0, b.0), a.0)) }
}

/// `pslld` by `amt mod 32`.
#[inline(always)]
pub fn i32x4_shl(a: v128, amt: u32) -> v128 {
    unsafe { v128(_mm_sll_epi32(a.0, count(amt, 32))) }
}

/// `psrad` by `amt mod 32`.
#[inline(always)]
pub fn i32x4_shr(a: v128, amt: u32) -> v128 {
    unsafe { v128(_mm_sra_epi32(a.0, count(amt, 32))) }
}

/// `psrld` by `amt mod 32`.
#[inline(always)]
pub fn u32x4_shr(a: v128, amt: u32) -> v128 {
    unsafe { v128(_mm_srl_epi32(a.0, count(amt, 32))) }
}

/// No lane is zero: `pcmpeqd` against zero, then `ptest` of that mask with itself.
#[inline(always)]
pub fn i32x4_all_true(a: v128) -> bool {
    unsafe {
        let zero_lanes = _mm_cmpeq_epi32(a.0, _mm_setzero_si128());
        ptest_zero(zero_lanes, zero_lanes)
    }
}

/// `movmskps`: bit `i` is the sign bit of lane `i`.
#[inline(always)]
pub fn i32x4_bitmask(a: v128) -> u8 {
    unsafe { _mm_movemask_ps(ps(a)) as u8 }
}

/// `pmaddwd`; like wasm it wraps only for all-`-32768` inputs (to `i32::MIN`).
#[inline(always)]
pub fn i32x4_dot_i16x8(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_madd_epi16(a.0, b.0)) }
}

#[inline(always)]
pub fn i32x4_extend_low_i16x8(a: v128) -> v128 {
    v128(pmovsxwd(a.0))
}

/// Two instructions: `punpckhwd a, a`, `psrad 16` (as [`i16x8_extend_high_i8x16`]).
#[inline(always)]
pub fn i32x4_extend_high_i16x8(a: v128) -> v128 {
    unsafe { v128(_mm_srai_epi32::<16>(_mm_unpackhi_epi16(a.0, a.0))) }
}

#[inline(always)]
pub fn u32x4_extend_low_u16x8(a: v128) -> v128 {
    v128(pmovzxwd(a.0))
}

/// `punpckhwd` with zero.
#[inline(always)]
pub fn u32x4_extend_high_u16x8(a: v128) -> v128 {
    unsafe { v128(_mm_unpackhi_epi16(a.0, _mm_setzero_si128())) }
}

/// Three instructions: `pmullw` and `pmulhw` give the low and high halves of the eight 32-bit
/// products, `punpcklwd` interleaves the low four. (`pmovsxwd` twice plus `pmulld` is four, and
/// `pmulld` is slow.) With [`i32x4_extmul_high_i16x8`] on the same inputs the two multiplies are
/// shared.
#[inline(always)]
pub fn i32x4_extmul_low_i16x8(a: v128, b: v128) -> v128 {
    unsafe {
        let lo = _mm_mullo_epi16(a.0, b.0);
        let hi = _mm_mulhi_epi16(a.0, b.0);
        v128(_mm_unpacklo_epi16(lo, hi))
    }
}

/// Three instructions: as [`i32x4_extmul_low_i16x8`], with `punpckhwd`.
#[inline(always)]
pub fn i32x4_extmul_high_i16x8(a: v128, b: v128) -> v128 {
    unsafe {
        let lo = _mm_mullo_epi16(a.0, b.0);
        let hi = _mm_mulhi_epi16(a.0, b.0);
        v128(_mm_unpackhi_epi16(lo, hi))
    }
}

/// `pmaddwd(a, 1)`.
#[inline(always)]
pub fn i32x4_extadd_pairwise_i16x8(a: v128) -> v128 {
    unsafe { v128(_mm_madd_epi16(a.0, _mm_set1_epi16(1))) }
}

/// Three instructions: `psrld 16` (high halves), `pblendw` with zero (low halves), `paddd`.
/// `pmaddwd` treats words as signed; correcting for that also takes three (`pxor`, `pmaddwd`,
/// `paddd`) plus two constants.
#[inline(always)]
pub fn u32x4_extadd_pairwise_u16x8(a: v128) -> v128 {
    unsafe {
        let low_halves = pblendw::<0xAA>(a.0, _mm_setzero_si128());
        v128(_mm_add_epi32(_mm_srli_epi32::<16>(a.0), low_halves))
    }
}

// ---------------------------------------------------------------------------------------------
// Whole-register bitwise ops.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn v128_and(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_and_si128(a.0, b.0)) }
}

/// `a & !b` (`pandn` takes the complemented operand first).
#[inline(always)]
pub fn v128_andnot(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_andnot_si128(b.0, a.0)) }
}

#[inline(always)]
pub fn v128_or(a: v128, b: v128) -> v128 {
    unsafe { v128(_mm_or_si128(a.0, b.0)) }
}

/// XOR with all-ones (SSE has no NOT; the all-ones register is a dependency-free idiom).
#[inline(always)]
pub fn v128_not(a: v128) -> v128 {
    unsafe { v128(_mm_xor_si128(a.0, ones())) }
}

/// Per bit `c ? v1 : v2`: `pand`, `pandn`, `por` (three instructions, two deep). `pblendvb`
/// selects whole bytes by their top bit, so it cannot implement a per-bit select.
#[inline(always)]
pub fn v128_bitselect(v1: v128, v2: v128, c: v128) -> v128 {
    unsafe {
        v128(_mm_or_si128(
            _mm_and_si128(v1.0, c.0),
            _mm_andnot_si128(c.0, v2.0),
        ))
    }
}

/// `ptest a, a`: true unless every bit is zero.
#[inline(always)]
pub fn v128_any_true(a: v128) -> bool {
    !ptest_zero(a.0, a.0)
}

// ---------------------------------------------------------------------------------------------
// Loads and stores (unaligned, as in wasm).
// ---------------------------------------------------------------------------------------------

/// Unaligned 16-byte load (`movdqu` / `movups`).
///
/// # Safety
/// `m` must be valid for a 16-byte read; no alignment is required.
#[inline(always)]
pub unsafe fn v128_load(m: *const v128) -> v128 {
    v128(_mm_loadu_si128(m as *const __m128i))
}

/// 32 bits into lane 0, the rest zeroed (`movd` / `movss` from memory).
///
/// # Safety
/// `m` must be valid for a 4-byte read; no alignment is required.
#[inline(always)]
pub unsafe fn v128_load32_zero(m: *const u32) -> v128 {
    v128(_mm_cvtsi32_si128(core::ptr::read_unaligned(m) as i32))
}

/// 64 bits into lane 0, the rest zeroed (`movq` / `movsd` from memory).
///
/// # Safety
/// `m` must be valid for an 8-byte read; no alignment is required.
#[inline(always)]
pub unsafe fn v128_load64_zero(m: *const u64) -> v128 {
    v128(_mm_cvtsi64_si128(core::ptr::read_unaligned(m) as i64))
}

/// Unaligned 16-byte store (`movdqu` / `movups`).
///
/// # Safety
/// `m` must be valid for a 16-byte write; no alignment is required.
#[inline(always)]
pub unsafe fn v128_store(m: *mut v128, a: v128) {
    _mm_storeu_si128(m as *mut __m128i, a.0)
}

/// Stores 32-bit lane `L`: `movd` / `movss` for lane 0, `pextrd` / `extractps` to memory otherwise.
///
/// # Safety
/// `m` must be valid for a 4-byte write; no alignment is required.
#[inline(always)]
pub unsafe fn v128_store32_lane<const L: usize>(v: v128, m: *mut u32) {
    core::ptr::write_unaligned(m, u32x4_extract_lane::<L>(v))
}

/// Stores 64-bit lane `L`: `movq` / `movlps` for lane 0, `pextrq` / `movhps` to memory for lane 1.
///
/// # Safety
/// `m` must be valid for an 8-byte write; no alignment is required.
#[inline(always)]
pub unsafe fn v128_store64_lane<const L: usize>(v: v128, m: *mut u64) {
    core::ptr::write_unaligned(m, u64x2_extract_lane::<L>(v))
}

// ---------------------------------------------------------------------------------------------
// Shuffles.
//
// A shuffle is first written as a byte pattern: `bytes[j]` is the index, in the 32-byte
// concatenation `a ++ b`, of the byte that lands in result byte `j`. `plan` maps the pattern to
// an instruction sequence (a `Plan`) at compile time, preferring the widest lane size the
// pattern allows: whole 32-bit lanes first, then 16-bit, then bytes; `pshufb` is the last resort.
// ---------------------------------------------------------------------------------------------

/// How a shuffle runs: instruction sequence `kind` applied to the operand pair `(x, y)` chosen by
/// `src` (one of `AA`, `AB`, `BA`, `BB`), with up to two immediates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    kind: u8,
    src: u8,
    imm0: u8,
    imm1: u8,
}

// Operand pairs `(x, y)`.
const AA: u8 = 0;
const AB: u8 = 1;
const BA: u8 = 2;
const BB: u8 = 3;

// Plan kinds (instruction counts in brackets).
/// `x` [0].
const K_X: u8 = 0;
/// `pshufd x, imm0` [1].
const K_PSHUFD: u8 = 1;
/// `pshuflw x, imm0` [1].
const K_PSHUFLW: u8 = 2;
/// `pshufhw x, imm0` [1].
const K_PSHUFHW: u8 = 3;
/// `shufps x, y, imm0` [1].
const K_SHUFPS: u8 = 4;
/// `blendps x, y, imm0` [1].
const K_BLENDPS: u8 = 5;
/// `pblendw x, y, imm0` [1].
const K_PBLENDW: u8 = 6;
/// `pblendvb x, y, mask` [1].
const K_PBLENDVB: u8 = 7;
/// `insertps x, y, imm0` [1].
const K_INSERTPS: u8 = 8;
/// `palignr`: bytes `imm0..imm0 + 16` of `x ++ y` [1].
const K_PALIGNR: u8 = 9;
/// `punpck{l,h}{bw,wd,dq} x, y`, which one by `imm0` (`U_*`) [1].
const K_UNPACK: u8 = 10;
/// `pshufb x, mask` (one source) [1].
const K_PSHUFB: u8 = 11;
/// `insertps (pshufd x, imm0), y, imm1` [2].
const K_PSHUFD_INSERTPS: u8 = 12;
/// `pshufd (shufps x, y, imm0), imm1` [2].
const K_SHUFPS_PSHUFD: u8 = 13;
/// `palignr` of `x ++ (y >> 8 * imm1)` by `imm0` bytes [2].
const K_PSRLDQ_PALIGNR: u8 = 14;
/// Even bytes of `x` then of `y`: `packuswb (x & 0xFF), (y & 0xFF)` per word [3].
const K_EVEN8: u8 = 15;
/// Odd bytes of `x` then of `y`: `packuswb (x >> 8), (y >> 8)` per word [3].
const K_ODD8: u8 = 16;
/// `pshufb a, mask_a | pshufb b, mask_b` (generic two-source fallback) [3].
const K_PSHUFB_OR: u8 = 17;
/// `plan16` found nothing better than the byte-level plans.
const K_NONE: u8 = 0xFF;

// `K_UNPACK` variants.
const U_LO8: u8 = 0;
const U_HI8: u8 = 1;
const U_LO16: u8 = 2;
const U_HI16: u8 = 3;
const U_LO32: u8 = 4;
const U_HI32: u8 = 5;

const fn mk(kind: u8, src: u8, imm0: u8, imm1: u8) -> Plan {
    Plan {
        kind,
        src,
        imm0,
        imm1,
    }
}

/// Instructions a plan executes (constants loaded from memory not counted).
#[cfg(test)]
const fn plan_cost(p: Plan) -> u32 {
    match p.kind {
        K_X => 0,
        K_PSHUFD_INSERTPS | K_SHUFPS_PSHUFD | K_PSRLDQ_PALIGNR => 2,
        K_EVEN8 | K_ODD8 | K_PSHUFB_OR => 3,
        _ => 1,
    }
}

/// Byte pattern of an `N`-lane shuffle (lanes of `16 / N` bytes) with lane indices `idx`.
const fn shuffle_bytes<const N: usize>(idx: [usize; N]) -> [u8; 16] {
    let w = 16 / N;
    let mut out = [0u8; 16];
    let mut j = 0;
    while j < 16 {
        out[j] = (idx[j / w] * w + j % w) as u8;
        j += 1;
    }
    out
}

/// Whether the pattern moves whole aligned `w`-byte lanes.
const fn lane_aligned(b: &[u8; 16], w: usize) -> bool {
    let mut j = 0;
    while j < 16 {
        let first = b[j - j % w] as usize;
        if first % w != 0 || b[j] as usize != first + j % w {
            return false;
        }
        j += 1;
    }
    true
}

/// Lane indices (in `a ++ b`, `N` lanes per operand) of a `lane_aligned` pattern.
const fn lane_indices<const N: usize>(b: &[u8; 16]) -> [u8; N] {
    let w = 16 / N;
    let mut out = [0u8; N];
    let mut k = 0;
    while k < N {
        out[k] = b[k * w] / w as u8;
        k += 1;
    }
    out
}

/// `pshufd` / `shufps` / `pshuflw` immediate: two bits per lane.
const fn imm2x4(l0: u8, l1: u8, l2: u8, l3: u8) -> u8 {
    (l0 & 3) | (l1 & 3) << 2 | (l2 & 3) << 4 | (l3 & 3) << 6
}

const fn plan(b: [u8; 16]) -> Plan {
    if lane_aligned(&b, 4) {
        return plan32(lane_indices::<4>(&b));
    }
    let mut na = 0;
    let mut j = 0;
    while j < 16 {
        if b[j] < 16 {
            na += 1;
        }
        j += 1;
    }
    // Rotations: one `palignr`, of `x ++ y` or of one operand with itself.
    if na == 16 || na == 0 {
        let off = if na == 16 { 0 } else { 16 };
        let n = b[0] - off;
        let mut rot = true;
        j = 0;
        while j < 16 {
            if b[j] - off != (n + j as u8) % 16 {
                rot = false;
            }
            j += 1;
        }
        if rot {
            return mk(K_PALIGNR, if na == 16 { AA } else { BB }, n, 0);
        }
    } else {
        let k = b[0];
        let mut rot = true;
        j = 0;
        while j < 16 {
            if b[j] != (k + j as u8) % 32 {
                rot = false;
            }
            j += 1;
        }
        if rot {
            return if k < 16 {
                mk(K_PALIGNR, AB, k, 0)
            } else {
                mk(K_PALIGNR, BA, k - 16, 0)
            };
        }
    }
    if lane_aligned(&b, 2) {
        let p = plan16(lane_indices::<8>(&b));
        if p.kind != K_NONE {
            return p;
        }
    }
    plan8(&b, na)
}

/// Whole 32-bit lanes `d` (indices into `a ++ b`, 0..8): at most two instructions.
const fn plan32(d: [u8; 4]) -> Plan {
    let mut na = 0;
    let mut j = 0;
    while j < 4 {
        if d[j] < 4 {
            na += 1;
        }
        j += 1;
    }
    let imm = imm2x4(d[0], d[1], d[2], d[3]);
    if na == 4 || na == 0 {
        let src = if na == 4 { AA } else { BB };
        return if imm == imm2x4(0, 1, 2, 3) {
            mk(K_X, src, 0, 0)
        } else {
            mk(K_PSHUFD, src, imm, 0)
        };
    }
    // Every lane stays in place: `blendps`.
    let mut in_place = true;
    let mut from_b = 0u8;
    j = 0;
    while j < 4 {
        if d[j] & 3 != j as u8 {
            in_place = false;
        }
        if d[j] >= 4 {
            from_b |= 1 << j;
        }
        j += 1;
    }
    if in_place {
        return mk(K_BLENDPS, AB, from_b, 0);
    }
    // Interleaves.
    let pat = (d[0] as u32) | (d[1] as u32) << 8 | (d[2] as u32) << 16 | (d[3] as u32) << 24;
    if pat == 0x0501_0400 {
        return mk(K_UNPACK, AB, U_LO32, 0); // 0, 4, 1, 5
    }
    if pat == 0x0105_0004 {
        return mk(K_UNPACK, BA, U_LO32, 0); // 4, 0, 5, 1
    }
    if pat == 0x0703_0602 {
        return mk(K_UNPACK, AB, U_HI32, 0); // 2, 6, 3, 7
    }
    if pat == 0x0307_0206 {
        return mk(K_UNPACK, BA, U_HI32, 0); // 6, 2, 7, 3
    }
    // Rotation of `a ++ b` or `b ++ a`: `palignr`.
    let k = d[0];
    if d[1] == (k + 1) % 8 && d[2] == (k + 2) % 8 && d[3] == (k + 3) % 8 {
        return if k < 4 {
            mk(K_PALIGNR, AB, 4 * k, 0)
        } else {
            mk(K_PALIGNR, BA, 4 * (k - 4), 0)
        };
    }
    // Two lanes of one operand, then two of the other: `shufps`.
    if d[0] < 4 && d[1] < 4 && d[2] >= 4 && d[3] >= 4 {
        return mk(K_SHUFPS, AB, imm, 0);
    }
    if d[0] >= 4 && d[1] >= 4 && d[2] < 4 && d[3] < 4 {
        return mk(K_SHUFPS, BA, imm, 0);
    }
    // Three lanes from `x`, one from `y`: `insertps`, after a `pshufd` unless `x`'s lanes are in
    // place.
    if na == 3 || na == 1 {
        let x_is_a = na == 3;
        let mut perm = [0u8; 4];
        let mut ins = 0u8;
        j = 0;
        while j < 4 {
            if (d[j] < 4) == x_is_a {
                perm[j] = d[j] & 3;
            } else {
                perm[j] = j as u8;
                ins = (d[j] & 3) << 6 | (j as u8) << 4;
            }
            j += 1;
        }
        let src = if x_is_a { AB } else { BA };
        let p = imm2x4(perm[0], perm[1], perm[2], perm[3]);
        return if p == imm2x4(0, 1, 2, 3) {
            mk(K_INSERTPS, src, ins, 0)
        } else {
            mk(K_PSHUFD_INSERTPS, src, p, ins)
        };
    }
    // Two lanes from each: `shufps` gathers them (`a`'s first), `pshufd` puts them in order.
    let mut sel = [0u8; 4];
    let mut perm = [0u8; 4];
    let mut ia = 0;
    let mut ib = 2;
    j = 0;
    while j < 4 {
        if d[j] < 4 {
            sel[ia] = d[j];
            perm[j] = ia as u8;
            ia += 1;
        } else {
            sel[ib] = d[j];
            perm[j] = ib as u8;
            ib += 1;
        }
        j += 1;
    }
    mk(
        K_SHUFPS_PSHUFD,
        AB,
        imm2x4(sel[0], sel[1], sel[2], sel[3]),
        imm2x4(perm[0], perm[1], perm[2], perm[3]),
    )
}

/// Whole 16-bit lanes `w` (indices into `a ++ b`, 0..16), not all 32-bit aligned: one
/// instruction, or `K_NONE`.
const fn plan16(w: [u8; 8]) -> Plan {
    let mut na = 0;
    let mut j = 0;
    while j < 8 {
        if w[j] < 8 {
            na += 1;
        }
        j += 1;
    }
    if na == 8 || na == 0 {
        let src = if na == 8 { AA } else { BB };
        let mut r = [0u8; 8];
        let (mut low_from_low, mut high_from_high) = (true, true);
        j = 0;
        while j < 8 {
            r[j] = w[j] & 7;
            if j < 4 && r[j] >= 4 {
                low_from_low = false;
            }
            if j >= 4 && r[j] < 4 {
                high_from_high = false;
            }
            j += 1;
        }
        let low_id = r[0] == 0 && r[1] == 1 && r[2] == 2 && r[3] == 3;
        let high_id = r[4] == 4 && r[5] == 5 && r[6] == 6 && r[7] == 7;
        if high_id && low_from_low {
            return mk(K_PSHUFLW, src, imm2x4(r[0], r[1], r[2], r[3]), 0);
        }
        if low_id && high_from_high {
            return mk(K_PSHUFHW, src, imm2x4(r[4], r[5], r[6], r[7]), 0);
        }
        let (mut lo, mut hi) = (true, true);
        j = 0;
        while j < 8 {
            if r[j] != j as u8 / 2 {
                lo = false;
            }
            if r[j] != 4 + j as u8 / 2 {
                hi = false;
            }
            j += 1;
        }
        if lo {
            return mk(K_UNPACK, src, U_LO16, 0);
        }
        if hi {
            return mk(K_UNPACK, src, U_HI16, 0);
        }
        return mk(K_NONE, 0, 0, 0);
    }
    // Every lane stays in place: `pblendw`.
    let mut in_place = true;
    let mut from_b = 0u8;
    j = 0;
    while j < 8 {
        if w[j] & 7 != j as u8 {
            in_place = false;
        }
        if w[j] >= 8 {
            from_b |= 1 << j;
        }
        j += 1;
    }
    if in_place {
        return mk(K_PBLENDW, AB, from_b, 0);
    }
    // Interleaves of `(a, b)` or `(b, a)`, low or high halves.
    let mut v = 0;
    while v < 4 {
        let (src, xo, yo) = if v < 2 { (AB, 0, 8) } else { (BA, 8, 0) };
        let half = if v % 2 == 0 { 0 } else { 4 };
        let mut ok = true;
        let mut t = 0;
        while t < 4 {
            if w[2 * t] != xo + half + t as u8 || w[2 * t + 1] != yo + half + t as u8 {
                ok = false;
            }
            t += 1;
        }
        if ok {
            return mk(K_UNPACK, src, if half == 0 { U_LO16 } else { U_HI16 }, 0);
        }
        v += 1;
    }
    mk(K_NONE, 0, 0, 0)
}

/// Byte patterns: one to three instructions. `na` is the number of bytes taken from `a`.
const fn plan8(b: &[u8; 16], na: u32) -> Plan {
    // Interleaves of any operand pair, low or high halves.
    let mut v = 0;
    while v < 8 {
        let src = (v / 2) as u8;
        let xo = if src == BA || src == BB { 16 } else { 0 };
        let yo = if src == AB || src == BB { 16 } else { 0 };
        let half = if v % 2 == 0 { 0 } else { 8 };
        let mut ok = true;
        let mut t = 0;
        while t < 8 {
            if b[2 * t] != xo + half + t as u8 || b[2 * t + 1] != yo + half + t as u8 {
                ok = false;
            }
            t += 1;
        }
        if ok {
            return mk(K_UNPACK, src, if half == 0 { U_LO8 } else { U_HI8 }, 0);
        }
        v += 1;
    }
    if na == 16 || na == 0 {
        return mk(K_PSHUFB, if na == 16 { AA } else { BB }, 0, 0);
    }
    // Every byte in place: `pblendvb`.
    let mut in_place = true;
    let mut j = 0;
    while j < 16 {
        if b[j] % 16 != j as u8 {
            in_place = false;
        }
        j += 1;
    }
    if in_place {
        return mk(K_PBLENDVB, AB, 0, 0);
    }
    // Even or odd bytes of `x`, then of `y`: `packuswb` after masking or shifting the words.
    let mut v = 0;
    while v < 4 {
        let (src, xo, yo) = if v < 2 { (AB, 0, 16) } else { (BA, 16, 0) };
        let odd = (v % 2) as u8;
        let mut ok = true;
        j = 0;
        while j < 16 {
            let want = if j < 8 {
                xo + 2 * j as u8
            } else {
                yo + 2 * (j as u8 - 8)
            };
            if b[j] != want + odd {
                ok = false;
            }
            j += 1;
        }
        if ok {
            return mk(if odd == 0 { K_EVEN8 } else { K_ODD8 }, src, 0, 0);
        }
        v += 1;
    }
    // The top `16 - k` bytes of `x` followed by `k` consecutive bytes of `y` from byte `m`:
    // `palignr` by `k` of `x ++ (y >> 8m)`.
    let mut v = 0;
    while v < 2 {
        let (src, xo, yo) = if v == 0 { (AB, 0u8, 16u8) } else { (BA, 16, 0) };
        v += 1;
        if b[0] < xo || b[0] >= xo + 16 {
            continue;
        }
        let k = b[0] - xo;
        if k == 0 {
            continue;
        }
        let split = 16 - k as usize;
        let mut ok = b[split] >= yo && b[split] < yo + 16;
        let m = b[split].wrapping_sub(yo);
        if m == 0 || m as usize + k as usize > 16 {
            ok = false;
        }
        j = 0;
        while j < 16 {
            let want = if j < split {
                xo + k + j as u8
            } else {
                yo.wrapping_add(m).wrapping_add((j - split) as u8)
            };
            if b[j] != want {
                ok = false;
            }
            j += 1;
        }
        if ok {
            return mk(K_PSRLDQ_PALIGNR, src, k, m);
        }
    }
    mk(K_PSHUFB_OR, AB, 0, 0)
}

/// `pshufb` mask for a one-source pattern.
const fn mask_one(b: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut j = 0;
    while j < 16 {
        out[j] = b[j] & 15;
        j += 1;
    }
    out
}

/// `pshufb` mask taking the bytes that come from `a` (`from_b == false`) or `b`, zero elsewhere.
const fn mask_part(b: [u8; 16], from_b: bool) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut j = 0;
    while j < 16 {
        out[j] = if (b[j] >= 16) == from_b {
            b[j] & 15
        } else {
            0x80
        };
        j += 1;
    }
    out
}

/// `pblendvb` mask: top bit set where the byte comes from `b`.
const fn mask_blend(b: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut j = 0;
    while j < 16 {
        out[j] = if b[j] >= 16 { 0x80 } else { 0 };
        j += 1;
    }
    out
}

const fn m128(b: [u8; 16]) -> __m128i {
    // SAFETY: `[u8; 16]` and `__m128i` are both 16 bytes of plain data.
    unsafe { core::mem::transmute::<[u8; 16], __m128i>(b) }
}

/// `match $imm` over every `u8`, calling `$f::<imm>$args` (immediates must be constants).
macro_rules! imm8 {
    (@ $imm:expr, $f:ident $args:tt; $($n:literal)*) => {
        match $imm { $($n => $f::<$n> $args,)* }
    };
    ($imm:expr, $f:ident $args:tt) => {
        imm8!(@ $imm, $f $args;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60
            61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89
            90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113
            114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
            136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157
            158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
            180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201
            202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
            224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245
            246 247 248 249 250 251 252 253 254 255)
    };
}

/// `match $imm` over `0..16`, calling `$f::<imm>$args`.
macro_rules! imm4 {
    (@ $imm:expr, $f:ident $args:tt; $($n:literal)*) => {
        match $imm { $($n => $f::<$n> $args,)* _ => unreachable!(), }
    };
    ($imm:expr, $f:ident $args:tt) => {
        imm4!(@ $imm, $f $args; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    };
}

/// Executes a plan. In the shuffle functions every plan field and mask is a `const` block, so
/// each `match` below is on a constant and only the planned instructions are emitted; the tests
/// run the same arms with runtime plans.
macro_rules! run_plan {
    ($a:expr, $b:expr, kind: $kind:expr, src: $src:expr, imm0: $imm0:expr, imm1: $imm1:expr,
     mask_one: $m1:expr, mask_a: $ma:expr, mask_b: $mb:expr, mask_blend: $mbl:expr $(,)?) => {{
        let a: __m128i = $a;
        let b: __m128i = $b;
        let (x, y) = match $src {
            AA => (a, a),
            AB => (a, b),
            BA => (b, a),
            _ => (b, b),
        };
        match $kind {
            K_X => x,
            K_PSHUFD => imm8!($imm0, pshufd(x)),
            K_PSHUFLW => imm8!($imm0, pshuflw(x)),
            K_PSHUFHW => imm8!($imm0, pshufhw(x)),
            K_SHUFPS => imm8!($imm0, shufps(x, y)),
            K_BLENDPS => imm4!($imm0, blendps(x, y)),
            K_PBLENDW => imm8!($imm0, pblendw(x, y)),
            K_PBLENDVB => pblendvb(x, y, $mbl),
            K_INSERTPS => imm8!($imm0, insertps(x, y)),
            K_PALIGNR => imm4!($imm0, palignr(x, y)),
            // SAFETY (the `unsafe` blocks in these arms): SSE2 intrinsics; see "Safety" in the
            // module docs.
            K_UNPACK => unsafe {
                match $imm0 {
                    U_LO8 => _mm_unpacklo_epi8(x, y),
                    U_HI8 => _mm_unpackhi_epi8(x, y),
                    U_LO16 => _mm_unpacklo_epi16(x, y),
                    U_HI16 => _mm_unpackhi_epi16(x, y),
                    U_LO32 => _mm_unpacklo_epi32(x, y),
                    _ => _mm_unpackhi_epi32(x, y),
                }
            },
            K_PSHUFB => pshufb(x, $m1),
            K_PSHUFD_INSERTPS => {
                let t = imm8!($imm0, pshufd(x));
                imm8!($imm1, insertps(t, y))
            }
            K_SHUFPS_PSHUFD => {
                let t = imm8!($imm0, shufps(x, y));
                imm8!($imm1, pshufd(t))
            }
            K_PSRLDQ_PALIGNR => {
                let t = imm4!($imm1, psrldq(y));
                imm4!($imm0, palignr(x, t))
            }
            K_EVEN8 => unsafe {
                let low_bytes = _mm_set1_epi16(0x00FF);
                _mm_packus_epi16(_mm_and_si128(x, low_bytes), _mm_and_si128(y, low_bytes))
            },
            K_ODD8 => unsafe { _mm_packus_epi16(_mm_srli_epi16::<8>(x), _mm_srli_epi16::<8>(y)) },
            K_PSHUFB_OR => {
                let (ma, mb): (__m128i, __m128i) = ($ma, $mb);
                unsafe { _mm_or_si128(pshufb(a, ma), pshufb(b, mb)) }
            }
            _ => unreachable!(),
        }
    }};
}

/// A shuffle with the byte pattern `$bytes` (a constant expression of the const generics).
macro_rules! shuffle {
    ($a:expr, $b:expr, $bytes:expr) => {
        v128(run_plan!(
            $a.0,
            $b.0,
            // (Parenthesized: an edition-2021 `expr` fragment cannot start with `const`.)
            kind: (const { plan($bytes).kind }),
            src: (const { plan($bytes).src }),
            imm0: (const { plan($bytes).imm0 }),
            imm1: (const { plan($bytes).imm1 }),
            mask_one: (const { m128(mask_one($bytes)) }),
            mask_a: (const { m128(mask_part($bytes, false)) }),
            mask_b: (const { m128(mask_part($bytes, true)) }),
            mask_blend: (const { m128(mask_blend($bytes)) }),
        ))
    };
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn i8x16_shuffle<
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
    const I4: usize,
    const I5: usize,
    const I6: usize,
    const I7: usize,
    const I8: usize,
    const I9: usize,
    const I10: usize,
    const I11: usize,
    const I12: usize,
    const I13: usize,
    const I14: usize,
    const I15: usize,
>(
    a: v128,
    b: v128,
) -> v128 {
    const {
        assert!(I0 < 32 && I1 < 32 && I2 < 32 && I3 < 32 && I4 < 32 && I5 < 32 && I6 < 32);
        assert!(I7 < 32 && I8 < 32 && I9 < 32 && I10 < 32 && I11 < 32 && I12 < 32);
        assert!(I13 < 32 && I14 < 32 && I15 < 32);
    };
    shuffle!(
        a,
        b,
        shuffle_bytes([I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15])
    )
}

#[inline(always)]
pub fn i16x8_shuffle<
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
    const I4: usize,
    const I5: usize,
    const I6: usize,
    const I7: usize,
>(
    a: v128,
    b: v128,
) -> v128 {
    const {
        assert!(I0 < 16 && I1 < 16 && I2 < 16 && I3 < 16);
        assert!(I4 < 16 && I5 < 16 && I6 < 16 && I7 < 16);
    };
    shuffle!(a, b, shuffle_bytes([I0, I1, I2, I3, I4, I5, I6, I7]))
}

#[inline(always)]
pub fn i32x4_shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
    a: v128,
    b: v128,
) -> v128 {
    const { assert!(I0 < 8 && I1 < 8 && I2 < 8 && I3 < 8) };
    shuffle!(a, b, shuffle_bytes([I0, I1, I2, I3]))
}

#[inline(always)]
pub fn i64x2_shuffle<const I0: usize, const I1: usize>(a: v128, b: v128) -> v128 {
    const { assert!(I0 < 4 && I1 < 4) };
    shuffle!(a, b, shuffle_bytes([I0, I1]))
}

// ---------------------------------------------------------------------------------------------
// Tests of the shuffle planner (host only). The conformance suite covers every op and every
// shuffle instantiation the kernel uses; these cover the planner for every other pattern by
// running the same `run_plan!` arms with runtime plans.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::simd::conformance::Rng;
    use std::collections::BTreeSet;
    use std::vec::Vec;

    fn run(bytes: [u8; 16], a: __m128i, b: __m128i) -> __m128i {
        let p = plan(bytes);
        run_plan!(
            a,
            b,
            kind: p.kind,
            src: p.src,
            imm0: p.imm0,
            imm1: p.imm1,
            mask_one: m128(mask_one(bytes)),
            mask_a: m128(mask_part(bytes, false)),
            mask_b: m128(mask_part(bytes, true)),
            mask_blend: m128(mask_blend(bytes)),
        )
    }

    fn to_bytes(v: __m128i) -> [u8; 16] {
        // SAFETY: 16 bytes of plain data either way.
        unsafe { core::mem::transmute(v) }
    }

    /// Runs `bytes` on three input pairs (distinct bytes, sign and saturation boundaries, random)
    /// and compares with the byte-by-byte definition. Returns the plan.
    fn check(bytes: [u8; 16], rng: &mut Rng) -> Plan {
        let mut inputs = [[[0u8; 16]; 2]; 3];
        for i in 0..16 {
            inputs[0][0][i] = i as u8;
            inputs[0][1][i] = 0x10 | i as u8;
            inputs[1][0][i] = [0x00, 0x7F, 0x80, 0xFF][i % 4];
            inputs[1][1][i] = [0xFF, 0x80, 0x7F, 0x01][i % 4];
        }
        for half in 0..2 {
            let r = (rng.next() as u128) << 64 | rng.next() as u128;
            inputs[2][half] = r.to_le_bytes();
        }
        for [a, b] in inputs {
            let want: [u8; 16] = core::array::from_fn(|j| {
                if bytes[j] < 16 {
                    a[bytes[j] as usize]
                } else {
                    b[bytes[j] as usize - 16]
                }
            });
            let got = to_bytes(run(bytes, m128(a), m128(b)));
            assert_eq!(got, want, "pattern {bytes:?} plan {:?}", plan(bytes));
        }
        plan(bytes)
    }

    fn random_lanes<const N: usize>(rng: &mut Rng, range: u64) -> [usize; N] {
        core::array::from_fn(|_| (rng.next() % range) as usize)
    }

    #[test]
    fn every_32_and_64_bit_shuffle_matches_and_takes_at_most_two_instructions() {
        let mut rng = Rng::new(32);
        for p in 0..8usize.pow(4) {
            let idx = [p % 8, p / 8 % 8, p / 64 % 8, p / 512];
            let plan = check(shuffle_bytes(idx), &mut rng);
            assert!(plan_cost(plan) <= 2, "{idx:?}: {plan:?}");
            assert!(
                plan.kind != K_PSHUFB && plan.kind != K_PSHUFB_OR,
                "{idx:?}: {plan:?}"
            );
        }
        for p in 0..16usize {
            check(shuffle_bytes([p % 4, p / 4]), &mut rng);
        }
    }

    #[test]
    fn random_and_structured_16_and_8_bit_shuffles_match() {
        let mut rng = Rng::new(16);
        let mut kinds = BTreeSet::new();
        let mut patterns: Vec<[u8; 16]> = Vec::new();
        // Miri (x86_64 hosts; it interprets the SSE intrinsics) runs a thinned random set.
        for _ in 0..if cfg!(miri) { 500 } else { 100_000 } {
            patterns.push(shuffle_bytes(random_lanes::<8>(&mut rng, 16)));
            patterns.push(shuffle_bytes(random_lanes::<8>(&mut rng, 8)));
            let from_b = random_lanes::<8>(&mut rng, 8).map(|i| i + 8);
            patterns.push(shuffle_bytes(from_b));
            patterns.push(shuffle_bytes(random_lanes::<16>(&mut rng, 32)));
            patterns.push(shuffle_bytes(random_lanes::<16>(&mut rng, 16)));
            // Half-preserving word permutations (`pshuflw` / `pshufhw`).
            let mut w: [usize; 8] = core::array::from_fn(|i| i);
            let s = (rng.next() % 2) as usize * 4;
            for i in 0..4 {
                w[s + i] = s + (rng.next() % 4) as usize;
            }
            patterns.push(shuffle_bytes(w));
            // Lane-in-place blends.
            let m = rng.next();
            patterns.push(shuffle_bytes::<8>(core::array::from_fn(|i| {
                i + 8 * ((m >> i) & 1) as usize
            })));
            patterns.push(shuffle_bytes::<16>(core::array::from_fn(|i| {
                i + 16 * ((m >> i) & 1) as usize
            })));
        }
        // Rotations, interleaves, even/odd bytes, shifted-in tails, 4x2 transposes.
        for k in 0..32 {
            patterns.push(shuffle_bytes::<16>(core::array::from_fn(|j| (k + j) % 32)));
            patterns.push(shuffle_bytes::<16>(core::array::from_fn(|j| (k + j) % 16)));
            patterns.push(shuffle_bytes::<16>(core::array::from_fn(|j| {
                16 + (k + j) % 16
            })));
            patterns.push(shuffle_bytes::<8>(core::array::from_fn(|j| (k + j) % 16)));
        }
        for (xo, yo) in [(0, 0), (0, 16), (16, 0), (16, 16)] {
            for half in [0, 8] {
                patterns.push(shuffle_bytes::<16>(core::array::from_fn(|j| {
                    if j % 2 == 0 {
                        xo + half + j / 2
                    } else {
                        yo + half + j / 2
                    }
                })));
                patterns.push(shuffle_bytes::<8>(core::array::from_fn(|j| {
                    if j % 2 == 0 {
                        xo / 2 + half / 2 + j / 2
                    } else {
                        yo / 2 + half / 2 + j / 2
                    }
                })));
            }
            for odd in [0, 1] {
                patterns.push(shuffle_bytes::<16>(core::array::from_fn(|j| {
                    (if j < 8 { xo + 2 * j } else { yo + 2 * (j - 8) }) + odd
                })));
            }
        }
        for k in 1..16 {
            for m in 1..=16 - k {
                for (xo, yo) in [(0, 16), (16, 0)] {
                    patterns.push(shuffle_bytes::<16>(core::array::from_fn(|j| {
                        if j < 16 - k {
                            xo + k + j
                        } else {
                            yo + m + j - (16 - k)
                        }
                    })));
                }
            }
        }
        patterns.push(shuffle_bytes([0, 4, 8, 12, 1, 5, 9, 13]));
        patterns.push(shuffle_bytes([2, 6, 10, 14, 3, 7, 11, 15]));
        for p in 0..8usize.pow(4) {
            patterns.push(shuffle_bytes([p % 8, p / 8 % 8, p / 64 % 8, p / 512]));
        }
        for bytes in patterns {
            let p = check(bytes, &mut rng);
            assert!(plan_cost(p) <= 3, "{bytes:?}: {p:?}");
            if lane_aligned(&bytes, 2) && plan_cost(p) == 3 {
                assert!(p.kind == K_PSHUFB_OR, "{bytes:?}: {p:?}");
            }
            kinds.insert(p.kind);
        }
        for k in K_X..=K_PSHUFB_OR {
            assert!(kinds.contains(&k), "no test pattern used plan kind {k}");
        }
    }

    /// The planned sequence for the patterns that matter most in the kernel (by call sites).
    #[test]
    fn kernel_hot_patterns_use_the_intended_instructions() {
        let cases: &[([u8; 16], Plan)] = &[
            (shuffle_bytes([1, 2, 3, 4]), mk(K_PALIGNR, AB, 4, 0)),
            (shuffle_bytes([1, 3, 5, 7]), mk(K_SHUFPS, AB, 0xDD, 0)),
            (shuffle_bytes([0, 2, 4, 6]), mk(K_SHUFPS, AB, 0x88, 0)),
            (shuffle_bytes([2, 3, 4, 5]), mk(K_PALIGNR, AB, 8, 0)),
            (shuffle_bytes([3, 4, 5, 6]), mk(K_PALIGNR, AB, 12, 0)),
            (
                shuffle_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                mk(K_PALIGNR, AB, 1, 0),
            ),
            (shuffle_bytes([0, 4, 1, 5]), mk(K_UNPACK, AB, U_LO32, 0)),
            (shuffle_bytes([2, 6, 3, 7]), mk(K_UNPACK, AB, U_HI32, 0)),
            (shuffle_bytes([0, 1, 6, 7]), mk(K_BLENDPS, AB, 0b1100, 0)),
            (shuffle_bytes([3, 5, 6, 7]), mk(K_INSERTPS, BA, 0xC0, 0)),
            (shuffle_bytes([0, 1, 4, 5]), mk(K_SHUFPS, AB, 0x44, 0)),
            (shuffle_bytes([7, 6, 5, 4]), mk(K_PSHUFD, BB, 0x1B, 0)),
            (
                shuffle_bytes([0, 0, 1, 1, 2, 2, 3, 3]),
                mk(K_UNPACK, AA, U_LO16, 0),
            ),
            (
                shuffle_bytes([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]),
                mk(K_ODD8, AB, 0, 0),
            ),
            (
                shuffle_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]),
                mk(K_PSRLDQ_PALIGNR, AB, 1, 1),
            ),
            (shuffle_bytes([1, 3]), mk(K_SHUFPS, AB, 0xEE, 0)),
        ];
        for (bytes, want) in cases {
            assert_eq!(plan(*bytes), *want, "{bytes:?}");
        }
    }
}
