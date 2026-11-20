//! ARM NEON backend for the SIMD layer (`aarch64`, `core::arch::aarch64` intrinsics only).
//!
//! Every function has the `core::arch::wasm32` name and signature and computes exactly what the
//! WebAssembly SIMD specification defines, as spelled out lane by lane in `simd::scalar` (the
//! oracle this backend is checked against by the conformance suite), NaN payloads excepted where
//! wasm leaves them open. Every wrapper is `#[inline(always)]` and lowers to the single natural
//! NEON instruction wherever one exists; the exceptions are listed under "Multi-instruction ops".
//!
//! # Representation
//!
//! `v128` is a `#[repr(transparent)]` newtype over `float32x4_t`: a real NEON register type, so
//! values live in `v` registers and every reinterpretation between lane types
//! (`vreinterpretq_*`) is free. It is 16 bytes, 16-byte aligned and `Copy` (asserted below), and
//! a distinct type, like `core::arch::wasm32::v128`, so kernel code cannot mix it with raw NEON
//! types by accident. `f32` lanes are the inner type because they dominate the kernel. Lane `i`
//! of every lane type occupies the same bytes as in wasm (little-endian, lane 0 at the lowest
//! address), so loads, stores, lane numbers and shuffle indices carry over unchanged.
//!
//! # Floating-point environment
//!
//! Results match wasm under the default AArch64 FPCR, all control bits clear, which is the state
//! a process starts in on macOS and Linux: round to nearest even, no flush-to-zero (`FZ = 0`),
//! NaN propagation rather than default NaN (`DN = 0`) and the standard (not x86-alternative,
//! `AH = 0`) handling of `FMIN`/`FMAX`/`FABS`/`FNEG`. Code that changes FPCR breaks the match.
//! Under that environment the NEON instructions used here are exactly the wasm operations:
//! `FMIN`/`FMAX` propagate NaN and order -0 below +0 (`f32x4.min`/`max`), `FRINTN`/`FRINTM` are
//! `nearest` (ties to even) and `floor`, `FCVTZS` is `trunc_sat` (saturating, NaN to 0),
//! `UCVTF`/`SCVTF` round to nearest even, `SQRDMULH` is `q15mulr_sat`, and `FABS`/`FNEG` only
//! touch the sign bit (NaN payloads preserved). LLVM never contracts the separate `fmul`/`fadd`
//! intrinsics into `fmla` without fast-math flags, which Rust never sets.
//!
//! # Shifts
//!
//! Shift counts are reduced modulo the lane width (as wasm requires) and applied with
//! `USHL`/`SSHL` by a splatted count, negated for right shifts. When the count is a constant,
//! which it is at every kernel call site, LLVM folds the whole sequence into one immediate
//! `SHL`/`USHR`/`SSHR` (or nothing for a count of 0 mod width). A count only known at run time
//! costs `and` + (`neg`) + `dup` + `sshl`/`ushl`.
//!
//! # Shuffles
//!
//! A const-generic shuffle is planned at compile time by `plan`, a `const fn` over its byte
//! mask, and dispatched on associated constants of the `Shuffle` trait, so each instantiation
//! compiles to the one arm it selects (rustc prunes the others even without optimisation). The
//! planner tries every single-instruction NEON permutation, in this order: the identity (no
//! instruction), `DUP` (element), `EXT`, `ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2` at 64/32/16/8-bit
//! elements, `REV64/REV32/REV16`, and `INS` (one element copied from either operand), each with
//! every operand order including one operand used twice. A pattern none of them produces falls
//! back to one `TBL` (one table register when every index selects the same operand, two
//! otherwise) with the indices as a constant vector. LLVM turns a `TBL` with constant indices
//! back into a generic shuffle and lowers that with its own cost model: for 4 x 32-bit patterns
//! that is two permutes from its perfect-shuffle tables (three for `<1, 2, 5, 6>`), otherwise it
//! keeps the `TBL`, whose constant is hoisted out of loops.
//!
//! # Multi-instruction ops
//!
//! Every op not listed here is at most one instruction (lane-0 extracts and identity shuffles
//! are none). Listed ops need more because NEON has no single instruction with the wasm
//! semantics:
//!
//! - `f32x4_pmin` / `f32x4_pmax`: `fcmgt` + `bsl`/`bit`/`bif`. Pseudo-min/max return one input's
//!   bits unchanged (a NaN in `b` loses, `+0`/`-0` ties keep `a`); `fmin`/`fminnm` disagree on
//!   NaN and signed zeros, so it takes a compare plus a bit select.
//! - `i32x4_ne`: `cmeq` + `mvn`. NEON has no "compare not equal"; LLVM folds the `mvn` into a
//!   following `and`/`bsl` (as `bic`/`bif`) when there is one.
//! - `i8x16_narrow_i16x8`, `u8x16_narrow_i16x8`, `i16x8_narrow_i32x4`: `sqxtn`/`sqxtun` +
//!   `sqxtn2`/`sqxtun2`. A saturating narrow reads one 128-bit source and writes half a register,
//!   so two sources take two.
//! - `i32x4_dot_i16x8`: `smull` + `smull2` + `addp`. NEON has no 16-bit dot product (`SDOT` is
//!   8-bit); the two widening multiplies produce the eight 32-bit products and one pairwise add
//!   sums adjacent pairs.
//! - `i8x16_bitmask`: `cmlt #0` + `and` (lane weights) + 3 x `addp` + `umov`. There is no
//!   movemask: isolate and weight the sign bits, fold sixteen weighted bytes into two with
//!   pairwise adds, move to a general register. The weight vector is a constant (literal-pool
//!   load) that LLVM hoists out of loops.
//! - `i32x4_bitmask`: `cmlt #0` + `and` (lane weights) + `addv` + `fmov`. The same for four
//!   lanes, with one across-lane add.
//! - `v128_any_true`: `umaxp` + `fmov` + `cmp`/`cbnz`. A pairwise max folds 128 bits into 64
//!   (non-zero iff any lane is), which is moved to a general register and tested.
//! - `i32x4_all_true`: `uminv` + `fmov` + `cmp`/`cbnz`. The across-lane unsigned minimum is
//!   non-zero iff every lane is.
//! - Shifts by a run-time count: `and` + (`neg`) + `dup` + `ushl`/`sshl`. NEON register shifts
//!   do not reduce the count modulo the lane width and shift right only by a negative count.
//!   Constant counts, as at every kernel call site, fold to one instruction (see "Shifts").
//! - Shuffles with no single-instruction form: one `TBL` plus its constant indices, or the pair
//!   of permutes LLVM prefers (see "Shuffles").
//! - `i32x4_splat` / `u32x4_splat` of a run-time value that is stored or used as `f32`:
//!   `fmov s, w` + `dup v.4s, v.s[0]`. LLVM rewrites a 32-bit integer splat consumed in the
//!   inner `f32` type into a splat of the reinterpreted scalar; feeding an integer op it stays
//!   one `dup v.4s, w`, and a literal splat is a materialised constant.
//!
//! Constructors (`f32x4(..)` and friends) with constant arguments are one `movi` or literal-pool
//! load (hoisted out of loops); with run-time arguments they cost one `ins` per lane after the
//! first. The bit-select instructions (`bsl`/`bit`/`bif`) overwrite one input, so LLVM adds a
//! register copy when all three inputs stay live, and `ins`-based shuffles likewise when the
//! overwritten operand is still needed.
//!
//! # Safety
//!
//! The NEON intrinsics are `unsafe fn` only because they carry `#[target_feature(enable =
//! "neon")]`. The wrappers cannot carry that attribute themselves: they must stay safe to call
//! from ordinary kernel code, and rustc rejects it together with `#[inline(always)]`. This module
//! is compiled only when `neon` is enabled for the whole crate (see `simd/mod.rs`), which is
//! exactly what those intrinsics require, so the `unsafe` blocks that just call intrinsics are
//! sound. The load and store ops are `unsafe fn`, as in
//! `core::arch::wasm32`, and additionally rely on their documented pointer contracts.

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;

#[cfg(target_endian = "big")]
compile_error!(
    "simd::neon maps wasm's little-endian lanes onto NEON lanes and needs a little-endian target"
);

/// 128-bit SIMD register: a NEON `q` register, lanes in wasm order (see the module docs).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct v128(float32x4_t);

const _: () = assert!(core::mem::size_of::<v128>() == 16 && core::mem::align_of::<v128>() == 16);

// ---------------------------------------------------------------------------------------------
// Lane-type views (free: `vreinterpretq_*` emit no instructions).
// ---------------------------------------------------------------------------------------------

macro_rules! views {
    ($($get:ident, $put:ident, $t:ty, $to:ident, $from:ident;)*) => {
        $(
            #[inline(always)]
            fn $get(v: v128) -> $t {
                // SAFETY: NEON intrinsic, enabled for this module (see the module docs).
                unsafe { $to(v.0) }
            }

            #[inline(always)]
            fn $put(x: $t) -> v128 {
                // SAFETY: as above.
                v128(unsafe { $from(x) })
            }
        )*
    };
}

views! {
    u8s, from_u8s, uint8x16_t, vreinterpretq_u8_f32, vreinterpretq_f32_u8;
    i8s, from_i8s, int8x16_t, vreinterpretq_s8_f32, vreinterpretq_f32_s8;
    u16s, from_u16s, uint16x8_t, vreinterpretq_u16_f32, vreinterpretq_f32_u16;
    i16s, from_i16s, int16x8_t, vreinterpretq_s16_f32, vreinterpretq_f32_s16;
    u32s, from_u32s, uint32x4_t, vreinterpretq_u32_f32, vreinterpretq_f32_u32;
    i32s, from_i32s, int32x4_t, vreinterpretq_s32_f32, vreinterpretq_f32_s32;
    u64s, from_u64s, uint64x2_t, vreinterpretq_u64_f32, vreinterpretq_f32_u64;
    f64s, from_f64s, float64x2_t, vreinterpretq_f64_f32, vreinterpretq_f32_f64;
}

#[inline(always)]
fn f32s(v: v128) -> float32x4_t {
    v.0
}

#[inline(always)]
fn from_f32s(x: float32x4_t) -> v128 {
    v128(x)
}

/// `lane_match!(n; 0, 1, 2 => 3; |L| body)` evaluates `body` with `L` a `const L: i32` equal to
/// `n`, which must be a constant (a const generic or an associated const): rustc then emits only
/// the matching arm, so a run-time lane number never reaches the generated code. The last
/// literal is the catch-all; callers guarantee the range (a compile-time assertion, or the
/// shuffle planner's construction).
macro_rules! lane_match {
    ($n:expr; $($l:literal),* => $last:literal; |$lane:ident| $body:expr) => {
        match $n {
            $( $l => {
                const $lane: i32 = $l;
                $body
            } )*
            _ => {
                const $lane: i32 = $last;
                $body
            }
        }
    };
}

// ---------------------------------------------------------------------------------------------
// Constructors (`v128.const`).
// ---------------------------------------------------------------------------------------------

// SAFETY (all constructors): a 16-byte array and a NEON vector have the same size, every bit
// pattern is valid for both, and lane `i` of the vector is element `i` of the array on a
// little-endian target (asserted above), which is wasm's lane order.

#[inline(always)]
pub const fn f32x4(a0: f32, a1: f32, a2: f32, a3: f32) -> v128 {
    v128(unsafe { core::mem::transmute::<[f32; 4], float32x4_t>([a0, a1, a2, a3]) })
}

#[inline(always)]
pub const fn f64x2(a0: f64, a1: f64) -> v128 {
    v128(unsafe { core::mem::transmute::<[f64; 2], float32x4_t>([a0, a1]) })
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub const fn i16x8(a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16) -> v128 {
    v128(unsafe { core::mem::transmute::<[i16; 8], float32x4_t>([a0, a1, a2, a3, a4, a5, a6, a7]) })
}

#[inline(always)]
pub const fn i32x4(a0: i32, a1: i32, a2: i32, a3: i32) -> v128 {
    v128(unsafe { core::mem::transmute::<[i32; 4], float32x4_t>([a0, a1, a2, a3]) })
}

#[inline(always)]
pub const fn u32x4(a0: u32, a1: u32, a2: u32, a3: u32) -> v128 {
    v128(unsafe { core::mem::transmute::<[u32; 4], float32x4_t>([a0, a1, a2, a3]) })
}

#[inline(always)]
pub const fn u64x2(a0: u64, a1: u64) -> v128 {
    v128(unsafe { core::mem::transmute::<[u64; 2], float32x4_t>([a0, a1]) })
}

// ---------------------------------------------------------------------------------------------
// Splats (`dup`), lane extraction (`mov`/`umov`, free for lane 0) and replacement (`ins`).
// ---------------------------------------------------------------------------------------------

// SAFETY (every `unsafe` block from here on that only calls NEON intrinsics): see the module docs.

#[inline(always)]
pub fn f32x4_splat(a: f32) -> v128 {
    from_f32s(unsafe { vdupq_n_f32(a) })
}

#[inline(always)]
pub fn f64x2_splat(a: f64) -> v128 {
    from_f64s(unsafe { vdupq_n_f64(a) })
}

#[inline(always)]
pub fn i8x16_splat(a: i8) -> v128 {
    from_i8s(unsafe { vdupq_n_s8(a) })
}

#[inline(always)]
pub fn u8x16_splat(a: u8) -> v128 {
    from_u8s(unsafe { vdupq_n_u8(a) })
}

#[inline(always)]
pub fn i16x8_splat(a: i16) -> v128 {
    from_i16s(unsafe { vdupq_n_s16(a) })
}

#[inline(always)]
pub fn u16x8_splat(a: u16) -> v128 {
    from_u16s(unsafe { vdupq_n_u16(a) })
}

#[inline(always)]
pub fn i32x4_splat(a: i32) -> v128 {
    from_i32s(unsafe { vdupq_n_s32(a) })
}

#[inline(always)]
pub fn u32x4_splat(a: u32) -> v128 {
    from_u32s(unsafe { vdupq_n_u32(a) })
}

#[inline(always)]
pub fn f32x4_extract_lane<const N: usize>(a: v128) -> f32 {
    const { assert!(N < 4) };
    let a = f32s(a);
    unsafe { lane_match!(N; 0, 1, 2 => 3; |L| vgetq_lane_f32::<L>(a)) }
}

#[inline(always)]
pub fn f64x2_extract_lane<const N: usize>(a: v128) -> f64 {
    const { assert!(N < 2) };
    let a = f64s(a);
    unsafe { lane_match!(N; 0 => 1; |L| vgetq_lane_f64::<L>(a)) }
}

#[inline(always)]
pub fn i32x4_extract_lane<const N: usize>(a: v128) -> i32 {
    const { assert!(N < 4) };
    let a = i32s(a);
    unsafe { lane_match!(N; 0, 1, 2 => 3; |L| vgetq_lane_s32::<L>(a)) }
}

#[inline(always)]
pub fn u32x4_extract_lane<const N: usize>(a: v128) -> u32 {
    const { assert!(N < 4) };
    let a = u32s(a);
    unsafe { lane_match!(N; 0, 1, 2 => 3; |L| vgetq_lane_u32::<L>(a)) }
}

#[inline(always)]
pub fn u64x2_extract_lane<const N: usize>(a: v128) -> u64 {
    const { assert!(N < 2) };
    let a = u64s(a);
    unsafe { lane_match!(N; 0 => 1; |L| vgetq_lane_u64::<L>(a)) }
}

#[inline(always)]
pub fn f32x4_replace_lane<const N: usize>(a: v128, val: f32) -> v128 {
    const { assert!(N < 4) };
    let a = f32s(a);
    from_f32s(unsafe { lane_match!(N; 0, 1, 2 => 3; |L| vsetq_lane_f32::<L>(val, a)) })
}

#[inline(always)]
pub fn i32x4_replace_lane<const N: usize>(a: v128, val: i32) -> v128 {
    const { assert!(N < 4) };
    let a = i32s(a);
    from_i32s(unsafe { lane_match!(N; 0, 1, 2 => 3; |L| vsetq_lane_s32::<L>(val, a)) })
}

// ---------------------------------------------------------------------------------------------
// f32x4.
// ---------------------------------------------------------------------------------------------

/// `fabs`: clears the sign bit only (NaN payloads preserved).
#[inline(always)]
pub fn f32x4_abs(a: v128) -> v128 {
    from_f32s(unsafe { vabsq_f32(f32s(a)) })
}

/// `fneg`: flips the sign bit only (NaN payloads preserved).
#[inline(always)]
pub fn f32x4_neg(a: v128) -> v128 {
    from_f32s(unsafe { vnegq_f32(f32s(a)) })
}

#[inline(always)]
pub fn f32x4_add(a: v128, b: v128) -> v128 {
    from_f32s(unsafe { vaddq_f32(f32s(a), f32s(b)) })
}

#[inline(always)]
pub fn f32x4_sub(a: v128, b: v128) -> v128 {
    from_f32s(unsafe { vsubq_f32(f32s(a), f32s(b)) })
}

#[inline(always)]
pub fn f32x4_mul(a: v128, b: v128) -> v128 {
    from_f32s(unsafe { vmulq_f32(f32s(a), f32s(b)) })
}

#[inline(always)]
pub fn f32x4_div(a: v128, b: v128) -> v128 {
    from_f32s(unsafe { vdivq_f32(f32s(a), f32s(b)) })
}

/// `fsqrt`: correctly rounded.
#[inline(always)]
pub fn f32x4_sqrt(a: v128) -> v128 {
    from_f32s(unsafe { vsqrtq_f32(f32s(a)) })
}

/// `frintm`: toward negative infinity; -0 stays -0.
#[inline(always)]
pub fn f32x4_floor(a: v128) -> v128 {
    from_f32s(unsafe { vrndmq_f32(f32s(a)) })
}

/// `frintn`: to nearest, ties to even; a zero result keeps the input's sign.
#[inline(always)]
pub fn f32x4_nearest(a: v128) -> v128 {
    from_f32s(unsafe { vrndnq_f32(f32s(a)) })
}

/// `fmin`: NaN if either input is NaN, and -0 is less than +0 (not `fminnm`).
#[inline(always)]
pub fn f32x4_min(a: v128, b: v128) -> v128 {
    from_f32s(unsafe { vminq_f32(f32s(a), f32s(b)) })
}

/// `fmax`: NaN if either input is NaN, and +0 is greater than -0 (not `fmaxnm`).
#[inline(always)]
pub fn f32x4_max(a: v128, b: v128) -> v128 {
    from_f32s(unsafe { vmaxq_f32(f32s(a), f32s(b)) })
}

/// Pseudo-minimum `b < a ? b : a`: `fcmgt` + bit select, returning one input's bits unchanged.
#[inline(always)]
pub fn f32x4_pmin(a: v128, b: v128) -> v128 {
    let (a, b) = (f32s(a), f32s(b));
    from_f32s(unsafe { vbslq_f32(vcltq_f32(b, a), b, a) })
}

/// Pseudo-maximum `a < b ? b : a`: `fcmgt` + bit select, returning one input's bits unchanged.
#[inline(always)]
pub fn f32x4_pmax(a: v128, b: v128) -> v128 {
    let (a, b) = (f32s(a), f32s(b));
    from_f32s(unsafe { vbslq_f32(vcltq_f32(a, b), b, a) })
}

#[inline(always)]
pub fn f32x4_lt(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcltq_f32(f32s(a), f32s(b)) })
}

#[inline(always)]
pub fn f32x4_le(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcleq_f32(f32s(a), f32s(b)) })
}

#[inline(always)]
pub fn f32x4_gt(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcgtq_f32(f32s(a), f32s(b)) })
}

#[inline(always)]
pub fn f32x4_ge(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcgeq_f32(f32s(a), f32s(b)) })
}

/// `scvtf`: round to nearest, ties to even.
#[inline(always)]
pub fn f32x4_convert_i32x4(a: v128) -> v128 {
    from_f32s(unsafe { vcvtq_f32_s32(i32s(a)) })
}

/// `ucvtf`: round to nearest, ties to even.
#[inline(always)]
pub fn f32x4_convert_u32x4(a: v128) -> v128 {
    from_f32s(unsafe { vcvtq_f32_u32(u32s(a)) })
}

/// `fcvtzs`: toward zero, saturating at the `i32` range, NaN to 0 (wasm `trunc_sat` exactly).
#[inline(always)]
pub fn i32x4_trunc_sat_f32x4(a: v128) -> v128 {
    from_i32s(unsafe { vcvtq_s32_f32(f32s(a)) })
}

// ---------------------------------------------------------------------------------------------
// f64x2.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn f64x2_add(a: v128, b: v128) -> v128 {
    from_f64s(unsafe { vaddq_f64(f64s(a), f64s(b)) })
}

#[inline(always)]
pub fn f64x2_sub(a: v128, b: v128) -> v128 {
    from_f64s(unsafe { vsubq_f64(f64s(a), f64s(b)) })
}

#[inline(always)]
pub fn f64x2_mul(a: v128, b: v128) -> v128 {
    from_f64s(unsafe { vmulq_f64(f64s(a), f64s(b)) })
}

#[inline(always)]
pub fn f64x2_sqrt(a: v128) -> v128 {
    from_f64s(unsafe { vsqrtq_f64(f64s(a)) })
}

// ---------------------------------------------------------------------------------------------
// i8x16 / u8x16.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn i8x16_add(a: v128, b: v128) -> v128 {
    from_i8s(unsafe { vaddq_s8(i8s(a), i8s(b)) })
}

#[inline(always)]
pub fn i8x16_sub(a: v128, b: v128) -> v128 {
    from_i8s(unsafe { vsubq_s8(i8s(a), i8s(b)) })
}

#[inline(always)]
pub fn i8x16_min(a: v128, b: v128) -> v128 {
    from_i8s(unsafe { vminq_s8(i8s(a), i8s(b)) })
}

#[inline(always)]
pub fn i8x16_max(a: v128, b: v128) -> v128 {
    from_i8s(unsafe { vmaxq_s8(i8s(a), i8s(b)) })
}

/// Bit `i` of the result is the sign bit of lane `i`: weight each negative lane by `1 << (i % 8)`
/// and fold the halves with three pairwise adds (NEON has no movemask).
#[inline(always)]
pub fn i8x16_bitmask(a: v128) -> u16 {
    // SAFETY: same size, every bit pattern valid (constant lane weights).
    const WEIGHTS: uint8x16_t = unsafe {
        core::mem::transmute::<[u8; 16], uint8x16_t>([
            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
        ])
    };
    unsafe {
        let w = vandq_u8(vcltzq_s8(i8s(a)), WEIGHTS);
        let s = vpaddq_u8(w, w); // byte k: lanes 2k, 2k+1
        let s = vpaddq_u8(s, s); // byte k: lanes 4k..4k+4
        let s = vpaddq_u8(s, s); // byte 0: lanes 0..8, byte 1: lanes 8..16
        vgetq_lane_u16::<0>(vreinterpretq_u16_u8(s))
    }
}

/// Lanes 0..8 from `a`, 8..16 from `b`, each `i16` saturated to `i8` (`sqxtn` + `sqxtn2`).
#[inline(always)]
pub fn i8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    from_i8s(unsafe { vqmovn_high_s16(vqmovn_s16(i16s(a)), i16s(b)) })
}

#[inline(always)]
pub fn u8x16_add_sat(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vqaddq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn u8x16_sub_sat(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vqsubq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn u8x16_min(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vminq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn u8x16_max(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vmaxq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn u8x16_lt(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vcltq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn u8x16_le(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vcleq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn u8x16_gt(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vcgtq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn u8x16_ge(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vcgeq_u8(u8s(a), u8s(b)) })
}

/// Lanes 0..8 from `a`, 8..16 from `b`, each signed `i16` saturated to `u8` (`sqxtun` +
/// `sqxtun2`; not `uqxtn`, which would read the input as unsigned).
#[inline(always)]
pub fn u8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vqmovun_high_s16(vqmovun_s16(i16s(a)), i16s(b)) })
}

// ---------------------------------------------------------------------------------------------
// i16x8 / u16x8.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn i16x8_add(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vaddq_s16(i16s(a), i16s(b)) })
}

/// Same op as `i16x8_add` (wasm `i16x8.add`); `core::arch::wasm32` names it both ways.
pub use self::i16x8_add as u16x8_add;

#[inline(always)]
pub fn i16x8_sub(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vsubq_s16(i16s(a), i16s(b)) })
}

#[inline(always)]
pub fn i16x8_sub_sat(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vqsubq_s16(i16s(a), i16s(b)) })
}

/// Wrapping 16-bit multiply (wasm `i16x8.mul`).
#[inline(always)]
pub fn u16x8_mul(a: v128, b: v128) -> v128 {
    from_u16s(unsafe { vmulq_u16(u16s(a), u16s(b)) })
}

#[inline(always)]
pub fn i16x8_min(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vminq_s16(i16s(a), i16s(b)) })
}

#[inline(always)]
pub fn i16x8_max(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vmaxq_s16(i16s(a), i16s(b)) })
}

#[inline(always)]
pub fn i16x8_lt(a: v128, b: v128) -> v128 {
    from_u16s(unsafe { vcltq_s16(i16s(a), i16s(b)) })
}

/// Shift left by `amt mod 16` (one `shl #imm` for a constant count).
#[inline(always)]
pub fn i16x8_shl(a: v128, amt: u32) -> v128 {
    from_u16s(unsafe { vshlq_u16(u16s(a), vdupq_n_s16((amt & 15) as i16)) })
}

/// Arithmetic shift right by `amt mod 16` (one `sshr #imm` for a constant count).
#[inline(always)]
pub fn i16x8_shr(a: v128, amt: u32) -> v128 {
    from_i16s(unsafe { vshlq_s16(i16s(a), vdupq_n_s16(-((amt & 15) as i16))) })
}

/// Logical shift right by `amt mod 16` (one `ushr #imm` for a constant count).
#[inline(always)]
pub fn u16x8_shr(a: v128, amt: u32) -> v128 {
    from_u16s(unsafe { vshlq_u16(u16s(a), vdupq_n_s16(-((amt & 15) as i16))) })
}

/// Q15 rounding multiply `sat_i16((a * b + 2^14) >> 15)`: `sqrdmulh` computes exactly that,
/// including the single saturating case `-32768 * -32768`.
#[inline(always)]
pub fn i16x8_q15mulr_sat(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vqrdmulhq_s16(i16s(a), i16s(b)) })
}

/// Lanes 0..4 from `a`, 4..8 from `b`, each `i32` saturated to `i16` (`sqxtn` + `sqxtn2`).
#[inline(always)]
pub fn i16x8_narrow_i32x4(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vqmovn_high_s32(vqmovn_s32(i32s(a)), i32s(b)) })
}

#[inline(always)]
pub fn i16x8_extend_low_i8x16(a: v128) -> v128 {
    from_i16s(unsafe { vmovl_s8(vget_low_s8(i8s(a))) })
}

#[inline(always)]
pub fn i16x8_extend_high_i8x16(a: v128) -> v128 {
    from_i16s(unsafe { vmovl_high_s8(i8s(a)) })
}

#[inline(always)]
pub fn u16x8_extend_low_u8x16(a: v128) -> v128 {
    from_u16s(unsafe { vmovl_u8(vget_low_u8(u8s(a))) })
}

#[inline(always)]
pub fn u16x8_extend_high_u8x16(a: v128) -> v128 {
    from_u16s(unsafe { vmovl_high_u8(u8s(a)) })
}

#[inline(always)]
pub fn i16x8_extmul_low_i8x16(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vmull_s8(vget_low_s8(i8s(a)), vget_low_s8(i8s(b))) })
}

#[inline(always)]
pub fn i16x8_extmul_high_i8x16(a: v128, b: v128) -> v128 {
    from_i16s(unsafe { vmull_high_s8(i8s(a), i8s(b)) })
}

#[inline(always)]
pub fn u16x8_extmul_low_u8x16(a: v128, b: v128) -> v128 {
    from_u16s(unsafe { vmull_u8(vget_low_u8(u8s(a)), vget_low_u8(u8s(b))) })
}

#[inline(always)]
pub fn u16x8_extmul_high_u8x16(a: v128, b: v128) -> v128 {
    from_u16s(unsafe { vmull_high_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn i16x8_extadd_pairwise_i8x16(a: v128) -> v128 {
    from_i16s(unsafe { vpaddlq_s8(i8s(a)) })
}

#[inline(always)]
pub fn u16x8_extadd_pairwise_u8x16(a: v128) -> v128 {
    from_u16s(unsafe { vpaddlq_u8(u8s(a)) })
}

// ---------------------------------------------------------------------------------------------
// i32x4 / u32x4.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn i32x4_add(a: v128, b: v128) -> v128 {
    from_i32s(unsafe { vaddq_s32(i32s(a), i32s(b)) })
}

#[inline(always)]
pub fn i32x4_sub(a: v128, b: v128) -> v128 {
    from_i32s(unsafe { vsubq_s32(i32s(a), i32s(b)) })
}

#[inline(always)]
pub fn i32x4_min(a: v128, b: v128) -> v128 {
    from_i32s(unsafe { vminq_s32(i32s(a), i32s(b)) })
}

#[inline(always)]
pub fn i32x4_max(a: v128, b: v128) -> v128 {
    from_i32s(unsafe { vmaxq_s32(i32s(a), i32s(b)) })
}

#[inline(always)]
pub fn u32x4_min(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vminq_u32(u32s(a), u32s(b)) })
}

#[inline(always)]
pub fn i32x4_eq(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vceqq_s32(i32s(a), i32s(b)) })
}

/// Same op as `i32x4_eq` (wasm `i32x4.eq`); `core::arch::wasm32` names it both ways.
pub use self::i32x4_eq as u32x4_eq;

/// `cmeq` + `mvn` (NEON has no "not equal" compare).
#[inline(always)]
pub fn i32x4_ne(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vmvnq_u32(vceqq_s32(i32s(a), i32s(b))) })
}

#[inline(always)]
pub fn i32x4_lt(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcltq_s32(i32s(a), i32s(b)) })
}

#[inline(always)]
pub fn i32x4_ge(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcgeq_s32(i32s(a), i32s(b)) })
}

#[inline(always)]
pub fn u32x4_lt(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcltq_u32(u32s(a), u32s(b)) })
}

#[inline(always)]
pub fn u32x4_ge(a: v128, b: v128) -> v128 {
    from_u32s(unsafe { vcgeq_u32(u32s(a), u32s(b)) })
}

/// Shift left by `amt mod 32` (one `shl #imm` for a constant count).
#[inline(always)]
pub fn i32x4_shl(a: v128, amt: u32) -> v128 {
    from_u32s(unsafe { vshlq_u32(u32s(a), vdupq_n_s32((amt & 31) as i32)) })
}

/// Arithmetic shift right by `amt mod 32` (one `sshr #imm` for a constant count).
#[inline(always)]
pub fn i32x4_shr(a: v128, amt: u32) -> v128 {
    from_i32s(unsafe { vshlq_s32(i32s(a), vdupq_n_s32(-((amt & 31) as i32))) })
}

/// Logical shift right by `amt mod 32` (one `ushr #imm` for a constant count).
#[inline(always)]
pub fn u32x4_shr(a: v128, amt: u32) -> v128 {
    from_u32s(unsafe { vshlq_u32(u32s(a), vdupq_n_s32(-((amt & 31) as i32))) })
}

/// Every lane non-zero: the unsigned across-lane minimum (`uminv`) is non-zero.
#[inline(always)]
pub fn i32x4_all_true(a: v128) -> bool {
    unsafe { vminvq_u32(u32s(a)) != 0 }
}

/// Bit `i` of the result is the sign bit of lane `i`: weight each negative lane by `1 << i` and
/// add across lanes.
#[inline(always)]
pub fn i32x4_bitmask(a: v128) -> u8 {
    // SAFETY: same size, every bit pattern valid (constant lane weights).
    const WEIGHTS: uint32x4_t =
        unsafe { core::mem::transmute::<[u32; 4], uint32x4_t>([1, 2, 4, 8]) };
    unsafe { vaddvq_u32(vandq_u32(vcltzq_s32(i32s(a)), WEIGHTS)) as u8 }
}

/// `a[2i]*b[2i] + a[2i+1]*b[2i+1]` in 32 bits (`smull` + `smull2` + `addp`, wrapping only for
/// all-`-32768` inputs, as in wasm).
#[inline(always)]
pub fn i32x4_dot_i16x8(a: v128, b: v128) -> v128 {
    let (a, b) = (i16s(a), i16s(b));
    unsafe {
        let lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        let hi = vmull_high_s16(a, b);
        from_i32s(vpaddq_s32(lo, hi))
    }
}

#[inline(always)]
pub fn i32x4_extend_low_i16x8(a: v128) -> v128 {
    from_i32s(unsafe { vmovl_s16(vget_low_s16(i16s(a))) })
}

#[inline(always)]
pub fn i32x4_extend_high_i16x8(a: v128) -> v128 {
    from_i32s(unsafe { vmovl_high_s16(i16s(a)) })
}

#[inline(always)]
pub fn u32x4_extend_low_u16x8(a: v128) -> v128 {
    from_u32s(unsafe { vmovl_u16(vget_low_u16(u16s(a))) })
}

#[inline(always)]
pub fn u32x4_extend_high_u16x8(a: v128) -> v128 {
    from_u32s(unsafe { vmovl_high_u16(u16s(a)) })
}

#[inline(always)]
pub fn i32x4_extmul_low_i16x8(a: v128, b: v128) -> v128 {
    from_i32s(unsafe { vmull_s16(vget_low_s16(i16s(a)), vget_low_s16(i16s(b))) })
}

#[inline(always)]
pub fn i32x4_extmul_high_i16x8(a: v128, b: v128) -> v128 {
    from_i32s(unsafe { vmull_high_s16(i16s(a), i16s(b)) })
}

#[inline(always)]
pub fn i32x4_extadd_pairwise_i16x8(a: v128) -> v128 {
    from_i32s(unsafe { vpaddlq_s16(i16s(a)) })
}

#[inline(always)]
pub fn u32x4_extadd_pairwise_u16x8(a: v128) -> v128 {
    from_u32s(unsafe { vpaddlq_u16(u16s(a)) })
}

// ---------------------------------------------------------------------------------------------
// Shuffles: each index selects a lane of the concatenation `a ++ b` (indices >= lanes pick `b`).
// ---------------------------------------------------------------------------------------------

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
    shuffle::<Sh16<I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15>>(a, b)
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
    shuffle::<Sh8<I0, I1, I2, I3, I4, I5, I6, I7>>(a, b)
}

#[inline(always)]
pub fn i32x4_shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
    a: v128,
    b: v128,
) -> v128 {
    const { assert!(I0 < 8 && I1 < 8 && I2 < 8 && I3 < 8) };
    shuffle::<Sh4<I0, I1, I2, I3>>(a, b)
}

#[inline(always)]
pub fn i64x2_shuffle<const I0: usize, const I1: usize>(a: v128, b: v128) -> v128 {
    const { assert!(I0 < 4 && I1 < 4) };
    shuffle::<Sh2<I0, I1>>(a, b)
}

// ---- Compile-time shuffle planning --------------------------------------------------------------
//
// The planner only runs during const evaluation. Its `const fn`s are `#[inline]` so that rustc
// does not also emit their never-called bodies into native rlibs (it would otherwise treat them as
// reachable from the public generic shuffles).

/// How a shuffle is implemented: one NEON instruction (or none), or a `TBL` fallback.
#[derive(Clone, Copy)]
struct Plan {
    kind: u8,
    /// Element size in bytes (`DUP`, `ZIP`/`UZP`/`TRN`, `INS`).
    esize: u8,
    /// `EXT`: byte offset; `DUP`: lane; `INS`: destination lane; `REV`: variant (see `REVS`).
    p0: u8,
    /// `INS`: source lane.
    p1: u8,
    /// First operand (`x`) is `b` rather than `a`.
    x_b: bool,
    /// Second operand (`y`) is `b` rather than `a`.
    y_b: bool,
    /// `TBL1` / `TBL2`: byte indices into the table.
    idx: [u8; 16],
}

const K_ID: u8 = 0; // x unchanged
const K_DUP: u8 = 1; // dup x.e[p0]
const K_EXT: u8 = 2; // ext x, y, #p0
const K_ZIP1: u8 = 3;
const K_ZIP2: u8 = 4;
const K_UZP1: u8 = 5;
const K_UZP2: u8 = 6;
const K_TRN1: u8 = 7;
const K_TRN2: u8 = 8;
const K_REV: u8 = 9; // rev{64,32,16} x, variant p0
const K_INS: u8 = 10; // x with x.e[p0] = y.e[p1]
const K_TBL1: u8 = 11; // tbl {x}, idx
const K_TBL2: u8 = 12; // tbl {a, b}, idx

/// `REV` variants as (block bytes, element bytes): `rev64.4s`, `rev64.8h`, `rev64.16b`,
/// `rev32.8h`, `rev32.16b`, `rev16.16b`.
const REVS: [(usize, usize); 6] = [(8, 4), (8, 2), (8, 1), (4, 2), (4, 1), (2, 1)];

/// Operand orders tried for two-operand instructions: (x is b, y is b).
const PAIRS: [(bool, bool); 4] = [(false, true), (true, false), (false, false), (true, true)];

/// Element sizes in bytes, widest first.
const ESIZES: [usize; 4] = [8, 4, 2, 1];

#[inline]
const fn mk(kind: u8, esize: usize, p0: usize, p1: usize, x_b: bool, y_b: bool) -> Plan {
    Plan {
        kind,
        esize: esize as u8,
        p0: p0 as u8,
        p1: p1 as u8,
        x_b,
        y_b,
        idx: [0; 16],
    }
}

/// Whether the instruction mask `m` (byte indices into `x ++ y`, 0..32) applied to the chosen
/// operands yields the target `t` (byte indices into `a ++ b`).
#[inline]
const fn fits(t: &[u8; 16], m: &[u8; 16], x_b: bool, y_b: bool) -> bool {
    let mut i = 0;
    while i < 16 {
        let from_b = if m[i] < 16 { x_b } else { y_b };
        let want = m[i] % 16 + if from_b { 16 } else { 0 };
        if t[i] != want {
            return false;
        }
        i += 1;
    }
    true
}

/// Byte masks (indices into `x ++ y`) of the fixed NEON permutations.
#[inline]
const fn mask_ext(n: usize) -> [u8; 16] {
    let mut m = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        m[i] = (n + i) as u8;
        i += 1;
    }
    m
}

#[inline]
const fn mask_dup(e: usize, lane: usize) -> [u8; 16] {
    let mut m = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        m[i] = (lane * e + i % e) as u8;
        i += 1;
    }
    m
}

/// `ZIP1`/`ZIP2` (`kind` 0/1), `UZP1`/`UZP2` (2/3), `TRN1`/`TRN2` (4/5) with `e`-byte elements.
#[inline]
const fn mask_perm(kind: usize, e: usize) -> [u8; 16] {
    let n = 16 / e;
    let hi = kind % 2;
    let mut m = [0u8; 16];
    let mut k = 0;
    while k < n {
        // Source element as an index into the 2n elements of `x ++ y`.
        let src = match kind / 2 {
            0 => (k % 2) * n + k / 2 + hi * (n / 2),
            1 => 2 * k + hi,
            _ => (k % 2) * n + k - k % 2 + hi,
        };
        let mut j = 0;
        while j < e {
            m[k * e + j] = (src * e + j) as u8;
            j += 1;
        }
        k += 1;
    }
    m
}

#[inline]
const fn mask_rev(blk: usize, e: usize) -> [u8; 16] {
    let mut m = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        m[i] = ((i / blk) * blk + (blk / e - 1 - (i % blk) / e) * e + i % e) as u8;
        i += 1;
    }
    m
}

/// Picks the cheapest implementation of the byte shuffle `t` (indices into `a ++ b`).
#[inline]
const fn plan(t: [u8; 16]) -> Plan {
    let t = &t;
    // No instruction: one operand unchanged.
    let id = mask_ext(0);
    if fits(t, &id, false, false) {
        return mk(K_ID, 0, 0, 0, false, false);
    }
    if fits(t, &id, true, true) {
        return mk(K_ID, 0, 0, 0, true, true);
    }
    // DUP (element).
    let mut s = 0;
    while s < 4 {
        let e = ESIZES[s];
        let mut lane = 0;
        while lane < 16 / e {
            let m = mask_dup(e, lane);
            if fits(t, &m, false, false) {
                return mk(K_DUP, e, lane, 0, false, false);
            }
            if fits(t, &m, true, true) {
                return mk(K_DUP, e, lane, 0, true, true);
            }
            lane += 1;
        }
        s += 1;
    }
    // EXT.
    let mut n = 1;
    while n < 16 {
        let m = mask_ext(n);
        let mut p = 0;
        while p < 4 {
            let (x_b, y_b) = PAIRS[p];
            if fits(t, &m, x_b, y_b) {
                return mk(K_EXT, 1, n, 0, x_b, y_b);
            }
            p += 1;
        }
        n += 1;
    }
    // ZIP / UZP / TRN.
    let mut kind = 0;
    while kind < 6 {
        let mut s = 0;
        while s < 4 {
            let e = ESIZES[s];
            let m = mask_perm(kind, e);
            let mut p = 0;
            while p < 4 {
                let (x_b, y_b) = PAIRS[p];
                if fits(t, &m, x_b, y_b) {
                    return mk(K_ZIP1 + kind as u8, e, 0, 0, x_b, y_b);
                }
                p += 1;
            }
            s += 1;
        }
        kind += 1;
    }
    // REV64 / REV32 / REV16.
    let mut r = 0;
    while r < REVS.len() {
        let m = mask_rev(REVS[r].0, REVS[r].1);
        if fits(t, &m, false, false) {
            return mk(K_REV, REVS[r].1, r, 0, false, false);
        }
        if fits(t, &m, true, true) {
            return mk(K_REV, REVS[r].1, r, 0, true, true);
        }
        r += 1;
    }
    // INS: x with exactly one element replaced by any element of a or b.
    let mut s = 0;
    while s < 4 {
        let e = ESIZES[s];
        let mut xi = 0;
        while xi < 2 {
            let base = (xi * 16) as u8;
            // The single destination element that differs from x, if there is exactly one.
            let mut dst = 16;
            let mut ok = true;
            let mut i = 0;
            while i < 16 {
                if t[i] != base + i as u8 {
                    if dst == 16 {
                        dst = i / e;
                    } else if dst != i / e {
                        ok = false;
                    }
                }
                i += 1;
            }
            if ok && dst < 16 {
                // Its bytes must be one aligned e-byte element of a or b.
                let first = t[dst * e] as usize;
                let mut whole = first % e == 0;
                let mut j = 1;
                while j < e {
                    if t[dst * e + j] as usize != first + j {
                        whole = false;
                    }
                    j += 1;
                }
                if whole {
                    return mk(K_INS, e, dst, (first % 16) / e, xi == 1, first >= 16);
                }
            }
            xi += 1;
        }
        s += 1;
    }
    // TBL: one table register when every index selects the same operand.
    let mut from_a = true;
    let mut from_b = true;
    let mut i = 0;
    while i < 16 {
        if t[i] < 16 {
            from_b = false;
        } else {
            from_a = false;
        }
        i += 1;
    }
    let kind = if from_a || from_b { K_TBL1 } else { K_TBL2 };
    let mut p = mk(kind, 1, 0, 0, from_b, from_b);
    let mut i = 0;
    while i < 16 {
        p.idx[i] = if from_b { t[i] - 16 } else { t[i] };
        i += 1;
    }
    p
}

/// Byte shuffle mask of a lane shuffle with `N` lanes (each `16 / N` bytes).
#[inline]
const fn lane_bytes<const N: usize>(lanes: [usize; N]) -> [u8; 16] {
    let w = 16 / N;
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = (lanes[i / w] * w + i % w) as u8;
        i += 1;
    }
    t
}

/// A shuffle pattern, planned once per instantiation. The individual fields are associated
/// constants so that `match` on them is resolved when the instantiation is generated.
trait Shuffle {
    const PLAN: Plan;
    const KIND: u8 = Self::PLAN.kind;
    const ESIZE: u8 = Self::PLAN.esize;
    const P0: u8 = Self::PLAN.p0;
    const P1: u8 = Self::PLAN.p1;
    const X_B: bool = Self::PLAN.x_b;
    const Y_B: bool = Self::PLAN.y_b;
    const IDX: [u8; 16] = Self::PLAN.idx;
}

struct Sh16<
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
>;

impl<
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
    > Shuffle for Sh16<I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15>
{
    const PLAN: Plan = plan(lane_bytes([
        I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15,
    ]));
}

struct Sh8<
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
    const I4: usize,
    const I5: usize,
    const I6: usize,
    const I7: usize,
>;

impl<
        const I0: usize,
        const I1: usize,
        const I2: usize,
        const I3: usize,
        const I4: usize,
        const I5: usize,
        const I6: usize,
        const I7: usize,
    > Shuffle for Sh8<I0, I1, I2, I3, I4, I5, I6, I7>
{
    const PLAN: Plan = plan(lane_bytes([I0, I1, I2, I3, I4, I5, I6, I7]));
}

struct Sh4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>;

impl<const I0: usize, const I1: usize, const I2: usize, const I3: usize> Shuffle
    for Sh4<I0, I1, I2, I3>
{
    const PLAN: Plan = plan(lane_bytes([I0, I1, I2, I3]));
}

struct Sh2<const I0: usize, const I1: usize>;

impl<const I0: usize, const I1: usize> Shuffle for Sh2<I0, I1> {
    const PLAN: Plan = plan(lane_bytes([I0, I1]));
}

/// `ZIP`/`UZP`/`TRN` at the plan's element size.
macro_rules! perm {
    ($s:ident, $x:expr, $y:expr, $f64:ident, $f32:ident, $f16:ident, $f8:ident) => {
        match $s::ESIZE {
            8 => from_u64s($f64(u64s($x), u64s($y))),
            4 => from_u32s($f32(u32s($x), u32s($y))),
            2 => from_u16s($f16(u16s($x), u16s($y))),
            _ => from_u8s($f8(u8s($x), u8s($y))),
        }
    };
}

/// Executes the plan of `S`. Every `match` scrutinee is an associated constant, so each
/// instantiation keeps exactly one arm.
#[inline(always)]
fn shuffle<S: Shuffle>(a: v128, b: v128) -> v128 {
    let x = if S::X_B { b } else { a };
    let y = if S::Y_B { b } else { a };
    // SAFETY: NEON intrinsics only (see the module docs); lane and offset constants are in range
    // by construction of `plan`.
    unsafe {
        match S::KIND {
            K_ID => x,
            K_DUP => match S::ESIZE {
                8 => from_u64s(lane_match!(S::P0; 0 => 1; |L| vdupq_laneq_u64::<L>(u64s(x)))),
                4 => from_u32s(lane_match!(S::P0; 0, 1, 2 => 3; |L| vdupq_laneq_u32::<L>(u32s(x)))),
                2 => from_u16s(lane_match!(S::P0; 0, 1, 2, 3, 4, 5, 6 => 7; |L| {
                    vdupq_laneq_u16::<L>(u16s(x))
                })),
                _ => from_u8s(
                    lane_match!(S::P0; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 => 15; |L| {
                        vdupq_laneq_u8::<L>(u8s(x))
                    }),
                ),
            },
            K_EXT => from_u8s(
                lane_match!(S::P0; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 => 15; |N| {
                    vextq_u8::<N>(u8s(x), u8s(y))
                }),
            ),
            K_ZIP1 => perm!(S, x, y, vzip1q_u64, vzip1q_u32, vzip1q_u16, vzip1q_u8),
            K_ZIP2 => perm!(S, x, y, vzip2q_u64, vzip2q_u32, vzip2q_u16, vzip2q_u8),
            K_UZP1 => perm!(S, x, y, vuzp1q_u64, vuzp1q_u32, vuzp1q_u16, vuzp1q_u8),
            K_UZP2 => perm!(S, x, y, vuzp2q_u64, vuzp2q_u32, vuzp2q_u16, vuzp2q_u8),
            K_TRN1 => perm!(S, x, y, vtrn1q_u64, vtrn1q_u32, vtrn1q_u16, vtrn1q_u8),
            K_TRN2 => perm!(S, x, y, vtrn2q_u64, vtrn2q_u32, vtrn2q_u16, vtrn2q_u8),
            K_REV => match S::P0 {
                0 => from_u32s(vrev64q_u32(u32s(x))),
                1 => from_u16s(vrev64q_u16(u16s(x))),
                2 => from_u8s(vrev64q_u8(u8s(x))),
                3 => from_u16s(vrev32q_u16(u16s(x))),
                4 => from_u8s(vrev32q_u8(u8s(x))),
                _ => from_u8s(vrev16q_u8(u8s(x))),
            },
            K_INS => match S::ESIZE {
                8 => from_u64s(lane_match!(S::P0; 0 => 1; |D| {
                    lane_match!(S::P1; 0 => 1; |E| vcopyq_laneq_u64::<D, E>(u64s(x), u64s(y)))
                })),
                4 => from_u32s(lane_match!(S::P0; 0, 1, 2 => 3; |D| {
                    lane_match!(S::P1; 0, 1, 2 => 3; |E| vcopyq_laneq_u32::<D, E>(u32s(x), u32s(y)))
                })),
                2 => from_u16s(lane_match!(S::P0; 0, 1, 2, 3, 4, 5, 6 => 7; |D| {
                    lane_match!(S::P1; 0, 1, 2, 3, 4, 5, 6 => 7; |E| {
                        vcopyq_laneq_u16::<D, E>(u16s(x), u16s(y))
                    })
                })),
                _ => from_u8s(
                    lane_match!(S::P0; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 => 15; |D| {
                        lane_match!(S::P1;
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 => 15; |E| {
                            vcopyq_laneq_u8::<D, E>(u8s(x), u8s(y))
                        })
                    }),
                ),
            },
            K_TBL1 => {
                let idx = core::mem::transmute::<[u8; 16], uint8x16_t>(S::IDX);
                from_u8s(vqtbl1q_u8(u8s(x), idx))
            }
            _ => {
                let idx = core::mem::transmute::<[u8; 16], uint8x16_t>(S::IDX);
                from_u8s(vqtbl2q_u8(uint8x16x2_t(u8s(a), u8s(b)), idx))
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Whole-register bitwise ops.
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn v128_and(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vandq_u8(u8s(a), u8s(b)) })
}

/// `a & !b` (`bic`).
#[inline(always)]
pub fn v128_andnot(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vbicq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn v128_or(a: v128, b: v128) -> v128 {
    from_u8s(unsafe { vorrq_u8(u8s(a), u8s(b)) })
}

#[inline(always)]
pub fn v128_not(a: v128) -> v128 {
    from_u8s(unsafe { vmvnq_u8(u8s(a)) })
}

/// Per bit: `c ? v1 : v2` (`bsl`, or `bit`/`bif` when that saves a register copy).
///
/// Written on 64-bit lanes: with byte or 32-bit lanes, LLVM fails to match the select when the
/// mask arrives through an `f32`-typed value it cannot see the origin of (a function argument,
/// for example) and emits `and` + `bic` + `orr`; on 64-bit lanes it emits one `bif` in every
/// context tried (opaque, compare, loaded and loop-carried masks).
#[inline(always)]
pub fn v128_bitselect(v1: v128, v2: v128, c: v128) -> v128 {
    from_u64s(unsafe { vbslq_u64(u64s(c), u64s(v1), u64s(v2)) })
}

/// Any bit set: fold 128 bits to 64 with a pairwise unsigned max (`umaxp`), then test them.
#[inline(always)]
pub fn v128_any_true(a: v128) -> bool {
    unsafe {
        let m = vpmaxq_u32(u32s(a), u32s(a));
        vgetq_lane_u64::<0>(vreinterpretq_u64_u32(m)) != 0
    }
}

// ---------------------------------------------------------------------------------------------
// Loads and stores (unaligned, as in wasm; AArch64 `ldr`/`str` allow any alignment).
// ---------------------------------------------------------------------------------------------

/// `ldr q`.
///
/// # Safety
/// `m` must be valid for a 16-byte read; no alignment is required.
#[inline(always)]
pub unsafe fn v128_load(m: *const v128) -> v128 {
    core::ptr::read_unaligned(m)
}

/// Loads 32 bits into lane 0 and zeroes the rest (`ldr s`, which clears the upper lanes).
///
/// # Safety
/// `m` must be valid for a 4-byte read; no alignment is required.
#[inline(always)]
pub unsafe fn v128_load32_zero(m: *const u32) -> v128 {
    from_u32s(vsetq_lane_u32::<0>(
        core::ptr::read_unaligned(m),
        vdupq_n_u32(0),
    ))
}

/// Loads 64 bits into lane 0 and zeroes the rest (`ldr d`, which clears the upper lane).
///
/// # Safety
/// `m` must be valid for an 8-byte read; no alignment is required.
#[inline(always)]
pub unsafe fn v128_load64_zero(m: *const u64) -> v128 {
    from_u64s(vsetq_lane_u64::<0>(
        core::ptr::read_unaligned(m),
        vdupq_n_u64(0),
    ))
}

/// `str q`.
///
/// # Safety
/// `m` must be valid for a 16-byte write; no alignment is required.
#[inline(always)]
pub unsafe fn v128_store(m: *mut v128, a: v128) {
    core::ptr::write_unaligned(m, a)
}

/// Stores 32-bit lane `L` (`str s` for lane 0, `st1 {v.s}[L]` otherwise).
///
/// # Safety
/// `m` must be valid for a 4-byte write; no alignment is required.
#[inline(always)]
pub unsafe fn v128_store32_lane<const L: usize>(v: v128, m: *mut u32) {
    const { assert!(L < 4) };
    let v = u32s(v);
    core::ptr::write_unaligned(m, lane_match!(L; 0, 1, 2 => 3; |N| vgetq_lane_u32::<N>(v)))
}

/// Stores 64-bit lane `L` (`str d` for lane 0, `st1 {v.d}[1]` otherwise).
///
/// # Safety
/// `m` must be valid for an 8-byte write; no alignment is required.
#[inline(always)]
pub unsafe fn v128_store64_lane<const L: usize>(v: v128, m: *mut u64) {
    const { assert!(L < 2) };
    let v = u64s(v);
    core::ptr::write_unaligned(m, lane_match!(L; 0 => 1; |N| vgetq_lane_u64::<N>(v)))
}

// ---------------------------------------------------------------------------------------------
// Tests of the shuffle planner (host only). Conformance against `simd::scalar` lives in
// `simd/conformance.rs`.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// (kind, element bytes, p0, p1, x is b, y is b) of a planned shuffle.
    fn shape<S: Shuffle>() -> (u8, u8, u8, u8, bool, bool) {
        (S::KIND, S::ESIZE, S::P0, S::P1, S::X_B, S::Y_B)
    }

    /// The kernel's hottest shuffle patterns (by call-site count) each plan to the one
    /// instruction that implements them.
    #[test]
    fn hot_kernel_shuffles_plan_to_one_instruction() {
        assert_eq!(shape::<Sh4<1, 2, 3, 4>>(), (K_EXT, 1, 4, 0, false, true));
        assert_eq!(shape::<Sh4<1, 3, 5, 7>>(), (K_UZP2, 4, 0, 0, false, true));
        assert_eq!(shape::<Sh4<0, 2, 4, 6>>(), (K_UZP1, 4, 0, 0, false, true));
        assert_eq!(shape::<Sh4<2, 3, 4, 5>>(), (K_EXT, 1, 8, 0, false, true));
        assert_eq!(shape::<Sh4<3, 4, 5, 6>>(), (K_EXT, 1, 12, 0, false, true));
        assert_eq!(
            shape::<Sh16<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>>(),
            (K_EXT, 1, 1, 0, false, true)
        );
        assert_eq!(shape::<Sh4<0, 4, 1, 5>>(), (K_ZIP1, 4, 0, 0, false, true));
        assert_eq!(shape::<Sh4<2, 6, 3, 7>>(), (K_ZIP2, 4, 0, 0, false, true));
        assert_eq!(shape::<Sh4<2, 3, 6, 7>>(), (K_ZIP2, 8, 0, 0, false, true));
        assert_eq!(shape::<Sh4<2, 3, 0, 1>>(), (K_EXT, 1, 8, 0, false, false));
        assert_eq!(shape::<Sh4<1, 0, 3, 2>>(), (K_REV, 4, 0, 0, false, false));
        assert_eq!(shape::<Sh4<0, 1, 4, 5>>(), (K_ZIP1, 8, 0, 0, false, true));
        assert_eq!(shape::<Sh2<0, 2>>(), (K_ZIP1, 8, 0, 0, false, true));
        assert_eq!(shape::<Sh4<3, 3, 3, 3>>(), (K_DUP, 4, 3, 0, false, false));
        assert_eq!(shape::<Sh4<0, 1, 6, 3>>(), (K_INS, 4, 2, 2, false, true));
        assert_eq!(shape::<Sh4<3, 5, 6, 7>>(), (K_INS, 4, 0, 3, true, false));
        assert_eq!(shape::<Sh4<0, 1, 6, 7>>(), (K_INS, 8, 1, 1, false, true));
        assert_eq!(
            shape::<Sh8<4, 4, 5, 5, 6, 6, 7, 7>>(),
            (K_ZIP2, 2, 0, 0, false, false)
        );
        assert_eq!(shape::<Sh4<0, 1, 2, 3>>(), (K_ID, 0, 0, 0, false, false));
        assert_eq!(shape::<Sh2<3, 2>>(), (K_EXT, 1, 8, 0, true, true));
    }

    /// Patterns with no single-instruction form fall back to `TBL`, one table register when
    /// only one operand is used.
    #[test]
    fn other_shuffles_fall_back_to_tbl() {
        assert_eq!(<Sh4<1, 2, 3, 3>>::KIND, K_TBL1);
        assert_eq!(<Sh4<0, 5, 2, 7>>::KIND, K_TBL2);
        assert_eq!(<Sh4<7, 6, 5, 4>>::KIND, K_TBL1);
        assert!(<Sh4<7, 6, 5, 4>>::X_B);
        assert_eq!(
            <Sh4<7, 6, 5, 4>>::IDX,
            [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
        );
    }

    /// Every byte mask the planner matches against is the documented NEON permutation.
    #[test]
    fn permutation_masks_follow_the_architecture() {
        assert_eq!(
            mask_perm(0, 4),
            [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23]
        ); // zip1.4s
        assert_eq!(
            mask_perm(1, 8),
            [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
        ); // zip2.2d
        assert_eq!(
            mask_perm(2, 2),
            [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29]
        ); // uzp1.8h
        assert_eq!(
            mask_perm(3, 1),
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        ); // uzp2.16b
        assert_eq!(
            mask_perm(4, 4),
            [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27]
        ); // trn1.4s
        assert_eq!(
            mask_perm(5, 2),
            [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31]
        ); // trn2.8h
        assert_eq!(
            mask_rev(8, 2),
            [6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9]
        ); // rev64.8h
        assert_eq!(
            mask_rev(2, 1),
            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
        ); // rev16.16b
    }
}
