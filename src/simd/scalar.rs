//! Portable scalar reference backend for the SIMD layer (`core` only, no `unsafe` beyond the
//! load/store ops whose `core::arch::wasm32` signatures are `unsafe fn`).
//!
//! Every function has the `core::arch::wasm32` name and signature and computes exactly what the
//! WebAssembly SIMD specification defines, lane by lane, written for clarity rather than speed.
//! It is the oracle the arch-specific backends are checked against.
//!
//! Lanes are numbered as in wasm: lane `i` of an `f32x4` occupies bytes `4*i..4*i+4` of the
//! 16-byte register in little-endian order, which is also its order in memory, so loading a
//! `v128` from an `[f32; 4]` yields lanes in array order.
//!
//! Floating point: `+ - * /` and the int/float conversions use Rust's IEEE-754 semantics
//! (round to nearest, ties to even, no flush-to-zero), which is what wasm requires. `core` has no
//! stable `sqrt`/`floor`/`round_ties_even` on Rust 1.99.0 (they sit behind the unstable
//! `core_float_math` feature), so the correctly rounded square root and the rounding functions
//! are implemented here on the raw bits. Where wasm leaves a NaN result's payload and sign open,
//! this backend returns a quieted input NaN, or the positive default NaN when no input is NaN.

#![allow(non_camel_case_types)]

#[cfg(target_endian = "big")]
compile_error!(
    "simd::scalar models wasm's little-endian lane layout and needs a little-endian target"
);

/// 128-bit SIMD register: sixteen bytes holding lanes in little-endian order.
///
/// 16 bytes, 16-byte aligned and `Copy`, like `core::arch::wasm32::v128`, so kernel code that
/// casts `*const f32` to `*const v128` or embeds `v128` in structs works unchanged.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct v128([u8; 16]);

const _: () = assert!(core::mem::size_of::<v128>() == 16 && core::mem::align_of::<v128>() == 16);

// ---------------------------------------------------------------------------------------------
// Lane views.
// ---------------------------------------------------------------------------------------------

macro_rules! lane_view {
    ($get:ident, $put:ident, $t:ty, $n:literal, $w:literal) => {
        #[inline]
        const fn $get(v: v128) -> [$t; $n] {
            let mut out = [<$t>::from_le_bytes([0; $w]); $n];
            let mut i = 0;
            while i < $n {
                let mut b = [0u8; $w];
                let mut j = 0;
                while j < $w {
                    b[j] = v.0[i * $w + j];
                    j += 1;
                }
                out[i] = <$t>::from_le_bytes(b);
                i += 1;
            }
            out
        }

        #[inline]
        const fn $put(lanes: [$t; $n]) -> v128 {
            let mut out = [0u8; 16];
            let mut i = 0;
            while i < $n {
                let b = lanes[i].to_le_bytes();
                let mut j = 0;
                while j < $w {
                    out[i * $w + j] = b[j];
                    j += 1;
                }
                i += 1;
            }
            v128(out)
        }
    };
}

lane_view!(get_u8, put_u8, u8, 16, 1);
lane_view!(get_i8, put_i8, i8, 16, 1);
lane_view!(get_u16, put_u16, u16, 8, 2);
lane_view!(get_i16, put_i16, i16, 8, 2);
lane_view!(get_u32, put_u32, u32, 4, 4);
lane_view!(get_i32, put_i32, i32, 4, 4);
lane_view!(get_f32, put_f32, f32, 4, 4);
lane_view!(get_u64, put_u64, u64, 2, 8);
lane_view!(get_f64, put_f64, f64, 2, 8);

/// `lanewise!(get => put, n, |a, b| expr)`: views each named `v128` operand through `get`,
/// evaluates `expr` per lane with the operands bound to that lane's values, and packs the `n`
/// results with `put`.
macro_rules! lanewise {
    ($get:ident => $put:ident, $n:literal, |$($x:ident),+| $body:expr) => {{
        $( let $x = $get($x); )+
        let mut out = [Default::default(); $n];
        let mut i = 0;
        while i < $n {
            out[i] = {
                $( let $x = $x[i]; )+
                $body
            };
            i += 1;
        }
        $put(out)
    }};
}

/// All-ones lane mask for `true`, all-zeros for `false`.
macro_rules! mask {
    ($c:expr) => {
        if $c {
            -1
        } else {
            0
        }
    };
}

// ---------------------------------------------------------------------------------------------
// Scalar IEEE-754 helpers.
// ---------------------------------------------------------------------------------------------

#[inline]
fn quiet_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() | 0x0040_0000)
}

#[inline]
fn quiet_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() | 0x0008_0000_0000_0000)
}

/// Integer square root: `(floor(sqrt(n)), sqrt(n) is not an integer)`, digit by digit.
fn isqrt_u128(n: u128) -> (u128, bool) {
    let mut rem = n;
    let mut root = 0u128;
    let mut bit = 1u128 << 126;
    while bit > n {
        bit >>= 2;
    }
    while bit != 0 {
        if rem >= root + bit {
            rem -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    (root, rem != 0)
}

/// Correctly rounded (ties to even) square root on the raw bits of a binary IEEE-754 format
/// with `mant` fraction bits and `expw` exponent bits (f32: 23/8, f64: 52/11).
fn sqrt_bits(bits: u64, mant: u32, expw: u32) -> u64 {
    let sign = 1u64 << (mant + expw);
    let exp_max = (1u64 << expw) - 1;
    let bias = (exp_max >> 1) as i32;
    let frac_mask = (1u64 << mant) - 1;
    let quiet = 1u64 << (mant - 1);
    let exp_field = (bits >> mant) & exp_max;
    let frac = bits & frac_mask;

    if exp_field == exp_max && frac != 0 {
        return bits | quiet; // NaN in, quieted NaN out
    }
    if bits & !sign == 0 {
        return bits; // sqrt(+-0) = +-0
    }
    if bits & sign != 0 {
        return (exp_max << mant) | quiet; // negative (including -inf): default NaN
    }
    if exp_field == exp_max {
        return bits; // sqrt(+inf) = +inf
    }

    // Positive finite x = m * 2^e with integer m normalised to [2^mant, 2^(mant+1)).
    let (mut m, mut e) = if exp_field == 0 {
        (frac, 1 - bias - mant as i32)
    } else {
        (frac | (1u64 << mant), exp_field as i32 - bias - mant as i32)
    };
    while m & (1u64 << mant) == 0 {
        m <<= 1;
        e -= 1;
    }
    if e & 1 != 0 {
        m <<= 1; // make the exponent even: m in [2^mant, 2^(mant+2))
        e -= 1;
    }
    // sqrt(x) = sqrt(m << 2k) * 2^(e/2 - k). With k = mant/2 + 3 the integer root has at least
    // mant + 3 bits: mant + 1 significand bits, a round bit and one more guard bit.
    let k = (mant / 2 + 3) as i32;
    let (r, inexact) = isqrt_u128((m as u128) << (2 * k));
    let r_bits = 128 - r.leading_zeros();
    let shift = r_bits - (mant + 1);
    let mut sig = (r >> shift) as u64;
    let dropped = r & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    // The exact root is r + delta with 0 <= delta < 1 and delta > 0 iff `inexact`.
    let round_up = dropped > half || (dropped == half && (inexact || sig & 1 == 1));
    let mut exp = e / 2 - k + shift as i32;
    if round_up {
        sig += 1;
        if sig == 1u64 << (mant + 1) {
            sig >>= 1;
            exp += 1;
        }
    }
    // The square root of a positive finite value is always a normal number.
    let biased = (exp + mant as i32 + bias) as u64;
    (biased << mant) | (sig & frac_mask)
}

#[inline]
fn sqrt_f32(x: f32) -> f32 {
    f32::from_bits(sqrt_bits(x.to_bits() as u64, 23, 8) as u32)
}

#[inline]
fn sqrt_f64(x: f64) -> f64 {
    f64::from_bits(sqrt_bits(x.to_bits(), 52, 11))
}

/// Rounds toward zero. Values with |x| >= 2^23 are already integral (or inf/NaN).
#[inline]
fn trunc_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    if e >= 23 {
        x
    } else if e < 0 {
        f32::from_bits(bits & 0x8000_0000)
    } else {
        f32::from_bits(bits & !((1u32 << (23 - e)) - 1))
    }
}

/// `f32.floor`: round toward negative infinity; -0 stays -0.
#[inline]
fn floor_f32(x: f32) -> f32 {
    if x.is_nan() {
        return quiet_f32(x);
    }
    let t = trunc_f32(x);
    // t > x only for negative non-integers, where |t| < 2^23 so t - 1 is exact.
    if t > x {
        t - 1.0
    } else {
        t
    }
}

/// `f32.nearest`: round to nearest integer, ties to even; the sign of a zero result follows x.
#[inline]
fn nearest_f32(x: f32) -> f32 {
    if x.is_nan() {
        return quiet_f32(x);
    }
    if !(f32::from_bits(x.to_bits() & 0x7FFF_FFFF) < 8_388_608.0) {
        return x; // |x| >= 2^23 (or inf): already integral
    }
    let t = trunc_f32(x);
    let frac = f32::from_bits((x - t).to_bits() & 0x7FFF_FFFF); // exact
    let t_is_odd = (t as i32) & 1 != 0;
    if frac > 0.5 || (frac == 0.5 && t_is_odd) {
        t + f32::from_bits(0x3F80_0000 | (x.to_bits() & 0x8000_0000)) // t + copysign(1, x)
    } else {
        t
    }
}

/// `f32x4.min` lane: NaN if either input is NaN, and -0 is less than +0.
#[inline]
fn wasm_fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        quiet_f32(a)
    } else if b.is_nan() {
        quiet_f32(b)
    } else if a == b {
        // Equal non-zero values have equal bits; for {+0, -0} OR-ing the bits yields -0.
        f32::from_bits(a.to_bits() | b.to_bits())
    } else if a < b {
        a
    } else {
        b
    }
}

/// `f32x4.max` lane: NaN if either input is NaN, and +0 is greater than -0.
#[inline]
fn wasm_fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        quiet_f32(a)
    } else if b.is_nan() {
        quiet_f32(b)
    } else if a == b {
        // Equal non-zero values have equal bits; for {+0, -0} AND-ing the bits yields +0.
        f32::from_bits(a.to_bits() & b.to_bits())
    } else if a > b {
        a
    } else {
        b
    }
}

#[inline]
fn sat_i16(x: i32) -> i16 {
    if x > i16::MAX as i32 {
        i16::MAX
    } else if x < i16::MIN as i32 {
        i16::MIN
    } else {
        x as i16
    }
}

#[inline]
fn sat_i8(x: i16) -> i8 {
    if x > i8::MAX as i16 {
        i8::MAX
    } else if x < i8::MIN as i16 {
        i8::MIN
    } else {
        x as i8
    }
}

#[inline]
fn sat_u8(x: i16) -> u8 {
    if x > u8::MAX as i16 {
        u8::MAX
    } else if x < 0 {
        0
    } else {
        x as u8
    }
}

// ---------------------------------------------------------------------------------------------
// Constructors (`v128.const`).
// ---------------------------------------------------------------------------------------------

#[inline]
pub const fn f32x4(a0: f32, a1: f32, a2: f32, a3: f32) -> v128 {
    put_f32([a0, a1, a2, a3])
}

#[inline]
pub const fn f64x2(a0: f64, a1: f64) -> v128 {
    put_f64([a0, a1])
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub const fn i16x8(a0: i16, a1: i16, a2: i16, a3: i16, a4: i16, a5: i16, a6: i16, a7: i16) -> v128 {
    put_i16([a0, a1, a2, a3, a4, a5, a6, a7])
}

#[inline]
pub const fn i32x4(a0: i32, a1: i32, a2: i32, a3: i32) -> v128 {
    put_i32([a0, a1, a2, a3])
}

#[inline]
pub const fn u32x4(a0: u32, a1: u32, a2: u32, a3: u32) -> v128 {
    put_u32([a0, a1, a2, a3])
}

#[inline]
pub const fn u64x2(a0: u64, a1: u64) -> v128 {
    put_u64([a0, a1])
}

// ---------------------------------------------------------------------------------------------
// Splats, lane extraction and replacement.
// ---------------------------------------------------------------------------------------------

#[inline]
pub fn f32x4_splat(a: f32) -> v128 {
    put_f32([a; 4])
}

#[inline]
pub fn f64x2_splat(a: f64) -> v128 {
    put_f64([a; 2])
}

#[inline]
pub fn i8x16_splat(a: i8) -> v128 {
    put_i8([a; 16])
}

#[inline]
pub fn u8x16_splat(a: u8) -> v128 {
    put_u8([a; 16])
}

#[inline]
pub fn i16x8_splat(a: i16) -> v128 {
    put_i16([a; 8])
}

#[inline]
pub fn u16x8_splat(a: u16) -> v128 {
    put_u16([a; 8])
}

#[inline]
pub fn i32x4_splat(a: i32) -> v128 {
    put_i32([a; 4])
}

#[inline]
pub fn u32x4_splat(a: u32) -> v128 {
    put_u32([a; 4])
}

#[inline]
pub fn f32x4_extract_lane<const N: usize>(a: v128) -> f32 {
    const { assert!(N < 4) };
    get_f32(a)[N]
}

#[inline]
pub fn f64x2_extract_lane<const N: usize>(a: v128) -> f64 {
    const { assert!(N < 2) };
    get_f64(a)[N]
}

#[inline]
pub fn i32x4_extract_lane<const N: usize>(a: v128) -> i32 {
    const { assert!(N < 4) };
    get_i32(a)[N]
}

#[inline]
pub fn u32x4_extract_lane<const N: usize>(a: v128) -> u32 {
    const { assert!(N < 4) };
    get_u32(a)[N]
}

#[inline]
pub fn u64x2_extract_lane<const N: usize>(a: v128) -> u64 {
    const { assert!(N < 2) };
    get_u64(a)[N]
}

#[inline]
pub fn f32x4_replace_lane<const N: usize>(a: v128, val: f32) -> v128 {
    const { assert!(N < 4) };
    let mut l = get_f32(a);
    l[N] = val;
    put_f32(l)
}

#[inline]
pub fn i32x4_replace_lane<const N: usize>(a: v128, val: i32) -> v128 {
    const { assert!(N < 4) };
    let mut l = get_i32(a);
    l[N] = val;
    put_i32(l)
}

// ---------------------------------------------------------------------------------------------
// f32x4.
// ---------------------------------------------------------------------------------------------

/// Clears the sign bit (a bit operation in wasm: NaN payloads are preserved).
#[inline]
pub fn f32x4_abs(a: v128) -> v128 {
    lanewise!(get_u32 => put_u32, 4, |a| a & 0x7FFF_FFFF)
}

/// Flips the sign bit (a bit operation in wasm: NaN payloads are preserved).
#[inline]
pub fn f32x4_neg(a: v128) -> v128 {
    lanewise!(get_u32 => put_u32, 4, |a| a ^ 0x8000_0000)
}

#[inline]
pub fn f32x4_add(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| a + b)
}

#[inline]
pub fn f32x4_sub(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| a - b)
}

#[inline]
pub fn f32x4_mul(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| a * b)
}

#[inline]
pub fn f32x4_div(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| a / b)
}

#[inline]
pub fn f32x4_sqrt(a: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a| sqrt_f32(a))
}

#[inline]
pub fn f32x4_floor(a: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a| floor_f32(a))
}

#[inline]
pub fn f32x4_nearest(a: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a| nearest_f32(a))
}

#[inline]
pub fn f32x4_min(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| wasm_fmin(a, b))
}

#[inline]
pub fn f32x4_max(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| wasm_fmax(a, b))
}

/// Pseudo-minimum `b < a ? b : a`: returns one input's bits unchanged (NaN in `b` never wins).
#[inline]
pub fn f32x4_pmin(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| if b < a { b } else { a })
}

/// Pseudo-maximum `a < b ? b : a`: returns one input's bits unchanged (NaN in `b` never wins).
#[inline]
pub fn f32x4_pmax(a: v128, b: v128) -> v128 {
    lanewise!(get_f32 => put_f32, 4, |a, b| if a < b { b } else { a })
}

#[inline]
pub fn f32x4_lt(a: v128, b: v128) -> v128 {
    let (a, b) = (get_f32(a), get_f32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] < b[i])))
}

#[inline]
pub fn f32x4_le(a: v128, b: v128) -> v128 {
    let (a, b) = (get_f32(a), get_f32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] <= b[i])))
}

#[inline]
pub fn f32x4_gt(a: v128, b: v128) -> v128 {
    let (a, b) = (get_f32(a), get_f32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] > b[i])))
}

#[inline]
pub fn f32x4_ge(a: v128, b: v128) -> v128 {
    let (a, b) = (get_f32(a), get_f32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] >= b[i])))
}

/// Signed `i32` to `f32`, rounding to nearest, ties to even.
#[inline]
pub fn f32x4_convert_i32x4(a: v128) -> v128 {
    put_f32(get_i32(a).map(|x| x as f32))
}

/// Unsigned `u32` to `f32`, rounding to nearest, ties to even.
#[inline]
pub fn f32x4_convert_u32x4(a: v128) -> v128 {
    put_f32(get_u32(a).map(|x| x as f32))
}

/// `f32` to `i32` toward zero, saturating at the `i32` range; NaN becomes 0.
#[inline]
pub fn i32x4_trunc_sat_f32x4(a: v128) -> v128 {
    // Rust's float-to-int `as` is defined as exactly this saturating conversion.
    put_i32(get_f32(a).map(|x| x as i32))
}

// ---------------------------------------------------------------------------------------------
// f64x2.
// ---------------------------------------------------------------------------------------------

#[inline]
pub fn f64x2_add(a: v128, b: v128) -> v128 {
    lanewise!(get_f64 => put_f64, 2, |a, b| a + b)
}

#[inline]
pub fn f64x2_sub(a: v128, b: v128) -> v128 {
    lanewise!(get_f64 => put_f64, 2, |a, b| a - b)
}

#[inline]
pub fn f64x2_mul(a: v128, b: v128) -> v128 {
    lanewise!(get_f64 => put_f64, 2, |a, b| a * b)
}

#[inline]
pub fn f64x2_sqrt(a: v128) -> v128 {
    lanewise!(get_f64 => put_f64, 2, |a| if a.is_nan() { quiet_f64(a) } else { sqrt_f64(a) })
}

// ---------------------------------------------------------------------------------------------
// i8x16 / u8x16.
// ---------------------------------------------------------------------------------------------

#[inline]
pub fn i8x16_add(a: v128, b: v128) -> v128 {
    lanewise!(get_i8 => put_i8, 16, |a, b| a.wrapping_add(b))
}

#[inline]
pub fn i8x16_sub(a: v128, b: v128) -> v128 {
    lanewise!(get_i8 => put_i8, 16, |a, b| a.wrapping_sub(b))
}

#[inline]
pub fn i8x16_min(a: v128, b: v128) -> v128 {
    lanewise!(get_i8 => put_i8, 16, |a, b| if a < b { a } else { b })
}

#[inline]
pub fn i8x16_max(a: v128, b: v128) -> v128 {
    lanewise!(get_i8 => put_i8, 16, |a, b| if a > b { a } else { b })
}

/// Bit `i` of the result is the sign bit of lane `i`.
#[inline]
pub fn i8x16_bitmask(a: v128) -> u16 {
    let l = get_i8(a);
    let mut m = 0u16;
    let mut i = 0;
    while i < 16 {
        m |= ((l[i] < 0) as u16) << i;
        i += 1;
    }
    m
}

/// Lanes 0..8 from `a`, 8..16 from `b`, each `i16` saturated to `i8`.
#[inline]
pub fn i8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i16(a), get_i16(b));
    put_i8(core::array::from_fn(|i| {
        sat_i8(if i < 8 { a[i] } else { b[i - 8] })
    }))
}

#[inline]
pub fn u8x16_add_sat(a: v128, b: v128) -> v128 {
    lanewise!(get_u8 => put_u8, 16, |a, b| a.saturating_add(b))
}

#[inline]
pub fn u8x16_sub_sat(a: v128, b: v128) -> v128 {
    lanewise!(get_u8 => put_u8, 16, |a, b| a.saturating_sub(b))
}

#[inline]
pub fn u8x16_min(a: v128, b: v128) -> v128 {
    lanewise!(get_u8 => put_u8, 16, |a, b| if a < b { a } else { b })
}

#[inline]
pub fn u8x16_max(a: v128, b: v128) -> v128 {
    lanewise!(get_u8 => put_u8, 16, |a, b| if a > b { a } else { b })
}

#[inline]
pub fn u8x16_lt(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u8(a), get_u8(b));
    put_i8(core::array::from_fn(|i| mask!(a[i] < b[i])))
}

#[inline]
pub fn u8x16_le(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u8(a), get_u8(b));
    put_i8(core::array::from_fn(|i| mask!(a[i] <= b[i])))
}

#[inline]
pub fn u8x16_gt(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u8(a), get_u8(b));
    put_i8(core::array::from_fn(|i| mask!(a[i] > b[i])))
}

#[inline]
pub fn u8x16_ge(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u8(a), get_u8(b));
    put_i8(core::array::from_fn(|i| mask!(a[i] >= b[i])))
}

/// Lanes 0..8 from `a`, 8..16 from `b`, each signed `i16` saturated to `u8`.
#[inline]
pub fn u8x16_narrow_i16x8(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i16(a), get_i16(b));
    put_u8(core::array::from_fn(|i| {
        sat_u8(if i < 8 { a[i] } else { b[i - 8] })
    }))
}

// ---------------------------------------------------------------------------------------------
// i16x8 / u16x8.
// ---------------------------------------------------------------------------------------------

#[inline]
pub fn i16x8_add(a: v128, b: v128) -> v128 {
    lanewise!(get_i16 => put_i16, 8, |a, b| a.wrapping_add(b))
}

/// Same op as `i16x8_add` (wasm `i16x8.add`); `core::arch::wasm32` names it both ways.
pub use self::i16x8_add as u16x8_add;

#[inline]
pub fn i16x8_sub(a: v128, b: v128) -> v128 {
    lanewise!(get_i16 => put_i16, 8, |a, b| a.wrapping_sub(b))
}

#[inline]
pub fn i16x8_sub_sat(a: v128, b: v128) -> v128 {
    lanewise!(get_i16 => put_i16, 8, |a, b| a.saturating_sub(b))
}

/// Wrapping 16-bit multiply (wasm `i16x8.mul`, named `u16x8_mul` in `core::arch::wasm32` too).
#[inline]
pub fn u16x8_mul(a: v128, b: v128) -> v128 {
    lanewise!(get_u16 => put_u16, 8, |a, b| a.wrapping_mul(b))
}

#[inline]
pub fn i16x8_min(a: v128, b: v128) -> v128 {
    lanewise!(get_i16 => put_i16, 8, |a, b| if a < b { a } else { b })
}

#[inline]
pub fn i16x8_max(a: v128, b: v128) -> v128 {
    lanewise!(get_i16 => put_i16, 8, |a, b| if a > b { a } else { b })
}

#[inline]
pub fn i16x8_lt(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i16(a), get_i16(b));
    put_i16(core::array::from_fn(|i| mask!(a[i] < b[i])))
}

/// Shift left by `amt mod 16`.
#[inline]
pub fn i16x8_shl(a: v128, amt: u32) -> v128 {
    put_u16(get_u16(a).map(|x| x << (amt & 15)))
}

/// Arithmetic shift right by `amt mod 16`.
#[inline]
pub fn i16x8_shr(a: v128, amt: u32) -> v128 {
    put_i16(get_i16(a).map(|x| x >> (amt & 15)))
}

/// Logical shift right by `amt mod 16`.
#[inline]
pub fn u16x8_shr(a: v128, amt: u32) -> v128 {
    put_u16(get_u16(a).map(|x| x >> (amt & 15)))
}

/// Q15 rounding multiply: `sat_i16((a * b + 2^14) >> 15)`; only `-32768 * -32768` saturates.
#[inline]
pub fn i16x8_q15mulr_sat(a: v128, b: v128) -> v128 {
    lanewise!(get_i16 => put_i16, 8, |a, b| sat_i16((a as i32 * b as i32 + 0x4000) >> 15))
}

/// Lanes 0..4 from `a`, 4..8 from `b`, each `i32` saturated to `i16`.
#[inline]
pub fn i16x8_narrow_i32x4(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i32(a), get_i32(b));
    put_i16(core::array::from_fn(|i| {
        sat_i16(if i < 4 { a[i] } else { b[i - 4] })
    }))
}

#[inline]
pub fn i16x8_extend_low_i8x16(a: v128) -> v128 {
    let a = get_i8(a);
    put_i16(core::array::from_fn(|i| a[i] as i16))
}

#[inline]
pub fn i16x8_extend_high_i8x16(a: v128) -> v128 {
    let a = get_i8(a);
    put_i16(core::array::from_fn(|i| a[i + 8] as i16))
}

#[inline]
pub fn u16x8_extend_low_u8x16(a: v128) -> v128 {
    let a = get_u8(a);
    put_u16(core::array::from_fn(|i| a[i] as u16))
}

#[inline]
pub fn u16x8_extend_high_u8x16(a: v128) -> v128 {
    let a = get_u8(a);
    put_u16(core::array::from_fn(|i| a[i + 8] as u16))
}

#[inline]
pub fn i16x8_extmul_low_i8x16(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i8(a), get_i8(b));
    put_i16(core::array::from_fn(|i| a[i] as i16 * b[i] as i16))
}

#[inline]
pub fn i16x8_extmul_high_i8x16(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i8(a), get_i8(b));
    put_i16(core::array::from_fn(|i| a[i + 8] as i16 * b[i + 8] as i16))
}

#[inline]
pub fn u16x8_extmul_low_u8x16(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u8(a), get_u8(b));
    put_u16(core::array::from_fn(|i| a[i] as u16 * b[i] as u16))
}

#[inline]
pub fn u16x8_extmul_high_u8x16(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u8(a), get_u8(b));
    put_u16(core::array::from_fn(|i| a[i + 8] as u16 * b[i + 8] as u16))
}

#[inline]
pub fn i16x8_extadd_pairwise_i8x16(a: v128) -> v128 {
    let a = get_i8(a);
    put_i16(core::array::from_fn(|i| {
        a[2 * i] as i16 + a[2 * i + 1] as i16
    }))
}

#[inline]
pub fn u16x8_extadd_pairwise_u8x16(a: v128) -> v128 {
    let a = get_u8(a);
    put_u16(core::array::from_fn(|i| {
        a[2 * i] as u16 + a[2 * i + 1] as u16
    }))
}

// ---------------------------------------------------------------------------------------------
// i32x4 / u32x4.
// ---------------------------------------------------------------------------------------------

#[inline]
pub fn i32x4_add(a: v128, b: v128) -> v128 {
    lanewise!(get_i32 => put_i32, 4, |a, b| a.wrapping_add(b))
}

#[inline]
pub fn i32x4_sub(a: v128, b: v128) -> v128 {
    lanewise!(get_i32 => put_i32, 4, |a, b| a.wrapping_sub(b))
}

#[inline]
pub fn i32x4_min(a: v128, b: v128) -> v128 {
    lanewise!(get_i32 => put_i32, 4, |a, b| if a < b { a } else { b })
}

#[inline]
pub fn i32x4_max(a: v128, b: v128) -> v128 {
    lanewise!(get_i32 => put_i32, 4, |a, b| if a > b { a } else { b })
}

#[inline]
pub fn u32x4_min(a: v128, b: v128) -> v128 {
    lanewise!(get_u32 => put_u32, 4, |a, b| if a < b { a } else { b })
}

#[inline]
pub fn i32x4_eq(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i32(a), get_i32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] == b[i])))
}

/// Same op as `i32x4_eq` (wasm `i32x4.eq`); `core::arch::wasm32` names it both ways.
pub use self::i32x4_eq as u32x4_eq;

#[inline]
pub fn i32x4_ne(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i32(a), get_i32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] != b[i])))
}

#[inline]
pub fn i32x4_lt(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i32(a), get_i32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] < b[i])))
}

#[inline]
pub fn i32x4_ge(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i32(a), get_i32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] >= b[i])))
}

#[inline]
pub fn u32x4_lt(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u32(a), get_u32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] < b[i])))
}

#[inline]
pub fn u32x4_ge(a: v128, b: v128) -> v128 {
    let (a, b) = (get_u32(a), get_u32(b));
    put_i32([0, 1, 2, 3].map(|i| mask!(a[i] >= b[i])))
}

/// Shift left by `amt mod 32`.
#[inline]
pub fn i32x4_shl(a: v128, amt: u32) -> v128 {
    put_u32(get_u32(a).map(|x| x << (amt & 31)))
}

/// Arithmetic shift right by `amt mod 32`.
#[inline]
pub fn i32x4_shr(a: v128, amt: u32) -> v128 {
    put_i32(get_i32(a).map(|x| x >> (amt & 31)))
}

/// Logical shift right by `amt mod 32`.
#[inline]
pub fn u32x4_shr(a: v128, amt: u32) -> v128 {
    put_u32(get_u32(a).map(|x| x >> (amt & 31)))
}

#[inline]
pub fn i32x4_all_true(a: v128) -> bool {
    get_i32(a).iter().all(|&x| x != 0)
}

/// Bit `i` of the result is the sign bit of lane `i`.
#[inline]
pub fn i32x4_bitmask(a: v128) -> u8 {
    let l = get_i32(a);
    ((l[0] < 0) as u8)
        | (((l[1] < 0) as u8) << 1)
        | (((l[2] < 0) as u8) << 2)
        | (((l[3] < 0) as u8) << 3)
}

/// `a[2i]*b[2i] + a[2i+1]*b[2i+1]` in 32 bits; only all-`-32768` inputs wrap (to `i32::MIN`).
#[inline]
pub fn i32x4_dot_i16x8(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i16(a), get_i16(b));
    put_i32(core::array::from_fn(|i| {
        (a[2 * i] as i32 * b[2 * i] as i32).wrapping_add(a[2 * i + 1] as i32 * b[2 * i + 1] as i32)
    }))
}

#[inline]
pub fn i32x4_extend_low_i16x8(a: v128) -> v128 {
    let a = get_i16(a);
    put_i32(core::array::from_fn(|i| a[i] as i32))
}

#[inline]
pub fn i32x4_extend_high_i16x8(a: v128) -> v128 {
    let a = get_i16(a);
    put_i32(core::array::from_fn(|i| a[i + 4] as i32))
}

#[inline]
pub fn u32x4_extend_low_u16x8(a: v128) -> v128 {
    let a = get_u16(a);
    put_u32(core::array::from_fn(|i| a[i] as u32))
}

#[inline]
pub fn u32x4_extend_high_u16x8(a: v128) -> v128 {
    let a = get_u16(a);
    put_u32(core::array::from_fn(|i| a[i + 4] as u32))
}

#[inline]
pub fn i32x4_extmul_low_i16x8(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i16(a), get_i16(b));
    put_i32(core::array::from_fn(|i| a[i] as i32 * b[i] as i32))
}

#[inline]
pub fn i32x4_extmul_high_i16x8(a: v128, b: v128) -> v128 {
    let (a, b) = (get_i16(a), get_i16(b));
    put_i32(core::array::from_fn(|i| a[i + 4] as i32 * b[i + 4] as i32))
}

#[inline]
pub fn i32x4_extadd_pairwise_i16x8(a: v128) -> v128 {
    let a = get_i16(a);
    put_i32(core::array::from_fn(|i| {
        a[2 * i] as i32 + a[2 * i + 1] as i32
    }))
}

#[inline]
pub fn u32x4_extadd_pairwise_u16x8(a: v128) -> v128 {
    let a = get_u16(a);
    put_u32(core::array::from_fn(|i| {
        a[2 * i] as u32 + a[2 * i + 1] as u32
    }))
}

// ---------------------------------------------------------------------------------------------
// Shuffles: each index selects a lane of the concatenation `a ++ b` (indices >= lanes pick `b`).
// ---------------------------------------------------------------------------------------------

#[inline]
fn shuffle_lanes<T: Copy, const N: usize>(a: [T; N], b: [T; N], idx: [usize; N]) -> [T; N] {
    core::array::from_fn(|i| if idx[i] < N { a[idx[i]] } else { b[idx[i] - N] })
}

#[inline]
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
    let idx = [
        I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15,
    ];
    put_u8(shuffle_lanes(get_u8(a), get_u8(b), idx))
}

#[inline]
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
    put_u16(shuffle_lanes(
        get_u16(a),
        get_u16(b),
        [I0, I1, I2, I3, I4, I5, I6, I7],
    ))
}

#[inline]
pub fn i32x4_shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
    a: v128,
    b: v128,
) -> v128 {
    const { assert!(I0 < 8 && I1 < 8 && I2 < 8 && I3 < 8) };
    put_u32(shuffle_lanes(get_u32(a), get_u32(b), [I0, I1, I2, I3]))
}

#[inline]
pub fn i64x2_shuffle<const I0: usize, const I1: usize>(a: v128, b: v128) -> v128 {
    const { assert!(I0 < 4 && I1 < 4) };
    put_u64(shuffle_lanes(get_u64(a), get_u64(b), [I0, I1]))
}

// ---------------------------------------------------------------------------------------------
// Whole-register bitwise ops.
// ---------------------------------------------------------------------------------------------

#[inline]
pub fn v128_and(a: v128, b: v128) -> v128 {
    lanewise!(get_u64 => put_u64, 2, |a, b| a & b)
}

/// `a & !b`.
#[inline]
pub fn v128_andnot(a: v128, b: v128) -> v128 {
    lanewise!(get_u64 => put_u64, 2, |a, b| a & !b)
}

#[inline]
pub fn v128_or(a: v128, b: v128) -> v128 {
    lanewise!(get_u64 => put_u64, 2, |a, b| a | b)
}

#[inline]
pub fn v128_not(a: v128) -> v128 {
    lanewise!(get_u64 => put_u64, 2, |a| !a)
}

/// Per bit: `c ? v1 : v2`.
#[inline]
pub fn v128_bitselect(v1: v128, v2: v128, c: v128) -> v128 {
    lanewise!(get_u64 => put_u64, 2, |v1, v2, c| (v1 & c) | (v2 & !c))
}

#[inline]
pub fn v128_any_true(a: v128) -> bool {
    a.0.iter().any(|&b| b != 0)
}

// ---------------------------------------------------------------------------------------------
// Loads and stores (unaligned, as in wasm).
// ---------------------------------------------------------------------------------------------

/// # Safety
/// `m` must be valid for a 16-byte read; no alignment is required.
#[inline]
pub unsafe fn v128_load(m: *const v128) -> v128 {
    core::ptr::read_unaligned(m)
}

/// Loads 32 bits into lane 0 and zeroes the rest.
///
/// # Safety
/// `m` must be valid for a 4-byte read; no alignment is required.
#[inline]
pub unsafe fn v128_load32_zero(m: *const u32) -> v128 {
    put_u32([core::ptr::read_unaligned(m), 0, 0, 0])
}

/// Loads 64 bits into lane 0 and zeroes the rest.
///
/// # Safety
/// `m` must be valid for an 8-byte read; no alignment is required.
#[inline]
pub unsafe fn v128_load64_zero(m: *const u64) -> v128 {
    put_u64([core::ptr::read_unaligned(m), 0])
}

/// # Safety
/// `m` must be valid for a 16-byte write; no alignment is required.
#[inline]
pub unsafe fn v128_store(m: *mut v128, a: v128) {
    core::ptr::write_unaligned(m, a)
}

/// Stores 32-bit lane `L`.
///
/// # Safety
/// `m` must be valid for a 4-byte write; no alignment is required.
#[inline]
pub unsafe fn v128_store32_lane<const L: usize>(v: v128, m: *mut u32) {
    const { assert!(L < 4) };
    core::ptr::write_unaligned(m, get_u32(v)[L])
}

/// Stores 64-bit lane `L`.
///
/// # Safety
/// `m` must be valid for an 8-byte write; no alignment is required.
#[inline]
pub unsafe fn v128_store64_lane<const L: usize>(v: v128, m: *mut u64) {
    const { assert!(L < 2) };
    core::ptr::write_unaligned(m, get_u64(v)[L])
}

// ---------------------------------------------------------------------------------------------
// Tests of the scalar math helpers against `std` (host only).
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    fn same_f32(a: f32, b: f32) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    fn same_f64(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    /// Pattern stride of the sampled f32 tests. Miri interprets every operation, so under Miri
    /// (`cargo miri test`, see docs/VERIFICATION_AND_BENCHMARKS.md) the sweeps are thinned to
    /// about 1,700 inputs; they still cross every exponent range and every dense window.
    const STEP: u32 = if cfg!(miri) { 4_000_037 } else { 9973 };
    /// Length of each dense window in [`f32_patterns`].
    const WINDOW: u32 = if cfg!(miri) { 0x80 } else { 0x1000 };
    /// Random inputs of `sqrt_f64_matches_std_correct_rounding`.
    const F64_SAMPLES: u64 = if cfg!(miri) { 4_000 } else { 2_000_000 };

    /// Visits every `step`-th f32 bit pattern, plus dense windows around subnormals, 0.5, 1.0
    /// and 2^23 (where rounding and normalisation change behaviour).
    fn f32_patterns(step: u32, mut f: impl FnMut(f32)) {
        let mut bits = 0u32;
        loop {
            f(f32::from_bits(bits));
            match bits.checked_add(step) {
                Some(b) => bits = b,
                None => break,
            }
        }
        for b in 0..=WINDOW {
            f(f32::from_bits(b)); // smallest subnormals
            f(f32::from_bits(b | 0x8000_0000));
            f(f32::from_bits(0x3F80_0000 + b)); // just above 1.0
            f(f32::from_bits(0x3F00_0000 - b)); // just below 0.5
            f(f32::from_bits(0x4B00_0000 - b)); // just below 2^23
        }
    }

    #[test]
    fn sqrt_f32_matches_std_correct_rounding() {
        f32_patterns(STEP, |x| {
            assert!(
                same_f32(sqrt_f32(x), x.sqrt()),
                "sqrt({x:e}) bits {:#010x}",
                x.to_bits()
            );
        });
    }

    /// All 2^32 inputs of `f32` sqrt, floor and nearest against `std`. About 2.5 minutes in
    /// release, plus the fat-LTO compile:
    /// `cargo test --release --target host-tuple -- --ignored exhaustive`.
    #[test]
    #[ignore]
    fn exhaustive_f32_sqrt_floor_nearest_match_std() {
        f32_patterns(1, |x| {
            assert!(
                same_f32(sqrt_f32(x), x.sqrt()),
                "sqrt bits {:#010x}",
                x.to_bits()
            );
            assert!(
                same_f32(floor_f32(x), x.floor()),
                "floor bits {:#010x}",
                x.to_bits()
            );
            assert!(
                same_f32(nearest_f32(x), x.round_ties_even()),
                "nearest bits {:#010x}",
                x.to_bits()
            );
        });
    }

    #[test]
    fn sqrt_f64_matches_std_correct_rounding() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for i in 0..F64_SAMPLES {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = match i % 4 {
                0 => f64::from_bits(s),
                1 => f64::from_bits(s >> 12), // subnormals and tiny values
                2 => (s >> 11) as f64,        // integers, incl. perfect squares
                _ => ((s >> 40) as f64).powi(2),
            };
            assert!(
                same_f64(sqrt_f64(x), x.sqrt()),
                "sqrt({x:e}) bits {:#018x}",
                x.to_bits()
            );
        }
        for x in [
            0.0,
            -0.0,
            1.0,
            2.0,
            4.0,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            f64::INFINITY,
        ] {
            assert!(same_f64(sqrt_f64(x), x.sqrt()));
        }
        assert!(sqrt_f64(-1.0).is_nan() && sqrt_f64(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn floor_and_nearest_match_std() {
        f32_patterns(STEP, |x| {
            assert!(same_f32(floor_f32(x), x.floor()), "floor({x:e})");
            assert!(
                same_f32(nearest_f32(x), x.round_ties_even()),
                "nearest({x:e})"
            );
        });
    }

    #[test]
    fn min_max_follow_wasm_zero_and_nan_rules() {
        assert_eq!(wasm_fmin(0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(wasm_fmin(-0.0, 0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(wasm_fmax(0.0, -0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(wasm_fmax(-0.0, 0.0).to_bits(), 0.0f32.to_bits());
        assert!(wasm_fmin(f32::NAN, 1.0).is_nan() && wasm_fmin(1.0, f32::NAN).is_nan());
        assert!(wasm_fmax(f32::NAN, 1.0).is_nan() && wasm_fmax(1.0, f32::NAN).is_nan());
    }
}
