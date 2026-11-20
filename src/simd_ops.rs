#![allow(dead_code)]

use crate::simd::*;

/// `[a[0], a[1], a[2], a[3]]` as one unaligned vector load. `f32x4(a[0], a[1], a[2], a[3])` is
/// the same vector, and aarch64 already compiles it to one load, but LLVM's wasm32 backend builds
/// it lane by lane (an `f32.load` and an `f32x4.replace_lane` per element after the first).
#[inline(always)]
pub(crate) fn load_f32x4(a: &[f32; 4]) -> v128 {
    // SAFETY: `a` is 16 valid, initialised bytes; `v128_load` is an unaligned load on every
    // backend (wasm `v128.load`, `read_unaligned`, `_mm_loadu_si128`); any bits are a valid `v128`.
    unsafe { v128_load(a.as_ptr() as *const v128) }
}

/// Stores the four lanes of `v` to `a` with one unaligned vector store (the counterpart of
/// `load_f32x4`; storing the four `f32x4_extract_lane` values is the same bytes).
#[inline(always)]
pub(crate) fn store_f32x4(a: &mut [f32; 4], v: v128) {
    // SAFETY: `a` is 16 writable bytes; `v128_store` is an unaligned store on every backend.
    unsafe { v128_store(a.as_mut_ptr() as *mut v128, v) }
}

#[inline(always)]
pub fn morton2d_4x4(bx: usize, by: usize) -> usize {
    let px = (bx & 1) | ((bx & 2) << 1);
    let py = (by & 1) | ((by & 2) << 1);
    px | (py << 1)
}

#[inline(always)]
pub unsafe fn hsum_f32x4(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

#[inline(always)]
unsafe fn hsum4_f32x4(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let t0 = f32x4_add(i32x4_shuffle::<0, 1, 4, 5>(a, b), i32x4_shuffle::<2, 3, 6, 7>(a, b));
    let t1 = f32x4_add(i32x4_shuffle::<0, 1, 4, 5>(c, d), i32x4_shuffle::<2, 3, 6, 7>(c, d));
    f32x4_add(i32x4_shuffle::<0, 2, 4, 6>(t0, t1), i32x4_shuffle::<1, 3, 5, 7>(t0, t1))
}

#[inline(always)]
fn hsum_i32x4(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v)
        + i32x4_extract_lane::<1>(v)
        + i32x4_extract_lane::<2>(v)
        + i32x4_extract_lane::<3>(v)
}

/// Multi-scale SIMD128 4-lane loader:
/// - `scale_px == 1` (SD 16x16): single `v128_load` of 4 horizontally adjacent texels.
/// - `scale_px == 2` (HD 32x32): exact 2x2 box-filtered average over an 8x2 pixel block.
#[inline(always)]
pub unsafe fn load_sum4_2x2_f32(ptr: *const f32, stride: usize) -> v128 {
    let r0a = v128_load(ptr as *const v128);
    let r0b = v128_load(ptr.add(4) as *const v128);
    let r1a = v128_load(ptr.add(stride) as *const v128);
    let r1b = v128_load(ptr.add(stride + 4) as *const v128);
    let s0 = f32x4_add(r0a, r1a);
    let s1 = f32x4_add(r0b, r1b);
    let evens = i32x4_shuffle::<0, 2, 4, 6>(s0, s1);
    let odds = i32x4_shuffle::<1, 3, 5, 7>(s0, s1);
    f32x4_add(evens, odds)
}

#[inline(always)]
pub unsafe fn load_pooled4_f32(ptr: *const f32, stride: usize, scale_px: usize) -> v128 {
    if scale_px <= 1 {
        v128_load(ptr as *const v128)
    } else {
        f32x4_mul(load_sum4_2x2_f32(ptr, stride), f32x4_splat(0.25))
    }
}

/// Hardware binary-prefix 2-adic ultrametric valuation using wasm32 `i32.clz`.
/// Enforces the strong triangle inequality: d_2(A, C) <= max(d_2(A, B), d_2(B, C))
/// where d_2(A, B) = 2^(-padic_ultrametric_val(A, B)) for A != B and 0 for A == B.
#[no_mangle]
#[inline(never)]
pub extern "C" fn padic_ultrametric_val(code_a: u32, code_b: u32) -> u32 {
    (code_a ^ code_b).leading_zeros()
}

#[inline(always)]
pub fn padic_distance_from_val(val: u32) -> f32 {
    if val >= 32 {
        0.0
    } else {
        // Exact IEEE-754 power of two: 2.0f32^(-val)
        let exp_bits = (127u32.saturating_sub(val)) << 23;
        f32::from_bits(exp_bits)
    }
}

/// Unpack 320x180 RGBA8 buffer into normalized f32 Luma, Chroma R-G, and Chroma B-Y planes
/// using 4-pixel-wide v128 SIMD128 operations.
#[inline(never)]
pub unsafe fn rgba_to_planar_simd128(
    rgba_ptr: *const u8,
    luma_ptr: *mut f32,
    rg_ptr: *mut f32,
    by_ptr: *mut f32,
    pixels: usize,
) {
    let mask_ff = u32x4_splat(0xFF);
    let inv_255 = f32x4_splat(1.0 / 255.0);
    let wr = f32x4_splat(0.299);
    let wg = f32x4_splat(0.587);
    let wb = f32x4_splat(0.114);

    let mut i = 0usize;
    while i + 4 <= pixels {
        let px = v128_load(rgba_ptr.add(i * 4) as *const v128);

        let r_u32 = v128_and(px, mask_ff);
        let g_u32 = v128_and(u32x4_shr(px, 8), mask_ff);
        let b_u32 = v128_and(u32x4_shr(px, 16), mask_ff);

        let r = f32x4_mul(f32x4_convert_u32x4(r_u32), inv_255);
        let g = f32x4_mul(f32x4_convert_u32x4(g_u32), inv_255);
        let b = f32x4_mul(f32x4_convert_u32x4(b_u32), inv_255);

        let y_rg = f32x4_add(f32x4_mul(wr, r), f32x4_mul(wg, g));
        let y = f32x4_add(y_rg, f32x4_mul(wb, b));
        let rg = f32x4_sub(r, g);
        let by = f32x4_sub(b, y);

        v128_store(luma_ptr.add(i) as *mut v128, y);
        v128_store(rg_ptr.add(i) as *mut v128, rg);
        v128_store(by_ptr.add(i) as *mut v128, by);

        i += 4;
    }
}

/// On-demand horizontal run of `n_tiles` contiguous `16x16` (clamped at frame bottom) tiles
/// RGBA8 -> planar `chroma_rg`, `chroma_by` using precomputed `luma_ptr` (`rg = (r - g)/255`,
/// `by = b/255 - y`, bit-identical to planar pass).
#[inline(always)]
pub unsafe fn rgba_tiles_run_to_chroma_simd128(
    rgba_ptr: *const u8,
    luma_ptr: *const f32,
    rg_ptr: *mut f32,
    by_ptr: *mut f32,
    stride: usize,
    x0: usize,
    n_tiles: usize,
    y0: usize,
    y_end: usize,
) {
    let inv_255 = f32x4_splat(1.0 / 255.0);
    for y in y0..y_end {
        let mut row = y * stride + x0;
        for _ in 0..n_tiles {
            let (r16, g16, b16) = deinterleave_rgb16_simd128(rgba_ptr.add(row * 4));
            let rg_lo = i16x8_sub(u16x8_extend_low_u8x16(r16), u16x8_extend_low_u8x16(g16));
            let rg_hi = i16x8_sub(u16x8_extend_high_u8x16(r16), u16x8_extend_high_u8x16(g16));
            v128_store(
                rg_ptr.add(row) as *mut v128,
                f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(rg_lo)), inv_255),
            );
            v128_store(
                rg_ptr.add(row + 4) as *mut v128,
                f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(rg_lo)), inv_255),
            );
            v128_store(
                rg_ptr.add(row + 8) as *mut v128,
                f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(rg_hi)), inv_255),
            );
            v128_store(
                rg_ptr.add(row + 12) as *mut v128,
                f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(rg_hi)), inv_255),
            );

            let b_lo = u16x8_extend_low_u8x16(b16);
            let b_hi = u16x8_extend_high_u8x16(b16);
            v128_store(
                by_ptr.add(row) as *mut v128,
                f32x4_sub(
                    f32x4_mul(f32x4_convert_u32x4(u32x4_extend_low_u16x8(b_lo)), inv_255),
                    v128_load(luma_ptr.add(row) as *const v128),
                ),
            );
            v128_store(
                by_ptr.add(row + 4) as *mut v128,
                f32x4_sub(
                    f32x4_mul(f32x4_convert_u32x4(u32x4_extend_high_u16x8(b_lo)), inv_255),
                    v128_load(luma_ptr.add(row + 4) as *const v128),
                ),
            );
            v128_store(
                by_ptr.add(row + 8) as *mut v128,
                f32x4_sub(
                    f32x4_mul(f32x4_convert_u32x4(u32x4_extend_low_u16x8(b_hi)), inv_255),
                    v128_load(luma_ptr.add(row + 8) as *const v128),
                ),
            );
            v128_store(
                by_ptr.add(row + 12) as *mut v128,
                f32x4_sub(
                    f32x4_mul(f32x4_convert_u32x4(u32x4_extend_high_u16x8(b_hi)), inv_255),
                    v128_load(luma_ptr.add(row + 12) as *const v128),
                ),
            );
            row += 16;
        }
    }
}

/// On-demand single `16x16` (clamped at frame bottom) tile RGBA8 -> planar `chroma_rg`, `chroma_by`
/// using precomputed `luma_ptr` (`rg = (r - g)/255`, `by = b/255 - y`, bit-identical to planar pass).
#[inline(always)]
pub unsafe fn rgba_tile16_to_chroma_simd128(
    rgba_ptr: *const u8,
    luma_ptr: *const f32,
    rg_ptr: *mut f32,
    by_ptr: *mut f32,
    stride: usize,
    x0: usize,
    y0: usize,
    y_end: usize,
) {
    rgba_tiles_run_to_chroma_simd128(rgba_ptr, luma_ptr, rg_ptr, by_ptr, stride, x0, 1, y0, y_end);
}

/// Fused single-pass RGBA8 -> planar `luma` unpack AND `160x90` box-pooled pyramid
/// construction in registers (`chroma_rg`/`chroma_by` deferred to lazy per-16x16-tile fill),
/// returning sampled `sum(y^2)` at `k % 16 == 0` for exposure-RMS estimation.
#[inline(never)]
pub unsafe fn rgba_to_planar_and_pool160_simd128(
    rgba_ptr: *const u8,
    luma_ptr: *mut f32,
    rg_ptr: *mut f32,
    by_ptr: *mut f32,
    pyr_luma_ptr: *mut f32,
    pyr_rg_ptr: *mut f32,
    pyr_by_ptr: *mut f32,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
) -> f32 {
    let dot_luma = i16x8(9798, 19235, 3735, 0, 9798, 19235, 3735, 0);
    let dot_rgb_fold = i16x8(1, -1, 1, 0, 1, -1, 1, 0);
    let inv_q15_255 = f32x4_splat(1.0 / (32768.0 * 255.0));

    macro_rules! unpack_luma_i16_4 {
        ($off:expr) => {{
            let px = v128_load(rgba_ptr.add($off * 4) as *const v128);
            let p01 = u16x8_extend_low_u8x16(px);
            let p23 = u16x8_extend_high_u8x16(px);
            let d01 = i32x4_dot_i16x8(p01, dot_luma);
            let d23 = i32x4_dot_i16x8(p23, dot_luma);
            let y_i = i32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(d01, d23),
                i32x4_shuffle::<1, 3, 5, 7>(d01, d23),
            );
            let y = f32x4_mul(f32x4_convert_i32x4(y_i), inv_q15_255);
            v128_store(luma_ptr.add($off) as *mut v128, y);
            (y, y_i, p01, p23)
        }};
    }

    macro_rules! horiz_pair2_i32 {
        ($a:expr, $b:expr) => {
            i32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>($a, $b),
                i32x4_shuffle::<1, 3, 5, 7>($a, $b),
            )
        };
    }

    macro_rules! sum2_rows_luma_col4 {
        ($off0:expr, $off1:expr) => {{
            let (y0, yi0, p01_0, p23_0) = unpack_luma_i16_4!($off0);
            let (_, yi1, p01_1, p23_1) = unpack_luma_i16_4!($off1);
            let v01 = i16x8_add(p01_0, p01_1);
            let v23 = i16x8_add(p23_0, p23_1);
            let s01 = i16x8_add(v01, i64x2_shuffle::<1, 0>(v01, v01));
            let s23 = i16x8_add(v23, i64x2_shuffle::<1, 0>(v23, v23));
            let rgb2 = i32x4_dot_i16x8(i64x2_shuffle::<0, 2>(s01, s23), dot_rgb_fold);
            (y0, i32x4_add(yi0, yi1), rgb2)
        }};
    }

    #[allow(unused_macros)]
    macro_rules! sum4_rows_luma_col4 {
        ($off:expr) => {{
            let o = $off;
            let (y0, yi0, p01_0, p23_0) = unpack_luma_i16_4!(o);
            let (y1, yi1, p01_1, p23_1) = unpack_luma_i16_4!(o + 640);
            let (y2, yi2, p01_2, p23_2) = unpack_luma_i16_4!(o + 1280);
            let (y3, yi3, p01_3, p23_3) = unpack_luma_i16_4!(o + 1920);
            let v01 = i16x8_add(i16x8_add(p01_0, p01_1), i16x8_add(p01_2, p01_3));
            let v23 = i16x8_add(i16x8_add(p23_0, p23_1), i16x8_add(p23_2, p23_3));
            let v4 = i16x8_add(v01, v23);
            let blk = i16x8_add(v4, i64x2_shuffle::<1, 0>(v4, v4));
            (
                y0,
                y1,
                y2,
                y3,
                i32x4_add(i32x4_add(yi0, yi1), i32x4_add(yi2, yi3)),
                blk,
            )
        }};
    }

    let mut y_sq_v = f32x4_splat(0.0);
    if scale_px <= 1 && frame_w == 320 && frame_h == 180 {
        let q_luma = f32x4_splat(1.0 / (4.0 * 32768.0 * 255.0));
        let q_255 = f32x4_splat(1.0 / (4.0 * 255.0));
        for py in 0..90usize {
            let r0 = py * 640;
            let r1 = r0 + 320;
            let pyr_row = py * 160;
            let mut px = 0usize;
            while px + 4 <= 160 {
                let off0 = r0 + px * 2;
                let off1 = r1 + px * 2;
                let (y0a, ya, rgb_ab) = sum2_rows_luma_col4!(off0, off1);
                if (px & 12) == 0 {
                    y_sq_v = f32x4_add(y_sq_v, f32x4_mul(y0a, y0a));
                }
                let (_, yb, rgb_cd) = sum2_rows_luma_col4!(off0 + 4, off1 + 4);
                let p = pyr_row + px;
                let pl = f32x4_mul(f32x4_convert_i32x4(horiz_pair2_i32!(ya, yb)), q_luma);
                let rg_i = i32x4_shuffle::<0, 2, 4, 6>(rgb_ab, rgb_cd);
                let b_i = i32x4_shuffle::<1, 3, 5, 7>(rgb_ab, rgb_cd);
                let prg = f32x4_mul(f32x4_convert_i32x4(rg_i), q_255);
                let pby = f32x4_sub(f32x4_mul(f32x4_convert_i32x4(b_i), q_255), pl);
                v128_store(pyr_luma_ptr.add(p) as *mut v128, pl);
                v128_store(pyr_rg_ptr.add(p) as *mut v128, prg);
                v128_store(pyr_by_ptr.add(p) as *mut v128, pby);
                px += 4;
            }
        }
        return hsum_f32x4(y_sq_v);
    }

    if scale_px == 2 && frame_w == 640 && frame_h == 360 {
        let s16_luma = f32x4_splat(1.0 / (16.0 * 32768.0 * 255.0));
        let s16_255 = f32x4_splat(1.0 / (16.0 * 255.0));
        macro_rules! unpack_hd_luma4 {
            ($off:expr, $add_ysq:expr) => {{
                let o = $off;
                let px = v128_load(rgba_ptr.add(o * 4) as *const v128);
                let p01 = u16x8_extend_low_u8x16(px);
                let p23 = u16x8_extend_high_u8x16(px);
                let d01 = i32x4_dot_i16x8(p01, dot_luma);
                let d23 = i32x4_dot_i16x8(p23, dot_luma);
                let y_i = i32x4_add(
                    i32x4_shuffle::<0, 2, 4, 6>(d01, d23),
                    i32x4_shuffle::<1, 3, 5, 7>(d01, d23),
                );
                let y = f32x4_mul(f32x4_convert_i32x4(y_i), inv_q15_255);
                v128_store(luma_ptr.add(o) as *mut v128, y);
                if $add_ysq {
                    y_sq_v = f32x4_add(y_sq_v, f32x4_mul(y, y));
                }
                (p01, p23)
            }};
        }
        macro_rules! sum4_hd_col4 {
            ($off:expr, $add_ysq:expr) => {{
                let o = $off;
                let (mut acc01, mut acc23) = unpack_hd_luma4!(o, $add_ysq);
                let (p01_1, p23_1) = unpack_hd_luma4!(o + 640, $add_ysq);
                acc01 = i16x8_add(acc01, p01_1);
                acc23 = i16x8_add(acc23, p23_1);
                let (p01_2, p23_2) = unpack_hd_luma4!(o + 1280, $add_ysq);
                acc01 = i16x8_add(acc01, p01_2);
                acc23 = i16x8_add(acc23, p23_2);
                let (p01_3, p23_3) = unpack_hd_luma4!(o + 1920, $add_ysq);
                acc01 = i16x8_add(acc01, p01_3);
                acc23 = i16x8_add(acc23, p23_3);
                let v4 = i16x8_add(acc01, acc23);
                i16x8_add(v4, i64x2_shuffle::<1, 0>(v4, v4))
            }};
        }
        macro_rules! pool_hd_col4 {
            ($r0:expr, $pyr_row:expr, $px:expr, $add_ysq:expr) => {{
                let off0 = $r0 + $px * 4;
                let ba = sum4_hd_col4!(off0, $add_ysq);
                let bb = sum4_hd_col4!(off0 + 4, false);
                let blk_ab = i64x2_shuffle::<0, 2>(ba, bb);
                let bc = sum4_hd_col4!(off0 + 8, false);
                let bd = sum4_hd_col4!(off0 + 12, false);
                let blk_cd = i64x2_shuffle::<0, 2>(bc, bd);
                let p = $pyr_row + $px;
                let pl = f32x4_mul(
                    f32x4_convert_i32x4(horiz_pair2_i32!(
                        i32x4_dot_i16x8(blk_ab, dot_luma),
                        i32x4_dot_i16x8(blk_cd, dot_luma)
                    )),
                    s16_luma,
                );
                let rgb_ab = i32x4_dot_i16x8(blk_ab, dot_rgb_fold);
                let rgb_cd = i32x4_dot_i16x8(blk_cd, dot_rgb_fold);
                let prg = f32x4_mul(
                    f32x4_convert_i32x4(i32x4_shuffle::<0, 2, 4, 6>(rgb_ab, rgb_cd)),
                    s16_255,
                );
                let pby = f32x4_sub(
                    f32x4_mul(
                        f32x4_convert_i32x4(i32x4_shuffle::<1, 3, 5, 7>(rgb_ab, rgb_cd)),
                        s16_255,
                    ),
                    pl,
                );
                v128_store(pyr_luma_ptr.add(p) as *mut v128, pl);
                v128_store(pyr_rg_ptr.add(p) as *mut v128, prg);
                v128_store(pyr_by_ptr.add(p) as *mut v128, pby);
            }};
        }
        for py in 0..90usize {
            let r0 = py * 2560;
            let pyr_row = py * 160;
            for px_blk in 0..10usize {
                let px = px_blk * 16;
                pool_hd_col4!(r0, pyr_row, px, true);
                pool_hd_col4!(r0, pyr_row, px + 4, false);
                pool_hd_col4!(r0, pyr_row, px + 8, false);
                pool_hd_col4!(r0, pyr_row, px + 12, false);
            }
        }
        return hsum_f32x4(y_sq_v);
    }

    let pixels = frame_w * frame_h;
    rgba_to_planar_simd128(rgba_ptr, luma_ptr, rg_ptr, by_ptr, pixels);
    pool_plane_to_160x90_simd128(luma_ptr, pyr_luma_ptr, scale_px);
    pool_plane_to_160x90_simd128(rg_ptr, pyr_rg_ptr, scale_px);
    pool_plane_to_160x90_simd128(by_ptr, pyr_by_ptr, scale_px);
    let mut k = 0usize;
    let mut y_sq = 0.0f32;
    while k < pixels {
        let v = *luma_ptr.add(k);
        y_sq += v * v;
        k += 16;
    }
    y_sq
}

/// Centered Rank-2 Guarded ECA Subspace Regression & Uncensored Evaluation:
/// Models static background + global illumination shift X = [b, 1]:
///   r = y - (a * b + c),  where (a, c) minimize ||y - (a * b + c)||_{2, Omega}^2
/// Computed via two-pass centered covariance/variance to avoid normal-equation cancellation.
/// Evaluates uncensored residual/signal energy strictly over `eval_mask_ptr` (defined from
/// Internal Pass-1 helper for Rank-2 Extensive Cancellation Algorithm (ECA):
/// computes optimal affine parameters `(alpha, offset_c)` over the stationary subset `Omega`
/// via single-pass anchor-shifted `f64x2` SIMD128 accumulators, or `None` on cold start (`raw_b2 < 1e-5`).
#[inline(always)]
pub unsafe fn eca_fit_affine_params_simd128(
    luma_ptr: *const f32,
    bg_luma_ptr: *const f32,
    eval_mask_ptr: *const u8,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *const u8,
    len: usize,
) -> Option<(f32, f32)> {
    eca_fit_affine_params_ex_simd128(
        luma_ptr,
        bg_luma_ptr,
        eval_mask_ptr,
        curr_rgba,
        prev_rgba,
        stat_streak_ptr,
        len,
        false,
    )
}

#[inline(never)]
pub unsafe fn eca_fit_affine_params_ex_simd128(
    luma_ptr: *const f32,
    bg_luma_ptr: *const f32,
    eval_mask_ptr: *const u8,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *const u8,
    len: usize,
    dense_swarm: bool,
) -> Option<(f32, f32)> {
    let guard_thresh = f32x4_splat(0.045);
    let guard_stat = f32x4_splat(0.15);
    let use_live = !curr_rgba.is_null() && !prev_rgba.is_null() && !stat_streak_ptr.is_null();

    let mut b_ref = 0.5f32;
    let mut y_ref = 0.5f32;
    if use_live {
        for k in 0..len {
            let bk = *bg_luma_ptr.add(k);
            let yk = *luma_ptr.add(k);
            let st = if *stat_streak_ptr.add(k) >= 11 {
                let b = k * 4;
                let max_d = (*curr_rgba.add(b))
                    .abs_diff(*prev_rgba.add(b))
                    .max((*curr_rgba.add(b + 1)).abs_diff(*prev_rgba.add(b + 1)))
                    .max((*curr_rgba.add(b + 2)).abs_diff(*prev_rgba.add(b + 2)));
                max_d <= 3
            } else {
                false
            };
            let g = if st { 0.15f32 } else { 0.045f32 };
            if (yk - bk).abs() < g {
                b_ref = bk;
                y_ref = yk;
                break;
            }
        }
    } else if !eval_mask_ptr.is_null() {
        for k in 0..len {
            let bk = *bg_luma_ptr.add(k);
            let yk = *luma_ptr.add(k);
            let g = if *eval_mask_ptr.add(k) != 0 {
                0.15f32
            } else {
                0.045f32
            };
            if (yk - bk).abs() < g {
                b_ref = bk;
                y_ref = yk;
                break;
            }
        }
    } else {
        for k in 0..len {
            let bk = *bg_luma_ptr.add(k);
            let yk = *luma_ptr.add(k);
            if (yk - bk).abs() < 0.045f32 {
                b_ref = bk;
                y_ref = yk;
                break;
            }
        }
    }
    let b_ref_vec = f32x4_splat(b_ref);
    let y_ref_vec = f32x4_splat(y_ref);

    let mut acc_raw_b2_0 = f32x4_splat(0.0);
    let mut acc_raw_b2_1 = f32x4_splat(0.0);
    let mut acc_cnt_i32_0 = i32x4_splat(0);
    let mut acc_cnt_i32_1 = i32x4_splat(0);
    let mut tot_u = 0.0f64;
    let mut tot_v = 0.0f64;
    let mut tot_uu = 0.0f64;
    let mut tot_uv = 0.0f64;
    let mut c_u0 = f32x4_splat(0.0);
    let mut c_v0 = f32x4_splat(0.0);
    let mut c_uu0 = f32x4_splat(0.0);
    let mut c_uv0 = f32x4_splat(0.0);
    let mut c_u1 = f32x4_splat(0.0);
    let mut c_v1 = f32x4_splat(0.0);
    let mut c_uu1 = f32x4_splat(0.0);
    let mut c_uv1 = f32x4_splat(0.0);
    let mut tail_raw_b2 = 0.0f32;
    let mut tail_cnt = 0i64;
    let mut tail_u = 0.0f64;
    let mut tail_v = 0.0f64;
    let mut tail_uu = 0.0f64;
    let mut tail_uv = 0.0f64;

    macro_rules! accum_omega_lane {
        ($lane:tt, $y:expr, $b:expr, $in_omega:expr) => {{
            let um = v128_and(f32x4_sub($b, b_ref_vec), $in_omega);
            let vm = v128_and(f32x4_sub($y, y_ref_vec), $in_omega);
            if $lane == 0 {
                acc_raw_b2_0 = f32x4_add(acc_raw_b2_0, f32x4_mul($b, $b));
                acc_cnt_i32_0 = i32x4_sub(acc_cnt_i32_0, $in_omega);
                c_u0 = f32x4_add(c_u0, um);
                c_v0 = f32x4_add(c_v0, vm);
                c_uu0 = f32x4_add(c_uu0, f32x4_mul(um, um));
                c_uv0 = f32x4_add(c_uv0, f32x4_mul(um, vm));
            } else {
                acc_raw_b2_1 = f32x4_add(acc_raw_b2_1, f32x4_mul($b, $b));
                acc_cnt_i32_1 = i32x4_sub(acc_cnt_i32_1, $in_omega);
                c_u1 = f32x4_add(c_u1, um);
                c_v1 = f32x4_add(c_v1, vm);
                c_uu1 = f32x4_add(c_uu1, f32x4_mul(um, um));
                c_uv1 = f32x4_add(c_uv1, f32x4_mul(um, vm));
            }
        }};
    }
    macro_rules! accum_f32x4_lane {
        ($lane:tt, $y:expr, $b:expr, $eff_guard:expr) => {{
            let pre_diff = f32x4_abs(f32x4_sub($y, $b));
            let in_omega = f32x4_lt(pre_diff, $eff_guard);
            accum_omega_lane!($lane, $y, $b, in_omega);
        }};
    }
    macro_rules! sum_f32x4_chunks {
        () => {{
            tot_u += hsum_f32x4(f32x4_add(c_u0, c_u1)) as f64;
            tot_v += hsum_f32x4(f32x4_add(c_v0, c_v1)) as f64;
            tot_uu += hsum_f32x4(f32x4_add(c_uu0, c_uu1)) as f64;
            tot_uv += hsum_f32x4(f32x4_add(c_uv0, c_uv1)) as f64;
        }};
    }
    macro_rules! flush_f32x4_chunk {
        () => {{
            sum_f32x4_chunks!();
            c_u0 = f32x4_splat(0.0);
            c_v0 = f32x4_splat(0.0);
            c_uu0 = f32x4_splat(0.0);
            c_uv0 = f32x4_splat(0.0);
            c_u1 = f32x4_splat(0.0);
            c_v1 = f32x4_splat(0.0);
            c_uu1 = f32x4_splat(0.0);
            c_uv1 = f32x4_splat(0.0);
        }};
    }

    // Single-pass anchor-shifted sufficient statistics over Omega (staggered coprime diagonal lattice for SD/HD)
    let use_stride13 = dense_swarm && len >= 131_072;
    let use_stride7 = !use_stride13 && len >= 131_072;
    let use_stride3 = len >= 32768;
    let step16 = if use_stride13 {
        208usize
    } else if use_stride7 {
        112usize
    } else if use_stride3 {
        48usize
    } else {
        16usize
    };
    let step_scale = if use_stride13 {
        13i64
    } else if use_stride7 {
        7i64
    } else if use_stride3 {
        3i64
    } else {
        1i64
    };
    let mut i = 0usize;
    let mut blk_rem = 64usize;
    if use_live {
        let rgb_mask = u32x4_splat(0x00FF_FFFF);
        let tol3 = u8x16_splat(3);
        let zero_i32 = i32x4_splat(0);
        let eleven_u8 = u8x16_splat(11);
        while i + 16 <= len {
            let y0 = v128_load(luma_ptr.add(i) as *const v128);
            let b0 = v128_load(bg_luma_ptr.add(i) as *const v128);
            let y1 = v128_load(luma_ptr.add(i + 4) as *const v128);
            let b1 = v128_load(bg_luma_ptr.add(i + 4) as *const v128);
            let y2 = v128_load(luma_ptr.add(i + 8) as *const v128);
            let b2 = v128_load(bg_luma_ptr.add(i + 8) as *const v128);
            let y3 = v128_load(luma_ptr.add(i + 12) as *const v128);
            let b3 = v128_load(bg_luma_ptr.add(i + 12) as *const v128);

            let d0 = f32x4_abs(f32x4_sub(y0, b0));
            let d1 = f32x4_abs(f32x4_sub(y1, b1));
            let d2 = f32x4_abs(f32x4_sub(y2, b2));
            let d3 = f32x4_abs(f32x4_sub(y3, b3));

            let mut w0 = f32x4_lt(d0, guard_thresh);
            let mut w1 = f32x4_lt(d1, guard_thresh);
            let mut w2 = f32x4_lt(d2, guard_thresh);
            let mut w3 = f32x4_lt(d3, guard_thresh);

            let mid0 = v128_andnot(f32x4_lt(d0, guard_stat), w0);
            let mid1 = v128_andnot(f32x4_lt(d1, guard_stat), w1);
            let mid2 = v128_andnot(f32x4_lt(d2, guard_stat), w2);
            let mid3 = v128_andnot(f32x4_lt(d3, guard_stat), w3);
            let mid_or = v128_or(v128_or(mid0, mid1), v128_or(mid2, mid3));
            if v128_any_true(mid_or) {
                let old_s = v128_load(stat_streak_ptr.add(i) as *const v128);
                let ge11 = u8x16_ge(old_s, eleven_u8);
                let mid16 = i8x16_narrow_i16x8(
                    i16x8_narrow_i32x4(mid0, mid1),
                    i16x8_narrow_i32x4(mid2, mid3),
                );
                let cand16 = v128_and(mid16, ge11);
                if v128_any_true(cand16) {
                    let cb = i * 4;
                    let c0 = v128_load(curr_rgba.add(cb) as *const v128);
                    let p0 = v128_load(prev_rgba.add(cb) as *const v128);
                    let c1 = v128_load(curr_rgba.add(cb + 16) as *const v128);
                    let p1 = v128_load(prev_rgba.add(cb + 16) as *const v128);
                    let c2 = v128_load(curr_rgba.add(cb + 32) as *const v128);
                    let p2 = v128_load(prev_rgba.add(cb + 32) as *const v128);
                    let c3 = v128_load(curr_rgba.add(cb + 48) as *const v128);
                    let p3 = v128_load(prev_rgba.add(cb + 48) as *const v128);
                    let a0 = v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask);
                    let a1 = v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask);
                    let a2 = v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask);
                    let a3 = v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask);
                    let le0 = i32x4_eq(u8x16_gt(a0, tol3), zero_i32);
                    let le1 = i32x4_eq(u8x16_gt(a1, tol3), zero_i32);
                    let le2 = i32x4_eq(u8x16_gt(a2, tol3), zero_i32);
                    let le3 = i32x4_eq(u8x16_gt(a3, tol3), zero_i32);
                    let lo8 = i16x8_extend_low_i8x16(cand16);
                    let hi8 = i16x8_extend_high_i8x16(cand16);
                    w0 = v128_or(w0, v128_and(i32x4_extend_low_i16x8(lo8), le0));
                    w1 = v128_or(w1, v128_and(i32x4_extend_high_i16x8(lo8), le1));
                    w2 = v128_or(w2, v128_and(i32x4_extend_low_i16x8(hi8), le2));
                    w3 = v128_or(w3, v128_and(i32x4_extend_high_i16x8(hi8), le3));
                }
            }
            accum_omega_lane!(0, y0, b0, w0);
            accum_omega_lane!(1, y1, b1, w1);
            accum_omega_lane!(0, y2, b2, w2);
            accum_omega_lane!(1, y3, b3, w3);

            i += step16;
            blk_rem -= 1;
            if blk_rem == 0 {
                flush_f32x4_chunk!();
                blk_rem = 64;
            }
        }
        while i < len {
            let y = *luma_ptr.add(i);
            let b = *bg_luma_ptr.add(i);
            tail_raw_b2 += b * b;
            let is_stat = if *stat_streak_ptr.add(i) >= 11 {
                let cb = i * 4;
                let max_d = (*curr_rgba.add(cb))
                    .abs_diff(*prev_rgba.add(cb))
                    .max((*curr_rgba.add(cb + 1)).abs_diff(*prev_rgba.add(cb + 1)))
                    .max((*curr_rgba.add(cb + 2)).abs_diff(*prev_rgba.add(cb + 2)));
                max_d <= 3
            } else {
                false
            };
            let eff_guard = if is_stat { 0.15f32 } else { 0.045f32 };
            if (y - b).abs() < eff_guard {
                let u = (b - b_ref) as f64;
                let v = (y - y_ref) as f64;
                tail_cnt += 1;
                tail_u += u;
                tail_v += v;
                tail_uu += u * u;
                tail_uv += u * v;
            }
            i += 1;
        }
    } else if !eval_mask_ptr.is_null() {
        while i + 16 <= len {
            let raw_m16 = v128_load(eval_mask_ptr.add(i) as *const v128);
            let lo8 = i16x8_extend_low_i8x16(raw_m16);
            let hi8 = i16x8_extend_high_i8x16(raw_m16);

            let y0 = v128_load(luma_ptr.add(i) as *const v128);
            let b0 = v128_load(bg_luma_ptr.add(i) as *const v128);
            let m0 = i32x4_extend_low_i16x8(lo8);
            accum_f32x4_lane!(0, y0, b0, v128_bitselect(guard_stat, guard_thresh, m0));

            let y1 = v128_load(luma_ptr.add(i + 4) as *const v128);
            let b1 = v128_load(bg_luma_ptr.add(i + 4) as *const v128);
            let m1 = i32x4_extend_high_i16x8(lo8);
            accum_f32x4_lane!(1, y1, b1, v128_bitselect(guard_stat, guard_thresh, m1));

            let y2 = v128_load(luma_ptr.add(i + 8) as *const v128);
            let b2 = v128_load(bg_luma_ptr.add(i + 8) as *const v128);
            let m2 = i32x4_extend_low_i16x8(hi8);
            accum_f32x4_lane!(0, y2, b2, v128_bitselect(guard_stat, guard_thresh, m2));

            let y3 = v128_load(luma_ptr.add(i + 12) as *const v128);
            let b3 = v128_load(bg_luma_ptr.add(i + 12) as *const v128);
            let m3 = i32x4_extend_high_i16x8(hi8);
            accum_f32x4_lane!(1, y3, b3, v128_bitselect(guard_stat, guard_thresh, m3));

            i += step16;
            blk_rem -= 1;
            if blk_rem == 0 {
                flush_f32x4_chunk!();
                blk_rem = 64;
            }
        }
        while i + 4 <= len {
            let y = v128_load(luma_ptr.add(i) as *const v128);
            let b = v128_load(bg_luma_ptr.add(i) as *const v128);
            let raw_m4 = v128_load32_zero(eval_mask_ptr.add(i) as *const u32);
            let mask_vec = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(raw_m4));
            accum_f32x4_lane!(0, y, b, v128_bitselect(guard_stat, guard_thresh, mask_vec));
            i += 4;
        }
        while i < len {
            let y = *luma_ptr.add(i);
            let b = *bg_luma_ptr.add(i);
            tail_raw_b2 += b * b;
            let eff_guard = if *eval_mask_ptr.add(i) != 0 {
                0.15f32
            } else {
                0.045f32
            };
            if (y - b).abs() < eff_guard {
                let u = (b - b_ref) as f64;
                let v = (y - y_ref) as f64;
                tail_cnt += 1;
                tail_u += u;
                tail_v += v;
                tail_uu += u * u;
                tail_uv += u * v;
            }
            i += 1;
        }
    } else {
        while i + 16 <= len {
            let y0 = v128_load(luma_ptr.add(i) as *const v128);
            let b0 = v128_load(bg_luma_ptr.add(i) as *const v128);
            accum_f32x4_lane!(0, y0, b0, guard_thresh);
            let y1 = v128_load(luma_ptr.add(i + 4) as *const v128);
            let b1 = v128_load(bg_luma_ptr.add(i + 4) as *const v128);
            accum_f32x4_lane!(1, y1, b1, guard_thresh);
            let y2 = v128_load(luma_ptr.add(i + 8) as *const v128);
            let b2 = v128_load(bg_luma_ptr.add(i + 8) as *const v128);
            accum_f32x4_lane!(0, y2, b2, guard_thresh);
            let y3 = v128_load(luma_ptr.add(i + 12) as *const v128);
            let b3 = v128_load(bg_luma_ptr.add(i + 12) as *const v128);
            accum_f32x4_lane!(1, y3, b3, guard_thresh);

            i += step16;
            blk_rem -= 1;
            if blk_rem == 0 {
                flush_f32x4_chunk!();
                blk_rem = 64;
            }
        }
        while i + 4 <= len {
            let y = v128_load(luma_ptr.add(i) as *const v128);
            let b = v128_load(bg_luma_ptr.add(i) as *const v128);
            accum_f32x4_lane!(0, y, b, guard_thresh);
            i += 4;
        }
        while i < len {
            let y = *luma_ptr.add(i);
            let b = *bg_luma_ptr.add(i);
            tail_raw_b2 += b * b;
            if (y - b).abs() < 0.045f32 {
                let u = (b - b_ref) as f64;
                let v = (y - y_ref) as f64;
                tail_cnt += 1;
                tail_u += u;
                tail_v += v;
                tail_uu += u * u;
                tail_uv += u * v;
            }
            i += 1;
        }
    }
    sum_f32x4_chunks!();
    let acc_cnt_i32 = i32x4_add(acc_cnt_i32_0, acc_cnt_i32_1);

    let raw_b2 = hsum_f32x4(f32x4_add(acc_raw_b2_0, acc_raw_b2_1)) + tail_raw_b2;
    let n_cnt = (i32x4_extract_lane::<0>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<1>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<2>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<3>(acc_cnt_i32) as i64)
        + tail_cnt;
    tot_u += tail_u;
    tot_v += tail_v;
    tot_uu += tail_uu;
    tot_uv += tail_uv;

    eca_fit_affine_finish(
        raw_b2,
        n_cnt,
        step_scale,
        b_ref,
        y_ref,
        tot_u,
        tot_v,
        tot_uu,
        tot_uv,
    )
}

#[inline(always)]
fn eca_fit_affine_finish(
    raw_b2: f32,
    n_cnt: i64,
    step_scale: i64,
    b_ref: f32,
    y_ref: f32,
    tot_u: f64,
    tot_v: f64,
    tot_uu: f64,
    tot_uv: f64,
) -> Option<(f32, f32)> {
    if raw_b2 < 1e-5 {
        return None;
    }
    let tot_cnt = n_cnt as f64;
    let params = if n_cnt * step_scale >= 256 {
        let inv_n = 1.0 / tot_cnt;
        let mu = tot_u * inv_n;
        let mv = tot_v * inv_n;
        let mean_b = (b_ref as f64) + mu;
        let mean_y = (y_ref as f64) + mv;
        let vbb = (tot_uu - tot_cnt * mu * mu).max(0.0);
        let cby = tot_uv - tot_cnt * mu * mv;
        if vbb > 1e-3 {
            let a = (cby / vbb).clamp(0.5, 1.5);
            (a as f32, (mean_y - a * mean_b) as f32)
        } else {
            // Rank-1 constant offset fallback when background spatial variance is near-zero
            (1.0, (mean_y - mean_b) as f32)
        }
    } else {
        (1.0, 0.0)
    };
    Some(params)
}

/// prior-frame stationarity and target guard zones, independent of current residual `r`)
/// BEFORE adapting `b`.
#[inline(never)]
pub unsafe fn eca_subspace_project_masked_simd128(
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *const u8,
    eca_res_ptr: *mut f32,
    len: usize,
    adapt_rate: f32,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> f32 {
    let one_vec = f32x4_splat(1.0);
    let Some((alpha, offset_c)) = eca_fit_affine_params_simd128(
        luma_ptr,
        bg_luma_ptr,
        eval_mask_ptr,
        core::ptr::null(),
        core::ptr::null(),
        core::ptr::null(),
        len,
    ) else {
        // Cold-start initialization: seed background template `b` with initial frame `y`
        let zero = f32x4_splat(0.0);
        let mut k = 0usize;
        while k + 4 <= len {
            let y = v128_load(luma_ptr.add(k) as *const v128);
            v128_store(bg_luma_ptr.add(k) as *mut v128, y);
            v128_store(eca_res_ptr.add(k) as *mut v128, zero);
            k += 4;
        }
        while k < len {
            *bg_luma_ptr.add(k) = *luma_ptr.add(k);
            *eca_res_ptr.add(k) = 0.0;
            k += 1;
        }
        if !out_err2.is_null() {
            *out_err2 = 0.0;
        }
        if !out_sig2.is_null() {
            *out_sig2 = 0.0;
        }
        if !out_eval_count.is_null() {
            *out_eval_count = 0;
        }
        return -1.0;
    };

    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);
    let v_rate_eval = f32x4_splat(0.80);
    let v_eval_gate = f32x4_splat(0.053);

    let mut acc_eval_err2 = f32x4_splat(0.0);
    let mut acc_eval_sig2 = f32x4_splat(0.0);
    let mut acc_eval_cnt = f32x4_splat(0.0);
    let mut tail_eval_err2 = 0.0f32;
    let mut tail_eval_sig2 = 0.0f32;
    let mut tail_eval_cnt = 0u32;

    // Pass 3: Project orthogonal residual, measure uncensored error on prior `eval_mask_ptr`, then adapt `b`
    let mut j = 0usize;
    if !eval_mask_ptr.is_null() {
        while j + 4 <= len {
            let y = v128_load(luma_ptr.add(j) as *const v128);
            let b = v128_load(bg_luma_ptr.add(j) as *const v128);

            let proj = f32x4_add(f32x4_mul(v_alpha, b), v_c);
            let diff = f32x4_sub(y, proj);
            let res = f32x4_abs(diff);
            v128_store(eca_res_ptr.add(j) as *mut v128, res);

            // Branchless sign-extension of 4 mask bytes (0xFF / 0x00) -> 4 x 32-bit lane masks
            let raw_m4 = v128_load32_zero(eval_mask_ptr.add(j) as *const u32);
            let mask_vec = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(raw_m4));
            let m_diff = v128_and(diff, mask_vec);
            let m_sig = v128_and(y, mask_vec);
            acc_eval_err2 = f32x4_add(acc_eval_err2, f32x4_mul(m_diff, m_diff));
            acc_eval_sig2 = f32x4_add(acc_eval_sig2, f32x4_mul(m_sig, m_sig));
            acc_eval_cnt = f32x4_add(acc_eval_cnt, v128_and(one_vec, mask_vec));

            let is_bg = f32x4_lt(res, v_motion_gate);
            let is_eval_heal = v128_and(mask_vec, f32x4_lt(res, v_eval_gate));
            let rate = v128_bitselect(v_rate_eval, v128_bitselect(v_rate_bg, v_rate_fg, is_bg), is_eval_heal);
            let b_next = f32x4_add(b, f32x4_mul(rate, f32x4_sub(y, b)));
            v128_store(bg_luma_ptr.add(j) as *mut v128, b_next);

            j += 4;
        }
        while j < len {
            let y = *luma_ptr.add(j);
            let b = *bg_luma_ptr.add(j);
            let res = (y - (alpha * b + offset_c)).abs();
            *eca_res_ptr.add(j) = res;
            let eval_on = *eval_mask_ptr.add(j) != 0;
            if eval_on {
                tail_eval_err2 += (y - (alpha * b + offset_c)) * (y - (alpha * b + offset_c));
                tail_eval_sig2 += y * y;
                tail_eval_cnt += 1;
            }
            let rate = if eval_on && res < 0.053 {
                0.80
            } else if res < 0.045 {
                adapt_rate
            } else {
                adapt_rate * 0.008
            };
            *bg_luma_ptr.add(j) = b + rate * (y - b);
            j += 1;
        }
    } else {
        while j + 4 <= len {
            let y = v128_load(luma_ptr.add(j) as *const v128);
            let b = v128_load(bg_luma_ptr.add(j) as *const v128);

            let proj = f32x4_add(f32x4_mul(v_alpha, b), v_c);
            let diff = f32x4_sub(y, proj);
            let res = f32x4_abs(diff);
            v128_store(eca_res_ptr.add(j) as *mut v128, res);

            let is_bg = f32x4_lt(res, v_motion_gate);
            let rate = v128_bitselect(v_rate_bg, v_rate_fg, is_bg);
            let b_next = f32x4_add(b, f32x4_mul(rate, f32x4_sub(y, b)));
            v128_store(bg_luma_ptr.add(j) as *mut v128, b_next);

            j += 4;
        }
        while j < len {
            let y = *luma_ptr.add(j);
            let b = *bg_luma_ptr.add(j);
            let res = (y - (alpha * b + offset_c)).abs();
            *eca_res_ptr.add(j) = res;
            let rate = if res < 0.045 { adapt_rate } else { adapt_rate * 0.008 };
            *bg_luma_ptr.add(j) = b + rate * (y - b);
            j += 1;
        }
    }

    let err2 = hsum_f32x4(acc_eval_err2) + tail_eval_err2;
    let sig2 = hsum_f32x4(acc_eval_sig2) + tail_eval_sig2;
    let eval_count = (hsum_f32x4(acc_eval_cnt) as u32).saturating_add(tail_eval_cnt);
    if !out_err2.is_null() {
        *out_err2 = err2;
    }
    if !out_sig2.is_null() {
        *out_sig2 = sig2;
    }
    if !out_eval_count.is_null() {
        *out_eval_count = eval_count;
    }

    if eval_count >= 1024 && sig2 > 1e-3 {
        err2 / sig2
    } else {
        -1.0
    }
}

/// Internal SIMD128 wrapper preserving the unmasked `eca_subspace_project_simd128` signature.
pub unsafe fn eca_subspace_project_simd128(
    luma_ptr: *const f32,
    bg_basis_ptr: *mut f32,
    eca_res_ptr: *mut f32,
    len: usize,
    adapt_rate: f32,
) -> f32 {
    eca_subspace_project_masked_simd128(
        luma_ptr,
        bg_basis_ptr,
        core::ptr::null(),
        eca_res_ptr,
        len,
        adapt_rate,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    )
}

/// Pure SIMD128 16-pixel-wide temporal stationarity streak & independent clutter-evaluation mask.
/// Defines Omega_stat(t) strictly from raw frame-to-frame RGB invariance (max(|dR|,|dG|,|dB|) <= 3
/// for >= 12 consecutive frames), 100% independent of ECA subspace output `eca_res` or `bg_luma`.
pub unsafe fn update_stat_streak_and_mask_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    pixels: usize,
) {
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let tol = u8x16_splat(3);
    let zero_i32 = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let min_streak = u8x16_splat(12);

    let mut p = 0usize;
    while p + 16 <= pixels {
        let base = p * 4;
        let c0 = v128_load(curr_rgba.add(base) as *const v128);
        let p0 = v128_load(prev_rgba.add(base) as *const v128);
        let c1 = v128_load(curr_rgba.add(base + 16) as *const v128);
        let p1 = v128_load(prev_rgba.add(base + 16) as *const v128);
        let c2 = v128_load(curr_rgba.add(base + 32) as *const v128);
        let p2 = v128_load(prev_rgba.add(base + 32) as *const v128);
        let c3 = v128_load(curr_rgba.add(base + 48) as *const v128);
        let p3 = v128_load(prev_rgba.add(base + 48) as *const v128);

        let a0 = v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask);
        let a1 = v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask);
        let a2 = v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask);
        let a3 = v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask);

        let within16 = i8x16_narrow_i16x8(
            i16x8_narrow_i32x4(
                i32x4_eq(u8x16_sub_sat(a0, tol), zero_i32),
                i32x4_eq(u8x16_sub_sat(a1, tol), zero_i32),
            ),
            i16x8_narrow_i32x4(
                i32x4_eq(u8x16_sub_sat(a2, tol), zero_i32),
                i32x4_eq(u8x16_sub_sat(a3, tol), zero_i32),
            ),
        );

        let old_streak = v128_load(stat_streak_ptr.add(p) as *const v128);
        let new_streak = v128_and(u8x16_add_sat(old_streak, one_u8), within16);
        v128_store(stat_streak_ptr.add(p) as *mut v128, new_streak);

        let eval16 = u8x16_ge(new_streak, min_streak);
        v128_store(eval_mask_ptr.add(p) as *mut v128, eval16);

        p += 16;
    }
    while p < pixels {
        let b = p * 4;
        let dr = (*curr_rgba.add(b)).abs_diff(*prev_rgba.add(b));
        let dg = (*curr_rgba.add(b + 1)).abs_diff(*prev_rgba.add(b + 1));
        let db = (*curr_rgba.add(b + 2)).abs_diff(*prev_rgba.add(b + 2));
        let max_d = dr.max(dg).max(db);
        let old_s = *stat_streak_ptr.add(p);
        let ns = if max_d <= 3 { old_s.saturating_add(1) } else { 0 };
        *stat_streak_ptr.add(p) = ns;
        *eval_mask_ptr.add(p) = if ns >= 12 { 0xFF } else { 0 };
        p += 1;
    }
}

/// SIMD128 4-tap unsigned fixed-point (`1/16` px phase, `w00 + w10 + w01 + w11 = 256`) bilinear
/// RGBA8 bilinear loader returning `(lo_u16, hi_u16)` lanes (`u16` in `0..=255`) for 4 horizontally adjacent pixels.
#[inline(always)]
unsafe fn bilinear_rgba4_vecs_i16_simd128(
    p00: v128,
    p10: v128,
    p01: v128,
    p11: v128,
    w00_v: v128,
    w10_v: v128,
    w01_v: v128,
    w11_v: v128,
    round_v: v128,
) -> (v128, v128) {
    let lo = u16x8_shr(
        u16x8_add(
            u16x8_add(
                u16x8_mul(u16x8_extend_low_u8x16(p00), w00_v),
                u16x8_mul(u16x8_extend_low_u8x16(p10), w10_v),
            ),
            u16x8_add(
                u16x8_add(
                    u16x8_mul(u16x8_extend_low_u8x16(p01), w01_v),
                    u16x8_mul(u16x8_extend_low_u8x16(p11), w11_v),
                ),
                round_v,
            ),
        ),
        8,
    );
    let hi = u16x8_shr(
        u16x8_add(
            u16x8_add(
                u16x8_mul(u16x8_extend_high_u8x16(p00), w00_v),
                u16x8_mul(u16x8_extend_high_u8x16(p10), w10_v),
            ),
            u16x8_add(
                u16x8_add(
                    u16x8_mul(u16x8_extend_high_u8x16(p01), w01_v),
                    u16x8_mul(u16x8_extend_high_u8x16(p11), w11_v),
                ),
                round_v,
            ),
        ),
        8,
    );
    (lo, hi)
}

#[inline(always)]
unsafe fn bilinear_rgba4_i16_simd128(
    p00_ptr: *const u8,
    row_bytes: usize,
    w00_v: v128,
    w10_v: v128,
    w01_v: v128,
    w11_v: v128,
    round_v: v128,
) -> (v128, v128) {
    let p00 = v128_load(p00_ptr as *const v128);
    let p10 = v128_load(p00_ptr.add(4) as *const v128);
    let p01 = v128_load(p00_ptr.add(row_bytes) as *const v128);
    let p11 = v128_load(p00_ptr.add(row_bytes + 4) as *const v128);
    bilinear_rgba4_vecs_i16_simd128(p00, p10, p01, p11, w00_v, w10_v, w01_v, w11_v, round_v)
}

#[inline(always)]
unsafe fn abs_rgb_to_sum_f32x4_simd128(abs_rgb: v128) -> v128 {
    let sum_u16 = u16x8_extadd_pairwise_u8x16(abs_rgb);
    let sum_u32 = u32x4_extadd_pairwise_u16x8(sum_u16);
    f32x4_convert_u32x4(sum_u32)
}

#[inline(always)]
unsafe fn abs_rgb_to_sum_u16x4_simd128(abs_rgb: v128) -> v128 {
    let sum_u16 = u16x8_extadd_pairwise_u8x16(abs_rgb);
    let sum_u32 = u32x4_extadd_pairwise_u16x8(sum_u16);
    i16x8_narrow_i32x4(sum_u32, sum_u32)
}

#[inline(always)]
unsafe fn rgb_abs_sum_f32x4_simd128(c_px: v128, p_px: v128, rgb_mask: v128) -> v128 {
    let abs_rgb = v128_and(v128_or(u8x16_sub_sat(c_px, p_px), u8x16_sub_sat(p_px, c_px)), rgb_mask);
    abs_rgb_to_sum_f32x4_simd128(abs_rgb)
}

#[inline(always)]
unsafe fn bilinear_rgba4_vecs_with_span_exp_simd128(
    p00: v128,
    p10: v128,
    p01: v128,
    p11: v128,
    w00_v: v128,
    w10_v: v128,
    w01_v: v128,
    w11_v: v128,
    round_v: v128,
    exp_active: bool,
    dg_q15_v: v128,
    off_i16_v: v128,
) -> (v128, v128, v128) {
    let cell_max = u8x16_max(u8x16_max(p00, p10), u8x16_max(p01, p11));
    let cell_min = u8x16_min(u8x16_min(p00, p10), u8x16_min(p01, p11));
    let cell_span = u8x16_sub_sat(cell_max, cell_min);

    let lo = u16x8_shr(
        u16x8_add(
            u16x8_add(
                u16x8_mul(u16x8_extend_low_u8x16(p00), w00_v),
                u16x8_mul(u16x8_extend_low_u8x16(p10), w10_v),
            ),
            u16x8_add(
                u16x8_add(
                    u16x8_mul(u16x8_extend_low_u8x16(p01), w01_v),
                    u16x8_mul(u16x8_extend_low_u8x16(p11), w11_v),
                ),
                round_v,
            ),
        ),
        8,
    );
    let hi = u16x8_shr(
        u16x8_add(
            u16x8_add(
                u16x8_mul(u16x8_extend_high_u8x16(p00), w00_v),
                u16x8_mul(u16x8_extend_high_u8x16(p10), w10_v),
            ),
            u16x8_add(
                u16x8_add(
                    u16x8_mul(u16x8_extend_high_u8x16(p01), w01_v),
                    u16x8_mul(u16x8_extend_high_u8x16(p11), w11_v),
                ),
                round_v,
            ),
        ),
        8,
    );
    let raw = u8x16_narrow_i16x8(lo, hi);
    let adj = if exp_active {
        let lo_adj = i16x8_add(lo, i16x8_add(i16x8_q15mulr_sat(lo, dg_q15_v), off_i16_v));
        let hi_adj = i16x8_add(hi, i16x8_add(i16x8_q15mulr_sat(hi, dg_q15_v), off_i16_v));
        u8x16_narrow_i16x8(lo_adj, hi_adj)
    } else {
        raw
    };
    (raw, adj, cell_span)
}

#[inline(always)]
fn bilinear_weights_u16(pan_dx_f: f32, pan_dy_f: f32) -> (i32, i32, u16, u16, u16, u16) {
    let shift_x = -pan_dx_f;
    let shift_y = -pan_dy_f;
    let ix0 = f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(shift_x))) as i32;
    let iy0 = f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(shift_y))) as i32;
    let fx = (shift_x - (ix0 as f32)).clamp(0.0, 1.0);
    let fy = (shift_y - (iy0 as f32)).clamp(0.0, 1.0);
    let wx1 = ((fx * 16.0 + 0.5) as u16).min(16);
    let wx0 = 16 - wx1;
    let wy1 = ((fy * 16.0 + 0.5) as u16).min(16);
    let wy0 = 16 - wy1;
    (ix0, iy0, wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1)
}

#[inline(always)]
fn exposure_q15_params(exp_gain: f32, exp_off: f32) -> (bool, i16, i16) {
    let dg = (exp_gain - 1.0).clamp(-0.5, 0.5) * 32768.0;
    let dg_q15 = if dg >= 0.0 {
        (dg + 0.5) as i16
    } else {
        (dg - 0.5) as i16
    };
    let off_f = exp_off.clamp(-0.35, 0.35) * 255.0;
    let off_i16 = if off_f >= 0.0 {
        (off_f + 0.5) as i16
    } else {
        (off_f - 0.5) as i16
    };
    (dg_q15 != 0 || off_i16 != 0, dg_q15, off_i16)
}

#[inline(always)]
unsafe fn apply_exposure_rgba4_simd128(px: v128, dg_q15_v: v128, off_i16_v: v128) -> v128 {
    let lo = u16x8_extend_low_u8x16(px);
    let hi = u16x8_extend_high_u8x16(px);
    let lo_adj = i16x8_add(lo, i16x8_add(i16x8_q15mulr_sat(lo, dg_q15_v), off_i16_v));
    let hi_adj = i16x8_add(hi, i16x8_add(i16x8_q15mulr_sat(hi, dg_q15_v), off_i16_v));
    u8x16_narrow_i16x8(lo_adj, hi_adj)
}

#[inline(always)]
fn apply_exposure_u8_scalar(v: u8, dg_q15: i16, off_i16: i16) -> u8 {
    let base = v as i32;
    let delta = (base * (dg_q15 as i32) + 16384) >> 15;
    (base + delta + (off_i16 as i32)).clamp(0, 255) as u8
}

#[inline(always)]
fn catmull_rom_weights(t: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    )
}

#[inline(always)]
unsafe fn zero_u8_row_simd128(dst: *mut u8, len: usize) {
    core::slice::from_raw_parts_mut(dst, len).fill(0);
}

#[inline(always)]
unsafe fn shift_u8_row_in_bounds_simd128(dst: *mut u8, src: *const u8, len: usize, reverse: bool) {
    if !reverse {
        let mut i = 0usize;
        while i + 64 <= len {
            let v0 = v128_load(src.add(i) as *const v128);
            let v1 = v128_load(src.add(i + 16) as *const v128);
            let v2 = v128_load(src.add(i + 32) as *const v128);
            let v3 = v128_load(src.add(i + 48) as *const v128);
            v128_store(dst.add(i) as *mut v128, v0);
            v128_store(dst.add(i + 16) as *mut v128, v1);
            v128_store(dst.add(i + 32) as *mut v128, v2);
            v128_store(dst.add(i + 48) as *mut v128, v3);
            i += 64;
        }
        while i + 16 <= len {
            v128_store(dst.add(i) as *mut v128, v128_load(src.add(i) as *const v128));
            i += 16;
        }
        while i < len {
            *dst.add(i) = *src.add(i);
            i += 1;
        }
    } else {
        let mut rem = len;
        while rem >= 64 {
            let off = rem - 64;
            let v3 = v128_load(src.add(off + 48) as *const v128);
            let v2 = v128_load(src.add(off + 32) as *const v128);
            let v1 = v128_load(src.add(off + 16) as *const v128);
            let v0 = v128_load(src.add(off) as *const v128);
            v128_store(dst.add(off + 48) as *mut v128, v3);
            v128_store(dst.add(off + 32) as *mut v128, v2);
            v128_store(dst.add(off + 16) as *mut v128, v1);
            v128_store(dst.add(off) as *mut v128, v0);
            rem -= 64;
        }
        while rem >= 16 {
            let off = rem - 16;
            v128_store(dst.add(off) as *mut v128, v128_load(src.add(off) as *const v128));
            rem -= 16;
        }
        while rem > 0 {
            rem -= 1;
            *dst.add(rem) = *src.add(rem);
        }
    }
}

#[inline(always)]
unsafe fn copy_f32_row_simd128(dst: *mut f32, src: *const f32, len: usize) {
    let mut i = 0usize;
    while i + 16 <= len {
        let v0 = v128_load(src.add(i) as *const v128);
        let v1 = v128_load(src.add(i + 4) as *const v128);
        let v2 = v128_load(src.add(i + 8) as *const v128);
        let v3 = v128_load(src.add(i + 12) as *const v128);
        v128_store(dst.add(i) as *mut v128, v0);
        v128_store(dst.add(i + 4) as *mut v128, v1);
        v128_store(dst.add(i + 8) as *mut v128, v2);
        v128_store(dst.add(i + 12) as *mut v128, v3);
        i += 16;
    }
    while i + 4 <= len {
        v128_store(dst.add(i) as *mut v128, v128_load(src.add(i) as *const v128));
        i += 4;
    }
    while i < len {
        *dst.add(i) = *src.add(i);
        i += 1;
    }
}

#[inline(always)]
unsafe fn shift_f32_row_in_bounds_simd128(
    dst: *mut f32,
    src: *const f32,
    len: usize,
    reverse: bool,
    exp_f_active: bool,
    exp_gain: f32,
    exp_off: f32,
    gain_v: v128,
    off_v: v128,
    zero_f: v128,
    one_f: v128,
) {
    if !reverse {
        let mut i = 0usize;
        if exp_f_active {
            while i + 4 <= len {
                let v = v128_load(src.add(i) as *const v128);
                let out_v = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(v, gain_v), off_v)));
                v128_store(dst.add(i) as *mut v128, out_v);
                i += 4;
            }
            while i < len {
                *dst.add(i) = ((*src.add(i)) * exp_gain + exp_off).clamp(0.0, 1.0);
                i += 1;
            }
        } else {
            while i + 4 <= len {
                v128_store(dst.add(i) as *mut v128, v128_load(src.add(i) as *const v128));
                i += 4;
            }
            while i < len {
                *dst.add(i) = *src.add(i);
                i += 1;
            }
        }
    } else {
        let mut rem = len;
        if exp_f_active {
            while rem >= 4 {
                let off = rem - 4;
                let v = v128_load(src.add(off) as *const v128);
                let out_v = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(v, gain_v), off_v)));
                v128_store(dst.add(off) as *mut v128, out_v);
                rem -= 4;
            }
            while rem > 0 {
                rem -= 1;
                *dst.add(rem) = ((*src.add(rem)) * exp_gain + exp_off).clamp(0.0, 1.0);
            }
        } else {
            while rem >= 4 {
                let off = rem - 4;
                v128_store(dst.add(off) as *mut v128, v128_load(src.add(off) as *const v128));
                rem -= 4;
            }
            while rem > 0 {
                rem -= 1;
                *dst.add(rem) = *src.add(rem);
            }
        }
    }
}

#[repr(C)]
struct CatmullCtx {
    wx0_v: v128,
    wx1_v: v128,
    wx2_v: v128,
    wx3_v: v128,
    wy0_v: v128,
    wy1_v: v128,
    wy2_v: v128,
    wy3_v: v128,
    gain_v: v128,
    off_v: v128,
    zero_f: v128,
    one_f: v128,
    frame_w: usize,
    w_i: i32,
    ix0: i32,
    fx: f32,
    wx0: f32,
    wx1: f32,
    wx2: f32,
    wx3: f32,
    wy0: f32,
    wy1: f32,
    wy2: f32,
    wy3: f32,
    exp_gain: f32,
    exp_off: f32,
    vx_lo: usize,
    vx_hi: usize,
    x_lo_h: usize,
    x_lim_h: i32,
    x_lim8_h: i32,
    vx_lim8_v: i32,
    vx_lim_v: i32,
}

#[inline(always)]
unsafe fn horiz_catmull_row_ctx<const FX_ZERO: bool>(
    ctx: &CatmullCtx,
    src_row: *const f32,
    dst_row: *mut f32,
) {
    let vx_lo = ctx.vx_lo;
    let vx_hi = ctx.vx_hi;
    if vx_lo >= vx_hi {
        return;
    }
    let ix0 = ctx.ix0;
    if FX_ZERO {
        copy_f32_row_simd128(
            dst_row.add(vx_lo),
            src_row.add(((vx_lo as i32) + ix0) as usize),
            vx_hi - vx_lo,
        );
        return;
    }
    let fx = ctx.fx;
    let wx0 = ctx.wx0;
    let wx1 = ctx.wx1;
    let wx2 = ctx.wx2;
    let wx3 = ctx.wx3;
    let wx0_v = ctx.wx0_v;
    let wx1_v = ctx.wx1_v;
    let wx2_v = ctx.wx2_v;
    let wx3_v = ctx.wx3_v;
    let w_i = ctx.w_i;

    let x_lo = ctx.x_lo_h;
    let mut x = vx_lo;
    while x < x_lo {
        let sx = (x as i32) + ix0;
        let s = sx as usize;
        *dst_row.add(x) = (1.0 - fx) * (*src_row.add(s)) + fx * (*src_row.add(s + 1));
        x += 1;
    }

    let x_lim_i = ctx.x_lim_h;
    if (x as i32) <= x_lim_i {
        let x_vec_h = ((x_lo + 3) & !3).min(vx_hi);
        while x < x_vec_h {
            let sx = (x as i32) + ix0;
            let s = sx as usize;
            *dst_row.add(x) = if sx >= 1 && sx + 2 < w_i {
                wx0 * (*src_row.add(s - 1))
                    + wx1 * (*src_row.add(s))
                    + wx2 * (*src_row.add(s + 1))
                    + wx3 * (*src_row.add(s + 2))
            } else {
                (1.0 - fx) * (*src_row.add(s)) + fx * (*src_row.add(s + 1))
            };
            x += 1;
        }
    }
    if (x as i32) <= x_lim_i {
        let mut cur4 = v128_load(src_row.add((((x as i32) + ix0) as usize) - 1) as *const v128);
        let x_lim8_i = ctx.x_lim8_h;
        while (x as i32) <= x_lim8_i {
            let s = ((x as i32) + ix0) as usize;
            let next4 = v128_load(src_row.add(s + 3) as *const v128);
            let next8 = v128_load(src_row.add(s + 7) as *const v128);
            let v0_a = i32x4_shuffle::<1, 2, 3, 4>(cur4, next4);
            let vp1_a = i32x4_shuffle::<2, 3, 4, 5>(cur4, next4);
            let vp2_a = i32x4_shuffle::<3, 4, 5, 6>(cur4, next4);
            let h_a = f32x4_add(
                f32x4_add(f32x4_mul(cur4, wx0_v), f32x4_mul(v0_a, wx1_v)),
                f32x4_add(f32x4_mul(vp1_a, wx2_v), f32x4_mul(vp2_a, wx3_v)),
            );
            v128_store(dst_row.add(x) as *mut v128, h_a);

            let v0_b = i32x4_shuffle::<1, 2, 3, 4>(next4, next8);
            let vp1_b = i32x4_shuffle::<2, 3, 4, 5>(next4, next8);
            let vp2_b = i32x4_shuffle::<3, 4, 5, 6>(next4, next8);
            let h_b = f32x4_add(
                f32x4_add(f32x4_mul(next4, wx0_v), f32x4_mul(v0_b, wx1_v)),
                f32x4_add(f32x4_mul(vp1_b, wx2_v), f32x4_mul(vp2_b, wx3_v)),
            );
            v128_store(dst_row.add(x + 4) as *mut v128, h_b);

            cur4 = next8;
            x += 8;
        }
        while (x as i32) <= x_lim_i {
            let s = ((x as i32) + ix0) as usize;
            let next4 = v128_load(src_row.add(s + 3) as *const v128);
            let vm1 = cur4;
            let v0 = i32x4_shuffle::<1, 2, 3, 4>(cur4, next4);
            let vp1 = i32x4_shuffle::<2, 3, 4, 5>(cur4, next4);
            let vp2 = i32x4_shuffle::<3, 4, 5, 6>(cur4, next4);
            let h_val = f32x4_add(
                f32x4_add(f32x4_mul(vm1, wx0_v), f32x4_mul(v0, wx1_v)),
                f32x4_add(f32x4_mul(vp1, wx2_v), f32x4_mul(vp2, wx3_v)),
            );
            v128_store(dst_row.add(x) as *mut v128, h_val);
            cur4 = next4;
            x += 4;
        }
    }

    while x < vx_hi {
        let sx = (x as i32) + ix0;
        let s = sx as usize;
        *dst_row.add(x) = if sx >= 1 && sx + 2 < w_i {
            wx0 * (*src_row.add(s - 1))
                + wx1 * (*src_row.add(s))
                + wx2 * (*src_row.add(s + 1))
                + wx3 * (*src_row.add(s + 2))
        } else {
            (1.0 - fx) * (*src_row.add(s)) + fx * (*src_row.add(s + 1))
        };
        x += 1;
    }
}

#[inline(never)]
unsafe fn horiz_catmull_1d_row_ctx<const EXP_ACTIVE: bool, const IN_PLACE: bool>(
    ctx: &CatmullCtx,
    dst_row: *mut f32,
    src_row: *const f32,
    cur_row: *const f32,
) {
    let frame_w = ctx.frame_w;
    let vx_lo = ctx.vx_lo;
    let vx_hi = ctx.vx_hi;
    if vx_lo >= vx_hi {
        copy_f32_row_simd128(dst_row, cur_row, frame_w);
        return;
    }
    let ix0 = ctx.ix0;
    let fx = ctx.fx;
    let wx0 = ctx.wx0;
    let wx1 = ctx.wx1;
    let wx2 = ctx.wx2;
    let wx3 = ctx.wx3;
    let wx0_v = ctx.wx0_v;
    let wx1_v = ctx.wx1_v;
    let wx2_v = ctx.wx2_v;
    let wx3_v = ctx.wx3_v;
    let gain_v = ctx.gain_v;
    let off_v = ctx.off_v;
    let zero_f = ctx.zero_f;
    let one_f = ctx.one_f;
    let exp_gain = ctx.exp_gain;
    let exp_off = ctx.exp_off;
    let w_i = ctx.w_i;

    #[inline(always)]
    fn clamp_exp_s(val: f32, exp_gain: f32, exp_off: f32, exp_active: bool) -> f32 {
        let c = val.clamp(0.0, 1.0);
        if exp_active {
            (c * exp_gain + exp_off).clamp(0.0, 1.0)
        } else {
            c
        }
    }

    let x_lo = ctx.x_lo_h;
    let x_lim_i = ctx.x_lim_h;
    let x_lim8_i = ctx.x_lim8_h;

    if IN_PLACE {
        let mut head_vals = [0.0f32; 4];
        let head_n = x_lo.saturating_sub(vx_lo);
        for k in 0..head_n {
            let s = ((vx_lo + k) as i32 + ix0) as usize;
            let v = (1.0 - fx) * (*src_row.add(s)) + fx * (*src_row.add(s + 1));
            head_vals[k] = clamp_exp_s(v, exp_gain, exp_off, EXP_ACTIVE);
        }
        let has_simd = (x_lo as i32) <= x_lim_i;
        let mut cur4 = if has_simd {
            v128_load(src_row.add((((x_lo as i32) + ix0) as usize) - 1) as *const v128)
        } else {
            zero_f
        };
        for k in 0..vx_lo {
            *dst_row.add(k) = *cur_row.add(k);
        }
        for k in 0..head_n {
            *dst_row.add(vx_lo + k) = head_vals[k];
        }
        let mut x = x_lo;
        if has_simd {
            while (x as i32) <= x_lim8_i {
                let s = ((x as i32) + ix0) as usize;
                let next4 = v128_load(src_row.add(s + 3) as *const v128);
                let next8 = v128_load(src_row.add(s + 7) as *const v128);
                let v0_a = i32x4_shuffle::<1, 2, 3, 4>(cur4, next4);
                let vp1_a = i32x4_shuffle::<2, 3, 4, 5>(cur4, next4);
                let vp2_a = i32x4_shuffle::<3, 4, 5, 6>(cur4, next4);
                let mut h_a = f32x4_add(
                    f32x4_add(f32x4_mul(cur4, wx0_v), f32x4_mul(v0_a, wx1_v)),
                    f32x4_add(f32x4_mul(vp1_a, wx2_v), f32x4_mul(vp2_a, wx3_v)),
                );
                h_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, h_a));
                if EXP_ACTIVE {
                    h_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(h_a, gain_v), off_v)));
                }
                v128_store(dst_row.add(x) as *mut v128, h_a);

                let v0_b = i32x4_shuffle::<1, 2, 3, 4>(next4, next8);
                let vp1_b = i32x4_shuffle::<2, 3, 4, 5>(next4, next8);
                let vp2_b = i32x4_shuffle::<3, 4, 5, 6>(next4, next8);
                let mut h_b = f32x4_add(
                    f32x4_add(f32x4_mul(next4, wx0_v), f32x4_mul(v0_b, wx1_v)),
                    f32x4_add(f32x4_mul(vp1_b, wx2_v), f32x4_mul(vp2_b, wx3_v)),
                );
                h_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, h_b));
                if EXP_ACTIVE {
                    h_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(h_b, gain_v), off_v)));
                }
                v128_store(dst_row.add(x + 4) as *mut v128, h_b);

                cur4 = next8;
                x += 8;
            }
            while (x as i32) <= x_lim_i {
                let s = ((x as i32) + ix0) as usize;
                let next4 = v128_load(src_row.add(s + 3) as *const v128);
                let v0 = i32x4_shuffle::<1, 2, 3, 4>(cur4, next4);
                let vp1 = i32x4_shuffle::<2, 3, 4, 5>(cur4, next4);
                let vp2 = i32x4_shuffle::<3, 4, 5, 6>(cur4, next4);
                let mut h_val = f32x4_add(
                    f32x4_add(f32x4_mul(cur4, wx0_v), f32x4_mul(v0, wx1_v)),
                    f32x4_add(f32x4_mul(vp1, wx2_v), f32x4_mul(vp2, wx3_v)),
                );
                h_val = f32x4_pmax(zero_f, f32x4_pmin(one_f, h_val));
                if EXP_ACTIVE {
                    h_val = f32x4_pmax(
                        zero_f,
                        f32x4_pmin(one_f, f32x4_add(f32x4_mul(h_val, gain_v), off_v)),
                    );
                }
                v128_store(dst_row.add(x) as *mut v128, h_val);
                cur4 = next4;
                x += 4;
            }
        }
        if x < vx_hi {
            let mut tbuf = [0.0f32; 12];
            let s0 = ((x as i32) + ix0 - 1) as usize;
            v128_store(tbuf.as_mut_ptr() as *mut v128, cur4);
            let mut k = s0 + 4;
            let mut ti = 4usize;
            while k < frame_w && ti < 12 {
                tbuf[ti] = *src_row.add(k);
                k += 1;
                ti += 1;
            }
            let mut tx = 1usize;
            while x < vx_hi {
                let sx = (x as i32) + ix0;
                let v = if sx >= 1 && sx + 2 < w_i {
                    wx0 * tbuf[tx - 1]
                        + wx1 * tbuf[tx]
                        + wx2 * tbuf[tx + 1]
                        + wx3 * tbuf[tx + 2]
                } else {
                    (1.0 - fx) * tbuf[tx] + fx * tbuf[tx + 1]
                };
                *dst_row.add(x) = clamp_exp_s(v, exp_gain, exp_off, EXP_ACTIVE);
                x += 1;
                tx += 1;
            }
        }
        if vx_hi < frame_w {
            copy_f32_row_simd128(dst_row.add(vx_hi), cur_row.add(vx_hi), frame_w - vx_hi);
        }
        return;
    }

    if vx_lo > 0 {
        copy_f32_row_simd128(dst_row, cur_row, vx_lo);
    }
    let mut x = vx_lo;
    while x < x_lo {
        let s = ((x as i32) + ix0) as usize;
        let v = (1.0 - fx) * (*src_row.add(s)) + fx * (*src_row.add(s + 1));
        *dst_row.add(x) = clamp_exp_s(v, exp_gain, exp_off, EXP_ACTIVE);
        x += 1;
    }
    if (x as i32) <= x_lim_i {
        let mut cur4 = v128_load(src_row.add((((x as i32) + ix0) as usize) - 1) as *const v128);
        while (x as i32) <= x_lim8_i {
            let s = ((x as i32) + ix0) as usize;
            let next4 = v128_load(src_row.add(s + 3) as *const v128);
            let next8 = v128_load(src_row.add(s + 7) as *const v128);
            let v0_a = i32x4_shuffle::<1, 2, 3, 4>(cur4, next4);
            let vp1_a = i32x4_shuffle::<2, 3, 4, 5>(cur4, next4);
            let vp2_a = i32x4_shuffle::<3, 4, 5, 6>(cur4, next4);
            let mut h_a = f32x4_add(
                f32x4_add(f32x4_mul(cur4, wx0_v), f32x4_mul(v0_a, wx1_v)),
                f32x4_add(f32x4_mul(vp1_a, wx2_v), f32x4_mul(vp2_a, wx3_v)),
            );
            h_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, h_a));
            if EXP_ACTIVE {
                h_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(h_a, gain_v), off_v)));
            }
            v128_store(dst_row.add(x) as *mut v128, h_a);

            let v0_b = i32x4_shuffle::<1, 2, 3, 4>(next4, next8);
            let vp1_b = i32x4_shuffle::<2, 3, 4, 5>(next4, next8);
            let vp2_b = i32x4_shuffle::<3, 4, 5, 6>(next4, next8);
            let mut h_b = f32x4_add(
                f32x4_add(f32x4_mul(next4, wx0_v), f32x4_mul(v0_b, wx1_v)),
                f32x4_add(f32x4_mul(vp1_b, wx2_v), f32x4_mul(vp2_b, wx3_v)),
            );
            h_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, h_b));
            if EXP_ACTIVE {
                h_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(h_b, gain_v), off_v)));
            }
            v128_store(dst_row.add(x + 4) as *mut v128, h_b);

            cur4 = next8;
            x += 8;
        }
        while (x as i32) <= x_lim_i {
            let s = ((x as i32) + ix0) as usize;
            let next4 = v128_load(src_row.add(s + 3) as *const v128);
            let v0 = i32x4_shuffle::<1, 2, 3, 4>(cur4, next4);
            let vp1 = i32x4_shuffle::<2, 3, 4, 5>(cur4, next4);
            let vp2 = i32x4_shuffle::<3, 4, 5, 6>(cur4, next4);
            let mut h_val = f32x4_add(
                f32x4_add(f32x4_mul(cur4, wx0_v), f32x4_mul(v0, wx1_v)),
                f32x4_add(f32x4_mul(vp1, wx2_v), f32x4_mul(vp2, wx3_v)),
            );
            h_val = f32x4_pmax(zero_f, f32x4_pmin(one_f, h_val));
            if EXP_ACTIVE {
                h_val = f32x4_pmax(
                    zero_f,
                    f32x4_pmin(one_f, f32x4_add(f32x4_mul(h_val, gain_v), off_v)),
                );
            }
            v128_store(dst_row.add(x) as *mut v128, h_val);
            cur4 = next4;
            x += 4;
        }
    }
    while x < vx_hi {
        let sx = (x as i32) + ix0;
        let s = sx as usize;
        let v = if sx >= 1 && sx + 2 < w_i {
            wx0 * (*src_row.add(s - 1))
                + wx1 * (*src_row.add(s))
                + wx2 * (*src_row.add(s + 1))
                + wx3 * (*src_row.add(s + 2))
        } else {
            (1.0 - fx) * (*src_row.add(s)) + fx * (*src_row.add(s + 1))
        };
        *dst_row.add(x) = clamp_exp_s(v, exp_gain, exp_off, EXP_ACTIVE);
        x += 1;
    }
    if vx_hi < frame_w {
        copy_f32_row_simd128(dst_row.add(vx_hi), cur_row.add(vx_hi), frame_w - vx_hi);
    }
}

#[inline(always)]
unsafe fn clamp_exp_copy_row_simd128(
    dst: *mut f32,
    src: *const f32,
    cur_row: *const f32,
    frame_w: usize,
    vx_lo: usize,
    vx_hi: usize,
    exp_f_active: bool,
    exp_gain: f32,
    exp_off: f32,
    gain_v: v128,
    off_v: v128,
    zero_f: v128,
    one_f: v128,
) {
    if vx_lo > 0 {
        copy_f32_row_simd128(dst, cur_row, vx_lo);
    }
    let mut x = vx_lo;
    if exp_f_active {
        while x + 4 <= vx_hi {
            let v = f32x4_pmax(zero_f, f32x4_pmin(one_f, v128_load(src.add(x) as *const v128)));
            let out_v = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(v, gain_v), off_v)));
            v128_store(dst.add(x) as *mut v128, out_v);
            x += 4;
        }
        while x < vx_hi {
            let v = (*src.add(x)).clamp(0.0, 1.0);
            *dst.add(x) = (v * exp_gain + exp_off).clamp(0.0, 1.0);
            x += 1;
        }
    } else {
        while x + 4 <= vx_hi {
            let v = f32x4_pmax(zero_f, f32x4_pmin(one_f, v128_load(src.add(x) as *const v128)));
            v128_store(dst.add(x) as *mut v128, v);
            x += 4;
        }
        while x < vx_hi {
            *dst.add(x) = (*src.add(x)).clamp(0.0, 1.0);
            x += 1;
        }
    }
    if vx_hi < frame_w {
        copy_f32_row_simd128(dst.add(vx_hi), cur_row.add(vx_hi), frame_w - vx_hi);
    }
}

#[inline(always)]
unsafe fn vert_catmull_row_simd128(
    bg_row: *mut f32,
    cur_row: *const f32,
    scratch: *const f32,
    frame_w: usize,
    frame_h: usize,
    sy: i32,
    ix0: i32,
    fy: f32,
    wy0: f32,
    wy1: f32,
    wy2: f32,
    wy3: f32,
    wy0_v: v128,
    wy1_v: v128,
    wy2_v: v128,
    wy3_v: v128,
    exp_f_active: bool,
    exp_gain: f32,
    exp_off: f32,
    gain_v: v128,
    off_v: v128,
    zero_f: v128,
    one_f: v128,
) {
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let s_row = sy as usize;
    if sy >= 1 && sy + 2 < h_i {
        let sr_m1 = scratch.add(((s_row - 1) & 7) * frame_w);
        let sr_0 = scratch.add((s_row & 7) * frame_w);
        let sr_p1 = scratch.add(((s_row + 1) & 7) * frame_w);
        let sr_p2 = scratch.add(((s_row + 2) & 7) * frame_w);

        let vx_lo = (-ix0).clamp(0, w_i) as usize;
        let mut x = 0usize;
        if vx_lo > 0 {
            copy_f32_row_simd128(bg_row, cur_row, vx_lo);
            x = vx_lo;
        }
        let vx_lim_i = (w_i - 4).min(w_i - 5 - ix0);
        if exp_f_active {
            while (x as i32) <= vx_lim_i {
                let vm1 = v128_load(sr_m1.add(x) as *const v128);
                let v0 = v128_load(sr_0.add(x) as *const v128);
                let vp1 = v128_load(sr_p1.add(x) as *const v128);
                let vp2 = v128_load(sr_p2.add(x) as *const v128);
                let interp = f32x4_pmax(
                    zero_f,
                    f32x4_pmin(
                        one_f,
                        f32x4_add(
                            f32x4_add(f32x4_mul(vm1, wy0_v), f32x4_mul(v0, wy1_v)),
                            f32x4_add(f32x4_mul(vp1, wy2_v), f32x4_mul(vp2, wy3_v)),
                        ),
                    ),
                );
                let adj = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(interp, gain_v), off_v)));
                v128_store(bg_row.add(x) as *mut v128, adj);
                x += 4;
            }
        } else {
            while (x as i32) <= vx_lim_i {
                let vm1 = v128_load(sr_m1.add(x) as *const v128);
                let v0 = v128_load(sr_0.add(x) as *const v128);
                let vp1 = v128_load(sr_p1.add(x) as *const v128);
                let vp2 = v128_load(sr_p2.add(x) as *const v128);
                let interp = f32x4_pmax(
                    zero_f,
                    f32x4_pmin(
                        one_f,
                        f32x4_add(
                            f32x4_add(f32x4_mul(vm1, wy0_v), f32x4_mul(v0, wy1_v)),
                            f32x4_add(f32x4_mul(vp1, wy2_v), f32x4_mul(vp2, wy3_v)),
                        ),
                    ),
                );
                v128_store(bg_row.add(x) as *mut v128, interp);
                x += 4;
            }
        }
        while x < frame_w {
            let sx = (x as i32) + ix0;
            if sx < 0 || sx + 1 >= w_i {
                *bg_row.add(x) = *cur_row.add(x);
            } else {
                let interp_s = (wy0 * (*sr_m1.add(x))
                    + wy1 * (*sr_0.add(x))
                    + wy2 * (*sr_p1.add(x))
                    + wy3 * (*sr_p2.add(x)))
                .clamp(0.0, 1.0);
                *bg_row.add(x) = if exp_f_active {
                    (interp_s * exp_gain + exp_off).clamp(0.0, 1.0)
                } else {
                    interp_s
                };
            }
            x += 1;
        }
    } else {
        let sr_0 = scratch.add((s_row & 7) * frame_w);
        let sr_p1 = scratch.add(((s_row + 1) & 7) * frame_w);
        for x in 0..frame_w {
            let sx = (x as i32) + ix0;
            if sx < 0 || sx + 1 >= w_i {
                *bg_row.add(x) = *cur_row.add(x);
            } else {
                let interp_s = ((1.0 - fy) * (*sr_0.add(x)) + fy * (*sr_p1.add(x))).clamp(0.0, 1.0);
                *bg_row.add(x) = if exp_f_active {
                    (interp_s * exp_gain + exp_off).clamp(0.0, 1.0)
                } else {
                    interp_s
                };
            }
        }
    }
}

#[inline(never)]
unsafe fn vert_catmull_2rows_ctx<const EXP_ACTIVE: bool>(
    ctx: &CatmullCtx,
    bg_row0: *mut f32,
    cur_row0: *const f32,
    bg_row1: *mut f32,
    cur_row1: *const f32,
    sr_a: *const f32,
    sr_b: *const f32,
    sr_c: *const f32,
    sr_d: *const f32,
    sr_e: *const f32,
) {
    let frame_w = ctx.frame_w;
    let w_i = ctx.w_i;
    let ix0 = ctx.ix0;
    let wy0 = ctx.wy0;
    let wy1 = ctx.wy1;
    let wy2 = ctx.wy2;
    let wy3 = ctx.wy3;
    let wy0_v = ctx.wy0_v;
    let wy1_v = ctx.wy1_v;
    let wy2_v = ctx.wy2_v;
    let wy3_v = ctx.wy3_v;
    let exp_gain = ctx.exp_gain;
    let exp_off = ctx.exp_off;
    let gain_v = ctx.gain_v;
    let off_v = ctx.off_v;
    let zero_f = ctx.zero_f;
    let one_f = ctx.one_f;

    let vx_lo = ctx.vx_lo;
    let vx_hi = ctx.vx_hi;
    let mut x = 0usize;
    if vx_lo > 0 {
        copy_f32_row_simd128(bg_row0, cur_row0, vx_lo);
        copy_f32_row_simd128(bg_row1, cur_row1, vx_lo);
        x = vx_lo;
    }
    let x_vec_v = ((vx_lo + 3) & !3).min(vx_hi);
    while x < x_vec_v {
        let va = *sr_a.add(x);
        let vb = *sr_b.add(x);
        let vc = *sr_c.add(x);
        let vd = *sr_d.add(x);
        let ve = *sr_e.add(x);
        let i0 = (wy0 * va + wy1 * vb + wy2 * vc + wy3 * vd).clamp(0.0, 1.0);
        let i1 = (wy0 * vb + wy1 * vc + wy2 * vd + wy3 * ve).clamp(0.0, 1.0);
        if EXP_ACTIVE {
            *bg_row0.add(x) = (i0 * exp_gain + exp_off).clamp(0.0, 1.0);
            *bg_row1.add(x) = (i1 * exp_gain + exp_off).clamp(0.0, 1.0);
        } else {
            *bg_row0.add(x) = i0;
            *bg_row1.add(x) = i1;
        }
        x += 1;
    }
    let vx_lim8_i = ctx.vx_lim8_v;
    let vx_lim_i = ctx.vx_lim_v;
    if EXP_ACTIVE {
        while (x as i32) <= vx_lim8_i {
            let va0 = v128_load(sr_a.add(x) as *const v128);
            let vb0 = v128_load(sr_b.add(x) as *const v128);
            let vc0 = v128_load(sr_c.add(x) as *const v128);
            let vd0 = v128_load(sr_d.add(x) as *const v128);
            let ve0 = v128_load(sr_e.add(x) as *const v128);
            let mid0 = f32x4_add(f32x4_mul(vb0, wy1_v), f32x4_mul(vc0, wy2_v));
            let s0_a = f32x4_add(mid0, f32x4_add(f32x4_mul(va0, wy0_v), f32x4_mul(vd0, wy3_v)));
            let i0_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, s0_a));
            let a0_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i0_a, gain_v), off_v)));
            v128_store(bg_row0.add(x) as *mut v128, a0_a);
            let s1_a = f32x4_add(
                f32x4_add(f32x4_mul(vb0, wy0_v), f32x4_mul(vc0, wy1_v)),
                f32x4_add(f32x4_mul(vd0, wy2_v), f32x4_mul(ve0, wy3_v)),
            );
            let i1_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, s1_a));
            let a1_a = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i1_a, gain_v), off_v)));
            v128_store(bg_row1.add(x) as *mut v128, a1_a);

            let x4 = x + 4;
            let va1 = v128_load(sr_a.add(x4) as *const v128);
            let vb1 = v128_load(sr_b.add(x4) as *const v128);
            let vc1 = v128_load(sr_c.add(x4) as *const v128);
            let vd1 = v128_load(sr_d.add(x4) as *const v128);
            let ve1 = v128_load(sr_e.add(x4) as *const v128);
            let mid1 = f32x4_add(f32x4_mul(vb1, wy1_v), f32x4_mul(vc1, wy2_v));
            let s0_b = f32x4_add(mid1, f32x4_add(f32x4_mul(va1, wy0_v), f32x4_mul(vd1, wy3_v)));
            let i0_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, s0_b));
            let a0_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i0_b, gain_v), off_v)));
            v128_store(bg_row0.add(x4) as *mut v128, a0_b);
            let s1_b = f32x4_add(
                f32x4_add(f32x4_mul(vb1, wy0_v), f32x4_mul(vc1, wy1_v)),
                f32x4_add(f32x4_mul(vd1, wy2_v), f32x4_mul(ve1, wy3_v)),
            );
            let i1_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, s1_b));
            let a1_b = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i1_b, gain_v), off_v)));
            v128_store(bg_row1.add(x4) as *mut v128, a1_b);
            x += 8;
        }
        while (x as i32) <= vx_lim_i {
            let va = v128_load(sr_a.add(x) as *const v128);
            let vb = v128_load(sr_b.add(x) as *const v128);
            let vc = v128_load(sr_c.add(x) as *const v128);
            let vd = v128_load(sr_d.add(x) as *const v128);
            let ve = v128_load(sr_e.add(x) as *const v128);
            let mid = f32x4_add(f32x4_mul(vb, wy1_v), f32x4_mul(vc, wy2_v));
            let s0 = f32x4_add(mid, f32x4_add(f32x4_mul(va, wy0_v), f32x4_mul(vd, wy3_v)));
            let i0 = f32x4_pmax(zero_f, f32x4_pmin(one_f, s0));
            let a0 = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i0, gain_v), off_v)));
            v128_store(bg_row0.add(x) as *mut v128, a0);

            let s1 = f32x4_add(
                f32x4_add(f32x4_mul(vb, wy0_v), f32x4_mul(vc, wy1_v)),
                f32x4_add(f32x4_mul(vd, wy2_v), f32x4_mul(ve, wy3_v)),
            );
            let i1 = f32x4_pmax(zero_f, f32x4_pmin(one_f, s1));
            let a1 = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i1, gain_v), off_v)));
            v128_store(bg_row1.add(x) as *mut v128, a1);
            x += 4;
        }
    } else {
        while (x as i32) <= vx_lim8_i {
            let va0 = v128_load(sr_a.add(x) as *const v128);
            let vb0 = v128_load(sr_b.add(x) as *const v128);
            let vc0 = v128_load(sr_c.add(x) as *const v128);
            let vd0 = v128_load(sr_d.add(x) as *const v128);
            let ve0 = v128_load(sr_e.add(x) as *const v128);
            let s0_a = f32x4_add(
                f32x4_add(f32x4_mul(va0, wy0_v), f32x4_mul(vb0, wy1_v)),
                f32x4_add(f32x4_mul(vc0, wy2_v), f32x4_mul(vd0, wy3_v)),
            );
            v128_store(
                bg_row0.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, s0_a)),
            );
            let s1_a = f32x4_add(
                f32x4_add(f32x4_mul(vb0, wy0_v), f32x4_mul(vc0, wy1_v)),
                f32x4_add(f32x4_mul(vd0, wy2_v), f32x4_mul(ve0, wy3_v)),
            );
            v128_store(
                bg_row1.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, s1_a)),
            );

            let x4 = x + 4;
            let va1 = v128_load(sr_a.add(x4) as *const v128);
            let vb1 = v128_load(sr_b.add(x4) as *const v128);
            let vc1 = v128_load(sr_c.add(x4) as *const v128);
            let vd1 = v128_load(sr_d.add(x4) as *const v128);
            let ve1 = v128_load(sr_e.add(x4) as *const v128);
            let s0_b = f32x4_add(
                f32x4_add(f32x4_mul(va1, wy0_v), f32x4_mul(vb1, wy1_v)),
                f32x4_add(f32x4_mul(vc1, wy2_v), f32x4_mul(vd1, wy3_v)),
            );
            v128_store(
                bg_row0.add(x4) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, s0_b)),
            );
            let s1_b = f32x4_add(
                f32x4_add(f32x4_mul(vb1, wy0_v), f32x4_mul(vc1, wy1_v)),
                f32x4_add(f32x4_mul(vd1, wy2_v), f32x4_mul(ve1, wy3_v)),
            );
            v128_store(
                bg_row1.add(x4) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, s1_b)),
            );
            x += 8;
        }
        while (x as i32) <= vx_lim_i {
            let va = v128_load(sr_a.add(x) as *const v128);
            let vb = v128_load(sr_b.add(x) as *const v128);
            let vc = v128_load(sr_c.add(x) as *const v128);
            let vd = v128_load(sr_d.add(x) as *const v128);
            let ve = v128_load(sr_e.add(x) as *const v128);
            let s0 = f32x4_add(
                f32x4_add(f32x4_mul(va, wy0_v), f32x4_mul(vb, wy1_v)),
                f32x4_add(f32x4_mul(vc, wy2_v), f32x4_mul(vd, wy3_v)),
            );
            v128_store(
                bg_row0.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, s0)),
            );
            let s1 = f32x4_add(
                f32x4_add(f32x4_mul(vb, wy0_v), f32x4_mul(vc, wy1_v)),
                f32x4_add(f32x4_mul(vd, wy2_v), f32x4_mul(ve, wy3_v)),
            );
            v128_store(
                bg_row1.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, s1)),
            );
            x += 4;
        }
    }
    while x < frame_w {
        let sx = (x as i32) + ix0;
        if sx < 0 || sx + 1 >= w_i {
            *bg_row0.add(x) = *cur_row0.add(x);
            *bg_row1.add(x) = *cur_row1.add(x);
        } else {
            let va = *sr_a.add(x);
            let vb = *sr_b.add(x);
            let vc = *sr_c.add(x);
            let vd = *sr_d.add(x);
            let ve = *sr_e.add(x);
            let i0 = (wy0 * va + wy1 * vb + wy2 * vc + wy3 * vd).clamp(0.0, 1.0);
            let i1 = (wy0 * vb + wy1 * vc + wy2 * vd + wy3 * ve).clamp(0.0, 1.0);
            if EXP_ACTIVE {
                *bg_row0.add(x) = (i0 * exp_gain + exp_off).clamp(0.0, 1.0);
                *bg_row1.add(x) = (i1 * exp_gain + exp_off).clamp(0.0, 1.0);
            } else {
                *bg_row0.add(x) = i0;
                *bg_row1.add(x) = i1;
            }
        }
        x += 1;
    }
}

#[inline(always)]
unsafe fn horiz_catmull_2rows_inline(
    ctx: &CatmullCtx,
    src_r0: *const f32,
    dst_r0: *mut f32,
    src_r1: *const f32,
    dst_r1: *mut f32,
) {
    let vx_lo = ctx.vx_lo;
    let vx_hi = ctx.vx_hi;
    if vx_lo >= vx_hi {
        return;
    }
    let ix0 = ctx.ix0;
    let fx = ctx.fx;
    let wx0 = ctx.wx0;
    let wx1 = ctx.wx1;
    let wx2 = ctx.wx2;
    let wx3 = ctx.wx3;
    let wx0_v = ctx.wx0_v;
    let wx1_v = ctx.wx1_v;
    let wx2_v = ctx.wx2_v;
    let wx3_v = ctx.wx3_v;
    let w_i = ctx.w_i;

    let x_lo = ctx.x_lo_h;
    let mut x = vx_lo;
    while x < x_lo {
        let sx = (x as i32) + ix0;
        let s = sx as usize;
        *dst_r0.add(x) = (1.0 - fx) * (*src_r0.add(s)) + fx * (*src_r0.add(s + 1));
        *dst_r1.add(x) = (1.0 - fx) * (*src_r1.add(s)) + fx * (*src_r1.add(s + 1));
        x += 1;
    }

    let x_lim_i = ctx.x_lim_h;
    if (x as i32) <= x_lim_i {
        let x_vec_h = ((x_lo + 3) & !3).min(vx_hi);
        while x < x_vec_h {
            let sx = (x as i32) + ix0;
            let s = sx as usize;
            if sx >= 1 && sx + 2 < w_i {
                *dst_r0.add(x) = wx0 * (*src_r0.add(s - 1))
                    + wx1 * (*src_r0.add(s))
                    + wx2 * (*src_r0.add(s + 1))
                    + wx3 * (*src_r0.add(s + 2));
                *dst_r1.add(x) = wx0 * (*src_r1.add(s - 1))
                    + wx1 * (*src_r1.add(s))
                    + wx2 * (*src_r1.add(s + 1))
                    + wx3 * (*src_r1.add(s + 2));
            } else {
                *dst_r0.add(x) = (1.0 - fx) * (*src_r0.add(s)) + fx * (*src_r0.add(s + 1));
                *dst_r1.add(x) = (1.0 - fx) * (*src_r1.add(s)) + fx * (*src_r1.add(s + 1));
            }
            x += 1;
        }
    }
    if (x as i32) <= x_lim_i {
        let init_off = (((x as i32) + ix0) as usize) - 1;
        let mut cur4_0 = v128_load(src_r0.add(init_off) as *const v128);
        let mut cur4_1 = v128_load(src_r1.add(init_off) as *const v128);
        let x_lim8_i = ctx.x_lim8_h;
        while (x as i32) <= x_lim8_i {
            let s = ((x as i32) + ix0) as usize;
            let next4_0 = v128_load(src_r0.add(s + 3) as *const v128);
            let next8_0 = v128_load(src_r0.add(s + 7) as *const v128);
            let v0_0a = i32x4_shuffle::<1, 2, 3, 4>(cur4_0, next4_0);
            let vp1_0a = i32x4_shuffle::<2, 3, 4, 5>(cur4_0, next4_0);
            let vp2_0a = i32x4_shuffle::<3, 4, 5, 6>(cur4_0, next4_0);
            let h_0a = f32x4_add(
                f32x4_add(f32x4_mul(cur4_0, wx0_v), f32x4_mul(v0_0a, wx1_v)),
                f32x4_add(f32x4_mul(vp1_0a, wx2_v), f32x4_mul(vp2_0a, wx3_v)),
            );
            v128_store(dst_r0.add(x) as *mut v128, h_0a);
            let v0_0b = i32x4_shuffle::<1, 2, 3, 4>(next4_0, next8_0);
            let vp1_0b = i32x4_shuffle::<2, 3, 4, 5>(next4_0, next8_0);
            let vp2_0b = i32x4_shuffle::<3, 4, 5, 6>(next4_0, next8_0);
            let h_0b = f32x4_add(
                f32x4_add(f32x4_mul(next4_0, wx0_v), f32x4_mul(v0_0b, wx1_v)),
                f32x4_add(f32x4_mul(vp1_0b, wx2_v), f32x4_mul(vp2_0b, wx3_v)),
            );
            v128_store(dst_r0.add(x + 4) as *mut v128, h_0b);
            cur4_0 = next8_0;

            let next4_1 = v128_load(src_r1.add(s + 3) as *const v128);
            let next8_1 = v128_load(src_r1.add(s + 7) as *const v128);
            let v0_1a = i32x4_shuffle::<1, 2, 3, 4>(cur4_1, next4_1);
            let vp1_1a = i32x4_shuffle::<2, 3, 4, 5>(cur4_1, next4_1);
            let vp2_1a = i32x4_shuffle::<3, 4, 5, 6>(cur4_1, next4_1);
            let h_1a = f32x4_add(
                f32x4_add(f32x4_mul(cur4_1, wx0_v), f32x4_mul(v0_1a, wx1_v)),
                f32x4_add(f32x4_mul(vp1_1a, wx2_v), f32x4_mul(vp2_1a, wx3_v)),
            );
            v128_store(dst_r1.add(x) as *mut v128, h_1a);
            let v0_1b = i32x4_shuffle::<1, 2, 3, 4>(next4_1, next8_1);
            let vp1_1b = i32x4_shuffle::<2, 3, 4, 5>(next4_1, next8_1);
            let vp2_1b = i32x4_shuffle::<3, 4, 5, 6>(next4_1, next8_1);
            let h_1b = f32x4_add(
                f32x4_add(f32x4_mul(next4_1, wx0_v), f32x4_mul(v0_1b, wx1_v)),
                f32x4_add(f32x4_mul(vp1_1b, wx2_v), f32x4_mul(vp2_1b, wx3_v)),
            );
            v128_store(dst_r1.add(x + 4) as *mut v128, h_1b);
            cur4_1 = next8_1;

            x += 8;
        }
        while (x as i32) <= x_lim_i {
            let s = ((x as i32) + ix0) as usize;
            let next4_0 = v128_load(src_r0.add(s + 3) as *const v128);
            let v0_0 = i32x4_shuffle::<1, 2, 3, 4>(cur4_0, next4_0);
            let vp1_0 = i32x4_shuffle::<2, 3, 4, 5>(cur4_0, next4_0);
            let vp2_0 = i32x4_shuffle::<3, 4, 5, 6>(cur4_0, next4_0);
            let h_0 = f32x4_add(
                f32x4_add(f32x4_mul(cur4_0, wx0_v), f32x4_mul(v0_0, wx1_v)),
                f32x4_add(f32x4_mul(vp1_0, wx2_v), f32x4_mul(vp2_0, wx3_v)),
            );
            v128_store(dst_r0.add(x) as *mut v128, h_0);
            cur4_0 = next4_0;

            let next4_1 = v128_load(src_r1.add(s + 3) as *const v128);
            let v0_1 = i32x4_shuffle::<1, 2, 3, 4>(cur4_1, next4_1);
            let vp1_1 = i32x4_shuffle::<2, 3, 4, 5>(cur4_1, next4_1);
            let vp2_1 = i32x4_shuffle::<3, 4, 5, 6>(cur4_1, next4_1);
            let h_1 = f32x4_add(
                f32x4_add(f32x4_mul(cur4_1, wx0_v), f32x4_mul(v0_1, wx1_v)),
                f32x4_add(f32x4_mul(vp1_1, wx2_v), f32x4_mul(vp2_1, wx3_v)),
            );
            v128_store(dst_r1.add(x) as *mut v128, h_1);
            cur4_1 = next4_1;

            x += 4;
        }
    }

    while x < vx_hi {
        let sx = (x as i32) + ix0;
        let s = sx as usize;
        if sx >= 1 && sx + 2 < w_i {
            *dst_r0.add(x) = wx0 * (*src_r0.add(s - 1))
                + wx1 * (*src_r0.add(s))
                + wx2 * (*src_r0.add(s + 1))
                + wx3 * (*src_r0.add(s + 2));
            *dst_r1.add(x) = wx0 * (*src_r1.add(s - 1))
                + wx1 * (*src_r1.add(s))
                + wx2 * (*src_r1.add(s + 1))
                + wx3 * (*src_r1.add(s + 2));
        } else {
            *dst_r0.add(x) = (1.0 - fx) * (*src_r0.add(s)) + fx * (*src_r0.add(s + 1));
            *dst_r1.add(x) = (1.0 - fx) * (*src_r1.add(s)) + fx * (*src_r1.add(s + 1));
        }
        x += 1;
    }
}

#[inline(always)]
unsafe fn vert_catmull_4rows_inline<const EXP_ACTIVE: bool>(
    ctx: &CatmullCtx,
    bg_r0: *mut f32,
    cur_r0: *const f32,
    bg_r1: *mut f32,
    cur_r1: *const f32,
    bg_r2: *mut f32,
    cur_r2: *const f32,
    bg_r3: *mut f32,
    cur_r3: *const f32,
    sr0: *const f32,
    sr1: *const f32,
    sr2: *const f32,
    sr3: *const f32,
    sr4: *const f32,
    sr5: *const f32,
    sr6: *const f32,
) {
    let frame_w = ctx.frame_w;
    let w_i = ctx.w_i;
    let ix0 = ctx.ix0;
    let wy0 = ctx.wy0;
    let wy1 = ctx.wy1;
    let wy2 = ctx.wy2;
    let wy3 = ctx.wy3;
    let wy0_v = ctx.wy0_v;
    let wy1_v = ctx.wy1_v;
    let wy2_v = ctx.wy2_v;
    let wy3_v = ctx.wy3_v;
    let exp_gain = ctx.exp_gain;
    let exp_off = ctx.exp_off;
    let gain_v = ctx.gain_v;
    let off_v = ctx.off_v;
    let zero_f = ctx.zero_f;
    let one_f = ctx.one_f;

    let vx_lo = ctx.vx_lo;
    let vx_hi = ctx.vx_hi;
    let mut x = 0usize;
    if vx_lo > 0 {
        copy_f32_row_simd128(bg_r0, cur_r0, vx_lo);
        copy_f32_row_simd128(bg_r1, cur_r1, vx_lo);
        copy_f32_row_simd128(bg_r2, cur_r2, vx_lo);
        copy_f32_row_simd128(bg_r3, cur_r3, vx_lo);
        x = vx_lo;
    }
    let x_vec_v = ((vx_lo + 3) & !3).min(vx_hi);
    while x < x_vec_v {
        let v0 = *sr0.add(x);
        let v1 = *sr1.add(x);
        let v2 = *sr2.add(x);
        let v3 = *sr3.add(x);
        let v4 = *sr4.add(x);
        let v5 = *sr5.add(x);
        let v6 = *sr6.add(x);
        let i0 = (wy0 * v0 + wy1 * v1 + wy2 * v2 + wy3 * v3).clamp(0.0, 1.0);
        let i1 = (wy0 * v1 + wy1 * v2 + wy2 * v3 + wy3 * v4).clamp(0.0, 1.0);
        let i2 = (wy0 * v2 + wy1 * v3 + wy2 * v4 + wy3 * v5).clamp(0.0, 1.0);
        let i3 = (wy0 * v3 + wy1 * v4 + wy2 * v5 + wy3 * v6).clamp(0.0, 1.0);
        if EXP_ACTIVE {
            *bg_r0.add(x) = (i0 * exp_gain + exp_off).clamp(0.0, 1.0);
            *bg_r1.add(x) = (i1 * exp_gain + exp_off).clamp(0.0, 1.0);
            *bg_r2.add(x) = (i2 * exp_gain + exp_off).clamp(0.0, 1.0);
            *bg_r3.add(x) = (i3 * exp_gain + exp_off).clamp(0.0, 1.0);
        } else {
            *bg_r0.add(x) = i0;
            *bg_r1.add(x) = i1;
            *bg_r2.add(x) = i2;
            *bg_r3.add(x) = i3;
        }
        x += 1;
    }
    let vx_lim_i = ctx.vx_lim_v;
    if EXP_ACTIVE {
        while (x as i32) <= vx_lim_i {
            let v0 = v128_load(sr0.add(x) as *const v128);
            let v1 = v128_load(sr1.add(x) as *const v128);
            let v2 = v128_load(sr2.add(x) as *const v128);
            let v3 = v128_load(sr3.add(x) as *const v128);
            let s0 = f32x4_add(
                f32x4_add(f32x4_mul(v0, wy0_v), f32x4_mul(v1, wy1_v)),
                f32x4_add(f32x4_mul(v2, wy2_v), f32x4_mul(v3, wy3_v)),
            );
            let i0 = f32x4_pmax(zero_f, f32x4_pmin(one_f, s0));
            v128_store(
                bg_r0.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i0, gain_v), off_v))),
            );
            let v4 = v128_load(sr4.add(x) as *const v128);
            let s1 = f32x4_add(
                f32x4_add(f32x4_mul(v1, wy0_v), f32x4_mul(v2, wy1_v)),
                f32x4_add(f32x4_mul(v3, wy2_v), f32x4_mul(v4, wy3_v)),
            );
            let i1 = f32x4_pmax(zero_f, f32x4_pmin(one_f, s1));
            v128_store(
                bg_r1.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i1, gain_v), off_v))),
            );
            let v5 = v128_load(sr5.add(x) as *const v128);
            let s2 = f32x4_add(
                f32x4_add(f32x4_mul(v2, wy0_v), f32x4_mul(v3, wy1_v)),
                f32x4_add(f32x4_mul(v4, wy2_v), f32x4_mul(v5, wy3_v)),
            );
            let i2 = f32x4_pmax(zero_f, f32x4_pmin(one_f, s2));
            v128_store(
                bg_r2.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i2, gain_v), off_v))),
            );
            let v6 = v128_load(sr6.add(x) as *const v128);
            let s3 = f32x4_add(
                f32x4_add(f32x4_mul(v3, wy0_v), f32x4_mul(v4, wy1_v)),
                f32x4_add(f32x4_mul(v5, wy2_v), f32x4_mul(v6, wy3_v)),
            );
            let i3 = f32x4_pmax(zero_f, f32x4_pmin(one_f, s3));
            v128_store(
                bg_r3.add(x) as *mut v128,
                f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i3, gain_v), off_v))),
            );
            x += 4;
        }
    } else {
        while (x as i32) <= vx_lim_i - 4 {
            let v0 = v128_load(sr0.add(x) as *const v128);
            let u0 = v128_load(sr0.add(x + 4) as *const v128);
            let v1 = v128_load(sr1.add(x) as *const v128);
            let u1 = v128_load(sr1.add(x + 4) as *const v128);
            let v2 = v128_load(sr2.add(x) as *const v128);
            let u2 = v128_load(sr2.add(x + 4) as *const v128);
            let v3 = v128_load(sr3.add(x) as *const v128);
            let u3 = v128_load(sr3.add(x + 4) as *const v128);
            let s0a = f32x4_add(
                f32x4_add(f32x4_mul(v0, wy0_v), f32x4_mul(v1, wy1_v)),
                f32x4_add(f32x4_mul(v2, wy2_v), f32x4_mul(v3, wy3_v)),
            );
            let s0b = f32x4_add(
                f32x4_add(f32x4_mul(u0, wy0_v), f32x4_mul(u1, wy1_v)),
                f32x4_add(f32x4_mul(u2, wy2_v), f32x4_mul(u3, wy3_v)),
            );
            v128_store(bg_r0.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s0a)));
            v128_store(bg_r0.add(x + 4) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s0b)));

            let v4 = v128_load(sr4.add(x) as *const v128);
            let u4 = v128_load(sr4.add(x + 4) as *const v128);
            let s1a = f32x4_add(
                f32x4_add(f32x4_mul(v1, wy0_v), f32x4_mul(v2, wy1_v)),
                f32x4_add(f32x4_mul(v3, wy2_v), f32x4_mul(v4, wy3_v)),
            );
            let s1b = f32x4_add(
                f32x4_add(f32x4_mul(u1, wy0_v), f32x4_mul(u2, wy1_v)),
                f32x4_add(f32x4_mul(u3, wy2_v), f32x4_mul(u4, wy3_v)),
            );
            v128_store(bg_r1.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s1a)));
            v128_store(bg_r1.add(x + 4) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s1b)));

            let v5 = v128_load(sr5.add(x) as *const v128);
            let u5 = v128_load(sr5.add(x + 4) as *const v128);
            let s2a = f32x4_add(
                f32x4_add(f32x4_mul(v2, wy0_v), f32x4_mul(v3, wy1_v)),
                f32x4_add(f32x4_mul(v4, wy2_v), f32x4_mul(v5, wy3_v)),
            );
            let s2b = f32x4_add(
                f32x4_add(f32x4_mul(u2, wy0_v), f32x4_mul(u3, wy1_v)),
                f32x4_add(f32x4_mul(u4, wy2_v), f32x4_mul(u5, wy3_v)),
            );
            v128_store(bg_r2.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s2a)));
            v128_store(bg_r2.add(x + 4) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s2b)));

            let v6 = v128_load(sr6.add(x) as *const v128);
            let u6 = v128_load(sr6.add(x + 4) as *const v128);
            let s3a = f32x4_add(
                f32x4_add(f32x4_mul(v3, wy0_v), f32x4_mul(v4, wy1_v)),
                f32x4_add(f32x4_mul(v5, wy2_v), f32x4_mul(v6, wy3_v)),
            );
            let s3b = f32x4_add(
                f32x4_add(f32x4_mul(u3, wy0_v), f32x4_mul(u4, wy1_v)),
                f32x4_add(f32x4_mul(u5, wy2_v), f32x4_mul(u6, wy3_v)),
            );
            v128_store(bg_r3.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s3a)));
            v128_store(bg_r3.add(x + 4) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s3b)));
            x += 8;
        }
        while (x as i32) <= vx_lim_i {
            let v0 = v128_load(sr0.add(x) as *const v128);
            let v1 = v128_load(sr1.add(x) as *const v128);
            let v2 = v128_load(sr2.add(x) as *const v128);
            let v3 = v128_load(sr3.add(x) as *const v128);
            let s0 = f32x4_add(
                f32x4_add(f32x4_mul(v0, wy0_v), f32x4_mul(v1, wy1_v)),
                f32x4_add(f32x4_mul(v2, wy2_v), f32x4_mul(v3, wy3_v)),
            );
            v128_store(bg_r0.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s0)));
            let v4 = v128_load(sr4.add(x) as *const v128);
            let s1 = f32x4_add(
                f32x4_add(f32x4_mul(v1, wy0_v), f32x4_mul(v2, wy1_v)),
                f32x4_add(f32x4_mul(v3, wy2_v), f32x4_mul(v4, wy3_v)),
            );
            v128_store(bg_r1.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s1)));
            let v5 = v128_load(sr5.add(x) as *const v128);
            let s2 = f32x4_add(
                f32x4_add(f32x4_mul(v2, wy0_v), f32x4_mul(v3, wy1_v)),
                f32x4_add(f32x4_mul(v4, wy2_v), f32x4_mul(v5, wy3_v)),
            );
            v128_store(bg_r2.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s2)));
            let v6 = v128_load(sr6.add(x) as *const v128);
            let s3 = f32x4_add(
                f32x4_add(f32x4_mul(v3, wy0_v), f32x4_mul(v4, wy1_v)),
                f32x4_add(f32x4_mul(v5, wy2_v), f32x4_mul(v6, wy3_v)),
            );
            v128_store(bg_r3.add(x) as *mut v128, f32x4_pmax(zero_f, f32x4_pmin(one_f, s3)));
            x += 4;
        }
    }
    while x < vx_hi {
        let sx = (x as i32) + ix0;
        if sx < 0 || sx + 1 >= w_i {
            *bg_r0.add(x) = *cur_r0.add(x);
            *bg_r1.add(x) = *cur_r1.add(x);
            *bg_r2.add(x) = *cur_r2.add(x);
            *bg_r3.add(x) = *cur_r3.add(x);
        } else {
            let v0 = *sr0.add(x);
            let v1 = *sr1.add(x);
            let v2 = *sr2.add(x);
            let v3 = *sr3.add(x);
            let v4 = *sr4.add(x);
            let v5 = *sr5.add(x);
            let v6 = *sr6.add(x);
            let i0 = (wy0 * v0 + wy1 * v1 + wy2 * v2 + wy3 * v3).clamp(0.0, 1.0);
            let i1 = (wy0 * v1 + wy1 * v2 + wy2 * v3 + wy3 * v4).clamp(0.0, 1.0);
            let i2 = (wy0 * v2 + wy1 * v3 + wy2 * v4 + wy3 * v5).clamp(0.0, 1.0);
            let i3 = (wy0 * v3 + wy1 * v4 + wy2 * v5 + wy3 * v6).clamp(0.0, 1.0);
            if EXP_ACTIVE {
                *bg_r0.add(x) = (i0 * exp_gain + exp_off).clamp(0.0, 1.0);
                *bg_r1.add(x) = (i1 * exp_gain + exp_off).clamp(0.0, 1.0);
                *bg_r2.add(x) = (i2 * exp_gain + exp_off).clamp(0.0, 1.0);
                *bg_r3.add(x) = (i3 * exp_gain + exp_off).clamp(0.0, 1.0);
            } else {
                *bg_r0.add(x) = i0;
                *bg_r1.add(x) = i1;
                *bg_r2.add(x) = i2;
                *bg_r3.add(x) = i3;
            }
        }
        x += 1;
    }
    if vx_hi < frame_w {
        copy_f32_row_simd128(bg_r0.add(vx_hi), cur_r0.add(vx_hi), frame_w - vx_hi);
        copy_f32_row_simd128(bg_r1.add(vx_hi), cur_r1.add(vx_hi), frame_w - vx_hi);
        copy_f32_row_simd128(bg_r2.add(vx_hi), cur_r2.add(vx_hi), frame_w - vx_hi);
        copy_f32_row_simd128(bg_r3.add(vx_hi), cur_r3.add(vx_hi), frame_w - vx_hi);
    }
}

/// Strictly causal 4-tap separable Catmull-Rom bicubic SIMD128 background advection (`bg_luma`,
/// `bg_stat_streak`, `bg_eval_mask`) with optional affine exposure `(gain, off)` on `bg_luma`.
/// Never reads `cur_luma(t)` on interior pixels; only out-of-bounds incoming pan borders seed from `cur_luma(t)`.
pub unsafe fn advect_bg_subpixel_simd128(
    bg_luma: *mut f32,
    scratch: *mut f32,
    cur_luma: *const f32,
    stat_streak: *mut u8,
    _eval_mask: *mut u8,
    frame_w: usize,
    frame_h: usize,
    pan_dx: i32,
    pan_dy: i32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    frac_mag: f32,
    exp_gain: f32,
    exp_off: f32,
) {
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let exp_f_active = (exp_gain - 1.0).abs() >= 1e-4 || exp_off.abs() >= 1e-4;
    let zero_f = f32x4_splat(0.0);
    let one_f = f32x4_splat(1.0);
    let gain_v = f32x4_splat(exp_gain);
    let off_v = f32x4_splat(exp_off);

    if frac_mag < 0.04 {
        if pan_dx == 0 && pan_dy == 0 {
            if !exp_f_active {
                return;
            }
            let total = frame_w * frame_h;
            let mut idx = 0usize;
            while idx + 4 <= total {
                let v = v128_load(bg_luma.add(idx) as *const v128);
                let out_v = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(v, gain_v), off_v)));
                v128_store(bg_luma.add(idx) as *mut v128, out_v);
                idx += 4;
            }
            while idx < total {
                *bg_luma.add(idx) = ((*bg_luma.add(idx)) * exp_gain + exp_off).clamp(0.0, 1.0);
                idx += 1;
            }
            return;
        }
        let x_lo = pan_dx.clamp(0, w_i) as usize;
        let x_hi = (w_i + pan_dx).clamp(x_lo as i32, w_i) as usize;
        let x_len = x_hi.saturating_sub(x_lo);
        let rev_x = pan_dx > 0;
        let right_len = frame_w.saturating_sub(x_hi);
        for yi in 0..frame_h {
            let y = if pan_dy > 0 { frame_h - 1 - yi } else { yi };
            let sy = (y as i32) - pan_dy;
            let row_d = y * frame_w;
            if sy < 0 || sy >= h_i {
                copy_f32_row_simd128(bg_luma.add(row_d), cur_luma.add(row_d), frame_w);
                zero_u8_row_simd128(stat_streak.add(row_d), frame_w);
            } else {
                let row_s = (sy as usize) * frame_w;
                if x_len > 0 {
                    let sx_start = ((x_lo as i32) - pan_dx) as usize;
                    shift_f32_row_in_bounds_simd128(
                        bg_luma.add(row_d + x_lo),
                        bg_luma.add(row_s + sx_start),
                        x_len,
                        rev_x,
                        exp_f_active,
                        exp_gain,
                        exp_off,
                        gain_v,
                        off_v,
                        zero_f,
                        one_f,
                    );
                    if pan_dx != 0 || pan_dy != 0 {
                        shift_u8_row_in_bounds_simd128(
                            stat_streak.add(row_d + x_lo),
                            stat_streak.add(row_s + sx_start),
                            x_len,
                            rev_x,
                        );
                    }
                }
                if x_lo > 0 {
                    copy_f32_row_simd128(bg_luma.add(row_d), cur_luma.add(row_d), x_lo);
                    zero_u8_row_simd128(stat_streak.add(row_d), x_lo);
                }
                if right_len > 0 {
                    copy_f32_row_simd128(bg_luma.add(row_d + x_hi), cur_luma.add(row_d + x_hi), right_len);
                    zero_u8_row_simd128(stat_streak.add(row_d + x_hi), right_len);
                }
            }
        }
        return;
    }

    let shift_x = -pan_dx_f;
    let shift_y = -pan_dy_f;
    let ix0 = f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(shift_x))) as i32;
    let iy0 = f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(shift_y))) as i32;
    let fx = (shift_x - (ix0 as f32)).clamp(0.0, 1.0);
    let fy = (shift_y - (iy0 as f32)).clamp(0.0, 1.0);

    // 1. Helpers to fuse stationarity streak shift/border zeroing into the single bg advection pass
    let x_lo_i = pan_dx.max(0).max(-ix0).min(w_i);
    let x_hi_i = (w_i + pan_dx).min(w_i - 1 - ix0).max(x_lo_i).min(w_i);
    let x_lo = x_lo_i as usize;
    let x_hi = x_hi_i as usize;
    let x_len = x_hi.saturating_sub(x_lo);
    let right_len = frame_w.saturating_sub(x_hi);
    let y_b_lo = (-iy0).clamp(0, h_i) as usize;
    let y_b_hi = ((h_i - 1 - iy0).clamp(0, h_i) as usize).max(y_b_lo);
    if pan_dx == 0 && pan_dy == 0 {
        if y_b_lo > 0 {
            core::ptr::write_bytes(stat_streak, 0, y_b_lo * frame_w);
        }
        if y_b_hi < frame_h {
            core::ptr::write_bytes(stat_streak.add(y_b_hi * frame_w), 0, (frame_h - y_b_hi) * frame_w);
        }
        if x_lo == 1 && right_len == 0 {
            for y in y_b_lo..y_b_hi {
                *stat_streak.add(y * frame_w) = 0;
            }
        } else if x_lo == 0 && right_len == 1 {
            for y in y_b_lo..y_b_hi {
                *stat_streak.add(y * frame_w + x_hi) = 0;
            }
        } else if x_lo > 0 || right_len > 0 {
            for y in y_b_lo..y_b_hi {
                let row_p = stat_streak.add(y * frame_w);
                if x_lo > 0 {
                    core::ptr::write_bytes(row_p, 0, x_lo);
                }
                if right_len > 0 {
                    core::ptr::write_bytes(row_p.add(x_hi), 0, right_len);
                }
            }
        }
    } else {
        let y_i_lo = pan_dy.clamp(0, h_i) as usize;
        let y_i_hi = (h_i + pan_dy).clamp(0, h_i) as usize;
        let y_valid_lo = y_b_lo.max(y_i_lo);
        let y_valid_hi = y_b_hi.min(y_i_hi).max(y_valid_lo);
        if x_lo == 0 && right_len == 0 && pan_dx == 0 {
            if y_valid_hi > y_valid_lo && x_len > 0 {
                let src_y = ((y_valid_lo as i32) - pan_dy) as usize;
                core::ptr::copy(
                    stat_streak.add(src_y * frame_w),
                    stat_streak.add(y_valid_lo * frame_w),
                    (y_valid_hi - y_valid_lo) * frame_w,
                );
            }
        } else {
            let sx_start = ((x_lo as i32) - pan_dx) as usize;
            let copy_streak_row = |y: usize| {
                let row_d = y * frame_w;
                if x_len > 0 {
                    let sy_i = (y as i32) - pan_dy;
                    let row_s = (sy_i as usize) * frame_w;
                    core::ptr::copy(
                        stat_streak.add(row_s + sx_start),
                        stat_streak.add(row_d + x_lo),
                        x_len,
                    );
                }
                if x_lo == 1 {
                    *stat_streak.add(row_d) = 0;
                } else if x_lo > 0 {
                    core::ptr::write_bytes(stat_streak.add(row_d), 0, x_lo);
                }
                if right_len == 1 {
                    *stat_streak.add(row_d + x_hi) = 0;
                } else if right_len > 0 {
                    core::ptr::write_bytes(stat_streak.add(row_d + x_hi), 0, right_len);
                }
            };
            if pan_dy > 0 {
                for y in (y_valid_lo..y_valid_hi).rev() {
                    copy_streak_row(y);
                }
            } else {
                for y in y_valid_lo..y_valid_hi {
                    copy_streak_row(y);
                }
            }
        }
        if y_valid_lo > 0 {
            core::ptr::write_bytes(stat_streak, 0, y_valid_lo * frame_w);
        }
        if y_valid_hi < frame_h {
            core::ptr::write_bytes(stat_streak.add(y_valid_hi * frame_w), 0, (frame_h - y_valid_hi) * frame_w);
        }
    }

    // 2. Fused 8-row L1D ring-buffered separable 4-tap Catmull-Rom passes + causal affine exposure
    let (wx0, wx1, wx2, wx3) = catmull_rom_weights(fx);
    let (wy0, wy1, wy2, wy3) = catmull_rom_weights(fy);
    let vx_lo = (-ix0).clamp(0, w_i) as usize;
    let vx_hi = (w_i - 1 - ix0).clamp(vx_lo as i32, w_i) as usize;
    let catmull_ctx = CatmullCtx {
        wx0_v: f32x4_splat(wx0),
        wx1_v: f32x4_splat(wx1),
        wx2_v: f32x4_splat(wx2),
        wx3_v: f32x4_splat(wx3),
        wy0_v: f32x4_splat(wy0),
        wy1_v: f32x4_splat(wy1),
        wy2_v: f32x4_splat(wy2),
        wy3_v: f32x4_splat(wy3),
        gain_v,
        off_v,
        zero_f,
        one_f,
        frame_w,
        w_i,
        ix0,
        fx,
        wx0,
        wx1,
        wx2,
        wx3,
        wy0,
        wy1,
        wy2,
        wy3,
        exp_gain,
        exp_off,
        vx_lo,
        vx_hi,
        x_lo_h: (1 - ix0).clamp(vx_lo as i32, vx_hi as i32) as usize,
        x_lim_h: ((vx_hi as i32) - 4).min(w_i - 7 - ix0),
        x_lim8_h: ((vx_hi as i32) - 8).min(w_i - 11 - ix0),
        vx_lim8_v: (w_i - 8).min(w_i - 9 - ix0),
        vx_lim_v: (w_i - 4).min(w_i - 5 - ix0),
    };

    if fy == 0.0 {
        if fx == 0.0 {
            if ix0 == 0 && iy0 == 0 {
                if exp_f_active {
                    let total = frame_w * frame_h;
                    let mut idx = 0usize;
                    while idx + 4 <= total {
                        let v = f32x4_pmax(zero_f, f32x4_pmin(one_f, v128_load(bg_luma.add(idx) as *const v128)));
                        let out_v = f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(v, gain_v), off_v)));
                        v128_store(bg_luma.add(idx) as *mut v128, out_v);
                        idx += 4;
                    }
                    while idx < total {
                        let v = (*bg_luma.add(idx)).clamp(0.0, 1.0);
                        *bg_luma.add(idx) = (v * exp_gain + exp_off).clamp(0.0, 1.0);
                        idx += 1;
                    }
                }
                return;
            }
            for yi in 0..frame_h {
                let y = if iy0 < 0 { frame_h - 1 - yi } else { yi };
                let sy = (y as i32) + iy0;
                let ro = y * frame_w;
                if sy < 0 || sy + 1 >= h_i {
                    copy_f32_row_simd128(bg_luma.add(ro), cur_luma.add(ro), frame_w);
                } else {
                    let s_row = bg_luma.add((sy as usize) * frame_w);
                    let d_row = bg_luma.add(ro);
                    let c_row = cur_luma.add(ro);
                    if vx_lo < vx_hi {
                        shift_f32_row_in_bounds_simd128(
                            d_row.add(vx_lo),
                            s_row.add(((vx_lo as i32) + ix0) as usize),
                            vx_hi - vx_lo,
                            iy0 == 0 && ix0 < 0,
                            exp_f_active,
                            exp_gain,
                            exp_off,
                            gain_v,
                            off_v,
                            zero_f,
                            one_f,
                        );
                    }
                    if vx_lo > 0 {
                        copy_f32_row_simd128(d_row, c_row, vx_lo);
                    }
                    if vx_hi < frame_w {
                        copy_f32_row_simd128(d_row.add(vx_hi), c_row.add(vx_hi), frame_w - vx_hi);
                    }
                }
            }
            return;
        }
        let can_inplace = iy0 != 0 || (ix0 >= -3 && frame_w >= 16);
        for yi in 0..frame_h {
            let y = if iy0 < 0 { frame_h - 1 - yi } else { yi };
            let sy = (y as i32) + iy0;
            let ro = y * frame_w;
            if sy < 0 || sy + 1 >= h_i {
                copy_f32_row_simd128(bg_luma.add(ro), cur_luma.add(ro), frame_w);
            } else if can_inplace {
                let s_row = bg_luma.add((sy as usize) * frame_w);
                let d_row = bg_luma.add(ro);
                let c_row = cur_luma.add(ro);
                match (exp_f_active, iy0 == 0) {
                    (false, true) => horiz_catmull_1d_row_ctx::<false, true>(&catmull_ctx, d_row, s_row, c_row),
                    (false, false) => horiz_catmull_1d_row_ctx::<false, false>(&catmull_ctx, d_row, s_row, c_row),
                    (true, true) => horiz_catmull_1d_row_ctx::<true, true>(&catmull_ctx, d_row, s_row, c_row),
                    (true, false) => horiz_catmull_1d_row_ctx::<true, false>(&catmull_ctx, d_row, s_row, c_row),
                }
            } else {
                horiz_catmull_row_ctx::<false>(&catmull_ctx, bg_luma.add((sy as usize) * frame_w), scratch);
                clamp_exp_copy_row_simd128(
                    bg_luma.add(ro),
                    scratch,
                    cur_luma.add(ro),
                    frame_w, vx_lo, vx_hi, exp_f_active, exp_gain, exp_off,
                    gain_v, off_v, zero_f, one_f,
                );
            }
        }
        return;
    }

    if fx == 0.0 {
        advect_2d_vert_only_simd128(&catmull_ctx, bg_luma, scratch, cur_luma, frame_h, iy0, fy, exp_f_active);
        return;
    }

    macro_rules! run_2d_catmull {
        ($expa:expr) => {{
            if iy0 >= 0 {
                let mut computed_up_to: i32 = -1;
                let mut y = 0usize;
                while y < frame_h {
                    let sy = (y as i32) + iy0;
                    let ro = y * frame_w;
                    if sy < 0 || sy + 1 >= h_i {
                        copy_f32_row_simd128(bg_luma.add(ro), cur_luma.add(ro), frame_w);
                        y += 1;
                        continue;
                    }
                    if y + 3 < frame_h && sy >= 1 && sy + 5 < h_i {
                        let r_end = sy + 5;
                        let mut r = (computed_up_to + 1).max(sy - 1);
                        while r + 1 <= r_end {
                            let ru = r as usize;
                            horiz_catmull_2rows_inline(
                                &catmull_ctx,
                                bg_luma.add(ru * frame_w),
                                scratch.add((ru & 7) * frame_w),
                                bg_luma.add((ru + 1) * frame_w),
                                scratch.add(((ru + 1) & 7) * frame_w),
                            );
                            r += 2;
                        }
                        if r <= r_end {
                            let ru = r as usize;
                            horiz_catmull_row_ctx::<false>(
                                &catmull_ctx,
                                bg_luma.add(ru * frame_w),
                                scratch.add((ru & 7) * frame_w),
                            );
                        }
                        if r_end > computed_up_to {
                            computed_up_to = r_end;
                        }
                        let s = sy as usize;
                        vert_catmull_4rows_inline::<$expa>(
                            &catmull_ctx,
                            bg_luma.add(ro),
                            cur_luma.add(ro),
                            bg_luma.add(ro + frame_w),
                            cur_luma.add(ro + frame_w),
                            bg_luma.add(ro + 2 * frame_w),
                            cur_luma.add(ro + 2 * frame_w),
                            bg_luma.add(ro + 3 * frame_w),
                            cur_luma.add(ro + 3 * frame_w),
                            scratch.add(((s - 1) & 7) * frame_w),
                            scratch.add((s & 7) * frame_w),
                            scratch.add(((s + 1) & 7) * frame_w),
                            scratch.add(((s + 2) & 7) * frame_w),
                            scratch.add(((s + 3) & 7) * frame_w),
                            scratch.add(((s + 4) & 7) * frame_w),
                            scratch.add(((s + 5) & 7) * frame_w),
                        );
                        y += 4;
                        continue;
                    }
                    if y + 1 < frame_h && sy >= 1 && sy + 3 < h_i {
                        let r_end = sy + 3;
                        let mut r = (computed_up_to + 1).max(sy - 1);
                        while r <= r_end {
                            let ru = r as usize;
                            horiz_catmull_row_ctx::<false>(
                                &catmull_ctx,
                                bg_luma.add(ru * frame_w),
                                scratch.add((ru & 7) * frame_w),
                            );
                            r += 1;
                        }
                        if r_end > computed_up_to {
                            computed_up_to = r_end;
                        }
                        let s = sy as usize;
                        vert_catmull_2rows_ctx::<$expa>(
                            &catmull_ctx,
                            bg_luma.add(ro),
                            cur_luma.add(ro),
                            bg_luma.add(ro + frame_w),
                            cur_luma.add(ro + frame_w),
                            scratch.add(((s - 1) & 7) * frame_w),
                            scratch.add((s & 7) * frame_w),
                            scratch.add(((s + 1) & 7) * frame_w),
                            scratch.add(((s + 2) & 7) * frame_w),
                            scratch.add(((s + 3) & 7) * frame_w),
                        );
                        y += 2;
                        continue;
                    }
                    let r_end = (sy + 2).min(h_i - 1);
                    let mut r = (computed_up_to + 1).max((sy - 1).max(0));
                    while r <= r_end {
                        let ru = r as usize;
                        horiz_catmull_row_ctx::<false>(
                            &catmull_ctx,
                            bg_luma.add(ru * frame_w),
                            scratch.add((ru & 7) * frame_w),
                        );
                        r += 1;
                    }
                    if r_end > computed_up_to {
                        computed_up_to = r_end;
                    }
                    vert_catmull_row_simd128(
                        bg_luma.add(ro), cur_luma.add(ro), scratch, frame_w, frame_h, sy, ix0, fy,
                        wy0, wy1, wy2, wy3, catmull_ctx.wy0_v, catmull_ctx.wy1_v, catmull_ctx.wy2_v, catmull_ctx.wy3_v,
                        $expa, exp_gain, exp_off, gain_v, off_v, zero_f, one_f,
                    );
                    y += 1;
                }
            } else {
                let mut computed_down_to: i32 = h_i;
                let mut y_next = frame_h;
                while y_next > 0 {
                    let y = y_next - 1;
                    let sy = (y as i32) + iy0;
                    let ro = y * frame_w;
                    if sy < 0 || sy + 1 >= h_i {
                        copy_f32_row_simd128(bg_luma.add(ro), cur_luma.add(ro), frame_w);
                        y_next -= 1;
                        continue;
                    }
                    if y >= 3 && sy >= 4 && sy + 2 < h_i {
                        let r_start = sy - 4;
                        let mut r = (computed_down_to - 1).min(sy + 2);
                        while r - 1 >= r_start {
                            let ru = r as usize;
                            horiz_catmull_2rows_inline(
                                &catmull_ctx,
                                bg_luma.add(ru * frame_w),
                                scratch.add((ru & 7) * frame_w),
                                bg_luma.add((ru - 1) * frame_w),
                                scratch.add(((ru - 1) & 7) * frame_w),
                            );
                            r -= 2;
                        }
                        if r >= r_start {
                            let ru = r as usize;
                            horiz_catmull_row_ctx::<false>(
                                &catmull_ctx,
                                bg_luma.add(ru * frame_w),
                                scratch.add((ru & 7) * frame_w),
                            );
                        }
                        if r_start < computed_down_to {
                            computed_down_to = r_start;
                        }
                        let s = sy as usize;
                        vert_catmull_4rows_inline::<$expa>(
                            &catmull_ctx,
                            bg_luma.add(ro - 3 * frame_w),
                            cur_luma.add(ro - 3 * frame_w),
                            bg_luma.add(ro - 2 * frame_w),
                            cur_luma.add(ro - 2 * frame_w),
                            bg_luma.add(ro - frame_w),
                            cur_luma.add(ro - frame_w),
                            bg_luma.add(ro),
                            cur_luma.add(ro),
                            scratch.add(((s - 4) & 7) * frame_w),
                            scratch.add(((s - 3) & 7) * frame_w),
                            scratch.add(((s - 2) & 7) * frame_w),
                            scratch.add(((s - 1) & 7) * frame_w),
                            scratch.add((s & 7) * frame_w),
                            scratch.add(((s + 1) & 7) * frame_w),
                            scratch.add(((s + 2) & 7) * frame_w),
                        );
                        y_next -= 4;
                        continue;
                    }
                    if y >= 1 && sy >= 2 && sy + 2 < h_i {
                        let r_start = sy - 2;
                        let mut r = (computed_down_to - 1).min(sy + 2);
                        while r >= r_start {
                            let ru = r as usize;
                            horiz_catmull_row_ctx::<false>(
                                &catmull_ctx,
                                bg_luma.add(ru * frame_w),
                                scratch.add((ru & 7) * frame_w),
                            );
                            r -= 1;
                        }
                        if r_start < computed_down_to {
                            computed_down_to = r_start;
                        }
                        let s = sy as usize;
                        vert_catmull_2rows_ctx::<$expa>(
                            &catmull_ctx,
                            bg_luma.add(ro - frame_w),
                            cur_luma.add(ro - frame_w),
                            bg_luma.add(ro),
                            cur_luma.add(ro),
                            scratch.add(((s - 2) & 7) * frame_w),
                            scratch.add(((s - 1) & 7) * frame_w),
                            scratch.add((s & 7) * frame_w),
                            scratch.add(((s + 1) & 7) * frame_w),
                            scratch.add(((s + 2) & 7) * frame_w),
                        );
                        y_next -= 2;
                        continue;
                    }
                    let r_start = (sy - 1).max(0);
                    let mut r = (computed_down_to - 1).min((sy + 2).min(h_i - 1));
                    while r >= r_start {
                        let ru = r as usize;
                        horiz_catmull_row_ctx::<false>(
                            &catmull_ctx,
                            bg_luma.add(ru * frame_w),
                            scratch.add((ru & 7) * frame_w),
                        );
                        r -= 1;
                    }
                    if r_start < computed_down_to {
                        computed_down_to = r_start;
                    }
                    vert_catmull_row_simd128(
                        bg_luma.add(ro), cur_luma.add(ro), scratch, frame_w, frame_h, sy, ix0, fy,
                        wy0, wy1, wy2, wy3, catmull_ctx.wy0_v, catmull_ctx.wy1_v, catmull_ctx.wy2_v, catmull_ctx.wy3_v,
                        $expa, exp_gain, exp_off, gain_v, off_v, zero_f, one_f,
                    );
                    y_next -= 1;
                }
            }
        }};
    }
    if exp_f_active {
        run_2d_catmull!(true);
    } else {
        run_2d_catmull!(false);
    }
}

#[inline(never)]
unsafe fn vert_only_2rows_simd128(
    ctx: &CatmullCtx,
    bg_row0: *mut f32,
    cur_row0: *const f32,
    bg_row1: *mut f32,
    cur_row1: *const f32,
    sr_a: *const f32,
    sr_b: *const f32,
    sr_c: *const f32,
    sr_d: *const f32,
    sr_e: *const f32,
    scratch: *mut f32,
    save_mode: u8,
    exp_f_active: bool,
) {
    let frame_w = ctx.frame_w;
    let ix0 = ctx.ix0;
    let vx_lo = ctx.vx_lo;
    let vx_hi = ctx.vx_hi;
    let wy0 = ctx.wy0;
    let wy1 = ctx.wy1;
    let wy2 = ctx.wy2;
    let wy3 = ctx.wy3;
    let wy0_v = ctx.wy0_v;
    let wy1_v = ctx.wy1_v;
    let wy2_v = ctx.wy2_v;
    let wy3_v = ctx.wy3_v;
    let exp_gain = ctx.exp_gain;
    let exp_off = ctx.exp_off;
    let gain_v = ctx.gain_v;
    let off_v = ctx.off_v;
    let zero_f = ctx.zero_f;
    let one_f = ctx.one_f;
    let use_a = save_mode == 1;
    let use_e = save_mode == 2;
    let save_vc = save_mode != 0;

    let finish_s = |v: f32| -> f32 {
        let i = v.clamp(0.0, 1.0);
        if exp_f_active { (i * exp_gain + exp_off).clamp(0.0, 1.0) } else { i }
    };
    let finish_v = |v: v128| -> v128 {
        let i = f32x4_pmax(zero_f, f32x4_pmin(one_f, v));
        if exp_f_active {
            f32x4_pmax(zero_f, f32x4_pmin(one_f, f32x4_add(f32x4_mul(i, gain_v), off_v)))
        } else {
            i
        }
    };

    if ix0 < 0 {
        let sh = (-ix0) as usize;
        let mut x = vx_hi;
        while x > vx_lo && (x & 3) != 0 {
            x -= 1;
            let s = x - sh;
            let va = *(if use_a { scratch.add(x) } else { sr_a.add(s) });
            let vb = *sr_b.add(s);
            let vc = *sr_c.add(s);
            let vd = *sr_d.add(s);
            let ve = *(if use_e { scratch.add(x) } else { sr_e.add(s) });
            if save_vc { *scratch.add(x) = vc; }
            *bg_row0.add(x) = finish_s(wy0 * va + wy1 * vb + wy2 * vc + wy3 * vd);
            *bg_row1.add(x) = finish_s(wy0 * vb + wy1 * vc + wy2 * vd + wy3 * ve);
        }
        while x >= vx_lo + 4 {
            let x0 = x - 4;
            let s0 = x0 - sh;
            let va = v128_load((if use_a { scratch.add(x0) } else { sr_a.add(s0) }) as *const v128);
            let vb = v128_load(sr_b.add(s0) as *const v128);
            let vc = v128_load(sr_c.add(s0) as *const v128);
            let vd = v128_load(sr_d.add(s0) as *const v128);
            let ve = v128_load((if use_e { scratch.add(x0) } else { sr_e.add(s0) }) as *const v128);
            if save_vc { v128_store(scratch.add(x0) as *mut v128, vc); }
            let s0_v = f32x4_add(
                f32x4_add(f32x4_mul(va, wy0_v), f32x4_mul(vb, wy1_v)),
                f32x4_add(f32x4_mul(vc, wy2_v), f32x4_mul(vd, wy3_v)),
            );
            let s1_v = f32x4_add(
                f32x4_add(f32x4_mul(vb, wy0_v), f32x4_mul(vc, wy1_v)),
                f32x4_add(f32x4_mul(vd, wy2_v), f32x4_mul(ve, wy3_v)),
            );
            v128_store(bg_row0.add(x0) as *mut v128, finish_v(s0_v));
            v128_store(bg_row1.add(x0) as *mut v128, finish_v(s1_v));
            x = x0;
        }
        while x > vx_lo {
            x -= 1;
            let s = x - sh;
            let va = *(if use_a { scratch.add(x) } else { sr_a.add(s) });
            let vb = *sr_b.add(s);
            let vc = *sr_c.add(s);
            let vd = *sr_d.add(s);
            let ve = *(if use_e { scratch.add(x) } else { sr_e.add(s) });
            if save_vc { *scratch.add(x) = vc; }
            *bg_row0.add(x) = finish_s(wy0 * va + wy1 * vb + wy2 * vc + wy3 * vd);
            *bg_row1.add(x) = finish_s(wy0 * vb + wy1 * vc + wy2 * vd + wy3 * ve);
        }
    } else {
        let ix0_u = ix0 as usize;
        let mut x = 0usize;
        while x + 4 <= vx_hi {
            let s0 = x + ix0_u;
            let va = v128_load((if use_a { scratch.add(x) } else { sr_a.add(s0) }) as *const v128);
            let vb = v128_load(sr_b.add(s0) as *const v128);
            let vc = v128_load(sr_c.add(s0) as *const v128);
            let vd = v128_load(sr_d.add(s0) as *const v128);
            let ve = v128_load((if use_e { scratch.add(x) } else { sr_e.add(s0) }) as *const v128);
            if save_vc { v128_store(scratch.add(x) as *mut v128, vc); }
            let s0_v = f32x4_add(
                f32x4_add(f32x4_mul(va, wy0_v), f32x4_mul(vb, wy1_v)),
                f32x4_add(f32x4_mul(vc, wy2_v), f32x4_mul(vd, wy3_v)),
            );
            let s1_v = f32x4_add(
                f32x4_add(f32x4_mul(vb, wy0_v), f32x4_mul(vc, wy1_v)),
                f32x4_add(f32x4_mul(vd, wy2_v), f32x4_mul(ve, wy3_v)),
            );
            v128_store(bg_row0.add(x) as *mut v128, finish_v(s0_v));
            v128_store(bg_row1.add(x) as *mut v128, finish_v(s1_v));
            x += 4;
        }
        while x < vx_hi {
            let s = x + ix0_u;
            let va = *(if use_a { scratch.add(x) } else { sr_a.add(s) });
            let vb = *sr_b.add(s);
            let vc = *sr_c.add(s);
            let vd = *sr_d.add(s);
            let ve = *(if use_e { scratch.add(x) } else { sr_e.add(s) });
            if save_vc { *scratch.add(x) = vc; }
            *bg_row0.add(x) = finish_s(wy0 * va + wy1 * vb + wy2 * vc + wy3 * vd);
            *bg_row1.add(x) = finish_s(wy0 * vb + wy1 * vc + wy2 * vd + wy3 * ve);
            x += 1;
        }
    }
    if vx_lo > 0 {
        copy_f32_row_simd128(bg_row0, cur_row0, vx_lo);
        copy_f32_row_simd128(bg_row1, cur_row1, vx_lo);
    }
    if vx_hi < frame_w {
        copy_f32_row_simd128(bg_row0.add(vx_hi), cur_row0.add(vx_hi), frame_w - vx_hi);
        copy_f32_row_simd128(bg_row1.add(vx_hi), cur_row1.add(vx_hi), frame_w - vx_hi);
    }
}

#[inline(never)]
unsafe fn vert_only_1row_simd128(
    ctx: &CatmullCtx,
    bg_row: *mut f32,
    cur_row: *const f32,
    sr0: *const f32,
    sr1: *const f32,
    sr2: *const f32,
    sr3: *const f32,
    scratch: *mut f32,
    fy: f32,
    use_4tap: bool,
    save_mode: u8, // bilinear: 1=save sr0, 2=save sr1 | 4tap: 1=sr0 is scratch, 2=sr3 is scratch
    exp_f_active: bool,
) {
    let frame_w = ctx.frame_w;
    let ix0 = ctx.ix0;
    let vx_lo = ctx.vx_lo;
    let vx_hi = ctx.vx_hi;
    let exp_gain = ctx.exp_gain;
    let exp_off = ctx.exp_off;
    let om_fy = 1.0 - fy;
    let wy0 = ctx.wy0;
    let wy1 = ctx.wy1;
    let wy2 = ctx.wy2;
    let wy3 = ctx.wy3;

    let step_x = |x: usize, s: usize| {
        let interp = if use_4tap {
            let va = *(if save_mode == 1 { scratch.add(x) } else { sr0.add(s) });
            let vb = *sr1.add(s);
            let vc = *sr2.add(s);
            let vd = *(if save_mode == 2 { scratch.add(x) } else { sr3.add(s) });
            (wy0 * va + wy1 * vb + wy2 * vc + wy3 * vd).clamp(0.0, 1.0)
        } else {
            let v0 = *sr0.add(s);
            let v1 = *sr1.add(s);
            if save_mode != 0 {
                *scratch.add(x) = if save_mode == 1 { v0 } else { v1 };
            }
            (om_fy * v0 + fy * v1).clamp(0.0, 1.0)
        };
        *bg_row.add(x) = if exp_f_active {
            (interp * exp_gain + exp_off).clamp(0.0, 1.0)
        } else {
            interp
        };
    };

    if ix0 < 0 {
        let sh = (-ix0) as usize;
        let mut x = vx_hi;
        while x > vx_lo {
            x -= 1;
            step_x(x, x - sh);
        }
    } else {
        let ix0_u = ix0 as usize;
        for x in 0..vx_hi {
            step_x(x, x + ix0_u);
        }
    }
    if vx_lo > 0 {
        copy_f32_row_simd128(bg_row, cur_row, vx_lo);
    }
    if vx_hi < frame_w {
        copy_f32_row_simd128(bg_row.add(vx_hi), cur_row.add(vx_hi), frame_w - vx_hi);
    }
}

#[inline(never)]
unsafe fn advect_2d_vert_only_simd128(
    ctx: &CatmullCtx,
    bg_luma: *mut f32,
    scratch: *mut f32,
    cur_luma: *const f32,
    frame_h: usize,
    iy0: i32,
    fy: f32,
    exp_f_active: bool,
) {
    let frame_w = ctx.frame_w;
    let h_i = frame_h as i32;
    if iy0 >= 0 {
        let sm_pair = if iy0 == 0 { 1u8 } else { 0u8 };
        let mut y = 0usize;
        while y < frame_h {
            let sy = (y as i32) + iy0;
            let ro = y * frame_w;
            if sy + 1 >= h_i {
                copy_f32_row_simd128(bg_luma.add(ro), cur_luma.add(ro), frame_w);
                y += 1;
                continue;
            }
            let s = sy as usize;
            if y + 1 < frame_h && sy >= 1 && sy + 3 < h_i {
                vert_only_2rows_simd128(
                    ctx,
                    bg_luma.add(ro),
                    cur_luma.add(ro),
                    bg_luma.add(ro + frame_w),
                    cur_luma.add(ro + frame_w),
                    if sm_pair == 1 { scratch } else { bg_luma.add((s - 1) * frame_w) },
                    bg_luma.add(s * frame_w),
                    bg_luma.add((s + 1) * frame_w),
                    bg_luma.add((s + 2) * frame_w),
                    bg_luma.add((s + 3) * frame_w),
                    scratch,
                    sm_pair,
                    exp_f_active,
                );
                y += 2;
                continue;
            }
            if sy >= 1 && sy + 2 < h_i {
                vert_only_1row_simd128(
                    ctx,
                    bg_luma.add(ro),
                    cur_luma.add(ro),
                    if sm_pair == 1 { scratch } else { bg_luma.add((s - 1) * frame_w) },
                    bg_luma.add(s * frame_w),
                    bg_luma.add((s + 1) * frame_w),
                    bg_luma.add((s + 2) * frame_w),
                    scratch,
                    fy,
                    true,
                    sm_pair,
                    exp_f_active,
                );
            } else {
                let save_bi = if iy0 == 0 && sy == 0 { 1u8 } else { 0u8 };
                vert_only_1row_simd128(
                    ctx,
                    bg_luma.add(ro),
                    cur_luma.add(ro),
                    bg_luma.add(s * frame_w),
                    bg_luma.add((s + 1) * frame_w),
                    cur_luma,
                    cur_luma,
                    scratch,
                    fy,
                    false,
                    save_bi,
                    exp_f_active,
                );
            }
            y += 1;
        }
    } else {
        let sm_pair = if iy0 == -1 { 2u8 } else { 0u8 };
        let mut y_next = frame_h;
        while y_next > 0 {
            let y = y_next - 1;
            let sy = (y as i32) + iy0;
            let ro = y * frame_w;
            if sy < 0 || sy + 1 >= h_i {
                copy_f32_row_simd128(bg_luma.add(ro), cur_luma.add(ro), frame_w);
                y_next -= 1;
                continue;
            }
            let s = sy as usize;
            if y >= 1 && sy >= 2 && sy + 2 < h_i {
                vert_only_2rows_simd128(
                    ctx,
                    bg_luma.add(ro - frame_w),
                    cur_luma.add(ro - frame_w),
                    bg_luma.add(ro),
                    cur_luma.add(ro),
                    bg_luma.add((s - 2) * frame_w),
                    bg_luma.add((s - 1) * frame_w),
                    bg_luma.add(s * frame_w),
                    bg_luma.add((s + 1) * frame_w),
                    if sm_pair == 2 { scratch } else { bg_luma.add((s + 2) * frame_w) },
                    scratch,
                    sm_pair,
                    exp_f_active,
                );
                y_next -= 2;
                continue;
            }
            if sy >= 1 && sy + 2 < h_i {
                vert_only_1row_simd128(
                    ctx,
                    bg_luma.add(ro),
                    cur_luma.add(ro),
                    bg_luma.add((s - 1) * frame_w),
                    bg_luma.add(s * frame_w),
                    bg_luma.add((s + 1) * frame_w),
                    if sm_pair == 2 { scratch } else { bg_luma.add((s + 2) * frame_w) },
                    scratch,
                    fy,
                    true,
                    sm_pair,
                    exp_f_active,
                );
            } else {
                let save_bi = if iy0 == -1 && sy + 2 >= h_i { 2u8 } else { 0u8 };
                vert_only_1row_simd128(
                    ctx,
                    bg_luma.add(ro),
                    cur_luma.add(ro),
                    bg_luma.add(s * frame_w),
                    bg_luma.add((s + 1) * frame_w),
                    cur_luma,
                    cur_luma,
                    scratch,
                    fy,
                    false,
                    save_bi,
                    exp_f_active,
                );
            }
            y_next -= 1;
        }
    }
}

#[inline(always)]
unsafe fn drop_b_royal_blue_simd128(
    abs: v128,
    c: v128,
    p_raw: v128,
    rgb_mask: v128,
    rg_mask: v128,
    ff_mask: v128,
    b_mask: v128,
    b_250: v128,
    splat_230: v128,
) -> v128 {
    let b_hi = u32x4_ge(v128_and(u8x16_max(p_raw, c), b_mask), b_250);
    let pr_r = v128_and(p_raw, ff_mask);
    let pr_g = v128_and(u32x4_shr(p_raw, 8), ff_mask);
    let rg_low = u32x4_lt(u32x4_min(pr_r, pr_g), splat_230);
    let drop_b = v128_and(b_hi, rg_low);
    let chan_mask = v128_bitselect(rg_mask, rgb_mask, drop_b);
    v128_and(abs, chan_mask)
}

#[inline(always)]
unsafe fn white_clip_mask_u32_simd128(
    c: v128,
    p_raw: v128,
    rg_mask: v128,
    splat_251: v128,
    splat_253: v128,
) -> v128 {
    let pr_hi = u8x16_ge(p_raw, splat_251);
    let c_hi = u8x16_ge(c, splat_253);
    let pr_wc = u32x4_eq(v128_and(pr_hi, rg_mask), rg_mask);
    let c_wc = u32x4_eq(v128_and(c_hi, rg_mask), rg_mask);
    v128_or(pr_wc, c_wc)
}

#[inline(always)]
unsafe fn pack4_le_tol16_simd128(
    v0: v128,
    v1: v128,
    v2: v128,
    v3: v128,
    tol: v128,
    zero_i32: v128,
) -> v128 {
    i8x16_narrow_i16x8(
        i16x8_narrow_i32x4(
            i32x4_eq(u8x16_sub_sat(v0, tol), zero_i32),
            i32x4_eq(u8x16_sub_sat(v1, tol), zero_i32),
        ),
        i16x8_narrow_i32x4(
            i32x4_eq(u8x16_sub_sat(v2, tol), zero_i32),
            i32x4_eq(u8x16_sub_sat(v3, tol), zero_i32),
        ),
    )
}

#[inline(always)]
unsafe fn pack4_max_u8_16_simd128(v0: v128, v1: v128, v2: v128, v3: v128, _ff_mask: v128) -> v128 {
    let rg01 = i8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29>(v0, v1);
    let rg23 = i8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29>(v2, v3);
    let r16 = i64x2_shuffle::<0, 2>(rg01, rg23);
    let g16 = i64x2_shuffle::<1, 3>(rg01, rg23);
    let b01 = i8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18, 22, 26, 30>(v0, v1);
    let b23 = i8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18, 22, 26, 30>(v2, v3);
    let b16 = i64x2_shuffle::<0, 2>(b01, b23);
    u8x16_max(u8x16_max(r16, g16), b16)
}

#[inline(always)]
unsafe fn pack_max_rgb4_to_i32(v: v128) -> i32 {
    let mx = u8x16_max(u8x16_max(v, u32x4_shr(v, 8)), u32x4_shr(v, 16));
    let p = i8x16_shuffle::<0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(mx, mx);
    i32x4_extract_lane::<0>(p)
}

#[inline(always)]
unsafe fn pack_wc4_to_i32(wc: v128) -> i32 {
    let n16 = i16x8_narrow_i32x4(wc, wc);
    let n8 = i8x16_narrow_i16x8(n16, n16);
    i32x4_extract_lane::<0>(n8)
}

#[inline(always)]
unsafe fn hysteresis_pan_16_next_simd128(
    max_d16: v128,
    sp_opt: Option<(v128, v128, v128)>,
    wc16: v128,
    tol_hard: v128,
    tol_mid: v128,
    tol_soft: v128,
    one_u8: v128,
    inc_u8: v128,
    six_u8: v128,
    fourteen_u8: v128,
    twelve_u8: v128,
    twenty_u8: v128,
    exp_regime: bool,
    old_s: v128,
) -> (v128, v128) {
    let mut hard16 = u8x16_le(max_d16, tol_hard);
    let mut mid16 = u8x16_le(max_d16, tol_mid);
    let mut soft16 = u8x16_le(max_d16, tol_soft);

    if exp_regime {
        hard16 = v128_andnot(hard16, wc16);
    }
    if let Some((max_sp16, tol_span_soft, tol_span_reset)) = sp_opt {
        let span_soft_ok = u8x16_le(max_sp16, tol_span_soft);
        let span_reset_ok = u8x16_le(max_sp16, tol_span_reset);
        hard16 = v128_and(v128_and(hard16, span_soft_ok), span_reset_ok);
        mid16 = v128_and(mid16, span_reset_ok);
        soft16 = v128_and(soft16, span_reset_ok);
    }

    let inc = u8x16_min(u8x16_add_sat(old_s, inc_u8), twenty_u8);
    let mid_dec = u8x16_sub_sat(old_s, one_u8);
    let soft_dec = u8x16_sub_sat(u8x16_min(old_s, fourteen_u8), six_u8);
    let s_mid = v128_bitselect(inc, mid_dec, hard16);
    let s_soft = v128_bitselect(s_mid, soft_dec, mid16);
    let new_s = v128_and(s_soft, soft16);

    let eval_raw = u8x16_ge(new_s, twelve_u8);
    let eval16 = if exp_regime {
        v128_andnot(eval_raw, wc16)
    } else {
        eval_raw
    };
    (new_s, eval16)
}

#[inline(always)]
unsafe fn hysteresis_pan_16_compute_simd128(
    max_d16: v128,
    sp_opt: Option<(v128, v128, v128)>,
    wc16: v128,
    tol_hard: v128,
    tol_mid: v128,
    tol_soft: v128,
    one_u8: v128,
    inc_u8: v128,
    six_u8: v128,
    fourteen_u8: v128,
    twelve_u8: v128,
    twenty_u8: v128,
    exp_regime: bool,
    stat_ptr: *mut u8,
) -> v128 {
    let old_s = v128_load(stat_ptr as *const v128);
    let (new_s, eval16) = hysteresis_pan_16_next_simd128(
        max_d16,
        sp_opt,
        wc16,
        tol_hard,
        tol_mid,
        tol_soft,
        one_u8,
        inc_u8,
        six_u8,
        fourteen_u8,
        twelve_u8,
        twenty_u8,
        exp_regime,
        old_s,
    );
    v128_store(stat_ptr as *mut v128, new_s);
    eval16
}

#[inline(always)]
unsafe fn hysteresis_pan_16_packed_simd128(
    max_d16: v128,
    sp_opt: Option<(v128, v128, v128)>,
    wc16: v128,
    tol_hard: v128,
    tol_mid: v128,
    tol_soft: v128,
    one_u8: v128,
    inc_u8: v128,
    six_u8: v128,
    fourteen_u8: v128,
    twelve_u8: v128,
    twenty_u8: v128,
    exp_regime: bool,
    stat_ptr: *mut u8,
    eval_ptr: *mut u8,
) {
    let eval16 = hysteresis_pan_16_compute_simd128(
        max_d16,
        sp_opt,
        wc16,
        tol_hard,
        tol_mid,
        tol_soft,
        one_u8,
        inc_u8,
        six_u8,
        fourteen_u8,
        twelve_u8,
        twenty_u8,
        exp_regime,
        stat_ptr,
    );
    v128_store(eval_ptr as *mut v128, eval16);
}

#[inline(always)]
unsafe fn hysteresis_pan_16_simd128(
    abs0: v128,
    abs1: v128,
    abs2: v128,
    abs3: v128,
    sp0_ok_masked: Option<(v128, v128, v128, v128, v128, v128)>,
    wc16: v128,
    tol_hard: v128,
    tol_mid: v128,
    tol_soft: v128,
    ff_mask: v128,
    one_u8: v128,
    inc_u8: v128,
    six_u8: v128,
    fourteen_u8: v128,
    twelve_u8: v128,
    twenty_u8: v128,
    exp_regime: bool,
    stat_ptr: *mut u8,
    eval_ptr: *mut u8,
) {
    let max_d16 = pack4_max_u8_16_simd128(abs0, abs1, abs2, abs3, ff_mask);
    let sp_opt = sp0_ok_masked.map(|(sp0, sp1, sp2, sp3, tol_span_soft, tol_span_reset)| {
        (
            pack4_max_u8_16_simd128(sp0, sp1, sp2, sp3, ff_mask),
            tol_span_soft,
            tol_span_reset,
        )
    });
    hysteresis_pan_16_packed_simd128(
        max_d16,
        sp_opt,
        wc16,
        tol_hard,
        tol_mid,
        tol_soft,
        one_u8,
        inc_u8,
        six_u8,
        fourteen_u8,
        twelve_u8,
        twenty_u8,
        exp_regime,
        stat_ptr,
        eval_ptr,
    );
}

#[inline(always)]
unsafe fn hysteresis_pan_scalar(
    stat_ptr: *mut u8,
    eval_ptr: *mut u8,
    max_d: u8,
    inc_u8: u8,
    hard: u8,
    mid: u8,
    soft: u8,
    white_clip: bool,
) {
    let old_s = *stat_ptr;
    let ns = if max_d <= hard {
        old_s.saturating_add(inc_u8).min(20)
    } else if max_d <= mid {
        old_s.saturating_sub(1)
    } else if max_d <= soft {
        old_s.min(14).saturating_sub(6)
    } else {
        0
    };
    *stat_ptr = ns;
    *eval_ptr = if ns >= 12 && !white_clip { 0xFF } else { 0 };
}

#[inline(always)]
fn bilinear_weights_1d_u16(pan_dx_f: f32, pan_dy_f: f32) -> (i32, i32, u16, u16, u16, u16) {
    let shift_x = -pan_dx_f;
    let shift_y = -pan_dy_f;
    let ix0 = f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(shift_x))) as i32;
    let iy0 = f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(shift_y))) as i32;
    let fx = (shift_x - (ix0 as f32)).clamp(0.0, 1.0);
    let fy = (shift_y - (iy0 as f32)).clamp(0.0, 1.0);
    let wx1 = ((fx * 16.0 + 0.5) as u16).min(16);
    let wx0 = 16 - wx1;
    let wy1 = ((fy * 16.0 + 0.5) as u16).min(16);
    let wy0 = 16 - wy1;
    (ix0, iy0, wx0, wx1, wy0, wy1)
}

#[inline(always)]
unsafe fn horiz_interp_span_simd128(
    r_cur: v128,
    r_shift: v128,
    wx0_v: v128,
    wx1_v: v128,
) -> (v128, v128, v128, v128) {
    let hmax = u8x16_max(r_cur, r_shift);
    let hmin = u8x16_min(r_cur, r_shift);
    let h_lo = u16x8_add(
        u16x8_mul(u16x8_extend_low_u8x16(r_cur), wx0_v),
        u16x8_mul(u16x8_extend_low_u8x16(r_shift), wx1_v),
    );
    let h_hi = u16x8_add(
        u16x8_mul(u16x8_extend_high_u8x16(r_cur), wx0_v),
        u16x8_mul(u16x8_extend_high_u8x16(r_shift), wx1_v),
    );
    (h_lo, h_hi, hmax, hmin)
}

#[inline(always)]
unsafe fn vert_finish_with_span_exp_simd128<const EXP_ACTIVE: bool>(
    h0: (v128, v128, v128, v128),
    h1: (v128, v128, v128, v128),
    wy0_v: v128,
    wy1_v: v128,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
) -> (v128, v128, v128) {
    let cell_span = u8x16_sub_sat(u8x16_max(h0.2, h1.2), u8x16_min(h0.3, h1.3));
    let lo = u16x8_shr(
        u16x8_add(
            u16x8_add(u16x8_mul(h0.0, wy0_v), u16x8_mul(h1.0, wy1_v)),
            round_v,
        ),
        8,
    );
    let hi = u16x8_shr(
        u16x8_add(
            u16x8_add(u16x8_mul(h0.1, wy0_v), u16x8_mul(h1.1, wy1_v)),
            round_v,
        ),
        8,
    );
    let raw = u8x16_narrow_i16x8(lo, hi);
    let adj = if EXP_ACTIVE {
        let lo_adj = i16x8_add(lo, i16x8_add(i16x8_q15mulr_sat(lo, dg_q15_v), off_i16_v));
        let hi_adj = i16x8_add(hi, i16x8_add(i16x8_q15mulr_sat(hi, dg_q15_v), off_i16_v));
        u8x16_narrow_i16x8(lo_adj, hi_adj)
    } else {
        raw
    };
    (raw, adj, cell_span)
}

#[inline(always)]
unsafe fn subpx_eval_quad_reg<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
>(
    h0: (v128, v128, v128, v128),
    h1: (v128, v128, v128, v128),
    c_px: v128,
    wy0_v: v128,
    wy1_v: v128,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
    rgb_mask: v128,
    rg_mask: v128,
    ff_mask: v128,
    b_mask: v128,
    b_250: v128,
    splat_230: v128,
    splat_251: v128,
    splat_253: v128,
) -> (v128, v128, v128, v128) {
    let (p_raw, p_adj, sp) = vert_finish_with_span_exp_simd128::<EXP_ACTIVE>(
        h0, h1, wy0_v, wy1_v, round_v, dg_q15_v, off_i16_v,
    );
    let mut abs = v128_and(v128_or(u8x16_sub_sat(c_px, p_adj), u8x16_sub_sat(p_adj, c_px)), rgb_mask);
    let rgb_sum = if WRITE_RGB {
        if RGB_U16 {
            abs_rgb_to_sum_u16x4_simd128(abs)
        } else {
            abs_rgb_to_sum_f32x4_simd128(abs)
        }
    } else {
        u32x4_splat(0)
    };
    let wc = if EFF_EXP {
        abs = drop_b_royal_blue_simd128(
            abs, c_px, p_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230,
        );
        white_clip_mask_u32_simd128(c_px, p_raw, rg_mask, splat_251, splat_253)
    } else {
        u32x4_splat(0)
    };
    (abs, sp, wc, rgb_sum)
}

#[inline(always)]
unsafe fn subpx_eval_quad<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
>(
    h0: (v128, v128, v128, v128),
    h1: (v128, v128, v128, v128),
    c_px: v128,
    rgb_out_ptr: *mut u8,
    wy0_v: v128,
    wy1_v: v128,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
    rgb_mask: v128,
    rg_mask: v128,
    ff_mask: v128,
    b_mask: v128,
    b_250: v128,
    splat_230: v128,
    splat_251: v128,
    splat_253: v128,
) -> (v128, v128, v128) {
    let (abs, sp, wc, rgb_sum) = subpx_eval_quad_reg::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
        h0, h1, c_px, wy0_v, wy1_v, round_v, dg_q15_v, off_i16_v, rgb_mask, rg_mask, ff_mask,
        b_mask, b_250, splat_230, splat_251, splat_253,
    );
    if WRITE_RGB {
        if RGB_U16 {
            v128_store64_lane::<0>(rgb_sum, rgb_out_ptr as *mut u64);
        } else {
            v128_store(rgb_out_ptr as *mut v128, rgb_sum);
        }
    }
    (abs, sp, wc)
}

#[inline(always)]
unsafe fn subpx_zero_rows<const WRITE_RGB: bool>(
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_sum_out: *mut f32,
    start_row: usize,
    end_row: usize,
    frame_w: usize,
) {
    if end_row <= start_row {
        return;
    }
    let off = start_row * frame_w;
    let count = (end_row - start_row) * frame_w;
    core::slice::from_raw_parts_mut(stat_streak_ptr.add(off), count).fill(0);
    core::slice::from_raw_parts_mut(eval_mask_ptr.add(off), count).fill(0);
    if WRITE_RGB {
        core::slice::from_raw_parts_mut(rgb_sum_out.add(off), count).fill(0.0);
    }
}

#[inline(always)]
unsafe fn subpx_scalar_tail<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_row_out: *mut u8,
    row_c: usize,
    row_p0: usize,
    row_bytes: usize,
    _frame_w: usize,
    w_i: i32,
    ix0: i32,
    x_start: usize,
    x_end: usize,
    w00: u16,
    w10: u16,
    w01: u16,
    w11: u16,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
    mid_u8: u8,
    soft_u8: u8,
    span_soft_u8: u8,
    span_reset_u8: u8,
) {
    let mut x = x_start;
    while x < x_end {
        let sx = (x as i32) + ix0;
        let idx = row_c + x;
        if sx >= 0 && sx + 1 < w_i {
            let c_b = idx * 4;
            let p00_b = (row_p0 + (sx as usize)) * 4;
            let p10_b = p00_b + 4;
            let p01_b = p00_b + row_bytes;
            let p11_b = p01_b + 4;
            let mut max_d = 0u8;
            let mut max_span = 0u8;
            let mut diff_sum = 0u32;
            let mut pv_raw3 = [0u8; 3];
            for ch in 0..3usize {
                let v00 = *prev_rgba.add(p00_b + ch);
                let v10 = *prev_rgba.add(p10_b + ch);
                let v01 = *prev_rgba.add(p01_b + ch);
                let v11 = *prev_rgba.add(p11_b + ch);
                let sp = v00.max(v10).max(v01).max(v11) - v00.min(v10).min(v01).min(v11);
                if sp > max_span {
                    max_span = sp;
                }
                let pv = ((w00 as u32) * (v00 as u32)
                    + (w10 as u32) * (v10 as u32)
                    + (w01 as u32) * (v01 as u32)
                    + (w11 as u32) * (v11 as u32)
                    + 128)
                    >> 8;
                let pv_raw = pv as u8;
                pv_raw3[ch] = pv_raw;
                let c_ch = *curr_rgba.add(c_b + ch);
                let pv_u8 = if EXP_ACTIVE {
                    apply_exposure_u8_scalar(pv_raw, dg_q15, off_i16)
                } else {
                    pv_raw
                };
                let d = c_ch.abs_diff(pv_u8);
                diff_sum += d as u32;
                if ch == 2
                    && EFF_EXP
                    && (pv_raw >= 250 || c_ch >= 250)
                    && pv_raw3[0].min(pv_raw3[1]) < 230
                {
                    continue;
                }
                if d > max_d {
                    max_d = d;
                }
            }
            if WRITE_RGB {
                if RGB_U16 {
                    *(rgb_row_out as *mut u16).add(x) = diff_sum as u16;
                } else {
                    *(rgb_row_out as *mut f32).add(x) = diff_sum as f32;
                }
            }
            let white_clip = EFF_EXP
                && ((pv_raw3[0] >= 251 && pv_raw3[1] >= 251)
                    || (*curr_rgba.add(c_b) >= 253 && *curr_rgba.add(c_b + 1) >= 253));
            if white_clip && max_d <= hard_u8 {
                max_d = mid_u8;
            }
            if max_span > span_reset_u8 {
                max_d = 255;
            } else if max_span > span_soft_u8 && max_d <= hard_u8 {
                max_d = mid_u8;
            }
            hysteresis_pan_scalar(
                stat_streak_ptr.add(idx),
                eval_mask_ptr.add(idx),
                max_d,
                2,
                hard_u8,
                mid_u8,
                soft_u8,
                white_clip,
            );
        } else {
            *stat_streak_ptr.add(idx) = 0;
            *eval_mask_ptr.add(idx) = 0;
            if WRITE_RGB {
                if RGB_U16 {
                    *(rgb_row_out as *mut u16).add(x) = 0;
                } else {
                    *(rgb_row_out as *mut f32).add(x) = 0.0;
                }
            }
        }
        x += 1;
    }
}

#[inline(always)]
unsafe fn subpx_eval_quad_2d<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
>(
    r0: v128,
    s0: v128,
    r1: v128,
    s1: v128,
    c_px: v128,
    rgb_out_ptr: *mut u8,
    w00_u8: v128,
    w10_u8: v128,
    w01_u8: v128,
    w11_u8: v128,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
    rgb_mask: v128,
    rg_mask: v128,
    ff_mask: v128,
    b_mask: v128,
    b_250: v128,
    splat_230: v128,
    splat_251: v128,
    splat_253: v128,
) -> (v128, v128, v128) {
    let sp = u8x16_sub_sat(
        u8x16_max(u8x16_max(r0, s0), u8x16_max(r1, s1)),
        u8x16_min(u8x16_min(r0, s0), u8x16_min(r1, s1)),
    );
    let lo_0 = u16x8_add(
        u16x8_extmul_low_u8x16(r0, w00_u8),
        u16x8_extmul_low_u8x16(s0, w10_u8),
    );
    let lo_1 = u16x8_add(
        u16x8_extmul_low_u8x16(r1, w01_u8),
        u16x8_extmul_low_u8x16(s1, w11_u8),
    );
    let hi_0 = u16x8_add(
        u16x8_extmul_high_u8x16(r0, w00_u8),
        u16x8_extmul_high_u8x16(s0, w10_u8),
    );
    let hi_1 = u16x8_add(
        u16x8_extmul_high_u8x16(r1, w01_u8),
        u16x8_extmul_high_u8x16(s1, w11_u8),
    );
    let lo = u16x8_shr(u16x8_add(u16x8_add(lo_0, lo_1), round_v), 8);
    let hi = u16x8_shr(u16x8_add(u16x8_add(hi_0, hi_1), round_v), 8);
    let p_raw = u8x16_narrow_i16x8(lo, hi);
    let p_adj = if EXP_ACTIVE {
        let lo_adj = i16x8_add(lo, i16x8_add(i16x8_q15mulr_sat(lo, dg_q15_v), off_i16_v));
        let hi_adj = i16x8_add(hi, i16x8_add(i16x8_q15mulr_sat(hi, dg_q15_v), off_i16_v));
        u8x16_narrow_i16x8(lo_adj, hi_adj)
    } else {
        p_raw
    };
    let mut abs = v128_and(v128_or(u8x16_sub_sat(c_px, p_adj), u8x16_sub_sat(p_adj, c_px)), rgb_mask);
    if WRITE_RGB {
        if RGB_U16 {
            v128_store64_lane::<0>(abs_rgb_to_sum_u16x4_simd128(abs), rgb_out_ptr as *mut u64);
        } else {
            v128_store(rgb_out_ptr as *mut v128, abs_rgb_to_sum_f32x4_simd128(abs));
        }
    }
    let wc = if EFF_EXP {
        abs = drop_b_royal_blue_simd128(
            abs, c_px, p_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230,
        );
        white_clip_mask_u32_simd128(c_px, p_raw, rg_mask, splat_251, splat_253)
    } else {
        u32x4_splat(0)
    };
    (abs, sp, wc)
}

#[inline(always)]
unsafe fn deinterleave_rgb16_simd128(ptr: *const u8) -> (v128, v128, v128) {
    let v0 = v128_load(ptr as *const v128);
    let v1 = v128_load(ptr.add(16) as *const v128);
    let v2 = v128_load(ptr.add(32) as *const v128);
    let v3 = v128_load(ptr.add(48) as *const v128);
    let rb01 = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(v0, v1);
    let ga01 = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(v0, v1);
    let rb23 = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(v2, v3);
    let ga23 = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(v2, v3);
    let r16 = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(rb01, rb23);
    let b16 = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(rb01, rb23);
    let g16 = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(ga01, ga23);
    (r16, g16, b16)
}

#[inline(always)]
unsafe fn eval_chan_16_planar<const EXP_ACTIVE: bool, const EXTMUL_MODE: u8>(
    r0: v128,
    s0: v128,
    r1: v128,
    s1: v128,
    c_ch: v128,
    w00_u8: v128,
    w10_u8: v128,
    w01_u8: v128,
    w11_u8: v128,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
) -> (v128, v128, v128) {
    let sp = u8x16_sub_sat(
        u8x16_max(u8x16_max(r0, s0), u8x16_max(r1, s1)),
        u8x16_min(u8x16_min(r0, s0), u8x16_min(r1, s1)),
    );
    let (lo, hi) = match EXTMUL_MODE {
        // w10 == 0 && w11 == 0
        1 => {
            let lo_s = u16x8_add(
                u16x8_extmul_low_u8x16(r0, w00_u8),
                u16x8_extmul_low_u8x16(r1, w01_u8),
            );
            let hi_s = u16x8_add(
                u16x8_extmul_high_u8x16(r0, w00_u8),
                u16x8_extmul_high_u8x16(r1, w01_u8),
            );
            (u16x8_shr(u16x8_add(lo_s, round_v), 8), u16x8_shr(u16x8_add(hi_s, round_v), 8))
        }
        // w01 == 0 && w11 == 0
        2 => {
            let lo_s = u16x8_add(
                u16x8_extmul_low_u8x16(r0, w00_u8),
                u16x8_extmul_low_u8x16(s0, w10_u8),
            );
            let hi_s = u16x8_add(
                u16x8_extmul_high_u8x16(r0, w00_u8),
                u16x8_extmul_high_u8x16(s0, w10_u8),
            );
            (u16x8_shr(u16x8_add(lo_s, round_v), 8), u16x8_shr(u16x8_add(hi_s, round_v), 8))
        }
        // w00 == 0 && w01 == 0
        3 => {
            let lo_s = u16x8_add(
                u16x8_extmul_low_u8x16(s0, w10_u8),
                u16x8_extmul_low_u8x16(s1, w11_u8),
            );
            let hi_s = u16x8_add(
                u16x8_extmul_high_u8x16(s0, w10_u8),
                u16x8_extmul_high_u8x16(s1, w11_u8),
            );
            (u16x8_shr(u16x8_add(lo_s, round_v), 8), u16x8_shr(u16x8_add(hi_s, round_v), 8))
        }
        // w00 == 0 && w10 == 0
        4 => {
            let lo_s = u16x8_add(
                u16x8_extmul_low_u8x16(r1, w01_u8),
                u16x8_extmul_low_u8x16(s1, w11_u8),
            );
            let hi_s = u16x8_add(
                u16x8_extmul_high_u8x16(r1, w01_u8),
                u16x8_extmul_high_u8x16(s1, w11_u8),
            );
            (u16x8_shr(u16x8_add(lo_s, round_v), 8), u16x8_shr(u16x8_add(hi_s, round_v), 8))
        }
        // General 2D bilinear
        _ => {
            let lo_0 = u16x8_add(
                u16x8_extmul_low_u8x16(r0, w00_u8),
                u16x8_extmul_low_u8x16(s0, w10_u8),
            );
            let lo_1 = u16x8_add(
                u16x8_extmul_low_u8x16(r1, w01_u8),
                u16x8_extmul_low_u8x16(s1, w11_u8),
            );
            let hi_0 = u16x8_add(
                u16x8_extmul_high_u8x16(r0, w00_u8),
                u16x8_extmul_high_u8x16(s0, w10_u8),
            );
            let hi_1 = u16x8_add(
                u16x8_extmul_high_u8x16(r1, w01_u8),
                u16x8_extmul_high_u8x16(s1, w11_u8),
            );
            (
                u16x8_shr(u16x8_add(u16x8_add(lo_0, lo_1), round_v), 8),
                u16x8_shr(u16x8_add(u16x8_add(hi_0, hi_1), round_v), 8),
            )
        }
    };
    let p_raw = u8x16_narrow_i16x8(lo, hi);
    let p_adj = if EXP_ACTIVE {
        let lo_adj = i16x8_add(lo, i16x8_add(i16x8_q15mulr_sat(lo, dg_q15_v), off_i16_v));
        let hi_adj = i16x8_add(hi, i16x8_add(i16x8_q15mulr_sat(hi, dg_q15_v), off_i16_v));
        u8x16_narrow_i16x8(lo_adj, hi_adj)
    } else {
        p_raw
    };
    let abs = v128_or(u8x16_sub_sat(c_ch, p_adj), u8x16_sub_sat(p_adj, c_ch));
    (abs, sp, p_raw)
}

#[repr(C)]
struct SubpxRowCtx {
    w00_u8: v128,
    w10_u8: v128,
    w01_u8: v128,
    w11_u8: v128,
    wx0_v: v128,
    wx1_v: v128,
    wy0_v: v128,
    wy1_v: v128,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
    zero_v: v128,
    tol_hard: v128,
    tol_mid: v128,
    tol_soft: v128,
    tol_span_soft: v128,
    tol_span_reset: v128,
    one_u8: v128,
    two_u8: v128,
    six_u8: v128,
    fourteen_u8: v128,
    twelve_u8: v128,
    twenty_u8: v128,
    u230: v128,
    u250: v128,
    u251: v128,
    u253: v128,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    h_cache_ptr: *mut v128,
    frame_w: usize,
    w_i: i32,
    row_bytes: usize,
    ix0: i32,
    iy0: i32,
    x_lo: usize,
    sx_lo: usize,
    n_blocks16: usize,
    x_simd_end: usize,
    quad_lim: usize,
    w00: u16,
    w10: u16,
    w01: u16,
    w11: u16,
    max_w: u16,
    extmul_mode: u8,
    write_h1: bool,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
    mid_u8: u8,
    soft_u8: u8,
    span_soft_u8: u8,
    span_reset_u8: u8,
}

impl SubpxRowCtx {
    #[inline(always)]
    unsafe fn new<const EFF_EXP: bool>(
        curr_rgba: *const u8,
        prev_rgba: *const u8,
        stat_streak_ptr: *mut u8,
        eval_mask_ptr: *mut u8,
        h_cache_ptr: *mut v128,
        frame_w: usize,
        w_i: i32,
        row_bytes: usize,
        ix0: i32,
        iy0: i32,
        x_lo: usize,
        n_blocks16: usize,
        x_simd_end: usize,
        wx0: u16,
        wx1: u16,
        wy0: u16,
        wy1: u16,
        dg_q15: i16,
        off_i16: i16,
        hard_u8: u8,
        mid_u8: u8,
        soft_u8: u8,
        span_soft_u8: u8,
        span_reset_u8: u8,
        allow_h_cache: bool,
    ) -> Self {
        let w00 = wx0 * wy0;
        let w10 = wx1 * wy0;
        let w01 = wx0 * wy1;
        let w11 = wx1 * wy1;
        let max_w = w00.max(w10).max(w01).max(w11);
        let extmul_mode: u8 = if w10 == 0 && w11 == 0 {
            1
        } else if w01 == 0 && w11 == 0 {
            2
        } else if w00 == 0 && w01 == 0 {
            3
        } else if w00 == 0 && w10 == 0 {
            4
        } else {
            0
        };
        let write_h1 = !h_cache_ptr.is_null() && (max_w > 255 || (allow_h_cache && n_blocks16 <= 40));
        let quad_lim = if w_i - 1 - ix0 > 0 {
            ((w_i - 1 - ix0) as usize).min(frame_w)
        } else {
            0
        };
        let zero_v = i32x4_splat(0);
        Self {
            w00_u8: u8x16_splat(w00 as u8),
            w10_u8: u8x16_splat(w10 as u8),
            w01_u8: u8x16_splat(w01 as u8),
            w11_u8: u8x16_splat(w11 as u8),
            wx0_v: u16x8_splat(wx0),
            wx1_v: u16x8_splat(wx1),
            wy0_v: u16x8_splat(wy0),
            wy1_v: u16x8_splat(wy1),
            round_v: u16x8_splat(128),
            dg_q15_v: i16x8_splat(dg_q15),
            off_i16_v: i16x8_splat(off_i16),
            zero_v,
            tol_hard: u8x16_splat(hard_u8),
            tol_mid: u8x16_splat(mid_u8),
            tol_soft: u8x16_splat(soft_u8),
            tol_span_soft: u8x16_splat(span_soft_u8),
            tol_span_reset: u8x16_splat(span_reset_u8),
            one_u8: u8x16_splat(1),
            two_u8: u8x16_splat(2),
            six_u8: u8x16_splat(6),
            fourteen_u8: u8x16_splat(14),
            twelve_u8: u8x16_splat(12),
            twenty_u8: u8x16_splat(20),
            u230: if EFF_EXP { u8x16_splat(230) } else { zero_v },
            u250: if EFF_EXP { u8x16_splat(250) } else { zero_v },
            u251: if EFF_EXP { u8x16_splat(251) } else { zero_v },
            u253: if EFF_EXP { u8x16_splat(253) } else { zero_v },
            curr_rgba,
            prev_rgba,
            stat_streak_ptr,
            eval_mask_ptr,
            h_cache_ptr,
            frame_w,
            w_i,
            row_bytes,
            ix0,
            iy0,
            x_lo,
            sx_lo: ((x_lo as i32) + ix0) as usize,
            n_blocks16,
            x_simd_end,
            quad_lim,
            w00,
            w10,
            w01,
            w11,
            max_w,
            extmul_mode,
            write_h1,
            dg_q15,
            off_i16,
            hard_u8,
            mid_u8,
            soft_u8,
            span_soft_u8,
            span_reset_u8,
        }
    }
}

#[inline(always)]
unsafe fn subpx_row_extmul_2d_inner<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
    const REUSE_H0: bool,
    const WRITE_H1: bool,
    const EXTMUL_MODE: u8,
>(
    ctx: &SubpxRowCtx,
    p0_ptr: *const u8,
    p1_ptr: *const u8,
    rgb_row_out: *mut u8,
    row_c: usize,
) {
    let w00_u8 = ctx.w00_u8;
    let w10_u8 = ctx.w10_u8;
    let w01_u8 = ctx.w01_u8;
    let w11_u8 = ctx.w11_u8;
    let round_v = u16x8_splat(128);
    let dg_q15_v = ctx.dg_q15_v;
    let off_i16_v = ctx.off_i16_v;

    let zero_v = i32x4_splat(0);
    let tol_hard = ctx.tol_hard;
    let tol_mid = ctx.tol_mid;
    let tol_soft = ctx.tol_soft;
    let tol_span_soft = ctx.tol_span_soft;
    let tol_span_reset = ctx.tol_span_reset;
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let fourteen_u8 = u8x16_splat(14);
    let twelve_u8 = u8x16_splat(12);
    let twenty_u8 = u8x16_splat(20);
    let u230 = u8x16_splat(230);
    let u250 = u8x16_splat(250);
    let u251 = u8x16_splat(251);
    let u253 = u8x16_splat(253);

    let stat_streak_ptr = ctx.stat_streak_ptr;
    let eval_mask_ptr = ctx.eval_mask_ptr;
    let h_cache_ptr = ctx.h_cache_ptr;
    let x_lo = ctx.x_lo;
    let n_blocks16 = ctx.n_blocks16;
    let c_row_ptr = ctx.curr_rgba.add(row_c * 4);
    if n_blocks16 == 0 {
        return;
    }

    macro_rules! eval_and_store_blk {
        ($x_blk:expr, $out_off:expr,
         $r0_r:expr, $s0_r:expr, $r1_r:expr, $s1_r:expr, $after_r:block,
         $r0_g:expr, $s0_g:expr, $r1_g:expr, $s1_g:expr, $after_g:block,
         $r0_b:expr, $s0_b:expr, $r1_b:expr, $s1_b:expr, $after_b:block) => {{
            let (c_r, c_g, c_b) = deinterleave_rgb16_simd128(c_row_ptr.add($x_blk * 4));
            let s0_r = $s0_r;
            let s1_r = $s1_r;
            let (abs_r, sp_r, p_raw_r) = eval_chan_16_planar::<EXP_ACTIVE, EXTMUL_MODE>(
                $r0_r, s0_r, $r1_r, s1_r, c_r, w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            );
            $after_r
            let mut max_d16 = abs_r;
            let mut max_sp16 = sp_r;
            let mut sum_lo = if WRITE_RGB { u16x8_extend_low_u8x16(abs_r) } else { zero_v };
            let mut sum_hi = if WRITE_RGB { u16x8_extend_high_u8x16(abs_r) } else { zero_v };

            let s0_g = $s0_g;
            let s1_g = $s1_g;
            let (abs_g, sp_g, p_raw_g) = eval_chan_16_planar::<EXP_ACTIVE, EXTMUL_MODE>(
                $r0_g, s0_g, $r1_g, s1_g, c_g, w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            );
            $after_g
            max_d16 = u8x16_max(max_d16, abs_g);
            max_sp16 = u8x16_max(max_sp16, sp_g);
            if WRITE_RGB {
                sum_lo = u16x8_add(sum_lo, u16x8_extend_low_u8x16(abs_g));
                sum_hi = u16x8_add(sum_hi, u16x8_extend_high_u8x16(abs_g));
            }

            let s0_b = $s0_b;
            let s1_b = $s1_b;
            let (mut abs_b, sp_b, p_raw_b) = eval_chan_16_planar::<EXP_ACTIVE, EXTMUL_MODE>(
                $r0_b, s0_b, $r1_b, s1_b, c_b, w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            );
            $after_b
            max_sp16 = u8x16_max(max_sp16, sp_b);

            if WRITE_RGB {
                sum_lo = u16x8_add(sum_lo, u16x8_extend_low_u8x16(abs_b));
                sum_hi = u16x8_add(sum_hi, u16x8_extend_high_u8x16(abs_b));
                if RGB_U16 {
                    let out_p = rgb_row_out.add($x_blk * 2);
                    v128_store(out_p as *mut v128, sum_lo);
                    v128_store(out_p.add(16) as *mut v128, sum_hi);
                } else {
                    let out_p = (rgb_row_out as *mut f32).add($x_blk);
                    v128_store(out_p as *mut v128, f32x4_convert_u32x4(u32x4_extend_low_u16x8(sum_lo)));
                    v128_store(out_p.add(4) as *mut v128, f32x4_convert_u32x4(u32x4_extend_high_u16x8(sum_lo)));
                    v128_store(out_p.add(8) as *mut v128, f32x4_convert_u32x4(u32x4_extend_low_u16x8(sum_hi)));
                    v128_store(out_p.add(12) as *mut v128, f32x4_convert_u32x4(u32x4_extend_high_u16x8(sum_hi)));
                }
            }

            let wc16 = if EFF_EXP {
                let b_hi = u8x16_ge(u8x16_max(p_raw_b, c_b), u250);
                let rg_low = u8x16_lt(u8x16_min(p_raw_r, p_raw_g), u230);
                abs_b = v128_andnot(abs_b, v128_and(b_hi, rg_low));
                let pr_wc = v128_and(u8x16_ge(p_raw_r, u251), u8x16_ge(p_raw_g, u251));
                let c_wc = v128_and(u8x16_ge(c_r, u253), u8x16_ge(c_g, u253));
                v128_or(pr_wc, c_wc)
            } else {
                zero_v
            };

            max_d16 = u8x16_max(max_d16, abs_b);
            hysteresis_pan_16_packed_simd128(
                max_d16,
                Some((max_sp16, tol_span_soft, tol_span_reset)),
                wc16,
                tol_hard,
                tol_mid,
                tol_soft,
                one_u8,
                two_u8,
                six_u8,
                fourteen_u8,
                twelve_u8,
                twenty_u8,
                EFF_EXP,
                stat_streak_ptr.add($out_off),
                eval_mask_ptr.add($out_off),
            );
        }};
    }

    let (mut r0_r, mut r0_g, mut r0_b) = if REUSE_H0 {
        (v128_load(h_cache_ptr), v128_load(h_cache_ptr.add(1)), v128_load(h_cache_ptr.add(2)))
    } else {
        deinterleave_rgb16_simd128(p0_ptr)
    };
    let (mut r1_r, mut r1_g, mut r1_b) = deinterleave_rgb16_simd128(p1_ptr);

    let last_blk = n_blocks16 - 1;
    for blk in 0..last_blk {
        let px_off = blk * 64;
        let x_blk = x_lo + blk * 16;
        if WRITE_H1 {
            let hc = h_cache_ptr.add(blk * 3);
            v128_store(hc, r1_r);
            v128_store(hc.add(1), r1_g);
            v128_store(hc.add(2), r1_b);
        }
        let (nr1_r, nr1_g, nr1_b) = deinterleave_rgb16_simd128(p1_ptr.add(px_off + 64));
        if REUSE_H0 {
            let hc_n = h_cache_ptr.add((blk + 1) * 3);
            let nr0_r = v128_load(hc_n);
            let nr0_g;
            let nr0_b;
            eval_and_store_blk!(
                x_blk, row_c + x_blk,
                r0_r, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_r, nr0_r),
                r1_r, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_r, nr1_r),
                { r0_r = nr0_r; r1_r = nr1_r; },
                r0_g, { nr0_g = v128_load(hc_n.add(1)); i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_g, nr0_g) },
                r1_g, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_g, nr1_g),
                { r0_g = nr0_g; r1_g = nr1_g; },
                r0_b, { nr0_b = v128_load(hc_n.add(2)); i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_b, nr0_b) },
                r1_b, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_b, nr1_b),
                { r0_b = nr0_b; r1_b = nr1_b; }
            );
        } else {
            let (nr0_r, nr0_g, nr0_b) = deinterleave_rgb16_simd128(p0_ptr.add(px_off + 64));
            eval_and_store_blk!(
                x_blk, row_c + x_blk,
                r0_r, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_r, nr0_r),
                r1_r, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_r, nr1_r),
                { r0_r = nr0_r; r1_r = nr1_r; },
                r0_g, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_g, nr0_g),
                r1_g, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_g, nr1_g),
                { r0_g = nr0_g; r1_g = nr1_g; },
                r0_b, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_b, nr0_b),
                r1_b, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_b, nr1_b),
                { r0_b = nr0_b; r1_b = nr1_b; }
            );
        }
    }

    let px_off = last_blk * 64;
    let x_blk = x_lo + last_blk * 16;
    let p0_tail = if REUSE_H0 {
        v128_load32_zero(h_cache_ptr.add(n_blocks16 * 3) as *const u32)
    } else {
        v128_load32_zero(p0_ptr.add(px_off + 64) as *const u32)
    };
    let p1_tail = v128_load32_zero(p1_ptr.add(px_off + 64) as *const u32);
    if WRITE_H1 {
        let hc = h_cache_ptr.add(last_blk * 3);
        v128_store(hc, r1_r);
        v128_store(hc.add(1), r1_g);
        v128_store(hc.add(2), r1_b);
        v128_store32_lane::<0>(p1_tail, h_cache_ptr.add(n_blocks16 * 3) as *mut u32);
    }

    eval_and_store_blk!(
        x_blk, row_c + x_blk,
        r0_r, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_r, p0_tail),
        r1_r, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_r, p1_tail),
        {},
        r0_g, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17>(r0_g, p0_tail),
        r1_g, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17>(r1_g, p1_tail),
        {},
        r0_b, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18>(r0_b, p0_tail),
        r1_b, i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18>(r1_b, p1_tail),
        {}
    );
}

#[inline(never)]
unsafe fn subpx_row_extmul_2d<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
    const REUSE_H0: bool,
    const WRITE_H1: bool,
    const EXTMUL_MODE: u8,
>(
    ctx: &SubpxRowCtx,
    p0_ptr: *const u8,
    p1_ptr: *const u8,
    rgb_row_out: *mut u8,
    row_c: usize,
) {
    subpx_row_extmul_2d_inner::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16, REUSE_H0, WRITE_H1, EXTMUL_MODE>(
        ctx, p0_ptr, p1_ptr, rgb_row_out, row_c,
    );
}

#[inline(never)]
unsafe fn subpx_row_blk_chunk<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
    const REUSE_H0: bool,
    const WRITE_H1: bool,
>(
    curr_rgba: *const u8,
    p0_ptr: *const u8,
    p1_ptr: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_row_out: *mut u8,
    h_cache_ptr: *mut v128,
    row_c: usize,
    x_lo: usize,
    n_q: usize,
    blk_lo: usize,
    blk_cnt: usize,
    is_tail_end: bool,
    wx0_v: v128,
    wx1_v: v128,
    wy0_v: v128,
    wy1_v: v128,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
    hard_u8: u8,
    mid_u8: u8,
    soft_u8: u8,
    span_soft_u8: u8,
    span_reset_u8: u8,
) {
    let zero_i32 = i32x4_splat(0);
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let rg_mask = if EFF_EXP { u32x4_splat(0x0000_FFFF) } else { zero_i32 };
    let ff_mask = u32x4_splat(0xFF);
    let b_mask = if EFF_EXP { u32x4_splat(0x00FF_0000) } else { zero_i32 };
    let b_250 = if EFF_EXP { u32x4_splat(250 << 16) } else { zero_i32 };
    let splat_230 = if EFF_EXP { u32x4_splat(230) } else { zero_i32 };
    let splat_251 = if EFF_EXP { u8x16_splat(251) } else { zero_i32 };
    let splat_253 = if EFF_EXP { u8x16_splat(253) } else { zero_i32 };

    let tol_hard = u8x16_splat(hard_u8);
    let tol_mid = u8x16_splat(mid_u8);
    let tol_soft = u8x16_splat(soft_u8);
    let tol_span_soft = u8x16_splat(span_soft_u8);
    let tol_span_reset = u8x16_splat(span_reset_u8);
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let fourteen_u8 = u8x16_splat(14);
    let twelve_u8 = u8x16_splat(12);
    let twenty_u8 = u8x16_splat(20);
    let out_base = row_c + x_lo + blk_lo * 16;

    let c_row_ptr = curr_rgba.add(row_c * 4);
    let q_start = blk_lo * 4;
    let mut r1_cur = v128_load(p1_ptr.add(q_start * 16) as *const v128);
    let mut r0_cur = if !REUSE_H0 {
        v128_load(p0_ptr.add(q_start * 16) as *const v128)
    } else {
        zero_i32
    };

    macro_rules! eval_quad_k {
        (@upd false, $dst:ident, $val:ident) => {
            $dst = $val;
        };
        (@upd true, $dst:ident, $val:ident) => {};
        ($q_expr:expr, $px_expr:expr, $is_last:tt) => {{
            let q = $q_expr;
            let px_x = $px_expr;
            let r1_nxt = if $is_last {
                v128_load32_zero(p1_ptr.add(n_q * 16) as *const u32)
            } else {
                v128_load(p1_ptr.add((q + 1) * 16) as *const v128)
            };
            let r1_shf = i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(
                r1_cur, r1_nxt,
            );
            let h1 = horiz_interp_span_simd128(r1_cur, r1_shf, wx0_v, wx1_v);
            eval_quad_k!(@upd $is_last, r1_cur, r1_nxt);

            let h0 = if REUSE_H0 {
                let hc = h_cache_ptr.add(q * 4);
                let cached = (
                    v128_load(hc),
                    v128_load(hc.add(1)),
                    v128_load(hc.add(2)),
                    v128_load(hc.add(3)),
                );
                if WRITE_H1 {
                    v128_store(hc, h1.0);
                    v128_store(hc.add(1), h1.1);
                    v128_store(hc.add(2), h1.2);
                    v128_store(hc.add(3), h1.3);
                }
                cached
            } else {
                let r0_nxt = if $is_last {
                    v128_load32_zero(p0_ptr.add(n_q * 16) as *const u32)
                } else {
                    v128_load(p0_ptr.add((q + 1) * 16) as *const v128)
                };
                let r0_shf =
                    i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(
                        r0_cur, r0_nxt,
                    );
                let h0_c = horiz_interp_span_simd128(r0_cur, r0_shf, wx0_v, wx1_v);
                eval_quad_k!(@upd $is_last, r0_cur, r0_nxt);
                if WRITE_H1 {
                    let hc = h_cache_ptr.add(q * 4);
                    v128_store(hc, h1.0);
                    v128_store(hc.add(1), h1.1);
                    v128_store(hc.add(2), h1.2);
                    v128_store(hc.add(3), h1.3);
                }
                h0_c
            };

            let ca = v128_load(c_row_ptr.add(px_x * 4) as *const v128);
            let rgb_p = if WRITE_RGB {
                if RGB_U16 {
                    rgb_row_out.add(px_x * 2)
                } else {
                    rgb_row_out.add(px_x * 4)
                }
            } else {
                core::ptr::null_mut()
            };
            subpx_eval_quad::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
                h0,
                h1,
                ca,
                rgb_p,
                wy0_v,
                wy1_v,
                round_v,
                dg_q15_v,
                off_i16_v,
                rgb_mask,
                rg_mask,
                ff_mask,
                b_mask,
                b_250,
                splat_230,
                splat_251,
                splat_253,
            )
        }};
    }

    macro_rules! eval_block {
        ($blk_local:expr, $tail_q3:tt) => {{
            let blk = blk_lo + $blk_local;
            let q_base = blk * 4;
            let x_blk = x_lo + blk * 16;
            let (a0, s0, w0) = eval_quad_k!(q_base, x_blk, false);
            let (a1, s1, w1) = eval_quad_k!(q_base + 1, x_blk + 4, false);
            let (a2, s2, w2) = eval_quad_k!(q_base + 2, x_blk + 8, false);
            let (a3, s3, w3) = eval_quad_k!(q_base + 3, x_blk + 12, $tail_q3);
            let max_d16 = pack4_max_u8_16_simd128(a0, a1, a2, a3, ff_mask);
            let max_sp16 = pack4_max_u8_16_simd128(s0, s1, s2, s3, ff_mask);
            let wc16 = if EFF_EXP {
                i8x16_narrow_i16x8(
                    i16x8_narrow_i32x4(w0, w1),
                    i16x8_narrow_i32x4(w2, w3),
                )
            } else {
                zero_i32
            };
            let out_off = out_base + $blk_local * 16;
            hysteresis_pan_16_packed_simd128(
                max_d16,
                Some((max_sp16, tol_span_soft, tol_span_reset)),
                wc16,
                tol_hard,
                tol_mid,
                tol_soft,
                one_u8,
                two_u8,
                six_u8,
                fourteen_u8,
                twelve_u8,
                twenty_u8,
                EFF_EXP,
                stat_streak_ptr.add(out_off),
                eval_mask_ptr.add(out_off),
            );
        }};
    }

    let fast_cnt = if is_tail_end { blk_cnt - 1 } else { blk_cnt };
    for blk_local in 0..fast_cnt {
        eval_block!(blk_local, false);
    }
    if is_tail_end {
        eval_block!(blk_cnt - 1, true);
    }
}

#[inline(always)]
unsafe fn subpx_eval_and_hyst_quads_2d_ctx<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
>(
    ctx: &SubpxRowCtx,
    rgb_row_out: *mut u8,
    row_c: usize,
    row_p0: usize,
    mut x: usize,
) -> usize {
    let quad_lim = ctx.quad_lim;
    if x + 4 > quad_lim {
        return x;
    }
    let w00_u8 = ctx.w00_u8;
    let w10_u8 = ctx.w10_u8;
    let w01_u8 = ctx.w01_u8;
    let w11_u8 = ctx.w11_u8;
    let zero_i32 = i32x4_splat(0);
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let rg_mask = if EFF_EXP { u32x4_splat(0x0000_FFFF) } else { zero_i32 };
    let ff_mask = u32x4_splat(0xFF);
    let b_mask = if EFF_EXP { u32x4_splat(0x00FF_0000) } else { zero_i32 };
    let b_250 = if EFF_EXP { u32x4_splat(250 << 16) } else { zero_i32 };
    let splat_230 = if EFF_EXP { u32x4_splat(230) } else { zero_i32 };
    let splat_251 = if EFF_EXP { u8x16_splat(251) } else { zero_i32 };
    let splat_253 = if EFF_EXP { u8x16_splat(253) } else { zero_i32 };
    let tol_hard = ctx.tol_hard;
    let tol_mid = ctx.tol_mid;
    let tol_soft = ctx.tol_soft;
    let tol_span_soft = ctx.tol_span_soft;
    let tol_span_reset = ctx.tol_span_reset;
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let fourteen_u8 = u8x16_splat(14);
    let twelve_u8 = u8x16_splat(12);
    let twenty_u8 = u8x16_splat(20);
    let round_v = u16x8_splat(128);
    let dg_q15_v = ctx.dg_q15_v;
    let off_i16_v = ctx.off_i16_v;
    let c_row_ptr = ctx.curr_rgba.add(row_c * 4);
    while x + 4 <= quad_lim {
        let sx = ((x as i32) + ctx.ix0) as usize;
        let q0_ptr = ctx.prev_rgba.add((row_p0 + sx) * 4);
        let q1_ptr = q0_ptr.add(ctx.row_bytes);
        let r0 = v128_load(q0_ptr as *const v128);
        let r1 = v128_load(q1_ptr as *const v128);
        let s0 = v128_load(q0_ptr.add(4) as *const v128);
        let s1 = v128_load(q1_ptr.add(4) as *const v128);
        let ca = v128_load(c_row_ptr.add(x * 4) as *const v128);
        let rgb_ptr = if WRITE_RGB {
            if RGB_U16 {
                rgb_row_out.add(x * 2)
            } else {
                rgb_row_out.add(x * 4)
            }
        } else {
            core::ptr::null_mut()
        };
        let (abs, sp, wc) = subpx_eval_quad_2d::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
            r0, s0, r1, s1, ca, rgb_ptr,
            w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230, splat_251, splat_253,
        );
        let max_d = pack4_max_u8_16_simd128(abs, zero_i32, zero_i32, zero_i32, ff_mask);
        let max_sp = pack4_max_u8_16_simd128(sp, zero_i32, zero_i32, zero_i32, ff_mask);
        let wc4 = if EFF_EXP {
            let n16 = i16x8_narrow_i32x4(wc, zero_i32);
            i8x16_narrow_i16x8(n16, zero_i32)
        } else {
            zero_i32
        };
        let mut hard4 = u8x16_le(max_d, tol_hard);
        let mut mid4 = u8x16_le(max_d, tol_mid);
        let mut soft4 = u8x16_le(max_d, tol_soft);
        if EFF_EXP {
            hard4 = v128_andnot(hard4, wc4);
        }
        let span_soft_ok = u8x16_le(max_sp, tol_span_soft);
        let span_reset_ok = u8x16_le(max_sp, tol_span_reset);
        hard4 = v128_and(v128_and(hard4, span_soft_ok), span_reset_ok);
        mid4 = v128_and(mid4, span_reset_ok);
        soft4 = v128_and(soft4, span_reset_ok);
        let out_off = row_c + x;
        let old_s = v128_load32_zero(ctx.stat_streak_ptr.add(out_off) as *const u32);
        let inc = u8x16_min(u8x16_add_sat(old_s, two_u8), twenty_u8);
        let mid_dec = u8x16_sub_sat(old_s, one_u8);
        let soft_dec = u8x16_sub_sat(u8x16_min(old_s, fourteen_u8), six_u8);
        let s_mid = v128_bitselect(inc, mid_dec, hard4);
        let s_soft = v128_bitselect(s_mid, soft_dec, mid4);
        let new_s = v128_and(s_soft, soft4);
        let eval_raw = u8x16_ge(new_s, twelve_u8);
        let eval4 = if EFF_EXP { v128_andnot(eval_raw, wc4) } else { eval_raw };
        v128_store32_lane::<0>(new_s, ctx.stat_streak_ptr.add(out_off) as *mut u32);
        v128_store32_lane::<0>(eval4, ctx.eval_mask_ptr.add(out_off) as *mut u32);
        x += 4;
    }
    x
}

#[inline(always)]
unsafe fn subpx_eval_and_hyst_quads_2d<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_row_out: *mut u8,
    row_c: usize,
    row_p0: usize,
    row_bytes: usize,
    ix0: i32,
    mut x: usize,
    quad_lim: usize,
    w00: u16,
    w10: u16,
    w01: u16,
    w11: u16,
    round_v: v128,
    dg_q15_v: v128,
    off_i16_v: v128,
    hard_u8: u8,
    mid_u8: u8,
    soft_u8: u8,
    span_soft_u8: u8,
    span_reset_u8: u8,
) -> usize {
    if x + 4 > quad_lim {
        return x;
    }
    let w00_u8 = u8x16_splat(w00 as u8);
    let w10_u8 = u8x16_splat(w10 as u8);
    let w01_u8 = u8x16_splat(w01 as u8);
    let w11_u8 = u8x16_splat(w11 as u8);
    let zero_i32 = i32x4_splat(0);
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let rg_mask = if EFF_EXP { u32x4_splat(0x0000_FFFF) } else { zero_i32 };
    let ff_mask = u32x4_splat(0xFF);
    let b_mask = if EFF_EXP { u32x4_splat(0x00FF_0000) } else { zero_i32 };
    let b_250 = if EFF_EXP { u32x4_splat(250 << 16) } else { zero_i32 };
    let splat_230 = if EFF_EXP { u32x4_splat(230) } else { zero_i32 };
    let splat_251 = if EFF_EXP { u8x16_splat(251) } else { zero_i32 };
    let splat_253 = if EFF_EXP { u8x16_splat(253) } else { zero_i32 };
    let tol_hard = u8x16_splat(hard_u8);
    let tol_mid = u8x16_splat(mid_u8);
    let tol_soft = u8x16_splat(soft_u8);
    let tol_span_soft = u8x16_splat(span_soft_u8);
    let tol_span_reset = u8x16_splat(span_reset_u8);
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let fourteen_u8 = u8x16_splat(14);
    let twelve_u8 = u8x16_splat(12);
    let twenty_u8 = u8x16_splat(20);
    let c_row_ptr = curr_rgba.add(row_c * 4);
    while x + 4 <= quad_lim {
        let sx = ((x as i32) + ix0) as usize;
        let q0_ptr = prev_rgba.add((row_p0 + sx) * 4);
        let q1_ptr = q0_ptr.add(row_bytes);
        let r0 = v128_load(q0_ptr as *const v128);
        let r1 = v128_load(q1_ptr as *const v128);
        let s0 = v128_load(q0_ptr.add(4) as *const v128);
        let s1 = v128_load(q1_ptr.add(4) as *const v128);
        let ca = v128_load(c_row_ptr.add(x * 4) as *const v128);
        let rgb_ptr = if WRITE_RGB {
            if RGB_U16 {
                rgb_row_out.add(x * 2)
            } else {
                rgb_row_out.add(x * 4)
            }
        } else {
            core::ptr::null_mut()
        };
        let (abs, sp, wc) = subpx_eval_quad_2d::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
            r0, s0, r1, s1, ca, rgb_ptr,
            w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230, splat_251, splat_253,
        );
        let max_d = pack4_max_u8_16_simd128(abs, zero_i32, zero_i32, zero_i32, ff_mask);
        let max_sp = pack4_max_u8_16_simd128(sp, zero_i32, zero_i32, zero_i32, ff_mask);
        let wc4 = if EFF_EXP {
            let n16 = i16x8_narrow_i32x4(wc, zero_i32);
            i8x16_narrow_i16x8(n16, zero_i32)
        } else {
            zero_i32
        };
        let mut hard4 = u8x16_le(max_d, tol_hard);
        let mut mid4 = u8x16_le(max_d, tol_mid);
        let mut soft4 = u8x16_le(max_d, tol_soft);
        if EFF_EXP {
            hard4 = v128_andnot(hard4, wc4);
        }
        let span_soft_ok = u8x16_le(max_sp, tol_span_soft);
        let span_reset_ok = u8x16_le(max_sp, tol_span_reset);
        hard4 = v128_and(v128_and(hard4, span_soft_ok), span_reset_ok);
        mid4 = v128_and(mid4, span_reset_ok);
        soft4 = v128_and(soft4, span_reset_ok);
        let out_off = row_c + x;
        let old_s = v128_load32_zero(stat_streak_ptr.add(out_off) as *const u32);
        let inc = u8x16_min(u8x16_add_sat(old_s, two_u8), twenty_u8);
        let mid_dec = u8x16_sub_sat(old_s, one_u8);
        let soft_dec = u8x16_sub_sat(u8x16_min(old_s, fourteen_u8), six_u8);
        let s_mid = v128_bitselect(inc, mid_dec, hard4);
        let s_soft = v128_bitselect(s_mid, soft_dec, mid4);
        let new_s = v128_and(s_soft, soft4);
        let eval_raw = u8x16_ge(new_s, twelve_u8);
        let eval4 = if EFF_EXP { v128_andnot(eval_raw, wc4) } else { eval_raw };
        v128_store32_lane::<0>(new_s, stat_streak_ptr.add(out_off) as *mut u32);
        v128_store32_lane::<0>(eval4, eval_mask_ptr.add(out_off) as *mut u32);
        x += 4;
    }
    x
}

#[inline(always)]
unsafe fn subpx_pan_streak_row_exact_inner<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
    const REUSE_H0: bool,
    const WRITE_H1: bool,
    const EXTMUL_MODE: u8,
>(
    ctx: &SubpxRowCtx,
    rgb_row_out: *mut u8,
    y: usize,
) {
    let row_c = y * ctx.frame_w;
    let sy = (y as i32) + ctx.iy0;
    let row_p0 = (sy as usize) * ctx.frame_w;
    if ctx.x_lo == 1 {
        *ctx.stat_streak_ptr.add(row_c) = 0;
        *ctx.eval_mask_ptr.add(row_c) = 0;
        if WRITE_RGB {
            if RGB_U16 {
                *(rgb_row_out as *mut u16) = 0;
            } else {
                *(rgb_row_out as *mut f32) = 0.0;
            }
        }
    } else if ctx.x_lo > 0 {
        core::slice::from_raw_parts_mut(ctx.stat_streak_ptr.add(row_c), ctx.x_lo).fill(0);
        core::slice::from_raw_parts_mut(ctx.eval_mask_ptr.add(row_c), ctx.x_lo).fill(0);
        if WRITE_RGB {
            if RGB_U16 {
                core::ptr::write_bytes(rgb_row_out, 0, ctx.x_lo * 2);
            } else {
                core::slice::from_raw_parts_mut(rgb_row_out as *mut f32, ctx.x_lo).fill(0.0);
            }
        }
    }
    if ctx.n_blocks16 > 0 {
        let p0_ptr = ctx.prev_rgba.add((row_p0 + ctx.sx_lo) * 4);
        let p1_ptr = p0_ptr.add(ctx.row_bytes);
        subpx_row_extmul_2d_inner::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16, REUSE_H0, WRITE_H1, EXTMUL_MODE>(
            ctx, p0_ptr, p1_ptr, rgb_row_out, row_c,
        );
    }
    let mut tail_x = ctx.x_simd_end;
    if tail_x >= ctx.x_lo && tail_x + 4 <= ctx.quad_lim {
        tail_x = subpx_eval_and_hyst_quads_2d_ctx::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
            ctx, rgb_row_out, row_c, row_p0, tail_x,
        );
    }
    if tail_x < ctx.frame_w {
        subpx_scalar_tail::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
            ctx.curr_rgba,
            ctx.prev_rgba,
            ctx.stat_streak_ptr,
            ctx.eval_mask_ptr,
            rgb_row_out,
            row_c,
            row_p0,
            ctx.row_bytes,
            ctx.frame_w,
            ctx.w_i,
            ctx.ix0,
            tail_x,
            ctx.frame_w,
            ctx.w00,
            ctx.w10,
            ctx.w01,
            ctx.w11,
            ctx.dg_q15,
            ctx.off_i16,
            ctx.hard_u8,
            ctx.mid_u8,
            ctx.soft_u8,
            ctx.span_soft_u8,
            ctx.span_reset_u8,
        );
    }
}

#[inline(never)]
unsafe fn subpx_pan_streak_row_exact<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
    const REUSE_H0: bool,
    const WRITE_H1: bool,
    const EXTMUL_MODE: u8,
>(
    ctx: &SubpxRowCtx,
    rgb_row_out: *mut u8,
    y: usize,
) {
    subpx_pan_streak_row_exact_inner::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16, REUSE_H0, WRITE_H1, EXTMUL_MODE>(
        ctx, rgb_row_out, y,
    );
}

#[inline(never)]
unsafe fn subpx_pan_streak_band_2d<const EXP_ACTIVE: bool, const EFF_EXP: bool>(
    ctx: &SubpxRowCtx,
    band_rgb_ptr: *mut u16,
    y0: usize,
    init_reuse_h0: bool,
) {
    let mut ry = 0usize;
    if !init_reuse_h0 {
        subpx_pan_streak_row_exact_inner::<EXP_ACTIVE, EFF_EXP, true, true, false, true, 0>(
            ctx,
            band_rgb_ptr as *mut u8,
            y0,
        );
        ry = 1;
    }
    while ry < 16 {
        subpx_pan_streak_row_exact_inner::<EXP_ACTIVE, EFF_EXP, true, true, true, true, 0>(
            ctx,
            band_rgb_ptr.add(ry * 640) as *mut u8,
            y0 + ry,
        );
        ry += 1;
    }
}

#[inline(never)]
unsafe fn subpx_pan_streak_row_ctx<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
    const RGB_U16: bool,
>(
    ctx: &SubpxRowCtx,
    rgb_row_out: *mut u8,
    y: usize,
    reuse_h0: bool,
) {
    let row_c = y * ctx.frame_w;
    let sy = (y as i32) + ctx.iy0;
    let row_p0 = (sy as usize) * ctx.frame_w;
    if ctx.x_lo > 0 {
        core::slice::from_raw_parts_mut(ctx.stat_streak_ptr.add(row_c), ctx.x_lo).fill(0);
        core::slice::from_raw_parts_mut(ctx.eval_mask_ptr.add(row_c), ctx.x_lo).fill(0);
        if WRITE_RGB {
            if RGB_U16 {
                core::ptr::write_bytes(rgb_row_out, 0, ctx.x_lo * 2);
            } else {
                core::slice::from_raw_parts_mut(rgb_row_out as *mut f32, ctx.x_lo).fill(0.0);
            }
        }
    }

    if ctx.n_blocks16 > 0 {
        let p0_ptr = ctx.prev_rgba.add((row_p0 + ctx.sx_lo) * 4);
        let p1_ptr = p0_ptr.add(ctx.row_bytes);
        if ctx.max_w <= 255 {
            let write_h1 = ctx.write_h1;
            let read_h0 = write_h1 && reuse_h0;
            macro_rules! call_extmul {
                ($rh0:expr, $wh1:expr, $mode:expr) => {
                    return subpx_pan_streak_row_exact::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16, $rh0, $wh1, $mode>(
                        ctx, rgb_row_out, y,
                    )
                };
            }
            match (read_h0, write_h1, ctx.extmul_mode) {
                (true, true, 1) => call_extmul!(true, true, 1),
                (true, true, 2) => call_extmul!(true, true, 2),
                (true, true, 3) => call_extmul!(true, true, 3),
                (true, true, 4) => call_extmul!(true, true, 4),
                (true, true, _) => call_extmul!(true, true, 0),
                (false, true, 1) => call_extmul!(false, true, 1),
                (false, true, 2) => call_extmul!(false, true, 2),
                (false, true, 3) => call_extmul!(false, true, 3),
                (false, true, 4) => call_extmul!(false, true, 4),
                (false, true, _) => call_extmul!(false, true, 0),
                (_, _, 1) => call_extmul!(false, false, 1),
                (_, _, 2) => call_extmul!(false, false, 2),
                (_, _, 3) => call_extmul!(false, false, 3),
                (_, _, 4) => call_extmul!(false, false, 4),
                _ => call_extmul!(false, false, 0),
            }
        } else {
            let n_q = ctx.n_blocks16 * 4;
            let write_h1 = ctx.write_h1;
            let read_h0 = write_h1 && reuse_h0;

            let mut blk_lo = 0usize;
            while blk_lo < ctx.n_blocks16 {
                let blk_cnt = (ctx.n_blocks16 - blk_lo).min(40);
                let is_tail_end = blk_lo + blk_cnt == ctx.n_blocks16;
                match (read_h0, write_h1) {
                    (true, true) => {
                        subpx_row_blk_chunk::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16, true, true>(
                            ctx.curr_rgba, p0_ptr, p1_ptr, ctx.stat_streak_ptr, ctx.eval_mask_ptr,
                            rgb_row_out, ctx.h_cache_ptr, row_c, ctx.x_lo, n_q, blk_lo, blk_cnt,
                            is_tail_end, ctx.wx0_v, ctx.wx1_v, ctx.wy0_v, ctx.wy1_v, ctx.round_v,
                            ctx.dg_q15_v, ctx.off_i16_v, ctx.hard_u8, ctx.mid_u8, ctx.soft_u8,
                            ctx.span_soft_u8, ctx.span_reset_u8,
                        );
                    }
                    (false, true) => {
                        subpx_row_blk_chunk::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16, false, true>(
                            ctx.curr_rgba, p0_ptr, p1_ptr, ctx.stat_streak_ptr, ctx.eval_mask_ptr,
                            rgb_row_out, ctx.h_cache_ptr, row_c, ctx.x_lo, n_q, blk_lo, blk_cnt,
                            is_tail_end, ctx.wx0_v, ctx.wx1_v, ctx.wy0_v, ctx.wy1_v, ctx.round_v,
                            ctx.dg_q15_v, ctx.off_i16_v, ctx.hard_u8, ctx.mid_u8, ctx.soft_u8,
                            ctx.span_soft_u8, ctx.span_reset_u8,
                        );
                    }
                    _ => {
                        subpx_row_blk_chunk::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16, false, false>(
                            ctx.curr_rgba, p0_ptr, p1_ptr, ctx.stat_streak_ptr, ctx.eval_mask_ptr,
                            rgb_row_out, ctx.h_cache_ptr, row_c, ctx.x_lo, n_q, blk_lo, blk_cnt,
                            is_tail_end, ctx.wx0_v, ctx.wx1_v, ctx.wy0_v, ctx.wy1_v, ctx.round_v,
                            ctx.dg_q15_v, ctx.off_i16_v, ctx.hard_u8, ctx.mid_u8, ctx.soft_u8,
                            ctx.span_soft_u8, ctx.span_reset_u8,
                        );
                    }
                }
                blk_lo += blk_cnt;
            }
        }
    }

    let mut tail_x = ctx.x_simd_end;
    if ctx.max_w <= 255 && tail_x >= ctx.x_lo && tail_x + 4 <= ctx.quad_lim {
        tail_x = subpx_eval_and_hyst_quads_2d::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
            ctx.curr_rgba, ctx.prev_rgba, ctx.stat_streak_ptr, ctx.eval_mask_ptr, rgb_row_out,
            row_c, row_p0, ctx.row_bytes, ctx.ix0, tail_x, ctx.quad_lim,
            ctx.w00, ctx.w10, ctx.w01, ctx.w11, ctx.round_v, ctx.dg_q15_v, ctx.off_i16_v,
            ctx.hard_u8, ctx.mid_u8, ctx.soft_u8, ctx.span_soft_u8, ctx.span_reset_u8,
        );
    }

    if tail_x < ctx.frame_w {
        subpx_scalar_tail::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, RGB_U16>(
            ctx.curr_rgba,
            ctx.prev_rgba,
            ctx.stat_streak_ptr,
            ctx.eval_mask_ptr,
            rgb_row_out,
            row_c,
            row_p0,
            ctx.row_bytes,
            ctx.frame_w,
            ctx.w_i,
            ctx.ix0,
            tail_x,
            ctx.frame_w,
            ctx.w00,
            ctx.w10,
            ctx.w01,
            ctx.w11,
            ctx.dg_q15,
            ctx.off_i16,
            ctx.hard_u8,
            ctx.mid_u8,
            ctx.soft_u8,
            ctx.span_soft_u8,
            ctx.span_reset_u8,
        );
    }
}

#[inline(never)]
unsafe fn update_stat_streak_pan_subpx_range_impl<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_sum_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    row_lo: usize,
    row_hi: usize,
    pan_dx_f: f32,
    pan_dy_f: f32,
    dg_q15: i16,
    off_i16: i16,
    scale_px: usize,
) {
    if row_hi <= row_lo {
        return;
    }
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;

    let (ix0, iy0, wx0, wx1, wy0, wy1) = bilinear_weights_1d_u16(pan_dx_f, pan_dy_f);
    let fast_pan = pan_dx_f.abs() + pan_dy_f.abs() >= 3.2 * (scale_px as f32);
    let hard_u8 = 4u8;
    let mid_u8 = 5u8;
    let soft_u8 = 8u8;
    let span_reset_u8 = if fast_pan { 68u8 } else { 35u8 };
    let span_soft_u8 = if fast_pan { 50u8 } else { 26u8 };
    let row_bytes = frame_w * 4;

    let rlo_i = row_lo as i32;
    let rhi_i = (row_hi.min(frame_h)) as i32;
    let y_start = ((-iy0).clamp(rlo_i, rhi_i)) as usize;
    let y_end = (((h_i - 1 - iy0).clamp(rlo_i, rhi_i)) as usize).max(y_start);
    subpx_zero_rows::<WRITE_RGB>(stat_streak_ptr, eval_mask_ptr, rgb_sum_out, row_lo, y_start, frame_w);

    let x_lo = ((-ix0).clamp(0, w_i)) as usize;
    let x_hi_lim = (w_i - 16).min(w_i - 17 - ix0);
    let n_blocks16 = if (x_lo as i32) <= x_hi_lim {
        (((x_hi_lim - (x_lo as i32)) >> 4) + 1) as usize
    } else {
        0
    };
    let x_simd_end = x_lo + n_blocks16 * 16;

    let subpx_ctx = SubpxRowCtx::new::<EFF_EXP>(
        curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, core::ptr::null_mut(),
        frame_w, w_i, row_bytes, ix0, iy0, x_lo, n_blocks16, x_simd_end,
        wx0, wx1, wy0, wy1, dg_q15, off_i16, hard_u8, mid_u8, soft_u8, span_soft_u8, span_reset_u8,
        false,
    );

    if subpx_ctx.max_w <= 255 {
        macro_rules! run_rows_mode {
            ($mode:expr) => {
                for y in y_start..y_end {
                    let rgb_row = if WRITE_RGB {
                        rgb_sum_out.add(y * frame_w) as *mut u8
                    } else {
                        core::ptr::null_mut()
                    };
                    subpx_pan_streak_row_exact::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, false, false, false, $mode>(
                        &subpx_ctx, rgb_row, y,
                    );
                }
            };
        }
        match subpx_ctx.extmul_mode {
            1 => run_rows_mode!(1),
            2 => run_rows_mode!(2),
            3 => run_rows_mode!(3),
            4 => run_rows_mode!(4),
            _ => run_rows_mode!(0),
        }
    } else {
        for y in y_start..y_end {
            let rgb_row = if WRITE_RGB {
                rgb_sum_out.add(y * frame_w) as *mut u8
            } else {
                core::ptr::null_mut()
            };
            subpx_pan_streak_row_ctx::<EXP_ACTIVE, EFF_EXP, WRITE_RGB, false>(
                &subpx_ctx,
                rgb_row,
                y,
                false,
            );
        }
    }

    subpx_zero_rows::<WRITE_RGB>(stat_streak_ptr, eval_mask_ptr, rgb_sum_out, y_end, row_hi, frame_w);
}

#[inline(never)]
unsafe fn update_stat_streak_pan_subpx_impl<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_sum_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    pan_dx_f: f32,
    pan_dy_f: f32,
    dg_q15: i16,
    off_i16: i16,
    scale_px: usize,
) {
    update_stat_streak_pan_subpx_range_impl::<EXP_ACTIVE, EFF_EXP, WRITE_RGB>(
        curr_rgba,
        prev_rgba,
        stat_streak_ptr,
        eval_mask_ptr,
        rgb_sum_out,
        frame_w,
        frame_h,
        0,
        frame_h,
        pan_dx_f,
        pan_dy_f,
        dg_q15,
        off_i16,
        scale_px,
    );
}

#[inline(never)]
unsafe fn update_stat_streak_pan_int_impl<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const WRITE_RGB: bool,
>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_sum_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    pan_dx: i32,
    pan_dy: i32,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
) {
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let rg_mask = u32x4_splat(0x0000_FFFF);
    let ff_mask = u32x4_splat(0xFF);
    let b_mask = u32x4_splat(0x00FF_0000);
    let b_250 = u32x4_splat(250 << 16);
    let splat_230 = u32x4_splat(230);
    let splat_251 = u8x16_splat(251);
    let splat_253 = u8x16_splat(253);

    let zero_i32 = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let fourteen_u8 = u8x16_splat(14);
    let twelve_u8 = u8x16_splat(12);
    let twenty_u8 = u8x16_splat(20);

    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let dg_q15_v = i16x8_splat(dg_q15);
    let off_i16_v = i16x8_splat(off_i16);

    let inc_scalar = 2u8;
    let mid_u8 = hard_u8 + 1;
    let soft_u8 = hard_u8 + 4;
    let tol_hard = u8x16_splat(hard_u8);
    let tol_mid = u8x16_splat(mid_u8);
    let tol_soft = u8x16_splat(soft_u8);

    for y in 0..frame_h {
        let sy = (y as i32) - pan_dy;
        let row_c = y * frame_w;
        if sy < 0 || sy >= h_i {
            core::slice::from_raw_parts_mut(stat_streak_ptr.add(row_c), frame_w).fill(0);
            core::slice::from_raw_parts_mut(eval_mask_ptr.add(row_c), frame_w).fill(0);
            if WRITE_RGB {
                core::slice::from_raw_parts_mut(rgb_sum_out.add(row_c), frame_w).fill(0.0);
            }
            continue;
        }
        let row_p = (sy as usize) * frame_w;
        let x_lo = (pan_dx.clamp(0, w_i)) as usize;
        let x_hi_lim = (w_i - 16).min(w_i - 16 + pan_dx);
        if x_lo > 0 {
            core::slice::from_raw_parts_mut(stat_streak_ptr.add(row_c), x_lo).fill(0);
            core::slice::from_raw_parts_mut(eval_mask_ptr.add(row_c), x_lo).fill(0);
            if WRITE_RGB {
                core::slice::from_raw_parts_mut(rgb_sum_out.add(row_c), x_lo).fill(0.0);
            }
        }
        let mut x = x_lo;
        while (x as i32) <= x_hi_lim {
            let sx = (x as i32) - pan_dx;
            let c_off = (row_c + x) * 4;
            let p_off = (row_p + (sx as usize)) * 4;
            let c0 = v128_load(curr_rgba.add(c_off) as *const v128);
            let p0_raw = v128_load(prev_rgba.add(p_off) as *const v128);
            let c1 = v128_load(curr_rgba.add(c_off + 16) as *const v128);
            let p1_raw = v128_load(prev_rgba.add(p_off + 16) as *const v128);
            let c2 = v128_load(curr_rgba.add(c_off + 32) as *const v128);
            let p2_raw = v128_load(prev_rgba.add(p_off + 32) as *const v128);
            let c3 = v128_load(curr_rgba.add(c_off + 48) as *const v128);
            let p3_raw = v128_load(prev_rgba.add(p_off + 48) as *const v128);
            let (p0, p1, p2, p3) = if EXP_ACTIVE {
                (
                    apply_exposure_rgba4_simd128(p0_raw, dg_q15_v, off_i16_v),
                    apply_exposure_rgba4_simd128(p1_raw, dg_q15_v, off_i16_v),
                    apply_exposure_rgba4_simd128(p2_raw, dg_q15_v, off_i16_v),
                    apply_exposure_rgba4_simd128(p3_raw, dg_q15_v, off_i16_v),
                )
            } else {
                (p0_raw, p1_raw, p2_raw, p3_raw)
            };
            let mut abs0 = v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask);
            let mut abs1 = v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask);
            let mut abs2 = v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask);
            let mut abs3 = v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask);
            if WRITE_RGB {
                v128_store(rgb_sum_out.add(row_c + x) as *mut v128, abs_rgb_to_sum_f32x4_simd128(abs0));
                v128_store(rgb_sum_out.add(row_c + x + 4) as *mut v128, abs_rgb_to_sum_f32x4_simd128(abs1));
                v128_store(rgb_sum_out.add(row_c + x + 8) as *mut v128, abs_rgb_to_sum_f32x4_simd128(abs2));
                v128_store(rgb_sum_out.add(row_c + x + 12) as *mut v128, abs_rgb_to_sum_f32x4_simd128(abs3));
            }
            let wc16 = if EFF_EXP {
                abs0 = drop_b_royal_blue_simd128(abs0, c0, p0_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
                abs1 = drop_b_royal_blue_simd128(abs1, c1, p1_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
                abs2 = drop_b_royal_blue_simd128(abs2, c2, p2_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
                abs3 = drop_b_royal_blue_simd128(abs3, c3, p3_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
                let wc0 = white_clip_mask_u32_simd128(c0, p0_raw, rg_mask, splat_251, splat_253);
                let wc1 = white_clip_mask_u32_simd128(c1, p1_raw, rg_mask, splat_251, splat_253);
                let wc2 = white_clip_mask_u32_simd128(c2, p2_raw, rg_mask, splat_251, splat_253);
                let wc3 = white_clip_mask_u32_simd128(c3, p3_raw, rg_mask, splat_251, splat_253);
                i8x16_narrow_i16x8(i16x8_narrow_i32x4(wc0, wc1), i16x8_narrow_i32x4(wc2, wc3))
            } else {
                zero_i32
            };

            hysteresis_pan_16_simd128(
                abs0,
                abs1,
                abs2,
                abs3,
                None,
                wc16,
                tol_hard,
                tol_mid,
                tol_soft,
                ff_mask,
                one_u8,
                two_u8,
                six_u8,
                fourteen_u8,
                twelve_u8,
                twenty_u8,
                EFF_EXP,
                stat_streak_ptr.add(row_c + x),
                eval_mask_ptr.add(row_c + x),
            );
            x += 16;
        }
        while x < frame_w {
            let sx = (x as i32) - pan_dx;
            let idx = row_c + x;
            if sx >= 0 && sx < w_i {
                let c_b = idx * 4;
                let p_b = (row_p + (sx as usize)) * 4;
                let pr_r = *prev_rgba.add(p_b);
                let pr_g = *prev_rgba.add(p_b + 1);
                let white_clip = EFF_EXP
                    && ((pr_r >= 251 && pr_g >= 251)
                        || (*curr_rgba.add(c_b) >= 253 && *curr_rgba.add(c_b + 1) >= 253));
                let mut max_d = 0u8;
                let mut diff_sum = 0.0f32;
                for ch in 0..3usize {
                    let pr_raw = *prev_rgba.add(p_b + ch);
                    let c_ch = *curr_rgba.add(c_b + ch);
                    let pv = if EXP_ACTIVE {
                        apply_exposure_u8_scalar(pr_raw, dg_q15, off_i16)
                    } else {
                        pr_raw
                    };
                    let d = c_ch.abs_diff(pv);
                    diff_sum += d as f32;
                    if ch == 2 && EFF_EXP && (pr_raw >= 250 || c_ch >= 250) && pr_r.min(pr_g) < 230 {
                        continue;
                    }
                    if d > max_d {
                        max_d = d;
                    }
                }
                if WRITE_RGB {
                    *rgb_sum_out.add(idx) = diff_sum;
                }
                if white_clip && max_d <= hard_u8 {
                    max_d = mid_u8;
                }
                hysteresis_pan_scalar(
                    stat_streak_ptr.add(idx),
                    eval_mask_ptr.add(idx),
                    max_d,
                    inc_scalar,
                    hard_u8,
                    mid_u8,
                    soft_u8,
                    white_clip,
                );
            } else {
                *stat_streak_ptr.add(idx) = 0;
                *eval_mask_ptr.add(idx) = 0;
                if WRITE_RGB {
                    *rgb_sum_out.add(idx) = 0.0;
                }
            }
            x += 1;
        }
    }
}

/// Pan- and exposure-compensated stationarity streak & evaluation mask: fast-paths to `update_stat_streak_and_mask_simd128`
/// only when both camera pan and global exposure regime are inactive.
/// When the pan/exposure path executes and `rgb_sum_out` is non-null, writes unmasked 3-channel RGB L1
/// differences (`|c - p|_RGB` in `0.0..=765.0`) into `rgb_sum_out` and returns `true`.
#[inline(never)]
pub unsafe fn update_stat_streak_and_mask_pan_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_sum_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    pan_dx: i32,
    pan_dy: i32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    frac_mag: f32,
    exp_gain: f32,
    exp_off: f32,
    exp_regime: bool,
    rms_y: f32,
    scale_px: usize,
) -> bool {
    let (exp_active, dg_q15, off_i16) = exposure_q15_params(exp_gain, exp_off);
    let pan_moved = pan_dx != 0 || pan_dy != 0 || frac_mag >= 0.04;
    if !pan_moved && !exp_active && !exp_regime {
        update_stat_streak_and_mask_simd128(
            curr_rgba,
            prev_rgba,
            stat_streak_ptr,
            eval_mask_ptr,
            frame_w * frame_h,
        );
        return false;
    }

    let write_rgb_sum = !rgb_sum_out.is_null();
    let eff_exp_regime = exp_regime || exp_active;
    if frac_mag >= 0.04 {
        match (exp_active, eff_exp_regime, write_rgb_sum) {
            (false, false, true) => update_stat_streak_pan_subpx_impl::<false, false, true>(
                curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
                frame_w, frame_h, pan_dx_f, pan_dy_f, dg_q15, off_i16, scale_px,
            ),
            (false, true, true) => update_stat_streak_pan_subpx_impl::<false, true, true>(
                curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
                frame_w, frame_h, pan_dx_f, pan_dy_f, dg_q15, off_i16, scale_px,
            ),
            (true, _, true) => update_stat_streak_pan_subpx_impl::<true, true, true>(
                curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
                frame_w, frame_h, pan_dx_f, pan_dy_f, dg_q15, off_i16, scale_px,
            ),
            (false, false, false) => update_stat_streak_pan_subpx_impl::<false, false, false>(
                curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
                frame_w, frame_h, pan_dx_f, pan_dy_f, dg_q15, off_i16, scale_px,
            ),
            (false, true, false) => update_stat_streak_pan_subpx_impl::<false, true, false>(
                curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
                frame_w, frame_h, pan_dx_f, pan_dy_f, dg_q15, off_i16, scale_px,
            ),
            (true, _, false) => update_stat_streak_pan_subpx_impl::<true, true, false>(
                curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
                frame_w, frame_h, pan_dx_f, pan_dy_f, dg_q15, off_i16, scale_px,
            ),
        }
        return write_rgb_sum;
    }

    let hard_u8 = if pan_moved || rms_y >= 0.52 { 4u8 } else { 3u8 };
    match (exp_active, eff_exp_regime, write_rgb_sum) {
        (false, false, true) => update_stat_streak_pan_int_impl::<false, false, true>(
            curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
            frame_w, frame_h, pan_dx, pan_dy, dg_q15, off_i16, hard_u8,
        ),
        (false, true, true) => update_stat_streak_pan_int_impl::<false, true, true>(
            curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
            frame_w, frame_h, pan_dx, pan_dy, dg_q15, off_i16, hard_u8,
        ),
        (true, _, true) => update_stat_streak_pan_int_impl::<true, true, true>(
            curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
            frame_w, frame_h, pan_dx, pan_dy, dg_q15, off_i16, hard_u8,
        ),
        (false, false, false) => update_stat_streak_pan_int_impl::<false, false, false>(
            curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
            frame_w, frame_h, pan_dx, pan_dy, dg_q15, off_i16, hard_u8,
        ),
        (false, true, false) => update_stat_streak_pan_int_impl::<false, true, false>(
            curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
            frame_w, frame_h, pan_dx, pan_dy, dg_q15, off_i16, hard_u8,
        ),
        (true, _, false) => update_stat_streak_pan_int_impl::<true, true, false>(
            curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_sum_out,
            frame_w, frame_h, pan_dx, pan_dy, dg_q15, off_i16, hard_u8,
        ),
    }
    write_rgb_sum
}

/// Fast SIMD128 16x16 block ECA + full-RGB temporal frame-difference energy grid
/// (`20x11` in 320x180 SD, `40x22` in 640x360 HD) and stationary background healing across all scanlines.
#[inline(never)]
pub unsafe fn compute_motion_grid_and_heal_bg_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg0_ptr: *mut f32,
    eca_res_ptr: *const f32,
    grid_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
) {
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let heal_gate_rgb = f32x4_splat(0.0110 * 765.0);
    let w_eca = f32x4_splat(0.75);
    let w_rgb = f32x4_splat(0.25 / 765.0);
    let heal_alpha = f32x4_splat(0.65);

    for gy in 0..grid_h {
        let base_y = gy * 16;
        let mut gx_base = 0usize;
        while gx_base < grid_w {
            let gx_chunk = (grid_w - gx_base).min(40);
            let mut row_acc = [f32x4_splat(0.0); 40];
            for ry in 0..16usize {
                let y_row = (base_y + ry) * frame_w;
                for gx_l in 0..gx_chunk {
                    let row_off = y_row + (gx_base + gx_l) * 16;
                    let mut acc = row_acc[gx_l];
                    for cx_vec in 0..4usize {
                        let p = row_off + cx_vec * 4;
                        let c_px = v128_load(curr_rgba.add(p * 4) as *const v128);
                        let p_px = v128_load(prev_rgba.add(p * 4) as *const v128);

                        // Full 3-channel RGB temporal delta (prevents blue-only blind spot)
                        let rgb_sum = rgb_abs_sum_f32x4_simd128(c_px, p_px, rgb_mask);

                        let eca = v128_load(eca_res_ptr.add(p) as *const v128);
                        let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);

                        let y_cur = v128_load(luma_ptr.add(p) as *const v128);
                        let bg0 = v128_load(bg0_ptr.add(p) as *const v128);
                        let bg_healed = f32x4_add(bg0, f32x4_mul(heal_alpha, f32x4_sub(y_cur, bg0)));
                        v128_store(
                            bg0_ptr.add(p) as *mut v128,
                            v128_bitselect(bg_healed, bg0, is_stat),
                        );

                        let gated_eca = v128_andnot(eca, is_stat);
                        let combined = f32x4_add(
                            f32x4_mul(gated_eca, w_eca),
                            f32x4_mul(rgb_sum, w_rgb),
                        );
                        acc = f32x4_add(acc, combined);
                    }
                    row_acc[gx_l] = acc;
                }
            }
            let grid_row = gy * grid_w + gx_base;
            for gx_l in 0..gx_chunk {
                *grid_out.add(grid_row + gx_l) = hsum_f32x4(row_acc[gx_l]);
            }
            gx_base += gx_chunk;
        }
    }

    // Heal stationary background pixels on remaining bottom scanlines (`grid_h * 16 .. frame_h`)
    for py in (grid_h * 16)..frame_h {
        let row_off = py * frame_w;
        let mut px = 0usize;
        while px + 4 <= frame_w {
            let p = row_off + px;
            let c_px = v128_load(curr_rgba.add(p * 4) as *const v128);
            let p_px = v128_load(prev_rgba.add(p * 4) as *const v128);
            let rgb_sum = rgb_abs_sum_f32x4_simd128(c_px, p_px, rgb_mask);
            let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
            let y_cur = v128_load(luma_ptr.add(p) as *const v128);
            let bg0 = v128_load(bg0_ptr.add(p) as *const v128);
            let bg_healed = f32x4_add(bg0, f32x4_mul(heal_alpha, f32x4_sub(y_cur, bg0)));
            v128_store(
                bg0_ptr.add(p) as *mut v128,
                v128_bitselect(bg_healed, bg0, is_stat),
            );
            px += 4;
        }
    }
}

#[inline(always)]
unsafe fn flush_pyr_eca_reg_simd128(
    gy: usize,
    gx: usize,
    ry: usize,
    scale_px: usize,
    q4: v128,
    q16: v128,
    r0: v128,
    r1: v128,
    r2: v128,
    r3: v128,
    pyr_a: &mut v128,
    pyr_b: &mut v128,
    pyr_eca_ptr: *mut f32,
) {
    let s01 = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(r0, r1),
        i32x4_shuffle::<1, 3, 5, 7>(r0, r1),
    );
    let s23 = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(r2, r3),
        i32x4_shuffle::<1, 3, 5, 7>(r2, r3),
    );
    if scale_px <= 1 {
        if (ry & 1) == 0 {
            *pyr_a = s01;
            *pyr_b = s23;
        } else {
            let v0 = f32x4_mul(f32x4_add(*pyr_a, s01), q4);
            let v1 = f32x4_mul(f32x4_add(*pyr_b, s23), q4);
            let dst = pyr_eca_ptr.add((gy * 8 + (ry >> 1)) * 160 + gx * 8);
            v128_store(dst as *mut v128, v0);
            v128_store(dst.add(4) as *mut v128, v1);
        }
    } else {
        let h4 = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(s01, s23),
            i32x4_shuffle::<1, 3, 5, 7>(s01, s23),
        );
        let m = ry & 3;
        if m == 0 {
            *pyr_a = h4;
        } else if m < 3 {
            *pyr_a = f32x4_add(*pyr_a, h4);
        } else {
            let out = f32x4_mul(f32x4_add(*pyr_a, h4), q16);
            v128_store(
                pyr_eca_ptr.add((gy * 4 + (ry >> 2)) * 160 + gx * 4) as *mut v128,
                out,
            );
        }
    }
}

#[inline(always)]
unsafe fn flush_pyr_scale2_phase_simd128<const PHASE: u8>(
    gy: usize,
    gx: usize,
    ry_blk: usize,
    q16: v128,
    s01: v128,
    s23: v128,
    pyr_a: &mut v128,
    pyr_eca_ptr: *mut f32,
) {
    let h4 = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s01, s23),
        i32x4_shuffle::<1, 3, 5, 7>(s01, s23),
    );
    match PHASE {
        0 => *pyr_a = h4,
        1 | 2 => *pyr_a = f32x4_add(*pyr_a, h4),
        _ => {
            let out = f32x4_mul(f32x4_add(*pyr_a, h4), q16);
            v128_store(
                pyr_eca_ptr.add((gy * 4 + ry_blk) * 160 + gx * 4) as *mut v128,
                out,
            );
        }
    }
}

#[inline(always)]
unsafe fn flush_pyr_scale2_ri_simd128(
    gy: usize,
    gx: usize,
    ry_blk: usize,
    ri: usize,
    q16: v128,
    s01: v128,
    s23: v128,
    pyr_a: &mut v128,
    pyr_eca_ptr: *mut f32,
) {
    let h4 = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s01, s23),
        i32x4_shuffle::<1, 3, 5, 7>(s01, s23),
    );
    if ri == 0 {
        *pyr_a = h4;
    } else if ri < 3 {
        *pyr_a = f32x4_add(*pyr_a, h4);
    } else {
        let out = f32x4_mul(f32x4_add(*pyr_a, h4), q16);
        v128_store(
            pyr_eca_ptr.add((gy * 4 + ry_blk) * 160 + gx * 4) as *mut v128,
            out,
        );
    }
}

#[inline(always)]
pub fn is_pan_or_exp_active(
    pan_dx: i32,
    pan_dy: i32,
    frac_mag: f32,
    exp_gain: f32,
    exp_off: f32,
    exp_regime: bool,
) -> bool {
    let (exp_active, _, _) = exposure_q15_params(exp_gain, exp_off);
    let pan_moved = pan_dx != 0 || pan_dy != 0 || frac_mag >= 0.04;
    pan_moved || exp_active || exp_regime
}

#[inline(always)]
unsafe fn stat_fused_chunk4<const HAS_MASK: bool, const UPDATE_STREAK: bool>(
    p: usize,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *const u8,
    eca_res_ptr: *mut f32,
    pre_abs: v128,
    pre_mv: v128,
    v_alpha: v128,
    v_c: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    rgb_mask: v128,
    heal_gate_rgb: v128,
    w_eca: v128,
    w_rgb: v128,
    heal_alpha: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
    tile_acc: &mut v128,
) -> v128 {
    let y_cur = v128_load(luma_ptr.add(p) as *const v128);
    let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);

    let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
    let diff = f32x4_sub(y_cur, proj);
    let res = f32x4_abs(diff);
    v128_store(eca_res_ptr.add(p) as *mut v128, res);

    let rate = if HAS_MASK {
        let mv = if UPDATE_STREAK {
            pre_mv
        } else {
            let raw_m4 = v128_load32_zero(eval_mask_ptr.add(p) as *const u32);
            i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(raw_m4))
        };
        let m_diff = v128_and(diff, mv);
        let m_sig = v128_and(y_cur, mv);
        *acc_eval_err2 = f32x4_add(*acc_eval_err2, f32x4_mul(m_diff, m_diff));
        *acc_eval_sig2 = f32x4_add(*acc_eval_sig2, f32x4_mul(m_sig, m_sig));
        *acc_eval_cnt_i = i32x4_sub(*acc_eval_cnt_i, mv);

        let is_bg = f32x4_lt(res, v_motion_gate);
        let is_eval_heal = v128_and(mv, f32x4_lt(res, v_eval_gate));
        v128_bitselect(v_rate_eval, v128_bitselect(v_rate_bg, v_rate_fg, is_bg), is_eval_heal)
    } else {
        let is_bg = f32x4_lt(res, v_motion_gate);
        v128_bitselect(v_rate_bg, v_rate_fg, is_bg)
    };
    let b_eca = f32x4_add(b_old, f32x4_mul(rate, f32x4_sub(y_cur, b_old)));

    let rgb_sum = if UPDATE_STREAK {
        abs_rgb_to_sum_f32x4_simd128(pre_abs)
    } else {
        let c_px = v128_load(curr_rgba.add(p * 4) as *const v128);
        let p_px = v128_load(prev_rgba.add(p * 4) as *const v128);
        rgb_abs_sum_f32x4_simd128(c_px, p_px, rgb_mask)
    };
    let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
    let bg_healed = f32x4_add(b_eca, f32x4_mul(heal_alpha, f32x4_sub(y_cur, b_eca)));
    v128_store(
        bg_luma_ptr.add(p) as *mut v128,
        v128_bitselect(bg_healed, b_eca, is_stat),
    );

    let gated_eca = v128_andnot(res, is_stat);
    let combined = f32x4_add(f32x4_mul(gated_eca, w_eca), f32x4_mul(rgb_sum, w_rgb));
    *tile_acc = f32x4_add(*tile_acc, combined);
    res
}

#[inline(always)]
unsafe fn stat_fused_grid_le40_row<
    const HAS_MASK: bool,
    const UPDATE_STREAK: bool,
    const FUSE_SCALE2: bool,
    const PYR_PHASE: u8,
>(
    gy: usize,
    ry: usize,
    ry_blk: usize,
    row_base: usize,
    grid_w: usize,
    scale_px: usize,
    fuse_pyr: bool,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eval_mask_const: *const u8,
    stat_streak_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    write_eval_mask: bool,
    v_alpha: v128,
    v_c: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    rgb_mask: v128,
    heal_gate_rgb: v128,
    w_eca: v128,
    w_rgb: v128,
    heal_alpha: v128,
    tol3: v128,
    zero_i32: v128,
    one_u8: v128,
    min_streak: v128,
    zero_v: v128,
    q4: v128,
    q16: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
    acc_row: &mut [v128; 40],
    pyr_a_row: &mut [v128; 40],
    pyr_b_row: &mut [v128; 20],
) {
    for gx in 0..grid_w {
        let p_base = row_base + gx * 16;
        let (a0, a1, a2, a3, mv0, mv1, mv2, mv3) = if UPDATE_STREAK {
            let cb = p_base * 4;
            let c0 = v128_load(curr_rgba.add(cb) as *const v128);
            let p0 = v128_load(prev_rgba.add(cb) as *const v128);
            let c1 = v128_load(curr_rgba.add(cb + 16) as *const v128);
            let p1 = v128_load(prev_rgba.add(cb + 16) as *const v128);
            let c2 = v128_load(curr_rgba.add(cb + 32) as *const v128);
            let p2 = v128_load(prev_rgba.add(cb + 32) as *const v128);
            let c3 = v128_load(curr_rgba.add(cb + 48) as *const v128);
            let p3 = v128_load(prev_rgba.add(cb + 48) as *const v128);

            let d0 = v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask);
            let d1 = v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask);
            let d2 = v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask);
            let d3 = v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask);

            let within16 = pack4_le_tol16_simd128(d0, d1, d2, d3, tol3, zero_i32);
            let old_streak = v128_load(stat_streak_ptr.add(p_base) as *const v128);
            let new_streak = v128_and(u8x16_add_sat(old_streak, one_u8), within16);
            v128_store(stat_streak_ptr.add(p_base) as *mut v128, new_streak);

            let eval16 = u8x16_ge(new_streak, min_streak);
            if write_eval_mask {
                v128_store(eval_mask_ptr.add(p_base) as *mut v128, eval16);
            }
            if HAS_MASK {
                let lo16 = i16x8_extend_low_i8x16(eval16);
                let hi16 = i16x8_extend_high_i8x16(eval16);
                (
                    d0,
                    d1,
                    d2,
                    d3,
                    i32x4_extend_low_i16x8(lo16),
                    i32x4_extend_high_i16x8(lo16),
                    i32x4_extend_low_i16x8(hi16),
                    i32x4_extend_high_i16x8(hi16),
                )
            } else {
                (d0, d1, d2, d3, zero_i32, zero_i32, zero_i32, zero_i32)
            }
        } else {
            (zero_v, zero_v, zero_v, zero_v, zero_v, zero_v, zero_v, zero_v)
        };

        let mut acc = acc_row[gx];
        let r0 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
            p_base, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_const, eca_res_ptr,
            a0, mv0, v_alpha, v_c, v_rate_bg, v_rate_fg, v_motion_gate, v_rate_eval, v_eval_gate,
            rgb_mask, heal_gate_rgb, w_eca, w_rgb, heal_alpha,
            acc_eval_err2, acc_eval_sig2, acc_eval_cnt_i, &mut acc,
        );
        let r1 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
            p_base + 4, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_const, eca_res_ptr,
            a1, mv1, v_alpha, v_c, v_rate_bg, v_rate_fg, v_motion_gate, v_rate_eval, v_eval_gate,
            rgb_mask, heal_gate_rgb, w_eca, w_rgb, heal_alpha,
            acc_eval_err2, acc_eval_sig2, acc_eval_cnt_i, &mut acc,
        );
        let r2 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
            p_base + 8, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_const, eca_res_ptr,
            a2, mv2, v_alpha, v_c, v_rate_bg, v_rate_fg, v_motion_gate, v_rate_eval, v_eval_gate,
            rgb_mask, heal_gate_rgb, w_eca, w_rgb, heal_alpha,
            acc_eval_err2, acc_eval_sig2, acc_eval_cnt_i, &mut acc,
        );
        let r3 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
            p_base + 12, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_const, eca_res_ptr,
            a3, mv3, v_alpha, v_c, v_rate_bg, v_rate_fg, v_motion_gate, v_rate_eval, v_eval_gate,
            rgb_mask, heal_gate_rgb, w_eca, w_rgb, heal_alpha,
            acc_eval_err2, acc_eval_sig2, acc_eval_cnt_i, &mut acc,
        );
        acc_row[gx] = acc;
        if FUSE_SCALE2 {
            let s01 = f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(r0, r1),
                i32x4_shuffle::<1, 3, 5, 7>(r0, r1),
            );
            let s23 = f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(r2, r3),
                i32x4_shuffle::<1, 3, 5, 7>(r2, r3),
            );
            flush_pyr_scale2_phase_simd128::<PYR_PHASE>(
                gy,
                gx,
                ry_blk,
                q16,
                s01,
                s23,
                &mut pyr_a_row[gx],
                pyr_eca_ptr,
            );
        } else if fuse_pyr {
            let mut dummy_b = zero_v;
            let pb = if scale_px <= 1 && gx < 20 {
                &mut pyr_b_row[gx]
            } else {
                &mut dummy_b
            };
            flush_pyr_eca_reg_simd128(
                gy,
                gx,
                ry,
                scale_px,
                q4,
                q16,
                r0,
                r1,
                r2,
                r3,
                &mut pyr_a_row[gx],
                pb,
                pyr_eca_ptr,
            );
        }
    }
}

#[inline(always)]
unsafe fn abs_rgb_to_sum_u32x4_simd128(abs_rgb: v128) -> v128 {
    u32x4_extadd_pairwise_u16x8(u16x8_extadd_pairwise_u8x16(abs_rgb))
}

#[inline(always)]
unsafe fn stat_fused_hd40_row<
    const HAS_MASK: bool,
    const UPDATE_STREAK: bool,
    const FIRST_GY: bool,
    const PYR_STAGE: u8,
    const LAST_GY: bool,
>(
    row_p0: usize,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eval_mask_const: *const u8,
    stat_streak_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    acc_eca_ptr: *mut v128,
    acc_rgb_ptr: *mut v128,
    pyr_a_ptr: *mut v128,
    pyr_out_row: *mut f32,
    grid_row: *mut f32,
    v_alpha: v128,
    v_c: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    v_rate_bg_heal: v128,
    v_rate_fg_heal: v128,
    v_rate_eval_heal: v128,
    w_eca: v128,
    w_rgb: v128,
    q16: v128,
    out_err2: &mut v128,
    out_sig2: &mut v128,
    out_cnt_i: &mut v128,
) {
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let tol3 = u8x16_splat(3);
    let zero_i32 = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let min_streak = u8x16_splat(12);
    let heal_gate_i32 = i32x4_splat(9);

    let mut acc_err2 = *out_err2;
    let mut acc_sig2 = *out_sig2;
    let mut row_cnt_u8 = zero_i32;

    let mut p_base = row_p0;
    for gx in 0..40usize {
        let cb = p_base * 4;
        let d0 = {
            let c0 = v128_load(curr_rgba.add(cb) as *const v128);
            let p0 = v128_load(prev_rgba.add(cb) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask)
        };
        let d1 = {
            let c1 = v128_load(curr_rgba.add(cb + 16) as *const v128);
            let p1 = v128_load(prev_rgba.add(cb + 16) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask)
        };
        let d2 = {
            let c2 = v128_load(curr_rgba.add(cb + 32) as *const v128);
            let p2 = v128_load(prev_rgba.add(cb + 32) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask)
        };
        let d3 = {
            let c3 = v128_load(curr_rgba.add(cb + 48) as *const v128);
            let p3 = v128_load(prev_rgba.add(cb + 48) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask)
        };

        let eval16 = if UPDATE_STREAK {
            let within16 = pack4_le_tol16_simd128(d0, d1, d2, d3, tol3, zero_i32);
            let old_streak = v128_load(stat_streak_ptr.add(p_base) as *const v128);
            let new_streak = v128_and(u8x16_add_sat(old_streak, one_u8), within16);
            v128_store(stat_streak_ptr.add(p_base) as *mut v128, new_streak);
            let ev16 = u8x16_ge(new_streak, min_streak);
            v128_store(eval_mask_ptr.add(p_base) as *mut v128, ev16);
            ev16
        } else if HAS_MASK {
            v128_load(eval_mask_const.add(p_base) as *const v128)
        } else {
            zero_i32
        };
        if HAS_MASK {
            row_cnt_u8 = i8x16_add(row_cnt_u8, v128_and(eval16, one_u8));
        }

        let s0 = abs_rgb_to_sum_u32x4_simd128(d0);
        let s1 = abs_rgb_to_sum_u32x4_simd128(d1);
        let s2 = abs_rgb_to_sum_u32x4_simd128(d2);
        let s3 = abs_rgb_to_sum_u32x4_simd128(d3);
        let cur_rgb = i32x4_add(i32x4_add(s0, s1), i32x4_add(s2, s3));
        *acc_rgb_ptr.add(gx) = if FIRST_GY {
            cur_rgb
        } else {
            i32x4_add(*acc_rgb_ptr.add(gx), cur_rgb)
        };
        let st0 = i32x4_lt(s0, heal_gate_i32);
        let st1 = i32x4_lt(s1, heal_gate_i32);
        let st2 = i32x4_lt(s2, heal_gate_i32);
        let st3 = i32x4_lt(s3, heal_gate_i32);

        macro_rules! eca_bg_vec4 {
            ($off:expr, $mv:expr, $st:expr) => {{
                let yv = v128_load(luma_ptr.add(p_base + $off) as *const v128);
                let bv = v128_load(bg_luma_ptr.add(p_base + $off) as *const v128);
                let df = f32x4_sub(yv, f32x4_add(f32x4_mul(v_alpha, bv), v_c));
                let rv = f32x4_abs(df);
                v128_store(eca_res_ptr.add(p_base + $off) as *mut v128, rv);
                let r_bg_s = v128_bitselect(v_rate_bg_heal, v_rate_bg, $st);
                let r_fg_s = v128_bitselect(v_rate_fg_heal, v_rate_fg, $st);
                let mut r_eff = v128_bitselect(r_bg_s, r_fg_s, f32x4_lt(rv, v_motion_gate));
                if HAS_MASK {
                    let r_ev_s = v128_bitselect(v_rate_eval_heal, v_rate_eval, $st);
                    r_eff = v128_bitselect(
                        r_ev_s,
                        r_eff,
                        v128_and($mv, f32x4_lt(rv, v_eval_gate)),
                    );
                }
                let bv_new = f32x4_add(bv, f32x4_mul(r_eff, f32x4_sub(yv, bv)));
                v128_store(
                    bg_luma_ptr.add(p_base + $off) as *mut v128,
                    bv_new,
                );
                if HAS_MASK {
                    (
                        rv,
                        v128_and(f32x4_mul(df, df), $mv),
                        v128_and(f32x4_mul(yv, yv), $mv),
                    )
                } else {
                    (rv, zero_i32, zero_i32)
                }
            }};
        }

        let (r0, e01, sy01, r1) = if HAS_MASK {
            let lo8 = i16x8_extend_low_i8x16(eval16);
            let mv0 = i32x4_extend_low_i16x8(lo8);
            let mv1 = i32x4_extend_high_i16x8(lo8);
            let (r0, e0, sy0) = eca_bg_vec4!(0, mv0, st0);
            let (r1, e1, sy1) = eca_bg_vec4!(4, mv1, st1);
            (r0, f32x4_add(e0, e1), f32x4_add(sy0, sy1), r1)
        } else {
            let (r0, _, _) = eca_bg_vec4!(0, zero_i32, st0);
            let (r1, _, _) = eca_bg_vec4!(4, zero_i32, st1);
            (r0, zero_i32, zero_i32, r1)
        };
        let s01 = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(r0, r1),
            i32x4_shuffle::<1, 3, 5, 7>(r0, r1),
        );
        let cur_r01 = f32x4_add(v128_andnot(r0, st0), v128_andnot(r1, st1));

        let (r2, e2, sy2, r3, e3, sy3) = if HAS_MASK {
            let hi8 = i16x8_extend_high_i8x16(eval16);
            let mv2 = i32x4_extend_low_i16x8(hi8);
            let mv3 = i32x4_extend_high_i16x8(hi8);
            let (r2, e2, sy2) = eca_bg_vec4!(8, mv2, st2);
            let (r3, e3, sy3) = eca_bg_vec4!(12, mv3, st3);
            (r2, e2, sy2, r3, e3, sy3)
        } else {
            let (r2, _, _) = eca_bg_vec4!(8, zero_i32, st2);
            let (r3, _, _) = eca_bg_vec4!(12, zero_i32, st3);
            (r2, zero_i32, zero_i32, r3, zero_i32, zero_i32)
        };
        if HAS_MASK {
            acc_err2 = f32x4_add(acc_err2, f32x4_add(e01, f32x4_add(e2, e3)));
            acc_sig2 = f32x4_add(acc_sig2, f32x4_add(sy01, f32x4_add(sy2, sy3)));
        }

        let cur_r = f32x4_add(
            cur_r01,
            f32x4_add(v128_andnot(r2, st2), v128_andnot(r3, st3)),
        );
        let acc_eca_x = if FIRST_GY {
            cur_r
        } else {
            f32x4_add(*acc_eca_ptr.add(gx), cur_r)
        };
        if LAST_GY {
            let acc_rgb_x = *acc_rgb_ptr.add(gx);
            let tile_v = f32x4_add(
                f32x4_mul(acc_eca_x, w_eca),
                f32x4_mul(f32x4_convert_u32x4(acc_rgb_x), w_rgb),
            );
            *grid_row.add(gx) = hsum_f32x4(tile_v);
        } else {
            *acc_eca_ptr.add(gx) = acc_eca_x;
        }

        let s23 = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(r2, r3),
            i32x4_shuffle::<1, 3, 5, 7>(r2, r3),
        );
        let h4 = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(s01, s23),
            i32x4_shuffle::<1, 3, 5, 7>(s01, s23),
        );
        if PYR_STAGE == 0 {
            *pyr_a_ptr.add(gx) = h4;
        } else if PYR_STAGE == 1 {
            *pyr_a_ptr.add(gx) = f32x4_add(*pyr_a_ptr.add(gx), h4);
        } else {
            let pyr_a = f32x4_add(*pyr_a_ptr.add(gx), h4);
            v128_store(pyr_out_row.add(gx * 4) as *mut v128, f32x4_mul(pyr_a, q16));
        }
        p_base += 16;
    }

    if HAS_MASK {
        *out_err2 = acc_err2;
        *out_sig2 = acc_sig2;
        let sum16 = u16x8_extadd_pairwise_u8x16(row_cnt_u8);
        *out_cnt_i = i32x4_add(*out_cnt_i, u32x4_extadd_pairwise_u16x8(sum16));
    }
}

#[inline(never)]
unsafe fn stat_fused_hd40_grid_loop<const HAS_MASK: bool, const UPDATE_STREAK: bool>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eval_mask_const: *const u8,
    stat_streak_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    grid_h: usize,
    alpha: f32,
    offset_c: f32,
    adapt_rate: f32,
    out_err2: &mut v128,
    out_sig2: &mut v128,
    out_cnt_i: &mut v128,
) {
    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);
    let v_rate_eval = f32x4_splat(0.80);
    let v_eval_gate = f32x4_splat(0.053);
    let w_eca = f32x4_splat(0.75);
    let w_rgb = f32x4_splat(0.25 / 765.0);
    let heal_alpha = f32x4_splat(0.65);
    let one_minus_heal = f32x4_sub(f32x4_splat(1.0), heal_alpha);
    let v_rate_bg_heal = f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_bg));
    let v_rate_fg_heal = f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_fg));
    let v_rate_eval_heal = f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_eval));
    let q16 = f32x4_splat(1.0 / 16.0);

    let zero_v = f32x4_splat(0.0);
    let mut acc_err2 = zero_v;
    let mut acc_sig2 = zero_v;
    let mut acc_cnt_i = i32x4_splat(0);

    let mut acc_r_buf = [i32x4_splat(0); 40];
    let acc_eca_ptr = acc_r_buf.as_mut_ptr() as *mut v128;
    let mut acc_rgb_buf = [i32x4_splat(0); 40];
    let acc_rgb_ptr = acc_rgb_buf.as_mut_ptr() as *mut v128;
    let mut pyr_a_buf = [i32x4_splat(0); 40];
    let pyr_a_ptr = pyr_a_buf.as_mut_ptr() as *mut v128;

    for gy in 0..grid_h {
        let base_y = gy * 16;
        let grid_row = grid_out.add(gy * 40);
        for ry_blk in 0..4usize {
            let pyr_out_row = pyr_eca_ptr.add((gy * 4 + ry_blk) * 160);
            let y_blk = base_y + ry_blk * 4;
            let row0 = y_blk * 640;
            if ry_blk == 0 {
                stat_fused_hd40_row::<HAS_MASK, UPDATE_STREAK, true, 0, false>(
                    row0, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q16, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            } else {
                stat_fused_hd40_row::<HAS_MASK, UPDATE_STREAK, false, 0, false>(
                    row0, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q16, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            }
            for ry_sub in 1..3usize {
                stat_fused_hd40_row::<HAS_MASK, UPDATE_STREAK, false, 1, false>(
                    row0 + ry_sub * 640, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q16, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            }
            if ry_blk == 3 {
                stat_fused_hd40_row::<HAS_MASK, UPDATE_STREAK, false, 2, true>(
                    row0 + 3 * 640, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q16, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            } else {
                stat_fused_hd40_row::<HAS_MASK, UPDATE_STREAK, false, 2, false>(
                    row0 + 3 * 640, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q16, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            }
        }
    }

    *out_err2 = f32x4_add(*out_err2, acc_err2);
    *out_sig2 = f32x4_add(*out_sig2, acc_sig2);
    *out_cnt_i = i32x4_add(*out_cnt_i, acc_cnt_i);
}

#[inline(always)]
unsafe fn stat_fused_sd20_row<
    const HAS_MASK: bool,
    const UPDATE_STREAK: bool,
    const FIRST_GY: bool,
    const PYR_ODD: bool,
    const LAST_GY: bool,
>(
    row_p0: usize,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eval_mask_const: *const u8,
    stat_streak_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    acc_eca_ptr: *mut v128,
    acc_rgb_ptr: *mut v128,
    pyr_a_ptr: *mut v128,
    pyr_b_ptr: *mut v128,
    pyr_out_row: *mut f32,
    grid_row: *mut f32,
    v_alpha: v128,
    v_c: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    v_rate_bg_heal: v128,
    v_rate_fg_heal: v128,
    v_rate_eval_heal: v128,
    w_eca: v128,
    w_rgb: v128,
    q4: v128,
    out_err2: &mut v128,
    out_sig2: &mut v128,
    out_cnt_i: &mut v128,
) {
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let tol3 = u8x16_splat(3);
    let zero_i32 = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let min_streak = u8x16_splat(12);
    let heal_gate_i32 = i32x4_splat(9);

    let mut acc_err2 = *out_err2;
    let mut acc_sig2 = *out_sig2;
    let mut row_cnt_u8 = zero_i32;

    let mut p_base = row_p0;
    for gx in 0..20usize {
        let cb = p_base * 4;
        let d0 = {
            let c0 = v128_load(curr_rgba.add(cb) as *const v128);
            let p0 = v128_load(prev_rgba.add(cb) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask)
        };
        let d1 = {
            let c1 = v128_load(curr_rgba.add(cb + 16) as *const v128);
            let p1 = v128_load(prev_rgba.add(cb + 16) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask)
        };
        let d2 = {
            let c2 = v128_load(curr_rgba.add(cb + 32) as *const v128);
            let p2 = v128_load(prev_rgba.add(cb + 32) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask)
        };
        let d3 = {
            let c3 = v128_load(curr_rgba.add(cb + 48) as *const v128);
            let p3 = v128_load(prev_rgba.add(cb + 48) as *const v128);
            v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask)
        };

        let eval16 = if UPDATE_STREAK {
            let within16 = pack4_le_tol16_simd128(d0, d1, d2, d3, tol3, zero_i32);
            let old_streak = v128_load(stat_streak_ptr.add(p_base) as *const v128);
            let new_streak = v128_and(u8x16_add_sat(old_streak, one_u8), within16);
            v128_store(stat_streak_ptr.add(p_base) as *mut v128, new_streak);
            let ev16 = u8x16_ge(new_streak, min_streak);
            v128_store(eval_mask_ptr.add(p_base) as *mut v128, ev16);
            ev16
        } else if HAS_MASK {
            v128_load(eval_mask_const.add(p_base) as *const v128)
        } else {
            zero_i32
        };
        if HAS_MASK {
            row_cnt_u8 = i8x16_add(row_cnt_u8, v128_and(eval16, one_u8));
        }

        let s0 = abs_rgb_to_sum_u32x4_simd128(d0);
        let s1 = abs_rgb_to_sum_u32x4_simd128(d1);
        let s2 = abs_rgb_to_sum_u32x4_simd128(d2);
        let s3 = abs_rgb_to_sum_u32x4_simd128(d3);
        let cur_rgb = i32x4_add(i32x4_add(s0, s1), i32x4_add(s2, s3));
        *acc_rgb_ptr.add(gx) = if FIRST_GY {
            cur_rgb
        } else {
            i32x4_add(*acc_rgb_ptr.add(gx), cur_rgb)
        };
        let st0 = i32x4_lt(s0, heal_gate_i32);
        let st1 = i32x4_lt(s1, heal_gate_i32);
        let st2 = i32x4_lt(s2, heal_gate_i32);
        let st3 = i32x4_lt(s3, heal_gate_i32);

        macro_rules! eca_bg_vec4_sd {
            ($off:expr, $mv:expr, $st:expr) => {{
                let yv = v128_load(luma_ptr.add(p_base + $off) as *const v128);
                let bv = v128_load(bg_luma_ptr.add(p_base + $off) as *const v128);
                let df = f32x4_sub(yv, f32x4_add(f32x4_mul(v_alpha, bv), v_c));
                let rv = f32x4_abs(df);
                v128_store(eca_res_ptr.add(p_base + $off) as *mut v128, rv);
                let r_bg_s = v128_bitselect(v_rate_bg_heal, v_rate_bg, $st);
                let r_fg_s = v128_bitselect(v_rate_fg_heal, v_rate_fg, $st);
                let mut r_eff = v128_bitselect(r_bg_s, r_fg_s, f32x4_lt(rv, v_motion_gate));
                if HAS_MASK {
                    let r_ev_s = v128_bitselect(v_rate_eval_heal, v_rate_eval, $st);
                    r_eff = v128_bitselect(
                        r_ev_s,
                        r_eff,
                        v128_and($mv, f32x4_lt(rv, v_eval_gate)),
                    );
                }
                let bv_new = f32x4_add(bv, f32x4_mul(r_eff, f32x4_sub(yv, bv)));
                v128_store(
                    bg_luma_ptr.add(p_base + $off) as *mut v128,
                    bv_new,
                );
                if HAS_MASK {
                    (
                        rv,
                        v128_and(f32x4_mul(df, df), $mv),
                        v128_and(f32x4_mul(yv, yv), $mv),
                    )
                } else {
                    (rv, zero_i32, zero_i32)
                }
            }};
        }

        let (r0, e01, sy01, r1) = if HAS_MASK {
            let lo8 = i16x8_extend_low_i8x16(eval16);
            let mv0 = i32x4_extend_low_i16x8(lo8);
            let mv1 = i32x4_extend_high_i16x8(lo8);
            let (r0, e0, sy0) = eca_bg_vec4_sd!(0, mv0, st0);
            let (r1, e1, sy1) = eca_bg_vec4_sd!(4, mv1, st1);
            (r0, f32x4_add(e0, e1), f32x4_add(sy0, sy1), r1)
        } else {
            let (r0, _, _) = eca_bg_vec4_sd!(0, zero_i32, st0);
            let (r1, _, _) = eca_bg_vec4_sd!(4, zero_i32, st1);
            (r0, zero_i32, zero_i32, r1)
        };
        let s01 = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(r0, r1),
            i32x4_shuffle::<1, 3, 5, 7>(r0, r1),
        );
        let cur_r01 = f32x4_add(v128_andnot(r0, st0), v128_andnot(r1, st1));

        let (r2, e2, sy2, r3, e3, sy3) = if HAS_MASK {
            let hi8 = i16x8_extend_high_i8x16(eval16);
            let mv2 = i32x4_extend_low_i16x8(hi8);
            let mv3 = i32x4_extend_high_i16x8(hi8);
            let (r2, e2, sy2) = eca_bg_vec4_sd!(8, mv2, st2);
            let (r3, e3, sy3) = eca_bg_vec4_sd!(12, mv3, st3);
            (r2, e2, sy2, r3, e3, sy3)
        } else {
            let (r2, _, _) = eca_bg_vec4_sd!(8, zero_i32, st2);
            let (r3, _, _) = eca_bg_vec4_sd!(12, zero_i32, st3);
            (r2, zero_i32, zero_i32, r3, zero_i32, zero_i32)
        };
        if HAS_MASK {
            acc_err2 = f32x4_add(acc_err2, f32x4_add(e01, f32x4_add(e2, e3)));
            acc_sig2 = f32x4_add(acc_sig2, f32x4_add(sy01, f32x4_add(sy2, sy3)));
        }

        let cur_r = f32x4_add(
            cur_r01,
            f32x4_add(v128_andnot(r2, st2), v128_andnot(r3, st3)),
        );
        let acc_eca_x = if FIRST_GY {
            cur_r
        } else {
            f32x4_add(*acc_eca_ptr.add(gx), cur_r)
        };
        if LAST_GY {
            let acc_rgb_x = *acc_rgb_ptr.add(gx);
            let tile_v = f32x4_add(
                f32x4_mul(acc_eca_x, w_eca),
                f32x4_mul(f32x4_convert_u32x4(acc_rgb_x), w_rgb),
            );
            *grid_row.add(gx) = hsum_f32x4(tile_v);
        } else {
            *acc_eca_ptr.add(gx) = acc_eca_x;
        }

        let s23 = f32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(r2, r3),
            i32x4_shuffle::<1, 3, 5, 7>(r2, r3),
        );
        if !PYR_ODD {
            *pyr_a_ptr.add(gx) = s01;
            *pyr_b_ptr.add(gx) = s23;
        } else {
            let v0 = f32x4_mul(f32x4_add(*pyr_a_ptr.add(gx), s01), q4);
            let v1 = f32x4_mul(f32x4_add(*pyr_b_ptr.add(gx), s23), q4);
            let dst = pyr_out_row.add(gx * 8);
            v128_store(dst as *mut v128, v0);
            v128_store(dst.add(4) as *mut v128, v1);
        }
        p_base += 16;
    }

    if HAS_MASK {
        *out_err2 = acc_err2;
        *out_sig2 = acc_sig2;
        let sum16 = u16x8_extadd_pairwise_u8x16(row_cnt_u8);
        *out_cnt_i = i32x4_add(*out_cnt_i, u32x4_extadd_pairwise_u16x8(sum16));
    }
}

#[inline(never)]
unsafe fn stat_fused_sd20_grid_loop<const HAS_MASK: bool, const UPDATE_STREAK: bool>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eval_mask_const: *const u8,
    stat_streak_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    grid_h: usize,
    alpha: f32,
    offset_c: f32,
    adapt_rate: f32,
    out_err2: &mut v128,
    out_sig2: &mut v128,
    out_cnt_i: &mut v128,
) {
    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);
    let v_rate_eval = f32x4_splat(0.80);
    let v_eval_gate = f32x4_splat(0.053);
    let w_eca = f32x4_splat(0.75);
    let w_rgb = f32x4_splat(0.25 / 765.0);
    let heal_alpha = f32x4_splat(0.65);
    let one_minus_heal = f32x4_sub(f32x4_splat(1.0), heal_alpha);
    let v_rate_bg_heal = f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_bg));
    let v_rate_fg_heal = f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_fg));
    let v_rate_eval_heal = f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_eval));
    let q4 = f32x4_splat(0.25);

    let zero_v = f32x4_splat(0.0);
    let mut acc_err2 = zero_v;
    let mut acc_sig2 = zero_v;
    let mut acc_cnt_i = i32x4_splat(0);

    let mut acc_r_buf = [i32x4_splat(0); 20];
    let acc_eca_ptr = acc_r_buf.as_mut_ptr() as *mut v128;
    let mut acc_rgb_buf = [i32x4_splat(0); 20];
    let acc_rgb_ptr = acc_rgb_buf.as_mut_ptr() as *mut v128;
    let mut pyr_a_buf = [i32x4_splat(0); 20];
    let pyr_a_ptr = pyr_a_buf.as_mut_ptr() as *mut v128;
    let mut pyr_b_buf = [i32x4_splat(0); 20];
    let pyr_b_ptr = pyr_b_buf.as_mut_ptr() as *mut v128;

    for gy in 0..grid_h {
        let base_y = gy * 16;
        let grid_row = grid_out.add(gy * 20);
        for ry_pair in 0..8usize {
            let pyr_out_row = pyr_eca_ptr.add((gy * 8 + ry_pair) * 160);
            let row0 = (base_y + ry_pair * 2) * 320;
            if ry_pair == 0 {
                stat_fused_sd20_row::<HAS_MASK, UPDATE_STREAK, true, false, false>(
                    row0, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_b_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q4, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            } else {
                stat_fused_sd20_row::<HAS_MASK, UPDATE_STREAK, false, false, false>(
                    row0, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_b_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q4, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            }
            if ry_pair == 7 {
                stat_fused_sd20_row::<HAS_MASK, UPDATE_STREAK, false, true, true>(
                    row0 + 320, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_b_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q4, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            } else {
                stat_fused_sd20_row::<HAS_MASK, UPDATE_STREAK, false, true, false>(
                    row0 + 320, curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eval_mask_ptr,
                    eval_mask_const, stat_streak_ptr, eca_res_ptr, acc_eca_ptr, acc_rgb_ptr,
                    pyr_a_ptr, pyr_b_ptr, pyr_out_row, grid_row, v_alpha, v_c, v_rate_bg, v_rate_fg,
                    v_motion_gate, v_rate_eval, v_eval_gate, v_rate_bg_heal, v_rate_fg_heal,
                    v_rate_eval_heal, w_eca, w_rgb, q4, &mut acc_err2, &mut acc_sig2, &mut acc_cnt_i,
                );
            }
        }
    }

    *out_err2 = f32x4_add(*out_err2, acc_err2);
    *out_sig2 = f32x4_add(*out_sig2, acc_sig2);
    *out_cnt_i = i32x4_add(*out_cnt_i, acc_cnt_i);
}

/// Fused stationary Rank-2 ECA projection, 16x16 motion-grid accumulation,
/// stationary background healing, and optional in-register stationarity streak update
/// without pan/exposure overhead or stack spills.
#[inline(never)]
unsafe fn eca_and_motion_grid_stat_fused_impl<const HAS_MASK: bool, const UPDATE_STREAK: bool>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    stat_streak_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    alpha: f32,
    offset_c: f32,
    adapt_rate: f32,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> f32 {
    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);
    let v_rate_eval = f32x4_splat(0.80);
    let v_eval_gate = f32x4_splat(0.053);
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let heal_gate_rgb = f32x4_splat(0.0110 * 765.0);
    let w_eca = f32x4_splat(0.75);
    let w_rgb = f32x4_splat(0.25 / 765.0);
    let heal_alpha = f32x4_splat(0.65);

    let tol3 = u8x16_splat(3);
    let zero_i32 = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let min_streak = u8x16_splat(12);
    let eval_mask_const = eval_mask_ptr as *const u8;
    let write_eval_mask = UPDATE_STREAK && !eval_mask_ptr.is_null();

    let mut acc_eval_err2 = f32x4_splat(0.0);
    let mut acc_eval_sig2 = f32x4_splat(0.0);
    let mut acc_eval_cnt_i = i32x4_splat(0);
    let fuse_pyr = !pyr_eca_ptr.is_null() && scale_px <= 2 && frame_w == 320 * scale_px;
    let zero_v = f32x4_splat(0.0);
    let q4 = f32x4_splat(0.25);
    let q16 = f32x4_splat(1.0 / 16.0);

    if fuse_pyr && scale_px == 2 && grid_w == 40 && (!UPDATE_STREAK || write_eval_mask) {
        stat_fused_hd40_grid_loop::<HAS_MASK, UPDATE_STREAK>(
            curr_rgba,
            prev_rgba,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            eval_mask_const,
            stat_streak_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
            grid_out,
            grid_h,
            alpha,
            offset_c,
            adapt_rate,
            &mut acc_eval_err2,
            &mut acc_eval_sig2,
            &mut acc_eval_cnt_i,
        );
    } else if fuse_pyr && scale_px == 1 && grid_w == 20 && (!UPDATE_STREAK || write_eval_mask) {
        stat_fused_sd20_grid_loop::<HAS_MASK, UPDATE_STREAK>(
            curr_rgba,
            prev_rgba,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            eval_mask_const,
            stat_streak_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
            grid_out,
            grid_h,
            alpha,
            offset_c,
            adapt_rate,
            &mut acc_eval_err2,
            &mut acc_eval_sig2,
            &mut acc_eval_cnt_i,
        );
    } else if grid_w <= 40 {
        for gy in 0..grid_h {
            let base_y = gy * 16;
            let mut acc_row = [zero_v; 40];
            let mut pyr_a_row = [zero_v; 40];
            let mut pyr_b_row = [zero_v; 20];
            for ry in 0..16usize {
                stat_fused_grid_le40_row::<HAS_MASK, UPDATE_STREAK, false, 0>(
                    gy,
                    ry,
                    ry >> 2,
                    (base_y + ry) * frame_w,
                    grid_w,
                    scale_px,
                    fuse_pyr,
                    curr_rgba,
                    prev_rgba,
                    luma_ptr,
                    bg_luma_ptr,
                    eval_mask_ptr,
                    eval_mask_const,
                    stat_streak_ptr,
                    eca_res_ptr,
                    pyr_eca_ptr,
                    write_eval_mask,
                    v_alpha,
                    v_c,
                    v_rate_bg,
                    v_rate_fg,
                    v_motion_gate,
                    v_rate_eval,
                    v_eval_gate,
                    rgb_mask,
                    heal_gate_rgb,
                    w_eca,
                    w_rgb,
                    heal_alpha,
                    tol3,
                    zero_i32,
                    one_u8,
                    min_streak,
                    zero_v,
                    q4,
                    q16,
                    &mut acc_eval_err2,
                    &mut acc_eval_sig2,
                    &mut acc_eval_cnt_i,
                    &mut acc_row,
                    &mut pyr_a_row,
                    &mut pyr_b_row,
                );
            }
            for gx in 0..grid_w {
                *grid_out.add(gy * grid_w + gx) = hsum_f32x4(acc_row[gx]);
            }
        }
    } else {
        for gy in 0..grid_h {
            let base_y = gy * 16;
            for gx in 0..grid_w {
                let block_off = base_y * frame_w + gx * 16;
                let mut acc = zero_v;
                let mut pyr_a = zero_v;
                let mut pyr_b = zero_v;
                for ry in 0..16usize {
                    let p_base = block_off + ry * frame_w;
                    let (a0, a1, a2, a3, mv0, mv1, mv2, mv3) = if UPDATE_STREAK {
                        let cb = p_base * 4;
                        let c0 = v128_load(curr_rgba.add(cb) as *const v128);
                        let p0 = v128_load(prev_rgba.add(cb) as *const v128);
                        let c1 = v128_load(curr_rgba.add(cb + 16) as *const v128);
                        let p1 = v128_load(prev_rgba.add(cb + 16) as *const v128);
                        let c2 = v128_load(curr_rgba.add(cb + 32) as *const v128);
                        let p2 = v128_load(prev_rgba.add(cb + 32) as *const v128);
                        let c3 = v128_load(curr_rgba.add(cb + 48) as *const v128);
                        let p3 = v128_load(prev_rgba.add(cb + 48) as *const v128);

                        let d0 = v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask);
                        let d1 = v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask);
                        let d2 = v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask);
                        let d3 = v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask);

                        let within16 = pack4_le_tol16_simd128(d0, d1, d2, d3, tol3, zero_i32);
                        let old_streak = v128_load(stat_streak_ptr.add(p_base) as *const v128);
                        let new_streak = v128_and(u8x16_add_sat(old_streak, one_u8), within16);
                        v128_store(stat_streak_ptr.add(p_base) as *mut v128, new_streak);

                        let eval16 = u8x16_ge(new_streak, min_streak);
                        if write_eval_mask {
                            v128_store(eval_mask_ptr.add(p_base) as *mut v128, eval16);
                        }
                        if HAS_MASK {
                            let lo16 = i16x8_extend_low_i8x16(eval16);
                            let hi16 = i16x8_extend_high_i8x16(eval16);
                            (
                                d0,
                                d1,
                                d2,
                                d3,
                                i32x4_extend_low_i16x8(lo16),
                                i32x4_extend_high_i16x8(lo16),
                                i32x4_extend_low_i16x8(hi16),
                                i32x4_extend_high_i16x8(hi16),
                            )
                        } else {
                            (d0, d1, d2, d3, zero_i32, zero_i32, zero_i32, zero_i32)
                        }
                    } else {
                        (zero_v, zero_v, zero_v, zero_v, zero_v, zero_v, zero_v, zero_v)
                    };

                    let r0 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
                        p_base,
                        curr_rgba,
                        prev_rgba,
                        luma_ptr,
                        bg_luma_ptr,
                        eval_mask_const,
                        eca_res_ptr,
                        a0,
                        mv0,
                        v_alpha,
                        v_c,
                        v_rate_bg,
                        v_rate_fg,
                        v_motion_gate,
                        v_rate_eval,
                        v_eval_gate,
                        rgb_mask,
                        heal_gate_rgb,
                        w_eca,
                        w_rgb,
                        heal_alpha,
                        &mut acc_eval_err2,
                        &mut acc_eval_sig2,
                        &mut acc_eval_cnt_i,
                        &mut acc,
                    );
                    let r1 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
                        p_base + 4,
                        curr_rgba,
                        prev_rgba,
                        luma_ptr,
                        bg_luma_ptr,
                        eval_mask_const,
                        eca_res_ptr,
                        a1,
                        mv1,
                        v_alpha,
                        v_c,
                        v_rate_bg,
                        v_rate_fg,
                        v_motion_gate,
                        v_rate_eval,
                        v_eval_gate,
                        rgb_mask,
                        heal_gate_rgb,
                        w_eca,
                        w_rgb,
                        heal_alpha,
                        &mut acc_eval_err2,
                        &mut acc_eval_sig2,
                        &mut acc_eval_cnt_i,
                        &mut acc,
                    );
                    let r2 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
                        p_base + 8,
                        curr_rgba,
                        prev_rgba,
                        luma_ptr,
                        bg_luma_ptr,
                        eval_mask_const,
                        eca_res_ptr,
                        a2,
                        mv2,
                        v_alpha,
                        v_c,
                        v_rate_bg,
                        v_rate_fg,
                        v_motion_gate,
                        v_rate_eval,
                        v_eval_gate,
                        rgb_mask,
                        heal_gate_rgb,
                        w_eca,
                        w_rgb,
                        heal_alpha,
                        &mut acc_eval_err2,
                        &mut acc_eval_sig2,
                        &mut acc_eval_cnt_i,
                        &mut acc,
                    );
                    let r3 = stat_fused_chunk4::<HAS_MASK, UPDATE_STREAK>(
                        p_base + 12,
                        curr_rgba,
                        prev_rgba,
                        luma_ptr,
                        bg_luma_ptr,
                        eval_mask_const,
                        eca_res_ptr,
                        a3,
                        mv3,
                        v_alpha,
                        v_c,
                        v_rate_bg,
                        v_rate_fg,
                        v_motion_gate,
                        v_rate_eval,
                        v_eval_gate,
                        rgb_mask,
                        heal_gate_rgb,
                        w_eca,
                        w_rgb,
                        heal_alpha,
                        &mut acc_eval_err2,
                        &mut acc_eval_sig2,
                        &mut acc_eval_cnt_i,
                        &mut acc,
                    );
                    if fuse_pyr {
                        flush_pyr_eca_reg_simd128(
                            gy,
                            gx,
                            ry,
                            scale_px,
                            q4,
                            q16,
                            r0,
                            r1,
                            r2,
                            r3,
                            &mut pyr_a,
                            &mut pyr_b,
                            pyr_eca_ptr,
                        );
                    }
                }
                *grid_out.add(gy * grid_w + gx) = hsum_f32x4(acc);
            }
        }
    }

    for py in (grid_h * 16)..frame_h {
        let row_off = py * frame_w;
        let mut px = 0usize;
        if UPDATE_STREAK {
            while px + 16 <= frame_w {
                let p_base = row_off + px;
                let cb = p_base * 4;
                let c0 = v128_load(curr_rgba.add(cb) as *const v128);
                let p0 = v128_load(prev_rgba.add(cb) as *const v128);
                let c1 = v128_load(curr_rgba.add(cb + 16) as *const v128);
                let p1 = v128_load(prev_rgba.add(cb + 16) as *const v128);
                let c2 = v128_load(curr_rgba.add(cb + 32) as *const v128);
                let p2 = v128_load(prev_rgba.add(cb + 32) as *const v128);
                let c3 = v128_load(curr_rgba.add(cb + 48) as *const v128);
                let p3 = v128_load(prev_rgba.add(cb + 48) as *const v128);

                let d0 = v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask);
                let d1 = v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask);
                let d2 = v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask);
                let d3 = v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask);

                let within16 = pack4_le_tol16_simd128(d0, d1, d2, d3, tol3, zero_i32);
                let old_streak = v128_load(stat_streak_ptr.add(p_base) as *const v128);
                let new_streak = v128_and(u8x16_add_sat(old_streak, one_u8), within16);
                v128_store(stat_streak_ptr.add(p_base) as *mut v128, new_streak);

                let eval16 = u8x16_ge(new_streak, min_streak);
                if write_eval_mask {
                    v128_store(eval_mask_ptr.add(p_base) as *mut v128, eval16);
                }
                let (mv0, mv1, mv2, mv3) = if HAS_MASK {
                    let lo16 = i16x8_extend_low_i8x16(eval16);
                    let hi16 = i16x8_extend_high_i8x16(eval16);
                    (
                        i32x4_extend_low_i16x8(lo16),
                        i32x4_extend_high_i16x8(lo16),
                        i32x4_extend_low_i16x8(hi16),
                        i32x4_extend_high_i16x8(hi16),
                    )
                } else {
                    (zero_i32, zero_i32, zero_i32, zero_i32)
                };

                let ds = [d0, d1, d2, d3];
                let mvs = [mv0, mv1, mv2, mv3];
                for k in 0..4usize {
                    let p = p_base + k * 4;
                    let y_cur = v128_load(luma_ptr.add(p) as *const v128);
                    let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);
                    let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
                    let diff = f32x4_sub(y_cur, proj);
                    let res = f32x4_abs(diff);
                    v128_store(eca_res_ptr.add(p) as *mut v128, res);

                    let rate = if HAS_MASK {
                        let mv = mvs[k];
                        let m_diff = v128_and(diff, mv);
                        let m_sig = v128_and(y_cur, mv);
                        acc_eval_err2 = f32x4_add(acc_eval_err2, f32x4_mul(m_diff, m_diff));
                        acc_eval_sig2 = f32x4_add(acc_eval_sig2, f32x4_mul(m_sig, m_sig));
                        acc_eval_cnt_i = i32x4_sub(acc_eval_cnt_i, mv);

                        let is_bg = f32x4_lt(res, v_motion_gate);
                        let is_eval_heal = v128_and(mv, f32x4_lt(res, v_eval_gate));
                        v128_bitselect(v_rate_eval, v128_bitselect(v_rate_bg, v_rate_fg, is_bg), is_eval_heal)
                    } else {
                        let is_bg = f32x4_lt(res, v_motion_gate);
                        v128_bitselect(v_rate_bg, v_rate_fg, is_bg)
                    };
                    let b_eca = f32x4_add(b_old, f32x4_mul(rate, f32x4_sub(y_cur, b_old)));
                    let rgb_sum = abs_rgb_to_sum_f32x4_simd128(ds[k]);
                    let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
                    let bg_healed = f32x4_add(b_eca, f32x4_mul(heal_alpha, f32x4_sub(y_cur, b_eca)));
                    v128_store(
                        bg_luma_ptr.add(p) as *mut v128,
                        v128_bitselect(bg_healed, b_eca, is_stat),
                    );
                }
                px += 16;
            }
        } else {
            while px + 4 <= frame_w {
                let p = row_off + px;
                let y_cur = v128_load(luma_ptr.add(p) as *const v128);
                let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);
                let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
                let diff = f32x4_sub(y_cur, proj);
                let res = f32x4_abs(diff);
                v128_store(eca_res_ptr.add(p) as *mut v128, res);

                let rate = if HAS_MASK {
                    let raw_m4 = v128_load32_zero(eval_mask_const.add(p) as *const u32);
                    let mv = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(raw_m4));
                    let m_diff = v128_and(diff, mv);
                    let m_sig = v128_and(y_cur, mv);
                    acc_eval_err2 = f32x4_add(acc_eval_err2, f32x4_mul(m_diff, m_diff));
                    acc_eval_sig2 = f32x4_add(acc_eval_sig2, f32x4_mul(m_sig, m_sig));
                    acc_eval_cnt_i = i32x4_sub(acc_eval_cnt_i, mv);

                    let is_bg = f32x4_lt(res, v_motion_gate);
                    let is_eval_heal = v128_and(mv, f32x4_lt(res, v_eval_gate));
                    v128_bitselect(v_rate_eval, v128_bitselect(v_rate_bg, v_rate_fg, is_bg), is_eval_heal)
                } else {
                    let is_bg = f32x4_lt(res, v_motion_gate);
                    v128_bitselect(v_rate_bg, v_rate_fg, is_bg)
                };
                let b_eca = f32x4_add(b_old, f32x4_mul(rate, f32x4_sub(y_cur, b_old)));

                let c_px = v128_load(curr_rgba.add(p * 4) as *const v128);
                let p_px = v128_load(prev_rgba.add(p * 4) as *const v128);
                let rgb_sum = rgb_abs_sum_f32x4_simd128(c_px, p_px, rgb_mask);
                let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
                let bg_healed = f32x4_add(b_eca, f32x4_mul(heal_alpha, f32x4_sub(y_cur, b_eca)));
                v128_store(
                    bg_luma_ptr.add(p) as *mut v128,
                    v128_bitselect(bg_healed, b_eca, is_stat),
                );
                px += 4;
            }
        }
    }

    let err2 = hsum_f32x4(acc_eval_err2);
    let sig2 = hsum_f32x4(acc_eval_sig2);
    let eval_count = hsum_i32x4(acc_eval_cnt_i) as u32;
    if !out_err2.is_null() {
        *out_err2 = err2;
    }
    if !out_sig2.is_null() {
        *out_sig2 = sig2;
    }
    if !out_eval_count.is_null() {
        *out_eval_count = eval_count;
    }

    if eval_count >= 1024 && sig2 > 1e-3 {
        err2 / sig2
    } else {
        -1.0
    }
}

pub unsafe fn eca_and_motion_grid_stat_fused_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    stat_streak_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    adapt_rate: f32,
    mask_warm: bool,
    update_streak: bool,
    dense_swarm: bool,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> f32 {
    let len = frame_w * frame_h;
    let do_streak = update_streak && !stat_streak_ptr.is_null() && frame_w == grid_w * 16;
    let fit_params = if do_streak {
        if mask_warm {
            eca_fit_affine_params_ex_simd128(
                luma_ptr,
                bg_luma_ptr,
                core::ptr::null(),
                curr_rgba,
                prev_rgba,
                stat_streak_ptr as *const u8,
                len,
                dense_swarm,
            )
        } else {
            eca_fit_affine_params_ex_simd128(
                luma_ptr,
                bg_luma_ptr,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                len,
                dense_swarm,
            )
        }
    } else {
        let fit_mask = if mask_warm {
            eval_mask_ptr as *const u8
        } else {
            core::ptr::null()
        };
        eca_fit_affine_params_ex_simd128(
            luma_ptr,
            bg_luma_ptr,
            fit_mask,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            len,
            dense_swarm,
        )
    };
    let Some((alpha, offset_c)) = fit_params else {
        let zero = f32x4_splat(0.0);
        let mut k = 0usize;
        while k + 4 <= len {
            let y = v128_load(luma_ptr.add(k) as *const v128);
            v128_store(bg_luma_ptr.add(k) as *mut v128, y);
            v128_store(eca_res_ptr.add(k) as *mut v128, zero);
            k += 4;
        }
        while k < len {
            *bg_luma_ptr.add(k) = *luma_ptr.add(k);
            *eca_res_ptr.add(k) = 0.0;
            k += 1;
        }
        if do_streak && !eval_mask_ptr.is_null() {
            update_stat_streak_and_mask_simd128(
                curr_rgba,
                prev_rgba,
                stat_streak_ptr,
                eval_mask_ptr,
                len,
            );
        }
        compute_motion_grid_and_heal_bg_simd128(
            curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, eca_res_ptr, grid_out, frame_w,
            frame_h, grid_w, grid_h,
        );
        if !pyr_eca_ptr.is_null() {
            pool_plane_to_160x90_simd128(eca_res_ptr, pyr_eca_ptr, scale_px);
        }
        if !out_err2.is_null() {
            *out_err2 = 0.0;
        }
        if !out_sig2.is_null() {
            *out_sig2 = 0.0;
        }
        if !out_eval_count.is_null() {
            *out_eval_count = 0;
        }
        return -1.0;
    };

    let has_mask = mask_warm && (do_streak || !eval_mask_ptr.is_null());
    match (has_mask, do_streak) {
        (true, true) => eca_and_motion_grid_stat_fused_impl::<true, true>(
            curr_rgba,
            prev_rgba,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            stat_streak_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
            grid_out,
            frame_w,
            frame_h,
            grid_w,
            grid_h,
            scale_px,
            alpha,
            offset_c,
            adapt_rate,
            out_err2,
            out_sig2,
            out_eval_count,
        ),
        (false, true) => eca_and_motion_grid_stat_fused_impl::<false, true>(
            curr_rgba,
            prev_rgba,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            stat_streak_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
            grid_out,
            frame_w,
            frame_h,
            grid_w,
            grid_h,
            scale_px,
            alpha,
            offset_c,
            adapt_rate,
            out_err2,
            out_sig2,
            out_eval_count,
        ),
        (true, false) => eca_and_motion_grid_stat_fused_impl::<true, false>(
            curr_rgba,
            prev_rgba,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            stat_streak_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
            grid_out,
            frame_w,
            frame_h,
            grid_w,
            grid_h,
            scale_px,
            alpha,
            offset_c,
            adapt_rate,
            out_err2,
            out_sig2,
            out_eval_count,
        ),
        (false, false) => eca_and_motion_grid_stat_fused_impl::<false, false>(
            curr_rgba,
            prev_rgba,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            stat_streak_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
            grid_out,
            frame_w,
            frame_h,
            grid_w,
            grid_h,
            scale_px,
            alpha,
            offset_c,
            adapt_rate,
            out_err2,
            out_sig2,
            out_eval_count,
        ),
    }
}

#[inline(always)]
unsafe fn filter_mask4_simd128(
    eval_mask_ptr: *mut u8,
    p: usize,
    y_cur: v128,
    b_old: v128,
    tol_v: v128,
    raw_m4: v128,
    mv: v128,
) -> v128 {
    let over = f32x4_gt(f32x4_abs(f32x4_sub(y_cur, b_old)), tol_v);
    let clear_m = v128_and(mv, over);
    if v128_any_true(clear_m) {
        let o16 = i16x8_narrow_i32x4(over, over);
        let o8 = i8x16_narrow_i16x8(o16, o16);
        let kept = v128_andnot(raw_m4, o8);
        core::ptr::write_unaligned(
            eval_mask_ptr.add(p) as *mut u32,
            u32x4_extract_lane::<0>(kept),
        );
    }
    v128_andnot(mv, over)
}

#[inline(always)]
unsafe fn pan_fused_interior_chunk4<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
>(
    p: usize,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    heal_alpha: v128,
    heal_gate_rgb: v128,
    w_eca: v128,
    w_rgb: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
    tile_acc: &mut v128,
) -> v128 {
    let rgb_sum = v128_load(eca_res_ptr.add(p) as *const v128);
    let y_cur = v128_load(luma_ptr.add(p) as *const v128);
    let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);

    let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
    let diff = f32x4_sub(y_cur, proj);
    let res = f32x4_abs(diff);
    v128_store(eca_res_ptr.add(p) as *mut v128, res);

    let mask_vec = if HAS_MASK {
        let raw_m4 = v128_load32_zero(eval_mask_ptr.add(p) as *const u32);
        let mut mv = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(raw_m4));
        if FILTER_MASK {
            mv = filter_mask4_simd128(
                eval_mask_ptr,
                p,
                y_cur,
                b_old,
                tol_v,
                raw_m4,
                mv,
            );
        }
        let m_diff = v128_and(diff, mv);
        let m_sig = v128_and(y_cur, mv);
        *acc_eval_err2 = f32x4_add(*acc_eval_err2, f32x4_mul(m_diff, m_diff));
        *acc_eval_sig2 = f32x4_add(*acc_eval_sig2, f32x4_mul(m_sig, m_sig));
        *acc_eval_cnt_i = i32x4_sub(*acc_eval_cnt_i, mv);
        mv
    } else {
        v_alpha
    };

    let is_bg = f32x4_lt(res, v_motion_gate);
    let rate = if ALLOW_EVAL_HEAL {
        let is_eval_heal = v128_and(mask_vec, f32x4_lt(res, v_eval_gate));
        v128_bitselect(v_rate_eval, v128_bitselect(v_rate_bg, v_rate_fg, is_bg), is_eval_heal)
    } else {
        v128_bitselect(v_rate_bg, v_rate_fg, is_bg)
    };
    let b_eca = f32x4_add(b_old, f32x4_mul(rate, f32x4_sub(y_cur, b_old)));

    let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
    let bg_healed = f32x4_add(b_eca, f32x4_mul(heal_alpha, f32x4_sub(y_cur, b_eca)));
    v128_store(
        bg_luma_ptr.add(p) as *mut v128,
        v128_bitselect(bg_healed, b_eca, is_stat),
    );

    let combined = if GATE_RGB {
        v128_andnot(f32x4_add(f32x4_mul(res, w_eca), f32x4_mul(rgb_sum, w_rgb)), is_stat)
    } else {
        f32x4_add(f32x4_mul(v128_andnot(res, is_stat), w_eca), f32x4_mul(rgb_sum, w_rgb))
    };
    *tile_acc = f32x4_add(*tile_acc, combined);
    res
}

#[inline(always)]
unsafe fn pan_fused_border_chunk4<const HAS_MASK: bool, const FILTER_MASK: bool>(
    p: usize,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
) -> v128 {
    let y_cur = v128_load(luma_ptr.add(p) as *const v128);
    let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);

    let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
    let diff = f32x4_sub(y_cur, proj);
    let res = f32x4_abs(diff);
    v128_store(eca_res_ptr.add(p) as *mut v128, res);

    if HAS_MASK {
        let raw_m4 = v128_load32_zero(eval_mask_ptr.add(p) as *const u32);
        let mut mask_vec = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(raw_m4));
        if FILTER_MASK {
            mask_vec = filter_mask4_simd128(
                eval_mask_ptr,
                p,
                y_cur,
                b_old,
                tol_v,
                raw_m4,
                mask_vec,
            );
        }
        let m_diff = v128_and(diff, mask_vec);
        let m_sig = v128_and(y_cur, mask_vec);
        *acc_eval_err2 = f32x4_add(*acc_eval_err2, f32x4_mul(m_diff, m_diff));
        *acc_eval_sig2 = f32x4_add(*acc_eval_sig2, f32x4_mul(m_sig, m_sig));
        *acc_eval_cnt_i = i32x4_sub(*acc_eval_cnt_i, mask_vec);
    }

    v128_store(bg_luma_ptr.add(p) as *mut v128, y_cur);
    res
}

#[inline(always)]
unsafe fn pan_fused_interior_chunk4_val_mv<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
>(
    p: usize,
    rgb_sum: v128,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eca_res_ptr: *mut f32,
    mut mv: v128,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    heal_alpha: v128,
    heal_gate_rgb: v128,
    w_eca: v128,
    w_rgb: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
    tile_acc: &mut v128,
) -> (v128, v128) {
    let y_cur = v128_load(luma_ptr.add(p) as *const v128);
    let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);

    let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
    let diff = f32x4_sub(y_cur, proj);
    let res = f32x4_abs(diff);
    v128_store(eca_res_ptr.add(p) as *mut v128, res);

    let dyb = f32x4_sub(y_cur, b_old);
    let mut over = v_alpha;
    if HAS_MASK {
        if FILTER_MASK {
            over = f32x4_gt(f32x4_abs(dyb), tol_v);
            mv = v128_andnot(mv, over);
        }
        let m_diff = v128_and(diff, mv);
        let m_sig = v128_and(y_cur, mv);
        *acc_eval_err2 = f32x4_add(*acc_eval_err2, f32x4_mul(m_diff, m_diff));
        *acc_eval_sig2 = f32x4_add(*acc_eval_sig2, f32x4_mul(m_sig, m_sig));
        *acc_eval_cnt_i = i32x4_sub(*acc_eval_cnt_i, mv);
    }

    let is_bg = f32x4_lt(res, v_motion_gate);
    let rate = if HAS_MASK && ALLOW_EVAL_HEAL {
        let is_eval_heal = v128_and(mv, f32x4_lt(res, v_eval_gate));
        v128_bitselect(v_rate_eval, v128_bitselect(v_rate_bg, v_rate_fg, is_bg), is_eval_heal)
    } else {
        v128_bitselect(v_rate_bg, v_rate_fg, is_bg)
    };
    let b_eca = f32x4_add(b_old, f32x4_mul(rate, dyb));

    let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
    let bg_healed = f32x4_add(b_eca, f32x4_mul(heal_alpha, f32x4_sub(y_cur, b_eca)));
    v128_store(
        bg_luma_ptr.add(p) as *mut v128,
        v128_bitselect(bg_healed, b_eca, is_stat),
    );

    let combined = if GATE_RGB {
        v128_andnot(f32x4_add(f32x4_mul(res, w_eca), f32x4_mul(rgb_sum, w_rgb)), is_stat)
    } else {
        f32x4_add(f32x4_mul(v128_andnot(res, is_stat), w_eca), f32x4_mul(rgb_sum, w_rgb))
    };
    *tile_acc = f32x4_add(*tile_acc, combined);
    (res, over)
}

#[inline(always)]
unsafe fn pan_fused_interior_chunk4_mv<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
>(
    p: usize,
    rgb_ptr: *const f32,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eca_res_ptr: *mut f32,
    mv: v128,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    heal_alpha: v128,
    heal_gate_rgb: v128,
    w_eca: v128,
    w_rgb: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
    tile_acc: &mut v128,
) -> (v128, v128) {
    pan_fused_interior_chunk4_val_mv::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK>(
        p,
        v128_load(rgb_ptr as *const v128),
        luma_ptr,
        bg_luma_ptr,
        eca_res_ptr,
        mv,
        v_alpha,
        v_c,
        tol_v,
        v_rate_bg,
        v_rate_fg,
        v_motion_gate,
        heal_alpha,
        heal_gate_rgb,
        w_eca,
        w_rgb,
        v_rate_eval,
        v_eval_gate,
        acc_eval_err2,
        acc_eval_sig2,
        acc_eval_cnt_i,
        tile_acc,
    )
}

#[inline(always)]
unsafe fn pan_fused_border_chunk4_mv<const HAS_MASK: bool, const FILTER_MASK: bool>(
    p: usize,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eca_res_ptr: *mut f32,
    mut mv: v128,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
) -> (v128, v128) {
    let y_cur = v128_load(luma_ptr.add(p) as *const v128);
    let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);

    let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
    let diff = f32x4_sub(y_cur, proj);
    let res = f32x4_abs(diff);
    v128_store(eca_res_ptr.add(p) as *mut v128, res);

    let mut over = v_alpha;
    if HAS_MASK {
        if FILTER_MASK {
            over = f32x4_gt(f32x4_abs(f32x4_sub(y_cur, b_old)), tol_v);
            mv = v128_andnot(mv, over);
        }
        let m_diff = v128_and(diff, mv);
        let m_sig = v128_and(y_cur, mv);
        *acc_eval_err2 = f32x4_add(*acc_eval_err2, f32x4_mul(m_diff, m_diff));
        *acc_eval_sig2 = f32x4_add(*acc_eval_sig2, f32x4_mul(m_sig, m_sig));
        *acc_eval_cnt_i = i32x4_sub(*acc_eval_cnt_i, mv);
    }

    v128_store(bg_luma_ptr.add(p) as *mut v128, y_cur);
    (res, over)
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PanFusedHd40Ctx {
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_rate_bg_heal: v128,
    v_rate_fg_heal: v128,
    v_motion_gate: v128,
    heal_gate_i16: v128,
    w_eca: v128,
    w_rgb: v128,
    v_rate_eval: v128,
    v_rate_eval_heal: v128,
    v_eval_gate: v128,
    q16: v128,
    acc_eval_err2: v128,
    acc_eval_sig2: v128,
    acc_eval_cnt_i: v128,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
}

#[inline(always)]
unsafe fn pan_fused_hd40_row_phased_inner<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
    const PYR_STAGE: u8,
    const HOIST_RGB: bool,
>(
    ctx: &mut PanFusedHd40Ctx,
    gy: usize,
    ry_blk: usize,
    row_off0: usize,
    int_lo: usize,
    int_hi: usize,
    rgb_row_ptr: *const u16,
    acc_row: &mut [v128; 40],
    acc_rgb_row: *mut v128,
    pyr_a_row: &mut [v128; 40],
) {
    let luma_ptr = ctx.luma_ptr;
    let bg_luma_ptr = ctx.bg_luma_ptr;
    let eval_mask_ptr = ctx.eval_mask_ptr;
    let eca_res_ptr = ctx.eca_res_ptr;
    let pyr_eca_ptr = ctx.pyr_eca_ptr;
    let v_alpha = ctx.v_alpha;
    let v_c = ctx.v_c;
    let tol_v = ctx.tol_v;
    let v_rate_bg = ctx.v_rate_bg;
    let v_rate_fg = ctx.v_rate_fg;
    let v_rate_bg_heal = ctx.v_rate_bg_heal;
    let v_rate_fg_heal = ctx.v_rate_fg_heal;
    let v_motion_gate = ctx.v_motion_gate;
    let heal_gate_i16 = ctx.heal_gate_i16;
    let w_eca = ctx.w_eca;
    let w_rgb = ctx.w_rgb;
    let v_rate_eval = ctx.v_rate_eval;
    let v_rate_eval_heal = ctx.v_rate_eval_heal;
    let v_eval_gate = ctx.v_eval_gate;
    let q16 = ctx.q16;

    let mut loc_err2 = ctx.acc_eval_err2;
    let mut loc_sig2 = ctx.acc_eval_sig2;
    let loc_cnt_i = ctx.acc_eval_cnt_i;
    let zero_v = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let mut row_cnt_u8 = zero_v;
    let mut dummy_cnt = zero_v;
    let pyr_row_off = (gy * 4 + ry_blk) * 160;

    macro_rules! flush_hd40_pyr {
        ($gx:expr, $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {{
            let s01 = f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>($r0, $r1),
                i32x4_shuffle::<1, 3, 5, 7>($r0, $r1),
            );
            let s23 = f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>($r2, $r3),
                i32x4_shuffle::<1, 3, 5, 7>($r2, $r3),
            );
            let h4 = f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(s01, s23),
                i32x4_shuffle::<1, 3, 5, 7>(s01, s23),
            );
            if PYR_STAGE == 0 {
                pyr_a_row[$gx] = h4;
            } else if PYR_STAGE == 1 {
                pyr_a_row[$gx] = f32x4_add(pyr_a_row[$gx], h4);
            } else {
                let out = f32x4_mul(f32x4_add(pyr_a_row[$gx], h4), q16);
                v128_store(pyr_eca_ptr.add(pyr_row_off + $gx * 4) as *mut v128, out);
            }
        }};
    }

    macro_rules! run_border_hd40 {
        ($lo:expr, $hi:expr) => {
            for gx in $lo..$hi {
                let row_off = row_off0 + gx * 16;
                let (raw_m16, mv0, mv1, mv2, mv3) = if HAS_MASK {
                    let raw_m16 = v128_load(eval_mask_ptr.add(row_off) as *const v128);
                    let lo8 = i16x8_extend_low_i8x16(raw_m16);
                    let hi8 = i16x8_extend_high_i8x16(raw_m16);
                    (
                        raw_m16,
                        i32x4_extend_low_i16x8(lo8),
                        i32x4_extend_high_i16x8(lo8),
                        i32x4_extend_low_i16x8(hi8),
                        i32x4_extend_high_i16x8(hi8),
                    )
                } else {
                    (zero_v, zero_v, zero_v, zero_v, zero_v)
                };
                let (r0, o0) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off, luma_ptr, bg_luma_ptr, eca_res_ptr, mv0,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                );
                let (r1, o1) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off + 4, luma_ptr, bg_luma_ptr, eca_res_ptr, mv1,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                );
                let (r2, o2) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off + 8, luma_ptr, bg_luma_ptr, eca_res_ptr, mv2,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                );
                let (r3, o3) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off + 12, luma_ptr, bg_luma_ptr, eca_res_ptr, mv3,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                );
                if HAS_MASK {
                    let out_m16 = if FILTER_MASK {
                        let o16 = i8x16_narrow_i16x8(i16x8_narrow_i32x4(o0, o1), i16x8_narrow_i32x4(o2, o3));
                        let m = v128_andnot(raw_m16, o16);
                        v128_store(eval_mask_ptr.add(row_off) as *mut v128, m);
                        m
                    } else {
                        raw_m16
                    };
                    row_cnt_u8 = i8x16_add(row_cnt_u8, v128_and(out_m16, one_u8));
                }
                flush_hd40_pyr!(gx, r0, r1, r2, r3);
            }
        };
    }

    run_border_hd40!(0, int_lo);
    macro_rules! eval_hd40_chunk4 {
        ($p:expr, $st:expr, $mv:expr) => {{
            let y_cur = v128_load(luma_ptr.add($p) as *const v128);
            let b_old = v128_load(bg_luma_ptr.add($p) as *const v128);
            let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
            let diff = f32x4_sub(y_cur, proj);
            let res = f32x4_abs(diff);
            v128_store(eca_res_ptr.add($p) as *mut v128, res);
            let mut over = v_alpha;
            let mut mv = $mv;
            if HAS_MASK {
                if FILTER_MASK {
                    over = f32x4_gt(f32x4_abs(f32x4_sub(y_cur, b_old)), tol_v);
                    mv = v128_andnot(mv, over);
                }
                let m_diff = v128_and(diff, mv);
                let m_sig = v128_and(y_cur, mv);
                loc_err2 = f32x4_add(loc_err2, f32x4_mul(m_diff, m_diff));
                loc_sig2 = f32x4_add(loc_sig2, f32x4_mul(m_sig, m_sig));
            }
            let r_bg_s = v128_bitselect(v_rate_bg_heal, v_rate_bg, $st);
            let r_fg_s = v128_bitselect(v_rate_fg_heal, v_rate_fg, $st);
            let mut r_eff = v128_bitselect(r_bg_s, r_fg_s, f32x4_lt(res, v_motion_gate));
            if HAS_MASK && ALLOW_EVAL_HEAL {
                let r_ev_s = v128_bitselect(v_rate_eval_heal, v_rate_eval, $st);
                r_eff = v128_bitselect(
                    r_ev_s,
                    r_eff,
                    v128_and(mv, f32x4_lt(res, v_eval_gate)),
                );
            }
            v128_store(
                bg_luma_ptr.add($p) as *mut v128,
                f32x4_add(b_old, f32x4_mul(r_eff, f32x4_sub(y_cur, b_old))),
            );
            (res, over)
        }};
    }
    for gx in int_lo..int_hi {
        let row_off = row_off0 + gx * 16;
        let rgb_vp = (rgb_row_ptr.add(gx * 16)) as *const v128;
        let rgb_u0 = v128_load(rgb_vp);
        let rgb_u1 = v128_load(rgb_vp.add(1));
        let st_u0 = i16x8_lt(rgb_u0, heal_gate_i16);
        let st_u1 = i16x8_lt(rgb_u1, heal_gate_i16);
        let st0 = i32x4_extend_low_i16x8(st_u0);
        let st1 = i32x4_extend_high_i16x8(st_u0);
        let st2 = i32x4_extend_low_i16x8(st_u1);
        let st3 = i32x4_extend_high_i16x8(st_u1);
        let cur_rgb16 = if GATE_RGB {
            i16x8_add(v128_andnot(rgb_u0, st_u0), v128_andnot(rgb_u1, st_u1))
        } else {
            i16x8_add(rgb_u0, rgb_u1)
        };

        let (r0, r1, r2, r3) = if HAS_MASK {
            let raw_m16 = v128_load(eval_mask_ptr.add(row_off) as *const v128);
            let lo8 = i16x8_extend_low_i8x16(raw_m16);
            let hi8 = i16x8_extend_high_i8x16(raw_m16);
            let (r0, o0) = eval_hd40_chunk4!(row_off, st0, i32x4_extend_low_i16x8(lo8));
            let (r1, o1) = eval_hd40_chunk4!(row_off + 4, st1, i32x4_extend_high_i16x8(lo8));
            let (r2, o2) = eval_hd40_chunk4!(row_off + 8, st2, i32x4_extend_low_i16x8(hi8));
            let (r3, o3) = eval_hd40_chunk4!(row_off + 12, st3, i32x4_extend_high_i16x8(hi8));
            let out_m16 = if FILTER_MASK {
                let o16 = i8x16_narrow_i16x8(i16x8_narrow_i32x4(o0, o1), i16x8_narrow_i32x4(o2, o3));
                let m = v128_andnot(raw_m16, o16);
                v128_store(eval_mask_ptr.add(row_off) as *mut v128, m);
                m
            } else {
                raw_m16
            };
            row_cnt_u8 = i8x16_add(row_cnt_u8, v128_and(out_m16, one_u8));
            (r0, r1, r2, r3)
        } else {
            let (r0, _) = eval_hd40_chunk4!(row_off, st0, zero_v);
            let (r1, _) = eval_hd40_chunk4!(row_off + 4, st1, zero_v);
            let (r2, _) = eval_hd40_chunk4!(row_off + 8, st2, zero_v);
            let (r3, _) = eval_hd40_chunk4!(row_off + 12, st3, zero_v);
            (r0, r1, r2, r3)
        };
        let cur_r = f32x4_add(
            f32x4_add(v128_andnot(r0, st0), v128_andnot(r1, st1)),
            f32x4_add(v128_andnot(r2, st2), v128_andnot(r3, st3)),
        );
        if HOIST_RGB {
            acc_row[gx] = f32x4_add(acc_row[gx], cur_r);
            let ar = acc_rgb_row.add(gx);
            *ar = i16x8_add(*ar, cur_rgb16);
        } else {
            let cur_rgb = u32x4_extadd_pairwise_u16x8(cur_rgb16);
            let cur_tile = f32x4_add(
                f32x4_mul(cur_r, w_eca),
                f32x4_mul(f32x4_convert_u32x4(cur_rgb), w_rgb),
            );
            acc_row[gx] = f32x4_add(acc_row[gx], cur_tile);
        }
        flush_hd40_pyr!(gx, r0, r1, r2, r3);
    }
    run_border_hd40!(int_hi, 40);
    ctx.acc_eval_err2 = loc_err2;
    ctx.acc_eval_sig2 = loc_sig2;
    if HAS_MASK {
        let sum16 = u16x8_extadd_pairwise_u8x16(row_cnt_u8);
        ctx.acc_eval_cnt_i = i32x4_add(loc_cnt_i, u32x4_extadd_pairwise_u16x8(sum16));
    } else {
        ctx.acc_eval_cnt_i = loc_cnt_i;
    }
}

#[inline(never)]
unsafe fn pan_fused_hd40_row_phased<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
    const PYR_STAGE: u8,
    const HOIST_RGB: bool,
>(
    ctx: &mut PanFusedHd40Ctx,
    gy: usize,
    ry_blk: usize,
    row_off0: usize,
    int_lo: usize,
    int_hi: usize,
    rgb_row_ptr: *const u16,
    acc_row: &mut [v128; 40],
    acc_rgb_row: *mut v128,
    pyr_a_row: &mut [v128; 40],
) {
    pan_fused_hd40_row_phased_inner::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, PYR_STAGE, HOIST_RGB>(
        ctx, gy, ry_blk, row_off0, int_lo, int_hi, rgb_row_ptr, acc_row, acc_rgb_row, pyr_a_row,
    );
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn pan_hd40_band<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
>(
    ctx: &mut PanFusedHd40Ctx,
    gy: usize,
    _gy_valid: bool,
    int_lo: usize,
    int_hi: usize,
    band_rgb_ptr: *const u16,
    grid_out: *mut f32,
    w_eca: v128,
    w_rgb: v128,
) {
    let zero_v = i32x4_splat(0);
    let mut loc_ctx = *ctx;
    let mut acc_row = [zero_v; 40];
    let mut pyr_a_row_buf = [zero_v; 40];
    let pyr_a_row = &mut pyr_a_row_buf;
    let mut acc_rgb_row_buf = [zero_v; 40];
    let acc_rgb_row = &mut acc_rgb_row_buf;
    let acc_rgb_ptr = acc_rgb_row.as_mut_ptr();
    let y0 = gy * 16;
    for ry_blk in 0..4usize {
        let ry0 = ry_blk * 4;
        pan_fused_hd40_row_phased_inner::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, 0, true>(
            &mut loc_ctx, gy, ry_blk, (y0 + ry0) * 640, int_lo, int_hi,
            band_rgb_ptr.add(ry0 * 640), &mut acc_row, acc_rgb_ptr, pyr_a_row,
        );
        pan_fused_hd40_row_phased_inner::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, 1, true>(
            &mut loc_ctx, gy, ry_blk, (y0 + ry0 + 1) * 640, int_lo, int_hi,
            band_rgb_ptr.add((ry0 + 1) * 640), &mut acc_row, acc_rgb_ptr, pyr_a_row,
        );
        pan_fused_hd40_row_phased_inner::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, 1, true>(
            &mut loc_ctx, gy, ry_blk, (y0 + ry0 + 2) * 640, int_lo, int_hi,
            band_rgb_ptr.add((ry0 + 2) * 640), &mut acc_row, acc_rgb_ptr, pyr_a_row,
        );
        pan_fused_hd40_row_phased_inner::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, 2, true>(
            &mut loc_ctx, gy, ry_blk, (y0 + ry0 + 3) * 640, int_lo, int_hi,
            band_rgb_ptr.add((ry0 + 3) * 640), &mut acc_row, acc_rgb_ptr, pyr_a_row,
        );
    }
    ctx.acc_eval_err2 = loc_ctx.acc_eval_err2;
    ctx.acc_eval_sig2 = loc_ctx.acc_eval_sig2;
    ctx.acc_eval_cnt_i = loc_ctx.acc_eval_cnt_i;
    for gx in 0..40usize {
        *grid_out.add(gy * 40 + gx) = if gx >= int_lo && gx < int_hi {
            let sum_rgb = u32x4_extadd_pairwise_u16x8(acc_rgb_row[gx]);
            hsum_f32x4(f32x4_add(
                f32x4_mul(acc_row[gx], w_eca),
                f32x4_mul(f32x4_convert_u32x4(sum_rgb), w_rgb),
            ))
        } else {
            0.0
        };
    }
}

#[inline(always)]
unsafe fn pan_fused_hd40_row<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
>(
    ctx: &mut PanFusedHd40Ctx,
    gy: usize,
    ry_blk: usize,
    phase: u8,
    row_off0: usize,
    int_lo: usize,
    int_hi: usize,
    rgb_row_ptr: *const u16,
    acc_row: &mut [v128; 40],
    pyr_a_row: &mut [v128; 40],
) {
    match phase {
        0 => pan_fused_hd40_row_phased::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, 0, false>(
            ctx, gy, ry_blk, row_off0, int_lo, int_hi, rgb_row_ptr, acc_row, core::ptr::null_mut(), pyr_a_row,
        ),
        1 | 2 => pan_fused_hd40_row_phased::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, 1, false>(
            ctx, gy, ry_blk, row_off0, int_lo, int_hi, rgb_row_ptr, acc_row, core::ptr::null_mut(), pyr_a_row,
        ),
        _ => pan_fused_hd40_row_phased::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, 2, false>(
            ctx, gy, ry_blk, row_off0, int_lo, int_hi, rgb_row_ptr, acc_row, core::ptr::null_mut(), pyr_a_row,
        ),
    }
}

#[inline(never)]
unsafe fn pan_fused_grid_le40_row<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
    const FUSE_SCALE2: bool,
    const PYR_PHASE: u8,
    const READ_RGB_ROW: bool,
    const RGB_U16: bool,
>(
    gy: usize,
    ry: usize,
    ry_blk: usize,
    row_off0: usize,
    grid_w: usize,
    scale_px: usize,
    fuse_pyr: bool,
    int_lo: usize,
    int_hi: usize,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    rgb_row_ptr: *const u8,
    pyr_eca_ptr: *mut f32,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    heal_alpha: v128,
    heal_gate_rgb: v128,
    w_eca: v128,
    w_rgb: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    zero_v: v128,
    q4: v128,
    q16: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
    acc_row: &mut [v128; 40],
    pyr_a_row: &mut [v128; 40],
    pyr_b_row: *mut v128,
) {
    let mut loc_err2 = *acc_eval_err2;
    let mut loc_sig2 = *acc_eval_sig2;
    let mut loc_cnt_i = *acc_eval_cnt_i;

    macro_rules! flush_block_pyr {
        ($gx:expr, $s01:expr, $s23:expr, $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {
            if FUSE_SCALE2 {
                if PYR_PHASE < 4 {
                    flush_pyr_scale2_phase_simd128::<PYR_PHASE>(
                        gy,
                        $gx,
                        ry_blk,
                        q16,
                        $s01,
                        $s23,
                        &mut pyr_a_row[$gx],
                        pyr_eca_ptr,
                    );
                } else {
                    flush_pyr_scale2_ri_simd128(
                        gy,
                        $gx,
                        ry_blk,
                        ry & 3,
                        q16,
                        $s01,
                        $s23,
                        &mut pyr_a_row[$gx],
                        pyr_eca_ptr,
                    );
                }
            } else if fuse_pyr {
                let mut dummy_b = zero_v;
                let pb = if scale_px <= 1 && $gx < 20 {
                    &mut *pyr_b_row.add($gx)
                } else {
                    &mut dummy_b
                };
                flush_pyr_eca_reg_simd128(
                    gy,
                    $gx,
                    ry,
                    scale_px,
                    q4,
                    q16,
                    $r0,
                    $r1,
                    $r2,
                    $r3,
                    &mut pyr_a_row[$gx],
                    pb,
                    pyr_eca_ptr,
                );
            }
        };
    }

    macro_rules! load_m16_mvs {
        ($row_off:expr) => {
            if HAS_MASK {
                let raw_m16 = v128_load(eval_mask_ptr.add($row_off) as *const v128);
                let lo8 = i16x8_extend_low_i8x16(raw_m16);
                let hi8 = i16x8_extend_high_i8x16(raw_m16);
                (
                    raw_m16,
                    i32x4_extend_low_i16x8(lo8),
                    i32x4_extend_high_i16x8(lo8),
                    i32x4_extend_low_i16x8(hi8),
                    i32x4_extend_high_i16x8(hi8),
                )
            } else {
                (zero_v, zero_v, zero_v, zero_v, zero_v)
            }
        };
    }

    macro_rules! finish_filter_m16 {
        ($row_off:expr, $raw_m16:expr, $o0:expr, $o1:expr, $o2:expr, $o3:expr) => {
            if HAS_MASK && FILTER_MASK {
                let o16 = i8x16_narrow_i16x8(
                    i16x8_narrow_i32x4($o0, $o1),
                    i16x8_narrow_i32x4($o2, $o3),
                );
                v128_store(
                    eval_mask_ptr.add($row_off) as *mut v128,
                    v128_andnot($raw_m16, o16),
                );
            }
        };
    }

    macro_rules! run_border_range {
        ($lo:expr, $hi:expr) => {
            for gx in $lo..$hi {
                let row_off = row_off0 + gx * 16;
                let (raw_m16, mv0, mv1, mv2, mv3) = load_m16_mvs!(row_off);
                let (r0, o0) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off, luma_ptr, bg_luma_ptr, eca_res_ptr, mv0,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i,
                );
                let (r1, o1) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off + 4, luma_ptr, bg_luma_ptr, eca_res_ptr, mv1,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i,
                );
                let (r2, o2) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off + 8, luma_ptr, bg_luma_ptr, eca_res_ptr, mv2,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i,
                );
                let (r3, o3) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                    row_off + 12, luma_ptr, bg_luma_ptr, eca_res_ptr, mv3,
                    v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i,
                );
                finish_filter_m16!(row_off, raw_m16, o0, o1, o2, o3);
                let s01 = if FUSE_SCALE2 {
                    f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(r0, r1),
                        i32x4_shuffle::<1, 3, 5, 7>(r0, r1),
                    )
                } else {
                    zero_v
                };
                let s23 = if FUSE_SCALE2 {
                    f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(r2, r3),
                        i32x4_shuffle::<1, 3, 5, 7>(r2, r3),
                    )
                } else {
                    zero_v
                };
                flush_block_pyr!(gx, s01, s23, r0, r1, r2, r3);
            }
        };
    }

    run_border_range!(0, int_lo);
    macro_rules! eval_le40_interior {
        ($gx:expr, $row_off:expr, $acc:expr, $hm:expr, $aeh:expr, $fm:expr, $mv0:expr, $mv1:expr, $mv2:expr, $mv3:expr) => {{
            if READ_RGB_ROW && RGB_U16 {
                let rgb_vp = (rgb_row_ptr as *const v128).add($gx * 2);
                let rgb_u0 = v128_load(rgb_vp);
                let (r0, o0) = pan_fused_interior_chunk4_val_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off, f32x4_convert_u32x4(u32x4_extend_low_u16x8(rgb_u0)),
                    luma_ptr, bg_luma_ptr, eca_res_ptr, $mv0,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                let (r1, o1) = pan_fused_interior_chunk4_val_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off + 4, f32x4_convert_u32x4(u32x4_extend_high_u16x8(rgb_u0)),
                    luma_ptr, bg_luma_ptr, eca_res_ptr, $mv1,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                let rgb_u1 = v128_load(rgb_vp.add(1));
                let (r2, o2) = pan_fused_interior_chunk4_val_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off + 8, f32x4_convert_u32x4(u32x4_extend_low_u16x8(rgb_u1)),
                    luma_ptr, bg_luma_ptr, eca_res_ptr, $mv2,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                let (r3, o3) = pan_fused_interior_chunk4_val_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off + 12, f32x4_convert_u32x4(u32x4_extend_high_u16x8(rgb_u1)),
                    luma_ptr, bg_luma_ptr, eca_res_ptr, $mv3,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                (r0, r1, r2, r3, o0, o1, o2, o3)
            } else {
                let rgb_base = if READ_RGB_ROW {
                    (rgb_row_ptr as *const f32).add($gx * 16)
                } else {
                    eca_res_ptr.add($row_off) as *const f32
                };
                let (r0, o0) = pan_fused_interior_chunk4_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off, rgb_base, luma_ptr, bg_luma_ptr, eca_res_ptr, $mv0,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                let (r1, o1) = pan_fused_interior_chunk4_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off + 4, rgb_base.add(4), luma_ptr, bg_luma_ptr, eca_res_ptr, $mv1,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                let (r2, o2) = pan_fused_interior_chunk4_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off + 8, rgb_base.add(8), luma_ptr, bg_luma_ptr, eca_res_ptr, $mv2,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                let (r3, o3) = pan_fused_interior_chunk4_mv::<$hm, $aeh, GATE_RGB, $fm>(
                    $row_off + 12, rgb_base.add(12), luma_ptr, bg_luma_ptr, eca_res_ptr, $mv3,
                    v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    &mut loc_err2, &mut loc_sig2, &mut loc_cnt_i, &mut $acc,
                );
                (r0, r1, r2, r3, o0, o1, o2, o3)
            }
        }};
    }
    for gx in int_lo..int_hi {
        let row_off = row_off0 + gx * 16;
        let mut acc = acc_row[gx];
        let (r0, r1, r2, r3) = if HAS_MASK {
            let raw_m16 = v128_load(eval_mask_ptr.add(row_off) as *const v128);
            if !v128_any_true(raw_m16) {
                let (r0, r1, r2, r3, _, _, _, _) = eval_le40_interior!(
                    gx, row_off, acc, false, false, false, zero_v, zero_v, zero_v, zero_v
                );
                (r0, r1, r2, r3)
            } else {
                let lo8 = i16x8_extend_low_i8x16(raw_m16);
                let hi8 = i16x8_extend_high_i8x16(raw_m16);
                let (r0, r1, r2, r3, o0, o1, o2, o3) = eval_le40_interior!(
                    gx, row_off, acc, true, ALLOW_EVAL_HEAL, FILTER_MASK,
                    i32x4_extend_low_i16x8(lo8),
                    i32x4_extend_high_i16x8(lo8),
                    i32x4_extend_low_i16x8(hi8),
                    i32x4_extend_high_i16x8(hi8)
                );
                finish_filter_m16!(row_off, raw_m16, o0, o1, o2, o3);
                (r0, r1, r2, r3)
            }
        } else {
            let (r0, r1, r2, r3, _, _, _, _) = eval_le40_interior!(
                gx, row_off, acc, false, false, false, zero_v, zero_v, zero_v, zero_v
            );
            (r0, r1, r2, r3)
        };
        acc_row[gx] = acc;
        let s01 = if FUSE_SCALE2 {
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(r0, r1),
                i32x4_shuffle::<1, 3, 5, 7>(r0, r1),
            )
        } else {
            zero_v
        };
        let s23 = if FUSE_SCALE2 {
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(r2, r3),
                i32x4_shuffle::<1, 3, 5, 7>(r2, r3),
            )
        } else {
            zero_v
        };
        flush_block_pyr!(gx, s01, s23, r0, r1, r2, r3);
    }
    run_border_range!(int_hi, grid_w);
    *acc_eval_err2 = loc_err2;
    *acc_eval_sig2 = loc_sig2;
    *acc_eval_cnt_i = loc_cnt_i;
}

#[inline(always)]
unsafe fn pan_fused_tail_rows_simd128<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const FILTER_MASK: bool,
>(
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    mask_luma_tol: f32,
    frame_w: usize,
    frame_h: usize,
    grid_h: usize,
    ix0: i32,
    iy0: i32,
    sx_limit: i32,
    sy_limit: i32,
    alpha: f32,
    offset_c: f32,
    adapt_rate: f32,
    heal_alpha_s: f32,
    heal_gate_s: f32,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    heal_alpha: v128,
    heal_gate_rgb: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
    tail_eval_err2: &mut f32,
    tail_eval_sig2: &mut f32,
    tail_eval_cnt: &mut u32,
) {
    let lane_ofs = i32x4(0, 1, 2, 3);
    let zero_i32 = i32x4_splat(0);
    let sx_limit_v = i32x4_splat(sx_limit);
    for py in (grid_h * 16)..frame_h {
        let sy = (py as i32) + iy0;
        let sy_valid = sy >= 0 && sy < sy_limit;
        let row_off = py * frame_w;
        let mut px = 0usize;
        while px + 4 <= frame_w {
            let p = row_off + px;
            let rgb_sum = v128_load(eca_res_ptr.add(p) as *const v128);
            let y_cur = v128_load(luma_ptr.add(p) as *const v128);
            let b_old = v128_load(bg_luma_ptr.add(p) as *const v128);

            let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
            let diff = f32x4_sub(y_cur, proj);
            let res = f32x4_abs(diff);
            v128_store(eca_res_ptr.add(p) as *mut v128, res);

            let mask_vec = if HAS_MASK {
                let raw_m4 = v128_load32_zero(eval_mask_ptr.add(p) as *const u32);
                let mut mv = i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(raw_m4));
                if FILTER_MASK {
                    mv = filter_mask4_simd128(
                        eval_mask_ptr,
                        p,
                        y_cur,
                        b_old,
                        tol_v,
                        raw_m4,
                        mv,
                    );
                }
                let m_diff = v128_and(diff, mv);
                let m_sig = v128_and(y_cur, mv);
                *acc_eval_err2 = f32x4_add(*acc_eval_err2, f32x4_mul(m_diff, m_diff));
                *acc_eval_sig2 = f32x4_add(*acc_eval_sig2, f32x4_mul(m_sig, m_sig));
                *acc_eval_cnt_i = i32x4_sub(*acc_eval_cnt_i, mv);
                mv
            } else {
                v_alpha
            };

            let bg_out = if sy_valid {
                let is_bg = f32x4_lt(res, v_motion_gate);
                let rate = if ALLOW_EVAL_HEAL {
                    let is_eval_heal = v128_and(mask_vec, f32x4_lt(res, v_eval_gate));
                    v128_bitselect(v_rate_eval, v128_bitselect(v_rate_bg, v_rate_fg, is_bg), is_eval_heal)
                } else {
                    v128_bitselect(v_rate_bg, v_rate_fg, is_bg)
                };
                let b_eca = f32x4_add(b_old, f32x4_mul(rate, f32x4_sub(y_cur, b_old)));
                let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
                let bg_healed = f32x4_add(b_eca, f32x4_mul(heal_alpha, f32x4_sub(y_cur, b_eca)));
                let sx_v = i32x4_add(i32x4_splat((px as i32) + ix0), lane_ofs);
                let in_x = v128_and(i32x4_ge(sx_v, zero_i32), i32x4_lt(sx_v, sx_limit_v));
                v128_bitselect(v128_bitselect(bg_healed, b_eca, is_stat), y_cur, in_x)
            } else {
                y_cur
            };
            v128_store(bg_luma_ptr.add(p) as *mut v128, bg_out);
            px += 4;
        }
        while px < frame_w {
            let p = row_off + px;
            let rgb_sum = *eca_res_ptr.add(p);
            let y_c = *luma_ptr.add(p);
            let b_old = *bg_luma_ptr.add(p);
            let diff = y_c - (alpha * b_old + offset_c);
            let res = diff.abs();
            *eca_res_ptr.add(p) = res;
            if HAS_MASK && *eval_mask_ptr.add(p) != 0 {
                if FILTER_MASK && (y_c - b_old).abs() > mask_luma_tol {
                    *eval_mask_ptr.add(p) = 0;
                } else {
                    *tail_eval_err2 += diff * diff;
                    *tail_eval_sig2 += y_c * y_c;
                    *tail_eval_cnt += 1;
                }
            }
            let sx = (px as i32) + ix0;
            if sy_valid && sx >= 0 && sx < sx_limit {
                let rate = if res < 0.045 { adapt_rate } else { adapt_rate * 0.008 };
                let b_eca = b_old + rate * (y_c - b_old);
                *bg_luma_ptr.add(p) = if rgb_sum < heal_gate_s {
                    b_eca + heal_alpha_s * (y_c - b_eca)
                } else {
                    b_eca
                };
            } else {
                *bg_luma_ptr.add(p) = y_c;
            }
            px += 1;
        }
    }
}

/// Specialized monomorphized inner kernel for `eca_and_motion_grid_pan_fused_simd128`.
/// Iterates block-by-block `(gy, gx, ry)` with compact register-only accumulators and
/// compile-time folded mask, eval-heal, gate, and luma-filter flags.
#[inline(never)]
unsafe fn eca_and_motion_grid_pan_fused_impl<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
>(
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    mask_luma_tol: f32,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    alpha: f32,
    offset_c: f32,
    adapt_rate: f32,
    heal_alpha_s: f32,
    heal_gate_s: f32,
    ix0: i32,
    iy0: i32,
    gx_lo: usize,
    gx_hi: usize,
    gy_lo: usize,
    gy_hi: usize,
    sx_limit: i32,
    sy_limit: i32,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> f32 {
    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let tol_v = f32x4_splat(mask_luma_tol);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);
    let heal_alpha = f32x4_splat(heal_alpha_s);
    let heal_gate_rgb = f32x4_splat(heal_gate_s);
    let w_eca = f32x4_splat(0.75);
    let w_rgb = f32x4_splat(0.25 / 765.0);
    let v_rate_eval = f32x4_splat(0.98);
    let v_eval_gate = f32x4_splat(0.062);

    let mut acc_eval_err2 = f32x4_splat(0.0);
    let mut acc_eval_sig2 = f32x4_splat(0.0);
    let mut acc_eval_cnt_i = i32x4_splat(0);
    let mut tail_eval_err2 = 0.0f32;
    let mut tail_eval_sig2 = 0.0f32;
    let mut tail_eval_cnt = 0u32;
    let fuse_pyr = !pyr_eca_ptr.is_null() && scale_px <= 2 && frame_w == 320 * scale_px;
    let zero_v = f32x4_splat(0.0);
    let q4 = f32x4_splat(0.25);
    let q16 = f32x4_splat(1.0 / 16.0);

    if fuse_pyr && scale_px == 2 && grid_w == 40 && frame_w == 640 {
        for gy in 0..grid_h {
            let base_y = gy * 16;
            let gy_valid = gy >= gy_lo && gy < gy_hi;
            let (int_lo, int_hi) = if gy_valid {
                let a = gx_lo.min(40);
                (a, gx_hi.min(40).max(a))
            } else {
                (0, 0)
            };
            let mut acc_row = [zero_v; 40];
            let mut pyr_a_row = [zero_v; 40];
            for ry_blk in 0..4usize {
                for ri in 0..4usize {
                    let ry = ry_blk * 4 + ri;
                    pan_fused_grid_le40_row::<
                        HAS_MASK,
                        ALLOW_EVAL_HEAL,
                        GATE_RGB,
                        FILTER_MASK,
                        true,
                        4,
                        false,
                        false,
                    >(
                        gy,
                        ry,
                        ry_blk,
                        (base_y + ry) * frame_w,
                        40,
                        2,
                        true,
                        int_lo,
                        int_hi,
                        luma_ptr,
                        bg_luma_ptr,
                        eval_mask_ptr,
                        eca_res_ptr,
                        core::ptr::null(),
                        pyr_eca_ptr,
                        v_alpha,
                        v_c,
                        tol_v,
                        v_rate_bg,
                        v_rate_fg,
                        v_motion_gate,
                        heal_alpha,
                        heal_gate_rgb,
                        w_eca,
                        w_rgb,
                        v_rate_eval,
                        v_eval_gate,
                        zero_v,
                        q4,
                        q16,
                        &mut acc_eval_err2,
                        &mut acc_eval_sig2,
                        &mut acc_eval_cnt_i,
                        &mut acc_row,
                        &mut pyr_a_row,
                        core::ptr::null_mut(),
                    );
                }
            }

            for gx in 0..40usize {
                *grid_out.add(gy * grid_w + gx) = if gx >= int_lo && gx < int_hi {
                    hsum_f32x4(acc_row[gx])
                } else {
                    0.0
                };
            }
        }
    } else if grid_w <= 40 && frame_w == grid_w * 16 {
        for gy in 0..grid_h {
            let base_y = gy * 16;
            let gy_valid = gy >= gy_lo && gy < gy_hi;
            let (int_lo, int_hi) = if gy_valid {
                let a = gx_lo.min(grid_w);
                (a, gx_hi.min(grid_w).max(a))
            } else {
                (0, 0)
            };
            let mut acc_row = [zero_v; 40];
            let mut pyr_a_row = [zero_v; 40];
            let mut pyr_b_row = [zero_v; 20];

            for ry in 0..16usize {
                pan_fused_grid_le40_row::<
                    HAS_MASK,
                    ALLOW_EVAL_HEAL,
                    GATE_RGB,
                    FILTER_MASK,
                    false,
                    0,
                    false,
                    false,
                >(
                    gy,
                    ry,
                    ry >> 2,
                    (base_y + ry) * frame_w,
                    grid_w,
                    scale_px,
                    fuse_pyr,
                    int_lo,
                    int_hi,
                    luma_ptr,
                    bg_luma_ptr,
                    eval_mask_ptr,
                    eca_res_ptr,
                    core::ptr::null(),
                    pyr_eca_ptr,
                    v_alpha,
                    v_c,
                    tol_v,
                    v_rate_bg,
                    v_rate_fg,
                    v_motion_gate,
                    heal_alpha,
                    heal_gate_rgb,
                    w_eca,
                    w_rgb,
                    v_rate_eval,
                    v_eval_gate,
                    zero_v,
                    q4,
                    q16,
                    &mut acc_eval_err2,
                    &mut acc_eval_sig2,
                    &mut acc_eval_cnt_i,
                    &mut acc_row,
                    &mut pyr_a_row,
                    pyr_b_row.as_mut_ptr(),
                );
            }

            for gx in 0..grid_w {
                *grid_out.add(gy * grid_w + gx) = if gx >= int_lo && gx < int_hi {
                    hsum_f32x4(acc_row[gx])
                } else {
                    0.0
                };
            }
        }
    } else {
        for gy in 0..grid_h {
            let base_y = gy * 16;
            let gy_valid = gy >= gy_lo && gy < gy_hi;
            for gx in 0..grid_w {
                let block_off = base_y * frame_w + gx * 16;
                let mut pyr_a = zero_v;
                let mut pyr_b = zero_v;
                if gy_valid && gx >= gx_lo && gx < gx_hi {
                    let mut acc = zero_v;
                    for ry in 0..16usize {
                        let row_off = block_off + ry * frame_w;
                        let r0 = pan_fused_interior_chunk4::<
                            HAS_MASK,
                            ALLOW_EVAL_HEAL,
                            GATE_RGB,
                            FILTER_MASK,
                        >(
                            row_off,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            v_rate_bg,
                            v_rate_fg,
                            v_motion_gate,
                            heal_alpha,
                            heal_gate_rgb,
                            w_eca,
                            w_rgb,
                            v_rate_eval,
                            v_eval_gate,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                            &mut acc,
                        );
                        let r1 = pan_fused_interior_chunk4::<
                            HAS_MASK,
                            ALLOW_EVAL_HEAL,
                            GATE_RGB,
                            FILTER_MASK,
                        >(
                            row_off + 4,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            v_rate_bg,
                            v_rate_fg,
                            v_motion_gate,
                            heal_alpha,
                            heal_gate_rgb,
                            w_eca,
                            w_rgb,
                            v_rate_eval,
                            v_eval_gate,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                            &mut acc,
                        );
                        let r2 = pan_fused_interior_chunk4::<
                            HAS_MASK,
                            ALLOW_EVAL_HEAL,
                            GATE_RGB,
                            FILTER_MASK,
                        >(
                            row_off + 8,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            v_rate_bg,
                            v_rate_fg,
                            v_motion_gate,
                            heal_alpha,
                            heal_gate_rgb,
                            w_eca,
                            w_rgb,
                            v_rate_eval,
                            v_eval_gate,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                            &mut acc,
                        );
                        let r3 = pan_fused_interior_chunk4::<
                            HAS_MASK,
                            ALLOW_EVAL_HEAL,
                            GATE_RGB,
                            FILTER_MASK,
                        >(
                            row_off + 12,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            v_rate_bg,
                            v_rate_fg,
                            v_motion_gate,
                            heal_alpha,
                            heal_gate_rgb,
                            w_eca,
                            w_rgb,
                            v_rate_eval,
                            v_eval_gate,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                            &mut acc,
                        );
                        if fuse_pyr {
                            flush_pyr_eca_reg_simd128(
                                gy,
                                gx,
                                ry,
                                scale_px,
                                q4,
                                q16,
                                r0,
                                r1,
                                r2,
                                r3,
                                &mut pyr_a,
                                &mut pyr_b,
                                pyr_eca_ptr,
                            );
                        }
                    }
                    *grid_out.add(gy * grid_w + gx) = hsum_f32x4(acc);
                } else {
                    *grid_out.add(gy * grid_w + gx) = 0.0;
                    for ry in 0..16usize {
                        let row_off = block_off + ry * frame_w;
                        let r0 = pan_fused_border_chunk4::<HAS_MASK, FILTER_MASK>(
                            row_off,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                        );
                        let r1 = pan_fused_border_chunk4::<HAS_MASK, FILTER_MASK>(
                            row_off + 4,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                        );
                        let r2 = pan_fused_border_chunk4::<HAS_MASK, FILTER_MASK>(
                            row_off + 8,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                        );
                        let r3 = pan_fused_border_chunk4::<HAS_MASK, FILTER_MASK>(
                            row_off + 12,
                            luma_ptr,
                            bg_luma_ptr,
                            eval_mask_ptr,
                            eca_res_ptr,
                            v_alpha,
                            v_c,
                            tol_v,
                            &mut acc_eval_err2,
                            &mut acc_eval_sig2,
                            &mut acc_eval_cnt_i,
                        );
                        if fuse_pyr {
                            flush_pyr_eca_reg_simd128(
                                gy,
                                gx,
                                ry,
                                scale_px,
                                q4,
                                q16,
                                r0,
                                r1,
                                r2,
                                r3,
                                &mut pyr_a,
                                &mut pyr_b,
                                pyr_eca_ptr,
                            );
                        }
                    }
                }
            }
            if grid_w * 16 < frame_w {
                for ry in 0..16usize {
                    let y = base_y + ry;
                    let y_row = y * frame_w;
                    let sy = (y as i32) + iy0;
                    let sy_valid = sy >= 0 && sy < sy_limit;
                    for px in (grid_w * 16)..frame_w {
                        let p = y_row + px;
                        let rgb_sum = *eca_res_ptr.add(p);
                        let y_c = *luma_ptr.add(p);
                        let b_old = *bg_luma_ptr.add(p);
                        let diff = y_c - (alpha * b_old + offset_c);
                        let res = diff.abs();
                        *eca_res_ptr.add(p) = res;
                        if HAS_MASK && *eval_mask_ptr.add(p) != 0 {
                            if FILTER_MASK && (y_c - b_old).abs() > mask_luma_tol {
                                *eval_mask_ptr.add(p) = 0;
                            } else {
                                tail_eval_err2 += diff * diff;
                                tail_eval_sig2 += y_c * y_c;
                                tail_eval_cnt += 1;
                            }
                        }
                        let sx = (px as i32) + ix0;
                        if sy_valid && sx >= 0 && sx < sx_limit {
                            let eval_on = HAS_MASK && ALLOW_EVAL_HEAL && *eval_mask_ptr.add(p) != 0;
                            let rate = if eval_on && res < 0.062 {
                                0.98
                            } else if res < 0.045 {
                                adapt_rate
                            } else {
                                adapt_rate * 0.008
                            };
                            let b_eca = b_old + rate * (y_c - b_old);
                            *bg_luma_ptr.add(p) = if rgb_sum < heal_gate_s {
                                b_eca + heal_alpha_s * (y_c - b_eca)
                            } else {
                                b_eca
                            };
                        } else {
                            *bg_luma_ptr.add(p) = y_c;
                        }
                    }
                }
            }
        }
    }

    pan_fused_tail_rows_simd128::<HAS_MASK, ALLOW_EVAL_HEAL, FILTER_MASK>(
        luma_ptr,
        bg_luma_ptr,
        eval_mask_ptr,
        eca_res_ptr,
        mask_luma_tol,
        frame_w,
        frame_h,
        grid_h,
        ix0,
        iy0,
        sx_limit,
        sy_limit,
        alpha,
        offset_c,
        adapt_rate,
        heal_alpha_s,
        heal_gate_s,
        v_alpha,
        v_c,
        tol_v,
        v_rate_bg,
        v_rate_fg,
        v_motion_gate,
        heal_alpha,
        heal_gate_rgb,
        v_rate_eval,
        v_eval_gate,
        &mut acc_eval_err2,
        &mut acc_eval_sig2,
        &mut acc_eval_cnt_i,
        &mut tail_eval_err2,
        &mut tail_eval_sig2,
        &mut tail_eval_cnt,
    );

    let err2 = hsum_f32x4(acc_eval_err2) + tail_eval_err2;
    let sig2 = hsum_f32x4(acc_eval_sig2) + tail_eval_sig2;
    let eval_count = (hsum_i32x4(acc_eval_cnt_i) as u32).saturating_add(tail_eval_cnt);
    if !out_err2.is_null() {
        *out_err2 = err2;
    }
    if !out_sig2.is_null() {
        *out_sig2 = sig2;
    }
    if !out_eval_count.is_null() {
        *out_eval_count = eval_count;
    }

    if eval_count >= 1024 && sig2 > 1e-3 {
        err2 / sig2
    } else {
        -1.0
    }
}

/// Fused pan- and exposure-compensated Rank-2 ECA projection, 16x16 motion-grid accumulation,
/// and stationary background healing. Expects precomputed unmasked 3-channel RGB L1 deltas
/// (`|c - p|_RGB` in `0.0..=765.0`) in `eca_res_ptr` on entry, overwriting `eca_res_ptr` in-place
/// with ECA orthogonal residuals in monotonic scanline order.
#[inline(never)]
pub unsafe fn eca_and_motion_grid_pan_fused_simd128(
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    mask_luma_tol: f32,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    adapt_rate: f32,
    pan_dx: i32,
    pan_dy: i32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    frac_mag: f32,
    exp_gain: f32,
    exp_off: f32,
    exp_regime: bool,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> f32 {
    let len = frame_w * frame_h;
    let (exp_active, _, _) = exposure_q15_params(exp_gain, exp_off);
    let pan_moved = pan_dx != 0 || pan_dy != 0 || frac_mag >= 0.04;
    let eff_exp_regime = exp_regime || exp_active;
    let subpx = frac_mag >= 0.04;
    let (ix0, iy0) = if subpx {
        let (ix, iy, _, _, _, _) = bilinear_weights_u16(pan_dx_f, pan_dy_f);
        (ix, iy)
    } else {
        (-pan_dx, -pan_dy)
    };
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let sx_limit = if subpx { w_i - 1 } else { w_i };
    let sy_limit = if subpx { h_i - 1 } else { h_i };
    let border_pad = if !subpx && pan_dx == 0 && pan_dy == 0 {
        0
    } else {
        2
    };
    let is_gx_valid = |base_x: i32, base_y: i32| -> bool {
        if subpx {
            base_x + ix0 >= 2
                && base_x + 17 + ix0 <= w_i - 2
                && base_y + iy0 >= 2
                && base_y + 16 + iy0 <= h_i - 2
        } else {
            base_x - pan_dx >= border_pad
                && base_x + 16 - pan_dx <= w_i - border_pad
                && base_y - pan_dy >= border_pad
                && base_y + 16 - pan_dy <= h_i - border_pad
        }
    };

    let heal_alpha_s = if pan_moved && eff_exp_regime {
        0.86f32
    } else if eff_exp_regime {
        0.84f32
    } else if pan_moved {
        0.78f32
    } else {
        0.65f32
    };
    let scalar_gate = if subpx {
        if eff_exp_regime { 0.026f32 } else { 0.014f32 }
    } else if eff_exp_regime {
        0.023f32
    } else {
        0.010f32
    };
    let heal_gate_s = scalar_gate * 765.0;
    let heal_gate_rgb = f32x4_splat(heal_gate_s);
    let gate_rgb_on_stat = subpx || eff_exp_regime;
    let w_rgb = f32x4_splat(0.25 / 765.0);

    let fit_mask_ptr = if !eval_mask_ptr.is_null() && mask_luma_tol > 0.0 && mask_luma_tol <= 0.045 {
        core::ptr::null()
    } else {
        eval_mask_ptr as *const u8
    };
    let Some((alpha, offset_c)) = eca_fit_affine_params_simd128(
        luma_ptr,
        bg_luma_ptr,
        fit_mask_ptr,
        core::ptr::null(),
        core::ptr::null(),
        core::ptr::null(),
        len,
    ) else {
        if !eval_mask_ptr.is_null() && mask_luma_tol > 0.0 {
            filter_bg_eval_mask_luma_simd128(eval_mask_ptr, luma_ptr, bg_luma_ptr, len, mask_luma_tol);
        }
        let zero = f32x4_splat(0.0);
        for gy in 0..grid_h {
            let base_y = gy * 16;
            let mut row_acc = [zero; 40];
            let mut gx_valid = [false; 40];
            for gx in 0..grid_w.min(40) {
                gx_valid[gx] = is_gx_valid((gx * 16) as i32, base_y as i32);
            }
            for gx in 40..grid_w {
                *grid_out.add(gy * grid_w + gx) = 0.0;
            }
            for ry in 0..16usize {
                let y = base_y + ry;
                let y_row = y * frame_w;
                for gx in 0..grid_w {
                    let row_off = y_row + gx * 16;
                    let valid = if gx < 40 {
                        gx_valid[gx]
                    } else {
                        is_gx_valid((gx * 16) as i32, base_y as i32)
                    };
                    if valid {
                        let mut acc = if gx < 40 { row_acc[gx] } else { zero };
                        for cx_vec in 0..4usize {
                            let p = row_off + cx_vec * 4;
                            let rgb_sum = v128_load(eca_res_ptr.add(p) as *const v128);
                            let y_cur = v128_load(luma_ptr.add(p) as *const v128);
                            v128_store(eca_res_ptr.add(p) as *mut v128, zero);
                            v128_store(bg_luma_ptr.add(p) as *mut v128, y_cur);
                            let eff_rgb = if gate_rgb_on_stat {
                                let is_stat = f32x4_lt(rgb_sum, heal_gate_rgb);
                                v128_andnot(rgb_sum, is_stat)
                            } else {
                                rgb_sum
                            };
                            acc = f32x4_add(acc, f32x4_mul(eff_rgb, w_rgb));
                        }
                        if gx < 40 {
                            row_acc[gx] = acc;
                        } else {
                            *grid_out.add(gy * grid_w + gx) += hsum_f32x4(acc);
                        }
                    } else {
                        for cx_vec in 0..4usize {
                            let p = row_off + cx_vec * 4;
                            let y_cur = v128_load(luma_ptr.add(p) as *const v128);
                            v128_store(eca_res_ptr.add(p) as *mut v128, zero);
                            v128_store(bg_luma_ptr.add(p) as *mut v128, y_cur);
                        }
                    }
                }
                for px in (grid_w * 16)..frame_w {
                    let p = y_row + px;
                    *eca_res_ptr.add(p) = 0.0;
                    *bg_luma_ptr.add(p) = *luma_ptr.add(p);
                }
            }
            for gx in 0..grid_w.min(40) {
                *grid_out.add(gy * grid_w + gx) = if gx_valid[gx] {
                    hsum_f32x4(row_acc[gx])
                } else {
                    0.0
                };
            }
        }
        let mut k = grid_h * 16 * frame_w;
        while k + 4 <= len {
            let y_cur = v128_load(luma_ptr.add(k) as *const v128);
            v128_store(bg_luma_ptr.add(k) as *mut v128, y_cur);
            v128_store(eca_res_ptr.add(k) as *mut v128, zero);
            k += 4;
        }
        while k < len {
            *bg_luma_ptr.add(k) = *luma_ptr.add(k);
            *eca_res_ptr.add(k) = 0.0;
            k += 1;
        }
        if !pyr_eca_ptr.is_null() {
            core::ptr::write_bytes(pyr_eca_ptr, 0, 160 * 90);
        }
        if !out_err2.is_null() {
            *out_err2 = 0.0;
        }
        if !out_sig2.is_null() {
            *out_sig2 = 0.0;
        }
        if !out_eval_count.is_null() {
            *out_eval_count = 0;
        }
        return -1.0;
    };

    let (min_bx, max_bx, min_by, max_by) = if subpx {
        (2 - ix0, w_i - 19 - ix0, 2 - iy0, h_i - 18 - iy0)
    } else {
        (
            border_pad + pan_dx,
            w_i - 16 - border_pad + pan_dx,
            border_pad + pan_dy,
            h_i - 16 - border_pad + pan_dy,
        )
    };
    let mut gx_lo = 0usize;
    while gx_lo < grid_w && ((gx_lo * 16) as i32) < min_bx {
        gx_lo += 1;
    }
    let mut gx_hi = gx_lo;
    while gx_hi < grid_w && ((gx_hi * 16) as i32) <= max_bx {
        gx_hi += 1;
    }
    let mut gy_lo = 0usize;
    while gy_lo < grid_h && ((gy_lo * 16) as i32) < min_by {
        gy_lo += 1;
    }
    let mut gy_hi = gy_lo;
    while gy_hi < grid_h && ((gy_hi * 16) as i32) <= max_by {
        gy_hi += 1;
    }

    let has_mask = !eval_mask_ptr.is_null();
    let allow_eval_heal = has_mask && !eff_exp_regime;
    let filter_mask = has_mask && mask_luma_tol > 0.0;
    macro_rules! call_pan_fused {
        ($hm:expr, $eh:expr, $gr:expr, $fm:expr) => {
            eca_and_motion_grid_pan_fused_impl::<$hm, $eh, $gr, $fm>(
                luma_ptr,
                bg_luma_ptr,
                eval_mask_ptr,
                mask_luma_tol,
                eca_res_ptr,
                pyr_eca_ptr,
                grid_out,
                frame_w,
                frame_h,
                grid_w,
                grid_h,
                scale_px,
                alpha,
                offset_c,
                adapt_rate,
                heal_alpha_s,
                heal_gate_s,
                ix0,
                iy0,
                gx_lo,
                gx_hi,
                gy_lo,
                gy_hi,
                sx_limit,
                sy_limit,
                out_err2,
                out_sig2,
                out_eval_count,
            )
        };
    }
    match (has_mask, allow_eval_heal, gate_rgb_on_stat, filter_mask) {
        (true, true, true, true) => call_pan_fused!(true, true, true, true),
        (true, true, true, false) => call_pan_fused!(true, true, true, false),
        (true, true, false, true) => call_pan_fused!(true, true, false, true),
        (true, true, false, false) => call_pan_fused!(true, true, false, false),
        (true, false, true, true) => call_pan_fused!(true, false, true, true),
        (true, false, true, false) => call_pan_fused!(true, false, true, false),
        (true, false, false, true) => call_pan_fused!(true, false, false, true),
        (true, false, false, false) => call_pan_fused!(true, false, false, false),
        (false, _, true, _) => call_pan_fused!(false, false, true, false),
        (false, _, false, _) => call_pan_fused!(false, false, false, false),
    }
}

#[inline(always)]
unsafe fn subpx_int_row_inner<
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
    const REUSE_H0: bool,
>(
    s_ctx: &SubpxRowCtx,
    eval_buf: *mut v128,
    rgb_u0_buf: *mut v128,
    rgb_u1_buf: *mut v128,
    y: usize,
    int_lo: usize,
    int_hi: usize,
) {
    let row_off0 = y * 640;
    let sy = (y as i32) + s_ctx.iy0;
    let row_p0 = (sy as usize) * 640;
    let eval_mask_ptr = s_ctx.eval_mask_ptr;
    let stat_streak_ptr = s_ctx.stat_streak_ptr;
    let h_cache_ptr = s_ctx.h_cache_ptr;

    if int_lo > 0 {
        let left_end = int_lo * 16;
        if s_ctx.x_lo == 1 {
            *stat_streak_ptr.add(row_off0) = 0;
            *eval_mask_ptr.add(row_off0) = 0;
        } else if s_ctx.x_lo > 0 {
            core::slice::from_raw_parts_mut(stat_streak_ptr.add(row_off0), s_ctx.x_lo).fill(0);
            core::slice::from_raw_parts_mut(eval_mask_ptr.add(row_off0), s_ctx.x_lo).fill(0);
        }
        let left_quad_lim = left_end.min(s_ctx.quad_lim);
        let next_left = if s_ctx.x_lo + 4 <= left_quad_lim {
            subpx_eval_and_hyst_quads_2d::<EXP_ACTIVE, EFF_EXP, false, true>(
                s_ctx.curr_rgba, s_ctx.prev_rgba, stat_streak_ptr, eval_mask_ptr,
                core::ptr::null_mut(), row_off0, row_p0, s_ctx.row_bytes, s_ctx.ix0,
                s_ctx.x_lo, left_quad_lim, s_ctx.w00, s_ctx.w10, s_ctx.w01, s_ctx.w11,
                s_ctx.round_v, s_ctx.dg_q15_v, s_ctx.off_i16_v,
                s_ctx.hard_u8, s_ctx.mid_u8, s_ctx.soft_u8, s_ctx.span_soft_u8, s_ctx.span_reset_u8,
            )
        } else {
            s_ctx.x_lo
        };
        if next_left < left_end {
            subpx_scalar_tail::<EXP_ACTIVE, EFF_EXP, false, true>(
                s_ctx.curr_rgba, s_ctx.prev_rgba, stat_streak_ptr, eval_mask_ptr,
                core::ptr::null_mut(), row_off0, row_p0, s_ctx.row_bytes, 640, s_ctx.w_i,
                s_ctx.ix0, next_left, left_end, s_ctx.w00, s_ctx.w10, s_ctx.w01, s_ctx.w11,
                s_ctx.dg_q15, s_ctx.off_i16, s_ctx.hard_u8, s_ctx.mid_u8, s_ctx.soft_u8,
                s_ctx.span_soft_u8, s_ctx.span_reset_u8,
            );
        }
    }

    let w00_u8 = s_ctx.w00_u8;
    let w10_u8 = s_ctx.w10_u8;
    let w01_u8 = s_ctx.w01_u8;
    let w11_u8 = s_ctx.w11_u8;
    let dg_q15_v = s_ctx.dg_q15_v;
    let off_i16_v = s_ctx.off_i16_v;
    let c_row_ptr = s_ctx.curr_rgba.add(row_off0 * 4);

    let sx_start = ((int_lo * 16) as i32 + s_ctx.ix0) as usize;
    let p0_ptr = s_ctx.prev_rgba.add((row_p0 + sx_start) * 4);
    let p1_ptr = p0_ptr.add(s_ctx.row_bytes);

    let (mut r0_r, mut r0_g, mut r0_b) = if REUSE_H0 {
        (v128_load(h_cache_ptr), v128_load(h_cache_ptr.add(1)), v128_load(h_cache_ptr.add(2)))
    } else {
        deinterleave_rgb16_simd128(p0_ptr)
    };
    let (mut r1_r, mut r1_g, mut r1_b) = deinterleave_rgb16_simd128(p1_ptr);

    macro_rules! eval_int_blk {
        ($gx:expr, $s0_r:expr, $s1_r:expr, $after_r:block,
         $s0_g:expr, $s1_g:expr, $after_g:block,
         $s0_b:expr, $s1_b:expr, $after_b:block) => {{
            let x_blk = $gx * 16;
            let row_off = row_off0 + x_blk;
            let round_v = u16x8_splat(128);
            let (c_r, c_g, c_b) = deinterleave_rgb16_simd128(c_row_ptr.add(x_blk * 4));
            let s0_r = $s0_r;
            let s1_r = $s1_r;
            let (abs_r, sp_r, p_raw_r) = eval_chan_16_planar::<EXP_ACTIVE, 0>(
                r0_r, s0_r, r1_r, s1_r, c_r, w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            );
            $after_r
            let mut rgb_lo = u16x8_extend_low_u8x16(abs_r);
            let mut rgb_hi = u16x8_extend_high_u8x16(abs_r);
            let mut max_d16 = abs_r;
            let mut max_sp16 = sp_r;

            let s0_g = $s0_g;
            let s1_g = $s1_g;
            let (abs_g, sp_g, p_raw_g) = eval_chan_16_planar::<EXP_ACTIVE, 0>(
                r0_g, s0_g, r1_g, s1_g, c_g, w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            );
            $after_g
            rgb_lo = u16x8_add(rgb_lo, u16x8_extend_low_u8x16(abs_g));
            rgb_hi = u16x8_add(rgb_hi, u16x8_extend_high_u8x16(abs_g));
            max_d16 = u8x16_max(max_d16, abs_g);
            max_sp16 = u8x16_max(max_sp16, sp_g);

            let s0_b = $s0_b;
            let s1_b = $s1_b;
            let (mut abs_b, sp_b, p_raw_b) = eval_chan_16_planar::<EXP_ACTIVE, 0>(
                r0_b, s0_b, r1_b, s1_b, c_b, w00_u8, w10_u8, w01_u8, w11_u8, round_v, dg_q15_v, off_i16_v,
            );
            $after_b
            rgb_lo = u16x8_add(rgb_lo, u16x8_extend_low_u8x16(abs_b));
            rgb_hi = u16x8_add(rgb_hi, u16x8_extend_high_u8x16(abs_b));
            max_sp16 = u8x16_max(max_sp16, sp_b);

            v128_store(rgb_u0_buf.add($gx) as *mut v128, rgb_lo);
            v128_store(rgb_u1_buf.add($gx) as *mut v128, rgb_hi);

            let wc16 = if EFF_EXP {
                let b_hi = u8x16_ge(u8x16_max(p_raw_b, c_b), u8x16_splat(250));
                let rg_low = u8x16_lt(u8x16_min(p_raw_r, p_raw_g), u8x16_splat(230));
                abs_b = v128_andnot(abs_b, v128_and(b_hi, rg_low));
                let pr_wc = v128_and(u8x16_ge(p_raw_r, u8x16_splat(251)), u8x16_ge(p_raw_g, u8x16_splat(251)));
                let c_wc = v128_and(u8x16_ge(c_r, u8x16_splat(253)), u8x16_ge(c_g, u8x16_splat(253)));
                v128_or(pr_wc, c_wc)
            } else {
                i32x4_splat(0)
            };
            max_d16 = u8x16_max(max_d16, abs_b);
            let eval16 = hysteresis_pan_16_compute_simd128(
                max_d16, Some((max_sp16, s_ctx.tol_span_soft, s_ctx.tol_span_reset)), wc16,
                s_ctx.tol_hard, s_ctx.tol_mid, s_ctx.tol_soft,
                u8x16_splat(1), u8x16_splat(2), u8x16_splat(6), u8x16_splat(14), u8x16_splat(12), u8x16_splat(20),
                EFF_EXP, stat_streak_ptr.add(row_off),
            );
            v128_store(eval_buf.add($gx) as *mut v128, eval16);
        }};
    }

    let n_int = int_hi - int_lo;
    let last_blk = n_int - 1;
    for blk in 0..last_blk {
        let gx = int_lo + blk;
        let px_off = blk * 64;
        let hc = h_cache_ptr.add(blk * 3);
        v128_store(hc, r1_r);
        v128_store(hc.add(1), r1_g);
        v128_store(hc.add(2), r1_b);
        let (nr1_r, nr1_g, nr1_b) = deinterleave_rgb16_simd128(p1_ptr.add(px_off + 64));
        if REUSE_H0 {
            let hc_n = h_cache_ptr.add((blk + 1) * 3);
            let nr0_r = v128_load(hc_n);
            let nr0_g;
            let nr0_b;
            eval_int_blk!(
                gx,
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_r, nr0_r),
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_r, nr1_r),
                { r0_r = nr0_r; r1_r = nr1_r; },
                { nr0_g = v128_load(hc_n.add(1)); i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_g, nr0_g) },
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_g, nr1_g),
                { r0_g = nr0_g; r1_g = nr1_g; },
                { nr0_b = v128_load(hc_n.add(2)); i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_b, nr0_b) },
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_b, nr1_b),
                { r0_b = nr0_b; r1_b = nr1_b; }
            );
        } else {
            let (nr0_r, nr0_g, nr0_b) = deinterleave_rgb16_simd128(p0_ptr.add(px_off + 64));
            eval_int_blk!(
                gx,
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_r, nr0_r),
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_r, nr1_r),
                { r0_r = nr0_r; r1_r = nr1_r; },
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_g, nr0_g),
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_g, nr1_g),
                { r0_g = nr0_g; r1_g = nr1_g; },
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_b, nr0_b),
                i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_b, nr1_b),
                { r0_b = nr0_b; r1_b = nr1_b; }
            );
        }
    }

    let last_px_off = last_blk * 64;
    let last_gx = int_lo + last_blk;
    let p0_tail = if REUSE_H0 {
        v128_load32_zero(h_cache_ptr.add(n_int * 3) as *const u32)
    } else {
        v128_load32_zero(p0_ptr.add(last_px_off + 64) as *const u32)
    };
    let p1_tail = v128_load32_zero(p1_ptr.add(last_px_off + 64) as *const u32);
    let hc = h_cache_ptr.add(last_blk * 3);
    v128_store(hc, r1_r);
    v128_store(hc.add(1), r1_g);
    v128_store(hc.add(2), r1_b);
    v128_store32_lane::<0>(p1_tail, h_cache_ptr.add(n_int * 3) as *mut u32);
    eval_int_blk!(
        last_gx,
        i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r0_r, p0_tail),
        i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(r1_r, p1_tail),
        {},
        i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17>(r0_g, p0_tail),
        i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17>(r1_g, p1_tail),
        {},
        i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18>(r0_b, p0_tail),
        i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18>(r1_b, p1_tail),
        {}
    );

    if int_hi < 40 {
        let right_start = int_hi * 16;
        let next_right = if right_start + 4 <= s_ctx.quad_lim {
            subpx_eval_and_hyst_quads_2d::<EXP_ACTIVE, EFF_EXP, false, true>(
                s_ctx.curr_rgba, s_ctx.prev_rgba, stat_streak_ptr, eval_mask_ptr,
                core::ptr::null_mut(), row_off0, row_p0, s_ctx.row_bytes, s_ctx.ix0,
                right_start, s_ctx.quad_lim, s_ctx.w00, s_ctx.w10, s_ctx.w01, s_ctx.w11,
                s_ctx.round_v, s_ctx.dg_q15_v, s_ctx.off_i16_v,
                s_ctx.hard_u8, s_ctx.mid_u8, s_ctx.soft_u8, s_ctx.span_soft_u8, s_ctx.span_reset_u8,
            )
        } else {
            right_start
        };
        if next_right < 640 {
            subpx_scalar_tail::<EXP_ACTIVE, EFF_EXP, false, true>(
                s_ctx.curr_rgba, s_ctx.prev_rgba, stat_streak_ptr, eval_mask_ptr,
                core::ptr::null_mut(), row_off0, row_p0, s_ctx.row_bytes, 640, s_ctx.w_i,
                s_ctx.ix0, next_right, 640, s_ctx.w00, s_ctx.w10, s_ctx.w01, s_ctx.w11,
                s_ctx.dg_q15, s_ctx.off_i16, s_ctx.hard_u8, s_ctx.mid_u8, s_ctx.soft_u8,
                s_ctx.span_soft_u8, s_ctx.span_reset_u8,
            );
        }
    }
}

#[inline(never)]
unsafe fn subpx_int_row_cold<const EXP_ACTIVE: bool, const EFF_EXP: bool>(
    s_ctx: &SubpxRowCtx,
    eval_buf: *mut v128,
    rgb_u0_buf: *mut v128,
    rgb_u1_buf: *mut v128,
    y: usize,
    int_lo: usize,
    int_hi: usize,
) {
    subpx_int_row_inner::<EXP_ACTIVE, EFF_EXP, false>(
        s_ctx, eval_buf, rgb_u0_buf, rgb_u1_buf, y, int_lo, int_hi,
    );
}

#[inline(never)]
unsafe fn subpx_int_row_hot<const EXP_ACTIVE: bool, const EFF_EXP: bool>(
    s_ctx: &SubpxRowCtx,
    eval_buf: *mut v128,
    rgb_u0_buf: *mut v128,
    rgb_u1_buf: *mut v128,
    y: usize,
    int_lo: usize,
    int_hi: usize,
) {
    subpx_int_row_inner::<EXP_ACTIVE, EFF_EXP, true>(
        s_ctx, eval_buf, rgb_u0_buf, rgb_u1_buf, y, int_lo, int_hi,
    );
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn subpx_hd40_band_fused<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const FILTER_MASK: bool,
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
>(
    s_ctx: &SubpxRowCtx,
    ctx: &mut PanFusedHd40Ctx,
    gy: usize,
    int_lo: usize,
    int_hi: usize,
    grid_out: *mut f32,
    w_eca: v128,
    w_rgb: v128,
    init_reuse_h0: bool,
) {
    let luma_ptr = ctx.luma_ptr;
    let bg_luma_ptr = ctx.bg_luma_ptr;
    let eval_mask_ptr = ctx.eval_mask_ptr;
    let eca_res_ptr = ctx.eca_res_ptr;
    let pyr_eca_ptr = ctx.pyr_eca_ptr;
    let v_alpha = ctx.v_alpha;
    let v_c = ctx.v_c;
    let tol_v = ctx.tol_v;
    let v_rate_bg = ctx.v_rate_bg;
    let v_rate_fg = ctx.v_rate_fg;
    let v_rate_bg_heal = ctx.v_rate_bg_heal;
    let v_rate_fg_heal = ctx.v_rate_fg_heal;
    let v_motion_gate = ctx.v_motion_gate;
    let heal_gate_i16 = ctx.heal_gate_i16;
    let v_rate_eval = ctx.v_rate_eval;
    let v_rate_eval_heal = ctx.v_rate_eval_heal;
    let v_eval_gate = ctx.v_eval_gate;
    let q16 = ctx.q16;

    let mut loc_err2 = ctx.acc_eval_err2;
    let mut loc_sig2 = ctx.acc_eval_sig2;
    let mut loc_cnt_i = ctx.acc_eval_cnt_i;
    let zero_v = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);

    let mut acc_row = [zero_v; 40];
    let mut pyr_a_row_buf = [zero_v; 40];
    let pyr_a_row = &mut pyr_a_row_buf;
    let mut acc_rgb_row_buf = [zero_v; 40];
    let acc_rgb_row = &mut acc_rgb_row_buf;
    let acc_rgb_ptr = acc_rgb_row.as_mut_ptr();
    let mut dummy_cnt = zero_v;
    let y0 = gy * 16;

    let eval_buf = s_ctx.h_cache_ptr.add(160);
    let rgb_u0_buf = s_ctx.h_cache_ptr.add(200);
    let rgb_u1_buf = s_ctx.h_cache_ptr.add(240);

    for ry_blk in 0..4usize {
        let mut blk_cnt_u8 = zero_v;
        let pyr_row_off = (gy * 4 + ry_blk) * 160;
        for ry_in in 0..4usize {
            let ry = ry_blk * 4 + ry_in;
            let y = y0 + ry;
            if ry == 0 && !init_reuse_h0 {
                subpx_int_row_cold::<EXP_ACTIVE, EFF_EXP>(
                    s_ctx, eval_buf, rgb_u0_buf, rgb_u1_buf, y, int_lo, int_hi,
                );
            } else {
                subpx_int_row_hot::<EXP_ACTIVE, EFF_EXP>(
                    s_ctx, eval_buf, rgb_u0_buf, rgb_u1_buf, y, int_lo, int_hi,
                );
            }
            let row_off0 = y * 640;

            macro_rules! flush_pyr {
                ($gx:expr, $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {{
                    let s01 = f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>($r0, $r1),
                        i32x4_shuffle::<1, 3, 5, 7>($r0, $r1),
                    );
                    let s23 = f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>($r2, $r3),
                        i32x4_shuffle::<1, 3, 5, 7>($r2, $r3),
                    );
                    let h4 = f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(s01, s23),
                        i32x4_shuffle::<1, 3, 5, 7>(s01, s23),
                    );
                    if ry_in == 0 {
                        pyr_a_row[$gx] = h4;
                    } else if ry_in < 3 {
                        pyr_a_row[$gx] = f32x4_add(pyr_a_row[$gx], h4);
                    } else {
                        let out = f32x4_mul(f32x4_add(pyr_a_row[$gx], h4), q16);
                        v128_store(pyr_eca_ptr.add(pyr_row_off + $gx * 4) as *mut v128, out);
                    }
                }};
            }

            macro_rules! run_border {
                ($lo:expr, $hi:expr) => {
                    for gx in $lo..$hi {
                        let row_off = row_off0 + gx * 16;
                        let (raw_m16, mv0, mv1, mv2, mv3) = if HAS_MASK {
                            let raw_m16 = v128_load(eval_mask_ptr.add(row_off) as *const v128);
                            let lo8 = i16x8_extend_low_i8x16(raw_m16);
                            let hi8 = i16x8_extend_high_i8x16(raw_m16);
                            (
                                raw_m16,
                                i32x4_extend_low_i16x8(lo8),
                                i32x4_extend_high_i16x8(lo8),
                                i32x4_extend_low_i16x8(hi8),
                                i32x4_extend_high_i16x8(hi8),
                            )
                        } else {
                            (zero_v, zero_v, zero_v, zero_v, zero_v)
                        };
                        let (r0, o0) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                            row_off, luma_ptr, bg_luma_ptr, eca_res_ptr, mv0,
                            v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                        );
                        let (r1, o1) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                            row_off + 4, luma_ptr, bg_luma_ptr, eca_res_ptr, mv1,
                            v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                        );
                        let (r2, o2) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                            row_off + 8, luma_ptr, bg_luma_ptr, eca_res_ptr, mv2,
                            v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                        );
                        let (r3, o3) = pan_fused_border_chunk4_mv::<HAS_MASK, FILTER_MASK>(
                            row_off + 12, luma_ptr, bg_luma_ptr, eca_res_ptr, mv3,
                            v_alpha, v_c, tol_v, &mut loc_err2, &mut loc_sig2, &mut dummy_cnt,
                        );
                        if HAS_MASK {
                            let out_m16 = if FILTER_MASK {
                                let o16 = i8x16_narrow_i16x8(i16x8_narrow_i32x4(o0, o1), i16x8_narrow_i32x4(o2, o3));
                                let m = v128_andnot(raw_m16, o16);
                                v128_store(eval_mask_ptr.add(row_off) as *mut v128, m);
                                m
                            } else {
                                raw_m16
                            };
                            blk_cnt_u8 = i8x16_add(blk_cnt_u8, v128_and(out_m16, one_u8));
                        }
                        flush_pyr!(gx, r0, r1, r2, r3);
                    }
                };
            }

            run_border!(0, int_lo);
            macro_rules! eval_hd40_chunk4 {
                ($p:expr, $st:expr, $mv:expr) => {{
                    let y_cur = v128_load(luma_ptr.add($p) as *const v128);
                    let b_old = v128_load(bg_luma_ptr.add($p) as *const v128);
                    let proj = f32x4_add(f32x4_mul(v_alpha, b_old), v_c);
                    let diff = f32x4_sub(y_cur, proj);
                    let res = f32x4_abs(diff);
                    v128_store(eca_res_ptr.add($p) as *mut v128, res);
                    let dyb = f32x4_sub(y_cur, b_old);
                    let mut over = v_alpha;
                    let mut mv = $mv;
                    if HAS_MASK {
                        if FILTER_MASK {
                            over = f32x4_gt(f32x4_abs(dyb), tol_v);
                            mv = v128_andnot(mv, over);
                        }
                        let m_diff = v128_and(diff, mv);
                        let m_sig = v128_and(y_cur, mv);
                        loc_err2 = f32x4_add(loc_err2, f32x4_mul(m_diff, m_diff));
                        loc_sig2 = f32x4_add(loc_sig2, f32x4_mul(m_sig, m_sig));
                    }
                    let r_bg_s = v128_bitselect(v_rate_bg_heal, v_rate_bg, $st);
                    let r_fg_s = v128_bitselect(v_rate_fg_heal, v_rate_fg, $st);
                    let mut r_eff = v128_bitselect(r_bg_s, r_fg_s, f32x4_lt(res, v_motion_gate));
                    if HAS_MASK && ALLOW_EVAL_HEAL {
                        let r_ev_s = v128_bitselect(v_rate_eval_heal, v_rate_eval, $st);
                        r_eff = v128_bitselect(
                            r_ev_s,
                            r_eff,
                            v128_and(mv, f32x4_lt(res, v_eval_gate)),
                        );
                    }
                    v128_store(
                        bg_luma_ptr.add($p) as *mut v128,
                        f32x4_add(b_old, f32x4_mul(r_eff, dyb)),
                    );
                    (res, over)
                }};
            }
            for gx in int_lo..int_hi {
                let row_off = row_off0 + gx * 16;
                let rgb_u0 = v128_load(rgb_u0_buf.add(gx));
                let rgb_u1 = v128_load(rgb_u1_buf.add(gx));
                let st_u0 = i16x8_lt(rgb_u0, heal_gate_i16);
                let st_u1 = i16x8_lt(rgb_u1, heal_gate_i16);
                let eval16 = v128_load(eval_buf.add(gx));
                let st0 = i32x4_extend_low_i16x8(st_u0);
                let st1 = i32x4_extend_high_i16x8(st_u0);
                let st2 = i32x4_extend_low_i16x8(st_u1);
                let st3 = i32x4_extend_high_i16x8(st_u1);
                let cur_rgb16 = i16x8_add(v128_andnot(rgb_u0, st_u0), v128_andnot(rgb_u1, st_u1));
                let (r0, r1, r2, r3) = if HAS_MASK {
                    let lo8 = i16x8_extend_low_i8x16(eval16);
                    let hi8 = i16x8_extend_high_i8x16(eval16);
                    let (r0, o0) = eval_hd40_chunk4!(row_off, st0, i32x4_extend_low_i16x8(lo8));
                    let (r1, o1) = eval_hd40_chunk4!(row_off + 4, st1, i32x4_extend_high_i16x8(lo8));
                    let (r2, o2) = eval_hd40_chunk4!(row_off + 8, st2, i32x4_extend_low_i16x8(hi8));
                    let (r3, o3) = eval_hd40_chunk4!(row_off + 12, st3, i32x4_extend_high_i16x8(hi8));
                    let out_m16 = if FILTER_MASK {
                        let o16 = i8x16_narrow_i16x8(i16x8_narrow_i32x4(o0, o1), i16x8_narrow_i32x4(o2, o3));
                        v128_andnot(eval16, o16)
                    } else {
                        eval16
                    };
                    v128_store(eval_mask_ptr.add(row_off) as *mut v128, out_m16);
                    blk_cnt_u8 = i8x16_add(blk_cnt_u8, v128_and(out_m16, one_u8));
                    (r0, r1, r2, r3)
                } else {
                    v128_store(eval_mask_ptr.add(row_off) as *mut v128, eval16);
                    let (r0, _) = eval_hd40_chunk4!(row_off, st0, zero_v);
                    let (r1, _) = eval_hd40_chunk4!(row_off + 4, st1, zero_v);
                    let (r2, _) = eval_hd40_chunk4!(row_off + 8, st2, zero_v);
                    let (r3, _) = eval_hd40_chunk4!(row_off + 12, st3, zero_v);
                    (r0, r1, r2, r3)
                };
                let cur_r = f32x4_add(
                    f32x4_add(v128_andnot(r0, st0), v128_andnot(r1, st1)),
                    f32x4_add(v128_andnot(r2, st2), v128_andnot(r3, st3)),
                );
                acc_row[gx] = f32x4_add(acc_row[gx], cur_r);
                let ar = acc_rgb_ptr.add(gx);
                *ar = i16x8_add(*ar, cur_rgb16);
                flush_pyr!(gx, r0, r1, r2, r3);
            }
            run_border!(int_hi, 40);
        }
        if HAS_MASK {
            let sum16 = u16x8_extadd_pairwise_u8x16(blk_cnt_u8);
            loc_cnt_i = i32x4_add(loc_cnt_i, u32x4_extadd_pairwise_u16x8(sum16));
        }
    }
    ctx.acc_eval_err2 = loc_err2;
    ctx.acc_eval_sig2 = loc_sig2;
    ctx.acc_eval_cnt_i = loc_cnt_i;
    for gx in 0..40usize {
        *grid_out.add(gy * 40 + gx) = if gx >= int_lo && gx < int_hi {
            let sum_rgb = u32x4_extadd_pairwise_u16x8(acc_rgb_row[gx]);
            hsum_f32x4(f32x4_add(
                f32x4_mul(acc_row[gx], w_eca),
                f32x4_mul(f32x4_convert_u32x4(sum_rgb), w_rgb),
            ))
        } else {
            0.0
        };
    }
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn pan_subpx_fused_hd40_no_scratch<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const FILTER_MASK: bool,
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
>(
    subpx_ctx: &SubpxRowCtx,
    hd40_ctx: &mut PanFusedHd40Ctx,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    grid_out: *mut f32,
    y_start: usize,
    y_end: usize,
    grid_h: usize,
    gx_lo: usize,
    gx_hi: usize,
    gy_lo: usize,
    gy_hi: usize,
    w_eca: v128,
    w_rgb: v128,
) {
    let zero_v = i32x4_splat(0);
    let mut rgb_row_buf = [zero_v; 80];
    let rgb_row_ptr = rgb_row_buf.as_mut_ptr() as *mut u16;
    let mut pyr_a_row_buf = [zero_v; 40];
    let pyr_a_row = &mut pyr_a_row_buf;
    let mut acc_rgb_row_buf = [zero_v; 40];
    let acc_rgb_row = &mut acc_rgb_row_buf;
    for gy in 0..grid_h {
        let gy_valid = gy >= gy_lo && gy < gy_hi;
        let (int_lo, int_hi) = if gy_valid {
            let a = gx_lo.min(40);
            (a, gx_hi.min(40).max(a))
        } else {
            (0, 0)
        };
        let mut acc_row = [zero_v; 40];
        if gy_valid {
            *acc_rgb_row = [zero_v; 40];
        }
        let acc_rgb_ptr = acc_rgb_row.as_mut_ptr();
        for ry_blk in 0..4usize {
            let ry0 = ry_blk * 4;
            for ry_in in 0..4usize {
                let y = gy * 16 + ry0 + ry_in;
                if y < y_start || y >= y_end {
                    subpx_zero_rows::<false>(
                        stat_streak_ptr,
                        eval_mask_ptr,
                        core::ptr::null_mut(),
                        y,
                        y + 1,
                        640,
                    );
                    core::ptr::write_bytes(rgb_row_ptr, 0, 640);
                } else {
                    subpx_pan_streak_row_ctx::<EXP_ACTIVE, EFF_EXP, true, true>(
                        subpx_ctx,
                        rgb_row_ptr as *mut u8,
                        y,
                        false,
                    );
                }
                match ry_in {
                    0 => pan_fused_hd40_row_phased::<HAS_MASK, ALLOW_EVAL_HEAL, true, FILTER_MASK, 0, true>(
                        hd40_ctx, gy, ry_blk, y * 640, int_lo, int_hi,
                        rgb_row_ptr as *const u16, &mut acc_row, acc_rgb_ptr, &mut *pyr_a_row,
                    ),
                    1 | 2 => pan_fused_hd40_row_phased::<HAS_MASK, ALLOW_EVAL_HEAL, true, FILTER_MASK, 1, true>(
                        hd40_ctx, gy, ry_blk, y * 640, int_lo, int_hi,
                        rgb_row_ptr as *const u16, &mut acc_row, acc_rgb_ptr, &mut *pyr_a_row,
                    ),
                    _ => pan_fused_hd40_row_phased::<HAS_MASK, ALLOW_EVAL_HEAL, true, FILTER_MASK, 2, true>(
                        hd40_ctx, gy, ry_blk, y * 640, int_lo, int_hi,
                        rgb_row_ptr as *const u16, &mut acc_row, acc_rgb_ptr, &mut *pyr_a_row,
                    ),
                }
            }
        }
        for gx in 0..40usize {
            *grid_out.add(gy * 40 + gx) = if gx >= int_lo && gx < int_hi {
                let sum_rgb = u32x4_extadd_pairwise_u16x8(acc_rgb_row[gx]);
                hsum_f32x4(f32x4_add(
                    f32x4_mul(acc_row[gx], w_eca),
                    f32x4_mul(f32x4_convert_u32x4(sum_rgb), w_rgb),
                ))
            } else {
                0.0
            };
        }
    }
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn pan_subpx_fused_grid_fallback<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const FILTER_MASK: bool,
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
>(
    subpx_ctx: &SubpxRowCtx,
    h_cache_ptr: *mut v128,
    cached_sy: &mut i32,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    stat_streak_ptr: *mut u8,
    grid_out: *mut f32,
    fuse_pyr: bool,
    frame_w: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    iy0: i32,
    y_start: usize,
    y_end: usize,
    gx_lo: usize,
    gx_hi: usize,
    gy_lo: usize,
    gy_hi: usize,
    v_alpha: v128,
    v_c: v128,
    tol_v: v128,
    v_rate_bg: v128,
    v_rate_fg: v128,
    v_motion_gate: v128,
    heal_alpha: v128,
    heal_gate_rgb: v128,
    w_eca: v128,
    w_rgb: v128,
    v_rate_eval: v128,
    v_eval_gate: v128,
    zero_v: v128,
    q4: v128,
    q16: v128,
    acc_eval_err2: &mut v128,
    acc_eval_sig2: &mut v128,
    acc_eval_cnt_i: &mut v128,
) {
    let mut rgb_row_uninit = [i32x4_splat(0); 80];
    let rgb_row_ptr = rgb_row_uninit.as_mut_ptr() as *mut u16;
    for gy in 0..grid_h {
        let gy_valid = gy >= gy_lo && gy < gy_hi;
        let (int_lo, int_hi) = if gy_valid {
            let a = gx_lo.min(grid_w);
            (a, gx_hi.min(grid_w).max(a))
        } else {
            (0, 0)
        };
        let mut acc_row = [zero_v; 40];
        let mut pyr_a_row = [zero_v; 40];
        let mut pyr_b_row = [zero_v; 20];
        for ry in 0..16usize {
            let y = gy * 16 + ry;
            if y < y_start || y >= y_end {
                *cached_sy = i32::MIN;
                subpx_zero_rows::<false>(
                    stat_streak_ptr,
                    eval_mask_ptr,
                    core::ptr::null_mut(),
                    y,
                    y + 1,
                    frame_w,
                );
                core::ptr::write_bytes(rgb_row_ptr, 0, frame_w);
            } else {
                let sy = (y as i32) + iy0;
                let reuse_h0 = !h_cache_ptr.is_null() && *cached_sy == sy;
                subpx_pan_streak_row_ctx::<EXP_ACTIVE, EFF_EXP, true, true>(
                    subpx_ctx,
                    rgb_row_ptr as *mut u8,
                    y,
                    reuse_h0,
                );
                if !h_cache_ptr.is_null() {
                    *cached_sy = sy + 1;
                }
            }
            pan_fused_grid_le40_row::<
                HAS_MASK, ALLOW_EVAL_HEAL, true, FILTER_MASK, false, 0, true, true,
            >(
                gy, ry, ry >> 2, y * frame_w, grid_w, scale_px, fuse_pyr, int_lo, int_hi,
                luma_ptr, bg_luma_ptr, eval_mask_ptr, eca_res_ptr, rgb_row_ptr as *const u8,
                pyr_eca_ptr, v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                zero_v, q4, q16, acc_eval_err2, acc_eval_sig2, acc_eval_cnt_i,
                &mut acc_row, &mut pyr_a_row, pyr_b_row.as_mut_ptr(),
            );
        }
        for gx in 0..grid_w {
            *grid_out.add(gy * grid_w + gx) = if gx >= int_lo && gx < int_hi {
                hsum_f32x4(acc_row[gx])
            } else {
                0.0
            };
        }
    }
}

#[inline(never)]
unsafe fn pan_streak_and_eca_fused_subpx_impl<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const FILTER_MASK: bool,
    const EXP_ACTIVE: bool,
    const EFF_EXP: bool,
>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    mask_luma_tol: f32,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    scratch_ptr: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    alpha: f32,
    offset_c: f32,
    adapt_rate: f32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    dg_q15: i16,
    off_i16: i16,
    ix0: i32,
    iy0: i32,
    wx0: u16,
    wx1: u16,
    wy0: u16,
    wy1: u16,
    gx_lo: usize,
    gx_hi: usize,
    gy_lo: usize,
    gy_hi: usize,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> f32 {
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let sx_limit = w_i - 1;
    let sy_limit = h_i - 1;
    let heal_alpha_s = if EFF_EXP { 0.86f32 } else { 0.78f32 };
    let scalar_gate = if EFF_EXP { 0.026f32 } else { 0.014f32 };
    let heal_gate_s = scalar_gate * 765.0;

    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let tol_v = f32x4_splat(mask_luma_tol);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);
    let heal_alpha = f32x4_splat(heal_alpha_s);
    let heal_gate_rgb = f32x4_splat(heal_gate_s);
    let w_eca = f32x4_splat(0.75);
    let w_rgb = f32x4_splat(0.25 / 765.0);
    let v_rate_eval = f32x4_splat(0.98);
    let v_eval_gate = f32x4_splat(0.062);

    let zero_v = f32x4_splat(0.0);
    let q4 = f32x4_splat(0.25);
    let q16 = f32x4_splat(1.0 / 16.0);
    let fuse_pyr = !pyr_eca_ptr.is_null() && scale_px <= 2 && frame_w == 320 * scale_px;

    let fast_pan = pan_dx_f.abs() + pan_dy_f.abs() >= 3.2 * (scale_px as f32);
    let hard_u8 = 4u8;
    let mid_u8 = 5u8;
    let soft_u8 = 8u8;
    let span_reset_u8 = if fast_pan { 68u8 } else { 35u8 };
    let span_soft_u8 = if fast_pan { 50u8 } else { 26u8 };
    let row_bytes = frame_w * 4;

    let y_start = ((-iy0).clamp(0, h_i)) as usize;
    let y_end = (((h_i - 1 - iy0).clamp(0, h_i)) as usize).max(y_start);
    let x_lo = ((-ix0).clamp(0, w_i)) as usize;
    let x_hi_lim = (w_i - 16).min(w_i - 17 - ix0);
    let n_blocks16 = if (x_lo as i32) <= x_hi_lim {
        (((x_hi_lim - (x_lo as i32)) >> 4) + 1) as usize
    } else {
        0
    };
    let x_simd_end = x_lo + n_blocks16 * 16;

    let mut acc_eval_err2 = zero_v;
    let mut acc_eval_sig2 = zero_v;
    let mut acc_eval_cnt_i = i32x4_splat(0);
    let mut tail_eval_err2 = 0.0f32;
    let mut tail_eval_sig2 = 0.0f32;
    let mut tail_eval_cnt = 0u32;

    let h_cache_ptr: *mut v128 = if !scratch_ptr.is_null() {
        (((scratch_ptr as usize) + 15) & !15) as *mut v128
    } else {
        core::ptr::null_mut()
    };
    let subpx_ctx = SubpxRowCtx::new::<EFF_EXP>(
        curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, h_cache_ptr,
        frame_w, w_i, row_bytes, ix0, iy0, x_lo, n_blocks16, x_simd_end,
        wx0, wx1, wy0, wy1, dg_q15, off_i16, hard_u8, mid_u8, soft_u8, span_soft_u8, span_reset_u8,
        true,
    );
    let mut cached_sy = i32::MIN;

    macro_rules! run_streak_row {
        ($y:expr, $rgb_dst:expr, $zero_rgb:expr, $rgb_u16:expr) => {
            if $y < y_start || $y >= y_end {
                cached_sy = i32::MIN;
                subpx_zero_rows::<$zero_rgb>(
                    stat_streak_ptr,
                    eval_mask_ptr,
                    if $zero_rgb { eca_res_ptr } else { core::ptr::null_mut() },
                    $y,
                    $y + 1,
                    frame_w,
                );
                if !$zero_rgb {
                    if $rgb_u16 {
                        core::ptr::write_bytes($rgb_dst as *mut u16, 0, frame_w);
                    } else {
                        core::ptr::write_bytes($rgb_dst as *mut f32, 0, frame_w);
                    }
                }
            } else {
                let sy = ($y as i32) + iy0;
                let reuse_h0 = !h_cache_ptr.is_null() && cached_sy == sy;
                subpx_pan_streak_row_ctx::<EXP_ACTIVE, EFF_EXP, true, $rgb_u16>(
                    &subpx_ctx,
                    $rgb_dst as *mut u8,
                    $y,
                    reuse_h0,
                );
                if !h_cache_ptr.is_null() {
                    cached_sy = sy + 1;
                }
            }
        };
    }

    if fuse_pyr && scale_px == 2 && grid_w == 40 && frame_w == 640 {
        let g_i = heal_gate_s as i32;
        let heal_gate_lim = (g_i + if (g_i as f32) < heal_gate_s { 1 } else { 0 }) as i16;
        let one_minus_heal = f32x4_sub(f32x4_splat(1.0), heal_alpha);
        let mut hd40_ctx = PanFusedHd40Ctx {
            v_alpha,
            v_c,
            tol_v,
            v_rate_bg,
            v_rate_fg,
            v_rate_bg_heal: f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_bg)),
            v_rate_fg_heal: f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_fg)),
            v_motion_gate,
            heal_gate_i16: i16x8_splat(heal_gate_lim),
            w_eca,
            w_rgb,
            v_rate_eval,
            v_rate_eval_heal: f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_eval)),
            v_eval_gate,
            q16,
            acc_eval_err2,
            acc_eval_sig2,
            acc_eval_cnt_i,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
        };
        if !scratch_ptr.is_null() {
            let band_rgb_ptr = h_cache_ptr.add(160) as *mut u16;
            let mut cached_fused_sy = i32::MIN;
            for gy in 0..grid_h {
                let gy_valid = gy >= gy_lo && gy < gy_hi;
                let (int_lo, int_hi) = if gy_valid {
                    let a = gx_lo.min(40);
                    (a, gx_hi.min(40).max(a))
                } else {
                    (0, 0)
                };
                let y0 = gy * 16;
                let y1 = y0 + 16;
                if gy_valid
                    && int_lo < int_hi
                    && subpx_ctx.max_w <= 255
                    && subpx_ctx.extmul_mode == 0
                    && y0 >= y_start
                    && y1 <= y_end
                {
                    let init_reuse = cached_fused_sy == (y0 as i32) + iy0;
                    subpx_hd40_band_fused::<HAS_MASK, ALLOW_EVAL_HEAL, FILTER_MASK, EXP_ACTIVE, EFF_EXP>(
                        &subpx_ctx,
                        &mut hd40_ctx,
                        gy,
                        int_lo,
                        int_hi,
                        grid_out,
                        w_eca,
                        w_rgb,
                        init_reuse,
                    );
                    cached_fused_sy = (y1 as i32) + iy0;
                    cached_sy = i32::MIN;
                } else {
                    cached_fused_sy = i32::MIN;
                    if subpx_ctx.max_w <= 255
                        && subpx_ctx.write_h1
                        && subpx_ctx.extmul_mode == 0
                        && y0 >= y_start
                        && y1 <= y_end
                    {
                        let init_reuse = cached_sy == (y0 as i32) + iy0;
                        subpx_pan_streak_band_2d::<EXP_ACTIVE, EFF_EXP>(
                            &subpx_ctx,
                            band_rgb_ptr,
                            y0,
                            init_reuse,
                        );
                        cached_sy = (y1 as i32) + iy0;
                    } else {
                        for ry in 0..16usize {
                            let y = y0 + ry;
                            run_streak_row!(y, band_rgb_ptr.add(ry * 640) as *mut u8, false, true);
                        }
                    }
                    pan_hd40_band::<HAS_MASK, ALLOW_EVAL_HEAL, true, FILTER_MASK>(
                        &mut hd40_ctx,
                        gy,
                        gy_valid,
                        int_lo,
                        int_hi,
                        band_rgb_ptr as *const u16,
                        grid_out,
                        w_eca,
                        w_rgb,
                    );
                }
            }
        } else {
            pan_subpx_fused_hd40_no_scratch::<HAS_MASK, ALLOW_EVAL_HEAL, FILTER_MASK, EXP_ACTIVE, EFF_EXP>(
                &subpx_ctx,
                &mut hd40_ctx,
                stat_streak_ptr,
                eval_mask_ptr,
                grid_out,
                y_start,
                y_end,
                grid_h,
                gx_lo,
                gx_hi,
                gy_lo,
                gy_hi,
                w_eca,
                w_rgb,
            );
        }
        acc_eval_err2 = hd40_ctx.acc_eval_err2;
        acc_eval_sig2 = hd40_ctx.acc_eval_sig2;
        acc_eval_cnt_i = hd40_ctx.acc_eval_cnt_i;
    } else {
        pan_subpx_fused_grid_fallback::<HAS_MASK, ALLOW_EVAL_HEAL, FILTER_MASK, EXP_ACTIVE, EFF_EXP>(
            &subpx_ctx,
            h_cache_ptr,
            &mut cached_sy,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
            stat_streak_ptr,
            grid_out,
            fuse_pyr,
            frame_w,
            grid_w,
            grid_h,
            scale_px,
            iy0,
            y_start,
            y_end,
            gx_lo,
            gx_hi,
            gy_lo,
            gy_hi,
            v_alpha,
            v_c,
            tol_v,
            v_rate_bg,
            v_rate_fg,
            v_motion_gate,
            heal_alpha,
            heal_gate_rgb,
            w_eca,
            w_rgb,
            v_rate_eval,
            v_eval_gate,
            zero_v,
            q4,
            q16,
            &mut acc_eval_err2,
            &mut acc_eval_sig2,
            &mut acc_eval_cnt_i,
        );
    }

    let grid_bot = grid_h * 16;
    if grid_bot < frame_h {
        for y in grid_bot..frame_h {
            run_streak_row!(y, eca_res_ptr.add(y * frame_w) as *mut u8, true, false);
        }
        pan_fused_tail_rows_simd128::<HAS_MASK, ALLOW_EVAL_HEAL, FILTER_MASK>(
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            eca_res_ptr,
            mask_luma_tol,
            frame_w,
            frame_h,
            grid_h,
            ix0,
            iy0,
            sx_limit,
            sy_limit,
            alpha,
            offset_c,
            adapt_rate,
            heal_alpha_s,
            heal_gate_s,
            v_alpha,
            v_c,
            tol_v,
            v_rate_bg,
            v_rate_fg,
            v_motion_gate,
            heal_alpha,
            heal_gate_rgb,
            v_rate_eval,
            v_eval_gate,
            &mut acc_eval_err2,
            &mut acc_eval_sig2,
            &mut acc_eval_cnt_i,
            &mut tail_eval_err2,
            &mut tail_eval_sig2,
            &mut tail_eval_cnt,
        );
    }

    let err2 = hsum_f32x4(acc_eval_err2) + tail_eval_err2;
    let sig2 = hsum_f32x4(acc_eval_sig2) + tail_eval_sig2;
    let eval_count = (hsum_i32x4(acc_eval_cnt_i) as u32).saturating_add(tail_eval_cnt);
    if !out_err2.is_null() {
        *out_err2 = err2;
    }
    if !out_sig2.is_null() {
        *out_sig2 = sig2;
    }
    if !out_eval_count.is_null() {
        *out_eval_count = eval_count;
    }
    if eval_count >= 1024 && sig2 > 1e-3 {
        err2 / sig2
    } else {
        -1.0
    }
}

#[inline(always)]
unsafe fn pan_subpx_eval_bit_scalar(
    p: usize,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *const u8,
    frame_w: usize,
    w_i: i32,
    h_i: i32,
    row_bytes: usize,
    ix0: i32,
    iy0: i32,
    w00: u16,
    w10: u16,
    w01: u16,
    w11: u16,
    exp_active: bool,
    eff_exp_regime: bool,
    dg_q15: i16,
    off_i16: i16,
    span_soft_u8: u8,
    span_reset_u8: u8,
) -> bool {
    let old_s = *stat_streak_ptr.add(p);
    if old_s < 10 {
        return false;
    }
    let y = p / frame_w;
    let x = p - y * frame_w;
    let sy = (y as i32) + iy0;
    let sx = (x as i32) + ix0;
    if sy < 0 || sy + 1 >= h_i || sx < 0 || sx + 1 >= w_i {
        return false;
    }
    let c_b = p * 4;
    let p00_b = ((sy as usize) * frame_w + (sx as usize)) * 4;
    let p10_b = p00_b + 4;
    let p01_b = p00_b + row_bytes;
    let p11_b = p01_b + 4;
    let mut max_d = 0u8;
    let mut max_span = 0u8;
    let mut pv_raw3 = [0u8; 3];
    for ch in 0..3usize {
        let v00 = *prev_rgba.add(p00_b + ch);
        let v10 = *prev_rgba.add(p10_b + ch);
        let v01 = *prev_rgba.add(p01_b + ch);
        let v11 = *prev_rgba.add(p11_b + ch);
        let sp = v00.max(v10).max(v01).max(v11) - v00.min(v10).min(v01).min(v11);
        if sp > max_span {
            max_span = sp;
        }
        let pv = ((w00 as u32) * (v00 as u32)
            + (w10 as u32) * (v10 as u32)
            + (w01 as u32) * (v01 as u32)
            + (w11 as u32) * (v11 as u32)
            + 128)
            >> 8;
        let pv_raw = pv as u8;
        pv_raw3[ch] = pv_raw;
        let c_ch = *curr_rgba.add(c_b + ch);
        let pv_u8 = if exp_active {
            apply_exposure_u8_scalar(pv_raw, dg_q15, off_i16)
        } else {
            pv_raw
        };
        let d = c_ch.abs_diff(pv_u8);
        if ch == 2
            && eff_exp_regime
            && (pv_raw >= 250 || c_ch >= 250)
            && pv_raw3[0].min(pv_raw3[1]) < 230
        {
            continue;
        }
        if d > max_d {
            max_d = d;
        }
    }
    let white_clip = eff_exp_regime
        && ((pv_raw3[0] >= 251 && pv_raw3[1] >= 251)
            || (*curr_rgba.add(c_b) >= 253 && *curr_rgba.add(c_b + 1) >= 253));
    if white_clip && max_d <= 4 {
        max_d = 5;
    }
    if max_span > span_reset_u8 {
        max_d = 255;
    } else if max_span > span_soft_u8 && max_d <= 4 {
        max_d = 5;
    }
    let ns = if max_d <= 4 {
        old_s.saturating_add(2).min(20)
    } else if max_d <= 5 {
        old_s.saturating_sub(1)
    } else if max_d <= 8 {
        old_s.min(14).saturating_sub(6)
    } else {
        0
    };
    ns >= 12 && !white_clip
}

#[inline(never)]
unsafe fn eca_fit_affine_params_pan_subpx_live_simd128(
    luma_ptr: *const f32,
    bg_luma_ptr: *const f32,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *const u8,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    pan_dx_f: f32,
    pan_dy_f: f32,
    ix0: i32,
    iy0: i32,
    wx0: u16,
    wx1: u16,
    wy0: u16,
    wy1: u16,
    exp_active: bool,
    eff_exp_regime: bool,
    dg_q15: i16,
    off_i16: i16,
) -> Option<(f32, f32)> {
    let len = frame_w * frame_h;
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let row_bytes = frame_w * 4;
    let w00 = wx0 * wy0;
    let w10 = wx1 * wy0;
    let w01 = wx0 * wy1;
    let w11 = wx1 * wy1;
    let fast_pan = pan_dx_f.abs() + pan_dy_f.abs() >= 3.2 * (scale_px as f32);
    let span_reset_u8 = if fast_pan { 68u8 } else { 35u8 };
    let span_soft_u8 = if fast_pan { 50u8 } else { 26u8 };

    let guard_thresh = f32x4_splat(0.045);
    let guard_stat = f32x4_splat(0.15);
    let ten_u8 = u8x16_splat(10);
    let four_i32 = i32x4_splat(4);

    let mut b_ref = 0.5f32;
    let mut y_ref = 0.5f32;
    for k in 0..len {
        let bk = *bg_luma_ptr.add(k);
        let yk = *luma_ptr.add(k);
        let ad = (yk - bk).abs();
        if ad < 0.045f32
            || (ad < 0.15f32
                && pan_subpx_eval_bit_scalar(
                    k, curr_rgba, prev_rgba, stat_streak_ptr, frame_w, w_i, h_i,
                    row_bytes, ix0, iy0, w00, w10, w01, w11, exp_active,
                    eff_exp_regime, dg_q15, off_i16, span_soft_u8, span_reset_u8,
                ))
        {
            b_ref = bk;
            y_ref = yk;
            break;
        }
    }
    let b_ref_vec = f32x4_splat(b_ref);
    let y_ref_vec = f32x4_splat(y_ref);

    let mut acc_raw_b2_0 = f32x4_splat(0.0);
    let mut acc_raw_b2_1 = f32x4_splat(0.0);
    let mut acc_cnt_i32_0 = i32x4_splat(0);
    let mut acc_cnt_i32_1 = i32x4_splat(0);
    let mut tot_u = 0.0f64;
    let mut tot_v = 0.0f64;
    let mut tot_uu = 0.0f64;
    let mut tot_uv = 0.0f64;
    let mut c_u0 = f32x4_splat(0.0);
    let mut c_v0 = f32x4_splat(0.0);
    let mut c_uu0 = f32x4_splat(0.0);
    let mut c_uv0 = f32x4_splat(0.0);
    let mut c_u1 = f32x4_splat(0.0);
    let mut c_v1 = f32x4_splat(0.0);
    let mut c_uu1 = f32x4_splat(0.0);
    let mut c_uv1 = f32x4_splat(0.0);

    let use_stride7 = len >= 131_072;
    let use_stride3 = !use_stride7 && len >= 32768;
    let step16 = if use_stride7 { 112usize } else if use_stride3 { 48usize } else { 16usize };
    let step_scale = if use_stride7 { 7i64 } else if use_stride3 { 3i64 } else { 1i64 };

    let mut i = 0usize;
    let mut blk_rem = 64usize;
    while i + 16 <= len {
        let y0 = v128_load(luma_ptr.add(i) as *const v128);
        let b0 = v128_load(bg_luma_ptr.add(i) as *const v128);
        let y1 = v128_load(luma_ptr.add(i + 4) as *const v128);
        let b1 = v128_load(bg_luma_ptr.add(i + 4) as *const v128);
        let y2 = v128_load(luma_ptr.add(i + 8) as *const v128);
        let b2 = v128_load(bg_luma_ptr.add(i + 8) as *const v128);
        let y3 = v128_load(luma_ptr.add(i + 12) as *const v128);
        let b3 = v128_load(bg_luma_ptr.add(i + 12) as *const v128);

        let d0 = f32x4_abs(f32x4_sub(y0, b0));
        let d1 = f32x4_abs(f32x4_sub(y1, b1));
        let d2 = f32x4_abs(f32x4_sub(y2, b2));
        let d3 = f32x4_abs(f32x4_sub(y3, b3));

        let mut w0 = f32x4_lt(d0, guard_thresh);
        let mut w1 = f32x4_lt(d1, guard_thresh);
        let mut w2 = f32x4_lt(d2, guard_thresh);
        let mut w3 = f32x4_lt(d3, guard_thresh);

        if i32x4_all_true(v128_and(v128_and(w0, w1), v128_and(w2, w3))) {
            acc_raw_b2_0 = f32x4_add(f32x4_add(acc_raw_b2_0, f32x4_mul(b0, b0)), f32x4_mul(b2, b2));
            acc_raw_b2_1 = f32x4_add(f32x4_add(acc_raw_b2_1, f32x4_mul(b1, b1)), f32x4_mul(b3, b3));
            acc_cnt_i32_0 = i32x4_add(acc_cnt_i32_0, four_i32);
            let u0 = f32x4_sub(b0, b_ref_vec);
            let v0 = f32x4_sub(y0, y_ref_vec);
            let u2 = f32x4_sub(b2, b_ref_vec);
            let v2 = f32x4_sub(y2, y_ref_vec);
            c_u0 = f32x4_add(f32x4_add(c_u0, u0), u2);
            c_v0 = f32x4_add(f32x4_add(c_v0, v0), v2);
            c_uu0 = f32x4_add(f32x4_add(c_uu0, f32x4_mul(u0, u0)), f32x4_mul(u2, u2));
            c_uv0 = f32x4_add(f32x4_add(c_uv0, f32x4_mul(u0, v0)), f32x4_mul(u2, v2));
            let u1 = f32x4_sub(b1, b_ref_vec);
            let v1 = f32x4_sub(y1, y_ref_vec);
            let u3 = f32x4_sub(b3, b_ref_vec);
            let v3 = f32x4_sub(y3, y_ref_vec);
            c_u1 = f32x4_add(f32x4_add(c_u1, u1), u3);
            c_v1 = f32x4_add(f32x4_add(c_v1, v1), v3);
            c_uu1 = f32x4_add(f32x4_add(c_uu1, f32x4_mul(u1, u1)), f32x4_mul(u3, u3));
            c_uv1 = f32x4_add(f32x4_add(c_uv1, f32x4_mul(u1, v1)), f32x4_mul(u3, v3));
        } else {
            let mid0 = v128_andnot(f32x4_lt(d0, guard_stat), w0);
            let mid1 = v128_andnot(f32x4_lt(d1, guard_stat), w1);
            let mid2 = v128_andnot(f32x4_lt(d2, guard_stat), w2);
            let mid3 = v128_andnot(f32x4_lt(d3, guard_stat), w3);
            let mid_or = v128_or(v128_or(mid0, mid1), v128_or(mid2, mid3));
            if v128_any_true(mid_or) {
                let old_s = v128_load(stat_streak_ptr.add(i) as *const v128);
                let ge10 = u8x16_ge(old_s, ten_u8);
                let mid16 = i8x16_narrow_i16x8(
                    i16x8_narrow_i32x4(mid0, mid1),
                    i16x8_narrow_i32x4(mid2, mid3),
                );
                let cmask = i8x16_bitmask(v128_and(mid16, ge10));
                if cmask != 0 {
                    let mut bits = [0u8; 16];
                    for k in 0..16usize {
                        if ((cmask >> k) & 1) != 0
                            && pan_subpx_eval_bit_scalar(
                                i + k, curr_rgba, prev_rgba, stat_streak_ptr, frame_w,
                                w_i, h_i, row_bytes, ix0, iy0, w00, w10, w01, w11,
                                exp_active, eff_exp_regime, dg_q15, off_i16,
                                span_soft_u8, span_reset_u8,
                            )
                        {
                            bits[k] = 0xFF;
                        }
                    }
                    let eval16 = v128_load(bits.as_ptr() as *const v128);
                    let lo8 = i16x8_extend_low_i8x16(eval16);
                    let hi8 = i16x8_extend_high_i8x16(eval16);
                    w0 = v128_or(w0, v128_and(mid0, i32x4_extend_low_i16x8(lo8)));
                    w1 = v128_or(w1, v128_and(mid1, i32x4_extend_high_i16x8(lo8)));
                    w2 = v128_or(w2, v128_and(mid2, i32x4_extend_low_i16x8(hi8)));
                    w3 = v128_or(w3, v128_and(mid3, i32x4_extend_high_i16x8(hi8)));
                }
            }
            macro_rules! accum_live_subpx {
                ($lane:tt, $yv:expr, $bv:expr, $wv:expr) => {{
                    let um = v128_and(f32x4_sub($bv, b_ref_vec), $wv);
                    let vm = v128_and(f32x4_sub($yv, y_ref_vec), $wv);
                    if $lane == 0 {
                        acc_raw_b2_0 = f32x4_add(acc_raw_b2_0, f32x4_mul($bv, $bv));
                        acc_cnt_i32_0 = i32x4_sub(acc_cnt_i32_0, $wv);
                        c_u0 = f32x4_add(c_u0, um);
                        c_v0 = f32x4_add(c_v0, vm);
                        c_uu0 = f32x4_add(c_uu0, f32x4_mul(um, um));
                        c_uv0 = f32x4_add(c_uv0, f32x4_mul(um, vm));
                    } else {
                        acc_raw_b2_1 = f32x4_add(acc_raw_b2_1, f32x4_mul($bv, $bv));
                        acc_cnt_i32_1 = i32x4_sub(acc_cnt_i32_1, $wv);
                        c_u1 = f32x4_add(c_u1, um);
                        c_v1 = f32x4_add(c_v1, vm);
                        c_uu1 = f32x4_add(c_uu1, f32x4_mul(um, um));
                        c_uv1 = f32x4_add(c_uv1, f32x4_mul(um, vm));
                    }
                }};
            }
            accum_live_subpx!(0, y0, b0, w0);
            accum_live_subpx!(1, y1, b1, w1);
            accum_live_subpx!(0, y2, b2, w2);
            accum_live_subpx!(1, y3, b3, w3);
        }

        i += step16;
        blk_rem -= 1;
        if blk_rem == 0 {
            tot_u += hsum_f32x4(f32x4_add(c_u0, c_u1)) as f64;
            tot_v += hsum_f32x4(f32x4_add(c_v0, c_v1)) as f64;
            tot_uu += hsum_f32x4(f32x4_add(c_uu0, c_uu1)) as f64;
            tot_uv += hsum_f32x4(f32x4_add(c_uv0, c_uv1)) as f64;
            c_u0 = f32x4_splat(0.0);
            c_v0 = f32x4_splat(0.0);
            c_uu0 = f32x4_splat(0.0);
            c_uv0 = f32x4_splat(0.0);
            c_u1 = f32x4_splat(0.0);
            c_v1 = f32x4_splat(0.0);
            c_uu1 = f32x4_splat(0.0);
            c_uv1 = f32x4_splat(0.0);
            blk_rem = 64;
        }
    }
    let mut tail_raw_b2 = 0.0f32;
    let mut tail_cnt = 0i64;
    let mut tail_u = 0.0f64;
    let mut tail_v = 0.0f64;
    let mut tail_uu = 0.0f64;
    let mut tail_uv = 0.0f64;
    while i < len {
        let y = *luma_ptr.add(i);
        let b = *bg_luma_ptr.add(i);
        tail_raw_b2 += b * b;
        let ad = (y - b).abs();
        if ad < 0.045f32
            || (ad < 0.15f32
                && pan_subpx_eval_bit_scalar(
                    i, curr_rgba, prev_rgba, stat_streak_ptr, frame_w, w_i, h_i,
                    row_bytes, ix0, iy0, w00, w10, w01, w11, exp_active,
                    eff_exp_regime, dg_q15, off_i16, span_soft_u8, span_reset_u8,
                ))
        {
            let u = (b - b_ref) as f64;
            let v = (y - y_ref) as f64;
            tail_cnt += 1;
            tail_u += u;
            tail_v += v;
            tail_uu += u * u;
            tail_uv += u * v;
        }
        i += 1;
    }

    tot_u += hsum_f32x4(f32x4_add(c_u0, c_u1)) as f64 + tail_u;
    tot_v += hsum_f32x4(f32x4_add(c_v0, c_v1)) as f64 + tail_v;
    tot_uu += hsum_f32x4(f32x4_add(c_uu0, c_uu1)) as f64 + tail_uu;
    tot_uv += hsum_f32x4(f32x4_add(c_uv0, c_uv1)) as f64 + tail_uv;
    let acc_cnt_i32 = i32x4_add(acc_cnt_i32_0, acc_cnt_i32_1);
    let raw_b2 = hsum_f32x4(f32x4_add(acc_raw_b2_0, acc_raw_b2_1)) + tail_raw_b2;
    let n_cnt = (i32x4_extract_lane::<0>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<1>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<2>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<3>(acc_cnt_i32) as i64)
        + tail_cnt;

    eca_fit_affine_finish(
        raw_b2, n_cnt, step_scale, b_ref, y_ref, tot_u, tot_v, tot_uu, tot_uv,
    )
}

/// True 1-pass fused subpixel pan/streak/ECA/pyr_eca/motion-grid kernel.
/// Returns `None` without mutating buffers if preconditions aren't met or on ECA cold-start,
/// allowing callers to fall back to the 2-pass pan pipeline cleanly.
#[inline(never)]
pub unsafe fn pan_streak_and_eca_fused_subpx_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    mask_warm: bool,
    mask_luma_tol: f32,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    scratch_ptr: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    adapt_rate: f32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    frac_mag: f32,
    exp_gain: f32,
    exp_off: f32,
    exp_regime: bool,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> Option<f32> {
    if frac_mag < 0.04
        || grid_w > 40
        || frame_w != grid_w * 16
        || frame_w > 640
        || frame_h < grid_h * 16
        || (mask_warm && mask_luma_tol > 0.045)
    {
        return None;
    }

    let (ix0, iy0, wx0, wx1, wy0, wy1) = bilinear_weights_1d_u16(pan_dx_f, pan_dy_f);
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let min_bx = 2 - ix0;
    let max_bx = w_i - 19 - ix0;
    let min_by = 2 - iy0;
    let max_by = h_i - 18 - iy0;
    let mut gx_lo = 0usize;
    while gx_lo < grid_w && ((gx_lo * 16) as i32) < min_bx {
        gx_lo += 1;
    }
    let mut gx_hi = gx_lo;
    while gx_hi < grid_w && ((gx_hi * 16) as i32) <= max_bx {
        gx_hi += 1;
    }
    let mut gy_lo = 0usize;
    while gy_lo < grid_h && ((gy_lo * 16) as i32) < min_by {
        gy_lo += 1;
    }
    let mut gy_hi = gy_lo;
    while gy_hi < grid_h && ((gy_hi * 16) as i32) <= max_by {
        gy_hi += 1;
    }
    if gx_lo >= gx_hi || gy_lo >= gy_hi {
        return None;
    }

    let len = frame_w * frame_h;
    let (exp_active, dg_q15, off_i16) = exposure_q15_params(exp_gain, exp_off);
    let eff_exp_regime = exp_regime || exp_active;
    let has_mask = mask_warm && !eval_mask_ptr.is_null();
    let filter_mask = has_mask && mask_luma_tol > 0.0;

    let (alpha, offset_c) = if has_mask && !filter_mask {
        eca_fit_affine_params_pan_subpx_live_simd128(
            luma_ptr,
            bg_luma_ptr,
            curr_rgba,
            prev_rgba,
            stat_streak_ptr,
            frame_w,
            frame_h,
            scale_px,
            pan_dx_f,
            pan_dy_f,
            ix0,
            iy0,
            wx0,
            wx1,
            wy0,
            wy1,
            exp_active,
            eff_exp_regime,
            dg_q15,
            off_i16,
        )?
    } else {
        eca_fit_affine_params_simd128(
            luma_ptr,
            bg_luma_ptr,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            len,
        )?
    };

    macro_rules! call_1pass {
        ($hm:expr, $eh:expr, $fm:expr, $ea:expr, $ee:expr) => {
            pan_streak_and_eca_fused_subpx_impl::<$hm, $eh, $fm, $ea, $ee>(
                curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, stat_streak_ptr, eval_mask_ptr,
                mask_luma_tol, eca_res_ptr, pyr_eca_ptr, grid_out, scratch_ptr,
                frame_w, frame_h, grid_w, grid_h, scale_px, alpha, offset_c, adapt_rate,
                pan_dx_f, pan_dy_f, dg_q15, off_i16, ix0, iy0, wx0, wx1, wy0, wy1,
                gx_lo, gx_hi, gy_lo, gy_hi, out_err2, out_sig2, out_eval_count,
            )
        };
    }
    let ratio = match (has_mask, filter_mask, exp_active, eff_exp_regime) {
        (true, true, false, false) => call_1pass!(true, true, true, false, false),
        (true, false, false, false) => call_1pass!(true, true, false, false, false),
        (true, true, false, true) => call_1pass!(true, false, true, false, true),
        (true, false, false, true) => call_1pass!(true, false, false, false, true),
        (true, true, true, _) => call_1pass!(true, false, true, true, true),
        (true, false, true, _) => call_1pass!(true, false, false, true, true),
        (false, _, false, false) => call_1pass!(false, false, false, false, false),
        (false, _, false, true) => call_1pass!(false, false, false, false, true),
        (false, _, true, _) => call_1pass!(false, false, false, true, true),
    };
    Some(ratio)
}

#[inline(always)]
unsafe fn pan_int_eval_bit_scalar(
    p: usize,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *const u8,
    frame_w: usize,
    frame_h: usize,
    pan_dx: i32,
    pan_dy: i32,
    exp_active: bool,
    eff_exp_regime: bool,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
) -> bool {
    let y = p / frame_w;
    let x = p - y * frame_w;
    let sy = (y as i32) - pan_dy;
    let sx = (x as i32) - pan_dx;
    if sy < 0 || sy >= frame_h as i32 || sx < 0 || sx >= frame_w as i32 {
        return false;
    }
    let c_b = p * 4;
    let p_b = ((sy as usize) * frame_w + (sx as usize)) * 4;
    let pr_r = *prev_rgba.add(p_b);
    let pr_g = *prev_rgba.add(p_b + 1);
    let white_clip = eff_exp_regime
        && ((pr_r >= 251 && pr_g >= 251)
            || (*curr_rgba.add(c_b) >= 253 && *curr_rgba.add(c_b + 1) >= 253));
    let mut max_d = 0u8;
    for ch in 0..3usize {
        let pr_raw = *prev_rgba.add(p_b + ch);
        let c_ch = *curr_rgba.add(c_b + ch);
        let pv = if exp_active {
            apply_exposure_u8_scalar(pr_raw, dg_q15, off_i16)
        } else {
            pr_raw
        };
        let d = c_ch.abs_diff(pv);
        if ch == 2 && eff_exp_regime && (pr_raw >= 250 || c_ch >= 250) && pr_r.min(pr_g) < 230 {
            continue;
        }
        if d > max_d {
            max_d = d;
        }
    }
    let mid_u8 = hard_u8 + 1;
    let soft_u8 = hard_u8 + 4;
    if white_clip && max_d <= hard_u8 {
        max_d = mid_u8;
    }
    let old_s = *stat_streak_ptr.add(p);
    let ns = if max_d <= hard_u8 {
        old_s.saturating_add(2).min(20)
    } else if max_d <= mid_u8 {
        old_s.saturating_sub(1)
    } else if max_d <= soft_u8 {
        old_s.min(14).saturating_sub(6)
    } else {
        0
    };
    ns >= 12 && !white_clip
}

#[inline(always)]
unsafe fn pan_int_block16_streak<
    const EFF_EXP: bool,
    const COMPUTE_RGB: bool,
    const MUTATE: bool,
>(
    c_off: usize,
    p_off: usize,
    stat_ptr: *mut u8,
    eval_ptr: *mut u8,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    exp_active: bool,
    dg_q15_v: v128,
    off_i16_v: v128,
    rgb_mask: v128,
    rg_mask: v128,
    ff_mask: v128,
    b_mask: v128,
    b_250: v128,
    splat_230: v128,
    splat_251: v128,
    splat_253: v128,
    tol_hard: v128,
    tol_mid: v128,
    tol_soft: v128,
    zero_i32: v128,
    one_u8: v128,
    two_u8: v128,
    six_u8: v128,
    fourteen_u8: v128,
    twelve_u8: v128,
    twenty_u8: v128,
) -> (v128, v128, v128) {
    let c0 = v128_load(curr_rgba.add(c_off) as *const v128);
    let p0_raw = v128_load(prev_rgba.add(p_off) as *const v128);
    let c1 = v128_load(curr_rgba.add(c_off + 16) as *const v128);
    let p1_raw = v128_load(prev_rgba.add(p_off + 16) as *const v128);
    let c2 = v128_load(curr_rgba.add(c_off + 32) as *const v128);
    let p2_raw = v128_load(prev_rgba.add(p_off + 32) as *const v128);
    let c3 = v128_load(curr_rgba.add(c_off + 48) as *const v128);
    let p3_raw = v128_load(prev_rgba.add(p_off + 48) as *const v128);
    let (p0, p1, p2, p3) = if EFF_EXP && exp_active {
        (
            apply_exposure_rgba4_simd128(p0_raw, dg_q15_v, off_i16_v),
            apply_exposure_rgba4_simd128(p1_raw, dg_q15_v, off_i16_v),
            apply_exposure_rgba4_simd128(p2_raw, dg_q15_v, off_i16_v),
            apply_exposure_rgba4_simd128(p3_raw, dg_q15_v, off_i16_v),
        )
    } else {
        (p0_raw, p1_raw, p2_raw, p3_raw)
    };
    let mut abs0 = v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask);
    let mut abs1 = v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask);
    let mut abs2 = v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask);
    let mut abs3 = v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask);

    let (rgb01, rgb23) = if COMPUTE_RGB {
        let u0 = abs_rgb_to_sum_u32x4_simd128(abs0);
        let u1 = abs_rgb_to_sum_u32x4_simd128(abs1);
        let u2 = abs_rgb_to_sum_u32x4_simd128(abs2);
        let u3 = abs_rgb_to_sum_u32x4_simd128(abs3);
        (i16x8_narrow_i32x4(u0, u1), i16x8_narrow_i32x4(u2, u3))
    } else {
        (zero_i32, zero_i32)
    };

    let wc16 = if EFF_EXP {
        abs0 = drop_b_royal_blue_simd128(abs0, c0, p0_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
        abs1 = drop_b_royal_blue_simd128(abs1, c1, p1_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
        abs2 = drop_b_royal_blue_simd128(abs2, c2, p2_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
        abs3 = drop_b_royal_blue_simd128(abs3, c3, p3_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
        let wc0 = white_clip_mask_u32_simd128(c0, p0_raw, rg_mask, splat_251, splat_253);
        let wc1 = white_clip_mask_u32_simd128(c1, p1_raw, rg_mask, splat_251, splat_253);
        let wc2 = white_clip_mask_u32_simd128(c2, p2_raw, rg_mask, splat_251, splat_253);
        let wc3 = white_clip_mask_u32_simd128(c3, p3_raw, rg_mask, splat_251, splat_253);
        i8x16_narrow_i16x8(i16x8_narrow_i32x4(wc0, wc1), i16x8_narrow_i32x4(wc2, wc3))
    } else {
        zero_i32
    };

    let max_d16 = pack4_max_u8_16_simd128(abs0, abs1, abs2, abs3, ff_mask);
    let old_s = v128_load(stat_ptr as *const v128);
    let (new_s, eval16) = hysteresis_pan_16_next_simd128(
        max_d16,
        None,
        wc16,
        tol_hard,
        tol_mid,
        tol_soft,
        one_u8,
        two_u8,
        six_u8,
        fourteen_u8,
        twelve_u8,
        twenty_u8,
        EFF_EXP,
        old_s,
    );
    if MUTATE {
        v128_store(stat_ptr as *mut v128, new_s);
        v128_store(eval_ptr as *mut v128, eval16);
    }
    (rgb01, rgb23, eval16)
}

#[inline(always)]
unsafe fn pan_int_eval_4_live_i32x4(
    cb: usize,
    pb: usize,
    old_s: v128,
    lane_quarter: usize,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    exp_active: bool,
    eff_exp: bool,
    dg_q15_v: v128,
    off_i16_v: v128,
    rgb_mask: v128,
    rg_mask: v128,
    ff_mask: v128,
    b_mask: v128,
    b_250: v128,
    splat_230: v128,
    splat_251: v128,
    splat_253: v128,
    tol_hard: v128,
    tol_mid: v128,
    tol_soft: v128,
    zero_i32: v128,
    one_u8: v128,
    two_u8: v128,
    six_u8: v128,
    fourteen_u8: v128,
    twelve_u8: v128,
    twenty_u8: v128,
) -> v128 {
    let c = v128_load(curr_rgba.add(cb) as *const v128);
    let p_raw = v128_load(prev_rgba.add(pb) as *const v128);
    let p = if exp_active {
        apply_exposure_rgba4_simd128(p_raw, dg_q15_v, off_i16_v)
    } else {
        p_raw
    };
    let mut abs = v128_and(v128_or(u8x16_sub_sat(c, p), u8x16_sub_sat(p, c)), rgb_mask);
    let wc32 = if eff_exp {
        abs = drop_b_royal_blue_simd128(abs, c, p_raw, rgb_mask, rg_mask, ff_mask, b_mask, b_250, splat_230);
        pack_wc4_to_i32(white_clip_mask_u32_simd128(c, p_raw, rg_mask, splat_251, splat_253))
    } else {
        0
    };
    let mx32 = pack_max_rgb4_to_i32(abs);
    let shift_bytes = old_s;
    let mx_v = i32x4_splat(mx32);
    let wc_v = i32x4_splat(wc32);
    let old_q = match lane_quarter {
        0 => shift_bytes,
        1 => i32x4_shuffle::<1, 1, 1, 1>(shift_bytes, shift_bytes),
        2 => i32x4_shuffle::<2, 2, 2, 2>(shift_bytes, shift_bytes),
        _ => i32x4_shuffle::<3, 3, 3, 3>(shift_bytes, shift_bytes),
    };
    let (_, eval16) = hysteresis_pan_16_next_simd128(
        mx_v,
        None,
        if eff_exp { wc_v } else { zero_i32 },
        tol_hard,
        tol_mid,
        tol_soft,
        one_u8,
        two_u8,
        six_u8,
        fourteen_u8,
        twelve_u8,
        twenty_u8,
        eff_exp,
        old_q,
    );
    i32x4_extend_low_i16x8(i16x8_extend_low_i8x16(eval16))
}

#[inline(never)]
unsafe fn eca_fit_affine_params_pan_int_live_simd128(
    luma_ptr: *const f32,
    bg_luma_ptr: *const f32,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *const u8,
    frame_w: usize,
    frame_h: usize,
    pan_dx: i32,
    pan_dy: i32,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
    exp_active: bool,
    eff_exp_regime: bool,
) -> Option<(f32, f32)> {
    let len = frame_w * frame_h;
    let guard_thresh = f32x4_splat(0.045);
    let guard_stat = f32x4_splat(0.15);

    let mut b_ref = 0.5f32;
    let mut y_ref = 0.5f32;
    for k in 0..len {
        let bk = *bg_luma_ptr.add(k);
        let yk = *luma_ptr.add(k);
        let ad = (yk - bk).abs();
        if ad < 0.045f32
            || (ad < 0.15f32
                && pan_int_eval_bit_scalar(
                    k,
                    curr_rgba,
                    prev_rgba,
                    stat_streak_ptr,
                    frame_w,
                    frame_h,
                    pan_dx,
                    pan_dy,
                    exp_active,
                    eff_exp_regime,
                    dg_q15,
                    off_i16,
                    hard_u8,
                ))
        {
            b_ref = bk;
            y_ref = yk;
            break;
        }
    }
    let b_ref_vec = f32x4_splat(b_ref);
    let y_ref_vec = f32x4_splat(y_ref);

    let mut acc_raw_b2_0 = f32x4_splat(0.0);
    let mut acc_raw_b2_1 = f32x4_splat(0.0);
    let mut acc_cnt_i32_0 = i32x4_splat(0);
    let mut acc_cnt_i32_1 = i32x4_splat(0);
    let mut tot_u = 0.0f64;
    let mut tot_v = 0.0f64;
    let mut tot_uu = 0.0f64;
    let mut tot_uv = 0.0f64;
    let mut c_u0 = f32x4_splat(0.0);
    let mut c_v0 = f32x4_splat(0.0);
    let mut c_uu0 = f32x4_splat(0.0);
    let mut c_uv0 = f32x4_splat(0.0);
    let mut c_u1 = f32x4_splat(0.0);
    let mut c_v1 = f32x4_splat(0.0);
    let mut c_uu1 = f32x4_splat(0.0);
    let mut c_uv1 = f32x4_splat(0.0);

    let use_stride7 = len >= 131_072;
    let use_stride3 = !use_stride7 && len >= 32768;
    let step16 = if use_stride7 { 112usize } else if use_stride3 { 48usize } else { 16usize };
    let step_scale = if use_stride7 { 7i64 } else if use_stride3 { 3i64 } else { 1i64 };

    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let rg_mask = u32x4_splat(0x0000_FFFF);
    let ff_mask = u32x4_splat(0xFF);
    let b_mask = u32x4_splat(0x00FF_0000);
    let b_250 = u32x4_splat(250 << 16);
    let splat_230 = u32x4_splat(230);
    let splat_251 = u8x16_splat(251);
    let splat_253 = u8x16_splat(253);
    let zero_i32 = i32x4_splat(0);
    let four_i32 = i32x4_splat(4);
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let ten_u8 = u8x16_splat(10);
    let twelve_u8 = u8x16_splat(12);
    let fourteen_u8 = u8x16_splat(14);
    let twenty_u8 = u8x16_splat(20);
    let dg_q15_v = i16x8_splat(dg_q15);
    let off_i16_v = i16x8_splat(off_i16);
    let tol_hard = u8x16_splat(hard_u8);
    let tol_mid = u8x16_splat(hard_u8 + 1);
    let tol_soft = u8x16_splat(hard_u8 + 4);
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;

    let mut i = 0usize;
    let mut blk_rem = 64usize;
    while i + 16 <= len {
        let y0 = v128_load(luma_ptr.add(i) as *const v128);
        let b0 = v128_load(bg_luma_ptr.add(i) as *const v128);
        let y1 = v128_load(luma_ptr.add(i + 4) as *const v128);
        let b1 = v128_load(bg_luma_ptr.add(i + 4) as *const v128);
        let y2 = v128_load(luma_ptr.add(i + 8) as *const v128);
        let b2 = v128_load(bg_luma_ptr.add(i + 8) as *const v128);
        let y3 = v128_load(luma_ptr.add(i + 12) as *const v128);
        let b3 = v128_load(bg_luma_ptr.add(i + 12) as *const v128);

        let d0 = f32x4_abs(f32x4_sub(y0, b0));
        let d1 = f32x4_abs(f32x4_sub(y1, b1));
        let d2 = f32x4_abs(f32x4_sub(y2, b2));
        let d3 = f32x4_abs(f32x4_sub(y3, b3));

        let mut w0 = f32x4_lt(d0, guard_thresh);
        let mut w1 = f32x4_lt(d1, guard_thresh);
        let mut w2 = f32x4_lt(d2, guard_thresh);
        let mut w3 = f32x4_lt(d3, guard_thresh);

        if i32x4_all_true(v128_and(v128_and(w0, w1), v128_and(w2, w3))) {
            acc_raw_b2_0 = f32x4_add(f32x4_add(acc_raw_b2_0, f32x4_mul(b0, b0)), f32x4_mul(b2, b2));
            acc_raw_b2_1 = f32x4_add(f32x4_add(acc_raw_b2_1, f32x4_mul(b1, b1)), f32x4_mul(b3, b3));
            acc_cnt_i32_0 = i32x4_add(acc_cnt_i32_0, four_i32);
            let u0 = f32x4_sub(b0, b_ref_vec);
            let v0 = f32x4_sub(y0, y_ref_vec);
            let u2 = f32x4_sub(b2, b_ref_vec);
            let v2 = f32x4_sub(y2, y_ref_vec);
            c_u0 = f32x4_add(f32x4_add(c_u0, u0), u2);
            c_v0 = f32x4_add(f32x4_add(c_v0, v0), v2);
            c_uu0 = f32x4_add(f32x4_add(c_uu0, f32x4_mul(u0, u0)), f32x4_mul(u2, u2));
            c_uv0 = f32x4_add(f32x4_add(c_uv0, f32x4_mul(u0, v0)), f32x4_mul(u2, v2));
            let u1 = f32x4_sub(b1, b_ref_vec);
            let v1 = f32x4_sub(y1, y_ref_vec);
            let u3 = f32x4_sub(b3, b_ref_vec);
            let v3 = f32x4_sub(y3, y_ref_vec);
            c_u1 = f32x4_add(f32x4_add(c_u1, u1), u3);
            c_v1 = f32x4_add(f32x4_add(c_v1, v1), v3);
            c_uu1 = f32x4_add(f32x4_add(c_uu1, f32x4_mul(u1, u1)), f32x4_mul(u3, u3));
            c_uv1 = f32x4_add(f32x4_add(c_uv1, f32x4_mul(u1, v1)), f32x4_mul(u3, v3));
        } else {
            let mid0 = v128_andnot(f32x4_lt(d0, guard_stat), w0);
            let mid1 = v128_andnot(f32x4_lt(d1, guard_stat), w1);
            let mid2 = v128_andnot(f32x4_lt(d2, guard_stat), w2);
            let mid3 = v128_andnot(f32x4_lt(d3, guard_stat), w3);
            let mid_or = v128_or(v128_or(mid0, mid1), v128_or(mid2, mid3));
            if v128_any_true(mid_or) {
                let old_s = v128_load(stat_streak_ptr.add(i) as *const v128);
                let ge10 = u8x16_ge(old_s, ten_u8);
                let mid16 = i8x16_narrow_i16x8(
                    i16x8_narrow_i32x4(mid0, mid1),
                    i16x8_narrow_i32x4(mid2, mid3),
                );
                let cmask = i8x16_bitmask(v128_and(mid16, ge10));
                if cmask != 0 {
                    let y = i / frame_w;
                    let x = i - y * frame_w;
                    let sy = (y as i32) - pan_dy;
                    let sx = (x as i32) - pan_dx;
                    if sy >= 0 && sy < h_i && sx >= 0 && sx + 16 <= w_i && x + 16 <= frame_w {
                        let cb = i * 4;
                        let pb = ((sy as usize) * frame_w + (sx as usize)) * 4;
                        macro_rules! eval4_q {
                            ($q:expr, $wv:expr, $midv:expr) => {{
                                let ev = pan_int_eval_4_live_i32x4(
                                    cb + $q * 16, pb + $q * 16, old_s, $q, curr_rgba, prev_rgba,
                                    exp_active, eff_exp_regime, dg_q15_v, off_i16_v, rgb_mask,
                                    rg_mask, ff_mask, b_mask, b_250, splat_230, splat_251,
                                    splat_253, tol_hard, tol_mid, tol_soft, zero_i32, one_u8,
                                    two_u8, six_u8, fourteen_u8, twelve_u8, twenty_u8,
                                );
                                $wv = v128_or($wv, v128_and($midv, ev));
                            }};
                        }
                        if (cmask & 0x000F) != 0 { eval4_q!(0, w0, mid0); }
                        if (cmask & 0x00F0) != 0 { eval4_q!(1, w1, mid1); }
                        if (cmask & 0x0F00) != 0 { eval4_q!(2, w2, mid2); }
                        if (cmask & 0xF000) != 0 { eval4_q!(3, w3, mid3); }
                    } else {
                        let mut bits = [0u8; 16];
                        for k in 0..16usize {
                            if ((cmask >> k) & 1) != 0
                                && pan_int_eval_bit_scalar(
                                    i + k, curr_rgba, prev_rgba, stat_streak_ptr,
                                    frame_w, frame_h, pan_dx, pan_dy, exp_active,
                                    eff_exp_regime, dg_q15, off_i16, hard_u8,
                                )
                            {
                                bits[k] = 0xFF;
                            }
                        }
                        let eval16 = v128_load(bits.as_ptr() as *const v128);
                        let lo8 = i16x8_extend_low_i8x16(eval16);
                        let hi8 = i16x8_extend_high_i8x16(eval16);
                        w0 = v128_or(w0, v128_and(mid0, i32x4_extend_low_i16x8(lo8)));
                        w1 = v128_or(w1, v128_and(mid1, i32x4_extend_high_i16x8(lo8)));
                        w2 = v128_or(w2, v128_and(mid2, i32x4_extend_low_i16x8(hi8)));
                        w3 = v128_or(w3, v128_and(mid3, i32x4_extend_high_i16x8(hi8)));
                    }
                }
            }
            macro_rules! accum_live_lane {
                ($lane:tt, $yv:expr, $bv:expr, $wv:expr) => {{
                    let um = v128_and(f32x4_sub($bv, b_ref_vec), $wv);
                    let vm = v128_and(f32x4_sub($yv, y_ref_vec), $wv);
                    if $lane == 0 {
                        acc_raw_b2_0 = f32x4_add(acc_raw_b2_0, f32x4_mul($bv, $bv));
                        acc_cnt_i32_0 = i32x4_sub(acc_cnt_i32_0, $wv);
                        c_u0 = f32x4_add(c_u0, um);
                        c_v0 = f32x4_add(c_v0, vm);
                        c_uu0 = f32x4_add(c_uu0, f32x4_mul(um, um));
                        c_uv0 = f32x4_add(c_uv0, f32x4_mul(um, vm));
                    } else {
                        acc_raw_b2_1 = f32x4_add(acc_raw_b2_1, f32x4_mul($bv, $bv));
                        acc_cnt_i32_1 = i32x4_sub(acc_cnt_i32_1, $wv);
                        c_u1 = f32x4_add(c_u1, um);
                        c_v1 = f32x4_add(c_v1, vm);
                        c_uu1 = f32x4_add(c_uu1, f32x4_mul(um, um));
                        c_uv1 = f32x4_add(c_uv1, f32x4_mul(um, vm));
                    }
                }};
            }
            accum_live_lane!(0, y0, b0, w0);
            accum_live_lane!(1, y1, b1, w1);
            accum_live_lane!(0, y2, b2, w2);
            accum_live_lane!(1, y3, b3, w3);
        }

        i += step16;
        blk_rem -= 1;
        if blk_rem == 0 {
            tot_u += hsum_f32x4(f32x4_add(c_u0, c_u1)) as f64;
            tot_v += hsum_f32x4(f32x4_add(c_v0, c_v1)) as f64;
            tot_uu += hsum_f32x4(f32x4_add(c_uu0, c_uu1)) as f64;
            tot_uv += hsum_f32x4(f32x4_add(c_uv0, c_uv1)) as f64;
            c_u0 = f32x4_splat(0.0);
            c_v0 = f32x4_splat(0.0);
            c_uu0 = f32x4_splat(0.0);
            c_uv0 = f32x4_splat(0.0);
            c_u1 = f32x4_splat(0.0);
            c_v1 = f32x4_splat(0.0);
            c_uu1 = f32x4_splat(0.0);
            c_uv1 = f32x4_splat(0.0);
            blk_rem = 64;
        }
    }
    let mut tail_raw_b2 = 0.0f32;
    let mut tail_cnt = 0i64;
    let mut tail_u = 0.0f64;
    let mut tail_v = 0.0f64;
    let mut tail_uu = 0.0f64;
    let mut tail_uv = 0.0f64;
    while i < len {
        let y = *luma_ptr.add(i);
        let b = *bg_luma_ptr.add(i);
        tail_raw_b2 += b * b;
        let ad = (y - b).abs();
        if ad < 0.045f32
            || (ad < 0.15f32
                && pan_int_eval_bit_scalar(
                    i, curr_rgba, prev_rgba, stat_streak_ptr, frame_w, frame_h,
                    pan_dx, pan_dy, exp_active, eff_exp_regime, dg_q15, off_i16, hard_u8,
                ))
        {
            let u = (b - b_ref) as f64;
            let v = (y - y_ref) as f64;
            tail_cnt += 1;
            tail_u += u;
            tail_v += v;
            tail_uu += u * u;
            tail_uv += u * v;
        }
        i += 1;
    }

    tot_u += hsum_f32x4(f32x4_add(c_u0, c_u1)) as f64 + tail_u;
    tot_v += hsum_f32x4(f32x4_add(c_v0, c_v1)) as f64 + tail_v;
    tot_uu += hsum_f32x4(f32x4_add(c_uu0, c_uu1)) as f64 + tail_uu;
    tot_uv += hsum_f32x4(f32x4_add(c_uv0, c_uv1)) as f64 + tail_uv;
    let acc_cnt_i32 = i32x4_add(acc_cnt_i32_0, acc_cnt_i32_1);
    let raw_b2 = hsum_f32x4(f32x4_add(acc_raw_b2_0, acc_raw_b2_1)) + tail_raw_b2;
    let n_cnt = (i32x4_extract_lane::<0>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<1>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<2>(acc_cnt_i32) as i64)
        + (i32x4_extract_lane::<3>(acc_cnt_i32) as i64)
        + tail_cnt;

    eca_fit_affine_finish(
        raw_b2, n_cnt, step_scale, b_ref, y_ref, tot_u, tot_v, tot_uu, tot_uv,
    )
}

#[inline(never)]
unsafe fn pan_int_block16_edge<const WRITE_RGB: bool>(
    row_c: usize,
    row_p_valid: bool,
    row_p: usize,
    x0: usize,
    frame_w: usize,
    pan_dx: i32,
    exp_active: bool,
    eff_exp: bool,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_sum_out: *mut f32,
) {
    if !row_p_valid {
        let p = row_c + x0;
        let z = i32x4_splat(0);
        v128_store(stat_streak_ptr.add(p) as *mut v128, z);
        v128_store(eval_mask_ptr.add(p) as *mut v128, z);
        if WRITE_RGB {
            let zf = f32x4_splat(0.0);
            v128_store(rgb_sum_out.add(p) as *mut v128, zf);
            v128_store(rgb_sum_out.add(p + 4) as *mut v128, zf);
            v128_store(rgb_sum_out.add(p + 8) as *mut v128, zf);
            v128_store(rgb_sum_out.add(p + 12) as *mut v128, zf);
        }
        return;
    }
    let w_i = frame_w as i32;
    let mid_u8 = hard_u8 + 1;
    let soft_u8 = hard_u8 + 4;
    for k in 0..16usize {
        let x = x0 + k;
        let idx = row_c + x;
        let sx = (x as i32) - pan_dx;
        if sx >= 0 && sx < w_i {
            let c_b = idx * 4;
            let p_b = (row_p + (sx as usize)) * 4;
            let pr_r = *prev_rgba.add(p_b);
            let pr_g = *prev_rgba.add(p_b + 1);
            let white_clip = eff_exp
                && ((pr_r >= 251 && pr_g >= 251)
                    || (*curr_rgba.add(c_b) >= 253 && *curr_rgba.add(c_b + 1) >= 253));
            let mut max_d = 0u8;
            let mut diff_sum = 0.0f32;
            for ch in 0..3usize {
                let pr_raw = *prev_rgba.add(p_b + ch);
                let c_ch = *curr_rgba.add(c_b + ch);
                let pv = if exp_active {
                    apply_exposure_u8_scalar(pr_raw, dg_q15, off_i16)
                } else {
                    pr_raw
                };
                let d = c_ch.abs_diff(pv);
                diff_sum += d as f32;
                if ch == 2 && eff_exp && (pr_raw >= 250 || c_ch >= 250) && pr_r.min(pr_g) < 230 {
                    continue;
                }
                if d > max_d {
                    max_d = d;
                }
            }
            if WRITE_RGB {
                *rgb_sum_out.add(idx) = diff_sum;
            }
            if white_clip && max_d <= hard_u8 {
                max_d = mid_u8;
            }
            hysteresis_pan_scalar(
                stat_streak_ptr.add(idx),
                eval_mask_ptr.add(idx),
                max_d,
                2u8,
                hard_u8,
                mid_u8,
                soft_u8,
                white_clip,
            );
        } else {
            *stat_streak_ptr.add(idx) = 0;
            *eval_mask_ptr.add(idx) = 0;
            if WRITE_RGB {
                *rgb_sum_out.add(idx) = 0.0;
            }
        }
    }
}

#[inline(never)]
unsafe fn pan_int_streak_row_u16<const EFF_EXP: bool>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    rgb_row_ptr: *mut u16,
    y: usize,
    frame_w: usize,
    grid_w: usize,
    w_i: i32,
    h_i: i32,
    pan_dx: i32,
    pan_dy: i32,
    int_lo: usize,
    int_hi: usize,
    exp_active: bool,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
) {
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let rg_mask = u32x4_splat(0x0000_FFFF);
    let ff_mask = u32x4_splat(0xFF);
    let b_mask = u32x4_splat(0x00FF_0000);
    let b_250 = u32x4_splat(250 << 16);
    let splat_230 = u32x4_splat(230);
    let splat_251 = u8x16_splat(251);
    let splat_253 = u8x16_splat(253);
    let zero_i32 = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let twelve_u8 = u8x16_splat(12);
    let fourteen_u8 = u8x16_splat(14);
    let twenty_u8 = u8x16_splat(20);
    let dg_q15_v = i16x8_splat(dg_q15);
    let off_i16_v = i16x8_splat(off_i16);
    let tol_hard = u8x16_splat(hard_u8);
    let tol_mid = u8x16_splat(hard_u8 + 1);
    let tol_soft = u8x16_splat(hard_u8 + 4);

    let sy = (y as i32) - pan_dy;
    let row_c = y * frame_w;
    let row_p_valid = sy >= 0 && sy < h_i;
    let row_p = if row_p_valid { (sy as usize) * frame_w } else { 0 };

    macro_rules! run_border_gx {
        ($gx:expr) => {{
            let x = $gx * 16;
            let row_off = row_c + x;
            let sx = (x as i32) - pan_dx;
            if row_p_valid && sx >= 0 && sx + 16 <= w_i {
                let p_off = (row_p + (sx as usize)) * 4;
                pan_int_block16_streak::<EFF_EXP, false, true>(
                    row_off * 4, p_off, stat_streak_ptr.add(row_off), eval_mask_ptr.add(row_off),
                    curr_rgba, prev_rgba, exp_active, dg_q15_v, off_i16_v, rgb_mask,
                    rg_mask, ff_mask, b_mask, b_250, splat_230, splat_251, splat_253,
                    tol_hard, tol_mid, tol_soft, zero_i32, one_u8, two_u8, six_u8,
                    fourteen_u8, twelve_u8, twenty_u8,
                );
            } else {
                pan_int_block16_edge::<false>(
                    row_c, row_p_valid, row_p, x, frame_w, pan_dx, exp_active, EFF_EXP,
                    dg_q15, off_i16, hard_u8, curr_rgba, prev_rgba, stat_streak_ptr,
                    eval_mask_ptr, core::ptr::null_mut(),
                );
            }
        }};
    }

    for gx in 0..int_lo {
        run_border_gx!(gx);
    }
    if int_lo < int_hi {
        for gx in int_lo..int_hi {
            let x = gx * 16;
            let sx = (x as i32) - pan_dx;
            let row_off = row_c + x;
            let p_off = (row_p + (sx as usize)) * 4;
            let (rgb01, rgb23, _) = pan_int_block16_streak::<EFF_EXP, true, true>(
                row_off * 4, p_off, stat_streak_ptr.add(row_off), eval_mask_ptr.add(row_off),
                curr_rgba, prev_rgba, exp_active, dg_q15_v, off_i16_v, rgb_mask,
                rg_mask, ff_mask, b_mask, b_250, splat_230, splat_251, splat_253,
                tol_hard, tol_mid, tol_soft, zero_i32, one_u8, two_u8, six_u8,
                fourteen_u8, twelve_u8, twenty_u8,
            );
            let rgb_vp = rgb_row_ptr.add(x) as *mut v128;
            v128_store(rgb_vp, rgb01);
            v128_store(rgb_vp.add(1), rgb23);
        }
    }
    for gx in int_hi.max(int_lo)..grid_w {
        run_border_gx!(gx);
    }
}

#[inline(never)]
unsafe fn pan_int_streak_tail_rows<const EFF_EXP: bool>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    w_i: i32,
    h_i: i32,
    pan_dx: i32,
    pan_dy: i32,
    exp_active: bool,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
) {
    let rgb_mask = u32x4_splat(0x00FF_FFFF);
    let rg_mask = u32x4_splat(0x0000_FFFF);
    let ff_mask = u32x4_splat(0xFF);
    let b_mask = u32x4_splat(0x00FF_0000);
    let b_250 = u32x4_splat(250 << 16);
    let splat_230 = u32x4_splat(230);
    let splat_251 = u8x16_splat(251);
    let splat_253 = u8x16_splat(253);
    let zero_i32 = i32x4_splat(0);
    let one_u8 = u8x16_splat(1);
    let two_u8 = u8x16_splat(2);
    let six_u8 = u8x16_splat(6);
    let twelve_u8 = u8x16_splat(12);
    let fourteen_u8 = u8x16_splat(14);
    let twenty_u8 = u8x16_splat(20);
    let dg_q15_v = i16x8_splat(dg_q15);
    let off_i16_v = i16x8_splat(off_i16);
    let tol_hard = u8x16_splat(hard_u8);
    let tol_mid = u8x16_splat(hard_u8 + 1);
    let tol_soft = u8x16_splat(hard_u8 + 4);

    for y in (grid_h * 16)..frame_h {
        let sy = (y as i32) - pan_dy;
        let row_c = y * frame_w;
        let row_p_valid = sy >= 0 && sy < h_i;
        let row_p = if row_p_valid { (sy as usize) * frame_w } else { 0 };
        for gx in 0..grid_w {
            let x = gx * 16;
            let sx = (x as i32) - pan_dx;
            let row_off = row_c + x;
            if row_p_valid && sx >= 0 && sx + 16 <= w_i {
                let p_off = (row_p + (sx as usize)) * 4;
                let (rgb01, rgb23, _) = pan_int_block16_streak::<EFF_EXP, true, true>(
                    row_off * 4, p_off, stat_streak_ptr.add(row_off), eval_mask_ptr.add(row_off),
                    curr_rgba, prev_rgba, exp_active, dg_q15_v, off_i16_v, rgb_mask,
                    rg_mask, ff_mask, b_mask, b_250, splat_230, splat_251, splat_253,
                    tol_hard, tol_mid, tol_soft, zero_i32, one_u8, two_u8, six_u8,
                    fourteen_u8, twelve_u8, twenty_u8,
                );
                v128_store(eca_res_ptr.add(row_off) as *mut v128, f32x4_convert_u32x4(u32x4_extend_low_u16x8(rgb01)));
                v128_store(eca_res_ptr.add(row_off + 4) as *mut v128, f32x4_convert_u32x4(u32x4_extend_high_u16x8(rgb01)));
                v128_store(eca_res_ptr.add(row_off + 8) as *mut v128, f32x4_convert_u32x4(u32x4_extend_low_u16x8(rgb23)));
                v128_store(eca_res_ptr.add(row_off + 12) as *mut v128, f32x4_convert_u32x4(u32x4_extend_high_u16x8(rgb23)));
            } else {
                pan_int_block16_edge::<true>(
                    row_c, row_p_valid, row_p, x, frame_w, pan_dx, exp_active, EFF_EXP,
                    dg_q15, off_i16, hard_u8, curr_rgba, prev_rgba, stat_streak_ptr,
                    eval_mask_ptr, eca_res_ptr,
                );
            }
        }
    }
}

#[inline(never)]
unsafe fn pan_streak_and_eca_fused_int_impl<
    const HAS_MASK: bool,
    const ALLOW_EVAL_HEAL: bool,
    const GATE_RGB: bool,
    const FILTER_MASK: bool,
    const EFF_EXP: bool,
>(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    mask_luma_tol: f32,
    alpha: f32,
    offset_c: f32,
    adapt_rate: f32,
    heal_alpha_s: f32,
    heal_gate_s: f32,
    pan_dx: i32,
    pan_dy: i32,
    gx_lo: usize,
    gx_hi: usize,
    gy_lo: usize,
    gy_hi: usize,
    exp_active: bool,
    dg_q15: i16,
    off_i16: i16,
    hard_u8: u8,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> Option<f32> {
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;

    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let tol_v = f32x4_splat(mask_luma_tol);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);
    let heal_alpha = f32x4_splat(heal_alpha_s);
    let heal_gate_rgb = f32x4_splat(heal_gate_s);
    let w_eca = f32x4_splat(0.75);
    let w_rgb = f32x4_splat(0.25 / 765.0);
    let v_rate_eval = f32x4_splat(0.98);
    let v_eval_gate = f32x4_splat(0.062);

    let zero_v = f32x4_splat(0.0);
    let q4 = f32x4_splat(0.25);
    let q16 = f32x4_splat(1.0 / 16.0);
    let fuse_pyr = !pyr_eca_ptr.is_null() && scale_px <= 2 && grid_w == 20 * scale_px;

    let mut acc_eval_err2 = zero_v;
    let mut acc_eval_sig2 = zero_v;
    let mut acc_eval_cnt_i = i32x4_splat(0);
    let mut rgb_row_uninit = [i32x4_splat(0); 80];
    let rgb_row_ptr = rgb_row_uninit.as_mut_ptr() as *mut u16;

    if fuse_pyr && scale_px == 2 && grid_w == 40 && frame_w == 640 {
        let g_i = heal_gate_s as i32;
        let heal_gate_lim = (g_i + if (g_i as f32) < heal_gate_s { 1 } else { 0 }) as i16;
        let one_minus_heal = f32x4_sub(f32x4_splat(1.0), heal_alpha);
        let mut hd40_ctx = PanFusedHd40Ctx {
            v_alpha,
            v_c,
            tol_v,
            v_rate_bg,
            v_rate_fg,
            v_rate_bg_heal: f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_bg)),
            v_rate_fg_heal: f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_fg)),
            v_motion_gate,
            heal_gate_i16: i16x8_splat(heal_gate_lim),
            w_eca,
            w_rgb,
            v_rate_eval,
            v_rate_eval_heal: f32x4_add(heal_alpha, f32x4_mul(one_minus_heal, v_rate_eval)),
            v_eval_gate,
            q16,
            acc_eval_err2,
            acc_eval_sig2,
            acc_eval_cnt_i,
            luma_ptr,
            bg_luma_ptr,
            eval_mask_ptr,
            eca_res_ptr,
            pyr_eca_ptr,
        };
        macro_rules! run_hd40_row {
            ($gy:expr, $ry:expr, $ry_blk:expr, $int_lo:expr, $int_hi:expr, $phase:expr, $acc_r:expr, $pa_r:expr) => {
                let y = $gy * 16 + $ry;
                pan_int_streak_row_u16::<EFF_EXP>(
                    curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_row_ptr, y,
                    frame_w, 40, w_i, h_i, pan_dx, pan_dy, $int_lo, $int_hi, exp_active,
                    dg_q15, off_i16, hard_u8,
                );
                pan_fused_hd40_row::<HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK>(
                    &mut hd40_ctx,
                    $gy,
                    $ry_blk,
                    $phase,
                    y * 640,
                    $int_lo,
                    $int_hi,
                    rgb_row_ptr as *const u16,
                    &mut $acc_r,
                    &mut $pa_r,
                );
            };
        }
        let mut pyr_a_row_buf = [zero_v; 40];
        let pyr_a_row = &mut pyr_a_row_buf;
        for gy in 0..grid_h {
            let gy_valid = gy >= gy_lo && gy < gy_hi;
            let (int_lo, int_hi) = if gy_valid {
                let a = gx_lo.min(40);
                (a, gx_hi.min(40).max(a))
            } else {
                (0, 0)
            };
            let mut acc_row = [zero_v; 40];
            for ry_blk in 0..4usize {
                let ry0 = ry_blk * 4;
                run_hd40_row!(gy, ry0, ry_blk, int_lo, int_hi, 0, acc_row, *pyr_a_row);
                run_hd40_row!(gy, ry0 + 1, ry_blk, int_lo, int_hi, 1, acc_row, *pyr_a_row);
                run_hd40_row!(gy, ry0 + 2, ry_blk, int_lo, int_hi, 2, acc_row, *pyr_a_row);
                run_hd40_row!(gy, ry0 + 3, ry_blk, int_lo, int_hi, 3, acc_row, *pyr_a_row);
            }
            for gx in 0..40usize {
                *grid_out.add(gy * 40 + gx) = if gx >= int_lo && gx < int_hi {
                    hsum_f32x4(acc_row[gx])
                } else {
                    0.0
                };
            }
        }
        acc_eval_err2 = hd40_ctx.acc_eval_err2;
        acc_eval_sig2 = hd40_ctx.acc_eval_sig2;
        acc_eval_cnt_i = hd40_ctx.acc_eval_cnt_i;
    } else {
        for gy in 0..grid_h {
            let gy_valid = gy >= gy_lo && gy < gy_hi;
            let (int_lo, int_hi) = if gy_valid {
                let a = gx_lo.min(grid_w);
                (a, gx_hi.min(grid_w).max(a))
            } else {
                (0, 0)
            };
            let mut acc_row = [zero_v; 40];
            let mut pyr_a_row = [zero_v; 40];
            let mut pyr_b_row = [zero_v; 20];
            for ry in 0..16usize {
                let y = gy * 16 + ry;
                pan_int_streak_row_u16::<EFF_EXP>(
                    curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, rgb_row_ptr, y,
                    frame_w, grid_w, w_i, h_i, pan_dx, pan_dy, int_lo, int_hi, exp_active,
                    dg_q15, off_i16, hard_u8,
                );
                pan_fused_grid_le40_row::<
                    HAS_MASK, ALLOW_EVAL_HEAL, GATE_RGB, FILTER_MASK, false, 0, true, true,
                >(
                    gy, ry, ry >> 2, y * frame_w, grid_w, scale_px, fuse_pyr, int_lo, int_hi,
                    luma_ptr, bg_luma_ptr, eval_mask_ptr, eca_res_ptr, rgb_row_ptr as *const u8,
                    pyr_eca_ptr, v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate,
                    heal_alpha, heal_gate_rgb, w_eca, w_rgb, v_rate_eval, v_eval_gate,
                    zero_v, q4, q16, &mut acc_eval_err2, &mut acc_eval_sig2, &mut acc_eval_cnt_i,
                    &mut acc_row, &mut pyr_a_row, pyr_b_row.as_mut_ptr(),
                );
            }
            for gx in 0..grid_w {
                *grid_out.add(gy * grid_w + gx) = if gx >= int_lo && gx < int_hi {
                    hsum_f32x4(acc_row[gx])
                } else {
                    0.0
                };
            }
        }
    }

    let grid_bot = grid_h * 16;
    if grid_bot < frame_h {
        let mut tail_eval_err2 = 0.0f32;
        let mut tail_eval_sig2 = 0.0f32;
        let mut tail_eval_cnt = 0u32;
        pan_int_streak_tail_rows::<EFF_EXP>(
            curr_rgba, prev_rgba, stat_streak_ptr, eval_mask_ptr, eca_res_ptr,
            frame_w, frame_h, grid_w, grid_h, w_i, h_i, pan_dx, pan_dy,
            exp_active, dg_q15, off_i16, hard_u8,
        );
        pan_fused_tail_rows_simd128::<HAS_MASK, ALLOW_EVAL_HEAL, FILTER_MASK>(
            luma_ptr, bg_luma_ptr, eval_mask_ptr, eca_res_ptr, mask_luma_tol, frame_w, frame_h,
            grid_h, -pan_dx, -pan_dy, w_i, h_i, alpha, offset_c, adapt_rate, heal_alpha_s,
            heal_gate_s, v_alpha, v_c, tol_v, v_rate_bg, v_rate_fg, v_motion_gate, heal_alpha,
            heal_gate_rgb, v_rate_eval, v_eval_gate, &mut acc_eval_err2, &mut acc_eval_sig2,
            &mut acc_eval_cnt_i, &mut tail_eval_err2, &mut tail_eval_sig2, &mut tail_eval_cnt,
        );
        let err2 = hsum_f32x4(acc_eval_err2) + tail_eval_err2;
        let sig2 = hsum_f32x4(acc_eval_sig2) + tail_eval_sig2;
        let eval_count = (hsum_i32x4(acc_eval_cnt_i) as u32).saturating_add(tail_eval_cnt);
        if !out_err2.is_null() { *out_err2 = err2; }
        if !out_sig2.is_null() { *out_sig2 = sig2; }
        if !out_eval_count.is_null() { *out_eval_count = eval_count; }
        return Some(if eval_count >= 1024 && sig2 > 1e-3 { err2 / sig2 } else { -1.0 });
    }

    let err2 = hsum_f32x4(acc_eval_err2);
    let sig2 = hsum_f32x4(acc_eval_sig2);
    let eval_count = hsum_i32x4(acc_eval_cnt_i) as u32;
    if !out_err2.is_null() { *out_err2 = err2; }
    if !out_sig2.is_null() { *out_sig2 = sig2; }
    if !out_eval_count.is_null() { *out_eval_count = eval_count; }
    Some(if eval_count >= 1024 && sig2 > 1e-3 { err2 / sig2 } else { -1.0 })
}

/// Fused single-pass integer-pan / exposure-hold stationarity streak + ECA + motion-grid + 160x90 pyramid.
#[inline(never)]
pub unsafe fn pan_streak_and_eca_fused_int_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg_luma_ptr: *mut f32,
    stat_streak_ptr: *mut u8,
    eval_mask_ptr: *mut u8,
    mask_warm: bool,
    mask_luma_tol: f32,
    eca_res_ptr: *mut f32,
    pyr_eca_ptr: *mut f32,
    grid_out: *mut f32,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    scale_px: usize,
    adapt_rate: f32,
    pan_dx: i32,
    pan_dy: i32,
    frac_mag: f32,
    exp_gain: f32,
    exp_off: f32,
    exp_regime: bool,
    rms_y: f32,
    out_err2: *mut f32,
    out_sig2: *mut f32,
    out_eval_count: *mut u32,
) -> Option<f32> {
    let (exp_active, dg_q15, off_i16) = exposure_q15_params(exp_gain, exp_off);
    let pan_moved = pan_dx != 0 || pan_dy != 0 || frac_mag >= 0.04;
    if frac_mag >= 0.04
        || grid_w == 0
        || grid_w > 40
        || frame_w != grid_w * 16
        || frame_h < grid_h * 16
        || (!pan_moved && !exp_active && !exp_regime)
    {
        return None;
    }

    let eff_exp_regime = exp_regime || exp_active;
    let hard_u8 = if pan_moved || rms_y >= 0.52 { 4u8 } else { 3u8 };
    let len = frame_w * frame_h;
    let has_mask = mask_warm && !eval_mask_ptr.is_null();
    let use_live_fit_mask = has_mask && !(mask_luma_tol > 0.0 && mask_luma_tol <= 0.045);

    let fit = if use_live_fit_mask {
        eca_fit_affine_params_pan_int_live_simd128(
            luma_ptr,
            bg_luma_ptr,
            curr_rgba,
            prev_rgba,
            stat_streak_ptr,
            frame_w,
            frame_h,
            pan_dx,
            pan_dy,
            dg_q15,
            off_i16,
            hard_u8,
            exp_active,
            eff_exp_regime,
        )
    } else {
        eca_fit_affine_params_simd128(
            luma_ptr,
            bg_luma_ptr,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            len,
        )
    };
    let Some((alpha, offset_c)) = fit else {
        return None;
    };

    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let border_pad = if pan_dx == 0 && pan_dy == 0 { 0 } else { 2 };
    let mut gx_lo = grid_w;
    let mut gx_hi = 0usize;
    for gx in 0..grid_w {
        let bx = (gx * 16) as i32;
        if bx - pan_dx >= border_pad && bx + 16 - pan_dx <= w_i - border_pad {
            if gx_lo == grid_w {
                gx_lo = gx;
            }
            gx_hi = gx + 1;
        }
    }
    if gx_lo > gx_hi {
        gx_lo = 0;
        gx_hi = 0;
    }
    let mut gy_lo = grid_h;
    let mut gy_hi = 0usize;
    for gy in 0..grid_h {
        let by = (gy * 16) as i32;
        if by - pan_dy >= border_pad && by + 16 - pan_dy <= h_i - border_pad {
            if gy_lo == grid_h {
                gy_lo = gy;
            }
            gy_hi = gy + 1;
        }
    }
    if gy_lo > gy_hi {
        gy_lo = 0;
        gy_hi = 0;
    }

    let heal_alpha_s = if pan_moved && eff_exp_regime {
        0.86f32
    } else if eff_exp_regime {
        0.84f32
    } else if pan_moved {
        0.78f32
    } else {
        0.65f32
    };
    let scalar_gate = if eff_exp_regime { 0.023f32 } else { 0.010f32 };
    let heal_gate_s = scalar_gate * 765.0;
    let filter_mask = has_mask && mask_luma_tol > 0.0;

    macro_rules! run_int_impl {
        ($hm:expr, $aeh:expr, $gr:expr, $fm:expr, $ee:expr) => {
            pan_streak_and_eca_fused_int_impl::<$hm, $aeh, $gr, $fm, $ee>(
                curr_rgba, prev_rgba, luma_ptr, bg_luma_ptr, stat_streak_ptr, eval_mask_ptr,
                eca_res_ptr, pyr_eca_ptr, grid_out, frame_w, frame_h, grid_w, grid_h, scale_px,
                mask_luma_tol, alpha, offset_c, adapt_rate, heal_alpha_s, heal_gate_s,
                pan_dx, pan_dy, gx_lo, gx_hi, gy_lo, gy_hi, exp_active, dg_q15, off_i16,
                hard_u8, out_err2, out_sig2, out_eval_count,
            )
        };
    }

    match (has_mask, filter_mask, eff_exp_regime) {
        (true, true, false) => run_int_impl!(true, true, false, true, false),
        (true, false, false) => run_int_impl!(true, true, false, false, false),
        (false, _, false) => run_int_impl!(false, false, false, false, false),
        (true, true, true) => run_int_impl!(true, false, true, true, true),
        (true, false, true) => run_int_impl!(true, false, true, false, true),
        (false, _, true) => run_int_impl!(false, false, true, false, true),
    }
}

/// Compute 32-bit Morton-interleaved 3-Scale Dyadic Haar & Deadzone Opponent-Chroma Signature:
/// - Bit 31:      Chromatic confidence regime (sat2 > 0.018); collapses low-chroma patches to canonical neutral prefix
/// - Bits 30..27: Deadzone thermometer opponent chroma (R-G, B-Y with +/-0.055 zero-crossing guard)
/// - Bits 26..24: Radial center-surround contrast, high-saturation tier, and core luminance polarity
/// - Bits 23..16: Scale-1 orthogonal 2D Haar wavelet contrasts (Hx, Hy, Hxy) on 2x2 Morton quadrants
/// - Bits 15..0:  Scale-2 dyadic parent-child 4x4 Morton Z-order texture bitplane (blk_luma[z] >= quad_l[z >> 2])
pub unsafe fn haar_morton_signature_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    x0: usize,
    y0: usize,
) -> u32 {
    haar_morton_signature_scaled_simd128(luma_ptr, rg_ptr, by_ptr, stride, x0, y0, 1)
}

pub unsafe fn haar_morton_signature_scaled_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    x0: usize,
    y0: usize,
    scale_px: usize,
) -> u32 {
    haar_morton_signature_hyst_simd128(luma_ptr, rg_ptr, by_ptr, stride, x0, y0, scale_px, 0, false)
}

pub unsafe fn haar_morton_signature_hyst_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    x0: usize,
    y0: usize,
    scale_px: usize,
    ref_sig: u32,
    use_hyst: bool,
) -> u32 {
    let mut blk_luma = [0.0f32; 16]; // Indexed in Morton Z-order z = 0..16
    let mut blk_rg = [0.0f32; 16];
    let mut blk_by = [0.0f32; 16];

    let inv_16 = 1.0 / 16.0;
    let cstep = 4 * scale_px;

    if scale_px == 2 {
        let inv_64 = 1.0 / 64.0;
        for by in 0..4usize {
            let mut acc_l0 = f32x4_splat(0.0);
            let mut acc_l1 = f32x4_splat(0.0);
            let mut acc_l2 = f32x4_splat(0.0);
            let mut acc_l3 = f32x4_splat(0.0);

            let mut acc_rg0 = f32x4_splat(0.0);
            let mut acc_rg1 = f32x4_splat(0.0);
            let mut acc_rg2 = f32x4_splat(0.0);
            let mut acc_rg3 = f32x4_splat(0.0);

            let mut acc_by0 = f32x4_splat(0.0);
            let mut acc_by1 = f32x4_splat(0.0);
            let mut acc_by2 = f32x4_splat(0.0);
            let mut acc_by3 = f32x4_splat(0.0);

            let y_base = y0 + by * 8;
            for r in 0..8usize {
                let row_base = (y_base + r) * stride + x0;
                let lp = luma_ptr.add(row_base);
                acc_l0 = f32x4_add(acc_l0, f32x4_add(v128_load(lp as *const v128), v128_load(lp.add(4) as *const v128)));
                acc_l1 = f32x4_add(acc_l1, f32x4_add(v128_load(lp.add(8) as *const v128), v128_load(lp.add(12) as *const v128)));
                acc_l2 = f32x4_add(acc_l2, f32x4_add(v128_load(lp.add(16) as *const v128), v128_load(lp.add(20) as *const v128)));
                acc_l3 = f32x4_add(acc_l3, f32x4_add(v128_load(lp.add(24) as *const v128), v128_load(lp.add(28) as *const v128)));

                let rp = rg_ptr.add(row_base);
                acc_rg0 = f32x4_add(acc_rg0, f32x4_add(v128_load(rp as *const v128), v128_load(rp.add(4) as *const v128)));
                acc_rg1 = f32x4_add(acc_rg1, f32x4_add(v128_load(rp.add(8) as *const v128), v128_load(rp.add(12) as *const v128)));
                acc_rg2 = f32x4_add(acc_rg2, f32x4_add(v128_load(rp.add(16) as *const v128), v128_load(rp.add(20) as *const v128)));
                acc_rg3 = f32x4_add(acc_rg3, f32x4_add(v128_load(rp.add(24) as *const v128), v128_load(rp.add(28) as *const v128)));

                let bp = by_ptr.add(row_base);
                acc_by0 = f32x4_add(acc_by0, f32x4_add(v128_load(bp as *const v128), v128_load(bp.add(4) as *const v128)));
                acc_by1 = f32x4_add(acc_by1, f32x4_add(v128_load(bp.add(8) as *const v128), v128_load(bp.add(12) as *const v128)));
                acc_by2 = f32x4_add(acc_by2, f32x4_add(v128_load(bp.add(16) as *const v128), v128_load(bp.add(20) as *const v128)));
                acc_by3 = f32x4_add(acc_by3, f32x4_add(v128_load(bp.add(24) as *const v128), v128_load(bp.add(28) as *const v128)));
            }

            let z0 = morton2d_4x4(0, by);
            let z1 = morton2d_4x4(1, by);
            let z2 = morton2d_4x4(2, by);
            let z3 = morton2d_4x4(3, by);

            let v_inv = f32x4_splat(inv_64);
            let vl = f32x4_mul(hsum4_f32x4(acc_l0, acc_l1, acc_l2, acc_l3), v_inv);
            let vrg = f32x4_mul(hsum4_f32x4(acc_rg0, acc_rg1, acc_rg2, acc_rg3), v_inv);
            let vby = f32x4_mul(hsum4_f32x4(acc_by0, acc_by1, acc_by2, acc_by3), v_inv);

            blk_luma[z0] = f32x4_extract_lane::<0>(vl);
            blk_luma[z1] = f32x4_extract_lane::<1>(vl);
            blk_luma[z2] = f32x4_extract_lane::<2>(vl);
            blk_luma[z3] = f32x4_extract_lane::<3>(vl);

            blk_rg[z0] = f32x4_extract_lane::<0>(vrg);
            blk_rg[z1] = f32x4_extract_lane::<1>(vrg);
            blk_rg[z2] = f32x4_extract_lane::<2>(vrg);
            blk_rg[z3] = f32x4_extract_lane::<3>(vrg);

            blk_by[z0] = f32x4_extract_lane::<0>(vby);
            blk_by[z1] = f32x4_extract_lane::<1>(vby);
            blk_by[z2] = f32x4_extract_lane::<2>(vby);
            blk_by[z3] = f32x4_extract_lane::<3>(vby);
        }
    } else {
        for by in 0..4usize {
            let mut acc_l0 = f32x4_splat(0.0);
            let mut acc_l1 = f32x4_splat(0.0);
            let mut acc_l2 = f32x4_splat(0.0);
            let mut acc_l3 = f32x4_splat(0.0);

            let mut acc_rg0 = f32x4_splat(0.0);
            let mut acc_rg1 = f32x4_splat(0.0);
            let mut acc_rg2 = f32x4_splat(0.0);
            let mut acc_rg3 = f32x4_splat(0.0);

            let mut acc_by0 = f32x4_splat(0.0);
            let mut acc_by1 = f32x4_splat(0.0);
            let mut acc_by2 = f32x4_splat(0.0);
            let mut acc_by3 = f32x4_splat(0.0);

            for ry in 0..4usize {
                let row_base = (y0 + (by * 4 + ry) * scale_px) * stride + x0;
                acc_l0 = f32x4_add(acc_l0, load_pooled4_f32(luma_ptr.add(row_base), stride, scale_px));
                acc_l1 = f32x4_add(acc_l1, load_pooled4_f32(luma_ptr.add(row_base + cstep), stride, scale_px));
                acc_l2 = f32x4_add(acc_l2, load_pooled4_f32(luma_ptr.add(row_base + cstep * 2), stride, scale_px));
                acc_l3 = f32x4_add(acc_l3, load_pooled4_f32(luma_ptr.add(row_base + cstep * 3), stride, scale_px));

                acc_rg0 = f32x4_add(acc_rg0, load_pooled4_f32(rg_ptr.add(row_base), stride, scale_px));
                acc_rg1 = f32x4_add(acc_rg1, load_pooled4_f32(rg_ptr.add(row_base + cstep), stride, scale_px));
                acc_rg2 = f32x4_add(acc_rg2, load_pooled4_f32(rg_ptr.add(row_base + cstep * 2), stride, scale_px));
                acc_rg3 = f32x4_add(acc_rg3, load_pooled4_f32(rg_ptr.add(row_base + cstep * 3), stride, scale_px));

                acc_by0 = f32x4_add(acc_by0, load_pooled4_f32(by_ptr.add(row_base), stride, scale_px));
                acc_by1 = f32x4_add(acc_by1, load_pooled4_f32(by_ptr.add(row_base + cstep), stride, scale_px));
                acc_by2 = f32x4_add(acc_by2, load_pooled4_f32(by_ptr.add(row_base + cstep * 2), stride, scale_px));
                acc_by3 = f32x4_add(acc_by3, load_pooled4_f32(by_ptr.add(row_base + cstep * 3), stride, scale_px));
            }

            let z0 = morton2d_4x4(0, by);
            let z1 = morton2d_4x4(1, by);
            let z2 = morton2d_4x4(2, by);
            let z3 = morton2d_4x4(3, by);

            let v_inv = f32x4_splat(inv_16);
            let vl = f32x4_mul(hsum4_f32x4(acc_l0, acc_l1, acc_l2, acc_l3), v_inv);
            let vrg = f32x4_mul(hsum4_f32x4(acc_rg0, acc_rg1, acc_rg2, acc_rg3), v_inv);
            let vby = f32x4_mul(hsum4_f32x4(acc_by0, acc_by1, acc_by2, acc_by3), v_inv);

            blk_luma[z0] = f32x4_extract_lane::<0>(vl);
            blk_luma[z1] = f32x4_extract_lane::<1>(vl);
            blk_luma[z2] = f32x4_extract_lane::<2>(vl);
            blk_luma[z3] = f32x4_extract_lane::<3>(vl);

            blk_rg[z0] = f32x4_extract_lane::<0>(vrg);
            blk_rg[z1] = f32x4_extract_lane::<1>(vrg);
            blk_rg[z2] = f32x4_extract_lane::<2>(vrg);
            blk_rg[z3] = f32x4_extract_lane::<3>(vrg);

            blk_by[z0] = f32x4_extract_lane::<0>(vby);
            blk_by[z1] = f32x4_extract_lane::<1>(vby);
            blk_by[z2] = f32x4_extract_lane::<2>(vby);
            blk_by[z3] = f32x4_extract_lane::<3>(vby);
        }
    }

    pack_haar_morton_from_blocks(&blk_luma, &blk_rg, &blk_by, ref_sig, use_hyst)
}

#[inline(always)]
fn pack_haar_morton_from_blocks(
    blk_luma: &[f32; 16],
    blk_rg: &[f32; 16],
    blk_by: &[f32; 16],
    ref_sig: u32,
    use_hyst: bool,
) -> u32 {
    // Center 8x8 blocks in Morton order are z in {3, 6, 9, 12}
    let center_l = 0.25 * (blk_luma[3] + blk_luma[6] + blk_luma[9] + blk_luma[12]);
    let center_rg = 0.25 * (blk_rg[3] + blk_rg[6] + blk_rg[9] + blk_rg[12]);
    let center_by = 0.25 * (blk_by[3] + blk_by[6] + blk_by[9] + blk_by[12]);

    // Scale-1 8x8 quadrant means (Morton quadrants q = 0..4 correspond to z = 4*q .. 4*q+3)
    // q=0: TL, q=1: TR, q=2: BL, q=3: BR
    let mut quad_l = [0.0f32; 4];
    let mut quad_c = [0.0f32; 4];
    let mut total_l = 0.0f32;
    let mut total_rg = 0.0f32;
    let mut total_by = 0.0f32;

    for q in 0..4usize {
        let b = q * 4;
        let ql = 0.25 * (blk_luma[b] + blk_luma[b + 1] + blk_luma[b + 2] + blk_luma[b + 3]);
        let qrg = 0.25 * (blk_rg[b] + blk_rg[b + 1] + blk_rg[b + 2] + blk_rg[b + 3]);
        let qby = 0.25 * (blk_by[b] + blk_by[b + 1] + blk_by[b + 2] + blk_by[b + 3]);
        quad_l[q] = ql;
        quad_c[q] = qrg - qby;
        total_l += ql;
        total_rg += qrg;
        total_by += qby;
    }

    let mean_rg = 0.65 * center_rg + 0.35 * (total_rg * 0.25);
    let mean_by = 0.65 * center_by + 0.35 * (total_by * 0.25);
    let outer_l = (total_l - center_l) / 3.0;
    let sat2 = mean_rg * mean_rg + mean_by * mean_by;
    // Narrow Schmitt-trigger band around 0.018 prevents MSB (bit 31, d_2 = 1.0) chatter on
    // borderline chroma patches while preserving occluder rejection on gray gantry signs.
    let sat_thresh = if use_hyst {
        if (ref_sig & (1 << 31)) != 0 {
            0.0155
        } else {
            0.0205
        }
    } else {
        0.018
    };
    let is_chromatic = sat2 > sat_thresh;

    let mut sig: u32 = 0;

    // Coarse Chromatic & Radial Regime (MSBs: bits 31..24)
    if is_chromatic {
        sig |= 1 << 31;
        // 2-bit deadzone thermometer per opponent axis avoids zero-crossing flips around 0.0
        if mean_rg > 0.055 {
            sig |= 1 << 30;
        }
        if mean_rg > -0.055 {
            sig |= 1 << 29;
        }
        if mean_by > 0.055 {
            sig |= 1 << 28;
        }
        if mean_by > -0.055 {
            sig |= 1 << 27;
        }
    } else {
        // Canonical neutral prefix `0_01_01` prevents arbitrary hue sign flips on gray/neutral targets
        sig |= (1 << 29) | (1 << 27);
    }

    if center_l > outer_l {
        sig |= 1 << 26;
    }
    if sat2 > 0.075 {
        sig |= 1 << 25;
    }
    if center_l > 0.45 {
        sig |= 1 << 24;
    }

    // Scale-1 Orthogonal 2D Haar Wavelet Basis on 2x2 Morton Quadrants (bits 23..16)
    let hx_l = (quad_l[1] + quad_l[3]) - (quad_l[0] + quad_l[2]);
    let hy_l = (quad_l[2] + quad_l[3]) - (quad_l[0] + quad_l[1]);
    let hxy_l = (quad_l[0] + quad_l[3]) - (quad_l[1] + quad_l[2]);

    let hx_c = (quad_c[1] + quad_c[3]) - (quad_c[0] + quad_c[2]);
    let hy_c = (quad_c[2] + quad_c[3]) - (quad_c[0] + quad_c[1]);
    let hxy_c = (quad_c[0] + quad_c[3]) - (quad_c[1] + quad_c[2]);

    if hx_l >= 0.0 {
        sig |= 1 << 23;
    }
    if hy_l >= 0.0 {
        sig |= 1 << 22;
    }
    if hxy_l >= 0.0 {
        sig |= 1 << 21;
    }
    if hx_l.abs() >= hy_l.abs() {
        sig |= 1 << 20;
    }

    if hx_c >= 0.0 {
        sig |= 1 << 19;
    }
    if hy_c >= 0.0 {
        sig |= 1 << 18;
    }
    if hxy_c >= 0.0 {
        sig |= 1 << 17;
    }
    if (center_rg - center_by) >= 0.0 {
        sig |= 1 << 16;
    }

    // Scale-2 True Dyadic Parent-Child 4x4 Morton Z-Order Texture Bitplane (LSBs: bits 15..0)
    for z in 0..16usize {
        let parent_q = z >> 2;
        if blk_luma[z] >= quad_l[parent_q] {
            sig |= 1 << (15 - z);
        }
    }

    sig
}

/// SIMD128 16x16 spatial AC texture variance (Luma + Opponent Chroma).
/// Rejects flat H.264 macroblocks, uniform shadows, and aperture-ambiguous regions before seeding.
#[inline]
pub unsafe fn patch16x16_ac_energy_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    scale_px: usize,
) -> f32 {
    let mut sum_l = f32x4_splat(0.0);
    let mut sum_l2 = f32x4_splat(0.0);
    let mut sum_rg = f32x4_splat(0.0);
    let mut sum_rg2 = f32x4_splat(0.0);
    let mut sum_by = f32x4_splat(0.0);
    let mut sum_by2 = f32x4_splat(0.0);
    let cstep = 4 * scale_px;

    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let vl = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
            let vrg = load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px);
            let vby = load_pooled4_f32(by_ptr.add(ioff), stride, scale_px);
            sum_l = f32x4_add(sum_l, vl);
            sum_l2 = f32x4_add(sum_l2, f32x4_mul(vl, vl));
            sum_rg = f32x4_add(sum_rg, vrg);
            sum_rg2 = f32x4_add(sum_rg2, f32x4_mul(vrg, vrg));
            sum_by = f32x4_add(sum_by, vby);
            sum_by2 = f32x4_add(sum_by2, f32x4_mul(vby, vby));
        }
    }

    let inv_n = 1.0 / 256.0;
    let ml = hsum_f32x4(sum_l) * inv_n;
    let mrg = hsum_f32x4(sum_rg) * inv_n;
    let mby = hsum_f32x4(sum_by) * inv_n;

    let var_l = (hsum_f32x4(sum_l2) * inv_n - ml * ml).max(0.0);
    let var_rg = (hsum_f32x4(sum_rg2) * inv_n - mrg * mrg).max(0.0);
    let var_by = (hsum_f32x4(sum_by2) * inv_n - mby * mby).max(0.0);
    var_l + 1.5 * (var_rg + var_by)
}

/// Continuous SIMD128 multi-scale Photometric SSD (16x16 SD / 32x32 HD) with clamped partial-DC
/// illumination compensation (`+/-0.14`) and without discrete signature jumps.
pub unsafe fn ssd16x16_photometric_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    scale_px: usize,
) -> f32 {
    let mut sum_il = f32x4_splat(0.0);
    let mut sum_tl = f32x4_splat(0.0);
    let cstep = 4 * scale_px;
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            sum_il = f32x4_add(
                sum_il,
                load_pooled4_f32(luma_ptr.add(img_row + vec_idx * cstep), stride, scale_px),
            );
            sum_tl = f32x4_add(sum_tl, v128_load(tpl_luma.add(tpl_row + vec_idx * 4) as *const v128));
        }
    }
    let dc_shift = ((hsum_f32x4(sum_il) - hsum_f32x4(sum_tl)) * (1.0 / 256.0)).clamp(-0.14, 0.14) * 0.70;
    let v_dc = f32x4_splat(dc_shift);

    let mut acc_ssd_l = f32x4_splat(0.0);
    let mut acc_ssd_c = f32x4_splat(0.0);

    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;

        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;

            let dl = f32x4_sub(
                f32x4_sub(load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px), v_dc),
                v128_load(tpl_luma.add(toff) as *const v128),
            );
            acc_ssd_l = f32x4_add(acc_ssd_l, f32x4_mul(dl, dl));

            let drg = f32x4_sub(
                load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px),
                v128_load(tpl_rg.add(toff) as *const v128),
            );
            let dby = f32x4_sub(
                load_pooled4_f32(by_ptr.add(ioff), stride, scale_px),
                v128_load(tpl_by.add(toff) as *const v128),
            );
            acc_ssd_c = f32x4_add(
                acc_ssd_c,
                f32x4_add(f32x4_mul(drg, drg), f32x4_mul(dby, dby)),
            );
        }
    }

    (hsum_f32x4(acc_ssd_l) + 1.85 * hsum_f32x4(acc_ssd_c)) * (1.0 / 256.0)
}

/// Multi-scale Optical Patch Tracker Cost using SIMD128 (`scale_px = 1` -> 16x16 SD, `scale_px = 2` -> 32x32 HD):
/// Combines clamped DC-compensated Luma SAD, Chroma (R-G, B-Y) SAD, ECA motion-residual reward,
/// and binary-prefix 2-adic ultrametric valuation penalty.
#[inline]
pub unsafe fn sad_patch_scaled_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_sig: u32,
    out_cand_sig: *mut u32,
    scale_px: usize,
) -> f32 {
    let mut sum_il = f32x4_splat(0.0);
    let mut sum_tl = f32x4_splat(0.0);
    let cstep = 4 * scale_px;
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            sum_il = f32x4_add(
                sum_il,
                load_pooled4_f32(luma_ptr.add(img_row + vec_idx * cstep), stride, scale_px),
            );
            sum_tl = f32x4_add(sum_tl, v128_load(tpl_luma.add(tpl_row + vec_idx * 4) as *const v128));
        }
    }
    let dc_shift = ((hsum_f32x4(sum_il) - hsum_f32x4(sum_tl)) * (1.0 / 256.0)).clamp(-0.14, 0.14) * 0.70;
    let v_dc = f32x4_splat(dc_shift);

    let mut acc_sad_l = f32x4_splat(0.0);
    let mut acc_sad_c = f32x4_splat(0.0);
    let mut acc_eca = f32x4_splat(0.0);

    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;

        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;

            let il = f32x4_sub(load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px), v_dc);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            acc_sad_l = f32x4_add(acc_sad_l, f32x4_abs(f32x4_sub(il, tl)));

            let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px);
            let trg = v128_load(tpl_rg.add(toff) as *const v128);
            let iby = load_pooled4_f32(by_ptr.add(ioff), stride, scale_px);
            let tby = v128_load(tpl_by.add(toff) as *const v128);

            let diff_rg = f32x4_abs(f32x4_sub(irg, trg));
            let diff_by = f32x4_abs(f32x4_sub(iby, tby));
            acc_sad_c = f32x4_add(acc_sad_c, f32x4_add(diff_rg, diff_by));

            let eca = load_pooled4_f32(eca_res_ptr.add(ioff), stride, scale_px);
            acc_eca = f32x4_add(acc_eca, eca);
        }
    }

    let sad_l = hsum_f32x4(acc_sad_l) * (1.0 / 256.0);
    let sad_c = hsum_f32x4(acc_sad_c) * (1.0 / 256.0);
    let motion_reward = (hsum_f32x4(acc_eca) * (1.0 / 256.0)).min(0.35);

    let cand_sig = haar_morton_signature_hyst_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, cx, cy, scale_px, tpl_sig, true,
    );
    if !out_cand_sig.is_null() {
        *out_cand_sig = cand_sig;
    }

    let val = padic_ultrametric_val(tpl_sig, cand_sig);
    let d2 = padic_distance_from_val(val);

    sad_l + 1.85 * sad_c + 0.55 * d2 - 0.18 * motion_reward
}

pub unsafe fn sad16x16_patch_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_sig: u32,
    out_cand_sig: *mut u32,
) -> f32 {
    sad_patch_scaled_simd128(
        luma_ptr,
        rg_ptr,
        by_ptr,
        eca_res_ptr,
        stride,
        cx,
        cy,
        tpl_luma,
        tpl_rg,
        tpl_by,
        tpl_sig,
        out_cand_sig,
        1,
    )
}

/// Unpack a rectangular ROI `(roi_x, roi_y, roi_w, roi_h)` (`roi_w` multiple of 4) from a buffered
/// RGBA8 ring slot into contiguous planar `roi_luma`, `roi_rg`, `roi_by` buffers for non-causal
/// backward re-tracking across `rgba_ring`.
pub unsafe fn rgba_roi_to_planar_simd128(
    rgba_ptr: *const u8,
    frame_stride: usize,
    roi_x: usize,
    roi_y: usize,
    roi_w: usize,
    roi_h: usize,
    roi_luma: *mut f32,
    roi_rg: *mut f32,
    roi_by: *mut f32,
) {
    let mask_ff = u32x4_splat(0xFF);
    let inv_255 = f32x4_splat(1.0 / 255.0);
    let wr = f32x4_splat(0.299);
    let wg = f32x4_splat(0.587);
    let wb = f32x4_splat(0.114);

    for ry in 0..roi_h {
        let src_row_px = (roi_y + ry) * frame_stride + roi_x;
        let dst_row = ry * roi_w;
        let mut rx = 0usize;
        while rx + 4 <= roi_w {
            let px = v128_load(rgba_ptr.add((src_row_px + rx) * 4) as *const v128);
            let r_u32 = v128_and(px, mask_ff);
            let g_u32 = v128_and(u32x4_shr(px, 8), mask_ff);
            let b_u32 = v128_and(u32x4_shr(px, 16), mask_ff);

            let r = f32x4_mul(f32x4_convert_u32x4(r_u32), inv_255);
            let g = f32x4_mul(f32x4_convert_u32x4(g_u32), inv_255);
            let b = f32x4_mul(f32x4_convert_u32x4(b_u32), inv_255);

            let y = f32x4_add(f32x4_add(f32x4_mul(wr, r), f32x4_mul(wg, g)), f32x4_mul(wb, b));
            let rg = f32x4_sub(r, g);
            let by = f32x4_sub(b, y);

            v128_store(roi_luma.add(dst_row + rx) as *mut v128, y);
            v128_store(roi_rg.add(dst_row + rx) as *mut v128, rg);
            v128_store(roi_by.add(dst_row + rx) as *mut v128, by);
            rx += 4;
        }
    }
}

#[inline(always)]
pub unsafe fn pool_plane_rows_to_160x90_simd128(
    src: *const f32,
    dst: *mut f32,
    scale_px: usize,
    py_start: usize,
    py_end: usize,
) {
    let py_end = py_end.min(90);
    if scale_px <= 1 {
        // 320x180 -> 160x90 exact 2x2 SIMD128 box downsample
        for py in py_start..py_end {
            let src_row = (py * 2) * 320;
            let dst_row = py * 160;
            let mut px = 0usize;
            while px + 4 <= 160 {
                let v = load_pooled4_f32(src.add(src_row + px * 2), 320, 2);
                v128_store(dst.add(dst_row + px) as *mut v128, v);
                px += 4;
            }
        }
    } else {
        // 640x360 -> 160x90 exact 4x4 register-only SIMD128 box downsample
        let q16 = f32x4_splat(1.0 / 16.0);
        for py in py_start..py_end {
            let r0 = (py * 4) * 640;
            let r2 = (py * 4 + 2) * 640;
            let dst_row = py * 160;
            let mut px = 0usize;
            while px + 4 <= 160 {
                let sx = px * 4;
                let top_a = load_sum4_2x2_f32(src.add(r0 + sx), 640);
                let top_b = load_sum4_2x2_f32(src.add(r0 + sx + 8), 640);
                let bot_a = load_sum4_2x2_f32(src.add(r2 + sx), 640);
                let bot_b = load_sum4_2x2_f32(src.add(r2 + sx + 8), 640);
                let s_a = f32x4_add(top_a, bot_a);
                let s_b = f32x4_add(top_b, bot_b);
                let evens = i32x4_shuffle::<0, 2, 4, 6>(s_a, s_b);
                let odds = i32x4_shuffle::<1, 3, 5, 7>(s_a, s_b);
                let v = f32x4_mul(f32x4_add(evens, odds), q16);
                v128_store(dst.add(dst_row + px) as *mut v128, v);
                px += 4;
            }
        }
    }
}

#[inline]
pub unsafe fn pool_plane_to_160x90_simd128(src: *const f32, dst: *mut f32, scale_px: usize) {
    pool_plane_rows_to_160x90_simd128(src, dst, scale_px, 0, 90);
}

/// Build the 160x90 multi-channel SIMD128 pyramid (`luma`, `rg`, `by`, `eca_res`) in linear memory.
pub unsafe fn build_pyramid_160x90_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_ptr: *const f32,
    pyr_luma: *mut f32,
    pyr_rg: *mut f32,
    pyr_by: *mut f32,
    pyr_eca: *mut f32,
    scale_px: usize,
) {
    pool_plane_to_160x90_simd128(luma_ptr, pyr_luma, scale_px);
    pool_plane_to_160x90_simd128(rg_ptr, pyr_rg, scale_px);
    pool_plane_to_160x90_simd128(by_ptr, pyr_by, scale_px);
    pool_plane_to_160x90_simd128(eca_ptr, pyr_eca, scale_px);
}

/// Radially-apodized SIMD128 2D Structure Tensor (Shi-Tomasi minor/major eigenvalues) over Luma + Chroma.
/// Returns `(lambda_min, lambda_max)`:
/// - Rejects flat walls/compression blocks (`lambda_min` small) AND 1D aperture edges like forearms,
///   room moldings, book straight borders, and vertical support posts (`lambda_min / lambda_max` small).
#[inline(never)]
pub unsafe fn patch16x16_shi_tomasi_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    scale_px: usize,
) -> (f32, f32) {
    let mut acc_jxx = f32x4_splat(0.0);
    let mut acc_jxy = f32x4_splat(0.0);
    let mut acc_jyy = f32x4_splat(0.0);
    let cstep = 4 * scale_px;
    let chroma_w = f32x4_splat(1.25);

    for ry in 2..14usize {
        let wy = if (4..12).contains(&ry) { 1.0 } else { 0.30 };
        let row_c = (cy + ry * scale_px) * stride + cx;
        let row_m = (cy + (ry - 1) * scale_px) * stride + cx;
        let row_p = (cy + (ry + 1) * scale_px) * stride + cx;

        for vec_idx in 0..4usize {
            let wx = if vec_idx == 1 || vec_idx == 2 { 1.0 } else { 0.35 };
            let vw = f32x4_splat(wx * wy);
            let off = vec_idx * cstep;

            let lx_p = load_pooled4_f32(luma_ptr.add(row_c + off + scale_px), stride, scale_px);
            let lx_m = load_pooled4_f32(luma_ptr.add(row_c + off - scale_px), stride, scale_px);
            let ly_p = load_pooled4_f32(luma_ptr.add(row_p + off), stride, scale_px);
            let ly_m = load_pooled4_f32(luma_ptr.add(row_m + off), stride, scale_px);
            let gx_l = f32x4_sub(lx_p, lx_m);
            let gy_l = f32x4_sub(ly_p, ly_m);

            let rgx = f32x4_sub(
                load_pooled4_f32(rg_ptr.add(row_c + off + scale_px), stride, scale_px),
                load_pooled4_f32(rg_ptr.add(row_c + off - scale_px), stride, scale_px),
            );
            let rgy = f32x4_sub(
                load_pooled4_f32(rg_ptr.add(row_p + off), stride, scale_px),
                load_pooled4_f32(rg_ptr.add(row_m + off), stride, scale_px),
            );

            let byx = f32x4_sub(
                load_pooled4_f32(by_ptr.add(row_c + off + scale_px), stride, scale_px),
                load_pooled4_f32(by_ptr.add(row_c + off - scale_px), stride, scale_px),
            );
            let byy = f32x4_sub(
                load_pooled4_f32(by_ptr.add(row_p + off), stride, scale_px),
                load_pooled4_f32(by_ptr.add(row_m + off), stride, scale_px),
            );

            let gxx = f32x4_add(
                f32x4_mul(gx_l, gx_l),
                f32x4_mul(chroma_w, f32x4_add(f32x4_mul(rgx, rgx), f32x4_mul(byx, byx))),
            );
            let gxy = f32x4_add(
                f32x4_mul(gx_l, gy_l),
                f32x4_mul(chroma_w, f32x4_add(f32x4_mul(rgx, rgy), f32x4_mul(byx, byy))),
            );
            let gyy = f32x4_add(
                f32x4_mul(gy_l, gy_l),
                f32x4_mul(chroma_w, f32x4_add(f32x4_mul(rgy, rgy), f32x4_mul(byy, byy))),
            );

            acc_jxx = f32x4_add(acc_jxx, f32x4_mul(vw, gxx));
            acc_jxy = f32x4_add(acc_jxy, f32x4_mul(vw, gxy));
            acc_jyy = f32x4_add(acc_jyy, f32x4_mul(vw, gyy));
        }
    }

    let inv_n = 1.0 / 128.0;
    let jxx = hsum_f32x4(acc_jxx) * inv_n;
    let jxy = hsum_f32x4(acc_jxy) * inv_n;
    let jyy = hsum_f32x4(acc_jyy) * inv_n;

    let tr = jxx + jyy;
    let diff = jxx - jyy;
    let disc = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat((diff * diff + 4.0 * jxy * jxy).max(0.0))));
    let lam_max = 0.5 * (tr + disc);
    let lam_min = (0.5 * (tr - disc)).max(0.0);
    (lam_min, lam_max)
}

/// Fast 8x8 foreground-weighted coarse SAD on the 160x90 SIMD128 pyramid (`16` v128 vectors total).
#[inline]
pub unsafe fn sad8x8_pyr160_simd128(
    pyr_luma: *const f32,
    pyr_rg: *const f32,
    pyr_by: *const f32,
    pyr_eca: *const f32,
    px: usize,
    py: usize,
    tpl8_luma: *const f32,
    tpl8_rg: *const f32,
    tpl8_by: *const f32,
    tpl8_w: *const f32,
) -> f32 {
    let mut sum_w_diff = f32x4_splat(0.0);
    let mut acc_c = f32x4_splat(0.0);
    let mut acc_e = f32x4_splat(0.0);

    macro_rules! load_pyr_vec {
        ($ry:expr, $v:expr) => {{
            let ioff = (py + $ry) * 160 + px + $v * 4;
            let toff = $ry * 8 + $v * 4;
            let vw = v128_load(tpl8_w.add(toff) as *const v128);
            let il = v128_load(pyr_luma.add(ioff) as *const v128);
            let tl = v128_load(tpl8_luma.add(toff) as *const v128);
            let diff_l = f32x4_sub(il, tl);
            sum_w_diff = f32x4_add(sum_w_diff, f32x4_mul(vw, diff_l));

            let drg = f32x4_abs(f32x4_sub(
                v128_load(pyr_rg.add(ioff) as *const v128),
                v128_load(tpl8_rg.add(toff) as *const v128),
            ));
            let dby = f32x4_abs(f32x4_sub(
                v128_load(pyr_by.add(ioff) as *const v128),
                v128_load(tpl8_by.add(toff) as *const v128),
            ));
            acc_c = f32x4_add(acc_c, f32x4_mul(vw, f32x4_add(drg, dby)));
            acc_e = f32x4_add(acc_e, f32x4_mul(vw, v128_load(pyr_eca.add(ioff) as *const v128)));
            diff_l
        }};
    }

    let dl0 = load_pyr_vec!(0, 0);
    let dl1 = load_pyr_vec!(0, 1);
    let dl2 = load_pyr_vec!(1, 0);
    let dl3 = load_pyr_vec!(1, 1);
    let dl4 = load_pyr_vec!(2, 0);
    let dl5 = load_pyr_vec!(2, 1);
    let dl6 = load_pyr_vec!(3, 0);
    let dl7 = load_pyr_vec!(3, 1);
    let dl8 = load_pyr_vec!(4, 0);
    let dl9 = load_pyr_vec!(4, 1);
    let dl10 = load_pyr_vec!(5, 0);
    let dl11 = load_pyr_vec!(5, 1);
    let dl12 = load_pyr_vec!(6, 0);
    let dl13 = load_pyr_vec!(6, 1);
    let dl14 = load_pyr_vec!(7, 0);
    let dl15 = load_pyr_vec!(7, 1);

    let dc = hsum_f32x4(sum_w_diff).clamp(-0.14, 0.14) * 0.70;
    let v_dc = f32x4_splat(dc);

    let mut acc_l0 = f32x4_splat(0.0);
    let mut acc_l1 = f32x4_splat(0.0);
    macro_rules! accum_sad_l {
        ($acc:expr, $dl:expr, $idx:expr) => {{
            let vw = v128_load(tpl8_w.add($idx * 4) as *const v128);
            $acc = f32x4_add($acc, f32x4_mul(vw, f32x4_abs(f32x4_sub($dl, v_dc))));
        }};
    }
    accum_sad_l!(acc_l0, dl0, 0);
    accum_sad_l!(acc_l1, dl1, 1);
    accum_sad_l!(acc_l0, dl2, 2);
    accum_sad_l!(acc_l1, dl3, 3);
    accum_sad_l!(acc_l0, dl4, 4);
    accum_sad_l!(acc_l1, dl5, 5);
    accum_sad_l!(acc_l0, dl6, 6);
    accum_sad_l!(acc_l1, dl7, 7);
    accum_sad_l!(acc_l0, dl8, 8);
    accum_sad_l!(acc_l1, dl9, 9);
    accum_sad_l!(acc_l0, dl10, 10);
    accum_sad_l!(acc_l1, dl11, 11);
    accum_sad_l!(acc_l0, dl12, 12);
    accum_sad_l!(acc_l1, dl13, 13);
    accum_sad_l!(acc_l0, dl14, 14);
    accum_sad_l!(acc_l1, dl15, 15);
    let sad_l = hsum_f32x4(f32x4_add(acc_l0, acc_l1));
    let sad_c = hsum_f32x4(acc_c);
    let mot = hsum_f32x4(acc_e).min(0.35);
    sad_l + 1.85 * sad_c - 0.16 * mot
}

/// Cutoff-pruned `sad8x8_pyr160_simd128`: evaluates chroma core 4 rows, then outer chroma
/// 4 rows, then DC-compensated luma, then ECA, pruning immediately at each exact lower bound.
#[inline]
pub unsafe fn sad8x8_pyr160_cutoff_simd128(
    pyr_luma: *const f32,
    pyr_rg: *const f32,
    pyr_by: *const f32,
    pyr_eca: *const f32,
    px: usize,
    py: usize,
    tpl8_luma: *const f32,
    tpl8_rg: *const f32,
    tpl8_by: *const f32,
    tpl8_w: *const f32,
    cutoff: f32,
) -> f32 {
    let mut acc_c0 = f32x4_splat(0.0);
    let mut acc_c1 = f32x4_splat(0.0);
    macro_rules! accum_chroma_row {
        ($ry:expr) => {{
            let ry: usize = $ry;
            let ioff = (py + ry) * 160 + px;
            let toff = ry * 8;
            let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
            let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
            let drg0 = f32x4_abs(f32x4_sub(
                v128_load(pyr_rg.add(ioff) as *const v128),
                v128_load(tpl8_rg.add(toff) as *const v128),
            ));
            let dby0 = f32x4_abs(f32x4_sub(
                v128_load(pyr_by.add(ioff) as *const v128),
                v128_load(tpl8_by.add(toff) as *const v128),
            ));
            acc_c0 = f32x4_add(acc_c0, f32x4_mul(vw0, f32x4_add(drg0, dby0)));
            let drg1 = f32x4_abs(f32x4_sub(
                v128_load(pyr_rg.add(ioff + 4) as *const v128),
                v128_load(tpl8_rg.add(toff + 4) as *const v128),
            ));
            let dby1 = f32x4_abs(f32x4_sub(
                v128_load(pyr_by.add(ioff + 4) as *const v128),
                v128_load(tpl8_by.add(toff + 4) as *const v128),
            ));
            acc_c1 = f32x4_add(acc_c1, f32x4_mul(vw1, f32x4_add(drg1, dby1)));
        }};
    }
    for ry in 2..6usize {
        accum_chroma_row!(ry);
    }
    if 1.85 * hsum_f32x4(f32x4_add(acc_c0, acc_c1)) - 0.0565f32 >= cutoff {
        return cutoff;
    }
    for ry in [0usize, 1, 6, 7] {
        accum_chroma_row!(ry);
    }
    let sad_c = hsum_f32x4(f32x4_add(acc_c0, acc_c1));
    let base_c = 1.85 * sad_c;
    if base_c - 0.0565f32 >= cutoff {
        return cutoff;
    }

    let mut sw0 = f32x4_splat(0.0);
    let mut sw1 = f32x4_splat(0.0);
    let mut dl_uninit = [i32x4_splat(0); 16];
    let dl = dl_uninit.as_mut_ptr() as *mut v128;
    for ry in 0..8usize {
        let ioff = (py + ry) * 160 + px;
        let toff = ry * 8;
        let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
        let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
        let d0 = f32x4_sub(
            v128_load(pyr_luma.add(ioff) as *const v128),
            v128_load(tpl8_luma.add(toff) as *const v128),
        );
        let d1 = f32x4_sub(
            v128_load(pyr_luma.add(ioff + 4) as *const v128),
            v128_load(tpl8_luma.add(toff + 4) as *const v128),
        );
        *dl.add(ry * 2) = d0;
        *dl.add(ry * 2 + 1) = d1;
        sw0 = f32x4_add(sw0, f32x4_mul(vw0, d0));
        sw1 = f32x4_add(sw1, f32x4_mul(vw1, d1));
    }
    let dc = hsum_f32x4(f32x4_add(sw0, sw1)).clamp(-0.14, 0.14) * 0.70;
    let v_dc = f32x4_splat(dc);
    let mut acc_l0 = f32x4_splat(0.0);
    let mut acc_l1 = f32x4_splat(0.0);
    for ry in 0..8usize {
        let toff = ry * 8;
        let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
        let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
        acc_l0 = f32x4_add(
            acc_l0,
            f32x4_mul(vw0, f32x4_abs(f32x4_sub(*dl.add(ry * 2), v_dc))),
        );
        acc_l1 = f32x4_add(
            acc_l1,
            f32x4_mul(vw1, f32x4_abs(f32x4_sub(*dl.add(ry * 2 + 1), v_dc))),
        );
    }
    let sad_l = hsum_f32x4(f32x4_add(acc_l0, acc_l1));
    let partial = sad_l + base_c;
    if partial - 0.0565f32 >= cutoff {
        return cutoff;
    }

    let mut acc_e0 = f32x4_splat(0.0);
    let mut acc_e1 = f32x4_splat(0.0);
    for ry in 0..8usize {
        let ioff = (py + ry) * 160 + px;
        let toff = ry * 8;
        let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
        let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
        acc_e0 = f32x4_add(acc_e0, f32x4_mul(vw0, v128_load(pyr_eca.add(ioff) as *const v128)));
        acc_e1 = f32x4_add(acc_e1, f32x4_mul(vw1, v128_load(pyr_eca.add(ioff + 4) as *const v128)));
    }
    let mot = hsum_f32x4(f32x4_add(acc_e0, acc_e1)).min(0.35);
    partial - 0.16 * mot
}

/// 5x5 batched `sad8x8_pyr160_cutoff_simd128` for interior `(cx0-2..=cx0+2, cy0-2..=cy0+2)`:
/// shares a single 12-row strip cache across all 25 shifts via in-register `i32x4_shuffle`,
/// evaluates core chroma rows `[2,3,4,5]` for all 5 `dx`s simultaneously, and lazy-loads
/// outer/luma rows only for chroma survivors.
pub unsafe fn sad8x8_pyr160_5x5_simd128(
    pyr_luma: *const f32,
    pyr_rg: *const f32,
    pyr_by: *const f32,
    pyr_eca: *const f32,
    cx0: usize,
    cy0: usize,
    tpl8_luma: *const f32,
    tpl8_rg: *const f32,
    tpl8_by: *const f32,
    tpl8_w: *const f32,
    skip_center: bool,
    init_best_c: f32,
    out_best_px: &mut i32,
    out_best_py: &mut i32,
    out_best_c: &mut f32,
) {
    let mut b_c = init_best_c;
    let mut b_px = cx0 as i32;
    let mut b_py = cy0 as i32;
    let x0 = cx0 - 2;
    let y0 = cy0 - 2;

    let zero = f32x4_splat(0.0);
    let mut rg_uninit = [[zero; 3]; 12];
    let mut by_uninit = [[zero; 3]; 12];
    let mut lum_uninit = [[zero; 3]; 12];
    let rg_c = rg_uninit.as_mut_ptr() as *mut [v128; 3];
    let by_c = by_uninit.as_mut_ptr() as *mut [v128; 3];
    let lum_c = lum_uninit.as_mut_ptr() as *mut [v128; 3];
    let mut loaded_c: u16 = 0;
    let mut loaded_l: u16 = 0;

    macro_rules! ensure_c {
        ($k:expr) => {{
            let k: usize = $k;
            let bit = 1u16 << k;
            if (loaded_c & bit) == 0 {
                loaded_c |= bit;
                let off = (y0 + k) * 160 + x0;
                *rg_c.add(k) = [
                    v128_load(pyr_rg.add(off) as *const v128),
                    v128_load(pyr_rg.add(off + 4) as *const v128),
                    v128_load(pyr_rg.add(off + 8) as *const v128),
                ];
                *by_c.add(k) = [
                    v128_load(pyr_by.add(off) as *const v128),
                    v128_load(pyr_by.add(off + 4) as *const v128),
                    v128_load(pyr_by.add(off + 8) as *const v128),
                ];
            }
        }};
    }
    macro_rules! ensure_l {
        ($k:expr) => {{
            let k: usize = $k;
            let bit = 1u16 << k;
            if (loaded_l & bit) == 0 {
                loaded_l |= bit;
                let off = (y0 + k) * 160 + x0;
                *lum_c.add(k) = [
                    v128_load(pyr_luma.add(off) as *const v128),
                    v128_load(pyr_luma.add(off + 4) as *const v128),
                    v128_load(pyr_luma.add(off + 8) as *const v128),
                ];
            }
        }};
    }
    #[inline(always)]
    fn sh5(r: &[v128; 3], s: usize) -> (v128, v128) {
        let (a, b, c) = (r[0], r[1], r[2]);
        match s {
            0 => (a, b),
            1 => (i32x4_shuffle::<1, 2, 3, 4>(a, b), i32x4_shuffle::<1, 2, 3, 4>(b, c)),
            2 => (i32x4_shuffle::<2, 3, 4, 5>(a, b), i32x4_shuffle::<2, 3, 4, 5>(b, c)),
            3 => (i32x4_shuffle::<3, 4, 5, 6>(a, b), i32x4_shuffle::<3, 4, 5, 6>(b, c)),
            _ => (b, c),
        }
    }

    let mut pre_trg0 = [zero; 4];
    let mut pre_trg1 = [zero; 4];
    let mut pre_tby0 = [zero; 4];
    let mut pre_tby1 = [zero; 4];
    let mut pre_vw0 = [zero; 4];
    let mut pre_vw1 = [zero; 4];
    for idx in 0..4usize {
        let toff = (idx + 2) * 8;
        pre_vw0[idx] = v128_load(tpl8_w.add(toff) as *const v128);
        pre_vw1[idx] = v128_load(tpl8_w.add(toff + 4) as *const v128);
        pre_trg0[idx] = v128_load(tpl8_rg.add(toff) as *const v128);
        pre_trg1[idx] = v128_load(tpl8_rg.add(toff + 4) as *const v128);
        pre_tby0[idx] = v128_load(tpl8_by.add(toff) as *const v128);
        pre_tby1[idx] = v128_load(tpl8_by.add(toff + 4) as *const v128);
    }

    for dy_i in [2usize, 1, 3, 0, 4] {
        let mut acc_c = [zero; 5];
        for ry_idx in 0..4usize {
            let ry = ry_idx + 2;
            ensure_c!(dy_i + ry);
            let vw0 = pre_vw0[ry_idx];
            let vw1 = pre_vw1[ry_idx];
            let trg0 = pre_trg0[ry_idx];
            let trg1 = pre_trg1[ry_idx];
            let tby0 = pre_tby0[ry_idx];
            let tby1 = pre_tby1[ry_idx];
            let rg_r = &*rg_c.add(dy_i + ry);
            let by_r = &*by_c.add(dy_i + ry);

            macro_rules! accum_s {
                ($s:expr, $irg0:expr, $irg1:expr, $iby0:expr, $iby1:expr) => {{
                    let d0 = f32x4_add(
                        f32x4_abs(f32x4_sub($irg0, trg0)),
                        f32x4_abs(f32x4_sub($iby0, tby0)),
                    );
                    let d1 = f32x4_add(
                        f32x4_abs(f32x4_sub($irg1, trg1)),
                        f32x4_abs(f32x4_sub($iby1, tby1)),
                    );
                    acc_c[$s] = f32x4_add(
                        acc_c[$s],
                        f32x4_add(f32x4_mul(vw0, d0), f32x4_mul(vw1, d1)),
                    );
                }};
            }
            let (ra, rb, rc) = (rg_r[0], rg_r[1], rg_r[2]);
            let (ba, bb, bc) = (by_r[0], by_r[1], by_r[2]);
            accum_s!(0, ra, rb, ba, bb);
            accum_s!(
                1,
                i32x4_shuffle::<1, 2, 3, 4>(ra, rb),
                i32x4_shuffle::<1, 2, 3, 4>(rb, rc),
                i32x4_shuffle::<1, 2, 3, 4>(ba, bb),
                i32x4_shuffle::<1, 2, 3, 4>(bb, bc)
            );
            if !skip_center || dy_i != 2 {
                accum_s!(
                    2,
                    i32x4_shuffle::<2, 3, 4, 5>(ra, rb),
                    i32x4_shuffle::<2, 3, 4, 5>(rb, rc),
                    i32x4_shuffle::<2, 3, 4, 5>(ba, bb),
                    i32x4_shuffle::<2, 3, 4, 5>(bb, bc)
                );
            }
            accum_s!(
                3,
                i32x4_shuffle::<3, 4, 5, 6>(ra, rb),
                i32x4_shuffle::<3, 4, 5, 6>(rb, rc),
                i32x4_shuffle::<3, 4, 5, 6>(ba, bb),
                i32x4_shuffle::<3, 4, 5, 6>(bb, bc)
            );
            accum_s!(4, rb, rc, bb, bc);
        }

        for s in [2usize, 1, 3, 0, 4] {
            if skip_center && dy_i == 2 && s == 2 {
                continue;
            }
            if 1.85 * hsum_f32x4(acc_c[s]) - 0.0565f32 >= b_c {
                continue;
            }
            ensure_c!(dy_i);
            ensure_c!(dy_i + 1);
            ensure_c!(dy_i + 6);
            ensure_c!(dy_i + 7);
            let mut acc_full = acc_c[s];
            for ry in [0usize, 1, 6, 7] {
                let toff = ry * 8;
                let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
                let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
                let trg0 = v128_load(tpl8_rg.add(toff) as *const v128);
                let trg1 = v128_load(tpl8_rg.add(toff + 4) as *const v128);
                let tby0 = v128_load(tpl8_by.add(toff) as *const v128);
                let tby1 = v128_load(tpl8_by.add(toff + 4) as *const v128);
                let (irg0, irg1) = sh5(&*rg_c.add(dy_i + ry), s);
                let (iby0, iby1) = sh5(&*by_c.add(dy_i + ry), s);
                let d0 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg0, trg0)),
                    f32x4_abs(f32x4_sub(iby0, tby0)),
                );
                let d1 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg1, trg1)),
                    f32x4_abs(f32x4_sub(iby1, tby1)),
                );
                acc_full = f32x4_add(acc_full, f32x4_add(f32x4_mul(vw0, d0), f32x4_mul(vw1, d1)));
            }
            let base_c = 1.85 * hsum_f32x4(acc_full);
            if base_c - 0.0565f32 >= b_c {
                continue;
            }

            let mut sw = zero;
            let mut dl_uninit = [i32x4_splat(0); 16];
            let dl = dl_uninit.as_mut_ptr() as *mut v128;
            for ry in 0..8usize {
                ensure_l!(dy_i + ry);
                let toff = ry * 8;
                let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
                let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
                let (il0, il1) = sh5(&*lum_c.add(dy_i + ry), s);
                let d0 = f32x4_sub(il0, v128_load(tpl8_luma.add(toff) as *const v128));
                let d1 = f32x4_sub(il1, v128_load(tpl8_luma.add(toff + 4) as *const v128));
                *dl.add(ry * 2) = d0;
                *dl.add(ry * 2 + 1) = d1;
                sw = f32x4_add(sw, f32x4_add(f32x4_mul(vw0, d0), f32x4_mul(vw1, d1)));
            }
            let dc = hsum_f32x4(sw).clamp(-0.14, 0.14) * 0.70;
            let v_dc = f32x4_splat(dc);
            let mut acc_l0 = zero;
            let mut acc_l1 = zero;
            for ry in 0..8usize {
                let toff = ry * 8;
                let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
                let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
                acc_l0 = f32x4_add(
                    acc_l0,
                    f32x4_mul(vw0, f32x4_abs(f32x4_sub(*dl.add(ry * 2), v_dc))),
                );
                acc_l1 = f32x4_add(
                    acc_l1,
                    f32x4_mul(vw1, f32x4_abs(f32x4_sub(*dl.add(ry * 2 + 1), v_dc))),
                );
            }
            let partial = hsum_f32x4(f32x4_add(acc_l0, acc_l1)) + base_c;
            if partial - 0.0565f32 >= b_c {
                continue;
            }

            let qx = x0 + s;
            let qy = y0 + dy_i;
            let mut acc_e0 = zero;
            let mut acc_e1 = zero;
            for ry in 0..8usize {
                let ioff = (qy + ry) * 160 + qx;
                let toff = ry * 8;
                let vw0 = v128_load(tpl8_w.add(toff) as *const v128);
                let vw1 = v128_load(tpl8_w.add(toff + 4) as *const v128);
                acc_e0 = f32x4_add(acc_e0, f32x4_mul(vw0, v128_load(pyr_eca.add(ioff) as *const v128)));
                acc_e1 = f32x4_add(acc_e1, f32x4_mul(vw1, v128_load(pyr_eca.add(ioff + 4) as *const v128)));
            }
            let score = partial - 0.16 * hsum_f32x4(f32x4_add(acc_e0, acc_e1)).min(0.35);
            if score < b_c {
                b_c = score;
                b_px = qx as i32;
                b_py = qy as i32;
            }
        }
    }
    *out_best_px = b_px;
    *out_best_py = b_py;
    *out_best_c = b_c;
}

/// Foreground-Masked SIMD128 Patch Tracker Cost (`tpl_weight` normalized to sum = 1.0):
/// Suppresses static background clutter in outer patch corners by >12x while computing weighted
/// DC-compensated Luma/Chroma SAD, weighted ECA motion reward, and 2-adic ultrametric valuation.
#[inline]
pub unsafe fn sad_patch_weighted_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    tpl_sig: u32,
    out_cand_sig: *mut u32,
    out_weighted_eca: *mut f32,
    scale_px: usize,
) -> f32 {
    let mut sum_w_diff = f32x4_splat(0.0);
    let cstep = 4 * scale_px;
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            sum_w_diff = f32x4_add(sum_w_diff, f32x4_mul(vw, f32x4_sub(il, tl)));
        }
    }
    let dc_shift = hsum_f32x4(sum_w_diff).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(dc_shift);

    let mut acc_sad_l = f32x4_splat(0.0);
    let mut acc_sad_c = f32x4_splat(0.0);
    let mut acc_eca = f32x4_splat(0.0);

    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;

        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);

            let il = f32x4_sub(load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px), v_dc);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, f32x4_abs(f32x4_sub(il, tl))));

            let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px);
            let trg = v128_load(tpl_rg.add(toff) as *const v128);
            let iby = load_pooled4_f32(by_ptr.add(ioff), stride, scale_px);
            let tby = v128_load(tpl_by.add(toff) as *const v128);

            let diff_c = f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
            acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));

            if !eca_res_ptr.is_null() {
                let eca = load_pooled4_f32(eca_res_ptr.add(ioff), stride, scale_px);
                acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca));
            }
        }
    }

    let sad_l = hsum_f32x4(acc_sad_l);
    let sad_c = hsum_f32x4(acc_sad_c);
    let w_eca = hsum_f32x4(acc_eca);
    if !out_weighted_eca.is_null() {
        *out_weighted_eca = w_eca;
    }
    let motion_reward = w_eca.min(0.85);

    let cand_sig = haar_morton_signature_hyst_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, cx, cy, scale_px, tpl_sig, true,
    );
    if !out_cand_sig.is_null() {
        *out_cand_sig = cand_sig;
    }

    let val = padic_ultrametric_val(tpl_sig, cand_sig);
    let d2 = padic_distance_from_val(val);

    sad_l + 1.85 * sad_c + 0.55 * d2 - 0.16 * motion_reward
}

/// Pinned-Anchor P-N Drift-Lock distance: computes foreground-weighted L1 photometric divergence
/// against the immutable birth anchor template (`anchor_luma`, `anchor_rg`, `anchor_by`).
#[inline]
pub unsafe fn weighted_anchor_l1_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    anchor_luma: *const f32,
    anchor_rg: *const f32,
    anchor_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
) -> f32 {
    let mut sum_w_diff = f32x4_splat(0.0);
    let cstep = 4 * scale_px;
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
            let al = v128_load(anchor_luma.add(toff) as *const v128);
            sum_w_diff = f32x4_add(sum_w_diff, f32x4_mul(vw, f32x4_sub(il, al)));
        }
    }
    let dc = hsum_f32x4(sum_w_diff).clamp(-0.16, 0.16) * 0.75;
    let v_dc = f32x4_splat(dc);

    let mut acc_l = f32x4_splat(0.0);
    let mut acc_c = f32x4_splat(0.0);
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let dl = f32x4_abs(f32x4_sub(
                f32x4_sub(load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px), v_dc),
                v128_load(anchor_luma.add(toff) as *const v128),
            ));
            acc_l = f32x4_add(acc_l, f32x4_mul(vw, dl));
            let dc_rg = f32x4_abs(f32x4_sub(
                load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px),
                v128_load(anchor_rg.add(toff) as *const v128),
            ));
            let dc_by = f32x4_abs(f32x4_sub(
                load_pooled4_f32(by_ptr.add(ioff), stride, scale_px),
                v128_load(anchor_by.add(toff) as *const v128),
            ));
            acc_c = f32x4_add(acc_c, f32x4_mul(vw, f32x4_add(dc_rg, dc_by)));
        }
    }
    hsum_f32x4(acc_l) + 1.65 * hsum_f32x4(acc_c)
}

/// Foreground-weighted continuous Photometric SSD (`tpl_weight` sum = 1.0) for sub-pixel refinement.
pub unsafe fn ssd16x16_weighted_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
) -> f32 {
    let mut sum_w_diff = f32x4_splat(0.0);
    let cstep = 4 * scale_px;
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            sum_w_diff = f32x4_add(sum_w_diff, f32x4_mul(vw, f32x4_sub(il, tl)));
        }
    }
    let dc_shift = hsum_f32x4(sum_w_diff).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(dc_shift);

    let mut acc_ssd_l = f32x4_splat(0.0);
    let mut acc_ssd_c = f32x4_splat(0.0);

    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);

            let dl = f32x4_sub(
                f32x4_sub(load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px), v_dc),
                v128_load(tpl_luma.add(toff) as *const v128),
            );
            acc_ssd_l = f32x4_add(acc_ssd_l, f32x4_mul(vw, f32x4_mul(dl, dl)));

            let drg = f32x4_sub(
                load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px),
                v128_load(tpl_rg.add(toff) as *const v128),
            );
            let dby = f32x4_sub(
                load_pooled4_f32(by_ptr.add(ioff), stride, scale_px),
                v128_load(tpl_by.add(toff) as *const v128),
            );
            acc_ssd_c = f32x4_add(
                acc_ssd_c,
                f32x4_mul(vw, f32x4_add(f32x4_mul(drg, drg), f32x4_mul(dby, dby))),
            );
        }
    }

    hsum_f32x4(acc_ssd_l) + 1.85 * hsum_f32x4(acc_ssd_c)
}

#[inline(always)]
unsafe fn sample_plane_bilinear(ptr: *const f32, stride: usize, fx: f32, fy: f32) -> f32 {
    let ix = fx as usize;
    let iy = fy as usize;
    let tx = fx - (ix as f32);
    let ty = fy - (iy as f32);
    let base = iy * stride + ix;
    let p00 = *ptr.add(base);
    let p10 = *ptr.add(base + 1);
    let p01 = *ptr.add(base + stride);
    let p11 = *ptr.add(base + stride + 1);
    let top = p00 + tx * (p10 - p00);
    let bot = p01 + tx * (p11 - p01);
    top + ty * (bot - top)
}

#[inline(always)]
unsafe fn ic_lk_sample1_blk<const S: usize, const MODE: u8>(
    ptr: *const f32,
    stride: usize,
    base: usize,
    is_tail: bool,
    w00: v128,
    w10: v128,
    w01: v128,
    w11: v128,
) -> v128 {
    if S == 1 {
        if MODE == 0 {
            v128_load(ptr.add(base) as *const v128)
        } else {
            let p00 = v128_load(ptr.add(base) as *const v128);
            let p10 = v128_load(ptr.add(base + 1) as *const v128);
            let p01 = v128_load(ptr.add(base + stride) as *const v128);
            let p11 = v128_load(ptr.add(base + stride + 1) as *const v128);
            f32x4_add(
                f32x4_add(f32x4_mul(p00, w00), f32x4_mul(p10, w10)),
                f32x4_add(f32x4_mul(p01, w01), f32x4_mul(p11, w11)),
            )
        }
    } else if !is_tail {
        let r0a = v128_load(ptr.add(base) as *const v128);
        let r0b = v128_load(ptr.add(base + 4) as *const v128);
        let r1a = v128_load(ptr.add(base + stride) as *const v128);
        let r1b = v128_load(ptr.add(base + stride + 4) as *const v128);
        if MODE == 1 {
            // Caller scales sum by 0.25 once after accumulation.
            let s0 = f32x4_add(r0a, r1a);
            let s1 = f32x4_add(r0b, r1b);
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(s0, s1),
                i32x4_shuffle::<1, 3, 5, 7>(s0, s1),
            )
        } else {
            let p00 = i32x4_shuffle::<0, 2, 4, 6>(r0a, r0b);
            let p10 = i32x4_shuffle::<1, 3, 5, 7>(r0a, r0b);
            let p01 = i32x4_shuffle::<0, 2, 4, 6>(r1a, r1b);
            let p11 = i32x4_shuffle::<1, 3, 5, 7>(r1a, r1b);
            f32x4_add(
                f32x4_add(f32x4_mul(p00, w00), f32x4_mul(p10, w10)),
                f32x4_add(f32x4_mul(p01, w01), f32x4_mul(p11, w11)),
            )
        }
    } else {
        let r0a = v128_load(ptr.add(base) as *const v128);
        let r1a = v128_load(ptr.add(base + stride) as *const v128);
        if MODE == 1 {
            // Caller scales sum by 0.25 once after accumulation.
            let s0 = f32x4_add(r0a, r1a);
            f32x4_add(
                i32x4_shuffle::<0, 0, 0, 2>(s0, s0),
                i32x4_shuffle::<1, 1, 1, 3>(s0, s0),
            )
        } else {
            let p00 = i32x4_shuffle::<0, 0, 0, 2>(r0a, r0a);
            let p10 = i32x4_shuffle::<1, 1, 1, 3>(r0a, r0a);
            let p01 = i32x4_shuffle::<0, 0, 0, 2>(r1a, r1a);
            let p11 = i32x4_shuffle::<1, 1, 1, 3>(r1a, r1a);
            f32x4_add(
                f32x4_add(f32x4_mul(p00, w00), f32x4_mul(p10, w10)),
                f32x4_add(f32x4_mul(p01, w01), f32x4_mul(p11, w11)),
            )
        }
    }
}

#[inline(never)]
unsafe fn ic_lk_grad_iter_simd<const S: usize, const MODE: u8>(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    best_tlx: usize,
    best_tly: usize,
    ix0: usize,
    iy0: usize,
    tx: f32,
    ty: f32,
    tpl_weight: *const f32,
    buf_wxl: &[v128; 56],
    buf_wyl: &[v128; 56],
    buf_wxrg: &[v128; 56],
    buf_wyrg: &[v128; 56],
    buf_wxby: &[v128; 56],
    buf_wyby: &[v128; 56],
    bx_tpl_raw: f32,
    by_tpl_raw: f32,
) -> (f32, f32, f32) {
    let omtx = 1.0 - tx;
    let omty = 1.0 - ty;
    let w00 = f32x4_splat(omtx * omty);
    let w10 = f32x4_splat(tx * omty);
    let w01 = f32x4_splat(omtx * ty);
    let w11 = f32x4_splat(tx * ty);
    let tail_mask = i32x4(0, 0, -1, -1);

    let base0 = (best_tly + iy0) * stride + best_tlx + ix0;
    let row_stride = S * stride;
    let tail_img = if S == 1 { 10 } else { 24 };

    let mut sum_w_il_v = f32x4_splat(0.0);
    let mut bx_l_v = f32x4_splat(0.0);
    let mut by_l_v = f32x4_splat(0.0);
    let mut bx_rg_v = f32x4_splat(0.0);
    let mut by_rg_v = f32x4_splat(0.0);
    let mut bx_by_v = f32x4_splat(0.0);
    let mut by_by_v = f32x4_splat(0.0);
    let mut iy_row = base0;

    for ry in 0..14usize {
        let trow = (ry + 1) * 16;
        let b0 = ry * 4;
        macro_rules! blk_l {
            ($c:expr, $rx_tpl:expr, $img_off:expr, $is_tail:expr) => {
                let k = b0 + $c;
                let mut w = v128_load(tpl_weight.add(trow + $rx_tpl) as *const v128);
                if $is_tail {
                    w = v128_and(w, tail_mask);
                }
                let il = ic_lk_sample1_blk::<S, MODE>(
                    luma_ptr, stride, iy_row + $img_off, $is_tail, w00, w10, w01, w11,
                );
                sum_w_il_v = f32x4_add(sum_w_il_v, f32x4_mul(w, il));
                bx_l_v = f32x4_add(bx_l_v, f32x4_mul(*buf_wxl.get_unchecked(k), il));
                by_l_v = f32x4_add(by_l_v, f32x4_mul(*buf_wyl.get_unchecked(k), il));
            };
        }
        blk_l!(0, 1, 0, false);
        blk_l!(1, 5, 4 * S, false);
        blk_l!(2, 9, 8 * S, false);
        blk_l!(3, 11, tail_img, true);

        macro_rules! blk_rg {
            ($c:expr, $img_off:expr, $is_tail:expr) => {
                let k = b0 + $c;
                let val = ic_lk_sample1_blk::<S, MODE>(
                    rg_ptr, stride, iy_row + $img_off, $is_tail, w00, w10, w01, w11,
                );
                bx_rg_v = f32x4_add(bx_rg_v, f32x4_mul(*buf_wxrg.get_unchecked(k), val));
                by_rg_v = f32x4_add(by_rg_v, f32x4_mul(*buf_wyrg.get_unchecked(k), val));
            };
        }
        blk_rg!(0, 0, false);
        blk_rg!(1, 4 * S, false);
        blk_rg!(2, 8 * S, false);
        blk_rg!(3, tail_img, true);

        macro_rules! blk_by {
            ($c:expr, $img_off:expr, $is_tail:expr) => {
                let k = b0 + $c;
                let val = ic_lk_sample1_blk::<S, MODE>(
                    by_ptr, stride, iy_row + $img_off, $is_tail, w00, w10, w01, w11,
                );
                bx_by_v = f32x4_add(bx_by_v, f32x4_mul(*buf_wxby.get_unchecked(k), val));
                by_by_v = f32x4_add(by_by_v, f32x4_mul(*buf_wyby.get_unchecked(k), val));
            };
        }
        blk_by!(0, 0, false);
        blk_by!(1, 4 * S, false);
        blk_by!(2, 8 * S, false);
        blk_by!(3, tail_img, true);

        iy_row += row_stride;
    }

    let bx_v = f32x4_add(bx_l_v, f32x4_add(bx_rg_v, bx_by_v));
    let by_v = f32x4_add(by_l_v, f32x4_add(by_rg_v, by_by_v));
    let scale = if S == 2 && MODE == 1 { 0.25 } else { 1.0 };
    (
        hsum_f32x4(sum_w_il_v) * scale,
        hsum_f32x4(bx_v) * scale - bx_tpl_raw,
        hsum_f32x4(by_v) * scale - by_tpl_raw,
    )
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct TplIcLkCache {
    pub buf_wxl: [v128; 56],
    pub buf_wyl: [v128; 56],
    pub buf_wxrg: [v128; 56],
    pub buf_wyrg: [v128; 56],
    pub buf_wxby: [v128; 56],
    pub buf_wyby: [v128; 56],
    pub h_xx_raw: f32,
    pub h_xy_raw: f32,
    pub h_yy_raw: f32,
    pub sum_w_tl: f32,
    pub sum_w_in: f32,
    pub gx_w_sum_raw: f32,
    pub gy_w_sum_raw: f32,
    pub tpl_inv_w: f32,
    pub tpl_mu_t: f32,
    pub full_inv_w: f32,
    pub full_mu_t: f32,
    pub bx_tpl_raw: f32,
    pub by_tpl_raw: f32,
}

#[inline(never)]
pub unsafe fn build_tpl_ic_lk_cache(
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    cache: &mut TplIcLkCache,
) {
    #[inline(always)]
    unsafe fn load_r4(ptr: *const f32, row: usize) -> [v128; 4] {
        let p = ptr.add(row * 16);
        [
            v128_load(p as *const v128),
            v128_load(p.add(4) as *const v128),
            v128_load(p.add(8) as *const v128),
            v128_load(p.add(12) as *const v128),
        ]
    }
    #[inline(always)]
    fn ext14(r: [v128; 4]) -> [v128; 4] {
        [
            i32x4_shuffle::<1, 2, 3, 4>(r[0], r[1]),
            i32x4_shuffle::<1, 2, 3, 4>(r[1], r[2]),
            i32x4_shuffle::<1, 2, 3, 4>(r[2], r[3]),
            i32x4_shuffle::<3, 4, 5, 6>(r[2], r[3]),
        ]
    }
    #[inline(always)]
    fn dx14(r: [v128; 4]) -> [v128; 4] {
        [
            f32x4_sub(i32x4_shuffle::<2, 3, 4, 5>(r[0], r[1]), r[0]),
            f32x4_sub(i32x4_shuffle::<2, 3, 4, 5>(r[1], r[2]), r[1]),
            f32x4_sub(i32x4_shuffle::<2, 3, 4, 5>(r[2], r[3]), r[2]),
            f32x4_sub(r[3], i32x4_shuffle::<2, 3, 4, 5>(r[2], r[3])),
        ]
    }
    #[inline(always)]
    fn dy14(prev: [v128; 4], next: [v128; 4]) -> [v128; 4] {
        ext14([
            f32x4_sub(next[0], prev[0]),
            f32x4_sub(next[1], prev[1]),
            f32x4_sub(next[2], prev[2]),
            f32x4_sub(next[3], prev[3]),
        ])
    }

    let zero_v = f32x4_splat(0.0);
    let mut ew0 = zero_v;
    let mut ew1 = zero_v;
    let mut ewt0 = zero_v;
    let mut ewt1 = zero_v;
    let mut ow0 = zero_v;
    let mut ow1 = zero_v;
    let mut owt0 = zero_v;
    let mut owt1 = zero_v;

    let mut l_prev = load_r4(tpl_luma, 0);
    let w_r0 = load_r4(tpl_weight, 0);
    ew0 = f32x4_add(ew0, f32x4_add(w_r0[0], w_r0[2]));
    ew1 = f32x4_add(ew1, f32x4_add(w_r0[1], w_r0[3]));
    ewt0 = f32x4_add(ewt0, f32x4_add(f32x4_mul(w_r0[0], l_prev[0]), f32x4_mul(w_r0[2], l_prev[2])));
    ewt1 = f32x4_add(ewt1, f32x4_add(f32x4_mul(w_r0[1], l_prev[1]), f32x4_mul(w_r0[3], l_prev[3])));

    let mut l_cur = load_r4(tpl_luma, 1);
    let mut rg_prev = load_r4(tpl_rg, 0);
    let mut rg_cur = load_r4(tpl_rg, 1);
    let mut by_prev = load_r4(tpl_by, 0);
    let mut by_cur = load_r4(tpl_by, 1);

    let v_14 = f32x4_splat(1.4);
    let tail_mask = i32x4(0, 0, -1, -1);

    let mut h_xx_v = zero_v;
    let mut h_xy_v = zero_v;
    let mut h_yy_v = zero_v;
    let mut sum_w_tl_v = zero_v;
    let mut sum_w_in_v = zero_v;
    let mut gx_w_sum_v = zero_v;
    let mut gy_w_sum_v = zero_v;
    let mut bx_tpl_v = zero_v;
    let mut by_tpl_v = zero_v;

    for row in 1..15usize {
        let l_next = load_r4(tpl_luma, row + 1);
        let rg_next = load_r4(tpl_rg, row + 1);
        let by_next = load_r4(tpl_by, row + 1);
        let w_row = load_r4(tpl_weight, row);
        if (row & 1) == 0 {
            ew0 = f32x4_add(ew0, f32x4_add(w_row[0], w_row[2]));
            ew1 = f32x4_add(ew1, f32x4_add(w_row[1], w_row[3]));
            ewt0 = f32x4_add(ewt0, f32x4_add(f32x4_mul(w_row[0], l_cur[0]), f32x4_mul(w_row[2], l_cur[2])));
            ewt1 = f32x4_add(ewt1, f32x4_add(f32x4_mul(w_row[1], l_cur[1]), f32x4_mul(w_row[3], l_cur[3])));
        } else {
            ow0 = f32x4_add(ow0, f32x4_add(w_row[0], w_row[2]));
            ow1 = f32x4_add(ow1, f32x4_add(w_row[1], w_row[3]));
            owt0 = f32x4_add(owt0, f32x4_add(f32x4_mul(w_row[0], l_cur[0]), f32x4_mul(w_row[2], l_cur[2])));
            owt1 = f32x4_add(owt1, f32x4_add(f32x4_mul(w_row[1], l_cur[1]), f32x4_mul(w_row[3], l_cur[3])));
        }

        let w_blk = ext14(w_row);
        let tl_blk = ext14(l_cur);
        let dtx_blk = dx14(l_cur);
        let dty_blk = dy14(l_prev, l_next);
        let trg_blk = ext14(rg_cur);
        let drgx_blk = dx14(rg_cur);
        let drgy_blk = dy14(rg_prev, rg_next);
        let tby_blk = ext14(by_cur);
        let dbyx_blk = dx14(by_cur);
        let dbyy_blk = dy14(by_prev, by_next);

        let b0 = (row - 1) * 4;
        for c in 0..4usize {
            let mut w = w_blk[c];
            if c == 3 {
                w = v128_and(w, tail_mask);
            }
            let k = b0 + c;
            let tl = tl_blk[c];
            sum_w_tl_v = f32x4_add(sum_w_tl_v, f32x4_mul(w, tl));
            sum_w_in_v = f32x4_add(sum_w_in_v, w);

            let dtx = dtx_blk[c];
            let dty = dty_blk[c];
            let wx_l = f32x4_mul(w, dtx);
            let wy_l = f32x4_mul(w, dty);
            cache.buf_wxl[k] = wx_l;
            cache.buf_wyl[k] = wy_l;
            gx_w_sum_v = f32x4_add(gx_w_sum_v, wx_l);
            gy_w_sum_v = f32x4_add(gy_w_sum_v, wy_l);
            h_xx_v = f32x4_add(h_xx_v, f32x4_mul(wx_l, dtx));
            h_xy_v = f32x4_add(h_xy_v, f32x4_mul(wx_l, dty));
            h_yy_v = f32x4_add(h_yy_v, f32x4_mul(wy_l, dty));

            let w_14 = f32x4_mul(w, v_14);
            let trg = trg_blk[c];
            let drgx = drgx_blk[c];
            let drgy = drgy_blk[c];
            let wx_rg = f32x4_mul(w_14, drgx);
            let wy_rg = f32x4_mul(w_14, drgy);
            cache.buf_wxrg[k] = wx_rg;
            cache.buf_wyrg[k] = wy_rg;
            h_xx_v = f32x4_add(h_xx_v, f32x4_mul(wx_rg, drgx));
            h_xy_v = f32x4_add(h_xy_v, f32x4_mul(wx_rg, drgy));
            h_yy_v = f32x4_add(h_yy_v, f32x4_mul(wy_rg, drgy));

            let tby = tby_blk[c];
            let dbyx = dbyx_blk[c];
            let dbyy = dbyy_blk[c];
            let wx_by = f32x4_mul(w_14, dbyx);
            let wy_by = f32x4_mul(w_14, dbyy);
            cache.buf_wxby[k] = wx_by;
            cache.buf_wyby[k] = wy_by;
            h_xx_v = f32x4_add(h_xx_v, f32x4_mul(wx_by, dbyx));
            h_xy_v = f32x4_add(h_xy_v, f32x4_mul(wx_by, dbyy));
            h_yy_v = f32x4_add(h_yy_v, f32x4_mul(wy_by, dbyy));

            bx_tpl_v = f32x4_add(
                bx_tpl_v,
                f32x4_add(
                    f32x4_mul(wx_l, tl),
                    f32x4_add(f32x4_mul(wx_rg, trg), f32x4_mul(wx_by, tby)),
                ),
            );
            by_tpl_v = f32x4_add(
                by_tpl_v,
                f32x4_add(
                    f32x4_mul(wy_l, tl),
                    f32x4_add(f32x4_mul(wy_rg, trg), f32x4_mul(wy_by, tby)),
                ),
            );
        }

        l_prev = l_cur;
        l_cur = l_next;
        rg_prev = rg_cur;
        rg_cur = rg_next;
        by_prev = by_cur;
        by_cur = by_next;
    }

    let w_r15 = load_r4(tpl_weight, 15);
    ow0 = f32x4_add(ow0, f32x4_add(w_r15[0], w_r15[2]));
    ow1 = f32x4_add(ow1, f32x4_add(w_r15[1], w_r15[3]));
    owt0 = f32x4_add(owt0, f32x4_add(f32x4_mul(w_r15[0], l_cur[0]), f32x4_mul(w_r15[2], l_cur[2])));
    owt1 = f32x4_add(owt1, f32x4_add(f32x4_mul(w_r15[1], l_cur[1]), f32x4_mul(w_r15[3], l_cur[3])));

    let even_w = f32x4_add(ew0, ew1);
    let even_wt = f32x4_add(ewt0, ewt1);
    let odd_w = f32x4_add(ow0, ow1);
    let odd_wt = f32x4_add(owt0, owt1);
    let inv_w = 1.0 / hsum_f32x4(even_w).max(1e-6);
    cache.tpl_inv_w = inv_w;
    cache.tpl_mu_t = hsum_f32x4(even_wt) * inv_w;
    let full_inv_w = 1.0 / hsum_f32x4(f32x4_add(even_w, odd_w)).max(1e-6);
    cache.full_inv_w = full_inv_w;
    cache.full_mu_t = hsum_f32x4(f32x4_add(even_wt, odd_wt)) * full_inv_w;

    cache.h_xx_raw = hsum_f32x4(h_xx_v);
    cache.h_xy_raw = hsum_f32x4(h_xy_v);
    cache.h_yy_raw = hsum_f32x4(h_yy_v);
    cache.sum_w_tl = hsum_f32x4(sum_w_tl_v);
    cache.sum_w_in = hsum_f32x4(sum_w_in_v);
    cache.gx_w_sum_raw = hsum_f32x4(gx_w_sum_v);
    cache.gy_w_sum_raw = hsum_f32x4(gy_w_sum_v);
    cache.bx_tpl_raw = hsum_f32x4(bx_tpl_v);
    cache.by_tpl_raw = hsum_f32x4(by_tpl_v);
}

#[inline(always)]
unsafe fn ic_lk_solve_from_cache(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    best_tlx: usize,
    best_tly: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
    cache: &TplIcLkCache,
) -> (f32, f32) {
    let s_f = scale_px as f32;
    let center_off = 0.5 * ((scale_px - 1) as f32);
    let inv_2sf = 0.5 / s_f;
    let inv_2sf2 = inv_2sf * inv_2sf;
    let h_xx = cache.h_xx_raw * inv_2sf2;
    let h_xy = cache.h_xy_raw * inv_2sf2;
    let h_yy = cache.h_yy_raw * inv_2sf2;
    let sum_w_tl = cache.sum_w_tl;
    let sum_w_in = cache.sum_w_in;
    let gx_w_sum = cache.gx_w_sum_raw * inv_2sf;
    let gy_w_sum = cache.gy_w_sum_raw * inv_2sf;

    let det = h_xx * h_yy - h_xy * h_xy;
    let tr = h_xx + h_yy;
    let max_shift = 0.62 * s_f;

    // Require genuine 2D Hessian conditioning (det > 1e-7 and condition ratio healthy)
    if det > 1e-7 && det > 0.02 * tr * tr && sum_w_in > 0.25 {
        let inv_det = 1.0 / det;
        let inv_win = 1.0 / sum_w_in;
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;

        for _iter in 0..2usize {
            let sx = (s_f + center_off + dx).max(0.0);
            let sy = (s_f + center_off + dy).max(0.0);
            let ix0 = sx as usize;
            let iy0 = sy as usize;
            let tx = sx - (ix0 as f32);
            let ty = sy - (iy0 as f32);

            let (sum_w_il, bx_raw, by_raw) = if scale_px == 2 {
                if tx == 0.5 && ty == 0.5 {
                    ic_lk_grad_iter_simd::<2, 1>(
                        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly, ix0, iy0, tx, ty,
                        tpl_weight,
                        &cache.buf_wxl, &cache.buf_wyl, &cache.buf_wxrg, &cache.buf_wyrg,
                        &cache.buf_wxby, &cache.buf_wyby,
                        cache.bx_tpl_raw, cache.by_tpl_raw,
                    )
                } else {
                    ic_lk_grad_iter_simd::<2, 2>(
                        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly, ix0, iy0, tx, ty,
                        tpl_weight,
                        &cache.buf_wxl, &cache.buf_wyl, &cache.buf_wxrg, &cache.buf_wyrg,
                        &cache.buf_wxby, &cache.buf_wyby,
                        cache.bx_tpl_raw, cache.by_tpl_raw,
                    )
                }
            } else if scale_px == 1 {
                if tx == 0.0 && ty == 0.0 {
                    ic_lk_grad_iter_simd::<1, 0>(
                        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly, ix0, iy0, tx, ty,
                        tpl_weight,
                        &cache.buf_wxl, &cache.buf_wyl, &cache.buf_wxrg, &cache.buf_wyrg,
                        &cache.buf_wxby, &cache.buf_wyby,
                        cache.bx_tpl_raw, cache.by_tpl_raw,
                    )
                } else {
                    ic_lk_grad_iter_simd::<1, 2>(
                        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly, ix0, iy0, tx, ty,
                        tpl_weight,
                        &cache.buf_wxl, &cache.buf_wyl, &cache.buf_wxrg, &cache.buf_wyrg,
                        &cache.buf_wxby, &cache.buf_wyby,
                        cache.bx_tpl_raw, cache.by_tpl_raw,
                    )
                }
            } else {
                let mut sw = 0.0f32;
                let mut bxr = 0.0f32;
                let mut byr = 0.0f32;
                for ry in 1..15usize {
                    let fy = (best_tly + ry * scale_px) as f32 + center_off + dy;
                    let trow = ry * 16;
                    for rx in 1..15usize {
                        let idx = trow + rx;
                        let fx = (best_tlx + rx * scale_px) as f32 + center_off + dx;
                        let w = *tpl_weight.add(idx);
                        let il = sample_plane_bilinear(luma_ptr, stride, fx, fy);
                        sw += w * il;
                        let err_l0 = il - *tpl_luma.add(idx);
                        let err_rg = sample_plane_bilinear(rg_ptr, stride, fx, fy) - *tpl_rg.add(idx);
                        let err_by = sample_plane_bilinear(by_ptr, stride, fx, fy) - *tpl_by.add(idx);
                        let gtx = *tpl_luma.add(idx + 1) - *tpl_luma.add(idx - 1);
                        let gty = *tpl_luma.add(idx + 16) - *tpl_luma.add(idx - 16);
                        let grgx = *tpl_rg.add(idx + 1) - *tpl_rg.add(idx - 1);
                        let grgy = *tpl_rg.add(idx + 16) - *tpl_rg.add(idx - 16);
                        let gbyx = *tpl_by.add(idx + 1) - *tpl_by.add(idx - 1);
                        let gbyy = *tpl_by.add(idx + 16) - *tpl_by.add(idx - 16);
                        bxr += w * (gtx * err_l0 + 1.4 * (grgx * err_rg + gbyx * err_by));
                        byr += w * (gty * err_l0 + 1.4 * (grgy * err_rg + gbyy * err_by));
                    }
                }
                (sw, bxr, byr)
            };

            let dc = ((sum_w_il - sum_w_tl) * inv_win).clamp(-0.14, 0.14) * 0.70;
            let bx = bx_raw * inv_2sf - dc * gx_w_sum;
            let by = by_raw * inv_2sf - dc * gy_w_sum;

            // Inverse-compositional Gauss-Newton step: I(x + p + dp) ~ T(x) => dp = -H^-1 b
            let step_x = -(h_yy * bx - h_xy * by) * inv_det;
            let step_y = -(-h_xy * bx + h_xx * by) * inv_det;
            dx = (dx + step_x).clamp(-max_shift, max_shift);
            dy = (dy + step_y).clamp(-max_shift, max_shift);
            if step_x.abs() < 0.015 && step_y.abs() < 0.015 {
                break;
            }
        }
        if dx.is_finite() && dy.is_finite() {
            return (dx, dy);
        }
    }

    // Fallback: Separable foreground-weighted parabolic SSD
    let c_0 = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let c_xm = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx - 1, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let c_xp = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx + 1, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let mut sub_dx = 0.0f32;
    let denom_x = 2.0 * (c_xm - 2.0 * c_0 + c_xp);
    if denom_x > 1e-5 {
        sub_dx = ((c_xm - c_xp) / denom_x).clamp(-0.48, 0.48);
    }

    let c_ym = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly - 1,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let c_yp = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly + 1,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let mut sub_dy = 0.0f32;
    let denom_y = 2.0 * (c_ym - 2.0 * c_0 + c_yp);
    if denom_y > 1e-5 {
        sub_dy = ((c_ym - c_yp) / denom_y).clamp(-0.48, 0.48);
    }
    (sub_dx, sub_dy)
}

#[inline]
pub unsafe fn refine_subpixel_ic_lk_cached(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    best_tlx: usize,
    best_tly: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
    cache: &mut TplIcLkCache,
    dirty: &mut u8,
) -> (f32, f32) {
    if *dirty != 0 {
        build_tpl_ic_lk_cache(tpl_luma, tpl_rg, tpl_by, tpl_weight, cache);
        *dirty = 0;
    }
    ic_lk_solve_from_cache(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px, cache,
    )
}

/// 2D Foreground-Weighted Inverse-Compositional Lucas-Kanade (IC-LK) Sub-Pixel Refinement
/// with separable weighted parabolic SSD fallback when template Hessian is ill-conditioned.
#[inline(never)]
pub unsafe fn refine_subpixel_ic_lk_weighted(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    best_tlx: usize,
    best_tly: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
) -> (f32, f32) {
    let z = i32x4_splat(0);
    let mut cache_val = TplIcLkCache {
        buf_wxl: [z; 56],
        buf_wyl: [z; 56],
        buf_wxrg: [z; 56],
        buf_wyrg: [z; 56],
        buf_wxby: [z; 56],
        buf_wyby: [z; 56],
        h_xx_raw: 0.0,
        h_xy_raw: 0.0,
        h_yy_raw: 0.0,
        sum_w_tl: 0.0,
        sum_w_in: 0.0,
        gx_w_sum_raw: 0.0,
        gy_w_sum_raw: 0.0,
        tpl_inv_w: 0.0,
        tpl_mu_t: 0.0,
        full_inv_w: 0.0,
        full_mu_t: 0.0,
        bx_tpl_raw: 0.0,
        by_tpl_raw: 0.0,
    };
    let cache = &mut cache_val;
    build_tpl_ic_lk_cache(tpl_luma, tpl_rg, tpl_by, tpl_weight, cache);
    ic_lk_solve_from_cache(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px, cache,
    )
}

#[inline]
pub unsafe fn sad64_demean_gain_norm_simd128(
    cur_patch: &[f32; 64],
    cur_mad: f32,
    prev_vals: &[f32; 64],
) -> f32 {
    let p_ptr = prev_vals.as_ptr();
    let c_ptr = cur_patch.as_ptr();
    let mut pv = [f32x4_splat(0.0); 16];
    let mut sum_v = f32x4_splat(0.0);
    for i in 0..16usize {
        let v = v128_load(p_ptr.add(i * 4) as *const v128);
        pv[i] = v;
        sum_v = f32x4_add(sum_v, v);
    }
    let mp = hsum_f32x4(sum_v) * (1.0 / 64.0);
    let mp_v = f32x4_splat(mp);
    let mut mad_v = f32x4_splat(0.0);
    for i in 0..16usize {
        let d = f32x4_sub(pv[i], mp_v);
        pv[i] = d;
        mad_v = f32x4_add(mad_v, f32x4_abs(d));
    }
    let mad_p = hsum_f32x4(mad_v);
    let g_scale = (cur_mad / mad_p.max(1e-3)).clamp(0.75, 1.33);
    let g_v = f32x4_splat(g_scale);
    let mut sad_v = f32x4_splat(0.0);
    for i in 0..16usize {
        let cur = v128_load(c_ptr.add(i * 4) as *const v128);
        let diff = f32x4_sub(cur, f32x4_mul(pv[i], g_v));
        sad_v = f32x4_add(sad_v, f32x4_abs(diff));
    }
    hsum_f32x4(sad_v)
}

#[inline]
pub unsafe fn eval_win_patch_sad_simd128(
    cur_q: &[f32; 64],
    mad_cq: f32,
    prev_win: &[f32],
    win_dim: usize,
    ox: usize,
    oy: usize,
    scale_px: usize,
) -> f32 {
    if scale_px <= 1
        && win_dim >= ox + 8
        && prev_win.len() >= (oy + 7) * win_dim + ox + 8
    {
        let w_ptr = prev_win.as_ptr();
        let c_ptr = cur_q.as_ptr();
        let zero_v = f32x4_splat(0.0);
        let mut pv = [zero_v; 16];
        let mut sum_v = zero_v;
        for r in 0..8usize {
            let row_ptr = w_ptr.add((oy + r) * win_dim + ox);
            let v0 = v128_load(row_ptr as *const v128);
            let v1 = v128_load(row_ptr.add(4) as *const v128);
            pv[r * 2] = v0;
            pv[r * 2 + 1] = v1;
            sum_v = f32x4_add(sum_v, f32x4_add(v0, v1));
        }
        let mp_v = f32x4_splat(hsum_f32x4(sum_v) * (1.0 / 64.0));
        let mut mad_v = zero_v;
        for i in 0..16usize {
            let d = f32x4_sub(pv[i], mp_v);
            pv[i] = d;
            mad_v = f32x4_add(mad_v, f32x4_abs(d));
        }
        let g_v = f32x4_splat((mad_cq / hsum_f32x4(mad_v).max(1e-3)).clamp(0.75, 1.33));
        let mut sad_v = zero_v;
        for i in 0..16usize {
            let cur = v128_load(c_ptr.add(i * 4) as *const v128);
            sad_v = f32x4_add(sad_v, f32x4_abs(f32x4_sub(cur, f32x4_mul(pv[i], g_v))));
        }
        return hsum_f32x4(sad_v) * (1.0 / 64.0);
    }

    if scale_px == 2
        && win_dim >= ox + 16
        && prev_win.len() >= (oy + 15) * win_dim + ox + 16
    {
        let w_ptr = prev_win.as_ptr();
        let c_ptr = cur_q.as_ptr();
        let zero_v = f32x4_splat(0.0);
        let mut pv = [zero_v; 16];
        let mut sum_v = zero_v;
        for r in 0..8usize {
            let phase = r & 1;
            let row0_base = (oy + 2 * r) * win_dim + ox + phase;
            let row1_base = row0_base + win_dim;
            for half in 0..2usize {
                let off = half * 8;
                let r0_ptr = w_ptr.add(row0_base + off);
                let r1_ptr = w_ptr.add(row1_base + off);
                let a0 = v128_load(r0_ptr as *const v128);
                let a1 = v128_load(r0_ptr.add(3) as *const v128);
                let b0 = v128_load(r1_ptr as *const v128);
                let b1 = v128_load(r1_ptr.add(3) as *const v128);
                let evens = i32x4_shuffle::<0, 5, 0, 5>(a0, a1);
                let odds = i32x4_shuffle::<2, 7, 2, 7>(b0, b1);
                let v = i32x4_shuffle::<0, 4, 1, 5>(evens, odds);
                pv[r * 2 + half] = v;
                sum_v = f32x4_add(sum_v, v);
            }
        }
        let mp_v = f32x4_splat(hsum_f32x4(sum_v) * (1.0 / 64.0));
        let mut mad_v = zero_v;
        for i in 0..16usize {
            let d = f32x4_sub(pv[i], mp_v);
            pv[i] = d;
            mad_v = f32x4_add(mad_v, f32x4_abs(d));
        }
        let g_v = f32x4_splat((mad_cq / hsum_f32x4(mad_v).max(1e-3)).clamp(0.75, 1.33));
        let mut sad_v = zero_v;
        for i in 0..16usize {
            let cur = v128_load(c_ptr.add(i * 4) as *const v128);
            sad_v = f32x4_add(sad_v, f32x4_abs(f32x4_sub(cur, f32x4_mul(pv[i], g_v))));
        }
        return hsum_f32x4(sad_v) * (1.0 / 64.0);
    }

    let phase_mask = scale_px.saturating_sub(1);
    let mut pv = [0.0f32; 64];
    let mut idx = 0usize;
    for r in 0..8usize {
        for c in 0..8usize {
            let ry = r * scale_px + (c & phase_mask);
            let rx = c * scale_px + (r & phase_mask);
            pv[idx] = prev_win[(oy + ry) * win_dim + (ox + rx)];
            idx += 1;
        }
    }
    sad64_demean_gain_norm_simd128(cur_q, mad_cq, &pv) * (1.0 / 64.0)
}

#[inline(always)]
pub unsafe fn reorder_patch64_eo_simd128(cur_q: &[f32; 64], cur_q_eo: &mut [f32; 64]) {
    let src = cur_q.as_ptr();
    let dst = cur_q_eo.as_mut_ptr();
    for r in 0..8usize {
        let a = v128_load(src.add(r * 8) as *const v128);
        let b = v128_load(src.add(r * 8 + 4) as *const v128);
        v128_store(dst.add(r * 8) as *mut v128, i32x4_shuffle::<0, 2, 4, 6>(a, b));
        v128_store(dst.add(r * 8 + 4) as *mut v128, i32x4_shuffle::<1, 3, 5, 7>(a, b));
    }
}

pub unsafe fn pack_win_p2_phases_simd128(
    prev_win: &[f32],
    win_dim: usize,
    planes: &mut [f32],
) -> (usize, usize) {
    let p_h = (win_dim + 1) / 2;
    let p_w = ((win_dim + 3) / 4) + 1;
    let plane_sz = p_h * p_w;
    let src_base = prev_win.as_ptr();
    let dst_base = planes.as_mut_ptr();
    for y in 0..win_dim {
        let row_off = (y >> 1) * p_w;
        let ph_y = (y & 1) * 4;
        let src_row = src_base.add(y * win_dim);
        let d0 = dst_base.add(ph_y * plane_sz + row_off);
        let d1 = dst_base.add((ph_y + 1) * plane_sz + row_off);
        let d2 = dst_base.add((ph_y + 2) * plane_sz + row_off);
        let d3 = dst_base.add((ph_y + 3) * plane_sz + row_off);
        let mut x = 0usize;
        while x + 16 <= win_dim {
            let v0 = v128_load(src_row.add(x) as *const v128);
            let v1 = v128_load(src_row.add(x + 4) as *const v128);
            let v2 = v128_load(src_row.add(x + 8) as *const v128);
            let v3 = v128_load(src_row.add(x + 12) as *const v128);
            let t0 = i32x4_shuffle::<0, 4, 1, 5>(v0, v1);
            let t1 = i32x4_shuffle::<2, 6, 3, 7>(v0, v1);
            let t2 = i32x4_shuffle::<0, 4, 1, 5>(v2, v3);
            let t3 = i32x4_shuffle::<2, 6, 3, 7>(v2, v3);
            let xq = x >> 2;
            v128_store(d0.add(xq) as *mut v128, i32x4_shuffle::<0, 1, 4, 5>(t0, t2));
            v128_store(d1.add(xq) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t0, t2));
            v128_store(d2.add(xq) as *mut v128, i32x4_shuffle::<0, 1, 4, 5>(t1, t3));
            v128_store(d3.add(xq) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t1, t3));
            x += 16;
        }
        while x < win_dim {
            *dst_base.add((ph_y + (x & 3)) * plane_sz + row_off + (x >> 2)) = *src_row.add(x);
            x += 1;
        }
    }
    (p_w, plane_sz)
}

#[inline]
pub unsafe fn eval_win_patch_sad_p2_packed_simd128(
    cur_q_eo: &[f32; 64],
    mad_cq: f32,
    planes: &[f32],
    p_w: usize,
    plane_sz: usize,
    ox: usize,
    oy: usize,
    max_sad64: f32,
) -> f32 {
    let base = planes.as_ptr();
    let c_ptr = cur_q_eo.as_ptr();
    let ph_y0 = (oy & 1) * 4;
    let ph_y1 = ((oy + 1) & 1) * 4;
    let row_y0 = (oy >> 1) * p_w;
    let row_y1 = ((oy + 1) >> 1) * p_w;
    let e0_ptr = base.add((ph_y0 + (ox & 3)) * plane_sz + row_y0 + (ox >> 2));
    let o0_ptr = base.add((ph_y1 + ((ox + 2) & 3)) * plane_sz + row_y1 + ((ox + 2) >> 2));
    let e1_ptr = base.add((ph_y0 + ((ox + 1) & 3)) * plane_sz + row_y0 + ((ox + 1) >> 2));
    let o1_ptr = base.add((ph_y1 + ((ox + 3) & 3)) * plane_sz + row_y1 + ((ox + 3) >> 2));
    let rk0 = 0usize;
    let rk1 = p_w;
    let rk2 = 2 * p_w;
    let rk3 = 3 * p_w;
    let rk4 = 4 * p_w;
    let rk5 = 5 * p_w;
    let rk6 = 6 * p_w;
    let rk7 = 7 * p_w;
    let p0 = v128_load(e0_ptr.add(rk0) as *const v128);
    let p1 = v128_load(o0_ptr.add(rk0) as *const v128);
    let p2 = v128_load(e1_ptr.add(rk1) as *const v128);
    let p3 = v128_load(o1_ptr.add(rk1) as *const v128);
    let p4 = v128_load(e0_ptr.add(rk2) as *const v128);
    let p5 = v128_load(o0_ptr.add(rk2) as *const v128);
    let p6 = v128_load(e1_ptr.add(rk3) as *const v128);
    let p7 = v128_load(o1_ptr.add(rk3) as *const v128);
    let p8 = v128_load(e0_ptr.add(rk4) as *const v128);
    let p9 = v128_load(o0_ptr.add(rk4) as *const v128);
    let p10 = v128_load(e1_ptr.add(rk5) as *const v128);
    let p11 = v128_load(o1_ptr.add(rk5) as *const v128);
    let p12 = v128_load(e0_ptr.add(rk6) as *const v128);
    let p13 = v128_load(o0_ptr.add(rk6) as *const v128);
    let p14 = v128_load(e1_ptr.add(rk7) as *const v128);
    let p15 = v128_load(o1_ptr.add(rk7) as *const v128);
    let s0_a = f32x4_add(p0, p1);
    let s1_a = f32x4_add(p2, p3);
    let s0_b = f32x4_add(p4, p5);
    let s1_b = f32x4_add(p6, p7);
    let s0_c = f32x4_add(p8, p9);
    let s1_c = f32x4_add(p10, p11);
    let s0_d = f32x4_add(p12, p13);
    let s1_d = f32x4_add(p14, p15);
    let sum0 = f32x4_add(f32x4_add(s0_a, s0_b), f32x4_add(s0_c, s0_d));
    let sum1 = f32x4_add(f32x4_add(s1_a, s1_b), f32x4_add(s1_c, s1_d));
    let mp_v = f32x4_splat(hsum_f32x4(f32x4_add(sum0, sum1)) * (1.0 / 64.0));
    let d0 = f32x4_sub(p0, mp_v);
    let d1 = f32x4_sub(p1, mp_v);
    let d2 = f32x4_sub(p2, mp_v);
    let d3 = f32x4_sub(p3, mp_v);
    let d4 = f32x4_sub(p4, mp_v);
    let d5 = f32x4_sub(p5, mp_v);
    let d6 = f32x4_sub(p6, mp_v);
    let d7 = f32x4_sub(p7, mp_v);
    let d8 = f32x4_sub(p8, mp_v);
    let d9 = f32x4_sub(p9, mp_v);
    let d10 = f32x4_sub(p10, mp_v);
    let d11 = f32x4_sub(p11, mp_v);
    let d12 = f32x4_sub(p12, mp_v);
    let d13 = f32x4_sub(p13, mp_v);
    let d14 = f32x4_sub(p14, mp_v);
    let d15 = f32x4_sub(p15, mp_v);
    let mad0 = f32x4_add(
        f32x4_add(f32x4_add(f32x4_abs(d0), f32x4_abs(d2)), f32x4_add(f32x4_abs(d4), f32x4_abs(d6))),
        f32x4_add(f32x4_add(f32x4_abs(d8), f32x4_abs(d10)), f32x4_add(f32x4_abs(d12), f32x4_abs(d14))),
    );
    let mad1 = f32x4_add(
        f32x4_add(f32x4_add(f32x4_abs(d1), f32x4_abs(d3)), f32x4_add(f32x4_abs(d5), f32x4_abs(d7))),
        f32x4_add(f32x4_add(f32x4_abs(d9), f32x4_abs(d11)), f32x4_add(f32x4_abs(d13), f32x4_abs(d15))),
    );
    let g_v = f32x4_splat((mad_cq / hsum_f32x4(f32x4_add(mad0, mad1)).max(1e-3)).clamp(0.75, 1.33));
    let zero_v = f32x4_splat(0.0);
    let mut sad0 = zero_v;
    let mut sad1 = zero_v;
    macro_rules! sad_pair {
        ($i:expr, $da:expr, $db:expr) => {{
            let ca = v128_load(c_ptr.add($i * 4) as *const v128);
            let cb = v128_load(c_ptr.add(($i + 1) * 4) as *const v128);
            sad0 = f32x4_add(sad0, f32x4_abs(f32x4_sub(ca, f32x4_mul($da, g_v))));
            sad1 = f32x4_add(sad1, f32x4_abs(f32x4_sub(cb, f32x4_mul($db, g_v))));
        }};
    }
    sad_pair!(0, d0, d1);
    sad_pair!(2, d2, d3);
    sad_pair!(4, d4, d5);
    sad_pair!(6, d6, d7);
    if max_sad64 < 1e8 && hsum_f32x4(f32x4_add(sad0, sad1)) >= max_sad64 {
        return 1e9f32;
    }
    sad_pair!(8, d8, d9);
    sad_pair!(10, d10, d11);
    sad_pair!(12, d12, d13);
    sad_pair!(14, d14, d15);
    hsum_f32x4(f32x4_add(sad0, sad1)) * (1.0 / 64.0)
}

#[inline]
pub unsafe fn subpx_patch_gain_norm_sad_simd128(
    cur_patch: &[f32; 64],
    cur_mad: f32,
    pl: &[f32],
    scale_px: usize,
    phase_mask: usize,
    off_x_base: usize,
    off_y_base: usize,
    w00: f32,
    w10: f32,
    w01: f32,
    w11: f32,
) -> f32 {
    if scale_px <= 1
        && pl.len() >= (7 + off_y_base + 1) * 18 + off_x_base + 9
    {
        let p_ptr = pl.as_ptr();
        let c_ptr = cur_patch.as_ptr();
        let w00_v = f32x4_splat(w00);
        let w10_v = f32x4_splat(w10);
        let w01_v = f32x4_splat(w01);
        let w11_v = f32x4_splat(w11);
        let mut pv = [f32x4_splat(0.0); 16];
        let mut sum_v = f32x4_splat(0.0);
        for r in 0..8usize {
            let row0 = (r + off_y_base) * 18 + off_x_base;
            let row1 = row0 + 18;
            for half in 0..2usize {
                let off = half * 4;
                let v00 = v128_load(p_ptr.add(row0 + off) as *const v128);
                let v10 = v128_load(p_ptr.add(row0 + off + 1) as *const v128);
                let v01 = v128_load(p_ptr.add(row1 + off) as *const v128);
                let v11 = v128_load(p_ptr.add(row1 + off + 1) as *const v128);
                let b = f32x4_add(
                    f32x4_add(f32x4_mul(w00_v, v00), f32x4_mul(w10_v, v10)),
                    f32x4_add(f32x4_mul(w01_v, v01), f32x4_mul(w11_v, v11)),
                );
                pv[r * 2 + half] = b;
                sum_v = f32x4_add(sum_v, b);
            }
        }
        let mp = hsum_f32x4(sum_v) * (1.0 / 64.0);
        let mp_v = f32x4_splat(mp);
        let mut mad_v = f32x4_splat(0.0);
        for i in 0..16usize {
            let d = f32x4_sub(pv[i], mp_v);
            pv[i] = d;
            mad_v = f32x4_add(mad_v, f32x4_abs(d));
        }
        let g_scale = (cur_mad / hsum_f32x4(mad_v).max(1e-3)).clamp(0.75, 1.33);
        let g_v = f32x4_splat(g_scale);
        let mut sad_v = f32x4_splat(0.0);
        for i in 0..16usize {
            let cur = v128_load(c_ptr.add(i * 4) as *const v128);
            let diff = f32x4_sub(cur, f32x4_mul(pv[i], g_v));
            sad_v = f32x4_add(sad_v, f32x4_abs(diff));
        }
        return hsum_f32x4(sad_v);
    }

    if scale_px == 2
        && pl.len() >= (16 + off_y_base) * 18 + off_x_base + 17
    {
        let p_ptr = pl.as_ptr();
        let c_ptr = cur_patch.as_ptr();
        let w00_v = f32x4_splat(w00);
        let w10_v = f32x4_splat(w10);
        let w01_v = f32x4_splat(w01);
        let w11_v = f32x4_splat(w11);
        let mut pv = [f32x4_splat(0.0); 16];
        let mut sum_v = f32x4_splat(0.0);
        for r in 0..8usize {
            let phase = r & 1;
            let y0_base = (2 * r + off_y_base) * 18 + off_x_base + phase;
            for half in 0..2usize {
                let xb = y0_base + half * 8;
                let r0_a = v128_load(p_ptr.add(xb) as *const v128);
                let r0_b = v128_load(p_ptr.add(xb + 4) as *const v128);
                let r1_a = v128_load(p_ptr.add(xb + 18) as *const v128);
                let r1_b = v128_load(p_ptr.add(xb + 22) as *const v128);
                let r2_a = v128_load(p_ptr.add(xb + 36) as *const v128);
                let r2_b = v128_load(p_ptr.add(xb + 40) as *const v128);

                let e00 = i32x4_shuffle::<0, 4, 1, 5>(r0_a, r0_b);
                let o00 = i32x4_shuffle::<2, 6, 3, 7>(r1_a, r1_b);
                let e01 = i32x4_shuffle::<0, 4, 1, 5>(r1_a, r1_b);
                let o01 = i32x4_shuffle::<2, 6, 3, 7>(r2_a, r2_b);

                let v00 = i32x4_shuffle::<0, 4, 1, 5>(e00, o00);
                let v10 = i32x4_shuffle::<2, 6, 3, 7>(e00, o00);
                let v01 = i32x4_shuffle::<0, 4, 1, 5>(e01, o01);
                let v11 = i32x4_shuffle::<2, 6, 3, 7>(e01, o01);

                let b = if w10 == 0.0 && w11 == 0.0 {
                    if w01 == 0.0 {
                        v00
                    } else {
                        f32x4_add(f32x4_mul(w00_v, v00), f32x4_mul(w01_v, v01))
                    }
                } else {
                    f32x4_add(
                        f32x4_add(f32x4_mul(w00_v, v00), f32x4_mul(w10_v, v10)),
                        f32x4_add(f32x4_mul(w01_v, v01), f32x4_mul(w11_v, v11)),
                    )
                };
                pv[r * 2 + half] = b;
                sum_v = f32x4_add(sum_v, b);
            }
        }
        let mp = hsum_f32x4(sum_v) * (1.0 / 64.0);
        let mp_v = f32x4_splat(mp);
        let mut mad_v = f32x4_splat(0.0);
        for i in 0..16usize {
            let d = f32x4_sub(pv[i], mp_v);
            pv[i] = d;
            mad_v = f32x4_add(mad_v, f32x4_abs(d));
        }
        let g_scale = (cur_mad / hsum_f32x4(mad_v).max(1e-3)).clamp(0.75, 1.33);
        let g_v = f32x4_splat(g_scale);
        let mut sad_v = f32x4_splat(0.0);
        for i in 0..16usize {
            let cur = v128_load(c_ptr.add(i * 4) as *const v128);
            let diff = f32x4_sub(cur, f32x4_mul(pv[i], g_v));
            sad_v = f32x4_add(sad_v, f32x4_abs(diff));
        }
        return hsum_f32x4(sad_v);
    }

    let mut prev_b = [0.0f32; 64];
    let mut q = 0usize;
    for r in 0..8usize {
        for c in 0..8usize {
            let ly = r * scale_px + (c & phase_mask) + off_y_base;
            let lx = c * scale_px + (r & phase_mask) + off_x_base;
            let r0 = ly * 18 + lx;
            let r1 = r0 + 18;
            prev_b[q] = w00 * pl[r0] + w10 * pl[r0 + 1] + w01 * pl[r1] + w11 * pl[r1 + 1];
            q += 1;
        }
    }
    sad64_demean_gain_norm_simd128(cur_patch, cur_mad, &prev_b)
}

pub unsafe fn eval_patch_raw_sad_rgba_simd128(
    cur_patch: &[f32; 64],
    cur_mad: f32,
    prev_rgba: &[u8],
    frame_w: usize,
    sx0: usize,
    sy0: usize,
    scale_px: usize,
    phase_mask: usize,
) -> f32 {
    let w_luma = i16x8(9798, 19235, 3735, 0, 9798, 19235, 3735, 0);
    let scale_luma = f32x4_splat(1.0 / (255.0 * 32768.0));

    #[inline(always)]
    unsafe fn rgba4_luma_dot(pk: v128, w_luma: v128, scale_luma: v128) -> v128 {
        let lo = i32x4_dot_i16x8(u16x8_extend_low_u8x16(pk), w_luma);
        let hi = i32x4_dot_i16x8(u16x8_extend_high_u8x16(pk), w_luma);
        let sum = i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(lo, hi),
            i32x4_shuffle::<1, 3, 5, 7>(lo, hi),
        );
        f32x4_mul(f32x4_convert_i32x4(sum), scale_luma)
    }

    if scale_px == 1 && phase_mask == 0 {
        let base_ptr = prev_rgba.as_ptr();
        let stride4 = frame_w * 4;
        let mut pv = [f32x4_splat(0.0); 16];
        let mut sum_v = f32x4_splat(0.0);
        for r in 0..8usize {
            let p = base_ptr.add((sy0 + r) * stride4 + sx0 * 4);
            let l0 = rgba4_luma_dot(v128_load(p as *const v128), w_luma, scale_luma);
            let l1 = rgba4_luma_dot(v128_load(p.add(16) as *const v128), w_luma, scale_luma);
            pv[r * 2] = l0;
            pv[r * 2 + 1] = l1;
            sum_v = f32x4_add(sum_v, f32x4_add(l0, l1));
        }
        let mp_v = f32x4_splat(hsum_f32x4(sum_v) * (1.0 / 64.0));
        let mut mad_v = f32x4_splat(0.0);
        for i in 0..16usize {
            let d = f32x4_sub(pv[i], mp_v);
            pv[i] = d;
            mad_v = f32x4_add(mad_v, f32x4_abs(d));
        }
        let g_v = f32x4_splat((cur_mad / hsum_f32x4(mad_v).max(1e-3)).clamp(0.75, 1.33));
        let c_ptr = cur_patch.as_ptr();
        let mut sad_v = f32x4_splat(0.0);
        for i in 0..16usize {
            let cur = v128_load(c_ptr.add(i * 4) as *const v128);
            sad_v = f32x4_add(sad_v, f32x4_abs(f32x4_sub(cur, f32x4_mul(pv[i], g_v))));
        }
        return hsum_f32x4(sad_v);
    }

    if scale_px == 2
        && phase_mask == 1
        && sx0 + 17 <= frame_w
        && ((sy0 + 15) * frame_w + sx0 + 17) * 4 <= prev_rgba.len()
    {
        let base_ptr = prev_rgba.as_ptr();
        let stride4 = frame_w * 4;
        let mut pv = [f32x4_splat(0.0); 16];
        let mut sum0 = f32x4_splat(0.0);
        let mut sum1 = f32x4_splat(0.0);
        for r in 0..8usize {
            let p0 = base_ptr.add(((sy0 + 2 * r) * frame_w + sx0 + (r & 1)) * 4);
            let p1 = p0.add(stride4);

            let ev0 = i32x4_shuffle::<0, 4, 0, 4>(
                v128_load(p0 as *const v128),
                v128_load(p0.add(16) as *const v128),
            );
            let od0 = i32x4_shuffle::<2, 6, 2, 6>(
                v128_load(p1 as *const v128),
                v128_load(p1.add(16) as *const v128),
            );
            let px0 = i32x4_shuffle::<0, 4, 1, 5>(ev0, od0);
            let l0 = rgba4_luma_dot(px0, w_luma, scale_luma);

            let ev1 = i32x4_shuffle::<0, 4, 0, 4>(
                v128_load(p0.add(32) as *const v128),
                v128_load(p0.add(48) as *const v128),
            );
            let od1 = i32x4_shuffle::<2, 6, 2, 6>(
                v128_load(p1.add(32) as *const v128),
                v128_load(p1.add(48) as *const v128),
            );
            let px1 = i32x4_shuffle::<0, 4, 1, 5>(ev1, od1);
            let l1 = rgba4_luma_dot(px1, w_luma, scale_luma);

            pv[r * 2] = l0;
            pv[r * 2 + 1] = l1;
            sum0 = f32x4_add(sum0, l0);
            sum1 = f32x4_add(sum1, l1);
        }
        let mp_v = f32x4_splat(hsum_f32x4(f32x4_add(sum0, sum1)) * (1.0 / 64.0));
        let mut mad0 = f32x4_splat(0.0);
        let mut mad1 = f32x4_splat(0.0);
        for i in (0..16usize).step_by(2) {
            let d0 = f32x4_sub(pv[i], mp_v);
            let d1 = f32x4_sub(pv[i + 1], mp_v);
            pv[i] = d0;
            pv[i + 1] = d1;
            mad0 = f32x4_add(mad0, f32x4_abs(d0));
            mad1 = f32x4_add(mad1, f32x4_abs(d1));
        }
        let g_v = f32x4_splat((cur_mad / hsum_f32x4(f32x4_add(mad0, mad1)).max(1e-3)).clamp(0.75, 1.33));
        let c_ptr = cur_patch.as_ptr();
        let mut sad0 = f32x4_splat(0.0);
        let mut sad1 = f32x4_splat(0.0);
        for i in (0..16usize).step_by(2) {
            let c0 = v128_load(c_ptr.add(i * 4) as *const v128);
            let c1 = v128_load(c_ptr.add((i + 1) * 4) as *const v128);
            sad0 = f32x4_add(sad0, f32x4_abs(f32x4_sub(c0, f32x4_mul(pv[i], g_v))));
            sad1 = f32x4_add(sad1, f32x4_abs(f32x4_sub(c1, f32x4_mul(pv[i + 1], g_v))));
        }
        return hsum_f32x4(f32x4_add(sad0, sad1));
    }

    let mut prev_vals = [0.0f32; 64];
    let mut q = 0usize;
    for r in 0..8usize {
        for c in 0..8usize {
            let ry = r * scale_px + (c & phase_mask);
            let rx = c * scale_px + (r & phase_mask);
            let off4 = ((sy0 + ry) * frame_w + (sx0 + rx)) * 4;
            let pr = prev_rgba[off4] as f32;
            let pg = prev_rgba[off4 + 1] as f32;
            let pb = prev_rgba[off4 + 2] as f32;
            prev_vals[q] = (0.299 * pr + 0.587 * pg + 0.114 * pb) * (1.0 / 255.0);
            q += 1;
        }
    }
    sad64_demean_gain_norm_simd128(cur_patch, cur_mad, &prev_vals)
}

#[inline]
pub unsafe fn patch_grad_tensor_span_simd128(
    luma: &[f32],
    frame_w: usize,
    cx: usize,
    cy: usize,
    span: usize,
) -> (f32, f32, f32, f32) {
    let mut sxx_v = f32x4_splat(0.0);
    let mut syy_v = f32x4_splat(0.0);
    let mut sxy_v = f32x4_splat(0.0);
    let mut l1_v = f32x4_splat(0.0);
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    let mut sxy = 0.0f32;
    let mut l1 = 0.0f32;

    if span > 0
        && frame_w >= cx + span + 1
        && luma.len() >= (cy + span) * frame_w + cx + span + 1
    {
        let l_ptr = luma.as_ptr();
        for r in 0..span {
            let roff = (cy + r) * frame_w + cx;
            let mut c = 0usize;
            while c + 4 <= span {
                let p = l_ptr.add(roff + c);
                let v = v128_load(p as *const v128);
                let vx = v128_load(p.add(1) as *const v128);
                let vy = v128_load(p.add(frame_w) as *const v128);
                let dx = f32x4_sub(vx, v);
                let dy = f32x4_sub(vy, v);
                sxx_v = f32x4_add(sxx_v, f32x4_mul(dx, dx));
                syy_v = f32x4_add(syy_v, f32x4_mul(dy, dy));
                sxy_v = f32x4_add(sxy_v, f32x4_mul(dx, dy));
                l1_v = f32x4_add(l1_v, f32x4_add(f32x4_abs(dx), f32x4_abs(dy)));
                c += 4;
            }
            while c < span {
                let v = *l_ptr.add(roff + c);
                let dx = *l_ptr.add(roff + c + 1) - v;
                let dy = *l_ptr.add(roff + frame_w + c) - v;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
                l1 += dx.abs() + dy.abs();
                c += 1;
            }
        }
        return (
            sxx + hsum_f32x4(sxx_v),
            syy + hsum_f32x4(syy_v),
            sxy + hsum_f32x4(sxy_v),
            l1 + hsum_f32x4(l1_v),
        );
    }

    for r in 0..span {
        let roff = (cy + r) * frame_w + cx;
        for c in 0..span {
            let v = luma[roff + c];
            let dx = luma[roff + c + 1] - v;
            let dy = luma[roff + frame_w + c] - v;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
            l1 += dx.abs() + dy.abs();
        }
    }
    (sxx, syy, sxy, l1)
}

#[inline]
pub unsafe fn patch_grad_tensor_span15_simd128(
    luma: &[f32],
    frame_w: usize,
    cx: usize,
    cy: usize,
) -> (f32, f32, f32, f32) {
    if frame_w < cx + 16 || luma.len() < (cy + 15) * frame_w + cx + 16 {
        return patch_grad_tensor_span_simd128(luma, frame_w, cx, cy, 15);
    }
    let l_ptr = luma.as_ptr();
    let mask_1110 = i32x4(-1, -1, -1, 0);
    let mut sxx_v = f32x4_splat(0.0);
    let mut syy_v = f32x4_splat(0.0);
    let mut sxy_v = f32x4_splat(0.0);
    let mut l1_v = f32x4_splat(0.0);
    let mut p = l_ptr.add(cy * frame_w + cx);
    let mut c0 = v128_load(p as *const v128);
    let mut c1 = v128_load(p.add(4) as *const v128);
    let mut c2 = v128_load(p.add(8) as *const v128);
    let mut c3 = v128_load(p.add(12) as *const v128);
    for _r in 0..15usize {
        p = p.add(frame_w);
        let n0 = v128_load(p as *const v128);
        let n1 = v128_load(p.add(4) as *const v128);
        let n2 = v128_load(p.add(8) as *const v128);
        let n3 = v128_load(p.add(12) as *const v128);

        let vx0 = i32x4_shuffle::<1, 2, 3, 4>(c0, c1);
        let vx1 = i32x4_shuffle::<1, 2, 3, 4>(c1, c2);
        let vx2 = i32x4_shuffle::<1, 2, 3, 4>(c2, c3);
        let vx3 = i32x4_shuffle::<1, 2, 3, 3>(c3, c3);

        macro_rules! acc {
            ($dx:expr, $dy:expr) => {{
                sxx_v = f32x4_add(sxx_v, f32x4_mul($dx, $dx));
                syy_v = f32x4_add(syy_v, f32x4_mul($dy, $dy));
                sxy_v = f32x4_add(sxy_v, f32x4_mul($dx, $dy));
                l1_v = f32x4_add(l1_v, f32x4_add(f32x4_abs($dx), f32x4_abs($dy)));
            }};
        }
        acc!(f32x4_sub(vx0, c0), f32x4_sub(n0, c0));
        acc!(f32x4_sub(vx1, c1), f32x4_sub(n1, c1));
        acc!(f32x4_sub(vx2, c2), f32x4_sub(n2, c2));
        let dx3 = v128_and(f32x4_sub(vx3, c3), mask_1110);
        let dy3 = v128_and(f32x4_sub(n3, c3), mask_1110);
        acc!(dx3, dy3);

        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    (
        hsum_f32x4(sxx_v),
        hsum_f32x4(syy_v),
        hsum_f32x4(sxy_v),
        hsum_f32x4(l1_v),
    )
}

#[inline(never)]
pub unsafe fn pick_textured_subpatch_5band_hd_simd128(
    luma: &[f32],
    frame_w: usize,
    base_x: usize,
    base_y: usize,
) -> (usize, usize, f32) {
    let l_ptr = luma.as_ptr();
    let mask_1110 = i32x4(-1, -1, -1, 0);
    let zero_v = f32x4_splat(0.0);

    // 5 disjoint Y bands covering oy=0 [0..15), oy=1 [4..19), oy=2 [8..23)
    const Y_BANDS: [(usize, usize); 5] = [(0, 4), (4, 8), (8, 15), (15, 19), (19, 23)];
    let mut band_stats = [[zero_v; 7]; 5];

    // Left half: strips 0..4 ([0..4), [4..8), [8..12), [12..15)) in 1 vertical sweep.
    {
        let mut p = l_ptr.add(base_y * frame_w + base_x);
        let mut c0 = v128_load(p as *const v128);
        let mut c1 = v128_load(p.add(4) as *const v128);
        let mut c2 = v128_load(p.add(8) as *const v128);
        let mut c3 = v128_load(p.add(12) as *const v128);
        let mut c4 = v128_load(p.add(16) as *const v128);
        for yb in 0..5usize {
            let (r0, r1) = Y_BANDS[yb];
            let mut sxx0 = zero_v;
            let mut sxx1 = zero_v;
            let mut sxx2 = zero_v;
            let mut sxx3 = zero_v;
            let mut syy0 = zero_v;
            let mut syy1 = zero_v;
            let mut syy2 = zero_v;
            let mut syy3 = zero_v;
            let mut sxy0 = zero_v;
            let mut sxy1 = zero_v;
            let mut sxy2 = zero_v;
            let mut sxy3 = zero_v;
            let mut l1_0 = zero_v;
            let mut l1_1 = zero_v;
            let mut l1_2 = zero_v;
            let mut l1_3 = zero_v;
            for _r in r0..r1 {
                p = p.add(frame_w);
                let n0 = v128_load(p as *const v128);
                let n1 = v128_load(p.add(4) as *const v128);
                let n2 = v128_load(p.add(8) as *const v128);
                let n3 = v128_load(p.add(12) as *const v128);
                let n4 = v128_load(p.add(16) as *const v128);

                let dx0 = f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c0, c1), c0);
                let dx1 = f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c1, c2), c1);
                let dx2 = f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c2, c3), c2);
                let dx3 = v128_and(f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c3, c4), c3), mask_1110);

                let dy0 = f32x4_sub(n0, c0);
                let dy1 = f32x4_sub(n1, c1);
                let dy2 = f32x4_sub(n2, c2);
                let dy3 = v128_and(f32x4_sub(n3, c3), mask_1110);

                sxx0 = f32x4_add(sxx0, f32x4_mul(dx0, dx0));
                sxx1 = f32x4_add(sxx1, f32x4_mul(dx1, dx1));
                sxx2 = f32x4_add(sxx2, f32x4_mul(dx2, dx2));
                sxx3 = f32x4_add(sxx3, f32x4_mul(dx3, dx3));

                syy0 = f32x4_add(syy0, f32x4_mul(dy0, dy0));
                syy1 = f32x4_add(syy1, f32x4_mul(dy1, dy1));
                syy2 = f32x4_add(syy2, f32x4_mul(dy2, dy2));
                syy3 = f32x4_add(syy3, f32x4_mul(dy3, dy3));

                sxy0 = f32x4_add(sxy0, f32x4_mul(dx0, dy0));
                sxy1 = f32x4_add(sxy1, f32x4_mul(dx1, dy1));
                sxy2 = f32x4_add(sxy2, f32x4_mul(dx2, dy2));
                sxy3 = f32x4_add(sxy3, f32x4_mul(dx3, dy3));

                l1_0 = f32x4_add(l1_0, f32x4_add(f32x4_abs(dx0), f32x4_abs(dy0)));
                l1_1 = f32x4_add(l1_1, f32x4_add(f32x4_abs(dx1), f32x4_abs(dy1)));
                l1_2 = f32x4_add(l1_2, f32x4_add(f32x4_abs(dx2), f32x4_abs(dy2)));
                l1_3 = f32x4_add(l1_3, f32x4_add(f32x4_abs(dx3), f32x4_abs(dy3)));

                c0 = n0;
                c1 = n1;
                c2 = n2;
                c3 = n3;
                c4 = n4;
            }
            band_stats[yb][0] = hsum4_f32x4(sxx0, syy0, sxy0, l1_0);
            band_stats[yb][1] = hsum4_f32x4(sxx1, syy1, sxy1, l1_1);
            band_stats[yb][2] = hsum4_f32x4(sxx2, syy2, sxy2, l1_2);
            band_stats[yb][3] = hsum4_f32x4(sxx3, syy3, sxy3, l1_3);
        }
    }

    // Right half: strips 4..7 ([15..19), [19..23), [23..27)) in 1 vertical sweep.
    {
        let mut p = l_ptr.add(base_y * frame_w + base_x + 12);
        let mut c3 = v128_load(p as *const v128);
        let mut c4 = v128_load(p.add(4) as *const v128);
        let mut c5 = v128_load(p.add(8) as *const v128);
        let mut c6 = v128_load(p.add(12) as *const v128);
        for yb in 0..5usize {
            let (r0, r1) = Y_BANDS[yb];
            let mut sxx4 = zero_v;
            let mut sxx5 = zero_v;
            let mut sxx6 = zero_v;
            let mut syy4 = zero_v;
            let mut syy5 = zero_v;
            let mut syy6 = zero_v;
            let mut sxy4 = zero_v;
            let mut sxy5 = zero_v;
            let mut sxy6 = zero_v;
            let mut l1_4 = zero_v;
            let mut l1_5 = zero_v;
            let mut l1_6 = zero_v;
            for _r in r0..r1 {
                p = p.add(frame_w);
                let n3 = v128_load(p as *const v128);
                let n4 = v128_load(p.add(4) as *const v128);
                let n5 = v128_load(p.add(8) as *const v128);
                let n6 = v128_load(p.add(12) as *const v128);

                let dx4 = f32x4_sub(c4, i32x4_shuffle::<3, 4, 5, 6>(c3, c4));
                let dx5 = f32x4_sub(c5, i32x4_shuffle::<3, 4, 5, 6>(c4, c5));
                let dx6 = f32x4_sub(c6, i32x4_shuffle::<3, 4, 5, 6>(c5, c6));

                let d_a3 = f32x4_sub(n3, c3);
                let d_a4 = f32x4_sub(n4, c4);
                let d_a5 = f32x4_sub(n5, c5);
                let d_a6 = f32x4_sub(n6, c6);

                let dy4 = i32x4_shuffle::<3, 4, 5, 6>(d_a3, d_a4);
                let dy5 = i32x4_shuffle::<3, 4, 5, 6>(d_a4, d_a5);
                let dy6 = i32x4_shuffle::<3, 4, 5, 6>(d_a5, d_a6);

                sxx4 = f32x4_add(sxx4, f32x4_mul(dx4, dx4));
                sxx5 = f32x4_add(sxx5, f32x4_mul(dx5, dx5));
                sxx6 = f32x4_add(sxx6, f32x4_mul(dx6, dx6));

                syy4 = f32x4_add(syy4, f32x4_mul(dy4, dy4));
                syy5 = f32x4_add(syy5, f32x4_mul(dy5, dy5));
                syy6 = f32x4_add(syy6, f32x4_mul(dy6, dy6));

                sxy4 = f32x4_add(sxy4, f32x4_mul(dx4, dy4));
                sxy5 = f32x4_add(sxy5, f32x4_mul(dx5, dy5));
                sxy6 = f32x4_add(sxy6, f32x4_mul(dx6, dy6));

                l1_4 = f32x4_add(l1_4, f32x4_add(f32x4_abs(dx4), f32x4_abs(dy4)));
                l1_5 = f32x4_add(l1_5, f32x4_add(f32x4_abs(dx5), f32x4_abs(dy5)));
                l1_6 = f32x4_add(l1_6, f32x4_add(f32x4_abs(dx6), f32x4_abs(dy6)));

                c3 = n3;
                c4 = n4;
                c5 = n5;
                c6 = n6;
            }
            band_stats[yb][4] = hsum4_f32x4(sxx4, syy4, sxy4, l1_4);
            band_stats[yb][5] = hsum4_f32x4(sxx5, syy5, sxy5, l1_5);
            band_stats[yb][6] = hsum4_f32x4(sxx6, syy6, sxy6, l1_6);
        }
    }

    // Prefix combine into 3 vertical sums per X strip: row_stats[oy][xb]
    let mut row_stats = [[zero_v; 7]; 3];
    for xb in 0..7usize {
        let shared_mid = f32x4_add(band_stats[1][xb], band_stats[2][xb]);
        row_stats[0][xb] = f32x4_add(band_stats[0][xb], shared_mid);
        row_stats[1][xb] = f32x4_add(shared_mid, band_stats[3][xb]);
        row_stats[2][xb] = f32x4_add(
            f32x4_add(band_stats[2][xb], band_stats[3][xb]),
            band_stats[4][xb],
        );
    }

    let mut best_tex2d = 0.0f32;
    let mut best_bx = base_x;
    let mut best_by = base_y;
    for oy in 0..3usize {
        let cy = base_y + oy * 4;
        for ox in 0..4usize {
            let v = f32x4_add(
                f32x4_add(f32x4_add(row_stats[oy][ox], row_stats[oy][ox + 1]), row_stats[oy][ox + 2]),
                row_stats[oy][ox + 3],
            );
            let l1 = f32x4_extract_lane::<3>(v);
            let upper_t2d = 0.5 * l1;
            if upper_t2d <= best_tex2d {
                continue;
            }
            let sxx = f32x4_extract_lane::<0>(v);
            let syy = f32x4_extract_lane::<1>(v);
            let sxy = f32x4_extract_lane::<2>(v);
            let tr = sxx + syy;
            let det = (sxx * syy - sxy * sxy).max(0.0);
            if det < 0.09566326 * tr * tr {
                continue;
            }
            if best_tex2d > 0.0 {
                let need = (best_tex2d * 0.999) / upper_t2d;
                let r_req = need * need;
                if det * (1.0 + r_req) * (1.0 + r_req) < r_req * tr * tr {
                    continue;
                }
            }
            let disc = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat((tr * tr - 4.0 * det).max(0.0))));
            let lam_min = 0.5 * (tr - disc);
            let lam_max = 0.5 * (tr + disc);
            let t2d = if lam_min >= 0.12 * lam_max {
                upper_t2d * f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(lam_min / lam_max.max(1e-8))))
            } else {
                0.0
            };
            if t2d > best_tex2d {
                best_tex2d = t2d;
                best_bx = base_x + ox * 4;
                best_by = cy;
            }
        }
    }
    (best_bx, best_by, best_tex2d)
}

pub unsafe fn pick_textured_subpatch_step16_hd_simd128(
    luma: &[f32],
    frame_w: usize,
    base_x: usize,
    base_y: usize,
) -> (usize, usize, f32) {
    let l_ptr = luma.as_ptr();
    let mask_1110 = i32x4(-1, -1, -1, 0);
    let zero_v = f32x4_splat(0.0);
    let mut best_tex2d = 0.0f32;
    let mut best_bx = base_x;
    let mut best_by = base_y;

    macro_rules! eval_cand {
        ($stats:expr, $cx:expr, $cy:expr) => {{
            let v = $stats;
            let sxx = f32x4_extract_lane::<0>(v);
            let syy = f32x4_extract_lane::<1>(v);
            let sxy = f32x4_extract_lane::<2>(v);
            let l1 = f32x4_extract_lane::<3>(v);
            let upper_t2d = 0.5 * l1;
            if upper_t2d > best_tex2d {
                let tr = sxx + syy;
                let det = (sxx * syy - sxy * sxy).max(0.0);
                if det >= 0.09566326 * tr * tr {
                    let ok = if best_tex2d > 0.0 {
                        let need = (best_tex2d * 0.999) / upper_t2d;
                        let r_req = need * need;
                        det * (1.0 + r_req) * (1.0 + r_req) >= r_req * tr * tr
                    } else {
                        true
                    };
                    if ok {
                        let disc = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(
                            (tr * tr - 4.0 * det).max(0.0),
                        )));
                        let lam_min = 0.5 * (tr - disc);
                        let lam_max = 0.5 * (tr + disc);
                        let t2d = if lam_min >= 0.12 * lam_max {
                            upper_t2d
                                * f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(
                                    lam_min / lam_max.max(1e-8),
                                )))
                        } else {
                            0.0
                        };
                        if t2d > best_tex2d {
                            best_tex2d = t2d;
                            best_bx = $cx;
                            best_by = $cy;
                        }
                    }
                }
            }
        }};
    }

    for oy in 0..3usize {
        let cy = base_y + oy * 16;
        for half in 0..2usize {
            let hx = base_x + half * 32;
            let mut sxx0 = zero_v;
            let mut syy0 = zero_v;
            let mut sxy0 = zero_v;
            let mut l1_0 = zero_v;
            let mut sxx1 = zero_v;
            let mut syy1 = zero_v;
            let mut sxy1 = zero_v;
            let mut l1_1 = zero_v;

            let mut p = l_ptr.add(cy * frame_w + hx);
            let mut c0 = v128_load(p as *const v128);
            let mut c1 = v128_load(p.add(4) as *const v128);
            let mut c2 = v128_load(p.add(8) as *const v128);
            let mut c3 = v128_load(p.add(12) as *const v128);
            let mut c4 = v128_load(p.add(16) as *const v128);
            let mut c5 = v128_load(p.add(20) as *const v128);
            let mut c6 = v128_load(p.add(24) as *const v128);
            let mut c7 = v128_load(p.add(28) as *const v128);

            for _r in 0..15usize {
                p = p.add(frame_w);
                {
                    let n0 = v128_load(p as *const v128);
                    let n1 = v128_load(p.add(4) as *const v128);
                    let n2 = v128_load(p.add(8) as *const v128);
                    let n3 = v128_load(p.add(12) as *const v128);
                    macro_rules! acc0 {
                        ($dx:expr, $dy:expr) => {{
                            let dx = $dx;
                            let dy = $dy;
                            sxx0 = f32x4_add(sxx0, f32x4_mul(dx, dx));
                            syy0 = f32x4_add(syy0, f32x4_mul(dy, dy));
                            sxy0 = f32x4_add(sxy0, f32x4_mul(dx, dy));
                            l1_0 = f32x4_add(l1_0, f32x4_add(f32x4_abs(dx), f32x4_abs(dy)));
                        }};
                    }
                    acc0!(f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c0, c1), c0), f32x4_sub(n0, c0));
                    acc0!(f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c1, c2), c1), f32x4_sub(n1, c1));
                    acc0!(f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c2, c3), c2), f32x4_sub(n2, c2));
                    acc0!(
                        v128_and(f32x4_sub(i32x4_shuffle::<1, 2, 3, 3>(c3, c3), c3), mask_1110),
                        v128_and(f32x4_sub(n3, c3), mask_1110)
                    );
                    c0 = n0;
                    c1 = n1;
                    c2 = n2;
                    c3 = n3;
                }
                {
                    let n4 = v128_load(p.add(16) as *const v128);
                    let n5 = v128_load(p.add(20) as *const v128);
                    let n6 = v128_load(p.add(24) as *const v128);
                    let n7 = v128_load(p.add(28) as *const v128);
                    macro_rules! acc1 {
                        ($dx:expr, $dy:expr) => {{
                            let dx = $dx;
                            let dy = $dy;
                            sxx1 = f32x4_add(sxx1, f32x4_mul(dx, dx));
                            syy1 = f32x4_add(syy1, f32x4_mul(dy, dy));
                            sxy1 = f32x4_add(sxy1, f32x4_mul(dx, dy));
                            l1_1 = f32x4_add(l1_1, f32x4_add(f32x4_abs(dx), f32x4_abs(dy)));
                        }};
                    }
                    acc1!(f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c4, c5), c4), f32x4_sub(n4, c4));
                    acc1!(f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c5, c6), c5), f32x4_sub(n5, c5));
                    acc1!(f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c6, c7), c6), f32x4_sub(n6, c6));
                    acc1!(
                        v128_and(f32x4_sub(i32x4_shuffle::<1, 2, 3, 3>(c7, c7), c7), mask_1110),
                        v128_and(f32x4_sub(n7, c7), mask_1110)
                    );
                    c4 = n4;
                    c5 = n5;
                    c6 = n6;
                    c7 = n7;
                }
            }
            eval_cand!(hsum4_f32x4(sxx0, syy0, sxy0, l1_0), hx, cy);
            eval_cand!(hsum4_f32x4(sxx1, syy1, sxy1, l1_1), hx + 16, cy);
        }
    }

    (best_bx, best_by, best_tex2d)
}

#[inline]
pub unsafe fn cam_pyr_tile_4x6x8_simd128(
    pyr_luma: &[f32],
    prev_pyr_luma: &[f32],
    pyr_w: usize,
    tx: usize,
    ty: usize,
    cur_sub_out: &mut [f32; 48],
) -> (f32, f32, f32, f32, f32, f32, bool) {
    let q25 = f32x4_splat(0.25);
    let zero = f32x4_splat(0.0);
    let mask_1110 = i32x4(0, -1, -1, -1);
    let p_cur = pyr_luma.as_ptr();
    let p_prv = prev_pyr_luma.as_ptr();

    let mut cur_v = [zero; 12];
    let mut prv_v = [zero; 12];
    let mut sum_c_v = zero;
    let mut sum_p_v = zero;

    for r in 0..6usize {
        let row0 = (ty + r * 2) * pyr_w + tx;
        let row1 = row0 + pyr_w;

        let c0a = v128_load(p_cur.add(row0) as *const v128);
        let c0b = v128_load(p_cur.add(row0 + 4) as *const v128);
        let c1a = v128_load(p_cur.add(row1) as *const v128);
        let c1b = v128_load(p_cur.add(row1 + 4) as *const v128);
        let cs_a = f32x4_add(c0a, c1a);
        let cs_b = f32x4_add(c0b, c1b);
        let c_lo = f32x4_mul(
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(cs_a, cs_b),
                i32x4_shuffle::<1, 3, 5, 7>(cs_a, cs_b),
            ),
            q25,
        );

        let c0c = v128_load(p_cur.add(row0 + 8) as *const v128);
        let c0d = v128_load(p_cur.add(row0 + 12) as *const v128);
        let c1c = v128_load(p_cur.add(row1 + 8) as *const v128);
        let c1d = v128_load(p_cur.add(row1 + 12) as *const v128);
        let cs_c = f32x4_add(c0c, c1c);
        let cs_d = f32x4_add(c0d, c1d);
        let c_hi = f32x4_mul(
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(cs_c, cs_d),
                i32x4_shuffle::<1, 3, 5, 7>(cs_c, cs_d),
            ),
            q25,
        );

        let p0a = v128_load(p_prv.add(row0) as *const v128);
        let p0b = v128_load(p_prv.add(row0 + 4) as *const v128);
        let p1a = v128_load(p_prv.add(row1) as *const v128);
        let p1b = v128_load(p_prv.add(row1 + 4) as *const v128);
        let ps_a = f32x4_add(p0a, p1a);
        let ps_b = f32x4_add(p0b, p1b);
        let p_lo = f32x4_mul(
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(ps_a, ps_b),
                i32x4_shuffle::<1, 3, 5, 7>(ps_a, ps_b),
            ),
            q25,
        );

        let p0c = v128_load(p_prv.add(row0 + 8) as *const v128);
        let p0d = v128_load(p_prv.add(row0 + 12) as *const v128);
        let p1c = v128_load(p_prv.add(row1 + 8) as *const v128);
        let p1d = v128_load(p_prv.add(row1 + 12) as *const v128);
        let ps_c = f32x4_add(p0c, p1c);
        let ps_d = f32x4_add(p0d, p1d);
        let p_hi = f32x4_mul(
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(ps_c, ps_d),
                i32x4_shuffle::<1, 3, 5, 7>(ps_c, ps_d),
            ),
            q25,
        );

        cur_v[r * 2] = c_lo;
        cur_v[r * 2 + 1] = c_hi;
        prv_v[r * 2] = p_lo;
        prv_v[r * 2 + 1] = p_hi;
        sum_c_v = f32x4_add(sum_c_v, f32x4_add(c_lo, c_hi));
        sum_p_v = f32x4_add(sum_p_v, f32x4_add(p_lo, p_hi));
    }

    let mc = hsum_f32x4(sum_c_v) * (1.0 / 48.0);
    let mp = hsum_f32x4(sum_p_v) * (1.0 / 48.0);
    let mc_v = f32x4_splat(mc);
    let mp_v = f32x4_splat(mp);

    let mut tex_mad_v = zero;
    let mut prev_mad_v = zero;
    let mut sxx_v = zero;
    let mut syy_v = zero;
    let mut sxy_v = zero;
    let mut gl1_v = zero;

    let mut prev_dc_lo = zero;
    let mut prev_dc_hi = zero;
    for r in 0..6usize {
        let dc_lo = f32x4_sub(cur_v[r * 2], mc_v);
        let dc_hi = f32x4_sub(cur_v[r * 2 + 1], mc_v);
        let dp_lo = f32x4_sub(prv_v[r * 2], mp_v);
        let dp_hi = f32x4_sub(prv_v[r * 2 + 1], mp_v);
        cur_v[r * 2] = dc_lo;
        cur_v[r * 2 + 1] = dc_hi;
        prv_v[r * 2] = dp_lo;
        prv_v[r * 2 + 1] = dp_hi;
        tex_mad_v = f32x4_add(tex_mad_v, f32x4_add(f32x4_abs(dc_lo), f32x4_abs(dc_hi)));
        prev_mad_v = f32x4_add(prev_mad_v, f32x4_add(f32x4_abs(dp_lo), f32x4_abs(dp_hi)));
        if r > 0 {
            let left_lo = i32x4_shuffle::<3, 0, 1, 2>(dc_lo, dc_lo);
            let dx_lo = v128_and(f32x4_sub(dc_lo, left_lo), mask_1110);
            let dy_lo = v128_and(f32x4_sub(dc_lo, prev_dc_lo), mask_1110);

            let left_hi = i32x4_shuffle::<3, 4, 5, 6>(dc_lo, dc_hi);
            let dx_hi = f32x4_sub(dc_hi, left_hi);
            let dy_hi = f32x4_sub(dc_hi, prev_dc_hi);

            sxx_v = f32x4_add(sxx_v, f32x4_add(f32x4_mul(dx_lo, dx_lo), f32x4_mul(dx_hi, dx_hi)));
            syy_v = f32x4_add(syy_v, f32x4_add(f32x4_mul(dy_lo, dy_lo), f32x4_mul(dy_hi, dy_hi)));
            sxy_v = f32x4_add(sxy_v, f32x4_add(f32x4_mul(dx_lo, dy_lo), f32x4_mul(dx_hi, dy_hi)));
            gl1_v = f32x4_add(
                gl1_v,
                f32x4_add(
                    f32x4_add(f32x4_abs(dx_lo), f32x4_abs(dy_lo)),
                    f32x4_add(f32x4_abs(dx_hi), f32x4_abs(dy_hi)),
                ),
            );
        }
        prev_dc_lo = dc_lo;
        prev_dc_hi = dc_hi;
    }

    let sxx = hsum_f32x4(sxx_v);
    let syy = hsum_f32x4(syy_v);
    let sxy = hsum_f32x4(sxy_v);
    let g_l1 = hsum_f32x4(gl1_v);
    let tr = sxx + syy;
    let det = (sxx * syy - sxy * sxy).max(0.0);
    let iso2d = det >= 0.055 * tr * tr;

    let tex_mad_raw = hsum_f32x4(tex_mad_v);
    let prev_mad_raw = hsum_f32x4(prev_mad_v);
    let g_v = f32x4_splat((tex_mad_raw / prev_mad_raw.max(1e-3)).clamp(0.75, 1.33));
    let mut zsad_v = zero;
    for k in 0..12usize {
        zsad_v = f32x4_add(
            zsad_v,
            f32x4_abs(f32x4_sub(cur_v[k], f32x4_mul(prv_v[k], g_v))),
        );
    }
    let zero_sad = hsum_f32x4(zsad_v) * (1.0 / 48.0);
    if zero_sad > 0.016 {
        let out_ptr = cur_sub_out.as_mut_ptr();
        for k in 0..12usize {
            v128_store(out_ptr.add(k * 4) as *mut v128, cur_v[k]);
        }
    }
    (
        zero_sad,
        mc,
        mp,
        tex_mad_raw * (1.0 / 48.0),
        prev_mad_raw * (1.0 / 48.0),
        g_l1,
        iso2d,
    )
}

#[inline]
pub unsafe fn unpack_rgba_luma_window_simd128(
    prev_rgba: &[u8],
    frame_w: usize,
    frame_h: usize,
    wx0: i32,
    wy0: i32,
    win_dim: usize,
    dst: &mut [f32],
) {
    if win_dim == 0 || frame_w == 0 || frame_h == 0 || dst.len() < win_dim * win_dim {
        return;
    }
    let w_max = (frame_w as i32) - 1;
    let h_max = (frame_h as i32) - 1;
    let w_i = frame_w as i32;
    let rgba_ptr = prev_rgba.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mask_ff = u32x4_splat(0xFF);
    let wr_255 = f32x4_splat(0.299 / 255.0);
    let wg_255 = f32x4_splat(0.587 / 255.0);
    let wb_255 = f32x4_splat(0.114 / 255.0);

    let in_bounds = wx0 >= 0
        && wy0 >= 0
        && (wx0 + (win_dim as i32)) <= w_i
        && (wy0 + (win_dim as i32)) <= (frame_h as i32)
        && (((wy0 as usize) + win_dim - 1) * frame_w + (wx0 as usize) + win_dim) * 4
            <= prev_rgba.len();
    if in_bounds {
        let ux0 = wx0 as usize;
        let uy0 = wy0 as usize;
        for wy in 0..win_dim {
            let roff = (uy0 + wy) * frame_w + ux0;
            let wrow = wy * win_dim;
            let mut wx = 0usize;
            while wx + 4 <= win_dim {
                let pk = v128_load(rgba_ptr.add((roff + wx) * 4) as *const v128);
                let r_f = f32x4_convert_u32x4(v128_and(pk, mask_ff));
                let g_f = f32x4_convert_u32x4(v128_and(u32x4_shr(pk, 8), mask_ff));
                let b_f = f32x4_convert_u32x4(v128_and(u32x4_shr(pk, 16), mask_ff));
                let l_v = f32x4_add(
                    f32x4_add(f32x4_mul(wr_255, r_f), f32x4_mul(wg_255, g_f)),
                    f32x4_mul(wb_255, b_f),
                );
                v128_store(dst_ptr.add(wrow + wx) as *mut v128, l_v);
                wx += 4;
            }
            while wx < win_dim {
                let off4 = (roff + wx) * 4;
                let pr = *rgba_ptr.add(off4) as f32;
                let pg = *rgba_ptr.add(off4 + 1) as f32;
                let pb = *rgba_ptr.add(off4 + 2) as f32;
                *dst_ptr.add(wrow + wx) = (0.299 * pr + 0.587 * pg + 0.114 * pb) * (1.0 / 255.0);
                wx += 1;
            }
        }
        return;
    }

    for wy in 0..win_dim {
        let py_c = (wy0 + (wy as i32)).clamp(0, h_max) as usize;
        let roff = py_c * frame_w;
        let wrow = wy * win_dim;
        let mut wx = 0usize;
        while wx + 4 <= win_dim {
            let gx = wx0 + (wx as i32);
            if gx >= 0
                && gx + 4 <= w_i
                && (roff + (gx as usize) + 4) * 4 <= prev_rgba.len()
            {
                let px_u = gx as usize;
                let pk = v128_load(rgba_ptr.add((roff + px_u) * 4) as *const v128);
                let r_f = f32x4_convert_u32x4(v128_and(pk, mask_ff));
                let g_f = f32x4_convert_u32x4(v128_and(u32x4_shr(pk, 8), mask_ff));
                let b_f = f32x4_convert_u32x4(v128_and(u32x4_shr(pk, 16), mask_ff));
                let l_v = f32x4_add(
                    f32x4_add(f32x4_mul(wr_255, r_f), f32x4_mul(wg_255, g_f)),
                    f32x4_mul(wb_255, b_f),
                );
                v128_store(dst_ptr.add(wrow + wx) as *mut v128, l_v);
                wx += 4;
            } else {
                let px_c = gx.clamp(0, w_max) as usize;
                let off4 = (roff + px_c) * 4;
                let pr = prev_rgba[off4] as f32;
                let pg = prev_rgba[off4 + 1] as f32;
                let pb = prev_rgba[off4 + 2] as f32;
                *dst_ptr.add(wrow + wx) = (0.299 * pr + 0.587 * pg + 0.114 * pb) * (1.0 / 255.0);
                wx += 1;
            }
        }
        while wx < win_dim {
            let px_c = (wx0 + (wx as i32)).clamp(0, w_max) as usize;
            let off4 = (roff + px_c) * 4;
            let pr = prev_rgba[off4] as f32;
            let pg = prev_rgba[off4 + 1] as f32;
            let pb = prev_rgba[off4 + 2] as f32;
            *dst_ptr.add(wrow + wx) = (0.299 * pr + 0.587 * pg + 0.114 * pb) * (1.0 / 255.0);
            wx += 1;
        }
    }
}

#[inline]
pub unsafe fn unpack_rgba_luma_win_p2_simd128(
    prev_rgba: &[u8],
    frame_w: usize,
    frame_h: usize,
    wx0: i32,
    wy0: i32,
    win_dim: usize,
    planes: &mut [f32],
) -> (usize, usize) {
    let p_h = (win_dim + 1) / 2;
    let p_w = ((win_dim + 3) / 4) + 1;
    let plane_sz = p_h * p_w;
    if win_dim == 0 || frame_w == 0 || frame_h == 0 || planes.len() < 8 * plane_sz {
        return (p_w, plane_sz);
    }
    let w_max = (frame_w as i32) - 1;
    let h_max = (frame_h as i32) - 1;
    let w_i = frame_w as i32;
    let rgba_ptr = prev_rgba.as_ptr();
    let dst_base = planes.as_mut_ptr();
    let w_luma = i16x8(9798, 19235, 3735, 0, 9798, 19235, 3735, 0);
    let scale_luma = f32x4_splat(1.0 / (255.0 * 32768.0));

    #[inline(always)]
    unsafe fn rgba4_luma_255(pk: v128, w_luma: v128, scale_luma: v128) -> v128 {
        let lo = i32x4_dot_i16x8(u16x8_extend_low_u8x16(pk), w_luma);
        let hi = i32x4_dot_i16x8(u16x8_extend_high_u8x16(pk), w_luma);
        let sum = i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(lo, hi),
            i32x4_shuffle::<1, 3, 5, 7>(lo, hi),
        );
        f32x4_mul(f32x4_convert_i32x4(sum), scale_luma)
    }

    let in_bounds = wx0 >= 0
        && wy0 >= 0
        && (wx0 + (win_dim as i32)) <= w_i
        && (wy0 + (win_dim as i32)) <= (frame_h as i32)
        && (((wy0 as usize) + win_dim - 1) * frame_w + (wx0 as usize) + win_dim) * 4
            <= prev_rgba.len();
    if in_bounds {
        let ux0 = wx0 as usize;
        let uy0 = wy0 as usize;
        for y in 0..win_dim {
            let row_px = (uy0 + y) * frame_w + ux0;
            let row_off = (y >> 1) * p_w;
            let ph_y = (y & 1) * 4;
            let d0 = dst_base.add(ph_y * plane_sz + row_off);
            let d1 = dst_base.add((ph_y + 1) * plane_sz + row_off);
            let d2 = dst_base.add((ph_y + 2) * plane_sz + row_off);
            let d3 = dst_base.add((ph_y + 3) * plane_sz + row_off);
            let mut x = 0usize;
            while x + 16 <= win_dim {
                let rp = rgba_ptr.add((row_px + x) * 4);
                let v0 = rgba4_luma_255(v128_load(rp as *const v128), w_luma, scale_luma);
                let v1 = rgba4_luma_255(v128_load(rp.add(16) as *const v128), w_luma, scale_luma);
                let v2 = rgba4_luma_255(v128_load(rp.add(32) as *const v128), w_luma, scale_luma);
                let v3 = rgba4_luma_255(v128_load(rp.add(48) as *const v128), w_luma, scale_luma);
                let t0 = i32x4_shuffle::<0, 4, 1, 5>(v0, v1);
                let t1 = i32x4_shuffle::<2, 6, 3, 7>(v0, v1);
                let t2 = i32x4_shuffle::<0, 4, 1, 5>(v2, v3);
                let t3 = i32x4_shuffle::<2, 6, 3, 7>(v2, v3);
                let xq = x >> 2;
                v128_store(d0.add(xq) as *mut v128, i32x4_shuffle::<0, 1, 4, 5>(t0, t2));
                v128_store(d1.add(xq) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t0, t2));
                v128_store(d2.add(xq) as *mut v128, i32x4_shuffle::<0, 1, 4, 5>(t1, t3));
                v128_store(d3.add(xq) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t1, t3));
                x += 16;
            }
            if x + 8 <= win_dim {
                let rp = rgba_ptr.add((row_px + x) * 4);
                let v0 = rgba4_luma_255(v128_load(rp as *const v128), w_luma, scale_luma);
                let v1 = rgba4_luma_255(v128_load(rp.add(16) as *const v128), w_luma, scale_luma);
                let t0 = i32x4_shuffle::<0, 4, 1, 5>(v0, v1);
                let t1 = i32x4_shuffle::<2, 6, 3, 7>(v0, v1);
                let xq = x >> 2;
                core::ptr::write_unaligned(d0.add(xq) as *mut u64, u64x2_extract_lane::<0>(t0));
                core::ptr::write_unaligned(d1.add(xq) as *mut u64, u64x2_extract_lane::<1>(t0));
                core::ptr::write_unaligned(d2.add(xq) as *mut u64, u64x2_extract_lane::<0>(t1));
                core::ptr::write_unaligned(d3.add(xq) as *mut u64, u64x2_extract_lane::<1>(t1));
                x += 8;
            }
            if x + 4 <= win_dim {
                let rp = rgba_ptr.add((row_px + x) * 4);
                let v0 = rgba4_luma_255(v128_load(rp as *const v128), w_luma, scale_luma);
                let xq = x >> 2;
                *d0.add(xq) = f32x4_extract_lane::<0>(v0);
                *d1.add(xq) = f32x4_extract_lane::<1>(v0);
                *d2.add(xq) = f32x4_extract_lane::<2>(v0);
                *d3.add(xq) = f32x4_extract_lane::<3>(v0);
                x += 4;
            }
            while x < win_dim {
                let off4 = (row_px + x) * 4;
                let l = (0.299 * (*rgba_ptr.add(off4) as f32)
                    + 0.587 * (*rgba_ptr.add(off4 + 1) as f32)
                    + 0.114 * (*rgba_ptr.add(off4 + 2) as f32))
                    * (1.0 / 255.0);
                *dst_base.add((ph_y + (x & 3)) * plane_sz + row_off + (x >> 2)) = l;
                x += 1;
            }
        }
        return (p_w, plane_sz);
    }

    for y in 0..win_dim {
        let py_c = (wy0 + (y as i32)).clamp(0, h_max) as usize;
        let roff = py_c * frame_w;
        let row_off = (y >> 1) * p_w;
        let ph_y = (y & 1) * 4;
        for x in 0..win_dim {
            let px_c = (wx0 + (x as i32)).clamp(0, w_max) as usize;
            let off4 = (roff + px_c) * 4;
            let l = (0.299 * (prev_rgba[off4] as f32)
                + 0.587 * (prev_rgba[off4 + 1] as f32)
                + 0.114 * (prev_rgba[off4 + 2] as f32))
                * (1.0 / 255.0);
            *dst_base.add((ph_y + (x & 3)) * plane_sz + row_off + (x >> 2)) = l;
        }
    }
    (p_w, plane_sz)
}

#[inline(always)]
unsafe fn render_eulerian_pip_impl<const EXACT: bool, const RESET: bool>(
    src_ptr: *const u8,
    ref_ptr: *mut f32,
    dst_ptr: *mut u8,
    ix0: usize,
    iy0: usize,
    frame_w: usize,
    pip_dim: usize,
    gain: f32,
    w00_v: v128,
    w10_v: v128,
    w01_v: v128,
    w11_v: v128,
    round_v: v128,
) {
    let row_bytes = frame_w * 4;
    let coeff_v = i16x8(9798, 19235, 3735, 0, 9798, 19235, 3735, 0);
    let inv_luma_scale = f32x4_splat(1.0 / (255.0 * 32768.0));
    let alpha_mask = u32x4_splat(0xFF00_0000);
    let gain_v = f32x4_splat(gain);
    let keep_v = f32x4_splat(0.80);
    let upd_v = f32x4_splat(0.20);
    let zero_v = f32x4_splat(0.0);
    let ar255_v = f32x4_splat(0.70 * 255.0);
    let sn255_v = f32x4_splat(0.65 * 255.0);
    let ag255_v = f32x4_splat(0.80 * 255.0);
    let spg255_v = f32x4_splat(0.45 * 255.0);
    let ab255_v = f32x4_splat(0.70 * 255.0);
    let spb255_v = f32x4_splat(0.85 * 255.0);
    let v255 = f32x4_splat(255.0);
    let vm255 = f32x4_splat(-255.0);
    let zero_i32 = i32x4_splat(0);

    for py in 0..pip_dim {
        let row0 = ((iy0 + py) * frame_w + ix0) * 4;
        let idx_row = py * pip_dim;
        let mut px = 0usize;
        while px < pip_dim {
            let p4 = px * 4;
            let (lo, hi) = if EXACT {
                let px_u8 = v128_load(src_ptr.add(row0 + p4) as *const v128);
                (
                    u16x8_extend_low_u8x16(px_u8),
                    u16x8_extend_high_u8x16(px_u8),
                )
            } else {
                bilinear_rgba4_i16_simd128(
                    src_ptr.add(row0 + p4),
                    row_bytes,
                    w00_v,
                    w10_v,
                    w01_v,
                    w11_v,
                    round_v,
                )
            };

            let d01 = i32x4_dot_i16x8(lo, coeff_v);
            let d23 = i32x4_dot_i16x8(hi, coeff_v);
            let rg_sum = i32x4_shuffle::<0, 2, 4, 6>(d01, d23);
            let b_sum = i32x4_shuffle::<1, 3, 5, 7>(d01, d23);
            let luma = f32x4_mul(
                f32x4_convert_i32x4(i32x4_add(rg_sum, b_sum)),
                inv_luma_scale,
            );
            let idx = idx_row + px;

            if RESET {
                v128_store(ref_ptr.add(idx) as *mut v128, luma);
                let packed = v128_or(u8x16_narrow_i16x8(lo, hi), alpha_mask);
                v128_store(dst_ptr.add(idx * 4) as *mut v128, packed);
            } else {
                let ref_prev = v128_load(ref_ptr.add(idx) as *const v128);
                let ref_next = f32x4_add(f32x4_mul(keep_v, ref_prev), f32x4_mul(upd_v, luma));
                v128_store(ref_ptr.add(idx) as *mut v128, ref_next);
                let dl = f32x4_sub(luma, ref_prev);

                let amp_l = f32x4_mul(gain_v, dl);
                let shear_pos = f32x4_pmax(amp_l, zero_v);
                let shear_neg = f32x4_pmax(f32x4_neg(amp_l), zero_v);

                let dr_v = i32x4_trunc_sat_f32x4(f32x4_pmax(
                    f32x4_pmin(
                        f32x4_add(f32x4_mul(amp_l, ar255_v), f32x4_mul(shear_neg, sn255_v)),
                        v255,
                    ),
                    vm255,
                ));
                let dg_v = i32x4_trunc_sat_f32x4(f32x4_pmax(
                    f32x4_pmin(
                        f32x4_add(f32x4_mul(amp_l, ag255_v), f32x4_mul(shear_pos, spg255_v)),
                        v255,
                    ),
                    vm255,
                ));
                let db_v = i32x4_trunc_sat_f32x4(f32x4_pmax(
                    f32x4_pmin(
                        f32x4_add(f32x4_mul(amp_l, ab255_v), f32x4_mul(shear_pos, spb255_v)),
                        v255,
                    ),
                    vm255,
                ));
                let rg16 = i16x8_narrow_i32x4(dr_v, dg_v);
                let b016 = i16x8_narrow_i32x4(db_v, zero_i32);
                let d_lo = i16x8_shuffle::<0, 4, 8, 12, 1, 5, 9, 13>(rg16, b016);
                let d_hi = i16x8_shuffle::<2, 6, 10, 14, 3, 7, 11, 15>(rg16, b016);
                let packed = v128_or(
                    u8x16_narrow_i16x8(i16x8_add(lo, d_lo), i16x8_add(hi, d_hi)),
                    alpha_mask,
                );
                v128_store(dst_ptr.add(idx * 4) as *mut v128, packed);
            }
            px += 4;
        }
    }
}

#[inline]
pub unsafe fn render_eulerian_pip_simd128(
    rgba: &[u8],
    frame_w: usize,
    frame_h: usize,
    sx0_f: f32,
    sy0_f: f32,
    pip_dim: usize,
    gain: f32,
    reset_ref: bool,
    ref_luma: &mut [f32],
    dst_rgba: &mut [u8],
) -> bool {
    if pip_dim == 0
        || (pip_dim & 3) != 0
        || frame_w < pip_dim + 2
        || frame_h < pip_dim + 2
        || sx0_f < 0.0
        || sy0_f < 0.0
    {
        return false;
    }
    let ix0 = sx0_f as usize;
    let iy0 = sy0_f as usize;
    if (ix0 + pip_dim - 1) as f32 >= (frame_w - 2) as f32
        || (iy0 + pip_dim - 1) as f32 >= (frame_h - 2) as f32
    {
        return false;
    }
    if ix0 + pip_dim + 1 > frame_w
        || iy0 + pip_dim + 1 > frame_h
        || ((iy0 + pip_dim) * frame_w + ix0 + pip_dim + 1) * 4 > rgba.len()
        || pip_dim * pip_dim > ref_luma.len()
        || pip_dim * pip_dim * 4 > dst_rgba.len()
    {
        return false;
    }

    let tx = sx0_f - (ix0 as f32);
    let ty = sy0_f - (iy0 as f32);
    let wx1 = ((tx * 16.0 + 0.5) as u16).min(16);
    let wx0 = 16 - wx1;
    let wy1 = ((ty * 16.0 + 0.5) as u16).min(16);
    let wy0 = 16 - wy1;
    let w00_v = u16x8_splat(wx0 * wy0);
    let w10_v = u16x8_splat(wx1 * wy0);
    let w01_v = u16x8_splat(wx0 * wy1);
    let w11_v = u16x8_splat(wx1 * wy1);
    let round_v = u16x8_splat(128);
    let exact_px = wx1 == 0 && wy1 == 0;

    let src_ptr = rgba.as_ptr();
    let ref_ptr = ref_luma.as_mut_ptr();
    let dst_ptr = dst_rgba.as_mut_ptr();
    match (exact_px, reset_ref) {
        (true, true) => render_eulerian_pip_impl::<true, true>(
            src_ptr, ref_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain, w00_v, w10_v, w01_v,
            w11_v, round_v,
        ),
        (true, false) => render_eulerian_pip_impl::<true, false>(
            src_ptr, ref_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain, w00_v, w10_v, w01_v,
            w11_v, round_v,
        ),
        (false, true) => render_eulerian_pip_impl::<false, true>(
            src_ptr, ref_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain, w00_v, w10_v, w01_v,
            w11_v, round_v,
        ),
        (false, false) => render_eulerian_pip_impl::<false, false>(
            src_ptr, ref_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain, w00_v, w10_v, w01_v,
            w11_v, round_v,
        ),
    }
    true
}

#[inline(always)]
unsafe fn pip_horiz4(r_cur: v128, s1: v128, wx0_u8: v128, wx1_u8: v128) -> (v128, v128) {
    let h_lo = i16x8_add(
        u16x8_extmul_low_u8x16(r_cur, wx0_u8),
        u16x8_extmul_low_u8x16(s1, wx1_u8),
    );
    let h_hi = i16x8_add(
        u16x8_extmul_high_u8x16(r_cur, wx0_u8),
        u16x8_extmul_high_u8x16(s1, wx1_u8),
    );
    (h_lo, h_hi)
}

#[inline(always)]
unsafe fn pip_vert4(
    h0_lo: v128,
    h0_hi: v128,
    h1_lo: v128,
    h1_hi: v128,
    fy_v: v128,
    round64_v: v128,
) -> (v128, v128) {
    let diff_lo = i16x8_sub(h1_lo, h0_lo);
    let diff_hi = i16x8_sub(h1_hi, h0_hi);
    let v_lo = i16x8_add(h0_lo, i16x8_q15mulr_sat(diff_lo, fy_v));
    let v_hi = i16x8_add(h0_hi, i16x8_q15mulr_sat(diff_hi, fy_v));
    let lo = u16x8_shr(i16x8_add(v_lo, round64_v), 7);
    let hi = u16x8_shr(i16x8_add(v_hi, round64_v), 7);
    (lo, hi)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pip_emit8<const RESET: bool, const FAST_GAIN: bool>(
    lo0: v128,
    hi0: v128,
    lo1: v128,
    hi1: v128,
    ref16_ptr: *mut i16,
    dst_ptr: *mut u8,
    idx: usize,
    coeff_v: v128,
    alpha_mask: v128,
    zero16: v128,
    upd_q15: v128,
    gain_q: v128,
    v_pos2048: v128,
    v_neg20400: v128,
    k_pos: v128,
    k_neg_abs: v128,
) {
    let d0 = i32x4_dot_i16x8(lo0, coeff_v);
    let d1 = i32x4_dot_i16x8(hi0, coeff_v);
    let l01 = u32x4_shr(
        i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(d0, d1),
            i32x4_shuffle::<1, 3, 5, 7>(d0, d1),
        ),
        8,
    );
    let d2 = i32x4_dot_i16x8(lo1, coeff_v);
    let d3 = i32x4_dot_i16x8(hi1, coeff_v);
    let l23 = u32x4_shr(
        i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(d2, d3),
            i32x4_shuffle::<1, 3, 5, 7>(d2, d3),
        ),
        8,
    );
    let l8 = i16x8_narrow_i32x4(l01, l23);

    if RESET {
        v128_store(ref16_ptr.add(idx) as *mut v128, l8);
        v128_store(
            dst_ptr.add(idx * 4) as *mut v128,
            v128_or(u8x16_narrow_i16x8(lo0, hi0), alpha_mask),
        );
        v128_store(
            dst_ptr.add(idx * 4 + 16) as *mut v128,
            v128_or(u8x16_narrow_i16x8(lo1, hi1), alpha_mask),
        );
    } else {
        let ref_prev = v128_load(ref16_ptr.add(idx) as *const v128);
        let dl = i16x8_sub_sat(l8, ref_prev);
        let ref_next = i16x8_add(ref_prev, i16x8_q15mulr_sat(dl, upd_q15));
        v128_store(ref16_ptr.add(idx) as *mut v128, ref_next);

        let q_raw = if FAST_GAIN {
            i16x8_q15mulr_sat(dl, gain_q)
        } else {
            let ql = i32x4_shr(i32x4_extmul_low_i16x8(dl, gain_q), 12);
            let qh = i32x4_shr(i32x4_extmul_high_i16x8(dl, gain_q), 12);
            i16x8_narrow_i32x4(ql, qh)
        };
        let q8 = i16x8_min(v_pos2048, i16x8_max(v_neg20400, q_raw));

        let q32_lo = i16x8_shuffle::<0, 0, 1, 1, 2, 2, 3, 3>(q8, q8);
        let q32_hi = i16x8_shuffle::<4, 4, 5, 5, 6, 6, 7, 7>(q8, q8);
        let q0_lo = i32x4_shuffle::<0, 0, 1, 1>(q32_lo, q32_lo);
        let q0_hi = i32x4_shuffle::<2, 2, 3, 3>(q32_lo, q32_lo);
        let m0_lo = i16x8_lt(q0_lo, zero16);
        let m0_hi = i16x8_lt(q0_hi, zero16);
        let d0_lo = i16x8_q15mulr_sat(q0_lo, v128_bitselect(k_neg_abs, k_pos, m0_lo));
        let d0_hi = i16x8_q15mulr_sat(q0_hi, v128_bitselect(k_neg_abs, k_pos, m0_hi));
        v128_store(
            dst_ptr.add(idx * 4) as *mut v128,
            v128_or(
                u8x16_narrow_i16x8(i16x8_add(lo0, d0_lo), i16x8_add(hi0, d0_hi)),
                alpha_mask,
            ),
        );

        let q1_lo = i32x4_shuffle::<0, 0, 1, 1>(q32_hi, q32_hi);
        let q1_hi = i32x4_shuffle::<2, 2, 3, 3>(q32_hi, q32_hi);
        let m1_lo = i16x8_lt(q1_lo, zero16);
        let m1_hi = i16x8_lt(q1_hi, zero16);
        let d1_lo = i16x8_q15mulr_sat(q1_lo, v128_bitselect(k_neg_abs, k_pos, m1_lo));
        let d1_hi = i16x8_q15mulr_sat(q1_hi, v128_bitselect(k_neg_abs, k_pos, m1_hi));
        v128_store(
            dst_ptr.add(idx * 4 + 16) as *mut v128,
            v128_or(
                u8x16_narrow_i16x8(i16x8_add(lo1, d1_lo), i16x8_add(hi1, d1_hi)),
                alpha_mask,
            ),
        );
    }
}

#[inline(never)]
unsafe fn render_eulerian_pip_exact_impl<const RESET: bool, const FAST_GAIN: bool>(
    src_ptr: *const u8,
    ref16_ptr: *mut i16,
    dst_ptr: *mut u8,
    ix0: usize,
    iy0: usize,
    frame_w: usize,
    pip_dim: usize,
    gain_q_scalar: i16,
) {
    let coeff_v = i16x8(9798, 19235, 3735, 0, 9798, 19235, 3735, 0);
    let alpha_mask = u32x4_splat(0xFF00_0000);
    let zero16 = i16x8_splat(0);
    let upd_q15 = i16x8_splat(6554);
    let gain_q = i16x8_splat(gain_q_scalar);
    let v_pos2048 = i16x8_splat(2048);
    let v_neg20400 = i16x8_splat(-20400);
    let k_pos = i16x8(5734, 10240, 12698, 0, 5734, 10240, 12698, 0);
    let k_neg_abs = i16x8(410, 6554, 5734, 0, 410, 6554, 5734, 0);
    let octs = pip_dim >> 3;

    for py in 0..pip_dim {
        let row0 = ((iy0 + py) * frame_w + ix0) * 4;
        let idx_row = py * pip_dim;
        for q in 0..octs {
            let p8 = q * 32;
            let px0 = v128_load(src_ptr.add(row0 + p8) as *const v128);
            let px1 = v128_load(src_ptr.add(row0 + p8 + 16) as *const v128);
            pip_emit8::<RESET, FAST_GAIN>(
                u16x8_extend_low_u8x16(px0),
                u16x8_extend_high_u8x16(px0),
                u16x8_extend_low_u8x16(px1),
                u16x8_extend_high_u8x16(px1),
                ref16_ptr,
                dst_ptr,
                idx_row + q * 8,
                coeff_v,
                alpha_mask,
                zero16,
                upd_q15,
                gain_q,
                v_pos2048,
                v_neg20400,
                k_pos,
                k_neg_abs,
            );
        }
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn render_eulerian_pip_subpx_impl<const RESET: bool, const FAST_GAIN: bool>(
    src_ptr: *const u8,
    ref16_ptr: *mut i16,
    dst_ptr: *mut u8,
    ix0: usize,
    iy0: usize,
    frame_w: usize,
    pip_dim: usize,
    gain_q_scalar: i16,
    wx0: i16,
    wx1: i16,
    fy_q15: i16,
) {
    let coeff_v = i16x8(9798, 19235, 3735, 0, 9798, 19235, 3735, 0);
    let alpha_mask = u32x4_splat(0xFF00_0000);
    let zero16 = i16x8_splat(0);
    let upd_q15 = i16x8_splat(6554);
    let gain_q = i16x8_splat(gain_q_scalar);
    let v_pos2048 = i16x8_splat(2048);
    let v_neg20400 = i16x8_splat(-20400);
    let k_pos = i16x8(5734, 10240, 12698, 0, 5734, 10240, 12698, 0);
    let k_neg_abs = i16x8(410, 6554, 5734, 0, 410, 6554, 5734, 0);
    let wx0_u8 = u8x16_splat(wx0 as u8);
    let wx1_u8 = u8x16_splat(wx1 as u8);
    let fy_v = i16x8_splat(fy_q15);
    let round64_v = i16x8_splat(64);

    let pairs = pip_dim >> 3;
    let stride_bytes = frame_w * 4;
    let base_ptr = src_ptr.add((iy0 * frame_w + ix0) * 4);

    for kp in 0..pairs {
        let mut row_base = base_ptr.add(kp * 32);
        let w0_0 = v128_load(row_base as *const v128);
        let w1_0 = v128_load(row_base.add(16) as *const v128);
        let w2_0 = v128_load32_zero(row_base.add(32) as *const u32);
        let nb0_0 = i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(w0_0, w1_0);
        let nb1_0 = i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(w1_0, w2_0);
        let (mut h0_lo0, mut h0_hi0) = pip_horiz4(w0_0, nb0_0, wx0_u8, wx1_u8);
        let (mut h0_lo1, mut h0_hi1) = pip_horiz4(w1_0, nb1_0, wx0_u8, wx1_u8);

        let mut out_idx = kp * 8;
        for _py in 0..pip_dim {
            row_base = row_base.add(stride_bytes);
            let w0 = v128_load(row_base as *const v128);
            let w1 = v128_load(row_base.add(16) as *const v128);
            let w2 = v128_load32_zero(row_base.add(32) as *const u32);
            let rnb0 = i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(w0, w1);
            let rnb1 = i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(w1, w2);
            let (h1_lo0, h1_hi0) = pip_horiz4(w0, rnb0, wx0_u8, wx1_u8);
            let (h1_lo1, h1_hi1) = pip_horiz4(w1, rnb1, wx0_u8, wx1_u8);
            let (r_lo0, r_hi0) = pip_vert4(h0_lo0, h0_hi0, h1_lo0, h1_hi0, fy_v, round64_v);
            let (r_lo1, r_hi1) = pip_vert4(h0_lo1, h0_hi1, h1_lo1, h1_hi1, fy_v, round64_v);
            h0_lo0 = h1_lo0;
            h0_hi0 = h1_hi0;
            h0_lo1 = h1_lo1;
            h0_hi1 = h1_hi1;
            pip_emit8::<RESET, FAST_GAIN>(
                r_lo0, r_hi0, r_lo1, r_hi1, ref16_ptr, dst_ptr, out_idx, coeff_v, alpha_mask,
                zero16, upd_q15, gain_q, v_pos2048, v_neg20400, k_pos, k_neg_abs,
            );
            out_idx += pip_dim;
        }
    }
}

#[inline]
pub unsafe fn render_eulerian_pip_fast_simd128(
    src_rgba: &[u8],
    frame_w: usize,
    frame_h: usize,
    x0: f32,
    y0: f32,
    pip_dim: usize,
    gain: f32,
    reset_ref: bool,
    ref_luma: &mut [f32],
    dst_rgba: &mut [u8],
) -> bool {
    let total_px = pip_dim * pip_dim;
    if (pip_dim & 7) != 0
        || pip_dim > 64
        || pip_dim < 8
        || x0 < 0.0
        || y0 < 0.0
        || ref_luma.len() < total_px
        || dst_rgba.len() < total_px * 4
    {
        return false;
    }
    let ix0 = x0 as usize;
    let iy0 = y0 as usize;
    let fx = x0 - (ix0 as f32);
    let fy = y0 - (iy0 as f32);
    let exact = fx < 1e-3 && fy < 1e-3;
    let req_w = if exact { ix0 + pip_dim } else { ix0 + pip_dim + 1 };
    let req_h = if exact { iy0 + pip_dim } else { iy0 + pip_dim + 1 };
    if req_w > frame_w || req_h > frame_h || src_rgba.len() < frame_w * frame_h * 4 {
        return false;
    }
    let src_ptr = src_rgba.as_ptr();
    let ref16_ptr = ref_luma.as_mut_ptr() as *mut i16;
    let dst_ptr = dst_rgba.as_mut_ptr();
    let gain_pos = if gain > 0.0 { gain.min(120.0) } else { 0.0 };
    let fast_gain = gain_pos < 31.9;
    let gain_q_scalar = if fast_gain {
        ((gain_pos * 1024.0 + 0.5) as i32).clamp(0, 32767) as i16
    } else {
        ((gain_pos * 128.0 + 0.5) as i32).clamp(0, 32767) as i16
    };
    let wx1 = ((fx * 128.0 + 0.5) as i16).clamp(0, 128);
    let wx0 = 128 - wx1;
    let fy_q15 = ((fy * 32768.0 + 0.5) as i32).clamp(0, 32767) as i16;
    match (exact, reset_ref, fast_gain) {
        (true, true, true) => render_eulerian_pip_exact_impl::<true, true>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar,
        ),
        (true, true, false) => render_eulerian_pip_exact_impl::<true, false>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar,
        ),
        (true, false, true) => render_eulerian_pip_exact_impl::<false, true>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar,
        ),
        (true, false, false) => render_eulerian_pip_exact_impl::<false, false>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar,
        ),
        (false, true, true) => render_eulerian_pip_subpx_impl::<true, true>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar, wx0, wx1, fy_q15,
        ),
        (false, true, false) => render_eulerian_pip_subpx_impl::<true, false>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar, wx0, wx1, fy_q15,
        ),
        (false, false, true) => render_eulerian_pip_subpx_impl::<false, true>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar, wx0, wx1, fy_q15,
        ),
        (false, false, false) => render_eulerian_pip_subpx_impl::<false, false>(
            src_ptr, ref16_ptr, dst_ptr, ix0, iy0, frame_w, pip_dim, gain_q_scalar, wx0, wx1, fy_q15,
        ),
    }
    true
}

/// Zero-Mean Foreground-Masked SIMD128 Patch Luma/Chroma SAD (`tpl_weight` normalized to sum = 1.0)
/// WITHOUT calling `haar_morton_signature_hyst_simd128` — designed for coarse hypothesis probes.
/// Applies weighted DC shift (`[-0.42, 0.42]`) and contrast gain (`[0.72, 1.45]`) STRICTLY to luma,
/// leaving `rg`/`by` unshifted and unscaled to preserve twin-chroma distractor rejection.
#[inline]
pub unsafe fn sad_patch_zm_score_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    out_weighted_eca: *mut f32,
    scale_px: usize,
) -> f32 {
    let cstep = 4 * scale_px;
    let mut sum_w = f32x4_splat(0.0);
    let mut sum_wi = f32x4_splat(0.0);
    let mut sum_wt = f32x4_splat(0.0);
    let mut acc_sad_c = f32x4_splat(0.0);
    let mut acc_eca = f32x4_splat(0.0);
    let has_eca = !eca_res_ptr.is_null();

    if scale_px <= 1 {
        let mut diff_l_uninit = [i32x4_splat(0); 64];
        let diff_l_ptr = diff_l_uninit.as_mut_ptr() as *mut v128;
        let mut k = 0usize;
        for ry in 0..16usize {
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 4;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = v128_load(luma_ptr.add(ioff) as *const v128);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                sum_w = f32x4_add(sum_w, vw);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                sum_wt = f32x4_add(sum_wt, f32x4_mul(vw, tl));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);

                let irg = v128_load(rg_ptr.add(ioff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let iby = v128_load(by_ptr.add(ioff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);
                let diff_c = f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));

                if has_eca {
                    let eca = v128_load(eca_res_ptr.add(ioff) as *const v128);
                    acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca));
                }
                k += 1;
            }
        }
        let inv_w = 1.0 / hsum_f32x4(sum_w).max(1e-6);
        let mu_i = hsum_f32x4(sum_wi) * inv_w;
        let mu_t = hsum_f32x4(sum_wt) * inv_w;
        let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
        let v_dc = f32x4_splat(dc);
        let mut acc_sad_l = f32x4_splat(0.0);
        for idx in 0..64usize {
            let vw = v128_load(tpl_weight.add(idx * 4) as *const v128);
            let dl = f32x4_abs(f32x4_sub(*diff_l_ptr.add(idx), v_dc));
            acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
        }
        let sad_l = hsum_f32x4(acc_sad_l);
        let sad_c = hsum_f32x4(acc_sad_c);
        let w_eca = hsum_f32x4(acc_eca);
        if !out_weighted_eca.is_null() {
            *out_weighted_eca = w_eca;
        }
        return sad_l + 1.85 * sad_c - 0.16 * w_eca.min(0.85);
    }

    let mut diff_l_uninit = [i32x4_splat(0); 64];
    let diff_l_ptr = diff_l_uninit.as_mut_ptr() as *mut v128;
    let mut k = 0usize;
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            sum_w = f32x4_add(sum_w, vw);
            sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
            sum_wt = f32x4_add(sum_wt, f32x4_mul(vw, tl));
            *diff_l_ptr.add(k) = f32x4_sub(il, tl);

            let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px);
            let trg = v128_load(tpl_rg.add(toff) as *const v128);
            let iby = load_pooled4_f32(by_ptr.add(ioff), stride, scale_px);
            let tby = v128_load(tpl_by.add(toff) as *const v128);
            let diff_c = f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
            acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));

            if has_eca {
                let eca = load_pooled4_f32(eca_res_ptr.add(ioff), stride, scale_px);
                acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca));
            }
            k += 1;
        }
    }

    let inv_w = 1.0 / hsum_f32x4(sum_w).max(1e-6);
    let mu_i = hsum_f32x4(sum_wi) * inv_w;
    let mu_t = hsum_f32x4(sum_wt) * inv_w;
    let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(dc);

    let mut acc_sad_l = f32x4_splat(0.0);
    for idx in 0..64usize {
        let vw = v128_load(tpl_weight.add(idx * 4) as *const v128);
        let dl = f32x4_abs(f32x4_sub(*diff_l_ptr.add(idx), v_dc));
        acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
    }

    let sad_l = hsum_f32x4(acc_sad_l);
    let sad_c = hsum_f32x4(acc_sad_c);
    let w_eca = hsum_f32x4(acc_eca);
    if !out_weighted_eca.is_null() {
        *out_weighted_eca = w_eca;
    }
    let motion_reward = w_eca.min(0.85);
    sad_l + 1.85 * sad_c - 0.16 * motion_reward
}

/// Precompute full 16x16 `(inv_w, mu_t)` stats for `sad_patch_zm_full_precomp_cutoff_simd128`.
#[inline]
pub unsafe fn tpl_full_stats_simd128(
    tpl_luma: *const f32,
    tpl_weight: *const f32,
) -> (f32, f32) {
    let mut sum_w0 = f32x4_splat(0.0);
    let mut sum_w1 = f32x4_splat(0.0);
    let mut sum_wt0 = f32x4_splat(0.0);
    let mut sum_wt1 = f32x4_splat(0.0);
    for idx in (0..64usize).step_by(2) {
        let off = idx * 4;
        let vw0 = v128_load(tpl_weight.add(off) as *const v128);
        let vw1 = v128_load(tpl_weight.add(off + 4) as *const v128);
        let tl0 = v128_load(tpl_luma.add(off) as *const v128);
        let tl1 = v128_load(tpl_luma.add(off + 4) as *const v128);
        sum_w0 = f32x4_add(sum_w0, vw0);
        sum_w1 = f32x4_add(sum_w1, vw1);
        sum_wt0 = f32x4_add(sum_wt0, f32x4_mul(vw0, tl0));
        sum_wt1 = f32x4_add(sum_wt1, f32x4_mul(vw1, tl1));
    }
    let inv_w = 1.0 / hsum_f32x4(f32x4_add(sum_w0, sum_w1)).max(1e-6);
    let mu_t = hsum_f32x4(f32x4_add(sum_wt0, sum_wt1)) * inv_w;
    (inv_w, mu_t)
}

#[inline(always)]
unsafe fn sad_patch_zm_full_precomp_cutoff_impl<const S: usize>(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    cutoff: f32,
    scale_px: usize,
) -> f32 {
    let cstep = 4 * scale_px;
    let v4 = f32x4_splat(4.0);
    let mut acc_sad_c = f32x4_splat(0.0);

    macro_rules! chroma_row {
        ($ry:expr) => {{
            let ry: usize = $ry;
            let tpl_row = ry * 16;
            if S == 1 {
                let img_row = (cy + ry) * stride + cx;
                for vec_idx in 0..4usize {
                    let ioff = img_row + vec_idx * 4;
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = v128_load(rg_ptr.add(ioff) as *const v128);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let iby = v128_load(by_ptr.add(ioff) as *const v128);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);
                    let diff_c =
                        f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                    acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));
                }
            } else if S == 2 {
                let img_row = (cy + ry * 2) * stride + cx;
                for vec_idx in 0..4usize {
                    let ioff = img_row + vec_idx * 8;
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = load_sum4_2x2_f32(rg_ptr.add(ioff), stride);
                    let trg = f32x4_mul(v128_load(tpl_rg.add(toff) as *const v128), v4);
                    let iby = load_sum4_2x2_f32(by_ptr.add(ioff), stride);
                    let tby = f32x4_mul(v128_load(tpl_by.add(toff) as *const v128), v4);
                    let diff_c =
                        f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                    acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));
                }
            } else {
                let img_row = (cy + ry * scale_px) * stride + cx;
                for vec_idx in 0..4usize {
                    let ioff = img_row + vec_idx * cstep;
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let iby = load_pooled4_f32(by_ptr.add(ioff), stride, scale_px);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);
                    let diff_c =
                        f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                    acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));
                }
            }
        }};
    }

    let c_scale = if S == 2 { 1.85 * 0.25 } else { 1.85 };
    for ry in 5usize..11usize {
        chroma_row!(ry);
    }
    if c_scale * hsum_f32x4(acc_sad_c) >= cutoff {
        return cutoff;
    }
    for ry in [3usize, 4, 11, 12] {
        chroma_row!(ry);
    }
    if c_scale * hsum_f32x4(acc_sad_c) >= cutoff {
        return cutoff;
    }
    for ry in [0usize, 1, 2, 13, 14, 15] {
        chroma_row!(ry);
    }
    let sad_c_weighted = c_scale * hsum_f32x4(acc_sad_c);
    if sad_c_weighted >= cutoff {
        return cutoff;
    }

    let mut sum_wi = f32x4_splat(0.0);
    let mut diff_l_uninit = [i32x4_splat(0); 64];
    let diff_l_ptr = diff_l_uninit.as_mut_ptr() as *mut v128;
    let mut k = 0usize;
    if S == 1 {
        for ry in 0..16usize {
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 4;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = v128_load(luma_ptr.add(ioff) as *const v128);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);
                k += 1;
            }
        }
    } else if S == 2 {
        for ry in 0..16usize {
            let img_row = (cy + ry * 2) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 8;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = load_sum4_2x2_f32(luma_ptr.add(ioff), stride);
                let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);
                k += 1;
            }
        }
    } else {
        for ry in 0..16usize {
            let img_row = (cy + ry * scale_px) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * cstep;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);
                k += 1;
            }
        }
    }

    let mu_i = hsum_f32x4(sum_wi) * (if S == 2 { 0.25 * inv_w } else { inv_w });
    let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(if S == 2 { 4.0 * dc } else { dc });
    let mut acc_sad_l = f32x4_splat(0.0);
    for idx in 0..64usize {
        let vw = v128_load(tpl_weight.add(idx * 4) as *const v128);
        let dl = f32x4_abs(f32x4_sub(*diff_l_ptr.add(idx), v_dc));
        acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
    }
    let sad_l = if S == 2 { 0.25 * hsum_f32x4(acc_sad_l) } else { hsum_f32x4(acc_sad_l) };
    sad_l + sad_c_weighted
}

/// Scale-1 (`320x180` SD) exact full-`16x16` zero-mean weighted SAD + ECA reward with chroma-first
/// lower-bound early exit and precomputed `(inv_w, mu_t)`. Returns `(score, w_eca)` where
/// `score >= cutoff` means the patch cannot beat `cutoff`.
#[inline(never)]
pub unsafe fn sad_patch_zm_full_sd_eca_cutoff_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    cutoff: f32,
) -> (f32, f32) {
    let has_eca = !eca_res_ptr.is_null();
    let eca_slack = if has_eca { 0.1361f32 } else { 0.0f32 };
    let c_cutoff = cutoff + eca_slack;

    let mut acc_sad_c0 = f32x4_splat(0.0);
    let mut acc_sad_c1 = f32x4_splat(0.0);
    macro_rules! chroma_row_sd {
        ($ry:expr) => {{
            let ry: usize = $ry;
            let tpl_row = ry * 16;
            let img_row = (cy + ry) * stride + cx;

            let vw0 = v128_load(tpl_weight.add(tpl_row) as *const v128);
            let dc0 = f32x4_add(
                f32x4_abs(f32x4_sub(
                    v128_load(rg_ptr.add(img_row) as *const v128),
                    v128_load(tpl_rg.add(tpl_row) as *const v128),
                )),
                f32x4_abs(f32x4_sub(
                    v128_load(by_ptr.add(img_row) as *const v128),
                    v128_load(tpl_by.add(tpl_row) as *const v128),
                )),
            );
            acc_sad_c0 = f32x4_add(acc_sad_c0, f32x4_mul(vw0, dc0));

            let vw1 = v128_load(tpl_weight.add(tpl_row + 4) as *const v128);
            let dc1 = f32x4_add(
                f32x4_abs(f32x4_sub(
                    v128_load(rg_ptr.add(img_row + 4) as *const v128),
                    v128_load(tpl_rg.add(tpl_row + 4) as *const v128),
                )),
                f32x4_abs(f32x4_sub(
                    v128_load(by_ptr.add(img_row + 4) as *const v128),
                    v128_load(tpl_by.add(tpl_row + 4) as *const v128),
                )),
            );
            acc_sad_c1 = f32x4_add(acc_sad_c1, f32x4_mul(vw1, dc1));

            let vw2 = v128_load(tpl_weight.add(tpl_row + 8) as *const v128);
            let dc2 = f32x4_add(
                f32x4_abs(f32x4_sub(
                    v128_load(rg_ptr.add(img_row + 8) as *const v128),
                    v128_load(tpl_rg.add(tpl_row + 8) as *const v128),
                )),
                f32x4_abs(f32x4_sub(
                    v128_load(by_ptr.add(img_row + 8) as *const v128),
                    v128_load(tpl_by.add(tpl_row + 8) as *const v128),
                )),
            );
            acc_sad_c0 = f32x4_add(acc_sad_c0, f32x4_mul(vw2, dc2));

            let vw3 = v128_load(tpl_weight.add(tpl_row + 12) as *const v128);
            let dc3 = f32x4_add(
                f32x4_abs(f32x4_sub(
                    v128_load(rg_ptr.add(img_row + 12) as *const v128),
                    v128_load(tpl_rg.add(tpl_row + 12) as *const v128),
                )),
                f32x4_abs(f32x4_sub(
                    v128_load(by_ptr.add(img_row + 12) as *const v128),
                    v128_load(tpl_by.add(tpl_row + 12) as *const v128),
                )),
            );
            acc_sad_c1 = f32x4_add(acc_sad_c1, f32x4_mul(vw3, dc3));
        }};
    }

    for ry in 4usize..12usize {
        chroma_row_sd!(ry);
    }
    if 1.85 * hsum_f32x4(f32x4_add(acc_sad_c0, acc_sad_c1)) >= c_cutoff {
        return (cutoff, 0.0);
    }
    for ry in 0usize..4usize {
        chroma_row_sd!(ry);
    }
    for ry in 12usize..16usize {
        chroma_row_sd!(ry);
    }
    let sad_c_weighted = 1.85 * hsum_f32x4(f32x4_add(acc_sad_c0, acc_sad_c1));
    if sad_c_weighted >= c_cutoff {
        return (cutoff, 0.0);
    }

    // Tight mathematical lower-bound slack: |dc| <= 0.048 => sum(vw*|dc|) <= 0.048/inv_w.
    let raw_cutoff = (c_cutoff - sad_c_weighted) + (0.048 / inv_w.max(1e-6));
    let mut sum_wi0 = f32x4_splat(0.0);
    let mut sum_wi1 = f32x4_splat(0.0);
    let mut acc_l_raw0 = f32x4_splat(0.0);
    let mut acc_l_raw1 = f32x4_splat(0.0);
    let mut diff_l_uninit = [i32x4_splat(0); 64];
    let diff_l_ptr = diff_l_uninit.as_mut_ptr() as *mut v128;

    macro_rules! luma_row_sd {
        ($ry:expr) => {{
            let ry: usize = $ry;
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            let dst_base = ry * 4;

            let vw0 = v128_load(tpl_weight.add(tpl_row) as *const v128);
            let il0 = v128_load(luma_ptr.add(img_row) as *const v128);
            let d0 = f32x4_sub(il0, v128_load(tpl_luma.add(tpl_row) as *const v128));
            sum_wi0 = f32x4_add(sum_wi0, f32x4_mul(vw0, il0));
            acc_l_raw0 = f32x4_add(acc_l_raw0, f32x4_mul(vw0, f32x4_abs(d0)));
            *diff_l_ptr.add(dst_base) = d0;

            let vw1 = v128_load(tpl_weight.add(tpl_row + 4) as *const v128);
            let il1 = v128_load(luma_ptr.add(img_row + 4) as *const v128);
            let d1 = f32x4_sub(il1, v128_load(tpl_luma.add(tpl_row + 4) as *const v128));
            sum_wi1 = f32x4_add(sum_wi1, f32x4_mul(vw1, il1));
            acc_l_raw1 = f32x4_add(acc_l_raw1, f32x4_mul(vw1, f32x4_abs(d1)));
            *diff_l_ptr.add(dst_base + 1) = d1;

            let vw2 = v128_load(tpl_weight.add(tpl_row + 8) as *const v128);
            let il2 = v128_load(luma_ptr.add(img_row + 8) as *const v128);
            let d2 = f32x4_sub(il2, v128_load(tpl_luma.add(tpl_row + 8) as *const v128));
            sum_wi0 = f32x4_add(sum_wi0, f32x4_mul(vw2, il2));
            acc_l_raw0 = f32x4_add(acc_l_raw0, f32x4_mul(vw2, f32x4_abs(d2)));
            *diff_l_ptr.add(dst_base + 2) = d2;

            let vw3 = v128_load(tpl_weight.add(tpl_row + 12) as *const v128);
            let il3 = v128_load(luma_ptr.add(img_row + 12) as *const v128);
            let d3 = f32x4_sub(il3, v128_load(tpl_luma.add(tpl_row + 12) as *const v128));
            sum_wi1 = f32x4_add(sum_wi1, f32x4_mul(vw3, il3));
            acc_l_raw1 = f32x4_add(acc_l_raw1, f32x4_mul(vw3, f32x4_abs(d3)));
            *diff_l_ptr.add(dst_base + 3) = d3;
        }};
    }

    for ry in 4usize..12usize {
        luma_row_sd!(ry);
    }
    if hsum_f32x4(f32x4_add(acc_l_raw0, acc_l_raw1)) >= raw_cutoff {
        return (cutoff, 0.0);
    }
    for ry in 0usize..4usize {
        luma_row_sd!(ry);
    }
    for ry in 12usize..16usize {
        luma_row_sd!(ry);
    }
    if hsum_f32x4(f32x4_add(acc_l_raw0, acc_l_raw1)) >= raw_cutoff {
        return (cutoff, 0.0);
    }

    let mu_i = hsum_f32x4(f32x4_add(sum_wi0, sum_wi1)) * inv_w;
    let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(dc);
    let mut acc_sad_l0 = f32x4_splat(0.0);
    let mut acc_sad_l1 = f32x4_splat(0.0);
    let mut idx = 0usize;
    while idx < 64 {
        let vw0 = v128_load(tpl_weight.add(idx * 4) as *const v128);
        let dl0 = f32x4_abs(f32x4_sub(*diff_l_ptr.add(idx), v_dc));
        acc_sad_l0 = f32x4_add(acc_sad_l0, f32x4_mul(vw0, dl0));

        let vw1 = v128_load(tpl_weight.add((idx + 1) * 4) as *const v128);
        let dl1 = f32x4_abs(f32x4_sub(*diff_l_ptr.add(idx + 1), v_dc));
        acc_sad_l1 = f32x4_add(acc_sad_l1, f32x4_mul(vw1, dl1));
        idx += 2;
    }
    let sad_l = hsum_f32x4(f32x4_add(acc_sad_l0, acc_sad_l1));
    let tot_no_eca = sad_l + sad_c_weighted;
    if tot_no_eca - eca_slack >= cutoff {
        return (cutoff, 0.0);
    }

    let w_eca = if has_eca {
        let mut acc_eca0 = f32x4_splat(0.0);
        let mut acc_eca1 = f32x4_splat(0.0);
        for ry in 0..16usize {
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            acc_eca0 = f32x4_add(
                acc_eca0,
                f32x4_mul(
                    v128_load(tpl_weight.add(tpl_row) as *const v128),
                    v128_load(eca_res_ptr.add(img_row) as *const v128),
                ),
            );
            acc_eca1 = f32x4_add(
                acc_eca1,
                f32x4_mul(
                    v128_load(tpl_weight.add(tpl_row + 4) as *const v128),
                    v128_load(eca_res_ptr.add(img_row + 4) as *const v128),
                ),
            );
            acc_eca0 = f32x4_add(
                acc_eca0,
                f32x4_mul(
                    v128_load(tpl_weight.add(tpl_row + 8) as *const v128),
                    v128_load(eca_res_ptr.add(img_row + 8) as *const v128),
                ),
            );
            acc_eca1 = f32x4_add(
                acc_eca1,
                f32x4_mul(
                    v128_load(tpl_weight.add(tpl_row + 12) as *const v128),
                    v128_load(eca_res_ptr.add(img_row + 12) as *const v128),
                ),
            );
        }
        hsum_f32x4(f32x4_add(acc_eca0, acc_eca1))
    } else {
        0.0
    };
    (tot_no_eca - 0.16 * w_eca.min(0.85), w_eca)
}

#[inline(always)]
unsafe fn sad_full_weca_at_sd_simd128(
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_weight: *const f32,
) -> f32 {
    if eca_res_ptr.is_null() {
        return 0.0;
    }
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    for ry in 0..16usize {
        let img_row = (cy + ry) * stride + cx;
        let tpl_row = ry * 16;
        acc0 = f32x4_add(
            acc0,
            f32x4_mul(
                v128_load(tpl_weight.add(tpl_row) as *const v128),
                v128_load(eca_res_ptr.add(img_row) as *const v128),
            ),
        );
        acc1 = f32x4_add(
            acc1,
            f32x4_mul(
                v128_load(tpl_weight.add(tpl_row + 4) as *const v128),
                v128_load(eca_res_ptr.add(img_row + 4) as *const v128),
            ),
        );
        acc0 = f32x4_add(
            acc0,
            f32x4_mul(
                v128_load(tpl_weight.add(tpl_row + 8) as *const v128),
                v128_load(eca_res_ptr.add(img_row + 8) as *const v128),
            ),
        );
        acc1 = f32x4_add(
            acc1,
            f32x4_mul(
                v128_load(tpl_weight.add(tpl_row + 12) as *const v128),
                v128_load(eca_res_ptr.add(img_row + 12) as *const v128),
            ),
        );
    }
    hsum_f32x4(f32x4_add(acc0, acc1))
}

#[inline(always)]
unsafe fn load5_full_sd_row_f32(ptr: *const f32) -> [v128; 5] {
    [
        v128_load(ptr as *const v128),
        v128_load(ptr.add(4) as *const v128),
        v128_load(ptr.add(8) as *const v128),
        v128_load(ptr.add(12) as *const v128),
        v128_load(ptr.add(16) as *const v128),
    ]
}

#[inline(always)]
unsafe fn shift5_sd_ring_vecs<const ST: usize>(
    row: &[v128; 5],
) -> ([v128; 4], [v128; 4], [v128; 4]) {
    let x0 = [row[0], row[1], row[2], row[3]];
    if ST == 1 {
        let x1 = [
            i32x4_shuffle::<1, 2, 3, 4>(row[0], row[1]),
            i32x4_shuffle::<1, 2, 3, 4>(row[1], row[2]),
            i32x4_shuffle::<1, 2, 3, 4>(row[2], row[3]),
            i32x4_shuffle::<1, 2, 3, 4>(row[3], row[4]),
        ];
        let x2 = [
            i32x4_shuffle::<2, 3, 4, 5>(row[0], row[1]),
            i32x4_shuffle::<2, 3, 4, 5>(row[1], row[2]),
            i32x4_shuffle::<2, 3, 4, 5>(row[2], row[3]),
            i32x4_shuffle::<2, 3, 4, 5>(row[3], row[4]),
        ];
        (x0, x1, x2)
    } else {
        let x1 = [
            i32x4_shuffle::<2, 3, 4, 5>(row[0], row[1]),
            i32x4_shuffle::<2, 3, 4, 5>(row[1], row[2]),
            i32x4_shuffle::<2, 3, 4, 5>(row[2], row[3]),
            i32x4_shuffle::<2, 3, 4, 5>(row[3], row[4]),
        ];
        let x2 = [row[1], row[2], row[3], row[4]];
        (x0, x1, x2)
    }
}

/// Fused multi-offset 8-neighbor full-`16x16` SD ring evaluator (`ST` in `{1, 2}`) around `(cx0, cy0)`.
/// Streams each image and template row once across overlapping horizontal and vertical offsets.
#[inline(never)]
pub unsafe fn sad_patch_zm_full_sd_ring3x3_simd128<const ST: usize>(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    out_b_tlx: &mut i32,
    out_b_tly: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    const RING_OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let has_eca = !eca_res_ptr.is_null();
    let eca_slack = if has_eca { 0.1361f32 } else { 0.0f32 };
    let c_cutoff = *out_best_cost + eca_slack;

    let zero = f32x4_splat(0.0);
    let mut sum_wi = [zero; 8];
    let mut acc_sad_c = [zero; 8];

    let x_base = cx0 - ST;
    let y_base = cy0 - ST;
    let s_rows = 16 + 2 * ST;

    macro_rules! load_sd_ring_row {
        ($ptr:expr) => {{
            if ST == 1 {
                load5_sd_row_f32($ptr)
            } else {
                load5_full_sd_row_f32($ptr)
            }
        }};
    }

    for s_row in 0..s_rows {
        let row_off = (y_base + s_row) * stride + x_base;
        let il_row = load_sd_ring_row!(luma_ptr.add(row_off));
        let irg_row = load_sd_ring_row!(rg_ptr.add(row_off));
        let iby_row = load_sd_ring_row!(by_ptr.add(row_off));

        let (il_x0, il_x1, il_x2) = shift5_sd_ring_vecs::<ST>(&il_row);
        let (rg_x0, rg_x1, rg_x2) = shift5_sd_ring_vecs::<ST>(&irg_row);
        let (by_x0, by_x1, by_x2) = shift5_sd_ring_vecs::<ST>(&iby_row);

        if s_row < 16 {
            let tpl_row = s_row * 16;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);

                let dc0 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x0[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x0[vec_idx], tby)),
                );
                sum_wi[0] = f32x4_add(sum_wi[0], f32x4_mul(vw, il_x0[vec_idx]));
                acc_sad_c[0] = f32x4_add(acc_sad_c[0], f32x4_mul(vw, dc0));

                let dc1 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x1[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x1[vec_idx], tby)),
                );
                sum_wi[1] = f32x4_add(sum_wi[1], f32x4_mul(vw, il_x1[vec_idx]));
                acc_sad_c[1] = f32x4_add(acc_sad_c[1], f32x4_mul(vw, dc1));

                let dc2 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x2[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x2[vec_idx], tby)),
                );
                sum_wi[2] = f32x4_add(sum_wi[2], f32x4_mul(vw, il_x2[vec_idx]));
                acc_sad_c[2] = f32x4_add(acc_sad_c[2], f32x4_mul(vw, dc2));
            }
        }
        if s_row >= ST && s_row < 16 + ST {
            let tpl_row = (s_row - ST) * 16;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);

                let dc3 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x0[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x0[vec_idx], tby)),
                );
                sum_wi[3] = f32x4_add(sum_wi[3], f32x4_mul(vw, il_x0[vec_idx]));
                acc_sad_c[3] = f32x4_add(acc_sad_c[3], f32x4_mul(vw, dc3));

                let dc4 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x2[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x2[vec_idx], tby)),
                );
                sum_wi[4] = f32x4_add(sum_wi[4], f32x4_mul(vw, il_x2[vec_idx]));
                acc_sad_c[4] = f32x4_add(acc_sad_c[4], f32x4_mul(vw, dc4));
            }
        }
        if s_row >= 2 * ST {
            let tpl_row = (s_row - 2 * ST) * 16;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);

                let dc5 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x0[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x0[vec_idx], tby)),
                );
                sum_wi[5] = f32x4_add(sum_wi[5], f32x4_mul(vw, il_x0[vec_idx]));
                acc_sad_c[5] = f32x4_add(acc_sad_c[5], f32x4_mul(vw, dc5));

                let dc6 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x1[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x1[vec_idx], tby)),
                );
                sum_wi[6] = f32x4_add(sum_wi[6], f32x4_mul(vw, il_x1[vec_idx]));
                acc_sad_c[6] = f32x4_add(acc_sad_c[6], f32x4_mul(vw, dc6));

                let dc7 = f32x4_add(
                    f32x4_abs(f32x4_sub(rg_x2[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(by_x2[vec_idx], tby)),
                );
                sum_wi[7] = f32x4_add(sum_wi[7], f32x4_mul(vw, il_x2[vec_idx]));
                acc_sad_c[7] = f32x4_add(acc_sad_c[7], f32x4_mul(vw, dc7));
            }
        }
    }

    let mut sad_c_w = [0.0f32; 8];
    let mut alive = [false; 8];
    let mut any_alive = false;
    let mut v_dc = [zero; 8];
    for c in 0..8usize {
        let sc = 1.85 * hsum_f32x4(acc_sad_c[c]);
        sad_c_w[c] = sc;
        if sc < c_cutoff {
            alive[c] = true;
            any_alive = true;
            let mu_i = hsum_f32x4(sum_wi[c]) * inv_w;
            let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
            v_dc[c] = f32x4_splat(dc);
        }
    }
    if !any_alive {
        return;
    }

    let top_alive = alive[0] || alive[1] || alive[2];
    let mid_alive = alive[3] || alive[4];
    let bot_alive = alive[5] || alive[6] || alive[7];

    let mut acc_sad_l = [zero; 8];
    for s_row in 0..s_rows {
        let need_top = top_alive && s_row < 16;
        let need_mid = mid_alive && s_row >= ST && s_row < 16 + ST;
        let need_bot = bot_alive && s_row >= 2 * ST;
        if !(need_top || need_mid || need_bot) {
            continue;
        }
        let row_off = (y_base + s_row) * stride + x_base;
        let il_row = load_sd_ring_row!(luma_ptr.add(row_off));
        let (il_x0, il_x1, il_x2) = shift5_sd_ring_vecs::<ST>(&il_row);

        if need_top {
            let tpl_row = s_row * 16;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                if alive[0] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x0[vec_idx], tl), v_dc[0]));
                    acc_sad_l[0] = f32x4_add(acc_sad_l[0], f32x4_mul(vw, dl));
                }
                if alive[1] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x1[vec_idx], tl), v_dc[1]));
                    acc_sad_l[1] = f32x4_add(acc_sad_l[1], f32x4_mul(vw, dl));
                }
                if alive[2] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x2[vec_idx], tl), v_dc[2]));
                    acc_sad_l[2] = f32x4_add(acc_sad_l[2], f32x4_mul(vw, dl));
                }
            }
        }
        if need_mid {
            let tpl_row = (s_row - ST) * 16;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                if alive[3] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x0[vec_idx], tl), v_dc[3]));
                    acc_sad_l[3] = f32x4_add(acc_sad_l[3], f32x4_mul(vw, dl));
                }
                if alive[4] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x2[vec_idx], tl), v_dc[4]));
                    acc_sad_l[4] = f32x4_add(acc_sad_l[4], f32x4_mul(vw, dl));
                }
            }
        }
        if need_bot {
            let tpl_row = (s_row - 2 * ST) * 16;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                if alive[5] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x0[vec_idx], tl), v_dc[5]));
                    acc_sad_l[5] = f32x4_add(acc_sad_l[5], f32x4_mul(vw, dl));
                }
                if alive[6] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x1[vec_idx], tl), v_dc[6]));
                    acc_sad_l[6] = f32x4_add(acc_sad_l[6], f32x4_mul(vw, dl));
                }
                if alive[7] {
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il_x2[vec_idx], tl), v_dc[7]));
                    acc_sad_l[7] = f32x4_add(acc_sad_l[7], f32x4_mul(vw, dl));
                }
            }
        }
    }

    let st_i = ST as i32;
    for c in 0..8usize {
        if !alive[c] || sad_c_w[c] >= *out_best_cost + eca_slack {
            continue;
        }
        let tot_no_eca = hsum_f32x4(acc_sad_l[c]) + sad_c_w[c];
        if tot_no_eca - eca_slack >= *out_best_cost {
            continue;
        }
        let (dx, dy) = RING_OFFSETS[c];
        let qx = (cx0 as i32 + dx * st_i) as usize;
        let qy = (cy0 as i32 + dy * st_i) as usize;
        let w_eca = sad_full_weca_at_sd_simd128(eca_res_ptr, stride, qx, qy, tpl_weight);
        let rc = tot_no_eca - 0.16 * w_eca.min(0.85);
        if rc < *out_best_cost {
            *out_best_cost = rc;
            *out_best_weca = w_eca;
            *out_b_tlx = qx as i32;
            *out_b_tly = qy as i32;
        }
    }
}

#[inline(never)]
pub unsafe fn sad_patch_zm_full_precomp_cutoff_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    cutoff: f32,
    scale_px: usize,
) -> f32 {
    if scale_px == 2 {
        sad_patch_zm_full_precomp_cutoff_impl::<2>(
            luma_ptr, rg_ptr, by_ptr, stride, cx, cy, tpl_luma, tpl_rg, tpl_by, tpl_weight,
            inv_w, mu_t, cutoff, 2,
        )
    } else if scale_px <= 1 {
        sad_patch_zm_full_precomp_cutoff_impl::<1>(
            luma_ptr, rg_ptr, by_ptr, stride, cx, cy, tpl_luma, tpl_rg, tpl_by, tpl_weight,
            inv_w, mu_t, cutoff, 1,
        )
    } else {
        sad_patch_zm_full_precomp_cutoff_impl::<0>(
            luma_ptr, rg_ptr, by_ptr, stride, cx, cy, tpl_luma, tpl_rg, tpl_by, tpl_weight,
            inv_w, mu_t, cutoff, scale_px,
        )
    }
}

/// Even-row (8x16 = 128-sample) fast normalized variant of `sad_patch_zm_score_simd128`
/// used for shift candidate comparison prior to exact full 16x16 evaluation on the winner.
#[inline]
pub unsafe fn tpl_half_stats_simd128(
    tpl_luma: *const f32,
    tpl_weight: *const f32,
) -> (f32, f32) {
    let mut sum_w0 = f32x4_splat(0.0);
    let mut sum_w1 = f32x4_splat(0.0);
    let mut sum_wt0 = f32x4_splat(0.0);
    let mut sum_wt1 = f32x4_splat(0.0);
    for row_i in 0..8usize {
        let tr = row_i * 32;
        let vw0 = v128_load(tpl_weight.add(tr) as *const v128);
        let vw1 = v128_load(tpl_weight.add(tr + 4) as *const v128);
        let vw2 = v128_load(tpl_weight.add(tr + 8) as *const v128);
        let vw3 = v128_load(tpl_weight.add(tr + 12) as *const v128);
        let tl0 = v128_load(tpl_luma.add(tr) as *const v128);
        let tl1 = v128_load(tpl_luma.add(tr + 4) as *const v128);
        let tl2 = v128_load(tpl_luma.add(tr + 8) as *const v128);
        let tl3 = v128_load(tpl_luma.add(tr + 12) as *const v128);
        sum_w0 = f32x4_add(sum_w0, f32x4_add(vw0, vw2));
        sum_w1 = f32x4_add(sum_w1, f32x4_add(vw1, vw3));
        sum_wt0 = f32x4_add(
            sum_wt0,
            f32x4_add(f32x4_mul(vw0, tl0), f32x4_mul(vw2, tl2)),
        );
        sum_wt1 = f32x4_add(
            sum_wt1,
            f32x4_add(f32x4_mul(vw1, tl1), f32x4_mul(vw3, tl3)),
        );
    }
    let inv_w = 1.0 / hsum_f32x4(f32x4_add(sum_w0, sum_w1)).max(1e-6);
    let mu_t = hsum_f32x4(f32x4_add(sum_wt0, sum_wt1)) * inv_w;
    (inv_w, mu_t)
}

/// Cutoff-pruned `sad_patch_zm_half` with precomputed `(inv_w, mu_t)`:
/// evaluates core chroma rows first, then outer chroma rows, then luma, then ECA,
/// skipping remaining planes whenever the exact lower bound reaches `cutoff`.
#[inline(always)]
unsafe fn sad_patch_zm_half_precomp_cutoff_impl<const S: usize>(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    cutoff: f32,
    out_w_eca: *mut f32,
    scale_px: usize,
) -> f32 {
    if !out_w_eca.is_null() {
        *out_w_eca = 0.0;
    }
    let has_eca = !eca_res_ptr.is_null();
    let eca_lb = if has_eca { 0.1365f32 } else { 0.0f32 };
    let cstep = 4 * scale_px;
    let inv_w_eff = if S == 2 { inv_w * 0.25 } else { inv_w };
    let v4 = f32x4_splat(4.0);
    let mut acc_sad_c = f32x4_splat(0.0);

    macro_rules! chroma_row {
        ($ry:expr) => {{
            let ry: usize = $ry;
            let tpl_row = ry * 16;
            if S == 1 {
                let img_row = (cy + ry) * stride + cx;
                for vec_idx in 0..4usize {
                    let ioff = img_row + vec_idx * 4;
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = v128_load(rg_ptr.add(ioff) as *const v128);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let iby = v128_load(by_ptr.add(ioff) as *const v128);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);
                    let diff_c =
                        f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                    acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));
                }
            } else if S == 2 {
                let img_row = (cy + ry * 2) * stride + cx;
                for vec_idx in 0..4usize {
                    let ioff = img_row + vec_idx * 8;
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = load_sum4_2x2_f32(rg_ptr.add(ioff), stride);
                    let trg = f32x4_mul(v128_load(tpl_rg.add(toff) as *const v128), v4);
                    let iby = load_sum4_2x2_f32(by_ptr.add(ioff), stride);
                    let tby = f32x4_mul(v128_load(tpl_by.add(toff) as *const v128), v4);
                    let diff_c =
                        f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                    acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));
                }
            } else {
                let img_row = (cy + ry * scale_px) * stride + cx;
                for vec_idx in 0..4usize {
                    let ioff = img_row + vec_idx * cstep;
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let iby = load_pooled4_f32(by_ptr.add(ioff), stride, scale_px);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);
                    let diff_c =
                        f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                    acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));
                }
            }
        }};
    }

    for ry in [4usize, 6usize, 8usize, 10usize] {
        chroma_row!(ry);
    }
    let sad_c_core = hsum_f32x4(acc_sad_c) * inv_w_eff;
    if 1.85 * sad_c_core - eca_lb >= cutoff {
        return cutoff;
    }
    for ry in [0usize, 2usize, 12usize, 14usize] {
        chroma_row!(ry);
    }

    let sad_c = hsum_f32x4(acc_sad_c) * inv_w_eff;
    if 1.85 * sad_c - eca_lb >= cutoff {
        return cutoff;
    }

    let mut sum_wi = f32x4_splat(0.0);
    let mut diff_l_uninit = [i32x4_splat(0); 32];
    let diff_l_ptr = diff_l_uninit.as_mut_ptr() as *mut v128;
    let mut k = 0usize;
    let mut ry = 0usize;
    if S == 1 {
        while ry < 16 {
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 4;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = v128_load(luma_ptr.add(ioff) as *const v128);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);
                k += 1;
            }
            ry += 2;
        }
    } else if S == 2 {
        while ry < 16 {
            let img_row = (cy + ry * 2) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 8;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = load_sum4_2x2_f32(luma_ptr.add(ioff), stride);
                let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);
                k += 1;
            }
            ry += 2;
        }
    } else {
        while ry < 16 {
            let img_row = (cy + ry * scale_px) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * cstep;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);
                k += 1;
            }
            ry += 2;
        }
    }

    let mu_i = hsum_f32x4(sum_wi) * inv_w_eff;
    let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(if S == 2 { 4.0 * dc } else { dc });
    let mut acc_sad_l = f32x4_splat(0.0);
    k = 0;
    ry = 0;
    while ry < 16 {
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
            let dl = f32x4_abs(f32x4_sub(*diff_l_ptr.add(k), v_dc));
            acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
            k += 1;
        }
        ry += 2;
    }

    let sad_l = hsum_f32x4(acc_sad_l) * inv_w_eff;
    let partial = sad_l + 1.85 * sad_c;
    if partial - eca_lb >= cutoff {
        return cutoff;
    }

    let mut w_eca = 0.0f32;
    if has_eca {
        let mut acc_eca = f32x4_splat(0.0);
        let mut ry_e = 0usize;
        if S == 1 {
            while ry_e < 16 {
                let img_row = (cy + ry_e) * stride + cx;
                let tpl_row = ry_e * 16;
                for vec_idx in 0..4usize {
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    let eca = v128_load(eca_res_ptr.add(img_row + vec_idx * 4) as *const v128);
                    acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca));
                }
                ry_e += 2;
            }
            w_eca = hsum_f32x4(acc_eca) * inv_w;
        } else if S == 2 {
            while ry_e < 16 {
                let img_row = (cy + ry_e * 2) * stride + cx;
                let tpl_row = ry_e * 16;
                for vec_idx in 0..4usize {
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    let eca4 = load_sum4_2x2_f32(eca_res_ptr.add(img_row + vec_idx * 8), stride);
                    acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca4));
                }
                ry_e += 2;
            }
            w_eca = hsum_f32x4(acc_eca) * (inv_w * 0.25);
        } else {
            while ry_e < 16 {
                let img_row = (cy + ry_e * scale_px) * stride + cx;
                let tpl_row = ry_e * 16;
                for vec_idx in 0..4usize {
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    let eca = load_pooled4_f32(eca_res_ptr.add(img_row + vec_idx * cstep), stride, scale_px);
                    acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca));
                }
                ry_e += 2;
            }
            w_eca = hsum_f32x4(acc_eca) * inv_w;
        }
        if !out_w_eca.is_null() {
            *out_w_eca = w_eca;
        }
    }
    partial - 0.16 * w_eca.min(0.85)
}

pub unsafe fn sad_patch_zm_half_precomp_cutoff_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    cutoff: f32,
    out_w_eca: *mut f32,
    scale_px: usize,
) -> f32 {
    if scale_px == 2 {
        sad_patch_zm_half_precomp_cutoff_impl::<2>(
            luma_ptr, rg_ptr, by_ptr, eca_res_ptr, stride, cx, cy, tpl_luma, tpl_rg, tpl_by,
            tpl_weight, inv_w, mu_t, cutoff, out_w_eca, 2,
        )
    } else if scale_px <= 1 {
        sad_patch_zm_half_precomp_cutoff_impl::<1>(
            luma_ptr, rg_ptr, by_ptr, eca_res_ptr, stride, cx, cy, tpl_luma, tpl_rg, tpl_by,
            tpl_weight, inv_w, mu_t, cutoff, out_w_eca, 1,
        )
    } else {
        sad_patch_zm_half_precomp_cutoff_impl::<0>(
            luma_ptr, rg_ptr, by_ptr, eca_res_ptr, stride, cx, cy, tpl_luma, tpl_rg, tpl_by,
            tpl_weight, inv_w, mu_t, cutoff, out_w_eca, scale_px,
        )
    }
}

pub unsafe fn sad_patch_zm_half_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    out_w_eca: *mut f32,
    scale_px: usize,
) -> f32 {
    let cstep = 4 * scale_px;
    let mut sum_w = f32x4_splat(0.0);
    let mut sum_wi = f32x4_splat(0.0);
    let mut sum_wt = f32x4_splat(0.0);
    let mut acc_sad_c = f32x4_splat(0.0);
    let mut acc_eca = f32x4_splat(0.0);
    let has_eca = !eca_res_ptr.is_null();

    let mut diff_l_uninit = [i32x4_splat(0); 32];
    let diff_l_ptr = diff_l_uninit.as_mut_ptr() as *mut v128;
    let mut k = 0usize;
    let mut ry = 0usize;

    if scale_px <= 1 {
        while ry < 16 {
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 4;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = v128_load(luma_ptr.add(ioff) as *const v128);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                sum_w = f32x4_add(sum_w, vw);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                sum_wt = f32x4_add(sum_wt, f32x4_mul(vw, tl));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);

                let irg = v128_load(rg_ptr.add(ioff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let iby = v128_load(by_ptr.add(ioff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);
                let diff_c = f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));

                if has_eca {
                    let eca = v128_load(eca_res_ptr.add(ioff) as *const v128);
                    acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca));
                }
                k += 1;
            }
            ry += 2;
        }
    } else {
        while ry < 16 {
            let img_row = (cy + ry * scale_px) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * cstep;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                sum_w = f32x4_add(sum_w, vw);
                sum_wi = f32x4_add(sum_wi, f32x4_mul(vw, il));
                sum_wt = f32x4_add(sum_wt, f32x4_mul(vw, tl));
                *diff_l_ptr.add(k) = f32x4_sub(il, tl);

                let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let iby = load_pooled4_f32(by_ptr.add(ioff), stride, scale_px);
                let tby = v128_load(tpl_by.add(toff) as *const v128);
                let diff_c = f32x4_add(f32x4_abs(f32x4_sub(irg, trg)), f32x4_abs(f32x4_sub(iby, tby)));
                acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_c));

                if has_eca {
                    let eca = load_pooled4_f32(eca_res_ptr.add(ioff), stride, scale_px);
                    acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca));
                }
                k += 1;
            }
            ry += 2;
        }
    }

    let inv_w = 1.0 / hsum_f32x4(sum_w).max(1e-6);
    let mu_i = hsum_f32x4(sum_wi) * inv_w;
    let mu_t = hsum_f32x4(sum_wt) * inv_w;
    let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(dc);

    let mut acc_sad_l = f32x4_splat(0.0);
    k = 0;
    ry = 0;
    while ry < 16 {
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
            let dl = f32x4_abs(f32x4_sub(*diff_l_ptr.add(k), v_dc));
            acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
            k += 1;
        }
        ry += 2;
    }

    let sad_l = hsum_f32x4(acc_sad_l) * inv_w;
    let sad_c = hsum_f32x4(acc_sad_c) * inv_w;
    let w_eca = hsum_f32x4(acc_eca) * inv_w;
    if !out_w_eca.is_null() {
        *out_w_eca = w_eca;
    }
    sad_l + 1.85 * sad_c - 0.16 * w_eca.min(0.85)
}

#[inline(always)]
unsafe fn pool5_row_f32(
    ptr: *const f32,
    stride: usize,
    scale_px: usize,
) -> [v128; 5] {
    if scale_px <= 1 {
        [
            v128_load(ptr as *const v128),
            v128_load(ptr.add(4) as *const v128),
            v128_load(ptr.add(8) as *const v128),
            v128_load(ptr.add(12) as *const v128),
            v128_load(ptr.add(16) as *const v128),
        ]
    } else {
        [
            load_pooled4_f32(ptr, stride, scale_px),
            load_pooled4_f32(ptr.add(8), stride, scale_px),
            load_pooled4_f32(ptr.add(16), stride, scale_px),
            load_pooled4_f32(ptr.add(24), stride, scale_px),
            load_pooled4_f32(ptr.add(32), stride, scale_px),
        ]
    }
}

#[inline(always)]
unsafe fn ring_shift_row_vecs(row: &[v128; 5], dx: i32) -> [v128; 4] {
    if dx < 0 {
        [row[0], row[1], row[2], row[3]]
    } else if dx > 0 {
        [row[1], row[2], row[3], row[4]]
    } else {
        [
            i32x4_shuffle::<2, 3, 4, 5>(row[0], row[1]),
            i32x4_shuffle::<2, 3, 4, 5>(row[1], row[2]),
            i32x4_shuffle::<2, 3, 4, 5>(row[2], row[3]),
            i32x4_shuffle::<2, 3, 4, 5>(row[3], row[4]),
        ]
    }
}

#[inline]
pub unsafe fn sad_half_weca_at_simd128(
    eca_res_ptr: *const f32,
    stride: usize,
    tx: usize,
    ty: usize,
    tpl_weight: *const f32,
    inv_w: f32,
    scale_px: usize,
) -> f32 {
    if eca_res_ptr.is_null() {
        return 0.0;
    }
    let rstep = 2 * scale_px;
    let mut acc_e = f32x4_splat(0.0);
    if scale_px <= 1 {
        for r in 0..8usize {
            let row_off = (ty + r * 2) * stride + tx;
            let tpl_row = r * 32;
            for vec_idx in 0..4usize {
                let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                let eca = v128_load(eca_res_ptr.add(row_off + vec_idx * 4) as *const v128);
                acc_e = f32x4_add(acc_e, f32x4_mul(vw, eca));
            }
        }
    } else if scale_px == 2 {
        for r in 0..8usize {
            let row_off = (ty + r * 4) * stride + tx;
            let tpl_row = r * 32;
            for vec_idx in 0..4usize {
                let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                let eca = load_sum4_2x2_f32(eca_res_ptr.add(row_off + vec_idx * 8), stride);
                acc_e = f32x4_add(acc_e, f32x4_mul(vw, eca));
            }
        }
        return hsum_f32x4(acc_e) * (inv_w * 0.25);
    } else {
        let cstep = 4 * scale_px;
        for r in 0..8usize {
            let row_off = (ty + r * rstep) * stride + tx;
            let tpl_row = r * 32;
            for vec_idx in 0..4usize {
                let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                let eca = load_pooled4_f32(
                    eca_res_ptr.add(row_off + vec_idx * cstep),
                    stride,
                    scale_px,
                );
                acc_e = f32x4_add(acc_e, f32x4_mul(vw, eca));
            }
        }
    }
    hsum_f32x4(acc_e) * inv_w
}

/// Evaluates all 8 `(-1..=1, -1..=1) \ {(0,0)}` neighbors at step `2 * scale_px` around `(cx0, cy0)`
/// sharing the overlapping `10 x 20` pooled window (`5.76x` fewer image loads + lazy exact ECA evaluation).
#[inline]
pub unsafe fn sad_patch_zm_half_ring3x3_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    let (inv_w, mu_t) = tpl_half_stats_simd128(tpl_luma, tpl_weight);
    sad_patch_zm_half_ring3x3_precomp_simd128(
        luma_ptr,
        rg_ptr,
        by_ptr,
        eca_res_ptr,
        stride,
        cx0,
        cy0,
        tpl_luma,
        tpl_rg,
        tpl_by,
        tpl_weight,
        inv_w,
        mu_t,
        scale_px,
        out_best_dx,
        out_best_dy,
        out_best_cost,
        out_best_weca,
    );
}

#[inline]
pub unsafe fn sad_patch_zm_half_ring3x3_precomp_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    scale_px: usize,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    if scale_px == 2 {
        sad_patch_zm_half_ring3x3_precomp_impl::<2>(
            luma_ptr, rg_ptr, by_ptr, eca_res_ptr, stride, cx0, cy0, tpl_luma, tpl_rg, tpl_by,
            tpl_weight, inv_w, mu_t, out_best_dx, out_best_dy, out_best_cost, out_best_weca,
        );
    } else if scale_px <= 1 {
        sad_patch_zm_half_ring3x3_precomp_impl::<1>(
            luma_ptr, rg_ptr, by_ptr, eca_res_ptr, stride, cx0, cy0, tpl_luma, tpl_rg, tpl_by,
            tpl_weight, inv_w, mu_t, out_best_dx, out_best_dy, out_best_cost, out_best_weca,
        );
    } else {
        sad_patch_zm_half_ring3x3_precomp_fallback_simd128(
            luma_ptr, rg_ptr, by_ptr, eca_res_ptr, stride, cx0, cy0, tpl_luma, tpl_rg, tpl_by,
            tpl_weight, inv_w, mu_t, scale_px, out_best_dx, out_best_dy, out_best_cost,
            out_best_weca,
        );
    }
}

#[inline(always)]
unsafe fn sad_patch_zm_half_ring3x3_precomp_impl<const SCALE_PX: usize>(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    const RING_OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let zero = f32x4_splat(0.0);
    let mut acc_sad_c = [zero; 8];
    let has_eca = !eca_res_ptr.is_null();
    let eca_lb = if has_eca { 0.136f32 } else { 0.0f32 };
    let cur_best = *out_best_cost;
    let cur_best_pad = cur_best + 1e-5;

    let st = 2 * SCALE_PX;
    let x_base = cx0 - st;
    let y_base = cy0 - st;

    // 1) Chroma pass FIRST across core rows [3,4,5,6] then outer rows with exact lower-bound pruning.
    // Core rows [3, 4] always have r5_alive == 0xFF (no mask branches); prune once after row 4.
    for r in [3usize, 4usize] {
        let row_off = (y_base + r * st) * stride + x_base;
        let irg_row = pool5_row_f32(rg_ptr.add(row_off), stride, SCALE_PX);
        let iby_row = pool5_row_f32(by_ptr.add(row_off), stride, SCALE_PX);
        let irg_mid = ring_shift_row_vecs(&irg_row, 0);
        let iby_mid = ring_shift_row_vecs(&iby_row, 0);

        let tpl_row_m1 = r * 32;
        for vec_idx in 0..4usize {
            let toff = tpl_row_m1 + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let trg = v128_load(tpl_rg.add(toff) as *const v128);
            let tby = v128_load(tpl_by.add(toff) as *const v128);

            let dc0 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
            );
            acc_sad_c[0] = f32x4_add(acc_sad_c[0], f32x4_mul(vw, dc0));

            let dc1 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
            );
            acc_sad_c[1] = f32x4_add(acc_sad_c[1], f32x4_mul(vw, dc1));

            let dc2 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
            );
            acc_sad_c[2] = f32x4_add(acc_sad_c[2], f32x4_mul(vw, dc2));
        }

        let tpl_row_0 = (r - 1) * 32;
        for vec_idx in 0..4usize {
            let toff = tpl_row_0 + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let trg = v128_load(tpl_rg.add(toff) as *const v128);
            let tby = v128_load(tpl_by.add(toff) as *const v128);

            let dc3 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
            );
            acc_sad_c[3] = f32x4_add(acc_sad_c[3], f32x4_mul(vw, dc3));

            let dc4 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
            );
            acc_sad_c[4] = f32x4_add(acc_sad_c[4], f32x4_mul(vw, dc4));
        }

        let tpl_row_p1 = (r - 2) * 32;
        for vec_idx in 0..4usize {
            let toff = tpl_row_p1 + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let trg = v128_load(tpl_rg.add(toff) as *const v128);
            let tby = v128_load(tpl_by.add(toff) as *const v128);

            let dc5 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
            );
            acc_sad_c[5] = f32x4_add(acc_sad_c[5], f32x4_mul(vw, dc5));

            let dc6 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
            );
            acc_sad_c[6] = f32x4_add(acc_sad_c[6], f32x4_mul(vw, dc6));

            let dc7 = f32x4_add(
                f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
            );
            acc_sad_c[7] = f32x4_add(acc_sad_c[7], f32x4_mul(vw, dc7));
        }
    }

    let mut r5_alive = 0u32;
    for c in 0..8usize {
        if 1.85 * (hsum_f32x4(acc_sad_c[c]) * inv_w) - eca_lb < cur_best_pad {
            r5_alive |= 1u32 << c;
        }
    }
    if r5_alive == 0 {
        return;
    }

    const TAIL_ROW_ORDER: [usize; 8] = [5, 6, 2, 7, 1, 8, 0, 9];
    let mut tail_single_start: usize = if (r5_alive & (r5_alive - 1)) == 0 {
        0
    } else {
        8
    };
    if tail_single_start == 8 {
        for t_idx in 0..8usize {
            let r = TAIL_ROW_ORDER[t_idx];
            let need_dy_m1 = r < 8 && (r5_alive & 0b0000_0111) != 0;
            let need_dy_0 = (1..=8).contains(&r) && (r5_alive & 0b0001_1000) != 0;
            let need_dy_p1 = r >= 2 && (r5_alive & 0b1110_0000) != 0;
            if !(need_dy_m1 || need_dy_0 || need_dy_p1) {
                continue;
            }
            let row_off = (y_base + r * st) * stride + x_base;
            let irg_row = pool5_row_f32(rg_ptr.add(row_off), stride, SCALE_PX);
            let iby_row = pool5_row_f32(by_ptr.add(row_off), stride, SCALE_PX);
            let need_mid = (need_dy_m1 && (r5_alive & 2) != 0) || (need_dy_p1 && (r5_alive & 64) != 0);
            let irg_mid = if need_mid { ring_shift_row_vecs(&irg_row, 0) } else { [zero; 4] };
            let iby_mid = if need_mid { ring_shift_row_vecs(&iby_row, 0) } else { [zero; 4] };

            if need_dy_m1 {
                let tpl_row = r * 32;
                let mask_m1 = r5_alive & 7;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);

                    if (mask_m1 & 1) != 0 {
                        let dc0 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                            f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                        );
                        acc_sad_c[0] = f32x4_add(acc_sad_c[0], f32x4_mul(vw, dc0));
                    }
                    if (mask_m1 & 2) != 0 {
                        let dc1 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                            f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
                        );
                        acc_sad_c[1] = f32x4_add(acc_sad_c[1], f32x4_mul(vw, dc1));
                    }
                    if (mask_m1 & 4) != 0 {
                        let dc2 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                            f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
                        );
                        acc_sad_c[2] = f32x4_add(acc_sad_c[2], f32x4_mul(vw, dc2));
                    }
                }
            }
            if need_dy_0 {
                let tpl_row = (r - 1) * 32;
                let mask_0 = r5_alive & 24;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);

                    if (mask_0 & 8) != 0 {
                        let dc3 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                            f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                        );
                        acc_sad_c[3] = f32x4_add(acc_sad_c[3], f32x4_mul(vw, dc3));
                    }
                    if (mask_0 & 16) != 0 {
                        let dc4 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                            f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
                        );
                        acc_sad_c[4] = f32x4_add(acc_sad_c[4], f32x4_mul(vw, dc4));
                    }
                }
            }
            if need_dy_p1 {
                let tpl_row = (r - 2) * 32;
                let mask_p1 = r5_alive & 224;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);

                    if (mask_p1 & 32) != 0 {
                        let dc5 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                            f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                        );
                        acc_sad_c[5] = f32x4_add(acc_sad_c[5], f32x4_mul(vw, dc5));
                    }
                    if (mask_p1 & 64) != 0 {
                        let dc6 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                            f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
                        );
                        acc_sad_c[6] = f32x4_add(acc_sad_c[6], f32x4_mul(vw, dc6));
                    }
                    if (mask_p1 & 128) != 0 {
                        let dc7 = f32x4_add(
                            f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                            f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
                        );
                        acc_sad_c[7] = f32x4_add(acc_sad_c[7], f32x4_mul(vw, dc7));
                    }
                }
            }
            if t_idx == 1 || t_idx == 3 || t_idx == 5 {
                let mut next_alive = 0u32;
                for c in 0..8usize {
                    if (r5_alive & (1u32 << c)) != 0
                        && 1.85 * (hsum_f32x4(acc_sad_c[c]) * inv_w) - eca_lb < cur_best_pad
                    {
                        next_alive |= 1u32 << c;
                    }
                }
                r5_alive = next_alive;
                if r5_alive == 0 {
                    return;
                }
                if (r5_alive & (r5_alive - 1)) == 0 {
                    tail_single_start = t_idx + 1;
                    break;
                }
            }
        }
    }
    if tail_single_start < 8 {
        let c = r5_alive.trailing_zeros() as usize;
        let (dx, dy) = RING_OFFSETS[c];
        let tx = (cx0 as isize + (dx as isize) * (st as isize)) as usize;
        let xstep = 4 * SCALE_PX;
        for t_idx in tail_single_start..8usize {
            let r = TAIL_ROW_ORDER[t_idx];
            let y_row_i = (r as isize) - ((dy as isize) + 1);
            if (0..8).contains(&y_row_i) {
                let img_row = (y_base + r * st) * stride + tx;
                let tpl_row = (y_row_i as usize) * 32;
                for vec_idx in 0..4usize {
                    let ioff = img_row + vec_idx * xstep;
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, SCALE_PX);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let iby = load_pooled4_f32(by_ptr.add(ioff), stride, SCALE_PX);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);
                    let dc = f32x4_add(
                        f32x4_abs(f32x4_sub(irg, trg)),
                        f32x4_abs(f32x4_sub(iby, tby)),
                    );
                    acc_sad_c[c] = f32x4_add(acc_sad_c[c], f32x4_mul(vw, dc));
                }
            }
            if (t_idx == 1 || t_idx == 3 || t_idx == 5)
                && 1.85 * (hsum_f32x4(acc_sad_c[c]) * inv_w) - eca_lb >= cur_best_pad
            {
                return;
            }
        }
    }

    let mut alive_mask = 0u32;
    let mut sad_c_arr = [0.0f32; 8];
    for c in 0..8usize {
        if (r5_alive & (1u32 << c)) == 0 {
            continue;
        }
        let sad_c = hsum_f32x4(acc_sad_c[c]) * inv_w;
        sad_c_arr[c] = sad_c;
        if 1.85 * sad_c - eca_lb < cur_best {
            alive_mask |= 1u32 << c;
        }
    }
    if alive_mask == 0 {
        return;
    }
    if (alive_mask & (alive_mask - 1)) == 0 {
        let c = alive_mask.trailing_zeros() as usize;
        let sad_c = sad_c_arr[c];
        let (dx, dy) = RING_OFFSETS[c];
        let tx = (cx0 as isize + (dx as isize) * (st as isize)) as usize;
        let ty = (cy0 as isize + (dy as isize) * (st as isize)) as usize;
        let mut il_single_uninit = [i32x4_splat(0); 32];
        let il_single = il_single_uninit.as_mut_ptr() as *mut v128;
        let mut sum_wi_1 = zero;
        if SCALE_PX <= 1 {
            let mut k = 0usize;
            for r in 0..8usize {
                let row0 = (ty + r * 2) * stride + tx;
                let tpl_row = r * 32;
                for vec_idx in 0..4usize {
                    let il = v128_load(luma_ptr.add(row0 + vec_idx * 4) as *const v128);
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    *il_single.add(k) = il;
                    sum_wi_1 = f32x4_add(sum_wi_1, f32x4_mul(vw, il));
                    k += 1;
                }
            }
            let mu_i = hsum_f32x4(sum_wi_1) * inv_w;
            let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
            let v_dc = f32x4_splat(dc);
            let mut acc_sad_l = zero;
            for r in [2usize, 3, 4, 5] {
                let tpl_row = r * 32;
                let k_r = r * 4;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let il = *il_single.add(k_r + vec_idx);
                    let tl = v128_load(tpl_luma.add(toff) as *const v128);
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                    acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
                }
            }
            if hsum_f32x4(acc_sad_l) * inv_w + 1.85 * sad_c - eca_lb >= *out_best_cost {
                return;
            }
            for r in [0usize, 1, 6, 7] {
                let tpl_row = r * 32;
                let k_r = r * 4;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let il = *il_single.add(k_r + vec_idx);
                    let tl = v128_load(tpl_luma.add(toff) as *const v128);
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                    acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
                }
            }
            let sad_l = hsum_f32x4(acc_sad_l) * inv_w;
            let partial = sad_l + 1.85 * sad_c;
            if partial - eca_lb < *out_best_cost {
                let w_eca = sad_half_weca_at_simd128(
                    eca_res_ptr, stride, tx, ty, tpl_weight, inv_w, SCALE_PX,
                );
                let score = partial - 0.16 * w_eca.min(0.85);
                if score < *out_best_cost {
                    *out_best_cost = score;
                    *out_best_dx = dx;
                    *out_best_dy = dy;
                    *out_best_weca = w_eca;
                }
            }
        } else {
            let mut k = 0usize;
            for r in 0..8usize {
                let row0 = (ty + r * 4) * stride + tx;
                let tpl_row = r * 32;
                for vec_idx in 0..4usize {
                    let il = load_sum4_2x2_f32(luma_ptr.add(row0 + vec_idx * 8), stride);
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    *il_single.add(k) = il;
                    sum_wi_1 = f32x4_add(sum_wi_1, f32x4_mul(vw, il));
                    k += 1;
                }
            }
            let inv_w4 = inv_w * 0.25;
            let v4 = f32x4_splat(4.0);
            let mu_i = hsum_f32x4(sum_wi_1) * inv_w4;
            let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
            let v_dc = f32x4_splat(4.0 * dc);
            let mut acc_sad_l = zero;
            for r in [2usize, 3, 4, 5] {
                let tpl_row = r * 32;
                let k_r = r * 4;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let il = *il_single.add(k_r + vec_idx);
                    let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                    acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
                }
            }
            if hsum_f32x4(acc_sad_l) * inv_w4 + 1.85 * sad_c - eca_lb >= *out_best_cost {
                return;
            }
            for r in [0usize, 1, 6, 7] {
                let tpl_row = r * 32;
                let k_r = r * 4;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let il = *il_single.add(k_r + vec_idx);
                    let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                    acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
                }
            }
            let sad_l = hsum_f32x4(acc_sad_l) * inv_w4;
            let partial = sad_l + 1.85 * sad_c;
            if partial - eca_lb < *out_best_cost {
                let w_eca =
                    sad_half_weca_at_simd128(eca_res_ptr, stride, tx, ty, tpl_weight, inv_w, SCALE_PX);
                let score = partial - 0.16 * w_eca.min(0.85);
                if score < *out_best_cost {
                    *out_best_cost = score;
                    *out_best_dx = dx;
                    *out_best_dy = dy;
                    *out_best_weca = w_eca;
                }
            }
        }
        return;
    }

    if SCALE_PX == 1 {
        sad_patch_zm_half_ring3x3_multi_impl::<1>(
            luma_ptr,
            eca_res_ptr,
            stride,
            cx0,
            cy0,
            tpl_luma,
            tpl_weight,
            inv_w,
            mu_t,
            1,
            alive_mask,
            &sad_c_arr,
            out_best_dx,
            out_best_dy,
            out_best_cost,
            out_best_weca,
        );
    } else if SCALE_PX == 2 {
        sad_patch_zm_half_ring3x3_multi_impl::<2>(
            luma_ptr,
            eca_res_ptr,
            stride,
            cx0,
            cy0,
            tpl_luma,
            tpl_weight,
            inv_w,
            mu_t,
            2,
            alive_mask,
            &sad_c_arr,
            out_best_dx,
            out_best_dy,
            out_best_cost,
            out_best_weca,
        );
    } else {
        sad_patch_zm_half_ring3x3_multi_impl::<0>(
            luma_ptr,
            eca_res_ptr,
            stride,
            cx0,
            cy0,
            tpl_luma,
            tpl_weight,
            inv_w,
            mu_t,
            SCALE_PX,
            alive_mask,
            &sad_c_arr,
            out_best_dx,
            out_best_dy,
            out_best_cost,
            out_best_weca,
        );
    }
}

#[inline(never)]
unsafe fn sad_patch_zm_half_ring3x3_precomp_fallback_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    scale_px: usize,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    const RING_OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let zero = f32x4_splat(0.0);
    let mut acc_sad_c = [zero; 8];
    let has_eca = !eca_res_ptr.is_null();
    let eca_lb = if has_eca { 0.136f32 } else { 0.0f32 };
    let cur_best = *out_best_cost;
    let cur_best_pad = cur_best + 1e-5;

    let st = 2 * scale_px;
    let x_base = cx0 - st;
    let y_base = cy0 - st;

    const ROW_ORDER: [usize; 10] = [3, 4, 5, 6, 2, 7, 1, 8, 0, 9];
    let mut r5_alive = 0xFFu32;
    for step in 0..10usize {
        let r = ROW_ORDER[step];
        let need_dy_m1 = r < 8 && (r5_alive & 0b0000_0111) != 0;
        let need_dy_0 = (1..=8).contains(&r) && (r5_alive & 0b0001_1000) != 0;
        let need_dy_p1 = r >= 2 && (r5_alive & 0b1110_0000) != 0;
        if !(need_dy_m1 || need_dy_0 || need_dy_p1) {
            continue;
        }
        let row_off = (y_base + r * st) * stride + x_base;
        let irg_row = pool5_row_f32(rg_ptr.add(row_off), stride, scale_px);
        let iby_row = pool5_row_f32(by_ptr.add(row_off), stride, scale_px);
        let need_mid = (need_dy_m1 && (r5_alive & 2) != 0) || (need_dy_p1 && (r5_alive & 64) != 0);
        let irg_mid = if need_mid { ring_shift_row_vecs(&irg_row, 0) } else { [zero; 4] };
        let iby_mid = if need_mid { ring_shift_row_vecs(&iby_row, 0) } else { [zero; 4] };

        if need_dy_m1 {
            let tpl_row = r * 32;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);

                let dc0 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                );
                acc_sad_c[0] = f32x4_add(acc_sad_c[0], f32x4_mul(vw, dc0));

                let dc1 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
                );
                acc_sad_c[1] = f32x4_add(acc_sad_c[1], f32x4_mul(vw, dc1));

                let dc2 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                    f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
                );
                acc_sad_c[2] = f32x4_add(acc_sad_c[2], f32x4_mul(vw, dc2));
            }
        }
        if need_dy_0 {
            let tpl_row = (r - 1) * 32;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);

                let dc3 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                );
                acc_sad_c[3] = f32x4_add(acc_sad_c[3], f32x4_mul(vw, dc3));

                let dc4 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                    f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
                );
                acc_sad_c[4] = f32x4_add(acc_sad_c[4], f32x4_mul(vw, dc4));
            }
        }
        if need_dy_p1 {
            let tpl_row = (r - 2) * 32;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);

                let dc5 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                );
                acc_sad_c[5] = f32x4_add(acc_sad_c[5], f32x4_mul(vw, dc5));

                let dc6 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
                );
                acc_sad_c[6] = f32x4_add(acc_sad_c[6], f32x4_mul(vw, dc6));

                let dc7 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_row[vec_idx + 1], trg)),
                    f32x4_abs(f32x4_sub(iby_row[vec_idx + 1], tby)),
                );
                acc_sad_c[7] = f32x4_add(acc_sad_c[7], f32x4_mul(vw, dc7));
            }
        }
        if step == 3 {
            r5_alive = 0;
            for c in 0..8usize {
                if 1.85 * (hsum_f32x4(acc_sad_c[c]) * inv_w) - eca_lb < cur_best_pad {
                    r5_alive |= 1u32 << c;
                }
            }
            if r5_alive == 0 {
                return;
            }
        }
    }

    let mut alive_mask = 0u32;
    let mut sad_c_arr = [0.0f32; 8];
    for c in 0..8usize {
        if (r5_alive & (1u32 << c)) == 0 {
            continue;
        }
        let sad_c = hsum_f32x4(acc_sad_c[c]) * inv_w;
        sad_c_arr[c] = sad_c;
        if 1.85 * sad_c - eca_lb < cur_best {
            alive_mask |= 1u32 << c;
        }
    }
    if alive_mask == 0 {
        return;
    }

    sad_patch_zm_half_ring3x3_multi_simd128(
        luma_ptr,
        eca_res_ptr,
        stride,
        cx0,
        cy0,
        tpl_luma,
        tpl_weight,
        inv_w,
        mu_t,
        scale_px,
        alive_mask,
        &sad_c_arr,
        out_best_dx,
        out_best_dy,
        out_best_cost,
        out_best_weca,
    );
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn sad_patch_zm_half_ring3x3_multi_simd128(
    luma_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    scale_px: usize,
    alive_mask: u32,
    sad_c_arr: &[f32; 8],
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    sad_patch_zm_half_ring3x3_multi_impl::<0>(
        luma_ptr,
        eca_res_ptr,
        stride,
        cx0,
        cy0,
        tpl_luma,
        tpl_weight,
        inv_w,
        mu_t,
        scale_px,
        alive_mask,
        sad_c_arr,
        out_best_dx,
        out_best_dy,
        out_best_cost,
        out_best_weca,
    );
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn sad_patch_zm_half_ring3x3_multi_impl<const SCALE_PX: usize>(
    luma_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    scale_px_dyn: usize,
    alive_mask: u32,
    sad_c_arr: &[f32; 8],
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    const RING_OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    let scale_px = if SCALE_PX == 0 { scale_px_dyn } else { SCALE_PX };
    let zero = f32x4_splat(0.0);
    let has_eca = !eca_res_ptr.is_null();
    let eca_lb = if has_eca { 0.136f32 } else { 0.0f32 };
    let st = 2 * scale_px;
    let x_base = cx0 - st;
    let y_base = cy0 - st;

    let need_m1 = (alive_mask & 0b0000_0111) != 0;
    let need_0 = (alive_mask & 0b0001_1000) != 0;
    let need_p1 = (alive_mask & 0b1110_0000) != 0;
    let mut sum_wi = [zero; 8];
    let mut il_uninit = [i32x4_splat(0); 50];
    let il_ptr = il_uninit.as_mut_ptr() as *mut v128;

    // 2) Luma pass ONLY for surviving rows/candidates.
    for r in 0..10usize {
        let use_m1 = r < 8 && need_m1;
        let use_0 = (1..=8).contains(&r) && need_0;
        let use_p1 = r >= 2 && need_p1;
        if !(use_m1 || use_0 || use_p1) {
            continue;
        }
        let row_off = (y_base + r * st) * stride + x_base;
        let il_row = pool5_row_f32(luma_ptr.add(row_off), stride, scale_px);
        for k in 0..5usize {
            *il_ptr.add(r * 5 + k) = il_row[k];
        }
        let need_mid = (use_m1 && (alive_mask & 2) != 0) || (use_p1 && (alive_mask & 64) != 0);
        let il_mid = if need_mid { ring_shift_row_vecs(&il_row, 0) } else { [zero; 4] };

        if use_m1 {
            let tpl_row = r * 32;
            let am0 = alive_mask & 7;
            if am0 == 7 {
                for vec_idx in 0..4usize {
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    sum_wi[0] = f32x4_add(sum_wi[0], f32x4_mul(vw, il_row[vec_idx]));
                    sum_wi[1] = f32x4_add(sum_wi[1], f32x4_mul(vw, il_mid[vec_idx]));
                    sum_wi[2] = f32x4_add(sum_wi[2], f32x4_mul(vw, il_row[vec_idx + 1]));
                }
            } else {
                for vec_idx in 0..4usize {
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    if (am0 & 1) != 0 {
                        sum_wi[0] = f32x4_add(sum_wi[0], f32x4_mul(vw, il_row[vec_idx]));
                    }
                    if (am0 & 2) != 0 {
                        sum_wi[1] = f32x4_add(sum_wi[1], f32x4_mul(vw, il_mid[vec_idx]));
                    }
                    if (am0 & 4) != 0 {
                        sum_wi[2] = f32x4_add(sum_wi[2], f32x4_mul(vw, il_row[vec_idx + 1]));
                    }
                }
            }
        }
        if use_0 {
            let tpl_row = (r - 1) * 32;
            let am1 = alive_mask & 24;
            for vec_idx in 0..4usize {
                let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                if (am1 & 8) != 0 {
                    sum_wi[3] = f32x4_add(sum_wi[3], f32x4_mul(vw, il_row[vec_idx]));
                }
                if (am1 & 16) != 0 {
                    sum_wi[4] = f32x4_add(sum_wi[4], f32x4_mul(vw, il_row[vec_idx + 1]));
                }
            }
        }
        if use_p1 {
            let tpl_row = (r - 2) * 32;
            let am2 = alive_mask & 224;
            if am2 == 224 {
                for vec_idx in 0..4usize {
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    sum_wi[5] = f32x4_add(sum_wi[5], f32x4_mul(vw, il_row[vec_idx]));
                    sum_wi[6] = f32x4_add(sum_wi[6], f32x4_mul(vw, il_mid[vec_idx]));
                    sum_wi[7] = f32x4_add(sum_wi[7], f32x4_mul(vw, il_row[vec_idx + 1]));
                }
            } else {
                for vec_idx in 0..4usize {
                    let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                    if (am2 & 32) != 0 {
                        sum_wi[5] = f32x4_add(sum_wi[5], f32x4_mul(vw, il_row[vec_idx]));
                    }
                    if (am2 & 64) != 0 {
                        sum_wi[6] = f32x4_add(sum_wi[6], f32x4_mul(vw, il_mid[vec_idx]));
                    }
                    if (am2 & 128) != 0 {
                        sum_wi[7] = f32x4_add(sum_wi[7], f32x4_mul(vw, il_row[vec_idx + 1]));
                    }
                }
            }
        }
    }

    let mut order = [0u8; 8];
    let mut n_ord = 0usize;
    let init_best = *out_best_cost;
    for c in 0..8usize {
        if (alive_mask & (1u32 << c)) != 0 && 1.85 * sad_c_arr[c] - eca_lb < init_best {
            let key = sad_c_arr[c];
            let mut pos = n_ord;
            while pos > 0 {
                let prev = order[pos - 1] as usize;
                if sad_c_arr[prev] <= key {
                    break;
                }
                order[pos] = order[pos - 1];
                pos -= 1;
            }
            order[pos] = c as u8;
            n_ord += 1;
        }
    }

    for oi in 0..n_ord {
        let c = order[oi] as usize;
        let sad_c = sad_c_arr[c];
        if 1.85 * sad_c - eca_lb >= *out_best_cost {
            break;
        }
        let mu_i = hsum_f32x4(sum_wi[c]) * inv_w;
        let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
        let v_dc = f32x4_splat(dc);
        let (dx, dy) = RING_OFFSETS[c];
        let r_off = (dy + 1) as usize;
        let mut acc_sad_l = zero;
        macro_rules! accum_luma_row {
            ($row_i:expr) => {{
                let row_i: usize = $row_i;
                let r = row_i + r_off;
                let il_p = il_ptr.add(r * 5);
                let tpl_row = row_i * 32;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let il = if dx < 0 {
                        *il_p.add(vec_idx)
                    } else if dx > 0 {
                        *il_p.add(vec_idx + 1)
                    } else {
                        i32x4_shuffle::<2, 3, 4, 5>(*il_p.add(vec_idx), *il_p.add(vec_idx + 1))
                    };
                    let tl = v128_load(tpl_luma.add(toff) as *const v128);
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                    acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
                }
            }};
        }
        for row_i in [2usize, 3, 4, 5] {
            accum_luma_row!(row_i);
        }
        if hsum_f32x4(acc_sad_l) * inv_w + 1.85 * sad_c - eca_lb >= *out_best_cost {
            continue;
        }
        for row_i in [0usize, 1, 6, 7] {
            accum_luma_row!(row_i);
        }
        let sad_l = hsum_f32x4(acc_sad_l) * inv_w;
        let partial = sad_l + 1.85 * sad_c;
        if partial - eca_lb >= *out_best_cost {
            continue;
        }
        let tx = (cx0 as isize + (dx as isize) * (st as isize)) as usize;
        let ty = (cy0 as isize + (dy as isize) * (st as isize)) as usize;
        let w_eca = sad_half_weca_at_simd128(
            eca_res_ptr,
            stride,
            tx,
            ty,
            tpl_weight,
            inv_w,
            scale_px,
        );
        let score = partial - 0.16 * w_eca.min(0.85);
        if score < *out_best_cost {
            *out_best_cost = score;
            *out_best_dx = dx;
            *out_best_dy = dy;
            *out_best_weca = w_eca;
        }
    }
}

#[inline(always)]
unsafe fn load5_sd_row_f32(ptr: *const f32) -> [v128; 5] {
    [
        v128_load(ptr as *const v128),
        v128_load(ptr.add(4) as *const v128),
        v128_load(ptr.add(8) as *const v128),
        v128_load(ptr.add(12) as *const v128),
        v128_load64_zero(ptr.add(16) as *const u64),
    ]
}

#[inline(always)]
unsafe fn shift5_row_vecs(row: &[v128; 5], shift: i32) -> [v128; 4] {
    if shift <= 0 {
        [row[0], row[1], row[2], row[3]]
    } else if shift == 1 {
        [
            i32x4_shuffle::<1, 2, 3, 4>(row[0], row[1]),
            i32x4_shuffle::<1, 2, 3, 4>(row[1], row[2]),
            i32x4_shuffle::<1, 2, 3, 4>(row[2], row[3]),
            i32x4_shuffle::<1, 2, 3, 4>(row[3], row[4]),
        ]
    } else {
        [
            i32x4_shuffle::<2, 3, 4, 5>(row[0], row[1]),
            i32x4_shuffle::<2, 3, 4, 5>(row[1], row[2]),
            i32x4_shuffle::<2, 3, 4, 5>(row[2], row[3]),
            i32x4_shuffle::<2, 3, 4, 5>(row[3], row[4]),
        ]
    }
}

#[inline(always)]
unsafe fn load9_hd_row_f32(ptr: *const f32) -> [v128; 9] {
    [
        v128_load(ptr as *const v128),
        v128_load(ptr.add(4) as *const v128),
        v128_load(ptr.add(8) as *const v128),
        v128_load(ptr.add(12) as *const v128),
        v128_load(ptr.add(16) as *const v128),
        v128_load(ptr.add(20) as *const v128),
        v128_load(ptr.add(24) as *const v128),
        v128_load(ptr.add(28) as *const v128),
        v128_load64_zero(ptr.add(32) as *const u64),
    ]
}

#[inline(always)]
unsafe fn fold_add_load9_2x2_pe_po<const NEED_POS: bool>(
    prev: &mut [v128; 9],
    ptr: *const f32,
    pe: &mut [v128; 4],
    pe_pos: &mut [v128; 4],
    po: &mut [v128; 4],
    out_pe_last: &mut f32,
) {
    let c0 = v128_load(ptr as *const v128);
    let s0 = f32x4_add(prev[0], c0);
    prev[0] = c0;
    let c1 = v128_load(ptr.add(4) as *const v128);
    let s1 = f32x4_add(prev[1], c1);
    prev[1] = c1;
    let e0 = i32x4_shuffle::<0, 2, 4, 6>(s0, s1);
    let o0 = i32x4_shuffle::<1, 3, 5, 7>(s0, s1);
    pe[0] = f32x4_add(e0, o0);

    let c2 = v128_load(ptr.add(8) as *const v128);
    let s2 = f32x4_add(prev[2], c2);
    prev[2] = c2;
    let c3 = v128_load(ptr.add(12) as *const v128);
    let s3 = f32x4_add(prev[3], c3);
    prev[3] = c3;
    let e1 = i32x4_shuffle::<0, 2, 4, 6>(s2, s3);
    let o1 = i32x4_shuffle::<1, 3, 5, 7>(s2, s3);
    pe[1] = f32x4_add(e1, o1);
    po[0] = f32x4_add(o0, i32x4_shuffle::<1, 2, 3, 4>(e0, e1));
    if NEED_POS {
        pe_pos[0] = i32x4_shuffle::<1, 2, 3, 4>(pe[0], pe[1]);
    }

    let c4 = v128_load(ptr.add(16) as *const v128);
    let s4 = f32x4_add(prev[4], c4);
    prev[4] = c4;
    let c5 = v128_load(ptr.add(20) as *const v128);
    let s5 = f32x4_add(prev[5], c5);
    prev[5] = c5;
    let e2 = i32x4_shuffle::<0, 2, 4, 6>(s4, s5);
    let o2 = i32x4_shuffle::<1, 3, 5, 7>(s4, s5);
    pe[2] = f32x4_add(e2, o2);
    po[1] = f32x4_add(o1, i32x4_shuffle::<1, 2, 3, 4>(e1, e2));
    if NEED_POS {
        pe_pos[1] = i32x4_shuffle::<1, 2, 3, 4>(pe[1], pe[2]);
    }

    let c6 = v128_load(ptr.add(24) as *const v128);
    let s6 = f32x4_add(prev[6], c6);
    prev[6] = c6;
    let c7 = v128_load(ptr.add(28) as *const v128);
    let s7 = f32x4_add(prev[7], c7);
    prev[7] = c7;
    let e3 = i32x4_shuffle::<0, 2, 4, 6>(s6, s7);
    let o3 = i32x4_shuffle::<1, 3, 5, 7>(s6, s7);
    pe[3] = f32x4_add(e3, o3);
    po[2] = f32x4_add(o2, i32x4_shuffle::<1, 2, 3, 4>(e2, e3));
    if NEED_POS {
        pe_pos[2] = i32x4_shuffle::<1, 2, 3, 4>(pe[2], pe[3]);
    }

    let c8 = v128_load64_zero(ptr.add(32) as *const u64);
    let s8 = f32x4_add(prev[8], c8);
    prev[8] = c8;
    po[3] = f32x4_add(o3, i32x4_shuffle::<1, 2, 3, 4>(e3, s8));
    if NEED_POS {
        let pe_last = f32x4_extract_lane::<0>(s8) + f32x4_extract_lane::<1>(s8);
        *out_pe_last = pe_last;
        pe_pos[3] = i32x4_shuffle::<1, 2, 3, 4>(pe[3], f32x4_splat(pe_last));
    }
}

#[inline(always)]
unsafe fn fold_add_load9_2x2_pe_only<const NEED_POS: bool>(
    prev: &mut [v128; 9],
    ptr: *const f32,
    pe: &mut [v128; 4],
    pe_pos: &mut [v128; 4],
    out_pe_last: &mut f32,
) {
    let c0 = v128_load(ptr as *const v128);
    let s0 = f32x4_add(prev[0], c0);
    prev[0] = c0;
    let c1 = v128_load(ptr.add(4) as *const v128);
    let s1 = f32x4_add(prev[1], c1);
    prev[1] = c1;
    pe[0] = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s0, s1),
        i32x4_shuffle::<1, 3, 5, 7>(s0, s1),
    );

    let c2 = v128_load(ptr.add(8) as *const v128);
    let s2 = f32x4_add(prev[2], c2);
    prev[2] = c2;
    let c3 = v128_load(ptr.add(12) as *const v128);
    let s3 = f32x4_add(prev[3], c3);
    prev[3] = c3;
    pe[1] = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s2, s3),
        i32x4_shuffle::<1, 3, 5, 7>(s2, s3),
    );
    if NEED_POS {
        pe_pos[0] = i32x4_shuffle::<1, 2, 3, 4>(pe[0], pe[1]);
    }

    let c4 = v128_load(ptr.add(16) as *const v128);
    let s4 = f32x4_add(prev[4], c4);
    prev[4] = c4;
    let c5 = v128_load(ptr.add(20) as *const v128);
    let s5 = f32x4_add(prev[5], c5);
    prev[5] = c5;
    pe[2] = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s4, s5),
        i32x4_shuffle::<1, 3, 5, 7>(s4, s5),
    );
    if NEED_POS {
        pe_pos[1] = i32x4_shuffle::<1, 2, 3, 4>(pe[1], pe[2]);
    }

    let c6 = v128_load(ptr.add(24) as *const v128);
    let s6 = f32x4_add(prev[6], c6);
    prev[6] = c6;
    let c7 = v128_load(ptr.add(28) as *const v128);
    let s7 = f32x4_add(prev[7], c7);
    prev[7] = c7;
    pe[3] = f32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s6, s7),
        i32x4_shuffle::<1, 3, 5, 7>(s6, s7),
    );
    if NEED_POS {
        pe_pos[2] = i32x4_shuffle::<1, 2, 3, 4>(pe[2], pe[3]);
    }

    let c8 = v128_load64_zero(ptr.add(32) as *const u64);
    let s8 = f32x4_add(prev[8], c8);
    prev[8] = c8;
    if NEED_POS {
        let pe_last = f32x4_extract_lane::<0>(s8) + f32x4_extract_lane::<1>(s8);
        *out_pe_last = pe_last;
        pe_pos[3] = i32x4_shuffle::<1, 2, 3, 4>(pe[3], f32x4_splat(pe_last));
    }
}

#[inline(never)]
unsafe fn sad_patch_zm_half_st1_ring_sd_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    let mut sum_wt = f32x4_splat(0.0);
    for row_i in 0..8usize {
        let tpl_row = row_i * 32;
        for vec_idx in 0..4usize {
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            sum_wt = f32x4_add(sum_wt, f32x4_mul(vw, tl));
        }
    }
    let mu_t = hsum_f32x4(sum_wt) * inv_w;
    const RING_OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let zero = f32x4_splat(0.0);
    let mut sum_wi = [zero; 8];
    let mut acc_sad_c = [zero; 8];
    let has_eca = !eca_res_ptr.is_null();
    let eca_lb = if has_eca { 0.136f32 } else { 0.0f32 };

    let x_base = cx0 - 1;
    let y_base = cy0 - 1;

    let mut il_uninit = [i32x4_splat(0); 85];
    let il_ptr = il_uninit.as_mut_ptr() as *mut v128;

    for s_row in 0..17usize {
        let row_off = (y_base + s_row) * stride + x_base;
        let il_row = load5_sd_row_f32(luma_ptr.add(row_off));
        for k in 0..5usize {
            *il_ptr.add(s_row * 5 + k) = il_row[k];
        }
        let irg_row = load5_sd_row_f32(rg_ptr.add(row_off));
        let iby_row = load5_sd_row_f32(by_ptr.add(row_off));

        let il_pos = shift5_row_vecs(&il_row, 2);
        let irg_pos = shift5_row_vecs(&irg_row, 2);
        let iby_pos = shift5_row_vecs(&iby_row, 2);

        if (s_row & 1) != 0 {
            let tpl_row = ((s_row - 1) >> 1) * 32;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let trg = v128_load(tpl_rg.add(toff) as *const v128);
                let tby = v128_load(tpl_by.add(toff) as *const v128);

                let dc3 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                );
                sum_wi[3] = f32x4_add(sum_wi[3], f32x4_mul(vw, il_row[vec_idx]));
                acc_sad_c[3] = f32x4_add(acc_sad_c[3], f32x4_mul(vw, dc3));

                let dc4 = f32x4_add(
                    f32x4_abs(f32x4_sub(irg_pos[vec_idx], trg)),
                    f32x4_abs(f32x4_sub(iby_pos[vec_idx], tby)),
                );
                sum_wi[4] = f32x4_add(sum_wi[4], f32x4_mul(vw, il_pos[vec_idx]));
                acc_sad_c[4] = f32x4_add(acc_sad_c[4], f32x4_mul(vw, dc4));
            }
        } else {
            let il_mid = shift5_row_vecs(&il_row, 1);
            let irg_mid = shift5_row_vecs(&irg_row, 1);
            let iby_mid = shift5_row_vecs(&iby_row, 1);

            let r_even = s_row >> 1;
            if r_even < 8 {
                let tpl_row = r_even * 32;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);

                    let dc0 = f32x4_add(
                        f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                        f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                    );
                    sum_wi[0] = f32x4_add(sum_wi[0], f32x4_mul(vw, il_row[vec_idx]));
                    acc_sad_c[0] = f32x4_add(acc_sad_c[0], f32x4_mul(vw, dc0));

                    let dc1 = f32x4_add(
                        f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                        f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
                    );
                    sum_wi[1] = f32x4_add(sum_wi[1], f32x4_mul(vw, il_mid[vec_idx]));
                    acc_sad_c[1] = f32x4_add(acc_sad_c[1], f32x4_mul(vw, dc1));

                    let dc2 = f32x4_add(
                        f32x4_abs(f32x4_sub(irg_pos[vec_idx], trg)),
                        f32x4_abs(f32x4_sub(iby_pos[vec_idx], tby)),
                    );
                    sum_wi[2] = f32x4_add(sum_wi[2], f32x4_mul(vw, il_pos[vec_idx]));
                    acc_sad_c[2] = f32x4_add(acc_sad_c[2], f32x4_mul(vw, dc2));
                }
            }
            if r_even >= 1 {
                let tpl_row = (r_even - 1) * 32;
                for vec_idx in 0..4usize {
                    let toff = tpl_row + vec_idx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let trg = v128_load(tpl_rg.add(toff) as *const v128);
                    let tby = v128_load(tpl_by.add(toff) as *const v128);

                    let dc5 = f32x4_add(
                        f32x4_abs(f32x4_sub(irg_row[vec_idx], trg)),
                        f32x4_abs(f32x4_sub(iby_row[vec_idx], tby)),
                    );
                    sum_wi[5] = f32x4_add(sum_wi[5], f32x4_mul(vw, il_row[vec_idx]));
                    acc_sad_c[5] = f32x4_add(acc_sad_c[5], f32x4_mul(vw, dc5));

                    let dc6 = f32x4_add(
                        f32x4_abs(f32x4_sub(irg_mid[vec_idx], trg)),
                        f32x4_abs(f32x4_sub(iby_mid[vec_idx], tby)),
                    );
                    sum_wi[6] = f32x4_add(sum_wi[6], f32x4_mul(vw, il_mid[vec_idx]));
                    acc_sad_c[6] = f32x4_add(acc_sad_c[6], f32x4_mul(vw, dc6));

                    let dc7 = f32x4_add(
                        f32x4_abs(f32x4_sub(irg_pos[vec_idx], trg)),
                        f32x4_abs(f32x4_sub(iby_pos[vec_idx], tby)),
                    );
                    sum_wi[7] = f32x4_add(sum_wi[7], f32x4_mul(vw, il_pos[vec_idx]));
                    acc_sad_c[7] = f32x4_add(acc_sad_c[7], f32x4_mul(vw, dc7));
                }
            }
        }
    }

    for c in 0..8usize {
        let sad_c = hsum_f32x4(acc_sad_c[c]) * inv_w;
        if 1.85 * sad_c - eca_lb >= *out_best_cost {
            continue;
        }
        let mu_i = hsum_f32x4(sum_wi[c]) * inv_w;
        let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
        let v_dc = f32x4_splat(dc);
        let (dx, dy) = RING_OFFSETS[c];
        let s_base = (dy + 1) as usize;
        let mut acc_sad_l = zero;
        for r in 0..8usize {
            let il_p = il_ptr.add((s_base + 2 * r) * 5);
            let tpl_row = r * 32;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let il = if dx < 0 {
                    *il_p.add(vec_idx)
                } else if dx == 0 {
                    i32x4_shuffle::<1, 2, 3, 4>(*il_p.add(vec_idx), *il_p.add(vec_idx + 1))
                } else {
                    i32x4_shuffle::<2, 3, 4, 5>(*il_p.add(vec_idx), *il_p.add(vec_idx + 1))
                };
                let tl = v128_load(tpl_luma.add(toff) as *const v128);
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
            }
        }
        let sad_l = hsum_f32x4(acc_sad_l) * inv_w;
        let partial = sad_l + 1.85 * sad_c;
        if partial - eca_lb >= *out_best_cost {
            continue;
        }
        let tx = (cx0 as isize + (dx as isize)) as usize;
        let ty = (cy0 as isize + (dy as isize)) as usize;
        let w_eca = sad_half_weca_at_simd128(eca_res_ptr, stride, tx, ty, tpl_weight, inv_w, 1);
        let score = partial - 0.16 * w_eca.min(0.85);
        if score < *out_best_cost {
            *out_best_cost = score;
            *out_best_dx = dx;
            *out_best_dy = dy;
            *out_best_weca = w_eca;
        }
    }
}

#[inline(never)]
unsafe fn sad_patch_zm_half_st1_ring_hd_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    const RING_OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let zero = f32x4_splat(0.0);
    let v4 = f32x4_splat(4.0);
    let inv_w4 = 0.25 * inv_w;
    let mut acc_sad_c = [zero; 8];
    let has_eca = !eca_res_ptr.is_null();
    let eca_lb = if has_eca { 0.136f32 } else { 0.0f32 };
    let cur_best = *out_best_cost;
    let cur_best_pad = cur_best + 1e-5;

    let x_base = cx0 - 1;
    let mut r4_alive;

    // 1) Chroma R-G + B-Y FIRST across core strips [2,3,4,5] then outer strips with midpoint prune.
    // Core strips [2, 3] always have r4_alive == 0xFF (no mask branches); prune once after strip 3.
    for r in [2usize, 3usize] {
        let row_a = (cy0 + r * 4 - 1) * stride + x_base;
        let tpl_row = r * 32;
        let vw = [
            v128_load(tpl_weight.add(tpl_row) as *const v128),
            v128_load(tpl_weight.add(tpl_row + 4) as *const v128),
            v128_load(tpl_weight.add(tpl_row + 8) as *const v128),
            v128_load(tpl_weight.add(tpl_row + 12) as *const v128),
        ];
        let trg = [
            f32x4_mul(v128_load(tpl_rg.add(tpl_row) as *const v128), v4),
            f32x4_mul(v128_load(tpl_rg.add(tpl_row + 4) as *const v128), v4),
            f32x4_mul(v128_load(tpl_rg.add(tpl_row + 8) as *const v128), v4),
            f32x4_mul(v128_load(tpl_rg.add(tpl_row + 12) as *const v128), v4),
        ];
        let tby = [
            f32x4_mul(v128_load(tpl_by.add(tpl_row) as *const v128), v4),
            f32x4_mul(v128_load(tpl_by.add(tpl_row + 4) as *const v128), v4),
            f32x4_mul(v128_load(tpl_by.add(tpl_row + 8) as *const v128), v4),
            f32x4_mul(v128_load(tpl_by.add(tpl_row + 12) as *const v128), v4),
        ];

        let mut prev_rg = load9_hd_row_f32(rg_ptr.add(row_a));
        let mut prev_by = load9_hd_row_f32(by_ptr.add(row_a));
        let mut pe_rg = [zero; 4];
        let mut pos_rg = [zero; 4];
        let mut po_rg = [zero; 4];
        let mut pe_by = [zero; 4];
        let mut pos_by = [zero; 4];
        let mut po_by = [zero; 4];
        let mut dummy_last = 0.0f32;

        // s = 0 (ab -> cands 0, 1, 2)
        fold_add_load9_2x2_pe_po::<true>(
            &mut prev_rg,
            rg_ptr.add(row_a + stride),
            &mut pe_rg,
            &mut pos_rg,
            &mut po_rg,
            &mut dummy_last,
        );
        fold_add_load9_2x2_pe_po::<true>(
            &mut prev_by,
            by_ptr.add(row_a + stride),
            &mut pe_by,
            &mut pos_by,
            &mut po_by,
            &mut dummy_last,
        );
        for j in 0..4usize {
            let d0 = f32x4_add(
                f32x4_abs(f32x4_sub(pe_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(pe_by[j], tby[j])),
            );
            acc_sad_c[0] = f32x4_add(acc_sad_c[0], f32x4_mul(vw[j], d0));
            let d1 = f32x4_add(
                f32x4_abs(f32x4_sub(po_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(po_by[j], tby[j])),
            );
            acc_sad_c[1] = f32x4_add(acc_sad_c[1], f32x4_mul(vw[j], d1));
            let d2 = f32x4_add(
                f32x4_abs(f32x4_sub(pos_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(pos_by[j], tby[j])),
            );
            acc_sad_c[2] = f32x4_add(acc_sad_c[2], f32x4_mul(vw[j], d2));
        }

        // s = 1 (bc -> cands 3, 4)
        fold_add_load9_2x2_pe_only::<true>(
            &mut prev_rg,
            rg_ptr.add(row_a + 2 * stride),
            &mut pe_rg,
            &mut pos_rg,
            &mut dummy_last,
        );
        fold_add_load9_2x2_pe_only::<true>(
            &mut prev_by,
            by_ptr.add(row_a + 2 * stride),
            &mut pe_by,
            &mut pos_by,
            &mut dummy_last,
        );
        for j in 0..4usize {
            let d3 = f32x4_add(
                f32x4_abs(f32x4_sub(pe_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(pe_by[j], tby[j])),
            );
            acc_sad_c[3] = f32x4_add(acc_sad_c[3], f32x4_mul(vw[j], d3));
            let d4 = f32x4_add(
                f32x4_abs(f32x4_sub(pos_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(pos_by[j], tby[j])),
            );
            acc_sad_c[4] = f32x4_add(acc_sad_c[4], f32x4_mul(vw[j], d4));
        }

        // s = 2 (cd -> cands 5, 6, 7)
        fold_add_load9_2x2_pe_po::<true>(
            &mut prev_rg,
            rg_ptr.add(row_a + 3 * stride),
            &mut pe_rg,
            &mut pos_rg,
            &mut po_rg,
            &mut dummy_last,
        );
        fold_add_load9_2x2_pe_po::<true>(
            &mut prev_by,
            by_ptr.add(row_a + 3 * stride),
            &mut pe_by,
            &mut pos_by,
            &mut po_by,
            &mut dummy_last,
        );
        for j in 0..4usize {
            let d5 = f32x4_add(
                f32x4_abs(f32x4_sub(pe_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(pe_by[j], tby[j])),
            );
            acc_sad_c[5] = f32x4_add(acc_sad_c[5], f32x4_mul(vw[j], d5));
            let d6 = f32x4_add(
                f32x4_abs(f32x4_sub(po_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(po_by[j], tby[j])),
            );
            acc_sad_c[6] = f32x4_add(acc_sad_c[6], f32x4_mul(vw[j], d6));
            let d7 = f32x4_add(
                f32x4_abs(f32x4_sub(pos_rg[j], trg[j])),
                f32x4_abs(f32x4_sub(pos_by[j], tby[j])),
            );
            acc_sad_c[7] = f32x4_add(acc_sad_c[7], f32x4_mul(vw[j], d7));
        }
    }

    {
        let mut next_alive = 0u32;
        for c in 0..8usize {
            if 1.85 * (hsum_f32x4(acc_sad_c[c]) * inv_w4) - eca_lb < cur_best_pad {
                next_alive |= 1u32 << c;
            }
        }
        r4_alive = next_alive;
        if r4_alive == 0 {
            return;
        }
    }

    const TAIL_STRIP_ORDER: [usize; 6] = [4, 5, 1, 6, 0, 7];
    let mut tail_single_start: usize = if (r4_alive & (r4_alive - 1)) == 0 {
        0
    } else {
        6
    };
    if tail_single_start == 6 {
        for t_idx in 0..6usize {
            let r = TAIL_STRIP_ORDER[t_idx];
            let need_s0 = (r4_alive & 0b0000_0111) != 0;
            let need_s1 = (r4_alive & 0b0001_1000) != 0;
            let need_s2 = (r4_alive & 0b1110_0000) != 0;
            let row_a = (cy0 + r * 4 - 1) * stride + x_base;
            let tpl_row = r * 32;
            let vw = [
                v128_load(tpl_weight.add(tpl_row) as *const v128),
                v128_load(tpl_weight.add(tpl_row + 4) as *const v128),
                v128_load(tpl_weight.add(tpl_row + 8) as *const v128),
                v128_load(tpl_weight.add(tpl_row + 12) as *const v128),
            ];
            let trg = [
                f32x4_mul(v128_load(tpl_rg.add(tpl_row) as *const v128), v4),
                f32x4_mul(v128_load(tpl_rg.add(tpl_row + 4) as *const v128), v4),
                f32x4_mul(v128_load(tpl_rg.add(tpl_row + 8) as *const v128), v4),
                f32x4_mul(v128_load(tpl_rg.add(tpl_row + 12) as *const v128), v4),
            ];
            let tby = [
                f32x4_mul(v128_load(tpl_by.add(tpl_row) as *const v128), v4),
                f32x4_mul(v128_load(tpl_by.add(tpl_row + 4) as *const v128), v4),
                f32x4_mul(v128_load(tpl_by.add(tpl_row + 8) as *const v128), v4),
                f32x4_mul(v128_load(tpl_by.add(tpl_row + 12) as *const v128), v4),
            ];

            let mut prev_rg = if need_s0 {
                load9_hd_row_f32(rg_ptr.add(row_a))
            } else {
                [zero; 9]
            };
            let mut prev_by = if need_s0 {
                load9_hd_row_f32(by_ptr.add(row_a))
            } else {
                [zero; 9]
            };
            let mut pe_rg = [zero; 4];
            let mut pos_rg = [zero; 4];
            let mut po_rg = [zero; 4];
            let mut pe_by = [zero; 4];
            let mut pos_by = [zero; 4];
            let mut po_by = [zero; 4];
            let mut dummy_last = 0.0f32;

            // s = 0 (ab -> cands 0, 1, 2)
            if need_s0 {
                let m0 = r4_alive & 7;
                match ((m0 & 2) != 0, (m0 & 4) != 0) {
                    (true, true) => {
                        fold_add_load9_2x2_pe_po::<true>(&mut prev_rg, rg_ptr.add(row_a + stride), &mut pe_rg, &mut pos_rg, &mut po_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_po::<true>(&mut prev_by, by_ptr.add(row_a + stride), &mut pe_by, &mut pos_by, &mut po_by, &mut dummy_last);
                    }
                    (true, false) => {
                        fold_add_load9_2x2_pe_po::<false>(&mut prev_rg, rg_ptr.add(row_a + stride), &mut pe_rg, &mut pos_rg, &mut po_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_po::<false>(&mut prev_by, by_ptr.add(row_a + stride), &mut pe_by, &mut pos_by, &mut po_by, &mut dummy_last);
                    }
                    (false, true) => {
                        fold_add_load9_2x2_pe_only::<true>(&mut prev_rg, rg_ptr.add(row_a + stride), &mut pe_rg, &mut pos_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_only::<true>(&mut prev_by, by_ptr.add(row_a + stride), &mut pe_by, &mut pos_by, &mut dummy_last);
                    }
                    (false, false) => {
                        fold_add_load9_2x2_pe_only::<false>(&mut prev_rg, rg_ptr.add(row_a + stride), &mut pe_rg, &mut pos_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_only::<false>(&mut prev_by, by_ptr.add(row_a + stride), &mut pe_by, &mut pos_by, &mut dummy_last);
                    }
                }
                if (m0 & 1) != 0 {
                    for j in 0..4usize {
                        let d0 = f32x4_add(
                            f32x4_abs(f32x4_sub(pe_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(pe_by[j], tby[j])),
                        );
                        acc_sad_c[0] = f32x4_add(acc_sad_c[0], f32x4_mul(vw[j], d0));
                    }
                }
                if (m0 & 2) != 0 {
                    for j in 0..4usize {
                        let d1 = f32x4_add(
                            f32x4_abs(f32x4_sub(po_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(po_by[j], tby[j])),
                        );
                        acc_sad_c[1] = f32x4_add(acc_sad_c[1], f32x4_mul(vw[j], d1));
                    }
                }
                if (m0 & 4) != 0 {
                    for j in 0..4usize {
                        let d2 = f32x4_add(
                            f32x4_abs(f32x4_sub(pos_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(pos_by[j], tby[j])),
                        );
                        acc_sad_c[2] = f32x4_add(acc_sad_c[2], f32x4_mul(vw[j], d2));
                    }
                }
            } else if need_s1 {
                prev_rg = load9_hd_row_f32(rg_ptr.add(row_a + stride));
                prev_by = load9_hd_row_f32(by_ptr.add(row_a + stride));
            }

            // s = 1 (bc -> cands 3, 4)
            if need_s1 {
                let m1 = r4_alive & 24;
                if (m1 & 16) != 0 {
                    fold_add_load9_2x2_pe_only::<true>(&mut prev_rg, rg_ptr.add(row_a + 2 * stride), &mut pe_rg, &mut pos_rg, &mut dummy_last);
                    fold_add_load9_2x2_pe_only::<true>(&mut prev_by, by_ptr.add(row_a + 2 * stride), &mut pe_by, &mut pos_by, &mut dummy_last);
                } else {
                    fold_add_load9_2x2_pe_only::<false>(&mut prev_rg, rg_ptr.add(row_a + 2 * stride), &mut pe_rg, &mut pos_rg, &mut dummy_last);
                    fold_add_load9_2x2_pe_only::<false>(&mut prev_by, by_ptr.add(row_a + 2 * stride), &mut pe_by, &mut pos_by, &mut dummy_last);
                }
                if (m1 & 8) != 0 {
                    for j in 0..4usize {
                        let d3 = f32x4_add(
                            f32x4_abs(f32x4_sub(pe_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(pe_by[j], tby[j])),
                        );
                        acc_sad_c[3] = f32x4_add(acc_sad_c[3], f32x4_mul(vw[j], d3));
                    }
                }
                if (m1 & 16) != 0 {
                    for j in 0..4usize {
                        let d4 = f32x4_add(
                            f32x4_abs(f32x4_sub(pos_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(pos_by[j], tby[j])),
                        );
                        acc_sad_c[4] = f32x4_add(acc_sad_c[4], f32x4_mul(vw[j], d4));
                    }
                }
            } else if need_s2 {
                prev_rg = load9_hd_row_f32(rg_ptr.add(row_a + 2 * stride));
                prev_by = load9_hd_row_f32(by_ptr.add(row_a + 2 * stride));
            }

            // s = 2 (cd -> cands 5, 6, 7)
            if need_s2 {
                let m2 = r4_alive & 224;
                match ((m2 & 64) != 0, (m2 & 128) != 0) {
                    (true, true) => {
                        fold_add_load9_2x2_pe_po::<true>(&mut prev_rg, rg_ptr.add(row_a + 3 * stride), &mut pe_rg, &mut pos_rg, &mut po_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_po::<true>(&mut prev_by, by_ptr.add(row_a + 3 * stride), &mut pe_by, &mut pos_by, &mut po_by, &mut dummy_last);
                    }
                    (true, false) => {
                        fold_add_load9_2x2_pe_po::<false>(&mut prev_rg, rg_ptr.add(row_a + 3 * stride), &mut pe_rg, &mut pos_rg, &mut po_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_po::<false>(&mut prev_by, by_ptr.add(row_a + 3 * stride), &mut pe_by, &mut pos_by, &mut po_by, &mut dummy_last);
                    }
                    (false, true) => {
                        fold_add_load9_2x2_pe_only::<true>(&mut prev_rg, rg_ptr.add(row_a + 3 * stride), &mut pe_rg, &mut pos_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_only::<true>(&mut prev_by, by_ptr.add(row_a + 3 * stride), &mut pe_by, &mut pos_by, &mut dummy_last);
                    }
                    (false, false) => {
                        fold_add_load9_2x2_pe_only::<false>(&mut prev_rg, rg_ptr.add(row_a + 3 * stride), &mut pe_rg, &mut pos_rg, &mut dummy_last);
                        fold_add_load9_2x2_pe_only::<false>(&mut prev_by, by_ptr.add(row_a + 3 * stride), &mut pe_by, &mut pos_by, &mut dummy_last);
                    }
                }
                if (m2 & 32) != 0 {
                    for j in 0..4usize {
                        let d5 = f32x4_add(
                            f32x4_abs(f32x4_sub(pe_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(pe_by[j], tby[j])),
                        );
                        acc_sad_c[5] = f32x4_add(acc_sad_c[5], f32x4_mul(vw[j], d5));
                    }
                }
                if (m2 & 64) != 0 {
                    for j in 0..4usize {
                        let d6 = f32x4_add(
                            f32x4_abs(f32x4_sub(po_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(po_by[j], tby[j])),
                        );
                        acc_sad_c[6] = f32x4_add(acc_sad_c[6], f32x4_mul(vw[j], d6));
                    }
                }
                if (m2 & 128) != 0 {
                    for j in 0..4usize {
                        let d7 = f32x4_add(
                            f32x4_abs(f32x4_sub(pos_rg[j], trg[j])),
                            f32x4_abs(f32x4_sub(pos_by[j], tby[j])),
                        );
                        acc_sad_c[7] = f32x4_add(acc_sad_c[7], f32x4_mul(vw[j], d7));
                    }
                }
            }

            if t_idx == 1 || t_idx == 3 {
                let mut next_alive = 0u32;
                for c in 0..8usize {
                    if (r4_alive & (1u32 << c)) != 0
                        && 1.85 * (hsum_f32x4(acc_sad_c[c]) * inv_w4) - eca_lb < cur_best_pad
                    {
                        next_alive |= 1u32 << c;
                    }
                }
                r4_alive = next_alive;
                if r4_alive == 0 {
                    return;
                }
                if (r4_alive & (r4_alive - 1)) == 0 {
                    tail_single_start = t_idx + 1;
                    break;
                }
            }
        }
    }
    if tail_single_start < 6 {
        let c = r4_alive.trailing_zeros() as usize;
        let (dx, dy) = RING_OFFSETS[c];
        let tx = (cx0 as isize + (dx as isize)) as usize;
        let ty = (cy0 as isize + (dy as isize)) as usize;
        for t_idx in tail_single_start..6usize {
            let r = TAIL_STRIP_ORDER[t_idx];
            let row0 = (ty + r * 4) * stride + tx;
            let tpl_row = r * 32;
            for vec_idx in 0..4usize {
                let ioff = row0 + vec_idx * 8;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let irg = load_sum4_2x2_f32(rg_ptr.add(ioff), stride);
                let trg = f32x4_mul(v128_load(tpl_rg.add(toff) as *const v128), v4);
                let iby = load_sum4_2x2_f32(by_ptr.add(ioff), stride);
                let tby = f32x4_mul(v128_load(tpl_by.add(toff) as *const v128), v4);
                let d = f32x4_add(
                    f32x4_abs(f32x4_sub(irg, trg)),
                    f32x4_abs(f32x4_sub(iby, tby)),
                );
                acc_sad_c[c] = f32x4_add(acc_sad_c[c], f32x4_mul(vw, d));
            }
            if (t_idx == 1 || t_idx == 3)
                && 1.85 * (hsum_f32x4(acc_sad_c[c]) * inv_w4) - eca_lb >= cur_best_pad
            {
                return;
            }
        }
    }

    let mut alive_mask = 0u32;
    let mut sad_c_arr = [0.0f32; 8];
    for c in 0..8usize {
        if (r4_alive & (1u32 << c)) == 0 {
            continue;
        }
        let sad_c = hsum_f32x4(acc_sad_c[c]) * inv_w4;
        sad_c_arr[c] = sad_c;
        if 1.85 * sad_c - eca_lb < cur_best {
            alive_mask |= 1u32 << c;
        }
    }
    if alive_mask == 0 {
        return;
    }

    let mut il_uninit = [i32x4_splat(0); 160];
    let il_ptr = il_uninit.as_mut_ptr() as *mut v128;

    if (alive_mask & (alive_mask - 1)) == 0 {
        let c = alive_mask.trailing_zeros() as usize;
        let sad_c = sad_c_arr[c];
        let (dx, dy) = RING_OFFSETS[c];
        let tx = (cx0 as isize + (dx as isize)) as usize;
        let ty = (cy0 as isize + (dy as isize)) as usize;
        let mut sum_wi_1 = zero;
        let il_single = il_ptr;
        let mut k = 0usize;
        for r in 0..8usize {
            let row0 = (ty + r * 4) * stride + tx;
            let tpl_row = r * 32;
            for vec_idx in 0..4usize {
                let il = load_sum4_2x2_f32(luma_ptr.add(row0 + vec_idx * 8), stride);
                let vw = v128_load(tpl_weight.add(tpl_row + vec_idx * 4) as *const v128);
                *il_single.add(k) = il;
                sum_wi_1 = f32x4_add(sum_wi_1, f32x4_mul(vw, il));
                k += 1;
            }
        }
        let mu_i = hsum_f32x4(sum_wi_1) * inv_w4;
        let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
        let v_dc = f32x4_splat(4.0 * dc);
        let mut acc_sad_l = zero;
        for r in [2usize, 3, 4, 5] {
            let tpl_row = r * 32;
            let k_r = r * 4;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let il = *il_single.add(k_r + vec_idx);
                let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
            }
        }
        if hsum_f32x4(acc_sad_l) * inv_w4 + 1.85 * sad_c - eca_lb >= *out_best_cost {
            return;
        }
        for r in [0usize, 1, 6, 7] {
            let tpl_row = r * 32;
            let k_r = r * 4;
            for vec_idx in 0..4usize {
                let toff = tpl_row + vec_idx * 4;
                let il = *il_single.add(k_r + vec_idx);
                let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
            }
        }
        let sad_l = hsum_f32x4(acc_sad_l) * inv_w4;
        let partial = sad_l + 1.85 * sad_c;
        if partial - eca_lb < *out_best_cost {
            let w_eca = sad_half_weca_at_simd128(eca_res_ptr, stride, tx, ty, tpl_weight, inv_w, 2);
            let score = partial - 0.16 * w_eca.min(0.85);
            if score < *out_best_cost {
                *out_best_cost = score;
                *out_best_dx = dx;
                *out_best_dy = dy;
                *out_best_weca = w_eca;
            }
        }
        return;
    }

    let need_m1 = (alive_mask & 0b0000_0111) != 0;
    let need_0 = (alive_mask & 0b0001_1000) != 0;
    let need_p1 = (alive_mask & 0b1110_0000) != 0;

    let mut sum_wi = [zero; 8];
    let mut pe4 = [0.0f32; 24];

    // 2) Luma pass ONLY for dy strips containing chroma-surviving candidates.
    for r in 0..8usize {
        let row_a = (cy0 + r * 4 - 1) * stride + x_base;
        let tpl_row = r * 32;
        let vw = [
            v128_load(tpl_weight.add(tpl_row) as *const v128),
            v128_load(tpl_weight.add(tpl_row + 4) as *const v128),
            v128_load(tpl_weight.add(tpl_row + 8) as *const v128),
            v128_load(tpl_weight.add(tpl_row + 12) as *const v128),
        ];

        let mut pe = [zero; 4];
        let mut pe_pos = [zero; 4];
        let mut po = [zero; 4];
        let il_r = il_ptr.add(r * 20);

        let mut prev = if need_m1 {
            load9_hd_row_f32(luma_ptr.add(row_a))
        } else {
            [zero; 9]
        };
        if need_m1 {
            let am0 = alive_mask & 7;
            match ((am0 & 2) != 0, (am0 & 4) != 0) {
                (true, true) => {
                    fold_add_load9_2x2_pe_po::<true>(&mut prev, luma_ptr.add(row_a + stride), &mut pe, &mut pe_pos, &mut po, &mut pe4[r * 3]);
                    for j in 0..4usize {
                        *il_r.add(j) = pe[j];
                        *il_r.add(4 + j) = po[j];
                        sum_wi[1] = f32x4_add(sum_wi[1], f32x4_mul(vw[j], po[j]));
                    }
                }
                (true, false) => {
                    fold_add_load9_2x2_pe_po::<false>(&mut prev, luma_ptr.add(row_a + stride), &mut pe, &mut pe_pos, &mut po, &mut pe4[r * 3]);
                    for j in 0..4usize {
                        *il_r.add(j) = pe[j];
                        *il_r.add(4 + j) = po[j];
                        sum_wi[1] = f32x4_add(sum_wi[1], f32x4_mul(vw[j], po[j]));
                    }
                }
                (false, true) => {
                    fold_add_load9_2x2_pe_only::<true>(&mut prev, luma_ptr.add(row_a + stride), &mut pe, &mut pe_pos, &mut pe4[r * 3]);
                    for j in 0..4usize {
                        *il_r.add(j) = pe[j];
                    }
                }
                (false, false) => {
                    fold_add_load9_2x2_pe_only::<false>(&mut prev, luma_ptr.add(row_a + stride), &mut pe, &mut pe_pos, &mut pe4[r * 3]);
                    for j in 0..4usize {
                        *il_r.add(j) = pe[j];
                    }
                }
            }
            if (am0 & 1) != 0 {
                for j in 0..4usize {
                    sum_wi[0] = f32x4_add(sum_wi[0], f32x4_mul(vw[j], pe[j]));
                }
            }
            if (am0 & 4) != 0 {
                for j in 0..4usize {
                    sum_wi[2] = f32x4_add(sum_wi[2], f32x4_mul(vw[j], pe_pos[j]));
                }
            }
        } else if need_0 {
            prev = load9_hd_row_f32(luma_ptr.add(row_a + stride));
        }

        if need_0 {
            if (alive_mask & 16) != 0 {
                fold_add_load9_2x2_pe_only::<true>(&mut prev, luma_ptr.add(row_a + 2 * stride), &mut pe, &mut pe_pos, &mut pe4[r * 3 + 1]);
                for j in 0..4usize {
                    *il_r.add(8 + j) = pe[j];
                    sum_wi[4] = f32x4_add(sum_wi[4], f32x4_mul(vw[j], pe_pos[j]));
                }
            } else {
                fold_add_load9_2x2_pe_only::<false>(&mut prev, luma_ptr.add(row_a + 2 * stride), &mut pe, &mut pe_pos, &mut pe4[r * 3 + 1]);
                for j in 0..4usize {
                    *il_r.add(8 + j) = pe[j];
                }
            }
            if (alive_mask & 8) != 0 {
                for j in 0..4usize {
                    sum_wi[3] = f32x4_add(sum_wi[3], f32x4_mul(vw[j], pe[j]));
                }
            }
        } else if need_p1 {
            prev = load9_hd_row_f32(luma_ptr.add(row_a + 2 * stride));
        }

        if need_p1 {
            let am2 = alive_mask & 224;
            match ((am2 & 64) != 0, (am2 & 128) != 0) {
                (true, true) => {
                    fold_add_load9_2x2_pe_po::<true>(&mut prev, luma_ptr.add(row_a + 3 * stride), &mut pe, &mut pe_pos, &mut po, &mut pe4[r * 3 + 2]);
                    for j in 0..4usize {
                        *il_r.add(12 + j) = pe[j];
                        *il_r.add(16 + j) = po[j];
                        sum_wi[6] = f32x4_add(sum_wi[6], f32x4_mul(vw[j], po[j]));
                    }
                }
                (true, false) => {
                    fold_add_load9_2x2_pe_po::<false>(&mut prev, luma_ptr.add(row_a + 3 * stride), &mut pe, &mut pe_pos, &mut po, &mut pe4[r * 3 + 2]);
                    for j in 0..4usize {
                        *il_r.add(12 + j) = pe[j];
                        *il_r.add(16 + j) = po[j];
                        sum_wi[6] = f32x4_add(sum_wi[6], f32x4_mul(vw[j], po[j]));
                    }
                }
                (false, true) => {
                    fold_add_load9_2x2_pe_only::<true>(&mut prev, luma_ptr.add(row_a + 3 * stride), &mut pe, &mut pe_pos, &mut pe4[r * 3 + 2]);
                    for j in 0..4usize {
                        *il_r.add(12 + j) = pe[j];
                    }
                }
                (false, false) => {
                    fold_add_load9_2x2_pe_only::<false>(&mut prev, luma_ptr.add(row_a + 3 * stride), &mut pe, &mut pe_pos, &mut pe4[r * 3 + 2]);
                    for j in 0..4usize {
                        *il_r.add(12 + j) = pe[j];
                    }
                }
            }
            if (am2 & 32) != 0 {
                for j in 0..4usize {
                    sum_wi[5] = f32x4_add(sum_wi[5], f32x4_mul(vw[j], pe[j]));
                }
            }
            if (am2 & 128) != 0 {
                for j in 0..4usize {
                    sum_wi[7] = f32x4_add(sum_wi[7], f32x4_mul(vw[j], pe_pos[j]));
                }
            }
        }
    }

    let mut order = [0u8; 8];
    let mut n_ord = 0usize;
    let init_best = *out_best_cost;
    for c in 0..8usize {
        if (alive_mask & (1u32 << c)) != 0 && 1.85 * sad_c_arr[c] - eca_lb < init_best {
            let key = sad_c_arr[c];
            let mut pos = n_ord;
            while pos > 0 {
                let prev = order[pos - 1] as usize;
                if sad_c_arr[prev] <= key {
                    break;
                }
                order[pos] = order[pos - 1];
                pos -= 1;
            }
            order[pos] = c as u8;
            n_ord += 1;
        }
    }

    for oi in 0..n_ord {
        let c = order[oi] as usize;
        let sad_c = sad_c_arr[c];
        if 1.85 * sad_c - eca_lb >= *out_best_cost {
            break;
        }
        let mu_i = hsum_f32x4(sum_wi[c]) * inv_w4;
        let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
        let v_dc = f32x4_splat(4.0 * dc);
        let (dx, dy) = RING_OFFSETS[c];
        let (sub_off, p4_grp) = if dy < 0 {
            (if dx == 0 { 4usize } else { 0usize }, 0usize)
        } else if dy == 0 {
            (8usize, 1usize)
        } else {
            (if dx == 0 { 16usize } else { 12usize }, 2usize)
        };

        let mut acc_sad_l = zero;
        macro_rules! accum_st1_row {
            ($r:expr) => {{
                let r: usize = $r;
                let il_p = il_ptr.add(r * 20 + sub_off);
                let tpl_row = r * 32;
                if dx <= 0 {
                    for vec_idx in 0..4usize {
                        let toff = tpl_row + vec_idx * 4;
                        let il = *il_p.add(vec_idx);
                        let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                        let vw = v128_load(tpl_weight.add(toff) as *const v128);
                        let dl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
                        acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
                    }
                } else {
                    let il0 = *il_p;
                    let il1 = *il_p.add(1);
                    let il2 = *il_p.add(2);
                    let il3 = *il_p.add(3);
                    let il_last = f32x4_splat(pe4[r * 3 + p4_grp]);
                    let shifted = [
                        i32x4_shuffle::<1, 2, 3, 4>(il0, il1),
                        i32x4_shuffle::<1, 2, 3, 4>(il1, il2),
                        i32x4_shuffle::<1, 2, 3, 4>(il2, il3),
                        i32x4_shuffle::<1, 2, 3, 4>(il3, il_last),
                    ];
                    for vec_idx in 0..4usize {
                        let toff = tpl_row + vec_idx * 4;
                        let tl = f32x4_mul(v128_load(tpl_luma.add(toff) as *const v128), v4);
                        let vw = v128_load(tpl_weight.add(toff) as *const v128);
                        let dl = f32x4_abs(f32x4_sub(f32x4_sub(shifted[vec_idx], tl), v_dc));
                        acc_sad_l = f32x4_add(acc_sad_l, f32x4_mul(vw, dl));
                    }
                }
            }};
        }
        for r in [2usize, 3, 4, 5] {
            accum_st1_row!(r);
        }
        if hsum_f32x4(acc_sad_l) * inv_w4 + 1.85 * sad_c - eca_lb >= *out_best_cost {
            continue;
        }
        for r in [0usize, 1, 6, 7] {
            accum_st1_row!(r);
        }
        let sad_l = hsum_f32x4(acc_sad_l) * inv_w4;
        let partial = sad_l + 1.85 * sad_c;
        if partial - eca_lb >= *out_best_cost {
            continue;
        }
        let tx = (cx0 as isize + (dx as isize)) as usize;
        let ty = (cy0 as isize + (dy as isize)) as usize;
        let w_eca = sad_half_weca_at_simd128(eca_res_ptr, stride, tx, ty, tpl_weight, inv_w, 2);
        let score = partial - 0.16 * w_eca.min(0.85);
        if score < *out_best_cost {
            *out_best_cost = score;
            *out_best_dx = dx;
            *out_best_dy = dy;
            *out_best_weca = w_eca;
        }
    }
}

/// Evaluates all 8 `(-1..=1, -1..=1) \ {(0,0)}` neighbors at step `1` around `(cx0, cy0)`
/// sharing loaded image rows (SD `17x18`, HD `8x4x34` 2x2-pooled) and exact Pass-2 bound skip.
#[inline]
pub unsafe fn sad_patch_zm_half_st1_ring_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    let (inv_w, mu_t) = tpl_half_stats_simd128(tpl_luma, tpl_weight);
    sad_patch_zm_half_st1_ring_precomp_simd128(
        luma_ptr,
        rg_ptr,
        by_ptr,
        eca_res_ptr,
        stride,
        cx0,
        cy0,
        tpl_luma,
        tpl_rg,
        tpl_by,
        tpl_weight,
        inv_w,
        mu_t,
        scale_px,
        out_best_dx,
        out_best_dy,
        out_best_cost,
        out_best_weca,
    );
}

#[inline]
pub unsafe fn sad_patch_zm_half_st1_ring_precomp_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx0: usize,
    cy0: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    inv_w: f32,
    mu_t: f32,
    scale_px: usize,
    out_best_dx: &mut i32,
    out_best_dy: &mut i32,
    out_best_cost: &mut f32,
    out_best_weca: &mut f32,
) {
    if scale_px <= 1 {
        sad_patch_zm_half_st1_ring_sd_simd128(
            luma_ptr,
            rg_ptr,
            by_ptr,
            eca_res_ptr,
            stride,
            cx0,
            cy0,
            tpl_luma,
            tpl_rg,
            tpl_by,
            tpl_weight,
            inv_w,
            out_best_dx,
            out_best_dy,
            out_best_cost,
            out_best_weca,
        );
    } else {
        sad_patch_zm_half_st1_ring_hd_simd128(
            luma_ptr,
            rg_ptr,
            by_ptr,
            eca_res_ptr,
            stride,
            cx0,
            cy0,
            tpl_luma,
            tpl_rg,
            tpl_by,
            tpl_weight,
            inv_w,
            mu_t,
            out_best_dx,
            out_best_dy,
            out_best_cost,
            out_best_weca,
        );
    }
}

#[inline]
pub unsafe fn weighted_eca_centroid_simd128(
    eca_res: *const f32,
    frame_w: usize,
    rx0: usize,
    rx1: usize,
    ry0: usize,
    ry1: usize,
) -> (f32, f32, f32) {
    let thresh = f32x4_splat(0.05);
    let four = f32x4_splat(4.0);
    let mut pw = 0.0f32;
    let mut pwx = 0.0f32;
    let mut pwy = 0.0f32;
    let x0_f = rx0 as f32;
    let x_init = f32x4(x0_f, x0_f + 1.0, x0_f + 2.0, x0_f + 3.0);

    for py in ry0..=ry1 {
        let row = py * frame_w;
        let mut vx = x_init;
        let mut acc_w = f32x4_splat(0.0);
        let mut acc_wx = f32x4_splat(0.0);
        let mut px = rx0;
        while px + 3 <= rx1 {
            let r = v128_load(eca_res.add(row + px) as *const v128);
            let mask = f32x4_gt(r, thresh);
            let w = v128_and(f32x4_mul(r, r), mask);
            acc_w = f32x4_add(acc_w, w);
            acc_wx = f32x4_add(acc_wx, f32x4_mul(w, vx));
            vx = f32x4_add(vx, four);
            px += 4;
        }
        let mut row_pw = hsum_f32x4(acc_w);
        let mut row_pwx = hsum_f32x4(acc_wx);
        while px <= rx1 {
            let r = *eca_res.add(row + px);
            if r > 0.05 {
                let w = r * r;
                row_pw += w;
                row_pwx += w * (px as f32);
            }
            px += 1;
        }
        pw += row_pw;
        pwx += row_pwx;
        pwy += row_pw * (py as f32);
    }
    (pw, pwx, pwy)
}

/// Full Zero-Mean Foreground-Masked SIMD128 Patch Tracker Cost with 2-adic Haar-Morton evaluation
/// (used at coarse winners / finalist scoring). Luma-only DC + contrast gain; unshifted `rg`/`by`.
#[inline]
pub unsafe fn sad_patch_zm_weighted_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    tpl_sig: u32,
    out_cand_sig: *mut u32,
    out_weighted_eca: *mut f32,
    scale_px: usize,
) -> f32 {
    let base_cost = sad_patch_zm_score_simd128(
        luma_ptr,
        rg_ptr,
        by_ptr,
        eca_res_ptr,
        stride,
        cx,
        cy,
        tpl_luma,
        tpl_rg,
        tpl_by,
        tpl_weight,
        out_weighted_eca,
        scale_px,
    );
    let cand_sig = haar_morton_signature_hyst_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, cx, cy, scale_px, tpl_sig, true,
    );
    if !out_cand_sig.is_null() {
        *out_cand_sig = cand_sig;
    }
    let val = padic_ultrametric_val(tpl_sig, cand_sig);
    let d2 = padic_distance_from_val(val);
    base_cost + 0.55 * d2
}

/// Zero-Mean Pinned-Anchor Drift-Lock L1 distance (`anchor_luma`, `anchor_rg`, `anchor_by`).
/// Applies luma-only DC (`[-0.42, 0.42]`) + contrast gain (`[0.72, 1.45]`) while keeping `rg`/`by` raw.
#[inline]
pub unsafe fn weighted_anchor_zm_l1_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    anchor_luma: *const f32,
    anchor_rg: *const f32,
    anchor_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
) -> f32 {
    let cstep = 4 * scale_px;
    let mut sum_w = f32x4_splat(0.0);
    let mut sum_wi = f32x4_splat(0.0);
    let mut sum_wa = f32x4_splat(0.0);
    let mut sum_wi2 = f32x4_splat(0.0);
    let mut sum_wa2 = f32x4_splat(0.0);
    let mut acc_c = f32x4_splat(0.0);

    if scale_px <= 1 {
        for ry in 0..16usize {
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 4;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il = v128_load(luma_ptr.add(ioff) as *const v128);
                let al = v128_load(anchor_luma.add(toff) as *const v128);
                let wi = f32x4_mul(vw, il);
                let wa = f32x4_mul(vw, al);
                sum_w = f32x4_add(sum_w, vw);
                sum_wi = f32x4_add(sum_wi, wi);
                sum_wa = f32x4_add(sum_wa, wa);
                sum_wi2 = f32x4_add(sum_wi2, f32x4_mul(wi, il));
                sum_wa2 = f32x4_add(sum_wa2, f32x4_mul(wa, al));

                let dc_rg = f32x4_abs(f32x4_sub(
                    v128_load(rg_ptr.add(ioff) as *const v128),
                    v128_load(anchor_rg.add(toff) as *const v128),
                ));
                let dc_by = f32x4_abs(f32x4_sub(
                    v128_load(by_ptr.add(ioff) as *const v128),
                    v128_load(anchor_by.add(toff) as *const v128),
                ));
                acc_c = f32x4_add(acc_c, f32x4_mul(vw, f32x4_add(dc_rg, dc_by)));
            }
        }
        let inv_w = 1.0 / hsum_f32x4(sum_w).max(1e-6);
        let mu_i = hsum_f32x4(sum_wi) * inv_w;
        let mu_a = hsum_f32x4(sum_wa) * inv_w;
        let var_i = (hsum_f32x4(sum_wi2) * inv_w - mu_i * mu_i).max(0.0);
        let var_a = (hsum_f32x4(sum_wa2) * inv_w - mu_a * mu_a).max(0.0);
        let std_i = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(var_i)));
        let std_a = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(var_a)));
        let gain = if std_a > 0.015 && std_i > 0.012 {
            (std_a / std_i).clamp(0.72, 1.45)
        } else {
            1.0
        };
        let dc = (mu_i - mu_a).clamp(-0.42, 0.42);
        let l_bias = mu_i - dc;
        let v_mui = f32x4_splat(mu_i);
        let v_gain = f32x4_splat(gain);
        let v_bias = f32x4_splat(l_bias);
        let mut acc_l = f32x4_splat(0.0);
        for ry in 0..16usize {
            let img_row = (cy + ry) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 4;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il_raw = v128_load(luma_ptr.add(ioff) as *const v128);
                let il_zm = f32x4_add(f32x4_mul(f32x4_sub(il_raw, v_mui), v_gain), v_bias);
                let dl = f32x4_abs(f32x4_sub(il_zm, v128_load(anchor_luma.add(toff) as *const v128)));
                acc_l = f32x4_add(acc_l, f32x4_mul(vw, dl));
            }
        }
        return hsum_f32x4(acc_l) + 1.65 * hsum_f32x4(acc_c);
    }

    let mut il_uninit = [i32x4_splat(0); 64];
    let il_ptr = il_uninit.as_mut_ptr() as *mut v128;

    if scale_px == 2 {
        let mut k = 0usize;
        for ry in 0..16usize {
            let img_row = (cy + ry * 2) * stride + cx;
            let tpl_row = ry * 16;
            for vec_idx in 0..4usize {
                let ioff = img_row + vec_idx * 8;
                let toff = tpl_row + vec_idx * 4;
                let vw = v128_load(tpl_weight.add(toff) as *const v128);
                let il4 = load_sum4_2x2_f32(luma_ptr.add(ioff), stride);
                *il_ptr.add(k) = il4;
                k += 1;
                let al = v128_load(anchor_luma.add(toff) as *const v128);
                let wi = f32x4_mul(vw, il4);
                let wa = f32x4_mul(vw, al);
                sum_w = f32x4_add(sum_w, vw);
                sum_wi = f32x4_add(sum_wi, wi);
                sum_wa = f32x4_add(sum_wa, wa);
                sum_wi2 = f32x4_add(sum_wi2, f32x4_mul(wi, il4));
                sum_wa2 = f32x4_add(sum_wa2, f32x4_mul(wa, al));

                let irg4 = load_sum4_2x2_f32(rg_ptr.add(ioff), stride);
                let iby4 = load_sum4_2x2_f32(by_ptr.add(ioff), stride);
                let v_q = f32x4_splat(0.25);
                let dc_rg = f32x4_abs(f32x4_sub(
                    f32x4_mul(irg4, v_q),
                    v128_load(anchor_rg.add(toff) as *const v128),
                ));
                let dc_by = f32x4_abs(f32x4_sub(
                    f32x4_mul(iby4, v_q),
                    v128_load(anchor_by.add(toff) as *const v128),
                ));
                acc_c = f32x4_add(acc_c, f32x4_mul(vw, f32x4_add(dc_rg, dc_by)));
            }
        }
        let inv_w = 1.0 / hsum_f32x4(sum_w).max(1e-6);
        let mu_i = hsum_f32x4(sum_wi) * (0.25 * inv_w);
        let mu_a = hsum_f32x4(sum_wa) * inv_w;
        let var_i = (hsum_f32x4(sum_wi2) * (0.0625 * inv_w) - mu_i * mu_i).max(0.0);
        let var_a = (hsum_f32x4(sum_wa2) * inv_w - mu_a * mu_a).max(0.0);
        let std_i = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(var_i)));
        let std_a = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(var_a)));
        let gain = if std_a > 0.015 && std_i > 0.012 {
            (std_a / std_i).clamp(0.72, 1.45)
        } else {
            1.0
        };
        let dc = (mu_i - mu_a).clamp(-0.42, 0.42);
        let l_bias = mu_i - dc;
        let v_gain_q = f32x4_splat(0.25 * gain);
        let v_bias_shift = f32x4_splat(l_bias - mu_i * gain);

        let mut acc_l = f32x4_splat(0.0);
        for idx in 0..64usize {
            let toff = idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let il4 = *il_ptr.add(idx);
            let il_zm = f32x4_add(f32x4_mul(il4, v_gain_q), v_bias_shift);
            let dl = f32x4_abs(f32x4_sub(il_zm, v128_load(anchor_luma.add(toff) as *const v128)));
            acc_l = f32x4_add(acc_l, f32x4_mul(vw, dl));
        }
        return hsum_f32x4(acc_l) + 1.65 * hsum_f32x4(acc_c);
    }

    let mut k = 0usize;
    for ry in 0..16usize {
        let img_row = (cy + ry * scale_px) * stride + cx;
        let tpl_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * cstep;
            let toff = tpl_row + vec_idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let il = load_pooled4_f32(luma_ptr.add(ioff), stride, scale_px);
            *il_ptr.add(k) = il;
            k += 1;
            let al = v128_load(anchor_luma.add(toff) as *const v128);
            let wi = f32x4_mul(vw, il);
            let wa = f32x4_mul(vw, al);
            sum_w = f32x4_add(sum_w, vw);
            sum_wi = f32x4_add(sum_wi, wi);
            sum_wa = f32x4_add(sum_wa, wa);
            sum_wi2 = f32x4_add(sum_wi2, f32x4_mul(wi, il));
            sum_wa2 = f32x4_add(sum_wa2, f32x4_mul(wa, al));

            let dc_rg = f32x4_abs(f32x4_sub(
                load_pooled4_f32(rg_ptr.add(ioff), stride, scale_px),
                v128_load(anchor_rg.add(toff) as *const v128),
            ));
            let dc_by = f32x4_abs(f32x4_sub(
                load_pooled4_f32(by_ptr.add(ioff), stride, scale_px),
                v128_load(anchor_by.add(toff) as *const v128),
            ));
            acc_c = f32x4_add(acc_c, f32x4_mul(vw, f32x4_add(dc_rg, dc_by)));
        }
    }

    let v_w_i_a_c = hsum4_f32x4(sum_w, sum_wi, sum_wa, acc_c);
    let inv_w = 1.0 / f32x4_extract_lane::<0>(v_w_i_a_c).max(1e-6);
    let mu_i = f32x4_extract_lane::<1>(v_w_i_a_c) * inv_w;
    let mu_a = f32x4_extract_lane::<2>(v_w_i_a_c) * inv_w;
    let acc_c_sum = f32x4_extract_lane::<3>(v_w_i_a_c);
    let v_var = hsum4_f32x4(sum_wi2, sum_wa2, sum_wi2, sum_wa2);
    let var_i = (f32x4_extract_lane::<0>(v_var) * inv_w - mu_i * mu_i).max(0.0);
    let var_a = (f32x4_extract_lane::<1>(v_var) * inv_w - mu_a * mu_a).max(0.0);
    let std_ia = f32x4_sqrt(f32x4_splat(var_i));
    let std_aa = f32x4_sqrt(f32x4_splat(var_a));
    let std_i = f32x4_extract_lane::<0>(std_ia);
    let std_a = f32x4_extract_lane::<0>(std_aa);
    let gain = if std_a > 0.015 && std_i > 0.012 {
        (std_a / std_i).clamp(0.72, 1.45)
    } else {
        1.0
    };
    let dc = (mu_i - mu_a).clamp(-0.42, 0.42);
    let l_bias = mu_i - dc;
    let v_gain = f32x4_splat(gain);
    let v_bias_shift = f32x4_splat(l_bias - mu_i * gain);

    let mut acc_l = f32x4_splat(0.0);
    for idx in 0..64usize {
        let toff = idx * 4;
        let vw = v128_load(tpl_weight.add(toff) as *const v128);
        let il_raw = *il_ptr.add(idx);
        let il_zm = f32x4_add(f32x4_mul(il_raw, v_gain), v_bias_shift);
        let dl = f32x4_abs(f32x4_sub(il_zm, v128_load(anchor_luma.add(toff) as *const v128)));
        acc_l = f32x4_add(acc_l, f32x4_mul(vw, dl));
    }
    hsum_f32x4(acc_l) + 1.65 * acc_c_sum
}

/// True iff both patches are confident chromatic (`bit 31 == 1`) AND occupy opposite poles
/// (`11` vs `00`, separated by >= 0.110 opponent units across the [-0.055, +0.055] deadzone).
#[inline(always)]
pub fn has_opposite_chroma_pole(sig_a: u32, sig_b: u32) -> bool {
    if ((sig_a & sig_b) & (1 << 31)) == 0 {
        return false;
    }
    let rg_a = (sig_a >> 29) & 0x3;
    let rg_b = (sig_b >> 29) & 0x3;
    let by_a = (sig_a >> 27) & 0x3;
    let by_b = (sig_b >> 27) & 0x3;
    (rg_a ^ rg_b) == 0x3 || (by_a ^ by_b) == 0x3
}

/// Fast opposite-chroma-pole check reading ONLY chroma planes `(rg, by)` and computing
/// center/total chroma means directly without reading luma or building Haar/Morton signatures.
#[inline]
pub unsafe fn has_opposite_chroma_pole_patch_simd128(
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    x0: usize,
    y0: usize,
    scale_px: usize,
    ref_sig: u32,
) -> bool {
    if (ref_sig & (1 << 31)) == 0 {
        return false;
    }
    let mut center_rg_v = f32x4_splat(0.0);
    let mut center_by_v = f32x4_splat(0.0);
    let mut outer_rg_v = f32x4_splat(0.0);
    let mut outer_by_v = f32x4_splat(0.0);

    let (center_rg, center_by, total_rg, total_by) = if scale_px == 2 {
        for r in 0..8usize {
            let row0 = (y0 + r) * stride + x0;
            let row1 = (y0 + 24 + r) * stride + x0;
            let rp0 = rg_ptr.add(row0);
            let rp1 = rg_ptr.add(row1);
            let bp0 = by_ptr.add(row0);
            let bp1 = by_ptr.add(row1);
            for k in 0..8usize {
                let off = k * 4;
                outer_rg_v = f32x4_add(outer_rg_v, f32x4_add(v128_load(rp0.add(off) as *const v128), v128_load(rp1.add(off) as *const v128)));
                outer_by_v = f32x4_add(outer_by_v, f32x4_add(v128_load(bp0.add(off) as *const v128), v128_load(bp1.add(off) as *const v128)));
            }
        }
        for r in 8..24usize {
            let row = (y0 + r) * stride + x0;
            let rp = rg_ptr.add(row);
            let bp = by_ptr.add(row);
            outer_rg_v = f32x4_add(outer_rg_v, f32x4_add(v128_load(rp as *const v128), v128_load(rp.add(4) as *const v128)));
            outer_rg_v = f32x4_add(outer_rg_v, f32x4_add(v128_load(rp.add(24) as *const v128), v128_load(rp.add(28) as *const v128)));
            center_rg_v = f32x4_add(center_rg_v, f32x4_add(v128_load(rp.add(8) as *const v128), v128_load(rp.add(12) as *const v128)));
            center_rg_v = f32x4_add(center_rg_v, f32x4_add(v128_load(rp.add(16) as *const v128), v128_load(rp.add(20) as *const v128)));

            outer_by_v = f32x4_add(outer_by_v, f32x4_add(v128_load(bp as *const v128), v128_load(bp.add(4) as *const v128)));
            outer_by_v = f32x4_add(outer_by_v, f32x4_add(v128_load(bp.add(24) as *const v128), v128_load(bp.add(28) as *const v128)));
            center_by_v = f32x4_add(center_by_v, f32x4_add(v128_load(bp.add(8) as *const v128), v128_load(bp.add(12) as *const v128)));
            center_by_v = f32x4_add(center_by_v, f32x4_add(v128_load(bp.add(16) as *const v128), v128_load(bp.add(20) as *const v128)));
        }
        let crg = hsum_f32x4(center_rg_v) * (1.0 / 256.0);
        let cby = hsum_f32x4(center_by_v) * (1.0 / 256.0);
        let trg = hsum_f32x4(f32x4_add(center_rg_v, outer_rg_v)) * (1.0 / 1024.0);
        let tby = hsum_f32x4(f32x4_add(center_by_v, outer_by_v)) * (1.0 / 1024.0);
        (crg, cby, trg, tby)
    } else {
        let cstep = 4 * scale_px;
        for by in 0..4usize {
            let mid_y = by == 1 || by == 2;
            for ry in 0..4usize {
                let row_base = (y0 + (by * 4 + ry) * scale_px) * stride + x0;
                let r0 = load_pooled4_f32(rg_ptr.add(row_base), stride, scale_px);
                let r1 = load_pooled4_f32(rg_ptr.add(row_base + cstep), stride, scale_px);
                let r2 = load_pooled4_f32(rg_ptr.add(row_base + cstep * 2), stride, scale_px);
                let r3 = load_pooled4_f32(rg_ptr.add(row_base + cstep * 3), stride, scale_px);
                let b0 = load_pooled4_f32(by_ptr.add(row_base), stride, scale_px);
                let b1 = load_pooled4_f32(by_ptr.add(row_base + cstep), stride, scale_px);
                let b2 = load_pooled4_f32(by_ptr.add(row_base + cstep * 2), stride, scale_px);
                let b3 = load_pooled4_f32(by_ptr.add(row_base + cstep * 3), stride, scale_px);
                outer_rg_v = f32x4_add(outer_rg_v, f32x4_add(r0, r3));
                outer_by_v = f32x4_add(outer_by_v, f32x4_add(b0, b3));
                if mid_y {
                    center_rg_v = f32x4_add(center_rg_v, f32x4_add(r1, r2));
                    center_by_v = f32x4_add(center_by_v, f32x4_add(b1, b2));
                } else {
                    outer_rg_v = f32x4_add(outer_rg_v, f32x4_add(r1, r2));
                    outer_by_v = f32x4_add(outer_by_v, f32x4_add(b1, b2));
                }
            }
        }
        let crg = hsum_f32x4(center_rg_v) * (1.0 / 64.0);
        let cby = hsum_f32x4(center_by_v) * (1.0 / 64.0);
        let trg = hsum_f32x4(f32x4_add(center_rg_v, outer_rg_v)) * (1.0 / 256.0);
        let tby = hsum_f32x4(f32x4_add(center_by_v, outer_by_v)) * (1.0 / 256.0);
        (crg, cby, trg, tby)
    };

    let mean_rg = 0.65 * center_rg + 0.35 * total_rg;
    let mean_by = 0.65 * center_by + 0.35 * total_by;
    if mean_rg * mean_rg + mean_by * mean_by <= 0.0155 {
        return false;
    }
    let rg_b = (((mean_rg > 0.055) as u32) << 1) | ((mean_rg > -0.055) as u32);
    let by_b = (((mean_by > 0.055) as u32) << 1) | ((mean_by > -0.055) as u32);
    let rg_a = (ref_sig >> 29) & 0x3;
    let by_a = (ref_sig >> 27) & 0x3;
    (rg_a ^ rg_b) == 0x3 || (by_a ^ by_b) == 0x3
}

/// Fused HD (`scale_px == 2`) winner evaluation: computes `sad_patch_zm_score`, `haar_morton_signature_hyst`,
/// and `weighted_anchor_zm_l1` from a single `load_pooled4_f32` pass over the 32x32 frame patch.
/// Returns `0` on base-SAD rejection, `1` on opposite-chroma-pole rejection (`*out_best_sig` written),
/// or `2` on acceptance (`out_best_base_c`, `out_anchor_l1`, `out_weighted_eca`, `out_best_sig` written).
#[inline]
pub unsafe fn compute_anchor_luma_stats_simd128(
    anchor_luma: *const f32,
    tpl_weight: *const f32,
    out_mu_a: *mut f32,
    out_std_a: *mut f32,
) {
    let mut sum_w = f32x4_splat(0.0);
    let mut sum_wa = f32x4_splat(0.0);
    let mut sum_wa2 = f32x4_splat(0.0);
    for idx in 0..64usize {
        let toff = idx * 4;
        let vw = v128_load(tpl_weight.add(toff) as *const v128);
        let al = v128_load(anchor_luma.add(toff) as *const v128);
        let wa = f32x4_mul(vw, al);
        sum_w = f32x4_add(sum_w, vw);
        sum_wa = f32x4_add(sum_wa, wa);
        sum_wa2 = f32x4_add(sum_wa2, f32x4_mul(wa, al));
    }
    let v_s = hsum4_f32x4(sum_w, sum_wa, sum_wa2, sum_w);
    let inv_w = 1.0 / f32x4_extract_lane::<0>(v_s).max(1e-6);
    let mu_a = f32x4_extract_lane::<1>(v_s) * inv_w;
    let var_a = (f32x4_extract_lane::<2>(v_s) * inv_w - mu_a * mu_a).max(0.0);
    *out_mu_a = mu_a;
    *out_std_a = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(var_a)));
}

/// Returns `0` on base-SAD rejection, `1` on opposite-chroma-pole rejection (`*out_best_sig` written),
/// or `2` on acceptance (`out_best_base_c`, `out_anchor_l1`, `out_weighted_eca`, `out_best_sig` written).
#[inline(never)]
pub unsafe fn eval_winner_patch_hd_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    eca_res_ptr: *const f32,
    stride: usize,
    cx: usize,
    cy: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    anchor_luma: *const f32,
    anchor_rg: *const f32,
    anchor_by: *const f32,
    anchor_sig: u32,
    inv_w: f32,
    mu_t: f32,
    mu_a: f32,
    std_a: f32,
    max_base_c: f32,
    out_best_base_c: *mut f32,
    out_anchor_l1: *mut f32,
    out_weighted_eca: *mut f32,
    out_best_sig: *mut u32,
) -> u8 {
    let mut blk_rg = [0.0f32; 16];
    let mut blk_by = [0.0f32; 16];
    let mut irg_uninit = [i32x4_splat(0); 64];
    let mut iby_uninit = [i32x4_splat(0); 64];
    let irg_ptr = irg_uninit.as_mut_ptr() as *mut v128;
    let iby_ptr = iby_uninit.as_mut_ptr() as *mut v128;

    let inv_16 = 1.0 / 16.0;
    let v_inv16 = f32x4_splat(inv_16);
    let zero_v = f32x4_splat(0.0);
    let mut acc_sad_c = zero_v;

    // Stage 1: Chroma SAD (center bands first) + opposite chroma pole gate; cache pooled chroma for Stage 3.
    for by in [1usize, 2, 0, 3] {
        let mut acc_rg0 = zero_v;
        let mut acc_rg1 = zero_v;
        let mut acc_rg2 = zero_v;
        let mut acc_rg3 = zero_v;

        let mut acc_by0 = zero_v;
        let mut acc_by1 = zero_v;
        let mut acc_by2 = zero_v;
        let mut acc_by3 = zero_v;

        for ry_in in 0..4usize {
            let ry = by * 4 + ry_in;
            let img_row = (cy + ry * 2) * stride + cx;
            let tpl_row = ry * 16;
            let k_row = ry * 4;

            macro_rules! step_chroma {
                ($vx:expr, $arg:ident, $aby:ident) => {{
                    let ioff = img_row + $vx * 8;
                    let toff = tpl_row + $vx * 4;
                    let vw = v128_load(tpl_weight.add(toff) as *const v128);
                    let irg = load_pooled4_f32(rg_ptr.add(ioff), stride, 2);
                    let iby = load_pooled4_f32(by_ptr.add(ioff), stride, 2);
                    *irg_ptr.add(k_row + $vx) = irg;
                    *iby_ptr.add(k_row + $vx) = iby;
                    $arg = f32x4_add($arg, irg);
                    $aby = f32x4_add($aby, iby);
                    let diff_tc = f32x4_add(
                        f32x4_abs(f32x4_sub(irg, v128_load(tpl_rg.add(toff) as *const v128))),
                        f32x4_abs(f32x4_sub(iby, v128_load(tpl_by.add(toff) as *const v128))),
                    );
                    acc_sad_c = f32x4_add(acc_sad_c, f32x4_mul(vw, diff_tc));
                }};
            }
            step_chroma!(0, acc_rg0, acc_by0);
            step_chroma!(1, acc_rg1, acc_by1);
            step_chroma!(2, acc_rg2, acc_by2);
            step_chroma!(3, acc_rg3, acc_by3);
        }

        let z0 = morton2d_4x4(0, by);
        let z1 = morton2d_4x4(1, by);
        let z2 = morton2d_4x4(2, by);
        let z3 = morton2d_4x4(3, by);

        let vrg = f32x4_mul(hsum4_f32x4(acc_rg0, acc_rg1, acc_rg2, acc_rg3), v_inv16);
        let vby = f32x4_mul(hsum4_f32x4(acc_by0, acc_by1, acc_by2, acc_by3), v_inv16);

        blk_rg[z0] = f32x4_extract_lane::<0>(vrg);
        blk_rg[z1] = f32x4_extract_lane::<1>(vrg);
        blk_rg[z2] = f32x4_extract_lane::<2>(vrg);
        blk_rg[z3] = f32x4_extract_lane::<3>(vrg);

        blk_by[z0] = f32x4_extract_lane::<0>(vby);
        blk_by[z1] = f32x4_extract_lane::<1>(vby);
        blk_by[z2] = f32x4_extract_lane::<2>(vby);
        blk_by[z3] = f32x4_extract_lane::<3>(vby);

        if by != 3 {
            let sad_c_mid_w = 1.85 * hsum_f32x4(acc_sad_c);
            if sad_c_mid_w >= 0.42 || sad_c_mid_w - 0.1361f32 >= max_base_c {
                return 0;
            }
        }
    }

    let sad_c = hsum_f32x4(acc_sad_c);
    let sad_c_weighted = 1.85 * sad_c;
    if sad_c_weighted >= 0.42 || sad_c_weighted - 0.1361f32 >= max_base_c {
        return 0;
    }

    let center_rg = 0.25 * (blk_rg[3] + blk_rg[6] + blk_rg[9] + blk_rg[12]);
    let center_by = 0.25 * (blk_by[3] + blk_by[6] + blk_by[9] + blk_by[12]);
    let mut total_rg = 0.0f32;
    let mut total_by = 0.0f32;
    for q in 0..4usize {
        let b = q * 4;
        total_rg += 0.25 * (blk_rg[b] + blk_rg[b + 1] + blk_rg[b + 2] + blk_rg[b + 3]);
        total_by += 0.25 * (blk_by[b] + blk_by[b + 1] + blk_by[b + 2] + blk_by[b + 3]);
    }
    let mean_rg = 0.65 * center_rg + 0.35 * (total_rg * 0.25);
    let mean_by = 0.65 * center_by + 0.35 * (total_by * 0.25);
    let sat_thresh = if (anchor_sig & (1 << 31)) != 0 { 0.0155 } else { 0.0205 };
    let mut csig_top = 0u32;
    if mean_rg * mean_rg + mean_by * mean_by > sat_thresh {
        csig_top |= 1 << 31;
        if mean_rg > 0.055 { csig_top |= 1 << 30; }
        if mean_rg > -0.055 { csig_top |= 1 << 29; }
        if mean_by > 0.055 { csig_top |= 1 << 28; }
        if mean_by > -0.055 { csig_top |= 1 << 27; }
    } else {
        csig_top |= (1 << 29) | (1 << 27);
    }
    if has_opposite_chroma_pole(anchor_sig, csig_top) {
        *out_best_sig = csig_top;
        return 1;
    }

    // Stage 2: Luma SAD FIRST (defer block sums & sum_wi2 until winner passes gate).
    let mut il_uninit = [i32x4_splat(0); 64];
    let il_ptr = il_uninit.as_mut_ptr() as *mut v128;
    let mut sum_wi0 = zero_v;
    let mut sum_wi1 = zero_v;
    let has_eca = !eca_res_ptr.is_null();

    for ry in 0..16usize {
        let img_row = (cy + ry * 2) * stride + cx;
        let tpl_row = ry * 16;
        let k_row = ry * 4;
        let il0 = load_pooled4_f32(luma_ptr.add(img_row), stride, 2);
        let il1 = load_pooled4_f32(luma_ptr.add(img_row + 8), stride, 2);
        let il2 = load_pooled4_f32(luma_ptr.add(img_row + 16), stride, 2);
        let il3 = load_pooled4_f32(luma_ptr.add(img_row + 24), stride, 2);
        *il_ptr.add(k_row) = il0;
        *il_ptr.add(k_row + 1) = il1;
        *il_ptr.add(k_row + 2) = il2;
        *il_ptr.add(k_row + 3) = il3;
        let vw0 = v128_load(tpl_weight.add(tpl_row) as *const v128);
        let vw1 = v128_load(tpl_weight.add(tpl_row + 4) as *const v128);
        let vw2 = v128_load(tpl_weight.add(tpl_row + 8) as *const v128);
        let vw3 = v128_load(tpl_weight.add(tpl_row + 12) as *const v128);
        sum_wi0 = f32x4_add(sum_wi0, f32x4_add(f32x4_mul(vw0, il0), f32x4_mul(vw2, il2)));
        sum_wi1 = f32x4_add(sum_wi1, f32x4_add(f32x4_mul(vw1, il1), f32x4_mul(vw3, il3)));
    }

    let mu_i = hsum_f32x4(f32x4_add(sum_wi0, sum_wi1)) * inv_w;
    let dc = (mu_i - mu_t).clamp(-0.12, 0.12) * 0.40;
    let v_dc = f32x4_splat(dc);

    let mut acc_sad_l0 = zero_v;
    let mut acc_sad_l1 = zero_v;
    macro_rules! accum_sad_l {
        ($idx:expr, $acc:ident) => {{
            let toff = $idx * 4;
            let vw = v128_load(tpl_weight.add(toff) as *const v128);
            let il = *il_ptr.add($idx);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            let dl_tpl = f32x4_abs(f32x4_sub(f32x4_sub(il, tl), v_dc));
            $acc = f32x4_add($acc, f32x4_mul(vw, dl_tpl));
        }};
    }
    // Check high-weight center rows [4..12] first before outer edge rows.
    let mut idx = 16usize;
    while idx < 48usize {
        accum_sad_l!(idx, acc_sad_l0);
        accum_sad_l!(idx + 1, acc_sad_l1);
        idx += 2;
    }
    let partial_mid = hsum_f32x4(f32x4_add(acc_sad_l0, acc_sad_l1)) + sad_c_weighted;
    if partial_mid >= 0.42 || partial_mid - 0.1361f32 >= max_base_c {
        return 0;
    }
    idx = 0usize;
    while idx < 16usize {
        accum_sad_l!(idx, acc_sad_l0);
        accum_sad_l!(idx + 1, acc_sad_l1);
        idx += 2;
    }
    idx = 48usize;
    while idx < 64usize {
        accum_sad_l!(idx, acc_sad_l0);
        accum_sad_l!(idx + 1, acc_sad_l1);
        idx += 2;
    }

    let sad_l = hsum_f32x4(f32x4_add(acc_sad_l0, acc_sad_l1));
    let partial = sad_l + sad_c_weighted;
    if partial >= 0.42 || partial - 0.1361f32 >= max_base_c {
        return 0;
    }
    let w_eca = if has_eca {
        let mut acc_eca = zero_v;
        for ry in 0..16usize {
            let img_row = (cy + ry * 2) * stride + cx;
            let tpl_row = ry * 16;
            for vx in 0..4usize {
                let vw = v128_load(tpl_weight.add(tpl_row + vx * 4) as *const v128);
                let eca4 = load_sum4_2x2_f32(eca_res_ptr.add(img_row + vx * 8), stride);
                acc_eca = f32x4_add(acc_eca, f32x4_mul(vw, eca4));
            }
        }
        hsum_f32x4(acc_eca) * 0.25
    } else {
        0.0
    };
    let motion_reward = w_eca.min(0.85);
    let best_base_c = partial - 0.16 * motion_reward;
    if best_base_c >= max_base_c + 1e-6f32 {
        return 0;
    }

    // Stage 3: Winner accepted -> Pack signature + single fused Anchor L1 pass.
    let mut blk_luma = [0.0f32; 16];
    let mut sum_wi2 = zero_v;
    for by in 0..4usize {
        let mut acc_l0 = zero_v;
        let mut acc_l1 = zero_v;
        let mut acc_l2 = zero_v;
        let mut acc_l3 = zero_v;
        for ry_in in 0..4usize {
            let ry = by * 4 + ry_in;
            let k_row = ry * 4;
            let tpl_row = ry * 16;
            let il0 = *il_ptr.add(k_row);
            let il1 = *il_ptr.add(k_row + 1);
            let il2 = *il_ptr.add(k_row + 2);
            let il3 = *il_ptr.add(k_row + 3);
            acc_l0 = f32x4_add(acc_l0, il0);
            acc_l1 = f32x4_add(acc_l1, il1);
            acc_l2 = f32x4_add(acc_l2, il2);
            acc_l3 = f32x4_add(acc_l3, il3);
            let vw0 = v128_load(tpl_weight.add(tpl_row) as *const v128);
            let vw1 = v128_load(tpl_weight.add(tpl_row + 4) as *const v128);
            let vw2 = v128_load(tpl_weight.add(tpl_row + 8) as *const v128);
            let vw3 = v128_load(tpl_weight.add(tpl_row + 12) as *const v128);
            sum_wi2 = f32x4_add(sum_wi2, f32x4_mul(vw0, f32x4_mul(il0, il0)));
            sum_wi2 = f32x4_add(sum_wi2, f32x4_mul(vw1, f32x4_mul(il1, il1)));
            sum_wi2 = f32x4_add(sum_wi2, f32x4_mul(vw2, f32x4_mul(il2, il2)));
            sum_wi2 = f32x4_add(sum_wi2, f32x4_mul(vw3, f32x4_mul(il3, il3)));
        }
        let vl = f32x4_mul(hsum4_f32x4(acc_l0, acc_l1, acc_l2, acc_l3), v_inv16);
        blk_luma[morton2d_4x4(0, by)] = f32x4_extract_lane::<0>(vl);
        blk_luma[morton2d_4x4(1, by)] = f32x4_extract_lane::<1>(vl);
        blk_luma[morton2d_4x4(2, by)] = f32x4_extract_lane::<2>(vl);
        blk_luma[morton2d_4x4(3, by)] = f32x4_extract_lane::<3>(vl);
    }
    let best_sig = pack_haar_morton_from_blocks(&blk_luma, &blk_rg, &blk_by, anchor_sig, true);

    let var_i = (hsum_f32x4(sum_wi2) * inv_w - mu_i * mu_i).max(0.0);
    let std_i = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(var_i)));
    let gain = if std_a > 0.015 && std_i > 0.012 {
        (std_a / std_i).clamp(0.72, 1.45)
    } else {
        1.0
    };
    let dc_a = (mu_i - mu_a).clamp(-0.42, 0.42);
    let l_bias = mu_i - dc_a;
    let v_gain = f32x4_splat(gain);
    let v_offset = f32x4_splat(l_bias - mu_i * gain);

    let mut acc_anchor_l = zero_v;
    let mut anchor_acc_c = zero_v;
    for idx in 0..64usize {
        let toff = idx * 4;
        let vw = v128_load(tpl_weight.add(toff) as *const v128);
        let il = *il_ptr.add(idx);
        let al = v128_load(anchor_luma.add(toff) as *const v128);
        let il_zm = f32x4_add(f32x4_mul(il, v_gain), v_offset);
        let dl_anc = f32x4_abs(f32x4_sub(il_zm, al));
        acc_anchor_l = f32x4_add(acc_anchor_l, f32x4_mul(vw, dl_anc));

        let irg = *irg_ptr.add(idx);
        let iby = *iby_ptr.add(idx);
        let diff_ac = f32x4_add(
            f32x4_abs(f32x4_sub(irg, v128_load(anchor_rg.add(toff) as *const v128))),
            f32x4_abs(f32x4_sub(iby, v128_load(anchor_by.add(toff) as *const v128))),
        );
        anchor_acc_c = f32x4_add(anchor_acc_c, f32x4_mul(vw, diff_ac));
    }

    *out_best_base_c = best_base_c;
    *out_anchor_l1 = hsum_f32x4(acc_anchor_l) + 1.65 * hsum_f32x4(anchor_acc_c);
    *out_weighted_eca = w_eca;
    *out_best_sig = best_sig;
    2
}

#[inline(always)]
unsafe fn sample_bilinear4_shifted(
    ptr: *const f32,
    row0: usize,
    row1: usize,
    ix0: usize,
    scale_px: usize,
    vw00: v128,
    vw10: v128,
    vw01: v128,
    vw11: v128,
) -> v128 {
    if scale_px <= 1 {
        let p00 = v128_load(ptr.add(row0 + ix0) as *const v128);
        let p10 = v128_load(ptr.add(row0 + ix0 + 1) as *const v128);
        let p01 = v128_load(ptr.add(row1 + ix0) as *const v128);
        let p11 = v128_load(ptr.add(row1 + ix0 + 1) as *const v128);
        f32x4_add(
            f32x4_add(f32x4_mul(vw00, p00), f32x4_mul(vw10, p10)),
            f32x4_add(f32x4_mul(vw01, p01), f32x4_mul(vw11, p11)),
        )
    } else {
        let r0a = v128_load(ptr.add(row0 + ix0) as *const v128);
        let r0b = v128_load(ptr.add(row0 + ix0 + 4) as *const v128);
        let r1a = v128_load(ptr.add(row1 + ix0) as *const v128);
        let r1b = v128_load(ptr.add(row1 + ix0 + 4) as *const v128);
        let p00 = i32x4_shuffle::<0, 2, 4, 6>(r0a, r0b);
        let p10 = i32x4_shuffle::<1, 3, 5, 7>(r0a, r0b);
        let p01 = i32x4_shuffle::<0, 2, 4, 6>(r1a, r1b);
        let p11 = i32x4_shuffle::<1, 3, 5, 7>(r1a, r1b);
        f32x4_add(
            f32x4_add(f32x4_mul(vw00, p00), f32x4_mul(vw10, p10)),
            f32x4_add(f32x4_mul(vw01, p01), f32x4_mul(vw11, p11)),
        )
    }
}

/// 3-Iteration SIMD128 Huber-IRLS Inverse-Compositional Lucas-Kanade (IC-LK) Sub-Pixel Refinement.
/// Uses adaptive joint-channel Huber IRLS (`delta in [0.032, 0.12]`, `wh <= 1.0`), luma-only DC compensation
/// (`dc in [-0.45, 0.45]`, leaving `rg`/`by` unshifted), and separable weighted SSD parabolic fallback.
/// Annotated `#[inline(never)]` to isolate stack usage (`<= 1 KiB`) under `-zstack-size=16384`.
#[inline(never)]
pub unsafe fn refine_subpixel_ic_lk_huber_simd128(
    luma_ptr: *const f32,
    rg_ptr: *const f32,
    by_ptr: *const f32,
    stride: usize,
    best_tlx: usize,
    best_tly: usize,
    tpl_luma: *const f32,
    tpl_rg: *const f32,
    tpl_by: *const f32,
    tpl_weight: *const f32,
    scale_px: usize,
) -> (f32, f32) {
    let s_f = scale_px as f32;
    let inv_2s = 0.5 / s_f;
    let inv_2s2 = inv_2s * inv_2s;
    let center_off = 0.5 * ((scale_px - 1) as f32);

    let mut sum_w_tl = f32x4_splat(0.0);
    let mut sum_w_in = f32x4_splat(0.0);
    let mut base_hxx_v = f32x4_splat(0.0);
    let mut base_hxy_v = f32x4_splat(0.0);
    let mut base_hyy_v = f32x4_splat(0.0);

    const LANE_MASK_TAIL: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    let v_tail_mask = v128_load(LANE_MASK_TAIL.as_ptr() as *const v128);
    let v_14 = f32x4_splat(1.4);

    // Quick template weighted mean + base Hessian check (defer `* inv_2s^2` outside the 56-quad loop)
    for ry in 1..15usize {
        let trow = ry * 16;
        for blk in 0..4usize {
            let rx = if blk < 3 { 1 + blk * 4 } else { 11 };
            let idx = trow + rx;
            let mut vw = v128_load(tpl_weight.add(idx) as *const v128);
            if blk == 3 {
                vw = f32x4_mul(vw, v_tail_mask);
            }
            let tl = v128_load(tpl_luma.add(idx) as *const v128);
            sum_w_in = f32x4_add(sum_w_in, vw);
            sum_w_tl = f32x4_add(sum_w_tl, f32x4_mul(vw, tl));

            let gtx = f32x4_sub(
                v128_load(tpl_luma.add(idx + 1) as *const v128),
                v128_load(tpl_luma.add(idx - 1) as *const v128),
            );
            let gty = f32x4_sub(
                v128_load(tpl_luma.add(idx + 16) as *const v128),
                v128_load(tpl_luma.add(idx - 16) as *const v128),
            );
            let grgx = f32x4_sub(
                v128_load(tpl_rg.add(idx + 1) as *const v128),
                v128_load(tpl_rg.add(idx - 1) as *const v128),
            );
            let grgy = f32x4_sub(
                v128_load(tpl_rg.add(idx + 16) as *const v128),
                v128_load(tpl_rg.add(idx - 16) as *const v128),
            );
            let gbyx = f32x4_sub(
                v128_load(tpl_by.add(idx + 1) as *const v128),
                v128_load(tpl_by.add(idx - 1) as *const v128),
            );
            let gbyy = f32x4_sub(
                v128_load(tpl_by.add(idx + 16) as *const v128),
                v128_load(tpl_by.add(idx - 16) as *const v128),
            );

            let jxx = f32x4_add(
                f32x4_mul(gtx, gtx),
                f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, grgx), f32x4_mul(gbyx, gbyx))),
            );
            let jxy = f32x4_add(
                f32x4_mul(gtx, gty),
                f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, grgy), f32x4_mul(gbyx, gbyy))),
            );
            let jyy = f32x4_add(
                f32x4_mul(gty, gty),
                f32x4_mul(v_14, f32x4_add(f32x4_mul(grgy, grgy), f32x4_mul(gbyy, gbyy))),
            );
            base_hxx_v = f32x4_add(base_hxx_v, f32x4_mul(vw, jxx));
            base_hxy_v = f32x4_add(base_hxy_v, f32x4_mul(vw, jxy));
            base_hyy_v = f32x4_add(base_hyy_v, f32x4_mul(vw, jyy));
        }
    }
    let base_hxx = hsum_f32x4(base_hxx_v) * inv_2s2;
    let base_hxy = hsum_f32x4(base_hxy_v) * inv_2s2;
    let base_hyy = hsum_f32x4(base_hyy_v) * inv_2s2;

    let w_in_total = hsum_f32x4(sum_w_in);
    let w_tl_total = hsum_f32x4(sum_w_tl);
    let base_det = base_hxx * base_hyy - base_hxy * base_hxy;
    let base_tr = base_hxx + base_hyy;
    let max_shift = 0.62 * s_f;

    if base_det > 1e-7 && base_det > 0.02 * base_tr * base_tr && w_in_total > 0.25 {
        let inv_win = 1.0 / w_in_total;
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        let mut huber_delta = 0.075f32;
        let mut converged_ok = true;
        let v_one = f32x4_splat(1.0);
        let v_eps = f32x4_splat(1e-6);
        let v_inv2s = f32x4_splat(inv_2s);
        let v_inv2s2 = f32x4_splat(inv_2s2);
        let mut il_uninit = [i32x4_splat(0); 56];
        let il_cache = il_uninit.as_mut_ptr() as *mut v128;

        for iter in 0..3usize {
            let fy0 = (best_tly + scale_px) as f32 + center_off + dy;
            let fx0 = (best_tlx + scale_px) as f32 + center_off + dx;
            let iy0_i = fy0 as isize;
            let ix0_i = fx0 as isize;
            if iy0_i < 0 || ix0_i < 0 {
                converged_ok = false;
                break;
            }
            let ty = fy0 - (iy0_i as f32);
            let tx = fx0 - (ix0_i as f32);
            let w00 = (1.0 - tx) * (1.0 - ty);
            let w10 = tx * (1.0 - ty);
            let w01 = (1.0 - tx) * ty;
            let w11 = tx * ty;
            let vw00 = f32x4_splat(w00);
            let vw10 = f32x4_splat(w10);
            let vw01 = f32x4_splat(w01);
            let vw11 = f32x4_splat(w11);
            let iy0 = iy0_i as usize;
            let ix0 = ix0_i as usize;

            // Step A: Luma-only weighted DC offset + cache `il` across Step B (saves 168 bilinear luma warps/call)
            let mut sum_w_il = f32x4_splat(0.0);
            if scale_px == 2 {
                let mut k = 0usize;
                for ry in 1..15usize {
                    let trow = ry * 16;
                    let row0 = (iy0 + (ry - 1) * 2) * stride + ix0;
                    let row1 = row0 + stride;
                    let lp0 = luma_ptr.add(row0);
                    let lp1 = luma_ptr.add(row1);
                    let r0_0 = v128_load(lp0 as *const v128);
                    let r0_1 = v128_load(lp0.add(4) as *const v128);
                    let r0_2 = v128_load(lp0.add(8) as *const v128);
                    let r0_3 = v128_load(lp0.add(12) as *const v128);
                    let r0_4 = v128_load(lp0.add(16) as *const v128);
                    let r0_5 = v128_load(lp0.add(20) as *const v128);
                    let r0_6 = v128_load(lp0.add(24) as *const v128);
                    let r1_0 = v128_load(lp1 as *const v128);
                    let r1_1 = v128_load(lp1.add(4) as *const v128);
                    let r1_2 = v128_load(lp1.add(8) as *const v128);
                    let r1_3 = v128_load(lp1.add(12) as *const v128);
                    let r1_4 = v128_load(lp1.add(16) as *const v128);
                    let r1_5 = v128_load(lp1.add(20) as *const v128);
                    let r1_6 = v128_load(lp1.add(24) as *const v128);
                    let b4 = |r0a: v128, r0b: v128, r1a: v128, r1b: v128| -> v128 {
                        let p00 = i32x4_shuffle::<0, 2, 4, 6>(r0a, r0b);
                        let p10 = i32x4_shuffle::<1, 3, 5, 7>(r0a, r0b);
                        let p01 = i32x4_shuffle::<0, 2, 4, 6>(r1a, r1b);
                        let p11 = i32x4_shuffle::<1, 3, 5, 7>(r1a, r1b);
                        f32x4_add(
                            f32x4_add(f32x4_mul(vw00, p00), f32x4_mul(vw10, p10)),
                            f32x4_add(f32x4_mul(vw01, p01), f32x4_mul(vw11, p11)),
                        )
                    };
                    let il0 = b4(r0_0, r0_1, r1_0, r1_1);
                    let il1 = b4(r0_2, r0_3, r1_2, r1_3);
                    let il2 = b4(r0_4, r0_5, r1_4, r1_5);
                    let il3 = b4(r0_5, r0_6, r1_5, r1_6);
                    *il_cache.add(k) = il0;
                    *il_cache.add(k + 1) = il1;
                    *il_cache.add(k + 2) = il2;
                    *il_cache.add(k + 3) = il3;
                    k += 4;
                    let vw0 = v128_load(tpl_weight.add(trow + 1) as *const v128);
                    let vw1 = v128_load(tpl_weight.add(trow + 5) as *const v128);
                    let vw2 = v128_load(tpl_weight.add(trow + 9) as *const v128);
                    let vw3 = f32x4_mul(v128_load(tpl_weight.add(trow + 11) as *const v128), v_tail_mask);
                    sum_w_il = f32x4_add(
                        sum_w_il,
                        f32x4_add(
                            f32x4_add(f32x4_mul(vw0, il0), f32x4_mul(vw1, il1)),
                            f32x4_add(f32x4_mul(vw2, il2), f32x4_mul(vw3, il3)),
                        ),
                    );
                }
            } else {
                let mut k = 0usize;
                for ry in 1..15usize {
                    let trow = ry * 16;
                    let row0 = (iy0 + (ry - 1) * scale_px) * stride;
                    let row1 = row0 + stride;
                    for blk in 0..4usize {
                        let rx = if blk < 3 { 1 + blk * 4 } else { 11 };
                        let idx = trow + rx;
                        let mut vw = v128_load(tpl_weight.add(idx) as *const v128);
                        if blk == 3 {
                            vw = f32x4_mul(vw, v_tail_mask);
                        }
                        let il = sample_bilinear4_shifted(
                            luma_ptr,
                            row0,
                            row1,
                            ix0 + (rx - 1) * scale_px,
                            scale_px,
                            vw00,
                            vw10,
                            vw01,
                            vw11,
                        );
                        *il_cache.add(k) = il;
                        k += 1;
                        sum_w_il = f32x4_add(sum_w_il, f32x4_mul(vw, il));
                    }
                }
            }
            let dc = ((hsum_f32x4(sum_w_il) - w_tl_total) * inv_win).clamp(-0.45, 0.45);
            let v_dc = f32x4_splat(dc);
            let v_delta = f32x4_splat(huber_delta);

            // Step B: Huber IRLS weighted Hessian & gradient RHS across 14x14 interior
            let mut acc_hxx = f32x4_splat(0.0);
            let mut acc_hxy = f32x4_splat(0.0);
            let mut acc_hyy = f32x4_splat(0.0);
            let mut acc_bx = f32x4_splat(0.0);
            let mut acc_by = f32x4_splat(0.0);
            let mut acc_wh = f32x4_splat(0.0);
            let mut acc_wh_e2 = f32x4_splat(0.0);

            if scale_px == 2 {
                let b4 = |r0a: v128, r0b: v128, r1a: v128, r1b: v128| -> v128 {
                    let p00 = i32x4_shuffle::<0, 2, 4, 6>(r0a, r0b);
                    let p10 = i32x4_shuffle::<1, 3, 5, 7>(r0a, r0b);
                    let p01 = i32x4_shuffle::<0, 2, 4, 6>(r1a, r1b);
                    let p11 = i32x4_shuffle::<1, 3, 5, 7>(r1a, r1b);
                    f32x4_add(
                        f32x4_add(f32x4_mul(vw00, p00), f32x4_mul(vw10, p10)),
                        f32x4_add(f32x4_mul(vw01, p01), f32x4_mul(vw11, p11)),
                    )
                };
                let mut k = 0usize;
                for ry in 1..15usize {
                    let trow = ry * 16;
                    let row0 = (iy0 + (ry - 1) * 2) * stride + ix0;
                    let row1 = row0 + stride;
                    let rgp0 = rg_ptr.add(row0);
                    let rgp1 = rg_ptr.add(row1);
                    let byp0 = by_ptr.add(row0);
                    let byp1 = by_ptr.add(row1);

                    let rg0_0 = v128_load(rgp0 as *const v128);
                    let rg0_1 = v128_load(rgp0.add(4) as *const v128);
                    let rg0_2 = v128_load(rgp0.add(8) as *const v128);
                    let rg0_3 = v128_load(rgp0.add(12) as *const v128);
                    let rg0_4 = v128_load(rgp0.add(16) as *const v128);
                    let rg0_5 = v128_load(rgp0.add(20) as *const v128);
                    let rg0_6 = v128_load(rgp0.add(24) as *const v128);
                    let rg1_0 = v128_load(rgp1 as *const v128);
                    let rg1_1 = v128_load(rgp1.add(4) as *const v128);
                    let rg1_2 = v128_load(rgp1.add(8) as *const v128);
                    let rg1_3 = v128_load(rgp1.add(12) as *const v128);
                    let rg1_4 = v128_load(rgp1.add(16) as *const v128);
                    let rg1_5 = v128_load(rgp1.add(20) as *const v128);
                    let rg1_6 = v128_load(rgp1.add(24) as *const v128);
                    let irg_row = [
                        b4(rg0_0, rg0_1, rg1_0, rg1_1),
                        b4(rg0_2, rg0_3, rg1_2, rg1_3),
                        b4(rg0_4, rg0_5, rg1_4, rg1_5),
                        b4(rg0_5, rg0_6, rg1_5, rg1_6),
                    ];

                    let by0_0 = v128_load(byp0 as *const v128);
                    let by0_1 = v128_load(byp0.add(4) as *const v128);
                    let by0_2 = v128_load(byp0.add(8) as *const v128);
                    let by0_3 = v128_load(byp0.add(12) as *const v128);
                    let by0_4 = v128_load(byp0.add(16) as *const v128);
                    let by0_5 = v128_load(byp0.add(20) as *const v128);
                    let by0_6 = v128_load(byp0.add(24) as *const v128);
                    let by1_0 = v128_load(byp1 as *const v128);
                    let by1_1 = v128_load(byp1.add(4) as *const v128);
                    let by1_2 = v128_load(byp1.add(8) as *const v128);
                    let by1_3 = v128_load(byp1.add(12) as *const v128);
                    let by1_4 = v128_load(byp1.add(16) as *const v128);
                    let by1_5 = v128_load(byp1.add(20) as *const v128);
                    let by1_6 = v128_load(byp1.add(24) as *const v128);
                    let iby_row = [
                        b4(by0_0, by0_1, by1_0, by1_1),
                        b4(by0_2, by0_3, by1_2, by1_3),
                        b4(by0_4, by0_5, by1_4, by1_5),
                        b4(by0_5, by0_6, by1_5, by1_6),
                    ];

                    for blk in 0..4usize {
                        let rx = if blk < 3 { 1 + blk * 4 } else { 11 };
                        let idx = trow + rx;
                        let mut vw = v128_load(tpl_weight.add(idx) as *const v128);
                        if blk == 3 {
                            vw = f32x4_mul(vw, v_tail_mask);
                        }
                        let il = *il_cache.add(k);
                        k += 1;
                        let irg = irg_row[blk];
                        let iby = iby_row[blk];

                        let err_l = f32x4_sub(f32x4_sub(il, v_dc), v128_load(tpl_luma.add(idx) as *const v128));
                        let err_rg = f32x4_sub(irg, v128_load(tpl_rg.add(idx) as *const v128));
                        let err_by = f32x4_sub(iby, v128_load(tpl_by.add(idx) as *const v128));

                        let e2 = f32x4_add(
                            f32x4_mul(err_l, err_l),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(err_rg, err_rg), f32x4_mul(err_by, err_by))),
                        );
                        let emag = f32x4_sqrt(f32x4_max(e2, v_eps));
                        let wh = f32x4_min(f32x4_div(v_delta, emag), v_one);
                        let weff = f32x4_mul(vw, wh);

                        acc_wh = f32x4_add(acc_wh, weff);
                        acc_wh_e2 = f32x4_add(acc_wh_e2, f32x4_mul(weff, e2));

                        let gtx = f32x4_sub(
                            v128_load(tpl_luma.add(idx + 1) as *const v128),
                            v128_load(tpl_luma.add(idx - 1) as *const v128),
                        );
                        let gty = f32x4_sub(
                            v128_load(tpl_luma.add(idx + 16) as *const v128),
                            v128_load(tpl_luma.add(idx - 16) as *const v128),
                        );
                        let grgx = f32x4_sub(
                            v128_load(tpl_rg.add(idx + 1) as *const v128),
                            v128_load(tpl_rg.add(idx - 1) as *const v128),
                        );
                        let grgy = f32x4_sub(
                            v128_load(tpl_rg.add(idx + 16) as *const v128),
                            v128_load(tpl_rg.add(idx - 16) as *const v128),
                        );
                        let gbyx = f32x4_sub(
                            v128_load(tpl_by.add(idx + 1) as *const v128),
                            v128_load(tpl_by.add(idx - 1) as *const v128),
                        );
                        let gbyy = f32x4_sub(
                            v128_load(tpl_by.add(idx + 16) as *const v128),
                            v128_load(tpl_by.add(idx - 16) as *const v128),
                        );

                        let jxx = f32x4_add(
                            f32x4_mul(gtx, gtx),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, grgx), f32x4_mul(gbyx, gbyx))),
                        );
                        let jxy = f32x4_add(
                            f32x4_mul(gtx, gty),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, grgy), f32x4_mul(gbyx, gbyy))),
                        );
                        let jyy = f32x4_add(
                            f32x4_mul(gty, gty),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgy, grgy), f32x4_mul(gbyy, gbyy))),
                        );
                        let jx_e = f32x4_add(
                            f32x4_mul(gtx, err_l),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, err_rg), f32x4_mul(gbyx, err_by))),
                        );
                        let jy_e = f32x4_add(
                            f32x4_mul(gty, err_l),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgy, err_rg), f32x4_mul(gbyy, err_by))),
                        );

                        acc_hxx = f32x4_add(acc_hxx, f32x4_mul(weff, jxx));
                        acc_hxy = f32x4_add(acc_hxy, f32x4_mul(weff, jxy));
                        acc_hyy = f32x4_add(acc_hyy, f32x4_mul(weff, jyy));
                        acc_bx = f32x4_add(acc_bx, f32x4_mul(weff, jx_e));
                        acc_by = f32x4_add(acc_by, f32x4_mul(weff, jy_e));
                    }
                }
            } else {
                let mut k = 0usize;
                for ry in 1..15usize {
                    let trow = ry * 16;
                    let row0 = (iy0 + (ry - 1) * scale_px) * stride;
                    let row1 = row0 + stride;
                    for blk in 0..4usize {
                        let rx = if blk < 3 { 1 + blk * 4 } else { 11 };
                        let idx = trow + rx;
                        let sx = ix0 + (rx - 1) * scale_px;

                        let mut vw = v128_load(tpl_weight.add(idx) as *const v128);
                        if blk == 3 {
                            vw = f32x4_mul(vw, v_tail_mask);
                        }

                        let il = *il_cache.add(k);
                        k += 1;
                        let irg = sample_bilinear4_shifted(rg_ptr, row0, row1, sx, scale_px, vw00, vw10, vw01, vw11);
                        let iby = sample_bilinear4_shifted(by_ptr, row0, row1, sx, scale_px, vw00, vw10, vw01, vw11);

                        let err_l = f32x4_sub(f32x4_sub(il, v_dc), v128_load(tpl_luma.add(idx) as *const v128));
                        let err_rg = f32x4_sub(irg, v128_load(tpl_rg.add(idx) as *const v128));
                        let err_by = f32x4_sub(iby, v128_load(tpl_by.add(idx) as *const v128));

                        let e2 = f32x4_add(
                            f32x4_mul(err_l, err_l),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(err_rg, err_rg), f32x4_mul(err_by, err_by))),
                        );
                        let emag = f32x4_sqrt(f32x4_max(e2, v_eps));
                        let wh = f32x4_min(f32x4_div(v_delta, emag), v_one);
                        let weff = f32x4_mul(vw, wh);

                        acc_wh = f32x4_add(acc_wh, weff);
                        acc_wh_e2 = f32x4_add(acc_wh_e2, f32x4_mul(weff, e2));

                        let gtx = f32x4_sub(
                            v128_load(tpl_luma.add(idx + 1) as *const v128),
                            v128_load(tpl_luma.add(idx - 1) as *const v128),
                        );
                        let gty = f32x4_sub(
                            v128_load(tpl_luma.add(idx + 16) as *const v128),
                            v128_load(tpl_luma.add(idx - 16) as *const v128),
                        );
                        let grgx = f32x4_sub(
                            v128_load(tpl_rg.add(idx + 1) as *const v128),
                            v128_load(tpl_rg.add(idx - 1) as *const v128),
                        );
                        let grgy = f32x4_sub(
                            v128_load(tpl_rg.add(idx + 16) as *const v128),
                            v128_load(tpl_rg.add(idx - 16) as *const v128),
                        );
                        let gbyx = f32x4_sub(
                            v128_load(tpl_by.add(idx + 1) as *const v128),
                            v128_load(tpl_by.add(idx - 1) as *const v128),
                        );
                        let gbyy = f32x4_sub(
                            v128_load(tpl_by.add(idx + 16) as *const v128),
                            v128_load(tpl_by.add(idx - 16) as *const v128),
                        );

                        let jxx = f32x4_add(
                            f32x4_mul(gtx, gtx),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, grgx), f32x4_mul(gbyx, gbyx))),
                        );
                        let jxy = f32x4_add(
                            f32x4_mul(gtx, gty),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, grgy), f32x4_mul(gbyx, gbyy))),
                        );
                        let jyy = f32x4_add(
                            f32x4_mul(gty, gty),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgy, grgy), f32x4_mul(gbyy, gbyy))),
                        );
                        let jx_e = f32x4_add(
                            f32x4_mul(gtx, err_l),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgx, err_rg), f32x4_mul(gbyx, err_by))),
                        );
                        let jy_e = f32x4_add(
                            f32x4_mul(gty, err_l),
                            f32x4_mul(v_14, f32x4_add(f32x4_mul(grgy, err_rg), f32x4_mul(gbyy, err_by))),
                        );

                        acc_hxx = f32x4_add(acc_hxx, f32x4_mul(weff, jxx));
                        acc_hxy = f32x4_add(acc_hxy, f32x4_mul(weff, jxy));
                        acc_hyy = f32x4_add(acc_hyy, f32x4_mul(weff, jyy));
                        acc_bx = f32x4_add(acc_bx, f32x4_mul(weff, jx_e));
                        acc_by = f32x4_add(acc_by, f32x4_mul(weff, jy_e));
                    }
                }
            }
            let h_vec = f32x4_mul(hsum4_f32x4(acc_hxx, acc_hxy, acc_hyy, acc_bx), v_inv2s2);
            let w_vec = hsum4_f32x4(acc_wh, acc_wh_e2, acc_bx, acc_by);
            let wh_tot = f32x4_extract_lane::<0>(w_vec);
            if wh_tot > 0.05 {
                let wh_rms = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(
                    (f32x4_extract_lane::<1>(w_vec) / wh_tot).max(1e-6),
                )));
                huber_delta = (1.345 * wh_rms).clamp(0.032, 0.12);
            }

            let h_xx = f32x4_extract_lane::<0>(h_vec);
            let h_xy = f32x4_extract_lane::<1>(h_vec);
            let h_yy = f32x4_extract_lane::<2>(h_vec);
            let det = h_xx * h_yy - h_xy * h_xy;
            let tr = h_xx + h_yy;
            if !(det > 1e-7 && det > 0.02 * tr * tr) {
                if iter == 0 {
                    converged_ok = false;
                }
                break;
            }
            let inv_det = 1.0 / det;
            let b_inv2s = f32x4_mul(w_vec, v_inv2s);
            let bx = f32x4_extract_lane::<2>(b_inv2s);
            let by = f32x4_extract_lane::<3>(b_inv2s);
            let step_x = -(h_yy * bx - h_xy * by) * inv_det;
            let step_y = -(-h_xy * bx + h_xx * by) * inv_det;
            dx = (dx + step_x).clamp(-max_shift, max_shift);
            dy = (dy + step_y).clamp(-max_shift, max_shift);
            if iter >= 1 && step_x.abs() < 0.015 && step_y.abs() < 0.015 {
                break;
            }
        }
        if converged_ok && dx.is_finite() && dy.is_finite() {
            return (dx, dy);
        }
    }

    // Fallback: Separable foreground-weighted parabolic SSD
    let c_0 = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let c_xm = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx - 1, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let c_xp = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx + 1, best_tly,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let mut sub_dx = 0.0f32;
    let denom_x = 2.0 * (c_xm - 2.0 * c_0 + c_xp);
    if denom_x > 1e-5 {
        sub_dx = ((c_xm - c_xp) / denom_x).clamp(-0.48, 0.48);
    }

    let c_ym = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly - 1,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let c_yp = ssd16x16_weighted_simd128(
        luma_ptr, rg_ptr, by_ptr, stride, best_tlx, best_tly + 1,
        tpl_luma, tpl_rg, tpl_by, tpl_weight, scale_px,
    );
    let mut sub_dy = 0.0f32;
    let denom_y = 2.0 * (c_ym - 2.0 * c_0 + c_yp);
    if denom_y > 1e-5 {
        sub_dy = ((c_ym - c_yp) / denom_y).clamp(-0.48, 0.48);
    }
    (sub_dx, sub_dy)
}

/// SIMD128 6x8x(7x9-1) coarse camera-pan tile search on the 160x90 luma pyramid.
/// Returns `(best_sad, best_dx, best_dy)`.
#[inline(always)]
unsafe fn eval_coarse_tile_vec12(
    cur: &[v128; 12],
    ps: &[v128; 12],
    tex_mad48: f32,
) -> f32 {
    let mut sum_ps = f32x4_splat(0.0);
    for k in 0..12usize {
        sum_ps = f32x4_add(sum_ps, ps[k]);
    }
    let mps = hsum_f32x4(sum_ps) * (1.0 / 48.0);
    let v_mps = f32x4_splat(mps);
    let mut mad_acc = f32x4_splat(0.0);
    let mut d_ps = [f32x4_splat(0.0); 12];
    for k in 0..12usize {
        let d = f32x4_sub(ps[k], v_mps);
        d_ps[k] = d;
        mad_acc = f32x4_add(mad_acc, f32x4_abs(d));
    }
    let mad_ps = hsum_f32x4(mad_acc);
    let g_shift = (tex_mad48 / mad_ps.max(1e-3)).clamp(0.75, 1.33);
    let v_g = f32x4_splat(g_shift);
    let mut sad_acc = f32x4_splat(0.0);
    for k in 0..12usize {
        let diff = f32x4_sub(cur[k], f32x4_mul(d_ps[k], v_g));
        sad_acc = f32x4_add(sad_acc, f32x4_abs(diff));
    }
    hsum_f32x4(sad_acc) * (1.0 / 48.0)
}

#[inline(always)]
unsafe fn eval_coarse_tile_vec12_sum4_presum(
    cur: &[v128; 12],
    ps: &[v128; 12],
    sum_ps: v128,
    tex_mad192: f32,
    best_sad48_tol: f32,
) -> f32 {
    let mps = hsum_f32x4(sum_ps) * (1.0 / 48.0);
    let v_mps = f32x4_splat(mps);
    let mut mad_acc = f32x4_splat(0.0);
    for k in 0..12usize {
        mad_acc = f32x4_add(mad_acc, f32x4_abs(f32x4_sub(ps[k], v_mps)));
    }
    let mad_ps = hsum_f32x4(mad_acc);
    let g_shift = (tex_mad192 / mad_ps.max(4e-3)).clamp(0.75, 1.33);
    let v_g = f32x4_splat(g_shift * 0.25);
    let mut sad_acc = f32x4_splat(0.0);
    for k in 0..3usize {
        let diff = f32x4_sub(cur[k], f32x4_mul(f32x4_sub(ps[k], v_mps), v_g));
        sad_acc = f32x4_add(sad_acc, f32x4_abs(diff));
    }
    if hsum_f32x4(sad_acc) >= best_sad48_tol {
        return 1e9f32;
    }
    for k in 3..6usize {
        let diff = f32x4_sub(cur[k], f32x4_mul(f32x4_sub(ps[k], v_mps), v_g));
        sad_acc = f32x4_add(sad_acc, f32x4_abs(diff));
    }
    if hsum_f32x4(sad_acc) >= best_sad48_tol {
        return 1e9f32;
    }
    for k in 6..12usize {
        let diff = f32x4_sub(cur[k], f32x4_mul(f32x4_sub(ps[k], v_mps), v_g));
        sad_acc = f32x4_add(sad_acc, f32x4_abs(diff));
    }
    hsum_f32x4(sad_acc) * (1.0 / 48.0)
}

pub unsafe fn eval_cam_coarse_3x3_simd128(
    cur_sub_ptr: *const f32,
    prev_pyr_ptr: *const f32,
    pyr_w: usize,
    pyr_h: usize,
    tx: usize,
    ty: usize,
    center_dx: i32,
    center_dy: i32,
    tex_mad: f32,
    zero_sad: f32,
    sad_cap: f32,
) -> (f32, i32, i32) {
    let mut best_sad = zero_sad;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let max_px = (pyr_w as i32) - 16;
    let max_py = (pyr_h as i32) - 12;

    let bx_i = (tx as i32) - center_dx - 1;
    let y_base_i = (ty as i32) - center_dy - 1;
    if bx_i >= 0 && bx_i + 18 <= (pyr_w as i32) && y_base_i >= 0 && y_base_i + 14 <= (pyr_h as i32)
    {
        let bx = bx_i as usize;
        let y_base = y_base_i as usize;
        let tex_mad192 = tex_mad * 192.0;
        let cur = [
            v128_load(cur_sub_ptr as *const v128),
            v128_load(cur_sub_ptr.add(4) as *const v128),
            v128_load(cur_sub_ptr.add(8) as *const v128),
            v128_load(cur_sub_ptr.add(12) as *const v128),
            v128_load(cur_sub_ptr.add(16) as *const v128),
            v128_load(cur_sub_ptr.add(20) as *const v128),
            v128_load(cur_sub_ptr.add(24) as *const v128),
            v128_load(cur_sub_ptr.add(28) as *const v128),
            v128_load(cur_sub_ptr.add(32) as *const v128),
            v128_load(cur_sub_ptr.add(36) as *const v128),
            v128_load(cur_sub_ptr.add(40) as *const v128),
            v128_load(cur_sub_ptr.add(44) as *const v128),
        ];
        macro_rules! hpair_sum {
            ($a:expr, $b:expr) => {
                f32x4_add(
                    i32x4_shuffle::<0, 2, 4, 6>($a, $b),
                    i32x4_shuffle::<1, 3, 5, 7>($a, $b),
                )
            };
        }
        let mut e_lo_uninit = [i32x4_splat(0); 13];
        let mut e_hi_uninit = [i32x4_splat(0); 13];
        let mut e2_hi_uninit = [i32x4_splat(0); 13];
        let mut o_lo_uninit = [i32x4_splat(0); 13];
        let mut o_hi_uninit = [i32x4_splat(0); 13];
        let e_lo = e_lo_uninit.as_mut_ptr() as *mut v128;
        let e_hi = e_hi_uninit.as_mut_ptr() as *mut v128;
        let e2_hi = e2_hi_uninit.as_mut_ptr() as *mut v128;
        let o_lo = o_lo_uninit.as_mut_ptr() as *mut v128;
        let o_hi = o_hi_uninit.as_mut_ptr() as *mut v128;

        let p0 = prev_pyr_ptr.add(y_base * pyr_w + bx);
        let mut r0 = v128_load(p0 as *const v128);
        let mut r1 = v128_load(p0.add(4) as *const v128);
        let mut r2 = v128_load(p0.add(8) as *const v128);
        let mut r3 = v128_load(p0.add(12) as *const v128);
        let mut r4 = v128_load(p0.add(14) as *const v128);
        for k in 0..13usize {
            let pk1 = prev_pyr_ptr.add((y_base + k + 1) * pyr_w + bx);
            let n0 = v128_load(pk1 as *const v128);
            let n1 = v128_load(pk1.add(4) as *const v128);
            let n2 = v128_load(pk1.add(8) as *const v128);
            let n3 = v128_load(pk1.add(12) as *const v128);
            let n4 = v128_load(pk1.add(14) as *const v128);
            let s0 = f32x4_add(r0, n0);
            let s1 = f32x4_add(r1, n1);
            let s2 = f32x4_add(r2, n2);
            let s3 = f32x4_add(r3, n3);
            let s4 = f32x4_add(r4, n4);

            let e0 = hpair_sum!(s0, s1);
            let e1 = hpair_sum!(s2, s3);
            let e_tail = hpair_sum!(s4, s4);
            *e_lo.add(k) = e0;
            *e_hi.add(k) = e1;
            *e2_hi.add(k) = i32x4_shuffle::<1, 2, 3, 5>(e1, e_tail);

            let t0 = i32x4_shuffle::<1, 2, 3, 4>(s0, s1);
            let t1 = i32x4_shuffle::<1, 2, 3, 4>(s1, s2);
            let t2 = i32x4_shuffle::<1, 2, 3, 4>(s2, s3);
            let t3 = i32x4_shuffle::<1, 2, 3, 6>(s3, s4);
            *o_lo.add(k) = hpair_sum!(t0, t1);
            *o_hi.add(k) = hpair_sum!(t2, t3);

            r0 = n0;
            r1 = n1;
            r2 = n2;
            r3 = n3;
            r4 = n4;
        }

        let mut ps = [f32x4_splat(0.0); 12];
        let cap_init = sad_cap.min(best_sad);
        let mut best_sad48_tol = cap_init * 48.0 + 1e-4;
        for &k0 in &[1usize, 0, 2] {
            let dy = center_dy + 1 - (k0 as i32);
            for &x_off in &[1usize, 0, 2] {
                let dx = center_dx + 1 - (x_off as i32);
                if dx != 0 || dy != 0 {
                    let mut sum_ps = f32x4_splat(0.0);
                    for r in 0..6usize {
                        let b_r = k0 + r * 2;
                        let (lo, hi) = if x_off == 1 {
                            (*o_lo.add(b_r), *o_hi.add(b_r))
                        } else if x_off == 0 {
                            (*e_lo.add(b_r), *e_hi.add(b_r))
                        } else {
                            let a = *e_lo.add(b_r);
                            let b = *e_hi.add(b_r);
                            (i32x4_shuffle::<1, 2, 3, 4>(a, b), *e2_hi.add(b_r))
                        };
                        ps[r * 2] = lo;
                        ps[r * 2 + 1] = hi;
                        sum_ps = f32x4_add(sum_ps, f32x4_add(lo, hi));
                    }
                    let sad = eval_coarse_tile_vec12_sum4_presum(
                        &cur,
                        &ps,
                        sum_ps,
                        tex_mad192,
                        best_sad48_tol,
                    );
                    if sad < best_sad && sad < cap_init {
                        best_sad = sad;
                        best_sad48_tol = sad * 48.0 + 1e-4;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
        }
        return (best_sad, best_dx, best_dy);
    }

    let tex_mad48 = tex_mad * 48.0;
    let v_quarter = f32x4_splat(0.25);
    for dy in (center_dy - 1)..=(center_dy + 1) {
        let pty_i = (ty as i32) - dy;
        if pty_i < 0 || pty_i > max_py {
            continue;
        }
        let pty = pty_i as usize;
        for dx in (center_dx - 1)..=(center_dx + 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let ptx_i = (tx as i32) - dx;
            if ptx_i < 0 || ptx_i > max_px {
                continue;
            }
            let ptx = ptx_i as usize;
            let mut ps = [f32x4_splat(0.0); 12];
            let mut sum_ps = f32x4_splat(0.0);
            for r in 0..6usize {
                let prow0 = (pty + r * 2) * pyr_w + ptx;
                let prow1 = prow0 + pyr_w;
                let a0 = v128_load(prev_pyr_ptr.add(prow0) as *const v128);
                let a1 = v128_load(prev_pyr_ptr.add(prow0 + 4) as *const v128);
                let a2 = v128_load(prev_pyr_ptr.add(prow0 + 8) as *const v128);
                let a3 = v128_load(prev_pyr_ptr.add(prow0 + 12) as *const v128);
                let b0 = v128_load(prev_pyr_ptr.add(prow1) as *const v128);
                let b1 = v128_load(prev_pyr_ptr.add(prow1 + 4) as *const v128);
                let b2 = v128_load(prev_pyr_ptr.add(prow1 + 8) as *const v128);
                let b3 = v128_load(prev_pyr_ptr.add(prow1 + 12) as *const v128);
                let s0 = f32x4_add(a0, b0);
                let s1 = f32x4_add(a1, b1);
                let s2 = f32x4_add(a2, b2);
                let s3 = f32x4_add(a3, b3);
                let lo = f32x4_mul(
                    f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(s0, s1),
                        i32x4_shuffle::<1, 3, 5, 7>(s0, s1),
                    ),
                    v_quarter,
                );
                let hi = f32x4_mul(
                    f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(s2, s3),
                        i32x4_shuffle::<1, 3, 5, 7>(s2, s3),
                    ),
                    v_quarter,
                );
                ps[r * 2] = lo;
                ps[r * 2 + 1] = hi;
                sum_ps = f32x4_add(sum_ps, f32x4_add(lo, hi));
            }
            let mps = hsum_f32x4(sum_ps) * (1.0 / 48.0);
            let v_mps = f32x4_splat(mps);
            let mut acc_mad = f32x4_splat(0.0);
            for k in 0..12usize {
                let d = f32x4_sub(ps[k], v_mps);
                ps[k] = d;
                acc_mad = f32x4_add(acc_mad, f32x4_abs(d));
            }
            let mad_ps = hsum_f32x4(acc_mad);
            let g_shift = (tex_mad48 / mad_ps.max(1e-3)).clamp(0.75, 1.33);
            let v_g = f32x4_splat(g_shift);
            let mut acc_sad = f32x4_splat(0.0);
            for k in 0..12usize {
                let vc = v128_load(cur_sub_ptr.add(k * 4) as *const v128);
                let diff = f32x4_sub(vc, f32x4_mul(ps[k], v_g));
                acc_sad = f32x4_add(acc_sad, f32x4_abs(diff));
            }
            let sad = hsum_f32x4(acc_sad) * (1.0 / 48.0);
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    (best_sad, best_dx, best_dy)
}

pub unsafe fn eval_cam_coarse_tile_simd128(
    cur_sub_ptr: *const f32,
    prev_pyr_ptr: *const f32,
    pyr_w: usize,
    pyr_h: usize,
    tx: usize,
    ty: usize,
    center_dx: i32,
    center_dy: i32,
    tex_mad: f32,
    zero_sad: f32,
    sad_cap: f32,
) -> (f32, i32, i32) {
    let mut best_sad = zero_sad;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let tex_mad48 = tex_mad * 48.0;
    let max_px = (pyr_w as i32) - 16;
    let max_py = (pyr_h as i32) - 12;

    let bx_i = (tx as i32) - center_dx - 4;
    if bx_i >= 0 && bx_i <= max_px - 8 && (ty as i32) - center_dy - 3 >= 0 && (ty as i32) - center_dy + 3 <= max_py {
        let bx = bx_i as usize;
        let tex_mad192 = tex_mad * 192.0;
        let cur = [
            v128_load(cur_sub_ptr as *const v128),
            v128_load(cur_sub_ptr.add(4) as *const v128),
            v128_load(cur_sub_ptr.add(8) as *const v128),
            v128_load(cur_sub_ptr.add(12) as *const v128),
            v128_load(cur_sub_ptr.add(16) as *const v128),
            v128_load(cur_sub_ptr.add(20) as *const v128),
            v128_load(cur_sub_ptr.add(24) as *const v128),
            v128_load(cur_sub_ptr.add(28) as *const v128),
            v128_load(cur_sub_ptr.add(32) as *const v128),
            v128_load(cur_sub_ptr.add(36) as *const v128),
            v128_load(cur_sub_ptr.add(40) as *const v128),
            v128_load(cur_sub_ptr.add(44) as *const v128),
        ];
        macro_rules! hpair_sum {
            ($a:expr, $b:expr) => {
                f32x4_add(
                    i32x4_shuffle::<0, 2, 4, 6>($a, $b),
                    i32x4_shuffle::<1, 3, 5, 7>($a, $b),
                )
            };
        }
        let y_base = ((ty as i32) - center_dy - 3) as usize;
        let mut all_e_uninit = [i32x4_splat(0); 51];
        let mut all_o_uninit = [i32x4_splat(0); 51];
        let all_e = all_e_uninit.as_mut_ptr() as *mut v128;
        let all_o = all_o_uninit.as_mut_ptr() as *mut v128;
        let p0 = prev_pyr_ptr.add(y_base * pyr_w + bx);
        let mut r0 = v128_load(p0 as *const v128);
        let mut r1 = v128_load(p0.add(4) as *const v128);
        let mut r2 = v128_load(p0.add(8) as *const v128);
        let mut r3 = v128_load(p0.add(12) as *const v128);
        let mut r4 = v128_load(p0.add(16) as *const v128);
        let mut r5 = v128_load(p0.add(20) as *const v128);
        for k in 0..17usize {
            let pk1 = prev_pyr_ptr.add((y_base + k + 1) * pyr_w + bx);
            let n0 = v128_load(pk1 as *const v128);
            let n1 = v128_load(pk1.add(4) as *const v128);
            let n2 = v128_load(pk1.add(8) as *const v128);
            let n3 = v128_load(pk1.add(12) as *const v128);
            let n4 = v128_load(pk1.add(16) as *const v128);
            let n5 = v128_load(pk1.add(20) as *const v128);
            let s0 = f32x4_add(r0, n0);
            let s1 = f32x4_add(r1, n1);
            let s2 = f32x4_add(r2, n2);
            let s3 = f32x4_add(r3, n3);
            let s4 = f32x4_add(r4, n4);
            let s5 = f32x4_add(r5, n5);
            let bk = k * 3;
            *all_e.add(bk) = hpair_sum!(s0, s1);
            *all_e.add(bk + 1) = hpair_sum!(s2, s3);
            *all_e.add(bk + 2) = hpair_sum!(s4, s5);

            let t0 = i32x4_shuffle::<1, 2, 3, 4>(s0, s1);
            let t1 = i32x4_shuffle::<1, 2, 3, 4>(s1, s2);
            let t2 = i32x4_shuffle::<1, 2, 3, 4>(s2, s3);
            let t3 = i32x4_shuffle::<1, 2, 3, 4>(s3, s4);
            let t4 = i32x4_shuffle::<1, 2, 3, 4>(s4, s5);
            let t5 = i32x4_shuffle::<1, 2, 3, 4>(s5, s5);
            *all_o.add(bk) = hpair_sum!(t0, t1);
            *all_o.add(bk + 1) = hpair_sum!(t2, t3);
            *all_o.add(bk + 2) = hpair_sum!(t4, t5);
            r0 = n0;
            r1 = n1;
            r2 = n2;
            r3 = n3;
            r4 = n4;
            r5 = n5;
        }
        let mut ps = [f32x4_splat(0.0); 12];
        let cap_init = sad_cap.min(best_sad);
        let mut best_sad48_tol = cap_init * 48.0 + 1e-4;
        for &k0 in &[3usize, 2, 4, 1, 5, 0, 6] {
            let dy = center_dy + 3 - (k0 as i32);
            for &step_e in &[2usize, 1, 3, 0, 4] {
                let dx = center_dx + 4 - (step_e as i32) * 2;
                if dx != 0 || dy != 0 {
                    let mut sum_ps = f32x4_splat(0.0);
                    for r in 0..6usize {
                        let b_r = (k0 + r * 2) * 3;
                        let a = *all_e.add(b_r);
                        let b = *all_e.add(b_r + 1);
                        let c = *all_e.add(b_r + 2);
                        let (lo, hi) = match step_e {
                            0 => (a, b),
                            1 => (i32x4_shuffle::<1, 2, 3, 4>(a, b), i32x4_shuffle::<1, 2, 3, 4>(b, c)),
                            2 => (i32x4_shuffle::<2, 3, 4, 5>(a, b), i32x4_shuffle::<2, 3, 4, 5>(b, c)),
                            3 => (i32x4_shuffle::<3, 4, 5, 6>(a, b), i32x4_shuffle::<3, 4, 5, 6>(b, c)),
                            _ => (b, c),
                        };
                        ps[r * 2] = lo;
                        ps[r * 2 + 1] = hi;
                        sum_ps = f32x4_add(sum_ps, f32x4_add(lo, hi));
                    }
                    let sad = eval_coarse_tile_vec12_sum4_presum(&cur, &ps, sum_ps, tex_mad192, best_sad48_tol);
                    if sad < best_sad && sad < cap_init {
                        best_sad = sad;
                        best_sad48_tol = sad * 48.0 + 1e-4;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
            for &step_o in &[1usize, 2, 0, 3] {
                let dx = center_dx + 3 - (step_o as i32) * 2;
                if dx != 0 || dy != 0 {
                    let mut sum_ps = f32x4_splat(0.0);
                    for r in 0..6usize {
                        let b_r = (k0 + r * 2) * 3;
                        let a = *all_o.add(b_r);
                        let b = *all_o.add(b_r + 1);
                        let c = *all_o.add(b_r + 2);
                        let (lo, hi) = match step_o {
                            0 => (a, b),
                            1 => (i32x4_shuffle::<1, 2, 3, 4>(a, b), i32x4_shuffle::<1, 2, 3, 4>(b, c)),
                            2 => (i32x4_shuffle::<2, 3, 4, 5>(a, b), i32x4_shuffle::<2, 3, 4, 5>(b, c)),
                            _ => (i32x4_shuffle::<3, 4, 5, 6>(a, b), i32x4_shuffle::<3, 4, 5, 6>(b, c)),
                        };
                        ps[r * 2] = lo;
                        ps[r * 2 + 1] = hi;
                        sum_ps = f32x4_add(sum_ps, f32x4_add(lo, hi));
                    }
                    let sad = eval_coarse_tile_vec12_sum4_presum(&cur, &ps, sum_ps, tex_mad192, best_sad48_tol);
                    if sad < best_sad && sad < cap_init {
                        best_sad = sad;
                        best_sad48_tol = sad * 48.0 + 1e-4;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
        }
        return (best_sad, best_dx, best_dy);
    }
    let v_quarter = f32x4_splat(0.25);

    for dy in (center_dy - 3)..=(center_dy + 3) {
        let pty_i = (ty as i32) - dy;
        if pty_i < 0 || pty_i > max_py {
            continue;
        }
        let pty = pty_i as usize;
        for dx in (center_dx - 4)..=(center_dx + 4) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let ptx_i = (tx as i32) - dx;
            if ptx_i < 0 || ptx_i > max_px {
                continue;
            }
            let ptx = ptx_i as usize;
            let mut ps = [f32x4_splat(0.0); 12];
            let mut sum_ps = f32x4_splat(0.0);

            for r in 0..6usize {
                let prow0 = (pty + r * 2) * pyr_w + ptx;
                let prow1 = prow0 + pyr_w;
                let a0 = v128_load(prev_pyr_ptr.add(prow0) as *const v128);
                let a1 = v128_load(prev_pyr_ptr.add(prow0 + 4) as *const v128);
                let a2 = v128_load(prev_pyr_ptr.add(prow0 + 8) as *const v128);
                let a3 = v128_load(prev_pyr_ptr.add(prow0 + 12) as *const v128);
                let b0 = v128_load(prev_pyr_ptr.add(prow1) as *const v128);
                let b1 = v128_load(prev_pyr_ptr.add(prow1 + 4) as *const v128);
                let b2 = v128_load(prev_pyr_ptr.add(prow1 + 8) as *const v128);
                let b3 = v128_load(prev_pyr_ptr.add(prow1 + 12) as *const v128);

                let s0 = f32x4_add(a0, b0);
                let s1 = f32x4_add(a1, b1);
                let s2 = f32x4_add(a2, b2);
                let s3 = f32x4_add(a3, b3);

                let lo = f32x4_mul(
                    f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(s0, s1),
                        i32x4_shuffle::<1, 3, 5, 7>(s0, s1),
                    ),
                    v_quarter,
                );
                let hi = f32x4_mul(
                    f32x4_add(
                        i32x4_shuffle::<0, 2, 4, 6>(s2, s3),
                        i32x4_shuffle::<1, 3, 5, 7>(s2, s3),
                    ),
                    v_quarter,
                );
                ps[r * 2] = lo;
                ps[r * 2 + 1] = hi;
                sum_ps = f32x4_add(sum_ps, f32x4_add(lo, hi));
            }

            let mps = hsum_f32x4(sum_ps) * (1.0 / 48.0);
            let v_mps = f32x4_splat(mps);
            let mut acc_mad = f32x4_splat(0.0);
            for k in 0..12usize {
                let d = f32x4_sub(ps[k], v_mps);
                ps[k] = d;
                acc_mad = f32x4_add(acc_mad, f32x4_abs(d));
            }
            let mad_ps = hsum_f32x4(acc_mad);
            let g_shift = (tex_mad48 / mad_ps.max(1e-3)).clamp(0.75, 1.33);
            let v_g = f32x4_splat(g_shift);

            let mut acc_sad = f32x4_splat(0.0);
            for k in 0..12usize {
                let vc = v128_load(cur_sub_ptr.add(k * 4) as *const v128);
                let diff = f32x4_sub(vc, f32x4_mul(ps[k], v_g));
                acc_sad = f32x4_add(acc_sad, f32x4_abs(diff));
            }
            let sad = hsum_f32x4(acc_sad) * (1.0 / 48.0);
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    (best_sad, best_dx, best_dy)
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
pub unsafe fn filter_bg_eval_mask_luma_simd128(
    bg_eval_mask: *mut u8,
    luma: *const f32,
    bg_luma: *const f32,
    count: usize,
    tol: f32,
) {
    let tol_v = f32x4_splat(tol);
    let mut i = 0usize;
    while i + 16 <= count {
        let m_ptr = bg_eval_mask.add(i) as *mut v128;
        let m16 = v128_load(m_ptr as *const v128);
        if v128_any_true(m16) {
            let d0 = f32x4_abs(f32x4_sub(
                v128_load(luma.add(i) as *const v128),
                v128_load(bg_luma.add(i) as *const v128),
            ));
            let d1 = f32x4_abs(f32x4_sub(
                v128_load(luma.add(i + 4) as *const v128),
                v128_load(bg_luma.add(i + 4) as *const v128),
            ));
            let d2 = f32x4_abs(f32x4_sub(
                v128_load(luma.add(i + 8) as *const v128),
                v128_load(bg_luma.add(i + 8) as *const v128),
            ));
            let d3 = f32x4_abs(f32x4_sub(
                v128_load(luma.add(i + 12) as *const v128),
                v128_load(bg_luma.add(i + 12) as *const v128),
            ));
            let o0 = f32x4_gt(d0, tol_v);
            let o1 = f32x4_gt(d1, tol_v);
            let o2 = f32x4_gt(d2, tol_v);
            let o3 = f32x4_gt(d3, tol_v);
            let over16 = i8x16_narrow_i16x8(
                i16x8_narrow_i32x4(o0, o1),
                i16x8_narrow_i32x4(o2, o3),
            );
            v128_store(m_ptr, v128_andnot(m16, over16));
        }
        i += 16;
    }
    while i < count {
        if *bg_eval_mask.add(i) != 0 && (*luma.add(i) - *bg_luma.add(i)).abs() > tol {
            *bg_eval_mask.add(i) = 0;
        }
        i += 1;
    }
}

#[inline(always)]
unsafe fn patch_grad_tensor_sd7_simd128(
    l_ptr: *const f32,
    frame_w: usize,
    cx: usize,
    cy: usize,
) -> (f32, f32, f32, f32) {
    let zero = f32x4_splat(0.0);
    let mask3 = u32x4(0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0);
    let mut sxx_v = zero;
    let mut syy_v = zero;
    let mut sxy_v = zero;
    let mut l1_v = zero;
    let mut roff = cy * frame_w + cx;
    for _ in 0..7usize {
        let p = l_ptr.add(roff);
        let v0 = v128_load(p as *const v128);
        let vx0 = v128_load(p.add(1) as *const v128);
        let vy0 = v128_load(p.add(frame_w) as *const v128);
        let dx0 = f32x4_sub(vx0, v0);
        let dy0 = f32x4_sub(vy0, v0);
        sxx_v = f32x4_add(sxx_v, f32x4_mul(dx0, dx0));
        syy_v = f32x4_add(syy_v, f32x4_mul(dy0, dy0));
        sxy_v = f32x4_add(sxy_v, f32x4_mul(dx0, dy0));
        l1_v = f32x4_add(l1_v, f32x4_add(f32x4_abs(dx0), f32x4_abs(dy0)));

        let v1 = v128_load(p.add(4) as *const v128);
        let vx1 = v128_load(p.add(5) as *const v128);
        let vy1 = v128_load(p.add(frame_w + 4) as *const v128);
        let dx1 = v128_and(f32x4_sub(vx1, v1), mask3);
        let dy1 = v128_and(f32x4_sub(vy1, v1), mask3);
        sxx_v = f32x4_add(sxx_v, f32x4_mul(dx1, dx1));
        syy_v = f32x4_add(syy_v, f32x4_mul(dy1, dy1));
        sxy_v = f32x4_add(sxy_v, f32x4_mul(dx1, dy1));
        l1_v = f32x4_add(l1_v, f32x4_add(f32x4_abs(dx1), f32x4_abs(dy1)));

        roff += frame_w;
    }
    (
        hsum_f32x4(sxx_v),
        hsum_f32x4(syy_v),
        hsum_f32x4(sxy_v),
        hsum_f32x4(l1_v),
    )
}

#[inline(always)]
unsafe fn patch_grad_tensor_hd8_simd128(
    l_ptr: *const f32,
    frame_w: usize,
    cx: usize,
    cy: usize,
) -> (f32, f32, f32, f32) {
    let zero = f32x4_splat(0.0);
    let mask3 = u32x4(0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0);
    let mut sxx_0 = zero;
    let mut sxx_1 = zero;
    let mut syy_0 = zero;
    let mut syy_1 = zero;
    let mut sxy_0 = zero;
    let mut sxy_1 = zero;
    let mut l1_0 = zero;
    let mut l1_1 = zero;
    let stride2 = frame_w * 2;
    let mut roff = cy * frame_w + cx;
    for _ in 0..8usize {
        let p = l_ptr.add(roff);
        let q = p.add(frame_w);

        let v0 = v128_load(p as *const v128);
        let v1 = v128_load(p.add(4) as *const v128);
        let v2 = v128_load(p.add(8) as *const v128);
        let v3 = v128_load(p.add(12) as *const v128);

        let vx0 = i32x4_shuffle::<1, 2, 3, 4>(v0, v1);
        let vy0 = v128_load(q as *const v128);
        let dx0 = f32x4_sub(vx0, v0);
        let dy0 = f32x4_sub(vy0, v0);
        sxx_0 = f32x4_add(sxx_0, f32x4_mul(dx0, dx0));
        syy_0 = f32x4_add(syy_0, f32x4_mul(dy0, dy0));
        sxy_0 = f32x4_add(sxy_0, f32x4_mul(dx0, dy0));
        l1_0 = f32x4_add(l1_0, f32x4_add(f32x4_abs(dx0), f32x4_abs(dy0)));

        let vx1 = i32x4_shuffle::<1, 2, 3, 4>(v1, v2);
        let vy1 = v128_load(q.add(4) as *const v128);
        let dx1 = f32x4_sub(vx1, v1);
        let dy1 = f32x4_sub(vy1, v1);
        sxx_1 = f32x4_add(sxx_1, f32x4_mul(dx1, dx1));
        syy_1 = f32x4_add(syy_1, f32x4_mul(dy1, dy1));
        sxy_1 = f32x4_add(sxy_1, f32x4_mul(dx1, dy1));
        l1_1 = f32x4_add(l1_1, f32x4_add(f32x4_abs(dx1), f32x4_abs(dy1)));

        let vx2 = i32x4_shuffle::<1, 2, 3, 4>(v2, v3);
        let vy2 = v128_load(q.add(8) as *const v128);
        let dx2 = f32x4_sub(vx2, v2);
        let dy2 = f32x4_sub(vy2, v2);
        sxx_0 = f32x4_add(sxx_0, f32x4_mul(dx2, dx2));
        syy_0 = f32x4_add(syy_0, f32x4_mul(dy2, dy2));
        sxy_0 = f32x4_add(sxy_0, f32x4_mul(dx2, dy2));
        l1_0 = f32x4_add(l1_0, f32x4_add(f32x4_abs(dx2), f32x4_abs(dy2)));

        let vx3 = i32x4_shuffle::<1, 2, 3, 3>(v3, v3);
        let vy3 = v128_load(q.add(12) as *const v128);
        let dx3 = v128_and(f32x4_sub(vx3, v3), mask3);
        let dy3 = v128_and(f32x4_sub(vy3, v3), mask3);
        sxx_1 = f32x4_add(sxx_1, f32x4_mul(dx3, dx3));
        syy_1 = f32x4_add(syy_1, f32x4_mul(dy3, dy3));
        sxy_1 = f32x4_add(sxy_1, f32x4_mul(dx3, dy3));
        l1_1 = f32x4_add(l1_1, f32x4_add(f32x4_abs(dx3), f32x4_abs(dy3)));

        roff += stride2;
    }
    const SCALE: f32 = 15.0 / 8.0;
    (
        hsum_f32x4(f32x4_add(sxx_0, sxx_1)) * SCALE,
        hsum_f32x4(f32x4_add(syy_0, syy_1)) * SCALE,
        hsum_f32x4(f32x4_add(sxy_0, sxy_1)) * SCALE,
        hsum_f32x4(f32x4_add(l1_0, l1_1)) * SCALE,
    )
}

#[inline(never)]
pub unsafe fn find_top_2d_patches_simd128(
    luma: &[f32],
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    min_t2d: f32,
    min_mad: f32,
    excl_x: &[f32; 32],
    excl_y: &[f32; 32],
    n_excl: usize,
    out_bx: &mut [usize; 16],
    out_by: &mut [usize; 16],
) -> usize {
    if frame_w <= 12 * scale_px || frame_h <= 12 * scale_px || luma.len() < frame_w * frame_h {
        return 0;
    }
    let step = 9 * scale_px;
    let span = 8 * scale_px - 1;
    let min_x = 2 * scale_px;
    let max_x = frame_w - 10 * scale_px;
    let min_y = 2 * scale_px;
    let max_y = frame_h - 10 * scale_px;
    let phase_mask = scale_px - 1;
    let half_span = 4 * scale_px;
    let fg_excl_r = (18 * scale_px) as f32;
    let fg_excl_r2 = fg_excl_r * fg_excl_r;
    let sep_r2 = ((20 * scale_px) as f32) * ((20 * scale_px) as f32);
    let mut out_t2d = [0.0f32; 16];
    let mut cnt = 0usize;
    let mut min_k = 0usize;
    let mut cur_min_t2d = min_t2d;
    let l_ptr = luma.as_ptr();

    let mut row_excl_x = [0.0f32; 32];
    let mut row_excl_dy2 = [0.0f32; 32];

    let mut cy = min_y;
    while cy <= max_y {
        let pcy = (cy + half_span) as f32;
        let mut row_excl_cnt = 0usize;
        for p in 0..n_excl {
            let dy = pcy - *excl_y.get_unchecked(p);
            let dy2 = dy * dy;
            if dy2 < fg_excl_r2 {
                *row_excl_x.get_unchecked_mut(row_excl_cnt) = *excl_x.get_unchecked(p);
                *row_excl_dy2.get_unchecked_mut(row_excl_cnt) = dy2;
                row_excl_cnt += 1;
            }
        }
        let mut cx = min_x;
        while cx <= max_x {
            let pcx = (cx + half_span) as f32;
            let mut near_fg = false;
            for p in 0..row_excl_cnt {
                let dx = pcx - *row_excl_x.get_unchecked(p);
                if dx * dx + *row_excl_dy2.get_unchecked(p) < fg_excl_r2 {
                    near_fg = true;
                    break;
                }
            }
            if near_fg {
                cx += step;
                continue;
            }

            let (sxx, syy, sxy, l1) = if scale_px == 1 {
                patch_grad_tensor_sd7_simd128(l_ptr, frame_w, cx, cy)
            } else if scale_px == 2 {
                patch_grad_tensor_hd8_simd128(l_ptr, frame_w, cx, cy)
            } else {
                patch_grad_tensor_span_simd128(luma, frame_w, cx, cy, span)
            };

            let upper_t2d = 0.5 * l1;
            if (cnt < 16 && upper_t2d < min_t2d) || (cnt == 16 && upper_t2d <= cur_min_t2d) {
                cx += step;
                continue;
            }

            let tr = sxx + syy;
            let det = (sxx * syy - sxy * sxy).max(0.0);
            if det < 0.0685871 * tr * tr {
                cx += step;
                continue;
            }
            let target_t2d = if cnt == 16 { cur_min_t2d } else { min_t2d };
            if target_t2d > 0.0 {
                let need = (target_t2d * 0.999) / upper_t2d;
                let r_req = need * need;
                if det * (1.0 + r_req) * (1.0 + r_req) < r_req * tr * tr {
                    cx += step;
                    continue;
                }
            }

            let disc = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat((tr * tr - 4.0 * det).max(0.0))));
            let lam_min = 0.5 * (tr - disc);
            let lam_max = 0.5 * (tr + disc);
            if lam_min < 0.08 * lam_max {
                cx += step;
                continue;
            }
            let ratio_sqrt = f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(lam_min / lam_max.max(1e-8))));
            let t2d = upper_t2d * ratio_sqrt;
            if (cnt < 16 && t2d < min_t2d) || (cnt == 16 && t2d <= cur_min_t2d) {
                cx += step;
                continue;
            }

            let mut near_k = usize::MAX;
            for k in 0..cnt {
                let dx = (out_bx[k] as f32) - (cx as f32);
                let dy = (out_by[k] as f32) - (cy as f32);
                if dx * dx + dy * dy < sep_r2 {
                    near_k = k;
                    break;
                }
            }
            if near_k != usize::MAX && t2d <= out_t2d[near_k] {
                cx += step;
                continue;
            }

            let mut v_arr = [f32x4_splat(0.0); 16];
            let mut sum_v = f32x4_splat(0.0);
            if scale_px <= 1 {
                for r in 0..8usize {
                    let row_p = l_ptr.add((cy + r) * frame_w + cx);
                    let v0 = v128_load(row_p as *const v128);
                    let v1 = v128_load(row_p.add(4) as *const v128);
                    v_arr[r * 2] = v0;
                    v_arr[r * 2 + 1] = v1;
                    sum_v = f32x4_add(sum_v, f32x4_add(v0, v1));
                }
            } else if scale_px == 2 {
                for r in 0..8usize {
                    let row_e = l_ptr.add((cy + r * 2) * frame_w + cx);
                    let row_o = row_e.add(frame_w);
                    let e0 = v128_load(row_e as *const v128);
                    let e1 = v128_load(row_e.add(4) as *const v128);
                    let e2 = v128_load(row_e.add(8) as *const v128);
                    let e3 = v128_load(row_e.add(12) as *const v128);
                    let o0 = v128_load(row_o as *const v128);
                    let o1 = v128_load(row_o.add(4) as *const v128);
                    let o2 = v128_load(row_o.add(8) as *const v128);
                    let o3 = v128_load(row_o.add(12) as *const v128);
                    let (v0, v1) = if (r & 1) == 0 {
                        let ev0 = i32x4_shuffle::<0, 2, 4, 6>(e0, e1);
                        let ov0 = i32x4_shuffle::<0, 2, 4, 6>(o0, o1);
                        let ev1 = i32x4_shuffle::<0, 2, 4, 6>(e2, e3);
                        let ov1 = i32x4_shuffle::<0, 2, 4, 6>(o2, o3);
                        (
                            i32x4_shuffle::<0, 5, 2, 7>(ev0, ov0),
                            i32x4_shuffle::<0, 5, 2, 7>(ev1, ov1),
                        )
                    } else {
                        let ev0 = i32x4_shuffle::<1, 3, 5, 7>(e0, e1);
                        let ov0 = i32x4_shuffle::<1, 3, 5, 7>(o0, o1);
                        let ev1 = i32x4_shuffle::<1, 3, 5, 7>(e2, e3);
                        let ov1 = i32x4_shuffle::<1, 3, 5, 7>(o2, o3);
                        (
                            i32x4_shuffle::<0, 5, 2, 7>(ev0, ov0),
                            i32x4_shuffle::<0, 5, 2, 7>(ev1, ov1),
                        )
                    };
                    v_arr[r * 2] = v0;
                    v_arr[r * 2 + 1] = v1;
                    sum_v = f32x4_add(sum_v, f32x4_add(v0, v1));
                }
            } else {
                let val_p = v_arr.as_mut_ptr() as *mut f32;
                let mut q = 0usize;
                for r in 0..8usize {
                    let by_r = cy + r * scale_px;
                    let rx_base = cx + (r & phase_mask);
                    for c in 0..8usize {
                        let ry = by_r + (c & phase_mask);
                        let rx = rx_base + c * scale_px;
                        *val_p.add(q) = *l_ptr.add(ry * frame_w + rx);
                        q += 1;
                    }
                }
                for i in 0..16usize {
                    sum_v = f32x4_add(sum_v, v_arr[i]);
                }
            }
            let mp = hsum_f32x4(sum_v) * (1.0 / 64.0);
            let v_mp = f32x4_splat(mp);
            let mut acc_mad = f32x4_splat(0.0);
            for i in 0..16usize {
                acc_mad = f32x4_add(acc_mad, f32x4_abs(f32x4_sub(v_arr[i], v_mp)));
            }
            let mad_c = hsum_f32x4(acc_mad);
            if mad_c >= min_mad {
                if near_k != usize::MAX {
                    out_bx[near_k] = cx;
                    out_by[near_k] = cy;
                    out_t2d[near_k] = t2d;
                    if cnt == 16 && near_k == min_k {
                        let mut mk = 0usize;
                        for k in 1..16usize {
                            if out_t2d[k] < out_t2d[mk] {
                                mk = k;
                            }
                        }
                        min_k = mk;
                        cur_min_t2d = out_t2d[mk];
                    }
                } else if cnt < 16 {
                    out_bx[cnt] = cx;
                    out_by[cnt] = cy;
                    out_t2d[cnt] = t2d;
                    cnt += 1;
                    if cnt == 16 {
                        let mut mk = 0usize;
                        for k in 1..16usize {
                            if out_t2d[k] < out_t2d[mk] {
                                mk = k;
                            }
                        }
                        min_k = mk;
                        cur_min_t2d = out_t2d[mk];
                    }
                } else {
                    out_bx[min_k] = cx;
                    out_by[min_k] = cy;
                    out_t2d[min_k] = t2d;
                    let mut mk = 0usize;
                    for k in 1..16usize {
                        if out_t2d[k] < out_t2d[mk] {
                            mk = k;
                        }
                    }
                    min_k = mk;
                    cur_min_t2d = out_t2d[mk];
                }
            }
            cx += step;
        }
        cy += step;
    }

    for i in 1..cnt {
        let kx = out_bx[i];
        let ky = out_by[i];
        let kt = out_t2d[i];
        let mut j = i;
        while j > 0 && out_t2d[j - 1] < kt {
            out_bx[j] = out_bx[j - 1];
            out_by[j] = out_by[j - 1];
            out_t2d[j] = out_t2d[j - 1];
            j -= 1;
        }
        out_bx[j] = kx;
        out_by[j] = ky;
        out_t2d[j] = kt;
    }
    cnt
}

/// Computes `(core_eca_sum, inner_eca_sum, core_eca_max)` over `[2*s .. 14*s) x [2*s .. 14*s)`
/// and inner `[5*s .. 11*s) x [5*s .. 11*s)` at `(qx, qy)` in `eca_res`.
#[inline(always)]
pub unsafe fn patch_eca_core_inner_max_simd128(
    eca_ptr: *const f32,
    frame_w: usize,
    qx: usize,
    qy: usize,
    scale_px: usize,
) -> (f32, f32, f32) {
    if scale_px == 2 {
        let mut acc_core = f32x4_splat(0.0);
        let mut acc_inner = f32x4_splat(0.0);
        let mut acc_max = f32x4_splat(0.0);

        for ry in 4..10usize {
            let rp = eca_ptr.add((qy + ry) * frame_w + qx);
            let v0 = v128_load(rp.add(4) as *const v128);
            let v1 = v128_load(rp.add(8) as *const v128);
            let v2 = v128_load(rp.add(12) as *const v128);
            let v3 = v128_load(rp.add(16) as *const v128);
            let v4 = v128_load(rp.add(20) as *const v128);
            let v5 = v128_load(rp.add(24) as *const v128);
            let s05 = f32x4_add(v0, v5);
            let s14 = f32x4_add(v1, v4);
            let s23 = f32x4_add(v2, v3);
            acc_core = f32x4_add(acc_core, f32x4_add(s05, f32x4_add(s14, s23)));
            let m05 = f32x4_pmax(v0, v5);
            let m14 = f32x4_pmax(v1, v4);
            let m23 = f32x4_pmax(v2, v3);
            acc_max = f32x4_pmax(acc_max, f32x4_pmax(m05, f32x4_pmax(m14, m23)));
        }
        for ry in 10..22usize {
            let rp = eca_ptr.add((qy + ry) * frame_w + qx);
            let v0 = v128_load(rp.add(4) as *const v128);
            let v1 = v128_load(rp.add(8) as *const v128);
            let v2 = v128_load(rp.add(12) as *const v128);
            let v3 = v128_load(rp.add(16) as *const v128);
            let v4 = v128_load(rp.add(20) as *const v128);
            let v5 = v128_load(rp.add(24) as *const v128);
            let s05 = f32x4_add(v0, v5);
            let s14 = f32x4_add(v1, v4);
            let s23 = f32x4_add(v2, v3);
            acc_core = f32x4_add(acc_core, f32x4_add(s05, f32x4_add(s14, s23)));
            let edges = i32x4_shuffle::<2, 3, 4, 5>(v1, v4);
            acc_inner = f32x4_add(acc_inner, f32x4_add(s23, edges));
            let m05 = f32x4_pmax(v0, v5);
            let m14 = f32x4_pmax(v1, v4);
            let m23 = f32x4_pmax(v2, v3);
            acc_max = f32x4_pmax(acc_max, f32x4_pmax(m05, f32x4_pmax(m14, m23)));
        }
        for ry in 22..28usize {
            let rp = eca_ptr.add((qy + ry) * frame_w + qx);
            let v0 = v128_load(rp.add(4) as *const v128);
            let v1 = v128_load(rp.add(8) as *const v128);
            let v2 = v128_load(rp.add(12) as *const v128);
            let v3 = v128_load(rp.add(16) as *const v128);
            let v4 = v128_load(rp.add(20) as *const v128);
            let v5 = v128_load(rp.add(24) as *const v128);
            let s05 = f32x4_add(v0, v5);
            let s14 = f32x4_add(v1, v4);
            let s23 = f32x4_add(v2, v3);
            acc_core = f32x4_add(acc_core, f32x4_add(s05, f32x4_add(s14, s23)));
            let m05 = f32x4_pmax(v0, v5);
            let m14 = f32x4_pmax(v1, v4);
            let m23 = f32x4_pmax(v2, v3);
            acc_max = f32x4_pmax(acc_max, f32x4_pmax(m05, f32x4_pmax(m14, m23)));
        }
        let m2 = f32x4_pmax(acc_max, i32x4_shuffle::<2, 3, 0, 1>(acc_max, acc_max));
        let max_v = f32x4_extract_lane::<0>(f32x4_pmax(m2, i32x4_shuffle::<1, 0, 3, 2>(m2, m2)));
        (hsum_f32x4(acc_core), hsum_f32x4(acc_inner), max_v)
    } else if scale_px == 1 {
        let mut acc_core = f32x4_splat(0.0);
        let mut acc_inner = f32x4_splat(0.0);
        let mut acc_max = f32x4_splat(0.0);
        let z = f32x4_splat(0.0);
        for ry in 2..14usize {
            let rp = eca_ptr.add((qy + ry) * frame_w + qx);
            let v0 = v128_load(rp.add(2) as *const v128);
            let v1 = v128_load(rp.add(6) as *const v128);
            let v2 = v128_load(rp.add(10) as *const v128);
            acc_core = f32x4_add(acc_core, f32x4_add(v0, f32x4_add(v1, v2)));
            acc_max = f32x4_pmax(acc_max, f32x4_pmax(v0, f32x4_pmax(v1, v2)));
            if ry >= 5 && ry < 11 {
                let e0 = i32x4_shuffle::<3, 5, 6, 7>(v0, z);
                let e2 = i32x4_shuffle::<0, 5, 6, 7>(v2, z);
                acc_inner = f32x4_add(acc_inner, f32x4_add(v1, f32x4_add(e0, e2)));
            }
        }
        let m2 = f32x4_pmax(acc_max, i32x4_shuffle::<2, 3, 0, 1>(acc_max, acc_max));
        let max_v = f32x4_extract_lane::<0>(f32x4_pmax(m2, i32x4_shuffle::<1, 0, 3, 2>(m2, m2)));
        (hsum_f32x4(acc_core), hsum_f32x4(acc_inner), max_v)
    } else {
        let core_lo = 2 * scale_px;
        let core_hi = 14 * scale_px;
        let inner_lo = 5 * scale_px;
        let inner_hi = 11 * scale_px;
        let mut c_eca = 0.0f32;
        let mut inner_eca = 0.0f32;
        let mut max_eca = 0.0f32;
        for ry in core_lo..core_hi {
            let roff = (qy + ry) * frame_w + qx;
            let in_y = ry >= inner_lo && ry < inner_hi;
            for rx in core_lo..core_hi {
                let v = *eca_ptr.add(roff + rx);
                c_eca += v;
                if in_y && rx >= inner_lo && rx < inner_hi {
                    inner_eca += v;
                }
                if v > max_eca {
                    max_eca = v;
                }
            }
        }
        (c_eca, inner_eca, max_eca)
    }
}

#[inline]
pub unsafe fn fit_affine_rgb_trimmed_simd128(
    bs: *const f32,
    ys: *const f32,
    n: usize,
    out: *mut [f32; 4],
) -> bool {
    if n < 64 {
        return false;
    }
    const THR1: f32 = 56.0 / 255.0;
    const THR2: f32 = 9.0 / 255.0;
    let v_thr1 = f32x4_splat(THR1);
    let v_thr2 = f32x4_splat(THR2);
    let one_i = i32x4_splat(1);
    let n8 = n & !7;
    let n4 = n & !3;

    // Pass 1: coarse inliers |y - b| <= THR1 -> means (mb, my)
    let mut sb0 = f32x4_splat(0.0);
    let mut sb1 = f32x4_splat(0.0);
    let mut sy0 = f32x4_splat(0.0);
    let mut sy1 = f32x4_splat(0.0);
    let mut n0_v = i32x4_splat(0);
    let mut i = 0usize;
    while i < n8 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let b1 = v128_load(bs.add(i + 4) as *const v128);
        let y1 = v128_load(ys.add(i + 4) as *const v128);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, b0)), v_thr1);
        let m1 = f32x4_le(f32x4_abs(f32x4_sub(y1, b1)), v_thr1);
        sb0 = f32x4_add(sb0, v128_and(b0, m0));
        sb1 = f32x4_add(sb1, v128_and(b1, m1));
        sy0 = f32x4_add(sy0, v128_and(y0, m0));
        sy1 = f32x4_add(sy1, v128_and(y1, m1));
        n0_v = i32x4_add(n0_v, i32x4_add(v128_and(one_i, m0), v128_and(one_i, m1)));
        i += 8;
    }
    if i < n4 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, b0)), v_thr1);
        sb0 = f32x4_add(sb0, v128_and(b0, m0));
        sy0 = f32x4_add(sy0, v128_and(y0, m0));
        n0_v = i32x4_add(n0_v, v128_and(one_i, m0));
        i += 4;
    }
    let mut sb = hsum_f32x4(f32x4_add(sb0, sb1));
    let mut sy = hsum_f32x4(f32x4_add(sy0, sy1));
    let mut n0 = hsum_i32x4(n0_v) as usize;
    while i < n {
        let b = *bs.add(i);
        let y = *ys.add(i);
        if (y - b).abs() <= THR1 {
            sb += b;
            sy += y;
            n0 += 1;
        }
        i += 1;
    }
    if n0 < 64 {
        return false;
    }
    let inv_n0 = 1.0 / (n0 as f32);
    let mb = sb * inv_n0;
    let my = sy * inv_n0;

    // Pass 2: coarse variance/covariance around (mb, my)
    let v_mb = f32x4_splat(mb);
    let v_my = f32x4_splat(my);
    let mut vbb0 = f32x4_splat(0.0);
    let mut vbb1 = f32x4_splat(0.0);
    let mut cby0 = f32x4_splat(0.0);
    let mut cby1 = f32x4_splat(0.0);
    i = 0;
    while i < n8 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let b1 = v128_load(bs.add(i + 4) as *const v128);
        let y1 = v128_load(ys.add(i + 4) as *const v128);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, b0)), v_thr1);
        let m1 = f32x4_le(f32x4_abs(f32x4_sub(y1, b1)), v_thr1);
        let db0 = v128_and(f32x4_sub(b0, v_mb), m0);
        let dy0 = v128_and(f32x4_sub(y0, v_my), m0);
        let db1 = v128_and(f32x4_sub(b1, v_mb), m1);
        let dy1 = v128_and(f32x4_sub(y1, v_my), m1);
        vbb0 = f32x4_add(vbb0, f32x4_mul(db0, db0));
        vbb1 = f32x4_add(vbb1, f32x4_mul(db1, db1));
        cby0 = f32x4_add(cby0, f32x4_mul(db0, dy0));
        cby1 = f32x4_add(cby1, f32x4_mul(db1, dy1));
        i += 8;
    }
    if i < n4 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, b0)), v_thr1);
        let db0 = v128_and(f32x4_sub(b0, v_mb), m0);
        let dy0 = v128_and(f32x4_sub(y0, v_my), m0);
        vbb0 = f32x4_add(vbb0, f32x4_mul(db0, db0));
        cby0 = f32x4_add(cby0, f32x4_mul(db0, dy0));
        i += 4;
    }
    let mut v_bb = hsum_f32x4(f32x4_add(vbb0, vbb1));
    let mut c_by = hsum_f32x4(f32x4_add(cby0, cby1));
    while i < n {
        let b = *bs.add(i);
        let y = *ys.add(i);
        if (y - b).abs() <= THR1 {
            let db = b - mb;
            let dy = y - my;
            v_bb += db * db;
            c_by += db * dy;
        }
        i += 1;
    }
    if v_bb * inv_n0 < 5e-5 {
        return false;
    }
    let g1 = (c_by / v_bb).clamp(0.60, 1.45);
    let o1 = (my - g1 * mb).clamp(-0.18, 0.18);

    // Pass 3: refined inliers |y - (g1*b + o1)| <= THR2 -> means (mb2, my2)
    let v_g1 = f32x4_splat(g1);
    let v_o1 = f32x4_splat(o1);
    let mut sb20 = f32x4_splat(0.0);
    let mut sb21 = f32x4_splat(0.0);
    let mut sy20 = f32x4_splat(0.0);
    let mut sy21 = f32x4_splat(0.0);
    let mut n2_v = i32x4_splat(0);
    i = 0;
    while i < n8 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let b1 = v128_load(bs.add(i + 4) as *const v128);
        let y1 = v128_load(ys.add(i + 4) as *const v128);
        let p0 = f32x4_add(f32x4_mul(v_g1, b0), v_o1);
        let p1 = f32x4_add(f32x4_mul(v_g1, b1), v_o1);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, p0)), v_thr2);
        let m1 = f32x4_le(f32x4_abs(f32x4_sub(y1, p1)), v_thr2);
        sb20 = f32x4_add(sb20, v128_and(b0, m0));
        sb21 = f32x4_add(sb21, v128_and(b1, m1));
        sy20 = f32x4_add(sy20, v128_and(y0, m0));
        sy21 = f32x4_add(sy21, v128_and(y1, m1));
        n2_v = i32x4_add(n2_v, i32x4_add(v128_and(one_i, m0), v128_and(one_i, m1)));
        i += 8;
    }
    if i < n4 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let p0 = f32x4_add(f32x4_mul(v_g1, b0), v_o1);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, p0)), v_thr2);
        sb20 = f32x4_add(sb20, v128_and(b0, m0));
        sy20 = f32x4_add(sy20, v128_and(y0, m0));
        n2_v = i32x4_add(n2_v, v128_and(one_i, m0));
        i += 4;
    }
    let mut sb2 = hsum_f32x4(f32x4_add(sb20, sb21));
    let mut sy2 = hsum_f32x4(f32x4_add(sy20, sy21));
    let mut n2 = hsum_i32x4(n2_v) as usize;
    while i < n {
        let b = *bs.add(i);
        let y = *ys.add(i);
        if (y - (g1 * b + o1)).abs() <= THR2 {
            sb2 += b;
            sy2 += y;
            n2 += 1;
        }
        i += 1;
    }
    if n2 < 64 || n2 * 10 < n0 * 6 {
        return false;
    }
    let inv_n2 = 1.0 / (n2 as f32);
    let mb2 = sb2 * inv_n2;
    let my2 = sy2 * inv_n2;

    // Pass 4: refined variance/covariance + pre_sse
    let v_mb2 = f32x4_splat(mb2);
    let v_my2 = f32x4_splat(my2);
    let mut vbb20 = f32x4_splat(0.0);
    let mut vbb21 = f32x4_splat(0.0);
    let mut cby20 = f32x4_splat(0.0);
    let mut cby21 = f32x4_splat(0.0);
    let mut psse0 = f32x4_splat(0.0);
    let mut psse1 = f32x4_splat(0.0);
    i = 0;
    while i < n8 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let b1 = v128_load(bs.add(i + 4) as *const v128);
        let y1 = v128_load(ys.add(i + 4) as *const v128);
        let p0 = f32x4_add(f32x4_mul(v_g1, b0), v_o1);
        let p1 = f32x4_add(f32x4_mul(v_g1, b1), v_o1);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, p0)), v_thr2);
        let m1 = f32x4_le(f32x4_abs(f32x4_sub(y1, p1)), v_thr2);
        let db0 = v128_and(f32x4_sub(b0, v_mb2), m0);
        let dy0 = v128_and(f32x4_sub(y0, v_my2), m0);
        let e0 = v128_and(f32x4_sub(y0, b0), m0);
        let db1 = v128_and(f32x4_sub(b1, v_mb2), m1);
        let dy1 = v128_and(f32x4_sub(y1, v_my2), m1);
        let e1 = v128_and(f32x4_sub(y1, b1), m1);
        vbb20 = f32x4_add(vbb20, f32x4_mul(db0, db0));
        vbb21 = f32x4_add(vbb21, f32x4_mul(db1, db1));
        cby20 = f32x4_add(cby20, f32x4_mul(db0, dy0));
        cby21 = f32x4_add(cby21, f32x4_mul(db1, dy1));
        psse0 = f32x4_add(psse0, f32x4_mul(e0, e0));
        psse1 = f32x4_add(psse1, f32x4_mul(e1, e1));
        i += 8;
    }
    if i < n4 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let p0 = f32x4_add(f32x4_mul(v_g1, b0), v_o1);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, p0)), v_thr2);
        let db0 = v128_and(f32x4_sub(b0, v_mb2), m0);
        let dy0 = v128_and(f32x4_sub(y0, v_my2), m0);
        let e0 = v128_and(f32x4_sub(y0, b0), m0);
        vbb20 = f32x4_add(vbb20, f32x4_mul(db0, db0));
        cby20 = f32x4_add(cby20, f32x4_mul(db0, dy0));
        psse0 = f32x4_add(psse0, f32x4_mul(e0, e0));
        i += 4;
    }
    let mut v_bb2 = hsum_f32x4(f32x4_add(vbb20, vbb21));
    let mut c_by2 = hsum_f32x4(f32x4_add(cby20, cby21));
    let mut pre_sse = hsum_f32x4(f32x4_add(psse0, psse1));
    while i < n {
        let b = *bs.add(i);
        let y = *ys.add(i);
        if (y - (g1 * b + o1)).abs() <= THR2 {
            let db = b - mb2;
            let dy = y - my2;
            v_bb2 += db * db;
            c_by2 += db * dy;
            let e0 = y - b;
            pre_sse += e0 * e0;
        }
        i += 1;
    }
    if v_bb2 * inv_n2 < 5e-5 {
        return false;
    }
    let g2 = (c_by2 / v_bb2).clamp(0.60, 1.45);
    let o2 = (my2 - g2 * mb2).clamp(-0.18, 0.18);

    // Pass 5: post_sse
    let v_g2 = f32x4_splat(g2);
    let v_o2 = f32x4_splat(o2);
    let mut post0 = f32x4_splat(0.0);
    let mut post1 = f32x4_splat(0.0);
    i = 0;
    while i < n8 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let b1 = v128_load(bs.add(i + 4) as *const v128);
        let y1 = v128_load(ys.add(i + 4) as *const v128);
        let p0 = f32x4_add(f32x4_mul(v_g1, b0), v_o1);
        let p1 = f32x4_add(f32x4_mul(v_g1, b1), v_o1);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, p0)), v_thr2);
        let m1 = f32x4_le(f32x4_abs(f32x4_sub(y1, p1)), v_thr2);
        let e0 = v128_and(f32x4_sub(y0, f32x4_add(f32x4_mul(v_g2, b0), v_o2)), m0);
        let e1 = v128_and(f32x4_sub(y1, f32x4_add(f32x4_mul(v_g2, b1), v_o2)), m1);
        post0 = f32x4_add(post0, f32x4_mul(e0, e0));
        post1 = f32x4_add(post1, f32x4_mul(e1, e1));
        i += 8;
    }
    if i < n4 {
        let b0 = v128_load(bs.add(i) as *const v128);
        let y0 = v128_load(ys.add(i) as *const v128);
        let p0 = f32x4_add(f32x4_mul(v_g1, b0), v_o1);
        let m0 = f32x4_le(f32x4_abs(f32x4_sub(y0, p0)), v_thr2);
        let e0 = v128_and(f32x4_sub(y0, f32x4_add(f32x4_mul(v_g2, b0), v_o2)), m0);
        post0 = f32x4_add(post0, f32x4_mul(e0, e0));
        i += 4;
    }
    let mut post_sse = hsum_f32x4(f32x4_add(post0, post1));
    while i < n {
        let b = *bs.add(i);
        let y = *ys.add(i);
        if (y - (g1 * b + o1)).abs() <= THR2 {
            let e = y - (g2 * b + o2);
            post_sse += e * e;
        }
        i += 1;
    }
    *out = [g2, o2, pre_sse * inv_n2, post_sse * inv_n2];
    true
}

#[inline(never)]
pub unsafe fn estimate_pan_exposure_symmetric_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    scratch: *mut f32,
    frame_w: usize,
    frame_h: usize,
    pan_dx_round: i32,
    pan_dy_round: i32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    out_go: *mut [f32; 2],
) {
    let floor_i32 = |v: f32| -> i32 {
        if v >= 0.0 {
            v as i32
        } else {
            let iv = v as i32;
            if (iv as f32) == v { iv } else { iv - 1 }
        }
    };
    let rem_x = pan_dx_f - (pan_dx_round as f32);
    let rem_y = pan_dy_f - (pan_dy_round as f32);
    let hx_c = 0.5 * rem_x;
    let hy_c = 0.5 * rem_y;
    let ox_c = floor_i32(hx_c);
    let oy_c = floor_i32(hy_c);
    let fx_c = (hx_c - (ox_c as f32)).clamp(0.0, 1.0);
    let fy_c = (hy_c - (oy_c as f32)).clamp(0.0, 1.0);
    let wc00 = (1.0 - fx_c) * (1.0 - fy_c);
    let wc10 = fx_c * (1.0 - fy_c);
    let wc01 = (1.0 - fx_c) * fy_c;
    let wc11 = fx_c * fy_c;

    let ox_p_sub = floor_i32(-hx_c);
    let oy_p_sub = floor_i32(-hy_c);
    let fx_p = (-hx_c - (ox_p_sub as f32)).clamp(0.0, 1.0);
    let fy_p = (-hy_c - (oy_p_sub as f32)).clamp(0.0, 1.0);
    let ox_p = -pan_dx_round + ox_p_sub;
    let oy_p = -pan_dy_round + oy_p_sub;
    let wp00 = (1.0 - fx_p) * (1.0 - fy_p);
    let wp10 = fx_p * (1.0 - fy_p);
    let wp01 = (1.0 - fx_p) * fy_p;
    let wp11 = fx_p * fy_p;

    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let step_y = (frame_h / 18).max(4);
    let step_x = (frame_w / 24).max(4);

    let gx_min_req = (-ox_c).max(-ox_p).max(step_x as i32) as usize;
    let gx_max_excl = (w_i - 1 - ox_c)
        .min(w_i - 1 - ox_p)
        .min(w_i - (step_x as i32));
    let gy_min_req = (-oy_c).max(-oy_p).max(step_y as i32) as usize;
    let gy_max_excl = (h_i - 1 - oy_c)
        .min(h_i - 1 - oy_p)
        .min(h_i - (step_y as i32));

    if gx_max_excl <= (gx_min_req as i32) || gy_max_excl <= (gy_min_req as i32) {
        *out_go = [1.0, 0.0];
        return;
    }
    let gx_start = ((gx_min_req + step_x - 1) / step_x) * step_x;
    let gy_start = ((gy_min_req + step_y - 1) / step_y) * step_y;
    let gx_end = gx_max_excl as usize;
    let gy_end = gy_max_excl as usize;

    let row_stride = frame_w * 4;
    let bs_ptr = scratch;
    let ys_ptr = scratch.add(2048);

    const INV_255: f32 = 1.0 / 255.0;
    let v_wc00 = f32x4_splat(wc00);
    let v_wc10 = f32x4_splat(wc10);
    let v_wc01 = f32x4_splat(wc01);
    let v_wc11 = f32x4_splat(wc11);
    let v_wp00 = f32x4_splat(wp00);
    let v_wp10 = f32x4_splat(wp10);
    let v_wp01 = f32x4_splat(wp01);
    let v_wp11 = f32x4_splat(wp11);
    let v_inv255 = f32x4_splat(INV_255);
    let v_004 = f32x4_splat(0.04);
    let v_092 = f32x4_splat(0.92);

    let mut count = 0usize;
    let mut gy = gy_start;
    while gy < gy_end && count + 3 <= 2048 {
        let yc0 = ((gy as i32) + oy_c) as usize;
        let yp0 = ((gy as i32) + oy_p) as usize;
        let row_c = yc0 * frame_w;
        let row_p = yp0 * frame_w;
        let mut gx = gx_start;
        while gx < gx_end && count + 3 <= 2048 {
            let xc0 = ((gx as i32) + ox_c) as usize;
            let xp0 = ((gx as i32) + ox_p) as usize;
            let c_ptr = curr_rgba.add((row_c + xc0) * 4);
            let p_ptr = prev_rgba.add((row_p + xp0) * 4);
            let pr0_v = v128_load64_zero(p_ptr as *const u64);
            let c00_v = v128_load32_zero(c_ptr as *const u32);
            let pr0_u64 = u64x2_extract_lane::<0>(pr0_v);
            let p00_u32 = pr0_u64 as u32;
            let c00_u32 = u32x4_extract_lane::<0>(c00_v);
            if (p00_u32 & 0xff).max((p00_u32 >> 8) & 0xff) < 236
                && (c00_u32 & 0xff).max((c00_u32 >> 8) & 0xff) < 236
            {
                let pr1_v = v128_load64_zero(p_ptr.add(row_stride) as *const u64);
                let pr1_u64 = u64x2_extract_lane::<0>(pr1_v);
                let g00 = ((p00_u32 >> 8) & 0xff) as u8;
                let g10 = ((pr0_u64 >> 40) & 0xff) as u8;
                let g01 = ((pr1_u64 >> 8) & 0xff) as u8;
                let g11 = ((pr1_u64 >> 40) & 0xff) as u8;
                let sp_g = g00.max(g10).max(g01).max(g11) - g00.min(g10).min(g01).min(g11);
                if sp_g <= 18 {
                    let cr0_v = v128_load64_zero(c_ptr as *const u64);
                    let cr1_v = v128_load64_zero(c_ptr.add(row_stride) as *const u64);
                    let p01_16 = u16x8_extend_low_u8x16(pr0_v);
                    let p23_16 = u16x8_extend_low_u8x16(pr1_v);
                    let p00 = f32x4_convert_u32x4(u32x4_extend_low_u16x8(p01_16));
                    let p10 = f32x4_convert_u32x4(u32x4_extend_high_u16x8(p01_16));
                    let p01 = f32x4_convert_u32x4(u32x4_extend_low_u16x8(p23_16));
                    let p11 = f32x4_convert_u32x4(u32x4_extend_high_u16x8(p23_16));

                    let c01_16 = u16x8_extend_low_u8x16(cr0_v);
                    let c23_16 = u16x8_extend_low_u8x16(cr1_v);
                    let c00 = f32x4_convert_u32x4(u32x4_extend_low_u16x8(c01_16));
                    let c10 = f32x4_convert_u32x4(u32x4_extend_high_u16x8(c01_16));
                    let c01 = f32x4_convert_u32x4(u32x4_extend_low_u16x8(c23_16));
                    let c11 = f32x4_convert_u32x4(u32x4_extend_high_u16x8(c23_16));

                    let y_sum = f32x4_add(
                        f32x4_add(
                            f32x4_add(f32x4_mul(v_wc00, c00), f32x4_mul(v_wc10, c10)),
                            f32x4_mul(v_wc01, c01),
                        ),
                        f32x4_mul(v_wc11, c11),
                    );
                    let y_vec = f32x4_mul(y_sum, v_inv255);
                    let b_sum = f32x4_add(
                        f32x4_add(
                            f32x4_add(f32x4_mul(v_wp00, p00), f32x4_mul(v_wp10, p10)),
                            f32x4_mul(v_wp01, p01),
                        ),
                        f32x4_mul(v_wp11, p11),
                    );
                    let b_vec = f32x4_mul(b_sum, v_inv255);

                    let skip_ch2 =
                        (p00_u32 & 0x00ff_0000) >= (238 << 16)
                            || (c00_u32 & 0x00ff_0000) >= (238 << 16);
                    let ok_b = v128_and(f32x4_ge(b_vec, v_004), f32x4_le(b_vec, v_092));
                    let ok_y = v128_and(f32x4_ge(y_vec, v_004), f32x4_le(y_vec, v_092));
                    let mask = (i32x4_bitmask(v128_and(ok_b, ok_y)) as usize)
                        & if skip_ch2 { 3 } else { 7 };

                    if mask == 7 && count + 4 <= 2048 {
                        v128_store(bs_ptr.add(count) as *mut v128, b_vec);
                        v128_store(ys_ptr.add(count) as *mut v128, y_vec);
                        count += 3;
                    } else if mask != 0 {
                        if (mask & 1) != 0 {
                            *bs_ptr.add(count) = f32x4_extract_lane::<0>(b_vec);
                            *ys_ptr.add(count) = f32x4_extract_lane::<0>(y_vec);
                            count += 1;
                        }
                        if (mask & 2) != 0 {
                            *bs_ptr.add(count) = f32x4_extract_lane::<1>(b_vec);
                            *ys_ptr.add(count) = f32x4_extract_lane::<1>(y_vec);
                            count += 1;
                        }
                        if (mask & 4) != 0 {
                            *bs_ptr.add(count) = f32x4_extract_lane::<2>(b_vec);
                            *ys_ptr.add(count) = f32x4_extract_lane::<2>(y_vec);
                            count += 1;
                        }
                    }
                }
            }
            gx += step_x;
        }
        gy += step_y;
    }

    let mut res = [0.0f32; 4];
    if fit_affine_rgb_trimmed_simd128(bs_ptr, ys_ptr, count, &mut res) {
        let g = res[0];
        let o = res[1];
        let pre_mse = res[2];
        let post_mse = res[3];
        if ((g - 1.0).abs() >= 0.003 || o.abs() >= 0.002) && post_mse < 0.97 * pre_mse {
            *out_go = [g, o];
            return;
        }
    }
    *out_go = [1.0, 0.0];
}

#[inline]
pub unsafe fn sample_luma_patch64_p2_simd128(
    luma_ptr: *const f32,
    frame_w: usize,
    bx: usize,
    by: usize,
    patch: &mut [f32; 64],
) -> f32 {
    let base = luma_ptr.add(by * frame_w + bx);
    let mut pv = [f32x4_splat(0.0); 16];
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    for r in 0..8usize {
        let p0 = base.add(2 * r * frame_w + (r & 1));
        let p1 = p0.add(frame_w);

        let ev0 = i32x4_shuffle::<0, 4, 0, 4>(
            v128_load(p0 as *const v128),
            v128_load(p0.add(4) as *const v128),
        );
        let od0 = i32x4_shuffle::<2, 6, 2, 6>(
            v128_load(p1 as *const v128),
            v128_load(p1.add(4) as *const v128),
        );
        let l0 = i32x4_shuffle::<0, 4, 1, 5>(ev0, od0);

        let ev1 = i32x4_shuffle::<0, 4, 0, 4>(
            v128_load(p0.add(8) as *const v128),
            v128_load(p0.add(12) as *const v128),
        );
        let od1 = i32x4_shuffle::<2, 6, 2, 6>(
            v128_load(p1.add(8) as *const v128),
            v128_load(p1.add(12) as *const v128),
        );
        let l1 = i32x4_shuffle::<0, 4, 1, 5>(ev1, od1);

        pv[r * 2] = l0;
        pv[r * 2 + 1] = l1;
        sum0 = f32x4_add(sum0, l0);
        sum1 = f32x4_add(sum1, l1);
    }
    let mp_v = f32x4_splat(hsum_f32x4(f32x4_add(sum0, sum1)) * (1.0 / 64.0));
    let mut m0 = f32x4_splat(0.0);
    let mut m1 = f32x4_splat(0.0);
    let dst = patch.as_mut_ptr();
    for i in (0..16usize).step_by(2) {
        let d0 = f32x4_sub(pv[i], mp_v);
        let d1 = f32x4_sub(pv[i + 1], mp_v);
        v128_store(dst.add(i * 4) as *mut v128, d0);
        v128_store(dst.add((i + 1) * 4) as *mut v128, d1);
        m0 = f32x4_add(m0, f32x4_abs(d0));
        m1 = f32x4_add(m1, f32x4_abs(d1));
    }
    hsum_f32x4(f32x4_add(m0, m1))
}

#[inline]
pub unsafe fn demean_mad64_simd128(patch: &mut [f32; 64]) -> f32 {
    let p = patch.as_mut_ptr();
    let mut s0 = f32x4_splat(0.0);
    let mut s1 = f32x4_splat(0.0);
    for i in (0..16usize).step_by(2) {
        s0 = f32x4_add(s0, v128_load(p.add(i * 4) as *const v128));
        s1 = f32x4_add(s1, v128_load(p.add((i + 1) * 4) as *const v128));
    }
    let mp_v = f32x4_splat(hsum_f32x4(f32x4_add(s0, s1)) * (1.0 / 64.0));
    let mut m0 = f32x4_splat(0.0);
    let mut m1 = f32x4_splat(0.0);
    for i in (0..16usize).step_by(2) {
        let d0 = f32x4_sub(v128_load(p.add(i * 4) as *const v128), mp_v);
        let d1 = f32x4_sub(v128_load(p.add((i + 1) * 4) as *const v128), mp_v);
        v128_store(p.add(i * 4) as *mut v128, d0);
        v128_store(p.add((i + 1) * 4) as *mut v128, d1);
        m0 = f32x4_add(m0, f32x4_abs(d0));
        m1 = f32x4_add(m1, f32x4_abs(d1));
    }
    hsum_f32x4(f32x4_add(m0, m1))
}

#[inline]
pub unsafe fn unpack_rgba_luma_win18_p2_simd128(
    prev_rgba: &[u8],
    frame_w: usize,
    frame_h: usize,
    wx0: i32,
    wy0: i32,
    dst: &mut [f32],
) {
    if frame_w == 0 || frame_h == 0 || dst.len() < 324 {
        return;
    }
    let w_i = frame_w as i32;
    let h_i = frame_h as i32;
    let w_max = w_i - 1;
    let h_max = h_i - 1;
    let rgba_ptr = prev_rgba.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let w_luma = i16x8(9798, 19235, 3735, 0, 9798, 19235, 3735, 0);
    let scale_luma = f32x4_splat(1.0 / (255.0 * 32768.0));

    #[inline(always)]
    unsafe fn rgba4_luma_dot(pk: v128, w_luma: v128, scale_luma: v128) -> v128 {
        let lo = i32x4_dot_i16x8(u16x8_extend_low_u8x16(pk), w_luma);
        let hi = i32x4_dot_i16x8(u16x8_extend_high_u8x16(pk), w_luma);
        let sum = i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(lo, hi),
            i32x4_shuffle::<1, 3, 5, 7>(lo, hi),
        );
        f32x4_mul(f32x4_convert_i32x4(sum), scale_luma)
    }

    let in_bounds = wx0 >= 0
        && wy0 >= 0
        && wx0 + 18 <= w_i
        && wy0 + 18 <= h_i
        && (((wy0 as usize) + 17) * frame_w + (wx0 as usize) + 18) * 4 <= prev_rgba.len();
    if in_bounds {
        let ux0 = wx0 as usize;
        let uy0 = wy0 as usize;
        for y in 0..18usize {
            let rp = rgba_ptr.add(((uy0 + y) * frame_w + ux0) * 4);
            let dr = dst_ptr.add(y * 18);
            let v0 = rgba4_luma_dot(v128_load(rp as *const v128), w_luma, scale_luma);
            let v1 = rgba4_luma_dot(v128_load(rp.add(16) as *const v128), w_luma, scale_luma);
            let v2 = rgba4_luma_dot(v128_load(rp.add(32) as *const v128), w_luma, scale_luma);
            let v3 = rgba4_luma_dot(v128_load(rp.add(48) as *const v128), w_luma, scale_luma);
            let v4 = rgba4_luma_dot(v128_load64_zero(rp.add(64) as *const u64), w_luma, scale_luma);

            let t0 = i32x4_shuffle::<0, 4, 1, 5>(v0, v1);
            let t1 = i32x4_shuffle::<2, 6, 3, 7>(v0, v1);
            let t2 = i32x4_shuffle::<0, 4, 1, 5>(v2, v3);
            let t3 = i32x4_shuffle::<2, 6, 3, 7>(v2, v3);

            v128_store(dr as *mut v128, i32x4_shuffle::<0, 1, 4, 5>(t0, t2));
            *dr.add(4) = f32x4_extract_lane::<0>(v4);
            v128_store(dr.add(5) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t0, t2));
            *dr.add(9) = f32x4_extract_lane::<1>(v4);
            v128_store(dr.add(10) as *mut v128, i32x4_shuffle::<0, 1, 4, 5>(t1, t3));
            v128_store(dr.add(14) as *mut v128, i32x4_shuffle::<2, 3, 6, 7>(t1, t3));
        }
        return;
    }

    const PH_OFF: [usize; 4] = [0, 5, 10, 14];
    for y in 0..18usize {
        let iy = (wy0 + (y as i32)).clamp(0, h_max) as usize;
        let roff = iy * frame_w;
        let dr = dst_ptr.add(y * 18);
        for x in 0..18usize {
            let ix = (wx0 + (x as i32)).clamp(0, w_max) as usize;
            let o4 = (roff + ix) * 4;
            let l = (0.299 * (prev_rgba[o4] as f32)
                + 0.587 * (prev_rgba[o4 + 1] as f32)
                + 0.114 * (prev_rgba[o4 + 2] as f32))
                * (1.0 / 255.0);
            *dr.add(PH_OFF[x & 3] + (x >> 2)) = l;
        }
    }
}

#[inline]
pub unsafe fn subpx_patch_gain_norm_sad_p2_18_simd128(
    cur_eo: &[f32; 64],
    cur_mad: f32,
    pl: &[f32],
    off_x_base: usize,
    off_y_base: usize,
    w00: f32,
    w10: f32,
    w01: f32,
    w11: f32,
) -> f32 {
    if pl.len() < (16 + off_y_base) * 18 {
        return 1e9;
    }
    let p_ptr = pl.as_ptr();
    let c_ptr = cur_eo.as_ptr();
    let (ea, eb, oa, ob, ea2, eb2, oa2, ob2) = if off_x_base == 0 {
        (0usize, 5usize, 10usize, 14usize, 5usize, 10usize, 14usize, 1usize)
    } else {
        (5usize, 10usize, 14usize, 1usize, 10usize, 14usize, 1usize, 6usize)
    };
    let r_base = off_y_base * 18;
    let p_ee_a0 = p_ptr.add(r_base + ea);
    let p_ee_b0 = p_ptr.add(r_base + eb);
    let p_eo_a0 = p_ptr.add(r_base + 18 + oa);
    let p_eo_b0 = p_ptr.add(r_base + 18 + ob);
    let p_oe_a0 = p_ptr.add(r_base + 36 + ea2);
    let p_oe_b0 = p_ptr.add(r_base + 36 + eb2);
    let p_oo_a0 = p_ptr.add(r_base + 54 + oa2);
    let p_oo_b0 = p_ptr.add(r_base + 54 + ob2);

    let mut pv = [f32x4_splat(0.0); 16];
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);

    if w10 == 0.0 && w11 == 0.0 {
        if w01 == 0.0 {
            for k in 0..4usize {
                let rk = k * 72;
                let v0 = v128_load(p_ee_a0.add(rk) as *const v128);
                let v1 = v128_load(p_eo_a0.add(rk) as *const v128);
                let v2 = v128_load(p_oe_a0.add(rk) as *const v128);
                let v3 = v128_load(p_oo_a0.add(rk) as *const v128);
                let j = k * 4;
                pv[j] = v0;
                pv[j + 1] = v1;
                pv[j + 2] = v2;
                pv[j + 3] = v3;
                sum0 = f32x4_add(sum0, f32x4_add(v0, v1));
                sum1 = f32x4_add(sum1, f32x4_add(v2, v3));
            }
        } else {
            let w00_v = f32x4_splat(w00);
            let w01_v = f32x4_splat(w01);
            for k in 0..4usize {
                let rk = k * 72;
                let rk1 = rk + 18;
                let v0 = f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_ee_a0.add(rk) as *const v128)),
                    f32x4_mul(w01_v, v128_load(p_ee_a0.add(rk1) as *const v128)),
                );
                let v1 = f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_eo_a0.add(rk) as *const v128)),
                    f32x4_mul(w01_v, v128_load(p_eo_a0.add(rk1) as *const v128)),
                );
                let v2 = f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_oe_a0.add(rk) as *const v128)),
                    f32x4_mul(w01_v, v128_load(p_oe_a0.add(rk1) as *const v128)),
                );
                let v3 = f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_oo_a0.add(rk) as *const v128)),
                    f32x4_mul(w01_v, v128_load(p_oo_a0.add(rk1) as *const v128)),
                );
                let j = k * 4;
                pv[j] = v0;
                pv[j + 1] = v1;
                pv[j + 2] = v2;
                pv[j + 3] = v3;
                sum0 = f32x4_add(sum0, f32x4_add(v0, v1));
                sum1 = f32x4_add(sum1, f32x4_add(v2, v3));
            }
        }
    } else {
        let w00_v = f32x4_splat(w00);
        let w10_v = f32x4_splat(w10);
        let w01_v = f32x4_splat(w01);
        let w11_v = f32x4_splat(w11);
        for k in 0..4usize {
            let rk = k * 72;
            let rk1 = rk + 18;
            let v0 = f32x4_add(
                f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_ee_a0.add(rk) as *const v128)),
                    f32x4_mul(w10_v, v128_load(p_ee_b0.add(rk) as *const v128)),
                ),
                f32x4_add(
                    f32x4_mul(w01_v, v128_load(p_ee_a0.add(rk1) as *const v128)),
                    f32x4_mul(w11_v, v128_load(p_ee_b0.add(rk1) as *const v128)),
                ),
            );
            let v1 = f32x4_add(
                f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_eo_a0.add(rk) as *const v128)),
                    f32x4_mul(w10_v, v128_load(p_eo_b0.add(rk) as *const v128)),
                ),
                f32x4_add(
                    f32x4_mul(w01_v, v128_load(p_eo_a0.add(rk1) as *const v128)),
                    f32x4_mul(w11_v, v128_load(p_eo_b0.add(rk1) as *const v128)),
                ),
            );
            let v2 = f32x4_add(
                f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_oe_a0.add(rk) as *const v128)),
                    f32x4_mul(w10_v, v128_load(p_oe_b0.add(rk) as *const v128)),
                ),
                f32x4_add(
                    f32x4_mul(w01_v, v128_load(p_oe_a0.add(rk1) as *const v128)),
                    f32x4_mul(w11_v, v128_load(p_oe_b0.add(rk1) as *const v128)),
                ),
            );
            let v3 = f32x4_add(
                f32x4_add(
                    f32x4_mul(w00_v, v128_load(p_oo_a0.add(rk) as *const v128)),
                    f32x4_mul(w10_v, v128_load(p_oo_b0.add(rk) as *const v128)),
                ),
                f32x4_add(
                    f32x4_mul(w01_v, v128_load(p_oo_a0.add(rk1) as *const v128)),
                    f32x4_mul(w11_v, v128_load(p_oo_b0.add(rk1) as *const v128)),
                ),
            );
            let j = k * 4;
            pv[j] = v0;
            pv[j + 1] = v1;
            pv[j + 2] = v2;
            pv[j + 3] = v3;
            sum0 = f32x4_add(sum0, f32x4_add(v0, v1));
            sum1 = f32x4_add(sum1, f32x4_add(v2, v3));
        }
    }

    let mp_v = f32x4_splat(hsum_f32x4(f32x4_add(sum0, sum1)) * (1.0 / 64.0));
    let mut mad0 = f32x4_splat(0.0);
    let mut mad1 = f32x4_splat(0.0);
    for i in (0..16usize).step_by(2) {
        let d0 = f32x4_sub(pv[i], mp_v);
        let d1 = f32x4_sub(pv[i + 1], mp_v);
        pv[i] = d0;
        pv[i + 1] = d1;
        mad0 = f32x4_add(mad0, f32x4_abs(d0));
        mad1 = f32x4_add(mad1, f32x4_abs(d1));
    }
    let g_v = f32x4_splat((cur_mad / hsum_f32x4(f32x4_add(mad0, mad1)).max(1e-3)).clamp(0.75, 1.33));
    let mut sad0 = f32x4_splat(0.0);
    let mut sad1 = f32x4_splat(0.0);
    for i in (0..16usize).step_by(2) {
        let c0 = v128_load(c_ptr.add(i * 4) as *const v128);
        let c1 = v128_load(c_ptr.add((i + 1) * 4) as *const v128);
        sad0 = f32x4_add(sad0, f32x4_abs(f32x4_sub(c0, f32x4_mul(pv[i], g_v))));
        sad1 = f32x4_add(sad1, f32x4_abs(f32x4_sub(c1, f32x4_mul(pv[i + 1], g_v))));
    }
    hsum_f32x4(f32x4_add(sad0, sad1))
}
