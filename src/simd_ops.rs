use core::arch::wasm32::*;

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

/// Centered Rank-2 Guarded ECA Subspace Regression & Uncensored Evaluation:
/// Models static background + global illumination shift X = [b, 1]:
///   r = y - (a * b + c),  where (a, c) minimize ||y - (a * b + c)||_{2, Omega}^2
/// Computed via two-pass centered covariance/variance to avoid normal-equation cancellation.
/// Evaluates uncensored residual/signal energy strictly over `eval_mask_ptr` (defined from
/// prior-frame stationarity and target guard zones, independent of current residual `r`)
/// BEFORE adapting `b`.
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
    let guard_thresh = f32x4_splat(0.045);
    let one_vec = f32x4_splat(1.0);

    let mut acc_raw_b2 = f32x4_splat(0.0);
    let mut acc_cnt = f32x4_splat(0.0);
    let mut acc_sum_b = f32x4_splat(0.0);
    let mut acc_sum_y = f32x4_splat(0.0);

    // Pass 1: Guarded clutter-region means (bar_b, bar_y) over Omega = {|y - b| < 0.045}
    let mut i = 0usize;
    while i + 4 <= len {
        let y = v128_load(luma_ptr.add(i) as *const v128);
        let b = v128_load(bg_luma_ptr.add(i) as *const v128);

        acc_raw_b2 = f32x4_add(acc_raw_b2, f32x4_mul(b, b));

        let pre_diff = f32x4_abs(f32x4_sub(y, b));
        let in_omega = f32x4_lt(pre_diff, guard_thresh);
        acc_cnt = f32x4_add(acc_cnt, v128_and(one_vec, in_omega));
        acc_sum_b = f32x4_add(acc_sum_b, v128_and(b, in_omega));
        acc_sum_y = f32x4_add(acc_sum_y, v128_and(y, in_omega));

        i += 4;
    }

    let raw_b2 = hsum_f32x4(acc_raw_b2);
    if raw_b2 < 1e-5 {
        // Cold-start initialization: seed background template `b` with initial frame `y`
        let zero = f32x4_splat(0.0);
        let mut k = 0usize;
        while k + 4 <= len {
            let y = v128_load(luma_ptr.add(k) as *const v128);
            v128_store(bg_luma_ptr.add(k) as *mut v128, y);
            v128_store(eca_res_ptr.add(k) as *mut v128, zero);
            k += 4;
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
    }

    let n_omega = hsum_f32x4(acc_cnt);
    let (alpha, offset_c) = if n_omega >= 256.0 {
        let inv_n = 1.0 / n_omega;
        let mean_b = hsum_f32x4(acc_sum_b) * inv_n;
        let mean_y = hsum_f32x4(acc_sum_y) * inv_n;
        let v_mb = f32x4_splat(mean_b);
        let v_my = f32x4_splat(mean_y);

        // Pass 2: Centered variance V_bb and covariance C_by over Omega
        let mut acc_vbb = f32x4_splat(0.0);
        let mut acc_cby = f32x4_splat(0.0);
        let mut m = 0usize;
        while m + 4 <= len {
            let y = v128_load(luma_ptr.add(m) as *const v128);
            let b = v128_load(bg_luma_ptr.add(m) as *const v128);
            let pre_diff = f32x4_abs(f32x4_sub(y, b));
            let in_omega = f32x4_lt(pre_diff, guard_thresh);

            let db = v128_and(f32x4_sub(b, v_mb), in_omega);
            let dy = v128_and(f32x4_sub(y, v_my), in_omega);
            acc_vbb = f32x4_add(acc_vbb, f32x4_mul(db, db));
            acc_cby = f32x4_add(acc_cby, f32x4_mul(db, dy));
            m += 4;
        }

        let vbb = hsum_f32x4(acc_vbb);
        let cby = hsum_f32x4(acc_cby);
        if vbb > 1e-3 {
            let a = (cby / vbb).clamp(0.5, 1.5);
            (a, mean_y - a * mean_b)
        } else {
            // Rank-1 constant offset fallback when background spatial variance is near-zero
            (1.0, mean_y - mean_b)
        }
    } else {
        (1.0, 0.0)
    };

    let v_alpha = f32x4_splat(alpha);
    let v_c = f32x4_splat(offset_c);
    let v_rate_bg = f32x4_splat(adapt_rate);
    let v_rate_fg = f32x4_splat(adapt_rate * 0.008);
    let v_motion_gate = f32x4_splat(0.045);

    let mut acc_eval_err2 = f32x4_splat(0.0);
    let mut acc_eval_sig2 = f32x4_splat(0.0);
    let mut acc_eval_cnt = f32x4_splat(0.0);

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
            let rate = v128_bitselect(v_rate_bg, v_rate_fg, is_bg);
            let b_next = f32x4_add(b, f32x4_mul(rate, f32x4_sub(y, b)));
            v128_store(bg_luma_ptr.add(j) as *mut v128, b_next);

            j += 4;
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
    }

    let err2 = hsum_f32x4(acc_eval_err2);
    let sig2 = hsum_f32x4(acc_eval_sig2);
    let eval_count = hsum_f32x4(acc_eval_cnt) as u32;
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

/// Exported C ABI wrapper preserving the exact `eca_subspace_project_simd128` signature.
#[no_mangle]
pub unsafe extern "C" fn eca_subspace_project_simd128(
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

        let d0 = u8x16_sub_sat(v128_and(v128_or(u8x16_sub_sat(c0, p0), u8x16_sub_sat(p0, c0)), rgb_mask), tol);
        let d1 = u8x16_sub_sat(v128_and(v128_or(u8x16_sub_sat(c1, p1), u8x16_sub_sat(p1, c1)), rgb_mask), tol);
        let d2 = u8x16_sub_sat(v128_and(v128_or(u8x16_sub_sat(c2, p2), u8x16_sub_sat(p2, c2)), rgb_mask), tol);
        let d3 = u8x16_sub_sat(v128_and(v128_or(u8x16_sub_sat(c3, p3), u8x16_sub_sat(p3, c3)), rgb_mask), tol);

        let s0 = i32x4_eq(d0, zero_i32);
        let s1 = i32x4_eq(d1, zero_i32);
        let s2 = i32x4_eq(d2, zero_i32);
        let s3 = i32x4_eq(d3, zero_i32);

        let stat16 = i8x16_narrow_i16x8(
            i16x8_narrow_i32x4(s0, s1),
            i16x8_narrow_i32x4(s2, s3),
        );

        let old_streak = v128_load(stat_streak_ptr.add(p) as *const v128);
        let new_streak = v128_and(u8x16_add_sat(old_streak, one_u8), stat16);
        v128_store(stat_streak_ptr.add(p) as *mut v128, new_streak);

        let eval16 = u8x16_ge(new_streak, min_streak);
        v128_store(eval_mask_ptr.add(p) as *mut v128, eval16);

        p += 16;
    }
}

/// Fast SIMD128 16x16 block ECA + full-RGB temporal frame-difference energy grid (20x11 blocks)
/// and stationary background healing across all 180 scanlines.
pub unsafe fn compute_motion_grid_and_heal_bg_simd128(
    curr_rgba: *const u8,
    prev_rgba: *const u8,
    luma_ptr: *const f32,
    bg0_ptr: *mut f32,
    eca_res_ptr: *const f32,
    grid_out: *mut f32, // 20 * 11 = 220 floats
) {
    let mask_ff = u32x4_splat(0xFF);
    let inv_765 = f32x4_splat(1.0 / (3.0 * 255.0));
    let heal_gate = f32x4_splat(0.006);
    let heal_alpha = f32x4_splat(0.65);

    for gy in 0..11usize {
        for gx in 0..20usize {
            let idx = gy * 20 + gx;
            let base_y = gy * 16;
            let base_x = gx * 16;
            let mut acc_energy = f32x4_splat(0.0);

            for ry in 0..16usize {
                let row_off = (base_y + ry) * 320 + base_x;
                for cx_vec in 0..4usize {
                    let p = row_off + cx_vec * 4;
                    let c_px = v128_load(curr_rgba.add(p * 4) as *const v128);
                    let p_px = v128_load(prev_rgba.add(p * 4) as *const v128);

                    // Full 3-channel RGB temporal delta (prevents blue-only blind spot)
                    let cr = f32x4_convert_u32x4(v128_and(c_px, mask_ff));
                    let pr = f32x4_convert_u32x4(v128_and(p_px, mask_ff));
                    let cg = f32x4_convert_u32x4(v128_and(u32x4_shr(c_px, 8), mask_ff));
                    let pg = f32x4_convert_u32x4(v128_and(u32x4_shr(p_px, 8), mask_ff));
                    let cb = f32x4_convert_u32x4(v128_and(u32x4_shr(c_px, 16), mask_ff));
                    let pb = f32x4_convert_u32x4(v128_and(u32x4_shr(p_px, 16), mask_ff));

                    let rgb_sum = f32x4_add(
                        f32x4_add(f32x4_abs(f32x4_sub(cr, pr)), f32x4_abs(f32x4_sub(cg, pg))),
                        f32x4_abs(f32x4_sub(cb, pb)),
                    );
                    let temp_diff = f32x4_mul(rgb_sum, inv_765);

                    let eca = v128_load(eca_res_ptr.add(p) as *const v128);
                    let is_stat = f32x4_lt(temp_diff, heal_gate);

                    let y_cur = v128_load(luma_ptr.add(p) as *const v128);
                    let bg0 = v128_load(bg0_ptr.add(p) as *const v128);
                    let bg_healed = f32x4_add(bg0, f32x4_mul(heal_alpha, f32x4_sub(y_cur, bg0)));
                    v128_store(
                        bg0_ptr.add(p) as *mut v128,
                        v128_bitselect(bg_healed, bg0, is_stat),
                    );

                    let gated_eca = v128_bitselect(f32x4_splat(0.0), eca, is_stat);
                    let combined = f32x4_add(
                        f32x4_mul(gated_eca, f32x4_splat(0.75)),
                        f32x4_mul(temp_diff, f32x4_splat(0.25)),
                    );
                    acc_energy = f32x4_add(acc_energy, combined);
                }
            }
            *grid_out.add(idx) = hsum_f32x4(acc_energy);
        }
    }

    // Heal stationary background pixels on bottom 4 scanlines (rows 176..180) as well
    for py in 176..180usize {
        let row_off = py * 320;
        let mut px = 0usize;
        while px + 4 <= 320 {
            let p = row_off + px;
            let c_px = v128_load(curr_rgba.add(p * 4) as *const v128);
            let p_px = v128_load(prev_rgba.add(p * 4) as *const v128);
            let cr = f32x4_convert_u32x4(v128_and(c_px, mask_ff));
            let pr = f32x4_convert_u32x4(v128_and(p_px, mask_ff));
            let cg = f32x4_convert_u32x4(v128_and(u32x4_shr(c_px, 8), mask_ff));
            let pg = f32x4_convert_u32x4(v128_and(u32x4_shr(p_px, 8), mask_ff));
            let cb = f32x4_convert_u32x4(v128_and(u32x4_shr(c_px, 16), mask_ff));
            let pb = f32x4_convert_u32x4(v128_and(u32x4_shr(p_px, 16), mask_ff));
            let rgb_sum = f32x4_add(
                f32x4_add(f32x4_abs(f32x4_sub(cr, pr)), f32x4_abs(f32x4_sub(cg, pg))),
                f32x4_abs(f32x4_sub(cb, pb)),
            );
            let temp_diff = f32x4_mul(rgb_sum, inv_765);
            let is_stat = f32x4_lt(temp_diff, heal_gate);
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
    let mut blk_luma = [0.0f32; 16]; // Indexed in Morton Z-order z = 0..16
    let mut blk_rg = [0.0f32; 16];
    let mut blk_by = [0.0f32; 16];

    let inv_16 = 1.0 / 16.0;

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
            let row_base = (y0 + by * 4 + ry) * stride + x0;
            acc_l0 = f32x4_add(acc_l0, v128_load(luma_ptr.add(row_base) as *const v128));
            acc_l1 = f32x4_add(acc_l1, v128_load(luma_ptr.add(row_base + 4) as *const v128));
            acc_l2 = f32x4_add(acc_l2, v128_load(luma_ptr.add(row_base + 8) as *const v128));
            acc_l3 = f32x4_add(acc_l3, v128_load(luma_ptr.add(row_base + 12) as *const v128));

            acc_rg0 = f32x4_add(acc_rg0, v128_load(rg_ptr.add(row_base) as *const v128));
            acc_rg1 = f32x4_add(acc_rg1, v128_load(rg_ptr.add(row_base + 4) as *const v128));
            acc_rg2 = f32x4_add(acc_rg2, v128_load(rg_ptr.add(row_base + 8) as *const v128));
            acc_rg3 = f32x4_add(acc_rg3, v128_load(rg_ptr.add(row_base + 12) as *const v128));

            acc_by0 = f32x4_add(acc_by0, v128_load(by_ptr.add(row_base) as *const v128));
            acc_by1 = f32x4_add(acc_by1, v128_load(by_ptr.add(row_base + 4) as *const v128));
            acc_by2 = f32x4_add(acc_by2, v128_load(by_ptr.add(row_base + 8) as *const v128));
            acc_by3 = f32x4_add(acc_by3, v128_load(by_ptr.add(row_base + 12) as *const v128));
        }

        let z0 = morton2d_4x4(0, by);
        let z1 = morton2d_4x4(1, by);
        let z2 = morton2d_4x4(2, by);
        let z3 = morton2d_4x4(3, by);

        blk_luma[z0] = hsum_f32x4(acc_l0) * inv_16;
        blk_luma[z1] = hsum_f32x4(acc_l1) * inv_16;
        blk_luma[z2] = hsum_f32x4(acc_l2) * inv_16;
        blk_luma[z3] = hsum_f32x4(acc_l3) * inv_16;

        blk_rg[z0] = hsum_f32x4(acc_rg0) * inv_16;
        blk_rg[z1] = hsum_f32x4(acc_rg1) * inv_16;
        blk_rg[z2] = hsum_f32x4(acc_rg2) * inv_16;
        blk_rg[z3] = hsum_f32x4(acc_rg3) * inv_16;

        blk_by[z0] = hsum_f32x4(acc_by0) * inv_16;
        blk_by[z1] = hsum_f32x4(acc_by1) * inv_16;
        blk_by[z2] = hsum_f32x4(acc_by2) * inv_16;
        blk_by[z3] = hsum_f32x4(acc_by3) * inv_16;
    }

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
    let is_chromatic = sat2 > 0.018;

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

/// Continuous SIMD128 16x16 Photometric SSD (Luma + Opponent Chroma) without discrete
/// signature jumps or motion-reward discontinuities. Used for smooth sub-pixel registration.
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
) -> f32 {
    let mut acc_ssd_l = f32x4_splat(0.0);
    let mut acc_ssd_c = f32x4_splat(0.0);

    for ry in 0..16usize {
        let img_row = (cy + ry) * stride + cx;
        let tpl_row = ry * 16;

        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * 4;
            let toff = tpl_row + vec_idx * 4;

            let dl = f32x4_sub(
                v128_load(luma_ptr.add(ioff) as *const v128),
                v128_load(tpl_luma.add(toff) as *const v128),
            );
            acc_ssd_l = f32x4_add(acc_ssd_l, f32x4_mul(dl, dl));

            let drg = f32x4_sub(
                v128_load(rg_ptr.add(ioff) as *const v128),
                v128_load(tpl_rg.add(toff) as *const v128),
            );
            let dby = f32x4_sub(
                v128_load(by_ptr.add(ioff) as *const v128),
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

/// 16x16 Optical Patch Tracker Cost using SIMD128:
/// Combines Luma SAD, Chroma (R-G, B-Y) SAD, ECA motion-residual reward,
/// and binary-prefix 2-adic ultrametric valuation penalty.
#[no_mangle]
pub unsafe extern "C" fn sad16x16_patch_simd128(
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
    let mut acc_sad_l = f32x4_splat(0.0);
    let mut acc_sad_c = f32x4_splat(0.0);
    let mut acc_eca = f32x4_splat(0.0);

    for ry in 0..16usize {
        let img_row = (cy + ry) * stride + cx;
        let tpl_row = ry * 16;

        for vec_idx in 0..4usize {
            let ioff = img_row + vec_idx * 4;
            let toff = tpl_row + vec_idx * 4;

            let il = v128_load(luma_ptr.add(ioff) as *const v128);
            let tl = v128_load(tpl_luma.add(toff) as *const v128);
            acc_sad_l = f32x4_add(acc_sad_l, f32x4_abs(f32x4_sub(il, tl)));

            let irg = v128_load(rg_ptr.add(ioff) as *const v128);
            let trg = v128_load(tpl_rg.add(toff) as *const v128);
            let iby = v128_load(by_ptr.add(ioff) as *const v128);
            let tby = v128_load(tpl_by.add(toff) as *const v128);

            let diff_rg = f32x4_abs(f32x4_sub(irg, trg));
            let diff_by = f32x4_abs(f32x4_sub(iby, tby));
            acc_sad_c = f32x4_add(acc_sad_c, f32x4_add(diff_rg, diff_by));

            let eca = v128_load(eca_res_ptr.add(ioff) as *const v128);
            acc_eca = f32x4_add(acc_eca, eca);
        }
    }

    let sad_l = hsum_f32x4(acc_sad_l) * (1.0 / 256.0);
    let sad_c = hsum_f32x4(acc_sad_c) * (1.0 / 256.0);
    let motion_reward = (hsum_f32x4(acc_eca) * (1.0 / 256.0)).min(0.35);

    let cand_sig = haar_morton_signature_simd128(luma_ptr, rg_ptr, by_ptr, stride, cx, cy);
    if !out_cand_sig.is_null() {
        *out_cand_sig = cand_sig;
    }

    let val = padic_ultrametric_val(tpl_sig, cand_sig);
    let d2 = padic_distance_from_val(val);

    sad_l + 1.85 * sad_c + 0.55 * d2 - 0.18 * motion_reward
}
