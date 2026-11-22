//! Coarse-to-fine camera pan, cut, and exposure estimation.

use super::*;

#[inline]
fn eval_single_patch_raw_sad(
    cur_patch: &[f32; 64],
    cur_mad: f32,
    px: usize,
    py: usize,
    prev_rgba: &[u8],
    frame_w: usize,
    scale_px: usize,
    phase_mask: usize,
    cdx: i32,
    cdy: i32,
) -> f32 {
    let frame_h = prev_rgba.len() / (frame_w * 4);
    let max_sx = (frame_w - 9 * scale_px) as i32;
    let max_sy = (frame_h - 9 * scale_px) as i32;
    let sx0 = ((px as i32) - cdx).clamp(0, max_sx) as usize;
    let sy0 = ((py as i32) - cdy).clamp(0, max_sy) as usize;
    unsafe {
        eval_patch_raw_sad_rgba_simd128(
            cur_patch,
            cur_mad,
            prev_rgba,
            frame_w,
            sx0,
            sy0,
            scale_px,
            phase_mask,
        )
    }
}

#[inline(always)]
fn eval_trimmed_sad_with<F: FnMut(usize) -> f32>(
    n_patches: usize,
    cutoff: f32,
    mut eval_one: F,
) -> f32 {
    // Trim the worst 2-3 patches when >= 4 patches exist so foreground targets crossing a tile never bias registration.
    // Maintaining the top `trim_cnt` largest values yields a strictly non-decreasing lower bound `sum - sum(top)`
    // for exact monotone early cutoff.
    let trim_cnt = if n_patches >= 6 {
        3usize
    } else if n_patches >= 4 {
        2usize
    } else {
        0usize
    };
    let mut sum = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    for pid in 0..n_patches {
        let s = eval_one(pid);
        sum += s;
        if trim_cnt == 3 {
            if s > b2 {
                if s > b1 {
                    b2 = b1;
                    if s > b0 {
                        b1 = b0;
                        b0 = s;
                    } else {
                        b1 = s;
                    }
                } else {
                    b2 = s;
                }
            }
            if pid >= 3 && (sum - (b0 + b1 + b2)) >= cutoff {
                return sum - (b0 + b1 + b2);
            }
        } else if trim_cnt == 2 {
            if s > b1 {
                if s > b0 {
                    b1 = b0;
                    b0 = s;
                } else {
                    b1 = s;
                }
            }
            if pid >= 2 && (sum - (b0 + b1)) >= cutoff {
                return sum - (b0 + b1);
            }
        } else if sum >= cutoff {
            return sum;
        }
    }
    sum - (b0 + b1 + b2)
}

#[inline]
fn eval_ref_patches_trimmed_sad(
    cur_patches: &[[f32; 64]; 8],
    cur_mads: &[f32; 8],
    patch_px: &[usize; 8],
    patch_py: &[usize; 8],
    n_patches: usize,
    prev_rgba: &[u8],
    frame_w: usize,
    scale_px: usize,
    phase_mask: usize,
    cdx: i32,
    cdy: i32,
    cutoff: f32,
) -> f32 {
    eval_trimmed_sad_with(n_patches, cutoff, |pid| {
        eval_single_patch_raw_sad(
            &cur_patches[pid],
            cur_mads[pid],
            patch_px[pid],
            patch_py[pid],
            prev_rgba,
            frame_w,
            scale_px,
            phase_mask,
            cdx,
            cdy,
        )
    })
}

#[derive(Clone, Copy)]
pub(crate) struct WarpModel {
    /// 0 = pure translation (Tier-0/1), 1 = 4-DoF similarity, 2 = 6-DoF affine, 3 = 8-DoF homography.
    pub(crate) tier: u8,
    /// Centered forward warp (prev -> cur) around `(cx, cy)`:
    ///   xr' = ((1+h0)*xr + h1*yr + h2) / (1 + h6*xr + h7*yr)
    ///   yr' = (h3*xr + (1+h4)*yr + h5) / (1 + h6*xr + h7*yr)
    pub(crate) h: [f32; 8],
}

impl WarpModel {
    #[inline(always)]
    pub(crate) const fn identity() -> Self {
        Self { tier: 0, h: [0.0; 8] }
    }

    #[inline(always)]
    pub(crate) fn from_similarity(tx: f32, ty: f32, a: f32, b: f32) -> Self {
        Self {
            tier: 1,
            h: [a, -b, tx, b, a, ty, 0.0, 0.0],
        }
    }

    #[inline(always)]
    pub(crate) fn warp_pt(&self, x: f32, y: f32, cx: f32, cy: f32) -> (f32, f32) {
        let xr = x - cx;
        let yr = y - cy;
        let denom = 1.0 + self.h[6] * xr + self.h[7] * yr;
        let inv_d = if denom.abs() > 0.25 { 1.0 / denom } else { 1.0 };
        let nx = cx + ((1.0 + self.h[0]) * xr + self.h[1] * yr + self.h[2]) * inv_d;
        let ny = cy + (self.h[3] * xr + (1.0 + self.h[4]) * yr + self.h[5]) * inv_d;
        (nx, ny)
    }

    #[inline(always)]
    pub(crate) fn warp_vel(&self, vx: f32, vy: f32, x: f32, y: f32, cx: f32, cy: f32) -> (f32, f32) {
        let xr = x - cx;
        let yr = y - cy;
        let a00 = 1.0 + self.h[0];
        let a01 = self.h[1];
        let a10 = self.h[3];
        let a11 = 1.0 + self.h[4];
        if self.tier <= 2 {
            return (a00 * vx + a01 * vy, a10 * vx + a11 * vy);
        }
        let d = 1.0 + self.h[6] * xr + self.h[7] * yr;
        let inv_d = if d.abs() > 0.25 { 1.0 / d } else { 1.0 };
        let u = (a00 * xr + a01 * yr + self.h[2]) * inv_d;
        let v = (a10 * xr + a11 * yr + self.h[5]) * inv_d;
        let j00 = (a00 - u * self.h[6]) * inv_d;
        let j01 = (a01 - u * self.h[7]) * inv_d;
        let j10 = (a10 - v * self.h[6]) * inv_d;
        let j11 = (a11 - v * self.h[7]) * inv_d;
        (j00 * vx + j01 * vy, j10 * vx + j11 * vy)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PanMotion {
    pub(crate) scene_cut: bool,
    pub(crate) dx: i32,
    pub(crate) dy: i32,
    pub(crate) dx_f: f32,
    pub(crate) dy_f: f32,
    pub(crate) frac_mag: f32,
    pub(crate) exp_gain: f32,
    pub(crate) exp_off: f32,
    pub(crate) exp_regime: bool,
    pub(crate) warp: WarpModel,
}

impl PanMotion {
    #[inline(always)]
    const fn stationary(exp_regime: bool) -> Self {
        Self {
            scene_cut: false,
            dx: 0,
            dy: 0,
            dx_f: 0.0,
            dy_f: 0.0,
            frac_mag: 0.0,
            exp_gain: 1.0,
            exp_off: 0.0,
            exp_regime,
            warp: WarpModel::identity(),
        }
    }

    #[inline(always)]
    const fn cut() -> Self {
        Self {
            scene_cut: true,
            dx: 0,
            dy: 0,
            dx_f: 0.0,
            dy_f: 0.0,
            frac_mag: 0.0,
            exp_gain: 1.0,
            exp_off: 0.0,
            exp_regime: false,
            warp: WarpModel::identity(),
        }
    }
}

#[inline(always)]
pub(crate) fn unpack_cam_hold(raw: u32) -> (u8, u8, f32, f32) {
    let cut_hold = (raw & 0xFF) as u8;
    let exp_hold = ((raw >> 8) & 0xFF) as u8;
    let vx_q = ((raw >> 16) & 0xFF) as u8 as i8 as i32;
    let vy_q = ((raw >> 24) & 0xFF) as u8 as i8 as i32;
    let unpack_v = |qi: i32| -> f32 {
        let aq = qi.abs();
        if aq <= 64 {
            (qi as f32) * 0.125
        } else {
            let mag = 8.0 + ((aq - 64) as f32);
            if qi >= 0 { mag } else { -mag }
        }
    };
    (cut_hold, exp_hold, unpack_v(vx_q), unpack_v(vy_q))
}

#[inline(always)]
pub(crate) fn pack_cam_hold(cut_hold: u8, exp_hold: u8, vx_f: f32, vy_f: f32) -> u32 {
    let pack_v = |v: f32| -> u8 {
        let av = v.abs();
        let q = if av <= 8.0 {
            let r = if v >= 0.0 {
                (v * 8.0 + 0.5) as i32
            } else {
                (v * 8.0 - 0.5) as i32
            };
            r.clamp(-64, 64)
        } else {
            let mag = (64 + ((av - 8.0 + 0.5) as i32)).min(127);
            if v >= 0.0 { mag } else { -mag }
        };
        q as i8 as u8
    };
    let qx = pack_v(vx_f);
    let qy = pack_v(vy_f);
    (cut_hold as u32)
        | ((exp_hold as u32) << 8)
        | ((qx as u32) << 16)
        | ((qy as u32) << 24)
}

#[inline]
fn fit_affine_rgb_trimmed(bs: &[f32], ys: &[f32]) -> Option<(f32, f32, f32, f32)> {
    let n = bs.len().min(ys.len());
    let mut out = [0.0f32; 4];
    unsafe {
        if fit_affine_rgb_trimmed_simd128(bs.as_ptr(), ys.as_ptr(), n, &mut out) {
            Some((out[0], out[1], out[2], out[3]))
        } else {
            None
        }
    }
}

#[inline]
fn eval_subpx_patch_gain_norm_sad(
    cur_patch: &[f32; 64],
    cur_mad: f32,
    pl: &[f32],
    tile_stride: usize,
    scale_px: usize,
    phase_mask: usize,
    off_x_base: usize,
    off_y_base: usize,
    w00: f32,
    w10: f32,
    w01: f32,
    w11: f32,
) -> f32 {
    unsafe {
        subpx_patch_gain_norm_sad_simd128(
            cur_patch, cur_mad, pl, tile_stride, scale_px, phase_mask, off_x_base, off_y_base, w00,
            w10, w01, w11,
        )
    }
}

#[inline(never)]
fn estimate_pan_exposure_symmetric(
    curr_rgba: &[u8],
    prev_rgba: &[u8],
    scratch: &mut [f32],
    frame_w: usize,
    frame_h: usize,
    pan_dx_round: i32,
    pan_dy_round: i32,
    pan_dx_f: f32,
    pan_dy_f: f32,
) -> (f32, f32) {
    if scratch.len() < 4096 || frame_w < 2 || frame_h < 2 {
        return (1.0, 0.0);
    }
    let mut go = [1.0f32, 0.0f32];
    unsafe {
        estimate_pan_exposure_symmetric_simd128(
            curr_rgba.as_ptr(),
            prev_rgba.as_ptr(),
            scratch.as_mut_ptr(),
            frame_w,
            frame_h,
            pan_dx_round,
            pan_dy_round,
            pan_dx_f,
            pan_dy_f,
            &mut go,
        );
    }
    (go[0], go[1])
}

#[inline]
fn pick_textured_subpatch(
    luma: &[f32],
    frame_w: usize,
    scale_px: usize,
    sub_step: usize,
    min_px: usize,
    max_px: usize,
    min_py: usize,
    max_py: usize,
    tx: usize,
    ty: usize,
) -> (usize, usize, f32) {
    let base_x = (tx * 2 * scale_px).clamp(min_px, max_px);
    let base_y = (ty * 2 * scale_px).clamp(min_py, max_py);
    if scale_px == 2
        && sub_step == 4
        && base_x + 12 <= max_px
        && base_y + 8 <= max_py
        && base_x + 28 <= frame_w
        && luma.len() >= (base_y + 23) * frame_w + base_x + 28
    {
        return unsafe { pick_textured_subpatch_5band_hd_simd128(luma, frame_w, base_x, base_y) };
    }
    if scale_px == 2
        && sub_step == 16
        && base_x + 48 <= max_px
        && base_y + 32 <= max_py
        && base_x + 64 <= frame_w
        && luma.len() >= (base_y + 47) * frame_w + base_x + 64
    {
        return unsafe { pick_textured_subpatch_step16_hd_simd128(luma, frame_w, base_x, base_y) };
    }
    let mut best_tex2d = 0.0f32;
    let mut best_bx = base_x;
    let mut best_by = base_y;
    let span = 8 * scale_px - 1;
    for oy in 0..3usize {
        let cy = (base_y + oy * sub_step).min(max_py);
        for ox in 0..4usize {
            let cx = (base_x + ox * sub_step).min(max_px);
            let (sxx, syy, sxy, l1) = if scale_px == 2 {
                unsafe { patch_grad_tensor_span15_simd128(luma, frame_w, cx, cy) }
            } else {
                unsafe { patch_grad_tensor_span_simd128(luma, frame_w, cx, cy, span) }
            };
            if 0.5 * l1 <= best_tex2d {
                continue;
            }
            let tr = sxx + syy;
            let det = (sxx * syy - sxy * sxy).max(0.0);
            if det < 0.09566326 * tr * tr {
                continue;
            }
            let upper_t2d = 0.5 * l1;
            if best_tex2d > 0.0 {
                let need = (best_tex2d * 0.999) / upper_t2d;
                let r_req = need * need;
                if det * (1.0 + r_req) * (1.0 + r_req) < r_req * tr * tr {
                    continue;
                }
            }
            let disc = wasm_sqrtf((tr * tr - 4.0 * det).max(0.0));
            let lam_min = 0.5 * (tr - disc);
            let lam_max = 0.5 * (tr + disc);
            let t2d = if lam_min >= 0.12 * lam_max {
                upper_t2d * wasm_sqrtf(lam_min / lam_max.max(1e-8))
            } else {
                0.0
            };
            if t2d > best_tex2d {
                best_tex2d = t2d;
                best_bx = cx;
                best_by = cy;
            }
        }
    }
    (best_bx, best_by, best_tex2d)
}

#[inline]
fn sample_cur_luma_patch(
    luma: &[f32],
    frame_w: usize,
    scale_px: usize,
    phase_mask: usize,
    bx: usize,
    by: usize,
) -> ([f32; 64], f32) {
    let mut patch = [0.0f32; 64];
    unsafe {
        if scale_px == 2
            && phase_mask == 1
            && bx + 17 <= frame_w
            && (by + 15) * frame_w + bx + 17 <= luma.len()
        {
            let mad_c = sample_luma_patch64_p2_simd128(luma.as_ptr(), frame_w, bx, by, &mut patch);
            return (patch, mad_c);
        }
        let lp = luma.as_ptr().add(by * frame_w + bx);
        let mut q = 0usize;
        for r in 0..8usize {
            let r_base = (r * scale_px) * frame_w + (r & phase_mask);
            for c in 0..8usize {
                let off = r_base + (c & phase_mask) * frame_w + c * scale_px;
                patch[q] = *lp.add(off);
                q += 1;
            }
        }
        let mad_c = demean_mad64_simd128(&mut patch);
        (patch, mad_c)
    }
}

#[inline]
fn find_top_2d_patches(
    luma: &[f32],
    targets: &[TargetTrack],
    peaks: &[CandidatePeak],
    num_peaks: usize,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    min_t2d: f32,
    min_mad: f32,
    out_bx: &mut [usize; 16],
    out_by: &mut [usize; 16],
) -> usize {
    let mv_excl = 0.5 * (scale_px as f32);
    let mv_excl2 = mv_excl * mv_excl;
    let exc_excl = (8 * scale_px) as f32;
    let exc_excl2 = exc_excl * exc_excl;
    let mut excl_x = [0.0f32; 32];
    let mut excl_y = [0.0f32; 32];
    let mut n_excl = 0usize;

    for t in targets.iter() {
        if t.active != 0
            && (t.confirmed != 0
                || t.max_excursion2 >= exc_excl2
                || (t.vel_x * t.vel_x + t.vel_y * t.vel_y) >= mv_excl2)
            && n_excl < 32
        {
            excl_x[n_excl] = t.pos_x;
            excl_y[n_excl] = t.pos_y;
            n_excl += 1;
        }
    }
    let min_pk_e = 65.0 * (scale_px as f32);
    for p in 0..num_peaks {
        if peaks[p].energy >= min_pk_e && n_excl < 32 {
            excl_x[n_excl] = peaks[p].x;
            excl_y[n_excl] = peaks[p].y;
            n_excl += 1;
        }
    }

    unsafe {
        simd_ops::find_top_2d_patches_simd128(
            luma, frame_w, frame_h, scale_px, min_t2d, min_mad, &excl_x, &excl_y, n_excl, out_bx, out_by,
        )
    }
}

#[inline]
fn find_best_win_shift_simd128(
    cur_q: &[f32; 64],
    mad_cq: f32,
    prev_win: &[f32],
    phase_scratch: &mut [f32],
    win_dim: usize,
    f_r: i32,
    scale_px: usize,
    z_sad: f32,
    hint_dx: i32,
    hint_dy: i32,
    eval_center: bool,
    p2_prepacked: (usize, usize),
) -> (f32, i32, i32) {
    let mut b_sad = z_sad;
    let mut b_dx = 0i32;
    let mut b_dy = 0i32;

    #[cfg(all(feature = "hires-720p", not(target_arch = "wasm32")))]
    const MAX_GRID: usize = 64;
    #[cfg(not(all(feature = "hires-720p", not(target_arch = "wasm32"))))]
    const MAX_GRID: usize = 33;
    let mut seen = [0u64; MAX_GRID];
    if !eval_center {
        let c = f_r as usize;
        seen[c] |= 1u64 << c;
    }

    let mut cur_q_eo = [0.0f32; 64];
    let prepacked = p2_prepacked.0 > 0;
    if prepacked {
        unsafe {
            reorder_patch64_eo_simd128(cur_q, &mut cur_q_eo);
        }
    }

    let eval_pt_init = |ox: usize, oy: usize| -> f32 {
        if prepacked {
            unsafe {
                eval_win_patch_sad_p2_packed_simd128(
                    &cur_q_eo,
                    mad_cq,
                    prev_win,
                    p2_prepacked.0,
                    p2_prepacked.1,
                    ox,
                    oy,
                    1.0e9f32,
                )
            }
        } else {
            unsafe { eval_win_patch_sad_simd128(cur_q, mad_cq, prev_win, win_dim, ox, oy, scale_px) }
        }
    };

    // Step B first: 5x5 neighborhood around previous-frame integer pan hint
    if eval_center || hint_dx != 0 || hint_dy != 0 {
        let hx = hint_dx.clamp(-f_r, f_r);
        let hy = hint_dy.clamp(-f_r, f_r);
        for cdy in (hy - 2).max(-f_r)..=(hy + 2).min(f_r) {
            let gy = (cdy + f_r) as usize;
            for cdx in (hx - 2).max(-f_r)..=(hx + 2).min(f_r) {
                let gx = (cdx + f_r) as usize;
                let bit = 1u64 << gx;
                if (seen[gy] & bit) == 0 {
                    seen[gy] |= bit;
                    let sad = eval_pt_init((f_r - cdx) as usize, (f_r - cdy) as usize);
                    if sad < b_sad {
                        b_sad = sad;
                        b_dx = cdx;
                        b_dy = cdy;
                    }
                }
            }
        }
        let stepb_lim = if eval_center {
            0.68
        } else if hint_dx != 0 || hint_dy != 0 {
            0.60
        } else {
            0.48
        };
        if b_sad < stepb_lim * z_sad && (z_sad - b_sad) > 0.0020 {
            return (b_sad, b_dx, b_dy);
        }
    }

    // Step A: Dense small-shift core (`-2..=2`) for exact 1-px parity at slow speeds
    let core_r = f_r.min(2);
    for cdy in -core_r..=core_r {
        let gy = (cdy + f_r) as usize;
        for cdx in -core_r..=core_r {
            let gx = (cdx + f_r) as usize;
            let bit = 1u64 << gx;
            if (seen[gy] & bit) == 0 {
                seen[gy] |= bit;
                let sad = eval_pt_init((f_r - cdx) as usize, (f_r - cdy) as usize);
                if sad < b_sad {
                    b_sad = sad;
                    b_dx = cdx;
                    b_dy = cdy;
                }
            }
        }
    }
    if f_r <= 2 || (b_sad < 0.54 * z_sad && (z_sad - b_sad) > 0.0020) {
        return (b_sad, b_dx, b_dy);
    }
    if !eval_center && hint_dx == 0 && hint_dy == 0 && z_sad < 0.015 && b_sad >= 0.90 * z_sad {
        let probe_r = f_r.min(4);
        let probes = [(-probe_r, 0), (probe_r, 0), (0, -probe_r), (0, probe_r)];
        let mut hit = false;
        for &(cdx, cdy) in &probes {
            let gy = (cdy + f_r) as usize;
            let gx = (cdx + f_r) as usize;
            let bit = 1u64 << gx;
            if (seen[gy] & bit) == 0 {
                seen[gy] |= bit;
                let sad = eval_pt_init((f_r - cdx) as usize, (f_r - cdy) as usize);
                if sad < b_sad {
                    b_sad = sad;
                    b_dx = cdx;
                    b_dy = cdy;
                }
                if sad < 0.90 * z_sad {
                    hit = true;
                }
            }
        }
        if !hit {
            return (b_sad, b_dx, b_dy);
        }
    }

    let (use_p2_pack, p_w, plane_sz) = if prepacked {
        (true, p2_prepacked.0, p2_prepacked.1)
    } else if scale_px == 2 && f_r >= 4 {
        let p_h = (win_dim + 1) / 2;
        let pw = ((win_dim + 3) / 4) + 1;
        let psz = p_h * pw;
        if phase_scratch.len() >= 8 * psz && win_dim >= 16 {
            unsafe {
                reorder_patch64_eo_simd128(cur_q, &mut cur_q_eo);
                pack_win_p2_phases_simd128(prev_win, win_dim, phase_scratch);
            }
            (true, pw, psz)
        } else {
            (false, 0, 0)
        }
    } else {
        (false, 0, 0)
    };

    let eval_sad = |ox: usize, oy: usize, max_sad64: f32| -> f32 {
        if use_p2_pack {
            let buf = if prepacked { prev_win } else { &*phase_scratch };
            unsafe {
                eval_win_patch_sad_p2_packed_simd128(
                    &cur_q_eo,
                    mad_cq,
                    buf,
                    p_w,
                    plane_sz,
                    ox,
                    oy,
                    max_sad64,
                )
            }
        } else {
            unsafe { eval_win_patch_sad_simd128(cur_q, mad_cq, prev_win, win_dim, ox, oy, scale_px) }
        }
    };

    // Step C: Outer ring stride-2 + top-3 3x3 refinement
    let mut o_sad = [1.0e9f32; 3];
    let mut o_dx = [0i32; 3];
    let mut o_dy = [0i32; 3];
    let mut cdy = -f_r;
    while cdy <= f_r {
        let gy = (cdy + f_r) as usize;
        let mut cdx = -f_r;
        while cdx <= f_r {
            let gx = (cdx + f_r) as usize;
            let bit = 1u64 << gx;
            if (seen[gy] & bit) == 0 {
                seen[gy] |= bit;
                let upper_tol = (b_sad * 1.35).min(1.02 * z_sad);
                let max_c = o_sad[2].min(upper_tol);
                let max_sad64 = if max_c < 1e7 { max_c * 64.0 + 1e-4 } else { 1.0e9f32 };
                let sad = eval_sad((f_r - cdx) as usize, (f_r - cdy) as usize, max_sad64);
                if sad < b_sad {
                    b_sad = sad;
                    b_dx = cdx;
                    b_dy = cdy;
                }
                if sad < o_sad[0] {
                    o_sad[2] = o_sad[1];
                    o_dx[2] = o_dx[1];
                    o_dy[2] = o_dy[1];
                    o_sad[1] = o_sad[0];
                    o_dx[1] = o_dx[0];
                    o_dy[1] = o_dy[0];
                    o_sad[0] = sad;
                    o_dx[0] = cdx;
                    o_dy[0] = cdy;
                } else if sad < o_sad[1] {
                    o_sad[2] = o_sad[1];
                    o_dx[2] = o_dx[1];
                    o_dy[2] = o_dy[1];
                    o_sad[1] = sad;
                    o_dx[1] = cdx;
                    o_dy[1] = cdy;
                } else if sad < o_sad[2] {
                    o_sad[2] = sad;
                    o_dx[2] = cdx;
                    o_dy[2] = cdy;
                }
            }
            cdx += 2;
        }
        cdy += 2;
    }

    let max_ti = if f_r <= 4 { 1 } else { 3 };
    for ti in 0..max_ti {
        if o_sad[ti] >= 1.0e8
            || (ti == 0 && o_sad[0] > (b_sad * 1.35).min(1.02 * z_sad))
            || (ti > 0 && o_sad[ti] > (b_sad * 1.25).min(0.95 * z_sad))
        {
            continue;
        }
        let cx = o_dx[ti];
        let cy = o_dy[ti];
        for oy_i in (cy - 1).max(-f_r)..=(cy + 1).min(f_r) {
            let gy = (oy_i + f_r) as usize;
            for ox_i in (cx - 1).max(-f_r)..=(cx + 1).min(f_r) {
                let gx = (ox_i + f_r) as usize;
                let bit = 1u64 << gx;
                if (seen[gy] & bit) == 0 {
                    seen[gy] |= bit;
                    let sad = eval_sad((f_r - ox_i) as usize, (f_r - oy_i) as usize, b_sad * 64.0 + 1e-4);
                    if sad < b_sad {
                        b_sad = sad;
                        b_dx = ox_i;
                        b_dy = oy_i;
                    }
                }
            }
        }
    }

    // Step D: Exhaustive fallback bypassed (Step A 5x5 + Step B + Step C stride-2 + 3x3 covers coherent pans)
    if false && b_sad >= 0.72 * z_sad && z_sad > 0.025 {
        for cdy in -f_r..=f_r {
            let gy = (cdy + f_r) as usize;
            for cdx in -f_r..=f_r {
                let gx = (cdx + f_r) as usize;
                if (seen[gy] & (1u64 << gx)) == 0 {
                    let sad = eval_sad((f_r - cdx) as usize, (f_r - cdy) as usize, b_sad * 64.0 + 1e-4);
                    if sad < b_sad {
                        b_sad = sad;
                        b_dx = cdx;
                        b_dy = cdy;
                    }
                }
            }
        }
    }

    (b_sad, b_dx, b_dy)
}

#[cold]
#[inline(never)]
unsafe fn cam_coarse_3x3_fallback(
    pyr_luma: &[f32],
    prev_pyr_luma: &[f32],
    coarse_cx: i32,
    coarse_cy: i32,
    pan_votes: &mut usize,
    vx_votes: &mut [i32; 16],
    vy_votes: &mut [i32; 16],
    vote_tx: &mut [usize; 16],
    vote_ty: &mut [usize; 16],
    best_cluster: &mut usize,
    best_anchor: &mut usize,
    sum_cx: &mut f32,
    sum_cy: &mut f32,
) {
    const TILE_X: [usize; 4] = [8, 48, 96, 136];
    const TILE_Y: [usize; 4] = [6, 26, 50, 70];
    let mut fb_any = false;
    for &ty in &TILE_Y {
        for &tx in &TILE_X {
            let mut cur_sub = [0.0f32; 48];
            let (zero_sad, _, _, tex_mad, _, g_l1, iso2d) = cam_pyr_tile_4x6x8_simd128(
                pyr_luma,
                prev_pyr_luma,
                PYR_W,
                tx,
                ty,
                &mut cur_sub,
            );
            if zero_sad <= 0.016 || zero_sad > 0.055 {
                continue;
            }
            if !(tex_mad > 0.006 && iso2d && g_l1 > 0.18) {
                continue;
            }
            fb_any = true;
            let sad_cap = (0.56 * zero_sad).min(zero_sad - 0.006);
            let (best_sad, best_dx, best_dy) = eval_cam_coarse_tile_simd128(
                cur_sub.as_ptr(),
                prev_pyr_luma.as_ptr(),
                PYR_W,
                PYR_H,
                tx,
                ty,
                coarse_cx,
                coarse_cy,
                tex_mad,
                zero_sad,
                sad_cap,
            );
            if best_sad < 0.56 * zero_sad && (zero_sad - best_sad) > 0.006 {
                let mut found = false;
                for k in 0..*pan_votes {
                    if vote_tx[k] == tx && vote_ty[k] == ty {
                        vx_votes[k] = best_dx;
                        vy_votes[k] = best_dy;
                        found = true;
                        break;
                    }
                }
                if !found {
                    let pv = *pan_votes;
                    vx_votes[pv] = best_dx;
                    vy_votes[pv] = best_dy;
                    vote_tx[pv] = tx;
                    vote_ty[pv] = ty;
                    *pan_votes = pv + 1;
                }
            }
        }
    }
    if !fb_any {
        return;
    }
    *best_cluster = 0;
    let pv = *pan_votes;
    if pv >= 5 {
        for i in 0..pv {
            let mut cnt = 0usize;
            let mut sx = 0i32;
            let mut sy = 0i32;
            for j in 0..pv {
                if (vx_votes[i] - vx_votes[j]).abs() <= 1 && (vy_votes[i] - vy_votes[j]).abs() <= 1 {
                    cnt += 1;
                    sx += vx_votes[j];
                    sy += vy_votes[j];
                }
            }
            if cnt > *best_cluster {
                *best_cluster = cnt;
                *best_anchor = i;
                *sum_cx = (sx as f32) / (cnt as f32);
                *sum_cy = (sy as f32) / (cnt as f32);
            }
        }
    }
}

#[inline(never)]
unsafe fn swarm_pan_flat_canvas_reject(
    a: &Arena,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
) -> bool {
    if frame_w < 8 || frame_h < 4 || scale_px == 0 {
        return false;
    }
    let w_m6 = frame_w.saturating_sub(6);
    let h_m3 = frame_h.saturating_sub(3);
    let r0 = a.luma[2 * frame_w + 2];
    let r1 = a.luma[h_m3 * frame_w + w_m6];
    let r2 = a.luma[2 * frame_w + (frame_w >> 1)];
    let bg_ref = if (r0 - r1).abs() <= 0.012 || (r0 - r2).abs() <= 0.012 {
        r0
    } else {
        r1
    };
    let y_step = 2 * scale_px;
    let n_x = (frame_w - 4) / 4;
    let n_y = (frame_h - 4) / y_step + 1;
    let tot_expected = n_x * 4 * n_y;
    let max_non_flat = tot_expected / 4;
    let mut flat_cnt = 0usize;
    let mut tot_cnt = 0usize;
    let mut y = 2usize;
    let (v_gthresh, v_cthresh, v_bg) = {
        use crate::simd::f32x4_splat;
        (f32x4_splat(0.003), f32x4_splat(0.012), f32x4_splat(bg_ref))
    };
    while y + 1 < frame_h {
        let row0 = y * frame_w;
        let row1 = row0 + frame_w;
        let mut x = 2usize;
        while x + 5 < frame_w {
            {
                use crate::simd::{
                    f32x4_abs, f32x4_add, f32x4_le, f32x4_sub, i32x4_bitmask, v128, v128_and,
                };
                let p0 = a.luma.as_ptr().add(row0 + x);
                let p1 = a.luma.as_ptr().add(row1 + x);
                let v0 = core::ptr::read_unaligned(p0 as *const v128);
                let vx = core::ptr::read_unaligned(p0.add(1) as *const v128);
                let vy = core::ptr::read_unaligned(p1 as *const v128);
                let gsum = f32x4_add(
                    f32x4_abs(f32x4_sub(vx, v0)),
                    f32x4_abs(f32x4_sub(vy, v0)),
                );
                let m = i32x4_bitmask(v128_and(
                    f32x4_le(gsum, v_gthresh),
                    f32x4_le(f32x4_abs(f32x4_sub(v0, v_bg)), v_cthresh),
                ));
                flat_cnt += (m as u32).count_ones() as usize;
                tot_cnt += 4;
            }
            x += 4;
        }
        if tot_cnt - flat_cnt > max_non_flat {
            return false;
        }
        y += y_step;
    }
    tot_cnt > 0 && flat_cnt * 4 >= tot_cnt * 3
}

/// Coarse 16-tile demeaned `160x90` pyramid camera-motion and scene-cut detector,
/// coupled with 1-px slow-pan fallback, gain-invariant eighth-pixel 2D bilinear SAD refinement,
/// symmetric exposure-step compensation, 4-tap SIMD128 bilinear background advection, and world-frame registration.
pub(crate) unsafe fn detect_camera_motion_coarse(
    a: &mut Arena,
    frame_w: usize,
    frame_h: usize,
    frame_pixels: usize,
    scale_px: usize,
    slot_cur: usize,
    slot_prev: usize,
    s: f32,
    step_dt: f32,
) -> PanMotion {
    const TILE_X: [usize; 4] = [8, 48, 96, 136];
    const TILE_Y: [usize; 4] = [6, 26, 50, 70];

    let f_idx = a.frame_idx;
    if f_idx == 0 {
        a.prev_pyr_luma.copy_from_slice(&a.pyr_luma);
        a.cam_motion_state = 0;
        a.last_pan_dx = 0;
        a.last_pan_dy = 0;
        a.cam_cut_hold = 0;
        a.hist_cam_x[0] = a.cum_cam_x;
        a.hist_cam_y[0] = a.cum_cam_y;
        return PanMotion::stationary(false);
    }

    let (cut_hold_in, mut exp_hold, prev_vx, prev_vy) = unpack_cam_hold(a.cam_cut_hold);
    let round_f = |v: f32| -> i32 {
        if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            (v - 0.5) as i32
        }
    };
    let prev_v2 = prev_vx * prev_vx + prev_vy * prev_vy;
    let (coarse_cx, coarse_cy) = if a.cam_pan_streak >= 1 && prev_v2 >= (2.0 * s) * (2.0 * s) {
        let inv_pyr_step = 1.0 / (2.0 * s);
        let max_cx = ((PYR_W as i32) / 3).max(4);
        let max_cy = ((PYR_H as i32) / 3).max(3);
        (
            round_f(prev_vx * inv_pyr_step).clamp(-max_cx, max_cx),
            round_f(prev_vy * inv_pyr_step).clamp(-max_cy, max_cy),
        )
    } else {
        (0, 0)
    };
    let (pred_dx, pred_dy) = if a.cam_pan_streak >= 1 && prev_v2 >= (2.0 * s) * (2.0 * s) {
        let lim_x = ((frame_w as i32) / 3).max(12);
        let lim_y = ((frame_h as i32) / 3).max(12);
        (
            round_f(prev_vx).clamp(-lim_x, lim_x),
            round_f(prev_vy).clamp(-lim_y, lim_y),
        )
    } else {
        (0, 0)
    };
    let pred_shifted = pred_dx != 0 || pred_dy != 0;
    let p_ok = |bx: usize, by: usize, edx: i32, edy: i32| -> bool {
        let sx = (bx as i32) - edx;
        let sy = (by as i32) - edy;
        let margin = (2 * scale_px) as i32 + 1;
        let span = (8 * scale_px) as i32;
        sx >= margin
            && sy >= margin
            && sx + span + margin <= (frame_w as i32)
            && sy + span + margin <= (frame_h as i32)
    };
    let wind_lab_lock = a.wind.get_mode() != crate::wind_tunnel::WIND_MODE_OFF;
    let mut changed_tiles = 0u32;
    let mut uniform_tiles = 0u32;
    let mut anchored_2d_tiles = 0u32;
    let mut hard_cut_tiles = 0u32;
    let mut pan_votes = 0usize;
    let mut vx_votes = [0i32; 16];
    let mut vy_votes = [0i32; 16];
    let mut vote_tx = [0usize; 16];
    let mut vote_ty = [0usize; 16];
    let mut subtle_tiles = 0usize;
    let mut sub_tx = [0usize; 16];
    let mut sub_ty = [0usize; 16];
    let mut stat_tiles = 0u32;
    let mut stat_tx = [0usize; 16];
    let mut stat_ty = [0usize; 16];
    let mut pos_exp_tiles = 0u32;
    let mut neg_exp_tiles = 0u32;
    let mut pos_gain_tiles = 0u32;
    let mut neg_gain_tiles = 0u32;

    for &ty in &TILE_Y {
        for &tx in &TILE_X {
            let mut cur_sub = [0.0f32; 48];
            let (zero_sad, mc, mp, tex_mad, prev_tex_mad, g_l1, iso2d) = unsafe {
                cam_pyr_tile_4x6x8_simd128(
                    &a.pyr_luma,
                    &a.prev_pyr_luma,
                    PYR_W,
                    tx,
                    ty,
                    &mut cur_sub,
                )
            };
            if tex_mad < 0.0042 {
                uniform_tiles += 1;
            }
            if tex_mad >= 0.0085 && g_l1 > 0.16 && iso2d && zero_sad <= 0.008 {
                anchored_2d_tiles += 1;
            }

            if zero_sad <= 0.014 {
                stat_tx[stat_tiles as usize] = tx;
                stat_ty[stat_tiles as usize] = ty;
                stat_tiles += 1;
                let dm = mc - mp;
                if dm > 0.003 {
                    pos_exp_tiles += 1;
                } else if dm < -0.003 {
                    neg_exp_tiles += 1;
                }
                if tex_mad > 0.006 {
                    let dt = tex_mad - prev_tex_mad;
                    if dt > 0.0009 {
                        pos_gain_tiles += 1;
                    } else if dt < -0.0009 {
                        neg_gain_tiles += 1;
                    }
                }
            }

            if tex_mad > 0.0045
                && iso2d
                && g_l1 > 0.14
                && zero_sad > 0.0018 / s
                && zero_sad < 0.055
            {
                sub_tx[subtle_tiles] = tx;
                sub_ty[subtle_tiles] = ty;
                subtle_tiles += 1;
            }

            if zero_sad <= 0.016 {
                continue;
            }
            changed_tiles += 1;

            let want_pan = !wind_lab_lock && tex_mad > 0.006 && iso2d && g_l1 > 0.18;
            let want_cut = zero_sad > 0.055;
            if !want_pan && !want_cut {
                continue;
            }
            let sad_cap = if want_cut {
                if want_pan {
                    (0.56 * zero_sad).min(zero_sad - 0.006).max(0.0365)
                } else {
                    0.0365
                }
            } else {
                (0.56 * zero_sad).min(zero_sad - 0.006)
            };
            let use_3x3 = a.cam_pan_streak >= 1 && want_pan && !want_cut;
            let (best_sad, best_dx, best_dy) = if use_3x3 {
                eval_cam_coarse_3x3_simd128(
                    cur_sub.as_ptr(),
                    a.prev_pyr_luma.as_ptr(),
                    PYR_W,
                    PYR_H,
                    tx,
                    ty,
                    coarse_cx,
                    coarse_cy,
                    tex_mad,
                    zero_sad,
                    sad_cap,
                )
            } else {
                eval_cam_coarse_tile_simd128(
                    cur_sub.as_ptr(),
                    a.prev_pyr_luma.as_ptr(),
                    PYR_W,
                    PYR_H,
                    tx,
                    ty,
                    coarse_cx,
                    coarse_cy,
                    tex_mad,
                    zero_sad,
                    sad_cap,
                )
            };

            if want_pan
                && best_sad < 0.56 * zero_sad
                && (zero_sad - best_sad) > 0.006
            {
                vx_votes[pan_votes] = best_dx;
                vy_votes[pan_votes] = best_dy;
                vote_tx[pan_votes] = tx;
                vote_ty[pan_votes] = ty;
                pan_votes += 1;
            } else if zero_sad > 0.055 && best_sad > 0.036 {
                hard_cut_tiles += 1;
            }
        }
    }


    let mut best_cluster = 0usize;
    let mut best_anchor = 0usize;
    let mut sum_cx = 0.0f32;
    let mut sum_cy = 0.0f32;
    if changed_tiles >= 6 && pan_votes >= 5 {
        for i in 0..pan_votes {
            let mut cnt = 0usize;
            let mut sx = 0i32;
            let mut sy = 0i32;
            for j in 0..pan_votes {
                if (vx_votes[i] - vx_votes[j]).abs() <= 1 && (vy_votes[i] - vy_votes[j]).abs() <= 1 {
                    cnt += 1;
                    sx += vx_votes[j];
                    sy += vy_votes[j];
                }
            }
            if cnt > best_cluster {
                best_cluster = cnt;
                best_anchor = i;
                sum_cx = (sx as f32) / (cnt as f32);
                sum_cy = (sy as f32) / (cnt as f32);
            }
        }
    }

    if a.cam_pan_streak >= 1 && changed_tiles >= 6 && best_cluster < 5 {
        cam_coarse_3x3_fallback(
            &a.pyr_luma,
            &a.prev_pyr_luma,
            coarse_cx,
            coarse_cy,
            &mut pan_votes,
            &mut vx_votes,
            &mut vy_votes,
            &mut vote_tx,
            &mut vote_ty,
            &mut best_cluster,
            &mut best_anchor,
            &mut sum_cx,
            &mut sum_cy,
        );
    }

    a.prev_pyr_luma.copy_from_slice(&a.pyr_luma);

    let min_px = 12 * scale_px;
    let max_px = frame_w - 26 * scale_px;
    let min_py = 10 * scale_px;
    let max_py = frame_h - 24 * scale_px;
    let sub_step = 8 * scale_px;
    let phase_mask = scale_px - 1;

    // Store 8x64 patch buffers in `a.roi_rg`/`a.roi_by` (unused during ego-motion) to save 4 KiB of Wasm shadow stack.
    let cur_patches: &mut [[f32; 64]; 8] = &mut *(a.roi_rg.as_mut_ptr() as *mut [[f32; 64]; 8]);
    for p in cur_patches.iter_mut() {
        p.fill(0.0);
    }
    let mut cur_mad = [0.0f32; 8];
    let mut patch_px = [0usize; 8];
    let mut patch_py = [0usize; 8];
    let mut n_patches = 0usize;
    let mut est_dx = 0i32;
    let mut est_dy = 0i32;
    let mut have_pan_candidate = false;
    let mut have_int_consensus = false;
    let mut subpixel_zero_probe = false;
    let mut sparse_used = false;
    let mut coast_only_pan = false;
    let swarm_pan_guard = a.swarm_mode_cached != 0 && a.micro_swarm.swarm_active != 0;

    if !wind_lab_lock && best_cluster >= 5 {
        est_dx = round_f(sum_cx * (2.0 * s));
        est_dy = round_f(sum_cy * (2.0 * s));
        let max_ref_patches = if swarm_pan_guard && best_cluster >= 6 && a.cam_pan_streak >= 2 {
            5
        } else {
            8
        };
        for j in 0..pan_votes {
            if n_patches >= max_ref_patches {
                break;
            }
            if (vx_votes[best_anchor] - vx_votes[j]).abs() > 1
                || (vy_votes[best_anchor] - vy_votes[j]).abs() > 1
            {
                continue;
            }
            let (best_bx, best_by, best_tex2d) = pick_textured_subpatch(
                &a.luma,
                frame_w,
                scale_px,
                sub_step,
                min_px,
                max_px,
                min_py,
                max_py,
                vote_tx[j],
                vote_ty[j],
            );
            if best_tex2d >= 0.08 * (scale_px as f32) && p_ok(best_bx, best_by, est_dx, est_dy) {
                let (patch, mad_c) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, best_bx, best_by);
                cur_patches[n_patches] = patch;
                cur_mad[n_patches] = mad_c;
                patch_px[n_patches] = best_bx;
                patch_py[n_patches] = best_by;
                n_patches += 1;
            }
        }
        have_pan_candidate = n_patches >= 2;
        have_int_consensus = have_pan_candidate;
    } else if !wind_lab_lock
        && hard_cut_tiles < 8
        && subtle_tiles >= 5
        && (!swarm_pan_guard || uniform_tiles < 9)
        && !(swarm_pan_guard && a.cam_pan_streak == 0 && stat_tiles >= 6)
    {
        // Coarse 2*s pyramid cells quantize away <= 1.5*s px/frame camera motion, so fall back
        // to full-resolution integer shifts buffered in `a.eca_res`.
        let prev_rgba = &a.rgba_ring[slot_prev];
        let f_r = 6 * (scale_px as i32);
        let win_dim = 8 * scale_px + ((2 * f_r) as usize);
        let mut fine_votes = 0usize;
        let mut f_dx = [0i32; 16];
        let mut f_dy = [0i32; 16];
        let mut f_bx = [0usize; 16];
        let mut f_by = [0usize; 16];
        let mut sub_cand_cnt = 0usize;
        let mut sub_cand_bx = [0usize; 16];
        let mut sub_cand_by = [0usize; 16];
        let min_t2d_sub = (if a.cam_pan_streak == 0 { 0.14 } else { 0.06 }) * (scale_px as f32);

        const SUB_ORDER: [usize; 16] = [0, 5, 10, 15, 2, 7, 8, 13, 1, 6, 11, 12, 3, 4, 9, 14];
        let mut rem_subtle = subtle_tiles;
        for &t in &SUB_ORDER {
            if t >= subtle_tiles {
                continue;
            }
            rem_subtle = rem_subtle.saturating_sub(1);
            let (best_bx, best_by, best_tex2d) = pick_textured_subpatch(
                &a.luma,
                frame_w,
                scale_px,
                sub_step,
                min_px,
                max_px,
                min_py,
                max_py,
                sub_tx[t],
                sub_ty[t],
            );
            if best_tex2d < min_t2d_sub {
                continue;
            }
            if pred_shifted && !p_ok(best_bx, best_by, pred_dx, pred_dy) {
                continue;
            }

            let (cur_q, mad_cq) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, best_bx, best_by);
            let z_sad = eval_single_patch_raw_sad(
                &cur_q,
                mad_cq,
                best_bx,
                best_by,
                prev_rgba,
                frame_w,
                scale_px,
                phase_mask,
                0,
                0,
            ) * (1.0 / 64.0);
            if z_sad <= 0.0020 / s {
                continue;
            }
            sub_cand_bx[sub_cand_cnt] = best_bx;
            sub_cand_by[sub_cand_cnt] = best_by;
            sub_cand_cnt += 1;
            if fine_votes + 1 + rem_subtle < 5 {
                continue;
            }

            const PREV_WIN_OFF: usize = 4608;
            let _ = win_dim;
            let f_r_t = if !pred_shifted && pan_votes <= 1 && z_sad < 0.012 {
                (2 * (scale_px as i32)).min(f_r)
            } else if (!pred_shifted && pan_votes <= 1)
                || (!pred_shifted && pan_votes < 4 && z_sad < 0.025)
                || (pred_shifted && a.cam_pan_streak >= 2 && z_sad < 0.015)
            {
                (2 * (scale_px as i32) + if pred_shifted { 0 } else { scale_px as i32 }).min(f_r)
            } else if pred_shifted && a.cam_pan_streak >= 2 {
                (3 * (scale_px as i32)).min(f_r)
            } else {
                f_r
            };
            let win_dim_t = 8 * scale_px + ((2 * f_r_t) as usize);
            let wx0 = (best_bx as i32) - pred_dx - f_r_t;
            let wy0 = (best_by as i32) - pred_dy - f_r_t;
            if z_sad <= 0.0012 {
                continue;
            }
            let (b_sad, r_dx, r_dy) = if scale_px == 2 && f_r_t >= 4 && win_dim_t >= 16 {
                let p_dims = unsafe {
                    unpack_rgba_luma_win_p2_simd128(
                        prev_rgba,
                        frame_w,
                        frame_h,
                        wx0,
                        wy0,
                        win_dim_t,
                        &mut a.eca_res[PREV_WIN_OFF..],
                    )
                };
                find_best_win_shift_simd128(
                    &cur_q,
                    mad_cq,
                    &a.eca_res[PREV_WIN_OFF..],
                    &mut [],
                    win_dim_t,
                    f_r_t,
                    scale_px,
                    z_sad,
                    a.last_pan_dx - pred_dx,
                    a.last_pan_dy - pred_dy,
                    pred_shifted,
                    p_dims,
                )
            } else {
                let (prev_win, phase_scratch) =
                    a.eca_res[PREV_WIN_OFF..].split_at_mut(win_dim_t * win_dim_t);
                unsafe {
                    unpack_rgba_luma_window_simd128(
                        prev_rgba, frame_w, frame_h, wx0, wy0, win_dim_t, prev_win,
                    );
                }
                find_best_win_shift_simd128(
                    &cur_q,
                    mad_cq,
                    prev_win,
                    phase_scratch,
                    win_dim_t,
                    f_r_t,
                    scale_px,
                    z_sad,
                    a.last_pan_dx - pred_dx,
                    a.last_pan_dy - pred_dy,
                    pred_shifted,
                    (0, 0),
                )
            };
            let b_dx = pred_dx + r_dx;
            let b_dy = pred_dy + r_dy;
            if (b_dx != 0 || b_dy != 0) && b_sad < 0.78 * z_sad && (z_sad - b_sad) > 0.0012 {
                f_dx[fine_votes] = b_dx;
                f_dy[fine_votes] = b_dy;
                f_bx[fine_votes] = best_bx;
                f_by[fine_votes] = best_by;
                fine_votes += 1;
                if fine_votes >= 6 {
                    let mut agree_cnt = 0usize;
                    for j in 0..fine_votes {
                        if (f_dx[j] - b_dx).abs() <= 1 && (f_dy[j] - b_dy).abs() <= 1 {
                            agree_cnt += 1;
                        }
                    }
                    if agree_cnt >= 6 {
                        break;
                    }
                }
            }
        }

        let mut fine_cluster = 0usize;
        let mut fine_anchor = 0usize;
        let mut fine_sum_x = 0.0f32;
        let mut fine_sum_y = 0.0f32;
        if fine_votes >= 5 {
            for i in 0..fine_votes {
                let mut cnt = 0usize;
                let mut sx = 0i32;
                let mut sy = 0i32;
                for j in 0..fine_votes {
                    if (f_dx[i] - f_dx[j]).abs() <= 1 && (f_dy[i] - f_dy[j]).abs() <= 1 {
                        cnt += 1;
                        sx += f_dx[j];
                        sy += f_dy[j];
                    }
                }
                if cnt > fine_cluster {
                    fine_cluster = cnt;
                    fine_anchor = i;
                    fine_sum_x = (sx as f32) / (cnt as f32);
                    fine_sum_y = (sy as f32) / (cnt as f32);
                }
            }
        }

        if fine_cluster >= 5 {
            est_dx = round_f(fine_sum_x);
            est_dy = round_f(fine_sum_y);
            for j in 0..fine_votes {
                if n_patches >= 8 {
                    break;
                }
                if (f_dx[fine_anchor] - f_dx[j]).abs() > 1 || (f_dy[fine_anchor] - f_dy[j]).abs() > 1 {
                    continue;
                }
                let bx = f_bx[j];
                let by = f_by[j];
                if p_ok(bx, by, est_dx, est_dy) {
                    let (patch, mad_c) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, bx, by);
                    cur_patches[n_patches] = patch;
                    cur_mad[n_patches] = mad_c;
                    patch_px[n_patches] = bx;
                    patch_py[n_patches] = by;
                    n_patches += 1;
                }
            }
            have_pan_candidate = n_patches >= 4;
            have_int_consensus = have_pan_candidate;
            subpixel_zero_probe = n_patches >= 5;
        } else if subtle_tiles >= 5
            && sub_cand_cnt >= 4
            && !(a.cam_pan_streak >= 1 && prev_v2 >= (1.5 * s) * (1.5 * s))
        {
            est_dx = 0;
            est_dy = 0;
            let take = sub_cand_cnt.min(8);
            for k in 0..take {
                let ci = (k * sub_cand_cnt) / take;
                let bx = sub_cand_bx[ci];
                let by = sub_cand_by[ci];
                let (patch, mad_c) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, bx, by);
                cur_patches[n_patches] = patch;
                cur_mad[n_patches] = mad_c;
                patch_px[n_patches] = bx;
                patch_py[n_patches] = by;
                n_patches += 1;
            }
            have_pan_candidate = true;
            subpixel_zero_probe = true;
        }
    }
    let mut sp_stat_landmarks = 0usize;
    let fast_pan_active = scale_px > 1 && a.cam_pan_streak >= 1 && prev_v2 >= (7.5 * s) * (7.5 * s);
    let mut sp_cand_ran = false;
    let mut sp_cand_cnt = 0usize;
    let mut sp_cand_bx = [0usize; 16];
    let mut sp_cand_by = [0usize; 16];
    if !wind_lab_lock && !swarm_pan_guard && !have_pan_candidate && hard_cut_tiles < 8 && uniform_tiles >= 9 && anchored_2d_tiles == 0 {
        const PREV_WIN_OFF: usize = 4608;
        let prev_rgba = &a.rgba_ring[slot_prev];
        let f_r = 6 * (scale_px as i32);
        let win_dim = 8 * scale_px + ((2 * f_r) as usize);
        sp_cand_cnt = find_top_2d_patches(
            &a.luma,
            &a.targets,
            &a.peaks,
            a.num_peaks,
            frame_w,
            frame_h,
            scale_px,
            0.075 * (scale_px as f32),
            0.40,
            &mut sp_cand_bx,
            &mut sp_cand_by,
        );
        sp_cand_ran = true;
        let cand_cnt = sp_cand_cnt;
        let cand_bx = &sp_cand_bx;
        let cand_by = &sp_cand_by;
        let mut sp_votes = 0usize;
        let mut sp_dx = [0i32; 16];
        let mut sp_dy = [0i32; 16];
        let mut sp_bx = [0usize; 16];
        let mut sp_by = [0usize; 16];

        for ci in 0..cand_cnt {
            let best_bx = cand_bx[ci];
            let best_by = cand_by[ci];
            let (cur_q, mad_cq) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, best_bx, best_by);
            if mad_cq < 0.40 {
                continue;
            }
            let z_sad = eval_single_patch_raw_sad(
                &cur_q,
                mad_cq,
                best_bx,
                best_by,
                prev_rgba,
                frame_w,
                scale_px,
                phase_mask,
                0,
                0,
            ) * (1.0 / 64.0);
            if z_sad <= 0.0040 / s {
                sp_stat_landmarks += 1;
                break;
            }
            if pred_shifted && !p_ok(best_bx, best_by, pred_dx, pred_dy) {
                continue;
            }
            if z_sad <= 0.0025 {
                continue;
            }
            let wx0 = (best_bx as i32) - pred_dx - f_r;
            let wy0 = (best_by as i32) - pred_dy - f_r;
            let (b_sad, r_dx, r_dy) = if scale_px == 2 && f_r >= 4 && win_dim >= 16 {
                let p_dims = unsafe {
                    unpack_rgba_luma_win_p2_simd128(
                        prev_rgba,
                        frame_w,
                        frame_h,
                        wx0,
                        wy0,
                        win_dim,
                        &mut a.eca_res[PREV_WIN_OFF..],
                    )
                };
                find_best_win_shift_simd128(
                    &cur_q,
                    mad_cq,
                    &a.eca_res[PREV_WIN_OFF..],
                    &mut [],
                    win_dim,
                    f_r,
                    scale_px,
                    z_sad,
                    a.last_pan_dx - pred_dx,
                    a.last_pan_dy - pred_dy,
                    pred_shifted,
                    p_dims,
                )
            } else {
                let (prev_win, phase_scratch) = a.eca_res[PREV_WIN_OFF..].split_at_mut(win_dim * win_dim);
                unsafe {
                    unpack_rgba_luma_window_simd128(prev_rgba, frame_w, frame_h, wx0, wy0, win_dim, prev_win);
                }
                find_best_win_shift_simd128(
                    &cur_q,
                    mad_cq,
                    prev_win,
                    phase_scratch,
                    win_dim,
                    f_r,
                    scale_px,
                    z_sad,
                    a.last_pan_dx - pred_dx,
                    a.last_pan_dy - pred_dy,
                    pred_shifted,
                    (0, 0),
                )
            };
            let b_dx = pred_dx + r_dx;
            let b_dy = pred_dy + r_dy;
            if (b_dx != 0 || b_dy != 0) && b_sad < 0.65 * z_sad && (z_sad - b_sad) > 0.0025 {
                sp_dx[sp_votes] = b_dx;
                sp_dy[sp_votes] = b_dy;
                sp_bx[sp_votes] = best_bx;
                sp_by[sp_votes] = best_by;
                sp_votes += 1;
            }
        }

        let mut best_sp = 0usize;
        let mut sp_anc = 0usize;
        let mut sp_sx = 0.0f32;
        let mut sp_sy = 0.0f32;
        let mut max_dist2 = 0.0f32;
        let min_dist2 = (40.0 * s) * (40.0 * s);
        for i in 0..sp_votes {
            let mut cnt = 0usize;
            let mut sx = 0i32;
            let mut sy = 0i32;
            let mut md2 = 0.0f32;
            for j in 0..sp_votes {
                if (sp_dx[i] - sp_dx[j]).abs() <= 1 && (sp_dy[i] - sp_dy[j]).abs() <= 1 {
                    cnt += 1;
                    sx += sp_dx[j];
                    sy += sp_dy[j];
                    let ddx = (sp_bx[i] as f32) - (sp_bx[j] as f32);
                    let ddy = (sp_by[i] as f32) - (sp_by[j] as f32);
                    let d2 = ddx * ddx + ddy * ddy;
                    if d2 > md2 {
                        md2 = d2;
                    }
                }
            }
            if cnt > best_sp && md2 >= min_dist2 {
                best_sp = cnt;
                sp_anc = i;
                sp_sx = (sx as f32) / (cnt as f32);
                sp_sy = (sy as f32) / (cnt as f32);
                max_dist2 = md2;
            }
        }
        let min_sp_votes = if scale_px > 1 || a.lab_mode == 1 { 3usize } else { 2usize };
        if sp_stat_landmarks == 0 && best_sp >= min_sp_votes && max_dist2 >= min_dist2 {
            est_dx = round_f(sp_sx);
            est_dy = round_f(sp_sy);
            for j in 0..sp_votes {
                if n_patches >= 8 {
                    break;
                }
                if (sp_dx[sp_anc] - sp_dx[j]).abs() > 1 || (sp_dy[sp_anc] - sp_dy[j]).abs() > 1 {
                    continue;
                }
                let bx = sp_bx[j];
                let by = sp_by[j];
                let (patch, mad_c) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, bx, by);
                cur_patches[n_patches] = patch;
                cur_mad[n_patches] = mad_c;
                patch_px[n_patches] = bx;
                patch_py[n_patches] = by;
                n_patches += 1;
            }
            have_pan_candidate = n_patches >= 2;
            have_int_consensus = have_pan_candidate;
            sparse_used = have_pan_candidate;
        }
    }

    let mut from_pan_fallback = false;
    if !wind_lab_lock
        && !have_pan_candidate
        && a.cam_pan_streak >= 1
        && anchored_2d_tiles == 0
        && (!swarm_pan_guard || uniform_tiles < 9)
        && (prev_vx * prev_vx + prev_vy * prev_vy) >= 0.06
    {
        est_dx = round_f(prev_vx);
        est_dy = round_f(prev_vy);
        for &ty in &[10usize, 30, 50, 68] {
            for &tx in &[12usize, 52, 92, 132] {
                if n_patches >= 8 {
                    break;
                }
                let (best_bx, best_by, best_tex2d) = pick_textured_subpatch(
                    &a.luma,
                    frame_w,
                    scale_px,
                    sub_step,
                    min_px,
                    max_px,
                    min_py,
                    max_py,
                    tx,
                    ty,
                );
                if best_tex2d >= 0.08 * (scale_px as f32) && p_ok(best_bx, best_by, est_dx, est_dy) {
                    let (patch, mad_c) =
                        sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, best_bx, best_by);
                    cur_patches[n_patches] = patch;
                    cur_mad[n_patches] = mad_c;
                    patch_px[n_patches] = best_bx;
                    patch_py[n_patches] = best_by;
                    n_patches += 1;
                }
            }
        }
        if n_patches < 5
            && (if swarm_pan_guard {
                fast_pan_active && (uniform_tiles + subtle_tiles as u32) >= 9
            } else {
                uniform_tiles >= 9 || (fast_pan_active && (uniform_tiles + subtle_tiles as u32) >= 9)
            })
            && hard_cut_tiles < 8
        {
            n_patches = 0;
            let mut reused = false;
            if !swarm_pan_guard && sp_cand_ran && sp_cand_cnt >= 8 {
                for ci in 0..sp_cand_cnt {
                    if n_patches >= 8 {
                        break;
                    }
                    let bx = sp_cand_bx[ci];
                    let by = sp_cand_by[ci];
                    if p_ok(bx, by, est_dx, est_dy) {
                        let (patch, mad_c) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, bx, by);
                        cur_patches[n_patches] = patch;
                        cur_mad[n_patches] = mad_c;
                        patch_px[n_patches] = bx;
                        patch_py[n_patches] = by;
                        n_patches += 1;
                    }
                }
                if n_patches >= 8 {
                    reused = true;
                } else {
                    n_patches = 0;
                }
            }
            if !reused {
                let mut cand_bx = [0usize; 16];
                let mut cand_by = [0usize; 16];
                let cand_cnt = find_top_2d_patches(
                    &a.luma,
                    &a.targets,
                    &a.peaks,
                    a.num_peaks,
                    frame_w,
                    frame_h,
                    scale_px,
                    0.065 * (scale_px as f32),
                    0.35,
                    &mut cand_bx,
                    &mut cand_by,
                );
                for ci in 0..cand_cnt.min(8) {
                    let bx = cand_bx[ci];
                    let by = cand_by[ci];
                    if p_ok(bx, by, est_dx, est_dy) {
                        let (patch, mad_c) = sample_cur_luma_patch(&a.luma, frame_w, scale_px, phase_mask, bx, by);
                        cur_patches[n_patches] = patch;
                        cur_mad[n_patches] = mad_c;
                        patch_px[n_patches] = bx;
                        patch_py[n_patches] = by;
                        n_patches += 1;
                    }
                }
            }
        }
        if n_patches >= 5 {
            have_pan_candidate = true;
            have_int_consensus = true;
            subpixel_zero_probe = true;
            from_pan_fallback = true;
        } else if !swarm_pan_guard && uniform_tiles >= 9 && a.cam_pan_streak >= 2 && n_patches >= 1 {
            have_pan_candidate = true;
            have_int_consensus = true;
            sparse_used = true;
        } else if !swarm_pan_guard
            && uniform_tiles >= 9
            && a.cam_pan_streak >= 2
            && anchored_2d_tiles == 0
            && hard_cut_tiles < 8
            && sp_stat_landmarks == 0
            && (prev_vx * prev_vx + prev_vy * prev_vy) >= 0.20
        {
            have_pan_candidate = true;
            coast_only_pan = true;
        }
    } else if have_pan_candidate && a.cam_pan_streak >= 1 {
        let dvx = (est_dx as f32) - prev_vx;
        let dvy = (est_dy as f32) - prev_vy;
        let max_dv = if prev_v2 >= (2.0 * s) * (2.0 * s) {
            4.5 * s
        } else {
            2.5 * s
        };
        if dvx * dvx + dvy * dvy > max_dv * max_dv {
            est_dx = round_f(prev_vx);
            est_dy = round_f(prev_vy);
        }
    }

    if swarm_pan_guard && have_pan_candidate && swarm_pan_flat_canvas_reject(a, frame_w, frame_h, scale_px) {
        have_pan_candidate = false;
        have_int_consensus = false;
        subpixel_zero_probe = false;
    }

    if have_pan_candidate {
        let ref_r = (2 * scale_px) as i32;
        let mut pan_dx = est_dx;
        let mut pan_dy = est_dy;
        let mut sub_x = 0.0f32;
        let mut sub_y = 0.0f32;
        let mut valid_pan = pan_dx != 0 || pan_dy != 0;
        let mut skip_a = usize::MAX;
        let mut skip_b = usize::MAX;
        let mut skip_c = usize::MAX;

        if coast_only_pan {
            let dec_vx = 0.94 * prev_vx;
            let dec_vy = 0.94 * prev_vy;
            pan_dx = round_f(dec_vx);
            pan_dy = round_f(dec_vy);
            sub_x = dec_vx - (pan_dx as f32);
            sub_y = dec_vy - (pan_dy as f32);
            valid_pan = true;
        } else if n_patches >= 2 || (sparse_used && a.cam_pan_streak >= 2 && n_patches == 1) {
            let prev_rgba = &a.rgba_ring[slot_prev];
            let zero_raw_sad = eval_ref_patches_trimmed_sad(
                &cur_patches,
                &cur_mad,
                &patch_px,
                &patch_py,
                n_patches,
                prev_rgba,
                frame_w,
                scale_px,
                phase_mask,
                0,
                0,
                1e9f32,
            );
            let mut best_ref_sad = 1e9f32;
            let mut best_raw_sad = 1e9f32;
            if have_int_consensus {
                const WIN24_STRIDE: usize = 672;
                let cur_eo: &mut [[f32; 64]; 8] = &mut *(a.roi_by.as_mut_ptr() as *mut [[f32; 64]; 8]);
                for p in cur_eo.iter_mut() {
                    p.fill(0.0);
                }
                let mut all_win24_ok =
                    scale_px == 2 && ref_r == 4 && a.eca_res.len() >= n_patches * WIN24_STRIDE;
                if all_win24_ok {
                    for pid in 0..n_patches {
                        let bx_c = (patch_px[pid] as i32) - est_dx;
                        let by_c = (patch_py[pid] as i32) - est_dy;
                        if bx_c < 4
                            || by_c < 4
                            || bx_c + 22 > (frame_w as i32)
                            || by_c + 22 > (frame_h as i32)
                        {
                            all_win24_ok = false;
                            break;
                        }
                    }
                }
                if all_win24_ok {
                    for pid in 0..n_patches {
                        let wx0 = (patch_px[pid] as i32) - est_dx - 4;
                        let wy0 = (patch_py[pid] as i32) - est_dy - 4;
                        unsafe {
                            reorder_patch64_eo_simd128(&cur_patches[pid], &mut cur_eo[pid]);
                            unpack_rgba_luma_win_p2_simd128(
                                prev_rgba,
                                frame_w,
                                frame_h,
                                wx0,
                                wy0,
                                24,
                                &mut a.eca_res[pid * WIN24_STRIDE..(pid + 1) * WIN24_STRIDE],
                            );
                        }
                    }
                }
                let eca_wins = &a.eca_res[..];
                let mut eval_ref_pt = |cdx: i32, cdy: i32, limit: f32| -> f32 {
                    let eff_limit = if limit > 0.0 { limit } else { best_ref_sad };
                    let reg = if a.cam_pan_streak >= 1 {
                        let dxp = (cdx as f32) - prev_vx;
                        let dyp = (cdy as f32) - prev_vy;
                        2e-3 * (dxp * dxp + dyp * dyp)
                    } else {
                        let dd2 = ((cdx - est_dx) * (cdx - est_dx) + (cdy - est_dy) * (cdy - est_dy)) as f32;
                        1e-4 * dd2
                    };
                    let raw_cutoff = if eff_limit < 1e8f32 {
                        eff_limit - reg + 1e-5f32
                    } else {
                        1e9f32
                    };
                    if raw_cutoff <= 0.0 {
                        return eff_limit + 1.0f32;
                    }
                    let raw_sad = if all_win24_ok {
                        let ox = (4 - (cdx - est_dx)) as usize;
                        let oy = (4 - (cdy - est_dy)) as usize;
                        eval_trimmed_sad_with(n_patches, raw_cutoff, |pid| unsafe {
                            eval_win_patch_sad_p2_packed_simd128(
                                &cur_eo[pid],
                                cur_mad[pid],
                                &eca_wins[pid * WIN24_STRIDE..],
                                7,
                                84,
                                ox,
                                oy,
                                1.0e9f32,
                            ) * 64.0
                        })
                    } else {
                        eval_ref_patches_trimmed_sad(
                            &cur_patches,
                            &cur_mad,
                            &patch_px,
                            &patch_py,
                            n_patches,
                            prev_rgba,
                            frame_w,
                            scale_px,
                            phase_mask,
                            cdx,
                            cdy,
                            raw_cutoff,
                        )
                    };
                    if raw_sad >= raw_cutoff {
                        return eff_limit + 1.0f32;
                    }
                    let total_sad = raw_sad + reg;
                    if total_sad < best_ref_sad {
                        best_ref_sad = total_sad;
                        best_raw_sad = raw_sad;
                        pan_dx = cdx;
                        pan_dy = cdy;
                    }
                    total_sad
                };

                if scale_px > 1 && ref_r <= 8 {
                    const MAX_REF: usize = 17;
                    let mut seen = [false; MAX_REF * MAX_REF];
                    let mut top_sad = [1.0e9f32; 2];
                    let mut top_dx = [est_dx; 2];
                    let mut top_dy = [est_dy; 2];
                    let mut oy = -ref_r;
                    while oy <= ref_r {
                        let mut ox = -ref_r;
                        while ox <= ref_r {
                            let gx = (ox + ref_r) as usize;
                            let gy = (oy + ref_r) as usize;
                            seen[gy * MAX_REF + gx] = true;
                            let total_sad = eval_ref_pt(est_dx + ox, est_dy + oy, top_sad[1]);
                            if total_sad < top_sad[0] {
                                top_sad[1] = top_sad[0];
                                top_dx[1] = top_dx[0];
                                top_dy[1] = top_dy[0];
                                top_sad[0] = total_sad;
                                top_dx[0] = ox;
                                top_dy[0] = oy;
                            } else if total_sad < top_sad[1] {
                                top_sad[1] = total_sad;
                                top_dx[1] = ox;
                                top_dy[1] = oy;
                            }
                            ox += 2;
                        }
                        oy += 2;
                    }
                    let mut ref_best = top_sad[0];
                    for ti in 0..2usize {
                        if top_sad[ti] >= 1.0e8 {
                            continue;
                        }
                        let cx = top_dx[ti];
                        let cy = top_dy[ti];
                        for fy in (cy - 1).max(-ref_r)..=(cy + 1).min(ref_r) {
                            for fx in (cx - 1).max(-ref_r)..=(cx + 1).min(ref_r) {
                                let gx = (fx + ref_r) as usize;
                                let gy = (fy + ref_r) as usize;
                                if !seen[gy * MAX_REF + gx] {
                                    seen[gy * MAX_REF + gx] = true;
                                    let s_pt = eval_ref_pt(est_dx + fx, est_dy + fy, -1.0f32);
                                    if s_pt < ref_best {
                                        ref_best = s_pt;
                                    }
                                }
                            }
                        }
                    }
                    if ref_best >= 0.68 * zero_raw_sad && zero_raw_sad > 0.006 {
                        for cdy in -ref_r..=ref_r {
                            let gy = (cdy + ref_r) as usize;
                            for cdx in -ref_r..=ref_r {
                                let gx = (cdx + ref_r) as usize;
                                if !seen[gy * MAX_REF + gx] {
                                    let _ = eval_ref_pt(est_dx + cdx, est_dy + cdy, -1.0f32);
                                }
                            }
                        }
                    }
                } else {
                    for cdy in (est_dy - ref_r)..=(est_dy + ref_r) {
                        for cdx in (est_dx - ref_r)..=(est_dx + ref_r) {
                            let _ = eval_ref_pt(cdx, cdy, -1.0f32);
                        }
                    }
                }
            }

            let int_ratio_lim = if sparse_used { 0.68 } else { 0.80 };
            let mut int_pan_ok =
                have_int_consensus && (pan_dx != 0 || pan_dy != 0) && best_raw_sad < int_ratio_lim * zero_raw_sad;
            if from_pan_fallback
                && !int_pan_ok
                && n_patches >= 4
                && scale_px > 1
                && prev_v2 >= (7.5 * s) * (7.5 * s)
            {
                let mut p_zero = [0.0f32; 8];
                for pid in 0..n_patches {
                    p_zero[pid] = eval_single_patch_raw_sad(
                        &cur_patches[pid],
                        cur_mad[pid],
                        patch_px[pid],
                        patch_py[pid],
                        prev_rgba,
                        frame_w,
                        scale_px,
                        phase_mask,
                        0,
                        0,
                    );
                }
                let eval_wide = |cdx: i32, cdy: i32| -> (f32, usize) {
                    let mut ss = [0.0f32; 8];
                    let mut zs = [0.0f32; 8];
                    let mut cnt = 0usize;
                    for pid in 0..n_patches {
                        if p_ok(patch_px[pid], patch_py[pid], cdx, cdy) {
                            ss[cnt] = eval_single_patch_raw_sad(
                                &cur_patches[pid],
                                cur_mad[pid],
                                patch_px[pid],
                                patch_py[pid],
                                prev_rgba,
                                frame_w,
                                scale_px,
                                phase_mask,
                                cdx,
                                cdy,
                            );
                            zs[cnt] = p_zero[pid];
                            cnt += 1;
                        }
                    }
                    if cnt < 4 {
                        return (1e9f32, cnt);
                    }
                    let trim = if cnt >= 6 { 2usize } else { 1usize };
                    for _ in 0..trim {
                        let mut mx = -1.0f32;
                        let mut mi = 0usize;
                        for k in 0..cnt {
                            if ss[k] > mx {
                                mx = ss[k];
                                mi = k;
                            }
                        }
                        ss[mi] = -1.0;
                    }
                    let mut sum_s = 0.0f32;
                    let mut sum_z = 0.0f32;
                    for k in 0..cnt {
                        if ss[k] >= 0.0 {
                            sum_s += ss[k];
                            sum_z += zs[k];
                        }
                    }
                    if sum_z <= 1e-4 {
                        return (1e9f32, cnt);
                    }
                    (sum_s / sum_z, cnt)
                };
                let lim_x = ((frame_w as i32) / 3).max(12);
                let lim_y = ((frame_h as i32) / 3).max(12);
                let sx_step = (2 * scale_px) as i32;
                let mut best_wscore = 1e9f32;
                let mut best_wratio = 1e9f32;
                let mut best_wx = 0i32;
                let mut best_wy = 0i32;
                for &alpha in &[0.25f32, 0.375, 0.50, 0.625, 0.75, 1.20, 1.35, 1.50] {
                    let cy0 = round_f(prev_vy * alpha).clamp(-lim_y, lim_y);
                    let cx0 = round_f(prev_vx * alpha);
                    for &ox in &[-sx_step, 0i32, sx_step] {
                        let cdx = (cx0 + ox).clamp(-lim_x, lim_x);
                        if cdx == 0 && cy0 == 0 {
                            continue;
                        }
                        let (rat, cnt) = eval_wide(cdx, cy0);
                        if cnt >= 4 && rat < 1e8 {
                            let dxp = (cdx as f32) - prev_vx;
                            let dyp = (cy0 as f32) - prev_vy;
                            let sc = rat + 0.015 * ((dxp * dxp + dyp * dyp) / prev_v2.max(1.0));
                            if sc < best_wscore {
                                best_wscore = sc;
                                best_wratio = rat;
                                best_wx = cdx;
                                best_wy = cy0;
                            }
                        }
                    }
                }
                if best_wratio < 0.90 {
                    let c0x = best_wx;
                    let c0y = best_wy;
                    let rad = (2 * scale_px) as i32;
                    let mut oy = -rad;
                    while oy <= rad {
                        let mut ox = -rad;
                        while ox <= rad {
                            let cdx = (c0x + ox).clamp(-lim_x, lim_x);
                            let cdy = (c0y + oy).clamp(-lim_y, lim_y);
                            if cdx != 0 || cdy != 0 {
                                let (rat, cnt) = eval_wide(cdx, cdy);
                                if cnt >= 4 && rat < 1e8 {
                                    let dxp = (cdx as f32) - prev_vx;
                                    let dyp = (cdy as f32) - prev_vy;
                                    let sc = rat + 0.015 * ((dxp * dxp + dyp * dyp) / prev_v2.max(1.0));
                                    if sc < best_wscore {
                                        best_wscore = sc;
                                        best_wratio = rat;
                                        best_wx = cdx;
                                        best_wy = cdy;
                                    }
                                }
                            }
                            ox += 2;
                        }
                        oy += 2;
                    }
                    let f0x = best_wx;
                    let f0y = best_wy;
                    for oy in -1..=1i32 {
                        for ox in -1..=1i32 {
                            let cdx = (f0x + ox).clamp(-lim_x, lim_x);
                            let cdy = (f0y + oy).clamp(-lim_y, lim_y);
                            if cdx != 0 || cdy != 0 {
                                let (rat, cnt) = eval_wide(cdx, cdy);
                                if cnt >= 4 && rat < 1e8 {
                                    let dxp = (cdx as f32) - prev_vx;
                                    let dyp = (cdy as f32) - prev_vy;
                                    let sc = rat + 0.015 * ((dxp * dxp + dyp * dyp) / prev_v2.max(1.0));
                                    if sc < best_wscore {
                                        best_wscore = sc;
                                        best_wratio = rat;
                                        best_wx = cdx;
                                        best_wy = cdy;
                                    }
                                }
                            }
                        }
                    }
                }
                if (best_wx != 0 || best_wy != 0) && best_wratio < int_ratio_lim.min(0.75) {
                    let mut kept = 0usize;
                    for pid in 0..n_patches {
                        if p_ok(patch_px[pid], patch_py[pid], best_wx, best_wy) {
                            if kept != pid {
                                cur_patches[kept] = cur_patches[pid];
                                cur_mad[kept] = cur_mad[pid];
                                patch_px[kept] = patch_px[pid];
                                patch_py[kept] = patch_py[pid];
                            }
                            kept += 1;
                        }
                    }
                    if kept >= 4 {
                        n_patches = kept;
                        pan_dx = best_wx;
                        pan_dy = best_wy;
                        int_pan_ok = true;
                    }
                }
            }
            if !int_pan_ok && !(subpixel_zero_probe && n_patches >= 5) {
                valid_pan = false;
            } else {
                if !int_pan_ok {
                    pan_dx = 0;
                    pan_dy = 0;
                    valid_pan = false;
                } else {
                    valid_pan = true;
                }
                let mut p_s0 = [0.0f32; 8];
                for pid in 0..n_patches {
                    p_s0[pid] = eval_single_patch_raw_sad(
                        &cur_patches[pid],
                        cur_mad[pid],
                        patch_px[pid],
                        patch_py[pid],
                        prev_rgba,
                        frame_w,
                        scale_px,
                        phase_mask,
                        pan_dx,
                        pan_dy,
                    );
                }
                let mut min_ps0 = 1e9f32;
                for pid in 0..n_patches {
                    if p_s0[pid] < min_ps0 {
                        min_ps0 = p_s0[pid];
                    }
                }
                let outlier_gate = 2.0 * min_ps0 + 0.12;
                if n_patches >= 4 {
                    let mut m1 = -1.0f32;
                    for pid in 0..n_patches {
                        if (int_pan_ok || p_s0[pid] > outlier_gate) && p_s0[pid] > m1 {
                            m1 = p_s0[pid];
                            skip_a = pid;
                        }
                    }
                }
                if n_patches >= 5 {
                    let mut m2 = -1.0f32;
                    for pid in 0..n_patches {
                        if pid != skip_a && (int_pan_ok || p_s0[pid] > outlier_gate) && p_s0[pid] > m2 {
                            m2 = p_s0[pid];
                            skip_b = pid;
                        }
                    }
                }
                if n_patches >= 6 {
                    let mut m3 = -1.0f32;
                    for pid in 0..n_patches {
                        if pid != skip_a && pid != skip_b && (int_pan_ok || p_s0[pid] > outlier_gate) && p_s0[pid] > m3 {
                            m3 = p_s0[pid];
                            skip_c = pid;
                        }
                    }
                }

                let mut clean_s0 = 0.0f32;
                let mut clean_cnt = 0usize;
                for pid in 0..n_patches {
                    if pid != skip_a && pid != skip_b && pid != skip_c {
                        clean_s0 += p_s0[pid];
                        clean_cnt += 1;
                    }
                }

                if clean_s0 > 1e-4 {
                    let w_max = (frame_w as i32) - 1;
                    let h_max = (frame_h as i32) - 1;
                    let win_span = 8 * scale_px + 2;
                    let tile_stride = win_span.max(18);
                    let tile_area = tile_stride * tile_stride;
                    let use_p2_18 = win_span == 18 && scale_px == 2 && phase_mask == 1;
                    let cur_patches_eo: &mut [[f32; 64]; 8] =
                        &mut *(a.roi_by.as_mut_ptr() as *mut [[f32; 64]; 8]);
                    for p in cur_patches_eo.iter_mut() {
                        p.fill(0.0);
                    }
                    let mut p_s0_gn = [0.0f32; 8];
                    let mut clean_s0_gn = 0.0f32;
                    for pid in 0..n_patches {
                        if pid == skip_a || pid == skip_b || pid == skip_c {
                            continue;
                        }
                        if use_p2_18 {
                            unsafe {
                                reorder_patch64_eo_simd128(
                                    &cur_patches[pid],
                                    &mut cur_patches_eo[pid],
                                );
                            }
                        }
                        let p_base = pid * tile_area;
                        let base_ix = (patch_px[pid] as i32) - pan_dx - 1;
                        let base_iy = (patch_py[pid] as i32) - pan_dy - 1;
                        if use_p2_18 {
                            unsafe {
                                unpack_rgba_luma_win18_p2_simd128(
                                    prev_rgba,
                                    frame_w,
                                    frame_h,
                                    base_ix,
                                    base_iy,
                                    &mut a.roi_luma[p_base..(p_base + 324)],
                                );
                            }
                        } else if win_span == 18 {
                            unsafe {
                                unpack_rgba_luma_window_simd128(
                                    prev_rgba,
                                    frame_w,
                                    frame_h,
                                    base_ix,
                                    base_iy,
                                    18,
                                    &mut a.roi_luma[p_base..(p_base + 324)],
                                );
                            }
                        } else {
                            for ly in 0..win_span {
                                let iy = (base_iy + (ly as i32)).clamp(0, h_max) as usize;
                                let r_off = iy * frame_w;
                                let l_row = p_base + ly * tile_stride;
                                for lx in 0..win_span {
                                    let ix = (base_ix + (lx as i32)).clamp(0, w_max) as usize;
                                    let o4 = (r_off + ix) * 4;
                                    a.roi_luma[l_row + lx] = (0.299 * (prev_rgba[o4] as f32)
                                        + 0.587 * (prev_rgba[o4 + 1] as f32)
                                        + 0.114 * (prev_rgba[o4 + 2] as f32))
                                        * (1.0 / 255.0);
                                }
                            }
                        }
                        let pl = &a.roi_luma[p_base..(p_base + tile_area)];
                        let s0g = if use_p2_18 {
                            unsafe {
                                subpx_patch_gain_norm_sad_p2_18_simd128(
                                    &cur_patches_eo[pid],
                                    cur_mad[pid],
                                    pl,
                                    1,
                                    1,
                                    1.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                )
                            }
                        } else {
                            eval_subpx_patch_gain_norm_sad(
                                &cur_patches[pid],
                                cur_mad[pid],
                                pl,
                                tile_stride,
                                scale_px,
                                phase_mask,
                                1,
                                1,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                            )
                        };
                        p_s0_gn[pid] = s0g;
                        clean_s0_gn += s0g;
                    }

                    let mut best_sub_sad = clean_s0_gn;
                    let mut best_hx = 0i32;
                    let mut best_hy = 0i32;
                    let mut best_agree = 0usize;

                    let eval_subpx_cand = |hx: i32,
                                           hy: i32,
                                           best_sub_sad: &mut f32,
                                           best_hx: &mut i32,
                                           best_hy: &mut i32,
                                           best_agree: &mut usize| {
                        let c_sx = (hx as f32) * 0.0625;
                        let c_sy = (hy as f32) * 0.0625;
                        let off_x_base = if hx > 0 { 0usize } else { 1usize };
                        let off_y_base = if hy > 0 { 0usize } else { 1usize };
                        let fx = if hx > 0 {
                            ((16 - hx) as f32) * 0.0625
                        } else {
                            ((-hx) as f32) * 0.0625
                        };
                        let fy = if hy > 0 {
                            ((16 - hy) as f32) * 0.0625
                        } else {
                            ((-hy) as f32) * 0.0625
                        };
                        let w00 = (1.0 - fx) * (1.0 - fy);
                        let w10 = fx * (1.0 - fy);
                        let w01 = (1.0 - fx) * fy;
                        let w11 = fx * fy;

                        let mut cand_sad = 1e-4 * (c_sx * c_sx + c_sy * c_sy);
                        let mut agree = 0usize;
                        for pid in 0..n_patches {
                            if pid == skip_a || pid == skip_b || pid == skip_c {
                                continue;
                            }
                            let p_base = pid * tile_area;
                            let pl = &a.roi_luma[p_base..(p_base + tile_area)];
                            let p_sad = if use_p2_18 {
                                unsafe {
                                    subpx_patch_gain_norm_sad_p2_18_simd128(
                                        &cur_patches_eo[pid],
                                        cur_mad[pid],
                                        pl,
                                        off_x_base,
                                        off_y_base,
                                        w00,
                                        w10,
                                        w01,
                                        w11,
                                    )
                                }
                            } else {
                                eval_subpx_patch_gain_norm_sad(
                                    &cur_patches[pid],
                                    cur_mad[pid],
                                    pl,
                                    tile_stride,
                                    scale_px,
                                    phase_mask,
                                    off_x_base,
                                    off_y_base,
                                    w00,
                                    w10,
                                    w01,
                                    w11,
                                )
                            };
                            if p_sad < 0.86 * p_s0_gn[pid] && (p_s0_gn[pid] - p_sad) > 0.015 / s {
                                agree += 1;
                            }
                            cand_sad += p_sad;
                            if cand_sad >= *best_sub_sad {
                                break;
                            }
                        }
                        if cand_sad < *best_sub_sad {
                            *best_sub_sad = cand_sad;
                            *best_hx = hx;
                            *best_hy = hy;
                            *best_agree = agree;
                        }
                    };

                    let skip_subpx_sweep = !int_pan_ok && (clean_cnt < 4 || clean_s0_gn <= 0.12);
                    if !skip_subpx_sweep {
                        // Stage 1: 1/4-px grid in [-12/16, +12/16]
                        for ky in -3i32..=3i32 {
                            let hy = ky * 4;
                            for kx in -3i32..=3i32 {
                                if kx == 0 && ky == 0 {
                                    continue;
                                }
                                eval_subpx_cand(
                                    kx * 4,
                                    hy,
                                    &mut best_sub_sad,
                                    &mut best_hx,
                                    &mut best_hy,
                                    &mut best_agree,
                                );
                            }
                        }

                        // Stage 2: 1/8-px refinement around Stage 1 optimum
                        let q1_hx = best_hx;
                        let q1_hy = best_hy;
                        for dhy in -1i32..=1i32 {
                            let hy = (q1_hy + dhy * 2).clamp(-15, 15);
                            for dhx in -1i32..=1i32 {
                                let hx = (q1_hx + dhx * 2).clamp(-15, 15);
                                if (hx == q1_hx && hy == q1_hy) || (hx == 0 && hy == 0) {
                                    continue;
                                }
                                eval_subpx_cand(
                                    hx,
                                    hy,
                                    &mut best_sub_sad,
                                    &mut best_hx,
                                    &mut best_hy,
                                    &mut best_agree,
                                );
                            }
                        }

                        // Stage 3: 1/16-px (0.0625 px) polish around Stage 2 optimum
                        let q2_hx = best_hx;
                        let q2_hy = best_hy;
                        for dhy in -1i32..=1i32 {
                            let hy = (q2_hy + dhy).clamp(-15, 15);
                            for dhx in -1i32..=1i32 {
                                let hx = (q2_hx + dhx).clamp(-15, 15);
                                if (hx == q2_hx && hy == q2_hy) || (hx == 0 && hy == 0) {
                                    continue;
                                }
                                eval_subpx_cand(
                                    hx,
                                    hy,
                                    &mut best_sub_sad,
                                    &mut best_hx,
                                    &mut best_hy,
                                    &mut best_agree,
                                );
                            }
                        }
                    }

                    if int_pan_ok {
                        if best_sub_sad < 0.99 * clean_s0_gn {
                            sub_x = (best_hx as f32) * 0.0625;
                            sub_y = (best_hy as f32) * 0.0625;
                        }
                    } else if clean_cnt >= 4
                        && ((best_hx.abs() >= 8 || best_hy.abs() >= 8)
                            || (clean_cnt >= 5
                                && (best_hx.abs() >= 2 || best_hy.abs() >= 2)
                                && best_sub_sad < 0.84 * clean_s0_gn
                                && best_agree * 4 >= clean_cnt * 3))
                        && clean_s0_gn > 0.12
                        && best_sub_sad < 0.84 * clean_s0_gn
                        && (clean_s0_gn - best_sub_sad) > 0.015 * (clean_cnt as f32)
                        && best_agree * 3 >= clean_cnt * 2
                    {
                        valid_pan = true;
                        sub_x = (best_hx as f32) * 0.0625;
                        sub_y = (best_hy as f32) * 0.0625;
                    }
                }
            }
            if !valid_pan
                && !swarm_pan_guard
                && uniform_tiles >= 9
                && a.cam_pan_streak >= 2
                && anchored_2d_tiles == 0
                && hard_cut_tiles < 8
                && sp_stat_landmarks == 0
                && (prev_vx * prev_vx + prev_vy * prev_vy) >= 0.20
            {
                let dec_vx = 0.94 * prev_vx;
                let dec_vy = 0.94 * prev_vy;
                pan_dx = round_f(dec_vx);
                pan_dy = round_f(dec_vy);
                sub_x = dec_vx - (pan_dx as f32);
                sub_y = dec_vy - (pan_dy as f32);
                valid_pan = true;
            }
        }

        let pan_dx_f = (pan_dx as f32) + sub_x;
        let pan_dy_f = (pan_dy as f32) + sub_y;
        let disp2 = pan_dx_f * pan_dx_f + pan_dy_f * pan_dy_f;
        if valid_pan && disp2 >= 0.12 * 0.12 {
            let pan_dx_round = if pan_dx_f >= 0.0 {
                (pan_dx_f + 0.5) as i32
            } else {
                (pan_dx_f - 0.5) as i32
            };
            let pan_dy_round = if pan_dy_f >= 0.0 {
                (pan_dy_f + 0.5) as i32
            } else {
                (pan_dy_f - 0.5) as i32
            };
            let frac_mag = (pan_dx_f - (pan_dx_round as f32)).abs()
                + (pan_dy_f - (pan_dy_round as f32)).abs();
            let (exp_gain, exp_off) = estimate_pan_exposure_symmetric(
                &a.rgba_ring[slot_cur],
                &a.rgba_ring[slot_prev],
                &mut a.eca_res[..4096],
                frame_w,
                frame_h,
                pan_dx_round,
                pan_dy_round,
                pan_dx_f,
                pan_dy_f,
            );
            let exp_fired = (exp_gain - 1.0).abs() >= 0.003 || exp_off.abs() >= 0.002;
            if exp_fired {
                exp_hold = 8;
            } else {
                exp_hold = exp_hold.saturating_sub(1);
            }
            let exp_regime = exp_hold > 0;

            // Check Tier-2 nested 4-DoF Similarity / 6-DoF Affine / 8-DoF Planar Homography only
            // when active camera motion is large enough for roll/zoom/tilt (> 0.5 px or < 8 anchored tiles).
            // Tier-0 (stationary) and Tier-1 (pure translation) remain completely untouched.
            let warp_fit = if disp2 >= 0.25 || anchored_2d_tiles < 8 {
                estimate_warp_tier2(
                    &a.luma,
                    &a.rgba_ring[slot_prev],
                    &a.targets,
                    frame_w,
                    frame_h,
                    pan_dx_f,
                    pan_dy_f,
                    anchored_2d_tiles,
                    0.80,
                )
            } else {
                None
            };

            let active_warp = if let Some(wm) = warp_fit {
                simd_ops::advect_bg_warp_simd128(
                    a.bg_luma.as_mut_ptr(),
                    a.eca_res.as_mut_ptr(),
                    a.luma.as_ptr(),
                    a.bg_stat_streak.as_mut_ptr(),
                    a.bg_eval_mask.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    wm.tier,
                    &wm.h,
                    exp_gain,
                    exp_off,
                );
                advect_tracks_warp(a, frame_w, frame_h, &wm);
                wm
            } else {
                // `a.eca_res` is overwritten immediately after in `chronos_pre_tick`, giving zero-allocation
                // full-frame scratch for strictly causal SIMD128 separable 4-tap Catmull-Rom background advection.
                advect_bg_subpixel_simd128(
                    a.bg_luma.as_mut_ptr(),
                    a.eca_res.as_mut_ptr(),
                    a.luma.as_ptr(),
                    a.bg_stat_streak.as_mut_ptr(),
                    a.bg_eval_mask.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    pan_dx_round,
                    pan_dy_round,
                    pan_dx_f,
                    pan_dy_f,
                    frac_mag,
                    exp_gain,
                    exp_off,
                );

                for m in 0..MAX_TARGETS {
                    if a.targets[m].active != 0 {
                        a.targets[m].pos_x += pan_dx_f;
                        a.targets[m].pos_y += pan_dy_f;
                        a.targets[m].pred_cx += pan_dx_f;
                        a.targets[m].pred_cy += pan_dy_f;
                        a.targets[m].entry_x += pan_dx_f;
                        a.targets[m].entry_y += pan_dy_f;
                        a.targets[m].birth_x += pan_dx_f;
                        a.targets[m].birth_y += pan_dy_f;
                        a.targets[m].step_ref_x += pan_dx_f;
                        a.targets[m].step_ref_y += pan_dy_f;
                        a.targets[m].stim_ref_x += pan_dx_f;
                        a.targets[m].stim_ref_y += pan_dy_f;
                        a.targets[m].osc_mean_x += pan_dx_f;
                        a.targets[m].osc_mean_y += pan_dy_f;
                    }
                    if a.kalman[m].active != 0 {
                        a.kalman[m].x += pan_dx_f;
                        a.kalman[m].y += pan_dy_f;
                    }
                    if a.imm[m].active != 0 {
                        for k in 0..4usize {
                            a.imm[m].x[k] += pan_dx_f;
                            a.imm[m].y[k] += pan_dy_f;
                        }
                        a.imm[m].eq_x += pan_dx_f;
                        a.imm[m].eq_y += pan_dy_f;
                        a.imm[m].piv_x += pan_dx_f;
                        a.imm[m].piv_y += pan_dy_f;
                        a.imm[m].pred_x += pan_dx_f;
                        a.imm[m].pred_y += pan_dy_f;
                    }
                }
                for di in 0..MAX_DORMANT {
                    if a.dormant[di].active != 0 {
                        a.dormant[di].last_x += pan_dx_f;
                        a.dormant[di].last_y += pan_dy_f;
                    }
                }
                WarpModel::identity()
            };

            a.last_pan_dx = pan_dx_round;
            a.last_pan_dy = pan_dy_round;
            a.cum_cam_x -= pan_dx_f;
            a.cum_cam_y -= pan_dy_f;
            a.hist_cam_x[(f_idx as usize) & HIST_MASK] = a.cum_cam_x;
            a.hist_cam_y[(f_idx as usize) & HIST_MASK] = a.cum_cam_y;

            let streak_disp = 0.15;
            if disp2 >= streak_disp * streak_disp {
                a.cam_pan_streak = if coast_only_pan {
                    a.cam_pan_streak.saturating_sub(1)
                } else if sparse_used {
                    a.cam_pan_streak.saturating_add(1).min(16)
                } else if a.cam_pan_streak >= 2 {
                    a.cam_pan_streak.saturating_add(2).min(8)
                } else {
                    a.cam_pan_streak.saturating_add(1)
                };
            }
            let disp_px = wasm_sqrtf(disp2);
            let inst_pan_v = disp_px / step_dt.max(1e-3);
            a.cam_pan_speed_px_s = if a.cam_pan_speed_px_s > 1.0 {
                0.55 * a.cam_pan_speed_px_s + 0.45 * inst_pan_v
            } else {
                inst_pan_v
            };
            if a.cam_pan_streak >= 2 || disp2 >= (2.2 * s) * (2.2 * s) || active_warp.tier >= 1 {
                a.cam_motion_state = 1;
            }
            let cut_hold_next = cut_hold_in.saturating_sub(1);
            a.cam_cut_hold = pack_cam_hold(cut_hold_next, exp_hold, pan_dx_f, pan_dy_f);
            return PanMotion {
                scene_cut: false,
                dx: pan_dx_round,
                dy: pan_dy_round,
                dx_f: pan_dx_f,
                dy_f: pan_dy_f,
                frac_mag,
                exp_gain,
                exp_off,
                exp_regime,
                warp: active_warp,
            };
        }
    }

    // Pure centered roll/zoom/tilt entry when Tier-1 found ~0 center translation but >= 12 tiles moved
    // and zero 2D-textured background tiles remained anchored (skipped during active micro-swarms
    // where 1,000+ moving foreground particles saturate tiles without camera motion).
    if f_idx >= 2
        && !wind_lab_lock
        && a.micro_swarm.swarm_active == 0
        && anchored_2d_tiles == 0
        && stat_tiles <= 4
        && hard_cut_tiles < 13
        && !swarm_pan_guard
    {
        if let Some(wm) = estimate_warp_tier2(
            &a.luma,
            &a.rgba_ring[slot_prev],
            &a.targets,
            frame_w,
            frame_h,
            0.0,
            0.0,
            anchored_2d_tiles,
            0.72,
        ) {
            simd_ops::advect_bg_warp_simd128(
                a.bg_luma.as_mut_ptr(),
                a.eca_res.as_mut_ptr(),
                a.luma.as_ptr(),
                a.bg_stat_streak.as_mut_ptr(),
                a.bg_eval_mask.as_mut_ptr(),
                frame_w,
                frame_h,
                wm.tier,
                &wm.h,
                1.0,
                0.0,
            );
            advect_tracks_warp(a, frame_w, frame_h, &wm);
            let w_dx = wm.h[2];
            let w_dy = wm.h[5];
            let w_dx_r = round_f(w_dx);
            let w_dy_r = round_f(w_dy);
            a.last_pan_dx = w_dx_r;
            a.last_pan_dy = w_dy_r;
            a.cum_cam_x -= w_dx;
            a.cum_cam_y -= w_dy;
            a.hist_cam_x[(f_idx as usize) & HIST_MASK] = a.cum_cam_x;
            a.hist_cam_y[(f_idx as usize) & HIST_MASK] = a.cum_cam_y;
            a.cam_pan_streak = a.cam_pan_streak.saturating_add(1).min(8);
            a.cam_motion_state = 1;
            let cut_hold_next = cut_hold_in.saturating_sub(1);
            exp_hold = exp_hold.saturating_sub(1);
            a.cam_cut_hold = pack_cam_hold(cut_hold_next, exp_hold, w_dx, w_dy);
            return PanMotion {
                scene_cut: false,
                dx: w_dx_r,
                dy: w_dy_r,
                dx_f: w_dx,
                dy_f: w_dy,
                frac_mag: (w_dx - (w_dx_r as f32)).abs() + (w_dy - (w_dy_r as f32)).abs(),
                exp_gain: 1.0,
                exp_off: 0.0,
                exp_regime: exp_hold > 0,
                warp: wm,
            };
        }
    }

    if f_idx >= 2 && hard_cut_tiles >= 13 {
        for m in 0..MAX_TARGETS {
            a.targets[m] = core::mem::zeroed();
            a.kalman[m] = core::mem::zeroed();
            a.imm[m].reset();
            a.ode_states[m].reset();
            a.pend_states[m].reset();
            a.prev_fiedler_track[m] = 0.0;
            a.all_pip_tid[m] = 0;
            a.all_pip_ref_luma[m].fill(0.0);
            for q in 0..8 {
                a.slot_meta[m * 8 + q] = 0.0;
            }
        }
        a.ode_scratch.reset();
        a.lyap_scratch.reset();
        a.collision_state.reset();
        a.telemetry.fill(0.0);
        a.telemetry[20] = core::mem::size_of::<Arena>() as f32;
        a.physics_telem_dirty_mask = 0;
        for di in 0..MAX_DORMANT {
            a.dormant[di] = core::mem::zeroed();
        }
        a.pip_ref_luma.fill(0.0);
        for i in 0..frame_pixels {
            a.bg_luma[i] = a.luma[i];
            a.eca_res[i] = 0.0;
            a.bg_stat_streak[i] = 0;
            a.bg_eval_mask[i] = 0;
        }
        for i in 0..PYR_PIXELS {
            a.pyr_eca[i] = 0.0;
        }
        a.num_peaks = 0;
        a.pip_target_id = 0;
        a.pip_hold_frames = 0;
        a.focus_target_id = 0;
        a.clutter_db = 0.0;
        a.clutter_db_valid = 0;
        a.clutter_eval_ratio = 0.0;
        a.eca_err2_ema = 0.0;
        a.eca_sig2_ema = 0.0;
        a.scene_gravity_px_s2 = 0.0;
        a.scene_gravity_conf = 0.0;
        a.scene_gravity_arcs = 0;
        a.last_grav_arc_track = 0;
        a.last_grav_arc_frame = 0;
        a.scatter_deg = 0.0;
        a.scatter_event_frame = 0;
        a.last_discont_frame = f_idx;
        a.discont_count = a.discont_count.saturating_add(1);
        a.cam_pan_streak = 0;
        a.cam_pan_speed_px_s = 0.0;
        a.last_pan_dx = 0;
        a.last_pan_dy = 0;
        a.cum_cam_x = 0.0;
        a.cum_cam_y = 0.0;
        a.hist_cam_x = [0.0; HIST_RING_LEN];
        a.hist_cam_y = [0.0; HIST_RING_LEN];
        a.cam_cut_hold = pack_cam_hold(10, 0, 0.0, 0.0);
        a.cam_motion_state = 2;
        return PanMotion::cut();
    }

    let mut stat_exp_gain = 1.0f32;
    let mut stat_exp_off = 0.0f32;
    let mut stat_exp_fired = false;
    let coherent_exp_cand = stat_tiles >= 8
        && (exp_hold > 0 || pos_exp_tiles >= 6 || neg_exp_tiles >= 6 || pos_gain_tiles >= 5 || neg_gain_tiles >= 5);
    let mut exp_n = 0usize;
    if coherent_exp_cand {
        let curr_rgba = &a.rgba_ring[slot_cur];
        let prev_rgba = &a.rgba_ring[slot_prev];
        const INV_255: f32 = 1.0 / 255.0;
        for si in 0..(stat_tiles as usize) {
            let tx = stat_tx[si];
            let ty = stat_ty[si];
            for r in 0..6usize {
                let py = (ty + r * 2) * scale_px;
                let row_b = py * frame_w;
                for c in 0..8usize {
                    let px = (tx + c * 2) * scale_px;
                    let o4 = (row_b + px) * 4;
                    if prev_rgba[o4].max(prev_rgba[o4 + 1]) >= 236
                        || curr_rgba[o4].max(curr_rgba[o4 + 1]) >= 236
                    {
                        continue;
                    }
                    for ch in 0..3usize {
                        if exp_n >= 2304 {
                            break;
                        }
                        if ch == 2 && (prev_rgba[o4 + 2] >= 238 || curr_rgba[o4 + 2] >= 238) {
                            continue;
                        }
                        let b = (prev_rgba[o4 + ch] as f32) * INV_255;
                        let y = (curr_rgba[o4 + ch] as f32) * INV_255;
                        if b >= 0.04 && b <= 0.92 && y >= 0.04 && y <= 0.92 {
                            a.eca_res[exp_n] = b;
                            a.eca_res[2304 + exp_n] = y;
                            exp_n += 1;
                        }
                    }
                }
            }
        }
    }
    let coherent_exp = coherent_exp_cand && exp_n >= 96;
    if coherent_exp {
        if let Some((cand_g, cand_o, pre_mse, post_mse)) =
            fit_affine_rgb_trimmed(&a.eca_res[..exp_n], &a.eca_res[2304..(2304 + exp_n)])
        {
            let explained_mse = pre_mse - post_mse;
            const POST_ABS: f32 = (4.20 / 255.0) * (4.20 / 255.0);
            let pass_exp = if exp_hold > 0 {
                ((cand_g - 1.0).abs() >= 0.003 || cand_o.abs() >= 0.002)
                    && explained_mse > (0.30 / 255.0) * (0.30 / 255.0)
                    && post_mse < POST_ABS
                    && post_mse < 0.98 * pre_mse
            } else {
                ((cand_g - 1.0).abs() >= 0.008 || cand_o.abs() >= 0.006)
                    && explained_mse > (2.50 / 255.0) * (2.50 / 255.0)
                    && post_mse < POST_ABS
                    && post_mse < 0.085 * pre_mse
            };
            if pass_exp {
                stat_exp_gain = cand_g;
                stat_exp_off = cand_o;
                stat_exp_fired = true;
                advect_bg_subpixel_simd128(
                    a.bg_luma.as_mut_ptr(),
                    a.eca_res.as_mut_ptr(),
                    a.luma.as_ptr(),
                    a.bg_stat_streak.as_mut_ptr(),
                    a.bg_eval_mask.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    cand_g,
                    cand_o,
                );
            }
        }
    }
    if stat_exp_fired {
        exp_hold = 8;
    } else {
        exp_hold = exp_hold.saturating_sub(1);
    }

    a.last_pan_dx = 0;
    a.last_pan_dy = 0;
    a.hist_cam_x[(f_idx as usize) & HIST_MASK] = a.cum_cam_x;
    a.hist_cam_y[(f_idx as usize) & HIST_MASK] = a.cum_cam_y;
    if wind_lab_lock {
        a.cam_pan_streak = 0;
        a.cam_pan_speed_px_s = 0.0;
    } else {
        a.cam_pan_streak = a.cam_pan_streak.saturating_sub(1);
        a.cam_pan_speed_px_s *= 0.65;
    }
    let cut_hold_next = cut_hold_in.saturating_sub(1);
    if cut_hold_next > 0 {
        a.cam_motion_state = 2;
    } else if a.cam_pan_streak >= 2 {
        a.cam_motion_state = 1;
    } else {
        a.cam_motion_state = 0;
    }
    let (save_vx, save_vy) = if a.cam_pan_streak > 0 {
        (prev_vx, prev_vy)
    } else {
        (0.0, 0.0)
    };
    a.cam_cut_hold = pack_cam_hold(cut_hold_next, exp_hold, save_vx, save_vy);
    PanMotion {
        scene_cut: false,
        dx: 0,
        dy: 0,
        dx_f: 0.0,
        dy_f: 0.0,
        frac_mag: 0.0,
        exp_gain: stat_exp_gain,
        exp_off: stat_exp_off,
        exp_regime: exp_hold > 0,
        warp: WarpModel::identity(),
    }
}

/// Bilinear luma sample from packed RGBA8 buffer (`0.299 R + 0.587 G + 0.114 B` normalized to `[0, 1]`).
#[inline(always)]
fn sample_rgba_luma_bilin(rgba: &[u8], w: usize, h: usize, x: f32, y: f32) -> Option<f32> {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
        return None;
    }
    let ix = x as usize;
    let iy = y as usize;
    if ix + 1 >= w || iy + 1 >= h {
        return None;
    }
    let fx = x - (ix as f32);
    let fy = y - (iy as f32);
    let luma_at = |px: usize, py: usize| -> f32 {
        let off = (py * w + px) * 4;
        (0.299 * (rgba[off] as f32) + 0.587 * (rgba[off + 1] as f32) + 0.114 * (rgba[off + 2] as f32))
            * (1.0 / 255.0)
    };
    let s00 = luma_at(ix, iy);
    let s10 = luma_at(ix + 1, iy);
    let s01 = luma_at(ix, iy + 1);
    let s11 = luma_at(ix + 1, iy + 1);
    let top = s00 + fx * (s10 - s00);
    let bot = s01 + fx * (s11 - s01);
    Some(top + fy * (bot - top))
}

const MAX_WARP_SAMPLES: usize = 24;

/// Solves a 3x3 symmetric positive-definite linear system via Cholesky (`L L^T x = b`).
#[inline(always)]
fn solve_spd_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let l00 = if a[0][0] > 1e-12 { wasm_sqrt_f64(a[0][0]) } else { return None };
    let l10 = a[1][0] / l00;
    let d11 = a[1][1] - l10 * l10;
    let l11 = if d11 > 1e-12 { wasm_sqrt_f64(d11) } else { return None };
    let l20 = a[2][0] / l00;
    let l21 = (a[2][1] - l20 * l10) / l11;
    let d22 = a[2][2] - l20 * l20 - l21 * l21;
    let l22 = if d22 > 1e-12 { wasm_sqrt_f64(d22) } else { return None };

    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let y2 = (b[2] - l20 * y0 - l21 * y1) / l22;

    let x2 = y2 / l22;
    let x1 = (y1 - l21 * x2) / l11;
    let x0 = (y0 - l10 * x1 - l20 * x2) / l00;
    if x0.is_finite() && x1.is_finite() && x2.is_finite() {
        Some([x0, x1, x2])
    } else {
        None
    }
}

/// Solves an 8x8 symmetric positive-definite linear system via in-place Cholesky (`L L^T x = b`).
#[inline(never)]
fn solve_spd_8x8(a: &mut [[f64; 8]; 8], b: &[f64; 8]) -> Option<[f64; 8]> {
    for i in 0..8usize {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= a[i][k] * a[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-12 {
                    return None;
                }
                a[i][j] = wasm_sqrt_f64(sum);
            } else {
                a[i][j] = sum / a[j][j];
            }
        }
    }
    let mut y = [0.0f64; 8];
    for i in 0..8usize {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i][k] * y[k];
        }
        y[i] = sum / a[i][i];
    }
    let mut x = [0.0f64; 8];
    for rev in 0..8usize {
        let i = 7 - rev;
        let mut sum = y[i];
        for k in (i + 1)..8usize {
            sum -= a[k][i] * x[k];
        }
        let xi = sum / a[i][i];
        if !xi.is_finite() {
            return None;
        }
        x[i] = xi;
    }
    Some(x)
}

/// Solves centered 4-DoF similarity `(tx, ty, a = s*cos(theta)-1, b = s*sin(theta))` from
/// matched patch displacements `(px_i, py_i, du_i, dv_i)` via 3-step Huber-IRLS on the
/// exact 4x4 SPD normal equations in `f64`.
#[inline(never)]
fn fit_similarity_4dof_irls(
    px: &[f32],
    py: &[f32],
    du: &[f32],
    dv: &[f32],
    n: usize,
    cx: f64,
    cy: f64,
) -> Option<(f32, f32, f32, f32, u8, usize)> {
    if n < 6 || n > MAX_WARP_SAMPLES {
        return None;
    }
    let mut wts = [1.0f64; MAX_WARP_SAMPLES];
    let mut p = [0.0f64; 4]; // [tx, ty, a, b]
    let delta = 0.75f64;

    for iter in 0..3usize {
        let mut s0 = 0.0f64;
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        let mut sr = 0.0f64;
        let mut bu = 0.0f64;
        let mut bv = 0.0f64;
        let mut ba = 0.0f64;
        let mut bb = 0.0f64;
        for i in 0..n {
            let wi = wts[i];
            let xr = (px[i] as f64) - cx;
            let yr = (py[i] as f64) - cy;
            let ui = du[i] as f64;
            let vi = dv[i] as f64;
            s0 += wi;
            sx += wi * xr;
            sy += wi * yr;
            sr += wi * (xr * xr + yr * yr);
            bu += wi * ui;
            bv += wi * vi;
            ba += wi * (xr * ui + yr * vi);
            bb += wi * (-yr * ui + xr * vi);
        }
        if s0 < 4.0 || sr < 100.0 {
            return None;
        }
        let inv_s0 = 1.0 / s0;
        let mx = sx * inv_s0;
        let my = sy * inv_s0;
        let mu = bu * inv_s0;
        let mv = bv * inv_s0;
        let denom = sr - s0 * (mx * mx + my * my);
        if !denom.is_finite() || denom < 100.0 {
            return None;
        }
        let a_val = (ba - (sx * mu + sy * mv)) / denom;
        let b_val = (bb - (-sy * mu + sx * mv)) / denom;
        let tx_val = mu - a_val * mx + b_val * my;
        let ty_val = mv - b_val * mx - a_val * my;
        if !tx_val.is_finite() || !ty_val.is_finite() || !a_val.is_finite() || !b_val.is_finite() {
            return None;
        }
        p = [tx_val, ty_val, a_val, b_val];

        if iter < 2 {
            for i in 0..n {
                let xr = (px[i] as f64) - cx;
                let yr = (py[i] as f64) - cy;
                let eu = (du[i] as f64) - (p[0] + p[2] * xr - p[3] * yr);
                let ev = (dv[i] as f64) - (p[1] + p[3] * xr + p[2] * yr);
                let err = wasm_sqrt_f64(eu * eu + ev * ev);
                wts[i] = if err <= delta { 1.0 } else { delta / err };
            }
        }
    }

    let mut quad_mask = 0u8;
    let mut inliers = 0usize;
    for i in 0..n {
        let xr = (px[i] as f64) - cx;
        let yr = (py[i] as f64) - cy;
        let eu = (du[i] as f64) - (p[0] + p[2] * xr - p[3] * yr);
        let ev = (dv[i] as f64) - (p[1] + p[3] * xr + p[2] * yr);
        if eu * eu + ev * ev <= 0.75 * 0.75 {
            inliers += 1;
            let qbit = (if xr >= 0.0 { 1u8 } else { 0u8 }) | (if yr >= 0.0 { 2u8 } else { 0u8 });
            quad_mask |= 1u8 << qbit;
        }
    }
    Some((p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32, quad_mask, inliers))
}

/// Solves centered 6-DoF Affine (`[h0, h1, h2, h3, h4, h5, 0, 0]`) via decoupled 3x3 SPD
/// Cholesky Huber-IRLS in `f64`.
#[inline(never)]
fn fit_affine_6dof_irls(
    px: &[f32],
    py: &[f32],
    du: &[f32],
    dv: &[f32],
    n: usize,
    cx: f64,
    cy: f64,
) -> Option<(WarpModel, u8, usize)> {
    if n < 8 || n > MAX_WARP_SAMPLES {
        return None;
    }
    let mut wts = [1.0f64; MAX_WARP_SAMPLES];
    let mut pu = [0.0f64; 3]; // [h0, h1, h2]
    let mut pv = [0.0f64; 3]; // [h3, h4, h5]
    let delta = 0.75f64;

    for iter in 0..3usize {
        let mut ata = [[0.0f64; 3]; 3];
        let mut bu = [0.0f64; 3];
        let mut bv = [0.0f64; 3];
        for i in 0..n {
            let wi = wts[i];
            let xr = (px[i] as f64) - cx;
            let yr = (py[i] as f64) - cy;
            let ui = du[i] as f64;
            let vi = dv[i] as f64;
            let r = [xr, yr, 1.0];
            for row in 0..3usize {
                bu[row] += wi * r[row] * ui;
                bv[row] += wi * r[row] * vi;
                for col in 0..3usize {
                    ata[row][col] += wi * r[row] * r[col];
                }
            }
        }
        ata[0][0] += 1e-9;
        ata[1][1] += 1e-9;
        ata[2][2] += 1e-9;
        pu = solve_spd_3x3(&ata, &bu)?;
        pv = solve_spd_3x3(&ata, &bv)?;

        if iter < 2 {
            for i in 0..n {
                let xr = (px[i] as f64) - cx;
                let yr = (py[i] as f64) - cy;
                let eu = (du[i] as f64) - (pu[0] * xr + pu[1] * yr + pu[2]);
                let ev = (dv[i] as f64) - (pv[0] * xr + pv[1] * yr + pv[2]);
                let err = wasm_sqrt_f64(eu * eu + ev * ev);
                wts[i] = if err <= delta { 1.0 } else { delta / err };
            }
        }
    }

    let det_j = (1.0 + pu[0]) * (1.0 + pv[1]) - pu[1] * pv[0];
    if !det_j.is_finite() || det_j < 0.80 || det_j > 1.25 {
        return None;
    }
    let mut quad_mask = 0u8;
    let mut inliers = 0usize;
    for i in 0..n {
        let xr = (px[i] as f64) - cx;
        let yr = (py[i] as f64) - cy;
        let eu = (du[i] as f64) - (pu[0] * xr + pu[1] * yr + pu[2]);
        let ev = (dv[i] as f64) - (pv[0] * xr + pv[1] * yr + pv[2]);
        if eu * eu + ev * ev <= 0.75 * 0.75 {
            inliers += 1;
            let qbit = (if xr >= 0.0 { 1u8 } else { 0u8 }) | (if yr >= 0.0 { 2u8 } else { 0u8 });
            quad_mask |= 1u8 << qbit;
        }
    }
    let wm = WarpModel {
        tier: 2,
        h: [
            pu[0] as f32,
            pu[1] as f32,
            pu[2] as f32,
            pv[0] as f32,
            pv[1] as f32,
            pv[2] as f32,
            0.0,
            0.0,
        ],
    };
    Some((wm, quad_mask, inliers))
}

/// Solves centered 8-DoF Planar Homography via 3-step Gauss-Newton Huber-IRLS seeded from an
/// initial affine/similarity `WarpModel` using in-place 8x8 Cholesky in `f64`.
#[inline(never)]
fn fit_homography_8dof_irls(
    px: &[f32],
    py: &[f32],
    du: &[f32],
    dv: &[f32],
    n: usize,
    cx: f64,
    cy: f64,
    init: &WarpModel,
) -> Option<(WarpModel, u8, usize)> {
    if n < 10 || n > MAX_WARP_SAMPLES {
        return None;
    }
    let mut h = [0.0f64; 8];
    for k in 0..8usize {
        h[k] = init.h[k] as f64;
    }
    let delta = 0.75f64;
    let inv_s = 1.0 / wasm_sqrt_f64(cx * cx + cy * cy).max(1.0);
    // Work in normalized coordinates `(xn, yn) = (xr * inv_s, yr * inv_s)` for crisp 8x8 conditioning
    let mut hn = [
        h[0],
        h[1],
        h[2] * inv_s,
        h[3],
        h[4],
        h[5] * inv_s,
        h[6] / inv_s,
        h[7] / inv_s,
    ];

    for _iter in 0..4usize {
        let mut ata = [[0.0f64; 8]; 8];
        let mut atb = [0.0f64; 8];
        for i in 0..n {
            let xn = ((px[i] as f64) - cx) * inv_s;
            let yn = ((py[i] as f64) - cy) * inv_s;
            // Observation `x'_n = xn + du*inv_s`, `y'_n = yn + dv*inv_s`
            let xp_obs = xn + (du[i] as f64) * inv_s;
            let yp_obs = yn + (dv[i] as f64) * inv_s;
            let d = 1.0 + hn[6] * xn + hn[7] * yn;
            if d.abs() < 0.5 {
                return None;
            }
            let inv_d = 1.0 / d;
            let xp = ((1.0 + hn[0]) * xn + hn[1] * yn + hn[2]) * inv_d;
            let yp = (hn[3] * xn + (1.0 + hn[4]) * yn + hn[5]) * inv_d;
            let rx = xp_obs - xp;
            let ry = yp_obs - yp;
            let err_px = wasm_sqrt_f64(rx * rx + ry * ry) / inv_s;
            let wi = if err_px <= delta { 1.0 } else { delta / err_px };

            let jx = [xn * inv_d, yn * inv_d, inv_d, 0.0, 0.0, 0.0, -xp * xn * inv_d, -xp * yn * inv_d];
            let jy = [0.0, 0.0, 0.0, xn * inv_d, yn * inv_d, inv_d, -yp * xn * inv_d, -yp * yn * inv_d];
            for r in 0..8usize {
                atb[r] += wi * (jx[r] * rx + jy[r] * ry);
                for c in 0..=r {
                    ata[r][c] += wi * (jx[r] * jx[c] + jy[r] * jy[c]);
                }
            }
        }
        for r in 0..8usize {
            ata[r][r] += 1e-8;
            for c in (r + 1)..8usize {
                ata[r][c] = ata[c][r];
            }
        }
        let step = solve_spd_8x8(&mut ata, &atb)?;
        let mut step_norm2 = 0.0f64;
        for k in 0..8usize {
            hn[k] += step[k];
            step_norm2 += step[k] * step[k];
        }
        if step_norm2 < 1e-12 {
            break;
        }
    }

    h = [
        hn[0],
        hn[1],
        hn[2] / inv_s,
        hn[3],
        hn[4],
        hn[5] / inv_s,
        hn[6] * inv_s,
        hn[7] * inv_s,
    ];
    // Projective horizon & corner-Jacobian safety bounds
    if !h.iter().all(|v| v.is_finite()) || (h[6].abs() * cx + h[7].abs() * cy) > 0.06 {
        return None;
    }
    let det_lin = (1.0 + h[0]) * (1.0 + h[4]) - h[1] * h[3];
    if det_lin < 0.80 || det_lin > 1.25 {
        return None;
    }
    for &(sx, sy) in &[(-cx, -cy), (cx, -cy), (-cx, cy), (cx, cy)] {
        let d = 1.0 + h[6] * sx + h[7] * sy;
        if d < 0.90 || d > 1.10 {
            return None;
        }
        let loc_det = det_lin / (d * d * d);
        if loc_det < 0.78 || loc_det > 1.28 {
            return None;
        }
    }

    let wm = WarpModel {
        tier: 3,
        h: [
            h[0] as f32,
            h[1] as f32,
            h[2] as f32,
            h[3] as f32,
            h[4] as f32,
            h[5] as f32,
            h[6] as f32,
            h[7] as f32,
        ],
    };
    let mut quad_mask = 0u8;
    let mut inliers = 0usize;
    let cxf = cx as f32;
    let cyf = cy as f32;
    for i in 0..n {
        let (wx, wy) = wm.warp_pt(px[i], py[i], cxf, cyf);
        let eu = (du[i] as f64) - ((wx - px[i]) as f64);
        let ev = (dv[i] as f64) - ((wy - py[i]) as f64);
        if eu * eu + ev * ev <= 0.75 * 0.75 {
            inliers += 1;
            let xr = (px[i] as f64) - cx;
            let yr = (py[i] as f64) - cy;
            let qbit = (if xr >= 0.0 { 1u8 } else { 0u8 }) | (if yr >= 0.0 { 2u8 } else { 0u8 });
            quad_mask |= 1u8 << qbit;
        }
    }
    Some((wm, quad_mask, inliers))
}

/// Evaluates photometric luma SAD of a `WarpModel` (or pure translation when `warp.is_none()`)
/// across the sampled background patches (`cur_luma` vs bilinear `prev_rgba`).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn eval_warp_photometric_sad(
    cur_luma: &[f32],
    prev_rgba: &[u8],
    px: &[f32],
    py: &[f32],
    n: usize,
    frame_w: usize,
    frame_h: usize,
    warp: Option<&WarpModel>,
    trans_dx: f32,
    trans_dy: f32,
) -> (f64, usize) {
    let cxf = 0.5 * ((frame_w - 1) as f32);
    let cyf = 0.5 * ((frame_h - 1) as f32);
    let mut sad_tot = 0.0f64;
    let mut eval_patches = 0usize;
    for i in 0..n {
        let x_base = (px[i] as usize).saturating_sub(4);
        let y_base = (py[i] as usize).saturating_sub(4);
        if x_base + 8 > frame_w || y_base + 8 > frame_h {
            continue;
        }
        let (su, sv) = if let Some(wm) = warp {
            let (wx, wy) = wm.warp_pt(px[i], py[i], cxf, cyf);
            (wx - px[i], wy - py[i])
        } else {
            (trans_dx, trans_dy)
        };
        let mut p_sad = 0.0f64;
        let mut p_ok = true;
        for j in (0..8usize).step_by(2) {
            let row = (y_base + j) * frame_w + x_base;
            let yf = (y_base + j) as f32;
            for k in (0..8usize).step_by(2) {
                let xf = (x_base + k) as f32;
                let cur = cur_luma[row + k] as f64;
                let Some(vp) = sample_rgba_luma_bilin(prev_rgba, frame_w, frame_h, xf - su, yf - sv) else {
                    p_ok = false;
                    break;
                };
                p_sad += (cur - vp as f64).abs();
            }
            if !p_ok {
                break;
            }
        }
        if p_ok {
            sad_tot += p_sad;
            eval_patches += 1;
        }
    }
    (sad_tot, eval_patches)
}

/// Estimates nested Tier-2 warp (`4-DoF Similarity -> 6-DoF Affine -> 8-DoF Planar Homography`)
/// from up to 24 (`6x4`) spatially distributed 8x8 2D-textured background patches using 4-iter
/// mean-removed IC-LK + Huber-IRLS and strict nested photometric SAD verification.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn estimate_warp_tier2(
    cur_luma: &[f32],
    prev_rgba: &[u8],
    targets: &[TargetTrack; MAX_TARGETS],
    frame_w: usize,
    frame_h: usize,
    init_dx: f32,
    init_dy: f32,
    anchored_2d_tiles: u32,
    sim_photom_ratio_max: f64,
) -> Option<WarpModel> {
    if frame_w < 64 || frame_h < 64 || anchored_2d_tiles >= 4 {
        return None;
    }
    let mut px = [0.0f32; MAX_WARP_SAMPLES];
    let mut py = [0.0f32; MAX_WARP_SAMPLES];
    let mut du = [0.0f32; MAX_WARP_SAMPLES];
    let mut dv = [0.0f32; MAX_WARP_SAMPLES];
    let mut n = 0usize;

    let cell_w = frame_w / 6;
    let cell_h = frame_h / 4;
    let margin = 14usize;

    for gy in 0..4usize {
        for gx in 0..6usize {
            let base_cx = (gx * cell_w + cell_w / 2).clamp(margin, frame_w - margin - 1);
            let base_cy = (gy * cell_h + cell_h / 2).clamp(margin, frame_h - margin - 1);

            // Pick highest-corner-score 8x8 subpatch among 3 local offsets inside the cell
            let mut best_cx = base_cx;
            let mut best_cy = base_cy;
            let mut best_h = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64); // (det, hxx, hxy, hyy, mean_t)
            for &(ox, oy) in &[(0isize, 0isize), (-6, 0), (6, 0)] {
                let cx0 = ((base_cx as isize) + ox).clamp(margin as isize, (frame_w - margin - 1) as isize) as usize;
                let cy0 = ((base_cy as isize) + oy).clamp(margin as isize, (frame_h - margin - 1) as isize) as usize;
                let mut on_target = false;
                for m in 0..MAX_TARGETS {
                    if targets[m].active != 0 && targets[m].confirmed != 0 {
                        let tdx = (cx0 as f32) - targets[m].pos_x;
                        let tdy = (cy0 as f32) - targets[m].pos_y;
                        if tdx * tdx + tdy * tdy < 28.0 * 28.0 {
                            on_target = true;
                            break;
                        }
                    }
                }
                if on_target {
                    continue;
                }
                let mut h_xx = 0.0f64;
                let mut h_xy = 0.0f64;
                let mut h_yy = 0.0f64;
                let mut mean_t = 0.0f64;
                let x_base = cx0 - 4;
                let y_base = cy0 - 4;
                for j in 0..8usize {
                    let row = (y_base + j) * frame_w + x_base;
                    for i in 0..8usize {
                        let idx = row + i;
                        let tc = cur_luma[idx] as f64;
                        let gxp = 0.5 * ((cur_luma[idx + 1] as f64) - (cur_luma[idx - 1] as f64));
                        let gyp = 0.5 * ((cur_luma[idx + frame_w] as f64) - (cur_luma[idx - frame_w] as f64));
                        mean_t += tc;
                        h_xx += gxp * gxp;
                        h_xy += gxp * gyp;
                        h_yy += gyp * gyp;
                    }
                }
                let tr = h_xx + h_yy;
                let det = h_xx * h_yy - h_xy * h_xy;
                if det.is_finite() && det > 1e-5 && tr > 1e-3 && det * 400.0 >= tr * tr && det > best_h.0 {
                    best_cx = cx0;
                    best_cy = cy0;
                    best_h = (det, h_xx, h_xy, h_yy, mean_t * (1.0 / 64.0));
                    if ox == 0 && oy == 0 && det > 2e-4 {
                        break;
                    }
                }
            }
            if best_h.0 <= 1e-5 {
                continue;
            }
            let (det, h_xx, h_xy, h_yy, mean_t) = best_h;
            let inv_det = 1.0 / det;
            let x_base = best_cx - 4;
            let y_base = best_cy - 4;

            // 4-iter affine-offset-normalized IC-LK against `prev_rgba` seeded at `(init_dx, init_dy)`
            let mut lx = init_dx as f64;
            let mut ly = init_dy as f64;
            let mut ok = true;
            for _iter in 0..4usize {
                let mut sum_i = 0.0f64;
                let mut bx = 0.0f64;
                let mut by = 0.0f64;
                for j in 0..8usize {
                    let sy = ((y_base + j) as f32) - (ly as f32);
                    for i in 0..8usize {
                        let sx = ((x_base + i) as f32) - (lx as f32);
                        match sample_rgba_luma_bilin(prev_rgba, frame_w, frame_h, sx, sy) {
                            Some(iv) => sum_i += iv as f64,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        break;
                    }
                }
                if !ok {
                    break;
                }
                let mean_i = sum_i * (1.0 / 64.0);
                for j in 0..8usize {
                    let row = (y_base + j) * frame_w + x_base;
                    let sy = ((y_base + j) as f32) - (ly as f32);
                    for i in 0..8usize {
                        let idx = row + i;
                        let sx = ((x_base + i) as f32) - (lx as f32);
                        let iv = match sample_rgba_luma_bilin(prev_rgba, frame_w, frame_h, sx, sy) {
                            Some(v) => v as f64,
                            None => {
                                ok = false;
                                break;
                            }
                        };
                        let res = (cur_luma[idx] as f64 - mean_t) - (iv - mean_i);
                        let gxp = 0.5 * ((cur_luma[idx + 1] as f64) - (cur_luma[idx - 1] as f64));
                        let gyp = 0.5 * ((cur_luma[idx + frame_w] as f64) - (cur_luma[idx - frame_w] as f64));
                        bx += gxp * res;
                        by += gyp * res;
                    }
                    if !ok {
                        break;
                    }
                }
                if !ok {
                    break;
                }
                let d_lx = (h_yy * bx - h_xy * by) * inv_det;
                let d_ly = (-h_xy * bx + h_xx * by) * inv_det;
                if !d_lx.is_finite() || !d_ly.is_finite() || d_lx.abs() > 4.0 || d_ly.abs() > 4.0 {
                    ok = false;
                    break;
                }
                lx += d_lx;
                ly += d_ly;
                if d_lx * d_lx + d_ly * d_ly < 1e-4 {
                    break;
                }
            }
            if ok && ((lx - init_dx as f64).abs() <= 10.0) && ((ly - init_dy as f64).abs() <= 10.0) {
                px[n] = best_cx as f32;
                py[n] = best_cy as f32;
                du[n] = lx as f32;
                dv[n] = ly as f32;
                n += 1;
            }
        }
    }

    let cx = 0.5 * ((frame_w - 1) as f64);
    let cy = 0.5 * ((frame_h - 1) as f64);
    let (tx, ty, a_sim, b_sim, quad_mask, inliers) =
        fit_similarity_4dof_irls(&px[..n], &py[..n], &du[..n], &dv[..n], n, cx, cy)?;

    let corner_disp = (a_sim.abs().max(b_sim.abs()) as f64) * wasm_sqrt_f64(cx * cx + cy * cy);
    if corner_disp < 0.50 || a_sim.abs() > 0.08 || b_sim.abs() > 0.08 {
        return None;
    }
    if inliers < 8 || quad_mask != 0b1111 {
        return None;
    }

    // Compare 4-DoF fit against inlier-optimal pure translation `(mean_u, mean_v)`
    let mut sum_u = 0.0f64;
    let mut sum_v = 0.0f64;
    let mut inl_cnt = 0usize;
    let mut sum_sim = 0.0f64;
    for i in 0..n {
        let xr = (px[i] as f64) - cx;
        let yr = (py[i] as f64) - cy;
        let eu_s = (du[i] as f64) - ((tx as f64) + (a_sim as f64) * xr - (b_sim as f64) * yr);
        let ev_s = (dv[i] as f64) - ((ty as f64) + (b_sim as f64) * xr + (a_sim as f64) * yr);
        let es = wasm_sqrt_f64(eu_s * eu_s + ev_s * ev_s);
        if es <= 0.75 {
            sum_u += du[i] as f64;
            sum_v += dv[i] as f64;
            sum_sim += es;
            inl_cnt += 1;
        }
    }
    if inl_cnt < 8 {
        return None;
    }
    let inv_inl = 1.0 / (inl_cnt as f64);
    let mean_u = sum_u * inv_inl;
    let mean_v = sum_v * inv_inl;
    let mut sum_trans_opt = 0.0f64;
    for i in 0..n {
        let xr = (px[i] as f64) - cx;
        let yr = (py[i] as f64) - cy;
        let eu_s = (du[i] as f64) - ((tx as f64) + (a_sim as f64) * xr - (b_sim as f64) * yr);
        let ev_s = (dv[i] as f64) - ((ty as f64) + (b_sim as f64) * xr + (a_sim as f64) * yr);
        if eu_s * eu_s + ev_s * ev_s <= 0.75 * 0.75 {
            let eu_t = (du[i] as f64) - mean_u;
            let ev_t = (dv[i] as f64) - mean_v;
            sum_trans_opt += wasm_sqrt_f64(eu_t * eu_t + ev_t * ev_t);
        }
    }
    if sum_trans_opt < 0.14 * (inl_cnt as f64) || sum_sim > 0.70 * sum_trans_opt {
        return None;
    }

    let sim_wm = WarpModel::from_similarity(tx, ty, a_sim, b_sim);
    let (sad_trans, eval_t) = eval_warp_photometric_sad(
        cur_luma, prev_rgba, &px[..n], &py[..n], n, frame_w, frame_h, None, init_dx, init_dy,
    );
    let (sad_sim, eval_s) = eval_warp_photometric_sad(
        cur_luma, prev_rgba, &px[..n], &py[..n], n, frame_w, frame_h, Some(&sim_wm), init_dx, init_dy,
    );
    let eval_patches = eval_t.min(eval_s);
    if eval_patches < 8
        || sad_trans <= 1e-4
        || sad_sim > sim_photom_ratio_max * sad_trans
        || (sad_trans - sad_sim) < 0.004 * ((eval_patches * 16) as f64)
    {
        return None;
    }

    let mut best_wm = sim_wm;
    let mut best_sad = sad_sim;

    // Nested promotion 1 -> 2: 6-DoF Affine (only if beats 4-DoF Similarity by >= 12% SAD)
    if n >= 10 {
        if let Some((aff_wm, aff_q, aff_inl)) =
            fit_affine_6dof_irls(&px[..n], &py[..n], &du[..n], &dv[..n], n, cx, cy)
        {
            if aff_inl >= 10 && aff_q == 0b1111 {
                let (sad_aff, eval_a) = eval_warp_photometric_sad(
                    cur_luma, prev_rgba, &px[..n], &py[..n], n, frame_w, frame_h, Some(&aff_wm), init_dx, init_dy,
                );
                if eval_a >= eval_patches
                    && sad_aff <= 0.88 * best_sad
                    && (best_sad - sad_aff) >= 0.002 * ((eval_a * 16) as f64)
                {
                    best_wm = aff_wm;
                    best_sad = sad_aff;
                }
            }
        }
    }

    // Nested promotion -> 3: 8-DoF Planar Homography (only if beats best lower-DoF by >= 12% SAD)
    if n >= 12 {
        if let Some((hom_wm, hom_q, hom_inl)) =
            fit_homography_8dof_irls(&px[..n], &py[..n], &du[..n], &dv[..n], n, cx, cy, &best_wm)
        {
            if hom_inl >= 12 && hom_q == 0b1111 {
                let (sad_hom, eval_h) = eval_warp_photometric_sad(
                    cur_luma, prev_rgba, &px[..n], &py[..n], n, frame_w, frame_h, Some(&hom_wm), init_dx, init_dy,
                );
                if eval_h >= eval_patches
                    && sad_hom <= 0.88 * best_sad
                    && (best_sad - sad_hom) >= 0.002 * ((eval_h * 16) as f64)
                {
                    best_wm = hom_wm;
                }
            }
        }
    }

    Some(best_wm)
}

#[inline(never)]
fn advect_tracks_warp(
    a: &mut Arena,
    frame_w: usize,
    frame_h: usize,
    wm: &WarpModel,
) {
    let cx = 0.5 * ((frame_w - 1) as f32);
    let cy = 0.5 * ((frame_h - 1) as f32);

    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            let t = &mut a.targets[m];
            let (px0, py0) = (t.pos_x, t.pos_y);
            let (ex0, ey0) = (t.entry_x, t.entry_y);
            (t.pos_x, t.pos_y) = wm.warp_pt(px0, py0, cx, cy);
            (t.pred_cx, t.pred_cy) = wm.warp_pt(t.pred_cx, t.pred_cy, cx, cy);
            (t.entry_x, t.entry_y) = wm.warp_pt(ex0, ey0, cx, cy);
            (t.birth_x, t.birth_y) = wm.warp_pt(t.birth_x, t.birth_y, cx, cy);
            (t.step_ref_x, t.step_ref_y) = wm.warp_pt(t.step_ref_x, t.step_ref_y, cx, cy);
            (t.stim_ref_x, t.stim_ref_y) = wm.warp_pt(t.stim_ref_x, t.stim_ref_y, cx, cy);
            (t.osc_mean_x, t.osc_mean_y) = wm.warp_pt(t.osc_mean_x, t.osc_mean_y, cx, cy);
            (t.vel_x, t.vel_y) = wm.warp_vel(t.vel_x, t.vel_y, px0, py0, cx, cy);
            (t.entry_vx, t.entry_vy) = wm.warp_vel(t.entry_vx, t.entry_vy, ex0, ey0, cx, cy);
            for s_i in 0..RING_LEN {
                if t.ring_vis[s_i] != 0 {
                    (t.ring_raw_x[s_i], t.ring_raw_y[s_i]) =
                        wm.warp_pt(t.ring_raw_x[s_i], t.ring_raw_y[s_i], cx, cy);
                }
            }
            for h_i in 0..HIST_RING_LEN {
                if t.arc_vis[h_i] != 0 {
                    (t.arc_raw_x[h_i], t.arc_raw_y[h_i]) =
                        wm.warp_pt(t.arc_raw_x[h_i], t.arc_raw_y[h_i], cx, cy);
                }
            }
        }
        if a.kalman[m].active != 0 {
            let (kx, ky) = (a.kalman[m].x, a.kalman[m].y);
            (a.kalman[m].x, a.kalman[m].y) = wm.warp_pt(kx, ky, cx, cy);
            (a.kalman[m].vx, a.kalman[m].vy) = wm.warp_vel(a.kalman[m].vx, a.kalman[m].vy, kx, ky, cx, cy);
        }
        if a.imm[m].active != 0 {
            for k in 0..4usize {
                let (ix, iy) = (a.imm[m].x[k], a.imm[m].y[k]);
                (a.imm[m].x[k], a.imm[m].y[k]) = wm.warp_pt(ix, iy, cx, cy);
                (a.imm[m].vx[k], a.imm[m].vy[k]) = wm.warp_vel(a.imm[m].vx[k], a.imm[m].vy[k], ix, iy, cx, cy);
            }
            (a.imm[m].eq_x, a.imm[m].eq_y) = wm.warp_pt(a.imm[m].eq_x, a.imm[m].eq_y, cx, cy);
            (a.imm[m].piv_x, a.imm[m].piv_y) = wm.warp_pt(a.imm[m].piv_x, a.imm[m].piv_y, cx, cy);
            let (prx, pry) = (a.imm[m].pred_x, a.imm[m].pred_y);
            (a.imm[m].pred_x, a.imm[m].pred_y) = wm.warp_pt(prx, pry, cx, cy);
            (a.imm[m].pred_vx, a.imm[m].pred_vy) =
                wm.warp_vel(a.imm[m].pred_vx, a.imm[m].pred_vy, prx, pry, cx, cy);
        }
    }
    for di in 0..MAX_DORMANT {
        if a.dormant[di].active != 0 {
            let (dx, dy) = (a.dormant[di].last_x, a.dormant[di].last_y);
            (a.dormant[di].last_x, a.dormant[di].last_y) = wm.warp_pt(dx, dy, cx, cy);
            (a.dormant[di].vx, a.dormant[di].vy) =
                wm.warp_vel(a.dormant[di].vx, a.dormant[di].vy, dx, dy, cx, cy);
        }
    }
}

#[cfg(test)]
mod similarity_tests {
    use super::*;

    #[test]
    fn recovers_synthetic_roll_and_zoom_with_outlier() {
        let (w, h) = (640usize, 360usize);
        let cx = 0.5 * ((w - 1) as f64);
        let cy = 0.5 * ((h - 1) as f64);
        let true_tx = 3.25f64;
        let true_ty = -1.75f64;
        let theta = 0.65f64.to_radians();
        let scale = 1.005f64;
        let true_a = scale * theta.cos() - 1.0;
        let true_b = scale * theta.sin();

        let mut px = [0.0f32; 16];
        let mut py = [0.0f32; 16];
        let mut du = [0.0f32; 16];
        let mut dv = [0.0f32; 16];
        let mut n = 0usize;
        for gy in 0..4usize {
            for gx in 0..4usize {
                let x = (gx * 160 + 80) as f64;
                let y = (gy * 90 + 45) as f64;
                let xr = x - cx;
                let yr = y - cy;
                px[n] = x as f32;
                py[n] = y as f32;
                if n == 5 {
                    du[n] = 18.0;
                    dv[n] = -12.0;
                } else {
                    du[n] = (true_tx + true_a * xr - true_b * yr) as f32;
                    dv[n] = (true_ty + true_b * xr + true_a * yr) as f32;
                }
                n += 1;
            }
        }
        let (tx, ty, a_est, b_est, qmask, inliers) =
            fit_similarity_4dof_irls(&px, &py, &du, &dv, n, cx, cy).expect("should fit");
        assert_eq!(qmask, 0b1111);
        assert_eq!(inliers, 15);
        assert!((tx as f64 - true_tx).abs() < 0.05, "tx={tx}");
        assert!((ty as f64 - true_ty).abs() < 0.05, "ty={ty}");
        assert!((a_est as f64 - true_a).abs() < 2e-4, "a={a_est} vs {true_a}");
        assert!((b_est as f64 - true_b).abs() < 2e-4, "b={b_est} vs {true_b}");
    }

    #[test]
    fn recovers_synthetic_6dof_affine_shear_and_anisotropic_scale() {
        let (w, h) = (640usize, 360usize);
        let cx = 0.5 * ((w - 1) as f64);
        let cy = 0.5 * ((h - 1) as f64);
        let true_h = [0.014f64, -0.018, 2.5, 0.009, -0.011, -1.4];
        let mut px = [0.0f32; 24];
        let mut py = [0.0f32; 24];
        let mut du = [0.0f32; 24];
        let mut dv = [0.0f32; 24];
        let mut n = 0usize;
        for gy in 0..4usize {
            for gx in 0..6usize {
                let x = (gx * 100 + 70) as f64;
                let y = (gy * 85 + 50) as f64;
                let xr = x - cx;
                let yr = y - cy;
                px[n] = x as f32;
                py[n] = y as f32;
                if n == 7 {
                    du[n] = -15.0;
                    dv[n] = 11.0;
                } else {
                    du[n] = (true_h[0] * xr + true_h[1] * yr + true_h[2]) as f32;
                    dv[n] = (true_h[3] * xr + true_h[4] * yr + true_h[5]) as f32;
                }
                n += 1;
            }
        }
        let (wm, qmask, inliers) =
            fit_affine_6dof_irls(&px[..n], &py[..n], &du[..n], &dv[..n], n, cx, cy).expect("affine fit");
        assert_eq!(wm.tier, 2);
        assert_eq!(qmask, 0b1111);
        assert_eq!(inliers, 23);
        assert!((wm.h[0] as f64 - true_h[0]).abs() < 2e-4);
        assert!((wm.h[1] as f64 - true_h[1]).abs() < 2e-4);
        assert!((wm.h[2] as f64 - true_h[2]).abs() < 0.05);
        assert!((wm.h[3] as f64 - true_h[3]).abs() < 2e-4);
        assert!((wm.h[4] as f64 - true_h[4]).abs() < 2e-4);
        assert!((wm.h[5] as f64 - true_h[5]).abs() < 0.05);
    }

    #[test]
    fn recovers_synthetic_8dof_planar_homography() {
        let (w, h) = (640usize, 360usize);
        let cx = 0.5 * ((w - 1) as f64);
        let cy = 0.5 * ((h - 1) as f64);
        let true_wm = WarpModel {
            tier: 3,
            h: [0.008, -0.012, 2.2, 0.011, 0.006, -1.6, 8.0e-5, -5.0e-5],
        };
        let mut px = [0.0f32; 24];
        let mut py = [0.0f32; 24];
        let mut du = [0.0f32; 24];
        let mut dv = [0.0f32; 24];
        let mut n = 0usize;
        for gy in 0..4usize {
            for gx in 0..6usize {
                let x = (gx * 100 + 70) as f32;
                let y = (gy * 85 + 50) as f32;
                px[n] = x;
                py[n] = y;
                if n == 11 {
                    du[n] = 14.0;
                    dv[n] = -9.0;
                } else {
                    let (wx, wy) = true_wm.warp_pt(x, y, cx as f32, cy as f32);
                    du[n] = wx - x;
                    dv[n] = wy - y;
                }
                n += 1;
            }
        }
        let (aff_wm, _, _) =
            fit_affine_6dof_irls(&px[..n], &py[..n], &du[..n], &dv[..n], n, cx, cy).expect("aff seed");
        let (wm, qmask, inliers) =
            fit_homography_8dof_irls(&px[..n], &py[..n], &du[..n], &dv[..n], n, cx, cy, &aff_wm)
                .expect("homography fit");
        assert_eq!(wm.tier, 3);
        assert_eq!(qmask, 0b1111);
        assert_eq!(inliers, 23);
        assert!((wm.h[6] - true_wm.h[6]).abs() < 2e-6, "h6={}", wm.h[6]);
        assert!((wm.h[7] - true_wm.h[7]).abs() < 2e-6, "h7={}", wm.h[7]);
        assert!((wm.h[2] - true_wm.h[2]).abs() < 0.05, "h2={}", wm.h[2]);
    }
}

