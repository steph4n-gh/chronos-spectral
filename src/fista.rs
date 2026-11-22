//! Non-causal Hermite occlusion bridges and 2-pass IRL1 Dual Projected-FISTA trend filter.

use super::*;

// ============================================================================
// Subsystem 4: Boundary Hermite Splines + Exact Beck & Teboulle Dual FISTA L1
// ============================================================================

#[inline(always)]
fn eval_primal_from_dual(
    y: &[f32; RING_LEN],
    inv_m: &[f32; RING_LEN],
    u: &[f32; 10],
    x_out: &mut [f32; RING_LEN],
    d2x_out: &mut [f32; 10],
) {
    for k in 0..RING_LEN {
        let mut d2t_u = 0.0f32;
        if k < 10 {
            d2t_u += u[k];
        }
        if k >= 1 && (k - 1) < 10 {
            d2t_u -= 2.0 * u[k - 1];
        }
        if k >= 2 && (k - 2) < 10 {
            d2t_u += u[k - 2];
        }
        x_out[k] = y[k] - inv_m[k] * d2t_u;
    }
    for k in 0..10usize {
        d2x_out[k] = x_out[k] - 2.0 * x_out[k + 1] + x_out[k + 2];
    }
}

/// Solves a single weighted dual FISTA stage (`2` BCM warm-start sweeps + `n_iters` O'Donoghue-restarted
/// Beck-Teboulle iterations) for heterogeneous per-knot SO(2) disk bounds `||(ux_k, uy_k)||_2 <= lam_k[k]`.
#[inline]
fn run_fista_2d_dual_stage(
    yx: &[f32; RING_LEN],
    yy: &[f32; RING_LEN],
    inv_m: &[f32; RING_LEN],
    q_diag: &[f32; 10],
    step_dual: f32,
    lam_k: &[f32; 10],
    n_iters: usize,
    ux: &mut [f32; 10],
    uy: &mut [f32; 10],
    xk: &mut [f32; RING_LEN],
    yk: &mut [f32; RING_LEN],
    d2x: &mut [f32; 10],
    d2y: &mut [f32; 10],
) -> f32 {
    for k in 0..10usize {
        let (pux, puy) = project_dual_disk_2d(ux[k], uy[k], lam_k[k]);
        ux[k] = pux;
        uy[k] = puy;
    }
    eval_primal_from_dual(yx, inv_m, ux, xk, d2x);
    eval_primal_from_dual(yy, inv_m, uy, yk, d2y);
    for _sweep in 0..2usize {
        for k in 0..10usize {
            let cur_d2x = xk[k] - 2.0 * xk[k + 1] + xk[k + 2];
            let cur_d2y = yk[k] - 2.0 * yk[k + 1] + yk[k + 2];
            let inv_qkk = 1.0 / q_diag[k];
            let (next_ux, next_uy) =
                project_dual_disk_2d(ux[k] + cur_d2x * inv_qkk, uy[k] + cur_d2y * inv_qkk, lam_k[k]);
            let dux = next_ux - ux[k];
            if dux != 0.0 {
                ux[k] = next_ux;
                xk[k] -= inv_m[k] * dux;
                xk[k + 1] += 2.0 * inv_m[k + 1] * dux;
                xk[k + 2] -= inv_m[k + 2] * dux;
            }
            let duy = next_uy - uy[k];
            if duy != 0.0 {
                uy[k] = next_uy;
                yk[k] -= inv_m[k] * duy;
                yk[k + 1] += 2.0 * inv_m[k + 1] * duy;
                yk[k + 2] -= inv_m[k + 2] * duy;
            }
        }
    }

    let mut vx_dual = *ux;
    let mut vy_dual = *uy;
    let mut tk = 1.0f32;
    for _iter in 0..n_iters {
        eval_primal_from_dual(yx, inv_m, &vx_dual, xk, d2x);
        eval_primal_from_dual(yy, inv_m, &vy_dual, yk, d2y);
        let mut ux_next = [0.0f32; 10];
        let mut uy_next = [0.0f32; 10];
        let mut restart_dot = 0.0f32;
        for k in 0..10usize {
            let (unx, uny) = project_dual_disk_2d(
                vx_dual[k] + step_dual * d2x[k],
                vy_dual[k] + step_dual * d2y[k],
                lam_k[k],
            );
            ux_next[k] = unx;
            uy_next[k] = uny;
            restart_dot += (-d2x[k]) * (unx - ux[k]) + (-d2y[k]) * (uny - uy[k]);
        }
        let (t_next, beta) = if restart_dot > 0.0 {
            (1.0f32, 0.0f32)
        } else {
            let tn = 0.5 * (1.0 + wasm_sqrtf(1.0 + 4.0 * tk * tk));
            (tn, (tk - 1.0) / tn)
        };
        for k in 0..10usize {
            vx_dual[k] = ux_next[k] + beta * (ux_next[k] - ux[k]);
            vy_dual[k] = uy_next[k] + beta * (uy_next[k] - uy[k]);
        }
        *ux = ux_next;
        *uy = uy_next;
        tk = t_next;
    }

    eval_primal_from_dual(yx, inv_m, ux, xk, d2x);
    eval_primal_from_dual(yy, inv_m, uy, yk, d2y);
    let mut dual_gap = 0.0f32;
    for k in 0..10usize {
        let d2_mag = wasm_sqrtf(d2x[k] * d2x[k] + d2y[k] * d2y[k]);
        dual_gap += (lam_k[k] * d2_mag - (ux[k] * d2x[k] + uy[k] * d2y[k])).max(0.0);
    }
    dual_gap
}

#[inline(never)]
pub(crate) fn solve_noncausal_fista_target(
    t: &mut TargetTrack,
    head_slot: usize,
    u_disp: usize,
    lambda: f32,
    fps: f32,
    frame_idx: u32,
    stimulus_armed: u32,
    stimulus_frame: u32,
    frame_w: usize,
    frame_h: usize,
    scale_f: f32,
    allow_ballistic: bool,
    scene_gravity_px_s2: f32,
    scene_gravity_conf: f32,
    hist_t_sec: &[f64; HIST_RING_LEN],
    hist_cam_x: &[f32; HIST_RING_LEN],
    hist_cam_y: &[f32; HIST_RING_LEN],
    last_discont_frame: u32,
) -> (bool, Option<f32>) {
    let mut yx = [0.0f32; RING_LEN];
    let mut yy = [0.0f32; RING_LEN];
    let mut cam_off_x = [0.0f32; RING_LEN];
    let mut cam_off_y = [0.0f32; RING_LEN];
    let mut vx = [0.0f32; RING_LEN];
    let mut vy = [0.0f32; RING_LEN];
    let mut vis = [0u8; RING_LEN];
    let mut weff = [0.0f32; RING_LEN];
    let mut t_rel_sec = [0.0f32; RING_LEN];
    let mut vis_count = 0usize;

    let t11_sec = hist_t_sec[(frame_idx as usize) & HIST_MASK];
    let fallback_dt = (1.0 / fps.max(8.0)).clamp(0.002, 0.25);
    let disp_f = frame_idx.saturating_sub((RING_LEN - 1 - u_disp) as u32);
    let cam_disp_x = hist_cam_x[(disp_f as usize) & HIST_MASK];
    let cam_disp_y = hist_cam_y[(disp_f as usize) & HIST_MASK];
    let cam_head_off_x = hist_cam_x[(frame_idx as usize) & HIST_MASK] - cam_disp_x;
    let cam_head_off_y = hist_cam_y[(frame_idx as usize) & HIST_MASK] - cam_disp_y;
    let entry_x_disp = t.entry_x + cam_head_off_x;
    let entry_y_disp = t.entry_y + cam_head_off_y;

    for u in 0..RING_LEN {
        let s = (head_slot + 1 + u) % RING_LEN;
        let mut v_u = t.ring_vis[s];
        if frame_idx >= (RING_LEN - 1 - u) as u32 {
            let f_u = frame_idx - (RING_LEN - 1 - u) as u32;
            if last_discont_frame > 0 && f_u < last_discont_frame {
                v_u = 0;
            }
            cam_off_x[u] = hist_cam_x[(f_u as usize) & HIST_MASK] - cam_disp_x;
            cam_off_y[u] = hist_cam_y[(f_u as usize) & HIST_MASK] - cam_disp_y;
            let dt_rel = (hist_t_sec[(f_u as usize) & HIST_MASK] - t11_sec) as f32;
            t_rel_sec[u] = if dt_rel.is_finite() {
                dt_rel
            } else {
                ((u as f32) - 11.0) * fallback_dt
            };
        } else {
            v_u = 0;
            t_rel_sec[u] = ((u as f32) - 11.0) * fallback_dt;
        }
        yx[u] = t.ring_raw_x[s] + cam_off_x[u];
        yy[u] = t.ring_raw_y[s] + cam_off_y[u];
        vx[u] = t.ring_vx[s];
        vy[u] = t.ring_vy[s];
        vis[u] = v_u;
        if v_u != 0 {
            weff[u] = 1.0;
            vis_count += 1;
        }
    }

    if vis_count == 0 {
        // Explicitly positive prior weights when entire 12-frame window is occluded (M > 0 guaranteed)
        for u in 0..RING_LEN {
            weff[u] = 0.50;
        }
    } else {
        // Scan for occluded intervals (u0, u1) pinned on both sides by visible detections:
        // Two-Point Boundary-Value Cubic Hermite Spline Bridge
        let tail_margin = 4.0 * scale_f;
        let mut u = 0usize;
        while u < RING_LEN {
            if vis[u] == 0 {
                let gap_start = u;
                while u < RING_LEN && vis[u] == 0 {
                    u += 1;
                }
                let gap_end = u;

                if gap_start > 0 && gap_end < RING_LEN {
                    let u0 = gap_start - 1;
                    let u1 = gap_end;
                    let dt = (u1 - u0) as f32;

                    let v0x = if u0 >= 1 && vis[u0 - 1] != 0 {
                        yx[u0] - yx[u0 - 1]
                    } else {
                        vx[u0]
                    };
                    let v0y = if u0 >= 1 && vis[u0 - 1] != 0 {
                        yy[u0] - yy[u0 - 1]
                    } else {
                        vy[u0]
                    };

                    // Provisional curvature-aware exit tangent on initial re-emergence frame u1 == 11 (revised next frame)
                    let (v1x, v1y) = if u1 + 1 < RING_LEN && vis[u1 + 1] != 0 {
                        (yx[u1 + 1] - yx[u1], yy[u1 + 1] - yy[u1])
                    } else {
                        let sec_vx = (yx[u1] - yx[u0]) / dt;
                        let sec_vy = (yy[u1] - yy[u0]) / dt;
                        if t.is_ballistic != 0 || t.entry_omega_f.abs() > 0.045 {
                            (1.65 * sec_vx - 0.65 * v0x, 1.65 * sec_vy - 0.65 * v0y)
                        } else {
                            (sec_vx, sec_vy)
                        }
                    };

                    for k in gap_start..gap_end {
                        let tau = ((k - u0) as f32) / dt;
                        let tau2 = tau * tau;
                        let tau3 = tau2 * tau;

                        let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
                        let h10 = tau3 - 2.0 * tau2 + tau;
                        let h01 = -2.0 * tau3 + 3.0 * tau2;
                        let h11 = tau3 - tau2;

                        yx[k] = h00 * yx[u0] + h10 * (dt * v0x) + h01 * yx[u1] + h11 * (dt * v1x);
                        yy[k] = h00 * yy[u0] + h10 * (dt * v0y) + h01 * yy[u1] + h11 * (dt * v1y);
                        weff[k] = 0.90;
                    }
                } else if gap_start > 0 {
                    let u0 = gap_start - 1;
                    let v_tail_x = if u0 >= 1 && vis[u0 - 1] != 0 {
                        yx[u0] - yx[u0 - 1]
                    } else {
                        vx[u0]
                    };
                    let v_tail_y = if u0 >= 1 && vis[u0 - 1] != 0 {
                        yy[u0] - yy[u0 - 1]
                    } else {
                        vy[u0]
                    };
                    for k in gap_start..RING_LEN {
                        let dt = (k - u0) as f32;
                        let (cx_k, cy_k, _, _) = predict_curved_step(
                            yx[u0],
                            yy[u0],
                            v_tail_x,
                            v_tail_y,
                            t.entry_omega_f,
                            t.grav_ay_frame,
                            t.is_ballistic,
                            dt,
                        );
                        yx[k] = cx_k.clamp(tail_margin, (frame_w as f32) - tail_margin);
                        yy[k] = cy_k.clamp(tail_margin, (frame_h as f32) - tail_margin);
                        weff[k] = 0.60;
                    }
                } else if gap_end < RING_LEN {
                    // Check if pre-occlusion entry boundary scrolled off the left of the 12-frame ring:
                    // keep evaluating the exact Two-Point Cubic Hermite Bridge across full occlusions up to 28 frames!
                    let u1 = gap_end;
                    let can_bridge_from_entry = t.has_entry != 0
                        && frame_idx >= (RING_LEN - 1 - u1) as u32
                        && {
                            let f_re = (frame_idx - (RING_LEN - 1 - u1) as u32) as usize;
                            let f_en = t.entry_frame as usize;
                            f_en < f_re && (f_re - f_en) <= 28
                        };

                    if can_bridge_from_entry {
                        let f_re = (frame_idx - (RING_LEN - 1 - u1) as u32) as usize;
                        let f_en = t.entry_frame as usize;
                        let dt = (f_re - f_en) as f32;
                        let v0x = t.entry_vx;
                        let v0y = t.entry_vy;
                        let (v1x, v1y) = if u1 + 1 < RING_LEN && vis[u1 + 1] != 0 {
                            (yx[u1 + 1] - yx[u1], yy[u1 + 1] - yy[u1])
                        } else {
                            let sec_vx = (yx[u1] - entry_x_disp) / dt;
                            let sec_vy = (yy[u1] - entry_y_disp) / dt;
                            if t.is_ballistic != 0 || t.entry_omega_f.abs() > 0.045 {
                                (1.65 * sec_vx - 0.65 * v0x, 1.65 * sec_vy - 0.65 * v0y)
                            } else {
                                (sec_vx, sec_vy)
                            }
                        };
                        for k in 0..gap_end {
                            if frame_idx >= (RING_LEN - 1 - k) as u32 {
                                let f_k = (frame_idx - (RING_LEN - 1 - k) as u32) as usize;
                                if f_k > f_en && f_k < f_re {
                                    let tau = ((f_k - f_en) as f32) / dt;
                                    let tau2 = tau * tau;
                                    let tau3 = tau2 * tau;
                                    let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
                                    let h10 = tau3 - 2.0 * tau2 + tau;
                                    let h01 = -2.0 * tau3 + 3.0 * tau2;
                                    let h11 = tau3 - tau2;
                                    yx[k] = (h00 * entry_x_disp + h10 * (dt * v0x) + h01 * yx[u1] + h11 * (dt * v1x))
                                        .clamp(tail_margin, (frame_w as f32) - tail_margin);
                                    yy[k] = (h00 * entry_y_disp + h10 * (dt * v0y) + h01 * yy[u1] + h11 * (dt * v1y))
                                        .clamp(tail_margin, (frame_h as f32) - tail_margin);
                                    weff[k] = 0.90;
                                    continue;
                                }
                            }
                            let dt_l = (gap_end - k) as f32;
                            yx[k] = (yx[u1] - dt_l * v1x).clamp(tail_margin, (frame_w as f32) - tail_margin);
                            yy[k] = (yy[u1] - dt_l * v1y).clamp(tail_margin, (frame_h as f32) - tail_margin);
                            weff[k] = 0.65;
                        }
                    } else {
                        let (v_lead_x, v_lead_y) = if gap_end + 1 < RING_LEN && vis[gap_end + 1] != 0 {
                            (yx[gap_end + 1] - yx[gap_end], yy[gap_end + 1] - yy[gap_end])
                        } else {
                            (vx[gap_end], vy[gap_end])
                        };
                        for k in 0..gap_end {
                            let dt = (gap_end - k) as f32;
                            yx[k] = (yx[gap_end] - dt * v_lead_x).clamp(tail_margin, (frame_w as f32) - tail_margin);
                            yy[k] = (yy[gap_end] - dt * v_lead_y).clamp(tail_margin, (frame_h as f32) - tail_margin);
                            weff[k] = 0.65;
                        }
                    }
                }
            } else {
                u += 1;
            }
        }
    }

    // Build M^{-1} = diag(1 / w_i^2) and exact Gershgorin bound L = max_k sum_j |Q_{kj}| for Q = D2 M^{-1} D2^T
    let mut inv_m = [1.0f32; RING_LEN];
    for i in 0..RING_LEN {
        let w = weff[i].max(0.25);
        inv_m[i] = 1.0 / (w * w);
    }

    let mut q_diag = [0.0f32; 10];
    let mut l_gersh = 16.0f32;
    for k in 0..10usize {
        let d0 = inv_m[k];
        let d1 = inv_m[k + 1];
        let d2 = inv_m[k + 2];
        let q_kk = d0 + 4.0 * d1 + d2;
        q_diag[k] = q_kk;

        let mut row_abs = q_kk;
        if k >= 2 {
            row_abs += d0;
        }
        if k >= 1 {
            row_abs += 2.0 * (d0 + d1);
        }
        if k + 1 < 10 {
            row_abs += 2.0 * (d1 + d2);
        }
        if k + 2 < 10 {
            row_abs += d2;
        }
        if row_abs > l_gersh {
            l_gersh = row_abs;
        }
    }
    let step_dual = 1.0 / l_gersh;
    // Scale-covariant SO(2)-isotropic dual disk ||(ux_k, uy_k)||_2 <= s * lambda
    let lam = (lambda * scale_f).max(0.02 * scale_f);

    // Shift warm-started dual variables left by 1 frame and project onto Euclidean disk ||u_k||_2 <= lam
    let mut ux = [0.0f32; 10];
    let mut uy = [0.0f32; 10];
    if t.age > 1 {
        for k in 0..9usize {
            let (pux, puy) = project_dual_disk_2d(t.fista_ux[k + 1], t.fista_uy[k + 1], lam);
            ux[k] = pux;
            uy[k] = puy;
        }
    }

    let mut xk = [0.0f32; RING_LEN];
    let mut yk = [0.0f32; RING_LEN];
    let mut d2x = [0.0f32; 10];
    let mut d2y = [0.0f32; 10];
    let lam_pass1 = [lam; 10];

    // Pass 1: Uniform SO(2)-isotropic Dual FISTA (`2` BCM warm-start sweeps + `6` FGP iterations)
    let mut dual_gap = run_fista_2d_dual_stage(
        &yx,
        &yy,
        &inv_m,
        &q_diag,
        step_dual,
        &lam_pass1,
        6,
        &mut ux,
        &mut uy,
        &mut xk,
        &mut yk,
        &mut d2x,
        &mut d2y,
    );

    // 2-Pass IRL1 Knot-Sharpening: relax dual disk radii ONLY when Pass-1 reveals a strong genuine
    // impulse/burnout/peg/chute kink (`min_w <= 0.48`), avoiding Pass-2 overhead on smooth motion.
    let tau_irl1 = 2.2 * scale_f;
    let mut lam_k = [lam; 10];
    let mut min_w = 1.0f32;
    let mut sharp_knots = 0u8;
    let mut knot_mask = 0u16;
    for k in 0..10usize {
        let r_k = wasm_sqrtf(d2x[k] * d2x[k] + d2y[k] * d2y[k]);
        let w_k = (tau_irl1 / (tau_irl1 + r_k)).clamp(0.08, 1.0);
        lam_k[k] = lam * w_k;
        if w_k < min_w {
            min_w = w_k;
        }
        if w_k <= 0.48 {
            sharp_knots += 1;
            knot_mask |= 1u16 << k;
        }
    }

    if min_w <= 0.48 {
        dual_gap = run_fista_2d_dual_stage(
            &yx,
            &yy,
            &inv_m,
            &q_diag,
            step_dual,
            &lam_k,
            4,
            &mut ux,
            &mut uy,
            &mut xk,
            &mut yk,
            &mut d2x,
            &mut d2y,
        );
    }

    for u in 0..RING_LEN {
        t.fista_x[u] = xk[u];
        t.fista_y[u] = yk[u];
    }
    t.fista_zx = d2x;
    t.fista_zy = d2y;
    t.fista_ux = ux;
    t.fista_uy = uy;
    t.fista_duality_gap = dual_gap;
    t.fista_sharp_knots = sharp_knots;
    t.fista_knot_mask = knot_mask;

    // Mode A: Full-Segment Closed-Form Quadratic Least-Squares Parabola + Scene Gravity Prior
    // Solves the centered 2x2 normal system for y(tau) = c0 + c1*tau + c2*tau^2 (with c2 = 0.5*g*dt_nom^2)
    // and x(tau) = x_bar + vx*tau across all impulse-free observations in [seg_start..=11] using
    // per-slot presentation timestamps `t_rel_sec[q]` so single-frame `dt` jitter never rescales history.
    let dt10_11 = (t_rel_sec[11] - t_rel_sec[10]).max(1e-3);
    let dt9_10 = (t_rel_sec[10] - t_rel_sec[9]).max(1e-3);
    let vx_9_10 = (xk[10] - xk[9]) / dt9_10;
    let vx_10_11 = (xk[11] - xk[10]) / dt10_11;
    let v_phys_x = vx_10_11 + 0.5 * (vx_10_11 - vx_9_10);
    let v_phys_y = (yk[11] - yk[10]) / dt10_11;

    let mut apex_found = false;
    let mut ballistic_found = false;
    let mut pooled_ay_phys: Option<f32> = None;

    if vis_count >= 3 && vis[11] != 0 {
        let mut seg_start = 0usize;
        if last_discont_frame > 0 && frame_idx >= last_discont_frame {
            let age_d = (frame_idx - last_discont_frame) as usize;
            if age_d < 11 {
                seg_start = 11 - age_d;
            }
        }
        if t.has_struck != 0 && frame_idx >= t.last_strike_event_frame {
            let age_s = (frame_idx - t.last_strike_event_frame) as usize;
            if age_s < 11 {
                seg_start = seg_start.max(11 - age_s);
            }
        }
        let mut miss_run = 0usize;
        let mut scan_u = 11usize;
        while scan_u > seg_start {
            scan_u -= 1;
            if vis[scan_u] == 0 {
                miss_run += 1;
                if miss_run >= 2 {
                    seg_start = seg_start.max(scan_u + 2);
                    break;
                }
            } else {
                miss_run = 0;
                if scan_u + 2 <= 11 && vis[scan_u + 1] != 0 && vis[scan_u + 2] != 0 {
                    let dta = (t_rel_sec[scan_u + 1] - t_rel_sec[scan_u]).max(1e-3);
                    let dtb = (t_rel_sec[scan_u + 2] - t_rel_sec[scan_u + 1]).max(1e-3);
                    let dt_ref = 0.5 * (dta + dtb);
                    let dvx = ((yx[scan_u + 2] - yx[scan_u + 1]) / dtb
                        - (yx[scan_u + 1] - yx[scan_u]) / dta)
                        * dt_ref;
                    let dvy = ((yy[scan_u + 2] - yy[scan_u + 1]) / dtb
                        - (yy[scan_u + 1] - yy[scan_u]) / dta)
                        * dt_ref;
                    let dvy_ceil = (14.0 * scale_f) * (dt_ref * 42.0).clamp(1.0, 2.2);
                    if dvx.abs() > 4.6 * scale_f || dvy < -3.2 * scale_f || dvy > dvy_ceil {
                        seg_start = seg_start.max(scan_u + 1);
                        break;
                    }
                }
            }
        }

        let bounded_osc = t.freq_valid != 0
            && t.tremor_coherence > 0.45
            && t.max_excursion2 < (92.0 * scale_f) * (92.0 * scale_f);
        let rigid_tether = t.pivot_valid > 0
            || t.retrace_turnaround > 0
            || (t.freq_valid != 0 && t.tremor_coherence > 0.36);

        if !bounded_osc {
            let start_min = seg_start.min(9);
            let mut all_pts = [0usize; RING_LEN];
            let mut n_all = 0usize;
            for q in start_min..=11usize {
                if vis[q] != 0 {
                    all_pts[n_all] = q;
                    n_all += 1;
                }
            }
            if n_all >= 3 {
                #[derive(Clone, Copy)]
                struct QuadSuf {
                    n_pts: usize,
                    q_first: usize,
                    q_last: usize,
                    min_y: f32,
                    max_y: f32,
                    ut1: f64,
                    ut2: f64,
                    ut3: f64,
                    ut4: f64,
                    ux0: f64,
                    ux1: f64,
                    uy0: f64,
                    uy1: f64,
                    uy2: f64,
                }
                let zero_qs = QuadSuf {
                    n_pts: 0,
                    q_first: 11,
                    q_last: 0,
                    min_y: 1e9,
                    max_y: -1e9,
                    ut1: 0.0,
                    ut2: 0.0,
                    ut3: 0.0,
                    ut4: 0.0,
                    ux0: 0.0,
                    ux1: 0.0,
                    uy0: 0.0,
                    uy1: 0.0,
                    uy2: 0.0,
                };
                let mut sufs = [zero_qs; 10];
                let mut acc = zero_qs;
                let mut q = 11usize;
                loop {
                    if vis[q] != 0 {
                        let t = t_rel_sec[q] as f64;
                        let t2 = t * t;
                        let x = yx[q] as f64;
                        let y = yy[q];
                        let y64 = y as f64;
                        if acc.n_pts == 0 {
                            acc.q_last = q;
                        }
                        acc.n_pts += 1;
                        acc.q_first = q;
                        if y < acc.min_y {
                            acc.min_y = y;
                        }
                        if y > acc.max_y {
                            acc.max_y = y;
                        }
                        acc.ut1 += t;
                        acc.ut2 += t2;
                        acc.ut3 += t2 * t;
                        acc.ut4 += t2 * t2;
                        acc.ux0 += x;
                        acc.ux1 += t * x;
                        acc.uy0 += y64;
                        acc.uy1 += t * y64;
                        acc.uy2 += t2 * y64;
                    }
                    if q <= 9 {
                        sufs[q] = acc;
                    }
                    if q == start_min {
                        break;
                    }
                    q -= 1;
                }

                let mut pt_off = 0usize;
                for u0 in start_min..=9usize {
                    while pt_off < n_all && all_pts[pt_off] < u0 {
                        pt_off += 1;
                    }
                    let sf = sufs[u0];
                    let n_pts = sf.n_pts;
                    if n_pts >= 4 && sf.q_last >= sf.q_first + 3 {
                        let pts = &all_pts[pt_off..n_all];
                        let q_first = sf.q_first;
                        let q_last = sf.q_last;
                        let dt_nom = ((t_rel_sec[q_last] - t_rel_sec[q_first])
                            / ((q_last - q_first).max(1) as f32))
                            .clamp(0.002, 0.25);
                        let inv_dt_nom = 1.0 / dt_nom;
                        let inv_n = 1.0 / (n_pts as f32);
                        let inv_n64 = 1.0 / (n_pts as f64);
                        let k = inv_dt_nom as f64;
                        let k2 = k * k;
                        let tb_s = sf.ut1 * inv_n64;
                        let tb2 = tb_s * tb_s;
                        let xb64 = sf.ux0 * inv_n64;
                        let yb64 = sf.uy0 * inv_n64;
                        let m2 = sf.ut2 - sf.ut1 * tb_s;
                        let m3 = sf.ut3 - 3.0 * tb_s * sf.ut2 + 2.0 * tb2 * sf.ut1;
                        let m4 = sf.ut4
                            - 4.0 * tb_s * sf.ut3
                            + 6.0 * tb2 * sf.ut2
                            - 3.0 * (tb2 * tb_s) * sf.ut1;
                        let mx1 = sf.ux1 - tb_s * sf.ux0;
                        let my1 = sf.uy1 - tb_s * sf.uy0;
                        let my2 = (sf.uy2 - 2.0 * tb_s * sf.uy1 + tb2 * sf.uy0) - yb64 * m2;

                        let t_bar = (tb_s * k) as f32;
                        let x_bar = xb64 as f32;
                        let y_bar = yb64 as f32;
                        let s2 = (m2 * k2) as f32;
                        let s3 = (m3 * k2 * k) as f32;
                        let s4 = (m4 * k2 * k2) as f32;
                        let x1 = (mx1 * k) as f32;
                        let y1 = (my1 * k) as f32;
                        let y2_raw = (my2 * k2) as f32;
                        let min_y = sf.min_y;
                        let max_y = sf.max_y;

                        let s2_n = s2 * inv_n;
                        let s4c = s4 - s2 * s2_n;
                        let det = s2 * s4c - s3 * s3;
                        if s2 > 1e-3 && det > 1e-3 {
                            let inv_det = 1.0 / det;
                            let c1 = (s4c * y1 - s3 * y2_raw) * inv_det;
                            let c2 = (s2 * y2_raw - s3 * y1) * inv_det;
                            let c0 = y_bar - c2 * s2_n;

                            let a_ball_y = 2.0 * c2;
                            let v_ball_y = c1 - 2.0 * c2 * t_bar;
                            let v_ball_x = x1 / s2;
                            let x11_fit = x_bar - v_ball_x * t_bar;
                            let y11_fit = c0 - c1 * t_bar + c2 * t_bar * t_bar;

                            let vy_phys_b = v_ball_y * inv_dt_nom;
                            let vx_phys_b = v_ball_x * inv_dt_nom;
                            let ay_phys_b = a_ball_y * inv_dt_nom * inv_dt_nom;

                            if ((max_y - min_y) >= 6.0 * scale_f
                                || vy_phys_b.abs() >= 42.0 * scale_f)
                                && ay_phys_b > 110.0 * scale_f
                                && ay_phys_b < 16500.0 * scale_f
                            {
                                let mut sum_ey2 = 0.0f32;
                                let mut sum_ex2 = 0.0f32;
                                let mut max_res_y = 0.0f32;
                                let mut max_res_x = 0.0f32;
                                for i in 0..n_pts {
                                    let q = pts[i];
                                    let tau = t_rel_sec[q] * inv_dt_nom - t_bar;
                                    let y_fit = c0 + c1 * tau + c2 * tau * tau;
                                    let x_fit = x_bar + v_ball_x * tau;
                                    let ey = (yy[q] - y_fit).abs();
                                    let ex = (yx[q] - x_fit).abs();
                                    sum_ey2 += ey * ey;
                                    sum_ex2 += ex * ex;
                                    if ey > max_res_y {
                                        max_res_y = ey;
                                    }
                                    if ex > max_res_x {
                                        max_res_x = ex;
                                    }
                                }
                                let rmse_y = wasm_sqrtf(sum_ey2 * inv_n);
                                let rmse_x = wasm_sqrtf(sum_ex2 * inv_n);

                                if rmse_y <= 2.4 * scale_f
                                    && rmse_x <= 2.6 * scale_f
                                    && max_res_y <= 3.6 * scale_f
                                    && max_res_x <= 3.8 * scale_f
                                {
                                    let prev_g_phys = if t.grav_ay_px_s2 > 1.0 {
                                        t.grav_ay_px_s2
                                    } else {
                                        t.grav_ay_frame * inv_dt_nom * inv_dt_nom
                                    };
                                    let apex_spanned = !rigid_tether
                                        && n_pts >= 6
                                        && (yy[pts[0]] - min_y) >= 3.8 * scale_f
                                        && (yy[pts[n_pts - 1]] - min_y) >= 3.8 * scale_f
                                        && ay_phys_b > 760.0 * scale_f;
                                    let consistent_curv = if n_pts >= 5 {
                                        let m_i = (n_pts - 2) / 2;
                                        let p0 = pts[0];
                                        let pm = pts[m_i];
                                        let p1 = pts[n_pts - 1];
                                        let dt_e1 = (t_rel_sec[pm] - t_rel_sec[p0]).max(1e-3);
                                        let dt_e2 = (t_rel_sec[p1] - t_rel_sec[pm]).max(1e-3);
                                        let vy_e1 = (yy[pm] - yy[p0]) / dt_e1;
                                        let vy_e2 = (yy[p1] - yy[pm]) / dt_e2;
                                        (vy_e2 - vy_e1) / (0.5 * (dt_e1 + dt_e2)) > 220.0 * scale_f
                                    } else {
                                        false
                                    };
                                    let ascending_launch = !rigid_tether
                                        && vy_phys_b < -18.0 * scale_f
                                        && ay_phys_b > 760.0 * scale_f
                                        && (t.age <= 14
                                            || scene_gravity_conf >= 0.35
                                            || apex_spanned
                                            || consistent_curv);
                                    let continuing_arc = !rigid_tether
                                        && (t.is_ballistic != 0 || apex_spanned)
                                        && rmse_y <= 2.30 * scale_f
                                        && (apex_spanned
                                            || (prev_g_phys > 760.0 * scale_f
                                                && ay_phys_b > 0.55 * prev_g_phys
                                                && ay_phys_b < 1.55 * prev_g_phys));

                                    if allow_ballistic && (ascending_launch || continuing_arc) {
                                        ballistic_found = true;
                                        t.is_ballistic = 2;
                                        t.grav_ay_px_s2 = if t.grav_ay_px_s2 > 10.0 {
                                            0.68 * t.grav_ay_px_s2 + 0.32 * ay_phys_b
                                        } else {
                                            ay_phys_b
                                        };
                                        t.grav_ay_frame = t.grav_ay_px_s2 * dt_nom * dt_nom;
                                        if n_pts >= 6
                                            && ((ascending_launch
                                                && rmse_y <= 1.45 * scale_f
                                                && rmse_x <= 1.75 * scale_f)
                                                || (apex_spanned
                                                    && rmse_y <= 2.25 * scale_f
                                                    && rmse_x <= 2.45 * scale_f))
                                        {
                                            pooled_ay_phys = Some(ay_phys_b);
                                        }
                                    }

                                    let shallow_road_cruise = !ballistic_found
                                        && t.pivot_valid == 0
                                        && ay_phys_b < 360.0 * scale_f
                                        && vx_phys_b.abs() > 1.75 * vy_phys_b.abs();
                                    if vy_phys_b < -18.0 * scale_f && !shallow_road_cruise {
                                        let ay_used = if n_pts == 4
                                            && scene_gravity_conf >= 0.50
                                            && scene_gravity_px_s2 > 320.0 * scale_f
                                            && ay_phys_b > 0.45 * scene_gravity_px_s2
                                            && ay_phys_b < 1.85 * scene_gravity_px_s2
                                        {
                                            0.78 * ay_phys_b + 0.22 * scene_gravity_px_s2
                                        } else {
                                            ay_phys_b
                                        };
                                        let t_apex_sec = (-vy_phys_b / ay_used).min(0.75);
                                        let x0_ap = 0.5 * (xk[11] + x11_fit);
                                        let y0_ap = 0.5 * (yk[11] + y11_fit);
                                        t.apex_frames = t_apex_sec * inv_dt_nom;
                                        t.apex_x = (x0_ap + vx_phys_b * t_apex_sec)
                                            .clamp(8.0 * scale_f, (frame_w as f32) - 8.0 * scale_f);
                                        t.apex_y = (y0_ap
                                            + vy_phys_b * t_apex_sec
                                            + 0.5 * ay_used * t_apex_sec * t_apex_sec)
                                            .clamp(6.0 * scale_f, (frame_h as f32) - 6.0 * scale_f);
                                        apex_found = true;
                                        break;
                                    }
                                    if continuing_arc && vy_phys_b > 15.0 * scale_f {
                                        break;
                                    }
                                }
                            }
                        }
                    } else if u0 == 9
                        && n_pts == 3
                        && vis[9] != 0
                        && vis[10] != 0
                        && yy[9] > yy[10]
                        && yy[10] > yy[11]
                        && (yy[9] - yy[11]) >= 12.0 * scale_f
                    {
                        // Fast 3-point Newtonian ascent [9, 10, 11] using exact presentation timestamp intervals
                        let dt1 = (t_rel_sec[10] - t_rel_sec[9]).max(1e-3);
                        let dt2 = (t_rel_sec[11] - t_rel_sec[10]).max(1e-3);
                        let dt_m = 0.5 * (dt1 + dt2);
                        let vy1_s = (yy[10] - yy[9]) / dt1;
                        let vy2_s = (yy[11] - yy[10]) / dt2;
                        let vx1_s = (yx[10] - yx[9]) / dt1;
                        let vx2_s = (yx[11] - yx[10]) / dt2;
                        if (vx2_s - vx1_s).abs() * dt_m < 2.8 * scale_f {
                            let ay_phys_b = (vy2_s - vy1_s) / dt_m;
                            let vy_phys_b = vy2_s + 0.5 * dt2 * ay_phys_b;
                            let vx_phys_b = (yx[11] - yx[9]) / (dt1 + dt2);
                            let shallow_road_cruise = (rigid_tether || ay_phys_b <= 760.0 * scale_f)
                                && t.pivot_valid == 0
                                && ay_phys_b < 360.0 * scale_f
                                && vx_phys_b.abs() > 1.75 * vy_phys_b.abs();
                            if vy_phys_b < -18.0 * scale_f
                                && ay_phys_b > 110.0 * scale_f
                                && ay_phys_b < 16500.0 * scale_f
                                && !shallow_road_cruise
                            {
                                let ay_used = if scene_gravity_conf >= 0.50
                                    && scene_gravity_px_s2 > 110.0 * scale_f
                                    && ay_phys_b > 0.40 * scene_gravity_px_s2
                                    && ay_phys_b < 1.95 * scene_gravity_px_s2
                                {
                                    0.72 * ay_phys_b + 0.28 * scene_gravity_px_s2
                                } else {
                                    ay_phys_b
                                };
                                if allow_ballistic
                                    && !rigid_tether
                                    && ay_used > 760.0 * scale_f
                                    && (t.age <= 14 || scene_gravity_conf >= 0.35)
                                {
                                    ballistic_found = true;
                                    t.is_ballistic = 1;
                                    t.grav_ay_px_s2 = ay_used;
                                    t.grav_ay_frame = ay_used * dt_m * dt_m;
                                }
                                let t_apex_sec = (-vy_phys_b / ay_used).min(0.75);
                                t.apex_frames = t_apex_sec / dt_m;
                                t.apex_x = (xk[11] + vx_phys_b * t_apex_sec)
                                    .clamp(8.0 * scale_f, (frame_w as f32) - 8.0 * scale_f);
                                t.apex_y = (yk[11]
                                    + vy_phys_b * t_apex_sec
                                    + 0.5 * ay_used * t_apex_sec * t_apex_sec)
                                    .clamp(6.0 * scale_f, (frame_h as f32) - 6.0 * scale_f);
                                apex_found = true;
                                break;
                            }
                        }
                    }
                }
            }

            // Constrained-arc wide-chord turnaround fallback (e.g. Galileo's pendulum decelerating in both x & y)
            if !apex_found {
                for u0 in [2usize, 3, 4, 5, 6, 7, 8] {
                    if u0 < seg_start {
                        continue;
                    }
                    let span = 11 - u0;
                    let um = (u0 + 11) / 2;
                    let ok_chord = if span >= 4 {
                        vis[u0] != 0 && vis[um] != 0 && yy[u0] > yy[um] && yy[um] > yy[11]
                    } else {
                        vis[8] != 0 && vis[9] != 0 && vis[10] != 0 && yy[8] > yy[9] && yy[9] > yy[10] && yy[10] > yy[11]
                    };
                    if !ok_chord {
                        continue;
                    }
                    let (vy1_s, vy2_s, dt_mid_s, dt_last_s, vx1_s, vx2_s, dt_tot_s, min_rise) = if span >= 4 {
                        let d1_s = (t_rel_sec[um] - t_rel_sec[u0]).max(1e-3);
                        let d2_s = (t_rel_sec[11] - t_rel_sec[um]).max(1e-3);
                        (
                            (yy[um] - yy[u0]) / d1_s,
                            (yy[11] - yy[um]) / d2_s,
                            0.5 * (d1_s + d2_s),
                            0.5 * d2_s,
                            (yx[um] - yx[u0]) / d1_s,
                            (yx[11] - yx[um]) / d2_s,
                            d1_s + d2_s,
                            6.0 * scale_f,
                        )
                    } else {
                        let d1_s = (t_rel_sec[9] - t_rel_sec[8]).max(1e-3);
                        let d2_s = (t_rel_sec[11] - t_rel_sec[10]).max(1e-3);
                        let dm_s = (0.5 * (t_rel_sec[11] + t_rel_sec[10]) - 0.5 * (t_rel_sec[9] + t_rel_sec[8])).max(1e-3);
                        (
                            (yy[9] - yy[8]) / d1_s,
                            (yy[11] - yy[10]) / d2_s,
                            dm_s,
                            0.5 * d2_s,
                            (yx[9] - yx[8]) / d1_s,
                            (yx[11] - yx[10]) / d2_s,
                            (t_rel_sec[11] - t_rel_sec[8]).max(1e-3),
                            9.0 * scale_f,
                        )
                    };
                    let dt_nom_c = dt_tot_s / (span as f32);
                    // Circular pendulum arcs transfer velocity between vx and vy continuously near turnarounds.
                    let dvx_lim = if t.pivot_valid > 0 {
                        (7.5 * scale_f).max(4.2)
                    } else {
                        3.5 * scale_f
                    };
                    let min_rise_eff = if t.pivot_valid > 0 && span >= 4 {
                        (4.2 * scale_f).max(2.0)
                    } else {
                        min_rise
                    };
                    if (vx2_s - vx1_s).abs() * dt_nom_c < dvx_lim && (yy[u0] - yy[11]) >= min_rise_eff {
                        let ay_phys_b = (vy2_s - vy1_s) / dt_mid_s;
                        let vy_phys_b = vy2_s + dt_last_s * ay_phys_b;
                        let vx_phys_b = (yx[11] - yx[u0]) / dt_tot_s;
                        let shallow_road_cruise = t.pivot_valid == 0
                            && ay_phys_b < 360.0 * scale_f
                            && vx_phys_b.abs() > 1.75 * vy_phys_b.abs();
                        let vy_lim = if t.pivot_valid > 0 { -10.0 * scale_f } else { -18.0 * scale_f };
                        let ay_min = if t.pivot_valid > 0 { 28.0 * scale_f } else { 110.0 * scale_f };
                        if vy_phys_b < vy_lim
                            && ay_phys_b > ay_min
                            && ay_phys_b < 16500.0 * scale_f
                            && !shallow_road_cruise
                        {
                            let mut obs_in_arc = 0usize;
                            let mut max_ry = 0.0f32;
                            let mut max_rx = 0.0f32;
                            for q in u0..=11usize {
                                if vis[q] != 0 {
                                    obs_in_arc += 1;
                                    let du_s = t_rel_sec[q];
                                    let ey = (yy[q] - (yy[11] + vy_phys_b * du_s + 0.5 * ay_phys_b * du_s * du_s)).abs();
                                    let ex = (yx[q] - (yx[11] + vx_phys_b * du_s)).abs();
                                    if ey > max_ry { max_ry = ey; }
                                    if ex > max_rx { max_rx = ex; }
                                }
                            }
                            let max_ry_lim = if t.pivot_valid > 0 {
                                (4.4 * scale_f).max(2.8)
                            } else {
                                (3.6 * scale_f).max(2.3)
                            };
                            let max_rx_lim = if t.pivot_valid > 0 {
                                (7.0 * scale_f).max(4.4)
                            } else {
                                (3.8 * scale_f).max(2.7)
                            };
                            if obs_in_arc >= 4 && max_ry <= max_ry_lim && max_rx <= max_rx_lim {
                                let t_apex_sec = (-vy_phys_b / ay_phys_b).min(0.75);
                                t.apex_frames = t_apex_sec / dt_nom_c.max(1e-3);
                                t.apex_x = (xk[11] + vx_phys_b * t_apex_sec).clamp(8.0 * scale_f, (frame_w as f32) - 8.0 * scale_f);
                                t.apex_y = (yk[11] + vy_phys_b * t_apex_sec + 0.5 * ay_phys_b * t_apex_sec * t_apex_sec).clamp(6.0 * scale_f, (frame_h as f32) - 6.0 * scale_f);
                                apex_found = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if vis[11] != 0 && !ballistic_found {
        if t.is_ballistic >= 2 && scale_f > 1.5 {
            // 1-frame hysteresis in HD so a single blurred frame doesn't wipe ballistic state
            t.is_ballistic = 1;
        } else {
            t.grav_ay_frame = 0.0;
            t.grav_ay_px_s2 = 0.0;
            t.is_ballistic = 0;
        }
    } else if vis[11] == 0 && t.coasting > 14 {
        t.is_ballistic = 0;
        t.grav_ay_frame = 0.0;
        t.grav_ay_px_s2 = 0.0;
    }

    if !apex_found {
        t.apex_frames = 0.0;
        t.apex_x = xk[11];
        t.apex_y = yk[11];
    }

    // Mode B: Stimulus-to-onset reflex latency (strictly when stimulus armed) + spontaneous step settle duration
    let mut consumed_stimulus = false;
    let stim_disp2 = (8.0 * scale_f) * (8.0 * scale_f);
    let step_disp2 = (18.0 * scale_f) * (18.0 * scale_f);
    let settle_thresh = 38.0 * scale_f;
    if vis[11] != 0 {
        if stimulus_armed != 0 && frame_idx > stimulus_frame {
            let elapsed_stim = frame_idx - stimulus_frame;
            if elapsed_stim <= 90 {
                let sdx = (xk[11] - cam_off_x[11]) - t.stim_ref_x;
                let sdy = (yk[11] - cam_off_y[11]) - t.stim_ref_y;
                if sdx * sdx + sdy * sdy > stim_disp2 {
                    t.saccade_latency_ms =
                        frame_dt_span_sec(hist_t_sec, stimulus_frame, frame_idx, fallback_dt) * 1000.0;
                    consumed_stimulus = true;
                }
            }
        }

        let cur_sx11 = xk[11] - cam_off_x[11];
        let cur_sy11 = yk[11] - cam_off_y[11];
        let jump_dx = cur_sx11 - t.step_ref_x;
        let jump_dy = cur_sy11 - t.step_ref_y;
        let min_rearm = ((0.40 * fps) as u32).max(12);
        if (jump_dx * jump_dx + jump_dy * jump_dy) > step_disp2
            && frame_idx.saturating_sub(t.step_onset_frame) > min_rearm
        {
            t.step_onset_frame = frame_idx;
            t.step_ref_x = cur_sx11;
            t.step_ref_y = cur_sy11;
        } else if t.step_onset_frame > 0 {
            let settle_speed_px_s = v_phys_x.abs() + v_phys_y.abs();
            let elapsed_frames = frame_idx.saturating_sub(t.step_onset_frame);
            let max_settle_frames = ((0.50 * fps) as u32).max(15);
            if elapsed_frames >= 2 && elapsed_frames <= max_settle_frames && settle_speed_px_s < settle_thresh {
                t.step_settle_ms =
                    frame_dt_span_sec(hist_t_sec, t.step_onset_frame, frame_idx, fallback_dt) * 1000.0;
                t.step_onset_frame = 0;
            }
        }
    }
    (consumed_stimulus, pooled_ay_phys)
}

