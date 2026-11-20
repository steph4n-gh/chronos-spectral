#![no_std]
pub mod simd;
pub(crate) mod simd_ops;
#[allow(dead_code)]
pub(crate) mod physics_id;
pub(crate) mod soa_tracks;
pub(crate) mod continuum_physics;
pub(crate) mod bos;
#[allow(dead_code)]
pub(crate) mod vibrometry;
#[allow(dead_code)]
pub(crate) mod hodge;

use crate::simd::{
    f32x4_abs, f32x4_add, f32x4_div, f32x4_extract_lane, f32x4_floor, f32x4_ge, f32x4_gt,
    f32x4_lt, f32x4_mul, f32x4_nearest, f32x4_pmax, f32x4_pmin, f32x4_splat, f32x4_sqrt,
    f32x4_sub, f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_splat, f64x2_sqrt, f64x2_sub,
    i32x4, i32x4_bitmask, i32x4_shuffle, v128, v128_bitselect, v128_load, v128_store,
};
#[cfg(target_arch = "wasm32")] use core::panic::PanicInfo;
use core::ptr::addr_of_mut;

#[cfg(target_arch = "wasm32")]
extern "C" {
    static __heap_base: u8;
}

#[inline(always)]
fn linked_static_end_bytes() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        core::ptr::addr_of!(__heap_base) as usize as u32
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}
pub use simd_ops::padic_ultrametric_val;
use simd_ops::{
    advect_bg_subpixel_simd128, build_tpl_ic_lk_cache, cam_pyr_tile_4x6x8_simd128,
    compute_motion_grid_and_heal_bg_simd128, demean_mad64_simd128,
    eca_and_motion_grid_pan_fused_simd128, eca_and_motion_grid_stat_fused_simd128,
    eca_subspace_project_masked_simd128, estimate_pan_exposure_symmetric_simd128,
    eval_cam_coarse_3x3_simd128, eval_cam_coarse_tile_simd128, eval_patch_raw_sad_rgba_simd128,
    eval_win_patch_sad_p2_packed_simd128, eval_win_patch_sad_simd128,
    eval_winner_patch_hd_simd128, fit_affine_rgb_trimmed_simd128,
    haar_morton_signature_hyst_simd128, haar_morton_signature_scaled_simd128,
    haar_morton_signature_simd128, has_opposite_chroma_pole,
    has_opposite_chroma_pole_patch_simd128, is_pan_or_exp_active, load_pooled4_f32,
    pack_win_p2_phases_simd128, padic_distance_from_val, pan_streak_and_eca_fused_subpx_simd128,
    patch16x16_shi_tomasi_simd128, patch_eca_core_inner_max_simd128,
    patch_grad_tensor_span15_simd128, patch_grad_tensor_span_simd128,
    pick_textured_subpatch_5band_hd_simd128, pick_textured_subpatch_step16_hd_simd128,
    pool_plane_rows_to_160x90_simd128,
    pool_plane_to_160x90_simd128, refine_subpixel_ic_lk_cached,
    render_eulerian_pip_fast_simd128, reorder_patch64_eo_simd128, rgba_roi_to_planar_simd128,
    rgba_tiles_run_to_chroma_simd128, rgba_to_planar_and_pool160_simd128, sad8x8_pyr160_5x5_simd128,
    sad8x8_pyr160_cutoff_simd128, sad_patch_zm_full_precomp_cutoff_simd128,
    sad_patch_zm_score_simd128, sample_luma_patch64_p2_simd128, tpl_full_stats_simd128,
    subpx_patch_gain_norm_sad_p2_18_simd128,
    subpx_patch_gain_norm_sad_simd128, TplIcLkCache, unpack_rgba_luma_win18_p2_simd128,
    unpack_rgba_luma_win_p2_simd128, unpack_rgba_luma_window_simd128,
    update_stat_streak_and_mask_pan_simd128, weighted_anchor_zm_l1_simd128,
};

#[inline(always)]
pub(crate) fn wasm_sqrtf(x: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
}

#[inline(always)]
pub(crate) fn wasm_floorf(x: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(x)))
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn wasm_nearestf(x: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_nearest(f32x4_splat(x)))
}

#[inline(always)]
pub(crate) fn wasm_sqrt_f64(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

#[inline(always)]
fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
}

#[inline(always)]
fn hypot_f64(a: f64, b: f64) -> f64 {
    let aa = abs_f64(a);
    let bb = abs_f64(b);
    let (hi, lo) = if aa >= bb { (aa, bb) } else { (bb, aa) };
    if hi == 0.0 {
        return 0.0;
    }
    let u = lo / hi;
    hi * wasm_sqrt_f64(1.0 + u * u)
}

#[cfg(target_arch = "wasm32")] #[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

pub const SD_FRAME_W: usize = 320;
pub const SD_FRAME_H: usize = 180;
pub const MAX_FRAME_W: usize = 640;
pub const MAX_FRAME_H: usize = 360;
pub const MAX_FRAME_PIXELS: usize = MAX_FRAME_W * MAX_FRAME_H; // 230,400
pub const MAX_FRAME_RGBA_BYTES: usize = MAX_FRAME_PIXELS * 4; // 921,600
pub const RING_LEN: usize = 12;
pub const MAX_TARGETS: usize = 16;
pub const DENSE_SWARM_HERO_CAP: usize = 3;
pub const GRAPH_NODES: usize = 32;
pub const MAX_KRYLOV: usize = GRAPH_NODES - 1;
pub const MAX_GRID_W: usize = 40;
pub const MAX_GRID_H: usize = 22;
pub const MAX_GRID_CELLS: usize = MAX_GRID_W * MAX_GRID_H; // 880
pub const MAX_PIP_DIM: usize = 64;
pub const MAX_PIP_PIXELS: usize = MAX_PIP_DIM * MAX_PIP_DIM; // 4,096
pub const PYR_W: usize = 160;
pub const PYR_H: usize = 90;
pub const PYR_PIXELS: usize = PYR_W * PYR_H; // 14,400
pub const MAX_ROI_DIM: usize = 96;
pub const MAX_ROI_PIXELS: usize = MAX_ROI_DIM * MAX_ROI_DIM; // 9,216
pub const MAX_HYPS: usize = 48;
pub const ASSIGN_COLS: usize = MAX_HYPS + MAX_TARGETS; // 64
pub const HIST_RING_LEN: usize = 64;
pub const HIST_MASK: usize = HIST_RING_LEN - 1;
pub const ARC_HIST_LEN: usize = HIST_RING_LEN;
pub const MAX_DORMANT: usize = 32;
pub const DORMANT_MAX_AGE: u32 = 1200;
pub const TELEM_PEAKS_START: usize = 928;
pub const TELEM_EXCUR_START: usize = 992;
pub const TELEM_KINETIC_START: usize = 1120;
pub const TELEM_GLOBAL_START: usize = 1248;
pub const TELEM_PERSIST_START: usize = 1264;
pub const TELEM_PHYSICS_START: usize = 1280;
pub const TELEM_ENSEMBLE_START: usize = 1536;
pub const TELEMETRY_LEN: usize = 1568;

#[inline(always)]
fn mode_dims(res_mode: u32) -> (usize, usize, usize, usize, usize, usize, f32) {
    if res_mode == 1 {
        (640, 360, 40, 22, 2, 64, 2.0)
    } else {
        (SD_FRAME_W, SD_FRAME_H, 20, 11, 1, 32, 1.0)
    }
}

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const HALF_PI: f32 = 1.57079632679489661923;

// ============================================================================
// Zero-dependency #![no_std] minimax math kernels (zero `env` imports)
// Tolerance bounds:
// - fast_atan2f: <= 2.04e-4 rad max error
// - fast_log10f: <= 6.5e-5 log10 units (<= 0.00065 dB), C^0 continuous across 2^k
// - fast_sinf:   <= 2.0e-6 rad max error on [-pi, pi]
// ============================================================================

#[inline(always)]
pub(crate) fn wrap_pi(mut a: f32) -> f32 {
    if !a.is_finite() {
        return 0.0;
    }
    if a > 32.0 || a < -32.0 {
        let turns = wasm_floorf((a + PI) * (1.0 / TWO_PI));
        a -= turns * TWO_PI;
    }
    while a > PI {
        a -= TWO_PI;
    }
    while a < -PI {
        a += TWO_PI;
    }
    a
}

#[inline(always)]
pub(crate) fn fast_sinf(x: f32) -> f32 {
    let mut a = wrap_pi(x);
    if a > HALF_PI {
        a = PI - a;
    } else if a < -HALF_PI {
        a = -PI - a;
    }
    let a2 = a * a;
    a * (1.0 + a2 * (-0.16666657 + a2 * (0.008333025 - a2 * 0.000198074)))
}

#[inline(always)]
pub(crate) fn fast_cosf(x: f32) -> f32 {
    fast_sinf(x + HALF_PI)
}

/// Trigonometric Constant-Turn-Rate & Velocity (CTRV) + gravitational free-flight step predictor.
/// Smoothly reduces to exact straight-line chord `(x0 + vx0 * dt_f, y0 + vy0 * dt_f)` as `omega_f -> 0`.
#[inline]
fn predict_curved_step(
    x0: f32,
    y0: f32,
    vx0: f32,
    vy0: f32,
    omega_f: f32,
    ay_grav_f: f32,
    is_ballistic: u8,
    dt_f: f32,
) -> (f32, f32, f32, f32) {
    if is_ballistic != 0 && ay_grav_f > 0.02 {
        let dt_c = dt_f.clamp(-12.0, 12.0);
        let px = x0 + vx0 * dt_c;
        let py = y0 + vy0 * dt_c + 0.5 * ay_grav_f * dt_c * dt_c;
        let evy = vy0 + ay_grav_f * dt_c;
        (px, py, vx0, evy)
    } else if omega_f.abs() > 0.045 && omega_f.abs() < 0.42 {
        let psi = (omega_f * dt_f).clamp(-1.25, 1.25);
        let (sinc, cosc) = if psi.abs() < 1e-2 {
            let p2 = psi * psi;
            (
                1.0 - p2 * (1.0 / 6.0) * (1.0 - p2 * (1.0 / 20.0)),
                0.5 * psi * (1.0 - p2 * (1.0 / 12.0)),
            )
        } else {
            (fast_sinf(psi) / psi, (1.0 - fast_cosf(psi)) / psi)
        };
        let dx = dt_f * (vx0 * sinc - vy0 * cosc);
        let dy = dt_f * (vx0 * cosc + vy0 * sinc);
        let cp = fast_cosf(psi);
        let sp = fast_sinf(psi);
        let evx = vx0 * cp - vy0 * sp;
        let evy = vx0 * sp + vy0 * cp;
        (x0 + dx, y0 + dy, evx, evy)
    } else {
        (x0 + vx0 * dt_f, y0 + vy0 * dt_f, vx0, vy0)
    }
}

#[inline(always)]
fn frame_dt_span_sec(
    hist_t_sec: &[f64; HIST_RING_LEN],
    f_a: u32,
    f_b: u32,
    fallback_dt: f32,
) -> f32 {
    let span = f_b.wrapping_sub(f_a);
    if span > 0 && span < (HIST_RING_LEN as u32) {
        let dt =
            (hist_t_sec[(f_b as usize) & HIST_MASK] - hist_t_sec[(f_a as usize) & HIST_MASK])
                as f32;
        if dt.is_finite() && dt > 1e-4 {
            return dt;
        }
    }
    let fallback_frames = if span > 0 && span < 0x8000_0000 { span } else { 1 };
    (fallback_frames as f32) * fallback_dt.max(1e-3)
}

#[inline(always)]
fn ring_slot_back_cursor(cursor: u64, offset: usize) -> usize {
    let s = (cursor % (RING_LEN as u64)) as usize;
    (s + RING_LEN - (offset % RING_LEN)) % RING_LEN
}

#[inline(always)]
pub(crate) fn fast_atan2f(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    let ax = x.abs();
    let ay = y.abs();
    let swap = ay > ax;
    let u = if swap { ax / (ay + 1e-20) } else { ay / (ax + 1e-20) };
    let u2 = u * u;
    let p = (((-0.0464964749 * u2 + 0.15931422) * u2 - 0.327622764) * u2 * u) + u;
    let mut ang = if swap { HALF_PI - p } else { p };
    if x < 0.0 {
        ang = PI - ang;
    }
    ang.copysign(y)
}

#[inline(always)]
fn fast_log2f(x: f32) -> f32 {
    let v = x.max(1e-12);
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000) - 1.0;
    // Endpoint-constrained minimax quartic for log2(1 + m) on [0, 1]: P(0) = 0, P(1) = 1.00000000
    let log2_m = (((-0.08231790 * m + 0.32402705) * m - 0.68169811) * m + 1.43998896) * m;
    exp as f32 + log2_m
}

#[inline(always)]
pub(crate) fn fast_log10f(x: f32) -> f32 {
    fast_log2f(x) * 0.30102999566
}

#[inline(always)]
pub(crate) fn fast_lnf(x: f32) -> f32 {
    fast_log2f(x) * 0.69314718056
}

#[inline(always)]
pub(crate) fn fast_expf(x: f32) -> f32 {
    let xc = x.clamp(-20.0, 20.0);
    let t = xc * 1.44269504089;
    let k = wasm_floorf(t);
    let f = t - k;
    let r = f * 0.69314718056;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * 0.04166667)));
    let ki = (k as i32 + 127).clamp(1, 254) as u32;
    poly * f32::from_bits(ki << 23)
}

const fn fast_expf_const(x: f32) -> f32 {
    let xc = if x < -20.0 { -20.0 } else if x > 20.0 { 20.0 } else { x };
    let t = xc * 1.44269504089;
    let ti = t as i32;
    let k = if t < (ti as f32) { (ti - 1) as f32 } else { ti as f32 };
    let f = t - k;
    let r = f * 0.69314718056;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * 0.04166667)));
    let ki_raw = (k as i32) + 127;
    let ki = if ki_raw < 1 { 1u32 } else if ki_raw > 254 { 254u32 } else { ki_raw as u32 };
    poly * f32::from_bits(ki << 23)
}

static TPL_RAD_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut ry = 0usize;
    while ry < 16 {
        let dy = (ry as f32) - 7.5;
        let mut rx = 0usize;
        while rx < 16 {
            let dx = (rx as f32) - 7.5;
            let r2 = (dx * dx + dy * dy) * (1.0 / 26.0);
            lut[ry * 16 + rx] = fast_expf_const(-r2);
            rx += 1;
        }
        ry += 1;
    }
    lut
};

/// Branchless L1 soft-thresholding operator emitting wasm32 `f32.abs`, `f32.max`, `f32.copysign`.
/// Also implements exact box projection onto [-lambda, lambda] via Moreau's identity:
///   clip(v, -lambda, lambda) = v - soft_threshold_branchless(v, lambda)
#[inline(always)]
pub fn soft_threshold_branchless(d2: f32, thresh: f32) -> f32 {
    let mag = (0.0f32).max(d2.abs() - thresh);
    mag.copysign(d2)
}

/// Exact SO(2)-isotropic Euclidean disk projection onto `||(ux, uy)||_2 <= lam` via radial Moreau identity.
#[inline(always)]
fn project_dual_disk_2d(ux: f32, uy: f32, lam: f32) -> (f32, f32) {
    let r2 = ux * ux + uy * uy;
    if lam >= 0.0 && r2 <= lam * lam {
        return (ux, uy);
    }
    let r = wasm_sqrtf(r2);
    let excess = soft_threshold_branchless(r, lam);
    if excess > 0.0 {
        let scale = (r - excess) / r;
        (ux * scale, uy * scale)
    } else {
        (ux, uy)
    }
}

// ============================================================================
// Contiguous Zero-Allocation Arena Structures (<= 64 Pages / 4,194,304 Bytes)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CandidatePeak {
    pub x: f32,
    pub y: f32,
    pub energy: f32,
    pub sig: u32,
    pub valid: u8,
    pub comp_id: u8,
    pub _pad: [u8; 2],
}

#[repr(C, align(16))]
pub struct TargetTrack {
    // 16x16 Adaptive Foreground-Weighted Luma/Chroma Templates + Immutable P-N Birth Anchor
    pub tpl_luma: [f32; 256],
    pub tpl_rg: [f32; 256],
    pub tpl_by: [f32; 256],
    pub anchor_luma: [f32; 256],
    pub anchor_rg: [f32; 256],
    pub anchor_by: [f32; 256],
    pub tpl_weight: [f32; 256],

    // Pooled 8x8 Coarse Pyramid Templates (`160x90` SIMD128 Pyramid)
    pub tpl8_luma: [f32; 64],
    pub tpl8_rg: [f32; 64],
    pub tpl8_by: [f32; 64],
    pub tpl8_w: [f32; 64],

    // 12-frame circular trajectory ring buffer
    pub ring_raw_x: [f32; RING_LEN],
    pub ring_raw_y: [f32; RING_LEN],
    pub ring_vx: [f32; RING_LEN],
    pub ring_vy: [f32; RING_LEN],
    pub ring_sig: [u32; RING_LEN],
    // ring_vis states: 0 = occluded coast, 1 = causal or birth lock, 2 = non-causal gap-recovered lock
    pub ring_vis: [u8; RING_LEN],
    pub last_ode_disc_frame: u32,

    // Warm-started Dual Projected-FISTA primal X, second-difference D2*X, and dual u in [-lambda, lambda]^10
    pub fista_x: [f32; RING_LEN],
    pub fista_y: [f32; RING_LEN],
    pub fista_zx: [f32; 10],
    pub fista_zy: [f32; 10],
    pub fista_ux: [f32; 10],
    pub fista_uy: [f32; 10],
    pub fista_duality_gap: f32,
    pub fista_sharp_knots: u8,
    pub fista_knot_mask: u16,
    pub ode_cv_streak: u8,

    // Non-causally revised 2D neuromuscular tremor circular history (for 15-tap Hilbert filter)
    pub tremor_hist_x: [f32; HIST_RING_LEN],
    pub tremor_hist_y: [f32; HIST_RING_LEN],
    pub tremor_valid: [u8; HIST_RING_LEN],

    // Slowly-varying 2D tremor covariance & sign-continuous principal axis (zero blind diagonals)
    pub cov_xx: f32,
    pub cov_xy: f32,
    pub cov_yy: f32,
    pub axis_x: f32,
    pub axis_y: f32,

    // Kinematics, lifecycle (`Tentative -> Confirmed`), spectral & 2-adic identity invariants
    pub active: u8,
    pub confirmed: u8,
    pub just_confirmed: u8,
    pub hit_mask_8: u8,
    pub static_streak: u8,
    pub bridged_at_disp: u8,
    pub owner_slot: u8,
    pub mismatch_streak: u8,
    pub phase_valid: u8,
    pub prev_phase_valid: u8,
    pub reemerged_last: u8,
    pub has_struck: u8,
    pub strike_fired: u8,
    pub strike_zone: u8,
    pub proximity_mask: u16,
    pub orig_birth_frame: u32,
    pub sustained_10s_counted: u8,
    pub sustained_60s_counted: u8,
    pub co_dup_streak: u8,
    pub has_entry: u8,
    pub is_ballistic: u8,
    pub pivot_valid: u8,
    pub turn_streak: u8,
    pub retrace_dirs: u8,
    pub dormant_reid: u8,
    pub grav_arc_counted: u8,

    pub track_id: u32,
    pub birth_frame: u32,
    pub anchor_sig: u32,
    pub curr_sig: u32,
    pub last_padic_val: u32,
    pub age: u32,
    pub coasting: u32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub acc_x: f32,
    pub acc_y: f32,
    pub prev_inst_vx: f32,
    pub prev_inst_vy: f32,
    pub pred_cx: f32,
    pub pred_cy: f32,
    pub entry_x: f32,
    pub entry_y: f32,
    pub entry_vx: f32,
    pub entry_vy: f32,
    pub entry_ax: f32,
    pub entry_ay: f32,
    pub entry_frame: u32,
    pub birth_x: f32,
    pub birth_y: f32,
    pub max_excursion2: f32,
    pub vis_eca_ema: f32,
    pub disp_fista_x: f32,
    pub disp_fista_y: f32,
    pub disp_raw_x: f32,
    pub disp_raw_y: f32,
    pub disp_vis: f32,

    // Analytic tremor & present-time phase prediction state
    pub tremor_r: f32,
    pub tremor_h: f32,
    pub tremor_phase: f32,
    pub prev_tremor_phase: f32,
    pub pred_phase_now: f32,
    pub prev_pred_phase_now: f32,
    pub tremor_amp: f32,
    pub tremor_omega_ema: f32,
    pub tremor_freq_hz: f32,
    pub tremor_coherence: f32,
    pub osc_mean_x: f32,
    pub osc_mean_y: f32,

    // Mode A/B/C Biophysical Lab outputs
    pub apex_x: f32,
    pub apex_y: f32,
    pub apex_frames: f32,
    pub l1_impulse_mag: f32,
    pub last_strike_event_frame: u32,
    pub strike_lead_frames: f32,
    pub strike_vel: f32,
    pub strike_x: f32,
    pub strike_y: f32,
    pub step_ref_x: f32,
    pub step_ref_y: f32,
    pub stim_ref_x: f32,
    pub stim_ref_y: f32,
    pub step_onset_frame: u32,
    pub saccade_latency_ms: f32,
    pub step_settle_ms: f32,

    // Trigonometric, curved-arc CTRV, ballistic gravity & collision angle kinematics
    pub arc_raw_x: [f32; ARC_HIST_LEN],
    pub arc_raw_y: [f32; ARC_HIST_LEN],
    pub arc_vis: [u8; ARC_HIST_LEN],
    pub cand_piv_streak: u8,
    pub retrace_turnaround: u8,
    pub freq_valid: u8,
    pub osc_cross_count: u8,
    pub osc_prev_sign: i8,
    pub trem_cross_count: u8,
    pub trem_prev_sign: i8,
    pub osc_regime: u8,
    pub cand_piv_x: f32,
    pub cand_piv_y: f32,
    pub cand_piv_r: f32,
    pub last_chord_dx: f32,
    pub last_chord_dy: f32,
    pub osc_last_cross_t: f64,
    pub osc_half_period_ema: f32,
    pub osc_period_jit_ema: f32,
    pub trem_last_cross_t: f64,
    pub trem_half_period_ema: f32,
    pub trem_period_jit_ema: f32,
    pub trem_prev_q: f32,
    pub trem_max_q: f32,
    pub osc_ext_min_x: f32,
    pub osc_ext_max_x: f32,
    pub pend_theta_mean: f32,
    pub pend_prev_theta: f32,
    pub pend_prev_q: f32,
    pub pend_max_q: f32,
    pub ang_vel_rad_f: f32,
    pub entry_omega_f: f32,
    pub grav_ay_frame: f32,
    pub grav_ay_px_s2: f32,
    pub heading_deg: f32,
    pub speed_px_s: f32,
    pub ang_vel_deg_s: f32,
    pub strike_deflect_deg: f32,
    pub strike_restitution: f32,
    pub strike_in_deg: f32,
    pub strike_out_vx: f32,
    pub strike_out_vy: f32,
    pub pivot_x: f32,
    pub pivot_y: f32,
    pub pivot_r: f32,
    pub reacq_count: u16,
    pub tpl_grad_dirty: u8,
    pub wave1_auto_owned: u8,
    pub anchor_mu_a: f32,
    pub anchor_std_a: f32,
    pub phys_stamp_tid: u32,
}

/// 4-Mode Interacting Multiple Model (IMM) state per active track:
/// Mode 0 = CV, Mode 1 = Ballistic+Drag, Mode 2 = CTRV, Mode 3 = Oscillatory/Pendulum.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImmTrack {
    pub active: u8,
    pub pivot_valid: u8,
    pub mode_dom: u8,
    pub _pad_imm: u8,
    pub mu: [f32; 4],
    pub x: [f32; 4],
    pub y: [f32; 4],
    pub vx: [f32; 4],
    pub vy: [f32; 4],
    pub pos_var: [f32; 4],
    pub omega_turn: f32,
    pub g_bal_s2: f32,
    pub beta_drag: f32,
    pub eq_x: f32,
    pub eq_y: f32,
    pub omega_osc: f32,
    pub piv_x: f32,
    pub piv_y: f32,
    pub piv_r: f32,
    pub pred_x: f32,
    pub pred_y: f32,
    pub pred_vx: f32,
    pub pred_vy: f32,
}

impl ImmTrack {
    pub const fn zeroed() -> Self {
        Self {
            active: 0,
            pivot_valid: 0,
            mode_dom: 0,
            _pad_imm: 0,
            mu: [0.25; 4],
            x: [0.0; 4],
            y: [0.0; 4],
            vx: [0.0; 4],
            vy: [0.0; 4],
            pos_var: [4.0; 4],
            omega_turn: 0.0,
            g_bal_s2: 980.0,
            beta_drag: 0.0,
            eq_x: 0.0,
            eq_y: 0.0,
            omega_osc: 3.1415927,
            piv_x: 0.0,
            piv_y: 0.0,
            piv_r: 0.0,
            pred_x: 0.0,
            pred_y: 0.0,
            pred_vx: 0.0,
            pred_vy: 0.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::zeroed();
        self.active = 0;
    }
}

/// Recently-lost confirmed track archive for 2-adic Haar-Morton + kinematic re-identification.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DormantTrack {
    pub active: u8,
    pub is_ballistic: u8,
    pub sustained_10s_counted: u8,
    pub sustained_60s_counted: u8,
    pub reacq_count: u16,
    pub _pad_d: u16,
    pub track_id: u32,
    pub anchor_sig: u32,
    pub curr_sig: u32,
    pub lost_frame: u32,
    pub orig_birth_frame: u32,
    pub last_x: f32,
    pub last_y: f32,
    pub vx: f32,
    pub vy: f32,
    pub grav_ay_frame: f32,
    pub age: u32,
    pub anchor_luma: [f32; 256],
    pub anchor_rg: [f32; 256],
    pub anchor_by: [f32; 256],
    pub tpl_weight: [f32; 256],
}

/// Constant-Velocity 2x2 State-Covariance Kinematic Kalman Baseline (`[x, v]` with `p_xx, p_xv, p_vv`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CausalKalmanTrack {
    pub active: u8,
    pub current_owner: u8,
    pub mismatch_streak: u8,
    pub _pad: u8,
    pub anchor_sig: u32,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub p_xx: f32,
    pub p_xv: f32,
    pub p_vv: f32,
    pub coasting: u32,
    pub ring_x: [f32; RING_LEN],
    pub ring_y: [f32; RING_LEN],
}

#[repr(C, align(16))]
pub struct LanczosScratch {
    pub sub_l: [f64; GRAPH_NODES * GRAPH_NODES],
    pub q: [f64; MAX_KRYLOV * GRAPH_NODES],
    pub z: [f64; MAX_KRYLOV * GRAPH_NODES],
    pub alpha: [f64; MAX_KRYLOV],
    pub beta: [f64; MAX_KRYLOV],
    pub d_diag: [f64; MAX_KRYLOV],
    pub e_sub: [f64; GRAPH_NODES],
    pub w: [f64; GRAPH_NODES],
    pub eig_vec: [f64; GRAPH_NODES],
    pub prev_track: [f64; MAX_TARGETS],
    pub act_idx: [usize; GRAPH_NODES],
    pub last_lambda2: f64,
    pub last_n: usize,
}

pub const WAVE1_ABI_VERSION: u32 = 1;
pub const WAVE1_TELEM_LEN: usize = 192;
pub const INGEST_CTRL_LEN: usize = 12;
pub const INGEST_RING_STRIDE_WORDS: u32 = 24;

pub const INGEST_FLAG_COMMITTED: u32 = 1 << 0;
pub const INGEST_FLAG_PTS_KNOWN: u32 = 1 << 1;
pub const INGEST_FLAG_CAPTURE_AGE: u32 = 1 << 2;
pub const INGEST_FLAG_HANDOFF_AGE: u32 = 1 << 3;
pub const INGEST_FLAG_GEOM_KNOWN: u32 = 1 << 4;

pub const FRAME_GAP_SEQ: u32 = 1 << 0;
pub const FRAME_GAP_NONINC_PTS: u32 = 1 << 1;
pub const FRAME_GAP_EXCESS_DT: u32 = 1 << 2;
pub const FRAME_GAP_SEEK_RESTART: u32 = 1 << 3;
pub const FRAME_GAP_GEOM_CHANGE: u32 = 1 << 4;
pub const FRAME_GAP_SCENE_CUT: u32 = 1 << 5;
pub const FRAME_GAP_WRAP: u32 = 1 << 6;

pub const INGEST_STATUS_EMPTY: u32 = 0;
pub const INGEST_STATUS_OK: u32 = 1;
pub const INGEST_STATUS_INVALID: u32 = 2;
pub const INGEST_STATUS_DUP: u32 = 3;
pub const INGEST_STATUS_PHASE_ERR: u32 = 4;

pub const STAMP_VALID: u32 = 1 << 0;
pub const STAMP_HELD: u32 = 1 << 1;
pub const WAVE1_PRODUCT_STAMPS_START: usize = 16;
pub const WAVE1_HERO_STAMPS_START: usize = 64;
pub const STAMP_STRIDE: usize = 8;

pub const PROD_MICRO: usize = 0;
pub const PROD_GRAD: usize = 1;
pub const PROD_FTLE: usize = 2;
pub const PROD_MODAL_FIELD: usize = 3;
pub const PROD_HERO_PHYSICS: usize = 4;
pub const PROD_BOS: usize = 5;

pub const WAVE2_ABI_VERSION: u32 = 1;
pub const WAVE2_TELEM_LEN: usize = 64;
pub const WAVE2_PRODUCT_STAMPS_START: usize = 16;
pub const PROD_W2_VIBRO: usize = 0;
pub const PROD_W2_HODGE: usize = 1;
pub const PROD_W2_CONSERVATION: usize = 2;

const CTRL_PENDING_VALID: usize = 0;
const CTRL_TICK_PHASE: usize = 1;
const CTRL_NOM_DT_US: usize = 2;
const CTRL_DUP_GUARD_ACTIVE: usize = 3;
const CTRL_HERO_PHYS_RAN: usize = 4;
const CTRL_HERO_PHYS_SLOT_MASK: usize = 5;
const CTRL_AUTO_HERO_MASK: usize = 6;
const CTRL_AUTO_HERO_MICRO_BASE: usize = 7; // 7, 8, 9 hold micro_tid for up to 3 auto-promoted Heroes
const CTRL_AUTO_HERO_SLOTS: usize = 10;     // packed bytes: (slot_m + 1) per binding k in 0..3, 0 = empty
const CTRL_PREV_EXCESS_DT_US: usize = 11;   // previous sustained slow-rate dt_us candidate for nominal relearning

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct IngestSlotMeta {
    pub flags: u32,
    pub source_epoch: u32,
    pub source_seq: u32,
    pub engine_frame_idx: u32,
    pub t_src_us_lo: u32,
    pub t_src_us_hi: u32,
    pub age_ms_bits: u32,
    pub src_w: u32,
    pub src_h: u32,
    pub crop_x0: u32,
    pub crop_y0: u32,
    pub crop_w: u32,
    pub crop_h: u32,
    pub orientation: u32,
    pub scan_dir: u32,
    pub engine_w: u32,
    pub engine_h: u32,
    pub dst_x: u32,
    pub dst_y: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub frame_gap_flag: u32,
    pub native_1to1_active: u32,
    pub continuity_epoch: u32,
}

impl IngestSlotMeta {
    #[inline(always)]
    const fn zeroed() -> Self {
        Self {
            flags: 0,
            source_epoch: 0,
            source_seq: 0,
            engine_frame_idx: 0,
            t_src_us_lo: 0,
            t_src_us_hi: 0,
            age_ms_bits: 0,
            src_w: 0,
            src_h: 0,
            crop_x0: 0,
            crop_y0: 0,
            crop_w: 0,
            crop_h: 0,
            orientation: 0,
            scan_dir: 0,
            engine_w: 0,
            engine_h: 0,
            dst_x: 0,
            dst_y: 0,
            dst_w: 0,
            dst_h: 0,
            frame_gap_flag: 0,
            native_1to1_active: 0,
            continuity_epoch: 0,
        }
    }
}

const _: () = assert!(core::mem::size_of::<IngestSlotMeta>() == 96);

#[repr(C, align(16))]
pub struct Arena {
    // 1. 12-Frame Circular RGBA Display & Lossless Pixel Ring Buffer (up to 640x360: 11,059,200 B)
    pub rgba_ring: [[u8; MAX_FRAME_RGBA_BYTES]; RING_LEN],

    // 2. Per-Frame Planar SIMD128 Luma & Opponent Chroma Buffers (2,764,800 B)
    pub luma: [f32; MAX_FRAME_PIXELS],
    pub chroma_rg: [f32; MAX_FRAME_PIXELS],
    pub chroma_by: [f32; MAX_FRAME_PIXELS],

    // 3. Guarded Rank-2 ECA Background Model X = [b, 1], Residual, & Independent Prior Masks (2,304,000 B)
    pub bg_luma: [f32; MAX_FRAME_PIXELS],
    pub eca_res: [f32; MAX_FRAME_PIXELS],
    pub bg_stat_streak: [u8; MAX_FRAME_PIXELS],
    pub bg_eval_mask: [u8; MAX_FRAME_PIXELS],

    // 3b. 160x90 Multi-Channel SIMD128 Coarse Pyramid + Camera-Motion Reference + Ring ROI Scratch
    pub pyr_luma: [f32; PYR_PIXELS],
    pub prev_pyr_luma: [f32; PYR_PIXELS],
    pub pyr_rg: [f32; PYR_PIXELS],
    pub pyr_by: [f32; PYR_PIXELS],
    pub pyr_eca: [f32; PYR_PIXELS],
    pub roi_luma: [f32; MAX_ROI_PIXELS],
    pub roi_rg: [f32; MAX_ROI_PIXELS],
    pub roi_by: [f32; MAX_ROI_PIXELS],

    // 4. Multi-Target 2-Adic + Spectral + FISTA State (8 targets)
    pub targets: [TargetTrack; MAX_TARGETS],
    pub kalman: [CausalKalmanTrack; MAX_TARGETS],

    // 5. Pre-Assignment 16x16 Track-Peak Graph Laplacian & Fiedler Eigenvector
    pub laplacian: [f32; GRAPH_NODES * GRAPH_NODES],
    pub fiedler_vec: [f32; GRAPH_NODES],
    pub prev_fiedler_track: [f32; MAX_TARGETS],

    // 6. FISTA-Stabilized Eulerian Temporal-Contrast PiP Buffers (Primary + All 8 Target Slots)
    pub pip_rgba: [u8; MAX_PIP_PIXELS * 4],
    pub pip_ref_luma: [f32; MAX_PIP_PIXELS],
    pub pip_target_id: u32,
    pub pip_hold_frames: u32,
    pub all_pip_rgba: [[u8; MAX_PIP_PIXELS * 4]; MAX_TARGETS],
    pub all_pip_ref_luma: [[f32; MAX_PIP_PIXELS]; MAX_TARGETS],
    pub all_pip_tid: [u32; MAX_TARGETS],
    pub slot_meta: [f32; MAX_TARGETS * 8],
    pub dormant: [DormantTrack; MAX_DORMANT],
    pub jv_cost: [[f32; ASSIGN_COLS]; MAX_TARGETS],
    pub eval_tlx: [[usize; MAX_HYPS]; MAX_TARGETS],
    pub eval_tly: [[usize; MAX_HYPS]; MAX_TARGETS],
    pub eval_raw: [[f32; MAX_HYPS]; MAX_TARGETS],
    pub eval_app: [[f32; MAX_HYPS]; MAX_TARGETS],
    pub eval_anc: [[f32; MAX_HYPS]; MAX_TARGETS],
    pub eval_eca: [[f32; MAX_HYPS]; MAX_TARGETS],
    pub eval_sig: [[u32; MAX_HYPS]; MAX_TARGETS],
    pub eval_ok: [[bool; MAX_HYPS]; MAX_TARGETS],
    pub jacobi_mat: [f32; GRAPH_NODES * GRAPH_NODES],
    pub jacobi_vecs: [f32; GRAPH_NODES * GRAPH_NODES],
    pub coast_reacq_events: u32,
    pub dormant_reid_events: u32,
    pub peak_simultaneous_confirmed: u32,
    pub longest_tenure_frames: u32,
    pub longest_tenure_tid: u32,
    pub sustained_10s_count: u32,
    pub sustained_60s_count: u32,
    pub last_reid_gap_frames: u32,
    pub reset_base_track_id: u32,

    // 7. Connected-Component Seeder Workspace (20x11 SD = 220 cells / 40x22 HD = 880 cells)
    pub cc_energy: [f32; MAX_GRID_CELLS],
    pub cc_queue: [u16; MAX_GRID_CELLS],
    pub cc_visited: [u8; MAX_GRID_CELLS],
    pub peaks: [CandidatePeak; MAX_TARGETS],
    pub num_peaks: usize,

    // 8. Exported Flat Telemetry Vector (zero-marshaling Float32Array view)
    pub telemetry: [f32; TELEMETRY_LEN],

    // 9. Alignment-packed scalar/scratch fields and Wave-1 state (`<= 20,702,240 B`)
    pub ic_lk_cache: [TplIcLkCache; MAX_TARGETS],
    pub micro_swarm: soa_tracks::MicroSwarmState,
    pub bos: bos::BosState,
    pub vibro: vibrometry::VibrometryState,
    pub hodge: hodge::HodgeState,
    pub ring_meta: [IngestSlotMeta; RING_LEN],
    pub pending_ingest_meta: IngestSlotMeta,
    pub row_luma_means: [[f32; MAX_FRAME_H]; RING_LEN],
    pub wave1_telem: [u32; WAVE1_TELEM_LEN],
    pub wave2_telem: [u32; WAVE2_TELEM_LEN],
    pub ingest_ctrl: [u32; INGEST_CTRL_LEN],
    pub continuum: continuum_physics::ContinuumState,
    pub lanczos_scratch: LanczosScratch,
    pub ode_scratch: physics_id::PhysicsIdScratch,
    pub lyap_scratch: physics_id::LyapunovScratch,
    pub collision_state: physics_id::CollisionAnalyzerState,
    pub ode_states: [physics_id::OdeTrackState; MAX_TARGETS],
    pub pend_states: [physics_id::PendulumState; MAX_TARGETS],
    pub imm: [ImmTrack; MAX_TARGETS],
    pub cum_time_sec: f64,
    pub prev_raw_pts_sec: f64,
    pub pending_pts_sec: f64,
    pub hist_t_sec: [f64; HIST_RING_LEN],
    pub phys_ts: [f64; HIST_RING_LEN],
    pub chroma_tile_mask: [u64; 24],
    pub ring_cursor: u64,
    pub res_mode: u32,
    pub frame_idx: u32,
    pub next_track_id: u32,
    pub lookahead_k: u32,
    pub last_display_slot: u32,
    pub eca_adapt_rate: f32,
    pub fista_lambda: f32,
    pub eulerian_gain: f32,
    pub phase_offset_rad: f32,
    pub audio_out_latency_sec: f32,
    pub spectral_gamma: f32,
    pub lab_mode: u32,
    pub clutter_db: f32,
    pub clutter_db_valid: u32,
    pub clutter_eval_ratio: f32,
    pub eca_err2_ema: f32,
    pub eca_sig2_ema: f32,
    pub fiedler_lambda2: f32,
    pub fiedler_gap: f32,
    pub padic_id_swaps: u32,
    pub kalman_id_swaps: u32,
    pub last_strobe_frame: u32,
    pub has_strobed: u32,
    pub stimulus_frame: u32,
    pub stimulus_armed: u32,
    pub focus_target_id: u32,
    pub scene_gravity_px_s2: f32,
    pub scene_gravity_conf: f32,
    pub scatter_deg: f32,
    pub scatter_x: f32,
    pub scatter_y: f32,
    pub scatter_th1_deg: f32,
    pub scatter_th2_deg: f32,
    pub scatter_event_frame: u32,
    pub has_explicit_pts: u32,
    pub last_discont_frame: u32,
    pub discont_count: u32,
    pub cam_motion_state: u32,
    pub cam_pan_streak: u32,
    pub cam_cut_hold: u32,
    pub cam_pan_speed_px_s: f32,
    pub last_pan_dx: i32,
    pub last_pan_dy: i32,
    pub cum_cam_x: f32,
    pub cum_cam_y: f32,
    pub hist_cam_x: [f32; HIST_RING_LEN],
    pub hist_cam_y: [f32; HIST_RING_LEN],
    pub scene_gravity_arcs: u32,
    pub last_grav_arc_track: u32,
    pub last_grav_arc_frame: u32,
    pub last_strike_tid: u32,
    pub last_strike_frame: u32,
    pub lyap_cache: [f32; MAX_TARGETS],
    pub phys_xs: [f32; HIST_RING_LEN],
    pub phys_ys: [f32; HIST_RING_LEN],
    pub active_rgba_slot: u32,
    pub last_pan_dx_f: f32,
    pub last_pan_dy_f: f32,
    pub inspected_hero_tid: u32,
    pub inspected_micro_tid_pin: u32,
    pub last_pre_tick_frame: u32,
    pub physics_telem_dirty_mask: u16,
    pub phys_vis: [u8; HIST_RING_LEN],
    pub continuum_was_active: u8,
    pub swarm_mode_cached: u8,
    pub swarm_telem_zeroed: u8,
    pub last_scene_cut: u8,
    pub pre_tick_seen: u8,
}

// Compile-time structural bound: Arena + Wave1 + Wave2 state <= 24,117,248 B (23.00 MiB) within 448 pages (29,360,128 B)
const _: () = assert!(core::mem::size_of::<Arena>() <= 24_117_248);

static mut ARENA: Arena = unsafe { core::mem::zeroed() };

#[inline(always)]
unsafe fn arena_mut() -> &'static mut Arena {
    &mut *addr_of_mut!(ARENA)
}

#[inline(never)]
unsafe fn fill_chroma_tiles_slow(
    chroma_tile_mask: &mut [u64; 24],
    rgba: &[u8; MAX_FRAME_RGBA_BYTES],
    luma: &[f32; MAX_FRAME_PIXELS],
    chroma_rg: &mut [f32; MAX_FRAME_PIXELS],
    chroma_by: &mut [f32; MAX_FRAME_PIXELS],
    ty0: usize,
    ty1: usize,
    need_bits: u64,
    frame_w: usize,
    frame_h: usize,
) {
    let rgba_ptr = rgba.as_ptr();
    let luma_ptr = luma.as_ptr();
    let rg_ptr = chroma_rg.as_mut_ptr();
    let by_ptr = chroma_by.as_mut_ptr();
    for ty in ty0..=ty1 {
        let mut todo = need_bits & !chroma_tile_mask[ty];
        if todo == 0 {
            continue;
        }
        chroma_tile_mask[ty] |= todo;
        let y0 = ty << 4;
        let y_end = (y0 + 16).min(frame_h);
        while todo != 0 {
            let tx0 = todo.trailing_zeros() as usize;
            let run = (!(todo >> tx0)).trailing_zeros() as usize;
            let run_mask = if run >= 64 {
                !0u64
            } else {
                ((1u64 << run) - 1) << tx0
            };
            todo &= !run_mask;
            rgba_tiles_run_to_chroma_simd128(
                rgba_ptr,
                luma_ptr,
                rg_ptr,
                by_ptr,
                frame_w,
                tx0 << 4,
                run,
                y0,
                y_end,
            );
        }
    }
}

#[inline(always)]
unsafe fn ensure_chroma_rect_fields(
    chroma_tile_mask: &mut [u64; 24],
    rgba: &[u8; MAX_FRAME_RGBA_BYTES],
    luma: &[f32; MAX_FRAME_PIXELS],
    chroma_rg: &mut [f32; MAX_FRAME_PIXELS],
    chroma_by: &mut [f32; MAX_FRAME_PIXELS],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    frame_w: usize,
    frame_h: usize,
) {
    if w == 0 || h == 0 || x >= frame_w || y >= frame_h {
        return;
    }
    let x1 = (x + w - 1).min(frame_w - 1);
    let y1 = (y + h - 1).min(frame_h - 1);
    let tx0 = x >> 4;
    let tx1 = (x1 >> 4).min(39);
    let ty0 = y >> 4;
    let ty1 = (y1 >> 4).min(22);
    let n_bits = tx1 - tx0 + 1;
    let need_bits = if n_bits >= 64 { !0u64 } else { ((1u64 << n_bits) - 1) << tx0 };
    for ty in ty0..=ty1 {
        if (need_bits & !chroma_tile_mask[ty]) != 0 {
            fill_chroma_tiles_slow(
                chroma_tile_mask,
                rgba,
                luma,
                chroma_rg,
                chroma_by,
                ty0,
                ty1,
                need_bits,
                frame_w,
                frame_h,
            );
            return;
        }
    }
}

macro_rules! ensure_chroma {
    ($a:expr, $x:expr, $y:expr, $w:expr, $h:expr, $fw:expr, $fh:expr) => {
        ensure_chroma_rect_fields(
            &mut $a.chroma_tile_mask,
            &$a.rgba_ring[$a.active_rgba_slot as usize],
            &$a.luma,
            &mut $a.chroma_rg,
            &mut $a.chroma_by,
            $x,
            $y,
            $w,
            $h,
            $fw,
            $fh,
        )
    };
}

// ============================================================================
// Helper Kernels: Foreground-Masked Templates, Pyramid Proposals, JV & Backtrack
// ============================================================================

const fn const_inv_sqrt_f64(x: f64) -> f64 {
    let mut y = if x >= 64.0 {
        0.1
    } else if x >= 16.0 {
        0.2
    } else {
        0.3
    };
    let mut i = 0usize;
    while i < 12 {
        y = 0.5 * y * (3.0 - x * y * y);
        i += 1;
    }
    y
}

#[repr(C, align(16))]
struct DeslipGeomLut {
    urx: [f32; 256],
    ury: [f32; 256],
    m_ge_1225: [i32; 256],
    m_ge_3600: [i32; 256],
}

const TPL_DESLIP_LUT: DeslipGeomLut = {
    let mut urx = [0.0f32; 256];
    let mut ury = [0.0f32; 256];
    let mut m_ge_1225 = [0i32; 256];
    let mut m_ge_3600 = [0i32; 256];
    let mut ry = 0usize;
    while ry < 16 {
        let dy = (ry as f64) - 7.5;
        let mut rx = 0usize;
        while rx < 16 {
            let dx = (rx as f64) - 7.5;
            let r2 = dx * dx + dy * dy;
            let idx = ry * 16 + rx;
            if r2 >= 12.25 {
                let inv_r = const_inv_sqrt_f64(r2);
                urx[idx] = (dx * inv_r) as f32;
                ury[idx] = (dy * inv_r) as f32;
                m_ge_1225[idx] = -1;
            }
            if r2 >= 36.0 {
                m_ge_3600[idx] = -1;
            }
            rx += 1;
        }
        ry += 1;
    }
    DeslipGeomLut { urx, ury, m_ge_1225, m_ge_3600 }
};

#[inline]
fn rebuild_tpl8_from_16(t: &mut TargetTrack) {
    t.tpl_grad_dirty = 1;
    unsafe {
        simd_ops::compute_anchor_luma_stats_simd128(
            t.anchor_luma.as_ptr(),
            t.tpl_weight.as_ptr(),
            &mut t.anchor_mu_a,
            &mut t.anchor_std_a,
        );
        let q25 = f32x4_splat(0.25);
        macro_rules! pool_plane {
            ($src:expr, $dst:expr, $scale:expr) => {
                for py8 in 0..8usize {
                    let r0 = ($src).add((py8 * 2) * 16);
                    let r1 = r0.add(16);
                    let d = ($dst).add(py8 * 8);

                    let a0 = v128_load(r0 as *const v128);
                    let a1 = v128_load(r0.add(4) as *const v128);
                    let a2 = v128_load(r0.add(8) as *const v128);
                    let a3 = v128_load(r0.add(12) as *const v128);

                    let b0 = v128_load(r1 as *const v128);
                    let b1 = v128_load(r1.add(4) as *const v128);
                    let b2 = v128_load(r1.add(8) as *const v128);
                    let b3 = v128_load(r1.add(12) as *const v128);

                    let s0 = f32x4_add(
                        f32x4_add(i32x4_shuffle::<0, 2, 4, 6>(a0, a1), i32x4_shuffle::<1, 3, 5, 7>(a0, a1)),
                        i32x4_shuffle::<0, 2, 4, 6>(b0, b1),
                    );
                    let lo = f32x4_add(s0, i32x4_shuffle::<1, 3, 5, 7>(b0, b1));

                    let s1 = f32x4_add(
                        f32x4_add(i32x4_shuffle::<0, 2, 4, 6>(a2, a3), i32x4_shuffle::<1, 3, 5, 7>(a2, a3)),
                        i32x4_shuffle::<0, 2, 4, 6>(b2, b3),
                    );
                    let hi = f32x4_add(s1, i32x4_shuffle::<1, 3, 5, 7>(b2, b3));

                    if $scale {
                        v128_store(d as *mut v128, f32x4_mul(q25, lo));
                        v128_store(d.add(4) as *mut v128, f32x4_mul(q25, hi));
                    } else {
                        v128_store(d as *mut v128, lo);
                        v128_store(d.add(4) as *mut v128, hi);
                    }
                }
            };
        }
        pool_plane!(t.tpl_luma.as_ptr(), t.tpl8_luma.as_mut_ptr(), true);
        pool_plane!(t.tpl_rg.as_ptr(), t.tpl8_rg.as_mut_ptr(), true);
        pool_plane!(t.tpl_by.as_ptr(), t.tpl8_by.as_mut_ptr(), true);
        pool_plane!(t.tpl_weight.as_ptr(), t.tpl8_w.as_mut_ptr(), false);
    }
}

unsafe fn capture_template_scaled(
    luma: &[f32; MAX_FRAME_PIXELS],
    rg: &[f32; MAX_FRAME_PIXELS],
    by: &[f32; MAX_FRAME_PIXELS],
    eca_res: &[f32; MAX_FRAME_PIXELS],
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    cx: usize,
    cy: usize,
    t: &mut TargetTrack,
) {
    let patch_dim = 16 * scale_px;
    let x0 = cx.min(frame_w - patch_dim);
    let y0 = cy.min(frame_h - patch_dim);
    let cstep = 4 * scale_px;
    let mut eca_patch = [0.0f32; 256];

    for ry in 0..16usize {
        let src_row = (y0 + ry * scale_px) * frame_w + x0;
        let dst_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = src_row + vec_idx * cstep;
            let doff = dst_row + vec_idx * 4;
            let vl = load_pooled4_f32(luma.as_ptr().add(ioff), frame_w, scale_px);
            let vrg = load_pooled4_f32(rg.as_ptr().add(ioff), frame_w, scale_px);
            let vby = load_pooled4_f32(by.as_ptr().add(ioff), frame_w, scale_px);
            let veca = load_pooled4_f32(eca_res.as_ptr().add(ioff), frame_w, scale_px);

            v128_store(t.tpl_luma.as_mut_ptr().add(doff) as *mut v128, vl);
            v128_store(t.tpl_rg.as_mut_ptr().add(doff) as *mut v128, vrg);
            v128_store(t.tpl_by.as_mut_ptr().add(doff) as *mut v128, vby);
            v128_store(t.anchor_luma.as_mut_ptr().add(doff) as *mut v128, vl);
            v128_store(t.anchor_rg.as_mut_ptr().add(doff) as *mut v128, vrg);
            v128_store(t.anchor_by.as_mut_ptr().add(doff) as *mut v128, vby);
            v128_store(eca_patch.as_mut_ptr().add(doff) as *mut v128, veca);
        }
    }

    // Radial Gaussian x ECA silhouette mask normalized to sum(tpl_weight) == 1.0
    let zero_v = f32x4_splat(0.0);
    let one_v = f32x4_splat(1.0);
    let inv_eca_v = f32x4_splat(1.0 / 0.075);
    let v_045 = f32x4_splat(0.045);
    let v_955 = f32x4_splat(0.955);
    let mut sum_w_v = zero_v;
    let mut i = 0usize;
    while i < 256 {
        let veca = v128_load(eca_patch.as_ptr().add(i) as *const v128);
        let w_rad = v128_load(TPL_RAD_LUT.as_ptr().add(i) as *const v128);
        let c01 = f32x4_pmin(one_v, f32x4_pmax(zero_v, f32x4_mul(veca, inv_eca_v)));
        let w_fg = f32x4_add(v_045, f32x4_mul(v_955, c01));
        let w = f32x4_mul(w_rad, w_fg);
        v128_store(t.tpl_weight.as_mut_ptr().add(i) as *mut v128, w);
        sum_w_v = f32x4_add(sum_w_v, w);
        i += 4;
    }
    let sum_w = f32x4_extract_lane::<0>(sum_w_v)
        + f32x4_extract_lane::<1>(sum_w_v)
        + f32x4_extract_lane::<2>(sum_w_v)
        + f32x4_extract_lane::<3>(sum_w_v);
    let v_inv_w = f32x4_splat(1.0 / sum_w.max(1e-6));
    let mut j = 0usize;
    while j < 256 {
        let w = v128_load(t.tpl_weight.as_ptr().add(j) as *const v128);
        v128_store(t.tpl_weight.as_mut_ptr().add(j) as *mut v128, f32x4_mul(w, v_inv_w));
        j += 4;
    }
    apply_radial_deslip_weights(t, 1.0);

    t.curr_sig = haar_morton_signature_simd128(
        t.tpl_luma.as_ptr(),
        t.tpl_rg.as_ptr(),
        t.tpl_by.as_ptr(),
        16,
        0,
        0,
    );
}

/// Suppresses 1D radial rod / glider-rail gradients on outer template texels (`r >= 3.5`)
/// where luma intensity gradients run perpendicular to the radial spoke vector `r_hat`,
/// while keeping compact core texels and radial-aligned target boundaries intact.
#[inline(never)]
fn apply_radial_deslip_weights(t: &mut TargetTrack, strength: f32) {
    let k_str = strength.clamp(0.0, 1.0);
    unsafe {
        let lum = t.tpl_luma.as_ptr();
        let w_ptr = t.tpl_weight.as_mut_ptr();
        let urx_ptr = TPL_DESLIP_LUT.urx.as_ptr();
        let ury_ptr = TPL_DESLIP_LUT.ury.as_ptr();
        let m12_ptr = TPL_DESLIP_LUT.m_ge_1225.as_ptr();
        let m36_ptr = TPL_DESLIP_LUT.m_ge_3600.as_ptr();

        let v_kstr = f32x4_splat(k_str);
        let v_one = f32x4_splat(1.0);
        let v_half = f32x4_splat(0.5);
        let v_072 = f32x4_splat(0.72);
        let v_g2_thr = f32x4_splat(0.00025);
        let v_eps = f32x4_splat(1e-6);
        let v_062 = f32x4_splat(0.62);
        let v_inv062 = f32x4_splat(1.0 / 0.62);
        let v_014 = f32x4_splat(0.14);
        let v_086 = f32x4_splat(0.86);
        let v_zero = f32x4_splat(0.0);
        let v_inv_dx0 = i32x4(
            1.0f32.to_bits() as i32,
            0.5f32.to_bits() as i32,
            0.5f32.to_bits() as i32,
            0.5f32.to_bits() as i32,
        );
        let v_inv_dx3 = i32x4(
            0.5f32.to_bits() as i32,
            0.5f32.to_bits() as i32,
            0.5f32.to_bits() as i32,
            1.0f32.to_bits() as i32,
        );

        let mut sum_w_v = v_zero;
        for ry in 0..16usize {
            let row = ry * 16;
            let ym = ry.saturating_sub(1) * 16;
            let yp = (ry + 1).min(15) * 16;
            let inv_dy = if ry == 0 || ry == 15 { v_one } else { v_half };

            let c0 = v128_load(lum.add(row) as *const v128);
            let c1 = v128_load(lum.add(row + 4) as *const v128);
            let c2 = v128_load(lum.add(row + 8) as *const v128);
            let c3 = v128_load(lum.add(row + 12) as *const v128);

            let gxs = [
                f32x4_mul(
                    f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c0, c1), i32x4_shuffle::<0, 0, 1, 2>(c0, c0)),
                    v_inv_dx0,
                ),
                f32x4_mul(
                    f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c1, c2), i32x4_shuffle::<3, 4, 5, 6>(c0, c1)),
                    v_half,
                ),
                f32x4_mul(
                    f32x4_sub(i32x4_shuffle::<1, 2, 3, 4>(c2, c3), i32x4_shuffle::<3, 4, 5, 6>(c1, c2)),
                    v_half,
                ),
                f32x4_mul(
                    f32x4_sub(i32x4_shuffle::<1, 2, 3, 3>(c3, c3), i32x4_shuffle::<3, 4, 5, 6>(c2, c3)),
                    v_inv_dx3,
                ),
            ];

            for c in 0..4usize {
                let off = row + c * 4;
                let w = v128_load(w_ptr.add(off) as *const v128);
                let gy = f32x4_mul(
                    f32x4_sub(v128_load(lum.add(yp + c * 4) as *const v128), v128_load(lum.add(ym + c * 4) as *const v128)),
                    inv_dy,
                );
                let gx = gxs[c];
                let g2 = f32x4_add(f32x4_mul(gx, gx), f32x4_mul(gy, gy));
                let g_mag = f32x4_pmax(f32x4_sqrt(g2), v_eps);
                let urx = v128_load(urx_ptr.add(off) as *const v128);
                let ury = v128_load(ury_ptr.add(off) as *const v128);
                let align_rad = f32x4_div(f32x4_abs(f32x4_add(f32x4_mul(gx, urx), f32x4_mul(gy, ury))), g_mag);
                let s_deslip = f32x4_pmin(v_one, f32x4_pmax(v_zero, f32x4_mul(align_rad, v_inv062)));
                let deslip_f = f32x4_add(v_014, f32x4_mul(v_086, f32x4_mul(s_deslip, s_deslip)));
                let w_grad = v128_bitselect(deslip_f, v_one, f32x4_lt(align_rad, v_062));
                let w_low_g = v128_bitselect(v_072, v_one, v128_load(m36_ptr.add(off) as *const v128));
                let w_factor = v128_bitselect(w_grad, w_low_g, f32x4_gt(g2, v_g2_thr));
                let mult = f32x4_sub(v_one, f32x4_mul(v_kstr, f32x4_sub(v_one, w_factor)));
                let w_out = v128_bitselect(f32x4_mul(w, mult), w, v128_load(m12_ptr.add(off) as *const v128));
                v128_store(w_ptr.add(off) as *mut v128, w_out);
                sum_w_v = f32x4_add(sum_w_v, w_out);
            }
        }

        let sum_w = f32x4_extract_lane::<0>(sum_w_v)
            + f32x4_extract_lane::<1>(sum_w_v)
            + f32x4_extract_lane::<2>(sum_w_v)
            + f32x4_extract_lane::<3>(sum_w_v);
        let v_inv_w = f32x4_splat(1.0 / sum_w.max(1e-6));
        let mut i = 0usize;
        while i < 256 {
            let w = v128_load(w_ptr.add(i) as *const v128);
            v128_store(w_ptr.add(i) as *mut v128, f32x4_mul(w, v_inv_w));
            i += 4;
        }
    }
    rebuild_tpl8_from_16(t);
}

unsafe fn blend_template_scaled(
    luma: &[f32; MAX_FRAME_PIXELS],
    rg: &[f32; MAX_FRAME_PIXELS],
    by: &[f32; MAX_FRAME_PIXELS],
    eca_res: &[f32; MAX_FRAME_PIXELS],
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    cx: usize,
    cy: usize,
    alpha: f32,
    t: &mut TargetTrack,
) {
    let patch_dim = 16 * scale_px;
    let x0 = cx.min(frame_w - patch_dim);
    let y0 = cy.min(frame_h - patch_dim);
    let cstep = 4 * scale_px;
    let v_a = f32x4_splat(alpha);
    let v_ia = f32x4_splat(1.0 - alpha);
    let zero_v = f32x4_splat(0.0);
    let one_v = f32x4_splat(1.0);
    let inv_eca_v = f32x4_splat(1.0 / 0.075);
    let v_045 = f32x4_splat(0.045);
    let v_955 = f32x4_splat(0.955);
    let v_96 = f32x4_splat(0.96);
    let v_04 = f32x4_splat(0.04);
    let mut sum_w_v = zero_v;

    for ry in 0..16usize {
        let src_row = (y0 + ry * scale_px) * frame_w + x0;
        let dst_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = src_row + vec_idx * cstep;
            let doff = dst_row + vec_idx * 4;

            let tl = v128_load(t.tpl_luma.as_ptr().add(doff) as *const v128);
            let il = load_pooled4_f32(luma.as_ptr().add(ioff), frame_w, scale_px);
            v128_store(
                t.tpl_luma.as_mut_ptr().add(doff) as *mut v128,
                f32x4_add(f32x4_mul(v_ia, tl), f32x4_mul(v_a, il)),
            );

            let trg = v128_load(t.tpl_rg.as_ptr().add(doff) as *const v128);
            let irg = load_pooled4_f32(rg.as_ptr().add(ioff), frame_w, scale_px);
            v128_store(
                t.tpl_rg.as_mut_ptr().add(doff) as *mut v128,
                f32x4_add(f32x4_mul(v_ia, trg), f32x4_mul(v_a, irg)),
            );

            let tby = v128_load(t.tpl_by.as_ptr().add(doff) as *const v128);
            let iby = load_pooled4_f32(by.as_ptr().add(ioff), frame_w, scale_px);
            v128_store(
                t.tpl_by.as_mut_ptr().add(doff) as *mut v128,
                f32x4_add(f32x4_mul(v_ia, tby), f32x4_mul(v_a, iby)),
            );

            let veca = load_pooled4_f32(eca_res.as_ptr().add(ioff), frame_w, scale_px);
            let w_rad = v128_load(TPL_RAD_LUT.as_ptr().add(doff) as *const v128);
            let c01 = f32x4_pmin(one_v, f32x4_pmax(zero_v, f32x4_mul(veca, inv_eca_v)));
            let w_fg = f32x4_add(v_045, f32x4_mul(v_955, c01));
            let target_w = f32x4_mul(w_rad, w_fg);
            let tw_prev = v128_load(t.tpl_weight.as_ptr().add(doff) as *const v128);
            let nw = f32x4_add(f32x4_mul(v_96, tw_prev), f32x4_mul(v_04, target_w));
            v128_store(t.tpl_weight.as_mut_ptr().add(doff) as *mut v128, nw);
            sum_w_v = f32x4_add(sum_w_v, nw);
        }
    }

    let sum_w = f32x4_extract_lane::<0>(sum_w_v)
        + f32x4_extract_lane::<1>(sum_w_v)
        + f32x4_extract_lane::<2>(sum_w_v)
        + f32x4_extract_lane::<3>(sum_w_v);
    let v_inv_w = f32x4_splat(1.0 / sum_w.max(1e-6));
    let mut j = 0usize;
    while j < 256 {
        let w = v128_load(t.tpl_weight.as_ptr().add(j) as *const v128);
        v128_store(t.tpl_weight.as_mut_ptr().add(j) as *mut v128, f32x4_mul(w, v_inv_w));
        j += 4;
    }
    rebuild_tpl8_from_16(t);
}

/// Exact `#![no_std]` Jonker-Volgenant / Kuhn-Munkres Shortest Augmenting Path Bipartite Solver
/// (`nr <= 8` tracks x `nc <= 32` hypotheses + per-track occlusion null columns, zero allocations).
fn solve_jv_assign_8x32(
    nr: usize,
    nc: usize,
    cost: &[[f32; ASSIGN_COLS]; MAX_TARGETS],
) -> [usize; MAX_TARGETS] {
    let mut row_match = [usize::MAX; MAX_TARGETS];
    if nr == 0 || nc == 0 {
        return row_match;
    }
    let nr = nr.min(MAX_TARGETS);
    let nc = nc.min(ASSIGN_COLS);
    if nr == 1 {
        let row = &cost[0][..nc];
        let mut best_j = 0usize;
        let mut best_v = row[0];
        for (j, &c) in row.iter().enumerate().skip(1) {
            if c < best_v {
                best_v = c;
                best_j = j;
            }
        }
        row_match[0] = best_j;
        return row_match;
    }

    let mut u = [0.0f32; MAX_TARGETS + 1];
    let mut v_buf = [0.0f32; ASSIGN_COLS + 1];
    let v = &mut v_buf[..=nc];
    let mut p_buf = [0usize; ASSIGN_COLS + 1];
    let p = &mut p_buf[..=nc];
    let mut way_buf = [0usize; ASSIGN_COLS + 1];
    let way = &mut way_buf[..=nc];
    let mut minv_buf = [1e30f32; ASSIGN_COLS + 1];
    let minv = &mut minv_buf[..=nc];
    let mut used_buf = [false; ASSIGN_COLS + 1];
    let used = &mut used_buf[..=nc];

    for i in 1..=nr {
        p[0] = i;
        let mut j0 = 0usize;
        minv.fill(1e30f32);
        used.fill(false);

        for _step in 0..=nc {
            used[j0] = true;
            let i0 = p[j0];
            if i0 == 0 {
                break;
            }
            let u_i0 = u[i0];
            let cost_row = &cost[i0 - 1][..nc];
            let mut delta = 1e30f32;
            let mut j1 = 0usize;

            for j in 1..=nc {
                if !used[j] {
                    let cur = cost_row[j - 1] - u_i0 - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta || j1 == 0 {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            if j1 == 0 {
                break;
            }

            for j in 0..=nc {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        for _aug in 0..=nc {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    for j in 1..=nc {
        if p[j] > 0 && p[j] <= nr {
            row_match[p[j] - 1] = j - 1;
        }
    }
    row_match
}

#[inline]
fn archive_dormant_track(
    a: &mut Arena,
    m: usize,
    f_idx: u32,
    last_x: f32,
    last_y: f32,
    s: f32,
) {
    let t = &a.targets[m];
    let is_user_pin = a.focus_target_id != 0 && t.track_id == a.focus_target_id;
    if t.confirmed == 0
        || t.track_id == 0
        || (!is_user_pin
            && t.dormant_reid == 0
            && (t.age < 8 || t.max_excursion2 < (12.0 * s) * (12.0 * s)))
    {
        return;
    }
    let tid = t.track_id;
    let mut best_di = 0usize;
    let mut oldest_f = u32::MAX;
    for di in 0..MAX_DORMANT {
        if a.dormant[di].active != 0 && a.dormant[di].track_id == tid {
            best_di = di;
            break;
        }
        if a.dormant[di].active == 0 {
            best_di = di;
            oldest_f = 0;
        } else if a.dormant[di].lost_frame < oldest_f {
            oldest_f = a.dormant[di].lost_frame;
            best_di = di;
        }
    }
    let vx = if t.coasting > 0 { t.entry_vx } else { t.vel_x };
    let vy = if t.coasting > 0 { t.entry_vy } else { t.vel_y };
    a.dormant[best_di] = DormantTrack {
        active: 1,
        is_ballistic: t.is_ballistic,
        sustained_10s_counted: t.sustained_10s_counted,
        sustained_60s_counted: t.sustained_60s_counted,
        reacq_count: t.reacq_count,
        _pad_d: 0,
        track_id: tid,
        anchor_sig: t.anchor_sig,
        curr_sig: t.curr_sig,
        lost_frame: f_idx,
        orig_birth_frame: t.orig_birth_frame,
        last_x,
        last_y,
        vx,
        vy,
        grav_ay_frame: t.grav_ay_frame,
        age: t.age,
        anchor_luma: t.anchor_luma,
        anchor_rg: t.anchor_rg,
        anchor_by: t.anchor_by,
        tpl_weight: t.tpl_weight,
    };
}

/// Searches the 32-slot DormantTrack graveyard using 2-adic Haar-Morton + 16x16 weighted anchor L1
/// over a 3-mode (loiter / linear / ballistic-arc) corridor. Returns the winning dormant index.
fn find_best_dormant_match(
    a: &Arena,
    m: usize,
    f_idx: u32,
    mcx: f32,
    mcy: f32,
    frame_w: usize,
    frame_h: usize,
    half_patch: f32,
    s: f32,
    strict_fast: bool,
) -> Option<usize> {
    let mut best_di: Option<usize> = None;
    let mut best_dscore = 1e30f32;
    let max_fw = (frame_w as f32) - half_patch;
    let max_fh = (frame_h as f32) - half_patch;

    let mut active_cnt = 0usize;
    for k in 0..MAX_TARGETS {
        if a.targets[k].active != 0 && a.targets[k].confirmed != 0 {
            active_cnt += 1;
        }
    }

    for di in 0..MAX_DORMANT {
        let d = &a.dormant[di];
        if d.active == 0 || f_idx <= d.lost_frame {
            continue;
        }
        let gap = f_idx.saturating_sub(d.lost_frame);
        if gap > DORMANT_MAX_AGE {
            continue;
        }

        let mut id_busy = false;
        for o in 0..MAX_TARGETS {
            if o != m
                && a.targets[o].active != 0
                && a.targets[o].confirmed != 0
                && a.targets[o].track_id == d.track_id
            {
                id_busy = true;
                break;
            }
        }
        if id_busy {
            continue;
        }

        let dt_lin = (gap as f32).min(120.0);
        let end_x = (d.last_x + dt_lin * d.vx).clamp(half_patch, max_fw);
        let end_y = (d.last_y + dt_lin * d.vy).clamp(half_patch, max_fh);
        let seg_dx = end_x - d.last_x;
        let seg_dy = end_y - d.last_y;
        let seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy;
        let d2_stop = (mcx - d.last_x) * (mcx - d.last_x) + (mcy - d.last_y) * (mcy - d.last_y);
        let mut min_d2 = if seg_len2 > 1.0 {
            let tau = (((mcx - d.last_x) * seg_dx + (mcy - d.last_y) * seg_dy) / seg_len2)
                .clamp(0.0, 1.05);
            let qx = d.last_x + tau * seg_dx;
            let qy = d.last_y + tau * seg_dy;
            ((mcx - qx) * (mcx - qx) + (mcy - qy) * (mcy - qy)).min(d2_stop)
        } else {
            d2_stop
        };

        if d.is_ballistic != 0 || d.grav_ay_frame.abs() > 0.005 {
            let dt_bal = (gap as f32).min(120.0);
            let p_bx = (d.last_x + 0.85 * dt_bal * d.vx).clamp(half_patch, max_fw);
            let p_by = (d.last_y + 0.85 * dt_bal * d.vy + 0.5 * d.grav_ay_frame * dt_bal * dt_bal)
                .clamp(half_patch, max_fh);
            let d2_bal = (mcx - p_bx) * (mcx - p_bx) + (mcy - p_by) * (mcy - p_by);
            if d2_bal < min_d2 {
                min_d2 = d2_bal;
            }
        }

        if a.lab_mode == 0 && d.is_ballistic == 0 && gap > 60 && d2_stop > (52.0 * s) * (52.0 * s) {
            continue;
        }
        let base_r = ((44.0 + 0.35 * (gap.min(240) as f32)) * s).min(140.0 * s);
        let max_r = if a.lab_mode == 1 && active_cnt >= 5 {
            (26.0 * s).min(base_r)
        } else if strict_fast {
            (56.0 * s).min(base_r)
        } else {
            base_r
        };
        if min_d2 > max_r * max_r {
            continue;
        }

        let reid_sig = unsafe {
            haar_morton_signature_hyst_simd128(
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                16,
                0,
                0,
                1,
                d.anchor_sig,
                true,
            )
        };
        if has_opposite_chroma_pole(d.anchor_sig, reid_sig) {
            continue;
        }

        let anc_l1 = unsafe {
            weighted_anchor_zm_l1_simd128(
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                16,
                0,
                0,
                d.anchor_luma.as_ptr(),
                d.anchor_rg.as_ptr(),
                d.anchor_by.as_ptr(),
                d.tpl_weight.as_ptr(),
                1,
            )
        };

        let rval = padic_ultrametric_val(d.anchor_sig, reid_sig)
            .max(padic_ultrametric_val(d.curr_sig, reid_sig));
        let min_rval = if (d.anchor_sig & (1 << 31)) != 0 { 10 } else { 12 };
        let app_ok = if strict_fast {
            rval >= min_rval && anc_l1 <= 0.118
        } else {
            (rval >= min_rval && anc_l1 <= 0.165)
                || (rval >= 14 && anc_l1 <= 0.195 && min_d2 <= (44.0 * s) * (44.0 * s))
        };
        if !app_ok {
            continue;
        }

        let dv2 = d.vx * d.vx + d.vy * d.vy;
        if gap <= 150
            && d.is_ballistic == 0
            && dv2 > (1.6 * s) * (1.6 * s)
            && (a.targets[m].vel_x * d.vx + a.targets[m].vel_y * d.vy) < 0.0
        {
            continue;
        }

        let dist_norm = wasm_sqrtf(min_d2) / (44.0 * s);
        let dscore = anc_l1 + 0.045 * dist_norm - 0.004 * ((rval as i32 - 10).max(0) as f32);
        if dscore < best_dscore {
            best_dscore = dscore;
            best_di = Some(di);
        }
    }
    best_di
}

/// Cached proposal fine score & refinement status per track proposal slot.
#[derive(Clone, Copy)]
struct PropFineCache {
    tlx: i16,
    tly: i16,
    flags: u8, // bit 0: valid, bit 1: st4_centered, bit 2: st1_centered
    half_c: f32,
    weca: f32,
}

impl PropFineCache {
    #[inline(always)]
    const fn empty() -> Self {
        Self { tlx: 0, tly: 0, flags: 0, half_c: 0.0, weca: 0.0 }
    }
}

#[inline(never)]
unsafe fn propose_track_candidates_pyr_ex(
    a: &mut Arena,
    m: usize,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
    tpl_inv_w: f32,
    tpl_mu_t: f32,
    out_cache: *mut [PropFineCache; 2],
) -> [(f32, f32, bool); 2] {
    let pyr_div = (2 * scale_px) as f32;
    let inv_pyr_div = 1.0 / pyr_div;
    let half_patch = 8.0 * s;
    let patch_dim = 16 * scale_px;
    let max_tlx_i = (frame_w - patch_dim - 1) as i32;
    let max_tly_i = (frame_h - patch_dim - 1) as i32;
    let max_px = (PYR_W - 8) as i32;
    let max_py = (PYR_H - 8) as i32;

    let t = &a.targets[m];
    let coasting = t.coasting;
    let pred_cx = t.pred_cx;
    let pred_cy = t.pred_cy;

    let mut roots = [(0i32, 0i32); 6];
    let mut n_roots = 0usize;

    let p0x = (((pred_cx - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
    let p0y = (((pred_cy - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
    roots[n_roots] = (p0x, p0y);
    n_roots += 1;

    if coasting == 0 && ((pred_cx - t.pos_x).abs() + (pred_cy - t.pos_y).abs() > 10.5 * s) {
        let sx = (((t.pos_x - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
        let sy = (((t.pos_y - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
        roots[n_roots] = (sx, sy);
        n_roots += 1;

        // Forward-heading extended root for confirmed airborne projectiles & returning periodic whips
        // (one-way vehicles strictly satisfy cur_exc2 == max_excursion2 AND dot(pos - birth, vel) > 0;
        // periodic oscillators violate one of those on ~96% of every swing regardless of birth phase!)
        let bdx_p = t.pos_x - t.birth_x;
        let bdy_p = t.pos_y - t.birth_y;
        let cur_exc2_p = bdx_p * bdx_p + bdy_p * bdy_p;
        let dot_birth_v = bdx_p * t.vel_x + bdy_p * t.vel_y;
        let returning_osc = t.max_excursion2 >= (22.0 * s) * (22.0 * s)
            && (cur_exc2_p < 0.85 * t.max_excursion2 || dot_birth_v < 0.0)
            && (t.pivot_valid > 0 || (t.phase_valid != 0 && t.tremor_coherence > 0.24));
        if (t.is_ballistic != 0 || returning_osc) && (t.vel_x.abs() + t.vel_y.abs() > 11.5 * s) {
            let fwd_grav = if t.is_ballistic != 0 { 0.5 * t.grav_ay_frame } else { 0.0 };
            let fwd_cx = pred_cx + 0.45 * t.vel_x;
            let fwd_cy = pred_cy + 0.45 * t.vel_y + fwd_grav;
            let fx = (((fwd_cx - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
            let fy = (((fwd_cy - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
            if (fx - p0x).abs() + (fy - p0y).abs() >= 3 && n_roots < roots.len() {
                roots[n_roots] = (fx, fy);
                n_roots += 1;
            }
        }
    } else if coasting > 0 {
        let ex = (((t.entry_x - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
        let ey = (((t.entry_y - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
        roots[n_roots] = (ex, ey);
        n_roots += 1;

        let cf = (a.frame_idx.saturating_sub(t.entry_frame).max(1)) as f32;
        let refl_cx = t.entry_x - cf * t.entry_vx;
        let refl_cy = t.entry_y - cf * t.entry_vy;
        let rx = (((refl_cx - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
        let ry = (((refl_cy - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
        roots[n_roots] = (rx, ry);
        n_roots += 1;

        // Linear-chord fallback strictly when CTRV/ballistic curved `pred_cx/pred_cy` is active
        if t.is_ballistic != 0 || t.entry_omega_f.abs() > 0.045 {
            let lin_cx = t.entry_x + cf * t.entry_vx;
            let lin_cy = t.entry_y + cf * t.entry_vy;
            let lx = (((lin_cx - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
            let ly = (((lin_cy - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
            if (lx - p0x).abs() + (ly - p0y).abs() >= 3 && n_roots < roots.len() {
                roots[n_roots] = (lx, ly);
                n_roots += 1;
            }
        }

        // Longitudinal speed-variation roots (braked 0.62x & accelerated 1.20x) under extended occlusion
        let disp_l1 = (pred_cx - t.entry_x).abs() + (pred_cy - t.entry_y).abs();
        if coasting >= 3 && disp_l1 >= 18.0 * s {
            let brk_cx = t.entry_x + 0.62 * (pred_cx - t.entry_x);
            let brk_cy = t.entry_y + 0.62 * (pred_cy - t.entry_y);
            let bx = (((brk_cx - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
            let by = (((brk_cy - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
            if (bx - p0x).abs() + (by - p0y).abs() >= 3 && n_roots < roots.len() {
                roots[n_roots] = (bx, by);
                n_roots += 1;
            }
            let acc_cx = t.entry_x + 1.20 * (pred_cx - t.entry_x);
            let acc_cy = t.entry_y + 1.20 * (pred_cy - t.entry_y);
            let ax = (((acc_cx - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_px);
            let ay = (((acc_cy - half_patch) * inv_pyr_div + 0.5) as i32).clamp(0, max_py);
            if (ax - p0x).abs() + (ay - p0y).abs() >= 3 && n_roots < roots.len() {
                roots[n_roots] = (ax, ay);
                n_roots += 1;
            }
        }
    }

    // Track top 2 pyramid minima separated by >= 5 pyramid pixels (`10 * s` screen px)
    let c0_pyr = sad8x8_pyr160_cutoff_simd128(
        a.pyr_luma.as_ptr(),
        a.pyr_rg.as_ptr(),
        a.pyr_by.as_ptr(),
        a.pyr_eca.as_ptr(),
        p0x as usize,
        p0y as usize,
        t.tpl8_luma.as_ptr(),
        t.tpl8_rg.as_ptr(),
        t.tpl8_by.as_ptr(),
        t.tpl8_w.as_ptr(),
        1e9f32,
    );
    let mut best1 = (p0x, p0y, c0_pyr);
    let mut best2 = (p0x, p0y, 1e9f32);

    for r_idx in 0..n_roots {
        let (rx, ry) = roots[r_idx];
        let rad_steps = if r_idx == 0 { 2i32 } else { 1i32 };
        let step_p = 3i32;
        for sy in -rad_steps..=rad_steps {
            for sx in -rad_steps..=rad_steps {
                if r_idx == 0 && sx == 0 && sy == 0 && rx == p0x && ry == p0y {
                    continue;
                }
                let qx = (rx + sx * step_p).clamp(0, max_px);
                let qy = (ry + sy * step_p).clamp(0, max_py);
                let d1 = (qx - best1.0) * (qx - best1.0) + (qy - best1.1) * (qy - best1.1);
                let cutoff = if d1 < 25 { best1.2 } else { best2.2 };
                let c_pyr = sad8x8_pyr160_cutoff_simd128(
                    a.pyr_luma.as_ptr(),
                    a.pyr_rg.as_ptr(),
                    a.pyr_by.as_ptr(),
                    a.pyr_eca.as_ptr(),
                    qx as usize,
                    qy as usize,
                    t.tpl8_luma.as_ptr(),
                    t.tpl8_rg.as_ptr(),
                    t.tpl8_by.as_ptr(),
                    t.tpl8_w.as_ptr(),
                    cutoff,
                );
                if c_pyr < best1.2 {
                    if d1 >= 25 {
                        best2 = best1;
                    }
                    best1 = (qx, qy, c_pyr);
                } else if d1 >= 25 && c_pyr < best2.2 {
                    best2 = (qx, qy, c_pyr);
                }
            }
        }
    }

    // Refine 1-pyramid-pixel (`2*s` screen px) neighborhood lazily around top 2 coarse minima
    let coarse_pyr = [(best1.0, best1.1, best1.2 != 1e9), (best2.0, best2.1, best2.2 < 1e8)];
    let mut best0_pyr = (best1.0, best1.1);

    // Fine full-resolution snap around the 1..2 pyramid winners
    let mut out = [(0.0f32, 0.0f32, false); 2];
    let mut best_c0 = 1e9f32;
    let step_coarse = scale_px as i32;
    for idx in 0..2usize {
        if !coarse_pyr[idx].2 {
            continue;
        }
        if idx == 1 && coasting == 0 && out[0].2 && best_c0 < 0.24 {
            let sd_stat =
                a.lab_mode == 0 && a.cam_motion_state == 0 && a.scene_gravity_conf < 0.70;
            if (scale_px > 1 && a.lab_mode != 0) || sd_stat {
                break;
            }
            if a.lab_mode == 1 && a.targets[m].freq_valid != 0 {
                let dx_p0 = out[0].0 - pred_cx;
                let dy_p0 = out[0].1 - pred_cy;
                let near_pred0 = dx_p0 * dx_p0 + dy_p0 * dy_p0 <= (4.5 * s) * (4.5 * s);
                if near_pred0 {
                    let mut isolated0 = true;
                    let mut j = 0usize;
                    while j < MAX_TARGETS {
                        if j != m && a.targets[j].active != 0 {
                            let ddx = out[0].0 - a.targets[j].pos_x;
                            let ddy = out[0].1 - a.targets[j].pos_y;
                            if ddx * ddx + ddy * ddy < (28.0 * s) * (28.0 * s) {
                                isolated0 = false;
                                break;
                            }
                        }
                        j += 1;
                    }
                    if isolated0 {
                        break;
                    }
                }
            }
        }
        let cx0 = coarse_pyr[idx].0;
        let cy0 = coarse_pyr[idx].1;
        let mut b_px = cx0;
        let mut b_py = cy0;
        let mut b_c = if idx == 0 { best1.2 } else { 1e9f32 };
        let r_ref = if idx == 0 { 2i32 } else { 1i32 };
        if idx == 0 && cx0 >= 2 && cx0 + 2 <= max_px && cy0 >= 2 && cy0 + 2 <= max_py {
            sad8x8_pyr160_5x5_simd128(
                a.pyr_luma.as_ptr(),
                a.pyr_rg.as_ptr(),
                a.pyr_by.as_ptr(),
                a.pyr_eca.as_ptr(),
                cx0 as usize,
                cy0 as usize,
                t.tpl8_luma.as_ptr(),
                t.tpl8_rg.as_ptr(),
                t.tpl8_by.as_ptr(),
                t.tpl8_w.as_ptr(),
                true,
                b_c,
                &mut b_px,
                &mut b_py,
                &mut b_c,
            );
        } else {
            let mut prev_qy = i32::MIN;
            for dy in -r_ref..=r_ref {
                let qy = (cy0 + dy).clamp(0, max_py);
                if qy == prev_qy {
                    continue;
                }
                prev_qy = qy;
                let mut prev_qx = i32::MIN;
                for dx in -r_ref..=r_ref {
                    if idx == 0 && dx == 0 && dy == 0 {
                        prev_qx = (cx0 + dx).clamp(0, max_px);
                        continue;
                    }
                    let qx = (cx0 + dx).clamp(0, max_px);
                    if qx == prev_qx {
                        continue;
                    }
                    prev_qx = qx;
                    if idx == 1 {
                        let d0 = (qx - best0_pyr.0) * (qx - best0_pyr.0)
                            + (qy - best0_pyr.1) * (qy - best0_pyr.1);
                        if d0 < 25 {
                            continue;
                        }
                    }
                    let c_pyr = sad8x8_pyr160_cutoff_simd128(
                        a.pyr_luma.as_ptr(),
                        a.pyr_rg.as_ptr(),
                        a.pyr_by.as_ptr(),
                        a.pyr_eca.as_ptr(),
                        qx as usize,
                        qy as usize,
                        t.tpl8_luma.as_ptr(),
                        t.tpl8_rg.as_ptr(),
                        t.tpl8_by.as_ptr(),
                        t.tpl8_w.as_ptr(),
                        b_c,
                    );
                    if c_pyr < b_c {
                        b_c = c_pyr;
                        b_px = qx;
                        b_py = qy;
                    }
                }
            }
        }
        if b_c >= 1e8 {
            continue;
        }
        if idx == 0 {
            best0_pyr = (b_px, b_py);
        }
        let mut b_tlx = (b_px * (2 * scale_px as i32)).clamp(1, max_tlx_i);
        let mut b_tly = (b_py * (2 * scale_px as i32)).clamp(1, max_tly_i);
        ensure_chroma!(a, b_tlx as usize, b_tly as usize, patch_dim, patch_dim, frame_w, frame_h);
        let mut best_weca = 0.0f32;
        let mut best_c = simd_ops::sad_patch_zm_half_precomp_cutoff_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            a.eca_res.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            t.tpl_luma.as_ptr(),
            t.tpl_rg.as_ptr(),
            t.tpl_by.as_ptr(),
            t.tpl_weight.as_ptr(),
            tpl_inv_w,
            tpl_mu_t,
            1e9f32,
            &mut best_weca,
            scale_px,
        );
        if idx == 0 && coasting == 0 {
            let p_tlx = ((pred_cx - half_patch) as i32).clamp(1, max_tlx_i);
            let p_tly = ((pred_cy - half_patch) as i32).clamp(1, max_tly_i);
            if p_tlx != b_tlx || p_tly != b_tly {
                ensure_chroma!(a, p_tlx as usize, p_tly as usize, patch_dim, patch_dim, frame_w, frame_h);
                let mut p_weca = 0.0f32;
                let c_pred = simd_ops::sad_patch_zm_half_precomp_cutoff_simd128(
                    a.luma.as_ptr(),
                    a.chroma_rg.as_ptr(),
                    a.chroma_by.as_ptr(),
                    a.eca_res.as_ptr(),
                    frame_w,
                    p_tlx as usize,
                    p_tly as usize,
                    t.tpl_luma.as_ptr(),
                    t.tpl_rg.as_ptr(),
                    t.tpl_by.as_ptr(),
                    t.tpl_weight.as_ptr(),
                    tpl_inv_w,
                    tpl_mu_t,
                    best_c,
                    &mut p_weca,
                    scale_px,
                );
                if c_pred < best_c {
                    b_tlx = p_tlx;
                    b_tly = p_tly;
                    best_c = c_pred;
                    best_weca = p_weca;
                }
            }
        }
        let mut st4_centered = false;
        let mut st1_centered = false;
        let skip_idx1_ring = idx == 1
            && (scale_px > 1 || a.lab_mode == 1)
            && ((out[0].2 && best_c0 < 0.26 && best_c > (best_c0 + 0.14).max(0.36))
                || best_c > 0.62);
        for st in [2 * step_coarse, 1i32] {
            if skip_idx1_ring {
                break;
            }
            let cx0 = b_tlx;
            let cy0 = b_tly;
            ensure_chroma!(
                a,
                (cx0 - st).max(1) as usize,
                (cy0 - st).max(1) as usize,
                patch_dim + 2 * (st as usize),
                patch_dim + 2 * (st as usize),
                frame_w,
                frame_h
            );
            if (st == (2 * scale_px) as i32 || st == 1)
                && scale_px <= 2
                && cx0 - st >= 1
                && cx0 + st <= max_tlx_i
                && cy0 - st >= 1
                && cy0 + st <= max_tly_i
            {
                let mut r_dx = 0i32;
                let mut r_dy = 0i32;
                let mut r_weca = best_weca;
                if st == 1 {
                    simd_ops::sad_patch_zm_half_st1_ring_precomp_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        cx0 as usize,
                        cy0 as usize,
                        t.tpl_luma.as_ptr(),
                        t.tpl_rg.as_ptr(),
                        t.tpl_by.as_ptr(),
                        t.tpl_weight.as_ptr(),
                        tpl_inv_w,
                        tpl_mu_t,
                        scale_px,
                        &mut r_dx,
                        &mut r_dy,
                        &mut best_c,
                        &mut r_weca,
                    );
                } else {
                    simd_ops::sad_patch_zm_half_ring3x3_precomp_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        cx0 as usize,
                        cy0 as usize,
                        t.tpl_luma.as_ptr(),
                        t.tpl_rg.as_ptr(),
                        t.tpl_by.as_ptr(),
                        t.tpl_weight.as_ptr(),
                        tpl_inv_w,
                        tpl_mu_t,
                        scale_px,
                        &mut r_dx,
                        &mut r_dy,
                        &mut best_c,
                        &mut r_weca,
                    );
                }
                best_weca = r_weca;
                if r_dx == 0 && r_dy == 0 {
                    if st == 1 {
                        st1_centered = true;
                    } else {
                        st4_centered = true;
                    }
                } else {
                    b_tlx = cx0 + r_dx * st;
                    b_tly = cy0 + r_dy * st;
                    if st == 1 {
                        st4_centered = false;
                        st1_centered = false;
                    }
                }
                continue;
            }
            let mut moved_fb = false;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let qx = (cx0 + dx * st).clamp(1, max_tlx_i) as usize;
                    let qy = (cy0 + dy * st).clamp(1, max_tly_i) as usize;
                    let mut rweca = 0.0f32;
                    let rc = simd_ops::sad_patch_zm_half_precomp_cutoff_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        qx,
                        qy,
                        t.tpl_luma.as_ptr(),
                        t.tpl_rg.as_ptr(),
                        t.tpl_by.as_ptr(),
                        t.tpl_weight.as_ptr(),
                        tpl_inv_w,
                        tpl_mu_t,
                        best_c,
                        &mut rweca,
                        scale_px,
                    );
                    if rc < best_c {
                        best_c = rc;
                        best_weca = rweca;
                        b_tlx = qx as i32;
                        b_tly = qy as i32;
                        moved_fb = true;
                    }
                }
            }
            if moved_fb && st == 1 {
                st4_centered = false;
                st1_centered = false;
            }
        }

        if best_c < 1e8 {
            if !out_cache.is_null() {
                (*out_cache)[idx] = PropFineCache {
                    tlx: b_tlx as i16,
                    tly: b_tly as i16,
                    flags: 1 | ((st4_centered as u8) << 1) | ((st1_centered as u8) << 2),
                    half_c: best_c,
                    weca: best_weca,
                };
            }
            if (t.anchor_sig & (1 << 31)) != 0
                && has_opposite_chroma_pole_patch_simd128(
                    a.chroma_rg.as_ptr(),
                    a.chroma_by.as_ptr(),
                    frame_w,
                    b_tlx as usize,
                    b_tly as usize,
                    scale_px,
                    t.anchor_sig,
                )
            {
                continue;
            }
            let fcx = (b_tlx as f32) + half_patch;
            let fcy = (b_tly as f32) + half_patch;
            if idx == 1 && out[0].2 {
                let ddx = fcx - out[0].0;
                let ddy = fcy - out[0].1;
                if ddx * ddx + ddy * ddy < (11.0 * s) * (11.0 * s) {
                    continue;
                }
            }
            out[idx] = (fcx, fcy, true);
            if idx == 0 {
                best_c0 = best_c;
            }
        }
    }
    out
}

/// Evaluate & locally snap (`±2*s -> ±1` px) hypothesis `(hx, hy)` for track `m`, returning
/// `(best_tlx, best_tly, raw_cost, app_err, anchor_l1, w_eca, cand_sig, valid)`.
#[inline(never)]
unsafe fn eval_track_hypothesis_fine_cached(
    a: &mut Arena,
    m: usize,
    hx: f32,
    hy: f32,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
    max_base_c: f32,
    tpl_inv_w: f32,
    tpl_mu_t: f32,
    prop_cache: *const [PropFineCache; 2],
) -> (usize, usize, f32, f32, f32, f32, u32, bool) {
    let half_patch = 8.0 * s;
    let patch_dim = 16 * scale_px;
    let max_tlx_i = (frame_w - patch_dim - 1) as i32;
    let max_tly_i = (frame_h - patch_dim - 1) as i32;

    let init_tlx = ((hx - half_patch + 0.5) as i32).clamp(1, max_tlx_i);
    let init_tly = ((hy - half_patch + 0.5) as i32).clamp(1, max_tly_i);
    let mut b_tlx = init_tlx;
    let mut b_tly = init_tly;
    let t = &a.targets[m];
    let mut best_weca = 0.0f32;
    if scale_px <= 1 {
        let full_inv_w = a.ic_lk_cache[m].full_inv_w;
        let full_mu_t = a.ic_lk_cache[m].full_mu_t;
        let center_cutoff = (max_base_c + 0.55).min(1.15) + 1e-4;
        ensure_chroma!(a, b_tlx as usize, b_tly as usize, patch_dim, patch_dim, frame_w, frame_h);
        let (mut best_base_c, c_weca) = simd_ops::sad_patch_zm_full_sd_eca_cutoff_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            a.eca_res.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            t.tpl_luma.as_ptr(),
            t.tpl_rg.as_ptr(),
            t.tpl_by.as_ptr(),
            t.tpl_weight.as_ptr(),
            full_inv_w,
            full_mu_t,
            center_cutoff,
        );
        best_weca = c_weca;
        if best_base_c >= center_cutoff {
            return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
        }
        let mut prop_hit = false;
        if a.lab_mode == 1 && !prop_cache.is_null() {
            let pc_arr = &*prop_cache;
            for k in 0..2usize {
                let pc = &pc_arr[k];
                if (pc.flags & 1) != 0
                    && (init_tlx - (pc.tlx as i32)).abs() <= 1
                    && (init_tly - (pc.tly as i32)).abs() <= 1
                {
                    prop_hit = true;
                    break;
                }
            }
        }
        let s_step = (2 * scale_px) as i32;
        for st in [s_step, 1i32] {
            if prop_hit && st == s_step {
                continue;
            }
            if st == 1 && best_base_c > (max_base_c + 0.42).min(0.98) {
                return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
            }
            let cx0 = b_tlx;
            let cy0 = b_tly;
            ensure_chroma!(
                a,
                (cx0 - st).max(1) as usize,
                (cy0 - st).max(1) as usize,
                patch_dim + 2 * (st as usize),
                patch_dim + 2 * (st as usize),
                frame_w,
                frame_h
            );
            if scale_px == 1
                && (st == 1 || st == 2)
                && cx0 - st >= 1
                && cx0 + st <= max_tlx_i
                && cy0 - st >= 1
                && cy0 + st <= max_tly_i
            {
                if st == 2 {
                    simd_ops::sad_patch_zm_full_sd_ring3x3_simd128::<2>(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        cx0 as usize,
                        cy0 as usize,
                        t.tpl_luma.as_ptr(),
                        t.tpl_rg.as_ptr(),
                        t.tpl_by.as_ptr(),
                        t.tpl_weight.as_ptr(),
                        full_inv_w,
                        full_mu_t,
                        &mut b_tlx,
                        &mut b_tly,
                        &mut best_base_c,
                        &mut best_weca,
                    );
                } else {
                    simd_ops::sad_patch_zm_full_sd_ring3x3_simd128::<1>(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        cx0 as usize,
                        cy0 as usize,
                        t.tpl_luma.as_ptr(),
                        t.tpl_rg.as_ptr(),
                        t.tpl_by.as_ptr(),
                        t.tpl_weight.as_ptr(),
                        full_inv_w,
                        full_mu_t,
                        &mut b_tlx,
                        &mut b_tly,
                        &mut best_base_c,
                        &mut best_weca,
                    );
                }
                continue;
            }
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let qx = (cx0 + dx * st).clamp(1, max_tlx_i) as usize;
                    let qy = (cy0 + dy * st).clamp(1, max_tly_i) as usize;
                    let (rc, rweca) = simd_ops::sad_patch_zm_full_sd_eca_cutoff_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        qx,
                        qy,
                        t.tpl_luma.as_ptr(),
                        t.tpl_rg.as_ptr(),
                        t.tpl_by.as_ptr(),
                        t.tpl_weight.as_ptr(),
                        full_inv_w,
                        full_mu_t,
                        best_base_c,
                    );
                    if rc < best_base_c {
                        best_base_c = rc;
                        best_weca = rweca;
                        b_tlx = qx as i32;
                        b_tly = qy as i32;
                    }
                }
            }
        }
        if best_base_c >= max_base_c {
            return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
        }
        let cand_sig = haar_morton_signature_hyst_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            scale_px,
            t.anchor_sig,
            true,
        );
        if has_opposite_chroma_pole(t.anchor_sig, cand_sig) {
            return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, cand_sig, false);
        }
        let val = padic_ultrametric_val(t.anchor_sig, cand_sig);
        let d2 = padic_distance_from_val(val);
        let best_rc = best_base_c + 0.55 * d2;
        let app_err = best_rc - 0.55 * d2 + 0.16 * best_weca.min(0.85);
        let anchor_l1 = weighted_anchor_zm_l1_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            t.anchor_luma.as_ptr(),
            t.anchor_rg.as_ptr(),
            t.anchor_by.as_ptr(),
            t.tpl_weight.as_ptr(),
            scale_px,
        );
        return (
            b_tlx as usize,
            b_tly as usize,
            best_rc,
            app_err,
            anchor_l1,
            best_weca,
            cand_sig,
            true,
        );
    }
    let pre_cutoff = (max_base_c + 0.38).min(0.95) + 1e-4;
    let mut skip_st4 = false;
    let mut skip_st1 = false;
    let mut cache_hit = false;
    if !prop_cache.is_null() {
        let cs = &*prop_cache;
        for ci in 0..2usize {
            let c = cs[ci];
            if (c.flags & 1) != 0 && (c.tlx as i32) == init_tlx && (c.tly as i32) == init_tly {
                cache_hit = true;
                skip_st4 = (c.flags & 2) != 0;
                skip_st1 = (c.flags & 4) != 0;
                if c.half_c > (max_base_c + 0.38).min(0.95) {
                    return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
                }
                best_weca = c.weca;
                break;
            }
        }
    }
    // Fast half-row center pre-screen eliminates distant/mismatched cross-track hypotheses in 1 SIMD call
    let mut best_half_c = if cache_hit {
        let cs = &*prop_cache;
        if (cs[0].flags & 1) != 0 && (cs[0].tlx as i32) == init_tlx && (cs[0].tly as i32) == init_tly {
            cs[0].half_c
        } else {
            cs[1].half_c
        }
    } else {
        ensure_chroma!(a, b_tlx as usize, b_tly as usize, patch_dim, patch_dim, frame_w, frame_h);
        simd_ops::sad_patch_zm_half_precomp_cutoff_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            a.eca_res.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            t.tpl_luma.as_ptr(),
            t.tpl_rg.as_ptr(),
            t.tpl_by.as_ptr(),
            t.tpl_weight.as_ptr(),
            tpl_inv_w,
            tpl_mu_t,
            pre_cutoff,
            &mut best_weca,
            scale_px,
        )
    };
    if best_half_c > (max_base_c + 0.38).min(0.95) {
        return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
    }
    let s_step = (2 * scale_px) as i32;
    for st in [s_step, 1i32] {
        if st == 1 {
            if best_half_c > (max_base_c + 0.18).min(0.82) {
                return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
            }
            if scale_px == 2 && !skip_st1 && !prop_cache.is_null() {
                let cs = &*prop_cache;
                for ci in 0..2usize {
                    let c = cs[ci];
                    if (c.flags & 5) == 5 && (c.tlx as i32) == b_tlx && (c.tly as i32) == b_tly {
                        best_half_c = c.half_c;
                        best_weca = c.weca;
                        skip_st1 = true;
                        break;
                    }
                }
            }
            if skip_st1 {
                continue;
            }
        } else if skip_st4 {
            continue;
        }
        let cx0 = b_tlx;
        let cy0 = b_tly;
        ensure_chroma!(
            a,
            (cx0 - st).max(1) as usize,
            (cy0 - st).max(1) as usize,
            patch_dim + 2 * (st as usize),
            patch_dim + 2 * (st as usize),
            frame_w,
            frame_h
        );
        if (st == s_step || st == 1)
            && scale_px <= 2
            && cx0 - st >= 1
            && cx0 + st <= max_tlx_i
            && cy0 - st >= 1
            && cy0 + st <= max_tly_i
        {
            let mut r_dx = 0i32;
            let mut r_dy = 0i32;
            if st == 1 {
                simd_ops::sad_patch_zm_half_st1_ring_precomp_simd128(
                    a.luma.as_ptr(),
                    a.chroma_rg.as_ptr(),
                    a.chroma_by.as_ptr(),
                    a.eca_res.as_ptr(),
                    frame_w,
                    cx0 as usize,
                    cy0 as usize,
                    t.tpl_luma.as_ptr(),
                    t.tpl_rg.as_ptr(),
                    t.tpl_by.as_ptr(),
                    t.tpl_weight.as_ptr(),
                    tpl_inv_w,
                    tpl_mu_t,
                    scale_px,
                    &mut r_dx,
                    &mut r_dy,
                    &mut best_half_c,
                    &mut best_weca,
                );
            } else {
                let mut search_c = best_half_c;
                simd_ops::sad_patch_zm_half_ring3x3_precomp_simd128(
                    a.luma.as_ptr(),
                    a.chroma_rg.as_ptr(),
                    a.chroma_by.as_ptr(),
                    a.eca_res.as_ptr(),
                    frame_w,
                    cx0 as usize,
                    cy0 as usize,
                    t.tpl_luma.as_ptr(),
                    t.tpl_rg.as_ptr(),
                    t.tpl_by.as_ptr(),
                    t.tpl_weight.as_ptr(),
                    tpl_inv_w,
                    tpl_mu_t,
                    scale_px,
                    &mut r_dx,
                    &mut r_dy,
                    &mut search_c,
                    &mut best_weca,
                );
                if r_dx != 0 || r_dy != 0 {
                    best_half_c = search_c;
                    skip_st1 = false;
                }
            }
            b_tlx = cx0 + r_dx * st;
            b_tly = cy0 + r_dy * st;
            continue;
        }
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let qx = (cx0 + dx * st).clamp(1, max_tlx_i) as usize;
                let qy = (cy0 + dy * st).clamp(1, max_tly_i) as usize;
                let mut rweca = 0.0f32;
                let rc = simd_ops::sad_patch_zm_half_precomp_cutoff_simd128(
                    a.luma.as_ptr(),
                    a.chroma_rg.as_ptr(),
                    a.chroma_by.as_ptr(),
                    a.eca_res.as_ptr(),
                    frame_w,
                    qx,
                    qy,
                    t.tpl_luma.as_ptr(),
                    t.tpl_rg.as_ptr(),
                    t.tpl_by.as_ptr(),
                    t.tpl_weight.as_ptr(),
                    tpl_inv_w,
                    tpl_mu_t,
                    best_half_c,
                    &mut rweca,
                    scale_px,
                );
                if rc < best_half_c {
                    best_half_c = rc;
                    best_weca = rweca;
                    b_tlx = qx as i32;
                    b_tly = qy as i32;
                }
            }
        }
    }
    let full_inv_w = a.ic_lk_cache[m].full_inv_w;
    let full_mu_t = a.ic_lk_cache[m].full_mu_t;
    let (best_base_c, anchor_l1, best_sig) = if scale_px == 2 {
        let mut base_c = 0.0f32;
        let mut anc_l1 = 0.0f32;
        let mut cand_sig = 0u32;
        let status = eval_winner_patch_hd_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            a.eca_res.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            t.tpl_luma.as_ptr(),
            t.tpl_rg.as_ptr(),
            t.tpl_by.as_ptr(),
            t.tpl_weight.as_ptr(),
            t.anchor_luma.as_ptr(),
            t.anchor_rg.as_ptr(),
            t.anchor_by.as_ptr(),
            t.anchor_sig,
            full_inv_w,
            full_mu_t,
            t.anchor_mu_a,
            t.anchor_std_a,
            max_base_c,
            &mut base_c,
            &mut anc_l1,
            &mut best_weca,
            &mut cand_sig,
        );
        if status == 0 {
            return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
        }
        if status == 1 {
            return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, cand_sig, false);
        }
        (base_c, anc_l1, cand_sig)
    } else {
        let base_c = sad_patch_zm_score_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            a.eca_res.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            t.tpl_luma.as_ptr(),
            t.tpl_rg.as_ptr(),
            t.tpl_by.as_ptr(),
            t.tpl_weight.as_ptr(),
            &mut best_weca,
            scale_px,
        );
        if base_c >= 1.18 {
            return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, t.curr_sig, false);
        }
        let cand_sig = haar_morton_signature_hyst_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            scale_px,
            t.anchor_sig,
            true,
        );
        if has_opposite_chroma_pole(t.anchor_sig, cand_sig) {
            return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, cand_sig, false);
        }
        let anc_l1 = weighted_anchor_zm_l1_simd128(
            a.luma.as_ptr(),
            a.chroma_rg.as_ptr(),
            a.chroma_by.as_ptr(),
            frame_w,
            b_tlx as usize,
            b_tly as usize,
            t.anchor_luma.as_ptr(),
            t.anchor_rg.as_ptr(),
            t.anchor_by.as_ptr(),
            t.tpl_weight.as_ptr(),
            scale_px,
        );
        (base_c, anc_l1, cand_sig)
    };

    let val = padic_ultrametric_val(t.anchor_sig, best_sig);
    let d2 = padic_distance_from_val(val);
    let best_rc = best_base_c + 0.55 * d2;
    let app_err = best_rc - 0.55 * d2 + 0.16 * best_weca.min(0.85);

    (
        b_tlx as usize,
        b_tly as usize,
        best_rc,
        app_err,
        anchor_l1,
        best_weca,
        best_sig,
        true,
    )
}

/// True Non-Causal Backward Ring Re-Tracker:
/// Walks backward across `a.rgba_ring` (`f_idx - 1 .. f_idx - max_back`) upon occlusion re-emergence
/// or track confirmation, recovering real sub-pixel detections inside `ring_raw_x/y` and `ring_vis`
/// BEFORE `solve_noncausal_fista_target` runs and before delayed display frame `t - K` renders.
#[inline(never)]
unsafe fn backtrack_gap_in_rgba_ring(
    a: &mut Arena,
    m: usize,
    f_idx: u32,
    max_back: usize,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
    use_entry_guide: bool,
) {
    if max_back == 0 || a.ring_cursor == 0 {
        return;
    }
    let roi_dim = 48 * scale_px; // 48 in SD, 96 in HD (<= MAX_ROI_DIM)
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;
    let max_lx = (roi_dim - patch_dim - 2) as i32;
    let max_ly = (roi_dim - patch_dim - 2) as i32;
    let steps_coarse = [8 * scale_px as i32, 4 * scale_px as i32, 0];
    let (full_inv_w, full_mu_t) = tpl_full_stats_simd128(
        a.targets[m].tpl_luma.as_ptr(),
        a.targets[m].tpl_weight.as_ptr(),
    );

    let exit_x = a.targets[m].pos_x;
    let exit_y = a.targets[m].pos_y;
    let exit_vx = a.targets[m].vel_x;
    let exit_vy = a.targets[m].vel_y;
    let mut cur_x = exit_x;
    let mut cur_y = exit_y;
    let mut last_hit_b = 0usize;
    let mut last_cam_back_x = 0.0f32;
    let mut last_cam_back_y = 0.0f32;
    let ent_x = a.targets[m].entry_x;
    let ent_y = a.targets[m].entry_y;
    let orig_birth = a.targets[m].birth_frame;
    let denom_b = (max_back + 1) as f32;

    for b in 1..=max_back {
        if a.ring_cursor < (b as u64) {
            break;
        }
        let past_f = f_idx.wrapping_sub(b as u32);
        if a.last_discont_frame > 0 && f_idx.wrapping_sub(a.last_discont_frame) < (b as u32) {
            break;
        }
        if !use_entry_guide && f_idx.wrapping_sub(orig_birth).saturating_add(2) < (b as u32) {
            break;
        }
        let slot_b = ring_slot_back_cursor(a.ring_cursor, b);
        let cam_back_x = a.cum_cam_x - a.hist_cam_x[(past_f as usize) & HIST_MASK];
        let cam_back_y = a.cum_cam_y - a.hist_cam_y[(past_f as usize) & HIST_MASK];
        if a.targets[m].ring_vis[slot_b] != 0 {
            if use_entry_guide || a.lab_mode != 2 {
                break;
            }
            cur_x = a.targets[m].ring_raw_x[slot_b];
            cur_y = a.targets[m].ring_raw_y[slot_b];
            last_cam_back_x = cam_back_x;
            last_cam_back_y = cam_back_y;
            last_hit_b = b;
            continue;
        }

        let (guide_x, guide_y) = if use_entry_guide {
            let alpha = (b as f32) / denom_b;
            let tau = 1.0 - alpha;
            let sag_y = if a.targets[m].is_ballistic != 0 {
                -0.5 * a.targets[m].grav_ay_frame * denom_b * denom_b * tau * alpha
            } else {
                0.0
            };
            (
                (1.0 - alpha) * exit_x + alpha * ent_x + cam_back_x,
                (1.0 - alpha) * exit_y + alpha * ent_y + sag_y + cam_back_y,
            )
        } else {
            let step_b = ((b - last_hit_b).max(1)) as f32;
            let b0 = last_hit_b as f32;
            let bf = b as f32;
            let step_grav = if a.targets[m].is_ballistic != 0 {
                0.5 * a.targets[m].grav_ay_frame * (bf * bf - b0 * b0)
            } else {
                0.0
            };
            (
                cur_x - exit_vx * step_b + (cam_back_x - last_cam_back_x),
                cur_y - exit_vy * step_b + step_grav + (cam_back_y - last_cam_back_y),
            )
        };

        let roi_x = ((guide_x as i32) - (roi_dim as i32) / 2)
            .clamp(0, (frame_w - roi_dim) as i32) as usize;
        let roi_y = ((guide_y as i32) - (roi_dim as i32) / 2)
            .clamp(0, (frame_h - roi_dim) as i32) as usize;

        rgba_roi_to_planar_simd128(
            a.rgba_ring[slot_b].as_ptr(),
            frame_w,
            roi_x,
            roi_y,
            roi_dim,
            roi_dim,
            a.roi_luma.as_mut_ptr(),
            a.roi_rg.as_mut_ptr(),
            a.roi_by.as_mut_ptr(),
        );

        let base_lx = ((guide_x - (roi_x as f32) - half_patch + 0.5) as i32).clamp(2, max_lx);
        let base_ly = ((guide_y - (roi_y as f32) - half_patch + 0.5) as i32).clamp(2, max_ly);

        let anchor_sig = a.targets[m].anchor_sig;
        let mut best_lx = base_lx;
        let mut best_ly = base_ly;
        let mut best_base_c = sad_patch_zm_full_precomp_cutoff_simd128(
            a.roi_luma.as_ptr(),
            a.roi_rg.as_ptr(),
            a.roi_by.as_ptr(),
            roi_dim,
            base_lx as usize,
            base_ly as usize,
            a.targets[m].tpl_luma.as_ptr(),
            a.targets[m].tpl_rg.as_ptr(),
            a.targets[m].tpl_by.as_ptr(),
            a.targets[m].tpl_weight.as_ptr(),
            full_inv_w,
            full_mu_t,
            1e9f32,
            scale_px,
        );

        for &dy in &[-steps_coarse[0], -steps_coarse[1], 0, steps_coarse[1], steps_coarse[0]] {
            for &dx in &[-steps_coarse[0], -steps_coarse[1], 0, steps_coarse[1], steps_coarse[0]] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let lx = (base_lx + dx).clamp(2, max_lx);
                let ly = (base_ly + dy).clamp(2, max_ly);
                if lx == base_lx && ly == base_ly {
                    continue;
                }
                let rc = sad_patch_zm_full_precomp_cutoff_simd128(
                    a.roi_luma.as_ptr(),
                    a.roi_rg.as_ptr(),
                    a.roi_by.as_ptr(),
                    roi_dim,
                    lx as usize,
                    ly as usize,
                    a.targets[m].tpl_luma.as_ptr(),
                    a.targets[m].tpl_rg.as_ptr(),
                    a.targets[m].tpl_by.as_ptr(),
                    a.targets[m].tpl_weight.as_ptr(),
                    full_inv_w,
                    full_mu_t,
                    best_base_c,
                    scale_px,
                );
                if rc < best_base_c {
                    best_base_c = rc;
                    best_lx = lx;
                    best_ly = ly;
                }
            }
        }

        for &st in &[2 * (scale_px as i32), 1i32] {
            let cx0 = best_lx;
            let cy0 = best_ly;
            for dy in [-st, 0, st] {
                for dx in [-st, 0, st] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let lx = (cx0 + dx).clamp(2, max_lx);
                    let ly = (cy0 + dy).clamp(2, max_ly);
                    if lx == cx0 && ly == cy0 {
                        continue;
                    }
                    let rc = sad_patch_zm_full_precomp_cutoff_simd128(
                        a.roi_luma.as_ptr(),
                        a.roi_rg.as_ptr(),
                        a.roi_by.as_ptr(),
                        roi_dim,
                        lx as usize,
                        ly as usize,
                        a.targets[m].tpl_luma.as_ptr(),
                        a.targets[m].tpl_rg.as_ptr(),
                        a.targets[m].tpl_by.as_ptr(),
                        a.targets[m].tpl_weight.as_ptr(),
                        full_inv_w,
                        full_mu_t,
                        best_base_c,
                        scale_px,
                    );
                    if rc < best_base_c {
                        best_base_c = rc;
                        best_lx = lx;
                        best_ly = ly;
                    }
                }
            }
        }

        let max_bt_err = if use_entry_guide { 0.22 } else { 0.135 };
        if best_base_c >= max_bt_err + 1e-4 {
            continue;
        }
        let best_sig = haar_morton_signature_hyst_simd128(
            a.roi_luma.as_ptr(),
            a.roi_rg.as_ptr(),
            a.roi_by.as_ptr(),
            roi_dim,
            best_lx as usize,
            best_ly as usize,
            scale_px,
            anchor_sig,
            true,
        );
        if has_opposite_chroma_pole(anchor_sig, best_sig) {
            continue;
        }
        let app_err = best_base_c;
        let anchor_l1 = weighted_anchor_zm_l1_simd128(
            a.roi_luma.as_ptr(),
            a.roi_rg.as_ptr(),
            a.roi_by.as_ptr(),
            roi_dim,
            best_lx as usize,
            best_ly as usize,
            a.targets[m].anchor_luma.as_ptr(),
            a.targets[m].anchor_rg.as_ptr(),
            a.targets[m].anchor_by.as_ptr(),
            a.targets[m].tpl_weight.as_ptr(),
            scale_px,
        );

        // Strict acceptance so genuinely occluded frames (inside pillar/behind book) remain vis == 0
        // for Hermite spline bridging while missed visible frames recover sub-pixel lock!
        if app_err < max_bt_err && anchor_l1 < max_bt_err {
            let tl_ptr = a.targets[m].tpl_luma.as_ptr();
            let trg_ptr = a.targets[m].tpl_rg.as_ptr();
            let tby_ptr = a.targets[m].tpl_by.as_ptr();
            let tw_ptr = a.targets[m].tpl_weight.as_ptr();
            let (sdx, sdy) = refine_subpixel_ic_lk_cached(
                a.roi_luma.as_ptr(),
                a.roi_rg.as_ptr(),
                a.roi_by.as_ptr(),
                roi_dim,
                best_lx as usize,
                best_ly as usize,
                tl_ptr,
                trg_ptr,
                tby_ptr,
                tw_ptr,
                scale_px,
                &mut a.ic_lk_cache[m],
                &mut a.targets[m].tpl_grad_dirty,
            );
            let fx = (roi_x + best_lx as usize) as f32 + half_patch + sdx;
            let fy = (roi_y + best_ly as usize) as f32 + half_patch + sdy;
            let step_span = ((b - last_hit_b).max(1)) as f32;
            a.targets[m].ring_raw_x[slot_b] = fx;
            a.targets[m].ring_raw_y[slot_b] = fy;
            a.targets[m].ring_vx[slot_b] = ((cur_x - last_cam_back_x) - (fx - cam_back_x)) / step_span;
            a.targets[m].ring_vy[slot_b] = ((cur_y - last_cam_back_y) - (fy - cam_back_y)) / step_span;
            a.targets[m].ring_sig[slot_b] = best_sig;
            a.targets[m].ring_vis[slot_b] = if use_entry_guide && max_back >= 2 { 2 } else { 1 };
            let arc_sb = (past_f as usize) & (ARC_HIST_LEN - 1);
            a.targets[m].arc_raw_x[arc_sb] = fx;
            a.targets[m].arc_raw_y[arc_sb] = fy;
            a.targets[m].arc_vis[arc_sb] = 1;
            if past_f < a.targets[m].birth_frame {
                a.targets[m].birth_frame = past_f;
            }
            cur_x = fx;
            cur_y = fy;
            last_cam_back_x = cam_back_x;
            last_cam_back_y = cam_back_y;
            last_hit_b = b;
        } else if !use_entry_guide {
            break;
        }
    }
}

#[inline(always)]
fn sample_rgba_bilinear(
    rgba: &[u8; MAX_FRAME_RGBA_BYTES],
    frame_w: usize,
    frame_h: usize,
    fx: f32,
    fy: f32,
) -> (f32, f32, f32, f32) {
    let x = fx.clamp(0.0, (frame_w - 2) as f32);
    let y = fy.clamp(0.0, (frame_h - 2) as f32);
    let ix = x as usize;
    let iy = y as usize;
    let tx = x - ix as f32;
    let ty = y - iy as f32;

    let w00 = (1.0 - tx) * (1.0 - ty) * (1.0 / 255.0);
    let w10 = tx * (1.0 - ty) * (1.0 / 255.0);
    let w01 = (1.0 - tx) * ty * (1.0 / 255.0);
    let w11 = tx * ty * (1.0 / 255.0);

    let i00 = (iy * frame_w + ix) * 4;
    let i10 = i00 + 4;
    let i01 = i00 + frame_w * 4;
    let i11 = i01 + 4;

    let r = rgba[i00] as f32 * w00
        + rgba[i10] as f32 * w10
        + rgba[i01] as f32 * w01
        + rgba[i11] as f32 * w11;
    let g = rgba[i00 + 1] as f32 * w00
        + rgba[i10 + 1] as f32 * w10
        + rgba[i01 + 1] as f32 * w01
        + rgba[i11 + 1] as f32 * w11;
    let b = rgba[i00 + 2] as f32 * w00
        + rgba[i10 + 2] as f32 * w10
        + rgba[i01 + 2] as f32 * w01
        + rgba[i11 + 2] as f32 * w11;
    let luma = 0.299 * r + 0.587 * g + 0.114 * b;
    (r, g, b, luma)
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum BodyConnectivity {
    SeparatedByValley,
    Unsplit,
}

#[inline]
fn border_axis_exiting(dist_from_clamp: f32, vn_out: f32, vt_par: f32, s: f32) -> bool {
    if dist_from_clamp < 0.0 {
        return true;
    }
    let outgoing = dist_from_clamp <= 1.8 * s
        && vn_out >= 0.20 * s
        && dist_from_clamp - 3.0 * vn_out <= 0.5 * s;
    let clamped_stop = dist_from_clamp <= 1.25 * s
        && vn_out >= -0.05 * s
        && vt_par <= 1.15 * s;
    outgoing || clamped_stop
}

#[inline]
fn is_border_clamped_exit(
    entry_x: f32,
    entry_y: f32,
    entry_vx: f32,
    entry_vy: f32,
    frame_w: usize,
    frame_h: usize,
    half_patch: f32,
    s: f32,
) -> bool {
    let max_cx = (frame_w as f32) - half_patch;
    let max_cy = (frame_h as f32) - half_patch;
    let abs_vx = entry_vx.abs();
    let abs_vy = entry_vy.abs();
    border_axis_exiting(entry_x - half_patch, -entry_vx, abs_vy, s)
        || border_axis_exiting(max_cx - entry_x, entry_vx, abs_vy, s)
        || border_axis_exiting(entry_y - half_patch, -entry_vy, abs_vx, s)
        || border_axis_exiting(max_cy - entry_y, entry_vy, abs_vx, s)
}

#[inline]
fn sample_pyr_bilinear(pyr_eca: &[f32; PYR_PIXELS], u: f32, v: f32) -> f32 {
    let uc = u.clamp(0.0, (PYR_W - 1) as f32);
    let vc = v.clamp(0.0, (PYR_H - 1) as f32);
    let x0 = (uc as usize).min(PYR_W - 2);
    let y0 = (vc as usize).min(PYR_H - 2);
    let fx = (uc - (x0 as f32)).clamp(0.0, 1.0);
    let fy = (vc - (y0 as f32)).clamp(0.0, 1.0);
    let r0 = y0 * PYR_W + x0;
    let r1 = r0 + PYR_W;
    let top = pyr_eca[r0] * (1.0 - fx) + pyr_eca[r0 + 1] * fx;
    let bot = pyr_eca[r1] * (1.0 - fx) + pyr_eca[r1 + 1] * fx;
    top * (1.0 - fy) + bot * fy
}

#[inline]
fn sample_pyr_max3x3(pyr_eca: &[f32; PYR_PIXELS], u: f32, v: f32) -> f32 {
    let cx = ((u + 0.5) as isize).clamp(0, (PYR_W - 1) as isize) as usize;
    let cy = ((v + 0.5) as isize).clamp(0, (PYR_H - 1) as isize) as usize;
    let x0 = cx.saturating_sub(1);
    let x1 = (cx + 1).min(PYR_W - 1);
    let y0 = cy.saturating_sub(1);
    let y1 = (cy + 1).min(PYR_H - 1);
    let mut best = 0.0f32;
    for y in y0..=y1 {
        let row = y * PYR_W;
        for x in x0..=x1 {
            let val = pyr_eca[row + x];
            if val > best {
                best = val;
            }
        }
    }
    best
}

// Perpendicular-only 3-tap ribbon plus 3-channel opponent-color background valley verification
// resolves narrow 4*s gaps between adjacent bodies while never splitting uniform or isoluminant interiors.
fn classify_eca_connectivity(
    pyr_eca: &[f32; PYR_PIXELS],
    luma: &[f32],
    rgba: &[u8],
    bg_luma: &[f32],
    bg_stat_streak: &[u8],
    frame_w: usize,
    frame_h: usize,
    s: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> BodyConnectivity {
    let inv_cell = 0.5 / s;
    let u0 = x0 * inv_cell;
    let v0 = y0 * inv_cell;
    let u1 = x1 * inv_cell;
    let v1 = y1 * inv_cell;
    let du = u1 - u0;
    let dv = v1 - v0;
    let d_pyr2 = du * du + dv * dv;
    if d_pyr2 < 7.0 * 7.0 || d_pyr2 > 13.0 * 13.0 {
        return BodyConnectivity::Unsplit;
    }
    let e0 = sample_pyr_max3x3(pyr_eca, u0, v0);
    let e1 = sample_pyr_max3x3(pyr_eca, u1, v1);
    let end_min = e0.min(e1);
    if end_min < 0.025 {
        return BodyConnectivity::Unsplit;
    }
    let d_pyr = wasm_sqrtf(d_pyr2);
    let nx = -dv / d_pyr;
    let ny = du / d_pyr;
    let step_px = (s as isize).max(1);
    let max_xi = (frame_w as isize) - 1;
    let max_yi = (frame_h as isize) - 1;
    let sample_color_avg = |ex: f32, ey: f32| -> (f32, f32, f32) {
        let exi = (ex as isize).clamp(step_px, max_xi - step_px);
        let eyi = (ey as isize).clamp(step_px, max_yi - step_px);
        let mut sl = 0.0f32;
        let mut srg = 0.0f32;
        let mut sby = 0.0f32;
        for &(ox, oy) in &[
            (0isize, 0isize),
            (-step_px, 0),
            (step_px, 0),
            (0, -step_px),
            (0, step_px),
        ] {
            let idx = ((eyi + oy) as usize) * frame_w + ((exi + ox) as usize);
            let p4 = idx * 4;
            let l = luma[idx];
            sl += l;
            srg += ((rgba[p4] as f32) - (rgba[p4 + 1] as f32)) * (1.0 / 255.0);
            sby += (rgba[p4 + 2] as f32) * (1.0 / 255.0) - l;
        }
        (sl * 0.2, srg * 0.2, sby * 0.2)
    };
    let (l0, rg0, by0) = sample_color_avg(x0, y0);
    let (l1, rg1, by1) = sample_color_avg(x1, y1);

    let mut found_bg_valley = false;
    for step in 3..=9usize {
        let t = (step as f32) * (1.0 / 12.0);
        let ut = u0 + du * t;
        let vt = v0 + dv * t;
        let c0 = sample_pyr_bilinear(pyr_eca, ut, vt);
        let cm = sample_pyr_bilinear(pyr_eca, ut - nx, vt - ny);
        let cp = sample_pyr_bilinear(pyr_eca, ut + nx, vt + ny);
        let rib_max = c0.max(cm.max(cp));
        if rib_max <= 0.14 * end_min && rib_max < 0.024 && !found_bg_valley {
            let xm = x0 + (x1 - x0) * t;
            let ym = y0 + (y1 - y0) * t;
            let xi = (xm as isize).clamp(step_px, max_xi - step_px);
            let yi = (ym as isize).clamp(step_px, max_yi - step_px);
            let mut bg_match = true;
            for &(ox, oy) in &[
                (0isize, 0isize),
                (-step_px, 0),
                (step_px, 0),
                (0, -step_px),
                (0, step_px),
            ] {
                let p_idx = ((yi + oy) as usize) * frame_w + ((xi + ox) as usize);
                if bg_stat_streak[p_idx] < 4 || (luma[p_idx] - bg_luma[p_idx]).abs() > 0.022 {
                    bg_match = false;
                    break;
                }
            }
            if bg_match {
                let (lm, rgm, bym) = sample_color_avg(xm, ym);
                let pd0 = (lm - l0).abs() + 1.5 * (rgm - rg0).abs() + 1.5 * (bym - by0).abs();
                let pd1 = (lm - l1).abs() + 1.5 * (rgm - rg1).abs() + 1.5 * (bym - by1).abs();
                if pd0 >= 0.12 && pd1 >= 0.12 {
                    found_bg_valley = true;
                }
            }
        }
    }

    if found_bg_valley {
        BodyConnectivity::SeparatedByValley
    } else {
        BodyConnectivity::Unsplit
    }
}

#[inline(always)]
fn div_rem_grid_w(cell: usize, grid_w: usize) -> (usize, usize) {
    if grid_w == 40 {
        let gy = (cell * 3277) >> 17;
        (cell - gy * 40, gy)
    } else if grid_w == 20 {
        let gy = (cell * 3277) >> 16;
        (cell - gy * 20, gy)
    } else {
        (cell % grid_w, cell / grid_w)
    }
}

/// Extract Connected Components / Energy Peaks over the active ECA motion-residual grid
/// (`20x11` in 320x180 SD, `40x22` in 640x360 HD).
#[inline(never)]
unsafe fn extract_connected_component_peaks(a: &mut Arena) {
    let (frame_w, frame_h, grid_w, grid_h, scale_px, _pip_dim, s) = mode_dims(a.res_mode);
    let grid_cells = grid_w * grid_h;
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;

    let macro_peak_cap = if dense_swarm_hero_cap_active(a) {
        6usize.min(MAX_TARGETS)
    } else {
        MAX_TARGETS
    };
    a.cc_visited[..grid_cells].fill(0);
    for p in 0..MAX_TARGETS {
        a.peaks[p].valid = 0;
        a.peaks[p].energy = 0.0;
    }
    a.num_peaks = 0;

    let mut sum_e = 0.0f32;
    let mut max_e = 0.0f32;
    for i in 0..grid_cells {
        let e = a.cc_energy[i];
        sum_e += e;
        if e > max_e {
            max_e = e;
        }
    }
    let mean_e = sum_e / (grid_cells as f32);
    let seed_floor = if a.lab_mode == 0
        && scale_px == 1
        && a.cam_pan_streak == 0
        && a.last_pan_dx == 0
        && a.last_pan_dy == 0
        && a.clutter_db_valid != 0
        && a.clutter_db >= 41.0
    {
        1.15
    } else if a.lab_mode == 0
        && scale_px == 1
        && a.cam_pan_streak == 0
        && a.clutter_db_valid != 0
        && a.clutter_db >= 38.0
    {
        1.50
    } else {
        2.2
    };
    let seed_thresh = (mean_e * 2.6).max(seed_floor);

    if max_e < seed_thresh {
        return;
    }

    let min_split_cells = 6 * scale_px * scale_px;
    let min_dg2 = 5 * (scale_px as i32) * (scale_px as i32);
    let mut next_comp_id = 1u8;

    let mut seed_mask = [0u64; 14];
    let n_words = (grid_cells + 63) >> 6;
    let v_thresh = f32x4_splat(seed_thresh);
    let cc_e_ptr = a.cc_energy.as_ptr();
    let mut ci = 0usize;
    for w in 0..n_words {
        let mut bits = 0u64;
        let end_c = (ci + 64).min(grid_cells);
        let mut sh = 0u32;
        while ci + 4 <= end_c {
            let mask4 = f32x4_ge(v128_load(cc_e_ptr.add(ci) as *const v128), v_thresh);
            bits |= (i32x4_bitmask(mask4) as u64) << sh;
            ci += 4;
            sh += 4;
        }
        while ci < end_c {
            if *cc_e_ptr.add(ci) >= seed_thresh {
                bits |= 1u64 << sh;
            }
            ci += 1;
            sh += 1;
        }
        seed_mask[w] = bits;
    }

    for w in 0..n_words {
        let mut bits = seed_mask[w];
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let start_idx = (w << 6) + b;
            if a.cc_visited[start_idx] != 0 {
                continue;
            }
        let comp_id = next_comp_id;
        next_comp_id = next_comp_id.saturating_add(1).min(30);

        let mut q_head = 0usize;
        let mut q_tail = 0usize;
        a.cc_queue[q_tail] = start_idx as u16;
        q_tail += 1;
        a.cc_visited[start_idx] = comp_id;

        let mut comp_cells = 0usize;
        let mut sum_w = 0.0f32;
        let mut sum_wx = 0.0f32;
        let mut sum_wy = 0.0f32;
        let mut best_cell_e = 0.0f32;
        let mut best_gx = 0usize;
        let mut best_gy = 0usize;
        let mut min_gx = grid_w;
        let mut max_gx = 0usize;
        let mut min_gy = grid_h;
        let mut max_gy = 0usize;

        while q_head < q_tail {
            let cur = a.cc_queue[q_head] as usize;
            q_head += 1;
            comp_cells += 1;

            let (gx, gy) = div_rem_grid_w(cur, grid_w);
            if gx < min_gx {
                min_gx = gx;
            }
            if gx > max_gx {
                max_gx = gx;
            }
            if gy < min_gy {
                min_gy = gy;
            }
            if gy > max_gy {
                max_gy = gy;
            }
            let e = a.cc_energy[cur];
            sum_w += e;
            sum_wx += e * ((gx * 16 + 8) as f32);
            sum_wy += e * ((gy * 16 + 8) as f32);

            if e > best_cell_e {
                best_cell_e = e;
                best_gx = gx;
                best_gy = gy;
            }

            let y_min = gy.saturating_sub(1);
            let y_max = (gy + 1).min(grid_h - 1);
            let x_min = gx.saturating_sub(1);
            let x_max = (gx + 1).min(grid_w - 1);

            for ny in y_min..=y_max {
                for nx in x_min..=x_max {
                    let nidx = ny * grid_w + nx;
                    if a.cc_visited[nidx] == 0 && a.cc_energy[nidx] >= seed_thresh * 0.65 {
                        a.cc_visited[nidx] = comp_id;
                        if q_tail < grid_cells {
                            a.cc_queue[q_tail] = nidx as u16;
                            q_tail += 1;
                        }
                    }
                }
            }
        }

        if comp_cells == 0 || comp_cells > (grid_cells * 65) / 100 {
            continue;
        }

        let mut sub_roots = [(0usize, 0usize, 0.0f32); 3];
        let mut num_sub = 0usize;

        if comp_cells >= min_split_cells {
            for qi in 0..q_tail {
                let cell = a.cc_queue[qi] as usize;
                let ce = a.cc_energy[cell];
                if ce < seed_thresh * 1.15 {
                    continue;
                }
                let (cgx, cgy) = div_rem_grid_w(cell, grid_w);
                let mut is_local_max = true;
                let ny0 = cgy.saturating_sub(1);
                let ny1 = (cgy + 1).min(grid_h - 1);
                let nx0 = cgx.saturating_sub(1);
                let nx1 = (cgx + 1).min(grid_w - 1);
                'nbr: for ny in ny0..=ny1 {
                    for nx in nx0..=nx1 {
                        if nx == cgx && ny == cgy {
                            continue;
                        }
                        let ne = a.cc_energy[ny * grid_w + nx];
                        if ne > ce || (ne == ce && (ny * grid_w + nx) < cell) {
                            is_local_max = false;
                            break 'nbr;
                        }
                    }
                }
                if !is_local_max {
                    continue;
                }
                let mut too_close_grid = false;
                for sr in 0..num_sub {
                    let dgx = cgx as i32 - sub_roots[sr].0 as i32;
                    let dgy = cgy as i32 - sub_roots[sr].1 as i32;
                    if dgx * dgx + dgy * dgy < min_dg2 {
                        if ce > sub_roots[sr].2 {
                            sub_roots[sr] = (cgx, cgy, ce);
                        }
                        too_close_grid = true;
                        break;
                    }
                }
                if !too_close_grid {
                    if num_sub < 3 {
                        sub_roots[num_sub] = (cgx, cgy, ce);
                        num_sub += 1;
                    } else {
                        let mut min_s = 0usize;
                        for sr in 1..3 {
                            if sub_roots[sr].2 < sub_roots[min_s].2 {
                                min_s = sr;
                            }
                        }
                        if ce > sub_roots[min_s].2 {
                            sub_roots[min_s] = (cgx, cgy, ce);
                        }
                    }
                }
            }
        }

        let r_margin = 10.0 * s;
        let c_margin = 9.0 * s;
        let centroid_win = 12 * scale_px;
        let mut sub_pts = [(0.0f32, 0.0f32, 0.0f32); 3];

        if num_sub >= 2 {
            for sr in 0..num_sub {
                let (sgx, sgy, sub_e) = sub_roots[sr];
                sub_pts[sr] = (
                    ((sgx * 16 + 8) as f32).clamp(r_margin, (frame_w as f32) - r_margin - 1.0),
                    ((sgy * 16 + 8) as f32).clamp(r_margin, (frame_h as f32) - r_margin - 1.0),
                    sub_e * 2.2 * s * s,
                );
            }
        } else {
            let mut valley_split_done = false;
            if comp_cells >= 2 && comp_cells <= 12 * scale_px * scale_px {
                let cells_to_pyr = 8 / scale_px;
                let pu0 = (min_gx * cells_to_pyr).max(2);
                let pu1 = ((max_gx + 1) * cells_to_pyr).min(PYR_W - 3);
                let pv0 = (min_gy * cells_to_pyr).max(2);
                let pv1 = ((max_gy + 1) * cells_to_pyr).min(PYR_H - 3);
                let mut loc = [(0usize, 0usize, 0.0f32); 6];
                let mut n_loc = 0usize;
                for pv in pv0..=pv1 {
                    let prow = pv * PYR_W;
                    for pu in pu0..=pu1 {
                        let pe = a.pyr_eca[prow + pu];
                        if pe < 0.14 {
                            continue;
                        }
                        let mut is_max = true;
                        'pmax: for dy in -1isize..=1isize {
                            let nr = ((pv as isize + dy) as usize) * PYR_W;
                            for dx in -1isize..=1isize {
                                if dx == 0 && dy == 0 {
                                    continue;
                                }
                                let idx_n = nr + ((pu as isize + dx) as usize);
                                let ne = a.pyr_eca[idx_n];
                                if ne > pe || (ne == pe && idx_n < prow + pu) {
                                    is_max = false;
                                    break 'pmax;
                                }
                            }
                        }
                        if !is_max {
                            continue;
                        }
                        if n_loc < 6 {
                            loc[n_loc] = (pu, pv, pe);
                            n_loc += 1;
                        } else {
                            let mut min_k = 0usize;
                            for k in 1..6 {
                                if loc[k].2 < loc[min_k].2 {
                                    min_k = k;
                                }
                            }
                            if pe > loc[min_k].2 {
                                loc[min_k] = (pu, pv, pe);
                            }
                        }
                    }
                }
                for i in 0..n_loc {
                    for j in (i + 1)..n_loc {
                        if loc[j].2 > loc[i].2 {
                            let tmp = loc[i];
                            loc[i] = loc[j];
                            loc[j] = tmp;
                        }
                    }
                }
                if n_loc >= 2 && loc[0].2 >= 0.16 {
                    let mut picked = [(0.0f32, 0.0f32, 0.0f32); 3];
                    picked[0] = (
                        ((loc[0].0 as f32) + 0.5) * 2.0 * s,
                        ((loc[0].1 as f32) + 0.5) * 2.0 * s,
                        loc[0].2,
                    );
                    let mut n_picked = 1usize;
                    for k in 1..n_loc {
                        if n_picked >= 3 || loc[k].2 < 0.42 * loc[0].2 {
                            break;
                        }
                        let cand_x = ((loc[k].0 as f32) + 0.5) * 2.0 * s;
                        let cand_y = ((loc[k].1 as f32) + 0.5) * 2.0 * s;
                        let mut all_sep = true;
                        for j in 0..n_picked {
                            let dx = cand_x - picked[j].0;
                            let dy = cand_y - picked[j].1;
                            let d2 = dx * dx + dy * dy;
                            let mode_c_in_pan = a.lab_mode == 2
                                && (a.cam_pan_streak >= 1 || a.last_pan_dx != 0 || a.last_pan_dy != 0);
                            let split_min_d = if mode_c_in_pan { 9.5 * s } else { 12.5 * s };
                            let mode_c_pair_sep = mode_c_in_pan
                                && picked[j].2 >= 0.22
                                && loc[k].2 >= 0.22
                                && d2 >= (10.5 * s) * (10.5 * s);
                            if d2 < split_min_d * split_min_d
                                || d2 > (25.0 * s) * (25.0 * s)
                                || (!mode_c_pair_sep
                                    && classify_eca_connectivity(
                                        &a.pyr_eca,
                                        &a.luma,
                                        &a.rgba_ring[a.active_rgba_slot as usize],
                                        &a.bg_luma,
                                        &a.bg_stat_streak,
                                        frame_w,
                                        frame_h,
                                        s,
                                        picked[j].0,
                                        picked[j].1,
                                        cand_x,
                                        cand_y,
                                    ) != BodyConnectivity::SeparatedByValley)
                            {
                                all_sep = false;
                                break;
                            }
                        }
                        if all_sep {
                            picked[n_picked] = (cand_x, cand_y, loc[k].2);
                            n_picked += 1;
                        }
                    }
                    if n_picked >= 2 {
                        let mut sum_pe = 0.0f32;
                        for j in 0..n_picked {
                            sum_pe += picked[j].2;
                        }
                        for j in 0..n_picked {
                            sub_pts[j] = (
                                picked[j]
                                    .0
                                    .clamp(r_margin, (frame_w as f32) - r_margin - 1.0),
                                picked[j]
                                    .1
                                    .clamp(r_margin, (frame_h as f32) - r_margin - 1.0),
                                sum_w * (picked[j].2 / sum_pe.max(1e-5)),
                            );
                        }
                        num_sub = n_picked;
                        valley_split_done = true;
                    }
                }
            }
            if !valley_split_done {
                let sgx = if num_sub == 1 { sub_roots[0].0 } else { best_gx };
                let sgy = if num_sub == 1 { sub_roots[0].1 } else { best_gy };
                sub_pts[0] = (
                    (0.5 * (sum_wx / sum_w.max(1e-5)) + 0.5 * ((sgx * 16 + 8) as f32))
                        .clamp(r_margin, (frame_w as f32) - r_margin - 1.0),
                    (0.5 * (sum_wy / sum_w.max(1e-5)) + 0.5 * ((sgy * 16 + 8) as f32))
                        .clamp(r_margin, (frame_h as f32) - r_margin - 1.0),
                    sum_w,
                );
                num_sub = 1;
            }
        }

        for sr in 0..num_sub {
            let (rough_x, rough_y, peak_e) = sub_pts[sr];
            let mut eff_win = centroid_win;
            if num_sub >= 2 {
                let mut min_d2 = 1e9f32;
                for o in 0..num_sub {
                    if o != sr {
                        let dx = rough_x - sub_pts[o].0;
                        let dy = rough_y - sub_pts[o].1;
                        let d2 = dx * dx + dy * dy;
                        if d2 < min_d2 {
                            min_d2 = d2;
                        }
                    }
                }
                let half_sep = (0.40 * wasm_sqrtf(min_d2)) as usize;
                eff_win = half_sep.clamp(5 * scale_px, centroid_win);
            }

            let rx0 = (rough_x as usize).saturating_sub(eff_win).max(2);
            let rx1 = ((rough_x as usize) + eff_win).min(frame_w - 3);
            let ry0 = (rough_y as usize).saturating_sub(eff_win).max(2);
            let ry1 = ((rough_y as usize) + eff_win).min(frame_h - 3);

            let (pw, pwx, pwy) = simd_ops::weighted_eca_centroid_simd128(
                a.eca_res.as_ptr(),
                frame_w,
                rx0,
                rx1,
                ry0,
                ry1,
            );

            let (cx, cy) = if pw > 1e-4 {
                (
                    (pwx / pw).clamp(c_margin, (frame_w as f32) - c_margin - 1.0),
                    (pwy / pw).clamp(c_margin, (frame_h as f32) - c_margin - 1.0),
                )
            } else {
                (rough_x, rough_y)
            };

            let tlx = ((cx - half_patch).max(0.0) as usize).min(frame_w - patch_dim);
            let tly = ((cy - half_patch).max(0.0) as usize).min(frame_h - patch_dim);
            ensure_chroma!(a, tlx, tly, patch_dim, patch_dim, frame_w, frame_h);
            let sig = haar_morton_signature_scaled_simd128(
                a.luma.as_ptr(),
                a.chroma_rg.as_ptr(),
                a.chroma_by.as_ptr(),
                frame_w,
                tlx,
                tly,
                scale_px,
            );

            let cand = CandidatePeak {
                x: cx,
                y: cy,
                energy: peak_e,
                sig,
                valid: 1,
                comp_id,
                _pad: [0; 2],
            };

            if a.num_peaks < macro_peak_cap {
                a.peaks[a.num_peaks] = cand;
                a.num_peaks += 1;
            } else {
                let mut min_slot = 0usize;
                let mut min_val = a.peaks[0].energy;
                for k in 1..macro_peak_cap {
                    if a.peaks[k].energy < min_val {
                        min_val = a.peaks[k].energy;
                        min_slot = k;
                    }
                }
                if cand.energy > min_val {
                    a.peaks[min_slot] = cand;
                }
            }
        }
        }
    }

    for i in 0..a.num_peaks {
        for j in (i + 1)..a.num_peaks {
            if a.peaks[j].energy > a.peaks[i].energy {
                let tmp = a.peaks[i];
                a.peaks[i] = a.peaks[j];
                a.peaks[j] = tmp;
            }
        }
    }
}

/// Complete fixed-array symmetric Jacobi eigensolver for the active 16x16 Laplacian subgraph.
/// Used by standalone `chronos_tick` so pre-assignment Fiedler costs are always certified
/// without seed symmetry blindness.
fn solve_fiedler_jacobi_16x16(a: &mut Arena) {
    let mut act = [0usize; GRAPH_NODES];
    let mut n = 0usize;
    for i in 0..GRAPH_NODES {
        if a.laplacian[i * GRAPH_NODES + i] > 1e-6 {
            act[n] = i;
            n += 1;
        }
    }

    for i in 0..GRAPH_NODES {
        a.fiedler_vec[i] = 0.0;
    }
    if n < 2 {
        a.fiedler_lambda2 = 0.0;
        a.fiedler_gap = 0.0;
        return;
    }

    for v in &mut a.jacobi_mat[..n * n] {
        *v = 0.0;
    }
    for v in &mut a.jacobi_vecs[..n * n] {
        *v = 0.0;
    }
    for r in 0..n {
        let ir = act[r];
        a.jacobi_vecs[r * n + r] = 1.0;
        for c in 0..n {
            a.jacobi_mat[r * n + c] = a.laplacian[ir * GRAPH_NODES + act[c]];
        }
    }
    let mat = &mut a.jacobi_mat;
    let vecs = &mut a.jacobi_vecs;

    for _sweep in 0..12usize {
        let mut max_off = 0.0f32;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[p * n + q];
                let abs_apq = apq.abs();
                if abs_apq > max_off {
                    max_off = abs_apq;
                }
                if abs_apq < 1e-7 {
                    continue;
                }
                let app = mat[p * n + p];
                let aqq = mat[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + wasm_sqrtf(1.0 + tau * tau))
                } else {
                    -1.0 / (-tau + wasm_sqrtf(1.0 + tau * tau))
                };
                let c = 1.0 / wasm_sqrtf(1.0 + t * t);
                let s = t * c;

                mat[p * n + p] = app - t * apq;
                mat[q * n + q] = aqq + t * apq;
                mat[p * n + q] = 0.0;
                mat[q * n + p] = 0.0;

                for k in 0..n {
                    if k != p && k != q {
                        let akp = mat[k * n + p];
                        let akq = mat[k * n + q];
                        let nkp = c * akp - s * akq;
                        let nkq = s * akp + c * akq;
                        mat[k * n + p] = nkp;
                        mat[p * n + k] = nkp;
                        mat[k * n + q] = nkq;
                        mat[q * n + k] = nkq;
                    }
                    let vkp = vecs[k * n + p];
                    let vkq = vecs[k * n + q];
                    vecs[k * n + p] = c * vkp - s * vkq;
                    vecs[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
        if max_off < 1e-6 {
            break;
        }
    }

    // Sort eigenvalues ascending: evals[0] ~ 0 (1_n), evals[1] = lambda_2, evals[2] = lambda_3
    let mut order = [0usize; GRAPH_NODES];
    for i in 0..n {
        order[i] = i;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if mat[order[j] * n + order[j]] < mat[order[i] * n + order[i]] {
                let tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    let c1 = order[1];
    let lambda2 = mat[c1 * n + c1].max(0.0);
    let lambda3 = if n >= 3 {
        mat[order[2] * n + order[2]].max(lambda2)
    } else {
        lambda2
    };

    // Orthogonalize against 1_n and normalize
    let mut mean = 0.0f32;
    for r in 0..n {
        mean += vecs[r * n + c1];
    }
    mean /= n as f32;

    let mut norm2 = 0.0f32;
    let mut fvec = [0.0f32; GRAPH_NODES];
    for r in 0..n {
        let v = vecs[r * n + c1] - mean;
        fvec[r] = v;
        norm2 += v * v;
    }
    let inv_norm = 1.0 / wasm_sqrtf(norm2.max(1e-12));
    for r in 0..n {
        fvec[r] *= inv_norm;
    }

    // Temporal sign continuity against prior track Fiedler components
    let mut dot_prev = 0.0f32;
    for r in 0..n {
        let idx = act[r];
        if idx < MAX_TARGETS {
            dot_prev += fvec[r] * a.prev_fiedler_track[idx];
        }
    }
    let sign = if dot_prev < -1e-4 || (dot_prev.abs() <= 1e-4 && fvec[0] < 0.0) {
        -1.0
    } else {
        1.0
    };

    for r in 0..n {
        let idx = act[r];
        let val = fvec[r] * sign;
        a.fiedler_vec[idx] = val;
        if idx < MAX_TARGETS {
            a.prev_fiedler_track[idx] = val;
        }
    }
    a.fiedler_lambda2 = lambda2;
    a.fiedler_gap = (lambda3 - lambda2).max(0.0);
}

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
fn solve_noncausal_fista_target(
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
                                        t.is_ballistic = 1;
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
        t.grav_ay_frame = 0.0;
        t.grav_ay_px_s2 = 0.0;
        t.is_ballistic = 0;
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

// ============================================================================
// Exported C ABI Entrypoints
// ============================================================================

#[inline(always)]
fn set_stamp_invalid(telem_stamps: &mut [u32], base: usize) {
    telem_stamps[base] = 0;
    telem_stamps[base + 1] = 0;
    telem_stamps[base + 2] = 0;
    telem_stamps[base + 3] = 0;
    telem_stamps[base + 4] = 0;
    telem_stamps[base + 5] = 0;
    telem_stamps[base + 6] = (-1.0f32).to_bits();
    telem_stamps[base + 7] = 0;
}

#[inline(always)]
fn set_stamp_fresh(telem_stamps: &mut [u32], base: usize, meta: &IngestSlotMeta) {
    let time_bits = (meta.flags & (INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_CAPTURE_AGE | INGEST_FLAG_HANDOFF_AGE)) << 1;
    telem_stamps[base] = STAMP_VALID | time_bits;
    telem_stamps[base + 1] = meta.continuity_epoch;
    telem_stamps[base + 2] = meta.source_epoch;
    telem_stamps[base + 3] = meta.source_seq;
    telem_stamps[base + 4] = meta.t_src_us_lo;
    telem_stamps[base + 5] = meta.t_src_us_hi;
    telem_stamps[base + 6] = meta.age_ms_bits;
    telem_stamps[base + 7] = meta.engine_frame_idx;
}

#[inline(always)]
fn hold_or_invalidate_stamp(telem_stamps: &mut [u32], base: usize, cur_cont_ep: u32) {
    if (telem_stamps[base] & STAMP_VALID) != 0 && telem_stamps[base + 1] == cur_cont_ep && cur_cont_ep != 0 {
        telem_stamps[base] |= STAMP_HELD;
    } else {
        set_stamp_invalid(telem_stamps, base);
    }
}

#[inline(always)]
fn invalidate_all_wave1_stamps(wave1_telem: &mut [u32; WAVE1_TELEM_LEN]) {
    let mut i = 0usize;
    while i < 6 {
        set_stamp_invalid(wave1_telem, WAVE1_PRODUCT_STAMPS_START + i * STAMP_STRIDE);
        i += 1;
    }
    let mut m = 0usize;
    while m < MAX_TARGETS {
        set_stamp_invalid(wave1_telem, WAVE1_HERO_STAMPS_START + m * STAMP_STRIDE);
        m += 1;
    }
}

#[inline(always)]
fn invalidate_stale_wave2_stamps(wave2_telem: &mut [u32; WAVE2_TELEM_LEN], cur_cont_ep: u32) {
    let mut i = 0usize;
    while i < 3 {
        let base = WAVE2_PRODUCT_STAMPS_START + i * STAMP_STRIDE;
        if (wave2_telem[base] & STAMP_VALID) != 0 && wave2_telem[base + 1] != cur_cont_ep {
            set_stamp_invalid(wave2_telem, base);
        }
        i += 1;
    }
}

#[inline(always)]
fn invalidate_stale_wave1_stamps(wave1_telem: &mut [u32; WAVE1_TELEM_LEN], cur_cont_ep: u32) {
    let mut i = 0usize;
    while i < 6 {
        let base = WAVE1_PRODUCT_STAMPS_START + i * STAMP_STRIDE;
        if (wave1_telem[base] & STAMP_VALID) != 0 && wave1_telem[base + 1] != cur_cont_ep {
            set_stamp_invalid(wave1_telem, base);
        }
        i += 1;
    }
    let mut m = 0usize;
    while m < MAX_TARGETS {
        let base = WAVE1_HERO_STAMPS_START + m * STAMP_STRIDE;
        if (wave1_telem[base] & STAMP_VALID) != 0 && wave1_telem[base + 1] != cur_cont_ep {
            set_stamp_invalid(wave1_telem, base);
        }
        m += 1;
    }
}

#[inline(always)]
fn oriented_extent(cw: u32, ch: u32, orientation: u32) -> (u32, u32) {
    if (orientation & 1) == 0 {
        (cw, ch)
    } else {
        (ch, cw)
    }
}

#[inline(never)]
unsafe fn compute_row_luma_means_simd128(
    luma: &[f32],
    out_means: &mut [f32; MAX_FRAME_H],
    frame_w: usize,
    frame_h: usize,
) {
    if frame_w == 0 || frame_h == 0 || luma.len() < frame_w * frame_h {
        out_means.fill(0.0);
        return;
    }
    let inv_fw = 1.0f32 / (frame_w as f32);
    let base_ptr = luma.as_ptr();
    let max_h = frame_h.min(MAX_FRAME_H);
    for ry in 0..max_h {
        let row_ptr = base_ptr.add(ry * frame_w);
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut acc2 = f32x4_splat(0.0);
        let mut acc3 = f32x4_splat(0.0);
        let mut rx = 0usize;
        while rx + 16 <= frame_w {
            acc0 = f32x4_add(acc0, v128_load(row_ptr.add(rx) as *const v128));
            acc1 = f32x4_add(acc1, v128_load(row_ptr.add(rx + 4) as *const v128));
            acc2 = f32x4_add(acc2, v128_load(row_ptr.add(rx + 8) as *const v128));
            acc3 = f32x4_add(acc3, v128_load(row_ptr.add(rx + 12) as *const v128));
            rx += 16;
        }
        let mut sum_v = f32x4_add(f32x4_add(acc0, acc1), f32x4_add(acc2, acc3));
        while rx + 4 <= frame_w {
            sum_v = f32x4_add(sum_v, v128_load(row_ptr.add(rx) as *const v128));
            rx += 4;
        }
        let mut rsum = f32x4_extract_lane::<0>(sum_v)
            + f32x4_extract_lane::<1>(sum_v)
            + f32x4_extract_lane::<2>(sum_v)
            + f32x4_extract_lane::<3>(sum_v);
        while rx < frame_w {
            rsum += *row_ptr.add(rx);
            rx += 1;
        }
        let mean = rsum * inv_fw;
        out_means[ry] = if mean.is_finite() {
            mean.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    if max_h < MAX_FRAME_H {
        out_means[max_h..MAX_FRAME_H].fill(0.0);
    }
}

#[inline(always)]
fn wave1_enabled_bits(a: &Arena) -> u32 {
    (if a.micro_swarm.get_micro_metrology_mode() != 0 {
        1
    } else {
        0
    }) | (if a.bos.get_mode() != bos::BOS_MODE_OFF {
        2
    } else {
        0
    })
}

#[inline(always)]
fn wave2_enabled_bits(a: &Arena) -> u32 {
    (if a.vibro.get_mode() != vibrometry::RS_MODE_OFF {
        1
    } else {
        0
    }) | (if a.hodge.get_mode() != hodge::HODGE_MODE_OFF {
        2
    } else {
        0
    }) | (if a.lab_mode == 2 {
        4
    } else {
        0
    })
}

#[inline(always)]
fn invalidate_all_wave2_stamps(wave2_telem: &mut [u32; WAVE2_TELEM_LEN]) {
    let mut i = 0usize;
    while i < 3 {
        set_stamp_invalid(wave2_telem, WAVE2_PRODUCT_STAMPS_START + i * STAMP_STRIDE);
        i += 1;
    }
}

#[inline(never)]
fn init_wave2_telemetry(a: &mut Arena) {
    a.wave2_telem.fill(0);
    a.wave2_telem[0] = WAVE2_ABI_VERSION;
    a.wave2_telem[1] = core::mem::size_of::<Arena>() as u32;
    a.wave2_telem[2] = wave2_enabled_bits(a);
    a.wave2_telem[3] = a.wave1_telem[6];
    invalidate_all_wave2_stamps(&mut a.wave2_telem);
}

#[inline(never)]
fn init_wave1_telemetry(a: &mut Arena, preserve_history: bool) {
    let prev_commits_lo = if preserve_history { a.wave1_telem[4] } else { 0 };
    let prev_commits_hi = if preserve_history { a.wave1_telem[5] } else { 0 };
    let prev_epoch = if preserve_history { a.wave1_telem[6] } else { 0 };
    let had_any = (prev_commits_lo | prev_commits_hi) != 0 || prev_epoch != 0;
    let next_epoch = if preserve_history && had_any {
        prev_epoch.wrapping_add(1).max(1)
    } else {
        0
    };

    a.pending_ingest_meta = IngestSlotMeta::zeroed();
    for s in 0..RING_LEN {
        a.ring_meta[s] = IngestSlotMeta::zeroed();
        a.row_luma_means[s].fill(0.0);
    }
    a.ingest_ctrl.fill(0);
    a.wave1_telem.fill(0);
    a.wave1_telem[0] = WAVE1_ABI_VERSION;
    a.wave1_telem[1] = core::mem::size_of::<Arena>() as u32;
    a.wave1_telem[2] = u32::MAX;
    a.wave1_telem[3] = u32::MAX;
    a.wave1_telem[4] = prev_commits_lo;
    a.wave1_telem[5] = prev_commits_hi;
    a.wave1_telem[6] = next_epoch;
    a.wave1_telem[7] = wave1_enabled_bits(a);
    a.wave1_telem[8] = INGEST_STATUS_EMPTY;
    a.wave1_telem[9] = 0;
    a.wave1_telem[10] = 0;
    a.wave1_telem[11] = INGEST_RING_STRIDE_WORDS;
    a.wave1_telem[12] = linked_static_end_bytes();
    a.wave1_telem[13] = 0;
    invalidate_all_wave1_stamps(&mut a.wave1_telem);
    init_wave2_telemetry(a);
}

#[no_mangle]
pub unsafe extern "C" fn chronos_init() {
    let a = arena_mut();
    core::ptr::write_bytes(a as *mut Arena as *mut u8, 0, core::mem::size_of::<Arena>());
    a.res_mode = 0;
    a.lookahead_k = 8;
    a.eca_adapt_rate = 0.035;
    a.fista_lambda = 1.45;
    a.eulerian_gain = 25.0;
    a.phase_offset_rad = -HALF_PI;
    a.audio_out_latency_sec = 0.0;
    a.spectral_gamma = 0.35;
    a.lab_mode = 0;
    a.next_track_id = 1;
    a.reset_base_track_id = 1;
    a.clutter_db = 0.0;
    a.clutter_db_valid = 0;
    a.micro_swarm.init_in_place();
    a.continuum.reset();
    a.bos.init_in_place();
    a.vibro.reset_all();
    a.hodge.reset();
    a.lanczos_scratch.prev_track = [0.0; MAX_TARGETS];
    a.lanczos_scratch.last_lambda2 = 0.0;
    a.lanczos_scratch.last_n = 0;
    a.swarm_mode_cached = 0;
    a.swarm_telem_zeroed = 1;
    a.telemetry[20] = core::mem::size_of::<Arena>() as f32;
    init_wave1_telemetry(a, false);
}

#[no_mangle]
pub unsafe extern "C" fn chronos_reset_tracks() {
    let a = arena_mut();
    for m in 0..MAX_TARGETS {
        a.targets[m] = core::mem::zeroed();
        a.kalman[m] = core::mem::zeroed();
        a.imm[m].reset();
        a.ode_states[m].reset();
        a.pend_states[m].reset();
        a.lyap_cache[m] = 0.0;
        a.prev_fiedler_track[m] = 0.0;
        a.all_pip_tid[m] = 0;
        for i in 0..MAX_PIP_PIXELS {
            a.all_pip_ref_luma[m][i] = 0.0;
        }
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
    for i in 0..MAX_FRAME_PIXELS {
        a.bg_luma[i] = 0.0;
        a.eca_res[i] = 0.0;
        a.bg_stat_streak[i] = 0;
        a.bg_eval_mask[i] = 0;
    }
    for i in 0..PYR_PIXELS {
        a.pyr_luma[i] = 0.0;
        a.prev_pyr_luma[i] = 0.0;
        a.pyr_rg[i] = 0.0;
        a.pyr_by[i] = 0.0;
        a.pyr_eca[i] = 0.0;
    }
    for i in 0..(GRAPH_NODES * GRAPH_NODES) {
        a.laplacian[i] = 0.0;
    }
    for i in 0..GRAPH_NODES {
        a.fiedler_vec[i] = 0.0;
    }
    for i in 0..MAX_PIP_PIXELS {
        a.pip_ref_luma[i] = 0.0;
    }
    a.num_peaks = 0;
    a.padic_id_swaps = 0;
    a.kalman_id_swaps = 0;
    a.pip_target_id = 0;
    a.pip_hold_frames = 0;
    a.frame_idx = 0;
    a.ring_cursor = 0;
    a.last_display_slot = 0;
    a.clutter_db = 0.0;
    a.clutter_db_valid = 0;
    a.clutter_eval_ratio = 0.0;
    a.eca_err2_ema = 0.0;
    a.eca_sig2_ema = 0.0;
    a.fiedler_lambda2 = 0.0;
    a.fiedler_gap = 0.0;
    a.last_strobe_frame = 0;
    a.has_strobed = 0;
    a.stimulus_frame = 0;
    a.stimulus_armed = 0;
    a.focus_target_id = 0;
    a.scene_gravity_px_s2 = 0.0;
    a.scene_gravity_conf = 0.0;
    a.scatter_deg = 0.0;
    a.scatter_x = 0.0;
    a.scatter_y = 0.0;
    a.scatter_th1_deg = 0.0;
    a.scatter_th2_deg = 0.0;
    a.scatter_event_frame = 0;
    a.cum_time_sec = 0.0;
    a.prev_raw_pts_sec = 0.0;
    a.pending_pts_sec = 0.0;
    a.hist_t_sec = [0.0; HIST_RING_LEN];
    a.has_explicit_pts = 0;
    a.last_discont_frame = 0;
    a.discont_count = 0;
    a.cam_motion_state = 0;
    a.cam_pan_streak = 0;
    a.cam_cut_hold = 0;
    a.cam_pan_speed_px_s = 0.0;
    a.last_pan_dx = 0;
    a.last_pan_dy = 0;
    a.cum_cam_x = 0.0;
    a.cum_cam_y = 0.0;
    a.hist_cam_x = [0.0; HIST_RING_LEN];
    a.hist_cam_y = [0.0; HIST_RING_LEN];
    a.scene_gravity_arcs = 0;
    a.last_grav_arc_track = 0;
    a.last_grav_arc_frame = 0;
    a.last_strike_tid = 0;
    a.last_strike_frame = 0;
    a.coast_reacq_events = 0;
    a.dormant_reid_events = 0;
    a.peak_simultaneous_confirmed = 0;
    a.longest_tenure_frames = 0;
    a.longest_tenure_tid = 0;
    a.sustained_10s_count = 0;
    a.sustained_60s_count = 0;
    a.last_reid_gap_frames = 0;
    a.reset_base_track_id = a.next_track_id;
    a.micro_swarm.reset_active_keep_mode();
    a.continuum.reset();
    a.bos.reset_keep_config();
    a.vibro.reset_continuity(0);
    a.hodge.reset_keep_mode();
    a.lanczos_scratch.prev_track = [0.0; MAX_TARGETS];
    a.lanczos_scratch.last_lambda2 = 0.0;
    a.lanczos_scratch.last_n = 0;
    a.micro_swarm.swarm_telem[1..soa_tracks::SWARM_TELEM_LEN].fill(0.0);
    a.continuum_was_active = 0;
    a.swarm_telem_zeroed = if a.swarm_mode_cached == 0 { 1 } else { 0 };
    a.last_scene_cut = 0;
    a.last_pan_dx_f = 0.0;
    a.last_pan_dy_f = 0.0;
    a.inspected_hero_tid = 0;
    a.inspected_micro_tid_pin = 0;
    a.pre_tick_seen = 0;
    a.last_pre_tick_frame = 0;
    init_wave1_telemetry(a, true);
}

/// Test hook, not part of the host API. Sets the engine frame counter `frame_idx` (the `u32`
/// that `chronos_post_tick` advances with wrapping) and the ingest ring cursor `ring_cursor`
/// (the monotonic `u64` that selects the ring slot), and nothing else, so test harnesses can
/// exercise the frame-index wrap at `u32::MAX` without running 2^32 frames. The cursor is
/// `ring_cursor_hi << 32 | ring_cursor_lo`; pass its current value to move only `frame_idx`.
///
/// Ring contents, frame history and track state are left as they are: the kernel sees both
/// counters jump, exactly as if the host had overwritten the two arena fields. The gates did
/// that by raw address until v3.1.0 (`scripts/gate_i5_bos.mjs`, `scripts/gate_i0_i4_ingest.mjs`),
/// which a recorded trace cannot replay on a native build; a call to this export replays on both.
#[no_mangle]
pub unsafe extern "C" fn chronos_test_hook_set_frame_counters(frame_idx: u32, ring_cursor_lo: u32, ring_cursor_hi: u32) {
    let a = arena_mut();
    a.frame_idx = frame_idx;
    a.ring_cursor = ((ring_cursor_hi as u64) << 32) | ring_cursor_lo as u64;
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_ingest_pts(pts_sec: f64) {
    if pts_sec.is_finite() && pts_sec >= 0.0 {
        let a = arena_mut();
        a.pending_pts_sec = pts_sec;
        a.has_explicit_pts = 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_focus_target(track_id: u32) {
    let a = arena_mut();
    if track_id != 0 && track_id == a.inspected_micro_tid_pin && a.inspected_hero_tid != 0 {
        a.focus_target_id = a.inspected_hero_tid;
    } else {
        a.focus_target_id = track_id;
        if !(track_id != 0 && track_id == a.inspected_hero_tid) {
            a.inspected_micro_tid_pin = 0;
            a.inspected_hero_tid = 0;
            a.micro_swarm.inspected_micro_tid = 0;
            a.micro_swarm.inspected_body_id = 0;
            a.micro_swarm.swarm_telem[11] = 0.0;
            a.micro_swarm.swarm_telem[12] = 0.0;
        }
    }
    if a.focus_target_id != 0 {
        for m in 0..MAX_TARGETS {
            if a.targets[m].active != 0 && a.targets[m].track_id == a.focus_target_id {
                a.targets[m].wave1_auto_owned = 0;
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_focus_target() -> u32 {
    arena_mut().focus_target_id
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_resolution(mode: u32) {
    let a = arena_mut();
    let next_mode = if mode != 0 { 1 } else { 0 };
    if a.res_mode != next_mode {
        a.res_mode = next_mode;
        chronos_reset_tracks();
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_resolution() -> u32 {
    arena_mut().res_mode
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_frame_w() -> u32 {
    mode_dims(arena_mut().res_mode).0 as u32
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_frame_h() -> u32 {
    mode_dims(arena_mut().res_mode).1 as u32
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_pip_dim() -> u32 {
    mode_dims(arena_mut().res_mode).5 as u32
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_ring_slot_ptr(slot: u32) -> *mut u8 {
    let a = arena_mut();
    a.rgba_ring[(slot as usize) % RING_LEN].as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_params(
    lookahead_k: u32,
    eca_adapt_rate: f32,
    fista_lambda: f32,
    eulerian_gain: f32,
    phase_offset_rad: f32,
    lab_mode: u32,
) {
    let a = arena_mut();
    a.lookahead_k = lookahead_k.min((RING_LEN - 1) as u32);
    if eca_adapt_rate.is_finite() {
        a.eca_adapt_rate = eca_adapt_rate.clamp(0.001, 0.25);
    }
    if fista_lambda.is_finite() {
        a.fista_lambda = fista_lambda.clamp(0.05, 12.0);
    }
    if eulerian_gain.is_finite() {
        a.eulerian_gain = eulerian_gain.clamp(1.0, 80.0);
    }
    if phase_offset_rad.is_finite() {
        a.phase_offset_rad = wrap_pi(phase_offset_rad);
    }
    a.lab_mode = lab_mode.min(2);
    a.wave2_telem[2] = wave2_enabled_bits(a);
    if a.lab_mode != 2 {
        a.collision_state.conservation.reset_continuity();
        set_stamp_invalid(
            &mut a.wave2_telem,
            WAVE2_PRODUCT_STAMPS_START + PROD_W2_CONSERVATION * STAMP_STRIDE,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_spectral_weight(gamma: f32) {
    let a = arena_mut();
    if gamma.is_finite() {
        a.spectral_gamma = gamma.clamp(0.0, 4.0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_audio_latency(latency_sec: f32) {
    let a = arena_mut();
    if latency_sec.is_finite() {
        a.audio_out_latency_sec = latency_sec.clamp(0.0, 0.25);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_trigger_stimulus() {
    let a = arena_mut();
    a.stimulus_frame = a.frame_idx;
    a.stimulus_armed = 1;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            a.targets[m].stim_ref_x = a.targets[m].pos_x;
            a.targets[m].stim_ref_y = a.targets[m].pos_y;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_ingest_rgba_ptr() -> *mut u8 {
    let a = arena_mut();
    let slot = (a.ring_cursor % (RING_LEN as u64)) as usize;
    a.rgba_ring[slot].as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_display_rgba_ptr() -> *const u8 {
    let a = arena_mut();
    a.rgba_ring[a.last_display_slot as usize].as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_eca_res_ptr() -> *const f32 {
    let a = arena_mut();
    a.eca_res.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_pip_rgba_ptr() -> *const u8 {
    let a = arena_mut();
    a.pip_rgba.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_all_pip_rgba_ptr() -> *const u8 {
    let a = arena_mut();
    a.all_pip_rgba.as_ptr() as *const u8
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_slot_meta_ptr() -> *const f32 {
    let a = arena_mut();
    a.slot_meta.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_laplacian_ptr() -> *mut f32 {
    let a = arena_mut();
    a.laplacian.as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_fiedler_ptr() -> *mut f32 {
    let a = arena_mut();
    a.fiedler_vec.as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_fiedler_lambda2(lambda2: f32) {
    let a = arena_mut();
    a.fiedler_lambda2 = if lambda2.is_finite() { lambda2.max(0.0) } else { 0.0 };
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_fiedler_spectrum(lambda2: f32, spectral_gap: f32) {
    let a = arena_mut();
    a.fiedler_lambda2 = if lambda2.is_finite() { lambda2.max(0.0) } else { 0.0 };
    a.fiedler_gap = if spectral_gap.is_finite() { spectral_gap.max(0.0) } else { 0.0 };
}

#[inline(always)]
unsafe fn dot_f64_padded4(a_ptr: *const f64, b_ptr: *const f64, n4: usize) -> f64 {
    let mut s0 = f64x2_splat(0.0);
    let mut s1 = f64x2_splat(0.0);
    let mut i = 0usize;
    while i < n4 {
        let a0 = v128_load(a_ptr.add(i) as *const v128);
        let b0 = v128_load(b_ptr.add(i) as *const v128);
        s0 = f64x2_add(s0, f64x2_mul(a0, b0));
        let a1 = v128_load(a_ptr.add(i + 2) as *const v128);
        let b1 = v128_load(b_ptr.add(i + 2) as *const v128);
        s1 = f64x2_add(s1, f64x2_mul(a1, b1));
        i += 4;
    }
    let s = f64x2_add(s0, s1);
    f64x2_extract_lane::<0>(s) + f64x2_extract_lane::<1>(s)
}

#[inline(always)]
unsafe fn sub_scaled_f64_padded4(w_ptr: *mut f64, q_ptr: *const f64, scale: f64, n4: usize) {
    let vs = f64x2_splat(scale);
    let mut i = 0usize;
    while i < n4 {
        let w0 = v128_load(w_ptr.add(i) as *const v128);
        let q0 = v128_load(q_ptr.add(i) as *const v128);
        v128_store(w_ptr.add(i) as *mut v128, f64x2_sub(w0, f64x2_mul(vs, q0)));
        let w1 = v128_load(w_ptr.add(i + 2) as *const v128);
        let q1 = v128_load(q_ptr.add(i + 2) as *const v128);
        v128_store(w_ptr.add(i + 2) as *mut v128, f64x2_sub(w1, f64x2_mul(vs, q1)));
        i += 4;
    }
}

unsafe fn find_orthogonal_basis_seed_f64(
    ls: &mut LanczosScratch,
    n: usize,
    n4: usize,
    num_existing: usize,
    out_offset: usize,
) -> bool {
    let inv_n = 1.0f64 / (n as f64);
    for cand in 0..n {
        for idx in 0..n {
            ls.w[idx] = if idx == cand { 1.0 } else { 0.0 } - inv_n;
        }
        for idx in n..n4 {
            ls.w[idx] = 0.0;
        }
        for _pass in 0..2usize {
            for k in 0..num_existing {
                let k_off = k * GRAPH_NODES;
                let dot = dot_f64_padded4(ls.q.as_ptr().add(k_off), ls.w.as_ptr(), n4);
                sub_scaled_f64_padded4(ls.w.as_mut_ptr(), ls.q.as_ptr().add(k_off), dot, n4);
            }
            let mut mean = 0.0f64;
            for idx in 0..n {
                mean += ls.w[idx];
            }
            mean *= inv_n;
            for idx in 0..n {
                ls.w[idx] -= mean;
            }
            for idx in n..n4 {
                ls.w[idx] = 0.0;
            }
        }
        let norm2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);
        if norm2 > 1e-12 {
            let inv_norm = 1.0 / wasm_sqrt_f64(norm2);
            for idx in 0..n {
                ls.q[out_offset + idx] = ls.w[idx] * inv_norm;
            }
            for idx in n..n4 {
                ls.q[out_offset + idx] = 0.0;
            }
            return true;
        }
    }
    false
}

unsafe fn solve_tridiag_tql2_f64(ls: &mut LanczosScratch, m_dim: usize) -> (usize, i32, f64, f64) {
    let m4 = (m_dim + 3) & !3;
    for i in 0..m_dim {
        ls.d_diag[i] = ls.alpha[i];
        ls.e_sub[i] = if i + 1 < m_dim { ls.beta[i] } else { 0.0 };
        let r_off = i * GRAPH_NODES;
        ls.z[r_off..(r_off + m4)].fill(0.0);
        ls.z[r_off + i] = 1.0;
    }

    for l in 0..m_dim {
        let mut iter = 0usize;
        while iter < 40 {
            iter += 1;
            let mut m = l;
            while m < m_dim - 1 {
                if abs_f64(ls.e_sub[m])
                    <= 1e-14 * (abs_f64(ls.d_diag[m]) + abs_f64(ls.d_diag[m + 1])) + 1e-30
                {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            let mut g = (ls.d_diag[l + 1] - ls.d_diag[l]) / (2.0 * ls.e_sub[l]);
            let ag = abs_f64(g);
            let mut r = if ag < 1e100 {
                wasm_sqrt_f64(g * g + 1.0)
            } else {
                ag
            };
            g = ls.d_diag[m] - ls.d_diag[l] + ls.e_sub[l] / (g + if g >= 0.0 { r } else { -r });
            let mut s = 1.0f64;
            let mut c = 1.0f64;
            let mut p = 0.0f64;
            for i in (l..m).rev() {
                let f = s * ls.e_sub[i];
                let b = c * ls.e_sub[i];
                let r2 = f * f + g * g;
                r = if r2 > 1e-200 && r2 < 1e200 {
                    wasm_sqrt_f64(r2)
                } else {
                    hypot_f64(f, g)
                };
                ls.e_sub[i + 1] = r;
                if r < 1e-30 {
                    ls.d_diag[i + 1] -= p;
                    ls.e_sub[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = ls.d_diag[i + 1] - p;
                r = (ls.d_diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                ls.d_diag[i + 1] = g + p;
                g = c * r - b;
                let vs = f64x2_splat(s);
                let vc = f64x2_splat(c);
                let z_ptr = ls.z.as_mut_ptr();
                let row_i = i * GRAPH_NODES;
                let row_i1 = row_i + GRAPH_NODES;
                let mut k = 0usize;
                while k < m4 {
                    let fz0 = v128_load(z_ptr.add(row_i1 + k) as *const v128);
                    let zk0 = v128_load(z_ptr.add(row_i + k) as *const v128);
                    v128_store(
                        z_ptr.add(row_i1 + k) as *mut v128,
                        f64x2_add(f64x2_mul(vs, zk0), f64x2_mul(vc, fz0)),
                    );
                    v128_store(
                        z_ptr.add(row_i + k) as *mut v128,
                        f64x2_sub(f64x2_mul(vc, zk0), f64x2_mul(vs, fz0)),
                    );
                    let fz1 = v128_load(z_ptr.add(row_i1 + k + 2) as *const v128);
                    let zk1 = v128_load(z_ptr.add(row_i + k + 2) as *const v128);
                    v128_store(
                        z_ptr.add(row_i1 + k + 2) as *mut v128,
                        f64x2_add(f64x2_mul(vs, zk1), f64x2_mul(vc, fz1)),
                    );
                    v128_store(
                        z_ptr.add(row_i + k + 2) as *mut v128,
                        f64x2_sub(f64x2_mul(vc, zk1), f64x2_mul(vs, fz1)),
                    );
                    k += 4;
                }
            }
            if r < 1e-30 && ls.e_sub[m] == 0.0 {
                continue;
            }
            ls.d_diag[l] -= p;
            ls.e_sub[l] = g;
            ls.e_sub[m] = 0.0;
        }
    }

    let mut best_col = 0usize;
    let mut sec_col = -1i32;
    let mut theta_max = -1e30f64;
    let mut theta_second = -1e30f64;
    for c in 0..m_dim {
        let val = ls.d_diag[c];
        if val > theta_max {
            theta_second = theta_max;
            sec_col = best_col as i32;
            theta_max = val;
            best_col = c;
        } else if val > theta_second {
            theta_second = val;
            sec_col = c as i32;
        }
    }
    (best_col, sec_col, theta_max, theta_second)
}

#[no_mangle]
pub unsafe extern "C" fn chronos_solve_fiedler_lanczos() -> f32 {
    let a = arena_mut();
    let ls = &mut a.lanczos_scratch;
    let mut n = 0usize;
    for i in 0..GRAPH_NODES {
        if a.laplacian[i * GRAPH_NODES + i] > 1e-6 {
            ls.act_idx[n] = i;
            n += 1;
        }
    }
    a.fiedler_vec.fill(0.0);
    if n < 2 {
        ls.last_n = 0;
        ls.last_lambda2 = 0.0;
        a.fiedler_lambda2 = 0.0;
        a.fiedler_gap = 0.0;
        return 0.0;
    }

    let n4 = (n + 3) & !3;
    let mut mu = 0.0f64;
    for idx in 0..n {
        let ia = ls.act_idx[idx] * GRAPH_NODES;
        let r_off = idx * GRAPH_NODES;
        let diag = a.laplacian[ia + ls.act_idx[idx]] as f64;
        ls.sub_l[r_off + idx] = diag;
        for b in 0..idx {
            let v = a.laplacian[ia + ls.act_idx[b]] as f64;
            ls.sub_l[r_off + b] = v;
            ls.sub_l[b * GRAPH_NODES + idx] = v;
        }
        for b in n..n4 {
            ls.sub_l[r_off + b] = 0.0;
        }
    }
    for idx in 0..n {
        let r_off = idx * GRAPH_NODES;
        let mut row_sum = ls.sub_l[r_off + idx];
        for b in 0..n {
            if idx != b {
                row_sum += abs_f64(ls.sub_l[r_off + b]);
            }
        }
        if row_sum > mu {
            mu = row_sum;
        }
    }
    mu += 0.1;

    let m_steps = (n - 1).min(MAX_KRYLOV);
    let inv_n = 1.0f64 / (n as f64);
    let mut mean0 = 0.0f64;
    for idx in 0..n {
        let val = ((idx + 1) as f64) - 0.5 * ((n + 1) as f64);
        ls.q[idx] = val;
        mean0 += val;
    }
    mean0 *= inv_n;
    let mut norm0 = 0.0f64;
    for idx in 0..n {
        ls.q[idx] -= mean0;
        norm0 += ls.q[idx] * ls.q[idx];
    }
    let inv_norm0 = 1.0f64 / wasm_sqrt_f64(norm0.max(1e-18));
    for idx in 0..n {
        ls.q[idx] *= inv_norm0;
    }
    for idx in n..n4 {
        ls.q[idx] = 0.0;
        ls.w[idx] = 0.0;
    }

    let mut actual_m = 0usize;
    for j in 0..m_steps {
        let q_off = j * GRAPH_NODES;
        let q_ptr = ls.q.as_ptr().add(q_off);
        for idx in 0..n {
            let sum_l = dot_f64_padded4(ls.sub_l.as_ptr().add(idx * GRAPH_NODES), q_ptr, n4);
            ls.w[idx] = mu * ls.q[q_off + idx] - sum_l;
        }
        let mut w_mean = 0.0f64;
        for idx in 0..n {
            w_mean += ls.w[idx];
        }
        w_mean *= inv_n;
        if j > 0 && ls.beta[j - 1] > 0.0 {
            let prev_off = (j - 1) * GRAPH_NODES;
            let beta_prev = ls.beta[j - 1];
            for idx in 0..n {
                ls.w[idx] = (ls.w[idx] - w_mean) - beta_prev * ls.q[prev_off + idx];
            }
        } else {
            for idx in 0..n {
                ls.w[idx] -= w_mean;
            }
        }
        for idx in n..n4 {
            ls.w[idx] = 0.0;
        }

        let alpha = dot_f64_padded4(q_ptr, ls.w.as_ptr(), n4);
        ls.alpha[j] = alpha;
        sub_scaled_f64_padded4(ls.w.as_mut_ptr(), q_ptr, alpha, n4);
        let pre_norm2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);

        for pass in 0..2usize {
            for k in 0..=j {
                let k_ptr = ls.q.as_ptr().add(k * GRAPH_NODES);
                let dot = dot_f64_padded4(k_ptr, ls.w.as_ptr(), n4);
                sub_scaled_f64_padded4(ls.w.as_mut_ptr(), k_ptr, dot, n4);
            }
            let mut wm = 0.0f64;
            for idx in 0..n {
                wm += ls.w[idx];
            }
            wm *= inv_n;
            for idx in 0..n {
                ls.w[idx] -= wm;
            }
            for idx in n..n4 {
                ls.w[idx] = 0.0;
            }
            let post2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);
            if pass == 0 && post2 > 0.5 * pre_norm2 {
                break;
            }
        }

        let beta2 = dot_f64_padded4(ls.w.as_ptr(), ls.w.as_ptr(), n4);
        let beta = wasm_sqrt_f64(beta2.max(0.0));
        actual_m = j + 1;

        if j + 1 >= m_steps {
            ls.beta[j] = 0.0;
            break;
        }
        let next_off = (j + 1) * GRAPH_NODES;
        if beta < 1e-8 {
            ls.beta[j] = 0.0;
            if !find_orthogonal_basis_seed_f64(ls, n, n4, j + 1, next_off) {
                break;
            }
        } else {
            ls.beta[j] = beta;
            let inv_beta = 1.0 / beta;
            for idx in 0..n {
                ls.q[next_off + idx] = ls.w[idx] * inv_beta;
            }
            for idx in n..n4 {
                ls.q[next_off + idx] = 0.0;
            }
        }
    }

    let (best_col, _, theta_max, theta_second) = solve_tridiag_tql2_f64(ls, actual_m);
    let lambda2 = (mu - theta_max).max(0.0);
    let lambda3 = if actual_m >= 2 {
        (mu - theta_second).max(lambda2)
    } else {
        lambda2
    };
    let spectral_gap = (lambda3 - lambda2).max(0.0);

    let mut eig_acc0 = [f64x2_splat(0.0); GRAPH_NODES / 2];
    let best_z_off = best_col * GRAPH_NODES;
    let n2 = (n + 1) >> 1;
    for j in 0..actual_m {
        let vz = f64x2_splat(ls.z[best_z_off + j]);
        let q_ptr = ls.q.as_ptr().add(j * GRAPH_NODES);
        for u in 0..n2 {
            let qv = v128_load(q_ptr.add(u * 2) as *const v128);
            eig_acc0[u] = f64x2_add(eig_acc0[u], f64x2_mul(qv, vz));
        }
    }
    for u in 0..n2 {
        v128_store(ls.eig_vec.as_mut_ptr().add(u * 2) as *mut v128, eig_acc0[u]);
    }
    for idx in n..n4 {
        ls.eig_vec[idx] = 0.0;
    }

    let mut dot_prev = 0.0f64;
    for idx in 0..n {
        let node_idx = ls.act_idx[idx];
        if node_idx < MAX_TARGETS {
            dot_prev += ls.eig_vec[idx] * ls.prev_track[node_idx];
        }
    }

    let sign_fix = if dot_prev < -1e-4 || (abs_f64(dot_prev) <= 1e-4 && ls.eig_vec[0] < 0.0) {
        -1.0f64
    } else {
        1.0f64
    };
    for idx in 0..n {
        let node_idx = ls.act_idx[idx];
        let val = ls.eig_vec[idx] * sign_fix;
        a.fiedler_vec[node_idx] = val as f32;
        if node_idx < MAX_TARGETS {
            ls.prev_track[node_idx] = val;
        }
    }

    ls.last_n = n;
    ls.last_lambda2 = lambda2;
    let l2_f = lambda2 as f32;
    let gap_f = spectral_gap as f32;
    a.fiedler_lambda2 = if l2_f.is_finite() { l2_f.max(0.0) } else { 0.0 };
    a.fiedler_gap = if gap_f.is_finite() { gap_f.max(0.0) } else { 0.0 };
    a.fiedler_lambda2
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_fiedler_res_norm() -> f32 {
    let a = arena_mut();
    let ls = &a.lanczos_scratch;
    let n = ls.last_n;
    if n < 2 {
        return 0.0;
    }
    let n4 = (n + 3) & !3;
    let lambda2 = ls.last_lambda2;
    let mut res2 = 0.0f64;
    for idx in 0..n {
        let lv = dot_f64_padded4(
            ls.sub_l.as_ptr().add(idx * GRAPH_NODES),
            ls.eig_vec.as_ptr(),
            n4,
        );
        let diff = lv - lambda2 * ls.eig_vec[idx];
        res2 += diff * diff;
    }
    wasm_sqrt_f64(res2) as f32
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_telemetry_ptr() -> *const f32 {
    let a = arena_mut();
    a.telemetry.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_swarm_mode(mode: u32) {
    let a = arena_mut();
    let sanitized = mode.min(2);
    a.micro_swarm.swarm_mode = sanitized;
    a.micro_swarm.swarm_telem[0] = sanitized as f32;
    a.swarm_mode_cached = sanitized as u8;
    if sanitized == 0 {
        a.micro_swarm.reset_active_keep_mode();
        a.continuum.reset();
        a.micro_swarm.swarm_telem[1..soa_tracks::SWARM_TELEM_LEN].fill(0.0);
        a.continuum_was_active = 0;
        a.inspected_hero_tid = 0;
        a.inspected_micro_tid_pin = 0;
        a.swarm_telem_zeroed = 1;
        a.hodge.reset_keep_mode();
        set_stamp_invalid(&mut a.wave1_telem, WAVE1_PRODUCT_STAMPS_START + PROD_MICRO * STAMP_STRIDE);
        set_stamp_invalid(&mut a.wave1_telem, WAVE1_PRODUCT_STAMPS_START + PROD_GRAD * STAMP_STRIDE);
        set_stamp_invalid(&mut a.wave1_telem, WAVE1_PRODUCT_STAMPS_START + PROD_FTLE * STAMP_STRIDE);
        set_stamp_invalid(&mut a.wave1_telem, WAVE1_PRODUCT_STAMPS_START + PROD_MODAL_FIELD * STAMP_STRIDE);
        set_stamp_invalid(
            &mut a.wave2_telem,
            WAVE2_PRODUCT_STAMPS_START + PROD_W2_HODGE * STAMP_STRIDE,
        );
        unsafe {
            release_wave1_owned_auto_heroes(a, a.frame_idx.saturating_sub(1), mode_dims(a.res_mode).6);
        }
    } else {
        a.swarm_telem_zeroed = 0;
        if sanitized == 2 {
            a.micro_swarm.swarm_active = 1;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_swarm_mode() -> u32 {
    arena_mut().micro_swarm.swarm_mode
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_micro_tracks_ptr() -> *const f32 {
    arena_mut().micro_swarm.export_micro.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_continuum_field_ptr() -> *const f32 {
    arena_mut().continuum.export_field.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_rigid_bodies_ptr() -> *const f32 {
    arena_mut().continuum.export_bodies.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_swarm_telemetry_ptr() -> *const f32 {
    arena_mut().micro_swarm.swarm_telem.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_modal_field_ptr() -> *const f32 {
    arena_mut().continuum.export_modal_field.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_ingest_meta(
    epoch: u32,
    seq: u32,
    t_lo: u32,
    t_hi: u32,
    age_ms: f32,
    flags: u32,
    src_w: u32,
    src_h: u32,
    x0: u32,
    y0: u32,
    cw: u32,
    ch: u32,
    orientation: u32,
    scan_dir: u32,
    dx: u32,
    dy: u32,
    dw: u32,
    dh: u32,
    gap_mask: u32,
    native_1to1: u32,
) -> u32 {
    let a = arena_mut();
    if a.ingest_ctrl[CTRL_TICK_PHASE] != 0 {
        a.ingest_ctrl[CTRL_PENDING_VALID] = 0;
        a.pending_ingest_meta = IngestSlotMeta::zeroed();
        a.wave1_telem[8] = INGEST_STATUS_PHASE_ERR;
        return 0;
    }

    let (frame_w, frame_h, ..) = mode_dims(a.res_mode);
    let ew = frame_w as u32;
    let eh = frame_h as u32;

    let crop_bad = x0.checked_add(cw).map_or(true, |v| v > src_w)
        || y0.checked_add(ch).map_or(true, |v| v > src_h);
    let dst_bad = dx.checked_add(dw).map_or(true, |v| v > ew)
        || dy.checked_add(dh).map_or(true, |v| v > eh);
    let flags_bad = (flags & !0x1eu32) != 0
        || (flags & 12) == 12
        || ((flags & 12) == 0 && age_ms != -1.0)
        || ((flags & 12) != 0 && age_ms < 0.0);
    let gap_mask_bad = (gap_mask & !0x3fu32) != 0;
    let native_bad = native_1to1 == 1
        && ((flags & INGEST_FLAG_GEOM_KNOWN) == 0
            || dx != 0
            || dy != 0
            || dw != ew
            || dh != eh
            || oriented_extent(cw, ch, orientation) != (ew, eh));

    if src_w == 0
        || src_h == 0
        || cw == 0
        || ch == 0
        || dw == 0
        || dh == 0
        || crop_bad
        || dst_bad
        || orientation > 7
        || (scan_dir != u32::MAX && scan_dir > 3)
        || native_1to1 > 1
        || !age_ms.is_finite()
        || flags_bad
        || gap_mask_bad
        || native_bad
    {
        a.ingest_ctrl[CTRL_PENDING_VALID] = 0;
        a.pending_ingest_meta = IngestSlotMeta::zeroed();
        a.wave1_telem[8] = INGEST_STATUS_INVALID;
        return 0;
    }

    if a.ingest_ctrl[CTRL_DUP_GUARD_ACTIVE] != 0
        && epoch != 0
        && (a.wave1_telem[2] as usize) < RING_LEN
    {
        let prev_slot = a.wave1_telem[2] as usize;
        let prev = &a.ring_meta[prev_slot];
        if (prev.flags & INGEST_FLAG_COMMITTED) != 0
            && prev.source_epoch == epoch
            && prev.source_seq == seq
        {
            a.ingest_ctrl[CTRL_PENDING_VALID] = 0;
            a.pending_ingest_meta = IngestSlotMeta::zeroed();
            a.wave1_telem[8] = INGEST_STATUS_DUP;
            return 0;
        }
    }

    a.pending_ingest_meta = IngestSlotMeta {
        flags,
        source_epoch: epoch,
        source_seq: seq,
        engine_frame_idx: 0,
        t_src_us_lo: t_lo,
        t_src_us_hi: t_hi,
        age_ms_bits: if (flags & 12) == 0 {
            (-1.0f32).to_bits()
        } else {
            age_ms.to_bits()
        },
        src_w,
        src_h,
        crop_x0: x0,
        crop_y0: y0,
        crop_w: cw,
        crop_h: ch,
        orientation,
        scan_dir,
        engine_w: ew,
        engine_h: eh,
        dst_x: dx,
        dst_y: dy,
        dst_w: dw,
        dst_h: dh,
        frame_gap_flag: gap_mask & 0x3f,
        native_1to1_active: native_1to1,
        continuity_epoch: 0,
    };
    a.ingest_ctrl[CTRL_PENDING_VALID] = 1;
    1
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_ring_meta_ptr() -> *const u32 {
    arena_mut().ring_meta.as_ptr() as *const u32
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_row_luma_ptr() -> *const f32 {
    arena_mut().row_luma_means.as_ptr() as *const f32
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_wave1_telemetry_ptr() -> *const u32 {
    arena_mut().wave1_telem.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_bos_mode(mode: u32) -> u32 {
    let a = arena_mut();
    let ok = a.bos.set_mode(mode);
    if a.bos.get_mode() != bos::BOS_MODE_OFF {
        a.wave1_telem[7] |= 2;
    } else {
        a.wave1_telem[7] &= !2;
    }
    ok
}

#[no_mangle]
pub unsafe extern "C" fn chronos_reset_bos_ref() {
    arena_mut().bos.reset_ref();
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_bos_gauge_roi(x0: u32, y0: u32, x1: u32, y1: u32) -> u32 {
    arena_mut().bos.set_gauge_roi(x0, y0, x1, y1)
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_bos_overlay_kind(kind: u32) -> u32 {
    arena_mut().bos.set_overlay_kind(kind)
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_bos_telemetry_ptr() -> *const u32 {
    arena_mut().bos.get_telemetry_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_bos_planes_ptr() -> *const f32 {
    arena_mut().bos.get_planes_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_bos_valid_ptr() -> *const u8 {
    arena_mut().bos.get_valid_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_bos_rgba_ptr() -> *const u8 {
    arena_mut().bos.get_rgba_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_bos_solve_manufactured_test(
    case_id: u32,
    a: f32,
    b: f32,
) -> u32 {
    let arena = arena_mut();
    let (frame_w, frame_h, _, _, _, _, _) = mode_dims(arena.res_mode);
    arena.bos.solve_manufactured_test(case_id, a, b, frame_w, frame_h)
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_micro_metrology_mode(mode: u32) -> u32 {
    let a = arena_mut();
    let ok = a.micro_swarm.set_micro_metrology_mode(mode);
    if ok != 0 {
        if a.micro_swarm.get_micro_metrology_mode() != 0 {
            a.wave1_telem[7] |= 1;
        } else {
            a.wave1_telem[7] &= !1;
            set_stamp_invalid(
                &mut a.wave1_telem,
                WAVE1_PRODUCT_STAMPS_START + PROD_MICRO * STAMP_STRIDE,
            );
            let s = mode_dims(a.res_mode).6;
            release_wave1_owned_auto_heroes(a, a.frame_idx.saturating_sub(1), s);
        }
    }
    ok
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_micro_cov_ptr() -> *const f32 {
    arena_mut().micro_swarm.micro_cov_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_micro_residual_ptr() -> *const f32 {
    arena_mut().micro_swarm.micro_residual_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_micro_loom_ptr() -> *const f32 {
    arena_mut().micro_swarm.micro_loom_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_micro_quality_ptr() -> *const u32 {
    arena_mut().micro_swarm.micro_quality_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_hero_candidates_ptr() -> *const u32 {
    arena_mut().micro_swarm.hero_candidates_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_wave2_telemetry_ptr() -> *const u32 {
    arena_mut().wave2_telem.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_rs_calib(
    mode: u32,
    dt_row_us: f32,
    exposure_us: f32,
    audio_rate_hz: f32,
    max_skew_us: f32,
) -> u32 {
    let a = arena_mut();
    let ok = a
        .vibro
        .set_calib(mode, dt_row_us, exposure_us, audio_rate_hz, max_skew_us);
    a.wave2_telem[2] = wave2_enabled_bits(a);
    set_stamp_invalid(
        &mut a.wave2_telem,
        WAVE2_PRODUCT_STAMPS_START + PROD_W2_VIBRO * STAMP_STRIDE,
    );
    ok
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_audio_staging_ptr() -> *mut f32 {
    arena_mut().vibro.audio_staging_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_push_audio_block(
    num_samples: u32,
    t_end_us_lo: u32,
    t_end_us_hi: u32,
    sample_rate_hz: f32,
    est_skew_us: f32,
) -> u32 {
    let t_end_us = ((t_end_us_hi as u64) << 32) | (t_end_us_lo as u64);
    let a = arena_mut();
    let ok = a
        .vibro
        .push_audio_block(num_samples, t_end_us, sample_rate_hz, est_skew_us);
    if (a.vibro.get_mode() == vibrometry::RS_MODE_RS_WELCH && ok == 0)
        || a.vibro.get_state() != vibrometry::RS_STATE_READY
    {
        set_stamp_invalid(
            &mut a.wave2_telem,
            WAVE2_PRODUCT_STAMPS_START + PROD_W2_VIBRO * STAMP_STRIDE,
        );
    }
    ok
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_vibro_telemetry_ptr() -> *const u32 {
    arena_mut().vibro.telemetry_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_vibro_planes_ptr() -> *const f32 {
    arena_mut().vibro.planes_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_hodge_mode(mode: u32) -> u32 {
    let a = arena_mut();
    let ok = a.hodge.set_mode(mode);
    a.wave2_telem[2] = wave2_enabled_bits(a);
    if a.hodge.get_mode() == hodge::HODGE_MODE_OFF {
        set_stamp_invalid(
            &mut a.wave2_telem,
            WAVE2_PRODUCT_STAMPS_START + PROD_W2_HODGE * STAMP_STRIDE,
        );
    }
    ok
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_hodge_telemetry_ptr() -> *const u32 {
    arena_mut().hodge.telemetry_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_hodge_planes_ptr() -> *const f32 {
    arena_mut().hodge.planes_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_hodge_inject_ptr() -> *mut f32 {
    arena_mut().hodge.inject_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_hodge_solve_injected(grid_dx: f32, grid_dy: f32) -> u32 {
    let a = arena_mut();
    a.hodge.live_ready = 0;
    set_stamp_invalid(
        &mut a.wave2_telem,
        WAVE2_PRODUCT_STAMPS_START + PROD_W2_HODGE * STAMP_STRIDE,
    );
    a.hodge.solve_injected(grid_dx, grid_dy)
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_conservation_telemetry_ptr() -> *const f32 {
    arena_mut().collision_state.conservation_telemetry_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_conservation_input_ptr() -> *mut f32 {
    arena_mut().collision_state.conservation_input_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_eval_occluded_pair() -> u32 {
    let a = arena_mut();
    set_stamp_invalid(
        &mut a.wave2_telem,
        WAVE2_PRODUCT_STAMPS_START + PROD_W2_CONSERVATION * STAMP_STRIDE,
    );
    a.collision_state.eval_staged_occluded_pair()
}

#[inline(never)]
unsafe fn promote_or_bind_hero_slot(
    a: &mut Arena,
    raw_mx: f32,
    raw_my: f32,
    raw_mvx: f32,
    raw_mvy: f32,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
) -> u32 {
    let mx = if raw_mx.is_finite() {
        raw_mx.clamp(0.0, (frame_w.saturating_sub(1)) as f32)
    } else {
        (frame_w as f32) * 0.5
    };
    let my = if raw_my.is_finite() {
        raw_my.clamp(0.0, (frame_h.saturating_sub(1)) as f32)
    } else {
        (frame_h as f32) * 0.5
    };
    let mvx = if raw_mvx.is_finite() { raw_mvx } else { 0.0 };
    let mvy = if raw_mvy.is_finite() { raw_mvy } else { 0.0 };

    let match_r2 = (20.0 * s) * (20.0 * s);
    let mut best_existing: Option<usize> = None;
    let mut best_d2 = match_r2;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 && a.targets[m].confirmed != 0 && a.targets[m].track_id != 0 {
            let dx = a.targets[m].pos_x - mx;
            let dy = a.targets[m].pos_y - my;
            let d2 = dx * dx + dy * dy;
            if d2.is_finite() && d2 < best_d2 {
                best_d2 = d2;
                best_existing = Some(m);
            }
        }
    }
    if let Some(m) = best_existing {
        return a.targets[m].track_id;
    }

    let dense_cap = dense_swarm_hero_cap_active(a);
    let eff_cap = if dense_cap { DENSE_SWARM_HERO_CAP } else { MAX_TARGETS };
    let mut act_cnt = 0usize;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            act_cnt += 1;
        }
    }

    let mut chosen: Option<usize> = None;
    if act_cnt < eff_cap {
        for m in 0..MAX_TARGETS {
            if a.targets[m].active == 0 {
                chosen = Some(m);
                break;
            }
        }
    }
    if chosen.is_none() {
        let mut worst_score = 1e9f32;
        let mut worst_any_score = 1e9f32;
        let mut worst_any_m = 0usize;
        for m in 0..MAX_TARGETS {
            let t = &a.targets[m];
            if dense_cap && t.active == 0 {
                continue;
            }
            let is_pinned = (a.focus_target_id != 0 && t.track_id == a.focus_target_id)
                || (a.inspected_hero_tid != 0 && t.track_id == a.inspected_hero_tid);
            let sc = t.vis_eca_ema
                + if t.confirmed != 0 { 0.04 } else { 0.0 }
                - 0.002 * (t.coasting as f32)
                + if is_pinned { 100.0 } else { 0.0 };
            if sc < worst_any_score {
                worst_any_score = sc;
                worst_any_m = m;
            }
            if !is_pinned && sc < worst_score {
                worst_score = sc;
                chosen = Some(m);
            }
        }
        if chosen.is_none() {
            chosen = Some(worst_any_m);
        }
    }

    let m = chosen.unwrap_or(0);
    let last_cursor = a.ring_cursor.saturating_sub(1);
    let f_last = if a.ring_cursor > 0 { a.frame_idx.wrapping_sub(1) } else { 0 };
    let slot_in = (last_cursor % (RING_LEN as u64)) as usize;
    let arc_slot_in = (last_cursor as usize) & (ARC_HIST_LEN - 1);
    let s2 = s * s;
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;

    if a.targets[m].active != 0 {
        if a.targets[m].confirmed != 0 {
            let ev_x = a.targets[m].pos_x;
            let ev_y = a.targets[m].pos_y;
            archive_dormant_track(a, m, f_last, ev_x, ev_y, s);
        }
        a.all_pip_tid[m] = 0;
        let mbase = m * 8;
        for q in 0..8 {
            a.slot_meta[mbase + q] = 0.0;
        }
    }
    let clear_bit = !(1u16 << m);
    for o in 0..MAX_TARGETS {
        a.targets[o].proximity_mask &= clear_bit;
    }

    let tlx = ((mx - half_patch).max(0.0) as usize).min(frame_w.saturating_sub(patch_dim));
    let tly = ((my - half_patch).max(0.0) as usize).min(frame_h.saturating_sub(patch_dim));
    ensure_chroma!(a, tlx, tly, patch_dim, patch_dim, frame_w, frame_h);
    core::ptr::write_bytes(
        &mut a.targets[m] as *mut TargetTrack as *mut u8,
        0,
        core::mem::size_of::<TargetTrack>(),
    );
    capture_template_scaled(
        &a.luma,
        &a.chroma_rg,
        &a.chroma_by,
        &a.eca_res,
        frame_w,
        frame_h,
        scale_px,
        tlx,
        tly,
        &mut a.targets[m],
    );
    simd_ops::compute_anchor_luma_stats_simd128(
        a.targets[m].anchor_luma.as_ptr(),
        a.targets[m].tpl_weight.as_ptr(),
        &mut a.targets[m].anchor_mu_a,
        &mut a.targets[m].anchor_std_a,
    );

    let hero_tid = a.next_track_id.max(1);
    a.next_track_id = hero_tid.wrapping_add(1).max(1);

    let t = &mut a.targets[m];
    t.active = 1;
    t.confirmed = 1;
    t.just_confirmed = 1;
    t.hit_mask_8 = 0x07;
    t.owner_slot = m as u8;
    t.track_id = hero_tid;
    t.birth_frame = f_last;
    t.orig_birth_frame = f_last;
    t.entry_frame = f_last;
    t.anchor_sig = t.curr_sig;
    t.last_padic_val = 32;
    t.age = 3;
    t.pos_x = mx;
    t.pos_y = my;
    t.vel_x = mvx;
    t.vel_y = mvy;
    t.prev_inst_vx = mvx;
    t.prev_inst_vy = mvy;
    t.pred_cx = mx;
    t.pred_cy = my;
    t.entry_x = mx;
    t.entry_y = my;
    t.entry_vx = mvx;
    t.entry_vy = mvy;
    t.birth_x = mx;
    t.birth_y = my;
    t.osc_mean_x = mx;
    t.osc_mean_y = my;
    t.osc_ext_min_x = mx;
    t.osc_ext_max_x = mx;
    t.disp_fista_x = mx;
    t.disp_fista_y = my;
    t.disp_raw_x = mx;
    t.disp_raw_y = my;
    t.disp_vis = 1.0;
    t.vis_eca_ema = 0.08;
    t.step_ref_x = mx;
    t.step_ref_y = my;
    t.stim_ref_x = mx;
    t.stim_ref_y = my;
    t.axis_y = 1.0;
    let asig = t.anchor_sig;
    for sr in 0..RING_LEN {
        t.ring_raw_x[sr] = mx;
        t.ring_raw_y[sr] = my;
        t.ring_sig[sr] = asig;
        t.ring_vis[sr] = if sr == slot_in { 1 } else { 0 };
        t.fista_x[sr] = mx;
        t.fista_y[sr] = my;
    }
    t.arc_raw_x[arc_slot_in] = mx;
    t.arc_raw_y[arc_slot_in] = my;
    t.arc_vis[arc_slot_in] = 1;

    a.imm[m].reset();
    a.imm[m].active = 1;
    a.imm[m].x = [mx; 4];
    a.imm[m].y = [my; 4];
    a.imm[m].vx = [mvx; 4];
    a.imm[m].vy = [mvy; 4];
    a.imm[m].pred_x = mx;
    a.imm[m].pred_y = my;
    a.imm[m].pred_vx = mvx;
    a.imm[m].pred_vy = mvy;
    a.ode_states[m].reset();
    a.pend_states[m].reset();
    physics_id::reset_collision_slot(&mut a.collision_state, m);

    a.kalman[m].active = 1;
    a.kalman[m].current_owner = m as u8;
    a.kalman[m].mismatch_streak = 0;
    a.kalman[m]._pad = 0;
    a.kalman[m].anchor_sig = asig;
    a.kalman[m].x = mx;
    a.kalman[m].y = my;
    a.kalman[m].vx = mvx;
    a.kalman[m].vy = mvy;
    a.kalman[m].p_xx = 4.0 * s2;
    a.kalman[m].p_xv = 0.0;
    a.kalman[m].p_vv = 8.0 * s2;
    a.kalman[m].coasting = 0;
    a.kalman[m].ring_x.fill(mx);
    a.kalman[m].ring_y.fill(my);

    let base = 32 + m * 56;
    a.telemetry[base] = 1.0;
    a.telemetry[base + 1] = hero_tid as f32;
    a.telemetry[base + 2] = mx;
    a.telemetry[base + 3] = my;
    a.telemetry[base + 4] = mx;
    a.telemetry[base + 5] = my;
    a.telemetry[base + 6] = mx;
    a.telemetry[base + 7] = my;
    a.telemetry[base + 8] = mx;
    a.telemetry[base + 9] = my;
    a.telemetry[base + 10] = 1.0;
    let mut active_confirmed = 0u32;
    for k in 0..MAX_TARGETS {
        if a.targets[k].active != 0 && a.targets[k].confirmed != 0 {
            active_confirmed += 1;
        }
    }
    a.telemetry[3] = active_confirmed as f32;
    a.ingest_ctrl[CTRL_HERO_PHYS_SLOT_MASK] &= !(1u32 << m);

    hero_tid
}

#[inline(never)]
unsafe fn refresh_hero_pin_bridge(
    a: &mut Arena,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
) {
    if a.inspected_micro_tid_pin == 0 {
        return;
    }
    let mut found_ti: Option<usize> = None;
    let mut ti = 0usize;
    while ti < soa_tracks::MAX_MICRO {
        if a.micro_swarm.tracks.state[ti] != soa_tracks::MICRO_STATE_INACTIVE
            && a.micro_swarm.tracks.track_id[ti] == a.inspected_micro_tid_pin
        {
            found_ti = Some(ti);
            break;
        }
        ti += 1;
    }
    if let Some(ti) = found_ti {
        a.micro_swarm.inspected_micro_tid = a.inspected_micro_tid_pin;
        a.micro_swarm.inspected_body_id = a.micro_swarm.tracks.body_id[ti] as u32;
        a.micro_swarm.swarm_telem[11] = a.micro_swarm.inspected_micro_tid as f32;
        a.micro_swarm.swarm_telem[12] = a.micro_swarm.inspected_body_id as f32;
        let mut hero_alive = false;
        if a.inspected_hero_tid != 0 {
            for m in 0..MAX_TARGETS {
                if a.targets[m].active != 0 && a.targets[m].track_id == a.inspected_hero_tid {
                    hero_alive = true;
                    break;
                }
            }
            if !hero_alive {
                for di in 0..MAX_DORMANT {
                    if a.dormant[di].active != 0 && a.dormant[di].track_id == a.inspected_hero_tid {
                        hero_alive = true;
                        break;
                    }
                }
            }
        }
        if !hero_alive && a.micro_swarm.tracks.state[ti] == soa_tracks::MICRO_STATE_CONFIRMED {
            let mx = a.micro_swarm.tracks.pos_x[ti];
            let my = a.micro_swarm.tracks.pos_y[ti];
            let mvx = a.micro_swarm.tracks.vel_x[ti];
            let mvy = a.micro_swarm.tracks.vel_y[ti];
            let hero_tid =
                promote_or_bind_hero_slot(a, mx, my, mvx, mvy, frame_w, frame_h, scale_px, s);
            for m in 0..MAX_TARGETS {
                if a.targets[m].active != 0 && a.targets[m].track_id == hero_tid {
                    a.targets[m].wave1_auto_owned = 0;
                }
            }
            a.inspected_hero_tid = hero_tid;
            a.focus_target_id = hero_tid;
        }
    } else {
        a.inspected_micro_tid_pin = 0;
        a.inspected_hero_tid = 0;
        a.micro_swarm.inspected_micro_tid = 0;
        a.micro_swarm.inspected_body_id = 0;
        a.micro_swarm.swarm_telem[11] = 0.0;
        a.micro_swarm.swarm_telem[12] = 0.0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_inspect_field_point(x: f32, y: f32) -> u32 {
    let a = arena_mut();
    let (frame_w, frame_h, _gw, _gh, scale_px, _pip_dim, s) = mode_dims(a.res_mode);
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= (frame_w as f32)
        || y >= (frame_h as f32)
    {
        return 0;
    }

    let micro_tid =
        soa_tracks::inspect_nearest_micro_track(&mut a.micro_swarm, x, y, frame_w, frame_h);
    if micro_tid == 0 {
        a.inspected_micro_tid_pin = 0;
        a.inspected_hero_tid = 0;
        return 0;
    }

    let mut mx = x;
    let mut my = y;
    let mut mvx = 0.0f32;
    let mut mvy = 0.0f32;
    let mut ti = 0usize;
    while ti < soa_tracks::MAX_MICRO {
        if a.micro_swarm.tracks.state[ti] == soa_tracks::MICRO_STATE_CONFIRMED
            && a.micro_swarm.tracks.track_id[ti] == micro_tid
        {
            mx = a.micro_swarm.tracks.pos_x[ti];
            my = a.micro_swarm.tracks.pos_y[ti];
            mvx = a.micro_swarm.tracks.vel_x[ti];
            mvy = a.micro_swarm.tracks.vel_y[ti];
            break;
        }
        ti += 1;
    }

    let hero_tid =
        promote_or_bind_hero_slot(a, mx, my, mvx, mvy, frame_w, frame_h, scale_px, s);
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 && a.targets[m].track_id == hero_tid {
            a.targets[m].wave1_auto_owned = 0;
        }
    }
    a.focus_target_id = hero_tid;
    a.inspected_hero_tid = hero_tid;
    a.inspected_micro_tid_pin = micro_tid;
    micro_tid
}

#[no_mangle]
pub unsafe extern "C" fn chronos_inspect_micro_track(track_id: u32) -> u32 {
    let a = arena_mut();
    let (frame_w, frame_h, _gw, _gh, scale_px, _pip_dim, s) = mode_dims(a.res_mode);
    let Some((mx, my, mvx, mvy)) =
        soa_tracks::inspect_micro_track_by_id(&mut a.micro_swarm, track_id, frame_w, frame_h)
    else {
        a.focus_target_id = 0;
        a.inspected_micro_tid_pin = 0;
        a.inspected_hero_tid = 0;
        return 0;
    };
    let hero_tid =
        promote_or_bind_hero_slot(a, mx, my, mvx, mvy, frame_w, frame_h, scale_px, s);
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 && a.targets[m].track_id == hero_tid {
            a.targets[m].wave1_auto_owned = 0;
        }
    }
    a.focus_target_id = hero_tid;
    a.inspected_hero_tid = hero_tid;
    a.inspected_micro_tid_pin = track_id;
    track_id
}


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
struct PanMotion {
    scene_cut: bool,
    dx: i32,
    dy: i32,
    dx_f: f32,
    dy_f: f32,
    frac_mag: f32,
    exp_gain: f32,
    exp_off: f32,
    exp_regime: bool,
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
        }
    }
}

#[inline(always)]
fn unpack_cam_hold(raw: u32) -> (u8, u8, f32, f32) {
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
fn pack_cam_hold(cut_hold: u8, exp_hold: u8, vx_f: f32, vy_f: f32) -> u32 {
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
            cur_patch, cur_mad, pl, scale_px, phase_mask, off_x_base, off_y_base, w00, w10, w01, w11,
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
unsafe fn detect_camera_motion_coarse(
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

            let want_pan = tex_mad > 0.006 && iso2d && g_l1 > 0.18;
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

    let mut cur_patches = [[0.0f32; 64]; 8];
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

    if best_cluster >= 5 {
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
    } else if hard_cut_tiles < 8
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
    if !swarm_pan_guard && !have_pan_candidate && hard_cut_tiles < 8 && uniform_tiles >= 9 && anchored_2d_tiles == 0 {
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
    if !have_pan_candidate
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
                let mut cur_eo = [[0.0f32; 64]; 8];
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
                    let use_p2_18 = win_span == 18 && scale_px == 2 && phase_mask == 1;
                    let mut cur_patches_eo = [[0.0f32; 64]; 8];
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
                        let p_base = pid * 324;
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
                                let l_row = p_base + ly * 18;
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
                        let pl = &a.roi_luma[p_base..(p_base + 324)];
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
                            let p_base = pid * 324;
                            let pl = &a.roi_luma[p_base..(p_base + 324)];
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
            if a.cam_pan_streak >= 2 || disp2 >= (2.2 * s) * (2.2 * s) {
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
    a.cam_pan_streak = a.cam_pan_streak.saturating_sub(1);
    a.cam_pan_speed_px_s *= 0.65;
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
    }
}

#[inline(always)]
fn dense_swarm_hero_cap_active(a: &Arena) -> bool {
    a.swarm_mode_cached != 0
        && a.micro_swarm.swarm_active != 0
        && a.micro_swarm.active_count >= 128
}

#[inline(never)]
unsafe fn retire_target_slot_full(a: &mut Arena, m: usize, f_idx: u32, s: f32) {
    if m >= MAX_TARGETS {
        return;
    }
    if a.targets[m].active != 0 && a.targets[m].confirmed != 0 {
        let ev_x = a.targets[m].pos_x;
        let ev_y = a.targets[m].pos_y;
        archive_dormant_track(a, m, f_idx, ev_x, ev_y, s);
    }
    core::ptr::write_bytes(
        &mut a.targets[m] as *mut TargetTrack as *mut u8,
        0,
        core::mem::size_of::<TargetTrack>(),
    );
    retire_target_slot(a, m);
    a.all_pip_tid[m] = 0;
    let mbase = m * 8;
    for q in 0..8 {
        a.slot_meta[mbase + q] = 0.0;
    }
    let tbase = 32 + m * 56;
    for j in 0..56 {
        a.telemetry[tbase + j] = 0.0;
    }
    let pbase = TELEM_PHYSICS_START + m * 16;
    for q in 0..16 {
        a.telemetry[pbase + q] = 0.0;
    }
    a.physics_telem_dirty_mask &= !(1u16 << m);
    a.ingest_ctrl[CTRL_HERO_PHYS_SLOT_MASK] &= !(1u32 << m);
    a.ingest_ctrl[CTRL_AUTO_HERO_MASK] &= !(1u32 << m);
    let mut slots_packed = a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS];
    for k in 0..DENSE_SWARM_HERO_CAP {
        let shift = k * 8;
        if ((slots_packed >> shift) & 0xFF) as usize == m + 1 {
            slots_packed &= !(0xFFu32 << shift);
            a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k] = 0;
        }
    }
    a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS] = slots_packed;
    set_stamp_invalid(&mut a.wave1_telem, WAVE1_HERO_STAMPS_START + m * STAMP_STRIDE);

    let mut active_confirmed = 0u32;
    for k in 0..MAX_TARGETS {
        if a.targets[k].active != 0 && a.targets[k].confirmed != 0 {
            active_confirmed += 1;
        }
    }
    a.telemetry[3] = active_confirmed as f32;
}

#[inline(never)]
fn enforce_dense_swarm_hero_cap(a: &mut Arena, cap: usize, f_idx: u32, s: f32) {
    let mut act_cnt = 0usize;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            act_cnt += 1;
        }
    }
    while act_cnt > cap {
        let mut best_evict: Option<usize> = None;
        let mut worst_score = 1.0e9f32;
        for m in 0..MAX_TARGETS {
            let t = &a.targets[m];
            if t.active == 0 {
                continue;
            }
            let is_pinned = (a.focus_target_id != 0 && t.track_id == a.focus_target_id)
                || (a.inspected_hero_tid != 0 && t.track_id == a.inspected_hero_tid);
            let sc = if is_pinned { 1000.0 } else { 0.0 }
                + if t.confirmed != 0 { 100.0 } else { 0.0 }
                + if t.coasting == 0 { 20.0 } else { 0.0 }
                + 0.01 * (t.age.min(500) as f32)
                + t.vis_eca_ema
                - 0.1 * (t.coasting as f32);
            if sc < worst_score {
                worst_score = sc;
                best_evict = Some(m);
            }
        }
        let Some(m) = best_evict else { break };
        unsafe {
            retire_target_slot_full(a, m, f_idx, s);
        }
        act_cnt -= 1;
    }
}

#[inline]
fn retire_target_slot(a: &mut Arena, m: usize) {
    a.targets[m].active = 0;
    a.targets[m].confirmed = 0;
    a.targets[m].wave1_auto_owned = 0;
    a.targets[m].phys_stamp_tid = 0;
    a.kalman[m].active = 0;
    a.imm[m].reset();
    a.ode_states[m].reset();
    a.pend_states[m].reset();
    a.lyap_cache[m] = 0.0;
    physics_id::reset_collision_slot(&mut a.collision_state, m);
    let clear_bit = !(1u16 << m);
    for o in 0..MAX_TARGETS {
        a.targets[o].proximity_mask &= clear_bit;
    }
}

/// 4-Mode IMM Interaction + Time-Update (`0 = CV`, `1 = Ballistic+Drag`, `2 = CTRV`, `3 = Osc/Pendulum`).
/// Uses 4th-order Taylor `(sinc, cosc)` when `|omega * dt| < 1e-2` and clamps all state variables finite.
#[inline(never)]
fn imm_predict_step(
    imm: &mut ImmTrack,
    dt_s: f32,
    s: f32,
    half_patch: f32,
    frame_w: f32,
    frame_h: f32,
    grav_hint_s2: f32,
    grav_conf: f32,
) {
    if imm.active == 0 {
        return;
    }
    const P_IMM: [[f32; 4]; 4] = [
        [0.92, 0.03, 0.03, 0.02],
        [0.03, 0.92, 0.03, 0.02],
        [0.04, 0.02, 0.90, 0.04],
        [0.03, 0.02, 0.03, 0.92],
    ];
    let safe_dt = dt_s.clamp(0.002, 0.25);
    let safe_s = s.max(1.0);
    let s2 = safe_s * safe_s;
    let v_lim = 64.0 * safe_s;

    let mut mu = [0.0f32; 4];
    let mut mu_sum = 0.0f32;
    for i in 0..4usize {
        let m_i = if imm.mu[i].is_finite() { imm.mu[i].clamp(1e-4, 1.0) } else { 0.25 };
        mu[i] = m_i;
        mu_sum += m_i;
    }
    let inv_ms = 1.0 / mu_sum.max(1e-6);
    for i in 0..4usize {
        mu[i] *= inv_ms;
    }

    let mut c_bar = [0.0f32; 4];
    for j in 0..4usize {
        let mut s_cb = 0.0f32;
        for i in 0..4usize {
            s_cb += P_IMM[i][j] * mu[i];
        }
        c_bar[j] = s_cb.max(1e-5);
    }

    let mut x0 = [0.0f32; 4];
    let mut y0 = [0.0f32; 4];
    let mut vx0 = [0.0f32; 4];
    let mut vy0 = [0.0f32; 4];
    let mut var0 = [0.0f32; 4];
    for j in 0..4usize {
        let inv_cb = 1.0 / c_bar[j];
        let mut mx = 0.0f32;
        let mut my = 0.0f32;
        let mut mvx = 0.0f32;
        let mut mvy = 0.0f32;
        for i in 0..4usize {
            let wij = P_IMM[i][j] * mu[i] * inv_cb;
            mx += wij * imm.x[i];
            my += wij * imm.y[i];
            mvx += wij * imm.vx[i];
            mvy += wij * imm.vy[i];
        }
        let mut mv = 0.0f32;
        for i in 0..4usize {
            let wij = P_IMM[i][j] * mu[i] * inv_cb;
            let dx = imm.x[i] - mx;
            let dy = imm.y[i] - my;
            mv += wij * (imm.pos_var[i] + 0.5 * (dx * dx + dy * dy));
        }
        x0[j] = mx;
        y0[j] = my;
        vx0[j] = mvx.clamp(-v_lim, v_lim);
        vy0[j] = mvy.clamp(-v_lim, v_lim);
        var0[j] = mv.clamp(0.5 * s2, 180.0 * s2);
    }

    let g_eff_s2 = if grav_conf > 0.35 && grav_hint_s2 >= 120.0 * safe_s {
        0.5 * grav_hint_s2 + 0.5 * imm.g_bal_s2
    } else {
        imm.g_bal_s2
    }
    .clamp(150.0 * safe_s, 3600.0 * safe_s);
    let g_step = g_eff_s2 * safe_dt * safe_dt;

    // Mode 0: Constant Velocity (CV)
    imm.x[0] = (x0[0] + vx0[0]).clamp(-64.0 * safe_s, frame_w + 64.0 * safe_s);
    imm.y[0] = (y0[0] + vy0[0]).clamp(-64.0 * safe_s, frame_h + 64.0 * safe_s);
    imm.vx[0] = vx0[0];
    imm.vy[0] = vy0[0];
    imm.pos_var[0] = (var0[0] + 0.85 * s2).clamp(0.5 * s2, 240.0 * s2);

    // Mode 1: Ballistic + Quadratic Aero Drag
    let spd1 = wasm_sqrtf(vx0[1] * vx0[1] + vy0[1] * vy0[1]);
    let drag_k = (imm.beta_drag * spd1).clamp(0.0, 0.22);
    let ax1 = -drag_k * vx0[1];
    let ay1 = g_step - drag_k * vy0[1];
    imm.x[1] = (x0[1] + vx0[1] + 0.5 * ax1).clamp(-64.0 * safe_s, frame_w + 64.0 * safe_s);
    imm.y[1] = (y0[1] + vy0[1] + 0.5 * ay1).clamp(-64.0 * safe_s, frame_h + 64.0 * safe_s);
    imm.vx[1] = (vx0[1] + ax1).clamp(-v_lim, v_lim);
    imm.vy[1] = (vy0[1] + ay1).clamp(-v_lim, v_lim);
    imm.pos_var[1] = (var0[1] + 0.70 * s2).clamp(0.5 * s2, 240.0 * safe_s);

    // Mode 2: Constant Turn-Rate & Velocity (CTRV) with 4th-order Taylor when |psi| < 1e-2
    let psi = imm.omega_turn.clamp(-0.35, 0.35);
    let (sinc, cosc) = if psi.abs() < 1e-2 {
        let p2 = psi * psi;
        (
            1.0 - p2 * (1.0 / 6.0) * (1.0 - p2 * (1.0 / 20.0)),
            0.5 * psi * (1.0 - p2 * (1.0 / 12.0)),
        )
    } else {
        (fast_sinf(psi) / psi, (1.0 - fast_cosf(psi)) / psi)
    };
    let cp = fast_cosf(psi);
    let sp = fast_sinf(psi);
    imm.x[2] = (x0[2] + (vx0[2] * sinc - vy0[2] * cosc))
        .clamp(-64.0 * safe_s, frame_w + 64.0 * safe_s);
    imm.y[2] = (y0[2] + (vx0[2] * cosc + vy0[2] * sinc))
        .clamp(-64.0 * safe_s, frame_h + 64.0 * safe_s);
    imm.vx[2] = (vx0[2] * cp - vy0[2] * sp).clamp(-v_lim, v_lim);
    imm.vy[2] = (vx0[2] * sp + vy0[2] * cp).clamp(-v_lim, v_lim);
    imm.pos_var[2] = (var0[2] + 0.90 * s2).clamp(0.5 * s2, 240.0 * safe_s);

    // Mode 3: Oscillatory / Pendulum (Velocity-Verlet harmonic + soft tether projection when pivoted)
    let k_spr = if imm.pivot_valid != 0 && imm.piv_r >= 12.0 * safe_s {
        (g_eff_s2 / imm.piv_r).clamp(0.5, 220.0) * safe_dt * safe_dt
    } else {
        let wdt = (imm.omega_osc * safe_dt).clamp(0.02, 0.48);
        wdt * wdt
    }
    .clamp(0.0004, 0.25);
    let dx_eq0 = x0[3] - imm.eq_x;
    let dy_eq0 = y0[3] - imm.eq_y;
    let mut px3 = x0[3] + vx0[3] - 0.5 * k_spr * dx_eq0;
    let mut py3 = y0[3] + vy0[3] - 0.5 * k_spr * dy_eq0;
    if imm.pivot_valid != 0 && imm.piv_r >= 12.0 * safe_s {
        let rdx = px3 - imm.piv_x;
        let rdy = py3 - imm.piv_y;
        let rd = wasm_sqrtf((rdx * rdx + rdy * rdy).max(1e-4));
        if rd > 0.5 * imm.piv_r && rd < 1.5 * imm.piv_r {
            let corr = 0.35 * (imm.piv_r / rd - 1.0);
            px3 += rdx * corr;
            py3 += rdy * corr;
        }
    }
    imm.x[3] = px3.clamp(-64.0 * safe_s, frame_w + 64.0 * safe_s);
    imm.y[3] = py3.clamp(-64.0 * safe_s, frame_h + 64.0 * safe_s);
    imm.vx[3] = (vx0[3] - 0.5 * k_spr * (dx_eq0 + (px3 - imm.eq_x))).clamp(-v_lim, v_lim);
    imm.vy[3] = (vy0[3] - 0.5 * k_spr * (dy_eq0 + (py3 - imm.eq_y))).clamp(-v_lim, v_lim);
    imm.pos_var[3] = (var0[3] + 0.85 * s2).clamp(0.5 * s2, 240.0 * safe_s);

    let mut cb_sum = 0.0f32;
    for j in 0..4usize {
        cb_sum += c_bar[j];
    }
    let inv_cb_sum = 1.0 / cb_sum.max(1e-6);
    let mut px_mix = 0.0f32;
    let mut py_mix = 0.0f32;
    let mut pvx_mix = 0.0f32;
    let mut pvy_mix = 0.0f32;
    let mut dom = 0usize;
    let mut max_mu = -1.0f32;
    for j in 0..4usize {
        c_bar[j] *= inv_cb_sum;
        imm.mu[j] = c_bar[j];
        if c_bar[j] > max_mu {
            max_mu = c_bar[j];
            dom = j;
        }
        px_mix += c_bar[j] * imm.x[j];
        py_mix += c_bar[j] * imm.y[j];
        pvx_mix += c_bar[j] * imm.vx[j];
        pvy_mix += c_bar[j] * imm.vy[j];
    }
    imm.mode_dom = dom as u8;
    imm.pred_x = if px_mix.is_finite() {
        px_mix.clamp(half_patch, frame_w - half_patch)
    } else {
        x0[0].clamp(half_patch, frame_w - half_patch)
    };
    imm.pred_y = if py_mix.is_finite() {
        py_mix.clamp(half_patch, frame_h - half_patch)
    } else {
        y0[0].clamp(half_patch, frame_h - half_patch)
    };
    imm.pred_vx = if pvx_mix.is_finite() { pvx_mix } else { 0.0 };
    imm.pred_vy = if pvy_mix.is_finite() { pvy_mix } else { 0.0 };
}

/// 4-Mode IMM Measurement Update (Gaussian likelihood posterior + per-mode Kalman correction).
#[inline(never)]
fn imm_update_obs(
    imm: &mut ImmTrack,
    obs_x: f32,
    obs_y: f32,
    inst_vx: f32,
    inst_vy: f32,
    prev_vx: f32,
    prev_vy: f32,
    t: &TargetTrack,
    dt_s: f32,
    s: f32,
    observed: bool,
) {
    let safe_dt = dt_s.clamp(0.002, 0.25);
    let safe_s = s.max(1.0);
    let s2 = safe_s * safe_s;
    let v_lim = 56.0 * safe_s;

    if !observed {
        if imm.active == 0 {
            return;
        }
        for j in 0..4usize {
            imm.pos_var[j] = (imm.pos_var[j] + 0.5 * s2).clamp(0.5 * s2, 240.0 * s2);
        }
        return;
    }

    if imm.active == 0 || !obs_x.is_finite() || !obs_y.is_finite() {
        imm.mu = [0.46, 0.18, 0.18, 0.18];
        imm.x = [obs_x; 4];
        imm.y = [obs_y; 4];
        imm.vx = [t.vel_x.clamp(-v_lim, v_lim); 4];
        imm.vy = [t.vel_y.clamp(-v_lim, v_lim); 4];
        imm.pos_var = [2.5 * s2; 4];
        imm.omega_turn = t.ang_vel_rad_f.clamp(-0.35, 0.35);
        imm.g_bal_s2 = 980.0 * safe_s;
        imm.beta_drag = 0.0;
        imm.eq_x = if t.osc_mean_x.is_finite() { t.osc_mean_x } else { obs_x };
        imm.eq_y = if t.osc_mean_y.is_finite() { t.osc_mean_y } else { obs_y };
        imm.omega_osc = 3.1415927;
        imm.pivot_valid = if t.pivot_valid > 0 && t.pivot_r >= 10.0 * safe_s { 1 } else { 0 };
        imm.piv_x = t.pivot_x;
        imm.piv_y = t.pivot_y;
        imm.piv_r = t.pivot_r;
        imm.pred_x = obs_x;
        imm.pred_y = obs_y;
        imm.pred_vx = t.vel_x;
        imm.pred_vy = t.vel_y;
        imm.mode_dom = 0;
        imm.active = 1;
        return;
    }

    imm.eq_x = if t.osc_mean_x.is_finite() { t.osc_mean_x } else { obs_x };
    imm.eq_y = if t.osc_mean_y.is_finite() { t.osc_mean_y } else { obs_y };
    if t.pivot_valid > 0 && t.pivot_r >= 10.0 * safe_s && t.pivot_x.is_finite() && t.pivot_y.is_finite() {
        imm.pivot_valid = 1;
        imm.piv_x = t.pivot_x;
        imm.piv_y = t.pivot_y;
        imm.piv_r = t.pivot_r;
    } else {
        imm.pivot_valid = 0;
    }
    if t.ang_vel_rad_f.abs() > 0.015 {
        imm.omega_turn = (0.70 * imm.omega_turn + 0.30 * t.ang_vel_rad_f).clamp(-0.35, 0.35);
    } else {
        imm.omega_turn *= 0.85;
    }
    if t.freq_valid != 0 && t.tremor_freq_hz > 0.15 {
        imm.omega_osc = (TWO_PI * t.tremor_freq_hz).clamp(0.8, 95.0);
    }
    if t.age >= 3 && t.coasting == 0 && t.reemerged_last == 0 {
        let ay_f = inst_vy - prev_vy;
        let g_obs_s2 = ay_f / (safe_dt * safe_dt);
        if (t.is_ballistic != 0 || imm.mu[1] > 0.20 || g_obs_s2 >= 180.0 * safe_s)
            && g_obs_s2 >= 150.0 * safe_s
            && g_obs_s2 <= 3600.0 * safe_s
        {
            imm.g_bal_s2 = (0.85 * imm.g_bal_s2 + 0.15 * g_obs_s2).clamp(150.0 * safe_s, 3600.0 * safe_s);
        }
        let spd = wasm_sqrtf(inst_vx * inst_vx + inst_vy * inst_vy);
        if spd >= 3.0 * safe_s {
            let ax_res = inst_vx - prev_vx;
            let ay_res = (inst_vy - prev_vy) - imm.g_bal_s2 * safe_dt * safe_dt;
            let a_tan = (ax_res * inst_vx + ay_res * inst_vy) / spd;
            if a_tan < -0.05 * safe_s {
                let beta_obs = (-a_tan / (spd * spd)).clamp(0.0, 0.025 / safe_s);
                imm.beta_drag = (0.88 * imm.beta_drag + 0.12 * beta_obs).clamp(0.0, 0.025 / safe_s);
            } else {
                imm.beta_drag *= 0.96;
            }
        }
    } else if t.reemerged_last != 0 {
        for j in 0..4usize {
            imm.vx[j] = (0.5 * imm.vx[j] + 0.5 * t.vel_x).clamp(-v_lim, v_lim);
            imm.vy[j] = (0.5 * imm.vy[j] + 0.5 * t.vel_y).clamp(-v_lim, v_lim);
        }
    }

    let meas_r = 0.75 * s2;
    let mut log_post = [0.0f32; 4];
    let mut max_lp = -1e30f32;
    for j in 0..4usize {
        let ex = obs_x - imm.x[j];
        let ey = obs_y - imm.y[j];
        let e2 = ex * ex + ey * ey;
        let s_j = (imm.pos_var[j] + meas_r).clamp(0.5 * s2, 260.0 * s2);
        let maha2 = (e2 / s_j).clamp(0.0, 24.0);
        let lp = fast_lnf(imm.mu[j].max(1e-4)) - fast_lnf(s_j) - 0.5 * maha2;
        log_post[j] = lp;
        if lp > max_lp {
            max_lp = lp;
        }
        let k_pos = (imm.pos_var[j] / s_j).clamp(0.10, 0.88);
        let k_vel = 0.42 * k_pos;
        imm.x[j] += k_pos * ex;
        imm.y[j] += k_pos * ey;
        imm.vx[j] = (imm.vx[j] + k_vel * ex).clamp(-v_lim, v_lim);
        imm.vy[j] = (imm.vy[j] + k_vel * ey).clamp(-v_lim, v_lim);
        imm.pos_var[j] = ((1.0 - k_pos) * imm.pos_var[j]).clamp(0.35 * s2, 120.0 * s2);
    }

    let mut w_sum = 0.0f32;
    let mut mu_new = [0.25f32; 4];
    for j in 0..4usize {
        let w = fast_expf((log_post[j] - max_lp).clamp(-20.0, 0.0)).max(1e-4);
        mu_new[j] = w;
        w_sum += w;
    }
    let inv_ws = 1.0 / w_sum.max(1e-6);
    let mut dom = 0usize;
    let mut max_mu = -1.0f32;
    let mut px_m = 0.0f32;
    let mut py_m = 0.0f32;
    let mut pvx_m = 0.0f32;
    let mut pvy_m = 0.0f32;
    for j in 0..4usize {
        let mj = mu_new[j] * inv_ws;
        imm.mu[j] = mj;
        if mj > max_mu {
            max_mu = mj;
            dom = j;
        }
        px_m += mj * imm.x[j];
        py_m += mj * imm.y[j];
        pvx_m += mj * imm.vx[j];
        pvy_m += mj * imm.vy[j];
    }
    imm.mode_dom = dom as u8;
    imm.pred_x = if px_m.is_finite() { px_m } else { obs_x };
    imm.pred_y = if py_m.is_finite() { py_m } else { obs_y };
    imm.pred_vx = if pvx_m.is_finite() { pvx_m } else { 0.0 };
    imm.pred_vy = if pvy_m.is_finite() { pvy_m } else { 0.0 };
}

/// Phase 1 of frame tick:
/// 1. Records monotonic presentation timestamps & detects discontinuities (`dt > 0.25 s`).
/// 2. Unpacks RGBA8 into SIMD128 Luma/Opponent-Chroma planes & runs ego-motion Pan / Cut registration.
/// 3. Evaluates guarded Rank-2 ECA on the pan-warped background & builds the `160x90` coarse pyramid.
/// 4. Extracts Connected-Component energy peaks and assembles the pre-assignment 16x16 Graph Laplacian.
#[no_mangle]
pub unsafe extern "C" fn chronos_pre_tick(dt_sec: f32) {
    let a = arena_mut();
    if a.ingest_ctrl[CTRL_TICK_PHASE] != 0 {
        a.wave1_telem[8] = INGEST_STATUS_PHASE_ERR;
        return;
    }
    a.wave1_telem[8] = INGEST_STATUS_EMPTY;

    let (frame_w, frame_h, grid_w, grid_h, scale_px, _pip_dim, s) = mode_dims(a.res_mode);
    let frame_pixels = frame_w * frame_h;
    let half_patch = 8.0 * s;
    let f_idx = a.frame_idx;
    let cursor = a.ring_cursor;
    let prev_pre_seen = a.pre_tick_seen;
    let prev_pre_frame = a.last_pre_tick_frame;
    let need_swarm_inval_check = a.swarm_mode_cached != 0 || a.swarm_telem_zeroed == 0;
    let engine_wrapped = prev_pre_seen != 0 && f_idx == 0 && prev_pre_frame == u32::MAX;
    let f_discont_wrap = prev_pre_seen != 0
        && !engine_wrapped
        && ((f_idx == 0
            && (prev_pre_frame != 0
                || (need_swarm_inval_check
                    && (a.micro_swarm.active_count != 0
                        || a.continuum.initialized != 0))))
            || (f_idx != prev_pre_frame && f_idx != prev_pre_frame.wrapping_add(1)));
    let mut discont_any = f_discont_wrap;
    if f_discont_wrap && need_swarm_inval_check {
        continuum_physics::invalidate_continuum_continuity(&mut a.micro_swarm, &mut a.continuum);
    }
    a.last_pre_tick_frame = f_idx;
    a.pre_tick_seen = 1;
    a.last_scene_cut = 0;
    a.last_pan_dx_f = 0.0;
    a.last_pan_dy_f = 0.0;
    let slot_in = ring_slot_back_cursor(cursor, 0);
    let slot_prev = if cursor == 0 {
        slot_in
    } else {
        ring_slot_back_cursor(cursor, 1)
    };
    let eff_k = (a.lookahead_k as u64).min(cursor).min((RING_LEN - 1) as u64) as usize;
    let slot_disp = ring_slot_back_cursor(cursor, eff_k);
    a.last_display_slot = slot_disp as u32;
    a.active_rgba_slot = slot_in as u32;
    a.chroma_tile_mask = if (scale_px <= 1 && frame_w == 320 && frame_h == 180)
        || (scale_px == 2 && frame_w == 640 && frame_h == 360)
    {
        [0u64; 24]
    } else {
        [!0u64; 24]
    };

    // Wave-1 100% unconditional ring ingest metadata consume & continuity check (prior to downstream solvers)
    let had_active_stream = a.ingest_ctrl[CTRL_DUP_GUARD_ACTIVE] != 0;
    let prev_slot = a.wave1_telem[2] as usize;
    let prev_meta = if prev_slot < RING_LEN {
        a.ring_meta[prev_slot]
    } else {
        IngestSlotMeta::zeroed()
    };
    a.ring_meta[slot_in].flags = 0;

    let staged_valid = a.ingest_ctrl[CTRL_PENDING_VALID] != 0
        && a.pending_ingest_meta.engine_w == (frame_w as u32)
        && a.pending_ingest_meta.engine_h == (frame_h as u32);
    let staged_copy = if staged_valid {
        let m = a.pending_ingest_meta;
        a.ingest_ctrl[CTRL_PENDING_VALID] = 0;
        a.pending_ingest_meta = IngestSlotMeta::zeroed();
        m
    } else {
        if a.ingest_ctrl[CTRL_PENDING_VALID] != 0 {
            a.ingest_ctrl[CTRL_PENDING_VALID] = 0;
            a.pending_ingest_meta = IngestSlotMeta::zeroed();
        }
        IngestSlotMeta::zeroed()
    };

    // Staged IngestSlotMeta is strictly authoritative over legacy chronos_set_ingest_pts.
    let legacy_pts_valid = !staged_valid && a.has_explicit_pts != 0;
    let legacy_pts_sec = a.pending_pts_sec;
    a.has_explicit_pts = 0;

    let mut meta = if staged_valid {
        staged_copy
    } else {
        let (flags, t_lo, t_hi) = if legacy_pts_valid {
            let t_us = (legacy_pts_sec.max(0.0) * 1_000_000.0 + 0.5) as u64;
            (
                INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_GEOM_KNOWN,
                t_us as u32,
                (t_us >> 32) as u32,
            )
        } else {
            (INGEST_FLAG_GEOM_KNOWN, 0u32, 0u32)
        };
        IngestSlotMeta {
            flags,
            source_epoch: 0,
            source_seq: f_idx,
            engine_frame_idx: f_idx,
            t_src_us_lo: t_lo,
            t_src_us_hi: t_hi,
            age_ms_bits: (-1.0f32).to_bits(),
            src_w: frame_w as u32,
            src_h: frame_h as u32,
            crop_x0: 0,
            crop_y0: 0,
            crop_w: frame_w as u32,
            crop_h: frame_h as u32,
            orientation: 0,
            scan_dir: u32::MAX,
            engine_w: frame_w as u32,
            engine_h: frame_h as u32,
            dst_x: 0,
            dst_y: 0,
            dst_w: frame_w as u32,
            dst_h: frame_h as u32,
            frame_gap_flag: 0,
            native_1to1_active: 0,
            continuity_epoch: 0,
        }
    };
    meta.flags &= !INGEST_FLAG_COMMITTED;
    meta.engine_frame_idx = f_idx;

    let mut gap = meta.frame_gap_flag & 0x3fu32;
    if engine_wrapped {
        gap |= FRAME_GAP_WRAP;
    } else if f_discont_wrap {
        gap |= FRAME_GAP_SEEK_RESTART;
    }

    if had_active_stream && (prev_meta.flags & INGEST_FLAG_COMMITTED) != 0 {
        if meta.source_epoch == prev_meta.source_epoch {
            if meta.source_epoch != 0 && meta.source_seq != prev_meta.source_seq.wrapping_add(1) {
                gap |= FRAME_GAP_SEQ;
            }
            if (meta.flags & INGEST_FLAG_PTS_KNOWN) != 0
                && (prev_meta.flags & INGEST_FLAG_PTS_KNOWN) != 0
            {
                let cur_us = ((meta.t_src_us_hi as u64) << 32) | (meta.t_src_us_lo as u64);
                let prev_us = ((prev_meta.t_src_us_hi as u64) << 32) | (prev_meta.t_src_us_lo as u64);
                if cur_us <= prev_us {
                    gap |= FRAME_GAP_NONINC_PTS;
                    a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
                } else {
                    let dt_us = cur_us - prev_us;
                    let nom = a.ingest_ctrl[CTRL_NOM_DT_US];
                    let seq_clean =
                        (gap & (FRAME_GAP_SEQ | FRAME_GAP_SEEK_RESTART | FRAME_GAP_WRAP)) == 0;
                    if dt_us > 250_000 {
                        gap |= FRAME_GAP_EXCESS_DT;
                        a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
                    } else if nom > 0 && (dt_us as u128) * 2 > (nom as u128) * 3 {
                        let clamped_us = dt_us.clamp(1_000, 250_000) as u32;
                        let prev_excess = a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US];
                        if seq_clean
                            && prev_excess > 0
                            && (clamped_us as u64) * 4 >= (prev_excess as u64) * 3
                            && (clamped_us as u64) * 4 <= (prev_excess as u64) * 5
                        {
                            a.ingest_ctrl[CTRL_NOM_DT_US] =
                                (((prev_excess as u64) + (clamped_us as u64)) / 2) as u32;
                            a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
                        } else {
                            gap |= FRAME_GAP_EXCESS_DT;
                            a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] =
                                if seq_clean { clamped_us } else { 0 };
                        }
                    } else if seq_clean {
                        let clamped_us = dt_us.clamp(1_000, 250_000) as u32;
                        let next_nom = if nom == 0 {
                            clamped_us
                        } else {
                            (((nom as u64) * 7 + (clamped_us as u64)) / 8) as u32
                        };
                        a.ingest_ctrl[CTRL_NOM_DT_US] = next_nom;
                        a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
                    } else {
                        a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
                    }
                }
            } else {
                a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
                if (meta.flags & INGEST_FLAG_PTS_KNOWN) == 0
                    && dt_sec.is_finite()
                    && dt_sec > 0.25
                {
                    gap |= FRAME_GAP_EXCESS_DT;
                }
            }
        } else {
            gap |= FRAME_GAP_SEEK_RESTART;
            a.ingest_ctrl[CTRL_NOM_DT_US] = 0;
            a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
        }

        let geom_changed = ((meta.flags & INGEST_FLAG_GEOM_KNOWN) != 0
            && (prev_meta.flags & INGEST_FLAG_GEOM_KNOWN) != 0
            && (meta.src_w != prev_meta.src_w
                || meta.src_h != prev_meta.src_h
                || meta.crop_x0 != prev_meta.crop_x0
                || meta.crop_y0 != prev_meta.crop_y0
                || meta.crop_w != prev_meta.crop_w
                || meta.crop_h != prev_meta.crop_h
                || meta.orientation != prev_meta.orientation
                || meta.scan_dir != prev_meta.scan_dir
                || meta.engine_w != prev_meta.engine_w
                || meta.engine_h != prev_meta.engine_h
                || meta.dst_x != prev_meta.dst_x
                || meta.dst_y != prev_meta.dst_y
                || meta.dst_w != prev_meta.dst_w
                || meta.dst_h != prev_meta.dst_h
                || meta.native_1to1_active != prev_meta.native_1to1_active))
            || meta.engine_w != prev_meta.engine_w
            || meta.engine_h != prev_meta.engine_h;
        if geom_changed {
            gap |= FRAME_GAP_GEOM_CHANGE;
            a.ingest_ctrl[CTRL_PREV_EXCESS_DT_US] = 0;
        }
    }

    // 0. Update rebased monotonic presentation timestamp ring `hist_t_sec` (`f64`),
    //    differencing `u64` timestamps in integer precision before any float conversion.
    let fallback_step = if dt_sec.is_finite() && dt_sec > 1e-4 {
        (dt_sec as f64).min(0.25)
    } else {
        1.0 / 60.0
    };
    let mut step_dt_f = fallback_step as f32;
    let is_initial_frame = f_idx == 0 && prev_pre_seen == 0;
    if is_initial_frame {
        a.cum_time_sec = 0.0;
        a.hist_t_sec[0] = 0.0;
        if staged_valid && (meta.flags & INGEST_FLAG_PTS_KNOWN) != 0 {
            let cur_us = ((meta.t_src_us_hi as u64) << 32) | (meta.t_src_us_lo as u64);
            a.prev_raw_pts_sec = (cur_us as f64) * 1e-6;
        } else if legacy_pts_valid {
            a.prev_raw_pts_sec = legacy_pts_sec;
        } else {
            a.prev_raw_pts_sec = 0.0;
        }
    } else if staged_valid && (meta.flags & INGEST_FLAG_PTS_KNOWN) != 0 {
        let cur_us = ((meta.t_src_us_hi as u64) << 32) | (meta.t_src_us_lo as u64);
        a.prev_raw_pts_sec = (cur_us as f64) * 1e-6;
        let has_prev_u64 = had_active_stream
            && (prev_meta.flags & INGEST_FLAG_COMMITTED) != 0
            && (prev_meta.flags & INGEST_FLAG_PTS_KNOWN) != 0;
        let raw_step = if has_prev_u64 {
            let prev_us = ((prev_meta.t_src_us_hi as u64) << 32) | (prev_meta.t_src_us_lo as u64);
            if cur_us >= prev_us {
                ((cur_us - prev_us) as f64) * 1e-6
            } else {
                -(((prev_us - cur_us) as f64) * 1e-6)
            }
        } else {
            fallback_step
        };
        if has_prev_u64 && (!raw_step.is_finite() || raw_step < -0.02 || raw_step > 0.25) {
            a.last_discont_frame = f_idx;
            a.discont_count = a.discont_count.saturating_add(1);
            continuum_physics::invalidate_continuum_continuity(&mut a.micro_swarm, &mut a.continuum);
            discont_any = true;
        }
        let mono_step = if raw_step.is_finite() && raw_step > 5e-4 {
            raw_step.min(1.0)
        } else {
            fallback_step
        };
        a.cum_time_sec += mono_step;
        a.hist_t_sec[(f_idx as usize) & HIST_MASK] = a.cum_time_sec;
        step_dt_f = (mono_step as f32).clamp(0.002, 0.25);
    } else if !staged_valid && legacy_pts_valid {
        let raw_step = legacy_pts_sec - a.prev_raw_pts_sec;
        a.prev_raw_pts_sec = legacy_pts_sec;
        if !raw_step.is_finite() || raw_step < -0.02 || raw_step > 0.25 {
            a.last_discont_frame = f_idx;
            a.discont_count = a.discont_count.saturating_add(1);
            continuum_physics::invalidate_continuum_continuity(&mut a.micro_swarm, &mut a.continuum);
            discont_any = true;
        }
        let mono_step = if raw_step.is_finite() && raw_step > 5e-4 {
            raw_step.min(1.0)
        } else {
            fallback_step
        };
        a.cum_time_sec += mono_step;
        a.hist_t_sec[(f_idx as usize) & HIST_MASK] = a.cum_time_sec;
        step_dt_f = (mono_step as f32).clamp(0.002, 0.25);
    } else {
        let raw_step = if dt_sec.is_finite() && dt_sec > 1e-4 {
            dt_sec as f64
        } else {
            1.0 / 60.0
        };
        if raw_step > 0.25 {
            a.last_discont_frame = f_idx;
            a.discont_count = a.discont_count.saturating_add(1);
            continuum_physics::invalidate_continuum_continuity(&mut a.micro_swarm, &mut a.continuum);
            discont_any = true;
        }
        a.cum_time_sec += raw_step.min(1.0);
        a.hist_t_sec[(f_idx as usize) & HIST_MASK] = a.cum_time_sec;
        step_dt_f = (raw_step as f32).clamp(0.002, 0.25);
    }

    if (gap
        & (FRAME_GAP_SEQ
            | FRAME_GAP_NONINC_PTS
            | FRAME_GAP_EXCESS_DT
            | FRAME_GAP_SEEK_RESTART
            | FRAME_GAP_GEOM_CHANGE
            | FRAME_GAP_SCENE_CUT))
        != 0
    {
        discont_any = true;
    }

    let mut epoch_bumped = false;
    let prev_cont_ep = a.wave1_telem[6];
    if a.wave1_telem[6] == 0 {
        a.wave1_telem[6] = 1;
        epoch_bumped = true;
    } else if discont_any {
        a.wave1_telem[6] = a.wave1_telem[6].wrapping_add(1).max(1);
        epoch_bumped = true;
        continuum_physics::invalidate_continuum_continuity(&mut a.micro_swarm, &mut a.continuum);
        a.inspected_hero_tid = 0;
        a.inspected_micro_tid_pin = 0;
    }

    // 1. Fused SIMD128 Planar Unpack + Coarse Luma/Chroma Pyramid (eliminates 3 full-frame read passes)
    let y_sq_sub = rgba_to_planar_and_pool160_simd128(
        a.rgba_ring[slot_in].as_ptr(),
        a.luma.as_mut_ptr(),
        a.chroma_rg.as_mut_ptr(),
        a.chroma_by.as_mut_ptr(),
        a.pyr_luma.as_mut_ptr(),
        a.pyr_rg.as_mut_ptr(),
        a.pyr_by.as_mut_ptr(),
        frame_w,
        frame_h,
        scale_px,
    );
    compute_row_luma_means_simd128(&a.luma, &mut a.row_luma_means[slot_in], frame_w, frame_h);
    let y_cnt_sub = ((frame_pixels + 15) / 16).max(1) as f32;
    let rms_y = wasm_sqrtf(y_sq_sub / y_cnt_sub);

    let pan = detect_camera_motion_coarse(
        a,
        frame_w,
        frame_h,
        frame_pixels,
        scale_px,
        slot_in,
        slot_prev,
        s,
        step_dt_f,
    );
    a.last_pan_dx_f = pan.dx_f;
    a.last_pan_dy_f = pan.dy_f;
    a.last_scene_cut = if pan.scene_cut { 1 } else { 0 };
    if pan.scene_cut {
        gap |= FRAME_GAP_SCENE_CUT;
        continuum_physics::invalidate_continuum_continuity(&mut a.micro_swarm, &mut a.continuum);
        a.inspected_hero_tid = 0;
        a.inspected_micro_tid_pin = 0;
        if !epoch_bumped {
            a.wave1_telem[6] = a.wave1_telem[6].wrapping_add(1).max(1);
        }
    }

    // Commit ring_meta[slot_in] and Wave-1 ingest counters BEFORE Rank-2 ECA / CC / Laplacian solvers.
    let committed_flags = meta.flags | INGEST_FLAG_COMMITTED;
    meta.flags = committed_flags;
    meta.frame_gap_flag = gap;
    meta.continuity_epoch = a.wave1_telem[6];
    if prev_cont_ep != 0 && meta.continuity_epoch != prev_cont_ep {
        invalidate_stale_wave1_stamps(&mut a.wave1_telem, meta.continuity_epoch);
        invalidate_stale_wave2_stamps(&mut a.wave2_telem, meta.continuity_epoch);
        a.vibro.reset_continuity(if pan.scene_cut {
            vibrometry::RS_REASON_FRAME_GAP
        } else {
            0
        });
        a.hodge.reset_keep_mode();
        a.collision_state.conservation.reset_continuity();
    }
    a.wave2_telem[3] = meta.continuity_epoch;
    a.ring_meta[slot_in] = meta;
    a.wave1_telem[0] = WAVE1_ABI_VERSION;
    a.wave1_telem[1] = core::mem::size_of::<Arena>() as u32;
    a.wave1_telem[2] = slot_in as u32;
    a.wave1_telem[3] = slot_disp as u32;
    let prev_commits = ((a.wave1_telem[5] as u64) << 32) | (a.wave1_telem[4] as u64);
    let next_commits = prev_commits.wrapping_add(1);
    a.wave1_telem[4] = next_commits as u32;
    a.wave1_telem[5] = (next_commits >> 32) as u32;
    a.wave1_telem[7] = wave1_enabled_bits(a);
    a.wave1_telem[8] = INGEST_STATUS_OK;
    a.wave1_telem[9] = committed_flags;
    a.wave1_telem[10] = frame_h as u32;
    a.wave1_telem[11] = INGEST_RING_STRIDE_WORDS;
    a.wave1_telem[12] = linked_static_end_bytes();
    a.wave1_telem[13] = 1;
    a.wave1_telem[14] = 0;
    a.wave1_telem[15] = 0;
    a.ingest_ctrl[CTRL_DUP_GUARD_ACTIVE] = 1;
    a.ingest_ctrl[CTRL_TICK_PHASE] = 1;
    a.ingest_ctrl[CTRL_HERO_PHYS_RAN] = 0;
    a.ingest_ctrl[CTRL_HERO_PHYS_SLOT_MASK] = 0;

    // 2. Construct independent stationary-clutter evaluation mask on pan-registered frames,
    //    then evaluate Rank-2 ECA `X = [b, 1]` against pan-warped `bg_luma`.
    let mut rgb_sum_cached = false;
    let mut pan_subpx_fused = false;
    if !pan.scene_cut {
        let stat_single_pass = cursor > 0
            && !is_pan_or_exp_active(
                pan.dx,
                pan.dy,
                pan.frac_mag,
                pan.exp_gain,
                pan.exp_off,
                pan.exp_regime,
            )
            && frame_w == grid_w * 16
            && frame_h >= grid_h * 16;
        let mask_warm = f_idx >= 12
            && (a.last_discont_frame == 0 || f_idx >= a.last_discont_frame.saturating_add(12));
        let mask_ptr = if mask_warm {
            a.bg_eval_mask.as_ptr()
        } else {
            core::ptr::null()
        };
        let mask_mut_ptr = if mask_warm {
            a.bg_eval_mask.as_mut_ptr()
        } else {
            core::ptr::null_mut()
        };
        let mask_luma_tol = if mask_warm
            && a.clutter_eval_ratio >= 0.14
            && (pan.dx != 0 || pan.dy != 0 || pan.frac_mag >= 0.04)
            && rms_y < 0.42
        {
            (0.027 * rms_y).clamp(0.0088, 0.0125)
        } else {
            0.0f32
        };
        let mut err2 = 0.0f32;
        let mut sig2 = 0.0f32;
        let mut eval_count = 0u32;
        let mut stat_fused = false;
        let subpx_fused_ratio = if cursor > 0
            && !stat_single_pass
            && frame_w == grid_w * 16
            && frame_h >= grid_h * 16
        {
            if pan.frac_mag >= 0.04 {
                pan_streak_and_eca_fused_subpx_simd128(
                    a.rgba_ring[slot_in].as_ptr(),
                    a.rgba_ring[slot_prev].as_ptr(),
                    a.luma.as_ptr(),
                    a.bg_luma.as_mut_ptr(),
                    a.bg_stat_streak.as_mut_ptr(),
                    a.bg_eval_mask.as_mut_ptr(),
                    mask_warm,
                    mask_luma_tol,
                    a.eca_res.as_mut_ptr(),
                    a.pyr_eca.as_mut_ptr(),
                    a.cc_energy.as_mut_ptr(),
                    a.roi_luma.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    grid_w,
                    grid_h,
                    scale_px,
                    a.eca_adapt_rate,
                    pan.dx_f,
                    pan.dy_f,
                    pan.frac_mag,
                    pan.exp_gain,
                    pan.exp_off,
                    pan.exp_regime,
                    &mut err2,
                    &mut sig2,
                    &mut eval_count,
                )
            } else {
                simd_ops::pan_streak_and_eca_fused_int_simd128(
                    a.rgba_ring[slot_in].as_ptr(),
                    a.rgba_ring[slot_prev].as_ptr(),
                    a.luma.as_ptr(),
                    a.bg_luma.as_mut_ptr(),
                    a.bg_stat_streak.as_mut_ptr(),
                    a.bg_eval_mask.as_mut_ptr(),
                    mask_warm,
                    mask_luma_tol,
                    a.eca_res.as_mut_ptr(),
                    a.pyr_eca.as_mut_ptr(),
                    a.cc_energy.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    grid_w,
                    grid_h,
                    scale_px,
                    a.eca_adapt_rate,
                    pan.dx,
                    pan.dy,
                    pan.frac_mag,
                    pan.exp_gain,
                    pan.exp_off,
                    pan.exp_regime,
                    rms_y,
                    &mut err2,
                    &mut sig2,
                    &mut eval_count,
                )
            }
        } else {
            None
        };
        if let Some(_r) = subpx_fused_ratio {
            pan_subpx_fused = true;
        } else {
            if cursor > 0 && !stat_single_pass {
                rgb_sum_cached = update_stat_streak_and_mask_pan_simd128(
                    a.rgba_ring[slot_in].as_ptr(),
                    a.rgba_ring[slot_prev].as_ptr(),
                    a.bg_stat_streak.as_mut_ptr(),
                    a.bg_eval_mask.as_mut_ptr(),
                    a.eca_res.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    pan.dx,
                    pan.dy,
                    pan.dx_f,
                    pan.dy_f,
                    pan.frac_mag,
                    pan.exp_gain,
                    pan.exp_off,
                    pan.exp_regime,
                    rms_y,
                    scale_px,
                );
            }
            stat_fused =
                !rgb_sum_cached && cursor > 0 && frame_w == grid_w * 16 && frame_h >= grid_h * 16;
            let _ratio = if rgb_sum_cached {
                eca_and_motion_grid_pan_fused_simd128(
                    a.luma.as_ptr(),
                    a.bg_luma.as_mut_ptr(),
                    mask_mut_ptr,
                    mask_luma_tol,
                    a.eca_res.as_mut_ptr(),
                    a.pyr_eca.as_mut_ptr(),
                    a.cc_energy.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    grid_w,
                    grid_h,
                    scale_px,
                    a.eca_adapt_rate,
                    pan.dx,
                    pan.dy,
                    pan.dx_f,
                    pan.dy_f,
                    pan.frac_mag,
                    pan.exp_gain,
                    pan.exp_off,
                    pan.exp_regime,
                    &mut err2,
                    &mut sig2,
                    &mut eval_count,
                )
            } else if stat_fused {
                eca_and_motion_grid_stat_fused_simd128(
                    a.rgba_ring[slot_in].as_ptr(),
                    a.rgba_ring[slot_prev].as_ptr(),
                    a.luma.as_ptr(),
                    a.bg_luma.as_mut_ptr(),
                    a.bg_eval_mask.as_mut_ptr(),
                    a.bg_stat_streak.as_mut_ptr(),
                    a.eca_res.as_mut_ptr(),
                    a.pyr_eca.as_mut_ptr(),
                    a.cc_energy.as_mut_ptr(),
                    frame_w,
                    frame_h,
                    grid_w,
                    grid_h,
                    scale_px,
                    a.eca_adapt_rate,
                    mask_warm,
                    stat_single_pass,
                    a.micro_swarm.swarm_active != 0 && a.micro_swarm.active_count >= 512,
                    &mut err2,
                    &mut sig2,
                    &mut eval_count,
                )
            } else {
                eca_subspace_project_masked_simd128(
                    a.luma.as_ptr(),
                    a.bg_luma.as_mut_ptr(),
                    mask_ptr,
                    a.eca_res.as_mut_ptr(),
                    frame_pixels,
                    a.eca_adapt_rate,
                    &mut err2,
                    &mut sig2,
                    &mut eval_count,
                )
            };
        }
        if (rgb_sum_cached || stat_fused || pan_subpx_fused)
            && scale_px <= 2
            && frame_w == 320 * scale_px
        {
            let py_start = grid_h * (8 / scale_px);
            if py_start < 90 {
                pool_plane_rows_to_160x90_simd128(
                    a.eca_res.as_ptr(),
                    a.pyr_eca.as_mut_ptr(),
                    scale_px,
                    py_start,
                    90,
                );
            }
        } else {
            pool_plane_to_160x90_simd128(a.eca_res.as_ptr(), a.pyr_eca.as_mut_ptr(), scale_px);
        }

        a.clutter_eval_ratio = (eval_count as f32) * (1.0 / (frame_pixels as f32));
        if eval_count >= 1024 && sig2 > 1e-3 {
            if a.clutter_db_valid == 0 {
                a.eca_err2_ema = err2;
                a.eca_sig2_ema = sig2;
                a.clutter_db_valid = 1;
            } else {
                a.eca_err2_ema = 0.85 * a.eca_err2_ema + 0.15 * err2;
                a.eca_sig2_ema = 0.85 * a.eca_sig2_ema + 0.15 * sig2;
            }
            let uncensored_ratio = (a.eca_err2_ema / a.eca_sig2_ema.max(1e-6)).max(1e-12);
            a.clutter_db = (-10.0 * fast_log10f(uncensored_ratio)).clamp(0.0, 96.0);
        } else if a.clutter_db_valid == 0 {
            a.clutter_db = 0.0;
        }
        if cursor > 0 && !rgb_sum_cached && !stat_fused && !pan_subpx_fused {
            compute_motion_grid_and_heal_bg_simd128(
                a.rgba_ring[slot_in].as_ptr(),
                a.rgba_ring[slot_prev].as_ptr(),
                a.luma.as_ptr(),
                a.bg_luma.as_mut_ptr(),
                a.eca_res.as_ptr(),
                a.cc_energy.as_mut_ptr(),
                frame_w,
                frame_h,
                grid_w,
                grid_h,
            );
        }
    }


    // 3. Connected-Component Peaks (active through camera pans!)
    let skip_hero_cc_peaks = if dense_swarm_hero_cap_active(a) && a.inspected_micro_tid_pin == 0 {
        let mut live_conf_heroes = 0usize;
        for m in 0..MAX_TARGETS {
            let t = &a.targets[m];
            if t.active != 0 && t.confirmed != 0 && t.coasting == 0 {
                live_conf_heroes += 1;
            }
        }
        live_conf_heroes >= DENSE_SWARM_HERO_CAP
    } else {
        false
    };
    if cursor > 0 && !pan.scene_cut && !skip_hero_cc_peaks {
        extract_connected_component_peaks(a);
    } else {
        a.num_peaks = 0;
    }

    // 4. Assemble Pre-Assignment 16x16 Track-Peak Graph Laplacian L = D - A
    //    Nodes 0..7: second-order ballistic predicted active tracks; Nodes 8..15: candidate peaks.
    for i in 0..(GRAPH_NODES * GRAPH_NODES) {
        a.laplacian[i] = 0.0;
    }

    let mut node_active = [false; GRAPH_NODES];
    let mut node_x = [0.0f32; GRAPH_NODES];
    let mut node_y = [0.0f32; GRAPH_NODES];
    let mut node_sig = [0u32; GRAPH_NODES];
    let lap_peak_cap = if dense_swarm_hero_cap_active(a) {
        5usize.min(MAX_TARGETS)
    } else {
        MAX_TARGETS
    };

    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            let coasting = a.targets[m].coasting;
            imm_predict_step(
                &mut a.imm[m],
                step_dt_f,
                s,
                half_patch,
                frame_w as f32,
                frame_h as f32,
                a.scene_gravity_px_s2,
                a.scene_gravity_conf,
            );
            let (leg_x, leg_y) = if coasting == 0 {
                let ay_pred = if a.targets[m].is_ballistic != 0 && a.targets[m].grav_ay_frame > 0.1 {
                    a.targets[m].prev_inst_vy + a.targets[m].grav_ay_frame
                } else {
                    a.targets[m].vel_y + 0.35 * a.targets[m].acc_y
                };
                (
                    (a.targets[m].pos_x + a.targets[m].vel_x + 0.35 * a.targets[m].acc_x)
                        .clamp(half_patch, (frame_w as f32) - half_patch),
                    (a.targets[m].pos_y + ay_pred)
                        .clamp(half_patch, (frame_h as f32) - half_patch),
                )
            } else {
                let cf = (f_idx.saturating_sub(a.targets[m].entry_frame).max(1)) as f32;
                let ca = cf.min(8.0);
                let (cx_raw, cy_raw, _, _) = predict_curved_step(
                    a.targets[m].entry_x,
                    a.targets[m].entry_y,
                    a.targets[m].entry_vx,
                    a.targets[m].entry_vy,
                    a.targets[m].entry_omega_f,
                    a.targets[m].grav_ay_frame,
                    a.targets[m].is_ballistic,
                    cf,
                );
                let use_lin_acc = a.targets[m].is_ballistic == 0
                    && a.targets[m].entry_omega_f.abs() <= 0.045;
                let ax_off = if use_lin_acc { 0.20 * ca * ca * a.targets[m].entry_ax } else { 0.0 };
                let ay_off = if use_lin_acc { 0.20 * ca * ca * a.targets[m].entry_ay } else { 0.0 };
                (
                    (cx_raw + ax_off).clamp(half_patch, (frame_w as f32) - half_patch),
                    (cy_raw + ay_off).clamp(half_patch, (frame_h as f32) - half_patch),
                )
            };
            let (px, py) = if a.imm[m].active != 0 && a.targets[m].age >= 4 {
                let w_imm = if coasting == 0 {
                    0.28
                } else if a.imm[m].mu[3] > 0.45 || a.imm[m].mu[1] > 0.55 {
                    0.35
                } else {
                    0.0
                };
                (
                    ((1.0 - w_imm) * leg_x + w_imm * a.imm[m].pred_x)
                        .clamp(half_patch, (frame_w as f32) - half_patch),
                    ((1.0 - w_imm) * leg_y + w_imm * a.imm[m].pred_y)
                        .clamp(half_patch, (frame_h as f32) - half_patch),
                )
            } else {
                (leg_x, leg_y)
            };
            a.targets[m].pred_cx = px;
            a.targets[m].pred_cy = py;

            node_active[m] = true;
            node_x[m] = px;
            node_y[m] = py;
            node_sig[m] = a.targets[m].anchor_sig;
        }
        if m < a.num_peaks && m < lap_peak_cap && a.peaks[m].valid != 0 {
            node_active[MAX_TARGETS + m] = true;
            node_x[MAX_TARGETS + m] = a.peaks[m].x;
            node_y[MAX_TARGETS + m] = a.peaks[m].y;
            node_sig[MAX_TARGETS + m] = a.peaks[m].sig;
        }
    }

    let lap_sigma = 28.0 * s;
    let inv_2sig2 = 1.0 / (2.0 * lap_sigma * lap_sigma);
    let mut deg = [0.0f32; GRAPH_NODES];
    for i in 0..GRAPH_NODES {
        if !node_active[i] {
            continue;
        }
        let xi = node_x[i];
        let yi = node_y[i];
        let si = node_sig[i];
        for j in (i + 1)..GRAPH_NODES {
            if !node_active[j] {
                continue;
            }
            let padic_aff = if has_opposite_chroma_pole(si, node_sig[j]) {
                0.0
            } else {
                let val = padic_ultrametric_val(si, node_sig[j]);
                (1.0 - padic_distance_from_val(val)).max(0.05)
            };

            let dx = xi - node_x[j];
            let dy = yi - node_y[j];
            let kin_aff = fast_expf(-(dx * dx + dy * dy) * inv_2sig2);

            // Macro-geometric spatial coupling (0.08 * kin_aff) + 2-adic ultrametric affinity
            let w = 1e-3 + 0.08 * kin_aff + padic_aff * (0.12 + 0.80 * kin_aff);
            a.laplacian[i * GRAPH_NODES + j] = -w;
            a.laplacian[j * GRAPH_NODES + i] = -w;
            deg[i] += w;
            deg[j] += w;
        }
        a.laplacian[i * GRAPH_NODES + i] = deg[i];
    }
}

/// Measures pre- and post-impact velocity chords around a sparse L1 acceleration impulse `strike_u`
/// inside the 12-frame ring using true per-slot presentation timestamps `hist_t_sec`, skipping the
/// exposure-straddling frame `strike_u` when multi-frame flanking chords exist.
/// Returns `(in_deg, signed_deflect_deg, speed_ratio, out_vx_s, out_vy_s, pre_valid)`.
fn measure_strike_kinematics(
    t: &TargetTrack,
    slot_in: usize,
    strike_u: usize,
    s: f32,
    f_idx: u32,
    hist_t_sec: &[f64; HIST_RING_LEN],
    last_discont_frame: u32,
    fallback_dt: f32,
) -> Option<(f32, f32, f32, f32, f32, bool)> {
    if strike_u == 0 || strike_u + 1 >= RING_LEN || (f_idx as usize) + strike_u < (RING_LEN - 1) {
        return None;
    }
    let mut rx = [0.0f32; RING_LEN];
    let mut ry = [0.0f32; RING_LEN];
    let mut rvis = [0u8; RING_LEN];
    for u in 0..RING_LEN {
        let su = (slot_in + 1 + u) % RING_LEN;
        rx[u] = t.ring_raw_x[su];
        ry[u] = t.ring_raw_y[su];
        let mut vu = t.ring_vis[su];
        if (f_idx as usize) + u >= (RING_LEN - 1) {
            let fu = ((f_idx as usize) + u - (RING_LEN - 1)) as u32;
            if last_discont_frame > 0 && fu < last_discont_frame {
                vu = 0;
            }
        } else {
            vu = 0;
        }
        rvis[u] = vu;
    }

    let u_b = if strike_u >= 2 && rvis[strike_u - 1] != 0 && rvis[strike_u - 2] != 0 {
        strike_u - 1
    } else {
        strike_u
    };
    let mut u_a = u_b;
    let min_a = u_b.saturating_sub(3);
    while u_a > min_a && rvis[u_a - 1] != 0 {
        u_a -= 1;
    }
    if u_a == u_b || rvis[u_a] == 0 || rvis[u_b] == 0 || (f_idx as usize) + u_a < (RING_LEN - 1) {
        return None;
    }
    let f_a = ((f_idx as usize) + u_a - (RING_LEN - 1)) as u32;
    let f_b = ((f_idx as usize) + u_b - (RING_LEN - 1)) as u32;
    let dt_pre_s = frame_dt_span_sec(hist_t_sec, f_a, f_b, fallback_dt);
    let vpre_x = (rx[u_b] - rx[u_a]) / dt_pre_s;
    let vpre_y = (ry[u_b] - ry[u_a]) / dt_pre_s;

    let u_c = if strike_u + 2 < RING_LEN && rvis[strike_u + 1] != 0 && rvis[strike_u + 2] != 0 {
        strike_u + 1
    } else {
        strike_u
    };
    let mut u_d = u_c;
    let max_d = (u_c + 3).min(RING_LEN - 1);
    while u_d < max_d && rvis[u_d + 1] != 0 {
        u_d += 1;
    }
    if u_d == u_c || rvis[u_c] == 0 || rvis[u_d] == 0 || (f_idx as usize) + u_c < (RING_LEN - 1) {
        return None;
    }
    let f_c = ((f_idx as usize) + u_c - (RING_LEN - 1)) as u32;
    let f_d = ((f_idx as usize) + u_d - (RING_LEN - 1)) as u32;
    let dt_post_s = frame_dt_span_sec(hist_t_sec, f_c, f_d, fallback_dt);
    let vpost_x = (rx[u_d] - rx[u_c]) / dt_post_s;
    let vpost_y = (ry[u_d] - ry[u_c]) / dt_post_s;

    let pre_mag = wasm_sqrtf(vpre_x * vpre_x + vpre_y * vpre_y);
    let post_mag = wasm_sqrtf(vpost_x * vpost_x + vpost_y * vpost_y);
    let inv_dt_ref = 1.0 / fallback_dt.max(1e-3);
    let rad_to_deg = 180.0 / PI;

    if pre_mag >= 0.75 * s * inv_dt_ref {
        let in_rad = fast_atan2f(-vpre_y, vpre_x);
        let in_deg = in_rad * rad_to_deg;
        let speed_ratio = (post_mag / pre_mag).clamp(0.0, 2.5);
        let defl_deg = if post_mag >= 0.40 * s * inv_dt_ref {
            let out_rad = fast_atan2f(-vpost_y, vpost_x);
            wrap_pi(out_rad - in_rad) * rad_to_deg
        } else {
            0.0
        };
        Some((in_deg, defl_deg, speed_ratio, vpost_x, vpost_y, true))
    } else if post_mag >= 1.15 * s * inv_dt_ref {
        // Object ball struck from near-rest: export outgoing velocity (px/s) for 2-body scatter wedge
        let out_deg = fast_atan2f(-vpost_y, vpost_x) * rad_to_deg;
        Some((out_deg, 0.0, 1.0, vpost_x, vpost_y, false))
    } else {
        None
    }
}

/// Exact 2x2 centered Kasa algebraic circle fit on raw sub-pixel IC-LK observations.
/// Unbiased by FISTA L1 piecewise-linear boundary flattening; returns `(xc, yc, rad, mean_y, sag)`.
fn fit_kasa_circle_subpx(
    pts_x: &[f32],
    pts_y: &[f32],
    n: usize,
    s: f32,
    min_sag_mult: f32,
    max_rmse_mult: f32,
) -> Option<(f32, f32, f32, f32, f32)> {
    if n < 7 {
        return None;
    }
    let inv_n = 1.0 / (n as f32);
    let mut mx = 0.0f32;
    let mut my = 0.0f32;
    let mut min_x = 1e9f32;
    let mut max_x = -1e9f32;
    let mut min_y = 1e9f32;
    let mut max_y = -1e9f32;
    for i in 0..n {
        let x = pts_x[i];
        let y = pts_y[i];
        mx += x;
        my += y;
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }
    let span = wasm_sqrtf((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y));
    if span < 14.0 * s {
        return None;
    }
    mx *= inv_n;
    my *= inv_n;

    let mut s_uu = 0.0f32;
    let mut s_uv = 0.0f32;
    let mut s_vv = 0.0f32;
    let mut b_u = 0.0f32;
    let mut b_v = 0.0f32;
    for i in 0..n {
        let ui = pts_x[i] - mx;
        let vi = pts_y[i] - my;
        let zi = ui * ui + vi * vi;
        s_uu += ui * ui;
        s_uv += ui * vi;
        s_vv += vi * vi;
        b_u += 0.5 * ui * zi;
        b_v += 0.5 * vi * zi;
    }
    let tr = s_uu + s_vv;
    let det = s_uu * s_vv - s_uv * s_uv;
    if tr <= 1.0 || det <= 5e-4 * tr * tr {
        return None;
    }
    let inv_det = 1.0 / det;
    let uc = (s_vv * b_u - s_uv * b_v) * inv_det;
    let vc = (s_uu * b_v - s_uv * b_u) * inv_det;
    let xc = mx + uc;
    let yc = my + vc;
    let rad = wasm_sqrtf(uc * uc + vc * vc + tr * inv_n);
    let sag = (span * span) / (8.0 * rad.max(1.0));
    if rad < 22.0 * s || rad > 440.0 * s || sag < min_sag_mult * s {
        return None;
    }
    let mut sum_er2 = 0.0f32;
    for i in 0..n {
        let rx = pts_x[i] - xc;
        let ry = pts_y[i] - yc;
        let er = wasm_sqrtf(rx * rx + ry * ry) - rad;
        sum_er2 += er * er;
    }
    let rmse_r = wasm_sqrtf(sum_er2 * inv_n);
    if rmse_r > max_rmse_mult * s {
        return None;
    }
    Some((xc, yc, rad, my, sag))
}

/// Evaluates per-track #![no_std] physics identification, Taubin arc & Galileo peg detection,
/// Jacobi AGM elliptic period, damping zeta/Q, energy & angular momentum, Kuramoto synchrony,
/// Lyapunov divergence, and two-body collision conservation, populating telemetry `1280..1568`.
#[inline(never)]
fn update_physics_and_telemetry(
    a: &mut Arena,
    f_idx: u32,
    frame_w: usize,
    frame_h: usize,
    s: f32,
    fps: f32,
    eff_k: usize,
    safe_dt: f32,
) -> bool {
    let half_patch = 8.0 * s;
    let disp_f = f_idx.wrapping_sub(eff_k as u32);
    let head_slot = (f_idx as usize) & HIST_MASK;
    let cam_hx = a.hist_cam_x[head_slot];
    let cam_hy = a.hist_cam_y[head_slot];
    let disp_cam_x = a.hist_cam_x[(disp_f as usize) & HIST_MASK];
    let disp_cam_y = a.hist_cam_y[(disp_f as usize) & HIST_MASK];
    let pan_dh_x = cam_hx - disp_cam_x;
    let pan_dh_y = cam_hy - disp_cam_y;
    let t_head_sec = a.hist_t_sec[head_slot] as f32;
    let t_disp_sec = a.hist_t_sec[(disp_f as usize) & HIST_MASK] as f32;
    let u_disp = (RING_LEN - 1).saturating_sub(eff_k);
    let f_at_u = |u: usize| -> u32 {
        f_idx.wrapping_sub((RING_LEN - 1 - u) as u32)
    };

    if dense_swarm_hero_cap_active(a)
        && a.inspected_micro_tid_pin == 0
        && a.focus_target_id == 0
        && (a.micro_swarm.confirmed_count >= 960 || f_idx < 21 || (f_idx & 7) != 5)
    {
        a.ingest_ctrl[CTRL_HERO_PHYS_SLOT_MASK] = 0;
        for m in 0..MAX_TARGETS {
            let bit = 1u16 << m;
            if a.targets[m].active != 0 {
                let (bx_cur, by_cur, vis_cur) = if a.targets[m].coasting == 0 {
                    (a.targets[m].pos_x, a.targets[m].pos_y, 1u8)
                } else {
                    (a.targets[m].arc_raw_x[head_slot], a.targets[m].arc_raw_y[head_slot], 0u8)
                };
                physics_id::push_chaos_reservoir(
                    &mut a.pend_states[m],
                    f_idx,
                    bx_cur + cam_hx,
                    by_cur + cam_hy,
                    a.hist_t_sec[head_slot],
                    vis_cur,
                );
            } else {
                a.lyap_cache[m] = 0.0;
                a.targets[m].phys_stamp_tid = 0;
            }
            if a.targets[m].active == 0
                || a.targets[m].confirmed == 0
                || a.telemetry[32 + m * 56] < 0.5
            {
                if a.targets[m].confirmed == 0 {
                    a.lyap_cache[m] = 0.0;
                    a.targets[m].phys_stamp_tid = 0;
                }
                if (a.physics_telem_dirty_mask & bit) != 0 {
                    let b = TELEM_PHYSICS_START + m * 16;
                    for q in 0..16 {
                        a.telemetry[b + q] = 0.0;
                    }
                    a.physics_telem_dirty_mask &= !bit;
                }
            }
        }
        let z_start = if a.physics_telem_dirty_mask == 0 {
            TELEM_ENSEMBLE_START
        } else {
            TELEM_ENSEMBLE_START + 14
        };
        for ti in z_start..TELEMETRY_LEN {
            a.telemetry[ti] = 0.0;
        }
        return false;
    }

    let mut taubin_fits = [physics_id::TaubinFitResult::zeroed(); MAX_TARGETS];
    let mut osc_samples = [physics_id::OscSample::zeroed(); MAX_TARGETS];
    let mut active_arr = [0u8; MAX_TARGETS];
    let mut conf_arr = [0u8; MAX_TARGETS];
    let mut bal_arr = [0u8; MAX_TARGETS];
    let mut piv_arr = [0u8; MAX_TARGETS];
    let mut span_arr = [0.0f32; MAX_TARGETS];
    let mut ode_jump_arr = [0u8; MAX_TARGETS];
    let mut peg_jump_arr = [false; MAX_TARGETS];
    let mut phys_reg_arr = [0u8; MAX_TARGETS];
    let mut phys_conf_arr = [0.0f32; MAX_TARGETS];
    let mut ode_r2_arr = [0.0f32; MAX_TARGETS];
    let mut drag_beta_arr = [0.0f32; MAX_TARGETS];
    let mut thrust_arr = [0.0f32; MAX_TARGETS];
    let mut period_arr = [0.0f32; MAX_TARGETS];
    let mut zeta_arr = [0.0f32; MAX_TARGETS];
    let mut q_arr = [0.0f32; MAX_TARGETS];
    let mut energy_arr = [0.0f32; MAX_TARGETS];
    let mut ang_mom_arr = [0.0f32; MAX_TARGETS];
    let mut disp_vx_arr = [0.0f32; MAX_TARGETS];
    let mut disp_vy_arr = [0.0f32; MAX_TARGETS];

    for m in 0..MAX_TARGETS {
        if a.targets[m].active == 0 {
            a.lyap_cache[m] = 0.0;
            continue;
        }
        active_arr[m] = 1;
        conf_arr[m] = a.targets[m].confirmed;
        bal_arr[m] = a.targets[m].is_ballistic;

        let (dvx, dvy) = if a.targets[m].age < 2 {
            (0.0, 0.0)
        } else if u_disp >= 1
            && u_disp + 1 < RING_LEN
            && a.ring_cursor >= ((RING_LEN - u_disp) as u64)
            && (a.last_discont_frame == 0
                || f_at_u(u_disp - 1).wrapping_sub(a.last_discont_frame) < 0x8000_0000)
        {
            let dt_2 = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp - 1), f_at_u(u_disp + 1), safe_dt);
            (
                (a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp - 1]) / dt_2,
                (a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp - 1]) / dt_2,
            )
        } else if u_disp + 1 < RING_LEN
            && a.ring_cursor >= ((RING_LEN - 1 - u_disp) as u64)
            && (a.last_discont_frame == 0
                || f_at_u(u_disp).wrapping_sub(a.last_discont_frame) < 0x8000_0000)
        {
            let dt_1 = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp), f_at_u(u_disp + 1), safe_dt);
            (
                (a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp]) / dt_1,
                (a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp]) / dt_1,
            )
        } else {
            let dt_1 = frame_dt_span_sec(&a.hist_t_sec, f_at_u(10), f_at_u(11), safe_dt);
            (
                (a.targets[m].fista_x[11] - a.targets[m].fista_x[10]) / dt_1,
                (a.targets[m].fista_y[11] - a.targets[m].fista_y[10]) / dt_1,
            )
        };
        disp_vx_arr[m] = if dvx.is_finite() { dvx } else { 0.0 };
        disp_vy_arr[m] = if dvy.is_finite() { dvy } else { 0.0 };

        let (bx_cur, by_cur, vis_cur) = if a.targets[m].coasting == 0 {
            (a.targets[m].pos_x, a.targets[m].pos_y, 1u8)
        } else {
            (a.targets[m].arc_raw_x[head_slot], a.targets[m].arc_raw_y[head_slot], 0u8)
        };
        physics_id::push_chaos_reservoir(
            &mut a.pend_states[m],
            f_idx,
            bx_cur + cam_hx,
            by_cur + cam_hy,
            a.hist_t_sec[head_slot],
            vis_cur,
        );

        if a.targets[m].confirmed == 0 {
            a.lyap_cache[m] = 0.0;
            continue;
        }

        let start_f = f_idx
            .saturating_sub((HIST_RING_LEN - 1) as u32)
            .max(a.targets[m].birth_frame)
            .max(a.last_discont_frame);
        let n_win = physics_id::extract_compensated_window_from_ring(
            &a.targets[m].arc_raw_x,
            &a.targets[m].arc_raw_y,
            &a.targets[m].arc_vis,
            &a.hist_cam_x,
            &a.hist_cam_y,
            &a.hist_t_sec,
            f_idx,
            start_f,
            &mut a.phys_xs,
            &mut a.phys_ys,
            &mut a.phys_ts,
            &mut a.phys_vis,
        );
        let frames_since_disc = if a.targets[m].last_ode_disc_frame == 0 {
            u32::MAX
        } else {
            f_idx.saturating_sub(a.targets[m].last_ode_disc_frame)
        };
        let vis_flicker = (a.targets[m].hit_mask_8 & 3) == 1;
        let age = a.targets[m].age;
        let is_bal = a.targets[m].is_ballistic != 0;
        let prior_fl = a.ode_states[m].prior_flight != 0;
        let is_osc_lock = a.targets[m].osc_regime == 2
            || (a.targets[m].osc_regime != 0 && a.targets[m].freq_valid != 0);
        let cv_now = a.lab_mode == 0
            && !is_bal
            && !prior_fl
            && a.targets[m].pivot_valid == 0
            && !is_osc_lock
            && a.ode_states[m].last_regime == physics_id::PHYS_REGIME_LINEAR_CV
            && a.ode_states[m].latched_jump_code == 0;
        let mut vis_idx = [0u8; 64];
        let mut n_vis = 0usize;
        let mut x0_arc = 0.0f32;
        let mut y0_arc = 0.0f32;
        let mut xf_arc = 0.0f32;
        let mut yf_arc = 0.0f32;
        let mut max_d2_arc = 0.0f32;
        let mut min_x = 1e9f32;
        let mut max_x = -1e9f32;
        let mut min_y = 1e9f32;
        let mut max_y = -1e9f32;
        let mut sum_y_win = 0.0f32;
        let mut vis_cnt = 0usize;
        let mut max_vis_step2 = 0.0f32;
        let mut prev_vx = 0.0f32;
        let mut prev_vy = 0.0f32;
        let mut has_prev_v = false;
        for i in 0..n_win {
            if a.phys_vis[i] != 0 {
                let x = a.phys_xs[i];
                let y = a.phys_ys[i];
                vis_idx[n_vis] = i as u8;
                if n_vis == 0 {
                    x0_arc = x;
                    y0_arc = y;
                    xf_arc = x;
                    yf_arc = y;
                } else {
                    let dx = x - x0_arc;
                    let dy = y - y0_arc;
                    let d2 = dx * dx + dy * dy;
                    if d2 > max_d2_arc {
                        max_d2_arc = d2;
                        xf_arc = x;
                        yf_arc = y;
                    }
                }
                n_vis += 1;
                if x.is_finite() && y.is_finite() {
                    if has_prev_v {
                        let sdx = x - prev_vx;
                        let sdy = y - prev_vy;
                        let sd2 = sdx * sdx + sdy * sdy;
                        if sd2 > max_vis_step2 {
                            max_vis_step2 = sd2;
                        }
                    }
                    prev_vx = x;
                    prev_vy = y;
                    has_prev_v = true;
                    if x < min_x { min_x = x; }
                    if x > max_x { max_x = x; }
                    if y < min_y { min_y = y; }
                    if y > max_y { max_y = y; }
                    sum_y_win += y;
                    vis_cnt += 1;
                }
            }
        }
        let ols_cv_dev = if !cv_now || n_win < 10 {
            true
        } else {
            let xs = &a.phys_xs[..n_win];
            let ys = &a.phys_ys[..n_win];
            let ts = &a.phys_ts[..n_win];
            if n_vis < 8 {
                true
            } else {
                let start_k = n_vis.saturating_sub(20);
                let n_fit = n_vis - start_k;
                let mut sum_t = 0.0f32;
                let mut sum_x = 0.0f32;
                let mut sum_y = 0.0f32;
                let t0 = ts[vis_idx[start_k] as usize];
                for k in start_k..n_vis {
                    let idx = vis_idx[k] as usize;
                    sum_t += (ts[idx] - t0) as f32;
                    sum_x += xs[idx];
                    sum_y += ys[idx];
                }
                let inv_n = 1.0f32 / (n_fit as f32);
                let mt = sum_t * inv_n;
                let mx = sum_x * inv_n;
                let my = sum_y * inv_n;
                let mut sxx = 0.0f32;
                let mut stx = 0.0f32;
                let mut sty = 0.0f32;
                for k in start_k..n_vis {
                    let idx = vis_idx[k] as usize;
                    let dt = (ts[idx] - t0) as f32 - mt;
                    sxx += dt * dt;
                    stx += dt * (xs[idx] - mx);
                    sty += dt * (ys[idx] - my);
                }
                if sxx < 0.02 {
                    true
                } else {
                    let inv_sxx = 1.0f32 / sxx;
                    let vx = stx * inv_sxx;
                    let vy = sty * inv_sxx;
                    let tol = 1.9 * s;
                    let tol_tail = 1.6 * s;
                    let tail_k = n_vis.saturating_sub(6);
                    let mut out_cnt = 0u32;
                    let mut dev = false;
                    for k in start_k..n_vis {
                        let idx = vis_idx[k] as usize;
                        let dt = (ts[idx] - t0) as f32 - mt;
                        let ex = (xs[idx] - (mx + vx * dt)).abs();
                        let ey = (ys[idx] - (my + vy * dt)).abs();
                        let lim = if k >= tail_k { tol_tail } else { tol };
                        if ex > lim || ey > lim {
                            out_cnt += 1;
                            if ex > 2.6 * s || ey > 2.6 * s || out_cnt >= 3 {
                                dev = true;
                                break;
                            }
                        }
                    }
                    dev
                }
            }
        };
        let stale_disc = cv_now
            && !ols_cv_dev
            && frames_since_disc < 8;
        let has_cv_dev = ols_cv_dev || (a.targets[m].fista_sharp_knots >= 2 && !stale_disc);
        let vis_reentry = a.targets[m].reemerged_last >= 2
            || (frames_since_disc >= 6
                && ((a.targets[m].reemerged_last != 0
                    && (!cv_now || has_cv_dev || a.targets[m].ode_cv_streak == 0))
                    || (vis_flicker && (!cv_now || has_cv_dev))));
        let dyn_vert = a.targets[m].vel_y.abs() >= 2.2 * s
            && (max_y - min_y) >= 10.0 * s
            && (!cv_now || a.targets[m].fista_sharp_knots >= 2 || a.targets[m].ode_cv_streak == 0);
        let vert_climb = vis_cnt >= 6
            && max_y > min_y
            && (max_y - min_y) >= 4.0 * s
            && a.targets[m].vel_y <= -0.20 * s
            && (max_y - min_y) >= 1.8 * (max_x - min_x).max(0.0);
        let eff_knots = if a.lab_mode != 0 {
            a.targets[m].fista_sharp_knots
        } else if a.targets[m].fista_sharp_knots == 0 {
            0
        } else if is_bal
            || prior_fl
            || dyn_vert
            || vert_climb
            || a.targets[m].pivot_valid > 0
        {
            a.targets[m].fista_sharp_knots
        } else if (!cv_now && age < 32 && frames_since_disc >= 4)
            || (has_cv_dev && a.targets[m].fista_sharp_knots >= 2 && frames_since_disc >= 6)
        {
            a.targets[m].fista_sharp_knots
        } else {
            let k_mask = match a.targets[m].ode_cv_streak {
                0 => 3usize,
                1..=3 => 7usize,
                _ => 15usize,
            };
            if (((f_idx as usize) + m) & k_mask) == 0 && frames_since_disc >= 6 {
                a.targets[m].fista_sharp_knots
            } else {
                0
            }
        };
        let mut run_disc = physics_id::should_run_ode_discovery(
            &a.ode_states[m],
            f_idx,
            m,
            age,
            eff_knots,
            vis_reentry,
        );
        if run_disc && a.lab_mode == 0 && !vert_climb && vis_cnt < 15 && a.ode_states[m].last_regime == 0 && a.ode_states[m].conf_ema == 0.0 {
            run_disc = false;
        } else if run_disc
            && a.lab_mode == 0
            && !vert_climb
            && age < 24
            && a.targets[m].last_ode_disc_frame != 0
            && frames_since_disc < 6
            && !is_bal
            && !dyn_vert
            && a.targets[m].reemerged_last < 2
            && a.targets[m].fista_sharp_knots < 2
        {
            run_disc = false;
        } else if run_disc
            && a.lab_mode == 0
            && !vert_climb
            && !is_bal
            && !prior_fl
            && !dyn_vert
            && !is_osc_lock
            && a.targets[m].pivot_valid == 0
            && a.targets[m].reemerged_last < 2
            && a.ode_states[m].latched_jump_code == 0
            && (a.ode_states[m].last_regime == physics_id::PHYS_REGIME_LINEAR_CV
                || a.ode_states[m].last_regime == physics_id::PHYS_REGIME_CURVED_CTRV)
        {
            if !has_cv_dev && eff_knots == 0 && !vis_reentry && a.ode_states[m].last_regime == physics_id::PHYS_REGIME_LINEAR_CV {
                if a.targets[m].coasting > 0 && age >= 20 && a.targets[m].ode_cv_streak >= 2 {
                    run_disc = false;
                } else if age >= 40
                    && a.targets[m].coasting == 0
                    && a.targets[m].ode_cv_streak >= 3
                    && a.ode_states[m].conf_ema >= 0.40
                {
                    run_disc = (((f_idx as usize) + m) & 31) == 0;
                } else if age >= 24
                    && a.targets[m].coasting == 0
                    && a.targets[m].ode_cv_streak >= 2
                    && a.ode_states[m].conf_ema >= 0.40
                {
                    run_disc = (((f_idx as usize) + m) & 15) == 0;
                } else if age >= 32 && a.ode_states[m].conf_ema >= 0.75 {
                    run_disc = (((f_idx as usize) + m) & 15) == 0;
                }
            } else if a.targets[m].coasting > 0 && age >= 20 && frames_since_disc < 16 {
                run_disc = false;
            } else if age >= 40 && eff_knots == 0 {
                run_disc = frames_since_disc >= 32
                    && (a.ode_states[m].conf_ema >= 0.25 || a.targets[m].ode_cv_streak >= 1 || frames_since_disc >= 40);
            } else if age >= 24 && eff_knots == 0 {
                run_disc = frames_since_disc >= 20;
            }
        }
        if !run_disc
            && ((a.ode_states[m].latched_jump_code == physics_id::JUMP_CHUTE_DEPLOY
                && a.ode_states[m].jump_hold_rem > 0)
                || (a.targets[m].vel_y > 1.5 * s
                    && a.targets[m].vel_y > 1.15 * a.targets[m].vel_x.abs()))
        {
            run_disc = true;
        }
        let ode_res = physics_id::step_ode_track_from_window(
            &mut a.ode_states[m],
            &a.phys_xs[..n_win],
            &a.phys_ys[..n_win],
            &a.phys_ts[..n_win],
            &vis_idx[..n_vis],
            n_vis,
            f_idx,
            s,
            run_disc,
            &mut a.ode_scratch,
        );
        if run_disc {
            a.targets[m].last_ode_disc_frame = f_idx;
            if cv_now
                && ode_res.phys_regime == physics_id::PHYS_REGIME_LINEAR_CV
                && ode_res.constraint_jump_code == 0
            {
                a.targets[m].ode_cv_streak = a.targets[m].ode_cv_streak.saturating_add(1).min(8);
            } else {
                a.targets[m].ode_cv_streak = 0;
            }
        }
        ode_jump_arr[m] = ode_res.constraint_jump_code;
        ode_r2_arr[m] = if ode_res.ode_r2.is_finite() { ode_res.ode_r2.clamp(0.0, 1.0) } else { 0.0 };
        drag_beta_arr[m] = if ode_res.aero_drag_beta_px_inv.is_finite() {
            ode_res.aero_drag_beta_px_inv.max(0.0)
        } else {
            0.0
        };
        thrust_arr[m] = if ode_res.thrust_accel_px_s2.is_finite() {
            ode_res.thrust_accel_px_s2.max(0.0)
        } else {
            0.0
        };

        let has_arc_bend = if n_win >= 7 {
            if n_vis < 7 {
                false
            } else if max_d2_arc <= (3.0 * s) * (3.0 * s) {
                true
            } else {
                let xs = &a.phys_xs[..n_win];
                let ys = &a.phys_ys[..n_win];
                let chord = wasm_sqrtf(max_d2_arc);
                let inv_c = 1.0 / chord;
                let ux = (xf_arc - x0_arc) * inv_c;
                let uy = (yf_arc - y0_arc) * inv_c;
                let thresh = 0.45 * s;
                let mut bent = false;
                for &idx_u8 in &vis_idx[..n_vis] {
                    let i = idx_u8 as usize;
                    let perp = (xs[i] - x0_arc) * uy - (ys[i] - y0_arc) * ux;
                    if perp >= thresh || perp <= -thresh {
                        bent = true;
                        break;
                    }
                }
                bent
            }
        } else {
            false
        };
        let mut tfit = if n_win >= 7 && has_arc_bend {
            physics_id::fit_taubin_circle_subpx(
                &a.phys_xs[..n_win],
                &a.phys_ys[..n_win],
                &a.phys_vis[..n_win],
                s,
                320.0,
            )
        } else {
            physics_id::TaubinFitResult::zeroed()
        };

        let sum_y = sum_y_win;
        let vis_ratio = if n_win > 0 { (vis_cnt as f32) / (n_win as f32) } else { 0.0 };
        let dx_span = if vis_cnt >= 2 { max_x - min_x } else { 0.0 };
        let dy_span = if vis_cnt >= 2 { max_y - min_y } else { 0.0 };
        span_arr[m] = wasm_sqrtf(dx_span * dx_span + dy_span * dy_span);
        let mean_y = if vis_cnt > 0 { sum_y / (vis_cnt as f32) } else { by_cur };

        let pan_active = a.cam_pan_streak >= 1 || a.cam_pan_speed_px_s >= 12.0 * s;
        let ctrv_pend_arc = ode_res.phys_regime == physics_id::PHYS_REGIME_CURVED_CTRV
            && a.targets[m].is_ballistic == 0
            && !pan_active
            && tfit.valid_circle != 0
            && tfit.radius >= 18.0 * s
            && tfit.yc < by_cur - 0.25 * tfit.radius
            && dy_span >= 8.5 * s
            && dx_span >= 0.80 * dy_span;
        let flight_or_ctrv = a.targets[m].is_ballistic != 0
            || ode_res.phys_regime == physics_id::PHYS_REGIME_BALLISTIC
            || ode_res.phys_regime == physics_id::PHYS_REGIME_BALLISTIC_DRAG
            || ode_res.phys_regime == physics_id::PHYS_REGIME_POWERED_BOOST
            || ode_res.phys_regime == physics_id::PHYS_REGIME_PARACHUTE
            || (ode_res.phys_regime == physics_id::PHYS_REGIME_CURVED_CTRV && !ctrv_pend_arc);
        let tethered_orig = tfit.valid_circle != 0
            && (a.lab_mode == 1 || (a.targets[m].birth_x - tfit.xc).abs() <= 1.35 * tfit.radius);
        let near_plumb = tfit.valid_circle != 0
            && (a.lab_mode == 1 || (min_x <= tfit.xc + 6.0 * s && max_x >= tfit.xc - 6.0 * s));
        let vert_sep = if tfit.valid_circle != 0 {
            if a.lab_mode == 1 { (mean_y - tfit.yc).abs() } else { mean_y - tfit.yc }
        } else {
            0.0
        };
        let osc_verified = a.targets[m].retrace_turnaround > 0
            || a.targets[m].freq_valid != 0
            || ode_res.phys_regime == physics_id::PHYS_REGIME_PENDULUM
            || a.targets[m].pivot_valid > 0;
        let taubin_piv_ok = physics_id::eval_pendulum_pivot_gate(
            &tfit,
            ode_res.phys_regime,
            vis_ratio,
            vert_sep,
            dx_span,
            dy_span,
            near_plumb,
            tethered_orig,
            osc_verified,
            s,
        );
        let plumb_ok = !flight_or_ctrv
            && n_win >= 10
            && vis_ratio >= 0.80
            && (a.targets[m].pivot_valid > 0
                || a.pend_states[m].has_stable_piv != 0
                || taubin_piv_ok
                || (tfit.valid_circle != 0
                    && tfit.smooth_arc != 0
                    && vert_sep >= (14.0 * s).max(0.42 * tfit.radius)
                    && dx_span >= 0.90 * dy_span
                    && tethered_orig));
        if (has_arc_bend && !(cv_now && !ols_cv_dev && tfit.valid_circle == 0))
            || a.pend_states[m].has_stable_piv != 0
            || a.pend_states[m].jump_hold > 0
            || a.pend_states[m].jump_cooldown > 0
            || a.pend_states[m].jump_cand_streak > 0
        {
            let (peg_fit, _peg_fired) = physics_id::update_galileo_peg_detector(
                &mut a.pend_states[m],
                &a.phys_xs[..n_win],
                &a.phys_ys[..n_win],
                &a.phys_ts[..n_win],
                &a.phys_vis[..n_win],
                n_win,
                f_idx,
                s,
                plumb_ok,
                bx_cur,
                by_cur,
            );
            if tfit.valid_circle == 0 && peg_fit.valid_circle != 0 {
                tfit = peg_fit;
            }
        }
        taubin_fits[m] = tfit;
        peg_jump_arr[m] = a.pend_states[m].jump_code == physics_id::JUMP_GALILEO_PEG;

        let stable_piv_ok = a.pend_states[m].has_stable_piv != 0
            && !flight_or_ctrv
            && a.pend_states[m].stable_piv_r > 1.0
            && (by_cur > a.pend_states[m].stable_piv_y + 8.0 * s || a.lab_mode == 1);
        let mode0_pend_geom_ok = a.lab_mode != 0
            || (!pan_active
                && n_win >= 12
                && ((max_vis_step2 <= (9.0 * s) * (9.0 * s)
                    && (a.pend_states[m].has_stable_piv != 0
                        || a.targets[m].retrace_turnaround > 0
                        || a.targets[m].freq_valid != 0
                        || a.targets[m].cand_piv_streak >= 3))
                    || (max_vis_step2 <= (24.0 * s) * (24.0 * s)
                        && ((a.targets[m].retrace_dirs == 3
                            && a.targets[m].retrace_turnaround > 0
                            && a.targets[m].age >= 20
                            && dy_span >= 10.0 * s)
                            || (a.targets[m].freq_valid != 0
                                && a.targets[m].tremor_coherence >= 0.36
                                && a.targets[m].tremor_freq_hz >= 0.20
                                && a.targets[m].tremor_freq_hz <= 1.25))))
                && dy_span >= 6.0 * s
                && dx_span >= 12.0 * s
                && a.targets[m].max_excursion2 >= (18.0 * s) * (18.0 * s));
        let target_piv_ok = a.targets[m].pivot_valid > 0
            && (by_cur > a.targets[m].pivot_y + 8.0 * s || a.lab_mode == 1);
        let has_pend_piv = !flight_or_ctrv
            && mode0_pend_geom_ok
            && (stable_piv_ok || taubin_piv_ok || target_piv_ok);
        if a.lab_mode == 0 && !mode0_pend_geom_ok {
            a.pend_states[m].has_stable_piv = 0;
            a.targets[m].pivot_valid = 0;
        }
        piv_arr[m] = if has_pend_piv { 1 } else { 0 };
        let (piv_xh, piv_yh, piv_r) = if stable_piv_ok {
            (
                a.pend_states[m].stable_piv_x,
                a.pend_states[m].stable_piv_y,
                a.pend_states[m].stable_piv_r,
            )
        } else if taubin_piv_ok {
            (tfit.xc, tfit.yc, tfit.radius)
        } else {
            (
                a.targets[m].pivot_x - pan_dh_x,
                a.targets[m].pivot_y - pan_dh_y,
                a.targets[m].pivot_r,
            )
        };
        if a.lab_mode == 0 && has_pend_piv && (stable_piv_ok || taubin_piv_ok) {
            a.targets[m].pivot_x = piv_xh + pan_dh_x;
            a.targets[m].pivot_y = piv_yh + pan_dh_y;
            a.targets[m].pivot_r = piv_r;
            if a.targets[m].pivot_valid == 0 {
                a.targets[m].pivot_valid = 20;
            }
        }

        let mut phys_regime = ode_res.phys_regime;
        let mut phys_conf = if ode_res.phys_regime_conf.is_finite() {
            ode_res.phys_regime_conf.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if (has_pend_piv
            || (a.lab_mode == 1
                && !flight_or_ctrv
                && a.targets[m].osc_regime == 2
                && a.targets[m].freq_valid != 0
                && a.targets[m].tremor_coherence >= 0.45))
            && phys_regime == physics_id::PHYS_REGIME_LINEAR_CV
        {
            phys_regime = physics_id::PHYS_REGIME_PENDULUM;
            phys_conf = phys_conf.max(0.75);
        }
        if a.lab_mode == 0 {
            if phys_regime == physics_id::PHYS_REGIME_PENDULUM && (pan_active || !has_pend_piv) {
                phys_regime = physics_id::PHYS_REGIME_LINEAR_CV;
                phys_conf = 0.0;
            }
            if phys_regime == physics_id::PHYS_REGIME_POWERED_BOOST
                && (a.targets[m].vel_y >= -0.22 * s
                    || a.targets[m].vel_y.abs() < 0.45 * a.targets[m].vel_x.abs()
                    || (a.lab_mode == 0
                        && (disp_vy_arr[m] > -8.0 * s
                            || disp_vy_arr[m].abs() < 0.95 * disp_vx_arr[m].abs()))
                    || (a.targets[m].vel_x * a.targets[m].vel_x
                        + a.targets[m].vel_y * a.targets[m].vel_y)
                        < (0.24 * s) * (0.24 * s)
                    || a.targets[m].max_excursion2 < (8.0 * s) * (8.0 * s))
            {
                phys_regime = physics_id::PHYS_REGIME_LINEAR_CV;
                phys_conf = 0.0;
            }
        }
        if a.lab_mode == 0
            && phys_regime == physics_id::PHYS_REGIME_PARACHUTE
            && (pan_active
                || disp_vy_arr[m] < 8.0 * s
                || disp_vy_arr[m] < 0.45 * disp_vx_arr[m].abs())
        {
            phys_regime = physics_id::PHYS_REGIME_LINEAR_CV;
            phys_conf = 0.0;
        }
        phys_reg_arr[m] = phys_regime;
        phys_conf_arr[m] = phys_conf;

        let mut g_pend = 0.0f32;
        if has_pend_piv {
            if let Some(g_arc) = physics_id::estimate_pendulum_gravity_from_arc(
                &a.phys_xs[..n_win],
                &a.phys_ys[..n_win],
                &a.phys_ts[..n_win],
                &a.phys_vis[..n_win],
                n_win,
                s,
            ) {
                a.pend_states[m].g_eff_ema = if a.pend_states[m].g_eff_ema <= 1.0 {
                    g_arc
                } else {
                    0.80 * a.pend_states[m].g_eff_ema + 0.20 * g_arc
                };
            }
            g_pend = if a.pend_states[m].g_eff_ema > 1.0 {
                a.pend_states[m].g_eff_ema
            } else if a.scene_gravity_conf >= 0.45 && a.scene_gravity_px_s2 > 1.0 {
                a.scene_gravity_px_s2
            } else if ode_res.est_g_px_s2 > 1.0 {
                ode_res.est_g_px_s2
            } else {
                0.0
            };
        }

        let osc_active = !flight_or_ctrv
            && phys_regime != physics_id::PHYS_REGIME_IMPULSIVE
            && (has_pend_piv
                || phys_regime == physics_id::PHYS_REGIME_PENDULUM
                || phys_regime == physics_id::PHYS_REGIME_MICRO_TREMOR
                || (a.targets[m].freq_valid != 0 && a.targets[m].osc_regime != 0));
        let t_obs_s = if a.targets[m].freq_valid != 0 && a.targets[m].tremor_freq_hz > 0.05 {
            1.0 / a.targets[m].tremor_freq_hz
        } else if ode_res.est_omega_rad_s > 0.3 {
            TWO_PI / ode_res.est_omega_rad_s
        } else {
            0.0
        };
        let elliptic_period_s = if !osc_active {
            0.0
        } else if has_pend_piv && piv_r > 1.0 {
            physics_id::compute_elliptic_period_s(
                piv_xh,
                piv_yh,
                piv_r,
                &a.phys_xs[..n_win],
                &a.phys_ys[..n_win],
                n_win,
                &mut a.pend_states[m].seg_peak_cos,
                g_pend,
                t_obs_s,
            )
        } else if t_obs_s.is_finite() && t_obs_s > 0.05 {
            t_obs_s.clamp(0.05, 30.0)
        } else {
            0.0
        };
        period_arr[m] = if elliptic_period_s.is_finite() {
            elliptic_period_s.max(0.0)
        } else {
            0.0
        };

        let omega_d = if elliptic_period_s > 0.05 {
            TWO_PI / elliptic_period_s
        } else if a.targets[m].tremor_omega_ema > 0.3 {
            a.targets[m].tremor_omega_ema
        } else if ode_res.est_omega_rad_s > 0.3 {
            ode_res.est_omega_rad_s
        } else {
            0.0
        };

        let dfx = a.targets[m].disp_fista_x;
        let dfy = a.targets[m].disp_fista_y;
        let piv_xd = piv_xh + pan_dh_x;
        let piv_yd = piv_yh + pan_dh_y;
        let (q_damp, min_amp_damp, t_damp) = if has_pend_piv && piv_r > 1.0 {
            (bx_cur - piv_xh, (3.5 * s).max(0.03 * piv_r), t_head_sec)
        } else if a.targets[m].osc_regime == 2 {
            (dfx - a.targets[m].osc_mean_x, 3.0 * s, t_disp_sec)
        } else {
            (
                a.targets[m].tremor_r,
                (0.18 * s).max(0.20 * a.targets[m].tremor_amp),
                t_disp_sec,
            )
        };
        let had_prev_piv = a.telemetry[1120 + m * 8 + 6] > -9e5;
        if osc_active
            && has_pend_piv
            && piv_r > 1.0
            && !had_prev_piv
            && a.pend_states[m].peak_count == 0
            && !peg_jump_arr[m]
            && n_win >= 12
        {
            for k in 0..(n_win - 1) {
                if a.phys_vis[k] != 0 {
                    let _ = physics_id::update_damping_zeta_q(
                        &mut a.pend_states[m],
                        a.phys_xs[k] - piv_xh,
                        a.phys_ts[k] as f32,
                        omega_d,
                        min_amp_damp,
                        true,
                    );
                }
            }
        }
        let (zeta, q_val) = physics_id::update_damping_zeta_q(
            &mut a.pend_states[m],
            q_damp,
            t_damp,
            omega_d,
            min_amp_damp,
            osc_active,
        );
        zeta_arr[m] = if zeta.is_finite() { zeta.clamp(0.0, 1.0) } else { 0.0 };
        q_arr[m] = if q_val.is_finite() { q_val.clamp(0.0, 999.0) } else { 0.0 };

        a.imm[m].beta_drag = drag_beta_arr[m].clamp(0.0, 0.25);
        if g_pend > 1.0 {
            a.imm[m].g_bal_s2 = 0.7 * a.imm[m].g_bal_s2 + 0.3 * g_pend.clamp(30.0 * s, 25000.0 * s);
        } else if ode_res.est_g_px_s2 > 1.0 {
            a.imm[m].g_bal_s2 = 0.7 * a.imm[m].g_bal_s2 + 0.3 * ode_res.est_g_px_s2.clamp(30.0 * s, 25000.0 * s);
        }
        if osc_active && omega_d > 0.3 {
            a.imm[m].omega_osc = 0.75 * a.imm[m].omega_osc + 0.25 * omega_d.clamp(0.5, 95.0);
        }

        let g_energy = if has_pend_piv && g_pend > 1.0 {
            g_pend
        } else if a.scene_gravity_conf >= 0.45 && a.scene_gravity_px_s2 > 1.0 {
            a.scene_gravity_px_s2
        } else if ode_res.est_g_px_s2 > 1.0 {
            ode_res.est_g_px_s2
        } else {
            0.0
        };
        let (ref_x, ref_y) = if phys_regime == physics_id::PHYS_REGIME_CURVED_CTRV && tfit.valid_circle != 0 {
            (tfit.xc + pan_dh_x, tfit.yc + pan_dh_y)
        } else if a.targets[m].osc_regime != 0 {
            (a.targets[m].osc_mean_x, a.targets[m].osc_mean_y)
        } else {
            (frame_w as f32 * 0.5, frame_h as f32 * 0.5)
        };
        let q_energy = if has_pend_piv {
            dfx - piv_xd
        } else if a.targets[m].osc_regime == 2 {
            dfx - a.targets[m].osc_mean_x
        } else {
            a.targets[m].tremor_r
        };
        let (spec_e, spec_l) = physics_id::compute_specific_energy_and_ang_mom(
            dfx,
            dfy,
            disp_vx_arr[m],
            disp_vy_arr[m],
            g_energy,
            frame_h as f32,
            has_pend_piv,
            piv_xd,
            piv_yd,
            piv_r,
            ref_x,
            ref_y,
            osc_active,
            omega_d,
            q_energy,
            a.targets[m].is_ballistic != 0,
        );
        energy_arr[m] = spec_e;
        ang_mom_arr[m] = spec_l;

        osc_samples[m] = physics_id::OscSample {
            active: a.targets[m].active,
            confirmed: a.targets[m].confirmed,
            coasting: if a.targets[m].coasting > 0 { 1 } else { 0 },
            phase_valid: a.targets[m].phase_valid,
            freq_valid: a.targets[m].freq_valid,
            is_ballistic: a.targets[m].is_ballistic,
            strike_fired: a.targets[m].strike_fired,
            phys_regime,
            phase_rad: a.targets[m].tremor_phase,
            freq_hz: a.targets[m].tremor_freq_hz,
            coherence: a.targets[m].tremor_coherence,
            amp_px: a.targets[m].tremor_amp,
        };
    }

    // 2. Two-Body Causal Collision & Conservation Analysis
    let col_arena = physics_id::CollisionArenaView {
        frame_idx: f_idx,
        continuity_epoch: a.wave1_telem[6],
        lookahead_k: eff_k as u8,
        scene_gravity_conf: a.scene_gravity_conf,
        scene_gravity_px_s2: a.scene_gravity_px_s2,
        frame_w: frame_w as f32,
        frame_h: frame_h as f32,
        scale_s: s,
        hist_t_sec: &a.hist_t_sec,
        hist_cam_x: &a.hist_cam_x,
        hist_cam_y: &a.hist_cam_y,
    };
    let dummy_track = physics_id::CollisionTrackView {
        active: 0,
        confirmed: 0,
        is_ballistic: 0,
        pivot_valid: 0,
        coasting: 0,
        track_id: 0,
        birth_frame: 0,
        orig_birth_frame: 0,
        last_discont_frame: 0,
        disp_world_x: 0.0,
        disp_world_y: 0.0,
        disp_vx: 0.0,
        disp_vy: 0.0,
        arc_raw_x: &a.targets[0].arc_raw_x,
        arc_raw_y: &a.targets[0].arc_raw_y,
        arc_vis: &a.targets[0].arc_vis,
    };
    let col_out = if a.lab_mode == 2 {
        let mut col_tracks = [dummy_track; MAX_TARGETS];
        for m in 0..MAX_TARGETS {
            let col_act = if a.lab_mode == 0 && a.targets[m].coasting >= 2 {
                0
            } else {
                a.targets[m].active
            };
            col_tracks[m] = physics_id::CollisionTrackView {
                active: col_act,
                confirmed: a.targets[m].confirmed,
                is_ballistic: a.targets[m].is_ballistic,
                pivot_valid: piv_arr[m],
                coasting: a.targets[m].coasting.min(255) as u8,
                track_id: a.targets[m].track_id,
                birth_frame: a.targets[m].birth_frame,
                orig_birth_frame: a.targets[m].orig_birth_frame,
                last_discont_frame: a.last_discont_frame,
                disp_world_x: a.targets[m].disp_fista_x,
                disp_world_y: a.targets[m].disp_fista_y,
                disp_vx: disp_vx_arr[m],
                disp_vy: disp_vy_arr[m],
                arc_raw_x: &a.targets[m].arc_raw_x,
                arc_raw_y: &a.targets[m].arc_raw_y,
                arc_vis: &a.targets[m].arc_vis,
            };
        }
        physics_id::analyze_two_body_collisions(&col_arena, &col_tracks, &mut a.collision_state)
    } else {
        physics_id::CollisionTickOutput::zeroed()
    };

    for m in 0..MAX_TARGETS {
        if a.targets[m].active == 0 || a.targets[m].confirmed == 0 {
            continue;
        }
        if col_out.regime_override[m] == physics_id::PHYS_REGIME_IMPULSIVE {
            phys_reg_arr[m] = physics_id::PHYS_REGIME_IMPULSIVE;
            phys_conf_arr[m] = phys_conf_arr[m].max(0.85);
            osc_samples[m].phys_regime = physics_id::PHYS_REGIME_IMPULSIVE;
        }
    }

    // 3. Ensemble Kuramoto Order, Ensemble Taubin Selection & Staggered 1:8 Lyapunov
    let (kuramoto_r, kuramoto_psi, kuramoto_cnt) =
        physics_id::compute_kuramoto_order(&osc_samples, MAX_TARGETS, s);
    let (taubin_delta_px, taubin_curv) = physics_id::select_ensemble_taubin_metrics(
        &taubin_fits,
        &active_arr,
        &conf_arr,
        &bal_arr,
        &piv_arr,
        &phys_reg_arr,
        &span_arr,
        s,
    );

    let mut max_lyap = 0.0f32;
    for m in 0..MAX_TARGETS {
        let t = &a.targets[m];
        let interior = t.active != 0
            && t.confirmed != 0
            && t.coasting == 0
            && t.is_ballistic == 0
            && t.disp_fista_x > half_patch + 6.0 * s
            && t.disp_fista_x < (frame_w as f32) - half_patch - 6.0 * s
            && t.disp_fista_y > half_patch + 6.0 * s
            && t.disp_fista_y < (frame_h as f32) - half_patch - 6.0 * s
            && a.cam_pan_streak == 0
            && a.cam_pan_speed_px_s < 12.0 * s
            && (phys_reg_arr[m] == physics_id::PHYS_REGIME_PENDULUM
                || phys_reg_arr[m] == physics_id::PHYS_REGIME_MICRO_TREMOR
                || phys_reg_arr[m] == physics_id::PHYS_REGIME_CURVED_CTRV);
        if !interior {
            a.lyap_cache[m] = 0.0;
        } else if (((f_idx as usize) + m) & 7) == 0 || a.pend_states[m].res_count == 32 {
            a.lyap_cache[m] = physics_id::compute_lyapunov_div_rate_from_reservoir(
                &a.pend_states[m],
                &mut a.lyap_scratch,
                phys_reg_arr[m],
                t.is_ballistic != 0,
                t.coasting != 0,
                t.confirmed != 0,
                true,
                s,
                fps,
            );
        }
        if a.lyap_cache[m] > max_lyap {
            max_lyap = a.lyap_cache[m];
        }
    }

    // 4. Write Locked Telemetry Schema (1280..1535 per-target + 1536..1549 ensemble + 1550..1567 == 0.0)
    for m in 0..MAX_TARGETS {
        let b = TELEM_PHYSICS_START + m * 16;
        let bit = 1u16 << m;
        if a.targets[m].active == 0 || a.targets[m].confirmed == 0 || a.telemetry[32 + m * 56] < 0.5 {
            a.targets[m].phys_stamp_tid = 0;
            a.ingest_ctrl[CTRL_HERO_PHYS_SLOT_MASK] &= !(1u32 << m);
            if (a.physics_telem_dirty_mask & bit) != 0 {
                for q in 0..16 {
                    a.telemetry[b + q] = 0.0;
                }
                a.physics_telem_dirty_mask &= !bit;
            }
            continue;
        }
        a.physics_telem_dirty_mask |= bit;
        a.targets[m].phys_stamp_tid = a.targets[m].track_id;
        a.ingest_ctrl[CTRL_HERO_PHYS_SLOT_MASK] |= 1u32 << m;
        let mut w0 = if a.imm[m].mu[0].is_finite() { a.imm[m].mu[0].max(0.0) } else { 0.25 };
        let mut w1 = if a.imm[m].mu[1].is_finite() { a.imm[m].mu[1].max(0.0) } else { 0.25 };
        let mut w23 = if a.imm[m].mu[2].is_finite() && a.imm[m].mu[3].is_finite() {
            (a.imm[m].mu[2] + a.imm[m].mu[3]).max(0.0)
        } else {
            0.50
        };
        let wsum = w0 + w1 + w23;
        if wsum > 1e-6 {
            let inv = 1.0 / wsum;
            w0 *= inv;
            w1 *= inv;
            w23 = (1.0 - w0 - w1).clamp(0.0, 1.0);
        } else {
            w0 = 0.25;
            w1 = 0.25;
            w23 = 0.50;
        }

        let col_e = col_out.restitution_e[m];
        let rest_e = if col_e > 0.0 && col_e.is_finite() {
            col_e.clamp(0.0, 1.0)
        } else if a.targets[m].has_struck != 0 && a.targets[m].strike_restitution.is_finite() {
            a.targets[m].strike_restitution.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Strict arbitration priority: 4 (COLLISION) > 1 (GALILEO_PEG) > 3 (CHUTE_DEPLOY) > 2 (BOOST_BURNOUT)
        let jump_code = if col_out.jump_code[m] == physics_id::JUMP_COLLISION {
            physics_id::JUMP_COLLISION
        } else if peg_jump_arr[m] {
            physics_id::JUMP_GALILEO_PEG
        } else if ode_jump_arr[m] == physics_id::JUMP_CHUTE_DEPLOY {
            physics_id::JUMP_CHUTE_DEPLOY
        } else if ode_jump_arr[m] == physics_id::JUMP_BOOST_BURNOUT {
            physics_id::JUMP_BOOST_BURNOUT
        } else {
            physics_id::JUMP_NONE
        };

        a.telemetry[b] = w0;
        a.telemetry[b + 1] = w1;
        a.telemetry[b + 2] = w23;
        a.telemetry[b + 3] = (a.targets[m].fista_sharp_knots.min(10)) as f32;
        a.telemetry[b + 4] = phys_reg_arr[m] as f32;
        a.telemetry[b + 5] = phys_conf_arr[m];
        a.telemetry[b + 6] = ode_r2_arr[m];
        a.telemetry[b + 7] = drag_beta_arr[m];
        a.telemetry[b + 8] = thrust_arr[m];
        a.telemetry[b + 9] = period_arr[m];
        a.telemetry[b + 10] = zeta_arr[m];
        a.telemetry[b + 11] = q_arr[m];
        a.telemetry[b + 12] = energy_arr[m];
        a.telemetry[b + 13] = ang_mom_arr[m];
        a.telemetry[b + 14] = rest_e;
        a.telemetry[b + 15] = jump_code as f32;
    }

    if a.lab_mode == 1 && eff_k >= 7 && a.ring_cursor >= (eff_k as u64) {
        let mut ode_freq_arr = [0.0f32; MAX_TARGETS];
        let mut best_pend_m: i32 = -1;
        let mut best_pend_score = -1.0f32;
        for m in 0..MAX_TARGETS {
            if conf_arr[m] != 0
                && a.telemetry[32 + m * 56] > 0.5
                && phys_reg_arr[m] == physics_id::PHYS_REGIME_PENDULUM
                && period_arr[m] >= 0.30
                && period_arr[m] <= 6.0
                && ode_r2_arr[m] >= 0.50
            {
                let f_ode = (1.0 / period_arr[m]).clamp(0.15, 15.0);
                ode_freq_arr[m] = f_ode;
                if a.slot_meta[m * 8 + 4] <= 0.0 {
                    a.slot_meta[m * 8 + 4] = f_ode;
                }
                let sc_m = span_arr[m] * ode_r2_arr[m];
                if sc_m > best_pend_score {
                    best_pend_score = sc_m;
                    best_pend_m = m as i32;
                }
            }
        }
        let cur_pm = a.telemetry[8] as i32;
        if cur_pm >= 0 && (cur_pm as usize) < MAX_TARGETS {
            let pm = cur_pm as usize;
            if a.targets[pm].freq_valid != 0 && a.targets[pm].osc_regime == 2 {
                a.telemetry[13] = a.targets[pm].tremor_freq_hz;
                a.telemetry[26] = a.targets[pm].phase_valid as f32;
                a.telemetry[TELEM_GLOBAL_START + 12] = 1.0;
            } else if ode_freq_arr[pm] > 0.0 {
                a.telemetry[13] = ode_freq_arr[pm];
                a.telemetry[26] = 1.0;
                a.telemetry[TELEM_GLOBAL_START + 12] = 1.0;
            } else if a.focus_target_id == 0 && best_pend_m >= 0 {
                let bm = best_pend_m as usize;
                a.telemetry[8] = best_pend_m as f32;
                a.telemetry[13] = ode_freq_arr[bm];
                a.telemetry[26] = 1.0;
                a.telemetry[TELEM_GLOBAL_START + 12] = 1.0;
            }
        }
    }

    a.telemetry[TELEM_ENSEMBLE_START] = kuramoto_r;
    a.telemetry[TELEM_ENSEMBLE_START + 1] = kuramoto_psi;
    a.telemetry[TELEM_ENSEMBLE_START + 2] = kuramoto_cnt;
    a.telemetry[TELEM_ENSEMBLE_START + 3] = col_out.ensemble_kinetic_px2_s2;
    a.telemetry[TELEM_ENSEMBLE_START + 4] = col_out.ensemble_mech_energy_px2_s2;
    a.telemetry[TELEM_ENSEMBLE_START + 5] = col_out.last_pair_mass_ratio;
    a.telemetry[TELEM_ENSEMBLE_START + 6] = col_out.last_pair_mom_resid;
    a.telemetry[TELEM_ENSEMBLE_START + 7] = col_out.last_pair_energy_retention;
    a.telemetry[TELEM_ENSEMBLE_START + 8] = col_out.last_pair_tid_a;
    a.telemetry[TELEM_ENSEMBLE_START + 9] = col_out.last_pair_tid_b;
    a.telemetry[TELEM_ENSEMBLE_START + 10] = col_out.last_pair_frame;
    a.telemetry[TELEM_ENSEMBLE_START + 11] = max_lyap;
    a.telemetry[TELEM_ENSEMBLE_START + 12] = taubin_delta_px;
    a.telemetry[TELEM_ENSEMBLE_START + 13] = taubin_curv;
    for ti in (TELEM_ENSEMBLE_START + 14)..TELEMETRY_LEN {
        a.telemetry[ti] = 0.0;
    }

    {
        use crate::simd::{f32x4_abs, f32x4_lt, f32x4_splat, v128, v128_and};
        const _: () = assert!(TELEMETRY_LEN % 4 == 0);
        let vinf = f32x4_splat(f32::INFINITY);
        let p = a.telemetry.as_mut_ptr() as *mut v128;
        let n4 = TELEMETRY_LEN / 4;
        // SAFETY: `a.telemetry` has `TELEMETRY_LEN` (`1568`, divisible by 4) `f32` lanes.
        unsafe {
            let mut i = 0usize;
            while i < n4 {
                let vp = p.add(i);
                let v = core::ptr::read_unaligned(vp);
                let fin = f32x4_lt(f32x4_abs(v), vinf);
                core::ptr::write_unaligned(vp, v128_and(v, fin));
                i += 1;
            }
        }
    }
    true
}

/// Exact sparse-compacted wrapper around `solve_jv_assign_8x32`.
///
/// Iteratively peels rows whose best private (`col_rows.count_ones() == 1`) column beats
/// or ties all shared columns, compacts remaining contested rows & shared columns strictly
/// monotonically in-place into `cost`, and solves only the reduced subproblem.
/// Stores each row's winning assignment cost in `out_cost[r]` (even when `cost` is mutated).
#[inline(never)]
fn solve_jv_assign_sparse_mut(
    nr: usize,
    n_hyps: usize,
    nc: usize,
    fin_hyps: &[u64; MAX_TARGETS],
    cost: &mut [[f32; ASSIGN_COLS]; MAX_TARGETS],
    out_cost: &mut [f32; MAX_TARGETS],
) -> [usize; MAX_TARGETS] {
    let mut out_assign = [usize::MAX; MAX_TARGETS];
    for i in 0..MAX_TARGETS {
        out_cost[i] = 1e9f32;
    }
    if nr == 0 {
        return out_assign;
    }
    if nr > MAX_TARGETS || nc > ASSIGN_COLS || nc == 0 || n_hyps > MAX_HYPS {
        let assign = solve_jv_assign_8x32(nr, nc, cost);
        for r in 0..nr.min(MAX_TARGETS) {
            let ac = assign[r];
            out_cost[r] = if ac < nc { cost[r][ac] } else { 1e9f32 };
        }
        return assign;
    }

    let mut col_rows = [0u16; ASSIGN_COLS];
    let mut row_cols = [0u64; MAX_TARGETS];
    let mut active_cols = 0u64;
    for r in 0..nr {
        let rm = fin_hyps[r] | (1u64 << (n_hyps + r));
        row_cols[r] = rm;
        active_cols |= rm;
        let mut m = rm;
        while m != 0 {
            let c = m.trailing_zeros() as usize;
            m &= m - 1;
            col_rows[c] |= 1u16 << r;
        }
    }

    let mut priv_c = [usize::MAX; MAX_TARGETS];
    let mut priv_v = [1e9f32; MAX_TARGETS];
    let mut shared_mask = 0u64;
    let mut ac = active_cols;
    while ac != 0 {
        let c = ac.trailing_zeros() as usize;
        ac &= ac - 1;
        let m = col_rows[c];
        let cnt = m.count_ones();
        if cnt >= 2 {
            shared_mask |= 1u64 << c;
        } else if cnt == 1 {
            let r = m.trailing_zeros() as usize;
            let v = cost[r][c];
            if v < priv_v[r] {
                priv_v[r] = v;
                priv_c[r] = c;
            }
        }
    }

    let mut alive_rows = ((1u32 << nr) - 1) as u16;
    loop {
        let mut peeled_any = false;
        let mut rem_r = alive_rows;
        while rem_r != 0 {
            let r = rem_r.trailing_zeros() as usize;
            rem_r &= rem_r - 1;
            let sm = row_cols[r] & shared_mask;
            let mut min_s = 1e9f32;
            let mut rem_s = sm;
            while rem_s != 0 {
                let c = rem_s.trailing_zeros() as usize;
                rem_s &= rem_s - 1;
                let v = cost[r][c];
                if v < min_s {
                    min_s = v;
                }
            }
            if sm == 0 || (priv_c[r] < nc && priv_v[r] <= min_s) {
                out_assign[r] = priv_c[r];
                out_cost[r] = priv_v[r];
                alive_rows &= !(1u16 << r);
                peeled_any = true;
                let mut upd = sm;
                while upd != 0 {
                    let c = upd.trailing_zeros() as usize;
                    upd &= upd - 1;
                    col_rows[c] &= !(1u16 << r);
                    let cr = col_rows[c];
                    if cr.count_ones() == 1 {
                        shared_mask &= !(1u64 << c);
                        let r2 = cr.trailing_zeros() as usize;
                        let v2 = cost[r2][c];
                        if v2 < priv_v[r2] {
                            priv_v[r2] = v2;
                            priv_c[r2] = c;
                        }
                    }
                }
            }
        }
        if !peeled_any || alive_rows == 0 {
            break;
        }
    }

    if alive_rows == 0 {
        return out_assign;
    }

    let mut sub_rows = [0usize; MAX_TARGETS];
    let mut sub_nr = 0usize;
    let mut rem_r = alive_rows;
    while rem_r != 0 {
        let r = rem_r.trailing_zeros() as usize;
        rem_r &= rem_r - 1;
        if priv_c[r] >= nc {
            let assign = solve_jv_assign_8x32(nr, nc, cost);
            for rr in 0..nr {
                let ac = assign[rr];
                out_cost[rr] = if ac < nc { cost[rr][ac] } else { 1e9f32 };
            }
            return assign;
        }
        sub_rows[sub_nr] = r;
        sub_nr += 1;
    }
    if sub_nr == 1 {
        let r = sub_rows[0];
        out_assign[r] = priv_c[r];
        out_cost[r] = priv_v[r];
        return out_assign;
    }

    let mut shared_cols = [0usize; ASSIGN_COLS];
    let mut sub_ns = 0usize;
    let mut rem_sc = shared_mask;
    while rem_sc != 0 {
        let c = rem_sc.trailing_zeros() as usize;
        rem_sc &= rem_sc - 1;
        shared_cols[sub_ns] = c;
        sub_ns += 1;
    }

    if sub_ns + sub_nr > ASSIGN_COLS {
        let assign = solve_jv_assign_8x32(nr, nc, cost);
        for rr in 0..nr {
            let ac = assign[rr];
            out_cost[rr] = if ac < nc { cost[rr][ac] } else { 1e9f32 };
        }
        return assign;
    }

    let sub_nc = sub_ns + sub_nr;
    for sr in 0..sub_nr {
        let r = sub_rows[sr];
        for sc in 0..sub_ns {
            cost[sr][sc] = cost[r][shared_cols[sc]];
        }
        let pv = priv_v[r];
        for k in 0..sub_nr {
            cost[sr][sub_ns + k] = if k == sr { pv } else { 1e9f32 };
        }
    }

    let sub_assign = solve_jv_assign_8x32(sub_nr, sub_nc, cost);
    for sr in 0..sub_nr {
        let r = sub_rows[sr];
        let sc = sub_assign[sr];
        if sc < sub_ns {
            out_assign[r] = shared_cols[sc];
            out_cost[r] = cost[sr][sc];
        } else {
            out_assign[r] = priv_c[r];
            out_cost[r] = priv_v[r];
        }
    }
    out_assign
}

#[inline(never)]
unsafe fn update_slot_pip_and_meta(
    a: &mut Arena,
    m: usize,
    mbase: usize,
    slot_disp: usize,
    frame_w: usize,
    frame_h: usize,
    pip_dim: usize,
    pip_center_off: f32,
    max_pip_x: f32,
    max_pip_y: f32,
    gain: f32,
) {
    let tid = a.targets[m].track_id;
    let fx = a.targets[m].disp_fista_x;
    let fy = a.targets[m].disp_fista_y;
    let reset_slot_ref = a.all_pip_tid[m] != tid;
    a.all_pip_tid[m] = tid;

    let pip_x0 = (fx - pip_center_off).clamp(0.0, max_pip_x);
    let pip_y0 = (fy - pip_center_off).clamp(0.0, max_pip_y);
    if !render_eulerian_pip_fast_simd128(
        &a.rgba_ring[slot_disp],
        frame_w,
        frame_h,
        pip_x0,
        pip_y0,
        pip_dim,
        gain,
        reset_slot_ref,
        &mut a.all_pip_ref_luma[m],
        &mut a.all_pip_rgba[m],
    ) {
        update_slot_pip_scalar_fallback(
            a,
            m,
            slot_disp,
            frame_w,
            frame_h,
            pip_dim,
            pip_x0,
            pip_y0,
            gain,
            reset_slot_ref,
        );
    }

    a.slot_meta[mbase] = a.targets[m].age as f32;
    a.slot_meta[mbase + 1] = a.targets[m].coasting as f32;
    a.slot_meta[mbase + 2] = a.targets[m].vis_eca_ema;
    a.slot_meta[mbase + 3] = a.targets[m].reacq_count as f32;
    a.slot_meta[mbase + 4] = if a.targets[m].freq_valid != 0 {
        a.targets[m].tremor_freq_hz
    } else {
        0.0
    };
    a.slot_meta[mbase + 5] = a.targets[m].tremor_amp;
    a.slot_meta[mbase + 6] = a.targets[m].fista_duality_gap;
    a.slot_meta[mbase + 7] = wasm_sqrtf(a.targets[m].max_excursion2);
}

#[cold]
#[inline(never)]
unsafe fn update_slot_pip_scalar_fallback(
    a: &mut Arena,
    m: usize,
    slot_disp: usize,
    frame_w: usize,
    frame_h: usize,
    pip_dim: usize,
    pip_x0: f32,
    pip_y0: f32,
    gain: f32,
    reset_slot_ref: bool,
) {
    for py in 0..pip_dim {
        for px in 0..pip_dim {
            let sx = pip_x0 + (px as f32);
            let sy = pip_y0 + (py as f32);
            let (r, g, b, luma) =
                sample_rgba_bilinear(&a.rgba_ring[slot_disp], frame_w, frame_h, sx, sy);
            let idx = py * pip_dim + px;

            if reset_slot_ref {
                a.all_pip_ref_luma[m][idx] = luma;
            }

            let dl = luma - a.all_pip_ref_luma[m][idx];
            a.all_pip_ref_luma[m][idx] = 0.80 * a.all_pip_ref_luma[m][idx] + 0.20 * luma;

            let amp_l = gain * dl;
            let shear_pos = amp_l.max(0.0);
            let shear_neg = (-amp_l).max(0.0);
            let out_r = (r + amp_l * 0.70 + shear_neg * 0.65).clamp(0.0, 1.0);
            let out_g = (g + amp_l * 0.80 + shear_pos * 0.45).clamp(0.0, 1.0);
            let out_b = (b + amp_l * 0.70 + shear_pos * 0.85).clamp(0.0, 1.0);

            let rgba_off = idx * 4;
            a.all_pip_rgba[m][rgba_off] = (out_r * 255.0) as u8;
            a.all_pip_rgba[m][rgba_off + 1] = (out_g * 255.0) as u8;
            a.all_pip_rgba[m][rgba_off + 2] = (out_b * 255.0) as u8;
            a.all_pip_rgba[m][rgba_off + 3] = 255;
        }
    }
}

/// Phase 2 of frame tick (executes after Fiedler vector & eigengap are written to Arena):
/// Performs spectrally-regularized 2-adic patch association, continuous SSD sub-pixel refinement,
/// 2x2 Kalman baseline tracking, Dual Projected-FISTA smoothing, non-causal Hilbert equalization,
/// and biophysical telemetry export.
#[no_mangle]
pub unsafe extern "C" fn chronos_post_tick(dt_sec: f32) -> u32 {
    let a = arena_mut();
    if a.ingest_ctrl[CTRL_TICK_PHASE] == 0 {
        a.wave1_telem[8] = INGEST_STATUS_PHASE_ERR;
        return a.last_display_slot;
    }

    let (frame_w, frame_h, grid_w, grid_h, scale_px, pip_dim, s) = mode_dims(a.res_mode);
    let s2 = s * s;
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;
    let s_i32 = scale_px as i32;

    let f_idx = a.frame_idx;
    let cursor = a.ring_cursor;
    let actual_dt = if cursor > 0 {
        let d = (a.hist_t_sec[(f_idx as usize) & HIST_MASK]
            - a.hist_t_sec[(f_idx.wrapping_sub(1) as usize) & HIST_MASK]) as f32;
        if d.is_finite() && d > 1e-4 {
            d
        } else if dt_sec.is_finite() {
            dt_sec
        } else {
            1.0 / 60.0
        }
    } else if dt_sec.is_finite() {
        dt_sec
    } else {
        1.0 / 60.0
    };
    let safe_dt = actual_dt.clamp(0.002, 0.15);
    let fps = (1.0 / safe_dt).clamp(8.0, 240.0);
    let dt_frames_60 = (safe_dt * 60.0).clamp(0.25, 6.5);

    let slot_in = ring_slot_back_cursor(cursor, 0);
    let arc_slot_in = (cursor & ((ARC_HIST_LEN - 1) as u64)) as usize;
    let eff_k = (a.lookahead_k as u64).min(cursor).min((RING_LEN - 1) as u64) as usize;
    let slot_disp = ring_slot_back_cursor(cursor, eff_k);
    a.last_display_slot = slot_disp as u32;
    a.active_rgba_slot = slot_in as u32;

    // Sanitize JS-written spectral buffer and eigenvalues against external NaN/Inf writes
    if !a.fiedler_lambda2.is_finite() || a.fiedler_lambda2 < 0.0 {
        a.fiedler_lambda2 = 0.0;
    }
    if !a.fiedler_gap.is_finite() || a.fiedler_gap < 0.0 {
        a.fiedler_gap = 0.0;
    }
    for fv in a.fiedler_vec.iter_mut() {
        if !fv.is_finite() {
            *fv = 0.0;
        }
    }

    if dense_swarm_hero_cap_active(a) {
        enforce_dense_swarm_hero_cap(a, DENSE_SWARM_HERO_CAP, f_idx, s);
    }

    let mut n_active_tracks = 0usize;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            n_active_tracks += 1;
        }
    }
    let n_graph_nodes = n_active_tracks + a.num_peaks;

    // Eigengap confidence weight for sign-invariant spectral cost C_spec = gamma * c_gap * (v_m - v_{8+p})^2
    // For n=2, L = [[w,-w],[-w,w]] has fixed v_2 = [1,-1]/sqrt(2) and lambda_2 = 2*w in [0, 2]:
    // collocated track-peak pairs have lambda_2 -> 2.0 (zero penalty), whereas cross-lane pulls drop lambda_2 < 1.75.
    let gap_conf = if n_graph_nodes >= 3 && a.fiedler_lambda2 > 1e-5 {
        let denom = a.fiedler_lambda2 + a.fiedler_gap + 1e-4;
        let g = if a.fiedler_gap > 0.0 {
            a.fiedler_gap / denom
        } else {
            0.0
        };
        g.clamp(0.0, 1.0)
    } else if n_graph_nodes == 2 && a.fiedler_lambda2 > 1e-5 {
        ((1.75 - a.fiedler_lambda2) * 0.85).clamp(0.0, 0.55)
    } else {
        0.0
    };
    let eff_spec_gamma = a.spectral_gamma * gap_conf;

    // ------------------------------------------------------------------------
    // 3. Pyramid Coarse-to-Fine Proposal + Jonker-Volgenant Global Bipartite
    //    Assignment + 2D IC-LK Sub-Pixel Refinement + Non-Causal Ring Backtrack
    // ------------------------------------------------------------------------
    let mut peak_claimed = [0u8; MAX_TARGETS];
    let cadence_scale = dt_frames_60.clamp(1.0, 1.85);
    let gate_cadence = dt_frames_60.clamp(1.0, 1.45);
    let spec_lookup_r2 = (24.0 * s) * (24.0 * s);
    let hyp_merge_r2 = (5.5 * s) * (5.5 * s);
    let v_clamp = 24.0 * s * cadence_scale;
    let a_clamp = 8.0 * s * cadence_scale;
    let iso_r2 = (32.0 * s) * (32.0 * s);
    let conflict_r2 = (10.5 * s) * (10.5 * s);
    let claim_r2 = if a.lab_mode == 2
        && (a.cam_pan_streak >= 1 || a.last_pan_dx != 0 || a.last_pan_dy != 0)
    {
        (10.0 * s) * (10.0 * s)
    } else {
        (22.0 * s) * (22.0 * s)
    };

    // Record pairwise spatial interaction history (`proximity_mask`) for internal swap self-audit
    let prox_now_r2 = (54.0 * s) * (54.0 * s);
    let prox_coast_r2 = (96.0 * s) * (96.0 * s);
    for i in 0..MAX_TARGETS {
        if a.targets[i].active == 0 {
            continue;
        }
        for j in (i + 1)..MAX_TARGETS {
            if a.targets[j].active == 0 {
                continue;
            }
            let dx = a.targets[i].pos_x - a.targets[j].pos_x;
            let dy = a.targets[i].pos_y - a.targets[j].pos_y;
            let pdx = a.targets[i].pred_cx - a.targets[j].pred_cx;
            let pdy = a.targets[i].pred_cy - a.targets[j].pred_cy;
            let close_now =
                (dx * dx + dy * dy < prox_now_r2) || (pdx * pdx + pdy * pdy < prox_now_r2);
            let close_coast = (a.targets[i].coasting > 0 || a.targets[j].coasting > 0)
                && (dx * dx + dy * dy < prox_coast_r2);
            if close_now || close_coast {
                a.targets[i].proximity_mask |= 1u16 << j;
                a.targets[j].proximity_mask |= 1u16 << i;
            }
        }
    }

    let mut act_tracks = [0usize; MAX_TARGETS];
    let mut nr = 0usize;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            a.targets[m].just_confirmed = 0;
            act_tracks[nr] = m;
            nr += 1;
        }
    }

    // Build shared deduplicated hypothesis pool `hyps[0..n_hyps]` (`<= MAX_HYPS = 24`):
    // 1) All valid ECA connected-component peaks `a.peaks[0..num_peaks]`
    // 2) Up to 2 distinct pyramid-refined patch proposals per active track
    let mut hyps_x = [0.0f32; MAX_HYPS];
    let mut hyps_y = [0.0f32; MAX_HYPS];
    let mut n_hyps = 0usize;

    for p in 0..a.num_peaks {
        if a.peaks[p].valid != 0 && n_hyps < MAX_HYPS {
            hyps_x[n_hyps] = a.peaks[p].x;
            hyps_y[n_hyps] = a.peaks[p].y;
            n_hyps += 1;
        }
    }

    let mut row_tpl_stats = [(0.0f32, 0.0f32); MAX_TARGETS];
    let mut row_prop_cache = [[PropFineCache::empty(); 2]; MAX_TARGETS];
    for r_i in 0..nr {
        let m = act_tracks[r_i];
        if a.targets[m].tpl_grad_dirty != 0 {
            let tl_ptr = a.targets[m].tpl_luma.as_ptr();
            let trg_ptr = a.targets[m].tpl_rg.as_ptr();
            let tby_ptr = a.targets[m].tpl_by.as_ptr();
            let tw_ptr = a.targets[m].tpl_weight.as_ptr();
            build_tpl_ic_lk_cache(tl_ptr, trg_ptr, tby_ptr, tw_ptr, &mut a.ic_lk_cache[m]);
            a.targets[m].tpl_grad_dirty = 0;
        }
        let tpl_inv_w = a.ic_lk_cache[m].tpl_inv_w;
        let tpl_mu_t = a.ic_lk_cache[m].tpl_mu_t;
        row_tpl_stats[r_i] = (tpl_inv_w, tpl_mu_t);
        let props = propose_track_candidates_pyr_ex(
            a,
            m,
            frame_w,
            frame_h,
            scale_px,
            s,
            tpl_inv_w,
            tpl_mu_t,
            &mut row_prop_cache[r_i],
        );
        for k in 0..2usize {
            if !props[k].2 || n_hyps >= MAX_HYPS {
                continue;
            }
            let px = props[k].0;
            let py = props[k].1;
            let mut merged = false;
            for h in 0..n_hyps {
                let dx = px - hyps_x[h];
                let dy = py - hyps_y[h];
                if dx * dx + dy * dy < hyp_merge_r2 {
                    // Snap existing peak hypothesis slightly toward finer template-aligned center
                    hyps_x[h] = 0.5 * (hyps_x[h] + px);
                    hyps_y[h] = 0.5 * (hyps_y[h] + py);
                    merged = true;
                    break;
                }
            }
            if !merged {
                hyps_x[n_hyps] = px;
                hyps_y[n_hyps] = py;
                n_hyps += 1;
            }
        }
    }

    // Assemble rectangular bipartite cost matrix `a.jv_cost[nr][n_hyps + nr]`
    // Cache fine evaluation per `(r_i, h)` so winning pairs need zero re-evaluation!
    let nc = n_hyps + nr;
    for r_i in 0..nr {
        a.jv_cost[r_i][..nc].fill(1e9);
        a.eval_ok[r_i][..n_hyps].fill(false);
    }
    let mut fin_hyps = [0u64; MAX_TARGETS];

    let vis_pred_r = 36.0 * s * gate_cadence;
    let vis_curr_r = 39.5 * s * gate_cadence;
    let vis_pred_r2 = vis_pred_r * vis_pred_r;
    let vis_curr_r2 = vis_curr_r * vis_curr_r;
    let inv_kin_scale2 = 1.0 / ((30.0 * s) * (30.0 * s));

    let no_scene_grav = a.scene_gravity_arcs == 0
        || (a.scene_gravity_arcs == 1
            && a.scene_gravity_conf < 0.25
            && a.scene_gravity_px_s2 < 2500.0 * s);

    #[derive(Copy, Clone, Default)]
    struct PostRowVis {
        m: u8,
        planar_oneway: bool,
        planar_birth_base: bool,
        planar_coast_oneway: bool,
        planar_lane_reacq: bool,
        anchor_chromatic: bool,
        v_track: f32,
        v_mag_t: f32,
        v_mag2_t: f32,
        max_cpar_vis: f32,
        max_cperp_vis: f32,
        b_dx_t: f32,
        b_dy_t: f32,
        b_len_t: f32,
        be_dx_t: f32,
        be_dy_t: f32,
        be_len_t: f32,
        speed_l1: f32,
    }
    let mut row_vis = [PostRowVis::default(); MAX_TARGETS];
    let mut row_null = [0.54f32; MAX_TARGETS];
    let mut row_mask = [0u64; MAX_TARGETS];
    let mut col_mask = [0u16; MAX_HYPS];

    for r_i in 0..nr {
        let m = act_tracks[r_i];
        let is_conf = a.targets[m].confirmed != 0;
        // Dedicated coasting / occlusion null column for row `r_i`
        // Confirmed tracks slightly outbid tentative tracks for contested hypotheses
        let null_c_r = if is_conf { 0.54 } else { 0.46 };
        a.jv_cost[r_i][n_hyps + r_i] = null_c_r;
        row_null[r_i] = null_c_r;

        let t = &a.targets[m];
        let coasting = t.coasting;
        let pred_cx = t.pred_cx;
        let pred_cy = t.pred_cy;
        let v_track = a.fiedler_vec[m];

        let v_ent = wasm_sqrtf(t.entry_vx * t.entry_vx + t.entry_vy * t.entry_vy);
        let cf = coasting as f32;
        let (r_entry_max, d_kin_max) = if a.lab_mode == 1 && nr >= 5 && t.confirmed != 0 {
            (
                (16.0 * s + cf * (v_ent + 1.5 * s)).min(26.0 * s),
                (20.0 * s + cf * 2.0 * s).min(28.0 * s),
            )
        } else {
            (
                (26.0 * s + cf * (v_ent.max(4.2 * s) + 2.4 * s)).min(148.0 * s),
                (34.0 * s + cf * 3.5 * s).min(118.0 * s),
            )
        };
        let r_entry_max2 = r_entry_max * r_entry_max;

        let confirmed_ballistic = t.is_ballistic != 0;
        let non_harmonic = t.pivot_valid == 0
            && ((a.lab_mode == 0
                && (t.retrace_dirs != 3
                    || (t.osc_ext_max_x - t.osc_ext_min_x).abs() < 20.0 * s))
                || !(t.phase_valid != 0 && t.tremor_coherence > 0.25));
        let v_mag2_t = t.vel_x * t.vel_x + t.vel_y * t.vel_y;
        let v_mag_t = wasm_sqrtf(v_mag2_t);
        let b_dx_t = t.pos_x - t.birth_x;
        let b_dy_t = t.pos_y - t.birth_y;
        let b_d2_t = b_dx_t * b_dx_t + b_dy_t * b_dy_t;
        let b_len_t = wasm_sqrtf(b_d2_t);
        let be_dx_t = t.entry_x - t.birth_x;
        let be_dy_t = t.entry_y - t.birth_y;
        let be_d2_t = be_dx_t * be_dx_t + be_dy_t * be_dy_t;
        let be_len_t = wasm_sqrtf(be_d2_t);
        let achromatic_track = (t.anchor_sig & (1 << 31)) == 0;
        let anchor_chromatic = !achromatic_track;
        let speed_l1 = t.vel_x.abs() + t.vel_y.abs();
        let planar_coast_oneway = a.lab_mode == 0
            && no_scene_grav
            && achromatic_track
            && t.confirmed != 0
            && non_harmonic
            && !confirmed_ballistic
            && t.age >= 10
            && be_d2_t >= (20.0 * s) * (20.0 * s);
        let planar_oneway = a.lab_mode == 0
            && no_scene_grav
            && t.confirmed != 0
            && non_harmonic
            && !confirmed_ballistic
            && t.age >= 10
            && b_d2_t >= (16.0 * s) * (16.0 * s)
            && v_mag2_t >= (1.15 * s) * (1.15 * s)
            && (b_dx_t * t.vel_x + b_dy_t * t.vel_y) > 0.45 * b_len_t * v_mag_t;
        let planar_birth_base = a.lab_mode == 0
            && no_scene_grav
            && t.confirmed != 0
            && non_harmonic
            && !confirmed_ballistic
            && t.age >= 10
            && b_d2_t >= (20.0 * s) * (20.0 * s);
        let max_par_vis = (2.0 * v_mag_t + 11.0 * s) * v_mag_t;
        let max_perp_vis = (11.0 * s + 0.30 * v_mag_t) * v_mag_t;
        let max_cpar_vis = (2.0 * v_mag_t + 12.0 * s) * v_mag_t;
        let max_cperp_vis = (12.0 * s + 0.30 * v_mag_t) * v_mag_t;

        let mut lin_cx = 0.0f32;
        let mut lin_cy = 0.0f32;
        let mut refl_x = 0.0f32;
        let mut refl_y = 0.0f32;
        let mut planar_linear_lane = false;
        let mut ux_e = 0.0f32;
        let mut uy_e = 0.0f32;
        let mut max_fwd_e = 0.0f32;
        let mut dx_ch = 0.0f32;
        let mut dy_ch = 0.0f32;
        let mut ch_len2 = 0.0f32;
        let mut ch_len = 0.0f32;
        let mut planar_lane_reacq = false;
        let mut mp_c = 0.0f32;

        if coasting > 0 {
            let cf2 = cf.min(3.0);
            let (ex_x, ex_y, _, _) = predict_curved_step(
                t.entry_x,
                t.entry_y,
                t.entry_vx,
                t.entry_vy,
                t.entry_omega_f,
                t.grav_ay_frame,
                t.is_ballistic,
                cf2,
            );
            let exited_border = cf >= 2.0
                && (ex_x < half_patch - 3.0 * s
                    || ex_x > (frame_w as f32) - half_patch + 3.0 * s
                    || ex_y < half_patch - 3.0 * s
                    || ex_y > (frame_h as f32) - half_patch + 3.0 * s
                    || (a.lab_mode == 0
                        && cf >= 3.0
                        && is_border_clamped_exit(
                            t.entry_x,
                            t.entry_y,
                            t.entry_vx,
                            t.entry_vy,
                            frame_w,
                            frame_h,
                            half_patch,
                            s,
                        )));
            if exited_border {
                continue;
            }
            planar_linear_lane = a.lab_mode == 0
                && !confirmed_ballistic
                && t.pivot_valid == 0
                && v_ent >= 1.15 * s
                && t.entry_omega_f.abs() < 0.02;
            if planar_linear_lane {
                ux_e = t.entry_vx / v_ent;
                uy_e = t.entry_vy / v_ent;
                max_fwd_e = cf * (v_ent * 1.45 + 1.2 * s) + 18.0 * s;
            }
            lin_cx = (t.entry_x + cf * t.entry_vx).clamp(half_patch, (frame_w as f32) - half_patch);
            lin_cy = (t.entry_y + cf * t.entry_vy).clamp(half_patch, (frame_h as f32) - half_patch);
            refl_x = t.entry_x - cf * t.entry_vx;
            refl_y = t.entry_y - cf * t.entry_vy;
            dx_ch = pred_cx - t.entry_x;
            dy_ch = pred_cy - t.entry_y;
            ch_len2 = dx_ch * dx_ch + dy_ch * dy_ch;
            if coasting >= 3 && ch_len2 > (18.0 * s) * (18.0 * s) {
                ch_len = wasm_sqrtf(ch_len2);
            }
            planar_lane_reacq = a.lab_mode == 0
                && !confirmed_ballistic
                && t.pivot_valid == 0
                && (t.entry_vx * t.entry_vx + t.entry_vy * t.entry_vy) >= (1.15 * s) * (1.15 * s)
                && t.entry_omega_f.abs() < 0.02;
            mp_c = ((cf + 1.0) * v_ent + 6.5 * s).max(11.0 * s);
        }
        row_vis[r_i] = PostRowVis {
            m: m as u8,
            planar_oneway,
            planar_birth_base,
            planar_coast_oneway,
            planar_lane_reacq,
            anchor_chromatic,
            v_track,
            v_mag_t,
            v_mag2_t,
            max_cpar_vis,
            max_cperp_vis,
            b_dx_t,
            b_dy_t,
            b_len_t,
            be_dx_t,
            be_dy_t,
            be_len_t,
            speed_l1,
        };

        for h in 0..n_hyps {
            let hx = hyps_x[h];
            let hy = hyps_y[h];

            let kin_cost = if coasting == 0 {
                let dxp = hx - pred_cx;
                let dyp = hy - pred_cy;
                let dxc = hx - t.pos_x;
                let dyc = hy - t.pos_y;
                let d2_curr = dxc * dxc + dyc * dyc;
                if planar_birth_base && d2_curr > (14.0 * s) * (14.0 * s) {
                    let vel_aligned = v_mag2_t >= (4.0 * s) * (4.0 * s)
                        && (dxc * t.vel_x + dyc * t.vel_y) > 0.50 * wasm_sqrtf(d2_curr) * v_mag_t;
                    if !vel_aligned && (dxc * b_dx_t + dyc * b_dy_t) < -4.0 * s * b_len_t {
                        continue;
                    }
                }
                if planar_oneway {
                    let par_vis = dxc * t.vel_x + dyc * t.vel_y;
                    let perp_vis = (-dxc * t.vel_y + dyc * t.vel_x).abs();
                    if par_vis < -4.2 * s * v_mag_t || par_vis > max_par_vis || perp_vis > max_perp_vis {
                        continue;
                    }
                }
                if t.confirmed != 0 && non_harmonic && v_mag2_t >= (8.5 * s) * (8.5 * s) {
                    // Anisotropic heading-aligned elliptical gate: expand longitudinal reach (1.08x)
                    // along velocity vector while tightening cross-track lateral radius (0.90x) & cost (1.24x).
                    let ux_h = t.vel_x / v_mag_t;
                    let uy_h = t.vel_y / v_mag_t;
                    let d_par = dxp * ux_h + dyp * uy_h;
                    let d_perp = -dxp * uy_h + dyp * ux_h;
                    let a_par = 1.08 * vis_pred_r;
                    let b_perp = 0.90 * vis_pred_r;
                    let norm_ell2 = (d_par * d_par) / (a_par * a_par)
                        + (d_perp * d_perp) / (b_perp * b_perp);
                    if norm_ell2 > 1.0 || d2_curr > vis_curr_r2 {
                        continue;
                    }
                    let d2_mahal = 0.94 * d_par * d_par + 1.24 * d_perp * d_perp;
                    0.036 * d2_mahal * inv_kin_scale2
                } else {
                    let (loc_pred_r2, loc_curr_r2) = if a.lab_mode == 1 && nr >= 5 && t.confirmed != 0
                    {
                        ((18.0 * s) * (18.0 * s), (20.0 * s) * (20.0 * s))
                    } else if a.lab_mode == 2 && t.confirmed != 0 {
                        let mp = (v_mag_t * 2.2 + 8.5 * s).max(11.0 * s);
                        let mc = mp + 2.0 * s;
                        ((mp * mp).min(vis_pred_r2), (mc * mc).min(vis_curr_r2))
                    } else {
                        (vis_pred_r2, vis_curr_r2)
                    };
                    let d2_pred = dxp * dxp + dyp * dyp;
                    if d2_pred > loc_pred_r2 || d2_curr > loc_curr_r2 {
                        continue;
                    }
                    let base_kin = 0.036 * d2_pred * inv_kin_scale2;
                    if a.lab_mode == 0
                        && t.confirmed != 0
                        && t.proximity_mask != 0
                        && !confirmed_ballistic
                        && t.pivot_valid == 0
                        && v_mag2_t >= (1.8 * s) * (1.8 * s)
                    {
                        let d_perp = (-dxp * t.vel_y + dyp * t.vel_x) / v_mag_t;
                        base_kin + 0.11 * (d_perp * d_perp) / ((10.0 * s) * (10.0 * s))
                    } else {
                        base_kin
                    }
                }
            } else {
                let dxe = hx - t.entry_x;
                let dye = hy - t.entry_y;
                let d2_ent = dxe * dxe + dye * dye;
                if d2_ent > r_entry_max2 {
                    continue;
                }
                if planar_coast_oneway
                    && d2_ent > (14.0 * s) * (14.0 * s)
                    && (dxe * be_dx_t + dye * be_dy_t) < -4.0 * s * be_len_t
                {
                    continue;
                }
                if planar_linear_lane {
                    let par_e = dxe * ux_e + dye * uy_e;
                    let perp_e = (-dxe * uy_e + dye * ux_e).abs();
                    if par_e < -4.0 * s || par_e > max_fwd_e || perp_e > 14.5 * s {
                        continue;
                    }
                }
                let d_ent = wasm_sqrtf(d2_ent) + 4.0 * s;
                let d_pass_curv =
                    wasm_sqrtf((hx - pred_cx) * (hx - pred_cx) + (hy - pred_cy) * (hy - pred_cy));
                let d_pass_lin =
                    wasm_sqrtf((hx - lin_cx) * (hx - lin_cx) + (hy - lin_cy) * (hy - lin_cy));
                let mut d_pass = d_pass_curv.min(d_pass_lin);
                if ch_len > 0.0 {
                    let tau = ((dxe * dx_ch + dye * dy_ch) / ch_len2).clamp(0.58, 1.22);
                    let cor_x = t.entry_x + tau * dx_ch;
                    let cor_y = t.entry_y + tau * dy_ch;
                    let d_perp_cor = wasm_sqrtf((hx - cor_x) * (hx - cor_x) + (hy - cor_y) * (hy - cor_y));
                    let d_long_dev = (tau - 1.0).abs() * ch_len;
                    d_pass = d_pass.min(d_perp_cor + 0.32 * d_long_dev);
                }
                let d_refl = wasm_sqrtf((hx - refl_x) * (hx - refl_x) + (hy - refl_y) * (hy - refl_y)) + 2.0 * s;
                let d_kin = if planar_linear_lane {
                    d_pass.min(d_ent)
                } else {
                    d_pass.min(d_ent).min(d_refl)
                };
                if a.lab_mode == 2 && t.confirmed != 0 && coasting <= 3 && d_pass > mp_c && (d_refl - 2.0 * s) > 7.5 * s {
                    continue;
                }
                if d_kin > d_kin_max {
                    continue;
                }
                0.048 * (d_kin * d_kin) * inv_kin_scale2 + 0.030
            };

            let null_cost = if is_conf { 0.54f32 } else { 0.46f32 };
            if kin_cost >= null_cost {
                continue;
            }
            if a.lab_mode == 0 {
                let (ref_x, ref_y) = if coasting > 0 {
                    (t.entry_x, t.entry_y)
                } else {
                    (t.pos_x, t.pos_y)
                };
                let hjdx = hx - ref_x;
                let hjdy = hy - ref_y;
                let big_jump_r2 = (16.0 * s) * (16.0 * s);
                if hjdx * hjdx + hjdy * hjdy > big_jump_r2 {
                    let own_r2 = (6.5 * s) * (6.5 * s);
                    let mut h_owned = false;
                    for o in 0..MAX_TARGETS {
                        if o != m
                            && a.targets[o].active != 0
                            && a.targets[o].confirmed != 0
                            && a.targets[o].coasting == 0
                            && a.targets[o].age >= 6
                        {
                            let hdx_p = hx - a.targets[o].pred_cx;
                            let hdy_p = hy - a.targets[o].pred_cy;
                            let hdx_c = hx - a.targets[o].pos_x;
                            let hdy_c = hy - a.targets[o].pos_y;
                            if hdx_p * hdx_p + hdy_p * hdy_p < own_r2
                                || hdx_c * hdx_c + hdy_c * hdy_c < own_r2
                            {
                                h_owned = true;
                                break;
                            }
                        }
                    }
                    if h_owned {
                        continue;
                    }
                }
            }

            a.eval_raw[r_i][h] = kin_cost;
            row_mask[r_i] |= 1u64 << h;
            col_mask[h] |= 1u16 << r_i;
        }
    }

    let mut row_ub = row_null;
    let mut pass2a_h = [-1i8; MAX_TARGETS];

    macro_rules! eval_post_cand {
        ($r_i:expr, $h:expr, $kin_cost:expr, $fin_hyps:expr) => {{
        let r_i: usize = $r_i;
        let h: usize = $h;
        let kin_cost: f32 = $kin_cost;
        let rv = &row_vis[r_i];
        let m = rv.m as usize;
        let coasting = a.targets[m].coasting;
        let t_pos_x = a.targets[m].pos_x;
        let t_pos_y = a.targets[m].pos_y;
        let t_vel_x = a.targets[m].vel_x;
        let t_vel_y = a.targets[m].vel_y;
        let v_track = rv.v_track;
        let v_mag_t = rv.v_mag_t;
        let v_mag2_t = rv.v_mag2_t;
        let max_cpar_vis = rv.max_cpar_vis;
        let max_cperp_vis = rv.max_cperp_vis;
        let b_dx_t = rv.b_dx_t;
        let b_dy_t = rv.b_dy_t;
        let b_len_t = rv.b_len_t;
        let be_dx_t = rv.be_dx_t;
        let be_dy_t = rv.be_dy_t;
        let be_len_t = rv.be_len_t;
        let speed_l1 = rv.speed_l1;
        let planar_oneway = rv.planar_oneway;
        let planar_birth_base = rv.planar_birth_base;
        let planar_coast_oneway = rv.planar_coast_oneway;
        let planar_lane_reacq = rv.planar_lane_reacq;
        let anchor_chromatic = rv.anchor_chromatic;
        let hx = hyps_x[h];
        let hy = hyps_y[h];
        let ok_cand: bool = 'eval_blk: {
            if coasting == 0
                && planar_oneway
                && hx >= half_patch + 1.0
                && hx <= (frame_w as f32) - half_patch - 1.0
                && hy >= half_patch + 1.0
                && hy <= (frame_h as f32) - half_patch - 1.0
            {
                let hdx = hx - t_pos_x;
                let hdy = hy - t_pos_y;
                let snap_pad = (2.0 * s + 1.55) * (t_vel_x.abs() + t_vel_y.abs());
                let hpar = hdx * t_vel_x + hdy * t_vel_y;
                let hperp = (-hdx * t_vel_y + hdy * t_vel_x).abs();
                if hpar + snap_pad < -3.6 * s * v_mag_t
                    || hpar - snap_pad > max_cpar_vis
                    || hperp - snap_pad > max_cperp_vis
                {
                    break 'eval_blk false;
                }
            }
            let r_pad_geom = 6.0 * s;
            if coasting == 0 && planar_birth_base {
                let hdxc_j = hx - t_pos_x;
                let hdyc_j = hy - t_pos_y;
                let hd2_jump = hdxc_j * hdxc_j + hdyc_j * hdyc_j;
                let min_jump_r = 14.0 * s + r_pad_geom;
                if hd2_jump > min_jump_r * min_jump_r {
                    let vel_maybe_aligned = v_mag2_t >= (4.0 * s) * (4.0 * s)
                        && (hdxc_j * t_vel_x + hdyc_j * t_vel_y) > 0.0;
                    if !vel_maybe_aligned
                        && (hdxc_j * b_dx_t + hdyc_j * b_dy_t) < -(4.0 * s + r_pad_geom) * b_len_t
                    {
                        break 'eval_blk false;
                    }
                }
            }
            if coasting > 0 && planar_coast_oneway {
                let hdxe = hx - a.targets[m].entry_x;
                let hdye = hy - a.targets[m].entry_y;
                let min_coast_r = 14.0 * s + r_pad_geom;
                if hdxe * hdxe + hdye * hdye > min_coast_r * min_coast_r
                    && (hdxe * be_dx_t + hdye * be_dy_t) < -(4.0 * s + r_pad_geom) * be_len_t
                {
                    break 'eval_blk false;
                }
            }
            let mut hyp_on_owner = false;
            let mut hj_big = false;
            if a.lab_mode == 0 {
                let (ref_x, ref_y) = if coasting > 0 {
                    (a.targets[m].entry_x, a.targets[m].entry_y)
                } else {
                    (t_pos_x, t_pos_y)
                };
                let hjdx = hx - ref_x;
                let hjdy = hy - ref_y;
                let big_jump_r2 = (16.0 * s) * (16.0 * s);
                hj_big = hjdx * hjdx + hjdy * hjdy > big_jump_r2;
                if hj_big {
                    let own_r2 = (6.5 * s) * (6.5 * s);
                    for o in 0..MAX_TARGETS {
                        if o != m
                            && a.targets[o].active != 0
                            && a.targets[o].confirmed != 0
                            && a.targets[o].coasting == 0
                            && a.targets[o].age >= 6
                        {
                            let hdx_p = hx - a.targets[o].pred_cx;
                            let hdy_p = hy - a.targets[o].pred_cy;
                            let hdx_c = hx - a.targets[o].pos_x;
                            let hdy_c = hy - a.targets[o].pos_y;
                            if hdx_p * hdx_p + hdy_p * hdy_p < own_r2
                                || hdx_c * hdx_c + hdy_c * hdy_c < own_r2
                            {
                                hyp_on_owner = true;
                                break;
                            }
                        }
                    }
                }
            }
            let hyp_owner_lb = if hyp_on_owner { 0.65f32 } else { 0.0f32 };
            let max_base_c = row_ub[r_i] - kin_cost - hyp_owner_lb;
            if max_base_c < -0.1361f32 {
                break 'eval_blk false;
            }
            let ts = row_tpl_stats[r_i];
            let (btlx, btly, raw_c, app_err, anc_l1, weca, csig, ok) =
                eval_track_hypothesis_fine_cached(
                    a,
                    m,
                    hx,
                    hy,
                    frame_w,
                    frame_h,
                    scale_px,
                    s,
                    max_base_c,
                    ts.0,
                    ts.1,
                    &row_prop_cache[r_i],
                );
            if !ok {
                break 'eval_blk false;
            }
            let t = &a.targets[m];

            let cand_cx = (btlx as f32) + half_patch;
            let cand_cy = (btly as f32) + half_patch;
            if coasting == 0 && planar_oneway {
                let cdxc = cand_cx - t.pos_x;
                let cdyc = cand_cy - t.pos_y;
                let cpar_vis = cdxc * t.vel_x + cdyc * t.vel_y;
                let cperp_vis = (-cdxc * t.vel_y + cdyc * t.vel_x).abs();
                if cpar_vis < -3.6 * s * v_mag_t || cpar_vis > max_cpar_vis || cperp_vis > max_cperp_vis {
                    break 'eval_blk false;
                }
            }
            let cdxc_j = cand_cx - t.pos_x;
            let cdyc_j = cand_cy - t.pos_y;
            let d2_jump = cdxc_j * cdxc_j + cdyc_j * cdyc_j;
            if coasting == 0
                && planar_birth_base
                && d2_jump > (14.0 * s) * (14.0 * s)
            {
                let vel_aligned = v_mag2_t >= (4.0 * s) * (4.0 * s)
                    && (cdxc_j * t.vel_x + cdyc_j * t.vel_y) > 0.50 * wasm_sqrtf(d2_jump) * v_mag_t;
                if !vel_aligned && (cdxc_j * b_dx_t + cdyc_j * b_dy_t) < -4.0 * s * b_len_t {
                    break 'eval_blk false;
                }
            }
            if coasting > 0 && planar_coast_oneway {
                let cdxe = cand_cx - t.entry_x;
                let cdye = cand_cy - t.entry_y;
                if cdxe * cdxe + cdye * cdye > (14.0 * s) * (14.0 * s)
                    && (cdxe * be_dx_t + cdye * be_dy_t) < -4.0 * s * be_len_t
                {
                    break 'eval_blk false;
                }
            }
            let far_static_surf =
                a.lab_mode == 0 && scale_px > 1 && d2_jump > (9.0 * s) * (9.0 * s) && weca < 0.008;
            let static_bg_occluder =
                if a.lab_mode == 0 && (weca < 0.016 || app_err >= 0.31 || scale_px == 1) {
                    let core_step = (2 * scale_px) as isize;
                let cxi = (cand_cx as isize).clamp(core_step, (frame_w as isize) - 1 - core_step);
                let cyi = (cand_cy as isize).clamp(core_step, (frame_h as isize) - 1 - core_step);
                let mut bg_core_hits = 0u32;
                for &(ox, oy) in &[
                    (0isize, 0isize),
                    (-core_step, 0),
                    (core_step, 0),
                    (0, -core_step),
                    (0, core_step),
                ] {
                    let pidx = ((cyi + oy) as usize) * frame_w + ((cxi + ox) as usize);
                    if a.bg_stat_streak[pidx] >= 10
                        && a.eca_res[pidx] < 0.015
                        && (a.luma[pidx] - a.bg_luma[pidx]).abs() < 0.028
                    {
                        bg_core_hits += 1;
                    }
                }
                bg_core_hits >= 4
            } else {
                false
            };
            let opp_pole_cand = has_opposite_chroma_pole(t.anchor_sig, csig);
            let c_val_reacq = padic_ultrametric_val(t.anchor_sig, csig);
            let vis_ok = if coasting == 0 {
                if far_static_surf || static_bg_occluder {
                    false
                } else if anchor_chromatic {
                    let max_anc_chr = if a.lab_mode == 0
                        && scale_px > 1
                        && speed_l1 > 2.5 * s
                        && weca > 0.035
                        && app_err < 0.135
                    {
                        0.46
                    } else {
                        0.42
                    };
                    app_err < 0.42 && anc_l1 < max_anc_chr && (weca > 0.011 || app_err < 0.135)
                } else {
                    let max_anc_vis = if a.lab_mode == 0
                        && scale_px > 1
                        && speed_l1 > 2.5 * s
                        && weca > 0.035
                        && app_err < 0.130
                    {
                        0.44
                    } else if speed_l1 > 2.5 * s {
                        0.36
                    } else {
                        0.30
                    };
                    app_err < 0.38 && anc_l1 < max_anc_vis && (weca > 0.011 || app_err < 0.125)
                }
            } else if opp_pole_cand || far_static_surf || static_bg_occluder {
                false
            } else if coasting <= 5 {
                let brief_fg = weca > 0.013 || (coasting <= 2 && app_err < 0.125 && weca > 0.007);
                if anchor_chromatic {
                    let max_anc = if a.lab_mode == 0 && (scale_px > 1 || planar_lane_reacq) && app_err < 0.24 {
                        0.44
                    } else {
                        0.36
                    };
                    brief_fg && app_err < 0.38 && anc_l1 < max_anc
                } else {
                    let max_anc = if a.lab_mode == 0 && (scale_px > 1 || planar_lane_reacq) && app_err < 0.22 && c_val_reacq >= 8 {
                        0.41
                    } else {
                        0.33
                    };
                    brief_fg && app_err < 0.34 && anc_l1 < max_anc
                }
            } else if anchor_chromatic {
                let max_anc = if a.lab_mode == 0 && (scale_px > 1 || planar_lane_reacq) && c_val_reacq >= 10 && app_err < 0.26 {
                    0.39
                } else {
                    0.32
                };
                weca > 0.018 && app_err < 0.34 && anc_l1 < max_anc
            } else {
                let max_anc = if a.lab_mode == 0 && (scale_px > 1 || planar_lane_reacq) && c_val_reacq >= 8 && app_err < 0.24 {
                    0.37
                } else {
                    0.28
                };
                weca > 0.018 && app_err < 0.30 && anc_l1 < max_anc
            };
            if !vis_ok {
                break 'eval_blk false;
            }
            let mut lm1_pen = 0.0f32;
            if a.lab_mode == 1 {
                let cpx = (btlx as f32) + half_patch;
                let cpy = (btly as f32) + half_patch;
                let mut min_pk_d2 = 1.0e12f32;
                for p in 0..a.num_peaks {
                    if a.peaks[p].valid != 0 {
                        let pdx = cpx - a.peaks[p].x;
                        let pdy = cpy - a.peaks[p].y;
                        let d2 = pdx * pdx + pdy * pdy;
                        if d2 < min_pk_d2 {
                            min_pk_d2 = d2;
                        }
                    }
                }
                let dy_lm1 = (cpy - t.pos_y).abs();
                let pdy_lm1 = (cpy - t.pred_cy).abs();
                let dx_lm1 = (cpx - t.pos_x).abs();
                let trig_lm1 = a.num_peaks > 0
                    && t.age >= 4
                    && min_pk_d2 > (6.5 * s) * (6.5 * s)
                    && dy_lm1 > 6.5 * s
                    && dy_lm1 <= 24.0 * s
                    && pdy_lm1 > 6.5 * s
                    && dy_lm1 > 1.45 * dx_lm1;
                if trig_lm1 {
                    lm1_pen = 0.25;
                }
                if a.num_peaks > 0 && min_pk_d2 > (10.0 * s) * (10.0 * s) && weca < 0.012 {
                    break 'eval_blk false;
                }
            }

            let mut spec_pen = 0.0f32;
            if eff_spec_gamma > 0.0 && a.num_peaks > 0 {
                let cand_px = (btlx as f32) + half_patch;
                let cand_py = (btly as f32) + half_patch;
                let mut near_p: Option<usize> = None;
                let mut near_d2 = spec_lookup_r2;
                for p in 0..a.num_peaks {
                    let pdx = cand_px - a.peaks[p].x;
                    let pdy = cand_py - a.peaks[p].y;
                    let pd2 = pdx * pdx + pdy * pdy;
                    if pd2 < near_d2 {
                        near_d2 = pd2;
                        near_p = Some(p);
                    }
                }
                if let Some(p) = near_p {
                    let dv = v_track - a.fiedler_vec[MAX_TARGETS + p];
                    spec_pen = eff_spec_gamma * (dv * dv);
                }
            }

            let mut owner_pen = hyp_owner_lb;
            if a.lab_mode == 0 && !hyp_on_owner {
                let (ref_x, ref_y) = if coasting > 0 {
                    (t.entry_x, t.entry_y)
                } else {
                    (t.pos_x, t.pos_y)
                };
                let jdx = cand_cx - ref_x;
                let jdy = cand_cy - ref_y;
                let big_jump_r2 = (16.0 * s) * (16.0 * s);
                if hj_big || jdx * jdx + jdy * jdy > big_jump_r2 {
                    let own_r2 = (6.5 * s) * (6.5 * s);
                    for o in 0..MAX_TARGETS {
                        if o != m
                            && a.targets[o].active != 0
                            && a.targets[o].confirmed != 0
                            && a.targets[o].coasting == 0
                            && a.targets[o].age >= 6
                        {
                            let odx_p = cand_cx - a.targets[o].pred_cx;
                            let ody_p = cand_cy - a.targets[o].pred_cy;
                            let odx_c = cand_cx - a.targets[o].pos_x;
                            let ody_c = cand_cy - a.targets[o].pos_y;
                            let hyp_hit = if !hj_big {
                                let hdx_p = hx - a.targets[o].pred_cx;
                                let hdy_p = hy - a.targets[o].pred_cy;
                                let hdx_c = hx - a.targets[o].pos_x;
                                let hdy_c = hy - a.targets[o].pos_y;
                                hdx_p * hdx_p + hdy_p * hdy_p < own_r2
                                    || hdx_c * hdx_c + hdy_c * hdy_c < own_r2
                            } else {
                                false
                            };
                            let cand_on_owner =
                                !has_opposite_chroma_pole(a.targets[o].anchor_sig, csig)
                                    && (odx_p * odx_p + ody_p * ody_p < own_r2
                                        || odx_c * odx_c + ody_c * ody_c < own_r2);
                            if hyp_hit || cand_on_owner {
                                owner_pen = 0.65;
                                break;
                            }
                        }
                    }
                }
            }
            let anc_pen = 0.30 * (anc_l1 - 0.045).max(0.0);
            let total_c = raw_c + anc_pen + kin_cost + spec_pen + owner_pen + lm1_pen;
            if total_c >= row_ub[r_i] {
                break 'eval_blk false;
            }

            a.jv_cost[r_i][h] = total_c;
            a.eval_tlx[r_i][h] = btlx;
            a.eval_tly[r_i][h] = btly;
            a.eval_raw[r_i][h] = raw_c;
            a.eval_app[r_i][h] = app_err;
            a.eval_anc[r_i][h] = anc_l1;
            a.eval_eca[r_i][h] = weca;
            a.eval_sig[r_i][h] = csig;
            a.eval_ok[r_i][h] = true;
            $fin_hyps[r_i] |= 1u64 << h;
            true
        };
        if !ok_cand {
            col_mask[h] &= !(1u16 << r_i);
        } else if (col_mask[h] & !(1u16 << r_i)) == 0 && a.jv_cost[r_i][h] < row_ub[r_i] {
            let ub = a.jv_cost[r_i][h];
            row_ub[r_i] = ub;
            let mut rem = row_mask[r_i];
            while rem != 0 {
                let hp = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                if a.eval_raw[r_i][hp] - 0.1361f32 >= ub {
                    row_mask[r_i] &= !(1u64 << hp);
                    col_mask[hp] &= !(1u16 << r_i);
                }
            }
        }
        ok_cand
        }};
    }

    for r_i in 0..nr {
        while row_mask[r_i] != 0 {
            let mut rem = row_mask[r_i];
            let mut best_h = 0usize;
            let mut best_k = 1e9f32;
            while rem != 0 {
                let h = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                let k = a.eval_raw[r_i][h];
                if k < best_k {
                    best_k = k;
                    best_h = h;
                }
            }
            if best_k - 0.1361f32 >= row_ub[r_i] {
                let mut prune = row_mask[r_i];
                while prune != 0 {
                    let hp = prune.trailing_zeros() as usize;
                    prune &= prune - 1;
                    col_mask[hp] &= !(1u16 << r_i);
                }
                row_mask[r_i] = 0;
                break;
            }
            row_mask[r_i] &= !(1u64 << best_h);
            if eval_post_cand!(r_i, best_h, best_k, fin_hyps) {
                pass2a_h[r_i] = best_h as i8;
                break;
            }
        }
    }
    for r_i in 0..nr {
        let ph = pass2a_h[r_i];
        if ph >= 0 {
            let h = ph as usize;
            if (col_mask[h] & !(1u16 << r_i)) == 0 && a.jv_cost[r_i][h] < row_ub[r_i] {
                let ub = a.jv_cost[r_i][h];
                row_ub[r_i] = ub;
                let mut rem = row_mask[r_i];
                while rem != 0 {
                    let hp = rem.trailing_zeros() as usize;
                    rem &= rem - 1;
                    if a.eval_raw[r_i][hp] - 0.1361f32 >= ub {
                        row_mask[r_i] &= !(1u64 << hp);
                        col_mask[hp] &= !(1u16 << r_i);
                    }
                }
            }
        }
    }
    // Pass 2b-clear: resolve contenders on Pass 2a winners early so uncontested row_ub tightens
    for r_i in 0..nr {
        let ph = pass2a_h[r_i];
        if ph >= 0 {
            let h = ph as usize;
            let mut others = col_mask[h] & !(1u16 << r_i);
            while others != 0 {
                let r_j = others.trailing_zeros() as usize;
                others &= others - 1;
                if (row_mask[r_j] & (1u64 << h)) == 0 {
                    continue;
                }
                row_mask[r_j] &= !(1u64 << h);
                let k_j = a.eval_raw[r_j][h];
                if k_j - 0.1361f32 >= row_ub[r_j] {
                    col_mask[h] &= !(1u16 << r_j);
                } else {
                    let _ = eval_post_cand!(r_j, h, k_j, fin_hyps);
                }
            }
            if (col_mask[h] & !(1u16 << r_i)) == 0 && a.jv_cost[r_i][h] < row_ub[r_i] {
                let ub = a.jv_cost[r_i][h];
                row_ub[r_i] = ub;
                let mut rem = row_mask[r_i];
                while rem != 0 {
                    let hp = rem.trailing_zeros() as usize;
                    rem &= rem - 1;
                    if a.eval_raw[r_i][hp] - 0.1361f32 >= ub {
                        row_mask[r_i] &= !(1u64 << hp);
                        col_mask[hp] &= !(1u16 << r_i);
                    }
                }
            }
        }
    }
    let mut safe_priv_ub = row_null;
    for _ub_it in 0..nr.min(4) {
        let mut changed = false;
        for r_i in 0..nr {
            let mut rem_f = fin_hyps[r_i];
            while rem_f != 0 {
                let c = rem_f.trailing_zeros() as usize;
                rem_f &= rem_f - 1;
                if (col_mask[c] & !(1u16 << r_i)) == 0 {
                    let v = a.jv_cost[r_i][c];
                    if v < safe_priv_ub[r_i] {
                        safe_priv_ub[r_i] = v;
                    }
                    if v < row_ub[r_i] {
                        row_ub[r_i] = v;
                        changed = true;
                    }
                }
            }
        }
        for r_i in 0..nr {
            let mut rem_f = fin_hyps[r_i];
            while rem_f != 0 {
                let h = rem_f.trailing_zeros() as usize;
                rem_f &= rem_f - 1;
                let c_best = a.jv_cost[r_i][h];
                if c_best >= row_ub[r_i] {
                    continue;
                }
                let mut others = col_mask[h] & !(1u16 << r_i);
                let mut steal_slack = 0.0f32;
                let mut bound_ok = true;
                while others != 0 {
                    let r_j = others.trailing_zeros() as usize;
                    others &= others - 1;
                    let other_c = if (fin_hyps[r_j] & (1u64 << h)) != 0 {
                        a.jv_cost[r_j][h]
                    } else if (row_mask[r_j] & (1u64 << h)) != 0 {
                        a.eval_raw[r_j][h] - 0.1361f32
                    } else {
                        bound_ok = false;
                        break;
                    };
                    let diff = safe_priv_ub[r_j] - other_c;
                    if diff > steal_slack {
                        steal_slack = diff;
                    }
                }
                if bound_ok {
                    let tight_ub = c_best + steal_slack;
                    if tight_ub < row_ub[r_i] {
                        row_ub[r_i] = tight_ub;
                        changed = true;
                    }
                }
            }
        }
        for r_i in 0..nr {
            let ub = row_ub[r_i];
            let mut rem_f = fin_hyps[r_i];
            while rem_f != 0 {
                let hp = rem_f.trailing_zeros() as usize;
                rem_f &= rem_f - 1;
                if a.jv_cost[r_i][hp] > ub + 1e-5 {
                    fin_hyps[r_i] &= !(1u64 << hp);
                    a.jv_cost[r_i][hp] = 1e9;
                    if (col_mask[hp] & (1u16 << r_i)) != 0 {
                        col_mask[hp] &= !(1u16 << r_i);
                        changed = true;
                    }
                }
            }
            let mut rem = row_mask[r_i];
            while rem != 0 {
                let hp = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                if a.eval_raw[r_i][hp] - 0.1361f32 >= ub {
                    row_mask[r_i] &= !(1u64 << hp);
                    col_mask[hp] &= !(1u16 << r_i);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for r_i in 0..nr {
        while row_mask[r_i] != 0 {
            let mut rem = row_mask[r_i];
            let mut best_h = 0usize;
            let mut best_k = 1e9f32;
            while rem != 0 {
                let h = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                let k = a.eval_raw[r_i][h];
                if k < best_k {
                    best_k = k;
                    best_h = h;
                }
            }
            if best_k - 0.1361f32 >= row_ub[r_i] {
                let mut prune = row_mask[r_i];
                while prune != 0 {
                    let hp = prune.trailing_zeros() as usize;
                    prune &= prune - 1;
                    col_mask[hp] &= !(1u16 << r_i);
                }
                row_mask[r_i] = 0;
                break;
            }
            row_mask[r_i] &= !(1u64 << best_h);
            if (col_mask[best_h] & !(1u16 << r_i)) != 0 {
            }
            let ok_cand = eval_post_cand!(r_i, best_h, best_k, fin_hyps);
            if !ok_cand {
                let cm = col_mask[best_h];
                if cm.count_ones() == 1 {
                    let r_o = cm.trailing_zeros() as usize;
                    if (fin_hyps[r_o] & (1u64 << best_h)) != 0 {
                        let v = a.jv_cost[r_o][best_h];
                        if v < safe_priv_ub[r_o] {
                            safe_priv_ub[r_o] = v;
                        }
                        if v < row_ub[r_o] {
                            row_ub[r_o] = v;
                        }
                    }
                }
            } else if row_mask[r_i] != 0 && (fin_hyps[r_i] & (1u64 << best_h)) != 0 {
                let mut others = col_mask[best_h] & !(1u16 << r_i);
                if others != 0 {
                    let mut steal_slack = 0.0f32;
                    let mut bound_ok = true;
                    while others != 0 {
                        let r_j = others.trailing_zeros() as usize;
                        others &= others - 1;
                        let other_c = if (fin_hyps[r_j] & (1u64 << best_h)) != 0 {
                            a.jv_cost[r_j][best_h]
                        } else if (row_mask[r_j] & (1u64 << best_h)) != 0 {
                            a.eval_raw[r_j][best_h] - 0.1361f32
                        } else {
                            bound_ok = false;
                            break;
                        };
                        let diff = safe_priv_ub[r_j] - other_c;
                        if diff > steal_slack {
                            steal_slack = diff;
                        }
                    }
                    if bound_ok {
                        let tight_ub = a.jv_cost[r_i][best_h] + steal_slack;
                        if tight_ub < row_ub[r_i] {
                            row_ub[r_i] = tight_ub;
                        }
                    }
                }
            }
        }
    }

    let mut jv_out_cost = [1e9f32; MAX_TARGETS];
    let row_assign = if nr > 0 {
        solve_jv_assign_sparse_mut(nr, n_hyps, nc, &fin_hyps, &mut a.jv_cost, &mut jv_out_cost)
    } else {
        [usize::MAX; MAX_TARGETS]
    };

    // Extract sub-pixel positions via 2D Foreground-Weighted IC-LK & enforce post-match spatial NMS
    // `(mcx, mcy, total_cost, raw_cost, app_err, anc_l1, weca, sig, val, vis)`
    let mut matched = [(0.0f32, 0.0f32, 1e9f32, 1e9f32, 1e9f32, 1e9f32, 0.0f32, 0u32, 0u32, false); MAX_TARGETS];
    for r_i in 0..nr {
        let m = act_tracks[r_i];
        let col = row_assign[r_i];
        if col < n_hyps && a.eval_ok[r_i][col] {
            let btlx = a.eval_tlx[r_i][col];
            let btly = a.eval_tly[r_i][col];
            let (mut sub_dx, mut sub_dy) = (0.0f32, 0.0f32);
            if btlx >= 2
                && btlx + patch_dim + 1 < frame_w
                && btly >= 2
                && btly + patch_dim + 1 < frame_h
            {
                ensure_chroma!(a, btlx - 2, btly - 2, patch_dim + 4, patch_dim + 4, frame_w, frame_h);
                let tl_ptr = a.targets[m].tpl_luma.as_ptr();
                let trg_ptr = a.targets[m].tpl_rg.as_ptr();
                let tby_ptr = a.targets[m].tpl_by.as_ptr();
                let tw_ptr = a.targets[m].tpl_weight.as_ptr();
                let (dx_lk, dy_lk) = refine_subpixel_ic_lk_cached(
                    a.luma.as_ptr(),
                    a.chroma_rg.as_ptr(),
                    a.chroma_by.as_ptr(),
                    frame_w,
                    btlx,
                    btly,
                    tl_ptr,
                    trg_ptr,
                    tby_ptr,
                    tw_ptr,
                    scale_px,
                    &mut a.ic_lk_cache[m],
                    &mut a.targets[m].tpl_grad_dirty,
                );
                sub_dx = dx_lk;
                sub_dy = dy_lk;
            }
            let mcx = (btlx as f32) + half_patch + sub_dx;
            let mcy = (btly as f32) + half_patch + sub_dy;
            let csig = a.eval_sig[r_i][col];
            let cval = padic_ultrametric_val(a.targets[m].anchor_sig, csig);
            matched[m] = (
                mcx,
                mcy,
                jv_out_cost[r_i],
                a.eval_raw[r_i][col],
                a.eval_app[r_i][col],
                a.eval_anc[r_i][col],
                a.eval_eca[r_i][col],
                csig,
                cval,
                true,
            );
        }
    }

    #[inline(always)]
    fn is_strictly_senior(t_pm: &TargetTrack, pm: usize, t_m: &TargetTrack, m: usize) -> bool {
        if t_pm.confirmed != t_m.confirmed {
            return t_pm.confirmed > t_m.confirmed;
        }
        let id_pm = if t_pm.track_id > 0 { t_pm.track_id } else { u32::MAX };
        let id_m = if t_m.track_id > 0 { t_m.track_id } else { u32::MAX };
        if id_pm != id_m {
            return id_pm < id_m;
        }
        if t_pm.birth_frame != t_m.birth_frame {
            return t_pm.birth_frame < t_m.birth_frame;
        }
        pm < m
    }

    let mut order = [0usize; MAX_TARGETS];
    for i in 0..MAX_TARGETS {
        order[i] = i;
    }
    for i in 0..MAX_TARGETS {
        for j in (i + 1)..MAX_TARGETS {
            let ci = a.targets[order[i]].confirmed;
            let cj = a.targets[order[j]].confirmed;
            let swap = if ci != cj {
                cj > ci
            } else {
                matched[order[j]].2 < matched[order[i]].2
            };
            if swap {
                let tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    // Precompute unbiased current-frame velocity so newborn tracks (vel=0 at age 1..2) match
    // their host body's velocity before fast_projectile_confirm runs, while preserving filtered
    // velocity on established and coasting tracks.
    let mut obs_vx = [0.0f32; MAX_TARGETS];
    let mut obs_vy = [0.0f32; MAX_TARGETS];
    for k in 0..MAX_TARGETS {
        if a.targets[k].active == 0 || !matched[k].9 {
            continue;
        }
        let mcx_k = matched[k].0;
        let mcy_k = matched[k].1;
        if a.targets[k].coasting == 0 && a.targets[k].age > 0 {
            let ivx = mcx_k - a.targets[k].pos_x;
            let ivy = mcy_k - a.targets[k].pos_y;
            if a.targets[k].age <= 1 || a.targets[k].reemerged_last != 0 {
                obs_vx[k] = ivx;
                obs_vy[k] = ivy;
            } else if a.targets[k].age <= 3 {
                obs_vx[k] = 0.50 * ivx + 0.50 * a.targets[k].prev_inst_vx;
                obs_vy[k] = 0.50 * ivy + 0.50 * a.targets[k].prev_inst_vy;
            } else {
                obs_vx[k] = 0.58 * a.targets[k].vel_x + 0.42 * ivx;
                obs_vy[k] = 0.58 * a.targets[k].vel_y + 0.42 * ivy;
            }
        } else if a.targets[k].coasting > 0 {
            let ev2 = a.targets[k].entry_vx * a.targets[k].entry_vx
                + a.targets[k].entry_vy * a.targets[k].entry_vy;
            let dxe_k = mcx_k - a.targets[k].entry_x;
            let dye_k = mcy_k - a.targets[k].entry_y;
            let near_entry = dxe_k * dxe_k + dye_k * dye_k < (20.0 * s) * (20.0 * s);
            let v_ent_k = wasm_sqrtf(ev2);
            if a.lab_mode != 0 && a.targets[k].coasting >= 6 && v_ent_k >= 1.2 * s && near_entry {
                obs_vx[k] = -0.85 * a.targets[k].entry_vx;
                obs_vy[k] = -0.85 * a.targets[k].entry_vy;
            } else {
                let dt_c = (a.targets[k].coasting + 1) as f32;
                obs_vx[k] = (mcx_k - a.targets[k].entry_x) / dt_c;
                obs_vy[k] = (mcy_k - a.targets[k].entry_y) / dt_c;
            }
        }
    }

    let mut backtrack_budget = 3usize;
    let mut comp_cells_cnt = [0u16; 32];
    let grid_total = grid_w * grid_h;
    for gidx in 0..grid_total {
        let cid = a.cc_visited[gidx] as usize;
        if cid > 0 && cid < 32 {
            comp_cells_cnt[cid] = comp_cells_cnt[cid].saturating_add(1);
        }
    }
    let mut comp_track_cnt = [0u8; 32];
    let mode_b_crowded = a.lab_mode == 1 && nr >= 5;
    let co_dup_r2 = (22.5 * s) * (22.5 * s);
    let mode1_rod_r2 = if a.lab_mode == 1 {
        if mode_b_crowded {
            (34.0 * s) * (34.0 * s)
        } else {
            (30.0 * s) * (30.0 * s)
        }
    } else {
        co_dup_r2
    };
    let co_dup_vel_tol = 1.85 * s * gate_cadence;

    let mut conflict_dup_reap = [false; MAX_TARGETS];
    let mut conflict_dup_inc = [false; MAX_TARGETS];
    let mut order_rank = [0usize; MAX_TARGETS];
    let mut match_cost = [0.0f32; MAX_TARGETS];
    for idx in 0..MAX_TARGETS {
        order_rank[order[idx]] = idx;
        match_cost[idx] = matched[idx].2;
    }
    let sen_wins_pair = |pm: usize, m: usize, targets: &[TargetTrack; MAX_TARGETS]| -> (bool, bool) {
        if has_opposite_chroma_pole(targets[m].anchor_sig, targets[pm].anchor_sig) {
            return (false, false);
        }
        if !is_strictly_senior(&targets[pm], pm, &targets[m], m) {
            return (false, false);
        }
        let pm_speed = obs_vx[pm].abs() + obs_vy[pm].abs();
        let eff_vel_tol = if mode_b_crowded
            || (targets[pm].phase_valid != 0)
            || (a.lab_mode == 1 && pm_speed >= 1.5 * s)
        {
            3.2 * s * gate_cadence
        } else {
            co_dup_vel_tol
        };
        let vel_compat = (obs_vx[m] - obs_vx[pm]).abs() + (obs_vy[m] - obs_vy[pm]).abs() < eff_vel_tol;
        let rod_compat =
            (obs_vx[m] - obs_vx[pm]).abs() + (obs_vy[m] - obs_vy[pm]).abs() < 5.8 * s * gate_cadence;
        let harmonic_host = (targets[pm].phase_valid != 0 || targets[pm].pivot_valid != 0)
            && targets[pm].coasting <= 1
            && (safe_dt <= 0.062 || rod_compat);
        let wins = if targets[m].confirmed == 0 && targets[pm].confirmed != 0 {
            vel_compat || (harmonic_host && match_cost[pm] <= match_cost[m] + 0.22)
        } else if harmonic_host {
            match_cost[pm] <= match_cost[m] + 0.18
        } else {
            vel_compat && targets[pm].coasting == 0 && match_cost[pm] <= match_cost[m] + 0.055
        };
        (wins, vel_compat)
    };

    let mut undecided = [false; MAX_TARGETS];
    for m in 0..MAX_TARGETS {
        undecided[m] = a.targets[m].active != 0 && matched[m].9;
    }
    for _step in 0..MAX_TARGETS {
        let mut pick: Option<usize> = None;
        let mut fallback: Option<usize> = None;
        for idx in 0..MAX_TARGETS {
            let u = order[idx];
            if !undecided[u] {
                continue;
            }
            if fallback.is_none() {
                fallback = Some(u);
            }
            let mut beaten = false;
            for w in 0..MAX_TARGETS {
                if w == u || !undecided[w] {
                    continue;
                }
                let dx = matched[u].0 - matched[w].0;
                let dy = matched[u].1 - matched[w].1;
                if dx * dx + dy * dy < conflict_r2 {
                    let (w_sen, _) = sen_wins_pair(w, u, &a.targets);
                    let (u_sen, _) = sen_wins_pair(u, w, &a.targets);
                    if w_sen || (!u_sen && order_rank[w] < order_rank[u]) {
                        beaten = true;
                        break;
                    }
                }
            }
            if !beaten {
                pick = Some(u);
                break;
            }
        }
        let u = match pick.or(fallback) {
            Some(slot) => slot,
            None => break,
        };
        undecided[u] = false;
        for v in 0..MAX_TARGETS {
            if v == u || !undecided[v] {
                continue;
            }
            let dx = matched[v].0 - matched[u].0;
            let dy = matched[v].1 - matched[u].1;
            if dx * dx + dy * dy < conflict_r2 {
                undecided[v] = false;
                matched[v].9 = false;
                let (u_sen, vel_compat) = sen_wins_pair(u, v, &a.targets);
                if u_sen && vel_compat {
                    let mature_distinct = a.lab_mode == 0
                        && scale_px > 1
                        && a.targets[v].age >= 32
                        && a.targets[v].max_excursion2 >= (32.0 * s) * (32.0 * s);
                    if a.targets[v].confirmed == 0 {
                        conflict_dup_reap[v] = true;
                    } else if !mature_distinct {
                        a.targets[v].co_dup_streak = a.targets[v].co_dup_streak.saturating_add(1);
                        conflict_dup_inc[v] = true;
                        if a.targets[v].co_dup_streak >= 3 {
                            conflict_dup_reap[v] = true;
                        }
                    }
                }
            }
        }
    }

    for idx in 0..MAX_TARGETS {
        let m = order[idx];
        if a.targets[m].active == 0 {
            continue;
        }
        if conflict_dup_reap[m] {
            retire_target_slot(a, m);
            continue;
        }
        let (mcx, mcy, _mtotal, mrawcost, mapp_err, manc_l1, mweca, msig, mval, vis) = matched[m];
        let mut co_moving_dup = false;

        if vis {
            let gx = ((mcx as usize) / 16).min(grid_w - 1);
            let gy = ((mcy as usize) / 16).min(grid_h - 1);
            let mut track_cid = a.cc_visited[gy * grid_w + gx] as usize;

            for pm in 0..MAX_TARGETS {
                if pm == m || a.targets[pm].active == 0 || !matched[pm].9 {
                    continue;
                }
                if !is_strictly_senior(&a.targets[pm], pm, &a.targets[m], m) {
                    continue;
                }
                if has_opposite_chroma_pole(a.targets[m].anchor_sig, a.targets[pm].anchor_sig) {
                    continue;
                }
                let dx = mcx - matched[pm].0;
                let dy = mcy - matched[pm].1;
                let d2 = dx * dx + dy * dy;
                if d2 >= co_dup_r2 && d2 >= mode1_rod_r2 {
                    continue;
                }
                if a.lab_mode == 1
                    && (a.targets[m].anchor_sig & (1 << 31)) == 0
                    && (a.targets[pm].anchor_sig & (1 << 31)) == 0
                {
                    let rod_dx_max = if mode_b_crowded { 18.0 * s } else { 14.0 * s };
                    let rod_vx_min = if mode_b_crowded { 1.35 * s } else { 2.0 * s };
                    let mv2 = obs_vx[m] * obs_vx[m] + obs_vy[m] * obs_vy[m];
                    let pmv2 = obs_vx[pm] * obs_vx[pm] + obs_vy[pm] * obs_vy[pm];
                    let rod_v_min2 = rod_vx_min * rod_vx_min;
                    if d2 < mode1_rod_r2 && mv2 >= rod_v_min2 && pmv2 >= rod_v_min2 {
                        let v_dot = obs_vx[m] * obs_vx[pm] + obs_vy[m] * obs_vy[pm];
                        let v_prod = wasm_sqrtf(mv2 * pmv2);
                        if v_dot > 0.78 * v_prod {
                            let dup_rod = if scale_px > 1 {
                                let inv_pm = 1.0 / wasm_sqrtf(pmv2);
                                let par_sep = (dx * obs_vx[pm] + dy * obs_vy[pm]).abs() * inv_pm;
                                par_sep < rod_dx_max
                            } else {
                                dx.abs() < 10.5 * s && (obs_vy[m] - obs_vy[pm]).abs() < 2.2 * s
                            };
                            if dup_rod {
                                co_moving_dup = true;
                                break;
                            }
                        }
                    }
                }
                if d2 >= co_dup_r2 {
                    continue;
                }
                if d2 >= (12.0 * s) * (12.0 * s)
                    && classify_eca_connectivity(
                        &a.pyr_eca,
                        &a.luma,
                        &a.rgba_ring[a.active_rgba_slot as usize],
                        &a.bg_luma,
                        &a.bg_stat_streak,
                        frame_w,
                        frame_h,
                        s,
                        mcx,
                        mcy,
                        matched[pm].0,
                        matched[pm].1,
                    ) == BodyConnectivity::SeparatedByValley
                {
                    continue;
                }
                let pm_speed = obs_vx[pm].abs() + obs_vy[pm].abs();
                let eff_vel_tol = if mode_b_crowded || (a.lab_mode == 1 && pm_speed >= 1.5 * s) {
                    3.2 * s * gate_cadence
                } else {
                    co_dup_vel_tol
                };
                let vel_diff = (obs_vx[m] - obs_vx[pm]).abs() + (obs_vy[m] - obs_vy[pm]).abs();
                if vel_diff < eff_vel_tol {
                    co_moving_dup = true;
                    break;
                }
            }
            for p in 0..a.num_peaks {
                let dx = a.peaks[p].x - mcx;
                let dy = a.peaks[p].y - mcy;
                let pd2 = dx * dx + dy * dy;
                if pd2 < claim_r2 {
                    let sep_body = pd2 >= (12.0 * s) * (12.0 * s)
                        && classify_eca_connectivity(
                            &a.pyr_eca,
                            &a.luma,
                            &a.rgba_ring[a.active_rgba_slot as usize],
                            &a.bg_luma,
                            &a.bg_stat_streak,
                            frame_w,
                            frame_h,
                            s,
                            mcx,
                            mcy,
                            a.peaks[p].x,
                            a.peaks[p].y,
                        ) == BodyConnectivity::SeparatedByValley;
                    if !sep_body {
                        peak_claimed[p] = 1;
                        if track_cid == 0 {
                            track_cid = a.peaks[p].comp_id as usize;
                        }
                    }
                }
            }
            if track_cid > 0 && track_cid < 32 {
                comp_track_cnt[track_cid] = comp_track_cnt[track_cid].saturating_add(1);
            }

            if co_moving_dup {
                a.targets[m].co_dup_streak = a.targets[m].co_dup_streak.saturating_add(1);
            } else {
                a.targets[m].co_dup_streak = a.targets[m].co_dup_streak.saturating_sub(1);
            }

            let prev_coasting = a.targets[m].coasting;
            let imm_prev_vx = a.targets[m].prev_inst_vx;
            let imm_prev_vy = a.targets[m].prev_inst_vy;
            let inst_vx = if prev_coasting == 0 && a.targets[m].age > 0 {
                mcx - a.targets[m].pos_x
            } else {
                0.0
            };
            let inst_vy = if prev_coasting == 0 && a.targets[m].age > 0 {
                mcy - a.targets[m].pos_y
            } else {
                0.0
            };

            if prev_coasting > 0 {
                // Distinguish elastic in-pillar bounce back through entry vs forward chord/arc pass-through
                let ev2 = a.targets[m].entry_vx * a.targets[m].entry_vx
                    + a.targets[m].entry_vy * a.targets[m].entry_vy;
                let dxe_m = mcx - a.targets[m].entry_x;
                let dye_m = mcy - a.targets[m].entry_y;
                let near_entry = dxe_m * dxe_m + dye_m * dye_m < (20.0 * s) * (20.0 * s);
                let v_ent_m = wasm_sqrtf(ev2);
                if a.lab_mode != 0 && prev_coasting >= 6 && v_ent_m >= 1.2 * s && near_entry {
                    a.targets[m].vel_x = (-0.85 * a.targets[m].entry_vx).clamp(-v_clamp, v_clamp);
                    a.targets[m].vel_y = (-0.85 * a.targets[m].entry_vy).clamp(-v_clamp, v_clamp);
                    a.targets[m].ang_vel_rad_f = 0.0;
                } else {
                    let dt_c = (prev_coasting + 1) as f32;
                    let (_, _, evx_arc, evy_arc) = predict_curved_step(
                        a.targets[m].entry_x,
                        a.targets[m].entry_y,
                        a.targets[m].entry_vx,
                        a.targets[m].entry_vy,
                        a.targets[m].entry_omega_f,
                        a.targets[m].grav_ay_frame,
                        a.targets[m].is_ballistic,
                        dt_c,
                    );
                    let sec_vx = (mcx - a.targets[m].entry_x) / dt_c;
                    let sec_vy = (mcy - a.targets[m].entry_y) / dt_c;
                    if a.targets[m].is_ballistic != 0 || a.targets[m].entry_omega_f.abs() > 0.045 {
                        a.targets[m].vel_x = (0.55 * sec_vx + 0.45 * evx_arc).clamp(-v_clamp, v_clamp);
                        a.targets[m].vel_y = (0.55 * sec_vy + 0.45 * evy_arc).clamp(-v_clamp, v_clamp);
                    } else {
                        a.targets[m].vel_x = sec_vx.clamp(-v_clamp, v_clamp);
                        a.targets[m].vel_y = sec_vy.clamp(-v_clamp, v_clamp);
                    }
                    a.targets[m].ang_vel_rad_f *= 0.50;
                }
                a.targets[m].acc_x = 0.0;
                a.targets[m].acc_y = 0.0;
                a.targets[m].prev_inst_vx = a.targets[m].vel_x;
                a.targets[m].prev_inst_vy = a.targets[m].vel_y;
                a.targets[m].last_chord_dx = 0.0;
                a.targets[m].last_chord_dy = 0.0;
                a.targets[m].reemerged_last = prev_coasting.min(255) as u8;
                if prev_coasting >= 2 && a.targets[m].confirmed != 0 {
                    a.targets[m].reacq_count = a.targets[m].reacq_count.saturating_add(1);
                    a.coast_reacq_events = a.coast_reacq_events.saturating_add(1);
                }
            } else if a.targets[m].reemerged_last != 0 {
                a.targets[m].vel_x = inst_vx.clamp(-v_clamp, v_clamp);
                a.targets[m].vel_y = inst_vy.clamp(-v_clamp, v_clamp);
                a.targets[m].acc_x = 0.0;
                a.targets[m].acc_y = 0.0;
                a.targets[m].prev_inst_vx = a.targets[m].vel_x;
                a.targets[m].prev_inst_vy = a.targets[m].vel_y;
                a.targets[m].ang_vel_rad_f *= 0.65;
                a.targets[m].reemerged_last = 0;
            } else {
                if a.targets[m].age >= 2 {
                    let prev_vx = a.targets[m].prev_inst_vx;
                    let prev_vy = a.targets[m].prev_inst_vy;
                    let inst_ax = (inst_vx - prev_vx).clamp(-a_clamp, a_clamp);
                    let inst_ay = (inst_vy - prev_vy).clamp(-a_clamp, a_clamp);
                    a.targets[m].acc_x = 0.65 * a.targets[m].acc_x + 0.35 * inst_ax;
                    a.targets[m].acc_y = 0.65 * a.targets[m].acc_y + 0.35 * inst_ay;

                    let min_turn_v2 = (2.2 * s) * (2.2 * s);
                    if mapp_err < 0.13
                        && inst_vx * inst_vx + inst_vy * inst_vy >= min_turn_v2
                        && prev_vx * prev_vx + prev_vy * prev_vy >= min_turn_v2
                    {
                        let dth = wrap_pi(fast_atan2f(inst_vy, inst_vx) - fast_atan2f(prev_vy, prev_vx));
                        if dth.abs() >= 0.018 && dth.abs() < 0.35 {
                            let same_sign = (dth > 0.0 && a.targets[m].ang_vel_rad_f >= -0.004)
                                || (dth < 0.0 && a.targets[m].ang_vel_rad_f <= 0.004);
                            if same_sign {
                                a.targets[m].turn_streak = a.targets[m].turn_streak.saturating_add(1);
                            } else {
                                a.targets[m].turn_streak = 1;
                            }
                            a.targets[m].ang_vel_rad_f =
                                (0.74 * a.targets[m].ang_vel_rad_f + 0.26 * dth).clamp(-0.24, 0.24);
                        } else if dth.abs() >= 0.35 {
                            a.targets[m].ang_vel_rad_f = 0.0;
                            a.targets[m].turn_streak = 0;
                        } else {
                            a.targets[m].ang_vel_rad_f *= 0.80;
                            a.targets[m].turn_streak = a.targets[m].turn_streak.saturating_sub(1);
                        }
                    } else if mapp_err >= 0.13 {
                        // Freeze turn-rate update during partial occlusion edge-clipping
                    } else {
                        a.targets[m].ang_vel_rad_f *= 0.78;
                        a.targets[m].turn_streak = 0;
                    }
                }
                if a.targets[m].grav_arc_counted == 1 && a.targets[m].vel_y > 2.0 * s {
                    a.targets[m].grav_arc_counted = 2;
                }
                if (a.targets[m].grav_arc_counted == 2 && inst_vy < -1.5 * s)
                    || (a.targets[m].prev_inst_vy > 1.5 * s && inst_vy < -1.5 * s)
                {
                    a.targets[m].grav_arc_counted = 0;
                }
                a.targets[m].prev_inst_vx = inst_vx;
                a.targets[m].prev_inst_vy = inst_vy;
                a.targets[m].vel_x =
                    (0.58 * a.targets[m].vel_x + 0.42 * inst_vx).clamp(-v_clamp, v_clamp);
                a.targets[m].vel_y =
                    (0.58 * a.targets[m].vel_y + 0.42 * inst_vy).clamp(-v_clamp, v_clamp);
            }

            a.targets[m].pos_x = mcx;
            a.targets[m].pos_y = mcy;
            a.targets[m].curr_sig = msig;
            a.targets[m].last_padic_val = mval;
            a.targets[m].coasting = 0;
            a.targets[m].age += 1;
            a.targets[m].hit_mask_8 = (a.targets[m].hit_mask_8 << 1) | 1;
            a.targets[m].vis_eca_ema = 0.72 * a.targets[m].vis_eca_ema + 0.28 * mweca;
            a.targets[m].osc_mean_x = 0.97 * a.targets[m].osc_mean_x + 0.03 * mcx;
            a.targets[m].osc_mean_y = 0.97 * a.targets[m].osc_mean_y + 0.03 * mcy;

            let bdx = mcx - a.targets[m].birth_x;
            let bdy = mcy - a.targets[m].birth_y;
            let exc2 = bdx * bdx + bdy * bdy;
            if exc2 > a.targets[m].max_excursion2 {
                a.targets[m].max_excursion2 = exc2;
            }

            if a.targets[m].confirmed == 0 && !co_moving_dup {
                let hits = a.targets[m].hit_mask_8.count_ones();
                let exp_regime = ((a.cam_cut_hold >> 8) & 0xFF) > 0;
                let min_exc = if a.cam_pan_streak >= 1 {
                    if a.targets[m].vis_eca_ema >= 0.12 {
                        3.6 * s
                    } else {
                        4.5 * s
                    }
                } else if exp_regime {
                    2.0 * s
                } else {
                    1.6 * s
                };
                let pan_vel_ok = a.cam_pan_streak < 2
                    || (a.targets[m].vel_x * a.targets[m].vel_x
                        + a.targets[m].vel_y * a.targets[m].vel_y)
                        >= (0.95 * s) * (0.95 * s);
                let std_confirm = pan_vel_ok
                    && hits >= 3
                    && ((a.targets[m].max_excursion2 >= min_exc * min_exc
                        && a.targets[m].vis_eca_ema >= 0.024)
                        || (a.cam_pan_streak == 0
                            && !exp_regime
                            && a.targets[m].max_excursion2 >= (0.85 * s) * (0.85 * s)
                            && a.targets[m].vis_eca_ema >= 0.11));
                let fast_projectile_confirm = pan_vel_ok
                    && hits >= 2
                    && ((a.cam_pan_streak == 0
                        && !exp_regime
                        && a.targets[m].max_excursion2 >= (3.2 * s) * (3.2 * s))
                        || a.targets[m].max_excursion2 >= (5.2 * s) * (5.2 * s))
                    && a.targets[m].vis_eca_ema >= 0.068;
                let fast_dormant_di = if !std_confirm
                    && !fast_projectile_confirm
                    && pan_vel_ok
                    && hits >= 2
                    && a.targets[m].vis_eca_ema >= 0.032
                    && a.targets[m].track_id == 0
                {
                    find_best_dormant_match(a, m, f_idx, mcx, mcy, frame_w, frame_h, half_patch, s, true)
                } else {
                    None
                };
                let dense_swarm_hero_confirm = a.swarm_mode_cached != 0
                    && a.micro_swarm.swarm_active != 0
                    && a.micro_swarm.active_count >= 192
                    && a.cam_pan_streak < 2
                    && a.targets[m].age >= 4
                    && a.targets[m].max_excursion2 >= (0.50 * s) * (0.50 * s);
                if std_confirm || fast_projectile_confirm || fast_dormant_di.is_some() || dense_swarm_hero_confirm {
                    a.targets[m].confirmed = 1;
                    a.targets[m].just_confirmed = 1;
                    if a.targets[m].track_id == 0 {
                        // Check 32-slot 2-Adic + Foreground-Weighted Dormant Re-ID Graveyard before burning a new Track ID
                        let best_di = fast_dormant_di.or_else(|| {
                            find_best_dormant_match(a, m, f_idx, mcx, mcy, frame_w, frame_h, half_patch, s, false)
                        });
                        if let Some(di) = best_di {
                            let d = a.dormant[di];
                            a.targets[m].track_id = d.track_id;
                            a.targets[m].dormant_reid = 1;
                            a.targets[m].reacq_count = d.reacq_count.saturating_add(1);
                            a.targets[m].age = a.targets[m].age.saturating_add(d.age);
                            a.targets[m].orig_birth_frame = d.orig_birth_frame;
                            a.targets[m].sustained_10s_counted = d.sustained_10s_counted;
                            a.targets[m].sustained_60s_counted = d.sustained_60s_counted;
                            a.targets[m].anchor_sig = d.anchor_sig;
                            a.targets[m].anchor_luma = d.anchor_luma;
                            a.targets[m].anchor_rg = d.anchor_rg;
                            a.targets[m].anchor_by = d.anchor_by;
                            a.targets[m].tpl_weight = d.tpl_weight;
                            a.targets[m].tpl_grad_dirty = 1;
                            simd_ops::compute_anchor_luma_stats_simd128(
                                a.targets[m].anchor_luma.as_ptr(),
                                a.targets[m].tpl_weight.as_ptr(),
                                &mut a.targets[m].anchor_mu_a,
                                &mut a.targets[m].anchor_std_a,
                            );
                            a.targets[m].arc_raw_x = [0.0; HIST_RING_LEN];
                            a.targets[m].arc_raw_y = [0.0; HIST_RING_LEN];
                            a.targets[m].arc_vis = [0; HIST_RING_LEN];
                            a.targets[m].tremor_hist_x = [0.0; HIST_RING_LEN];
                            a.targets[m].tremor_hist_y = [0.0; HIST_RING_LEN];
                            a.targets[m].tremor_valid = [0; HIST_RING_LEN];
                            a.targets[m].fista_sharp_knots = 0;
                            a.targets[m].fista_knot_mask = 0;
                            a.targets[m].last_ode_disc_frame = 0;
                            a.targets[m].ode_cv_streak = 0;
                            a.imm[m].reset();
                            a.ode_states[m].reset();
                            a.pend_states[m].reset();
                            physics_id::reset_collision_slot(&mut a.collision_state, m);
                            a.dormant_reid_events = a.dormant_reid_events.saturating_add(1);
                            a.last_reid_gap_frames = f_idx.saturating_sub(d.lost_frame);
                            a.dormant[di].active = 0;
                        } else {
                            a.targets[m].track_id = a.next_track_id;
                            a.next_track_id += 1;
                        }
                    }
                }
            }

            imm_update_obs(
                &mut a.imm[m],
                mcx,
                mcy,
                inst_vx,
                inst_vy,
                imm_prev_vx,
                imm_prev_vy,
                &a.targets[m],
                safe_dt,
                s,
                true,
            );

            a.targets[m].ring_raw_x[slot_in] = mcx;
            a.targets[m].ring_raw_y[slot_in] = mcy;
            a.targets[m].ring_vx[slot_in] = a.targets[m].vel_x;
            a.targets[m].ring_vy[slot_in] = a.targets[m].vel_y;
            a.targets[m].ring_sig[slot_in] = msig;
            a.targets[m].ring_vis[slot_in] = 1;
            a.targets[m].arc_raw_x[arc_slot_in] = mcx;
            a.targets[m].arc_raw_y[arc_slot_in] = mcy;
            a.targets[m].arc_vis[arc_slot_in] = 1;

            // Non-causal backward ring re-tracking across `rgba_ring` on re-emergence or confirmation
            if backtrack_budget > 0 && a.targets[m].confirmed != 0 {
                if prev_coasting >= 1 {
                    let nb = (prev_coasting as usize).min(eff_k).min(RING_LEN - 2);
                    backtrack_gap_in_rgba_ring(a, m, f_idx, nb, frame_w, frame_h, scale_px, s, true);
                    backtrack_budget -= 1;
                } else if a.targets[m].just_confirmed != 0 {
                    let nb = (a.ring_cursor.min(eff_k as u64) as usize).min(RING_LEN - 2);
                    backtrack_gap_in_rgba_ring(a, m, f_idx, nb, frame_w, frame_h, scale_px, s, false);
                    backtrack_budget -= 1;
                }
            }

            // P-N Dual-Anchor Drift-Locked Template Update:
            // Only adapt `tpl_*` when confirmed, isolated, faithful to live template (`mapp_err < 0.135`, never occluders!),
            // allowing gradual perspective scale tracking in HD (`manc_l1 < 0.34`) while keeping `anchor_sig` immutable!
            let persp_fg = a.lab_mode == 0
                && scale_px > 1
                && (inst_vx * inst_vx + inst_vy * inst_vy) > (1.5 * s) * (1.5 * s)
                && mweca > 0.035;
            let tpl_gate = (mrawcost < 0.16
                && mapp_err < 0.16
                && manc_l1 < 0.20
                && mweca > 0.025
                && mval >= 8)
                || (persp_fg
                    && mrawcost < 0.165
                    && mapp_err < 0.135
                    && manc_l1 < 0.34
                    && mval >= 8);
            if a.targets[m].confirmed != 0 && tpl_gate {
                let mut isolated = true;
                for o in 0..MAX_TARGETS {
                    if o != m && a.targets[o].active != 0 {
                        let dx = a.targets[o].pos_x - mcx;
                        let dy = a.targets[o].pos_y - mcy;
                        if dx * dx + dy * dy < iso_r2 {
                            isolated = false;
                            break;
                        }
                    }
                }
                if isolated {
                    let tlx = ((mcx - half_patch).clamp(0.0, (frame_w - patch_dim) as f32)) as usize;
                    let tly = ((mcy - half_patch).clamp(0.0, (frame_h - patch_dim) as f32)) as usize;
                    let adapt_rate = if persp_fg && manc_l1 >= 0.18 {
                        0.035
                    } else {
                        0.025
                    };
                    ensure_chroma!(a, tlx, tly, patch_dim, patch_dim, frame_w, frame_h);
                    blend_template_scaled(
                        &a.luma,
                        &a.chroma_rg,
                        &a.chroma_by,
                        &a.eca_res,
                        frame_w,
                        frame_h,
                        scale_px,
                        tlx,
                        tly,
                        adapt_rate,
                        &mut a.targets[m],
                    );
                    if (f_idx & 3) == 0 {
                        apply_radial_deslip_weights(&mut a.targets[m], 0.25);
                    }
                }
            }

            // Static-Background Pin, Duplicate Fragment & Tentative Failure Reaper
            let cx_i = (mcx as usize).min(frame_w - 1);
            let cy_i = (mcy as usize).min(frame_h - 1);
            let c_pidx = cy_i * frame_w + cx_i;
            let center_stat = a.bg_eval_mask[c_pidx] != 0
                || (a.clutter_eval_ratio >= 0.14 && a.bg_stat_streak[c_pidx] >= 8);
            let step2 = inst_vx * inst_vx + inst_vy * inst_vy;
            let in_pan_or_exp = a.cam_pan_streak >= 2 || ((a.cam_cut_hold >> 8) & 0xFF) > 0;
            let vel_mag2_now =
                a.targets[m].vel_x * a.targets[m].vel_x + a.targets[m].vel_y * a.targets[m].vel_y;
            let osc_dx = mcx - a.targets[m].osc_mean_x;
            let osc_dy = mcy - a.targets[m].osc_mean_y;
            let osc_d2 = osc_dx * osc_dx + osc_dy * osc_dy;
            let pan_world_stat = in_pan_or_exp
                && prev_coasting == 0
                && step2 < (0.82 * s) * (0.82 * s)
                && vel_mag2_now < (0.80 * s) * (0.80 * s)
                && (a.targets[m].max_excursion2 < (14.0 * s) * (14.0 * s)
                    || osc_d2 < (3.4 * s) * (3.4 * s));
            if pan_world_stat
                || (prev_coasting == 0
                    && step2 < (0.38 * s) * (0.38 * s)
                    && (mweca < 0.011 || center_stat))
            {
                let inc = if pan_world_stat || (mweca < 0.006 && center_stat) { 2u8 } else { 1u8 };
                a.targets[m].static_streak = a.targets[m].static_streak.saturating_add(inc);
            } else if (!in_pan_or_exp && step2 > (0.68 * s) * (0.68 * s))
                || (step2 > (1.85 * s) * (1.85 * s) && vel_mag2_now > (1.35 * s) * (1.35 * s))
            {
                a.targets[m].static_streak = a.targets[m].static_streak.saturating_sub(2);
            }

            let active_oscillator = a.cam_pan_streak < 2
                && !in_pan_or_exp
                && a.targets[m].phase_valid != 0
                && a.targets[m].tremor_amp > 0.30 * s
                && a.targets[m].tremor_coherence > 0.30;
            let slot_age = f_idx.saturating_sub(a.targets[m].birth_frame).saturating_add(1);
            let dense_swarm_hero = a.swarm_mode_cached != 0
                && a.micro_swarm.swarm_active != 0
                && a.micro_swarm.active_count >= 192;
            let traveled_body = a.targets[m].dormant_reid != 0
                || a.targets[m].max_excursion2 >= (20.0 * s) * (20.0 * s);
            let max_stat_streak = if traveled_body { 135 } else { 18 };
            let unconfirmed_expired =
                a.targets[m].confirmed == 0 && slot_age >= if dense_swarm_hero { 10 } else { 7 };
            let mode_b_wake_pinned = a.lab_mode == 1
                && a.stimulus_armed == 0
                && ((a.targets[m].anchor_sig & (1 << 31)) == 0
                    || a.targets[m].max_excursion2 < (4.8 * s) * (4.8 * s))
                && a.targets[m].pivot_valid == 0
                && a.targets[m].freq_valid == 0
                && slot_age >= 18
                && a.targets[m].max_excursion2 < (9.5 * s) * (9.5 * s);
            let aperture_1d_bg = if a.lab_mode == 2
                && mweca < 0.020
                && !active_oscillator
            {
                let tlx_st = ((mcx - half_patch).clamp(0.0, (frame_w - patch_dim) as f32)) as usize;
                let tly_st = ((mcy - half_patch).clamp(0.0, (frame_h - patch_dim) as f32)) as usize;
                ensure_chroma!(
                    a,
                    tlx_st.saturating_sub(scale_px),
                    tly_st.saturating_sub(scale_px),
                    patch_dim + 2 * scale_px,
                    patch_dim + 2 * scale_px,
                    frame_w,
                    frame_h
                );
                let (st_lmin, st_lmax) = unsafe {
                    patch16x16_shi_tomasi_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        frame_w,
                        tlx_st,
                        tly_st,
                        scale_px,
                    )
                };
                st_lmin < 0.015 * st_lmax
            } else {
                false
            };
            let static_pinned = (!active_oscillator
                && (a.targets[m].static_streak >= max_stat_streak
                    || (in_pan_or_exp && a.targets[m].static_streak >= 14)
                    || (slot_age >= 18
                        && a.targets[m].max_excursion2 < (4.2 * s) * (4.2 * s))
                    || (a.cam_pan_streak >= 2
                        && ((slot_age >= 8 && a.targets[m].max_excursion2 < (5.5 * s) * (5.5 * s))
                            || (slot_age >= 11 && a.targets[m].max_excursion2 < (8.5 * s) * (8.5 * s))))))
                || aperture_1d_bg
                || mode_b_wake_pinned;
            let mature_distinct_dup = a.lab_mode == 0
                && scale_px > 1
                && a.targets[m].age >= 32
                && a.targets[m].max_excursion2 >= (32.0 * s) * (32.0 * s);
            let dup_redundant = (a.targets[m].confirmed == 0 && co_moving_dup)
                || (!mature_distinct_dup && a.targets[m].co_dup_streak >= 3);
            if unconfirmed_expired || static_pinned || dup_redundant {
                if static_pinned && traveled_body && !aperture_1d_bg && !dup_redundant && !unconfirmed_expired {
                    archive_dormant_track(a, m, f_idx, mcx, mcy, s);
                }
                retire_target_slot(a, m);
                continue;
            }

            // Internal self-audit for cross-target chromatic/2-adic signature inversion
            if a.targets[m].confirmed != 0 {
                let own_val = padic_ultrametric_val(a.targets[m].anchor_sig, msig);
                let mut other_best_slot = a.targets[m].owner_slot as usize;
                let mut other_best_val = own_val;
                let cand_chromatic = (msig & (1 << 31)) != 0;
                let min_swap_val = if cand_chromatic { 6 } else { 12 };
                for o in 0..MAX_TARGETS {
                    if o != m
                        && a.targets[o].active != 0
                        && a.targets[o].confirmed != 0
                        && (a.targets[m].proximity_mask & (1u16 << o)) != 0
                    {
                        let v = padic_ultrametric_val(a.targets[o].anchor_sig, msig);
                        if !has_opposite_chroma_pole(a.targets[o].anchor_sig, msig)
                            && ((has_opposite_chroma_pole(a.targets[m].anchor_sig, msig) && v >= 5)
                                || (v > other_best_val + 4 && v >= min_swap_val))
                        {
                            other_best_val = v;
                            other_best_slot = o;
                        }
                    }
                }
                if other_best_slot != (a.targets[m].owner_slot as usize) {
                    a.targets[m].mismatch_streak += 1;
                    if a.targets[m].mismatch_streak >= 2 {
                        a.padic_id_swaps += 1;
                        a.targets[m].owner_slot = other_best_slot as u8;
                        a.targets[m].mismatch_streak = 0;
                    }
                } else {
                    a.targets[m].mismatch_streak = 0;
                }
            }
        } else {
            if a.targets[m].coasting == 0 {
                a.targets[m].entry_x = a.targets[m].pos_x;
                a.targets[m].entry_y = a.targets[m].pos_y;
                a.targets[m].entry_vx = a.targets[m].vel_x;
                a.targets[m].entry_vy = a.targets[m].vel_y;
                a.targets[m].entry_ax = a.targets[m].acc_x;
                a.targets[m].entry_ay = a.targets[m].acc_y;
                a.targets[m].entry_omega_f = if a.targets[m].turn_streak >= 3
                    && !(a.targets[m].phase_valid != 0 && a.targets[m].tremor_coherence > 0.25)
                {
                    a.targets[m].ang_vel_rad_f
                } else {
                    0.0
                };
                a.targets[m].entry_frame = f_idx.saturating_sub(1);
                a.targets[m].has_entry = if a.targets[m].age > 0 { 1 } else { 0 };
                a.targets[m].reemerged_last = 0;
            }
            a.targets[m].coasting += 1;
            if a.lab_mode == 0 && scale_px > 1 && !conflict_dup_inc[m] {
                a.targets[m].co_dup_streak = a.targets[m].co_dup_streak.saturating_sub(1);
            }
            if a.targets[m].coasting >= 3 {
                a.targets[m].grav_arc_counted = 0;
            }
            a.targets[m].age += 1;
            a.targets[m].hit_mask_8 <<= 1;

            let cf = a.targets[m].coasting as f32;
            let (raw_coast_x, raw_coast_y, coast_vx, coast_vy) = predict_curved_step(
                a.targets[m].entry_x,
                a.targets[m].entry_y,
                a.targets[m].entry_vx,
                a.targets[m].entry_vy,
                a.targets[m].entry_omega_f,
                a.targets[m].grav_ay_frame,
                a.targets[m].is_ballistic,
                cf,
            );
            let exited_viewport = raw_coast_x < half_patch - 3.0 * s
                || raw_coast_x > (frame_w as f32) - half_patch + 3.0 * s
                || raw_coast_y < half_patch - 3.0 * s
                || raw_coast_y > (frame_h as f32) - half_patch + 3.0 * s
                || (a.lab_mode == 0
                    && is_border_clamped_exit(
                        a.targets[m].entry_x,
                        a.targets[m].entry_y,
                        a.targets[m].entry_vx,
                        a.targets[m].entry_vy,
                        frame_w,
                        frame_h,
                        half_patch,
                        s,
                    ));
            let coast_x = raw_coast_x.clamp(half_patch, (frame_w as f32) - half_patch);
            let coast_y = raw_coast_y.clamp(half_patch, (frame_h as f32) - half_patch);

            a.targets[m].ring_raw_x[slot_in] = coast_x;
            a.targets[m].ring_raw_y[slot_in] = coast_y;
            a.targets[m].ring_vx[slot_in] = coast_vx;
            a.targets[m].ring_vy[slot_in] = coast_vy;
            a.targets[m].ring_sig[slot_in] = a.targets[m].curr_sig;
            a.targets[m].ring_vis[slot_in] = 0;
            a.targets[m].arc_vis[arc_slot_in] = 0;

            let traveled_coast = a.targets[m].max_excursion2 >= (18.0 * s) * (18.0 * s);
            let dense_swarm_coast = a.swarm_mode_cached != 0
                && a.micro_swarm.swarm_active != 0
                && a.micro_swarm.active_count >= 192;
            let max_coast = if a.targets[m].confirmed == 0 {
                if dense_swarm_coast { 4 } else { 2 }
            } else if exited_viewport {
                (eff_k as u32 + 2).max(4)
            } else if a.targets[m].age < 10 {
                16
            } else if traveled_coast {
                90
            } else if a.targets[m].max_excursion2 < (5.0 * s) * (5.0 * s)
                && f_idx.saturating_sub(a.targets[m].birth_frame) >= 10
            {
                4
            } else {
                64
            };
            if a.targets[m].coasting >= max_coast
                || (a.targets[m].confirmed == 0
                    && a.targets[m].age >= if dense_swarm_coast { 10 } else { 6 })
            {
                if !exited_viewport && a.targets[m].confirmed != 0 {
                    archive_dormant_track(a, m, f_idx, coast_x, coast_y, s);
                }
                retire_target_slot(a, m);
            } else {
                imm_update_obs(
                    &mut a.imm[m],
                    coast_x,
                    coast_y,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    &a.targets[m],
                    safe_dt,
                    s,
                    false,
                );
            }
        }
    }

    // ------------------------------------------------------------------------
    // 4. Kinematic-Only Constant-Velocity 2x2 Covariance Kalman Baseline
    // ------------------------------------------------------------------------
    let mut kal_peak_used = [0u8; MAX_TARGETS];
    let kal_margin = 6.0 * s;
    let kal_gate_d2 = (64.0 * s) * (64.0 * s);
    for m in 0..MAX_TARGETS {
        if a.kalman[m].active == 0 {
            continue;
        }
        let dt_f = dt_frames_60;
        let pred_x = (a.kalman[m].x + a.kalman[m].vx * dt_f).clamp(kal_margin, (frame_w as f32) - kal_margin);
        let pred_y = (a.kalman[m].y + a.kalman[m].vy * dt_f).clamp(kal_margin, (frame_h as f32) - kal_margin);

        // Exact P^- = F P F^T + Q for F = [[1, dt_f], [0, 1]]
        let p_xx_m = a.kalman[m].p_xx
            + 2.0 * dt_f * a.kalman[m].p_xv
            + dt_f * dt_f * a.kalman[m].p_vv
            + 0.25 * s2 * dt_f;
        let p_xv_m = a.kalman[m].p_xv + dt_f * a.kalman[m].p_vv;
        let p_vv_m = a.kalman[m].p_vv + 0.04 * s2 * dt_f;

        let mut best_p: Option<usize> = None;
        let mut best_d2 = kal_gate_d2;
        for p in 0..a.num_peaks {
            if kal_peak_used[p] != 0 {
                continue;
            }
            let dx = a.peaks[p].x - pred_x;
            let dy = a.peaks[p].y - pred_y;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best_p = Some(p);
            }
        }

        if let Some(p) = best_p {
            kal_peak_used[p] = 1;
            let zx = a.peaks[p].x;
            let zy = a.peaks[p].y;
            let e_conf = (a.peaks[p].energy * (1.0 / (52.0 * s2))).clamp(0.12, 1.0);
            let e_conf2 = e_conf * e_conf;
            let r_meas = (4.0 * s2) / (e_conf2 * e_conf2);
            let s_innov = p_xx_m + r_meas;
            let k_x = p_xx_m / s_innov;
            let k_v = p_xv_m / s_innov;

            let innov_x = zx - pred_x;
            let innov_y = zy - pred_y;
            a.kalman[m].x = pred_x + k_x * innov_x;
            a.kalman[m].y = pred_y + k_x * innov_y;
            a.kalman[m].vx = (a.kalman[m].vx + k_v * innov_x).clamp(-v_clamp, v_clamp);
            a.kalman[m].vy = (a.kalman[m].vy + k_v * innov_y).clamp(-v_clamp, v_clamp);
            a.kalman[m].p_xx = ((1.0 - k_x) * p_xx_m).max(1e-3 * s2);
            a.kalman[m].p_xv = (1.0 - k_x) * p_xv_m;
            a.kalman[m].p_vv = (p_vv_m - k_v * p_xv_m).max(1e-4 * s2);
            a.kalman[m].coasting = 0;

            let ksig = a.peaks[p].sig;
            let cur_owner = a.kalman[m].current_owner as usize;
            let cur_owner_sig = if cur_owner < MAX_TARGETS && a.targets[cur_owner].active != 0 {
                a.targets[cur_owner].anchor_sig
            } else {
                a.kalman[m].anchor_sig
            };
            let own_val = padic_ultrametric_val(cur_owner_sig, ksig);

            let mut hijacked_by = cur_owner;
            let mut best_other_val = own_val;
            for o in 0..MAX_TARGETS {
                if o != cur_owner && a.targets[o].active != 0 {
                    let osig = a.targets[o].anchor_sig;
                    let v = padic_ultrametric_val(osig, ksig);
                    if !has_opposite_chroma_pole(osig, ksig)
                        && (has_opposite_chroma_pole(cur_owner_sig, ksig) || v > best_other_val + 2)
                        && v >= 2
                    {
                        best_other_val = v;
                        hijacked_by = o;
                    }
                }
            }
            if hijacked_by != cur_owner {
                a.kalman[m].mismatch_streak += 1;
                if a.kalman[m].mismatch_streak >= 2 {
                    a.kalman_id_swaps += 1;
                    a.kalman[m].current_owner = hijacked_by as u8;
                    a.kalman[m].mismatch_streak = 0;
                }
            } else {
                a.kalman[m].mismatch_streak = 0;
            }
        } else {
            a.kalman[m].x = pred_x;
            a.kalman[m].y = pred_y;
            a.kalman[m].p_xx = p_xx_m.min(400.0 * s2);
            a.kalman[m].p_xv = p_xv_m.clamp(-80.0 * s2, 80.0 * s2);
            a.kalman[m].p_vv = p_vv_m.min(64.0 * s2);
            a.kalman[m].coasting += 1;
            if a.kalman[m].coasting > 20 {
                a.kalman[m].vx *= 0.80;
                a.kalman[m].vy *= 0.80;
            }
            if a.kalman[m].coasting > 28 && a.targets[m].active != 0 && a.targets[m].coasting == 0 {
                a.kalman[m].x = a.targets[m].pos_x;
                a.kalman[m].y = a.targets[m].pos_y;
                a.kalman[m].vx = a.targets[m].vel_x;
                a.kalman[m].vy = a.targets[m].vel_y;
                a.kalman[m].coasting = 0;
            }
        }

        a.kalman[m].ring_x[slot_in] = a.kalman[m].x;
        a.kalman[m].ring_y[slot_in] = a.kalman[m].y;
    }

    // ------------------------------------------------------------------------
    // 5. Auto-Seed New Moving Physical Targets from Unclaimed ECA Peaks
    //    Gated by 2D Shi-Tomasi Structure-Tensor Cornerness (eliminates 1D forearm/wall edges)
    //    and active during ego-motion pans via interior border & energy guards.
    // ------------------------------------------------------------------------
    let active_pan_seed = a.cam_pan_streak >= 1 || a.last_pan_dx != 0 || a.last_pan_dy != 0;
    let dense_hero_cap = dense_swarm_hero_cap_active(a);
    let eff_hero_cap = if dense_hero_cap {
        DENSE_SWARM_HERO_CAP
    } else {
        MAX_TARGETS
    };
    let mut act_hero_cnt = 0usize;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            act_hero_cnt += 1;
        }
    }
    if f_idx >= 4
        && (!dense_hero_cap || act_hero_cnt < eff_hero_cap)
        && (a.lab_mode == 2
            || a.last_discont_frame == 0
            || f_idx >= a.last_discont_frame.saturating_add(3))
    {
        let seed_close_r2 = if a.lab_mode == 2 {
            (10.0 * s) * (10.0 * s)
        } else {
            (19.0 * s) * (19.0 * s)
        };
        let seed_chroma_r2 = if a.lab_mode == 2 {
            (13.5 * s) * (13.5 * s)
        } else {
            (28.0 * s) * (28.0 * s)
        };
        let seed_neutral_r2 = if a.lab_mode == 1 {
            (44.0 * s) * (44.0 * s)
        } else if a.lab_mode == 2 {
            (12.0 * s) * (12.0 * s)
        } else {
            (27.5 * s) * (27.5 * s)
        };
        let large_comp_sep_r2 = (34.0 * s) * (34.0 * s);
        let coast_guard_r2 = (54.0 * s) * (54.0 * s);
        let single_obj_cells = (48 * scale_px * scale_px) as u16;
        let min_patch_eca = 144.0 * s2 * 0.054;
        let min_inner6_eca = 36.0 * s2 * 0.080;
        let min_peak_energy = 3.2 * s * wasm_sqrtf(s);
        let eff_min_peak_energy = if active_pan_seed {
            1.28 * min_peak_energy
        } else {
            min_peak_energy
        };
        let st_min_tlx = 3 * s_i32;
        let st_max_tlx = (frame_w - 19 * scale_px) as i32;
        let st_min_tly = 3 * s_i32;
        let st_max_tly = (frame_h - 19 * scale_px) as i32;
        let snap_step = 4 * s_i32;
        let mut births_this_frame = 0usize;

        let pan_margin_l = if a.last_pan_dx > 0 { 18.0 * s } else { 9.0 * s };
        let pan_margin_r = if a.last_pan_dx < 0 { 18.0 * s } else { 9.0 * s };
        let pan_margin_t = if a.last_pan_dy > 0 { 14.0 * s } else { 8.0 * s };
        let pan_margin_b = if a.last_pan_dy < 0 {
            if a.lab_mode == 2 { 7.5 * s } else { 14.0 * s }
        } else {
            8.0 * s
        };
        let allow_sd_base = !active_pan_seed
            && a.lab_mode == 0
            && scale_px == 1
            && a.clutter_db_valid != 0
            && a.clutter_db >= 38.0;
        let quiet_sd_base = allow_sd_base && a.clutter_db >= 41.0;

        for p in 0..a.num_peaks {
            if births_this_frame >= 2
                || (dense_hero_cap && act_hero_cnt >= eff_hero_cap)
                || peak_claimed[p] != 0
                || (a.peaks[p].energy < eff_min_peak_energy
                    && !(allow_sd_base && a.peaks[p].energy >= if quiet_sd_base { 1.25 } else { 1.65 }))
            {
                continue;
            }
            let raw_px = a.peaks[p].x;
            let raw_py = a.peaks[p].y;
            if active_pan_seed
                && (raw_px < pan_margin_l
                    || raw_px > (frame_w as f32) - pan_margin_r
                    || raw_py < pan_margin_t
                    || raw_py > (frame_h as f32) - pan_margin_b)
            {
                continue;
            }
            let psig = a.peaks[p].sig;
            let p_comp = a.peaks[p].comp_id;
            let peak_chromatic = (psig & (1 << 31)) != 0;
            let min_neu_pan_e = 1.65;
            if active_pan_seed && a.cam_pan_streak >= 2 && !peak_chromatic && a.peaks[p].energy < min_neu_pan_e * min_peak_energy {
                continue;
            }
            let comp_sz = if (p_comp as usize) > 0 && (p_comp as usize) < 32 {
                comp_cells_cnt[p_comp as usize]
            } else {
                999
            };
            let quiet_scene_hd = !active_pan_seed
                && a.lab_mode == 0
                && scale_px > 1
                && a.clutter_db_valid != 0
                && a.clutter_db >= 48.0
                && a.clutter_eval_ratio >= 0.85;
            let eca_norm_denom = if a.lab_mode == 0 && scale_px > 1 && comp_sz <= 14 {
                144.0 * s
            } else {
                144.0 * s2
            };

            // Check centered candidate first; only search local 3x3 (`±4*s`) when center lacks
            // strong isotropic 2D cornerness (e.g. peak centroid fell on a 1D edge)
            let base_tlx = ((raw_px - half_patch + 0.5) as i32).clamp(st_min_tlx, st_max_tlx);
            let base_tly = ((raw_py - half_patch + 0.5) as i32).clamp(st_min_tly, st_max_tly);
            let mut best_tlx = base_tlx as usize;
            let mut best_tly = base_tly as usize;
            let mut best_st_score = -1.0f32;
            let mut best_patch_eca = 0.0f32;
            let mut best_inner_eca = 0.0f32;
            let mut best_outer_eca = 0.0f32;

            let (center_c_eca, center_inner_eca, center_max_eca) = patch_eca_core_inner_max_simd128(
                a.eca_res.as_ptr(),
                frame_w,
                base_tlx as usize,
                base_tly as usize,
                scale_px,
            );
            let center_isolum_chroma = peak_chromatic
                && a.peaks[p].energy >= 2.2 * min_peak_energy
                && center_c_eca >= 0.16 * min_patch_eca;
            let center_compact_hd = !active_pan_seed
                && a.lab_mode == 0
                && scale_px > 1
                && comp_sz <= 10
                && a.peaks[p].energy >= 16.0
                && center_c_eca >= 12.5
                && center_max_eca >= 0.30;
            let center_fog_hd = quiet_scene_hd
                && comp_sz <= 14
                && raw_py < (frame_h as f32) - 20.0 * s
                && a.peaks[p].energy >= 6.5
                && center_c_eca >= 5.5
                && center_max_eca >= 0.12;
            let center_compact_sd = allow_sd_base
                && comp_sz <= if quiet_sd_base { 6 } else { 4 }
                && raw_py > 18.0
                && raw_py < (frame_h as f32) - 18.0
                && a.peaks[p].energy >= if quiet_sd_base { 1.25 } else { 1.65 }
                && (center_inner_eca >= if quiet_sd_base { 0.85 } else { 1.35 }
                    || (quiet_sd_base && center_c_eca >= 2.1))
                && center_max_eca >= if quiet_sd_base { 0.065 } else { 0.09 };
            let center_eca_pass = if a.peaks[p].energy < eff_min_peak_energy {
                center_compact_sd
            } else {
                center_c_eca >= min_patch_eca
                    || (center_inner_eca >= min_inner6_eca && center_max_eca >= 0.16)
                    || center_isolum_chroma
                    || center_compact_hd
                    || center_fog_hd
                    || center_compact_sd
            };
            let mut center_lam_min = -1.0f32;
            let mut center_lam_max = 0.0f32;

            {
                let qx = base_tlx as usize;
                let qy = base_tly as usize;
                if center_eca_pass {
                    ensure_chroma!(
                        a,
                        qx.saturating_sub(scale_px),
                        qy.saturating_sub(scale_px),
                        patch_dim + 2 * scale_px,
                        patch_dim + 2 * scale_px,
                        frame_w,
                        frame_h
                    );
                    let (lam_min, lam_max) = patch16x16_shi_tomasi_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        frame_w,
                        qx,
                        qy,
                        scale_px,
                    );
                    center_lam_min = lam_min;
                    center_lam_max = lam_max;
                    let min_center_lam = if center_fog_hd {
                        0.00018
                    } else if center_compact_sd {
                        if quiet_sd_base { 0.00028 } else { 0.00040 }
                    } else {
                        0.00050
                    };
                    if lam_min >= min_center_lam && lam_min >= 0.20 * lam_max {
                        best_st_score = lam_min;
                        best_patch_eca = center_c_eca.max(center_inner_eca * 3.0);
                        best_inner_eca = center_inner_eca;
                        best_outer_eca = (center_c_eca - center_inner_eca).max(0.0);
                    }
                }
            }

            if best_st_score <= 0.0 {
                for dy in [-snap_step, 0, snap_step] {
                    for dx in [-snap_step, 0, snap_step] {
                        let qx = (base_tlx + dx).clamp(st_min_tlx, st_max_tlx) as usize;
                        let qy = (base_tly + dy).clamp(st_min_tly, st_max_tly) as usize;
                        let is_center_coord =
                            qx == (base_tlx as usize) && qy == (base_tly as usize);

                        let (p_eca, inner_eca, fog_hd_pass, compact_sd_pass, lam_min, lam_max) =
                            if is_center_coord {
                                if !center_eca_pass {
                                    continue;
                                }
                                (
                                    center_c_eca,
                                    center_inner_eca,
                                    center_fog_hd,
                                    center_compact_sd,
                                    center_lam_min,
                                    center_lam_max,
                                )
                            } else {
                                let (p_eca, inner_eca, max_eca) = patch_eca_core_inner_max_simd128(
                                    a.eca_res.as_ptr(),
                                    frame_w,
                                    qx,
                                    qy,
                                    scale_px,
                                );
                                let isolum_chroma_pass = peak_chromatic
                                    && a.peaks[p].energy >= 2.2 * min_peak_energy
                                    && p_eca >= 0.16 * min_patch_eca;
                                let compact_hd_pass = !active_pan_seed
                                    && a.lab_mode == 0
                                    && scale_px > 1
                                    && comp_sz <= 10
                                    && a.peaks[p].energy >= 16.0
                                    && p_eca >= 12.5
                                    && max_eca >= 0.30;
                                let fog_hd_pass = quiet_scene_hd
                                    && comp_sz <= 14
                                    && raw_py < (frame_h as f32) - 20.0 * s
                                    && a.peaks[p].energy >= 6.5
                                    && p_eca >= 5.5
                                    && max_eca >= 0.12;
                                let compact_sd_pass = allow_sd_base
                                    && comp_sz <= if quiet_sd_base { 6 } else { 4 }
                                    && raw_py > 18.0
                                    && raw_py < (frame_h as f32) - 18.0
                                    && a.peaks[p].energy >= if quiet_sd_base { 1.25 } else { 1.65 }
                                    && (inner_eca >= if quiet_sd_base { 0.85 } else { 1.35 }
                                        || (quiet_sd_base && p_eca >= 2.1))
                                    && max_eca >= if quiet_sd_base { 0.065 } else { 0.09 };
                                let eca_pass = if a.peaks[p].energy < eff_min_peak_energy {
                                    compact_sd_pass
                                } else {
                                    p_eca >= min_patch_eca
                                        || (inner_eca >= min_inner6_eca && max_eca >= 0.16)
                                        || isolum_chroma_pass
                                        || compact_hd_pass
                                        || fog_hd_pass
                                        || compact_sd_pass
                                };
                                if !eca_pass {
                                    continue;
                                }
                                ensure_chroma!(
                                    a,
                                    qx.saturating_sub(scale_px),
                                    qy.saturating_sub(scale_px),
                                    patch_dim + 2 * scale_px,
                                    patch_dim + 2 * scale_px,
                                    frame_w,
                                    frame_h
                                );
                                let (lam_min, lam_max) = patch16x16_shi_tomasi_simd128(
                                    a.luma.as_ptr(),
                                    a.chroma_rg.as_ptr(),
                                    a.chroma_by.as_ptr(),
                                    frame_w,
                                    qx,
                                    qy,
                                    scale_px,
                                );
                                (p_eca, inner_eca, fog_hd_pass, compact_sd_pass, lam_min, lam_max)
                            };
                        let eff_eca = p_eca.max(inner_eca * 3.0);

                        // Require genuine 2D corner/blob structure (reject flat blocks & 1D straight edges)
                        let min_snap_lam = if fog_hd_pass {
                            0.00010
                        } else if compact_sd_pass && quiet_sd_base {
                            0.00016
                        } else {
                            0.00022
                        };
                        if lam_min < min_snap_lam || lam_min < 0.072 * lam_max {
                            continue;
                        }
                        let iso_ratio = (lam_min / lam_max.max(1e-7)).min(0.55);
                        let center_bias = if dx == 0 && dy == 0 { 1.15 } else { 1.0 };
                        let score = lam_min * iso_ratio * (0.4 + eff_eca / eca_norm_denom) * center_bias;
                        if score > best_st_score {
                            best_st_score = score;
                            best_tlx = qx;
                            best_tly = qy;
                            best_patch_eca = eff_eca;
                            best_inner_eca = inner_eca;
                            best_outer_eca = (p_eca - inner_eca).max(0.0);
                        }
                    }
                }
            }

            if best_st_score <= 0.0 {
                continue;
            }

            let px = (best_tlx as f32) + half_patch;
            let py = (best_tly as f32) + half_patch;
            let pgx = ((px as usize) / 16).min(grid_w - 1);
            let pgy = ((py as usize) / 16).min(grid_h - 1);
            let cid = {
                let c_grid = a.cc_visited[pgy * grid_w + pgx] as usize;
                if c_grid > 0 && c_grid < 32 {
                    c_grid
                } else {
                    (p_comp as usize).min(31)
                }
            };
            // Crisp small projectile splitting off a large >28-cell human torso/arm component
            let large_comp = cid > 0 && comp_cells_cnt[cid] > single_obj_cells;
            let compact_projectile = large_comp
                && best_inner_eca >= min_inner6_eca
                && best_inner_eca * 1.65 >= best_outer_eca;

            let mut too_close = false;
            let mut has_same_pole_in_comp = false;
            for m in 0..MAX_TARGETS {
                if a.targets[m].active != 0 {
                    let m_exited_border = if a.targets[m].coasting > 0 {
                        let cf_m = (a.targets[m].coasting as f32).min(3.0);
                        let mex_x = a.targets[m].entry_x + cf_m * a.targets[m].entry_vx;
                        let mex_y = a.targets[m].entry_y + cf_m * a.targets[m].entry_vy;
                        mex_x < half_patch - 3.0 * s
                            || mex_x > (frame_w as f32) - half_patch + 3.0 * s
                            || mex_y < half_patch - 3.0 * s
                            || mex_y > (frame_h as f32) - half_patch + 3.0 * s
                            || (a.lab_mode == 0
                                && is_border_clamped_exit(
                                    a.targets[m].entry_x,
                                    a.targets[m].entry_y,
                                    a.targets[m].entry_vx,
                                    a.targets[m].entry_vy,
                                    frame_w,
                                    frame_h,
                                    half_patch,
                                    s,
                                ))
                    } else {
                        false
                    };
                    if a.targets[m].coasting >= 2 && m_exited_border {
                        continue;
                    }
                    let dx = a.targets[m].pos_x - px;
                    let dy = a.targets[m].pos_y - py;
                    let dist2 = dx * dx + dy * dy;
                    let min_seed_d = if a.lab_mode == 2 {
                        8.5 * s
                    } else {
                        12.0 * s
                    };
                    if dist2 < min_seed_d * min_seed_d {
                        too_close = true;
                        break;
                    }
                    let opp_chroma = has_opposite_chroma_pole(a.targets[m].anchor_sig, psig);
                    let conn = if a.targets[m].coasting == 0 && dist2 < (26.5 * s) * (26.5 * s) {
                        classify_eca_connectivity(
                            &a.pyr_eca,
                            &a.luma,
                            &a.rgba_ring[a.active_rgba_slot as usize],
                            &a.bg_luma,
                            &a.bg_stat_streak,
                            frame_w,
                            frame_h,
                            s,
                            a.targets[m].pos_x,
                            a.targets[m].pos_y,
                            px,
                            py,
                        )
                    } else {
                        BodyConnectivity::Unsplit
                    };
                    if conn != BodyConnectivity::SeparatedByValley {
                        if dist2 < seed_close_r2 {
                            too_close = true;
                            break;
                        }
                        if cid > 0 && !opp_chroma {
                            let tgx = ((a.targets[m].pos_x as usize) / 16).min(grid_w - 1);
                            let tgy = ((a.targets[m].pos_y as usize) / 16).min(grid_h - 1);
                            if a.cc_visited[tgy * grid_w + tgx] as usize == cid {
                                has_same_pole_in_comp = true;
                                if !compact_projectile && a.lab_mode != 2 && dist2 < large_comp_sep_r2 {
                                    too_close = true;
                                    break;
                                }
                            }
                        }
                        let val = padic_ultrametric_val(a.targets[m].anchor_sig, psig);
                        let same_chroma = peak_chromatic && !opp_chroma && val >= 12;
                        if same_chroma && dist2 < seed_chroma_r2 {
                            too_close = true;
                            break;
                        }
                        let same_neutral_body = !peak_chromatic
                            && ((a.targets[m].anchor_sig & (1 << 31)) == 0)
                            && val >= 8;
                        if same_neutral_body && dist2 < seed_neutral_r2 {
                            too_close = true;
                            break;
                        }
                    }
                    let val = padic_ultrametric_val(a.targets[m].anchor_sig, psig);
                    if a.targets[m].coasting > 0 && a.targets[m].coasting <= 56 {
                        if !m_exited_border {
                            let edx = a.targets[m].entry_x - px;
                            let edy = a.targets[m].entry_y - py;
                            let ed2 = edx * edx + edy * edy;
                            let pdx = a.targets[m].pred_cx - px;
                            let pdy = a.targets[m].pred_cy - py;
                            let pd2 = pdx * pdx + pdy * pdy;
                            let pred_guard = !opp_chroma
                                && val >= 8
                                && pd2 < (34.0 * s) * (34.0 * s);
                            let ev2 = a.targets[m].entry_vx * a.targets[m].entry_vx
                                + a.targets[m].entry_vy * a.targets[m].entry_vy;
                            let corridor_gantry_guard = if a.lab_mode == 0
                                && !opp_chroma
                                && a.targets[m].is_ballistic == 0
                                && a.scene_gravity_arcs == 0
                                && a.targets[m].coasting <= 22
                                && ev2 > (0.9 * s) * (0.9 * s)
                            {
                                let inv_ev = 1.0 / wasm_sqrtf(ev2);
                                let ux = a.targets[m].entry_vx * inv_ev;
                                let uy = a.targets[m].entry_vy * inv_ev;
                                let along = (edx * ux + edy * uy).abs();
                                let cross = (-edx * uy + edy * ux).abs();
                                along < 68.0 * s && cross < 20.0 * s
                            } else {
                                false
                            };
                            if (a.targets[m].coasting <= 28 && !opp_chroma && ed2 < coast_guard_r2)
                                || corridor_gantry_guard
                                || pred_guard
                            {
                                too_close = true;
                                break;
                            }
                        }
                    }
                }
            }
            if !too_close && cid > 0 && !compact_projectile && has_same_pole_in_comp {
                let max_in_comp = if a.lab_mode == 1 && !peak_chromatic {
                    2u8
                } else if comp_cells_cnt[cid] > single_obj_cells {
                    4u8
                } else {
                    3u8
                };
                if comp_track_cnt[cid] >= max_in_comp {
                    too_close = true;
                }
            }
            if too_close {
                continue;
            }

            let mut free_slot: Option<usize> = None;
            for m in 0..MAX_TARGETS {
                if a.targets[m].active == 0 {
                    free_slot = Some(m);
                    break;
                }
            }
            if free_slot.is_none() {
                let cand_eca = best_patch_eca / eca_norm_denom;
                if cand_eca > 0.085 || compact_projectile {
                    let mut worst_m: Option<usize> = None;
                    let mut worst_score = 1e9f32;
                    for m in 0..MAX_TARGETS {
                        let t = &a.targets[m];
                        let is_pinned = a.focus_target_id != 0 && t.track_id == a.focus_target_id;
                        let ev_slot_age = f_idx.saturating_sub(t.birth_frame).saturating_add(1);
                        let evictable = !is_pinned
                            && (t.confirmed == 0
                                || t.coasting >= 56
                                || (ev_slot_age >= 20 && t.max_excursion2 < (3.6 * s) * (3.6 * s)));
                        if evictable {
                            let sc = t.vis_eca_ema
                                + if t.confirmed != 0 { 0.04 } else { 0.0 }
                                - 0.002 * (t.coasting as f32);
                            if sc < worst_score {
                                worst_score = sc;
                                worst_m = Some(m);
                            }
                        }
                    }
                    free_slot = worst_m;
                }
            }
            if let Some(m) = free_slot {
                let was_inactive = a.targets[m].active == 0;
                if a.targets[m].active != 0 && a.targets[m].confirmed != 0 {
                    let ev_x = a.targets[m].pos_x;
                    let ev_y = a.targets[m].pos_y;
                    archive_dormant_track(a, m, f_idx, ev_x, ev_y, s);
                }
                let clear_bit = !(1u16 << m);
                for o in 0..MAX_TARGETS {
                    a.targets[o].proximity_mask &= clear_bit;
                }
                let tlx = best_tlx;
                let tly = best_tly;

                ensure_chroma!(a, tlx, tly, patch_dim, patch_dim, frame_w, frame_h);
                a.targets[m] = core::mem::zeroed();
                capture_template_scaled(
                    &a.luma,
                    &a.chroma_rg,
                    &a.chroma_by,
                    &a.eca_res,
                    frame_w,
                    frame_h,
                    scale_px,
                    tlx,
                    tly,
                    &mut a.targets[m],
                );

                a.targets[m].active = 1;
                a.targets[m].confirmed = 0; // Tentative until M-of-N (`3` hits + real motion; Track ID allocated on confirm)
                a.targets[m].wave1_auto_owned = 0;
                a.targets[m].phys_stamp_tid = 0;
                a.targets[m].just_confirmed = 0;
                a.targets[m].hit_mask_8 = 0x01;
                a.targets[m].static_streak = 0;
                a.targets[m].owner_slot = m as u8;
                a.targets[m].track_id = 0;
                a.targets[m].birth_frame = f_idx;
                a.targets[m].anchor_sig = a.targets[m].curr_sig;
                a.targets[m].orig_birth_frame = f_idx;
                a.targets[m].sustained_10s_counted = 0;
                a.targets[m].sustained_60s_counted = 0;
                a.targets[m].last_padic_val = 32;
                a.targets[m].age = 1;
                a.targets[m].pos_x = px;
                a.targets[m].pos_y = py;
                a.targets[m].pred_cx = px;
                a.targets[m].pred_cy = py;
                a.targets[m].entry_x = px;
                a.targets[m].entry_y = py;
                a.targets[m].birth_x = px;
                a.targets[m].birth_y = py;
                a.targets[m].osc_mean_x = px;
                a.targets[m].osc_mean_y = py;
                a.targets[m].osc_ext_min_x = px;
                a.targets[m].osc_ext_max_x = px;
                a.targets[m].max_excursion2 = 0.0;
                a.targets[m].vis_eca_ema = (best_patch_eca / eca_norm_denom).min(0.35);
                a.targets[m].step_ref_x = px;
                a.targets[m].step_ref_y = py;
                a.targets[m].stim_ref_x = px;
                a.targets[m].stim_ref_y = py;
                a.targets[m].axis_x = 0.0;
                a.targets[m].axis_y = 1.0; // Default vertical principal tremor axis

                // Only birth slot `slot_in` has observed visibility; pre-birth history is marked 0!
                for sr in 0..RING_LEN {
                    a.targets[m].ring_raw_x[sr] = px;
                    a.targets[m].ring_raw_y[sr] = py;
                    a.targets[m].ring_sig[sr] = a.targets[m].anchor_sig;
                    a.targets[m].ring_vis[sr] = if sr == slot_in { 1 } else { 0 };
                    a.targets[m].fista_x[sr] = px;
                    a.targets[m].fista_y[sr] = py;
                }
                a.targets[m].arc_raw_x = [0.0; HIST_RING_LEN];
                a.targets[m].arc_raw_y = [0.0; HIST_RING_LEN];
                a.targets[m].arc_vis = [0; HIST_RING_LEN];
                a.targets[m].tremor_hist_x = [0.0; HIST_RING_LEN];
                a.targets[m].tremor_hist_y = [0.0; HIST_RING_LEN];
                a.targets[m].tremor_valid = [0; HIST_RING_LEN];
                a.targets[m].arc_raw_x[arc_slot_in] = px;
                a.targets[m].arc_raw_y[arc_slot_in] = py;
                a.targets[m].arc_vis[arc_slot_in] = 1;

                a.imm[m].reset();
                a.ode_states[m].reset();
                a.pend_states[m].reset();
                physics_id::reset_collision_slot(&mut a.collision_state, m);

                a.kalman[m] = CausalKalmanTrack {
                    active: 1,
                    current_owner: m as u8,
                    mismatch_streak: 0,
                    _pad: 0,
                    anchor_sig: a.targets[m].anchor_sig,
                    x: px,
                    y: py,
                    vx: 0.0,
                    vy: 0.0,
                    p_xx: 10.0 * s2,
                    p_xv: 0.0,
                    p_vv: 16.0 * s2,
                    coasting: 0,
                    ring_x: [px; RING_LEN],
                    ring_y: [py; RING_LEN],
                };
                if cid > 0 && cid < 32 {
                    comp_track_cnt[cid] = comp_track_cnt[cid].saturating_add(1);
                }
                if was_inactive {
                    act_hero_cnt += 1;
                }
                births_this_frame += 1;
            }
        }
    }

    // ------------------------------------------------------------------------
    // 6. Dual FISTA Trend Filter + Non-Causal Tremor Revision + Equalized Hilbert
    // ------------------------------------------------------------------------
    let u_disp = (RING_LEN - 1).saturating_sub(eff_k);
    let mut primary_idx: i32 = -1;
    let mut best_primary_score = -1.0f32;

    let mut perc_mask: u32 = 0;
    let mut perc_lead: f32 = 0.0;
    let mut perc_zone: f32 = 0.0;
    let mut perc_vel: f32 = 0.0;
    let mut any_consumed_stimulus = false;

    // Exact 15-tap Hamming-windowed Hilbert taps: h[n] = (2 / (pi * n)) * (0.54 + 0.46 * cos(pi * n / 8))
    const H_TAPS: [(usize, f32); 4] = [
        (1, 0.61432828),
        (3, 0.15194721),
        (5, 0.04634154),
        (7, 0.01046016),
    ];

    for m in 0..MAX_TARGETS {
        a.targets[m].strike_fired = 0;
        if a.targets[m].active == 0 {
            continue;
        }

        let (consumed, pooled_ay) = solve_noncausal_fista_target(
            &mut a.targets[m],
            slot_in,
            u_disp,
            a.fista_lambda,
            fps,
            f_idx,
            a.stimulus_armed,
            a.stimulus_frame,
            frame_w,
            frame_h,
            s,
            a.lab_mode != 1,
            a.scene_gravity_px_s2,
            a.scene_gravity_conf,
            &a.hist_t_sec,
            &a.hist_cam_x,
            &a.hist_cam_y,
            a.last_discont_frame,
        );
        if consumed {
            any_consumed_stimulus = true;
        }
        if let Some(ay_obs) = pooled_ay {
            if a.targets[m].confirmed != 0 {
                let tid = a.targets[m].track_id;
                if a.scene_gravity_arcs == 0 || a.scene_gravity_px_s2 <= 10.0 * s {
                    a.scene_gravity_px_s2 = ay_obs;
                    a.scene_gravity_conf = 0.24;
                    a.scene_gravity_arcs = 1;
                    a.last_grav_arc_track = tid;
                    a.last_grav_arc_frame = f_idx;
                    a.targets[m].grav_arc_counted = if a.targets[m].vel_y > 2.0 * s { 2 } else { 1 };
                } else {
                    let rel_diff =
                        (ay_obs - a.scene_gravity_px_s2).abs() / a.scene_gravity_px_s2.max(1.0);
                    let same_arc_cont = a.targets[m].grav_arc_counted != 0;
                    let new_indep_arc = a.targets[m].grav_arc_counted == 0
                        && (tid != a.last_grav_arc_track
                            || f_idx >= a.last_grav_arc_frame.saturating_add(4));

                    if rel_diff < 0.34 {
                        a.scene_gravity_px_s2 = 0.80 * a.scene_gravity_px_s2 + 0.20 * ay_obs;
                        if same_arc_cont {
                            a.last_grav_arc_frame = f_idx;
                            if a.scene_gravity_arcs < 2 {
                                a.scene_gravity_conf = (a.scene_gravity_conf + 0.02).min(0.32);
                            } else {
                                a.scene_gravity_conf = (a.scene_gravity_conf + 0.03).min(1.0);
                            }
                        } else if new_indep_arc {
                            a.scene_gravity_arcs = a.scene_gravity_arcs.saturating_add(1);
                            a.last_grav_arc_track = tid;
                            a.last_grav_arc_frame = f_idx;
                            a.targets[m].grav_arc_counted =
                                if a.targets[m].vel_y > 2.0 * s { 2 } else { 1 };
                            a.scene_gravity_conf = (a.scene_gravity_conf + 0.34).min(1.0);
                        }
                    } else if rel_diff >= 0.42 {
                        a.scene_gravity_conf *= 0.62;
                        a.scene_gravity_px_s2 = 0.88 * a.scene_gravity_px_s2 + 0.12 * ay_obs;
                        if a.scene_gravity_conf < 0.18 {
                            a.scene_gravity_px_s2 = ay_obs;
                            a.scene_gravity_conf = 0.22;
                            a.scene_gravity_arcs = 1;
                            a.last_grav_arc_track = tid;
                            a.last_grav_arc_frame = f_idx;
                            for o in 0..MAX_TARGETS {
                                a.targets[o].grav_arc_counted = 0;
                            }
                            a.targets[m].grav_arc_counted =
                                if a.targets[m].vel_y > 2.0 * s { 2 } else { 1 };
                        }
                    } else if same_arc_cont {
                        a.last_grav_arc_frame = f_idx;
                    }
                }
            }
        }

        let s_disp = slot_disp;
        let disp_f = f_idx.wrapping_sub(eff_k as u32);
        let fx = a.targets[m].fista_x[u_disp];
        let fy = a.targets[m].fista_y[u_disp];
        let rx = a.targets[m].ring_raw_x[s_disp];
        let ry = a.targets[m].ring_raw_y[s_disp];
        let vis_d = if a.last_discont_frame > 0 && disp_f < a.last_discont_frame {
            0
        } else {
            a.targets[m].ring_vis[s_disp]
        };

        let mut has_past_vis = false;
        let mut has_future_vis = false;
        for u in 0..u_disp {
            let su = (slot_in + 1 + u) % RING_LEN;
            if a.targets[m].ring_vis[su] != 0 {
                if a.ring_cursor + (u as u64) >= ((RING_LEN - 1) as u64) {
                    let fu = f_idx.wrapping_sub((RING_LEN - 1 - u) as u32);
                    if a.last_discont_frame == 0
                        || (fu.wrapping_sub(a.last_discont_frame) < 0x8000_0000)
                    {
                        has_past_vis = true;
                    }
                }
            }
        }
        if !has_past_vis && a.ring_cursor >= (eff_k as u64) {
            let disp_cursor = a.ring_cursor - (eff_k as u64);
            for back in 1..=48u32 {
                if disp_cursor < (back as u64) {
                    break;
                }
                let pf = disp_f.wrapping_sub(back);
                if pf.wrapping_sub(a.targets[m].birth_frame) >= 0x8000_0000
                    || (a.last_discont_frame > 0
                        && pf.wrapping_sub(a.last_discont_frame) >= 0x8000_0000)
                {
                    break;
                }
                if a.targets[m].arc_vis[(pf as usize) & (ARC_HIST_LEN - 1)] != 0 {
                    has_past_vis = true;
                    break;
                }
            }
        }
        for u in (u_disp + 1)..RING_LEN {
            let su = (slot_in + 1 + u) % RING_LEN;
            if a.targets[m].ring_vis[su] != 0 {
                if a.ring_cursor + (u as u64) >= ((RING_LEN - 1) as u64) {
                    let fu = f_idx.wrapping_sub((RING_LEN - 1 - u) as u32);
                    if a.last_discont_frame == 0
                        || (fu.wrapping_sub(a.last_discont_frame) < 0x8000_0000)
                    {
                        has_future_vis = true;
                    }
                }
            }
        }
        a.targets[m].bridged_at_disp =
            if (vis_d == 0 || vis_d == 2) && has_past_vis && has_future_vis {
                1
            } else {
                0
            };
        a.targets[m].disp_fista_x = fx;
        a.targets[m].disp_fista_y = fy;
        a.targets[m].disp_raw_x = rx;
        a.targets[m].disp_raw_y = ry;
        a.targets[m].disp_vis = if vis_d != 0 { 1.0 } else { 0.0 };

        // 6a. Mode C: Scan lookahead window for sparse D2*X acceleration impulses, deduplicated on physical event time & space
        let mut max_z = 0.0f32;
        let mut max_raw_dv = 0.0f32;
        let mut strike_u = u_disp;
        let scan_start = u_disp.min(9);
        let mut raw_dv_arr = [0.0f32; 10];
        let mut raw_dv_ok = [0u8; 10];

        for zk in 0..10usize {
            if a.ring_cursor + (zk as u64) < ((RING_LEN - 1) as u64) {
                continue;
            }
            let f_z0 = f_idx.wrapping_sub((RING_LEN - 1 - zk) as u32);
            if a.last_discont_frame > 0 && f_z0.wrapping_sub(a.last_discont_frame) >= 0x8000_0000 {
                continue;
            }
            let su0 = (slot_in + 1 + zk) % RING_LEN;
            let su1 = (slot_in + 1 + zk + 1) % RING_LEN;
            let su2 = (slot_in + 1 + zk + 2) % RING_LEN;
            if a.targets[m].ring_vis[su0] == 0
                || a.targets[m].ring_vis[su1] == 0
                || a.targets[m].ring_vis[su2] == 0
            {
                continue;
            }
            let dta = frame_dt_span_sec(&a.hist_t_sec, f_z0, f_z0 + 1, safe_dt);
            let dtb = frame_dt_span_sec(&a.hist_t_sec, f_z0 + 1, f_z0 + 2, safe_dt);
            let dt_m = 0.5 * (dta + dtb);
            let c0x = a.hist_cam_x[(f_z0 as usize) & HIST_MASK];
            let c1x = a.hist_cam_x[((f_z0 + 1) as usize) & HIST_MASK];
            let c2x = a.hist_cam_x[((f_z0 + 2) as usize) & HIST_MASK];
            let c0y = a.hist_cam_y[(f_z0 as usize) & HIST_MASK];
            let c1y = a.hist_cam_y[((f_z0 + 1) as usize) & HIST_MASK];
            let c2y = a.hist_cam_y[((f_z0 + 2) as usize) & HIST_MASK];
            let wx0 = a.targets[m].ring_raw_x[su0] + c0x;
            let wx1 = a.targets[m].ring_raw_x[su1] + c1x;
            let wx2 = a.targets[m].ring_raw_x[su2] + c2x;
            let wy0 = a.targets[m].ring_raw_y[su0] + c0y;
            let wy1 = a.targets[m].ring_raw_y[su1] + c1y;
            let wy2 = a.targets[m].ring_raw_y[su2] + c2y;
            let dvx = ((wx2 - wx1) / dtb - (wx1 - wx0) / dta) * dt_m;
            let dvy = ((wy2 - wy1) / dtb - (wy1 - wy0) / dta) * dt_m;
            let raw_dv = wasm_sqrtf(dvx * dvx + dvy * dvy);
            raw_dv_arr[zk] = raw_dv;
            raw_dv_ok[zk] = 1;
            if zk >= scan_start {
                let rate_norm_z = (1.0 / (dt_m * 60.0)).clamp(0.15, 16.0);
                let zmag = wasm_sqrtf(
                    a.targets[m].fista_zx[zk] * a.targets[m].fista_zx[zk]
                        + a.targets[m].fista_zy[zk] * a.targets[m].fista_zy[zk],
                ) * (rate_norm_z * rate_norm_z);
                if zmag > max_z {
                    max_z = zmag;
                    max_raw_dv = raw_dv;
                    strike_u = zk + 1;
                }
            }
        }
        let mut flank_dv = 0.0f32;
        if strike_u >= 1 && strike_u <= 10 {
            let b_zk = strike_u - 1;
            let mut sum_f = 0.0f32;
            let mut cnt_f = 0.0f32;
            if b_zk >= 2 && raw_dv_ok[b_zk - 2] != 0 {
                sum_f += raw_dv_arr[b_zk - 2];
                cnt_f += 1.0;
            } else if b_zk >= 1 && raw_dv_ok[b_zk - 1] != 0 {
                sum_f += raw_dv_arr[b_zk - 1];
                cnt_f += 1.0;
            }
            if b_zk + 2 < 10 && raw_dv_ok[b_zk + 2] != 0 {
                sum_f += raw_dv_arr[b_zk + 2];
                cnt_f += 1.0;
            } else if b_zk + 1 < 10 && raw_dv_ok[b_zk + 1] != 0 {
                sum_f += raw_dv_arr[b_zk + 1];
                cnt_f += 1.0;
            }
            if cnt_f > 0.0 {
                flank_dv = sum_f / cnt_f;
            }
        }
        a.targets[m].l1_impulse_mag = max_z;

        // Enforce temporal impulse sparsity against both flanking intervals (`flank_dv`):
        // rigid collisions concentrate Delta-v into 1-2 impact frames, whereas harmonic pendulum/metronome turnarounds have smooth C^infty acceleration!
        let min_z_strike = if a.lab_mode == 1 { 2.80 * s } else { 0.85 * s };
        let min_dv_strike = if a.lab_mode == 1 { 4.60 * s } else { 1.50 * s };
        let dirac_sparse = if a.lab_mode == 2 {
            a.targets[m].pivot_valid == 0 || max_raw_dv >= 2.0 * flank_dv + 0.55 * s
        } else if a.lab_mode == 1 {
            max_raw_dv >= 2.65 * flank_dv + 0.85 * s && a.targets[m].retrace_dirs != 3
        } else {
            max_raw_dv >= 2.05 * flank_dv + 0.55 * s
        };

        if a.targets[m].confirmed != 0
            && a.targets[m].max_excursion2 >= (8.0 * s) * (8.0 * s)
            && max_z > min_z_strike
            && max_raw_dv >= min_dv_strike
            && dirac_sparse
            && (f_idx as usize) + strike_u >= (RING_LEN - 1)
        {
            let event_frame = ((f_idx as usize) + strike_u - (RING_LEN - 1)) as u32;
            let min_refractory_frames = ((0.13 * fps) as u32).max(6);
            let mut allowed = a.targets[m].has_struck == 0
                || event_frame >= a.targets[m].last_strike_event_frame.saturating_add(min_refractory_frames);

            let sx_pos = a.targets[m].fista_x[strike_u];
            let sy_pos = a.targets[m].fista_y[strike_u];
            let mut paired_partner: Option<usize> = None;
            // Tight close-contact radius (<= 30 SD px / 60 HD px within <= 3 frames) prevents distant non-contact scatter pairings
            let partner_r2 = (30.0 * s) * (30.0 * s);
            let spatial_dedup_r2 = (26.0 * s) * (26.0 * s);

            for o in 0..MAX_TARGETS {
                if o != m && a.targets[o].active != 0 && a.targets[o].has_struck != 0 {
                    let df = event_frame.abs_diff(a.targets[o].last_strike_event_frame);
                    if df <= 3 {
                        let dx = sx_pos - a.targets[o].strike_x;
                        let dy = sy_pos - a.targets[o].strike_y;
                        let d2 = dx * dx + dy * dy;
                        if d2 <= partner_r2 && paired_partner.is_none() {
                            paired_partner = Some(o);
                        }
                        if allowed && df <= 2 && d2 <= spatial_dedup_r2 {
                            allowed = false;
                        }
                    }
                }
            }

            let same_event_refine = a.targets[m].has_struck != 0
                && event_frame.abs_diff(a.targets[m].last_strike_event_frame) <= 1;

            if allowed || same_event_refine || paired_partner.is_some() {
                if let Some((in_d, defl_d, speed_rat, ovx_s, ovy_s, pre_ok)) =
                    measure_strike_kinematics(
                        &a.targets[m],
                        slot_in,
                        strike_u,
                        s,
                        f_idx,
                        &a.hist_t_sec,
                        a.last_discont_frame,
                        safe_dt,
                    )
                {
                    if pre_ok || a.targets[m].has_struck == 0 || allowed {
                        a.targets[m].strike_in_deg = in_d;
                        a.targets[m].strike_deflect_deg = defl_d;
                        a.targets[m].strike_restitution = speed_rat;
                    }
                    a.targets[m].strike_out_vx = ovx_s;
                    a.targets[m].strike_out_vy = ovy_s;

                    if let Some(o) = paired_partner {
                        let o_vx = a.targets[o].strike_out_vx;
                        let o_vy = a.targets[o].strike_out_vy;
                        let min_scatter_v2 = (20.0 * s) * (20.0 * s);
                        let m_v2 = ovx_s * ovx_s + ovy_s * ovy_s;
                        let o_v2 = o_vx * o_vx + o_vy * o_vy;
                        if m_v2 >= min_scatter_v2 && o_v2 >= min_scatter_v2 {
                            let rad_to_deg = 180.0 / PI;
                            let th1 = fast_atan2f(-o_vy, o_vx) * rad_to_deg;
                            let th2 = fast_atan2f(-ovy_s, ovx_s) * rad_to_deg;
                            let sc_deg = (wrap_pi((th1 - th2) * (PI / 180.0)) * rad_to_deg).abs();
                            a.scatter_deg = sc_deg;
                            a.scatter_x = 0.5 * (sx_pos + a.targets[o].strike_x);
                            a.scatter_y = 0.5 * (sy_pos + a.targets[o].strike_y);
                            a.scatter_th1_deg = th1;
                            a.scatter_th2_deg = th2;
                            a.scatter_event_frame = event_frame;
                        }
                    }
                }
            }

            if allowed {
                a.targets[m].has_struck = 1;
                a.targets[m].last_strike_event_frame = event_frame;
                a.targets[m].strike_fired = 1;
                if a.lab_mode == 2 {
                    a.targets[m].pivot_valid = 0;
                    a.targets[m].cand_piv_streak = 0;
                    a.targets[m].retrace_turnaround = 0;
                    a.targets[m].retrace_dirs = 0;
                    a.targets[m].freq_valid = 0;
                    a.targets[m].osc_cross_count = 0;
                    a.targets[m].osc_prev_sign = 0;
                }
                a.last_strike_tid = a.targets[m].track_id;
                a.last_strike_frame = event_frame;

                let qx = if sx_pos >= (frame_w as f32 * 0.5) { 1u8 } else { 0u8 };
                let qy = if sy_pos >= (frame_h as f32 * 0.5) { 2u8 } else { 0u8 };
                let zone = qx + qy;
                let lead_frames = (strike_u.saturating_sub(u_disp)) as f32;
                let vel = (max_z / (5.0 * s)).clamp(0.2, 1.0);

                a.targets[m].strike_zone = zone;
                a.targets[m].strike_lead_frames = lead_frames;
                a.targets[m].strike_vel = vel;
                a.targets[m].strike_x = sx_pos;
                a.targets[m].strike_y = sy_pos;

                if perc_mask == 0 {
                    perc_lead = lead_frames;
                    perc_zone = zone as f32;
                    perc_vel = vel;
                }
                perc_mask |= 1 << m;
            }
        }

        // 6b. Timestamp-exact display-time velocity v(t - K), flight angle theta(t - K), & angular rate
        let f_at_u = |u: usize| -> u32 {
            f_idx.wrapping_sub((RING_LEN - 1 - u) as u32)
        };
        let (fvx_sec, fvy_sec) = if u_disp >= 1
            && u_disp + 1 < RING_LEN
            && a.ring_cursor >= ((RING_LEN - u_disp) as u64)
            && (a.last_discont_frame == 0
                || f_at_u(u_disp - 1).wrapping_sub(a.last_discont_frame) < 0x8000_0000)
        {
            let dt_2 = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp - 1), f_at_u(u_disp + 1), safe_dt);
            (
                (a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp - 1]) / dt_2,
                (a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp - 1]) / dt_2,
            )
        } else if u_disp + 1 < RING_LEN
            && a.ring_cursor >= ((RING_LEN - 1 - u_disp) as u64)
            && (a.last_discont_frame == 0
                || f_at_u(u_disp).wrapping_sub(a.last_discont_frame) < 0x8000_0000)
        {
            let dt_1 = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp), f_at_u(u_disp + 1), safe_dt);
            (
                (a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp]) / dt_1,
                (a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp]) / dt_1,
            )
        } else {
            let dt_1 = frame_dt_span_sec(&a.hist_t_sec, f_at_u(10), f_at_u(11), safe_dt);
            (
                (a.targets[m].fista_x[11] - a.targets[m].fista_x[10]) / dt_1,
                (a.targets[m].fista_y[11] - a.targets[m].fista_y[10]) / dt_1,
            )
        };
        let fspeed_px_s = wasm_sqrtf(fvx_sec * fvx_sec + fvy_sec * fvy_sec);
        a.targets[m].speed_px_s = fspeed_px_s;
        if fspeed_px_s >= 18.0 * s {
            a.targets[m].heading_deg = fast_atan2f(-fvy_sec, fvx_sec) * (180.0 / PI);
        }
        if u_disp >= 2
            && u_disp + 2 < RING_LEN
            && a.ring_cursor >= ((RING_LEN + 1 - u_disp) as u64)
            && (a.last_discont_frame == 0
                || f_at_u(u_disp - 2).wrapping_sub(a.last_discont_frame) < 0x8000_0000)
        {
            let dt_p = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp - 2), f_at_u(u_disp), 2.0 * safe_dt);
            let dt_n = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp), f_at_u(u_disp + 2), 2.0 * safe_dt);
            let vpx = (a.targets[m].fista_x[u_disp] - a.targets[m].fista_x[u_disp - 2]) / dt_p;
            let vpy = (a.targets[m].fista_y[u_disp] - a.targets[m].fista_y[u_disp - 2]) / dt_p;
            let vnx = (a.targets[m].fista_x[u_disp + 2] - a.targets[m].fista_x[u_disp]) / dt_n;
            let vny = (a.targets[m].fista_y[u_disp + 2] - a.targets[m].fista_y[u_disp]) / dt_n;
            let min_w2 = (36.0 * s) * (36.0 * s);
            if vpx * vpx + vpy * vpy >= min_w2 && vnx * vnx + vny * vny >= min_w2 {
                let dth_d = wrap_pi(fast_atan2f(-vny, vnx) - fast_atan2f(-vpy, vpx));
                if dth_d.abs() < 0.78 {
                    let inst_w_deg = (dth_d / (0.5 * (dt_p + dt_n))) * (180.0 / PI);
                    a.targets[m].ang_vel_deg_s =
                        0.70 * a.targets[m].ang_vel_deg_s + 0.30 * inst_w_deg.clamp(-360.0, 360.0);
                } else {
                    a.targets[m].ang_vel_deg_s *= 0.65;
                }
            } else {
                a.targets[m].ang_vel_deg_s *= 0.78;
            }
        } else if u_disp >= 1
            && u_disp + 1 < RING_LEN
            && a.ring_cursor >= ((RING_LEN - u_disp) as u64)
            && (a.last_discont_frame == 0
                || f_at_u(u_disp - 1).wrapping_sub(a.last_discont_frame) < 0x8000_0000)
        {
            let dt_p = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp - 1), f_at_u(u_disp), safe_dt);
            let dt_n = frame_dt_span_sec(&a.hist_t_sec, f_at_u(u_disp), f_at_u(u_disp + 1), safe_dt);
            let vpx = (a.targets[m].fista_x[u_disp] - a.targets[m].fista_x[u_disp - 1]) / dt_p;
            let vpy = (a.targets[m].fista_y[u_disp] - a.targets[m].fista_y[u_disp - 1]) / dt_p;
            let vnx = (a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp]) / dt_n;
            let vny = (a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp]) / dt_n;
            let min_w2 = (36.0 * s) * (36.0 * s);
            if vpx * vpx + vpy * vpy >= min_w2 && vnx * vnx + vny * vny >= min_w2 {
                let dth_d = wrap_pi(fast_atan2f(-vny, vnx) - fast_atan2f(-vpy, vpx));
                if dth_d.abs() < 0.55 {
                    let inst_w_deg = (dth_d / (0.5 * (dt_p + dt_n))) * (180.0 / PI);
                    a.targets[m].ang_vel_deg_s =
                        0.68 * a.targets[m].ang_vel_deg_s + 0.32 * inst_w_deg.clamp(-360.0, 360.0);
                } else {
                    a.targets[m].ang_vel_deg_s *= 0.65;
                }
            } else {
                a.targets[m].ang_vel_deg_s *= 0.78;
            }
        } else {
            a.targets[m].ang_vel_deg_s *= 0.78;
        }

        // Keep display-time Kasa pivot & turnaround bounds registered in `disp_f`'s screen frame
        let disp_cam_x = a.hist_cam_x[(disp_f as usize) & HIST_MASK];
        let disp_cam_y = a.hist_cam_y[(disp_f as usize) & HIST_MASK];
        if disp_f > 0 {
            let prev_disp_cam_x = a.hist_cam_x[((disp_f - 1) as usize) & HIST_MASK];
            let prev_disp_cam_y = a.hist_cam_y[((disp_f - 1) as usize) & HIST_MASK];
            let d_pan_x = -(disp_cam_x - prev_disp_cam_x);
            let d_pan_y = -(disp_cam_y - prev_disp_cam_y);
            if d_pan_x != 0.0 || d_pan_y != 0.0 {
                a.targets[m].pivot_x += d_pan_x;
                a.targets[m].pivot_y += d_pan_y;
                a.targets[m].cand_piv_x += d_pan_x;
                a.targets[m].cand_piv_y += d_pan_y;
                a.targets[m].osc_ext_min_x += d_pan_x;
                a.targets[m].osc_ext_max_x += d_pan_x;
            }
        }

        // 6c. Synchronized `t - K` Kasa 2x2 Circle Fit with Center-Stability & Turnaround Gate
        // Evaluates history strictly up to `disp_f = f_idx - eff_k` so bob & virtual pivot match in sample time,
        // breaking across strikes/discontinuities/kinks and requiring oscillatory turnaround (or Mode B plumb swing).
        let mut pivot_updated = false;
        let recent_strike_at_disp = a.targets[m].has_struck != 0
            && disp_f <= a.targets[m].last_strike_event_frame.saturating_add(5);
        if a.ring_cursor >= (eff_k as u64)
            && !(a.micro_swarm.confirmed_count >= 960
                && a.inspected_micro_tid_pin == 0
                && a.focus_target_id == 0)
            && vis_d != 0
            && a.targets[m].age >= 8
            && a.targets[m].strike_fired == 0
            && !recent_strike_at_disp
            && (a.targets[m].is_ballistic == 0
                || a.lab_mode == 1
                || a.targets[m].retrace_turnaround > 0
                || a.targets[m].pivot_valid > 0)
        {
            let max_win = ARC_HIST_LEN.saturating_sub(eff_k).min(36);
            let mut rev_x = [0.0f32; 36];
            let mut rev_y = [0.0f32; 36];
            let mut n_rev = 0usize;
            let mut miss_a = 0usize;
            let mut any_gap_in_rev = false;
            let mut gap_in_recent = false;
            let max_kink = (7.2 * s) * (safe_dt * 60.0).clamp(0.85, 2.25);

            for back in 0..max_win {
                if (disp_f as usize) < back {
                    break;
                }
                let fa = disp_f - (back as u32);
                if fa < a.targets[m].birth_frame {
                    break;
                }
                if a.last_discont_frame > 0 && fa < a.last_discont_frame {
                    break;
                }
                if a.targets[m].has_struck != 0
                    && a.lab_mode == 2
                    && fa <= a.targets[m].last_strike_event_frame
                {
                    break;
                }
                let sa = (fa as usize) & (ARC_HIST_LEN - 1);
                if a.targets[m].arc_vis[sa] != 0 {
                    miss_a = 0;
                    let ax_u = a.targets[m].arc_raw_x[sa] + (a.hist_cam_x[sa] - disp_cam_x);
                    let ay_u = a.targets[m].arc_raw_y[sa] + (a.hist_cam_y[sa] - disp_cam_y);
                    if n_rev > 0 {
                        let dx1 = rev_x[n_rev - 1] - ax_u;
                        let dy1 = rev_y[n_rev - 1] - ay_u;
                        if dx1 * dx1 + dy1 * dy1 > (42.0 * s) * (42.0 * s) {
                            break;
                        }
                        if n_rev >= 2 {
                            let dx0 = rev_x[n_rev - 2] - rev_x[n_rev - 1];
                            let dy0 = rev_y[n_rev - 2] - rev_y[n_rev - 1];
                            let kx = dx0 - dx1;
                            let ky = dy0 - dy1;
                            if kx * kx + ky * ky > max_kink * max_kink
                                && (dx0 * dx1 + dy0 * dy1) > 0.0
                            {
                                break;
                            }
                        }
                    }
                    rev_x[n_rev] = ax_u;
                    rev_y[n_rev] = ay_u;
                    n_rev += 1;
                } else {
                    any_gap_in_rev = true;
                    if back < 10 {
                        gap_in_recent = true;
                    }
                    miss_a += 1;
                    if miss_a >= 3 {
                        break;
                    }
                }
            }

            if n_rev >= 7 {
                // Reverse into chronological order `full_x[0..n_rev]` ending at `disp_f`
                let mut full_x = [0.0f32; 36];
                let mut full_y = [0.0f32; 36];
                for i in 0..n_rev {
                    full_x[i] = rev_x[n_rev - 1 - i];
                    full_y[i] = rev_y[n_rev - 1 - i];
                }
                let n_recent = n_rev.min(12);
                let rec_off = n_rev - n_recent;

                // Detect smooth 180-deg tether turnaround (strictly contiguous in Mode A/C so highway occlusion snaps never trigger)
                let allow_turn_scan = a.lab_mode != 2
                    && !recent_strike_at_disp
                    && (a.lab_mode == 1 || (!gap_in_recent && a.targets[m].coasting == 0));
                if allow_turn_scan {
                    let cdx = full_x[n_rev - 1] - full_x[n_rev - 4];
                    let cdy = full_y[n_rev - 1] - full_y[n_rev - 4];
                    let cn2 = cdx * cdx + cdy * cdy;
                    let min_chord = if a.lab_mode == 1 { 4.2 * s } else { 5.2 * s };
                    let dot_thresh = if a.lab_mode == 1 { -0.76 } else { -0.78 };
                    if cn2 >= min_chord * min_chord {
                        let cn = wasm_sqrtf(cn2);
                        let ux_c = cdx / cn;
                        let uy_c = cdy / cn;
                        let pn2 = a.targets[m].last_chord_dx * a.targets[m].last_chord_dx
                            + a.targets[m].last_chord_dy * a.targets[m].last_chord_dy;
                        if pn2 > 0.25 {
                            let dot_c = ux_c * a.targets[m].last_chord_dx
                                + uy_c * a.targets[m].last_chord_dy;
                            if dot_c < dot_thresh {
                                a.targets[m].retrace_turnaround = 48;
                            }
                        }
                        if a.targets[m].last_chord_dx == 0.0
                            && a.targets[m].last_chord_dy == 0.0
                        {
                            a.targets[m].last_chord_dx = ux_c;
                            a.targets[m].last_chord_dy = uy_c;
                        } else {
                            let bx = 0.72 * a.targets[m].last_chord_dx + 0.28 * ux_c;
                            let by = 0.72 * a.targets[m].last_chord_dy + 0.28 * uy_c;
                            let bn = wasm_sqrtf((bx * bx + by * by).max(1e-8));
                            a.targets[m].last_chord_dx = bx / bn;
                            a.targets[m].last_chord_dy = by / bn;
                        }
                    }

                    // Also check explicit x-direction reversal inside `full_x[0..n_rev]`
                    if a.lab_mode == 1 || !any_gap_in_rev {
                        let mut min_ix = 0usize;
                        let mut max_ix = 0usize;
                        for i in 1..n_rev {
                            if full_x[i] < full_x[min_ix] {
                                min_ix = i;
                            }
                            if full_x[i] > full_x[max_ix] {
                                max_ix = i;
                            }
                        }
                        let min_pad = if a.lab_mode == 1 { 2usize } else { 3usize };
                        let min_exc = if a.lab_mode == 1 { 5.0 * s } else { 6.5 * s };
                        let rev_right = max_ix >= min_pad
                            && max_ix + min_pad < n_rev
                            && (full_x[max_ix] - full_x[0]) >= min_exc
                            && (full_x[max_ix] - full_x[n_rev - 1]) >= min_exc;
                        let rev_left = min_ix >= min_pad
                            && min_ix + min_pad < n_rev
                            && (full_x[0] - full_x[min_ix]) >= min_exc
                            && (full_x[n_rev - 1] - full_x[min_ix]) >= min_exc;
                        if rev_right {
                            a.targets[m].retrace_turnaround = 48;
                            let r_max = full_x[max_ix];
                            a.targets[m].osc_ext_max_x = if (a.targets[m].retrace_dirs & 2) == 0 {
                                r_max
                            } else {
                                0.72 * a.targets[m].osc_ext_max_x + 0.28 * r_max
                            };
                            a.targets[m].retrace_dirs |= 2;
                        }
                        if rev_left {
                            a.targets[m].retrace_turnaround = 48;
                            let l_min = full_x[min_ix];
                            a.targets[m].osc_ext_min_x = if (a.targets[m].retrace_dirs & 1) == 0 {
                                l_min
                            } else {
                                0.72 * a.targets[m].osc_ext_min_x + 0.28 * l_min
                            };
                            a.targets[m].retrace_dirs |= 1;
                        }
                    }
                }

                let kasa_rmse_mult = if a.targets[m].retrace_turnaround > 0 || a.lab_mode == 1 {
                    3.6
                } else {
                    2.25
                };
                let circle_cand = fit_kasa_circle_subpx(
                    &full_x[rec_off..],
                    &full_y[rec_off..],
                    n_recent,
                    s,
                    3.2,
                    kasa_rmse_mult,
                )
                .map(|c| (c, rec_off, n_recent))
                .or_else(|| {
                    fit_kasa_circle_subpx(&full_x, &full_y, n_rev, s, 1.8, kasa_rmse_mult)
                        .map(|c| (c, 0usize, n_rev))
                });

                if let Some(((xc, yc, rad, my_c, _sag), fit_off, fit_n)) = circle_cand {
                    let r_dx = full_x[n_rev - 1] - full_x[rec_off];
                    let r_dy = full_y[n_rev - 1] - full_y[rec_off];
                    let r_chord2 = r_dx * r_dx + r_dy * r_dy;
                    let sag_expected = r_chord2 / (8.0 * rad.max(8.0 * s));
                    let mut recent_curvature_consistent = true;
                    if a.lab_mode == 0 && n_recent >= 9 && sag_expected >= 1.1 * s {
                        let inv_rc = 1.0 / wasm_sqrtf(r_chord2.max(1e-6));
                        let mut rec_sag = 0.0f32;
                        for i in (rec_off + 1)..(n_rev - 1) {
                            let px_d = full_x[i] - full_x[rec_off];
                            let py_d = full_y[i] - full_y[rec_off];
                            let perp = (px_d * r_dy - py_d * r_dx).abs() * inv_rc;
                            if perp > rec_sag {
                                rec_sag = perp;
                            }
                        }
                        if rec_sag < 0.35 * sag_expected {
                            recent_curvature_consistent = false;
                            a.targets[m].cand_piv_streak = 0;
                            a.targets[m].pivot_valid = 0;
                        }
                    }

                    let mut min_px = 1e9f32;
                    let mut max_px = -1e9f32;
                    let mut min_py = 1e9f32;
                    let mut max_py = -1e9f32;
                    for i in 0..fit_n {
                        let px_i = full_x[fit_off + i];
                        let py_i = full_y[fit_off + i];
                        if px_i < min_px {
                            min_px = px_i;
                        }
                        if px_i > max_px {
                            max_px = px_i;
                        }
                        if py_i < min_py {
                            min_py = py_i;
                        }
                        if py_i > max_py {
                            max_py = py_i;
                        }
                    }
                    let dx_span = max_px - min_px;
                    let dy_span = max_py - min_py;
                    let vert_sep = if a.lab_mode == 1 {
                        (my_c - yc).abs()
                    } else {
                        my_c - yc
                    };
                    let near_plumb_line = a.lab_mode == 1
                        || (min_px <= xc + 6.0 * s && max_px >= xc - 6.0 * s);
                    let tethered_origin = a.lab_mode == 1
                        || (a.targets[m].birth_x - xc).abs() <= 1.35 * rad;
                    let plumb_geom = recent_curvature_consistent
                        && tethered_origin
                        && rad <= 265.0 * s
                        && (a.lab_mode != 0 || dy_span >= 5.0 * s)
                        && dx_span >= 1.05 * dy_span
                        && vert_sep >= (14.0 * s).max(0.42 * rad)
                        && near_plumb_line;

                    if plumb_geom {
                        let tol = (0.18 * rad).max(6.0 * s);
                        if a.targets[m].cand_piv_r > 1.0
                            && (xc - a.targets[m].cand_piv_x).abs() <= tol
                            && (yc - a.targets[m].cand_piv_y).abs() <= tol
                            && (rad - a.targets[m].cand_piv_r).abs() <= 0.25 * rad
                        {
                            a.targets[m].cand_piv_streak =
                                a.targets[m].cand_piv_streak.saturating_add(1).min(32);
                            a.targets[m].cand_piv_x = 0.65 * a.targets[m].cand_piv_x + 0.35 * xc;
                            a.targets[m].cand_piv_y = 0.65 * a.targets[m].cand_piv_y + 0.35 * yc;
                            a.targets[m].cand_piv_r = 0.65 * a.targets[m].cand_piv_r + 0.35 * rad;
                        } else {
                            a.targets[m].cand_piv_streak = 1;
                            a.targets[m].cand_piv_x = xc;
                            a.targets[m].cand_piv_y = yc;
                            a.targets[m].cand_piv_r = rad;
                        }

                        // Symmetric plumb-extremum crossing (`min_px < xc < max_px` with y turning at `xc`)
                        let y_start = full_y[fit_off];
                        let y_end = full_y[fit_off + fit_n - 1];
                        let plumb_crossed = min_px < xc - 4.0 * s
                            && max_px > xc + 4.0 * s
                            && if my_c > yc {
                                (max_py - y_start) >= 1.8 * s && (max_py - y_end) >= 1.8 * s
                            } else {
                                (y_start - min_py) >= 1.8 * s && (y_end - min_py) >= 1.8 * s
                            };

                        let osc_verified = if a.lab_mode == 0 {
                            (a.targets[m].retrace_dirs == 3 && a.targets[m].retrace_turnaround > 0)
                                || (a.targets[m].freq_valid != 0
                                    && a.targets[m].tremor_coherence >= 0.36
                                    && a.targets[m].tremor_freq_hz >= 0.20
                                    && a.targets[m].tremor_freq_hz <= 1.25)
                        } else {
                            a.targets[m].retrace_turnaround > 0
                                || (a.lab_mode == 1 && plumb_crossed)
                        };
                        let center_ready = a.targets[m].cand_piv_streak >= 3
                            || (a.targets[m].pivot_valid > 0 && a.targets[m].cand_piv_streak >= 2);

                        if osc_verified && center_ready {
                            if a.targets[m].pivot_valid == 0 || a.targets[m].pivot_r <= 1.0 {
                                a.targets[m].pivot_x = a.targets[m].cand_piv_x;
                                a.targets[m].pivot_y = a.targets[m].cand_piv_y;
                                a.targets[m].pivot_r = a.targets[m].cand_piv_r;
                            } else {
                                let alpha_p = if (rad - a.targets[m].pivot_r).abs()
                                    > 0.24 * a.targets[m].pivot_r
                                {
                                    0.76
                                } else {
                                    0.24
                                };
                                a.targets[m].pivot_x =
                                    (1.0 - alpha_p) * a.targets[m].pivot_x + alpha_p * xc;
                                a.targets[m].pivot_y =
                                    (1.0 - alpha_p) * a.targets[m].pivot_y + alpha_p * yc;
                                a.targets[m].pivot_r =
                                    (1.0 - alpha_p) * a.targets[m].pivot_r + alpha_p * rad;
                            }
                            a.targets[m].pivot_valid = 26;
                            a.targets[m].is_ballistic = 0;
                            pivot_updated = true;
                        }
                    }
                }
            }
        }
        if !pivot_updated {
            let bottom_piv_ballistic = a.targets[m].is_ballistic != 0
                && a.targets[m].retrace_turnaround == 0
                && a.targets[m].pivot_y > fy;
            if a.targets[m].coasting > 4
                || bottom_piv_ballistic
                || (a.lab_mode == 2 && recent_strike_at_disp)
            {
                a.targets[m].pivot_valid = 0;
                a.targets[m].cand_piv_streak = 0;
            } else {
                a.targets[m].pivot_valid = a.targets[m].pivot_valid.saturating_sub(1);
            }
            if a.targets[m].coasting > 6 {
                a.targets[m].retrace_turnaround = a.targets[m].retrace_turnaround.saturating_sub(1);
            }
        }

        // 6d. Stationary-Definition Tremor History + Rigid-Tether / Resampled-Hilbert Periodicity
        // `tremor_hist` NEVER switches definitions (strictly FISTA interior micro-residual `raw - fista_int`).
        let rev_start = if a.targets[m].age <= 8 { 0usize } else { 3usize };
        let slope_int_x = 0.25 * (a.targets[m].fista_x[8] - a.targets[m].fista_x[4]);
        let slope_int_y = 0.25 * (a.targets[m].fista_y[8] - a.targets[m].fista_y[4]);
        for u in rev_start..RING_LEN {
            if a.ring_cursor + (u as u64) < ((RING_LEN - 1) as u64) {
                continue;
            }
            let sample_f = f_idx.wrapping_sub((RING_LEN - 1 - u) as u32);
            let h_slot = (sample_f as usize) & HIST_MASK;
            let su = (slot_in + 1 + u) % RING_LEN;
            let post_disc = a.last_discont_frame == 0
                || sample_f.wrapping_sub(a.last_discont_frame) < 0x8000_0000;
            if post_disc
                && a.targets[m].ring_vis[su] != 0
                && sample_f.wrapping_sub(a.targets[m].birth_frame) < 0x8000_0000
            {
                let (base_x, base_y) = if u > 8 && a.targets[m].age > 8 {
                    let du = (u - 8) as f32;
                    (
                        a.targets[m].fista_x[8] + du * slope_int_x,
                        a.targets[m].fista_y[8] + du * slope_int_y,
                    )
                } else {
                    (a.targets[m].fista_x[u], a.targets[m].fista_y[u])
                };
                let raw_wx = a.targets[m].ring_raw_x[su] + (a.hist_cam_x[h_slot] - disp_cam_x);
                let raw_wy = a.targets[m].ring_raw_y[su] + (a.hist_cam_y[h_slot] - disp_cam_y);
                a.targets[m].tremor_hist_x[h_slot] = raw_wx - base_x;
                a.targets[m].tremor_hist_y[h_slot] = raw_wy - base_y;
                a.targets[m].tremor_valid[h_slot] = 1;
            } else {
                a.targets[m].tremor_hist_x[h_slot] = 0.0;
                a.targets[m].tremor_hist_y[h_slot] = 0.0;
                a.targets[m].tremor_valid[h_slot] = 0;
            }
        }

        let disp_h_slot = ((f_idx as usize).wrapping_add(HIST_RING_LEN).wrapping_sub(eff_k)) & HIST_MASK;
        let disp_valid = a.ring_cursor >= (eff_k as u64) && a.targets[m].tremor_valid[disp_h_slot] != 0;
        if disp_valid {
            let rdx = a.targets[m].tremor_hist_x[disp_h_slot];
            let rdy = a.targets[m].tremor_hist_y[disp_h_slot];
            a.targets[m].cov_xx = 0.92 * a.targets[m].cov_xx + 0.08 * (rdx * rdx);
            a.targets[m].cov_xy = 0.92 * a.targets[m].cov_xy + 0.08 * (rdx * rdy);
            a.targets[m].cov_yy = 0.92 * a.targets[m].cov_yy + 0.08 * (rdy * rdy);

            let tr = a.targets[m].cov_xx + a.targets[m].cov_yy;
            if tr > 1e-5 {
                let diff = a.targets[m].cov_xx - a.targets[m].cov_yy;
                let cxy = a.targets[m].cov_xy;
                let disc = wasm_sqrtf((diff * diff + 4.0 * cxy * cxy).max(0.0));
                let lam1 = 0.5 * (tr + disc);

                let (mut vx_p, mut vy_p) = if cxy.abs() > 1e-6 {
                    let cand1_x = lam1 - a.targets[m].cov_yy;
                    let cand1_y = cxy;
                    let cand2_x = cxy;
                    let cand2_y = lam1 - a.targets[m].cov_xx;
                    if cand1_x * cand1_x + cand1_y * cand1_y >= cand2_x * cand2_x + cand2_y * cand2_y {
                        (cand1_x, cand1_y)
                    } else {
                        (cand2_x, cand2_y)
                    }
                } else if a.targets[m].cov_xx > a.targets[m].cov_yy {
                    (1.0, 0.0)
                } else {
                    (0.0, 1.0)
                };

                let vnorm = wasm_sqrtf((vx_p * vx_p + vy_p * vy_p).max(1e-12));
                vx_p /= vnorm;
                vy_p /= vnorm;

                if vx_p * a.targets[m].axis_x + vy_p * a.targets[m].axis_y < 0.0 {
                    vx_p = -vx_p;
                    vy_p = -vy_p;
                }
                let bx = 0.88 * a.targets[m].axis_x + 0.12 * vx_p;
                let by = 0.88 * a.targets[m].axis_y + 0.12 * vy_p;
                let bnorm = wasm_sqrtf((bx * bx + by * by).max(1e-12));
                a.targets[m].axis_x = bx / bnorm;
                a.targets[m].axis_y = by / bnorm;
            }
        }

        a.targets[m].prev_phase_valid = a.targets[m].phase_valid;
        let prev_regime = a.targets[m].osc_regime;
        let disp_lag_sec = ((a.hist_t_sec[(f_idx as usize) & HIST_MASK]
            - a.hist_t_sec[(disp_f as usize) & HIST_MASK])
            .max(0.0)) as f32;

        let osc_span_min = if a.lab_mode == 1 { 10.0 * s } else { 24.0 * s };
        let bounded_bi_turn = a.lab_mode != 2
            && (a.targets[m].retrace_dirs & 3) == 3
            && (a.targets[m].osc_ext_max_x - a.targets[m].osc_ext_min_x) >= osc_span_min
            && fx >= a.targets[m].osc_ext_min_x - 18.0 * s
            && fx <= a.targets[m].osc_ext_max_x + 18.0 * s;
        let macro_osc = a.targets[m].pivot_valid > 0 || bounded_bi_turn;

        let disp_obs_or_bridged = vis_d != 0 || a.targets[m].bridged_at_disp != 0;
        if macro_osc && eff_k >= 7 && disp_obs_or_bridged && a.ring_cursor >= (eff_k as u64) {
            // Regime 2: Verified Macro-Excursion Harmonic Oscillator (Kasa plumb or bi-directional turnaround midpoint + >=3 exact f64 zero-crossings)
            let center_x = if bounded_bi_turn {
                0.5 * (a.targets[m].osc_ext_min_x + a.targets[m].osc_ext_max_x)
            } else {
                a.targets[m].pivot_x
            };

            // Backfill recent zero-crossing history from `arc_raw_x` + `hist_t_sec` (bounded strictly to un-overwritten 32-slot ring!)
            if a.targets[m].osc_cross_count == 0 && disp_f > a.targets[m].birth_frame + 4 {
                let mut start_bf = f_idx
                    .saturating_sub((ARC_HIST_LEN - 1) as u32)
                    .max(a.targets[m].birth_frame)
                    .max(a.last_discont_frame);
                if a.lab_mode != 1 {
                    start_bf = start_bf.max(a.targets[m].last_strike_event_frame);
                }
                let mut bf_sign: i8 = 0;
                let mut bf_max_q = 0.0f32;
                let mut bf_prev_q = 0.0f32;
                let mut bf_prev_f = start_bf;
                for fb in start_bf..disp_f {
                    let sb = (fb as usize) & (ARC_HIST_LEN - 1);
                    if a.targets[m].arc_vis[sb] == 0 {
                        continue;
                    }
                    let qb = (a.targets[m].arc_raw_x[sb] + (a.hist_cam_x[sb] - disp_cam_x)) - center_x;
                    if qb.abs() > bf_max_q {
                        bf_max_q = qb.abs();
                    }
                    if bf_sign == 0 {
                        if qb.abs() >= (if a.lab_mode == 1 { 2.0 * s } else { 4.0 * s }) {
                            bf_sign = if qb > 0.0 { 1 } else { -1 };
                            bf_max_q = qb.abs();
                        }
                    } else {
                        let crossed_b = (bf_sign > 0 && qb <= 0.0) || (bf_sign < 0 && qb >= 0.0);
                        if crossed_b && bf_max_q >= (if a.lab_mode == 1 { 2.5 * s } else { 5.0 * s }) {
                            let tb_curr = a.hist_t_sec[(fb as usize) & HIST_MASK];
                            let tb_prev = a.hist_t_sec[(bf_prev_f as usize) & HIST_MASK];
                            let frac_b = if (bf_prev_q - qb).abs() > 1e-4 && bf_prev_q * qb <= 0.0 {
                                (bf_prev_q.abs() / (bf_prev_q.abs() + qb.abs())) as f64
                            } else {
                                0.5
                            };
                            let tb_cross =
                                tb_prev + frac_b.clamp(0.0, 1.0) * (tb_curr - tb_prev).max(0.0);
                            if a.targets[m].osc_cross_count >= 1 {
                                let dt_hb = (tb_cross - a.targets[m].osc_last_cross_t) as f32;
                                if dt_hb >= 0.08 && dt_hb <= 1.25 {
                                    if a.targets[m].osc_cross_count == 1
                                        || a.targets[m].osc_half_period_ema <= 0.01
                                    {
                                        a.targets[m].osc_half_period_ema = dt_hb;
                                        a.targets[m].osc_period_jit_ema = 0.20;
                                    } else {
                                        let rj = (dt_hb - a.targets[m].osc_half_period_ema).abs()
                                            / a.targets[m].osc_half_period_ema.max(0.05);
                                        a.targets[m].osc_period_jit_ema =
                                            if a.targets[m].osc_cross_count == 2 {
                                                rj
                                            } else {
                                                0.65 * a.targets[m].osc_period_jit_ema + 0.35 * rj
                                            };
                                        a.targets[m].osc_half_period_ema =
                                            0.60 * a.targets[m].osc_half_period_ema + 0.40 * dt_hb;
                                    }
                                    a.targets[m].osc_cross_count =
                                        a.targets[m].osc_cross_count.saturating_add(1).min(64);
                                }
                            } else {
                                a.targets[m].osc_cross_count = 1;
                            }
                            a.targets[m].osc_last_cross_t = tb_cross;
                            bf_sign = -bf_sign;
                            bf_max_q = qb.abs();
                        }
                    }
                    bf_prev_q = qb;
                    bf_prev_f = fb;
                }
                if bf_sign != 0 {
                    a.targets[m].osc_prev_sign = bf_sign;
                    a.targets[m].pend_max_q = bf_max_q;
                    a.targets[m].pend_prev_q = bf_prev_q;
                }
            }

            let q_disp = fx - center_x;
            if disp_obs_or_bridged {
                if q_disp.abs() > a.targets[m].pend_max_q {
                    a.targets[m].pend_max_q = q_disp.abs();
                }
                if a.targets[m].osc_prev_sign == 0 {
                    if q_disp.abs() >= (if a.lab_mode == 1 { 2.0 * s } else { 4.0 * s }) {
                        a.targets[m].osc_prev_sign = if q_disp > 0.0 { 1 } else { -1 };
                        a.targets[m].pend_max_q = q_disp.abs();
                    }
                } else {
                    let ps = a.targets[m].osc_prev_sign;
                    let crossed = (ps > 0 && q_disp <= 0.0) || (ps < 0 && q_disp >= 0.0);
                    if crossed
                        && a.targets[m].pend_max_q >= (if a.lab_mode == 1 { 2.5 * s } else { 5.0 * s })
                    {
                        let prev_q = a.targets[m].pend_prev_q;
                        let t_curr = a.hist_t_sec[(disp_f as usize) & HIST_MASK];
                        let t_prev = if disp_f > 0 {
                            a.hist_t_sec[((disp_f - 1) as usize) & HIST_MASK]
                        } else {
                            t_curr - (safe_dt as f64)
                        };
                        let frac = if (prev_q - q_disp).abs() > 1e-4 && prev_q * q_disp <= 0.0 {
                            (prev_q.abs() / (prev_q.abs() + q_disp.abs())) as f64
                        } else {
                            0.5
                        };
                        let t_cross = t_prev + frac.clamp(0.0, 1.0) * (t_curr - t_prev).max(0.0);
                        let mut bounce_skip = false;
                        if a.targets[m].osc_cross_count >= 1 {
                            let dt_half = (t_cross - a.targets[m].osc_last_cross_t) as f32;
                            let hp_prev = a.targets[m].osc_half_period_ema;
                            let eff_half = if a.lab_mode == 1
                                && a.targets[m].osc_cross_count >= 3
                                && a.targets[m].osc_period_jit_ema <= 0.34
                                && hp_prev >= 0.20
                                && dt_half < 0.55 * hp_prev
                            {
                                bounce_skip = true;
                                None
                            } else if dt_half >= 0.08 && dt_half <= 1.25 {
                                Some(dt_half)
                            } else if a.targets[m].osc_cross_count >= 3
                                && a.targets[m].osc_period_jit_ema <= 0.34
                                && hp_prev >= 0.20
                                && dt_half >= 1.65 * hp_prev
                                && dt_half <= 2.35 * hp_prev
                            {
                                Some(0.5 * dt_half)
                            } else {
                                None
                            };
                            if let Some(half_sec) = eff_half {
                                if a.targets[m].osc_cross_count == 1 || hp_prev <= 0.01 {
                                    a.targets[m].osc_half_period_ema = half_sec;
                                    a.targets[m].osc_period_jit_ema = 0.20;
                                } else {
                                    let rj = (half_sec - hp_prev).abs() / hp_prev.max(0.05);
                                    a.targets[m].osc_period_jit_ema =
                                        if a.targets[m].osc_cross_count == 2 {
                                            rj
                                        } else {
                                            0.65 * a.targets[m].osc_period_jit_ema + 0.35 * rj
                                        };
                                    a.targets[m].osc_half_period_ema =
                                        0.60 * hp_prev + 0.40 * half_sec;
                                }
                                a.targets[m].osc_cross_count =
                                    a.targets[m].osc_cross_count.saturating_add(1).min(64);
                            } else if !bounce_skip {
                                a.targets[m].osc_cross_count = 1;
                                a.targets[m].osc_half_period_ema = 0.0;
                                a.targets[m].osc_period_jit_ema = 1.0;
                            }
                        } else {
                            a.targets[m].osc_cross_count = 1;
                        }
                        if !bounce_skip {
                            a.targets[m].osc_last_cross_t = t_cross;
                        }
                        a.targets[m].osc_prev_sign = -ps;
                        a.targets[m].pend_max_q = q_disp.abs();
                    }
                }
                a.targets[m].pend_prev_q = q_disp;
            }

            let period_locked = a.targets[m].osc_cross_count >= 3
                && a.targets[m].osc_half_period_ema >= 0.08
                && a.targets[m].osc_period_jit_ema <= 0.34;
            if period_locked {
                let f_teth = (1.0 / (2.0 * a.targets[m].osc_half_period_ema)).clamp(0.15, 15.0);
                let omega_teth = TWO_PI * f_teth;
                let h_disp = -fvx_sec / omega_teth.max(0.5);
                let amp_teth = wasm_sqrtf(q_disp * q_disp + h_disp * h_disp);
                let prev_phi = a.targets[m].tremor_phase;
                let phi = fast_atan2f(h_disp, q_disp);
                let phi_now = wrap_pi(phi + omega_teth * (disp_lag_sec + a.audio_out_latency_sec));

                a.targets[m].prev_tremor_phase = prev_phi;
                a.targets[m].prev_pred_phase_now = a.targets[m].pred_phase_now;
                a.targets[m].tremor_r = q_disp;
                a.targets[m].tremor_h = h_disp;
                a.targets[m].tremor_phase = phi;
                a.targets[m].pred_phase_now = phi_now;
                a.targets[m].tremor_amp = amp_teth;
                a.targets[m].tremor_omega_ema = omega_teth;
                a.targets[m].tremor_freq_hz = f_teth;
                a.targets[m].tremor_coherence =
                    (1.0 - a.targets[m].osc_period_jit_ema * 1.4).clamp(0.55, 0.99);
                a.targets[m].phase_valid = 1;
                a.targets[m].freq_valid = 1;
                a.targets[m].osc_regime = 2;
            } else {
                let omega_guess = 5.0f32;
                let h_guess = -fvx_sec / omega_guess;
                a.targets[m].tremor_r = q_disp;
                a.targets[m].tremor_h = h_guess;
                a.targets[m].tremor_phase = fast_atan2f(h_guess, q_disp);
                a.targets[m].tremor_amp = wasm_sqrtf(q_disp * q_disp + h_guess * h_guess);
                a.targets[m].phase_valid = if a.lab_mode == 1 { 1 } else { 0 };
                a.targets[m].freq_valid = 0;
                a.targets[m].osc_regime = if a.lab_mode == 1 { 2 } else { 0 };
            }
        } else {
            // Regime 1: Timestamp-Resampled 15-Tap Hilbert Filter on Stationary FISTA Micro-Residual
            let ax = a.targets[m].axis_x;
            let ay = a.targets[m].axis_y;
            let r_disp = if disp_valid {
                ax * a.targets[m].tremor_hist_x[disp_h_slot]
                    + ay * a.targets[m].tremor_hist_y[disp_h_slot]
            } else {
                0.0
            };

            let mut active_tap_mask = 0u8;
            let mut hilb_raw = 0.0f32;
            let mut dt_hilb = safe_dt;
            let no_disc_in_hilb =
                a.last_discont_frame == 0 || disp_f >= a.last_discont_frame.saturating_add(7);

            if eff_k >= 7 && disp_valid && a.ring_cursor >= ((eff_k + 7) as u64) && no_disc_in_hilb {
                let t_c = a.hist_t_sec[(disp_f as usize) & HIST_MASK];
                let dt_core = ((a.hist_t_sec[(disp_f.wrapping_add(3) as usize) & HIST_MASK]
                    - a.hist_t_sec[(disp_f.wrapping_sub(3) as usize) & HIST_MASK])
                    * (1.0 / 6.0)) as f32;
                if dt_core.is_finite() && dt_core > 1e-3 && dt_core < 0.25 {
                    dt_hilb = dt_core;
                }

                // Sample 1D projected residual over [disp_f - 7 ..= disp_f + 7] and linearly resample at uniform +/- odd_k * dt_hilb
                let mut win_t = [0.0f32; 15];
                let mut win_v = [0.0f32; 15];
                let mut win_ok = [0u8; 15];
                for idx_w in 0..15usize {
                    let fw = (disp_f + (idx_w as u32)) - 7;
                    let sw = (fw as usize) & HIST_MASK;
                    win_t[idx_w] = (a.hist_t_sec[sw] - t_c) as f32;
                    win_ok[idx_w] = a.targets[m].tremor_valid[sw];
                    if win_ok[idx_w] != 0 {
                        win_v[idx_w] = ax * a.targets[m].tremor_hist_x[sw]
                            + ay * a.targets[m].tremor_hist_y[sw];
                    }
                }

                let interp_tap = |target_dt: f32| -> Option<f32> {
                    for j in 0..14usize {
                        if win_ok[j] != 0 && win_ok[j + 1] != 0 {
                            let t0 = win_t[j];
                            let t1 = win_t[j + 1];
                            if target_dt >= t0 - 1e-4 && target_dt <= t1 + 1e-4 && (t1 - t0) > 1e-5
                            {
                                let alpha = ((target_dt - t0) / (t1 - t0)).clamp(0.0, 1.0);
                                return Some(win_v[j] + alpha * (win_v[j + 1] - win_v[j]));
                            }
                        }
                    }
                    None
                };

                for (tap_idx, &(odd_k, coeff)) in H_TAPS.iter().enumerate() {
                    let tau_k = (odd_k as f32) * dt_hilb;
                    if let (Some(past_val), Some(fut_val)) = (interp_tap(-tau_k), interp_tap(tau_k))
                    {
                        hilb_raw += coeff * (past_val - fut_val);
                        active_tap_mask |= 1u8 << tap_idx;
                    }
                }
            }

            if (active_tap_mask & 0b0011) == 0b0011 && vis_d != 0 {
                if prev_regime == 2 {
                    a.targets[m].tremor_omega_ema = 0.0;
                    a.targets[m].tremor_coherence = 0.0;
                }
                let f_raw = if a.targets[m].tremor_freq_hz >= 1.0 && prev_regime != 2 {
                    a.targets[m].tremor_freq_hz
                } else {
                    8.5
                };
                let f_est = f_raw.min(0.42 / dt_hilb.max(1e-3)).max(1.0);
                let omega_hat = TWO_PI * f_est * dt_hilb;
                let mut g_gain = 0.0f32;
                for (tap_idx, &(odd_k, coeff)) in H_TAPS.iter().enumerate() {
                    if (active_tap_mask & (1u8 << tap_idx)) != 0 {
                        g_gain += 2.0 * coeff * fast_sinf((odd_k as f32) * omega_hat);
                    }
                }
                let hilb = if g_gain > 0.25 {
                    hilb_raw / g_gain
                } else {
                    hilb_raw
                };

                let prev_phi = a.targets[m].tremor_phase;
                let phi = fast_atan2f(hilb, r_disp);
                let amp = wasm_sqrtf(r_disp * r_disp + hilb * hilb);
                let dt_disp = if disp_f > 0 {
                    frame_dt_span_sec(&a.hist_t_sec, disp_f - 1, disp_f, dt_hilb)
                } else {
                    dt_hilb
                };

                if a.targets[m].prev_phase_valid != 0 && prev_regime == 1 && amp > 0.025 * s {
                    let dphi = wrap_pi(phi - prev_phi);
                    let inst_omega = dphi / dt_disp.max(1e-3);
                    if a.targets[m].tremor_omega_ema == 0.0 {
                        a.targets[m].tremor_omega_ema = inst_omega;
                    } else {
                        a.targets[m].tremor_omega_ema =
                            0.85 * a.targets[m].tremor_omega_ema + 0.15 * inst_omega;
                    }
                    let pred_step = a.targets[m].tremor_omega_ema * dt_disp;
                    let phase_jit = wrap_pi(dphi - pred_step).abs();
                    let inst_coh = (1.0 - phase_jit * 0.63661975).clamp(0.0, 1.0);
                    a.targets[m].tremor_coherence =
                        0.86 * a.targets[m].tremor_coherence + 0.14 * inst_coh;
                    a.targets[m].tremor_freq_hz =
                        (a.targets[m].tremor_omega_ema.abs() / TWO_PI).clamp(0.0, 30.0);
                } else if amp <= 0.015 * s {
                    a.targets[m].tremor_omega_ema *= 0.85;
                    a.targets[m].tremor_coherence *= 0.85;
                    if a.targets[m].tremor_omega_ema.abs() < 0.5 {
                        a.targets[m].tremor_freq_hz = 0.0;
                    }
                }

                // Multi-cycle micro-tremor periodicity gate (`freq_valid` isolated from Regime 2 `osc_*` state;
                // requires >= 0.42*s px physical tremor amplitude so sub-pixel grid aliasing on translating objects never triggers!)
                if r_disp.abs() > a.targets[m].trem_max_q {
                    a.targets[m].trem_max_q = r_disp.abs();
                }
                if a.targets[m].trem_prev_sign == 0 {
                    if r_disp.abs() >= 0.30 * s {
                        a.targets[m].trem_prev_sign = if r_disp > 0.0 { 1 } else { -1 };
                        a.targets[m].trem_max_q = r_disp.abs();
                    }
                } else {
                    let ps = a.targets[m].trem_prev_sign;
                    let crossed = (ps > 0 && r_disp <= 0.0) || (ps < 0 && r_disp >= 0.0);
                    if crossed && a.targets[m].trem_max_q >= 0.36 * s && amp >= 0.42 * s {
                        let prev_q = a.targets[m].trem_prev_q;
                        let t_curr = a.hist_t_sec[(disp_f as usize) & HIST_MASK];
                        let t_prev = if disp_f > 0 {
                            a.hist_t_sec[((disp_f - 1) as usize) & HIST_MASK]
                        } else {
                            t_curr - (dt_disp as f64)
                        };
                        let frac = if (prev_q - r_disp).abs() > 1e-4 && prev_q * r_disp <= 0.0 {
                            (prev_q.abs() / (prev_q.abs() + r_disp.abs())) as f64
                        } else {
                            0.5
                        };
                        let t_cross = t_prev + frac.clamp(0.0, 1.0) * (t_curr - t_prev).max(0.0);
                        if a.targets[m].trem_cross_count >= 1 {
                            let dt_h = (t_cross - a.targets[m].trem_last_cross_t) as f32;
                            if dt_h >= 0.015 && dt_h <= 0.35 {
                                if a.targets[m].trem_half_period_ema <= 0.005 {
                                    a.targets[m].trem_half_period_ema = dt_h;
                                    a.targets[m].trem_period_jit_ema = 0.10;
                                } else {
                                    let rj = (dt_h - a.targets[m].trem_half_period_ema).abs()
                                        / a.targets[m].trem_half_period_ema.max(0.01);
                                    a.targets[m].trem_period_jit_ema =
                                        0.68 * a.targets[m].trem_period_jit_ema + 0.32 * rj;
                                    a.targets[m].trem_half_period_ema =
                                        0.65 * a.targets[m].trem_half_period_ema + 0.35 * dt_h;
                                }
                                a.targets[m].trem_cross_count =
                                    a.targets[m].trem_cross_count.saturating_add(1).min(64);
                            } else {
                                a.targets[m].trem_cross_count = 1;
                                a.targets[m].trem_half_period_ema = 0.0;
                            }
                        } else {
                            a.targets[m].trem_cross_count = 1;
                        }
                        a.targets[m].trem_last_cross_t = t_cross;
                        a.targets[m].trem_prev_sign = -ps;
                        a.targets[m].trem_max_q = r_disp.abs();
                    }
                }
                a.targets[m].trem_prev_q = r_disp;

                let tr_speed_f = (a.targets[m].vel_x.abs() + a.targets[m].vel_y.abs()) / s;
                let dfx = a.targets[m].disp_fista_x;
                let dfy = a.targets[m].disp_fista_y;
                let border_clamped_disp = dfx <= half_patch + 6.0 * s
                    || dfx >= (frame_w as f32) - half_patch - 6.0 * s
                    || dfy <= half_patch + 6.0 * s
                    || dfy >= (frame_h as f32) - half_patch - 6.0 * s;
                a.targets[m].freq_valid = if a.targets[m].coasting == 0
                    && !recent_strike_at_disp
                    && !border_clamped_disp
                    && a.targets[m].trem_cross_count >= 5
                    && a.targets[m].tremor_coherence >= 0.58
                    && a.targets[m].trem_period_jit_ema <= 0.28
                    && amp >= 0.42 * s
                    && (tr_speed_f < 1.85 || amp >= 0.85 * s)
                    && !(a.lab_mode != 1
                        && a.targets[m].retrace_dirs != 3
                        && a.targets[m].max_excursion2 >= (30.0 * s) * (30.0 * s))
                    && !(a.lab_mode == 2
                        && a.targets[m].max_excursion2 >= (30.0 * s) * (30.0 * s))
                {
                    1
                } else {
                    0
                };

                let lead_time_sec = disp_lag_sec + a.audio_out_latency_sec;
                let phi_now = wrap_pi(phi + a.targets[m].tremor_omega_ema * lead_time_sec);

                a.targets[m].prev_tremor_phase = prev_phi;
                a.targets[m].prev_pred_phase_now = a.targets[m].pred_phase_now;
                a.targets[m].tremor_r = r_disp;
                a.targets[m].tremor_h = hilb;
                a.targets[m].tremor_phase = phi;
                a.targets[m].pred_phase_now = phi_now;
                a.targets[m].tremor_amp = amp;
                a.targets[m].phase_valid = 1;
                a.targets[m].osc_regime = 1;
            } else {
                a.targets[m].phase_valid = 0;
                a.targets[m].freq_valid = 0;
                a.targets[m].osc_regime = 0;
                a.targets[m].tremor_r = r_disp;
                a.targets[m].tremor_h = 0.0;
                a.targets[m].tremor_amp = 0.0;
                a.targets[m].tremor_coherence *= 0.88;
                if !disp_valid || vis_d == 0 {
                    a.targets[m].tremor_freq_hz = 0.0;
                }
            }
        }

        if a.targets[m].confirmed != 0 {
            let pinned_bonus = if a.focus_target_id != 0 && a.targets[m].track_id == a.focus_target_id {
                10_000.0
            } else {
                0.0
            };
            let exc_px = wasm_sqrtf(a.targets[m].max_excursion2) / s;
            let speed_px = (a.targets[m].vel_x.abs() + a.targets[m].vel_y.abs()) / s;
            let pip_bonus = if a.pip_target_id != 0 && a.targets[m].track_id == a.pip_target_id {
                let inc_moving = speed_px >= 0.45
                    || a.targets[m].bridged_at_disp != 0
                    || a.targets[m].coasting <= 30;
                let mode1_unverified = a.lab_mode == 1
                    && a.targets[m].freq_valid == 0
                    && a.targets[m].pivot_valid == 0
                    && exc_px < 12.0;
                if mode1_unverified || (!inc_moving && a.lab_mode != 1) {
                    20.0
                } else if vis_d != 0 || a.targets[m].bridged_at_disp != 0 {
                    140.0 + if a.pip_hold_frames < 48 { 65.0 } else { 0.0 }
                } else if a.targets[m].coasting <= 30 {
                    95.0
                } else {
                    24.0
                }
            } else {
                0.0
            };
            let mode_bonus = match a.lab_mode {
                0 => {
                    let apex_b = if a.targets[m].apex_frames > 0.5 && exc_px >= 6.0 {
                        95.0
                    } else {
                        0.0
                    };
                    let bridge_b = if a.targets[m].bridged_at_disp != 0 && exc_px >= 6.0 {
                        85.0
                    } else {
                        0.0
                    };
                    apex_b + bridge_b
                }
                1 => {
                    let osc_b = if a.targets[m].phase_valid != 0 && a.targets[m].tremor_coherence > 0.30 {
                        85.0 * a.targets[m].tremor_coherence
                            + (a.targets[m].tremor_amp / s).min(12.0) * 4.0
                    } else {
                        0.0
                    };
                    let freq_b = if a.targets[m].freq_valid != 0 {
                        if a.targets[m].osc_regime == 2 || a.targets[m].pivot_valid > 0 {
                            420.0
                        } else {
                            60.0
                        }
                    } else if a.targets[m].osc_regime == 2 && exc_px >= 12.0 * s {
                        95.0
                    } else {
                        0.0
                    };
                    let piv_b = if a.targets[m].pivot_valid > 0 && exc_px >= 12.0 {
                        115.0 + exc_px.min(90.0) * 0.75
                    } else {
                        0.0
                    };
                    osc_b + freq_b + piv_b
                }
                _ => {
                    if exc_px >= 8.0 && (speed_px >= 0.55 || a.targets[m].strike_fired != 0) {
                        (a.targets[m].l1_impulse_mag / s).min(8.0) * 16.0
                            + if a.targets[m].strike_fired != 0 { 80.0 } else { 0.0 }
                    } else {
                        0.0
                    }
                }
            };
            let vis_bonus = if vis_d != 0 {
                45.0
            } else if a.targets[m].bridged_at_disp != 0 {
                40.0
            } else {
                0.0
            };
            let chroma_bonus = if (a.targets[m].anchor_sig & (1 << 31)) != 0 {
                22.0
            } else {
                0.0
            };
            let parked_penalty = if a.lab_mode != 1
                && speed_px < 0.55
                && a.targets[m].bridged_at_disp == 0
                && a.targets[m].strike_fired == 0
                && a.targets[m].apex_frames <= 0.1
            {
                -95.0
            } else {
                0.0
            };
            let motion_bonus = exc_px.min(140.0) * 0.65
                + speed_px.min(24.0) * 3.8
                + chroma_bonus
                + parked_penalty
                + if exc_px < 5.5 && a.targets[m].tremor_amp < 0.35 * s {
                    -70.0
                } else {
                    0.0
                };
            let score = pinned_bonus
                + (a.targets[m].age.min(36) as f32)
                + vis_bonus
                + a.targets[m].vis_eca_ema.min(0.25) * 90.0
                + pip_bonus
                + motion_bonus
                + mode_bonus;
            if score > best_primary_score {
                best_primary_score = score;
                primary_idx = m as i32;
            }
        }
    }

    if a.focus_target_id != 0 {
        let mut pin_alive = false;
        for m in 0..MAX_TARGETS {
            if a.targets[m].active != 0
                && a.targets[m].confirmed != 0
                && a.targets[m].track_id == a.focus_target_id
            {
                pin_alive = true;
                break;
            }
        }
        if !pin_alive {
            for di in 0..MAX_DORMANT {
                if a.dormant[di].active != 0
                    && a.dormant[di].track_id == a.focus_target_id
                    && f_idx.saturating_sub(a.dormant[di].lost_frame) <= DORMANT_MAX_AGE
                {
                    pin_alive = true;
                    break;
                }
            }
        }
        if !pin_alive {
            a.focus_target_id = 0;
        }
    }

    if any_consumed_stimulus || (a.stimulus_armed != 0 && f_idx.saturating_sub(a.stimulus_frame) > 90) {
        a.stimulus_armed = 0;
    }

    // ------------------------------------------------------------------------
    // 7. FISTA-Stabilized Eulerian Temporal-Contrast PiP (All 16 Slots + Primary)
    // ------------------------------------------------------------------------
    let mut strobe_fired = 0.0f32;
    let pip_center_off = 0.5 * (pip_dim as f32) - 0.5;
    let pip_bytes = pip_dim * pip_dim * 4;
    let gain = a.eulerian_gain;
    let max_pip_x = (frame_w.saturating_sub(pip_dim + 2)) as f32;
    let max_pip_y = (frame_h.saturating_sub(pip_dim + 2)) as f32;

    let dense_cap_odd = dense_swarm_hero_cap_active(a) && (f_idx & 1) != 0;
    let cold_dense_pip_defer =
        a.micro_swarm.confirmed_count >= 960 && a.inspected_micro_tid_pin == 0 && a.ring_cursor < 9;
    let mut conf_hero_cnt = 0u32;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 && a.targets[m].confirmed != 0 {
            conf_hero_cnt += 1;
        }
    }
    let multi_hero_stagger = conf_hero_cnt >= 4;
    for m in 0..MAX_TARGETS {
        let mbase = m * 8;
        if a.targets[m].active == 0 || a.targets[m].confirmed == 0 {
            a.all_pip_tid[m] = 0;
            for q in 0..8 {
                a.slot_meta[mbase + q] = 0.0;
            }
            continue;
        }
        let not_born_at_disp = !((a.telemetry[32 + m * 56] > 0.5)
            && (a.telemetry[32 + m * 56 + 1] - (a.targets[m].track_id as f32)).abs() < 0.25)
            && a.ring_cursor >= (eff_k as u64)
            && f_idx.wrapping_sub(a.targets[m].birth_frame) < (eff_k as u32)
            && a.targets[m].dormant_reid == 0
            && a.targets[m].is_ballistic == 0
            && a.targets[m].disp_vis < 0.5
            && a.targets[m].bridged_at_disp == 0;
        let exited_past_disp = if not_born_at_disp {
            true
        } else if a.targets[m].coasting >= 2
            && a.targets[m].disp_vis < 0.5
            && a.targets[m].bridged_at_disp == 0
        {
            let cf_now = a.targets[m].coasting as f32;
            let (ex_now_x, ex_now_y, _, _) = predict_curved_step(
                a.targets[m].entry_x,
                a.targets[m].entry_y,
                a.targets[m].entry_vx,
                a.targets[m].entry_vy,
                a.targets[m].entry_omega_f,
                a.targets[m].grav_ay_frame,
                a.targets[m].is_ballistic,
                cf_now.min(3.0),
            );
            ex_now_x < half_patch - 3.0 * s
                || ex_now_x > (frame_w as f32) - half_patch + 3.0 * s
                || ex_now_y < half_patch - 3.0 * s
                || ex_now_y > (frame_h as f32) - half_patch + 3.0 * s
        } else {
            false
        };
        if !exited_past_disp {
            let sec_pip_mask = if conf_hero_cnt >= 7 { 7usize } else { 3usize };
            let secondary_stagger =
                multi_hero_stagger && ((((f_idx as usize).wrapping_add(m)) & sec_pip_mask) != 0);
            let skip_occ_bg_pip = cold_dense_pip_defer
                || ((m as i32) != primary_idx
                    && a.all_pip_tid[m] == a.targets[m].track_id
                    && ((a.targets[m].coasting >= 2 && a.targets[m].disp_vis < 0.5)
                        || dense_cap_odd
                        || secondary_stagger));
            if skip_occ_bg_pip {
                a.slot_meta[mbase] = a.targets[m].age as f32;
                a.slot_meta[mbase + 1] = a.targets[m].coasting as f32;
                a.slot_meta[mbase + 2] = a.targets[m].vis_eca_ema;
                a.slot_meta[mbase + 3] = a.targets[m].reacq_count as f32;
                a.slot_meta[mbase + 4] = if a.targets[m].freq_valid != 0 {
                    a.targets[m].tremor_freq_hz
                } else {
                    0.0
                };
                a.slot_meta[mbase + 5] = a.targets[m].tremor_amp;
                a.slot_meta[mbase + 6] = a.targets[m].fista_duality_gap;
                a.slot_meta[mbase + 7] = wasm_sqrtf(a.targets[m].max_excursion2);
            } else {
                update_slot_pip_and_meta(
                    a,
                    m,
                    mbase,
                    slot_disp,
                    frame_w,
                    frame_h,
                    pip_dim,
                    pip_center_off,
                    max_pip_x,
                    max_pip_y,
                    gain,
                );
            }
        } else {
            a.all_pip_tid[m] = 0;
            for q in 0..8 {
                a.slot_meta[mbase + q] = 0.0;
            }
        }
    }

    if primary_idx >= 0 {
        let pm = primary_idx as usize;
        let tid = a.targets[pm].track_id;
        if cold_dense_pip_defer {
            a.pip_target_id = 0;
            a.pip_hold_frames = 0;
        } else {
            if a.pip_target_id == tid {
                a.pip_hold_frames = a.pip_hold_frames.saturating_add(1);
            } else {
                a.pip_target_id = tid;
                a.pip_hold_frames = 1;
            }

            if a.all_pip_tid[pm] == tid && tid != 0 {
                a.pip_rgba[..pip_bytes].copy_from_slice(&a.all_pip_rgba[pm][..pip_bytes]);
            } else {
                let fx = a.targets[pm].disp_fista_x;
                let fy = a.targets[pm].disp_fista_y;
                let reset_ref = a.pip_hold_frames <= 1;
                let pip_x0 = (fx - pip_center_off).clamp(0.0, max_pip_x);
                let pip_y0 = (fy - pip_center_off).clamp(0.0, max_pip_y);
                if !unsafe {
                    render_eulerian_pip_fast_simd128(
                        &a.rgba_ring[slot_disp],
                        frame_w,
                        frame_h,
                        pip_x0,
                        pip_y0,
                        pip_dim,
                        gain,
                        reset_ref,
                        &mut a.pip_ref_luma,
                        &mut a.pip_rgba,
                    )
                } {
                    for py in 0..pip_dim {
                        for px in 0..pip_dim {
                            let sx = pip_x0 + (px as f32);
                            let sy = pip_y0 + (py as f32);
                            let (r, g, b, luma) =
                                sample_rgba_bilinear(&a.rgba_ring[slot_disp], frame_w, frame_h, sx, sy);
                            let idx = py * pip_dim + px;
                            if reset_ref {
                                a.pip_ref_luma[idx] = luma;
                            }
                            let dl = luma - a.pip_ref_luma[idx];
                            a.pip_ref_luma[idx] = 0.80 * a.pip_ref_luma[idx] + 0.20 * luma;
                            let amp_l = gain * dl;
                            let shear_pos = amp_l.max(0.0);
                            let shear_neg = (-amp_l).max(0.0);
                            let out_r = (r + amp_l * 0.70 + shear_neg * 0.65).clamp(0.0, 1.0);
                            let out_g = (g + amp_l * 0.80 + shear_pos * 0.45).clamp(0.0, 1.0);
                            let out_b = (b + amp_l * 0.70 + shear_pos * 0.85).clamp(0.0, 1.0);
                            let rgba_off = idx * 4;
                            a.pip_rgba[rgba_off] = (out_r * 255.0) as u8;
                            a.pip_rgba[rgba_off + 1] = (out_g * 255.0) as u8;
                            a.pip_rgba[rgba_off + 2] = (out_b * 255.0) as u8;
                            a.pip_rgba[rgba_off + 3] = 255;
                        }
                    }
                }
            }
        }

        // Present-time Phase-Locked Strobe trigger (using forward-predicted phi_now)
        if a.targets[pm].phase_valid != 0 && a.targets[pm].prev_phase_valid != 0 {
            let prev_err = wrap_pi(a.targets[pm].prev_pred_phase_now - a.phase_offset_rad);
            let curr_err = wrap_pi(a.targets[pm].pred_phase_now - a.phase_offset_rad);
            let min_strobe_gap = ((0.045 * fps) as u32).max(3);
            let strobe_ready =
                a.has_strobed == 0 || f_idx.saturating_sub(a.last_strobe_frame) >= min_strobe_gap;

            if prev_err < 0.0
                && curr_err >= 0.0
                && (curr_err - prev_err) < 1.8
                && a.targets[pm].tremor_amp > 0.025 * s
                && strobe_ready
            {
                strobe_fired = 1.0;
                a.last_strobe_frame = f_idx;
                a.has_strobed = 1;
            }
        }
    } else {
        a.pip_hold_frames = 0;
        for i in 0..(pip_dim * pip_dim) {
            let off = i * 4;
            a.pip_rgba[off] = 8;
            a.pip_rgba[off + 1] = 14;
            a.pip_rgba[off + 2] = 22;
            a.pip_rgba[off + 3] = 255;
        }
    }

    // ------------------------------------------------------------------------
    // 8. Populate Zero-Copy Flat Telemetry Array (Confirmed Tracks Only)
    // ------------------------------------------------------------------------
    let mut active_count = 0u32;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 && a.targets[m].confirmed != 0 {
            active_count += 1;
            let tenure = f_idx
                .saturating_sub(a.targets[m].orig_birth_frame)
                .max(a.targets[m].age);
            if tenure > a.longest_tenure_frames {
                a.longest_tenure_frames = tenure;
                a.longest_tenure_tid = a.targets[m].track_id;
            }
            if a.targets[m].sustained_10s_counted == 0 && tenure >= 600 {
                a.targets[m].sustained_10s_counted = 1;
                a.sustained_10s_count = a.sustained_10s_count.saturating_add(1);
            }
            if a.targets[m].sustained_60s_counted == 0 && tenure >= 3600 {
                a.targets[m].sustained_60s_counted = 1;
                a.sustained_60s_count = a.sustained_60s_count.saturating_add(1);
            }
        }
    }
    if active_count > a.peak_simultaneous_confirmed {
        a.peak_simultaneous_confirmed = active_count;
    }

    a.telemetry[0] = f_idx as f32;
    a.telemetry[1] = eff_k as f32;
    a.telemetry[2] = slot_disp as f32;
    a.telemetry[3] = active_count as f32;
    a.telemetry[4] = a.clutter_db;
    a.telemetry[5] = a.padic_id_swaps as f32;
    a.telemetry[6] = a.kalman_id_swaps as f32;
    a.telemetry[7] = a.fiedler_lambda2;
    a.telemetry[8] = primary_idx as f32;

    if primary_idx >= 0 {
        let pm = primary_idx as usize;
        a.telemetry[9] = a.targets[pm].tremor_r;
        a.telemetry[10] = a.targets[pm].tremor_h;
        a.telemetry[11] = a.targets[pm].tremor_phase;
        a.telemetry[12] = a.targets[pm].tremor_amp;
        a.telemetry[13] = if a.targets[pm].freq_valid != 0 {
            a.targets[pm].tremor_freq_hz
        } else {
            0.0
        };
        a.telemetry[19] = a.targets[pm].saccade_latency_ms;
        a.telemetry[25] = a.targets[pm].fista_duality_gap;
        a.telemetry[26] = a.targets[pm].phase_valid as f32;
        a.telemetry[27] = a.targets[pm].pred_phase_now;
        a.telemetry[28] = a.targets[pm].step_settle_ms;
    } else {
        a.telemetry[9] = 0.0;
        a.telemetry[10] = 0.0;
        a.telemetry[11] = 0.0;
        a.telemetry[12] = 0.0;
        a.telemetry[13] = 0.0;
        a.telemetry[19] = 0.0;
        a.telemetry[25] = 0.0;
        a.telemetry[26] = 0.0;
        a.telemetry[27] = 0.0;
        a.telemetry[28] = 0.0;
    }

    a.telemetry[14] = strobe_fired;
    a.telemetry[15] = perc_mask as f32;
    a.telemetry[16] = perc_lead;
    a.telemetry[17] = perc_zone;
    a.telemetry[18] = perc_vel;
    a.telemetry[20] = (core::mem::size_of::<Arena>()) as f32;
    a.telemetry[21] = a.num_peaks as f32;
    a.telemetry[22] = a.clutter_db_valid as f32;
    a.telemetry[23] = a.clutter_eval_ratio;
    a.telemetry[24] = a.fiedler_gap;
    a.telemetry[29] = a.stimulus_armed as f32;
    a.telemetry[30] = frame_w as f32;
    a.telemetry[31] = frame_h as f32;

    for m in 0..MAX_TARGETS {
        let base = 32 + m * 56;
        let ebase = TELEM_EXCUR_START + m * 8;
        let kbase = TELEM_KINETIC_START + m * 8;
        let not_born_at_disp = !((a.telemetry[32 + m * 56] > 0.5)
            && (a.telemetry[32 + m * 56 + 1] - (a.targets[m].track_id as f32)).abs() < 0.25)
            && a.ring_cursor >= (eff_k as u64)
            && f_idx.wrapping_sub(a.targets[m].birth_frame) < (eff_k as u32)
            && a.targets[m].dormant_reid == 0
            && a.targets[m].is_ballistic == 0
            && a.targets[m].disp_vis < 0.5
            && a.targets[m].bridged_at_disp == 0;
        let exited_past_disp = if not_born_at_disp {
            true
        } else if a.targets[m].coasting >= 2
            && a.targets[m].disp_vis < 0.5
            && a.targets[m].bridged_at_disp == 0
        {
            let cf_now = a.targets[m].coasting as f32;
            let (ex_now_x, ex_now_y, _, _) = predict_curved_step(
                a.targets[m].entry_x,
                a.targets[m].entry_y,
                a.targets[m].entry_vx,
                a.targets[m].entry_vy,
                a.targets[m].entry_omega_f,
                a.targets[m].grav_ay_frame,
                a.targets[m].is_ballistic,
                cf_now.min(3.0),
            );
            ex_now_x < half_patch - 3.0 * s
                || ex_now_x > (frame_w as f32) - half_patch + 3.0 * s
                || ex_now_y < half_patch - 3.0 * s
                || ex_now_y > (frame_h as f32) - half_patch + 3.0 * s
        } else {
            false
        };
        if a.targets[m].active == 0 || a.targets[m].confirmed == 0 || exited_past_disp {
            a.telemetry[base] = 0.0;
            for q in 0..8 {
                a.telemetry[ebase + q] = 0.0;
                a.telemetry[kbase + q] = 0.0;
            }
            a.telemetry[kbase + 6] = -1.0e6;
            a.telemetry[kbase + 7] = -1.0e6;
            continue;
        }
        let sig = a.targets[m].curr_sig;
        a.telemetry[base] = 1.0;
        a.telemetry[base + 1] = a.targets[m].track_id as f32;
        a.telemetry[base + 2] = a.targets[m].disp_fista_x;
        a.telemetry[base + 3] = a.targets[m].disp_fista_y;
        a.telemetry[base + 4] = a.targets[m].disp_raw_x;
        a.telemetry[base + 5] = a.targets[m].disp_raw_y;
        a.telemetry[base + 6] = a.kalman[m].ring_x[slot_disp];
        a.telemetry[base + 7] = a.kalman[m].ring_y[slot_disp];
        a.telemetry[base + 8] = a.targets[m].pos_x;
        a.telemetry[base + 9] = a.targets[m].pos_y;
        a.telemetry[base + 10] = a.targets[m].disp_vis;
        a.telemetry[base + 11] = a.targets[m].bridged_at_disp as f32;
        a.telemetry[base + 12] = ((sig >> 16) & 0xFFFF) as f32;
        a.telemetry[base + 13] = (sig & 0xFFFF) as f32;
        a.telemetry[base + 14] = a.targets[m].last_padic_val as f32;
        a.telemetry[base + 15] = a.targets[m].apex_x;
        a.telemetry[base + 16] = a.targets[m].apex_y;
        a.telemetry[base + 17] = a.targets[m].apex_frames;
        a.telemetry[base + 18] = a.targets[m].l1_impulse_mag;
        a.telemetry[base + 19] = a.fiedler_vec[m];

        for u in 0..RING_LEN {
            let su = (slot_in + 1 + u) % RING_LEN;
            a.telemetry[base + 20 + u] = a.targets[m].fista_x[u];
            a.telemetry[base + 32 + u] = a.targets[m].fista_y[u];
            a.telemetry[base + 44 + u] = if a.targets[m].ring_vis[su] != 0 { 1.0 } else { 0.0 };
        }

        a.telemetry[ebase] = a.targets[m].strike_fired as f32;
        a.telemetry[ebase + 1] = a.targets[m].strike_zone as f32;
        a.telemetry[ebase + 2] = a.targets[m].strike_lead_frames;
        a.telemetry[ebase + 3] = a.targets[m].strike_vel;
        a.telemetry[ebase + 4] = a.targets[m].strike_x;
        a.telemetry[ebase + 5] = a.targets[m].strike_y;
        a.telemetry[ebase + 6] = a.targets[m].tremor_coherence;
        a.telemetry[ebase + 7] = fast_atan2f(a.targets[m].axis_y, a.targets[m].axis_x);

        a.telemetry[kbase] = a.targets[m].heading_deg;
        a.telemetry[kbase + 1] = a.targets[m].speed_px_s;
        a.telemetry[kbase + 2] = a.targets[m].ang_vel_deg_s;
        a.telemetry[kbase + 3] = a.targets[m].strike_deflect_deg;
        a.telemetry[kbase + 4] = a.targets[m].strike_restitution;
        a.telemetry[kbase + 5] = a.targets[m].strike_in_deg;
        if a.targets[m].pivot_valid > 0 {
            a.telemetry[kbase + 6] = a.targets[m].pivot_x;
            a.telemetry[kbase + 7] = a.targets[m].pivot_y;
        } else {
            a.telemetry[kbase + 6] = -1.0e6;
            a.telemetry[kbase + 7] = -1.0e6;
        }
    }

    a.telemetry[TELEM_GLOBAL_START] = a.scene_gravity_px_s2;
    a.telemetry[TELEM_GLOBAL_START + 1] = a.scene_gravity_conf;
    a.telemetry[TELEM_GLOBAL_START + 2] = a.scatter_deg;
    a.telemetry[TELEM_GLOBAL_START + 3] = a.scatter_x;
    a.telemetry[TELEM_GLOBAL_START + 4] = a.scatter_y;
    a.telemetry[TELEM_GLOBAL_START + 5] = a.scatter_th1_deg;
    a.telemetry[TELEM_GLOBAL_START + 6] = a.scatter_th2_deg;
    a.telemetry[TELEM_GLOBAL_START + 7] = a.scatter_event_frame as f32;

    let disp_f_out = f_idx.wrapping_sub(eff_k as u32);
    let disp_lag_ms = ((a.hist_t_sec[(f_idx as usize) & HIST_MASK]
        - a.hist_t_sec[(disp_f_out as usize) & HIST_MASK])
        .max(0.0)
        * 1000.0) as f32;
    a.telemetry[TELEM_GLOBAL_START + 8] = disp_lag_ms;
    a.telemetry[TELEM_GLOBAL_START + 9] = a.cam_motion_state as f32;
    a.telemetry[TELEM_GLOBAL_START + 10] = a.cam_pan_speed_px_s;
    a.telemetry[TELEM_GLOBAL_START + 11] = a.scene_gravity_arcs as f32;
    a.telemetry[TELEM_GLOBAL_START + 12] = if primary_idx >= 0 {
        a.targets[primary_idx as usize].freq_valid as f32
    } else {
        0.0
    };
    a.telemetry[TELEM_GLOBAL_START + 13] = a.last_strike_tid as f32;
    a.telemetry[TELEM_GLOBAL_START + 14] = a.last_strike_frame as f32;
    a.telemetry[TELEM_GLOBAL_START + 15] = a.discont_count as f32;

    let mut dormant_active_count = 0u32;
    for di in 0..MAX_DORMANT {
        if a.dormant[di].active != 0
            && f_idx.saturating_sub(a.dormant[di].lost_frame) <= DORMANT_MAX_AGE
        {
            dormant_active_count += 1;
        }
    }
    a.telemetry[TELEM_PERSIST_START] = a.next_track_id.saturating_sub(a.reset_base_track_id) as f32;
    a.telemetry[TELEM_PERSIST_START + 1] = a.coast_reacq_events as f32;
    a.telemetry[TELEM_PERSIST_START + 2] = a.dormant_reid_events as f32;
    a.telemetry[TELEM_PERSIST_START + 3] = a.peak_simultaneous_confirmed as f32;
    a.telemetry[TELEM_PERSIST_START + 4] = a.longest_tenure_frames as f32;
    a.telemetry[TELEM_PERSIST_START + 5] = a.longest_tenure_tid as f32;
    a.telemetry[TELEM_PERSIST_START + 6] = a.sustained_10s_count as f32;
    a.telemetry[TELEM_PERSIST_START + 7] = a.sustained_60s_count as f32;
    a.telemetry[TELEM_PERSIST_START + 8] = dormant_active_count as f32;
    a.telemetry[TELEM_PERSIST_START + 9] = a.last_reid_gap_frames as f32;

    for p in 0..MAX_TARGETS {
        let pbase = TELEM_PEAKS_START + p * 4;
        if p < a.num_peaks && a.peaks[p].valid != 0 {
            a.telemetry[pbase] = a.peaks[p].x;
            a.telemetry[pbase + 1] = a.peaks[p].y;
            a.telemetry[pbase + 2] = a.peaks[p].energy;
            a.telemetry[pbase + 3] = ((a.peaks[p].sig >> 16) & 0xFFFF) as f32;
        } else {
            a.telemetry[pbase] = -1.0;
            a.telemetry[pbase + 1] = -1.0;
            a.telemetry[pbase + 2] = 0.0;
            a.telemetry[pbase + 3] = 0.0;
        }
    }

    let prev_cons_seq = a.collision_state.conservation_live_event_seq();
    let hero_phys_ran =
        update_physics_and_telemetry(a, f_idx, frame_w, frame_h, s, fps, eff_k, safe_dt);
    a.ingest_ctrl[CTRL_HERO_PHYS_RAN] = if hero_phys_ran { 1 } else { 0 };
    let new_cons_seq = a.collision_state.conservation_live_event_seq();
    let cons_fresh = new_cons_seq != prev_cons_seq && new_cons_seq != 0;

    let need_px = frame_w * frame_h;
    let rgba_bytes = need_px * 4;
    let swarm_active = if a.swarm_mode_cached != 0 {
        a.swarm_telem_zeroed = 0;
        let meta_in = a.ring_meta[slot_in];
        if (meta_in.flags & INGEST_FLAG_PTS_KNOWN) != 0 {
            a.micro_swarm.set_ingest_clock(
                meta_in.t_src_us_lo,
                meta_in.t_src_us_hi,
                meta_in.continuity_epoch,
            );
        } else {
            a.micro_swarm.ingest_clock_valid = 0;
            a.micro_swarm.ingest_continuity_epoch = meta_in.continuity_epoch;
            if a.micro_swarm.metrology.metrology_ctrl[1] != meta_in.continuity_epoch {
                a.micro_swarm.invalidate_continuity();
                a.micro_swarm.metrology.metrology_ctrl[1] = meta_in.continuity_epoch;
            }
        }
        soa_tracks::soa_step(
            &mut a.micro_swarm,
            &a.luma[..need_px],
            &a.bg_luma[..need_px],
            &a.eca_res[..need_px],
            &a.rgba_ring[slot_in][..rgba_bytes],
            &a.cc_energy[..grid_w * grid_h],
            frame_w,
            frame_h,
            safe_dt,
            f_idx,
            a.last_pan_dx_f,
            a.last_pan_dy_f,
            a.last_scene_cut != 0,
        )
    } else {
        if a.swarm_telem_zeroed == 0 {
            a.micro_swarm.reset_active_keep_mode();
            a.micro_swarm.cut_cooldown = 0;
            a.micro_swarm.probe_backoff = 0;
            a.micro_swarm.probe_fail_streak = 0;
            a.micro_swarm.swarm_telem[1..16].fill(0.0);
            a.swarm_telem_zeroed = 1;
        }
        false
    };
    if swarm_active {
        continuum_physics::continuum_step(
            &mut a.micro_swarm,
            &mut a.continuum,
            f_idx,
            frame_w,
            frame_h,
            safe_dt,
        );
        a.continuum_was_active = 1;
        if a.inspected_micro_tid_pin != 0 {
            refresh_hero_pin_bridge(a, frame_w, frame_h, scale_px, s);
        }
    } else if a.continuum_was_active != 0 {
        a.continuum.reset();
        a.micro_swarm.swarm_telem[16..soa_tracks::SWARM_TELEM_LEN].fill(0.0);
        a.continuum_was_active = 0;
        a.inspected_hero_tid = 0;
        a.inspected_micro_tid_pin = 0;
    }
    if dense_swarm_hero_cap_active(a) {
        enforce_dense_swarm_hero_cap(a, DENSE_SWARM_HERO_CAP, f_idx, s);
    }
    if a.micro_swarm.get_micro_metrology_mode() != 0
        || a.ingest_ctrl[CTRL_AUTO_HERO_MASK] != 0
        || a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS] != 0
    {
        reconcile_wave1_hero_promotions(a, f_idx, frame_w, frame_h, scale_px, s, swarm_active);
    }

    let bos_ran = if a.bos.get_mode() != bos::BOS_MODE_OFF {
        let meta = a.ring_meta[slot_in];
        let ext_scene_cut =
            a.last_scene_cut != 0 || (meta.frame_gap_flag & FRAME_GAP_SCENE_CUT) != 0;
        let ext_pan_or_cut = ext_scene_cut || a.cam_motion_state != 0;
        a.bos.step(
            &a.luma[..need_px],
            frame_w,
            frame_h,
            meta.continuity_epoch,
            meta.source_epoch,
            meta.source_seq,
            meta.t_src_us_lo,
            meta.t_src_us_hi,
            meta.frame_gap_flag,
            ext_pan_or_cut,
            ext_scene_cut,
        )
    } else {
        false
    };

    let (hodge_ran, hodge_grad_held) = run_wave2_post_tick(
        a,
        f_idx,
        slot_in,
        frame_w,
        frame_h,
        grid_w,
        grid_h,
        swarm_active,
    );

    finalize_wave1_stamps(a, f_idx, slot_in, slot_disp, swarm_active, bos_ran);
    finalize_wave2_stamps(a, slot_in, swarm_active, hodge_ran, hodge_grad_held, cons_fresh, new_cons_seq);

    a.ring_cursor = a.ring_cursor.wrapping_add(1);
    a.frame_idx = f_idx.wrapping_add(1);
    slot_disp as u32
}

#[inline(never)]
fn run_wave2_post_tick(
    a: &mut Arena,
    f_idx: u32,
    slot_in: usize,
    frame_w: usize,
    frame_h: usize,
    grid_w: usize,
    grid_h: usize,
    swarm_active: bool,
) -> (bool, bool) {
    let need_px = frame_w * frame_h;
    if a.vibro.get_mode() != vibrometry::RS_MODE_OFF {
        let meta = a.ring_meta[slot_in];
        let vmeta = vibrometry::VibroFrameMeta {
            flags: meta.flags,
            continuity_epoch: meta.continuity_epoch,
            t_src_us: ((meta.t_src_us_hi as u64) << 32) | (meta.t_src_us_lo as u64),
            src_w: meta.src_w,
            src_h: meta.src_h,
            crop_x0: meta.crop_x0,
            crop_y0: meta.crop_y0,
            crop_w: meta.crop_w,
            crop_h: meta.crop_h,
            orientation: meta.orientation,
            scan_dir: meta.scan_dir,
            native_1to1: meta.native_1to1_active,
            gap_flag: meta.frame_gap_flag,
            cam_motion_state: a.cam_motion_state,
        };
        a.vibro.ingest_and_solve_frame(
            &a.luma[..need_px],
            &a.row_luma_means[slot_in],
            frame_w,
            frame_h,
            &vmeta,
        );
    }

    let (grad_valid, grad_held, _) = a.continuum.grad_product_status();
    let _ = (grid_w, grid_h);
    let hodge_ran = if swarm_active && a.hodge.get_mode() != hodge::HODGE_MODE_OFF {
        if grad_valid && !grad_held {
            let gdx = (frame_w as f32) / (hodge::HODGE_W as f32);
            let gdy = (frame_h as f32) / (hodge::HODGE_H as f32);
            a.hodge.solve_from_continuum(
                &a.continuum.export_field,
                &a.continuum.cell_valid_mask,
                gdx,
                gdy,
                f_idx,
            )
        } else if !grad_valid {
            a.hodge.reset_keep_mode();
            false
        } else {
            false
        }
    } else {
        if a.hodge.get_mode() != hodge::HODGE_MODE_OFF && !swarm_active {
            a.hodge.reset_keep_mode();
        }
        false
    };
    (hodge_ran, grad_held)
}

#[inline(never)]
fn finalize_wave2_stamps(
    a: &mut Arena,
    slot_in: usize,
    swarm_active: bool,
    hodge_ran: bool,
    hodge_grad_held: bool,
    cons_fresh: bool,
    cons_seq: u32,
) {
    let meta = a.ring_meta[slot_in];
    let cont_ep = meta.continuity_epoch;
    a.wave2_telem[0] = WAVE2_ABI_VERSION;
    a.wave2_telem[1] = core::mem::size_of::<Arena>() as u32;
    a.wave2_telem[2] = wave2_enabled_bits(a);
    a.wave2_telem[3] = cont_ep;

    // 0 (word 16): RS Vibrometry & Welch Coherence
    let vibro_base = WAVE2_PRODUCT_STAMPS_START + PROD_W2_VIBRO * STAMP_STRIDE;
    if a.vibro.get_mode() != vibrometry::RS_MODE_OFF
        && a.vibro.get_state() == vibrometry::RS_STATE_READY
    {
        if a.vibro.fresh_solve != 0 {
            set_stamp_fresh(&mut a.wave2_telem, vibro_base, &meta);
        } else {
            hold_or_invalidate_stamp(&mut a.wave2_telem, vibro_base, cont_ep);
        }
    } else {
        set_stamp_invalid(&mut a.wave2_telem, vibro_base);
    }

    // 1 (word 24): 40x22 Hodge Decomposition (live_ready only; never spoofed by injected test solves)
    let hodge_base = WAVE2_PRODUCT_STAMPS_START + PROD_W2_HODGE * STAMP_STRIDE;
    let (grad_valid, _, _) = a.continuum.grad_product_status();
    if swarm_active
        && a.hodge.get_mode() != hodge::HODGE_MODE_OFF
        && a.hodge.live_ready != 0
        && a.hodge.get_state() == hodge::HODGE_STATE_READY
    {
        if hodge_ran && !hodge_grad_held {
            set_stamp_fresh(&mut a.wave2_telem, hodge_base, &meta);
        } else if grad_valid {
            if (a.wave2_telem[hodge_base] & STAMP_VALID) != 0
                && a.wave2_telem[hodge_base + 1] == cont_ep
            {
                hold_or_invalidate_stamp(&mut a.wave2_telem, hodge_base, cont_ep);
            } else if hodge_ran && hodge_grad_held {
                // Mirror originating observation from PROD_GRAD if Hodge solved from held gradient
                let grad_base = WAVE1_PRODUCT_STAMPS_START + PROD_GRAD * STAMP_STRIDE;
                if (a.wave1_telem[grad_base] & STAMP_VALID) != 0
                    && a.wave1_telem[grad_base + 1] == cont_ep
                {
                    for k in 0..STAMP_STRIDE {
                        a.wave2_telem[hodge_base + k] = a.wave1_telem[grad_base + k];
                    }
                    a.wave2_telem[hodge_base] |= STAMP_HELD;
                } else {
                    set_stamp_invalid(&mut a.wave2_telem, hodge_base);
                }
            } else {
                set_stamp_invalid(&mut a.wave2_telem, hodge_base);
            }
        } else {
            set_stamp_invalid(&mut a.wave2_telem, hodge_base);
        }
    } else {
        set_stamp_invalid(&mut a.wave2_telem, hodge_base);
    }

    // 2 (word 32): Organ 5 Live Occluded-Pair Conservation (never spoofed by staged test calls)
    let cons_base = WAVE2_PRODUCT_STAMPS_START + PROD_W2_CONSERVATION * STAMP_STRIDE;
    if a.lab_mode == 2 && cons_seq != 0 {
        if cons_fresh {
            set_stamp_fresh(&mut a.wave2_telem, cons_base, &meta);
        } else {
            hold_or_invalidate_stamp(&mut a.wave2_telem, cons_base, cont_ep);
        }
    } else {
        set_stamp_invalid(&mut a.wave2_telem, cons_base);
    }
}

#[inline(never)]
unsafe fn release_wave1_owned_auto_heroes(a: &mut Arena, f_idx: u32, s: f32) {
    let auto_mask = a.ingest_ctrl[CTRL_AUTO_HERO_MASK];
    for m in 0..MAX_TARGETS {
        let bit_m = 1u32 << m;
        if ((auto_mask & bit_m) != 0 || a.targets[m].wave1_auto_owned != 0)
            && a.targets[m].active != 0
        {
            let htid = a.targets[m].track_id;
            let pinned_by_op = (a.focus_target_id != 0 && htid == a.focus_target_id)
                || (a.inspected_hero_tid != 0 && htid == a.inspected_hero_tid);
            if pinned_by_op {
                a.targets[m].wave1_auto_owned = 0;
            } else if a.targets[m].wave1_auto_owned != 0 {
                retire_target_slot_full(a, m, f_idx, s);
            }
        }
    }
    a.ingest_ctrl[CTRL_AUTO_HERO_MASK] = 0;
    a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS] = 0;
    for k in 0..DENSE_SWARM_HERO_CAP {
        a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k] = 0;
    }
    a.micro_swarm.sync_bound_hero_tids(&[]);
}

#[inline(never)]
unsafe fn reconcile_wave1_hero_promotions(
    a: &mut Arena,
    f_idx: u32,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
    swarm_active: bool,
) {
    if !swarm_active || a.micro_swarm.get_micro_metrology_mode() == 0 {
        release_wave1_owned_auto_heroes(a, f_idx, s);
        return;
    }

    let cand_cnt =
        (a.micro_swarm.hero_candidates_slice()[0] as usize).min(soa_tracks::MAX_HERO_CANDIDATES);
    let mut bound_tids = [0u32; DENSE_SWARM_HERO_CAP];
    let mut bound_len = 0usize;

    // 1. Validate existing auto-promoted Hero bindings and release hysteresis exits
    let mut auto_mask = a.ingest_ctrl[CTRL_AUTO_HERO_MASK];
    let mut slots_packed = a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS];
    for k in 0..DENSE_SWARM_HERO_CAP {
        let shift = k * 8;
        let raw_slot = ((slots_packed >> shift) & 0xFF) as usize;
        let mtid = a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k];
        if raw_slot == 0 || mtid == 0 || (raw_slot - 1) >= MAX_TARGETS {
            a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k] = 0;
            slots_packed &= !(0xFFu32 << shift);
            continue;
        }
        let m = raw_slot - 1;
        let bit_m = 1u32 << m;
        if a.targets[m].active == 0
            || a.targets[m].wave1_auto_owned == 0
            || a.targets[m].track_id == 0
            || (auto_mask & bit_m) == 0
        {
            a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k] = 0;
            slots_packed &= !(0xFFu32 << shift);
            auto_mask &= !bit_m;
            if a.targets[m].active != 0 && a.targets[m].wave1_auto_owned != 0 {
                a.targets[m].wave1_auto_owned = 0;
            }
            continue;
        }
        let htid = a.targets[m].track_id;
        let pinned_by_op = (a.focus_target_id != 0 && htid == a.focus_target_id)
            || (a.inspected_hero_tid != 0 && htid == a.inspected_hero_tid);
        if pinned_by_op {
            a.targets[m].wave1_auto_owned = 0;
            auto_mask &= !bit_m;
            a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k] = 0;
            slots_packed &= !(0xFFu32 << shift);
            continue;
        }
        let mut still_cand = false;
        {
            let cand_slice = a.micro_swarm.hero_candidates_slice();
            let mut c = 0usize;
            while c < cand_cnt {
                let base = soa_tracks::HERO_CAND_HEADER_LEN + c * soa_tracks::HERO_CAND_STRIDE;
                if cand_slice[base] == mtid && (cand_slice[base + 2] & 1) != 0 {
                    still_cand = true;
                    break;
                }
                c += 1;
            }
        }
        if still_cand {
            bound_tids[bound_len] = mtid;
            bound_len += 1;
        } else {
            retire_target_slot_full(a, m, f_idx, s);
            auto_mask = a.ingest_ctrl[CTRL_AUTO_HERO_MASK];
            slots_packed = a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS];
        }
    }
    a.ingest_ctrl[CTRL_AUTO_HERO_MASK] = auto_mask;
    a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS] = slots_packed;

    // 2. Count current Hero slots & enforce dense cap + operator pin priority
    let dense_cap = dense_swarm_hero_cap_active(a);
    let eff_cap = if dense_cap { DENSE_SWARM_HERO_CAP } else { MAX_TARGETS };
    let mut act_heroes = 0usize;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            act_heroes += 1;
        }
    }
    let skip_new_promo = dense_cap && a.inspected_micro_tid_pin != 0 && act_heroes >= eff_cap;
    let match_r2 = (20.0 * s) * (20.0 * s);

    if !skip_new_promo {
        let mut c = 0usize;
        while c < cand_cnt && bound_len < DENSE_SWARM_HERO_CAP {
            let (cand_tid, ti, flags) = {
                let cand_slice = a.micro_swarm.hero_candidates_slice();
                let base = soa_tracks::HERO_CAND_HEADER_LEN + c * soa_tracks::HERO_CAND_STRIDE;
                (cand_slice[base], cand_slice[base + 1] as usize, cand_slice[base + 2])
            };
            c += 1;
            if cand_tid == 0 || (flags & 1) == 0 || ti >= soa_tracks::MAX_MICRO {
                continue;
            }
            let mut already_bound = false;
            for b in 0..bound_len {
                if bound_tids[b] == cand_tid {
                    already_bound = true;
                    break;
                }
            }
            if already_bound {
                continue;
            }
            if a.micro_swarm.tracks.state[ti] == soa_tracks::MICRO_STATE_INACTIVE
                || a.micro_swarm.tracks.track_id[ti] != cand_tid
            {
                continue;
            }
            let mx = a.micro_swarm.tracks.pos_x[ti];
            let my = a.micro_swarm.tracks.pos_y[ti];
            let mvx = a.micro_swarm.tracks.vel_x[ti];
            let mvy = a.micro_swarm.tracks.vel_y[ti];

            let mut existing_match = false;
            for m in 0..MAX_TARGETS {
                if a.targets[m].active != 0 && a.targets[m].confirmed != 0 {
                    let dx = a.targets[m].pos_x - mx;
                    let dy = a.targets[m].pos_y - my;
                    let d2 = dx * dx + dy * dy;
                    if d2.is_finite() && d2 < match_r2 {
                        existing_match = true;
                        break;
                    }
                }
            }
            if existing_match || act_heroes >= eff_cap {
                continue;
            }
            let mut free_k: Option<usize> = None;
            for k in 0..DENSE_SWARM_HERO_CAP {
                if a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k] == 0 {
                    free_k = Some(k);
                    break;
                }
            }
            if let Some(k) = free_k {
                let hero_tid =
                    promote_or_bind_hero_slot(a, mx, my, mvx, mvy, frame_w, frame_h, scale_px, s);
                for m in 0..MAX_TARGETS {
                    if a.targets[m].active != 0 && a.targets[m].track_id == hero_tid {
                        let shift = k * 8;
                        a.targets[m].wave1_auto_owned = 1;
                        a.targets[m].phys_stamp_tid = 0;
                        a.ingest_ctrl[CTRL_AUTO_HERO_MASK] |= 1u32 << m;
                        a.ingest_ctrl[CTRL_AUTO_HERO_MICRO_BASE + k] = cand_tid;
                        a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS] =
                            (a.ingest_ctrl[CTRL_AUTO_HERO_SLOTS] & !(0xFFu32 << shift))
                                | (((m + 1) as u32) << shift);
                        act_heroes += 1;
                        bound_tids[bound_len] = cand_tid;
                        bound_len += 1;
                        break;
                    }
                }
            }
        }
    }

    a.micro_swarm.sync_bound_hero_tids(&bound_tids[..bound_len]);
}

#[inline(never)]
fn finalize_wave1_stamps(
    a: &mut Arena,
    _f_idx: u32,
    slot_in: usize,
    slot_disp: usize,
    swarm_active: bool,
    bos_ran: bool,
) {
    if a.ingest_ctrl[CTRL_TICK_PHASE] == 0 {
        a.wave1_telem[8] = INGEST_STATUS_PHASE_ERR;
        return;
    }
    a.ingest_ctrl[CTRL_TICK_PHASE] = 0;

    let mut meta = a.ring_meta[slot_in];
    let committed_flags = meta.flags | INGEST_FLAG_COMMITTED;
    meta.flags = committed_flags;
    a.ring_meta[slot_in].flags = committed_flags;

    let met_mode = a.micro_swarm.get_micro_metrology_mode();
    let cont_ep = meta.continuity_epoch;
    a.wave1_telem[0] = WAVE1_ABI_VERSION;
    a.wave1_telem[1] = core::mem::size_of::<Arena>() as u32;
    a.wave1_telem[2] = slot_in as u32;
    a.wave1_telem[3] = slot_disp as u32;
    a.wave1_telem[6] = cont_ep;
    a.wave1_telem[7] = wave1_enabled_bits(a);
    a.wave1_telem[8] = INGEST_STATUS_OK;
    a.wave1_telem[9] = committed_flags;
    a.wave1_telem[10] = mode_dims(a.res_mode).1 as u32;
    a.wave1_telem[11] = INGEST_RING_STRIDE_WORDS;
    a.wave1_telem[12] = linked_static_end_bytes();
    a.wave1_telem[13] = 0;
    a.wave1_telem[14] = 0;
    a.wave1_telem[15] = 0;

    // 0: micro metrology diagnostics (valid only when swarm active AND metrology mode != 0)
    let micro_base = WAVE1_PRODUCT_STAMPS_START + PROD_MICRO * STAMP_STRIDE;
    if swarm_active && met_mode != 0 {
        set_stamp_fresh(&mut a.wave1_telem, micro_base, &meta);
    } else {
        set_stamp_invalid(&mut a.wave1_telem, micro_base);
    }

    // 1: grad
    let grad_base = WAVE1_PRODUCT_STAMPS_START + PROD_GRAD * STAMP_STRIDE;
    let (grad_valid, grad_held, _) = a.continuum.grad_product_status();
    if swarm_active && grad_valid {
        if !grad_held {
            set_stamp_fresh(&mut a.wave1_telem, grad_base, &meta);
        } else {
            hold_or_invalidate_stamp(&mut a.wave1_telem, grad_base, cont_ep);
        }
    } else {
        set_stamp_invalid(&mut a.wave1_telem, grad_base);
    }

    // 2: FTLE
    let ftle_base = WAVE1_PRODUCT_STAMPS_START + PROD_FTLE * STAMP_STRIDE;
    let (ftle_valid, ftle_held, _) = a.continuum.ftle_product_status();
    if swarm_active && ftle_valid {
        if !ftle_held {
            set_stamp_fresh(&mut a.wave1_telem, ftle_base, &meta);
        } else {
            hold_or_invalidate_stamp(&mut a.wave1_telem, ftle_base, cont_ep);
        }
    } else {
        set_stamp_invalid(&mut a.wave1_telem, ftle_base);
    }

    // 3: modal_field
    let modal_base = WAVE1_PRODUCT_STAMPS_START + PROD_MODAL_FIELD * STAMP_STRIDE;
    let (modal_valid, modal_held, _) = a.continuum.modal_product_status();
    if swarm_active && modal_valid {
        if !modal_held {
            set_stamp_fresh(&mut a.wave1_telem, modal_base, &meta);
        } else if a.continuum.export_modal_dirty != 0 {
            hold_or_invalidate_stamp(&mut a.wave1_telem, modal_base, cont_ep);
        } else {
            set_stamp_invalid(&mut a.wave1_telem, modal_base);
        }
    } else {
        set_stamp_invalid(&mut a.wave1_telem, modal_base);
    }

    // 4: hero_physics
    let hp_base = WAVE1_PRODUCT_STAMPS_START + PROD_HERO_PHYSICS * STAMP_STRIDE;
    if a.telemetry[3] >= 0.5 {
        if a.ingest_ctrl[CTRL_HERO_PHYS_RAN] != 0 {
            set_stamp_fresh(&mut a.wave1_telem, hp_base, &meta);
        } else if a.physics_telem_dirty_mask != 0 {
            hold_or_invalidate_stamp(&mut a.wave1_telem, hp_base, cont_ep);
        } else {
            set_stamp_invalid(&mut a.wave1_telem, hp_base);
        }
    } else {
        set_stamp_invalid(&mut a.wave1_telem, hp_base);
    }

    // 5: BOS
    let bos_base = WAVE1_PRODUCT_STAMPS_START + PROD_BOS * STAMP_STRIDE;
    if bos_ran && a.bos.get_state() == bos::BOS_STATE_READY {
        set_stamp_fresh(&mut a.wave1_telem, bos_base, &meta);
    } else {
        set_stamp_invalid(&mut a.wave1_telem, bos_base);
    }

    // Hero slot stamps (16 slots): strictly driven by per-slot Hero physics computation + track_id continuity
    let fresh_phys_mask = a.ingest_ctrl[CTRL_HERO_PHYS_SLOT_MASK];
    for m in 0..MAX_TARGETS {
        let h_base = WAVE1_HERO_STAMPS_START + m * STAMP_STRIDE;
        let slot_active = a.targets[m].active != 0
            && a.targets[m].confirmed != 0
            && a.telemetry[32 + m * 56] > 0.5;
        if !slot_active {
            set_stamp_invalid(&mut a.wave1_telem, h_base);
        } else if (fresh_phys_mask & (1u32 << m)) != 0 {
            set_stamp_fresh(&mut a.wave1_telem, h_base, &meta);
        } else if a.targets[m].phys_stamp_tid != 0
            && a.targets[m].phys_stamp_tid == a.targets[m].track_id
        {
            hold_or_invalidate_stamp(&mut a.wave1_telem, h_base, cont_ep);
        } else {
            set_stamp_invalid(&mut a.wave1_telem, h_base);
        }
    }

    // Wave-2 header & stamps are finalized in finalize_wave2_stamps() right after finalize_wave1_stamps().
}

/// Complete single-call Wasm frame tick (runs pre_tick -> certified internal Jacobi Fiedler solve -> post_tick).
#[no_mangle]
pub unsafe extern "C" fn chronos_tick(dt_sec: f32) -> u32 {
    let a = arena_mut();
    if a.ingest_ctrl[CTRL_TICK_PHASE] != 0 {
        a.wave1_telem[8] = INGEST_STATUS_PHASE_ERR;
        return INGEST_STATUS_PHASE_ERR;
    }
    chronos_pre_tick(dt_sec);
    solve_fiedler_jacobi_16x16(arena_mut());
    chronos_post_tick(dt_sec)
}

// Native builds that are not linked into a `std` binary (for example `cargo build --target
// host-tuple` of this cdylib). Native test builds and `native/` library builds (feature
// `native-lib`) link into `std`, which provides the panic handler and the C library. Kept at the
// end of the file: panic `Location` line numbers are part of the shipped Wasm, so lines above
// must not move.
#[cfg(not(any(target_arch = "wasm32", test, feature = "native-lib")))]
mod native_cdylib_runtime {
    // A `no_std` cdylib links with `-nodefaultlibs`; the C library provides `abort` and the
    // `memcpy` / `memset` / `bzero` calls LLVM emits.
    #[cfg_attr(target_vendor = "apple", link(name = "System"))]
    #[cfg_attr(all(unix, not(target_vendor = "apple")), link(name = "c"))]
    extern "C" {
        fn abort() -> !;
    }

    #[panic_handler]
    fn panic(_info: &core::panic::PanicInfo) -> ! {
        // SAFETY: C `abort` takes no arguments and never returns.
        unsafe { abort() }
    }

    // The precompiled `libcore` is built with `panic = "unwind"` and references this symbol from
    // code that an un-LTO'd (debug) link keeps. This crate aborts on panic, so it is never called.
    #[no_mangle]
    extern "C" fn rust_eh_personality() {}
}

// Native-only support for host tools (pristine-arena reset, native region layout); see
// `src/native.rs`. Declared last for the same reason as the module above.
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
