#![no_std]

pub mod simd_ops;

use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_floor, f32x4_mul, f32x4_splat, f32x4_sqrt, v128, v128_load, v128_store};
use core::panic::PanicInfo;
use core::ptr::addr_of_mut;
pub use simd_ops::{eca_subspace_project_simd128, padic_ultrametric_val, sad16x16_patch_simd128};
use simd_ops::{
    build_pyramid_160x90_simd128, compute_motion_grid_and_heal_bg_simd128,
    eca_subspace_project_masked_simd128, haar_morton_signature_scaled_simd128,
    haar_morton_signature_simd128, load_pooled4_f32, padic_distance_from_val,
    patch16x16_shi_tomasi_simd128, refine_subpixel_ic_lk_weighted, rgba_roi_to_planar_simd128,
    rgba_to_planar_simd128, sad8x8_pyr160_simd128, sad_patch_weighted_simd128,
    update_stat_streak_and_mask_simd128, weighted_anchor_l1_simd128,
};

#[inline(always)]
fn wasm_sqrtf(x: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
}

#[inline(always)]
fn wasm_floorf(x: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(x)))
}

#[panic_handler]
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
pub const MAX_TARGETS: usize = 8;
pub const GRAPH_NODES: usize = 16;
pub const MAX_GRID_W: usize = 40;
pub const MAX_GRID_H: usize = 22;
pub const MAX_GRID_CELLS: usize = MAX_GRID_W * MAX_GRID_H; // 880
pub const MAX_PIP_DIM: usize = 32;
pub const MAX_PIP_PIXELS: usize = MAX_PIP_DIM * MAX_PIP_DIM; // 1,024
pub const PYR_W: usize = 160;
pub const PYR_H: usize = 90;
pub const PYR_PIXELS: usize = PYR_W * PYR_H; // 14,400
pub const MAX_ROI_DIM: usize = 96;
pub const MAX_ROI_PIXELS: usize = MAX_ROI_DIM * MAX_ROI_DIM; // 9,216
pub const MAX_HYPS: usize = 24;
pub const ASSIGN_COLS: usize = MAX_HYPS + MAX_TARGETS; // 32
pub const TELEMETRY_LEN: usize = 648;

#[inline(always)]
fn mode_dims(res_mode: u32) -> (usize, usize, usize, usize, usize, usize, f32) {
    if res_mode == 1 {
        (640, 360, 40, 22, 2, 32, 2.0)
    } else {
        (SD_FRAME_W, SD_FRAME_H, 20, 11, 1, 16, 1.0)
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
fn wrap_pi(mut a: f32) -> f32 {
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
fn fast_sinf(x: f32) -> f32 {
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
fn fast_cosf(x: f32) -> f32 {
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
        let dt_c = dt_f.min(10.0);
        let px = x0 + vx0 * dt_f;
        let py = y0 + vy0 * dt_f + 0.5 * ay_grav_f * dt_c * dt_c;
        let evy = vy0 + ay_grav_f * dt_c;
        (px, py, vx0, evy)
    } else if omega_f.abs() > 0.045 && omega_f.abs() < 0.42 {
        let psi = (omega_f * dt_f).clamp(-1.25, 1.25);
        let (sinc, cosc) = if psi.abs() < 1e-3 {
            (1.0 - psi * psi * (1.0 / 6.0), 0.5 * psi)
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
fn fast_atan2f(y: f32, x: f32) -> f32 {
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
fn fast_log10f(x: f32) -> f32 {
    let v = x.max(1e-12);
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000) - 1.0;
    // Endpoint-constrained minimax quartic for log2(1 + m) on [0, 1]: P(0) = 0, P(1) = 1.00000000
    let log2_m = (((-0.08231790 * m + 0.32402705) * m - 0.68169811) * m + 1.43998896) * m;
    (exp as f32 + log2_m) * 0.30102999566
}

#[inline(always)]
fn fast_expf(x: f32) -> f32 {
    let xc = x.clamp(-20.0, 20.0);
    let t = xc * 1.44269504089;
    let k = wasm_floorf(t);
    let f = t - k;
    let r = f * 0.69314718056;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * 0.04166667)));
    let ki = (k as i32 + 127).clamp(1, 254) as u32;
    poly * f32::from_bits(ki << 23)
}

/// Branchless L1 soft-thresholding operator emitting wasm32 `f32.abs`, `f32.max`, `f32.copysign`.
/// Also implements exact box projection onto [-lambda, lambda] via Moreau's identity:
///   clip(v, -lambda, lambda) = v - soft_threshold_branchless(v, lambda)
#[inline(always)]
pub fn soft_threshold_branchless(d2: f32, thresh: f32) -> f32 {
    let mag = (0.0f32).max(d2.abs() - thresh);
    mag.copysign(d2)
}

/// True iff both patches are confident chromatic (`bit 31 == 1`) AND occupy opposite poles
/// (`11` vs `00`, separated by >= 0.110 opponent units across the [-0.055, +0.055] deadzone).
#[inline(always)]
fn has_opposite_chroma_pole(sig_a: u32, sig_b: u32) -> bool {
    if ((sig_a & sig_b) & (1 << 31)) == 0 {
        return false;
    }
    let rg_a = (sig_a >> 29) & 0x3;
    let rg_b = (sig_b >> 29) & 0x3;
    let by_a = (sig_a >> 27) & 0x3;
    let by_b = (sig_b >> 27) & 0x3;
    (rg_a ^ rg_b) == 0x3 || (by_a ^ by_b) == 0x3
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
    pub ring_vis: [u8; RING_LEN],
    pub _pad0: [u8; 4],

    // Warm-started Dual Projected-FISTA primal X, second-difference D2*X, and dual u in [-lambda, lambda]^10
    pub fista_x: [f32; RING_LEN],
    pub fista_y: [f32; RING_LEN],
    pub fista_zx: [f32; 10],
    pub fista_zy: [f32; 10],
    pub fista_ux: [f32; 10],
    pub fista_uy: [f32; 10],
    pub fista_duality_gap: f32,

    // Non-causally revised 2D neuromuscular tremor circular history (for 15-tap Hilbert filter)
    pub tremor_hist_x: [f32; 32],
    pub tremor_hist_y: [f32; 32],
    pub tremor_valid: [u8; 32],

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
    pub proximity_mask: u8,
    pub co_dup_streak: u8,
    pub has_entry: u8,
    pub is_ballistic: u8,
    pub pivot_valid: u8,
    pub turn_streak: u8,
    pub _pad_kin: [u8; 1],

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
    pub arc_raw_x: [f32; 24],
    pub arc_raw_y: [f32; 24],
    pub arc_vis: [u8; 24],
    pub ang_vel_rad_f: f32,
    pub entry_omega_f: f32,
    pub grav_ay_frame: f32,
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

    // 3b. 160x90 Multi-Channel SIMD128 Coarse Pyramid + Non-Causal Ring ROI Scratch (340,992 B)
    pub pyr_luma: [f32; PYR_PIXELS],
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

    // 6. FISTA-Stabilized Eulerian Temporal-Contrast PiP Buffers (16x16 SD / 32x32 HD)
    pub pip_rgba: [u8; MAX_PIP_PIXELS * 4],
    pub pip_ref_luma: [f32; MAX_PIP_PIXELS],
    pub pip_target_id: u32,

    // 7. Connected-Component Seeder Workspace (20x11 SD = 220 cells / 40x22 HD = 880 cells)
    pub cc_energy: [f32; MAX_GRID_CELLS],
    pub cc_queue: [u16; MAX_GRID_CELLS],
    pub cc_visited: [u8; MAX_GRID_CELLS],
    pub peaks: [CandidatePeak; MAX_TARGETS],
    pub num_peaks: usize,

    // 8. Exported Flat Telemetry Vector (zero-marshaling Float32Array view)
    pub telemetry: [f32; TELEMETRY_LEN],

    // 9. Scalar Control, Resolution Mode, Spectral & Timing State
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
}

// Compile-time structural bound: Arena + 16 KiB stack + .rodata <= 256 pages (16,777,216 B)
const _: () = assert!(core::mem::size_of::<Arena>() <= 16_720_000);

static mut ARENA: Arena = unsafe { core::mem::zeroed() };

#[inline(always)]
unsafe fn arena_mut() -> &'static mut Arena {
    &mut *addr_of_mut!(ARENA)
}

// ============================================================================
// Helper Kernels: Foreground-Masked Templates, Pyramid Proposals, JV & Backtrack
// ============================================================================

#[inline]
fn rebuild_tpl8_from_16(t: &mut TargetTrack) {
    for py8 in 0..8usize {
        let r0 = (py8 * 2) * 16;
        let r1 = (py8 * 2 + 1) * 16;
        let drow = py8 * 8;
        for px8 in 0..8usize {
            let c0 = px8 * 2;
            let c1 = c0 + 1;
            let i00 = r0 + c0;
            let i01 = r0 + c1;
            let i10 = r1 + c0;
            let i11 = r1 + c1;
            let didx = drow + px8;

            t.tpl8_luma[didx] =
                0.25 * (t.tpl_luma[i00] + t.tpl_luma[i01] + t.tpl_luma[i10] + t.tpl_luma[i11]);
            t.tpl8_rg[didx] =
                0.25 * (t.tpl_rg[i00] + t.tpl_rg[i01] + t.tpl_rg[i10] + t.tpl_rg[i11]);
            t.tpl8_by[didx] =
                0.25 * (t.tpl_by[i00] + t.tpl_by[i01] + t.tpl_by[i10] + t.tpl_by[i11]);
            t.tpl8_w[didx] =
                t.tpl_weight[i00] + t.tpl_weight[i01] + t.tpl_weight[i10] + t.tpl_weight[i11];
        }
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
    let mut sum_w = 0.0f32;
    for ry in 0..16usize {
        let dy = (ry as f32) - 7.5;
        let trow = ry * 16;
        for rx in 0..16usize {
            let dx = (rx as f32) - 7.5;
            let r2 = (dx * dx + dy * dy) * (1.0 / 26.0);
            let w_rad = fast_expf(-r2);
            let w_fg = 0.045 + 0.955 * (eca_patch[trow + rx] * (1.0 / 0.075)).clamp(0.0, 1.0);
            let w = w_rad * w_fg;
            t.tpl_weight[trow + rx] = w;
            sum_w += w;
        }
    }
    let inv_w = 1.0 / sum_w.max(1e-6);
    for i in 0..256usize {
        t.tpl_weight[i] *= inv_w;
    }
    rebuild_tpl8_from_16(t);

    t.curr_sig = haar_morton_signature_simd128(
        t.tpl_luma.as_ptr(),
        t.tpl_rg.as_ptr(),
        t.tpl_by.as_ptr(),
        16,
        0,
        0,
    );
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
    let mut eca_patch = [0.0f32; 256];

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
            v128_store(eca_patch.as_mut_ptr().add(doff) as *mut v128, veca);
        }
    }

    let mut sum_w = 0.0f32;
    for ry in 0..16usize {
        let dy = (ry as f32) - 7.5;
        let trow = ry * 16;
        for rx in 0..16usize {
            let dx = (rx as f32) - 7.5;
            let r2 = (dx * dx + dy * dy) * (1.0 / 26.0);
            let w_rad = fast_expf(-r2);
            let w_fg = 0.045 + 0.955 * (eca_patch[trow + rx] * (1.0 / 0.075)).clamp(0.0, 1.0);
            let target_w = w_rad * w_fg;
            let nw = 0.96 * t.tpl_weight[trow + rx] + 0.04 * target_w;
            t.tpl_weight[trow + rx] = nw;
            sum_w += nw;
        }
    }
    let inv_w = 1.0 / sum_w.max(1e-6);
    for i in 0..256usize {
        t.tpl_weight[i] *= inv_w;
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
    let mut u = [0.0f32; MAX_TARGETS + 1];
    let mut v = [0.0f32; ASSIGN_COLS + 1];
    let mut p = [0usize; ASSIGN_COLS + 1];
    let mut way = [0usize; ASSIGN_COLS + 1];

    for i in 1..=nr {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = [1e30f32; ASSIGN_COLS + 1];
        let mut used = [false; ASSIGN_COLS + 1];

        for _step in 0..=nc {
            used[j0] = true;
            let i0 = p[j0];
            if i0 == 0 {
                break;
            }
            let mut delta = 1e30f32;
            let mut j1 = 0usize;

            for j in 1..=nc {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
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

    let mut row_match = [usize::MAX; MAX_TARGETS];
    for j in 1..=nc {
        if p[j] > 0 && p[j] <= nr {
            row_match[p[j] - 1] = j - 1;
        }
    }
    row_match
}

/// Propose up to 2 spatially distinct (`>= 11 * s` apart) candidate centers `(cx, cy)` for track `m`
/// via 2-stage search over the `160x90` SIMD128 pyramid + fine full-resolution refinement.
unsafe fn propose_track_candidates_pyr(
    a: &Arena,
    m: usize,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
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

        let cf = coasting as f32;
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
    }

    // Track top 2 pyramid minima separated by >= 5 pyramid pixels (`10 * s` screen px)
    let mut best1 = (p0x, p0y, 1e9f32);
    let mut best2 = (p0x, p0y, 1e9f32);

    for r_idx in 0..n_roots {
        let (rx, ry) = roots[r_idx];
        let rad_steps = if r_idx == 0 { 2i32 } else { 1i32 };
        let step_p = 3i32;
        for sy in -rad_steps..=rad_steps {
            for sx in -rad_steps..=rad_steps {
                let qx = (rx + sx * step_p).clamp(0, max_px);
                let qy = (ry + sy * step_p).clamp(0, max_py);
                let c_pyr = sad8x8_pyr160_simd128(
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
                );
                let d1 = (qx - best1.0) * (qx - best1.0) + (qy - best1.1) * (qy - best1.1);
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

    // Refine 1-pyramid-pixel (`2*s` screen px) neighborhood around top 2 coarse minima
    let mut refined_pyr = [(best1.0, best1.1, best1.2 != 1e9), (best2.0, best2.1, best2.2 < 1e8)];
    for idx in 0..2usize {
        if !refined_pyr[idx].2 {
            continue;
        }
        let cx0 = refined_pyr[idx].0;
        let cy0 = refined_pyr[idx].1;
        let mut b_px = cx0;
        let mut b_py = cy0;
        let mut b_c = 1e9f32;
        for dy in -2i32..=2i32 {
            for dx in -2i32..=2i32 {
                let qx = (cx0 + dx).clamp(0, max_px);
                let qy = (cy0 + dy).clamp(0, max_py);
                if idx == 1 {
                    let d0 = (qx - refined_pyr[0].0) * (qx - refined_pyr[0].0)
                        + (qy - refined_pyr[0].1) * (qy - refined_pyr[0].1);
                    if d0 < 25 {
                        continue;
                    }
                }
                let c_pyr = sad8x8_pyr160_simd128(
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
                );
                if c_pyr < b_c {
                    b_c = c_pyr;
                    b_px = qx;
                    b_py = qy;
                }
            }
        }
        refined_pyr[idx] = (b_px, b_py, b_c < 1e8);
    }

    // Fine full-resolution snap around the 1..2 pyramid winners
    let mut out = [(0.0f32, 0.0f32, false); 2];
    let step_coarse = scale_px as i32;
    for idx in 0..2usize {
        if !refined_pyr[idx].2 {
            continue;
        }
        let mut b_tlx = (refined_pyr[idx].0 * (2 * scale_px as i32)).clamp(1, max_tlx_i);
        let mut b_tly = (refined_pyr[idx].1 * (2 * scale_px as i32)).clamp(1, max_tly_i);
        if idx == 0 && coasting == 0 {
            // Also compete exact predicted center directly at full resolution
            let p_tlx = ((pred_cx - half_patch) as i32).clamp(1, max_tlx_i);
            let p_tly = ((pred_cy - half_patch) as i32).clamp(1, max_tly_i);
            let mut cs0 = 0u32;
            let mut we0 = 0.0f32;
            let c_pyr_full = sad_patch_weighted_simd128(
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
                t.anchor_sig,
                &mut cs0,
                &mut we0,
                scale_px,
            );
            let c_pred_full = sad_patch_weighted_simd128(
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
                t.anchor_sig,
                &mut cs0,
                &mut we0,
                scale_px,
            );
            if c_pred_full < c_pyr_full {
                b_tlx = p_tlx;
                b_tly = p_tly;
            }
        }

        let mut best_c = 1e9f32;
        for &st in &[2 * step_coarse, 1i32] {
            let cx0 = b_tlx;
            let cy0 = b_tly;
            for dy in [-st, 0, st] {
                for dx in [-st, 0, st] {
                    let qx = (cx0 + dx).clamp(1, max_tlx_i) as usize;
                    let qy = (cy0 + dy).clamp(1, max_tly_i) as usize;
                    let mut csig = 0u32;
                    let mut weca = 0.0f32;
                    let rc = sad_patch_weighted_simd128(
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
                        t.anchor_sig,
                        &mut csig,
                        &mut weca,
                        scale_px,
                    );
                    if has_opposite_chroma_pole(t.anchor_sig, csig) {
                        continue;
                    }
                    if rc < best_c {
                        best_c = rc;
                        b_tlx = qx as i32;
                        b_tly = qy as i32;
                    }
                }
            }
        }

        if best_c < 1e8 {
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
        }
    }
    out
}

/// Evaluate & locally snap (`±2*s -> ±1` px) hypothesis `(hx, hy)` for track `m`, returning
/// `(best_tlx, best_tly, raw_cost, app_err, anchor_l1, w_eca, cand_sig, valid)`.
unsafe fn eval_track_hypothesis_fine(
    a: &Arena,
    m: usize,
    hx: f32,
    hy: f32,
    frame_w: usize,
    frame_h: usize,
    scale_px: usize,
    s: f32,
) -> (usize, usize, f32, f32, f32, f32, u32, bool) {
    let half_patch = 8.0 * s;
    let patch_dim = 16 * scale_px;
    let max_tlx_i = (frame_w - patch_dim - 1) as i32;
    let max_tly_i = (frame_h - patch_dim - 1) as i32;
    let t = &a.targets[m];

    let mut b_tlx = ((hx - half_patch + 0.5) as i32).clamp(1, max_tlx_i);
    let mut b_tly = ((hy - half_patch + 0.5) as i32).clamp(1, max_tly_i);
    let mut best_rc = 1e9f32;
    let mut best_sig = t.curr_sig;
    let mut best_weca = 0.0f32;

    // Fast center pre-screen eliminates distant/mismatched cross-track hypotheses in 1 SIMD call
    {
        let qx = b_tlx as usize;
        let qy = b_tly as usize;
        let mut csig = 0u32;
        let mut weca = 0.0f32;
        let rc0 = sad_patch_weighted_simd128(
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
            t.anchor_sig,
            &mut csig,
            &mut weca,
            scale_px,
        );
        if rc0 > 1.18 {
            return (qx, qy, 1e9, 1e9, 1e9, 0.0, csig, false);
        }
        if !has_opposite_chroma_pole(t.anchor_sig, csig) {
            best_rc = rc0;
            best_sig = csig;
            best_weca = weca;
        }
    }

    let s_step = (2 * scale_px) as i32;
    {
        let cx0 = b_tlx;
        let cy0 = b_tly;
        for dy in [-s_step, 0, s_step] {
            for dx in [-s_step, 0, s_step] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let qx = (cx0 + dx).clamp(1, max_tlx_i) as usize;
                let qy = (cy0 + dy).clamp(1, max_tly_i) as usize;
                let mut csig = 0u32;
                let mut weca = 0.0f32;
                let rc = sad_patch_weighted_simd128(
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
                    t.anchor_sig,
                    &mut csig,
                    &mut weca,
                    scale_px,
                );
                if has_opposite_chroma_pole(t.anchor_sig, csig) {
                    continue;
                }
                if rc < best_rc {
                    best_rc = rc;
                    b_tlx = qx as i32;
                    b_tly = qy as i32;
                    best_sig = csig;
                    best_weca = weca;
                }
            }
        }
    }
    if best_rc < 1e8 {
        let cx0 = b_tlx;
        let cy0 = b_tly;
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let qx = (cx0 + dx).clamp(1, max_tlx_i) as usize;
                let qy = (cy0 + dy).clamp(1, max_tly_i) as usize;
                let mut csig = 0u32;
                let mut weca = 0.0f32;
                let rc = sad_patch_weighted_simd128(
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
                    t.anchor_sig,
                    &mut csig,
                    &mut weca,
                    scale_px,
                );
                if has_opposite_chroma_pole(t.anchor_sig, csig) {
                    continue;
                }
                if rc < best_rc {
                    best_rc = rc;
                    b_tlx = qx as i32;
                    b_tly = qy as i32;
                    best_sig = csig;
                    best_weca = weca;
                }
            }
        }
    }

    if best_rc >= 1e8 {
        return (b_tlx as usize, b_tly as usize, 1e9, 1e9, 1e9, 0.0, best_sig, false);
    }

    let val = padic_ultrametric_val(t.anchor_sig, best_sig);
    let d2 = padic_distance_from_val(val);
    let app_err = best_rc - 0.55 * d2 + 0.16 * best_weca.min(0.85);
    let anchor_l1 = weighted_anchor_l1_simd128(
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
    if max_back == 0 || f_idx == 0 {
        return;
    }
    let roi_dim = 48 * scale_px; // 48 in SD, 96 in HD (<= MAX_ROI_DIM)
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;
    let max_lx = (roi_dim - patch_dim - 2) as i32;
    let max_ly = (roi_dim - patch_dim - 2) as i32;
    let steps_coarse = [8 * scale_px as i32, 4 * scale_px as i32, 0];

    let mut cur_x = a.targets[m].pos_x;
    let mut cur_y = a.targets[m].pos_y;
    let ent_x = a.targets[m].entry_x;
    let ent_y = a.targets[m].entry_y;
    let denom_b = (max_back + 1) as f32;

    for b in 1..=max_back {
        if (f_idx as usize) < b {
            break;
        }
        let past_f = f_idx - (b as u32);
        let slot_b = (past_f as usize) % RING_LEN;
        if a.targets[m].ring_vis[slot_b] != 0 {
            break;
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
                (1.0 - alpha) * cur_x + alpha * ent_x,
                (1.0 - alpha) * cur_y + alpha * ent_y + sag_y,
            )
        } else {
            let step_grav = if a.targets[m].is_ballistic != 0 {
                0.5 * a.targets[m].grav_ay_frame
            } else {
                0.0
            };
            (
                cur_x - a.targets[m].vel_x,
                cur_y - a.targets[m].vel_y + step_grav,
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
        let mut best_rc = 1e9f32;
        let mut best_sig = anchor_sig;

        for &dy in &[-steps_coarse[0], -steps_coarse[1], 0, steps_coarse[1], steps_coarse[0]] {
            for &dx in &[-steps_coarse[0], -steps_coarse[1], 0, steps_coarse[1], steps_coarse[0]] {
                let lx = (base_lx + dx).clamp(2, max_lx) as usize;
                let ly = (base_ly + dy).clamp(2, max_ly) as usize;
                let mut csig = 0u32;
                let rc = sad_patch_weighted_simd128(
                    a.roi_luma.as_ptr(),
                    a.roi_rg.as_ptr(),
                    a.roi_by.as_ptr(),
                    core::ptr::null(),
                    roi_dim,
                    lx,
                    ly,
                    a.targets[m].tpl_luma.as_ptr(),
                    a.targets[m].tpl_rg.as_ptr(),
                    a.targets[m].tpl_by.as_ptr(),
                    a.targets[m].tpl_weight.as_ptr(),
                    anchor_sig,
                    &mut csig,
                    core::ptr::null_mut(),
                    scale_px,
                );
                if !has_opposite_chroma_pole(anchor_sig, csig) && rc < best_rc {
                    best_rc = rc;
                    best_lx = lx as i32;
                    best_ly = ly as i32;
                    best_sig = csig;
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
                    let lx = (cx0 + dx).clamp(2, max_lx) as usize;
                    let ly = (cy0 + dy).clamp(2, max_ly) as usize;
                    let mut csig = 0u32;
                    let rc = sad_patch_weighted_simd128(
                        a.roi_luma.as_ptr(),
                        a.roi_rg.as_ptr(),
                        a.roi_by.as_ptr(),
                        core::ptr::null(),
                        roi_dim,
                        lx,
                        ly,
                        a.targets[m].tpl_luma.as_ptr(),
                        a.targets[m].tpl_rg.as_ptr(),
                        a.targets[m].tpl_by.as_ptr(),
                        a.targets[m].tpl_weight.as_ptr(),
                        anchor_sig,
                        &mut csig,
                        core::ptr::null_mut(),
                        scale_px,
                    );
                    if !has_opposite_chroma_pole(anchor_sig, csig) && rc < best_rc {
                        best_rc = rc;
                        best_lx = lx as i32;
                        best_ly = ly as i32;
                        best_sig = csig;
                    }
                }
            }
        }

        if best_rc >= 1e8 {
            continue;
        }
        let val = padic_ultrametric_val(anchor_sig, best_sig);
        let app_err = best_rc - 0.55 * padic_distance_from_val(val);
        let anchor_l1 = weighted_anchor_l1_simd128(
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
        if app_err < 0.22 && anchor_l1 < 0.22 {
            let (sdx, sdy) = refine_subpixel_ic_lk_weighted(
                a.roi_luma.as_ptr(),
                a.roi_rg.as_ptr(),
                a.roi_by.as_ptr(),
                roi_dim,
                best_lx as usize,
                best_ly as usize,
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                a.targets[m].tpl_weight.as_ptr(),
                scale_px,
            );
            let fx = (roi_x + best_lx as usize) as f32 + half_patch + sdx;
            let fy = (roi_y + best_ly as usize) as f32 + half_patch + sdy;
            a.targets[m].ring_raw_x[slot_b] = fx;
            a.targets[m].ring_raw_y[slot_b] = fy;
            a.targets[m].ring_vx[slot_b] = cur_x - fx;
            a.targets[m].ring_vy[slot_b] = cur_y - fy;
            a.targets[m].ring_sig[slot_b] = best_sig;
            a.targets[m].ring_vis[slot_b] = 1;
            if past_f < a.targets[m].birth_frame {
                a.targets[m].birth_frame = past_f;
            }
            cur_x = fx;
            cur_y = fy;
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

/// Extract Connected Components / Energy Peaks over the active ECA motion-residual grid
/// (`20x11` in 320x180 SD, `40x22` in 640x360 HD).
unsafe fn extract_connected_component_peaks(a: &mut Arena) {
    let (frame_w, frame_h, grid_w, grid_h, scale_px, _pip_dim, s) = mode_dims(a.res_mode);
    let grid_cells = grid_w * grid_h;
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;

    for i in 0..grid_cells {
        a.cc_visited[i] = 0;
    }
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
    let seed_thresh = (mean_e * 2.6).max(2.2);

    if max_e < seed_thresh {
        return;
    }

    let min_split_cells = 6 * scale_px * scale_px;
    let min_dg2 = 5 * (scale_px as i32) * (scale_px as i32);
    let mut next_comp_id = 1u8;

    for start_idx in 0..grid_cells {
        if a.cc_visited[start_idx] != 0 || a.cc_energy[start_idx] < seed_thresh {
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

        while q_head < q_tail {
            let cur = a.cc_queue[q_head] as usize;
            q_head += 1;
            comp_cells += 1;

            let gx = cur % grid_w;
            let gy = cur / grid_w;
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

        // Split multi-target or extended foreground components into up to 3 distinct local 3x3
        // grid maxima separated by scale-invariant screen distance, preventing nearby crossing
        // targets from collapsing into a single midpoint centroid.
        let mut sub_roots = [(0usize, 0usize, 0.0f32); 3];
        let mut num_sub = 0usize;

        if comp_cells >= min_split_cells {
            for qi in 0..q_tail {
                let cell = a.cc_queue[qi] as usize;
                let ce = a.cc_energy[cell];
                if ce < seed_thresh * 1.15 {
                    continue;
                }
                let cgx = cell % grid_w;
                let cgy = cell / grid_w;
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

        if num_sub == 0 {
            sub_roots[0] = (best_gx, best_gy, sum_w);
            num_sub = 1;
        }

        let r_margin = 10.0 * s;
        let c_margin = 9.0 * s;
        let centroid_win = 12 * scale_px;

        for sr in 0..num_sub {
            let (sgx, sgy, sub_e) = sub_roots[sr];
            let rough_x = if num_sub == 1 {
                (0.5 * (sum_wx / sum_w.max(1e-5)) + 0.5 * ((sgx * 16 + 8) as f32))
                    .clamp(r_margin, (frame_w as f32) - r_margin - 1.0)
            } else {
                ((sgx * 16 + 8) as f32).clamp(r_margin, (frame_w as f32) - r_margin - 1.0)
            };
            let rough_y = if num_sub == 1 {
                (0.5 * (sum_wy / sum_w.max(1e-5)) + 0.5 * ((sgy * 16 + 8) as f32))
                    .clamp(r_margin, (frame_h as f32) - r_margin - 1.0)
            } else {
                ((sgy * 16 + 8) as f32).clamp(r_margin, (frame_h as f32) - r_margin - 1.0)
            };

            let rx0 = (rough_x as usize).saturating_sub(centroid_win).max(2);
            let rx1 = ((rough_x as usize) + centroid_win).min(frame_w - 3);
            let ry0 = (rough_y as usize).saturating_sub(centroid_win).max(2);
            let ry1 = ((rough_y as usize) + centroid_win).min(frame_h - 3);

            let mut pw = 0.0f32;
            let mut pwx = 0.0f32;
            let mut pwy = 0.0f32;
            for py in ry0..=ry1 {
                let row = py * frame_w;
                for px in rx0..=rx1 {
                    let r = a.eca_res[row + px];
                    if r > 0.05 {
                        let w = r * r;
                        pw += w;
                        pwx += w * (px as f32);
                        pwy += w * (py as f32);
                    }
                }
            }

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
                energy: if num_sub == 1 { sum_w } else { sub_e * 2.2 * s * s },
                sig,
                valid: 1,
                comp_id,
                _pad: [0; 2],
            };

            if a.num_peaks < MAX_TARGETS {
                a.peaks[a.num_peaks] = cand;
                a.num_peaks += 1;
            } else {
                let mut min_slot = 0usize;
                let mut min_val = a.peaks[0].energy;
                for k in 1..MAX_TARGETS {
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

    let mut mat = [0.0f32; GRAPH_NODES * GRAPH_NODES];
    let mut vecs = [0.0f32; GRAPH_NODES * GRAPH_NODES];
    for r in 0..n {
        let ir = act[r];
        vecs[r * n + r] = 1.0;
        for c in 0..n {
            mat[r * n + c] = a.laplacian[ir * GRAPH_NODES + act[c]];
        }
    }

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

fn solve_noncausal_fista_target(
    t: &mut TargetTrack,
    head_slot: usize,
    lambda: f32,
    fps: f32,
    frame_idx: u32,
    stimulus_armed: u32,
    stimulus_frame: u32,
    frame_w: usize,
    frame_h: usize,
    scale_f: f32,
    scene_gravity_px_s2: f32,
    scene_gravity_conf: f32,
) -> (bool, Option<f32>) {
    let mut yx = [0.0f32; RING_LEN];
    let mut yy = [0.0f32; RING_LEN];
    let mut vx = [0.0f32; RING_LEN];
    let mut vy = [0.0f32; RING_LEN];
    let mut vis = [0u8; RING_LEN];
    let mut weff = [0.0f32; RING_LEN];
    let mut vis_count = 0usize;

    for u in 0..RING_LEN {
        let s = (head_slot + 1 + u) % RING_LEN;
        yx[u] = t.ring_raw_x[s];
        yy[u] = t.ring_raw_y[s];
        vx[u] = t.ring_vx[s];
        vy[u] = t.ring_vy[s];
        vis[u] = t.ring_vis[s];
        if vis[u] != 0 {
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
                        && (frame_idx as usize) + u1 >= (RING_LEN - 1)
                        && {
                            let f_re = (frame_idx as usize) + u1 - (RING_LEN - 1);
                            let f_en = t.entry_frame as usize;
                            f_en < f_re && (f_re - f_en) <= 28
                        };

                    if can_bridge_from_entry {
                        let f_re = (frame_idx as usize) + u1 - (RING_LEN - 1);
                        let f_en = t.entry_frame as usize;
                        let dt = (f_re - f_en) as f32;
                        let v0x = t.entry_vx;
                        let v0y = t.entry_vy;
                        let (v1x, v1y) = if u1 + 1 < RING_LEN && vis[u1 + 1] != 0 {
                            (yx[u1 + 1] - yx[u1], yy[u1 + 1] - yy[u1])
                        } else {
                            let sec_vx = (yx[u1] - t.entry_x) / dt;
                            let sec_vy = (yy[u1] - t.entry_y) / dt;
                            if t.is_ballistic != 0 || t.entry_omega_f.abs() > 0.045 {
                                (1.65 * sec_vx - 0.65 * v0x, 1.65 * sec_vy - 0.65 * v0y)
                            } else {
                                (sec_vx, sec_vy)
                            }
                        };
                        for k in 0..gap_end {
                            if (frame_idx as usize) + k >= (RING_LEN - 1) {
                                let f_k = (frame_idx as usize) + k - (RING_LEN - 1);
                                if f_k > f_en && f_k < f_re {
                                    let tau = ((f_k - f_en) as f32) / dt;
                                    let tau2 = tau * tau;
                                    let tau3 = tau2 * tau;
                                    let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
                                    let h10 = tau3 - 2.0 * tau2 + tau;
                                    let h01 = -2.0 * tau3 + 3.0 * tau2;
                                    let h11 = tau3 - tau2;
                                    yx[k] = (h00 * t.entry_x + h10 * (dt * v0x) + h01 * yx[u1] + h11 * (dt * v1x))
                                        .clamp(tail_margin, (frame_w as f32) - tail_margin);
                                    yy[k] = (h00 * t.entry_y + h10 * (dt * v0y) + h01 * yy[u1] + h11 * (dt * v1y))
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
    // Scale-covariant dual box [-s * lambda, s * lambda] ensures X_HD(t) = s * X_SD(t)
    let lam = (lambda * scale_f).max(0.02 * scale_f);

    // Shift warm-started dual variables left by 1 frame and project onto [-lam, lam]
    let mut ux = [0.0f32; 10];
    let mut uy = [0.0f32; 10];
    if t.age > 1 {
        for k in 0..9usize {
            let vx_prev = t.fista_ux[k + 1];
            let vy_prev = t.fista_uy[k + 1];
            ux[k] = vx_prev - soft_threshold_branchless(vx_prev, lam);
            uy[k] = vy_prev - soft_threshold_branchless(vy_prev, lam);
        }
    }

    let mut xk = [0.0f32; RING_LEN];
    let mut yk = [0.0f32; RING_LEN];
    let mut d2x = [0.0f32; 10];
    let mut d2y = [0.0f32; 10];

    // 2 Exact Forward-Backward Coordinate-Minimization Warm-Start Sweeps on separable box [-lam, lam]^10
    eval_primal_from_dual(&yx, &inv_m, &ux, &mut xk, &mut d2x);
    eval_primal_from_dual(&yy, &inv_m, &uy, &mut yk, &mut d2y);
    for _sweep in 0..2usize {
        for k in 0..10usize {
            let cur_d2x = xk[k] - 2.0 * xk[k + 1] + xk[k + 2];
            let cur_d2y = yk[k] - 2.0 * yk[k + 1] + yk[k + 2];
            let inv_qkk = 1.0 / q_diag[k];

            let cand_ux = ux[k] + cur_d2x * inv_qkk;
            let next_ux = cand_ux - soft_threshold_branchless(cand_ux, lam);
            let dux = next_ux - ux[k];
            if dux != 0.0 {
                ux[k] = next_ux;
                xk[k] -= inv_m[k] * dux;
                xk[k + 1] += 2.0 * inv_m[k + 1] * dux;
                xk[k + 2] -= inv_m[k + 2] * dux;
            }

            let cand_uy = uy[k] + cur_d2y * inv_qkk;
            let next_uy = cand_uy - soft_threshold_branchless(cand_uy, lam);
            let duy = next_uy - uy[k];
            if duy != 0.0 {
                uy[k] = next_uy;
                yk[k] -= inv_m[k] * duy;
                yk[k + 1] += 2.0 * inv_m[k + 1] * duy;
                yk[k + 2] -= inv_m[k + 2] * duy;
            }
        }
    }

    // 6 Exact Beck & Teboulle Dual Projected-FISTA (FGP) Iterations with Moreau Box Projection
    let mut vx_dual = ux;
    let mut vy_dual = uy;
    let mut tk = 1.0f32;

    for _iter in 0..6usize {
        eval_primal_from_dual(&yx, &inv_m, &vx_dual, &mut xk, &mut d2x);
        eval_primal_from_dual(&yy, &inv_m, &vy_dual, &mut yk, &mut d2y);

        let mut ux_next = [0.0f32; 10];
        let mut uy_next = [0.0f32; 10];
        let mut restart_dot = 0.0f32;

        for k in 0..10usize {
            // Dual gradient grad_k = (Q v - D2 Y)_k = -(D2 X(v))_k
            let tilde_ux = vx_dual[k] + step_dual * d2x[k];
            let tilde_uy = vy_dual[k] + step_dual * d2y[k];

            // Moreau decomposition: clip(v, -lam, lam) = v - soft_threshold_branchless(v, lam)
            let unx = tilde_ux - soft_threshold_branchless(tilde_ux, lam);
            let uny = tilde_uy - soft_threshold_branchless(tilde_uy, lam);

            ux_next[k] = unx;
            uy_next[k] = uny;

            // O'Donoghue-Candes gradient restart check: grad(v) . (u_next - u) > 0
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
        ux = ux_next;
        uy = uy_next;
        tk = t_next;
    }

    // Recover primal X = Y - M^{-1} D2^T u and certified duality gap P(X) - D(u) = lam ||D2 X||_1 - u^T D2 X
    eval_primal_from_dual(&yx, &inv_m, &ux, &mut xk, &mut d2x);
    eval_primal_from_dual(&yy, &inv_m, &uy, &mut yk, &mut d2y);

    let mut dual_gap = 0.0f32;
    for k in 0..10usize {
        dual_gap += (lam * d2x[k].abs() - ux[k] * d2x[k]).max(0.0);
        dual_gap += (lam * d2y[k].abs() - uy[k] * d2y[k]).max(0.0);
    }

    t.fista_x = xk;
    t.fista_y = yk;
    t.fista_zx = d2x;
    t.fista_zy = d2y;
    t.fista_ux = ux;
    t.fista_uy = uy;
    t.fista_duality_gap = dual_gap;

    // Mode A: Full-Segment Closed-Form Quadratic Least-Squares Parabola + Scene Gravity Prior
    // Solves the centered 2x2 normal system for y(tau) = c0 + c1*tau + c2*tau^2 (with c2 = 0.5*g)
    // and x(tau) = x_bar + vx*tau across all impulse-free observations in [seg_start..=11],
    // retaining a 3-point Newtonian ascent fallback at [9, 10, 11] right after hand release.
    let a_frame_x = xk[11] - 2.0 * xk[10] + xk[9];
    let v_frame_x = (xk[11] - xk[10]) + 0.5 * a_frame_x;
    let v_frame_y = yk[11] - yk[10];
    let v_phys_x = v_frame_x * fps;
    let v_phys_y = v_frame_y * fps;

    let mut apex_found = false;
    let mut ballistic_found = false;
    let mut pooled_ay_phys: Option<f32> = None;

    if vis_count >= 3 && vis[11] != 0 {
        let mut seg_start = 0usize;
        let mut miss_run = 0usize;
        let mut scan_u = 11usize;
        while scan_u > 0 {
            scan_u -= 1;
            if vis[scan_u] == 0 {
                miss_run += 1;
                if miss_run >= 2 {
                    seg_start = scan_u + 2;
                    break;
                }
            } else {
                miss_run = 0;
                if scan_u + 2 <= 11 && vis[scan_u + 1] != 0 && vis[scan_u + 2] != 0 {
                    let dvx = (yx[scan_u + 2] - yx[scan_u + 1]) - (yx[scan_u + 1] - yx[scan_u]);
                    let dvy = (yy[scan_u + 2] - yy[scan_u + 1]) - (yy[scan_u + 1] - yy[scan_u]);
                    if dvx.abs() > 3.8 * scale_f || dvy < -2.8 * scale_f || dvy > 14.0 * scale_f {
                        seg_start = scan_u + 1;
                        break;
                    }
                }
            }
        }

        let bounded_osc = t.phase_valid != 0
            && t.tremor_coherence > 0.45
            && t.max_excursion2 < (92.0 * scale_f) * (92.0 * scale_f);
        let rigid_tether = t.pivot_valid > 0
            || (t.phase_valid != 0 && t.tremor_coherence > 0.36);

        if !bounded_osc {
            let start_min = seg_start.min(9);
            for u0 in start_min..=9usize {
                let mut pts = [0usize; RING_LEN];
                let mut n_pts = 0usize;
                for q in u0..=11usize {
                    if vis[q] != 0 {
                        pts[n_pts] = q;
                        n_pts += 1;
                    }
                }

                if n_pts >= 4 && (pts[n_pts - 1] - pts[0]) >= 3 {
                    let inv_n = 1.0 / (n_pts as f32);
                    let mut sum_t = 0.0f32;
                    let mut sum_x = 0.0f32;
                    let mut sum_y = 0.0f32;
                    let mut min_y = 1e9f32;
                    let mut max_y = -1e9f32;
                    for i in 0..n_pts {
                        let q = pts[i];
                        let ti = (q as f32) - 11.0;
                        let xi = yx[q];
                        let yi = yy[q];
                        sum_t += ti;
                        sum_x += xi;
                        sum_y += yi;
                        if yi < min_y {
                            min_y = yi;
                        }
                        if yi > max_y {
                            max_y = yi;
                        }
                    }
                    let t_bar = sum_t * inv_n;
                    let x_bar = sum_x * inv_n;
                    let y_bar = sum_y * inv_n;

                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    let mut s4 = 0.0f32;
                    let mut x1 = 0.0f32;
                    let mut y1 = 0.0f32;
                    let mut y2_raw = 0.0f32;
                    for i in 0..n_pts {
                        let q = pts[i];
                        let tau = ((q as f32) - 11.0) - t_bar;
                        let tau2 = tau * tau;
                        let dx = yx[q] - x_bar;
                        let dy = yy[q] - y_bar;
                        s2 += tau2;
                        s3 += tau2 * tau;
                        s4 += tau2 * tau2;
                        x1 += tau * dx;
                        y1 += tau * dy;
                        y2_raw += tau2 * dy;
                    }
                    let s2_n = s2 * inv_n;
                    let s4c = s4 - s2 * s2_n;
                    let det = s2 * s4c - s3 * s3;
                    if s2 > 1e-3 && det > 1e-3 {
                        // Centered y_bar gives sum(dy) == 0, so sum((tau^2 - s2/N)*dy) == y2_raw
                        let inv_det = 1.0 / det;
                        let c1 = (s4c * y1 - s3 * y2_raw) * inv_det;
                        let c2 = (s2 * y2_raw - s3 * y1) * inv_det;
                        let c0 = y_bar - c2 * s2_n;

                        let a_ball_y = 2.0 * c2;
                        let v_ball_y = c1 - 2.0 * c2 * t_bar;
                        let v_ball_x = x1 / s2;
                        let x11_fit = x_bar - v_ball_x * t_bar;
                        let y11_fit = c0 - c1 * t_bar + c2 * t_bar * t_bar;

                        let vy_phys_b = v_ball_y * fps;
                        let vx_phys_b = v_ball_x * fps;
                        let ay_phys_b = a_ball_y * fps * fps;

                        if ((max_y - min_y) >= 6.0 * scale_f || vy_phys_b.abs() >= 42.0 * scale_f)
                            && ay_phys_b > 110.0 * scale_f
                            && ay_phys_b < 16500.0 * scale_f
                        {
                            let mut sum_ey2 = 0.0f32;
                            let mut sum_ex2 = 0.0f32;
                            let mut max_res_y = 0.0f32;
                            let mut max_res_x = 0.0f32;
                            for i in 0..n_pts {
                                let q = pts[i];
                                let tau = ((q as f32) - 11.0) - t_bar;
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
                                let prev_g_phys = t.grav_ay_frame * fps * fps;
                                let ascending_launch = !rigid_tether
                                    && vy_phys_b < -18.0 * scale_f
                                    && ay_phys_b > 760.0 * scale_f;
                                let continuing_arc = !rigid_tether
                                    && t.is_ballistic != 0
                                    && rmse_y <= 1.85 * scale_f
                                    && prev_g_phys > 760.0 * scale_f
                                    && ay_phys_b > 0.55 * prev_g_phys
                                    && ay_phys_b < 1.55 * prev_g_phys;

                                if ascending_launch || continuing_arc {
                                    ballistic_found = true;
                                    t.is_ballistic = 1;
                                    t.grav_ay_frame = if t.grav_ay_frame > 0.01 {
                                        0.68 * t.grav_ay_frame + 0.32 * a_ball_y
                                    } else {
                                        a_ball_y
                                    };
                                    if ascending_launch
                                        && n_pts >= 5
                                        && rmse_y <= 1.9 * scale_f
                                        && rmse_x <= 2.2 * scale_f
                                    {
                                        pooled_ay_phys = Some(ay_phys_b);
                                    }
                                }

                                if vy_phys_b < -18.0 * scale_f {
                                    let ay_used = if n_pts == 4
                                        && scene_gravity_conf >= 0.55
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
                                    t.apex_frames = t_apex_sec * fps;
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
                    // Fast 3-point Newtonian ascent [9, 10, 11] for steep 24-30fps throws right after release
                    let vy1 = yy[10] - yy[9];
                    let vy2 = yy[11] - yy[10];
                    let vx1 = yx[10] - yx[9];
                    let vx2 = yx[11] - yx[10];
                    if (vx2 - vx1).abs() < 2.8 * scale_f {
                        let a_ball_y = vy2 - vy1;
                        let v_ball_y = vy2 + 0.5 * a_ball_y;
                        let v_ball_x = 0.5 * (vx1 + vx2);
                        let vy_phys_b = v_ball_y * fps;
                        let vx_phys_b = v_ball_x * fps;
                        let ay_phys_b = a_ball_y * fps * fps;
                        if vy_phys_b < -18.0 * scale_f
                            && ay_phys_b > 110.0 * scale_f
                            && ay_phys_b < 16500.0 * scale_f
                        {
                            let ay_used = if scene_gravity_conf >= 0.55
                                && scene_gravity_px_s2 > 110.0 * scale_f
                                && ay_phys_b > 0.40 * scene_gravity_px_s2
                                && ay_phys_b < 1.95 * scene_gravity_px_s2
                            {
                                0.72 * ay_phys_b + 0.28 * scene_gravity_px_s2
                            } else {
                                ay_phys_b
                            };
                            if !rigid_tether && ay_used > 760.0 * scale_f {
                                ballistic_found = true;
                                t.is_ballistic = 1;
                                t.grav_ay_frame = ay_used / (fps * fps);
                            }
                            let t_apex_sec = (-vy_phys_b / ay_used).min(0.75);
                            t.apex_frames = t_apex_sec * fps;
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

            // Constrained-arc wide-chord turnaround fallback (e.g. Galileo's pendulum decelerating in both x & y)
            if !apex_found {
                for u0 in [2usize, 3, 4, 5, 6, 7, 8] {
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
                    let (vy1, vy2, dt_mid, dt_last, vx1, vx2, min_rise) = if span >= 4 {
                        let d1 = (um - u0) as f32;
                        let d2 = (11 - um) as f32;
                        ((yy[um] - yy[u0]) / d1, (yy[11] - yy[um]) / d2, 0.5 * (d1 + d2), d2, (yx[um] - yx[u0]) / d1, (yx[11] - yx[um]) / d2, 6.0 * scale_f)
                    } else {
                        (yy[9] - yy[8], yy[11] - yy[10], 2.0, 1.0, yx[9] - yx[8], yx[11] - yx[10], 9.0 * scale_f)
                    };
                    if (vx2 - vx1).abs() < 3.5 * scale_f && (yy[u0] - yy[11]) >= min_rise {
                        let a_ball_y = (vy2 - vy1) / dt_mid;
                        let v_ball_y = vy2 + 0.5 * dt_last * a_ball_y;
                        let v_ball_x = (yx[11] - yx[u0]) / (span as f32);
                        let vy_phys_b = v_ball_y * fps;
                        let vx_phys_b = v_ball_x * fps;
                        let ay_phys_b = a_ball_y * fps * fps;
                        if vy_phys_b < -18.0 * scale_f && ay_phys_b > 110.0 * scale_f && ay_phys_b < 16500.0 * scale_f {
                            let mut obs_in_arc = 0usize;
                            let mut max_ry = 0.0f32;
                            let mut max_rx = 0.0f32;
                            for q in u0..=11usize {
                                if vis[q] != 0 {
                                    obs_in_arc += 1;
                                    let du = (q as f32) - 11.0;
                                    let ey = (yy[q] - (yy[11] + v_ball_y * du + 0.5 * a_ball_y * du * du)).abs();
                                    let ex = (yx[q] - (yx[11] + v_ball_x * du)).abs();
                                    if ey > max_ry { max_ry = ey; }
                                    if ex > max_rx { max_rx = ex; }
                                }
                            }
                            if obs_in_arc >= 4 && max_ry <= 3.6 * scale_f && max_rx <= 3.8 * scale_f {
                                let t_apex_sec = (-vy_phys_b / ay_phys_b).min(0.75);
                                t.apex_frames = t_apex_sec * fps;
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
        t.is_ballistic = 0;
    } else if vis[11] == 0 && t.coasting > 14 {
        t.is_ballistic = 0;
        t.grav_ay_frame = 0.0;
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
    let frame_ms = 1000.0 / fps.max(1.0);
    if vis[11] != 0 {
        if stimulus_armed != 0 && frame_idx > stimulus_frame {
            let elapsed_stim = frame_idx - stimulus_frame;
            if elapsed_stim <= 90 {
                let sdx = xk[11] - t.stim_ref_x;
                let sdy = yk[11] - t.stim_ref_y;
                if sdx * sdx + sdy * sdy > stim_disp2 {
                    t.saccade_latency_ms = (elapsed_stim as f32) * frame_ms;
                    consumed_stimulus = true;
                }
            }
        }

        let jump_dx = xk[11] - t.step_ref_x;
        let jump_dy = yk[11] - t.step_ref_y;
        let min_rearm = ((0.40 * fps) as u32).max(12);
        if (jump_dx * jump_dx + jump_dy * jump_dy) > step_disp2
            && frame_idx.saturating_sub(t.step_onset_frame) > min_rearm
        {
            t.step_onset_frame = frame_idx;
            t.step_ref_x = xk[11];
            t.step_ref_y = yk[11];
        } else if t.step_onset_frame > 0 {
            let settle_speed_px_s = v_phys_x.abs() + v_phys_y.abs();
            let elapsed_frames = frame_idx.saturating_sub(t.step_onset_frame);
            let max_settle_frames = ((0.50 * fps) as u32).max(15);
            if elapsed_frames >= 2 && elapsed_frames <= max_settle_frames && settle_speed_px_s < settle_thresh {
                t.step_settle_ms = (elapsed_frames as f32) * frame_ms;
                t.step_onset_frame = 0;
            }
        }
    }
    (consumed_stimulus, pooled_ay_phys)
}

// ============================================================================
// Exported C ABI Entrypoints
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn chronos_init() {
    let a = arena_mut();
    core::ptr::write_bytes(a as *mut Arena as *mut u8, 0, 1);
    *a = core::mem::zeroed();
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
    a.clutter_db = 0.0;
    a.clutter_db_valid = 0;
}

#[no_mangle]
pub unsafe extern "C" fn chronos_reset_tracks() {
    let a = arena_mut();
    for m in 0..MAX_TARGETS {
        a.targets[m] = core::mem::zeroed();
        a.kalman[m] = core::mem::zeroed();
        a.prev_fiedler_track[m] = 0.0;
    }
    for i in 0..MAX_FRAME_PIXELS {
        a.bg_luma[i] = 0.0;
        a.eca_res[i] = 0.0;
        a.bg_stat_streak[i] = 0;
        a.bg_eval_mask[i] = 0;
    }
    for i in 0..PYR_PIXELS {
        a.pyr_luma[i] = 0.0;
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
    a.frame_idx = 0;
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
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_focus_target(track_id: u32) {
    arena_mut().focus_target_id = track_id;
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
            a.targets[m].stim_ref_x = a.targets[m].fista_x[RING_LEN - 1];
            a.targets[m].stim_ref_y = a.targets[m].fista_y[RING_LEN - 1];
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_ingest_rgba_ptr() -> *mut u8 {
    let a = arena_mut();
    let slot = (a.frame_idx as usize) % RING_LEN;
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
    if lambda2.is_finite() {
        a.fiedler_lambda2 = lambda2.max(0.0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_set_fiedler_spectrum(lambda2: f32, spectral_gap: f32) {
    let a = arena_mut();
    if lambda2.is_finite() {
        a.fiedler_lambda2 = lambda2.max(0.0);
    }
    if spectral_gap.is_finite() {
        a.fiedler_gap = spectral_gap.max(0.0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_get_telemetry_ptr() -> *const f32 {
    let a = arena_mut();
    a.telemetry.as_ptr()
}

/// Phase 1 of frame tick:
/// 1. Unpacks RGBA8 into SIMD128 Luma/Opponent-Chroma planes.
/// 2. Evaluates uncensored clutter suppression over independent prior-frame `bg_eval_mask`
///    and performs guarded rank-2 background regression/cancellation.
/// 3. Extracts Connected-Component energy peaks and updates `bg_stat_streak` for next frame.
/// 4. Assembles the pre-assignment 16x16 Track-Peak Graph Laplacian L = D - A so the Fiedler
///    eigensolver runs BEFORE track-peak association.
#[no_mangle]
pub unsafe extern "C" fn chronos_pre_tick(_dt_sec: f32) {
    let a = arena_mut();
    let (frame_w, frame_h, grid_w, grid_h, scale_px, _pip_dim, s) = mode_dims(a.res_mode);
    let frame_pixels = frame_w * frame_h;
    let half_patch = 8.0 * s;
    let f_idx = a.frame_idx;
    let slot_in = (f_idx as usize) % RING_LEN;
    let slot_prev = if f_idx == 0 {
        slot_in
    } else {
        ((f_idx as usize) + RING_LEN - 1) % RING_LEN
    };
    let eff_k = (a.lookahead_k.min(f_idx) as usize).min(RING_LEN - 1);
    let slot_disp = ((f_idx as usize) + RING_LEN - eff_k) % RING_LEN;
    a.last_display_slot = slot_disp as u32;

    // 1. SIMD128 Planar Unpack
    rgba_to_planar_simd128(
        a.rgba_ring[slot_in].as_ptr(),
        a.luma.as_mut_ptr(),
        a.chroma_rg.as_mut_ptr(),
        a.chroma_by.as_mut_ptr(),
        frame_pixels,
    );

    // 2. Construct independent stationary-clutter evaluation mask strictly from raw frame-to-frame
    //    RGB temporal invariance (>= 12 consecutive frames), 100% independent of ECA output `eca_res`.
    if f_idx > 0 {
        update_stat_streak_and_mask_simd128(
            a.rgba_ring[slot_in].as_ptr(),
            a.rgba_ring[slot_prev].as_ptr(),
            a.bg_stat_streak.as_mut_ptr(),
            a.bg_eval_mask.as_mut_ptr(),
            frame_pixels,
        );
    }

    let mut err2 = 0.0f32;
    let mut sig2 = 0.0f32;
    let mut eval_count = 0u32;
    let _ratio = eca_subspace_project_masked_simd128(
        a.luma.as_ptr(),
        a.bg_luma.as_mut_ptr(),
        if f_idx >= 12 {
            a.bg_eval_mask.as_ptr()
        } else {
            core::ptr::null()
        },
        a.eca_res.as_mut_ptr(),
        frame_pixels,
        a.eca_adapt_rate,
        &mut err2,
        &mut sig2,
        &mut eval_count,
    );

    // Build 160x90 SIMD128 multi-channel coarse pyramid (`luma`, `rg`, `by`, `eca_res`)
    build_pyramid_160x90_simd128(
        a.luma.as_ptr(),
        a.chroma_rg.as_ptr(),
        a.chroma_by.as_ptr(),
        a.eca_res.as_ptr(),
        a.pyr_luma.as_mut_ptr(),
        a.pyr_rg.as_mut_ptr(),
        a.pyr_by.as_mut_ptr(),
        a.pyr_eca.as_mut_ptr(),
        scale_px,
    );

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

    // 3. Motion Grid, Stationary Background Healing & Connected-Component Peaks
    if f_idx > 0 {
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

    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
            let coasting = a.targets[m].coasting;
            let (px, py) = if coasting == 0 {
                (
                    (a.targets[m].pos_x + a.targets[m].vel_x + 0.35 * a.targets[m].acc_x)
                        .clamp(half_patch, (frame_w as f32) - half_patch),
                    (a.targets[m].pos_y + a.targets[m].vel_y + 0.35 * a.targets[m].acc_y)
                        .clamp(half_patch, (frame_h as f32) - half_patch),
                )
            } else {
                let cf = coasting as f32;
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
            a.targets[m].pred_cx = px;
            a.targets[m].pred_cy = py;

            node_active[m] = true;
            node_x[m] = px;
            node_y[m] = py;
            node_sig[m] = a.targets[m].anchor_sig;
        }
        if m < a.num_peaks && a.peaks[m].valid != 0 {
            node_active[8 + m] = true;
            node_x[8 + m] = a.peaks[m].x;
            node_y[8 + m] = a.peaks[m].y;
            node_sig[8 + m] = a.peaks[m].sig;
        }
    }

    let lap_sigma = 28.0 * s;
    let inv_2sig2 = 1.0 / (2.0 * lap_sigma * lap_sigma);
    for i in 0..GRAPH_NODES {
        if !node_active[i] {
            continue;
        }
        let mut deg = 0.0f32;
        for j in 0..GRAPH_NODES {
            if i == j || !node_active[j] {
                continue;
            }
            let padic_aff = if has_opposite_chroma_pole(node_sig[i], node_sig[j]) {
                0.0
            } else {
                let val = padic_ultrametric_val(node_sig[i], node_sig[j]);
                (1.0 - padic_distance_from_val(val)).max(0.05)
            };

            let dx = node_x[i] - node_x[j];
            let dy = node_y[i] - node_y[j];
            let kin_aff = fast_expf(-(dx * dx + dy * dy) * inv_2sig2);

            // Macro-geometric spatial coupling (0.08 * kin_aff) + 2-adic ultrametric affinity
            let w = 1e-3 + 0.08 * kin_aff + padic_aff * (0.12 + 0.80 * kin_aff);
            a.laplacian[i * GRAPH_NODES + j] = -w;
            deg += w;
        }
        a.laplacian[i * GRAPH_NODES + i] = deg;
    }
}

/// Measures pre- and post-impact velocity chords around a sparse L1 acceleration impulse `strike_u`
/// inside the 12-frame ring, skipping the exposure-straddling frame `strike_u` when multi-frame
/// flanking chords exist. Returns `(in_deg, signed_deflect_deg, restitution, out_vx, out_vy, pre_valid)`.
fn measure_strike_kinematics(
    t: &TargetTrack,
    slot_in: usize,
    strike_u: usize,
    s: f32,
) -> Option<(f32, f32, f32, f32, f32, bool)> {
    if strike_u == 0 || strike_u + 1 >= RING_LEN {
        return None;
    }
    let mut rx = [0.0f32; RING_LEN];
    let mut ry = [0.0f32; RING_LEN];
    let mut rvis = [0u8; RING_LEN];
    for u in 0..RING_LEN {
        let su = (slot_in + 1 + u) % RING_LEN;
        rx[u] = t.ring_raw_x[su];
        ry[u] = t.ring_raw_y[su];
        rvis[u] = t.ring_vis[su];
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
    if u_a == u_b || rvis[u_a] == 0 || rvis[u_b] == 0 {
        return None;
    }
    let dt_pre = (u_b - u_a) as f32;
    let vpre_x = (rx[u_b] - rx[u_a]) / dt_pre;
    let vpre_y = (ry[u_b] - ry[u_a]) / dt_pre;

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
    if u_d == u_c || rvis[u_c] == 0 || rvis[u_d] == 0 {
        return None;
    }
    let dt_post = (u_d - u_c) as f32;
    let vpost_x = (rx[u_d] - rx[u_c]) / dt_post;
    let vpost_y = (ry[u_d] - ry[u_c]) / dt_post;

    let pre_mag = wasm_sqrtf(vpre_x * vpre_x + vpre_y * vpre_y);
    let post_mag = wasm_sqrtf(vpost_x * vpost_x + vpost_y * vpost_y);
    let rad_to_deg = 180.0 / PI;

    if pre_mag >= 0.75 * s {
        let in_rad = fast_atan2f(-vpre_y, vpre_x);
        let in_deg = in_rad * rad_to_deg;
        let rest = (post_mag / pre_mag).clamp(0.0, 2.5);
        let defl_deg = if post_mag >= 0.40 * s {
            let out_rad = fast_atan2f(-vpost_y, vpost_x);
            wrap_pi(out_rad - in_rad) * rad_to_deg
        } else {
            0.0
        };
        Some((in_deg, defl_deg, rest, vpost_x, vpost_y, true))
    } else if post_mag >= 1.15 * s {
        // Object ball struck from near-rest: export outgoing velocity for 2-body scatter wedge
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
    if rmse_r > 2.25 * s {
        return None;
    }
    Some((xc, yc, rad, my, sag))
}

/// Phase 2 of frame tick (executes after Fiedler vector & eigengap are written to Arena):
/// Performs spectrally-regularized 2-adic patch association, continuous SSD sub-pixel refinement,
/// 2x2 Kalman baseline tracking, Dual Projected-FISTA smoothing, non-causal Hilbert equalization,
/// and biophysical telemetry export.
#[no_mangle]
pub unsafe extern "C" fn chronos_post_tick(dt_sec: f32) -> u32 {
    let a = arena_mut();
    let (frame_w, frame_h, grid_w, grid_h, scale_px, pip_dim, s) = mode_dims(a.res_mode);
    let s2 = s * s;
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;
    let s_i32 = scale_px as i32;

    let f_idx = a.frame_idx;
    let safe_dt = if dt_sec.is_finite() {
        dt_sec.clamp(0.002, 0.15)
    } else {
        1.0 / 60.0
    };
    let fps = (1.0 / safe_dt).clamp(8.0, 240.0);
    let dt_frames_60 = (safe_dt * 60.0).clamp(0.25, 6.5);

    let slot_in = (f_idx as usize) % RING_LEN;
    let arc_slot_in = (f_idx as usize) % 24;
    let eff_k = (a.lookahead_k.min(f_idx) as usize).min(RING_LEN - 1);
    let slot_disp = ((f_idx as usize) + RING_LEN - eff_k) % RING_LEN;
    a.last_display_slot = slot_disp as u32;

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
            0.5
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
    let claim_r2 = (22.0 * s) * (22.0 * s);

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
                a.targets[i].proximity_mask |= 1u8 << j;
                a.targets[j].proximity_mask |= 1u8 << i;
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

    for r_i in 0..nr {
        let m = act_tracks[r_i];
        let props = propose_track_candidates_pyr(a, m, frame_w, frame_h, scale_px, s);
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

    // Assemble rectangular bipartite cost matrix `cost_mat[nr][n_hyps + nr]`
    // Cache fine evaluation per `(r_i, h)` so winning pairs need zero re-evaluation!
    let nc = n_hyps + nr;
    let mut cost_mat = [[1e9f32; ASSIGN_COLS]; MAX_TARGETS];
    let mut eval_tlx = [[0usize; MAX_HYPS]; MAX_TARGETS];
    let mut eval_tly = [[0usize; MAX_HYPS]; MAX_TARGETS];
    let mut eval_raw = [[1e9f32; MAX_HYPS]; MAX_TARGETS];
    let mut eval_app = [[1e9f32; MAX_HYPS]; MAX_TARGETS];
    let mut eval_anc = [[1e9f32; MAX_HYPS]; MAX_TARGETS];
    let mut eval_eca = [[0.0f32; MAX_HYPS]; MAX_TARGETS];
    let mut eval_sig = [[0u32; MAX_HYPS]; MAX_TARGETS];
    let mut eval_ok = [[false; MAX_HYPS]; MAX_TARGETS];

    let vis_pred_r = 36.0 * s * gate_cadence;
    let vis_curr_r = 39.5 * s * gate_cadence;
    let vis_pred_r2 = vis_pred_r * vis_pred_r;
    let vis_curr_r2 = vis_curr_r * vis_curr_r;
    let inv_kin_scale2 = 1.0 / ((30.0 * s) * (30.0 * s));

    for r_i in 0..nr {
        let m = act_tracks[r_i];
        let t = &a.targets[m];
        let coasting = t.coasting;
        let pred_cx = t.pred_cx;
        let pred_cy = t.pred_cy;
        let v_track = a.fiedler_vec[m];

        // Dedicated coasting / occlusion null column for row `r_i`
        // Confirmed tracks slightly outbid tentative tracks for contested hypotheses
        cost_mat[r_i][n_hyps + r_i] = if t.confirmed != 0 { 0.54 } else { 0.46 };

        let v_ent = wasm_sqrtf(t.entry_vx * t.entry_vx + t.entry_vy * t.entry_vy);
        let cf = coasting as f32;
        let r_entry_max = (26.0 * s + cf * (v_ent.max(4.2 * s) + 2.4 * s)).min(148.0 * s);
        let r_entry_max2 = r_entry_max * r_entry_max;
        let d_kin_max = (34.0 * s + cf * 3.5 * s).min(118.0 * s);

        for h in 0..n_hyps {
            let hx = hyps_x[h];
            let hy = hyps_y[h];

            let kin_cost = if coasting == 0 {
                let dxp = hx - pred_cx;
                let dyp = hy - pred_cy;
                let dxc = hx - t.pos_x;
                let dyc = hy - t.pos_y;
                let d2_curr = dxc * dxc + dyc * dyc;
                let v_mag2 = t.vel_x * t.vel_x + t.vel_y * t.vel_y;
                let non_harmonic = t.pivot_valid == 0
                    && !(t.phase_valid != 0 && t.tremor_coherence > 0.25);
                if t.confirmed != 0 && non_harmonic && v_mag2 >= (8.5 * s) * (8.5 * s) {
                    // Anisotropic heading-aligned elliptical gate: expand longitudinal reach (1.08x)
                    // along velocity vector while tightening cross-track lateral radius (0.90x) & cost (1.24x).
                    let v_mag = wasm_sqrtf(v_mag2);
                    let ux_h = t.vel_x / v_mag;
                    let uy_h = t.vel_y / v_mag;
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
                    let d2_pred = dxp * dxp + dyp * dyp;
                    if d2_pred > vis_pred_r2 || d2_curr > vis_curr_r2 {
                        continue;
                    }
                    0.036 * d2_pred * inv_kin_scale2
                }
            } else {
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
                        || ex_y > (frame_h as f32) - half_patch + 3.0 * s);
                if exited_border {
                    continue;
                }
                let dxe = hx - t.entry_x;
                let dye = hy - t.entry_y;
                let d2_ent = dxe * dxe + dye * dye;
                if d2_ent > r_entry_max2 {
                    continue;
                }
                let d_ent = wasm_sqrtf(d2_ent) + 4.0 * s;
                let d_pass_curv =
                    wasm_sqrtf((hx - pred_cx) * (hx - pred_cx) + (hy - pred_cy) * (hy - pred_cy));
                let lin_cx = (t.entry_x + cf * t.entry_vx).clamp(half_patch, (frame_w as f32) - half_patch);
                let lin_cy = (t.entry_y + cf * t.entry_vy).clamp(half_patch, (frame_h as f32) - half_patch);
                let d_pass_lin =
                    wasm_sqrtf((hx - lin_cx) * (hx - lin_cx) + (hy - lin_cy) * (hy - lin_cy));
                let d_pass = d_pass_curv.min(d_pass_lin);
                let refl_x = t.entry_x - cf * t.entry_vx;
                let refl_y = t.entry_y - cf * t.entry_vy;
                let d_refl = wasm_sqrtf((hx - refl_x) * (hx - refl_x) + (hy - refl_y) * (hy - refl_y)) + 2.0 * s;
                let d_kin = d_pass.min(d_ent).min(d_refl);
                if d_kin > d_kin_max {
                    continue;
                }
                0.048 * (d_kin * d_kin) * inv_kin_scale2 + 0.030
            };

            let (btlx, btly, raw_c, app_err, anc_l1, weca, csig, ok) =
                eval_track_hypothesis_fine(a, m, hx, hy, frame_w, frame_h, scale_px, s);
            if !ok {
                continue;
            }

            let anchor_chromatic = (t.anchor_sig & (1 << 31)) != 0;
            let speed_l1 = t.vel_x.abs() + t.vel_y.abs();
            let vis_ok = if coasting == 0 {
                if anchor_chromatic {
                    app_err < 0.42 && anc_l1 < 0.42 && (weca > 0.011 || app_err < 0.135)
                } else {
                    let max_anc_vis = if speed_l1 > 2.5 * s { 0.36 } else { 0.30 };
                    app_err < 0.38 && anc_l1 < max_anc_vis && (weca > 0.011 || app_err < 0.125)
                }
            } else if anchor_chromatic {
                weca > 0.022 && app_err < 0.32 && anc_l1 < 0.30
            } else {
                weca > 0.022 && app_err < 0.28 && anc_l1 < 0.27
            };
            if !vis_ok {
                continue;
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
                    let dv = v_track - a.fiedler_vec[8 + p];
                    spec_pen = eff_spec_gamma * (dv * dv);
                }
            }

            let anc_pen = 0.30 * (anc_l1 - 0.045).max(0.0);
            let total_c = raw_c + anc_pen + kin_cost + spec_pen;

            cost_mat[r_i][h] = total_c;
            eval_tlx[r_i][h] = btlx;
            eval_tly[r_i][h] = btly;
            eval_raw[r_i][h] = raw_c;
            eval_app[r_i][h] = app_err;
            eval_anc[r_i][h] = anc_l1;
            eval_eca[r_i][h] = weca;
            eval_sig[r_i][h] = csig;
            eval_ok[r_i][h] = true;
        }
    }

    let row_assign = if nr > 0 {
        solve_jv_assign_8x32(nr, nc, &cost_mat)
    } else {
        [usize::MAX; MAX_TARGETS]
    };

    // Extract sub-pixel positions via 2D Foreground-Weighted IC-LK & enforce post-match spatial NMS
    // `(mcx, mcy, total_cost, raw_cost, app_err, anc_l1, weca, sig, val, vis)`
    let mut matched = [(0.0f32, 0.0f32, 1e9f32, 1e9f32, 1e9f32, 1e9f32, 0.0f32, 0u32, 0u32, false); MAX_TARGETS];
    for r_i in 0..nr {
        let m = act_tracks[r_i];
        let col = row_assign[r_i];
        if col < n_hyps && eval_ok[r_i][col] {
            let btlx = eval_tlx[r_i][col];
            let btly = eval_tly[r_i][col];
            let (mut sub_dx, mut sub_dy) = (0.0f32, 0.0f32);
            if btlx >= 2
                && btlx + patch_dim + 1 < frame_w
                && btly >= 2
                && btly + patch_dim + 1 < frame_h
            {
                let (dx_lk, dy_lk) = refine_subpixel_ic_lk_weighted(
                    a.luma.as_ptr(),
                    a.chroma_rg.as_ptr(),
                    a.chroma_by.as_ptr(),
                    frame_w,
                    btlx,
                    btly,
                    a.targets[m].tpl_luma.as_ptr(),
                    a.targets[m].tpl_rg.as_ptr(),
                    a.targets[m].tpl_by.as_ptr(),
                    a.targets[m].tpl_weight.as_ptr(),
                    scale_px,
                );
                sub_dx = dx_lk;
                sub_dy = dy_lk;
            }
            let mcx = (btlx as f32) + half_patch + sub_dx;
            let mcy = (btly as f32) + half_patch + sub_dy;
            let csig = eval_sig[r_i][col];
            let cval = padic_ultrametric_val(a.targets[m].anchor_sig, csig);
            matched[m] = (
                mcx,
                mcy,
                cost_mat[r_i][col],
                eval_raw[r_i][col],
                eval_app[r_i][col],
                eval_anc[r_i][col],
                eval_eca[r_i][col],
                csig,
                cval,
                true,
            );
        }
    }

    let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7];
    for i in 0..MAX_TARGETS {
        for j in (i + 1)..MAX_TARGETS {
            // Confirmed tracks win spatial tie-breaks before raw assignment cost
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

    let mut backtrack_budget = 2usize;
    let mut comp_cells_cnt = [0u16; 32];
    let grid_total = grid_w * grid_h;
    for gidx in 0..grid_total {
        let cid = a.cc_visited[gidx] as usize;
        if cid > 0 && cid < 32 {
            comp_cells_cnt[cid] = comp_cells_cnt[cid].saturating_add(1);
        }
    }
    let mut comp_track_cnt = [0u8; 32];
    let co_dup_r2 = (22.5 * s) * (22.5 * s);

    for idx in 0..MAX_TARGETS {
        let m = order[idx];
        if a.targets[m].active == 0 {
            continue;
        }
        let (mcx, mcy, _mtotal, mrawcost, mapp_err, manc_l1, mweca, msig, mval, mut vis) = matched[m];
        let mut co_moving_dup = false;
        let mut track_cid = 0usize;

        if vis {
            let gx = ((mcx as usize) / 16).min(grid_w - 1);
            let gy = ((mcy as usize) / 16).min(grid_h - 1);
            track_cid = a.cc_visited[gy * grid_w + gx] as usize;

            for prev_idx in 0..idx {
                let pm = order[prev_idx];
                if a.targets[pm].active != 0 && matched[pm].9 {
                    let dx = mcx - matched[pm].0;
                    let dy = mcy - matched[pm].1;
                    let d2 = dx * dx + dy * dy;
                    if d2 < conflict_r2 {
                        vis = false;
                        matched[m].9 = false;
                        break;
                    }
                    if d2 < co_dup_r2
                        && !has_opposite_chroma_pole(a.targets[m].anchor_sig, a.targets[pm].anchor_sig)
                        && ((a.targets[m].vel_x - a.targets[pm].vel_x).abs()
                            + (a.targets[m].vel_y - a.targets[pm].vel_y).abs()
                            < 1.8 * s)
                    {
                        co_moving_dup = true;
                    }
                }
            }
        }

        if vis {
            for p in 0..a.num_peaks {
                let dx = a.peaks[p].x - mcx;
                let dy = a.peaks[p].y - mcy;
                if dx * dx + dy * dy < claim_r2 {
                    peak_claimed[p] = 1;
                    if track_cid == 0 {
                        track_cid = a.peaks[p].comp_id as usize;
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
                let ent_speed = a.targets[m].entry_vx.abs() + a.targets[m].entry_vy.abs();
                let near_entry = (mcx - a.targets[m].entry_x).abs() < 20.0 * s
                    && (mcy - a.targets[m].entry_y).abs() < 20.0 * s;
                if prev_coasting >= 6 && ent_speed >= 1.5 * s && near_entry {
                    a.targets[m].vel_x = (-0.85 * a.targets[m].entry_vx).clamp(-v_clamp, v_clamp);
                    a.targets[m].vel_y = (0.50 * a.targets[m].entry_vy).clamp(-v_clamp, v_clamp);
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
                a.targets[m].reemerged_last = 1;
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

            if a.targets[m].confirmed == 0 {
                let hits = a.targets[m].hit_mask_8.count_ones();
                let std_confirm = hits >= 3
                    && (a.targets[m].max_excursion2 >= (1.5 * s) * (1.5 * s)
                        || a.targets[m].vis_eca_ema >= 0.032);
                let fast_projectile_confirm = hits >= 2
                    && a.targets[m].max_excursion2 >= (3.2 * s) * (3.2 * s)
                    && a.targets[m].vis_eca_ema >= 0.068;
                if std_confirm || fast_projectile_confirm {
                    a.targets[m].confirmed = 1;
                    a.targets[m].just_confirmed = 1;
                }
            }

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
                    let nb = (f_idx as usize).min(eff_k).min(RING_LEN - 2);
                    backtrack_gap_in_rgba_ring(a, m, f_idx, nb, frame_w, frame_h, scale_px, s, false);
                    backtrack_budget -= 1;
                }
            }

            // P-N Dual-Anchor Drift-Locked Template Update:
            // Only adapt `tpl_*` when confirmed, isolated, genuinely moving, and faithful to birth anchor!
            if a.targets[m].confirmed != 0
                && mrawcost < 0.16
                && mapp_err < 0.16
                && manc_l1 < 0.20
                && mweca > 0.025
                && mval >= 8
            {
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
                        0.025,
                        &mut a.targets[m],
                    );
                }
            }

            // Static-Background Pin, Duplicate Fragment & Tentative Failure Reaper
            let cx_i = (mcx as usize).min(frame_w - 1);
            let cy_i = (mcy as usize).min(frame_h - 1);
            let center_stat = a.bg_eval_mask[cy_i * frame_w + cx_i] != 0;
            let step2 = inst_vx * inst_vx + inst_vy * inst_vy;
            if prev_coasting == 0
                && step2 < (0.38 * s) * (0.38 * s)
                && (mweca < 0.011 || center_stat)
            {
                let inc = if mweca < 0.006 && center_stat { 2u8 } else { 1u8 };
                a.targets[m].static_streak = a.targets[m].static_streak.saturating_add(inc);
            } else if step2 > (0.68 * s) * (0.68 * s) {
                a.targets[m].static_streak = a.targets[m].static_streak.saturating_sub(2);
            }

            let active_oscillator = a.targets[m].phase_valid != 0
                && a.targets[m].tremor_amp > 0.30 * s
                && a.targets[m].tremor_coherence > 0.30;
            let traveled_body = a.targets[m].max_excursion2 >= (20.0 * s) * (20.0 * s);
            let max_stat_streak = if traveled_body { 135 } else { 18 };
            let unconfirmed_expired = a.targets[m].confirmed == 0 && a.targets[m].age >= 7;
            let static_pinned = !active_oscillator
                && (a.targets[m].static_streak >= max_stat_streak
                    || (a.targets[m].age >= 24
                        && a.targets[m].max_excursion2 < (4.8 * s) * (4.8 * s)));
            let dup_redundant =
                (a.targets[m].confirmed == 0 && co_moving_dup) || a.targets[m].co_dup_streak >= 3;
            if unconfirmed_expired || static_pinned || dup_redundant {
                a.targets[m].active = 0;
                a.targets[m].confirmed = 0;
                a.kalman[m].active = 0;
                let clear_bit = !(1u8 << m);
                for o in 0..MAX_TARGETS {
                    a.targets[o].proximity_mask &= clear_bit;
                }
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
                        && (a.targets[m].proximity_mask & (1u8 << o)) != 0
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
                || raw_coast_y > (frame_h as f32) - half_patch + 3.0 * s;
            let coast_x = raw_coast_x.clamp(half_patch, (frame_w as f32) - half_patch);
            let coast_y = raw_coast_y.clamp(half_patch, (frame_h as f32) - half_patch);

            a.targets[m].ring_raw_x[slot_in] = coast_x;
            a.targets[m].ring_raw_y[slot_in] = coast_y;
            a.targets[m].ring_vx[slot_in] = coast_vx;
            a.targets[m].ring_vy[slot_in] = coast_vy;
            a.targets[m].ring_sig[slot_in] = a.targets[m].curr_sig;
            a.targets[m].ring_vis[slot_in] = 0;
            a.targets[m].arc_vis[arc_slot_in] = 0;

            let max_coast = if a.targets[m].confirmed == 0 {
                2
            } else if exited_viewport {
                (eff_k as u32 + 2).max(4)
            } else if a.targets[m].age < 10 {
                16
            } else {
                64
            };
            if a.targets[m].coasting >= max_coast
                || (a.targets[m].confirmed == 0 && a.targets[m].age >= 6)
            {
                a.targets[m].active = 0;
                a.targets[m].confirmed = 0;
                a.kalman[m].active = 0;
                let clear_bit = !(1u8 << m);
                for o in 0..MAX_TARGETS {
                    a.targets[o].proximity_mask &= clear_bit;
                }
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
    // ------------------------------------------------------------------------
    if f_idx >= 4 {
        let seed_close_r2 = (19.0 * s) * (19.0 * s);
        let seed_chroma_r2 = (28.0 * s) * (28.0 * s);
        let seed_neutral_r2 = (27.5 * s) * (27.5 * s);
        let large_comp_sep_r2 = (34.0 * s) * (34.0 * s);
        let coast_guard_r2 = (54.0 * s) * (54.0 * s);
        let single_obj_cells = (48 * scale_px * scale_px) as u16;
        let core_lo = 2 * scale_px;
        let core_hi = 14 * scale_px;
        let inner_lo = 5 * scale_px;
        let inner_hi = 11 * scale_px;
        let min_patch_eca = 144.0 * s2 * 0.054;
        let min_inner6_eca = 36.0 * s2 * 0.080;
        let min_peak_energy = 3.2 * s * wasm_sqrtf(s);
        let st_min_tlx = 3 * s_i32;
        let st_max_tlx = (frame_w - 19 * scale_px) as i32;
        let st_min_tly = 3 * s_i32;
        let st_max_tly = (frame_h - 19 * scale_px) as i32;
        let snap_step = 4 * s_i32;
        let mut births_this_frame = 0usize;

        for p in 0..a.num_peaks {
            if births_this_frame >= 2 || peak_claimed[p] != 0 || a.peaks[p].energy < min_peak_energy {
                continue;
            }
            let raw_px = a.peaks[p].x;
            let raw_py = a.peaks[p].y;
            let psig = a.peaks[p].sig;
            let p_comp = a.peaks[p].comp_id;
            let peak_chromatic = (psig & (1 << 31)) != 0;

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

            {
                let qx = base_tlx as usize;
                let qy = base_tly as usize;
                let mut c_eca = 0.0f32;
                let mut inner_eca = 0.0f32;
                let mut max_eca = 0.0f32;
                for ry in core_lo..core_hi {
                    let roff = (qy + ry) * frame_w + qx;
                    let in_y = ry >= inner_lo && ry < inner_hi;
                    for rx in core_lo..core_hi {
                        let v = a.eca_res[roff + rx];
                        c_eca += v;
                        if in_y && rx >= inner_lo && rx < inner_hi {
                            inner_eca += v;
                        }
                        if v > max_eca {
                            max_eca = v;
                        }
                    }
                }
                let isolum_chroma_pass = peak_chromatic
                    && a.peaks[p].energy >= 2.2 * min_peak_energy
                    && c_eca >= 0.16 * min_patch_eca;
                let eca_pass = c_eca >= min_patch_eca
                    || (inner_eca >= min_inner6_eca && max_eca >= 0.16)
                    || isolum_chroma_pass;
                if eca_pass {
                    let (lam_min, lam_max) = patch16x16_shi_tomasi_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        frame_w,
                        qx,
                        qy,
                        scale_px,
                    );
                    if lam_min >= 0.00050 && lam_min >= 0.20 * lam_max {
                        best_st_score = lam_min;
                        best_patch_eca = c_eca.max(inner_eca * 3.0);
                        best_inner_eca = inner_eca;
                        best_outer_eca = (c_eca - inner_eca).max(0.0);
                    }
                }
            }

            if best_st_score <= 0.0 {
                for dy in [-snap_step, 0, snap_step] {
                    for dx in [-snap_step, 0, snap_step] {
                        let qx = (base_tlx + dx).clamp(st_min_tlx, st_max_tlx) as usize;
                        let qy = (base_tly + dy).clamp(st_min_tly, st_max_tly) as usize;

                        let mut p_eca = 0.0f32;
                        let mut inner_eca = 0.0f32;
                        let mut max_eca = 0.0f32;
                        for ry in core_lo..core_hi {
                            let roff = (qy + ry) * frame_w + qx;
                            let in_y = ry >= inner_lo && ry < inner_hi;
                            for rx in core_lo..core_hi {
                                let v = a.eca_res[roff + rx];
                                p_eca += v;
                                if in_y && rx >= inner_lo && rx < inner_hi {
                                    inner_eca += v;
                                }
                                if v > max_eca {
                                    max_eca = v;
                                }
                            }
                        }
                        let isolum_chroma_pass = peak_chromatic
                            && a.peaks[p].energy >= 2.2 * min_peak_energy
                            && p_eca >= 0.16 * min_patch_eca;
                        if p_eca < min_patch_eca
                            && !(inner_eca >= min_inner6_eca && max_eca >= 0.16)
                            && !isolum_chroma_pass
                        {
                            continue;
                        }
                        let eff_eca = p_eca.max(inner_eca * 3.0);

                        let (lam_min, lam_max) = patch16x16_shi_tomasi_simd128(
                            a.luma.as_ptr(),
                            a.chroma_rg.as_ptr(),
                            a.chroma_by.as_ptr(),
                            frame_w,
                            qx,
                            qy,
                            scale_px,
                        );
                        // Require genuine 2D corner/blob structure (reject flat blocks & 1D straight edges)
                        if lam_min < 0.00022 || lam_min < 0.072 * lam_max {
                            continue;
                        }
                        let iso_ratio = (lam_min / lam_max.max(1e-7)).min(0.55);
                        let center_bias = if dx == 0 && dy == 0 { 1.15 } else { 1.0 };
                        let score = lam_min * iso_ratio * (0.4 + eff_eca / (144.0 * s2)) * center_bias;
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
                    let dx = a.targets[m].pos_x - px;
                    let dy = a.targets[m].pos_y - py;
                    let dist2 = dx * dx + dy * dy;
                    if dist2 < seed_close_r2 {
                        too_close = true;
                        break;
                    }
                    let opp_chroma = has_opposite_chroma_pole(a.targets[m].anchor_sig, psig);
                    if cid > 0 && !opp_chroma {
                        let tgx = ((a.targets[m].pos_x as usize) / 16).min(grid_w - 1);
                        let tgy = ((a.targets[m].pos_y as usize) / 16).min(grid_h - 1);
                        if a.cc_visited[tgy * grid_w + tgx] as usize == cid {
                            has_same_pole_in_comp = true;
                            if !compact_projectile && dist2 < large_comp_sep_r2 {
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
                    let same_neutral_body =
                        !peak_chromatic && ((a.targets[m].anchor_sig & (1 << 31)) == 0) && val >= 8;
                    if same_neutral_body && dist2 < seed_neutral_r2 {
                        too_close = true;
                        break;
                    }
                    if a.targets[m].coasting > 0 && a.targets[m].coasting <= 28 {
                        let cf_m = (a.targets[m].coasting as f32).min(3.0);
                        let mex_x = a.targets[m].entry_x + cf_m * a.targets[m].entry_vx;
                        let mex_y = a.targets[m].entry_y + cf_m * a.targets[m].entry_vy;
                        let m_exited_border = mex_x < half_patch - 3.0 * s
                            || mex_x > (frame_w as f32) - half_patch + 3.0 * s
                            || mex_y < half_patch - 3.0 * s
                            || mex_y > (frame_h as f32) - half_patch + 3.0 * s;
                        if !m_exited_border {
                            let edx = a.targets[m].entry_x - px;
                            let edy = a.targets[m].entry_y - py;
                            let ed2 = edx * edx + edy * edy;
                            if (!opp_chroma && ed2 < coast_guard_r2)
                                || (a.targets[m].coasting <= 22
                                    && edy.abs() < 20.0 * s
                                    && edx.abs() < 68.0 * s)
                            {
                                too_close = true;
                                break;
                            }
                        }
                    }
                }
            }
            if !too_close && cid > 0 && !compact_projectile && has_same_pole_in_comp {
                let max_in_comp = if comp_cells_cnt[cid] > single_obj_cells { 4u8 } else { 3u8 };
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
                let cand_eca = best_patch_eca / (144.0 * s2);
                if cand_eca > 0.085 || compact_projectile {
                    let mut worst_m: Option<usize> = None;
                    let mut worst_score = 1e9f32;
                    for m in 0..MAX_TARGETS {
                        let t = &a.targets[m];
                        let evictable = t.confirmed == 0
                            || t.coasting >= 24
                            || (t.age >= 20 && t.max_excursion2 < (3.6 * s) * (3.6 * s));
                        if evictable {
                            let sc = t.vis_eca_ema
                                + if t.confirmed != 0 { 0.02 } else { 0.0 }
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
                let clear_bit = !(1u8 << m);
                for o in 0..MAX_TARGETS {
                    a.targets[o].proximity_mask &= clear_bit;
                }
                let tlx = best_tlx;
                let tly = best_tly;

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

                let tid = a.next_track_id;
                a.next_track_id += 1;

                a.targets[m].active = 1;
                a.targets[m].confirmed = 0; // Tentative until M-of-N (`3` hits + real motion)
                a.targets[m].just_confirmed = 0;
                a.targets[m].hit_mask_8 = 0x01;
                a.targets[m].static_streak = 0;
                a.targets[m].owner_slot = m as u8;
                a.targets[m].track_id = tid;
                a.targets[m].birth_frame = f_idx;
                a.targets[m].anchor_sig = a.targets[m].curr_sig;
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
                a.targets[m].max_excursion2 = 0.0;
                a.targets[m].vis_eca_ema = (best_patch_eca / (144.0 * s2)).min(0.35);
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
                a.targets[m].arc_raw_x[arc_slot_in] = px;
                a.targets[m].arc_raw_y[arc_slot_in] = py;
                a.targets[m].arc_vis[arc_slot_in] = 1;

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
            a.fista_lambda,
            fps,
            f_idx,
            a.stimulus_armed,
            a.stimulus_frame,
            frame_w,
            frame_h,
            s,
            a.scene_gravity_px_s2,
            a.scene_gravity_conf,
        );
        if consumed {
            any_consumed_stimulus = true;
        }
        if let Some(ay_obs) = pooled_ay {
            if a.targets[m].confirmed != 0 {
                if a.scene_gravity_conf < 0.05 || a.scene_gravity_px_s2 <= 10.0 * s {
                    a.scene_gravity_px_s2 = ay_obs;
                    a.scene_gravity_conf = 0.35;
                } else {
                    let rel_diff = (ay_obs - a.scene_gravity_px_s2).abs() / a.scene_gravity_px_s2;
                    if rel_diff < 0.45 {
                        a.scene_gravity_px_s2 = 0.84 * a.scene_gravity_px_s2 + 0.16 * ay_obs;
                        a.scene_gravity_conf = (a.scene_gravity_conf + 0.12).min(1.0);
                    } else {
                        a.scene_gravity_px_s2 = 0.94 * a.scene_gravity_px_s2 + 0.06 * ay_obs;
                    }
                }
            }
        }

        let s_disp = slot_disp;
        let fx = a.targets[m].fista_x[u_disp];
        let fy = a.targets[m].fista_y[u_disp];
        let rx = a.targets[m].ring_raw_x[s_disp];
        let ry = a.targets[m].ring_raw_y[s_disp];
        let vis_d = a.targets[m].ring_vis[s_disp];

        let mut has_past_vis = false;
        let mut has_future_vis = false;
        for u in 0..u_disp {
            let su = (slot_in + 1 + u) % RING_LEN;
            if a.targets[m].ring_vis[su] != 0 {
                has_past_vis = true;
            }
        }
        if !has_past_vis && a.targets[m].has_entry != 0 && (f_idx as usize) >= eff_k {
            let disp_f = f_idx - (eff_k as u32);
            if a.targets[m].entry_frame < disp_f
                && disp_f.saturating_sub(a.targets[m].entry_frame) <= 28
            {
                has_past_vis = true;
            }
        }
        for u in (u_disp + 1)..RING_LEN {
            let su = (slot_in + 1 + u) % RING_LEN;
            if a.targets[m].ring_vis[su] != 0 {
                has_future_vis = true;
            }
        }
        a.targets[m].bridged_at_disp = if vis_d == 0 && has_past_vis && has_future_vis { 1 } else { 0 };
        a.targets[m].disp_fista_x = fx;
        a.targets[m].disp_fista_y = fy;
        a.targets[m].disp_raw_x = rx;
        a.targets[m].disp_raw_y = ry;
        a.targets[m].disp_vis = if vis_d != 0 { 1.0 } else { 0.0 };

        // Detect bounded macroscopic stationary-pivot harmonic oscillators (e.g. metronomes / cantilevers)
        // vs. translating carriers with micro-tremor (where FISTA interior baseline removes drift).
        let mut vis_12 = 0usize;
        let mut sum_12_x = 0.0f32;
        let mut sum_12_y = 0.0f32;
        let mut min_12_x = 1e9f32;
        let mut max_12_x = -1e9f32;
        let mut min_12_y = 1e9f32;
        let mut max_12_y = -1e9f32;
        for u in 0..RING_LEN {
            let su = (slot_in + 1 + u) % RING_LEN;
            if a.targets[m].ring_vis[su] != 0 {
                let vx_u = a.targets[m].ring_raw_x[su];
                let vy_u = a.targets[m].ring_raw_y[su];
                vis_12 += 1;
                sum_12_x += vx_u;
                sum_12_y += vy_u;
                if vx_u < min_12_x {
                    min_12_x = vx_u;
                }
                if vx_u > max_12_x {
                    max_12_x = vx_u;
                }
                if vy_u < min_12_y {
                    min_12_y = vy_u;
                }
                if vy_u > max_12_y {
                    max_12_y = vy_u;
                }
            }
        }
        let macro_osc = if vis_12 >= 6 && a.targets[m].age >= 12 {
            let inv_v = 1.0 / (vis_12 as f32);
            let mean_12_x = sum_12_x * inv_v;
            let mean_12_y = sum_12_y * inv_v;
            let dx_sp = max_12_x - min_12_x;
            let dy_sp = max_12_y - min_12_y;
            let span2 = dx_sp * dx_sp + dy_sp * dy_sp;
            let sh_x = mean_12_x - a.targets[m].osc_mean_x;
            let sh_y = mean_12_y - a.targets[m].osc_mean_y;
            let shift2 = sh_x * sh_x + sh_y * sh_y;
            span2 > (8.0 * s) * (8.0 * s) && shift2 < (4.5 * s) * (4.5 * s)
        } else {
            false
        };

        // Revise tremor residual history across [f_idx - 8, f_idx], preserving centered interior
        // FISTA residuals (never overwriting with left-boundary u=0..2 free-boundary taper) and
        // avoiding right-boundary u=9..11 endpoint pinning on provisional lookahead taps.
        let rev_start = if macro_osc || a.targets[m].age <= 8 { 0usize } else { 3usize };
        let slope_int_x = 0.25 * (a.targets[m].fista_x[8] - a.targets[m].fista_x[4]);
        let slope_int_y = 0.25 * (a.targets[m].fista_y[8] - a.targets[m].fista_y[4]);
        for u in rev_start..RING_LEN {
            if (f_idx as usize) + u < (RING_LEN - 1) {
                continue;
            }
            let sample_idx = (f_idx as usize) + u - (RING_LEN - 1);
            let h_slot = sample_idx & 31;
            let su = (slot_in + 1 + u) % RING_LEN;
            if a.targets[m].ring_vis[su] != 0 && sample_idx >= (a.targets[m].birth_frame as usize) {
                let (base_x, base_y) = if macro_osc {
                    (a.targets[m].osc_mean_x, a.targets[m].osc_mean_y)
                } else if u > 8 && a.targets[m].age > 8 {
                    let du = (u - 8) as f32;
                    (
                        a.targets[m].fista_x[8] + du * slope_int_x,
                        a.targets[m].fista_y[8] + du * slope_int_y,
                    )
                } else {
                    (a.targets[m].fista_x[u], a.targets[m].fista_y[u])
                };
                a.targets[m].tremor_hist_x[h_slot] = a.targets[m].ring_raw_x[su] - base_x;
                a.targets[m].tremor_hist_y[h_slot] = a.targets[m].ring_raw_y[su] - base_y;
                a.targets[m].tremor_valid[h_slot] = 1;
            } else if macro_osc
                && a.targets[m].coasting <= 4
                && sample_idx >= (a.targets[m].birth_frame as usize)
            {
                a.targets[m].tremor_hist_x[h_slot] = a.targets[m].fista_x[u] - a.targets[m].osc_mean_x;
                a.targets[m].tremor_hist_y[h_slot] = a.targets[m].fista_y[u] - a.targets[m].osc_mean_y;
                a.targets[m].tremor_valid[h_slot] = 1;
            } else {
                a.targets[m].tremor_hist_x[h_slot] = 0.0;
                a.targets[m].tremor_hist_y[h_slot] = 0.0;
                a.targets[m].tremor_valid[h_slot] = 0;
            }
        }

        // Update 2D covariance & sign-continuous principal tremor axis (eliminates diagonal blind spot)
        let disp_h_slot = ((f_idx as usize) + 32 - eff_k) & 31;
        let disp_valid = (f_idx as usize) >= eff_k && a.targets[m].tremor_valid[disp_h_slot] != 0;
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

        let ax = a.targets[m].axis_x;
        let ay = a.targets[m].axis_y;
        let r_disp = if disp_valid {
            ax * a.targets[m].tremor_hist_x[disp_h_slot]
                + ay * a.targets[m].tremor_hist_y[disp_h_slot]
        } else {
            0.0
        };

        // Centered 15-tap Hilbert filter gated strictly on eff_k >= 7 and valid core taps (±1, ±3)
        a.targets[m].prev_phase_valid = a.targets[m].phase_valid;
        let mut active_tap_mask = 0u8;
        let mut hilb_raw = 0.0f32;
        if eff_k >= 7 && disp_valid && (f_idx as usize) >= eff_k + 7 {
            for (tap_idx, &(odd_k, coeff)) in H_TAPS.iter().enumerate() {
                let p_slot = (disp_h_slot + 32 - odd_k) & 31;
                let f_slot = (disp_h_slot + odd_k) & 31;
                if a.targets[m].tremor_valid[p_slot] != 0 && a.targets[m].tremor_valid[f_slot] != 0 {
                    let past_val =
                        ax * a.targets[m].tremor_hist_x[p_slot] + ay * a.targets[m].tremor_hist_y[p_slot];
                    let fut_val =
                        ax * a.targets[m].tremor_hist_x[f_slot] + ay * a.targets[m].tremor_hist_y[f_slot];
                    hilb_raw += coeff * (past_val - fut_val);
                    active_tap_mask |= 1u8 << tap_idx;
                }
            }
        }

        if (active_tap_mask & 0b0011) == 0b0011 {
            // Narrowband FIR gain equalization G(omega_hat) over active tap pairs
            let f_est = if a.targets[m].tremor_freq_hz >= 1.0 {
                a.targets[m].tremor_freq_hz
            } else {
                8.5
            };
            let omega_hat = TWO_PI * f_est / fps;
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

            if a.targets[m].prev_phase_valid != 0 && amp > 0.025 * s {
                // Signed unwrapped phase difference (no max(0) upward noise bias)
                let dphi = wrap_pi(phi - prev_phi);
                let inst_omega = dphi * fps;
                if a.targets[m].tremor_omega_ema == 0.0 {
                    a.targets[m].tremor_omega_ema = inst_omega;
                } else {
                    a.targets[m].tremor_omega_ema =
                        0.85 * a.targets[m].tremor_omega_ema + 0.15 * inst_omega;
                }
                let pred_step = a.targets[m].tremor_omega_ema / fps;
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

            // Predict present-time phase phi_now(t + tau_out) from delayed phi(t - K)
            let lead_time_sec = (eff_k as f32) / fps + a.audio_out_latency_sec;
            let phi_now = wrap_pi(phi + a.targets[m].tremor_omega_ema * lead_time_sec);

            a.targets[m].prev_tremor_phase = prev_phi;
            a.targets[m].prev_pred_phase_now = a.targets[m].pred_phase_now;
            a.targets[m].tremor_r = r_disp;
            a.targets[m].tremor_h = hilb;
            a.targets[m].tremor_phase = phi;
            a.targets[m].pred_phase_now = phi_now;
            a.targets[m].tremor_amp = amp;
            a.targets[m].phase_valid = 1;
        } else {
            a.targets[m].phase_valid = 0;
            a.targets[m].tremor_r = r_disp;
            a.targets[m].tremor_h = 0.0;
            a.targets[m].tremor_amp = 0.0;
            a.targets[m].tremor_coherence *= 0.88;
            if !disp_valid {
                a.targets[m].tremor_freq_hz = 0.0;
            }
        }

        // Mode C: Scan lookahead window for sparse D2*X acceleration impulses, deduplicated on physical event time & space
        let mut max_z = 0.0f32;
        let mut max_raw_dv = 0.0f32;
        let mut strike_u = u_disp;
        let scan_start = u_disp.min(9);
        let rate_norm = (fps / 60.0) * (fps / 60.0);

        for zk in scan_start..10usize {
            let su0 = (slot_in + 1 + zk) % RING_LEN;
            let su1 = (slot_in + 1 + zk + 1) % RING_LEN;
            let su2 = (slot_in + 1 + zk + 2) % RING_LEN;
            // Require observed visibility across the 3-point stencil so occluded priors never fire strikes
            if a.targets[m].ring_vis[su0] == 0
                || a.targets[m].ring_vis[su1] == 0
                || a.targets[m].ring_vis[su2] == 0
            {
                continue;
            }
            let dvx = (a.targets[m].ring_raw_x[su2] - a.targets[m].ring_raw_x[su1])
                - (a.targets[m].ring_raw_x[su1] - a.targets[m].ring_raw_x[su0]);
            let dvy = (a.targets[m].ring_raw_y[su2] - a.targets[m].ring_raw_y[su1])
                - (a.targets[m].ring_raw_y[su1] - a.targets[m].ring_raw_y[su0]);
            let raw_dv = wasm_sqrtf(dvx * dvx + dvy * dvy);
            let zmag = wasm_sqrtf(
                a.targets[m].fista_zx[zk] * a.targets[m].fista_zx[zk]
                    + a.targets[m].fista_zy[zk] * a.targets[m].fista_zy[zk],
            ) * rate_norm;
            if zmag > max_z {
                max_z = zmag;
                max_raw_dv = raw_dv;
                strike_u = zk + 1;
            }
        }
        a.targets[m].l1_impulse_mag = max_z;

        if a.targets[m].confirmed != 0
            && a.targets[m].max_excursion2 >= (8.0 * s) * (8.0 * s)
            && max_z > 0.85 * s
            && max_raw_dv >= 1.50 * s
            && (f_idx as usize) + strike_u >= (RING_LEN - 1)
        {
            let event_frame = ((f_idx as usize) + strike_u - (RING_LEN - 1)) as u32;
            let min_refractory_frames = ((0.13 * fps) as u32).max(6);
            let mut allowed = a.targets[m].has_struck == 0
                || event_frame >= a.targets[m].last_strike_event_frame.saturating_add(min_refractory_frames);

            let sx_pos = a.targets[m].fista_x[strike_u];
            let sy_pos = a.targets[m].fista_y[strike_u];
            let mut paired_partner: Option<usize> = None;
            let partner_r2 = (58.0 * s) * (58.0 * s);
            let spatial_dedup_r2 = (42.0 * s) * (42.0 * s);

            for o in 0..MAX_TARGETS {
                if o != m && a.targets[o].active != 0 && a.targets[o].has_struck != 0 {
                    let df = if event_frame >= a.targets[o].last_strike_event_frame {
                        event_frame - a.targets[o].last_strike_event_frame
                    } else {
                        a.targets[o].last_strike_event_frame - event_frame
                    };
                    if df <= 5 {
                        let dx = sx_pos - a.targets[o].strike_x;
                        let dy = sy_pos - a.targets[o].strike_y;
                        let d2 = dx * dx + dy * dy;
                        if d2 <= partner_r2 && paired_partner.is_none() {
                            paired_partner = Some(o);
                        }
                        if allowed && df <= 4 && d2 <= spatial_dedup_r2 {
                            allowed = false;
                        }
                    }
                }
            }

            let same_event_refine = a.targets[m].has_struck != 0
                && event_frame.abs_diff(a.targets[m].last_strike_event_frame) <= 1;

            if allowed || same_event_refine || paired_partner.is_some() {
                if let Some((in_d, defl_d, rest_e, ovx, ovy, pre_ok)) =
                    measure_strike_kinematics(&a.targets[m], slot_in, strike_u, s)
                {
                    if pre_ok || a.targets[m].has_struck == 0 || allowed {
                        a.targets[m].strike_in_deg = in_d;
                        a.targets[m].strike_deflect_deg = defl_d;
                        a.targets[m].strike_restitution = rest_e;
                    }
                    a.targets[m].strike_out_vx = ovx;
                    a.targets[m].strike_out_vy = ovy;

                    if let Some(o) = paired_partner {
                        let o_vx = a.targets[o].strike_out_vx;
                        let o_vy = a.targets[o].strike_out_vy;
                        let m_v2 = ovx * ovx + ovy * ovy;
                        let o_v2 = o_vx * o_vx + o_vy * o_vy;
                        if m_v2 >= (0.65 * s) * (0.65 * s) && o_v2 >= (0.65 * s) * (0.65 * s) {
                            let rad_to_deg = 180.0 / PI;
                            let th1 = fast_atan2f(-o_vy, o_vx) * rad_to_deg;
                            let th2 = fast_atan2f(-ovy, ovx) * rad_to_deg;
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
                a.targets[m].pivot_valid = 0;

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

        // Non-causal display-time trigonometric velocity, flight angle theta(t - K), & angular rate
        let (fvx_d, fvy_d) = if u_disp >= 1 && u_disp + 1 < RING_LEN {
            (
                0.5 * (a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp - 1]),
                0.5 * (a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp - 1]),
            )
        } else if u_disp + 1 < RING_LEN {
            (
                a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp],
                a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp],
            )
        } else {
            (
                a.targets[m].fista_x[11] - a.targets[m].fista_x[10],
                a.targets[m].fista_y[11] - a.targets[m].fista_y[10],
            )
        };
        let fspeed_f = wasm_sqrtf(fvx_d * fvx_d + fvy_d * fvy_d);
        a.targets[m].speed_px_s = fspeed_f * fps;
        if fspeed_f >= 0.32 * s {
            a.targets[m].heading_deg = fast_atan2f(-fvy_d, fvx_d) * (180.0 / PI);
        }
        if u_disp >= 1 && u_disp + 1 < RING_LEN {
            let vpx = a.targets[m].fista_x[u_disp] - a.targets[m].fista_x[u_disp - 1];
            let vpy = a.targets[m].fista_y[u_disp] - a.targets[m].fista_y[u_disp - 1];
            let vnx = a.targets[m].fista_x[u_disp + 1] - a.targets[m].fista_x[u_disp];
            let vny = a.targets[m].fista_y[u_disp + 1] - a.targets[m].fista_y[u_disp];
            let min_w2 = (0.55 * s) * (0.55 * s);
            if vpx * vpx + vpy * vpy >= min_w2 && vnx * vnx + vny * vny >= min_w2 {
                let dth_d = wrap_pi(fast_atan2f(-vny, vnx) - fast_atan2f(-vpy, vpx));
                let inst_w_deg = dth_d * fps * (180.0 / PI);
                a.targets[m].ang_vel_deg_s =
                    0.62 * a.targets[m].ang_vel_deg_s + 0.38 * inst_w_deg.clamp(-720.0, 720.0);
            } else {
                a.targets[m].ang_vel_deg_s *= 0.78;
            }
        } else {
            a.targets[m].ang_vel_deg_s *= 0.78;
        }

        // Mode B: Kasa 2x2 Algebraic Circle Fit on 24-frame raw sub-pixel history (with 12-frame peg-snap priority)
        // Note: back-and-forth harmonic oscillation on a rigid tether (e.g. 2.38 Hz metronome) always satisfies
        // (x - xc)^2 + (y - yc)^2 = R^2 even across high-acceleration turnarounds!
        let mut pivot_updated = false;
        let smooth_or_osc = max_z <= 0.85 * s
            || macro_osc
            || (a.targets[m].phase_valid != 0 && a.targets[m].tremor_coherence > 0.28);
        if vis_12 >= 6 && a.targets[m].age >= 8 && a.targets[m].strike_fired == 0 && smooth_or_osc {
            let mut full_x = [0.0f32; 24];
            let mut full_y = [0.0f32; 24];
            let mut n_full = 0usize;
            let mut recent_x = [0.0f32; 12];
            let mut recent_y = [0.0f32; 12];
            let mut n_recent = 0usize;

            let mut u24 = 24usize;
            let mut miss24 = 0usize;
            while u24 > 0 {
                u24 -= 1;
                let sa = (arc_slot_in + 1 + u24) % 24;
                if a.targets[m].arc_vis[sa] != 0 {
                    miss24 = 0;
                    let ax_u = a.targets[m].arc_raw_x[sa];
                    let ay_u = a.targets[m].arc_raw_y[sa];
                    if n_full > 0 {
                        let jump2 = (ax_u - full_x[n_full - 1]) * (ax_u - full_x[n_full - 1])
                            + (ay_u - full_y[n_full - 1]) * (ay_u - full_y[n_full - 1]);
                        if jump2 > (38.0 * s) * (38.0 * s) {
                            break;
                        }
                    }
                    full_x[n_full] = ax_u;
                    full_y[n_full] = ay_u;
                    n_full += 1;
                    if u24 >= 12 && n_recent < 12 {
                        recent_x[n_recent] = ax_u;
                        recent_y[n_recent] = ay_u;
                        n_recent += 1;
                    }
                } else {
                    miss24 += 1;
                    if miss24 >= 3 {
                        break;
                    }
                }
            }

            // Prefer recent 12-frame arc when strongly curved (fast peg transition); else use up to 24 frames!
            let circle_cand = fit_kasa_circle_subpx(&recent_x, &recent_y, n_recent, s, 3.2)
                .or_else(|| fit_kasa_circle_subpx(&full_x, &full_y, n_full, s, 1.8));

            if let Some((xc, yc, rad, my_c, _sag)) = circle_cand {
                // True pendulum/metronome tethers sit within ~67 deg of vertical plumb (|my_c - yc| >= 0.38 * R)
                // with physical arm radius <= 235*s, rejecting sideways-curving highway lanes & rail grazes.
                let hanging_pend = rad <= 235.0 * s
                    && (my_c - yc) >= (14.0 * s).max(0.38 * rad)
                    && a.targets[m].is_ballistic == 0;
                let inverted_metro = rad <= 235.0 * s
                    && (yc - my_c) >= (14.0 * s).max(0.38 * rad)
                    && a.targets[m].is_ballistic == 0
                    && (macro_osc
                        || (a.targets[m].phase_valid != 0
                            && a.targets[m].tremor_coherence > 0.28));
                if hanging_pend || inverted_metro {
                    if a.targets[m].pivot_valid == 0 || a.targets[m].pivot_r <= 1.0 {
                        a.targets[m].pivot_x = xc;
                        a.targets[m].pivot_y = yc;
                        a.targets[m].pivot_r = rad;
                    } else {
                        let alpha_p =
                            if (rad - a.targets[m].pivot_r).abs() > 0.26 * a.targets[m].pivot_r {
                                0.72
                            } else {
                                0.22
                            };
                        a.targets[m].pivot_x =
                            (1.0 - alpha_p) * a.targets[m].pivot_x + alpha_p * xc;
                        a.targets[m].pivot_y =
                            (1.0 - alpha_p) * a.targets[m].pivot_y + alpha_p * yc;
                        a.targets[m].pivot_r =
                            (1.0 - alpha_p) * a.targets[m].pivot_r + alpha_p * rad;
                    }
                    a.targets[m].pivot_valid = 18;
                    pivot_updated = true;
                }
            }
        }
        if !pivot_updated {
            let bottom_piv_ballistic =
                a.targets[m].is_ballistic != 0 && a.targets[m].pivot_y > fy;
            if a.targets[m].coasting > 4 || bottom_piv_ballistic {
                a.targets[m].pivot_valid = 0;
            } else {
                a.targets[m].pivot_valid = a.targets[m].pivot_valid.saturating_sub(1);
            }
        }

        if a.targets[m].confirmed != 0 {
            let pinned_bonus = if a.focus_target_id != 0 && a.targets[m].track_id == a.focus_target_id {
                10_000.0
            } else {
                0.0
            };
            let pip_bonus = if a.targets[m].track_id == a.pip_target_id {
                24.0
            } else {
                0.0
            };
            let exc_px = wasm_sqrtf(a.targets[m].max_excursion2) / s;
            let speed_px = (a.targets[m].vel_x.abs() + a.targets[m].vel_y.abs()) / s;
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
                    let piv_b = if a.targets[m].pivot_valid > 0 && exc_px >= 12.0 {
                        45.0
                    } else {
                        0.0
                    };
                    osc_b + piv_b
                }
                _ => {
                    if exc_px >= 8.0 {
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
            let motion_bonus = exc_px.min(140.0) * 0.65
                + speed_px.min(20.0) * 2.5
                + chroma_bonus
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
            a.focus_target_id = 0;
        }
    }

    if any_consumed_stimulus || (a.stimulus_armed != 0 && f_idx.saturating_sub(a.stimulus_frame) > 90) {
        a.stimulus_armed = 0;
    }

    // ------------------------------------------------------------------------
    // 7. FISTA-Stabilized Eulerian Temporal-Contrast PiP & Present-Phase Strobe
    // ------------------------------------------------------------------------
    let mut strobe_fired = 0.0f32;
    let pip_center_off = 0.5 * (pip_dim as f32) - 0.5;

    if primary_idx >= 0 {
        let pm = primary_idx as usize;
        let tid = a.targets[pm].track_id;
        let fx = a.targets[pm].disp_fista_x;
        let fy = a.targets[pm].disp_fista_y;
        let reset_ref = a.pip_target_id != tid;
        a.pip_target_id = tid;
        let gain = a.eulerian_gain;

        for py in 0..pip_dim {
            for px in 0..pip_dim {
                let sx = fx - pip_center_off + (px as f32);
                let sy = fy - pip_center_off + (py as f32);
                let (r, g, b, luma) = sample_rgba_bilinear(&a.rgba_ring[slot_disp], frame_w, frame_h, sx, sy);
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
        }
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
        a.telemetry[13] = a.targets[pm].tremor_freq_hz;
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
        let ebase = 512 + m * 8;
        let kbase = 576 + m * 8;
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
        let exited_past_disp = a.targets[m].coasting >= 2
            && a.targets[m].disp_vis < 0.5
            && a.targets[m].bridged_at_disp == 0
            && (ex_now_x < half_patch - 3.0 * s
                || ex_now_x > (frame_w as f32) - half_patch + 3.0 * s
                || ex_now_y < half_patch - 3.0 * s
                || ex_now_y > (frame_h as f32) - half_patch + 3.0 * s);
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
            a.telemetry[base + 44 + u] = a.targets[m].ring_vis[su] as f32;
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

    a.telemetry[640] = a.scene_gravity_px_s2;
    a.telemetry[641] = a.scene_gravity_conf;
    a.telemetry[642] = a.scatter_deg;
    a.telemetry[643] = a.scatter_x;
    a.telemetry[644] = a.scatter_y;
    a.telemetry[645] = a.scatter_th1_deg;
    a.telemetry[646] = a.scatter_th2_deg;
    a.telemetry[647] = a.scatter_event_frame as f32;

    for p in 0..MAX_TARGETS {
        let pbase = 480 + p * 4;
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

    a.frame_idx = f_idx.wrapping_add(1);
    slot_disp as u32
}

/// Complete single-call Wasm frame tick (runs pre_tick -> certified internal Jacobi Fiedler solve -> post_tick).
#[no_mangle]
pub unsafe extern "C" fn chronos_tick(dt_sec: f32) -> u32 {
    chronos_pre_tick(dt_sec);
    solve_fiedler_jacobi_16x16(arena_mut());
    chronos_post_tick(dt_sec)
}
