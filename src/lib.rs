#![no_std]

pub mod simd_ops;

use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_floor, f32x4_mul, f32x4_splat, f32x4_sqrt, v128, v128_load, v128_store};
use core::panic::PanicInfo;
use core::ptr::addr_of_mut;
pub use simd_ops::{eca_subspace_project_simd128, padic_ultrametric_val, sad16x16_patch_simd128};
use simd_ops::{
    compute_motion_grid_and_heal_bg_simd128, eca_subspace_project_masked_simd128,
    haar_morton_signature_scaled_simd128, haar_morton_signature_simd128, hsum_f32x4,
    load_pooled4_f32, padic_distance_from_val, patch16x16_ac_energy_simd128,
    rgba_to_planar_simd128, sad_patch_scaled_simd128, ssd16x16_photometric_simd128,
    update_stat_streak_and_mask_simd128,
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
pub const TELEMETRY_LEN: usize = 576;

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
    pub _pad: [u8; 3],
}

#[repr(C, align(16))]
pub struct TargetTrack {
    // 16x16 Luma + Chroma R-G + Chroma B-Y reference templates (3,072 bytes)
    pub tpl_luma: [f32; 256],
    pub tpl_rg: [f32; 256],
    pub tpl_by: [f32; 256],

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

    // Kinematics, spectral & 2-adic identity invariants
    pub active: u8,
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
    pub _pad1: u8,

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
    pub pred_cx: f32,
    pub pred_cy: f32,
    pub entry_x: f32,
    pub entry_y: f32,
    pub entry_vx: f32,
    pub entry_vy: f32,
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

    // Mode A/B/C Biophysical Lab outputs
    pub apex_x: f32,
    pub apex_y: f32,
    pub apex_frames: f32,
    pub l1_impulse_mag: f32,
    pub last_strike_event_frame: u32,
    pub strike_lead_frames: f32,
    pub strike_vel: f32,
    pub step_ref_x: f32,
    pub step_ref_y: f32,
    pub stim_ref_x: f32,
    pub stim_ref_y: f32,
    pub step_onset_frame: u32,
    pub saccade_latency_ms: f32,
    pub step_settle_ms: f32,
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
}

// Compile-time structural bound: Arena + 16 KiB stack + .rodata <= 256 pages (16,777,216 B)
const _: () = assert!(core::mem::size_of::<Arena>() <= 16_720_000);

static mut ARENA: Arena = unsafe { core::mem::zeroed() };

#[inline(always)]
unsafe fn arena_mut() -> &'static mut Arena {
    &mut *addr_of_mut!(ARENA)
}

// ============================================================================
// Helper Kernels: Patch Copy, Bilinear Subpixel Sampling, Connected Components
// ============================================================================

unsafe fn capture_template_scaled(
    luma: &[f32; MAX_FRAME_PIXELS],
    rg: &[f32; MAX_FRAME_PIXELS],
    by: &[f32; MAX_FRAME_PIXELS],
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
    for ry in 0..16usize {
        let src_row = (y0 + ry * scale_px) * frame_w + x0;
        let dst_row = ry * 16;
        for vec_idx in 0..4usize {
            let ioff = src_row + vec_idx * cstep;
            let doff = dst_row + vec_idx * 4;
            v128_store(
                t.tpl_luma.as_mut_ptr().add(doff) as *mut v128,
                load_pooled4_f32(luma.as_ptr().add(ioff), frame_w, scale_px),
            );
            v128_store(
                t.tpl_rg.as_mut_ptr().add(doff) as *mut v128,
                load_pooled4_f32(rg.as_ptr().add(ioff), frame_w, scale_px),
            );
            v128_store(
                t.tpl_by.as_mut_ptr().add(doff) as *mut v128,
                load_pooled4_f32(by.as_ptr().add(ioff), frame_w, scale_px),
            );
        }
    }
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

    let min_split_cells = 8 * scale_px * scale_px;
    let min_dg2 = 5 * (scale_px as i32) * (scale_px as i32);

    for start_idx in 0..grid_cells {
        if a.cc_visited[start_idx] != 0 || a.cc_energy[start_idx] < seed_thresh {
            continue;
        }

        let mut q_head = 0usize;
        let mut q_tail = 0usize;
        a.cc_queue[q_tail] = start_idx as u16;
        q_tail += 1;
        a.cc_visited[start_idx] = 1;

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
                        a.cc_visited[nidx] = 1;
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
                _pad: [0; 3],
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
) -> bool {
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

                    // Provisional secant tangent on initial re-emergence frame u1 == 11 (revised next frame)
                    let (v1x, v1y) = if u1 + 1 < RING_LEN && vis[u1 + 1] != 0 {
                        (yx[u1 + 1] - yx[u1], yy[u1 + 1] - yy[u1])
                    } else {
                        ((yx[u1] - yx[u0]) / dt, (yy[u1] - yy[u0]) / dt)
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
                        yx[k] = (yx[u0] + dt * v_tail_x).clamp(tail_margin, (frame_w as f32) - tail_margin);
                        yy[k] = (yy[u0] + dt * v_tail_y).clamp(tail_margin, (frame_h as f32) - tail_margin);
                        weff[k] = 0.60;
                    }
                } else if gap_end < RING_LEN {
                    let (v_lead_x, v_lead_y) = if gap_end + 1 < RING_LEN && vis[gap_end + 1] != 0 {
                        (yx[gap_end + 1] - yx[gap_end], yy[gap_end + 1] - yy[gap_end])
                    } else {
                        (vx[gap_end], vy[gap_end])
                    };
                    for k in 0..gap_end {
                        let dt = (gap_end - k) as f32;
                        yx[k] = yx[gap_end] - dt * v_lead_x;
                        yy[k] = yy[gap_end] - dt * v_lead_y;
                        weff[k] = 0.65;
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

    // Mode A: Physical-units Parabolic Ballistic Apex Predictor with half-step velocity correction
    // Screen Y increases downward: ascending throw has v_phys_y < 0 and downward gravity a_phys_y > 0.
    let a_frame_x = xk[11] - 2.0 * xk[10] + xk[9];
    let a_frame_y = 0.5 * ((yk[11] - 2.0 * yk[10] + yk[9]) + (yk[10] - 2.0 * yk[9] + yk[8]));
    let v_frame_x = (xk[11] - xk[10]) + 0.5 * a_frame_x;
    let v_frame_y = (yk[11] - yk[10]) + 0.5 * a_frame_y;

    let v_phys_x = v_frame_x * fps;
    let v_phys_y = v_frame_y * fps;
    let a_phys_y = a_frame_y * fps * fps;

    if vis_count > 0 && vis[11] != 0 && v_phys_y < -20.0 * scale_f && a_phys_y > 120.0 * scale_f {
        let t_apex_sec = (-v_phys_y / a_phys_y).min(0.75);
        t.apex_frames = t_apex_sec * fps;
        t.apex_x = (xk[11] + v_phys_x * t_apex_sec).clamp(8.0 * scale_f, (frame_w as f32) - 8.0 * scale_f);
        t.apex_y = (yk[11] + v_phys_y * t_apex_sec + 0.5 * a_phys_y * t_apex_sec * t_apex_sec)
            .clamp(6.0 * scale_f, (frame_h as f32) - 6.0 * scale_f);
    } else {
        t.apex_frames = 0.0;
        t.apex_x = xk[11];
        t.apex_y = yk[11];
    }

    // Mode B: Stimulus-to-onset reflex latency (when stimulus armed) + spontaneous step settle duration
    let mut consumed_stimulus = false;
    let stim_disp2 = (8.0 * scale_f) * (8.0 * scale_f);
    let step_disp2 = (18.0 * scale_f) * (18.0 * scale_f);
    let settle_thresh = 38.0 * scale_f;
    if vis[11] != 0 {
        if stimulus_armed != 0 && frame_idx > stimulus_frame {
            let elapsed_stim = frame_idx - stimulus_frame;
            if elapsed_stim <= 90 {
                let sdx = xk[11] - t.stim_ref_x;
                let sdy = yk[11] - t.stim_ref_y;
                if sdx * sdx + sdy * sdy > stim_disp2 {
                    t.saccade_latency_ms = (elapsed_stim as f32) * (1000.0 / fps.max(24.0));
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
                t.step_settle_ms = (elapsed_frames as f32) * (1000.0 / fps.max(24.0));
                if t.saccade_latency_ms <= 0.0 {
                    t.saccade_latency_ms = t.step_settle_ms;
                }
                t.step_onset_frame = 0;
            }
        }
    }
    consumed_stimulus
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
    let (frame_w, frame_h, grid_w, grid_h, _scale_px, _pip_dim, s) = mode_dims(a.res_mode);
    let frame_pixels = frame_w * frame_h;
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
    //    Nodes 0..7: predicted active tracks; Nodes 8..15: current candidate peaks.
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
            let px = if coasting == 0 {
                a.targets[m].pos_x + a.targets[m].vel_x
            } else {
                a.targets[m].entry_x + (coasting as f32) * a.targets[m].entry_vx
            };
            let py = if coasting == 0 {
                a.targets[m].pos_y + a.targets[m].vel_y
            } else {
                a.targets[m].entry_y + (coasting as f32) * a.targets[m].entry_vy
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

/// Phase 2 of frame tick (executes after Fiedler vector & eigengap are written to Arena):
/// Performs spectrally-regularized 2-adic patch association, continuous SSD sub-pixel refinement,
/// 2x2 Kalman baseline tracking, Dual Projected-FISTA smoothing, non-causal Hilbert equalization,
/// and biophysical telemetry export.
#[no_mangle]
pub unsafe extern "C" fn chronos_post_tick(dt_sec: f32) -> u32 {
    let a = arena_mut();
    let (frame_w, frame_h, _grid_w, _grid_h, scale_px, pip_dim, s) = mode_dims(a.res_mode);
    let s2 = s * s;
    let patch_dim = 16 * scale_px;
    let half_patch = 8.0 * s;
    let max_tlx_f = (frame_w - patch_dim - 1) as f32;
    let max_tly_f = (frame_h - patch_dim - 1) as f32;
    let max_tlx_i = (frame_w - patch_dim - 1) as i32;
    let max_tly_i = (frame_h - patch_dim - 1) as i32;
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
    let eff_k = (a.lookahead_k.min(f_idx) as usize).min(RING_LEN - 1);
    let slot_disp = ((f_idx as usize) + RING_LEN - eff_k) % RING_LEN;
    a.last_display_slot = slot_disp as u32;

    // Eigengap confidence weight for sign-invariant spectral cost C_spec = gamma * c_gap * (v_m - v_{8+p})^2
    let gap_conf = if a.fiedler_lambda2 > 1e-5 {
        let denom = a.fiedler_lambda2 + a.fiedler_gap + 1e-4;
        let g = if a.fiedler_gap > 0.0 {
            a.fiedler_gap / denom
        } else {
            // Fallback when caller only sets lambda2 without lambda3 gap
            0.5
        };
        g.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eff_spec_gamma = a.spectral_gamma * gap_conf;

    // ------------------------------------------------------------------------
    // 3. Spectrally-Regularized 2-Adic Ultrametric + SIMD128 Patch Tracking
    // ------------------------------------------------------------------------
    let mut peak_claimed = [0u8; MAX_TARGETS];
    // (match_cx, match_cy, total_cost, raw_patch_cost, best_sig, best_val, is_visible)
    let mut track_matched_pos = [(0.0f32, 0.0f32, 1e9f32, 1e9f32, 0u32, 0u32, false); MAX_TARGETS];
    let spec_lookup_r2 = (24.0 * s) * (24.0 * s);
    let fine_steps_sd = [3i32, 1i32];
    let fine_steps_hd = [6i32, 2i32, 1i32];
    let fine_steps: &[i32] = if scale_px == 2 {
        &fine_steps_hd
    } else {
        &fine_steps_sd
    };

    for m in 0..MAX_TARGETS {
        if a.targets[m].active == 0 {
            continue;
        }
        let anchor_sig = a.targets[m].anchor_sig;
        let coasting = a.targets[m].coasting;
        let pred_cx = a.targets[m].pred_cx;
        let pred_cy = a.targets[m].pred_cy;
        let v_track = a.fiedler_vec[m];

        let mut seeds_x = [0.0f32; 10];
        let mut seeds_y = [0.0f32; 10];
        let mut n_seeds = 0usize;

        seeds_x[n_seeds] = pred_cx;
        seeds_y[n_seeds] = pred_cy;
        n_seeds += 1;

        if coasting > 0 {
            seeds_x[n_seeds] = a.targets[m].entry_x;
            seeds_y[n_seeds] = a.targets[m].entry_y;
            n_seeds += 1;
            if coasting <= 6 && n_seeds < 10 {
                seeds_x[n_seeds] = a.targets[m].entry_x - (coasting as f32) * a.targets[m].entry_vx;
                seeds_y[n_seeds] = a.targets[m].entry_y - (coasting as f32) * a.targets[m].entry_vy;
                n_seeds += 1;
            }
        }

        for p in 0..a.num_peaks {
            if n_seeds >= 10 {
                break;
            }
            let px = a.peaks[p].x;
            let py = a.peaks[p].y;
            let ref_x = if coasting == 0 { pred_cx } else { a.targets[m].entry_x };
            let ref_y = if coasting == 0 { pred_cy } else { a.targets[m].entry_y };
            let dist2 = (px - ref_x) * (px - ref_x) + (py - ref_y) * (py - ref_y);
            let max_r2 = if coasting == 0 {
                (48.0 * s) * (48.0 * s)
            } else {
                (180.0 * s) * (180.0 * s)
            };
            if dist2 <= max_r2 {
                seeds_x[n_seeds] = px;
                seeds_y[n_seeds] = py;
                n_seeds += 1;
            }
        }

        let mut best_total_cost = 1e9f32;
        let mut best_raw_cost = 1e9f32;
        let mut best_tlx = ((pred_cx - half_patch).clamp(1.0, max_tlx_f)) as usize;
        let mut best_tly = ((pred_cy - half_patch).clamp(1.0, max_tly_f)) as usize;
        let mut best_sig = a.targets[m].curr_sig;

        // Coarse search across kinematic prediction, entry anchor, and candidate ECA peaks
        for seed_idx in 0..n_seeds {
            let base_tlx = ((seeds_x[seed_idx] - half_patch).clamp(2.0, max_tlx_f - 1.0)) as i32;
            let base_tly = ((seeds_y[seed_idx] - half_patch).clamp(2.0, max_tly_f - 1.0)) as i32;

            let step = if seed_idx == 0 { 6 * s_i32 } else { 4 * s_i32 };
            for dy in [-step, 0, step] {
                for dx in [-step, 0, step] {
                    let cx = (base_tlx + dx).clamp(1, max_tlx_i) as usize;
                    let cy = (base_tly + dy).clamp(1, max_tly_i) as usize;
                    let mut csig = 0u32;
                    let raw_cost = sad_patch_scaled_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        cx,
                        cy,
                        a.targets[m].tpl_luma.as_ptr(),
                        a.targets[m].tpl_rg.as_ptr(),
                        a.targets[m].tpl_by.as_ptr(),
                        anchor_sig,
                        &mut csig,
                        scale_px,
                    );
                    if has_opposite_chroma_pole(anchor_sig, csig) {
                        continue;
                    }

                    let mut spec_pen = 0.0f32;
                    if eff_spec_gamma > 0.0 && a.num_peaks > 0 {
                        let cand_px = (cx as f32) + half_patch;
                        let cand_py = (cy as f32) + half_patch;
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

                    let total_cost = raw_cost + spec_pen;
                    if total_cost < best_total_cost {
                        best_total_cost = total_cost;
                        best_raw_cost = raw_cost;
                        best_tlx = cx;
                        best_tly = cy;
                        best_sig = csig;
                    }
                }
            }
        }

        // Hierarchical fine refinement (`3 -> 1` in SD, `6 -> 2 -> 1` in HD) around winning patch
        for &step in fine_steps {
            let center_x = best_tlx as i32;
            let center_y = best_tly as i32;
            for dy in [-step, 0, step] {
                for dx in [-step, 0, step] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let cx = (center_x + dx).clamp(1, max_tlx_i) as usize;
                    let cy = (center_y + dy).clamp(1, max_tly_i) as usize;
                    let mut csig = 0u32;
                    let raw_cost = sad_patch_scaled_simd128(
                        a.luma.as_ptr(),
                        a.chroma_rg.as_ptr(),
                        a.chroma_by.as_ptr(),
                        a.eca_res.as_ptr(),
                        frame_w,
                        cx,
                        cy,
                        a.targets[m].tpl_luma.as_ptr(),
                        a.targets[m].tpl_rg.as_ptr(),
                        a.targets[m].tpl_by.as_ptr(),
                        anchor_sig,
                        &mut csig,
                        scale_px,
                    );
                    if has_opposite_chroma_pole(anchor_sig, csig) {
                        continue;
                    }

                    let mut spec_pen = 0.0f32;
                    if eff_spec_gamma > 0.0 && a.num_peaks > 0 {
                        let cand_px = (cx as f32) + half_patch;
                        let cand_py = (cy as f32) + half_patch;
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

                    let total_cost = raw_cost + spec_pen;
                    if total_cost < best_total_cost {
                        best_total_cost = total_cost;
                        best_raw_cost = raw_cost;
                        best_tlx = cx;
                        best_tly = cy;
                        best_sig = csig;
                    }
                }
            }
        }

        let best_val = padic_ultrametric_val(anchor_sig, best_sig);
        let d2 = padic_distance_from_val(best_val);

        let mut acc_eca = f32x4_splat(0.0);
        let cstep = 4 * scale_px;
        for ry in 0..16usize {
            let r_off = (best_tly + ry * scale_px) * frame_w + best_tlx;
            for vec_idx in 0..4usize {
                acc_eca = f32x4_add(
                    acc_eca,
                    load_pooled4_f32(a.eca_res.as_ptr().add(r_off + vec_idx * cstep), frame_w, scale_px),
                );
            }
        }
        let eca_mean = hsum_f32x4(acc_eca) * (1.0 / 256.0);
        let app_err = best_raw_cost - 0.55 * d2 + 0.18 * eca_mean.min(0.35);

        let is_visible = best_total_cost < 1e8
            && app_err < 0.25
            && !has_opposite_chroma_pole(anchor_sig, best_sig)
            && (eca_mean > 0.010 || app_err < 0.095);

        // Smooth Continuous Photometric SSD 1D Parabolic Sub-Pixel Refinement (free of discrete jumps)
        let mut sub_dx = 0.0f32;
        let mut sub_dy = 0.0f32;
        if is_visible
            && best_tlx >= 2
            && best_tlx + patch_dim + 1 < frame_w
            && best_tly >= 2
            && best_tly + patch_dim + 1 < frame_h
        {
            let c_0 = ssd16x16_photometric_simd128(
                a.luma.as_ptr(),
                a.chroma_rg.as_ptr(),
                a.chroma_by.as_ptr(),
                frame_w,
                best_tlx,
                best_tly,
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                scale_px,
            );
            let c_xm = ssd16x16_photometric_simd128(
                a.luma.as_ptr(),
                a.chroma_rg.as_ptr(),
                a.chroma_by.as_ptr(),
                frame_w,
                best_tlx - 1,
                best_tly,
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                scale_px,
            );
            let c_xp = ssd16x16_photometric_simd128(
                a.luma.as_ptr(),
                a.chroma_rg.as_ptr(),
                a.chroma_by.as_ptr(),
                frame_w,
                best_tlx + 1,
                best_tly,
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                scale_px,
            );
            let denom_x = 2.0 * (c_xm - 2.0 * c_0 + c_xp);
            if denom_x > 1e-5 {
                sub_dx = ((c_xm - c_xp) / denom_x).clamp(-0.48, 0.48);
            }

            let c_ym = ssd16x16_photometric_simd128(
                a.luma.as_ptr(),
                a.chroma_rg.as_ptr(),
                a.chroma_by.as_ptr(),
                frame_w,
                best_tlx,
                best_tly - 1,
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                scale_px,
            );
            let c_yp = ssd16x16_photometric_simd128(
                a.luma.as_ptr(),
                a.chroma_rg.as_ptr(),
                a.chroma_by.as_ptr(),
                frame_w,
                best_tlx,
                best_tly + 1,
                a.targets[m].tpl_luma.as_ptr(),
                a.targets[m].tpl_rg.as_ptr(),
                a.targets[m].tpl_by.as_ptr(),
                scale_px,
            );
            let denom_y = 2.0 * (c_ym - 2.0 * c_0 + c_yp);
            if denom_y > 1e-5 {
                sub_dy = ((c_ym - c_yp) / denom_y).clamp(-0.48, 0.48);
            }
        }

        let match_cx = (best_tlx as f32) + half_patch + sub_dx;
        let match_cy = (best_tly as f32) + half_patch + sub_dy;
        track_matched_pos[m] = (
            match_cx,
            match_cy,
            best_total_cost,
            best_raw_cost,
            best_sig,
            best_val,
            is_visible,
        );
    }

    // Spatial conflict resolution ordered by spectrally-regularized `best_total_cost`
    // Also record pairwise spatial interaction history (`proximity_mask`) so internal swap
    // auditing only attributes cross-track identity swaps between tracks that physically met.
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
            let close_now = (dx * dx + dy * dy < prox_now_r2) || (pdx * pdx + pdy * pdy < prox_now_r2);
            let close_coast = (a.targets[i].coasting > 0 || a.targets[j].coasting > 0)
                && (dx * dx + dy * dy < prox_coast_r2);
            if close_now || close_coast {
                a.targets[i].proximity_mask |= 1u8 << j;
                a.targets[j].proximity_mask |= 1u8 << i;
            }
        }
    }

    let mut order = [0usize, 1, 2, 3, 4, 5, 6, 7];
    for i in 0..MAX_TARGETS {
        for j in (i + 1)..MAX_TARGETS {
            if track_matched_pos[order[j]].2 < track_matched_pos[order[i]].2 {
                let tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    let conflict_r2 = (11.0 * s) * (11.0 * s);
    let claim_r2 = (22.0 * s) * (22.0 * s);
    let v_clamp = 18.0 * s;
    let iso_r2 = (32.0 * s) * (32.0 * s);

    for idx in 0..MAX_TARGETS {
        let m = order[idx];
        if a.targets[m].active == 0 {
            continue;
        }
        let (mcx, mcy, _mtotal, mrawcost, msig, mval, mut vis) = track_matched_pos[m];

        if vis {
            for prev_idx in 0..idx {
                let pm = order[prev_idx];
                if a.targets[pm].active != 0 && track_matched_pos[pm].6 {
                    let dx = mcx - track_matched_pos[pm].0;
                    let dy = mcy - track_matched_pos[pm].1;
                    if dx * dx + dy * dy < conflict_r2 {
                        vis = false;
                        track_matched_pos[m].6 = false;
                        break;
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
                }
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
                // First re-emergence frame: mark provisional state; Hermite bridge uses chord tangent
                a.targets[m].vel_x = 0.0;
                a.targets[m].vel_y = 0.0;
                a.targets[m].reemerged_last = 1;
            } else if a.targets[m].reemerged_last != 0 {
                // Second visible frame after occlusion: lock directly onto observed post-emergence velocity
                a.targets[m].vel_x = inst_vx.clamp(-v_clamp, v_clamp);
                a.targets[m].vel_y = inst_vy.clamp(-v_clamp, v_clamp);
                a.targets[m].reemerged_last = 0;
            } else {
                a.targets[m].vel_x = (0.62 * a.targets[m].vel_x + 0.38 * inst_vx).clamp(-v_clamp, v_clamp);
                a.targets[m].vel_y = (0.62 * a.targets[m].vel_y + 0.38 * inst_vy).clamp(-v_clamp, v_clamp);
            }

            a.targets[m].pos_x = mcx;
            a.targets[m].pos_y = mcy;
            a.targets[m].curr_sig = msig;
            a.targets[m].last_padic_val = mval;
            a.targets[m].coasting = 0;
            a.targets[m].age += 1;

            a.targets[m].ring_raw_x[slot_in] = mcx;
            a.targets[m].ring_raw_y[slot_in] = mcy;
            a.targets[m].ring_vx[slot_in] = a.targets[m].vel_x;
            a.targets[m].ring_vy[slot_in] = a.targets[m].vel_y;
            a.targets[m].ring_sig[slot_in] = msig;
            a.targets[m].ring_vis[slot_in] = 1;

            if mrawcost < 0.06 && mval >= 8 {
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
                        frame_w,
                        frame_h,
                        scale_px,
                        tlx,
                        tly,
                        0.02,
                        &mut a.targets[m],
                    );
                }
            }

            // Internal self-audit for cross-target chromatic/2-adic signature inversion
            // Restricted to tracks with recorded spatial proximity and adjusted for the 5-bit
            // canonical neutral prefix (`0_01_01`) when `bit 31 == 0`.
            let own_val = padic_ultrametric_val(a.targets[m].anchor_sig, msig);
            let mut other_best_slot = a.targets[m].owner_slot as usize;
            let mut other_best_val = own_val;
            let cand_chromatic = (msig & (1 << 31)) != 0;
            let min_swap_val = if cand_chromatic { 6 } else { 12 };
            for o in 0..MAX_TARGETS {
                if o != m
                    && a.targets[o].active != 0
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
        } else {
            if a.targets[m].coasting == 0 {
                a.targets[m].entry_x = a.targets[m].pos_x;
                a.targets[m].entry_y = a.targets[m].pos_y;
                a.targets[m].entry_vx = a.targets[m].vel_x;
                a.targets[m].entry_vy = a.targets[m].vel_y;
                a.targets[m].reemerged_last = 0;
            }
            a.targets[m].coasting += 1;
            a.targets[m].age += 1;

            let coast_x = (a.targets[m].entry_x
                + (a.targets[m].coasting as f32) * a.targets[m].entry_vx)
                .clamp(half_patch, (frame_w as f32) - half_patch);
            let coast_y = (a.targets[m].entry_y
                + (a.targets[m].coasting as f32) * a.targets[m].entry_vy)
                .clamp(half_patch, (frame_h as f32) - half_patch);

            a.targets[m].ring_raw_x[slot_in] = coast_x;
            a.targets[m].ring_raw_y[slot_in] = coast_y;
            a.targets[m].ring_vx[slot_in] = a.targets[m].entry_vx;
            a.targets[m].ring_vy[slot_in] = a.targets[m].entry_vy;
            a.targets[m].ring_sig[slot_in] = a.targets[m].curr_sig;
            a.targets[m].ring_vis[slot_in] = 0;

            if (a.targets[m].age < 6 && a.targets[m].coasting >= 3) || a.targets[m].coasting > 75 {
                a.targets[m].active = 0;
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
        }

        a.kalman[m].ring_x[slot_in] = a.kalman[m].x;
        a.kalman[m].ring_y[slot_in] = a.kalman[m].y;
    }

    // ------------------------------------------------------------------------
    // 5. Auto-Seed New Moving Physical Targets from Unclaimed ECA Peaks
    // ------------------------------------------------------------------------
    if f_idx >= 4 {
        let seed_close_r2 = (28.0 * s) * (28.0 * s);
        let seed_chroma_r2 = (72.0 * s) * (72.0 * s);
        let seed_neutral_r2 = (46.0 * s) * (46.0 * s);
        let core_lo = 2 * scale_px;
        let core_hi = 14 * scale_px;
        let min_patch_eca = 144.0 * s2 * 0.055;

        for p in 0..a.num_peaks {
            if peak_claimed[p] != 0 || a.peaks[p].energy < 3.6 * s {
                continue;
            }
            let px = a.peaks[p].x;
            let py = a.peaks[p].y;
            let psig = a.peaks[p].sig;
            let peak_chromatic = (psig & (1 << 31)) != 0;

            let mut too_close = false;
            for m in 0..MAX_TARGETS {
                if a.targets[m].active != 0 {
                    let dx = a.targets[m].pos_x - px;
                    let dy = a.targets[m].pos_y - py;
                    let dist2 = dx * dx + dy * dy;
                    if dist2 < seed_close_r2 {
                        too_close = true;
                        break;
                    }
                    let val = padic_ultrametric_val(a.targets[m].anchor_sig, psig);
                    let same_chroma = peak_chromatic && !has_opposite_chroma_pole(a.targets[m].anchor_sig, psig) && val >= 5;
                    if same_chroma && dist2 < seed_chroma_r2 {
                        too_close = true;
                        break;
                    }
                    let same_neutral_body = !peak_chromatic && ((a.targets[m].anchor_sig & (1 << 31)) == 0) && val >= 8;
                    if same_neutral_body && dist2 < seed_neutral_r2 {
                        too_close = true;
                        break;
                    }
                    if a.targets[m].coasting > 0 && a.targets[m].coasting <= 32 {
                        let edx = a.targets[m].entry_x - px;
                        let edy = a.targets[m].entry_y - py;
                        if !has_opposite_chroma_pole(a.targets[m].anchor_sig, psig)
                            || (edy.abs() < 24.0 * s && edx.abs() < 120.0 * s)
                        {
                            too_close = true;
                            break;
                        }
                    }
                }
            }
            if too_close {
                continue;
            }

            let check_tlx = ((px - half_patch).clamp(1.0, max_tlx_f)) as usize;
            let check_tly = ((py - half_patch).clamp(1.0, max_tly_f)) as usize;
            let mut patch_eca = 0.0f32;
            for ry in core_lo..core_hi {
                let roff = (check_tly + ry) * frame_w + check_tlx;
                for rx in core_lo..core_hi {
                    patch_eca += a.eca_res[roff + rx];
                }
            }
            if patch_eca < min_patch_eca {
                continue;
            }
            let ac_energy = patch16x16_ac_energy_simd128(
                a.luma.as_ptr(),
                a.chroma_rg.as_ptr(),
                a.chroma_by.as_ptr(),
                frame_w,
                check_tlx,
                check_tly,
                scale_px,
            );
            if ac_energy < 0.00055 {
                continue;
            }

            let mut free_slot: Option<usize> = None;
            for m in 0..MAX_TARGETS {
                if a.targets[m].active == 0 {
                    free_slot = Some(m);
                    break;
                }
            }
            if let Some(m) = free_slot {
                let clear_bit = !(1u8 << m);
                for o in 0..MAX_TARGETS {
                    a.targets[o].proximity_mask &= clear_bit;
                }
                let tlx = check_tlx;
                let tly = check_tly;

                a.targets[m] = core::mem::zeroed();
                capture_template_scaled(
                    &a.luma,
                    &a.chroma_rg,
                    &a.chroma_by,
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

        let consumed = solve_noncausal_fista_target(
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
        );
        if consumed {
            any_consumed_stimulus = true;
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

        // Revise tremor residual history across [f_idx - 8, f_idx], preserving centered interior
        // FISTA residuals (never overwriting with left-boundary u=0..2 free-boundary taper) and
        // avoiding right-boundary u=9..11 endpoint pinning on provisional lookahead taps.
        let rev_start = if a.targets[m].age > 8 { 3usize } else { 0usize };
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
                let (base_x, base_y) = if u > 8 && a.targets[m].age > 8 {
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

        // Centered 15-tap Hilbert filter gated strictly on eff_k >= 7 and valid observed taps
        a.targets[m].prev_phase_valid = a.targets[m].phase_valid;
        let mut valid_pairs = 0usize;
        let mut hilb_raw = 0.0f32;
        if eff_k >= 7 && disp_valid && (f_idx as usize) >= eff_k + 7 {
            for &(odd_k, coeff) in H_TAPS.iter() {
                let p_slot = (disp_h_slot + 32 - odd_k) & 31;
                let f_slot = (disp_h_slot + odd_k) & 31;
                if a.targets[m].tremor_valid[p_slot] != 0 && a.targets[m].tremor_valid[f_slot] != 0 {
                    let past_val =
                        ax * a.targets[m].tremor_hist_x[p_slot] + ay * a.targets[m].tremor_hist_y[p_slot];
                    let fut_val =
                        ax * a.targets[m].tremor_hist_x[f_slot] + ay * a.targets[m].tremor_hist_y[f_slot];
                    hilb_raw += coeff * (past_val - fut_val);
                    valid_pairs += 1;
                }
            }
        }

        if valid_pairs == 4 {
            // Narrowband FIR gain equalization G(omega_hat) eliminates 120Hz passband droop & 2f phase ripple
            let f_est = if a.targets[m].tremor_freq_hz >= 2.0 {
                a.targets[m].tremor_freq_hz
            } else {
                8.5
            };
            let omega_hat = TWO_PI * f_est / fps;
            let mut g_gain = 0.0f32;
            for &(odd_k, coeff) in H_TAPS.iter() {
                g_gain += 2.0 * coeff * fast_sinf((odd_k as f32) * omega_hat);
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
                a.targets[m].tremor_freq_hz =
                    (a.targets[m].tremor_omega_ema.abs() / TWO_PI).clamp(0.0, 30.0);
            } else if amp <= 0.015 * s {
                a.targets[m].tremor_omega_ema *= 0.85;
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
            if !disp_valid {
                a.targets[m].tremor_freq_hz = 0.0;
            }
        }

        // Mode C: Scan lookahead window for sparse D2*X acceleration impulses, deduplicated on physical event time
        let mut max_z = 0.0f32;
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
            let zmag = wasm_sqrtf(
                a.targets[m].fista_zx[zk] * a.targets[m].fista_zx[zk]
                    + a.targets[m].fista_zy[zk] * a.targets[m].fista_zy[zk],
            ) * rate_norm;
            if zmag > max_z {
                max_z = zmag;
                strike_u = zk + 1;
            }
        }
        a.targets[m].l1_impulse_mag = max_z;

        if max_z > 0.85 * s && (f_idx as usize) + strike_u >= (RING_LEN - 1) {
            let event_frame = ((f_idx as usize) + strike_u - (RING_LEN - 1)) as u32;
            let min_refractory_frames = ((0.13 * fps) as u32).max(6);
            let allowed = a.targets[m].has_struck == 0
                || event_frame >= a.targets[m].last_strike_event_frame.saturating_add(min_refractory_frames);

            if allowed {
                a.targets[m].has_struck = 1;
                a.targets[m].last_strike_event_frame = event_frame;
                a.targets[m].strike_fired = 1;

                let sx_pos = a.targets[m].fista_x[strike_u];
                let sy_pos = a.targets[m].fista_y[strike_u];
                let qx = if sx_pos >= (frame_w as f32 * 0.5) { 1u8 } else { 0u8 };
                let qy = if sy_pos >= (frame_h as f32 * 0.5) { 2u8 } else { 0u8 };
                let zone = qx + qy;
                let lead_frames = (strike_u.saturating_sub(u_disp)) as f32;
                let vel = (max_z / (5.0 * s)).clamp(0.2, 1.0);

                a.targets[m].strike_zone = zone;
                a.targets[m].strike_lead_frames = lead_frames;
                a.targets[m].strike_vel = vel;

                if perc_mask == 0 {
                    perc_lead = lead_frames;
                    perc_zone = zone as f32;
                    perc_vel = vel;
                }
                perc_mask |= 1 << m;
            }
        }

        let score = (a.targets[m].age.min(120) as f32) + if vis_d != 0 { 50.0 } else { 0.0 };
        if score > best_primary_score {
            best_primary_score = score;
            primary_idx = m as i32;
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
    // 8. Populate Zero-Copy Flat Telemetry Array
    // ------------------------------------------------------------------------
    let mut active_count = 0u32;
    for m in 0..MAX_TARGETS {
        if a.targets[m].active != 0 {
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
        if a.targets[m].active == 0 {
            a.telemetry[base] = 0.0;
            for q in 0..8 {
                a.telemetry[ebase + q] = 0.0;
            }
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
        a.telemetry[ebase + 4] = a.targets[m].last_strike_event_frame as f32;
        a.telemetry[ebase + 5] = a.targets[m].fista_duality_gap;
        a.telemetry[ebase + 6] = a.targets[m].axis_x;
        a.telemetry[ebase + 7] = a.targets[m].axis_y;
    }

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
