//! In-Situ Physics Identification Engine (`#![no_std]`, zero host imports, zero heap allocation).
//!
//! Combines the three locked Wave-2 physics subsystems (`#16` / `#17`):
//! - Section A: Weak-form STRidge + AICc Governing-ODE Discovery (`phys_regime` `0..=8`, drag `beta`,
//!   thrust `accel`, burnout `JUMP_BOOST_BURNOUT = 2`, parachute deploy `JUMP_CHUTE_DEPLOY = 3`).
//! - Section B: Taubin Circle Fit + Kasa bias diagnostic, Galileo Peg Radius Jump (`JUMP_GALILEO_PEG = 1`),
//!   Jacobi AGM Elliptic Period, Damping `zeta` / `Q`, Specific Energy & Angular Momentum,
//!   Bounded Rosenstein Lyapunov Divergence (`64`-slot stride-2 reservoir), and Kuramoto Order `R_sync`.
//! - Section C: Causal Two-Body Elastic/Inelastic Collision & Momentum/Energy Conservation Analyzer
//!   (`JUMP_COLLISION = 4`, `PHYS_REGIME_IMPULSIVE = 7`, inverse mass ratio, restitution `e_n`).

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{
    f32x4, f32x4_add, f32x4_extract_lane, f32x4_gt, f32x4_lt, f32x4_mul, f32x4_nearest,
    f32x4_splat, f32x4_sub, f64x2_extract_lane, f64x2_splat, f64x2_sqrt, v128, v128_bitselect,
};
#[cfg(not(target_arch = "wasm32"))]
use core::arch::wasm32::{f64x2_extract_lane, f64x2_splat, f64x2_sqrt};

#[allow(unused_imports)]
use crate::{
    fast_atan2f, fast_cosf, fast_expf, fast_log10f, fast_sinf,
    wasm_floorf, wasm_sqrtf, wrap_pi,
};

pub const HIST_RING_LEN: usize = 64;
pub const HIST_MASK: usize = 63;
const HIST_RING_MASK: usize = HIST_MASK;
pub const MAX_TARGETS: usize = 16;
pub const MAX_WEAK_ROWS: usize = 12;

pub const PHYS_REGIME_LINEAR_CV: u8 = 0;
pub const PHYS_REGIME_BALLISTIC: u8 = 1;
pub const PHYS_REGIME_BALLISTIC_DRAG: u8 = 2;
pub const PHYS_REGIME_POWERED_BOOST: u8 = 3;
pub const PHYS_REGIME_PARACHUTE: u8 = 4;
pub const PHYS_REGIME_PENDULUM: u8 = 5;
pub const PHYS_REGIME_MICRO_TREMOR: u8 = 6;
pub const PHYS_REGIME_IMPULSIVE: u8 = 7;
pub const PHYS_REGIME_CURVED_CTRV: u8 = 8;

pub const JUMP_NONE: u8 = 0;
pub const JUMP_GALILEO_PEG: u8 = 1;
pub const JUMP_BOOST_BURNOUT: u8 = 2;
pub const JUMP_CHUTE_DEPLOY: u8 = 3;
pub const JUMP_COLLISION: u8 = 4;

const CONSTRAINT_JUMP_TWO_BODY: u8 = JUMP_COLLISION;

const DIRECT_COLLISION_RADIUS_SD: f32 = 36.0;
const CHAIN_COLLISION_RADIUS_SD: f32 = 75.0;
const PAIR_SCAN_PROX_SD: f32 = 105.0;
const BLUR_HALF_WIN: u32 = 2;
const TWO_PI_64: f64 = 6.28318530717958647692;

// =============================================================================
// Shared Zero-Import Crate Fast-Math Adapters
// =============================================================================

#[inline(always)]
fn sqrt_f32(x: f32) -> f32 {
    wasm_sqrtf(x.max(0.0))
}

#[inline(always)]
fn wasm_sqrt_f64(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

#[inline(always)]
fn fast_lnf(x: f32) -> f32 {
    fast_log10f(x) * 2.30258509299
}

#[inline(always)]
fn wasm_sqrtd(x: f64) -> f64 {
    wasm_sqrt_f64(x.max(0.0))
}

#[inline(always)]
fn abs_f32(x: f32) -> f32 {
    x.abs()
}

#[inline(always)]
fn min_f32(a: f32, b: f32) -> f32 {
    a.min(b)
}

#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    a.max(b)
}

#[inline(always)]
fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    x.clamp(lo, hi)
}

#[inline(always)]
fn hypot_f32(x: f32, y: f32) -> f32 {
    sqrt_f32(x * x + y * y)
}

#[inline(always)]
fn hypot_f64(x: f64, y: f64) -> f64 {
    let ax = x.abs();
    let ay = y.abs();
    let hi = ax.max(ay);
    if hi == 0.0 {
        0.0
    } else {
        let r = ax.min(ay) / hi;
        hi * wasm_sqrtd(1.0 + r * r)
    }
}

#[inline(always)]
fn f64_fits_f32(x: f64) -> bool {
    x.is_finite() && x.abs() <= (f32::MAX as f64)
}

#[inline(always)]
fn fast_sin_cos(x: f32) -> (f32, f32) {
    (fast_sinf(x), fast_cosf(x))
}

// =============================================================================
// Section A: Weak-Form STRidge + AICc In-Situ Governing-ODE Discovery
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OdeTrackState {
    pub prior_flight: u8,
    pub last_regime: u8,
    pub latched_jump_code: u8,
    pub jump_hold_rem: u8,
    pub burnout_cooldown: u8,
    pub chute_cooldown: u8,
    pub has_ticked: u8,
    pub _pad: u8,
    pub last_tick_frame: u32,
    pub beta_ema: f32,
    pub thrust_ema: f32,
    pub latched_delta_beta: f32,
    pub omega_est_rad_s: f32,
    pub g_eff_px_s2: f32,
    pub sigma_pos_px: f32,
    pub conf_ema: f32,
    pub ode_r2_ema: f32,
}

impl OdeTrackState {
    pub const fn zeroed() -> Self {
        Self::new()
    }
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        *self = Self::zeroed();
    }
    #[allow(dead_code)]
    pub const fn new() -> Self {
        Self {
            prior_flight: 0,
            last_regime: PHYS_REGIME_LINEAR_CV,
            latched_jump_code: JUMP_NONE,
            jump_hold_rem: 0,
            burnout_cooldown: 0,
            chute_cooldown: 0,
            has_ticked: 0,
            _pad: 0,
            last_tick_frame: 0,
            beta_ema: 0.0,
            thrust_ema: 0.0,
            latched_delta_beta: 0.0,
            omega_est_rad_s: 0.0,
            g_eff_px_s2: 0.0,
            sigma_pos_px: 0.0,
            conf_ema: 0.0,
            ode_r2_ema: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OdeResult {
    pub phys_regime: u8,
    pub constraint_jump_code: u8,
    pub active_terms: u8,
    pub eff_samples: u8,
    pub phys_regime_conf: f32,
    pub ode_r2: f32,
    pub aero_drag_beta_px_inv: f32,
    pub thrust_accel_px_s2: f32,
    pub delta_beta_px_inv: f32,
    pub est_g_px_s2: f32,
    pub est_omega_rad_s: f32,
    pub sigma_pos_px: f32,
}

impl OdeResult {
    pub const fn zeroed() -> Self {
        Self::zero(0.0)
    }
    #[allow(dead_code)]
    pub const fn new() -> Self {
        Self::zero(0.0)
    }
    pub const fn zero(sigma: f32) -> Self {
        Self {
            phys_regime: PHYS_REGIME_LINEAR_CV,
            constraint_jump_code: JUMP_NONE,
            active_terms: 0,
            eff_samples: 0,
            phys_regime_conf: 0.0,
            ode_r2: 0.0,
            aero_drag_beta_px_inv: 0.0,
            thrust_accel_px_s2: 0.0,
            delta_beta_px_inv: 0.0,
            est_g_px_s2: 0.0,
            est_omega_rad_s: 0.0,
            sigma_pos_px: sigma,
        }
    }
}

/// Reusable Arena-resident scratchpad shared sequentially across all 16 active targets.
/// Keeps Wasm call stack near zero (< 1 KiB).
#[repr(C)]
pub struct PhysicsIdScratch {
    pub ts_v: [f32; HIST_RING_LEN],
    pub xs_v: [f32; HIST_RING_LEN],
    pub ys_v: [f32; HIST_RING_LEN],
    pub t_arr: [f32; HIST_RING_LEN],
    pub xc: [f32; HIST_RING_LEN],
    pub yc: [f32; HIST_RING_LEN],
    pub vx: [f32; HIST_RING_LEN],
    pub vy: [f32; HIST_RING_LEN],
    pub u1: [f32; HIST_RING_LEN],
    pub sim_x: [f32; HIST_RING_LEN],
    pub sim_y: [f32; HIST_RING_LEN],
    pub tmp_n: [f32; HIST_RING_LEN],
    pub clean_idx: [u8; HIST_RING_LEN],
    // Per-row weak weights (64 entries max, single-row reuse)
    pub w_alpha: [f32; HIST_RING_LEN],
    pub w_beta: [f32; HIST_RING_LEN],
    pub w_gamma: [f32; HIST_RING_LEN],
    // Macro weak design bank (M <= 12 -> 2M <= 24 rows)
    pub y_vec: [f32; 2 * MAX_WEAK_ROWS],
    pub x_trans: [[f32; 4]; 2 * MAX_WEAK_ROWS],
    pub x_ctrv: [[f32; 2]; 2 * MAX_WEAK_ROWS],
    pub y_u1: [f32; MAX_WEAK_ROWS],
    pub x_harm1d: [[f32; 3]; MAX_WEAK_ROWS],
    pub y_u2: [f32; MAX_WEAK_ROWS],
    pub x_u2: [[f32; 2]; MAX_WEAK_ROWS],
    // Fast weak bank (M <= 12 rows) for micro-tremor (6) & multi-band pendulum seeds
    pub y_fast: [f32; MAX_WEAK_ROWS],
    pub x_fast: [[f32; 4]; MAX_WEAK_ROWS],
}

impl PhysicsIdScratch {
    pub const fn zeroed() -> Self {
        Self::new()
    }
    /// In-place array fill avoids constructing a 5.1 KiB struct literal on the shadow stack.
    pub fn reset(&mut self) {
        self.ts_v.fill(0.0);
        self.xs_v.fill(0.0);
        self.ys_v.fill(0.0);
        self.t_arr.fill(0.0);
        self.xc.fill(0.0);
        self.yc.fill(0.0);
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.u1.fill(0.0);
        self.sim_x.fill(0.0);
        self.sim_y.fill(0.0);
        self.tmp_n.fill(0.0);
        self.clean_idx.fill(0);
        self.w_alpha.fill(0.0);
        self.w_beta.fill(0.0);
        self.w_gamma.fill(0.0);
        self.y_vec.fill(0.0);
        self.x_trans.fill([0.0; 4]);
        self.x_ctrv.fill([0.0; 2]);
        self.y_u1.fill(0.0);
        self.x_harm1d.fill([0.0; 3]);
        self.y_u2.fill(0.0);
        self.x_u2.fill([0.0; 2]);
        self.y_fast.fill(0.0);
        self.x_fast.fill([0.0; 4]);
    }
    pub const fn new() -> Self {
        Self {
            ts_v: [0.0; HIST_RING_LEN],
            xs_v: [0.0; HIST_RING_LEN],
            ys_v: [0.0; HIST_RING_LEN],
            t_arr: [0.0; HIST_RING_LEN],
            xc: [0.0; HIST_RING_LEN],
            yc: [0.0; HIST_RING_LEN],
            vx: [0.0; HIST_RING_LEN],
            vy: [0.0; HIST_RING_LEN],
            u1: [0.0; HIST_RING_LEN],
            sim_x: [0.0; HIST_RING_LEN],
            sim_y: [0.0; HIST_RING_LEN],
            tmp_n: [0.0; HIST_RING_LEN],
            clean_idx: [0; HIST_RING_LEN],
            w_alpha: [0.0; HIST_RING_LEN],
            w_beta: [0.0; HIST_RING_LEN],
            w_gamma: [0.0; HIST_RING_LEN],
            y_vec: [0.0; 2 * MAX_WEAK_ROWS],
            x_trans: [[0.0; 4]; 2 * MAX_WEAK_ROWS],
            x_ctrv: [[0.0; 2]; 2 * MAX_WEAK_ROWS],
            y_u1: [0.0; MAX_WEAK_ROWS],
            x_harm1d: [[0.0; 3]; MAX_WEAK_ROWS],
            y_u2: [0.0; MAX_WEAK_ROWS],
            x_u2: [[0.0; 2]; MAX_WEAK_ROWS],
            y_fast: [0.0; MAX_WEAK_ROWS],
            x_fast: [[0.0; 4]; MAX_WEAK_ROWS],
        }
    }
}

pub struct OdeWindowInput<'a> {
    pub x_world: &'a [f32],
    pub y_world: &'a [f32],
    pub t_sec: &'a [f64],
    pub vis: &'a [u8],
    pub scale_s: f32,
}

#[inline]
fn solve_3x3(mut a: [[f32; 3]; 3], mut b: [f32; 3]) -> Option<[f32; 3]> {
    let mut c = 0usize;
    while c < 3 {
        let mut piv = c;
        let mut best = abs_f32(a[c][c]);
        let mut r = c + 1;
        while r < 3 {
            let v = abs_f32(a[r][c]);
            if v > best {
                best = v;
                piv = r;
            }
            r += 1;
        }
        if best < 1.0e-12 {
            return None;
        }
        if piv != c {
            let tmp_row = a[c];
            a[c] = a[piv];
            a[piv] = tmp_row;
            let tb = b[c];
            b[c] = b[piv];
            b[piv] = tb;
        }
        let inv = 1.0 / a[c][c];
        let mut r2 = c + 1;
        while r2 < 3 {
            let fac = a[r2][c] * inv;
            let mut k = c;
            while k < 3 {
                a[r2][k] -= fac * a[c][k];
                k += 1;
            }
            b[r2] -= fac * b[c];
            r2 += 1;
        }
        c += 1;
    }
    let mut x = [0.0f32; 3];
    let mut i = 3usize;
    while i > 0 {
        i -= 1;
        let mut s = b[i];
        let mut j = i + 1;
        while j < 3 {
            s -= a[i][j] * x[j];
            j += 1;
        }
        if abs_f32(a[i][i]) < 1.0e-12 {
            return None;
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

#[inline]
fn solve_4x4_dim(mut a: [[f32; 4]; 4], mut b: [f32; 4], p: usize) -> Option<[f32; 4]> {
    if p == 0 || p > 4 {
        return None;
    }
    let mut c = 0usize;
    while c < p {
        let mut piv = c;
        let mut best = abs_f32(a[c][c]);
        let mut r = c + 1;
        while r < p {
            let v = abs_f32(a[r][c]);
            if v > best {
                best = v;
                piv = r;
            }
            r += 1;
        }
        if best < 1.0e-12 {
            return None;
        }
        if piv != c {
            let tmp_row = a[c];
            a[c] = a[piv];
            a[piv] = tmp_row;
            let tb = b[c];
            b[c] = b[piv];
            b[piv] = tb;
        }
        let inv = 1.0 / a[c][c];
        let mut r2 = c + 1;
        while r2 < p {
            let fac = a[r2][c] * inv;
            let mut k = c;
            while k < p {
                a[r2][k] -= fac * a[c][k];
                k += 1;
            }
            b[r2] -= fac * b[c];
            r2 += 1;
        }
        c += 1;
    }
    let mut x = [0.0f32; 4];
    let mut i = p;
    while i > 0 {
        i -= 1;
        let mut s = b[i];
        let mut j = i + 1;
        while j < p {
            s -= a[i][j] * x[j];
            j += 1;
        }
        if abs_f32(a[i][i]) < 1.0e-12 {
            return None;
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Column-normalized ridge solve over subset `cols[..m]` -> (xi[4], xi_norm[4], rss).
#[inline]
fn fit_ridge_subset<const C: usize>(
    rows: &[[f32; C]],
    n_rows: usize,
    y: &[f32],
    cols: &[usize],
    lam: f32,
) -> Option<([f32; 4], [f32; 4], f32)> {
    let m = cols.len();
    if m == 0 || m > 4 || n_rows == 0 {
        return None;
    }
    let mut col_norms = [0.0f32; 4];
    let mut i = 0usize;
    while i < m {
        let gc = cols[i];
        let mut ss = 0.0f32;
        let mut r = 0usize;
        while r < n_rows {
            let v = rows[r][gc];
            ss += v * v;
            r += 1;
        }
        let cn = sqrt_f32(ss);
        if cn <= 1.0e-10 {
            return None;
        }
        col_norms[i] = cn;
        i += 1;
    }
    let mut gram = [[0.0f32; 4]; 4];
    let mut rhs = [0.0f32; 4];
    let mut r = 0usize;
    while r < n_rows {
        let yr = y[r];
        let mut a = 0usize;
        while a < m {
            let va = rows[r][cols[a]] / col_norms[a];
            rhs[a] += va * yr;
            let mut b = a;
            while b < m {
                gram[a][b] += va * (rows[r][cols[b]] / col_norms[b]);
                b += 1;
            }
            a += 1;
        }
        r += 1;
    }
    i = 0;
    while i < m {
        gram[i][i] += lam;
        let mut j = 0usize;
        while j < i {
            gram[i][j] = gram[j][i];
            j += 1;
        }
        i += 1;
    }
    let xi_norm = solve_4x4_dim(gram, rhs, m)?;
    let mut xi = [0.0f32; 4];
    i = 0;
    while i < m {
        xi[i] = xi_norm[i] / col_norms[i];
        i += 1;
    }
    let mut rss = 0.0f32;
    r = 0;
    while r < n_rows {
        let mut pred = 0.0f32;
        let mut a = 0usize;
        while a < m {
            pred += rows[r][cols[a]] * xi[a];
            a += 1;
        }
        let e = y[r] - pred;
        rss += e * e;
        r += 1;
    }
    Some((xi, xi_norm, rss))
}

/// Unified column-normalized STRidge with unregularized final polish (`lam = 1e-6`).
fn stridge_cols<const C: usize>(
    rows: &[[f32; C]],
    n_rows: usize,
    y: &[f32],
    col_map: &[usize],
    thresh_frac: f32,
    lam: f32,
    nonneg_mask: u8,
) -> ([f32; 4], usize, f32) {
    let p = col_map.len();
    let mut y_ss = 0.0f32;
    let mut r0 = 0usize;
    while r0 < n_rows {
        y_ss += y[r0] * y[r0];
        r0 += 1;
    }
    let y_norm = sqrt_f32(y_ss);
    if y_norm < 1.0e-10 || p == 0 {
        return ([0.0; 4], 0, y_ss);
    }
    let mut active_cols = [0usize; 4];
    let mut active_local = [0usize; 4];
    let mut n_act = 0usize;
    let mut j = 0usize;
    while j < p && j < 4 {
        let gc = col_map[j];
        let mut cn2 = 0.0f32;
        let mut r = 0usize;
        while r < n_rows {
            let v = rows[r][gc];
            cn2 += v * v;
            r += 1;
        }
        if cn2 > 1.0e-12 {
            active_cols[n_act] = gc;
            active_local[n_act] = j;
            n_act += 1;
        }
        j += 1;
    }
    if n_act == 0 {
        return ([0.0; 4], 0, y_ss);
    }

    let mut iter = 0usize;
    while iter < 6 && n_act > 0 {
        let sol = fit_ridge_subset(rows, n_rows, y, &active_cols[..n_act], lam);
        let (xi_sub, xi_norm, _) = match sol {
            Some(v) => v,
            None => {
                n_act -= 1;
                iter += 1;
                continue;
            }
        };
        let mut next_n = 0usize;
        let mut changed = false;
        let mut idx = 0usize;
        while idx < n_act {
            let loc = active_local[idx];
            let contrib = abs_f32(xi_norm[idx]);
            let mut keep = contrib >= thresh_frac * y_norm;
            if (nonneg_mask & (1u8 << loc)) != 0 && xi_sub[idx] <= 0.0 {
                keep = false;
            }
            if keep {
                active_cols[next_n] = active_cols[idx];
                active_local[next_n] = loc;
                next_n += 1;
            } else {
                changed = true;
            }
            idx += 1;
        }
        n_act = next_n;
        if !changed {
            break;
        }
        iter += 1;
    }

    // Final polish with tiny lam = 1e-6 to eliminate ridge shrinkage bias
    let mut final_xi = [0.0f32; 4];
    let mut rss = y_ss;
    if n_act > 0 {
        if let Some((xi_p, _, rss_p)) = fit_ridge_subset(rows, n_rows, y, &active_cols[..n_act], 1.0e-6) {
            let mut bad = false;
            let mut idx = 0usize;
            while idx < n_act {
                let loc = active_local[idx];
                final_xi[loc] = xi_p[idx];
                if (nonneg_mask & (1u8 << loc)) != 0 && xi_p[idx] <= 0.0 {
                    bad = true;
                }
                idx += 1;
            }
            if !bad {
                rss = rss_p;
            } else {
                let mut next_n = 0usize;
                idx = 0;
                while idx < n_act {
                    let loc = active_local[idx];
                    if !((nonneg_mask & (1u8 << loc)) != 0 && final_xi[loc] <= 0.0) {
                        active_cols[next_n] = active_cols[idx];
                        active_local[next_n] = loc;
                        next_n += 1;
                    }
                    idx += 1;
                }
                n_act = next_n;
                final_xi = [0.0; 4];
                if n_act > 0 {
                    if let Some((xi2, _, rss2)) = fit_ridge_subset(rows, n_rows, y, &active_cols[..n_act], 1.0e-6) {
                        let mut k = 0usize;
                        while k < n_act {
                            final_xi[active_local[k]] = xi2[k];
                            k += 1;
                        }
                        rss = rss2;
                    }
                }
            }
        } else {
            n_act = 0;
        }
    }
    (final_xi, n_act, rss)
}

#[inline]
fn compute_aicc(rss_wf: f32, n_wf: usize, k_params: usize, sigma_wf_sq: f32) -> f32 {
    let nf = max_f32(n_wf as f32, 4.0);
    let kf = k_params as f32;
    let denom = max_f32(nf - kf - 1.0, 1.0);
    let mse_eff = sigma_wf_sq + rss_wf / nf;
    nf * fast_lnf(mse_eff) + 2.0 * kf + (2.0 * kf * (kf + 1.0)) / denom
}

// =============================================================================
// Position Smoothers, Trajectory Fits, Weak-Form Moment-Matched Quadrature
// =============================================================================

/// Exact centered OLS quadratic fit for slice `[start..end)` -> (cx[3], cy[3], t_mid, rss)
fn fit_quad_xy(t: &[f32], x: &[f32], y: &[f32], start: usize, end: usize) -> Option<([f32; 3], [f32; 3], f32, f32)> {
    let n = end.saturating_sub(start);
    if n < 5 {
        return None;
    }
    let mut sum_t = 0.0f32;
    let mut i = start;
    while i < end {
        sum_t += t[i];
        i += 1;
    }
    let tm = sum_t / (n as f32);
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut bx0 = 0.0f32;
    let mut bx1 = 0.0f32;
    let mut bx2 = 0.0f32;
    let mut by0 = 0.0f32;
    let mut by1 = 0.0f32;
    let mut by2 = 0.0f32;
    let tm64 = tm as f64;
    let x0 = x[start] as f64;
    let y0 = y[start] as f64;
    let mut su1 = 0.0f64;
    let mut su2 = 0.0f64;
    let mut su3 = 0.0f64;
    let mut su4 = 0.0f64;
    let mut sdx = 0.0f64;
    let mut sdx_u = 0.0f64;
    let mut sdx_u2 = 0.0f64;
    let mut sdy = 0.0f64;
    let mut sdy_u = 0.0f64;
    let mut sdy_u2 = 0.0f64;
    let mut sdxy2 = 0.0f64;
    i = start;
    while i < end {
        let xi = x[i];
        let yi = y[i];
        let u = t[i] - tm;
        let u2 = u * u;
        s1 += u;
        s2 += u2;
        s3 += u2 * u;
        s4 += u2 * u2;
        bx0 += xi;
        bx1 += u * xi;
        bx2 += u2 * xi;
        by0 += yi;
        by1 += u * yi;
        by2 += u2 * yi;

        let u64 = (t[i] as f64) - tm64;
        let u2_64 = u64 * u64;
        let dx = (xi as f64) - x0;
        let dy = (yi as f64) - y0;
        su1 += u64;
        su2 += u2_64;
        su3 += u2_64 * u64;
        su4 += u2_64 * u2_64;
        sdx += dx;
        sdx_u += dx * u64;
        sdx_u2 += dx * u2_64;
        sdy += dy;
        sdy_u += dy * u64;
        sdy_u2 += dy * u2_64;
        sdxy2 += dx * dx + dy * dy;
        i += 1;
    }
    let nf = n as f32;
    let mat = [
        [nf + 1.0e-8, s1, s2],
        [s1, s2 + 1.0e-8, s3],
        [s2, s3, s4 + 1.0e-8],
    ];
    let cx = solve_3x3(mat, [bx0, bx1, bx2])?;
    let cy = solve_3x3(mat, [by0, by1, by2])?;
    let nf64 = n as f64;
    let cx0 = (cx[0] as f64) - x0;
    let cx1 = cx[1] as f64;
    let cx2 = cx[2] as f64;
    let cy0 = (cy[0] as f64) - y0;
    let cy1 = cy[1] as f64;
    let cy2 = cy[2] as f64;
    let qx = cx0 * (nf64 * cx0 + 2.0 * (su1 * cx1 + su2 * cx2))
        + cx1 * (su2 * cx1 + 2.0 * su3 * cx2)
        + cx2 * (su4 * cx2)
        - 2.0 * (cx0 * sdx + cx1 * sdx_u + cx2 * sdx_u2);
    let qy = cy0 * (nf64 * cy0 + 2.0 * (su1 * cy1 + su2 * cy2))
        + cy1 * (su2 * cy1 + 2.0 * su3 * cy2)
        + cy2 * (su4 * cy2)
        - 2.0 * (cy0 * sdy + cy1 * sdy_u + cy2 * sdy_u2);
    let rss = max_f32((sdxy2 + qx + qy) as f32, 0.0);
    Some((cx, cy, tm, rss))
}

#[inline]
fn fit_linear_xy(t: &[f32], x: &[f32], y: &[f32], start: usize, end: usize) -> (f32, f32, f32, f32, f32, f32) {
    let n = end.saturating_sub(start);
    if n < 2 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let inv_n = 1.0 / (n as f32);
    let mut mt = 0.0f32;
    let mut mx = 0.0f32;
    let mut my = 0.0f32;
    let mut i = start;
    while i < end {
        mt += t[i];
        mx += x[i];
        my += y[i];
        i += 1;
    }
    mt *= inv_n;
    mx *= inv_n;
    my *= inv_n;
    let mut denom = 0.0f32;
    let mut num_x = 0.0f32;
    let mut num_y = 0.0f32;
    let mut tss = 0.0f32;
    i = start;
    while i < end {
        let dt = t[i] - mt;
        let dx = x[i] - mx;
        let dy = y[i] - my;
        denom += dt * dt;
        num_x += dt * dx;
        num_y += dt * dy;
        tss += dx * dx + dy * dy;
        i += 1;
    }
    let inv_d = if denom > 1.0e-12 { 1.0 / denom } else { 0.0 };
    let slx = num_x * inv_d;
    let sly = num_y * inv_d;
    let ix = mx - slx * mt;
    let iy = my - sly * mt;
    let mut rss = 0.0f32;
    i = start;
    while i < end {
        let ex = x[i] - (ix + slx * t[i]);
        let ey = y[i] - (iy + sly * t[i]);
        rss += ex * ex + ey * ey;
        i += 1;
    }
    (ix, slx, iy, sly, rss, tss)
}

/// 1D ordinary least squares `y(t) = inter + slope * t` on `[start, end)`, returning `(slope, rss, syy)`.
#[inline]
fn fit_linear_1d_slice(t: &[f32], y: &[f32], start: usize, end: usize) -> (f32, f32, f32) {
    let n = end.saturating_sub(start);
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let inv_n = 1.0 / (n as f32);
    let mut mt = 0.0f32;
    let mut my = 0.0f32;
    let mut i = start;
    while i < end {
        mt += t[i];
        my += y[i];
        i += 1;
    }
    mt *= inv_n;
    my *= inv_n;
    let mut denom = 0.0f32;
    let mut num = 0.0f32;
    let mut syy = 0.0f32;
    i = start;
    while i < end {
        let dt = t[i] - mt;
        let dy = y[i] - my;
        denom += dt * dt;
        num += dt * dy;
        syy += dy * dy;
        i += 1;
    }
    let sl = if denom > 1.0e-12 { num / denom } else { 0.0 };
    let mut rss = 0.0f32;
    i = start;
    while i < end {
        let e = (y[i] - my) - sl * (t[i] - mt);
        rss += e * e;
        i += 1;
    }
    (sl, rss, syy)
}

#[inline]
fn fit_linear_x_only(t: &[f32], x: &[f32], n: usize) -> f32 {
    fit_linear_1d_slice(t, x, 0, n).1
}

fn insertion_sort_f32(arr: &mut [f32], n: usize) {
    let mut i = 1usize;
    while i < n {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
        i += 1;
    }
}

/// Local 7-point (1-u^2)^2 weighted quadratic velocity smoother + trimmed 3rd-diff noise sigma_pos.
fn smooth_vel_and_sigma(sc: &mut PhysicsIdScratch, n: usize) -> f32 {
    let dt_med = (sc.t_arr[n - 1] - sc.t_arr[0]) / max_f32((n - 1) as f32, 1.0);
    let hw = max_f32(0.08, 2.8 * dt_med);
    let mut i = 0usize;
    while i < n {
        let lo = i.saturating_sub(3);
        let hi = if i + 4 < n { i + 4 } else { n };
        let tc = sc.t_arr[i];
        let local_hw = max_f32(hw, max_f32(tc - sc.t_arr[lo], sc.t_arr[hi - 1] - tc) * 1.15);
        let local_hw = max_f32(local_hw, 1.0e-3);
        let mut s = [0.0f32; 5];
        let mut bx = [0.0f32; 3];
        let mut by = [0.0f32; 3];
        let mut j = lo;
        while j < hi {
            let dt = sc.t_arr[j] - tc;
            let u = abs_f32(dt) / local_hw;
            if u < 1.0 {
                let om = 1.0 - u * u;
                let w = om * om;
                let d2 = dt * dt;
                s[0] += w;
                s[1] += w * dt;
                s[2] += w * d2;
                s[3] += w * d2 * dt;
                s[4] += w * d2 * d2;
                let xj = sc.xc[j];
                let yj = sc.yc[j];
                bx[0] += w * xj;
                bx[1] += w * dt * xj;
                bx[2] += w * d2 * xj;
                by[0] += w * yj;
                by[1] += w * dt * yj;
                by[2] += w * d2 * yj;
            }
            j += 1;
        }
        let mat = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
        let cx = solve_3x3(mat, bx);
        let cy = solve_3x3(mat, by);
        if let (Some(vx_c), Some(vy_c)) = (cx, cy) {
            sc.vx[i] = vx_c[1];
            sc.vy[i] = vy_c[1];
        } else {
            let j0 = i.saturating_sub(1);
            let j1 = if i + 1 < n { i + 1 } else { n - 1 };
            let dt_d = max_f32(sc.t_arr[j1] - sc.t_arr[j0], 1.0e-4);
            sc.vx[i] = (sc.xc[j1] - sc.xc[j0]) / dt_d;
            sc.vy[i] = (sc.yc[j1] - sc.yc[j0]) / dt_d;
        }
        i += 1;
    }

    if n < 6 {
        return 0.25;
    }
    let mut cnt = 0usize;
    let inv20 = 0.05f32;
    let mut k = 1usize;
    while k + 2 < n {
        if (sc.t_arr[k + 2] - sc.t_arr[k - 1]) < 4.5 * dt_med {
            let d3x = sc.xc[k + 2] - 3.0 * sc.xc[k + 1] + 3.0 * sc.xc[k] - sc.xc[k - 1];
            let d3y = sc.yc[k + 2] - 3.0 * sc.yc[k + 1] + 3.0 * sc.yc[k] - sc.yc[k - 1];
            sc.tmp_n[cnt] = (d3x * d3x + d3y * d3y) * inv20;
            cnt += 1;
        }
        k += 1;
    }
    if cnt == 0 {
        return 0.25;
    }
    insertion_sort_f32(&mut sc.tmp_n, cnt);
    let keep = {
        let q = (cnt * 3) / 4;
        if q < 3 { if cnt < 3 { cnt } else { 3 } } else { q }
    };
    let mut sum_sq = 0.0f32;
    let mut idx = 0usize;
    while idx < keep {
        sum_sq += sc.tmp_n[idx];
        idx += 1;
    }
    let sigma = sqrt_f32(sum_sq / (keep as f32));
    clamp_f32(sigma, 0.16, 3.0)
}

/// Gamma-weighted moment-matched weak weights phi(u)=(1-u^2)^2 over [lo_t, hi_t].
fn build_weak_weights(sc: &mut PhysicsIdScratch, n: usize, center_t: f32, half_h: f32) -> Option<(usize, usize, f32)> {
    let inv_h = 1.0 / max_f32(half_h, 1.0e-4);
    let inv_h2 = inv_h * inv_h;
    let mut i0 = n;
    let mut i1 = 0usize;
    let mut sp = 0.0f32;
    let mut i = 0usize;
    while i < n {
        let dt = sc.t_arr[i] - center_t;
        let u = dt * inv_h;
        if u > -1.0 && u < 1.0 {
            if i0 == n { i0 = i; }
            i1 = i + 1;
            let w = if i == 0 {
                sc.t_arr[1] - sc.t_arr[0]
            } else if i + 1 == n {
                sc.t_arr[n - 1] - sc.t_arr[n - 2]
            } else {
                0.5 * (sc.t_arr[i + 1] - sc.t_arr[i - 1])
            };
            let om = 1.0 - u * u;
            let phi = w * om * om;
            sc.w_gamma[i] = phi;
            sc.w_beta[i] = w * (-4.0 * u * om * inv_h);
            sc.w_alpha[i] = w * (4.0 * (3.0 * u * u - 1.0) * inv_h2);
            sp += phi;
        }
        i += 1;
    }
    if i1 <= i0 || (i1 - i0) < 6 || sp <= 1.0e-12 {
        return None;
    }
    let inv_sp = 1.0 / sp;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut g3 = 0.0f32;
    let mut g4 = 0.0f32;
    let mut sb = [0.0f32; 3];
    let mut sa = [0.0f32; 3];
    let mut k = i0;
    while k < i1 {
        let g = sc.w_gamma[k] * inv_sp;
        let b0 = -sc.w_beta[k] * inv_sp;
        let a0 = sc.w_alpha[k] * inv_sp;
        sc.w_gamma[k] = g;
        sc.w_beta[k] = b0;
        sc.w_alpha[k] = a0;
        let dt = sc.t_arr[k] - center_t;
        let dt2 = dt * dt;
        g1 += g * dt;
        g2 += g * dt2;
        g3 += g * dt2 * dt;
        g4 += g * dt2 * dt2;
        sb[0] += b0;
        sb[1] += b0 * dt;
        sb[2] += b0 * 0.5 * dt2;
        sa[0] += a0;
        sa[1] += a0 * dt;
        sa[2] += a0 * 0.5 * dt2;
        k += 1;
    }
    sb[1] -= 1.0;
    sa[2] -= 1.0;
    let mat = [
        [1.0f32, g1, 0.5 * g2],
        [g1, g2, 0.5 * g3],
        [0.5 * g2, 0.5 * g3, 0.25 * g4],
    ];
    let d_beta = solve_3x3(mat, sb)?;
    let d_alpha = solve_3x3(mat, sa)?;
    let mut amp_sq = 0.0f32;
    k = i0;
    while k < i1 {
        let dt = sc.t_arr[k] - center_t;
        let p2 = 0.5 * dt * dt;
        let g = sc.w_gamma[k];
        sc.w_beta[k] -= (d_beta[0] + d_beta[1] * dt + d_beta[2] * p2) * g;
        let ak = sc.w_alpha[k] - (d_alpha[0] + d_alpha[1] * dt + d_alpha[2] * p2) * g;
        sc.w_alpha[k] = ak;
        amp_sq += ak * ak;
        k += 1;
    }
    let amp = sqrt_f32(amp_sq);
    if amp < 1.0e-6 { None } else { Some((i0, i1, amp)) }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn fast_sin4_wasm(vx: v128) -> v128 {
    use core::f32::consts::{FRAC_PI_2, PI, TAU};
    let v_inv_2pi = f32x4_splat(1.0 / TAU);
    let v_2pi = f32x4_splat(TAU);
    let v_pi = f32x4_splat(PI);
    let v_neg_pi = f32x4_splat(-PI);
    let v_hpi = f32x4_splat(FRAC_PI_2);
    let v_nhpi = f32x4_splat(-FRAC_PI_2);
    let vn = f32x4_nearest(f32x4_mul(vx, v_inv_2pi));
    let va0 = f32x4_sub(vx, f32x4_mul(vn, v_2pi));
    let gt_hpi = f32x4_gt(va0, v_hpi);
    let lt_nhpi = f32x4_lt(va0, v_nhpi);
    let va1 = v128_bitselect(f32x4_sub(v_pi, va0), va0, gt_hpi);
    let a = v128_bitselect(f32x4_sub(v_neg_pi, va0), va1, lt_nhpi);
    let a2 = f32x4_mul(a, a);
    let poly = f32x4_add(
        f32x4_splat(1.0),
        f32x4_mul(
            a2,
            f32x4_add(
                f32x4_splat(-0.16666657),
                f32x4_mul(
                    a2,
                    f32x4_add(f32x4_splat(0.008333025), f32x4_mul(a2, f32x4_splat(-0.000198074))),
                ),
            ),
        ),
    );
    f32x4_mul(a, poly)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn hsum_f32x4_f64(v: v128) -> f64 {
    (f32x4_extract_lane::<0>(v) as f64)
        + (f32x4_extract_lane::<1>(v) as f64)
        + (f32x4_extract_lane::<2>(v) as f64)
        + (f32x4_extract_lane::<3>(v) as f64)
}

/// Evaluate 1D sinusoid u1(t) = c0 + c1*cos(w*t) + c2*sin(w*t) at candidate omega + local refinement.
/// Returns (r2, rms, amp).
#[inline(never)]
fn eval_sinusoid_single(
    t_arr: &[f32],
    u: &[f32],
    n: usize,
    omega: f32,
    u_sum_f32: f32,
    sum_u2: f64,
    tss: f32,
) -> (f32, f32, f32) {
    let lim = n.min(t_arr.len()).min(u.len());
    if omega <= 0.1 || omega > 60.0 || lim < 5 {
        return (0.0, 999.0, 0.0);
    }
    let mut i = 0usize;
    #[cfg(target_arch = "wasm32")]
    let (mut sc_sum, mut ss_sum, mut scc_sum, mut scs_sum, mut sss_sum, mut b1_sum, mut b2_sum) = {
        let v_omega = f32x4_splat(omega);
        let v_hpi = f32x4_splat(core::f32::consts::FRAC_PI_2);
        let mut v_sc = f32x4_splat(0.0);
        let mut v_ss = f32x4_splat(0.0);
        let mut v_scc = f32x4_splat(0.0);
        let mut v_scs = f32x4_splat(0.0);
        let mut v_sss = f32x4_splat(0.0);
        let mut v_b1 = f32x4_splat(0.0);
        let mut v_b2 = f32x4_splat(0.0);
        while i + 4 <= lim {
            let vt = f32x4(t_arr[i], t_arr[i + 1], t_arr[i + 2], t_arr[i + 3]);
            let vu = f32x4(u[i], u[i + 1], u[i + 2], u[i + 3]);
            let vx = f32x4_mul(v_omega, vt);
            let vs = fast_sin4_wasm(vx);
            let vc = fast_sin4_wasm(f32x4_add(vx, v_hpi));
            v_sc = f32x4_add(v_sc, vc);
            v_ss = f32x4_add(v_ss, vs);
            v_scc = f32x4_add(v_scc, f32x4_mul(vc, vc));
            v_scs = f32x4_add(v_scs, f32x4_mul(vc, vs));
            v_sss = f32x4_add(v_sss, f32x4_mul(vs, vs));
            v_b1 = f32x4_add(v_b1, f32x4_mul(vc, vu));
            v_b2 = f32x4_add(v_b2, f32x4_mul(vs, vu));
            i += 4;
        }
        (
            hsum_f32x4_f64(v_sc),
            hsum_f32x4_f64(v_ss),
            hsum_f32x4_f64(v_scc),
            hsum_f32x4_f64(v_scs),
            hsum_f32x4_f64(v_sss),
            hsum_f32x4_f64(v_b1),
            hsum_f32x4_f64(v_b2),
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (mut sc_sum, mut ss_sum, mut scc_sum, mut scs_sum, mut sss_sum, mut b1_sum, mut b2_sum) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);

    while i < lim {
        let (s_p, c_p) = fast_sin_cos(omega * t_arr[i]);
        let sf = s_p as f64;
        let cf = c_p as f64;
        let uf = u[i] as f64;
        sc_sum += cf;
        ss_sum += sf;
        scc_sum += cf * cf;
        scs_sum += cf * sf;
        sss_sum += sf * sf;
        b1_sum += cf * uf;
        b2_sum += sf * uf;
        i += 1;
    }

    let nf_32 = lim as f32;
    let sc = sc_sum as f32;
    let ss = ss_sum as f32;
    let scc = scc_sum as f32;
    let scs = scs_sum as f32;
    let sss = sss_sum as f32;
    let a3 = [
        [nf_32 + 1.0e-7, sc, ss],
        [sc, scc + 1.0e-7, scs],
        [ss, scs, sss + 1.0e-7],
    ];
    let b3 = [u_sum_f32, b1_sum as f32, b2_sum as f32];
    let coef = match solve_3x3(a3, b3) {
        Some(v) => v,
        None => return (0.0, 999.0, 0.0),
    };
    let c0 = coef[0] as f64;
    let c1 = coef[1] as f64;
    let c2 = coef[2] as f64;
    let b0_64 = u_sum_f32 as f64;
    let nf_64 = lim as f64;
    let quad = c0 * (nf_64 * c0 + 2.0 * (sc_sum * c1 + ss_sum * c2))
        + c1 * (scc_sum * c1 + 2.0 * scs_sum * c2)
        + c2 * (sss_sum * c2);
    let lin = 2.0 * (c0 * b0_64 + c1 * b1_sum + c2 * b2_sum);
    let rss = max_f32((sum_u2 - lin + quad) as f32, 0.0);
    let r2 = max_f32(0.0, 1.0 - rss / tss);
    let rms = sqrt_f32(rss / (lim as f32));
    let amp = hypot_f32(coef[1], coef[2]);
    (r2, rms, amp)
}

#[inline(never)]
fn eval_sinusoid_pos(t_arr: &[f32], u: &[f32], n: usize, omega_seed: f32) -> (f32, f32, f32, f32) {
    let lim = n.min(t_arr.len()).min(u.len());
    if omega_seed <= 0.1 || omega_seed > 60.0 || lim < 5 {
        return (0.0, 999.0, 0.0, omega_seed);
    }
    let mut u_sum = 0.0f64;
    let mut sum_u2 = 0.0f64;
    let mut i = 0usize;
    while i < lim {
        let ui = u[i] as f64;
        u_sum += ui;
        sum_u2 += ui * ui;
        i += 1;
    }
    let tss = max_f32((sum_u2 - (u_sum * u_sum) / (lim as f64)) as f32, 1.0e-6);
    let u_sum_f32 = u_sum as f32;

    const MULTS: [f32; 7] = [0.72, 0.82, 0.91, 1.0, 1.09, 1.19, 1.32];
    let mut best_r2 = 0.0f32;
    let mut best_rms = 999.0f32;
    let mut best_amp = 0.0f32;
    let mut best_w = omega_seed;
    let mut m = 0usize;
    while m < 7 {
        let wt = omega_seed * MULTS[m];
        let (r2, rms, amp) = eval_sinusoid_single(t_arr, u, lim, wt, u_sum_f32, sum_u2, tss);
        if rms < best_rms {
            best_r2 = r2;
            best_rms = rms;
            best_amp = amp;
            best_w = wt;
        }
        m += 1;
    }
    let dw = 0.04 * best_w;
    let tries = [best_w - dw, best_w + dw, best_w - 0.5 * dw, best_w + 0.5 * dw];
    let mut k = 0usize;
    while k < 4 {
        let wt = tries[k];
        let (r2, rms, amp) = eval_sinusoid_single(t_arr, u, lim, wt, u_sum_f32, sum_u2, tss);
        if rms < best_rms {
            best_r2 = r2;
            best_rms = rms;
            best_amp = amp;
            best_w = wt;
        }
        k += 1;
    }
    (best_r2, best_rms, best_amp, best_w)
}

#[derive(Clone, Copy)]
struct LinearAccum {
    cnt: f64,
    st: f64,
    st2: f64,
    sx: f64,
    sy: f64,
    stx: f64,
    sty: f64,
    sxy2: f64,
}

impl LinearAccum {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            cnt: 0.0,
            st: 0.0,
            st2: 0.0,
            sx: 0.0,
            sy: 0.0,
            stx: 0.0,
            sty: 0.0,
            sxy2: 0.0,
        }
    }

    #[inline(always)]
    fn add_pt(&mut self, t: f64, x: f64, y: f64) {
        self.cnt += 1.0;
        self.st += t;
        self.st2 += t * t;
        self.sx += x;
        self.sy += y;
        self.stx += t * x;
        self.sty += t * y;
        self.sxy2 += x * x + y * y;
    }

    #[inline(always)]
    fn sub_pt(&mut self, t: f64, x: f64, y: f64) {
        self.cnt -= 1.0;
        self.st -= t;
        self.st2 -= t * t;
        self.sx -= x;
        self.sy -= y;
        self.stx -= t * x;
        self.sty -= t * y;
        self.sxy2 -= x * x + y * y;
    }

    #[inline(always)]
    fn solve(&self) -> (f32, f32, f32, f32, f32) {
        let inv_n = 1.0 / self.cnt;
        let mt = self.st * inv_n;
        let mx = self.sx * inv_n;
        let my = self.sy * inv_n;
        let denom = self.st2 - self.st * mt;
        let num_x = self.stx - self.st * mx;
        let num_y = self.sty - self.st * my;
        let tss = (self.sxy2 - (self.sx * mx + self.sy * my)).max(0.0);
        if denom > 1.0e-12 {
            let inv_d = 1.0 / denom;
            let slx = num_x * inv_d;
            let sly = num_y * inv_d;
            let ix = mx - slx * mt;
            let iy = my - sly * mt;
            let rss = (tss - (slx * num_x + sly * num_y)).max(0.0);
            (ix as f32, slx as f32, iy as f32, sly as f32, rss as f32)
        } else {
            (mx as f32, 0.0, my as f32, 0.0, tss as f32)
        }
    }
}

#[inline(never)]
fn find_best_impulsive_kink(
    t: &[f32],
    x: &[f32],
    y: &[f32],
    n: usize,
    min_side: usize,
) -> Option<(usize, f32, f32, f32, f32, f32, f32, f32)> {
    let lim = n.min(t.len()).min(x.len()).min(y.len());
    if lim <= 2 * min_side {
        return None;
    }
    let mut left = LinearAccum::zero();
    let mut right = LinearAccum::zero();
    let mut i = 0usize;
    while i < min_side {
        left.add_pt(t[i] as f64, x[i] as f64, y[i] as f64);
        i += 1;
    }
    i = min_side + 1;
    while i < lim {
        right.add_pt(t[i] as f64, x[i] as f64, y[i] as f64);
        i += 1;
    }
    let mut best_obj = 1.0e18f32;
    let mut best: Option<(usize, f32, f32, f32, f32, f32, f32, f32)> = None;
    let end_k = lim - min_side;
    let mut k = min_side;
    while k < end_k {
        if k >= 4 && (lim - (k + 1)) >= 4 {
            let (ix1, vx1, iy1, vy1, rss1) = left.solve();
            let (ix2, vx2, iy2, vy2, rss2) = right.solve();
            let rss_split = rss1 + rss2;
            let tk = t[k];
            let x1k = ix1 + vx1 * tk;
            let y1k = iy1 + vy1 * tk;
            let x2k = ix2 + vx2 * tk;
            let y2k = iy2 + vy2 * tk;
            let pos_gap = hypot_f32(x2k - x1k, y2k - y1k);
            let obj = rss_split + 2.0 * pos_gap * pos_gap;
            if obj < best_obj {
                best_obj = obj;
                best = Some((
                    k,
                    rss_split,
                    hypot_f32(vx2 - vx1, vy2 - vy1),
                    pos_gap,
                    vx1,
                    vy1,
                    vx2,
                    vy2,
                ));
            }
        }
        if k + 1 < end_k {
            left.add_pt(t[k] as f64, x[k] as f64, y[k] as f64);
            right.sub_pt(t[k + 1] as f64, x[k + 1] as f64, y[k + 1] as f64);
        }
        k += 1;
    }
    best
}

/// Forward RK2 Verlet rollout of a = (0, g_val) - beta_val * |v| * v with linear drift subtraction.
fn verlet_drag_pos_rms(sc: &mut PhysicsIdScratch, n: usize, g_val: f32, beta_val: f32) -> f32 {
    let m0 = if n / 3 < 6 { if n < 6 { n } else { 6 } } else if n / 3 > 14 { 14 } else { n / 3 };
    let (mut x_c, mut y_c, mut vx_c, mut vy_c) = match fit_quad_xy(&sc.t_arr, &sc.xc, &sc.yc, 0, m0) {
        Some((cx0, cy0, tq, _)) => (
            cx0[0] + cx0[1] * (-tq) + cx0[2] * (tq * tq),
            cy0[0] + cy0[1] * (-tq) + cy0[2] * (tq * tq),
            cx0[1] - 2.0 * cx0[2] * tq,
            cy0[1] - 2.0 * cy0[2] * tq,
        ),
        None => (sc.xc[0], sc.yc[0], sc.vx[0], sc.vy[0]),
    };
    sc.sim_x[0] = sc.xc[0] - x_c;
    sc.sim_y[0] = sc.yc[0] - y_c;
    let mut i = 1usize;
    while i < n {
        let dt_step = max_f32(sc.t_arr[i] - sc.t_arr[i - 1], 1.0e-4);
        let h = 0.5 * dt_step;
        let mut sub = 0usize;
        while sub < 2 {
            let sp1 = hypot_f32(vx_c, vy_c);
            let ax1 = -beta_val * sp1 * vx_c;
            let ay1 = g_val - beta_val * sp1 * vy_c;
            let vx_m = vx_c + 0.5 * h * ax1;
            let vy_m = vy_c + 0.5 * h * ay1;
            let sp2 = hypot_f32(vx_m, vy_m);
            let ax2 = -beta_val * sp2 * vx_m;
            let ay2 = g_val - beta_val * sp2 * vy_m;
            x_c += h * vx_m;
            y_c += h * vy_m;
            vx_c += h * ax2;
            vy_c += h * ay2;
            sub += 1;
        }
        sc.sim_x[i] = sc.xc[i] - x_c;
        sc.sim_y[i] = sc.yc[i] - y_c;
        i += 1;
    }
    let (_, _, _, _, rss_drift, _) = fit_linear_xy(&sc.t_arr, &sc.sim_x, &sc.sim_y, 0, n);
    sqrt_f32(rss_drift / (n as f32))
}

// =============================================================================
// Public Core API for Wave-2 src/physics_id.rs Integration
// =============================================================================

fn discover_from_normalized_vis(nv: usize, s: f32, prior_flight: u8, sc: &mut PhysicsIdScratch) -> OdeResult {
    if nv < 15 {
        return OdeResult::zero(0.0);
    }

    // 1. Teleport segmentation (keeps occlusions and high-G oscillations intact while resetting genuine ID swaps)
    let mut start_k = 0usize;
    let mut i = 1usize;
    while i < nv {
        let dt = max_f32(sc.ts_v[i] - sc.ts_v[i - 1], 1.0e-4);
        let dist = hypot_f32(sc.xs_v[i] - sc.xs_v[i - 1], sc.ys_v[i] - sc.ys_v[i - 1]);
        if dt > 0.65 {
            start_k = i;
        } else if i >= 2 && (i - start_k) >= 2 {
            let dt_prev = max_f32(sc.ts_v[i - 1] - sc.ts_v[i - 2], 1.0e-4);
            let vx_p = (sc.xs_v[i - 1] - sc.xs_v[i - 2]) / dt_prev;
            let vy_p = (sc.ys_v[i - 1] - sc.ys_v[i - 2]) / dt_prev;
            let pred_err = hypot_f32(sc.xs_v[i] - (sc.xs_v[i - 1] + vx_p * dt), sc.ys_v[i] - (sc.ys_v[i - 1] + vy_p * dt));
            if dt <= 0.10 && pred_err > 52.0 && dist > 38.0 {
                start_k = i;
            } else if dt > 0.10 && dist > 75.0 && pred_err > 65.0 {
                start_k = i;
            }
        } else if dist > 75.0 {
            start_k = i;
        }
        i += 1;
    }

    // 2. Single-frame outlier spike cleanup
    let seg_len = nv - start_k;
    if seg_len < 15 {
        return OdeResult::zero(0.0);
    }
    let mut nc = 0usize;
    sc.clean_idx[0] = start_k as u8;
    nc += 1;
    let mut idx = start_k + 1;
    while idx + 1 < nv {
        let prev_i = sc.clean_idx[nc - 1] as usize;
        let t0 = sc.ts_v[prev_i];
        let t1 = sc.ts_v[idx];
        let t2 = sc.ts_v[idx + 1];
        let frac = (t1 - t0) / max_f32(t2 - t0, 1.0e-4);
        let px_int = sc.xs_v[prev_i] + frac * (sc.xs_v[idx + 1] - sc.xs_v[prev_i]);
        let py_int = sc.ys_v[prev_i] + frac * (sc.ys_v[idx + 1] - sc.ys_v[prev_i]);
        if hypot_f32(sc.xs_v[idx] - px_int, sc.ys_v[idx] - py_int) <= 16.0 {
            sc.clean_idx[nc] = idx as u8;
            nc += 1;
        }
        idx += 1;
    }
    sc.clean_idx[nc] = (nv - 1) as u8;
    nc += 1;

    let n = nc;
    if n < 15 {
        return OdeResult::zero(0.0);
    }
    let t_base = sc.ts_v[sc.clean_idx[0] as usize];
    let mut xm = 0.0f32;
    let mut ym = 0.0f32;
    i = 0;
    while i < n {
        let k = sc.clean_idx[i] as usize;
        sc.t_arr[i] = sc.ts_v[k] - t_base;
        let xv = sc.xs_v[k];
        let yv = sc.ys_v[k];
        sc.xc[i] = xv;
        sc.yc[i] = yv;
        xm += xv;
        ym += yv;
        i += 1;
    }
    let inv_n = 1.0 / (n as f32);
    xm *= inv_n;
    ym *= inv_n;
    i = 0;
    while i < n {
        sc.xc[i] -= xm;
        sc.yc[i] -= ym;
        i += 1;
    }
    let t_span = sc.t_arr[n - 1];
    if t_span < 0.25 {
        return OdeResult::zero(0.0);
    }

    // 3. Smooth velocities + position noise sigma_pos
    let sigma_pos = smooth_vel_and_sigma(sc, n);
    let (_, _, _, _, rss_lin_pos, tss_pos) = fit_linear_xy(&sc.t_arr, &sc.xc, &sc.yc, 0, n);
    let rms_lin_pos = sqrt_f32(rss_lin_pos * inv_n);
    let (cx_q, cy_q, tm_q, rss_quad_pos) = match fit_quad_xy(&sc.t_arr, &sc.xc, &sc.yc, 0, n) {
        Some(v) => v,
        None => return OdeResult::zero(sigma_pos * s),
    };
    let rms_quad_pos = sqrt_f32(rss_quad_pos * inv_n);
    let ax_q = 2.0 * cx_q[2];
    let ay_q = 2.0 * cy_q[2];
    let vx_q = cx_q[1];
    let vy_q = cy_q[1];
    let spd_q = hypot_f32(vx_q, vy_q);

    let disp_net = hypot_f32(sc.xc[n - 1] - sc.xc[0], sc.yc[n - 1] - sc.yc[0]);
    let mut path_len = 0.0f32;
    i = 1;
    while i < n {
        path_len += hypot_f32(sc.xc[i] - sc.xc[i - 1], sc.yc[i] - sc.yc[i - 1]);
        i += 1;
    }
    let _ = path_len;
    let r2_lin_pos = clamp_f32(1.0 - rss_lin_pos / max_f32(tss_pos, 1.0e-6), 0.0, 1.0);
    let r2_quad_pos = clamp_f32(1.0 - rss_quad_pos / max_f32(tss_pos, 1.0e-6), 0.0, 1.0);

    // 4. Piecewise-CV kink search (Regime 7)
    let mut min_side = n / 4;
    if min_side < 5 { min_side = 5; }
    if min_side > 8 { min_side = 8; }
    let mut imp_k = 0usize;
    let mut imp_rss = 0.0f32;
    let mut imp_dv = 0.0f32;
    let mut imp_gap = 0.0f32;
    let mut imp_vx1 = 0.0f32;
    let mut imp_vy1 = 0.0f32;
    let mut imp_vx2 = 0.0f32;
    let mut imp_vy2 = 0.0f32;
    let mut has_imp = false;
    if n > 2 * min_side && rms_quad_pos > max_f32(2.5, 3.0 * sigma_pos) {
        if let Some((bk, brss, bdv, bgap, bvx1, bvy1, bvx2, bvy2)) =
            find_best_impulsive_kink(&sc.t_arr, &sc.xc, &sc.yc, n, min_side)
        {
            imp_k = bk;
            imp_rss = brss;
            imp_dv = bdv;
            imp_gap = bgap;
            imp_vx1 = bvx1;
            imp_vy1 = bvy1;
            imp_vx2 = bvx2;
            imp_vy2 = bvy2;
            has_imp = true;
        }
    }

    // 5. PCA Principal Axis & Weak-Form Macro Bank (H_wide)
    let mut cxx = 0.0f32;
    let mut cyy = 0.0f32;
    let mut cxy = 0.0f32;
    i = 0;
    while i < n {
        cxx += sc.xc[i] * sc.xc[i];
        cyy += sc.yc[i] * sc.yc[i];
        cxy += sc.xc[i] * sc.yc[i];
        i += 1;
    }
    cxx *= inv_n;
    cyy *= inv_n;
    cxy *= inv_n;
    let tr = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = sqrt_f32(max_f32(0.0, 0.25 * tr * tr - det));
    let lam1 = 0.5 * tr + disc;
    let (e1x, e1y) = if abs_f32(cxy) > 1.0e-8 || abs_f32(cxx - lam1) > 1.0e-8 {
        let ang = 0.5 * fast_atan2f(2.0 * cxy, cxx - cyy);
        let (sa, ca) = fast_sin_cos(ang);
        (ca, sa)
    } else {
        (1.0f32, 0.0f32)
    };
    let mut u1_min = 1.0e18f32;
    let mut u1_max = -1.0e18f32;
    let mut u1_abs_max = 0.0f32;
    let mut vu1_abs_max = 0.0f32;
    i = 0;
    while i < n {
        let u = sc.xc[i] * e1x + sc.yc[i] * e1y;
        let vu = sc.vx[i] * e1x + sc.vy[i] * e1y;
        sc.u1[i] = u;
        if u < u1_min { u1_min = u; }
        if u > u1_max { u1_max = u; }
        if abs_f32(u) > u1_abs_max { u1_abs_max = abs_f32(u); }
        if abs_f32(vu) > vu1_abs_max { vu1_abs_max = abs_f32(vu); }
        i += 1;
    }
    let app_u1 = u1_max - u1_min;

    let dt_med = t_span / ((n - 1) as f32);
    let h_wide = max_f32(0.32 * t_span, 0.22);
    let h_mid = max_f32(0.18 * t_span, 0.14);
    let h_fast = max_f32(min_f32(0.14 * t_span, 0.125), 3.6 * dt_med);

    // First pass: accumulate macro weak rows directly into sc buffers
    let mut m_rows = 0usize;
    let mut m_idx = 0usize;
    while m_idx < MAX_WEAK_ROWS {
        let c_m = t_span * (0.14 + 0.72 * ((m_idx as f32) / 11.0));
        if let Some((i0, i1, amp_a)) = build_weak_weights(sc, n, c_m, h_wide) {
            let inv_s = 1.0 / amp_a;
            let mut ax_w = 0.0f32;
            let mut ay_w = 0.0f32;
            let mut vx_w = 0.0f32;
            let mut vy_w = 0.0f32;
            let mut x_w = 0.0f32;
            let mut y_w = 0.0f32;
            let mut dx_w = 0.0f32;
            let mut dy_w = 0.0f32;
            let mut vhx_w = 0.0f32;
            let mut vhy_w = 0.0f32;
            let mut u1_a = 0.0f32;
            let mut u1_g = 0.0f32;
            let mut u1_b = 0.0f32;
            let mut u1sq_g = 0.0f32;
            let mut k = i0;
            while k < i1 {
                let wa = sc.w_alpha[k];
                let wb = sc.w_beta[k];
                let wg = sc.w_gamma[k];
                let xk = sc.xc[k];
                let yk = sc.yc[k];
                let vxk = sc.vx[k];
                let vyk = sc.vy[k];
                let uk = sc.u1[k];
                ax_w += wa * xk;
                ay_w += wa * yk;
                vx_w += wb * xk;
                vy_w += wb * yk;
                x_w += wg * xk;
                y_w += wg * yk;
                let sp = hypot_f32(vxk, vyk);
                dx_w -= wg * sp * vxk;
                dy_w -= wg * sp * vyk;
                if sp > 1.0e-2 {
                    let isp = 1.0 / sp;
                    vhx_w += wg * (vxk * isp);
                    vhy_w += wg * (vyk * isp);
                }
                u1_a += wa * uk;
                u1_g += wg * uk;
                u1_b += wb * uk;
                u1sq_g += wg * (uk * uk);
                k += 1;
            }
            let r_x = m_rows;
            // We'll place y-rows at m_rows_final + r later; store x-rows at 0..m_rows and y-rows temporarily at 12..12+m_rows, then pack cleanly!
            sc.y_vec[r_x] = ax_w * inv_s;
            sc.y_vec[MAX_WEAK_ROWS + r_x] = ay_w * inv_s;
            sc.x_trans[r_x] = [inv_s, 0.0, dx_w * inv_s, -(vx_w * inv_s)];
            sc.x_trans[MAX_WEAK_ROWS + r_x] = [0.0, inv_s, dy_w * inv_s, -(vy_w * inv_s)];
            sc.x_ctrv[r_x] = [-(vy_w * inv_s), vhx_w * inv_s];
            sc.x_ctrv[MAX_WEAK_ROWS + r_x] = [vx_w * inv_s, vhy_w * inv_s];
            let _ = (x_w, y_w);
            sc.y_u1[r_x] = u1_a * inv_s;
            sc.x_harm1d[r_x] = [inv_s, -(u1_g * inv_s), -(u1_b * inv_s)];
            sc.y_u2[r_x] = (-e1y) * sc.y_vec[r_x] + e1x * sc.y_vec[MAX_WEAK_ROWS + r_x];
            sc.x_u2[r_x] = [inv_s, u1sq_g * inv_s];
            m_rows += 1;
        }
        m_idx += 1;
    }
    if m_rows < 6 {
        let mut r = OdeResult::zero(sigma_pos * s);
        r.phys_regime_conf = r2_lin_pos;
        r.ode_r2 = r2_lin_pos;
        r.eff_samples = n as u8;
        return r;
    }
    // Compact y-rows from MAX_WEAK_ROWS.. into m_rows..2*m_rows if m_rows < MAX_WEAK_ROWS
    if m_rows < MAX_WEAK_ROWS {
        let mut r = 0usize;
        while r < m_rows {
            sc.y_vec[m_rows + r] = sc.y_vec[MAX_WEAK_ROWS + r];
            sc.x_trans[m_rows + r] = sc.x_trans[MAX_WEAK_ROWS + r];
            sc.x_ctrv[m_rows + r] = sc.x_ctrv[MAX_WEAK_ROWS + r];
            r += 1;
        }
    }
    let n_wf = 2 * m_rows;
    let sigma_wf_sq = max_f32(sigma_pos * sigma_pos, 0.08);
    let mut rss_wf_cv = 0.0f32;
    i = 0;
    while i < n_wf {
        rss_wf_cv += sc.y_vec[i] * sc.y_vec[i];
        i += 1;
    }
    let aicc_cv = compute_aicc(rss_wf_cv, n_wf, 0, sigma_wf_sq);

    // 6. Solve compact STRidge candidates via unified column-normalized stridge_cols
    let (xi_bd, k_bd, rss_bd) = stridge_cols(&sc.x_trans, n_wf, &sc.y_vec, &[1, 2], 0.10, 1.0e-3, 0b11);
    let (xi_b, k_b, rss_b) = stridge_cols(&sc.x_trans, n_wf, &sc.y_vec, &[1], 0.05, 1.0e-3, 0b1);
    let (xi_bst, k_bst, rss_bst) = stridge_cols(&sc.x_trans, n_wf, &sc.y_vec, &[0, 1, 2], 0.12, 1.0e-3, 0b100);
    let (xi_ctrv, k_ctrv, rss_ctrv) = stridge_cols(&sc.x_ctrv, n_wf, &sc.y_vec, &[0, 1], 0.12, 1.0e-3, 0);
    let (xi_h1, k_h1, rss_h1_u1) = stridge_cols(&sc.x_harm1d, m_rows, &sc.y_u1, &[0, 1, 2], 0.10, 1.0e-3, 0b010);
    let (_, _, rss_h1_u2) = stridge_cols(&sc.x_u2, m_rows, &sc.y_u2, &[0, 1], 0.0, 1.0e-3, 0);
    let rss_pend_2d = rss_h1_u1 + rss_h1_u2;
    let k_pend = k_h1 + 2;

    // 7. Physical Trajectory Features & Zero Crossings
    let mut mean_vx = 0.0f32;
    let mut mean_vy = 0.0f32;
    i = 0;
    while i < n {
        mean_vx += sc.vx[i];
        mean_vy += sc.vy[i];
        i += 1;
    }
    mean_vx *= inv_n;
    mean_vy *= inv_n;
    let mean_spd = hypot_f32(mean_vx, mean_vy);
    let mut min_dot = 1.0e18f32;
    let mut zc_u1 = 0u32;
    let mut last_sv = 0i32;
    let mut zc_pos_u1 = 0u32;
    let mut last_sp = 0i32;
    let mut vmin = 1.0e18f32;
    let mut vmax = 0.0f32;
    i = 0;
    while i < n {
        let dot = sc.vx[i] * mean_vx + sc.vy[i] * mean_vy;
        if dot < min_dot { min_dot = dot; }
        let sp = hypot_f32(sc.vx[i], sc.vy[i]);
        if sp < vmin { vmin = sp; }
        if sp > vmax { vmax = sp; }
        let vu = sc.vx[i] * e1x + sc.vy[i] * e1y;
        if abs_f32(vu) > 0.18 * vu1_abs_max {
            let sv = if vu > 0.0 { 1 } else { -1 };
            if last_sv != 0 && sv != last_sv {
                zc_u1 += 1;
            }
            last_sv = sv;
        }
        if abs_f32(sc.u1[i]) > 0.20 * u1_abs_max {
            let sp_u = if sc.u1[i] > 0.0 { 1 } else { -1 };
            if last_sp != 0 && sp_u != last_sp {
                zc_pos_u1 += 1;
            }
            last_sp = sp_u;
        }
        i += 1;
    }
    let _ = (vmin, vmax);
    let unidir_ratio = min_dot / max_f32(mean_spd * mean_spd, 1.0e-3);
    let i_h0 = if n - 1 < 3 { n - 1 } else { 3 };
    let i_h1 = if n >= 4 { n - 4 } else { 0 };
    let ang0 = fast_atan2f(sc.vy[i_h0], sc.vx[i_h0]);
    let ang1 = fast_atan2f(sc.vy[i_h1], sc.vx[i_h1]);
    let (sdh, cdh) = fast_sin_cos(ang1 - ang0);
    let dheading = fast_atan2f(sdh, cdh);

    let mut tss_u1_wf = 0.0f32;
    i = 0;
    while i < m_rows {
        tss_u1_wf += sc.y_u1[i] * sc.y_u1[i];
        i += 1;
    }

    // 8. Candidate Table (regime, aicc, beta_sd, thrust_sd, r2_wf, active_k, g_sd, omega)
    let mut c_reg = [0u8; 9];
    let mut c_aicc = [1.0e18f32; 9];
    let mut c_beta = [0.0f32; 9];
    let mut c_thr = [0.0f32; 9];
    let mut c_r2 = [0.0f32; 9];
    let mut c_k = [0u8; 9];
    let mut c_g = [0.0f32; 9];
    let mut c_w = [0.0f32; 9];
    let mut n_cand = 0usize;

    // Candidate 0: CV
    c_reg[n_cand] = PHYS_REGIME_LINEAR_CV;
    c_aicc[n_cand] = aicc_cv;
    c_r2[n_cand] = r2_lin_pos;
    n_cand += 1;

    let sagitta_rms = rms_lin_pos;
    let has_real_accel = (sagitta_rms > max_f32(1.15 * sigma_pos, 0.45)) && ((rss_wf_cv / (n_wf as f32)) > 1.35 * sigma_wf_sq);
    let quad_tol = min_f32(max_f32(1.1, 3.8 * sigma_pos), 2.3);
    let mut smooth_quad_ok = (rms_quad_pos <= quad_tol) && (r2_quad_pos >= 0.82);
    let mut max_quad_dev = 0.0f32;
    i = 0;
    while i < n {
        let u = sc.t_arr[i] - tm_q;
        let qx = cx_q[0] + cx_q[1] * u + cx_q[2] * (u * u);
        let qy = cy_q[0] + cy_q[1] * u + cy_q[2] * (u * u);
        let d = hypot_f32(sc.xc[i] - qx, sc.yc[i] - qy);
        if d > max_quad_dev {
            max_quad_dev = d;
        }
        i += 1;
    }
    if max_quad_dev > min_f32(max_f32(4.5, 5.5 * sigma_pos), 7.5) && rms_quad_pos > 1.1 {
        smooth_quad_ok = false;
    }

    // Candidate 1: Ballistic
    let g_b = xi_b[0];
    if has_real_accel && smooth_quad_ok && k_b == 1 && g_b > 12.0 && abs_f32(ax_q) < 0.45 * max_f32(g_b, 1.0) && ay_q > 10.0 {
        c_reg[n_cand] = PHYS_REGIME_BALLISTIC;
        c_aicc[n_cand] = compute_aicc(rss_b, n_wf, 1, sigma_wf_sq);
        c_r2[n_cand] = 1.0 - rss_b / max_f32(rss_wf_cv, 1.0e-6);
        c_k[n_cand] = 1;
        c_g[n_cand] = g_b;
        n_cand += 1;
    }

    // Candidate 2: Ballistic + AeroDrag
    let g_bd = xi_bd[0];
    let beta_bd = xi_bd[1];
    let mut v_rms_sq = 0.0f32;
    i = 0;
    while i < n {
        v_rms_sq += sc.vx[i] * sc.vx[i] + sc.vy[i] * sc.vy[i];
        i += 1;
    }
    v_rms_sq *= inv_n;
    if has_real_accel
        && k_bd == 2
        && g_bd > 12.0
        && abs_f32(ax_q) < 0.65 * max_f32(g_bd, 1.0)
        && beta_bd >= 0.0004
        && beta_bd <= 0.25
        && (beta_bd * v_rms_sq) >= 0.08 * g_bd
        && rss_bd < 0.35 * max_f32(rss_b, 1.0e-6)
    {
        let rms_drag_v = verlet_drag_pos_rms(sc, n, g_bd, beta_bd);
        let drag_tol = min_f32(max_f32(2.6, 5.5 * sigma_pos), 4.2);
        if rms_drag_v <= drag_tol && rms_drag_v < 0.38 * max_f32(rms_quad_pos, 1.0e-3) {
            c_reg[n_cand] = PHYS_REGIME_BALLISTIC_DRAG;
            c_aicc[n_cand] = compute_aicc(rss_bd, n_wf, 2, sigma_wf_sq);
            c_beta[n_cand] = beta_bd;
            c_r2[n_cand] = 1.0 - rss_bd / max_f32(rss_wf_cv, 1.0e-6);
            c_k[n_cand] = 2;
            c_g[n_cand] = g_bd;
            n_cand += 1;
        }
    }

    // Candidate 3: Powered Boost / Thrust
    // Fit unpruned 2-axis thrust [ax, ay] in case drag-STRidge [0, 1, 2] zeroes a smaller axis during pitchover.
    let (xi_bst2, k_bst2, rss_bst2) = stridge_cols(&sc.x_trans, n_wf, &sc.y_vec, &[0, 1], 0.03, 1.0e-3, 0);
    let use_bst2 = (rss_bst2 < 0.85 * rss_bst) && (k_bst2 == 2) && (k_bst <= 1);
    let ax_bst = if use_bst2 { xi_bst2[0] } else { xi_bst[0] };
    let ay_bst = if use_bst2 { xi_bst2[1] } else { xi_bst[1] };
    let beta_bst = if use_bst2 { 0.0 } else { xi_bst[2] };
    let rss_bst_eff = if use_bst2 { rss_bst2 } else { rss_bst };
    let k_bst_eff = if use_bst2 { k_bst2 } else { k_bst };
    let amag_bst = hypot_f32(ax_bst, ay_bst);
    let (vdx, vdy) = if spd_q > 2.0 { (vx_q / spd_q, vy_q / spd_q) } else { (0.0, -1.0) };
    let at_bst = ax_bst * vdx + ay_bst * vdy;
    let sp0 = hypot_f32(sc.vx[0], sc.vy[0]);
    let sp1 = hypot_f32(sc.vx[n - 1], sc.vy[n - 1]);
    // Boost trajectories with thrust ramp/ignition jerk often show 1.2..3.0 sd_px parabolic RMS while remaining strictly smooth and convex.
    let boost_tol = min_f32(max_f32(2.8, 4.5 * sigma_pos), 3.4);
    let boost_smooth_ok = smooth_quad_ok
        || ((rms_quad_pos <= boost_tol)
            && (r2_quad_pos >= 0.96)
            && (sigma_pos <= 0.95)
            && (rms_quad_pos <= 0.45 * rms_lin_pos)
            && (max_quad_dev <= 8.5));
    let upward_accel = (ay_bst < -3.0) && (abs_f32(vy_q) >= 1.2 * abs_f32(vx_q) || abs_f32(ay_bst) >= 1.8 * abs_f32(ax_bst)) && (vy_q < -3.0 || abs_f32(vx_q) < 1.5);
    let retro_burn = (ay_bst < -35.0) && (abs_f32(vx_q) <= 0.35 * (abs_f32(vy_q) + 3.0)) && (abs_f32(ax_bst) <= 0.35 * abs_f32(ay_bst)) && (rms_quad_pos < 1.5);
    let strong_climb_thrust = (vy_q < -10.0) && (at_bst > 35.0) && (abs_f32(vy_q) > 0.8 * abs_f32(vx_q));
    let at_quad = ax_q * vdx + ay_q * vdy;
    let extreme_boost = (at_bst > 120.0) && (at_quad > 40.0) && (rms_quad_pos < 1.5) && (sp1 > 1.5 * sp0 + 15.0);
    let not_osc = (zc_u1 == 0) && boost_smooth_ok && (disp_net >= 10.0) && (n >= 15);
    let boost_cand_ok = has_real_accel && not_osc && (upward_accel || retro_burn || strong_climb_thrust || extreme_boost);
    let mut boost_cand_added = false;
    let omega_ctrv = xi_ctrv[0];
    let ct_ctrv = xi_ctrv[1];
    // Climbing gravity/pitch turn must not lose to constant-speed CTRV when axial thrust and upward speed rise.
    let climbing_pitch_boost = (vy_q < -10.0)
        && (abs_f32(vy_q) >= 0.75 * abs_f32(vx_q))
        && (ct_ctrv > 25.0 || at_bst > 28.0)
        && (ay_q < -3.0)
        && (sp1 > sp0 + 6.0);
    if boost_cand_ok {
        let mut spike_clean = smooth_quad_ok;
        if !spike_clean {
            let mut has_gap_flip = false;
            let mut gi = 2usize;
            while gi + 2 < n {
                if (sc.t_arr[gi] - sc.t_arr[gi - 1]) > 0.14 {
                    has_gap_flip = true;
                    break;
                }
                gi += 1;
            }
            let mut max_step = 0.0f32;
            let mut med_sum = 0.0f32;
            gi = 1;
            while gi < n {
                let d = hypot_f32(sc.xc[gi] - sc.xc[gi - 1], sc.yc[gi] - sc.yc[gi - 1])
                    / max_f32(sc.t_arr[gi] - sc.t_arr[gi - 1], 1.0e-4);
                if d > max_step {
                    max_step = d;
                }
                med_sum += d;
                gi += 1;
            }
            let avg_step = med_sum / ((n - 1) as f32);
            spike_clean = !has_gap_flip && (max_step <= 4.0 * max_f32(avg_step, 15.0));
        }
        if spike_clean {
            let kb = if k_bst_eff < 1 { 1 } else { k_bst_eff };
            c_reg[n_cand] = PHYS_REGIME_POWERED_BOOST;
            c_aicc[n_cand] = compute_aicc(rss_bst_eff, n_wf, kb, sigma_wf_sq);
            c_beta[n_cand] = beta_bst;
            c_thr[n_cand] = amag_bst;
            c_r2[n_cand] = 1.0 - rss_bst_eff / max_f32(rss_wf_cv, 1.0e-6);
            c_k[n_cand] = kb as u8;
            n_cand += 1;
            boost_cand_added = true;
        }
    }

    // Candidate 8: Curved CTRV
    let an_mean = abs_f32(omega_ctrv) * spd_q;
    if has_real_accel
        && smooth_quad_ok
        && k_ctrv >= 1
        && !(boost_cand_added && climbing_pitch_boost)
        && abs_f32(omega_ctrv) > 0.08
        && an_mean > max_f32(3.0, 0.85 * abs_f32(ct_ctrv))
        && unidir_ratio > 0.05
        && abs_f32(dheading) > 0.16
        && zc_u1 == 0
    {
        c_reg[n_cand] = PHYS_REGIME_CURVED_CTRV;
        c_aicc[n_cand] = compute_aicc(rss_ctrv, n_wf, k_ctrv, sigma_wf_sq);
        c_r2[n_cand] = 1.0 - rss_ctrv / max_f32(rss_wf_cv, 1.0e-6);
        c_k[n_cand] = k_ctrv as u8;
        c_w[n_cand] = abs_f32(omega_ctrv);
        n_cand += 1;
    }

    // Pre-check Impulsive candidate
    let mut imp_valid = false;
    let mut rms_imp_cand = 999.0f32;
    if has_imp {
        rms_imp_cand = sqrt_f32(imp_rss * inv_n);
        let sp1_i = hypot_f32(imp_vx1, imp_vy1);
        let sp2_i = hypot_f32(imp_vx2, imp_vy2);
        let cos_t = (imp_vx1 * imp_vx2 + imp_vy1 * imp_vy2) / max_f32(sp1_i * sp2_i, 1.0e-6);
        let sharp_kink = (cos_t < 0.94) || (imp_dv > max_f32(40.0, 0.45 * (sp1_i + sp2_i)));
        if imp_k >= 7
            && (n - imp_k) >= 8
            && imp_dv > 25.0
            && imp_gap < 4.5
            && rms_imp_cand < min_f32(max_f32(1.1, 2.5 * sigma_pos), 1.6)
            && imp_rss < 0.20 * rss_quad_pos
            && rms_quad_pos > max_f32(2.5, 3.0 * sigma_pos)
            && sharp_kink
        {
            imp_valid = true;
        }
    }

    // Detect occlusion-blind direction flip
    let mut occl_flip = false;
    if n >= 7 {
        let mut gi = 3usize;
        while gi + 3 < n {
            if (sc.t_arr[gi] - sc.t_arr[gi - 1]) > 0.14 {
                let dt1 = max_f32(sc.t_arr[gi - 1] - sc.t_arr[gi - 3], 1.0e-4);
                let dt2 = max_f32(sc.t_arr[gi + 2] - sc.t_arr[gi], 1.0e-4);
                let vx1_g = (sc.xc[gi - 1] - sc.xc[gi - 3]) / dt1;
                let vy1_g = (sc.yc[gi - 1] - sc.yc[gi - 3]) / dt1;
                let vx2_g = (sc.xc[gi + 2] - sc.xc[gi]) / dt2;
                let vy2_g = (sc.yc[gi + 2] - sc.yc[gi]) / dt2;
                if (vx1_g * vx2_g + vy1_g * vy2_g) < -20.0 {
                    occl_flip = true;
                    break;
                }
            }
            gi += 1;
        }
    }

    // Candidate 5: Nonlinear / Harmonic Pendulum across Macro + Mid + Fast weak banks
    let k_stiff = xi_h1[1];
    let harm_r2_u1 = 1.0 - rss_h1_u1 / max_f32(tss_u1_wf, 1.0e-6);
    let (mut best_sin_r2, mut best_sin_rms, mut _best_sin_amp, mut best_omega) =
        eval_sinusoid_pos(&sc.t_arr, &sc.u1, n, sqrt_f32(max_f32(k_stiff, 0.0)));
    let mut best_wf_harm_r2 = harm_r2_u1;

    // Evaluate H_mid & H_fast banks directly into sc.y_fast / sc.x_fast
    let mut m_fast = 0usize;
    let mut r2_fast_h = 0.0f32;
    let mut xi_fast_4 = [0.0f32; 4];
    let mut k_fast_4 = 0usize;
    let mut rss_fast_4 = 0.0f32;
    let mut tss_fast = 0.0f32;
    let banks = [h_mid, h_fast];
    let mut bi = 0usize;
    while bi < 2 {
        let hb = banks[bi];
        if bi == 0 || h_fast < 0.85 * h_wide {
            let mut mb = 0usize;
            let mut tb_ss = 0.0f32;
            let mut mi = 0usize;
            while mi < MAX_WEAK_ROWS {
                let cm = t_span * (0.12 + 0.76 * ((mi as f32) / 11.0));
                if let Some((i0, i1, amp_b)) = build_weak_weights(sc, n, cm, hb) {
                    let inv_b = 1.0 / amp_b;
                    let mut ua = 0.0f32;
                    let mut ug = 0.0f32;
                    let mut ub = 0.0f32;
                    let mut u3g = 0.0f32;
                    let mut k = i0;
                    while k < i1 {
                        let uk = sc.u1[k];
                        ua += sc.w_alpha[k] * uk;
                        ug += sc.w_gamma[k] * uk;
                        ub += sc.w_beta[k] * uk;
                        u3g += sc.w_gamma[k] * (uk * uk * uk);
                        k += 1;
                    }
                    let yv = ua * inv_b;
                    sc.y_fast[mb] = yv;
                    tb_ss += yv * yv;
                    sc.x_fast[mb] = [inv_b, -(ug * inv_b), -(ub * inv_b), -(u3g * inv_b)];
                    // Also copy 3-col view into x_u2? We can solve 3-col directly via stridge_solve(&x_fast, ..., p=3)!
                    mb += 1;
                }
                mi += 1;
            }
            if mb >= 6 {
                let (xi_b3, _kb3, rss_b3) = stridge_cols(&sc.x_fast, mb, &sc.y_fast, &[0, 1, 2], 0.08, 1.0e-3, 0b010);
                let r2_b3 = 1.0 - rss_b3 / max_f32(tb_ss, 1.0e-6);
                if xi_b3[1] > 0.25 {
                    let (sr2, srms, samp, sw) = eval_sinusoid_pos(&sc.t_arr, &sc.u1, n, sqrt_f32(xi_b3[1]));
                    if sr2 > best_sin_r2 || (abs_f32(sr2 - best_sin_r2) < 0.01 && srms < best_sin_rms) {
                        best_sin_r2 = sr2;
                        best_sin_rms = srms;
                        _best_sin_amp = samp;
                        best_omega = sw;
                        best_wf_harm_r2 = max_f32(best_wf_harm_r2, r2_b3);
                    }
                }
                if bi == 1 {
                    m_fast = mb;
                    tss_fast = tb_ss;
                    let (xf4, kf4, rf4) = stridge_cols(&sc.x_fast, mb, &sc.y_fast, &[0, 1, 2, 3], 0.10, 1.0e-3, 0b0010);
                    xi_fast_4 = xf4;
                    k_fast_4 = kf4;
                    rss_fast_4 = rf4;
                    r2_fast_h = 1.0 - rf4 / max_f32(tb_ss, 1.0e-6);
                }
            }
        }
        bi += 1;
    }

    // Prominent spatial extrema reversal counter on u1 (NO fake fabrication from noisy zc!)
    let prom_thr = max_f32(10.0, 0.22 * app_u1);
    let mut n_ext = 0usize;
    let mut prev_ext = sc.u1[0];
    let mut dir_ext = 0i32;
    let mut cand_ext = sc.u1[0];
    let mut rev_travel = 0.0f32;
    let mut prev_prev_ext = sc.u1[0];
    i = 1;
    while i < n {
        let val = sc.u1[i];
        if dir_ext == 0 {
            if val - sc.u1[0] >= prom_thr {
                dir_ext = 1;
                cand_ext = val;
            } else if sc.u1[0] - val >= prom_thr {
                dir_ext = -1;
                cand_ext = val;
            }
        } else if dir_ext == 1 {
            if val > cand_ext {
                cand_ext = val;
            } else if cand_ext - val >= prom_thr {
                // Confirmed interior peak `cand_ext`
                if n_ext >= 1 {
                    let rt = min_f32(abs_f32(prev_ext - prev_prev_ext), abs_f32(cand_ext - prev_ext));
                    if rt > rev_travel { rev_travel = rt; }
                }
                prev_prev_ext = prev_ext;
                prev_ext = cand_ext;
                n_ext += 1;
                dir_ext = -1;
                cand_ext = val;
            }
        } else if val < cand_ext {
            cand_ext = val;
        } else if val - cand_ext >= prom_thr {
            if n_ext >= 1 {
                let rt = min_f32(abs_f32(prev_ext - prev_prev_ext), abs_f32(cand_ext - prev_ext));
                if rt > rev_travel { rev_travel = rt; }
            }
            prev_prev_ext = prev_ext;
            prev_ext = cand_ext;
            n_ext += 1;
            dir_ext = 1;
            cand_ext = val;
        }
        i += 1;
    }
    if n_ext >= 1 {
        let rt = min_f32(abs_f32(prev_ext - prev_prev_ext), abs_f32(sc.u1[n - 1] - prev_ext));
        if rt > rev_travel { rev_travel = rt; }
    }
    // Wait: exact extrema list including last `cand_ext` only after threshold, matching Python!
    // In Python: append cand_ext ONLY on reversal, then append u1[-1] at end -> interior extrema count == n_ext!
    // Look carefully: on a 1-peak turnaround (0 -> +A -> 0), at apex `cand_ext = +A`, then when `cand_ext - val >= prom_thr` triggers, `prev_ext` IS `+A`, `n_ext == 1`! And at loop end we only checked `prev_ext` (`+A`) against `prev_prev_ext` (`u1[0]`) and `u1[-1]`! That matches Python 100%!
    let n_reversals = n_ext;

    // Step-speed spike ratio
    let mut n_steps = 0usize;
    let mut max_step_spd = 0.0f32;
    i = 1;
    while i < n {
        let d_st = hypot_f32(sc.xc[i] - sc.xc[i - 1], sc.yc[i] - sc.yc[i - 1]);
        let dt_st = max_f32(sc.t_arr[i] - sc.t_arr[i - 1], 1.0e-4);
        let spd_st = d_st / dt_st;
        sc.tmp_n[n_steps] = spd_st;
        if spd_st > max_step_spd { max_step_spd = spd_st; }
        n_steps += 1;
        i += 1;
    }
    let med_step_spd = if n_steps > 0 {
        insertion_sort_f32(&mut sc.tmp_n, n_steps);
        sc.tmp_n[n_steps / 2]
    } else {
        0.0
    };

    let mut unoccl_big_u1 = 0usize;
    let step_dt_cap = max_f32(1.65 * dt_med, 0.055);
    i = 1;
    while i < n {
        if (sc.t_arr[i] - sc.t_arr[i - 1]) <= step_dt_cap && abs_f32(sc.u1[i] - sc.u1[i - 1]) > 0.25 * app_u1 {
            unoccl_big_u1 += 1;
        }
        i += 1;
    }

    let not_pure_vertical_apogee = (abs_f32(e1x) >= 0.42) || (n_reversals >= 2);
    let rss_lin_u1 = fit_linear_x_only(&sc.t_arr, &sc.u1, n);
    let pend_tol = max_f32(
        min_f32(max_f32(2.2, 4.5 * sigma_pos), 3.6),
        (if n_reversals >= 2 { 0.105 } else { 0.055 }) * app_u1,
    );
    let pend_shape_ok = if n_reversals >= 2 {
        (best_sin_r2 >= 0.84)
            && (best_sin_rms <= pend_tol)
            && !occl_flip
            && (unoccl_big_u1 <= 1)
            && (n >= 32 || sigma_pos < 1.8)
            && (max_step_spd <= 5.0 * max_f32(med_step_spd, 15.0))
    } else if n_reversals == 1 {
        (best_sin_r2 >= 0.96)
            && (best_sin_rms <= pend_tol)
            && !occl_flip
            && (unoccl_big_u1 <= 1)
            && (sigma_pos < 1.2)
            && (max_step_spd <= 3.6 * max_f32(med_step_spd, 12.0))
            && ((best_sin_rms <= 0.88 * rms_quad_pos) || (best_sin_r2 >= 0.992 && best_sin_rms <= 2.4))
    } else {
        false
    };
    let mut is_pend_valid = has_real_accel
        && (app_u1 >= 12.0)
        && (best_omega > 0.5)
        && (best_wf_harm_r2 >= 0.55)
        && (rev_travel >= prom_thr)
        && not_pure_vertical_apogee
        && pend_shape_ok
        && ((best_sin_rms * best_sin_rms * (n as f32)) < 0.35 * max_f32(rss_lin_u1, 1.0e-6));
    if imp_valid && is_pend_valid && rms_imp_cand < 0.72 * best_sin_rms {
        is_pend_valid = false;
    }
    if is_pend_valid {
        let ratio_sig = best_sin_rms / max_f32(sigma_pos, 0.35);
        let sin_wf_equiv = min_f32(rss_pend_2d, ratio_sig * ratio_sig * (n_wf as f32) * 0.25 + 1.0);
        let r2_p = max_f32(1.0 - rss_pend_2d / max_f32(rss_wf_cv, 1.0e-6), best_sin_r2);
        let aicc_p = compute_aicc(sin_wf_equiv, n_wf, k_pend, sigma_wf_sq);
        c_reg[n_cand] = PHYS_REGIME_PENDULUM;
        c_aicc[n_cand] = aicc_p;
        c_r2[n_cand] = r2_p;
        c_k[n_cand] = k_pend as u8;
        c_w[n_cand] = best_omega;
        n_cand += 1;
    }

    // Candidate 6: Micro-Tremor / Oscillator
    if m_fast >= 6 && app_u1 < 12.0 && disp_net < 14.0 && (zc_u1 >= 2 || zc_pos_u1 >= 2) {
        let k_stiff_f = xi_fast_4[1];
        if k_stiff_f > 2.0 && r2_fast_h >= 0.65 && app_u1 >= 0.35 {
            let w_f = sqrt_f32(k_stiff_f);
            let (sr2_f, _srms_f, _samp_f, sw_f) = eval_sinusoid_pos(&sc.t_arr, &sc.u1, n, w_f);
            let aicc_f0 = compute_aicc(tss_fast, m_fast, 0, sigma_wf_sq);
            let aicc_f1 = compute_aicc(rss_fast_4, m_fast, k_fast_4, sigma_wf_sq);
            if (aicc_f0 - aicc_f1) > 4.0 && sr2_f >= 0.70 {
                let daicc_gain = (aicc_f0 - aicc_f1) * ((n_wf as f32) / (m_fast as f32));
                c_reg[n_cand] = PHYS_REGIME_MICRO_TREMOR;
                c_aicc[n_cand] = aicc_cv - daicc_gain;
                c_r2[n_cand] = max_f32(r2_fast_h, sr2_f);
                c_k[n_cand] = k_fast_4 as u8;
                c_w[n_cand] = sw_f;
                n_cand += 1;
            }
        }
    }

    // Candidate 7: Impulsive / Collisional
    if imp_valid {
        let aicc_imp = compute_aicc(imp_rss * ((n_wf as f32) / (n as f32)) * 0.5, n_wf, 4, sigma_wf_sq);
        let r2_imp = clamp_f32(1.0 - imp_rss / max_f32(tss_pos, 1.0e-6), 0.0, 1.0);
        c_reg[n_cand] = PHYS_REGIME_IMPULSIVE;
        c_aicc[n_cand] = aicc_imp;
        c_r2[n_cand] = r2_imp;
        c_k[n_cand] = 4;
        n_cand += 1;
    }

    // 9. Constraint / Regime Jumps: Code 2 (Boost Burnout) & Code 3 (Canopy Deploy Delta-beta)
    let mut jump_code = JUMP_NONE;
    let mut delta_beta_sd = 0.0f32;
    let mut burn_split_k = 0usize;
    let mut has_burn_split = false;
    let mut chute_beta_post = 0.0f32;
    let mut best_burn_score = 0.0f32;
    let mut best_chute_score = 0.0f32;

    let can_burn = !is_pend_valid && rms_quad_pos > max_f32(0.8, 1.3 * sigma_pos);
    let can_chute = (sc.yc[n - 1] - sc.yc[0]) > 15.0;
    if n >= 18 && (can_burn || can_chute) {
        let mut sp_k = 8usize;
        while sp_k < n - 7 {
            if let (Some((c1x, c1y, tm1, r_q1)), Some((c2x, c2y, tm2, r_q2))) =
                (fit_quad_xy(&sc.t_arr, &sc.xc, &sc.yc, 0, sp_k), fit_quad_xy(&sc.t_arr, &sc.xc, &sc.yc, sp_k, n))
            {
                let v1x = c1x[1];
                let a1x = 2.0 * c1x[2];
                let v1y = c1y[1];
                let a1y = 2.0 * c1y[2];
                let v2x = c2x[1];
                let a2x = 2.0 * c2x[2];
                let v2y = c2y[1];
                let a2y = 2.0 * c2y[2];
                let rms_split_q = sqrt_f32((r_q1 + r_q2) * inv_n);
                let vy_b1 = v1y + a1y * (sc.t_arr[sp_k] - tm1);
                let vy_b2 = v2y + a2y * (sc.t_arr[sp_k] - tm2);

                // Code 2: Boost Burnout
                let vert_dom = abs_f32(a1y) > 1.3 * abs_f32(a1x)
                    && abs_f32(a2y) > 1.3 * abs_f32(a2x)
                    && abs_f32(vy_b1) > 0.8 * abs_f32(v1x);
                let smooth_split = rms_split_q <= max_f32(1.6, 2.5 * sigma_pos)
                    && rms_split_q < 0.50 * rms_quad_pos
                    && rms_quad_pos > max_f32(0.8, 1.3 * sigma_pos);
                let climb_before = a1y < -12.0 || (vy_b1 < -15.0 && (v1x * c1x[2] + v1y * c1y[2]) < 0.0);
                if vert_dom
                    && smooth_split
                    && climb_before
                    && a2y > 15.0
                    && abs_f32(vy_b2 - vy_b1) < max_f32(45.0, 0.40 * abs_f32(vy_b1) + 15.0)
                    && vy_b1 < -8.0
                    && (a2y - a1y) > 35.0
                {
                    let score = (rms_quad_pos / max_f32(rms_split_q, 0.1)) * (a2y - a1y);
                    if score > best_burn_score {
                        best_burn_score = score;
                        burn_split_k = sp_k;
                        has_burn_split = true;
                    }
                }

                // Code 3: Parachute Canopy Deploy
                if (n - sp_k) >= 10 && (sc.yc[n - 1] - sc.yc[0]) > 15.0 && v2y > 18.0 && abs_f32(v2y) > 1.5 * abs_f32(v2x) {
                    let i_s0 = if sp_k + 1 < n - 1 { sp_k + 1 } else { n - 1 };
                    let s_am = sp_k + (n - sp_k) / 2;
                    let i_b1 = if n >= 2 { n - 2 } else { 0 };
                    let va_start = sc.vy[i_s0];
                    let va_mid = sc.vy[s_am];
                    let vb_end = sc.vy[i_b1];
                    let ta_dt = max_f32(sc.t_arr[s_am] - sc.t_arr[i_s0], 1.0e-3);
                    let tb_dt = max_f32(sc.t_arr[i_b1] - sc.t_arr[s_am], 1.0e-3);

                    let mut int_vv_a = 0.0f32;
                    let mut j = i_s0;
                    while j < s_am {
                        let dtv = sc.t_arr[j + 1] - sc.t_arr[j];
                        let sja = hypot_f32(sc.vx[j], sc.vy[j]) * sc.vy[j];
                        let sjb = hypot_f32(sc.vx[j + 1], sc.vy[j + 1]) * sc.vy[j + 1];
                        int_vv_a += 0.5 * dtv * (sja + sjb);
                        j += 1;
                    }
                    let mut int_vv_b = 0.0f32;
                    j = s_am;
                    while j < i_b1 {
                        let dtv = sc.t_arr[j + 1] - sc.t_arr[j];
                        let sja = hypot_f32(sc.vx[j], sc.vy[j]) * sc.vy[j];
                        let sjb = hypot_f32(sc.vx[j + 1], sc.vy[j + 1]) * sc.vy[j + 1];
                        int_vv_b += 0.5 * dtv * (sja + sjb);
                        j += 1;
                    }
                    let dva = va_mid - va_start;
                    let dvb = vb_end - va_mid;
                    let det_c = ta_dt * (-int_vv_b) - (-int_vv_a) * tb_dt;

                    let pk_lo = sp_k.saturating_sub(2);
                    let pk_hi = if sp_k + 4 < n { sp_k + 4 } else { n };
                    let mut v_peak_post = -1.0e18f32;
                    let mut k_pk = pk_lo;
                    while k_pk < pk_hi {
                        if sc.vy[k_pk] > v_peak_post { v_peak_post = sc.vy[k_pk]; }
                        k_pk += 1;
                    }
                    let end_lo = n.saturating_sub(5);
                    let mut sum_end = 0.0f32;
                    let mut k_en = end_lo;
                    while k_en < n {
                        sum_end += sc.vy[k_en];
                        k_en += 1;
                    }
                    let v_end_post = sum_end / ((n - end_lo) as f32);
                    let mut min_vy_post = 1.0e18f32;
                    let mut k_mn = sp_k;
                    while k_mn < n {
                        if sc.vy[k_mn] < min_vy_post { min_vy_post = sc.vy[k_mn]; }
                        k_mn += 1;
                    }

                    // Require clean pre-chute arc and sustained downward post-chute travel
                    // (tolerant of a single canopy-inflation recoil sample when median vy > 25).
                    let j_split_c3 = if sp_k > 0 && sp_k < n {
                        hypot_f32(sc.xc[sp_k] - sc.xc[sp_k - 1], sc.yc[sp_k] - sc.yc[sp_k - 1])
                    } else {
                        999.0
                    };
                    let mut max_step_pre_c3 = 0.0f32;
                    let mut k_mp = 1usize;
                    while k_mp < sp_k {
                        let d = hypot_f32(sc.xc[k_mp] - sc.xc[k_mp - 1], sc.yc[k_mp] - sc.yc[k_mp - 1]);
                        if d > max_step_pre_c3 {
                            max_step_pre_c3 = d;
                        }
                        k_mp += 1;
                    }
                    let n_post_c3 = n.saturating_sub(sp_k);
                    let dy_post_c3 = sc.yc[n - 1] - sc.yc[sp_k - 1];
                    let mut n_gt10_c3 = 0usize;
                    let mut n_gt25_c3 = 0usize;
                    let mut k_p3 = sp_k;
                    while k_p3 < n {
                        if sc.vy[k_p3] > 10.0 {
                            n_gt10_c3 += 1;
                        }
                        if sc.vy[k_p3] > 25.0 {
                            n_gt25_c3 += 1;
                        }
                        k_p3 += 1;
                    }
                    let tail_lo_c3 = if n > 8 && n - 8 >= sp_k { n - 8 } else { sp_k };
                    let (v_end_lin_y_c3, _, _) = fit_linear_1d_slice(&sc.t_arr, &sc.yc, tail_lo_c3, n);
                    let post_descend_ok = min_vy_post > 10.0
                        || (n_post_c3 >= 10
                            && j_split_c3 < 13.0
                            && max_step_pre_c3 < 13.0
                            && dy_post_c3 > 25.0
                            && v_end_lin_y_c3 > 10.0
                            && (n_gt10_c3 * 10) >= (n_post_c3 * 7)
                            && (n_gt25_c3 * 2) > n_post_c3
                            && v_peak_post > 110.0);

                    if post_descend_ok
                        && v_peak_post > max_f32(40.0, 1.40 * max_f32(v_end_post, 5.0))
                        && (v_peak_post - v_end_post) > 25.0
                        && v_end_post > 15.0
                    {
                        let (mut g_p, mut beta_p) = if abs_f32(det_c) > 1.0e-3 {
                            ((dva * (-int_vv_b) - (-int_vv_a) * dvb) / det_c, (ta_dt * dvb - dva * tb_dt) / det_c)
                        } else {
                            (720.0, 0.0)
                        };
                        if g_p < 30.0 || g_p > 2500.0 || beta_p < 0.001 || beta_p > 0.3 {
                            let g_ref = if a1y > 30.0 { a1y } else { 720.0 };
                            if int_vv_a > 10.0 {
                                beta_p = max_f32(0.0, (g_ref * ta_dt - dva) / int_vv_a);
                                g_p = g_ref;
                            }
                        }
                        let b_pre = if v1y > 10.0 && a1y > 0.0 {
                            max_f32(0.0, (g_p - a1y) / max_f32(v1y * abs_f32(v1y), 100.0))
                        } else {
                            0.0
                        };
                        let db = beta_p - b_pre;
                        let rms_pre_q = sqrt_f32(r_q1 / (sp_k as f32));
                        if db >= 0.002
                            && db <= 0.25
                            && (beta_p * v_peak_post * v_peak_post) > 1.35 * g_p
                            && dva < -20.0
                            && rms_pre_q < max_f32(2.2, 3.0 * sigma_pos)
                            && (min_vy_post > 10.0 || a1y > 38.0)
                        {
                            let sc_v = (v_peak_post - v_end_post) * db;
                            if sc_v > best_chute_score {
                                best_chute_score = sc_v;
                                delta_beta_sd = db;
                                chute_beta_post = beta_p;
                            }
                        }
                    }
                }
            }
            sp_k += 1;
        }
    }

    if best_chute_score > 0.0 && delta_beta_sd > 0.002 {
        jump_code = JUMP_CHUTE_DEPLOY;
    } else if best_burn_score > 0.0 && !is_pend_valid {
        jump_code = JUMP_BOOST_BURNOUT;
        if has_burn_split {
            if let Some((_cx2, cy2, _, rq2)) = fit_quad_xy(&sc.t_arr, &sc.xc, &sc.yc, burn_split_k, n) {
                let rq1 = match fit_quad_xy(&sc.t_arr, &sc.xc, &sc.yc, 0, burn_split_k) {
                    Some((_, _, _, r1)) => r1,
                    None => 0.0,
                };
                let a2y = 2.0 * cy2[2];
                let rss_burn_2 = rq1 + rq2;
                if a2y > 15.0 {
                    c_reg[n_cand] = PHYS_REGIME_BALLISTIC;
                    c_aicc[n_cand] = compute_aicc(rss_burn_2 * ((n_wf as f32) / (n as f32)) * 0.5, n_wf, 4, sigma_wf_sq);
                    c_r2[n_cand] = clamp_f32(1.0 - rss_burn_2 / max_f32(tss_pos, 1.0e-6), 0.0, 1.0);
                    c_k[n_cand] = 4;
                    c_g[n_cand] = a2y;
                    n_cand += 1;
                }
            }
        }
    }

    // Candidate 4: Parachute / Terminal Descent
    let saw_flight = prior_flight != 0 || jump_code == JUMP_BOOST_BURNOUT || jump_code == JUMP_CHUTE_DEPLOY;
    let is_downward = vy_q > 12.0 && abs_f32(vy_q) > 1.8 * abs_f32(vx_q);
    let dy_tot_4 = sc.yc[n - 1] - sc.yc[0];
    let vy_avg_4 = dy_tot_4 / max_f32(sc.t_arr[n - 1] - sc.t_arr[0], 0.05);
    let tail_dy_4 = if n >= 8 { sc.yc[n - 1] - sc.yc[n - 8] } else { dy_tot_4 };
    let steady_chute = saw_flight
        && n >= 16
        && dy_tot_4 > 12.0
        && vy_avg_4 > 12.0
        && tail_dy_4 > 3.0
        && abs_f32(ay_q) < 0.35 * max_f32(720.0, abs_f32(ay_q))
        && rms_lin_pos < max_f32(3.5, 4.0 * sigma_pos);
    let open_chute_drag = smooth_quad_ok && beta_bd >= 0.015 && ay_q < -25.0 && vy_q > 15.0 && abs_f32(vy_q) > 2.2 * abs_f32(vx_q) && rss_bd < 0.5 * rss_b;

    // Steady terminal descent with lateral canopy sway (direct canopy acquisition without prior boost).
    let mut sway_chute = false;
    let mut rss_sway_eff = rss_wf_cv;
    if n >= 32 && !is_pend_valid && !imp_valid && dy_tot_4 > 33.0 && tail_dy_4 > 3.0 {
        let mut max_dx_s = 0.0f32;
        let mut max_dy_s = 0.0f32;
        let mut ii = 1usize;
        while ii < n {
            max_dx_s = max_f32(max_dx_s, abs_f32(sc.xc[ii] - sc.xc[ii - 1]));
            max_dy_s = max_f32(max_dy_s, abs_f32(sc.yc[ii] - sc.yc[ii - 1]));
            ii += 1;
        }
        if max_dx_s < 14.0 && max_dy_s < 14.0 {
            let (vx_1d, rss_lx, _) = fit_linear_1d_slice(&sc.t_arr, &sc.xc, 0, n);
            let (vy_1d, rss_ly, syy) = fit_linear_1d_slice(&sc.t_arr, &sc.yc, 0, n);
            if vy_1d > 17.5 && vy_1d < 55.0 && abs_f32(vy_1d) > 0.90 * abs_f32(vx_1d) {
                let rms_lx = sqrt_f32(rss_lx * inv_n);
                let rms_ly = sqrt_f32(rss_ly * inv_n);
                let r2_y = 1.0 - rss_ly / max_f32(syy, 1.0e-6);
                if r2_y > 0.80 && rms_ly < 5.0 && rms_lx > 5.5 {
                    // Reject sharp single-corner collisions via 2-piece linear fit on x(t).
                    let mut best_hinge_rss = 1.0e18f32;
                    let mut k_h = 6usize;
                    while k_h + 5 < n {
                        let (_, r1, _) = fit_linear_1d_slice(&sc.t_arr, &sc.xc, 0, k_h);
                        let (_, r2, _) = fit_linear_1d_slice(&sc.t_arr, &sc.xc, k_h, n);
                        if r1 + r2 < best_hinge_rss {
                            best_hinge_rss = r1 + r2;
                        }
                        k_h += 3;
                    }
                    let hinge_rms_x = sqrt_f32(best_hinge_rss * inv_n);
                    if hinge_rms_x > 2.2 {
                        sway_chute = true;
                        rss_sway_eff = rss_ly + min_f32(best_hinge_rss, 0.35 * rss_lx);
                    }
                }
            }
        }
    }

    if (is_downward || sway_chute) && !is_pend_valid && !imp_valid && (jump_code == JUMP_CHUTE_DEPLOY || steady_chute || open_chute_drag || sway_chute) {
        let base_rss = min_f32(rss_wf_cv, rss_bd);
        let rss_4 = if jump_code == JUMP_CHUTE_DEPLOY {
            base_rss * 0.35
        } else if sway_chute {
            min_f32(base_rss * 0.70, rss_sway_eff)
        } else {
            base_rss * 0.70
        };
        let b_out = if chute_beta_post > 0.0 {
            chute_beta_post
        } else if beta_bd > 0.0 {
            beta_bd
        } else {
            delta_beta_sd
        };
        c_reg[n_cand] = PHYS_REGIME_PARACHUTE;
        c_aicc[n_cand] = compute_aicc(rss_4, n_wf, 2, sigma_wf_sq);
        c_beta[n_cand] = b_out;
        c_r2[n_cand] = max_f32(r2_lin_pos, 1.0 - rss_4 / max_f32(rss_wf_cv, 1.0e-6));
        c_k[n_cand] = 2;
        n_cand += 1;
    }

    // 10. Pick lowest AICc (require >= 3.0 delta-AICc to depart from Inertial CV)
    let mut best_idx = 0usize;
    let mut k = 1usize;
    while k < n_cand {
        if c_aicc[k] < c_aicc[best_idx] {
            best_idx = k;
        }
        k += 1;
    }
    if c_reg[best_idx] != PHYS_REGIME_LINEAR_CV && (aicc_cv - c_aicc[best_idx]) < 3.0 {
        best_idx = 0;
    }
    let best_reg = c_reg[best_idx];
    let best_aicc = c_aicc[best_idx];
    let best_beta = c_beta[best_idx];
    let best_thr = c_thr[best_idx];
    let best_r2_wf = c_r2[best_idx];
    let best_k = c_k[best_idx];
    let best_g = c_g[best_idx];
    let best_w = c_w[best_idx];

    // Akaike weight softmax confidence
    let mut sum_w = 0.0f32;
    k = 0;
    while k < n_cand {
        let da = clamp_f32(c_aicc[k] - best_aicc, 0.0, 50.0);
        sum_w += fast_expf(-0.5 * da);
        k += 1;
    }
    let conf_aicc = 1.0 / max_f32(sum_w, 1.0);
    let best_r2 = if best_reg == PHYS_REGIME_LINEAR_CV {
        if disp_net < 2.5 {
            max_f32(0.0, 1.0 - rss_lin_pos / max_f32(tss_pos + 1.0, 1.0))
        } else {
            r2_lin_pos
        }
    } else if best_reg == PHYS_REGIME_IMPULSIVE {
        best_r2_wf
    } else {
        let extra_q = if best_reg >= PHYS_REGIME_BALLISTIC && best_reg <= PHYS_REGIME_POWERED_BOOST || best_reg == PHYS_REGIME_CURVED_CTRV {
            r2_quad_pos
        } else {
            0.0
        };
        clamp_f32(max_f32(best_r2_wf, extra_q), 0.0, 1.0)
    };
    let conf = clamp_f32(conf_aicc * (0.35 + 0.65 * best_r2), 0.0, 1.0);

    // Pre-ignition stationary pad prefix dilutes full-window quadratic fit into false LINEAR_CV; inspect post-pad tail.
    let (fin_reg, fin_k, fin_conf, fin_r2, fin_thr) = if best_reg == PHYS_REGIME_LINEAR_CV
        && zc_u1 == 0
        && !occl_flip
        && unoccl_big_u1 == 0
        && !imp_valid
        && !is_pend_valid
        && sigma_pos <= 0.85
        && sp1 > 1.8 * sp0 + 25.0
        && vy_q < -10.0
        && abs_f32(vy_q) >= 1.1 * abs_f32(vx_q)
        && n >= 15
    {
        let mut k_pad = 0usize;
        while k_pad + 9 <= n && hypot_f32(sc.xc[k_pad] - sc.xc[0], sc.yc[k_pad] - sc.yc[0]) <= 5.5 {
            k_pad += 1;
        }
        let mut res = (best_reg, best_k, conf, best_r2, best_thr);
        if k_pad >= 3 && k_pad + 8 <= n && (sc.t_arr[n - 1] - sc.t_arr[k_pad]) >= 0.24 {
            if let Some((cx_t, cy_t, _, rq_t)) = fit_quad_xy(&sc.t_arr, &sc.xc, &sc.yc, k_pad, n) {
                let (_, _, _, _, rl_t, tss_t) = fit_linear_xy(&sc.t_arr, &sc.xc, &sc.yc, k_pad, n);
                let nt = (n - k_pad) as f32;
                let rms_qt = sqrt_f32(rq_t / nt);
                let rms_lt = sqrt_f32(rl_t / nt);
                let r2_qt = clamp_f32(1.0 - rq_t / max_f32(tss_t, 1.0e-6), 0.0, 1.0);
                let ax_t = 2.0 * cx_t[2];
                let ay_t = 2.0 * cy_t[2];
                let vx_t = cx_t[1];
                let vy_t = cy_t[1];
                let tail_tol = min_f32(max_f32(1.6, 3.8 * sigma_pos), 2.5);
                if ay_t < -40.0
                    && vy_t < -15.0
                    && abs_f32(vy_t) >= 1.1 * abs_f32(vx_t)
                    && r2_qt >= 0.96
                    && rms_qt <= tail_tol
                    && rms_qt < 0.38 * max_f32(rms_lt, 1.0e-3)
                    && max_step_spd <= 4.5 * max_f32(med_step_spd, 15.0)
                {
                    res = (
                        PHYS_REGIME_POWERED_BOOST,
                        2u8,
                        clamp_f32(r2_qt, 0.72, 0.99),
                        r2_qt,
                        hypot_f32(ax_t, ay_t),
                    );
                }
            }
        }
        res
    } else {
        (best_reg, best_k, conf, best_r2, best_thr)
    };

    OdeResult {
        phys_regime: fin_reg,
        constraint_jump_code: jump_code,
        active_terms: fin_k,
        eff_samples: n as u8,
        phys_regime_conf: fin_conf,
        ode_r2: fin_r2,
        aero_drag_beta_px_inv: best_beta / s,
        thrust_accel_px_s2: fin_thr * s,
        delta_beta_px_inv: delta_beta_sd / s,
        est_g_px_s2: best_g * s,
        est_omega_rad_s: best_w,
        sigma_pos_px: sigma_pos * s,
    }
}


/// Extracts chronological camera-compensated world trajectory `[0..cnt)` from `HIST_RING_LEN = 64` rings.
/// Slot index is `f & 63`; camera pan `(hist_cam[f] - hist_cam[head_f])` moves all samples into `head_f`'s frame.
pub fn extract_compensated_window_from_ring(
    arc_raw_x: &[f32; 64],
    arc_raw_y: &[f32; 64],
    arc_vis: &[u8; 64],
    hist_cam_x: &[f32; 64],
    hist_cam_y: &[f32; 64],
    hist_t_sec: &[f64; 64],
    head_f: u32,
    start_f: u32,
    out_x: &mut [f32; 64],
    out_y: &mut [f32; 64],
    out_t: &mut [f64; 64],
    out_vis: &mut [u8; 64],
) -> usize {
    if head_f < start_f {
        out_vis.fill(0);
        return 0;
    }
    let head_slot = (head_f as usize) & HIST_MASK;
    let cam_hx = hist_cam_x[head_slot];
    let cam_hy = hist_cam_y[head_slot];
    let min_ring = head_f.saturating_sub((HIST_RING_LEN - 1) as u32);
    let lower = if min_ring > start_f { min_ring } else { start_f };
    let mut cnt = 0usize;
    let mut f = lower;
    while f <= head_f && cnt < HIST_RING_LEN {
        let slot = (f as usize) & HIST_MASK;
        out_x[cnt] = arc_raw_x[slot] + (hist_cam_x[slot] - cam_hx);
        out_y[cnt] = arc_raw_y[slot] + (hist_cam_y[slot] - cam_hy);
        out_t[cnt] = hist_t_sec[slot];
        out_vis[cnt] = arc_vis[slot];
        cnt += 1;
        f += 1;
    }
    let mut k = cnt;
    while k < HIST_RING_LEN {
        out_vis[k] = 0;
        k += 1;
    }
    cnt
}

/// Zero-stack helper that extracts visible compensated samples normalized by `scale_s` directly into `PhysicsIdScratch`.
#[allow(dead_code)]
fn extract_compensated_vis_to_scratch(
    arc_raw_x: &[f32; 64],
    arc_raw_y: &[f32; 64],
    arc_vis: &[u8; 64],
    hist_cam_x: &[f32; 64],
    hist_cam_y: &[f32; 64],
    hist_t_sec: &[f64; 64],
    head_f: u32,
    start_f: u32,
    s: f32,
    sc: &mut PhysicsIdScratch,
) -> usize {
    if head_f < start_f {
        return 0;
    }
    let head_slot = (head_f as usize) & HIST_MASK;
    let cam_hx = hist_cam_x[head_slot];
    let cam_hy = hist_cam_y[head_slot];
    let min_ring = head_f.saturating_sub((HIST_RING_LEN - 1) as u32);
    let lower = if min_ring > start_f { min_ring } else { start_f };
    let mut nv = 0usize;
    let mut t0_sec = 0.0f64;
    let mut has_t0 = false;
    let mut f = lower;
    while f <= head_f && nv < HIST_RING_LEN {
        let slot = (f as usize) & HIST_MASK;
        if arc_vis[slot] != 0 {
            let wx = arc_raw_x[slot] + (hist_cam_x[slot] - cam_hx);
            let wy = arc_raw_y[slot] + (hist_cam_y[slot] - cam_hy);
            let ts = hist_t_sec[slot];
            if wx.is_finite() && wy.is_finite() && ts.is_finite() {
                if !has_t0 {
                    t0_sec = ts;
                    has_t0 = true;
                }
                sc.ts_v[nv] = (ts - t0_sec) as f32;
                sc.xs_v[nv] = wx / s;
                sc.ys_v[nv] = wy / s;
                nv += 1;
            }
        }
        f += 1;
    }
    nv
}

#[inline(never)]
pub fn discover_ode_window(input: &OdeWindowInput<'_>, prior_flight: u8, sc: &mut PhysicsIdScratch) -> OdeResult {
    let n_tot = input.x_world.len().min(input.y_world.len()).min(input.t_sec.len()).min(input.vis.len());
    if n_tot < 15 || n_tot > HIST_RING_LEN {
        return OdeResult::zero(0.0);
    }
    let s = max_f32(input.scale_s, 0.25);
    let mut nv = 0usize;
    let mut t0_f64 = 0.0f64;
    let mut has_t0 = false;
    let mut i = 0usize;
    while i < n_tot {
        if input.vis[i] != 0 && input.x_world[i].is_finite() && input.y_world[i].is_finite() && input.t_sec[i].is_finite() {
            if !has_t0 {
                t0_f64 = input.t_sec[i];
                has_t0 = true;
            }
            sc.ts_v[nv] = (input.t_sec[i] - t0_f64) as f32;
            sc.xs_v[nv] = input.x_world[i] / s;
            sc.ys_v[nv] = input.y_world[i] / s;
            nv += 1;
        }
        i += 1;
    }
    discover_from_normalized_vis(nv, s, prior_flight, sc)
}

#[inline]
pub fn should_run_ode_discovery(
    st: &OdeTrackState,
    frame_idx: u32,
    slot_m: usize,
    age_frames: u32,
    fista_sharp_knots: u8,
    vis_reentry: bool,
) -> bool {
    (((frame_idx as usize) + slot_m) & 3) == 0
        || age_frames < 24
        || fista_sharp_knots > 0
        || vis_reentry
        || (st.latched_jump_code == JUMP_CHUTE_DEPLOY && st.jump_hold_rem > 0)
}

#[inline]
fn begin_ode_tick_countdowns(state: &mut OdeTrackState, frame_idx: u32) -> u8 {
    let dt_f: u8 = if state.has_ticked == 0 {
        1
    } else {
        let diff = frame_idx.saturating_sub(state.last_tick_frame);
        if diff > 255 { 255 } else { diff as u8 }
    };
    state.has_ticked = 1;
    state.last_tick_frame = frame_idx;
    state.burnout_cooldown = state.burnout_cooldown.saturating_sub(dt_f);
    state.chute_cooldown = state.chute_cooldown.saturating_sub(dt_f);
    dt_f
}

#[inline]
fn emit_cached_ode_step(state: &mut OdeTrackState, dt_f: u8) -> OdeResult {
    let mut res = OdeResult::zero(state.sigma_pos_px);
    res.phys_regime = state.last_regime;
    res.phys_regime_conf = state.conf_ema;
    res.ode_r2 = state.ode_r2_ema;
    res.aero_drag_beta_px_inv = if state.last_regime == PHYS_REGIME_BALLISTIC_DRAG
        || state.last_regime == PHYS_REGIME_PARACHUTE
        || state.last_regime == PHYS_REGIME_POWERED_BOOST
    {
        state.beta_ema
    } else {
        0.0
    };
    res.thrust_accel_px_s2 = if state.last_regime == PHYS_REGIME_POWERED_BOOST { state.thrust_ema } else { 0.0 };
    res.est_g_px_s2 = state.g_eff_px_s2;
    res.est_omega_rad_s = state.omega_est_rad_s;
    if state.jump_hold_rem > 0 {
        state.jump_hold_rem = state.jump_hold_rem.saturating_sub(dt_f);
        if state.jump_hold_rem > 0 {
            res.constraint_jump_code = state.latched_jump_code;
            if state.latched_jump_code == JUMP_CHUTE_DEPLOY {
                res.delta_beta_px_inv = state.latched_delta_beta;
            }
        } else {
            state.latched_jump_code = JUMP_NONE;
            state.latched_delta_beta = 0.0;
        }
    }
    res
}

#[inline]
fn apply_discovered_ode_step(state: &mut OdeTrackState, dt_f: u8, mut res: OdeResult) -> OdeResult {
    if res.phys_regime >= PHYS_REGIME_BALLISTIC && res.phys_regime <= PHYS_REGIME_PARACHUTE {
        state.prior_flight = 1;
    }
    let raw_jump = res.constraint_jump_code;
    let raw_dbeta = res.delta_beta_px_inv;
    if raw_jump == JUMP_BOOST_BURNOUT && state.burnout_cooldown == 0 {
        state.prior_flight = 1;
        state.latched_jump_code = JUMP_BOOST_BURNOUT;
        state.jump_hold_rem = 15;
        state.burnout_cooldown = 56;
        state.latched_delta_beta = 0.0;
    } else if raw_jump == JUMP_CHUTE_DEPLOY && state.chute_cooldown == 0 {
        state.prior_flight = 1;
        state.latched_jump_code = JUMP_CHUTE_DEPLOY;
        state.jump_hold_rem = 15;
        state.chute_cooldown = 56;
        state.latched_delta_beta = raw_dbeta;
    } else {
        if (raw_jump == JUMP_BOOST_BURNOUT && state.burnout_cooldown > 0)
            || (raw_jump == JUMP_CHUTE_DEPLOY && state.chute_cooldown > 0)
        {
            res.constraint_jump_code = JUMP_NONE;
            res.delta_beta_px_inv = 0.0;
        }
        if state.jump_hold_rem > 0 {
            state.jump_hold_rem = state.jump_hold_rem.saturating_sub(dt_f);
            if state.jump_hold_rem > 0 {
                if raw_jump == JUMP_CHUTE_DEPLOY
                    && state.latched_jump_code == JUMP_CHUTE_DEPLOY
                    && raw_dbeta > state.latched_delta_beta
                {
                    state.latched_delta_beta = raw_dbeta;
                }
                res.constraint_jump_code = state.latched_jump_code;
                if state.latched_jump_code == JUMP_CHUTE_DEPLOY {
                    res.delta_beta_px_inv = state.latched_delta_beta;
                }
            } else {
                state.latched_jump_code = JUMP_NONE;
                state.latched_delta_beta = 0.0;
            }
        }
    }
    state.last_regime = res.phys_regime;
    state.sigma_pos_px = res.sigma_pos_px;
    state.conf_ema = if state.conf_ema == 0.0 { res.phys_regime_conf } else { 0.7 * state.conf_ema + 0.3 * res.phys_regime_conf };
    state.ode_r2_ema = if state.ode_r2_ema == 0.0 { res.ode_r2 } else { 0.7 * state.ode_r2_ema + 0.3 * res.ode_r2 };
    if res.aero_drag_beta_px_inv > 0.0 {
        state.beta_ema = if state.beta_ema == 0.0 { res.aero_drag_beta_px_inv } else { 0.75 * state.beta_ema + 0.25 * res.aero_drag_beta_px_inv };
    }
    if res.thrust_accel_px_s2 > 0.0 {
        state.thrust_ema = if state.thrust_ema == 0.0 { res.thrust_accel_px_s2 } else { 0.75 * state.thrust_ema + 0.25 * res.thrust_accel_px_s2 };
    }
    if res.est_g_px_s2 > 0.0 {
        state.g_eff_px_s2 = res.est_g_px_s2;
    }
    if res.est_omega_rad_s > 0.0 {
        state.omega_est_rad_s = res.est_omega_rad_s;
    }
    res
}

pub fn step_ode_track(
    state: &mut OdeTrackState,
    input: &OdeWindowInput<'_>,
    frame_idx: u32,
    run_discovery: bool,
    sc: &mut PhysicsIdScratch,
) -> OdeResult {
    let dt_f = begin_ode_tick_countdowns(state, frame_idx);
    if !run_discovery {
        return emit_cached_ode_step(state, dt_f);
    }
    let res = discover_ode_window(input, state.prior_flight, sc);
    apply_discovered_ode_step(state, dt_f, res)
}

#[allow(dead_code)]
#[inline(never)]
pub fn step_ode_track_from_ring(
    st: &mut OdeTrackState,
    arc_raw_x: &[f32; 64],
    arc_raw_y: &[f32; 64],
    arc_vis: &[u8; 64],
    hist_cam_x: &[f32; 64],
    hist_cam_y: &[f32; 64],
    hist_t_sec: &[f64; 64],
    head_f: u32,
    start_f: u32,
    scale_s: f32,
    run_discovery: bool,
    scratch: &mut PhysicsIdScratch,
) -> OdeResult {
    let dt_f = begin_ode_tick_countdowns(st, head_f);
    if !run_discovery {
        return emit_cached_ode_step(st, dt_f);
    }
    let s = max_f32(scale_s, 0.25);
    let nv = extract_compensated_vis_to_scratch(
        arc_raw_x,
        arc_raw_y,
        arc_vis,
        hist_cam_x,
        hist_cam_y,
        hist_t_sec,
        head_f,
        start_f,
        s,
        scratch,
    );
    let res = discover_from_normalized_vis(nv, s, st.prior_flight, scratch);
    apply_discovered_ode_step(st, dt_f, res)
}

#[inline(never)]
pub fn step_ode_track_from_window(
    st: &mut OdeTrackState,
    xs: &[f32],
    ys: &[f32],
    ts: &[f64],
    vis_idx: &[u8],
    n_vis: usize,
    head_f: u32,
    scale_s: f32,
    run_discovery: bool,
    scratch: &mut PhysicsIdScratch,
) -> OdeResult {
    let dt_f = begin_ode_tick_countdowns(st, head_f);
    if !run_discovery {
        return emit_cached_ode_step(st, dt_f);
    }
    let s = max_f32(scale_s, 0.25);
    let inv_s = 1.0f32 / s;
    let lim = n_vis.min(vis_idx.len()).min(HIST_RING_LEN);
    let mut nv = 0usize;
    let mut t0_sec = 0.0f64;
    let mut has_t0 = false;
    for k in 0..lim {
        let i = vis_idx[k] as usize;
        let wx = xs[i];
        let wy = ys[i];
        let t_i = ts[i];
        if wx.is_finite() && wy.is_finite() && t_i.is_finite() {
            if !has_t0 {
                t0_sec = t_i;
                has_t0 = true;
            }
            scratch.ts_v[nv] = (t_i - t0_sec) as f32;
            scratch.xs_v[nv] = wx * inv_s;
            scratch.ys_v[nv] = wy * inv_s;
            nv += 1;
        }
    }
    let res = discover_from_normalized_vis(nv, s, st.prior_flight, scratch);
    apply_discovered_ode_step(st, dt_f, res)
}


// =============================================================================
// Section B: Taubin Circle Fit, Galileo Peg Jump, Jacobi AGM Elliptic Period,
// Damping zeta/Q, Specific Energy & Angular Momentum, Lyapunov, Kuramoto
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TaubinFitResult {
    pub valid_circle: u8,
    pub smooth_arc: u8,
    pub _pad: [u8; 2],
    pub xc: f32,
    pub yc: f32,
    pub radius: f32,
    pub curvature_inv_px: f32,
    pub rmse_px: f32,
    pub sagitta_px: f32,
    pub span_px: f32,
    pub mean_y: f32,
    pub taubin_vs_kasa_delta_px: f32,
}

impl TaubinFitResult {
    pub const fn zeroed() -> Self {
        Self {
            valid_circle: 0,
            smooth_arc: 0,
            _pad: [0; 2],
            xc: 0.0,
            yc: 0.0,
            radius: 0.0,
            curvature_inv_px: 0.0,
            rmse_px: 0.0,
            sagitta_px: 0.0,
            span_px: 0.0,
            mean_y: 0.0,
            taubin_vs_kasa_delta_px: 0.0,
        }
    }
    #[allow(dead_code)]
    pub const fn zero() -> Self {
        Self::zeroed()
    }
    #[allow(dead_code)]
    pub const fn new() -> Self {
        Self::zeroed()
    }
}

/// Checks smooth visible arc traversal without isolated ID-switch/occlusion teleport spikes (0 B stack alloc).
#[inline]
pub fn check_smooth_arc_traversal_vis(xs: &[f32], ys: &[f32], vis: &[u8], s: f32) -> (bool, f32, f32) {
    let m = xs.len().min(ys.len()).min(vis.len());
    let mut n = 0usize;
    let mut min_x = 1e9f32;
    let mut max_x = -1e9f32;
    let mut min_y = 1e9f32;
    let mut max_y = -1e9f32;
    let mut path_len = 0.0f32;
    let mut max_step = 0.0f32;
    let mut active_steps = 0usize;
    let step_thresh = 0.30 * s;
    let floor_step = 2.8 * s;
    let mut prev2_d = 0.0f32;
    let mut prev_d = 0.0f32;
    let mut prev_x = 0.0f32;
    let mut prev_y = 0.0f32;

    for i in 0..m {
        if vis[i] == 0 {
            continue;
        }
        let x = xs[i];
        let y = ys[i];
        if !x.is_finite() || !y.is_finite() {
            return (false, 0.0, 0.0);
        }
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
        if n > 0 {
            let dx = x - prev_x;
            let dy = y - prev_y;
            let d = wasm_sqrtf(dx * dx + dy * dy);
            path_len += d;
            if d > max_step {
                max_step = d;
            }
            if d >= step_thresh {
                active_steps += 1;
            }
            if n == 2 {
                if prev_d > 3.2 * d.max(floor_step) {
                    return (false, 0.0, 0.0);
                }
            } else if n >= 3 {
                let nbr = prev2_d.max(d).max(floor_step);
                if prev_d > 2.8 * nbr {
                    return (false, 0.0, 0.0);
                }
            }
            prev2_d = prev_d;
            prev_d = d;
        }
        prev_x = x;
        prev_y = y;
        n += 1;
    }
    if n < 7 {
        return (false, 0.0, 0.0);
    }
    if prev_d > 3.2 * prev2_d.max(floor_step) {
        return (false, 0.0, 0.0);
    }
    let span = wasm_sqrtf((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y));
    let min_active = (n / 2).max(4);
    let smooth = span >= 14.0 * s
        && path_len >= 14.0 * s
        && active_steps >= min_active
        && max_step <= 0.42 * path_len;
    (smooth, span, path_len)
}

/// Exact 3x3 symmetric Jacobi eigen-decomposition in `f64` (zero trig / zero host imports).
/// Returns `(lambda_min, lambda_mid, eigenvector_min)` for symmetric `S`.
#[inline]
fn jacobi_3x3_min_eig_f64(mut a: [[f64; 3]; 3]) -> (f64, f64, [f64; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

    for _sweep in 0..10 {
        let mut max_off = 0.0f64;
        for &(p, q) in &PAIRS {
            let apq = a[p][q];
            let abs_apq = apq.abs();
            if abs_apq > max_off {
                max_off = abs_apq;
            }
            if abs_apq < 1e-14 {
                continue;
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + wasm_sqrtd(1.0 + tau * tau))
            } else {
                -1.0 / (-tau + wasm_sqrtd(1.0 + tau * tau))
            };
            let c = 1.0 / wasm_sqrtd(1.0 + t * t);
            let s = t * c;

            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for k in 0..3 {
                if k != p && k != q {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    let nkp = c * akp - s * akq;
                    let nkq = s * akp + c * akq;
                    a[k][p] = nkp;
                    a[p][k] = nkp;
                    a[k][q] = nkq;
                    a[q][k] = nkq;
                }
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
        if max_off < 1e-13 {
            break;
        }
    }

    let e0 = a[0][0];
    let e1 = a[1][1];
    let e2 = a[2][2];
    let (i0, i1) = if e0 <= e1 && e0 <= e2 {
        (0, if e1 <= e2 { 1 } else { 2 })
    } else if e1 <= e0 && e1 <= e2 {
        (1, if e0 <= e2 { 0 } else { 2 })
    } else {
        (2, if e0 <= e1 { 0 } else { 1 })
    };
    (a[i0][i0], a[i1][i1], [v[0][i0], v[1][i0], v[2][i0]])
}

/// Exact zero-allocation Taubin algebraic circle fit + Kåsa comparison.

fn fit_taubin_circle_subpx_core(
    xs: &[f32],
    ys: &[f32],
    vis: &[u8],
    s_in: f32,
    min_sag_mult: f32,
    max_rmse_mult: f32,
    min_rad_mult: f32,
    max_rad_mult: f32,
) -> TaubinFitResult {
    let mut out = TaubinFitResult::zeroed();
    let s = s_in.max(0.25);
    let m = xs.len().min(ys.len()).min(vis.len());
    if m < 7 {
        return out;
    }

    let (smooth, span, _path_len) = check_smooth_arc_traversal_vis(xs, ys, vis, s);
    out.span_px = span;
    if !smooth {
        return out;
    }
    out.smooth_arc = 1;

    let mut mx = 0.0f64;
    let mut my = 0.0f64;
    let mut nv = 0usize;
    for i in 0..m {
        if vis[i] != 0 && xs[i].is_finite() && ys[i].is_finite() {
            mx += xs[i] as f64;
            my += ys[i] as f64;
            nv += 1;
        }
    }
    if nv < 7 {
        return out;
    }
    let inv_n = 1.0f64 / (nv as f64);
    mx *= inv_n;
    my *= inv_n;
    out.mean_y = my as f32;

    let mut s_xx = 0.0f64;
    let mut s_yy = 0.0f64;
    let mut s_xy = 0.0f64;
    let mut s_xz = 0.0f64;
    let mut s_yz = 0.0f64;
    let mut s_zz = 0.0f64;
    for i in 0..m {
        if vis[i] != 0 && xs[i].is_finite() && ys[i].is_finite() {
            let ux = (xs[i] as f64) - mx;
            let vy = (ys[i] as f64) - my;
            let zz = ux * ux + vy * vy;
            s_xx += ux * ux;
            s_yy += vy * vy;
            s_xy += ux * vy;
            s_xz += ux * zz;
            s_yz += vy * zz;
            s_zz += zz * zz;
        }
    }
    let l2 = (s_xx + s_yy) * inv_n;
    if l2 <= (1e-6 * (s as f64) * (s as f64)) {
        return out;
    }
    let l_scale = wasm_sqrtd(l2);
    let inv_l = 1.0 / l_scale;
    let inv_l2 = 1.0 / l2;
    let inv_n_l2 = inv_n * inv_l2;
    let inv_n_l3 = inv_n_l2 * inv_l;
    let inv_n_l4 = inv_n_l2 * inv_l2;
    let mxx = s_xx * inv_n_l2;
    let myy = s_yy * inv_n_l2;
    let mxy = s_xy * inv_n_l2;
    let mxz = s_xz * inv_n_l3;
    let myz = s_yz * inv_n_l3;
    let mzz = s_zz * inv_n_l4;
    let zbar = mxx + myy;

    if zbar < 1e-12 {
        return out;
    }
    let czz = (mzz - zbar * zbar).max(0.0);
    let sz = 2.0 * wasm_sqrtd(zbar);

    let sym = [
        [mxx, mxy, mxz / sz],
        [mxy, myy, myz / sz],
        [mxz / sz, myz / sz, czz / (4.0 * zbar)],
    ];
    let (_lam0, lam1, q) = jacobi_3x3_min_eig_f64(sym);
    if lam1 < 1e-6 {
        return out;
    }

    let b_coef = q[0];
    let c_coef = q[1];
    let a_coef = q[2] / sz;
    let kappa_d = (2.0 * a_coef.abs()) * inv_l;
    let kappa = (kappa_d as f32).clamp(0.0, 1.0 / (4.0 * s.max(0.5)));
    out.curvature_inv_px = kappa;

    let min_rad = min_rad_mult * s;
    let max_rad = max_rad_mult * s;
    if kappa_d <= (1.0 / (max_rad as f64)) || a_coef.abs() < 1e-9 {
        return out;
    }

    let rad_d = 1.0 / kappa_d;
    let rad = rad_d as f32;
    let xc_d = mx - (b_coef * l_scale) / (2.0 * a_coef);
    let yc_d = my - (c_coef * l_scale) / (2.0 * a_coef);
    if !xc_d.is_finite() || !yc_d.is_finite() || !rad.is_finite() {
        return out;
    }

    let sag = (span * span) / (8.0 * rad.max(1.0));
    out.sagitta_px = sag;
    if rad < min_rad || rad > max_rad || sag < min_sag_mult * s {
        return out;
    }

    let mut sum_er2 = 0.0f64;
    for i in 0..m {
        if vis[i] != 0 && xs[i].is_finite() && ys[i].is_finite() {
            let dx = (xs[i] as f64) - xc_d;
            let dy = (ys[i] as f64) - yc_d;
            let er = wasm_sqrtd(dx * dx + dy * dy) - rad_d;
            sum_er2 += er * er;
        }
    }
    let rmse = wasm_sqrtd(sum_er2 * inv_n) as f32;
    out.rmse_px = rmse;
    if !rmse.is_finite() || rmse > max_rmse_mult * s {
        return out;
    }

    out.xc = xc_d as f32;
    out.yc = yc_d as f32;
    out.radius = rad;
    out.valid_circle = 1;

    let det_k = mxx * myy - mxy * mxy;
    let tr_k = mxx + myy;
    if det_k > 1e-5 * tr_k * tr_k {
        let bu = 0.5 * mxz;
        let bv = 0.5 * myz;
        let uc_k = (myy * bu - mxy * bv) / det_k;
        let vc_k = (mxx * bv - mxy * bu) / det_k;
        let xc_k = mx + uc_k * l_scale;
        let yc_k = my + vc_k * l_scale;
        if xc_k.is_finite() && yc_k.is_finite() {
            let dcx = xc_d - xc_k;
            let dcy = yc_d - yc_k;
            out.taubin_vs_kasa_delta_px = wasm_sqrtd(dcx * dcx + dcy * dcy) as f32;
        }
    }

    out
}

pub fn fit_taubin_circle_subpx(
    xs: &[f32],
    ys: &[f32],
    vis: &[u8],
    s: f32,
    max_rad_mult: f32,
) -> TaubinFitResult {
    fit_taubin_circle_subpx_core(xs, ys, vis, s, 1.4, 2.2, 18.0, max_rad_mult)
}

pub fn eval_pendulum_pivot_gate(
    fit: &TaubinFitResult,
    phys_regime: u8,
    vis_ratio: f32,
    vert_sep_px: f32,
    dx_span_px: f32,
    dy_span_px: f32,
    near_plumb: bool,
    tethered_origin: bool,
    osc_verified: bool,
    s: f32,
) -> bool {
    if fit.valid_circle == 0 || fit.smooth_arc == 0 {
        return false;
    }
    if phys_regime == 8 || phys_regime == 1 || phys_regime == 2 || phys_regime == 3 || phys_regime == 4 || phys_regime == 7 {
        return false;
    }
    if !vis_ratio.is_finite() || vis_ratio < 0.85 {
        return false;
    }
    let rad = fit.radius;
    let sc = s.max(0.25);
    if !rad.is_finite() || rad > 265.0 * sc || !fit.rmse_px.is_finite() || fit.rmse_px > 1.5 * sc {
        return false;
    }
    let plumb_geom = tethered_origin
        && dx_span_px >= 1.05 * dy_span_px
        && vert_sep_px >= (14.0 * sc).max(0.42 * rad)
        && near_plumb;
    plumb_geom && osc_verified
}

pub fn select_ensemble_taubin_metrics(
    fits: &[TaubinFitResult; 16],
    active: &[u8; 16],
    confirmed: &[u8; 16],
    is_ballistic: &[u8; 16],
    piv_valid: &[u8; 16],
    phys_regime: &[u8; 16],
    span_px: &[f32; 16],
    s: f32,
) -> (f32, f32) {
    let sc = s.max(0.25);
    let mut best_tier = 0u8;
    let mut best_score = -1.0f32;
    let mut best_idx: Option<usize> = None;
    let mut max_smooth_kappa = 0.0f32;

    for m in 0..MAX_TARGETS {
        if active[m] == 0 || confirmed[m] == 0 || is_ballistic[m] != 0 {
            continue;
        }
        let fit = &fits[m];
        if fit.smooth_arc != 0 && fit.curvature_inv_px > max_smooth_kappa {
            max_smooth_kappa = fit.curvature_inv_px;
        }
        if fit.valid_circle == 0 {
            continue;
        }
        let tier = if piv_valid[m] != 0 || phys_regime[m] == PHYS_REGIME_PENDULUM { 2u8 } else { 1u8 };
        let sp = if span_px[m] > 0.0 { span_px[m] } else { fit.span_px };
        let score = sp / fit.rmse_px.max(0.25 * sc);
        if tier > best_tier || (tier == best_tier && score > best_score) {
            best_tier = tier;
            best_score = score;
            best_idx = Some(m);
        }
    }

    if let Some(m) = best_idx {
        (fits[m].taubin_vs_kasa_delta_px, fits[m].curvature_inv_px)
    } else {
        (0.0, max_smooth_kappa)
    }
}

/// Exact Gaussian Arithmetic-Geometric Mean ratio `1.0 / agm(1.0, cos(theta_0 / 2))`
/// equal to `T(theta_0) / T_0 = (2 / pi) * K(sin(theta_0 / 2))`.
/// Computed purely from `cos_theta0 = r_apex . e_eq` via half-angle identity `cos(t/2) = sqrt((1 + cos_t)/2)`.
#[inline]
pub fn elliptic_period_agm_factor_from_cos(cos_theta0: f64) -> f64 {
    let c = cos_theta0.clamp(-0.995, 1.0);
    let cos_half = wasm_sqrtd(0.5 * (1.0 + c)).clamp(0.05, 1.0);
    let mut a = 1.0f64;
    let mut b = cos_half;
    for _ in 0..8 {
        if (a - b).abs() <= 1e-13 * a {
            break;
        }
        let next_a = 0.5 * (a + b);
        let next_b = wasm_sqrtd(a * b);
        a = next_a;
        b = next_b;
    }
    1.0 / a.max(1e-6)
}

/// Estimates local gravity `g_eff` (px/s^2) along a hanging pendulum arc using first-order
/// energy conservation `0.5 * v^2 = g_eff * y + const` with 5-point Savitzky-Golay
/// velocity and position smoothing (zero noisy 2nd derivatives and reduced EIV bias).
/// Returns `Some(g_eff_px_s2)` if R^2 >= 0.72 and positive physical gravity.

pub fn estimate_pendulum_gravity_from_arc(
    xs: &[f32],
    ys: &[f32],
    ts: &[f64],
    vis: &[u8],
    n: usize,
    s: f32,
) -> Option<f32> {
    let m_len = n.min(xs.len()).min(ys.len()).min(ts.len()).min(vis.len());
    if m_len < 10 {
        return None;
    }
    let m = m_len - 4;
    let mut sum_y = 0.0f64;
    let mut sum_e = 0.0f64;
    let mut min_y = 1e9f64;
    let mut max_y = -1e9f64;
    let mut valid_m = 0usize;
    for i in 2..(m_len - 2) {
        if vis[i - 2] == 0 || vis[i - 1] == 0 || vis[i] == 0 || vis[i + 1] == 0 || vis[i + 2] == 0 {
            continue;
        }
        if !xs[i - 2].is_finite() || !xs[i - 1].is_finite() || !xs[i + 1].is_finite() || !xs[i + 2].is_finite()
            || !ys[i - 2].is_finite() || !ys[i - 1].is_finite() || !ys[i].is_finite() || !ys[i + 1].is_finite() || !ys[i + 2].is_finite()
        {
            continue;
        }
        let dt_span = ts[i + 2] - ts[i - 2];
        if !dt_span.is_finite() || dt_span <= 1e-3 || dt_span > 0.5 {
            continue;
        }
        let dt = 0.25 * dt_span;
        let denom = 10.0 * dt;
        let vx = ((-2.0 * (xs[i - 2] as f64) - (xs[i - 1] as f64)
            + (xs[i + 1] as f64) + 2.0 * (xs[i + 2] as f64))) / denom;
        let vy = ((-2.0 * (ys[i - 2] as f64) - (ys[i - 1] as f64)
            + (ys[i + 1] as f64) + 2.0 * (ys[i + 2] as f64))) / denom;
        let ke = 0.5 * (vx * vx + vy * vy);
        let y = (-3.0 * (ys[i - 2] as f64) + 12.0 * (ys[i - 1] as f64)
            + 17.0 * (ys[i] as f64) + 12.0 * (ys[i + 1] as f64) - 3.0 * (ys[i + 2] as f64)) / 35.0;
        sum_y += y;
        sum_e += ke;
        valid_m += 1;
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }
    let sc = s.max(0.25) as f64;
    if valid_m < 6.min(m) || (max_y - min_y) < (4.0 * sc) {
        return None;
    }
    let mf = valid_m as f64;
    let my = sum_y / mf;
    let me = sum_e / mf;
    let mut s_yy = 0.0f64;
    let mut s_ye = 0.0f64;
    let mut s_ee = 0.0f64;
    for i in 2..(m_len - 2) {
        if vis[i - 2] == 0 || vis[i - 1] == 0 || vis[i] == 0 || vis[i + 1] == 0 || vis[i + 2] == 0 {
            continue;
        }
        if !xs[i - 2].is_finite() || !xs[i - 1].is_finite() || !xs[i + 1].is_finite() || !xs[i + 2].is_finite()
            || !ys[i - 2].is_finite() || !ys[i - 1].is_finite() || !ys[i].is_finite() || !ys[i + 1].is_finite() || !ys[i + 2].is_finite()
        {
            continue;
        }
        let dt_span = ts[i + 2] - ts[i - 2];
        if !dt_span.is_finite() || dt_span <= 1e-3 || dt_span > 0.5 {
            continue;
        }
        let dt = 0.25 * dt_span;
        let denom = 10.0 * dt;
        let vx = ((-2.0 * (xs[i - 2] as f64) - (xs[i - 1] as f64)
            + (xs[i + 1] as f64) + 2.0 * (xs[i + 2] as f64))) / denom;
        let vy = ((-2.0 * (ys[i - 2] as f64) - (ys[i - 1] as f64)
            + (ys[i + 1] as f64) + 2.0 * (ys[i + 2] as f64))) / denom;
        let ke = 0.5 * (vx * vx + vy * vy);
        let y = (-3.0 * (ys[i - 2] as f64) + 12.0 * (ys[i - 1] as f64)
            + 17.0 * (ys[i] as f64) + 12.0 * (ys[i + 1] as f64) - 3.0 * (ys[i + 2] as f64)) / 35.0;
        let de = ke - me;
        let dy = y - my;
        s_yy += dy * dy;
        s_ye += dy * de;
        s_ee += de * de;
    }
    if s_yy < 1e-6 || s_ee < 1e-6 {
        return None;
    }
    let g_est = s_ye / s_yy;
    let r2 = (s_ye * s_ye) / (s_yy * s_ee);
    if g_est.is_finite() && r2 >= 0.72 && g_est > (15.0 * sc) && g_est < (25000.0 * sc) {
        Some(g_est as f32)
    } else {
        None
    }
}

pub fn compute_elliptic_period_s(
    xc: f32,
    yc: f32,
    rad: f32,
    xs: &[f32],
    ys: &[f32],
    n: usize,
    seg_peak_cos: &mut f32,
    g_eff: f32,
    t_obs_s: f32,
) -> f32 {
    let m = n.min(xs.len()).min(ys.len());
    if m == 0 || !rad.is_finite() || rad <= 1.0 {
        return if t_obs_s.is_finite() { t_obs_s.clamp(0.0, 30.0) } else { 0.0 };
    }
    let win_start = m.saturating_sub(12);
    let mut sum_dy = 0.0f64;
    for i in win_start..m {
        if ys[i].is_finite() {
            sum_dy += (ys[i] - yc) as f64;
        }
    }
    let eq_sign_y = if sum_dy >= 0.0 { 1.0f64 } else { -1.0f64 };

    let mut min_cos_theta = (*seg_peak_cos as f64).clamp(-0.995, 1.0);
    for i in win_start..m {
        if !xs[i].is_finite() || !ys[i].is_finite() {
            continue;
        }
        let rx = (xs[i] - xc) as f64;
        let ry = (ys[i] - yc) as f64;
        let rnorm = wasm_sqrtd((rx * rx + ry * ry).max(1.0));
        if (rnorm - (rad as f64)).abs() <= 0.12 * (rad as f64) {
            let c = ((ry * eq_sign_y) / rnorm).clamp(-0.995, 1.0);
            if c < min_cos_theta {
                min_cos_theta = c;
            }
        }
    }
    *seg_peak_cos = min_cos_theta as f32;

    if eq_sign_y > 0.0 && g_eff.is_finite() && g_eff > 1.0 {
        let t0 = TWO_PI_64 * wasm_sqrtd((rad as f64) / (g_eff as f64));
        let factor = elliptic_period_agm_factor_from_cos(min_cos_theta);
        return ((t0 * factor) as f32).clamp(0.05, 30.0);
    }
    if t_obs_s.is_finite() && t_obs_s > 0.05 {
        return t_obs_s.clamp(0.05, 30.0);
    }
    0.0
}


#[repr(C)]
#[derive(Clone, Copy)]
pub struct PendulumState {
    pub stable_piv_x: f32,
    pub stable_piv_y: f32,
    pub stable_piv_r: f32,
    pub has_stable_piv: u8,
    pub jump_cand_streak: u8,
    pub jump_cooldown: u8,
    pub jump_code: u8,
    pub jump_hold: u8,
    pub peg_has_ticked: u8,
    pub res_head: u8,
    pub res_count: u8,
    pub jump_cand_r: f32,
    pub g_eff_ema: f32,
    pub seg_peak_cos: f32,
    pub last_peg_frame: u32,
    pub res_last_frame: u32,
    // Peak amplitude ring for damping zeta & Q estimation across multiple cycles
    pub peak_t: [f32; 8],
    pub peak_a: [f32; 8],
    pub peak_count: u8,
    pub peak_head: u8,
    pub peak_last_sign: i8,
    pub res_has_last: u8,
    pub peak_cur_max: f32,
    pub peak_cur_t: f32,
    // 64-slot stride-2 chronological world-space chaos reservoir (covers 128 video frames)
    pub res_x: [f32; HIST_RING_LEN],
    pub res_y: [f32; HIST_RING_LEN],
    pub res_t: [f64; HIST_RING_LEN],
    pub res_vis: [u8; HIST_RING_LEN],
}

impl PendulumState {
    pub const fn zeroed() -> Self {
        Self {
            stable_piv_x: 0.0,
            stable_piv_y: 0.0,
            stable_piv_r: 0.0,
            has_stable_piv: 0,
            jump_cand_streak: 0,
            jump_cooldown: 0,
            jump_code: 0,
            jump_hold: 0,
            peg_has_ticked: 0,
            res_head: 0,
            res_count: 0,
            jump_cand_r: 0.0,
            g_eff_ema: 0.0,
            seg_peak_cos: 1.0,
            last_peg_frame: 0,
            res_last_frame: 0,
            peak_t: [0.0; 8],
            peak_a: [0.0; 8],
            peak_count: 0,
            peak_head: 0,
            peak_last_sign: 0,
            res_has_last: 0,
            peak_cur_max: 0.0,
            peak_cur_t: 0.0,
            res_x: [0.0; HIST_RING_LEN],
            res_y: [0.0; HIST_RING_LEN],
            res_t: [0.0; HIST_RING_LEN],
            res_vis: [0; HIST_RING_LEN],
        }
    }
    #[allow(dead_code)]
    pub const fn zero() -> Self {
        Self::zeroed()
    }
    #[allow(dead_code)]
    pub const fn new() -> Self {
        Self::zeroed()
    }
    /// In-place reset avoids stack allocation of ~1.2 KiB.
    pub fn reset(&mut self) {
        self.stable_piv_x = 0.0;
        self.stable_piv_y = 0.0;
        self.stable_piv_r = 0.0;
        self.has_stable_piv = 0;
        self.jump_cand_streak = 0;
        self.jump_cooldown = 0;
        self.jump_code = 0;
        self.jump_hold = 0;
        self.peg_has_ticked = 0;
        self.res_head = 0;
        self.res_count = 0;
        self.jump_cand_r = 0.0;
        self.g_eff_ema = 0.0;
        self.seg_peak_cos = 1.0;
        self.last_peg_frame = 0;
        self.res_last_frame = 0;
        self.peak_t.fill(0.0);
        self.peak_a.fill(0.0);
        self.peak_count = 0;
        self.peak_head = 0;
        self.peak_last_sign = 0;
        self.res_has_last = 0;
        self.peak_cur_max = 0.0;
        self.peak_cur_t = 0.0;
        self.res_x.fill(0.0);
        self.res_y.fill(0.0);
        self.res_t.fill(0.0);
        self.res_vis.fill(0);
    }
}

#[inline(never)]
pub fn update_galileo_peg_detector(
    st: &mut PendulumState,
    xs: &[f32],
    ys: &[f32],
    ts: &[f64],
    vis: &[u8],
    n: usize,
    frame_idx: u32,
    s: f32,
    plumb_ok: bool,
    bx: f32,
    by: f32,
) -> (TaubinFitResult, bool) {
    let dt_f: u8 = if st.peg_has_ticked == 0 {
        1
    } else if frame_idx > st.last_peg_frame {
        let diff = frame_idx - st.last_peg_frame;
        if diff > 255 { 255 } else { diff as u8 }
    } else {
        1
    };
    st.peg_has_ticked = 1;
    st.last_peg_frame = frame_idx;
    st.jump_cooldown = st.jump_cooldown.saturating_sub(dt_f);
    if st.jump_hold > 0 {
        st.jump_hold = st.jump_hold.saturating_sub(dt_f);
        if st.jump_hold == 0 && st.jump_code == JUMP_GALILEO_PEG {
            st.jump_code = JUMP_NONE;
        }
    }

    let sc = s.max(0.25);
    let mut m = n.min(xs.len()).min(ys.len()).min(vis.len());
    if !ts.is_empty() {
        m = m.min(ts.len());
    }
    let rec_len = m.min(16);
    let rec_off = m.saturating_sub(rec_len);
    let rec_fit = fit_taubin_circle_subpx_core(
        &xs[rec_off..m],
        &ys[rec_off..m],
        &vis[rec_off..m],
        sc,
        1.4,
        2.0,
        22.0,
        320.0,
    );
    let mut best_fit = rec_fit;
    if st.has_stable_piv == 0 && best_fit.valid_circle == 0 && m > rec_len {
        let wide_len = m.min(24);
        let wide_off = m - wide_len;
        let wide_fit = fit_taubin_circle_subpx_core(
            &xs[wide_off..m],
            &ys[wide_off..m],
            &vis[wide_off..m],
            sc,
            1.2,
            2.2,
            24.0,
            320.0,
        );
        if wide_fit.valid_circle != 0 {
            best_fit = wide_fit;
        }
    }

    let mut fired_peg_jump = false;
    // Discrete pixel/chroma 4:2:0 floor keeps SD (s=0.5) robust across decoder filters without altering HD (s=1.0).
    let init_rmse_lim = (1.2 * sc).max(0.85);
    let jump_rmse_lim = (1.15 * sc).max(0.82);
    if best_fit.valid_circle != 0 && plumb_ok {
        let r_new = best_fit.radius;
        if st.has_stable_piv == 0 {
            if st.jump_cand_r > 1.0 && (r_new - st.jump_cand_r).abs() <= 0.08 * st.jump_cand_r {
                st.jump_cand_streak = st.jump_cand_streak.saturating_add(1);
                st.jump_cand_r = 0.65 * st.jump_cand_r + 0.35 * r_new;
                if st.jump_cand_streak >= 3 && best_fit.rmse_px <= init_rmse_lim {
                    st.stable_piv_x = best_fit.xc;
                    st.stable_piv_y = best_fit.yc;
                    st.stable_piv_r = r_new;
                    st.has_stable_piv = 1;
                    st.jump_cand_streak = 0;
                    st.jump_cand_r = 0.0;
                    st.seg_peak_cos = 1.0;
                }
            } else if best_fit.rmse_px <= init_rmse_lim {
                st.jump_cand_r = r_new;
                st.jump_cand_streak = 1;
            }
        } else {
            let r_old = st.stable_piv_r;
            let ratio = r_old.max(r_new) / r_old.min(r_new).max(1.0);
            let diff_r = (r_new - r_old).abs();

            let v1x = st.stable_piv_x - bx;
            let v1y = st.stable_piv_y - by;
            let v2x = best_fit.xc - bx;
            let v2y = best_fit.yc - by;
            let d1 = wasm_sqrtf((v1x * v1x + v1y * v1y).max(1.0));
            let d2 = wasm_sqrtf((v2x * v2x + v2y * v2y).max(1.0));
            let cos_align = (v1x * v2x + v1y * v2y) / (d1 * d2);

            if ratio < 1.25 {
                st.stable_piv_x = 0.80 * st.stable_piv_x + 0.20 * best_fit.xc;
                st.stable_piv_y = 0.80 * st.stable_piv_y + 0.20 * best_fit.yc;
                st.stable_piv_r = 0.80 * st.stable_piv_r + 0.20 * r_new;
                st.jump_cand_streak = 0;
                st.jump_cand_r = 0.0;
            } else if ratio >= 1.38
                && diff_r >= 24.0 * sc
                && best_fit.rmse_px <= jump_rmse_lim
                && cos_align >= 0.70
            {
                if st.jump_cand_r > 1.0 && (r_new - st.jump_cand_r).abs() <= 0.06 * r_new {
                    st.jump_cand_streak = st.jump_cand_streak.saturating_add(1);
                    st.jump_cand_r = 0.55 * st.jump_cand_r + 0.45 * r_new;
                } else {
                    st.jump_cand_r = r_new;
                    st.jump_cand_streak = 1;
                }
                if st.jump_cand_streak >= 3 && st.jump_cooldown == 0 {
                    fired_peg_jump = true;
                    st.jump_code = JUMP_GALILEO_PEG;
                    st.jump_hold = 15;
                    st.jump_cooldown = 18;
                    st.stable_piv_x = best_fit.xc;
                    st.stable_piv_y = best_fit.yc;
                    st.stable_piv_r = r_new;
                    st.jump_cand_streak = 0;
                    st.jump_cand_r = 0.0;
                    st.seg_peak_cos = 1.0;
                }
            } else {
                st.jump_cand_streak = 0;
                st.jump_cand_r = 0.0;
            }
        }
    }

    (best_fit, fired_peg_jump)
}

pub fn update_damping_zeta_q(
    st: &mut PendulumState,
    q: f32,
    t_sec: f32,
    omega_d: f32,
    min_amp: f32,
    osc_active: bool,
) -> (f32, f32) {
    if !osc_active || !q.is_finite() || !t_sec.is_finite() || !omega_d.is_finite() || omega_d <= 0.3 {
        return (0.0, 0.0);
    }
    let sign: i8 = if q > min_amp * 0.25 {
        1
    } else if q < -min_amp * 0.25 {
        -1
    } else {
        0
    };
    let aq = q.abs();

    if st.peak_last_sign == 0 || sign == st.peak_last_sign {
        if aq > st.peak_cur_max {
            st.peak_cur_max = aq;
            st.peak_cur_t = t_sec;
        }
        if sign != 0 && st.peak_last_sign == 0 {
            st.peak_last_sign = sign;
            st.peak_cur_max = aq;
            st.peak_cur_t = t_sec;
        }
    } else if sign != 0 && sign != st.peak_last_sign {
        if st.peak_cur_max >= min_amp {
            let slot = (st.peak_head as usize) & 7;
            st.peak_t[slot] = st.peak_cur_t;
            st.peak_a[slot] = st.peak_cur_max;
            st.peak_head = st.peak_head.wrapping_add(1);
            if st.peak_count < 8 {
                st.peak_count += 1;
            }
        }
        st.peak_last_sign = sign;
        st.peak_cur_max = aq;
        st.peak_cur_t = t_sec;
    }

    if st.peak_count < 3 {
        return (0.0, 0.0);
    }

    let cnt = st.peak_count as usize;
    let mut sum_t = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut t0 = 0.0f32;
    for k in 0..cnt {
        let idx = ((st.peak_head as usize).wrapping_sub(cnt).wrapping_add(k)) & 7;
        if k == 0 {
            t0 = st.peak_t[idx];
        }
        sum_t += st.peak_t[idx] - t0;
        sum_y += fast_lnf(st.peak_a[idx].max(1e-3));
    }
    let inv_c = 1.0 / (cnt as f32);
    let mt = sum_t * inv_c;
    let my = sum_y * inv_c;
    let mut s_tt = 0.0f32;
    let mut s_ty = 0.0f32;
    for k in 0..cnt {
        let idx = ((st.peak_head as usize).wrapping_sub(cnt).wrapping_add(k)) & 7;
        let dt = (st.peak_t[idx] - t0) - mt;
        let dy = fast_lnf(st.peak_a[idx].max(1e-3)) - my;
        s_tt += dt * dt;
        s_ty += dt * dy;
    }
    if s_tt <= 1e-4 {
        return (0.0, 0.0);
    }
    let sigma = (-s_ty / s_tt).max(0.0);
    if sigma <= 1e-4 {
        return (0.0, 999.0);
    }
    let zeta = (sigma / wasm_sqrtf(sigma * sigma + omega_d * omega_d)).clamp(0.0, 1.0);
    let q_val = if zeta <= (1.0 / 1998.0) {
        999.0
    } else {
        (0.5 / zeta).clamp(0.0, 999.0)
    };
    (zeta, q_val)
}

#[inline]
pub fn compute_specific_energy_and_ang_mom(
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    g_eff: f32,
    frame_h: f32,
    has_pivot: bool,
    piv_x: f32,
    piv_y: f32,
    piv_r: f32,
    ref_x: f32,
    ref_y: f32,
    osc_active: bool,
    omega_osc: f32,
    q_disp: f32,
    is_ballistic: bool,
) -> (f32, f32) {
    if !x.is_finite() || !y.is_finite() || !vx.is_finite() || !vy.is_finite() {
        return (0.0, 0.0);
    }
    let ke = 0.5 * (vx * vx + vy * vy);
    let pe = if has_pivot && g_eff.is_finite() && g_eff > 1.0 && y > piv_y {
        let y_bottom = piv_y + piv_r;
        g_eff * (y_bottom - y).max(0.0)
    } else if is_ballistic && g_eff.is_finite() && g_eff > 1.0 {
        g_eff * (frame_h - y).max(0.0)
    } else if osc_active && omega_osc.is_finite() && omega_osc > 0.3 && q_disp.is_finite() {
        0.5 * omega_osc * omega_osc * q_disp * q_disp
    } else {
        0.0
    };
    let specific_energy = (ke + pe).max(0.0);
    let (ox, oy) = if has_pivot { (piv_x, piv_y) } else { (ref_x, ref_y) };
    let specific_ang_mom = (x - ox) * vy - (y - oy) * vx;
    (
        if specific_energy.is_finite() { specific_energy } else { 0.0 },
        if specific_ang_mom.is_finite() { specific_ang_mom } else { 0.0 },
    )
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LyapunovScratch {
    pub rx: [f32; HIST_RING_LEN],
    pub ry: [f32; HIST_RING_LEN],
    pub rt: [f64; HIST_RING_LEN],
    pub rvis: [u8; HIST_RING_LEN],
    pub zx: [f32; HIST_RING_LEN],
    pub zy: [f32; HIST_RING_LEN],
    pub zvx: [f32; HIST_RING_LEN],
    pub zvy: [f32; HIST_RING_LEN],
    pub steps: [f32; HIST_RING_LEN],
    pub step_sorted: [f32; HIST_RING_LEN],
    pub ok: [u8; HIST_RING_LEN],
}

impl LyapunovScratch {
    pub const fn zeroed() -> Self {
        Self {
            rx: [0.0; HIST_RING_LEN],
            ry: [0.0; HIST_RING_LEN],
            rt: [0.0; HIST_RING_LEN],
            rvis: [0; HIST_RING_LEN],
            zx: [0.0; HIST_RING_LEN],
            zy: [0.0; HIST_RING_LEN],
            zvx: [0.0; HIST_RING_LEN],
            zvy: [0.0; HIST_RING_LEN],
            steps: [0.0; HIST_RING_LEN],
            step_sorted: [0.0; HIST_RING_LEN],
            ok: [0; HIST_RING_LEN],
        }
    }
    #[allow(dead_code)]
    pub const fn zero() -> Self {
        Self::zeroed()
    }
    #[allow(dead_code)]
    pub const fn new() -> Self {
        Self::zeroed()
    }
    pub fn reset(&mut self) {
        self.rx.fill(0.0);
        self.ry.fill(0.0);
        self.rt.fill(0.0);
        self.rvis.fill(0);
        self.zx.fill(0.0);
        self.zy.fill(0.0);
        self.zvx.fill(0.0);
        self.zvy.fill(0.0);
        self.steps.fill(0.0);
        self.step_sorted.fill(0.0);
        self.ok.fill(0);
    }
}

/// Captures every even frame (`(frame_idx & 1) == 0`) into the 64-slot stride-2 chronological reservoir.
#[inline(always)]
pub fn push_chaos_reservoir(st: &mut PendulumState, frame_idx: u32, x: f32, y: f32, t_sec: f64, vis: u8) {
    if !t_sec.is_finite() {
        return;
    }
    if st.res_has_last != 0 {
        if frame_idx < st.res_last_frame {
            st.res_head = 0;
            st.res_count = 0;
            st.res_has_last = 0;
        } else if frame_idx == st.res_last_frame {
            return;
        }
    }
    if (frame_idx & 1) != 0 {
        return;
    }
    let slot = (st.res_head as usize) & HIST_MASK;
    st.res_x[slot] = if x.is_finite() { x } else { 0.0 };
    st.res_y[slot] = if y.is_finite() { y } else { 0.0 };
    st.res_t[slot] = t_sec;
    st.res_vis[slot] = if vis != 0 && x.is_finite() && y.is_finite() { 1 } else { 0 };
    st.res_head = st.res_head.wrapping_add(1);
    if st.res_count < HIST_RING_LEN as u8 {
        st.res_count += 1;
    }
    st.res_last_frame = frame_idx;
    st.res_has_last = 1;
}

/// Core Rosenstein 4D Lyapunov divergence computation reading chronologically unwrapped samples
/// in `scratch.{rx, ry, rt, rvis}[0..n]`.
fn compute_lyapunov_div_rate_scratch(
    n: usize,
    s_in: f32,
    fps_hint: f32,
    scratch: &mut LyapunovScratch,
) -> f32 {
    if n < 32 || n > HIST_RING_LEN {
        return 0.0;
    }
    let s = s_in.max(0.25);

    let mut mx = 0.0f32;
    let mut my = 0.0f32;
    let mut min_x = 1e9f32;
    let mut max_x = -1e9f32;
    let mut min_y = 1e9f32;
    let mut max_y = -1e9f32;
    let mut cnt = 0usize;
    let mut first_idx = n;
    let mut last_idx = 0usize;
    let mut step_cnt = 0usize;
    let mut path_len = 0.0f32;

    for i in 0..n {
        if scratch.rvis[i] != 0 && scratch.rx[i].is_finite() && scratch.ry[i].is_finite() {
            let x = scratch.rx[i];
            let y = scratch.ry[i];
            mx += x;
            my += y;
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
            cnt += 1;
            if first_idx == n {
                first_idx = i;
            }
            if i > 0 && scratch.rvis[i - 1] != 0 {
                let dx = x - scratch.rx[i - 1];
                let dy = y - scratch.ry[i - 1];
                let d = wasm_sqrtf(dx * dx + dy * dy);
                scratch.steps[step_cnt] = d;
                scratch.step_sorted[step_cnt] = d;
                step_cnt += 1;
                path_len += d;
            }
            last_idx = i;
        }
    }
    if cnt < 28 || cnt * 100 < n * 68 || step_cnt < 24 || last_idx <= first_idx + 24 {
        return 0.0;
    }
    mx /= cnt as f32;
    my /= cnt as f32;

    let bw = max_x - min_x;
    let bh = max_y - min_y;
    let bbox_diag = wasm_sqrtf(bw * bw + bh * bh);
    if bbox_diag < 18.0 * s {
        return 0.0;
    }

    for i in 1..step_cnt {
        let key = scratch.step_sorted[i];
        let mut j = i;
        while j > 0 && scratch.step_sorted[j - 1] > key {
            scratch.step_sorted[j] = scratch.step_sorted[j - 1];
            j -= 1;
        }
        scratch.step_sorted[j] = key;
    }
    let med_step = scratch.step_sorted[step_cnt >> 1];
    if med_step < 0.8 * s {
        return 0.0;
    }

    let end_dx = scratch.rx[last_idx] - scratch.rx[first_idx];
    let end_dy = scratch.ry[last_idx] - scratch.ry[first_idx];
    let end_dist = wasm_sqrtf(end_dx * end_dx + end_dy * end_dy);
    if path_len < 2.2 * bbox_diag || path_len < 2.0 * end_dist {
        return 0.0;
    }

    let raw_dt = (scratch.rt[last_idx] - scratch.rt[first_idx]) as f32 / ((last_idx - first_idx) as f32);
    let dt_nom = if raw_dt.is_finite() && raw_dt > 1e-3 {
        raw_dt.clamp(0.004, 0.20)
    } else {
        (2.0 / fps_hint.clamp(10.0, 240.0)).clamp(0.004, 0.20)
    };
    let fps_eff = 1.0 / dt_nom;
    // Physical velocity ceiling (~1270*s px/s) invariant across stride-1/stride-2 and 30/60fps
    let step_cap = 42.0 * s * (dt_nom / 0.033).clamp(0.5, 4.0);

    for k in 1..(step_cnt - 1) {
        let nbr = scratch.steps[k - 1].max(scratch.steps[k + 1]).max(3.0 * s);
        if scratch.steps[k] > 5.0 * nbr || scratch.steps[k] > step_cap {
            return 0.0;
        }
    }

    scratch.ok.fill(0);
    let mut l2_sum = 0.0f32;
    let mut v2_sum = 0.0f32;
    let mut ok_cnt = 0usize;

    for i in 1..(n - 1) {
        if scratch.rvis[i - 1] != 0 && scratch.rvis[i] != 0 && scratch.rvis[i + 1] != 0 {
            let dt = (scratch.rt[i + 1] - scratch.rt[i - 1]) as f32;
            if dt.is_finite() && dt > 1e-3 && dt < 0.50 {
                let vx = (scratch.rx[i + 1] - scratch.rx[i - 1]) / dt;
                let vy = (scratch.ry[i + 1] - scratch.ry[i - 1]) / dt;
                scratch.zx[i] = scratch.rx[i] - mx;
                scratch.zy[i] = scratch.ry[i] - my;
                scratch.zvx[i] = vx;
                scratch.zvy[i] = vy;
                scratch.ok[i] = 1;
                l2_sum += scratch.zx[i] * scratch.zx[i] + scratch.zy[i] * scratch.zy[i];
                v2_sum += vx * vx + vy * vy;
                ok_cnt += 1;
            }
        }
    }
    if ok_cnt < 24 {
        return 0.0;
    }
    let l_rms = wasm_sqrtf(l2_sum / (ok_cnt as f32));
    let v_rms = wasm_sqrtf(v2_sum / (ok_cnt as f32));
    if l_rms < 6.0 * s || v_rms < 10.0 * s {
        return 0.0;
    }
    let inv_l = 1.0 / l_rms;
    let inv_v = 1.0 / v_rms;
    for i in 1..(n - 1) {
        if scratch.ok[i] != 0 {
            scratch.zx[i] *= inv_l;
            scratch.zy[i] *= inv_l;
            scratch.zvx[i] *= inv_v;
            scratch.zvy[i] *= inv_v;
        }
    }

    let k_evol = ((0.45 * fps_eff + 0.5) as usize).clamp(5, 14);
    let theiler = ((0.32 * fps_eff + 0.5) as usize).clamp(5, 16);
    if n <= k_evol + theiler + 4 {
        return 0.0;
    }

    let sigma_z = (1.5 * s * inv_l).max(0.05);
    let sigma_z2 = sigma_z * sigma_z;
    let max_d0_2 = 0.65 * 0.65;

    let mut div_sum = [0.0f32; 16];
    let mut div_cnt = [0u16; 16];
    let mut matched_pairs = 0usize;

    let mut i = 1usize;
    while i + k_evol < n - 1 {
        let mut i_seg_ok = true;
        for k in 0..=k_evol {
            if scratch.ok[i + k] == 0 {
                i_seg_ok = false;
                break;
            }
        }
        let vi_n = wasm_sqrtf(scratch.zvx[i] * scratch.zvx[i] + scratch.zvy[i] * scratch.zvy[i]);
        if !i_seg_ok || vi_n < 0.25 {
            i += 2;
            continue;
        }

        let mut best_j = usize::MAX;
        let mut best_d2 = max_d0_2;
        let mut j = 1usize;
        while j + k_evol < n - 1 {
            if i.abs_diff(j) >= theiler && scratch.ok[j] != 0 {
                let vj_n = wasm_sqrtf(scratch.zvx[j] * scratch.zvx[j] + scratch.zvy[j] * scratch.zvy[j]);
                let vdot = scratch.zvx[i] * scratch.zvx[j] + scratch.zvy[i] * scratch.zvy[j];
                if vj_n >= 0.25 && vdot >= 0.45 * vi_n * vj_n {
                    let dx = scratch.zx[i] - scratch.zx[j];
                    let dy = scratch.zy[i] - scratch.zy[j];
                    let dvx = scratch.zvx[i] - scratch.zvx[j];
                    let dvy = scratch.zvy[i] - scratch.zvy[j];
                    let d2 = dx * dx + dy * dy + dvx * dvx + dvy * dvy;
                    if d2 < best_d2 {
                        let mut j_seg_ok = true;
                        for k in 1..=k_evol {
                            if scratch.ok[j + k] == 0 {
                                j_seg_ok = false;
                                break;
                            }
                        }
                        if j_seg_ok {
                            best_d2 = d2;
                            best_j = j;
                        }
                    }
                }
            }
            j += 1;
        }

        if best_j != usize::MAX {
            matched_pairs += 1;
            for k in 0..=k_evol {
                let dx = scratch.zx[i + k] - scratch.zx[best_j + k];
                let dy = scratch.zy[i + k] - scratch.zy[best_j + k];
                let dvx = scratch.zvx[i + k] - scratch.zvx[best_j + k];
                let dvy = scratch.zvy[i + k] - scratch.zvy[best_j + k];
                let reg_d = wasm_sqrtf(dx * dx + dy * dy + dvx * dvx + dvy * dvy + sigma_z2);
                div_sum[k] += fast_lnf(reg_d);
                div_cnt[k] += 1;
            }
        }
        i += 2;
    }

    if matched_pairs < 5 || div_cnt[0] == 0 || div_cnt[k_evol] == 0 {
        return 0.0;
    }

    let mut y_curve = [0.0f32; 16];
    for k in 0..=k_evol {
        if div_cnt[k] == 0 {
            return 0.0;
        }
        y_curve[k] = div_sum[k] / (div_cnt[k] as f32);
    }

    let mut k_peak = k_evol;
    for k in 4..k_evol {
        if y_curve[k + 1] < y_curve[k] - 0.04 {
            k_peak = k;
            break;
        }
    }

    let grow_from_1 = y_curve[k_peak] - y_curve[1];
    if y_curve[k_peak] < -1.05 || grow_from_1 < 0.18 {
        return 0.0;
    }

    let k_start = 1usize;
    let m_k = k_peak - k_start + 1;
    if m_k < 4 {
        return 0.0;
    }

    let mut sum_t = 0.0f32;
    let mut sum_y = 0.0f32;
    for k in k_start..=k_peak {
        sum_t += (k as f32) * dt_nom;
        sum_y += y_curve[k];
    }
    let inv_m = 1.0 / (m_k as f32);
    let mt = sum_t * inv_m;
    let my = sum_y * inv_m;
    let mut s_tt = 0.0f32;
    let mut s_yy = 0.0f32;
    let mut s_ty = 0.0f32;
    for k in k_start..=k_peak {
        let dt = (k as f32) * dt_nom - mt;
        let dy = y_curve[k] - my;
        s_tt += dt * dt;
        s_yy += dy * dy;
        s_ty += dt * dy;
    }
    if s_tt <= 1e-6 || s_yy <= 1e-6 {
        return 0.0;
    }
    let r2 = (s_ty * s_ty) / (s_tt * s_yy);
    let lam = s_ty / s_tt;
    if !lam.is_finite() || r2 < 0.55 || lam <= 0.15 {
        return 0.0;
    }
    lam.clamp(0.0, 10.0)
}

#[inline(never)]
pub fn compute_lyapunov_div_rate_from_reservoir(
    st: &PendulumState,
    scratch: &mut LyapunovScratch,
    phys_regime: u8,
    is_ballistic: bool,
    coasting: bool,
    confirmed: bool,
    interior: bool,
    s: f32,
    fps_hint: f32,
) -> f32 {
    if !interior || !confirmed || coasting || is_ballistic {
        return 0.0;
    }
    if phys_regime != PHYS_REGIME_PENDULUM
        && phys_regime != PHYS_REGIME_MICRO_TREMOR
        && phys_regime != PHYS_REGIME_CURVED_CTRV
    {
        return 0.0;
    }
    let n = (st.res_count as usize).min(HIST_RING_LEN);
    if n < 32 {
        return 0.0;
    }
    let start = (st.res_head as usize).wrapping_sub(n) & HIST_MASK;
    for k in 0..n {
        let idx = (start + k) & HIST_MASK;
        scratch.rx[k] = st.res_x[idx];
        scratch.ry[k] = st.res_y[idx];
        scratch.rt[k] = st.res_t[idx];
        scratch.rvis[k] = st.res_vis[idx];
    }
    compute_lyapunov_div_rate_scratch(n, s, fps_hint, scratch)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OscSample {
    pub active: u8,
    pub confirmed: u8,
    pub coasting: u8,
    pub phase_valid: u8,
    pub freq_valid: u8,
    pub is_ballistic: u8,
    pub strike_fired: u8,
    pub phys_regime: u8,
    pub phase_rad: f32,
    pub freq_hz: f32,
    pub coherence: f32,
    pub amp_px: f32,
}

impl OscSample {
    pub const fn zeroed() -> Self {
        Self {
            active: 0,
            confirmed: 0,
            coasting: 0,
            phase_valid: 0,
            freq_valid: 0,
            is_ballistic: 0,
            strike_fired: 0,
            phys_regime: 0,
            phase_rad: 0.0,
            freq_hz: 0.0,
            coherence: 0.0,
            amp_px: 0.0,
        }
    }
}

pub fn compute_kuramoto_order(samples: &[OscSample], n_samples: usize, s: f32) -> (f32, f32, f32) {
    let sc = s.max(0.25);
    let m = n_samples.min(samples.len()).min(MAX_TARGETS);
    let mut sum_c = 0.0f32;
    let mut sum_s = 0.0f32;
    let mut osc_cnt = 0u32;

    for i in 0..m {
        let it = &samples[i];
        let is_osc_regime = it.phys_regime == PHYS_REGIME_PENDULUM || it.phys_regime == PHYS_REGIME_MICRO_TREMOR;
        if it.active != 0
            && it.confirmed != 0
            && it.coasting == 0
            && it.phase_valid != 0
            && it.freq_valid != 0
            && it.is_ballistic == 0
            && it.strike_fired == 0
            && is_osc_regime
            && it.coherence >= 0.45
            && it.amp_px >= 0.38 * sc
            && it.freq_hz >= 0.20
            && it.freq_hz <= 20.0
            && it.phase_rad.is_finite()
        {
            sum_c += fast_cosf(it.phase_rad);
            sum_s += fast_sinf(it.phase_rad);
            osc_cnt += 1;
        }
    }

    if osc_cnt == 0 {
        return (0.0, 0.0, 0.0);
    }
    if osc_cnt == 1 {
        return (0.0, 0.0, 1.0);
    }
    let inv_n = 1.0 / (osc_cnt as f32);
    let mc = sum_c * inv_n;
    let ms = sum_s * inv_n;
    let r = wasm_sqrtf(mc * mc + ms * ms).clamp(0.0, 1.0);
    let psi = fast_atan2f(ms, mc);
    (r, psi, osc_cnt as f32)
}


// =============================================================================
// Section C: Two-Body Causal Collision & Conservation Analyzer
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollisionPairEvent {
    pub valid: u8,
    pub chain_transfer: u8, // 0 = direct contact, 1 = collinear multi-sphere chain (Newton's cradle)
    pub slot_a: u8,
    pub slot_b: u8,
    pub tid_a: u32,
    pub tid_b: u32,
    pub event_frame: u32,
    pub tau_c_sec: f32,
    pub sep_c_px: f32,
    pub nx: f32,
    pub ny: f32,
    pub normal_restitution_e: f32,
    pub mass_ratio_b_over_a: f32,
    pub mom_resid: f32,
    pub energy_retention: f32,
    pub approach_speed_px_s: f32,
    pub dv_a_mag: f32,
    pub dv_b_mag: f32,
    pub score: f32,
}

impl CollisionPairEvent {
    pub const fn zeroed() -> Self {
        Self {
            valid: 0,
            chain_transfer: 0,
            slot_a: 0,
            slot_b: 0,
            tid_a: 0,
            tid_b: 0,
            event_frame: 0,
            tau_c_sec: 0.0,
            sep_c_px: 0.0,
            nx: 0.0,
            ny: 0.0,
            normal_restitution_e: 0.0,
            mass_ratio_b_over_a: 0.0,
            mom_resid: 0.0,
            energy_retention: 0.0,
            approach_speed_px_s: 0.0,
            dv_a_mag: 0.0,
            dv_b_mag: 0.0,
            score: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TargetCollisionOut {
    pub active_collision: u8,      // 1 during pulse window around f_c
    pub phys_regime_override: u8,  // PHYS_REGIME_IMPULSIVE (7) around impact, else 0
    pub constraint_jump_code: u8,  // CONSTRAINT_JUMP_TWO_BODY (4) or 0
    pub _pad: u8,
    pub normal_restitution_e: f32, // latched e_n from last collision involving this target
    pub partner_tid: u32,
    pub last_collision_frame: u32,
}

impl TargetCollisionOut {
    pub const fn zeroed() -> Self {
        Self {
            active_collision: 0,
            phys_regime_override: 0,
            constraint_jump_code: 0,
            _pad: 0,
            normal_restitution_e: 0.0,
            partner_tid: 0,
            last_collision_frame: 0,
        }
    }
}

pub const CONSERVATION_TELEM_LEN: usize = 32;
pub const CONSERVATION_INPUT_LEN: usize = 32;

pub const OCCL_STATUS_NONE: u32 = 0;
pub const OCCL_STATUS_UNIQUE: u32 = 1;
pub const OCCL_STATUS_AMBIGUOUS: u32 = 2;
pub const OCCL_STATUS_VIOLATED: u32 = 3;

pub const HYP_PASSTHROUGH_KEEP: u32 = 1 << 0;
pub const HYP_PASSTHROUGH_SWAP: u32 = 1 << 1;
pub const HYP_COLLISION_KEEP: u32 = 1 << 2;
pub const HYP_COLLISION_SWAP: u32 = 1 << 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlotObsHistory {
    pub t_sec: [f64; 8],
    pub wx: [f32; 8],
    pub wy: [f32; 8],
    pub tid: u32,
    pub orig_birth: u32,
    pub last_vis_frame: u32,
    pub prev_coasting: u8,
    pub vis_streak: u8,
    pub head: u8,
    pub active: u8,
}

impl SlotObsHistory {
    pub const fn zeroed() -> Self {
        Self {
            t_sec: [0.0; 8],
            wx: [0.0; 8],
            wy: [0.0; 8],
            tid: 0,
            orig_birth: 0,
            last_vis_frame: 0,
            prev_coasting: 0,
            vis_streak: 0,
            head: 0,
            active: 0,
        }
    }

    pub fn reset_in_place(&mut self) {
        self.t_sec.fill(0.0);
        self.wx.fill(0.0);
        self.wy.fill(0.0);
        self.tid = 0;
        self.orig_birth = 0;
        self.last_vis_frame = 0;
        self.prev_coasting = 0;
        self.vis_streak = 0;
        self.head = 0;
        self.active = 0;
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OcclPairTrackState {
    pub pre_t_sec: f64,
    pub tid_a: u32,
    pub tid_b: u32,
    pub orig_birth_a: u32,
    pub orig_birth_b: u32,
    pub start_frame: u32,
    pub last_seen_frame: u32,
    pub occl_start_frame: u32,
    pub cooldown_until: u32,
    pub pre_x1: f32,
    pub pre_y1: f32,
    pub pre_vx1: f32,
    pub pre_vy1: f32,
    pub pre_x2: f32,
    pub pre_y2: f32,
    pub pre_vx2: f32,
    pub pre_vy2: f32,
    pub pre_rmse: f32,
    pub slot_a: u8,
    pub slot_b: u8,
    pub active: u8,
    pub has_pre: u8,
    pub both_occluded: u8,
    pub occl_seen_mask: u8,
    pub max_coast_a: u8,
    pub max_coast_b: u8,
    pub co_coast_frames: u8,
    pub _pad: [u8; 3],
}

impl OcclPairTrackState {
    pub const fn zeroed() -> Self {
        Self {
            pre_t_sec: 0.0,
            tid_a: 0,
            tid_b: 0,
            orig_birth_a: 0,
            orig_birth_b: 0,
            start_frame: 0,
            last_seen_frame: 0,
            occl_start_frame: 0,
            cooldown_until: 0,
            pre_x1: 0.0,
            pre_y1: 0.0,
            pre_vx1: 0.0,
            pre_vy1: 0.0,
            pre_x2: 0.0,
            pre_y2: 0.0,
            pre_vx2: 0.0,
            pre_vy2: 0.0,
            pre_rmse: 0.0,
            slot_a: 0,
            slot_b: 0,
            active: 0,
            has_pre: 0,
            both_occluded: 0,
            occl_seen_mask: 0,
            max_coast_a: 0,
            max_coast_b: 0,
            co_coast_frames: 0,
            _pad: [0; 3],
        }
    }

    pub fn reset_in_place(&mut self) {
        self.pre_t_sec = 0.0;
        self.tid_a = 0;
        self.tid_b = 0;
        self.orig_birth_a = 0;
        self.orig_birth_b = 0;
        self.start_frame = 0;
        self.last_seen_frame = 0;
        self.occl_start_frame = 0;
        self.cooldown_until = 0;
        self.pre_x1 = 0.0;
        self.pre_y1 = 0.0;
        self.pre_vx1 = 0.0;
        self.pre_vy1 = 0.0;
        self.pre_x2 = 0.0;
        self.pre_y2 = 0.0;
        self.pre_vx2 = 0.0;
        self.pre_vy2 = 0.0;
        self.pre_rmse = 0.0;
        self.slot_a = 0;
        self.slot_b = 0;
        self.active = 0;
        self.has_pre = 0;
        self.both_occluded = 0;
        self.occl_seen_mask = 0;
        self.max_coast_a = 0;
        self.max_coast_b = 0;
        self.co_coast_frames = 0;
        self._pad = [0; 3];
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OccludedConservationState {
    pub telem: [f32; CONSERVATION_TELEM_LEN],
    pub input: [f32; CONSERVATION_INPUT_LEN],
    pub track_pairs: [OcclPairTrackState; 8],
    pub slot_obs: [SlotObsHistory; MAX_TARGETS],
    pub live_event_seq: u32,
    pub last_frame_idx: u32,
    pub last_discont_seen: u32,
    pub last_continuity_epoch: u32,
    pub has_last_frame: u8,
    pub _pad: [u8; 3],
}

const _: () = assert!(core::mem::size_of::<OccludedConservationState>() <= 8_192);

#[derive(Clone, Copy)]
struct AssignConservationEval {
    j_net_mag: f64,
    mom_resid_norm: f64,
    ang_mom_resid_norm_trap: f64,
    ang_mom_resid_norm_h1: f64,
    d_lz_trap: f64,
    dk: f64,
    delta_ke_ratio: f64,
    eps_k: f64,
    l_env_max: f64,
    l_scale: f64,
    h0_ok: bool,
    h0_chi2: f64,
    h0_imp_resid: f64,
    h1_ok: bool,
    h1_chi2: f64,
    h1_imp_resid: f64,
    h1_tau: f64,
    h1_e: f64,
    finite_ok: bool,
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn eval_occluded_assignment(
    x1_pre: f64,
    y1_pre: f64,
    vx1_pre: f64,
    vy1_pre: f64,
    m1: f64,
    r1: f64,
    x2_pre: f64,
    y2_pre: f64,
    vx2_pre: f64,
    vy2_pre: f64,
    m2: f64,
    r2: f64,
    x1_out: f64,
    y1_out: f64,
    vx1_out: f64,
    vy1_out: f64,
    x2_out: f64,
    y2_out: f64,
    vx2_out: f64,
    vy2_out: f64,
    dt: f64,
    g_y: f64,
    drag_beta_max: f64,
    a_np: f64,
    a_env_1: f64,
    a_env_2: f64,
    v_drag_1: f64,
    v_drag_2: f64,
    sigma_pos: f64,
    sigma_vel: f64,
    j_env_max: f64,
) -> AssignConservationEval {
    let x_ref = 0.5 * (x1_pre + x2_pre);
    let y_ref = 0.5 * (y1_pre + y2_pre);

    let v1_pre_sq = vx1_pre * vx1_pre + vy1_pre * vy1_pre;
    let v2_pre_sq = vx2_pre * vx2_pre + vy2_pre * vy2_pre;
    let v1_out_sq = vx1_out * vx1_out + vy1_out * vy1_out;
    let v2_out_sq = vx2_out * vx2_out + vy2_out * vy2_out;

    let v1_pre_mag = hypot_f64(vx1_pre, vy1_pre);
    let v2_pre_mag = hypot_f64(vx2_pre, vy2_pre);
    let v1_out_mag = hypot_f64(vx1_out, vy1_out);
    let v2_out_mag = hypot_f64(vx2_out, vy2_out);

    // Net non-gravitational impulse across [0, dt]
    let p_pre_x = m1 * vx1_pre + m2 * vx2_pre;
    let p_pre_y = m1 * vy1_pre + m2 * vy2_pre;
    let p_post_x = m1 * vx1_out + m2 * vx2_out;
    let p_post_y = m1 * vy1_out + m2 * vy2_out;
    let j_grav_y = (m1 + m2) * g_y * dt;
    let j_net_x = p_post_x - p_pre_x;
    let j_net_y = (p_post_y - p_pre_y) - j_grav_y;
    let j_net_mag = hypot_f64(j_net_x, j_net_y);

    let p_scale = (0.5
        * (m1 * (v1_pre_mag + v1_out_mag)
            + m2 * (v2_pre_mag + v2_out_mag)
            + (m1 + m2) * g_y.abs() * dt))
        .max(j_env_max)
        .max(1.0e-3 * (m1 + m2));
    let mom_resid_norm = if j_net_mag <= 1.0e-6 || p_scale <= 0.0 {
        0.0
    } else {
        j_net_mag / p_scale
    };

    // Angular momentum about fixed inertial origin (x_ref, y_ref) with gravity torque
    let l_pre = m1 * ((x1_pre - x_ref) * vy1_pre - (y1_pre - y_ref) * vx1_pre)
        + m2 * ((x2_pre - x_ref) * vy2_pre - (y2_pre - y_ref) * vx2_pre);
    let l_post = m1 * ((x1_out - x_ref) * vy1_out - (y1_out - y_ref) * vx1_out)
        + m2 * ((x2_out - x_ref) * vy2_out - (y2_out - y_ref) * vx2_out);
    let tau_grav_trap = (m1 * (0.5 * (x1_pre + x1_out) - x_ref)
        + m2 * (0.5 * (x2_pre + x2_out) - x_ref))
        * g_y
        * dt;
    let d_lz_trap = (l_post - l_pre) - tau_grav_trap;

    let arm1_pre = hypot_f64(x1_pre - x_ref, y1_pre - y_ref) + r1;
    let arm1_post = hypot_f64(x1_out - x_ref, y1_out - y_ref) + r1;
    let arm1 = arm1_pre.max(arm1_post);
    let arm2_pre = hypot_f64(x2_pre - x_ref, y2_pre - y_ref) + r2;
    let arm2_post = hypot_f64(x2_out - x_ref, y2_out - y_ref) + r2;
    let arm2 = arm2_pre.max(arm2_post);
    let arm_eff = (0.5 * (arm1 + arm2)).max((r1 + r2).max(1.0));

    let g_half = 0.5 * g_y * dt;
    let sigma_l_pos = sigma_pos
        * hypot_f64(
            m1 * hypot_f64(hypot_f64(vx1_pre, vy1_pre), hypot_f64(v1_out_mag, g_half)),
            m2 * hypot_f64(hypot_f64(vx2_pre, vy2_pre), hypot_f64(v2_out_mag, g_half)),
        );
    let sigma_l_vel = sigma_vel
        * hypot_f64(m1 * hypot_f64(arm1_pre, arm1_post), m2 * hypot_f64(arm2_pre, arm2_post));
    let l_env_max = dt * (m1 * a_env_1 * arm1 + m2 * a_env_2 * arm2)
        + 3.0 * hypot_f64(sigma_l_pos, sigma_l_vel)
        + 1.0e-3 * (m1 + m2) * arm_eff * (1.0 + 0.01 * (v_drag_1 + v_drag_2));
    let l_scale = (l_pre.abs() + l_post.abs()).max((p_scale * arm_eff).max(l_env_max));
    let ang_mom_resid_norm_trap = if d_lz_trap.abs() <= 1.0e-5 || l_scale <= 0.0 {
        0.0
    } else {
        d_lz_trap.abs() / l_scale
    };

    // Gravity-compensated mechanical energy change dK
    let k_pre = 0.5 * (m1 * v1_pre_sq + m2 * v2_pre_sq);
    let k_post_raw = 0.5 * (m1 * v1_out_sq + m2 * v2_out_sq);
    let w_grav = m1 * g_y * (y1_out - y1_pre) + m2 * g_y * (y2_out - y2_pre);
    let k_post_grav_work = k_post_raw - w_grav;
    let dk = k_post_grav_work - k_pre;
    let k_ref = k_pre.max(k_post_grav_work.abs().max(1.0e-3 * (m1 + m2)));
    let delta_ke_ratio = if dk.abs() <= 1.0e-5 || k_ref <= 0.0 { 0.0 } else { dk / k_ref };

    let sigma_k_vel = sigma_vel
        * hypot_f64(m1 * hypot_f64(v1_pre_mag, v1_out_mag), m2 * hypot_f64(v2_pre_mag, v2_out_mag))
        + 2.0 * (m1 + m2) * sigma_vel * sigma_vel;
    let sigma_k_pos = 1.4142135623730951 * hypot_f64(m1, m2) * g_y.abs() * sigma_pos;
    let w_env_pos = persp_scale_bound_term_f64(
        a_np,
        dt,
        m1,
        m2,
        v1_pre_mag,
        v1_out_mag,
        v2_pre_mag,
        v2_out_mag,
    );
    let eps_k = w_env_pos
        + 3.0 * hypot_f64(sigma_k_vel, sigma_k_pos)
        + 1.0e-3 * k_pre.max(m1 + m2);

    // Individual gravity-compensated velocity jumps & position propagation residuals
    let dv1_x = vx1_out - vx1_pre;
    let dv1_y = vy1_out - (vy1_pre + g_y * dt);
    let dv2_x = vx2_out - vx2_pre;
    let dv2_y = vy2_out - (vy2_pre + g_y * dt);
    let vel_err_1 = hypot_f64(dv1_x, dv1_y);
    let vel_err_2 = hypot_f64(dv2_x, dv2_y);

    let g_dt2_half = 0.5 * g_y * dt * dt;
    let dp1_x = (x1_pre + vx1_pre * dt) - x1_out;
    let dp1_y = (y1_pre + vy1_pre * dt + g_dt2_half) - y1_out;
    let dp2_x = (x2_pre + vx2_pre * dt) - x2_out;
    let dp2_y = (y2_pre + vy2_pre * dt + g_dt2_half) - y2_out;
    let pos_err_1 = hypot_f64(dp1_x, dp1_y);
    let pos_err_2 = hypot_f64(dp2_x, dp2_y);

    // H0: Ballistic pass-through without contact
    let sqrt2 = 1.4142135623730951f64;
    let pos_noise = 3.0 * hypot_f64(sqrt2 * sigma_pos, sigma_vel * dt);
    let vel_noise = 3.0 * sqrt2 * sigma_vel;
    let num_eps_pos = 1.0e-3 * (1.0 + 0.01 * (v_drag_1 + v_drag_2) * dt);
    let num_eps_vel = 1.0e-3 * (1.0 + 0.01 * (v_drag_1 + v_drag_2));
    let pos_tol_1 = 0.5 * a_env_1 * dt * dt + pos_noise + num_eps_pos;
    let pos_tol_2 = 0.5 * a_env_2 * dt * dt + pos_noise + num_eps_pos;
    let vel_tol_1 = a_env_1 * dt + vel_noise + num_eps_vel;
    let vel_tol_2 = a_env_2 * dt + vel_noise + num_eps_vel;

    let dk_1 = 0.5 * m1 * (v1_out_sq - v1_pre_sq) - m1 * g_y * (y1_out - y1_pre);
    let dk_2 = 0.5 * m2 * (v2_out_sq - v2_pre_sq) - m2 * g_y * (y2_out - y2_pre);
    let eps_k1 = a_np * dt * m1 * 0.5 * (v1_pre_mag + v1_out_mag)
        + 0.5 * m1 * (a_np * dt) * (a_np * dt)
        + 3.0 * sigma_vel * m1 * hypot_f64(v1_pre_mag, v1_out_mag)
        + 3.0 * sqrt2 * m1 * g_y.abs() * sigma_pos
        + 2.0 * m1 * sigma_vel * sigma_vel
        + 1.0e-3 * (0.5 * m1 * v1_pre_sq).max(m1);
    let eps_k2 = a_np * dt * m2 * 0.5 * (v2_pre_mag + v2_out_mag)
        + 0.5 * m2 * (a_np * dt) * (a_np * dt)
        + 3.0 * sigma_vel * m2 * hypot_f64(v2_pre_mag, v2_out_mag)
        + 3.0 * sqrt2 * m2 * g_y.abs() * sigma_pos
        + 2.0 * m2 * sigma_vel * sigma_vel
        + 1.0e-3 * (0.5 * m2 * v2_pre_sq).max(m2);

    let h0_cons_ok = j_net_mag <= j_env_max
        && d_lz_trap.abs() <= l_env_max
        && dk <= eps_k
        && dk_1 <= eps_k1
        && dk_2 <= eps_k2;
    let h0_ok = pos_err_1 <= pos_tol_1
        && pos_err_2 <= pos_tol_2
        && vel_err_1 <= vel_tol_1
        && vel_err_2 <= vel_tol_2
        && h0_cons_ok;

    let r_p1 = pos_err_1 / pos_tol_1;
    let r_p2 = pos_err_2 / pos_tol_2;
    let r_v1 = vel_err_1 / vel_tol_1;
    let r_v2 = vel_err_2 / vel_tol_2;
    let mut h0_chi2 = 9.0 * (r_p1 * r_p1 + r_p2 * r_p2 + r_v1 * r_v1 + r_v2 * r_v2);
    if !h0_cons_ok {
        let r_j = j_net_mag / j_env_max;
        let r_l = d_lz_trap.abs() / l_env_max;
        let r_k = dk.max(0.0) / eps_k;
        h0_chi2 += 9.0 * (1.0 + r_j * r_j + r_l * r_l + r_k * r_k);
    }
    let h0_imp_resid = m1 * vel_err_1 + m2 * vel_err_2;

    // H1: Hidden two-body collision at tau in [0, dt]
    let j12_x = 0.5 * (m1 * dv1_x - m2 * dv2_x);
    let j12_y = 0.5 * (m1 * dv1_y - m2 * dv2_y);
    let j12_mag = hypot_f64(j12_x, j12_y);
    let sigma_j12 = 0.5 * 1.4142135623730951 * hypot_f64(m1, m2) * sigma_vel;
    let j12_min = (3.0 * sigma_j12 + 0.5 * (m1 * a_env_1 + m2 * a_env_2) * dt)
        .max(1.0e-2 * (m1 + m2));
    let (nx, ny) = if j12_mag > 1.0e-12 {
        (j12_x / j12_mag, j12_y / j12_mag)
    } else {
        (1.0, 0.0)
    };

    let r1_post0_x = x1_out - vx1_out * dt;
    let r1_post0_y = y1_out - vy1_out * dt + g_dt2_half;
    let r2_post0_x = x2_out - vx2_out * dt;
    let r2_post0_y = y2_out - vy2_out * dt + g_dt2_half;

    let gap1_0x = x1_pre - r1_post0_x;
    let gap1_0y = y1_pre - r1_post0_y;
    let gap2_0x = x2_pre - r2_post0_x;
    let gap2_0y = y2_pre - r2_post0_y;

    let dr0_x = x1_pre - x2_pre;
    let dr0_y = y1_pre - y2_pre;
    let dr_post0_x = r1_post0_x - r2_post0_x;
    let dr_post0_y = r1_post0_y - r2_post0_y;

    let vrel_pre_x = vx1_pre - vx2_pre;
    let vrel_pre_y = vy1_pre - vy2_pre;
    let vrel_out_x = vx1_out - vx2_out;
    let vrel_out_y = vy1_out - vy2_out;

    let dv_sq = dv1_x * dv1_x + dv1_y * dv1_y + dv2_x * dv2_x + dv2_y * dv2_y;
    let vrel_pre_sq = vrel_pre_x * vrel_pre_x + vrel_pre_y * vrel_pre_y;
    let vrel_out_sq = vrel_out_x * vrel_out_x + vrel_out_y * vrel_out_y;

    let r_sum = r1 + r2;
    let cx_off = r_sum * nx;
    let cy_off = r_sum * ny;

    let a_tau = dv_sq + 0.5 * (vrel_pre_sq + vrel_out_sq);
    let b_tau = (gap1_0x * dv1_x + gap1_0y * dv1_y)
        + (gap2_0x * dv2_x + gap2_0y * dv2_y)
        - 0.5
            * (((dr0_x - cx_off) * vrel_pre_x + (dr0_y - cy_off) * vrel_pre_y)
                + ((dr_post0_x - cx_off) * vrel_out_x + (dr_post0_y - cy_off) * vrel_out_y));
    let tau = if a_tau > 1.0e-12 {
        (b_tau / a_tau).clamp(0.0, dt)
    } else {
        0.5 * dt
    };

    let kink1 = hypot_f64(gap1_0x - dv1_x * tau, gap1_0y - dv1_y * tau);
    let kink2 = hypot_f64(gap2_0x - dv2_x * tau, gap2_0y - dv2_y * tau);
    let dr_pre_tx = dr0_x + vrel_pre_x * tau;
    let dr_pre_ty = dr0_y + vrel_pre_y * tau;
    let dr_post_tx = dr_post0_x + vrel_out_x * tau;
    let dr_post_ty = dr_post0_y + vrel_out_y * tau;
    let dr_mid_tx = 0.5 * (dr_pre_tx + dr_post_tx);
    let dr_mid_ty = 0.5 * (dr_pre_ty + dr_post_ty);

    let sep_pre = hypot_f64(dr_pre_tx, dr_pre_ty);
    let sep_post = hypot_f64(dr_post_tx, dr_post_ty);
    let sep_mid = hypot_f64(dr_mid_tx, dr_mid_ty);

    let dt_rem = dt - tau;
    let sigma_kink = wasm_sqrtd(
        2.0 * sigma_pos * sigma_pos + sigma_vel * sigma_vel * (tau * tau + dt_rem * dt_rem),
    );
    let env_kink_1 = 0.5 * a_env_1 * (tau * tau + dt_rem * dt_rem);
    let env_kink_2 = 0.5 * a_env_2 * (tau * tau + dt_rem * dt_rem);
    let kink_tol_1 = env_kink_1 + 3.0 * sqrt2 * sigma_kink + num_eps_pos;
    let kink_tol_2 = env_kink_2 + 3.0 * sqrt2 * sigma_kink + num_eps_pos;
    let contact_pos_tol = 0.5 * (env_kink_1 + env_kink_2) + 3.0 * sqrt2 * sigma_kink + num_eps_pos;

    // Exact piecewise gravity torque at contact instant tau for H1
    let x1_c = 0.5 * ((x1_pre + vx1_pre * tau) + (x1_out - vx1_out * dt_rem));
    let x2_c = 0.5 * ((x2_pre + vx2_pre * tau) + (x2_out - vx2_out * dt_rem));
    let tau_grav_h1 = (m1
        * (0.5 * ((x1_pre - x_ref) + (x1_c - x_ref)) * tau
            + 0.5 * ((x1_c - x_ref) + (x1_out - x_ref)) * dt_rem)
        + m2 * (0.5 * ((x2_pre - x_ref) + (x2_c - x_ref)) * tau
            + 0.5 * ((x2_c - x_ref) + (x2_out - x_ref)) * dt_rem))
        * g_y;
    let d_lz_h1 = (l_post - l_pre) - tau_grav_h1;
    let ang_mom_resid_norm_h1 = if d_lz_h1.abs() <= 1.0e-5 || l_scale <= 0.0 {
        0.0
    } else {
        d_lz_h1.abs() / l_scale
    };

    // Normal approach, separation, and joint velocity + position-chord restitution
    let vn_pre = vrel_pre_x * nx + vrel_pre_y * ny;
    let vn_post = vrel_out_x * nx + vrel_out_y * ny;
    let v_close_vel = -vn_pre;
    let v_close_min = (2.0 * sigma_vel).max(1.0e-3);
    let dv_rel_tol = 3.0 * sqrt2 * sigma_vel + 0.5 * (a_env_1 + a_env_2) * dt + num_eps_vel;

    let dr_end_x = x1_out - x2_out;
    let dr_end_y = y1_out - y2_out;
    let dr0_n = dr0_x * nx + dr0_y * ny;
    let dr_end_n = dr_end_x * nx + dr_end_y * ny;
    let end_sep_tol = 3.0 * sqrt2 * sigma_pos + num_eps_pos;
    let end_side_ok = dr0_n >= r_sum - end_sep_tol && dr_end_n >= r_sum - end_sep_tol;

    let a_env_sum = a_env_1 + a_env_2;
    let var_vel = 2.0 * sigma_vel * sigma_vel + 0.25 * a_env_sum * a_env_sum * dt * dt + 1.0e-12;
    let w_vel = 1.0 / var_vel;

    let (v_close_fit, alpha_pre) = if tau > 1.0e-4 {
        let vc_chord = (dr0_n - r_sum) / tau;
        let var_c_pre = (4.0 * sigma_pos * sigma_pos) / (tau * tau)
            + 0.25 * a_env_sum * a_env_sum * tau * tau
            + 1.0e-12;
        let w_c = 1.0 / var_c_pre;
        let a_p = w_c / (w_vel + w_c);
        (v_close_vel + a_p * (vc_chord - v_close_vel), a_p)
    } else {
        (v_close_vel, 0.0)
    };

    let (vn_post_fit, alpha_post) = if dt_rem > 1.0e-4 {
        let vp_chord = (dr_end_n - r_sum) / dt_rem;
        let var_c_post = (4.0 * sigma_pos * sigma_pos) / (dt_rem * dt_rem)
            + 0.25 * a_env_sum * a_env_sum * dt_rem * dt_rem
            + 1.0e-12;
        let w_c = 1.0 / var_c_post;
        let a_p = w_c / (w_vel + w_c);
        (vn_post + a_p * (vp_chord - vn_post), a_p)
    } else {
        (vn_post, 0.0)
    };

    let e_fit = if v_close_fit > 1.0e-6 {
        vn_post_fit / v_close_fit
    } else {
        -1.0
    };

    // Scale-aware f32 input quantization slack (4 half-ULPs of f32; never widens noisy fits)
    const F32_HALF_ULP: f64 = 0.5 * (f32::EPSILON as f64);
    let pos_mag = x1_pre
        .abs()
        .max(y1_pre.abs())
        .max(x2_pre.abs())
        .max(y2_pre.abs())
        .max(x1_out.abs())
        .max(y1_out.abs())
        .max(x2_out.abs())
        .max(y2_out.abs())
        .max(r_sum)
        .max(1.0);
    let vel_mag = v1_pre_mag
        .max(v2_pre_mag)
        .max(v1_out_mag)
        .max(v2_out_mag)
        .max(1.0);
    let ulp_pos = 4.0 * F32_HALF_ULP * pos_mag;
    let ulp_vel = 4.0 * F32_HALF_ULP * vel_mag;
    let dv_ulp_pre = ulp_vel + if tau > 1.0e-4 { alpha_pre * (ulp_pos / tau) } else { 0.0 };
    let dv_ulp_post = ulp_vel + if dt_rem > 1.0e-4 { alpha_post * (ulp_pos / dt_rem) } else { 0.0 };
    let e_ulp_tol = if v_close_fit > 1.0e-6 {
        ((dv_ulp_post + dv_ulp_pre) / v_close_fit).min(5.0e-5) + 1.0e-12
    } else {
        1.0e-12
    };

    let mu = (m1 * m2) / (m1 + m2);
    let vc_pos = v_close_fit.max(0.0);
    let e_phys = e_fit.clamp(0.0, 1.0);
    let dk_expected = -0.5 * mu * (1.0 - e_phys * e_phys) * vc_pos * vc_pos;
    let w_drag_max =
        drag_beta_max * dt * (m1 * v_drag_1 * v_drag_1 + m2 * v_drag_2 * v_drag_2);
    let dk_consistent = dk <= eps_k && dk >= dk_expected - w_drag_max - 1.5 * eps_k;

    let norm_dot = dr_mid_tx * nx + dr_mid_ty * ny;
    let norm_side_ok = if sep_mid > 1.0e-6 && r_sum > contact_pos_tol {
        norm_dot >= 0.85 * sep_mid - contact_pos_tol
    } else {
        norm_dot >= -contact_pos_tol
    };
    let sep_mid_ok = (sep_mid - r_sum).abs() <= contact_pos_tol;
    let sep_pre_ok =
        sep_pre >= r_sum - 1.25 * contact_pos_tol && sep_pre <= r_sum + 1.25 * contact_pos_tol;
    let sep_post_ok =
        sep_post >= r_sum - 1.25 * contact_pos_tol && sep_post <= r_sum + 1.25 * contact_pos_tol;
    let contact_ok = kink1 <= kink_tol_1
        && kink2 <= kink_tol_2
        && sep_mid_ok
        && sep_pre_ok
        && sep_post_ok
        && end_side_ok
        && norm_side_ok;

    let h1_dyn_ok = j12_mag > j12_min
        && v_close_fit > v_close_min
        && vn_post_fit >= -dv_ulp_post
        && e_fit >= -e_ulp_tol
        && e_fit <= 1.0 + e_ulp_tol
        && (v_close_vel - v_close_fit).abs() <= dv_rel_tol
        && (vn_post - vn_post_fit).abs() <= dv_rel_tol;
    let h1_ok = j_net_mag <= j_env_max
        && d_lz_h1.abs() <= l_env_max
        && dk_consistent
        && h1_dyn_ok
        && contact_ok;

    let r_j = j_net_mag / j_env_max;
    let r_lh1 = d_lz_h1.abs() / l_env_max;
    let r_k1 = kink1 / kink_tol_1;
    let r_k2 = kink2 / kink_tol_2;
    let r_sep = (sep_mid - r_sum).abs() / contact_pos_tol;
    let r_dk = dk.max(0.0) / eps_k;
    let mut h1_chi2 = 9.0
        * (r_j * r_j + r_lh1 * r_lh1 + r_k1 * r_k1 + r_k2 * r_k2 + r_sep * r_sep + r_dk * r_dk);
    if !h1_ok {
        if !h1_dyn_ok || !norm_side_ok || !end_side_ok || !dk_consistent {
            h1_chi2 += 16.0;
        }
        if !sep_pre_ok || !sep_post_ok {
            h1_chi2 += 9.0;
        }
    }

    let e_out = if h1_ok { e_fit.clamp(0.0, 1.0) } else { e_fit };
    let finite_ok = f64_fits_f32(j_net_mag)
        && f64_fits_f32(mom_resid_norm)
        && f64_fits_f32(ang_mom_resid_norm_trap)
        && f64_fits_f32(ang_mom_resid_norm_h1)
        && f64_fits_f32(delta_ke_ratio)
        && f64_fits_f32(h0_chi2)
        && f64_fits_f32(h0_imp_resid)
        && f64_fits_f32(h1_chi2)
        && f64_fits_f32(tau)
        && f64_fits_f32(e_out)
        && f64_fits_f32(eps_k)
        && f64_fits_f32(l_env_max)
        && f64_fits_f32(l_scale);

    AssignConservationEval {
        j_net_mag,
        mom_resid_norm,
        ang_mom_resid_norm_trap,
        ang_mom_resid_norm_h1,
        d_lz_trap,
        dk,
        delta_ke_ratio,
        eps_k,
        l_env_max,
        l_scale,
        h0_ok,
        h0_chi2,
        h0_imp_resid,
        h1_ok,
        h1_chi2,
        h1_imp_resid: j_net_mag,
        h1_tau: tau,
        h1_e: e_out,
        finite_ok,
    }
}

#[inline(always)]
fn persp_scale_bound_term_f64(
    a_np: f64,
    dt: f64,
    m1: f64,
    m2: f64,
    v1_pre_mag: f64,
    v1_out_mag: f64,
    v2_pre_mag: f64,
    v2_out_mag: f64,
) -> f64 {
    a_np * dt * (m1 * 0.5 * (v1_pre_mag + v1_out_mag) + m2 * 0.5 * (v2_pre_mag + v2_out_mag))
        + 0.5 * (m1 + m2) * (a_np * dt) * (a_np * dt)
}

#[inline(never)]
fn eval_occluded_pair_core(
    inp: &[f32; CONSERVATION_INPUT_LEN],
    telem: &mut [f32; CONSERVATION_TELEM_LEN],
) -> u32 {
    telem.fill(0.0);

    let mut i = 0usize;
    while i < CONSERVATION_INPUT_LEN - 1 {
        if !inp[i].is_finite() {
            return OCCL_STATUS_NONE;
        }
        i += 1;
    }

    let x1_pre = inp[0] as f64;
    let y1_pre = inp[1] as f64;
    let vx1_pre = inp[2] as f64;
    let vy1_pre = inp[3] as f64;
    let m1_in = inp[4] as f64;
    let r1 = inp[5] as f64;

    let x2_pre = inp[6] as f64;
    let y2_pre = inp[7] as f64;
    let vx2_pre = inp[8] as f64;
    let vy2_pre = inp[9] as f64;
    let m2_in = inp[10] as f64;
    let r2 = inp[11] as f64;

    let x_a = inp[12] as f64;
    let y_a = inp[13] as f64;
    let vx_a = inp[14] as f64;
    let vy_a = inp[15] as f64;

    let x_b = inp[16] as f64;
    let y_b = inp[17] as f64;
    let vx_b = inp[18] as f64;
    let vy_b = inp[19] as f64;

    let dt = inp[20] as f64;
    let g_y = inp[21] as f64;
    let drag_beta_max = inp[22] as f64;
    let spin_magnus_bound = inp[23] as f64;
    let persp_scale_bound = inp[24] as f64;
    let sigma_pos = inp[25] as f64;
    let sigma_vel = inp[26] as f64;
    let tid_a = inp[27];
    let tid_b = inp[28];
    let event_frame = inp[29];
    let mass_known_flag = inp[30];

    if dt <= 0.0
        || r1 < 0.0
        || r2 < 0.0
        || drag_beta_max < 0.0
        || spin_magnus_bound < 0.0
        || persp_scale_bound < 0.0
        || sigma_pos < 0.0
        || sigma_vel < 0.0
    {
        return OCCL_STATUS_NONE;
    }

    let (m1, m2) = if mass_known_flag > 0.5 {
        if m1_in <= 0.0 || m2_in <= 0.0 {
            return OCCL_STATUS_NONE;
        }
        (m1_in, m2_in)
    } else {
        if m1_in < 0.0 || m2_in < 0.0 {
            return OCCL_STATUS_NONE;
        }
        (1.0f64, 1.0f64)
    };

    let mass_ratio = m2 / m1;
    if !f64_fits_f32(mass_ratio) {
        return OCCL_STATUS_NONE;
    }

    // Pre-state-only drag speed bounds (strictly non-circular)
    let v1_pre_mag = hypot_f64(vx1_pre, vy1_pre);
    let v2_pre_mag = hypot_f64(vx2_pre, vy2_pre);
    let v_drag_1 = v1_pre_mag.max(hypot_f64(vx1_pre, vy1_pre + g_y * dt));
    let v_drag_2 = v2_pre_mag.max(hypot_f64(vx2_pre, vy2_pre + g_y * dt));

    let a_np = spin_magnus_bound + persp_scale_bound;
    let a_env_1 = drag_beta_max * v_drag_1 + a_np;
    let a_env_2 = drag_beta_max * v_drag_2 + a_np;

    let sigma_j = 1.4142135623730951 * hypot_f64(m1, m2) * sigma_vel;
    let j_num_eps = 1.0e-4 * (m1 + m2) * (1.0 + 0.01 * (v_drag_1 + v_drag_2));
    let j_env_max = dt * (m1 * a_env_1 + m2 * a_env_2) + 3.0 * sigma_j + j_num_eps;
    if !f64_fits_f32(j_env_max) {
        return OCCL_STATUS_NONE;
    }

    let keep = eval_occluded_assignment(
        x1_pre, y1_pre, vx1_pre, vy1_pre, m1, r1, x2_pre, y2_pre, vx2_pre, vy2_pre, m2, r2, x_a,
        y_a, vx_a, vy_a, x_b, y_b, vx_b, vy_b, dt, g_y, drag_beta_max, a_np, a_env_1, a_env_2,
        v_drag_1, v_drag_2, sigma_pos, sigma_vel, j_env_max,
    );
    let swap = eval_occluded_assignment(
        x1_pre, y1_pre, vx1_pre, vy1_pre, m1, r1, x2_pre, y2_pre, vx2_pre, vy2_pre, m2, r2, x_b,
        y_b, vx_b, vy_b, x_a, y_a, vx_a, vy_a, dt, g_y, drag_beta_max, a_np, a_env_1, a_env_2,
        v_drag_1, v_drag_2, sigma_pos, sigma_vel, j_env_max,
    );
    if !keep.finite_ok || !swap.finite_ok {
        return OCCL_STATUS_NONE;
    }

    let mut mask: u32 = 0;
    if keep.h0_ok {
        mask |= HYP_PASSTHROUGH_KEEP;
    }
    if swap.h0_ok {
        mask |= HYP_PASSTHROUGH_SWAP;
    }
    if keep.h1_ok {
        mask |= HYP_COLLISION_KEEP;
    }
    if swap.h1_ok {
        mask |= HYP_COLLISION_SWAP;
    }

    let admissible_count = mask.count_ones();
    let status = if admissible_count == 1 {
        OCCL_STATUS_UNIQUE
    } else if admissible_count >= 2 {
        OCCL_STATUS_AMBIGUOUS
    } else {
        OCCL_STATUS_VIOLATED
    };

    let keep_any = keep.h0_ok || keep.h1_ok;
    let swap_any = swap.h0_ok || swap.h1_ok;
    let use_swap = if keep_any && !swap_any {
        false
    } else if swap_any && !keep_any {
        true
    } else {
        let kj = keep.j_net_mag / j_env_max;
        let kk = keep.dk.max(0.0) / keep.eps_k;
        let sj = swap.j_net_mag / j_env_max;
        let sk = swap.dk.max(0.0) / swap.eps_k;
        (sj * sj + sk * sk) < (kj * kj + kk * kk)
    };
    let top = if use_swap { &swap } else { &keep };

    let top_ang_mom_resid = if !use_swap {
        if keep.h1_ok && !keep.h0_ok {
            keep.ang_mom_resid_norm_h1
        } else if keep.h0_ok && !keep.h1_ok {
            keep.ang_mom_resid_norm_trap
        } else {
            keep.ang_mom_resid_norm_trap.min(keep.ang_mom_resid_norm_h1)
        }
    } else if swap.h1_ok && !swap.h0_ok {
        swap.ang_mom_resid_norm_h1
    } else if swap.h0_ok && !swap.h1_ok {
        swap.ang_mom_resid_norm_trap
    } else {
        swap.ang_mom_resid_norm_trap.min(swap.ang_mom_resid_norm_h1)
    };
    let _ = (top.d_lz_trap, top.l_env_max, top.l_scale);

    let (est_tau_sec, est_restitution_e) = if keep.h1_ok && swap.h1_ok {
        if swap.h1_chi2 < keep.h1_chi2 {
            (swap.h1_tau, swap.h1_e)
        } else {
            (keep.h1_tau, keep.h1_e)
        }
    } else if keep.h1_ok {
        (keep.h1_tau, keep.h1_e)
    } else if swap.h1_ok {
        (swap.h1_tau, swap.h1_e)
    } else {
        (0.0, 0.0)
    };

    telem[0] = status as f32;
    telem[1] = mask as f32;
    telem[2] = admissible_count as f32;
    telem[3] = top.j_net_mag as f32;
    telem[4] = j_env_max as f32;
    telem[5] = top.mom_resid_norm as f32;
    telem[6] = top_ang_mom_resid as f32;
    telem[7] = top.delta_ke_ratio as f32;
    telem[8] = keep.h0_chi2 as f32;
    telem[9] = swap.h0_chi2 as f32;
    telem[10] = keep.h1_chi2 as f32;
    telem[11] = swap.h1_chi2 as f32;
    telem[12] = est_tau_sec as f32;
    telem[13] = est_restitution_e as f32;
    telem[14] = tid_a;
    telem[15] = tid_b;
    telem[16] = dt as f32;
    telem[17] = event_frame;
    telem[18] = keep.h0_imp_resid as f32;
    telem[19] = swap.h0_imp_resid as f32;
    telem[20] = keep.h1_imp_resid as f32;
    telem[21] = swap.h1_imp_resid as f32;
    telem[22] = g_y as f32;
    telem[23] = mass_ratio as f32;

    status
}

impl OccludedConservationState {
    pub const fn zeroed() -> Self {
        Self {
            telem: [0.0; CONSERVATION_TELEM_LEN],
            input: [0.0; CONSERVATION_INPUT_LEN],
            track_pairs: [OcclPairTrackState::zeroed(); 8],
            slot_obs: [SlotObsHistory::zeroed(); MAX_TARGETS],
            live_event_seq: 0,
            last_frame_idx: 0,
            last_discont_seen: 0,
            last_continuity_epoch: 0,
            has_last_frame: 0,
            _pad: [0; 3],
        }
    }

    pub fn clear_tracking_state(&mut self) {
        let mut p = 0usize;
        while p < 8 {
            self.track_pairs[p].reset_in_place();
            p += 1;
        }
        let mut m = 0usize;
        while m < MAX_TARGETS {
            self.slot_obs[m].reset_in_place();
            m += 1;
        }
        self.last_frame_idx = 0;
        self.last_discont_seen = 0;
        self.has_last_frame = 0;
    }

    pub fn reset_continuity(&mut self) {
        self.telem.fill(0.0);
        self.live_event_seq = 0;
        self.last_continuity_epoch = 0;
        self.clear_tracking_state();
    }

    pub fn reset(&mut self) {
        self.telem.fill(0.0);
        self.input.fill(0.0);
        self.live_event_seq = 0;
        self.last_continuity_epoch = 0;
        self.clear_tracking_state();
    }

    pub fn eval_staged(&mut self) -> u32 {
        eval_occluded_pair_core(&self.input, &mut self.telem)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollisionAnalyzerState {
    pub last_event: CollisionPairEvent,
    pub target_last_e: [f32; MAX_TARGETS],
    pub target_last_frame: [u32; MAX_TARGETS],
    pub target_detect_frame: [u32; MAX_TARGETS],
    pub target_last_tid: [u32; MAX_TARGETS],
    pub target_partner_tid: [u32; MAX_TARGETS],
    pub recent_pairs: [RecentCollisionPair; 8],
    pub recent_cursor: u8,
    pub _pad: [u8; 3],
    pub conservation: OccludedConservationState,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RecentCollisionPair {
    pub tid_lo: u32,
    pub tid_hi: u32,
    pub event_frame: u32,
}

impl RecentCollisionPair {
    pub const fn zeroed() -> Self {
        Self {
            tid_lo: 0,
            tid_hi: 0,
            event_frame: 0,
        }
    }
}

impl CollisionAnalyzerState {
    pub const fn zeroed() -> Self {
        Self {
            last_event: CollisionPairEvent::zeroed(),
            target_last_e: [0.0; MAX_TARGETS],
            target_last_frame: [0; MAX_TARGETS],
            target_detect_frame: [0; MAX_TARGETS],
            target_last_tid: [0; MAX_TARGETS],
            target_partner_tid: [0; MAX_TARGETS],
            recent_pairs: [RecentCollisionPair::zeroed(); 8],
            recent_cursor: 0,
            _pad: [0; 3],
            conservation: OccludedConservationState::zeroed(),
        }
    }

    #[allow(dead_code)]
    pub fn reset_slot(&mut self, slot_m: usize) {
        reset_collision_slot(self, slot_m);
    }
    pub fn reset(&mut self) {
        self.last_event = CollisionPairEvent::zeroed();
        self.target_last_e.fill(0.0);
        self.target_last_frame.fill(0);
        self.target_detect_frame.fill(0);
        self.target_last_tid.fill(0);
        self.target_partner_tid.fill(0);
        self.recent_pairs.fill(RecentCollisionPair::zeroed());
        self.recent_cursor = 0;
        self._pad = [0; 3];
        self.conservation.reset();
    }

    #[inline(always)]
    pub fn conservation_telemetry_ptr(&self) -> *const f32 {
        self.conservation.telem.as_ptr()
    }

    #[inline(always)]
    pub fn conservation_input_mut_ptr(&mut self) -> *mut f32 {
        self.conservation.input.as_mut_ptr()
    }

    #[inline(never)]
    pub fn eval_staged_occluded_pair(&mut self) -> u32 {
        self.conservation.eval_staged()
    }

    #[inline(always)]
    pub fn conservation_live_event_seq(&self) -> u32 {
        self.conservation.live_event_seq
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollisionTickOutput {
    pub jump_code: [u8; MAX_TARGETS],
    pub regime_override: [u8; MAX_TARGETS],
    pub restitution_e: [f32; MAX_TARGETS],
    pub pair_event: CollisionPairEvent,
    pub has_pair_event: u8,
    pub _pad_out: [u8; 3],
    pub ensemble_kinetic_px2_s2: f32,
    pub ensemble_mech_energy_px2_s2: f32,
    pub last_pair_mass_ratio: f32,
    pub last_pair_mom_resid: f32,
    pub last_pair_energy_retention: f32,
    pub last_pair_tid_a: f32,
    pub last_pair_tid_b: f32,
    pub last_pair_frame: f32,
    pub targets: [TargetCollisionOut; MAX_TARGETS],
}

impl CollisionTickOutput {
    pub const fn zeroed() -> Self {
        Self {
            jump_code: [0; MAX_TARGETS],
            regime_override: [0; MAX_TARGETS],
            restitution_e: [0.0; MAX_TARGETS],
            pair_event: CollisionPairEvent::zeroed(),
            has_pair_event: 0,
            _pad_out: [0; 3],
            ensemble_kinetic_px2_s2: 0.0,
            ensemble_mech_energy_px2_s2: 0.0,
            last_pair_mass_ratio: 0.0,
            last_pair_mom_resid: 0.0,
            last_pair_energy_retention: 0.0,
            last_pair_tid_a: 0.0,
            last_pair_tid_b: 0.0,
            last_pair_frame: 0.0,
            targets: [TargetCollisionOut::zeroed(); MAX_TARGETS],
        }
    }
}

#[derive(Clone, Copy)]
struct OlsRay {
    valid: bool,
    px0: f32,     // position extrapolated to tau = 0 (t_ref = t(f_c))
    py0: f32,
    vx: f32,
    vy: f32,
    rmse: f32,
    span_sec: f32,
}

#[derive(Clone, Copy)]
struct SamplePt {
    tau: f32,
    wx: f32,
    wy: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollisionArenaView<'a> {
    pub frame_idx: u32,
    pub continuity_epoch: u32,
    pub lookahead_k: u8,
    pub scene_gravity_conf: f32,
    pub scene_gravity_px_s2: f32,
    pub frame_w: f32,
    pub frame_h: f32,
    pub scale_s: f32,
    pub hist_t_sec: &'a [f64; 64],
    pub hist_cam_x: &'a [f32; 64],
    pub hist_cam_y: &'a [f32; 64],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollisionTrackView<'a> {
    pub active: u8,
    pub confirmed: u8,
    pub is_ballistic: u8,
    pub pivot_valid: u8,
    pub coasting: u8,
    pub track_id: u32,
    pub birth_frame: u32,
    pub orig_birth_frame: u32,
    pub last_discont_frame: u32,
    pub disp_world_x: f32,
    pub disp_world_y: f32,
    pub disp_vx: f32,
    pub disp_vy: f32,
    pub arc_raw_x: &'a [f32; 64],
    pub arc_raw_y: &'a [f32; 64],
    pub arc_vis: &'a [u8; 64],
}

#[inline(always)]
pub fn reset_collision_slot(st: &mut CollisionAnalyzerState, slot_m: usize) {
    if slot_m < MAX_TARGETS {
        let old_tid = st.target_last_tid[slot_m];
        let obs_tid = st.conservation.slot_obs[slot_m].tid;
        st.target_last_e[slot_m] = 0.0;
        st.target_last_frame[slot_m] = 0;
        st.target_detect_frame[slot_m] = 0;
        st.target_last_tid[slot_m] = 0;
        st.target_partner_tid[slot_m] = 0;
        st.conservation.slot_obs[slot_m].reset_in_place();
        for p in 0..8 {
            let pair = &mut st.conservation.track_pairs[p];
            if pair.active != 0
                && (pair.slot_a as usize == slot_m
                    || pair.slot_b as usize == slot_m
                    || (old_tid != 0 && (pair.tid_a == old_tid || pair.tid_b == old_tid))
                    || (obs_tid != 0 && (pair.tid_a == obs_tid || pair.tid_b == obs_tid)))
            {
                pair.reset_in_place();
            }
        }
    }
}

#[inline(always)]
fn derive_nominal_dt_sec(frame_idx: u32, hist_t_sec: &[f64; 64]) -> f32 {
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    let max_k = frame_idx.min(12);
    let mut k = 0u32;
    while k < max_k {
        let f1 = frame_idx - k;
        let f0 = f1 - 1;
        let dt = (hist_t_sec[(f1 as usize) & HIST_RING_MASK] - hist_t_sec[(f0 as usize) & HIST_RING_MASK]) as f32;
        if dt.is_finite() && dt > 0.002 && dt < 0.25 {
            sum += dt;
            cnt += 1;
        }
        k += 1;
    }
    if cnt > 0 {
        (sum / (cnt as f32)).clamp(1.0 / 240.0, 0.25)
    } else {
        1.0 / 30.0
    }
}

#[inline(always)]
fn earliest_valid_frame(a: &CollisionArenaView<'_>, t: &CollisionTrackView<'_>) -> u32 {
    let ring_min = a.frame_idx.saturating_sub((HIST_RING_LEN - 2) as u32);
    ring_min.max(t.last_discont_frame).max(t.birth_frame)
}

#[inline(always)]
fn effective_grav_ay(a: &CollisionArenaView<'_>, t: &CollisionTrackView<'_>) -> f32 {
    if a.scene_gravity_conf >= 0.45 && t.is_ballistic != 0 && t.pivot_valid == 0 {
        a.scene_gravity_px_s2
    } else {
        0.0
    }
}

fn collect_window_samples(
    a: &CollisionArenaView<'_>,
    t: &CollisionTrackView<'_>,
    f_start: u32,
    f_end: u32,
    t_ref_sec: f64,
    grav_ay: f32,
    out: &mut [SamplePt; 10],
) -> usize {
    let min_f = earliest_valid_frame(a, t);
    let fs = f_start.max(min_f);
    let fe = f_end.min(a.frame_idx);
    if fs > fe {
        return 0;
    }
    let mut count = 0usize;
    let mut f = fs;
    while f <= fe && count < 10 {
        let slot = (f as usize) & HIST_RING_MASK;
        if t.arc_vis[slot] != 0 {
            let tau = (a.hist_t_sec[slot] - t_ref_sec) as f32;
            let wx = t.arc_raw_x[slot] + a.hist_cam_x[slot];
            // Subtract ballistic gravity drift (0.5 * g * tau^2) if in free-fall so velocity is exact at tau=0
            let wy = (t.arc_raw_y[slot] + a.hist_cam_y[slot]) - 0.5 * grav_ay * tau * tau;
            out[count] = SamplePt { tau, wx, wy };
            count += 1;
        }
        f += 1;
    }
    count
}

fn fit_ols_line(pts: &[SamplePt], dt_nom: f32) -> OlsRay {
    let n = pts.len();
    if n < 3 {
        return OlsRay {
            valid: false,
            px0: 0.0,
            py0: 0.0,
            vx: 0.0,
            vy: 0.0,
            rmse: 1e9,
            span_sec: 0.0,
        };
    }
    let mut tau_min = pts[0].tau;
    let mut tau_max = pts[0].tau;
    let mut st = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for i in 0..n {
        let p = pts[i];
        if p.tau < tau_min {
            tau_min = p.tau;
        }
        if p.tau > tau_max {
            tau_max = p.tau;
        }
        st += p.tau;
        sx += p.wx;
        sy += p.wy;
    }
    let span = tau_max - tau_min;
    if span < 1.4 * dt_nom {
        return OlsRay {
            valid: false,
            px0: 0.0,
            py0: 0.0,
            vx: 0.0,
            vy: 0.0,
            rmse: 1e9,
            span_sec: span,
        };
    }
    let inv_n = 1.0 / (n as f32);
    let tb = st * inv_n;
    let xb = sx * inv_n;
    let yb = sy * inv_n;
    let mut s_tt = 0.0f32;
    let mut s_tx = 0.0f32;
    let mut s_ty = 0.0f32;
    for i in 0..n {
        let dt = pts[i].tau - tb;
        s_tt += dt * dt;
        s_tx += dt * (pts[i].wx - xb);
        s_ty += dt * (pts[i].wy - yb);
    }
    if s_tt <= 1e-8 {
        return OlsRay {
            valid: false,
            px0: 0.0,
            py0: 0.0,
            vx: 0.0,
            vy: 0.0,
            rmse: 1e9,
            span_sec: span,
        };
    }
    let vx = s_tx / s_tt;
    let vy = s_ty / s_tt;
    let px0 = xb - vx * tb;
    let py0 = yb - vy * tb;
    let mut sse = 0.0f32;
    for i in 0..n {
        let ex = pts[i].wx - (px0 + vx * pts[i].tau);
        let ey = pts[i].wy - (py0 + vy * pts[i].tau);
        sse += ex * ex + ey * ey;
    }
    let dof = (n - 2).max(1) as f32;
    let rmse = sqrt_f32(sse / dof);
    OlsRay {
        valid: true,
        px0,
        py0,
        vx,
        vy,
        rmse,
        span_sec: span,
    }
}

/// 1-pass far-outer peel: protects post-window against secondary cushion bounce (or pre-window against prior event)
/// by peeling ONLY the outermost frame(s) if RMSE drops > 45% and stays >= 4 samples.
fn fit_ray_with_outer_peel(
    pts: &mut [SamplePt; 10],
    len: usize,
    peel_from_left: bool,
    dt_nom: f32,
    s: f32,
) -> (OlsRay, usize, usize) {
    // Active slice is pts[lo..hi]
    let mut lo = 0usize;
    let mut hi = len;
    let mut ray = fit_ols_line(&pts[lo..hi], dt_nom);
    if !ray.valid {
        return (ray, lo, hi);
    }
    if ray.rmse > 1.50 * s && (hi - lo) >= 5 {
        let (cand_lo, cand_hi) = if peel_from_left {
            (lo + 1, hi)
        } else {
            (lo, hi - 1)
        };
        let cand = fit_ols_line(&pts[cand_lo..cand_hi], dt_nom);
        if cand.valid && cand.rmse < 0.60 * ray.rmse {
            lo = cand_lo;
            hi = cand_hi;
            ray = cand;
            if ray.rmse > 0.85 * s && (hi - lo) >= 5 {
                let (c2_lo, c2_hi) = if peel_from_left {
                    (lo + 1, hi)
                } else {
                    (lo, hi - 1)
                };
                let c2 = fit_ols_line(&pts[c2_lo..c2_hi], dt_nom);
                if c2.valid && c2.rmse < 0.50 * ray.rmse {
                    lo = c2_lo;
                    hi = c2_hi;
                    ray = c2;
                }
            }
        }
    }
    (ray, lo, hi)
}

/// Solve 3x3 symmetric linear system A * c = rhs via Gaussian elimination with partial pivoting.
fn compare_smooth_quad_vs_c0_kink(pre: &[SamplePt], post: &[SamplePt], tau_c: f32) -> (f32, f32, f32) {
    let n_tot = pre.len() + post.len();
    if n_tot < 6 {
        return (1e9, 1e9, 0.0);
    }
    // Scale time u = tau / scale_t so conditioning is O(1)!
    let mut scale_t = 0.05f32;
    for i in 0..pre.len() {
        let a = abs_f32(pre[i].tau);
        if a > scale_t {
            scale_t = a;
        }
    }
    for i in 0..post.len() {
        let a = abs_f32(post[i].tau);
        if a > scale_t {
            scale_t = a;
        }
    }
    let inv_t = 1.0 / scale_t;
    let uc = tau_c * inv_t;

    // Basis 1: Quadratic [1, u, u^2]
    let mut mq = [[0.0f32; 3]; 3];
    let mut rqx = [0.0f32; 3];
    let mut rqy = [0.0f32; 3];
    // Basis 2: C^0 Kink at uc: [1, du, |du|] where du = u - uc (exact equal DOF = 3 per axis!)
    let mut mk = [[0.0f32; 3]; 3];
    let mut rkx = [0.0f32; 3];
    let mut rky = [0.0f32; 3];

    let mut accum_pt = |p: SamplePt| {
        let u = p.tau * inv_t;
        let u2 = u * u;
        let bq = [1.0f32, u, u2];
        let du = u - uc;
        let bk = [1.0f32, du, abs_f32(du)];
        for r in 0..3 {
            rqx[r] += bq[r] * p.wx;
            rqy[r] += bq[r] * p.wy;
            rkx[r] += bk[r] * p.wx;
            rky[r] += bk[r] * p.wy;
            for c in 0..3 {
                mq[r][c] += bq[r] * bq[c];
                mk[r][c] += bk[r] * bk[c];
            }
        }
    };
    for i in 0..pre.len() {
        accum_pt(pre[i]);
    }
    for i in 0..post.len() {
        accum_pt(post[i]);
    }

    let Some(cqx) = solve_3x3(mq, rqx) else {
        return (1e9, 1e9, 0.0);
    };
    let Some(cqy) = solve_3x3(mq, rqy) else {
        return (1e9, 1e9, 0.0);
    };
    let Some(ckx) = solve_3x3(mk, rkx) else {
        return (1e9, 1e9, 0.0);
    };
    let Some(cky) = solve_3x3(mk, rky) else {
        return (1e9, 1e9, 0.0);
    };

    let mut sse_q = 0.0f32;
    let mut sse_k = 0.0f32;
    let eval_pt = |p: SamplePt, sq: &mut f32, sk: &mut f32| {
        let u = p.tau * inv_t;
        let du = u - uc;
        let adu = abs_f32(du);
        let qx = cqx[0] + u * (cqx[1] + u * cqx[2]);
        let qy = cqy[0] + u * (cqy[1] + u * cqy[2]);
        let kx = ckx[0] + du * ckx[1] + adu * ckx[2];
        let ky = cky[0] + du * cky[1] + adu * cky[2];
        *sq += (p.wx - qx) * (p.wx - qx) + (p.wy - qy) * (p.wy - qy);
        *sk += (p.wx - kx) * (p.wx - kx) + (p.wy - ky) * (p.wy - ky);
    };
    for i in 0..pre.len() {
        eval_pt(pre[i], &mut sse_q, &mut sse_k);
    }
    for i in 0..post.len() {
        eval_pt(post[i], &mut sse_q, &mut sse_k);
    }
    let dof = (n_tot - 3).max(1) as f32;
    let rmse_q = sqrt_f32(sse_q / dof);
    let rmse_k = sqrt_f32(sse_k / dof);
    // Midpoint sag between smooth quadratic at u=uc and kink vertex (ckx[0], cky[0]):
    let qcx = cqx[0] + uc * (cqx[1] + uc * cqx[2]);
    let qcy = cqy[0] + uc * (cqy[1] + uc * cqy[2]);
    let vertex_sep = sqrt_f32((qcx - ckx[0]) * (qcx - ckx[0]) + (qcy - cky[0]) * (qcy - cky[0]));
    (rmse_q, rmse_k, vertex_sep)
}

/// Stationary-birth fallback: handles scenes where a completely stationary object ball (`t_born`)
/// wasn't seeded by motion wakeup until struck at `f_c`, while the incoming striker (`t_str`)
/// has clean pre- and post-impact rays on both sides of `f_c`.
#[inline(never)]
fn evaluate_stationary_birth_one_side(
    a: &CollisionArenaView<'_>,
    t_str: &CollisionTrackView<'_>,
    t_born: &CollisionTrackView<'_>,
    slot_str: usize,
    slot_born: usize,
    f_c: u32,
) -> CollisionPairEvent {
    let mut ev = CollisionPairEvent::zeroed();
    let born_earliest = t_born.birth_frame;
    // Born partner must have actually appeared around or shortly after f_c (never a pre-existing track)
    if born_earliest + 2 < f_c || born_earliest > f_c + 8 {
        return ev;
    }
    let s = if a.scale_s > 0.1 { a.scale_s } else { (a.frame_w / 320.0).max(0.25) };
    let s2 = s * s;
    let dt_nom = derive_nominal_dt_sec(a.frame_idx, a.hist_t_sec);
    let win_w: u32 = if dt_nom < 0.025 { 8 } else { 6 };
    let post_reach = BLUR_HALF_WIN + win_w;
    if f_c < 5 || f_c + post_reach > a.frame_idx {
        return ev;
    }
    let slot_fc = (f_c as usize) & HIST_RING_MASK;
    let t_ref_sec = a.hist_t_sec[slot_fc];
    let grav_s = effective_grav_ay(a, t_str);
    let grav_b = effective_grav_ay(a, t_born);

    let pre_start = f_c.saturating_sub(post_reach);
    let pre_end = f_c - (BLUR_HALF_WIN + 1);
    let post_start = f_c + (BLUR_HALF_WIN + 1);
    let post_end = f_c + post_reach;

    let mut bs_pre = [SamplePt { tau: 0.0, wx: 0.0, wy: 0.0 }; 10];
    let mut bs_post = [SamplePt { tau: 0.0, wx: 0.0, wy: 0.0 }; 10];
    let mut bb_post = [SamplePt { tau: 0.0, wx: 0.0, wy: 0.0 }; 10];

    let ns_pre = collect_window_samples(a, t_str, pre_start, pre_end, t_ref_sec, grav_s, &mut bs_pre);
    let ns_post = collect_window_samples(a, t_str, post_start, post_end, t_ref_sec, grav_s, &mut bs_post);
    let nb_post = collect_window_samples(a, t_born, post_start, post_end, t_ref_sec, grav_b, &mut bb_post);
    if ns_pre < 4 || ns_post < 4 || nb_post < 4 {
        return ev;
    }

    let (rs_pre, s0l, s0h) = fit_ray_with_outer_peel(&mut bs_pre, ns_pre, true, dt_nom, s);
    let (rs_post, s1l, s1h) = fit_ray_with_outer_peel(&mut bs_post, ns_post, false, dt_nom, s);
    let (rb_post, _, _) = fit_ray_with_outer_peel(&mut bb_post, nb_post, false, dt_nom, s);
    if !rs_pre.valid || !rs_post.valid || !rb_post.valid {
        return ev;
    }
    if rs_pre.rmse > 1.8 * s || rs_post.rmse > 1.8 * s || rb_post.rmse > 1.5 * s {
        return ev;
    }

    let dva_x = rs_post.vx - rs_pre.vx;
    let dva_y = rs_post.vy - rs_pre.vy;
    let dvb_x = rb_post.vx; // v_born_pre == 0
    let dvb_y = rb_post.vy;
    let dva_2 = dva_x * dva_x + dva_y * dva_y;
    let dvb_2 = dvb_x * dvb_x + dvb_y * dvb_y;
    let dva_mag = sqrt_f32(dva_2);
    let dvb_mag = sqrt_f32(dvb_2);
    if dva_mag < 35.0 * s || dvb_mag < 30.0 * s {
        return ev;
    }

    let cos_opp = (dva_x * dvb_x + dva_y * dvb_y) / (dva_mag * dvb_mag);
    if cos_opp > -0.75 {
        return ev;
    }

    let dpa_x = rs_pre.px0 - rs_post.px0;
    let dpa_y = rs_pre.py0 - rs_post.py0;
    let tau_c = (dpa_x * dva_x + dpa_y * dva_y) / dva_2;
    if abs_f32(tau_c) > 0.75 * dt_nom {
        return ev;
    }

    let pa_pre_cx = rs_pre.px0 + rs_pre.vx * tau_c;
    let pa_pre_cy = rs_pre.py0 + rs_pre.vy * tau_c;
    let pa_post_cx = rs_post.px0 + rs_post.vx * tau_c;
    let pa_post_cy = rs_post.py0 + rs_post.vy * tau_c;
    let c0_gap_a = sqrt_f32((pa_pre_cx - pa_post_cx) * (pa_pre_cx - pa_post_cx) + (pa_pre_cy - pa_post_cy) * (pa_pre_cy - pa_post_cy));
    if c0_gap_a > 6.5 * s {
        return ev;
    }

    let pa_cx = 0.5 * (pa_pre_cx + pa_post_cx);
    let pa_cy = 0.5 * (pa_pre_cy + pa_post_cy);
    let pb_cx = rb_post.px0 + rb_post.vx * tau_c;
    let pb_cy = rb_post.py0 + rb_post.vy * tau_c;
    let rc_x = pb_cx - pa_cx;
    let rc_y = pb_cy - pa_cy;
    let sep_c = sqrt_f32(rc_x * rc_x + rc_y * rc_y);
    if sep_c > DIRECT_COLLISION_RADIUS_SD * s {
        return ev;
    }

    let jx = dvb_x - dva_x;
    let jy = dvb_y - dva_y;
    let j_mag = sqrt_f32(jx * jx + jy * jy);
    if j_mag < 55.0 * s {
        return ev;
    }
    let jnx = jx / j_mag;
    let jny = jy / j_mag;
    let (nx, ny) = if sep_c >= 16.0 * s {
        let rnx = rc_x / sep_c;
        let rny = rc_y / sep_c;
        if rnx * jnx + rny * jny < 0.65 {
            return ev;
        }
        (rnx, rny)
    } else {
        // For small-radius single-sided births, post-only pb(tau_c) couples tau_c noise into
        // tangential angle via (vb+ - va_avg)_t; j_hat is tau_c-invariant.
        (jnx, jny)
    };

    let va_n_pre = rs_pre.vx * nx + rs_pre.vy * ny;
    let va_n_post = rs_post.vx * nx + rs_post.vy * ny;
    let vb_n_post = rb_post.vx * nx + rb_post.vy * ny;
    let v_close = va_n_pre; // vb_n_pre == 0
    if v_close < 20.0 * s {
        return ev;
    }
    let e_n_raw = (vb_n_post - va_n_post) / v_close;
    if e_n_raw < 0.15 || e_n_raw > 1.20 {
        return ev;
    }
    let e_n = e_n_raw.clamp(0.0, 1.0);

    let dva_n = dva_x * nx + dva_y * ny;
    let dvb_n = dvb_x * nx + dvb_y * ny;
    if dva_n > -12.0 * s || dvb_n < 12.0 * s {
        return ev;
    }

    // Smooth curve rejection on striker track
    let (q_rmse_a, k_rmse_a, _) = compare_smooth_quad_vs_c0_kink(&bs_pre[s0l..s0h], &bs_post[s1l..s1h], tau_c);
    if q_rmse_a + 0.12 * s < k_rmse_a && q_rmse_a < 1.15 * rs_pre.rmse.max(rs_post.rmse).max(0.25 * s) {
        return ev;
    }

    let min_span = rs_pre.span_sec.min(rs_post.span_sec).min(rb_post.span_sec);
    let sigma0 = (2.5 * s).max(0.08 * s / min_span.max(0.02));
    let mass_ratio = sqrt_f32((dva_n * dva_n + sigma0 * sigma0) / (dvb_n * dvb_n + sigma0 * sigma0)).clamp(0.05, 20.0);
    if mass_ratio < 0.45 || mass_ratio > 2.20 {
        return ev;
    }

    let ma = 1.0 / (1.0 + mass_ratio);
    let mb = mass_ratio * ma;
    let dp_x = ma * dva_x + mb * dvb_x;
    let dp_y = ma * dva_y + mb * dvb_y;
    let dp_mag = sqrt_f32(dp_x * dp_x + dp_y * dp_y);
    let va_pre_mag = sqrt_f32(rs_pre.vx * rs_pre.vx + rs_pre.vy * rs_pre.vy);
    let va_post_mag = sqrt_f32(rs_post.vx * rs_post.vx + rs_post.vy * rs_post.vy);
    let p_scale = 0.5 * (ma * (va_pre_mag + va_post_mag) + mb * dvb_mag) + 2.5 * s;
    let mom_resid = dp_mag / p_scale;
    if mom_resid > 0.35 {
        return ev;
    }

    let ke_pre = 0.5 * ma * (va_pre_mag * va_pre_mag);
    let ke_post = 0.5 * (ma * (va_post_mag * va_post_mag) + mb * (dvb_mag * dvb_mag));
    let energy_ret_raw = ke_post / ke_pre.max(25.0 * s2);
    if energy_ret_raw < 0.20 || energy_ret_raw > 1.20 {
        return ev;
    }

    let simult_pen = abs_f32(tau_c) / dt_nom;
    let fit_pen = k_rmse_a / s + c0_gap_a / s;
    ev.valid = 1;
    ev.chain_transfer = 0;
    ev.slot_a = slot_str as u8;
    ev.slot_b = slot_born as u8;
    ev.tid_a = t_str.track_id;
    ev.tid_b = t_born.track_id;
    ev.event_frame = f_c;
    ev.tau_c_sec = tau_c;
    ev.sep_c_px = sep_c;
    ev.nx = nx;
    ev.ny = ny;
    ev.normal_restitution_e = e_n;
    ev.mass_ratio_b_over_a = mass_ratio;
    ev.mom_resid = mom_resid;
    ev.energy_retention = energy_ret_raw.clamp(0.0, 1.10);
    ev.approach_speed_px_s = v_close;
    ev.dv_a_mag = dva_mag;
    ev.dv_b_mag = dvb_mag;
    ev.score = 0.75 * (j_mag / s) / (1.0 + 0.5 * simult_pen + 0.35 * fit_pen);
    ev
}

/// Evaluate a candidate collision of pair (slot_a, slot_b) centered at integer frame `f_c`.
#[inline(never)]
fn evaluate_pair_at_frame_ref(
    a: &CollisionArenaView<'_>,
    ta: &CollisionTrackView<'_>,
    tb: &CollisionTrackView<'_>,
    slot_a: usize,
    slot_b: usize,
    f_c: u32,
) -> CollisionPairEvent {
    let mut ev = CollisionPairEvent::zeroed();
    let dt_nom = derive_nominal_dt_sec(a.frame_idx, a.hist_t_sec);
    let win_w: u32 = if dt_nom < 0.025 { 8 } else { 6 };
    let post_reach = BLUR_HALF_WIN + win_w;
    if f_c < 5 || f_c + post_reach > a.frame_idx {
        ev.score = -1.0;
        return ev;
    }
    if (ta.last_discont_frame > 0 && f_c < ta.last_discont_frame + 5)
        || (tb.last_discont_frame > 0 && f_c < tb.last_discont_frame + 5)
    {
        ev.score = -2.0;
        return ev;
    }
    let s = if a.scale_s > 0.1 { a.scale_s } else { (a.frame_w / 320.0).max(0.25) };
    let s2 = s * s;

    let slot_fc = (f_c as usize) & HIST_RING_MASK;
    let t_ref_sec = a.hist_t_sec[slot_fc];
    let grav_a = effective_grav_ay(a, ta);
    let grav_b = effective_grav_ay(a, tb);

    let pre_start = f_c.saturating_sub(post_reach);
    let pre_end = f_c - (BLUR_HALF_WIN + 1);
    let post_start = f_c + (BLUR_HALF_WIN + 1);
    let post_end = f_c + post_reach;

    let mut buf_a_pre = [SamplePt { tau: 0.0, wx: 0.0, wy: 0.0 }; 10];
    let mut buf_a_post = [SamplePt { tau: 0.0, wx: 0.0, wy: 0.0 }; 10];
    let mut buf_b_pre = [SamplePt { tau: 0.0, wx: 0.0, wy: 0.0 }; 10];
    let mut buf_b_post = [SamplePt { tau: 0.0, wx: 0.0, wy: 0.0 }; 10];

    let na_pre = collect_window_samples(a, ta, pre_start, pre_end, t_ref_sec, grav_a, &mut buf_a_pre);
    let na_post = collect_window_samples(a, ta, post_start, post_end, t_ref_sec, grav_a, &mut buf_a_post);
    let nb_pre = collect_window_samples(a, tb, pre_start, pre_end, t_ref_sec, grav_b, &mut buf_b_pre);
    let nb_post = collect_window_samples(a, tb, post_start, post_end, t_ref_sec, grav_b, &mut buf_b_post);

    if na_pre < 3 || na_post < 3 || nb_pre < 3 || nb_post < 3 {
        if na_pre >= 4 && na_post >= 4 && nb_pre < 3 && nb_post >= 4 {
            return evaluate_stationary_birth_one_side(a, ta, tb, slot_a, slot_b, f_c);
        }
        if nb_pre >= 4 && nb_post >= 4 && na_pre < 3 && na_post >= 4 {
            return evaluate_stationary_birth_one_side(a, tb, ta, slot_b, slot_a, f_c);
        }
        ev.score = -3.0;
        return ev;
    }

    let (ra_pre, a0l, a0h) = fit_ray_with_outer_peel(&mut buf_a_pre, na_pre, true, dt_nom, s);
    let (ra_post, a1l, a1h) = fit_ray_with_outer_peel(&mut buf_a_post, na_post, false, dt_nom, s);
    let (rb_pre, b0l, b0h) = fit_ray_with_outer_peel(&mut buf_b_pre, nb_pre, true, dt_nom, s);
    let (rb_post, b1l, b1h) = fit_ray_with_outer_peel(&mut buf_b_post, nb_post, false, dt_nom, s);

    if !ra_pre.valid || !ra_post.valid || !rb_pre.valid || !rb_post.valid {
        ev.score = -4.0;
        return ev;
    }

    // Track fit quality gate (reject wild noisy glints / ID-jump glitches)
    let max_rmse = 2.5 * s;
    if ra_pre.rmse > max_rmse || ra_post.rmse > max_rmse || rb_pre.rmse > max_rmse || rb_post.rmse > max_rmse {
        ev.score = -5.0;
        return ev;
    }

    // Impulse vectors
    let dva_x = ra_post.vx - ra_pre.vx;
    let dva_y = ra_post.vy - ra_pre.vy;
    let dvb_x = rb_post.vx - rb_pre.vx;
    let dvb_y = rb_post.vy - rb_pre.vy;

    let dva_2 = dva_x * dva_x + dva_y * dva_y;
    let dvb_2 = dvb_x * dvb_x + dvb_y * dvb_y;
    let dva_mag = sqrt_f32(dva_2);
    let dvb_mag = sqrt_f32(dvb_2);
    let dv_max = if dva_mag > dvb_mag { dva_mag } else { dvb_mag };
    let dv_min = if dva_mag < dvb_mag { dva_mag } else { dvb_mag };

    // 1. Both bodies must show genuine physical velocity change (Action-Reaction floor)
    if dv_max < 28.0 * s || dv_min < (8.0 * s).max(0.05 * dv_max) {
        ev.score = -6.0;
        return ev;
    }

    // 2. Action-Reaction opposing vector impulse gate
    let cos_opp = (dva_x * dvb_x + dva_y * dvb_y) / (dva_mag * dvb_mag);
    if cos_opp > -0.65 {
        ev.score = -7.0;
        return ev;
    }

    // 3. Sub-frame collision instant tau_c and simultaneity / C^0 kink continuity
    let dpa_x = ra_pre.px0 - ra_post.px0;
    let dpa_y = ra_pre.py0 - ra_post.py0;
    let dpb_x = rb_pre.px0 - rb_post.px0;
    let dpb_y = rb_pre.py0 - rb_post.py0;

    let tau_ca = (dpa_x * dva_x + dpa_y * dva_y) / dva_2;
    let tau_cb = (dpb_x * dvb_x + dpb_y * dvb_y) / dvb_2;
    let tau_c = (dpa_x * dva_x + dpa_y * dva_y + dpb_x * dvb_x + dpb_y * dvb_y) / (dva_2 + dvb_2);

    // Enforce frame centering so f_c snaps to the nearest integer frame to t_c (avoiding early detection at f_c - 2 where +/-2 blur leaks into window!)
    let max_tau_indiv = (1.6 * dt_nom).max(0.055);
    if abs_f32(tau_ca) > max_tau_indiv || abs_f32(tau_cb) > max_tau_indiv || abs_f32(tau_c) > 0.75 * dt_nom {
        ev.score = -8.0;
        return ev;
    }
    if abs_f32(tau_ca - tau_cb) > (1.8 * dt_nom).max(0.065) {
        ev.score = -9.0;
        return ev;
    }

    // Evaluate extrapolated positions at sub-frame instant tau_c
    let pa_pre_cx = ra_pre.px0 + ra_pre.vx * tau_c;
    let pa_pre_cy = ra_pre.py0 + ra_pre.vy * tau_c;
    let pa_post_cx = ra_post.px0 + ra_post.vx * tau_c;
    let pa_post_cy = ra_post.py0 + ra_post.vy * tau_c;

    let pb_pre_cx = rb_pre.px0 + rb_pre.vx * tau_c;
    let pb_pre_cy = rb_pre.py0 + rb_pre.vy * tau_c;
    let pb_post_cx = rb_post.px0 + rb_post.vx * tau_c;
    let pb_post_cy = rb_post.py0 + rb_post.vy * tau_c;

    let c0_gap_a = sqrt_f32((pa_pre_cx - pa_post_cx) * (pa_pre_cx - pa_post_cx) + (pa_pre_cy - pa_post_cy) * (pa_pre_cy - pa_post_cy));
    let c0_gap_b = sqrt_f32((pb_pre_cx - pb_post_cx) * (pb_pre_cx - pb_post_cx) + (pb_pre_cy - pb_post_cy) * (pb_pre_cy - pb_post_cy));
    if c0_gap_a > 6.5 * s || c0_gap_b > 6.5 * s {
        ev.score = -10.0;
        return ev;
    }

    // 4. Average pre+post separation vector at tau_c pointing from A -> B
    let rc_x = 0.5 * ((pb_pre_cx - pa_pre_cx) + (pb_post_cx - pa_post_cx));
    let rc_y = 0.5 * ((pb_pre_cy - pa_pre_cy) + (pb_post_cy - pa_post_cy));
    let sep_c = sqrt_f32(rc_x * rc_x + rc_y * rc_y);

    if sep_c > CHAIN_COLLISION_RADIUS_SD * s {
        ev.score = -11.0;
        return ev;
    }

    // Relative impulse transfer vector j = dv_b - dv_a (points A -> B under normal force)
    let jx = dvb_x - dva_x;
    let jy = dvb_y - dva_y;
    let j_mag = sqrt_f32(jx * jx + jy * jy);
    if j_mag < 36.0 * s {
        ev.score = -12.0;
        return ev;
    }
    let jnx = jx / j_mag;
    let jny = jy / j_mag;

    let is_chain = sep_c > DIRECT_COLLISION_RADIUS_SD * s;
    let (nx, ny) = if sep_c >= 4.0 * s {
        let rnx = rc_x / sep_c;
        let rny = rc_y / sep_c;
        let align = rnx * jnx + rny * jny;
        // Require separation vector to align with normal impulse direction!
        // Rejects lateral lane overtaking (r_c perpendicular to dv) and crossed post-pass trajectories!
        if is_chain {
            if align < 0.94 {
                ev.score = -13.0;
                return ev;
            }
        } else if align < 0.60 {
            ev.score = -14.0;
            return ev;
        }
        (rnx, rny)
    } else {
        (jnx, jny)
    };
    let tx = -ny;
    let ty = nx;

    // 5. Normal projection & Non-penetration + Restitution gate
    let va_n_pre = ra_pre.vx * nx + ra_pre.vy * ny;
    let vb_n_pre = rb_pre.vx * nx + rb_pre.vy * ny;
    let va_n_post = ra_post.vx * nx + ra_post.vy * ny;
    let vb_n_post = rb_post.vx * nx + rb_post.vy * ny;

    let v_rel_n_pre = vb_n_pre - va_n_pre;   // must be < 0 (A closing on B along A->B normal)
    let v_rel_n_post = vb_n_post - va_n_post; // must be >= 0 (separating after impact)
    let v_close = -v_rel_n_pre;
    if v_close < 15.0 * s {
        ev.score = -15.0;
        return ev;
    }
    let e_n_raw = v_rel_n_post / v_close; // = -(v2n+ - v1n+) / (v2n- - v1n-)
    if e_n_raw < -0.15 || e_n_raw > 1.30 {
        ev.score = -16.0;
        return ev;
    }
    let e_n = e_n_raw.clamp(0.0, 1.0);

    let dva_n = dva_x * nx + dva_y * ny;
    let dvb_n = dvb_x * nx + dvb_y * ny;
    let dva_t = dva_x * tx + dva_y * ty;
    let dvb_t = dvb_x * tx + dvb_y * ty;

    // Must obey directional normal push: A decelerates along n (dva_n < 0), B accelerates along n (dvb_n > 0)
    if dva_n > -6.0 * s || dvb_n < 6.0 * s {
        ev.score = -17.0;
        return ev;
    }

    // If multi-sphere chain (36s < d <= 75s), enforce strict 1D Newton's cradle kinematics
    if is_chain {
        if e_n_raw < 0.55 {
            ev.score = -18.0;
            return ev;
        }
        if abs_f32(dva_t) > 0.25 * abs_f32(dva_n) || abs_f32(dvb_t) > 0.25 * abs_f32(dvb_n) {
            ev.score = -19.0;
            return ev;
        }
    }

    // 6. Smooth Curve vs. Genuine Dirac Kink Rejection (eliminates rockets / curving traffic / pendulums)
    let (q_rmse_a, k_rmse_a, _) = compare_smooth_quad_vs_c0_kink(&buf_a_pre[a0l..a0h], &buf_a_post[a1l..a1h], tau_c);
    let (q_rmse_b, k_rmse_b, _) = compare_smooth_quad_vs_c0_kink(&buf_b_pre[b0l..b0h], &buf_b_post[b1l..b1h], tau_c);
    // For genuine impact, C^0 kink residuals stay small and quadratic does NOT fit substantially better than kink!
    // (When smooth constant acceleration acts, q_rmse << k_rmse!)
    let quad_beats_kink_a = q_rmse_a + 0.12 * s < k_rmse_a && q_rmse_a < 1.15 * ra_pre.rmse.max(ra_post.rmse).max(0.25 * s);
    let quad_beats_kink_b = q_rmse_b + 0.12 * s < k_rmse_b && q_rmse_b < 1.15 * rb_pre.rmse.max(rb_post.rmse).max(0.25 * s);
    if quad_beats_kink_a && quad_beats_kink_b {
        ev.score = -20.0;
        return ev;
    }

    // 7. SNR-conditioned symmetric mass ratio m_b / m_a = -dv_an / dv_bn
    let min_span = ra_pre.span_sec.min(ra_post.span_sec).min(rb_pre.span_sec).min(rb_post.span_sec);
    let sigma0 = (2.5 * s).max(0.08 * s / min_span.max(0.02));
    let mass_ratio_raw = sqrt_f32((dva_n * dva_n + sigma0 * sigma0) / (dvb_n * dvb_n + sigma0 * sigma0));
    let mass_ratio = mass_ratio_raw.clamp(0.05, 20.0);

    if is_chain && (mass_ratio < 0.65 || mass_ratio > 1.55) {
        ev.score = -21.0;
        return ev;
    }

    let ma = 1.0 / (1.0 + mass_ratio);
    let mb = mass_ratio * ma;

    // 8. 2D Linear momentum conservation residual
    let dp_x = ma * dva_x + mb * dvb_x;
    let dp_y = ma * dva_y + mb * dvb_y;
    let dp_mag = sqrt_f32(dp_x * dp_x + dp_y * dp_y);
    let va_pre_mag = sqrt_f32(ra_pre.vx * ra_pre.vx + ra_pre.vy * ra_pre.vy);
    let va_post_mag = sqrt_f32(ra_post.vx * ra_post.vx + ra_post.vy * ra_post.vy);
    let vb_pre_mag = sqrt_f32(rb_pre.vx * rb_pre.vx + rb_pre.vy * rb_pre.vy);
    let vb_post_mag = sqrt_f32(rb_post.vx * rb_post.vx + rb_post.vy * rb_post.vy);
    let p_scale = 0.5 * (ma * (va_pre_mag + va_post_mag) + mb * (vb_pre_mag + vb_post_mag)) + 2.5 * s;
    let mom_resid = dp_mag / p_scale;
    if mom_resid > 0.45 {
        ev.score = -22.0;
        return ev;
    }

    // 9. Kinetic / mechanical energy retention E+ / E- at tau_c (equal U_g at tau_c cancels!)
    let ke_pre = 0.5 * (ma * (va_pre_mag * va_pre_mag) + mb * (vb_pre_mag * vb_pre_mag));
    let ke_post = 0.5 * (ma * (va_post_mag * va_post_mag) + mb * (vb_post_mag * vb_post_mag));
    let energy_ret_raw = ke_post / ke_pre.max(25.0 * s2);
    if energy_ret_raw > 1.30 {
        ev.score = -23.0;
        return ev;
    }
    let energy_ret = energy_ret_raw.clamp(0.0, 1.10);

    // 10. Event scoring to select exact center frame f_c across +/- 2 frame search
    let simult_pen = abs_f32(tau_c) / dt_nom + abs_f32(tau_ca - tau_cb) / dt_nom;
    let fit_pen = (k_rmse_a + k_rmse_b) / s + (c0_gap_a + c0_gap_b) / s;
    let score = (j_mag / s) / (1.0 + 0.5 * simult_pen + 0.35 * fit_pen);

    ev.valid = 1;
    ev.chain_transfer = if is_chain { 1 } else { 0 };
    ev.slot_a = slot_a as u8;
    ev.slot_b = slot_b as u8;
    ev.tid_a = ta.track_id;
    ev.tid_b = tb.track_id;
    ev.event_frame = f_c;
    ev.tau_c_sec = tau_c;
    ev.sep_c_px = sep_c;
    ev.nx = nx;
    ev.ny = ny;
    ev.normal_restitution_e = e_n;
    ev.mass_ratio_b_over_a = mass_ratio;
    ev.mom_resid = mom_resid;
    ev.energy_retention = energy_ret;
    ev.approach_speed_px_s = v_close;
    ev.dv_a_mag = dva_mag;
    ev.dv_b_mag = dvb_mag;
    ev.score = score;
    ev
}

fn pair_in_cooldown(state: &CollisionAnalyzerState, tid_a: u32, tid_b: u32, f_c: u32) -> bool {
    let (lo, hi) = if tid_a <= tid_b { (tid_a, tid_b) } else { (tid_b, tid_a) };
    for i in 0..8 {
        let r = state.recent_pairs[i];
        if r.tid_lo == lo && r.tid_hi == hi && r.event_frame > 0 {
            let diff = if f_c >= r.event_frame { f_c - r.event_frame } else { r.event_frame - f_c };
            if diff <= 12 {
                return true;
            }
        }
    }
    false
}

fn record_collision_event(state: &mut CollisionAnalyzerState, ev: CollisionPairEvent, detect_frame: u32) {
    let (lo, hi) = if ev.tid_a <= ev.tid_b {
        (ev.tid_a, ev.tid_b)
    } else {
        (ev.tid_b, ev.tid_a)
    };
    let c = (state.recent_cursor as usize) & 7;
    state.recent_pairs[c] = RecentCollisionPair {
        tid_lo: lo,
        tid_hi: hi,
        event_frame: ev.event_frame,
    };
    state.recent_cursor = ((c + 1) & 7) as u8;
    state.last_event = ev;

    let sa = ev.slot_a as usize;
    let sb = ev.slot_b as usize;
    if sa < MAX_TARGETS {
        state.target_last_e[sa] = ev.normal_restitution_e;
        state.target_last_frame[sa] = ev.event_frame;
        state.target_detect_frame[sa] = detect_frame;
        state.target_last_tid[sa] = ev.tid_a;
        state.target_partner_tid[sa] = ev.tid_b;
    }
    if sb < MAX_TARGETS {
        state.target_last_e[sb] = ev.normal_restitution_e;
        state.target_last_frame[sb] = ev.event_frame;
        state.target_detect_frame[sb] = detect_frame;
        state.target_last_tid[sb] = ev.tid_b;
        state.target_partner_tid[sb] = ev.tid_a;
    }
}

fn fit_obs_history_slice(
    obs: &SlotObsHistory,
    k_start: usize,
    k_end: usize,
    t_ref: f64,
    grav_ay: f32,
    dt_nom: f32,
) -> OlsRay {
    let n_stored = (obs.vis_streak as usize).min(8);
    let hi = k_end.min(n_stored);
    if k_start >= hi || hi - k_start < 3 {
        return OlsRay {
            valid: false,
            px0: 0.0,
            py0: 0.0,
            vx: 0.0,
            vy: 0.0,
            rmse: 1e9,
            span_sec: 0.0,
        };
    }
    let base = if obs.vis_streak > 8 {
        obs.head as usize
    } else {
        0
    };
    let mut pts = [SamplePt {
        tau: 0.0,
        wx: 0.0,
        wy: 0.0,
    }; 8];
    let mut cnt = 0usize;
    let mut k = k_start;
    while k < hi && cnt < 8 {
        let idx = (base + k) & 7;
        let tau = (obs.t_sec[idx] - t_ref) as f32;
        let wx = obs.wx[idx];
        let wy = obs.wy[idx] - 0.5 * grav_ay * tau * tau;
        pts[cnt] = SamplePt { tau, wx, wy };
        cnt += 1;
        k += 1;
    }
    fit_ols_line(&pts[..cnt], dt_nom)
}

#[inline(always)]
fn expire_pairs_for_slot_or_tid(
    pairs: &mut [OcclPairTrackState; 8],
    slot_m: usize,
    tid_old: u32,
    tid_new: u32,
) {
    for p in 0..8 {
        let pair = &mut pairs[p];
        if pair.active != 0
            && (pair.slot_a as usize == slot_m
                || pair.slot_b as usize == slot_m
                || (tid_old != 0 && (pair.tid_a == tid_old || pair.tid_b == tid_old))
                || (tid_new != 0 && (pair.tid_a == tid_new || pair.tid_b == tid_new)))
        {
            pair.reset_in_place();
        }
    }
}

#[inline(never)]
fn update_live_occluded_pairs(
    arena: &CollisionArenaView<'_>,
    tracks: &[CollisionTrackView<'_>; 16],
    cons: &mut OccludedConservationState,
    s: f32,
    dt_nom: f32,
) {
    if cons.has_last_frame != 0 && arena.frame_idx == cons.last_frame_idx {
        return;
    }

    let mut max_discont: u32 = 0;
    for m in 0..MAX_TARGETS {
        if tracks[m].last_discont_frame > max_discont {
            max_discont = tracks[m].last_discont_frame;
        }
    }
    if (cons.has_last_frame != 0 && arena.frame_idx != cons.last_frame_idx.wrapping_add(1))
        || (cons.last_continuity_epoch != 0
            && arena.continuity_epoch != cons.last_continuity_epoch)
        || (max_discont > 0
            && (max_discont != cons.last_discont_seen || max_discont == arena.frame_idx))
    {
        cons.reset_continuity();
        cons.last_discont_seen = max_discont;
    }
    cons.last_continuity_epoch = arena.continuity_epoch;
    cons.has_last_frame = 1;
    cons.last_frame_idx = arena.frame_idx;

    let f_slot = (arena.frame_idx as usize) & HIST_RING_MASK;
    let t_now = arena.hist_t_sec[f_slot];
    let cam_now_x = arena.hist_cam_x[f_slot];
    let cam_now_y = arena.hist_cam_y[f_slot];

    // Capture live unoccluded observations once per committed frame, keyed by (track_id, orig_birth_frame).
    for m in 0..MAX_TARGETS {
        let tm = &tracks[m];
        if tm.active == 0 || tm.confirmed == 0 || tm.track_id == 0 {
            if cons.slot_obs[m].active != 0 {
                let old_tid = cons.slot_obs[m].tid;
                expire_pairs_for_slot_or_tid(&mut cons.track_pairs, m, old_tid, 0);
                cons.slot_obs[m].reset_in_place();
            }
            continue;
        }

        if cons.slot_obs[m].active == 0
            || cons.slot_obs[m].tid != tm.track_id
            || cons.slot_obs[m].orig_birth != tm.orig_birth_frame
        {
            let old_tid = cons.slot_obs[m].tid;
            expire_pairs_for_slot_or_tid(&mut cons.track_pairs, m, old_tid, tm.track_id);
            cons.slot_obs[m].reset_in_place();
            cons.slot_obs[m].tid = tm.track_id;
            cons.slot_obs[m].orig_birth = tm.orig_birth_frame;
            cons.slot_obs[m].active = 1;
            cons.slot_obs[m].prev_coasting = tm.coasting;
        }

        if tm.coasting == 0 {
            if cons.slot_obs[m].prev_coasting > 0
                || (cons.slot_obs[m].vis_streak > 0
                    && arena.frame_idx != cons.slot_obs[m].last_vis_frame.wrapping_add(1))
            {
                cons.slot_obs[m].vis_streak = 0;
                cons.slot_obs[m].head = 0;
            }
            let wx = tm.arc_raw_x[f_slot] + cam_now_x;
            let wy = tm.arc_raw_y[f_slot] + cam_now_y;
            if tm.arc_vis[f_slot] != 0 && wx.is_finite() && wy.is_finite() && t_now.is_finite() {
                let h = (cons.slot_obs[m].head as usize) & 7;
                cons.slot_obs[m].t_sec[h] = t_now;
                cons.slot_obs[m].wx[h] = wx;
                cons.slot_obs[m].wy[h] = wy;
                cons.slot_obs[m].head = ((h + 1) & 7) as u8;
                cons.slot_obs[m].vis_streak = cons.slot_obs[m].vis_streak.saturating_add(1);
                cons.slot_obs[m].last_vis_frame = arena.frame_idx;
            }
        }
        cons.slot_obs[m].prev_coasting = tm.coasting;
    }

    // Expire retired, mismatched, or timed-out pending pairs.
    for p in 0..8 {
        let pair = &mut cons.track_pairs[p];
        if pair.active == 0 {
            continue;
        }
        let sa = pair.slot_a as usize;
        let sb = pair.slot_b as usize;
        let valid_slots = sa < MAX_TARGETS
            && sb < MAX_TARGETS
            && sa != sb
            && tracks[sa].active != 0
            && tracks[sa].confirmed != 0
            && tracks[sb].active != 0
            && tracks[sb].confirmed != 0
            && tracks[sa].track_id == pair.tid_a
            && tracks[sb].track_id == pair.tid_b
            && tracks[sa].orig_birth_frame == pair.orig_birth_a
            && tracks[sb].orig_birth_frame == pair.orig_birth_b;
        let timed_out = arena.frame_idx.saturating_sub(pair.last_seen_frame) > 90
            || (pair.occl_seen_mask != 0
                && arena.frame_idx.saturating_sub(pair.occl_start_frame) > 90);
        if !valid_slots || timed_out {
            pair.reset_in_place();
        }
    }

    let prox_r = PAIR_SCAN_PROX_SD * s;
    let prox_r2 = prox_r * prox_r;
    let far_r2 = (1.35 * prox_r) * (1.35 * prox_r);

    for i in 0..MAX_TARGETS {
        let ti = &tracks[i];
        if ti.active == 0 || ti.confirmed == 0 || ti.track_id == 0 {
            continue;
        }
        for j in (i + 1)..MAX_TARGETS {
            let tj = &tracks[j];
            if tj.active == 0 || tj.confirmed == 0 || tj.track_id == 0 {
                continue;
            }
            let (sa, sb) = if ti.track_id <= tj.track_id {
                (i, j)
            } else {
                (j, i)
            };
            let ta = &tracks[sa];
            let tb = &tracks[sb];

            let mut found_idx: Option<usize> = None;
            for p in 0..8 {
                let pr = &cons.track_pairs[p];
                if pr.active != 0 && pr.tid_a == ta.track_id && pr.tid_b == tb.track_id {
                    found_idx = Some(p);
                    break;
                }
            }

            let ca = ta.coasting;
            let cb = tb.coasting;
            let g_pair = if arena.scene_gravity_conf >= 0.45
                && (ta.is_ballistic != 0 || tb.is_ballistic != 0)
                && ta.pivot_valid == 0
                && tb.pivot_valid == 0
            {
                arena.scene_gravity_px_s2
            } else {
                0.0
            };

            if ca == 0 && cb == 0 {
                let ax = ta.arc_raw_x[f_slot] + cam_now_x;
                let ay = ta.arc_raw_y[f_slot] + cam_now_y;
                let bx = tb.arc_raw_x[f_slot] + cam_now_x;
                let by = tb.arc_raw_y[f_slot] + cam_now_y;
                let dist2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay);

                let Some(p_idx) = found_idx else {
                    if dist2 <= prox_r2 && ta.arc_vis[f_slot] != 0 && tb.arc_vis[f_slot] != 0 {
                        let mut alloc_p = None;
                        for p in 0..8 {
                            if cons.track_pairs[p].active == 0 {
                                alloc_p = Some(p);
                                break;
                            }
                        }
                        if alloc_p.is_none() {
                            let mut best_evict = 0usize;
                            let mut best_score = u64::MAX;
                            for p in 0..8 {
                                let pr = &cons.track_pairs[p];
                                let prio: u64 = if pr.both_occluded != 0 {
                                    2u64 << 40
                                } else if pr.has_pre != 0 {
                                    1u64 << 40
                                } else {
                                    0
                                };
                                let sc = prio | (pr.last_seen_frame as u64);
                                if sc < best_score {
                                    best_score = sc;
                                    best_evict = p;
                                }
                            }
                            alloc_p = Some(best_evict);
                        }
                        if let Some(p) = alloc_p {
                            let pr = &mut cons.track_pairs[p];
                            pr.reset_in_place();
                            pr.tid_a = ta.track_id;
                            pr.tid_b = tb.track_id;
                            pr.orig_birth_a = ta.orig_birth_frame;
                            pr.orig_birth_b = tb.orig_birth_frame;
                            pr.slot_a = sa as u8;
                            pr.slot_b = sb as u8;
                            pr.active = 1;
                            pr.start_frame = arena.frame_idx;
                            pr.last_seen_frame = arena.frame_idx;
                        }
                    }
                    continue;
                };

                cons.track_pairs[p_idx].slot_a = sa as u8;
                cons.track_pairs[p_idx].slot_b = sb as u8;
                cons.track_pairs[p_idx].last_seen_frame = arena.frame_idx;

                if cons.track_pairs[p_idx].both_occluded != 0 {
                    if cons.slot_obs[sa].vis_streak >= 5 && cons.slot_obs[sb].vis_streak >= 5 {
                        let na = (cons.slot_obs[sa].vis_streak as usize).min(8);
                        let nb = (cons.slot_obs[sb].vis_streak as usize).min(8);
                        let ka0 = if cons.track_pairs[p_idx].max_coast_a == 0 {
                            na.saturating_sub(4)
                        } else if cons.slot_obs[sa].vis_streak <= 8 {
                            1
                        } else {
                            na.saturating_sub(6)
                        };
                        let kb0 = if cons.track_pairs[p_idx].max_coast_b == 0 {
                            nb.saturating_sub(4)
                        } else if cons.slot_obs[sb].vis_streak <= 8 {
                            1
                        } else {
                            nb.saturating_sub(6)
                        };
                        let ra_post = fit_obs_history_slice(
                            &cons.slot_obs[sa],
                            ka0,
                            na,
                            t_now,
                            g_pair,
                            dt_nom,
                        );
                        let rb_post = fit_obs_history_slice(
                            &cons.slot_obs[sb],
                            kb0,
                            nb,
                            t_now,
                            g_pair,
                            dt_nom,
                        );
                        let occl_dt = (t_now - cons.track_pairs[p_idx].pre_t_sec) as f32;
                        if ra_post.valid
                            && rb_post.valid
                            && ra_post.rmse <= 4.0 * s
                            && rb_post.rmse <= 4.0 * s
                            && occl_dt > dt_nom
                        {
                            let rmse_max = cons.track_pairs[p_idx]
                                .pre_rmse
                                .max(ra_post.rmse)
                                .max(rb_post.rmse);
                            let sigma_pos = max_f32(1.2 * s, rmse_max);
                            let sigma_vel = clamp_f32(
                                max_f32(rmse_max, 0.35 * s) / max_f32(1.5 * dt_nom, 0.015),
                                4.0 * s,
                                35.0 * s,
                            );
                            let pr = cons.track_pairs[p_idx];
                            let mut live_in = [0.0f32; CONSERVATION_INPUT_LEN];
                            live_in[0] = pr.pre_x1;
                            live_in[1] = pr.pre_y1;
                            live_in[2] = pr.pre_vx1;
                            live_in[3] = pr.pre_vy1;
                            live_in[4] = 1.0;
                            live_in[5] = 6.0 * s;
                            live_in[6] = pr.pre_x2;
                            live_in[7] = pr.pre_y2;
                            live_in[8] = pr.pre_vx2;
                            live_in[9] = pr.pre_vy2;
                            live_in[10] = 1.0;
                            live_in[11] = 6.0 * s;
                            live_in[12] = ra_post.px0;
                            live_in[13] = ra_post.py0;
                            live_in[14] = ra_post.vx;
                            live_in[15] = ra_post.vy;
                            live_in[16] = rb_post.px0;
                            live_in[17] = rb_post.py0;
                            live_in[18] = rb_post.vx;
                            live_in[19] = rb_post.vy;
                            live_in[20] = occl_dt;
                            live_in[21] = g_pair;
                            live_in[22] = 0.05;
                            live_in[23] = 2.0 * s;
                            live_in[24] = 2.0 * s;
                            live_in[25] = sigma_pos;
                            live_in[26] = sigma_vel;
                            live_in[27] = pr.tid_a as f32;
                            live_in[28] = pr.tid_b as f32;
                            live_in[29] = pr.occl_start_frame as f32;
                            live_in[30] = 0.0;
                            live_in[31] = 0.0;

                            let st_code = eval_occluded_pair_core(&live_in, &mut cons.telem);
                            if st_code != OCCL_STATUS_NONE {
                                let ns = cons.live_event_seq.wrapping_add(1);
                                cons.live_event_seq = if ns == 0 { 1 } else { ns };
                            }
                        }

                        let pr = &mut cons.track_pairs[p_idx];
                        pr.both_occluded = 0;
                        pr.occl_seen_mask = 0;
                        pr.has_pre = 0;
                        pr.max_coast_a = 0;
                        pr.max_coast_b = 0;
                        pr.co_coast_frames = 0;
                        pr.cooldown_until = arena.frame_idx.saturating_add(12);
                    }
                } else if dist2 > far_r2 {
                    cons.track_pairs[p_idx].reset_in_place();
                } else {
                    cons.track_pairs[p_idx].occl_seen_mask = 0;
                    cons.track_pairs[p_idx].max_coast_a = 0;
                    cons.track_pairs[p_idx].max_coast_b = 0;
                    if arena.frame_idx >= cons.track_pairs[p_idx].cooldown_until
                        && cons.slot_obs[sa].vis_streak >= 4
                        && cons.slot_obs[sb].vis_streak >= 4
                    {
                        let na = (cons.slot_obs[sa].vis_streak as usize).min(8);
                        let nb = (cons.slot_obs[sb].vis_streak as usize).min(8);
                        let use_a = if na >= 5 { na - 1 } else { na };
                        let use_b = if nb >= 5 { nb - 1 } else { nb };
                        let ref_f = if na >= 5 && nb >= 5 {
                            arena.frame_idx.saturating_sub(1)
                        } else {
                            arena.frame_idx
                        };
                        let t_pre_ref = arena.hist_t_sec[(ref_f as usize) & HIST_RING_MASK];
                        let ra_pre = fit_obs_history_slice(
                            &cons.slot_obs[sa],
                            0,
                            use_a,
                            t_pre_ref,
                            g_pair,
                            dt_nom,
                        );
                        let rb_pre = fit_obs_history_slice(
                            &cons.slot_obs[sb],
                            0,
                            use_b,
                            t_pre_ref,
                            g_pair,
                            dt_nom,
                        );
                        if ra_pre.valid
                            && rb_pre.valid
                            && ra_pre.rmse <= 3.5 * s
                            && rb_pre.rmse <= 3.5 * s
                        {
                            let pr = &mut cons.track_pairs[p_idx];
                            pr.pre_x1 = ra_pre.px0;
                            pr.pre_y1 = ra_pre.py0;
                            pr.pre_vx1 = ra_pre.vx;
                            pr.pre_vy1 = ra_pre.vy;
                            pr.pre_x2 = rb_pre.px0;
                            pr.pre_y2 = rb_pre.py0;
                            pr.pre_vx2 = rb_pre.vx;
                            pr.pre_vy2 = rb_pre.vy;
                            pr.pre_t_sec = t_pre_ref;
                            pr.pre_rmse = max_f32(ra_pre.rmse, rb_pre.rmse);
                            pr.has_pre = 1;
                        }
                    }
                }
            } else if let Some(p_idx) = found_idx {
                let pr = &mut cons.track_pairs[p_idx];
                pr.slot_a = sa as u8;
                pr.slot_b = sb as u8;
                if pr.has_pre != 0 {
                    pr.last_seen_frame = arena.frame_idx;
                    if pr.occl_seen_mask == 0 {
                        pr.occl_start_frame = arena.frame_idx;
                    }
                    if ca > 0 {
                        pr.occl_seen_mask |= 1;
                        if ca > pr.max_coast_a {
                            pr.max_coast_a = ca;
                        }
                    }
                    if cb > 0 {
                        pr.occl_seen_mask |= 2;
                        if cb > pr.max_coast_b {
                            pr.max_coast_b = cb;
                        }
                    }
                    if ca > 0 && cb > 0 {
                        pr.co_coast_frames = pr.co_coast_frames.saturating_add(1);
                    }
                    let both_coast_ok = pr.co_coast_frames >= 1
                        && pr.occl_seen_mask == 3
                        && (pr.max_coast_a >= 2 || pr.max_coast_b >= 2);
                    if both_coast_ok {
                        pr.both_occluded = 1;
                    }
                }
            }
        }
    }
}

#[inline(never)]
pub fn analyze_two_body_collisions(
    arena: &CollisionArenaView<'_>,
    tracks: &[CollisionTrackView<'_>; 16],
    st: &mut CollisionAnalyzerState,
) -> CollisionTickOutput {
    let mut out = CollisionTickOutput::zeroed();
    let s = if arena.scale_s > 0.1 {
        arena.scale_s
    } else {
        (arena.frame_w / 320.0).max(0.25)
    };
    let disp_f = arena.frame_idx.saturating_sub(arena.lookahead_k as u32);
    let dt_nom = derive_nominal_dt_sec(arena.frame_idx, arena.hist_t_sec);
    let win_w: u32 = if dt_nom < 0.025 { 8 } else { 6 };
    let post_reach: u32 = BLUR_HALF_WIN + win_w;

    // 1. Refresh target slot identity & clear discontinuity-reset slots
    for m in 0..MAX_TARGETS {
        let tm = &tracks[m];
        if tm.last_discont_frame > 0 && arena.frame_idx == tm.last_discont_frame {
            reset_collision_slot(st, m);
            st.target_last_tid[m] = tm.track_id;
        } else if tm.active != 0 && st.target_last_tid[m] != 0 && st.target_last_tid[m] != tm.track_id {
            reset_collision_slot(st, m);
            st.target_last_tid[m] = tm.track_id;
        }
    }

    // 2. Scan causal band strictly satisfying f_c + 2 + win_w < frame_idx
    if arena.frame_idx >= post_reach + 6 {
        let f_hi = arena.frame_idx - post_reach;
        let f_lo = arena.frame_idx.saturating_sub(post_reach + 7).max(6);
        let scan_prox2 = (PAIR_SCAN_PROX_SD * s) * (PAIR_SCAN_PROX_SD * s);
        let mut best_ev = CollisionPairEvent::zeroed();

        for i in 0..MAX_TARGETS {
            let ti = &tracks[i];
            if ti.active == 0 || ti.confirmed == 0 {
                continue;
            }
            for j in (i + 1)..MAX_TARGETS {
                let tj = &tracks[j];
                if tj.active == 0 || tj.confirmed == 0 {
                    continue;
                }
                let mut near = false;
                let mut f = f_lo;
                let scan_hi = (f_hi + BLUR_HALF_WIN).min(arena.frame_idx);
                while f <= scan_hi {
                    let sf = (f as usize) & HIST_RING_MASK;
                    if ti.arc_vis[sf] != 0 && tj.arc_vis[sf] != 0 {
                        let dx = tj.arc_raw_x[sf] - ti.arc_raw_x[sf];
                        let dy = tj.arc_raw_y[sf] - ti.arc_raw_y[sf];
                        if dx * dx + dy * dy <= scan_prox2 {
                            near = true;
                            break;
                        }
                    }
                    f += 1;
                }
                if !near {
                    let dfx = tj.disp_world_x - ti.disp_world_x;
                    let dfy = tj.disp_world_y - ti.disp_world_y;
                    if dfx * dfx + dfy * dfy > (1.5 * PAIR_SCAN_PROX_SD * s) * (1.5 * PAIR_SCAN_PROX_SD * s) {
                        continue;
                    }
                }

                let mut pair_best = CollisionPairEvent::zeroed();
                let mut fc = f_lo;
                while fc <= f_hi {
                    if !pair_in_cooldown(st, ti.track_id, tj.track_id, fc) {
                        let ev = evaluate_pair_at_frame_ref(arena, ti, tj, i, j, fc);
                        if ev.valid != 0 && (pair_best.valid == 0 || ev.score > pair_best.score) {
                            pair_best = ev;
                        }
                    }
                    fc += 1;
                }
                if pair_best.valid != 0 && pair_best.event_frame < f_hi && (best_ev.valid == 0 || pair_best.score > best_ev.score) {
                    best_ev = pair_best;
                }
            }
        }
        if best_ev.valid != 0 {
            record_collision_event(st, best_ev, arena.frame_idx);
            out.pair_event = best_ev;
            out.has_pair_event = 1;
        }
    }

    // 3. Populate per-target outputs and locked top-level arrays
    for m in 0..MAX_TARGETS {
        let tm = &tracks[m];
        if tm.active == 0 || tm.confirmed == 0 {
            continue;
        }
        if st.target_last_tid[m] == tm.track_id && st.target_last_frame[m] > 0 {
            let fc = st.target_last_frame[m];
            let df = st.target_detect_frame[m];
            let e_latched = st.target_last_e[m];
            out.targets[m].normal_restitution_e = e_latched;
            out.restitution_e[m] = e_latched;
            out.targets[m].partner_tid = st.target_partner_tid[m];
            out.targets[m].last_collision_frame = fc;
            let in_disp_jump = disp_f.saturating_add(1) >= fc && disp_f <= fc.saturating_add(12);
            let in_causal_jump = df > 0 && arena.frame_idx >= df && arena.frame_idx < df.saturating_add(12);
            if in_disp_jump || in_causal_jump {
                out.targets[m].constraint_jump_code = CONSTRAINT_JUMP_TWO_BODY;
                out.targets[m].active_collision = 1;
                out.jump_code[m] = CONSTRAINT_JUMP_TWO_BODY;
            }
            let in_disp_reg = disp_f.saturating_add(2) >= fc && disp_f <= fc.saturating_add(6);
            let in_causal_reg = df > 0 && arena.frame_idx >= df && arena.frame_idx < df.saturating_add(5);
            if in_disp_reg || in_causal_reg {
                out.targets[m].phys_regime_override = PHYS_REGIME_IMPULSIVE;
                out.regime_override[m] = PHYS_REGIME_IMPULSIVE;
            }
        }
    }

    // 4. Ensemble specific kinetic & mechanical energy across active confirmed targets
    let h_ref = arena.frame_h;
    let g_scene = if arena.scene_gravity_conf >= 0.45 {
        arena.scene_gravity_px_s2
    } else {
        0.0
    };
    let mut k_ens = 0.0f32;
    let mut u_ens = 0.0f32;
    for m in 0..MAX_TARGETS {
        let tm = &tracks[m];
        if tm.active == 0 || tm.confirmed == 0 {
            continue;
        }
        let v2 = tm.disp_vx * tm.disp_vx + tm.disp_vy * tm.disp_vy;
        k_ens += 0.5 * v2;
        if g_scene > 0.0 || g_scene < 0.0 {
            let height = h_ref - tm.disp_world_y;
            u_ens += g_scene * height;
        }
    }
    out.ensemble_kinetic_px2_s2 = k_ens;
    out.ensemble_mech_energy_px2_s2 = k_ens + u_ens;

    if st.last_event.valid != 0 {
        out.last_pair_mass_ratio = st.last_event.mass_ratio_b_over_a;
        out.last_pair_mom_resid = st.last_event.mom_resid;
        out.last_pair_energy_retention = st.last_event.energy_retention;
        out.last_pair_tid_a = st.last_event.tid_a as f32;
        out.last_pair_tid_b = st.last_event.tid_b as f32;
        out.last_pair_frame = st.last_event.event_frame as f32;
    }

    update_live_occluded_pairs(arena, tracks, &mut st.conservation, s, dt_nom);

    out
}

