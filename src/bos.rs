//! Controlled Tripod Background-Oriented Schlieren (`BOS`) Engine.
//!
//! Pure `#![no_std]` zero-allocation, zero-import implementation inside static `BosState`:
//! - `16`-frame clear running-mean reference warmup + frozen spatial anchor guard (`>= 8` patches).
//! - Dense `128x64` (`N = 8192`) `8x8` affine-brightness-invariant IC-LK (`<= 2` iters) with
//!   marginal variance `var_x, var_y` (`64 KiB`), saturation/occlusion/aperture gating.
//! - Mode `1` permanently frozen reference; Mode `2` selective clear-region maintenance (`120` clear
//!   samples, outside frozen anchor halos, `alpha = 1/4096`).
//! - Exact plaquette curl integrability diagnostic (`curl_rms`, `max_abs_curl`).
//! - Variational inhomogeneous-Neumann Poisson solver (`L = A^T W A`, `b = A^T W eps_bar`):
//!   - Direct path: separable orthonormal `128x64` DCT-II/III via Makhoul `N/2`-pt complex FFT.
//!   - General path: BFS positive-edge component labeling + `f64` DCT-preconditioned PCG (`<= 16` iters)
//!     with true relative residual verification (`<= 1e-6`) and max-overlap gauge-ROI pinning.
//! - Outputs apparent refractive displacement (`px`) and relative potential `Phi` (`px^2`); never
//!   fabricates uncalibrated gas density or temperature without external optical/Gladstone-Dale constants.

use crate::simd::{
    f32x4, f32x4_add, f32x4_extract_lane, f32x4_floor, f32x4_gt, f32x4_lt, f32x4_max, f32x4_min,
    f32x4_mul, f32x4_splat, f32x4_sqrt, f32x4_sub, f64x2, f64x2_add, f64x2_extract_lane,
    f64x2_mul, f64x2_splat, f64x2_sqrt, f64x2_sub, i32x4_bitmask, v128, v128_and,
};

pub const BOS_GRID_W: usize = 128;
pub const BOS_GRID_H: usize = 64;
pub const BOS_NODES: usize = BOS_GRID_W * BOS_GRID_H; // 8,192
pub const BOS_MAX_FRAME_W: usize = 640;
pub const BOS_MAX_FRAME_H: usize = 360;
pub const BOS_MAX_PIXELS: usize = BOS_MAX_FRAME_W * BOS_MAX_FRAME_H; // 230,400
pub const BOS_TELEM_LEN: usize = 32;
pub const BOS_PLANES_LEN: usize = 6 * BOS_NODES; // 49,152

pub const BOS_PLANE_EPS_X: usize = 0;
pub const BOS_PLANE_EPS_Y: usize = 1;
pub const BOS_PLANE_PHI: usize = 2;
pub const BOS_PLANE_VAR_X: usize = 3;
pub const BOS_PLANE_VAR_Y: usize = 4;
pub const BOS_PLANE_CURL: usize = 5;

pub const BOS_VALID_DISP: u8 = 1 << 0;
pub const BOS_VALID_PHI: u8 = 1 << 1;
pub const BOS_VALID_CURL: u8 = 1 << 2;

pub const BOS_MODE_OFF: u32 = 0;
pub const BOS_MODE_FROZEN: u32 = 1;
pub const BOS_MODE_MAINT: u32 = 2;

pub const BOS_STATE_OFF: u32 = 0;
pub const BOS_STATE_WARMING: u32 = 1;
pub const BOS_STATE_READY: u32 = 2;
pub const BOS_STATE_DISTURBED: u32 = 3;

pub const BOS_REASON_ANCHOR_SUPPORT: u32 = 1 << 0;
pub const BOS_REASON_EXT_PAN_CUT: u32 = 1 << 1;
pub const BOS_REASON_INT_SHIFT: u32 = 1 << 2;
pub const BOS_REASON_PHOTOM_MSE: u32 = 1 << 3;
pub const BOS_REASON_DISCONT: u32 = 1 << 4;
pub const BOS_REASON_NO_GAUGE: u32 = 1 << 5;
pub const BOS_REASON_NONCONV: u32 = 1 << 6;
pub const BOS_REASON_LOW_TEX: u32 = 1 << 7;

pub const BOS_OVERLAY_MAG: u32 = 0;
pub const BOS_OVERLAY_PHI: u32 = 1;
pub const BOS_OVERLAY_CURL: u32 = 2;

pub const BOS_SOLVER_NONE: u32 = 0;
pub const BOS_SOLVER_DIRECT: u32 = 1;
pub const BOS_SOLVER_PCG: u32 = 2;

const BOS_WARMUP_FRAMES: u32 = 16;
const BOS_MAINT_STREAK_REQ: u8 = 120;
const BOS_MAINT_ALPHA: f32 = 1.0 / 4096.0;
const BOS_MAX_ANCHORS: usize = 16;
const BOS_MIN_ANCHORS: usize = 8;
const BOS_PCG_MAX_ITERS: u32 = 16;

const GAP_BIT_SEEK: u32 = 1 << 3;
const GAP_BIT_GEOM: u32 = 1 << 4;
const GAP_BIT_CUT: u32 = 1 << 5;
#[allow(dead_code)]
const GAP_BIT_WRAP: u32 = 1 << 6;
// Clean 32-bit counter wrap (GAP_BIT_WRAP = 0x40) preserves the frozen BOS reference.
const GAP_RESET_MASK: u32 = GAP_BIT_SEEK | GAP_BIT_GEOM | GAP_BIT_CUT;

const PI_64: f64 = 3.14159265358979323846;
const TWO_PI_64: f64 = 6.28318530717958647692;
const HALF_PI_64: f64 = 1.57079632679489661923;
const PI_32: f32 = 3.14159265358979323846;
const TWO_PI_32: f32 = 6.28318530717958647692;
const HALF_PI_32: f32 = 1.57079632679489661923;

#[inline(always)]
fn sqrt_f32(x: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
}

#[inline(always)]
fn floor_f32(x: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_floor(f32x4_splat(x)))
}

#[inline(always)]
fn sqrt_f64(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

#[inline(always)]
fn sin_f32(x: f32) -> f32 {
    let mut a = x;
    if a > 32.0 || a < -32.0 {
        let turns = floor_f32((a + PI_32) * (1.0 / TWO_PI_32));
        a -= turns * TWO_PI_32;
    }
    while a > PI_32 {
        a -= TWO_PI_32;
    }
    while a < -PI_32 {
        a += TWO_PI_32;
    }
    if a > HALF_PI_32 {
        a = PI_32 - a;
    } else if a < -HALF_PI_32 {
        a = -PI_32 - a;
    }
    let a2 = a * a;
    a * (1.0 + a2 * (-0.16666657 + a2 * (0.008333025 - a2 * 0.000198074)))
}

#[inline(always)]
fn cos_f32(x: f32) -> f32 {
    sin_f32(x + HALF_PI_32)
}

const fn const_cos_f64(x: f64) -> f64 {
    let mut a = x;
    while a > PI_64 {
        a -= TWO_PI_64;
    }
    while a < -PI_64 {
        a += TWO_PI_64;
    }
    if a < 0.0 {
        a = -a;
    }
    let sign = if a > HALF_PI_64 {
        a = PI_64 - a;
        -1.0
    } else {
        1.0
    };
    let x2 = a * a;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1usize;
    while k <= 12 {
        let d = ((2 * k - 1) as f64) * ((2 * k) as f64);
        term = -term * x2 / d;
        sum += term;
        k += 1;
    }
    sign * sum
}

const fn const_sin_f64(x: f64) -> f64 {
    const_cos_f64(x - HALF_PI_64)
}

const fn const_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = if x > 1.0 { x * 0.5 } else { 1.0 };
    let mut i = 0usize;
    while i < 24 {
        y = 0.5 * (y + x / y);
        i += 1;
    }
    y
}

struct DctTables {
    bitrev64: [u8; 64],
    bitrev32: [u8; 32],
    tw64_re: [f64; 64],
    tw64_im: [f64; 64],
    wn128_re: [f64; 64],
    wn128_im: [f64; 64],
    wd128_re: [f64; 64],
    wd128_im: [f64; 64],
    sin2_x: [f64; 128],
    sin2_y: [f64; 64],
    sqrt_128: f64,
    sqrt_64: f64,
    scale_fwd_128: f64,
    scale_fwd_64: f64,
    scale_dc_128: f64,
    scale_dc_64: f64,
    scale_nyq_128: f64,
    scale_nyq_64: f64,
    inv_32: f64,
    inv_64: f64,
}

impl DctTables {
    const fn new() -> Self {
        let mut bitrev64 = [0u8; 64];
        let mut i = 0usize;
        while i < 64 {
            let mut r = 0usize;
            let mut b = 0usize;
            while b < 6 {
                r = (r << 1) | ((i >> b) & 1);
                b += 1;
            }
            bitrev64[i] = r as u8;
            i += 1;
        }
        let mut bitrev32 = [0u8; 32];
        let mut i = 0usize;
        while i < 32 {
            let mut r = 0usize;
            let mut b = 0usize;
            while b < 5 {
                r = (r << 1) | ((i >> b) & 1);
                b += 1;
            }
            bitrev32[i] = r as u8;
            i += 1;
        }

        let mut tw64_re = [0.0f64; 64];
        let mut tw64_im = [0.0f64; 64];
        let mut wn128_re = [0.0f64; 64];
        let mut wn128_im = [0.0f64; 64];
        let mut wd128_re = [0.0f64; 64];
        let mut wd128_im = [0.0f64; 64];
        let mut k = 0usize;
        while k < 64 {
            let ang64 = -TWO_PI_64 * (k as f64) / 64.0;
            tw64_re[k] = const_cos_f64(ang64);
            tw64_im[k] = const_sin_f64(ang64);

            let ang128 = -TWO_PI_64 * (k as f64) / 128.0;
            wn128_re[k] = const_cos_f64(ang128);
            wn128_im[k] = const_sin_f64(ang128);

            let ang_d = -PI_64 * (k as f64) / 256.0;
            wd128_re[k] = const_cos_f64(ang_d);
            wd128_im[k] = const_sin_f64(ang_d);
            k += 1;
        }

        let mut sin2_x = [0.0f64; 128];
        let mut k = 0usize;
        while k < 128 {
            let s = const_sin_f64(PI_64 * (k as f64) / 256.0);
            sin2_x[k] = 4.0 * s * s;
            k += 1;
        }
        let mut sin2_y = [0.0f64; 64];
        let mut l = 0usize;
        while l < 64 {
            let s = const_sin_f64(PI_64 * (l as f64) / 128.0);
            sin2_y[l] = 4.0 * s * s;
            l += 1;
        }

        let sqrt_128 = const_sqrt_f64(128.0);
        let sqrt_64 = const_sqrt_f64(64.0);
        let scale_fwd_128 = const_sqrt_f64(2.0 / 128.0);
        let scale_fwd_64 = const_sqrt_f64(2.0 / 64.0);
        let inv_sqrt2 = const_sqrt_f64(0.5);

        Self {
            bitrev64,
            bitrev32,
            tw64_re,
            tw64_im,
            wn128_re,
            wn128_im,
            wd128_re,
            wd128_im,
            sin2_x,
            sin2_y,
            sqrt_128,
            sqrt_64,
            scale_fwd_128,
            scale_fwd_64,
            scale_dc_128: const_sqrt_f64(1.0 / 128.0),
            scale_dc_64: const_sqrt_f64(1.0 / 64.0),
            scale_nyq_128: inv_sqrt2 * scale_fwd_128,
            scale_nyq_64: inv_sqrt2 * scale_fwd_64,
            inv_32: 1.0 / 32.0,
            inv_64: 1.0 / 64.0,
        }
    }
}

static DCT_TABLES: DctTables = DctTables::new();

#[repr(C, align(16))]
pub struct BosCtrl {
    pub requested_mode: u32,
    pub overlay_kind: u32,
    pub last_continuity_epoch: u32,
    pub warmup_count: u32,
    pub clear_consec_frames: u32,
    pub anchor_base_mse: f32,
    pub gauge_x0: u16,
    pub gauge_y0: u16,
    pub gauge_x1: u16,
    pub gauge_y1: u16,
    pub last_frame_w: u16,
    pub last_frame_h: u16,
    pub anchor_count: u8,
    pub initialized: u8,
    pub _pad: [u8; 2],
    pub anchor_node_idx: [u16; BOS_MAX_ANCHORS],
    pub _reserved: [u32; 14],
}

const _: () = assert!(core::mem::size_of::<BosCtrl>() == 128);

/// Static zero-allocation Background-Oriented Schlieren (`BOS`) kernel state (`1,777,920 B`).
#[repr(C, align(16))]
pub struct BosState {
    // f64 solver planes (6 * 65,536 + 65,536 + 4,096 = 462,848 B)
    pub solve_x: [f64; BOS_NODES],
    pub solve_b: [f64; BOS_NODES],
    pub solve_r: [f64; BOS_NODES],
    pub solve_p: [f64; BOS_NODES],
    pub solve_z: [f64; BOS_NODES],
    pub solve_lp: [f64; BOS_NODES],
    pub comp_sum: [f64; BOS_NODES],
    pub fft_line: [f64; 512],

    // f32 / u32 buffers (921,600 + 32,768 + 196,608 + 65,536 + 128 = 1,216,640 B)
    pub ref_luma: [f32; BOS_MAX_PIXELS],
    pub ref_noise_var: [f32; BOS_NODES],
    pub planes: [f32; BOS_PLANES_LEN],
    pub w_x: [f32; BOS_NODES],
    pub w_y: [f32; BOS_NODES],
    pub telem: [u32; BOS_TELEM_LEN],

    // u16 / u8 buffers (49,152 + 32,768 + 16,384 = 98,304 B)
    pub comp_id: [u16; BOS_NODES],
    pub bfs_queue: [u16; BOS_NODES],
    pub comp_count: [u16; BOS_NODES],
    pub rgba: [u8; 4 * BOS_NODES],
    pub valid_mask: [u8; BOS_NODES],
    pub clear_streak: [u8; BOS_NODES],

    // 128 B private control state
    pub ctrl: BosCtrl,
}

const _: () = assert!(core::mem::size_of::<BosState>() <= 1_777_920);

#[inline(always)]
fn node_coord(i: usize, j: usize, frame_w: usize, frame_h: usize) -> (f32, f32) {
    let hx = (frame_w as f32) * (1.0 / (BOS_GRID_W as f32));
    let hy = (frame_h as f32) * (1.0 / (BOS_GRID_H as f32));
    (
        ((i as f32) + 0.5) * hx - 0.5,
        ((j as f32) + 0.5) * hy - 0.5,
    )
}

#[inline(always)]
fn sample_bilinear_checked(
    img: &[f32],
    w: usize,
    h: usize,
    x: f32,
    y: f32,
) -> Option<f32> {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
        return None;
    }
    let w_lim = (w - 1) as f32;
    let h_lim = (h - 1) as f32;
    if x >= w_lim || y >= h_lim {
        return None;
    }
    let ix = floor_f32(x) as usize;
    let iy = floor_f32(y) as usize;
    if ix + 1 >= w || iy + 1 >= h {
        return None;
    }
    let fx = x - (ix as f32);
    let fy = y - (iy as f32);
    let r0 = iy * w + ix;
    let r1 = r0 + w;
    let v00 = img[r0];
    let v10 = img[r0 + 1];
    let v01 = img[r1];
    let v11 = img[r1 + 1];
    let t0 = v00 + fx * (v10 - v00);
    let t1 = v01 + fx * (v11 - v01);
    Some(t0 + fy * (t1 - t0))
}

#[inline(always)]
fn sample_bilinear_unsat_checked(
    img: &[f32],
    w: usize,
    h: usize,
    x: f32,
    y: f32,
) -> Option<f32> {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
        return None;
    }
    let w_lim = (w - 1) as f32;
    let h_lim = (h - 1) as f32;
    if x >= w_lim || y >= h_lim {
        return None;
    }
    let ix = floor_f32(x) as usize;
    let iy = floor_f32(y) as usize;
    if ix + 1 >= w || iy + 1 >= h {
        return None;
    }
    let fx = x - (ix as f32);
    let fy = y - (iy as f32);
    let r0 = iy * w + ix;
    let r1 = r0 + w;
    let v00 = img[r0];
    let v10 = img[r0 + 1];
    let v01 = img[r1];
    let v11 = img[r1 + 1];
    if !(is_unsaturated(v00)
        && is_unsaturated(v10)
        && is_unsaturated(v01)
        && is_unsaturated(v11))
    {
        return None;
    }
    let t0 = v00 + fx * (v10 - v00);
    let t1 = v01 + fx * (v11 - v01);
    Some(t0 + fy * (t1 - t0))
}

#[inline(always)]
fn is_unsaturated(v: f32) -> bool {
    v.is_finite() && v > 0.008 && v < 0.992
}

/// `img[start..start + N]` as an array: one bounds check, after which constant indices need none.
#[inline(always)]
fn row_n<const N: usize>(img: &[f32], start: usize) -> &[f32; N] {
    img[start..start + N].try_into().unwrap()
}

/// `[a[i], a[i + 1], a[i + 2], a[i + 3]]` as one vector (a single unaligned vector load).
#[inline(always)]
fn ld4<const N: usize>(a: &[f32; N], i: usize) -> v128 {
    f32x4(a[i], a[i + 1], a[i + 2], a[i + 3])
}

/// Stores the four lanes of `v` to `a[i..i + 4]`.
#[inline(always)]
fn st4<const N: usize>(a: &mut [f32; N], i: usize, v: v128) {
    a[i] = f32x4_extract_lane::<0>(v);
    a[i + 1] = f32x4_extract_lane::<1>(v);
    a[i + 2] = f32x4_extract_lane::<2>(v);
    a[i + 3] = f32x4_extract_lane::<3>(v);
}

struct PatchRefData {
    t: [f32; 64],
    gx_raw: [f32; 64],
    gy_raw: [f32; 64],
    gx_perp: [f32; 64],
    gy_perp: [f32; 64],
    m_cnt: usize,
    mu_t: f64,
    var_sum_t: f64,
    ref_var: f64,
    h_xx: f64,
    h_xy: f64,
    h_yy: f64,
    det: f64,
    tr: f64,
    valid_flag: [u8; 64],
}

impl PatchRefData {
    #[inline(always)]
    fn empty() -> Self {
        Self {
            t: [0.0; 64],
            gx_raw: [0.0; 64],
            gy_raw: [0.0; 64],
            gx_perp: [0.0; 64],
            gy_perp: [0.0; 64],
            m_cnt: 0,
            mu_t: 0.0,
            var_sum_t: 0.0,
            ref_var: 0.0,
            h_xx: 0.0,
            h_xy: 0.0,
            h_yy: 0.0,
            det: 0.0,
            tr: 0.0,
            valid_flag: [0; 64],
        }
    }
}

struct CurPatchFit {
    gain: f64,
    offset: f64,
    bx: f64,
    by: f64,
    h_xx: f64,
    h_xy: f64,
    h_yy: f64,
    det: f64,
    cnt: usize,
}

/// Evaluates an `8x8` reference patch centered at `(cx, cy)`, builds the `[T, 1]`-orthogonal
/// spatial gradient basis, and checks contrast and `kappa(H) <= 30` conditioning.
#[inline(never)]
fn build_ref_patch_8x8(
    ref_luma: &[f32],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    out: &mut PatchRefData,
) -> bool {
    out.m_cnt = 0;
    let fast_in_bounds = cx >= 5.0
        && cx <= ((w - 6) as f32)
        && cy >= 5.0
        && cy <= ((h - 6) as f32);

    let mut sum_t = 0.0f64;

    if fast_in_bounds {
        let sx0 = cx - 4.5;
        let sy0 = cy - 4.5;
        let ix = floor_f32(sx0) as usize;
        let iy = floor_f32(sy0) as usize;
        let fx = sx0 - (ix as f32);
        let fy = sy0 - (iy as f32);
        let w00 = f32x4_splat((1.0 - fx) * (1.0 - fy));
        let w10 = f32x4_splat(fx * (1.0 - fy));
        let w01 = f32x4_splat((1.0 - fx) * fy);
        let w11 = f32x4_splat(fx * fy);

        // Bilinear 10x10 window. Each source row is read through an 11-element array (one bounds
        // check per row), so every `ld4` of four consecutive elements is a single vector load.
        let mut win10 = [0.0f32; 100];
        let mut ry = 0usize;
        while ry < 10 {
            let r0 = (iy + ry) * w + ix;
            let a = row_n::<11>(ref_luma, r0);
            let b = row_n::<11>(ref_luma, r0 + w);
            // Columns 0..4
            let v_lo = f32x4_add(
                f32x4_add(f32x4_mul(w00, ld4(a, 0)), f32x4_mul(w10, ld4(a, 1))),
                f32x4_add(f32x4_mul(w01, ld4(b, 0)), f32x4_mul(w11, ld4(b, 1))),
            );
            // Columns 4..8
            let v_hi = f32x4_add(
                f32x4_add(f32x4_mul(w00, ld4(a, 4)), f32x4_mul(w10, ld4(a, 5))),
                f32x4_add(f32x4_mul(w01, ld4(b, 4)), f32x4_mul(w11, ld4(b, 5))),
            );

            let dst = ry * 10;
            st4(&mut win10, dst, v_lo);
            st4(&mut win10, dst + 4, v_hi);

            let c8 = (1.0 - fx) * (1.0 - fy) * a[8]
                + fx * (1.0 - fy) * a[9]
                + (1.0 - fx) * fy * b[8]
                + fx * fy * b[9];
            let c9 = (1.0 - fx) * (1.0 - fy) * a[9]
                + fx * (1.0 - fy) * a[10]
                + (1.0 - fx) * fy * b[9]
                + fx * fy * b[10];
            win10[dst + 8] = c8;
            win10[dst + 9] = c9;
            ry += 1;
        }

        // Centre samples, central differences and the saturation mask, four pixels at a time.
        // `v > 0.008 && v < 0.992` is false for NaN and both infinities, exactly as
        // `is_unsaturated`. Invalid pixels get `gx_raw = gy_raw = +0.0` (the mask clears every
        // bit); their `t` slot is written too, but `t[k]` is only ever read where `valid_flag[k]`
        // is set (directly, or through a current-patch mask that is only set there).
        let half = f32x4_splat(0.5);
        let sat_lo = f32x4_splat(0.008);
        let sat_hi = f32x4_splat(0.992);
        let mut k = 0usize;
        let mut py = 0usize;
        while py < 8 {
            let r_prev = py * 10;
            let r_curr = (py + 1) * 10;
            let r_next = (py + 2) * 10;
            let mut px = 0usize;
            while px < 8 {
                let tv = ld4(&win10, r_curr + px + 1);
                let gx = f32x4_mul(
                    half,
                    f32x4_sub(ld4(&win10, r_curr + px + 2), ld4(&win10, r_curr + px)),
                );
                let gy = f32x4_mul(
                    half,
                    f32x4_sub(ld4(&win10, r_next + px + 1), ld4(&win10, r_prev + px + 1)),
                );
                let ok = v128_and(f32x4_gt(tv, sat_lo), f32x4_lt(tv, sat_hi));
                st4(&mut out.t, k, tv);
                st4(&mut out.gx_raw, k, v128_and(gx, ok));
                st4(&mut out.gy_raw, k, v128_and(gy, ok));
                let m = i32x4_bitmask(ok);
                out.valid_flag[k] = m & 1;
                out.valid_flag[k + 1] = (m >> 1) & 1;
                out.valid_flag[k + 2] = (m >> 2) & 1;
                out.valid_flag[k + 3] = (m >> 3) & 1;
                out.m_cnt += m.count_ones() as usize;
                {
                    // In pixel order, valid pixels only: the same additions as before.
                    if (m & 1) != 0 {
                        sum_t += f32x4_extract_lane::<0>(tv) as f64;
                    }
                    if (m & 2) != 0 {
                        sum_t += f32x4_extract_lane::<1>(tv) as f64;
                    }
                    if (m & 4) != 0 {
                        sum_t += f32x4_extract_lane::<2>(tv) as f64;
                    }
                    if (m & 8) != 0 {
                        sum_t += f32x4_extract_lane::<3>(tv) as f64;
                    }
                }
                k += 4;
                px += 4;
            }
            py += 1;
        }
    } else {
        let mut k = 0usize;
        let mut py = 0usize;
        while py < 8 {
            let sy = cy - 3.5 + (py as f32);
            let mut px = 0usize;
            while px < 8 {
                let sx = cx - 3.5 + (px as f32);
                let mut ok = false;
                if let (Some(tc), Some(txm), Some(txp), Some(tym), Some(typ)) = (
                    sample_bilinear_checked(ref_luma, w, h, sx, sy),
                    sample_bilinear_checked(ref_luma, w, h, sx - 1.0, sy),
                    sample_bilinear_checked(ref_luma, w, h, sx + 1.0, sy),
                    sample_bilinear_checked(ref_luma, w, h, sx, sy - 1.0),
                    sample_bilinear_checked(ref_luma, w, h, sx, sy + 1.0),
                ) {
                    if is_unsaturated(tc) {
                        out.valid_flag[k] = 1;
                        out.t[k] = tc;
                        out.gx_raw[k] = 0.5 * (txp - txm);
                        out.gy_raw[k] = 0.5 * (typ - tym);
                        sum_t += tc as f64;
                        out.m_cnt += 1;
                        ok = true;
                    }
                }
                if !ok {
                    out.valid_flag[k] = 0;
                    out.gx_raw[k] = 0.0;
                    out.gy_raw[k] = 0.0;
                }
                k += 1;
                px += 1;
            }
            py += 1;
        }
    }

    if out.m_cnt < 40 {
        return false;
    }

    let inv_m = 1.0 / (out.m_cnt as f64);
    let mu_t = sum_t * inv_m;
    out.mu_t = mu_t;

    let mut var_sum_t = 0.0f64;
    let mut mean_gx = 0.0f64;
    let mut mean_gy = 0.0f64;
    let mut dot_tgx = 0.0f64;
    let mut dot_tgy = 0.0f64;
    let mut k = 0usize;
    while k < 64 {
        if out.valid_flag[k] != 0 {
            let dt = (out.t[k] as f64) - mu_t;
            let gx = out.gx_raw[k] as f64;
            let gy = out.gy_raw[k] as f64;
            var_sum_t += dt * dt;
            mean_gx += gx;
            mean_gy += gy;
            dot_tgx += dt * gx;
            dot_tgy += dt * gy;
        }
        k += 1;
    }

    let ref_var = var_sum_t * inv_m;
    if !ref_var.is_finite() || ref_var < 6.4e-5 || var_sum_t < 1e-6 {
        return false;
    }
    out.var_sum_t = var_sum_t;
    out.ref_var = ref_var;

    mean_gx *= inv_m;
    mean_gy *= inv_m;
    let a_gx = dot_tgx / var_sum_t;
    let a_gy = dot_tgy / var_sum_t;

    let mut h_xx = 0.0f64;
    let mut h_xy = 0.0f64;
    let mut h_yy = 0.0f64;
    let mut k = 0usize;
    while k < 64 {
        if out.valid_flag[k] != 0 {
            let dt = (out.t[k] as f64) - mu_t;
            let gxp = (out.gx_raw[k] as f64) - mean_gx - a_gx * dt;
            let gyp = (out.gy_raw[k] as f64) - mean_gy - a_gy * dt;
            out.gx_perp[k] = gxp as f32;
            out.gy_perp[k] = gyp as f32;
            h_xx += gxp * gxp;
            h_xy += gxp * gyp;
            h_yy += gyp * gyp;
        } else {
            out.gx_perp[k] = 0.0;
            out.gy_perp[k] = 0.0;
        }
        k += 1;
    }

    let tr = h_xx + h_yy;
    let det = h_xx * h_yy - h_xy * h_xy;
    if !tr.is_finite() || !det.is_finite() || tr <= 0.0 || det <= 0.0 {
        return false;
    }
    // Exact kappa(H) <= 30 check: det / tr^2 >= 30 / 31^2 = 30 / 961
    if det * 961.0 < 30.0 * tr * tr {
        return false;
    }
    let disc = sqrt_f64(((h_xx - h_yy) * (h_xx - h_yy) + 4.0 * h_xy * h_xy).max(0.0));
    let lambda_min = 0.5 * (tr - disc);
    if lambda_min < 5e-5 * (out.m_cnt as f64) {
        return false;
    }

    out.h_xx = h_xx;
    out.h_xy = h_xy;
    out.h_yy = h_yy;
    out.det = det;
    out.tr = tr;
    true
}

/// Samples current `8x8` patch at `(cx + dx, cy + dy)` and solves one affine-invariant IC-LK
/// gradient + `(gain, offset)` fit against precomputed `PatchRefData`.
/// When current saturation/border masking reduces active samples (`cnt < ref_p.m_cnt`),
/// re-projects raw reference gradients against `[ref, 1]` and checks/returns `H_sub` on surviving pixels.
#[inline(never)]
fn eval_cur_patch_8x8(
    cur_luma: &[f32],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    dx: f32,
    dy: f32,
    ref_p: &PatchRefData,
    out_cur: &mut [f32; 64],
    out_cur_mask: &mut [u8; 64],
) -> Option<CurPatchFit> {
    let sx0 = cx - 3.5 + dx;
    let sy0 = cy - 3.5 + dy;
    let fast_ok = ref_p.m_cnt == 64
        && sx0 >= 0.0
        && sy0 >= 0.0
        && sx0 + 7.0 < ((w - 1) as f32)
        && sy0 + 7.0 < ((h - 1) as f32);

    let mut cnt = 0usize;
    let mut sum_i = 0.0f64;
    let mut sum_t_sub = 0.0f64;

    if fast_ok {
        let ix = floor_f32(sx0) as usize;
        let iy = floor_f32(sy0) as usize;
        let fx = sx0 - (ix as f32);
        let fy = sy0 - (iy as f32);
        let w00 = f32x4_splat((1.0 - fx) * (1.0 - fy));
        let w10 = f32x4_splat(fx * (1.0 - fy));
        let w01 = f32x4_splat((1.0 - fx) * fy);
        let w11 = f32x4_splat(fx * fy);

        let mut tex_min = f32x4_splat(0.5);
        let mut tex_max = f32x4_splat(0.5);
        let mut acc_i = f32x4_splat(0.0);
        let mut py = 0usize;
        while py < 8 {
            // Rows read through 9-element arrays: one bounds check per row, vector loads.
            let r0 = (iy + py) * w + ix;
            let ra = row_n::<9>(cur_luma, r0);
            let rb = row_n::<9>(cur_luma, r0 + w);
            let a0 = ld4(ra, 0);
            let a1 = ld4(ra, 1);
            let a2 = ld4(rb, 0);
            let a3 = ld4(rb, 1);
            let v_lo = f32x4_add(
                f32x4_add(f32x4_mul(w00, a0), f32x4_mul(w10, a1)),
                f32x4_add(f32x4_mul(w01, a2), f32x4_mul(w11, a3)),
            );

            let b0 = ld4(ra, 4);
            let b1 = ld4(ra, 5);
            let b2 = ld4(rb, 4);
            let b3 = ld4(rb, 5);
            let v_hi = f32x4_add(
                f32x4_add(f32x4_mul(w00, b0), f32x4_mul(w10, b1)),
                f32x4_add(f32x4_mul(w01, b2), f32x4_mul(w11, b3)),
            );

            let base = py * 8;
            st4(out_cur, base, v_lo);
            st4(out_cur, base + 4, v_hi);

            let row_lo = f32x4_min(
                f32x4_min(a0, b0),
                f32x4_min(b1, f32x4_min(a2, f32x4_min(b2, b3))),
            );
            let row_hi = f32x4_max(
                f32x4_max(a0, b0),
                f32x4_max(b1, f32x4_max(a2, f32x4_max(b2, b3))),
            );
            tex_min = f32x4_min(tex_min, row_lo);
            tex_max = f32x4_max(tex_max, row_hi);
            acc_i = f32x4_add(acc_i, f32x4_add(v_lo, v_hi));
            py += 1;
        }

        let all_ok = is_unsaturated(f32x4_extract_lane::<0>(tex_min))
            && is_unsaturated(f32x4_extract_lane::<1>(tex_min))
            && is_unsaturated(f32x4_extract_lane::<2>(tex_min))
            && is_unsaturated(f32x4_extract_lane::<3>(tex_min))
            && is_unsaturated(f32x4_extract_lane::<0>(tex_max))
            && is_unsaturated(f32x4_extract_lane::<1>(tex_max))
            && is_unsaturated(f32x4_extract_lane::<2>(tex_max))
            && is_unsaturated(f32x4_extract_lane::<3>(tex_max));

        if all_ok {
            sum_i = (f32x4_extract_lane::<0>(acc_i)
                + f32x4_extract_lane::<1>(acc_i)
                + f32x4_extract_lane::<2>(acc_i)
                + f32x4_extract_lane::<3>(acc_i)) as f64;
            let mut k = 0usize;
            while k < 64 {
                out_cur_mask[k] = 1;
                k += 1;
            }
            let mu_t = ref_p.mu_t;
            let mut sum_ti = 0.0f64;
            let mut bx = 0.0f64;
            let mut by = 0.0f64;
            let mut k = 0usize;
            while k < 64 {
                let iv = out_cur[k] as f64;
                sum_ti += ((ref_p.t[k] as f64) - mu_t) * iv;
                bx += (ref_p.gx_perp[k] as f64) * iv;
                by += (ref_p.gy_perp[k] as f64) * iv;
                k += 1;
            }
            let gain = sum_ti / ref_p.var_sum_t;
            let offset = (sum_i * (1.0 / 64.0)) - gain * mu_t;
            return Some(CurPatchFit {
                gain,
                offset,
                bx,
                by,
                h_xx: ref_p.h_xx,
                h_xy: ref_p.h_xy,
                h_yy: ref_p.h_yy,
                det: ref_p.det,
                cnt: 64,
            });
        }
    }

    // Masked / partially saturated fallback
    let mut k = 0usize;
    let mut py = 0usize;
    while py < 8 {
        let wy = cy - 3.5 + dy + (py as f32);
        let mut px = 0usize;
        while px < 8 {
            let wx = cx - 3.5 + dx + (px as f32);
            let mut ok = false;
            if ref_p.valid_flag[k] != 0 {
                if let Some(cv) = sample_bilinear_unsat_checked(cur_luma, w, h, wx, wy) {
                    out_cur[k] = cv;
                    out_cur_mask[k] = 1;
                    sum_i += cv as f64;
                    sum_t_sub += ref_p.t[k] as f64;
                    cnt += 1;
                    ok = true;
                }
            }
            if !ok {
                out_cur_mask[k] = 0;
                out_cur[k] = 0.0;
            }
            k += 1;
            px += 1;
        }
        py += 1;
    }

    if cnt < 40 {
        return None;
    }

    let inv_c = 1.0 / (cnt as f64);
    let mu_t_sub = sum_t_sub * inv_c;
    let mu_i_sub = sum_i * inv_c;
    let mut var_t_sub = 0.0f64;
    let mut sum_ti = 0.0f64;
    let mut k = 0usize;
    while k < 64 {
        if out_cur_mask[k] != 0 {
            let dt = (ref_p.t[k] as f64) - mu_t_sub;
            let di = (out_cur[k] as f64) - mu_i_sub;
            var_t_sub += dt * dt;
            sum_ti += dt * di;
        }
        k += 1;
    }
    if !var_t_sub.is_finite() || var_t_sub < 1e-6 || (var_t_sub * inv_c) < 6.4e-5 {
        return None;
    }
    let gain = sum_ti / var_t_sub;
    let offset = mu_i_sub - gain * mu_t_sub;
    if !gain.is_finite() || !offset.is_finite() {
        return None;
    }

    if cnt == ref_p.m_cnt {
        let mut bx = 0.0f64;
        let mut by = 0.0f64;
        let mut k = 0usize;
        while k < 64 {
            if out_cur_mask[k] != 0 {
                let res = (out_cur[k] as f64) - (gain * (ref_p.t[k] as f64) + offset);
                bx += (ref_p.gx_perp[k] as f64) * res;
                by += (ref_p.gy_perp[k] as f64) * res;
            }
            k += 1;
        }
        return Some(CurPatchFit {
            gain,
            offset,
            bx,
            by,
            h_xx: ref_p.h_xx,
            h_xy: ref_p.h_xy,
            h_yy: ref_p.h_yy,
            det: ref_p.det,
            cnt,
        });
    }

    // Re-project raw gradients against [ref, 1] and rebuild/condition H_sub on surviving pixels
    let mut mean_gx = 0.0f64;
    let mut mean_gy = 0.0f64;
    let mut dot_tgx = 0.0f64;
    let mut dot_tgy = 0.0f64;
    let mut k = 0usize;
    while k < 64 {
        if out_cur_mask[k] != 0 {
            let dt = (ref_p.t[k] as f64) - mu_t_sub;
            let gx = ref_p.gx_raw[k] as f64;
            let gy = ref_p.gy_raw[k] as f64;
            mean_gx += gx;
            mean_gy += gy;
            dot_tgx += dt * gx;
            dot_tgy += dt * gy;
        }
        k += 1;
    }
    mean_gx *= inv_c;
    mean_gy *= inv_c;
    let a_gx = dot_tgx / var_t_sub;
    let a_gy = dot_tgy / var_t_sub;

    let mut hs_xx = 0.0f64;
    let mut hs_xy = 0.0f64;
    let mut hs_yy = 0.0f64;
    let mut bx = 0.0f64;
    let mut by = 0.0f64;
    let mut k = 0usize;
    while k < 64 {
        if out_cur_mask[k] != 0 {
            let dt = (ref_p.t[k] as f64) - mu_t_sub;
            let gxp = (ref_p.gx_raw[k] as f64) - mean_gx - a_gx * dt;
            let gyp = (ref_p.gy_raw[k] as f64) - mean_gy - a_gy * dt;
            hs_xx += gxp * gxp;
            hs_xy += gxp * gyp;
            hs_yy += gyp * gyp;
            let res = (out_cur[k] as f64) - (gain * (ref_p.t[k] as f64) + offset);
            bx += gxp * res;
            by += gyp * res;
        }
        k += 1;
    }

    let tr_s = hs_xx + hs_yy;
    let det_s = hs_xx * hs_yy - hs_xy * hs_xy;
    if !tr_s.is_finite() || !det_s.is_finite() || tr_s <= 0.0 || det_s <= 0.0 {
        return None;
    }
    if det_s * 961.0 < 30.0 * tr_s * tr_s {
        return None;
    }
    let disc_s = sqrt_f64(((hs_xx - hs_yy) * (hs_xx - hs_yy) + 4.0 * hs_xy * hs_xy).max(0.0));
    let lambda_min_s = 0.5 * (tr_s - disc_s);
    if lambda_min_s < 5e-5 * (cnt as f64) {
        return None;
    }

    Some(CurPatchFit {
        gain,
        offset,
        bx,
        by,
        h_xx: hs_xx,
        h_xy: hs_xy,
        h_yy: hs_yy,
        det: det_s,
        cnt,
    })
}

impl BosState {
    pub fn init_in_place(&mut self) {
        self.solve_x.fill(0.0);
        self.solve_b.fill(0.0);
        self.solve_r.fill(0.0);
        self.solve_p.fill(0.0);
        self.solve_z.fill(0.0);
        self.solve_lp.fill(0.0);
        self.comp_sum.fill(0.0);
        self.fft_line.fill(0.0);
        self.ref_luma.fill(0.0);
        self.ref_noise_var.fill(0.0);
        self.planes.fill(0.0);
        self.w_x.fill(0.0);
        self.w_y.fill(0.0);
        self.telem.fill(0);
        self.comp_id.fill(0);
        self.bfs_queue.fill(0);
        self.comp_count.fill(0);
        self.rgba.fill(0);
        self.valid_mask.fill(0);
        self.clear_streak.fill(0);
        self.ctrl = BosCtrl {
            requested_mode: BOS_MODE_OFF,
            overlay_kind: BOS_OVERLAY_MAG,
            last_continuity_epoch: 0,
            warmup_count: 0,
            clear_consec_frames: 0,
            anchor_base_mse: 0.0,
            gauge_x0: 0,
            gauge_y0: 0,
            gauge_x1: 0,
            gauge_y1: 0,
            last_frame_w: 0,
            last_frame_h: 0,
            anchor_count: 0,
            initialized: 1,
            _pad: [0; 2],
            anchor_node_idx: [0; BOS_MAX_ANCHORS],
            _reserved: [0; 14],
        };
        self.telem[0] = 1;
        self.telem[1] = BOS_MODE_OFF;
        self.telem[2] = BOS_STATE_OFF;
        self.telem[3] = BOS_REASON_NO_GAUGE;
    }

    pub fn reset_keep_config(&mut self) {
        let req_mode = self.ctrl.requested_mode;
        let ov_kind = self.ctrl.overlay_kind;
        let gx0 = self.ctrl.gauge_x0;
        let gy0 = self.ctrl.gauge_y0;
        let gx1 = self.ctrl.gauge_x1;
        let gy1 = self.ctrl.gauge_y1;
        let ref_epoch = self.telem[11].wrapping_add(1);
        self.init_in_place();
        self.ctrl.requested_mode = req_mode;
        self.ctrl.overlay_kind = ov_kind;
        self.ctrl.gauge_x0 = gx0;
        self.ctrl.gauge_y0 = gy0;
        self.ctrl.gauge_x1 = gx1;
        self.ctrl.gauge_y1 = gy1;
        self.telem[1] = req_mode;
        self.telem[11] = ref_epoch;
        let has_gauge = self.has_valid_gauge();
        if req_mode == BOS_MODE_OFF {
            self.telem[2] = BOS_STATE_OFF;
            self.telem[3] = if has_gauge { 0 } else { BOS_REASON_NO_GAUGE };
        } else if !has_gauge {
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = BOS_REASON_NO_GAUGE;
        } else {
            self.telem[2] = BOS_STATE_WARMING;
            self.telem[3] = 0;
        }
    }

    #[inline(always)]
    pub fn has_valid_gauge(&self) -> bool {
        self.ctrl.gauge_x0 < self.ctrl.gauge_x1
            && self.ctrl.gauge_y0 < self.ctrl.gauge_y1
            && (self.ctrl.gauge_x1 as usize) <= BOS_GRID_W
            && (self.ctrl.gauge_y1 as usize) <= BOS_GRID_H
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn get_mode(&self) -> u32 {
        self.ctrl.requested_mode
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn get_state(&self) -> u32 {
        self.telem[2]
    }

    #[inline(always)]
    pub fn get_telemetry_ptr(&self) -> *const u32 {
        self.telem.as_ptr()
    }

    #[inline(always)]
    pub fn get_planes_ptr(&self) -> *const f32 {
        self.planes.as_ptr()
    }

    #[inline(always)]
    pub fn get_valid_ptr(&self) -> *const u8 {
        self.valid_mask.as_ptr()
    }

    #[inline(always)]
    pub fn get_rgba_ptr(&self) -> *const u8 {
        self.rgba.as_ptr()
    }

    #[inline(never)]
    fn invalidate_outputs(&mut self) {
        self.planes.fill(0.0);
        self.valid_mask.fill(0);
        self.rgba.fill(0);
        self.w_x.fill(0.0);
        self.w_y.fill(0.0);
        self.telem[4] = 0;
        self.telem[5] = 0;
        self.telem[6] = 0;
        self.telem[7] = 0;
        self.telem[8] = 0;
        self.telem[9] = BOS_SOLVER_NONE;
        for idx in 16..24 {
            self.telem[idx] = 0;
        }
    }

    pub fn set_mode(&mut self, mode: u32) -> u32 {
        if mode > BOS_MODE_MAINT {
            return 0;
        }
        let prev = self.ctrl.requested_mode;
        self.ctrl.requested_mode = mode;
        self.telem[0] = 1;
        self.telem[1] = mode;

        if mode == BOS_MODE_OFF {
            self.ctrl.warmup_count = 0;
            self.ctrl.clear_consec_frames = 0;
            self.ctrl.anchor_count = 0;
            self.clear_streak.fill(0);
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_OFF;
            self.telem[3] = if self.has_valid_gauge() { 0 } else { BOS_REASON_NO_GAUGE };
            self.telem[10] = 0;
            self.telem[12] = 0;
            self.telem[13] = 0;
            self.telem[14] = 0;
            self.telem[15] = 0;
            return 1;
        }

        if prev == BOS_MODE_OFF {
            self.ctrl.warmup_count = 0;
            self.ctrl.clear_consec_frames = 0;
            self.ctrl.anchor_count = 0;
            self.clear_streak.fill(0);
            self.telem[10] = 0;
            self.telem[11] = self.telem[11].wrapping_add(1);
            self.invalidate_outputs();
            if self.has_valid_gauge() {
                self.telem[2] = BOS_STATE_WARMING;
                self.telem[3] = 0;
            } else {
                self.telem[2] = BOS_STATE_DISTURBED;
                self.telem[3] = BOS_REASON_NO_GAUGE;
            }
        } else if mode == BOS_MODE_FROZEN {
            self.ctrl.clear_consec_frames = 0;
        }
        1
    }

    pub fn reset_ref(&mut self) {
        self.ctrl.warmup_count = 0;
        self.ctrl.clear_consec_frames = 0;
        self.ctrl.anchor_count = 0;
        self.ctrl.anchor_base_mse = 0.0;
        self.ref_noise_var.fill(0.0);
        self.clear_streak.fill(0);
        self.invalidate_outputs();
        self.telem[10] = 0;
        self.telem[11] = self.telem[11].wrapping_add(1);
        if self.ctrl.requested_mode == BOS_MODE_OFF {
            self.telem[2] = BOS_STATE_OFF;
            self.telem[3] = if self.has_valid_gauge() { 0 } else { BOS_REASON_NO_GAUGE };
        } else if !self.has_valid_gauge() {
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = BOS_REASON_NO_GAUGE;
        } else {
            self.telem[2] = BOS_STATE_WARMING;
            self.telem[3] = 0;
        }
    }

    pub fn set_gauge_roi(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) -> u32 {
        if x0 >= x1 || y0 >= y1 || x1 > (BOS_GRID_W as u32) || y1 > (BOS_GRID_H as u32) {
            return 0;
        }
        let changed = self.ctrl.gauge_x0 as u32 != x0
            || self.ctrl.gauge_y0 as u32 != y0
            || self.ctrl.gauge_x1 as u32 != x1
            || self.ctrl.gauge_y1 as u32 != y1;
        self.ctrl.gauge_x0 = x0 as u16;
        self.ctrl.gauge_y0 = y0 as u16;
        self.ctrl.gauge_x1 = x1 as u16;
        self.ctrl.gauge_y1 = y1 as u16;
        self.telem[3] &= !BOS_REASON_NO_GAUGE;
        if changed {
            self.reset_ref();
        }
        1
    }

    pub fn set_overlay_kind(&mut self, kind: u32) -> u32 {
        if kind > BOS_OVERLAY_CURL {
            return 0;
        }
        self.ctrl.overlay_kind = kind;
        self.render_bos_rgba();
        1
    }

    /// Deterministically selects `8..=16` spatially distributed high-contrast anchor patches
    /// inside the declared `gauge_roi` on `ref_luma`.
    #[inline(never)]
    fn select_anchors_in_gauge(&mut self, frame_w: usize, frame_h: usize) -> usize {
        let gx0 = self.ctrl.gauge_x0 as usize;
        let gy0 = self.ctrl.gauge_y0 as usize;
        let gx1 = self.ctrl.gauge_x1 as usize;
        let gy1 = self.ctrl.gauge_y1 as usize;
        self.ctrl.anchor_count = 0;
        self.ctrl.anchor_node_idx = [0; BOS_MAX_ANCHORS];
        if gx0 >= gx1 || gy0 >= gy1 || gx1 > BOS_GRID_W || gy1 > BOS_GRID_H {
            return 0;
        }

        let gw = gx1 - gx0;
        let gh = gy1 - gy0;
        let (nx_bins, ny_bins) = if gw >= 4 && gh >= 4 {
            (4usize, 4usize)
        } else if gw >= 8 && gh >= 2 {
            (8usize, 2usize)
        } else if gh >= 8 && gw >= 2 {
            (2usize, 8usize)
        } else {
            (gw.min(4), gh.min(4))
        };

        let mut p_ref = PatchRefData::empty();
        let mut selected = 0usize;

        let mut by = 0usize;
        while by < ny_bins {
            let y_start = gy0 + (by * gh) / ny_bins;
            let y_end = (gy0 + ((by + 1) * gh) / ny_bins).max(y_start + 1).min(gy1);
            let mut bx = 0usize;
            while bx < nx_bins {
                let x_start = gx0 + (bx * gw) / nx_bins;
                let x_end = (gx0 + ((bx + 1) * gw) / nx_bins).max(x_start + 1).min(gx1);
                let mut best_score = 0.0f64;
                let mut best_node = u16::MAX;
                let mut j = y_start;
                while j < y_end {
                    let mut i = x_start;
                    while i < x_end {
                        let (cx, cy) = node_coord(i, j, frame_w, frame_h);
                        if cx >= 5.0
                            && cx <= ((frame_w - 6) as f32)
                            && cy >= 5.0
                            && cy <= ((frame_h - 6) as f32)
                        {
                            if build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, &mut p_ref)
                                && p_ref.m_cnt == 64
                            {
                                let disc = sqrt_f64(
                                    ((p_ref.h_xx - p_ref.h_yy) * (p_ref.h_xx - p_ref.h_yy)
                                        + 4.0 * p_ref.h_xy * p_ref.h_xy)
                                        .max(0.0),
                                );
                                let lmin = 0.5 * (p_ref.tr - disc);
                                if lmin > best_score {
                                    best_score = lmin;
                                    best_node = (j * BOS_GRID_W + i) as u16;
                                }
                            }
                        }
                        i += 1;
                    }
                    j += 1;
                }
                if best_node != u16::MAX && selected < BOS_MAX_ANCHORS {
                    self.ctrl.anchor_node_idx[selected] = best_node;
                    selected += 1;
                }
                bx += 1;
            }
            by += 1;
        }

        // Second pass if some spatial bins were textureless so we still gather >= 8 anchors if available.
        if selected < BOS_MAX_ANCHORS {
            let mut best_scores = [0.0f64; BOS_MAX_ANCHORS];
            let mut best_nodes = [u16::MAX; BOS_MAX_ANCHORS];
            let mut cand_cnt = 0usize;
            let mut j = gy0;
            while j < gy1 {
                let mut i = gx0;
                while i < gx1 {
                    let node = (j * BOS_GRID_W + i) as u16;
                    let mut dup = false;
                    let mut k = 0usize;
                    while k < selected {
                        if self.ctrl.anchor_node_idx[k] == node {
                            dup = true;
                            break;
                        }
                        k += 1;
                    }
                    if !dup {
                        let (cx, cy) = node_coord(i, j, frame_w, frame_h);
                        if cx >= 5.0
                            && cx <= ((frame_w - 6) as f32)
                            && cy >= 5.0
                            && cy <= ((frame_h - 6) as f32)
                            && build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, &mut p_ref)
                            && p_ref.m_cnt == 64
                        {
                            let disc = sqrt_f64(
                                ((p_ref.h_xx - p_ref.h_yy) * (p_ref.h_xx - p_ref.h_yy)
                                    + 4.0 * p_ref.h_xy * p_ref.h_xy)
                                    .max(0.0),
                            );
                            let lmin = 0.5 * (p_ref.tr - disc);
                            if cand_cnt < BOS_MAX_ANCHORS {
                                best_scores[cand_cnt] = lmin;
                                best_nodes[cand_cnt] = node;
                                cand_cnt += 1;
                            } else {
                                let mut min_pos = 0usize;
                                let mut c = 1usize;
                                while c < BOS_MAX_ANCHORS {
                                    if best_scores[c] < best_scores[min_pos] {
                                        min_pos = c;
                                    }
                                    c += 1;
                                }
                                if lmin > best_scores[min_pos] {
                                    best_scores[min_pos] = lmin;
                                    best_nodes[min_pos] = node;
                                }
                            }
                        }
                    }
                    i += 1;
                }
                j += 1;
            }
            while selected < BOS_MAX_ANCHORS && cand_cnt > 0 {
                let mut max_pos = 0usize;
                let mut c = 1usize;
                while c < cand_cnt {
                    if best_scores[c] > best_scores[max_pos] {
                        max_pos = c;
                    }
                    c += 1;
                }
                if best_scores[max_pos] <= 0.0 || best_nodes[max_pos] == u16::MAX {
                    break;
                }
                let node = best_nodes[max_pos];
                best_scores[max_pos] = -1.0;
                // Prefer spaced anchors first when gauge has plenty of nodes
                let ni = (node as usize) & 127;
                let nj = (node as usize) >> 7;
                let (ncx, ncy) = node_coord(ni, nj, frame_w, frame_h);
                let mut too_close = false;
                if gw * gh >= 32 {
                    let min_sep = if frame_w >= 640 { 4.5f32 } else { 2.2f32 };
                    let mut k = 0usize;
                    while k < selected {
                        let ak = self.ctrl.anchor_node_idx[k] as usize;
                        let (acx, acy) = node_coord(ak & 127, ak >> 7, frame_w, frame_h);
                        if (ncx - acx).abs() < min_sep && (ncy - acy).abs() < min_sep {
                            too_close = true;
                            break;
                        }
                        k += 1;
                    }
                }
                if !too_close {
                    self.ctrl.anchor_node_idx[selected] = node;
                    selected += 1;
                }
                cand_cnt -= 1;
            }
        }

        self.ctrl.anchor_count = selected as u8;
        selected
    }

    /// Helper evaluating joint gain/offset and photometric MSE over frozen anchors at `(tx, ty)`.
    #[inline(never)]
    fn eval_anchor_mse_at(
        &self,
        cur_luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        tx: f32,
        ty: f32,
        p_r: &mut PatchRefData,
        cb: &mut [f32; 64],
        cm: &mut [u8; 64],
    ) -> (f64, f64, usize) {
        let k_cnt = self.ctrl.anchor_count as usize;
        let mut sum_t = 0.0f64;
        let mut sum_i = 0.0f64;
        let mut n_tot = 0usize;

        let mut a = 0usize;
        while a < k_cnt {
            let node = self.ctrl.anchor_node_idx[a] as usize;
            let (cx, cy) = node_coord(node & 127, node >> 7, frame_w, frame_h);
            if build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, p_r) {
                if eval_cur_patch_8x8(cur_luma, frame_w, frame_h, cx, cy, tx, ty, p_r, cb, cm)
                    .is_some()
                {
                    let mut k = 0usize;
                    while k < 64 {
                        if cm[k] != 0 {
                            sum_t += p_r.t[k] as f64;
                            sum_i += cb[k] as f64;
                            n_tot += 1;
                        }
                        k += 1;
                    }
                }
            }
            a += 1;
        }
        if n_tot < BOS_MIN_ANCHORS * 40 {
            return (1.0, 0.0, 0);
        }
        let mu_t = sum_t / (n_tot as f64);
        let mu_i = sum_i / (n_tot as f64);
        let mut var_t = 0.0f64;
        let mut cov_ti = 0.0f64;
        let mut a = 0usize;
        while a < k_cnt {
            let node = self.ctrl.anchor_node_idx[a] as usize;
            let (cx, cy) = node_coord(node & 127, node >> 7, frame_w, frame_h);
            if build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, p_r) {
                if eval_cur_patch_8x8(cur_luma, frame_w, frame_h, cx, cy, tx, ty, p_r, cb, cm)
                    .is_some()
                {
                    let mut k = 0usize;
                    while k < 64 {
                        if cm[k] != 0 {
                            let dt = (p_r.t[k] as f64) - mu_t;
                            let di = (cb[k] as f64) - mu_i;
                            var_t += dt * dt;
                            cov_ti += dt * di;
                        }
                        k += 1;
                    }
                }
            }
            a += 1;
        }
        if var_t < 1e-6 {
            return (1.0, 0.0, 0);
        }
        let g = cov_ti / var_t;
        let off = mu_i - g * mu_t;
        let mut mse_sum = 0.0f64;
        let mut a = 0usize;
        while a < k_cnt {
            let node = self.ctrl.anchor_node_idx[a] as usize;
            let (cx, cy) = node_coord(node & 127, node >> 7, frame_w, frame_h);
            if build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, p_r) {
                if eval_cur_patch_8x8(cur_luma, frame_w, frame_h, cx, cy, tx, ty, p_r, cb, cm)
                    .is_some()
                {
                    let mut k = 0usize;
                    while k < 64 {
                        if cm[k] != 0 {
                            let r = (cb[k] as f64) - (g * (p_r.t[k] as f64) + off);
                            mse_sum += r * r;
                        }
                        k += 1;
                    }
                }
            }
            a += 1;
        }
        (mse_sum / (n_tot as f64), g, n_tot)
    }

    /// Estimates global subpixel translation `(shift_x, shift_y)` and global gain/offset-corrected
    /// photometric MSE jointly across the frozen anchor patches against `ref_luma`.
    #[inline(never)]
    fn evaluate_frozen_anchors(
        &self,
        cur_luma: &[f32],
        frame_w: usize,
        frame_h: usize,
    ) -> (f32, f32, f32, f32, bool) {
        let k_cnt = self.ctrl.anchor_count as usize;
        if k_cnt < BOS_MIN_ANCHORS {
            return (0.0, 0.0, 1.0, 1.0, false);
        }

        let mut p_ref = PatchRefData::empty();
        let mut cur_buf = [0.0f32; 64];
        let mut cur_mask = [0u8; 64];

        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        let mut gain_est = 1.0f64;

        let mut iter = 0usize;
        while iter < 3 {
            let mut h_xx = 0.0f64;
            let mut h_xy = 0.0f64;
            let mut h_yy = 0.0f64;
            let mut bx = 0.0f64;
            let mut by = 0.0f64;
            let mut sum_g = 0.0f64;
            let mut g_cnt = 0usize;

            let mut a = 0usize;
            while a < k_cnt {
                let node = self.ctrl.anchor_node_idx[a] as usize;
                let (cx, cy) = node_coord(node & 127, node >> 7, frame_w, frame_h);
                if build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, &mut p_ref) {
                    if let Some(fit) = eval_cur_patch_8x8(
                        cur_luma,
                        frame_w,
                        frame_h,
                        cx,
                        cy,
                        dx,
                        dy,
                        &p_ref,
                        &mut cur_buf,
                        &mut cur_mask,
                    ) {
                        if fit.gain.is_finite() && fit.gain >= 0.35 && fit.gain <= 2.8 {
                            h_xx += fit.h_xx;
                            h_xy += fit.h_xy;
                            h_yy += fit.h_yy;
                            bx += fit.bx / fit.gain;
                            by += fit.by / fit.gain;
                            sum_g += fit.gain;
                            g_cnt += 1;
                        }
                    }
                }
                a += 1;
            }

            if g_cnt < BOS_MIN_ANCHORS {
                break;
            }
            gain_est = sum_g / (g_cnt as f64);
            let det = h_xx * h_yy - h_xy * h_xy;
            if !det.is_finite() || det <= 1e-10 {
                return (dx, dy, 1.0, gain_est as f32, false);
            }
            let inv_det = 1.0 / det;
            let step_x = (-(h_yy * bx - h_xy * by) * inv_det) as f32;
            let step_y = (-(-h_xy * bx + h_xx * by) * inv_det) as f32;
            if !step_x.is_finite() || !step_y.is_finite() {
                return (dx, dy, 1.0, gain_est as f32, false);
            }
            dx = (dx + step_x).clamp(-3.5, 3.5);
            dy = (dy + step_y).clamp(-3.5, 3.5);
            if step_x * step_x + step_y * step_y < 1e-4 {
                break;
            }
            iter += 1;
        }

        // Final joint gain/offset and MSE evaluation at converged (dx, dy)
        let (mut mse, mut gain_final, n_tot) = self.eval_anchor_mse_at(
            cur_luma,
            frame_w,
            frame_h,
            dx,
            dy,
            &mut p_ref,
            &mut cur_buf,
            &mut cur_mask,
        );
        if n_tot == 0 {
            return (dx, dy, 1.0, gain_est as f32, false);
        }

        // Only if MSE exceeds threshold, probe coarse integer offsets +/-4 px to catch large shocks.
        let mse_gate = (4.0 * (self.ctrl.anchor_base_mse as f64)).max(1e-4);
        if mse > mse_gate {
            let mut best_c_mse = mse;
            let mut best_cx = dx;
            let mut best_cy = dy;
            let mut oy = -4i32;
            while oy <= 4 {
                let mut ox = -4i32;
                while ox <= 4 {
                    if ox != 0 || oy != 0 {
                        let (cmse, cg, cn) = self.eval_anchor_mse_at(
                            cur_luma,
                            frame_w,
                            frame_h,
                            ox as f32,
                            oy as f32,
                            &mut p_ref,
                            &mut cur_buf,
                            &mut cur_mask,
                        );
                        if cn > 0 && cg >= 0.5 && cg <= 2.0 && cmse < best_c_mse {
                            best_c_mse = cmse;
                            best_cx = ox as f32;
                            best_cy = oy as f32;
                            gain_final = cg;
                        }
                    }
                    ox += 2;
                }
                oy += 2;
            }
            if best_c_mse < mse {
                dx = best_cx;
                dy = best_cy;
                mse = best_c_mse;
            }
        }

        (dx, dy, mse as f32, gain_final as f32, true)
    }

    /// Dense `128x64` `8x8` affine-brightness-invariant IC-LK displacement + marginal variance.
    #[inline(never)]
    fn extract_dense_lk(
        &mut self,
        cur_luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        maint_mode: bool,
    ) -> (u32, f32) {
        let mut p_ref = PatchRefData::empty();
        let mut cur_buf = [0.0f32; 64];
        let mut cur_mask = [0u8; 64];

        let gx0 = self.ctrl.gauge_x0 as usize;
        let gy0 = self.ctrl.gauge_y0 as usize;
        let gx1 = self.ctrl.gauge_x1 as usize;
        let gy1 = self.ctrl.gauge_y1 as usize;

        let mut valid_nodes = 0u32;
        let mut disp2_sum = 0.0f64;

        let mut j = 0usize;
        while j < BOS_GRID_H {
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let idx = j * BOS_GRID_W + i;
                let (cx, cy) = node_coord(i, j, frame_w, frame_h);
                let mut accepted = false;
                let mut node_clear_ok = false;

                if build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, &mut p_ref) {
                    let mut ex = 0.0f32;
                    let mut ey = 0.0f32;
                    let mut last_gain = 1.0f64;
                    let mut iter_ok = true;

                    let mut it = 0usize;
                    while it < 2 {
                        if let Some(fit) = eval_cur_patch_8x8(
                            cur_luma,
                            frame_w,
                            frame_h,
                            cx,
                            cy,
                            ex,
                            ey,
                            &p_ref,
                            &mut cur_buf,
                            &mut cur_mask,
                        ) {
                            if !fit.gain.is_finite() || fit.gain < 0.5 || fit.gain > 2.0 {
                                iter_ok = false;
                                break;
                            }
                            last_gain = fit.gain;
                            let inv_det = 1.0 / fit.det;
                            let step_x = (-(fit.h_yy * fit.bx - fit.h_xy * fit.by)
                                * inv_det
                                / fit.gain) as f32;
                            let step_y = (-(-fit.h_xy * fit.bx + fit.h_xx * fit.by)
                                * inv_det
                                / fit.gain) as f32;
                            if !step_x.is_finite() || !step_y.is_finite() {
                                iter_ok = false;
                                break;
                            }
                            ex += step_x;
                            ey += step_y;
                            if ex * ex + ey * ey > 2.25 {
                                iter_ok = false;
                                break;
                            }
                            if step_x * step_x + step_y * step_y < 1e-4 {
                                break;
                            }
                        } else {
                            iter_ok = false;
                            break;
                        }
                        it += 1;
                    }

                    if iter_ok && (ex * ex + ey * ey) <= 2.25 {
                        // Evaluate variance and residual consistency at the exact final warp (ex, ey).
                        if let Some(final_fit) = eval_cur_patch_8x8(
                            cur_luma,
                            frame_w,
                            frame_h,
                            cx,
                            cy,
                            ex,
                            ey,
                            &p_ref,
                            &mut cur_buf,
                            &mut cur_mask,
                        ) {
                            let m_cnt = final_fit.cnt;
                            let gain = final_fit.gain;
                            let offset = final_fit.offset;
                            if gain.is_finite() && gain >= 0.5 && gain <= 2.0 && m_cnt >= 40 {
                                let mut sum_r2 = 0.0f64;
                                let mut k = 0usize;
                                while k < 64 {
                                    if cur_mask[k] != 0 {
                                        let r = (cur_buf[k] as f64)
                                            - (gain * (p_ref.t[k] as f64) + offset);
                                        sum_r2 += r * r;
                                    }
                                    k += 1;
                                }
                                let final_mse = sum_r2 / (m_cnt as f64);
                                let noise_base = self.ref_noise_var[idx] as f64;
                                let res_limit = (25.0 * noise_base)
                                    .max(0.035 * 0.035)
                                    .max(0.35 * p_ref.ref_var);
                                if final_mse.is_finite() && final_mse <= res_limit {
                                    let dof = (m_cnt - 4).max(1) as f64;
                                    let inv_det = 1.0 / final_fit.det;
                                    let sigma_ref2 = (sum_r2 / dof) / (gain * gain);
                                    let var_floor = 0.015 * 0.015;
                                    let vx = (sigma_ref2 * final_fit.h_yy * inv_det + var_floor)
                                        .max(1e-4);
                                    let vy = (sigma_ref2 * final_fit.h_xx * inv_det + var_floor)
                                        .max(1e-4);
                                    if ex.is_finite()
                                        && ey.is_finite()
                                        && vx.is_finite()
                                        && vy.is_finite()
                                        && vx <= 0.1225
                                        && vy <= 0.1225
                                    {
                                        self.planes[BOS_PLANE_EPS_X * BOS_NODES + idx] = ex;
                                        self.planes[BOS_PLANE_EPS_Y * BOS_NODES + idx] = ey;
                                        self.planes[BOS_PLANE_VAR_X * BOS_NODES + idx] = vx as f32;
                                        self.planes[BOS_PLANE_VAR_Y * BOS_NODES + idx] = vy as f32;
                                        self.valid_mask[idx] = BOS_VALID_DISP;
                                        valid_nodes += 1;
                                        disp2_sum += (ex as f64) * (ex as f64) + (ey as f64) * (ey as f64);
                                        accepted = true;
                                        if maint_mode
                                            && i >= gx0
                                            && i < gx1
                                            && j >= gy0
                                            && j < gy1
                                            && (ex * ex + ey * ey) <= 0.25 * 0.25
                                        {
                                            node_clear_ok = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let _ = last_gain;
                }

                if !accepted {
                    self.planes[BOS_PLANE_EPS_X * BOS_NODES + idx] = 0.0;
                    self.planes[BOS_PLANE_EPS_Y * BOS_NODES + idx] = 0.0;
                    self.planes[BOS_PLANE_VAR_X * BOS_NODES + idx] = 0.0;
                    self.planes[BOS_PLANE_VAR_Y * BOS_NODES + idx] = 0.0;
                    self.valid_mask[idx] = 0;
                }
                if node_clear_ok {
                    self.clear_streak[idx] = self.clear_streak[idx].saturating_add(1);
                } else {
                    self.clear_streak[idx] = 0;
                }
                i += 1;
            }
            j += 1;
        }

        let rms_disp = if valid_nodes > 0 {
            sqrt_f64(disp2_sum / (valid_nodes as f64)) as f32
        } else {
            0.0
        };
        (valid_nodes, rms_disp)
    }

    /// Mode 2 selective maintenance: updates each eligible pixel INSIDE `gauge_roi` and OUTSIDE
    /// frozen anchor halos ONCE per frame after `120` consecutive clear samples (`alpha = 1/4096`).
    #[inline(never)]
    fn update_mode2_reference(&mut self, cur_luma: &[f32], frame_w: usize, frame_h: usize) {
        if self.ctrl.requested_mode != BOS_MODE_MAINT || self.ctrl.clear_consec_frames < 120 {
            return;
        }
        let gx0 = self.ctrl.gauge_x0 as usize;
        let gy0 = self.ctrl.gauge_y0 as usize;
        let gx1 = self.ctrl.gauge_x1 as usize;
        let gy1 = self.ctrl.gauge_y1 as usize;
        if gx0 >= gx1 || gy0 >= gy1 || gx1 > BOS_GRID_W || gy1 > BOS_GRID_H {
            return;
        }

        let hx = (frame_w as f32) / (BOS_GRID_W as f32);
        let hy = (frame_h as f32) / (BOS_GRID_H as f32);
        let px0 = ((gx0 as f32) * hx) as usize;
        let px1 = ((((gx1 as f32) * hx) + 0.999) as usize).min(frame_w);
        let py0 = ((gy0 as f32) * hy) as usize;
        let py1 = ((((gy1 as f32) * hy) + 0.999) as usize).min(frame_h);

        let k_cnt = self.ctrl.anchor_count as usize;
        let inv_hx = 1.0 / hx;
        let inv_hy = 1.0 / hy;

        let mut py = py0;
        while py < py1 {
            let fy = (py as f32 + 0.5) * inv_hy;
            let nj = (floor_f32(fy) as usize).clamp(gy0, gy1 - 1);
            let row_off = py * frame_w;
            let mut px = px0;
            while px < px1 {
                let mut inside_anchor_halo = false;
                let pxf = px as f32;
                let pyf = py as f32;
                let mut a = 0usize;
                while a < k_cnt {
                    let node = self.ctrl.anchor_node_idx[a] as usize;
                    let (acx, acy) = node_coord(node & 127, node >> 7, frame_w, frame_h);
                    if (pxf - acx).abs() <= 6.0 && (pyf - acy).abs() <= 6.0 {
                        inside_anchor_halo = true;
                        break;
                    }
                    a += 1;
                }
                if !inside_anchor_halo {
                    let fx = (px as f32 + 0.5) * inv_hx;
                    let ni = (floor_f32(fx) as usize).clamp(gx0, gx1 - 1);
                    let node = nj * BOS_GRID_W + ni;
                    if (self.valid_mask[node] & BOS_VALID_DISP) != 0
                        && self.clear_streak[node] >= BOS_MAINT_STREAK_REQ
                    {
                        let idx = row_off + px;
                        self.ref_luma[idx] += (cur_luma[idx] - self.ref_luma[idx]) * BOS_MAINT_ALPHA;
                    }
                }
                px += 1;
            }
            py += 1;
        }
    }

    /// Computes exact discrete curl plaquettes and assembles variational edge weights + Neumann RHS.
    #[inline(never)]
    fn compute_curl_and_assemble_neumann(
        &mut self,
        frame_w: usize,
        frame_h: usize,
        use_existing_weights: bool,
    ) -> (u32, f32, f32, bool) {
        let hx = (frame_w as f64) / (BOS_GRID_W as f64);
        let hy = (frame_h as f64) / (BOS_GRID_H as f64);
        let inv_hx = 1.0 / hx;
        let inv_hy = 1.0 / hy;

        // Clear curl plane, curl bit, phi plane, phi bit, solve_b
        let mut idx = 0usize;
        while idx < BOS_NODES {
            self.planes[BOS_PLANE_CURL * BOS_NODES + idx] = 0.0;
            self.planes[BOS_PLANE_PHI * BOS_NODES + idx] = 0.0;
            self.valid_mask[idx] &= BOS_VALID_DISP;
            self.solve_b[idx] = 0.0;
            idx += 1;
        }

        // 1. Discrete curl on 4-valid-corner plaquettes
        let mut curl_cnt = 0u32;
        let mut curl_sum2 = 0.0f64;
        let mut max_abs_curl = 0.0f64;

        let mut j = 0usize;
        while j < BOS_GRID_H - 1 {
            let mut i = 0usize;
            while i < BOS_GRID_W - 1 {
                let tl = j * BOS_GRID_W + i;
                let tr = tl + 1;
                let bl = tl + BOS_GRID_W;
                let br = bl + 1;
                if (self.valid_mask[tl] & BOS_VALID_DISP) != 0
                    && (self.valid_mask[tr] & BOS_VALID_DISP) != 0
                    && (self.valid_mask[bl] & BOS_VALID_DISP) != 0
                    && (self.valid_mask[br] & BOS_VALID_DISP) != 0
                {
                    let ex_tl = self.planes[BOS_PLANE_EPS_X * BOS_NODES + tl] as f64;
                    let ex_tr = self.planes[BOS_PLANE_EPS_X * BOS_NODES + tr] as f64;
                    let ex_bl = self.planes[BOS_PLANE_EPS_X * BOS_NODES + bl] as f64;
                    let ex_br = self.planes[BOS_PLANE_EPS_X * BOS_NODES + br] as f64;

                    let ey_tl = self.planes[BOS_PLANE_EPS_Y * BOS_NODES + tl] as f64;
                    let ey_tr = self.planes[BOS_PLANE_EPS_Y * BOS_NODES + tr] as f64;
                    let ey_bl = self.planes[BOS_PLANE_EPS_Y * BOS_NODES + bl] as f64;
                    let ey_br = self.planes[BOS_PLANE_EPS_Y * BOS_NODES + br] as f64;

                    let ey_left = 0.5 * (ey_tl + ey_bl);
                    let ey_right = 0.5 * (ey_tr + ey_br);
                    let ex_top = 0.5 * (ex_tl + ex_tr);
                    let ex_bot = 0.5 * (ex_bl + ex_br);

                    let omega = (ey_right - ey_left) * inv_hx - (ex_bot - ex_top) * inv_hy;
                    let omega_f32 = omega as f32;
                    if ex_tl.is_finite()
                        && ex_tr.is_finite()
                        && ex_bl.is_finite()
                        && ex_br.is_finite()
                        && ey_tl.is_finite()
                        && ey_tr.is_finite()
                        && ey_bl.is_finite()
                        && ey_br.is_finite()
                        && omega.is_finite()
                        && omega_f32.is_finite()
                    {
                        self.planes[BOS_PLANE_CURL * BOS_NODES + tl] = omega_f32;
                        self.valid_mask[tl] |= BOS_VALID_CURL;
                        curl_cnt += 1;
                        curl_sum2 += omega * omega;
                        let ao = omega.abs();
                        if ao > max_abs_curl {
                            max_abs_curl = ao;
                        }
                    }
                }
                i += 1;
            }
            j += 1;
        }
        let mut curl_rms = if curl_cnt > 0 {
            sqrt_f64(curl_sum2 / (curl_cnt as f64)) as f32
        } else {
            0.0
        };
        let mut max_curl_f32 = max_abs_curl as f32;
        if !curl_rms.is_finite() || !max_curl_f32.is_finite() {
            let mut k = 0usize;
            while k < BOS_NODES {
                self.planes[BOS_PLANE_CURL * BOS_NODES + k] = 0.0;
                self.valid_mask[k] &= !BOS_VALID_CURL;
                k += 1;
            }
            curl_cnt = 0;
            curl_rms = 0.0;
            max_curl_f32 = 0.0;
        }

        // 2. Variational edge weights & RHS b = A^T W eps_bar
        const V0: f64 = 0.01;
        const V_MIN: f64 = 1e-4;
        if !use_existing_weights {
            self.w_x.fill(0.0);
            self.w_y.fill(0.0);
        }

        let mut all_valid = true;
        let mut wx_ref = -1.0f32;
        let mut wy_ref = -1.0f32;
        let mut uniform_weights = true;

        let mut j = 0usize;
        while j < BOS_GRID_H {
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let u = j * BOS_GRID_W + i;
                let u_valid = (self.valid_mask[u] & BOS_VALID_DISP) != 0;
                if !u_valid {
                    all_valid = false;
                }
                if i + 1 < BOS_GRID_W {
                    let v = u + 1;
                    if u_valid && (self.valid_mask[v] & BOS_VALID_DISP) != 0 {
                        let w_f32 = if use_existing_weights {
                            self.w_x[u]
                        } else {
                            let su = sqrt_f64(
                                (self.planes[BOS_PLANE_VAR_X * BOS_NODES + u] as f64).max(0.0),
                            );
                            let sv = sqrt_f64(
                                (self.planes[BOS_PLANE_VAR_X * BOS_NODES + v] as f64).max(0.0),
                            );
                            let ve = 0.25 * (su + sv) * (su + sv);
                            (V0 / ve.max(V_MIN)).clamp(0.25, 1.0) as f32
                        };
                        self.w_x[u] = w_f32;
                        if w_f32 > 0.0 && w_f32.is_finite() {
                            if wx_ref < 0.0 {
                                wx_ref = w_f32;
                            } else if w_f32 != wx_ref {
                                uniform_weights = false;
                            }
                            let w_e = w_f32 as f64;
                            let eps_bar = 0.5
                                * ((self.planes[BOS_PLANE_EPS_X * BOS_NODES + u] as f64)
                                    + (self.planes[BOS_PLANE_EPS_X * BOS_NODES + v] as f64));
                            let flux = w_e * eps_bar * inv_hx;
                            self.solve_b[u] -= flux;
                            self.solve_b[v] += flux;
                        } else {
                            uniform_weights = false;
                        }
                    } else {
                        self.w_x[u] = 0.0;
                        uniform_weights = false;
                    }
                }
                if j + 1 < BOS_GRID_H {
                    let v = u + BOS_GRID_W;
                    if u_valid && (self.valid_mask[v] & BOS_VALID_DISP) != 0 {
                        let w_f32 = if use_existing_weights {
                            self.w_y[u]
                        } else {
                            let su = sqrt_f64(
                                (self.planes[BOS_PLANE_VAR_Y * BOS_NODES + u] as f64).max(0.0),
                            );
                            let sv = sqrt_f64(
                                (self.planes[BOS_PLANE_VAR_Y * BOS_NODES + v] as f64).max(0.0),
                            );
                            let ve = 0.25 * (su + sv) * (su + sv);
                            (V0 / ve.max(V_MIN)).clamp(0.25, 1.0) as f32
                        };
                        self.w_y[u] = w_f32;
                        if w_f32 > 0.0 && w_f32.is_finite() {
                            if wy_ref < 0.0 {
                                wy_ref = w_f32;
                            } else if w_f32 != wy_ref {
                                uniform_weights = false;
                            }
                            let w_e = w_f32 as f64;
                            let eps_bar = 0.5
                                * ((self.planes[BOS_PLANE_EPS_Y * BOS_NODES + u] as f64)
                                    + (self.planes[BOS_PLANE_EPS_Y * BOS_NODES + v] as f64));
                            let flux = w_e * eps_bar * inv_hy;
                            self.solve_b[u] -= flux;
                            self.solve_b[v] += flux;
                        } else {
                            uniform_weights = false;
                        }
                    } else {
                        self.w_y[u] = 0.0;
                        uniform_weights = false;
                    }
                }
                i += 1;
            }
            j += 1;
        }

        let is_direct = all_valid && uniform_weights && wx_ref > 0.0 && wy_ref > 0.0;
        (curl_cnt, curl_rms, max_curl_f32, is_direct)
    }

    /// Iterative BFS 4-connected positive-edge component labeling (`comp_id` in `1..=num_comps`).
    /// Isolated nodes with 0 incident positive edges keep `comp_id = 0`.
    #[inline(never)]
    fn label_pos_edge_components(&mut self) -> u32 {
        self.comp_id.fill(0);
        self.comp_count.fill(0);
        let mut num_comps = 0u32;

        let mut u = 0usize;
        while u < BOS_NODES {
            if self.comp_id[u] == 0 {
                let i = u & (BOS_GRID_W - 1);
                let j = u >> 7;
                let has_edge = (i + 1 < BOS_GRID_W && self.w_x[u] > 0.0)
                    || (i > 0 && self.w_x[u - 1] > 0.0)
                    || (j + 1 < BOS_GRID_H && self.w_y[u] > 0.0)
                    || (j > 0 && self.w_y[u - BOS_GRID_W] > 0.0);
                if has_edge {
                    num_comps += 1;
                    let cid = num_comps as u16;
                    let mut qh = 0usize;
                    let mut qt = 0usize;
                    self.comp_id[u] = cid;
                    self.bfs_queue[qt] = u as u16;
                    qt += 1;
                    let mut c_size = 0u16;

                    while qh < qt {
                        let cur = self.bfs_queue[qh] as usize;
                        qh += 1;
                        c_size += 1;
                        let ci = cur & (BOS_GRID_W - 1);
                        let cj = cur >> 7;
                        if ci + 1 < BOS_GRID_W && self.w_x[cur] > 0.0 && self.comp_id[cur + 1] == 0
                        {
                            self.comp_id[cur + 1] = cid;
                            self.bfs_queue[qt] = (cur + 1) as u16;
                            qt += 1;
                        }
                        if ci > 0 && self.w_x[cur - 1] > 0.0 && self.comp_id[cur - 1] == 0 {
                            self.comp_id[cur - 1] = cid;
                            self.bfs_queue[qt] = (cur - 1) as u16;
                            qt += 1;
                        }
                        if cj + 1 < BOS_GRID_H
                            && self.w_y[cur] > 0.0
                            && self.comp_id[cur + BOS_GRID_W] == 0
                        {
                            self.comp_id[cur + BOS_GRID_W] = cid;
                            self.bfs_queue[qt] = (cur + BOS_GRID_W) as u16;
                            qt += 1;
                        }
                        if cj > 0 && self.w_y[cur - BOS_GRID_W] > 0.0 && self.comp_id[cur - BOS_GRID_W] == 0
                        {
                            self.comp_id[cur - BOS_GRID_W] = cid;
                            self.bfs_queue[qt] = (cur - BOS_GRID_W) as u16;
                            qt += 1;
                        }
                    }
                    self.comp_count[num_comps as usize] = c_size;
                }
            }
            u += 1;
        }
        num_comps
    }

    /// Iterative Cooley-Tukey radix-2 `f64x2` forward complex FFT on `fft_line` (`m = 64` or `32`).
    /// `fft_line[0..2*m]` holds real lanes (`f64x2` paired), `fft_line[2*m..4*m]` holds imag lanes.
    /// Note: caller writes input in bit-reversed order; `m` must be `64` or `32`.
    #[inline(always)]
    fn fft_radix2_paired(&mut self, m: usize) {
        let mut half = 1usize;
        let mut step_tw = 32usize;
        while half < m {
            let len = half << 1;
            let mut i = 0usize;
            while i < m {
                let mut k = 0usize;
                let mut tw_idx = 0usize;
                while k < half {
                    let wr = f64x2_splat(DCT_TABLES.tw64_re[tw_idx]);
                    let wi = f64x2_splat(DCT_TABLES.tw64_im[tw_idx]);
                    let p0 = i + k;
                    let p1 = p0 + half;
                    let u_r = f64x2(self.fft_line[p0 * 2], self.fft_line[p0 * 2 + 1]);
                    let u_i = f64x2(self.fft_line[(m + p0) * 2], self.fft_line[(m + p0) * 2 + 1]);
                    let b_r = f64x2(self.fft_line[p1 * 2], self.fft_line[p1 * 2 + 1]);
                    let b_i = f64x2(self.fft_line[(m + p1) * 2], self.fft_line[(m + p1) * 2 + 1]);

                    let v_r = f64x2_sub(f64x2_mul(b_r, wr), f64x2_mul(b_i, wi));
                    let v_i = f64x2_add(f64x2_mul(b_r, wi), f64x2_mul(b_i, wr));

                    let s_r = f64x2_add(u_r, v_r);
                    let s_i = f64x2_add(u_i, v_i);
                    let d_r = f64x2_sub(u_r, v_r);
                    let d_i = f64x2_sub(u_i, v_i);

                    self.fft_line[p0 * 2] = f64x2_extract_lane::<0>(s_r);
                    self.fft_line[p0 * 2 + 1] = f64x2_extract_lane::<1>(s_r);
                    self.fft_line[(m + p0) * 2] = f64x2_extract_lane::<0>(s_i);
                    self.fft_line[(m + p0) * 2 + 1] = f64x2_extract_lane::<1>(s_i);

                    self.fft_line[p1 * 2] = f64x2_extract_lane::<0>(d_r);
                    self.fft_line[p1 * 2 + 1] = f64x2_extract_lane::<1>(d_r);
                    self.fft_line[(m + p1) * 2] = f64x2_extract_lane::<0>(d_i);
                    self.fft_line[(m + p1) * 2 + 1] = f64x2_extract_lane::<1>(d_i);

                    k += 1;
                    tw_idx += step_tw;
                }
                i += len;
            }
            half = len;
            step_tw >>= 1;
        }
    }

    /// Paired orthonormal 1D DCT-II across two rows (`N = 128`, `m = 64`) or two columns (`N = 64`, `m = 32`).
    /// Operates in-place on `solve_z` using `fft_line` (`f64x2` SIMD128).
    #[inline(never)]
    fn dct2_2d_solve_z(&mut self) {
        // 1. 64 rows of N = 128 (32 pairs)
        let mut r = 0usize;
        while r < BOS_GRID_H {
            let r0 = r * BOS_GRID_W;
            let r1 = r0 + BOS_GRID_W;
            let mut dst = 0usize;
            while dst < 64 {
                let p = DCT_TABLES.bitrev64[dst] as usize;
                let (e_idx, o_idx) = if p < 32 {
                    (4 * p, 4 * p + 2)
                } else {
                    let q = 63 - p;
                    (4 * q + 3, 4 * q + 1)
                };
                self.fft_line[dst * 2] = self.solve_z[r0 + e_idx];
                self.fft_line[dst * 2 + 1] = self.solve_z[r1 + e_idx];
                self.fft_line[(64 + dst) * 2] = self.solve_z[r0 + o_idx];
                self.fft_line[(64 + dst) * 2 + 1] = self.solve_z[r1 + o_idx];
                dst += 1;
            }
            self.fft_radix2_paired(64);

            let e0 = f64x2(self.fft_line[0], self.fft_line[1]);
            let o0 = f64x2(self.fft_line[128], self.fft_line[129]);
            let dc = f64x2_mul(f64x2_add(e0, o0), f64x2_splat(DCT_TABLES.scale_dc_128));
            let nyq = f64x2_mul(f64x2_sub(e0, o0), f64x2_splat(DCT_TABLES.scale_nyq_128));
            self.solve_z[r0] = f64x2_extract_lane::<0>(dc);
            self.solve_z[r1] = f64x2_extract_lane::<1>(dc);
            self.solve_z[r0 + 64] = f64x2_extract_lane::<0>(nyq);
            self.solve_z[r1 + 64] = f64x2_extract_lane::<1>(nyq);

            let half = f64x2_splat(0.5);
            let sc = f64x2_splat(DCT_TABLES.scale_fwd_128);
            let mut k = 1usize;
            while k < 64 {
                let mk = 64 - k;
                let zkr = f64x2(self.fft_line[k * 2], self.fft_line[k * 2 + 1]);
                let zki = f64x2(self.fft_line[(64 + k) * 2], self.fft_line[(64 + k) * 2 + 1]);
                let zmr = f64x2(self.fft_line[mk * 2], self.fft_line[mk * 2 + 1]);
                let zmi = f64x2(-self.fft_line[(64 + mk) * 2], -self.fft_line[(64 + mk) * 2 + 1]);

                let er = f64x2_mul(half, f64x2_add(zkr, zmr));
                let ei = f64x2_mul(half, f64x2_add(zki, zmi));
                let or = f64x2_mul(half, f64x2_sub(zki, zmi));
                let oi = f64x2_mul(half, f64x2_sub(zmr, zkr));

                let wnr = f64x2_splat(DCT_TABLES.wn128_re[k]);
                let wni = f64x2_splat(DCT_TABLES.wn128_im[k]);
                let yr = f64x2_add(er, f64x2_sub(f64x2_mul(or, wnr), f64x2_mul(oi, wni)));
                let yi = f64x2_add(ei, f64x2_add(f64x2_mul(or, wni), f64x2_mul(oi, wnr)));

                let wdr = f64x2_splat(DCT_TABLES.wd128_re[k]);
                let wdi = f64x2_splat(DCT_TABLES.wd128_im[k]);
                let xk = f64x2_mul(sc, f64x2_sub(f64x2_mul(yr, wdr), f64x2_mul(yi, wdi)));
                let xnk = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.scale_fwd_128),
                    f64x2_add(f64x2_mul(yr, wdi), f64x2_mul(yi, wdr)),
                );
                self.solve_z[r0 + k] = f64x2_extract_lane::<0>(xk);
                self.solve_z[r1 + k] = f64x2_extract_lane::<1>(xk);
                self.solve_z[r0 + 128 - k] = f64x2_extract_lane::<0>(xnk);
                self.solve_z[r1 + 128 - k] = f64x2_extract_lane::<1>(xnk);
                k += 1;
            }
            r += 2;
        }

        // 2. 128 columns of N = 64 (64 adjacent column pairs)
        let mut c = 0usize;
        while c < BOS_GRID_W {
            let mut dst = 0usize;
            while dst < 32 {
                let p = DCT_TABLES.bitrev32[dst] as usize;
                let (e_idx, o_idx) = if p < 16 {
                    (4 * p, 4 * p + 2)
                } else {
                    let q = 31 - p;
                    (4 * q + 3, 4 * q + 1)
                };
                let erow = e_idx * BOS_GRID_W + c;
                let orow = o_idx * BOS_GRID_W + c;
                self.fft_line[dst * 2] = self.solve_z[erow];
                self.fft_line[dst * 2 + 1] = self.solve_z[erow + 1];
                self.fft_line[(32 + dst) * 2] = self.solve_z[orow];
                self.fft_line[(32 + dst) * 2 + 1] = self.solve_z[orow + 1];
                dst += 1;
            }
            self.fft_radix2_paired(32);

            let e0 = f64x2(self.fft_line[0], self.fft_line[1]);
            let o0 = f64x2(self.fft_line[64], self.fft_line[65]);
            let dc = f64x2_mul(f64x2_add(e0, o0), f64x2_splat(DCT_TABLES.scale_dc_64));
            let nyq = f64x2_mul(f64x2_sub(e0, o0), f64x2_splat(DCT_TABLES.scale_nyq_64));
            self.solve_z[c] = f64x2_extract_lane::<0>(dc);
            self.solve_z[c + 1] = f64x2_extract_lane::<1>(dc);
            self.solve_z[32 * BOS_GRID_W + c] = f64x2_extract_lane::<0>(nyq);
            self.solve_z[32 * BOS_GRID_W + c + 1] = f64x2_extract_lane::<1>(nyq);

            let half = f64x2_splat(0.5);
            let sc = f64x2_splat(DCT_TABLES.scale_fwd_64);
            let mut k = 1usize;
            while k < 32 {
                let mk = 32 - k;
                let zkr = f64x2(self.fft_line[k * 2], self.fft_line[k * 2 + 1]);
                let zki = f64x2(self.fft_line[(32 + k) * 2], self.fft_line[(32 + k) * 2 + 1]);
                let zmr = f64x2(self.fft_line[mk * 2], self.fft_line[mk * 2 + 1]);
                let zmi = f64x2(-self.fft_line[(32 + mk) * 2], -self.fft_line[(32 + mk) * 2 + 1]);

                let er = f64x2_mul(half, f64x2_add(zkr, zmr));
                let ei = f64x2_mul(half, f64x2_add(zki, zmi));
                let or = f64x2_mul(half, f64x2_sub(zki, zmi));
                let oi = f64x2_mul(half, f64x2_sub(zmr, zkr));

                let wnr = f64x2_splat(DCT_TABLES.wn128_re[k * 2]);
                let wni = f64x2_splat(DCT_TABLES.wn128_im[k * 2]);
                let yr = f64x2_add(er, f64x2_sub(f64x2_mul(or, wnr), f64x2_mul(oi, wni)));
                let yi = f64x2_add(ei, f64x2_add(f64x2_mul(or, wni), f64x2_mul(oi, wnr)));

                let wdr = f64x2_splat(DCT_TABLES.wd128_re[k * 2]);
                let wdi = f64x2_splat(DCT_TABLES.wd128_im[k * 2]);
                let xk = f64x2_mul(sc, f64x2_sub(f64x2_mul(yr, wdr), f64x2_mul(yi, wdi)));
                let xnk = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.scale_fwd_64),
                    f64x2_add(f64x2_mul(yr, wdi), f64x2_mul(yi, wdr)),
                );
                let row_k = k * BOS_GRID_W + c;
                let row_nk = (64 - k) * BOS_GRID_W + c;
                self.solve_z[row_k] = f64x2_extract_lane::<0>(xk);
                self.solve_z[row_k + 1] = f64x2_extract_lane::<1>(xk);
                self.solve_z[row_nk] = f64x2_extract_lane::<0>(xnk);
                self.solve_z[row_nk + 1] = f64x2_extract_lane::<1>(xnk);
                k += 1;
            }
            c += 2;
        }
    }

    /// Paired orthonormal 2D DCT-III (inverse of DCT-II) across `solve_z` using `fft_line`.
    #[inline(never)]
    fn dct3_2d_solve_z(&mut self) {
        // 1. 128 columns of N = 64 (64 pairs)
        let s0_64 = f64x2_splat(DCT_TABLES.sqrt_64);
        let sk_64 = f64x2_splat(DCT_TABLES.sqrt_64 * 0.70710678118654752440);
        let half = f64x2_splat(0.5);

        let mut c = 0usize;
        while c < BOS_GRID_W {
            let y0 = f64x2_mul(s0_64, f64x2(self.solve_z[c], self.solve_z[c + 1]));
            let ym = f64x2_mul(
                s0_64,
                f64x2(
                    self.solve_z[32 * BOS_GRID_W + c],
                    self.solve_z[32 * BOS_GRID_W + c + 1],
                ),
            );
            // Store yRe[0..32] at fft_line[256..320], yIm[0..32] at fft_line[320..384]
            self.fft_line[256] = f64x2_extract_lane::<0>(y0);
            self.fft_line[257] = f64x2_extract_lane::<1>(y0);
            self.fft_line[320] = 0.0;
            self.fft_line[321] = 0.0;
            let mut k = 1usize;
            while k < 32 {
                let row_k = k * BOS_GRID_W + c;
                let row_nk = (64 - k) * BOS_GRID_W + c;
                let ur = f64x2_mul(sk_64, f64x2(self.solve_z[row_k], self.solve_z[row_k + 1]));
                let ui = f64x2_mul(sk_64, f64x2(-self.solve_z[row_nk], -self.solve_z[row_nk + 1]));
                let wdr = f64x2_splat(DCT_TABLES.wd128_re[k * 2]);
                let wdi = f64x2_splat(-DCT_TABLES.wd128_im[k * 2]); // +pi*k/(2N)
                let yr = f64x2_sub(f64x2_mul(ur, wdr), f64x2_mul(ui, wdi));
                let yi = f64x2_add(f64x2_mul(ur, wdi), f64x2_mul(ui, wdr));
                self.fft_line[256 + k * 2] = f64x2_extract_lane::<0>(yr);
                self.fft_line[256 + k * 2 + 1] = f64x2_extract_lane::<1>(yr);
                self.fft_line[320 + k * 2] = f64x2_extract_lane::<0>(yi);
                self.fft_line[320 + k * 2 + 1] = f64x2_extract_lane::<1>(yi);
                k += 1;
            }

            // Build bit-reversed conj(Z) directly into fft_line[0..128]
            let z0r = f64x2_mul(half, f64x2_add(y0, ym));
            let z0i = f64x2_mul(half, f64x2_sub(y0, ym));
            self.fft_line[0] = f64x2_extract_lane::<0>(z0r);
            self.fft_line[1] = f64x2_extract_lane::<1>(z0r);
            self.fft_line[64] = -f64x2_extract_lane::<0>(z0i); // conj
            self.fft_line[65] = -f64x2_extract_lane::<1>(z0i);

            let mut k = 1usize;
            while k < 32 {
                let dst = DCT_TABLES.bitrev32[k] as usize;
                let mk = 32 - k;
                let ykr = f64x2(self.fft_line[256 + k * 2], self.fft_line[256 + k * 2 + 1]);
                let yki = f64x2(self.fft_line[320 + k * 2], self.fft_line[320 + k * 2 + 1]);
                let ymr = f64x2(self.fft_line[256 + mk * 2], self.fft_line[256 + mk * 2 + 1]);
                let ymi = f64x2(-self.fft_line[320 + mk * 2], -self.fft_line[320 + mk * 2 + 1]);

                let er = f64x2_mul(half, f64x2_add(ykr, ymr));
                let ei = f64x2_mul(half, f64x2_add(yki, ymi));
                let dr = f64x2_mul(half, f64x2_sub(ykr, ymr));
                let di = f64x2_mul(half, f64x2_sub(yki, ymi));
                let wnr = f64x2_splat(DCT_TABLES.wn128_re[k * 2]);
                let wni = f64x2_splat(-DCT_TABLES.wn128_im[k * 2]); // +2*pi*k/N
                let or = f64x2_sub(f64x2_mul(dr, wnr), f64x2_mul(di, wni));
                let oi = f64x2_add(f64x2_mul(dr, wni), f64x2_mul(di, wnr));

                let zr = f64x2_sub(er, oi);
                let zi = f64x2_add(ei, or);
                self.fft_line[dst * 2] = f64x2_extract_lane::<0>(zr);
                self.fft_line[dst * 2 + 1] = f64x2_extract_lane::<1>(zr);
                self.fft_line[(32 + dst) * 2] = -f64x2_extract_lane::<0>(zi); // conj
                self.fft_line[(32 + dst) * 2 + 1] = -f64x2_extract_lane::<1>(zi);
                k += 1;
            }
            self.fft_radix2_paired(32);

            let inv_m = f64x2_splat(DCT_TABLES.inv_32);
            let mut r = 0usize;
            while r < 32 {
                let ye = f64x2_mul(inv_m, f64x2(self.fft_line[r * 2], self.fft_line[r * 2 + 1]));
                let yo = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.inv_32),
                    f64x2(self.fft_line[(32 + r) * 2], self.fft_line[(32 + r) * 2 + 1]),
                );
                let (e_idx, o_idx) = if r < 16 {
                    (4 * r, 4 * r + 2)
                } else {
                    let q = 31 - r;
                    (4 * q + 3, 4 * q + 1)
                };
                let erow = e_idx * BOS_GRID_W + c;
                let orow = o_idx * BOS_GRID_W + c;
                self.solve_z[erow] = f64x2_extract_lane::<0>(ye);
                self.solve_z[erow + 1] = f64x2_extract_lane::<1>(ye);
                self.solve_z[orow] = f64x2_extract_lane::<0>(yo);
                self.solve_z[orow + 1] = f64x2_extract_lane::<1>(yo);
                r += 1;
            }
            c += 2;
        }

        // 2. 64 rows of N = 128 (32 pairs)
        let s0_128 = f64x2_splat(DCT_TABLES.sqrt_128);
        let sk_128 = f64x2_splat(DCT_TABLES.sqrt_128 * 0.70710678118654752440);

        let mut r = 0usize;
        while r < BOS_GRID_H {
            let r0 = r * BOS_GRID_W;
            let r1 = r0 + BOS_GRID_W;
            let y0 = f64x2_mul(s0_128, f64x2(self.solve_z[r0], self.solve_z[r1]));
            let ym = f64x2_mul(s0_128, f64x2(self.solve_z[r0 + 64], self.solve_z[r1 + 64]));
            // Store yRe[0..64] at fft_line[256..384], yIm[0..64] at fft_line[384..512]
            self.fft_line[256] = f64x2_extract_lane::<0>(y0);
            self.fft_line[257] = f64x2_extract_lane::<1>(y0);
            self.fft_line[384] = 0.0;
            self.fft_line[385] = 0.0;
            let mut k = 1usize;
            while k < 64 {
                let ur = f64x2_mul(sk_128, f64x2(self.solve_z[r0 + k], self.solve_z[r1 + k]));
                let ui = f64x2_mul(
                    sk_128,
                    f64x2(-self.solve_z[r0 + 128 - k], -self.solve_z[r1 + 128 - k]),
                );
                let wdr = f64x2_splat(DCT_TABLES.wd128_re[k]);
                let wdi = f64x2_splat(-DCT_TABLES.wd128_im[k]);
                let yr = f64x2_sub(f64x2_mul(ur, wdr), f64x2_mul(ui, wdi));
                let yi = f64x2_add(f64x2_mul(ur, wdi), f64x2_mul(ui, wdr));
                self.fft_line[256 + k * 2] = f64x2_extract_lane::<0>(yr);
                self.fft_line[256 + k * 2 + 1] = f64x2_extract_lane::<1>(yr);
                self.fft_line[384 + k * 2] = f64x2_extract_lane::<0>(yi);
                self.fft_line[384 + k * 2 + 1] = f64x2_extract_lane::<1>(yi);
                k += 1;
            }

            let z0r = f64x2_mul(half, f64x2_add(y0, ym));
            let z0i = f64x2_mul(half, f64x2_sub(y0, ym));
            self.fft_line[0] = f64x2_extract_lane::<0>(z0r);
            self.fft_line[1] = f64x2_extract_lane::<1>(z0r);
            self.fft_line[128] = -f64x2_extract_lane::<0>(z0i);
            self.fft_line[129] = -f64x2_extract_lane::<1>(z0i);

            let mut k = 1usize;
            while k < 64 {
                let dst = DCT_TABLES.bitrev64[k] as usize;
                let mk = 64 - k;
                let ykr = f64x2(self.fft_line[256 + k * 2], self.fft_line[256 + k * 2 + 1]);
                let yki = f64x2(self.fft_line[384 + k * 2], self.fft_line[384 + k * 2 + 1]);
                let ymr = f64x2(self.fft_line[256 + mk * 2], self.fft_line[256 + mk * 2 + 1]);
                let ymi = f64x2(-self.fft_line[384 + mk * 2], -self.fft_line[384 + mk * 2 + 1]);

                let er = f64x2_mul(half, f64x2_add(ykr, ymr));
                let ei = f64x2_mul(half, f64x2_add(yki, ymi));
                let dr = f64x2_mul(half, f64x2_sub(ykr, ymr));
                let di = f64x2_mul(half, f64x2_sub(yki, ymi));
                let wnr = f64x2_splat(DCT_TABLES.wn128_re[k]);
                let wni = f64x2_splat(-DCT_TABLES.wn128_im[k]);
                let or = f64x2_sub(f64x2_mul(dr, wnr), f64x2_mul(di, wni));
                let oi = f64x2_add(f64x2_mul(dr, wni), f64x2_mul(di, wnr));

                let zr = f64x2_sub(er, oi);
                let zi = f64x2_add(ei, or);
                self.fft_line[dst * 2] = f64x2_extract_lane::<0>(zr);
                self.fft_line[dst * 2 + 1] = f64x2_extract_lane::<1>(zr);
                self.fft_line[(64 + dst) * 2] = -f64x2_extract_lane::<0>(zi);
                self.fft_line[(64 + dst) * 2 + 1] = -f64x2_extract_lane::<1>(zi);
                k += 1;
            }
            self.fft_radix2_paired(64);

            let inv_m = f64x2_splat(DCT_TABLES.inv_64);
            let mut idx_r = 0usize;
            while idx_r < 64 {
                let ye = f64x2_mul(inv_m, f64x2(self.fft_line[idx_r * 2], self.fft_line[idx_r * 2 + 1]));
                let yo = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.inv_64),
                    f64x2(
                        self.fft_line[(64 + idx_r) * 2],
                        self.fft_line[(64 + idx_r) * 2 + 1],
                    ),
                );
                let (e_idx, o_idx) = if idx_r < 32 {
                    (4 * idx_r, 4 * idx_r + 2)
                } else {
                    let q = 63 - idx_r;
                    (4 * q + 3, 4 * q + 1)
                };
                self.solve_z[r0 + e_idx] = f64x2_extract_lane::<0>(ye);
                self.solve_z[r1 + e_idx] = f64x2_extract_lane::<1>(ye);
                self.solve_z[r0 + o_idx] = f64x2_extract_lane::<0>(yo);
                self.solve_z[r1 + o_idx] = f64x2_extract_lane::<1>(yo);
                idx_r += 1;
            }
            r += 2;
        }
    }

    /// Runs `L_0^\dagger` in-place on `solve_z` using uniform positive weights `(wx_c, wy_c)`
    /// and grid spacings `(hx, hy)`.
    #[inline(never)]
    fn apply_dct_poisson_solve_z(&mut self, wx_c: f64, wy_c: f64, hx: f64, hy: f64) {
        self.dct2_2d_solve_z();
        let inv_hx2 = wx_c / (hx * hx);
        let inv_hy2 = wy_c / (hy * hy);
        let mut l = 0usize;
        while l < BOS_GRID_H {
            let ly = DCT_TABLES.sin2_y[l] * inv_hy2;
            let row = l * BOS_GRID_W;
            let mut k = 0usize;
            while k < BOS_GRID_W {
                if k == 0 && l == 0 {
                    self.solve_z[0] = 0.0;
                } else {
                    let lam = DCT_TABLES.sin2_x[k] * inv_hx2 + ly;
                    self.solve_z[row + k] /= lam;
                }
                k += 1;
            }
            l += 1;
        }
        self.dct3_2d_solve_z();
    }

    /// Applies sparse weighted Neumann Laplacian `solve_lp = L * p` (`1..=BOS_NODES`).
    #[inline(never)]
    fn apply_laplacian_p_to_lp(&mut self, inv_hx2: f64, inv_hy2: f64, use_x_buf: bool) {
        self.solve_lp.fill(0.0);
        let mut j = 0usize;
        while j < BOS_GRID_H {
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let u = j * BOS_GRID_W + i;
                let vu = if use_x_buf { self.solve_x[u] } else { self.solve_p[u] };
                if i + 1 < BOS_GRID_W && self.w_x[u] > 0.0 {
                    let v1 = if use_x_buf {
                        self.solve_x[u + 1]
                    } else {
                        self.solve_p[u + 1]
                    };
                    let fl = (self.w_x[u] as f64) * (vu - v1) * inv_hx2;
                    self.solve_lp[u] += fl;
                    self.solve_lp[u + 1] -= fl;
                }
                if j + 1 < BOS_GRID_H && self.w_y[u] > 0.0 {
                    let v1 = if use_x_buf {
                        self.solve_x[u + BOS_GRID_W]
                    } else {
                        self.solve_p[u + BOS_GRID_W]
                    };
                    let fl = (self.w_y[u] as f64) * (vu - v1) * inv_hy2;
                    self.solve_lp[u] += fl;
                    self.solve_lp[u + BOS_GRID_W] -= fl;
                }
                i += 1;
            }
            j += 1;
        }
    }

    /// Componentwise zero-mean projector `P` (zeroing isolated nodes with `comp_id == 0`).
    #[inline(never)]
    fn project_comp_zero_mean(&mut self, target_is_z: bool, num_comps: u32) {
        let nc = num_comps as usize;
        if nc == 0 {
            if target_is_z {
                self.solve_z.fill(0.0);
            } else {
                self.solve_r.fill(0.0);
            }
            return;
        }
        self.comp_sum[1..=nc].fill(0.0);
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            if cid > 0 {
                let val = if target_is_z { self.solve_z[u] } else { self.solve_r[u] };
                self.comp_sum[cid] += val;
            }
            u += 1;
        }
        let mut c = 1usize;
        while c <= nc {
            let cnt = self.comp_count[c] as f64;
            self.comp_sum[c] = if cnt > 0.0 { self.comp_sum[c] / cnt } else { 0.0 };
            c += 1;
        }
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            if target_is_z {
                self.solve_z[u] = if cid > 0 {
                    self.solve_z[u] - self.comp_sum[cid]
                } else {
                    0.0
                };
            } else {
                self.solve_r[u] = if cid > 0 {
                    self.solve_r[u] - self.comp_sum[cid]
                } else {
                    0.0
                };
            }
            u += 1;
        }
    }

    #[inline(never)]
    fn project_comp_zero_mean_x(&mut self, num_comps: u32) {
        let nc = num_comps as usize;
        if nc == 0 {
            self.solve_x.fill(0.0);
            return;
        }
        self.comp_sum[1..=nc].fill(0.0);
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            if cid > 0 {
                self.comp_sum[cid] += self.solve_x[u];
            }
            u += 1;
        }
        let mut c = 1usize;
        while c <= nc {
            let cnt = self.comp_count[c] as f64;
            self.comp_sum[c] = if cnt > 0.0 { self.comp_sum[c] / cnt } else { 0.0 };
            c += 1;
        }
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            self.solve_x[u] = if cid > 0 {
                self.solve_x[u] - self.comp_sum[cid]
            } else {
                0.0
            };
            u += 1;
        }
    }

    /// Runs Direct DCT or DCT-Preconditioned PCG (`<= 16` iters), verifies true relative
    /// residual `<= 1e-6`, and pins potential gauge to the valid component with max overlap
    /// in `(gx0..gx1, gy0..gy1)`.
    #[inline(never)]
    fn solve_poisson_and_pin_gauge(
        &mut self,
        frame_w: usize,
        frame_h: usize,
        is_direct: bool,
        num_comps: u32,
        gx0: usize,
        gy0: usize,
        gx1: usize,
        gy1: usize,
    ) -> bool {
        let hx = (frame_w as f64) / (BOS_GRID_W as f64);
        let hy = (frame_h as f64) / (BOS_GRID_H as f64);
        let inv_hx2 = 1.0 / (hx * hx);
        let inv_hy2 = 1.0 / (hy * hy);

        self.solve_x.fill(0.0);
        self.telem[6] = num_comps;
        self.telem[7] = 0;
        self.telem[8] = 0;
        self.telem[9] = BOS_SOLVER_NONE;
        self.telem[16] = 0;

        if num_comps == 0 {
            self.telem[3] |= BOS_REASON_NO_GAUGE;
            return false;
        }

        let mut bnorm2 = 0.0f64;
        let mut u = 0usize;
        while u < BOS_NODES {
            let bv = self.solve_b[u];
            if !bv.is_finite() {
                self.telem[3] |= BOS_REASON_NONCONV;
                return false;
            }
            bnorm2 += bv * bv;
            u += 1;
        }

        let mut converged = false;
        let mut true_res = 0.0f64;

        if bnorm2 == 0.0 {
            self.telem[9] = if is_direct { BOS_SOLVER_DIRECT } else { BOS_SOLVER_PCG };
            self.telem[8] = 0;
            converged = true;
            true_res = 0.0;
        } else if is_direct {
            self.telem[9] = BOS_SOLVER_DIRECT;
            self.telem[8] = 0;
            self.solve_z.copy_from_slice(&self.solve_b);
            self.apply_dct_poisson_solve_z(self.w_x[0] as f64, self.w_y[0] as f64, hx, hy);
            self.solve_x.copy_from_slice(&self.solve_z);
            self.project_comp_zero_mean_x(num_comps);
            self.apply_laplacian_p_to_lp(inv_hx2, inv_hy2, true);
            let mut r2 = 0.0f64;
            let mut u = 0usize;
            while u < BOS_NODES {
                let d = self.solve_b[u] - self.solve_lp[u];
                r2 += d * d;
                u += 1;
            }
            true_res = sqrt_f64(r2 / bnorm2);
            converged = true_res.is_finite() && true_res <= 1e-6;
        } else {
            self.telem[9] = BOS_SOLVER_PCG;
            self.solve_r.copy_from_slice(&self.solve_b);
            self.project_comp_zero_mean(false, num_comps);

            // z_0 = P L_0^\dagger P r_0
            self.solve_z.copy_from_slice(&self.solve_r);
            self.project_comp_zero_mean(true, num_comps);
            self.apply_dct_poisson_solve_z(1.0, 1.0, hx, hy);
            self.project_comp_zero_mean(true, num_comps);
            self.solve_p.copy_from_slice(&self.solve_z);

            let mut rz_old = 0.0f64;
            let mut u = 0usize;
            while u < BOS_NODES {
                rz_old += self.solve_r[u] * self.solve_z[u];
                u += 1;
            }
            if !rz_old.is_finite() || rz_old <= 0.0 {
                self.telem[3] |= BOS_REASON_NONCONV;
                return false;
            }

            let mut iters_done = 0u32;
            let mut breakdown = false;
            let mut it = 1u32;
            while it <= BOS_PCG_MAX_ITERS {
                iters_done = it;
                self.apply_laplacian_p_to_lp(inv_hx2, inv_hy2, false);
                let mut p_lp = 0.0f64;
                let mut p_p = 0.0f64;
                let mut u = 0usize;
                while u < BOS_NODES {
                    let pv = self.solve_p[u];
                    p_lp += pv * self.solve_lp[u];
                    p_p += pv * pv;
                    u += 1;
                }
                if !p_lp.is_finite() || p_lp <= 1e-30 * p_p || p_lp <= 0.0 {
                    breakdown = true;
                    break;
                }
                let alpha = rz_old / p_lp;
                if !alpha.is_finite() {
                    breakdown = true;
                    break;
                }
                let mut rn2 = 0.0f64;
                let mut u = 0usize;
                while u < BOS_NODES {
                    self.solve_x[u] += alpha * self.solve_p[u];
                    let nr = self.solve_r[u] - alpha * self.solve_lp[u];
                    self.solve_r[u] = nr;
                    rn2 += nr * nr;
                    u += 1;
                }
                if rn2 <= 2.5e-13 * bnorm2 {
                    break;
                }
                if it == BOS_PCG_MAX_ITERS {
                    break;
                }

                self.solve_z.copy_from_slice(&self.solve_r);
                self.project_comp_zero_mean(true, num_comps);
                self.apply_dct_poisson_solve_z(1.0, 1.0, hx, hy);
                self.project_comp_zero_mean(true, num_comps);

                let mut rz_new = 0.0f64;
                let mut u = 0usize;
                while u < BOS_NODES {
                    rz_new += self.solve_r[u] * self.solve_z[u];
                    u += 1;
                }
                if !rz_new.is_finite() || rz_new <= 0.0 {
                    breakdown = true;
                    break;
                }
                let beta = rz_new / rz_old;
                let mut u = 0usize;
                while u < BOS_NODES {
                    self.solve_p[u] = self.solve_z[u] + beta * self.solve_p[u];
                    u += 1;
                }
                rz_old = rz_new;
                it += 1;
            }
            self.telem[8] = iters_done;

            if !breakdown {
                self.project_comp_zero_mean_x(num_comps);
                self.apply_laplacian_p_to_lp(inv_hx2, inv_hy2, true);
                let mut r2 = 0.0f64;
                let mut u = 0usize;
                while u < BOS_NODES {
                    let d = self.solve_b[u] - self.solve_lp[u];
                    r2 += d * d;
                    u += 1;
                }
                true_res = sqrt_f64(r2 / bnorm2);
                converged = true_res.is_finite() && (true_res as f32).is_finite() && true_res <= 1e-6;
            }
        }

        let tr_f32 = true_res as f32;
        if true_res.is_finite() && tr_f32.is_finite() {
            self.telem[16] = tr_f32.to_bits();
        } else {
            self.telem[16] = 0.0f32.to_bits();
            converged = false;
        }

        if !converged {
            self.telem[3] |= BOS_REASON_NONCONV;
            return false;
        }

        // Select component with maximum valid overlap (>= 2 nodes) in gauge_roi
        // (equal-overlap tie-break: component's global lowest grid node index).
        if gx0 >= gx1 || gy0 >= gy1 || gx1 > BOS_GRID_W || gy1 > BOS_GRID_H {
            self.telem[3] |= BOS_REASON_NO_GAUGE;
            return false;
        }
        let nc = num_comps as usize;
        self.comp_count[1..=nc].fill(0); // temporarily reuse comp_count for gauge overlap count
        self.comp_sum[1..=nc].fill(0.0); // temporarily reuse comp_sum for gauge sum
        let mut best_cid = 0usize;
        let mut best_overlap = 0u16;
        let mut best_min_idx = usize::MAX;
        // Record each component's global lowest grid node index using bfs_queue[1..=nc] (free after BFS)
        self.bfs_queue[1..=nc].fill(u16::MAX);
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            if cid > 0 && self.bfs_queue[cid] == u16::MAX {
                self.bfs_queue[cid] = u as u16;
            }
            u += 1;
        }

        let mut j = gy0;
        while j < gy1 {
            let mut i = gx0;
            while i < gx1 {
                let u = j * BOS_GRID_W + i;
                let cid = self.comp_id[u] as usize;
                if cid > 0 {
                    self.comp_count[cid] = self.comp_count[cid].saturating_add(1);
                    self.comp_sum[cid] += self.solve_x[u];
                }
                i += 1;
            }
            j += 1;
        }

        let mut c = 1usize;
        while c <= nc {
            let ov = self.comp_count[c];
            let m_idx = self.bfs_queue[c] as usize;
            if ov >= 2 && (ov > best_overlap || (ov == best_overlap && m_idx < best_min_idx)) {
                best_overlap = ov;
                best_cid = c;
                best_min_idx = m_idx;
            }
            c += 1;
        }

        if best_cid == 0 || best_overlap < 2 {
            self.telem[3] |= BOS_REASON_NO_GAUGE;
            self.telem[7] = 0;
            return false;
        }

        let gauge_mean = self.comp_sum[best_cid] / (best_overlap as f64);
        if !gauge_mean.is_finite() {
            self.telem[3] |= BOS_REASON_NONCONV;
            self.telem[7] = 0;
            return false;
        }
        // Validate EVERY node in best_cid converts finitely to f32 before marking ANY VALID_PHI.
        let mut all_phi_finite = true;
        let mut u = 0usize;
        while u < BOS_NODES {
            if self.comp_id[u] as usize == best_cid {
                let val = (self.solve_x[u] - gauge_mean) as f32;
                if !val.is_finite() {
                    all_phi_finite = false;
                    break;
                }
            }
            u += 1;
        }
        if !all_phi_finite {
            let mut k = 0usize;
            while k < BOS_NODES {
                self.planes[BOS_PLANE_PHI * BOS_NODES + k] = 0.0;
                self.valid_mask[k] &= !BOS_VALID_PHI;
                k += 1;
            }
            self.telem[3] |= BOS_REASON_NONCONV;
            self.telem[7] = 0;
            return false;
        }
        let mut u = 0usize;
        while u < BOS_NODES {
            if self.comp_id[u] as usize == best_cid {
                let val = (self.solve_x[u] - gauge_mean) as f32;
                self.planes[BOS_PLANE_PHI * BOS_NODES + u] = val;
                self.valid_mask[u] |= BOS_VALID_PHI;
            } else {
                self.planes[BOS_PLANE_PHI * BOS_NODES + u] = 0.0;
                self.valid_mask[u] &= !BOS_VALID_PHI;
            }
            u += 1;
        }
        self.telem[7] = best_overlap as u32;
        true
    }

    /// Renders `128x64` RGBA8 overlay buffer (`alpha == 0` wherever rendered quantity is invalid).
    #[inline(never)]
    pub fn render_bos_rgba(&mut self) {
        let kind = self.ctrl.overlay_kind;
        let state = self.telem[2];
        if state != BOS_STATE_READY {
            self.rgba.fill(0);
            return;
        }

        let mut max_abs_scale = if kind == BOS_OVERLAY_PHI {
            1.0f32
        } else if kind == BOS_OVERLAY_CURL {
            0.05f32
        } else {
            1.25f32
        };

        if kind == BOS_OVERLAY_PHI || kind == BOS_OVERLAY_CURL {
            let req_bit = if kind == BOS_OVERLAY_PHI {
                BOS_VALID_PHI
            } else {
                BOS_VALID_CURL
            };
            let plane_off = if kind == BOS_OVERLAY_PHI {
                BOS_PLANE_PHI * BOS_NODES
            } else {
                BOS_PLANE_CURL * BOS_NODES
            };
            let mut mx = 0.0f32;
            let mut u = 0usize;
            while u < BOS_NODES {
                if (self.valid_mask[u] & req_bit) != 0 {
                    let v = self.planes[plane_off + u].abs();
                    if v > mx {
                        mx = v;
                    }
                }
                u += 1;
            }
            if mx > max_abs_scale {
                max_abs_scale = mx;
            }
        }
        let inv_scale = 1.0 / max_abs_scale;

        let mut u = 0usize;
        while u < BOS_NODES {
            let off = u * 4;
            let vm = self.valid_mask[u];
            if kind == BOS_OVERLAY_MAG {
                if (vm & BOS_VALID_DISP) == 0 {
                    self.rgba[off] = 0;
                    self.rgba[off + 1] = 0;
                    self.rgba[off + 2] = 0;
                    self.rgba[off + 3] = 0;
                } else {
                    let ex = self.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                    let ey = self.planes[BOS_PLANE_EPS_Y * BOS_NODES + u];
                    let mag = sqrt_f32(ex * ex + ey * ey);
                    let t = (mag * inv_scale).clamp(0.0, 1.0);
                    let (r, g, b) = if t < 0.35 {
                        let s = t * (1.0 / 0.35);
                        (12.0 + 18.0 * s, 32.0 + 148.0 * s, 56.0 + 174.0 * s)
                    } else if t < 0.70 {
                        let s = (t - 0.35) * (1.0 / 0.35);
                        (30.0 + 218.0 * s, 180.0 + 28.0 * s, 230.0 - 170.0 * s)
                    } else {
                        let s = (t - 0.70) * (1.0 / 0.30);
                        (248.0 + 7.0 * s, 208.0 + 42.0 * s, 60.0 + 175.0 * s)
                    };
                    let a = (18.0 + 222.0 * t).clamp(18.0, 240.0);
                    self.rgba[off] = r as u8;
                    self.rgba[off + 1] = g as u8;
                    self.rgba[off + 2] = b as u8;
                    self.rgba[off + 3] = a as u8;
                }
            } else if kind == BOS_OVERLAY_PHI {
                if (vm & BOS_VALID_PHI) == 0 {
                    self.rgba[off] = 0;
                    self.rgba[off + 1] = 0;
                    self.rgba[off + 2] = 0;
                    self.rgba[off + 3] = 0;
                } else {
                    let v = (self.planes[BOS_PLANE_PHI * BOS_NODES + u] * inv_scale).clamp(-1.0, 1.0);
                    let av = v.abs();
                    let (r, g, b) = if v >= 0.0 {
                        (30.0 + 225.0 * av, 40.0 + 115.0 * av, 52.0 - 16.0 * av)
                    } else {
                        (30.0 + 12.0 * av, 40.0 + 145.0 * av, 52.0 + 200.0 * av)
                    };
                    let a = (32.0 + 204.0 * av).clamp(32.0, 236.0);
                    self.rgba[off] = r as u8;
                    self.rgba[off + 1] = g as u8;
                    self.rgba[off + 2] = b as u8;
                    self.rgba[off + 3] = a as u8;
                }
            } else {
                if (vm & BOS_VALID_CURL) == 0 {
                    self.rgba[off] = 0;
                    self.rgba[off + 1] = 0;
                    self.rgba[off + 2] = 0;
                    self.rgba[off + 3] = 0;
                } else {
                    let v = (self.planes[BOS_PLANE_CURL * BOS_NODES + u] * inv_scale).clamp(-1.0, 1.0);
                    let av = v.abs();
                    let (r, g, b) = if v >= 0.0 {
                        (26.0 + 48.0 * av, 36.0 + 205.0 * av, 44.0 + 70.0 * av)
                    } else {
                        (26.0 + 204.0 * av, 36.0 + 26.0 * av, 44.0 + 186.0 * av)
                    };
                    let a = (28.0 + 206.0 * av).clamp(28.0, 234.0);
                    self.rgba[off] = r as u8;
                    self.rgba[off + 1] = g as u8;
                    self.rgba[off + 2] = b as u8;
                    self.rgba[off + 3] = a as u8;
                }
            }
            u += 1;
        }
    }

    /// Executes one full BOS frame step using already-ingested planar `luma`.
    /// Returns `true` when `state == BOS_STATE_READY` with valid displacement measurements.
    #[inline(never)]
    pub fn step(
        &mut self,
        cur_luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        continuity_epoch: u32,
        source_epoch: u32,
        source_seq: u32,
        pts_lo: u32,
        pts_hi: u32,
        frame_gap_flag: u32,
        ext_pan_or_cut: bool,
        ext_scene_cut: bool,
    ) -> bool {
        if self.ctrl.initialized == 0 {
            self.init_in_place();
        }
        self.telem[0] = 1;
        self.telem[1] = self.ctrl.requested_mode;

        if self.ctrl.requested_mode == BOS_MODE_OFF {
            self.telem[2] = BOS_STATE_OFF;
            return false;
        }

        self.telem[12] = source_seq;
        self.telem[13] = source_epoch;
        self.telem[14] = pts_lo;
        self.telem[15] = pts_hi;

        let need_px = frame_w.saturating_mul(frame_h);
        if frame_w < 16
            || frame_h < 16
            || frame_w > BOS_MAX_FRAME_W
            || frame_h > BOS_MAX_FRAME_H
            || cur_luma.len() < need_px
        {
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = BOS_REASON_DISCONT;
            return false;
        }

        if !self.has_valid_gauge() {
            self.ctrl.warmup_count = 0;
            self.ctrl.clear_consec_frames = 0;
            self.ctrl.anchor_count = 0;
            self.telem[10] = 0;
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = BOS_REASON_NO_GAUGE;
            return false;
        }

        // Check hard continuity resets (geometry change, seek, cut, engine counter wrap).
        // Timing-only gaps preserve the frozen reference while clearing consecutive clear streak.
        let geom_changed = self.ctrl.last_frame_w != 0
            && (self.ctrl.last_frame_w as usize != frame_w
                || self.ctrl.last_frame_h as usize != frame_h);
        let hard_reset = geom_changed || ext_scene_cut || (frame_gap_flag & GAP_RESET_MASK) != 0;
        self.ctrl.last_frame_w = frame_w as u16;
        self.ctrl.last_frame_h = frame_h as u16;

        if hard_reset {
            self.ctrl.last_continuity_epoch = continuity_epoch;
            self.ctrl.warmup_count = 0;
            self.ctrl.clear_consec_frames = 0;
            self.ctrl.anchor_count = 0;
            self.ctrl.anchor_base_mse = 0.0;
            self.ref_noise_var.fill(0.0);
            self.clear_streak.fill(0);
            self.invalidate_outputs();
            self.telem[10] = 0;
            self.telem[11] = self.telem[11].wrapping_add(1);
            self.telem[2] = BOS_STATE_DISTURBED;
            let mut rsn = BOS_REASON_DISCONT;
            if ext_scene_cut || (frame_gap_flag & GAP_BIT_CUT) != 0 || ext_pan_or_cut {
                rsn |= BOS_REASON_EXT_PAN_CUT;
            }
            self.telem[3] = rsn;
            return false;
        }

        if continuity_epoch != self.ctrl.last_continuity_epoch {
            self.ctrl.last_continuity_epoch = continuity_epoch;
            self.ctrl.clear_consec_frames = 0;
            self.clear_streak.fill(0);
        }

        // 1. 16-Frame Reference Warmup Phase
        if self.ctrl.warmup_count < BOS_WARMUP_FRAMES {
            self.invalidate_outputs();
            if ext_pan_or_cut {
                self.ctrl.warmup_count = 0;
                self.ctrl.anchor_count = 0;
                self.telem[10] = 0;
                self.telem[2] = BOS_STATE_DISTURBED;
                self.telem[3] = BOS_REASON_EXT_PAN_CUT;
                return false;
            }

            if self.ctrl.warmup_count == 0 {
                self.ref_luma[..need_px].copy_from_slice(&cur_luma[..need_px]);
                self.ref_noise_var.fill(0.0);
                self.clear_streak.fill(0);
                self.ctrl.anchor_base_mse = 0.0;
                let n_anch = self.select_anchors_in_gauge(frame_w, frame_h);
                if n_anch < BOS_MIN_ANCHORS {
                    self.ctrl.warmup_count = 0;
                    self.ctrl.anchor_count = 0;
                    self.telem[10] = 0;
                    self.telem[2] = BOS_STATE_DISTURBED;
                    self.telem[3] = BOS_REASON_ANCHOR_SUPPORT | BOS_REASON_LOW_TEX;
                    return false;
                }
                self.ctrl.warmup_count = 1;
                self.telem[10] = 1;
                self.telem[2] = BOS_STATE_WARMING;
                self.telem[3] = 0;
                return false;
            }

            // Warmup frames 1..15: verify stability against running mean before averaging
            let (sx, sy, amse, again, a_ok) =
                self.evaluate_frozen_anchors(cur_luma, frame_w, frame_h);
            let s_mag = sqrt_f32(sx * sx + sy * sy);
            self.telem[19] = if sx.is_finite() { sx.to_bits() } else { 0 };
            self.telem[20] = if sy.is_finite() { sy.to_bits() } else { 0 };
            self.telem[21] = if amse.is_finite() { amse.to_bits() } else { 0 };

            if !a_ok
                || !sx.is_finite()
                || !sy.is_finite()
                || !s_mag.is_finite()
                || s_mag > 0.35
                || !amse.is_finite()
                || amse > 0.045 * 0.045
                || !again.is_finite()
                || again < 0.5
                || again > 2.0
            {
                self.ctrl.warmup_count = 0;
                self.ctrl.anchor_count = 0;
                self.telem[10] = 0;
                self.telem[2] = BOS_STATE_DISTURBED;
                let mut rsn = 0u32;
                if !a_ok {
                    rsn |= BOS_REASON_ANCHOR_SUPPORT;
                }
                if !sx.is_finite() || !sy.is_finite() || !s_mag.is_finite() || s_mag > 0.35 {
                    rsn |= BOS_REASON_INT_SHIFT;
                }
                if !amse.is_finite()
                    || amse > 0.045 * 0.045
                    || !again.is_finite()
                    || again < 0.5
                    || again > 2.0
                {
                    rsn |= BOS_REASON_PHOTOM_MSE;
                }
                self.telem[3] = rsn;
                return false;
            }

            // Accumulate per-node clear noise baseline before updating ref_luma
            self.accumulate_warmup_node_noise(cur_luma, frame_w, frame_h);
            self.ctrl.anchor_base_mse += amse;

            let n_next = self.ctrl.warmup_count + 1;
            let alpha = 1.0f32 / (n_next as f32);
            let mut p = 0usize;
            while p < need_px {
                self.ref_luma[p] += (cur_luma[p] - self.ref_luma[p]) * alpha;
                p += 1;
            }
            self.ctrl.warmup_count = n_next;
            self.telem[10] = n_next;

            if n_next == BOS_WARMUP_FRAMES {
                self.ctrl.anchor_base_mse *= 1.0 / 15.0;
                let mut u = 0usize;
                while u < BOS_NODES {
                    self.ref_noise_var[u] *= 1.0 / 15.0;
                    u += 1;
                }
                let n_anch = self.select_anchors_in_gauge(frame_w, frame_h);
                if n_anch < BOS_MIN_ANCHORS {
                    self.ctrl.warmup_count = 0;
                    self.ctrl.anchor_count = 0;
                    self.telem[10] = 0;
                    self.telem[2] = BOS_STATE_DISTURBED;
                    self.telem[3] = BOS_REASON_ANCHOR_SUPPORT | BOS_REASON_LOW_TEX;
                    return false;
                }
            }
            self.telem[2] = BOS_STATE_WARMING;
            self.telem[3] = 0;
            return false;
        }

        // 2. Active Tripod Stability Guard over Frozen Anchors
        self.telem[10] = self.ctrl.warmup_count;
        let (sx, sy, amse, again, a_ok) =
            self.evaluate_frozen_anchors(cur_luma, frame_w, frame_h);
        let s_mag = sqrt_f32(sx * sx + sy * sy);
        let mse_thresh = (4.0 * self.ctrl.anchor_base_mse).max(1e-4);

        let mut reason = 0u32;
        if !a_ok || (self.ctrl.anchor_count as usize) < BOS_MIN_ANCHORS {
            reason |= BOS_REASON_ANCHOR_SUPPORT;
        }
        if ext_pan_or_cut {
            reason |= BOS_REASON_EXT_PAN_CUT;
        }
        if !sx.is_finite() || !sy.is_finite() || !s_mag.is_finite() || s_mag > 0.35 {
            reason |= BOS_REASON_INT_SHIFT;
        }
        if !amse.is_finite()
            || amse > mse_thresh
            || !again.is_finite()
            || again < 0.5
            || again > 2.0
        {
            reason |= BOS_REASON_PHOTOM_MSE;
        }

        if reason != 0 {
            self.ctrl.clear_consec_frames = 0;
            self.clear_streak.fill(0);
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = reason;
            self.telem[19] = if sx.is_finite() { sx.to_bits() } else { 0 };
            self.telem[20] = if sy.is_finite() { sy.to_bits() } else { 0 };
            self.telem[21] = if amse.is_finite() { amse.to_bits() } else { 0 };
            return false;
        }

        // 3. Dense 128x64 Subpixel Refractive Displacement Extraction
        let maint_mode = self.ctrl.requested_mode == BOS_MODE_MAINT;
        let (valid_disp, rms_disp) =
            self.extract_dense_lk(cur_luma, frame_w, frame_h, maint_mode);
        let valid_frac = (valid_disp as f32) * (1.0 / (BOS_NODES as f32));
        if !valid_frac.is_finite() || !rms_disp.is_finite() {
            self.ctrl.clear_consec_frames = 0;
            self.clear_streak.fill(0);
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = BOS_REASON_NONCONV;
            return false;
        }

        self.telem[2] = BOS_STATE_READY;
        self.telem[3] = 0;
        self.telem[4] = valid_disp;
        self.telem[19] = sx.to_bits();
        self.telem[20] = sy.to_bits();
        self.telem[21] = amse.to_bits();
        self.telem[22] = valid_frac.to_bits();
        self.telem[23] = rms_disp.to_bits();

        if valid_disp == 0 {
            self.ctrl.clear_consec_frames = 0;
            self.telem[3] = BOS_REASON_LOW_TEX;
            self.telem[5] = 0;
            self.telem[6] = 0;
            self.telem[7] = 0;
            self.telem[8] = 0;
            self.telem[9] = BOS_SOLVER_NONE;
            self.telem[16] = 0;
            self.telem[17] = 0;
            self.telem[18] = 0;
            self.rgba.fill(0);
            return false;
        }

        // Check whether declared clear gauge region itself remained clear for Mode 2
        if maint_mode {
            let gx0 = self.ctrl.gauge_x0 as usize;
            let gy0 = self.ctrl.gauge_y0 as usize;
            let gx1 = self.ctrl.gauge_x1 as usize;
            let gy1 = self.ctrl.gauge_y1 as usize;
            let mut gauge_clear = 0usize;
            let mut gauge_tot = 0usize;
            let mut j = gy0;
            while j < gy1 {
                let mut i = gx0;
                while i < gx1 {
                    let u = j * BOS_GRID_W + i;
                    gauge_tot += 1;
                    if self.clear_streak[u] > 0 {
                        gauge_clear += 1;
                    }
                    i += 1;
                }
                j += 1;
            }
            if gauge_tot > 0 && gauge_clear * 2 >= gauge_tot {
                self.ctrl.clear_consec_frames = self.ctrl.clear_consec_frames.saturating_add(1);
                if self.ctrl.clear_consec_frames >= 120 {
                    self.update_mode2_reference(cur_luma, frame_w, frame_h);
                }
            } else {
                self.ctrl.clear_consec_frames = 0;
            }
        } else {
            self.ctrl.clear_consec_frames = 0;
        }

        // 4. Curl Diagnostic + Variational Neumann Poisson Solve + Gauge Pinning
        let (curl_cnt, curl_rms, max_curl, is_direct) =
            self.compute_curl_and_assemble_neumann(frame_w, frame_h, false);
        self.telem[5] = curl_cnt;
        self.telem[17] = curl_rms.to_bits();
        self.telem[18] = max_curl.to_bits();

        let num_comps = self.label_pos_edge_components();
        let gx0 = self.ctrl.gauge_x0 as usize;
        let gy0 = self.ctrl.gauge_y0 as usize;
        let gx1 = self.ctrl.gauge_x1 as usize;
        let gy1 = self.ctrl.gauge_y1 as usize;
        let _phi_ok = self.solve_poisson_and_pin_gauge(
            frame_w, frame_h, is_direct, num_comps, gx0, gy0, gx1, gy1,
        );

        // 5. Render Overlay
        self.render_bos_rgba();
        true
    }

    #[inline(never)]
    fn accumulate_warmup_node_noise(&mut self, cur_luma: &[f32], frame_w: usize, frame_h: usize) {
        let mut p_ref = PatchRefData::empty();
        let mut cur_buf = [0.0f32; 64];
        let mut cur_mask = [0u8; 64];
        let mut j = 0usize;
        while j < BOS_GRID_H {
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let idx = j * BOS_GRID_W + i;
                let (cx, cy) = node_coord(i, j, frame_w, frame_h);
                if build_ref_patch_8x8(&self.ref_luma, frame_w, frame_h, cx, cy, &mut p_ref) {
                    if let Some(fit) = eval_cur_patch_8x8(
                        cur_luma,
                        frame_w,
                        frame_h,
                        cx,
                        cy,
                        0.0,
                        0.0,
                        &p_ref,
                        &mut cur_buf,
                        &mut cur_mask,
                    ) {
                        if fit.gain.is_finite()
                            && fit.gain >= 0.5
                            && fit.gain <= 2.0
                            && fit.cnt >= 40
                        {
                            let mut s2 = 0.0f64;
                            let mut k = 0usize;
                            while k < 64 {
                                if cur_mask[k] != 0 {
                                    let r = (cur_buf[k] as f64)
                                        - (fit.gain * (p_ref.t[k] as f64) + fit.offset);
                                    s2 += r * r;
                                }
                                k += 1;
                            }
                            self.ref_noise_var[idx] += (s2 / (fit.cnt as f64)) as f32;
                        }
                    }
                }
                i += 1;
            }
            j += 1;
        }
    }

    /// Runs the exact compiled Wasm Poisson solver on manufactured synthetic fields (`case_id = 0..=9`).
    /// Never called in the live frame tick.
    #[inline(never)]
    pub fn solve_manufactured_test(
        &mut self,
        case_id: u32,
        param_a: f32,
        param_b: f32,
        frame_w_in: usize,
        frame_h_in: usize,
    ) -> u32 {
        if case_id > 9 || !param_a.is_finite() || !param_b.is_finite() {
            return 0;
        }
        if self.ctrl.initialized == 0 {
            self.init_in_place();
        }
        let frame_w = if frame_w_in >= 64 && frame_w_in <= BOS_MAX_FRAME_W {
            frame_w_in
        } else {
            BOS_MAX_FRAME_W
        };
        let frame_h = if frame_h_in >= 64 && frame_h_in <= BOS_MAX_FRAME_H {
            frame_h_in
        } else {
            BOS_MAX_FRAME_H
        };

        self.invalidate_outputs();
        self.telem[0] = 1;
        self.telem[2] = BOS_STATE_READY;
        self.telem[3] = 0;

        let mut use_existing_weights = false;
        let mut valid_disp = 0u32;
        let mut disp2_sum = 0.0f64;

        let mut j = 0usize;
        while j < BOS_GRID_H {
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let u = j * BOS_GRID_W + i;
                let in_hole = j >= 24 && j < 40 && i >= 55 && i < 73;
                let in_split = i >= 60 && i < 65;
                let node_valid = match case_id {
                    0 | 7 | 8 => true,
                    1 | 3 | 6 | 9 => !in_hole,
                    2 => !in_split,
                    4 => false,
                    5 => i == 10 && j == 10,
                    _ => false,
                };
                if node_valid {
                    let (ex, ey, vx, vy) = match case_id {
                        7 => (0.0f32, 0.0f32, 0.01f32, 0.01f32),
                        8 => {
                            let (cx, cy) = node_coord(i, j, frame_w, frame_h);
                            (param_a * cx, param_b * cy, 0.01f32, 0.01f32)
                        }
                        3 => (
                            param_a,
                            param_b,
                            0.01f32 / (0.5 + 0.25 * (1.0 + sin_f32((i as f32) * (1.0 / 13.0)))),
                            0.01f32 / (0.5 + 0.25 * (1.0 + cos_f32((j as f32) * (1.0 / 8.0)))),
                        ),
                        _ => (param_a, param_b, 0.01f32, 0.01f32),
                    };
                    if !ex.is_finite()
                        || !ey.is_finite()
                        || !vx.is_finite()
                        || !vy.is_finite()
                        || vx <= 0.0
                        || vy <= 0.0
                    {
                        self.invalidate_outputs();
                        self.telem[2] = BOS_STATE_DISTURBED;
                        self.telem[3] = BOS_REASON_NONCONV;
                        return 0;
                    }
                    self.planes[BOS_PLANE_EPS_X * BOS_NODES + u] = ex;
                    self.planes[BOS_PLANE_EPS_Y * BOS_NODES + u] = ey;
                    self.planes[BOS_PLANE_VAR_X * BOS_NODES + u] = vx;
                    self.planes[BOS_PLANE_VAR_Y * BOS_NODES + u] = vy;
                    self.valid_mask[u] = BOS_VALID_DISP;
                    valid_disp += 1;
                    disp2_sum += (ex as f64) * (ex as f64) + (ey as f64) * (ey as f64);
                }
                i += 1;
            }
            j += 1;
        }

        if case_id == 9 {
            // Pathological checkerboard edge-weight jump (1.0 vs 1e-5) that stalls 16-iter PCG
            use_existing_weights = true;
            let mut j = 0usize;
            while j < BOS_GRID_H {
                let mut i = 0usize;
                while i < BOS_GRID_W {
                    let u = j * BOS_GRID_W + i;
                    let u_ok = (self.valid_mask[u] & BOS_VALID_DISP) != 0;
                    let w = if ((i + j) & 1) != 0 { 1e-5f32 } else { 1.0f32 };
                    if i + 1 < BOS_GRID_W && u_ok && (self.valid_mask[u + 1] & BOS_VALID_DISP) != 0
                    {
                        self.w_x[u] = w;
                    }
                    if j + 1 < BOS_GRID_H
                        && u_ok
                        && (self.valid_mask[u + BOS_GRID_W] & BOS_VALID_DISP) != 0
                    {
                        self.w_y[u] = w;
                    }
                    i += 1;
                }
                j += 1;
            }
        }

        let valid_frac = (valid_disp as f32) * (1.0 / (BOS_NODES as f32));
        let rms_disp = if valid_disp > 0 {
            sqrt_f64(disp2_sum / (valid_disp as f64)) as f32
        } else {
            0.0
        };
        if !valid_frac.is_finite() || !rms_disp.is_finite() {
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = BOS_REASON_NONCONV;
            return 0;
        }
        self.telem[4] = valid_disp;
        self.telem[22] = valid_frac.to_bits();
        self.telem[23] = rms_disp.to_bits();

        let (curl_cnt, curl_rms, max_curl, is_direct) =
            self.compute_curl_and_assemble_neumann(frame_w, frame_h, use_existing_weights);
        if !curl_rms.is_finite() || !max_curl.is_finite() {
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_DISTURBED;
            self.telem[3] = BOS_REASON_NONCONV;
            return 0;
        }
        self.telem[5] = curl_cnt;
        self.telem[17] = curl_rms.to_bits();
        self.telem[18] = max_curl.to_bits();

        let num_comps = self.label_pos_edge_components();
        let (gx0, gy0, gx1, gy1) = if case_id == 6 {
            // Force empty gauge overlap inside masked-out hole
            (55usize, 24usize, 73usize, 40usize)
        } else if self.has_valid_gauge() {
            (
                self.ctrl.gauge_x0 as usize,
                self.ctrl.gauge_y0 as usize,
                self.ctrl.gauge_x1 as usize,
                self.ctrl.gauge_y1 as usize,
            )
        } else {
            (0usize, 0usize, BOS_GRID_W, BOS_GRID_H)
        };

        let _ = self.solve_poisson_and_pin_gauge(
            frame_w, frame_h, is_direct, num_comps, gx0, gy0, gx1, gy1,
        );
        self.render_bos_rgba();
        1
    }
}
