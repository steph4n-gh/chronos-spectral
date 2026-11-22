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
    f32x4_add, f32x4_extract_lane, f32x4_floor, f32x4_gt, f32x4_lt, f32x4_max, f32x4_min,
    f32x4_mul, f32x4_splat, f32x4_sqrt, f32x4_sub, f64x2, f64x2_add, f64x2_extract_lane,
    f64x2_mul, f64x2_splat, f64x2_sqrt, f64x2_sub, i32x4_bitmask, v128, v128_and,
};

pub const BOS_GRID_W: usize = 256;
pub const BOS_GRID_H: usize = 128;
pub const BOS_NODES: usize = BOS_GRID_W * BOS_GRID_H; // 32,768
pub const BOS_MAX_FRAME_W: usize = crate::MAX_FRAME_W;
pub const BOS_MAX_FRAME_H: usize = crate::MAX_FRAME_H;
pub const BOS_MAX_PIXELS: usize = BOS_MAX_FRAME_W * BOS_MAX_FRAME_H;
pub const BOS_TELEM_LEN: usize = 32;
pub const BOS_PLANES_LEN: usize = 6 * BOS_NODES; // 196,608

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
const BOS_PCG_MAX_ITERS: u32 = 24;

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
    bitrev128: [u8; 128],
    bitrev64: [u8; 64],
    tw128_re: [f64; 128],
    tw128_im: [f64; 128],
    wn256_re: [f64; 128],
    wn256_im: [f64; 128],
    wd256_re: [f64; 128],
    wd256_im: [f64; 128],
    sin2_x: [f64; 256],
    sin2_y: [f64; 128],
    sqrt_256: f64,
    sqrt_128: f64,
    scale_fwd_256: f64,
    scale_fwd_128: f64,
    scale_dc_256: f64,
    scale_dc_128: f64,
    scale_nyq_256: f64,
    scale_nyq_128: f64,
    inv_64: f64,
    inv_128: f64,
}

impl DctTables {
    const fn new() -> Self {
        let mut bitrev128 = [0u8; 128];
        let mut i = 0usize;
        while i < 128 {
            let mut r = 0usize;
            let mut b = 0usize;
            while b < 7 {
                r = (r << 1) | ((i >> b) & 1);
                b += 1;
            }
            bitrev128[i] = r as u8;
            i += 1;
        }
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

        let mut tw128_re = [0.0f64; 128];
        let mut tw128_im = [0.0f64; 128];
        let mut wn256_re = [0.0f64; 128];
        let mut wn256_im = [0.0f64; 128];
        let mut wd256_re = [0.0f64; 128];
        let mut wd256_im = [0.0f64; 128];
        let mut k = 0usize;
        while k < 128 {
            let ang128 = -TWO_PI_64 * (k as f64) / 128.0;
            tw128_re[k] = const_cos_f64(ang128);
            tw128_im[k] = const_sin_f64(ang128);

            let ang256 = -TWO_PI_64 * (k as f64) / 256.0;
            wn256_re[k] = const_cos_f64(ang256);
            wn256_im[k] = const_sin_f64(ang256);

            let ang_d = -PI_64 * (k as f64) / 512.0;
            wd256_re[k] = const_cos_f64(ang_d);
            wd256_im[k] = const_sin_f64(ang_d);
            k += 1;
        }

        let mut sin2_x = [0.0f64; 256];
        let mut k = 0usize;
        while k < 256 {
            let s = const_sin_f64(PI_64 * (k as f64) / 512.0);
            sin2_x[k] = 4.0 * s * s;
            k += 1;
        }
        let mut sin2_y = [0.0f64; 128];
        let mut l = 0usize;
        while l < 128 {
            let s = const_sin_f64(PI_64 * (l as f64) / 256.0);
            sin2_y[l] = 4.0 * s * s;
            l += 1;
        }

        let sqrt_256 = const_sqrt_f64(256.0);
        let sqrt_128 = const_sqrt_f64(128.0);
        let scale_fwd_256 = const_sqrt_f64(2.0 / 256.0);
        let scale_fwd_128 = const_sqrt_f64(2.0 / 128.0);
        let inv_sqrt2 = const_sqrt_f64(0.5);

        Self {
            bitrev128,
            bitrev64,
            tw128_re,
            tw128_im,
            wn256_re,
            wn256_im,
            wd256_re,
            wd256_im,
            sin2_x,
            sin2_y,
            sqrt_256,
            sqrt_128,
            scale_fwd_256,
            scale_fwd_128,
            scale_dc_256: const_sqrt_f64(1.0 / 256.0),
            scale_dc_128: const_sqrt_f64(1.0 / 128.0),
            scale_nyq_256: inv_sqrt2 * scale_fwd_256,
            scale_nyq_128: inv_sqrt2 * scale_fwd_128,
            inv_64: 1.0 / 64.0,
            inv_128: 1.0 / 128.0,
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
    /// Generation of `BosState::ref_luma`: advanced (never to 0) by every change of it, and kept
    /// across `init_in_place`, so `BosRefCache` entries tagged with an older value are never reused.
    /// `0` only in a zeroed arena, where the cache is empty too.
    pub ref_gen: u32,
    pub stereo_ref_ready: u8,
    pub stereo_ref_fresh: u8,
    pub _pad_stereo: [u8; 2],
    pub stereo_ref_epoch: u32,
    pub stereo_ref_frame_idx: u32,
    pub _reserved: [u32; 10],
}

const _: () = assert!(core::mem::size_of::<BosCtrl>() == 128);

/// Static zero-allocation Background-Oriented Schlieren (`BOS`) kernel state (`256x128`).
#[repr(C, align(16))]
pub struct BosState {
    // f64 solver planes (6 * 262,144 + 8,192 = 1,581,056 B; solve_lp is reused for component sums)
    pub solve_x: [f64; BOS_NODES],
    pub solve_b: [f64; BOS_NODES],
    pub solve_r: [f64; BOS_NODES],
    pub solve_p: [f64; BOS_NODES],
    pub solve_z: [f64; BOS_NODES],
    pub solve_lp: [f64; BOS_NODES],
    pub fft_line: [f64; 1024],

    // f32 / u32 buffers
    pub ref_luma: [f32; BOS_MAX_PIXELS],
    pub ref_noise_var: [f32; BOS_NODES],
    pub planes: [f32; BOS_PLANES_LEN],
    pub w_x: [f32; BOS_NODES],
    pub w_y: [f32; BOS_NODES],
    pub telem: [u32; BOS_TELEM_LEN],

    // u16 / u8 buffers
    pub comp_id: [u16; BOS_NODES],
    pub bfs_queue: [u16; BOS_NODES],
    pub comp_count: [u16; BOS_NODES],
    pub rgba: [u8; 4 * BOS_NODES],
    pub valid_mask: [u8; BOS_NODES],
    pub clear_streak: [u8; BOS_NODES],

    // 128 B private control state
    pub ctrl: BosCtrl,
}

#[cfg(not(all(feature = "hires-720p", not(target_arch = "wasm32"))))]
const _: () = assert!(core::mem::size_of::<BosState>() <= 4_120_000);

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
    crate::simd_ops::load_f32x4(a[i..i + 4].try_into().unwrap())
}

/// Stores the four lanes of `v` to `a[i..i + 4]` (a single unaligned vector store).
#[inline(always)]
fn st4<const N: usize>(a: &mut [f32; N], i: usize, v: v128) {
    crate::simd_ops::store_f32x4((&mut a[i..i + 4]).try_into().unwrap(), v);
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

#[inline(always)]
fn next_gen(g: u32) -> u32 {
    let n = g.wrapping_add(1);
    if n == 0 {
        1
    } else {
        n
    }
}

/// What `ref_patch_restore` needs to rebuild one node's `PatchRefData` without the reductions.
/// Compact `f32` layout (`44 B`) saves `1.18 MiB` at `256x128` (`32,768` nodes).
#[repr(C, align(4))]
#[derive(Clone, Copy)]
pub struct BosRefCacheEntry {
    mu_t: f32,
    var_sum_t: f32,
    mean_gx: f32,
    mean_gy: f32,
    a_gx: f32,
    a_gy: f32,
    h_xx: f32,
    h_xy: f32,
    h_yy: f32,
    /// `BosCtrl::ref_gen` the entry was computed for (`0`: empty).
    gen: u32,
    /// The result of the reference-patch checks (`0`: the node has no usable reference patch).
    ok: u32,
}

const _: () = assert!(core::mem::size_of::<BosRefCacheEntry>() == 44);

/// Per-node reference-patch reductions (`build_ref_patch_node`), `1,441,808 B`.
#[repr(C, align(16))]
pub struct BosRefCache {
    entries: [BosRefCacheEntry; BOS_NODES],
    /// Largest `ref_gen` seen; a smaller one means the counter wrapped, so every tag is cleared.
    last_gen: u32,
    /// Frame geometry the entries were computed for (node centres depend on it).
    frame_w: u32,
    frame_h: u32,
    _pad: u32,
}

const _: () = assert!(core::mem::size_of::<BosRefCache>() == 1_441_808);

impl BosRefCache {
    #[inline(always)]
    fn sync(&mut self, ref_gen: u32, w: usize, h: usize) {
        if ref_gen < self.last_gen || self.frame_w != w as u32 || self.frame_h != h as u32 {
            let mut n = 0usize;
            while n < BOS_NODES {
                self.entries[n].gen = 0;
                n += 1;
            }
            self.frame_w = w as u32;
            self.frame_h = h as u32;
        }
        self.last_gen = ref_gen;
    }
}

pub const BOS_PYR_W: usize = 160;
pub const BOS_PYR_H: usize = 90;
pub const BOS_COARSE_W: usize = 32;
pub const BOS_COARSE_H: usize = 16;

/// Arena-tail 160x90 reference pyramid and 32x16 coarse displacement scratch for 2-level
/// coarse-to-fine BOS IC-LK (extends capture range from 1.5 px to ~8 px on strong plumes while
/// staying bit-identical on <= 0.45 px displacements).
#[repr(C, align(16))]
pub struct BosPyrScratch {
    pub ref_pyr: [f32; BOS_PYR_W * BOS_PYR_H],
    pub coarse_uv: [f32; BOS_COARSE_W * BOS_COARSE_H * 2],
    pub coarse_ok: [u8; BOS_COARSE_W * BOS_COARSE_H],
    pub tmp_t: [f32; 16],
    pub tmp_gx: [f32; 16],
    pub tmp_gy: [f32; 16],
    pub tmp_cur: [f32; 16],
    pub ref_gen: u32,
    pub frame_w: u32,
    pub frame_h: u32,
    pub valid: u32,
}

impl BosPyrScratch {
    #[inline(never)]
    pub fn reset(&mut self) {
        self.coarse_ok.fill(0);
        self.coarse_uv.fill(0.0);
        self.ref_gen = 0;
        self.frame_w = 0;
        self.frame_h = 0;
        self.valid = 0;
    }
}

#[inline(never)]
fn sync_bos_ref_pyr(ref_luma: &[f32], frame_w: usize, frame_h: usize, ref_gen: u32, pyr: &mut BosPyrScratch) {
    if pyr.valid != 0
        && pyr.ref_gen == ref_gen
        && pyr.frame_w == frame_w as u32
        && pyr.frame_h == frame_h as u32
    {
        return;
    }
    if frame_w == 640 && frame_h == 360 {
        let mut py = 0usize;
        while py < BOS_PYR_H {
            let y0 = py * 4;
            let r0 = y0 * 640;
            let r1 = (y0 + 1) * 640;
            let r2 = (y0 + 2) * 640;
            let r3 = (y0 + 3) * 640;
            let mut px = 0usize;
            while px < BOS_PYR_W {
                let x0 = px * 4;
                let s = ref_luma[r0 + x0]
                    + ref_luma[r0 + x0 + 1]
                    + ref_luma[r0 + x0 + 2]
                    + ref_luma[r0 + x0 + 3]
                    + ref_luma[r1 + x0]
                    + ref_luma[r1 + x0 + 1]
                    + ref_luma[r1 + x0 + 2]
                    + ref_luma[r1 + x0 + 3]
                    + ref_luma[r2 + x0]
                    + ref_luma[r2 + x0 + 1]
                    + ref_luma[r2 + x0 + 2]
                    + ref_luma[r2 + x0 + 3]
                    + ref_luma[r3 + x0]
                    + ref_luma[r3 + x0 + 1]
                    + ref_luma[r3 + x0 + 2]
                    + ref_luma[r3 + x0 + 3];
                pyr.ref_pyr[py * BOS_PYR_W + px] = s * (1.0 / 16.0);
                px += 1;
            }
            py += 1;
        }
    } else if frame_w == 1280 && frame_h == 720 {
        let mut py = 0usize;
        while py < BOS_PYR_H {
            let y0 = py * 8;
            let mut px = 0usize;
            while px < BOS_PYR_W {
                let x0 = px * 8;
                let mut s = 0.0f32;
                let mut dy = 0usize;
                while dy < 8 {
                    let r = (y0 + dy) * 1280 + x0;
                    s += ref_luma[r]
                        + ref_luma[r + 1]
                        + ref_luma[r + 2]
                        + ref_luma[r + 3]
                        + ref_luma[r + 4]
                        + ref_luma[r + 5]
                        + ref_luma[r + 6]
                        + ref_luma[r + 7];
                    dy += 1;
                }
                pyr.ref_pyr[py * BOS_PYR_W + px] = s * (1.0 / 64.0);
                px += 1;
            }
            py += 1;
        }
    } else if frame_w == 320 && frame_h == 180 {
        let mut py = 0usize;
        while py < BOS_PYR_H {
            let y0 = py * 2;
            let r0 = y0 * 320;
            let r1 = (y0 + 1) * 320;
            let mut px = 0usize;
            while px < BOS_PYR_W {
                let x0 = px * 2;
                let s = ref_luma[r0 + x0]
                    + ref_luma[r0 + x0 + 1]
                    + ref_luma[r1 + x0]
                    + ref_luma[r1 + x0 + 1];
                pyr.ref_pyr[py * BOS_PYR_W + px] = s * 0.25;
                px += 1;
            }
            py += 1;
        }
    } else {
        let sx = (frame_w as f32) * (1.0 / (BOS_PYR_W as f32));
        let sy = (frame_h as f32) * (1.0 / (BOS_PYR_H as f32));
        let mut py = 0usize;
        while py < BOS_PYR_H {
            let fy = ((py as f32) + 0.5) * sy - 0.5;
            let mut px = 0usize;
            while px < BOS_PYR_W {
                let fx = ((px as f32) + 0.5) * sx - 0.5;
                pyr.ref_pyr[py * BOS_PYR_W + px] =
                    sample_bilinear_checked(ref_luma, frame_w, frame_h, fx.max(0.0).min((frame_w - 2) as f32), fy.max(0.0).min((frame_h - 2) as f32))
                        .unwrap_or(0.0);
                px += 1;
            }
            py += 1;
        }
    }
    pyr.ref_gen = ref_gen;
    pyr.frame_w = frame_w as u32;
    pyr.frame_h = frame_h as u32;
    pyr.valid = 1;
}

#[inline(never)]
fn compute_bos_coarse_pyr_lk(
    cur_pyr: &[f32],
    pyr: &mut BosPyrScratch,
    frame_w: usize,
    frame_h: usize,
) {
    pyr.coarse_ok.fill(0);
    pyr.coarse_uv.fill(0.0);
    if pyr.valid == 0 || cur_pyr.len() < BOS_PYR_W * BOS_PYR_H {
        return;
    }
    let scale_x = (frame_w as f32) * (1.0 / (BOS_PYR_W as f32));
    let scale_y = (frame_h as f32) * (1.0 / (BOS_PYR_H as f32));
    let step_cx = (BOS_PYR_W as f32) * (1.0 / (BOS_COARSE_W as f32));
    let step_cy = (BOS_PYR_H as f32) * (1.0 / (BOS_COARSE_H as f32));

    let mut cj = 0usize;
    while cj < BOS_COARSE_H {
        let pcy = ((cj as f32) + 0.5) * step_cy - 0.5;
        let mut ci = 0usize;
        while ci < BOS_COARSE_W {
            let pcx = ((ci as f32) + 0.5) * step_cx - 0.5;
            let cidx = cj * BOS_COARSE_W + ci;

            // Sample 4x4 reference patch & finite-difference gradients
            let mut ok_ref = true;
            let mut sum_t = 0.0f32;
            let mut k = 0usize;
            let mut dy = 0usize;
            while dy < 4 {
                let sy = pcy + (dy as f32) - 1.5;
                let mut dx = 0usize;
                while dx < 4 {
                    let sx = pcx + (dx as f32) - 1.5;
                    if let Some(v) = sample_bilinear_unsat_checked(&pyr.ref_pyr, BOS_PYR_W, BOS_PYR_H, sx, sy) {
                        pyr.tmp_t[k] = v;
                        sum_t += v;
                    } else {
                        ok_ref = false;
                        break;
                    }
                    k += 1;
                    dx += 1;
                }
                if !ok_ref {
                    break;
                }
                dy += 1;
            }
            if ok_ref {
                let mu_t = sum_t * (1.0 / 16.0);
                let mut var_t = 0.0f32;
                let mut h_xx = 0.0f32;
                let mut h_xy = 0.0f32;
                let mut h_yy = 0.0f32;
                let mut r = 0usize;
                while r < 4 {
                    let mut c = 0usize;
                    while c < 4 {
                        let idx = r * 4 + c;
                        let dt = pyr.tmp_t[idx] - mu_t;
                        var_t += dt * dt;
                        let gx = if c == 0 {
                            pyr.tmp_t[idx + 1] - pyr.tmp_t[idx]
                        } else if c == 3 {
                            pyr.tmp_t[idx] - pyr.tmp_t[idx - 1]
                        } else {
                            0.5 * (pyr.tmp_t[idx + 1] - pyr.tmp_t[idx - 1])
                        };
                        let gy = if r == 0 {
                            pyr.tmp_t[idx + 4] - pyr.tmp_t[idx]
                        } else if r == 3 {
                            pyr.tmp_t[idx] - pyr.tmp_t[idx - 4]
                        } else {
                            0.5 * (pyr.tmp_t[idx + 4] - pyr.tmp_t[idx - 4])
                        };
                        pyr.tmp_gx[idx] = gx;
                        pyr.tmp_gy[idx] = gy;
                        h_xx += gx * gx;
                        h_xy += gx * gy;
                        h_yy += gy * gy;
                        c += 1;
                    }
                    r += 1;
                }
                let tr = h_xx + h_yy;
                let det = h_xx * h_yy - h_xy * h_xy;
                if var_t > 16.0 * 1.0e-5 && tr > 0.002 && det > 1e-6 * tr * tr + 1e-10 {
                    let inv_det = 1.0 / det;
                    let mut pex = 0.0f32;
                    let mut pey = 0.0f32;
                    let mut it_ok = true;
                    let mut it = 0usize;
                    while it < 2 {
                        let mut sum_c = 0.0f32;
                        let mut ok_c = true;
                        let mut p = 0usize;
                        let mut oy = 0usize;
                        while oy < 4 {
                            let sy = pcy + (oy as f32) - 1.5 + pey;
                            let mut ox = 0usize;
                            while ox < 4 {
                                let sx = pcx + (ox as f32) - 1.5 + pex;
                                if let Some(v) = sample_bilinear_unsat_checked(cur_pyr, BOS_PYR_W, BOS_PYR_H, sx, sy) {
                                    pyr.tmp_cur[p] = v;
                                    sum_c += v;
                                } else {
                                    ok_c = false;
                                    break;
                                }
                                p += 1;
                                ox += 1;
                            }
                            if !ok_c {
                                break;
                            }
                            oy += 1;
                        }
                        if !ok_c {
                            it_ok = false;
                            break;
                        }
                        let mu_c = sum_c * (1.0 / 16.0);
                        let mut cov_ct = 0.0f32;
                        let mut j = 0usize;
                        while j < 16 {
                            cov_ct += (pyr.tmp_cur[j] - mu_c) * (pyr.tmp_t[j] - mu_t);
                            j += 1;
                        }
                        let gain = cov_ct / var_t;
                        if !gain.is_finite() || gain < 0.5 || gain > 2.0 {
                            it_ok = false;
                            break;
                        }
                        let offset = mu_c - gain * mu_t;
                        let mut bx = 0.0f32;
                        let mut by = 0.0f32;
                        let mut j = 0usize;
                        while j < 16 {
                            let res = pyr.tmp_cur[j] - (gain * pyr.tmp_t[j] + offset);
                            bx += pyr.tmp_gx[j] * res;
                            by += pyr.tmp_gy[j] * res;
                            j += 1;
                        }
                        let sx_step = -(h_yy * bx - h_xy * by) * inv_det / gain;
                        let sy_step = -(-h_xy * bx + h_xx * by) * inv_det / gain;
                        if !sx_step.is_finite() || !sy_step.is_finite() {
                            it_ok = false;
                            break;
                        }
                        pex += sx_step;
                        pey += sy_step;
                        if pex * pex + pey * pey > 5.0625 {
                            it_ok = false;
                            break;
                        }
                        if sx_step * sx_step + sy_step * sy_step < 1e-4 {
                            break;
                        }
                        it += 1;
                    }
                    if it_ok && (pex * pex + pey * pey) <= 5.0625 {
                        // Verify final coarse residual
                        let mut sum_c = 0.0f32;
                        let mut ok_f = true;
                        let mut p = 0usize;
                        let mut oy = 0usize;
                        while oy < 4 {
                            let sy = pcy + (oy as f32) - 1.5 + pey;
                            let mut ox = 0usize;
                            while ox < 4 {
                                let sx = pcx + (ox as f32) - 1.5 + pex;
                                if let Some(v) = sample_bilinear_unsat_checked(cur_pyr, BOS_PYR_W, BOS_PYR_H, sx, sy) {
                                    pyr.tmp_cur[p] = v;
                                    sum_c += v;
                                } else {
                                    ok_f = false;
                                    break;
                                }
                                p += 1;
                                ox += 1;
                            }
                            if !ok_f {
                                break;
                            }
                            oy += 1;
                        }
                        if ok_f {
                            let mu_c = sum_c * (1.0 / 16.0);
                            let mut cov_ct = 0.0f32;
                            let mut j = 0usize;
                            while j < 16 {
                                cov_ct += (pyr.tmp_cur[j] - mu_c) * (pyr.tmp_t[j] - mu_t);
                                j += 1;
                            }
                            let gain = cov_ct / var_t;
                            if gain.is_finite() && gain >= 0.5 && gain <= 2.0 {
                                let offset = mu_c - gain * mu_t;
                                let mut mse = 0.0f32;
                                let mut j = 0usize;
                                while j < 16 {
                                    let r = pyr.tmp_cur[j] - (gain * pyr.tmp_t[j] + offset);
                                    mse += r * r;
                                    j += 1;
                                }
                                mse *= 1.0 / 16.0;
                                let ref_var = var_t * (1.0 / 16.0);
                                let mse_lim = (0.04 * 0.04f32).max(0.30 * ref_var);
                                if mse.is_finite() && mse <= mse_lim {
                                    pyr.coarse_uv[cidx * 2] = pex * scale_x;
                                    pyr.coarse_uv[cidx * 2 + 1] = pey * scale_y;
                                    pyr.coarse_ok[cidx] = 1;
                                }
                            }
                        }
                    }
                }
            }
            ci += 1;
        }
        cj += 1;
    }
}

#[inline(always)]
fn sample_bos_coarse_uv(pyr: &BosPyrScratch, i: usize, j: usize) -> (f32, f32, bool) {
    if pyr.valid == 0 {
        return (0.0, 0.0, false);
    }
    let fx = (((i as f32) + 0.5) * ((BOS_COARSE_W as f32) / (BOS_GRID_W as f32)) - 0.5)
        .max(0.0)
        .min((BOS_COARSE_W - 1) as f32);
    let fy = (((j as f32) + 0.5) * ((BOS_COARSE_H as f32) / (BOS_GRID_H as f32)) - 0.5)
        .max(0.0)
        .min((BOS_COARSE_H - 1) as f32);
    let x0 = floor_f32(fx) as usize;
    let y0 = floor_f32(fy) as usize;
    let x1 = (x0 + 1).min(BOS_COARSE_W - 1);
    let y1 = (y0 + 1).min(BOS_COARSE_H - 1);
    let tx = fx - (x0 as f32);
    let ty = fy - (y0 as f32);

    let ws = [
        (1.0 - tx) * (1.0 - ty),
        tx * (1.0 - ty),
        (1.0 - tx) * ty,
        tx * ty,
    ];
    let idxs = [
        y0 * BOS_COARSE_W + x0,
        y0 * BOS_COARSE_W + x1,
        y1 * BOS_COARSE_W + x0,
        y1 * BOS_COARSE_W + x1,
    ];
    let mut w_sum = 0.0f32;
    let mut u_sum = 0.0f32;
    let mut v_sum = 0.0f32;
    let mut k = 0usize;
    while k < 4 {
        let id = idxs[k];
        if pyr.coarse_ok[id] != 0 {
            let w = ws[k];
            w_sum += w;
            u_sum += w * pyr.coarse_uv[id * 2];
            v_sum += w * pyr.coarse_uv[id * 2 + 1];
        }
        k += 1;
    }
    if w_sum > 0.25 {
        let inv = 1.0 / w_sum;
        (u_sum * inv, v_sum * inv, true)
    } else {
        (0.0, 0.0, false)
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

/// First phase of the reference patch at `(cx, cy)`: samples the `8x8` patch and its central
/// differences into `out` (`m_cnt`, `valid_flag`, `t`, `gx_raw`, `gy_raw`). With `WITH_SUM` it
/// also returns `sum_t`, the sum of the valid samples in pixel order (a sequential `f64` chain;
/// skipped when the cached reductions are reused).
#[inline(always)]
fn ref_patch_samples<const WITH_SUM: bool>(
    ref_luma: &[f32],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    out: &mut PatchRefData,
) -> f64 {
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
                if WITH_SUM {
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
                        if WITH_SUM {
                            sum_t += tc as f64;
                        }
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
    sum_t
}

/// Second phase: the order-dependent `f64` reductions over the samples (`mu_t`, the variance,
/// gradient means and `T`-projections, then `H` of the `[T, 1]`-orthogonal gradients) and the
/// contrast and conditioning checks. Records every value `ref_patch_restore` needs in `e`.
#[inline(always)]
fn ref_patch_reduce(out: &mut PatchRefData, sum_t: f64, e: &mut BosRefCacheEntry) -> bool {
    if out.m_cnt < 40 {
        return false;
    }

    let inv_m = 1.0 / (out.m_cnt as f64);
    let mu_t = sum_t * inv_m;
    out.mu_t = mu_t;
    e.mu_t = mu_t as f32;

    let mut var_sum_t0 = f64x2_splat(0.0);
    let mut var_sum_t1 = f64x2_splat(0.0);
    let mut mean_gx0 = f64x2_splat(0.0);
    let mut mean_gx1 = f64x2_splat(0.0);
    let mut mean_gy0 = f64x2_splat(0.0);
    let mut mean_gy1 = f64x2_splat(0.0);
    let mut dot_tgx0 = f64x2_splat(0.0);
    let mut dot_tgx1 = f64x2_splat(0.0);
    let mut dot_tgy0 = f64x2_splat(0.0);
    let mut dot_tgy1 = f64x2_splat(0.0);
    let v_mu_t = f64x2_splat(mu_t);
    let mut k = 0usize;
    while k < 64 {
        let m0 = f64x2(
            if out.valid_flag[k] != 0 { 1.0 } else { 0.0 },
            if out.valid_flag[k + 1] != 0 { 1.0 } else { 0.0 },
        );
        let dt0 = f64x2_mul(
            f64x2_sub(f64x2(out.t[k] as f64, out.t[k + 1] as f64), v_mu_t),
            m0,
        );
        let gx0 = f64x2_mul(f64x2(out.gx_raw[k] as f64, out.gx_raw[k + 1] as f64), m0);
        let gy0 = f64x2_mul(f64x2(out.gy_raw[k] as f64, out.gy_raw[k + 1] as f64), m0);
        var_sum_t0 = f64x2_add(var_sum_t0, f64x2_mul(dt0, dt0));
        mean_gx0 = f64x2_add(mean_gx0, gx0);
        mean_gy0 = f64x2_add(mean_gy0, gy0);
        dot_tgx0 = f64x2_add(dot_tgx0, f64x2_mul(dt0, gx0));
        dot_tgy0 = f64x2_add(dot_tgy0, f64x2_mul(dt0, gy0));

        let m1 = f64x2(
            if out.valid_flag[k + 2] != 0 { 1.0 } else { 0.0 },
            if out.valid_flag[k + 3] != 0 { 1.0 } else { 0.0 },
        );
        let dt1 = f64x2_mul(
            f64x2_sub(f64x2(out.t[k + 2] as f64, out.t[k + 3] as f64), v_mu_t),
            m1,
        );
        let gx1 = f64x2_mul(f64x2(out.gx_raw[k + 2] as f64, out.gx_raw[k + 3] as f64), m1);
        let gy1 = f64x2_mul(f64x2(out.gy_raw[k + 2] as f64, out.gy_raw[k + 3] as f64), m1);
        var_sum_t1 = f64x2_add(var_sum_t1, f64x2_mul(dt1, dt1));
        mean_gx1 = f64x2_add(mean_gx1, gx1);
        mean_gy1 = f64x2_add(mean_gy1, gy1);
        dot_tgx1 = f64x2_add(dot_tgx1, f64x2_mul(dt1, gx1));
        dot_tgy1 = f64x2_add(dot_tgy1, f64x2_mul(dt1, gy1));
        k += 4;
    }
    let hsum2 = |v0: v128, v1: v128| -> f64 {
        let s = f64x2_add(v0, v1);
        f64x2_extract_lane::<0>(s) + f64x2_extract_lane::<1>(s)
    };
    let var_sum_t = hsum2(var_sum_t0, var_sum_t1);
    let mut mean_gx = hsum2(mean_gx0, mean_gx1);
    let mut mean_gy = hsum2(mean_gy0, mean_gy1);
    let dot_tgx = hsum2(dot_tgx0, dot_tgx1);
    let dot_tgy = hsum2(dot_tgy0, dot_tgy1);

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
    e.var_sum_t = var_sum_t as f32;
    e.mean_gx = mean_gx as f32;
    e.mean_gy = mean_gy as f32;
    e.a_gx = a_gx as f32;
    e.a_gy = a_gy as f32;

    let v_mean_gx = f64x2_splat(mean_gx);
    let v_mean_gy = f64x2_splat(mean_gy);
    let v_a_gx = f64x2_splat(a_gx);
    let v_a_gy = f64x2_splat(a_gy);
    let mut h_xx0 = f64x2_splat(0.0);
    let mut h_xx1 = f64x2_splat(0.0);
    let mut h_xy0 = f64x2_splat(0.0);
    let mut h_xy1 = f64x2_splat(0.0);
    let mut h_yy0 = f64x2_splat(0.0);
    let mut h_yy1 = f64x2_splat(0.0);
    let mut k = 0usize;
    while k < 64 {
        let m0 = f64x2(
            if out.valid_flag[k] != 0 { 1.0 } else { 0.0 },
            if out.valid_flag[k + 1] != 0 { 1.0 } else { 0.0 },
        );
        let dt0 = f64x2_sub(f64x2(out.t[k] as f64, out.t[k + 1] as f64), v_mu_t);
        let gxp0 = f64x2_mul(
            f64x2_sub(
                f64x2_sub(f64x2(out.gx_raw[k] as f64, out.gx_raw[k + 1] as f64), v_mean_gx),
                f64x2_mul(v_a_gx, dt0),
            ),
            m0,
        );
        let gyp0 = f64x2_mul(
            f64x2_sub(
                f64x2_sub(f64x2(out.gy_raw[k] as f64, out.gy_raw[k + 1] as f64), v_mean_gy),
                f64x2_mul(v_a_gy, dt0),
            ),
            m0,
        );
        out.gx_perp[k] = f64x2_extract_lane::<0>(gxp0) as f32;
        out.gx_perp[k + 1] = f64x2_extract_lane::<1>(gxp0) as f32;
        out.gy_perp[k] = f64x2_extract_lane::<0>(gyp0) as f32;
        out.gy_perp[k + 1] = f64x2_extract_lane::<1>(gyp0) as f32;
        h_xx0 = f64x2_add(h_xx0, f64x2_mul(gxp0, gxp0));
        h_xy0 = f64x2_add(h_xy0, f64x2_mul(gxp0, gyp0));
        h_yy0 = f64x2_add(h_yy0, f64x2_mul(gyp0, gyp0));

        let m1 = f64x2(
            if out.valid_flag[k + 2] != 0 { 1.0 } else { 0.0 },
            if out.valid_flag[k + 3] != 0 { 1.0 } else { 0.0 },
        );
        let dt1 = f64x2_sub(f64x2(out.t[k + 2] as f64, out.t[k + 3] as f64), v_mu_t);
        let gxp1 = f64x2_mul(
            f64x2_sub(
                f64x2_sub(f64x2(out.gx_raw[k + 2] as f64, out.gx_raw[k + 3] as f64), v_mean_gx),
                f64x2_mul(v_a_gx, dt1),
            ),
            m1,
        );
        let gyp1 = f64x2_mul(
            f64x2_sub(
                f64x2_sub(f64x2(out.gy_raw[k + 2] as f64, out.gy_raw[k + 3] as f64), v_mean_gy),
                f64x2_mul(v_a_gy, dt1),
            ),
            m1,
        );
        out.gx_perp[k + 2] = f64x2_extract_lane::<0>(gxp1) as f32;
        out.gx_perp[k + 3] = f64x2_extract_lane::<1>(gxp1) as f32;
        out.gy_perp[k + 2] = f64x2_extract_lane::<0>(gyp1) as f32;
        out.gy_perp[k + 3] = f64x2_extract_lane::<1>(gyp1) as f32;
        h_xx1 = f64x2_add(h_xx1, f64x2_mul(gxp1, gxp1));
        h_xy1 = f64x2_add(h_xy1, f64x2_mul(gxp1, gyp1));
        h_yy1 = f64x2_add(h_yy1, f64x2_mul(gyp1, gyp1));
        k += 4;
    }
    let h_xx = hsum2(h_xx0, h_xx1);
    let h_xy = hsum2(h_xy0, h_xy1);
    let h_yy = hsum2(h_yy0, h_yy1);

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
    e.h_xx = h_xx as f32;
    e.h_xy = h_xy as f32;
    e.h_yy = h_yy as f32;
    true
}

/// Second phase from a cache entry recorded by `ref_patch_reduce` for the same node and the same
/// `ref_luma`: SIMD128 `f32x4` per-pixel orthogonalization without reductions.
#[inline(always)]
fn ref_patch_restore(out: &mut PatchRefData, e: &BosRefCacheEntry) -> bool {
    let inv_m = 1.0 / (out.m_cnt as f64);
    let mu_t = e.mu_t as f64;
    out.mu_t = mu_t;
    out.var_sum_t = e.var_sum_t as f64;
    out.ref_var = (e.var_sum_t as f64) * inv_m;
    let v_mu = f32x4_splat(e.mu_t);
    let v_mgx = f32x4_splat(e.mean_gx);
    let v_mgy = f32x4_splat(e.mean_gy);
    let v_agx = f32x4_splat(e.a_gx);
    let v_agy = f32x4_splat(e.a_gy);
    if out.m_cnt == 64 {
        let mut k = 0usize;
        while k < 64 {
            let dt = f32x4_sub(ld4(&out.t, k), v_mu);
            let gxp = f32x4_sub(f32x4_sub(ld4(&out.gx_raw, k), v_mgx), f32x4_mul(v_agx, dt));
            let gyp = f32x4_sub(f32x4_sub(ld4(&out.gy_raw, k), v_mgy), f32x4_mul(v_agy, dt));
            st4(&mut out.gx_perp, k, gxp);
            st4(&mut out.gy_perp, k, gyp);
            k += 4;
        }
    } else {
        let (mean_gx, mean_gy, a_gx, a_gy) = (e.mean_gx, e.mean_gy, e.a_gx, e.a_gy);
        let mut k = 0usize;
        while k < 64 {
            if out.valid_flag[k] != 0 {
                let dt = out.t[k] - e.mu_t;
                out.gx_perp[k] = out.gx_raw[k] - mean_gx - a_gx * dt;
                out.gy_perp[k] = out.gy_raw[k] - mean_gy - a_gy * dt;
            } else {
                out.gx_perp[k] = 0.0;
                out.gy_perp[k] = 0.0;
            }
            k += 1;
        }
    }
    let (h_xx, h_xy, h_yy) = (e.h_xx as f64, e.h_xy as f64, e.h_yy as f64);
    out.h_xx = h_xx;
    out.h_xy = h_xy;
    out.h_yy = h_yy;
    out.det = h_xx * h_yy - h_xy * h_xy;
    out.tr = h_xx + h_yy;
    true
}

/// The reference patch of BOS node `node` (`PatchRefData` as above; `false` when it fails the
/// contrast or conditioning checks, in which case callers read nothing from `out`).
///
/// The patch depends only on `ref_luma`, the frame geometry and the node, so the results of the
/// reductions are kept per node in `cache`, tagged with `ref_gen` (`BosCtrl::ref_gen`, advanced on
/// every change of `ref_luma`). A tagged entry skips the three sequential `f64` reductions, which
/// dominate the cost; everything else is recomputed exactly. `ref_gen == 0` disables reuse.
#[inline(never)]
fn build_ref_patch_node(
    cache: &mut BosRefCache,
    ref_gen: u32,
    ref_luma: &[f32],
    w: usize,
    h: usize,
    node: usize,
    out: &mut PatchRefData,
) -> bool {
    cache.sync(ref_gen, w, h);
    let (cx, cy) = node_coord(node % BOS_GRID_W, node / BOS_GRID_W, w, h);
    let e = &mut cache.entries[node];
    if ref_gen != 0 && e.gen == ref_gen {
        if e.ok == 0 {
            return false;
        }
        ref_patch_samples::<false>(ref_luma, w, h, cx, cy, out);
        ref_patch_restore(out, e)
    } else {
        let sum_t = ref_patch_samples::<true>(ref_luma, w, h, cx, cy, out);
        let ok = ref_patch_reduce(out, sum_t, e);
        e.gen = ref_gen;
        e.ok = ok as u32;
        ok
    }
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
            let v_mu_t = f64x2_splat(mu_t);
            let mut sum_ti0 = f64x2_splat(0.0);
            let mut sum_ti1 = f64x2_splat(0.0);
            let mut bx0 = f64x2_splat(0.0);
            let mut bx1 = f64x2_splat(0.0);
            let mut by0 = f64x2_splat(0.0);
            let mut by1 = f64x2_splat(0.0);
            let mut k = 0usize;
            while k < 64 {
                let iv0 = f64x2(out_cur[k] as f64, out_cur[k + 1] as f64);
                let dt0 = f64x2_sub(f64x2(ref_p.t[k] as f64, ref_p.t[k + 1] as f64), v_mu_t);
                sum_ti0 = f64x2_add(sum_ti0, f64x2_mul(dt0, iv0));
                bx0 = f64x2_add(bx0, f64x2_mul(f64x2(ref_p.gx_perp[k] as f64, ref_p.gx_perp[k + 1] as f64), iv0));
                by0 = f64x2_add(by0, f64x2_mul(f64x2(ref_p.gy_perp[k] as f64, ref_p.gy_perp[k + 1] as f64), iv0));

                let iv1 = f64x2(out_cur[k + 2] as f64, out_cur[k + 3] as f64);
                let dt1 = f64x2_sub(f64x2(ref_p.t[k + 2] as f64, ref_p.t[k + 3] as f64), v_mu_t);
                sum_ti1 = f64x2_add(sum_ti1, f64x2_mul(dt1, iv1));
                bx1 = f64x2_add(bx1, f64x2_mul(f64x2(ref_p.gx_perp[k + 2] as f64, ref_p.gx_perp[k + 3] as f64), iv1));
                by1 = f64x2_add(by1, f64x2_mul(f64x2(ref_p.gy_perp[k + 2] as f64, ref_p.gy_perp[k + 3] as f64), iv1));
                k += 4;
            }
            let sum_ti_v = f64x2_add(sum_ti0, sum_ti1);
            let bx_v = f64x2_add(bx0, bx1);
            let by_v = f64x2_add(by0, by1);
            let sum_ti = f64x2_extract_lane::<0>(sum_ti_v) + f64x2_extract_lane::<1>(sum_ti_v);
            let bx = f64x2_extract_lane::<0>(bx_v) + f64x2_extract_lane::<1>(bx_v);
            let by = f64x2_extract_lane::<0>(by_v) + f64x2_extract_lane::<1>(by_v);
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

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn refine_best_match_subpx(
    cur_luma: &[f32],
    frame_w: usize,
    icx: isize,
    icy: usize,
    t16: &[f32; 16],
    var_t16: f32,
    t1: &[f32; 16],
    v11: f32,
    v01: f32,
    has_t1: bool,
    dx_lo: isize,
    dx_hi: isize,
    mut best_dx: isize,
    mut best_sc: f32,
) -> (isize, f32, f32) {
    let mut nb = [[0.0f32; 16]; 3];
    let mut nb_var = [0.0f32; 3];
    let mut nb_d0 = [0.0f32; 3];
    let mut nb_ok = [false; 3];
    let mut best_sc_ref = best_sc;
    let center = best_dx;

    let mut k = 0usize;
    while k < 3 {
        let dx_k = center + (k as isize) - 1;
        if dx_k >= dx_lo && dx_k <= dx_hi {
            let tx = icx + dx_k;
            if tx >= 4 && tx + 4 < (frame_w as isize) {
                let mut sum_c = 0.0f32;
                let sx0 = (tx as usize) - 3;
                for (r4, dst_row) in nb[k].chunks_exact_mut(4).enumerate() {
                    let sy = icy - 3 + r4 * 2;
                    let ro = sy * frame_w + sx0;
                    if let Some(sl) = cur_luma.get(ro..ro + 7) {
                        let [v0, _, v1, _, v2, _, v3] = match <[f32; 7]>::try_from(sl) {
                            Ok(arr) => arr,
                            Err(_) => [0.0; 7],
                        };
                        dst_row[0] = v0;
                        dst_row[1] = v1;
                        dst_row[2] = v2;
                        dst_row[3] = v3;
                        sum_c += v0 + v1 + v2 + v3;
                    }
                }
                let mu_c = sum_c * (1.0 / 16.0);
                let mut d0 = 0.0f32;
                let mut d1 = 0.0f32;
                let mut var_c = 0.0f32;
                for ((slot, &t0_v), &t1_v) in
                    nb[k].iter_mut().zip(t16.iter()).zip(t1.iter())
                {
                    let d = *slot - mu_c;
                    *slot = d;
                    d0 += t0_v * d;
                    if has_t1 {
                        d1 += t1_v * d;
                    }
                    var_c += d * d;
                }
                if var_c > 5.0e-5 {
                    nb_var[k] = var_c;
                    nb_d0[k] = d0;
                    nb_ok[k] = true;
                    if d0 > 0.0 {
                        let sc = (d0 * d0) / (var_t16 * var_c);
                        if sc > best_sc {
                            best_sc = sc;
                            best_dx = dx_k;
                        }
                        if sc > best_sc_ref {
                            best_sc_ref = sc;
                        }
                    }
                    if has_t1 && (d0 > 0.0 || d1 > 0.0) {
                        let a = v11 * d0 - v01 * d1;
                        let b = var_t16 * d1 - v01 * d0;
                        if a > 0.0 && b > 0.0 {
                            let w = (b / (a + b)).clamp(0.0, 1.0);
                            let omw = 1.0 - w;
                            let num = omw * d0 + w * d1;
                            let den_t =
                                omw * omw * var_t16 + 2.0 * omw * w * v01 + w * w * v11;
                            if num > 0.0 && den_t > 5.0e-5 {
                                let sc_sub = (num * num) / (den_t * var_c);
                                if sc_sub > best_sc_ref {
                                    best_sc_ref = sc_sub;
                                }
                            }
                        }
                    }
                }
            }
        }
        k += 1;
    }

    let mut p = 0usize;
    while p < 2 {
        let a_i = p;
        let b_i = p + 1;
        if nb_ok[a_i] && nb_ok[b_i] && (nb_d0[a_i] > 0.0 || nb_d0[b_i] > 0.0) {
            let mut v_ab = 0.0f32;
            let mut kk = 0usize;
            while kk < 16 {
                v_ab += nb[a_i][kk] * nb[b_i][kk];
                kk += 1;
            }
            let va = nb_var[a_i];
            let vb = nb_var[b_i];
            let det = va * vb - v_ab * v_ab;
            if det > 1.0e-6 * va * vb {
                let da = nb_d0[a_i];
                let db = nb_d0[b_i];
                let ca = vb * da - v_ab * db;
                let cb = va * db - v_ab * da;
                if ca > 0.0 && cb > 0.0 {
                    let w = (cb / (ca + cb)).clamp(0.0, 1.0);
                    let omw = 1.0 - w;
                    let num = omw * da + w * db;
                    let den = omw * omw * va + 2.0 * omw * w * v_ab + w * w * vb;
                    if num > 0.0 && den > 5.0e-5 {
                        let sc_rev = (num * num) / (var_t16 * den);
                        if sc_rev > best_sc_ref {
                            best_sc_ref = sc_rev;
                        }
                    }
                }
            }
        }
        p += 1;
    }

    (best_dx, best_sc, best_sc_ref)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn screen_patch_autocorr_1d(
    img: &[f32],
    frame_w: usize,
    icx: isize,
    icy: usize,
    t16: &[f32; 16],
    mu_t16: f32,
    var_t16: f32,
    t1: &[f32; 16],
    v11: f32,
    v01: f32,
    has_t1: bool,
    max_auto: isize,
    thresh: f32,
    mut best_rival: f32,
) -> f32 {
    let mut c_prev = [[0.0f32; 16]; 2];
    let mut mu_prev = [0.0f32; 2];
    let mut var_prev = [0.0f32; 2];
    let mut d0_prev = [0.0f32; 2];
    let mut sc_prev = [0.0f32; 2];
    let mut cs_prev = [[0.0f32; 4]; 2];
    let mut cq_prev = [[0.0f32; 4]; 2];
    let mut cd_prev = [[0.0f32; 4]; 2];
    let mut has_prev = [false; 2];

    let mut s_auto = 4isize;
    'auto_scan: while s_auto <= max_auto {
        let mut dir = 0usize;
        while dir < 2 {
            let ax = if dir == 0 {
                icx + s_auto
            } else {
                icx - s_auto
            };
            if ax >= 4 && ax + 4 < (frame_w as isize) {
                let mut c16 = [0.0f32; 16];
                let mut cs0 = 0.0f32;
                let mut cs1 = 0.0f32;
                let mut cs2 = 0.0f32;
                let mut cs3 = 0.0f32;
                let mut cq0 = 0.0f32;
                let mut cq1 = 0.0f32;
                let mut cq2 = 0.0f32;
                let mut cq3 = 0.0f32;
                let mut cd0 = 0.0f32;
                let mut cd1 = 0.0f32;
                let mut cd2 = 0.0f32;
                let mut cd3 = 0.0f32;
                let sx0 = (ax as usize) - 3;
                for (r4, (c_row, t_row)) in
                    c16.chunks_exact_mut(4).zip(t16.chunks_exact(4)).enumerate()
                {
                    let sy = icy - 3 + r4 * 2;
                    let ro = sy * frame_w + sx0;
                    if let Some(sl) = img.get(ro..ro + 7) {
                        let [p0, _, p1, _, p2, _, p3] = match <[f32; 7]>::try_from(sl) {
                            Ok(arr) => arr,
                            Err(_) => [0.0; 7],
                        };
                        let v0 = p0 - mu_t16;
                        let v1 = p1 - mu_t16;
                        let v2 = p2 - mu_t16;
                        let v3 = p3 - mu_t16;
                        c_row[0] = v0;
                        c_row[1] = v1;
                        c_row[2] = v2;
                        c_row[3] = v3;
                        cs0 += v0;
                        cs1 += v1;
                        cs2 += v2;
                        cs3 += v3;
                        cq0 += v0 * v0;
                        cq1 += v1 * v1;
                        cq2 += v2 * v2;
                        cq3 += v3 * v3;
                        cd0 += t_row[0] * v0;
                        cd1 += t_row[1] * v1;
                        cd2 += t_row[2] * v2;
                        cd3 += t_row[3] * v3;
                    }
                }
                let cs = [cs0, cs1, cs2, cs3];
                let cq = [cq0, cq1, cq2, cq3];
                let cd = [cd0, cd1, cd2, cd3];
                let sum_c = cs0 + cs1 + cs2 + cs3;
                let sum_q = cq0 + cq1 + cq2 + cq3;
                let d0 = cd0 + cd1 + cd2 + cd3;
                let mu_c = sum_c * (1.0 / 16.0);
                let var_c16 = sum_q - sum_c * mu_c;
                if var_c16 > 5.0e-5
                    && var_c16 >= 0.25 * var_t16
                    && var_c16 <= 4.0 * var_t16
                {
                    let sc_int = if d0 > 0.0 {
                        (d0 * d0) / (var_t16 * var_c16)
                    } else {
                        0.0
                    };
                    if sc_int > best_rival {
                        best_rival = sc_int;
                        if best_rival > thresh {
                            break 'auto_scan;
                        }
                    }
                    // Forward convex sub-pixel projection: c_cur onto span{t(icx), t(icx+1)}.
                    // Skip dir==0 at s_auto==4 so the tested lag stays >= 4.0 px (avoiding the
                    // main autocorrelation peak shoulder at 3.0..4.0 px on smooth natural edges).
                    if has_t1 && !(dir == 0 && s_auto == 4) && sc_int >= 0.04 * thresh {
                        let mut d1 = 0.0f32;
                        let mut kk = 0usize;
                        while kk < 16 {
                            d1 += t1[kk] * c16[kk];
                            kk += 1;
                        }
                        if d0 > 0.0 || d1 > 0.0 {
                            let a = v11 * d0 - v01 * d1;
                            let b = var_t16 * d1 - v01 * d0;
                            if a > 0.0 && b > 0.0 {
                                let w = (b / (a + b)).clamp(0.0, 1.0);
                                let omw = 1.0 - w;
                                let num = omw * d0 + w * d1;
                                let den_t =
                                    omw * omw * var_t16 + 2.0 * omw * w * v01 + w * w * v11;
                                if num > 0.0 && den_t > 5.0e-5 {
                                    let sc_sub = (num * num) / (den_t * var_c16);
                                    if sc_sub > best_rival {
                                        best_rival = sc_sub;
                                        if best_rival > thresh {
                                            break 'auto_scan;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if has_prev[dir] {
                        // Reverse convex sub-pixel projection: t(icx) onto span{c(ax_prev), c(ax)}
                        if sc_prev[dir] + sc_int >= 0.25 * thresh
                            && (d0_prev[dir] > 0.0 || d0 > 0.0)
                        {
                            let mut dot_pc = 0.0f32;
                            let mut kp = 0usize;
                            while kp < 16 {
                                dot_pc += c_prev[dir][kp] * c16[kp];
                                kp += 1;
                            }
                            let v_pc = dot_pc - 16.0 * mu_prev[dir] * mu_c;
                            let vp = var_prev[dir];
                            let det_c = vp * var_c16 - v_pc * v_pc;
                            if det_c > 1.0e-6 * vp * var_c16 {
                                let dp = d0_prev[dir];
                                let ac = var_c16 * dp - v_pc * d0;
                                let bc = vp * d0 - v_pc * dp;
                                if ac > 0.0 && bc > 0.0 {
                                    let wc = (bc / (ac + bc)).clamp(0.0, 1.0);
                                    let omwc = 1.0 - wc;
                                    let numc = omwc * dp + wc * d0;
                                    let denc = omwc * omwc * vp
                                        + 2.0 * omwc * wc * v_pc
                                        + wc * wc * var_c16;
                                    if numc > 0.0 && denc > 5.0e-5 {
                                        let sc_rev = (numc * numc) / (var_t16 * denc);
                                        if sc_rev > best_rival {
                                            best_rival = sc_rev;
                                            if best_rival > thresh {
                                                break 'auto_scan;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Monotone 4-column seam-split check (catches fractional-period nearest-neighbor
                        // wrap seams where left m columns match lag s-1 and right 4-m match lag s, or vice versa)
                        let cp_d = cd_prev[dir];
                        let d_ub = cp_d[0].max(cd[0])
                            + cp_d[1].max(cd[1])
                            + cp_d[2].max(cd[2])
                            + cp_d[3].max(cd[3]);
                        if d_ub > 0.0
                            && d_ub * d_ub > 0.15 * thresh * var_t16 * var_prev[dir].min(var_c16)
                        {
                            let cp_s = cs_prev[dir];
                            let cp_q = cq_prev[dir];
                            let sum_ps = cp_s[0] + cp_s[1] + cp_s[2] + cp_s[3];
                            let sum_pq = cp_q[0] + cp_q[1] + cp_q[2] + cp_q[3];
                            let sum_pd = cp_d[0] + cp_d[1] + cp_d[2] + cp_d[3];
                            let mut ps = 0.0f32;
                            let mut pq = 0.0f32;
                            let mut pd = 0.0f32;
                            let mut cs_pre = 0.0f32;
                            let mut cq_pre = 0.0f32;
                            let mut cd_pre = 0.0f32;
                            let mut m = 0usize;
                            while m < 3 {
                                ps += cp_s[m];
                                pq += cp_q[m];
                                pd += cp_d[m];
                                cs_pre += cs[m];
                                cq_pre += cq[m];
                                cd_pre += cd[m];
                                // Split 1: [prev(0..=m), cur(m+1..4)]
                                let d_a = pd + (d0 - cd_pre);
                                if d_a > 0.0 {
                                    let s_a = ps + (sum_c - cs_pre);
                                    let q_a = pq + (sum_q - cq_pre);
                                    let v_a = q_a - s_a * s_a * (1.0 / 16.0);
                                    if v_a > 5.0e-5 {
                                        let sc_a = (d_a * d_a) / (var_t16 * v_a);
                                        if sc_a > best_rival {
                                            best_rival = sc_a;
                                            if best_rival > thresh {
                                                break 'auto_scan;
                                            }
                                        }
                                    }
                                }
                                // Split 2: [cur(0..=m), prev(m+1..4)]
                                let d_b = cd_pre + (sum_pd - pd);
                                if d_b > 0.0 {
                                    let s_b = cs_pre + (sum_ps - ps);
                                    let q_b = cq_pre + (sum_pq - pq);
                                    let v_b = q_b - s_b * s_b * (1.0 / 16.0);
                                    if v_b > 5.0e-5 {
                                        let sc_b = (d_b * d_b) / (var_t16 * v_b);
                                        if sc_b > best_rival {
                                            best_rival = sc_b;
                                            if best_rival > thresh {
                                                break 'auto_scan;
                                            }
                                        }
                                    }
                                }
                                m += 1;
                            }
                        }
                    }
                    c_prev[dir] = c16;
                    mu_prev[dir] = mu_c;
                    var_prev[dir] = var_c16;
                    d0_prev[dir] = d0;
                    sc_prev[dir] = sc_int;
                    cs_prev[dir] = cs;
                    cq_prev[dir] = cq;
                    cd_prev[dir] = cd;
                    has_prev[dir] = true;
                } else {
                    has_prev[dir] = false;
                }
            } else {
                has_prev[dir] = false;
            }
            dir += 1;
        }
        s_auto += 1;
    }
    best_rival
}

impl BosState {
    /// Advances `ctrl.ref_gen` after `ref_luma` changed (skipping `0`, which means "no cache").
    #[inline(always)]
    fn bump_ref_gen(&mut self) {
        self.ctrl.ref_gen = next_gen(self.ctrl.ref_gen);
    }

    pub fn init_in_place(&mut self) {
        let next_ref_gen = next_gen(self.ctrl.ref_gen);
        self.solve_x.fill(0.0);
        self.solve_b.fill(0.0);
        self.solve_r.fill(0.0);
        self.solve_p.fill(0.0);
        self.solve_z.fill(0.0);
        self.solve_lp.fill(0.0);
        self.solve_lp.fill(0.0);
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
            ref_gen: next_ref_gen,
            stereo_ref_ready: 0,
            stereo_ref_fresh: 0,
            _pad_stereo: [0; 2],
            stereo_ref_epoch: 0,
            stereo_ref_frame_idx: u32::MAX,
            _reserved: [0; 10],
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
        self.invalidate_stereo_ref();
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
    fn select_anchors_in_gauge(&mut self, cache: &mut BosRefCache, frame_w: usize, frame_h: usize) -> usize {
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
                            if build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, j * BOS_GRID_W + i, &mut p_ref)
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
                            && build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, node as usize, &mut p_ref)
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
                let ni = (node as usize) & 255;
                let nj = (node as usize) >> 8;
                let (ncx, ncy) = node_coord(ni, nj, frame_w, frame_h);
                let mut too_close = false;
                if gw * gh >= 32 {
                    let min_sep = if frame_w >= 640 { 4.5f32 } else { 2.2f32 };
                    let mut k = 0usize;
                    while k < selected {
                        let ak = self.ctrl.anchor_node_idx[k] as usize;
                        let (acx, acy) = node_coord(ak & 255, ak >> 8, frame_w, frame_h);
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
        cache: &mut BosRefCache,
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
            let (cx, cy) = node_coord(node & 255, node >> 8, frame_w, frame_h);
            if build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, node, p_r) {
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
            let (cx, cy) = node_coord(node & 255, node >> 8, frame_w, frame_h);
            if build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, node, p_r) {
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
            let (cx, cy) = node_coord(node & 255, node >> 8, frame_w, frame_h);
            if build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, node, p_r) {
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
        cache: &mut BosRefCache,
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
                let (cx, cy) = node_coord(node & 255, node >> 8, frame_w, frame_h);
                if build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, node, &mut p_ref) {
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
            cache,
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
                            cache,
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
        cache: &mut BosRefCache,
        pyr: &BosPyrScratch,
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

                if build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, idx, &mut p_ref) {
                    let (cx0, cy0, c_ok) = sample_bos_coarse_uv(pyr, i, j);
                    let use_coarse = c_ok && (cx0 * cx0 + cy0 * cy0) > (0.45 * 0.45);
                    let mut ex = if use_coarse { cx0 } else { 0.0f32 };
                    let mut ey = if use_coarse { cy0 } else { 0.0f32 };
                    let mut last_gain = 1.0f64;
                    let mut iter_ok = true;
                    let mut reusable_fit = None;

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
                                * (1.0 / fit.gain)) as f32;
                            let step_y = (-(-fit.h_xy * fit.bx + fit.h_xx * fit.by)
                                * inv_det
                                * (1.0 / fit.gain)) as f32;
                            if !step_x.is_finite() || !step_y.is_finite() {
                                iter_ok = false;
                                break;
                            }
                            ex += step_x;
                            ey += step_y;
                            let within_cap = if use_coarse {
                                let ddx = ex - cx0;
                                let ddy = ey - cy0;
                                (ddx * ddx + ddy * ddy) <= 4.0 && (ex * ex + ey * ey) <= 64.0
                            } else {
                                (ex * ex + ey * ey) <= 2.25
                            };
                            if !within_cap {
                                iter_ok = false;
                                break;
                            }
                            let step2 = step_x * step_x + step_y * step_y;
                            if step2 < 1.0e-6 {
                                reusable_fit = Some(fit);
                                break;
                            }
                            if step2 < 1.0e-5 {
                                break;
                            }
                        } else {
                            iter_ok = false;
                            break;
                        }
                        it += 1;
                    }

                    let final_within_cap = if use_coarse {
                        let ddx = ex - cx0;
                        let ddy = ey - cy0;
                        (ddx * ddx + ddy * ddy) <= 4.0 && (ex * ex + ey * ey) <= 64.0
                    } else {
                        (ex * ex + ey * ey) <= 2.25
                    };
                    if iter_ok && final_within_cap {
                        // Reuse converged static patch fit (< 0.001 px step); otherwise evaluate at final (ex, ey).
                        let final_fit_opt = if reusable_fit.is_some() {
                            reusable_fit
                        } else {
                            eval_cur_patch_8x8(
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
                            )
                        };
                        if let Some(final_fit) = final_fit_opt {
                            let m_cnt = final_fit.cnt;
                            let gain = final_fit.gain;
                            let offset = final_fit.offset;
                            if gain.is_finite() && gain >= 0.5 && gain <= 2.0 && m_cnt >= 40 {
                                let v_gain = f64x2_splat(gain);
                                let v_off = f64x2_splat(offset);
                                let mut s2_0 = f64x2_splat(0.0);
                                let mut s2_1 = f64x2_splat(0.0);
                                if m_cnt == 64 {
                                    let mut k = 0usize;
                                    while k < 64 {
                                        let r0 = f64x2_sub(
                                            f64x2(cur_buf[k] as f64, cur_buf[k + 1] as f64),
                                            f64x2_add(f64x2_mul(v_gain, f64x2(p_ref.t[k] as f64, p_ref.t[k + 1] as f64)), v_off),
                                        );
                                        let r1 = f64x2_sub(
                                            f64x2(cur_buf[k + 2] as f64, cur_buf[k + 3] as f64),
                                            f64x2_add(f64x2_mul(v_gain, f64x2(p_ref.t[k + 2] as f64, p_ref.t[k + 3] as f64)), v_off),
                                        );
                                        s2_0 = f64x2_add(s2_0, f64x2_mul(r0, r0));
                                        s2_1 = f64x2_add(s2_1, f64x2_mul(r1, r1));
                                        k += 4;
                                    }
                                } else {
                                    let mut k = 0usize;
                                    while k < 64 {
                                        let m0 = f64x2(
                                            if cur_mask[k] != 0 { 1.0 } else { 0.0 },
                                            if cur_mask[k + 1] != 0 { 1.0 } else { 0.0 },
                                        );
                                        let r0 = f64x2_mul(
                                            f64x2_sub(
                                                f64x2(cur_buf[k] as f64, cur_buf[k + 1] as f64),
                                                f64x2_add(f64x2_mul(v_gain, f64x2(p_ref.t[k] as f64, p_ref.t[k + 1] as f64)), v_off),
                                            ),
                                            m0,
                                        );
                                        let m1 = f64x2(
                                            if cur_mask[k + 2] != 0 { 1.0 } else { 0.0 },
                                            if cur_mask[k + 3] != 0 { 1.0 } else { 0.0 },
                                        );
                                        let r1 = f64x2_mul(
                                            f64x2_sub(
                                                f64x2(cur_buf[k + 2] as f64, cur_buf[k + 3] as f64),
                                                f64x2_add(f64x2_mul(v_gain, f64x2(p_ref.t[k + 2] as f64, p_ref.t[k + 3] as f64)), v_off),
                                            ),
                                            m1,
                                        );
                                        s2_0 = f64x2_add(s2_0, f64x2_mul(r0, r0));
                                        s2_1 = f64x2_add(s2_1, f64x2_mul(r1, r1));
                                        k += 4;
                                    }
                                }
                                let s2_v = f64x2_add(s2_0, s2_1);
                                let sum_r2 = f64x2_extract_lane::<0>(s2_v) + f64x2_extract_lane::<1>(s2_v);
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
    fn update_mode2_reference(
        &mut self,
        cache: &mut BosRefCache,
        cur_luma: &[f32],
        frame_w: usize,
        frame_h: usize,
    ) {
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
        let mut mod_x0 = usize::MAX;
        let mut mod_y0 = usize::MAX;
        let mut mod_x1 = 0usize;
        let mut mod_y1 = 0usize;

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
                    let (acx, acy) = node_coord(node & 255, node >> 8, frame_w, frame_h);
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
                        if px < mod_x0 {
                            mod_x0 = px;
                        }
                        if px > mod_x1 {
                            mod_x1 = px;
                        }
                        if py < mod_y0 {
                            mod_y0 = py;
                        }
                        if py > mod_y1 {
                            mod_y1 = py;
                        }
                    }
                }
                px += 1;
            }
            py += 1;
        }
        if mod_x0 <= mod_x1 && mod_y0 <= mod_y1 {
            self.ctrl.stereo_ref_ready = 0;
            self.ctrl.stereo_ref_fresh = 0;
            self.ctrl.stereo_ref_epoch = 0;
            // Each node (i, j) samples a 10x10 bilinear window covering [cx - 5.5, cx + 6.5]
            // (conservative 6.5 px radius around node_coord). Invalidate only overlapping nodes
            // instead of flushing the entire 8,192-node cache.
            let x_lo = (mod_x0 as f32) - 6.5;
            let x_hi = (mod_x1 as f32) + 6.5;
            let y_lo = (mod_y0 as f32) - 6.5;
            let y_hi = (mod_y1 as f32) + 6.5;
            let ni_lo = floor_f32((x_lo + 0.5) * inv_hx - 0.5).max(0.0) as usize;
            let ni_hi = (floor_f32((x_hi + 0.5) * inv_hx + 0.5).max(0.0) as usize).min(BOS_GRID_W - 1);
            let nj_lo = floor_f32((y_lo + 0.5) * inv_hy - 0.5).max(0.0) as usize;
            let nj_hi = (floor_f32((y_hi + 0.5) * inv_hy + 0.5).max(0.0) as usize).min(BOS_GRID_H - 1);
            let mut j = nj_lo;
            while j <= nj_hi {
                let row_n = j * BOS_GRID_W;
                let mut i = ni_lo;
                while i <= ni_hi {
                    cache.entries[row_n + i].gen = 0;
                    i += 1;
                }
                j += 1;
            }
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

        // 1. Discrete curl on 4-valid-corner plaquettes + exact 2x2 Kelvin-Stokes macro-plaquette
        // average matching the 8x8 LK patch support on 256x128 (`5.0 x 5.625 px` loop).
        let mut curl_cnt = 0u32;

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
                    }
                }
                i += 1;
            }
            j += 1;
        }

        let mut curl_sum2 = 0.0f64;
        let mut max_abs_curl = 0.0f64;
        j = 0;
        while j < BOS_GRID_H - 1 {
            let mut i = 0usize;
            while i < BOS_GRID_W - 1 {
                let tl = j * BOS_GRID_W + i;
                if (self.valid_mask[tl] & BOS_VALID_CURL) != 0 {
                    let mut omega = self.planes[BOS_PLANE_CURL * BOS_NODES + tl] as f64;
                    if i + 2 < BOS_GRID_W && j + 2 < BOS_GRID_H {
                        let tr = tl + 1;
                        let bl = tl + BOS_GRID_W;
                        let br = bl + 1;
                        if (self.valid_mask[tr] & BOS_VALID_CURL) != 0
                            && (self.valid_mask[bl] & BOS_VALID_CURL) != 0
                            && (self.valid_mask[br] & BOS_VALID_CURL) != 0
                        {
                            omega = 0.25
                                * (omega
                                    + (self.planes[BOS_PLANE_CURL * BOS_NODES + tr] as f64)
                                    + (self.planes[BOS_PLANE_CURL * BOS_NODES + bl] as f64)
                                    + (self.planes[BOS_PLANE_CURL * BOS_NODES + br] as f64));
                            self.planes[BOS_PLANE_CURL * BOS_NODES + tl] = omega as f32;
                        }
                    }
                    curl_sum2 += omega * omega;
                    let ao = omega.abs();
                    if ao > max_abs_curl {
                        max_abs_curl = ao;
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
                let j = u >> 8;
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
                        let cj = cur >> 8;
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
    /// Note: caller writes input in bit-reversed order; `m` must be `128` or `64`.
    #[inline(always)]
    fn fft_radix2_paired(&mut self, m: usize) {
        let mut half = 1usize;
        let mut step_tw = 64usize;
        while half < m {
            let len = half << 1;
            let mut i = 0usize;
            while i < m {
                let mut k = 0usize;
                let mut tw_idx = 0usize;
                while k < half {
                    let wr = f64x2_splat(DCT_TABLES.tw128_re[tw_idx]);
                    let wi = f64x2_splat(DCT_TABLES.tw128_im[tw_idx]);
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

    /// Paired orthonormal 1D DCT-II across two rows (`N = 256`, `m = 128`) or two columns (`N = 128`, `m = 64`).
    /// Operates in-place on `solve_z` using `fft_line` (`f64x2` SIMD128).
    #[inline(never)]
    fn dct2_2d_solve_z(&mut self) {
        // 1. 128 rows of N = 256 (64 pairs)
        let mut r = 0usize;
        while r < BOS_GRID_H {
            let r0 = r * BOS_GRID_W;
            let r1 = r0 + BOS_GRID_W;
            let mut dst = 0usize;
            while dst < 128 {
                let p = DCT_TABLES.bitrev128[dst] as usize;
                let (e_idx, o_idx) = if p < 64 {
                    (4 * p, 4 * p + 2)
                } else {
                    let q = 127 - p;
                    (4 * q + 3, 4 * q + 1)
                };
                self.fft_line[dst * 2] = self.solve_z[r0 + e_idx];
                self.fft_line[dst * 2 + 1] = self.solve_z[r1 + e_idx];
                self.fft_line[(128 + dst) * 2] = self.solve_z[r0 + o_idx];
                self.fft_line[(128 + dst) * 2 + 1] = self.solve_z[r1 + o_idx];
                dst += 1;
            }
            self.fft_radix2_paired(128);

            let e0 = f64x2(self.fft_line[0], self.fft_line[1]);
            let o0 = f64x2(self.fft_line[256], self.fft_line[257]);
            let dc = f64x2_mul(f64x2_add(e0, o0), f64x2_splat(DCT_TABLES.scale_dc_256));
            let nyq = f64x2_mul(f64x2_sub(e0, o0), f64x2_splat(DCT_TABLES.scale_nyq_256));
            self.solve_z[r0] = f64x2_extract_lane::<0>(dc);
            self.solve_z[r1] = f64x2_extract_lane::<1>(dc);
            self.solve_z[r0 + 128] = f64x2_extract_lane::<0>(nyq);
            self.solve_z[r1 + 128] = f64x2_extract_lane::<1>(nyq);

            let half = f64x2_splat(0.5);
            let sc = f64x2_splat(DCT_TABLES.scale_fwd_256);
            let mut k = 1usize;
            while k < 128 {
                let mk = 128 - k;
                let zkr = f64x2(self.fft_line[k * 2], self.fft_line[k * 2 + 1]);
                let zki = f64x2(self.fft_line[(128 + k) * 2], self.fft_line[(128 + k) * 2 + 1]);
                let zmr = f64x2(self.fft_line[mk * 2], self.fft_line[mk * 2 + 1]);
                let zmi = f64x2(-self.fft_line[(128 + mk) * 2], -self.fft_line[(128 + mk) * 2 + 1]);

                let er = f64x2_mul(half, f64x2_add(zkr, zmr));
                let ei = f64x2_mul(half, f64x2_add(zki, zmi));
                let or = f64x2_mul(half, f64x2_sub(zki, zmi));
                let oi = f64x2_mul(half, f64x2_sub(zmr, zkr));

                let wnr = f64x2_splat(DCT_TABLES.wn256_re[k]);
                let wni = f64x2_splat(DCT_TABLES.wn256_im[k]);
                let yr = f64x2_add(er, f64x2_sub(f64x2_mul(or, wnr), f64x2_mul(oi, wni)));
                let yi = f64x2_add(ei, f64x2_add(f64x2_mul(or, wni), f64x2_mul(oi, wnr)));

                let wdr = f64x2_splat(DCT_TABLES.wd256_re[k]);
                let wdi = f64x2_splat(DCT_TABLES.wd256_im[k]);
                let xk = f64x2_mul(sc, f64x2_sub(f64x2_mul(yr, wdr), f64x2_mul(yi, wdi)));
                let xnk = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.scale_fwd_256),
                    f64x2_add(f64x2_mul(yr, wdi), f64x2_mul(yi, wdr)),
                );
                self.solve_z[r0 + k] = f64x2_extract_lane::<0>(xk);
                self.solve_z[r1 + k] = f64x2_extract_lane::<1>(xk);
                self.solve_z[r0 + 256 - k] = f64x2_extract_lane::<0>(xnk);
                self.solve_z[r1 + 256 - k] = f64x2_extract_lane::<1>(xnk);
                k += 1;
            }
            r += 2;
        }

        // 2. 256 columns of N = 128 (128 adjacent column pairs)
        let mut c = 0usize;
        while c < BOS_GRID_W {
            let mut dst = 0usize;
            while dst < 64 {
                let p = DCT_TABLES.bitrev64[dst] as usize;
                let (e_idx, o_idx) = if p < 32 {
                    (4 * p, 4 * p + 2)
                } else {
                    let q = 63 - p;
                    (4 * q + 3, 4 * q + 1)
                };
                let erow = e_idx * BOS_GRID_W + c;
                let orow = o_idx * BOS_GRID_W + c;
                self.fft_line[dst * 2] = self.solve_z[erow];
                self.fft_line[dst * 2 + 1] = self.solve_z[erow + 1];
                self.fft_line[(64 + dst) * 2] = self.solve_z[orow];
                self.fft_line[(64 + dst) * 2 + 1] = self.solve_z[orow + 1];
                dst += 1;
            }
            self.fft_radix2_paired(64);

            let e0 = f64x2(self.fft_line[0], self.fft_line[1]);
            let o0 = f64x2(self.fft_line[128], self.fft_line[129]);
            let dc = f64x2_mul(f64x2_add(e0, o0), f64x2_splat(DCT_TABLES.scale_dc_128));
            let nyq = f64x2_mul(f64x2_sub(e0, o0), f64x2_splat(DCT_TABLES.scale_nyq_128));
            self.solve_z[c] = f64x2_extract_lane::<0>(dc);
            self.solve_z[c + 1] = f64x2_extract_lane::<1>(dc);
            self.solve_z[64 * BOS_GRID_W + c] = f64x2_extract_lane::<0>(nyq);
            self.solve_z[64 * BOS_GRID_W + c + 1] = f64x2_extract_lane::<1>(nyq);

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

                let wnr = f64x2_splat(DCT_TABLES.wn256_re[k * 2]);
                let wni = f64x2_splat(DCT_TABLES.wn256_im[k * 2]);
                let yr = f64x2_add(er, f64x2_sub(f64x2_mul(or, wnr), f64x2_mul(oi, wni)));
                let yi = f64x2_add(ei, f64x2_add(f64x2_mul(or, wni), f64x2_mul(oi, wnr)));

                let wdr = f64x2_splat(DCT_TABLES.wd256_re[k * 2]);
                let wdi = f64x2_splat(DCT_TABLES.wd256_im[k * 2]);
                let xk = f64x2_mul(sc, f64x2_sub(f64x2_mul(yr, wdr), f64x2_mul(yi, wdi)));
                let xnk = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.scale_fwd_128),
                    f64x2_add(f64x2_mul(yr, wdi), f64x2_mul(yi, wdr)),
                );
                let row_k = k * BOS_GRID_W + c;
                let row_nk = (128 - k) * BOS_GRID_W + c;
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
        // 1. 256 columns of N = 128 (128 pairs)
        let s0_128 = f64x2_splat(DCT_TABLES.sqrt_128);
        let sk_128 = f64x2_splat(DCT_TABLES.sqrt_128 * 0.70710678118654752440);
        let half = f64x2_splat(0.5);

        let mut c = 0usize;
        while c < BOS_GRID_W {
            let y0 = f64x2_mul(s0_128, f64x2(self.solve_z[c], self.solve_z[c + 1]));
            let ym = f64x2_mul(
                s0_128,
                f64x2(
                    self.solve_z[64 * BOS_GRID_W + c],
                    self.solve_z[64 * BOS_GRID_W + c + 1],
                ),
            );
            // Store yRe[0..64] at fft_line[512..640], yIm[0..64] at fft_line[640..768]
            self.fft_line[512] = f64x2_extract_lane::<0>(y0);
            self.fft_line[513] = f64x2_extract_lane::<1>(y0);
            self.fft_line[640] = 0.0;
            self.fft_line[641] = 0.0;
            let mut k = 1usize;
            while k < 64 {
                let row_k = k * BOS_GRID_W + c;
                let row_nk = (128 - k) * BOS_GRID_W + c;
                let ur = f64x2_mul(sk_128, f64x2(self.solve_z[row_k], self.solve_z[row_k + 1]));
                let ui = f64x2_mul(sk_128, f64x2(-self.solve_z[row_nk], -self.solve_z[row_nk + 1]));
                let wdr = f64x2_splat(DCT_TABLES.wd256_re[k * 2]);
                let wdi = f64x2_splat(-DCT_TABLES.wd256_im[k * 2]);
                let yr = f64x2_sub(f64x2_mul(ur, wdr), f64x2_mul(ui, wdi));
                let yi = f64x2_add(f64x2_mul(ur, wdi), f64x2_mul(ui, wdr));
                self.fft_line[512 + k * 2] = f64x2_extract_lane::<0>(yr);
                self.fft_line[512 + k * 2 + 1] = f64x2_extract_lane::<1>(yr);
                self.fft_line[640 + k * 2] = f64x2_extract_lane::<0>(yi);
                self.fft_line[640 + k * 2 + 1] = f64x2_extract_lane::<1>(yi);
                k += 1;
            }

            // Build bit-reversed conj(Z) directly into fft_line[0..256]
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
                let ykr = f64x2(self.fft_line[512 + k * 2], self.fft_line[512 + k * 2 + 1]);
                let yki = f64x2(self.fft_line[640 + k * 2], self.fft_line[640 + k * 2 + 1]);
                let ymr = f64x2(self.fft_line[512 + mk * 2], self.fft_line[512 + mk * 2 + 1]);
                let ymi = f64x2(-self.fft_line[640 + mk * 2], -self.fft_line[640 + mk * 2 + 1]);

                let er = f64x2_mul(half, f64x2_add(ykr, ymr));
                let ei = f64x2_mul(half, f64x2_add(yki, ymi));
                let dr = f64x2_mul(half, f64x2_sub(ykr, ymr));
                let di = f64x2_mul(half, f64x2_sub(yki, ymi));
                let wnr = f64x2_splat(DCT_TABLES.wn256_re[k * 2]);
                let wni = f64x2_splat(-DCT_TABLES.wn256_im[k * 2]);
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
            let mut r = 0usize;
            while r < 64 {
                let ye = f64x2_mul(inv_m, f64x2(self.fft_line[r * 2], self.fft_line[r * 2 + 1]));
                let yo = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.inv_64),
                    f64x2(self.fft_line[(64 + r) * 2], self.fft_line[(64 + r) * 2 + 1]),
                );
                let (e_idx, o_idx) = if r < 32 {
                    (4 * r, 4 * r + 2)
                } else {
                    let q = 63 - r;
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

        // 2. 128 rows of N = 256 (64 pairs)
        let s0_256 = f64x2_splat(DCT_TABLES.sqrt_256);
        let sk_256 = f64x2_splat(DCT_TABLES.sqrt_256 * 0.70710678118654752440);

        let mut r = 0usize;
        while r < BOS_GRID_H {
            let r0 = r * BOS_GRID_W;
            let r1 = r0 + BOS_GRID_W;
            let y0 = f64x2_mul(s0_256, f64x2(self.solve_z[r0], self.solve_z[r1]));
            let ym = f64x2_mul(s0_256, f64x2(self.solve_z[r0 + 128], self.solve_z[r1 + 128]));
            // Store yRe[0..128] at fft_line[512..768], yIm[0..128] at fft_line[768..1024]
            self.fft_line[512] = f64x2_extract_lane::<0>(y0);
            self.fft_line[513] = f64x2_extract_lane::<1>(y0);
            self.fft_line[768] = 0.0;
            self.fft_line[769] = 0.0;
            let mut k = 1usize;
            while k < 128 {
                let ur = f64x2_mul(sk_256, f64x2(self.solve_z[r0 + k], self.solve_z[r1 + k]));
                let ui = f64x2_mul(
                    sk_256,
                    f64x2(-self.solve_z[r0 + 256 - k], -self.solve_z[r1 + 256 - k]),
                );
                let wdr = f64x2_splat(DCT_TABLES.wd256_re[k]);
                let wdi = f64x2_splat(-DCT_TABLES.wd256_im[k]);
                let yr = f64x2_sub(f64x2_mul(ur, wdr), f64x2_mul(ui, wdi));
                let yi = f64x2_add(f64x2_mul(ur, wdi), f64x2_mul(ui, wdr));
                self.fft_line[512 + k * 2] = f64x2_extract_lane::<0>(yr);
                self.fft_line[512 + k * 2 + 1] = f64x2_extract_lane::<1>(yr);
                self.fft_line[768 + k * 2] = f64x2_extract_lane::<0>(yi);
                self.fft_line[768 + k * 2 + 1] = f64x2_extract_lane::<1>(yi);
                k += 1;
            }

            let z0r = f64x2_mul(half, f64x2_add(y0, ym));
            let z0i = f64x2_mul(half, f64x2_sub(y0, ym));
            self.fft_line[0] = f64x2_extract_lane::<0>(z0r);
            self.fft_line[1] = f64x2_extract_lane::<1>(z0r);
            self.fft_line[256] = -f64x2_extract_lane::<0>(z0i);
            self.fft_line[257] = -f64x2_extract_lane::<1>(z0i);

            let mut k = 1usize;
            while k < 128 {
                let dst = DCT_TABLES.bitrev128[k] as usize;
                let mk = 128 - k;
                let ykr = f64x2(self.fft_line[512 + k * 2], self.fft_line[512 + k * 2 + 1]);
                let yki = f64x2(self.fft_line[768 + k * 2], self.fft_line[768 + k * 2 + 1]);
                let ymr = f64x2(self.fft_line[512 + mk * 2], self.fft_line[512 + mk * 2 + 1]);
                let ymi = f64x2(-self.fft_line[768 + mk * 2], -self.fft_line[768 + mk * 2 + 1]);

                let er = f64x2_mul(half, f64x2_add(ykr, ymr));
                let ei = f64x2_mul(half, f64x2_add(yki, ymi));
                let dr = f64x2_mul(half, f64x2_sub(ykr, ymr));
                let di = f64x2_mul(half, f64x2_sub(yki, ymi));
                let wnr = f64x2_splat(DCT_TABLES.wn256_re[k]);
                let wni = f64x2_splat(-DCT_TABLES.wn256_im[k]);
                let or = f64x2_sub(f64x2_mul(dr, wnr), f64x2_mul(di, wni));
                let oi = f64x2_add(f64x2_mul(dr, wni), f64x2_mul(di, wnr));

                let zr = f64x2_sub(er, oi);
                let zi = f64x2_add(ei, or);
                self.fft_line[dst * 2] = f64x2_extract_lane::<0>(zr);
                self.fft_line[dst * 2 + 1] = f64x2_extract_lane::<1>(zr);
                self.fft_line[(128 + dst) * 2] = -f64x2_extract_lane::<0>(zi);
                self.fft_line[(128 + dst) * 2 + 1] = -f64x2_extract_lane::<1>(zi);
                k += 1;
            }
            self.fft_radix2_paired(128);

            let inv_m = f64x2_splat(DCT_TABLES.inv_128);
            let mut idx_r = 0usize;
            while idx_r < 128 {
                let ye = f64x2_mul(inv_m, f64x2(self.fft_line[idx_r * 2], self.fft_line[idx_r * 2 + 1]));
                let yo = f64x2_mul(
                    f64x2_splat(-DCT_TABLES.inv_128),
                    f64x2(
                        self.fft_line[(128 + idx_r) * 2],
                        self.fft_line[(128 + idx_r) * 2 + 1],
                    ),
                );
                let (e_idx, o_idx) = if idx_r < 64 {
                    (4 * idx_r, 4 * idx_r + 2)
                } else {
                    let q = 127 - idx_r;
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
        self.solve_lp[1..=nc].fill(0.0);
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            if cid > 0 {
                let val = if target_is_z { self.solve_z[u] } else { self.solve_r[u] };
                self.solve_lp[cid] += val;
            }
            u += 1;
        }
        let mut c = 1usize;
        while c <= nc {
            let cnt = self.comp_count[c] as f64;
            self.solve_lp[c] = if cnt > 0.0 { self.solve_lp[c] / cnt } else { 0.0 };
            c += 1;
        }
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            if target_is_z {
                self.solve_z[u] = if cid > 0 {
                    self.solve_z[u] - self.solve_lp[cid]
                } else {
                    0.0
                };
            } else {
                self.solve_r[u] = if cid > 0 {
                    self.solve_r[u] - self.solve_lp[cid]
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
        self.solve_lp[1..=nc].fill(0.0);
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            if cid > 0 {
                self.solve_lp[cid] += self.solve_x[u];
            }
            u += 1;
        }
        let mut c = 1usize;
        while c <= nc {
            let cnt = self.comp_count[c] as f64;
            self.solve_lp[c] = if cnt > 0.0 { self.solve_lp[c] / cnt } else { 0.0 };
            c += 1;
        }
        let mut u = 0usize;
        while u < BOS_NODES {
            let cid = self.comp_id[u] as usize;
            self.solve_x[u] = if cid > 0 {
                self.solve_x[u] - self.solve_lp[cid]
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
        self.solve_lp[1..=nc].fill(0.0); // temporarily reuse solve_lp for gauge sum
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
                    self.solve_lp[cid] += self.solve_x[u];
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

        let gauge_mean = self.solve_lp[best_cid] / (best_overlap as f64);
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
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        cur_luma: &[f32],
        cur_pyr: &[f32],
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
        cache: &mut BosRefCache,
        pyr: &mut BosPyrScratch,
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
            self.ctrl.stereo_ref_ready = 0;
            self.ctrl.stereo_ref_fresh = 0;
            self.ctrl.stereo_ref_epoch = 0;
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
                self.ctrl.stereo_ref_ready = 0;
                self.ctrl.stereo_ref_fresh = 0;
                self.ctrl.stereo_ref_epoch = 0;
                self.ref_luma[..need_px].copy_from_slice(&cur_luma[..need_px]);
                self.bump_ref_gen();
                self.ref_noise_var.fill(0.0);
                self.clear_streak.fill(0);
                self.ctrl.anchor_base_mse = 0.0;
                let n_anch = self.select_anchors_in_gauge(cache, frame_w, frame_h);
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
                self.evaluate_frozen_anchors(cache, cur_luma, frame_w, frame_h);
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
            self.accumulate_warmup_node_noise(cache, cur_luma, frame_w, frame_h);
            self.ctrl.anchor_base_mse += amse;

            let n_next = self.ctrl.warmup_count + 1;
            let alpha = 1.0f32 / (n_next as f32);
            self.ctrl.stereo_ref_ready = 0;
            self.ctrl.stereo_ref_fresh = 0;
            self.ctrl.stereo_ref_epoch = 0;
            let mut p = 0usize;
            while p < need_px {
                self.ref_luma[p] += (cur_luma[p] - self.ref_luma[p]) * alpha;
                p += 1;
            }
            self.bump_ref_gen();
            self.ctrl.warmup_count = n_next;
            self.telem[10] = n_next;

            if n_next == BOS_WARMUP_FRAMES {
                self.ctrl.anchor_base_mse *= 1.0 / 15.0;
                let mut u = 0usize;
                while u < BOS_NODES {
                    self.ref_noise_var[u] *= 1.0 / 15.0;
                    u += 1;
                }
                let n_anch = self.select_anchors_in_gauge(cache, frame_w, frame_h);
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
            self.evaluate_frozen_anchors(cache, cur_luma, frame_w, frame_h);
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

        // 3. Dense 128x64 Subpixel Refractive Displacement Extraction (with 2-level coarse-to-fine pyramid)
        let maint_mode = self.ctrl.requested_mode == BOS_MODE_MAINT;
        sync_bos_ref_pyr(&self.ref_luma, frame_w, frame_h, self.ctrl.ref_gen, pyr);
        compute_bos_coarse_pyr_lk(cur_pyr, pyr, frame_w, frame_h);
        let (valid_disp, rms_disp) =
            self.extract_dense_lk(cache, pyr, cur_luma, frame_w, frame_h, maint_mode);
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
                    self.update_mode2_reference(cache, cur_luma, frame_w, frame_h);
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
    fn accumulate_warmup_node_noise(
        &mut self,
        cache: &mut BosRefCache,
        cur_luma: &[f32],
        frame_w: usize,
        frame_h: usize,
    ) {
        let mut p_ref = PatchRefData::empty();
        let mut cur_buf = [0.0f32; 64];
        let mut cur_mask = [0u8; 64];
        let mut j = 0usize;
        while j < BOS_GRID_H {
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let idx = j * BOS_GRID_W + i;
                let (cx, cy) = node_coord(i, j, frame_w, frame_h);
                if build_ref_patch_node(cache, self.ctrl.ref_gen, &self.ref_luma, frame_w, frame_h, idx, &mut p_ref) {
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
                            let v_gain = f64x2_splat(fit.gain);
                            let v_off = f64x2_splat(fit.offset);
                            let mut s2_0 = f64x2_splat(0.0);
                            let mut s2_1 = f64x2_splat(0.0);
                            let mut k = 0usize;
                            while k < 64 {
                                let m0 = f64x2(
                                    if cur_mask[k] != 0 { 1.0 } else { 0.0 },
                                    if cur_mask[k + 1] != 0 { 1.0 } else { 0.0 },
                                );
                                let r0 = f64x2_mul(
                                    f64x2_sub(
                                        f64x2(cur_buf[k] as f64, cur_buf[k + 1] as f64),
                                        f64x2_add(f64x2_mul(v_gain, f64x2(p_ref.t[k] as f64, p_ref.t[k + 1] as f64)), v_off),
                                    ),
                                    m0,
                                );
                                let m1 = f64x2(
                                    if cur_mask[k + 2] != 0 { 1.0 } else { 0.0 },
                                    if cur_mask[k + 3] != 0 { 1.0 } else { 0.0 },
                                );
                                let r1 = f64x2_mul(
                                    f64x2_sub(
                                        f64x2(cur_buf[k + 2] as f64, cur_buf[k + 3] as f64),
                                        f64x2_add(f64x2_mul(v_gain, f64x2(p_ref.t[k + 2] as f64, p_ref.t[k + 3] as f64)), v_off),
                                    ),
                                    m1,
                                );
                                s2_0 = f64x2_add(s2_0, f64x2_mul(r0, r0));
                                s2_1 = f64x2_add(s2_1, f64x2_mul(r1, r1));
                                k += 4;
                            }
                            let s2_v = f64x2_add(s2_0, s2_1);
                            let s2 = f64x2_extract_lane::<0>(s2_v) + f64x2_extract_lane::<1>(s2_v);
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
                let in_hole = j >= 48 && j < 80 && i >= 110 && i < 146;
                let in_split = i >= 120 && i < 130;
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
            (110usize, 48usize, 146usize, 80usize)
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

    #[inline(always)]
    pub fn has_fresh_stereo_ref(
        &self,
        frame_w: usize,
        frame_h: usize,
        frame_idx: u32,
        continuity_epoch: u32,
    ) -> bool {
        self.ctrl.stereo_ref_ready != 0
            && self.ctrl.stereo_ref_fresh != 0
            && self.ctrl.stereo_ref_frame_idx == frame_idx
            && self.ctrl.stereo_ref_epoch == continuity_epoch
            && (self.ctrl.last_frame_w as usize) == frame_w
            && (self.ctrl.last_frame_h as usize) == frame_h
            && frame_w >= 16
            && frame_h >= 16
    }

    #[inline(always)]
    pub fn invalidate_stereo_ref(&mut self) {
        self.ctrl.stereo_ref_ready = 0;
        self.ctrl.stereo_ref_fresh = 0;
        self.ctrl.stereo_ref_epoch = 0;
        self.ctrl.stereo_ref_frame_idx = u32::MAX;
    }

    /// Capture an RGBA8 frame directly as the Left rectified stereo reference bound to
    /// `(frame_idx, continuity_epoch)`, without advancing `ring_cursor`. Invalidates BOS tripod
    /// state cleanly so BOS never misinterprets stereo `ref_luma` as a static Schlieren background.
    #[inline(never)]
    pub fn capture_stereo_left_rgba(
        &mut self,
        rgba: &[u8],
        frame_w: usize,
        frame_h: usize,
        frame_idx: u32,
        continuity_epoch: u32,
    ) -> u32 {
        let need_px = frame_w.saturating_mul(frame_h);
        if frame_w < 16
            || frame_h < 16
            || frame_w > BOS_MAX_FRAME_W
            || frame_h > BOS_MAX_FRAME_H
            || rgba.len() < need_px.saturating_mul(4)
        {
            return 0;
        }
        if self.ctrl.initialized == 0 {
            self.init_in_place();
        }
        const INV255: f32 = 1.0 / 255.0;
        let mut p = 0usize;
        while p < need_px {
            let o = p * 4;
            let r = rgba[o] as f32;
            let g = rgba[o + 1] as f32;
            let b = rgba[o + 2] as f32;
            self.ref_luma[p] = (0.299 * r + 0.587 * g + 0.114 * b) * INV255;
            p += 1;
        }
        self.bump_ref_gen();
        self.ref_noise_var.fill(1.0e-5);
        self.ctrl.warmup_count = 0;
        self.ctrl.anchor_count = 0;
        self.ctrl.clear_consec_frames = 0;
        self.telem[2] = BOS_STATE_OFF;
        self.telem[4] = 0;
        self.ctrl.last_frame_w = frame_w as u16;
        self.ctrl.last_frame_h = frame_h as u16;
        self.ctrl.stereo_ref_ready = 1;
        self.ctrl.stereo_ref_fresh = 1;
        self.ctrl.stereo_ref_epoch = continuity_epoch;
        self.ctrl.stereo_ref_frame_idx = frame_idx;
        1
    }

    /// Option B: `256x128` Calibrated Rectified Stereo Disparity & Metric Depth (`SURVEYED_STEREO`).
    ///
    /// - Consumes the single-pair Left reference (`stereo_ref_fresh = 0`).
    /// - Coarse-to-fine 1D horizontal ZNCC epipolar search (`d <= max_disp_px`) with peak-uniqueness
    ///   rejection (`second_sc <= 0.88 * best_sc`) and reverse R->L pyramid consistency check.
    /// - SIMD128 `8x8` affine-brightness-invariant IC-LK subpixel refiner + strict `|ey| <= 0.25 px`
    ///   and `d <= max_disp_px` bounds.
    /// - 720p-safe scanline visibility / half-occlusion culling via `self.solve_x[..frame_w]`.
    /// - Left-reference metric `Z` in `BOS_PLANE_PHI`, `1/Z` in `BOS_PLANE_CURL`, local fit
    ///   uncertainty `sigma_Z` in `BOS_PLANE_VAR_Y`; Right-frame z-buffered depth colormap in `rgba`.
    /// - Never writes `BOS_STATE_READY` into `self.telem[2]` so BOS Schlieren ABI stays isolated.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_stereo(
        &mut self,
        cache: &mut BosRefCache,
        pyr: &mut BosPyrScratch,
        cur_luma: &[f32],
        cur_pyr: &[f32],
        frame_w: usize,
        frame_h: usize,
        f_px: f32,
        baseline_m: f32,
        max_disp_px: f32,
        sign: i32,
    ) -> (u32, f32, f32, f32, f32) {
        // Consume pair immediately so unpaired subsequent frames fail closed
        let was_fresh = self.ctrl.stereo_ref_ready != 0 && self.ctrl.stereo_ref_fresh != 0;
        self.ctrl.stereo_ref_fresh = 0;

        let need_px = frame_w.saturating_mul(frame_h);
        if !was_fresh
            || (self.ctrl.last_frame_w as usize) != frame_w
            || (self.ctrl.last_frame_h as usize) != frame_h
            || cur_luma.len() < need_px
            || !f_px.is_finite()
            || f_px < 10.0
            || !baseline_m.is_finite()
            || baseline_m < 1e-4
            || !max_disp_px.is_finite()
            || max_disp_px < 1.0
            || max_disp_px > 64.0
        {
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_OFF;
            return (0, 0.0, 0.0, 0.0, 0.0);
        }

        let max_d = max_disp_px;
        let wide_mode = max_d > 2.5;
        let has_pyr = wide_mode && cur_pyr.len() >= BOS_PYR_W * BOS_PYR_H;
        if has_pyr {
            sync_bos_ref_pyr(&self.ref_luma, frame_w, frame_h, self.ctrl.ref_gen, pyr);
        }

        let scale_px_x = (frame_w as f32) * (1.0 / (BOS_PYR_W as f32));
        let inv_scale_x = (BOS_PYR_W as f32) / (frame_w as f32);
        let inv_scale_y = (BOS_PYR_H as f32) / (frame_h as f32);
        let max_dc = (floor_f32(max_d * inv_scale_x) as isize) + 1;
        let dc_lo = if sign > 0 { 0 } else { -max_dc };
        let dc_hi = if sign < 0 { 0 } else { max_dc };
        let dx_int_lo_glob = if sign > 0 { 0isize } else { -(floor_f32(max_d) as isize) };
        let dx_int_hi_glob = if sign < 0 { 0isize } else { floor_f32(max_d) as isize };

        let mut p_ref = PatchRefData::empty();
        let mut cur_buf = [0.0f32; 64];
        let mut cur_mask = [0u8; 64];

        self.planes.fill(0.0);
        self.valid_mask.fill(0);

        let mut raw_valid = 0u32;
        let mut pos_cnt = 0u32;
        let mut neg_cnt = 0u32;

        let mut j = 0usize;
        while j < BOS_GRID_H {
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let idx = j * BOS_GRID_W + i;
                let (cx, cy) = node_coord(i, j, frame_w, frame_h);
                if build_ref_patch_node(
                    cache,
                    self.ctrl.ref_gen,
                    &self.ref_luma,
                    frame_w,
                    frame_h,
                    idx,
                    &mut p_ref,
                ) {
                    let mut init_ex = 0.0f32;
                    let mut coarse_ok = true;
                    if wide_mode {
                        let mut center_dx = 0isize;
                        let mut pyr_found = false;
                        if has_pyr {
                            let pcx = floor_f32(cx * inv_scale_x + 0.5) as isize;
                            let pcy = floor_f32(cy * inv_scale_y + 0.5) as isize;
                            if pcx >= 1
                                && pcx + 1 < (BOS_PYR_W as isize)
                                && pcy >= 1
                                && pcy + 1 < (BOS_PYR_H as isize)
                            {
                                let mut rt = [0.0f32; 9];
                                let mut sum_rt = 0.0f32;
                                let mut r = 0usize;
                                while r < 3 {
                                    let row_off =
                                        ((pcy as usize) + r - 1) * BOS_PYR_W + (pcx as usize) - 1;
                                    rt[r * 3] = pyr.ref_pyr[row_off];
                                    rt[r * 3 + 1] = pyr.ref_pyr[row_off + 1];
                                    rt[r * 3 + 2] = pyr.ref_pyr[row_off + 2];
                                    sum_rt += rt[r * 3] + rt[r * 3 + 1] + rt[r * 3 + 2];
                                    r += 1;
                                }
                                let mu_rt = sum_rt * (1.0 / 9.0);
                                let mut var_rt = 0.0f32;
                                let mut k = 0usize;
                                while k < 9 {
                                    rt[k] -= mu_rt;
                                    var_rt += rt[k] * rt[k];
                                    k += 1;
                                }
                                if var_rt > 2.0e-5 {
                                    // Store scores across dc_lo..=dc_hi to enforce true peak uniqueness (72 covers sign==0 up to 320px/64px)
                                    let mut dc_scores = [-1.0f32; 72];
                                    let mut best_score = -1.0f32;
                                    let mut best_dc = 0isize;
                                    let mut best_c_patch = [0.0f32; 9];
                                    let mut best_c_var = 0.0f32;
                                    let mut dc = dc_lo;
                                    while dc <= dc_hi {
                                        let slot_dc = (dc - dc_lo) as usize;
                                        let ccx = pcx + dc;
                                        if ccx >= 1 && ccx + 1 < (BOS_PYR_W as isize) {
                                            let mut sum_c = 0.0f32;
                                            let mut cv = [0.0f32; 9];
                                            let mut rr = 0usize;
                                            while rr < 3 {
                                                let ro = ((pcy as usize) + rr - 1) * BOS_PYR_W
                                                    + (ccx as usize)
                                                    - 1;
                                                let v0 = cur_pyr[ro];
                                                let v1 = cur_pyr[ro + 1];
                                                let v2 = cur_pyr[ro + 2];
                                                cv[rr * 3] = v0;
                                                cv[rr * 3 + 1] = v1;
                                                cv[rr * 3 + 2] = v2;
                                                sum_c += v0 + v1 + v2;
                                                rr += 1;
                                            }
                                            let mu_c = sum_c * (1.0 / 9.0);
                                            let mut cov = 0.0f32;
                                            let mut var_c = 0.0f32;
                                            let mut kk = 0usize;
                                            while kk < 9 {
                                                let d = cv[kk] - mu_c;
                                                cv[kk] = d;
                                                cov += rt[kk] * d;
                                                var_c += d * d;
                                                kk += 1;
                                            }
                                            if cov > 0.0 && var_c > 2.0e-5 {
                                                let sc = (cov * cov) / (var_rt * var_c);
                                                if slot_dc < 72 {
                                                    dc_scores[slot_dc] = sc;
                                                }
                                                if sc > best_score {
                                                    best_score = sc;
                                                    best_dc = dc;
                                                    best_c_patch = cv;
                                                    best_c_var = var_c;
                                                }
                                            }
                                        }
                                        dc += 1;
                                    }

                                    // Uniqueness check: reject ambiguous repetitive textures
                                    let mut second_sc = -1.0f32;
                                    let mut dc2 = dc_lo;
                                    while dc2 <= dc_hi {
                                        if (dc2 - best_dc).abs() >= 2 {
                                            let slot_dc = (dc2 - dc_lo) as usize;
                                            if slot_dc < 72 && dc_scores[slot_dc] > second_sc {
                                                second_sc = dc_scores[slot_dc];
                                            }
                                        }
                                        dc2 += 1;
                                    }

                                    if best_score > 0.30 && second_sc <= 0.85 * best_score {
                                        // Reverse R->L pyramid consistency check
                                        let rcx = pcx + best_dc;
                                        let mut rev_best_sc = -1.0f32;
                                        let mut rev_best_dc = 0isize;
                                        let mut rdc = -dc_hi;
                                        while rdc <= -dc_lo {
                                            let lcx = rcx + rdc;
                                            if lcx >= 1 && lcx + 1 < (BOS_PYR_W as isize) {
                                                let mut sum_l = 0.0f32;
                                                let mut lv = [0.0f32; 9];
                                                let mut rr = 0usize;
                                                while rr < 3 {
                                                    let ro = ((pcy as usize) + rr - 1) * BOS_PYR_W
                                                        + (lcx as usize)
                                                        - 1;
                                                    let v0 = pyr.ref_pyr[ro];
                                                    let v1 = pyr.ref_pyr[ro + 1];
                                                    let v2 = pyr.ref_pyr[ro + 2];
                                                    lv[rr * 3] = v0;
                                                    lv[rr * 3 + 1] = v1;
                                                    lv[rr * 3 + 2] = v2;
                                                    sum_l += v0 + v1 + v2;
                                                    rr += 1;
                                                }
                                                let mu_l = sum_l * (1.0 / 9.0);
                                                let mut cov = 0.0f32;
                                                let mut var_l = 0.0f32;
                                                let mut kk = 0usize;
                                                while kk < 9 {
                                                    let d = lv[kk] - mu_l;
                                                    cov += best_c_patch[kk] * d;
                                                    var_l += d * d;
                                                    kk += 1;
                                                }
                                                if cov > 0.0 && var_l > 2.0e-5 {
                                                    let sc = (cov * cov) / (best_c_var * var_l);
                                                    if sc > rev_best_sc {
                                                        rev_best_sc = sc;
                                                        rev_best_dc = rdc;
                                                    }
                                                }
                                            }
                                            rdc += 1;
                                        }
                                        if rev_best_sc > 0.25 && (rev_best_dc + best_dc).abs() <= 1
                                        {
                                            center_dx =
                                                floor_f32((best_dc as f32) * scale_px_x + 0.5)
                                                    as isize;
                                            pyr_found = true;
                                        } else {
                                            coarse_ok = false;
                                        }
                                    } else if second_sc > 0.85 * best_score && best_score > 0.30 {
                                        // Explicitly reject periodic/ambiguous pyramid peak instead of falling through
                                        coarse_ok = false;
                                    }
                                }
                            }
                        }

                        // Stage 1b: Full-res integer 1D scan over strided 4x4 subsample with uniqueness check
                        if coarse_ok {
                            let icx = floor_f32(cx + 0.5) as isize;
                            let icy = floor_f32(cy + 0.5) as isize;
                            if icx >= 4
                                && icx + 4 < (frame_w as isize)
                                && icy >= 4
                                && icy + 4 < (frame_h as isize)
                            {
                                let mut t16 = [0.0f32; 16];
                                let mut sum_t16 = 0.0f32;
                                let mut q = 0usize;
                                let mut py4 = 0usize;
                                while py4 < 4 {
                                    let sy = (icy as usize) - 3 + py4 * 2;
                                    let row_o = sy * frame_w;
                                    let mut px4 = 0usize;
                                    while px4 < 4 {
                                        let sx = (icx as usize) - 3 + px4 * 2;
                                        let v = self.ref_luma[row_o + sx];
                                        t16[q] = v;
                                        sum_t16 += v;
                                        q += 1;
                                        px4 += 1;
                                    }
                                    py4 += 1;
                                }
                                let mu_t16 = sum_t16 * (1.0 / 16.0);
                                let mut var_t16 = 0.0f32;
                                let mut qq = 0usize;
                                while qq < 16 {
                                    t16[qq] -= mu_t16;
                                    var_t16 += t16[qq] * t16[qq];
                                    qq += 1;
                                }
                                if var_t16 > 5.0e-5 {
                                    let (scan_lo, scan_hi, step) = if pyr_found {
                                        (
                                            (center_dx - 4).max(dx_int_lo_glob),
                                            (center_dx + 4).min(dx_int_hi_glob),
                                            1isize,
                                        )
                                    } else {
                                        (dx_int_lo_glob, dx_int_hi_glob, 2isize)
                                    };
                                    let mut best_sc = -1.0f32;
                                    let mut best_dx = 0isize;
                                    let mut dx_i = scan_lo;
                                    while dx_i <= scan_hi {
                                        let tx = icx + dx_i;
                                        if tx >= 4 && tx + 4 < (frame_w as isize) {
                                            let mut c16 = [0.0f32; 16];
                                            let mut sum_c16 = 0.0f32;
                                            let mut idx16 = 0usize;
                                            let mut r4 = 0usize;
                                            while r4 < 4 {
                                                let sy = (icy as usize) - 3 + r4 * 2;
                                                let ro = sy * frame_w;
                                                let mut c4 = 0usize;
                                                while c4 < 4 {
                                                    let sx = (tx as usize) - 3 + c4 * 2;
                                                    let v = cur_luma[ro + sx];
                                                    c16[idx16] = v;
                                                    sum_c16 += v;
                                                    idx16 += 1;
                                                    c4 += 1;
                                                }
                                                r4 += 1;
                                            }
                                            let mu_c16 = sum_c16 * (1.0 / 16.0);
                                            let mut cov = 0.0f32;
                                            let mut var_c16 = 0.0f32;
                                            let mut kk = 0usize;
                                            while kk < 16 {
                                                let d = c16[kk] - mu_c16;
                                                cov += t16[kk] * d;
                                                var_c16 += d * d;
                                                kk += 1;
                                            }
                                            if cov > 0.0 && var_c16 > 5.0e-5 {
                                                let sc = (cov * cov) / (var_t16 * var_c16);
                                                if sc > best_sc {
                                                    best_sc = sc;
                                                    best_dx = dx_i;
                                                }
                                            }
                                        }
                                        dx_i += step;
                                    }

                                    // Sub-pixel refine best_dx match score (apples-to-apples with sub-pixel
                                    // autocorrelation so sharp aperiodic textures at half-pixel disparity are
                                    // never falsely rejected), then screen Right rivals and Left autocorrelation.
                                    let mut best_sc_ref = best_sc;
                                    let mut rival_sc = -1.0f32;
                                    if best_sc > 0.15 {
                                        let mut t1 = [0.0f32; 16];
                                        let mut v11 = 0.0f32;
                                        let mut v01 = 0.0f32;
                                        let mut has_t1 = false;
                                        if icx + 5 < (frame_w as isize) {
                                            let mut sum_t1 = 0.0f32;
                                            let mut q1 = 0usize;
                                            let mut py1 = 0usize;
                                            while py1 < 4 {
                                                let sy = (icy as usize) - 3 + py1 * 2;
                                                let row_o = sy * frame_w;
                                                let mut px1 = 0usize;
                                                while px1 < 4 {
                                                    let sx = (icx as usize) - 2 + px1 * 2;
                                                    let v = self.ref_luma[row_o + sx];
                                                    t1[q1] = v;
                                                    sum_t1 += v;
                                                    q1 += 1;
                                                    px1 += 1;
                                                }
                                                py1 += 1;
                                            }
                                            let mu_t1 = sum_t1 * (1.0 / 16.0);
                                            let mut qq1 = 0usize;
                                            while qq1 < 16 {
                                                let d1 = t1[qq1] - mu_t1;
                                                t1[qq1] = d1;
                                                v11 += d1 * d1;
                                                v01 += t16[qq1] * d1;
                                                qq1 += 1;
                                            }
                                            let det_t01 = var_t16 * v11 - v01 * v01;
                                            if v11 > 5.0e-5 && det_t01 > 1.0e-6 * var_t16 * v11 {
                                                has_t1 = true;
                                            }
                                        }
                                        let (ref_dx, _ref_int, ref_sub) = refine_best_match_subpx(
                                            cur_luma,
                                            frame_w,
                                            icx,
                                            icy as usize,
                                            &t16,
                                            var_t16,
                                            &t1,
                                            v11,
                                            v01,
                                            has_t1,
                                            dx_int_lo_glob,
                                            dx_int_hi_glob,
                                            best_dx,
                                            best_sc,
                                        );
                                        best_dx = ref_dx;
                                        best_sc_ref = ref_sub;

                                        if best_sc_ref > 0.40 {
                                            let rival_thresh = 0.85 * best_sc_ref;
                                            let mut r_dx = dx_int_lo_glob;
                                            while r_dx <= dx_int_hi_glob {
                                                if (r_dx - best_dx).abs() >= 4 {
                                                    let tx = icx + r_dx;
                                                    if tx >= 4 && tx + 4 < (frame_w as isize) {
                                                        let mut c16 = [0.0f32; 16];
                                                        let mut sum_c16 = 0.0f32;
                                                        let mut idx16 = 0usize;
                                                        let mut r4 = 0usize;
                                                        while r4 < 4 {
                                                            let sy = (icy as usize) - 3 + r4 * 2;
                                                            let ro = sy * frame_w;
                                                            let mut c4 = 0usize;
                                                            while c4 < 4 {
                                                                let sx = (tx as usize) - 3 + c4 * 2;
                                                                let v = cur_luma[ro + sx];
                                                                c16[idx16] = v;
                                                                sum_c16 += v;
                                                                idx16 += 1;
                                                                c4 += 1;
                                                            }
                                                            r4 += 1;
                                                        }
                                                        let mu_c16 = sum_c16 * (1.0 / 16.0);
                                                        let mut cov = 0.0f32;
                                                        let mut var_c16 = 0.0f32;
                                                        let mut kk = 0usize;
                                                        while kk < 16 {
                                                            let d = c16[kk] - mu_c16;
                                                            cov += t16[kk] * d;
                                                            var_c16 += d * d;
                                                            kk += 1;
                                                        }
                                                        if cov > 0.0
                                                            && var_c16 > 5.0e-5
                                                            && var_c16 >= 0.25 * var_t16
                                                            && var_c16 <= 4.0 * var_t16
                                                        {
                                                            let sc =
                                                                (cov * cov) / (var_t16 * var_c16);
                                                            if sc > rival_sc {
                                                                rival_sc = sc;
                                                                if rival_sc > rival_thresh {
                                                                    break;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                r_dx += 2;
                                            }
                                            if rival_sc <= rival_thresh {
                                                let max_auto = (floor_f32(2.0 * max_d) as isize
                                                    + 1)
                                                    .clamp(4, 128);
                                                rival_sc = screen_patch_autocorr_1d(
                                                    &self.ref_luma,
                                                    frame_w,
                                                    icx,
                                                    icy as usize,
                                                    &t16,
                                                    mu_t16,
                                                    var_t16,
                                                    &t1,
                                                    v11,
                                                    v01,
                                                    has_t1,
                                                    max_auto,
                                                    rival_thresh,
                                                    rival_sc,
                                                );
                                            }
                                        }
                                    }

                                    if best_sc_ref > 0.40 && rival_sc <= 0.85 * best_sc_ref {
                                        init_ex = best_dx as f32;
                                    } else {
                                        if best_sc_ref > 0.40 && rival_sc > 0.85 * best_sc_ref {
                                            self.valid_mask[idx] = 0x80;
                                        }
                                        coarse_ok = false;
                                    }
                                } else {
                                    coarse_ok = false;
                                }
                            } else {
                                coarse_ok = false;
                            }
                        }
                    }

                    // Stage 2: SIMD128 8x8 Affine-Brightness-Invariant IC-LK Subpixel Refinement
                    if coarse_ok {
                        let mut ex = init_ex;
                        let mut ey = 0.0f32;
                        let mut iter_ok = true;
                        let mut last_step2 = 0.0f32;
                        let mut reusable_fit = None;
                        let max_iters = 4usize;
                        let mut it = 0usize;
                        while it < max_iters {
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
                                let inv_det = 1.0 / fit.det;
                                let step_x = (-(fit.h_yy * fit.bx - fit.h_xy * fit.by)
                                    * inv_det
                                    * (1.0 / fit.gain)) as f32;
                                let step_y = (-(-fit.h_xy * fit.bx + fit.h_xx * fit.by)
                                    * inv_det
                                    * (1.0 / fit.gain)) as f32;
                                if !step_x.is_finite() || !step_y.is_finite() {
                                    iter_ok = false;
                                    break;
                                }
                                ex += step_x;
                                ey += step_y;
                                if (ex - init_ex).abs() > 2.5
                                    || ex.abs() > max_d + 0.05
                                    || ey.abs() > 1.25
                                {
                                    iter_ok = false;
                                    break;
                                }
                                let step2 = step_x * step_x + step_y * step_y;
                                last_step2 = step2;
                                if step2 < 1.0e-6 {
                                    reusable_fit = Some(fit);
                                    break;
                                }
                                if step2 < 1.0e-5 {
                                    break;
                                }
                            } else {
                                iter_ok = false;
                                break;
                            }
                            it += 1;
                        }

                        // Stage 3: Strict Rectified Vertical Epipolar Gate (|ey| <= 0.25 px),
                        // IC-LK convergence check, and strict disparity bound |ex| <= max_d
                        if iter_ok && last_step2 <= 0.02 && ey.abs() <= 0.25 && ex.abs() <= max_d {
                            let final_fit_opt = if reusable_fit.is_some() {
                                reusable_fit
                            } else {
                                eval_cur_patch_8x8(
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
                                )
                            };
                            if let Some(final_fit) = final_fit_opt {
                                let m_cnt = final_fit.cnt;
                                let gain = final_fit.gain;
                                let offset = final_fit.offset;
                                if gain.is_finite() && gain >= 0.5 && gain <= 2.0 && m_cnt >= 40 {
                                    let v_gain = f64x2_splat(gain);
                                    let v_off = f64x2_splat(offset);
                                    let mut s2_0 = f64x2_splat(0.0);
                                    let mut s2_1 = f64x2_splat(0.0);
                                    let mut k = 0usize;
                                    while k < 64 {
                                        let m0 = f64x2(
                                            if cur_mask[k] != 0 { 1.0 } else { 0.0 },
                                            if cur_mask[k + 1] != 0 { 1.0 } else { 0.0 },
                                        );
                                        let r0 = f64x2_mul(
                                            f64x2_sub(
                                                f64x2(cur_buf[k] as f64, cur_buf[k + 1] as f64),
                                                f64x2_add(
                                                    f64x2_mul(
                                                        v_gain,
                                                        f64x2(
                                                            p_ref.t[k] as f64,
                                                            p_ref.t[k + 1] as f64,
                                                        ),
                                                    ),
                                                    v_off,
                                                ),
                                            ),
                                            m0,
                                        );
                                        let m1 = f64x2(
                                            if cur_mask[k + 2] != 0 { 1.0 } else { 0.0 },
                                            if cur_mask[k + 3] != 0 { 1.0 } else { 0.0 },
                                        );
                                        let r1 = f64x2_mul(
                                            f64x2_sub(
                                                f64x2(
                                                    cur_buf[k + 2] as f64,
                                                    cur_buf[k + 3] as f64,
                                                ),
                                                f64x2_add(
                                                    f64x2_mul(
                                                        v_gain,
                                                        f64x2(
                                                            p_ref.t[k + 2] as f64,
                                                            p_ref.t[k + 3] as f64,
                                                        ),
                                                    ),
                                                    v_off,
                                                ),
                                            ),
                                            m1,
                                        );
                                        s2_0 = f64x2_add(s2_0, f64x2_mul(r0, r0));
                                        s2_1 = f64x2_add(s2_1, f64x2_mul(r1, r1));
                                        k += 4;
                                    }
                                    let s2_v = f64x2_add(s2_0, s2_1);
                                    let sum_r2 = f64x2_extract_lane::<0>(s2_v)
                                        + f64x2_extract_lane::<1>(s2_v);
                                    let final_mse = sum_r2 / (m_cnt as f64);
                                    let res_limit = (0.035 * 0.035f64).max(0.35 * p_ref.ref_var);
                                    if final_mse.is_finite() && final_mse <= res_limit {
                                        let dof = (m_cnt - 4).max(1) as f64;
                                        let inv_det = 1.0 / final_fit.det;
                                        let sigma_ref2 = (sum_r2 / dof) / (gain * gain);
                                        let vx = (sigma_ref2 * final_fit.h_yy * inv_det + 2.25e-4)
                                            .max(1e-4);
                                        if ex.is_finite()
                                            && ey.is_finite()
                                            && vx.is_finite()
                                            && vx <= 0.1225
                                        {
                                            self.planes[BOS_PLANE_EPS_X * BOS_NODES + idx] = ex;
                                            self.planes[BOS_PLANE_EPS_Y * BOS_NODES + idx] = ey;
                                            self.planes[BOS_PLANE_VAR_X * BOS_NODES + idx] =
                                                vx as f32;
                                            self.valid_mask[idx] = BOS_VALID_DISP;
                                            raw_valid += 1;
                                            if ex > 0.05 {
                                                pos_cnt += 1;
                                            } else if ex < -0.05 {
                                                neg_cnt += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                i += 1;
            }
            j += 1;
        }

        // On rows with periodic rejections (`0x80`), verify surviving nodes in Right `cur_luma`
        // (catches half-image large periods where the second cycle only appears in the shifted
        // Right frame, skipping rows below the threshold on clean aperiodic scenes),
        // then bridge narrow (~7.5 px) seam cracks flanked on BOTH horizontal sides by `0x80`,
        // and clear the temporary `0x80` bit before downstream stages.
        let max_auto_r = (floor_f32(4.0 * max_d) as isize + 1).clamp(4, 140);
        let bridge_r: usize = if frame_w <= 320 {
            6
        } else if frame_w <= 640 {
            3
        } else {
            2
        };
        let mut bj = 0usize;
        while bj < BOS_GRID_H {
            let row = bj * BOS_GRID_W;
            let mut row_periodic = 0u32;
            let mut pi = 0usize;
            while pi < BOS_GRID_W {
                if (self.valid_mask[row + pi] & 0x80) != 0 {
                    row_periodic += 1;
                }
                pi += 1;
            }
            if wide_mode && row_periodic >= 4 {
                let mut ri = 0usize;
                while ri < BOS_GRID_W {
                    let u = row + ri;
                    if (self.valid_mask[u] & BOS_VALID_DISP) != 0 {
                        let (cx, cy) = node_coord(ri, bj, frame_w, frame_h);
                        let ex = self.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                        let rx = floor_f32(cx + ex + 0.5) as isize;
                        let icy = floor_f32(cy + 0.5) as isize;
                        if rx >= 4
                            && rx + 4 < (frame_w as isize)
                            && icy >= 4
                            && icy + 4 < (frame_h as isize)
                        {
                            let icy_u = icy as usize;
                            let mut tr16 = [0.0f32; 16];
                            let mut sum_tr16 = 0.0f32;
                            let mut q = 0usize;
                            let mut py4 = 0usize;
                            while py4 < 4 {
                                let sy = icy_u - 3 + py4 * 2;
                                let ro = sy * frame_w;
                                let mut px4 = 0usize;
                                while px4 < 4 {
                                    let sx = (rx as usize) - 3 + px4 * 2;
                                    let v = cur_luma[ro + sx];
                                    tr16[q] = v;
                                    sum_tr16 += v;
                                    q += 1;
                                    px4 += 1;
                                }
                                py4 += 1;
                            }
                            let mu_tr16 = sum_tr16 * (1.0 / 16.0);
                            let mut var_tr16 = 0.0f32;
                            let mut qq = 0usize;
                            while qq < 16 {
                                tr16[qq] -= mu_tr16;
                                var_tr16 += tr16[qq] * tr16[qq];
                                qq += 1;
                            }
                            if var_tr16 > 5.0e-5 {
                                let mut tr1 = [0.0f32; 16];
                                let mut vr11 = 0.0f32;
                                let mut vr01 = 0.0f32;
                                let mut has_tr1 = false;
                                if rx + 5 < (frame_w as isize) {
                                    let mut sum_tr1 = 0.0f32;
                                    let mut q1 = 0usize;
                                    let mut py1 = 0usize;
                                    while py1 < 4 {
                                        let sy = icy_u - 3 + py1 * 2;
                                        let ro = sy * frame_w;
                                        let mut px1 = 0usize;
                                        while px1 < 4 {
                                            let sx = (rx as usize) - 2 + px1 * 2;
                                            let v = cur_luma[ro + sx];
                                            tr1[q1] = v;
                                            sum_tr1 += v;
                                            q1 += 1;
                                            px1 += 1;
                                        }
                                        py1 += 1;
                                    }
                                    let mu_tr1 = sum_tr1 * (1.0 / 16.0);
                                    let mut qq1 = 0usize;
                                    while qq1 < 16 {
                                        let d1 = tr1[qq1] - mu_tr1;
                                        tr1[qq1] = d1;
                                        vr11 += d1 * d1;
                                        vr01 += tr16[qq1] * d1;
                                        qq1 += 1;
                                    }
                                    let det_r01 = var_tr16 * vr11 - vr01 * vr01;
                                    if vr11 > 5.0e-5 && det_r01 > 1.0e-6 * var_tr16 * vr11 {
                                        has_tr1 = true;
                                    }
                                }
                                let r_sc = screen_patch_autocorr_1d(
                                    cur_luma,
                                    frame_w,
                                    rx,
                                    icy_u,
                                    &tr16,
                                    mu_tr16,
                                    var_tr16,
                                    &tr1,
                                    vr11,
                                    vr01,
                                    has_tr1,
                                    max_auto_r,
                                    0.85,
                                    -1.0,
                                );
                                if r_sc > 0.85 {
                                    self.valid_mask[u] = 0x80;
                                    raw_valid = raw_valid.saturating_sub(1);
                                    if ex > 0.05 {
                                        pos_cnt = pos_cnt.saturating_sub(1);
                                    } else if ex < -0.05 {
                                        neg_cnt = neg_cnt.saturating_sub(1);
                                    }
                                }
                            }
                        }
                    }
                    ri += 1;
                }
            }
            let mut bi = 0usize;
            while bi < BOS_GRID_W {
                let u = row + bi;
                if (self.valid_mask[u] & BOS_VALID_DISP) != 0 {
                    let mut has_l = false;
                    let mut kl = 1usize;
                    while kl <= bridge_r && bi >= kl {
                        if (self.valid_mask[u - kl] & 0x80) != 0 {
                            has_l = true;
                            break;
                        }
                        kl += 1;
                    }
                    if has_l {
                        let mut has_r = false;
                        let mut kr = 1usize;
                        while kr <= bridge_r && bi + kr < BOS_GRID_W {
                            if (self.valid_mask[u + kr] & 0x80) != 0 {
                                has_r = true;
                                break;
                            }
                            kr += 1;
                        }
                        if has_r {
                            let ex = self.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                            self.valid_mask[u] = 0;
                            raw_valid = raw_valid.saturating_sub(1);
                            if ex > 0.05 {
                                pos_cnt = pos_cnt.saturating_sub(1);
                            } else if ex < -0.05 {
                                neg_cnt = neg_cnt.saturating_sub(1);
                            }
                        }
                    }
                }
                bi += 1;
            }
            let mut ci = 0usize;
            while ci < BOS_GRID_W {
                self.valid_mask[row + ci] &= !0x80;
                ci += 1;
            }
            bj += 1;
        }

        if raw_valid == 0 {
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_OFF;
            return (0, 0.0, 0.0, 0.0, 0.0);
        }

        let eff_sign: f32 = if sign < 0 {
            -1.0
        } else if sign > 0 {
            1.0
        } else if neg_cnt >= pos_cnt {
            -1.0
        } else {
            1.0
        };

        // Stage 4: 720p-Safe Scanline Visibility / Half-Occlusion Culling + Metric Depth Z = fB / d
        // Reuse `self.solve_x[0..frame_w]` (`32,768` f64s, covers 320/640/1280 with 0 stack!).
        let zbuf_len = frame_w.min(BOS_NODES);
        let fb = f_px * baseline_m;
        let mut valid_stereo = 0u32;
        let mut epi_sq_sum = 0.0f64;
        let mut min_z = f32::INFINITY;
        let mut max_z = 0.0f32;
        let mut min_iz = f32::INFINITY;
        let mut max_iz = 0.0f32;

        let mut j = 0usize;
        while j < BOS_GRID_H {
            if wide_mode {
                let mut bx = 0usize;
                while bx < zbuf_len {
                    self.solve_x[bx] = 0.0;
                    bx += 1;
                }
                let mut i = 0usize;
                while i < BOS_GRID_W {
                    let u = j * BOS_GRID_W + i;
                    if (self.valid_mask[u] & BOS_VALID_DISP) != 0 {
                        let ex = self.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                        let d = eff_sign * ex;
                        if d >= 0.05 && d <= max_d {
                            let (cx, _) = node_coord(i, j, frame_w, frame_h);
                            let xr = floor_f32(cx + ex + 0.5) as isize;
                            if xr >= 0 && (xr as usize) < zbuf_len {
                                let xru = xr as usize;
                                if (d as f64) > self.solve_x[xru] {
                                    self.solve_x[xru] = d as f64;
                                }
                            }
                        }
                    }
                    i += 1;
                }
            }

            let mut i = 0usize;
            while i < BOS_GRID_W {
                let u = j * BOS_GRID_W + i;
                if (self.valid_mask[u] & BOS_VALID_DISP) != 0 {
                    let ex = self.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                    let ey = self.planes[BOS_PLANE_EPS_Y * BOS_NODES + u];
                    let vx = self.planes[BOS_PLANE_VAR_X * BOS_NODES + u];
                    let d = eff_sign * ex;
                    if d >= 0.05 && d <= max_d {
                        let mut occluded = false;
                        if wide_mode {
                            let (cx, _) = node_coord(i, j, frame_w, frame_h);
                            let xr = floor_f32(cx + ex + 0.5) as isize;
                            if xr >= 0 && (xr as usize) < zbuf_len {
                                let xru = xr as usize;
                                let x0 = xru.saturating_sub(2);
                                let x1 = (xru + 2).min(zbuf_len - 1);
                                let mut d_front = 0.0f64;
                                let mut xx = x0;
                                while xx <= x1 {
                                    if self.solve_x[xx] > d_front {
                                        d_front = self.solve_x[xx];
                                    }
                                    xx += 1;
                                }
                                if d_front > (d as f64) + 2.0 {
                                    occluded = true;
                                }
                            }
                        }
                        if !occluded {
                            let z = fb / d;
                            let inv_z = d / fb;
                            let sigma_z = ((z * z) / fb) * sqrt_f32(vx.max(1e-6));
                            if z.is_finite()
                                && z > 0.0
                                && inv_z.is_finite()
                                && inv_z > 1e-6
                                && sigma_z.is_finite()
                            {
                                self.planes[BOS_PLANE_PHI * BOS_NODES + u] = z;
                                self.planes[BOS_PLANE_CURL * BOS_NODES + u] = inv_z;
                                self.planes[BOS_PLANE_VAR_Y * BOS_NODES + u] = sigma_z;
                                self.valid_mask[u] |= BOS_VALID_PHI;
                                valid_stereo += 1;
                                epi_sq_sum += (ey as f64) * (ey as f64);
                                if z < min_z {
                                    min_z = z;
                                }
                                if z > max_z {
                                    max_z = z;
                                }
                                if inv_z < min_iz {
                                    min_iz = inv_z;
                                }
                                if inv_z > max_iz {
                                    max_iz = inv_z;
                                }
                            }
                        }
                    }
                }
                i += 1;
            }
            j += 1;
        }

        if valid_stereo == 0 {
            self.invalidate_outputs();
            self.telem[2] = BOS_STATE_OFF;
            return (0, 0.0, 0.0, 0.0, 0.0);
        }

        // Coprime spatial sampling across all columns & rows for median Z (in `self.fft_line[..512]`)
        let mut samp_n = 0usize;
        let mut step_k = 0usize;
        while step_k < BOS_NODES && samp_n < 512 {
            let u = (step_k.wrapping_mul(67)) & (BOS_NODES - 1);
            if (self.valid_mask[u] & BOS_VALID_PHI) != 0 {
                self.fft_line[samp_n] = self.planes[BOS_PLANE_PHI * BOS_NODES + u] as f64;
                samp_n += 1;
            }
            step_k += 1;
        }
        let mut gap = samp_n >> 1;
        while gap > 0 {
            let mut ii = gap;
            while ii < samp_n {
                let key = self.fft_line[ii];
                let mut jj = ii;
                while jj >= gap && self.fft_line[jj - gap] > key {
                    self.fft_line[jj] = self.fft_line[jj - gap];
                    jj -= gap;
                }
                self.fft_line[jj] = key;
                ii += 1;
            }
            gap >>= 1;
        }
        let med_z = if samp_n > 0 {
            self.fft_line[samp_n >> 1] as f32
        } else {
            min_z
        };
        let epi_rms = sqrt_f64(epi_sq_sum / (valid_stereo as f64)) as f32;

        // Keep BOS Schlieren ABI isolated (`BOS_STATE_OFF`; stereo telemetry lives in `DepthState::telem`)
        self.telem[2] = BOS_STATE_OFF;
        self.telem[4] = 0;

        // Stage 5: Render Right-frame z-buffered 256x128 perceptual inverse-depth heatmap into `self.rgba`
        self.rgba.fill(0);
        let iz_span = (max_iz - min_iz).max(1e-4);
        let inv_span = 1.0 / iz_span;
        let inv_hx = (BOS_GRID_W as f32) / (frame_w as f32);
        let mut jj = 0usize;
        while jj < BOS_GRID_H {
            let mut bi = 0usize;
            while bi < BOS_GRID_W {
                self.solve_x[bi] = 0.0;
                bi += 1;
            }
            let mut ii = 0usize;
            while ii < BOS_GRID_W {
                let u = jj * BOS_GRID_W + ii;
                if (self.valid_mask[u] & BOS_VALID_PHI) != 0 {
                    let iz = self.planes[BOS_PLANE_CURL * BOS_NODES + u];
                    let ex = self.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                    let (cx_l, _) = node_coord(ii, jj, frame_w, frame_h);
                    let cx_r = cx_l + ex;
                    if cx_r >= 0.0 && cx_r < (frame_w as f32) {
                        let ir = (floor_f32(cx_r * inv_hx) as isize)
                            .clamp(0, (BOS_GRID_W - 1) as isize) as usize;
                        if (iz as f64) > self.solve_x[ir] {
                            self.solve_x[ir] = iz as f64;
                            let t = ((iz - min_iz) * inv_span).clamp(0.0, 1.0);
                            let (r, g, b) = if t < 0.35 {
                                let s = t * (1.0 / 0.35);
                                (18.0 + 18.0 * s, 44.0 + 132.0 * s, 118.0 + 106.0 * s)
                            } else if t < 0.70 {
                                let s = (t - 0.35) * (1.0 / 0.35);
                                (36.0 + 206.0 * s, 176.0 + 42.0 * s, 224.0 - 154.0 * s)
                            } else {
                                let s = (t - 0.70) * (1.0 / 0.30);
                                (242.0 + 13.0 * s, 218.0 - 138.0 * s, 70.0 - 38.0 * s)
                            };
                            let off = (jj * BOS_GRID_W + ir) * 4;
                            self.rgba[off] = r as u8;
                            self.rgba[off + 1] = g as u8;
                            self.rgba[off + 2] = b as u8;
                            self.rgba[off + 3] = 220;
                        }
                    }
                }
                ii += 1;
            }
            jj += 1;
        }

        (valid_stereo, epi_rms, med_z, min_z, max_z)
    }
}
