//! Zero-ML Depth Sensing, Focus-of-Expansion / Time-to-Contact, Stereo Bridge & External Depth ABI.
//!
//! Pure `#![no_std]` zero-allocation, zero-`unsafe`, zero-import implementation inside static `DepthState`:
//! - Option A: Monocular motion-parallax inverse depth (`1/Z`) over `MicroTrackSoA` using an `8`-frame
//!   contiguous `u16 Q5` OLS sliding window (true elapsed `dt`), a fail-closed 3-DoF Huber-IRLS
//!   pure-rotation gate with hysteresis (`REFUSED_PURE_ROTATION` / `REFUSED_NO_BASELINE`), a 4-term
//!   bilinear cross-flow purity check (`[1, s_n, p_n, s_n*p_n]`) that rejects mixed rotation+translation
//!   and oblique contamination before emitting `MONO_METRIC`, epipolar independent-motion rejection
//!   (`MOVING_EPIPOLAR_OUTLIER`), and intrinsics-free RANSAC Focus of Expansion (FoE) + `f64`
//!   multi-frame `1/r(t)` optical looming Time-to-Contact (`tau = Z / -dZ/dt`, assumes fixed `f`).
//! - Option B: Strict paired bridge to `BosState::step_stereo` (`256x128` coarse-to-fine 1D epipolar
//!   ZNCC + uniqueness + L-R consistency + SIMD128 `8x8` affine IC-LK + `|eps_y| <= 0.25 px` +
//!   720p-safe scanline occlusion culling -> `SURVEYED_STEREO`), reprojected to Right-frame `64x36`
//!   coordinates for Hero-16 & display alignment.
//! - Option C: Consume-once, epoch-bound external depth-plane ingest (`<= 128x64` `f32`) with strict
//!   provenance firewall (`SURVEYED_LIDAR`, `SURVEYED_STEREO`, `ASSUMED_NEURAL_PRIOR`).

use crate::bos::{
    BosPyrScratch, BosRefCache, BosState, BOS_GRID_H, BOS_GRID_W, BOS_NODES, BOS_PLANE_CURL,
    BOS_PLANE_EPS_X, BOS_PLANE_PHI, BOS_VALID_PHI,
};
use crate::soa_tracks::{MicroSwarmState, MAX_MICRO, MICRO_STATE_CONFIRMED};
use crate::{wasm_floorf, wasm_sqrt_f64, wasm_sqrtf, MAX_TARGETS};

pub const DEPTH_GRID_W: usize = 64;
pub const DEPTH_GRID_H: usize = 36;
pub const DEPTH_CELLS: usize = DEPTH_GRID_W * DEPTH_GRID_H; // 2,304
pub const DEPTH_FIELD_STRIDE: usize = 2; // [inv_depth, confidence_or_weight]
pub const DEPTH_TRACK_STRIDE: usize = 2; // [inv_depth, ttc_sec]
pub const DEPTH_HERO_STRIDE: usize = 4; // [depth_z, inv_depth, ttc_sec, provenance]
pub const DEPTH_TELEM_LEN: usize = 32;

pub const EXT_DEPTH_MAX_W: usize = 128;
pub const EXT_DEPTH_MAX_H: usize = 64;
pub const EXT_DEPTH_MAX_CELLS: usize = EXT_DEPTH_MAX_W * EXT_DEPTH_MAX_H; // 8,192

pub const DEPTH_HIST_LEN: usize = 8;
const DEPTH_Q5_SCALE: f32 = 32.0;
const DEPTH_Q5_INV: f32 = 1.0 / 32.0;

const MAX_SOLVE_TRACKS: usize = 512;
const MAX_FOE_LINES: usize = 96;
const MAX_TTC_FITS: usize = 64;
const TTC_MAX_WINDOW_FRAMES: u32 = 96;

// Operating modes (`chronos_set_depth_mode`)
pub const DEPTH_MODE_OFF: u32 = 0;
pub const DEPTH_MODE_MONO_PARALLAX_TTC: u32 = 1;
pub const DEPTH_MODE_STEREO_BOS: u32 = 2;
pub const DEPTH_MODE_EXTERNAL: u32 = 3;
pub const DEPTH_MODE_AUTO_FUSED: u32 = 4;

// Telemetry state (`telem[2]`)
pub const DEPTH_STATE_OFF: u32 = 0;
pub const DEPTH_STATE_WARMING: u32 = 1;
pub const DEPTH_STATE_READY_PARALLAX: u32 = 2;
pub const DEPTH_STATE_READY_TTC_ONLY: u32 = 3;
pub const DEPTH_STATE_REFUSED_PURE_ROTATION: u32 = 4;
pub const DEPTH_STATE_REFUSED_NO_BASELINE: u32 = 5;
pub const DEPTH_STATE_READY_STEREO: u32 = 6;
pub const DEPTH_STATE_READY_EXTERNAL: u32 = 7;

// Fail-closed reason mask (`telem[3]`)
pub const DEPTH_REASON_PURE_ROTATION: u32 = 1 << 0;
pub const DEPTH_REASON_NO_BASELINE: u32 = 1 << 1;
pub const DEPTH_REASON_LOW_TRACKS: u32 = 1 << 2;
pub const DEPTH_REASON_NO_INTRINSICS: u32 = 1 << 3;
pub const DEPTH_REASON_EXT_STALE: u32 = 1 << 4;
pub const DEPTH_REASON_ASSUMED_NEURAL_PRIOR: u32 = 1 << 5;
pub const DEPTH_REASON_STEREO_NOT_READY: u32 = 1 << 6;
pub const DEPTH_REASON_MIXED_ROT_TRANS: u32 = 1 << 7;
pub const DEPTH_REASON_BAD_DT: u32 = 1 << 8;

// Provenance tags (`telem[4]` & `export_hero_depth[..+3]`)
pub const DEPTH_PROV_NONE: u32 = 0;
pub const DEPTH_PROV_MONO_UP_TO_SCALE: u32 = 1;
pub const DEPTH_PROV_MONO_METRIC: u32 = 2;
pub const DEPTH_PROV_INTRINSICS_FREE_TTC: u32 = 3;
pub const DEPTH_PROV_SURVEYED_STEREO: u32 = 4;
pub const DEPTH_PROV_SURVEYED_LIDAR: u32 = 5;
pub const DEPTH_PROV_ASSUMED_NEURAL_PRIOR: u32 = 6;

// External ingest provenance argument (`chronos_set_depth_external_meta`)
pub const EXT_PROV_SURVEYED_LIDAR: u32 = 1;
pub const EXT_PROV_SURVEYED_STEREO: u32 = 2;
pub const EXT_PROV_ASSUMED_NEURAL_PRIOR: u32 = 3;

// Per-track flags (`export_track_flags[i]`)
pub const DEPTH_FLAG_VALID: u8 = 1 << 0;
pub const DEPTH_FLAG_METRIC: u8 = 1 << 1;
pub const DEPTH_FLAG_TTC_VALID: u8 = 1 << 2;
pub const DEPTH_FLAG_MOVING_OUTLIER: u8 = 1 << 3;
#[allow(dead_code)]
pub const DEPTH_FLAG_EXTERNAL_PRIOR: u8 = 1 << 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TtcTrackFit {
    pub t0: f64,
    pub sum_dt: f64,
    pub sum_dt2: f64,
    pub sum_invr: f64,
    pub sum_invr2: f64,
    pub sum_dt_invr: f64,
    pub tid: u32,
    pub n: u32,
    pub last_seen_frame: u32,
    pub foe_x: f32,
    pub foe_y: f32,
    pub last_r2: f32,
}

impl TtcTrackFit {
    pub const fn zeroed() -> Self {
        Self {
            t0: 0.0,
            sum_dt: 0.0,
            sum_dt2: 0.0,
            sum_invr: 0.0,
            sum_invr2: 0.0,
            sum_dt_invr: 0.0,
            tid: 0,
            n: 0,
            last_seen_frame: 0,
            foe_x: 0.0,
            foe_y: 0.0,
            last_r2: 0.0,
        }
    }
}

#[repr(C, align(16))]
pub struct DepthState {
    // Exported read-only buffers
    pub telem: [u32; DEPTH_TELEM_LEN],
    pub export_field: [f32; DEPTH_CELLS * DEPTH_FIELD_STRIDE],
    pub export_track_depth: [f32; MAX_MICRO * DEPTH_TRACK_STRIDE],
    pub export_track_flags: [u8; MAX_MICRO],
    pub export_hero_depth: [f32; MAX_TARGETS * DEPTH_HERO_STRIDE],

    // Host-writable external depth staging buffer (Option C: <= 128x64 f32, 32 KiB)
    pub ext_depth: [f32; EXT_DEPTH_MAX_CELLS],

    // Per-track 8-frame sliding OLS window + birth anchor (Option A, 129 KiB)
    pub hist_x: [[u16; DEPTH_HIST_LEN]; MAX_MICRO],
    pub hist_y: [[u16; DEPTH_HIST_LEN]; MAX_MICRO],
    pub hist_tid: [u32; MAX_MICRO],
    pub hist_len: [u8; MAX_MICRO],
    pub anc_x_q5: [u16; MAX_MICRO],
    pub anc_y_q5: [u16; MAX_MICRO],
    pub anc_age: [u16; MAX_MICRO],

    // Multi-frame f64 1/r(t) TTC linear fits for top approaching tracks (4.5 KiB)
    pub ttc_fits: [TtcTrackFit; MAX_TTC_FITS],

    // Global ring timestamps for the 8-frame OLS window
    pub hist_t_sec: [f64; DEPTH_HIST_LEN],

    // Compact per-frame scratch (zero heap, zero large stack frames; ~15 KiB)
    pub cand_slot: [u16; MAX_SOLVE_TRACKS],
    pub cand_x: [f32; MAX_SOLVE_TRACKS],
    pub cand_y: [f32; MAX_SOLVE_TRACKS],
    pub cand_vx: [f32; MAX_SOLVE_TRACKS],
    pub cand_vy: [f32; MAX_SOLVE_TRACKS],
    pub cand_w: [f32; MAX_SOLVE_TRACKS],
    pub scratch_sort: [f32; MAX_SOLVE_TRACKS],
    pub line_nx: [f32; MAX_FOE_LINES],
    pub line_ny: [f32; MAX_FOE_LINES],
    pub line_c: [f32; MAX_FOE_LINES],
    pub line_disp: [f32; MAX_FOE_LINES],
    pub line_slot: [u16; MAX_FOE_LINES],

    // Control & calibration parameters (strictly separated mono speed vs stereo baseline!)
    pub mode: u32,
    pub calib_f_px: f32,
    pub calib_speed_m_s: f32,
    pub calib_baseline_m: f32,
    pub stereo_max_disp_px: f32,
    pub stereo_sign: i32,
    pub ext_w: u32,
    pub ext_h: u32,
    pub ext_f_px: f32,
    pub ext_prov: u32,
    pub ext_seq: u32,
    pub ext_epoch: u32,
    pub ext_frame_idx: u32,
    pub ext_valid: u8,
    pub hist_head: u8,
    pub hist_valid_frames: u8,
    pub foe_valid: u8,
    pub parallax_locked: u8,
    pub stereo_touched_bos: u8,
    pub _pad_ctrl: [u8; 2],
    pub foe_x_ema: f32,
    pub foe_y_ema: f32,
    pub last_frame_idx: u32,
    pub last_continuity_epoch: u32,
    pub last_frame_w: u16,
    pub last_frame_h: u16,
    pub sim_time_s: f64,
}

const _: () = assert!(core::mem::size_of::<DepthState>() <= 240_000);

#[inline(always)]
fn quant_q5(v: f32) -> u16 {
    let q = (v * DEPTH_Q5_SCALE + 0.5).clamp(0.0, 65535.0);
    q as u16
}

#[inline(always)]
fn dequant_q5(q: u16) -> f32 {
    (q as f32) * DEPTH_Q5_INV
}

#[inline(never)]
fn median_in_place(buf: &mut [f32], n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let n = n.min(buf.len());
    let mut gap = n >> 1;
    while gap > 0 {
        let mut i = gap;
        while i < n {
            let key = buf[i];
            let mut j = i;
            while j >= gap && buf[j - gap] > key {
                buf[j] = buf[j - gap];
                j -= gap;
            }
            buf[j] = key;
            i += 1;
        }
        gap >>= 1;
    }
    let mid = n >> 1;
    if (n & 1) == 1 {
        buf[mid]
    } else {
        0.5 * (buf[mid - 1] + buf[mid])
    }
}

/// Deterministic `4x4` linear system solver with partial pivoting (`a * x = b`).
#[inline(never)]
fn solve_4x4_f64(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    let mut col = 0usize;
    while col < 4 {
        let mut piv = col;
        let mut best = a[col][col].abs();
        let mut r = col + 1;
        while r < 4 {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
            r += 1;
        }
        if !best.is_finite() || best < 1e-12 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let inv_diag = 1.0 / a[col][col];
        let mut row = col + 1;
        while row < 4 {
            let f = a[row][col] * inv_diag;
            a[row][col] = 0.0;
            let mut c = col + 1;
            while c < 4 {
                a[row][c] -= f * a[col][c];
                c += 1;
            }
            b[row] -= f * b[col];
            row += 1;
        }
        col += 1;
    }
    let mut x = [0.0f64; 4];
    let mut r = 4usize;
    while r > 0 {
        r -= 1;
        let mut sum = b[r];
        let mut c = r + 1;
        while c < 4 {
            sum -= a[r][c] * x[c];
            c += 1;
        }
        let v = sum / a[r][r];
        if !v.is_finite() {
            return None;
        }
        x[r] = v;
    }
    Some(x)
}

#[inline(never)]
fn solve_5x5_f64(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> Option<[f64; 5]> {
    let mut col = 0usize;
    while col < 5 {
        let mut piv = col;
        let mut best = a[col][col].abs();
        let mut r = col + 1;
        while r < 5 {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
            r += 1;
        }
        if !best.is_finite() || best < 1e-12 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let inv_diag = 1.0 / a[col][col];
        let mut row = col + 1;
        while row < 5 {
            let f = a[row][col] * inv_diag;
            a[row][col] = 0.0;
            let mut c = col + 1;
            while c < 5 {
                a[row][c] -= f * a[col][c];
                c += 1;
            }
            b[row] -= f * b[col];
            row += 1;
        }
        col += 1;
    }
    let mut x = [0.0f64; 5];
    let mut r = 5usize;
    while r > 0 {
        r -= 1;
        let mut sum = b[r];
        let mut c = r + 1;
        while c < 5 {
            sum -= a[r][c] * x[c];
            c += 1;
        }
        let v = sum / a[r][r];
        if !v.is_finite() {
            return None;
        }
        x[r] = v;
    }
    Some(x)
}

impl DepthState {
    #[inline(never)]
    pub fn reset_all(&mut self) {
        self.mode = DEPTH_MODE_OFF;
        self.calib_f_px = 0.0;
        self.calib_speed_m_s = 0.0;
        self.calib_baseline_m = 0.0;
        self.stereo_max_disp_px = 32.0;
        self.stereo_sign = 0;
        self.ext_w = 0;
        self.ext_h = 0;
        self.ext_f_px = 0.0;
        self.ext_prov = 0;
        self.ext_seq = 0;
        self.ext_epoch = 0;
        self.ext_frame_idx = u32::MAX;
        self.ext_valid = 0;
        self.stereo_touched_bos = 0;
        self.ext_depth.fill(0.0);
        self.reset_temporal_keep_config();
    }

    #[inline(never)]
    pub fn reset_temporal_keep_config(&mut self) {
        self.hist_tid.fill(0);
        self.hist_len.fill(0);
        self.anc_age.fill(0);
        let mut k = 0usize;
        while k < MAX_TTC_FITS {
            self.ttc_fits[k] = TtcTrackFit::zeroed();
            k += 1;
        }
        self.hist_t_sec.fill(0.0);
        self.hist_head = 0;
        self.hist_valid_frames = 0;
        self.foe_valid = 0;
        self.parallax_locked = 0;
        self.foe_x_ema = 0.0;
        self.foe_y_ema = 0.0;
        self.last_frame_idx = u32::MAX;
        self.last_frame_w = 0;
        self.last_frame_h = 0;
        self.sim_time_s = 0.0;
        // Never allow stale external planes to survive reset_tracks / resolution change / cut
        self.ext_valid = 0;
        self.ext_frame_idx = u32::MAX;
        self.clear_outputs();
    }

    #[inline(never)]
    pub fn clear_outputs(&mut self) {
        self.export_field.fill(0.0);
        self.export_track_depth.fill(0.0);
        self.export_track_flags.fill(0);
        self.export_hero_depth.fill(0.0);
        self.telem.fill(0);
        self.telem[0] = 1;
        self.telem[1] = self.mode;
        self.telem[2] = if self.mode == DEPTH_MODE_OFF {
            DEPTH_STATE_OFF
        } else {
            DEPTH_STATE_WARMING
        };
        self.telem[26] = self.calib_f_px.to_bits();
        self.telem[27] = if self.mode == DEPTH_MODE_STEREO_BOS {
            self.calib_baseline_m.to_bits()
        } else {
            self.calib_speed_m_s.to_bits()
        };
    }

    pub fn set_mode(&mut self, mode: u32) -> u32 {
        if mode > DEPTH_MODE_AUTO_FUSED {
            return 0;
        }
        let changed = self.mode != mode;
        self.mode = mode;
        if changed {
            self.reset_temporal_keep_config();
        } else {
            self.telem[0] = 1;
            self.telem[1] = mode;
        }
        1
    }

    pub fn get_mode(&self) -> u32 {
        self.mode
    }

    pub fn set_calib(
        &mut self,
        f_px: f32,
        speed_m_s: f32,
        baseline_m: f32,
        max_disp_px: f32,
        stereo_sign: i32,
    ) -> u32 {
        let f_ok = f_px.is_finite() && (f_px == 0.0 || (f_px >= 10.0 && f_px <= 1.0e5));
        let spd_ok =
            speed_m_s.is_finite() && (speed_m_s == 0.0 || (speed_m_s >= 1e-4 && speed_m_s <= 1.0e5));
        let base_ok = baseline_m.is_finite()
            && (baseline_m == 0.0 || (baseline_m >= 1e-4 && baseline_m <= 1.0e4));
        let disp_ok = max_disp_px.is_finite() && max_disp_px >= 1.0 && max_disp_px <= 64.0;
        if !f_ok || !spd_ok || !base_ok || !disp_ok || stereo_sign < -1 || stereo_sign > 1 {
            return 0;
        }
        self.calib_f_px = f_px;
        self.calib_speed_m_s = speed_m_s;
        self.calib_baseline_m = baseline_m;
        self.stereo_max_disp_px = max_disp_px;
        self.stereo_sign = stereo_sign;
        self.telem[26] = self.calib_f_px.to_bits();
        self.telem[27] = if self.mode == DEPTH_MODE_STEREO_BOS {
            self.calib_baseline_m.to_bits()
        } else {
            self.calib_speed_m_s.to_bits()
        };
        1
    }

    /// Stage external depth-plane metadata bound strictly to `(frame_idx, continuity_epoch)`.
    /// On ANY invalid argument, clears `ext_valid = 0` so mutated staging bytes can NEVER inherit
    /// a previously accepted plane's provenance or sequence number.
    pub fn set_external_meta(
        &mut self,
        w: u32,
        h: u32,
        f_px: f32,
        provenance: u32,
        seq: u32,
        frame_idx: u32,
        continuity_epoch: u32,
    ) -> u32 {
        if w < 4
            || h < 4
            || w > (EXT_DEPTH_MAX_W as u32)
            || h > (EXT_DEPTH_MAX_H as u32)
            || (w as usize).saturating_mul(h as usize) > EXT_DEPTH_MAX_CELLS
            || !f_px.is_finite()
            || f_px < 0.0
            || f_px > 1.0e5
            || provenance < EXT_PROV_SURVEYED_LIDAR
            || provenance > EXT_PROV_ASSUMED_NEURAL_PRIOR
        {
            self.ext_valid = 0;
            self.ext_frame_idx = u32::MAX;
            return 0;
        }
        self.ext_w = w;
        self.ext_h = h;
        self.ext_f_px = f_px;
        self.ext_prov = provenance;
        self.ext_seq = seq;
        self.ext_epoch = continuity_epoch;
        self.ext_frame_idx = frame_idx;
        self.ext_valid = 1;
        1
    }

    pub fn telemetry_ptr(&self) -> *const u32 {
        self.telem.as_ptr()
    }

    pub fn field_ptr(&self) -> *const f32 {
        self.export_field.as_ptr()
    }

    pub fn tracks_ptr(&self) -> *const f32 {
        self.export_track_depth.as_ptr()
    }

    pub fn track_flags_ptr(&self) -> *const u8 {
        self.export_track_flags.as_ptr()
    }

    pub fn hero_depth_ptr(&self) -> *const f32 {
        self.export_hero_depth.as_ptr()
    }

    pub fn ingest_mut_ptr(&mut self) -> *mut f32 {
        self.ext_depth.as_mut_ptr()
    }

    /// Populate per-Hero depth & TTC export (`export_hero_depth[m*4..+4]`) using foreground-preserving
    /// median/upper-quartile `1/Z` sampling (avoids silhouette blending with background) and nearest
    /// valid TTC micro-track within `16 px`.
    #[inline(never)]
    fn update_hero_depth_from_field(
        &mut self,
        ms: &MicroSwarmState,
        hero_pos_active: &[(bool, f32, f32); MAX_TARGETS],
        frame_w: usize,
        frame_h: usize,
        provenance: u32,
        scale_is_metric: bool,
    ) {
        self.export_hero_depth.fill(0.0);
        let mut mask = 0u32;
        if frame_w < 16 || frame_h < 16 {
            self.telem[14] = 0;
            return;
        }
        let sx = (DEPTH_GRID_W as f32) / (frame_w as f32);
        let sy = (DEPTH_GRID_H as f32) / (frame_h as f32);
        let mut m = 0usize;
        while m < MAX_TARGETS {
            let (act, hx, hy) = hero_pos_active[m];
            if act && hx.is_finite() && hy.is_finite() {
                let gx = (wasm_floorf(hx * sx) as isize).clamp(0, (DEPTH_GRID_W - 1) as isize) as usize;
                let gy = (wasm_floorf(hy * sy) as isize).clamp(0, (DEPTH_GRID_H - 1) as isize) as usize;

                // Prefer exact center cell when strongly supported; otherwise foreground-biased 3x3 robust sample
                let c_base = (gy * DEPTH_GRID_W + gx) * DEPTH_FIELD_STRIDE;
                let c_iz = self.export_field[c_base];
                let c_cf = self.export_field[c_base + 1];
                let mut inv_z_pick = 0.0f32;
                if c_iz.is_finite() && c_iz > 1e-6 && c_cf.is_finite() && c_cf >= 0.25 {
                    inv_z_pick = c_iz;
                } else {
                    let mut neigh = [0.0f32; 9];
                    let mut nn = 0usize;
                    let y0 = gy.saturating_sub(1);
                    let y1 = (gy + 1).min(DEPTH_GRID_H - 1);
                    let x0 = gx.saturating_sub(1);
                    let x1 = (gx + 1).min(DEPTH_GRID_W - 1);
                    let mut yy = y0;
                    while yy <= y1 {
                        let mut xx = x0;
                        while xx <= x1 {
                            let base = (yy * DEPTH_GRID_W + xx) * DEPTH_FIELD_STRIDE;
                            let iz = self.export_field[base];
                            let cf = self.export_field[base + 1];
                            if iz.is_finite() && iz > 1e-6 && cf.is_finite() && cf > 0.0 && nn < 9 {
                                neigh[nn] = iz;
                                nn += 1;
                            }
                            xx += 1;
                        }
                        yy += 1;
                    }
                    if nn > 0 {
                        // Foreground-preserving upper-median in inverse-depth (larger 1/Z = closer object)
                        let _ = median_in_place(&mut neigh, nn);
                        let fg_idx = (nn * 2) / 3;
                        inv_z_pick = neigh[fg_idx.min(nn - 1)];
                    }
                }

                // Nearest valid micro-track TTC within 16 px
                let mut best_ttc = 0.0f32;
                let mut best_d2 = 16.0f32 * 16.0f32;
                if self.telem[7] > 0 {
                    let mut ti = 0usize;
                    while ti < MAX_MICRO {
                        if (self.export_track_flags[ti] & DEPTH_FLAG_TTC_VALID) != 0 {
                            let dx = ms.tracks.pos_x[ti] - hx;
                            let dy = ms.tracks.pos_y[ti] - hy;
                            let d2 = dx * dx + dy * dy;
                            if d2 < best_d2 {
                                let ttc = self.export_track_depth[ti * DEPTH_TRACK_STRIDE + 1];
                                if ttc.is_finite() && ttc > 0.0 {
                                    best_d2 = d2;
                                    best_ttc = ttc;
                                }
                            }
                        }
                        ti += 1;
                    }
                }

                if inv_z_pick > 1e-6 || best_ttc > 0.0 {
                    let hb = m * DEPTH_HERO_STRIDE;
                    if inv_z_pick > 1e-6 {
                        let z_val = 1.0 / inv_z_pick;
                        if z_val.is_finite() && z_val > 0.0 {
                            self.export_hero_depth[hb] = z_val;
                            self.export_hero_depth[hb + 1] = inv_z_pick;
                        }
                    }
                    self.export_hero_depth[hb + 2] = best_ttc;
                    self.export_hero_depth[hb + 3] = if inv_z_pick > 1e-6 {
                        if scale_is_metric {
                            provenance as f32
                        } else {
                            DEPTH_PROV_MONO_UP_TO_SCALE as f32
                        }
                    } else {
                        DEPTH_PROV_INTRINSICS_FREE_TTC as f32
                    };
                    mask |= 1u32 << m;
                }
            }
            m += 1;
        }
        self.telem[14] = mask;
    }

    /// Option C: Consume-once, epoch-bound bilinear inverse-depth resample of host-ingested `ext_depth`
    /// (`w x h`, positive metric Z in meters) into `export_field` (`64x36`) with strict provenance firewall.
    #[inline(never)]
    fn step_external(
        &mut self,
        ms: &MicroSwarmState,
        hero_pos_active: &[(bool, f32, f32); MAX_TARGETS],
        frame_idx: u32,
        frame_w: usize,
        frame_h: usize,
        continuity_epoch: u32,
    ) -> bool {
        self.export_field.fill(0.0);
        self.export_track_depth.fill(0.0);
        self.export_track_flags.fill(0);
        self.export_hero_depth.fill(0.0);
        self.telem.fill(0);
        self.telem[0] = 1;
        self.telem[1] = self.mode;
        self.telem[15] = continuity_epoch;

        let fresh = self.ext_valid != 0
            && self.ext_w >= 4
            && self.ext_h >= 4
            && self.ext_epoch == continuity_epoch
            && self.ext_frame_idx == frame_idx;
        // Consume immediately so a missing/rejected next-frame commit fails closed with EXT_STALE
        self.ext_valid = 0;
        self.ext_frame_idx = u32::MAX;

        if !fresh {
            self.telem[2] = DEPTH_STATE_WARMING;
            self.telem[3] = DEPTH_REASON_EXT_STALE;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        let ew = self.ext_w as usize;
        let eh = self.ext_h as usize;
        let prov = match self.ext_prov {
            EXT_PROV_SURVEYED_LIDAR => DEPTH_PROV_SURVEYED_LIDAR,
            EXT_PROV_SURVEYED_STEREO => DEPTH_PROV_SURVEYED_STEREO,
            _ => DEPTH_PROV_ASSUMED_NEURAL_PRIOR,
        };
        let mut reason = 0u32;
        if prov == DEPTH_PROV_ASSUMED_NEURAL_PRIOR {
            reason |= DEPTH_REASON_ASSUMED_NEURAL_PRIOR;
        }

        let mut valid_cells = 0u32;
        let mut z_min = f32::INFINITY;
        let mut z_max = 0.0f32;
        let sx = (ew as f32) / (DEPTH_GRID_W as f32);
        let sy = (eh as f32) / (DEPTH_GRID_H as f32);

        let mut gy = 0usize;
        while gy < DEPTH_GRID_H {
            let fy_c = ((gy as f32 + 0.5) * sy - 0.5).clamp(0.0, (eh - 1) as f32);
            let y0 = wasm_floorf(fy_c) as usize;
            let y1 = (y0 + 1).min(eh - 1);
            let wy = fy_c - (y0 as f32);
            let mut gx = 0usize;
            while gx < DEPTH_GRID_W {
                let fx_c = ((gx as f32 + 0.5) * sx - 0.5).clamp(0.0, (ew - 1) as f32);
                let x0 = wasm_floorf(fx_c) as usize;
                let x1 = (x0 + 1).min(ew - 1);
                let wx = fx_c - (x0 as f32);

                let z00 = self.ext_depth[y0 * ew + x0];
                let z10 = self.ext_depth[y0 * ew + x1];
                let z01 = self.ext_depth[y1 * ew + x0];
                let z11 = self.ext_depth[y1 * ew + x1];

                let mut w_tot = 0.0f32;
                let mut iz_tot = 0.0f32;
                let samples = [
                    (z00, (1.0 - wx) * (1.0 - wy)),
                    (z10, wx * (1.0 - wy)),
                    (z01, (1.0 - wx) * wy),
                    (z11, wx * wy),
                ];
                let mut s_idx = 0usize;
                while s_idx < 4 {
                    let (zv, w) = samples[s_idx];
                    if zv.is_finite() && zv > 0.02 && zv < 1.0e5 {
                        iz_tot += w / zv;
                        w_tot += w;
                    }
                    s_idx += 1;
                }
                if w_tot >= 0.5 {
                    let inv_z = iz_tot / w_tot;
                    if inv_z.is_finite() && inv_z > 1e-6 {
                        let z_val = 1.0 / inv_z;
                        let base = (gy * DEPTH_GRID_W + gx) * DEPTH_FIELD_STRIDE;
                        self.export_field[base] = inv_z;
                        self.export_field[base + 1] = w_tot;
                        valid_cells += 1;
                        if z_val < z_min {
                            z_min = z_val;
                        }
                        if z_val > z_max {
                            z_max = z_val;
                        }
                    }
                }
                gx += 1;
            }
            gy += 1;
        }

        self.telem[2] = if valid_cells > 0 {
            DEPTH_STATE_READY_EXTERNAL
        } else {
            DEPTH_STATE_WARMING
        };
        self.telem[3] = if valid_cells > 0 {
            reason
        } else {
            reason | DEPTH_REASON_EXT_STALE
        };
        self.telem[4] = if valid_cells > 0 { prov } else { DEPTH_PROV_NONE };
        self.telem[9] = valid_cells;
        self.telem[10] = self.ext_w;
        self.telem[11] = self.ext_h;
        self.telem[12] = self.ext_seq;
        self.telem[26] = if self.ext_f_px > 0.0 {
            self.ext_f_px.to_bits()
        } else {
            self.calib_f_px.to_bits()
        };
        self.telem[27] = self.calib_speed_m_s.to_bits();
        self.telem[30] = if valid_cells > 0 && z_min.is_finite() {
            z_min.to_bits()
        } else {
            0
        };
        self.telem[31] = if valid_cells > 0 && z_max.is_finite() {
            z_max.to_bits()
        } else {
            0
        };

        if valid_cells > 0 {
            self.update_hero_depth_from_field(ms, hero_pos_active, frame_w, frame_h, prov, true);
        }
        valid_cells > 0
    }

    /// Option B: Run `256x128` calibrated stereo disparity on `BosState` (explicitly paired Left ref
    /// ONLY — never auto-captures mono video frames!) and reproject valid inverse depth into
    /// Right-camera `64x36` `export_field` coordinates with foreground z-buffer priority.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn step_stereo_bridge(
        &mut self,
        ms: &MicroSwarmState,
        bos: &mut BosState,
        bos_cache: &mut BosRefCache,
        bos_pyr: &mut BosPyrScratch,
        cur_luma: &[f32],
        cur_pyr: &[f32],
        hero_pos_active: &[(bool, f32, f32); MAX_TARGETS],
        frame_idx: u32,
        frame_w: usize,
        frame_h: usize,
        continuity_epoch: u32,
    ) -> bool {
        self.export_field.fill(0.0);
        self.export_track_depth.fill(0.0);
        self.export_track_flags.fill(0);
        self.export_hero_depth.fill(0.0);
        self.telem.fill(0);
        self.telem[0] = 1;
        self.telem[1] = self.mode;
        self.telem[15] = continuity_epoch;
        self.telem[26] = self.calib_f_px.to_bits();
        self.telem[27] = self.calib_baseline_m.to_bits();

        if self.calib_f_px < 10.0 || self.calib_baseline_m < 1e-4 {
            self.telem[2] = DEPTH_STATE_WARMING;
            self.telem[3] = DEPTH_REASON_NO_INTRINSICS;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }
        if !bos.has_fresh_stereo_ref(frame_w, frame_h, frame_idx, continuity_epoch) {
            bos.invalidate_stereo_ref();
            self.telem[2] = DEPTH_STATE_WARMING;
            self.telem[3] = DEPTH_REASON_STEREO_NOT_READY;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        self.stereo_touched_bos = 1;
        let (valid_nodes, epi_rms, med_z, min_z, max_z) = bos.step_stereo(
            bos_cache,
            bos_pyr,
            cur_luma,
            cur_pyr,
            frame_w,
            frame_h,
            self.calib_f_px,
            self.calib_baseline_m,
            self.stereo_max_disp_px,
            self.stereo_sign,
        );

        // Enforce minimum spatial coverage (>= 512 nodes, ~1.5% of 256x128) so sparse wrong-sign /
        // out-of-range noise matches never report READY_STEREO.
        if valid_nodes < 512 {
            self.telem[2] = DEPTH_STATE_WARMING;
            self.telem[3] = DEPTH_REASON_STEREO_NOT_READY;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        // Reproject 256x128 Left-reference stereo depth into Right-image 64x36 `export_field`
        // with foreground-wins visibility (`max inv_z`) so Right-frame Hero-16 & HUD align exactly!
        let hx = (frame_w as f32) * (1.0 / (BOS_GRID_W as f32));
        let hy = (frame_h as f32) * (1.0 / (BOS_GRID_H as f32));
        let sx_grid = (DEPTH_GRID_W as f32) / (frame_w as f32);
        let sy_grid = (DEPTH_GRID_H as f32) / (frame_h as f32);

        // Pass 1: find closest (maximum) inverse-depth in each Right-frame 64x36 cell
        let mut bj = 0usize;
        while bj < BOS_GRID_H {
            let cy_l = ((bj as f32) + 0.5) * hy - 0.5;
            let gy = (wasm_floorf(cy_l * sy_grid) as isize).clamp(0, (DEPTH_GRID_H - 1) as isize) as usize;
            let mut bi = 0usize;
            while bi < BOS_GRID_W {
                let u = bj * BOS_GRID_W + bi;
                if (bos.valid_mask[u] & BOS_VALID_PHI) != 0 {
                    let iz = bos.planes[BOS_PLANE_CURL * BOS_NODES + u];
                    let z = bos.planes[BOS_PLANE_PHI * BOS_NODES + u];
                    let ex = bos.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                    if iz.is_finite() && iz > 1e-6 && z.is_finite() && z > 0.0 && ex.is_finite() {
                        let cx_r = ((bi as f32) + 0.5) * hx - 0.5 + ex;
                        if cx_r >= 0.0 && cx_r < (frame_w as f32) {
                            let gx = (wasm_floorf(cx_r * sx_grid) as isize)
                                .clamp(0, (DEPTH_GRID_W - 1) as isize) as usize;
                            let base = (gy * DEPTH_GRID_W + gx) * DEPTH_FIELD_STRIDE;
                            if iz > self.export_field[base] {
                                self.export_field[base] = iz;
                            }
                        }
                    }
                }
                bi += 1;
            }
            bj += 1;
        }

        // Pass 2: average foreground-consistent samples (within 20% of cell max inv_z) into clean mean
        // Temporarily store sum_iz in `base` only after reading max_iz; wait — we only need confidence
        // count in `base + 1`, and keeping the foreground max-consistent mean via a clean 2-slot trick:
        // keep foreground `max_iz` directly (already denoised by 8x8 patch!) and accumulate support count in `base + 1`.
        bj = 0;
        while bj < BOS_GRID_H {
            let cy_l = ((bj as f32) + 0.5) * hy - 0.5;
            let gy = (wasm_floorf(cy_l * sy_grid) as isize).clamp(0, (DEPTH_GRID_H - 1) as isize) as usize;
            let mut bi = 0usize;
            while bi < BOS_GRID_W {
                let u = bj * BOS_GRID_W + bi;
                if (bos.valid_mask[u] & BOS_VALID_PHI) != 0 {
                    let iz = bos.planes[BOS_PLANE_CURL * BOS_NODES + u];
                    let ex = bos.planes[BOS_PLANE_EPS_X * BOS_NODES + u];
                    if iz.is_finite() && iz > 1e-6 && ex.is_finite() {
                        let cx_r = ((bi as f32) + 0.5) * hx - 0.5 + ex;
                        if cx_r >= 0.0 && cx_r < (frame_w as f32) {
                            let gx = (wasm_floorf(cx_r * sx_grid) as isize)
                                .clamp(0, (DEPTH_GRID_W - 1) as isize) as usize;
                            let base = (gy * DEPTH_GRID_W + gx) * DEPTH_FIELD_STRIDE;
                            let max_iz = self.export_field[base];
                            if iz >= 0.80 * max_iz {
                                self.export_field[base + 1] += 1.0;
                            }
                        }
                    }
                }
                bi += 1;
            }
            bj += 1;
        }

        let mut valid_cells = 0u32;
        let mut c = 0usize;
        while c < DEPTH_CELLS {
            let base = c * DEPTH_FIELD_STRIDE;
            let cnt = self.export_field[base + 1];
            if cnt > 0.0 && self.export_field[base] > 1e-6 {
                self.export_field[base + 1] = (cnt * (1.0 / 16.0)).min(1.0);
                valid_cells += 1;
            } else {
                self.export_field[base] = 0.0;
                self.export_field[base + 1] = 0.0;
            }
            c += 1;
        }

        // Fail closed if extreme calibration or sparse noise produced < 32 representable field cells
        if valid_cells < 32 {
            self.export_field.fill(0.0);
            self.telem[2] = DEPTH_STATE_WARMING;
            self.telem[3] = DEPTH_REASON_STEREO_NOT_READY;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        self.telem[2] = DEPTH_STATE_READY_STEREO;
        self.telem[3] = 0;
        self.telem[4] = DEPTH_PROV_SURVEYED_STEREO;
        self.telem[9] = valid_cells;
        self.telem[13] = valid_nodes;
        self.telem[28] = epi_rms.to_bits();
        self.telem[29] = med_z.to_bits();
        self.telem[30] = min_z.to_bits();
        self.telem[31] = max_z.to_bits();

        self.update_hero_depth_from_field(
            ms,
            hero_pos_active,
            frame_w,
            frame_h,
            DEPTH_PROV_SURVEYED_STEREO,
            true,
        );
        true
    }

    /// Option A: Monocular Motion Parallax + Fail-Closed 3-DoF Pure-Rotation Gate + Cross-Flow
    /// Mixed-Motion Rejection + Intrinsics-Free RANSAC FoE & `f64` `1/r(t)` Optical Looming TTC.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn step_mono_parallax_ttc(
        &mut self,
        ms: &MicroSwarmState,
        hero_pos_active: &[(bool, f32, f32); MAX_TARGETS],
        frame_idx: u32,
        frame_w: usize,
        frame_h: usize,
        actual_dt_sec: f32,
        frame_t_sec: f64,
        continuity_epoch: u32,
        scene_cut: bool,
    ) -> bool {
        let geom_changed = self.last_frame_w != 0
            && (self.last_frame_w as usize != frame_w || self.last_frame_h as usize != frame_h);
        // Require strict 1-frame contiguity for the mono sliding window so AUTO_FUSED interleaving
        // with Stereo/External cannot compress time deltas and shrink metric depth!
        if scene_cut
            || geom_changed
            || continuity_epoch != self.last_continuity_epoch
            || (self.last_frame_idx != u32::MAX
                && frame_idx != self.last_frame_idx.wrapping_add(1))
        {
            self.reset_temporal_keep_config();
            self.last_continuity_epoch = continuity_epoch;
        }
        self.last_frame_w = frame_w as u16;
        self.last_frame_h = frame_h as u16;

        // Require true elapsed dt inside supported physical range [2 ms, 1.0 s]; never bias with 150 ms clamp!
        if !actual_dt_sec.is_finite() || actual_dt_sec < 0.002 || actual_dt_sec > 1.0 {
            self.reset_temporal_keep_config();
            self.telem[3] = DEPTH_REASON_BAD_DT;
            self.telem[15] = continuity_epoch;
            return false;
        }

        if ms.swarm_active == 0 || frame_w < 32 || frame_h < 32 {
            self.clear_outputs();
            self.telem[3] = DEPTH_REASON_LOW_TRACKS;
            self.telem[15] = continuity_epoch;
            return false;
        }

        // Use the engine's authoritative f64 presentation clock (`hist_t_sec`) directly
        if frame_t_sec.is_finite() && (self.hist_valid_frames == 0 || frame_t_sec > self.sim_time_s) {
            self.sim_time_s = frame_t_sec;
        } else {
            self.sim_time_s += actual_dt_sec as f64;
        }
        self.last_frame_idx = frame_idx;
        let head = (self.hist_head as usize + 1) & (DEPTH_HIST_LEN - 1);
        self.hist_head = head as u8;
        self.hist_t_sec[head] = self.sim_time_s;
        if (self.hist_valid_frames as usize) < DEPTH_HIST_LEN {
            self.hist_valid_frames += 1;
        }

        let res_scale = ((frame_w as f32) * (1.0 / 640.0)).clamp(0.5, 2.0);
        let chord_min_disp = 20.0 * res_scale;
        let chord_min_spd = 5.0 * res_scale;

        // 1. Update contiguous 8-frame Q5 ring + birth anchors and gather spatially-spread OLS candidates
        let mut n_cand = 0usize;
        let mut n_lines = 0usize;

        let mut i = 0usize;
        while i < MAX_MICRO {
            let st = ms.tracks.state[i];
            let tid = ms.tracks.track_id[i];
            let px = ms.tracks.pos_x[i];
            let py = ms.tracks.pos_y[i];
            let clean = st == MICRO_STATE_CONFIRMED
                && tid != 0
                && ms.tracks.coasting[i] == 0
                && ms.tracks.reacq_flag[i] == 0
                && px.is_finite()
                && py.is_finite()
                && px >= 0.0
                && py >= 0.0
                && px < (frame_w as f32)
                && py < (frame_h as f32);
            if !clean {
                self.hist_len[i] = 0;
                if self.hist_tid[i] != 0 && (st == 0 || tid != self.hist_tid[i]) {
                    self.hist_tid[i] = 0;
                    self.anc_age[i] = 0;
                }
                self.export_track_depth[i * DEPTH_TRACK_STRIDE] = 0.0;
                self.export_track_depth[i * DEPTH_TRACK_STRIDE + 1] = 0.0;
                self.export_track_flags[i] = 0;
                i += 1;
                continue;
            }

            let xq = quant_q5(px);
            let yq = quant_q5(py);
            if self.hist_tid[i] != tid {
                self.hist_tid[i] = tid;
                self.hist_len[i] = 1;
                self.anc_x_q5[i] = xq;
                self.anc_y_q5[i] = yq;
                self.anc_age[i] = 1;
                self.hist_x[i][head] = xq;
                self.hist_y[i][head] = yq;
            } else {
                self.hist_x[i][head] = xq;
                self.hist_y[i][head] = yq;
                if (self.hist_len[i] as usize) < DEPTH_HIST_LEN {
                    self.hist_len[i] += 1;
                }
                self.anc_age[i] = self.anc_age[i].saturating_add(1);
                // Staggered midpoint anchor refresh: prevents lockstep FoE/TTC dropout at frame 120
                // while keeping heading-change latency bounded and preserving >=50% chord length
                let refresh_lim = 160u16 + (((i as u16) & 31) * 4);
                if self.anc_age[i] > refresh_lim {
                    let old_ax = dequant_q5(self.anc_x_q5[i]);
                    let old_ay = dequant_q5(self.anc_y_q5[i]);
                    self.anc_x_q5[i] = quant_q5(0.5 * (old_ax + px));
                    self.anc_y_q5[i] = quant_q5(0.5 * (old_ay + py));
                    self.anc_age[i] = 60;
                }
            }

            if (self.hist_len[i] as usize) == DEPTH_HIST_LEN
                && (self.hist_valid_frames as usize) == DEPTH_HIST_LEN
            {
                let mut mt = 0.0f64;
                let mut mx = 0.0f64;
                let mut my = 0.0f64;
                let mut k = 0usize;
                while k < DEPTH_HIST_LEN {
                    let idx_k = (head + 1 + k) & (DEPTH_HIST_LEN - 1);
                    mt += self.hist_t_sec[idx_k];
                    mx += dequant_q5(self.hist_x[i][idx_k]) as f64;
                    my += dequant_q5(self.hist_y[i][idx_k]) as f64;
                    k += 1;
                }
                let inv_n = 1.0 / (DEPTH_HIST_LEN as f64);
                mt *= inv_n;
                mx *= inv_n;
                my *= inv_n;
                let mut stt = 0.0f64;
                let mut stx = 0.0f64;
                let mut sty = 0.0f64;
                k = 0;
                while k < DEPTH_HIST_LEN {
                    let idx_k = (head + 1 + k) & (DEPTH_HIST_LEN - 1);
                    let dt0 = self.hist_t_sec[idx_k] - mt;
                    let dx0 = (dequant_q5(self.hist_x[i][idx_k]) as f64) - mx;
                    let dy0 = (dequant_q5(self.hist_y[i][idx_k]) as f64) - my;
                    stt += dt0 * dt0;
                    stx += dt0 * dx0;
                    sty += dt0 * dy0;
                    k += 1;
                }
                if stt > 1e-6 && n_cand < MAX_SOLVE_TRACKS {
                    let vx_ols = stx / stt;
                    let vy_ols = sty / stt;
                    let mut rss = 0.0f64;
                    k = 0;
                    while k < DEPTH_HIST_LEN {
                        let idx_k = (head + 1 + k) & (DEPTH_HIST_LEN - 1);
                        let dt0 = self.hist_t_sec[idx_k] - mt;
                        let ex = (dequant_q5(self.hist_x[i][idx_k]) as f64) - (mx + vx_ols * dt0);
                        let ey = (dequant_q5(self.hist_y[i][idx_k]) as f64) - (my + vy_ols * dt0);
                        rss += ex * ex + ey * ey;
                        k += 1;
                    }
                    let rms_lim = (0.65 * res_scale) as f64;
                    if rss * inv_n <= rms_lim * rms_lim {
                        self.cand_slot[n_cand] = i as u16;
                        self.cand_x[n_cand] = mx as f32;
                        self.cand_y[n_cand] = my as f32;
                        self.cand_vx[n_cand] = vx_ols as f32;
                        self.cand_vy[n_cand] = vy_ols as f32;
                        self.cand_w[n_cand] = 1.0;
                        n_cand += 1;
                    }
                }
            }

            if self.anc_age[i] >= 12 {
                let ax = dequant_q5(self.anc_x_q5[i]);
                let ay = dequant_q5(self.anc_y_q5[i]);
                let dx = px - ax;
                let dy = py - ay;
                let disp = wasm_sqrtf(dx * dx + dy * dy);
                let spd = wasm_sqrtf(
                    ms.tracks.vel_x[i] * ms.tracks.vel_x[i] + ms.tracks.vel_y[i] * ms.tracks.vel_y[i],
                );
                if disp >= chord_min_disp && spd >= chord_min_spd {
                    let inv_len = 1.0 / disp;
                    let nx = dy * inv_len;
                    let ny = -dx * inv_len;
                    let c_val = nx * (0.5 * (ax + px)) + ny * (0.5 * (ay + py));
                    if n_lines < MAX_FOE_LINES {
                        self.line_nx[n_lines] = nx;
                        self.line_ny[n_lines] = ny;
                        self.line_c[n_lines] = c_val;
                        self.line_disp[n_lines] = disp;
                        self.line_slot[n_lines] = i as u16;
                        n_lines += 1;
                    } else {
                        let mut min_k = 0usize;
                        let mut min_d = self.line_disp[0];
                        let mut k = 1usize;
                        while k < MAX_FOE_LINES {
                            if self.line_disp[k] < min_d {
                                min_d = self.line_disp[k];
                                min_k = k;
                            }
                            k += 1;
                        }
                        if disp > min_d {
                            self.line_nx[min_k] = nx;
                            self.line_ny[min_k] = ny;
                            self.line_c[min_k] = c_val;
                            self.line_disp[min_k] = disp;
                            self.line_slot[min_k] = i as u16;
                        }
                    }
                }
            }
            i += 1;
        }

        // Clear per-frame export outputs before solving
        self.export_field.fill(0.0);
        self.export_track_depth.fill(0.0);
        self.export_track_flags.fill(0);
        self.export_hero_depth.fill(0.0);
        self.telem.fill(0);
        self.telem[0] = 1;
        self.telem[1] = self.mode;
        self.telem[15] = continuity_epoch;
        self.telem[26] = self.calib_f_px.to_bits();
        self.telem[27] = self.calib_speed_m_s.to_bits();

        if n_cand < 8 {
            self.parallax_locked = 0;
            if n_lines >= 4 {
                let (foe_ok, foe_x, foe_y, foe_inliers, ttc_valid_cnt, min_ttc, best_r2) =
                    self.solve_foe_and_ttc(ms, n_lines, frame_idx, frame_w, frame_h, res_scale);
                self.telem[7] = ttc_valid_cnt;
                self.telem[8] = foe_inliers;
                if foe_ok {
                    self.telem[22] = foe_x.to_bits();
                    self.telem[23] = foe_y.to_bits();
                }
                if ttc_valid_cnt > 0 {
                    self.telem[2] = DEPTH_STATE_READY_TTC_ONLY;
                    self.telem[3] = DEPTH_REASON_LOW_TRACKS;
                    self.telem[4] = DEPTH_PROV_INTRINSICS_FREE_TTC;
                    self.telem[24] = min_ttc.to_bits();
                    self.telem[25] = best_r2.to_bits();
                    self.update_hero_depth_from_field(
                        ms,
                        hero_pos_active,
                        frame_w,
                        frame_h,
                        DEPTH_PROV_INTRINSICS_FREE_TTC,
                        false,
                    );
                    return true;
                }
            }
            self.telem[2] = DEPTH_STATE_WARMING;
            self.telem[3] = DEPTH_REASON_LOW_TRACKS;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        // Compute median flow speed first
        let mut k = 0usize;
        while k < n_cand {
            let vx = self.cand_vx[k];
            let vy = self.cand_vy[k];
            self.scratch_sort[k] = wasm_sqrtf(vx * vx + vy * vy);
            k += 1;
        }
        let med_speed = median_in_place(&mut self.scratch_sort, n_cand);
        self.telem[18] = med_speed.to_bits();

        // 2. If focal length is uncalibrated (`f_px < 10`), pure-rotation verification is impossible:
        // fail closed on parallax depth and run ONLY intrinsics-free FoE/TTC optical looming!
        if self.calib_f_px < 10.0 {
            self.parallax_locked = 0;
            let (foe_ok, foe_x, foe_y, foe_inliers, ttc_valid_cnt, min_ttc, best_r2) =
                self.solve_foe_and_ttc(ms, n_lines, frame_idx, frame_w, frame_h, res_scale);
            self.telem[7] = ttc_valid_cnt;
            self.telem[8] = foe_inliers;
            if foe_ok {
                self.telem[22] = foe_x.to_bits();
                self.telem[23] = foe_y.to_bits();
            }
            if ttc_valid_cnt > 0 {
                self.telem[2] = DEPTH_STATE_READY_TTC_ONLY;
                self.telem[3] = DEPTH_REASON_NO_INTRINSICS;
                self.telem[4] = DEPTH_PROV_INTRINSICS_FREE_TTC;
                self.telem[24] = min_ttc.to_bits();
                self.telem[25] = best_r2.to_bits();
                self.update_hero_depth_from_field(
                    ms,
                    hero_pos_active,
                    frame_w,
                    frame_h,
                    DEPTH_PROV_INTRINSICS_FREE_TTC,
                    false,
                );
                return true;
            }
            self.telem[2] = if med_speed < 1.5 * res_scale {
                DEPTH_STATE_REFUSED_NO_BASELINE
            } else {
                DEPTH_STATE_WARMING
            };
            self.telem[3] = DEPTH_REASON_NO_INTRINSICS
                | if med_speed < 1.5 * res_scale {
                    DEPTH_REASON_NO_BASELINE
                } else {
                    0
                };
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        // 3. Fail-Closed 3-DoF Pure-Rotation Huber-IRLS Check with Relative Conditioning Guard
        let cx = (frame_w as f32) * 0.5;
        let cy = (frame_h as f32) * 0.5;
        let f_eff = self.calib_f_px;
        let inv_f = 1.0 / (f_eff as f64);
        let f64_eff = f_eff as f64;

        let mut omega = [0.0f64; 3];
        let mut rot_solve_ok = false;
        let mut it = 0usize;
        while it < 3 {
            let mut m = [0.0f64; 9];
            let mut rhs = [0.0f64; 3];
            k = 0;
            while k < n_cand {
                let w = self.cand_w[k] as f64;
                let xn = ((self.cand_x[k] - cx) as f64) * inv_f;
                let yn = ((self.cand_y[k] - cy) as f64) * inv_f;
                let au0 = f64_eff * xn * yn;
                let au1 = -f64_eff * (1.0 + xn * xn);
                let au2 = -f64_eff * yn;
                let av0 = f64_eff * (1.0 + yn * yn);
                let av1 = -f64_eff * xn * yn;
                let av2 = f64_eff * xn;
                let bu = self.cand_vx[k] as f64;
                let bv = self.cand_vy[k] as f64;

                m[0] += w * (au0 * au0 + av0 * av0);
                m[1] += w * (au0 * au1 + av0 * av1);
                m[2] += w * (au0 * au2 + av0 * av2);
                m[4] += w * (au1 * au1 + av1 * av1);
                m[5] += w * (au1 * au2 + av1 * av2);
                m[8] += w * (au2 * au2 + av2 * av2);
                rhs[0] += w * (au0 * bu + av0 * bv);
                rhs[1] += w * (au1 * bu + av1 * bv);
                rhs[2] += w * (au2 * bu + av2 * bv);
                k += 1;
            }
            m[3] = m[1];
            m[6] = m[2];
            m[7] = m[5];

            let det = m[0] * (m[4] * m[8] - m[5] * m[7])
                - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6]);
            let m_scale = (m[0] + m[4] + m[8]) * (1.0 / 3.0);
            if !det.is_finite()
                || !m_scale.is_finite()
                || m_scale < 1e-6
                || det.abs() < 1e-9 * m_scale * m_scale * m_scale
            {
                rot_solve_ok = false;
                break;
            }
            let det0 = rhs[0] * (m[4] * m[8] - m[5] * m[7])
                - m[1] * (rhs[1] * m[8] - m[5] * rhs[2])
                + m[2] * (rhs[1] * m[7] - m[4] * rhs[2]);
            let det1 = m[0] * (rhs[1] * m[8] - m[5] * rhs[2])
                - rhs[0] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * rhs[2] - rhs[1] * m[6]);
            let det2 = m[0] * (m[4] * rhs[2] - rhs[1] * m[7])
                - m[1] * (m[3] * rhs[2] - rhs[1] * m[6])
                + rhs[0] * (m[3] * m[7] - m[4] * m[6]);
            let inv_det = 1.0 / det;
            omega = [det0 * inv_det, det1 * inv_det, det2 * inv_det];
            rot_solve_ok = omega[0].is_finite() && omega[1].is_finite() && omega[2].is_finite();
            if !rot_solve_ok {
                break;
            }

            let huber_d = (1.5 * res_scale) as f32;
            k = 0;
            while k < n_cand {
                let xn = ((self.cand_x[k] - cx) as f64) * inv_f;
                let yn = ((self.cand_y[k] - cy) as f64) * inv_f;
                let pu = f64_eff * (xn * yn * omega[0] - (1.0 + xn * xn) * omega[1] - yn * omega[2]);
                let pv = f64_eff * ((1.0 + yn * yn) * omega[0] - xn * yn * omega[1] + xn * omega[2]);
                let du = (self.cand_vx[k] as f64) - pu;
                let dv = (self.cand_vy[k] as f64) - pv;
                let r = wasm_sqrt_f64(du * du + dv * dv) as f32;
                self.cand_w[k] = if r > huber_d { huber_d / r } else { 1.0 };
                k += 1;
            }
            it += 1;
        }

        // Fail closed if the 3x3 rotation solve was ill-conditioned!
        if !rot_solve_ok {
            self.parallax_locked = 0;
            self.telem[2] = DEPTH_STATE_REFUSED_NO_BASELINE;
            self.telem[3] = DEPTH_REASON_NO_BASELINE;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        k = 0;
        while k < n_cand {
            let xn = ((self.cand_x[k] - cx) as f64) * inv_f;
            let yn = ((self.cand_y[k] - cy) as f64) * inv_f;
            let pu = f64_eff * (xn * yn * omega[0] - (1.0 + xn * xn) * omega[1] - yn * omega[2]);
            let pv = f64_eff * ((1.0 + yn * yn) * omega[0] - xn * yn * omega[1] + xn * omega[2]);
            let du = (self.cand_vx[k] as f64) - pu;
            let dv = (self.cand_vy[k] as f64) - pv;
            self.scratch_sort[k] = wasm_sqrt_f64(du * du + dv * dv) as f32;
            k += 1;
        }
        let med_rot_resid = median_in_place(&mut self.scratch_sort, n_cand);
        let rot_ratio = med_rot_resid / med_speed.max(1e-3);

        self.telem[16] = med_rot_resid.to_bits();
        self.telem[17] = rot_ratio.to_bits();
        self.telem[19] = (omega[0] as f32).to_bits();
        self.telem[20] = (omega[1] as f32).to_bits();
        self.telem[21] = (omega[2] as f32).to_bits();

        let oldest_idx = (head + 1) & (DEPTH_HIST_LEN - 1);
        let win_dur_s = (self.hist_t_sec[head] - self.hist_t_sec[oldest_idx]) as f32;
        let rate_scale = (0.2333 / win_dur_s.max(0.05)).clamp(0.18, 2.0);
        let spd_floor = 1.5 * res_scale * rate_scale;
        let no_baseline = med_speed < spd_floor;

        // Hysteresis + framerate-normalized OLS noise floor prevents shallow scenes & low-fps streams
        // from flickering or falsely refusing valid parallax
        let pure_rotation = if no_baseline {
            false
        } else if self.parallax_locked != 0 {
            med_rot_resid <= 1.25 * res_scale * rate_scale || rot_ratio <= 0.10
        } else {
            med_rot_resid <= 1.65 * res_scale * rate_scale || rot_ratio <= 0.13
        };

        if pure_rotation {
            self.parallax_locked = 0;
            self.telem[2] = DEPTH_STATE_REFUSED_PURE_ROTATION;
            self.telem[3] = DEPTH_REASON_PURE_ROTATION;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        // 4. Run Intrinsics-Free RANSAC Focus of Expansion & f64 1/r(t) TTC over long looming chords
        let (foe_ok, foe_x, foe_y, foe_inliers, ttc_valid_cnt, min_ttc, best_r2) =
            self.solve_foe_and_ttc(ms, n_lines, frame_idx, frame_w, frame_h, res_scale);

        self.telem[7] = ttc_valid_cnt;
        self.telem[8] = foe_inliers;
        if foe_ok {
            self.telem[22] = foe_x.to_bits();
            self.telem[23] = foe_y.to_bits();
        }
        if ttc_valid_cnt > 0 {
            self.telem[24] = min_ttc.to_bits();
            self.telem[25] = best_r2.to_bits();
        }

        if no_baseline {
            self.parallax_locked = 0;
            if ttc_valid_cnt > 0 {
                self.telem[2] = DEPTH_STATE_READY_TTC_ONLY;
                self.telem[3] = DEPTH_REASON_NO_BASELINE;
                self.telem[4] = DEPTH_PROV_INTRINSICS_FREE_TTC;
                self.update_hero_depth_from_field(
                    ms,
                    hero_pos_active,
                    frame_w,
                    frame_h,
                    DEPTH_PROV_INTRINSICS_FREE_TTC,
                    false,
                );
                return true;
            }
            self.telem[2] = DEPTH_STATE_REFUSED_NO_BASELINE;
            self.telem[3] = DEPTH_REASON_NO_BASELINE;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        // 5. Verified Pure-Translation Motion-Parallax Solver (unified metric_scale condition)
        let metric_scale = self.calib_f_px >= 10.0 && self.calib_speed_m_s >= 1e-4;
        let denom = if metric_scale {
            self.calib_f_px * self.calib_speed_m_s
        } else {
            f_eff
        };
        let inv_denom = 1.0 / denom.max(1e-3);

        // Compute 2x2 velocity covariance on robust inliers
        let mut sxx = 0.0f64;
        let mut sxy = 0.0f64;
        let mut syy = 0.0f64;
        let mut sum_vx = 0.0f64;
        let mut sum_vy = 0.0f64;
        let mut cov_n = 0u32;
        let spd_cap = (3.0 * med_speed).max(15.0 * res_scale);
        k = 0;
        while k < n_cand {
            let vx = self.cand_vx[k];
            let vy = self.cand_vy[k];
            let spd = wasm_sqrtf(vx * vx + vy * vy);
            if spd > spd_floor && spd <= spd_cap {
                let vxd = vx as f64;
                let vyd = vy as f64;
                sxx += vxd * vxd;
                sxy += vxd * vyd;
                syy += vyd * vyd;
                sum_vx += vxd;
                sum_vy += vyd;
                cov_n += 1;
            }
            k += 1;
        }

        let trace = sxx + syy;
        let diff = sxx - syy;
        let disc = wasm_sqrt_f64((diff * diff + 4.0 * sxy * sxy).max(0.0));
        let lam_max = 0.5 * (trace + disc);
        let collinearity = if trace > 1e-9 { lam_max / trace } else { 0.0 };

        let mut valid_depths = 0u32;
        let mut moving_rejected = 0u32;
        let mut z_min = f32::INFINITY;
        let mut z_max = 0.0f32;
        let mut mixed_motion_refused = false;

        let sx_grid = (DEPTH_GRID_W as f32) / (frame_w as f32);
        let sy_grid = (DEPTH_GRID_H as f32) / (frame_h as f32);

        if cov_n >= 8 && collinearity >= 0.94 {
            // Stable principal eigenvector for arbitrary direction (including exact vertical sxy == 0!)
            let (mut dx_u, mut dy_u) = if sxx >= syy {
                let vx_e = lam_max - syy;
                let vy_e = sxy;
                let nrm = wasm_sqrt_f64(vx_e * vx_e + vy_e * vy_e);
                if nrm > 1e-12 {
                    (vx_e / nrm, vy_e / nrm)
                } else {
                    (1.0, 0.0)
                }
            } else {
                let vx_e = sxy;
                let vy_e = lam_max - sxx;
                let nrm = wasm_sqrt_f64(vx_e * vx_e + vy_e * vy_e);
                if nrm > 1e-12 {
                    (vx_e / nrm, vy_e / nrm)
                } else {
                    (0.0, 1.0)
                }
            };
            if dx_u * sum_vx + dy_u * sum_vy < 0.0 {
                dx_u = -dx_u;
                dy_u = -dy_u;
            }
            let dxf = dx_u as f32;
            let dyf = dy_u as f32;
            let proj_floor = 2.0 * res_scale * rate_scale;
            let epi_tol = (2.0 * res_scale * rate_scale).max(0.22 * med_speed);

            // Cross-flow 4-term bilinear purity regression `v_perp ~ b0 + b1*s_n + b2*p_n + b3*(s_n*p_n)`
            // PLUS orthogonal-axis rotation check (`v_perp` vs `v_proj` linear slope & p80(|v_perp|))
            // over static inliers (excludes independent movers):
            //   |b1|/f = roll omega_z; |b2|/f = oblique axial Tz/Z; |b3|/f = transverse yaw/pitch omega_p!
            let mut a4 = [[0.0f64; 4]; 4];
            let mut b4 = [0.0f64; 4];
            let mut n_purity = 0usize;
            let mut max_sn = 0.0f64;
            let mut max_pn = 0.0f64;
            let mut sum_pr = 0.0f64;
            let mut sum_pe = 0.0f64;
            let mut sum_pr2 = 0.0f64;
            let mut sum_pr_pe = 0.0f64;
            k = 0;
            while k < n_cand {
                let vx = self.cand_vx[k];
                let vy = self.cand_vy[k];
                let proj = vx * dxf + vy * dyf;
                let v_perp = (-vx * dyf + vy * dxf) as f64;
                if proj > proj_floor && (v_perp.abs() as f32) <= epi_tol.max(4.0 * res_scale * rate_scale) {
                    let xn = ((self.cand_x[k] - cx) as f64) * inv_f;
                    let yn = ((self.cand_y[k] - cy) as f64) * inv_f;
                    let sn = xn * dx_u + yn * dy_u;
                    let pn = -xn * dy_u + yn * dx_u;
                    if sn.abs() > max_sn {
                        max_sn = sn.abs();
                    }
                    if pn.abs() > max_pn {
                        max_pn = pn.abs();
                    }
                    let phi = [1.0, sn, pn, sn * pn];
                    let mut r = 0usize;
                    while r < 4 {
                        let mut c = 0usize;
                        while c < 4 {
                            a4[r][c] += phi[r] * phi[c];
                            c += 1;
                        }
                        b4[r] += phi[r] * v_perp;
                        r += 1;
                    }
                    let prd = proj as f64;
                    sum_pr += prd;
                    sum_pe += v_perp;
                    sum_pr2 += prd * prd;
                    sum_pr_pe += prd * v_perp;
                    self.scratch_sort[n_purity] = v_perp.abs() as f32;
                    n_purity += 1;
                }
                k += 1;
            }

            let mut pure_lateral_ok = n_purity >= 12 && max_sn >= 0.12 && max_pn >= 0.08;
            if pure_lateral_ok {
                let nf = n_purity as f64;
                let var_pr = sum_pr2 - (sum_pr * sum_pr) / nf;
                let cov_pr_pe = sum_pr_pe - (sum_pr * sum_pe) / nf;
                let perp_proj_slope = if var_pr > 1.0 {
                    (cov_pr_pe / var_pr).abs() as f32
                } else {
                    0.0
                };
                let _ = median_in_place(&mut self.scratch_sort, n_purity);
                let p80_perp = self.scratch_sort[((n_purity * 4) / 5).min(n_purity - 1)];
                let ang_speed = (med_speed / f_eff).max(1e-4);
                let rot_tol = (0.04 * ang_speed).clamp(0.0012, 0.004);
                let obl_tol = (0.06 * ang_speed).clamp(0.0020, 0.006);
                if let Some(beta) = solve_4x4_f64(a4, b4) {
                    let omega_z_est = (beta[1].abs() * inv_f) as f32;
                    let oblique_est = (beta[2].abs() * inv_f) as f32;
                    let omega_p_est = (beta[3].abs() * inv_f) as f32;
                    if omega_p_est > rot_tol
                        || omega_z_est > rot_tol
                        || oblique_est > obl_tol
                        || perp_proj_slope > 0.05
                        || (p80_perp / med_speed.max(1e-3)) > 0.06
                    {
                        pure_lateral_ok = false;
                        mixed_motion_refused = true;
                    }
                } else {
                    pure_lateral_ok = false;
                    mixed_motion_refused = true;
                }
            }

            if pure_lateral_ok {
                k = 0;
                while k < n_cand {
                    let slot = self.cand_slot[k] as usize;
                    let vx = self.cand_vx[k];
                    let vy = self.cand_vy[k];
                    let proj = vx * dxf + vy * dyf;
                    let perp = (-vx * dyf + vy * dxf).abs();
                    if perp > epi_tol || proj <= proj_floor {
                        self.export_track_flags[slot] |= DEPTH_FLAG_MOVING_OUTLIER;
                        moving_rejected += 1;
                    } else {
                        let inv_z = proj * inv_denom;
                        if inv_z.is_finite() && inv_z > 1e-6 {
                            let z_val = 1.0 / inv_z;
                            self.export_track_depth[slot * DEPTH_TRACK_STRIDE] = inv_z;
                            let mut fl = DEPTH_FLAG_VALID;
                            if metric_scale {
                                fl |= DEPTH_FLAG_METRIC;
                            }
                            self.export_track_flags[slot] |= fl;
                            valid_depths += 1;
                            if z_val < z_min {
                                z_min = z_val;
                            }
                            if z_val > z_max {
                                z_max = z_val;
                            }
                            let px = ms.tracks.pos_x[slot];
                            let py = ms.tracks.pos_y[slot];
                            let gx = (wasm_floorf(px * sx_grid) as isize)
                                .clamp(0, (DEPTH_GRID_W - 1) as isize) as usize;
                            let gy = (wasm_floorf(py * sy_grid) as isize)
                                .clamp(0, (DEPTH_GRID_H - 1) as isize) as usize;
                            let y0 = gy.saturating_sub(1);
                            let y1 = (gy + 1).min(DEPTH_GRID_H - 1);
                            let x0 = gx.saturating_sub(1);
                            let x1 = (gx + 1).min(DEPTH_GRID_W - 1);
                            let mut yy = y0;
                            while yy <= y1 {
                                let mut xx = x0;
                                while xx <= x1 {
                                    let cell = yy * DEPTH_GRID_W + xx;
                                    let wt = if xx == gx && yy == gy { 4.0 } else { 1.0 };
                                    let fbase = cell * DEPTH_FIELD_STRIDE;
                                    self.export_field[fbase] += wt * inv_z;
                                    self.export_field[fbase + 1] += wt;
                                    xx += 1;
                                }
                                yy += 1;
                            }
                        }
                    }
                    k += 1;
                }
            }
        } else if foe_ok && foe_inliers >= 4 {
            // Pure radial translation around FoE:
            //  1. 3-DoF tangential rotational-nullspace regression around FoE + tighter med/p80
            //     tangential checks (plus 5-DoF quadratic veto when FoE is strongly off-center).
            //  2. Inverse-depth relief observability guard (p90/p10 of inv_tau >= 1.65) so shallow
            //     scenes where small yaw mimics an off-center FoE shift fail closed.
            //  3. Near-FoE emission exclusion (r >= 72*res_scale, r_dot >= 5.5*res_scale*rate_scale)
            //     so tiny-r tracks right on the FoE do not emit 40%+ noisy metric depths.
            let min_r_fit = 18.0 * res_scale;
            let min_r_emit = 72.0 * res_scale;
            let rad_proj_floor = 2.0 * res_scale * rate_scale;
            let emit_r_dot_floor = 5.5 * res_scale * rate_scale;
            let ux = ((foe_x - cx) as f64) * inv_f;
            let uy = ((foe_y - cy) as f64) * inv_f;
            let mut m_rot = [0.0f64; 9];
            let mut rhs_rot = [0.0f64; 3];
            let mut a5 = [[0.0f64; 5]; 5];
            let mut b5 = [0.0f64; 5];
            let mut tan_n = 0usize;
            k = 0;
            while k < n_cand {
                let dx = self.cand_x[k] - foe_x;
                let dy = self.cand_y[k] - foe_y;
                let r = wasm_sqrtf(dx * dx + dy * dy);
                if r > min_r_fit {
                    let inv_r = 1.0 / r;
                    let rx = dx * inv_r;
                    let ry = dy * inv_r;
                    let vx = self.cand_vx[k];
                    let vy = self.cand_vy[k];
                    let r_dot = vx * rx + vy * ry;
                    if r_dot > rad_proj_floor {
                        let v_tan = -vx * ry + vy * rx;
                        self.scratch_sort[tan_n] = v_tan.abs();
                        tan_n += 1;
                        let xn = ((self.cand_x[k] - cx) as f64) * inv_f;
                        let yn = ((self.cand_y[k] - cy) as f64) * inv_f;
                        let rx64 = rx as f64;
                        let ry64 = ry as f64;
                        let au0 = f64_eff * xn * yn;
                        let au1 = -f64_eff * (1.0 + xn * xn);
                        let au2 = -f64_eff * yn;
                        let av0 = f64_eff * (1.0 + yn * yn);
                        let av1 = -f64_eff * xn * yn;
                        let av2 = f64_eff * xn;
                        let c0 = -au0 * ry64 + av0 * rx64;
                        let c1 = -au1 * ry64 + av1 * rx64;
                        let c2 = -au2 * ry64 + av2 * rx64;
                        let bt = v_tan as f64;
                        m_rot[0] += c0 * c0;
                        m_rot[1] += c0 * c1;
                        m_rot[2] += c0 * c2;
                        m_rot[4] += c1 * c1;
                        m_rot[5] += c1 * c2;
                        m_rot[8] += c2 * c2;
                        rhs_rot[0] += c0 * bt;
                        rhs_rot[1] += c1 * bt;
                        rhs_rot[2] += c2 * bt;

                        let phi5 = [
                            -ry64,
                            rx64,
                            f64_eff * (-xn * yn * ry64 + yn * yn * rx64),
                            f64_eff * (xn * xn * ry64 - xn * yn * rx64),
                            c2,
                        ];
                        let mut rr = 0usize;
                        while rr < 5 {
                            let mut cc = 0usize;
                            while cc < 5 {
                                a5[rr][cc] += phi5[rr] * phi5[cc];
                                cc += 1;
                            }
                            b5[rr] += phi5[rr] * bt;
                            rr += 1;
                        }
                    }
                }
                k += 1;
            }
            m_rot[3] = m_rot[1];
            m_rot[6] = m_rot[2];
            m_rot[7] = m_rot[5];

            let (med_tan, p80_tan) = if tan_n >= 8 {
                let med = median_in_place(&mut self.scratch_sort, tan_n);
                let p80 = self.scratch_sort[((tan_n * 4) / 5).min(tan_n - 1)];
                (med, p80)
            } else {
                (999.0, 999.0)
            };

            // Measure inverse-depth relief spread over well-conditioned radial tracks (r >= min_r_emit)
            let mut tau_n = 0usize;
            k = 0;
            while k < n_cand {
                let dx = self.cand_x[k] - foe_x;
                let dy = self.cand_y[k] - foe_y;
                let r = wasm_sqrtf(dx * dx + dy * dy);
                if r >= min_r_emit {
                    let inv_r = 1.0 / r;
                    let rx = dx * inv_r;
                    let ry = dy * inv_r;
                    let vx = self.cand_vx[k];
                    let vy = self.cand_vy[k];
                    let r_dot = vx * rx + vy * ry;
                    let perp = (-vx * ry + vy * rx).abs();
                    if r_dot >= emit_r_dot_floor && perp <= (3.0 * res_scale).max(0.25 * med_speed) {
                        self.scratch_sort[tau_n] = r_dot * inv_r;
                        tau_n += 1;
                    }
                }
                k += 1;
            }
            let radial_relief_ok = if tau_n >= 10 {
                let _ = median_in_place(&mut self.scratch_sort, tau_n);
                let p10 = self.scratch_sort[tau_n / 10];
                let p90 = self.scratch_sort[((tau_n * 9) / 10).min(tau_n - 1)];
                p10 > 1e-5 && (p90 / p10) >= 1.65
            } else {
                false
            };

            let mut rad_rot_clean = true;
            if tan_n >= 12 {
                let det_r = m_rot[0] * (m_rot[4] * m_rot[8] - m_rot[5] * m_rot[7])
                    - m_rot[1] * (m_rot[3] * m_rot[8] - m_rot[5] * m_rot[6])
                    + m_rot[2] * (m_rot[3] * m_rot[7] - m_rot[4] * m_rot[6]);
                let ms_r = (m_rot[0] + m_rot[4] + m_rot[8]) * (1.0 / 3.0);
                let ang_spd = (med_speed / f_eff).max(1e-4);
                let r_tol = (0.04 * ang_spd).clamp(0.0015, 0.004);
                if det_r.is_finite()
                    && ms_r.is_finite()
                    && ms_r > 1e-6
                    && det_r.abs() >= 1e-9 * ms_r * ms_r * ms_r
                {
                    let d0 = rhs_rot[0] * (m_rot[4] * m_rot[8] - m_rot[5] * m_rot[7])
                        - m_rot[1] * (rhs_rot[1] * m_rot[8] - m_rot[5] * rhs_rot[2])
                        + m_rot[2] * (rhs_rot[1] * m_rot[7] - m_rot[4] * rhs_rot[2]);
                    let d1 = m_rot[0] * (rhs_rot[1] * m_rot[8] - m_rot[5] * rhs_rot[2])
                        - rhs_rot[0] * (m_rot[3] * m_rot[8] - m_rot[5] * m_rot[6])
                        + m_rot[2] * (m_rot[3] * rhs_rot[2] - rhs_rot[1] * m_rot[6]);
                    let d2 = m_rot[0] * (m_rot[4] * rhs_rot[2] - rhs_rot[1] * m_rot[7])
                        - m_rot[1] * (m_rot[3] * rhs_rot[2] - rhs_rot[1] * m_rot[6])
                        + rhs_rot[0] * (m_rot[3] * m_rot[7] - m_rot[4] * m_rot[6]);
                    let inv_dr = 1.0 / det_r;
                    let wx_t = (d0 * inv_dr).abs() as f32;
                    let wy_t = (d1 * inv_dr).abs() as f32;
                    let wz_t = (d2 * inv_dr).abs() as f32;
                    if wx_t > r_tol || wy_t > r_tol || wz_t > r_tol {
                        rad_rot_clean = false;
                    }
                }
                if rad_rot_clean && (ux * ux + uy * uy) >= 0.01 {
                    if let Some(beta5) = solve_5x5_f64(a5, b5) {
                        let r_tol_q = (0.08 * ang_spd).clamp(0.003, 0.008);
                        let wx_q = beta5[2].abs() as f32;
                        let wy_q = beta5[3].abs() as f32;
                        let wz_q = beta5[4].abs() as f32;
                        if wx_q > r_tol_q || wy_q > r_tol_q || wz_q > r_tol {
                            rad_rot_clean = false;
                        }
                    }
                }
            }
            let pure_radial_ok = tan_n >= 12
                && radial_relief_ok
                && rad_rot_clean
                && med_tan <= 0.60 * res_scale * rate_scale
                && (med_tan / med_speed.max(1e-3)) <= 0.04
                && p80_tan <= 1.25 * res_scale * rate_scale
                && (p80_tan / med_speed.max(1e-3)) <= 0.055;
            if !pure_radial_ok {
                mixed_motion_refused = true;
            } else {
                // Exact 3D axial component Tz = ||T|| / sqrt(1 + ((fx-cx)/f)^2 + ((fy-cy)/f)^2)
                let t_hat_z = (1.0 / wasm_sqrt_f64(1.0 + ux * ux + uy * uy)) as f32;
                let tz_scale = self.calib_speed_m_s * t_hat_z;

                k = 0;
                while k < n_cand {
                    let slot = self.cand_slot[k] as usize;
                    let dx = self.cand_x[k] - foe_x;
                    let dy = self.cand_y[k] - foe_y;
                    let r = wasm_sqrtf(dx * dx + dy * dy);
                    if r >= min_r_emit {
                        let inv_r = 1.0 / r;
                        let rx = dx * inv_r;
                        let ry = dy * inv_r;
                        let vx = self.cand_vx[k];
                        let vy = self.cand_vy[k];
                        let r_dot = vx * rx + vy * ry;
                        let perp = (-vx * ry + vy * rx).abs();
                        if perp > (3.0 * res_scale).max(0.25 * med_speed) || r_dot <= rad_proj_floor {
                            self.export_track_flags[slot] |= DEPTH_FLAG_MOVING_OUTLIER;
                            moving_rejected += 1;
                        } else if r_dot >= emit_r_dot_floor {
                            let inv_tau = r_dot * inv_r;
                            let inv_z = if metric_scale && tz_scale > 1e-4 {
                                inv_tau / tz_scale
                            } else {
                                inv_tau
                            };
                            if inv_z.is_finite() && inv_z > 1e-6 {
                                let z_val = 1.0 / inv_z;
                                self.export_track_depth[slot * DEPTH_TRACK_STRIDE] = inv_z;
                                let mut fl = DEPTH_FLAG_VALID;
                                if metric_scale {
                                    fl |= DEPTH_FLAG_METRIC;
                                }
                                self.export_track_flags[slot] |= fl;
                                valid_depths += 1;
                                if z_val < z_min {
                                    z_min = z_val;
                                }
                                if z_val > z_max {
                                    z_max = z_val;
                                }
                                let px = ms.tracks.pos_x[slot];
                                let py = ms.tracks.pos_y[slot];
                                let gx = (wasm_floorf(px * sx_grid) as isize)
                                    .clamp(0, (DEPTH_GRID_W - 1) as isize) as usize;
                                let gy = (wasm_floorf(py * sy_grid) as isize)
                                    .clamp(0, (DEPTH_GRID_H - 1) as isize) as usize;
                                let cell = gy * DEPTH_GRID_W + gx;
                                let fbase = cell * DEPTH_FIELD_STRIDE;
                                self.export_field[fbase] += inv_z;
                                self.export_field[fbase + 1] += 1.0;
                            }
                        }
                    }
                    k += 1;
                }
            }
        } else {
            // Neither pure lateral nor centered pure radial -> oblique/mixed motion: fail closed!
            mixed_motion_refused = true;
        }

        // Normalize 64x36 sparse inverse-depth field in-place
        let mut valid_cells = 0u32;
        let mut c = 0usize;
        while c < DEPTH_CELLS {
            let base = c * DEPTH_FIELD_STRIDE;
            let wt = self.export_field[base + 1];
            if wt > 0.0 {
                self.export_field[base] /= wt;
                self.export_field[base + 1] = wt.min(16.0) * (1.0 / 16.0);
                valid_cells += 1;
            }
            c += 1;
        }

        if valid_depths > 0 {
            self.parallax_locked = 1;
        } else {
            self.parallax_locked = 0;
        }

        if mixed_motion_refused && valid_depths == 0 {
            if ttc_valid_cnt > 0 {
                self.telem[2] = DEPTH_STATE_READY_TTC_ONLY;
                self.telem[3] = DEPTH_REASON_MIXED_ROT_TRANS;
                self.telem[4] = DEPTH_PROV_INTRINSICS_FREE_TTC;
                self.update_hero_depth_from_field(
                    ms,
                    hero_pos_active,
                    frame_w,
                    frame_h,
                    DEPTH_PROV_INTRINSICS_FREE_TTC,
                    false,
                );
                return true;
            }
            self.telem[2] = DEPTH_STATE_REFUSED_PURE_ROTATION;
            self.telem[3] = DEPTH_REASON_MIXED_ROT_TRANS;
            self.telem[4] = DEPTH_PROV_NONE;
            return false;
        }

        let prov = if valid_depths > 0 {
            if metric_scale {
                DEPTH_PROV_MONO_METRIC
            } else {
                DEPTH_PROV_MONO_UP_TO_SCALE
            }
        } else if ttc_valid_cnt > 0 {
            DEPTH_PROV_INTRINSICS_FREE_TTC
        } else {
            DEPTH_PROV_NONE
        };

        self.telem[2] = if valid_depths > 0 {
            DEPTH_STATE_READY_PARALLAX
        } else if ttc_valid_cnt > 0 {
            DEPTH_STATE_READY_TTC_ONLY
        } else {
            DEPTH_STATE_WARMING
        };
        self.telem[3] = if !metric_scale && valid_depths > 0 {
            DEPTH_REASON_NO_INTRINSICS
        } else {
            0
        };
        self.telem[4] = prov;
        self.telem[5] = valid_depths;
        self.telem[6] = moving_rejected;
        self.telem[9] = valid_cells;
        self.telem[30] = if valid_depths > 0 && z_min.is_finite() {
            z_min.to_bits()
        } else {
            0
        };
        self.telem[31] = if valid_depths > 0 && z_max.is_finite() {
            z_max.to_bits()
        } else {
            0
        };

        if valid_cells > 0 || ttc_valid_cnt > 0 {
            self.update_hero_depth_from_field(
                ms,
                hero_pos_active,
                frame_w,
                frame_h,
                prov,
                metric_scale,
            );
        }
        valid_depths > 0 || ttc_valid_cnt > 0
    }

    /// Deterministic RANSAC line-intersection Focus of Expansion + `f64` bounded-window `1/r(t)`
    /// optical looming Time-to-Contact (requires spatially distributed angular support around FoE).
    #[inline(never)]
    fn solve_foe_and_ttc(
        &mut self,
        ms: &MicroSwarmState,
        n_lines: usize,
        frame_idx: u32,
        frame_w: usize,
        frame_h: usize,
        res_scale: f32,
    ) -> (bool, f32, f32, u32, u32, f32, f32) {
        if n_lines < 3 {
            return (false, 0.0, 0.0, 0, 0, 0.0, 0.0);
        }
        // Sort line indices by descending chord length so pairwise RANSAC truly searches longest chords
        let mut ord = [0u8; MAX_FOE_LINES];
        let mut ii = 0usize;
        while ii < n_lines {
            ord[ii] = ii as u8;
            ii += 1;
        }
        let lim = n_lines.min(48);
        let mut a = 0usize;
        while a < lim {
            let mut best_k = a;
            let mut best_d = self.line_disp[ord[a] as usize];
            let mut b = a + 1;
            while b < n_lines {
                let d = self.line_disp[ord[b] as usize];
                if d > best_d {
                    best_d = d;
                    best_k = b;
                }
                b += 1;
            }
            if best_k != a {
                ord.swap(a, best_k);
            }
            a += 1;
        }

        let wf = frame_w as f32;
        let hf = frame_h as f32;
        let inlier_tol = 10.0 * res_scale;

        let mut best_inliers = 0u32;
        let mut best_fx = 0.0f32;
        let mut best_fy = 0.0f32;

        a = 0;
        while a < lim {
            let ia = ord[a] as usize;
            let nx1 = self.line_nx[ia];
            let ny1 = self.line_ny[ia];
            let c1 = self.line_c[ia];
            let mut b = a + 1;
            while b < lim {
                let ib = ord[b] as usize;
                let nx2 = self.line_nx[ib];
                let ny2 = self.line_ny[ib];
                let det = nx1 * ny2 - ny1 * nx2;
                if det.abs() >= 0.14 {
                    let c2 = self.line_c[ib];
                    let inv_d = 1.0 / det;
                    let fx = (ny2 * c1 - ny1 * c2) * inv_d;
                    let fy = (-nx2 * c1 + nx1 * c2) * inv_d;
                    if fx >= 0.0 && fx <= wf && fy >= 0.0 && fy <= hf {
                        let mut cnt = 0u32;
                        let mut quad_mask = 0u8;
                        let mut m = 0usize;
                        while m < n_lines {
                            let err = (self.line_nx[m] * fx + self.line_ny[m] * fy - self.line_c[m]).abs();
                            if err < inlier_tol {
                                let sl = self.line_slot[m] as usize;
                                let ax = dequant_q5(self.anc_x_q5[sl]);
                                let ay = dequant_q5(self.anc_y_q5[sl]);
                                let px = ms.tracks.pos_x[sl];
                                let py = ms.tracks.pos_y[sl];
                                let rdx = px - fx;
                                let rdy = py - fy;
                                if rdx * (px - ax) + rdy * (py - ay) > 0.0 {
                                    cnt += 1;
                                    let q = (if rdx >= 0.0 { 1u8 } else { 0u8 })
                                        | (if rdy >= 0.0 { 2u8 } else { 0u8 });
                                    quad_mask |= 1u8 << q;
                                }
                            }
                            m += 1;
                        }
                        // Require spatial support across >= 2 quadrants around candidate FoE
                        if cnt > best_inliers && quad_mask.count_ones() >= 2 {
                            best_inliers = cnt;
                            best_fx = fx;
                            best_fy = fy;
                        }
                    }
                }
                b += 1;
            }
            a += 1;
        }

        if best_inliers < 3 {
            return (false, 0.0, 0.0, best_inliers, 0, 0.0, 0.0);
        }

        // Least-squares FoE refinement over consensus inliers
        let mut a00 = 0.0f64;
        let mut a01 = 0.0f64;
        let mut a11 = 0.0f64;
        let mut b0 = 0.0f64;
        let mut b1 = 0.0f64;
        let mut m = 0usize;
        while m < n_lines {
            let err = (self.line_nx[m] * best_fx + self.line_ny[m] * best_fy - self.line_c[m]).abs();
            if err < inlier_tol {
                let sl = self.line_slot[m] as usize;
                let ax = dequant_q5(self.anc_x_q5[sl]);
                let ay = dequant_q5(self.anc_y_q5[sl]);
                let px = ms.tracks.pos_x[sl];
                let py = ms.tracks.pos_y[sl];
                if (px - best_fx) * (px - ax) + (py - best_fy) * (py - ay) > 0.0 {
                    let nx = self.line_nx[m] as f64;
                    let ny = self.line_ny[m] as f64;
                    let c = self.line_c[m] as f64;
                    a00 += nx * nx;
                    a01 += nx * ny;
                    a11 += ny * ny;
                    b0 += nx * c;
                    b1 += ny * c;
                }
            }
            m += 1;
        }
        let det = a00 * a11 - a01 * a01;
        let (mut foe_x, mut foe_y) = if det.abs() > 1e-6 {
            (
                ((a11 * b0 - a01 * b1) / det) as f32,
                ((-a01 * b0 + a00 * b1) / det) as f32,
            )
        } else {
            (best_fx, best_fy)
        };
        if !foe_x.is_finite() || !foe_y.is_finite() || foe_x < 0.0 || foe_x > wf || foe_y < 0.0 || foe_y > hf {
            foe_x = best_fx;
            foe_y = best_fy;
        }

        let ema_gate = 40.0 * res_scale;
        if self.foe_valid != 0 {
            let dfx = foe_x - self.foe_x_ema;
            let dfy = foe_y - self.foe_y_ema;
            if dfx * dfx + dfy * dfy < ema_gate * ema_gate {
                self.foe_x_ema = 0.75 * self.foe_x_ema + 0.25 * foe_x;
                self.foe_y_ema = 0.75 * self.foe_y_ema + 0.25 * foe_y;
            } else {
                self.foe_x_ema = foe_x;
                self.foe_y_ema = foe_y;
            }
        } else {
            self.foe_x_ema = foe_x;
            self.foe_y_ema = foe_y;
            self.foe_valid = 1;
        }
        let fx = self.foe_x_ema;
        let fy = self.foe_y_ema;

        // Running f64 bounded-window 1/r(t) linear regression fits on FoE-consistent approaching tracks
        let cur_t = self.sim_time_s;
        let mut ttc_valid_cnt = 0u32;
        let mut min_ttc = f32::INFINITY;
        let mut best_r2 = 0.0f32;
        let track_tol = 12.0 * res_scale;
        let min_r_ttc = 20.0 * res_scale;
        let foe_reset_tol = 16.0 * res_scale;

        m = 0;
        while m < n_lines {
            let err = (self.line_nx[m] * fx + self.line_ny[m] * fy - self.line_c[m]).abs();
            if err < track_tol {
                let sl = self.line_slot[m] as usize;
                let tid = ms.tracks.track_id[sl];
                let px = ms.tracks.pos_x[sl];
                let py = ms.tracks.pos_y[sl];
                let vx = ms.tracks.vel_x[sl];
                let vy = ms.tracks.vel_y[sl];
                let dx = px - fx;
                let dy = py - fy;
                let r = wasm_sqrtf(dx * dx + dy * dy);
                if r > min_r_ttc {
                    let r_dot = (dx * vx + dy * vy) / r;
                    if r_dot > 3.0 * res_scale {
                        let mut hit = MAX_TTC_FITS;
                        let mut free = MAX_TTC_FITS;
                        let mut oldest = 0usize;
                        let mut oldest_f = u32::MAX;
                        let mut f_idx = 0usize;
                        while f_idx < MAX_TTC_FITS {
                            if self.ttc_fits[f_idx].tid == tid {
                                hit = f_idx;
                                break;
                            }
                            if self.ttc_fits[f_idx].tid == 0 && free == MAX_TTC_FITS {
                                free = f_idx;
                            }
                            if self.ttc_fits[f_idx].last_seen_frame < oldest_f {
                                oldest_f = self.ttc_fits[f_idx].last_seen_frame;
                                oldest = f_idx;
                            }
                            f_idx += 1;
                        }
                        let slot_k = if hit < MAX_TTC_FITS {
                            hit
                        } else if free < MAX_TTC_FITS {
                            free
                        } else {
                            oldest
                        };

                        // Reset on new track, non-contiguous gap (>1 frame), FoE jump, or window cap
                        let prev = self.ttc_fits[slot_k];
                        let need_reset = prev.tid != tid
                            || frame_idx.saturating_sub(prev.last_seen_frame) > 1
                            || (prev.foe_x - fx).abs() > foe_reset_tol
                            || (prev.foe_y - fy).abs() > foe_reset_tol
                            || prev.n >= TTC_MAX_WINDOW_FRAMES;
                        if need_reset {
                            let inv_r = 1.0 / (r as f64);
                            self.ttc_fits[slot_k] = TtcTrackFit {
                                t0: cur_t,
                                sum_dt: 0.0,
                                sum_dt2: 0.0,
                                sum_invr: inv_r,
                                sum_invr2: inv_r * inv_r,
                                sum_dt_invr: 0.0,
                                tid,
                                n: 1,
                                last_seen_frame: frame_idx,
                                foe_x: fx,
                                foe_y: fy,
                                last_r2: 0.0,
                            };
                        } else {
                            let fit = &mut self.ttc_fits[slot_k];
                            let fdx = (px - fit.foe_x) as f64;
                            let fdy = (py - fit.foe_y) as f64;
                            let fr = wasm_sqrt_f64(fdx * fdx + fdy * fdy).max(1.0);
                            let inv_r = 1.0 / fr;
                            let dt0 = cur_t - fit.t0;
                            fit.n = fit.n.saturating_add(1);
                            fit.last_seen_frame = frame_idx;
                            fit.sum_dt += dt0;
                            fit.sum_dt2 += dt0 * dt0;
                            fit.sum_invr += inv_r;
                            fit.sum_invr2 += inv_r * inv_r;
                            fit.sum_dt_invr += dt0 * inv_r;

                            if fit.n >= 12 {
                                let nf = fit.n as f64;
                                let sxx = fit.sum_dt2 - (fit.sum_dt * fit.sum_dt) / nf;
                                let syy = fit.sum_invr2 - (fit.sum_invr * fit.sum_invr) / nf;
                                let sxy = fit.sum_dt_invr - (fit.sum_dt * fit.sum_invr) / nf;
                                if sxx > 1e-6 && syy > 1e-12 && sxy < 0.0 {
                                    let slope = sxy / sxx;
                                    let intercept = (fit.sum_invr - slope * fit.sum_dt) / nf;
                                    let r2 = ((sxy * sxy) / (sxx * syy)).clamp(0.0, 1.0) as f32;
                                    fit.last_r2 = r2;
                                    if r2 >= 0.90 && slope < -1e-6 {
                                        let t_contact_rel = -intercept / slope;
                                        let tau_fit = (t_contact_rel - dt0) as f32;
                                        if tau_fit.is_finite() && tau_fit > 0.05 && tau_fit < 60.0 {
                                            self.export_track_depth[sl * DEPTH_TRACK_STRIDE + 1] =
                                                tau_fit;
                                            self.export_track_flags[sl] |= DEPTH_FLAG_TTC_VALID;
                                            ttc_valid_cnt += 1;
                                            if tau_fit < min_ttc {
                                                min_ttc = tau_fit;
                                            }
                                            if r2 > best_r2 {
                                                best_r2 = r2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            m += 1;
        }

        // Expire stale TTC fits not seen for > 1 frame (contiguous fit hygiene)
        let mut f_idx = 0usize;
        while f_idx < MAX_TTC_FITS {
            if self.ttc_fits[f_idx].tid != 0
                && frame_idx.saturating_sub(self.ttc_fits[f_idx].last_seen_frame) > 1
            {
                self.ttc_fits[f_idx] = TtcTrackFit::zeroed();
            }
            f_idx += 1;
        }

        (
            true,
            fx,
            fy,
            best_inliers,
            ttc_valid_cnt,
            if ttc_valid_cnt > 0 { min_ttc } else { 0.0 },
            best_r2,
        )
    }

    /// Top-level post-tick depth dispatch. Zero cost when `self.mode == DEPTH_MODE_OFF`.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        ms: &MicroSwarmState,
        bos: &mut BosState,
        bos_cache: &mut BosRefCache,
        bos_pyr: &mut BosPyrScratch,
        cur_luma: &[f32],
        cur_pyr: &[f32],
        hero_pos_active: &[(bool, f32, f32); MAX_TARGETS],
        frame_idx: u32,
        frame_w: usize,
        frame_h: usize,
        actual_dt_sec: f32,
        frame_t_sec: f64,
        continuity_epoch: u32,
        scene_cut: bool,
    ) -> bool {
        let ran = match self.mode {
            DEPTH_MODE_OFF => false,
            DEPTH_MODE_MONO_PARALLAX_TTC => self.step_mono_parallax_ttc(
                ms,
                hero_pos_active,
                frame_idx,
                frame_w,
                frame_h,
                actual_dt_sec,
                frame_t_sec,
                continuity_epoch,
                scene_cut,
            ),
            DEPTH_MODE_STEREO_BOS => self.step_stereo_bridge(
                ms,
                bos,
                bos_cache,
                bos_pyr,
                cur_luma,
                cur_pyr,
                hero_pos_active,
                frame_idx,
                frame_w,
                frame_h,
                continuity_epoch,
            ),
            DEPTH_MODE_EXTERNAL => self.step_external(
                ms,
                hero_pos_active,
                frame_idx,
                frame_w,
                frame_h,
                continuity_epoch,
            ),
            DEPTH_MODE_AUTO_FUSED => {
                // Priority with clean fallthrough: Surveyed Stereo -> Fresh External -> Monocular Parallax/TTC
                if self.calib_f_px >= 10.0
                    && self.calib_baseline_m >= 1e-4
                    && bos.has_fresh_stereo_ref(frame_w, frame_h, frame_idx, continuity_epoch)
                    && self.step_stereo_bridge(
                        ms,
                        bos,
                        bos_cache,
                        bos_pyr,
                        cur_luma,
                        cur_pyr,
                        hero_pos_active,
                        frame_idx,
                        frame_w,
                        frame_h,
                        continuity_epoch,
                    )
                {
                    // Stereo won frame `frame_idx`: expire any staged external plane for this frame
                    // so it can NEVER be published as fresh on frame_idx + 1!
                    self.ext_valid = 0;
                    self.ext_frame_idx = u32::MAX;
                    true
                } else if self.ext_valid != 0
                    && self.ext_epoch == continuity_epoch
                    && self.ext_frame_idx == frame_idx
                    && self.step_external(
                        ms,
                        hero_pos_active,
                        frame_idx,
                        frame_w,
                        frame_h,
                        continuity_epoch,
                    )
                {
                    true
                } else {
                    self.step_mono_parallax_ttc(
                        ms,
                        hero_pos_active,
                        frame_idx,
                        frame_w,
                        frame_h,
                        actual_dt_sec,
                        frame_t_sec,
                        continuity_epoch,
                        scene_cut,
                    )
                }
            }
            _ => false,
        };
        // Unconditionally expire any external staging that was bound to this or an earlier frame
        if self.ext_valid != 0 && self.ext_frame_idx <= frame_idx {
            self.ext_valid = 0;
            self.ext_frame_idx = u32::MAX;
        }
        ran
    }
}
