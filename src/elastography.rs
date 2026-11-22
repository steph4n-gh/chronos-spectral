//! kHz Optical Elastography, Wrap-Free Mechanical Power Flow & Lock-In Acoustic BOS (`Wave D1`).
//!
//! Pure `#![no_std]`, zero-allocation, zero-`unsafe`, zero-import implementation inside static
//! `ElastoState` (`<= 4,096 B` permanent footprint) borrowing `BosState`'s `f64` solver scratch,
//! reference luma, and output planes under strict mutual exclusion (`bos_scratch_owner` in `{4, 5}`),
//! reusing `VibrometryState`'s audio ring & D4 rolling-shutter geometry, `BosState`'s SIMD128
//! `256x128` IC-LK & paired radix-2 DCT Neumann Poisson solver, and `HodgeState`'s 2-complex
//! discrete Hodge decomposition for structural power-flow source/sink localization.

use crate::bos::{node_coord, BosRefCache, BosState, BOS_GRID_H, BOS_GRID_W, BOS_NODES};
use crate::hodge::{HodgeState, HODGE_H, HODGE_V, HODGE_W};
use crate::vibrometry::{
    engine_to_sensor_xy, ln_f64, oriented_extent, rel_us_f64, sin_cos_f64, sinc_exposure_h,
    VibroFrameMeta, VibrometryState, RS_REASON_AUDIO_UNDERRUN, RS_REASON_AXIS_TRANSPOSED,
    RS_REASON_BAD_CALIB, RS_REASON_CAMERA_MOTION, RS_REASON_CLOCK_SKEW, RS_REASON_EXPOSURE_NULL,
    RS_REASON_FRAME_GAP, RS_REASON_LOW_GRAD, RS_REASON_NOT_NATIVE_1TO1, RS_REASON_NO_PTS,
    RS_REASON_NO_SCAN_DIR,
};
use crate::{fast_atan2f, wasm_sqrt_f64};

pub const ELASTO_W: usize = HODGE_W; // 64
pub const ELASTO_H: usize = HODGE_H; // 36
pub const ELASTO_V: usize = HODGE_V; // 2,304
pub const ELASTO_TELEM_LEN: usize = 32;
pub const ELASTO_MAX_FREQS: usize = 4;

// Operating modes (`chronos_set_elasto_mode`)
pub const ELASTO_MODE_OFF: u32 = 0;
pub const ELASTO_MODE_SURFACE_RS_AUDIO: u32 = 1;
pub const ELASTO_MODE_SURFACE_RS_SELF: u32 = 2;
pub const ELASTO_MODE_SURFACE_FRAMERATE: u32 = 3;
pub const ELASTO_MODE_ACOUSTIC_BOS: u32 = 4;

// State enum (`telem[0]`)
pub const ELASTO_STATE_OFF: u32 = 0;
pub const ELASTO_STATE_UNCALIBRATED: u32 = 1;
pub const ELASTO_STATE_WARMING: u32 = 2;
pub const ELASTO_STATE_READY: u32 = 3;
pub const ELASTO_STATE_REFUSED: u32 = 4;

// Constitutive dispersion class (`telem[3]`)
pub const ELASTO_CLASS_NONE: u32 = 0;
pub const ELASTO_CLASS_KIRCHHOFF_PLATE: u32 = 1;
pub const ELASTO_CLASS_MEMBRANE_SHEAR: u32 = 2;
pub const ELASTO_CLASS_DISPERSIVE_UNKNOWN: u32 = 3;

// Fail-closed reason bits (`telem[2]`, bits 0..10 match RS_REASON_*)
pub const ELASTO_REASON_SPATIAL_ALIASING: u32 = 1 << 16;
pub const ELASTO_REASON_WAVELENGTH_TOO_LONG: u32 = 1 << 17;
pub const ELASTO_REASON_STANDING_WAVE_DOMINANT: u32 = 1 << 18;
pub const ELASTO_REASON_MUTEX_CONFLICT: u32 = 1 << 19;
pub const ELASTO_REASON_LOW_COH_COVERAGE: u32 = 1 << 20;

// Telemetry word indices (`[u32; 32]`; 8..=27 store `f32::to_bits()`)
pub const ELASTO_TELEM_STATE: usize = 0;
pub const ELASTO_TELEM_MODE: usize = 1;
pub const ELASTO_TELEM_REASON_MASK: usize = 2;
pub const ELASTO_TELEM_CLASS: usize = 3;
pub const ELASTO_TELEM_VALID_CELLS: usize = 4;
pub const ELASTO_TELEM_LOCKED_FREQS: usize = 5;
pub const ELASTO_TELEM_WELCH_SEGS: usize = 6;
pub const ELASTO_TELEM_BETA1_HOLES: usize = 7;
pub const ELASTO_TELEM_F0_HZ: usize = 8;
pub const ELASTO_TELEM_F1_HZ: usize = 9;
pub const ELASTO_TELEM_F2_HZ: usize = 10;
pub const ELASTO_TELEM_F3_HZ: usize = 11;
pub const ELASTO_TELEM_MED_C: usize = 12;
pub const ELASTO_TELEM_DISPERSION_ALPHA: usize = 13;
pub const ELASTO_TELEM_ANISOTROPY: usize = 14;
pub const ELASTO_TELEM_MEAN_COH: usize = 15;
pub const ELASTO_TELEM_SOURCE_X: usize = 16;
pub const ELASTO_TELEM_SOURCE_Y: usize = 17;
pub const ELASTO_TELEM_SINK_X: usize = 18;
pub const ELASTO_TELEM_SINK_Y: usize = 19;
pub const ELASTO_TELEM_COEXACT_FRAC: usize = 20;
pub const ELASTO_TELEM_EXACT_FRAC: usize = 21;
pub const ELASTO_TELEM_MED_K: usize = 22;
pub const ELASTO_TELEM_SINC_H0: usize = 23;
pub const ELASTO_TELEM_ABOS_CURL_RMS: usize = 24;
pub const ELASTO_TELEM_ABOS_PHI_RMS: usize = 25;
pub const ELASTO_TELEM_C_P10: usize = 26;
pub const ELASTO_TELEM_C_P90: usize = 27;
pub const ELASTO_TELEM_GRID_W: usize = 28;
pub const ELASTO_TELEM_GRID_H: usize = 29;
pub const ELASTO_TELEM_PROVENANCE: usize = 30;
pub const ELASTO_TELEM_RESERVED31: usize = 31;

// Surface mode output plane indices inside `BosState.planes` (stride `ELASTO_V = 2,304`)
pub const ELASTO_SURF_PLANE_C: usize = 0;
pub const ELASTO_SURF_PLANE_COH: usize = 1;
pub const ELASTO_SURF_PLANE_DIV: usize = 2;
pub const ELASTO_SURF_PLANE_AMP: usize = 3;
pub const ELASTO_SURF_PLANE_PHASE: usize = 4;
pub const ELASTO_SURF_PLANE_PHI: usize = 5;

// Acoustic BOS output plane indices inside `BosState.planes` (stride `BOS_NODES = 32,768`)
pub const ELASTO_ABOS_PLANE_PHI_MAG: usize = 0;
pub const ELASTO_ABOS_PLANE_PHI_PHASE: usize = 1;
pub const ELASTO_ABOS_PLANE_PHI_RE: usize = 2;
pub const ELASTO_ABOS_PLANE_PHI_IM: usize = 3;
pub const ELASTO_ABOS_PLANE_EPS_MAG: usize = 4;
pub const ELASTO_ABOS_PLANE_CURL_MAG: usize = 5;

// Surface sliding segment parameters:
//   `K_SEGS = 8` segments x `MAX_FREQS = 4` x `2,304` cells x `2 (re, im)` = `147,456` f64
//   Packed cleanly across `BosState.solve_{x,b,r,p,z}` (`5 * 32,768 = 163,840` f64),
//   leaving `solve_z[16_384..32_768]` (`16,384` f64) AND all of `solve_lp[0..32_768]` (`32,768` f64)
//   free for per-segment normal equations, Helmholtz/Poynting scratch, and Hodge power-flow arrays!
const K_SEGS: usize = 8;
const SEG_FRAMES: u32 = 4;
const REF_WARM_FRAMES: u8 = 16;
const FRAMERATE_HIST_LEN: usize = 24;

// Offsets inside `bos.solve_lp` (`32,768` f64) for active segment normal equations
// (persisted across the `SEG_FRAMES` frames of the current segment):
//   G[36][64]        : 0 .. 2,304
//   rhs[2,304][8]    : 2,304 .. 20,736
// Ephemeral solve scratch inside `bos.solve_z[16_384..32_768]` (`16,384` f64 = 7.1 grids)
// + `bos.solve_lp[20_736..32_768]` (`12,032` f64 = 5.2 grids):
const LP_G_OFF: usize = 0; // 2,304
const LP_RHS_OFF: usize = 2_304; // 18,432 -> ends 20,736
const LP_KX_OFF: usize = 20_736; // 2,304 -> ends 23,040
const LP_KY_OFF: usize = 23_040; // 2,304 -> ends 25,344
const LP_K0_OFF: usize = 25_344; // 2,304 -> ends 27,648
const LP_K1_OFF: usize = 27_648; // 2,304 -> ends 29,952
const LP_SORT_OFF: usize = 29_952; // 2,816 -> ends 32,768

// Ephemeral complex wavefield & Hodge scratch in `bos.solve_z[16_384..32_768]`
const Z_BASE: usize = 16_384;
const Z_URE_OFF: usize = Z_BASE; // 2,304
const Z_UIM_OFF: usize = Z_BASE + 2_304; // 2,304
const Z_IX_OFF: usize = Z_BASE + 4_608; // 2,304
const Z_IY_OFF: usize = Z_BASE + 6_912; // 2,304
const Z_PHI_OFF: usize = Z_BASE + 9_216; // 2,304
const Z_DIV_OFF: usize = Z_BASE + 11_520; // 2,304
const Z_COH_OFF: usize = Z_BASE + 13_824; // 2,304 -> ends 32,512 <= 32,768

// 9-point smooth C^3 compact-support weak test function phi(s) = (1 - s^2)^4 on [-1, 1],
// sampled at s_k = (k - 4) / 5 (`k = 0..8`; endpoints `k = -1, 9` sit at `s = +-1` where phi=phi'=phi''=0).
const HELM_PHI: [f64; 9] = [
    0.016777216,
    0.16777216,
    0.49787136,
    0.84934656,
    1.0,
    0.84934656,
    0.49787136,
    0.16777216,
    0.016777216,
];
// phi''(s_k) / 25, mean-adjusted so sum == 0 on constants (for weak-form Laplacian `<Delta psi, u>`)
const HELM_D2PHI: [f64; 9] = [
    0.11907072000000001,
    0.20729856,
    0.010977280000000002,
    -0.2174976,
    -0.23969792,
    -0.2174976,
    0.010977280000000002,
    0.20729856,
    0.11907072000000001,
];

const TWO_PI_64: f64 = 6.28318530717958647692;
const PI_64: f64 = 3.14159265358979323846;

#[inline(always)]
fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
}

#[inline(always)]
fn pack_f32(v: f64) -> u32 {
    let f = v as f32;
    if f.is_finite() {
        f.to_bits()
    } else {
        0
    }
}

/// Static zero-allocation kHz Optical Elastography, Structural Power Flow & Lock-In Acoustic BOS state.
#[repr(C, align(16))]
pub struct ElastoState {
    pub telem: [u32; ELASTO_TELEM_LEN],
    pub mode: u32,
    pub fresh_solve: u8,
    pub live_ready: u8,
    pub ref_warm_frames: u8,
    pub has_explicit_freqs: u8,
    pub explicit_freq_count: u32,
    pub explicit_freqs: [f32; ELASTO_MAX_FREQS],
    pub active_freq_count: u32,
    pub _pad0: u32,
    pub active_freqs: [f64; ELASTO_MAX_FREQS],
    pub welch_head: u32,
    pub welch_count: u32,
    pub seg_frame_acc: u32,
    pub last_continuity_epoch: u32,
    pub t0_us: u64,
    pub global_ax_x: f64,
    pub global_ax_y: f64,
    pub global_ax_locked: u8,
    pub _pad1: [u8; 3],
    pub last_frame_w: u16,
    pub last_frame_h: u16,
    pub last_orientation: u32,
    pub last_scan_dir: u32,
    pub fr_hist_head: u32,
    pub fr_hist_count: u32,
    pub fr_hist_t: [f64; FRAMERATE_HIST_LEN],
    pub abos_frames_acc: u32,
    pub abos_solve_countdown: u32,
    pub abos_freq_hz: f64,
    pub abos_valid_nodes: u32,
    pub _reserved: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<ElastoState>() <= 4_096);

/// Read/write helper into the `147,456`-entry `f64` segment ring spanning `BosState.solve_{x,b,r,p,z}`.
/// Layout index: `idx = ((seg * ELASTO_MAX_FREQS + m) * 2 + part) * ELASTO_V + v` (`part = 0` re, `1` im).
#[inline(always)]
fn seg_ring_read(bos: &BosState, idx: usize) -> f64 {
    let bank = idx >> 15;
    let off = idx & 32_767;
    match bank {
        0 => bos.solve_x[off],
        1 => bos.solve_b[off],
        2 => bos.solve_r[off],
        3 => bos.solve_p[off],
        _ => bos.solve_z[off],
    }
}

#[inline(always)]
fn seg_ring_write(bos: &mut BosState, idx: usize, val: f64) {
    let bank = idx >> 15;
    let off = idx & 32_767;
    match bank {
        0 => bos.solve_x[off] = val,
        1 => bos.solve_b[off] = val,
        2 => bos.solve_r[off] = val,
        3 => bos.solve_p[off] = val,
        _ => bos.solve_z[off] = val,
    }
}

#[inline(always)]
fn seg_ring_idx(seg: usize, m: usize, part: usize, v: usize) -> usize {
    ((seg * ELASTO_MAX_FREQS + m) * 2 + part) * ELASTO_V + v
}

/// Read/write helper for the `24`-frame `64x36` Framerate history ring inside `bos.w_x` (`14`) + `bos.w_y` (`10`).
#[inline(always)]
fn fr_ring_read(bos: &BosState, slot: usize, v: usize) -> f64 {
    let idx = slot * ELASTO_V + v;
    if idx < BOS_NODES {
        bos.w_x[idx] as f64
    } else {
        bos.w_y[idx - BOS_NODES] as f64
    }
}

#[inline(always)]
fn fr_ring_write(bos: &mut BosState, slot: usize, v: usize, val: f32) {
    let idx = slot * ELASTO_V + v;
    if idx < BOS_NODES {
        bos.w_x[idx] = val;
    } else {
        bos.w_y[idx - BOS_NODES] = val;
    }
}

impl ElastoState {
    pub fn init_in_place(&mut self) {
        self.telem.fill(0);
        self.mode = ELASTO_MODE_OFF;
        self.fresh_solve = 0;
        self.live_ready = 0;
        self.ref_warm_frames = 0;
        self.has_explicit_freqs = 0;
        self.explicit_freq_count = 0;
        self.explicit_freqs = [0.0; ELASTO_MAX_FREQS];
        self.active_freq_count = 0;
        self._pad0 = 0;
        self.active_freqs = [0.0; ELASTO_MAX_FREQS];
        self.welch_head = 0;
        self.welch_count = 0;
        self.seg_frame_acc = 0;
        self.last_continuity_epoch = 0;
        self.t0_us = 0;
        self.global_ax_x = 1.0;
        self.global_ax_y = 0.0;
        self.global_ax_locked = 0;
        self._pad1 = [0; 3];
        self.last_frame_w = 0;
        self.last_frame_h = 0;
        self.last_orientation = 0;
        self.last_scan_dir = 0;
        self.fr_hist_head = 0;
        self.fr_hist_count = 0;
        self.fr_hist_t = [0.0; FRAMERATE_HIST_LEN];
        self.abos_frames_acc = 0;
        self.abos_solve_countdown = 0;
        self.abos_freq_hz = 0.0;
        self.abos_valid_nodes = 0;
        self._reserved = [0; 3];
    }

    #[inline(always)]
    pub fn get_mode(&self) -> u32 {
        self.mode
    }

    #[inline(always)]
    pub fn get_state(&self) -> u32 {
        self.telem[ELASTO_TELEM_STATE]
    }

    #[inline(always)]
    pub fn get_telemetry_ptr(&self) -> *const u32 {
        self.telem.as_ptr()
    }

    pub fn set_mode(&mut self, mode: u32) -> u32 {
        if mode > ELASTO_MODE_ACOUSTIC_BOS {
            return 0;
        }
        if mode != self.mode {
            self.mode = mode;
            self.reset_runtime(0);
        }
        1
    }

    pub fn set_lock_freqs(&mut self, count: u32, f0: f32, f1: f32, f2: f32, f3: f32) -> u32 {
        if count > (ELASTO_MAX_FREQS as u32) {
            return 0;
        }
        if count == 0 {
            self.has_explicit_freqs = 0;
            self.explicit_freq_count = 0;
            self.explicit_freqs = [0.0; ELASTO_MAX_FREQS];
            self.reset_accumulators_only();
            return 1;
        }
        let raw = [f0, f1, f2, f3];
        let mut clean = [0.0f32; ELASTO_MAX_FREQS];
        let n = count as usize;
        let mut i = 0usize;
        while i < n {
            if !raw[i].is_finite() || raw[i] <= 0.1 || raw[i] > 50_000.0 {
                return 0;
            }
            clean[i] = raw[i];
            i += 1;
        }
        // Sort ascending
        i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n {
                if clean[j] < clean[i] {
                    let t = clean[i];
                    clean[i] = clean[j];
                    clean[j] = t;
                }
                j += 1;
            }
            i += 1;
        }
        self.has_explicit_freqs = 1;
        self.explicit_freq_count = count;
        self.explicit_freqs = clean;
        self.reset_accumulators_only();
        1
    }

    pub fn reset_keep_config(&mut self) {
        self.reset_runtime(0);
    }

    fn reset_accumulators_only(&mut self) {
        self.active_freq_count = 0;
        self.active_freqs = [0.0; ELASTO_MAX_FREQS];
        self.welch_head = 0;
        self.welch_count = 0;
        self.seg_frame_acc = 0;
        self.abos_frames_acc = 0;
        self.abos_solve_countdown = 0;
        self.abos_freq_hz = 0.0;
        self.fresh_solve = 0;
        self.live_ready = 0;
    }

    pub fn reset_runtime(&mut self, reason: u32) {
        self.ref_warm_frames = 0;
        self.t0_us = 0;
        self.global_ax_x = 1.0;
        self.global_ax_y = 0.0;
        self.global_ax_locked = 0;
        self.fr_hist_head = 0;
        self.fr_hist_count = 0;
        self.fr_hist_t = [0.0; FRAMERATE_HIST_LEN];
        self.abos_valid_nodes = 0;
        self.reset_accumulators_only();
        self.telem.fill(0);
        self.telem[ELASTO_TELEM_MODE] = self.mode;
        if self.mode == ELASTO_MODE_OFF {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_OFF;
        } else if reason != 0 {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_REFUSED;
            self.telem[ELASTO_TELEM_REASON_MASK] = reason;
        } else {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
        }
        let (gw, gh) = if self.mode == ELASTO_MODE_ACOUSTIC_BOS {
            (BOS_GRID_W as u32, BOS_GRID_H as u32)
        } else {
            (ELASTO_W as u32, ELASTO_H as u32)
        };
        self.telem[ELASTO_TELEM_GRID_W] = gw;
        self.telem[ELASTO_TELEM_GRID_H] = gh;
    }

    pub fn fail_refused(&mut self, reason: u32, bos: &mut BosState, clear_planes: bool) {
        self.fresh_solve = 0;
        self.live_ready = 0;
        if clear_planes {
            bos.planes.fill(0.0);
            bos.valid_mask.fill(0);
        }
        self.telem.fill(0);
        self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_REFUSED;
        self.telem[ELASTO_TELEM_MODE] = self.mode;
        self.telem[ELASTO_TELEM_REASON_MASK] = reason;
        let (gw, gh) = if self.mode == ELASTO_MODE_ACOUSTIC_BOS {
            (BOS_GRID_W as u32, BOS_GRID_H as u32)
        } else {
            (ELASTO_W as u32, ELASTO_H as u32)
        };
        self.telem[ELASTO_TELEM_GRID_W] = gw;
        self.telem[ELASTO_TELEM_GRID_H] = gh;
    }

    /// Main per-frame post-tick entry point. Returns `true` when fresh telemetry/planes were published.
    #[inline(never)]
    pub fn step_frame(
        &mut self,
        luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        frame_t_sec: f64,
        meta: &VibroFrameMeta,
        vibro: &mut VibrometryState,
        bos: &mut BosState,
        bos_cache: &mut BosRefCache,
        hodge: &mut HodgeState,
        has_scratch: bool,
    ) -> bool {
        self.fresh_solve = 0;
        if self.mode == ELASTO_MODE_OFF {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_OFF;
            self.telem[ELASTO_TELEM_MODE] = ELASTO_MODE_OFF;
            self.live_ready = 0;
            return false;
        }
        if !has_scratch {
            self.telem.fill(0);
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_REFUSED;
            self.telem[ELASTO_TELEM_MODE] = self.mode;
            self.telem[ELASTO_TELEM_REASON_MASK] = ELASTO_REASON_MUTEX_CONFLICT;
            self.live_ready = 0;
            return false;
        }
        if frame_w < 64
            || frame_h < 36
            || frame_w > crate::MAX_FRAME_W
            || frame_h > crate::MAX_FRAME_H
            || luma.len() < frame_w * frame_h
        {
            self.fail_refused(RS_REASON_NOT_NATIVE_1TO1, bos, true);
            return false;
        }

        // Continuity / geometry / camera-motion guard
        let geom_changed = self.last_frame_w != 0
            && (self.last_frame_w as usize != frame_w
                || self.last_frame_h as usize != frame_h
                || self.last_orientation != meta.orientation
                || self.last_scan_dir != meta.scan_dir);
        let hard_gap = (meta.gap_flag & !0x40u32) != 0;
        let epoch_changed =
            self.ref_warm_frames > 0 && meta.continuity_epoch != self.last_continuity_epoch;
        if geom_changed || hard_gap || epoch_changed {
            self.reset_runtime(if hard_gap { RS_REASON_FRAME_GAP } else { 0 });
            bos.clear_solver_scratch();
            bos.planes.fill(0.0);
            bos.valid_mask.fill(0);
            bos.ref_noise_var.fill(0.0);
            bos.w_x.fill(0.0);
            bos.w_y.fill(0.0);
        }
        self.last_frame_w = frame_w as u16;
        self.last_frame_h = frame_h as u16;
        self.last_orientation = meta.orientation;
        self.last_scan_dir = meta.scan_dir;
        self.last_continuity_epoch = meta.continuity_epoch;

        if meta.cam_motion_state != 0 && self.mode != ELASTO_MODE_SURFACE_FRAMERATE {
            self.reset_runtime(RS_REASON_CAMERA_MOTION);
            bos.clear_solver_scratch();
            bos.planes.fill(0.0);
            bos.valid_mask.fill(0);
            return false;
        }

        match self.mode {
            ELASTO_MODE_SURFACE_RS_AUDIO | ELASTO_MODE_SURFACE_RS_SELF => self.step_surface_rs(
                luma,
                frame_w,
                frame_h,
                meta,
                vibro,
                bos,
                hodge,
                self.mode == ELASTO_MODE_SURFACE_RS_AUDIO,
            ),
            ELASTO_MODE_SURFACE_FRAMERATE => {
                self.step_surface_framerate(luma, frame_w, frame_h, frame_t_sec, bos, hodge)
            }
            ELASTO_MODE_ACOUSTIC_BOS => {
                self.step_acoustic_bos(luma, frame_w, frame_h, meta, vibro, bos, bos_cache)
            }
            _ => false,
        }
    }

    /// Validates rolling-shutter calibration & D4 sensor scan-line mapping `(q_start, q_step, scan_along_ey)`.
    fn validate_rs_geometry(
        &self,
        frame_w: usize,
        frame_h: usize,
        meta: &VibroFrameMeta,
        vibro: &VibrometryState,
    ) -> Result<(f64, f64, f64, f64, bool), u32> {
        let mut rsn = 0u32;
        if vibro.calib_valid == 0 || !vibro.dt_row_us.is_finite() || vibro.dt_row_us <= 0.0 {
            rsn |= RS_REASON_BAD_CALIB;
        }
        if vibro.skew_disturbed != 0 {
            rsn |= RS_REASON_CLOCK_SKEW;
        }
        if vibro.rate_disturbed != 0 {
            rsn |= RS_REASON_BAD_CALIB;
        }
        if (meta.flags & 2) == 0 {
            rsn |= RS_REASON_NO_PTS;
        }
        if meta.native_1to1 != 1 {
            rsn |= RS_REASON_NOT_NATIVE_1TO1;
        }
        if meta.scan_dir > 3 {
            rsn |= RS_REASON_NO_SCAN_DIR;
        }
        let crop_ok = meta.orientation <= 7
            && meta.src_w > 0
            && meta.src_h > 0
            && meta.crop_w > 0
            && meta.crop_h > 0
            && meta.crop_x0.checked_add(meta.crop_w).map_or(false, |v| v <= meta.src_w)
            && meta.crop_y0.checked_add(meta.crop_h).map_or(false, |v| v <= meta.src_h)
            && oriented_extent(meta.crop_w, meta.crop_h, meta.orientation)
                == (frame_w as u32, frame_h as u32);
        if !crop_ok {
            rsn |= RS_REASON_NOT_NATIVE_1TO1;
        }
        if rsn != 0 {
            return Err(rsn);
        }

        let ow = frame_w as u32;
        let o = meta.orientation & 7;
        let (sx_tl, sy_tl) =
            engine_to_sensor_xy(0, 0, ow, meta.crop_w, meta.crop_h, meta.crop_x0, meta.crop_y0, o);
        let (sx_dx, sy_dx) =
            engine_to_sensor_xy(1, 0, ow, meta.crop_w, meta.crop_h, meta.crop_x0, meta.crop_y0, o);
        let (sx_dy, sy_dy) =
            engine_to_sensor_xy(0, 1, ow, meta.crop_w, meta.crop_h, meta.crop_x0, meta.crop_y0, o);

        let (q_tl, dq_ex, dq_ey) = match meta.scan_dir {
            0 => (
                sy_tl as i64,
                (sy_dx as i64) - (sy_tl as i64),
                (sy_dy as i64) - (sy_tl as i64),
            ),
            1 => (
                (meta.src_h - 1 - sy_tl) as i64,
                (sy_tl as i64) - (sy_dx as i64),
                (sy_tl as i64) - (sy_dy as i64),
            ),
            2 => (
                sx_tl as i64,
                (sx_dx as i64) - (sx_tl as i64),
                (sx_dy as i64) - (sx_tl as i64),
            ),
            _ => (
                (meta.src_w - 1 - sx_tl) as i64,
                (sx_tl as i64) - (sx_dx as i64),
                (sx_tl as i64) - (sx_dy as i64),
            ),
        };
        if dq_ey != 0 && dq_ex == 0 {
            Ok((
                vibro.dt_row_us as f64,
                vibro.exposure_us as f64,
                q_tl as f64,
                dq_ey as f64,
                true,
            ))
        } else if dq_ex != 0 && dq_ey == 0 {
            Ok((
                vibro.dt_row_us as f64,
                vibro.exposure_us as f64,
                q_tl as f64,
                dq_ex as f64,
                false,
            ))
        } else {
            Err(RS_REASON_AXIS_TRANSPOSED)
        }
    }

    /// Stage 1 + 2A + 3A: `64x36` 2D Rolling-Shutter Surface Elastography & Hodge Power Flow
    /// (Mode 1 `SURFACE_RS_AUDIO` and Mode 2 `SURFACE_RS_SELF`).
    #[inline(never)]
    fn step_surface_rs(
        &mut self,
        luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        meta: &VibroFrameMeta,
        vibro: &mut VibrometryState,
        bos: &mut BosState,
        hodge: &mut HodgeState,
        use_audio: bool,
    ) -> bool {
        let (dt_row_us, exposure_us, q_start, q_step, scan_along_ey) =
            match self.validate_rs_geometry(frame_w, frame_h, meta, vibro) {
                Ok(g) => g,
                Err(rsn) => {
                    self.telem.fill(0);
                    self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_UNCALIBRATED;
                    self.telem[ELASTO_TELEM_MODE] = self.mode;
                    self.telem[ELASTO_TELEM_REASON_MASK] = rsn;
                    self.telem[ELASTO_TELEM_GRID_W] = ELASTO_W as u32;
                    self.telem[ELASTO_TELEM_GRID_H] = ELASTO_H as u32;
                    bos.planes.fill(0.0);
                    bos.valid_mask.fill(0);
                    return false;
                }
            };
        if !scan_along_ey {
            self.fail_refused(RS_REASON_AXIS_TRANSPOSED, bos, true);
            return false;
        }

        let need_px = frame_w * frame_h;
        // 1. 16-frame static reference warmup into `bos.ref_luma`
        if self.ref_warm_frames < REF_WARM_FRAMES {
            if self.ref_warm_frames == 0 {
                bos.ref_luma[..need_px].copy_from_slice(&luma[..need_px]);
                bos.clear_solver_scratch();
                bos.planes.fill(0.0);
                bos.valid_mask.fill(0);
                self.t0_us = meta.t_src_us;
            } else {
                let alpha = 1.0f32 / ((self.ref_warm_frames as f32) + 1.0);
                let mut p = 0usize;
                while p < need_px {
                    bos.ref_luma[p] += (luma[p] - bos.ref_luma[p]) * alpha;
                    p += 1;
                }
            }
            bos.bump_ref_gen();
            self.ref_warm_frames += 1;
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
            self.telem[ELASTO_TELEM_MODE] = self.mode;
            self.telem[ELASTO_TELEM_REASON_MASK] = 0;
            self.telem[ELASTO_TELEM_GRID_W] = ELASTO_W as u32;
            self.telem[ELASTO_TELEM_GRID_H] = ELASTO_H as u32;
            return false;
        }

        // 2. Stage 1: Doppler-free frequency lock (explicit probe freqs OR audio Welch PSD peaks)
        let f_opt_max = 0.48 * (1_000_000.0 / dt_row_us);
        let mut locked = [0.0f64; ELASTO_MAX_FREQS];
        let mut n_locked = 0usize;
        if self.has_explicit_freqs != 0 && self.explicit_freq_count > 0 {
            let cnt = (self.explicit_freq_count as usize).min(ELASTO_MAX_FREQS);
            let mut i = 0usize;
            while i < cnt {
                let f = self.explicit_freqs[i] as f64;
                if f > 0.1 && f <= f_opt_max {
                    locked[n_locked] = f;
                    n_locked += 1;
                }
                i += 1;
            }
        } else if use_audio {
            n_locked = vibro.detect_top_audio_peaks(f_opt_max, &mut locked);
        } else {
            // Mode 2 without explicit frequencies: cannot lock without a reference frequency yet
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
            self.telem[ELASTO_TELEM_REASON_MASK] = RS_REASON_BAD_CALIB;
            return false;
        }

        if n_locked == 0 {
            self.telem[ELASTO_TELEM_STATE] = if use_audio {
                ELASTO_STATE_WARMING
            } else {
                ELASTO_STATE_REFUSED
            };
            self.telem[ELASTO_TELEM_REASON_MASK] = if use_audio {
                RS_REASON_AUDIO_UNDERRUN
            } else {
                RS_REASON_BAD_CALIB
            };
            return false;
        }

        // Exposure sinc guard on primary frequency `f0` (and keep all locked freqs with |H(f)| >= 0.05)
        let h0_sinc = sinc_exposure_h(locked[0], exposure_us);
        if abs_f64(h0_sinc) < 0.05 {
            self.fail_refused(RS_REASON_EXPOSURE_NULL, bos, true);
            self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(locked[0]);
            self.telem[ELASTO_TELEM_SINC_H0] = pack_f32(h0_sinc);
            return false;
        }
        // Filter secondary tones against exposure nulls
        let mut keep_n = 1usize;
        let mut fi = 1usize;
        while fi < n_locked {
            if abs_f64(sinc_exposure_h(locked[fi], exposure_us)) >= 0.05 {
                locked[keep_n] = locked[fi];
                keep_n += 1;
            }
            fi += 1;
        }
        n_locked = keep_n;

        // If locked frequency set changed significantly (> 3 Hz), reset Welch segment ring
        let mut freqs_changed = self.active_freq_count as usize != n_locked;
        if !freqs_changed {
            let mut m = 0usize;
            while m < n_locked {
                if abs_f64(self.active_freqs[m] - locked[m]) > 3.0 {
                    freqs_changed = true;
                    break;
                }
                m += 1;
            }
        }
        if freqs_changed {
            self.active_freq_count = n_locked as u32;
            self.active_freqs = locked;
            self.welch_head = 0;
            self.welch_count = 0;
            self.seg_frame_acc = 0;
            bos.clear_solver_scratch();
        }
        let m_cnt = self.active_freq_count as usize;
        let dim = 2 * m_cnt;

        // 3. Per-frame audio unit reference phasors `(x_re[m], x_im[m])` anchored at `meta.t_src_us`
        let mut x_re = [0.0f64; ELASTO_MAX_FREQS];
        let mut x_im = [0.0f64; ELASTO_MAX_FREQS];
        if use_audio {
            let mut m = 0usize;
            while m < m_cnt {
                match vibro.audio_phasor_at(self.active_freqs[m], meta.t_src_us) {
                    Some((ur, ui, _pxx)) => {
                        x_re[m] = ur;
                        x_im[m] = ui;
                    }
                    None => {
                        self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
                        self.telem[ELASTO_TELEM_REASON_MASK] = RS_REASON_AUDIO_UNDERRUN;
                        return false;
                    }
                }
                m += 1;
            }
        } else {
            let rel_sec = rel_us_f64(meta.t_src_us, self.t0_us) * 1.0e-6;
            let mut m = 0usize;
            while m < m_cnt {
                let (si, co) = sin_cos_f64(TWO_PI_64 * self.active_freqs[m] * rel_sec);
                x_re[m] = co;
                x_im[m] = si;
                m += 1;
            }
        }

        // 4. Zero current segment normal equations `(G, rhs)` at start of each `SEG_FRAMES` chunk
        if self.seg_frame_acc == 0 {
            bos.solve_lp[LP_G_OFF..LP_RHS_OFF + ELASTO_V * 8].fill(0.0);
        }

        // Lock global dominant displacement projection axis `(global_ax_x, global_ax_y)` on first active frame
        if self.global_ax_locked == 0 {
            let mut sxx = 0.0f64;
            let mut sxy = 0.0f64;
            let mut syy = 0.0f64;
            let mut r = 1usize;
            while r + 1 < ELASTO_H {
                let yc = ((r as f64) + 0.5) * (frame_h as f64) / (ELASTO_H as f64);
                let y0 = (yc as usize).saturating_sub(2).max(1).min(frame_h - 5);
                let mut c = 1usize;
                while c + 1 < ELASTO_W {
                    let xc = ((c as f64) + 0.5) * (frame_w as f64) / (ELASTO_W as f64);
                    let x0 = (xc as usize).saturating_sub(4).max(1).min(frame_w - 10);
                    if let Some((ux, uy)) = eval_strip_iclk(luma, &bos.ref_luma, frame_w, frame_h, x0, y0, 8, 4) {
                        sxx += ux * ux;
                        sxy += ux * uy;
                        syy += uy * uy;
                    }
                    c += 1;
                }
                r += 1;
            }
            let tr = sxx + syy;
            if tr > 1.0e-4 {
                let diff = sxx - syy;
                let disc = wasm_sqrt_f64(diff * diff + 4.0 * sxy * sxy);
                let lam1 = 0.5 * (tr + disc);
                let (mut vx, mut vy) = if sxx >= syy {
                    (lam1 - syy, sxy)
                } else {
                    (sxy, lam1 - sxx)
                };
                let norm = wasm_sqrt_f64(vx * vx + vy * vy);
                if norm > 1.0e-9 {
                    vx /= norm;
                    vy /= norm;
                    if vx < -1.0e-4 || (abs_f64(vx) <= 1.0e-4 && vy < 0.0) {
                        vx = -vx;
                        vy = -vy;
                    }
                    self.global_ax_x = vx;
                    self.global_ax_y = vy;
                    self.global_ax_locked = 1;
                }
            }
        }
        let ax_x = self.global_ax_x;
        let ax_y = self.global_ax_y;

        // 5. Sub-cell `16x4` 90-band IC-LK + symmetric +-16-row (`1.12 ms`) multi-tone demodulation:
        //    First evaluate `64 x 90` (`5,760`) 64-pixel affine IC-LK patches into `bos.w_x` / `bos.clear_streak`,
        //    then pool `9` symmetric bands (`+-4` bands = `+-16` rows) into each `64x36` cell with
        //    even flat-top weights `[0.5, 1, 1, 1, 1, 1, 1, 1, 0.5]`. Because the pooling window is symmetric
        //    around cell center `yc`, spatial `sin(ky * dy)` cancels identically (zero phase bias) while
        //    `1.12 ms` of rolling-shutter readout makes multi-tone carriers orthogonal even when
        //    `(f2 - f1) mod fps == 0` or `f mod (fps/2) == 0`!
        const N_BANDS: usize = 90;
        const POOL_R: isize = 4;
        const BAND_W: [f64; 9] = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5];
        let cell_w_f = (frame_w as f64) / (ELASTO_W as f64);
        let cell_h_f = (frame_h as f64) / (ELASTO_H as f64);
        let band_h_f = (frame_h as f64) / (N_BANDS as f64);
        let patch_h = (band_h_f as usize).clamp(2, 4);
        let patch_w = 16usize.min(frame_w.saturating_sub(4));
        let dt_row_sec = dt_row_us * 1.0e-6;

        // 5a. Compute 64x90 scalar projected displacements into `bos.w_x[0..5760]` & `bos.clear_streak[0..5760]`
        let mut band_ymid = [0.0f64; N_BANDS];
        let mut b = 0usize;
        while b < N_BANDS {
            let yc_b = ((b as f64) + 0.5) * band_h_f;
            let y0 = ((yc_b - 0.5 * (patch_h as f64)) as isize)
                .clamp(1, (frame_h - patch_h - 2) as isize) as usize;
            band_ymid[b] = (y0 as f64) + 0.5 * ((patch_h - 1) as f64);
            let row_off = b * ELASTO_W;
            let mut c = 0usize;
            while c < ELASTO_W {
                let xc = ((c as f64) + 0.5) * cell_w_f;
                let x0 = ((xc - 0.5 * (patch_w as f64)) as isize)
                    .clamp(1, (frame_w - patch_w - 2) as isize) as usize;
                if let Some((ux, uy)) =
                    eval_strip_iclk(luma, &bos.ref_luma, frame_w, frame_h, x0, y0, patch_w, patch_h)
                {
                    bos.w_x[row_off + c] = (ux * ax_x + uy * ax_y) as f32;
                    bos.clear_streak[row_off + c] = 1;
                } else {
                    bos.w_x[row_off + c] = 0.0;
                    bos.clear_streak[row_off + c] = 0;
                }
                c += 1;
            }
            b += 1;
        }

        // 5b. Zero per-cell valid weight tracker `bos.w_y[0..2304]` at segment start
        if self.seg_frame_acc == 0 {
            bos.w_y[0..ELASTO_V].fill(0.0);
        }

        // 5c. Accumulate row-shared `G[r]` and per-cell `rhs[v]` over the symmetric 9-band window
        let mut r = 0usize;
        while r < ELASTO_H {
            let yc = ((r as f64) + 0.5) * cell_h_f;
            let b_center = ((yc / band_h_f) - 0.5) as isize;
            let g_off = LP_G_OFF + r * 64;

            let mut k = 0usize;
            while k < 9 {
                let bi = b_center + (k as isize) - POOL_R;
                if bi >= 0 && bi < (N_BANDS as isize) {
                    let bu = bi as usize;
                    let w = BAND_W[k];
                    let q_row = q_start + band_ymid[bu] * q_step;
                    let t_scan = q_row * dt_row_sec;

                    let mut basis = [0.0f64; 8];
                    let mut m = 0usize;
                    while m < m_cnt {
                        let (si, co) = sin_cos_f64(TWO_PI_64 * self.active_freqs[m] * t_scan);
                        basis[2 * m] = x_re[m] * co - x_im[m] * si;
                        basis[2 * m + 1] = x_re[m] * si + x_im[m] * co;
                        m += 1;
                    }

                    let mut a = 0usize;
                    while a < dim {
                        let wba = w * basis[a];
                        let mut bb = a;
                        while bb < dim {
                            bos.solve_lp[g_off + a * dim + bb] += wba * basis[bb];
                            bb += 1;
                        }
                        a += 1;
                    }

                    let band_off = bu * ELASTO_W;
                    let mut c = 0usize;
                    while c < ELASTO_W {
                        if bos.clear_streak[band_off + c] != 0 {
                            let wu = w * (bos.w_x[band_off + c] as f64);
                            let v = r * ELASTO_W + c;
                            let rhs_off = LP_RHS_OFF + v * 8;
                            let mut a2 = 0usize;
                            while a2 < dim {
                                bos.solve_lp[rhs_off + a2] += basis[a2] * wu;
                                a2 += 1;
                            }
                            bos.w_y[v] += w as f32;
                        }
                        c += 1;
                    }
                }
                k += 1;
            }
            r += 1;
        }

        self.seg_frame_acc += 1;
        if self.seg_frame_acc < SEG_FRAMES {
            if self.live_ready == 0 {
                self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
                self.telem[ELASTO_TELEM_WELCH_SEGS] = self.welch_count;
            }
            return false;
        }
        self.seg_frame_acc = 0;

        // 6. Complete Welch segment: factor `G[r]` (<= 8x8 Cholesky) once per row and solve all `2,304` cells
        let seg = (self.welch_head as usize) % K_SEGS;
        let mut h_inv = [1.0f64; ELASTO_MAX_FREQS];
        let mut m = 0usize;
        while m < m_cnt {
            let hm = sinc_exposure_h(self.active_freqs[m], exposure_us);
            h_inv[m] = if abs_f64(hm) >= 0.05 { 1.0 / hm } else { 0.0 };
            m += 1;
        }

        r = 0;
        while r < ELASTO_H {
            let yc = ((r as f64) + 0.5) * cell_h_f;
            let b_center = ((yc / band_h_f) - 0.5) as isize;
            let mut row_w_max = 0.0f64;
            let mut k0 = 0usize;
            while k0 < 9 {
                let bi = b_center + (k0 as isize) - POOL_R;
                if bi >= 0 && bi < (N_BANDS as isize) {
                    row_w_max += BAND_W[k0];
                }
                k0 += 1;
            }
            row_w_max *= SEG_FRAMES as f64;

            let g_off = LP_G_OFF + r * 64;
            let mut g_mat = [0.0f64; 64];
            let mut a = 0usize;
            while a < dim {
                let mut bb = a;
                while bb < dim {
                    let val = bos.solve_lp[g_off + a * dim + bb];
                    g_mat[a * dim + bb] = val;
                    g_mat[bb * dim + a] = val;
                    bb += 1;
                }
                g_mat[a * dim + a] += 1.0e-5 * row_w_max.max(1.0);
                a += 1;
            }
            let mut chol = [0.0f64; 64];
            let chol_ok = cholesky_factor(&g_mat, &mut chol, dim);

            let mut c = 0usize;
            while c < ELASTO_W {
                let v = r * ELASTO_W + c;
                let w_got = bos.w_y[v] as f64;
                if chol_ok && w_got >= 0.70 * row_w_max {
                    let w_fix = row_w_max / w_got;
                    let rhs_off = LP_RHS_OFF + v * 8;
                    let mut rhs = [0.0f64; 8];
                    let mut k = 0usize;
                    while k < dim {
                        rhs[k] = bos.solve_lp[rhs_off + k] * w_fix;
                        k += 1;
                    }
                    let mut sol = [0.0f64; 8];
                    cholesky_solve(&chol, &rhs, &mut sol, dim);
                    let mut mm = 0usize;
                    while mm < m_cnt {
                        let scale = h_inv[mm];
                        let re = sol[2 * mm] * scale;
                        let im = sol[2 * mm + 1] * scale;
                        seg_ring_write(bos, seg_ring_idx(seg, mm, 0, v), if re.is_finite() { re } else { 0.0 });
                        seg_ring_write(bos, seg_ring_idx(seg, mm, 1, v), if im.is_finite() { im } else { 0.0 });
                        mm += 1;
                    }
                } else {
                    let mut mm = 0usize;
                    while mm < m_cnt {
                        seg_ring_write(bos, seg_ring_idx(seg, mm, 0, v), 0.0);
                        seg_ring_write(bos, seg_ring_idx(seg, mm, 1, v), 0.0);
                        mm += 1;
                    }
                }
                c += 1;
            }
            r += 1;
        }

        self.welch_head = ((seg + 1) % K_SEGS) as u32;
        if (self.welch_count as usize) < K_SEGS {
            self.welch_count += 1;
        }

        if (self.welch_count as usize) < K_SEGS {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
            self.telem[ELASTO_TELEM_WELCH_SEGS] = self.welch_count;
            self.telem[ELASTO_TELEM_LOCKED_FREQS] = m_cnt as u32;
            self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(self.active_freqs[0]);
            if m_cnt >= 2 {
                self.telem[ELASTO_TELEM_F1_HZ] = pack_f32(self.active_freqs[1]);
            }
            return false;
        }

        // 7. Stage 3A: Average complex wavefields across `K_SEGS = 8`, compute coherence,
        //    wrap-free Poynting power flow, Hodge source/sink decomposition, wavenumber & dispersion!
        self.solve_surface_stage3a(frame_w, frame_h, h0_sinc, bos, hodge)
    }

    /// Stage 2A + 3A for Mode 3 (`SURFACE_FRAMERATE`): ordinary 30/60 fps or slow-motion videos
    /// (water ripples, Faraday waves, Chladni plates, laboratory OCE shear waves) without RS calibration!
    #[inline(never)]
    fn step_surface_framerate(
        &mut self,
        luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        frame_t_sec: f64,
        bos: &mut BosState,
        hodge: &mut HodgeState,
    ) -> bool {
        let cell_w = (frame_w / ELASTO_W).max(1);
        let cell_h = (frame_h / ELASTO_H).max(1);
        let inv_area = 1.0f32 / ((cell_w * cell_h) as f32);

        let slot = (self.fr_hist_head as usize) % FRAMERATE_HIST_LEN;
        let is_first = self.fr_hist_count == 0;
        let mut r = 0usize;
        while r < ELASTO_H {
            let y0 = r * cell_h;
            let mut c = 0usize;
            while c < ELASTO_W {
                let x0 = c * cell_w;
                let mut sum = 0.0f32;
                let mut dy = 0usize;
                while dy < cell_h {
                    let row = (y0 + dy) * frame_w + x0;
                    let mut dx = 0usize;
                    while dx < cell_w {
                        sum += luma[row + dx];
                        dx += 1;
                    }
                    dy += 1;
                }
                let mean = sum * inv_area;
                let v = r * ELASTO_W + c;
                if is_first {
                    bos.ref_noise_var[v] = mean;
                } else {
                    bos.ref_noise_var[v] += (mean - bos.ref_noise_var[v]) * (1.0 / 24.0);
                }
                fr_ring_write(bos, slot, v, mean - bos.ref_noise_var[v]);
                c += 1;
            }
            r += 1;
        }
        self.fr_hist_t[slot] = frame_t_sec;
        self.fr_hist_head = ((slot + 1) % FRAMERATE_HIST_LEN) as u32;
        if (self.fr_hist_count as usize) < FRAMERATE_HIST_LEN {
            self.fr_hist_count += 1;
        }

        if (self.fr_hist_count as usize) < FRAMERATE_HIST_LEN {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
            self.telem[ELASTO_TELEM_MODE] = self.mode;
            self.telem[ELASTO_TELEM_GRID_W] = ELASTO_W as u32;
            self.telem[ELASTO_TELEM_GRID_H] = ELASTO_H as u32;
            return false;
        }

        self.seg_frame_acc += 1;
        if self.live_ready != 0 && self.seg_frame_acc < 4 {
            return false;
        }
        self.seg_frame_acc = 0;

        // Compute average dt across the 24-frame ring
        let oldest = self.fr_hist_head as usize % FRAMERATE_HIST_LEN;
        let newest = (oldest + FRAMERATE_HIST_LEN - 1) % FRAMERATE_HIST_LEN;
        let span = self.fr_hist_t[newest] - self.fr_hist_t[oldest];
        if !span.is_finite() || span <= 1.0e-3 {
            self.fail_refused(RS_REASON_FRAME_GAP, bos, true);
            return false;
        }
        let dt = span / ((FRAMERATE_HIST_LEN - 1) as f64);
        let fps = 1.0 / dt;

        // Pick dominant temporal DFT bin `k in 2..=10` (or explicit lock freq)
        let mut best_k = 2usize;
        let mut best_pow = -1.0f64;
        if self.has_explicit_freqs != 0 && self.explicit_freq_count > 0 {
            let f0 = (self.explicit_freqs[0] as f64).clamp(0.5, 0.45 * fps);
            self.active_freqs[0] = f0;
            self.active_freq_count = 1;
        } else {
            let mut k = 2usize;
            while k <= 10 {
                let mut cos_w = [0.0f64; FRAMERATE_HIST_LEN];
                let mut sin_w = [0.0f64; FRAMERATE_HIST_LEN];
                let mut n = 0usize;
                while n < FRAMERATE_HIST_LEN {
                    let hann = 0.5
                        * (1.0
                            - sin_cos_f64(TWO_PI_64 * (n as f64) / ((FRAMERATE_HIST_LEN - 1) as f64)).1);
                    let (si, co) =
                        sin_cos_f64(TWO_PI_64 * (k as f64) * (n as f64) / (FRAMERATE_HIST_LEN as f64));
                    cos_w[n] = hann * co;
                    sin_w[n] = hann * si;
                    n += 1;
                }
                let mut tot_p = 0.0f64;
                let mut v = 0usize;
                while v < ELASTO_V {
                    let mut re = 0.0f64;
                    let mut im = 0.0f64;
                    let mut n2 = 0usize;
                    while n2 < FRAMERATE_HIST_LEN {
                        let sl = (oldest + n2) % FRAMERATE_HIST_LEN;
                        let val = fr_ring_read(bos, sl, v);
                        re += val * cos_w[n2];
                        im += val * sin_w[n2];
                        n2 += 1;
                    }
                    tot_p += re * re + im * im;
                    v += 4; // stride 4 for fast peak search
                }
                if tot_p > best_pow {
                    best_pow = tot_p;
                    best_k = k;
                }
                k += 1;
            }
            self.active_freqs[0] = (best_k as f64) * fps / (FRAMERATE_HIST_LEN as f64);
            self.active_freq_count = 1;
        }

        // Compute `K_SEGS = 8` overlapping 16-frame sub-window complex phasors into the segment ring
        // so Stage 3A coherence & elastography pipeline is 100% shared!
        let f0 = self.active_freqs[0];
        const SUB_WIN: usize = 16;
        let mut seg = 0usize;
        while seg < K_SEGS {
            let mut cos_w = [0.0f64; SUB_WIN];
            let mut sin_w = [0.0f64; SUB_WIN];
            let mut n = 0usize;
            while n < SUB_WIN {
                let sl = (oldest + seg + n) % FRAMERATE_HIST_LEN;
                let t_rel = self.fr_hist_t[sl] - self.fr_hist_t[oldest];
                let hann = 0.5 * (1.0 - sin_cos_f64(TWO_PI_64 * (n as f64) / ((SUB_WIN - 1) as f64)).1);
                let (si, co) = sin_cos_f64(TWO_PI_64 * f0 * t_rel);
                cos_w[n] = hann * co;
                sin_w[n] = hann * si;
                n += 1;
            }
            let mut v = 0usize;
            while v < ELASTO_V {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                let mut n2 = 0usize;
                while n2 < SUB_WIN {
                    let sl = (oldest + seg + n2) % FRAMERATE_HIST_LEN;
                    let val = fr_ring_read(bos, sl, v);
                    re += val * cos_w[n2];
                    im += val * sin_w[n2];
                    n2 += 1;
                }
                seg_ring_write(bos, seg_ring_idx(seg, 0, 0, v), re * (2.0 / (SUB_WIN as f64)));
                seg_ring_write(bos, seg_ring_idx(seg, 0, 1, v), im * (2.0 / (SUB_WIN as f64)));
                v += 1;
            }
            seg += 1;
        }
        self.welch_count = K_SEGS as u32;
        self.solve_surface_stage3a(frame_w, frame_h, 1.0, bos, hodge)
    }

    /// Shared Stage 3A for all `64x36` Surface Modes (`1, 2, 3`):
    /// Complex cross-segment averaging -> coherence gating -> wrap-free Poynting wavevector +
    /// weak-form `9x9` Helmholtz wavenumber -> Hodge mechanical power flow (sources/sinks/beta1) ->
    /// constitutive dispersion classification (`KIRCHHOFF_PLATE` vs `MEMBRANE_SHEAR`).
    #[inline(never)]
    fn solve_surface_stage3a(
        &mut self,
        frame_w: usize,
        frame_h: usize,
        h0_sinc: f64,
        bos: &mut BosState,
        hodge: &mut HodgeState,
    ) -> bool {
        let m_cnt = (self.active_freq_count as usize).clamp(1, ELASTO_MAX_FREQS);
        let dx = (frame_w as f64) / (ELASTO_W as f64);
        let dy = (frame_h as f64) / (ELASTO_H as f64);
        let inv_k = 1.0 / (K_SEGS as f64);
        // Critical cross-segment coherence threshold (`gamma2_crit = 0.45` for `K = 8`)
        let coh_crit = 0.45f64;

        // Compute complex mean `u_tilde` and cross-segment phase coherence for frequency `m = 0`
        // (and store `k_eff` per cell for `m = 0` into `LP_K0_OFF`, and for `m = 1` into `LP_K1_OFF`)
        let mut max_amp0 = 0.0f64;
        let mut v = 0usize;
        while v < ELASTO_V {
            let mut sr = 0.0f64;
            let mut si = 0.0f64;
            let mut sp = 0.0f64;
            let mut s = 0usize;
            while s < K_SEGS {
                let re = seg_ring_read(bos, seg_ring_idx(s, 0, 0, v));
                let im = seg_ring_read(bos, seg_ring_idx(s, 0, 1, v));
                sr += re;
                si += im;
                sp += re * re + im * im;
                s += 1;
            }
            let mr = sr * inv_k;
            let mi = si * inv_k;
            let m2 = mr * mr + mi * mi;
            let mp = sp * inv_k;
            let coh = if mp > 1.0e-14 { (m2 / mp).clamp(0.0, 1.0) } else { 0.0 };
            let amp = wasm_sqrt_f64(m2);
            if amp > max_amp0 {
                max_amp0 = amp;
            }
            bos.solve_z[Z_URE_OFF + v] = mr;
            bos.solve_z[Z_UIM_OFF + v] = mi;
            bos.solve_z[Z_COH_OFF + v] = coh;
            v += 1;
        }

        // Mark valid coherent cells (`coh >= coh_crit` AND amplitude `>= 0.06 * max_amp0` so zero-amplitude
        // cavities like a guitar soundhole cleanly form `beta1` topological holes!)
        let amp_floor = (0.06 * max_amp0).max(1.0e-4);
        let mut valid_cnt = 0usize;
        let mut coh_sum = 0.0f64;
        v = 0;
        while v < ELASTO_V {
            let mr = bos.solve_z[Z_URE_OFF + v];
            let mi = bos.solve_z[Z_UIM_OFF + v];
            let amp = wasm_sqrt_f64(mr * mr + mi * mi);
            let coh = bos.solve_z[Z_COH_OFF + v];
            if coh >= coh_crit && amp >= amp_floor {
                bos.valid_mask[v] = 1;
                valid_cnt += 1;
                coh_sum += coh;
            } else {
                bos.valid_mask[v] = 0;
            }
            v += 1;
        }

        if valid_cnt < 48 {
            self.fail_refused(ELASTO_REASON_LOW_COH_COVERAGE, bos, true);
            self.telem[ELASTO_TELEM_LOCKED_FREQS] = m_cnt as u32;
            self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(self.active_freqs[0]);
            self.telem[ELASTO_TELEM_WELCH_SEGS] = self.welch_count;
            return false;
        }

        // Compute wrap-free Poynting power flow `(Ix, Iy)`, wavevector `(kx, ky)`, and wavenumber `k0`
        // for primary frequency `f0`
        let omega0 = TWO_PI_64 * self.active_freqs[0];
        let (alias_med_rad, standing_frac, aniso_ratio) = compute_wavefield_poynting_and_k(
            bos,
            omega0,
            dx,
            dy,
            LP_K0_OFF,
            true,
        );

        // Run Hodge decomposition on mechanical power flow `(Ix, Iy)` over active coherent mask!
        // Uses disjoint slices of `bos.solve_z` and `bos.valid_mask` (zero stack arrays!).
        let (hodge_ok, beta1_holes, exact_frac, coexact_frac) = {
            let (left, right) = bos.solve_z.split_at_mut(Z_PHI_OFF);
            let ix_sl = &left[Z_IX_OFF..Z_IX_OFF + ELASTO_V];
            let iy_sl = &left[Z_IY_OFF..Z_IY_OFF + ELASTO_V];
            let (phi_sl, div_rest) = right.split_at_mut(ELASTO_V);
            let div_sl = &mut div_rest[0..ELASTO_V];
            hodge.solve_power_flow_hodge(
                ix_sl,
                iy_sl,
                &bos.valid_mask[0..ELASTO_V],
                dx,
                dy,
                phi_sl,
                div_sl,
            )
        };
        let _ = hodge_ok;

        // Smooth interior divergence `div(I)` slightly (`3x3` Gaussian) to localize exciter source (+)
        // and crack/damping sink (-) cleanly inside the active interior (ignoring open frame border edges!)
        let (src_x, src_y, sink_x, sink_y) = localize_power_source_sink(bos, dx, dy);

        // Multi-frequency dispersion exponent `alpha = median(ln(k_1 / k_0) / ln(f_1 / f_0))`
        let mut disp_alpha = 0.0f64;
        let mut class_code = ELASTO_CLASS_NONE;
        if m_cnt >= 2 && self.active_freqs[1] > self.active_freqs[0] * 1.15 {
            // Save primary `u_re, u_im, coh` into `bos.planes` first!
            v = 0;
            while v < ELASTO_V {
                let mr = bos.solve_z[Z_URE_OFF + v];
                let mi = bos.solve_z[Z_UIM_OFF + v];
                let amp = wasm_sqrt_f64(mr * mr + mi * mi);
                bos.planes[ELASTO_SURF_PLANE_AMP * ELASTO_V + v] =
                    if bos.valid_mask[v] != 0 { amp as f32 } else { 0.0 };
                bos.planes[ELASTO_SURF_PLANE_PHASE * ELASTO_V + v] = if bos.valid_mask[v] != 0 {
                    fast_atan2f(mi as f32, mr as f32)
                } else {
                    0.0
                };
                bos.planes[ELASTO_SURF_PLANE_COH * ELASTO_V + v] = bos.solve_z[Z_COH_OFF + v] as f32;
                bos.planes[ELASTO_SURF_PLANE_DIV * ELASTO_V + v] =
                    if bos.valid_mask[v] != 0 { bos.solve_z[Z_DIV_OFF + v] as f32 } else { 0.0 };
                bos.planes[ELASTO_SURF_PLANE_PHI * ELASTO_V + v] =
                    if bos.valid_mask[v] != 0 { bos.solve_z[Z_PHI_OFF + v] as f32 } else { 0.0 };
                v += 1;
            }

            // Compute `m = 1` complex wavefield into `Z_URE_OFF / Z_UIM_OFF` and `k1` into `LP_K1_OFF`
            let mut max_amp1 = 0.0f64;
            v = 0;
            while v < ELASTO_V {
                let mut sr = 0.0f64;
                let mut si = 0.0f64;
                let mut sp = 0.0f64;
                let mut s = 0usize;
                while s < K_SEGS {
                    let re = seg_ring_read(bos, seg_ring_idx(s, 1, 0, v));
                    let im = seg_ring_read(bos, seg_ring_idx(s, 1, 1, v));
                    sr += re;
                    si += im;
                    sp += re * re + im * im;
                    s += 1;
                }
                let mr = sr * inv_k;
                let mi = si * inv_k;
                let m2 = mr * mr + mi * mi;
                let mp = sp * inv_k;
                let coh1 = if mp > 1.0e-14 { (m2 / mp).clamp(0.0, 1.0) } else { 0.0 };
                let amp1 = wasm_sqrt_f64(m2);
                if amp1 > max_amp1 {
                    max_amp1 = amp1;
                }
                bos.solve_z[Z_URE_OFF + v] = mr;
                bos.solve_z[Z_UIM_OFF + v] = mi;
                bos.solve_z[Z_COH_OFF + v] = coh1;
                v += 1;
            }
            let omega1 = TWO_PI_64 * self.active_freqs[1];
            let _ = compute_wavefield_poynting_and_k(bos, omega1, dx, dy, LP_K1_OFF, false);

            let ln_fratio = ln_f64(self.active_freqs[1] / self.active_freqs[0]);
            let mut n_alpha = 0usize;
            if ln_fratio > 0.05 {
                let amp1_floor = (0.06 * max_amp1).max(1.0e-4);
                v = 0;
                while v < ELASTO_V {
                    if bos.valid_mask[v] != 0 && bos.solve_z[Z_COH_OFF + v] >= coh_crit {
                        let a1 = wasm_sqrt_f64(
                            bos.solve_z[Z_URE_OFF + v] * bos.solve_z[Z_URE_OFF + v]
                                + bos.solve_z[Z_UIM_OFF + v] * bos.solve_z[Z_UIM_OFF + v],
                        );
                        let k0 = bos.solve_lp[LP_K0_OFF + v];
                        let k1 = bos.solve_lp[LP_K1_OFF + v];
                        if a1 >= amp1_floor && k0 > 5.0e-3 && k1 > 5.0e-3 && n_alpha < ELASTO_V {
                            let al = ln_f64(k1 / k0) / ln_fratio;
                            if al.is_finite() && al > 0.05 && al < 2.5 {
                                bos.solve_lp[LP_SORT_OFF + n_alpha] = al;
                                n_alpha += 1;
                            }
                        }
                    }
                    v += 1;
                }
            }
            if n_alpha >= 24 {
                disp_alpha = median_in_place(&mut bos.solve_lp[LP_SORT_OFF..LP_SORT_OFF + n_alpha], n_alpha);
                if disp_alpha >= 0.35 && disp_alpha <= 0.68 {
                    class_code = ELASTO_CLASS_KIRCHHOFF_PLATE;
                } else if disp_alpha >= 0.78 && disp_alpha <= 1.25 {
                    class_code = ELASTO_CLASS_MEMBRANE_SHEAR;
                } else {
                    class_code = ELASTO_CLASS_DISPERSIVE_UNKNOWN;
                }
            } else {
                class_code = ELASTO_CLASS_DISPERSIVE_UNKNOWN;
            }
        } else {
            // Single frequency: publish `m = 0` planes directly
            v = 0;
            while v < ELASTO_V {
                let mr = bos.solve_z[Z_URE_OFF + v];
                let mi = bos.solve_z[Z_UIM_OFF + v];
                let amp = wasm_sqrt_f64(mr * mr + mi * mi);
                bos.planes[ELASTO_SURF_PLANE_AMP * ELASTO_V + v] =
                    if bos.valid_mask[v] != 0 { amp as f32 } else { 0.0 };
                bos.planes[ELASTO_SURF_PLANE_PHASE * ELASTO_V + v] = if bos.valid_mask[v] != 0 {
                    fast_atan2f(mi as f32, mr as f32)
                } else {
                    0.0
                };
                bos.planes[ELASTO_SURF_PLANE_COH * ELASTO_V + v] = bos.solve_z[Z_COH_OFF + v] as f32;
                bos.planes[ELASTO_SURF_PLANE_DIV * ELASTO_V + v] =
                    if bos.valid_mask[v] != 0 { bos.solve_z[Z_DIV_OFF + v] as f32 } else { 0.0 };
                bos.planes[ELASTO_SURF_PLANE_PHI * ELASTO_V + v] =
                    if bos.valid_mask[v] != 0 { bos.solve_z[Z_PHI_OFF + v] as f32 } else { 0.0 };
                v += 1;
            }
        }

        // Median wavenumber `k_med` & fail-closed long-wavelength / spatial-aliasing guards!
        let mut nk = 0usize;
        v = 0;
        while v < ELASTO_V {
            if bos.valid_mask[v] != 0 {
                let kv = bos.solve_lp[LP_K0_OFF + v];
                if kv.is_finite() && kv > 1.0e-5 {
                    bos.solve_lp[LP_SORT_OFF + nk] = kv;
                    nk += 1;
                }
            }
            v += 1;
        }
        let k_med = if nk >= 16 {
            median_in_place(&mut bos.solve_lp[LP_SORT_OFF..LP_SORT_OFF + nk], nk)
        } else {
            0.0
        };
        let l_fov = (frame_w.min(frame_h)) as f64;
        let mut reason_mask = 0u32;
        if k_med * l_fov < 0.5 * PI_64 {
            reason_mask |= ELASTO_REASON_WAVELENGTH_TOO_LONG;
        }
        if alias_med_rad > 1.2 {
            reason_mask |= ELASTO_REASON_SPATIAL_ALIASING;
        }
        if standing_frac > 0.70 {
            reason_mask |= ELASTO_REASON_STANDING_WAVE_DOMINANT;
        }

        if (reason_mask & ELASTO_REASON_WAVELENGTH_TOO_LONG) != 0 {
            // Rigid-body oscillation: refuse wave speed & constitutive class!
            bos.planes[ELASTO_SURF_PLANE_C * ELASTO_V..(ELASTO_SURF_PLANE_C + 1) * ELASTO_V].fill(0.0);
            self.telem.fill(0);
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_REFUSED;
            self.telem[ELASTO_TELEM_MODE] = self.mode;
            self.telem[ELASTO_TELEM_REASON_MASK] = reason_mask;
            self.telem[ELASTO_TELEM_VALID_CELLS] = valid_cnt as u32;
            self.telem[ELASTO_TELEM_LOCKED_FREQS] = m_cnt as u32;
            self.telem[ELASTO_TELEM_WELCH_SEGS] = self.welch_count;
            self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(self.active_freqs[0]);
            self.telem[ELASTO_TELEM_MED_K] = pack_f32(k_med);
            self.telem[ELASTO_TELEM_MEAN_COH] = pack_f32(coh_sum / (valid_cnt as f64));
            self.telem[ELASTO_TELEM_SINC_H0] = pack_f32(h0_sinc);
            self.telem[ELASTO_TELEM_GRID_W] = ELASTO_W as u32;
            self.telem[ELASTO_TELEM_GRID_H] = ELASTO_H as u32;
            self.live_ready = 0;
            return false;
        }

        // Publish local wave speed `c(x,y; f0) = omega0 / k0(x,y)` into Plane 0 and compute p10/p50/p90
        let k_min_clip = (0.15 * k_med).max(1.0e-3);
        let k_max_clip = (5.0 * k_med).max(0.5);
        let mut nc = 0usize;
        v = 0;
        while v < ELASTO_V {
            let kv = bos.solve_lp[LP_K0_OFF + v];
            if bos.valid_mask[v] != 0 && kv.is_finite() && kv >= k_min_clip && kv <= k_max_clip {
                let cv = omega0 / kv;
                bos.planes[ELASTO_SURF_PLANE_C * ELASTO_V + v] = cv as f32;
                bos.solve_lp[LP_SORT_OFF + nc] = cv;
                nc += 1;
            } else {
                bos.planes[ELASTO_SURF_PLANE_C * ELASTO_V + v] = 0.0;
            }
            v += 1;
        }
        let (c_p10, c_med, c_p90) = if nc >= 8 {
            sort_f64_in_place(&mut bos.solve_lp[LP_SORT_OFF..LP_SORT_OFF + nc]);
            (
                bos.solve_lp[LP_SORT_OFF + nc / 10],
                bos.solve_lp[LP_SORT_OFF + nc / 2],
                bos.solve_lp[LP_SORT_OFF + (nc * 9) / 10],
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_READY;
        self.telem[ELASTO_TELEM_MODE] = self.mode;
        self.telem[ELASTO_TELEM_REASON_MASK] = reason_mask;
        self.telem[ELASTO_TELEM_CLASS] = class_code;
        self.telem[ELASTO_TELEM_VALID_CELLS] = valid_cnt as u32;
        self.telem[ELASTO_TELEM_LOCKED_FREQS] = m_cnt as u32;
        self.telem[ELASTO_TELEM_WELCH_SEGS] = self.welch_count;
        self.telem[ELASTO_TELEM_BETA1_HOLES] = beta1_holes;
        self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(self.active_freqs[0]);
        self.telem[ELASTO_TELEM_F1_HZ] = if m_cnt >= 2 { pack_f32(self.active_freqs[1]) } else { 0 };
        self.telem[ELASTO_TELEM_F2_HZ] = if m_cnt >= 3 { pack_f32(self.active_freqs[2]) } else { 0 };
        self.telem[ELASTO_TELEM_F3_HZ] = if m_cnt >= 4 { pack_f32(self.active_freqs[3]) } else { 0 };
        self.telem[ELASTO_TELEM_MED_C] = pack_f32(c_med);
        self.telem[ELASTO_TELEM_DISPERSION_ALPHA] = pack_f32(disp_alpha);
        self.telem[ELASTO_TELEM_ANISOTROPY] = pack_f32(aniso_ratio);
        self.telem[ELASTO_TELEM_MEAN_COH] = pack_f32(coh_sum / (valid_cnt as f64));
        self.telem[ELASTO_TELEM_SOURCE_X] = pack_f32(src_x);
        self.telem[ELASTO_TELEM_SOURCE_Y] = pack_f32(src_y);
        self.telem[ELASTO_TELEM_SINK_X] = pack_f32(sink_x);
        self.telem[ELASTO_TELEM_SINK_Y] = pack_f32(sink_y);
        self.telem[ELASTO_TELEM_COEXACT_FRAC] = pack_f32(coexact_frac);
        self.telem[ELASTO_TELEM_EXACT_FRAC] = pack_f32(exact_frac);
        self.telem[ELASTO_TELEM_MED_K] = pack_f32(k_med);
        self.telem[ELASTO_TELEM_SINC_H0] = pack_f32(h0_sinc);
        self.telem[ELASTO_TELEM_ABOS_CURL_RMS] = 0;
        self.telem[ELASTO_TELEM_ABOS_PHI_RMS] = 0;
        self.telem[ELASTO_TELEM_C_P10] = pack_f32(c_p10);
        self.telem[ELASTO_TELEM_C_P90] = pack_f32(c_p90);
        self.telem[ELASTO_TELEM_GRID_W] = ELASTO_W as u32;
        self.telem[ELASTO_TELEM_GRID_H] = ELASTO_H as u32;
        self.telem[ELASTO_TELEM_PROVENANCE] = 1;
        self.telem[ELASTO_TELEM_RESERVED31] = 0;

        self.live_ready = 1;
        self.fresh_solve = 1;
        true
    }

    /// Stage 2B: `256x128` Lock-In Acoustic BOS (Mode 4).
    /// Accumulates audio-phase-locked complex refraction `(\tilde{eps}_x, \tilde{eps}_y)` inside
    /// `BosState.solve_{x,b,r,p,lp}` and runs dual-pass paired-SIMD128 DCT Neumann Poisson inside
    /// `BosState.solve_z` without overwriting the running lock-in accumulators!
    #[inline(never)]
    fn step_acoustic_bos(
        &mut self,
        luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        meta: &VibroFrameMeta,
        vibro: &mut VibrometryState,
        bos: &mut BosState,
        bos_cache: &mut BosRefCache,
    ) -> bool {
        let (dt_row_us, exposure_us, q_start, q_step, scan_along_ey) =
            match self.validate_rs_geometry(frame_w, frame_h, meta, vibro) {
                Ok(g) => g,
                Err(rsn) => {
                    self.telem.fill(0);
                    self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_UNCALIBRATED;
                    self.telem[ELASTO_TELEM_MODE] = self.mode;
                    self.telem[ELASTO_TELEM_REASON_MASK] = rsn;
                    self.telem[ELASTO_TELEM_GRID_W] = BOS_GRID_W as u32;
                    self.telem[ELASTO_TELEM_GRID_H] = BOS_GRID_H as u32;
                    bos.planes.fill(0.0);
                    bos.valid_mask.fill(0);
                    return false;
                }
            };
        if !scan_along_ey {
            self.fail_refused(RS_REASON_AXIS_TRANSPOSED, bos, true);
            return false;
        }

        let need_px = frame_w * frame_h;
        // 1. 16-frame tripod reference warmup into `bos.ref_luma`
        if self.ref_warm_frames < REF_WARM_FRAMES {
            if self.ref_warm_frames == 0 {
                bos.ref_luma[..need_px].copy_from_slice(&luma[..need_px]);
                bos.clear_solver_scratch();
                bos.planes.fill(0.0);
                bos.valid_mask.fill(0);
                bos.ref_noise_var[0..BOS_GRID_H * 3].fill(0.0);
                self.abos_frames_acc = 0;
                self.abos_solve_countdown = 0;
            } else {
                let alpha = 1.0f32 / ((self.ref_warm_frames as f32) + 1.0);
                let mut p = 0usize;
                while p < need_px {
                    bos.ref_luma[p] += (luma[p] - bos.ref_luma[p]) * alpha;
                    p += 1;
                }
            }
            bos.bump_ref_gen();
            self.ref_warm_frames += 1;
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
            self.telem[ELASTO_TELEM_MODE] = self.mode;
            self.telem[ELASTO_TELEM_GRID_W] = BOS_GRID_W as u32;
            self.telem[ELASTO_TELEM_GRID_H] = BOS_GRID_H as u32;
            return false;
        }

        // 2. Lock primary acoustic frequency `f0`
        let f_opt_max = 0.48 * (1_000_000.0 / dt_row_us);
        let f0 = if self.has_explicit_freqs != 0 && self.explicit_freq_count > 0 {
            let f = self.explicit_freqs[0] as f64;
            if f > 0.1 && f <= f_opt_max { f } else { 0.0 }
        } else {
            let mut pks = [0.0f64; ELASTO_MAX_FREQS];
            let np = vibro.detect_top_audio_peaks(f_opt_max, &mut pks);
            if np > 0 { pks[0] } else { 0.0 }
        };
        if f0 <= 0.1 {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
            self.telem[ELASTO_TELEM_REASON_MASK] = RS_REASON_AUDIO_UNDERRUN;
            return false;
        }
        let h0_sinc = sinc_exposure_h(f0, exposure_us);
        if abs_f64(h0_sinc) < 0.05 {
            self.fail_refused(RS_REASON_EXPOSURE_NULL, bos, true);
            self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(f0);
            self.telem[ELASTO_TELEM_SINC_H0] = pack_f32(h0_sinc);
            return false;
        }
        if abs_f64(f0 - self.abos_freq_hz) > 3.0 {
            self.abos_freq_hz = f0;
            self.abos_frames_acc = 0;
            self.abos_solve_countdown = 0;
            bos.clear_solver_scratch();
            bos.ref_noise_var[0..BOS_GRID_H * 3].fill(0.0);
        }

        let Some((x_re, x_im, _pxx)) = vibro.audio_phasor_at(f0, meta.t_src_us) else {
            self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
            self.telem[ELASTO_TELEM_REASON_MASK] = RS_REASON_AUDIO_UNDERRUN;
            return false;
        };

        // 3. Single-frame `256x128` affine IC-LK into `bos.w_x`, `bos.w_y`, `bos.clear_streak`
        let valid_lk = bos.extract_dense_lk_raw(bos_cache, luma, frame_w, frame_h);
        if valid_lk < 1024 {
            self.fail_refused(RS_REASON_LOW_GRAD, bos, true);
            return false;
        }

        // 4. Accumulate row-time rolling-shutter audio-locked complex refraction into
        //    `solve_x (ex*C), solve_b (ex*S), solve_r (ey*C), solve_p (ey*S), solve_lp (cnt)`
        //    and per-row 2x2 Gram matrix `(C^2, C*S, S^2)` into `bos.ref_noise_var[0..3*128]`
        let dt_row_sec = dt_row_us * 1.0e-6;
        let mut j = 0usize;
        while j < BOS_GRID_H {
            let (_, cy) = node_coord(0, j, frame_w, frame_h);
            let q_row = q_start + (cy as f64) * q_step;
            let (si, co) = sin_cos_f64(TWO_PI_64 * f0 * q_row * dt_row_sec);
            let cj = x_re * co - x_im * si;
            let sj = x_re * si + x_im * co;

            bos.ref_noise_var[j * 3] += (cj * cj) as f32;
            bos.ref_noise_var[j * 3 + 1] += (cj * sj) as f32;
            bos.ref_noise_var[j * 3 + 2] += (sj * sj) as f32;

            let row_off = j * BOS_GRID_W;
            let mut i = 0usize;
            while i < BOS_GRID_W {
                let u = row_off + i;
                if bos.clear_streak[u] != 0 {
                    let ex = bos.w_x[u] as f64;
                    let ey = bos.w_y[u] as f64;
                    bos.solve_x[u] += ex * cj;
                    bos.solve_b[u] += ex * sj;
                    bos.solve_r[u] += ey * cj;
                    bos.solve_p[u] += ey * sj;
                    bos.solve_lp[u] += 1.0;
                }
                i += 1;
            }
            j += 1;
        }
        self.abos_frames_acc += 1;
        self.abos_solve_countdown += 1;

        // Require >= 24 locked frames (~8 Welch segments) before first publish, then refresh every 8 frames
        if self.abos_frames_acc < 24 || (self.live_ready != 0 && self.abos_solve_countdown < 8) {
            if self.live_ready == 0 {
                self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_WARMING;
                self.telem[ELASTO_TELEM_WELCH_SEGS] = self.abos_frames_acc / 3;
                self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(f0);
            }
            return false;
        }
        self.abos_solve_countdown = 0;

        // 5. Invert Hann-weighted +-3 row-pooled 2x2 lock-in normal matrix (zero stroboscopic striping
        //    even when `f0 mod fps == 0` like 1,200 Hz at 30 fps!), divide by `H(f0)`, and write
        //    `(eps_x_re, eps_y_re, eps_x_im, eps_y_im)` temporarily into `bos.planes[0..4*BOS_NODES]`!
        const POOL_W: [f64; 7] = [1.0, 3.0, 6.0, 8.0, 6.0, 3.0, 1.0];
        let inv_h0 = 1.0 / h0_sinc;
        let inv_facc = 1.0 / (self.abos_frames_acc as f64);
        let mut valid_out = 0u32;

        j = 0;
        while j < BOS_GRID_H {
            // Pool per-row 2x2 Gram matrix over `[j-3..=j+3]`
            let mut g00 = 0.0f64;
            let mut g01 = 0.0f64;
            let mut g11 = 0.0f64;
            let mut w_tot = 0.0f64;
            let mut k = 0usize;
            while k < 7 {
                let jj = (j as isize) + (k as isize) - 3;
                if jj >= 0 && jj < (BOS_GRID_H as isize) {
                    let ju = jj as usize;
                    let w = POOL_W[k];
                    g00 += w * (bos.ref_noise_var[ju * 3] as f64);
                    g01 += w * (bos.ref_noise_var[ju * 3 + 1] as f64);
                    g11 += w * (bos.ref_noise_var[ju * 3 + 2] as f64);
                    w_tot += w;
                }
                k += 1;
            }
            let reg = 1.0e-4 * w_tot * (self.abos_frames_acc as f64);
            g00 += reg;
            g11 += reg;
            let det = g00 * g11 - g01 * g01;
            let det_ok = det.is_finite() && det > 1.0e-9 * (g00 * g11).max(1.0);
            let inv_det = if det_ok { 1.0 / det } else { 0.0 };

            let mut i = 0usize;
            while i < BOS_GRID_W {
                let u = j * BOS_GRID_W + i;
                if det_ok && bos.solve_lp[u] * inv_facc >= 0.45 {
                    let mut sx_c = 0.0f64;
                    let mut sx_s = 0.0f64;
                    let mut sy_c = 0.0f64;
                    let mut sy_s = 0.0f64;
                    let mut w_ok = 0.0f64;
                    k = 0;
                    while k < 7 {
                        let jj = (j as isize) + (k as isize) - 3;
                        if jj >= 0 && jj < (BOS_GRID_H as isize) {
                            let ju = jj as usize;
                            let un = ju * BOS_GRID_W + i;
                            let frac = bos.solve_lp[un] * inv_facc;
                            if frac >= 0.35 {
                                let w = POOL_W[k] / frac;
                                sx_c += w * bos.solve_x[un];
                                sx_s += w * bos.solve_b[un];
                                sy_c += w * bos.solve_r[un];
                                sy_s += w * bos.solve_p[un];
                                w_ok += POOL_W[k];
                            }
                        }
                        k += 1;
                    }
                    if w_ok >= 0.5 * w_tot {
                        let scale = (w_tot / w_ok) * inv_h0 * inv_det;
                        let ex_re = (g11 * sx_c - g01 * sx_s) * scale;
                        let ex_im = (-g01 * sx_c + g00 * sx_s) * scale;
                        let ey_re = (g11 * sy_c - g01 * sy_s) * scale;
                        let ey_im = (-g01 * sy_c + g00 * sy_s) * scale;
                        bos.planes[u] = ex_re as f32;
                        bos.planes[BOS_NODES + u] = ey_re as f32;
                        bos.planes[2 * BOS_NODES + u] = ex_im as f32;
                        bos.planes[3 * BOS_NODES + u] = ey_im as f32;
                        bos.planes[ELASTO_ABOS_PLANE_EPS_MAG * BOS_NODES + u] =
                            wasm_sqrt_f64(ex_re * ex_re + ex_im * ex_im + ey_re * ey_re + ey_im * ey_im)
                                as f32;
                        bos.valid_mask[u] = 7;
                        valid_out += 1;
                    } else {
                        bos.planes[u] = 0.0;
                        bos.planes[BOS_NODES + u] = 0.0;
                        bos.planes[2 * BOS_NODES + u] = 0.0;
                        bos.planes[3 * BOS_NODES + u] = 0.0;
                        bos.planes[ELASTO_ABOS_PLANE_EPS_MAG * BOS_NODES + u] = 0.0;
                        bos.valid_mask[u] = 0;
                    }
                } else {
                    bos.planes[u] = 0.0;
                    bos.planes[BOS_NODES + u] = 0.0;
                    bos.planes[2 * BOS_NODES + u] = 0.0;
                    bos.planes[3 * BOS_NODES + u] = 0.0;
                    bos.planes[ELASTO_ABOS_PLANE_EPS_MAG * BOS_NODES + u] = 0.0;
                    bos.valid_mask[u] = 0;
                }
                i += 1;
            }
            j += 1;
        }

        let hx = (frame_w as f64) / (BOS_GRID_W as f64);
        let hy = (frame_h as f64) / (BOS_GRID_H as f64);
        let inv_hx = 1.0 / hx;
        let inv_hy = 1.0 / hy;

        // 6. Complex curl integrability diagnostic `|nabla x \tilde{eps}|` into Plane 5
        let mut curl2_sum = 0.0f64;
        let mut curl_cnt = 0usize;
        bos.planes[ELASTO_ABOS_PLANE_CURL_MAG * BOS_NODES..(ELASTO_ABOS_PLANE_CURL_MAG + 1) * BOS_NODES]
            .fill(0.0);
        j = 0;
        while j + 1 < BOS_GRID_H {
            let mut i = 0usize;
            while i + 1 < BOS_GRID_W {
                let u00 = j * BOS_GRID_W + i;
                let u10 = u00 + 1;
                let u01 = u00 + BOS_GRID_W;
                let u11 = u01 + 1;
                if bos.valid_mask[u00] != 0
                    && bos.valid_mask[u10] != 0
                    && bos.valid_mask[u01] != 0
                    && bos.valid_mask[u11] != 0
                {
                    let dy_re = 0.5
                        * (((bos.planes[BOS_NODES + u10] + bos.planes[BOS_NODES + u11]) as f64)
                            - ((bos.planes[BOS_NODES + u00] + bos.planes[BOS_NODES + u01]) as f64))
                        * inv_hx;
                    let dx_re = 0.5
                        * (((bos.planes[u01] + bos.planes[u11]) as f64)
                            - ((bos.planes[u00] + bos.planes[u10]) as f64))
                        * inv_hy;
                    let cr = dy_re - dx_re;

                    let dy_im = 0.5
                        * (((bos.planes[3 * BOS_NODES + u10] + bos.planes[3 * BOS_NODES + u11]) as f64)
                            - ((bos.planes[3 * BOS_NODES + u00] + bos.planes[3 * BOS_NODES + u01]) as f64))
                        * inv_hx;
                    let dx_im = 0.5
                        * (((bos.planes[2 * BOS_NODES + u01] + bos.planes[2 * BOS_NODES + u11]) as f64)
                            - ((bos.planes[2 * BOS_NODES + u00] + bos.planes[2 * BOS_NODES + u10]) as f64))
                        * inv_hy;
                    let ci = dy_im - dx_im;
                    let c2 = cr * cr + ci * ci;
                    curl2_sum += c2;
                    curl_cnt += 1;
                    bos.planes[ELASTO_ABOS_PLANE_CURL_MAG * BOS_NODES + u00] = wasm_sqrt_f64(c2) as f32;
                }
                i += 1;
            }
            j += 1;
        }
        let curl_rms = if curl_cnt > 0 {
            wasm_sqrt_f64(curl2_sum / (curl_cnt as f64))
        } else {
            0.0
        };

        // 7. Dual-pass exact `256x128` DCT Neumann Poisson solve in `bos.solve_z`
        //    (NEVER touches `solve_{x,b,r,p,lp}` lock-in accumulators!)
        // Pass A: Real part `Delta Phi_re = div(eps_re)` -> store into `bos.w_x` temporarily
        assemble_bos_neumann_rhs_into_z(bos, 0, BOS_NODES, inv_hx, inv_hy);
        bos.apply_dct_poisson_solve_z(1.0, 1.0, hx, hy);
        let mut mean_re = 0.0f64;
        let mut u = 0usize;
        while u < BOS_NODES {
            mean_re += bos.solve_z[u];
            u += 1;
        }
        mean_re /= BOS_NODES as f64;
        u = 0;
        while u < BOS_NODES {
            bos.w_x[u] = (bos.solve_z[u] - mean_re) as f32;
            u += 1;
        }

        // Pass B: Imag part `Delta Phi_im = div(eps_im)` -> store into `bos.w_y` temporarily
        assemble_bos_neumann_rhs_into_z(bos, 2 * BOS_NODES, 3 * BOS_NODES, inv_hx, inv_hy);
        bos.apply_dct_poisson_solve_z(1.0, 1.0, hx, hy);
        let mut mean_im = 0.0f64;
        u = 0;
        while u < BOS_NODES {
            mean_im += bos.solve_z[u];
            u += 1;
        }
        mean_im /= BOS_NODES as f64;
        u = 0;
        while u < BOS_NODES {
            bos.w_y[u] = (bos.solve_z[u] - mean_im) as f32;
            u += 1;
        }

        // Publish final complex acoustic refractive potential `(|Phi|, phase, Phi_re, Phi_im)`
        let mut phi2_sum = 0.0f64;
        u = 0;
        while u < BOS_NODES {
            let pr = bos.w_x[u] as f64;
            let pi = bos.w_y[u] as f64;
            let pm2 = pr * pr + pi * pi;
            let pm = wasm_sqrt_f64(pm2);
            bos.planes[ELASTO_ABOS_PLANE_PHI_MAG * BOS_NODES + u] = pm as f32;
            bos.planes[ELASTO_ABOS_PLANE_PHI_PHASE * BOS_NODES + u] = fast_atan2f(pi as f32, pr as f32);
            bos.planes[ELASTO_ABOS_PLANE_PHI_RE * BOS_NODES + u] = pr as f32;
            bos.planes[ELASTO_ABOS_PLANE_PHI_IM * BOS_NODES + u] = pi as f32;
            phi2_sum += pm2;
            u += 1;
        }
        let phi_rms = wasm_sqrt_f64(phi2_sum / (BOS_NODES as f64));

        self.telem[ELASTO_TELEM_STATE] = ELASTO_STATE_READY;
        self.telem[ELASTO_TELEM_MODE] = self.mode;
        self.telem[ELASTO_TELEM_REASON_MASK] = 0;
        self.telem[ELASTO_TELEM_CLASS] = ELASTO_CLASS_NONE;
        self.telem[ELASTO_TELEM_VALID_CELLS] = valid_out;
        self.telem[ELASTO_TELEM_LOCKED_FREQS] = 1;
        self.telem[ELASTO_TELEM_WELCH_SEGS] = self.abos_frames_acc / 3;
        self.telem[ELASTO_TELEM_BETA1_HOLES] = 0;
        self.telem[ELASTO_TELEM_F0_HZ] = pack_f32(f0);
        self.telem[ELASTO_TELEM_F1_HZ] = 0;
        self.telem[ELASTO_TELEM_F2_HZ] = 0;
        self.telem[ELASTO_TELEM_F3_HZ] = 0;
        self.telem[ELASTO_TELEM_SINC_H0] = pack_f32(h0_sinc);
        self.telem[ELASTO_TELEM_ABOS_CURL_RMS] = pack_f32(curl_rms);
        self.telem[ELASTO_TELEM_ABOS_PHI_RMS] = pack_f32(phi_rms);
        self.telem[ELASTO_TELEM_GRID_W] = BOS_GRID_W as u32;
        self.telem[ELASTO_TELEM_GRID_H] = BOS_GRID_H as u32;
        self.telem[ELASTO_TELEM_PROVENANCE] = 1;
        self.abos_valid_nodes = valid_out;
        self.live_ready = 1;
        self.fresh_solve = 1;
        true
    }
}

/// Evaluates affine-brightness-demeaned 1-step IC-LK `(ux, uy)` on a thin `sw x sh` strip at `(x0, y0)`
/// (`I_cur(x, y) - I_ref(x, y) ≈ -Ix * ux - Iy * uy`, matching `generate_v33_demos.mjs` `src(x - u)`).
#[inline(always)]
fn eval_strip_iclk(
    cur: &[f32],
    ref_img: &[f32],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    sw: usize,
    sh: usize,
) -> Option<(f64, f64)> {
    if x0 < 1 || y0 < 1 || x0 + sw + 1 >= w || y0 + sh + 1 >= h {
        return None;
    }
    let mut n = 0usize;
    let mut sum_r = 0.0f64;
    let mut sum_c = 0.0f64;
    let mut dy = 0usize;
    while dy < sh {
        let row = (y0 + dy) * w + x0;
        let mut dx = 0usize;
        while dx < sw {
            sum_r += ref_img[row + dx] as f64;
            sum_c += cur[row + dx] as f64;
            n += 1;
            dx += 1;
        }
        dy += 1;
    }
    let inv_n = 1.0 / (n as f64);
    let mu_r = sum_r * inv_n;
    let mu_c = sum_c * inv_n;

    let mut hxx = 0.0f64;
    let mut hxy = 0.0f64;
    let mut hyy = 0.0f64;
    let mut bx = 0.0f64;
    let mut by = 0.0f64;
    dy = 0;
    while dy < sh {
        let row = (y0 + dy) * w + x0;
        let mut dx = 0usize;
        while dx < sw {
            let p = row + dx;
            let gx = 0.5 * ((ref_img[p + 1] - ref_img[p - 1]) as f64);
            let gy = 0.5 * ((ref_img[p + w] - ref_img[p - w]) as f64);
            let diff = ((cur[p] as f64) - mu_c) - ((ref_img[p] as f64) - mu_r);
            hxx += gx * gx;
            hxy += gx * gy;
            hyy += gy * gy;
            // I_cur(x) = I_ref(x - u) ≈ I_ref(x) - u . grad I_ref -> -diff ≈ u . grad I_ref
            bx -= gx * diff;
            by -= gy * diff;
            dx += 1;
        }
        dy += 1;
    }
    let det = hxx * hyy - hxy * hxy;
    let tr = hxx + hyy;
    if !det.is_finite() || tr < 1.0e-4 || det <= 1.0e-7 * tr * tr {
        return None;
    }
    let inv_det = 1.0 / det;
    let ux = (hyy * bx - hxy * by) * inv_det;
    let uy = (-hxy * bx + hxx * by) * inv_det;
    if !ux.is_finite() || !uy.is_finite() || ux * ux + uy * uy > 9.0 {
        return None;
    }
    Some((ux, uy))
}

#[inline(never)]
fn cholesky_factor(a: &[f64; 64], l: &mut [f64; 64], n: usize) -> bool {
    l.fill(0.0);
    let mut i = 0usize;
    while i < n {
        let mut j = 0usize;
        while j <= i {
            let mut sum = a[i * n + j];
            let mut k = 0usize;
            while k < j {
                sum -= l[i * n + k] * l[j * n + k];
                k += 1;
            }
            if i == j {
                if !sum.is_finite() || sum <= 1.0e-12 {
                    return false;
                }
                l[i * n + j] = wasm_sqrt_f64(sum);
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
            j += 1;
        }
        i += 1;
    }
    true
}

#[inline(never)]
fn cholesky_solve(l: &[f64; 64], b: &[f64; 8], x: &mut [f64; 8], n: usize) {
    let mut y = [0.0f64; 8];
    let mut i = 0usize;
    while i < n {
        let mut sum = b[i];
        let mut k = 0usize;
        while k < i {
            sum -= l[i * n + k] * y[k];
            k += 1;
        }
        y[i] = sum / l[i * n + i];
        i += 1;
    }
    let mut i_rev = n;
    while i_rev > 0 {
        i_rev -= 1;
        let mut sum = y[i_rev];
        let mut k = i_rev + 1;
        while k < n {
            sum -= l[k * n + i_rev] * x[k];
            k += 1;
        }
        x[i_rev] = sum / l[i_rev * n + i_rev];
    }
}

/// Computes wrap-free Poynting power flow `(Ix, Iy)` (when `write_power_flow`), exact local
/// edge-phase wavevector `(kx, ky)`, `9x9` weak-form Helmholtz wavenumber `k_helm`, combined `k_eff`
/// into `bos.solve_lp[k_out_off..]`, median edge phase step (for aliasing check), standing-wave
/// fraction, and anisotropic directional speed ratio `c_x / c_y`.
#[inline(never)]
fn compute_wavefield_poynting_and_k(
    bos: &mut BosState,
    omega: f64,
    dx: f64,
    dy: f64,
    k_out_off: usize,
    write_power_flow: bool,
) -> (f64, f64, f64) {
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;
    let inv_dx2 = inv_dx * inv_dx;
    let inv_dy2 = inv_dy * inv_dy;

    bos.solve_lp[LP_KX_OFF..LP_KX_OFF + ELASTO_V].fill(0.0);
    bos.solve_lp[LP_KY_OFF..LP_KY_OFF + ELASTO_V].fill(0.0);
    bos.solve_lp[k_out_off..k_out_off + ELASTO_V].fill(0.0);
    if write_power_flow {
        bos.solve_z[Z_IX_OFF..Z_IX_OFF + ELASTO_V].fill(0.0);
        bos.solve_z[Z_IY_OFF..Z_IY_OFF + ELASTO_V].fill(0.0);
    }

    let mut n_phase_steps = 0usize;
    let mut r = 0usize;
    while r < ELASTO_H {
        let mut c = 0usize;
        while c < ELASTO_W {
            let v = r * ELASTO_W + c;
            if bos.valid_mask[v] == 0 {
                c += 1;
                continue;
            }
            let ur = bos.solve_z[Z_URE_OFF + v];
            let ui = bos.solve_z[Z_UIM_OFF + v];

            // Wrap-free edge phase increments via local perpendicular-smoothed complex cross-products
            // (`sum w * (u_a* u_b)` before `atan2`), eliminating Rician noise bias on `hypot(kx, ky)`
            // without any phase-unwrapping artifacts.
            let mut cr_x = 0.0f64;
            let mut ci_x = 0.0f64;
            let mut ix_sum = 0.0f64;
            let mut nx = 0usize;
            let mut dr = -1isize;
            while dr <= 1 {
                let rr_i = (r as isize) + dr;
                if rr_i >= 0 && rr_i < (ELASTO_H as isize) {
                    let wt = if dr == 0 { 2.0 } else { 1.0 };
                    let vc = (rr_i as usize) * ELASTO_W + c;
                    if bos.valid_mask[vc] != 0 {
                        let ucr = bos.solve_z[Z_URE_OFF + vc];
                        let uci = bos.solve_z[Z_UIM_OFF + vc];
                        if c > 0 && bos.valid_mask[vc - 1] != 0 {
                            let lr = bos.solve_z[Z_URE_OFF + vc - 1];
                            let li = bos.solve_z[Z_UIM_OFF + vc - 1];
                            let cr = lr * ucr + li * uci;
                            let ci = lr * uci - li * ucr;
                            cr_x += wt * cr;
                            ci_x += wt * ci;
                            if dr == 0 {
                                ix_sum += omega * ci;
                                nx += 1;
                            }
                        }
                        if c + 1 < ELASTO_W && bos.valid_mask[vc + 1] != 0 {
                            let rr = bos.solve_z[Z_URE_OFF + vc + 1];
                            let ri = bos.solve_z[Z_UIM_OFF + vc + 1];
                            let cr = ucr * rr + uci * ri;
                            let ci = ucr * ri - uci * rr;
                            cr_x += wt * cr;
                            ci_x += wt * ci;
                            if dr == 0 {
                                let dph = fast_atan2f(ci as f32, cr as f32) as f64;
                                ix_sum += omega * ci;
                                nx += 1;
                                if n_phase_steps < ELASTO_V {
                                    bos.solve_lp[LP_SORT_OFF + n_phase_steps] = abs_f64(dph);
                                    n_phase_steps += 1;
                                }
                            }
                        }
                    }
                }
                dr += 1;
            }
            let kx = if cr_x * cr_x + ci_x * ci_x > 1.0e-18 {
                (fast_atan2f(ci_x as f32, cr_x as f32) as f64) * inv_dx
            } else {
                0.0
            };

            let mut cr_y = 0.0f64;
            let mut ci_y = 0.0f64;
            let mut iy_sum = 0.0f64;
            let mut ny = 0usize;
            let mut dc = -1isize;
            while dc <= 1 {
                let cc_i = (c as isize) + dc;
                if cc_i >= 0 && cc_i < (ELASTO_W as isize) {
                    let wt = if dc == 0 { 2.0 } else { 1.0 };
                    let vc = r * ELASTO_W + (cc_i as usize);
                    if bos.valid_mask[vc] != 0 {
                        let ucr = bos.solve_z[Z_URE_OFF + vc];
                        let uci = bos.solve_z[Z_UIM_OFF + vc];
                        if r > 0 && bos.valid_mask[vc - ELASTO_W] != 0 {
                            let tr = bos.solve_z[Z_URE_OFF + vc - ELASTO_W];
                            let ti = bos.solve_z[Z_UIM_OFF + vc - ELASTO_W];
                            let cr = tr * ucr + ti * uci;
                            let ci = tr * uci - ti * ucr;
                            cr_y += wt * cr;
                            ci_y += wt * ci;
                            if dc == 0 {
                                iy_sum += omega * ci;
                                ny += 1;
                            }
                        }
                        if r + 1 < ELASTO_H && bos.valid_mask[vc + ELASTO_W] != 0 {
                            let br = bos.solve_z[Z_URE_OFF + vc + ELASTO_W];
                            let bi = bos.solve_z[Z_UIM_OFF + vc + ELASTO_W];
                            let cr = ucr * br + uci * bi;
                            let ci = ucr * bi - uci * br;
                            cr_y += wt * cr;
                            ci_y += wt * ci;
                            if dc == 0 {
                                iy_sum += omega * ci;
                                ny += 1;
                            }
                        }
                    }
                }
                dc += 1;
            }
            let ky = if cr_y * cr_y + ci_y * ci_y > 1.0e-18 {
                (fast_atan2f(ci_y as f32, cr_y as f32) as f64) * inv_dy
            } else {
                0.0
            };
            let _ = (ur, ui);

            bos.solve_lp[LP_KX_OFF + v] = kx;
            bos.solve_lp[LP_KY_OFF + v] = ky;
            if write_power_flow {
                bos.solve_z[Z_IX_OFF + v] = if nx > 0 { ix_sum / (nx as f64) } else { 0.0 };
                bos.solve_z[Z_IY_OFF + v] = if ny > 0 { iy_sum / (ny as f64) } else { 0.0 };
            }
            c += 1;
        }
        r += 1;
    }

    let alias_med = if n_phase_steps >= 16 {
        median_in_place(&mut bos.solve_lp[LP_SORT_OFF..LP_SORT_OFF + n_phase_steps], n_phase_steps)
    } else {
        0.0
    };

    // Combine sharp Poynting wavenumber `k_poynt = hypot(kx, ky)` with `9x9` weak-form Helmholtz
    // wavenumber `k_helm` (automatic standing-wave support!)
    let mut standing_cnt = 0usize;
    let mut total_k_cnt = 0usize;
    r = 0;
    while r < ELASTO_H {
        let mut c = 0usize;
        while c < ELASTO_W {
            let v = r * ELASTO_W + c;
            if bos.valid_mask[v] == 0 {
                c += 1;
                continue;
            }
            let kx = bos.solve_lp[LP_KX_OFF + v];
            let ky = bos.solve_lp[LP_KY_OFF + v];
            let k_poynt = wasm_sqrt_f64(kx * kx + ky * ky);

            // Evaluate compact-support `9x9` weak-form Helmholtz `k_helm^2 = -Re(<Delta psi, u> <psi, u>*) / |<psi, u>|^2`
            let mut k_helm = 0.0f64;
            if r >= 4 && r + 4 < ELASTO_H && c >= 4 && c + 4 < ELASTO_W {
                let mut full_ok = true;
                let mut ur_p = 0.0f64;
                let mut ui_p = 0.0f64;
                let mut lr_p = 0.0f64;
                let mut li_p = 0.0f64;
                let mut iy = 0usize;
                'hw: while iy < 9 {
                    let py = HELM_PHI[iy];
                    let d2y = HELM_D2PHI[iy] * inv_dy2;
                    let rr = r - 4 + iy;
                    let mut ix = 0usize;
                    while ix < 9 {
                        let vn = rr * ELASTO_W + (c - 4 + ix);
                        if bos.valid_mask[vn] == 0 {
                            full_ok = false;
                            break 'hw;
                        }
                        let px = HELM_PHI[ix];
                        let d2x = HELM_D2PHI[ix] * inv_dx2;
                        let pxy = px * py;
                        let lap_w = d2x * py + px * d2y;
                        let ur = bos.solve_z[Z_URE_OFF + vn];
                        let ui = bos.solve_z[Z_UIM_OFF + vn];
                        ur_p += pxy * ur;
                        ui_p += pxy * ui;
                        lr_p += lap_w * ur;
                        li_p += lap_w * ui;
                        ix += 1;
                    }
                    iy += 1;
                }
                let p2 = ur_p * ur_p + ui_p * ui_p;
                if full_ok && p2 > 1.0e-12 {
                    let kh2 = -(lr_p * ur_p + li_p * ui_p) / p2;
                    if kh2.is_finite() && kh2 > 0.0 {
                        k_helm = wasm_sqrt_f64(kh2);
                    }
                }
            }

            let k_eff = if k_poynt > 0.004 && (k_helm <= 1.0e-6 || k_poynt >= 0.30 * k_helm) {
                k_poynt
            } else if k_helm > 0.004 {
                standing_cnt += 1;
                k_helm
            } else {
                k_poynt
            };
            bos.solve_lp[k_out_off + v] = k_eff;
            if k_eff > 0.004 {
                total_k_cnt += 1;
            }
            c += 1;
        }
        r += 1;
    }

    let standing_frac = if total_k_cnt > 0 {
        (standing_cnt as f64) / (total_k_cnt as f64)
    } else {
        0.0
    };

    // Directional anisotropy ratio `c_x / c_y = med(|k_y| on vertical rays) / med(|k_x| on horizontal rays)`
    let mut aniso = 1.0f64;
    if write_power_flow {
        let mut nx_ray = 0usize;
        let mut v = 0usize;
        while v < ELASTO_V {
            if bos.valid_mask[v] != 0 {
                let akx = abs_f64(bos.solve_lp[LP_KX_OFF + v]);
                let aky = abs_f64(bos.solve_lp[LP_KY_OFF + v]);
                if akx > 2.0 * aky && akx > 0.005 && nx_ray < ELASTO_V {
                    bos.solve_lp[LP_SORT_OFF + nx_ray] = akx;
                    nx_ray += 1;
                }
            }
            v += 1;
        }
        let med_kx = if nx_ray >= 12 {
            median_in_place(&mut bos.solve_lp[LP_SORT_OFF..LP_SORT_OFF + nx_ray], nx_ray)
        } else {
            0.0
        };

        let mut ny_ray = 0usize;
        v = 0;
        while v < ELASTO_V {
            if bos.valid_mask[v] != 0 {
                let akx = abs_f64(bos.solve_lp[LP_KX_OFF + v]);
                let aky = abs_f64(bos.solve_lp[LP_KY_OFF + v]);
                if aky > 2.0 * akx && aky > 0.005 && ny_ray < ELASTO_V {
                    bos.solve_lp[LP_SORT_OFF + ny_ray] = aky;
                    ny_ray += 1;
                }
            }
            v += 1;
        }
        let med_ky = if ny_ray >= 12 {
            median_in_place(&mut bos.solve_lp[LP_SORT_OFF..LP_SORT_OFF + ny_ray], ny_ray)
        } else {
            0.0
        };
        if med_kx > 1.0e-4 && med_ky > 1.0e-4 {
            // Since c = omega / k, c_x / c_y = k_y / k_x!
            aniso = (med_ky / med_kx).clamp(0.1, 10.0);
        }
    }

    (alias_med, standing_frac, aniso)
}

/// Localizes primary interior Hodge power-flow source (`argmax` of smoothed `+div(I)`)
/// and primary interior crack/damping sink (`argmin` of energy-normalized local loss factor
/// `div(I) / (E_local + E_floor)` inside the strict interior `r in 6..H-6, c in 6..W-6` so
/// open frame outflow borders never mask an internal crack/damping absorber).
#[inline(never)]
fn localize_power_source_sink(bos: &BosState, dx: f64, dy: f64) -> (f64, f64, f64, f64) {
    let mut best_pos = -1.0e30f64;
    let mut best_loss = 1.0e30f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut kx = 0.0f64;
    let mut ky = 0.0f64;

    let mut max_e = 0.0f64;
    let mut v0 = 0usize;
    while v0 < ELASTO_V {
        if bos.valid_mask[v0] != 0 {
            let ur = bos.solve_z[Z_URE_OFF + v0];
            let ui = bos.solve_z[Z_UIM_OFF + v0];
            let e = ur * ur + ui * ui;
            if e > max_e {
                max_e = e;
            }
        }
        v0 += 1;
    }
    let e_floor = (0.04 * max_e).max(1.0e-5);

    // 5x5 Gaussian smoothing over interior valid cells to localize distributed sources/sinks cleanly
    const G5: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    let mut r = 4usize;
    while r + 4 < ELASTO_H {
        let mut c = 4usize;
        while c + 4 < ELASTO_W {
            let v = r * ELASTO_W + c;
            if bos.valid_mask[v] != 0
                && bos.valid_mask[v - 1] != 0
                && bos.valid_mask[v + 1] != 0
                && bos.valid_mask[v - ELASTO_W] != 0
                && bos.valid_mask[v + ELASTO_W] != 0
            {
                let mut sum_d = 0.0f64;
                let mut sum_e = 0.0f64;
                let mut sum_w = 0.0f64;
                let mut dy_i = 0usize;
                while dy_i < 5 {
                    let rr = r - 2 + dy_i;
                    let wy = G5[dy_i];
                    let mut dx_i = 0usize;
                    while dx_i < 5 {
                        let cc = c - 2 + dx_i;
                        let vn = rr * ELASTO_W + cc;
                        // Only pool strictly interior vertices (away from frame boundary & void rims)
                        if cc >= 2
                            && cc + 2 < ELASTO_W
                            && rr >= 2
                            && rr + 2 < ELASTO_H
                            && bos.valid_mask[vn] != 0
                            && bos.valid_mask[vn - 1] != 0
                            && bos.valid_mask[vn + 1] != 0
                            && bos.valid_mask[vn - ELASTO_W] != 0
                            && bos.valid_mask[vn + ELASTO_W] != 0
                        {
                            let w = wy * G5[dx_i];
                            let ur = bos.solve_z[Z_URE_OFF + vn];
                            let ui = bos.solve_z[Z_UIM_OFF + vn];
                            sum_d += w * bos.solve_z[Z_DIV_OFF + vn];
                            sum_e += w * (ur * ur + ui * ui);
                            sum_w += w;
                        }
                        dx_i += 1;
                    }
                    dy_i += 1;
                }
                if sum_w >= 64.0 {
                    let div_val = sum_d / sum_w;
                    let e_val = sum_e / sum_w;
                    let px = ((c as f64) + 0.5) * dx;
                    let py = ((r as f64) + 0.5) * dy;
                    if div_val > best_pos {
                        best_pos = div_val;
                        sx = px;
                        sy = py;
                    }
                    // Energy-normalized dissipation rate `div(I) / (E + E_floor)` strictly inside
                    // the plate interior (`r in 6..H-6, c in 6..W-6`) so open edges don't win
                    if r >= 6 && r + 6 < ELASTO_H && c >= 6 && c + 6 < ELASTO_W {
                        let loss_rate = div_val / (e_val + e_floor);
                        if loss_rate < best_loss {
                            best_loss = loss_rate;
                            kx = px;
                            ky = py;
                        }
                    }
                }
            }
            c += 1;
        }
        r += 1;
    }
    (sx, sy, kx, ky)
}

/// Assembles uniform-weight Neumann Poisson RHS `b = A^T eps` directly into `bos.solve_z`
/// from `bos.planes[ex_off..]` and `bos.planes[ey_off..]` (matches `BosState::compute_curl_and_assemble_neumann`).
#[inline(never)]
fn assemble_bos_neumann_rhs_into_z(
    bos: &mut BosState,
    ex_off: usize,
    ey_off: usize,
    inv_hx: f64,
    inv_hy: f64,
) {
    bos.solve_z.fill(0.0);
    let mut j = 0usize;
    while j < BOS_GRID_H {
        let row = j * BOS_GRID_W;
        let mut i = 0usize;
        while i < BOS_GRID_W {
            let u = row + i;
            if i + 1 < BOS_GRID_W {
                let v = u + 1;
                let eps_bar = 0.5 * ((bos.planes[ex_off + u] as f64) + (bos.planes[ex_off + v] as f64));
                let flux = eps_bar * inv_hx;
                bos.solve_z[u] -= flux;
                bos.solve_z[v] += flux;
            }
            if j + 1 < BOS_GRID_H {
                let v = u + BOS_GRID_W;
                let eps_bar = 0.5 * ((bos.planes[ey_off + u] as f64) + (bos.planes[ey_off + v] as f64));
                let flux = eps_bar * inv_hy;
                bos.solve_z[u] -= flux;
                bos.solve_z[v] += flux;
            }
            i += 1;
        }
        j += 1;
    }
}

#[inline(never)]
fn sort_f64_in_place(buf: &mut [f64]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let mut gap = n / 2;
    while gap > 0 {
        let mut i = gap;
        while i < n {
            let tmp = buf[i];
            let mut j = i;
            while j >= gap && buf[j - gap] > tmp {
                buf[j] = buf[j - gap];
                j -= gap;
            }
            buf[j] = tmp;
            i += 1;
        }
        gap /= 2;
    }
}

#[inline(never)]
fn median_in_place(buf: &mut [f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    sort_f64_in_place(&mut buf[..n]);
    buf[n / 2]
}
