//! Deterministic 8-Axis Physics Integrity Verifier & Anti-Gaming RLVR Reward Engine (`chronos-audit/1`).
//!
//! Lives in the `std` C-ABI crate (`native/capi`) so the `#![no_std]` Wasm kernel stays at
//! `+0 B` Arena and `101` exports while providing zero-drift, applicability-gated per-axis
//! verdicts, timestamped violation spans, and anti-gaming reward scoring across C, C++, and Python.

use chronos_spectral::native::ChronosEngine;

pub const CHRONOS_ABI_VERSION: u32 = 4;
pub const CHRONOS_AUDIT_SCHEMA_VERSION: u32 = 1;
pub const CHRONOS_AUDIT_AXIS_COUNT: usize = 8;
pub const MAX_AUDIT_SPANS: usize = 512;

pub const AXIS_TRAJECTORY: u32 = 0;
pub const AXIS_COLLISION: u32 = 1;
pub const AXIS_DEPTH_PARALLAX: u32 = 2;
pub const AXIS_FLUID_DIV: u32 = 3;
pub const AXIS_FLUID_MOMENTUM: u32 = 4;
pub const AXIS_BG_SHIMMER: u32 = 5;
pub const AXIS_TEMPORAL_CADENCE: u32 = 6;
pub const AXIS_AV_SYNC: u32 = 7;

pub const APPL_UNKNOWN: u32 = 0;
pub const APPL_ABSENT: u32 = 1;
pub const APPL_INSUFFICIENT: u32 = 2;
pub const APPL_ELIGIBLE: u32 = 3;

pub const VERDICT_NONE: u32 = 0;
pub const VERDICT_PASS: u32 = 1;
pub const VERDICT_WARN: u32 = 2;
pub const VERDICT_VIOLATED: u32 = 3;

pub const GRADE_NA: u32 = 0;
pub const GRADE_PASS: u32 = 1;
pub const GRADE_WARN: u32 = 2;
pub const GRADE_FAIL: u32 = 3;

pub const CLAIM_3D_TRANSLATION: u32 = 1 << 0;
pub const CLAIM_FIXED_FOCAL: u32 = 1 << 1;
pub const CLAIM_PLANAR_FLUID: u32 = 1 << 2;
pub const CLAIM_STATIC_BG: u32 = 1 << 3;
pub const CLAIM_DIRECT_AUDIO: u32 = 1 << 4;
pub const CLAIM_CONTINUOUS_CAPTURE: u32 = 1 << 5;

#[allow(dead_code)]
pub const NA_NO_EVIDENCE: u32 = 1 << 0;
#[allow(dead_code)]
pub const NA_TOO_SHORT: u32 = 1 << 1;
#[allow(dead_code)]
pub const NA_CAMERA_MOTION: u32 = 1 << 2;
#[allow(dead_code)]
pub const NA_HARD_CUT: u32 = 1 << 3;
#[allow(dead_code)]
pub const NA_LOW_TEXTURE: u32 = 1 << 4;
#[allow(dead_code)]
pub const NA_AMBIGUOUS: u32 = 1 << 5;
#[allow(dead_code)]
pub const NA_NO_AUDIO: u32 = 1 << 6;
#[allow(dead_code)]
pub const NA_NO_CALIB: u32 = 1 << 7;
#[allow(dead_code)]
pub const NA_DISABLED: u32 = 1 << 8;

// Specific violation codes
#[allow(dead_code)]
pub const CODE_G_SIGN_FLIP: u32 = 101;
#[allow(dead_code)]
pub const CODE_G_ACCEL_DRIFT: u32 = 102;
#[allow(dead_code)]
pub const CODE_ODE_LAW_BREAK: u32 = 103;
#[allow(dead_code)]
pub const CODE_SUPERELASTIC_RESTITUTION: u32 = 201;
#[allow(dead_code)]
pub const CODE_CONSERVATION_VIOLATED: u32 = 202;
#[allow(dead_code)]
pub const CODE_FAKE_3D_PURE_YAW: u32 = 301;
#[allow(dead_code)]
pub const CODE_STEREO_EPIPOLAR_BREAK: u32 = 302;
#[allow(dead_code)]
pub const CODE_TTC_COLLAPSE: u32 = 303;
#[allow(dead_code)]
pub const CODE_FLUID_MASS_LEAK: u32 = 401;
#[allow(dead_code)]
pub const CODE_FLUID_NONCONSERVATIVE: u32 = 501;
#[allow(dead_code)]
pub const CODE_BG_SHIMMER_CURL: u32 = 601;
#[allow(dead_code)]
pub const CODE_BG_SOLVER_NONCONV: u32 = 602;
#[allow(dead_code)]
pub const CODE_FROZEN_MID_MOTION: u32 = 701;
#[allow(dead_code)]
pub const CODE_PTS_CADENCE_BREAK: u32 = 702;
#[allow(dead_code)]
pub const CODE_TELEPORT_JUMP: u32 = 703;
#[allow(dead_code)]
pub const CODE_AV_IMPACT_LAG: u32 = 801;
#[allow(dead_code)]
pub const CODE_AV_RS_CLOCK_SKEW: u32 = 803;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChronosAuditConfig {
    pub struct_size: u32,
    pub profile_flags: u32,
    pub required_axes_mask: u32,
    pub expected_frames: u32,
    pub min_applicable_frames: u32,
    pub span_merge_gap_frames: u32,
    pub fail_severity: f32,
    pub min_coverage: f32,
    pub reserved: [u32; 2],
}

impl Default for ChronosAuditConfig {
    fn default() -> Self {
        Self {
            struct_size: core::mem::size_of::<Self>() as u32,
            profile_flags: 0,
            required_axes_mask: 0,
            expected_frames: 0,
            min_applicable_frames: 8,
            span_merge_gap_frames: 4,
            fail_severity: 0.5,
            min_coverage: 0.15,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ChronosAuditAxis {
    pub axis_id: u32,
    pub applicability: u32,
    pub verdict: u32,
    pub na_reason_mask: u32,
    pub applicable_frames: u32,
    pub warned_frames: u32,
    pub violated_frames: u32,
    pub span_count: u32,
    pub score: f32,
    pub worst_residual: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ChronosAuditSpan {
    pub axis_id: u32,
    pub code: u32,
    pub f0: u32,
    pub f1: u32,
    pub t0_us: u64,
    pub t1_us: u64,
    pub track_id: i32,
    pub severity: f32,
    pub bbox: [f32; 4],
    pub measured: f32,
    pub expected: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChronosAuditSummary {
    pub struct_size: u32,
    pub abi_version: u32,
    pub schema_version: u32,
    pub profile_flags: u32,
    pub required_axes_mask: u32,
    pub known_modalities: u32,
    pub present_modalities: u32,
    pub frames_seen: u32,
    pub frames_committed: u32,
    pub frames_rejected: u32,
    pub shots: u32,
    pub audio_blocks: u32,
    pub audio_underruns: u32,
    pub axis_count: u32,
    pub applicable_axes: u32,
    pub violated_axes: u32,
    pub warned_axes: u32,
    pub span_count: u32,
    pub spans_dropped: u32,
    pub overall_grade: u32,
    pub coverage_permille: u32,
    pub reward_milli: i32,
    pub integrity: f32,
    pub reserved: u32,
    pub axes: [ChronosAuditAxis; CHRONOS_AUDIT_AXIS_COUNT],
}

const _: () = assert!(core::mem::size_of::<ChronosAuditConfig>() == 40);
const _: () = assert!(core::mem::size_of::<ChronosAuditAxis>() == 40);
const _: () = assert!(core::mem::size_of::<ChronosAuditSpan>() == 64);
const _: () = assert!(core::mem::size_of::<ChronosAuditSummary>() == 416);

#[derive(Clone, Copy)]
struct TrackPrev {
    tid: u32,
    x: f32,
    y: f32,
    speed_px_s: f32,
    frame: u32,
    epoch: u32,
}

pub struct ClipAuditor {
    pub active: bool,
    pub finalized: bool,
    cfg: ChronosAuditConfig,

    frames_seen: u32,
    frames_committed: u32,
    frames_rejected: u32,
    last_epoch: u32,
    shots: u32,
    last_pts_us: u64,
    has_last_pts: bool,
    dt_hist_us: Vec<u64>,

    // Lightweight downsampled frame digest (64 luma samples) for frozen-duplicate mid-motion check
    prev_luma_sig: [f32; 64],
    has_prev_sig: bool,
    frozen_run: u32,

    // Per-axis accumulators
    appl_frames: [u32; CHRONOS_AUDIT_AXIS_COUNT],
    warn_frames: [u32; CHRONOS_AUDIT_AXIS_COUNT],
    viol_frames: [u32; CHRONOS_AUDIT_AXIS_COUNT],
    na_reasons: [u32; CHRONOS_AUDIT_AXIS_COUNT],
    worst_resid: [f32; CHRONOS_AUDIT_AXIS_COUNT],
    has_resid: [bool; CHRONOS_AUDIT_AXIS_COUNT],

    // Consecutive-frame streak guards (Sol 6.2: require >= 3 fresh estimates for continuous violations)
    streak_fluid_div: u32,
    streak_fluid_ns: u32,
    streak_bg_shimmer: u32,
    streak_pure_yaw: u32,
    depth_ready_frames: u32,
    depth_yaw_frames: u32,
    ttc_was_locked: bool,
    ttc_drop_streak: u32,
    ttc_max_r2: f32,

    // Trajectory / Gravity tracking
    g_samples: Vec<(u32, u64, f32)>,
    ode_r2_sum: f64,
    ode_r2_cnt: u32,
    ode_low_streak: u32,
    prev_tracks: [TrackPrev; 16],

    // Audio-Video sync tracking
    audio_blocks: u32,
    audio_underruns: u32,
    audio_rate_hz: f32,
    visual_impact_us: Vec<(u32, u64, i32)>,
    audio_peak_us: Vec<u64>,
    audio_env_hist: Vec<f32>,
    last_audio_peak_us: u64,
    rs_coh_sum: f64,
    rs_coh_cnt: u32,
    rs_skew_seen: bool,

    // Modalities
    known_modalities: u32,
    present_modalities: u32,

    // Bounded timestamped violation spans
    spans: Vec<ChronosAuditSpan>,
    spans_dropped: u32,
    cached_summary: ChronosAuditSummary,
}

impl ClipAuditor {
    pub fn new() -> Self {
        Self {
            active: false,
            finalized: false,
            cfg: ChronosAuditConfig::default(),
            frames_seen: 0,
            frames_committed: 0,
            frames_rejected: 0,
            last_epoch: 0,
            shots: 0,
            last_pts_us: 0,
            has_last_pts: false,
            dt_hist_us: Vec::with_capacity(64),
            prev_luma_sig: [0.0; 64],
            has_prev_sig: false,
            frozen_run: 0,
            appl_frames: [0; CHRONOS_AUDIT_AXIS_COUNT],
            warn_frames: [0; CHRONOS_AUDIT_AXIS_COUNT],
            viol_frames: [0; CHRONOS_AUDIT_AXIS_COUNT],
            na_reasons: [0; CHRONOS_AUDIT_AXIS_COUNT],
            worst_resid: [-1.0; CHRONOS_AUDIT_AXIS_COUNT],
            has_resid: [false; CHRONOS_AUDIT_AXIS_COUNT],
            streak_fluid_div: 0,
            streak_fluid_ns: 0,
            streak_bg_shimmer: 0,
            streak_pure_yaw: 0,
            depth_ready_frames: 0,
            depth_yaw_frames: 0,
            ttc_was_locked: false,
            ttc_drop_streak: 0,
            ttc_max_r2: 0.0,
            g_samples: Vec::new(),
            ode_r2_sum: 0.0,
            ode_r2_cnt: 0,
            ode_low_streak: 0,
            prev_tracks: [TrackPrev {
                tid: 0,
                x: 0.0,
                y: 0.0,
                speed_px_s: 0.0,
                frame: 0,
                epoch: 0,
            }; 16],
            audio_blocks: 0,
            audio_underruns: 0,
            audio_rate_hz: 0.0,
            visual_impact_us: Vec::new(),
            audio_peak_us: Vec::new(),
            audio_env_hist: Vec::with_capacity(128),
            last_audio_peak_us: 0,
            rs_coh_sum: 0.0,
            rs_coh_cnt: 0,
            rs_skew_seen: false,
            known_modalities: 0,
            present_modalities: 0,
            spans: Vec::new(),
            spans_dropped: 0,
            cached_summary: Self::empty_summary(ChronosAuditConfig::default()),
        }
    }

    fn empty_summary(cfg: ChronosAuditConfig) -> ChronosAuditSummary {
        let mut axes = [ChronosAuditAxis::default(); CHRONOS_AUDIT_AXIS_COUNT];
        for (i, ax) in axes.iter_mut().enumerate() {
            ax.axis_id = i as u32;
            ax.score = -1.0;
            ax.worst_residual = -1.0;
        }
        ChronosAuditSummary {
            struct_size: core::mem::size_of::<ChronosAuditSummary>() as u32,
            abi_version: CHRONOS_ABI_VERSION,
            schema_version: CHRONOS_AUDIT_SCHEMA_VERSION,
            profile_flags: cfg.profile_flags,
            required_axes_mask: cfg.required_axes_mask,
            known_modalities: 0,
            present_modalities: 0,
            frames_seen: 0,
            frames_committed: 0,
            frames_rejected: 0,
            shots: 0,
            audio_blocks: 0,
            audio_underruns: 0,
            axis_count: CHRONOS_AUDIT_AXIS_COUNT as u32,
            applicable_axes: 0,
            violated_axes: 0,
            warned_axes: 0,
            span_count: 0,
            spans_dropped: 0,
            overall_grade: GRADE_NA,
            coverage_permille: 0,
            reward_milli: i32::MIN,
            integrity: -1.0,
            reserved: 0,
            axes,
        }
    }

    pub fn begin(&mut self, cfg: ChronosAuditConfig) {
        *self = Self::new();
        self.cfg = cfg;
        if self.cfg.min_applicable_frames == 0 {
            self.cfg.min_applicable_frames = 8;
        }
        if self.cfg.fail_severity <= 0.0 || !self.cfg.fail_severity.is_finite() {
            self.cfg.fail_severity = 0.5;
        }
        if self.cfg.min_coverage < 0.0 || !self.cfg.min_coverage.is_finite() {
            self.cfg.min_coverage = 0.15;
        }
        self.active = true;
        self.finalized = false;
        self.cached_summary = Self::empty_summary(self.cfg);
    }

    pub fn end(&mut self) {
        self.active = false;
    }

    pub fn record_audio_block(&mut self, samples: &[f32], t_end_us: u64, rate_hz: f32, ok: bool) {
        if !self.active {
            return;
        }
        self.known_modalities |= 1 << 4;
        if !ok {
            self.audio_underruns = self.audio_underruns.saturating_add(1);
            return;
        }
        self.present_modalities |= 1 << 4;
        self.audio_blocks = self.audio_blocks.saturating_add(1);
        self.audio_rate_hz = rate_hz;
        if samples.is_empty() || !rate_hz.is_finite() || rate_hz < 4000.0 {
            return;
        }
        // Scan 10 ms sub-windows for sharp acoustic impact transients (>= 6x robust MAD above baseline)
        let win = ((rate_hz * 0.010).round() as usize).max(16);
        let dur_us = ((samples.len() as f64) * 1_000_000.0 / (rate_hz as f64)).round() as u64;
        let t_start_us = t_end_us.saturating_sub(dur_us);
        let mut pos = 0usize;
        while pos < samples.len() {
            let end = (pos + win).min(samples.len());
            let mut sum_sq = 0.0f64;
            let mut peak = 0.0f32;
            for &s in &samples[pos..end] {
                if s.is_finite() {
                    let a = s.abs();
                    if a > peak {
                        peak = a;
                    }
                    sum_sq += (s as f64) * (s as f64);
                }
            }
            let rms = (sum_sq / ((end - pos).max(1) as f64)).sqrt() as f32;
            let env = 0.5 * (rms + peak);
            let t_mid_us = t_start_us.saturating_add(
                (((pos + end) as f64 * 0.5) * 1_000_000.0 / (rate_hz as f64)).round() as u64,
            );
            if self.audio_env_hist.len() >= 12 {
                let mut sorted = self.audio_env_hist.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let med = sorted[sorted.len() / 2];
                let mut devs: Vec<f32> = sorted.iter().map(|&v| (v - med).abs()).collect();
                devs.sort_by(|a, b| a.total_cmp(b));
                let mad = (devs[devs.len() / 2] * 1.4826).max(0.005);
                if env > med + 6.0 * mad
                    && env > 0.08
                    && (self.audio_peak_us.is_empty()
                        || t_mid_us.saturating_sub(self.last_audio_peak_us) >= 60_000)
                {
                    if self.audio_peak_us.len() < 256 {
                        self.audio_peak_us.push(t_mid_us);
                    }
                    self.last_audio_peak_us = t_mid_us;
                }
            }
            if self.audio_env_hist.len() >= 128 {
                self.audio_env_hist.remove(0);
            }
            self.audio_env_hist.push(env);
            pos = end;
        }
    }

    pub fn sample_pre_step_luma_sig(&self, rgba: &[u8], w: usize, h: usize) -> [f32; 64] {
        let mut sig = [0.0f32; 64];
        if w == 0 || h == 0 || rgba.len() < w * h * 4 {
            return sig;
        }
        let mut idx = 0usize;
        for gy in 0..8usize {
            let y0 = (gy * h) / 8;
            let y1 = (((gy + 1) * h) / 8).max(y0 + 1);
            for gx in 0..8usize {
                let x0 = (gx * w) / 8;
                let x1 = (((gx + 1) * w) / 8).max(x0 + 1);
                let mut sum = 0u64;
                let mut cnt = 0u32;
                let mut py = y0;
                while py < y1 {
                    let row = py * w;
                    let mut px = x0;
                    while px < x1 {
                        let off = (row + px) * 4;
                        let y = (rgba[off] as u64 * 77
                            + rgba[off + 1] as u64 * 150
                            + rgba[off + 2] as u64 * 29)
                            >> 8;
                        sum += y;
                        cnt += 1;
                        px += 2;
                    }
                    py += 2;
                }
                sig[idx] = if cnt > 0 { (sum as f32) / (cnt as f32) } else { 0.0 };
                idx += 1;
            }
        }
        sig
    }

    fn update_resid(&mut self, axis: usize, val: f32) {
        if val.is_finite() {
            if !self.has_resid[axis] || val > self.worst_resid[axis] {
                self.worst_resid[axis] = val;
                self.has_resid[axis] = true;
            }
        }
    }

    fn record_span(
        &mut self,
        axis_id: u32,
        code: u32,
        frame: u32,
        t_us: u64,
        dt_us: u64,
        track_id: i32,
        severity: f32,
        bbox: [f32; 4],
        measured: f32,
        expected: f32,
    ) {
        let sev = if severity.is_finite() {
            severity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let t1 = t_us.saturating_add(dt_us.max(1));
        let gap = self.cfg.span_merge_gap_frames;
        if let Some(last) = self.spans.iter_mut().rev().find(|s| {
            s.axis_id == axis_id
                && s.code == code
                && s.track_id == track_id
                && frame <= s.f1.saturating_add(gap.max(1))
        }) {
            last.f1 = frame;
            last.t1_us = last.t1_us.max(t1);
            if sev > last.severity {
                last.severity = sev;
                last.measured = measured;
                last.expected = expected;
                last.bbox = bbox;
            }
            return;
        }
        if self.spans.len() < MAX_AUDIT_SPANS {
            self.spans.push(ChronosAuditSpan {
                axis_id,
                code,
                f0: frame,
                f1: frame,
                t0_us: t_us,
                t1_us: t1,
                track_id,
                severity: sev,
                bbox,
                measured,
                expected,
            });
        } else {
            self.spans_dropped = self.spans_dropped.saturating_add(1);
        }
    }

    pub fn observe_step(
        &mut self,
        eng: &ChronosEngine,
        dt_sec: f32,
        pts_us_opt: Option<u64>,
        luma_sig: [f32; 64],
    ) {
        if !self.active {
            return;
        }
        self.finalized = false;
        let frame_idx = self.frames_seen;
        self.frames_seen = self.frames_seen.saturating_add(1);

        let w1 = eng.wave1_telemetry();
        let w2 = eng.wave2_telemetry();
        let telem = eng.telemetry();
        let (fw, fh) = eng.frame_dims();
        let inv_w = 1.0 / (fw.max(1) as f32);
        let inv_h = 1.0 / (fh.max(1) as f32);

        let ingest_status = w1[8];
        if ingest_status == 2 || ingest_status == 3 || ingest_status == 4 {
            self.frames_rejected = self.frames_rejected.saturating_add(1);
            return;
        }
        self.frames_committed = self.frames_committed.saturating_add(1);

        let epoch = w1[6].max(1);
        let epoch_changed = self.last_epoch != 0 && epoch != self.last_epoch;
        if self.shots == 0 || epoch_changed {
            self.shots = self.shots.saturating_add(1);
            self.streak_fluid_div = 0;
            self.streak_fluid_ns = 0;
            self.streak_bg_shimmer = 0;
            self.streak_pure_yaw = 0;
            self.ttc_was_locked = false;
            self.ttc_drop_streak = 0;
            self.frozen_run = 0;
            self.has_prev_sig = false;
            self.ode_low_streak = 0;
            for pt in &mut self.prev_tracks {
                pt.tid = 0;
            }
        }
        self.last_epoch = epoch;

        let dt_us = ((dt_sec.clamp(0.001, 1.0) as f64) * 1_000_000.0).round() as u64;
        let t_us = match pts_us_opt {
            Some(p) => p,
            None => {
                if self.has_last_pts {
                    self.last_pts_us.saturating_add(dt_us)
                } else {
                    0
                }
            }
        };

        // ---- Axis 6: Temporal Cadence & Mid-Motion Stutter ----
        let cam_motion_state = telem[1257].round() as i32;
        let active_n = telem[3].round() as i32;
        let mut max_track_speed_px_f = 0.0f32;
        for m in 0..16usize {
            let base = 32 + m * 56;
            if telem[base] > 0.5 {
                let kbase = 1120 + m * 8;
                let spd_f = telem[kbase + 1] * dt_sec;
                if spd_f.is_finite() && spd_f > max_track_speed_px_f {
                    max_track_speed_px_f = spd_f;
                }
            }
        }
        let (luma_mad, max_tile_diff) = if self.has_prev_sig {
            let mut acc = 0.0f32;
            let mut mxd = 0.0f32;
            for k in 0..64 {
                let d = (self.prev_luma_sig[k] - luma_sig[k]).abs();
                acc += d;
                if d > mxd {
                    mxd = d;
                }
            }
            (acc / (64.0 * 255.0), mxd)
        } else {
            (1.0, 255.0)
        };
        let scene_has_motion = active_n > 0 && max_track_speed_px_f >= 1.5;
        if self.has_last_pts && !epoch_changed {
            if t_us <= self.last_pts_us {
                self.appl_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                self.viol_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                self.update_resid(AXIS_TEMPORAL_CADENCE as usize, 1.0);
                self.record_span(
                    AXIS_TEMPORAL_CADENCE,
                    CODE_PTS_CADENCE_BREAK,
                    frame_idx,
                    t_us,
                    dt_us,
                    -1,
                    0.95,
                    [0.0, 0.0, 1.0, 1.0],
                    0.0,
                    dt_us as f32,
                );
            } else {
                let step_us = t_us - self.last_pts_us;
                if self.dt_hist_us.len() >= 8 {
                    let mut tmp = self.dt_hist_us.clone();
                    tmp.sort_unstable();
                    let med_dt = tmp[tmp.len() / 2].max(1000);
                    let ratio = (step_us as f32) / (med_dt as f32);
                    if scene_has_motion {
                        self.appl_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                        self.update_resid(AXIS_TEMPORAL_CADENCE as usize, (ratio - 1.0).abs());
                    }
                    if ratio > 2.6 || ratio < 0.35 {
                        self.viol_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                        self.record_span(
                            AXIS_TEMPORAL_CADENCE,
                            CODE_PTS_CADENCE_BREAK,
                            frame_idx,
                            t_us,
                            step_us,
                            -1,
                            ((ratio - 1.0).abs() / 3.0).clamp(0.55, 1.0),
                            [0.0, 0.0, 1.0, 1.0],
                            step_us as f32,
                            med_dt as f32,
                        );
                    }
                } else if scene_has_motion {
                    self.appl_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                }
                if self.dt_hist_us.len() >= 64 {
                    self.dt_hist_us.remove(0);
                }
                self.dt_hist_us.push(step_us);
            }
        }
        if self.has_prev_sig && scene_has_motion && luma_mad <= 0.0006 && max_tile_diff <= 0.35 {
            self.frozen_run = self.frozen_run.saturating_add(1);
            if self.frozen_run >= 2 {
                self.appl_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                self.update_resid(AXIS_TEMPORAL_CADENCE as usize, self.frozen_run as f32);
                if (self.cfg.profile_flags & CLAIM_CONTINUOUS_CAPTURE) != 0 {
                    self.viol_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                    self.record_span(
                        AXIS_TEMPORAL_CADENCE,
                        CODE_FROZEN_MID_MOTION,
                        frame_idx,
                        t_us,
                        dt_us,
                        -1,
                        0.85,
                        [0.0, 0.0, 1.0, 1.0],
                        luma_mad,
                        max_track_speed_px_f,
                    );
                } else {
                    self.warn_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                }
            }
        } else {
            self.frozen_run = 0;
        }
        self.prev_luma_sig = luma_sig;
        self.has_prev_sig = true;
        self.last_pts_us = t_us;
        self.has_last_pts = true;

        // ---- Axis 0: Trajectory Dynamics & Gravity + Axis 1: Collisions + Teleport Check ----
        if active_n > 0 {
            self.known_modalities |= 1 << 0;
            self.present_modalities |= 1 << 0;
        }
        let g_hat = telem[1248];
        let g_conf = telem[1249];
        let g_arcs = telem[1259].round() as u32;
        if cam_motion_state == 0 && g_conf >= 0.65 && g_arcs >= 1 && g_hat.is_finite() && g_hat.abs() > 10.0 {
            self.g_samples.push((frame_idx, t_us, g_hat));
        }

        let mut frame_had_traj = false;
        let mut frame_best_r2 = 0.0f32;
        let mut frame_had_r2 = false;
        for m in 0..16usize {
            let base = 32 + m * 56;
            if telem[base] <= 0.5 {
                self.prev_tracks[m].tid = 0;
                continue;
            }
            let tid = telem[base + 1].round() as u32;
            let px = telem[base + 2];
            let py = telem[base + 3];
            let vis = telem[base + 10];
            let bridged = telem[base + 11];
            let ebase = 992 + m * 8;
            let kbase = 1120 + m * 8;
            let pbase = 1280 + m * 16;
            let spd = telem[kbase + 1];
            let r2 = telem[pbase + 6];
            let rest_e = telem[pbase + 14];
            let strike_fired = telem[ebase] > 0.5;

            if vis > 0.5 && bridged < 0.5 && spd >= 12.0 {
                frame_had_traj = true;
                if r2.is_finite() && r2 > 0.0 && r2 <= 1.001 {
                    if !frame_had_r2 || r2 > frame_best_r2 {
                        frame_best_r2 = r2.min(1.0);
                        frame_had_r2 = true;
                    }
                }
            }

            // Sudden interior teleport check (unbridged same-ID jump >> predicted reach without cut)
            let prev = self.prev_tracks[m];
            if prev.tid == tid
                && prev.epoch == epoch
                && frame_idx == prev.frame.saturating_add(1)
                && vis > 0.5
                && bridged < 0.5
                && cam_motion_state == 0
            {
                let dx = px - prev.x;
                let dy = py - prev.y;
                let jump = (dx * dx + dy * dy).sqrt();
                let reach = (prev.speed_px_s * dt_sec * 4.5).max(48.0 * (fw as f32 / 640.0));
                if jump > reach && jump.is_finite() {
                    self.appl_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                    self.viol_frames[AXIS_TEMPORAL_CADENCE as usize] += 1;
                    self.update_resid(AXIS_TEMPORAL_CADENCE as usize, jump / reach);
                    self.record_span(
                        AXIS_TEMPORAL_CADENCE,
                        CODE_TELEPORT_JUMP,
                        frame_idx,
                        t_us,
                        dt_us,
                        tid as i32,
                        (jump / (reach * 2.0)).clamp(0.55, 1.0),
                        [
                            (px * inv_w - 0.04).clamp(0.0, 1.0),
                            (py * inv_h - 0.04).clamp(0.0, 1.0),
                            0.08,
                            0.08,
                        ],
                        jump,
                        reach,
                    );
                }
            }
            self.prev_tracks[m] = TrackPrev {
                tid,
                x: px,
                y: py,
                speed_px_s: spd,
                frame: frame_idx,
                epoch,
            };

            // Collisions & visual impacts
            if strike_fired {
                if self.visual_impact_us.len() < 128
                    && self
                        .visual_impact_us
                        .last()
                        .map_or(true, |last| t_us.saturating_sub(last.1) >= 80_000)
                {
                    self.visual_impact_us.push((frame_idx, t_us, tid as i32));
                }
                if rest_e.is_finite() && rest_e > 0.05 && rest_e < 5.0 {
                    self.appl_frames[AXIS_COLLISION as usize] += 1;
                    let excess = (rest_e - 1.0).max(0.0);
                    self.update_resid(AXIS_COLLISION as usize, excess);
                    if rest_e > 1.18 {
                        self.viol_frames[AXIS_COLLISION as usize] += 1;
                        self.record_span(
                            AXIS_COLLISION,
                            CODE_SUPERELASTIC_RESTITUTION,
                            frame_idx,
                            t_us,
                            dt_us,
                            tid as i32,
                            ((rest_e - 1.0) / 0.5).clamp(0.55, 1.0),
                            [
                                (px * inv_w - 0.04).clamp(0.0, 1.0),
                                (py * inv_h - 0.04).clamp(0.0, 1.0),
                                0.08,
                                0.08,
                            ],
                            rest_e,
                            1.0,
                        );
                    }
                }
            }
        }

        if frame_had_traj && cam_motion_state == 0 {
            self.appl_frames[AXIS_TRAJECTORY as usize] += 1;
            if frame_had_r2 {
                self.ode_r2_sum += frame_best_r2 as f64;
                self.ode_r2_cnt += 1;
                let resid = (1.0 - frame_best_r2).max(0.0);
                self.update_resid(AXIS_TRAJECTORY as usize, resid);
                if frame_best_r2 < 0.72 {
                    self.ode_low_streak += 1;
                    if self.ode_low_streak >= 4 {
                        self.viol_frames[AXIS_TRAJECTORY as usize] += 1;
                        self.record_span(
                            AXIS_TRAJECTORY,
                            CODE_ODE_LAW_BREAK,
                            frame_idx,
                            t_us,
                            dt_us,
                            -1,
                            (1.0 - frame_best_r2).clamp(0.55, 1.0),
                            [0.0, 0.0, 1.0, 1.0],
                            frame_best_r2,
                            0.92,
                        );
                    }
                } else {
                    self.ode_low_streak = 0;
                }
            }
        } else if frame_had_traj && cam_motion_state != 0 {
            self.na_reasons[AXIS_TRAJECTORY as usize] |= NA_CAMERA_MOTION;
        }

        // ---- Axis 2: 3D Parallax, Pure-Yaw & Optical Looming TTC ----
        let dt_t = eng.depth_telemetry();
        let depth_mode = dt_t[1];
        if depth_mode != 0 {
            self.known_modalities |= 1 << 1;
            let ds = dt_t[2];
            let rot_ratio = f32::from_bits(dt_t[17]);
            let ttc_r2 = f32::from_bits(dt_t[25]);
            let epi_rms = f32::from_bits(dt_t[28]);
            if ds == 2 || ds == 3 {
                // READY_PARALLAX or READY_TTC_ONLY
                self.present_modalities |= 1 << 1;
                self.depth_ready_frames += 1;
                self.streak_pure_yaw = 0;
                self.appl_frames[AXIS_DEPTH_PARALLAX as usize] += 1;
                if ds == 3 && ttc_r2.is_finite() && ttc_r2 > 0.85 {
                    self.ttc_was_locked = true;
                    self.ttc_drop_streak = 0;
                    if ttc_r2 > self.ttc_max_r2 {
                        self.ttc_max_r2 = ttc_r2;
                    }
                    self.update_resid(AXIS_DEPTH_PARALLAX as usize, (1.0 - ttc_r2).max(0.0));
                }
                if depth_mode == 2 && epi_rms.is_finite() && epi_rms > 2.5 {
                    self.viol_frames[AXIS_DEPTH_PARALLAX as usize] += 1;
                    self.update_resid(AXIS_DEPTH_PARALLAX as usize, epi_rms);
                    self.record_span(
                        AXIS_DEPTH_PARALLAX,
                        CODE_STEREO_EPIPOLAR_BREAK,
                        frame_idx,
                        t_us,
                        dt_us,
                        -1,
                        (epi_rms / 5.0).clamp(0.55, 1.0),
                        [0.0, 0.0, 1.0, 1.0],
                        epi_rms,
                        1.5,
                    );
                }
            } else if ds == 4 {
                // DEPTH_STATE_REFUSED_PURE_ROTATION
                self.depth_yaw_frames += 1;
                self.streak_pure_yaw += 1;
                if (self.cfg.profile_flags & CLAIM_3D_TRANSLATION) != 0 {
                    self.appl_frames[AXIS_DEPTH_PARALLAX as usize] += 1;
                    let resid = if rot_ratio.is_finite() {
                        (1.0 - rot_ratio).clamp(0.0, 1.0)
                    } else {
                        0.9
                    };
                    self.update_resid(AXIS_DEPTH_PARALLAX as usize, resid);
                    if self.streak_pure_yaw >= 3 {
                        self.viol_frames[AXIS_DEPTH_PARALLAX as usize] += 1;
                        self.record_span(
                            AXIS_DEPTH_PARALLAX,
                            CODE_FAKE_3D_PURE_YAW,
                            frame_idx,
                            t_us,
                            dt_us,
                            -1,
                            0.85,
                            [0.0, 0.0, 1.0, 1.0],
                            rot_ratio,
                            0.25,
                        );
                    }
                } else {
                    self.na_reasons[AXIS_DEPTH_PARALLAX as usize] |= NA_CAMERA_MOTION;
                }
            } else if ds == 1 && self.ttc_was_locked && (self.cfg.profile_flags & CLAIM_FIXED_FOCAL) != 0 {
                self.ttc_drop_streak += 1;
                if self.ttc_drop_streak >= 6 {
                    self.appl_frames[AXIS_DEPTH_PARALLAX as usize] += 1;
                    self.viol_frames[AXIS_DEPTH_PARALLAX as usize] += 1;
                    self.update_resid(AXIS_DEPTH_PARALLAX as usize, 1.0);
                    self.record_span(
                        AXIS_DEPTH_PARALLAX,
                        CODE_TTC_COLLAPSE,
                        frame_idx,
                        t_us,
                        dt_us,
                        -1,
                        0.75,
                        [0.0, 0.0, 1.0, 1.0],
                        0.0,
                        self.ttc_max_r2,
                    );
                }
            }
        } else {
            self.na_reasons[AXIS_DEPTH_PARALLAX as usize] |= NA_DISABLED;
        }

        // ---- Axes 3 & 4: Fluid Mass Conservation (div u) & Navier-Stokes Curl Integrability ----
        let wt = eng.wind_telemetry();
        let wind_mode = wt[1];
        if wind_mode != 0 {
            self.known_modalities |= 1 << 2;
            let ws = wt[0];
            let w_reason = wt[2];
            let w_cells = wt[4];
            let div_frac = f32::from_bits(wt[22]);
            let ns_resid = f32::from_bits(wt[21]);
            let low_support = (w_reason & (1 << 9)) != 0;
            let wind_stamp_fresh = (w2[16 + 4 * 8] & 3) == 1;
            let is_ready = (ws == 2 || ws == 3) && wind_stamp_fresh && !low_support;
            let is_div_refusal = ws == 4 && (w_reason & (1 << 3)) != 0 && w_cells >= 16;
            let is_ns_refusal = ws == 4 && (w_reason & (1 << 4)) != 0 && w_cells >= 16;

            if is_ready || is_div_refusal {
                self.present_modalities |= 1 << 2;
                self.appl_frames[AXIS_FLUID_DIV as usize] += 1;
                self.update_resid(AXIS_FLUID_DIV as usize, div_frac);
                if is_div_refusal || (div_frac.is_finite() && div_frac > 0.25) {
                    self.streak_fluid_div += 1;
                    if self.streak_fluid_div >= 3 {
                        if (self.cfg.profile_flags & CLAIM_PLANAR_FLUID) != 0 {
                            self.viol_frames[AXIS_FLUID_DIV as usize] += 1;
                            self.record_span(
                                AXIS_FLUID_DIV,
                                CODE_FLUID_MASS_LEAK,
                                frame_idx,
                                t_us,
                                dt_us,
                                -1,
                                (div_frac / 0.5).clamp(0.55, 1.0),
                                [0.0, 0.0, 1.0, 1.0],
                                div_frac,
                                0.25,
                            );
                        } else {
                            // Under unknown 3D support (e.g. Daphnia laser sheet), warn rather than falsely claiming mass loss
                            self.warn_frames[AXIS_FLUID_DIV as usize] += 1;
                        }
                    }
                } else {
                    self.streak_fluid_div = 0;
                }
            }

            if is_ready || is_ns_refusal {
                self.present_modalities |= 1 << 2;
                self.appl_frames[AXIS_FLUID_MOMENTUM as usize] += 1;
                self.update_resid(AXIS_FLUID_MOMENTUM as usize, ns_resid);
                if is_ns_refusal || (ns_resid.is_finite() && ns_resid > 0.35) {
                    self.streak_fluid_ns += 1;
                    if self.streak_fluid_ns >= 3 {
                        if (self.cfg.profile_flags & CLAIM_PLANAR_FLUID) != 0 {
                            self.viol_frames[AXIS_FLUID_MOMENTUM as usize] += 1;
                            self.record_span(
                                AXIS_FLUID_MOMENTUM,
                                CODE_FLUID_NONCONSERVATIVE,
                                frame_idx,
                                t_us,
                                dt_us,
                                -1,
                                (ns_resid / 0.7).clamp(0.55, 1.0),
                                [0.0, 0.0, 1.0, 1.0],
                                ns_resid,
                                0.35,
                            );
                        } else {
                            self.warn_frames[AXIS_FLUID_MOMENTUM as usize] += 1;
                        }
                    }
                } else {
                    self.streak_fluid_ns = 0;
                }
            }
        } else {
            self.na_reasons[AXIS_FLUID_DIV as usize] |= NA_DISABLED;
            self.na_reasons[AXIS_FLUID_MOMENTUM as usize] |= NA_DISABLED;
        }

        // ---- Axis 5: Static Background Non-Conservative Shimmer (BOS Plaquette Curl) ----
        let bt = eng.bos_telemetry();
        let bs = bt[2];
        let b_reason = bt[3];
        if bs != 0 {
            self.known_modalities |= 1 << 3;
            let curl_rms = f32::from_bits(bt[17]);
            let tripod_broken = (b_reason & 0x0e) != 0; // EXT_PAN_CUT | INT_SHIFT | PHOTOM_MSE
            let nonconv = (b_reason & 0x40) != 0;
            if bs == 2 && !tripod_broken {
                self.present_modalities |= 1 << 3;
                self.appl_frames[AXIS_BG_SHIMMER as usize] += 1;
                self.update_resid(AXIS_BG_SHIMMER as usize, curl_rms);
                if (curl_rms.is_finite() && curl_rms > 0.06) || nonconv {
                    self.streak_bg_shimmer += 1;
                    if self.streak_bg_shimmer >= 3 {
                        if (self.cfg.profile_flags & CLAIM_STATIC_BG) != 0 || curl_rms > 0.08 {
                            self.viol_frames[AXIS_BG_SHIMMER as usize] += 1;
                            self.record_span(
                                AXIS_BG_SHIMMER,
                                if nonconv {
                                    CODE_BG_SOLVER_NONCONV
                                } else {
                                    CODE_BG_SHIMMER_CURL
                                },
                                frame_idx,
                                t_us,
                                dt_us,
                                -1,
                                (curl_rms / 0.15).clamp(0.55, 1.0),
                                [0.0, 0.0, 1.0, 1.0],
                                curl_rms,
                                0.06,
                            );
                        } else {
                            self.warn_frames[AXIS_BG_SHIMMER as usize] += 1;
                        }
                    }
                } else {
                    self.streak_bg_shimmer = 0;
                }
            } else if tripod_broken {
                self.na_reasons[AXIS_BG_SHIMMER as usize] |= NA_CAMERA_MOTION;
            }
        } else {
            self.na_reasons[AXIS_BG_SHIMMER as usize] |= NA_DISABLED;
        }

        // ---- Axis 7 (online RS part): Rolling-Shutter Audio Coherence / Clock Skew ----
        let vt = eng.vibro_telemetry();
        let et = eng.elasto_telemetry();
        if vt[0] == 3 || et[0] == 3 {
            let coh = if et[0] == 3 {
                f32::from_bits(et[15])
            } else {
                f32::from_bits(vt[12])
            };
            if coh.is_finite() && coh > 0.0 {
                self.appl_frames[AXIS_AV_SYNC as usize] += 1;
                self.rs_coh_sum += coh as f64;
                self.rs_coh_cnt += 1;
                self.update_resid(AXIS_AV_SYNC as usize, (1.0 - coh).max(0.0));
            }
        }
        if (vt[2] & (1 << 4)) != 0 || (et[2] & (1 << 4)) != 0 {
            self.rs_skew_seen = true;
            self.appl_frames[AXIS_AV_SYNC as usize] += 1;
            self.viol_frames[AXIS_AV_SYNC as usize] += 1;
            self.record_span(
                AXIS_AV_SYNC,
                CODE_AV_RS_CLOCK_SKEW,
                frame_idx,
                t_us,
                dt_us,
                -1,
                0.9,
                [0.0, 0.0, 1.0, 1.0],
                1.0,
                0.0,
            );
        }
    }

    pub fn finalize(&mut self) -> ChronosAuditSummary {
        if self.finalized {
            return self.cached_summary;
        }
        self.finalized = true;

        // Post-pass 1: Cross-arc gravity sign-flip / magnitude-drift check
        if self.g_samples.len() >= 6 {
            let mut g_vals: Vec<f32> = self.g_samples.iter().map(|s| s.2).collect();
            g_vals.sort_by(|a, b| a.total_cmp(b));
            let g_med = g_vals[g_vals.len() / 2];
            if g_med.abs() > 10.0 {
                let mut max_rel_drift = 0.0f32;
                let mut sign_flip = None;
                let mut drift_sample = None;
                for &(f, tus, gv) in &self.g_samples {
                    if gv * g_med < 0.0 {
                        sign_flip = Some((f, tus, gv));
                    }
                    let rel = (gv - g_med).abs() / g_med.abs();
                    if rel > max_rel_drift {
                        max_rel_drift = rel;
                        drift_sample = Some((f, tus, gv));
                    }
                }
                self.update_resid(AXIS_TRAJECTORY as usize, max_rel_drift);
                if let Some((f, tus, gv)) = sign_flip {
                    self.viol_frames[AXIS_TRAJECTORY as usize] =
                        self.viol_frames[AXIS_TRAJECTORY as usize].max(1);
                    self.record_span(
                        AXIS_TRAJECTORY,
                        CODE_G_SIGN_FLIP,
                        f,
                        tus,
                        33_333,
                        -1,
                        1.0,
                        [0.0, 0.0, 1.0, 1.0],
                        gv,
                        g_med,
                    );
                } else if max_rel_drift > 0.30 {
                    if let Some((f, tus, gv)) = drift_sample {
                        self.viol_frames[AXIS_TRAJECTORY as usize] =
                            self.viol_frames[AXIS_TRAJECTORY as usize].max(1);
                        self.record_span(
                            AXIS_TRAJECTORY,
                            CODE_G_ACCEL_DRIFT,
                            f,
                            tus,
                            33_333,
                            -1,
                            (max_rel_drift / 0.6).clamp(0.55, 1.0),
                            [0.0, 0.0, 1.0, 1.0],
                            gv,
                            g_med,
                        );
                    }
                }
            }
        }

        // Post-pass 2: Audio-Video discrete impact transient synchronization check
        if self.audio_blocks == 0 {
            self.na_reasons[AXIS_AV_SYNC as usize] |= NA_NO_AUDIO;
        } else if self.visual_impact_us.len() >= 2 && !self.audio_peak_us.is_empty() {
            let mut lags_ms: Vec<(u32, u64, i32, f32)> = Vec::new();
            for &(vf, v_us, tid) in &self.visual_impact_us {
                let mut best_lag_ms = f32::MAX;
                let mut best_signed = 0.0f32;
                for &a_us in &self.audio_peak_us {
                    let signed = ((a_us as i64) - (v_us as i64)) as f32 * 0.001;
                    if signed.abs() < best_lag_ms {
                        best_lag_ms = signed.abs();
                        best_signed = signed;
                    }
                }
                if best_lag_ms <= 450.0 {
                    lags_ms.push((vf, v_us, tid, best_signed));
                }
            }
            if lags_ms.len() >= 2 {
                self.appl_frames[AXIS_AV_SYNC as usize] =
                    self.appl_frames[AXIS_AV_SYNC as usize].max(lags_ms.len() as u32);
                let mut abs_lags: Vec<f32> = lags_ms.iter().map(|x| x.3.abs()).collect();
                abs_lags.sort_by(|a, b| a.total_cmp(b));
                let med_abs_lag = abs_lags[abs_lags.len() / 2];
                self.update_resid(AXIS_AV_SYNC as usize, med_abs_lag);
                let tol_ms = 55.0f32;
                if med_abs_lag > tol_ms {
                    for &(vf, v_us, tid, signed) in &lags_ms {
                        if signed.abs() > tol_ms {
                            if (self.cfg.profile_flags & CLAIM_DIRECT_AUDIO) != 0 {
                                self.viol_frames[AXIS_AV_SYNC as usize] += 1;
                                self.record_span(
                                    AXIS_AV_SYNC,
                                    CODE_AV_IMPACT_LAG,
                                    vf,
                                    v_us,
                                    33_333,
                                    tid,
                                    (signed.abs() / 180.0).clamp(0.55, 1.0),
                                    [0.0, 0.0, 1.0, 1.0],
                                    signed,
                                    0.0,
                                );
                            } else {
                                self.warn_frames[AXIS_AV_SYNC as usize] += 1;
                            }
                        }
                    }
                }
            } else {
                self.na_reasons[AXIS_AV_SYNC as usize] |= NA_NO_EVIDENCE;
            }
        } else if self.rs_coh_cnt == 0 && !self.rs_skew_seen {
            self.na_reasons[AXIS_AV_SYNC as usize] |= NA_NO_EVIDENCE;
        }

        // Assemble final summary
        let mut sum = Self::empty_summary(self.cfg);
        sum.known_modalities = self.known_modalities;
        sum.present_modalities = self.present_modalities;
        sum.frames_seen = self.frames_seen;
        sum.frames_committed = self.frames_committed;
        sum.frames_rejected = self.frames_rejected;
        sum.shots = self.shots;
        sum.audio_blocks = self.audio_blocks;
        sum.audio_underruns = self.audio_underruns;
        sum.span_count = self.spans.len() as u32;
        sum.spans_dropped = self.spans_dropped;

        let min_f = self.cfg.min_applicable_frames;
        let mut score_sum = 0.0f64;
        let mut score_cnt = 0u32;
        let mut any_fail_span = false;

        for ax_i in 0..CHRONOS_AUDIT_AXIS_COUNT {
            let af = self.appl_frames[ax_i];
            let wf = self.warn_frames[ax_i];
            let vf = self.viol_frames[ax_i];
            let span_cnt = self.spans.iter().filter(|s| s.axis_id == ax_i as u32).count() as u32;
            let max_sev = self
                .spans
                .iter()
                .filter(|s| s.axis_id == ax_i as u32)
                .map(|s| s.severity)
                .fold(0.0f32, f32::max);
            if max_sev >= self.cfg.fail_severity {
                any_fail_span = true;
            }

            // Discrete event axes (collision, AV impact sync) require >= 1..2 events; continuous axes require >= min_f
            let req_frames = match ax_i as u32 {
                AXIS_COLLISION | AXIS_AV_SYNC => 1u32,
                _ => min_f,
            };

            let (appl, verdict, score) = if af >= req_frames || vf > 0 {
                let viol_frac = (vf as f32) / (af.max(vf).max(1) as f32);
                let warn_frac = (wf as f32) / (af.max(wf).max(1) as f32);
                let sc = if vf > 0 {
                    (1.0 - (0.5 * viol_frac + 0.5 * max_sev.max(0.35))).clamp(0.0, 0.85)
                } else if wf > 0 {
                    (1.0 - 0.35 * warn_frac).clamp(0.5, 0.98)
                } else if ax_i as u32 == AXIS_TRAJECTORY && self.ode_r2_cnt > 0 {
                    ((self.ode_r2_sum / (self.ode_r2_cnt as f64)) as f32).clamp(0.0, 1.0)
                } else if ax_i as u32 == AXIS_AV_SYNC && self.rs_coh_cnt > 0 {
                    ((self.rs_coh_sum / (self.rs_coh_cnt as f64)) as f32).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let vd = if vf > 0 {
                    VERDICT_VIOLATED
                } else if wf > 0 {
                    VERDICT_WARN
                } else {
                    VERDICT_PASS
                };
                (APPL_ELIGIBLE, vd, sc)
            } else if (self.na_reasons[ax_i] & (NA_NO_AUDIO | NA_DISABLED)) != 0 {
                (APPL_ABSENT, VERDICT_NONE, -1.0)
            } else if af > 0 {
                self.na_reasons[ax_i] |= NA_TOO_SHORT;
                (APPL_INSUFFICIENT, VERDICT_NONE, -1.0)
            } else {
                self.na_reasons[ax_i] |= NA_NO_EVIDENCE;
                (APPL_UNKNOWN, VERDICT_NONE, -1.0)
            };

            sum.axes[ax_i] = ChronosAuditAxis {
                axis_id: ax_i as u32,
                applicability: appl,
                verdict,
                na_reason_mask: if appl == APPL_ELIGIBLE {
                    0
                } else {
                    self.na_reasons[ax_i]
                },
                applicable_frames: af,
                warned_frames: wf,
                violated_frames: vf,
                span_count: span_cnt,
                score,
                worst_residual: if self.has_resid[ax_i] {
                    self.worst_resid[ax_i]
                } else {
                    -1.0
                },
            };

            if appl == APPL_ELIGIBLE {
                sum.applicable_axes += 1;
                score_sum += score as f64;
                score_cnt += 1;
                if verdict == VERDICT_VIOLATED {
                    sum.violated_axes += 1;
                } else if verdict == VERDICT_WARN {
                    sum.warned_axes += 1;
                }
            }
        }

        let denom_frames = self.cfg.expected_frames.max(self.frames_committed).max(1);
        let max_appl_f = self.appl_frames.iter().copied().max().unwrap_or(0);
        let cov = ((max_appl_f as f64) / (denom_frames as f64)).clamp(0.0, 1.0);
        sum.coverage_permille = (cov * 1000.0).round() as u32;

        if score_cnt > 0 {
            sum.integrity = (score_sum / (score_cnt as f64)) as f32;
        } else {
            sum.integrity = -1.0;
        }

        sum.overall_grade = if sum.applicable_axes == 0 {
            GRADE_NA
        } else if sum.violated_axes > 0 || any_fail_span {
            GRADE_FAIL
        } else if sum.warned_axes > 0 || (cov as f32) < self.cfg.min_coverage {
            GRADE_WARN
        } else {
            GRADE_PASS
        };

        // Anti-gaming RLVR reward contract (Astra 6.2 + Sol 6.2 + Opus 5.5):
        // - Unprofiled wild clip (`required_axes_mask == 0`): INT32_MIN (null) when unscorable or below coverage floor;
        //   otherwise `round((2 * integrity - 1) * 1000)` in [-1000, +1000].
        // - Profiled clip (`required_axes_mask != 0`): fixed denominator over ALL required axes.
        //   PASS = +1000, WARN = 0, VIOLATED or MISSING/INSUFFICIENT = -1000.
        //   Blank/frozen/static/noisy/truncated clips CANNOT improve reward by suppressing evidence!
        let req_mask = self.cfg.required_axes_mask & 0xff;
        if req_mask == 0 {
            if sum.applicable_axes == 0 || (cov as f32) < self.cfg.min_coverage {
                sum.reward_milli = i32::MIN;
            } else {
                sum.reward_milli = ((sum.integrity * 1000.0).round() as i32).clamp(0, 1000);
            }
        } else {
            let mut req_total = 0i32;
            let mut req_sum = 0i32;
            let mut any_missing_req = false;
            for ax_i in 0..CHRONOS_AUDIT_AXIS_COUNT {
                if (req_mask & (1u32 << ax_i)) != 0 {
                    req_total += 1;
                    let ax = &sum.axes[ax_i];
                    if ax.applicability != APPL_ELIGIBLE {
                        any_missing_req = true;
                        req_sum -= 1000;
                    } else {
                        match ax.verdict {
                            VERDICT_PASS => req_sum += 1000,
                            VERDICT_WARN => {}
                            _ => req_sum -= 1000,
                        }
                    }
                }
            }
            if any_missing_req || (cov as f32) < self.cfg.min_coverage || req_total == 0 {
                sum.reward_milli = -1000;
            } else {
                sum.reward_milli = (req_sum / req_total).clamp(-1000, 1000);
            }
        }

        self.cached_summary = sum;
        sum
    }

    pub fn summary(&self) -> ChronosAuditSummary {
        self.cached_summary
    }

    pub fn spans(&self) -> &[ChronosAuditSpan] {
        &self.spans
    }
}
