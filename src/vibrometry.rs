//! Calibrated Rolling-Shutter Vibrometry & Multi-Segment Welch Audio Coherence (`Organ 4` / `Gate I2`).
//!
//! Pure `#![no_std]`, zero host imports, zero heap allocation inside static `VibrometryState`.

pub const VIBRO_BANDS: usize = 16;
pub const VIBRO_MAX_ROWS: usize = 360;
pub const VIBRO_HIST_FRAMES: usize = 32;
pub const VIBRO_SPEC_BINS: usize = 64;
pub const VIBRO_WELCH_SLOTS: usize = 16;
pub const AUDIO_STAGING_LEN: usize = 4096;
pub const AUDIO_RING_LEN: usize = 32768;
pub const VIBRO_TELEM_LEN: usize = 32;
pub const VIBRO_CELL_W: usize = 40;
pub const VIBRO_CELL_H: usize = 22;
pub const VIBRO_CELLS: usize = VIBRO_CELL_W * VIBRO_CELL_H; // 880
pub const VIBRO_PLANES_LEN: usize = 3 * VIBRO_CELLS + VIBRO_SPEC_BINS; // 2,704

pub const VIBRO_PLANE_AMP: usize = 0;
pub const VIBRO_PLANE_COHERENCE: usize = VIBRO_CELLS;
pub const VIBRO_PLANE_PHASE: usize = 2 * VIBRO_CELLS;
pub const VIBRO_PLANE_SPECTRUM: usize = 3 * VIBRO_CELLS;

pub const RS_STATE_OFF: u32 = 0;
pub const RS_STATE_UNCALIBRATED: u32 = 1;
pub const RS_STATE_WARMING: u32 = 2;
pub const RS_STATE_READY: u32 = 3;
pub const RS_STATE_DISTURBED: u32 = 4;

pub const RS_MODE_OFF: u32 = 0;
pub const RS_MODE_RS_ONLY: u32 = 1;
pub const RS_MODE_RS_WELCH: u32 = 2;

pub const RS_REASON_BAD_CALIB: u32 = 1 << 0;
pub const RS_REASON_NO_SCAN_DIR: u32 = 1 << 1;
pub const RS_REASON_NOT_NATIVE_1TO1: u32 = 1 << 2;
pub const RS_REASON_FRAME_GAP: u32 = 1 << 3;
pub const RS_REASON_CAMERA_MOTION: u32 = 1 << 4;
pub const RS_REASON_CLOCK_SKEW: u32 = 1 << 5;
pub const RS_REASON_EXPOSURE_NULL: u32 = 1 << 6;
pub const RS_REASON_LOW_GRAD: u32 = 1 << 7;
pub const RS_REASON_AUDIO_UNDERRUN: u32 = 1 << 8;

pub const AUDIO_FFT_N: usize = 1024;

#[repr(C, align(16))]
pub struct VibrometryState {
    // C-ABI flat export buffers
    pub telem: [u32; VIBRO_TELEM_LEN],
    pub planes: [f32; VIBRO_PLANES_LEN],
    pub audio_staging: [f32; AUDIO_STAGING_LEN],

    // Audio sample ring & timestamps
    pub audio_ring: [f32; AUDIO_RING_LEN],
    pub audio_write_cursor: u64,
    pub audio_total_samples: u64,
    pub audio_last_t_end_us: u64,
    pub audio_last_skew_us: f32,
    pub audio_ring_rate_hz: f32,

    // Row-resolved 16-band x 360-row reference & displacement history
    pub ref_band_luma: [f32; VIBRO_BANDS * VIBRO_MAX_ROWS],
    pub ref_band_gx2: [f32; VIBRO_BANDS * VIBRO_MAX_ROWS],
    pub disp_hist: [f32; VIBRO_HIST_FRAMES * VIBRO_BANDS * VIBRO_MAX_ROWS],
    pub t_src_hist_us: [u64; VIBRO_HIST_FRAMES],
    pub frame_rows_hist: [u16; VIBRO_HIST_FRAMES],
    pub hist_cursor: u32,
    pub hist_count: u32,

    // Multi-segment Welch FFT & cross-spectral accumulators (zero stack bloat)
    pub fft_re: [f64; AUDIO_FFT_N],
    pub fft_im: [f64; AUDIO_FFT_N],
    pub welch_pxx: [f64; VIBRO_SPEC_BINS * VIBRO_WELCH_SLOTS],
    pub welch_pyy: [f64; VIBRO_BANDS * VIBRO_SPEC_BINS * VIBRO_WELCH_SLOTS],
    pub welch_pxy_re: [f64; VIBRO_BANDS * VIBRO_SPEC_BINS * VIBRO_WELCH_SLOTS],
    pub welch_pxy_im: [f64; VIBRO_BANDS * VIBRO_SPEC_BINS * VIBRO_WELCH_SLOTS],
    pub welch_cursor: u32,
    pub welch_count: u32,
    pub last_audio_seg_end_sample: u64,

    // Calibrated acquisition configuration
    pub mode: u32,
    pub dt_row_us: f32,
    pub exposure_us: f32,
    pub calib_audio_rate_hz: f32,
    pub max_skew_us: f32,
    pub calib_valid: u8,
    pub ref_warm_frames: u8,
    pub active_frames_in_run: u32,
    pub last_frame_w: u16,
    pub last_frame_h: u16,
    pub last_scan_dir: u32,
    pub last_orientation: u32,
    pub last_crop_y0: u32,
}

const _: () = assert!(core::mem::size_of::<VibrometryState>() <= 1_450_000);

impl VibrometryState {
    pub fn reset_all(&mut self) {
        let ptr = self as *mut Self as *mut u8;
        unsafe {
            core::ptr::write_bytes(ptr, 0, core::mem::size_of::<Self>());
        }
    }

    pub fn reset_continuity(&mut self, reason_bits: u32) {
        self.ref_band_luma.fill(0.0);
        self.ref_band_gx2.fill(0.0);
        self.disp_hist.fill(0.0);
        self.t_src_hist_us.fill(0);
        self.frame_rows_hist.fill(0);
        self.hist_cursor = 0;
        self.hist_count = 0;
        self.welch_pxx.fill(0.0);
        self.welch_pyy.fill(0.0);
        self.welch_pxy_re.fill(0.0);
        self.welch_pxy_im.fill(0.0);
        self.welch_cursor = 0;
        self.welch_count = 0;
        self.ref_warm_frames = 0;
        self.active_frames_in_run = 0;
        self.planes.fill(0.0);

        let keep_mode = self.mode;
        let keep_calib_valid = self.calib_valid;
        self.telem.fill(0);
        self.telem[1] = keep_mode;
        self.telem[14] = self.dt_row_us.to_bits();
        self.telem[15] = self.exposure_us.to_bits();
        self.telem[16] = self.calib_audio_rate_hz.to_bits();
        self.telem[17] = self.max_skew_us.to_bits();
        if keep_mode == RS_MODE_OFF {
            self.telem[0] = RS_STATE_OFF;
        } else if keep_calib_valid == 0 {
            self.telem[0] = RS_STATE_UNCALIBRATED;
            self.telem[2] = reason_bits | RS_REASON_BAD_CALIB;
        } else if reason_bits != 0 {
            self.telem[0] = RS_STATE_DISTURBED;
            self.telem[2] = reason_bits;
        } else {
            self.telem[0] = RS_STATE_WARMING;
        }
    }

    pub fn set_calib(
        &mut self,
        mode: u32,
        dt_row_us: f32,
        exposure_us: f32,
        audio_rate_hz: f32,
        max_skew_us: f32,
    ) -> u32 {
        let sanitized_mode = mode.min(RS_MODE_RS_WELCH);
        if sanitized_mode == RS_MODE_OFF {
            self.mode = RS_MODE_OFF;
            self.calib_valid = 0;
            self.reset_continuity(0);
            return 1;
        }

        let calib_ok = dt_row_us.is_finite()
            && dt_row_us > 0.0
            && dt_row_us <= 1000.0
            && exposure_us.is_finite()
            && exposure_us > 0.0
            && exposure_us <= 200_000.0
            && max_skew_us.is_finite()
            && max_skew_us >= 0.0
            && (sanitized_mode == RS_MODE_RS_ONLY
                || (audio_rate_hz.is_finite()
                    && audio_rate_hz >= 4000.0
                    && audio_rate_hz <= 192_000.0));

        self.mode = sanitized_mode;
        if !calib_ok {
            self.calib_valid = 0;
            self.dt_row_us = 0.0;
            self.exposure_us = 0.0;
            self.calib_audio_rate_hz = 0.0;
            self.max_skew_us = 0.0;
            self.reset_continuity(RS_REASON_BAD_CALIB);
            return 0;
        }

        self.dt_row_us = dt_row_us;
        self.exposure_us = exposure_us;
        self.calib_audio_rate_hz = audio_rate_hz;
        self.max_skew_us = max_skew_us;
        self.calib_valid = 1;
        self.reset_continuity(0);
        1
    }

    #[inline(always)]
    pub fn get_mode(&self) -> u32 {
        self.mode
    }

    #[inline(always)]
    pub fn get_state(&self) -> u32 {
        self.telem[0]
    }

    #[inline(always)]
    pub fn audio_staging_mut_ptr(&mut self) -> *mut f32 {
        self.audio_staging.as_mut_ptr()
    }

    #[inline(never)]
    pub fn push_audio_block(
        &mut self,
        num_samples: u32,
        t_end_us: u64,
        sample_rate_hz: f32,
        est_skew_us: f32,
    ) -> u32 {
        let n = num_samples as usize;
        if n == 0
            || n > AUDIO_STAGING_LEN
            || !sample_rate_hz.is_finite()
            || sample_rate_hz < 4000.0
            || sample_rate_hz > 192_000.0
            || !est_skew_us.is_finite()
        {
            return 0;
        }
        let mut i = 0usize;
        while i < n {
            let s = self.audio_staging[i];
            let idx = (self.audio_write_cursor as usize) & (AUDIO_RING_LEN - 1);
            self.audio_ring[idx] = if s.is_finite() { s } else { 0.0 };
            self.audio_write_cursor = self.audio_write_cursor.wrapping_add(1);
            i += 1;
        }
        self.audio_total_samples = self.audio_total_samples.saturating_add(n as u64);
        self.audio_last_t_end_us = t_end_us;
        self.audio_ring_rate_hz = sample_rate_hz;
        self.audio_last_skew_us = est_skew_us;
        self.telem[10] = est_skew_us.to_bits();
        self.telem[18] = self.audio_total_samples as u32;
        1
    }

    #[inline(never)]
    pub fn ingest_and_solve_frame(
        &mut self,
        _luma: &[f32],
        _row_luma: &[f32; VIBRO_MAX_ROWS],
        frame_w: usize,
        frame_h: usize,
        _t_src_us: u64,
        scan_dir: u32,
        orientation: u32,
        crop_y0: u32,
        _native_1to1: u32,
        _gap_flag: u32,
        _cam_motion_state: u32,
    ) {
        self.last_frame_w = frame_w as u16;
        self.last_frame_h = frame_h as u16;
        self.last_scan_dir = scan_dir;
        self.last_orientation = orientation;
        self.last_crop_y0 = crop_y0;
        if self.mode == RS_MODE_OFF {
            self.telem[0] = RS_STATE_OFF;
            return;
        }
        if self.calib_valid == 0 {
            self.telem[0] = RS_STATE_UNCALIBRATED;
            self.telem[2] |= RS_REASON_BAD_CALIB;
            return;
        }
        self.active_frames_in_run = self.active_frames_in_run.saturating_add(1);
        self.telem[19] = self.active_frames_in_run;
        self.telem[0] = RS_STATE_WARMING;
    }

    #[inline(always)]
    pub fn telemetry_ptr(&self) -> *const u32 {
        self.telem.as_ptr()
    }

    #[inline(always)]
    pub fn planes_ptr(&self) -> *const f32 {
        self.planes.as_ptr()
    }
}
