//! Calibrated Rolling-Shutter Vibrometry & Multi-Segment Welch Audio Coherence (`Organ 4` / `Gate I2`).
//!
//! Pure `#![no_std]`, zero host imports, zero heap allocation inside static `VibrometryState`.

pub const VIBRO_BANDS: usize = 16;
pub const VIBRO_MAX_ROWS: usize = 360;
pub const VIBRO_MAX_PIXELS: usize = 640 * 360; // 230,400
pub const VIBRO_HIST_FRAMES: usize = 32;
pub const VIBRO_SPEC_BINS: usize = 64;
pub const VIBRO_WELCH_SLOTS: usize = 12;
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
pub const RS_REASON_AXIS_TRANSPOSED: u32 = 1 << 9;
pub const RS_REASON_NO_PTS: u32 = 1 << 10;

pub const AUDIO_FFT_N: usize = 1024;
const AUDIO_HOP: usize = 512;
const MAX_SEG_OBS: usize = 1536;
const REF_WARM_TARGET: u8 = 16;

const PI_32: f32 = 3.14159265358979323846;
const PI_64: f64 = 3.14159265358979323846;
const TWO_PI_64: f64 = 6.28318530717958647692;
const HALF_PI_64: f64 = 1.57079632679489661923;
const INV_TWO_PI_64: f64 = 0.15915494309189533577;
const LN_ALPHA_BIN_63: f64 = -8.743363716677859;

#[inline(always)]
fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

#[inline(always)]
fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
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

#[inline(always)]
fn sin_cos_f64(rad: f64) -> (f64, f64) {
    if !rad.is_finite() || abs_f64(rad) > 1.0e12 {
        return (0.0, 1.0);
    }
    let k = (rad * INV_TWO_PI_64 + if rad >= 0.0 { 0.5 } else { -0.5 }) as i64;
    let mut a = rad - (k as f64) * TWO_PI_64;
    while a > PI_64 {
        a -= TWO_PI_64;
    }
    while a < -PI_64 {
        a += TWO_PI_64;
    }
    let cos_sign = if a > HALF_PI_64 {
        a = PI_64 - a;
        -1.0
    } else if a < -HALF_PI_64 {
        a = -PI_64 - a;
        -1.0
    } else {
        1.0
    };
    let a2 = a * a;
    let s = a
        * (1.0
            + a2 * (-1.0 / 6.0
                + a2 * (1.0 / 120.0
                    + a2 * (-1.0 / 5040.0
                        + a2 * (1.0 / 362880.0
                            + a2 * (-1.0 / 39916800.0
                                + a2 * (1.0 / 6227020800.0 - a2 * (1.0 / 1307674368000.0))))))));
    let c = cos_sign
        * (1.0
            + a2 * (-0.5
                + a2 * (1.0 / 24.0
                    + a2 * (-1.0 / 720.0
                        + a2 * (1.0 / 40320.0
                            + a2 * (-1.0 / 3628800.0
                                + a2 * (1.0 / 479001600.0
                                    + a2 * (-1.0 / 87178291200.0
                                        + a2 * (1.0 / 20922789888000.0)))))))));
    (s, c)
}

#[inline(always)]
fn exp_neg_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    if x >= 0.0 {
        return 1.0;
    }
    if x < -700.0 {
        return 0.0;
    }
    const LN2_64: f64 = 0.6931471805599453;
    const INV_LN2_64: f64 = 1.4426950408889634;
    let k = (x * INV_LN2_64 - 0.5) as i64;
    let r = x - (k as f64) * LN2_64;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut n = 1usize;
    while n <= 14 {
        term *= r / (n as f64);
        sum += term;
        n += 1;
    }
    let exp_biased = 1023i64 + k;
    if exp_biased <= 0 {
        return 0.0;
    }
    sum * f64::from_bits((exp_biased as u64) << 52)
}

#[inline(always)]
fn sinc_exposure_h(freq_hz: f64, exposure_us: f64) -> f64 {
    let x = PI_64 * freq_hz * exposure_us * 1.0e-6;
    if abs_f64(x) < 1.0e-6 {
        1.0 - (x * x) * (1.0 / 6.0)
    } else {
        let (s, _) = sin_cos_f64(x);
        s / x
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

#[inline(always)]
fn engine_to_sensor_xy(
    ex: u32,
    ey: u32,
    ow: u32,
    cw: u32,
    ch: u32,
    crop_x0: u32,
    crop_y0: u32,
    orientation: u32,
) -> (u32, u32) {
    let o = orientation & 7;
    let rx = if (o & 4) != 0 { ow - 1 - ex } else { ex };
    let ry = ey;
    let (xc, yc) = match o & 3 {
        0 => (rx, ry),
        1 => (ry, ch - 1 - rx),
        2 => (cw - 1 - rx, ch - 1 - ry),
        _ => (cw - 1 - ry, rx),
    };
    (crop_x0 + xc, crop_y0 + yc)
}

#[inline(always)]
fn rotate_disp_to_sensor(u_ex: f32, u_ey: f32, orientation: u32) -> (f32, f32) {
    match orientation & 7 {
        0 => (u_ex, u_ey),
        1 => (u_ey, -u_ex),
        2 => (-u_ex, -u_ey),
        3 => (-u_ey, u_ex),
        4 => (-u_ex, u_ey),
        5 => (u_ey, u_ex),
        6 => (u_ex, -u_ey),
        _ => (-u_ey, -u_ex),
    }
}

struct Fft1024Tables {
    bitrev: [u16; AUDIO_FFT_N],
    tw_re: [f64; AUDIO_FFT_N / 2],
    tw_im: [f64; AUDIO_FFT_N / 2],
    hann: [f64; AUDIO_FFT_N],
}

impl Fft1024Tables {
    const fn new() -> Self {
        let mut bitrev = [0u16; AUDIO_FFT_N];
        let mut i = 0usize;
        while i < AUDIO_FFT_N {
            let mut r = 0usize;
            let mut b = 0usize;
            while b < 10 {
                r = (r << 1) | ((i >> b) & 1);
                b += 1;
            }
            bitrev[i] = r as u16;
            i += 1;
        }

        let mut tw_re = [0.0f64; AUDIO_FFT_N / 2];
        let mut tw_im = [0.0f64; AUDIO_FFT_N / 2];
        let mut k = 0usize;
        while k < AUDIO_FFT_N / 2 {
            let ang = -TWO_PI_64 * (k as f64) / (AUDIO_FFT_N as f64);
            tw_re[k] = const_cos_f64(ang);
            tw_im[k] = const_sin_f64(ang);
            k += 1;
        }

        let mut hann = [0.0f64; AUDIO_FFT_N];
        let mut n = 0usize;
        while n < AUDIO_FFT_N {
            hann[n] = 0.5 * (1.0 - const_cos_f64(TWO_PI_64 * (n as f64) / (AUDIO_FFT_N as f64)));
            n += 1;
        }

        Self {
            bitrev,
            tw_re,
            tw_im,
            hann,
        }
    }
}

static FFT1024_TABLES: Fft1024Tables = Fft1024Tables::new();

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VibroFrameMeta {
    pub flags: u32,
    pub continuity_epoch: u32,
    pub t_src_us: u64,
    pub src_w: u32,
    pub src_h: u32,
    pub crop_x0: u32,
    pub crop_y0: u32,
    pub crop_w: u32,
    pub crop_h: u32,
    pub orientation: u32,
    pub scan_dir: u32,
    pub native_1to1: u32,
    pub gap_flag: u32,
    pub cam_motion_state: u32,
}

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

    // Full-frame stationary reference luma (u16 q16) & row-resolved signed displacement history
    pub ref_luma_q16: [u16; VIBRO_MAX_PIXELS],
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

    // Internal Arena-backed scratch & continuity alignment state
    audio_interval_start_sample: u64,
    run_t0_us: u64,
    q_start_hist: [u32; VIBRO_HIST_FRAMES],
    q_step_hist: [i32; VIBRO_HIST_FRAMES],
    seg_x_re: [f64; VIBRO_SPEC_BINS],
    seg_x_im: [f64; VIBRO_SPEC_BINS],
    seg_y_re: [f64; VIBRO_BANDS * VIBRO_SPEC_BINS],
    seg_y_im: [f64; VIBRO_BANDS * VIBRO_SPEC_BINS],
    mean_pxx: [f64; VIBRO_SPEC_BINS],
    band_mean_pyy: [f64; VIBRO_SPEC_BINS],
    band_mean_pxy_re: [f64; VIBRO_SPEC_BINS],
    band_mean_pxy_im: [f64; VIBRO_SPEC_BINS],
    tmp_cos: [f64; VIBRO_SPEC_BINS],
    tmp_sin: [f64; VIBRO_SPEC_BINS],
    band_mean_y: [f64; VIBRO_BANDS],
    rs_tmp_re: [f64; VIBRO_BANDS],
    rs_tmp_im: [f64; VIBRO_BANDS],
    rs_pow_sum: [f64; VIBRO_BANDS],
    rs_refine_scores: [f64; 9],
    band_row_ux: [f32; VIBRO_MAX_ROWS],
    band_row_uy: [f32; VIBRO_MAX_ROWS],
    band_row_ok: [u8; VIBRO_MAX_ROWS],
    cell_grad_ok: [u8; VIBRO_BANDS * VIBRO_CELL_H],
    band_grad_ok: [u8; VIBRO_BANDS],
    band_axis_locked: [u8; VIBRO_BANDS],
    band_ax_x: [f32; VIBRO_BANDS],
    band_ax_y: [f32; VIBRO_BANDS],
    band_amp: [f32; VIBRO_BANDS],
    band_phase: [f32; VIBRO_BANDS],
    band_coh: [f32; VIBRO_BANDS],
    obs_t_rel_us: [f64; MAX_SEG_OBS],
    obs_w: [f64; MAX_SEG_OBS],
    obs_slot: [u16; MAX_SEG_OBS],
    obs_ey: [u16; MAX_SEG_OBS],
    last_src_w: u32,
    last_src_h: u32,
    last_crop_x0: u32,
    single_seg_k1_coh: f32,
    audio_has_prev_t: u8,
    skew_disturbed: u8,
}

const _: () = assert!(core::mem::size_of::<VibrometryState>() <= 1_750_000);

impl VibrometryState {
    pub fn reset_all(&mut self) {
        let ptr = self as *mut Self as *mut u8;
        unsafe {
            core::ptr::write_bytes(ptr, 0, core::mem::size_of::<Self>());
        }
    }

    #[inline(never)]
    fn reset_audio_welch_interval(&mut self) {
        self.welch_pxx.fill(0.0);
        self.welch_pyy.fill(0.0);
        self.welch_pxy_re.fill(0.0);
        self.welch_pxy_im.fill(0.0);
        self.welch_cursor = 0;
        self.welch_count = 0;
        self.single_seg_k1_coh = 0.0;
        self.audio_interval_start_sample = self.audio_total_samples;
        self.last_audio_seg_end_sample = self.audio_total_samples;
        self.audio_last_t_end_us = 0;
        self.audio_has_prev_t = 0;
    }

    #[inline(never)]
    pub fn reset_continuity(&mut self, reason_bits: u32) {
        self.ref_luma_q16.fill(0);
        self.disp_hist.fill(0.0);
        self.t_src_hist_us.fill(0);
        self.frame_rows_hist.fill(0);
        self.q_start_hist.fill(0);
        self.q_step_hist.fill(0);
        self.hist_cursor = 0;
        self.hist_count = 0;
        self.run_t0_us = 0;
        self.ref_warm_frames = 0;
        self.active_frames_in_run = 0;
        self.cell_grad_ok.fill(0);
        self.band_grad_ok.fill(0);
        self.band_axis_locked.fill(0);
        self.band_ax_x.fill(0.0);
        self.band_ax_y.fill(0.0);
        self.band_amp.fill(0.0);
        self.band_phase.fill(0.0);
        self.band_coh.fill(0.0);
        self.planes.fill(0.0);
        self.reset_audio_welch_interval();
        self.skew_disturbed = 0;

        let keep_mode = self.mode;
        let keep_calib_valid = self.calib_valid;
        self.telem.fill(0);
        self.telem[1] = keep_mode;
        self.telem[10] = self.audio_last_skew_us.to_bits();
        self.telem[13] = if keep_mode == RS_MODE_RS_WELCH && keep_calib_valid != 0 {
            (self.calib_audio_rate_hz / (AUDIO_FFT_N as f32)).to_bits()
        } else {
            0
        };
        self.telem[14] = self.dt_row_us.to_bits();
        self.telem[15] = self.exposure_us.to_bits();
        self.telem[16] = self.calib_audio_rate_hz.to_bits();
        self.telem[17] = self.max_skew_us.to_bits();
        self.telem[18] = self.audio_total_samples as u32;
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
        if n == 0 || n > AUDIO_STAGING_LEN {
            if self.mode == RS_MODE_RS_WELCH && self.calib_valid != 0 {
                self.telem[2] |= RS_REASON_AUDIO_UNDERRUN;
            }
            return 0;
        }
        if !sample_rate_hz.is_finite() || sample_rate_hz < 4000.0 || sample_rate_hz > 192_000.0 {
            if self.mode != RS_MODE_OFF {
                self.telem[2] |= RS_REASON_BAD_CALIB;
            }
            return 0;
        }
        if self.calib_valid != 0
            && self.mode == RS_MODE_RS_WELCH
            && abs_f32(sample_rate_hz - self.calib_audio_rate_hz) > 1.0
        {
            self.telem[0] = RS_STATE_DISTURBED;
            self.telem[2] |= RS_REASON_BAD_CALIB;
            return 0;
        }
        if !est_skew_us.is_finite() {
            self.skew_disturbed = 1;
            self.reset_audio_welch_interval();
            self.planes.fill(0.0);
            self.telem[3] = 0;
            self.telem[4] = 0;
            self.telem[21] = 0;
            if self.mode != RS_MODE_OFF {
                self.telem[0] = RS_STATE_DISTURBED;
                self.telem[2] |= RS_REASON_CLOCK_SKEW;
            }
            return 0;
        }

        self.audio_last_skew_us = est_skew_us;
        self.telem[10] = est_skew_us.to_bits();

        if self.calib_valid != 0 && abs_f32(est_skew_us) > self.max_skew_us {
            self.skew_disturbed = 1;
            self.reset_audio_welch_interval();
            self.planes.fill(0.0);
            self.telem[3] = 0;
            self.telem[4] = 0;
            self.telem[21] = 0;
            if self.mode != RS_MODE_OFF {
                self.telem[0] = RS_STATE_DISTURBED;
                self.telem[2] |= RS_REASON_CLOCK_SKEW;
            }
            return 0;
        }

        if self.audio_has_prev_t != 0 {
            let dt_audio_us = (n as f64) * 1_000_000.0 / (sample_rate_hz as f64);
            if t_end_us <= self.audio_last_t_end_us {
                self.reset_audio_welch_interval();
                self.planes.fill(0.0);
                self.telem[3] = 0;
                self.telem[4] = 0;
                self.telem[21] = 0;
                if self.mode == RS_MODE_RS_WELCH && self.calib_valid != 0 {
                    self.telem[2] |= RS_REASON_AUDIO_UNDERRUN;
                    if self.telem[0] == RS_STATE_READY {
                        self.telem[0] = RS_STATE_WARMING;
                    }
                }
                return 0;
            }
            let step_us = (t_end_us - self.audio_last_t_end_us) as f64;
            if abs_f64(step_us - dt_audio_us) > 2.5 * dt_audio_us
                || abs_f32(sample_rate_hz - self.audio_ring_rate_hz) > 1.0
            {
                self.reset_audio_welch_interval();
                self.planes.fill(0.0);
                self.telem[3] = 0;
                self.telem[4] = 0;
                self.telem[21] = 0;
                if self.mode == RS_MODE_RS_WELCH && self.calib_valid != 0 {
                    self.telem[2] |= RS_REASON_AUDIO_UNDERRUN;
                    if self.telem[0] == RS_STATE_READY {
                        self.telem[0] = RS_STATE_WARMING;
                    }
                }
            }
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
        let ring_floor = self.audio_total_samples.saturating_sub(AUDIO_RING_LEN as u64);
        if self.audio_interval_start_sample < ring_floor {
            self.audio_interval_start_sample = ring_floor;
        }
        if self.last_audio_seg_end_sample < self.audio_interval_start_sample {
            self.last_audio_seg_end_sample = self.audio_interval_start_sample;
        }
        self.skew_disturbed = 0;
        self.telem[2] &= !RS_REASON_CLOCK_SKEW;
        self.audio_last_t_end_us = t_end_us;
        self.audio_has_prev_t = 1;
        self.audio_ring_rate_hz = sample_rate_hz;
        self.telem[18] = self.audio_total_samples as u32;
        self.telem[13] = (sample_rate_hz / (AUDIO_FFT_N as f32)).to_bits();
        1
    }

    #[inline(never)]
    pub fn ingest_and_solve_frame(
        &mut self,
        luma: &[f32],
        _row_luma: &[f32; VIBRO_MAX_ROWS],
        frame_w: usize,
        frame_h: usize,
        meta: &VibroFrameMeta,
    ) {
        if self.mode == RS_MODE_OFF {
            self.telem[0] = RS_STATE_OFF;
            return;
        }
        if self.calib_valid == 0 {
            self.planes.fill(0.0);
            self.telem[0] = RS_STATE_UNCALIBRATED;
            self.telem[2] |= RS_REASON_BAD_CALIB;
            self.telem[4] = 0;
            self.telem[21] = 0;
            return;
        }
        if frame_w < 16
            || frame_w > 640
            || frame_h < VIBRO_CELL_H
            || frame_h > VIBRO_MAX_ROWS
            || luma.len() < frame_w * frame_h
        {
            self.reset_continuity(RS_REASON_NOT_NATIVE_1TO1);
            self.telem[0] = RS_STATE_UNCALIBRATED;
            return;
        }

        let geom_changed = (self.ref_warm_frames > 0 || self.active_frames_in_run > 0)
            && (self.last_frame_w as usize != frame_w
                || self.last_frame_h as usize != frame_h
                || self.last_scan_dir != meta.scan_dir
                || self.last_orientation != meta.orientation
                || self.last_crop_x0 != meta.crop_x0
                || self.last_crop_y0 != meta.crop_y0
                || self.last_src_w != meta.src_w
                || self.last_src_h != meta.src_h);
        if geom_changed {
            self.reset_continuity(RS_REASON_FRAME_GAP);
        }

        self.last_frame_w = frame_w as u16;
        self.last_frame_h = frame_h as u16;
        self.last_scan_dir = meta.scan_dir;
        self.last_orientation = meta.orientation;
        self.last_crop_x0 = meta.crop_x0;
        self.last_crop_y0 = meta.crop_y0;
        self.last_src_w = meta.src_w;
        self.last_src_h = meta.src_h;

        let hard_gap = (meta.gap_flag & !0x40u32) != 0;
        if hard_gap {
            self.reset_continuity(RS_REASON_FRAME_GAP);
        }

        let mut uncalib_reasons = 0u32;
        if (meta.flags & 2) == 0 {
            uncalib_reasons |= RS_REASON_NO_PTS;
        }
        if meta.native_1to1 != 1 {
            uncalib_reasons |= RS_REASON_NOT_NATIVE_1TO1;
        }
        if meta.scan_dir > 3 {
            uncalib_reasons |= RS_REASON_NO_SCAN_DIR;
        }
        let crop_valid = meta.orientation <= 7
            && meta.src_w > 0
            && meta.src_h > 0
            && meta.crop_w > 0
            && meta.crop_h > 0
            && meta.crop_x0.checked_add(meta.crop_w).map_or(false, |v| v <= meta.src_w)
            && meta.crop_y0.checked_add(meta.crop_h).map_or(false, |v| v <= meta.src_h)
            && oriented_extent(meta.crop_w, meta.crop_h, meta.orientation)
                == (frame_w as u32, frame_h as u32);
        if !crop_valid {
            uncalib_reasons |= RS_REASON_NOT_NATIVE_1TO1;
        }

        let ow = frame_w as u32;
        let oh = frame_h as u32;
        let o = meta.orientation & 7;
        let mut q_start = 0u32;
        let mut q_step = 1i32;

        if uncalib_reasons == 0 {
            let (sx_tl, sy_tl) = engine_to_sensor_xy(
                0,
                0,
                ow,
                meta.crop_w,
                meta.crop_h,
                meta.crop_x0,
                meta.crop_y0,
                o,
            );
            let (sx_tr, sy_tr) = engine_to_sensor_xy(
                ow - 1,
                0,
                ow,
                meta.crop_w,
                meta.crop_h,
                meta.crop_x0,
                meta.crop_y0,
                o,
            );
            let (sx_bl, sy_bl) = engine_to_sensor_xy(
                0,
                oh - 1,
                ow,
                meta.crop_w,
                meta.crop_h,
                meta.crop_x0,
                meta.crop_y0,
                o,
            );
            let (sx_br, sy_br) = engine_to_sensor_xy(
                ow - 1,
                oh - 1,
                ow,
                meta.crop_w,
                meta.crop_h,
                meta.crop_x0,
                meta.crop_y0,
                o,
            );

            let aligned = if meta.scan_dir <= 1 {
                (o & 1) == 0 && sy_tl == sy_tr && sy_bl == sy_br
            } else {
                (o & 1) == 1 && sx_tl == sx_tr && sx_bl == sx_br
            };
            if !aligned {
                uncalib_reasons |= RS_REASON_AXIS_TRANSPOSED;
            } else {
                let (c0, c1, extent) = if meta.scan_dir <= 1 {
                    let (_, sy_1) = engine_to_sensor_xy(
                        0,
                        1,
                        ow,
                        meta.crop_w,
                        meta.crop_h,
                        meta.crop_x0,
                        meta.crop_y0,
                        o,
                    );
                    (sy_tl, sy_1, meta.src_h)
                } else {
                    let (sx_1, _) = engine_to_sensor_xy(
                        0,
                        1,
                        ow,
                        meta.crop_w,
                        meta.crop_h,
                        meta.crop_x0,
                        meta.crop_y0,
                        o,
                    );
                    (sx_tl, sx_1, meta.src_w)
                };
                let (q0, q1) = if meta.scan_dir == 0 || meta.scan_dir == 2 {
                    (c0, c1)
                } else {
                    (extent - 1 - c0, extent - 1 - c1)
                };
                q_start = q0;
                q_step = if oh > 1 && q1 < q0 { -1 } else { 1 };
            }
        }

        if uncalib_reasons != 0 {
            let gap_bit = if hard_gap { RS_REASON_FRAME_GAP } else { 0 };
            self.reset_continuity(0);
            self.telem[0] = RS_STATE_UNCALIBRATED;
            self.telem[2] = uncalib_reasons | gap_bit;
            return;
        }

        if hard_gap || meta.cam_motion_state != 0 {
            let mut rsn = if hard_gap { RS_REASON_FRAME_GAP } else { 0 };
            if meta.cam_motion_state != 0 {
                rsn |= RS_REASON_CAMERA_MOTION;
            }
            self.reset_continuity(rsn);
            return;
        }

        if self.hist_count > 0 {
            let prev_slot = ((self.hist_cursor as usize) + VIBRO_HIST_FRAMES - 1) % VIBRO_HIST_FRAMES;
            if meta.t_src_us <= self.t_src_hist_us[prev_slot] {
                self.reset_continuity(RS_REASON_FRAME_GAP);
                return;
            }
        }

        if self.mode == RS_MODE_RS_WELCH && self.skew_disturbed != 0 {
            self.planes.fill(0.0);
            self.telem[0] = RS_STATE_DISTURBED;
            self.telem[2] |= RS_REASON_CLOCK_SKEW;
            self.telem[4] = 0;
            self.telem[21] = 0;
            return;
        }

        self.active_frames_in_run = self.active_frames_in_run.saturating_add(1);
        self.telem[19] = self.active_frames_in_run;
        if self.run_t0_us == 0 && self.ref_warm_frames == 0 {
            self.run_t0_us = meta.t_src_us;
        }

        if self.ref_warm_frames < REF_WARM_TARGET {
            self.warmup_accumulate_and_lock(luma, frame_w, frame_h);
            return;
        }

        let mut ok_bands = 0u32;
        let mut b = 0usize;
        while b < VIBRO_BANDS {
            if self.band_grad_ok[b] != 0 {
                ok_bands += 1;
            }
            b += 1;
        }
        if ok_bands == 0 {
            self.planes.fill(0.0);
            self.telem[0] = RS_STATE_DISTURBED;
            self.telem[2] = RS_REASON_LOW_GRAD;
            self.telem[4] = 0;
            self.telem[21] = 0;
            return;
        }

        let slot = (self.hist_cursor as usize) % VIBRO_HIST_FRAMES;
        self.extract_signed_band_displacements(luma, frame_w, frame_h, o, slot);
        self.t_src_hist_us[slot] = meta.t_src_us;
        self.frame_rows_hist[slot] = frame_h as u16;
        self.q_start_hist[slot] = q_start;
        self.q_step_hist[slot] = q_step;
        self.hist_cursor = ((slot + 1) % VIBRO_HIST_FRAMES) as u32;
        if (self.hist_count as usize) < VIBRO_HIST_FRAMES {
            self.hist_count += 1;
        }

        if self.mode == RS_MODE_RS_ONLY {
            self.solve_rs_only(frame_h);
        } else {
            self.advance_welch_segments();
            self.finalize_welch_solution();
        }
    }

    #[inline(never)]
    fn warmup_accumulate_and_lock(&mut self, luma: &[f32], frame_w: usize, frame_h: usize) {
        let need_px = frame_w * frame_h;
        let warm_idx = self.ref_warm_frames;
        let mut p = 0usize;
        if warm_idx == 0 {
            while p < need_px {
                let q = (luma[p].clamp(0.0, 1.0) * 4095.0 + 0.5) as u16;
                self.ref_luma_q16[p] = q;
                p += 1;
            }
        } else {
            while p < need_px {
                let q = (luma[p].clamp(0.0, 1.0) * 4095.0 + 0.5) as u16;
                self.ref_luma_q16[p] = self.ref_luma_q16[p].saturating_add(q);
                p += 1;
            }
        }
        self.ref_warm_frames = warm_idx + 1;
        self.hist_cursor = 0;
        self.hist_count = 0;

        if self.ref_warm_frames == REF_WARM_TARGET {
            const REF_INV: f32 = 1.0 / ((REF_WARM_TARGET as f32) * 4095.0);
            self.cell_grad_ok.fill(0);
            self.band_grad_ok.fill(0);
            let mut b = 0usize;
            while b < VIBRO_BANDS {
                let bx0 = (b * frame_w / VIBRO_BANDS + 1).min(frame_w - 1);
                let bx1 = ((b + 1) * frame_w / VIBRO_BANDS)
                    .saturating_sub(1)
                    .min(frame_w - 1);
                let ncols = bx1.saturating_sub(bx0);
                let row_floor = 2.0e-5 * (ncols as f32);
                let mut band_good_rows = 0usize;

                let mut cy = 0usize;
                while cy < VIBRO_CELL_H {
                    let ry0 = cy * frame_h / VIBRO_CELL_H;
                    let ry1 = (cy + 1) * frame_h / VIBRO_CELL_H;
                    let mut cell_good = 0usize;
                    let mut ey = ry0;
                    while ey < ry1 {
                        let r = ey * frame_w;
                        let ey_m = ey.saturating_sub(1);
                        let ey_p = (ey + 1).min(frame_h - 1);
                        let rm = ey_m * frame_w;
                        let rp = ey_p * frame_w;
                        let dy_scale = if ey_p > ey_m {
                            1.0 / ((ey_p - ey_m) as f32)
                        } else {
                            0.0
                        };
                        let mut sum_g2 = 0.0f32;
                        let mut ex = bx0;
                        while ex < bx1 {
                            let gx = 0.5
                                * ((self.ref_luma_q16[r + ex + 1] as f32)
                                    - (self.ref_luma_q16[r + ex - 1] as f32))
                                * REF_INV;
                            let gy = dy_scale
                                * ((self.ref_luma_q16[rp + ex] as f32)
                                    - (self.ref_luma_q16[rm + ex] as f32))
                                * REF_INV;
                            sum_g2 += gx * gx + gy * gy;
                            ex += 1;
                        }
                        if ncols > 0 && sum_g2 >= row_floor {
                            cell_good += 1;
                            band_good_rows += 1;
                        }
                        ey += 1;
                    }
                    let cell_rows = ry1.saturating_sub(ry0).max(1);
                    if cell_good * 2 >= cell_rows {
                        self.cell_grad_ok[b * VIBRO_CELL_H + cy] = 1;
                    }
                    cy += 1;
                }
                if band_good_rows * 4 >= frame_h {
                    self.band_grad_ok[b] = 1;
                }
                b += 1;
            }
        }

        self.planes.fill(0.0);
        self.telem[0] = RS_STATE_WARMING;
        self.telem[2] = 0;
        self.telem[3] = 0;
        self.telem[4] = 0;
        self.telem[21] = 0;
    }

    #[inline(never)]
    fn extract_signed_band_displacements(
        &mut self,
        luma: &[f32],
        frame_w: usize,
        frame_h: usize,
        orientation: u32,
        slot: usize,
    ) {
        const REF_INV: f32 = 1.0 / ((REF_WARM_TARGET as f32) * 4095.0);
        let slot_base = slot * VIBRO_BANDS * VIBRO_MAX_ROWS;
        let mut b = 0usize;
        while b < VIBRO_BANDS {
            let dst_base = slot_base + b * VIBRO_MAX_ROWS;
            if self.band_grad_ok[b] == 0 {
                self.disp_hist[dst_base..dst_base + frame_h].fill(0.0);
                b += 1;
                continue;
            }
            let bx0 = (b * frame_w / VIBRO_BANDS + 1).min(frame_w - 1);
            let bx1 = ((b + 1) * frame_w / VIBRO_BANDS)
                .saturating_sub(1)
                .min(frame_w - 1);
            let ncols = bx1.saturating_sub(bx0);
            let row_floor = 2.0e-5 * (ncols as f32);

            let mut m_xx = 0.0f32;
            let mut m_xy = 0.0f32;
            let mut m_yy = 0.0f32;
            let mut ey = 0usize;
            while ey < frame_h {
                let r = ey * frame_w;
                let ey_m = ey.saturating_sub(1);
                let ey_p = (ey + 1).min(frame_h - 1);
                let rm = ey_m * frame_w;
                let rp = ey_p * frame_w;
                let dy_half = if ey_p > ey_m {
                    0.5 / ((ey_p - ey_m) as f32)
                } else {
                    0.0
                };
                let mut s_xx = 0.0f32;
                let mut s_xy = 0.0f32;
                let mut s_yy = 0.0f32;
                let mut b_x = 0.0f32;
                let mut b_y = 0.0f32;
                let mut ex = bx0;
                while ex < bx1 {
                    let cur_c = luma[r + ex];
                    let ref_c = (self.ref_luma_q16[r + ex] as f32) * REF_INV;
                    let di = cur_c - ref_c;

                    let cur_dx = luma[r + ex + 1] - luma[r + ex - 1];
                    let ref_dx = ((self.ref_luma_q16[r + ex + 1] as f32)
                        - (self.ref_luma_q16[r + ex - 1] as f32))
                        * REF_INV;
                    let gx = 0.25 * (cur_dx + ref_dx);

                    let cur_dy = luma[rp + ex] - luma[rm + ex];
                    let ref_dy = ((self.ref_luma_q16[rp + ex] as f32)
                        - (self.ref_luma_q16[rm + ex] as f32))
                        * REF_INV;
                    let gy = dy_half * (cur_dy + ref_dy);

                    s_xx += gx * gx;
                    s_xy += gx * gy;
                    s_yy += gy * gy;
                    b_x -= gx * di;
                    b_y -= gy * di;
                    ex += 1;
                }

                let tr = s_xx + s_yy;
                if ncols > 0 && tr >= row_floor {
                    let det = s_xx * s_yy - s_xy * s_xy;
                    let (mut u_ex, mut u_ey) = if det > 0.04 * tr * tr {
                        ((s_yy * b_x - s_xy * b_y) / det, (s_xx * b_y - s_xy * b_x) / det)
                    } else {
                        (b_x / tr, b_y / tr)
                    };
                    u_ex = u_ex.clamp(-3.0, 3.0);
                    u_ey = u_ey.clamp(-3.0, 3.0);
                    let (u_sx, u_sy) = rotate_disp_to_sensor(u_ex, u_ey, orientation);
                    self.band_row_ux[ey] = u_sx;
                    self.band_row_uy[ey] = u_sy;
                    self.band_row_ok[ey] = 1;
                    m_xx += u_sx * u_sx;
                    m_xy += u_sx * u_sy;
                    m_yy += u_sy * u_sy;
                } else {
                    self.band_row_ux[ey] = 0.0;
                    self.band_row_uy[ey] = 0.0;
                    self.band_row_ok[ey] = 0;
                }
                ey += 1;
            }

            if self.band_axis_locked[b] == 0 {
                let tr_m = m_xx + m_yy;
                if tr_m >= 1.0e-4 * (frame_h as f32) {
                    let diff_m = m_xx - m_yy;
                    let disc_m =
                        crate::wasm_sqrtf((diff_m * diff_m + 4.0 * m_xy * m_xy).max(0.0));
                    let lam1 = 0.5 * (tr_m + disc_m);
                    let (mut vx, mut vy) = if m_xx >= m_yy {
                        (lam1 - m_yy, m_xy)
                    } else {
                        (m_xy, lam1 - m_xx)
                    };
                    let v_norm = crate::wasm_sqrtf(vx * vx + vy * vy);
                    if v_norm > 1.0e-8 {
                        vx /= v_norm;
                        vy /= v_norm;
                        if vx < -1.0e-4 || (abs_f32(vx) <= 1.0e-4 && vy < 0.0) {
                            vx = -vx;
                            vy = -vy;
                        }
                        self.band_ax_x[b] = vx;
                        self.band_ax_y[b] = vy;
                        self.band_axis_locked[b] = 1;
                    }
                }
            }

            let (ax_x, ax_y) = if self.band_axis_locked[b] != 0 {
                (self.band_ax_x[b], self.band_ax_y[b])
            } else {
                (1.0, 0.0)
            };

            let mut ey2 = 0usize;
            while ey2 < frame_h {
                let d = if self.band_row_ok[ey2] != 0 {
                    (self.band_row_ux[ey2] * ax_x + self.band_row_uy[ey2] * ax_y).clamp(-3.0, 3.0)
                } else {
                    0.0
                };
                self.disp_hist[dst_base + ey2] = d;
                ey2 += 1;
            }
            b += 1;
        }
    }

    #[inline(never)]
    fn fft1024_inplace(&mut self) {
        let mut i = 0usize;
        while i < AUDIO_FFT_N {
            let j = FFT1024_TABLES.bitrev[i] as usize;
            if i < j {
                let tr = self.fft_re[i];
                self.fft_re[i] = self.fft_re[j];
                self.fft_re[j] = tr;
                let ti = self.fft_im[i];
                self.fft_im[i] = self.fft_im[j];
                self.fft_im[j] = ti;
            }
            i += 1;
        }

        let mut len = 2usize;
        while len <= AUDIO_FFT_N {
            let half = len >> 1;
            let step = AUDIO_FFT_N / len;
            let mut base = 0usize;
            while base < AUDIO_FFT_N {
                let mut j = 0usize;
                while j < half {
                    let tw_idx = j * step;
                    let wr = FFT1024_TABLES.tw_re[tw_idx];
                    let wi = FFT1024_TABLES.tw_im[tw_idx];
                    let u = base + j;
                    let v = u + half;
                    let vr = self.fft_re[v];
                    let vi = self.fft_im[v];
                    let tr = wr * vr - wi * vi;
                    let ti = wr * vi + wi * vr;
                    let ur = self.fft_re[u];
                    let ui = self.fft_im[u];
                    self.fft_re[u] = ur + tr;
                    self.fft_im[u] = ui + ti;
                    self.fft_re[v] = ur - tr;
                    self.fft_im[v] = ui - ti;
                    j += 1;
                }
                base += len;
            }
            len <<= 1;
        }
    }

    #[inline(never)]
    fn eval_segment_optical_quadrature(&mut self, obs_count: usize, w_sum: f64, df_hz: f64) {
        self.band_mean_y.fill(0.0);
        let mut j = 0usize;
        while j < obs_count {
            let slot_base = (self.obs_slot[j] as usize) * VIBRO_BANDS * VIBRO_MAX_ROWS;
            let ey = self.obs_ey[j] as usize;
            let wj = self.obs_w[j];
            let mut b = 0usize;
            while b < VIBRO_BANDS {
                if self.band_grad_ok[b] != 0 {
                    let y = self.disp_hist[slot_base + b * VIBRO_MAX_ROWS + ey] as f64;
                    self.band_mean_y[b] += wj * y;
                }
                b += 1;
            }
            j += 1;
        }
        let inv_w_sum = 1.0 / w_sum;
        let mut b = 0usize;
        while b < VIBRO_BANDS {
            self.band_mean_y[b] *= inv_w_sum;
            b += 1;
        }

        self.seg_y_re.fill(0.0);
        self.seg_y_im.fill(0.0);
        let scale_y = 2.0 * inv_w_sum;

        j = 0;
        while j < obs_count {
            let t_rel_sec = self.obs_t_rel_us[j] * 1.0e-6;
            let dtheta = -TWO_PI_64 * df_hz * t_rel_sec;
            let (s1, c1) = sin_cos_f64(dtheta);
            let mut wk_re = c1;
            let mut wk_im = s1;
            let mut k = 1usize;
            while k < VIBRO_SPEC_BINS {
                self.tmp_cos[k] = wk_re;
                self.tmp_sin[k] = wk_im;
                let n_re = wk_re * c1 - wk_im * s1;
                let n_im = wk_re * s1 + wk_im * c1;
                wk_re = n_re;
                wk_im = n_im;
                k += 1;
            }

            let slot_base = (self.obs_slot[j] as usize) * VIBRO_BANDS * VIBRO_MAX_ROWS;
            let ey = self.obs_ey[j] as usize;
            let wj = self.obs_w[j];
            let mut b = 0usize;
            while b < VIBRO_BANDS {
                if self.band_grad_ok[b] != 0 {
                    let y = (self.disp_hist[slot_base + b * VIBRO_MAX_ROWS + ey] as f64)
                        - self.band_mean_y[b];
                    let wy = wj * y * scale_y;
                    let row_b = b * VIBRO_SPEC_BINS;
                    let mut k2 = 1usize;
                    while k2 < VIBRO_SPEC_BINS {
                        self.seg_y_re[row_b + k2] += wy * self.tmp_cos[k2];
                        self.seg_y_im[row_b + k2] += wy * self.tmp_sin[k2];
                        k2 += 1;
                    }
                }
                b += 1;
            }
            j += 1;
        }
    }

    #[inline(never)]
    fn advance_welch_segments(&mut self) {
        let h_cnt = self.hist_count as usize;
        if h_cnt == 0
            || self.audio_has_prev_t == 0
            || self.audio_total_samples.saturating_sub(self.audio_interval_start_sample)
                < (AUDIO_FFT_N as u64)
        {
            return;
        }

        let oldest_hist = ((self.hist_cursor as usize) + VIBRO_HIST_FRAMES - h_cnt) % VIBRO_HIST_FRAMES;
        let newest_hist = ((self.hist_cursor as usize) + VIBRO_HIST_FRAMES - 1) % VIBRO_HIST_FRAMES;

        let dt_row = self.dt_row_us as f64;
        let half_exp = 0.5 * (self.exposure_us as f64);

        let old_rows = self.frame_rows_hist[oldest_hist] as i64;
        let old_q_end = if self.q_step_hist[oldest_hist] >= 0 {
            self.q_start_hist[oldest_hist] as f64
        } else {
            (self.q_start_hist[oldest_hist] as i64 + (old_rows - 1) * (self.q_step_hist[oldest_hist] as i64)) as f64
        };
        let opt_t_min_us = (self.t_src_hist_us[oldest_hist] as f64) + old_q_end * dt_row + half_exp;

        let new_rows = self.frame_rows_hist[newest_hist] as i64;
        let new_q_max = if self.q_step_hist[newest_hist] >= 0 {
            (self.q_start_hist[newest_hist] as i64 + (new_rows - 1) * (self.q_step_hist[newest_hist] as i64)) as f64
        } else {
            self.q_start_hist[newest_hist] as f64
        };
        let opt_t_max_us = (self.t_src_hist_us[newest_hist] as f64) + new_q_max * dt_row + half_exp;

        let fs = self.audio_ring_rate_hz as f64;
        let dt_audio_us = 1_000_000.0 / fs;
        let t_win_us = (AUDIO_FFT_N as f64) * dt_audio_us;
        let df_hz = fs / (AUDIO_FFT_N as f64);

        let mut s0 = if self.last_audio_seg_end_sample <= self.audio_interval_start_sample {
            self.audio_interval_start_sample
        } else {
            self.last_audio_seg_end_sample - (AUDIO_HOP as u64)
        };
        let ring_floor = self.audio_total_samples.saturating_sub(AUDIO_RING_LEN as u64);
        let min_s0 = self.audio_interval_start_sample.max(ring_floor);
        if s0 < min_s0 {
            s0 = min_s0;
        }

        while s0 + (AUDIO_FFT_N as u64) <= self.audio_total_samples {
            let t_seg_start_us = (self.audio_last_t_end_us as f64)
                - ((self.audio_total_samples - s0) as f64) * dt_audio_us;
            let t_seg_end_us = t_seg_start_us + t_win_us;

            if t_seg_end_us > opt_t_max_us + dt_row {
                if self.audio_total_samples - s0 > ((AUDIO_RING_LEN - AUDIO_FFT_N) as u64) {
                    self.last_audio_seg_end_sample = s0 + (AUDIO_FFT_N as u64);
                    s0 += AUDIO_HOP as u64;
                    continue;
                }
                break;
            }
            if t_seg_end_us <= opt_t_min_us {
                self.last_audio_seg_end_sample = s0 + (AUDIO_FFT_N as u64);
                s0 += AUDIO_HOP as u64;
                continue;
            }

            let mut obs_count = 0usize;
            let mut m = 0usize;
            while m < h_cnt && obs_count < MAX_SEG_OBS {
                let slot = (oldest_hist + m) % VIBRO_HIST_FRAMES;
                let rows = self.frame_rows_hist[slot] as usize;
                if rows > 0 {
                    let t_src = self.t_src_hist_us[slot] as f64;
                    let qs = self.q_start_hist[slot] as i64;
                    let qstep = self.q_step_hist[slot] as i64;
                    let mut k = 0usize;
                    while k < rows && obs_count < MAX_SEG_OBS {
                        let ey = if qstep >= 0 { k } else { rows - 1 - k };
                        let q = (qs + (ey as i64) * qstep) as f64;
                        let t_row = t_src + q * dt_row + half_exp;
                        if t_row >= t_seg_start_us && t_row <= t_seg_end_us {
                            self.obs_t_rel_us[obs_count] = t_row - t_seg_start_us;
                            self.obs_slot[obs_count] = slot as u16;
                            self.obs_ey[obs_count] = ey as u16;
                            obs_count += 1;
                        }
                        k += 1;
                    }
                }
                m += 1;
            }

            if obs_count < 48 {
                self.last_audio_seg_end_sample = s0 + (AUDIO_FFT_N as u64);
                s0 += AUDIO_HOP as u64;
                continue;
            }

            let mut i = 1usize;
            while i < obs_count {
                let key_t = self.obs_t_rel_us[i];
                let key_s = self.obs_slot[i];
                let key_e = self.obs_ey[i];
                let mut p = i;
                while p > 0 && self.obs_t_rel_us[p - 1] > key_t {
                    self.obs_t_rel_us[p] = self.obs_t_rel_us[p - 1];
                    self.obs_slot[p] = self.obs_slot[p - 1];
                    self.obs_ey[p] = self.obs_ey[p - 1];
                    p -= 1;
                }
                self.obs_t_rel_us[p] = key_t;
                self.obs_slot[p] = key_s;
                self.obs_ey[p] = key_e;
                i += 1;
            }

            let dt_cap = 2.5 * dt_row;
            let mut w_sum = 0.0f64;
            let mut t_cent_num = 0.0f64;
            let mut j = 0usize;
            while j < obs_count {
                let dt_j = if j == 0 {
                    (self.obs_t_rel_us[1] - self.obs_t_rel_us[0]).min(dt_cap)
                } else if j + 1 == obs_count {
                    (self.obs_t_rel_us[j] - self.obs_t_rel_us[j - 1]).min(dt_cap)
                } else {
                    let dl = (self.obs_t_rel_us[j] - self.obs_t_rel_us[j - 1]).min(dt_cap);
                    let dr = (self.obs_t_rel_us[j + 1] - self.obs_t_rel_us[j]).min(dt_cap);
                    0.5 * (dl + dr)
                };
                let tr = self.obs_t_rel_us[j];
                let (_, c_hann) = sin_cos_f64(TWO_PI_64 * tr / t_win_us);
                let w_hann = 0.5 * (1.0 - c_hann);
                let w_tot = w_hann * dt_j.max(0.0);
                self.obs_w[j] = w_tot;
                w_sum += w_tot;
                t_cent_num += w_tot * tr;
                j += 1;
            }

            let t_centroid = if w_sum > 0.0 { t_cent_num / w_sum } else { 0.0 };
            if w_sum < 0.11 * t_win_us || abs_f64(t_centroid - 0.5 * t_win_us) > 0.30 * t_win_us {
                self.last_audio_seg_end_sample = s0 + (AUDIO_FFT_N as u64);
                s0 += AUDIO_HOP as u64;
                continue;
            }

            let ring_base = self.audio_write_cursor - (self.audio_total_samples - s0);
            let mut n = 0usize;
            while n < AUDIO_FFT_N {
                let idx = ((ring_base + (n as u64)) as usize) & (AUDIO_RING_LEN - 1);
                self.fft_re[n] = (self.audio_ring[idx] as f64) * FFT1024_TABLES.hann[n];
                self.fft_im[n] = 0.0;
                n += 1;
            }
            self.fft1024_inplace();

            const SCALE_X: f64 = 2.0 / 512.0;
            let mut k = 1usize;
            while k < VIBRO_SPEC_BINS {
                self.seg_x_re[k] = self.fft_re[k] * SCALE_X;
                self.seg_x_im[k] = self.fft_im[k] * SCALE_X;
                k += 1;
            }

            self.eval_segment_optical_quadrature(obs_count, w_sum, df_hz);

            let w_slot = (self.welch_cursor as usize) % VIBRO_WELCH_SLOTS;
            let pxx_base = w_slot * VIBRO_SPEC_BINS;
            let mut k_aud = 1usize;
            let mut max_pxx_seg = -1.0f64;
            k = 1;
            while k < VIBRO_SPEC_BINS {
                let xr = self.seg_x_re[k];
                let xi = self.seg_x_im[k];
                let pxx = xr * xr + xi * xi;
                self.welch_pxx[pxx_base + k] = pxx;
                if pxx > max_pxx_seg {
                    max_pxx_seg = pxx;
                    k_aud = k;
                }
                k += 1;
            }

            let mut max_pyy_aud = -1.0f64;
            let mut best_k1_coh = 0.0f64;
            let xr_a = self.seg_x_re[k_aud];
            let xi_a = self.seg_x_im[k_aud];
            let pxx_a = xr_a * xr_a + xi_a * xi_a;

            let mut b = 0usize;
            while b < VIBRO_BANDS {
                if self.band_grad_ok[b] != 0 {
                    let src_b = b * VIBRO_SPEC_BINS;
                    let dst_b = (w_slot * VIBRO_BANDS + b) * VIBRO_SPEC_BINS;
                    let mut k2 = 1usize;
                    while k2 < VIBRO_SPEC_BINS {
                        let xr = self.seg_x_re[k2];
                        let xi = self.seg_x_im[k2];
                        let yr = self.seg_y_re[src_b + k2];
                        let yi = self.seg_y_im[src_b + k2];
                        let pyy = yr * yr + yi * yi;
                        let pxy_r = yr * xr + yi * xi;
                        let pxy_i = yi * xr - yr * xi;
                        self.welch_pyy[dst_b + k2] = pyy;
                        self.welch_pxy_re[dst_b + k2] = pxy_r;
                        self.welch_pxy_im[dst_b + k2] = pxy_i;
                        k2 += 1;
                    }

                    let yr_a = self.seg_y_re[src_b + k_aud];
                    let yi_a = self.seg_y_im[src_b + k_aud];
                    let pyy_a = yr_a * yr_a + yi_a * yi_a;
                    if pyy_a > max_pyy_aud {
                        max_pyy_aud = pyy_a;
                        let denom = pxx_a * pyy_a;
                        if denom > 1.0e-30 {
                            let cr = yr_a * xr_a + yi_a * xi_a;
                            let ci = yi_a * xr_a - yr_a * xi_a;
                            best_k1_coh = ((cr * cr + ci * ci) / denom).clamp(0.0, 1.0);
                        } else {
                            best_k1_coh = 0.0;
                        }
                    }
                }
                b += 1;
            }
            self.single_seg_k1_coh = best_k1_coh as f32;

            self.welch_cursor = ((w_slot + 1) % VIBRO_WELCH_SLOTS) as u32;
            if (self.welch_count as usize) < VIBRO_WELCH_SLOTS {
                self.welch_count += 1;
            }
            self.last_audio_seg_end_sample = s0 + (AUDIO_FFT_N as u64);
            s0 += AUDIO_HOP as u64;
        }
    }

    #[inline(never)]
    fn finalize_welch_solution(&mut self) {
        let k_segs = self.welch_count;
        self.telem[3] = k_segs;
        self.telem[9] = self.single_seg_k1_coh.to_bits();

        if self.audio_has_prev_t == 0
            || self.audio_total_samples.saturating_sub(self.audio_interval_start_sample)
                < (AUDIO_FFT_N as u64)
        {
            self.planes.fill(0.0);
            self.telem[0] = RS_STATE_WARMING;
            self.telem[2] |= RS_REASON_AUDIO_UNDERRUN;
            self.telem[4] = 0;
            self.telem[6] = 0;
            self.telem[7] = (1.0f32).to_bits();
            self.telem[13] = (self.calib_audio_rate_hz / (AUDIO_FFT_N as f32)).to_bits();
            self.telem[21] = 0;
            return;
        }

        let df_hz = (self.audio_ring_rate_hz as f64) / (AUDIO_FFT_N as f64);
        self.telem[13] = (df_hz as f32).to_bits();

        if k_segs < 8 {
            self.planes.fill(0.0);
            self.telem[0] = RS_STATE_WARMING;
            self.telem[4] = 0;
            self.telem[6] = 0;
            self.telem[7] = (1.0f32).to_bits();
            self.telem[21] = 0;
            return;
        }

        let inv_k = 1.0 / (k_segs as f64);
        let k_eff = 0.95 * (k_segs as f64);
        let gamma2_crit = 1.0 - exp_neg_f64(LN_ALPHA_BIN_63 / (k_eff - 1.0));
        let gamma2_crit_f32 = gamma2_crit as f32;

        self.mean_pxx.fill(0.0);
        let mut s = 0usize;
        while s < (k_segs as usize) {
            let pxx_base = s * VIBRO_SPEC_BINS;
            let mut k = 1usize;
            while k < VIBRO_SPEC_BINS {
                self.mean_pxx[k] += self.welch_pxx[pxx_base + k];
                k += 1;
            }
            s += 1;
        }

        let mut max_pxx = 0.0f64;
        let mut k_audio_peak = 1usize;
        let mut k = 1usize;
        while k < VIBRO_SPEC_BINS {
            let v = self.mean_pxx[k] * inv_k;
            self.mean_pxx[k] = v;
            if v > max_pxx {
                max_pxx = v;
                k_audio_peak = k;
            }
            k += 1;
        }
        let eps_xx = (1.0e-5 * max_pxx).max(1.0e-12);

        self.planes[VIBRO_PLANE_SPECTRUM..VIBRO_PLANE_SPECTRUM + VIBRO_SPEC_BINS].fill(0.0);
        let mut max_coh_gamma2 = -1.0f64;
        let mut dom_b = 0usize;
        let mut dom_k = 1usize;

        let mut b = 0usize;
        while b < VIBRO_BANDS {
            if self.band_grad_ok[b] == 0 {
                b += 1;
                continue;
            }
            self.band_mean_pyy.fill(0.0);
            self.band_mean_pxy_re.fill(0.0);
            self.band_mean_pxy_im.fill(0.0);
            let mut s2 = 0usize;
            while s2 < (k_segs as usize) {
                let src = (s2 * VIBRO_BANDS + b) * VIBRO_SPEC_BINS;
                let mut k2 = 1usize;
                while k2 < VIBRO_SPEC_BINS {
                    self.band_mean_pyy[k2] += self.welch_pyy[src + k2];
                    self.band_mean_pxy_re[k2] += self.welch_pxy_re[src + k2];
                    self.band_mean_pxy_im[k2] += self.welch_pxy_im[src + k2];
                    k2 += 1;
                }
                s2 += 1;
            }

            let mut max_pyy_b = 0.0f64;
            let mut k2 = 1usize;
            while k2 < VIBRO_SPEC_BINS {
                self.band_mean_pyy[k2] *= inv_k;
                self.band_mean_pxy_re[k2] *= inv_k;
                self.band_mean_pxy_im[k2] *= inv_k;
                if self.band_mean_pyy[k2] > max_pyy_b {
                    max_pyy_b = self.band_mean_pyy[k2];
                }
                k2 += 1;
            }
            let eps_yy_b = (1.0e-4 * max_pyy_b).max(1.0e-6);

            k2 = 1;
            while k2 < VIBRO_SPEC_BINS {
                let cr = self.band_mean_pxy_re[k2];
                let ci = self.band_mean_pxy_im[k2];
                let denom = (self.mean_pxx[k2] + eps_xx) * (self.band_mean_pyy[k2] + eps_yy_b);
                let g2 = ((cr * cr + ci * ci) / denom).clamp(0.0, 1.0);
                let prev_spec = self.planes[VIBRO_PLANE_SPECTRUM + k2];
                if (g2 as f32) > prev_spec {
                    self.planes[VIBRO_PLANE_SPECTRUM + k2] = g2 as f32;
                }
                if g2 > max_coh_gamma2 {
                    max_coh_gamma2 = g2;
                    dom_b = b;
                    dom_k = k2;
                }
                k2 += 1;
            }
            b += 1;
        }
        if max_coh_gamma2 < 0.0 {
            max_coh_gamma2 = 0.0;
        }

        if max_coh_gamma2 < gamma2_crit && max_pxx > 1.0e-6 {
            let h_aud = sinc_exposure_h((k_audio_peak as f64) * df_hz, self.exposure_us as f64);
            if abs_f64(h_aud) < 0.05 {
                dom_k = k_audio_peak;
            }
        }

        let dom_freq_hz = (dom_k as f64) * df_hz;
        let h = sinc_exposure_h(dom_freq_hz, self.exposure_us as f64);
        let h_abs = abs_f64(h);
        let mut reason_bits = self.telem[2] & !RS_REASON_AUDIO_UNDERRUN & !RS_REASON_EXPOSURE_NULL;
        if h_abs < 0.05 {
            reason_bits |= RS_REASON_EXPOSURE_NULL;
        }

        let mut dom_raw_amp_null = 0.0f32;
        let mut dom_raw_phase_null = 0.0f32;
        let m_pxx_dom = self.mean_pxx[dom_k] + eps_xx;

        b = 0;
        while b < VIBRO_BANDS {
            if self.band_grad_ok[b] == 0 {
                self.band_coh[b] = 0.0;
                self.band_amp[b] = 0.0;
                self.band_phase[b] = 0.0;
                b += 1;
                continue;
            }

            let mut max_pyy_b = 0.0f64;
            let mut k2 = 1usize;
            while k2 < VIBRO_SPEC_BINS {
                let mut sum_p = 0.0f64;
                let mut s2 = 0usize;
                while s2 < (k_segs as usize) {
                    sum_p += self.welch_pyy[(s2 * VIBRO_BANDS + b) * VIBRO_SPEC_BINS + k2];
                    s2 += 1;
                }
                let mp = sum_p * inv_k;
                if mp > max_pyy_b {
                    max_pyy_b = mp;
                }
                k2 += 1;
            }
            let eps_yy_b = (1.0e-4 * max_pyy_b).max(1.0e-6);

            let mut sum_pyy_d = 0.0f64;
            let mut sum_cr_d = 0.0f64;
            let mut sum_ci_d = 0.0f64;
            let mut s2 = 0usize;
            while s2 < (k_segs as usize) {
                let src = (s2 * VIBRO_BANDS + b) * VIBRO_SPEC_BINS + dom_k;
                sum_pyy_d += self.welch_pyy[src];
                sum_cr_d += self.welch_pxy_re[src];
                sum_ci_d += self.welch_pxy_im[src];
                s2 += 1;
            }
            let m_pyy_d = sum_pyy_d * inv_k;
            let cr_d = sum_cr_d * inv_k;
            let ci_d = sum_ci_d * inv_k;
            let c2_d = cr_d * cr_d + ci_d * ci_d;
            let g2_b = (c2_d / (m_pxx_dom * (m_pyy_d + eps_yy_b))).clamp(0.0, 1.0);
            let coh_b = g2_b as f32;
            self.band_coh[b] = coh_b;

            let amp_raw = crate::wasm_sqrt_f64((c2_d / m_pxx_dom).max(0.0)) as f32;
            let phase_raw = if c2_d > 1.0e-24 {
                crate::fast_atan2f(ci_d as f32, cr_d as f32)
            } else {
                0.0
            };
            if b == dom_b {
                dom_raw_amp_null = crate::wasm_sqrt_f64(m_pyy_d.max(0.0)) as f32;
                dom_raw_phase_null = phase_raw;
            }

            if h_abs >= 0.05 && coh_b >= gamma2_crit_f32 {
                self.band_amp[b] = amp_raw / (h_abs as f32);
                self.band_phase[b] = if h < 0.0 {
                    crate::wrap_pi(phase_raw + PI_32)
                } else {
                    phase_raw
                };
            } else if h_abs < 0.05 && coh_b >= gamma2_crit_f32 {
                self.band_amp[b] = amp_raw;
                self.band_phase[b] = phase_raw;
            } else {
                self.band_amp[b] = 0.0;
                self.band_phase[b] = 0.0;
            }
            b += 1;
        }

        let (valid_cells, coupled_cells) = self.project_bands_to_grid(true, gamma2_crit_f32);
        let (dom_amp_px, dom_phase_rad) = if h_abs < 0.05 {
            (dom_raw_amp_null, dom_raw_phase_null)
        } else {
            (self.band_amp[dom_b], self.band_phase[dom_b])
        };

        self.telem[0] = RS_STATE_READY;
        self.telem[2] = reason_bits;
        self.telem[4] = valid_cells;
        self.telem[5] = (dom_freq_hz as f32).to_bits();
        self.telem[6] = (max_coh_gamma2 as f32).to_bits();
        self.telem[7] = gamma2_crit_f32.to_bits();
        self.telem[8] = (h as f32).to_bits();
        self.telem[11] = dom_phase_rad.to_bits();
        self.telem[12] = dom_amp_px.to_bits();
        self.telem[20] = dom_b as u32;
        self.telem[21] = coupled_cells;
    }

    #[inline(never)]
    fn eval_rs_band_power_at_freq(&mut self, b: usize, freq_hz: f64, m_frames: usize) -> f64 {
        let h_cnt = self.hist_count as usize;
        let oldest = ((self.hist_cursor as usize) + VIBRO_HIST_FRAMES - h_cnt) % VIBRO_HIST_FRAMES;
        let start_m = h_cnt - m_frames;
        let dt_row = self.dt_row_us as f64;
        let mut inc_pow = 0.0f64;

        let mut m = start_m;
        while m < h_cnt {
            let slot = (oldest + m) % VIBRO_HIST_FRAMES;
            let rows = self.frame_rows_hist[slot] as usize;
            if rows > 1 {
                let slot_base = slot * VIBRO_BANDS * VIBRO_MAX_ROWS + b * VIBRO_MAX_ROWS;
                let denom = (rows - 1) as f64;
                let mut w_sum = 0.0f64;
                let mut y_wsum = 0.0f64;
                let mut ey = 0usize;
                while ey < rows {
                    let (_, c_h) = sin_cos_f64(TWO_PI_64 * (ey as f64) / denom);
                    let w = 0.5 * (1.0 - c_h);
                    w_sum += w;
                    y_wsum += w * (self.disp_hist[slot_base + ey] as f64);
                    ey += 1;
                }
                if w_sum > 0.0 {
                    let inv_w = 1.0 / w_sum;
                    let mean_y = y_wsum * inv_w;
                    let t_src = self.t_src_hist_us[slot];
                    let t_base_sec = ((t_src - self.run_t0_us) as f64) * 1.0e-6;
                    let qs = self.q_start_hist[slot] as i64;
                    let qstep = self.q_step_hist[slot] as i64;
                    let mut re = 0.0f64;
                    let mut im = 0.0f64;
                    ey = 0;
                    while ey < rows {
                        let (_, c_h) = sin_cos_f64(TWO_PI_64 * (ey as f64) / denom);
                        let w = 0.5 * (1.0 - c_h);
                        let q = (qs + (ey as i64) * qstep) as f64;
                        let t_sec = t_base_sec + (q * dt_row + 0.5 * (self.exposure_us as f64)) * 1.0e-6;
                        let (s, c) = sin_cos_f64(-TWO_PI_64 * freq_hz * t_sec);
                        let wy = w * ((self.disp_hist[slot_base + ey] as f64) - mean_y);
                        re += wy * c;
                        im += wy * s;
                        ey += 1;
                    }
                    let scale = 2.0 * inv_w;
                    re *= scale;
                    im *= scale;
                    inc_pow += re * re + im * im;
                }
            }
            m += 1;
        }
        inc_pow / (m_frames.max(1) as f64)
    }

    #[inline(never)]
    fn refine_rs_only_freq(&mut self, dom_b: usize, dom_k: usize, df_hz: f64, m_frames: usize) -> f64 {
        let mut lo = ((dom_k as f64) - 1.0) * df_hz;
        let min_lo = 0.5 * df_hz;
        if lo < min_lo {
            lo = min_lo;
        }
        let mut hi = ((dom_k + 1) as f64) * df_hz;
        let mut best_f = (dom_k as f64) * df_hz;
        let mut best_j = 4usize;
        let mut step = 0.0f64;

        let mut stage = 0usize;
        while stage < 2 {
            step = (hi - lo) * 0.125;
            let mut best_s = -1.0f64;
            let mut j = 0usize;
            while j < 9 {
                let fj = lo + (j as f64) * step;
                let s = self.eval_rs_band_power_at_freq(dom_b, fj, m_frames);
                self.rs_refine_scores[j] = s;
                if s > best_s {
                    best_s = s;
                    best_f = fj;
                    best_j = j;
                }
                j += 1;
            }
            if stage == 0 {
                lo = (best_f - step).max(min_lo);
                hi = best_f + step;
            }
            stage += 1;
        }

        if best_j > 0 && best_j < 8 {
            let s_l = self.rs_refine_scores[best_j - 1];
            let s_m = self.rs_refine_scores[best_j];
            let s_r = self.rs_refine_scores[best_j + 1];
            let denom = s_l - 2.0 * s_m + s_r;
            if denom < -1.0e-24 {
                let delta = (0.5 * (s_l - s_r) / denom).clamp(-0.5, 0.5);
                best_f += delta * step;
            }
        }
        best_f
    }

    #[inline(never)]
    fn solve_rs_only(&mut self, frame_h: usize) {
        let dt_row = self.dt_row_us as f64;
        let df_hz = (48_000.0f64 / (AUDIO_FFT_N as f64))
            .min(0.45 * (1_000_000.0 / dt_row) / (VIBRO_SPEC_BINS as f64));
        self.telem[13] = (df_hz as f32).to_bits();

        let slot = ((self.hist_cursor as usize) + VIBRO_HIST_FRAMES - 1) % VIBRO_HIST_FRAMES;
        let slot_base = slot * VIBRO_BANDS * VIBRO_MAX_ROWS;
        let denom = (frame_h.saturating_sub(1)).max(1) as f64;
        let t_src = self.t_src_hist_us[slot];
        let t_base_sec = ((t_src - self.run_t0_us) as f64) * 1.0e-6;
        let qs = self.q_start_hist[slot] as i64;
        let qstep = self.q_step_hist[slot] as i64;

        self.band_mean_y.fill(0.0);
        let mut w_sum = 0.0f64;
        let mut ey = 0usize;
        while ey < frame_h {
            let (_, c_h) = sin_cos_f64(TWO_PI_64 * (ey as f64) / denom);
            let w = 0.5 * (1.0 - c_h);
            w_sum += w;
            let mut b = 0usize;
            while b < VIBRO_BANDS {
                if self.band_grad_ok[b] != 0 {
                    self.band_mean_y[b] += w * (self.disp_hist[slot_base + b * VIBRO_MAX_ROWS + ey] as f64);
                }
                b += 1;
            }
            ey += 1;
        }
        let inv_w_sum = if w_sum > 0.0 { 1.0 / w_sum } else { 0.0 };
        let mut b = 0usize;
        while b < VIBRO_BANDS {
            self.band_mean_y[b] *= inv_w_sum;
            b += 1;
        }

        self.seg_y_re.fill(0.0);
        self.seg_y_im.fill(0.0);
        let scale_y = 2.0 * inv_w_sum;
        ey = 0;
        while ey < frame_h {
            let (_, c_h) = sin_cos_f64(TWO_PI_64 * (ey as f64) / denom);
            let w = 0.5 * (1.0 - c_h);
            let q = (qs + (ey as i64) * qstep) as f64;
            let t_sec = t_base_sec + (q * dt_row + 0.5 * (self.exposure_us as f64)) * 1.0e-6;
            let (s1, c1) = sin_cos_f64(-TWO_PI_64 * df_hz * t_sec);
            let mut wk_re = c1;
            let mut wk_im = s1;
            let mut k = 1usize;
            while k < VIBRO_SPEC_BINS {
                self.tmp_cos[k] = wk_re;
                self.tmp_sin[k] = wk_im;
                let n_re = wk_re * c1 - wk_im * s1;
                let n_im = wk_re * s1 + wk_im * c1;
                wk_re = n_re;
                wk_im = n_im;
                k += 1;
            }
            let mut b2 = 0usize;
            while b2 < VIBRO_BANDS {
                if self.band_grad_ok[b2] != 0 {
                    let y = (self.disp_hist[slot_base + b2 * VIBRO_MAX_ROWS + ey] as f64)
                        - self.band_mean_y[b2];
                    let wy = w * y * scale_y;
                    let row_b = b2 * VIBRO_SPEC_BINS;
                    let mut k2 = 1usize;
                    while k2 < VIBRO_SPEC_BINS {
                        self.seg_y_re[row_b + k2] += wy * self.tmp_cos[k2];
                        self.seg_y_im[row_b + k2] += wy * self.tmp_sin[k2];
                        k2 += 1;
                    }
                }
                b2 += 1;
            }
            ey += 1;
        }

        let w_slot = (self.welch_cursor as usize) % VIBRO_WELCH_SLOTS;
        b = 0;
        while b < VIBRO_BANDS {
            let src_b = b * VIBRO_SPEC_BINS;
            let dst_b = (w_slot * VIBRO_BANDS + b) * VIBRO_SPEC_BINS;
            let mut k = 1usize;
            while k < VIBRO_SPEC_BINS {
                let yr = self.seg_y_re[src_b + k];
                let yi = self.seg_y_im[src_b + k];
                self.welch_pyy[dst_b + k] = yr * yr + yi * yi;
                k += 1;
            }
            b += 1;
        }
        self.welch_cursor = ((w_slot + 1) % VIBRO_WELCH_SLOTS) as u32;
        if (self.welch_count as usize) < VIBRO_WELCH_SLOTS {
            self.welch_count += 1;
        }

        self.telem[3] = self.hist_count;
        if self.hist_count < 8 {
            self.planes.fill(0.0);
            self.telem[0] = RS_STATE_WARMING;
            self.telem[2] = 0;
            self.telem[4] = 0;
            self.telem[21] = 0;
            return;
        }

        let n_w = self.welch_count.max(1) as usize;
        let inv_w = 1.0 / (n_w as f64);
        self.planes[VIBRO_PLANE_SPECTRUM..VIBRO_PLANE_SPECTRUM + VIBRO_SPEC_BINS].fill(0.0);
        let mut max_pyy = -1.0f64;
        let mut dom_b = 0usize;
        let mut dom_k = 1usize;

        b = 0;
        while b < VIBRO_BANDS {
            if self.band_grad_ok[b] == 0 {
                b += 1;
                continue;
            }
            let mut k = 1usize;
            while k < VIBRO_SPEC_BINS {
                let mut sum_p = 0.0f64;
                let mut s = 0usize;
                while s < n_w {
                    sum_p += self.welch_pyy[(s * VIBRO_BANDS + b) * VIBRO_SPEC_BINS + k];
                    s += 1;
                }
                let mp = sum_p * inv_w;
                let amp_spec = crate::wasm_sqrt_f64(mp.max(0.0)) as f32;
                let prev = self.planes[VIBRO_PLANE_SPECTRUM + k];
                if amp_spec > prev {
                    self.planes[VIBRO_PLANE_SPECTRUM + k] = amp_spec;
                }
                if mp > max_pyy {
                    max_pyy = mp;
                    dom_b = b;
                    dom_k = k;
                }
                k += 1;
            }
            b += 1;
        }

        if max_pyy <= 0.0 || crate::wasm_sqrt_f64(max_pyy) < 1.5e-3 {
            self.band_amp.fill(0.0);
            self.band_phase.fill(0.0);
            self.band_coh.fill(0.0);
            let (valid_cells, _) = self.project_bands_to_grid(false, 0.0);
            self.telem[0] = RS_STATE_READY;
            self.telem[2] = 0;
            self.telem[4] = valid_cells;
            self.telem[5] = 0;
            self.telem[6] = 0;
            self.telem[7] = 0;
            self.telem[8] = (1.0f32).to_bits();
            self.telem[9] = 0;
            self.telem[11] = 0;
            self.telem[12] = 0;
            self.telem[20] = 0;
            self.telem[21] = 0;
            return;
        }

        let m_frames = (self.hist_count as usize).min(8);
        let dom_freq_hz = self.refine_rs_only_freq(dom_b, dom_k, df_hz, m_frames);
        let h = sinc_exposure_h(dom_freq_hz, self.exposure_us as f64);
        let h_abs = abs_f64(h);
        let mut reason_bits = 0u32;
        if h_abs < 0.05 {
            reason_bits |= RS_REASON_EXPOSURE_NULL;
        }

        self.rs_pow_sum.fill(0.0);
        let h_cnt = self.hist_count as usize;
        let oldest = ((self.hist_cursor as usize) + VIBRO_HIST_FRAMES - h_cnt) % VIBRO_HIST_FRAMES;
        let start_m = h_cnt - m_frames;
        let mut m = start_m;
        while m < h_cnt {
            let cur_slot = (oldest + m) % VIBRO_HIST_FRAMES;
            let rows = self.frame_rows_hist[cur_slot] as usize;
            if rows > 1 {
                let s_base = cur_slot * VIBRO_BANDS * VIBRO_MAX_ROWS;
                let r_denom = (rows - 1) as f64;
                let mut row_w_sum = 0.0f64;
                self.band_mean_y.fill(0.0);
                let mut ey2 = 0usize;
                while ey2 < rows {
                    let (_, c_h) = sin_cos_f64(TWO_PI_64 * (ey2 as f64) / r_denom);
                    let w = 0.5 * (1.0 - c_h);
                    row_w_sum += w;
                    let mut b2 = 0usize;
                    while b2 < VIBRO_BANDS {
                        if self.band_grad_ok[b2] != 0 {
                            self.band_mean_y[b2] +=
                                w * (self.disp_hist[s_base + b2 * VIBRO_MAX_ROWS + ey2] as f64);
                        }
                        b2 += 1;
                    }
                    ey2 += 1;
                }
                if row_w_sum > 0.0 {
                    let inv_rw = 1.0 / row_w_sum;
                    let mut b2 = 0usize;
                    while b2 < VIBRO_BANDS {
                        self.band_mean_y[b2] *= inv_rw;
                        b2 += 1;
                    }
                    self.rs_tmp_re.fill(0.0);
                    self.rs_tmp_im.fill(0.0);
                    let t_b_sec = ((self.t_src_hist_us[cur_slot] - self.run_t0_us) as f64) * 1.0e-6;
                    let qs_m = self.q_start_hist[cur_slot] as i64;
                    let qstep_m = self.q_step_hist[cur_slot] as i64;
                    ey2 = 0;
                    while ey2 < rows {
                        let (_, c_h) = sin_cos_f64(TWO_PI_64 * (ey2 as f64) / r_denom);
                        let w = 0.5 * (1.0 - c_h);
                        let q = (qs_m + (ey2 as i64) * qstep_m) as f64;
                        let t_sec =
                            t_b_sec + (q * dt_row + 0.5 * (self.exposure_us as f64)) * 1.0e-6;
                        let (s, c) = sin_cos_f64(-TWO_PI_64 * dom_freq_hz * t_sec);
                        let mut b3 = 0usize;
                        while b3 < VIBRO_BANDS {
                            if self.band_grad_ok[b3] != 0 {
                                let wy = w
                                    * ((self.disp_hist[s_base + b3 * VIBRO_MAX_ROWS + ey2] as f64)
                                        - self.band_mean_y[b3]);
                                self.rs_tmp_re[b3] += wy * c;
                                self.rs_tmp_im[b3] += wy * s;
                            }
                            b3 += 1;
                        }
                        ey2 += 1;
                    }
                    let sc = 2.0 * inv_rw;
                    let mut b3 = 0usize;
                    while b3 < VIBRO_BANDS {
                        self.rs_tmp_re[b3] *= sc;
                        self.rs_tmp_im[b3] *= sc;
                        self.rs_pow_sum[b3] += self.rs_tmp_re[b3] * self.rs_tmp_re[b3]
                            + self.rs_tmp_im[b3] * self.rs_tmp_im[b3];
                        b3 += 1;
                    }
                }
            }
            m += 1;
        }

        let inv_m = 1.0 / (m_frames.max(1) as f64);
        b = 0;
        while b < VIBRO_BANDS {
            self.band_coh[b] = 0.0;
            if self.band_grad_ok[b] == 0 {
                self.band_amp[b] = 0.0;
                self.band_phase[b] = 0.0;
                b += 1;
                continue;
            }
            let amp_raw = crate::wasm_sqrt_f64((self.rs_pow_sum[b] * inv_m).max(0.0)) as f32;
            if amp_raw < 1.5e-3 {
                self.band_amp[b] = 0.0;
                self.band_phase[b] = 0.0;
                b += 1;
                continue;
            }
            let phase_raw =
                crate::fast_atan2f(self.rs_tmp_im[b] as f32, self.rs_tmp_re[b] as f32);
            if h_abs >= 0.05 {
                self.band_amp[b] = amp_raw / (h_abs as f32);
                self.band_phase[b] = if h < 0.0 {
                    crate::wrap_pi(phase_raw + PI_32)
                } else {
                    phase_raw
                };
            } else {
                self.band_amp[b] = amp_raw;
                self.band_phase[b] = phase_raw;
            }
            b += 1;
        }

        let (valid_cells, _) = self.project_bands_to_grid(false, 0.0);
        self.telem[0] = RS_STATE_READY;
        self.telem[2] = reason_bits;
        self.telem[4] = valid_cells;
        self.telem[5] = (dom_freq_hz as f32).to_bits();
        self.telem[6] = 0;
        self.telem[7] = 0;
        self.telem[8] = (h as f32).to_bits();
        self.telem[9] = 0;
        self.telem[11] = self.band_phase[dom_b].to_bits();
        self.telem[12] = self.band_amp[dom_b].to_bits();
        self.telem[20] = dom_b as u32;
        self.telem[21] = 0;
    }

    #[inline(never)]
    fn project_bands_to_grid(&mut self, check_coh: bool, gamma2_crit: f32) -> (u32, u32) {
        let mut valid_cells = 0u32;
        let mut coupled_cells = 0u32;
        let mut cy = 0usize;
        while cy < VIBRO_CELL_H {
            let mut cx = 0usize;
            while cx < VIBRO_CELL_W {
                let cell = cy * VIBRO_CELL_W + cx;
                let c_x0 = 2 * cx;
                let c_x1 = c_x0 + 2;
                let b0 = c_x0 / 5;
                let b1 = (c_x1 - 1) / 5;

                let mut w_sum = 0.0f32;
                let mut coh_sum = 0.0f32;
                let mut ph_re = 0.0f32;
                let mut ph_im = 0.0f32;
                let mut b = b0;
                while b <= b1 {
                    let b_x0 = 5 * b;
                    let b_x1 = b_x0 + 5;
                    let ov_px = c_x1.min(b_x1).saturating_sub(c_x0.max(b_x0));
                    let ov = (ov_px as f32) * 0.5;
                    if ov > 0.0 && self.cell_grad_ok[b * VIBRO_CELL_H + cy] != 0 {
                        w_sum += ov;
                        coh_sum += ov * self.band_coh[b];
                        let (s, c) = sin_cos_f64(self.band_phase[b] as f64);
                        ph_re += ov * self.band_amp[b] * (c as f32);
                        ph_im += ov * self.band_amp[b] * (s as f32);
                    }
                    b += 1;
                }

                if w_sum > 0.0 {
                    let inv_w = 1.0 / w_sum;
                    let cell_coh = coh_sum * inv_w;
                    let cell_amp = crate::wasm_sqrtf(ph_re * ph_re + ph_im * ph_im) * inv_w;
                    let cell_phase = if cell_amp > 1.0e-6 {
                        crate::fast_atan2f(ph_im, ph_re)
                    } else {
                        0.0
                    };
                    if check_coh {
                        self.planes[VIBRO_PLANE_COHERENCE + cell] = cell_coh;
                        if cell_coh >= gamma2_crit {
                            self.planes[VIBRO_PLANE_AMP + cell] = cell_amp;
                            self.planes[VIBRO_PLANE_PHASE + cell] = cell_phase;
                            valid_cells += 1;
                            coupled_cells += 1;
                        } else {
                            self.planes[VIBRO_PLANE_AMP + cell] = 0.0;
                            self.planes[VIBRO_PLANE_PHASE + cell] = 0.0;
                        }
                    } else {
                        self.planes[VIBRO_PLANE_COHERENCE + cell] = 0.0;
                        self.planes[VIBRO_PLANE_AMP + cell] = cell_amp;
                        self.planes[VIBRO_PLANE_PHASE + cell] = cell_phase;
                        valid_cells += 1;
                    }
                } else {
                    self.planes[VIBRO_PLANE_AMP + cell] = 0.0;
                    self.planes[VIBRO_PLANE_COHERENCE + cell] = 0.0;
                    self.planes[VIBRO_PLANE_PHASE + cell] = 0.0;
                }
                cx += 1;
            }
            cy += 1;
        }
        (valid_cells, coupled_cells)
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
