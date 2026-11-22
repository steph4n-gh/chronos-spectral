//! Multi-instance C ABI (`ChronosEngine*`) for the native CHRONOS-SPECTRAL kernel (v3.4.0, ABI 4).
//!
//! Kept in its own crate (`native/capi/`) so the core `#![no_std]` Wasm module stays zero-import
//! and `native/replay/kernel.rs`'s export census over `src/*.rs` stays untouched.

mod audit;

use audit::{
    ChronosAuditConfig, ChronosAuditSpan, ChronosAuditSummary, ClipAuditor, CHRONOS_ABI_VERSION,
};
use chronos_spectral::native::{ChronosEngine, NativeIngestMeta};
use core::ffi::{c_char, c_void};

/// Opaque handle to an independent heap-allocated `CapiEngine` (~23.3 MB `Arena` + `ClipAuditor`).
pub type ChronosEngineHandle = c_void;

struct CapiEngine {
    eng: ChronosEngine,
    auditor: ClipAuditor,
    pending_pts_us: Option<u64>,
    last_luma_sig: [f32; 64],
}

impl CapiEngine {
    fn new() -> Self {
        Self {
            eng: ChronosEngine::new(),
            auditor: ClipAuditor::new(),
            pending_pts_us: None,
            last_luma_sig: [0.0; 64],
        }
    }

    fn reset(&mut self) {
        self.eng.reset();
        self.auditor = ClipAuditor::new();
        self.pending_pts_us = None;
        self.last_luma_sig = [0.0; 64];
    }
}

/// Zero-copy read-only descriptor of a kernel output field/buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChronosFieldView {
    pub ptr: *const c_void,
    pub len: usize,
    pub elem_size: u32,
    /// `1 = u8`, `2 = u32`, `3 = f32`
    pub elem_kind: u32,
    pub rows: u32,
    pub cols: u32,
    pub stride: u32,
}

impl ChronosFieldView {
    const EMPTY: Self = Self {
        ptr: core::ptr::null(),
        len: 0,
        elem_size: 0,
        elem_kind: 0,
        rows: 0,
        cols: 0,
        stride: 0,
    };
}

/// 96-byte explicit ingest metadata mirroring Wasm `chronos_set_ingest_meta`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChronosIngestMeta {
    pub struct_size: u32,
    pub flags: u32,
    pub pts_us: u64,
    pub epoch: u32,
    pub seq: u32,
    pub age_ms: f32,
    pub src_w: u32,
    pub src_h: u32,
    pub crop_x: u32,
    pub crop_y: u32,
    pub crop_w: u32,
    pub crop_h: u32,
    pub dst_x: u32,
    pub dst_y: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub orientation: u32,
    pub scan_dir: u32,
    pub gap_mask: u32,
    pub native_1to1: u32,
    pub reserved: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<ChronosIngestMeta>() == 96);

const VERSION_STR: &[u8] = concat!("3.4.0\0").as_bytes();

#[no_mangle]
pub extern "C" fn chronos_abi_version() -> u32 {
    CHRONOS_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn chronos_version_string() -> *const c_char {
    VERSION_STR.as_ptr() as *const c_char
}

/// Layout code for `ChronosFieldKind` (`0 = FLAT`, `1 = ROWS_COLS`, `2 = PLANES`, `3 = INTERLEAVED`).
#[no_mangle]
pub extern "C" fn chronos_field_layout(kind: u32) -> u32 {
    match kind {
        6 | 9 | 21 | 23 => 2, // BOS/Hodge/Wind/Elasto planes: (stride, rows, cols)
        3 | 16 => 3,          // Continuum / Depth field: (rows, cols, stride)
        1 | 2 | 4 | 7 | 17 | 19 | 24 => 1, // (rows, cols)
        _ => 0,               // flat 1D vector
    }
}

#[no_mangle]
pub extern "C" fn chronos_engine_create() -> *mut ChronosEngineHandle {
    let eng = Box::new(CapiEngine::new());
    Box::into_raw(eng) as *mut ChronosEngineHandle
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_reset(handle: *mut ChronosEngineHandle) {
    if let Some(c) = (handle as *mut CapiEngine).as_mut() {
        c.reset();
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_destroy(handle: *mut ChronosEngineHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut CapiEngine));
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_resolution(handle: *mut ChronosEngineHandle, hd: u32) {
    if let Some(c) = (handle as *mut CapiEngine).as_mut() {
        c.eng.set_resolution(hd != 0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_resolution_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) {
    if let Some(c) = (handle as *mut CapiEngine).as_mut() {
        c.eng.set_resolution_mode(mode);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_params(
    handle: *mut ChronosEngineHandle,
    lookahead_k: u32,
    eca_adapt_rate: f32,
    fista_lambda: f32,
    eulerian_gain: f32,
    phase_offset_rad: f32,
    lab_mode: u32,
) {
    if let Some(c) = (handle as *mut CapiEngine).as_mut() {
        c.eng.set_params(
            lookahead_k,
            eca_adapt_rate,
            fista_lambda,
            eulerian_gain,
            phase_offset_rad,
            lab_mode,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_swarm_mode(handle: *mut ChronosEngineHandle, mode: u32) {
    if let Some(c) = (handle as *mut CapiEngine).as_mut() {
        c.eng.set_swarm_mode(mode);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_bos_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_bos_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_hodge_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_hodge_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_wind_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_wind_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_wind_fluid_density(
    handle: *mut ChronosEngineHandle,
    rho_kg_m3: f32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_wind_fluid_density(rho_kg_m3),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_elasto_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_elasto_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_elasto_lock_freqs(
    handle: *mut ChronosEngineHandle,
    count: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_elasto_lock_freqs(count, f0, f1, f2, f3),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_depth_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_depth_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_depth_calib(
    handle: *mut ChronosEngineHandle,
    f_px: f32,
    speed_m_s: f32,
    baseline_m: f32,
    max_disp_px: f32,
    stereo_sign: i32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c
            .eng
            .set_depth_calib(f_px, speed_m_s, baseline_m, max_disp_px, stereo_sign),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_depth_capture_stereo_left(
    handle: *mut ChronosEngineHandle,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.capture_stereo_left(),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_depth_external_meta(
    handle: *mut ChronosEngineHandle,
    w: u32,
    h: u32,
    f_px: f32,
    provenance: u32,
    seq: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_depth_external_meta(w, h, f_px, provenance, seq),
        None => 0,
    }
}

/// Zero-copy writable pointer to the external depth-plane staging buffer (`*out_len = 8192` floats).
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_depth_ingest_mut_ptr(
    handle: *mut ChronosEngineHandle,
    out_len: *mut usize,
) -> *mut f32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return core::ptr::null_mut();
    };
    let buf = c.eng.depth_ingest_mut();
    if !out_len.is_null() {
        *out_len = buf.len();
    }
    buf.as_mut_ptr()
}

/// Zero-copy writable pointer to the 4,096-sample mono `f32` audio staging buffer.
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_audio_staging_mut_ptr(
    handle: *mut ChronosEngineHandle,
    out_len: *mut usize,
) -> *mut f32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return core::ptr::null_mut();
    };
    let buf = c.eng.audio_staging_mut();
    if !out_len.is_null() {
        *out_len = buf.len();
    }
    buf.as_mut_ptr()
}

/// Commit `num_samples` (`1..=4096`) staged in `audio_staging_mut_ptr`. Returns 1 on accept, 0 on reject.
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_push_audio_block(
    handle: *mut ChronosEngineHandle,
    num_samples: u32,
    t_end_us: u64,
    sample_rate_hz: f32,
    est_skew_us: f32,
) -> u32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return 0;
    };
    let ok = c
        .eng
        .push_audio_block(num_samples, t_end_us, sample_rate_hz, est_skew_us);
    if c.auditor.active {
        let n = (num_samples as usize).min(4096);
        let samples: Vec<f32> = c.eng.audio_staging_mut()[..n].to_vec();
        c.auditor
            .record_audio_block(&samples, t_end_us, sample_rate_hz, ok != 0);
    }
    ok
}

/// Copying audio helper: splits arbitrary `num_samples` into `<= 4096`-sample blocks automatically.
/// Returns 0 on success, -1 null handle, -2 invalid args or rejected block.
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_push_audio(
    handle: *mut ChronosEngineHandle,
    mono: *const f32,
    num_samples: usize,
    t_end_us: u64,
    sample_rate_hz: f32,
) -> i32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return -1;
    };
    if mono.is_null()
        || num_samples == 0
        || !sample_rate_hz.is_finite()
        || sample_rate_hz < 4000.0
    {
        return -2;
    }
    let slice = core::slice::from_raw_parts(mono, num_samples);
    let total_dur_us = ((num_samples as f64) * 1_000_000.0 / (sample_rate_hz as f64)).round() as u64;
    let t_start_us = t_end_us.saturating_sub(total_dur_us);
    let mut off = 0usize;
    while off < num_samples {
        let chunk = (num_samples - off).min(4096);
        let end_idx = off + chunk;
        let blk_end_us = if end_idx == num_samples {
            t_end_us
        } else {
            t_start_us.saturating_add(
                ((end_idx as f64) * 1_000_000.0 / (sample_rate_hz as f64)).round() as u64,
            )
        };
        c.eng.audio_staging_mut()[..chunk].copy_from_slice(&slice[off..end_idx]);
        let ok = c
            .eng
            .push_audio_block(chunk as u32, blk_end_us, sample_rate_hz, 0.0);
        if c.auditor.active {
            c.auditor
                .record_audio_block(&slice[off..end_idx], blk_end_us, sample_rate_hz, ok != 0);
        }
        if ok == 0 {
            return -2;
        }
        off = end_idx;
    }
    0
}

/// Stage explicit ingest metadata (`ChronosIngestMeta`, 96 B). Returns 0 on accept, negative on error.
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_ingest_meta(
    handle: *mut ChronosEngineHandle,
    meta: *const ChronosIngestMeta,
) -> i32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return -1;
    };
    let Some(m) = meta.as_ref() else {
        return -2;
    };
    if (m.struct_size as usize) < core::mem::size_of::<ChronosIngestMeta>() {
        return -3;
    }
    let ok = c.eng.set_ingest_meta(NativeIngestMeta {
        epoch: m.epoch,
        seq: m.seq,
        pts_us: m.pts_us,
        age_ms: m.age_ms,
        flags: m.flags,
        src_w: m.src_w,
        src_h: m.src_h,
        crop_x: m.crop_x,
        crop_y: m.crop_y,
        crop_w: m.crop_w,
        crop_h: m.crop_h,
        orientation: m.orientation,
        scan_dir: m.scan_dir,
        dst_x: m.dst_x,
        dst_y: m.dst_y,
        dst_w: m.dst_w,
        dst_h: m.dst_h,
        gap_mask: m.gap_mask,
        native_1to1: m.native_1to1,
    });
    if ok != 0 {
        c.pending_pts_us = Some(m.pts_us);
        0
    } else {
        -2
    }
}

/// Deterministic integer box/area resample + letterbox (`fit`: `0 = letterbox`, `1 = stretch`)
/// from an arbitrary-size strided RGBA8 frame directly into the engine's next ring slot.
/// Updates `inout_meta->src_w/src_h/crop_*/dst_*` when non-null. Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_ingest_rgba_strided(
    handle: *mut ChronosEngineHandle,
    rgba: *const u8,
    src_w: u32,
    src_h: u32,
    src_stride_bytes: usize,
    fit: u32,
    inout_meta: *mut ChronosIngestMeta,
) -> i32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return -1;
    };
    if rgba.is_null() || src_w == 0 || src_h == 0 || fit > 1 {
        return -2;
    }
    let sw = src_w as usize;
    let sh = src_h as usize;
    let min_stride = sw.saturating_mul(4);
    let stride = if src_stride_bytes == 0 {
        min_stride
    } else {
        src_stride_bytes
    };
    if stride < min_stride {
        return -2;
    }
    let total_src_bytes = match stride.checked_mul(sh) {
        Some(v) => v,
        None => return -2,
    };
    let src = core::slice::from_raw_parts(rgba, total_src_bytes);
    let (ew, eh) = c.eng.frame_dims();

    let (dx, dy, dw, dh) = if fit == 0 {
        // Aspect-preserving letterbox using exact integer arithmetic
        let lhs = (sw as u64) * (eh as u64);
        let rhs = (ew as u64) * (sh as u64);
        let (rw, rh) = if lhs >= rhs {
            let rh = (((sh as u64) * (ew as u64) + (sw as u64) / 2) / (sw as u64))
                .clamp(1, eh as u64) as usize;
            (ew, rh)
        } else {
            let rw = (((sw as u64) * (eh as u64) + (sh as u64) / 2) / (sh as u64))
                .clamp(1, ew as u64) as usize;
            (rw, eh)
        };
        ((ew - rw) / 2, (eh - rh) / 2, rw, rh)
    } else {
        (0, 0, ew, eh)
    };

    let dst = c.eng.ingest_rgba_mut();
    dst.fill(0);
    // Set alpha=255 across the whole canvas
    for p in dst.chunks_exact_mut(4) {
        p[3] = 255;
    }

    if sw == dw && sh == dh && dx == 0 && dy == 0 && stride == ew * 4 {
        dst.copy_from_slice(&src[..ew * eh * 4]);
    } else {
        // Deterministic integer area-box average (downscale) / bilinear-nearest (upscale)
        for oy in 0..dh {
            let sy0 = (oy * sh) / dh;
            let sy1 = (((oy + 1) * sh + dh - 1) / dh).clamp(sy0 + 1, sh);
            let dst_row = (dy + oy) * ew;
            for ox in 0..dw {
                let sx0 = (ox * sw) / dw;
                let sx1 = (((ox + 1) * sw + dw - 1) / dw).clamp(sx0 + 1, sw);
                let mut sr = 0u64;
                let mut sg = 0u64;
                let mut sb = 0u64;
                let mut cnt = 0u64;
                for sy in sy0..sy1 {
                    let row_off = sy * stride;
                    for sx in sx0..sx1 {
                        let off = row_off + sx * 4;
                        sr += src[off] as u64;
                        sg += src[off + 1] as u64;
                        sb += src[off + 2] as u64;
                        cnt += 1;
                    }
                }
                let doff = (dst_row + dx + ox) * 4;
                dst[doff] = ((sr + cnt / 2) / cnt) as u8;
                dst[doff + 1] = ((sg + cnt / 2) / cnt) as u8;
                dst[doff + 2] = ((sb + cnt / 2) / cnt) as u8;
                dst[doff + 3] = 255;
            }
        }
    }

    if let Some(m) = inout_meta.as_mut() {
        m.src_w = src_w;
        m.src_h = src_h;
        m.crop_x = 0;
        m.crop_y = 0;
        m.crop_w = src_w;
        m.crop_h = src_h;
        m.dst_x = dx as u32;
        m.dst_y = dy as u32;
        m.dst_w = dw as u32;
        m.dst_h = dh as u32;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_seed_target(
    handle: *mut ChronosEngineHandle,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.seed_target(x, y, w, h),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_reset_tracks(handle: *mut ChronosEngineHandle) {
    if let Some(c) = (handle as *mut CapiEngine).as_mut() {
        c.eng.reset_tracks();
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_frame_w(handle: *const ChronosEngineHandle) -> u32 {
    match (handle as *const CapiEngine).as_ref() {
        Some(c) => c.eng.frame_dims().0 as u32,
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_frame_h(handle: *const ChronosEngineHandle) -> u32 {
    match (handle as *const CapiEngine).as_ref() {
        Some(c) => c.eng.frame_dims().1 as u32,
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_ingest_rgba(
    handle: *mut ChronosEngineHandle,
    rgba: *const u8,
    len: usize,
) -> i32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return -1;
    };
    if rgba.is_null() {
        return -2;
    }
    let slice = core::slice::from_raw_parts(rgba, len);
    match c.eng.write_rgba(slice) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_step(
    handle: *mut ChronosEngineHandle,
    dt_sec: f32,
) -> u32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return 0;
    };
    let sig = if c.auditor.active {
        let (w, h) = c.eng.frame_dims();
        c.auditor.sample_pre_step_luma_sig(c.eng.ingest_rgba_mut(), w, h)
    } else {
        [0.0; 64]
    };
    let slot = c.eng.step(dt_sec);
    let pts = c.pending_pts_us.take();
    if c.auditor.active {
        c.last_luma_sig = sig;
        c.auditor.observe_step(&c.eng, dt_sec, pts, sig);
    }
    slot
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_step_with_pts(
    handle: *mut ChronosEngineHandle,
    pts_sec: f64,
    dt_sec: f32,
) -> u32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return 0;
    };
    let sig = if c.auditor.active {
        let (w, h) = c.eng.frame_dims();
        c.auditor.sample_pre_step_luma_sig(c.eng.ingest_rgba_mut(), w, h)
    } else {
        [0.0; 64]
    };
    let slot = c.eng.step_with_pts(pts_sec, dt_sec);
    c.pending_pts_us = None;
    if c.auditor.active {
        let pts_us = if pts_sec.is_finite() && pts_sec >= 0.0 {
            Some((pts_sec * 1_000_000.0 + 0.5) as u64)
        } else {
            None
        };
        c.last_luma_sig = sig;
        c.auditor.observe_step(&c.eng, dt_sec, pts_us, sig);
    }
    slot
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_telemetry_ptr(
    handle: *const ChronosEngineHandle,
) -> *const f32 {
    match (handle as *const CapiEngine).as_ref() {
        Some(c) => c.eng.telemetry().as_ptr(),
        None => core::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn chronos_engine_telemetry_len() -> usize {
    chronos_spectral::TELEMETRY_LEN
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_bos_gauge_roi(
    handle: *mut ChronosEngineHandle,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_bos_gauge_roi(x0, y0, x1, y1),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_micro_metrology_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c.eng.set_micro_metrology_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_rs_calib(
    handle: *mut ChronosEngineHandle,
    mode: u32,
    dt_row_us: f32,
    exposure_us: f32,
    audio_rate_hz: f32,
    max_skew_us: f32,
) -> u32 {
    match (handle as *mut CapiEngine).as_mut() {
        Some(c) => c
            .eng
            .set_rs_calib(mode, dt_row_us, exposure_us, audio_rate_hz, max_skew_us),
        None => 0,
    }
}

/// Zero-copy writable pointer to the next RGBA8 ingest ring slot (`*out_len = frame_w * frame_h * 4`).
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_ingest_rgba_mut_ptr(
    handle: *mut ChronosEngineHandle,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return core::ptr::null_mut();
    };
    let buf = c.eng.ingest_rgba_mut();
    if !out_len.is_null() {
        *out_len = buf.len();
    }
    buf.as_mut_ptr()
}

/// Zero-copy read-only field view (`kind`: 0..=24; see `ChronosFieldKind` in `chronos.h`).
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_field_view(
    handle: *const ChronosEngineHandle,
    kind: u32,
) -> ChronosFieldView {
    let Some(c) = (handle as *const CapiEngine).as_ref() else {
        return ChronosFieldView::EMPTY;
    };
    match c.eng.field_view_desc(kind) {
        Some(d) => ChronosFieldView {
            ptr: d.ptr as *const c_void,
            len: d.len,
            elem_size: d.elem_size,
            elem_kind: d.elem_kind,
            rows: d.rows,
            cols: d.cols,
            stride: d.stride,
        },
        None => ChronosFieldView::EMPTY,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_audit_begin(
    handle: *mut ChronosEngineHandle,
    cfg: *const ChronosAuditConfig,
) -> i32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return -1;
    };
    let parsed = if cfg.is_null() {
        ChronosAuditConfig::default()
    } else {
        let m = &*cfg;
        if (m.struct_size as usize) < core::mem::size_of::<ChronosAuditConfig>() {
            return -3;
        }
        *m
    };
    c.auditor.begin(parsed);
    0
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_audit_end(handle: *mut ChronosEngineHandle) {
    if let Some(c) = (handle as *mut CapiEngine).as_mut() {
        c.auditor.end();
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_audit_finalize(handle: *mut ChronosEngineHandle) -> i32 {
    let Some(c) = (handle as *mut CapiEngine).as_mut() else {
        return -1;
    };
    let _ = c.auditor.finalize();
    0
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_audit_summary(
    handle: *const ChronosEngineHandle,
    out: *mut ChronosAuditSummary,
) -> i32 {
    let Some(c) = (handle as *const CapiEngine).as_ref() else {
        return -1;
    };
    let Some(dst) = out.as_mut() else {
        return -2;
    };
    *dst = c.auditor.summary();
    0
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_audit_spans(
    handle: *const ChronosEngineHandle,
    start: usize,
    out: *mut ChronosAuditSpan,
    cap: usize,
) -> usize {
    let Some(c) = (handle as *const CapiEngine).as_ref() else {
        return 0;
    };
    let spans = c.auditor.spans();
    if start >= spans.len() || out.is_null() || cap == 0 {
        return 0;
    }
    let count = (spans.len() - start).min(cap);
    let dst = core::slice::from_raw_parts_mut(out, count);
    dst.copy_from_slice(&spans[start..start + count]);
    count
}
