//! Multi-instance C ABI (`ChronosEngine*`) for the native CHRONOS-SPECTRAL kernel.
//!
//! Kept in its own crate (`native/capi/`) so the core `#![no_std]` Wasm module stays zero-import
//! and `native/replay/kernel.rs`'s export census over `src/*.rs` stays untouched.

use chronos_spectral::native::ChronosEngine;
use core::ffi::c_void;

/// Opaque handle to an independent heap-allocated `ChronosEngine` (~23.3 MB `Arena`).
pub type ChronosEngineHandle = c_void;

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

#[no_mangle]
pub extern "C" fn chronos_engine_create() -> *mut ChronosEngineHandle {
    let eng = Box::new(ChronosEngine::new());
    Box::into_raw(eng) as *mut ChronosEngineHandle
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_destroy(handle: *mut ChronosEngineHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut ChronosEngine));
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_resolution(handle: *mut ChronosEngineHandle, hd: u32) {
    if let Some(eng) = (handle as *mut ChronosEngine).as_mut() {
        eng.set_resolution(hd != 0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_resolution_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) {
    if let Some(eng) = (handle as *mut ChronosEngine).as_mut() {
        eng.set_resolution_mode(mode);
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
    if let Some(eng) = (handle as *mut ChronosEngine).as_mut() {
        eng.set_params(
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
    if let Some(eng) = (handle as *mut ChronosEngine).as_mut() {
        eng.set_swarm_mode(mode);
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_bos_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_bos_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_hodge_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_hodge_mode(mode),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_depth_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_depth_mode(mode),
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
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_depth_calib(f_px, speed_m_s, baseline_m, max_disp_px, stereo_sign),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_depth_capture_stereo_left(
    handle: *mut ChronosEngineHandle,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.capture_stereo_left(),
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
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_depth_external_meta(w, h, f_px, provenance, seq),
        None => 0,
    }
}

/// Zero-copy writable pointer to the external depth-plane staging buffer (`*out_len = 8192` floats).
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_depth_ingest_mut_ptr(
    handle: *mut ChronosEngineHandle,
    out_len: *mut usize,
) -> *mut f32 {
    let Some(eng) = (handle as *mut ChronosEngine).as_mut() else {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return core::ptr::null_mut();
    };
    let buf = eng.depth_ingest_mut();
    if !out_len.is_null() {
        *out_len = buf.len();
    }
    buf.as_mut_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_seed_target(
    handle: *mut ChronosEngineHandle,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.seed_target(x, y, w, h),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_reset_tracks(handle: *mut ChronosEngineHandle) {
    if let Some(eng) = (handle as *mut ChronosEngine).as_mut() {
        eng.reset_tracks();
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_frame_w(handle: *const ChronosEngineHandle) -> u32 {
    match (handle as *const ChronosEngine).as_ref() {
        Some(eng) => eng.frame_dims().0 as u32,
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_frame_h(handle: *const ChronosEngineHandle) -> u32 {
    match (handle as *const ChronosEngine).as_ref() {
        Some(eng) => eng.frame_dims().1 as u32,
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_ingest_rgba(
    handle: *mut ChronosEngineHandle,
    rgba: *const u8,
    len: usize,
) -> i32 {
    let Some(eng) = (handle as *mut ChronosEngine).as_mut() else {
        return -1;
    };
    if rgba.is_null() {
        return -2;
    }
    let slice = core::slice::from_raw_parts(rgba, len);
    match eng.write_rgba(slice) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_step(
    handle: *mut ChronosEngineHandle,
    dt_sec: f32,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.step(dt_sec),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_step_with_pts(
    handle: *mut ChronosEngineHandle,
    pts_sec: f64,
    dt_sec: f32,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.step_with_pts(pts_sec, dt_sec),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_telemetry_ptr(
    handle: *const ChronosEngineHandle,
) -> *const f32 {
    match (handle as *const ChronosEngine).as_ref() {
        Some(eng) => eng.telemetry().as_ptr(),
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
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_bos_gauge_roi(x0, y0, x1, y1),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn chronos_engine_set_micro_metrology_mode(
    handle: *mut ChronosEngineHandle,
    mode: u32,
) -> u32 {
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_micro_metrology_mode(mode),
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
    match (handle as *mut ChronosEngine).as_mut() {
        Some(eng) => eng.set_rs_calib(mode, dt_row_us, exposure_us, audio_rate_hz, max_skew_us),
        None => 0,
    }
}

/// Zero-copy writable pointer to the next RGBA8 ingest ring slot (`*out_len = frame_w * frame_h * 4`).
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_ingest_rgba_mut_ptr(
    handle: *mut ChronosEngineHandle,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(eng) = (handle as *mut ChronosEngine).as_mut() else {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return core::ptr::null_mut();
    };
    let buf = eng.ingest_rgba_mut();
    if !out_len.is_null() {
        *out_len = buf.len();
    }
    buf.as_mut_ptr()
}

/// Zero-copy read-only field view (`kind`: 0..=19; see `ChronosFieldKind` in `chronos.h`).
#[no_mangle]
pub unsafe extern "C" fn chronos_engine_field_view(
    handle: *const ChronosEngineHandle,
    kind: u32,
) -> ChronosFieldView {
    let Some(eng) = (handle as *const ChronosEngine).as_ref() else {
        return ChronosFieldView::EMPTY;
    };
    match eng.field_view_desc(kind) {
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
