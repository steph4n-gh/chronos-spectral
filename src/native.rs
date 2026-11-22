//! Native-only kernel support for host tools (the CSTRACE replay driver in `native/replay/`).
//!
//! Compiled only off wasm32 (declared at the very end of `lib.rs`, so no line of the shipped
//! Wasm's sources moves). It gives a host tool the two things the C ABI cannot:
//!
//! * [`reset_to_pristine`]: return the global arena to the state of a fresh Wasm instance (the
//!   trace `RESET` record). `ARENA` is initialised to all-zero bytes and is the kernel's only
//!   mutable static, so zeroing it is exactly that state.
//! * [`regions`]: the native build's own layout of every region a getter exports: the backing
//!   field's address and its byte length taken from the native type (`size_of_val`), so a replay
//!   tool can check a trace's region table against this build before touching memory.

use crate::{Arena, ARENA, RING_LEN};
use core::mem::{size_of, size_of_val};
use core::ptr::addr_of_mut;

/// Returns the kernel to its fresh-instance state: every byte of the arena zero, as in the
/// static initialiser `ARENA = zeroed()`. No `chronos_init` is implied (a fresh Wasm instance
/// has not run it either).
///
/// # Safety
/// The kernel is a global singleton: no other thread may be inside a kernel function, and no
/// reference into the arena may be live.
pub unsafe fn reset_to_pristine() {
    core::ptr::write_bytes(addr_of_mut!(ARENA) as *mut u8, 0, size_of::<Arena>());
}

/// Byte size of the arena in this build (differs between wasm32 and 64-bit hosts because the
/// arena contains `usize` fields).
pub fn arena_bytes() -> usize {
    size_of::<Arena>()
}

/// Address range of the arena in this build.
pub fn arena_range() -> (usize, usize) {
    // Only the address is taken; no reference into the arena is created.
    let lo = addr_of_mut!(ARENA) as usize;
    (lo, lo + size_of::<Arena>())
}

/// How a region's getter is called and which ring slots it may point at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// A fixed field; the getter must return exactly `field_addr`.
    Fixed,
    /// An indexed fixed field (`chronos_get_fovea_rgba_ptr(arg)`): getter takes `arg` and returns `field_addr`.
    Indexed(u32),
    /// `chronos_get_ring_slot_ptr(arg)`: ring slot `arg`.
    RingSlot(u32),
    /// A moving view onto one of the ring slots (`ingest_rgba`, `display_rgba`).
    RingAlias,
    /// A fixed view aliasing another digested field (e.g. `elasto_planes -> bos_planes`).
    FixedAlias,
}

/// One exported region, as laid out in this build.
#[derive(Clone, Copy, Debug)]
pub struct NativeRegion {
    /// Region name, as in the trace region table (`scripts/trace_lib.mjs`, `REGION_SPECS`).
    pub name: &'static str,
    /// Getter export that returns the region pointer.
    pub getter: &'static str,
    pub kind: RegionKind,
    /// Element code of the getter's pointee type: 1 = u8, 2 = u32, 3 = f32.
    pub elem: u8,
    /// Whether the getter returns `*mut` (host-writable).
    pub host_writable: bool,
    /// Start address of the backing field (the ring slot for ring regions; ring slot 0 for
    /// aliases).
    pub field_addr: usize,
    /// Byte length of the backing field (one ring slot for ring regions and aliases).
    pub byte_len: usize,
}

const U8: u8 = 1;
const U32: u8 = 2;
const F32: u8 = 3;

/// Every region exported by a pointer getter, in the order of `REGION_SPECS` in
/// `scripts/trace_lib.mjs` (ring slots expanded). Addresses and lengths come from the native
/// types; nothing here is hard-coded.
pub fn regions() -> [NativeRegion; RING_LEN + 48] {
    // SAFETY: only addresses and sizes of fields are taken; no reads or writes.
    let a: &Arena = unsafe { &*addr_of_mut!(ARENA) };
    macro_rules! fixed {
        ($name:literal, $getter:literal, $elem:expr, $mut:expr, $field:expr) => {
            NativeRegion {
                name: $name,
                getter: $getter,
                kind: RegionKind::Fixed,
                elem: $elem,
                host_writable: $mut,
                field_addr: $field.as_ptr() as usize,
                byte_len: size_of_val(&$field),
            }
        };
    }
    let slot = |k: usize| NativeRegion {
        name: RING_SLOT_NAMES[k],
        getter: "chronos_get_ring_slot_ptr",
        kind: RegionKind::RingSlot(k as u32),
        elem: U8,
        host_writable: true,
        field_addr: a.rgba_ring[k].as_ptr() as usize,
        byte_len: size_of_val(&a.rgba_ring[k]),
    };
    let alias = |name: &'static str, getter: &'static str, host_writable: bool| NativeRegion {
        name,
        getter,
        kind: RegionKind::RingAlias,
        elem: U8,
        host_writable,
        field_addr: a.rgba_ring[0].as_ptr() as usize,
        byte_len: size_of_val(&a.rgba_ring[0]),
    };
    [
        slot(0),
        slot(1),
        slot(2),
        slot(3),
        slot(4),
        slot(5),
        slot(6),
        slot(7),
        slot(8),
        slot(9),
        slot(10),
        slot(11),
        alias("ingest_rgba", "chronos_get_ingest_rgba_ptr", true),
        alias("display_rgba", "chronos_get_display_rgba_ptr", false),
        fixed!("eca_res", "chronos_get_eca_res_ptr", F32, false, a.eca_res),
        fixed!(
            "pip_rgba",
            "chronos_get_pip_rgba_ptr",
            U8,
            false,
            a.pip_rgba
        ),
        fixed!(
            "all_pip_rgba",
            "chronos_get_all_pip_rgba_ptr",
            U8,
            false,
            a.all_pip_rgba
        ),
        fixed!(
            "slot_meta",
            "chronos_get_slot_meta_ptr",
            F32,
            false,
            a.slot_meta
        ),
        fixed!(
            "laplacian",
            "chronos_get_laplacian_ptr",
            F32,
            true,
            a.laplacian
        ),
        fixed!(
            "fiedler",
            "chronos_get_fiedler_ptr",
            F32,
            true,
            a.fiedler_vec
        ),
        fixed!(
            "telemetry",
            "chronos_get_telemetry_ptr",
            F32,
            false,
            a.telemetry
        ),
        fixed!(
            "micro_tracks",
            "chronos_get_micro_tracks_ptr",
            F32,
            false,
            a.micro_swarm.export_micro
        ),
        fixed!(
            "continuum_field",
            "chronos_get_continuum_field_ptr",
            F32,
            false,
            a.continuum.export_field
        ),
        fixed!(
            "rigid_bodies",
            "chronos_get_rigid_bodies_ptr",
            F32,
            false,
            a.continuum.export_bodies
        ),
        fixed!(
            "swarm_telemetry",
            "chronos_get_swarm_telemetry_ptr",
            F32,
            false,
            a.micro_swarm.swarm_telem
        ),
        fixed!(
            "modal_field",
            "chronos_get_modal_field_ptr",
            F32,
            false,
            a.continuum.export_modal_field
        ),
        fixed!(
            "ring_meta",
            "chronos_get_ring_meta_ptr",
            U32,
            false,
            a.ring_meta
        ),
        fixed!(
            "row_luma",
            "chronos_get_row_luma_ptr",
            F32,
            false,
            a.row_luma_means
        ),
        fixed!(
            "wave1_telemetry",
            "chronos_get_wave1_telemetry_ptr",
            U32,
            false,
            a.wave1_telem
        ),
        fixed!(
            "bos_telemetry",
            "chronos_get_bos_telemetry_ptr",
            U32,
            false,
            a.bos.telem
        ),
        fixed!(
            "bos_planes",
            "chronos_get_bos_planes_ptr",
            F32,
            false,
            a.bos.planes
        ),
        fixed!(
            "bos_valid",
            "chronos_get_bos_valid_ptr",
            U8,
            false,
            a.bos.valid_mask
        ),
        fixed!(
            "bos_rgba",
            "chronos_get_bos_rgba_ptr",
            U8,
            false,
            a.bos.rgba
        ),
        fixed!(
            "micro_cov",
            "chronos_get_micro_cov_ptr",
            F32,
            false,
            a.micro_swarm.metrology.export_micro_cov
        ),
        fixed!(
            "micro_residual",
            "chronos_get_micro_residual_ptr",
            F32,
            false,
            a.micro_swarm.metrology.export_micro_residual
        ),
        fixed!(
            "micro_loom",
            "chronos_get_micro_loom_ptr",
            F32,
            false,
            a.micro_swarm.metrology.export_micro_loom_d2
        ),
        fixed!(
            "micro_quality",
            "chronos_get_micro_quality_ptr",
            U32,
            false,
            a.micro_swarm.metrology.export_micro_quality
        ),
        fixed!(
            "hero_candidates",
            "chronos_get_hero_candidates_ptr",
            U32,
            false,
            a.micro_swarm.metrology.export_hero_candidates
        ),
        fixed!(
            "wave2_telemetry",
            "chronos_get_wave2_telemetry_ptr",
            U32,
            false,
            a.wave2_telem
        ),
        fixed!(
            "audio_staging",
            "chronos_get_audio_staging_ptr",
            F32,
            true,
            a.vibro.audio_staging
        ),
        fixed!(
            "vibro_telemetry",
            "chronos_get_vibro_telemetry_ptr",
            U32,
            false,
            a.vibro.telem
        ),
        fixed!(
            "vibro_planes",
            "chronos_get_vibro_planes_ptr",
            F32,
            false,
            a.vibro.planes
        ),
        fixed!(
            "hodge_telemetry",
            "chronos_get_hodge_telemetry_ptr",
            U32,
            false,
            a.hodge.telem
        ),
        fixed!(
            "hodge_planes",
            "chronos_get_hodge_planes_ptr",
            F32,
            false,
            a.hodge.planes
        ),
        fixed!(
            "hodge_inject",
            "chronos_get_hodge_inject_ptr",
            F32,
            true,
            a.hodge.inject_buf
        ),
        fixed!(
            "conservation_telemetry",
            "chronos_get_conservation_telemetry_ptr",
            F32,
            false,
            a.collision_state.conservation.telem
        ),
        fixed!(
            "conservation_input",
            "chronos_get_conservation_input_ptr",
            F32,
            true,
            a.collision_state.conservation.input
        ),
        NativeRegion {
            name: "fovea_rgba[0]",
            getter: "chronos_get_fovea_rgba_ptr",
            kind: RegionKind::Indexed(0),
            elem: U8,
            host_writable: true,
            field_addr: a.fovea.rgba.as_ptr() as usize,
            byte_len: size_of_val(&a.fovea.rgba),
        },
        fixed!(
            "fovea_request",
            "chronos_get_fovea_request_ptr",
            U32,
            false,
            a.fovea.request_meta
        ),
        fixed!(
            "depth_telemetry",
            "chronos_get_depth_telemetry_ptr",
            U32,
            false,
            a.depth.telem
        ),
        fixed!(
            "depth_field",
            "chronos_get_depth_field_ptr",
            F32,
            false,
            a.depth.export_field
        ),
        fixed!(
            "depth_tracks",
            "chronos_get_depth_tracks_ptr",
            F32,
            false,
            a.depth.export_track_depth
        ),
        fixed!(
            "depth_track_flags",
            "chronos_get_depth_track_flags_ptr",
            U8,
            false,
            a.depth.export_track_flags
        ),
        fixed!(
            "depth_hero",
            "chronos_get_depth_hero_ptr",
            F32,
            false,
            a.depth.export_hero_depth
        ),
        fixed!(
            "depth_ingest",
            "chronos_get_depth_ingest_ptr",
            F32,
            true,
            a.depth.ext_depth
        ),
        fixed!(
            "wind_telemetry",
            "chronos_get_wind_telemetry_ptr",
            U32,
            false,
            a.wind.telem
        ),
        fixed!(
            "wind_planes",
            "chronos_get_wind_planes_ptr",
            F32,
            false,
            a.wind.planes
        ),
        fixed!(
            "elasto_telemetry",
            "chronos_get_elasto_telemetry_ptr",
            U32,
            false,
            a.elasto.telem
        ),
        NativeRegion {
            name: "elasto_planes",
            getter: "chronos_get_elasto_planes_ptr",
            kind: RegionKind::FixedAlias,
            elem: F32,
            host_writable: false,
            field_addr: a.bos.planes.as_ptr() as usize,
            byte_len: size_of_val(&a.bos.planes),
        },
        NativeRegion {
            name: "elasto_valid",
            getter: "chronos_get_elasto_valid_ptr",
            kind: RegionKind::FixedAlias,
            elem: U8,
            host_writable: false,
            field_addr: a.bos.valid_mask.as_ptr() as usize,
            byte_len: size_of_val(&a.bos.valid_mask),
        },
    ]
}

const RING_SLOT_NAMES: [&str; RING_LEN] = [
    "ring_slot[0]",
    "ring_slot[1]",
    "ring_slot[2]",
    "ring_slot[3]",
    "ring_slot[4]",
    "ring_slot[5]",
    "ring_slot[6]",
    "ring_slot[7]",
    "ring_slot[8]",
    "ring_slot[9]",
    "ring_slot[10]",
    "ring_slot[11]",
];

#[cfg(any(test, feature = "native-lib"))]
mod engine {
    extern crate alloc;

    use alloc::alloc::{alloc_zeroed, handle_alloc_error, Layout};
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use crate::{
        Arena, INGEST_FLAG_GEOM_KNOWN, INGEST_FLAG_PTS_KNOWN, MAX_TARGETS, RING_LEN,
        TELEMETRY_LEN, TELEM_KINETIC_START, TELEM_PHYSICS_START,
    };

    /// Zero-copy descriptor of a kernel output plane/buffer inside `ChronosEngine`.
    #[derive(Clone, Copy, Debug)]
    pub struct FieldViewDesc {
        pub ptr: *const u8,
        pub len: usize,
        pub elem_size: u32,
        /// `1 = u8`, `2 = u32`, `3 = f32`
        pub elem_kind: u32,
        pub rows: u32,
        pub cols: u32,
        pub stride: u32,
    }

    /// Snapshot of one active, confirmed Hero-16 track after `step()`.
    #[derive(Clone, Copy, Debug)]
    pub struct TrackSnapshot {
        pub slot: usize,
        pub track_id: u32,
        pub x: f32,
        pub y: f32,
        /// Velocity in pixels per frame (`px/f`).
        pub vx: f32,
        /// Velocity in pixels per frame (`px/f`).
        pub vy: f32,
        /// Velocity in pixels per second (`px/s`, derived from kinetic telemetry).
        pub vx_px_s: f32,
        /// Velocity in pixels per second (`px/s`, derived from kinetic telemetry).
        pub vy_px_s: f32,
        pub vis: f32,
        pub bridged: f32,
        pub regime: u32,
        pub ode_r2: f32,
    }

    /// Heap-allocated, multi-instance native CHRONOS-SPECTRAL engine (~23.3 MB `Arena`).
    ///
    /// Every instance owns its own `Arena` with zero shared mutable globals, so multiple
    /// engines can run concurrently on separate threads at independent resolutions and modes.
    pub struct ChronosEngine {
        arena: Box<Arena>,
        ingest_seq: u32,
    }

    // SAFETY: `ChronosEngine` owns its `Arena` exclusively with no shared static state.
    unsafe impl Send for ChronosEngine {}

    impl ChronosEngine {
        /// Allocate a fresh zeroed `Arena` directly on the heap and initialize it.
        pub fn new() -> Self {
            let layout = Layout::new::<Arena>();
            // SAFETY: `Arena` is valid when all-zero (same invariant as `static mut ARENA`),
            // and `alloc_zeroed` avoids placing the ~23.3 MB struct on the calling thread's stack.
            let ptr = unsafe { alloc_zeroed(layout) as *mut Arena };
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            let mut arena = unsafe { Box::from_raw(ptr) };
            unsafe {
                crate::init_on(&mut arena);
            }
            Self {
                arena,
                ingest_seq: 0,
            }
        }

        pub fn set_resolution(&mut self, hd: bool) {
            unsafe {
                crate::set_resolution_on(&mut self.arena, if hd { 1 } else { 0 });
            }
        }

        /// Set resolution mode: `0 = 320x180 SD`, `1 = 640x360 HD`, `2 = 1280x720` (requires `hires-720p` feature).
        pub fn set_resolution_mode(&mut self, mode: u32) {
            unsafe {
                crate::set_resolution_on(&mut self.arena, mode);
            }
        }

        pub fn resolution_mode(&self) -> u32 {
            self.arena.res_mode
        }

        pub fn set_params(
            &mut self,
            lookahead_k: u32,
            eca_adapt_rate: f32,
            fista_lambda: f32,
            eulerian_gain: f32,
            phase_offset_rad: f32,
            lab_mode: u32,
        ) {
            unsafe {
                crate::set_params_on(
                    &mut self.arena,
                    lookahead_k,
                    eca_adapt_rate,
                    fista_lambda,
                    eulerian_gain,
                    phase_offset_rad,
                    lab_mode,
                );
            }
        }

        pub fn set_swarm_mode(&mut self, mode: u32) {
            unsafe {
                crate::set_swarm_mode_on(&mut self.arena, mode);
            }
        }

        pub fn set_bos_mode(&mut self, mode: u32) -> u32 {
            unsafe { crate::set_bos_mode_on(&mut self.arena, mode) }
        }

        pub fn set_bos_gauge_roi(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) -> u32 {
            unsafe { crate::set_bos_gauge_roi_on(&mut self.arena, x0, y0, x1, y1) }
        }

        pub fn set_micro_metrology_mode(&mut self, mode: u32) -> u32 {
            unsafe { crate::set_micro_metrology_mode_on(&mut self.arena, mode) }
        }

        pub fn set_rs_calib(
            &mut self,
            mode: u32,
            dt_row_us: f32,
            exposure_us: f32,
            audio_rate_hz: f32,
            max_skew_us: f32,
        ) -> u32 {
            unsafe {
                crate::set_rs_calib_on(
                    &mut self.arena,
                    mode,
                    dt_row_us,
                    exposure_us,
                    audio_rate_hz,
                    max_skew_us,
                )
            }
        }

        pub fn set_hodge_mode(&mut self, mode: u32) -> u32 {
            unsafe { crate::set_hodge_mode_on(&mut self.arena, mode) }
        }

        pub fn set_wind_mode(&mut self, mode: u32) -> u32 {
            unsafe { crate::set_wind_mode_on(&mut self.arena, mode) }
        }

        pub fn set_wind_fluid_density(&mut self, rho_kg_m3: f32) -> u32 {
            unsafe { crate::set_wind_fluid_density_on(&mut self.arena, rho_kg_m3) }
        }

        pub fn set_elasto_mode(&mut self, mode: u32) -> u32 {
            unsafe { crate::set_elasto_mode_on(&mut self.arena, mode) }
        }

        pub fn set_elasto_lock_freqs(
            &mut self,
            count: u32,
            f0: f32,
            f1: f32,
            f2: f32,
            f3: f32,
        ) -> u32 {
            unsafe { crate::set_elasto_lock_freqs_on(&mut self.arena, count, f0, f1, f2, f3) }
        }

        pub fn set_depth_mode(&mut self, mode: u32) -> u32 {
            unsafe { crate::set_depth_mode_on(&mut self.arena, mode) }
        }

        pub fn get_depth_mode(&self) -> u32 {
            self.arena.depth.get_mode()
        }

        pub fn set_depth_calib(
            &mut self,
            f_px: f32,
            speed_m_s: f32,
            baseline_m: f32,
            max_disp_px: f32,
            stereo_sign: i32,
        ) -> u32 {
            unsafe {
                crate::set_depth_calib_on(
                    &mut self.arena,
                    f_px,
                    speed_m_s,
                    baseline_m,
                    max_disp_px,
                    stereo_sign,
                )
            }
        }

        pub fn capture_stereo_left(&mut self) -> u32 {
            unsafe { crate::depth_capture_stereo_left_on(&mut self.arena) }
        }

        pub fn set_depth_external_meta(
            &mut self,
            w: u32,
            h: u32,
            f_px: f32,
            provenance: u32,
            seq: u32,
        ) -> u32 {
            unsafe {
                crate::set_depth_external_meta_on(&mut self.arena, w, h, f_px, provenance, seq)
            }
        }

        /// Zero-copy writable slice to the external depth-plane staging buffer (`8,192` floats, <= `128x64`).
        pub fn depth_ingest_mut(&mut self) -> &mut [f32] {
            &mut self.arena.depth.ext_depth
        }

        pub fn seed_target(&mut self, x: f32, y: f32, w: f32, h: f32) -> u32 {
            unsafe { crate::seed_target_on(&mut self.arena, x, y, w, h) }
        }

        pub fn reset_tracks(&mut self) {
            unsafe {
                crate::reset_tracks_on(&mut self.arena);
            }
        }

        pub fn frame_dims(&self) -> (usize, usize) {
            let (w, h, ..) = crate::mode_dims(self.arena.res_mode);
            (w, h)
        }

        /// Zero-copy writable slice to the next RGBA8 ingest ring slot (`frame_w * frame_h * 4` bytes).
        pub fn ingest_rgba_mut(&mut self) -> &mut [u8] {
            let (w, h) = self.frame_dims();
            let expected = w * h * 4;
            let slot = (self.arena.ring_cursor % (RING_LEN as u64)) as usize;
            &mut self.arena.rgba_ring[slot][..expected]
        }

        pub fn write_rgba(&mut self, rgba: &[u8]) -> Result<(), &'static str> {
            let dst = self.ingest_rgba_mut();
            if rgba.len() != dst.len() {
                return Err("rgba length does not match active frame dimensions");
            }
            dst.copy_from_slice(rgba);
            Ok(())
        }

        pub fn step(&mut self, dt_sec: f32) -> u32 {
            unsafe { crate::tick_on(&mut self.arena, dt_sec) }
        }

        pub fn step_with_pts(&mut self, pts_sec: f64, dt_sec: f32) -> u32 {
            let (w, h) = self.frame_dims();
            let ew = w as u32;
            let eh = h as u32;
            self.ingest_seq = self.ingest_seq.wrapping_add(1).max(1);
            let t_us = if pts_sec.is_finite() && pts_sec >= 0.0 {
                (pts_sec * 1_000_000.0 + 0.5) as u64
            } else {
                0u64
            };
            let t_lo = t_us as u32;
            let t_hi = (t_us >> 32) as u32;
            unsafe {
                let _ = crate::set_ingest_meta_on(
                    &mut self.arena,
                    1,
                    self.ingest_seq,
                    t_lo,
                    t_hi,
                    -1.0,
                    INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_GEOM_KNOWN,
                    ew,
                    eh,
                    0,
                    0,
                    ew,
                    eh,
                    0,
                    u32::MAX,
                    0,
                    0,
                    ew,
                    eh,
                    0,
                    1,
                );
                crate::tick_on(&mut self.arena, dt_sec)
            }
        }

        pub fn telemetry(&self) -> &[f32; TELEMETRY_LEN] {
            &self.arena.telemetry
        }

        pub fn swarm_telemetry(&self) -> &[f32; 64] {
            &self.arena.micro_swarm.swarm_telem
        }

        pub fn eca_res(&self) -> &[f32] {
            let (w, h) = self.frame_dims();
            &self.arena.eca_res[..w * h]
        }

        pub fn micro_tracks(&self) -> &[f32] {
            &self.arena.micro_swarm.export_micro
        }

        pub fn continuum_field(&self) -> &[f32] {
            &self.arena.continuum.export_field
        }

        pub fn rigid_bodies(&self) -> &[f32] {
            &self.arena.continuum.export_bodies
        }

        pub fn bos_planes(&self) -> &[f32] {
            &self.arena.bos.planes
        }

        pub fn bos_valid(&self) -> &[u8] {
            &self.arena.bos.valid_mask
        }

        pub fn bos_telemetry(&self) -> &[u32] {
            &self.arena.bos.telem
        }

        pub fn hodge_planes(&self) -> &[f32] {
            &self.arena.hodge.planes
        }

        pub fn hodge_telemetry(&self) -> &[u32] {
            &self.arena.hodge.telem
        }

        pub fn vibro_planes(&self) -> &[f32] {
            &self.arena.vibro.planes
        }

        pub fn vibro_telemetry(&self) -> &[u32] {
            &self.arena.vibro.telem
        }

        pub fn wave1_telemetry(&self) -> &[u32] {
            &self.arena.wave1_telem
        }

        pub fn wave2_telemetry(&self) -> &[u32] {
            &self.arena.wave2_telem
        }

        pub fn depth_telemetry(&self) -> &[u32] {
            &self.arena.depth.telem
        }

        pub fn depth_field(&self) -> &[f32] {
            &self.arena.depth.export_field
        }

        pub fn depth_tracks(&self) -> &[f32] {
            &self.arena.depth.export_track_depth
        }

        pub fn depth_track_flags(&self) -> &[u8] {
            &self.arena.depth.export_track_flags
        }

        pub fn depth_hero(&self) -> &[f32] {
            &self.arena.depth.export_hero_depth
        }

        pub fn wind_telemetry(&self) -> &[u32] {
            &self.arena.wind.telem
        }

        pub fn wind_planes(&self) -> &[f32] {
            &self.arena.wind.planes
        }

        pub fn elasto_telemetry(&self) -> &[u32] {
            &self.arena.elasto.telem
        }

        pub fn elasto_planes(&self) -> &[f32] {
            if self.arena.elasto.get_mode() == crate::elastography::ELASTO_MODE_ACOUSTIC_BOS {
                &self.arena.bos.planes
            } else {
                &self.arena.bos.planes[..6 * crate::elastography::ELASTO_V]
            }
        }

        pub fn elasto_valid(&self) -> &[u8] {
            if self.arena.elasto.get_mode() == crate::elastography::ELASTO_MODE_ACOUSTIC_BOS {
                &self.arena.bos.valid_mask
            } else {
                &self.arena.bos.valid_mask[..crate::elastography::ELASTO_V]
            }
        }

        /// Zero-copy field descriptor for C/Python bindings (`kind`: 0..=24).
        pub fn field_view_desc(&self, kind: u32) -> Option<FieldViewDesc> {
            let (fw, fh) = self.frame_dims();
            let desc = match kind {
                0 => FieldViewDesc {
                    ptr: self.arena.telemetry.as_ptr() as *const u8,
                    len: self.arena.telemetry.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: self.arena.telemetry.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                1 => FieldViewDesc {
                    ptr: self.arena.eca_res.as_ptr() as *const u8,
                    len: fw * fh,
                    elem_size: 4,
                    elem_kind: 3,
                    rows: fh as u32,
                    cols: fw as u32,
                    stride: 1,
                },
                2 => FieldViewDesc {
                    ptr: self.arena.micro_swarm.export_micro.as_ptr() as *const u8,
                    len: self.arena.micro_swarm.export_micro.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::soa_tracks::MAX_MICRO as u32,
                    cols: crate::soa_tracks::MICRO_EXPORT_STRIDE as u32,
                    stride: crate::soa_tracks::MICRO_EXPORT_STRIDE as u32,
                },
                3 => FieldViewDesc {
                    ptr: self.arena.continuum.export_field.as_ptr() as *const u8,
                    len: self.arena.continuum.export_field.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::continuum_physics::FIELD_GRID_H as u32,
                    cols: crate::continuum_physics::FIELD_GRID_W as u32,
                    stride: crate::continuum_physics::FIELD_EXPORT_STRIDE as u32,
                },
                4 => FieldViewDesc {
                    ptr: self.arena.continuum.export_bodies.as_ptr() as *const u8,
                    len: self.arena.continuum.export_bodies.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::continuum_physics::MAX_RIGID_BODIES as u32,
                    cols: crate::continuum_physics::BODY_EXPORT_STRIDE as u32,
                    stride: crate::continuum_physics::BODY_EXPORT_STRIDE as u32,
                },
                5 => FieldViewDesc {
                    ptr: self.arena.micro_swarm.swarm_telem.as_ptr() as *const u8,
                    len: self.arena.micro_swarm.swarm_telem.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: self.arena.micro_swarm.swarm_telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                6 => FieldViewDesc {
                    ptr: self.arena.bos.planes.as_ptr() as *const u8,
                    len: self.arena.bos.planes.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::bos::BOS_GRID_H as u32,
                    cols: crate::bos::BOS_GRID_W as u32,
                    stride: 6,
                },
                7 => FieldViewDesc {
                    ptr: self.arena.bos.valid_mask.as_ptr(),
                    len: self.arena.bos.valid_mask.len(),
                    elem_size: 1,
                    elem_kind: 1,
                    rows: crate::bos::BOS_GRID_H as u32,
                    cols: crate::bos::BOS_GRID_W as u32,
                    stride: 1,
                },
                8 => FieldViewDesc {
                    ptr: self.arena.bos.telem.as_ptr() as *const u8,
                    len: self.arena.bos.telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.bos.telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                9 => FieldViewDesc {
                    ptr: self.arena.hodge.planes.as_ptr() as *const u8,
                    len: self.arena.hodge.planes.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::hodge::HODGE_H as u32,
                    cols: crate::hodge::HODGE_W as u32,
                    stride: 7,
                },
                10 => FieldViewDesc {
                    ptr: self.arena.hodge.telem.as_ptr() as *const u8,
                    len: self.arena.hodge.telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.hodge.telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                11 => FieldViewDesc {
                    ptr: self.arena.vibro.planes.as_ptr() as *const u8,
                    len: self.arena.vibro.planes.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: self.arena.vibro.planes.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                12 => FieldViewDesc {
                    ptr: self.arena.vibro.telem.as_ptr() as *const u8,
                    len: self.arena.vibro.telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.vibro.telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                13 => FieldViewDesc {
                    ptr: self.arena.wave1_telem.as_ptr() as *const u8,
                    len: self.arena.wave1_telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.wave1_telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                14 => FieldViewDesc {
                    ptr: self.arena.wave2_telem.as_ptr() as *const u8,
                    len: self.arena.wave2_telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.wave2_telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                15 => FieldViewDesc {
                    ptr: self.arena.depth.telem.as_ptr() as *const u8,
                    len: self.arena.depth.telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.depth.telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                16 => FieldViewDesc {
                    ptr: self.arena.depth.export_field.as_ptr() as *const u8,
                    len: self.arena.depth.export_field.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::depth::DEPTH_GRID_H as u32,
                    cols: crate::depth::DEPTH_GRID_W as u32,
                    stride: crate::depth::DEPTH_FIELD_STRIDE as u32,
                },
                17 => FieldViewDesc {
                    ptr: self.arena.depth.export_track_depth.as_ptr() as *const u8,
                    len: self.arena.depth.export_track_depth.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::soa_tracks::MAX_MICRO as u32,
                    cols: crate::depth::DEPTH_TRACK_STRIDE as u32,
                    stride: crate::depth::DEPTH_TRACK_STRIDE as u32,
                },
                18 => FieldViewDesc {
                    ptr: self.arena.depth.export_track_flags.as_ptr(),
                    len: self.arena.depth.export_track_flags.len(),
                    elem_size: 1,
                    elem_kind: 1,
                    rows: crate::soa_tracks::MAX_MICRO as u32,
                    cols: 1,
                    stride: 1,
                },
                19 => FieldViewDesc {
                    ptr: self.arena.depth.export_hero_depth.as_ptr() as *const u8,
                    len: self.arena.depth.export_hero_depth.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: MAX_TARGETS as u32,
                    cols: crate::depth::DEPTH_HERO_STRIDE as u32,
                    stride: crate::depth::DEPTH_HERO_STRIDE as u32,
                },
                20 => FieldViewDesc {
                    ptr: self.arena.wind.telem.as_ptr() as *const u8,
                    len: self.arena.wind.telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.wind.telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                21 => FieldViewDesc {
                    ptr: self.arena.wind.planes.as_ptr() as *const u8,
                    len: self.arena.wind.planes.len(),
                    elem_size: 4,
                    elem_kind: 3,
                    rows: crate::wind_tunnel::WIND_H as u32,
                    cols: crate::wind_tunnel::WIND_W as u32,
                    stride: 3,
                },
                22 => FieldViewDesc {
                    ptr: self.arena.elasto.telem.as_ptr() as *const u8,
                    len: self.arena.elasto.telem.len(),
                    elem_size: 4,
                    elem_kind: 2,
                    rows: self.arena.elasto.telem.len() as u32,
                    cols: 1,
                    stride: 1,
                },
                23 => {
                    let is_abos = self.arena.elasto.get_mode()
                        == crate::elastography::ELASTO_MODE_ACOUSTIC_BOS;
                    FieldViewDesc {
                        ptr: self.arena.bos.planes.as_ptr() as *const u8,
                        len: if is_abos {
                            self.arena.bos.planes.len()
                        } else {
                            6 * crate::elastography::ELASTO_V
                        },
                        elem_size: 4,
                        elem_kind: 3,
                        rows: if is_abos {
                            crate::bos::BOS_GRID_H as u32
                        } else {
                            crate::elastography::ELASTO_H as u32
                        },
                        cols: if is_abos {
                            crate::bos::BOS_GRID_W as u32
                        } else {
                            crate::elastography::ELASTO_W as u32
                        },
                        stride: 6,
                    }
                }
                24 => {
                    let is_abos = self.arena.elasto.get_mode()
                        == crate::elastography::ELASTO_MODE_ACOUSTIC_BOS;
                    FieldViewDesc {
                        ptr: self.arena.bos.valid_mask.as_ptr(),
                        len: if is_abos {
                            self.arena.bos.valid_mask.len()
                        } else {
                            crate::elastography::ELASTO_V
                        },
                        elem_size: 1,
                        elem_kind: 1,
                        rows: if is_abos {
                            crate::bos::BOS_GRID_H as u32
                        } else {
                            crate::elastography::ELASTO_H as u32
                        },
                        cols: if is_abos {
                            crate::bos::BOS_GRID_W as u32
                        } else {
                            crate::elastography::ELASTO_W as u32
                        },
                        stride: 1,
                    }
                }
                _ => return None,
            };
            Some(desc)
        }

        pub fn active_tracks(&self) -> Vec<TrackSnapshot> {
            let mut out = Vec::with_capacity(MAX_TARGETS);
            for m in 0..MAX_TARGETS {
                let base = 32 + m * 56;
                if self.arena.telemetry[base] > 0.5 {
                    let kbase = TELEM_KINETIC_START + m * 8;
                    let pbase = TELEM_PHYSICS_START + m * 16;
                    let t = &self.arena.targets[m];
                    let heading_rad = self.arena.telemetry[kbase] * (core::f32::consts::PI / 180.0);
                    let speed_px_s = self.arena.telemetry[kbase + 1];
                    let vx_px_s = crate::fast_cosf(heading_rad) * speed_px_s;
                    let vy_px_s = crate::fast_sinf(heading_rad) * speed_px_s;
                    out.push(TrackSnapshot {
                        slot: m,
                        track_id: self.arena.telemetry[base + 1] as u32,
                        x: self.arena.telemetry[base + 2],
                        y: self.arena.telemetry[base + 3],
                        vx: t.vel_x,
                        vy: t.vel_y,
                        vx_px_s,
                        vy_px_s,
                        vis: self.arena.telemetry[base + 10],
                        bridged: self.arena.telemetry[base + 11],
                        regime: self.arena.telemetry[pbase + 4] as u32,
                        ode_r2: self.arena.telemetry[pbase + 6],
                    });
                }
            }
            out
        }
    }

    impl Default for ChronosEngine {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(any(test, feature = "native-lib"))]
pub use engine::{ChronosEngine, FieldViewDesc, TrackSnapshot};
