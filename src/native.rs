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
    /// `chronos_get_ring_slot_ptr(arg)`: ring slot `arg`.
    RingSlot(u32),
    /// A moving view onto one of the ring slots (`ingest_rgba`, `display_rgba`).
    RingAlias,
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
pub fn regions() -> [NativeRegion; RING_LEN + 35] {
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
