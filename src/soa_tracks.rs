//! CHRONOS-1000 Wave 1 Micro-Swarm SoA Tracker (`MAX_MICRO = 1536`).
//!
//! Pure `#![no_std]` crate submodule: zero heap allocations, zero external host imports,
//! zero call-stack buffers >= 64 B (all working buffers live inside `MicroSwarmState`),
//! and top-level pipeline phases marked `#[inline(never)]` to preserve `STACK_MAX_PATH`.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[cfg(not(target_arch = "wasm32"))]
use crate::fast_expf;
#[cfg(target_arch = "wasm32")]
use crate::simd_ops::hsum_f32x4;
use crate::{wasm_floorf, wasm_sqrtf};

pub const MAX_MICRO: usize = 1536;
pub const MAX_MICRO_PEAKS: usize = 2048;
pub const MAX_CSR_K: usize = 8;
pub const SWARM_TELEM_LEN: usize = 64;

pub const MICRO_CELL_W: usize = 32;
pub const MICRO_CELL_H: usize = 18;
pub const MICRO_CELLS: usize = MICRO_CELL_W * MICRO_CELL_H; // 576
pub const MICRO_EXPORT_STRIDE: usize = 8;

pub const MICRO_STATE_INACTIVE: u8 = 0;
pub const MICRO_STATE_CONFIRMED: u8 = 1;
pub const MICRO_STATE_TENTATIVE: u8 = 2;

const MAX_RAW_PEAKS: usize = 4096;
const NMS_BITMAP_WORDS: usize = 3600; // 360 rows * 10 u64/row (covers up to 640x360)
const SCORE_RING_STRIDE: usize = 640;
const SCORE_RING_ROWS: usize = 3;

const R_NMS_HD: usize = 3; // Euclidean < 3 px HD (`>= 3 px` survives -> >85% recall on 1,200)
const R_NMS_SD: usize = 2; // Euclidean < 2 px SD

const MAX_COAST_FRAMES: u8 = 8;
const AUCTION_SCALE_Q: f32 = 4096.0;
const COAST_Q_CONF: i32 = 3277; // round(0.80 * 4096)
const COAST_Q_TENT: i32 = 2458; // round(0.60 * 4096)
const AMBIG_MARGIN_Q: i32 = 82; // < 0.02 * 4096 margin marks match ambiguous
const AUTO_OCC_CELLS_PROBE: u32 = 192;
const AUTO_ENTRY_PEAKS: usize = 768;
const AUTO_HOLD_PEAKS: usize = 512;
const AUTO_CORR_RADIUS_Q: f32 = 2.5;
const AUTO_CORR_SAD_MAX: f32 = 0.30;
const AUTO_CORR_SHIFT_MAX_Q: f32 = 10.0;
const AUTO_PROBE_MAX_BACKOFF: u8 = 32;
const MAX_PRELIM_CANDS: usize = 8;

/// Packed 16-byte Stage-1 CSR pre-filter candidate so insertion shifts move via a single `v128`.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct PrelimCand {
    pub score: f32,
    pub d2: f32,
    pub padic_pen: f32,
    pub idx: i32,
}

impl PrelimCand {
    pub const EMPTY: Self = Self {
        score: 0.0,
        d2: 0.0,
        padic_pen: 0.0,
        idx: -1,
    };
}
const MAX_MICRO_ID_F32: u32 = 16_777_215; // Exact contiguous f32 integer ID range
pub const FTLE_INVALID_FRAME: u32 = u32::MAX;
pub const FTLE_INVALID_VAL: f32 = -999.0;
pub const FTLE_BANK_PERIOD: u32 = 20;
pub const FTLE_BANK_STAGGER: u32 = 10;

/// Structure-of-Arrays micro-track storage (`1536` slots, zero heap).
#[repr(C, align(16))]
pub struct MicroTrackSoA {
    // Hot kinematics (velocities in px/s; `prev_x/y` = t-1 state, `last_obs_x/y` = last true measurement)
    pub pos_x: [f32; MAX_MICRO],
    pub pos_y: [f32; MAX_MICRO],
    pub prev_x: [f32; MAX_MICRO],
    pub prev_y: [f32; MAX_MICRO],
    pub last_obs_x: [f32; MAX_MICRO],
    pub last_obs_y: [f32; MAX_MICRO],
    pub vel_x: [f32; MAX_MICRO],
    pub vel_y: [f32; MAX_MICRO],
    pub pred_x: [f32; MAX_MICRO],
    pub pred_y: [f32; MAX_MICRO],
    pub imm_w_cv: [f32; MAX_MICRO],
    pub imm_w_bal: [f32; MAX_MICRO],
    pub imm_w_osc: [f32; MAX_MICRO],

    // Staggered 20-frame FTLE anchors (`A` resets %20==0, `B` resets %20==10, `FTLE_INVALID_FRAME` until sync)
    pub ftle_a_x0: [f32; MAX_MICRO],
    pub ftle_a_y0: [f32; MAX_MICRO],
    pub ftle_b_x0: [f32; MAX_MICRO],
    pub ftle_b_y0: [f32; MAX_MICRO],
    pub ftle_val: [f32; MAX_MICRO],
    pub ftle_a_frame: [u32; MAX_MICRO],
    pub ftle_b_frame: [u32; MAX_MICRO],

    // IDs, signatures, opponent chroma & lifecycle
    pub track_id: [u32; MAX_MICRO],
    pub anchor_sig: [u32; MAX_MICRO],
    pub curr_sig: [u32; MAX_MICRO],
    pub birth_frame: [u32; MAX_MICRO],
    pub chroma_rg: [f32; MAX_MICRO],
    pub chroma_by: [f32; MAX_MICRO],
    pub age: [u16; MAX_MICRO],
    pub state: [u8; MAX_MICRO],      // 0=inactive, 1=confirmed, 2=tentative/coasting
    pub coasting: [u8; MAX_MICRO],   // 0=matched, 1..=8=coasting frames
    pub reacq_flag: [u8; MAX_MICRO], // 1 if re-acquired from coasting on current frame
    pub ambiguous_flag: [u8; MAX_MICRO], // 1 if weak auction margin (<0.02) or repaired; FTLE must skip
    pub body_id: [u8; MAX_MICRO],    // 0=unassigned, 1..=32 written by continuum_physics
    pub regime: [u8; MAX_MICRO],     // 0=CV, 1=Ballistic, 2=Oscillatory

    // Virtual modal accelerometer bandpass & quadrature state per track
    pub eq_x: [f32; MAX_MICRO],
    pub eq_y: [f32; MAX_MICRO],
    pub bp_i: [f32; MAX_MICRO],
    pub bp_q: [f32; MAX_MICRO],
    pub bp_prev_y: [f32; MAX_MICRO],
    pub disp_y: [f32; MAX_MICRO],
    pub modal_phase: [f32; MAX_MICRO],
    pub modal_freq_hz: [f32; MAX_MICRO],
    pub modal_amp: [f32; MAX_MICRO],

    // 16-byte aligned zero-mean quantized 8x8 appearance patches (96 KiB)
    pub tpl8_i8: [[i8; 64]; MAX_MICRO],
}

impl MicroTrackSoA {
    pub const fn zeroed() -> Self {
        Self {
            pos_x: [0.0; MAX_MICRO],
            pos_y: [0.0; MAX_MICRO],
            prev_x: [0.0; MAX_MICRO],
            prev_y: [0.0; MAX_MICRO],
            last_obs_x: [0.0; MAX_MICRO],
            last_obs_y: [0.0; MAX_MICRO],
            vel_x: [0.0; MAX_MICRO],
            vel_y: [0.0; MAX_MICRO],
            pred_x: [0.0; MAX_MICRO],
            pred_y: [0.0; MAX_MICRO],
            imm_w_cv: [0.0; MAX_MICRO],
            imm_w_bal: [0.0; MAX_MICRO],
            imm_w_osc: [0.0; MAX_MICRO],
            ftle_a_x0: [0.0; MAX_MICRO],
            ftle_a_y0: [0.0; MAX_MICRO],
            ftle_b_x0: [0.0; MAX_MICRO],
            ftle_b_y0: [0.0; MAX_MICRO],
            ftle_val: [0.0; MAX_MICRO],
            ftle_a_frame: [FTLE_INVALID_FRAME; MAX_MICRO],
            ftle_b_frame: [FTLE_INVALID_FRAME; MAX_MICRO],
            track_id: [0; MAX_MICRO],
            anchor_sig: [0; MAX_MICRO],
            curr_sig: [0; MAX_MICRO],
            birth_frame: [0; MAX_MICRO],
            chroma_rg: [0.0; MAX_MICRO],
            chroma_by: [0.0; MAX_MICRO],
            age: [0; MAX_MICRO],
            state: [0; MAX_MICRO],
            coasting: [0; MAX_MICRO],
            reacq_flag: [0; MAX_MICRO],
            ambiguous_flag: [0; MAX_MICRO],
            body_id: [0; MAX_MICRO],
            regime: [0; MAX_MICRO],
            eq_x: [0.0; MAX_MICRO],
            eq_y: [0.0; MAX_MICRO],
            bp_i: [0.0; MAX_MICRO],
            bp_q: [0.0; MAX_MICRO],
            bp_prev_y: [0.0; MAX_MICRO],
            disp_y: [0.0; MAX_MICRO],
            modal_phase: [0.0; MAX_MICRO],
            modal_freq_hz: [0.0; MAX_MICRO],
            modal_amp: [0.0; MAX_MICRO],
            tpl8_i8: [[0; 64]; MAX_MICRO],
        }
    }
}

const CSR_CELL_W: usize = 64;
const CSR_CELL_H: usize = 36;
const CSR_CELLS: usize = CSR_CELL_W * CSR_CELL_H;

/// Complete Arena-resident state and zero-stack scratch for `soa_tracks`.
#[repr(C, align(16))]
pub struct MicroSwarmState {
    pub tracks: MicroTrackSoA,
    pub export_micro: [f32; MAX_MICRO * MICRO_EXPORT_STRIDE],
    pub swarm_telem: [f32; SWARM_TELEM_LEN],

    // Extracted sub-pixel micro-peaks
    pub peak_x: [f32; MAX_MICRO_PEAKS],
    pub peak_y: [f32; MAX_MICRO_PEAKS],
    pub peak_energy: [f32; MAX_MICRO_PEAKS],
    pub peak_rg: [f32; MAX_MICRO_PEAKS],
    pub peak_by: [f32; MAX_MICRO_PEAKS],
    pub peak_sig: [u32; MAX_MICRO_PEAKS],
    pub peak_morton: [u16; MAX_MICRO_PEAKS],
    pub peak_order: [u16; MAX_MICRO_PEAKS],
    pub peak_sort_tmp: [u16; MAX_MICRO_PEAKS],
    pub peak_next: [i16; MAX_MICRO_PEAKS],
    pub peak_matched: [u8; MAX_MICRO_PEAKS],
    pub peak_tpl8: [[i8; 64]; MAX_MICRO_PEAKS],

    // 32x18 spatial hashes & radix histogram
    pub cell_head: [i16; MICRO_CELLS],
    pub track_cell_head: [i16; MICRO_CELLS],
    pub track_next: [i16; MAX_MICRO],
    pub radix_cnt: [u16; 256],

    // NMS bitmask + 3-row active score ring (zero stack >= 64 B)
    pub nms_bitmap: [u64; NMS_BITMAP_WORDS],
    pub score_ring: [f32; SCORE_RING_STRIDE * SCORE_RING_ROWS],
    pub raw_sort_order: [u16; MAX_RAW_PEAKS],
    pub raw_sort_tmp: [u16; MAX_RAW_PEAKS],

    // Sparse CSR (k <= 8) candidate matrix & Bertsekas Q12 Integer Forward-Reverse Auction scratch
    // `csr_cost` is re-used at start of frame as 4096-entry pre-NMS raw candidate float min-heap.
    pub csr_cost: [f32; MAX_MICRO * MAX_CSR_K],     // 12,288 f32s (exact Q12 integers during Auction)
    pub csr_peak_idx: [i16; MAX_MICRO * MAX_CSR_K], // 12,288 i16s
    pub rev_head: [i16; MAX_MICRO_PEAKS],           // Peak->edge linked head for Reverse Auction cleanup
    pub rev_next: [i16; MAX_MICRO * MAX_CSR_K],     // Next edge index in `0..12288`
    pub csr_len: [u8; MAX_MICRO],
    pub peak_price_q: [i32; MAX_MICRO_PEAKS],
    pub peak_owner: [i16; MAX_MICRO_PEAKS],
    pub track_match: [i16; MAX_MICRO],
    pub auction_q: [u16; MAX_MICRO],
    pub track_order: [u16; MAX_MICRO],
    pub track_morton: [u16; MAX_MICRO],

    // Arena-resident IC-LK & CSR compatibility pre-filter scratch (prevents >= 64 B stack arrays)
    pub lk_tpl_f: [f32; 64],
    pub lk_gx: [f32; 64],
    pub lk_gy: [f32; 64],
    pub lk_warp_f: [f32; 64],
    pub prelim_cands: [PrelimCand; MAX_PRELIM_CANDS],

    // Policy & run telemetry scalars
    pub swarm_mode: u32, // 0=Off (default), 1=Auto, 2=Force
    pub swarm_active: u8,
    pub low_density_streak: u8,
    pub probe_streak: u8,
    pub probe_backoff: u8,
    pub probe_fail_streak: u8,
    pub cut_cooldown: u8,
    pub id_wrapped: u8,
    pub auction_incomplete: u8,
    pub probe_peak_count: u16,
    pub num_peaks: usize,
    pub next_micro_id: u32,
    pub reset_base_micro_id: u32,
    pub total_births_lifetime: u32,
    pub active_count: u32,
    pub confirmed_count: u32,
    pub tentative_count: u32,
    pub coasting_count: u32,
    pub births_count: u32,
    pub peak_simultaneous_confirmed: u32,
    pub auction_steps: u32,
    pub auction_eps_q: u32,
    pub reacq_events: u32,
    pub ambiguous_count: u32,
    pub raw_heap_overflow_cnt: u32,
    pub csr_overflow_cnt: u32,
    pub auction_cap_hits: u32,
    pub auction_repairs: u32,
    pub inspected_micro_tid: u32,
    pub inspected_body_id: u32,
    pub mean_ic_lk_res: f32,

    // Contiguous cell-sorted peak AoS/SoA slices populated in `build_peak_morton_cell_hash`
    // (`+ 4` tail padding allows branch-free masked 4-wide SIMD loads on short cells)
    pub cell_off: [u16; CSR_CELLS + 2],
    pub cell_px: [f32; MAX_MICRO_PEAKS + 4],
    pub cell_py: [f32; MAX_MICRO_PEAKS + 4],
    pub cell_psig: [u32; MAX_MICRO_PEAKS],
    pub cell_pidx: [u16; MAX_MICRO_PEAKS],
}

impl MicroSwarmState {
    pub const fn zeroed() -> Self {
        Self {
            tracks: MicroTrackSoA::zeroed(),
            export_micro: [0.0; MAX_MICRO * MICRO_EXPORT_STRIDE],
            swarm_telem: [0.0; SWARM_TELEM_LEN],
            peak_x: [0.0; MAX_MICRO_PEAKS],
            peak_y: [0.0; MAX_MICRO_PEAKS],
            peak_energy: [0.0; MAX_MICRO_PEAKS],
            peak_rg: [0.0; MAX_MICRO_PEAKS],
            peak_by: [0.0; MAX_MICRO_PEAKS],
            peak_sig: [0; MAX_MICRO_PEAKS],
            peak_morton: [0; MAX_MICRO_PEAKS],
            peak_order: [0; MAX_MICRO_PEAKS],
            peak_sort_tmp: [0; MAX_MICRO_PEAKS],
            peak_next: [-1; MAX_MICRO_PEAKS],
            peak_matched: [0; MAX_MICRO_PEAKS],
            peak_tpl8: [[0; 64]; MAX_MICRO_PEAKS],
            cell_head: [-1; MICRO_CELLS],
            track_cell_head: [-1; MICRO_CELLS],
            track_next: [-1; MAX_MICRO],
            radix_cnt: [0; 256],
            nms_bitmap: [0; NMS_BITMAP_WORDS],
            score_ring: [0.0; SCORE_RING_STRIDE * SCORE_RING_ROWS],
            raw_sort_order: [0; MAX_RAW_PEAKS],
            raw_sort_tmp: [0; MAX_RAW_PEAKS],
            csr_cost: [0.0; MAX_MICRO * MAX_CSR_K],
            csr_peak_idx: [-1; MAX_MICRO * MAX_CSR_K],
            rev_head: [-1; MAX_MICRO_PEAKS],
            rev_next: [-1; MAX_MICRO * MAX_CSR_K],
            csr_len: [0; MAX_MICRO],
            peak_price_q: [0; MAX_MICRO_PEAKS],
            peak_owner: [-1; MAX_MICRO_PEAKS],
            track_match: [-1; MAX_MICRO],
            auction_q: [0; MAX_MICRO],
            track_order: [0; MAX_MICRO],
            track_morton: [0; MAX_MICRO],
            lk_tpl_f: [0.0; 64],
            lk_gx: [0.0; 64],
            lk_gy: [0.0; 64],
            lk_warp_f: [0.0; 64],
            prelim_cands: [PrelimCand::EMPTY; MAX_PRELIM_CANDS],
            swarm_mode: 0,
            swarm_active: 0,
            low_density_streak: 0,
            probe_streak: 0,
            probe_backoff: 0,
            probe_fail_streak: 0,
            cut_cooldown: 0,
            id_wrapped: 0,
            auction_incomplete: 0,
            probe_peak_count: 0,
            num_peaks: 0,
            next_micro_id: 1,
            reset_base_micro_id: 1,
            total_births_lifetime: 0,
            active_count: 0,
            confirmed_count: 0,
            tentative_count: 0,
            coasting_count: 0,
            births_count: 0,
            peak_simultaneous_confirmed: 0,
            auction_steps: 0,
            auction_eps_q: 0,
            reacq_events: 0,
            ambiguous_count: 0,
            raw_heap_overflow_cnt: 0,
            csr_overflow_cnt: 0,
            auction_cap_hits: 0,
            auction_repairs: 0,
            inspected_micro_tid: 0,
            inspected_body_id: 0,
            mean_ic_lk_res: 0.0,
            cell_off: [0; CSR_CELLS + 2],
            cell_px: [0.0; MAX_MICRO_PEAKS + 4],
            cell_py: [0.0; MAX_MICRO_PEAKS + 4],
            cell_psig: [0; MAX_MICRO_PEAKS],
            cell_pidx: [0; MAX_MICRO_PEAKS],
        }
    }

    /// Cold-boot-only zero-initializer for `MicroSwarmState` before `continuum_physics` runs.
    /// Zeroes the entire struct including `swarm_telem[0..64]`; runtime resets must use
    /// `reset()` or `reset_active_keep_mode()` to preserve continuum telemetry `[16..63]`.
    #[inline(never)]
    pub fn init_in_place(&mut self) {
        // SAFETY: `MicroSwarmState` is plain data (primitives/arrays); zeroing in place avoids ~400 KiB stack moves.
        unsafe {
            core::ptr::write_bytes(
                self as *mut Self as *mut u8,
                0,
                core::mem::size_of::<Self>(),
            );
        }
        self.tracks.ftle_a_frame.fill(FTLE_INVALID_FRAME);
        self.tracks.ftle_b_frame.fill(FTLE_INVALID_FRAME);
        self.peak_next.fill(-1);
        self.cell_head.fill(-1);
        self.track_cell_head.fill(-1);
        self.track_next.fill(-1);
        self.csr_peak_idx.fill(-1);
        self.rev_head.fill(-1);
        self.rev_next.fill(-1);
        self.peak_owner.fill(-1);
        self.track_match.fill(-1);
        self.prelim_cands.fill(PrelimCand::EMPTY);
        self.next_micro_id = 1;
        self.reset_base_micro_id = 1;
    }

    /// Invalidates staggered FTLE anchor banks and modal/continuity state across tracks
    /// on frame gaps, seeks, or invalid frames without touching `swarm_mode` or `[16..63]`.
    #[inline(never)]
    pub fn invalidate_continuity(&mut self) {
        let mut i = 0usize;
        while i < MAX_MICRO {
            self.tracks.ftle_a_frame[i] = FTLE_INVALID_FRAME;
            self.tracks.ftle_b_frame[i] = FTLE_INVALID_FRAME;
            self.tracks.ftle_val[i] = 0.0;
            self.tracks.bp_i[i] = 0.0;
            self.tracks.bp_q[i] = 0.0;
            self.tracks.bp_prev_y[i] = 0.0;
            self.tracks.disp_y[i] = 0.0;
            self.tracks.modal_phase[i] = 0.0;
            self.tracks.modal_freq_hz[i] = 0.0;
            self.tracks.modal_amp[i] = 0.0;
            i += 1;
        }
    }

    #[inline(never)]
    pub fn reset(&mut self) {
        let saved_mode = self.swarm_mode;
        self.reset_active_keep_mode();
        self.swarm_mode = saved_mode;
        self.next_micro_id = 1;
        self.reset_base_micro_id = 1;
        self.total_births_lifetime = 0;
        self.id_wrapped = 0;
        self.auction_incomplete = 0;
        self.cut_cooldown = 0;
        self.probe_backoff = 0;
        self.probe_fail_streak = 0;
        self.peak_simultaneous_confirmed = 0;
    }

    #[inline(never)]
    pub fn reset_active_keep_mode(&mut self) {
        let mut i = 0;
        while i < MAX_MICRO {
            self.tracks.state[i] = MICRO_STATE_INACTIVE;
            self.tracks.track_id[i] = 0;
            self.tracks.coasting[i] = 0;
            self.tracks.reacq_flag[i] = 0;
            self.tracks.ambiguous_flag[i] = 0;
            self.tracks.body_id[i] = 0;
            self.tracks.age[i] = 0;
            self.tracks.ftle_val[i] = 0.0;
            self.tracks.ftle_a_frame[i] = FTLE_INVALID_FRAME;
            self.tracks.ftle_b_frame[i] = FTLE_INVALID_FRAME;
            self.track_next[i] = -1;
            i += 1;
        }
        let mut c = 0;
        while c < MICRO_CELLS {
            self.cell_head[c] = -1;
            self.track_cell_head[c] = -1;
            c += 1;
        }
        let mut e = 0;
        while e < MAX_MICRO * MICRO_EXPORT_STRIDE {
            self.export_micro[e] = 0.0;
            e += 1;
        }
        // Restrict tracker resets strictly to `[0..=15]`; never overwrite continuum `[16..=41]` or reserved `[42..=63]`.
        self.swarm_telem[0] = self.swarm_mode as f32;
        let mut t = 1;
        while t < 16 {
            self.swarm_telem[t] = 0.0;
            t += 1;
        }
        self.swarm_active = if self.swarm_mode == 2 { 1 } else { 0 };
        self.low_density_streak = 0;
        self.probe_streak = 0;
        self.probe_peak_count = 0;
        self.num_peaks = 0;
        self.active_count = 0;
        self.confirmed_count = 0;
        self.tentative_count = 0;
        self.coasting_count = 0;
        self.births_count = 0;
        self.auction_steps = 0;
        self.auction_eps_q = 0;
        self.auction_incomplete = 0;
        self.reacq_events = 0;
        self.ambiguous_count = 0;
        self.raw_heap_overflow_cnt = 0;
        self.csr_overflow_cnt = 0;
        self.auction_cap_hits = 0;
        self.auction_repairs = 0;
        self.inspected_micro_tid = 0;
        self.inspected_body_id = 0;
        self.mean_ic_lk_res = 0.0;
    }
}

#[inline(always)]
fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline(always)]
fn spread_bits_8(mut x: u32) -> u32 {
    x &= 0xFF;
    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    x = (x | (x << 1)) & 0x5555;
    x
}

/// Computes a 16-bit spatial Morton Z-code `(qy[7..0] interleaved with qx[7..0])`.
#[inline(always)]
pub fn morton16_xy(x: f32, y: f32, inv_w_255: f32, inv_h_255: f32) -> u16 {
    let qx = clamp_f32(x * inv_w_255, 0.0, 255.0) as u32;
    let qy = clamp_f32(y * inv_h_255, 0.0, 255.0) as u32;
    (spread_bits_8(qx) | (spread_bits_8(qy) << 1)) as u16
}

#[inline(always)]
fn cell_coord(pos: f32, inv_span_cells: f32, max_cell: usize) -> usize {
    let idx = (pos * inv_span_cells) as i32;
    if idx < 0 {
        0
    } else if idx as usize > max_cell {
        max_cell
    } else {
        idx as usize
    }
}

/// Overflow-free 8x8 `i8` patch SAD using exact two's-complement `i8x16_max - i8x16_min` (`0..=254u8`).
/// Summing all 4 pairwise-widened chunks in `u16x8` stays <= 2,040 (< 65,535) and defers the u32 widen.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn patch_8x8_sad_preloaded_i8_simd128(
    va0: v128,
    va1: v128,
    va2: v128,
    va3: v128,
    b: &[i8; 64],
) -> u32 {
    let pb = b.as_ptr() as *const v128;

    let vb0 = v128_load(pb);
    let d0 = i8x16_sub(i8x16_max(va0, vb0), i8x16_min(va0, vb0));

    let vb1 = v128_load(pb.add(1));
    let d1 = i8x16_sub(i8x16_max(va1, vb1), i8x16_min(va1, vb1));

    let vb2 = v128_load(pb.add(2));
    let d2 = i8x16_sub(i8x16_max(va2, vb2), i8x16_min(va2, vb2));

    let vb3 = v128_load(pb.add(3));
    let d3 = i8x16_sub(i8x16_max(va3, vb3), i8x16_min(va3, vb3));

    let s01 = u16x8_add(
        u16x8_extadd_pairwise_u8x16(d0),
        u16x8_extadd_pairwise_u8x16(d1),
    );
    let s23 = u16x8_add(
        u16x8_extadd_pairwise_u8x16(d2),
        u16x8_extadd_pairwise_u8x16(d3),
    );
    let acc32 = u32x4_extadd_pairwise_u16x8(u16x8_add(s01, s23));

    let shuf1 = i32x4_shuffle::<2, 3, 0, 1>(acc32, acc32);
    let s1 = i32x4_add(acc32, shuf1);
    let shuf2 = i32x4_shuffle::<1, 0, 3, 2>(s1, s1);
    let s2 = i32x4_add(s1, shuf2);
    i32x4_extract_lane::<0>(s2) as u32
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub fn patch_8x8_sad_i8_simd128(a: &[i8; 64], b: &[i8; 64]) -> u32 {
    unsafe {
        let pa = a.as_ptr() as *const v128;
        patch_8x8_sad_preloaded_i8_simd128(
            v128_load(pa),
            v128_load(pa.add(1)),
            v128_load(pa.add(2)),
            v128_load(pa.add(3)),
            b,
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub fn patch_8x8_sad_i8_simd128(a: &[i8; 64], b: &[i8; 64]) -> u32 {
    let mut sum = 0u32;
    let mut i = 0;
    while i < 64 {
        let d = (a[i] as i32) - (b[i] as i32);
        sum += d.unsigned_abs();
        i += 1;
    }
    sum
}

/// Inline minimum 2-adic distance across current & anchor signatures using hardware `clz`.
#[inline(always)]
fn padic_min_inline(sig_curr: u32, sig_anc: u32, p_sig: u32) -> f32 {
    let val = (sig_curr ^ p_sig)
        .leading_zeros()
        .max((sig_anc ^ p_sig).leading_zeros());
    if val >= 32 {
        0.0
    } else {
        f32::from_bits((127u32 - val) << 23)
    }
}

/// Computes local 8-point `2x2` Shi-Tomasi / Noble cornerness (`det / trace`) inside `[x-2..x+2]`.
/// Vanishes on straight edges at any angle while peaking on blobs, speckles, and polygon corners.
/// Caller guarantees `2 <= x < w - 2`, `2 <= y < h - 2`, and `luma.len() >= w * h`.
#[inline(always)]
fn local_noble_corner(luma: &[f32], w: usize, x: usize, y: usize) -> (f32, f32) {
    debug_assert!(x >= 2 && x + 2 < w && y >= 2 && (y + 3) * w <= luma.len());
    #[cfg(target_arch = "wasm32")]
    let (sxx, syy, sxy) = unsafe {
        let wi = w as isize;
        let lp = luma.as_ptr().offset((y as isize) * wi + (x as isize));
        let mask3 = i32x4_replace_lane::<3>(i32x4_splat(-1), 0);
        let mask_mid = i32x4_replace_lane::<1>(mask3, 0);

        let rm2_m1 = core::ptr::read_unaligned(lp.offset(-2 * wi - 1) as *const v128);
        let rm1_m2 = core::ptr::read_unaligned(lp.offset(-wi - 2) as *const v128);
        let rm1_m1 = core::ptr::read_unaligned(lp.offset(-wi - 1) as *const v128);
        let rm1_p2 = *lp.offset(-wi + 2);
        let r0_m2 = core::ptr::read_unaligned(lp.offset(-2) as *const v128);
        let r0_m1 = core::ptr::read_unaligned(lp.offset(-1) as *const v128);
        let r0_p2 = *lp.offset(2);
        let rp1_m2 = core::ptr::read_unaligned(lp.offset(wi - 2) as *const v128);
        let rp1_m1 = core::ptr::read_unaligned(lp.offset(wi - 1) as *const v128);
        let rp1_p2 = *lp.offset(wi + 2);
        let rp2_m1 = core::ptr::read_unaligned(lp.offset(2 * wi - 1) as *const v128);

        let gx_t = v128_and(
            f32x4_sub(
                f32x4_replace_lane::<2>(i32x4_shuffle::<2, 3, 3, 3>(rm1_m2, rm1_m2), rm1_p2),
                rm1_m2,
            ),
            mask3,
        );
        let gx_m = v128_and(
            f32x4_sub(
                f32x4_replace_lane::<2>(i32x4_shuffle::<2, 3, 3, 3>(r0_m2, r0_m2), r0_p2),
                r0_m2,
            ),
            mask_mid,
        );
        let gx_b = v128_and(
            f32x4_sub(
                f32x4_replace_lane::<2>(i32x4_shuffle::<2, 3, 3, 3>(rp1_m2, rp1_m2), rp1_p2),
                rp1_m2,
            ),
            mask3,
        );

        let gy_t = v128_and(f32x4_sub(r0_m1, rm2_m1), mask3);
        let gy_m = v128_and(f32x4_sub(rp1_m1, rm1_m1), mask_mid);
        let gy_b = v128_and(f32x4_sub(rp2_m1, r0_m1), mask3);

        let vxx = f32x4_add(
            f32x4_mul(gx_t, gx_t),
            f32x4_add(f32x4_mul(gx_m, gx_m), f32x4_mul(gx_b, gx_b)),
        );
        let vyy = f32x4_add(
            f32x4_mul(gy_t, gy_t),
            f32x4_add(f32x4_mul(gy_m, gy_m), f32x4_mul(gy_b, gy_b)),
        );
        let vxy = f32x4_add(
            f32x4_mul(gx_t, gy_t),
            f32x4_add(f32x4_mul(gx_m, gy_m), f32x4_mul(gx_b, gy_b)),
        );
        (hsum_f32x4(vxx), hsum_f32x4(vyy), hsum_f32x4(vxy))
    };

    #[cfg(not(target_arch = "wasm32"))]
    let (sxx, syy, sxy) = {
        let mut sxx = 0.0f32;
        let mut syy = 0.0f32;
        let mut sxy = 0.0f32;

        const OFFSETS: [(isize, isize); 8] = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ];
        let wi = w as isize;
        let base = (y as isize) * wi + (x as isize);
        let lp = luma.as_ptr();
        let mut k = 0;
        while k < 8 {
            let p = base + OFFSETS[k].1 * wi + OFFSETS[k].0;
            let (gx, gy) = unsafe {
                (
                    *lp.offset(p + 1) - *lp.offset(p - 1),
                    *lp.offset(p + wi) - *lp.offset(p - wi),
                )
            };
            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
            k += 1;
        }
        (sxx, syy, sxy)
    };
    let trace = sxx + syy;
    let det = sxx * syy - sxy * sxy;
    let noble = if trace > 1e-6 && det > 0.0 {
        det / trace
    } else {
        0.0
    };
    (wasm_sqrtf(noble), trace)
}

/// Evaluates `det / trace` Noble cornerness across 4 consecutive pixels `[x..x+4]` using SIMD128,
/// masking inactive lanes and using per-lane squared thresholds without taking `sqrt`.
/// Caller guarantees `x >= 2`, `x + 5 < w`, `y >= 2`, and `(y + 3) * w <= luma.len()`.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn local_noble_from_rows_4x(
    lp_q: *const f32,
    wi: isize,
    rm1_a: v128,
    rm1_b: v128,
    r0_a: v128,
    r0_b: v128,
    rp1_a: v128,
    rp1_b: v128,
    active_mask: v128,
    vmin_c2: v128,
) -> v128 {
    let rm1_0 = i32x4_shuffle::<2, 3, 4, 5>(rm1_a, rm1_b);
    let r0_0 = i32x4_shuffle::<2, 3, 4, 5>(r0_a, r0_b);
    let rp1_0 = i32x4_shuffle::<2, 3, 4, 5>(rp1_a, rp1_b);

    let gxt_lo = f32x4_sub(rm1_0, rm1_a);
    let gxt_hi = f32x4_sub(rm1_b, rm1_0);
    let gxm_lo = f32x4_sub(r0_0, r0_a);
    let gxm_hi = f32x4_sub(r0_b, r0_0);
    let gxb_lo = f32x4_sub(rp1_0, rp1_a);
    let gxb_hi = f32x4_sub(rp1_b, rp1_0);

    let tb_xx_lo = f32x4_add(f32x4_mul(gxt_lo, gxt_lo), f32x4_mul(gxb_lo, gxb_lo));
    let tb_xx_hi = f32x4_add(f32x4_mul(gxt_hi, gxt_hi), f32x4_mul(gxb_hi, gxb_hi));
    let tb_xx_c = i32x4_shuffle::<1, 2, 3, 6>(tb_xx_lo, tb_xx_hi);
    let sxx = f32x4_add(
        f32x4_add(tb_xx_lo, f32x4_add(tb_xx_c, tb_xx_hi)),
        f32x4_add(f32x4_mul(gxm_lo, gxm_lo), f32x4_mul(gxm_hi, gxm_hi)),
    );

    // Since det/trace = (sxx*syy - sxy^2)/(sxx+syy) < min(sxx, syy), skip rows +-2 & syy/sxy/div
    // whenever no active lane in `sxx` can exceed `0.999 * min_c2` (zero false rejections).
    let vzero = f32x4_splat(0.0);
    let vmin_guard = f32x4_mul(vmin_c2, f32x4_splat(0.999));
    let cand_x = v128_and(active_mask, f32x4_gt(sxx, vmin_guard));
    if !v128_any_true(cand_x) {
        return vzero;
    }

    let p_m2 = lp_q.offset(-2 * wi);
    let rm2_a = core::ptr::read_unaligned(p_m2.offset(-2) as *const v128);
    let rm2_b = core::ptr::read_unaligned(p_m2.offset(2) as *const v128);
    let p_p2 = lp_q.offset(2 * wi);
    let rp2_a = core::ptr::read_unaligned(p_p2.offset(-2) as *const v128);
    let rp2_b = core::ptr::read_unaligned(p_p2.offset(2) as *const v128);

    let dy_ta = f32x4_sub(r0_a, rm2_a);
    let dy_tb = f32x4_sub(r0_b, rm2_b);
    let gyt_lo = i32x4_shuffle::<1, 2, 3, 4>(dy_ta, dy_tb);
    let gyt_hi = i32x4_shuffle::<3, 4, 5, 6>(dy_ta, dy_tb);

    let dy_ma = f32x4_sub(rp1_a, rm1_a);
    let dy_mb = f32x4_sub(rp1_b, rm1_b);
    let gym_lo = i32x4_shuffle::<1, 2, 3, 4>(dy_ma, dy_mb);
    let gym_hi = i32x4_shuffle::<3, 4, 5, 6>(dy_ma, dy_mb);

    let dy_ba = f32x4_sub(rp2_a, r0_a);
    let dy_bb = f32x4_sub(rp2_b, r0_b);
    let gyb_lo = i32x4_shuffle::<1, 2, 3, 4>(dy_ba, dy_bb);
    let gyb_hi = i32x4_shuffle::<3, 4, 5, 6>(dy_ba, dy_bb);

    let tb_yy_lo = f32x4_add(f32x4_mul(gyt_lo, gyt_lo), f32x4_mul(gyb_lo, gyb_lo));
    let tb_yy_hi = f32x4_add(f32x4_mul(gyt_hi, gyt_hi), f32x4_mul(gyb_hi, gyb_hi));
    let tb_yy_c = i32x4_shuffle::<1, 2, 3, 6>(tb_yy_lo, tb_yy_hi);
    let syy = f32x4_add(
        f32x4_add(tb_yy_lo, f32x4_add(tb_yy_c, tb_yy_hi)),
        f32x4_add(f32x4_mul(gym_lo, gym_lo), f32x4_mul(gym_hi, gym_hi)),
    );
    let cand_xy = v128_and(cand_x, f32x4_gt(syy, vmin_guard));
    if !v128_any_true(cand_xy) {
        return vzero;
    }

    let tb_xy_lo = f32x4_add(f32x4_mul(gxt_lo, gyt_lo), f32x4_mul(gxb_lo, gyb_lo));
    let tb_xy_hi = f32x4_add(f32x4_mul(gxt_hi, gyt_hi), f32x4_mul(gxb_hi, gyb_hi));
    let tb_xy_c = i32x4_shuffle::<1, 2, 3, 6>(tb_xy_lo, tb_xy_hi);
    let sxy = f32x4_add(
        f32x4_add(tb_xy_lo, f32x4_add(tb_xy_c, tb_xy_hi)),
        f32x4_add(f32x4_mul(gxm_lo, gym_lo), f32x4_mul(gxm_hi, gym_hi)),
    );

    let trace = f32x4_add(sxx, syy);
    let det = f32x4_sub(f32x4_mul(sxx, syy), f32x4_mul(sxy, sxy));
    let veps = f32x4_splat(1e-6);
    let likely = v128_and(
        cand_xy,
        v128_and(f32x4_gt(trace, veps), f32x4_gt(det, f32x4_mul(trace, vmin_c2))),
    );
    if !v128_any_true(likely) {
        return vzero;
    }
    v128_and(f32x4_div(det, f32x4_pmax(trace, veps)), likely)
}

#[inline(always)]
unsafe fn local_noble_corner_4x(
    luma_ptr: *const f32,
    w: usize,
    x: usize,
    y: usize,
    active_mask: v128,
    vmin_c2: v128,
) -> v128 {
    let wi = w as isize;
    let lp = luma_ptr.offset((y as isize) * wi + (x as isize));
    let rm1_a = core::ptr::read_unaligned(lp.offset(-wi - 2) as *const v128);
    let rm1_b = core::ptr::read_unaligned(lp.offset(-wi + 2) as *const v128);
    let r0_a = core::ptr::read_unaligned(lp.offset(-2) as *const v128);
    let r0_b = core::ptr::read_unaligned(lp.offset(2) as *const v128);
    let rp1_a = core::ptr::read_unaligned(lp.offset(wi - 2) as *const v128);
    let rp1_b = core::ptr::read_unaligned(lp.offset(wi + 2) as *const v128);
    local_noble_from_rows_4x(
        lp,
        wi,
        rm1_a,
        rm1_b,
        r0_a,
        r0_b,
        rp1_a,
        rp1_b,
        active_mask,
        vmin_c2,
    )
}

#[inline(always)]
unsafe fn nms_is_clear_ptr(bitmap: *const u64, x: usize, y: usize, words_per_row: usize) -> bool {
    let w = *bitmap.add(y * words_per_row + (x >> 6));
    (w & (1u64 << (x & 63))) == 0
}

#[inline(always)]
unsafe fn nms_mark_disk_ptr(
    bitmap: *mut u64,
    cx: usize,
    cy: usize,
    r_nms: usize,
    w: usize,
    h: usize,
    words_per_row: usize,
) {
    let r_is = (r_nms as isize) - 1;
    let y_lo = (cy as isize - r_is).max(0) as usize;
    let y_hi = (cy as isize + r_is).min(h as isize - 1) as usize;
    let x_hi_lim = (w - 1) as isize;
    if r_nms <= 3 {
        let dx_max = r_is;
        let x0 = (cx as isize - dx_max).max(0) as usize;
        let x1 = (cx as isize + dx_max).min(x_hi_lim) as usize;
        let w0 = x0 >> 6;
        let w1 = x1 >> 6;
        let b0 = x0 & 63;
        let b1 = x1 & 63;
        if w0 == w1 {
            let width = b1 - b0 + 1;
            let mask = ((1u64 << width) - 1) << b0;
            let mut yy = y_lo;
            while yy <= y_hi {
                *bitmap.add(yy * words_per_row + w0) |= mask;
                yy += 1;
            }
        } else {
            let mask0 = (!0u64) << b0;
            let mask1 = if b1 == 63 { !0u64 } else { (1u64 << (b1 + 1)) - 1 };
            let mut yy = y_lo;
            while yy <= y_hi {
                let row_off = yy * words_per_row;
                *bitmap.add(row_off + w0) |= mask0;
                *bitmap.add(row_off + w1) |= mask1;
                yy += 1;
            }
        }
        return;
    }

    const DX_R5: [isize; 5] = [4, 4, 4, 3, 2];
    let r2_limit = (r_nms * r_nms) as isize;
    let mut yy = y_lo;
    while yy <= y_hi {
        let ady = (yy as isize - cy as isize).abs();
        let dx_max = if r_nms == 5 {
            if ady < 5 {
                *DX_R5.as_ptr().add(ady as usize)
            } else {
                0
            }
        } else {
            let rem = r2_limit - 1 - ady * ady;
            wasm_floorf(wasm_sqrtf(rem.max(0) as f32)) as isize
        };
        let x0 = (cx as isize - dx_max).max(0) as usize;
        let x1 = (cx as isize + dx_max).min(x_hi_lim) as usize;
        let row_off = yy * words_per_row;
        let w0 = x0 >> 6;
        let w1 = x1 >> 6;
        let b0 = x0 & 63;
        let b1 = x1 & 63;
        if w0 == w1 {
            let width = b1 - b0 + 1;
            let mask = if width >= 64 {
                !0u64
            } else {
                ((1u64 << width) - 1) << b0
            };
            *bitmap.add(row_off + w0) |= mask;
        } else {
            let mask0 = (!0u64) << b0;
            let mask1 = if b1 == 63 {
                !0u64
            } else {
                (1u64 << (b1 + 1)) - 1
            };
            *bitmap.add(row_off + w0) |= mask0;
            *bitmap.add(row_off + w1) |= mask1;
        }
        yy += 1;
    }
}

#[inline(always)]
unsafe fn raw_heap_sift_down_ptr(
    scores: *mut f32,
    xy_packed: *mut u32,
    len: usize,
    mut idx: usize,
) {
    let sc = *scores.add(idx);
    let xy0 = *xy_packed.add(idx);
    loop {
        let left = idx * 2 + 1;
        if left >= len {
            break;
        }
        let right = left + 1;
        let child = if right < len && *scores.add(right) < *scores.add(left) {
            right
        } else {
            left
        };
        if *scores.add(child) >= sc {
            break;
        }
        *scores.add(idx) = *scores.add(child);
        *xy_packed.add(idx) = *xy_packed.add(child);
        idx = child;
    }
    *scores.add(idx) = sc;
    *xy_packed.add(idx) = xy0;
}

#[inline(never)]
unsafe fn raw_heap_push_overflow_ptr(
    scores: *mut f32,
    xy_packed: *mut u32,
    mut heap_state: u64,
    score: f32,
    xy: u32,
) -> u64 {
    if ((heap_state >> 16) & 1) == 0 {
        let mut k = MAX_RAW_PEAKS >> 1;
        while k > 0 {
            k -= 1;
            raw_heap_sift_down_ptr(scores, xy_packed, MAX_RAW_PEAKS, k);
        }
        heap_state |= 1u64 << 16;
    }
    let trunc = ((heap_state >> 32) as u32).saturating_add(1);
    heap_state = (heap_state & 0xFFFF_FFFF) | ((trunc as u64) << 32);
    if score > *scores {
        *scores = score;
        *xy_packed = xy;
        raw_heap_sift_down_ptr(scores, xy_packed, MAX_RAW_PEAKS, 0);
    }
    heap_state
}

#[inline(always)]
unsafe fn raw_heap_push_lazy_packed(
    scores: *mut f32,
    xy_packed: *mut u32,
    heap_state: u64,
    score: f32,
    xy: u32,
) -> u64 {
    let len = (heap_state & 0xFFFF) as usize;
    if len < MAX_RAW_PEAKS {
        *scores.add(len) = score;
        *xy_packed.add(len) = xy;
        (heap_state & !0xFFFFu64) | ((len + 1) as u64)
    } else if (heap_state & 0x1_0000) == 0 || score > *scores {
        raw_heap_push_overflow_ptr(scores, xy_packed, heap_state, score, xy)
    } else {
        let trunc = ((heap_state >> 32) as u32).saturating_add(1);
        (heap_state & 0xFFFF_FFFF) | ((trunc as u64) << 32)
    }
}

/// 2-pass LSD radix sort for `u16` keys, descending or ascending.
#[inline(never)]
fn radix_sort_u16_keys(
    keys: &[u16],
    n: usize,
    descending: bool,
    order: &mut [u16],
    tmp: &mut [u16],
    cnt: &mut [u16; 256],
) {
    assert!(n <= 65535 && keys.len() >= n && order.len() >= n && tmp.len() >= n);
    if n <= 1 {
        if n == 1 {
            order[0] = 0;
        }
        return;
    }

    let keys_p = keys.as_ptr();
    let order_p = order.as_mut_ptr();
    let tmp_p = tmp.as_mut_ptr();
    let cnt_p = cnt.as_mut_ptr();

    // SAFETY: Slice lengths `>= n`, `n <= 65535`, and `cnt.len() == 256` are guaranteed;
    // all byte indices stay in `0..256` and cumulative write positions stay in `0..n`.
    unsafe {
        // Pass 0: low byte directly from `keys[0..n]` (skips identity init + gather on `order`)
        core::ptr::write_bytes(cnt_p, 0, 256);
        let mut i = 0usize;
        while i + 2 <= n {
            let b0 = (*keys_p.add(i) & 0xFF) as usize;
            let b1 = (*keys_p.add(i + 1) & 0xFF) as usize;
            *cnt_p.add(b0) += 1;
            *cnt_p.add(b1) += 1;
            i += 2;
        }
        if i < n {
            let b = (*keys_p.add(i) & 0xFF) as usize;
            *cnt_p.add(b) += 1;
        }
        let mut acc = 0usize;
        if descending {
            let mut b = 256usize;
            while b > 0 {
                b -= 1;
                let c = *cnt_p.add(b) as usize;
                *cnt_p.add(b) = acc as u16;
                acc += c;
            }
        } else {
            let mut b = 0usize;
            while b < 256 {
                let c = *cnt_p.add(b) as usize;
                *cnt_p.add(b) = acc as u16;
                acc += c;
                b += 1;
            }
        }
        i = 0;
        while i < n {
            let b = (*keys_p.add(i) & 0xFF) as usize;
            let pos = *cnt_p.add(b) as usize;
            *cnt_p.add(b) = (pos + 1) as u16;
            *tmp_p.add(pos) = i as u16;
            i += 1;
        }

        // Pass 1: high byte
        core::ptr::write_bytes(cnt_p, 0, 256);
        i = 0;
        while i + 2 <= n {
            let idx0 = *tmp_p.add(i) as usize;
            let idx1 = *tmp_p.add(i + 1) as usize;
            let b0 = (*keys_p.add(idx0) >> 8) as usize;
            let b1 = (*keys_p.add(idx1) >> 8) as usize;
            *cnt_p.add(b0) += 1;
            *cnt_p.add(b1) += 1;
            i += 2;
        }
        if i < n {
            let idx = *tmp_p.add(i) as usize;
            let b = (*keys_p.add(idx) >> 8) as usize;
            *cnt_p.add(b) += 1;
        }
        acc = 0;
        if descending {
            let mut b = 256usize;
            while b > 0 {
                b -= 1;
                let c = *cnt_p.add(b) as usize;
                *cnt_p.add(b) = acc as u16;
                acc += c;
            }
        } else {
            let mut b = 0usize;
            while b < 256 {
                let c = *cnt_p.add(b) as usize;
                *cnt_p.add(b) = acc as u16;
                acc += c;
                b += 1;
            }
        }
        i = 0;
        while i < n {
            let idx = *tmp_p.add(i);
            let b = (*keys_p.add(idx as usize) >> 8) as usize;
            let pos = *cnt_p.add(b) as usize;
            *cnt_p.add(b) = (pos + 1) as u16;
            *order_p.add(pos) = idx;
            i += 1;
        }
    }
}

/// Samples a bilinear 8x8 patch around sub-pixel `(px, py)` (scalar builds scratch into `lk_buf`),
/// quantizes to zero-mean `[i8; 64]` in `dst_tpl8`, and returns 32-bit Haar-Morton code.
#[inline(always)]
fn extract_8x8_patch_and_sig_simd128(
    luma: &[f32],
    w: usize,
    px: f32,
    py: f32,
    rg: f32,
    by: f32,
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))] lk_buf: &mut [f32; 64],
    dst_tpl8: &mut [i8; 64],
) -> u32 {
    const HAAR_SPREAD: [u32; 16] = [
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54,
        0x55,
    ];
    let sx = px - 3.5;
    let sy = py - 3.5;
    let x0 = (sx.max(0.0) as u32) as usize;
    let y0 = (sy.max(0.0) as u32) as usize;
    let fx = sx - (x0 as f32);
    let fy = sy - (y0 as f32);

    // Chromatic bits [31..27] matching `simd_ops::has_opposite_chroma_pole`
    let sat2 = rg * rg + by * by;
    let is_chr = if sat2 > 0.018 { 1u32 } else { 0u32 };
    let rg_p = if rg > 0.055 {
        2u32
    } else if rg < -0.055 {
        1u32
    } else {
        0u32
    };
    let by_p = if by > 0.055 {
        2u32
    } else if by < -0.055 {
        1u32
    } else {
        0u32
    };

    #[cfg(target_arch = "wasm32")]
    unsafe {
        let vfx = f32x4_splat(fx);
        let vfy = f32x4_splat(fy);
        let vquarter = f32x4_splat(0.25);
        let mut vacc = f32x4_splat(0.0);
        let ptr = luma.as_ptr();

        let p_init = y0 * w + x0;
        let top0 = core::ptr::read_unaligned(ptr.add(p_init) as *const v128);
        let top1 = core::ptr::read_unaligned(ptr.add(p_init + 4) as *const v128);
        let top0_s = i32x4_shuffle::<1, 2, 3, 4>(top0, top1);
        let top1_s = core::ptr::read_unaligned(ptr.add(p_init + 5) as *const v128);
        let mut htop_a = f32x4_add(top0, f32x4_mul(vfx, f32x4_sub(top0_s, top0)));
        let mut htop_b = f32x4_add(top1, f32x4_mul(vfx, f32x4_sub(top1_s, top1)));

        macro_rules! interp_row {
            (@row $y_off:expr, $advance:tt) => {{
                let p1 = (y0 + $y_off + 1) * w + x0;
                let bot0 = core::ptr::read_unaligned(ptr.add(p1) as *const v128);
                let bot1 = core::ptr::read_unaligned(ptr.add(p1 + 4) as *const v128);
                let bot0_s = i32x4_shuffle::<1, 2, 3, 4>(bot0, bot1);
                let bot1_s = core::ptr::read_unaligned(ptr.add(p1 + 5) as *const v128);
                let hbot_a = f32x4_add(bot0, f32x4_mul(vfx, f32x4_sub(bot0_s, bot0)));
                let hbot_b = f32x4_add(bot1, f32x4_mul(vfx, f32x4_sub(bot1_s, bot1)));
                let pa = f32x4_add(htop_a, f32x4_mul(vfy, f32x4_sub(hbot_a, htop_a)));
                let pb = f32x4_add(htop_b, f32x4_mul(vfy, f32x4_sub(hbot_b, htop_b)));
                interp_row!(@adv $advance, hbot_a, hbot_b);
                vacc = f32x4_add(vacc, f32x4_add(pa, pb));
                (pa, pb)
            }};
            (@adv true, $ba:expr, $bb:expr) => {
                htop_a = $ba;
                htop_b = $bb;
            };
            (@adv false, $ba:expr, $bb:expr) => {};
        }

        macro_rules! m4_pair {
            ($ra0:expr, $rb0:expr, $ra1:expr, $rb1:expr) => {{
                let sl = f32x4_add($ra0, $ra1);
                let sr = f32x4_add($rb0, $rb1);
                let evens = i32x4_shuffle::<0, 2, 4, 6>(sl, sr);
                let odds = i32x4_shuffle::<1, 3, 5, 7>(sl, sr);
                f32x4_mul(f32x4_add(evens, odds), vquarter)
            }};
        }

        let (r0a, r0b) = interp_row!(@row 0, true);
        let (r1a, r1b) = interp_row!(@row 1, true);
        let m4_0 = m4_pair!(r0a, r0b, r1a, r1b);

        let (r2a, r2b) = interp_row!(@row 2, true);
        let (r3a, r3b) = interp_row!(@row 3, true);
        let m4_1 = m4_pair!(r2a, r2b, r3a, r3b);

        let (r4a, r4b) = interp_row!(@row 4, true);
        let (r5a, r5b) = interp_row!(@row 5, true);
        let m4_2 = m4_pair!(r4a, r4b, r5a, r5b);

        let (r6a, r6b) = interp_row!(@row 6, true);
        let (r7a, r7b) = interp_row!(@row 7, false);
        let m4_3 = m4_pair!(r6a, r6b, r7a, r7b);

        let mean = hsum_f32x4(vacc) * (1.0 / 64.0);
        let vmean = f32x4_splat(mean);
        let b0 = i32x4_bitmask(f32x4_gt(m4_0, vmean)) as u32;
        let b1 = i32x4_bitmask(f32x4_gt(m4_1, vmean)) as u32;
        let b2 = i32x4_bitmask(f32x4_gt(m4_2, vmean)) as u32;
        let b3 = i32x4_bitmask(f32x4_gt(m4_3, vmean)) as u32;
        let b16_bits = ((b0 & 3) | ((b0 & 12) << 2))
            | (((b1 & 3) | ((b1 & 12) << 2)) << 2)
            | (((b2 & 3) | ((b2 & 12) << 2)) << 8)
            | (((b3 & 3) | ((b3 & 12) << 2)) << 10);

        let top_p = f32x4_add(m4_0, m4_1);
        let bot_p = f32x4_add(m4_2, m4_3);
        let mid_p = f32x4_add(m4_1, m4_2);
        let out_tb = f32x4_add(m4_0, m4_3);

        // Quad sums [q00, q01, q10, q11] -> Scale-1 Haar bits via SIMD compare + spread LUT
        let q_vec = f32x4_mul(
            f32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(top_p, bot_p),
                i32x4_shuffle::<1, 3, 5, 7>(top_p, bot_p),
            ),
            vquarter,
        );
        let qa = i32x4_shuffle::<1, 3, 2, 3>(q_vec, q_vec); // [q01m, q11m, q10m, q11m]
        let qb = i32x4_shuffle::<0, 2, 0, 1>(q_vec, q_vec); // [q00m, q10m, q00m, q01m]
        let hdiff = f32x4_sub(qa, qb);
        let hpos = (i32x4_bitmask(f32x4_gt(hdiff, f32x4_splat(0.015))) as usize) & 15;
        let hneg = (i32x4_bitmask(f32x4_lt(hdiff, f32x4_splat(-0.015))) as usize) & 15;
        let haar_bits = HAAR_SPREAD[hpos] | (HAAR_SPREAD[hneg] << 1);

        let center_sum = f32x4_extract_lane::<1>(mid_p) + f32x4_extract_lane::<2>(mid_p);
        let outer_sum = hsum_f32x4(out_tb)
            + f32x4_extract_lane::<0>(mid_p)
            + f32x4_extract_lane::<3>(mid_p);
        let cm = center_sum * 0.25;
        let om = outer_sum * (1.0 / 12.0);
        let co_bits = if cm - om > 0.03 {
            6u32
        } else if om - cm > 0.03 {
            2u32
        } else {
            4u32
        };

        // Quantize directly from live NEON registers -> `[i8; 64]` with zero `lk_buf` round-trip
        let vscale = f32x4_splat(127.0);
        let vmin_i8 = i8x16_splat(-127);
        let dst_ptr = dst_tpl8.as_mut_ptr() as *mut v128;

        macro_rules! quant4 {
            ($idx:expr, $p0:expr, $p1:expr, $p2:expr, $p3:expr) => {{
                let r0 = f32x4_nearest(f32x4_mul(f32x4_sub($p0, vmean), vscale));
                let r1 = f32x4_nearest(f32x4_mul(f32x4_sub($p1, vmean), vscale));
                let r2 = f32x4_nearest(f32x4_mul(f32x4_sub($p2, vmean), vscale));
                let r3 = f32x4_nearest(f32x4_mul(f32x4_sub($p3, vmean), vscale));
                let i01 = i16x8_narrow_i32x4(i32x4_trunc_sat_f32x4(r0), i32x4_trunc_sat_f32x4(r1));
                let i23 = i16x8_narrow_i32x4(i32x4_trunc_sat_f32x4(r2), i32x4_trunc_sat_f32x4(r3));
                core::ptr::write_unaligned(
                    dst_ptr.add($idx),
                    i8x16_max(vmin_i8, i8x16_narrow_i16x8(i01, i23)),
                );
            }};
        }

        quant4!(0, r0a, r0b, r1a, r1b);
        quant4!(1, r2a, r2b, r3a, r3b);
        quant4!(2, r4a, r4b, r5a, r5b);
        quant4!(3, r6a, r6b, r7a, r7b);

        return (is_chr << 31)
            | (rg_p << 29)
            | (by_p << 27)
            | (co_bits << 24)
            | (haar_bits << 16)
            | b16_bits;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;
        let mut sum = 0.0f32;
        let mut r = 0usize;
        while r < 8 {
            let row0 = (y0 + r) * w + x0;
            let row1 = row0 + w;
            let mut c = 0usize;
            while c < 8 {
                let v = luma[row0 + c] * w00
                    + luma[row0 + c + 1] * w10
                    + luma[row1 + c] * w01
                    + luma[row1 + c + 1] * w11;
                lk_buf[r * 8 + c] = v;
                sum += v;
                c += 1;
            }
            r += 1;
        }
        let mean = sum * (1.0 / 64.0);

        let mut b16_bits = 0u32;
        let mut q00 = 0.0f32;
        let mut q01 = 0.0f32;
        let mut q10 = 0.0f32;
        let mut q11 = 0.0f32;
        let mut center_sum = 0.0f32;
        let mut outer_sum = 0.0f32;

        let mut by_i = 0usize;
        while by_i < 4 {
            let r0 = by_i * 2;
            let mut bx_i = 0usize;
            while bx_i < 4 {
                let c0 = bx_i * 2;
                let s4 = lk_buf[r0 * 8 + c0]
                    + lk_buf[r0 * 8 + c0 + 1]
                    + lk_buf[(r0 + 1) * 8 + c0]
                    + lk_buf[(r0 + 1) * 8 + c0 + 1];
                let m4 = s4 * 0.25;
                let z = ((by_i & 1) << 1) | (bx_i & 1) | ((by_i & 2) << 2) | ((bx_i & 2) << 1);
                if m4 > mean {
                    b16_bits |= 1u32 << z;
                }
                if by_i < 2 {
                    if bx_i < 2 {
                        q00 += m4;
                    } else {
                        q01 += m4;
                    }
                } else if bx_i < 2 {
                    q10 += m4;
                } else {
                    q11 += m4;
                }
                if (by_i == 1 || by_i == 2) && (bx_i == 1 || bx_i == 2) {
                    center_sum += m4;
                } else {
                    outer_sum += m4;
                }
                bx_i += 1;
            }
            by_i += 1;
        }

        let q00m = q00 * 0.25;
        let q01m = q01 * 0.25;
        let q10m = q10 * 0.25;
        let q11m = q11 * 0.25;
        let mut haar_bits = 0u32;
        if q01m - q00m > 0.015 {
            haar_bits |= 1 << 0;
        } else if q00m - q01m > 0.015 {
            haar_bits |= 1 << 1;
        }
        if q11m - q10m > 0.015 {
            haar_bits |= 1 << 2;
        } else if q10m - q11m > 0.015 {
            haar_bits |= 1 << 3;
        }
        if q10m - q00m > 0.015 {
            haar_bits |= 1 << 4;
        } else if q00m - q10m > 0.015 {
            haar_bits |= 1 << 5;
        }
        if q11m - q01m > 0.015 {
            haar_bits |= 1 << 6;
        } else if q01m - q11m > 0.015 {
            haar_bits |= 1 << 7;
        }

        let cm = center_sum * 0.25;
        let om = outer_sum * (1.0 / 12.0);
        let co_bits = if cm - om > 0.03 {
            6u32
        } else if om - cm > 0.03 {
            2u32
        } else {
            4u32
        };

        let mut i = 0usize;
        while i < 64 {
            let q = clamp_f32((lk_buf[i] - mean) * 127.0, -127.0, 127.0);
            let r = if q >= 0.0 { q + 0.5 } else { q - 0.5 };
            dst_tpl8[i] = r as i8;
            i += 1;
        }

        let _ = HAAR_SPREAD[0];
        (is_chr << 31)
            | (rg_p << 29)
            | (by_p << 27)
            | (co_bits << 24)
            | (haar_bits << 16)
            | b16_bits
    }
}

/// Samples opponent chroma `(rg, by)` directly from the `2x2` block around `(ix, iy)` in `rgba`.
#[inline(always)]
fn sample_chroma_2x2_rgba(rgba: &[u8], w: usize, ix: usize, iy: usize) -> (f32, f32) {
    let idx00 = (iy * w + ix) * 4;
    let idx01 = idx00 + w * 4;
    debug_assert!(idx01 + 8 <= rgba.len());

    #[cfg(target_arch = "wasm32")]
    unsafe {
        let rp = rgba.as_ptr();
        let p0 = core::ptr::read_unaligned(rp.add(idx00) as *const u64);
        let p1 = core::ptr::read_unaligned(rp.add(idx01) as *const u64);
        let row01 = u64x2(p0, p1);
        let lo16 = u16x8_extend_low_u8x16(row01);
        let hi16 = u16x8_extend_high_u8x16(row01);
        let sum02_13 = u16x8_add(lo16, hi16);
        let p_hi = i32x4_shuffle::<2, 3, 2, 3>(sum02_13, sum02_13);
        let sum16 = u16x8_add(sum02_13, p_hi);
        let sum32 = u32x4_extend_low_u16x8(sum16);
        let rgba_f = f32x4_mul(f32x4_convert_u32x4(sum32), f32x4_splat(0.25 / 255.0));
        let r = f32x4_extract_lane::<0>(rgba_f);
        let g = f32x4_extract_lane::<1>(rgba_f);
        let b = f32x4_extract_lane::<2>(rgba_f);
        return (r - g, 0.5 * (r + g) - b);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let idx10 = idx00 + 4;
        let idx11 = idx01 + 4;
        let r = (rgba[idx00] as f32 + rgba[idx10] as f32 + rgba[idx01] as f32 + rgba[idx11] as f32)
            * (0.25 / 255.0);
        let g = (rgba[idx00 + 1] as f32
            + rgba[idx10 + 1] as f32
            + rgba[idx01 + 1] as f32
            + rgba[idx11 + 1] as f32)
            * (0.25 / 255.0);
        let b = (rgba[idx00 + 2] as f32
            + rgba[idx10 + 2] as f32
            + rgba[idx01 + 2] as f32
            + rgba[idx11 + 2] as f32)
            * (0.25 / 255.0);
        (r - g, 0.5 * (r + g) - b)
    }
}

/// 1D quadratic parabolic sub-pixel offset `[-0.5, +0.5]` when `v0` is a true extremum.
#[inline(always)]
fn parabolic_1d(vm: f32, v0: f32, vp: f32) -> f32 {
    let denom = vm - 2.0 * v0 + vp;
    if denom.abs() > 1e-5 && denom.is_sign_negative() {
        clamp_f32(0.5 * (vm - vp) / denom, -0.5, 0.5)
    } else {
        0.0
    }
}

#[inline(never)]
fn refine_peak_subpixel_noble_packed(luma: &[f32], w: usize, ix: usize, iy: usize) -> u64 {
    // Polygon vertex / non-extremum shape: refine on 2D cornerness
    #[cfg(target_arch = "wasm32")]
    let (cl, cc, cr) = if ix >= 3 && ix + 5 <= w && iy >= 2 && (iy + 3) * w <= luma.len() {
        unsafe {
            let vc = f32x4_sqrt(local_noble_corner_4x(
                luma.as_ptr(),
                w,
                ix - 1,
                iy,
                i32x4(-1, -1, -1, 0),
                f32x4_splat(0.0),
            ));
            (
                f32x4_extract_lane::<0>(vc),
                f32x4_extract_lane::<1>(vc),
                f32x4_extract_lane::<2>(vc),
            )
        }
    } else {
        (
            local_noble_corner(luma, w, ix - 1, iy).0,
            local_noble_corner(luma, w, ix, iy).0,
            local_noble_corner(luma, w, ix + 1, iy).0,
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (cl, cc, cr) = (
        local_noble_corner(luma, w, ix - 1, iy).0,
        local_noble_corner(luma, w, ix, iy).0,
        local_noble_corner(luma, w, ix + 1, iy).0,
    );
    let (ct, _) = local_noble_corner(luma, w, ix, iy - 1);
    let (cb, _) = local_noble_corner(luma, w, ix, iy + 1);
    let px = ix as f32 + parabolic_1d(cl, cc, cr);
    let py = iy as f32 + parabolic_1d(ct, cc, cb);
    (px.to_bits() as u64) | ((py.to_bits() as u64) << 32)
}

/// 3x3 sub-pixel parabolic refinement on `luma` extremum (blobs/birds/speckles) or
/// local `corner` field (polygon vertices).
#[inline(always)]
fn refine_peak_subpixel(luma: &[f32], w: usize, ix: usize, iy: usize) -> (f32, f32) {
    debug_assert!(ix >= 1 && ix + 1 < w && iy >= 1 && (iy + 2) * w <= luma.len());
    let (vc, vl, vr, vt, vb, vtl, vtr, vbl, vbr) = unsafe {
        let lp = luma.as_ptr().add(iy * w + ix);
        let wi = w as isize;
        (
            *lp,
            *lp.offset(-1),
            *lp.offset(1),
            *lp.offset(-wi),
            *lp.offset(wi),
            *lp.offset(-wi - 1),
            *lp.offset(-wi + 1),
            *lp.offset(wi - 1),
            *lp.offset(wi + 1),
        )
    };

    let is_luma_max = vc >= vl
        && vc >= vr
        && vc >= vt
        && vc >= vb
        && vc >= vtl
        && vc >= vtr
        && vc >= vbl
        && vc >= vbr;
    let is_luma_min = vc <= vl
        && vc <= vr
        && vc <= vt
        && vc <= vb
        && vc <= vtl
        && vc <= vtr
        && vc <= vbl
        && vc <= vbr;

    if is_luma_max || is_luma_min {
        let sign = if is_luma_max { 1.0f32 } else { -1.0f32 };
        let dx = 2.0 * vc - vl - vr;
        let dy = 2.0 * vc - vt - vb;
        let sx = dx * sign;
        let sy = dy * sign;
        let ox = if sx > 1.0e-5 {
            clamp_f32((0.5 * (vr - vl) * sign) / sx, -0.5, 0.5)
        } else {
            0.0
        };
        let oy = if sy > 1.0e-5 {
            clamp_f32((0.5 * (vb - vt) * sign) / sy, -0.5, 0.5)
        } else {
            0.0
        };
        return (ix as f32 + ox, iy as f32 + oy);
    }

    let packed = refine_peak_subpixel_noble_packed(luma, w, ix, iy);
    (
        f32::from_bits(packed as u32),
        f32::from_bits((packed >> 32) as u32),
    )
}

/// Ultra-lightweight Auto-Density coarse cell occupancy check (`~0.3 us` via SIMD128 over
/// `cc_energy[..880]`). Used when `swarm_mode == 1 && swarm_active == 0`.
#[inline(always)]
fn count_occupied_cc_cells_simd128(cc_energy: &[f32]) -> u32 {
    let n = cc_energy.len().min(880);
    if n == 0 {
        return 0;
    }
    let thresh = 0.01f32;
    let mut occ = 0u32;
    let mut i = 0usize;
    #[cfg(target_arch = "wasm32")]
    unsafe {
        let vth = f32x4_splat(thresh);
        let ptr = cc_energy.as_ptr();
        let mut vsum = i32x4_splat(0);
        while i + 16 <= n {
            let m0 = f32x4_gt(core::ptr::read_unaligned(ptr.add(i) as *const v128), vth);
            let m1 = f32x4_gt(core::ptr::read_unaligned(ptr.add(i + 4) as *const v128), vth);
            let m2 = f32x4_gt(core::ptr::read_unaligned(ptr.add(i + 8) as *const v128), vth);
            let m3 = f32x4_gt(core::ptr::read_unaligned(ptr.add(i + 12) as *const v128), vth);
            vsum = i32x4_sub(vsum, i32x4_add(i32x4_add(m0, m1), i32x4_add(m2, m3)));
            i += 16;
            if i == 704 || (n <= 240 && i == 32) {
                let partial = (i32x4_extract_lane::<0>(vsum)
                    + i32x4_extract_lane::<1>(vsum)
                    + i32x4_extract_lane::<2>(vsum)
                    + i32x4_extract_lane::<3>(vsum)) as u32;
                if partial + ((n - i) as u32) < AUTO_OCC_CELLS_PROBE {
                    return partial;
                }
            }
        }
        while i + 4 <= n {
            vsum = i32x4_sub(
                vsum,
                f32x4_gt(core::ptr::read_unaligned(ptr.add(i) as *const v128), vth),
            );
            i += 4;
        }
        occ += (i32x4_extract_lane::<0>(vsum)
            + i32x4_extract_lane::<1>(vsum)
            + i32x4_extract_lane::<2>(vsum)
            + i32x4_extract_lane::<3>(vsum)) as u32;
    }
    while i < n {
        if cc_energy[i] > thresh {
            occ += 1;
        }
        i += 1;
    }
    occ
}

/// Cheap pre-gate for `Auto` probe frames (`~0.015 ms`): scans ~20 evenly spaced interior rows
/// and confirms at least 10 thresholded pixels exhibit crisp 2D Noble cornerness before running
/// the full-frame extractor, preventing heavy peak-extraction work on smooth low-corner scenes.
#[inline(never)]
fn auto_probe_corner_pregate(
    luma: &[f32],
    bg_luma: &[f32],
    eca_res: &[f32],
    w: usize,
    h: usize,
    pan_mag: f32,
) -> bool {
    if w < 16 || h < 16 || luma.len() < w * h || bg_luma.len() < w * h || eca_res.len() < w * h {
        return false;
    }
    let q = (w as f32) * (1.0 / 640.0);
    if pan_mag > 1.8 * q {
        return false;
    }
    let is_hd = w > 400;
    let k = if is_hd { 4usize } else { 2usize };
    let row_step = if is_hd { 16usize } else { 8usize };
    let col_step = if is_hd { 6usize } else { 4usize };
    let min_dark = if is_hd { 40u32 } else { 18u32 };
    let kw = k * w;
    let y0 = k + 2;
    let y1 = h.saturating_sub(k + 2);
    let x0 = k + 2;
    let x1 = w.saturating_sub(k + 2);
    if y0 >= y1 || x0 >= x1 {
        return false;
    }

    let mut dark_mov = 0u32;
    let mut bright_mov = 0u32;
    let mut y = y0;
    unsafe {
        let ep = eca_res.as_ptr();
        let lp = luma.as_ptr();
        while y < y1 {
            let row = y * w;
            let mut x = x0;
            while x < x1 {
                let i = row + x;
                if (*ep.add(i)).abs() > 0.015 {
                    let c = *lp.add(i);
                    let dl = *lp.add(i - k) - c;
                    let dr = *lp.add(i + k) - c;
                    let du = *lp.add(i - kw) - c;
                    let dd = *lp.add(i + kw) - c;
                    let min_d = dl.min(dr).min(du).min(dd);
                    if min_d > 0.043 {
                        dark_mov += 1;
                    } else {
                        let min_b = (-dl).min(-dr).min(-du).min(-dd);
                        if min_b > 0.043 {
                            bright_mov += 1;
                        }
                    }
                }
                x += col_step;
            }
            y += row_step;
        }
    }
    let any_mov = dark_mov + bright_mov;
    dark_mov >= min_dark && dark_mov * 4 >= any_mov * 3
}

#[inline(always)]
fn find_best_probe_peak(
    ms: &MicroSwarmState,
    ti: usize,
    px: f32,
    py: f32,
    rad2: f32,
    sad_max_u32: u32,
    x_lo: usize,
    x_hi: usize,
    y_lo: usize,
    y_hi: usize,
) -> i32 {
    let mut best_pi = -1i32;
    let mut best_sad = u32::MAX;
    let mut best_d2 = f32::MAX;

    let mut cy = y_lo;
    while cy <= y_hi {
        let row = cy * MICRO_CELL_W;
        let mut cx = x_lo;
        while cx <= x_hi {
            let mut p_idx = ms.cell_head[row + cx];
            while p_idx >= 0 {
                let p = p_idx as usize;
                if ms.peak_matched[p] == 0 {
                    let dx = ms.peak_x[p] - px;
                    let dy = ms.peak_y[p] - py;
                    let d2 = dx * dx + dy * dy;
                    if d2 <= rad2 {
                        let sad =
                            patch_8x8_sad_i8_simd128(&ms.tracks.tpl8_i8[ti], &ms.peak_tpl8[p]);
                        if sad <= sad_max_u32
                            && (sad < best_sad || (sad == best_sad && d2 < best_d2))
                        {
                            best_sad = sad;
                            best_d2 = d2;
                            best_pi = p as i32;
                        }
                    }
                }
                p_idx = ms.peak_next[p];
            }
            cx += 1;
        }
        cy += 1;
    }
    best_pi
}

#[inline(never)]
fn eval_probe_shift(
    ms: &mut MicroSwarmState,
    probe_n: usize,
    off_x: f32,
    off_y: f32,
    rad: f32,
    rad2: f32,
    sad_max_u32: u32,
    inv_cw: f32,
    inv_ch: f32,
) -> usize {
    ms.peak_matched[..ms.num_peaks].fill(0);
    let mut matched = 0usize;
    let mut ti = 0usize;
    while ti < probe_n {
        let px = ms.tracks.pos_x[ti] + off_x;
        let py = ms.tracks.pos_y[ti] + off_y;
        if px.is_finite() && py.is_finite() {
            let x_lo = cell_coord(px - rad, inv_cw, MICRO_CELL_W - 1);
            let x_hi = cell_coord(px + rad, inv_cw, MICRO_CELL_W - 1);
            let y_lo = cell_coord(py - rad, inv_ch, MICRO_CELL_H - 1);
            let y_hi = cell_coord(py + rad, inv_ch, MICRO_CELL_H - 1);
            let best_pi =
                find_best_probe_peak(ms, ti, px, py, rad2, sad_max_u32, x_lo, x_hi, y_lo, y_hi);
            if best_pi >= 0 {
                ms.peak_matched[best_pi as usize] = 1;
                matched += 1;
            }
        }
        ti += 1;
    }
    matched
}

/// Evaluates local flow-vector coherence inside `r_assoc = 6.0 * scale`: allows non-rigid
/// vortices (`Lamb-Oseen`), rotating bodies, and multi-cluster swarms while rejecting uniform
/// random clutter (`<= 8%` vs `>= 50%` required) by requiring `>= 3` spatial neighbors (`<= 36*scale`)
/// whose displacement vector agrees within `<= 1.8 * scale`. Uses idle `tracks`/`track_cell_head`.
#[inline(never)]
fn eval_probe_local_consensus(
    ms: &mut MicroSwarmState,
    probe_n: usize,
    pan_dx_f: f32,
    pan_dy_f: f32,
    scale: f32,
    sad_max_u32: u32,
    inv_cw: f32,
    inv_ch: f32,
) -> usize {
    let r_assoc = 6.0 * scale;
    let r_assoc2 = r_assoc * r_assoc;
    let r_nbr2 = (36.0 * scale) * (36.0 * scale);
    let r_coh2 = (1.8 * scale) * (1.8 * scale);

    ms.peak_matched[..ms.num_peaks].fill(0);
    ms.track_cell_head.fill(-1);
    ms.track_next[..probe_n].fill(-1);
    ms.tracks.regime[..probe_n].fill(0);

    let mut ti = 0usize;
    while ti < probe_n {
        let sx = ms.tracks.pos_x[ti];
        let sy = ms.tracks.pos_y[ti];
        let px = sx + pan_dx_f;
        let py = sy + pan_dy_f;
        if px.is_finite() && py.is_finite() {
            let cx0 = cell_coord(sx, inv_cw, MICRO_CELL_W - 1);
            let cy0 = cell_coord(sy, inv_ch, MICRO_CELL_H - 1);
            let cell = cy0 * MICRO_CELL_W + cx0;
            ms.track_next[ti] = ms.track_cell_head[cell];
            ms.track_cell_head[cell] = ti as i16;

            let x_lo = cell_coord(px - r_assoc, inv_cw, MICRO_CELL_W - 1);
            let x_hi = cell_coord(px + r_assoc, inv_cw, MICRO_CELL_W - 1);
            let y_lo = cell_coord(py - r_assoc, inv_ch, MICRO_CELL_H - 1);
            let y_hi = cell_coord(py + r_assoc, inv_ch, MICRO_CELL_H - 1);
            let best_pi =
                find_best_probe_peak(ms, ti, px, py, r_assoc2, sad_max_u32, x_lo, x_hi, y_lo, y_hi);
            if best_pi >= 0 {
                let p = best_pi as usize;
                ms.peak_matched[p] = 1;
                ms.tracks.regime[ti] = 1;
                ms.tracks.vel_x[ti] = ms.peak_x[p] - px;
                ms.tracks.vel_y[ti] = ms.peak_y[p] - py;
            }
        }
        ti += 1;
    }

    let mut coherent = 0usize;
    ti = 0;
    while ti < probe_n {
        if ms.tracks.regime[ti] != 0 {
            let sx = ms.tracks.pos_x[ti];
            let sy = ms.tracks.pos_y[ti];
            let dx_i = ms.tracks.vel_x[ti];
            let dy_i = ms.tracks.vel_y[ti];
            let cx0 = cell_coord(sx, inv_cw, MICRO_CELL_W - 1) as isize;
            let cy0 = cell_coord(sy, inv_ch, MICRO_CELL_H - 1) as isize;
            let x_lo = (cx0 - 1).max(0) as usize;
            let x_hi = (cx0 + 1).min((MICRO_CELL_W - 1) as isize) as usize;
            let y_lo = (cy0 - 1).max(0) as usize;
            let y_hi = (cy0 + 1).min((MICRO_CELL_H - 1) as isize) as usize;

            let mut agree = 0u32;
            let mut cy = y_lo;
            while cy <= y_hi && agree < 3 {
                let row = cy * MICRO_CELL_W;
                let mut cx = x_lo;
                while cx <= x_hi && agree < 3 {
                    let mut idx = ms.track_cell_head[row + cx];
                    while idx >= 0 {
                        let tk = idx as usize;
                        idx = ms.track_next[tk];
                        if tk != ti && ms.tracks.regime[tk] != 0 {
                            let nx = ms.tracks.pos_x[tk] - sx;
                            let ny = ms.tracks.pos_y[tk] - sy;
                            if nx * nx + ny * ny <= r_nbr2 {
                                let vx = ms.tracks.vel_x[tk] - dx_i;
                                let vy = ms.tracks.vel_y[tk] - dy_i;
                                if vx * vx + vy * vy <= r_coh2 {
                                    agree += 1;
                                    if agree >= 3 {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    cx += 1;
                }
                cy += 1;
            }
            if agree >= 3 {
                coherent += 1;
            }
        }
        ti += 1;
    }

    ms.tracks.regime[..probe_n].fill(0);
    ms.tracks.vel_x[..probe_n].fill(0.0);
    ms.tracks.vel_y[..probe_n].fill(0.0);
    coherent
}

/// Checks genuine 2-frame spatial + `8x8` patch SAD peak correspondence for `Auto` entry (`mode=1`).
/// Compensates caller pan `(pan_dx_f, pan_dy_f)` and estimates a dominant coherent bulk shift
/// within `+-10*q` via idle `radix_cnt` `16x16` shift histogram before enforcing strict `2.5*q`
/// radius and `SAD <= 0.30` 1-to-1 matching (`>= max(AUTO_ENTRY_PEAKS / 2, 50% of probe_n)`).
#[inline(never)]
fn probe_temporal_correspondence(
    ms: &mut MicroSwarmState,
    w: usize,
    h: usize,
    pan_dx_f: f32,
    pan_dy_f: f32,
) -> bool {
    let n = ms.num_peaks;
    let store_n = if n < MAX_MICRO { n } else { MAX_MICRO };

    if ms.probe_streak == 0 || ms.probe_peak_count == 0 {
        let mut i = 0usize;
        while i < store_n {
            ms.tracks.pos_x[i] = ms.peak_x[i];
            ms.tracks.pos_y[i] = ms.peak_y[i];
            ms.tracks.curr_sig[i] = ms.peak_sig[i];
            ms.tracks.tpl8_i8[i].copy_from_slice(&ms.peak_tpl8[i]);
            i += 1;
        }
        ms.probe_peak_count = store_n as u16;
        return false;
    }

    let probe_n = (ms.probe_peak_count as usize).min(MAX_MICRO);
    if probe_n < AUTO_ENTRY_PEAKS || n < AUTO_ENTRY_PEAKS {
        let mut i = 0usize;
        while i < store_n {
            ms.tracks.pos_x[i] = ms.peak_x[i];
            ms.tracks.pos_y[i] = ms.peak_y[i];
            ms.tracks.curr_sig[i] = ms.peak_sig[i];
            ms.tracks.tpl8_i8[i].copy_from_slice(&ms.peak_tpl8[i]);
            i += 1;
        }
        ms.probe_peak_count = store_n as u16;
        return false;
    }

    // Hash Frame-2 extracted peaks directly into uniform `32x18` cells
    ms.cell_head[..MICRO_CELLS].fill(-1);
    let inv_cw = (MICRO_CELL_W as f32) / (w as f32);
    let inv_ch = (MICRO_CELL_H as f32) / (h as f32);
    let mut pi = 0usize;
    while pi < n {
        let cx = cell_coord(ms.peak_x[pi], inv_cw, MICRO_CELL_W - 1);
        let cy = cell_coord(ms.peak_y[pi], inv_ch, MICRO_CELL_H - 1);
        let cell = cy * MICRO_CELL_W + cx;
        ms.peak_next[pi] = ms.cell_head[cell];
        ms.cell_head[cell] = pi as i16;
        pi += 1;
    }

    let q = (w as f32) * (1.0 / 640.0);
    let scale = if q > 0.5 { q } else { 0.5 };
    let rad = AUTO_CORR_RADIUS_Q * scale;
    let rad2 = rad * rad;
    let sad_max_u32 = (AUTO_CORR_SAD_MAX * (64.0 * 64.0)) as u32;

    let mut matched = eval_probe_shift(
        ms,
        probe_n,
        pan_dx_f,
        pan_dy_f,
        rad,
        rad2,
        sad_max_u32,
        inv_cw,
        inv_ch,
    );

    if !(matched >= (AUTO_ENTRY_PEAKS / 2) && matched * 10 >= probe_n * 5) {
        // Estimate coherent residual bulk shift within `+-AUTO_CORR_SHIFT_MAX_Q * scale`
        // using a `16x16` quantized displacement histogram stored inside idle `radix_cnt[0..256]`.
        let shift_max = AUTO_CORR_SHIFT_MAX_Q * scale;
        let inv_bin = 8.0 / shift_max;
        ms.radix_cnt.fill(0);

        let mut ti = 0usize;
        while ti < probe_n {
            let px = ms.tracks.pos_x[ti] + pan_dx_f;
            let py = ms.tracks.pos_y[ti] + pan_dy_f;
            if px.is_finite() && py.is_finite() {
                let x_lo = cell_coord(px - shift_max, inv_cw, MICRO_CELL_W - 1);
                let x_hi = cell_coord(px + shift_max, inv_cw, MICRO_CELL_W - 1);
                let y_lo = cell_coord(py - shift_max, inv_ch, MICRO_CELL_H - 1);
                let y_hi = cell_coord(py + shift_max, inv_ch, MICRO_CELL_H - 1);

                let mut cy = y_lo;
                while cy <= y_hi {
                    let row = cy * MICRO_CELL_W;
                    let mut cx = x_lo;
                    while cx <= x_hi {
                        let mut p_idx = ms.cell_head[row + cx];
                        while p_idx >= 0 {
                            let p = p_idx as usize;
                            let dx = ms.peak_x[p] - px;
                            let dy = ms.peak_y[p] - py;
                            if dx.abs() <= shift_max && dy.abs() <= shift_max {
                                let sad = patch_8x8_sad_i8_simd128(
                                    &ms.tracks.tpl8_i8[ti],
                                    &ms.peak_tpl8[p],
                                );
                                if sad <= sad_max_u32 {
                                    let bx = ((dx + shift_max) * inv_bin) as isize;
                                    let by = ((dy + shift_max) * inv_bin) as isize;
                                    let ux = bx.clamp(0, 15) as usize;
                                    let uy = by.clamp(0, 15) as usize;
                                    let bin = (uy << 4) | ux;
                                    ms.radix_cnt[bin] = ms.radix_cnt[bin].saturating_add(1);
                                }
                            }
                            p_idx = ms.peak_next[p];
                        }
                        cx += 1;
                    }
                    cy += 1;
                }
            }
            ti += 1;
        }

        let mut best_bin = 0usize;
        let mut best_cnt = 0u16;
        let mut b = 0usize;
        while b < 256 {
            let c = ms.radix_cnt[b];
            if c > best_cnt {
                best_cnt = c;
                best_bin = b;
            }
            b += 1;
        }

        if best_cnt >= 16 {
            let best_bx = (best_bin & 15) as isize;
            let best_by = (best_bin >> 4) as isize;
            let mut sum_dx = 0.0f32;
            let mut sum_dy = 0.0f32;
            let mut cnt = 0u32;

            ti = 0;
            while ti < probe_n {
                let px = ms.tracks.pos_x[ti] + pan_dx_f;
                let py = ms.tracks.pos_y[ti] + pan_dy_f;
                if px.is_finite() && py.is_finite() {
                    let x_lo = cell_coord(px - shift_max, inv_cw, MICRO_CELL_W - 1);
                    let x_hi = cell_coord(px + shift_max, inv_cw, MICRO_CELL_W - 1);
                    let y_lo = cell_coord(py - shift_max, inv_ch, MICRO_CELL_H - 1);
                    let y_hi = cell_coord(py + shift_max, inv_ch, MICRO_CELL_H - 1);

                    let mut cy = y_lo;
                    while cy <= y_hi {
                        let row = cy * MICRO_CELL_W;
                        let mut cx = x_lo;
                        while cx <= x_hi {
                            let mut p_idx = ms.cell_head[row + cx];
                            while p_idx >= 0 {
                                let p = p_idx as usize;
                                let dx = ms.peak_x[p] - px;
                                let dy = ms.peak_y[p] - py;
                                if dx.abs() <= shift_max && dy.abs() <= shift_max {
                                    let ux = (((dx + shift_max) * inv_bin) as isize).clamp(0, 15);
                                    let uy = (((dy + shift_max) * inv_bin) as isize).clamp(0, 15);
                                    if (ux - best_bx).abs() <= 1 && (uy - best_by).abs() <= 1 {
                                        let sad = patch_8x8_sad_i8_simd128(
                                            &ms.tracks.tpl8_i8[ti],
                                            &ms.peak_tpl8[p],
                                        );
                                        if sad <= sad_max_u32 {
                                            sum_dx += dx;
                                            sum_dy += dy;
                                            cnt += 1;
                                        }
                                    }
                                }
                                p_idx = ms.peak_next[p];
                            }
                            cx += 1;
                        }
                        cy += 1;
                    }
                }
                ti += 1;
            }

            if cnt > 0 {
                let inv_cnt = 1.0 / (cnt as f32);
                let sx = pan_dx_f + clamp_f32(sum_dx * inv_cnt, -shift_max, shift_max);
                let sy = pan_dy_f + clamp_f32(sum_dy * inv_cnt, -shift_max, shift_max);
                let m_shift = eval_probe_shift(
                    ms,
                    probe_n,
                    sx,
                    sy,
                    rad,
                    rad2,
                    sad_max_u32,
                    inv_cw,
                    inv_ch,
                );
                if m_shift > matched {
                    matched = m_shift;
                }
            }
        }
    }

    if !(matched >= (AUTO_ENTRY_PEAKS / 2) && matched * 10 >= probe_n * 5) {
        let m_cons = eval_probe_local_consensus(
            ms,
            probe_n,
            pan_dx_f,
            pan_dy_f,
            scale,
            sad_max_u32,
            inv_cw,
            inv_ch,
        );
        if m_cons > matched {
            matched = m_cons;
        }
    }

    if matched >= (AUTO_ENTRY_PEAKS / 2) && matched * 10 >= probe_n * 5 {
        ms.probe_peak_count = 0;
        true
    } else {
        // Refresh probe snapshot with Frame-2 peaks for subsequent candidate pair
        let mut i = 0usize;
        while i < store_n {
            ms.tracks.pos_x[i] = ms.peak_x[i];
            ms.tracks.pos_y[i] = ms.peak_y[i];
            ms.tracks.curr_sig[i] = ms.peak_sig[i];
            ms.tracks.tpl8_i8[i].copy_from_slice(&ms.peak_tpl8[i]);
            i += 1;
        }
        ms.probe_peak_count = store_n as u16;
        false
    }
}

/// Extracts up to `MAX_MICRO_PEAKS = 2048` fine sub-pixel peaks from `luma` / `eca_res` / `bg_luma`.
/// Combines foreground moving blobs and subtle vibrational/textured corners, maintains a
/// 4096-entry global min-heap over 3x3 local maxima (via idle CSR scratch), applies exact
/// Euclidean NMS disk (`r_nms = 3` HD / `2` SD), and computes sub-pixel refinement + 8x8 i8
/// patch + 32-bit Haar-Morton + 16-bit spatial Morton Z-code ONLY for survivors.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn extract_micro_peaks_simd128(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    bg_luma: &[f32],
    eca_res: &[f32],
    rgba: &[u8],
    w: usize,
    h: usize,
    pan_mag: f32,
) {
    let raw_heap_len = extract_phase1a_raw(ms, luma, bg_luma, eca_res, w, h, pan_mag);
    extract_phase1b_sort(ms, raw_heap_len);
    extract_phase1c_patches(ms, luma, rgba, w, h, raw_heap_len);
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_cand_4px_rows(
    sr_row: *mut f32,
    px: usize,
    lp_q: *const f32,
    wi: isize,
    va: v128,
    vd: v128,
    e: v128,
    d: v128,
    min_c_eca_2: f32,
    min_c_trem_2: f32,
    rm1_a: v128,
    rm1_b: v128,
    r0_a: v128,
    r0_b: v128,
    rp1_a: v128,
    rp1_b: v128,
) -> bool {
    let v_maxfin = f32x4_splat(1.0e30);
    let fin_a = f32x4_lt(va, v_maxfin);
    let fin_d = f32x4_lt(vd, v_maxfin);
    let eca = v128_and(e, fin_a);
    let m = v128_or(eca, v128_and(d, fin_d));
    let vmin_c2 = v128_bitselect(
        f32x4_splat(min_c_eca_2),
        f32x4_splat(min_c_trem_2),
        eca,
    );
    let vzero = f32x4_splat(0.0);
    let vnoble = local_noble_from_rows_4x(
        lp_q,
        wi,
        rm1_a,
        rm1_b,
        r0_a,
        r0_b,
        rp1_a,
        rp1_b,
        m,
        vmin_c2,
    );
    let ok_c = v128_and(m, f32x4_gt(vnoble, vzero));
    if !v128_any_true(ok_c) {
        return false;
    }
    let diff = f32x4_pmax(
        v128_and(va, fin_a),
        f32x4_mul(v128_and(vd, fin_d), f32x4_splat(0.85)),
    );
    let g = f32x4_add(f32x4_splat(0.35), f32x4_mul(f32x4_splat(1.50), diff));
    let vgain_sq = f32x4_mul(g, g);
    let cand_s2 = f32x4_mul(vnoble, vgain_sq);
    let valid = v128_and(ok_c, f32x4_le(cand_s2, v_maxfin));
    if !v128_any_true(valid) {
        return false;
    }

    core::ptr::write_unaligned(sr_row.add(px) as *mut v128, v128_and(cand_s2, valid));
    true
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_cand_4px(
    lum_base: *const f32,
    sr_row: *mut f32,
    w: usize,
    px: usize,
    y: usize,
    va: v128,
    vd: v128,
    e: v128,
    d: v128,
    min_c_eca_2: f32,
    min_c_trem_2: f32,
) -> bool {
    let wi = w as isize;
    let lp_q = lum_base.offset((y as isize) * wi + (px as isize));
    let rm1_a = core::ptr::read_unaligned(lp_q.offset(-wi - 2) as *const v128);
    let rm1_b = core::ptr::read_unaligned(lp_q.offset(-wi + 2) as *const v128);
    let r0_a = core::ptr::read_unaligned(lp_q.offset(-2) as *const v128);
    let r0_b = core::ptr::read_unaligned(lp_q.offset(2) as *const v128);
    let rp1_a = core::ptr::read_unaligned(lp_q.offset(wi - 2) as *const v128);
    let rp1_b = core::ptr::read_unaligned(lp_q.offset(wi + 2) as *const v128);
    eval_cand_4px_rows(
        sr_row,
        px,
        lp_q,
        wi,
        va,
        vd,
        e,
        d,
        min_c_eca_2,
        min_c_trem_2,
        rm1_a,
        rm1_b,
        r0_a,
        r0_b,
        rp1_a,
        rp1_b,
    )
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_cand_scalar_range(
    luma: &[f32],
    bg_base: *const f32,
    eca_base: *const f32,
    sr_p: *mut f32,
    lum_p: *const f32,
    w: usize,
    x_start: usize,
    x_stop: usize,
    y: usize,
    adaptive_floor: f32,
    tremor_floor: f32,
    min_c_eca: f32,
    min_c_trem: f32,
) -> u32 {
    let base_c = x_start >> 2;
    let mut hit_mask = 0u32;
    let row_off = y * w;
    let mut px = x_start;
    while px < x_stop {
        let idx = row_off + px;
        let e_val = (*eca_base.add(idx)).abs();
        let t_val = (*lum_p.add(px) - *bg_base.add(idx)).abs();
        if (e_val.is_finite() && e_val > adaptive_floor)
            || (t_val.is_finite() && t_val > tremor_floor)
        {
            let (corner, _) = local_noble_corner(luma, w, px, y);
            let min_c = if e_val > adaptive_floor {
                min_c_eca
            } else {
                min_c_trem
            };
            if corner.is_finite() && corner > min_c {
                let diff_s = if e_val.is_finite() && e_val > t_val * 0.85 {
                    e_val
                } else {
                    t_val * 0.85
                };
                let cand_s = corner * (0.35 + 1.50 * diff_s);
                let cand_s2 = cand_s * cand_s;
                if cand_s2.is_finite() {
                    *sr_p.add(px) = cand_s2;
                    let off = (px >> 2) - base_c;
                    if off < 32 {
                        hit_mask |= 1u32 << off;
                    }
                }
            }
        }
        px += 1;
    }
    hit_mask
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn extract_phase1a_3x3_max(
    sp: *const f32,
    raw_scores: *mut f32,
    raw_xy: *mut u32,
    mut heap_state: u64,
    r_top: usize,
    r_mid: usize,
    r_bot: usize,
    ym1: usize,
    x_end: usize,
    m0: u64,
    m1: u64,
    m2: u64,
) -> u64 {
    let x_limit = x_end - 1;
    let y_hi_bits = ((ym1 as u16) as u32) << 16;
    #[cfg(target_arch = "wasm32")]
    let mut vfloor = f32x4_splat(if (heap_state & 0x1_0000) != 0 { *raw_scores } else { 0.0 });
    let mut w_idx = 0usize;
    while w_idx < 3 {
        let mut bits = match w_idx {
            0 => m0,
            1 => m1,
            _ => m2,
        };
        let base_x = w_idx << 8;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let cx4 = base_x + (b << 2);

            #[cfg(target_arch = "wasm32")]
            {
                let p_mid = sp.add(r_mid + cx4);
                let vs = core::ptr::read_unaligned(p_mid as *const v128);
                let above_floor = f32x4_gt(vs, vfloor);
                if !v128_any_true(above_floor) {
                    continue;
                }
                let m_l = core::ptr::read_unaligned(p_mid.sub(1) as *const v128);
                let m_r = core::ptr::read_unaligned(p_mid.add(1) as *const v128);
                let mid_win = v128_and(above_floor, v128_and(f32x4_ge(vs, m_l), f32x4_gt(vs, m_r)));
                if v128_any_true(mid_win) {
                    let p_top = sp.add(r_top + cx4);
                    let top_l = core::ptr::read_unaligned(p_top.sub(1) as *const v128);
                    let top_r = core::ptr::read_unaligned(p_top.add(1) as *const v128);
                    let top_c = i32x4_shuffle::<1, 2, 5, 6>(top_l, top_r);
                    let top_max = f32x4_pmax(f32x4_pmax(top_l, top_r), top_c);
                    let top_win = v128_and(mid_win, f32x4_ge(vs, top_max));
                    if v128_any_true(top_win) {
                        let p_bot = sp.add(r_bot + cx4);
                        let bot_l = core::ptr::read_unaligned(p_bot.sub(1) as *const v128);
                        let bot_r = core::ptr::read_unaligned(p_bot.add(1) as *const v128);
                        let bot_c = i32x4_shuffle::<1, 2, 5, 6>(bot_l, bot_r);
                        let bot_max = f32x4_pmax(f32x4_pmax(bot_l, bot_r), bot_c);

                        let win = v128_and(top_win, f32x4_gt(vs, bot_max));
                        let mut mm = i32x4_bitmask(win);
                        let any_hit = mm != 0;
                        while mm != 0 {
                            let lane = mm.trailing_zeros() as usize;
                            mm &= mm - 1;
                            let cx = cx4 + lane;
                            if cx >= 6 && cx < x_limit {
                                let s0 = *p_mid.add(lane);
                                heap_state = raw_heap_push_lazy_packed(
                                    raw_scores,
                                    raw_xy,
                                    heap_state,
                                    s0,
                                    ((cx as u16) as u32) | y_hi_bits,
                                );
                            }
                        }
                        if any_hit && (heap_state & 0x1_0000) != 0 {
                            vfloor = f32x4_splat(*raw_scores);
                        }
                    }
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut lane = 0usize;
                while lane < 4 {
                    let cx = cx4 + lane;
                    if cx >= 6 && cx < x_limit {
                        let s0 = *sp.add(r_mid + cx);
                        if s0.is_finite()
                            && s0 > 0.0
                            && s0 >= *sp.add(r_mid + cx - 1)
                            && s0 > *sp.add(r_mid + cx + 1)
                            && s0 >= *sp.add(r_top + cx - 1)
                            && s0 >= *sp.add(r_top + cx)
                            && s0 >= *sp.add(r_top + cx + 1)
                            && s0 > *sp.add(r_bot + cx - 1)
                            && s0 > *sp.add(r_bot + cx)
                            && s0 > *sp.add(r_bot + cx + 1)
                        {
                            heap_state = raw_heap_push_lazy_packed(
                                raw_scores,
                                raw_xy,
                                heap_state,
                                s0,
                                ((cx as u16) as u32) | y_hi_bits,
                            );
                        }
                    }
                    lane += 1;
                }
            }
        }
        w_idx += 1;
    }

    heap_state
}

#[inline(never)]
pub fn extract_phase1a_raw(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    bg_luma: &[f32],
    eca_res: &[f32],
    w: usize,
    h: usize,
    pan_mag: f32,
) -> usize {
    const _: () = assert!(
        SCORE_RING_STRIDE <= 768
            && SCORE_RING_STRIDE % 16 == 0
            && SCORE_RING_ROWS == 3
            && core::mem::offset_of!(MicroSwarmState, csr_peak_idx) % 4 == 0
    );
    let words_per_row = (w + 63) >> 6;
    assert!(
        w >= 16
            && w <= SCORE_RING_STRIDE
            && h >= 16
            && h <= 360
            && luma.len() >= w * h
            && bg_luma.len() >= w * h
            && eca_res.len() >= w * h
            && ms.nms_bitmap.len() >= h * words_per_row
            && ms.csr_cost.len() >= MAX_RAW_PEAKS
            && ms.csr_peak_idx.len() >= MAX_RAW_PEAKS * 2
            && ms.score_ring.len() >= SCORE_RING_ROWS * SCORE_RING_STRIDE
    );
    let q = (w as f32) * (1.0 / 640.0);
    let pan_norm = clamp_f32(pan_mag / q.max(0.5), 0.0, 2.0);
    let sat_active = ms.swarm_active != 0
        && ((ms.confirmed_count >= 1020 && ms.num_peaks >= 1420) || ms.num_peaks >= 1650);
    let sat_boost = if sat_active { 0.012f32 } else { 0.0f32 };
    let adaptive_floor =
        (if w <= 320 { 0.048f32 } else { 0.045f32 })
            + 0.025f32 * clamp_f32(pan_norm - 0.25, 0.0, 1.5)
            + sat_boost;
    let tremor_floor = 0.028f32 + 0.040f32 * pan_norm + 0.5f32 * sat_boost;

    // Clear NMS bitmap for active resolution; `score_ring` rows clear lazily by row/16-px tile mask
    let bmp_clear_len = h * words_per_row;
    unsafe {
        core::ptr::write_bytes(ms.nms_bitmap.as_mut_ptr(), 0, bmp_clear_len);
    }

    let mut heap_state = 0u64;
    let raw_scores_p = ms.csr_cost.as_mut_ptr();
    let raw_xy_p = ms.csr_peak_idx.as_mut_ptr() as *mut u32;
    let sr_base = ms.score_ring.as_mut_ptr();
    let lum_base = luma.as_ptr();
    let bg_base = bg_luma.as_ptr();
    let eca_base = eca_res.as_ptr();

    // Scan rows `5..h-5`, keeping a rotating 3-row score ring and exact 4-px chunk bitmaps
    // `(m0..m3: 256 chunks = 1024 px)` so 3x3 local-max tests and ring clears skip empty chunks.
    let x_end = w - 5;
    let y_end = h - 5;
    let mut top_m0 = 0u64;
    let mut top_m1 = 0u64;
    let mut top_m2 = 0u64;
    let mut mid_m0 = 0u64;
    let mut mid_m1 = 0u64;
    let mut mid_m2 = 0u64;
    let mut cur_m0 = 0u64;
    let mut cur_m1 = 0u64;
    let mut cur_m2 = 0u64;
    let mut top_ring_row = 0usize;
    let mut mid_ring_row = 1usize;
    let mut cur_ring_row = 2usize;

    let min_c_eca = if sat_active { 0.018f32 } else { 0.014f32 };
    let min_c_trem = if w <= 320 || sat_active {
        0.025f32
    } else {
        0.022f32
    };
    let min_c_eca_2 = min_c_eca * min_c_eca;
    let min_c_trem_2 = min_c_trem * min_c_trem;
    #[cfg(target_arch = "wasm32")]
    let (vadapt, vtrem) = (f32x4_splat(adaptive_floor), f32x4_splat(tremor_floor));

    let mut y = 5usize;
    while y < y_end {
        let cur_off = cur_ring_row * SCORE_RING_STRIDE;
        let row_off = y * w;
        let mut cm0 = 0u64;
        let mut cm1 = 0u64;
        let mut cm2 = 0u64;

        macro_rules! mark_chunk {
            ($px:expr) => {{
                let px = $px;
                let bit = 1u64 << ((px >> 2) & 63);
                if px < 256 {
                    cm0 |= bit;
                } else if px < 512 {
                    cm1 |= bit;
                } else {
                    cm2 |= bit;
                }
            }};
        }

        if y < 8 {
            unsafe {
                core::ptr::write_bytes(sr_base.add(cur_off), 0, w);
            }
        } else {
            let dirty = cur_m0.count_ones() + cur_m1.count_ones() + cur_m2.count_ones();
            if dirty != 0 {
                #[cfg(target_arch = "wasm32")]
                if dirty <= 16 {
                    unsafe {
                        let vzero_sr = f32x4_splat(0.0);
                        let sr_clr = sr_base.add(cur_off);
                        let mut wi = 0usize;
                        while wi < 3 {
                            let mut m = match wi {
                                0 => cur_m0,
                                1 => cur_m1,
                                _ => cur_m2,
                            };
                            let base_x = wi << 8;
                            while m != 0 {
                                let b = m.trailing_zeros() as usize;
                                m &= m - 1;
                                core::ptr::write_unaligned(
                                    sr_clr.add(base_x + (b << 2)) as *mut v128,
                                    vzero_sr,
                                );
                            }
                            wi += 1;
                        }
                    }
                } else {
                    unsafe {
                        core::ptr::write_bytes(sr_base.add(cur_off), 0, w);
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                unsafe {
                    core::ptr::write_bytes(sr_base.add(cur_off), 0, w);
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        let x = unsafe {
            let eca_p = eca_base.add(row_off);
            let lum_p = lum_base.add(row_off);
            let bg_p = bg_base.add(row_off);
            let sr_p = sr_base.add(cur_off);

            let bx_end = 8usize.min(x_end);
            if 5 < bx_end
                && eval_cand_scalar_range(
                    luma,
                    bg_base,
                    eca_base,
                    sr_p,
                    lum_p,
                    w,
                    5,
                    bx_end,
                    y,
                    adaptive_floor,
                    tremor_floor,
                    min_c_eca,
                    min_c_trem,
                ) != 0
            {
                cm0 |= 1u64 << 1;
            }

            let mut x = bx_end;
            let wi = w as isize;
            let row_lp = lum_base.offset((y as isize) * wi);
            while x + 8 <= x_end {
                let va0 = f32x4_abs(core::ptr::read_unaligned(eca_p.add(x) as *const v128));
                let vd0 = f32x4_abs(f32x4_sub(
                    core::ptr::read_unaligned(lum_p.add(x) as *const v128),
                    core::ptr::read_unaligned(bg_p.add(x) as *const v128),
                ));
                let ee0 = f32x4_gt(va0, vadapt);
                let dd0 = f32x4_gt(vd0, vtrem);
                let hit0 = v128_or(ee0, dd0);

                let x1 = x + 4;
                let va1 = f32x4_abs(core::ptr::read_unaligned(eca_p.add(x1) as *const v128));
                let vd1 = f32x4_abs(f32x4_sub(
                    core::ptr::read_unaligned(lum_p.add(x1) as *const v128),
                    core::ptr::read_unaligned(bg_p.add(x1) as *const v128),
                ));
                let ee1 = f32x4_gt(va1, vadapt);
                let dd1 = f32x4_gt(vd1, vtrem);
                let hit1 = v128_or(ee1, dd1);

                if v128_any_true(v128_or(hit0, hit1)) {
                    let mut run0 = v128_any_true(hit0);
                    let mut do_next = v128_any_true(hit1);
                    let mut has_prev = false;
                    let mut px_q = x;
                    let mut lp_q = row_lp.add(x);
                    let mut cva = va0;
                    let mut cvd = vd0;
                    let mut cee = ee0;
                    let mut cdd = dd0;
                    let mut rm1_b = ee0;
                    let mut r0_b = ee0;
                    let mut rp1_b = ee0;
                    loop {
                        if run0 {
                            let (rm1_a, r0_a, rp1_a) = if has_prev {
                                (rm1_b, r0_b, rp1_b)
                            } else {
                                (
                                    core::ptr::read_unaligned(lp_q.offset(-wi - 2) as *const v128),
                                    core::ptr::read_unaligned(lp_q.offset(-2) as *const v128),
                                    core::ptr::read_unaligned(lp_q.offset(wi - 2) as *const v128),
                                )
                            };
                            rm1_b = core::ptr::read_unaligned(lp_q.offset(-wi + 2) as *const v128);
                            r0_b = core::ptr::read_unaligned(lp_q.offset(2) as *const v128);
                            rp1_b = core::ptr::read_unaligned(lp_q.offset(wi + 2) as *const v128);
                            if eval_cand_4px_rows(
                                sr_p,
                                px_q,
                                lp_q,
                                wi,
                                cva,
                                cvd,
                                cee,
                                cdd,
                                min_c_eca_2,
                                min_c_trem_2,
                                rm1_a,
                                rm1_b,
                                r0_a,
                                r0_b,
                                rp1_a,
                                rp1_b,
                            ) {
                                mark_chunk!(px_q);
                            }
                            has_prev = true;
                        }
                        if !do_next {
                            break;
                        }
                        do_next = false;
                        run0 = true;
                        px_q = x1;
                        lp_q = row_lp.add(x1);
                        cva = va1;
                        cvd = vd1;
                        cee = ee1;
                        cdd = dd1;
                    }
                }
                x += 8;
            }
            if x + 4 <= x_end {
                let va = f32x4_abs(core::ptr::read_unaligned(eca_p.add(x) as *const v128));
                let vd = f32x4_abs(f32x4_sub(
                    core::ptr::read_unaligned(lum_p.add(x) as *const v128),
                    core::ptr::read_unaligned(bg_p.add(x) as *const v128),
                ));
                let ee = f32x4_gt(va, vadapt);
                let dd = f32x4_gt(vd, vtrem);
                if v128_any_true(v128_or(ee, dd))
                    && eval_cand_4px(
                        lum_base,
                        sr_p,
                        w,
                        x,
                        y,
                        va,
                        vd,
                        ee,
                        dd,
                        min_c_eca_2,
                        min_c_trem_2,
                    )
                {
                    mark_chunk!(x);
                }
                x += 4;
            }
            x
        };
        #[cfg(not(target_arch = "wasm32"))]
        let x = {
            let mut x = 5usize;
            while x + 4 <= x_end {
                let mut mask = 0i32;
                let mut k = 0;
                while k < 4 {
                    if unsafe { *eca_base.add(row_off + x + k) }.abs() > adaptive_floor
                        || (unsafe { *lum_base.add(row_off + x + k) - *bg_base.add(row_off + x + k) })
                            .abs()
                            > tremor_floor
                    {
                        mask |= 1 << k;
                    }
                    k += 1;
                }
                if mask != 0 {
                    let mut lane = 0usize;
                    while lane < 4 {
                        if (mask & (1 << lane)) != 0 {
                            let px = x + lane;
                            let idx = row_off + px;
                            let e_val = unsafe { *eca_base.add(idx) }.abs();
                            let t_val = unsafe { *lum_base.add(idx) - *bg_base.add(idx) }.abs();
                            let (corner, _) = local_noble_corner(luma, w, px, y);
                            let min_c = if e_val > adaptive_floor {
                                min_c_eca
                            } else {
                                min_c_trem
                            };
                            if corner.is_finite() && corner > min_c {
                                let diff_s = if e_val.is_finite() && e_val > t_val * 0.85 {
                                    e_val
                                } else {
                                    t_val * 0.85
                                };
                                let cand_s = corner * (0.35 + 1.50 * diff_s);
                                let cand_s2 = cand_s * cand_s;
                                if cand_s2.is_finite() {
                                    unsafe {
                                        *sr_base.add(cur_off + px) = cand_s2;
                                    }
                                    mark_chunk!(px);
                                }
                            }
                        }
                        lane += 1;
                    }
                }
                x += 4;
            }
            x
        };
        if x < x_end {
            let base_c = x >> 2;
            let mut hm = unsafe {
                eval_cand_scalar_range(
                    luma,
                    bg_base,
                    eca_base,
                    sr_base.add(cur_off),
                    lum_base.add(row_off),
                    w,
                    x,
                    x_end,
                    y,
                    adaptive_floor,
                    tremor_floor,
                    min_c_eca,
                    min_c_trem,
                )
            };
            while hm != 0 {
                let off = hm.trailing_zeros() as usize;
                hm &= hm - 1;
                let ch = base_c + off;
                let bit = 1u64 << (ch & 63);
                if ch < 64 {
                    cm0 |= bit;
                } else if ch < 128 {
                    cm1 |= bit;
                } else {
                    cm2 |= bit;
                }
            }
        }
        cur_m0 = cm0;
        cur_m1 = cm1;
        cur_m2 = cm2;

        // Test previous row `y - 1` for strict 3x3 local maxima across dirty 4-px chunks only
        if y >= 7 && (mid_m0 | mid_m1 | mid_m2) != 0 {
            unsafe {
                heap_state = extract_phase1a_3x3_max(
                    sr_base,
                    raw_scores_p,
                    raw_xy_p,
                    heap_state,
                    top_ring_row * SCORE_RING_STRIDE,
                    mid_ring_row * SCORE_RING_STRIDE,
                    cur_off,
                    y - 1,
                    x_end,
                    mid_m0,
                    mid_m1,
                    mid_m2,
                );
            }
        }
        let nt0 = mid_m0;
        mid_m0 = cur_m0;
        cur_m0 = top_m0;
        top_m0 = nt0;
        let nt1 = mid_m1;
        mid_m1 = cur_m1;
        cur_m1 = top_m1;
        top_m1 = nt1;
        let nt2 = mid_m2;
        mid_m2 = cur_m2;
        cur_m2 = top_m2;
        top_m2 = nt2;

        let next_top = mid_ring_row;
        mid_ring_row = cur_ring_row;
        cur_ring_row = top_ring_row;
        top_ring_row = next_top;
        y += 1;
    }
    ms.raw_heap_overflow_cnt = (heap_state >> 32) as u32;
    (heap_state & 0xFFFF) as usize
}

#[inline(never)]
pub fn extract_phase1b_sort(ms: &mut MicroSwarmState, raw_heap_len: usize) {
    assert!(raw_heap_len <= MAX_RAW_PEAKS);
    assert!(ms.csr_cost.len() >= MAX_RAW_PEAKS);
    assert!(ms.raw_sort_order.len() >= MAX_RAW_PEAKS);
    assert!(ms.raw_sort_tmp.len() >= MAX_RAW_PEAKS);
    assert!(ms.radix_cnt.len() >= 256);

    if raw_heap_len == 1 {
        ms.raw_sort_order[0] = 0;
        return;
    } else if raw_heap_len == 0 {
        return;
    }

    let scores_p = ms.csr_cost.as_ptr();
    let order_p = ms.raw_sort_order.as_mut_ptr();
    let tmp_p = ms.raw_sort_tmp.as_mut_ptr();
    let cnt_p = ms.radix_cnt.as_mut_ptr();

    // Sort `raw_heap_len <= 4096` raw local maxima descending by high 16 bits of positive IEEE-754 float bits.
    // SAFETY: `raw_heap_len <= MAX_RAW_PEAKS`, all slot indices inserted into `tmp_p`/`order_p` are
    // `< raw_heap_len <= MAX_RAW_PEAKS`, and radix byte indices are `< 256`.
    unsafe {
        core::ptr::write_bytes(cnt_p, 0, 256);
        let mut i = 0usize;
        while i + 2 <= raw_heap_len {
            let k0 = ((*scores_p.add(i)).to_bits() >> 16) as u16;
            let k1 = ((*scores_p.add(i + 1)).to_bits() >> 16) as u16;
            *cnt_p.add((k0 & 0xFF) as usize) += 1;
            *cnt_p.add((k1 & 0xFF) as usize) += 1;
            i += 2;
        }
        if i < raw_heap_len {
            let key = ((*scores_p.add(i)).to_bits() >> 16) as u16;
            *cnt_p.add((key & 0xFF) as usize) += 1;
        }
        let mut acc = 0usize;
        let mut b = 256usize;
        while b > 0 {
            b -= 1;
            let c = *cnt_p.add(b) as usize;
            *cnt_p.add(b) = acc as u16;
            acc += c;
        }
        i = 0;
        while i < raw_heap_len {
            let key = ((*scores_p.add(i)).to_bits() >> 16) as u16;
            let cslot = cnt_p.add((key & 0xFF) as usize);
            let pos = *cslot as usize;
            *cslot = (pos + 1) as u16;
            *tmp_p.add(pos) = i as u16;
            i += 1;
        }
        core::ptr::write_bytes(cnt_p, 0, 256);
        i = 0;
        while i + 2 <= raw_heap_len {
            let idx0 = *tmp_p.add(i) as usize;
            let idx1 = *tmp_p.add(i + 1) as usize;
            let k0 = ((*scores_p.add(idx0)).to_bits() >> 16) as u16;
            let k1 = ((*scores_p.add(idx1)).to_bits() >> 16) as u16;
            *cnt_p.add((k0 >> 8) as usize) += 1;
            *cnt_p.add((k1 >> 8) as usize) += 1;
            i += 2;
        }
        if i < raw_heap_len {
            let idx = *tmp_p.add(i) as usize;
            let key = ((*scores_p.add(idx)).to_bits() >> 16) as u16;
            *cnt_p.add((key >> 8) as usize) += 1;
        }
        acc = 0;
        b = 256;
        while b > 0 {
            b -= 1;
            let c = *cnt_p.add(b) as usize;
            *cnt_p.add(b) = acc as u16;
            acc += c;
        }
        i = 0;
        while i < raw_heap_len {
            let idx = *tmp_p.add(i);
            let key = ((*scores_p.add(idx as usize)).to_bits() >> 16) as u16;
            let cslot = cnt_p.add((key >> 8) as usize);
            let pos = *cslot as usize;
            *cslot = (pos + 1) as u16;
            *order_p.add(pos) = idx;
            i += 1;
        }

        // Exact IEEE-754 float tie-break within equal upper-16-bit buckets
        let mut run_start = 0usize;
        while run_start < raw_heap_len {
            let bucket = ((*scores_p.add(*order_p.add(run_start) as usize)).to_bits() >> 16) as u16;
            let mut run_end = run_start + 1;
            while run_end < raw_heap_len
                && ((*scores_p.add(*order_p.add(run_end) as usize)).to_bits() >> 16) as u16
                    == bucket
            {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            if run_len > 16 {
                // O(B) lower-16-bit descending radix pass for dense same-top-16 buckets
                core::ptr::write_bytes(cnt_p, 0, 256);
                let mut j = run_start;
                while j < run_end {
                    let lo = ((*scores_p.add(*order_p.add(j) as usize)).to_bits() & 0xFF) as usize;
                    *cnt_p.add(lo) += 1;
                    j += 1;
                }
                let mut acc2 = 0usize;
                let mut b2 = 256usize;
                while b2 > 0 {
                    b2 -= 1;
                    let c = *cnt_p.add(b2) as usize;
                    *cnt_p.add(b2) = acc2 as u16;
                    acc2 += c;
                }
                j = run_start;
                while j < run_end {
                    let idx = *order_p.add(j);
                    let lo = ((*scores_p.add(idx as usize)).to_bits() & 0xFF) as usize;
                    let cslot = cnt_p.add(lo);
                    let pos = *cslot as usize;
                    *cslot = (pos + 1) as u16;
                    *tmp_p.add(pos) = idx;
                    j += 1;
                }
                core::ptr::write_bytes(cnt_p, 0, 256);
                j = 0;
                while j < run_len {
                    let hi = (((*scores_p.add(*tmp_p.add(j) as usize)).to_bits() >> 8) & 0xFF)
                        as usize;
                    *cnt_p.add(hi) += 1;
                    j += 1;
                }
                acc2 = 0;
                b2 = 256;
                while b2 > 0 {
                    b2 -= 1;
                    let c = *cnt_p.add(b2) as usize;
                    *cnt_p.add(b2) = acc2 as u16;
                    acc2 += c;
                }
                j = 0;
                while j < run_len {
                    let idx = *tmp_p.add(j);
                    let hi = (((*scores_p.add(idx as usize)).to_bits() >> 8) & 0xFF) as usize;
                    let cslot = cnt_p.add(hi);
                    let pos = *cslot as usize;
                    *cslot = (pos + 1) as u16;
                    *order_p.add(run_start + pos) = idx;
                    j += 1;
                }
            } else if run_len > 1 {
                let mut j = run_start + 1;
                while j < run_end {
                    let key_idx = *order_p.add(j);
                    let key_val = *scores_p.add(key_idx as usize);
                    let mut k = j;
                    while k > run_start && *scores_p.add(*order_p.add(k - 1) as usize) < key_val {
                        *order_p.add(k) = *order_p.add(k - 1);
                        k -= 1;
                    }
                    *order_p.add(k) = key_idx;
                    j += 1;
                }
            }
            run_start = run_end;
        }
    }
}

#[inline(never)]
pub fn extract_phase1c_patches(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    rgba: &[u8],
    w: usize,
    h: usize,
    raw_heap_len: usize,
) {
    let words_per_row = (w + 63) >> 6;
    assert!(
        w >= 16
            && w <= SCORE_RING_STRIDE
            && h >= 16
            && h <= 360
            && luma.len() >= w * h
            && rgba.len() >= w * h * 4
            && raw_heap_len <= MAX_RAW_PEAKS
            && ms.raw_sort_order.len() >= MAX_RAW_PEAKS
            && ms.csr_peak_idx.len() >= MAX_RAW_PEAKS * 2
            && ms.csr_cost.len() >= MAX_RAW_PEAKS
            && ms.nms_bitmap.len() >= h * words_per_row
            && ms.cell_head.len() >= h
            && ms.peak_x.len() >= MAX_MICRO_PEAKS
            && ms.peak_y.len() >= MAX_MICRO_PEAKS
            && ms.peak_energy.len() >= MAX_MICRO_PEAKS
            && ms.peak_matched.len() >= MAX_MICRO_PEAKS
            && ms.peak_morton.len() >= MAX_MICRO_PEAKS
            && ms.peak_order.len() >= MAX_MICRO_PEAKS
            && ms.peak_next.len() >= MAX_MICRO_PEAKS
            && ms.peak_rg.len() >= MAX_MICRO_PEAKS
            && ms.peak_by.len() >= MAX_MICRO_PEAKS
            && ms.peak_sig.len() >= MAX_MICRO_PEAKS
            && ms.peak_tpl8.len() >= MAX_MICRO_PEAKS
    );
    let r_nms = if w > 320 { R_NMS_HD } else { R_NMS_SD };
    let y_end = h - 5;

    // Apply Euclidean NMS disk & populate final <= 2048 peaks with sub-pixel + 8x8 patch + signatures
    let eff_r_nms = if w <= 320 && raw_heap_len >= 1200 {
        3usize
    } else {
        r_nms
    };
    let inv_w_255 = 255.0 / (w as f32);
    let inv_h_255 = 255.0 / (h as f32);

    unsafe {
        let sort_p = ms.raw_sort_order.as_ptr();
        let csr_xy_p = ms.csr_peak_idx.as_ptr() as *const u32;
        let csr_cost_p = ms.csr_cost.as_ptr();
        let bmp_p = ms.nms_bitmap.as_mut_ptr();
        let ch_p = ms.cell_head.as_mut_ptr();
        let mut r = 5usize;
        while r < y_end {
            *ch_p.add(r) = -1;
            r += 1;
        }

        let px_p = ms.peak_x.as_mut_ptr();
        let py_p = ms.peak_y.as_mut_ptr();
        let pe_p = ms.peak_energy.as_mut_ptr();
        let pm_p = ms.peak_matched.as_mut_ptr();
        let pmort_p = ms.peak_morton.as_mut_ptr();
        let pord_p = ms.peak_order.as_mut_ptr();
        let pnext_p = ms.peak_next.as_mut_ptr();
        let prg_p = ms.peak_rg.as_mut_ptr();
        let pby_p = ms.peak_by.as_mut_ptr();
        let psig_p = ms.peak_sig.as_mut_ptr();
        let ptpl_p = ms.peak_tpl8.as_mut_ptr();

    let mut n_peaks = 0usize;
    let peak_cap = if ms.swarm_active != 0 && ms.confirmed_count >= 1030 && raw_heap_len >= 1950 {
        1664usize
    } else if ms.swarm_active != 0 && ms.active_count >= 1024 && raw_heap_len >= 2200 {
        1792usize
    } else {
        MAX_MICRO_PEAKS
    };
    let mut r_idx = 0usize;
    while r_idx < raw_heap_len && n_peaks < peak_cap {
            let raw_slot = *sort_p.add(r_idx) as usize;
            if raw_slot < MAX_RAW_PEAKS {
                let xy = *csr_xy_p.add(raw_slot);
                let ix = (xy & 0xFFFF) as usize;
                let iy = (xy >> 16) as usize;
                if ix >= 5 && ix + 5 < w && iy >= 5 && iy + 5 < h
                    && nms_is_clear_ptr(bmp_p, ix, iy, words_per_row)
                {
                    let sc = *csr_cost_p.add(raw_slot);
                    let (px, py) = refine_peak_subpixel(luma, w, ix, iy);
                    if px >= 0.0 && px < 1.0e30 && py >= 0.0 && py < 1.0e30 {
                        nms_mark_disk_ptr(bmp_p, ix, iy, eff_r_nms, w, h, words_per_row);
                        *px_p.add(n_peaks) = px;
                        *py_p.add(n_peaks) = py;
                        *pe_p.add(n_peaks) = sc;
                        *pm_p.add(n_peaks) = 0;
                        *pmort_p.add(n_peaks) = morton16_xy(px, py, inv_w_255, inv_h_255);
                        *pord_p.add(n_peaks) = ix as u16;
                        *pnext_p.add(n_peaks) = *ch_p.add(iy);
                        *ch_p.add(iy) = n_peaks as i16;
                        n_peaks += 1;
                    }
                }
            }
            r_idx += 1;
        }

        // Extract chroma & 8x8 templates in top-to-bottom scanline order for `rgba`/`luma` cache locality
        let mut ry = 5usize;
        while ry < y_end {
            let mut p_idx = *ch_p.add(ry);
            while p_idx >= 0 {
                let k = p_idx as usize;
                p_idx = *pnext_p.add(k);
                let ix = *pord_p.add(k) as usize;
                let px = *px_p.add(k);
                let py = *py_p.add(k);
                let (rg, by) = sample_chroma_2x2_rgba(rgba, w, ix, ry);
                *prg_p.add(k) = rg;
                *pby_p.add(k) = by;
                *psig_p.add(k) = extract_8x8_patch_and_sig_simd128(
                    luma,
                    w,
                    px,
                    py,
                    rg,
                    by,
                    &mut ms.lk_tpl_f,
                    &mut *ptpl_p.add(k),
                );
            }
            ry += 1;
        }

        ms.num_peaks = n_peaks;
    }
}

/// 2-pass Morton radix sort of extracted peaks + build `64x36` contiguous cell-sorted arrays.
#[inline(never)]
pub fn build_peak_morton_cell_hash(ms: &mut MicroSwarmState, w: usize, h: usize) {
    let n = ms.num_peaks.min(MAX_MICRO_PEAKS);
    assert!(ms.cell_off.len() >= CSR_CELLS + 2);
    if n == 0 {
        unsafe {
            core::ptr::write_bytes(ms.cell_off.as_mut_ptr(), 0, CSR_CELLS + 2);
        }
        ms.cell_head.fill(-1);
        return;
    }
    assert!(ms.peak_order.len() >= n && ms.peak_x.len() >= n && ms.peak_y.len() >= n);
    assert!(ms.peak_next.len() >= n);
    assert!(
        ms.cell_px.len() >= n && ms.cell_py.len() >= n && ms.cell_pidx.len() >= n
    );

    radix_sort_u16_keys(
        &ms.peak_morton,
        n,
        false,
        &mut ms.peak_order,
        &mut ms.peak_sort_tmp,
        &mut ms.radix_cnt,
    );

    let inv_cw = (CSR_CELL_W as f32) / (w as f32);
    let inv_ch = (CSR_CELL_H as f32) / (h as f32);

    let pord_p = ms.peak_order.as_ptr();
    let px_p = ms.peak_x.as_ptr();
    let py_p = ms.peak_y.as_ptr();
    let pnext_p = ms.peak_next.as_mut_ptr();
    let coff_p = ms.cell_off.as_mut_ptr();
    let cpx_p = ms.cell_px.as_mut_ptr();
    let cpy_p = ms.cell_py.as_mut_ptr();
    let cidx_p = ms.cell_pidx.as_mut_ptr();

    // SAFETY: `n <= MAX_MICRO_PEAKS`, all indices in `peak_order[0..n]` are `< n`, all cell
    // indices stay in `0..CSR_CELLS`, and slice lengths are asserted above.
    unsafe {
        core::ptr::write_bytes(coff_p, 0, CSR_CELLS + 2);
        let mut k = 0usize;
        while k < n {
            let pi = *pord_p.add(k) as usize;
            let pkx = *px_p.add(pi);
            let pky = *py_p.add(pi);
            if !pkx.is_finite() || !pky.is_finite() {
                *pnext_p.add(pi) = -1;
            } else {
                let cx = cell_coord(pkx, inv_cw, CSR_CELL_W - 1);
                let cy = cell_coord(pky, inv_ch, CSR_CELL_H - 1);
                let cell = cy * CSR_CELL_W + cx;
                *pnext_p.add(pi) = cell as i16;
                *coff_p.add(cell + 2) += 1;
            }
            k += 1;
        }

        let mut acc = 0u16;
        let mut c = 2usize;
        while c < CSR_CELLS + 2 {
            acc += *coff_p.add(c);
            *coff_p.add(c) = acc;
            c += 1;
        }

        // Populate contiguous cell slices in forward Morton order using in-place offset cursor `cell_off[c + 1]`
        let mut k2 = 0usize;
        while k2 < n {
            let pi = *pord_p.add(k2) as usize;
            let cell = *pnext_p.add(pi);
            if cell >= 0 {
                let cu = cell as usize;
                let cur_slot = coff_p.add(cu + 1);
                let dst = *cur_slot as usize;
                *cur_slot = (dst + 1) as u16;
                *cpx_p.add(dst) = *px_p.add(pi);
                *cpy_p.add(dst) = *py_p.add(pi);
                *cidx_p.add(dst) = pi as u16;
            }
            k2 += 1;
        }
    }
}

/// 4-wide SIMD128 IMM prediction + camera pan advection across all `1536` micro slots.
#[inline(never)]
pub fn imm_predict_micro_simd128(
    ms: &mut MicroSwarmState,
    w: usize,
    h: usize,
    safe_dt: f32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    scene_cut: bool,
) {
    if scene_cut {
        ms.reset_active_keep_mode();
        return;
    }

    let max_x = (w - 2) as f32;
    let max_y = (h - 2) as f32;
    let inv_w_255 = 255.0 / (w as f32);
    let inv_h_255 = 255.0 / (h as f32);

    #[cfg(target_arch = "wasm32")]
    unsafe {
        let v_pan_x = f32x4_splat(pan_dx_f);
        let v_pan_y = f32x4_splat(pan_dy_f);
        let v_dt = f32x4_splat(safe_dt);
        let v_bal_dt = f32x4_splat(1.06 * safe_dt);
        let v_osc_k = f32x4_splat(0.95);
        let v_osc_dt = f32x4_splat(0.72 * safe_dt);
        let v_zero = f32x4_splat(0.0);
        let v_max_x = f32x4_splat(max_x);
        let v_max_y = f32x4_splat(max_y);

        let p_pos_x = ms.tracks.pos_x.as_mut_ptr() as *mut v128;
        let p_pos_y = ms.tracks.pos_y.as_mut_ptr() as *mut v128;
        let p_prev_x = ms.tracks.prev_x.as_mut_ptr() as *mut v128;
        let p_prev_y = ms.tracks.prev_y.as_mut_ptr() as *mut v128;
        let p_lobs_x = ms.tracks.last_obs_x.as_mut_ptr() as *mut v128;
        let p_lobs_y = ms.tracks.last_obs_y.as_mut_ptr() as *mut v128;
        let p_vel_x = ms.tracks.vel_x.as_ptr() as *const v128;
        let p_vel_y = ms.tracks.vel_y.as_ptr() as *const v128;
        let p_pred_x = ms.tracks.pred_x.as_mut_ptr() as *mut v128;
        let p_pred_y = ms.tracks.pred_y.as_mut_ptr() as *mut v128;
        let p_eq_x = ms.tracks.eq_x.as_mut_ptr() as *mut v128;
        let p_eq_y = ms.tracks.eq_y.as_mut_ptr() as *mut v128;
        let p_ax0 = ms.tracks.ftle_a_x0.as_mut_ptr() as *mut v128;
        let p_ay0 = ms.tracks.ftle_a_y0.as_mut_ptr() as *mut v128;
        let p_bx0 = ms.tracks.ftle_b_x0.as_mut_ptr() as *mut v128;
        let p_by0 = ms.tracks.ftle_b_y0.as_mut_ptr() as *mut v128;
        let p_w_cv = ms.tracks.imm_w_cv.as_ptr() as *const v128;
        let p_w_bal = ms.tracks.imm_w_bal.as_ptr() as *const v128;
        let p_w_osc = ms.tracks.imm_w_osc.as_ptr() as *const v128;
        let state_ptr = ms.tracks.state.as_ptr();

        let has_pan = pan_dx_f != 0.0 || pan_dy_f != 0.0;
        let mut blk = 0usize;
        while blk < MAX_MICRO / 4 {
            let base = blk * 4;
            let s4 = core::ptr::read_unaligned(state_ptr.add(base) as *const u32);
            if s4 == 0 {
                blk += 1;
                continue;
            }
            let all_active = (s4 & 0xFF) != 0
                && (s4 & 0xFF00) != 0
                && (s4 & 0xFF_0000) != 0
                && (s4 & 0xFF00_0000) != 0;
            if all_active {
                let (x, y, eqx, eqy) = if has_pan {
                    let x = f32x4_add(v128_load(p_pos_x.add(blk)), v_pan_x);
                    let y = f32x4_add(v128_load(p_pos_y.add(blk)), v_pan_y);
                    let eqx = f32x4_add(v128_load(p_eq_x.add(blk)), v_pan_x);
                    let eqy = f32x4_add(v128_load(p_eq_y.add(blk)), v_pan_y);
                    v128_store(p_pos_x.add(blk), x);
                    v128_store(p_pos_y.add(blk), y);
                    v128_store(p_lobs_x.add(blk), f32x4_add(v128_load(p_lobs_x.add(blk)), v_pan_x));
                    v128_store(p_lobs_y.add(blk), f32x4_add(v128_load(p_lobs_y.add(blk)), v_pan_y));
                    v128_store(p_eq_x.add(blk), eqx);
                    v128_store(p_eq_y.add(blk), eqy);
                    v128_store(p_ax0.add(blk), f32x4_add(v128_load(p_ax0.add(blk)), v_pan_x));
                    v128_store(p_ay0.add(blk), f32x4_add(v128_load(p_ay0.add(blk)), v_pan_y));
                    v128_store(p_bx0.add(blk), f32x4_add(v128_load(p_bx0.add(blk)), v_pan_x));
                    v128_store(p_by0.add(blk), f32x4_add(v128_load(p_by0.add(blk)), v_pan_y));
                    (x, y, eqx, eqy)
                } else {
                    (
                        v128_load(p_pos_x.add(blk)),
                        v128_load(p_pos_y.add(blk)),
                        v128_load(p_eq_x.add(blk)),
                        v128_load(p_eq_y.add(blk)),
                    )
                };
                v128_store(p_prev_x.add(blk), x);
                v128_store(p_prev_y.add(blk), y);

                let vx = v128_load(p_vel_x.add(blk));
                let vy = v128_load(p_vel_y.add(blk));
                let w_cv = v128_load(p_w_cv.add(blk));
                let w_bal = v128_load(p_w_bal.add(blk));
                let w_osc = v128_load(p_w_osc.add(blk));

                let x_cv = f32x4_add(x, f32x4_mul(vx, v_dt));
                let y_cv = f32x4_add(y, f32x4_mul(vy, v_dt));

                let x_bal = f32x4_add(x, f32x4_mul(vx, v_bal_dt));
                let y_bal = f32x4_add(y, f32x4_mul(vy, v_bal_dt));

                let x_osc = f32x4_add(eqx, f32x4_add(f32x4_mul(f32x4_sub(x, eqx), v_osc_k), f32x4_mul(vx, v_osc_dt)));
                let y_osc = f32x4_add(eqy, f32x4_add(f32x4_mul(f32x4_sub(y, eqy), v_osc_k), f32x4_mul(vy, v_osc_dt)));

                let px = f32x4_add(f32x4_mul(w_cv, x_cv), f32x4_add(f32x4_mul(w_bal, x_bal), f32x4_mul(w_osc, x_osc)));
                let py = f32x4_add(f32x4_mul(w_cv, y_cv), f32x4_add(f32x4_mul(w_bal, y_bal), f32x4_mul(w_osc, y_osc)));

                v128_store(p_pred_x.add(blk), f32x4_pmin(v_max_x, f32x4_pmax(v_zero, px)));
                v128_store(p_pred_y.add(blk), f32x4_pmin(v_max_y, f32x4_pmax(v_zero, py)));
            } else {
                let mut lane = 0usize;
                while lane < 4 {
                    let i = base + lane;
                    if *state_ptr.add(i) != MICRO_STATE_INACTIVE {
                        let (x, y, eqx, eqy) = if has_pan {
                            let x = ms.tracks.pos_x[i] + pan_dx_f;
                            let y = ms.tracks.pos_y[i] + pan_dy_f;
                            let eqx = ms.tracks.eq_x[i] + pan_dx_f;
                            let eqy = ms.tracks.eq_y[i] + pan_dy_f;
                            ms.tracks.pos_x[i] = x;
                            ms.tracks.pos_y[i] = y;
                            ms.tracks.last_obs_x[i] += pan_dx_f;
                            ms.tracks.last_obs_y[i] += pan_dy_f;
                            ms.tracks.eq_x[i] = eqx;
                            ms.tracks.eq_y[i] = eqy;
                            ms.tracks.ftle_a_x0[i] += pan_dx_f;
                            ms.tracks.ftle_a_y0[i] += pan_dy_f;
                            ms.tracks.ftle_b_x0[i] += pan_dx_f;
                            ms.tracks.ftle_b_y0[i] += pan_dy_f;
                            (x, y, eqx, eqy)
                        } else {
                            (
                                ms.tracks.pos_x[i],
                                ms.tracks.pos_y[i],
                                ms.tracks.eq_x[i],
                                ms.tracks.eq_y[i],
                            )
                        };
                        ms.tracks.prev_x[i] = x;
                        ms.tracks.prev_y[i] = y;

                        let vx = ms.tracks.vel_x[i];
                        let vy = ms.tracks.vel_y[i];
                        let x_cv = x + vx * safe_dt;
                        let y_cv = y + vy * safe_dt;
                        let x_bal = x + 1.06 * vx * safe_dt;
                        let y_bal = y + 1.06 * vy * safe_dt;
                        let x_osc = eqx + 0.95 * (x - eqx) + 0.72 * vx * safe_dt;
                        let y_osc = eqy + 0.95 * (y - eqy) + 0.72 * vy * safe_dt;

                        let px = ms.tracks.imm_w_cv[i] * x_cv
                            + ms.tracks.imm_w_bal[i] * x_bal
                            + ms.tracks.imm_w_osc[i] * x_osc;
                        let py = ms.tracks.imm_w_cv[i] * y_cv
                            + ms.tracks.imm_w_bal[i] * y_bal
                            + ms.tracks.imm_w_osc[i] * y_osc;
                        ms.tracks.pred_x[i] = clamp_f32(px, 0.0, max_x);
                        ms.tracks.pred_y[i] = clamp_f32(py, 0.0, max_y);
                    }
                    lane += 1;
                }
            }
            blk += 1;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut i = 0usize;
        while i < MAX_MICRO {
            if ms.tracks.state[i] != MICRO_STATE_INACTIVE {
                let x = ms.tracks.pos_x[i] + pan_dx_f;
                let y = ms.tracks.pos_y[i] + pan_dy_f;
                let eqx = ms.tracks.eq_x[i] + pan_dx_f;
                let eqy = ms.tracks.eq_y[i] + pan_dy_f;
                ms.tracks.prev_x[i] = x;
                ms.tracks.prev_y[i] = y;
                ms.tracks.pos_x[i] = x;
                ms.tracks.pos_y[i] = y;
                ms.tracks.last_obs_x[i] += pan_dx_f;
                ms.tracks.last_obs_y[i] += pan_dy_f;
                ms.tracks.eq_x[i] = eqx;
                ms.tracks.eq_y[i] = eqy;
                ms.tracks.ftle_a_x0[i] += pan_dx_f;
                ms.tracks.ftle_a_y0[i] += pan_dy_f;
                ms.tracks.ftle_b_x0[i] += pan_dx_f;
                ms.tracks.ftle_b_y0[i] += pan_dy_f;

                let vx = ms.tracks.vel_x[i];
                let vy = ms.tracks.vel_y[i];
                let x_cv = x + vx * safe_dt;
                let y_cv = y + vy * safe_dt;
                let x_bal = x + 1.06 * vx * safe_dt;
                let y_bal = y + 1.06 * vy * safe_dt;
                let x_osc = eqx + 0.95 * (x - eqx) + 0.72 * vx * safe_dt;
                let y_osc = eqy + 0.95 * (y - eqy) + 0.72 * vy * safe_dt;

                let px = ms.tracks.imm_w_cv[i] * x_cv
                    + ms.tracks.imm_w_bal[i] * x_bal
                    + ms.tracks.imm_w_osc[i] * x_osc;
                let py = ms.tracks.imm_w_cv[i] * y_cv
                    + ms.tracks.imm_w_bal[i] * y_bal
                    + ms.tracks.imm_w_osc[i] * y_osc;
                ms.tracks.pred_x[i] = clamp_f32(px, 0.0, max_x);
                ms.tracks.pred_y[i] = clamp_f32(py, 0.0, max_y);
            }
            i += 1;
        }
    }

    // Collect active slots (deactivating any corrupted non-finite slot) & sort by Morton order
    let mut act_n = 0usize;
    unsafe {
        core::ptr::write_bytes(ms.tracks.reacq_flag.as_mut_ptr(), 0, MAX_MICRO);
        let st_p = ms.tracks.state.as_mut_ptr();
        let px_p = ms.tracks.pos_x.as_ptr();
        let py_p = ms.tracks.pos_y.as_ptr();
        let lx_p = ms.tracks.last_obs_x.as_ptr();
        let ly_p = ms.tracks.last_obs_y.as_ptr();
        let vx_p = ms.tracks.vel_x.as_ptr();
        let vy_p = ms.tracks.vel_y.as_ptr();
        let eqx_p = ms.tracks.eq_x.as_ptr();
        let eqy_p = ms.tracks.eq_y.as_ptr();
        let prx_p = ms.tracks.pred_x.as_ptr();
        let pry_p = ms.tracks.pred_y.as_ptr();
        let wcv_p = ms.tracks.imm_w_cv.as_ptr();
        let wbal_p = ms.tracks.imm_w_bal.as_ptr();
        let wosc_p = ms.tracks.imm_w_osc.as_ptr();
        let tm_p = ms.track_morton.as_mut_ptr();
        let aq_p = ms.auction_q.as_mut_ptr();
        let mut slot = 0usize;
        while slot < MAX_MICRO {
            if *st_p.add(slot) != MICRO_STATE_INACTIVE {
                let finite_ok = (*px_p.add(slot)).is_finite()
                    && (*py_p.add(slot)).is_finite()
                    && (*lx_p.add(slot)).is_finite()
                    && (*ly_p.add(slot)).is_finite()
                    && (*vx_p.add(slot)).is_finite()
                    && (*vy_p.add(slot)).is_finite()
                    && (*eqx_p.add(slot)).is_finite()
                    && (*eqy_p.add(slot)).is_finite()
                    && (*prx_p.add(slot)).is_finite()
                    && (*pry_p.add(slot)).is_finite()
                    && (*wcv_p.add(slot)).is_finite()
                    && (*wbal_p.add(slot)).is_finite()
                    && (*wosc_p.add(slot)).is_finite();
                if !finite_ok {
                    *st_p.add(slot) = MICRO_STATE_INACTIVE;
                    ms.tracks.track_id[slot] = 0;
                    ms.tracks.age[slot] = 0;
                    ms.tracks.coasting[slot] = 0;
                    ms.tracks.ambiguous_flag[slot] = 0;
                    ms.tracks.body_id[slot] = 0;
                    ms.tracks.ftle_a_frame[slot] = FTLE_INVALID_FRAME;
                    ms.tracks.ftle_b_frame[slot] = FTLE_INVALID_FRAME;
                    ms.tracks.ftle_val[slot] = 0.0;
                    slot += 1;
                    continue;
                }
                *tm_p.add(act_n) =
                    morton16_xy(*prx_p.add(slot), *pry_p.add(slot), inv_w_255, inv_h_255);
                *aq_p.add(act_n) = slot as u16;
                act_n += 1;
            }
            slot += 1;
        }
    }
    ms.active_count = act_n as u32;
    if act_n > 0 {
        radix_sort_u16_keys(
            &ms.track_morton,
            act_n,
            false,
            &mut ms.track_order,
            &mut ms.peak_sort_tmp,
            &mut ms.radix_cnt,
        );
        // Remap `track_order[k]` directly to physical track slot `[0..MAX_MICRO)`
        unsafe {
            let to_p = ms.track_order.as_mut_ptr();
            let aq_p = ms.auction_q.as_ptr();
            let mut k = 0usize;
            while k < act_n {
                *to_p.add(k) = *aq_p.add(*to_p.add(k) as usize);
                k += 1;
            }
        }
    }
}

/// Gathers up to `MAX_CSR_K = 8` lowest-cost Q12 candidates per active track over exact gate-projected
/// `32x18` cells. Stage 1 filters by squared innovation distance `d2 <= r_gate2` keeping up to
/// `MAX_PRELIM_CANDS = 8` survivors ordered by an O(1) compatibility proxy (`distance + 2-adic + pole`);
/// Stage 2 evaluates SIMD128 `8x8` `i8` SAD + opponent-chroma before Q12 integer conversion
/// (`cost_q = round(4096 * cost) < c_coast_q`).
#[inline(never)]
pub fn gather_csr_candidates(ms: &mut MicroSwarmState, w: usize, h: usize, safe_dt: f32) {
    let act_n = ms.active_count as usize;
    ms.csr_len.fill(0);
    ms.csr_overflow_cnt = 0;
    if act_n == 0 || ms.num_peaks == 0 || !(safe_dt > 0.0 && safe_dt < 1.0e15) {
        return;
    }

    assert!(act_n <= MAX_MICRO);
    assert!(ms.track_order.len() >= MAX_MICRO);
    assert!(ms.csr_len.len() >= MAX_MICRO);
    assert!(ms.tracks.pred_x.len() >= MAX_MICRO && ms.tracks.pred_y.len() >= MAX_MICRO);
    assert!(ms.tracks.vel_x.len() >= MAX_MICRO && ms.tracks.vel_y.len() >= MAX_MICRO);
    assert!(ms.tracks.coasting.len() >= MAX_MICRO && ms.tracks.state.len() >= MAX_MICRO);
    assert!(ms.tracks.age.len() >= MAX_MICRO);
    assert!(ms.tracks.curr_sig.len() >= MAX_MICRO && ms.tracks.anchor_sig.len() >= MAX_MICRO);
    assert!(ms.tracks.chroma_rg.len() >= MAX_MICRO && ms.tracks.chroma_by.len() >= MAX_MICRO);
    assert!(ms.tracks.tpl8_i8.len() >= MAX_MICRO);
    assert!(ms.cell_off.len() >= CSR_CELLS + 2);
    assert!(ms.cell_px.len() >= MAX_MICRO_PEAKS + 4 && ms.cell_py.len() >= MAX_MICRO_PEAKS + 4);
    assert!(ms.cell_pidx.len() >= MAX_MICRO_PEAKS);
    assert!(ms.prelim_cands.len() >= MAX_PRELIM_CANDS);
    assert!(ms.peak_tpl8.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_sig.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_rg.len() >= MAX_MICRO_PEAKS && ms.peak_by.len() >= MAX_MICRO_PEAKS);
    assert!(ms.csr_cost.len() >= MAX_MICRO * MAX_CSR_K);
    assert!(ms.csr_peak_idx.len() >= MAX_MICRO * MAX_CSR_K);

    let q = (w as f32) * (1.0 / 640.0);
    let ch = (h as f32) * (1.0 / (CSR_CELL_H as f32));
    let inv_cw = (CSR_CELL_W as f32) / (w as f32);
    let inv_ch = (CSR_CELL_H as f32) / (h as f32);
    let dense_swarm_fast = ms.swarm_active != 0 && ms.active_count >= 960 && ms.num_peaks >= 960;
    let dense_clutter = ms.active_count >= 1024 && ms.num_peaks >= 1500;
    let is_dense = ms.active_count >= 512;
    let prelim_cap = if dense_swarm_fast { 3 } else { MAX_PRELIM_CANDS };
    let row_cap = if dense_swarm_fast { 3 } else { MAX_CSR_K };

    // Precompute quadratic distance gate multipliers `(u, u^2)` for confirmed & tentative thresholds
    let max_prelim_conf = ((COAST_Q_CONF as f32) - 0.5) * (1.0 / AUCTION_SCALE_Q);
    let max_prelim_tent = ((COAST_Q_TENT as f32) - 0.5) * (1.0 / AUCTION_SCALE_Q);
    let u_conf = ((-0.65 + wasm_sqrtf((0.4225 + 0.60 * max_prelim_conf).max(0.0)))
        * (1.0 / 0.30))
        .max(0.0);
    let u_tent = ((-0.65 + wasm_sqrtf((0.4225 + 0.60 * max_prelim_tent).max(0.0)))
        * (1.0 / 0.30))
        .max(0.0);
    let u2_conf = u_conf * u_conf * (1.0 + 1e-5);
    let u_conf_pad = u_conf * (1.0 + 2e-5);
    let u2_tent = u_tent * u_tent * (1.0 + 1e-5);
    let u_tent_pad = u_tent * (1.0 + 2e-5);

    let order_ptr = ms.track_order.as_ptr();
    let pred_x_ptr = ms.tracks.pred_x.as_ptr();
    let pred_y_ptr = ms.tracks.pred_y.as_ptr();
    let vel_x_ptr = ms.tracks.vel_x.as_ptr();
    let vel_y_ptr = ms.tracks.vel_y.as_ptr();
    let coast_ptr = ms.tracks.coasting.as_ptr();
    let state_ptr = ms.tracks.state.as_ptr();
    let age_ptr = ms.tracks.age.as_ptr();
    let curr_sig_ptr = ms.tracks.curr_sig.as_ptr();
    let anc_sig_ptr = ms.tracks.anchor_sig.as_ptr();

    let mut ord_i = 0usize;
    while ord_i < act_n {
        let (
            ti,
            px,
            py,
            vx,
            vy,
            coast,
            is_conf_live,
            age,
            t_sig_curr,
            t_sig_anc,
        ) = unsafe {
            let ti = *order_ptr.add(ord_i) as usize;
            (
                ti,
                *pred_x_ptr.add(ti),
                *pred_y_ptr.add(ti),
                *vel_x_ptr.add(ti),
                *vel_y_ptr.add(ti),
                *coast_ptr.add(ti),
                *state_ptr.add(ti) == MICRO_STATE_CONFIRMED && *coast_ptr.add(ti) == 0,
                *age_ptr.add(ti),
                *curr_sig_ptr.add(ti),
                *anc_sig_ptr.add(ti),
            )
        };
        if !(px.abs() < 1.0e15 && py.abs() < 1.0e15 && vx.abs() < 1.0e15 && vy.abs() < 1.0e15) {
            unsafe { *ms.csr_len.as_mut_ptr().add(ti) = 0 };
            ord_i += 1;
            continue;
        }
        let speed = wasm_sqrtf(vx * vx + vy * vy);
        let step_dist = speed * safe_dt;
        let base_gate = if is_conf_live {
            if dense_clutter {
                if age >= 2 {
                    (5.8 * q + 1.35 * step_dist).min(11.5 * q)
                } else {
                    10.0 * q
                }
            } else if is_dense {
                if age >= 2 {
                    (6.2 * q + 1.45 * step_dist).min(13.5 * q)
                } else {
                    10.8 * q
                }
            } else {
                15.0 * q
            }
        } else if coast == 0 {
            if dense_clutter {
                if age >= 2 {
                    (6.2 * q + 1.35 * step_dist).min(12.2 * q)
                } else {
                    7.2 * q
                }
            } else if is_dense {
                if age >= 2 {
                    (7.0 * q + 1.45 * step_dist).min(14.5 * q)
                } else {
                    11.5 * q
                }
            } else {
                19.0 * q
            }
        } else {
            let max_c = if dense_clutter {
                if coast >= 3 {
                    9.5 * q
                } else if coast >= 2 {
                    10.5 * q
                } else {
                    13.5 * q
                }
            } else {
                (18.0 + 4.0 * (coast as f32)).min(42.0) * q
            };
            if dense_clutter {
                if step_dist < 1.5 * q && age >= 3 {
                    (6.2 * q + 1.0 * (coast as f32) * q).min(9.5 * q)
                } else {
                    (6.4 * q + (1.4 * q + 0.55 * step_dist) * (coast as f32).min(2.5)).min(max_c)
                }
            } else if is_dense {
                if step_dist < 1.5 * q && age >= 3 {
                    (6.2 * q + 1.0 * (coast as f32) * q).min(9.5 * q)
                } else {
                    (6.8 * q + (1.5 * q + 0.65 * step_dist) * (coast as f32).min(4.0)).min(max_c)
                }
            } else {
                max_c
            }
        };
        let r_gate = base_gate
            + if dense_clutter {
                (0.20 * step_dist).min(6.0 * q)
            } else {
                (0.25 * step_dist).min(12.0 * q)
            };
        let r_gate2 = r_gate * r_gate;
        if !(r_gate >= 0.0 && r_gate < 1.0e15 && r_gate2 < 1.0e30) {
            unsafe { *ms.csr_len.as_mut_ptr().add(ti) = 0 };
            ord_i += 1;
            continue;
        }
        let inv_r_gate = 1.0 / r_gate.max(1.0);
        let c_coast_q: i32 = if is_conf_live { COAST_Q_CONF } else { COAST_Q_TENT };
        let c_coast_f = c_coast_q as f32;

        let cx0 = cell_coord(px, inv_cw, CSR_CELL_W - 1);
        let cy0 = cell_coord(py, inv_ch, CSR_CELL_H - 1);
        let (init_d2_lim, r_box) = if r_gate >= 1.0 {
            if is_conf_live {
                let d2l = if u_conf < 1.0 { r_gate2 * u2_conf } else { r_gate2 };
                let rb = if u_conf < 1.0 { r_gate * u_conf_pad } else { r_gate };
                (d2l, rb)
            } else {
                let d2l = if u_tent < 1.0 { r_gate2 * u2_tent } else { r_gate2 };
                let rb = if u_tent < 1.0 { r_gate * u_tent_pad } else { r_gate };
                (d2l, rb)
            }
        } else {
            (r_gate2, r_gate)
        };

        // Exact tightened gate projection onto 64x36 grid cells on both axes (aspect-safe)
        let x_lo = cell_coord(px - r_box, inv_cw, CSR_CELL_W - 1);
        let x_hi = cell_coord(px + r_box, inv_cw, CSR_CELL_W - 1).max(x_lo);
        let y_lo = cell_coord(py - r_box, inv_ch, CSR_CELL_H - 1);
        let y_hi = cell_coord(py + r_box, inv_ch, CSR_CELL_H - 1).max(y_lo);

        // SAFETY: All `MicroSwarmState` slice lengths are asserted once at entry to
        // `gather_csr_candidates`, `ti < MAX_MICRO`, `cx0/x_lo..=x_hi < CSR_CELL_W`, and
        // `cy0/y_lo..=y_hi < CSR_CELL_H`.
        let prelim_len = unsafe {
            if prelim_cap == 3 {
                let (tight_hit_d2, tight_cap_d2, tight_r) =
                    if is_conf_live && *ms.tracks.ambiguous_flag.as_ptr().add(ti) == 0 {
                        (1.21 * q * q, 14.44 * q * q, 3.8 * q + 0.015)
                    } else {
                        (-1.0, init_d2_lim, 0.0)
                    };
                gather_csr_stage1_track::<true>(
                    ms,
                    px,
                    py,
                    cx0,
                    cy0,
                    x_lo,
                    x_hi,
                    y_lo,
                    y_hi,
                    inv_cw,
                    inv_ch,
                    ch,
                    init_d2_lim,
                    3,
                    tight_hit_d2,
                    tight_cap_d2,
                    tight_r,
                )
            } else {
                gather_csr_stage1_track::<false>(
                    ms,
                    px,
                    py,
                    cx0,
                    cy0,
                    x_lo,
                    x_hi,
                    y_lo,
                    y_hi,
                    inv_cw,
                    inv_ch,
                    ch,
                    init_d2_lim,
                    prelim_cap,
                    -1.0,
                    init_d2_lim,
                    0.0,
                )
            }
        };

        if prelim_len == 0 {
            unsafe { *ms.csr_len.as_mut_ptr().add(ti) = 0 };
            ord_i += 1;
            continue;
        }

        unsafe {
            gather_csr_stage2_track(
                ms,
                ti,
                prelim_len,
                row_cap,
                c_coast_f,
                r_gate,
                inv_r_gate,
                coast,
                t_sig_curr,
                t_sig_anc,
            );
        }
        ord_i += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn push_stage1_cand(
    prelim_ptr: *mut PrelimCand,
    prelim_len: &mut usize,
    cap_lim: usize,
    d2_lim: &mut f32,
    idx_ptr: *const u16,
    ki: usize,
    d2: f32,
) -> bool {
    let l = *prelim_len;
    if l < cap_lim {
        let mut ins = l;
        while ins > 0 && d2 < (*prelim_ptr.add(ins - 1)).d2 {
            *prelim_ptr.add(ins) = *prelim_ptr.add(ins - 1);
            ins -= 1;
        }
        *prelim_ptr.add(ins) = PrelimCand {
            score: d2,
            d2,
            padic_pen: 0.0,
            idx: *idx_ptr.add(ki) as i32,
        };
        let next_len = l + 1;
        *prelim_len = next_len;
        if next_len == cap_lim {
            let wd2 = (*prelim_ptr.add(cap_lim - 1)).d2;
            if wd2 < *d2_lim {
                *d2_lim = wd2;
                return true;
            }
        }
    } else if d2 < *d2_lim {
        let prev_lim = *d2_lim;
        let mut ins = cap_lim - 1;
        while ins > 0 && d2 < (*prelim_ptr.add(ins - 1)).d2 {
            *prelim_ptr.add(ins) = *prelim_ptr.add(ins - 1);
            ins -= 1;
        }
        *prelim_ptr.add(ins) = PrelimCand {
            score: d2,
            d2,
            padic_pen: 0.0,
            idx: *idx_ptr.add(ki) as i32,
        };
        let wd2 = (*prelim_ptr.add(cap_lim - 1)).d2;
        if wd2 < prev_lim {
            *d2_lim = wd2;
            return true;
        }
    }
    false
}

/// # Safety
/// Caller must assert all `MicroSwarmState` slice lengths (`cell_off`, `cell_px`, `cell_py`,
/// `cell_pidx`, `prelim_cands`) and ensure `cx0/x_lo..=x_hi < CSR_CELL_W`, `cy0/y_lo..=y_hi < CSR_CELL_H`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn gather_csr_stage1_track<const TOP3: bool>(
    ms: &mut MicroSwarmState,
    px: f32,
    py: f32,
    cx0: usize,
    cy0: usize,
    x_lo: usize,
    x_hi: usize,
    y_lo: usize,
    y_hi: usize,
    inv_cw: f32,
    inv_ch: f32,
    ch: f32,
    init_d2_lim: f32,
    prelim_cap: usize,
    tight_hit_d2: f32,
    tight_cap_d2: f32,
    tight_r: f32,
) -> usize {
    let cap_lim = prelim_cap.clamp(1, MAX_PRELIM_CANDS);

    let single_cell = x_lo == x_hi && y_lo == y_hi;
    let c0 = cy0 * CSR_CELL_W + cx0;
    let cell_off_ptr = ms.cell_off.as_ptr();
    let c0_s = *cell_off_ptr.add(c0) as usize;
    let c0_e = *cell_off_ptr.add(c0 + 1) as usize;
    if single_cell && c0_e <= c0_s {
        return 0;
    }

    let mut prelim_len = 0usize;
    let mut d2_lim = init_d2_lim;
    let (mut d0, mut d1, mut d2v) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let (mut i0, mut i1, mut i2) = (0i32, 0i32, 0i32);

    let px_ptr = ms.cell_px.as_ptr();
    let py_ptr = ms.cell_py.as_ptr();
    let idx_ptr = ms.cell_pidx.as_ptr();
    let prelim_ptr = ms.prelim_cands.as_mut_ptr();

    #[cfg(target_arch = "wasm32")]
    let v_px = f32x4_splat(px);
    #[cfg(target_arch = "wasm32")]
    let v_py = f32x4_splat(py);
    #[cfg(target_arch = "wasm32")]
    let mut v_d2_lim = f32x4_splat(d2_lim);

    macro_rules! push_stage1_hit {
        ($ki:expr, $d2:expr) => {{
            let ki = $ki;
            let d2 = $d2;
            if TOP3 {
                if prelim_len < 3 {
                    let idx_val = *idx_ptr.add(ki) as i32;
                    if d2 < d0 {
                        d2v = d1;
                        i2 = i1;
                        d1 = d0;
                        i1 = i0;
                        d0 = d2;
                        i0 = idx_val;
                        if d2 < tight_hit_d2 && tight_cap_d2 < d2_lim {
                            d2_lim = tight_cap_d2;
                            #[cfg(target_arch = "wasm32")]
                            {
                                v_d2_lim = f32x4_splat(d2_lim);
                            }
                        }
                    } else if d2 < d1 {
                        d2v = d1;
                        i2 = i1;
                        d1 = d2;
                        i1 = idx_val;
                    } else {
                        d2v = d2;
                        i2 = idx_val;
                    }
                    prelim_len += 1;
                    if prelim_len == 3 && d2v < d2_lim {
                        d2_lim = d2v;
                        #[cfg(target_arch = "wasm32")]
                        {
                            v_d2_lim = f32x4_splat(d2_lim);
                        }
                    }
                } else if d2 < d2_lim {
                    let idx_val = *idx_ptr.add(ki) as i32;
                    if d2 < d0 {
                        d2v = d1;
                        i2 = i1;
                        d1 = d0;
                        i1 = i0;
                        d0 = d2;
                        i0 = idx_val;
                        if d2 < tight_hit_d2 && tight_cap_d2 < d2_lim {
                            d2_lim = tight_cap_d2;
                            #[cfg(target_arch = "wasm32")]
                            {
                                v_d2_lim = f32x4_splat(d2_lim);
                            }
                        }
                    } else if d2 < d1 {
                        d2v = d1;
                        i2 = i1;
                        d1 = d2;
                        i1 = idx_val;
                    } else {
                        d2v = d2;
                        i2 = idx_val;
                    }
                    if d2v < d2_lim {
                        d2_lim = d2v;
                        #[cfg(target_arch = "wasm32")]
                        {
                            v_d2_lim = f32x4_splat(d2_lim);
                        }
                    }
                }
            } else if push_stage1_cand(
                prelim_ptr,
                &mut prelim_len,
                cap_lim,
                &mut d2_lim,
                idx_ptr,
                ki,
                d2,
            ) {
                #[cfg(target_arch = "wasm32")]
                {
                    v_d2_lim = f32x4_splat(d2_lim);
                }
            }
        }};
    }

    let mut bx_lo = x_lo;
    let mut bx_hi = x_hi;
    let mut by_lo = y_lo;
    let mut by_hi = y_hi;
    let row0 = cy0 * CSR_CELL_W;
    let mut seg_phase = 0u8;
    let mut cy_cur = 0usize;
    let mut seg_s = c0_s;
    let mut seg_e = c0_e;
    let mut tight_d2_lim = init_d2_lim;

    loop {
        if seg_s < seg_e {
            let mut k = seg_s;
            let e = seg_e.min(MAX_MICRO_PEAKS);
            #[cfg(target_arch = "wasm32")]
            while k + 4 <= e {
                let vx = core::ptr::read_unaligned(px_ptr.add(k) as *const v128);
                let vy = core::ptr::read_unaligned(py_ptr.add(k) as *const v128);
                let dx = f32x4_sub(vx, v_px);
                let dy = f32x4_sub(vy, v_py);
                let vd2 = f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy));
                let hit_v = f32x4_le(vd2, v_d2_lim);
                if v128_any_true(hit_v) {
                    let mut mask = (i32x4_bitmask(hit_v) as u32) & 0xF;
                    while mask != 0 {
                        let lane = mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        let d2 = match lane {
                            0 => f32x4_extract_lane::<0>(vd2),
                            1 => f32x4_extract_lane::<1>(vd2),
                            2 => f32x4_extract_lane::<2>(vd2),
                            _ => f32x4_extract_lane::<3>(vd2),
                        };
                        push_stage1_hit!(k + lane, d2);
                    }
                }
                k += 4;
            }

            while k < e {
                let ldx = *px_ptr.add(k) - px;
                let ldy = *py_ptr.add(k) - py;
                let d2 = ldx * ldx + ldy * ldy;
                if d2 <= d2_lim {
                    push_stage1_hit!(k, d2);
                }
                k += 1;
            }
        }

        if seg_phase == 0 {
            if single_cell || y_lo > y_hi {
                break;
            }
            if d2_lim < tight_d2_lim {
                tight_d2_lim = d2_lim;
                let r_tight = if TOP3 && (d2_lim - tight_cap_d2).abs() <= 1e-6 {
                    tight_r
                } else {
                    wasm_sqrtf(d2_lim.max(0.0)) + 0.015
                };
                by_lo = by_lo.max(cell_coord(py - r_tight, inv_ch, cy0));
                by_hi = by_hi.min(cell_coord(py + r_tight, inv_ch, CSR_CELL_H - 1)).max(cy0);
                bx_lo = bx_lo.max(cell_coord(px - r_tight, inv_cw, cx0));
                bx_hi = bx_hi.min(cell_coord(px + r_tight, inv_cw, CSR_CELL_W - 1)).max(cx0);
            }
            seg_phase = 1;
            if bx_lo < cx0 {
                let ls = *cell_off_ptr.add(row0 + bx_lo) as usize;
                if ls < c0_s {
                    seg_s = ls;
                    seg_e = c0_s;
                    continue;
                }
            }
        }
        if seg_phase <= 1 {
            seg_phase = 2;
            if bx_hi > cx0 {
                let re = *cell_off_ptr.add(row0 + bx_hi + 1) as usize;
                if c0_e < re {
                    seg_s = c0_e;
                    seg_e = re;
                    continue;
                }
            }
        }
        if seg_phase <= 2 {
            if y_lo >= y_hi {
                break;
            }
            if d2_lim < tight_d2_lim {
                tight_d2_lim = d2_lim;
                let r_tight = if TOP3 && (d2_lim - tight_cap_d2).abs() <= 1e-6 {
                    tight_r
                } else {
                    wasm_sqrtf(d2_lim.max(0.0)) + 0.015
                };
                by_lo = by_lo.max(cell_coord(py - r_tight, inv_ch, cy0));
                by_hi = by_hi.min(cell_coord(py + r_tight, inv_ch, CSR_CELL_H - 1)).max(cy0);
                bx_lo = bx_lo.max(cell_coord(px - r_tight, inv_cw, cx0));
                bx_hi = bx_hi.min(cell_coord(px + r_tight, inv_cw, CSR_CELL_W - 1)).max(cx0);
            }
            seg_phase = 3;
            if cy0 > by_lo {
                let r_cur = cy0 - 1;
                let dy_row = (py - ((r_cur + 1) as f32) * ch - 0.01).max(0.0);
                if dy_row * dy_row <= d2_lim {
                    let ro = r_cur * CSR_CELL_W;
                    let ss = *cell_off_ptr.add(ro + bx_lo) as usize;
                    let ee = *cell_off_ptr.add(ro + bx_hi + 1) as usize;
                    if ss < ee {
                        seg_s = ss;
                        seg_e = ee;
                        continue;
                    }
                }
            }
        }
        if seg_phase <= 3 {
            seg_phase = 4;
            if cy0 < by_hi {
                let r_cur = cy0 + 1;
                let dy_row = ((r_cur as f32) * ch - py - 0.01).max(0.0);
                if dy_row * dy_row <= d2_lim {
                    let ro = r_cur * CSR_CELL_W;
                    let ss = *cell_off_ptr.add(ro + bx_lo) as usize;
                    let ee = *cell_off_ptr.add(ro + bx_hi + 1) as usize;
                    if ss < ee {
                        seg_s = ss;
                        seg_e = ee;
                        continue;
                    }
                }
            }
        }
        if seg_phase <= 4 {
            if by_lo + 1 >= cy0 && by_hi <= cy0 + 1 {
                break;
            }
            if d2_lim < tight_d2_lim {
                tight_d2_lim = d2_lim;
                let r_tight = wasm_sqrtf(d2_lim.max(0.0)) + 0.015;
                by_lo = by_lo.max(cell_coord(py - r_tight, inv_ch, cy0));
                by_hi = by_hi.min(cell_coord(py + r_tight, inv_ch, CSR_CELL_H - 1)).max(cy0);
                bx_lo = bx_lo.max(cell_coord(px - r_tight, inv_cw, cx0));
                bx_hi = bx_hi.min(cell_coord(px + r_tight, inv_cw, CSR_CELL_W - 1)).max(cx0);
            }
            seg_phase = 5;
            cy_cur = by_lo;
        }
        let mut found_row = false;
        while cy_cur <= by_hi {
            let r_cur = cy_cur;
            cy_cur += 1;
            if r_cur + 1 < cy0 || r_cur > cy0 + 1 {
                let dy_row = if r_cur < cy0 {
                    (py - ((r_cur + 1) as f32) * ch - 0.01).max(0.0)
                } else {
                    ((r_cur as f32) * ch - py - 0.01).max(0.0)
                };
                if dy_row * dy_row <= d2_lim {
                    let ro = r_cur * CSR_CELL_W;
                    let ss = *cell_off_ptr.add(ro + bx_lo) as usize;
                    let ee = *cell_off_ptr.add(ro + bx_hi + 1) as usize;
                    if ss < ee {
                        seg_s = ss;
                        seg_e = ee;
                        found_row = true;
                        break;
                    }
                }
            }
        }
        if !found_row {
            break;
        }
    }

    if TOP3 {
        if prelim_len >= 1 {
            *prelim_ptr = PrelimCand { score: d0, d2: d0, padic_pen: 0.0, idx: i0 };
        }
        if prelim_len >= 2 {
            *prelim_ptr.add(1) = PrelimCand { score: d1, d2: d1, padic_pen: 0.0, idx: i1 };
        }
        if prelim_len >= 3 {
            *prelim_ptr.add(2) = PrelimCand { score: d2v, d2: d2v, padic_pen: 0.0, idx: i2 };
        }
    }

    prelim_len
}

/// # Safety
/// Caller must assert all `MicroSwarmState` slice lengths (`prelim_cands`, `tracks.chroma_*`,
/// `tracks.tpl8_i8`, `peak_*`, `csr_*`), `ti < MAX_MICRO`, `prelim_len <= MAX_PRELIM_CANDS`,
/// and `row_cap <= MAX_CSR_K`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn gather_csr_stage2_track(
    ms: &mut MicroSwarmState,
    ti: usize,
    prelim_len: usize,
    row_cap: usize,
    c_coast_f: f32,
    r_gate: f32,
    inv_r_gate: f32,
    coast: u8,
    t_sig_curr: u32,
    t_sig_anc: u32,
) {
    // Stage 2: full distance + 2-adic/pole + patch SAD + opponent chroma on sorted `<= prelim_cap` survivors
    let csr_base = ti * MAX_CSR_K;
    let mut row_len = 0usize;
    let mut worst_q = i32::MAX;
    let t_rg = *ms.tracks.chroma_rg.as_ptr().add(ti);
    let t_by = *ms.tracks.chroma_by.as_ptr().add(ti);
    let t_has_chroma = (t_sig_curr as i32) < 0;
    let opp_rg = ((t_sig_curr >> 29) & 3) ^ 3;
    let opp_by = ((t_sig_curr >> 27) & 3) ^ 3;
    let inv_sad_scale = 1.0f32 / (64.0 * 64.0);

    let prelim_ptr = ms.prelim_cands.as_ptr();
    let track_tpl_ptr = ms.tracks.tpl8_i8.as_ptr();
    let peak_tpl_ptr = ms.peak_tpl8.as_ptr();
    let peak_sig_ptr = ms.peak_sig.as_ptr();
    let peak_rg_ptr = ms.peak_rg.as_ptr();
    let peak_by_ptr = ms.peak_by.as_ptr();
    let csr_cost_ptr = ms.csr_cost.as_mut_ptr() as *mut i32;
    let csr_peak_ptr = ms.csr_peak_idx.as_mut_ptr();

    let tpl_t = &*track_tpl_ptr.add(ti);
    #[cfg(target_arch = "wasm32")]
    let mut tpl_loaded = false;
    #[cfg(target_arch = "wasm32")]
    let (mut t0, mut t1, mut t2, mut t3) = {
        let z = i8x16_splat(0);
        (z, z, z, z)
    };
    let gate_en = r_gate >= 1.0;
    let mut lb_cost = (c_coast_f - 0.5) * (1.0 / AUCTION_SCALE_Q);
    let mut pk = 0usize;
    while pk < prelim_len {
        let cand = *prelim_ptr.add(pk);
        let pi = (cand.idx as usize).min(MAX_MICRO_PEAKS - 1);
        let d2 = cand.d2;
        pk += 1;

        let dist = wasm_sqrtf(d2);
        let dist_norm = dist * inv_r_gate;
        let base_dist_cost = 0.65 * dist_norm + 0.15 * dist_norm * dist_norm;
        if gate_en && base_dist_cost >= lb_cost {
            break;
        }
        let p_sig = *peak_sig_ptr.add(pi);
        let pole_pen = if t_has_chroma
            && (p_sig as i32) < 0
            && (((p_sig >> 29) & 3) == opp_rg || ((p_sig >> 27) & 3) == opp_by)
        {
            0.55
        } else {
            0.0
        };
        let pre_padic_cost = base_dist_cost + pole_pen;
        if gate_en && pre_padic_cost >= lb_cost {
            continue;
        }
        let padic_min = padic_min_inline(t_sig_curr, t_sig_anc, p_sig);
        let dist_only_cost = pre_padic_cost + 0.15 * padic_min;
        if gate_en && dist_only_cost >= lb_cost {
            continue;
        }

        #[cfg(target_arch = "wasm32")]
        let sad = {
            if !tpl_loaded {
                let pa = tpl_t.as_ptr() as *const v128;
                t0 = v128_load(pa);
                t1 = v128_load(pa.add(1));
                t2 = v128_load(pa.add(2));
                t3 = v128_load(pa.add(3));
                tpl_loaded = true;
            }
            patch_8x8_sad_preloaded_i8_simd128(
                t0,
                t1,
                t2,
                t3,
                &*peak_tpl_ptr.add(pi),
            )
        };
        #[cfg(not(target_arch = "wasm32"))]
        let sad = patch_8x8_sad_i8_simd128(tpl_t, &*peak_tpl_ptr.add(pi));
        if coast == 0 && sad > 1966 {
            continue;
        }
        let sad_norm = (sad as f32) * inv_sad_scale;
        let pre_chroma_cost = dist_only_cost + 0.45 * sad_norm;
        if gate_en && pre_chroma_cost >= lb_cost {
            continue;
        }

        let drg = t_rg - *peak_rg_ptr.add(pi);
        let dby = t_by - *peak_by_ptr.add(pi);
        let chr2 = drg * drg + dby * dby;
        let rem_chr = (lb_cost - pre_chroma_cost) * 10.0;
        if rem_chr >= 0.0 && chr2 > rem_chr * rem_chr + 1e-5 {
            continue;
        }
        let chroma_d = if chr2 > 1e-8 { wasm_sqrtf(chr2) } else { 0.0 };
        let cost = pre_chroma_cost + 0.10 * chroma_d;
        let cost_scaled = cost * AUCTION_SCALE_Q + 0.5;
        if !(cost_scaled >= 0.0 && cost_scaled < c_coast_f) {
            continue;
        }
        let cost_q = cost_scaled as i32;

        if row_len < row_cap || cost_q < worst_q {
            let mut ins = if row_len < row_cap {
                let l = row_len;
                row_len += 1;
                l
            } else {
                row_cap - 1
            };
            while ins > 0 && cost_q < *csr_cost_ptr.add(csr_base + ins - 1) {
                *csr_cost_ptr.add(csr_base + ins) = *csr_cost_ptr.add(csr_base + ins - 1);
                *csr_peak_ptr.add(csr_base + ins) = *csr_peak_ptr.add(csr_base + ins - 1);
                ins -= 1;
            }
            *csr_cost_ptr.add(csr_base + ins) = cost_q;
            *csr_peak_ptr.add(csr_base + ins) = pi as i16;
            if row_len == row_cap {
                worst_q = *csr_cost_ptr.add(csr_base + row_cap - 1);
                lb_cost = ((worst_q as f32) - 0.5) * (1.0 / AUCTION_SCALE_Q);
            }
        }
    }
    *ms.csr_len.as_mut_ptr().add(ti) = row_len as u8;
}

/// Bounded Q12 integer Bertsekas Forward-Reverse `epsilon`-scaling Asymmetric Auction matcher
/// with private coasting surplus (`price = 0`), reverse unowned-price cleanup, cap-repair fallback,
/// and ambiguity margin flagging.
#[inline(never)]
pub fn solve_epsilon_auction(ms: &mut MicroSwarmState) {
    let act_n = ms.active_count as usize;
    let num_peaks = ms.num_peaks;
    assert!(MAX_MICRO_PEAKS.is_power_of_two());
    assert!(MAX_CSR_K == 8);
    assert!(act_n <= MAX_MICRO);
    assert!(num_peaks <= MAX_MICRO_PEAKS);
    assert!(ms.track_order.len() >= MAX_MICRO);
    assert!(ms.track_match.len() >= MAX_MICRO);
    assert!(ms.tracks.ambiguous_flag.len() >= MAX_MICRO);
    assert!(ms.tracks.state.len() >= MAX_MICRO);
    assert!(ms.tracks.coasting.len() >= MAX_MICRO);
    assert!(ms.csr_len.len() >= MAX_MICRO);
    assert!(ms.csr_peak_idx.len() >= MAX_MICRO * MAX_CSR_K);
    assert!(ms.csr_cost.len() >= MAX_MICRO * MAX_CSR_K);
    assert!(ms.rev_head.len() >= MAX_MICRO_PEAKS);
    assert!(ms.rev_next.len() >= MAX_MICRO * MAX_CSR_K);
    assert!(ms.peak_price_q.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_owner.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_matched.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_sort_tmp.len() >= MAX_MICRO_PEAKS);
    assert!(ms.auction_q.len() >= MAX_MICRO);
    assert!(ms.cell_px.len() >= MAX_MICRO);

    let track_order_ptr = ms.track_order.as_ptr();
    let track_match_ptr = ms.track_match.as_mut_ptr();
    let ambig_ptr = ms.tracks.ambiguous_flag.as_mut_ptr();
    let state_ptr = ms.tracks.state.as_ptr();
    let coasting_ptr = ms.tracks.coasting.as_ptr();
    let csr_len_ptr = ms.csr_len.as_ptr();
    let csr_peak_ptr = ms.csr_peak_idx.as_ptr();
    let csr_cost_ptr = ms.csr_cost.as_ptr() as *const i32;
    let match_cost_ptr = ms.cell_px.as_mut_ptr() as *mut i32;
    let rev_head_ptr = ms.rev_head.as_mut_ptr();
    let rev_next_ptr = ms.rev_next.as_mut_ptr();
    let peak_price_ptr = ms.peak_price_q.as_mut_ptr();
    let peak_owner_ptr = ms.peak_owner.as_mut_ptr();
    let peak_matched_ptr = ms.peak_matched.as_mut_ptr();
    let peak_sort_tmp_ptr = ms.peak_sort_tmp.as_mut_ptr();
    let auction_q_ptr = ms.auction_q.as_mut_ptr();

    // SAFETY: All lengths are asserted above; all track indices (`0..MAX_MICRO`), CSR offsets
    // (`0..MAX_MICRO*MAX_CSR_K`), and peak indices (`0..num_peaks <= MAX_MICRO_PEAKS`) are bounded.
    unsafe {
        core::ptr::write_bytes(track_match_ptr, 0xFF, MAX_MICRO);
        core::ptr::write_bytes(ambig_ptr, 0, MAX_MICRO);
    }
    ms.auction_steps = 0;
    ms.auction_eps_q = 0;
    ms.auction_incomplete = 0;
    ms.auction_cap_hits = 0;
    ms.auction_repairs = 0;
    ms.ambiguous_count = 0;
    if act_n == 0 || num_peaks == 0 {
        return;
    }

    const PEAK_Q_MASK: usize = MAX_MICRO_PEAKS - 1;
    const EPS_PHASES_NORM: [i32; 3] = [384, 48, 4];
    const EPS_PHASES_DENSE: [i32; 3] = [320, 16, 16];
    let dense_swarm_fast = ms.swarm_active != 0 && act_n >= 960 && num_peaks >= 960;
    let eps_phases: &[i32; 3] = if dense_swarm_fast {
        &EPS_PHASES_DENSE
    } else {
        &EPS_PHASES_NORM
    };
    let n_phases = if dense_swarm_fast { 2usize } else { 3usize };
    let last_ph = n_phases - 1;
    let max_steps = (16 * act_n + 64) as u32;
    let mut total_steps = 0u32;
    let rev_sweep = (ms.next_micro_id & 1) != 0;
    let mut last_completed_eps_q = 0i32;
    let mut completed_final = false;
    let mut cap_triggered = false;
    let mut rev_built = false;

    unsafe {
        core::ptr::write_bytes(peak_price_ptr, 0, num_peaks);

        let mut ph = 0usize;
        while ph < n_phases && total_steps < max_steps {
            // Preserve already converged phase if remaining step budget cannot sustain a pass
            if ph > 0 && (max_steps - total_steps) < ((act_n + 16) as u32) {
                cap_triggered = true;
                break;
            }
            let eps_q = eps_phases[ph];
            if ph > 0 {
                let eps_prev = eps_phases[ph - 1];
                let relax = eps_prev + (eps_prev >> 2);
                #[cfg(target_arch = "wasm32")]
                let mut j = {
                    let vrelax = i32x4_splat(relax);
                    let vzero = i32x4_splat(0);
                    let mut j = 0usize;
                    while j + 4 <= num_peaks {
                        let vp = core::ptr::read_unaligned(peak_price_ptr.add(j) as *const v128);
                        core::ptr::write_unaligned(
                            peak_price_ptr.add(j) as *mut v128,
                            i32x4_max(i32x4_sub(vp, vrelax), vzero),
                        );
                        j += 4;
                    }
                    j
                };
                #[cfg(not(target_arch = "wasm32"))]
                let mut j = 0usize;
                while j < num_peaks {
                    let p = (*peak_price_ptr.add(j)).saturating_sub(relax);
                    *peak_price_ptr.add(j) = if p > 0 { p } else { 0 };
                    j += 1;
                }
                core::ptr::write_bytes(track_match_ptr, 0xFF, MAX_MICRO);
            }
            core::ptr::write_bytes(peak_owner_ptr, 0xFF, num_peaks);

            let mut q_head = 0usize;
            let mut q_tail = 0usize;
            let mut q_count = 0usize;
            let rev_phase = rev_sweep ^ ((ph & 1) != 0);
            let mut k = 0usize;
            while k < act_n {
                let ord_idx = if rev_phase { act_n - 1 - k } else { k };
                let ti = *track_order_ptr.add(ord_idx) as usize;
                if *csr_len_ptr.add(ti) > 0 {
                    *auction_q_ptr.add(q_tail) = ti as u16;
                    q_tail += 1;
                    if q_tail == MAX_MICRO {
                        q_tail = 0;
                    }
                    q_count += 1;
                }
                k += 1;
            }

            // Forward Auction pass
            let mut any_conflict = false;
            while q_count > 0 && total_steps < max_steps {
                let ti = *auction_q_ptr.add(q_head) as usize;
                q_head += 1;
                if q_head == MAX_MICRO {
                    q_head = 0;
                }
                q_count -= 1;
                total_steps += 1;

                let row_len = *csr_len_ptr.add(ti) as usize;
                let csr_base = ti * MAX_CSR_K;
                let c_coast_q =
                    if *state_ptr.add(ti) == MICRO_STATE_CONFIRMED && *coasting_ptr.add(ti) == 0 {
                        COAST_Q_CONF
                    } else {
                        COAST_Q_TENT
                    };

                const _: () = assert!(COAST_Q_CONF <= 4095 && COAST_Q_TENT <= 4095);
                let mut best_pi: isize = -1;
                let mut best_key: i32 = 4095; // net_q = 0, bonus = 4095 (private coast)
                let mut second_key: i32 = 4095;
                let mut best_cost_q: i32 = i32::MAX;

                let mut j = 0usize;
                while j < row_len {
                    let e = csr_base + j;
                    let pi = *csr_peak_ptr.add(e) as usize;
                    let cost_q = *csr_cost_ptr.add(e);
                    let net_q = (c_coast_q - cost_q) - *peak_price_ptr.add(pi);
                    let key = (net_q << 12) + (4095 - cost_q);
                    if key > best_key {
                        second_key = best_key;
                        best_key = key;
                        best_pi = pi as isize;
                        best_cost_q = cost_q;
                    } else if key > second_key {
                        second_key = key;
                    }
                    j += 1;
                }

                if best_pi >= 0 {
                    let win_pi = best_pi as usize;
                    let bid = ((best_key >> 12) - (second_key >> 12)) + eps_q;
                    let p_slot = peak_price_ptr.add(win_pi);
                    *p_slot = (*p_slot).saturating_add(bid);

                    let prev_owner = *peak_owner_ptr.add(win_pi);
                    if prev_owner >= 0 {
                        any_conflict = true;
                        let prev_ti = prev_owner as usize;
                        *track_match_ptr.add(prev_ti) = -1;
                        *auction_q_ptr.add(q_tail) = prev_ti as u16;
                        q_tail += 1;
                        if q_tail == MAX_MICRO {
                            q_tail = 0;
                        }
                        q_count += 1;
                    }
                    *peak_owner_ptr.add(win_pi) = ti as i16;
                    *track_match_ptr.add(ti) = win_pi as i16;
                    *match_cost_ptr.add(ti) = best_cost_q;
                }
            }

            // Fast global-optimality exit: on ph == 0 (`prices == 0`), zero contentions means exact epsilon = 0 optimum!
            if ph == 0 && !any_conflict && q_count == 0 {
                last_completed_eps_q = 4;
                completed_final = true;
                break;
            }

            // Reverse Auction cleanup pass on unowned peaks with `price > 0`: restores `0`-reserve
            // price complementary slackness (`peak_sort_tmp` = queue, `peak_matched` = queued flag).
            // On ph == 0 all `price > 0` peaks received bids from empty initial prices and remain owned.
            if q_count == 0 && total_steps < max_steps {
                if ph == 0 {
                    last_completed_eps_q = eps_q;
                    ph += 1;
                    continue;
                }
                let mut rq_head = 0usize;
                let mut rq_tail = 0usize;
                let mut rq_count = 0usize;
                core::ptr::write_bytes(peak_matched_ptr, 0, num_peaks);

                let mut pi = 0usize;
                while pi < num_peaks {
                    if *peak_owner_ptr.add(pi) < 0 && *peak_price_ptr.add(pi) > 0 {
                        *peak_sort_tmp_ptr.add(rq_tail) = pi as u16;
                        rq_tail = (rq_tail + 1) & PEAK_Q_MASK;
                        rq_count += 1;
                        *peak_matched_ptr.add(pi) = 1;
                    }
                    pi += 1;
                }

                // Lazily build reverse CSR (`rev_head` / `rev_next`) once per frame only when needed
                if rq_count > 0 && !rev_built {
                    core::ptr::write_bytes(rev_head_ptr, 0xFF, num_peaks);
                    let mut rk = 0usize;
                    while rk < act_n {
                        let rti = *track_order_ptr.add(rk) as usize;
                        let rlen = *csr_len_ptr.add(rti) as usize;
                        let csr_base = rti * MAX_CSR_K;
                        let mut rj = 0usize;
                        while rj < rlen {
                            let e = csr_base + rj;
                            let rpi = *csr_peak_ptr.add(e) as usize;
                            if rpi < num_peaks {
                                *rev_next_ptr.add(e) = *rev_head_ptr.add(rpi);
                                *rev_head_ptr.add(rpi) = e as i16;
                            } else {
                                *rev_next_ptr.add(e) = -1;
                            }
                            rj += 1;
                        }
                        rk += 1;
                    }
                    rev_built = true;
                }

                while rq_count > 0 && total_steps < max_steps {
                    let pi = *peak_sort_tmp_ptr.add(rq_head) as usize;
                    rq_head = (rq_head + 1) & PEAK_Q_MASK;
                    rq_count -= 1;
                    *peak_matched_ptr.add(pi) = 0;

                    if *peak_owner_ptr.add(pi) >= 0 || *peak_price_ptr.add(pi) <= 0 {
                        continue;
                    }
                    total_steps += 1;

                    let mut best_ti: isize = -1;
                    let mut best_key: i32 = 4095;
                    let mut sec_key: i32 = 4095;
                    let mut best_edge_cost: i32 = i32::MAX;

                    let mut e_idx = *rev_head_ptr.add(pi);
                    while e_idx >= 0 {
                        let e = e_idx as usize;
                        let ti = e >> 3;
                        let edge_cost = *csr_cost_ptr.add(e);
                        let cur_m = *track_match_ptr.add(ti);
                        // When matched (common case), c_coast_ti cancels cleanly out of w_e - u_ti!
                        let b = if cur_m >= 0 {
                            (*match_cost_ptr.add(ti) + *peak_price_ptr.add(cur_m as usize))
                                - edge_cost
                        } else {
                            let c_coast_ti = if *state_ptr.add(ti) == MICRO_STATE_CONFIRMED
                                && *coasting_ptr.add(ti) == 0
                            {
                                COAST_Q_CONF
                            } else {
                                COAST_Q_TENT
                            };
                            c_coast_ti - edge_cost
                        };
                        let key = (b << 12) + (4095 - edge_cost);
                        if key > best_key {
                            sec_key = best_key;
                            best_key = key;
                            best_ti = ti as isize;
                            best_edge_cost = edge_cost;
                        } else if key > sec_key {
                            sec_key = key;
                        }

                        e_idx = *rev_next_ptr.add(e);
                    }

                    if best_ti >= 0 {
                        let b_ti = best_ti as usize;
                        let cap_p = ((sec_key >> 12) - eps_q).max(0);
                        let p_slot = peak_price_ptr.add(pi);
                        let new_p = (*p_slot).min(cap_p);
                        *p_slot = new_p;

                        let prev_m = *track_match_ptr.add(b_ti);
                        if prev_m >= 0 {
                            let pm = prev_m as usize;
                            *peak_owner_ptr.add(pm) = -1;
                            if *peak_price_ptr.add(pm) > 0 && *peak_matched_ptr.add(pm) == 0 {
                                *peak_sort_tmp_ptr.add(rq_tail) = pm as u16;
                                rq_tail = (rq_tail + 1) & PEAK_Q_MASK;
                                rq_count += 1;
                                *peak_matched_ptr.add(pm) = 1;
                            }
                        }
                        *track_match_ptr.add(b_ti) = pi as i16;
                        *match_cost_ptr.add(b_ti) = best_edge_cost;
                        *peak_owner_ptr.add(pi) = b_ti as i16;
                    } else {
                        *peak_price_ptr.add(pi) = 0;
                    }
                }

                if rq_count == 0 && total_steps <= max_steps {
                    last_completed_eps_q = eps_q;
                    if ph == last_ph {
                        completed_final = true;
                    }
                } else {
                    cap_triggered = true;
                    break;
                }
            } else {
                cap_triggered = true;
                break;
            }
            ph += 1;
        }

        if total_steps >= max_steps && !completed_final {
            cap_triggered = true;
        }

        // Budget cap guard: retain 1-to-1 owners, run 1 deterministic greedy repair pass,
        // zero all unowned peak prices, and record explicit incomplete phase instead of
        // reporting a prior phase's `eps_q` as certified.
        if cap_triggered || !completed_final {
            ms.auction_incomplete = 1;
            ms.auction_eps_q = 0;
            ms.auction_cap_hits = ms.auction_cap_hits.saturating_add(1);
            let mut k = 0usize;
            while k < act_n {
                let ti = *track_order_ptr.add(k) as usize;
                if *track_match_ptr.add(ti) < 0 {
                    let row_len = *csr_len_ptr.add(ti) as usize;
                    let csr_base = ti * MAX_CSR_K;
                    let mut j = 0usize;
                    while j < row_len {
                        let pi = *csr_peak_ptr.add(csr_base + j) as usize;
                        if *peak_owner_ptr.add(pi) < 0 {
                            *peak_owner_ptr.add(pi) = ti as i16;
                            *track_match_ptr.add(ti) = pi as i16;
                            *ambig_ptr.add(ti) = 1;
                            ms.auction_repairs = ms.auction_repairs.saturating_add(1);
                            break;
                        }
                        j += 1;
                    }
                }
                k += 1;
            }
            let mut pi = 0usize;
            while pi < num_peaks {
                if *peak_owner_ptr.add(pi) < 0 {
                    *peak_price_ptr.add(pi) = 0;
                }
                pi += 1;
            }
        } else {
            ms.auction_incomplete = 0;
            ms.auction_eps_q = last_completed_eps_q as u32;
        }

        // Ambiguity margin check (< AMBIG_MARGIN_Q ~ 0.02 normalized gap) across both matched
        // assignment margin AND raw sorted CSR best-vs-second-best candidate separation
        let mut ambig = 0u32;
        let mut k = 0usize;
        while k < act_n {
            let ti = *track_order_ptr.add(k) as usize;
            let mpi = *track_match_ptr.add(ti);
            if mpi >= 0 {
                if *ambig_ptr.add(ti) == 0 {
                    let row_len = *csr_len_ptr.add(ti) as usize;
                    let csr_base = ti * MAX_CSR_K;
                    let c_coast_q = if *state_ptr.add(ti) == MICRO_STATE_CONFIRMED
                        && *coasting_ptr.add(ti) == 0
                    {
                        COAST_Q_CONF
                    } else {
                        COAST_Q_TENT
                    };
                    if row_len >= 2 && *csr_peak_ptr.add(csr_base) == mpi {
                        let c0 = *csr_cost_ptr.add(csr_base);
                        let c1 = *csr_cost_ptr.add(csr_base + 1);
                        if c1 - c0 < AMBIG_MARGIN_Q || c_coast_q - c0 < AMBIG_MARGIN_Q {
                            *ambig_ptr.add(ti) = 1;
                        }
                    } else if row_len == 1 && *csr_peak_ptr.add(csr_base) == mpi {
                        let c0 = *csr_cost_ptr.add(csr_base);
                        if c_coast_q - c0 < AMBIG_MARGIN_Q {
                            *ambig_ptr.add(ti) = 1;
                        }
                    } else {
                        if row_len >= 2 {
                            let c0 = *csr_cost_ptr.add(csr_base);
                            let c1 = *csr_cost_ptr.add(csr_base + 1);
                            if c1 - c0 < AMBIG_MARGIN_Q {
                                *ambig_ptr.add(ti) = 1;
                            }
                        }
                        if *ambig_ptr.add(ti) == 0 {
                            let mut matched_cost_q = c_coast_q;
                            let mut match_j = usize::MAX;
                            let mut j = 0usize;
                            while j < row_len {
                                let ej = csr_base + j;
                                if *csr_peak_ptr.add(ej) == mpi {
                                    matched_cost_q = *csr_cost_ptr.add(ej);
                                    match_j = j;
                                    break;
                                }
                                j += 1;
                            }
                            if c_coast_q - matched_cost_q < AMBIG_MARGIN_Q {
                                *ambig_ptr.add(ti) = 1;
                            } else if match_j < row_len {
                                // `csr_cost` within a track row is sorted ascending, so only adjacent entries can be closer
                                if (match_j > 0
                                    && matched_cost_q
                                        - (*csr_cost_ptr.add(csr_base + match_j - 1))
                                        < AMBIG_MARGIN_Q)
                                    || (match_j + 1 < row_len
                                        && (*csr_cost_ptr.add(csr_base + match_j + 1))
                                            - matched_cost_q
                                            < AMBIG_MARGIN_Q)
                                {
                                    *ambig_ptr.add(ti) = 1;
                                }
                            } else {
                                j = 0;
                                while j < row_len {
                                    let alt_q = *csr_cost_ptr.add(csr_base + j);
                                    if (alt_q - matched_cost_q).abs() < AMBIG_MARGIN_Q {
                                        *ambig_ptr.add(ti) = 1;
                                        break;
                                    }
                                    j += 1;
                                }
                            }
                        }
                    }
                }
                if *ambig_ptr.add(ti) != 0 {
                    ambig += 1;
                }
            }
            k += 1;
        }

        ms.ambiguous_count = ambig;
    }
    ms.auction_steps = total_steps;
}

/// Evaluates 8x8 bilinearly warped patch at `(cx, cy)` and accumulates true Huber IRLS
/// objective, L1 mean residual, Hessian `(H_xx, H_xy, H_yy)`, and gradient residual `(b_x, b_y)`.
#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn eval_lk_patch_wasm<const NEED_GRAD: bool, const ROWS4: bool>(
    luma: &[f32],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    tpl_e2: f32,
    lk_tpl_f: &[f32; 64],
    lk_gx: &[f32; 64],
    lk_gy: &[f32; 64],
    lk_warp_f: &mut [f32; 64],
) -> f32 {
    const HUBER_DELTA: f32 = 0.08;
    let sx = cx - 3.5;
    let sy = if ROWS4 { cy - 1.5 } else { cy - 3.5 };
    let max_sy = if ROWS4 { (h as f32) - 5.0 } else { (h as f32) - 9.0 };
    if !(sx >= 0.0 && sy >= 0.0 && sx <= (w as f32) - 9.0 && sy <= max_sy) {
        lk_warp_f[1] = f32::INFINITY;
        lk_warp_f[2] = 0.0;
        lk_warp_f[3] = 0.0;
        return f32::INFINITY;
    }
    let ix = (sx as u32) as usize;
    let iy = (sy as u32) as usize;
    let fx = sx - (ix as f32);
    let fy = sy - (iy as f32);

    #[inline(always)]
    unsafe fn load_horiz_row(lp: *const f32, p_row: usize, vfx: v128) -> (v128, v128) {
        let p = lp.add(p_row);
        let r0 = core::ptr::read_unaligned(p as *const v128);
        let r1 = core::ptr::read_unaligned(p.add(4) as *const v128);
        let r0_s = i32x4_shuffle::<1, 2, 3, 4>(r0, r1);
        let r1_s = core::ptr::read_unaligned(p.add(5) as *const v128);
        (
            f32x4_add(r0, f32x4_mul(f32x4_sub(r0_s, r0), vfx)),
            f32x4_add(r1, f32x4_mul(f32x4_sub(r1_s, r1), vfx)),
        )
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn accum_lk_row<const NEED_GRAD: bool>(
        ch: usize,
        iw0: v128,
        iw1: v128,
        tp: *const v128,
        gxp: *const v128,
        gyp: *const v128,
        vmean: v128,
        vhuber: v128,
        vnhuber: v128,
        vhalf: v128,
        vabs: &mut v128,
        vrho: &mut v128,
        vw2: &mut v128,
        vwt: &mut v128,
        vb_x: &mut v128,
        vb_y: &mut v128,
    ) {
        let tpl0 = core::ptr::read_unaligned(tp.add(ch));
        let tpl1 = core::ptr::read_unaligned(tp.add(ch + 1));
        let iw_zm0 = f32x4_sub(iw0, vmean);
        let iw_zm1 = f32x4_sub(iw1, vmean);
        let err0 = f32x4_sub(iw_zm0, tpl0);
        let err1 = f32x4_sub(iw_zm1, tpl1);
        let aerr0 = f32x4_abs(err0);
        let aerr1 = f32x4_abs(err1);
        let alo0 = f32x4_pmin(aerr0, vhuber);
        let alo1 = f32x4_pmin(aerr1, vhuber);
        let rho0 = f32x4_mul(alo0, f32x4_sub(aerr0, f32x4_mul(vhalf, alo0)));
        let rho1 = f32x4_mul(alo1, f32x4_sub(aerr1, f32x4_mul(vhalf, alo1)));
        *vabs = f32x4_add(*vabs, f32x4_add(aerr0, aerr1));
        *vrho = f32x4_add(*vrho, f32x4_add(rho0, rho1));
        *vw2 = f32x4_add(
            *vw2,
            f32x4_add(f32x4_mul(iw_zm0, iw_zm0), f32x4_mul(iw_zm1, iw_zm1)),
        );
        *vwt = f32x4_add(
            *vwt,
            f32x4_add(f32x4_mul(iw_zm0, tpl0), f32x4_mul(iw_zm1, tpl1)),
        );
        if NEED_GRAD {
            let gx0 = core::ptr::read_unaligned(gxp.add(ch));
            let gx1 = core::ptr::read_unaligned(gxp.add(ch + 1));
            let gy0 = core::ptr::read_unaligned(gyp.add(ch));
            let gy1 = core::ptr::read_unaligned(gyp.add(ch + 1));
            let psi0 = f32x4_pmin(f32x4_pmax(err0, vnhuber), vhuber);
            let psi1 = f32x4_pmin(f32x4_pmax(err1, vnhuber), vhuber);
            *vb_x = f32x4_add(
                *vb_x,
                f32x4_add(f32x4_mul(gx0, psi0), f32x4_mul(gx1, psi1)),
            );
            *vb_y = f32x4_add(
                *vb_y,
                f32x4_add(f32x4_mul(gy0, psi0), f32x4_mul(gy1, psi1)),
            );
        }
    }

    let inv_n = if ROWS4 { 1.0f32 / 32.0 } else { 1.0f32 / 64.0 };
    let (b_x, b_y, abs_sum, rho_sum, warp_e2, dot_wt) = unsafe {
        let lp = luma.as_ptr();
        let tp = lk_tpl_f.as_ptr() as *const v128;
        let gxp = lk_gx.as_ptr() as *const v128;
        let gyp = lk_gy.as_ptr() as *const v128;
        let wp = lk_warp_f.as_mut_ptr() as *mut v128;
        let vfx = f32x4_splat(fx);
        let vfy = f32x4_splat(fy);

        let mut p_row = iy * w + ix;
        let (mut hp0, mut hp1) = load_horiz_row(lp, p_row, vfx);
        let mut vsum = f32x4_splat(0.0);

        if ROWS4 {
            macro_rules! load_4_row {
                (adv) => {{
                    p_row += w;
                    let (hn0, hn1) = load_horiz_row(lp, p_row, vfx);
                    let wa = f32x4_add(hp0, f32x4_mul(f32x4_sub(hn0, hp0), vfy));
                    let wb = f32x4_add(hp1, f32x4_mul(f32x4_sub(hn1, hp1), vfy));
                    vsum = f32x4_add(vsum, f32x4_add(wa, wb));
                    hp0 = hn0;
                    hp1 = hn1;
                    (wa, wb)
                }};
                () => {{
                    p_row += w;
                    let (hn0, hn1) = load_horiz_row(lp, p_row, vfx);
                    let wa = f32x4_add(hp0, f32x4_mul(f32x4_sub(hn0, hp0), vfy));
                    let wb = f32x4_add(hp1, f32x4_mul(f32x4_sub(hn1, hp1), vfy));
                    vsum = f32x4_add(vsum, f32x4_add(wa, wb));
                    (wa, wb)
                }};
            }
            let (w0a, w0b) = load_4_row!(adv);
            let (w1a, w1b) = load_4_row!(adv);
            let (w2a, w2b) = load_4_row!(adv);
            let (w3a, w3b) = load_4_row!();
            let mean_i = hsum_f32x4(vsum) * inv_n;
            if !(mean_i.abs() < 1.0e30) {
                lk_warp_f[1] = f32::INFINITY;
                lk_warp_f[2] = 0.0;
                lk_warp_f[3] = 0.0;
                return f32::INFINITY;
            }
            let vmean = f32x4_splat(mean_i);
            let vhuber = f32x4_splat(HUBER_DELTA);
            let vnhuber = f32x4_splat(-HUBER_DELTA);
            let vhalf = f32x4_splat(0.5);
            let mut vb_x = f32x4_splat(0.0);
            let mut vb_y = f32x4_splat(0.0);
            let mut vabs = f32x4_splat(0.0);
            let mut vrho = f32x4_splat(0.0);
            let mut vw2 = f32x4_splat(0.0);
            let mut vwt = f32x4_splat(0.0);
            accum_lk_row::<NEED_GRAD>(
                0, w0a, w0b, tp, gxp, gyp, vmean, vhuber, vnhuber, vhalf, &mut vabs, &mut vrho,
                &mut vw2, &mut vwt, &mut vb_x, &mut vb_y,
            );
            accum_lk_row::<NEED_GRAD>(
                2, w1a, w1b, tp, gxp, gyp, vmean, vhuber, vnhuber, vhalf, &mut vabs, &mut vrho,
                &mut vw2, &mut vwt, &mut vb_x, &mut vb_y,
            );
            accum_lk_row::<NEED_GRAD>(
                4, w2a, w2b, tp, gxp, gyp, vmean, vhuber, vnhuber, vhalf, &mut vabs, &mut vrho,
                &mut vw2, &mut vwt, &mut vb_x, &mut vb_y,
            );
            accum_lk_row::<NEED_GRAD>(
                6, w3a, w3b, tp, gxp, gyp, vmean, vhuber, vnhuber, vhalf, &mut vabs, &mut vrho,
                &mut vw2, &mut vwt, &mut vb_x, &mut vb_y,
            );
            (
                if NEED_GRAD { hsum_f32x4(vb_x) } else { 0.0 },
                if NEED_GRAD { hsum_f32x4(vb_y) } else { 0.0 },
                hsum_f32x4(vabs),
                hsum_f32x4(vrho),
                hsum_f32x4(vw2),
                hsum_f32x4(vwt),
            )
        } else {
            let mut rr = 0usize;
            while rr < 8 {
                p_row += w;
                let (hn0, hn1) = load_horiz_row(lp, p_row, vfx);
                let wa = f32x4_add(hp0, f32x4_mul(f32x4_sub(hn0, hp0), vfy));
                let wb = f32x4_add(hp1, f32x4_mul(f32x4_sub(hn1, hp1), vfy));
                core::ptr::write_unaligned(wp.add(rr * 2), wa);
                core::ptr::write_unaligned(wp.add(rr * 2 + 1), wb);
                vsum = f32x4_add(vsum, f32x4_add(wa, wb));
                hp0 = hn0;
                hp1 = hn1;
                rr += 1;
            }

            let mean_i = hsum_f32x4(vsum) * inv_n;
            if !(mean_i.abs() < 1.0e30) {
                lk_warp_f[1] = f32::INFINITY;
                lk_warp_f[2] = 0.0;
                lk_warp_f[3] = 0.0;
                return f32::INFINITY;
            }

            let vmean = f32x4_splat(mean_i);
            let vhuber = f32x4_splat(HUBER_DELTA);
            let vnhuber = f32x4_splat(-HUBER_DELTA);
            let vhalf = f32x4_splat(0.5);
            let mut vb_x = f32x4_splat(0.0);
            let mut vb_y = f32x4_splat(0.0);
            let mut vabs = f32x4_splat(0.0);
            let mut vrho = f32x4_splat(0.0);
            let mut vw2 = f32x4_splat(0.0);
            let mut vwt = f32x4_splat(0.0);

            let wpc = wp as *const v128;
            let mut ch = 0usize;
            while ch < 16 {
                let iw0 = core::ptr::read_unaligned(wpc.add(ch));
                let iw1 = core::ptr::read_unaligned(wpc.add(ch + 1));
                accum_lk_row::<NEED_GRAD>(
                    ch, iw0, iw1, tp, gxp, gyp, vmean, vhuber, vnhuber, vhalf, &mut vabs, &mut vrho,
                    &mut vw2, &mut vwt, &mut vb_x, &mut vb_y,
                );
                ch += 2;
            }

            (
                if NEED_GRAD { hsum_f32x4(vb_x) } else { 0.0 },
                if NEED_GRAD { hsum_f32x4(vb_y) } else { 0.0 },
                hsum_f32x4(vabs),
                hsum_f32x4(vrho),
                hsum_f32x4(vw2),
                hsum_f32x4(vwt),
            )
        }
    };

    let loss = rho_sum * inv_n;
    let abs_res = abs_sum * inv_n;
    let min_e2 = if ROWS4 { 0.0004 } else { 0.0008 };
    let cond_ok = loss >= 0.0
        && loss < 1.0e30
        && abs_res < 1.0e30
        && b_x.abs() < 1.0e30
        && b_y.abs() < 1.0e30
        && warp_e2 >= min_e2
        && warp_e2 < 1.0e30
        && warp_e2 >= 0.05 * tpl_e2
        && dot_wt > 0.0
        && dot_wt < 1.0e30
        && dot_wt * dot_wt >= 0.04 * warp_e2 * tpl_e2;
    let loss_out = if cond_ok { loss } else { f32::INFINITY };
    lk_warp_f[1] = abs_res;
    lk_warp_f[2] = b_x;
    lk_warp_f[3] = b_y;
    loss_out
}

/// Evaluates 8x8 bilinearly warped patch at `(cx, cy)` and accumulates true Huber IRLS
/// objective, L1 mean residual, Hessian `(H_xx, H_xy, H_yy)`, and gradient residual `(b_x, b_y)`.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn eval_lk_patch<const NEED_GRAD: bool>(
    luma: &[f32],
    w: usize,
    h: usize,
    cx: f32,
    cy: f32,
    tpl_e2: f32,
    lk_tpl_f: &[f32; 64],
    lk_gx: &[f32; 64],
    lk_gy: &[f32; 64],
    lk_warp_f: &mut [f32; 64],
) -> (f32, f32, f32, f32, bool) {
    #[cfg(target_arch = "wasm32")]
    {
        let loss = eval_lk_patch_wasm::<NEED_GRAD, false>(
            luma, w, h, cx, cy, tpl_e2, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
        );
        (
            loss,
            lk_warp_f[1],
            lk_warp_f[2],
            lk_warp_f[3],
            loss < 1.0e30,
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        const HUBER_DELTA: f32 = 0.08;
        if !cx.is_finite() || !cy.is_finite() {
            return (f32::INFINITY, f32::INFINITY, 0.0, 0.0, false);
        }
        let sx = cx - 3.5;
        let sy = cy - 3.5;
        if sx < 0.0 || sy < 0.0 || sx > (w as f32) - 9.0 || sy > (h as f32) - 9.0 {
            return (f32::INFINITY, f32::INFINITY, 0.0, 0.0, false);
        }
        let ix = wasm_floorf(sx) as usize;
        let iy = wasm_floorf(sy) as usize;
        let fx = sx - (ix as f32);
        let fy = sy - (iy as f32);
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        let mut sum_i = 0.0f32;
        let mut rr = 0usize;
        while rr < 8 {
            let p0 = (iy + rr) * w + ix;
            let p1 = p0 + w;
            let mut cc = 0usize;
            while cc < 8 {
                let iw = luma[p0 + cc] * w00
                    + luma[p0 + cc + 1] * w10
                    + luma[p1 + cc] * w01
                    + luma[p1 + cc + 1] * w11;
                lk_warp_f[rr * 8 + cc] = iw;
                sum_i += iw;
                cc += 1;
            }
            rr += 1;
        }
        let mean_i = sum_i * (1.0 / 64.0);
        if !mean_i.is_finite() {
            return (f32::INFINITY, f32::INFINITY, 0.0, 0.0, false);
        }

        let (mut b_x, mut b_y, mut abs_sum, mut rho_sum, mut warp_e2, mut dot_wt) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut k = 0usize;
        while k < 64 {
            let iw_zm = lk_warp_f[k] - mean_i;
            let tpl = lk_tpl_f[k];
            let err = iw_zm - tpl;
            let aerr = err.abs();
            let alo = if aerr < HUBER_DELTA { aerr } else { HUBER_DELTA };
            abs_sum += aerr;
            rho_sum += alo * (aerr - 0.5 * alo);
            warp_e2 += iw_zm * iw_zm;
            dot_wt += iw_zm * tpl;
            if NEED_GRAD {
                let psi = clamp_f32(err, -HUBER_DELTA, HUBER_DELTA);
                b_x += lk_gx[k] * psi;
                b_y += lk_gy[k] * psi;
            }
            k += 1;
        }

        let loss = rho_sum * (1.0 / 64.0);
        let abs_res = abs_sum * (1.0 / 64.0);
        let cond_ok = loss.is_finite()
            && abs_res.is_finite()
            && b_x.is_finite()
            && b_y.is_finite()
            && warp_e2.is_finite()
            && dot_wt.is_finite()
            && warp_e2 >= 0.0008
            && warp_e2 >= 0.05 * tpl_e2
            && dot_wt > 0.0
            && dot_wt * dot_wt >= 0.04 * warp_e2 * tpl_e2;
        (loss, abs_res, b_x, b_y, cond_ok)
    }
}

/// Runs 2-iteration SIMD128 Huber IC-LK sub-pixel refinement on `luma` using `ms.tracks.tpl8_i8[ti]`
/// and arena-resident `lk_*` scratch (zero stack buffers >= 64 B). Enforces zero-mean projected
/// template gradients, monotonic true-Huber-loss atomic checkpointing with final-step verification,
/// radial vector step clamp `0.75 * q`, total clamp `1.5 * q`, and dual Hessian/SNR conditioning.
/// Returns `(refined_x, refined_y, mean_abs_res, gradient_trace, accepted)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn refine_ic_lk_8x8_2iter(
    luma: &[f32],
    w: usize,
    h: usize,
    q: f32,
    tpl_i8: &[i8; 64],
    x0: f32,
    y0: f32,
    max_shift: f32,
    lk_tpl_f: &mut [f32; 64],
    lk_gx: &mut [f32; 64],
    lk_gy: &mut [f32; 64],
    lk_warp_f: &mut [f32; 64],
    fast_lk: bool,
) -> (f32, f32, f32, f32, bool) {
    #[cfg(target_arch = "wasm32")]
    {
        let tr = refine_ic_lk_8x8_2iter_wasm(
            luma, w, h, q, tpl_i8, x0, y0, max_shift, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
            fast_lk,
        );
        (
            lk_warp_f[0],
            lk_warp_f[1],
            lk_warp_f[2],
            if tr > 0.0 { tr } else { 0.0 },
            tr > 0.0,
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    refine_ic_lk_8x8_2iter_scalar(
        luma, w, h, q, tpl_i8, x0, y0, max_shift, lk_tpl_f, lk_gx, lk_gy, lk_warp_f, fast_lk,
    )
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn refine_ic_lk_8x8_2iter_wasm(
    luma: &[f32],
    w: usize,
    h: usize,
    q: f32,
    tpl_i8: &[i8; 64],
    x0: f32,
    y0: f32,
    max_shift: f32,
    lk_tpl_f: &mut [f32; 64],
    lk_gx: &mut [f32; 64],
    lk_gy: &mut [f32; 64],
    lk_warp_f: &mut [f32; 64],
    fast_lk: bool,
) -> f32 {
    if fast_lk {
        refine_ic_lk_8x8_2iter_wasm_impl::<true>(
            luma, w, h, q, tpl_i8, x0, y0, max_shift, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
        )
    } else {
        refine_ic_lk_8x8_2iter_wasm_impl::<false>(
            luma, w, h, q, tpl_i8, x0, y0, max_shift, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
        )
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn prep_ic_lk_tpl<const FAST_4X8: bool>(
    tpl_i8: &[i8; 64],
    lk_tpl_f: &mut [f32; 64],
    lk_gx: &mut [f32; 64],
    lk_gy: &mut [f32; 64],
    out_stats: &mut [f32; 64],
) {
    let n_pix = if FAST_4X8 { 32.0f32 } else { 64.0f32 };
    let inv_n = if FAST_4X8 { 1.0f32 / 32.0 } else { 1.0f32 / 64.0 };
    let src = tpl_i8.as_ptr() as *const v128;
    let tp = lk_tpl_f.as_mut_ptr() as *mut v128;
    let gxp = lk_gx.as_mut_ptr() as *mut v128;
    let gyp = lk_gy.as_mut_ptr() as *mut v128;
    let b0 = core::ptr::read_unaligned(src);
    let b1 = core::ptr::read_unaligned(src.add(1));
    let b2 = core::ptr::read_unaligned(src.add(2));
    let b3 = core::ptr::read_unaligned(src.add(3));
    let (p1_32, psq) = if FAST_4X8 {
        let p1 = i16x8_add(
            i16x8_extadd_pairwise_i8x16(b1),
            i16x8_extadd_pairwise_i8x16(b2),
        );
        let sq1 = i32x4_add(
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(b1, b1)),
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(b1, b1)),
        );
        let sq2 = i32x4_add(
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(b2, b2)),
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(b2, b2)),
        );
        (i32x4_extadd_pairwise_i16x8(p1), i32x4_add(sq1, sq2))
    } else {
        let p1 = i16x8_add(
            i16x8_add(i16x8_extadd_pairwise_i8x16(b0), i16x8_extadd_pairwise_i8x16(b1)),
            i16x8_add(i16x8_extadd_pairwise_i8x16(b2), i16x8_extadd_pairwise_i8x16(b3)),
        );
        let sq0 = i32x4_add(
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(b0, b0)),
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(b0, b0)),
        );
        let sq1 = i32x4_add(
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(b1, b1)),
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(b1, b1)),
        );
        let sq2 = i32x4_add(
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(b2, b2)),
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(b2, b2)),
        );
        let sq3 = i32x4_add(
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(b3, b3)),
            i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(b3, b3)),
        );
        (
            i32x4_extadd_pairwise_i16x8(p1),
            i32x4_add(i32x4_add(sq0, sq1), i32x4_add(sq2, sq3)),
        )
    };
    let s1 = i32x4_extract_lane::<0>(p1_32)
        + i32x4_extract_lane::<1>(p1_32)
        + i32x4_extract_lane::<2>(p1_32)
        + i32x4_extract_lane::<3>(p1_32);
    let s2 = i32x4_extract_lane::<0>(psq)
        + i32x4_extract_lane::<1>(psq)
        + i32x4_extract_lane::<2>(psq)
        + i32x4_extract_lane::<3>(psq);
    let inv127 = 1.0f32 / 127.0;
    let vinv127 = f32x4_splat(inv127);
    let mean_tpl = (s1 as f32) * (inv127 * inv_n);
    let tpl_e2 = ((s2 as f32) * (inv127 * inv127) - n_pix * mean_tpl * mean_tpl).max(0.0);
    let vmean_tpl = f32x4_splat(mean_tpl);

    let sx_l = f32x4_replace_lane::<0>(f32x4_splat(0.5), 1.0);
    let sx_r = f32x4_replace_lane::<3>(f32x4_splat(0.5), 1.0);
    let vsy_half = f32x4_splat(0.5);
    let mut vsum_gx = f32x4_splat(0.0);
    let mut vsum_gy = f32x4_splat(0.0);
    let mut vh_xx = f32x4_splat(0.0);
    let mut vh_xy = f32x4_splat(0.0);
    let mut vh_yy = f32x4_splat(0.0);

    macro_rules! cvt_row_f32 {
        ($x16:expr) => {{
            let x16 = $x16;
            (
                f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(x16)), vinv127),
                f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(x16)), vinv127),
            )
        }};
    }

    if FAST_4X8 {
        let (mut r_prev_l, mut r_prev_r) = cvt_row_f32!(i16x8_extend_high_i8x16(b0));
        let (mut lvec, mut rvec) = cvt_row_f32!(i16x8_extend_low_i8x16(b1));
        macro_rules! emit_mid {
            ($r:expr, $next_l:expr, $next_r:expr, adv) => {{
                let cm_l = i32x4_shuffle::<0, 0, 1, 2>(lvec, lvec);
                let cp_l = i32x4_shuffle::<1, 2, 3, 4>(lvec, rvec);
                let cm_r = i32x4_shuffle::<3, 4, 5, 6>(lvec, rvec);
                let cp_r = i32x4_shuffle::<1, 2, 3, 3>(rvec, rvec);
                let gx_l = f32x4_mul(f32x4_sub(cp_l, cm_l), sx_l);
                let gx_r = f32x4_mul(f32x4_sub(cp_r, cm_r), sx_r);
                let gy_l = f32x4_mul(f32x4_sub($next_l, r_prev_l), vsy_half);
                let gy_r = f32x4_mul(f32x4_sub($next_r, r_prev_r), vsy_half);
                core::ptr::write_unaligned(tp.add($r * 2), f32x4_sub(lvec, vmean_tpl));
                core::ptr::write_unaligned(tp.add($r * 2 + 1), f32x4_sub(rvec, vmean_tpl));
                core::ptr::write_unaligned(gxp.add($r * 2), gx_l);
                core::ptr::write_unaligned(gxp.add($r * 2 + 1), gx_r);
                core::ptr::write_unaligned(gyp.add($r * 2), gy_l);
                core::ptr::write_unaligned(gyp.add($r * 2 + 1), gy_r);
                vsum_gx = f32x4_add(vsum_gx, f32x4_add(gx_l, gx_r));
                vsum_gy = f32x4_add(vsum_gy, f32x4_add(gy_l, gy_r));
                vh_xx = f32x4_add(vh_xx, f32x4_add(f32x4_mul(gx_l, gx_l), f32x4_mul(gx_r, gx_r)));
                vh_xy = f32x4_add(vh_xy, f32x4_add(f32x4_mul(gx_l, gy_l), f32x4_mul(gx_r, gy_r)));
                vh_yy = f32x4_add(vh_yy, f32x4_add(f32x4_mul(gy_l, gy_l), f32x4_mul(gy_r, gy_r)));
                r_prev_l = lvec;
                r_prev_r = rvec;
                lvec = $next_l;
                rvec = $next_r;
            }};
            ($r:expr, $next_l:expr, $next_r:expr, last) => {{
                let cm_l = i32x4_shuffle::<0, 0, 1, 2>(lvec, lvec);
                let cp_l = i32x4_shuffle::<1, 2, 3, 4>(lvec, rvec);
                let cm_r = i32x4_shuffle::<3, 4, 5, 6>(lvec, rvec);
                let cp_r = i32x4_shuffle::<1, 2, 3, 3>(rvec, rvec);
                let gx_l = f32x4_mul(f32x4_sub(cp_l, cm_l), sx_l);
                let gx_r = f32x4_mul(f32x4_sub(cp_r, cm_r), sx_r);
                let gy_l = f32x4_mul(f32x4_sub($next_l, r_prev_l), vsy_half);
                let gy_r = f32x4_mul(f32x4_sub($next_r, r_prev_r), vsy_half);
                core::ptr::write_unaligned(tp.add($r * 2), f32x4_sub(lvec, vmean_tpl));
                core::ptr::write_unaligned(tp.add($r * 2 + 1), f32x4_sub(rvec, vmean_tpl));
                core::ptr::write_unaligned(gxp.add($r * 2), gx_l);
                core::ptr::write_unaligned(gxp.add($r * 2 + 1), gx_r);
                core::ptr::write_unaligned(gyp.add($r * 2), gy_l);
                core::ptr::write_unaligned(gyp.add($r * 2 + 1), gy_r);
                vsum_gx = f32x4_add(vsum_gx, f32x4_add(gx_l, gx_r));
                vsum_gy = f32x4_add(vsum_gy, f32x4_add(gy_l, gy_r));
                vh_xx = f32x4_add(vh_xx, f32x4_add(f32x4_mul(gx_l, gx_l), f32x4_mul(gx_r, gx_r)));
                vh_xy = f32x4_add(vh_xy, f32x4_add(f32x4_mul(gx_l, gy_l), f32x4_mul(gx_r, gy_r)));
                vh_yy = f32x4_add(vh_yy, f32x4_add(f32x4_mul(gy_l, gy_l), f32x4_mul(gy_r, gy_r)));
            }};
        }
        let (n3l, n3r) = cvt_row_f32!(i16x8_extend_high_i8x16(b1));
        emit_mid!(0, n3l, n3r, adv);
        let (n4l, n4r) = cvt_row_f32!(i16x8_extend_low_i8x16(b2));
        emit_mid!(1, n4l, n4r, adv);
        let (n5l, n5r) = cvt_row_f32!(i16x8_extend_high_i8x16(b2));
        emit_mid!(2, n5l, n5r, adv);
        let (n6l, n6r) = cvt_row_f32!(i16x8_extend_low_i8x16(b3));
        emit_mid!(3, n6l, n6r, last);
    } else {
        let (mut r_prev_l, mut r_prev_r) = cvt_row_f32!(i16x8_extend_low_i8x16(b0));
        let mut lvec = r_prev_l;
        let mut rvec = r_prev_r;
        macro_rules! emit_tpl_row {
            ($r:expr, $next_l:expr, $next_r:expr, edge_adv) => {{
                let cm_l = i32x4_shuffle::<0, 0, 1, 2>(lvec, lvec);
                let cp_l = i32x4_shuffle::<1, 2, 3, 4>(lvec, rvec);
                let cm_r = i32x4_shuffle::<3, 4, 5, 6>(lvec, rvec);
                let cp_r = i32x4_shuffle::<1, 2, 3, 3>(rvec, rvec);
                let gx_l = f32x4_mul(f32x4_sub(cp_l, cm_l), sx_l);
                let gx_r = f32x4_mul(f32x4_sub(cp_r, cm_r), sx_r);
                let gy_l = f32x4_sub($next_l, r_prev_l);
                let gy_r = f32x4_sub($next_r, r_prev_r);
                core::ptr::write_unaligned(tp.add($r * 2), f32x4_sub(lvec, vmean_tpl));
                core::ptr::write_unaligned(tp.add($r * 2 + 1), f32x4_sub(rvec, vmean_tpl));
                core::ptr::write_unaligned(gxp.add($r * 2), gx_l);
                core::ptr::write_unaligned(gxp.add($r * 2 + 1), gx_r);
                core::ptr::write_unaligned(gyp.add($r * 2), gy_l);
                core::ptr::write_unaligned(gyp.add($r * 2 + 1), gy_r);
                vsum_gx = f32x4_add(vsum_gx, f32x4_add(gx_l, gx_r));
                vsum_gy = f32x4_add(vsum_gy, f32x4_add(gy_l, gy_r));
                vh_xx = f32x4_add(vh_xx, f32x4_add(f32x4_mul(gx_l, gx_l), f32x4_mul(gx_r, gx_r)));
                vh_xy = f32x4_add(vh_xy, f32x4_add(f32x4_mul(gx_l, gy_l), f32x4_mul(gx_r, gy_r)));
                vh_yy = f32x4_add(vh_yy, f32x4_add(f32x4_mul(gy_l, gy_l), f32x4_mul(gy_r, gy_r)));
                r_prev_l = lvec;
                r_prev_r = rvec;
                lvec = $next_l;
                rvec = $next_r;
            }};
            ($r:expr, $next_l:expr, $next_r:expr, last) => {{
                let cm_l = i32x4_shuffle::<0, 0, 1, 2>(lvec, lvec);
                let cp_l = i32x4_shuffle::<1, 2, 3, 4>(lvec, rvec);
                let cm_r = i32x4_shuffle::<3, 4, 5, 6>(lvec, rvec);
                let cp_r = i32x4_shuffle::<1, 2, 3, 3>(rvec, rvec);
                let gx_l = f32x4_mul(f32x4_sub(cp_l, cm_l), sx_l);
                let gx_r = f32x4_mul(f32x4_sub(cp_r, cm_r), sx_r);
                let gy_l = f32x4_sub($next_l, r_prev_l);
                let gy_r = f32x4_sub($next_r, r_prev_r);
                core::ptr::write_unaligned(tp.add($r * 2), f32x4_sub(lvec, vmean_tpl));
                core::ptr::write_unaligned(tp.add($r * 2 + 1), f32x4_sub(rvec, vmean_tpl));
                core::ptr::write_unaligned(gxp.add($r * 2), gx_l);
                core::ptr::write_unaligned(gxp.add($r * 2 + 1), gx_r);
                core::ptr::write_unaligned(gyp.add($r * 2), gy_l);
                core::ptr::write_unaligned(gyp.add($r * 2 + 1), gy_r);
                vsum_gx = f32x4_add(vsum_gx, f32x4_add(gx_l, gx_r));
                vsum_gy = f32x4_add(vsum_gy, f32x4_add(gy_l, gy_r));
                vh_xx = f32x4_add(vh_xx, f32x4_add(f32x4_mul(gx_l, gx_l), f32x4_mul(gx_r, gx_r)));
                vh_xy = f32x4_add(vh_xy, f32x4_add(f32x4_mul(gx_l, gy_l), f32x4_mul(gx_r, gy_r)));
                vh_yy = f32x4_add(vh_yy, f32x4_add(f32x4_mul(gy_l, gy_l), f32x4_mul(gy_r, gy_r)));
            }};
            ($r:expr, $next_l:expr, $next_r:expr, mid) => {{
                let cm_l = i32x4_shuffle::<0, 0, 1, 2>(lvec, lvec);
                let cp_l = i32x4_shuffle::<1, 2, 3, 4>(lvec, rvec);
                let cm_r = i32x4_shuffle::<3, 4, 5, 6>(lvec, rvec);
                let cp_r = i32x4_shuffle::<1, 2, 3, 3>(rvec, rvec);
                let gx_l = f32x4_mul(f32x4_sub(cp_l, cm_l), sx_l);
                let gx_r = f32x4_mul(f32x4_sub(cp_r, cm_r), sx_r);
                let gy_l = f32x4_mul(f32x4_sub($next_l, r_prev_l), vsy_half);
                let gy_r = f32x4_mul(f32x4_sub($next_r, r_prev_r), vsy_half);
                core::ptr::write_unaligned(tp.add($r * 2), f32x4_sub(lvec, vmean_tpl));
                core::ptr::write_unaligned(tp.add($r * 2 + 1), f32x4_sub(rvec, vmean_tpl));
                core::ptr::write_unaligned(gxp.add($r * 2), gx_l);
                core::ptr::write_unaligned(gxp.add($r * 2 + 1), gx_r);
                core::ptr::write_unaligned(gyp.add($r * 2), gy_l);
                core::ptr::write_unaligned(gyp.add($r * 2 + 1), gy_r);
                vsum_gx = f32x4_add(vsum_gx, f32x4_add(gx_l, gx_r));
                vsum_gy = f32x4_add(vsum_gy, f32x4_add(gy_l, gy_r));
                vh_xx = f32x4_add(vh_xx, f32x4_add(f32x4_mul(gx_l, gx_l), f32x4_mul(gx_r, gx_r)));
                vh_xy = f32x4_add(vh_xy, f32x4_add(f32x4_mul(gx_l, gy_l), f32x4_mul(gx_r, gy_r)));
                vh_yy = f32x4_add(vh_yy, f32x4_add(f32x4_mul(gy_l, gy_l), f32x4_mul(gy_r, gy_r)));
                r_prev_l = lvec;
                r_prev_r = rvec;
                lvec = $next_l;
                rvec = $next_r;
            }};
        }
        let (n1l, n1r) = cvt_row_f32!(i16x8_extend_high_i8x16(b0));
        emit_tpl_row!(0, n1l, n1r, edge_adv);
        let (n2l, n2r) = cvt_row_f32!(i16x8_extend_low_i8x16(b1));
        emit_tpl_row!(1, n2l, n2r, mid);
        let (n3l, n3r) = cvt_row_f32!(i16x8_extend_high_i8x16(b1));
        emit_tpl_row!(2, n3l, n3r, mid);
        let (n4l, n4r) = cvt_row_f32!(i16x8_extend_low_i8x16(b2));
        emit_tpl_row!(3, n4l, n4r, mid);
        let (n5l, n5r) = cvt_row_f32!(i16x8_extend_high_i8x16(b2));
        emit_tpl_row!(4, n5l, n5r, mid);
        let (n6l, n6r) = cvt_row_f32!(i16x8_extend_low_i8x16(b3));
        emit_tpl_row!(5, n6l, n6r, mid);
        let (n7l, n7r) = cvt_row_f32!(i16x8_extend_high_i8x16(b3));
        emit_tpl_row!(6, n7l, n7r, mid);
        emit_tpl_row!(7, lvec, rvec, last);
    }
    let mgx = hsum_f32x4(vsum_gx) * inv_n;
    let mgy = hsum_f32x4(vsum_gy) * inv_n;
    out_stats[0] = tpl_e2;
    out_stats[1] = (hsum_f32x4(vh_xx) - n_pix * mgx * mgx).max(0.0);
    out_stats[2] = hsum_f32x4(vh_xy) - n_pix * mgx * mgy;
    out_stats[3] = (hsum_f32x4(vh_yy) - n_pix * mgy * mgy).max(0.0);
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
unsafe fn prep_ic_lk_tpl_8x8(
    tpl_i8: &[i8; 64],
    lk_tpl_f: &mut [f32; 64],
    lk_gx: &mut [f32; 64],
    lk_gy: &mut [f32; 64],
    out_stats: &mut [f32; 64],
) {
    prep_ic_lk_tpl::<false>(tpl_i8, lk_tpl_f, lk_gx, lk_gy, out_stats);
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn refine_ic_lk_8x8_2iter_wasm_impl<const FAST_4X8: bool>(
    luma: &[f32],
    w: usize,
    h: usize,
    q: f32,
    tpl_i8: &[i8; 64],
    x0: f32,
    y0: f32,
    max_shift: f32,
    lk_tpl_f: &mut [f32; 64],
    lk_gx: &mut [f32; 64],
    lk_gy: &mut [f32; 64],
    lk_warp_f: &mut [f32; 64],
) -> f32 {
    assert!(luma.len() >= w * h);
    let min_c = 4.5f32;
    let max_wx = (w as f32) - 6.0;
    let max_wy = (h as f32) - 6.0;
    if !(q > 0.0
        && q < 1.0e30
        && x0 >= min_c
        && x0 <= max_wx
        && y0 >= min_c
        && y0 <= max_wy)
    {
        lk_warp_f[0] = x0;
        lk_warp_f[1] = y0;
        lk_warp_f[2] = 0.0;
        return -1.0;
    }

    unsafe {
        if FAST_4X8 {
            prep_ic_lk_tpl::<true>(tpl_i8, lk_tpl_f, lk_gx, lk_gy, lk_warp_f);
        } else {
            prep_ic_lk_tpl_8x8(tpl_i8, lk_tpl_f, lk_gx, lk_gy, lk_warp_f);
        }
    }
    let tpl_e2 = lk_warp_f[0];
    let h_xx = lk_warp_f[1];
    let h_xy = lk_warp_f[2];
    let h_yy = lk_warp_f[3];

    let tr_b = h_xx + h_yy;
    let det_raw = h_xx * h_yy - h_xy * h_xy;
    if !(tpl_e2 < 1.0e30
        && tr_b > 1e-5
        && tr_b < 1.0e30
        && det_raw > 1e-10
        && det_raw < 1.0e30
        && tr_b * tr_b <= 22.05 * det_raw)
    {
        lk_warp_f[0] = x0;
        lk_warp_f[1] = y0;
        lk_warp_f[2] = 0.0;
        return -1.0;
    }

    let snr_bound = 0.25 * q;
    let inv_snr2 = 1.0 / (snr_bound * snr_bound);
    let min_sigma_r = 0.004f32;
    let thresh_scale = if FAST_4X8 { 0.5f32 } else { 1.0f32 };
    let thresh0 = thresh_scale * (min_sigma_r * min_sigma_r) * inv_snr2;
    if tr_b < 2.0 * thresh0 || det_raw < thresh0 * (tr_b - thresh0) {
        lk_warp_f[0] = x0;
        lk_warp_f[1] = y0;
        lk_warp_f[2] = 0.0;
        return -1.0;
    }

    let loss0 = eval_lk_patch_wasm::<true, FAST_4X8>(
        luma, w, h, x0, y0, tpl_e2, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
    );
    let abs0 = lk_warp_f[1];
    let bx0 = lk_warp_f[2];
    let by0 = lk_warp_f[3];
    if !(loss0 < 1.0e30) {
        let safe_abs = if abs0 >= 0.0 && abs0 < 1.0e30 {
            abs0
        } else {
            0.0
        };
        lk_warp_f[0] = x0;
        lk_warp_f[1] = y0;
        lk_warp_f[2] = safe_abs;
        return -1.0;
    }

    let mut acc_x = x0;
    let mut acc_y = y0;
    let mut acc_loss = loss0;
    let mut acc_abs = abs0;
    let mut acc_bx = bx0;
    let mut acc_by = by0;

    let step_lim = 0.75 * q;
    let max_tot = if max_shift.abs() < 1.0e30 {
        max_shift.max(0.0).min(1.5 * q)
    } else {
        0.0
    };
    let max_tot2 = max_tot * max_tot;
    let damp = 0.005 * tr_b + 1e-6;
    let d_xx = h_xx + damp;
    let d_yy = h_yy + damp;
    let det = d_xx * d_yy - h_xy * h_xy;
    if det > 1e-12 && det < 1.0e30 {
        let inv_det = 1.0 / det;
        let mut iter = 0usize;
        while iter < 2 {
            let mut dx = -(d_yy * acc_bx - h_xy * acc_by) * inv_det;
            let mut dy = -(d_xx * acc_by - h_xy * acc_bx) * inv_det;
            if !(dx.abs() < 1.0e30 && dy.abs() < 1.0e30) {
                break;
            }
            let step2 = dx * dx + dy * dy;
            if step2 < if iter == 0 { 9e-4 } else { 1.6e-3 } {
                break;
            }
            if step2 > step_lim * step_lim {
                let sc = step_lim / wasm_sqrtf(step2);
                dx *= sc;
                dy *= sc;
            }
            let cand_x = clamp_f32(acc_x + dx, min_c, max_wx);
            let cand_y = clamp_f32(acc_y + dy, min_c, max_wy);
            let shift2 = (cand_x - x0) * (cand_x - x0) + (cand_y - y0) * (cand_y - y0);
            if !(shift2 <= max_tot2) {
                break;
            }
            if (cand_x - acc_x) * (cand_x - acc_x) + (cand_y - acc_y) * (cand_y - acc_y) < 1e-4 {
                break;
            }
            if iter == 0
                && (step2 < 0.004
                    || if FAST_4X8 {
                        step2 < 0.18 && acc_abs < 0.085
                    } else {
                        step2 < 0.036 && acc_abs < 0.075
                    })
            {
                acc_x = cand_x;
                acc_y = cand_y;
                break;
            }

            let stop_after_eval =
                !FAST_4X8 && iter == 0 && step2 < 0.09 && acc_abs < 0.065;
            let lc = if iter == 0 && !FAST_4X8 && !stop_after_eval {
                eval_lk_patch_wasm::<true, false>(
                    luma, w, h, cand_x, cand_y, tpl_e2, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
                )
            } else {
                eval_lk_patch_wasm::<false, FAST_4X8>(
                    luma, w, h, cand_x, cand_y, tpl_e2, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
                )
            };
            if !(lc < 1.0e30) || lc > acc_loss {
                break;
            }
            acc_x = cand_x;
            acc_y = cand_y;
            acc_loss = lc;
            acc_abs = lk_warp_f[1];
            if FAST_4X8 || stop_after_eval {
                break;
            }
            acc_bx = lk_warp_f[2];
            acc_by = lk_warp_f[3];
            iter += 1;
        }
    }

    // Dual noise/Hessian check on final verified checkpoint: sigma_r / sqrt(lambda_min) <= 0.25 * q
    let sigma_r = (1.2533 * acc_abs).max(0.004);
    let thresh1 = thresh_scale * (sigma_r * sigma_r) * inv_snr2;
    if tr_b < 2.0 * thresh1 || det_raw < thresh1 * (tr_b - thresh1) {
        lk_warp_f[0] = x0;
        lk_warp_f[1] = y0;
        lk_warp_f[2] = abs0;
        return -1.0;
    }

    lk_warp_f[0] = acc_x;
    lk_warp_f[1] = acc_y;
    lk_warp_f[2] = acc_abs;
    if FAST_4X8 { 2.0 * tr_b } else { tr_b }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn refine_ic_lk_8x8_2iter_scalar(
    luma: &[f32],
    w: usize,
    h: usize,
    q: f32,
    tpl_i8: &[i8; 64],
    x0: f32,
    y0: f32,
    max_shift: f32,
    lk_tpl_f: &mut [f32; 64],
    lk_gx: &mut [f32; 64],
    lk_gy: &mut [f32; 64],
    lk_warp_f: &mut [f32; 64],
    fast_lk: bool,
) -> (f32, f32, f32, f32, bool) {
    assert!(luma.len() >= w * h);
    let min_c = 4.5f32;
    let max_wx = (w as f32) - 6.0;
    let max_wy = (h as f32) - 6.0;
    if !q.is_finite()
        || q <= 0.0
        || !x0.is_finite()
        || !y0.is_finite()
        || x0 < min_c
        || x0 > max_wx
        || y0 < min_c
        || y0 > max_wy
    {
        return (x0, y0, 0.0, 0.0, false);
    }

    let (tpl_e2, mut h_xx, mut h_xy, mut h_yy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut sum_tpl = 0.0f32;
    let mut t2 = 0.0f32;
    let mut i = 0usize;
    while i < 64 {
        let v = (tpl_i8[i] as f32) * (1.0 / 127.0);
        lk_tpl_f[i] = v;
        sum_tpl += v;
        t2 += v * v;
        i += 1;
    }
    let mean_tpl = sum_tpl * (1.0 / 64.0);
    let _ = tpl_e2;
    let tpl_e2 = (t2 - 64.0 * mean_tpl * mean_tpl).max(0.0);

    let mut sum_gx = 0.0f32;
    let mut sum_gy = 0.0f32;
    let mut r = 0usize;
    while r < 8 {
        let rm = if r == 0 { 0 } else { r - 1 };
        let rp = if r == 7 { 7 } else { r + 1 };
        let y_scale = if r == 0 || r == 7 { 1.0 } else { 0.5 };
        let mut c = 0usize;
        while c < 8 {
            let cm = if c == 0 { 0 } else { c - 1 };
            let cp = if c == 7 { 7 } else { c + 1 };
            let x_scale = if c == 0 || c == 7 { 1.0 } else { 0.5 };
            let gx = (lk_tpl_f[r * 8 + cp] - lk_tpl_f[r * 8 + cm]) * x_scale;
            let gy = (lk_tpl_f[rp * 8 + c] - lk_tpl_f[rm * 8 + c]) * y_scale;
            lk_gx[r * 8 + c] = gx;
            lk_gy[r * 8 + c] = gy;
            sum_gx += gx;
            sum_gy += gy;
            h_xx += gx * gx;
            h_xy += gx * gy;
            h_yy += gy * gy;
            c += 1;
        }
        r += 1;
    }
    let mut ci = 0usize;
    while ci < 64 {
        lk_tpl_f[ci] -= mean_tpl;
        ci += 1;
    }
    let mean_gx = sum_gx * (1.0 / 64.0);
    let mean_gy = sum_gy * (1.0 / 64.0);
    h_xx = (h_xx - 64.0 * mean_gx * mean_gx).max(0.0);
    h_xy -= 64.0 * mean_gx * mean_gy;
    h_yy = (h_yy - 64.0 * mean_gy * mean_gy).max(0.0);

    let tr_b = h_xx + h_yy;
    let det_raw = h_xx * h_yy - h_xy * h_xy;
    if !tpl_e2.is_finite()
        || !tr_b.is_finite()
        || !det_raw.is_finite()
        || tr_b <= 1e-5
        || det_raw <= 1e-10
        || tr_b * tr_b > 22.05 * det_raw
    {
        return (x0, y0, 0.0, 0.0, false);
    }

    let diff_h = h_xx - h_yy;
    let disc_h = wasm_sqrtf((diff_h * diff_h + 4.0 * h_xy * h_xy).max(0.0));
    let lambda_min = (0.5 * (tr_b - disc_h)).max(0.0);
    let snr_bound = 0.25 * q;
    let min_sigma_r = 0.004f32;
    if min_sigma_r * min_sigma_r > snr_bound * snr_bound * lambda_min {
        return (x0, y0, 0.0, tr_b, false);
    }

    let (loss0, abs0, bx0, by0, ok0) = eval_lk_patch::<true>(
        luma, w, h, x0, y0, tpl_e2, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
    );
    if !ok0 {
        let safe_abs = if abs0.is_finite() { abs0 } else { 0.0 };
        return (x0, y0, safe_abs, tr_b, false);
    }

    let mut acc_x = x0;
    let mut acc_y = y0;
    let mut acc_loss = loss0;
    let mut acc_abs = abs0;
    let mut acc_bx = bx0;
    let mut acc_by = by0;

    let step_lim = 0.75 * q;
    let max_tot = if max_shift.abs() < 1.0e30 {
        max_shift.max(0.0).min(1.5 * q)
    } else {
        0.0
    };
    let max_tot2 = max_tot * max_tot;
    let damp = 0.005 * tr_b + 1e-6;
    let d_xx = h_xx + damp;
    let d_yy = h_yy + damp;
    let det = d_xx * d_yy - h_xy * h_xy;
    if det > 1e-12 && det < 1.0e30 {
        let inv_det = 1.0 / det;
        let mut iter = 0usize;
        while iter < 2 {
            let mut dx = -(d_yy * acc_bx - h_xy * acc_by) * inv_det;
            let mut dy = -(d_xx * acc_by - h_xy * acc_bx) * inv_det;
            if !(dx.abs() < 1.0e30 && dy.abs() < 1.0e30) {
                break;
            }
            let step2 = dx * dx + dy * dy;
            if step2 < if iter == 0 { 9e-4 } else { 1.6e-3 } {
                break;
            }
            if step2 > step_lim * step_lim {
                let sc = step_lim / wasm_sqrtf(step2);
                dx *= sc;
                dy *= sc;
            }
            let cand_x = clamp_f32(acc_x + dx, min_c, max_wx);
            let cand_y = clamp_f32(acc_y + dy, min_c, max_wy);
            let shift2 = (cand_x - x0) * (cand_x - x0) + (cand_y - y0) * (cand_y - y0);
            if !(shift2 <= max_tot2) {
                break;
            }
            if (cand_x - acc_x) * (cand_x - acc_x) + (cand_y - acc_y) * (cand_y - acc_y) < 1e-4 {
                break;
            }
            if iter == 0
                && (step2 < 0.004
                    || if fast_lk {
                        step2 < 0.49 && acc_abs < 0.115
                    } else {
                        step2 < 0.036 && acc_abs < 0.075
                    })
            {
                acc_x = cand_x;
                acc_y = cand_y;
                break;
            }

            let (lc, ac, bxc, byc, okc) = if iter == 0 && !fast_lk {
                eval_lk_patch::<true>(
                    luma, w, h, cand_x, cand_y, tpl_e2, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
                )
            } else {
                eval_lk_patch::<false>(
                    luma, w, h, cand_x, cand_y, tpl_e2, lk_tpl_f, lk_gx, lk_gy, lk_warp_f,
                )
            };
            if !okc || lc > acc_loss {
                break;
            }
            acc_x = cand_x;
            acc_y = cand_y;
            acc_loss = lc;
            acc_abs = ac;
            if fast_lk {
                break;
            }
            acc_bx = bxc;
            acc_by = byc;
            iter += 1;
        }
    }

    let sigma_r = (1.2533 * acc_abs).max(0.004);
    if sigma_r * sigma_r > snr_bound * snr_bound * lambda_min {
        return (x0, y0, abs0, tr_b, false);
    }

    (acc_x, acc_y, acc_abs, tr_b, true)
}

#[inline(always)]
unsafe fn apply_coast_or_retire_raw(
    ftle_a_frame_ptr: *mut u32,
    ftle_b_frame_ptr: *mut u32,
    ftle_val_ptr: *mut f32,
    coasting_ptr: *mut u8,
    age_ptr: *mut u16,
    state_ptr: *mut u8,
    track_id_ptr: *mut u32,
    ambig_ptr: *mut u8,
    body_id_ptr: *mut u8,
    pos_x_ptr: *mut f32,
    pos_y_ptr: *mut f32,
    vel_x_ptr: *mut f32,
    vel_y_ptr: *mut f32,
    cell_head_ptr: *mut i16,
    track_next_ptr: *mut i16,
    ti: usize,
    px: f32,
    py: f32,
    max_x: f32,
    max_y: f32,
    inv_cw: f32,
    inv_ch: f32,
) {
    // Invalidate FTLE immediately on first missed/coasting frame
    *ftle_a_frame_ptr.add(ti) = FTLE_INVALID_FRAME;
    *ftle_b_frame_ptr.add(ti) = FTLE_INVALID_FRAME;

    let c_next = (*coasting_ptr.add(ti)).saturating_add(1);
    let max_coast = if *age_ptr.add(ti) <= 2 {
        3
    } else {
        MAX_COAST_FRAMES
    };
    if c_next > max_coast
        || !px.is_finite()
        || !py.is_finite()
        || px < 2.0
        || px > max_x
        || py < 2.0
        || py > max_y
    {
        *state_ptr.add(ti) = MICRO_STATE_INACTIVE;
        *track_id_ptr.add(ti) = 0;
        *coasting_ptr.add(ti) = 0;
        *ambig_ptr.add(ti) = 0;
        *body_id_ptr.add(ti) = 0;
        *age_ptr.add(ti) = 0;
        *ftle_val_ptr.add(ti) = 0.0;
    } else {
        *ftle_val_ptr.add(ti) = FTLE_INVALID_VAL;
        *coasting_ptr.add(ti) = c_next;
        *state_ptr.add(ti) = MICRO_STATE_TENTATIVE;
        let clamped_x = clamp_f32(px, 2.0, max_x);
        let clamped_y = clamp_f32(py, 2.0, max_y);
        *pos_x_ptr.add(ti) = clamped_x;
        *pos_y_ptr.add(ti) = clamped_y;
        *vel_x_ptr.add(ti) *= 0.92;
        *vel_y_ptr.add(ti) *= 0.92;

        let cx = cell_coord(clamped_x, inv_cw, MICRO_CELL_W - 1);
        let cy = cell_coord(clamped_y, inv_ch, MICRO_CELL_H - 1);
        let cell = cy * MICRO_CELL_W + cx;
        let head_slot = cell_head_ptr.add(cell);
        *track_next_ptr.add(ti) = *head_slot;
        *head_slot = ti as i16;
    }
}

/// Inserts alive tracks into `track_cell_head[576]` and `track_next[1536]` (`32x18`).
#[allow(dead_code)]
#[inline(never)]
pub fn build_track_cell_hash(ms: &mut MicroSwarmState, w: usize, h: usize) {
    ms.track_cell_head.fill(-1);
    ms.track_next.fill(-1);
    let inv_cw = (MICRO_CELL_W as f32) / (w as f32);
    let inv_ch = (MICRO_CELL_H as f32) / (h as f32);

    let mut ti = MAX_MICRO;
    while ti > 0 {
        ti -= 1;
        if ms.tracks.state[ti] != MICRO_STATE_INACTIVE {
            let cx = cell_coord(ms.tracks.pos_x[ti], inv_cw, MICRO_CELL_W - 1);
            let cy = cell_coord(ms.tracks.pos_y[ti], inv_ch, MICRO_CELL_H - 1);
            let cell = cy * MICRO_CELL_W + cx;
            ms.track_next[ti] = ms.track_cell_head[cell];
            ms.track_cell_head[cell] = ti as i16;
        }
    }
}

#[inline(always)]
unsafe fn unlink_track_cell_raw(
    pos_x_ptr: *const f32,
    pos_y_ptr: *const f32,
    cell_head_ptr: *mut i16,
    track_next_ptr: *mut i16,
    ti: usize,
    inv_cw: f32,
    inv_ch: f32,
) {
    let cx = cell_coord(*pos_x_ptr.add(ti), inv_cw, MICRO_CELL_W - 1);
    let cy = cell_coord(*pos_y_ptr.add(ti), inv_ch, MICRO_CELL_H - 1);
    let cell = cy * MICRO_CELL_W + cx;
    let mut prev_idx: i16 = -1;
    let mut cur = *cell_head_ptr.add(cell);
    while cur >= 0 {
        let cu = cur as usize;
        let nxt = *track_next_ptr.add(cu);
        if cu == ti {
            if prev_idx < 0 {
                *cell_head_ptr.add(cell) = nxt;
            } else {
                *track_next_ptr.add(prev_idx as usize) = nxt;
            }
            *track_next_ptr.add(ti) = -1;
            break;
        }
        prev_idx = cur;
        cur = nxt;
    }
}

#[inline(always)]
fn unlink_track_cell(ms: &mut MicroSwarmState, ti: usize, inv_cw: f32, inv_ch: f32) {
    assert!(ti < MAX_MICRO);
    assert!(ms.tracks.pos_x.len() >= MAX_MICRO);
    assert!(ms.tracks.pos_y.len() >= MAX_MICRO);
    assert!(ms.track_cell_head.len() >= MICRO_CELLS);
    assert!(ms.track_next.len() >= MAX_MICRO);
    unsafe {
        unlink_track_cell_raw(
            ms.tracks.pos_x.as_ptr(),
            ms.tracks.pos_y.as_ptr(),
            ms.track_cell_head.as_mut_ptr(),
            ms.track_next.as_mut_ptr(),
            ti,
            inv_cw,
            inv_ch,
        );
    }
}

#[inline(always)]
unsafe fn has_nearby_active_track_raw_lim(
    state_ptr: *const u8,
    pos_x_ptr: *const f32,
    pos_y_ptr: *const f32,
    cell_head_ptr: *const i16,
    track_next_ptr: *const i16,
    px: f32,
    py: f32,
    r_lim: f32,
    r2_limit: f32,
    inv_cw: f32,
    inv_ch: f32,
) -> bool {
    let x_lo = cell_coord(px - r_lim, inv_cw, MICRO_CELL_W - 1);
    let x_hi = cell_coord(px + r_lim, inv_cw, MICRO_CELL_W - 1);
    let y_lo = cell_coord(py - r_lim, inv_ch, MICRO_CELL_H - 1);
    let y_hi = cell_coord(py + r_lim, inv_ch, MICRO_CELL_H - 1);

    let mut cy = y_lo;
    while cy <= y_hi {
        let row_off = cy * MICRO_CELL_W;
        let mut cx = x_lo;
        while cx <= x_hi {
            let mut idx = *cell_head_ptr.add(row_off + cx);
            while idx >= 0 {
                let ti = idx as usize;
                idx = *track_next_ptr.add(ti);
                if *state_ptr.add(ti) == MICRO_STATE_INACTIVE {
                    continue;
                }
                let dx = *pos_x_ptr.add(ti) - px;
                let dy = *pos_y_ptr.add(ti) - py;
                if (dx * dx + dy * dy) < r2_limit {
                    return true;
                }
            }
            cx += 1;
        }
        cy += 1;
    }
    false
}

#[inline(always)]
unsafe fn has_nearby_active_track_raw(
    state_ptr: *const u8,
    pos_x_ptr: *const f32,
    pos_y_ptr: *const f32,
    cell_head_ptr: *const i16,
    track_next_ptr: *const i16,
    px: f32,
    py: f32,
    r2_limit: f32,
    inv_cw: f32,
    inv_ch: f32,
) -> bool {
    has_nearby_active_track_raw_lim(
        state_ptr,
        pos_x_ptr,
        pos_y_ptr,
        cell_head_ptr,
        track_next_ptr,
        px,
        py,
        wasm_sqrtf(r2_limit.max(0.0)),
        r2_limit,
        inv_cw,
        inv_ch,
    )
}

/// Allocates a contiguous exact-f32 micro-track ID in `1..=MAX_MICRO_ID_F32` (`16,777,215`).
/// On 24-bit integer wrap, guarantees no active slot collision via linear scan.
#[inline(always)]
fn alloc_micro_track_id(ms: &mut MicroSwarmState) -> u32 {
    let mut cand = ms.next_micro_id;
    if cand == 0 || cand > MAX_MICRO_ID_F32 {
        cand = 1;
        ms.id_wrapped = 1;
    }
    if ms.id_wrapped != 0 {
        let mut tries = 0usize;
        while tries <= MAX_MICRO {
            let mut clash = false;
            let mut s = 0usize;
            while s < MAX_MICRO {
                if ms.tracks.state[s] != MICRO_STATE_INACTIVE && ms.tracks.track_id[s] == cand {
                    clash = true;
                    break;
                }
                s += 1;
            }
            if !clash {
                break;
            }
            cand += 1;
            if cand > MAX_MICRO_ID_F32 {
                cand = 1;
            }
            tries += 1;
        }
    }
    let mut next = cand + 1;
    if next > MAX_MICRO_ID_F32 {
        next = 1;
        ms.id_wrapped = 1;
    }
    ms.next_micro_id = next;
    ms.total_births_lifetime = ms.total_births_lifetime.saturating_add(1);
    cand
}

/// Applies 2-iter SIMD128 IC-LK sub-pixel refinement, IMM probability updates, lifecycle
/// transitions, global synchronized `20f`/`10f` FTLE anchor banks (strict reset on first missed,
/// coasting, or ambiguous frame), and greedy seeding of unmatched peaks.
#[inline(never)]
pub fn update_lifecycle_and_seed(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    w: usize,
    h: usize,
    safe_dt: f32,
    frame_idx: u32,
) {
    let (mut res_sum, mut res_cnt, reacq_cnt, mature_cnt, shock_cnt) =
        update_step1_matched(ms, luma, w, h, safe_dt);
    update_step2_coast_merge_direct(
        ms,
        luma,
        w,
        h,
        safe_dt,
        frame_idx,
        &mut res_sum,
        &mut res_cnt,
    );
    let births = update_step3_seed(ms, w, h, frame_idx);
    update_step4_ftle_tallies(
        ms,
        frame_idx,
        res_sum,
        res_cnt,
        reacq_cnt,
        mature_cnt,
        shock_cnt,
        births,
    );
}

#[inline(never)]
pub fn update_step1_matched(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    w: usize,
    h: usize,
    safe_dt: f32,
) -> (f32, u32, u32, u32, u32) {
    let q = (w as f32) * (1.0 / 640.0);
    let inv_dt = 1.0 / safe_dt.max(1e-4);
    let inv_2sig2 = 1.0 / (2.0 * (2.5 * q) * (2.5 * q)).max(0.5);
    let act_n = ms.active_count as usize;
    let num_peaks = ms.num_peaks;
    assert!(luma.len() >= w * h);
    assert!(act_n <= MAX_MICRO);
    assert!(num_peaks <= MAX_MICRO_PEAKS);
    assert!(ms.track_order.len() >= MAX_MICRO);
    assert!(ms.track_match.len() >= MAX_MICRO);
    assert!(ms.peak_matched.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_x.len() >= MAX_MICRO_PEAKS && ms.peak_y.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_rg.len() >= MAX_MICRO_PEAKS && ms.peak_by.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_sig.len() >= MAX_MICRO_PEAKS && ms.peak_tpl8.len() >= MAX_MICRO_PEAKS);
    assert!(ms.tracks.pos_x.len() >= MAX_MICRO && ms.tracks.pos_y.len() >= MAX_MICRO);
    assert!(ms.tracks.vel_x.len() >= MAX_MICRO && ms.tracks.vel_y.len() >= MAX_MICRO);
    assert!(ms.tracks.prev_x.len() >= MAX_MICRO && ms.tracks.prev_y.len() >= MAX_MICRO);
    assert!(ms.tracks.eq_x.len() >= MAX_MICRO && ms.tracks.eq_y.len() >= MAX_MICRO);
    assert!(ms.tracks.last_obs_x.len() >= MAX_MICRO && ms.tracks.last_obs_y.len() >= MAX_MICRO);
    assert!(ms.tracks.coasting.len() >= MAX_MICRO && ms.tracks.reacq_flag.len() >= MAX_MICRO);
    assert!(ms.tracks.age.len() >= MAX_MICRO && ms.tracks.state.len() >= MAX_MICRO);
    assert!(ms.tracks.ambiguous_flag.len() >= MAX_MICRO && ms.tracks.regime.len() >= MAX_MICRO);
    assert!(ms.tracks.ftle_a_frame.len() >= MAX_MICRO && ms.tracks.ftle_b_frame.len() >= MAX_MICRO);
    assert!(ms.tracks.ftle_val.len() >= MAX_MICRO);
    assert!(ms.tracks.imm_w_cv.len() >= MAX_MICRO && ms.tracks.imm_w_bal.len() >= MAX_MICRO);
    assert!(ms.tracks.imm_w_osc.len() >= MAX_MICRO);
    assert!(ms.tracks.curr_sig.len() >= MAX_MICRO);
    assert!(ms.tracks.chroma_rg.len() >= MAX_MICRO && ms.tracks.chroma_by.len() >= MAX_MICRO);
    assert!(ms.tracks.tpl8_i8.len() >= MAX_MICRO);

    let track_order_ptr = ms.track_order.as_ptr();
    let track_match_ptr = ms.track_match.as_ptr();
    let peak_matched_ptr = ms.peak_matched.as_mut_ptr();
    let peak_x_ptr = ms.peak_x.as_ptr();
    let peak_y_ptr = ms.peak_y.as_ptr();
    let peak_rg_ptr = ms.peak_rg.as_ptr();
    let peak_by_ptr = ms.peak_by.as_ptr();
    let peak_sig_ptr = ms.peak_sig.as_ptr();
    let peak_tpl8_ptr = ms.peak_tpl8.as_ptr();

    let pos_x_ptr = ms.tracks.pos_x.as_mut_ptr();
    let pos_y_ptr = ms.tracks.pos_y.as_mut_ptr();
    let vel_x_ptr = ms.tracks.vel_x.as_mut_ptr();
    let vel_y_ptr = ms.tracks.vel_y.as_mut_ptr();
    let prev_x_ptr = ms.tracks.prev_x.as_ptr();
    let prev_y_ptr = ms.tracks.prev_y.as_ptr();
    let eq_x_ptr = ms.tracks.eq_x.as_ptr();
    let eq_y_ptr = ms.tracks.eq_y.as_ptr();
    let last_obs_x_ptr = ms.tracks.last_obs_x.as_mut_ptr();
    let last_obs_y_ptr = ms.tracks.last_obs_y.as_mut_ptr();
    let coasting_ptr = ms.tracks.coasting.as_mut_ptr();
    let reacq_flag_ptr = ms.tracks.reacq_flag.as_mut_ptr();
    let age_ptr = ms.tracks.age.as_mut_ptr();
    let state_ptr = ms.tracks.state.as_mut_ptr();
    let ambig_ptr = ms.tracks.ambiguous_flag.as_mut_ptr();
    let regime_ptr = ms.tracks.regime.as_mut_ptr();
    let ftle_a_frame_ptr = ms.tracks.ftle_a_frame.as_mut_ptr();
    let ftle_b_frame_ptr = ms.tracks.ftle_b_frame.as_mut_ptr();
    let ftle_val_ptr = ms.tracks.ftle_val.as_mut_ptr();
    let imm_cv_ptr = ms.tracks.imm_w_cv.as_mut_ptr();
    let imm_bal_ptr = ms.tracks.imm_w_bal.as_mut_ptr();
    let imm_osc_ptr = ms.tracks.imm_w_osc.as_mut_ptr();
    let curr_sig_ptr = ms.tracks.curr_sig.as_mut_ptr();
    let chroma_rg_ptr = ms.tracks.chroma_rg.as_mut_ptr();
    let chroma_by_ptr = ms.tracks.chroma_by.as_mut_ptr();
    let tpl8_ptr = ms.tracks.tpl8_i8.as_mut_ptr();

    let fast_lk = (act_n >= 1024 && num_peaks >= 1500) || (act_n >= 1176 && num_peaks >= 1184);
    let advect_dist2 = (3.0 * q) * (3.0 * q);
    let mut res_sum = 0.0f32;
    let mut res_cnt = 0u32;
    let mut reacq_cnt = 0u32;
    let mut mature_cnt = 0u32;
    let mut shock_cnt = 0u32;

    // SAFETY: All pointers come from disjoint `ms` fields whose lengths were asserted above;
    // indices `ti < MAX_MICRO` and `pi < num_peaks <= MAX_MICRO_PEAKS` stay strictly in bounds.
    unsafe {
        core::ptr::write_bytes(peak_matched_ptr, 0, num_peaks);

        // Step 1: update matched tracks + compute 2-iter IC-LK sub-pixel refinement
        let mut ord_i = 0usize;
        while ord_i < act_n {
            let ti = *track_order_ptr.add(ord_i) as usize;
            let match_pi = *track_match_ptr.add(ti);
            if match_pi >= 0 {
                let pi = match_pi as usize;
                *peak_matched_ptr.add(pi) = 1;
                let px0 = *peak_x_ptr.add(pi);
                let py0 = *peak_y_ptr.add(pi);
                let (nx, ny, lk_res, ok) =
                    if fast_lk && *state_ptr.add(ti) != MICRO_STATE_CONFIRMED {
                    (px0, py0, -1.0, false)
                } else {
                        let track_fast_lk = fast_lk
                            || (act_n >= 800 && {
                                let dxe = px0 - *eq_x_ptr.add(ti);
                                let dye = py0 - *eq_y_ptr.add(ti);
                                dxe * dxe + dye * dye >= advect_dist2
                            });

                        let (rx, ry, lk_res, _, ok) = refine_ic_lk_8x8_2iter(
                            luma,
                            w,
                            h,
                            q,
                            &*tpl8_ptr.add(ti),
                            px0,
                            py0,
                            1.25 * q,
                            &mut ms.lk_tpl_f,
                            &mut ms.lk_gx,
                            &mut ms.lk_gy,
                            &mut ms.lk_warp_f,
                            track_fast_lk,
                        );
                        let nx = if ok && rx >= 0.0 && rx < 1.0e30 {
                            rx
                        } else {
                            px0
                        };
                        let ny = if ok && ry >= 0.0 && ry < 1.0e30 {
                            ry
                        } else {
                            py0
                        };
                        (nx, ny, lk_res, ok)
                    };
                if lk_res >= 0.0 && lk_res < 1.0e30 {
                    res_sum += lk_res;
                    res_cnt += 1;
                }

                let vx_prior = *vel_x_ptr.add(ti);
                let vy_prior = *vel_y_ptr.add(ti);
                let was_coasting = *coasting_ptr.add(ti);
                if was_coasting > 0 {
                    *reacq_flag_ptr.add(ti) = 1;
                    reacq_cnt += 1;
                    const INV_COAST_MUL: [f32; 9] = [
                        1.0,
                        1.0 / 2.0,
                        1.0 / 3.0,
                        1.0 / 4.0,
                        1.0 / 5.0,
                        1.0 / 6.0,
                        1.0 / 7.0,
                        1.0 / 8.0,
                        1.0 / 9.0,
                    ];
                    let eff_inv_dt = if (was_coasting as usize) <= 8 && safe_dt >= 1e-4 {
                        inv_dt * INV_COAST_MUL[was_coasting as usize]
                    } else {
                        1.0 / (safe_dt * ((was_coasting as f32) + 1.0)).max(1e-4)
                    };
                    let obs_vx = (nx - *last_obs_x_ptr.add(ti)) * eff_inv_dt;
                    let obs_vy = (ny - *last_obs_y_ptr.add(ti)) * eff_inv_dt;
                    *vel_x_ptr.add(ti) = 0.45 * vx_prior + 0.55 * obs_vx;
                    *vel_y_ptr.add(ti) = 0.45 * vy_prior + 0.55 * obs_vy;
                    *coasting_ptr.add(ti) = 0;
                    // Wait for next global synchronized bank tick before resuming FTLE
                    *ftle_a_frame_ptr.add(ti) = FTLE_INVALID_FRAME;
                    *ftle_b_frame_ptr.add(ti) = FTLE_INVALID_FRAME;
                    *ftle_val_ptr.add(ti) = FTLE_INVALID_VAL;
                } else {
                    *reacq_flag_ptr.add(ti) = 0;
                    let obs_vx = (nx - *last_obs_x_ptr.add(ti)) * inv_dt;
                    let obs_vy = (ny - *last_obs_y_ptr.add(ti)) * inv_dt;
                    let gain = if *age_ptr.add(ti) <= 1 { 0.90 } else { 0.72 };
                    *vel_x_ptr.add(ti) = vx_prior + gain * (obs_vx - vx_prior);
                    *vel_y_ptr.add(ti) = vy_prior + gain * (obs_vy - vy_prior);
                }

                if !ok || *ambig_ptr.add(ti) != 0 {
                    *ftle_a_frame_ptr.add(ti) = FTLE_INVALID_FRAME;
                    *ftle_b_frame_ptr.add(ti) = FTLE_INVALID_FRAME;
                    *ftle_val_ptr.add(ti) = FTLE_INVALID_VAL;
                }

                // Update IMM probabilities from prior-velocity model innovation residuals
                let x_old = *prev_x_ptr.add(ti);
                let y_old = *prev_y_ptr.add(ti);
                let e_cv2 = (nx - (x_old + vx_prior * safe_dt))
                    * (nx - (x_old + vx_prior * safe_dt))
                    + (ny - (y_old + vy_prior * safe_dt)) * (ny - (y_old + vy_prior * safe_dt));
                let e_bal2 = (nx - (x_old + 1.06 * vx_prior * safe_dt))
                    * (nx - (x_old + 1.06 * vx_prior * safe_dt))
                    + (ny - (y_old + 1.06 * vy_prior * safe_dt))
                        * (ny - (y_old + 1.06 * vy_prior * safe_dt));
                let ex_osc = *eq_x_ptr.add(ti)
                    + 0.95 * (x_old - *eq_x_ptr.add(ti))
                    + 0.72 * vx_prior * safe_dt;
                let ey_osc = *eq_y_ptr.add(ti)
                    + 0.95 * (y_old - *eq_y_ptr.add(ti))
                    + 0.72 * vy_prior * safe_dt;
                let e_osc2 = (nx - ex_osc) * (nx - ex_osc) + (ny - ey_osc) * (ny - ey_osc);

                #[cfg(target_arch = "wasm32")]
                {
                    let x = f32x4_mul(
                        f32x4_splat(-inv_2sig2),
                        f32x4(e_cv2, e_bal2, e_osc2, 20.0),
                    );
                    let xc = f32x4_pmin(f32x4_splat(20.0), f32x4_pmax(f32x4_splat(-20.0), x));
                    let t = f32x4_mul(xc, f32x4_splat(1.44269504089));
                    let k = f32x4_floor(t);
                    let r = f32x4_mul(f32x4_sub(t, k), f32x4_splat(0.69314718056));
                    let c1 = f32x4_splat(1.0);
                    let poly = f32x4_add(
                        c1,
                        f32x4_mul(
                            r,
                            f32x4_add(
                                c1,
                                f32x4_mul(
                                    r,
                                    f32x4_add(
                                        f32x4_splat(0.5),
                                        f32x4_mul(
                                            r,
                                            f32x4_add(
                                                f32x4_splat(0.16666667),
                                                f32x4_mul(r, f32x4_splat(0.04166667)),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    );
                    let ki = i32x4_shl(
                        i32x4_min(
                            i32x4_splat(254),
                            i32x4_max(
                                i32x4_splat(1),
                                i32x4_add(i32x4_trunc_sat_f32x4(k), i32x4_splat(127)),
                            ),
                        ),
                        23,
                    );
                    let exps = f32x4_mul(poly, ki);
                    let wv = f32x4(
                        *imm_cv_ptr.add(ti),
                        *imm_bal_ptr.add(ti),
                        *imm_osc_ptr.add(ti),
                        0.0,
                    );
                    let lv = f32x4_mul(wv, exps);
                    let sum_l = hsum_f32x4(lv) + 1e-6;
                    if sum_l > 0.0 && sum_l < 1.0e30 {
                        let wc = f32x4_replace_lane::<3>(
                            f32x4_pmin(
                                f32x4_splat(0.90),
                                f32x4_pmax(
                                    f32x4_splat(0.05),
                                    f32x4_mul(lv, f32x4_splat(1.0 / sum_l)),
                                ),
                            ),
                            0.0,
                        );
                        let wsum = hsum_f32x4(wc);
                        let wn = f32x4_mul(wc, f32x4_splat(1.0 / wsum));
                        let w0 = f32x4_extract_lane::<0>(wn);
                        let w1 = f32x4_extract_lane::<1>(wn);
                        let w2 = f32x4_extract_lane::<2>(wn);
                        *imm_cv_ptr.add(ti) = w0;
                        *imm_bal_ptr.add(ti) = w1;
                        *imm_osc_ptr.add(ti) = w2;
                        *regime_ptr.add(ti) = if w2 > w0 && w2 > w1 {
                            2
                        } else if w1 > w0 {
                            1
                        } else {
                            0
                        };
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let l_cv = *imm_cv_ptr.add(ti) * fast_expf(-e_cv2 * inv_2sig2);
                    let l_bal = *imm_bal_ptr.add(ti) * fast_expf(-e_bal2 * inv_2sig2);
                    let l_osc = *imm_osc_ptr.add(ti) * fast_expf(-e_osc2 * inv_2sig2);
                    let sum_l = l_cv + l_bal + l_osc + 1e-6;
                    if sum_l > 0.0 && sum_l < 1.0e30 {
                        let inv_sum_l = 1.0 / sum_l;
                        let w0 = clamp_f32(l_cv * inv_sum_l, 0.05, 0.90);
                        let w1 = clamp_f32(l_bal * inv_sum_l, 0.05, 0.90);
                        let w2 = clamp_f32(l_osc * inv_sum_l, 0.05, 0.90);
                        let inv_wsum = 1.0 / (w0 + w1 + w2);
                        *imm_cv_ptr.add(ti) = w0 * inv_wsum;
                        *imm_bal_ptr.add(ti) = w1 * inv_wsum;
                        *imm_osc_ptr.add(ti) = w2 * inv_wsum;
                        *regime_ptr.add(ti) = if w2 > w0 && w2 > w1 {
                            2
                        } else if w1 > w0 {
                            1
                        } else {
                            0
                        };
                    }
                }

                let prev_confirmed_mature = was_coasting == 0
                    && *state_ptr.add(ti) == MICRO_STATE_CONFIRMED
                    && *age_ptr.add(ti) >= 4;
                let pk_rg = *peak_rg_ptr.add(pi);
                let pk_by = *peak_by_ptr.add(pi);
                let d_rg = pk_rg - *chroma_rg_ptr.add(ti);
                let d_by = pk_by - *chroma_by_ptr.add(ti);
                let d_chr2 = d_rg * d_rg + d_by * d_by;
                let mut chr_shock = false;
                if prev_confirmed_mature {
                    mature_cnt += 1;
                    if d_chr2 >= 0.045 * 0.045 {
                        shock_cnt += 1;
                        chr_shock = true;
                        *ambig_ptr.add(ti) = 1;
                        *ftle_a_frame_ptr.add(ti) = FTLE_INVALID_FRAME;
                        *ftle_b_frame_ptr.add(ti) = FTLE_INVALID_FRAME;
                        *ftle_val_ptr.add(ti) = FTLE_INVALID_VAL;
                    }
                }

                *pos_x_ptr.add(ti) = nx;
                *pos_y_ptr.add(ti) = ny;
                *last_obs_x_ptr.add(ti) = nx;
                *last_obs_y_ptr.add(ti) = ny;
                *curr_sig_ptr.add(ti) = *peak_sig_ptr.add(pi);
                *chroma_rg_ptr.add(ti) = pk_rg;
                *chroma_by_ptr.add(ti) = pk_by;
                let new_age = (*age_ptr.add(ti)).saturating_add(1);
                *age_ptr.add(ti) = new_age;
                *state_ptr.add(ti) = if (ok || act_n >= 192) && new_age >= 2 {
                    MICRO_STATE_CONFIRMED
                } else {
                    MICRO_STATE_TENTATIVE
                };

                // Symmetric truncation-toward-zero template refresh ONLY when IC-LK accepted cleanly
                if ok && lk_res < 0.14 && !chr_shock {
                    #[cfg(target_arch = "wasm32")]
                    {
                        let dp = (*tpl8_ptr.add(ti)).as_mut_ptr() as *mut v128;
                        let sp = (*peak_tpl8_ptr.add(pi)).as_ptr() as *const v128;
                        let v3 = i16x8_splat(3);
                        let mut c = 0usize;
                        while c < 4 {
                            let vd = core::ptr::read_unaligned(dp.add(c));
                            let vs = core::ptr::read_unaligned(sp.add(c));
                            let d_lo = i16x8_extend_low_i8x16(vd);
                            let d_hi = i16x8_extend_high_i8x16(vd);
                            let s_lo = i16x8_extend_low_i8x16(vs);
                            let s_hi = i16x8_extend_high_i8x16(vs);
                            let b_lo = i16x8_add(i16x8_add(i16x8_shl(d_lo, 1), d_lo), s_lo);
                            let b_hi = i16x8_add(i16x8_add(i16x8_shl(d_hi, 1), d_hi), s_hi);
                            let q_lo = i16x8_shr(
                                i16x8_add(b_lo, v128_and(i16x8_shr(b_lo, 15), v3)),
                                2,
                            );
                            let q_hi = i16x8_shr(
                                i16x8_add(b_hi, v128_and(i16x8_shr(b_hi, 15), v3)),
                                2,
                            );
                            core::ptr::write_unaligned(dp.add(c), i8x16_narrow_i16x8(q_lo, q_hi));
                            c += 1;
                        }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let dst = &mut *tpl8_ptr.add(ti);
                        let src = &*peak_tpl8_ptr.add(pi);
                        let mut b = 0usize;
                        while b < 64 {
                            let blended = (dst[b] as i32) * 3 + (src[b] as i32);
                            dst[b] = (blended / 4).clamp(-127, 127) as i8;
                            b += 1;
                        }
                    }
                }
            }
            ord_i += 1;
        }
    }
    (res_sum, res_cnt, reacq_cnt, mature_cnt, shock_cnt)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn update_step2_direct_locks(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    w: usize,
    h: usize,
    q: f32,
    inv_dt: f32,
    inv_cw: f32,
    inv_ch: f32,
    max_x: f32,
    max_y: f32,
    r_dup2: f32,
    act_n: usize,
    res_sum: &mut f32,
    res_cnt: &mut u32,
) {
    let track_order_ptr = ms.track_order.as_ptr();
    let track_match_ptr = ms.track_match.as_mut_ptr();
    let track_next_ptr = ms.track_next.as_mut_ptr();
    let cell_head_ptr = ms.track_cell_head.as_mut_ptr();
    let score_ring_ptr = ms.score_ring.as_mut_ptr();
    let csr_cost_ptr = ms.csr_cost.as_mut_ptr();

    let pos_x_ptr = ms.tracks.pos_x.as_mut_ptr();
    let pos_y_ptr = ms.tracks.pos_y.as_mut_ptr();
    let vel_x_ptr = ms.tracks.vel_x.as_mut_ptr();
    let vel_y_ptr = ms.tracks.vel_y.as_mut_ptr();
    let pred_x_ptr = ms.tracks.pred_x.as_ptr();
    let pred_y_ptr = ms.tracks.pred_y.as_ptr();
    let last_obs_x_ptr = ms.tracks.last_obs_x.as_mut_ptr();
    let last_obs_y_ptr = ms.tracks.last_obs_y.as_mut_ptr();
    let state_ptr = ms.tracks.state.as_mut_ptr();
    let coasting_ptr = ms.tracks.coasting.as_mut_ptr();
    let reacq_flag_ptr = ms.tracks.reacq_flag.as_mut_ptr();
    let age_ptr = ms.tracks.age.as_mut_ptr();
    let track_id_ptr = ms.tracks.track_id.as_mut_ptr();
    let ambig_ptr = ms.tracks.ambiguous_flag.as_mut_ptr();
    let body_id_ptr = ms.tracks.body_id.as_mut_ptr();
    let ftle_a_frame_ptr = ms.tracks.ftle_a_frame.as_mut_ptr();
    let ftle_b_frame_ptr = ms.tracks.ftle_b_frame.as_mut_ptr();
    let ftle_val_ptr = ms.tracks.ftle_val.as_mut_ptr();
    let tpl8_ptr = ms.tracks.tpl8_i8.as_ptr();

    let fast_lk = act_n >= 1024 && ms.num_peaks >= 1500;
    let mut accept_cands = 0usize;
    let mut ord_i = 0usize;
    while ord_i < act_n {
        let ti = *track_order_ptr.add(ord_i) as usize;
        if *track_match_ptr.add(ti) == -1 {
            let px = *pred_x_ptr.add(ti);
            let py = *pred_y_ptr.add(ti);
            if has_nearby_active_track_raw(
                state_ptr,
                pos_x_ptr,
                pos_y_ptr,
                cell_head_ptr,
                track_next_ptr,
                px,
                py,
                r_dup2,
                inv_cw,
                inv_ch,
            ) {
                apply_coast_or_retire_raw(
                    ftle_a_frame_ptr,
                    ftle_b_frame_ptr,
                    ftle_val_ptr,
                    coasting_ptr,
                    age_ptr,
                    state_ptr,
                    track_id_ptr,
                    ambig_ptr,
                    body_id_ptr,
                    pos_x_ptr,
                    pos_y_ptr,
                    vel_x_ptr,
                    vel_y_ptr,
                    cell_head_ptr,
                    track_next_ptr,
                    ti,
                    px,
                    py,
                    max_x,
                    max_y,
                    inv_cw,
                    inv_ch,
                );
                *track_match_ptr.add(ti) = -2;
            } else {
                let (lx, ly, lk_res, grad_tr, ok) = refine_ic_lk_8x8_2iter(
                    luma,
                    w,
                    h,
                    q,
                    &*tpl8_ptr.add(ti),
                    px,
                    py,
                    1.5 * q,
                    &mut ms.lk_tpl_f,
                    &mut ms.lk_gx,
                    &mut ms.lk_gy,
                    &mut ms.lk_warp_f,
                    fast_lk,
                );
                if ok
                    && lk_res >= 0.0
                    && lk_res < 0.08
                    && grad_tr > 0.016
                    && grad_tr < 1.0e30
                    && lx >= 0.0
                    && lx < 1.0e30
                    && ly >= 0.0
                    && ly < 1.0e30
                    && !has_nearby_active_track_raw(
                        state_ptr,
                        pos_x_ptr,
                        pos_y_ptr,
                        cell_head_ptr,
                        track_next_ptr,
                        lx,
                        ly,
                        r_dup2,
                        inv_cw,
                        inv_ch,
                    )
                {
                    *score_ring_ptr.add(ti) = lx;
                    *csr_cost_ptr.add(ti * MAX_CSR_K + 1) = ly;
                    *csr_cost_ptr.add(ti * MAX_CSR_K) = lk_res;
                    *track_match_ptr.add(ti) = -3;
                    accept_cands += 1;
                } else {
                    apply_coast_or_retire_raw(
                        ftle_a_frame_ptr,
                        ftle_b_frame_ptr,
                        ftle_val_ptr,
                        coasting_ptr,
                        age_ptr,
                        state_ptr,
                        track_id_ptr,
                        ambig_ptr,
                        body_id_ptr,
                        pos_x_ptr,
                        pos_y_ptr,
                        vel_x_ptr,
                        vel_y_ptr,
                        cell_head_ptr,
                        track_next_ptr,
                        ti,
                        px,
                        py,
                        max_x,
                        max_y,
                        inv_cw,
                        inv_ch,
                    );
                    *track_match_ptr.add(ti) = -2;
                }
            }
        }
        ord_i += 1;
    }

    // Pass 2C: accept non-colliding direct IC-LK locks or fall back to coasting
    if accept_cands > 0 {
        ord_i = 0;
        while ord_i < act_n {
            let ti = *track_order_ptr.add(ord_i) as usize;
            if *track_match_ptr.add(ti) == -3 {
                let px = *pred_x_ptr.add(ti);
                let py = *pred_y_ptr.add(ti);
                let lx = *score_ring_ptr.add(ti);
                let ly = *csr_cost_ptr.add(ti * MAX_CSR_K + 1);
                let lk_res = *csr_cost_ptr.add(ti * MAX_CSR_K);

                if !has_nearby_active_track_raw(
                    state_ptr,
                    pos_x_ptr,
                    pos_y_ptr,
                    cell_head_ptr,
                    track_next_ptr,
                    px,
                    py,
                    r_dup2,
                    inv_cw,
                    inv_ch,
                ) && !has_nearby_active_track_raw(
                    state_ptr,
                    pos_x_ptr,
                    pos_y_ptr,
                    cell_head_ptr,
                    track_next_ptr,
                    lx,
                    ly,
                    r_dup2,
                    inv_cw,
                    inv_ch,
                ) {
                    let was_coasting = *coasting_ptr.add(ti);
                    let obs_vx = (lx - *last_obs_x_ptr.add(ti)) * inv_dt;
                    let obs_vy = (ly - *last_obs_y_ptr.add(ti)) * inv_dt;
                    *vel_x_ptr.add(ti) = 0.35 * *vel_x_ptr.add(ti) + 0.65 * obs_vx;
                    *vel_y_ptr.add(ti) = 0.35 * *vel_y_ptr.add(ti) + 0.65 * obs_vy;
                    *pos_x_ptr.add(ti) = lx;
                    *pos_y_ptr.add(ti) = ly;
                    *last_obs_x_ptr.add(ti) = lx;
                    *last_obs_y_ptr.add(ti) = ly;
                    *coasting_ptr.add(ti) = 0;
                    *reacq_flag_ptr.add(ti) = if was_coasting > 0 { 1 } else { 0 };
                    *age_ptr.add(ti) = (*age_ptr.add(ti)).saturating_add(1);
                    *state_ptr.add(ti) = MICRO_STATE_CONFIRMED;
                    *res_sum += lk_res;
                    *res_cnt += 1;

                    let cx = cell_coord(lx, inv_cw, MICRO_CELL_W - 1);
                    let cy = cell_coord(ly, inv_ch, MICRO_CELL_H - 1);
                    let cell = cy * MICRO_CELL_W + cx;
                    let head_slot = cell_head_ptr.add(cell);
                    *track_next_ptr.add(ti) = *head_slot;
                    *head_slot = ti as i16;
                } else {
                    apply_coast_or_retire_raw(
                        ftle_a_frame_ptr,
                        ftle_b_frame_ptr,
                        ftle_val_ptr,
                        coasting_ptr,
                        age_ptr,
                        state_ptr,
                        track_id_ptr,
                        ambig_ptr,
                        body_id_ptr,
                        pos_x_ptr,
                        pos_y_ptr,
                        vel_x_ptr,
                        vel_y_ptr,
                        cell_head_ptr,
                        track_next_ptr,
                        ti,
                        px,
                        py,
                        max_x,
                        max_y,
                        inv_cw,
                        inv_ch,
                    );
                }
            }
            ord_i += 1;
        }
    }
}

#[inline(never)]
pub fn update_step2_coast_merge_direct(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    w: usize,
    h: usize,
    safe_dt: f32,
    frame_idx: u32,
    res_sum: &mut f32,
    res_cnt: &mut u32,
) {
    let q = (w as f32) * (1.0 / 640.0);
    let inv_dt = 1.0 / safe_dt.max(1e-4);
    let inv_cw = (MICRO_CELL_W as f32) / (w as f32);
    let inv_ch = (MICRO_CELL_H as f32) / (h as f32);
    let r_nms = if w > 320 {
        R_NMS_HD as f32
    } else {
        R_NMS_SD as f32
    };
    let max_x = (w as f32) - 2.0;
    let max_y = (h as f32) - 2.0;
    let act_n = ms.active_count as usize;
    assert!(act_n <= MAX_MICRO);
    assert!(ms.track_order.len() >= MAX_MICRO);
    assert!(ms.track_match.len() >= MAX_MICRO);
    assert!(ms.track_next.len() >= MAX_MICRO);
    assert!(ms.track_cell_head.len() >= MICRO_CELLS);
    assert!(ms.cell_off.len() >= CSR_CELLS + 2);
    assert!(ms.cell_px.len() >= MAX_MICRO + 4 && ms.cell_py.len() >= MAX_MICRO + 4);
    assert!(ms.cell_pidx.len() >= MAX_MICRO);
    assert!(ms.csr_peak_idx.len() >= MAX_MICRO);
    assert!(ms.score_ring.len() >= MAX_MICRO);
    assert!(ms.csr_cost.len() >= MAX_MICRO * MAX_CSR_K);
    assert!(ms.tracks.pos_x.len() >= MAX_MICRO && ms.tracks.pos_y.len() >= MAX_MICRO);
    assert!(ms.tracks.vel_x.len() >= MAX_MICRO && ms.tracks.vel_y.len() >= MAX_MICRO);
    assert!(ms.tracks.pred_x.len() >= MAX_MICRO && ms.tracks.pred_y.len() >= MAX_MICRO);
    assert!(ms.tracks.last_obs_x.len() >= MAX_MICRO && ms.tracks.last_obs_y.len() >= MAX_MICRO);
    assert!(ms.tracks.state.len() >= MAX_MICRO && ms.tracks.coasting.len() >= MAX_MICRO);
    assert!(ms.tracks.reacq_flag.len() >= MAX_MICRO);
    assert!(ms.tracks.age.len() >= MAX_MICRO && ms.tracks.track_id.len() >= MAX_MICRO);
    assert!(ms.tracks.ambiguous_flag.len() >= MAX_MICRO && ms.tracks.body_id.len() >= MAX_MICRO);
    assert!(ms.tracks.ftle_a_frame.len() >= MAX_MICRO && ms.tracks.ftle_b_frame.len() >= MAX_MICRO);
    assert!(ms.tracks.ftle_val.len() >= MAX_MICRO && ms.tracks.tpl8_i8.len() >= MAX_MICRO);

    let track_order_ptr = ms.track_order.as_ptr();
    let track_match_ptr = ms.track_match.as_mut_ptr();
    let track_next_ptr = ms.track_next.as_mut_ptr();
    let cell_head_ptr = ms.track_cell_head.as_mut_ptr();

    let pos_x_ptr = ms.tracks.pos_x.as_mut_ptr();
    let pos_y_ptr = ms.tracks.pos_y.as_mut_ptr();
    let vel_x_ptr = ms.tracks.vel_x.as_mut_ptr();
    let vel_y_ptr = ms.tracks.vel_y.as_mut_ptr();
    let pred_x_ptr = ms.tracks.pred_x.as_ptr();
    let pred_y_ptr = ms.tracks.pred_y.as_ptr();
    let state_ptr = ms.tracks.state.as_mut_ptr();
    let coasting_ptr = ms.tracks.coasting.as_mut_ptr();
    let age_ptr = ms.tracks.age.as_mut_ptr();
    let track_id_ptr = ms.tracks.track_id.as_mut_ptr();
    let ambig_ptr = ms.tracks.ambiguous_flag.as_mut_ptr();
    let body_id_ptr = ms.tracks.body_id.as_mut_ptr();
    let ftle_a_frame_ptr = ms.tracks.ftle_a_frame.as_mut_ptr();
    let ftle_b_frame_ptr = ms.tracks.ftle_b_frame.as_mut_ptr();
    let ftle_val_ptr = ms.tracks.ftle_val.as_mut_ptr();

    // Step 2: Handle unmatched tracks (coast/retire non-direct-lock & intrinsic IC-LK failures
    // BEFORE accepting direct locks, so direct locks cannot land within `< r_dup` of a surviving coaster).
    let r_dup2 = (r_nms - 0.5) * (r_nms - 0.5);
    let mut direct_cands = 0usize;

    // SAFETY: All pointers below come from disjoint fields of `ms`, lengths are asserted above,
    // and track indices stored in `track_order`, `track_cell_head`, and `track_next` are in `0..MAX_MICRO`.
    unsafe {
        core::ptr::write_bytes(cell_head_ptr, 0xFF, MICRO_CELLS);

        // Pass 2A: insert matched tracks and finalize unmatched tracks ineligible for direct lock
        let mut ord_i = 0usize;
        while ord_i < act_n {
            let ti = *track_order_ptr.add(ord_i) as usize;
            if *track_match_ptr.add(ti) >= 0 {
                let cx = cell_coord(*pos_x_ptr.add(ti), inv_cw, MICRO_CELL_W - 1);
                let cy = cell_coord(*pos_y_ptr.add(ti), inv_ch, MICRO_CELL_H - 1);
                let cell = cy * MICRO_CELL_W + cx;
                let head_slot = cell_head_ptr.add(cell);
                *track_next_ptr.add(ti) = *head_slot;
                *head_slot = ti as i16;
            } else if !(*state_ptr.add(ti) == MICRO_STATE_CONFIRMED
                && *coasting_ptr.add(ti) == 0
                && *ambig_ptr.add(ti) == 0)
            {
                let px = *pred_x_ptr.add(ti);
                let py = *pred_y_ptr.add(ti);
                apply_coast_or_retire_raw(
                    ftle_a_frame_ptr,
                    ftle_b_frame_ptr,
                    ftle_val_ptr,
                    coasting_ptr,
                    age_ptr,
                    state_ptr,
                    track_id_ptr,
                    ambig_ptr,
                    body_id_ptr,
                    pos_x_ptr,
                    pos_y_ptr,
                    vel_x_ptr,
                    vel_y_ptr,
                    cell_head_ptr,
                    track_next_ptr,
                    ti,
                    px,
                    py,
                    max_x,
                    max_y,
                    inv_cw,
                    inv_ch,
                );
                *track_match_ptr.add(ti) = -2;
            } else {
                direct_cands += 1;
            }
            ord_i += 1;
        }

        // Prune co-moving duplicate tracks on saturated silhouettes (retire younger track < 5.25*q
        // while preserving crossing paths via velocity consistency) via contiguous 64x36 SIMD cells.
        if act_n >= 800 && ms.num_peaks >= 1420 && (frame_idx & 1) == 1 {
            let cell_off_ptr = ms.cell_off.as_mut_ptr();
            let fine_cell_ptr = ms.csr_peak_idx.as_mut_ptr();
            let cpx_ptr = ms.cell_px.as_mut_ptr();
            let cpy_ptr = ms.cell_py.as_mut_ptr();
            let cidx_ptr = ms.cell_pidx.as_mut_ptr();
            core::ptr::write_bytes(cell_off_ptr, 0, CSR_CELLS + 2);
            let inv_fcw = (CSR_CELL_W as f32) / (w as f32);
            let inv_fch = (CSR_CELL_H as f32) / (h as f32);
            ord_i = 0;
            while ord_i < act_n {
                let ti = *track_order_ptr.add(ord_i) as usize;
                if *track_match_ptr.add(ti) >= 0 {
                    let fcx = cell_coord(*pos_x_ptr.add(ti), inv_fcw, CSR_CELL_W - 1);
                    let fcy = cell_coord(*pos_y_ptr.add(ti), inv_fch, CSR_CELL_H - 1);
                    let fcell = fcy * CSR_CELL_W + fcx;
                    *fine_cell_ptr.add(ti) = fcell as i16;
                    *cell_off_ptr.add(fcell + 2) += 1;
                }
                ord_i += 1;
            }
            let mut acc = 0u16;
            let mut fc = 2usize;
            while fc < CSR_CELLS + 2 {
                acc += *cell_off_ptr.add(fc);
                *cell_off_ptr.add(fc) = acc;
                fc += 1;
            }
            ord_i = 0;
            while ord_i < act_n {
                let ti = *track_order_ptr.add(ord_i) as usize;
                if *track_match_ptr.add(ti) >= 0 {
                    let fcell = *fine_cell_ptr.add(ti) as usize;
                    let dst = *cell_off_ptr.add(fcell + 1) as usize;
                    *cell_off_ptr.add(fcell + 1) = (dst + 1) as u16;
                    *cpx_ptr.add(dst) = *pos_x_ptr.add(ti);
                    *cpy_ptr.add(dst) = *pos_y_ptr.add(ti);
                    *cidx_ptr.add(dst) = ti as u16;
                }
                ord_i += 1;
            }

            macro_rules! retire_dup_slot {
                ($slot:expr) => {{
                    let victim = $slot;
                    unlink_track_cell_raw(
                        pos_x_ptr,
                        pos_y_ptr,
                        cell_head_ptr,
                        track_next_ptr,
                        victim,
                        inv_cw,
                        inv_ch,
                    );
                    *state_ptr.add(victim) = MICRO_STATE_INACTIVE;
                    *track_id_ptr.add(victim) = 0;
                    *coasting_ptr.add(victim) = 0;
                    *ambig_ptr.add(victim) = 0;
                    *body_id_ptr.add(victim) = 0;
                    *age_ptr.add(victim) = 0;
                    *ftle_a_frame_ptr.add(victim) = FTLE_INVALID_FRAME;
                    *ftle_b_frame_ptr.add(victim) = FTLE_INVALID_FRAME;
                    *ftle_val_ptr.add(victim) = 0.0;
                    *track_match_ptr.add(victim) = -2;
                }};
            }

            let r_merge = 5.25 * q;
            let r_merge2 = r_merge * r_merge;
            let dv_lim = 4.0 * q * inv_dt;
            let dv_lim2 = dv_lim * dv_lim;
            #[cfg(target_arch = "wasm32")]
            let vr_merge2 = f32x4_splat(r_merge2);

            ord_i = 0;
            while ord_i < act_n {
                let ti = *track_order_ptr.add(ord_i) as usize;
                if *track_match_ptr.add(ti) >= 0 {
                    let px = *pos_x_ptr.add(ti);
                    let py = *pos_y_ptr.add(ti);
                    let x_lo = cell_coord(px - r_merge, inv_fcw, CSR_CELL_W - 1);
                    let x_hi = cell_coord(px + r_merge, inv_fcw, CSR_CELL_W - 1);
                    let y_lo = cell_coord(py - r_merge, inv_fch, CSR_CELL_H - 1);
                    let y_hi = cell_coord(py + r_merge, inv_fch, CSR_CELL_H - 1);
                    #[cfg(target_arch = "wasm32")]
                    let (v_px, v_py) = (f32x4_splat(px), f32x4_splat(py));
                    let mut dup = false;
                    let mut cy = y_lo;
                    while cy <= y_hi && !dup {
                        let row_off = cy * CSR_CELL_W;
                        let s_seg = *cell_off_ptr.add(row_off + x_lo) as usize;
                        let e_seg = (*cell_off_ptr.add(row_off + x_hi + 1) as usize).min(MAX_MICRO);
                        let mut k = s_seg;

                        #[cfg(target_arch = "wasm32")]
                        while k < e_seg {
                            let dx = f32x4_sub(
                                core::ptr::read_unaligned(cpx_ptr.add(k) as *const v128),
                                v_px,
                            );
                            let dy = f32x4_sub(
                                core::ptr::read_unaligned(cpy_ptr.add(k) as *const v128),
                                v_py,
                            );
                            let vd2 = f32x4_add(f32x4_mul(dx, dx), f32x4_mul(dy, dy));
                            let mut mask = i32x4_bitmask(f32x4_lt(vd2, vr_merge2)) as u32;
                            let rem = e_seg - k;
                            if rem < 4 {
                                mask &= (1u32 << rem) - 1;
                            }
                            while mask != 0 {
                                let lane = mask.trailing_zeros() as usize;
                                mask &= mask - 1;
                                let tj = *cidx_ptr.add(k + lane) as usize;
                                if tj > ti && *state_ptr.add(tj) != MICRO_STATE_INACTIVE {
                                    let dvx = *vel_x_ptr.add(tj) - *vel_x_ptr.add(ti);
                                    let dvy = *vel_y_ptr.add(tj) - *vel_y_ptr.add(ti);
                                    if dvx * dvx + dvy * dvy <= dv_lim2 {
                                        let age_j = *age_ptr.add(tj);
                                        let age_i = *age_ptr.add(ti);
                                        let tid_j = *track_id_ptr.add(tj);
                                        if age_j > age_i
                                            || (age_j == age_i && tid_j < *track_id_ptr.add(ti))
                                        {
                                            dup = true;
                                            break;
                                        } else {
                                            retire_dup_slot!(tj);
                                        }
                                    }
                                }
                            }
                            if dup {
                                break;
                            }
                            k += 4;
                        }

                        #[cfg(not(target_arch = "wasm32"))]
                        while k < e_seg {
                            let dx = *cpx_ptr.add(k) - px;
                            let dx2 = dx * dx;
                            if dx2 < r_merge2 {
                                let dy = *cpy_ptr.add(k) - py;
                                if dx2 + dy * dy < r_merge2 {
                                    let tj = *cidx_ptr.add(k) as usize;
                                    if tj > ti && *state_ptr.add(tj) != MICRO_STATE_INACTIVE {
                                        let dvx = *vel_x_ptr.add(tj) - *vel_x_ptr.add(ti);
                                        let dvy = *vel_y_ptr.add(tj) - *vel_y_ptr.add(ti);
                                        if dvx * dvx + dvy * dvy <= dv_lim2 {
                                            let age_j = *age_ptr.add(tj);
                                            let age_i = *age_ptr.add(ti);
                                            let tid_j = *track_id_ptr.add(tj);
                                            if age_j > age_i
                                                || (age_j == age_i && tid_j < *track_id_ptr.add(ti))
                                            {
                                                dup = true;
                                                break;
                                            } else {
                                                retire_dup_slot!(tj);
                                            }
                                        }
                                    }
                                }
                            }
                            k += 1;
                        }
                        cy += 1;
                    }
                    if dup {
                        retire_dup_slot!(ti);
                    }
                }
                ord_i += 1;
            }
        }
    }

    // Pass 2B/2C: cold direct IC-LK path extracted so `update_step2_coast_merge_direct` stays lean
    if direct_cands > 0 {
        unsafe {
            update_step2_direct_locks(
                ms, luma, w, h, q, inv_dt, inv_cw, inv_ch, max_x, max_y, r_dup2, act_n, res_sum,
                res_cnt,
            );
        }
    }
}

#[inline(never)]
pub fn update_step3_seed(
    ms: &mut MicroSwarmState,
    w: usize,
    h: usize,
    frame_idx: u32,
) -> u32 {
    let inv_cw = (MICRO_CELL_W as f32) / (w as f32);
    let inv_ch = (MICRO_CELL_H as f32) / (h as f32);
    let r_nms = if w > 320 {
        R_NMS_HD as f32
    } else {
        R_NMS_SD as f32
    };
    let act_n = ms.active_count as usize;

    // Step 3: Greedy seed strongest unmatched peaks into free slots (evicting stale >=3f coasters if full)
    let mut births = 0u32;
    let mut free_cursor = 0usize;
    let mut evict_cursor = 0usize;
    let dense_q4_seed = ms.swarm_active != 0 && ms.confirmed_count >= 1045 && ms.num_peaks >= 1500;
    let r_seed = if dense_q4_seed {
        r_nms + 1.5
    } else if act_n >= 800 && ms.num_peaks >= 1500 {
        r_nms + 0.5
    } else {
        r_nms
    };
    assert!(ms.num_peaks <= MAX_MICRO_PEAKS);
    assert!(ms.peak_matched.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_x.len() >= MAX_MICRO_PEAKS && ms.peak_y.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_rg.len() >= MAX_MICRO_PEAKS && ms.peak_by.len() >= MAX_MICRO_PEAKS);
    assert!(ms.peak_sig.len() >= MAX_MICRO_PEAKS && ms.peak_tpl8.len() >= MAX_MICRO_PEAKS);
    assert!(ms.tracks.state.len() >= MAX_MICRO && ms.tracks.coasting.len() >= MAX_MICRO);
    assert!(ms.tracks.pos_x.len() >= MAX_MICRO && ms.tracks.pos_y.len() >= MAX_MICRO);
    assert!(ms.track_cell_head.len() >= MICRO_CELLS && ms.track_next.len() >= MAX_MICRO);

    let r_seed2 = r_seed * r_seed;
    let r_seed_lim = r_seed.max(0.0);
    let mut pi = 0usize;
    unsafe {
        let pm_p = ms.peak_matched.as_ptr();
        let px_p = ms.peak_x.as_ptr();
        let py_p = ms.peak_y.as_ptr();
        let st_p = ms.tracks.state.as_ptr();
        let coast_p = ms.tracks.coasting.as_ptr();
        let tx_p = ms.tracks.pos_x.as_ptr();
        let ty_p = ms.tracks.pos_y.as_ptr();
        let head_p = ms.track_cell_head.as_ptr();
        let next_p = ms.track_next.as_ptr();
        let n_pks = ms.num_peaks;
        while pi < n_pks {
            if *pm_p.add(pi) == 0 {
                while free_cursor < MAX_MICRO && *st_p.add(free_cursor) != MICRO_STATE_INACTIVE {
                    free_cursor += 1;
                }
                if free_cursor >= MAX_MICRO {
                    if dense_q4_seed && pi >= 1360 {
                        break;
                    }
                    while evict_cursor < MAX_MICRO
                        && !(*st_p.add(evict_cursor) != MICRO_STATE_INACTIVE
                            && *coast_p.add(evict_cursor) >= 3)
                    {
                        evict_cursor += 1;
                    }
                    if evict_cursor >= MAX_MICRO {
                        break;
                    }
                }
                let px = *px_p.add(pi);
                let py = *py_p.add(pi);
                let near = has_nearby_active_track_raw_lim(
                    st_p, tx_p, ty_p, head_p, next_p, px, py, r_seed_lim, r_seed2, inv_cw,
                    inv_ch,
                );
                if !near {
                    let mut used_eviction = false;
                    let ti = if free_cursor < MAX_MICRO {
                        let idx = free_cursor;
                        free_cursor += 1;
                        idx
                    } else {
                        let idx = evict_cursor;
                        evict_cursor += 1;
                        used_eviction = true;
                        idx
                    };
                    if used_eviction {
                        unlink_track_cell(ms, ti, inv_cw, inv_ch);
                    }

                    ms.tracks.track_id[ti] = alloc_micro_track_id(ms);
                    ms.tracks.birth_frame[ti] = frame_idx;
                    ms.tracks.age[ti] = 1;
                    ms.tracks.state[ti] = MICRO_STATE_TENTATIVE;
                    ms.tracks.coasting[ti] = 0;
                    ms.tracks.reacq_flag[ti] = 0;
                    ms.tracks.ambiguous_flag[ti] = 0;
                    ms.tracks.body_id[ti] = 0;
                    ms.tracks.regime[ti] = 0;

                    ms.tracks.pos_x[ti] = px;
                    ms.tracks.pos_y[ti] = py;
                    ms.tracks.prev_x[ti] = px;
                    ms.tracks.prev_y[ti] = py;
                    ms.tracks.last_obs_x[ti] = px;
                    ms.tracks.last_obs_y[ti] = py;
                    ms.tracks.pred_x[ti] = px;
                    ms.tracks.pred_y[ti] = py;
                    ms.tracks.vel_x[ti] = 0.0;
                    ms.tracks.vel_y[ti] = 0.0;
                    ms.tracks.imm_w_cv[ti] = 0.60;
                    ms.tracks.imm_w_bal[ti] = 0.20;
                    ms.tracks.imm_w_osc[ti] = 0.20;

                    ms.tracks.ftle_a_x0[ti] = px;
                    ms.tracks.ftle_a_y0[ti] = py;
                    ms.tracks.ftle_b_x0[ti] = px;
                    ms.tracks.ftle_b_y0[ti] = py;
                    ms.tracks.ftle_a_frame[ti] = FTLE_INVALID_FRAME;
                    ms.tracks.ftle_b_frame[ti] = FTLE_INVALID_FRAME;
                    ms.tracks.ftle_val[ti] = FTLE_INVALID_VAL;

                    let sig = *ms.peak_sig.as_ptr().add(pi);
                    ms.tracks.anchor_sig[ti] = sig;
                    ms.tracks.curr_sig[ti] = sig;
                    ms.tracks.chroma_rg[ti] = *ms.peak_rg.as_ptr().add(pi);
                    ms.tracks.chroma_by[ti] = *ms.peak_by.as_ptr().add(pi);

                    ms.tracks.eq_x[ti] = px;
                    ms.tracks.eq_y[ti] = py;
                    ms.tracks.bp_i[ti] = 0.0;
                    ms.tracks.bp_q[ti] = 0.0;
                    ms.tracks.bp_prev_y[ti] = 0.0;
                    ms.tracks.disp_y[ti] = 0.0;
                    ms.tracks.modal_phase[ti] = 0.0;
                    ms.tracks.modal_freq_hz[ti] = 0.0;
                    ms.tracks.modal_amp[ti] = 0.0;
                    ms.tracks.tpl8_i8[ti] = *ms.peak_tpl8.as_ptr().add(pi);

                    let cx = cell_coord(px, inv_cw, MICRO_CELL_W - 1);
                    let cy = cell_coord(py, inv_ch, MICRO_CELL_H - 1);
                    let cell = cy * MICRO_CELL_W + cx;
                    ms.track_next[ti] = ms.track_cell_head[cell];
                    ms.track_cell_head[cell] = ti as i16;
                    births += 1;
                }
            }
            pi += 1;
        }
    }
    births
}

#[inline(never)]
pub fn update_step4_ftle_tallies(
    ms: &mut MicroSwarmState,
    frame_idx: u32,
    res_sum: f32,
    res_cnt: u32,
    reacq_cnt: u32,
    mature_cnt: u32,
    shock_cnt: u32,
    births: u32,
) {
    // Step 4: Staggered 20-frame FTLE anchor banks (A@0, B@10 + immediate seed/backfill) & tallies
    let cohort_shock = mature_cnt >= 16 && shock_cnt * 4 >= mature_cnt;
    let phase20 = frame_idx % FTLE_BANK_PERIOD;
    let mut act_cnt = 0u32;
    let mut conf_cnt = 0u32;
    let mut tent_cnt = 0u32;
    let mut coast_cnt = 0u32;

    assert!(ms.tracks.state.len() >= MAX_MICRO && ms.tracks.coasting.len() >= MAX_MICRO);
    assert!(ms.tracks.ambiguous_flag.len() >= MAX_MICRO && ms.tracks.reacq_flag.len() >= MAX_MICRO);
    assert!(
        ms.tracks.ftle_a_frame.len() >= MAX_MICRO
            && ms.tracks.ftle_b_frame.len() >= MAX_MICRO
            && ms.tracks.ftle_val.len() >= MAX_MICRO
    );
    assert!(
        ms.tracks.ftle_a_x0.len() >= MAX_MICRO
            && ms.tracks.ftle_a_y0.len() >= MAX_MICRO
            && ms.tracks.ftle_b_x0.len() >= MAX_MICRO
            && ms.tracks.ftle_b_y0.len() >= MAX_MICRO
    );
    assert!(ms.tracks.pos_x.len() >= MAX_MICRO && ms.tracks.pos_y.len() >= MAX_MICRO);

    let st_p = ms.tracks.state.as_ptr();
    let coast_p = ms.tracks.coasting.as_ptr();
    let ambig_p = ms.tracks.ambiguous_flag.as_mut_ptr();
    let reacq_p = ms.tracks.reacq_flag.as_ptr();
    let fa_frame_p = ms.tracks.ftle_a_frame.as_mut_ptr();
    let fb_frame_p = ms.tracks.ftle_b_frame.as_mut_ptr();
    let fval_p = ms.tracks.ftle_val.as_mut_ptr();
    let fa_x0_p = ms.tracks.ftle_a_x0.as_mut_ptr();
    let fa_y0_p = ms.tracks.ftle_a_y0.as_mut_ptr();
    let fb_x0_p = ms.tracks.ftle_b_x0.as_mut_ptr();
    let fb_y0_p = ms.tracks.ftle_b_y0.as_mut_ptr();
    let px_p = ms.tracks.pos_x.as_ptr();
    let py_p = ms.tracks.pos_y.as_ptr();

    // SAFETY: All pointers come from disjoint `MicroSwarmState.tracks` arrays asserted `>= MAX_MICRO`.
    unsafe {
        let mut ti = 0usize;
        while ti < MAX_MICRO {
            let st = *st_p.add(ti);
            if st != MICRO_STATE_INACTIVE {
                act_cnt += 1;
                if st == MICRO_STATE_CONFIRMED {
                    conf_cnt += 1;
                } else if *coast_p.add(ti) > 0 {
                    coast_cnt += 1;
                } else {
                    tent_cnt += 1;
                }

                if cohort_shock {
                    *ambig_p.add(ti) = 1;
                }

                if st != MICRO_STATE_CONFIRMED
                    || *coast_p.add(ti) > 0
                    || *ambig_p.add(ti) != 0
                    || *reacq_p.add(ti) != 0
                    || cohort_shock
                {
                    *fa_frame_p.add(ti) = FTLE_INVALID_FRAME;
                    *fb_frame_p.add(ti) = FTLE_INVALID_FRAME;
                    *fval_p.add(ti) = FTLE_INVALID_VAL;
                } else {
                    let fa = *fa_frame_p.add(ti);
                    let fb = *fb_frame_p.add(ti);
                    if fa == FTLE_INVALID_FRAME && fb == FTLE_INVALID_FRAME {
                        *fa_x0_p.add(ti) = *px_p.add(ti);
                        *fa_y0_p.add(ti) = *py_p.add(ti);
                        *fa_frame_p.add(ti) = frame_idx;
                    } else if phase20 == 0
                        || (fa != FTLE_INVALID_FRAME
                            && fb != FTLE_INVALID_FRAME
                            && frame_idx.saturating_sub(fa) >= FTLE_BANK_PERIOD
                            && frame_idx.saturating_sub(fb) >= FTLE_BANK_STAGGER)
                    {
                        *fa_x0_p.add(ti) = *px_p.add(ti);
                        *fa_y0_p.add(ti) = *py_p.add(ti);
                        *fa_frame_p.add(ti) = frame_idx;
                    } else if phase20 == FTLE_BANK_STAGGER
                        || (fa != FTLE_INVALID_FRAME
                            && (fb == FTLE_INVALID_FRAME
                                || frame_idx.saturating_sub(fb) >= FTLE_BANK_PERIOD)
                            && frame_idx.saturating_sub(fa) >= FTLE_BANK_STAGGER)
                    {
                        *fb_x0_p.add(ti) = *px_p.add(ti);
                        *fb_y0_p.add(ti) = *py_p.add(ti);
                        *fb_frame_p.add(ti) = frame_idx;
                    }
                }
            }
            ti += 1;
        }
    }
    if cohort_shock {
        ms.ambiguous_count = conf_cnt.max(ms.ambiguous_count);
    }

    ms.active_count = act_cnt;
    ms.confirmed_count = conf_cnt;
    ms.tentative_count = tent_cnt;
    ms.coasting_count = coast_cnt;
    ms.births_count = births;
    if conf_cnt > ms.peak_simultaneous_confirmed {
        ms.peak_simultaneous_confirmed = conf_cnt;
    }
    ms.reacq_events = reacq_cnt;
    ms.mean_ic_lk_res = if res_cnt > 0 {
        res_sum / (res_cnt as f32)
    } else {
        0.0
    };
}

/// Packs `MicroTrackSoA` into the flat `1536 * 8` C-ABI export buffer and fills `swarm_telem[0..16]`.
/// Call again after `continuum_physics` updates `body_id`, `ftle_val`, and `swarm_telem[16..38]`.
#[inline(never)]
pub fn pack_export_micro(ms: &mut MicroSwarmState) {
    let mut act_cnt = 0u32;
    let mut conf_cnt = 0u32;
    let mut tent_cnt = 0u32;
    let mut coast_cnt = 0u32;

    assert!(ms.export_micro.len() >= MAX_MICRO * MICRO_EXPORT_STRIDE);
    assert!(ms.tracks.state.len() >= MAX_MICRO && ms.tracks.track_id.len() >= MAX_MICRO);
    assert!(ms.tracks.pos_x.len() >= MAX_MICRO && ms.tracks.pos_y.len() >= MAX_MICRO);
    assert!(ms.tracks.vel_x.len() >= MAX_MICRO && ms.tracks.vel_y.len() >= MAX_MICRO);
    assert!(ms.tracks.age.len() >= MAX_MICRO && ms.tracks.coasting.len() >= MAX_MICRO);
    assert!(ms.tracks.ambiguous_flag.len() >= MAX_MICRO && ms.tracks.body_id.len() >= MAX_MICRO);
    assert!(
        ms.tracks.ftle_a_frame.len() >= MAX_MICRO
            && ms.tracks.ftle_b_frame.len() >= MAX_MICRO
            && ms.tracks.ftle_val.len() >= MAX_MICRO
    );

    #[cfg(target_arch = "wasm32")]
    let vzero = f32x4_splat(0.0);
    let exp_ptr = ms.export_micro.as_mut_ptr();
    let state_p = ms.tracks.state.as_mut_ptr();
    let tid_p = ms.tracks.track_id.as_mut_ptr();
    let px_p = ms.tracks.pos_x.as_ptr();
    let py_p = ms.tracks.pos_y.as_ptr();
    let vx_p = ms.tracks.vel_x.as_ptr();
    let vy_p = ms.tracks.vel_y.as_ptr();
    let age_p = ms.tracks.age.as_mut_ptr();
    let coast_p = ms.tracks.coasting.as_mut_ptr();
    let ambig_p = ms.tracks.ambiguous_flag.as_mut_ptr();
    let body_p = ms.tracks.body_id.as_mut_ptr();
    let ftle_a_p = ms.tracks.ftle_a_frame.as_mut_ptr();
    let ftle_b_p = ms.tracks.ftle_b_frame.as_mut_ptr();
    let ftle_val_p = ms.tracks.ftle_val.as_mut_ptr();

    // SAFETY: All pointers come from disjoint `MicroSwarmState` fields asserted `>= MAX_MICRO`.
    unsafe {
        let mut i = 0usize;
        while i < MAX_MICRO {
            let base = i * MICRO_EXPORT_STRIDE;
            let mut st = *state_p.add(i);
            if st != MICRO_STATE_INACTIVE {
                let px = *px_p.add(i);
                let py = *py_p.add(i);
                let vx = *vx_p.add(i);
                let vy = *vy_p.add(i);
                if !px.is_finite() || !py.is_finite() || !vx.is_finite() || !vy.is_finite() {
                    *state_p.add(i) = MICRO_STATE_INACTIVE;
                    *tid_p.add(i) = 0;
                    *age_p.add(i) = 0;
                    *coast_p.add(i) = 0;
                    *ambig_p.add(i) = 0;
                    *body_p.add(i) = 0;
                    *ftle_a_p.add(i) = FTLE_INVALID_FRAME;
                    *ftle_b_p.add(i) = FTLE_INVALID_FRAME;
                    *ftle_val_p.add(i) = 0.0;
                    st = MICRO_STATE_INACTIVE;
                }
            }
            if st == MICRO_STATE_INACTIVE {
                #[cfg(target_arch = "wasm32")]
                {
                    let ep = exp_ptr.add(base) as *mut v128;
                    core::ptr::write_unaligned(ep, vzero);
                    core::ptr::write_unaligned(ep.add(1), vzero);
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    *exp_ptr.add(base) = 0.0;
                    *exp_ptr.add(base + 1) = 0.0;
                    *exp_ptr.add(base + 2) = 0.0;
                    *exp_ptr.add(base + 3) = 0.0;
                    *exp_ptr.add(base + 4) = 0.0;
                    *exp_ptr.add(base + 5) = 0.0;
                    *exp_ptr.add(base + 6) = 0.0;
                    *exp_ptr.add(base + 7) = 0.0;
                }
            } else {
                act_cnt += 1;
                if st == MICRO_STATE_CONFIRMED {
                    conf_cnt += 1;
                } else if *coast_p.add(i) > 0 {
                    coast_cnt += 1;
                } else {
                    tent_cnt += 1;
                }
                let fv = *ftle_val_p.add(i);
                let ftle = if fv.is_finite() {
                    fv
                } else {
                    *ftle_val_p.add(i) = FTLE_INVALID_VAL;
                    FTLE_INVALID_VAL
                };
                #[cfg(target_arch = "wasm32")]
                {
                    let ep = exp_ptr.add(base) as *mut v128;
                    core::ptr::write_unaligned(
                        ep,
                        f32x4(
                            *tid_p.add(i) as f32,
                            *px_p.add(i),
                            *py_p.add(i),
                            *vx_p.add(i),
                        ),
                    );
                    core::ptr::write_unaligned(
                        ep.add(1),
                        f32x4(
                            *vy_p.add(i),
                            st as f32,
                            *body_p.add(i) as f32,
                            ftle,
                        ),
                    );
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    *exp_ptr.add(base) = *tid_p.add(i) as f32;
                    *exp_ptr.add(base + 1) = *px_p.add(i);
                    *exp_ptr.add(base + 2) = *py_p.add(i);
                    *exp_ptr.add(base + 3) = *vx_p.add(i);
                    *exp_ptr.add(base + 4) = *vy_p.add(i);
                    *exp_ptr.add(base + 5) = st as f32;
                    *exp_ptr.add(base + 6) = *body_p.add(i) as f32;
                    *exp_ptr.add(base + 7) = ftle;
                }
            }
            i += 1;
        }
    }

    ms.active_count = act_cnt;
    ms.confirmed_count = conf_cnt;
    ms.tentative_count = tent_cnt;
    ms.coasting_count = coast_cnt;
    if conf_cnt > ms.peak_simultaneous_confirmed {
        ms.peak_simultaneous_confirmed = conf_cnt;
    }

    ms.swarm_telem[0] = ms.swarm_mode as f32;
    ms.swarm_telem[1] = ms.swarm_active as f32;
    ms.swarm_telem[2] = ms.num_peaks as f32;
    ms.swarm_telem[3] = ms.active_count as f32;
    ms.swarm_telem[4] = ms.confirmed_count as f32;
    ms.swarm_telem[5] = ms.peak_simultaneous_confirmed as f32;
    ms.swarm_telem[6] = ms.total_births_lifetime as f32;
    ms.swarm_telem[7] = ms.coasting_count as f32;
    ms.swarm_telem[8] = ms.auction_steps as f32;
    ms.swarm_telem[9] = ms.reacq_events as f32;
    ms.swarm_telem[10] = if ms.mean_ic_lk_res.is_finite() {
        ms.mean_ic_lk_res
    } else {
        0.0
    };
    ms.swarm_telem[11] = ms.inspected_micro_tid as f32;
    ms.swarm_telem[12] = ms.inspected_body_id as f32;
    ms.swarm_telem[13] = ms.births_count as f32;
    ms.swarm_telem[14] = ms.tentative_count as f32;
    ms.swarm_telem[15] = ms.ambiguous_count as f32;
}

/// Click-to-inspect helper: rejects non-finite / out-of-bounds `(x, y)` WITHOUT mutating
/// existing inspection state; otherwise finds nearest confirmed micro-track strictly `< 24 * q` px
/// of `(x, y)` and records `(inspected_micro_tid, inspected_body_id)`. Returns `track_id` or `0`.
#[inline(never)]
pub fn inspect_nearest_micro_track(ms: &mut MicroSwarmState, x: f32, y: f32, w: usize, h: usize) -> u32 {
    if w < 16
        || h < 16
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= (w as f32)
        || y >= (h as f32)
    {
        return 0;
    }
    if ms.swarm_active == 0 || ms.active_count == 0 {
        ms.inspected_micro_tid = 0;
        ms.inspected_body_id = 0;
        ms.swarm_telem[11] = 0.0;
        ms.swarm_telem[12] = 0.0;
        return 0;
    }
    let q = (w as f32) * (1.0 / 640.0);
    let r_lim = 24.0 * q;
    let mut best_d2 = r_lim * r_lim;
    let mut best_ti: Option<usize> = None;

    let inv_cw = (MICRO_CELL_W as f32) / (w as f32);
    let inv_ch = (MICRO_CELL_H as f32) / (h as f32);
    let cx0 = cell_coord(x, inv_cw, MICRO_CELL_W - 1) as isize;
    let cy0 = cell_coord(y, inv_ch, MICRO_CELL_H - 1) as isize;
    let x_lo = (cx0 - 2).max(0) as usize;
    let x_hi = (cx0 + 2).min((MICRO_CELL_W - 1) as isize) as usize;
    let y_lo = (cy0 - 2).max(0) as usize;
    let y_hi = (cy0 + 2).min((MICRO_CELL_H - 1) as isize) as usize;

    let mut cy = y_lo;
    while cy <= y_hi {
        let mut cx = x_lo;
        while cx <= x_hi {
            let mut idx = ms.track_cell_head[cy * MICRO_CELL_W + cx];
            while idx >= 0 {
                let ti = idx as usize;
                idx = ms.track_next[ti];
                if ms.tracks.state[ti] == MICRO_STATE_CONFIRMED {
                    let dx = ms.tracks.pos_x[ti] - x;
                    let dy = ms.tracks.pos_y[ti] - y;
                    let d2 = dx * dx + dy * dy;
                    if d2.is_finite() && d2 < best_d2 {
                        best_d2 = d2;
                        best_ti = Some(ti);
                    }
                }
            }
            cx += 1;
        }
        cy += 1;
    }

    if let Some(ti) = best_ti {
        ms.inspected_micro_tid = ms.tracks.track_id[ti];
        ms.inspected_body_id = ms.tracks.body_id[ti] as u32;
        ms.swarm_telem[11] = ms.inspected_micro_tid as f32;
        ms.swarm_telem[12] = ms.inspected_body_id as f32;
        ms.inspected_micro_tid
    } else {
        ms.inspected_micro_tid = 0;
        ms.inspected_body_id = 0;
        ms.swarm_telem[11] = 0.0;
        ms.swarm_telem[12] = 0.0;
        0
    }
}

pub fn inspect_micro_track_by_id(
    ms: &mut MicroSwarmState,
    track_id: u32,
    w: usize,
    h: usize,
) -> Option<(f32, f32, f32, f32)> {
    if track_id == 0 || w < 16 || h < 16 || ms.swarm_active == 0 || ms.active_count == 0 {
        ms.inspected_micro_tid = 0;
        ms.inspected_body_id = 0;
        ms.swarm_telem[11] = 0.0;
        ms.swarm_telem[12] = 0.0;
        return None;
    }
    let mut ti = 0usize;
    while ti < MAX_MICRO {
        if ms.tracks.state[ti] == MICRO_STATE_CONFIRMED && ms.tracks.track_id[ti] == track_id {
            let mx = ms.tracks.pos_x[ti];
            let my = ms.tracks.pos_y[ti];
            let mvx = ms.tracks.vel_x[ti];
            let mvy = ms.tracks.vel_y[ti];
            if mx.is_finite()
                && my.is_finite()
                && mx >= 0.0
                && my >= 0.0
                && mx < (w as f32)
                && my < (h as f32)
            {
                ms.inspected_micro_tid = track_id;
                ms.inspected_body_id = ms.tracks.body_id[ti] as u32;
                ms.swarm_telem[11] = track_id as f32;
                ms.swarm_telem[12] = ms.inspected_body_id as f32;
                return Some((mx, my, mvx, mvy));
            }
            break;
        }
        ti += 1;
    }
    ms.inspected_micro_tid = 0;
    ms.inspected_body_id = 0;
    ms.swarm_telem[11] = 0.0;
    ms.swarm_telem[12] = 0.0;
    None
}

/// Top-level Wave 1 `soa_tracks` step. Returns `true` when CHRONOS-1000 swarm tracking ran
/// for this frame (so caller runs `continuum_physics` afterwards).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn soa_step(
    ms: &mut MicroSwarmState,
    luma: &[f32],
    bg_luma: &[f32],
    eca_res: &[f32],
    rgba: &[u8],
    cc_energy: &[f32],
    w: usize,
    h: usize,
    safe_dt: f32,
    frame_idx: u32,
    pan_dx_f: f32,
    pan_dy_f: f32,
    scene_cut: bool,
) -> bool {
    if ms.swarm_mode == 0 {
        if ms.swarm_active != 0
            || ms.active_count != 0
            || ms.probe_streak != 0
            || ms.cut_cooldown != 0
            || ms.probe_backoff != 0
            || ms.probe_fail_streak != 0
        {
            ms.reset_active_keep_mode();
            ms.cut_cooldown = 0;
            ms.probe_backoff = 0;
            ms.probe_fail_streak = 0;
        } else {
            ms.swarm_telem[..16].fill(0.0);
        }
        return false;
    }

    let need_px = w.saturating_mul(h);
    let valid_inputs = w >= 16
        && h >= 16
        && w <= SCORE_RING_STRIDE
        && h <= 360
        && h * ((w + 63) >> 6) <= NMS_BITMAP_WORDS
        && safe_dt.is_finite()
        && safe_dt > 0.0
        && pan_dx_f.is_finite()
        && pan_dy_f.is_finite()
        && luma.len() >= need_px
        && bg_luma.len() >= need_px
        && eca_res.len() >= need_px
        && rgba.len() >= need_px.saturating_mul(4);
    if !valid_inputs {
        ms.export_micro.fill(0.0);
        ms.tracks.body_id.fill(0);
        ms.invalidate_continuity();
        ms.swarm_telem[1..16].fill(0.0);
        ms.swarm_telem[0] = ms.swarm_mode as f32;
        return false;
    }

    let q = (w as f32) * (1.0 / 640.0);
    let pan_mag = wasm_sqrtf(pan_dx_f * pan_dx_f + pan_dy_f * pan_dy_f);

    if ms.swarm_mode == 1 {
        if scene_cut {
            ms.reset_active_keep_mode();
            ms.cut_cooldown = 3;
            ms.probe_backoff = 0;
            ms.probe_fail_streak = 0;
            ms.swarm_telem[0] = 1.0;
            return false;
        }
        if ms.swarm_active == 0 {
            let ensure_auto_zero = |ms: &mut MicroSwarmState| {
                if ms.swarm_telem[0] != 1.0
                    || ms.swarm_telem[1] != 0.0
                    || ms.swarm_telem[2] != 0.0
                    || ms.swarm_telem[3] != 0.0
                {
                    ms.swarm_telem[1..16].fill(0.0);
                    ms.swarm_telem[0] = 1.0;
                }
            };
            if ms.cut_cooldown > 0 {
                ms.cut_cooldown -= 1;
                ms.probe_streak = 0;
                ms.probe_peak_count = 0;
                ensure_auto_zero(ms);
                return false;
            }
            if pan_mag > 2.0 * q {
                ms.probe_streak = 0;
                ms.probe_peak_count = 0;
                ms.probe_backoff = 0;
                ms.probe_fail_streak = 0;
                ensure_auto_zero(ms);
                return false;
            }
            let occ_cc = count_occupied_cc_cells_simd128(cc_energy);
            if occ_cc < AUTO_OCC_CELLS_PROBE {
                ms.probe_streak = 0;
                ms.probe_peak_count = 0;
                ms.probe_backoff = 0;
                ms.probe_fail_streak = 0;
                ensure_auto_zero(ms);
                return false;
            }
            if ms.probe_backoff > 0 {
                ms.probe_backoff -= 1;
                ensure_auto_zero(ms);
                return false;
            }
            if !auto_probe_corner_pregate(luma, bg_luma, eca_res, w, h, pan_mag) {
                let shift = ms.probe_fail_streak.min(3);
                ms.probe_backoff = (4u8 << shift).min(AUTO_PROBE_MAX_BACKOFF);
                ms.probe_fail_streak = ms.probe_fail_streak.saturating_add(1);
                ms.probe_streak = 0;
                ms.probe_peak_count = 0;
                ensure_auto_zero(ms);
                return false;
            }
            extract_micro_peaks_simd128(ms, luma, bg_luma, eca_res, rgba, w, h, pan_mag);
            if ms.num_peaks < AUTO_ENTRY_PEAKS {
                let shift = ms.probe_fail_streak.min(3);
                ms.probe_backoff = (4u8 << shift).min(AUTO_PROBE_MAX_BACKOFF);
                ms.probe_fail_streak = ms.probe_fail_streak.saturating_add(1);
                ms.probe_streak = 0;
                ms.probe_peak_count = 0;
                ensure_auto_zero(ms);
                return false;
            }
            if !probe_temporal_correspondence(ms, w, h, pan_dx_f, pan_dy_f) {
                if ms.probe_streak == 0 {
                    ms.probe_streak = 1;
                } else if ms.probe_fail_streak == 0 {
                    // Single correspondence glitch or transition: keep current frame snapshot for 1 immediate retry
                    ms.probe_fail_streak = 1;
                    ms.probe_streak = 1;
                } else {
                    let shift = ms.probe_fail_streak.saturating_sub(1).min(3);
                    ms.probe_backoff = (4u8 << shift).min(AUTO_PROBE_MAX_BACKOFF);
                    ms.probe_fail_streak = ms.probe_fail_streak.saturating_add(1);
                    ms.probe_streak = 0;
                    ms.probe_peak_count = 0;
                }
                ensure_auto_zero(ms);
                return false;
            }
            // Correspondence passed: enter swarm tracking cleanly
            ms.tracks.state.fill(MICRO_STATE_INACTIVE);
            ms.tracks.track_id.fill(0);
            ms.swarm_active = 1;
            ms.low_density_streak = 0;
            ms.probe_streak = 0;
            ms.probe_backoff = 0;
            ms.probe_fail_streak = 0;
            ms.probe_peak_count = 0;
        } else {
            extract_micro_peaks_simd128(ms, luma, bg_luma, eca_res, rgba, w, h, pan_mag);
            build_peak_morton_cell_hash(ms, w, h);
            imm_predict_micro_simd128(ms, w, h, safe_dt, pan_dx_f, pan_dy_f, false);
            gather_csr_candidates(ms, w, h, safe_dt);
            solve_epsilon_auction(ms);
            update_lifecycle_and_seed(ms, luma, w, h, safe_dt, frame_idx);
            pack_export_micro(ms);

            let observed_cnt = ms.confirmed_count.saturating_add(ms.tentative_count);
            if ms.num_peaks >= AUTO_HOLD_PEAKS && observed_cnt >= (AUTO_HOLD_PEAKS / 2) as u32 {
                ms.low_density_streak = 0;
            } else {
                ms.low_density_streak = ms.low_density_streak.saturating_add(1);
                if ms.low_density_streak >= 8 || ms.active_count == 0 {
                    ms.reset_active_keep_mode();
                    ms.swarm_telem[0] = 1.0;
                    return false;
                }
            }
            return true;
        }
    } else {
        if scene_cut {
            ms.reset_active_keep_mode();
        }
        ms.swarm_active = 1;
        ms.low_density_streak = 0;
        ms.probe_streak = 0;
        extract_micro_peaks_simd128(ms, luma, bg_luma, eca_res, rgba, w, h, pan_mag);
    }

    build_peak_morton_cell_hash(ms, w, h);
    imm_predict_micro_simd128(ms, w, h, safe_dt, pan_dx_f, pan_dy_f, false);
    gather_csr_candidates(ms, w, h, safe_dt);
    solve_epsilon_auction(ms);
    update_lifecycle_and_seed(ms, luma, w, h, safe_dt, frame_idx);
    pack_export_micro(ms);
    true
}
