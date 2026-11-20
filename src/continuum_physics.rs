//! Pillar 3 Continuum Physics Engine for CHRONOS-SPECTRAL (`CHRONOS-1000`).
//!
//! Operates zero-allocation in `#![no_std]` Arena memory on `MicroSwarmState` + `ContinuumState`
//! without modifying track assignment internals:
//! 1. Sparse `k`-NN Interaction Graph (`k = 6`, `32x18` cell-accelerated with exact ring pruning).
//! 2. Persistent Rigid-Motion Clustering & Exact `SO(2)` / `SE(2)` Procrustes Solver (`1 << 2`).
//! 3. Adaptive `40x22` Eulerian Velocity Gradient & Robust Staggered FTLE Solver (`1 << 0`, `1 << 1`).
//! 4. Canonical-Axis PLL Modal Virtual Accelerometers & Swarm Kuramoto / CoM (`1 << 3`, `1 << 4`).
//!
//! Diagnostic Transparency: never clamps or fabricates metrics on degenerate inputs; always emits
//! raw finite values (`0.0` when undefined) and clears the matching `diagnostic_validity_mask` bit.

use crate::soa_tracks::{
    inv_2x2_quad_f64, update_hero_candidates_hysteresis, MicroSwarmState, FTLE_INVALID_FRAME,
    MAX_MICRO, METROLOGY_OFF, MICRO_CELL_H, MICRO_CELL_W, MICRO_EXPORT_STRIDE,
    MICRO_STATE_CONFIRMED, MICRO_STATE_INACTIVE, QUAL_BENCH_RES_USABLE,
    QUAL_BENCH_TRANSLATION_ONLY, QUAL_COV_USABLE, QUAL_NEIGHBOR_MASK, QUAL_NEIGHBOR_SHIFT,
    SWARM_TELEM_LEN,
};
use crate::{
    fast_atan2f, fast_cosf, fast_expf, fast_lnf, fast_sinf, wasm_floorf, wasm_sqrt_f64,
    wasm_sqrtf, wrap_pi,
};

pub const MICRO_KNN_K: usize = 6;
pub const MAX_RIGID_BODIES: usize = 32;
pub const FIELD_GRID_W: usize = 40;
pub const FIELD_GRID_H: usize = 22;
pub const FIELD_CELLS: usize = FIELD_GRID_W * FIELD_GRID_H; // 880
pub const FIELD_EXPORT_STRIDE: usize = 6;
pub const BODY_EXPORT_STRIDE: usize = 8;

pub const VALIDITY_BIT_GRAD: u32 = 1 << 0;
pub const VALIDITY_BIT_FTLE: u32 = 1 << 1;
pub const VALIDITY_BIT_RIGID: u32 = 1 << 2;
pub const VALIDITY_BIT_MODAL: u32 = 1 << 3;
pub const VALIDITY_BIT_KURAMOTO: u32 = 1 << 4;

pub const CELL_VALID_GRAD: u8 = 1 << 0;
pub const CELL_VALID_FTLE: u8 = 1 << 1;
pub const CELL_VALID_MODAL: u8 = 1 << 2;

const FTLE_TIME_RING: usize = 32;
const FIELD_CAND_CAP: usize = 24;
const CAND_BUF_CAP: usize = 96;
const MIN_RIGID_MEMBERS: usize = 6;
const MAX_RIGID_CORES: usize = MAX_MICRO / MIN_RIGID_MEMBERS; // 256
const RIGID_PERSIST_STREAK: u8 = 8;
const KNN_STRAIN_THRESH: f32 = 0.035;
const KNN_STRAIN_SCALE: f32 = 32_767.5; // maps [0.0, 2.0] into u16
const INV_KNN_STRAIN_SCALE: f32 = 1.0 / KNN_STRAIN_SCALE;
const KNN_STRAIN_THRESH_Q15: u16 = 1147; // smallest u16 with (q15 as f32)/32767.5 >= 0.035
const TWO_PI: f32 = 6.283185307179586;
const GRAD_FLAG_PASS2: u8 = 1 << 0;
const GRAD_FLAG_INCOHERENT_CORE: u8 = 1 << 1;

/// Compact AoS per-slot sparse k-NN interaction history (`80 B`, 0 padding).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct KnnSlot {
    pub tid: [u32; MICRO_KNN_K],
    pub ref_dist: [f32; MICRO_KNN_K],
    pub idx: [i16; MICRO_KNN_K],
    pub strain_q15: [u16; MICRO_KNN_K],
    pub rigid_streak: [u8; MICRO_KNN_K],
    pub count: u8,
    pub persist_mask: u8,
}

impl KnnSlot {
    pub const ZEROED: Self = Self {
        tid: [0; MICRO_KNN_K],
        ref_dist: [0.0; MICRO_KNN_K],
        idx: [0; MICRO_KNN_K],
        strain_q15: [0; MICRO_KNN_K],
        rigid_streak: [0; MICRO_KNN_K],
        count: 0,
        persist_mask: 0,
    };

    pub const EMPTY: Self = Self {
        tid: [0; MICRO_KNN_K],
        ref_dist: [0.0; MICRO_KNN_K],
        idx: [-1; MICRO_KNN_K],
        strain_q15: [0; MICRO_KNN_K],
        rigid_streak: [0; MICRO_KNN_K],
        count: 0,
        persist_mask: 0,
    };
}

/// Persistent Arena-resident buffers and scratch space for Pillar 3 continuum physics.
/// All arrays `>= 64 B` live inside this struct so call-stack usage stays tiny (`<= 14,336 B`).
#[repr(C, align(16))]
pub struct ContinuumState {
    // C-ABI flat export buffers
    pub export_field: [f32; FIELD_CELLS * FIELD_EXPORT_STRIDE],
    pub export_modal_field: [f32; FIELD_CELLS],
    pub export_bodies: [f32; MAX_RIGID_BODIES * BODY_EXPORT_STRIDE],
    pub cell_valid_mask: [u8; FIELD_CELLS],

    // Sparse k-NN interaction graph (per track slot, k = 6)
    pub knn: [KnnSlot; MAX_MICRO],

    // Per-slot lifecycle & rotation accumulation tracking for FTLE, SE(2), & k-NN resets
    pub prev_slot_tid: [u32; MAX_MICRO],
    pub slot_clean_since_frame: [u32; MAX_MICRO],
    pub slot_dtheta_a: [f32; MAX_MICRO],
    pub slot_dtheta_b: [f32; MAX_MICRO],
    pub prev_slot_body_rank: [i8; MAX_MICRO],
    pub slot_dtheta_valid: [u8; MAX_MICRO],

    // 40x22 contiguous CSR spatial hash (reused across Grad-V, FTLE anchor banks, and Modal field)
    pub field_grid_x: [f32; MAX_MICRO],
    pub field_grid_y: [f32; MAX_MICRO],
    pub field_grid_aux: [[f32; 2]; MAX_MICRO],
    pub field_cell_start: [u16; FIELD_CELLS + 1],
    pub field_grid_ti: [u16; MAX_MICRO],
    pub field_grid_src_ti: [u16; MAX_MICRO],

    // Union-Find (`< 0` stores `-size`, `>= 0` stores parent) & component workspace for SE(2)
    pub uf_parent: [i16; MAX_MICRO],
    pub comp_head: [i16; MAX_MICRO],
    pub comp_next: [i16; MAX_MICRO],
    pub member_mask: [u8; MAX_MICRO],
    pub cand_roots: [u16; MAX_RIGID_CORES],
    pub rigid_seeds: [u16; MAX_MICRO],
    pub rigid_seed_n: u16,
    pub clean_slots: [u16; MAX_MICRO],
    pub conf_slots: [u16; MAX_MICRO],
    pub clean_slots_len: u16,
    pub conf_slots_len: u16,
    pub active_max_slot: u16,
    pub prev_active_max_slot: u16,
    pub lists_frame_idx: u32,

    // Top-K rigid body selection & ID-based cumulative rotation scratch
    pub body_root_rank: [u16; MAX_RIGID_BODIES],
    pub body_cnt_rank: [u16; MAX_RIGID_BODIES],
    pub body_cx_rank: [f32; MAX_RIGID_BODIES],
    pub body_cy_rank: [f32; MAX_RIGID_BODIES],
    pub body_vx_rank: [f32; MAX_RIGID_BODIES],
    pub body_vy_rank: [f32; MAX_RIGID_BODIES],
    pub body_omega_rank: [f32; MAX_RIGID_BODIES],
    pub body_rmse_rank: [f32; MAX_RIGID_BODIES],
    pub body_dtheta1f_rank: [f32; MAX_RIGID_BODIES],
    pub body_cos1f_rank: [f32; MAX_RIGID_BODIES],
    pub body_sin1f_rank: [f32; MAX_RIGID_BODIES],
    pub prev_body_cum_theta: [f32; MAX_RIGID_BODIES],
    pub body_cum_theta_tmp: [f32; MAX_RIGID_BODIES],
    pub body_overlap_cnt: [u16; MAX_RIGID_BODIES],
    pub body_rank_len: usize,
    pub top_body_cum_theta: f32,
    pub prev_body_cnt: u8,

    // Field cell candidate & robust FTLE IRLS scratch (max 24 tracks per cell)
    pub cell_cands_ti: [u16; FIELD_CAND_CAP],
    pub cell_cands_d2: [f32; FIELD_CAND_CAP],
    pub ftle_cands_ti: [u16; FIELD_CAND_CAP],
    pub ftle_cands_d2: [f32; FIELD_CAND_CAP],
    pub ftle_cands_bank: [u8; FIELD_CAND_CAP],
    pub ftle_cands_w: [f32; FIELD_CAND_CAP],
    pub ftle_cands_res: [f32; FIELD_CAND_CAP],
    pub ftle_sorted_res: [f32; FIELD_CAND_CAP],
    pub cand_dx: [f32; FIELD_CAND_CAP],
    pub cand_dy: [f32; FIELD_CAND_CAP],
    pub cand_du: [f32; FIELD_CAND_CAP],
    pub cand_dv: [f32; FIELD_CAND_CAP],
    pub cand_buf_ti: [u16; CAND_BUF_CAP],
    pub cand_buf_d2: [f32; CAND_BUF_CAP],
    pub cand_buf_dx: [f32; CAND_BUF_CAP],
    pub cand_buf_dy: [f32; CAND_BUF_CAP],

    // Modal virtual accelerometer per-slot state (no heap, zero 2nd-diff accel)
    pub modal_axis_xx: [f32; MAX_MICRO],
    pub modal_axis_xy: [f32; MAX_MICRO],
    pub modal_axis_yy: [f32; MAX_MICRO],
    pub modal_u_dc: [f32; MAX_MICRO],
    pub modal_err_var: [f32; MAX_MICRO],
    pub modal_cross_age_s: [f32; MAX_MICRO],
    pub modal_ema_period_s: [f32; MAX_MICRO],
    pub modal_tent_period_01ms: [u16; MAX_MICRO],
    pub modal_prev_ax_q13: [i16; MAX_MICRO],
    pub modal_zc_state: [i8; MAX_MICRO],
    pub modal_cycles: [u8; MAX_MICRO],

    // Per-frame exact timestamp ring for multi-frame T_sec
    pub sim_time_s: f64,
    pub time_ring_s: [f64; FTLE_TIME_RING],
    pub time_ring_frame: [u32; FTLE_TIME_RING],
    pub last_frame_idx: u32,
    pub clean_epoch_frame: u32,
    pub initialized: u8,

    // Scalar diagnostics exposed in struct & swarm_telem[16..=41]
    pub diagnostic_validity_mask: u32,
    pub valid_field_cells: u32,
    pub max_abs_vorticity_s1: f32,
    pub rms_vorticity_s1: f32,
    pub rms_divergence_s1: f32,
    pub max_ftle_s1: f32,
    pub ftle_cells: u32,
    pub rigid_body_count: u32,
    pub top_rigid_omega_rad_s: f32,
    pub top_rigid_members: u32,
    pub top_rigid_se2_rmse_px: f32,
    pub top_rigid_theta_rad: f32,
    pub modal_dom_freq_hz: f32,
    pub modal_phase_coherence: f32,
    pub modal_nematic_r2: f32,
    pub modal_valid_sensors: u32,
    pub kuramoto_r: f32,
    pub kuramoto_psi: f32,
    pub com_x: f32,
    pub com_y: f32,
    pub com_vx: f32,
    pub com_vy: f32,
    pub swarm_velocity_dispersion: f32,
    pub mean_strain: f32,
    pub knn_edge_count: u32,
    pub ftle_mature_tracks: u32,

    // Cached 15 Hz Eulerian gradient & FTLE summary for dense-swarm staggered split
    pub cached_grad_ftle_frame: u32,
    pub cached_ftle_frame: u32,
    pub cached_grad_ftle_mask: u32,
    pub cached_valid_field_cells: u32,
    pub cached_max_abs_vorticity_s1: f32,
    pub cached_rms_vorticity_s1: f32,
    pub cached_rms_divergence_s1: f32,
    pub cached_max_ftle_s1: f32,
    pub cached_ftle_cells: u32,
    pub cached_knn_frame: u32,
    pub cached_modal_frame: u32,
    pub cached_modal_cells: u32,
    pub export_field_dirty: u8,
    pub export_modal_dirty: u8,
}

// Keep ContinuumState <= 284 KiB so 320-page Arena headroom (>2 MiB) stays comfortable.
const _: () = assert!(core::mem::size_of::<ContinuumState>() <= 290_816);

#[inline(always)]
fn pack_strain_q15(strain: f32) -> u16 {
    ((0.0f32.max(strain).min(2.0)) * KNN_STRAIN_SCALE + 0.5) as u16
}

#[inline(always)]
fn unpack_strain_q15(q15: u16) -> f32 {
    (q15 as f32) * INV_KNN_STRAIN_SCALE
}

#[inline(always)]
fn min_continuum_age(ms: &MicroSwarmState) -> u16 {
    if ms.num_peaks >= 1550 && ms.confirmed_count >= 1015 {
        6
    } else {
        1
    }
}

#[inline(always)]
pub(crate) fn is_clean_confirmed(ms: &MicroSwarmState, i: usize) -> bool {
    ms.tracks.state[i] == MICRO_STATE_CONFIRMED
        && ms.tracks.coasting[i] == 0
        && ms.tracks.reacq_flag[i] == 0
        && ms.tracks.ambiguous_flag[i] == 0
}

#[inline(always)]
fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline(always)]
fn clamped_grid_coord(pos: f32, scale: f32, max_idx: usize) -> usize {
    let p = pos * scale;
    if p > 0.0 {
        (p as usize).min(max_idx)
    } else {
        0
    }
}

#[inline(always)]
fn sym2x2_eigenvalues(gxx: f32, gxy: f32, gyy: f32) -> (f32, f32) {
    let half_tr = 0.5 * (gxx + gyy);
    let half_diff = 0.5 * (gxx - gyy);
    let disc = wasm_sqrtf((half_diff * half_diff + gxy * gxy).max(0.0));
    (half_tr - disc, half_tr + disc)
}

#[inline(always)]
fn is_well_conditioned_cov2_f32(gxx: f32, gxy: f32, gyy: f32, min_eig_px2: f32) -> bool {
    if !gxx.is_finite() || !gxy.is_finite() || !gyy.is_finite() {
        return false;
    }
    let tr = gxx + gyy;
    if tr <= 1e-6 || tr < 2.0 * min_eig_px2 {
        return false;
    }
    let det = gxx * gyy - gxy * gxy;
    if det <= 0.0 || det < (0.05 / 1.1025) * tr * tr {
        return false;
    }
    let am = gxx - min_eig_px2;
    let cm = gyy - min_eig_px2;
    am >= 0.0 && cm >= 0.0 && am * cm >= gxy * gxy
}

impl ContinuumState {
    /// Const all-zero initializer for static `Arena` `.bss` placement (`reset()` runs on first frame).
    pub const fn zeroed() -> Self {
        Self {
            export_field: [0.0; FIELD_CELLS * FIELD_EXPORT_STRIDE],
            export_modal_field: [0.0; FIELD_CELLS],
            export_bodies: [0.0; MAX_RIGID_BODIES * BODY_EXPORT_STRIDE],
            cell_valid_mask: [0; FIELD_CELLS],
            knn: [KnnSlot::ZEROED; MAX_MICRO],
            prev_slot_tid: [0; MAX_MICRO],
            slot_clean_since_frame: [0; MAX_MICRO],
            slot_dtheta_a: [0.0; MAX_MICRO],
            slot_dtheta_b: [0.0; MAX_MICRO],
            prev_slot_body_rank: [0; MAX_MICRO],
            slot_dtheta_valid: [0; MAX_MICRO],
            field_grid_x: [0.0; MAX_MICRO],
            field_grid_y: [0.0; MAX_MICRO],
            field_grid_aux: [[0.0; 2]; MAX_MICRO],
            field_cell_start: [0; FIELD_CELLS + 1],
            field_grid_ti: [0; MAX_MICRO],
            field_grid_src_ti: [0; MAX_MICRO],
            uf_parent: [0; MAX_MICRO],
            comp_head: [0; MAX_MICRO],
            comp_next: [0; MAX_MICRO],
            member_mask: [0; MAX_MICRO],
            cand_roots: [0; MAX_RIGID_CORES],
            rigid_seeds: [0; MAX_MICRO],
            rigid_seed_n: 0,
            clean_slots: [0; MAX_MICRO],
            conf_slots: [0; MAX_MICRO],
            clean_slots_len: 0,
            conf_slots_len: 0,
            active_max_slot: 0,
            prev_active_max_slot: MAX_MICRO as u16,
            lists_frame_idx: u32::MAX,
            body_root_rank: [0; MAX_RIGID_BODIES],
            body_cnt_rank: [0; MAX_RIGID_BODIES],
            body_cx_rank: [0.0; MAX_RIGID_BODIES],
            body_cy_rank: [0.0; MAX_RIGID_BODIES],
            body_vx_rank: [0.0; MAX_RIGID_BODIES],
            body_vy_rank: [0.0; MAX_RIGID_BODIES],
            body_omega_rank: [0.0; MAX_RIGID_BODIES],
            body_rmse_rank: [0.0; MAX_RIGID_BODIES],
            body_dtheta1f_rank: [0.0; MAX_RIGID_BODIES],
            body_cos1f_rank: [1.0; MAX_RIGID_BODIES],
            body_sin1f_rank: [0.0; MAX_RIGID_BODIES],
            prev_body_cum_theta: [0.0; MAX_RIGID_BODIES],
            body_cum_theta_tmp: [0.0; MAX_RIGID_BODIES],
            body_overlap_cnt: [0; MAX_RIGID_BODIES],
            body_rank_len: 0,
            top_body_cum_theta: 0.0,
            prev_body_cnt: 0,
            cell_cands_ti: [0; FIELD_CAND_CAP],
            cell_cands_d2: [0.0; FIELD_CAND_CAP],
            ftle_cands_ti: [0; FIELD_CAND_CAP],
            ftle_cands_d2: [0.0; FIELD_CAND_CAP],
            ftle_cands_bank: [0; FIELD_CAND_CAP],
            ftle_cands_w: [0.0; FIELD_CAND_CAP],
            ftle_cands_res: [0.0; FIELD_CAND_CAP],
            ftle_sorted_res: [0.0; FIELD_CAND_CAP],
            cand_dx: [0.0; FIELD_CAND_CAP],
            cand_dy: [0.0; FIELD_CAND_CAP],
            cand_du: [0.0; FIELD_CAND_CAP],
            cand_dv: [0.0; FIELD_CAND_CAP],
            cand_buf_ti: [0; CAND_BUF_CAP],
            cand_buf_d2: [0.0; CAND_BUF_CAP],
            cand_buf_dx: [0.0; CAND_BUF_CAP],
            cand_buf_dy: [0.0; CAND_BUF_CAP],
            modal_axis_xx: [0.0; MAX_MICRO],
            modal_axis_xy: [0.0; MAX_MICRO],
            modal_axis_yy: [0.0; MAX_MICRO],
            modal_u_dc: [0.0; MAX_MICRO],
            modal_err_var: [0.0; MAX_MICRO],
            modal_cross_age_s: [0.0; MAX_MICRO],
            modal_ema_period_s: [0.0; MAX_MICRO],
            modal_tent_period_01ms: [0; MAX_MICRO],
            modal_prev_ax_q13: [0; MAX_MICRO],
            modal_zc_state: [0; MAX_MICRO],
            modal_cycles: [0; MAX_MICRO],
            sim_time_s: 0.0,
            time_ring_s: [0.0; FTLE_TIME_RING],
            time_ring_frame: [u32::MAX; FTLE_TIME_RING],
            last_frame_idx: u32::MAX,
            clean_epoch_frame: 0,
            initialized: 0,
            diagnostic_validity_mask: 0,
            valid_field_cells: 0,
            max_abs_vorticity_s1: 0.0,
            rms_vorticity_s1: 0.0,
            rms_divergence_s1: 0.0,
            max_ftle_s1: 0.0,
            ftle_cells: 0,
            rigid_body_count: 0,
            top_rigid_omega_rad_s: 0.0,
            top_rigid_members: 0,
            top_rigid_se2_rmse_px: 0.0,
            top_rigid_theta_rad: 0.0,
            modal_dom_freq_hz: 0.0,
            modal_phase_coherence: 0.0,
            modal_nematic_r2: 0.0,
            modal_valid_sensors: 0,
            kuramoto_r: 0.0,
            kuramoto_psi: 0.0,
            com_x: 0.0,
            com_y: 0.0,
            com_vx: 0.0,
            com_vy: 0.0,
            swarm_velocity_dispersion: 0.0,
            mean_strain: 0.0,
            knn_edge_count: 0,
            ftle_mature_tracks: 0,
            cached_grad_ftle_frame: u32::MAX,
            cached_ftle_frame: u32::MAX,
            cached_grad_ftle_mask: 0,
            cached_valid_field_cells: 0,
            cached_max_abs_vorticity_s1: 0.0,
            cached_rms_vorticity_s1: 0.0,
            cached_rms_divergence_s1: 0.0,
            cached_max_ftle_s1: 0.0,
            cached_ftle_cells: 0,
            cached_knn_frame: u32::MAX,
            cached_modal_frame: u32::MAX,
            cached_modal_cells: 0,
            export_field_dirty: 0,
            export_modal_dirty: 0,
        }
    }

    /// Zeroes and re-initializes all ContinuumState buffers in-place without stack allocation.
    #[inline(never)]
    pub fn reset(&mut self) {
        clear_export_inactive(self);

        self.knn.fill(KnnSlot::EMPTY);
        self.prev_slot_tid.fill(0);
        self.slot_clean_since_frame.fill(u32::MAX);
        self.slot_dtheta_a.fill(0.0);
        self.slot_dtheta_b.fill(0.0);
        self.prev_slot_body_rank.fill(-1);
        self.slot_dtheta_valid.fill(0);

        self.field_grid_x.fill(0.0);
        self.field_grid_y.fill(0.0);
        self.field_grid_aux.fill([0.0; 2]);
        self.field_cell_start.fill(0);
        self.field_grid_ti.fill(0);
        self.field_grid_src_ti.fill(0);
        self.uf_parent.fill(-1);
        self.comp_head.fill(-1);
        self.comp_next.fill(-1);
        self.member_mask.fill(0);
        self.cand_roots.fill(0);
        self.rigid_seeds.fill(0);
        self.rigid_seed_n = 0;
        self.clean_slots.fill(0);
        self.conf_slots.fill(0);
        self.clean_slots_len = 0;
        self.conf_slots_len = 0;
        self.active_max_slot = 0;
        self.prev_active_max_slot = MAX_MICRO as u16;
        self.lists_frame_idx = u32::MAX;

        self.body_root_rank.fill(0);
        self.body_cnt_rank.fill(0);
        self.body_cx_rank.fill(0.0);
        self.body_cy_rank.fill(0.0);
        self.body_vx_rank.fill(0.0);
        self.body_vy_rank.fill(0.0);
        self.body_omega_rank.fill(0.0);
        self.body_rmse_rank.fill(0.0);
        self.body_dtheta1f_rank.fill(0.0);
        self.body_cos1f_rank.fill(1.0);
        self.body_sin1f_rank.fill(0.0);
        self.prev_body_cum_theta.fill(0.0);
        self.body_cum_theta_tmp.fill(0.0);
        self.body_overlap_cnt.fill(0);
        self.body_rank_len = 0;
        self.top_body_cum_theta = 0.0;
        self.prev_body_cnt = 0;

        self.cell_cands_ti.fill(0);
        self.cell_cands_d2.fill(0.0);
        self.ftle_cands_ti.fill(0);
        self.ftle_cands_d2.fill(0.0);
        self.ftle_cands_bank.fill(0);
        self.ftle_cands_w.fill(0.0);
        self.ftle_cands_res.fill(0.0);
        self.ftle_sorted_res.fill(0.0);
        self.cand_dx.fill(0.0);
        self.cand_dy.fill(0.0);
        self.cand_du.fill(0.0);
        self.cand_dv.fill(0.0);

        self.modal_axis_xx.fill(0.0);
        self.modal_axis_xy.fill(0.0);
        self.modal_axis_yy.fill(0.0);
        self.modal_u_dc.fill(0.0);
        self.modal_err_var.fill(0.0);
        self.modal_cross_age_s.fill(0.0);
        self.modal_ema_period_s.fill(0.0);
        self.modal_tent_period_01ms.fill(0);
        self.modal_prev_ax_q13.fill(i16::MIN);
        self.modal_zc_state.fill(0);
        self.modal_cycles.fill(0);

        self.sim_time_s = 0.0;
        self.time_ring_s.fill(0.0);
        self.time_ring_frame.fill(u32::MAX);
        self.last_frame_idx = u32::MAX;
        self.clean_epoch_frame = 0;
        self.initialized = 1;
        self.cached_grad_ftle_frame = u32::MAX;
        self.cached_ftle_frame = u32::MAX;
        self.cached_grad_ftle_mask = 0;
        self.cached_valid_field_cells = 0;
        self.cached_max_abs_vorticity_s1 = 0.0;
        self.cached_rms_vorticity_s1 = 0.0;
        self.cached_rms_divergence_s1 = 0.0;
        self.cached_max_ftle_s1 = 0.0;
        self.cached_ftle_cells = 0;
        self.cached_knn_frame = u32::MAX;
        self.cached_modal_frame = u32::MAX;
        self.cached_modal_cells = 0;
        self.export_field_dirty = 0;
        self.export_modal_dirty = 0;

        self.clear_diagnostics();
    }

    #[inline(always)]
    fn clear_diagnostics(&mut self) {
        self.diagnostic_validity_mask = 0;
        self.valid_field_cells = 0;
        self.max_abs_vorticity_s1 = 0.0;
        self.rms_vorticity_s1 = 0.0;
        self.rms_divergence_s1 = 0.0;
        self.max_ftle_s1 = 0.0;
        self.ftle_cells = 0;
        self.rigid_body_count = 0;
        self.top_rigid_omega_rad_s = 0.0;
        self.top_rigid_members = 0;
        self.top_rigid_se2_rmse_px = 0.0;
        self.top_rigid_theta_rad = 0.0;
        self.modal_dom_freq_hz = 0.0;
        self.modal_phase_coherence = 0.0;
        self.modal_nematic_r2 = 0.0;
        self.modal_valid_sensors = 0;
        self.kuramoto_r = 0.0;
        self.kuramoto_psi = 0.0;
        self.com_x = 0.0;
        self.com_y = 0.0;
        self.com_vx = 0.0;
        self.com_vy = 0.0;
        self.swarm_velocity_dispersion = 0.0;
        self.mean_strain = 0.0;
        self.knn_edge_count = 0;
        self.ftle_mature_tracks = 0;
    }

    /// Strict anchor-timestamp lookup in the circular timestamp ring.
    /// Returns `(true, elapsed_sec)` iff the exact `anchor_frame` timestamp is intact.
    #[inline(always)]
    fn elapsed_since_anchor(&self, anchor_frame: u32, curr_frame: u32) -> (bool, f32) {
        if curr_frame <= anchor_frame {
            return (false, 0.0);
        }
        let df = curr_frame - anchor_frame;
        if (df as usize) < FTLE_TIME_RING {
            let slot = (anchor_frame as usize) & (FTLE_TIME_RING - 1);
            if self.time_ring_frame[slot] == anchor_frame {
                let dt = (self.sim_time_s - self.time_ring_s[slot]) as f32;
                if dt.is_finite() && dt > 1e-4 {
                    return (true, dt);
                }
            }
        }
        (false, 0.0)
    }

    #[inline(always)]
    pub fn grad_product_status(&self) -> (bool, bool, u32) {
        let valid = (self.diagnostic_validity_mask & VALIDITY_BIT_GRAD) != 0
            && self.cached_grad_ftle_frame != u32::MAX;
        if !valid {
            return (false, false, u32::MAX);
        }
        let held = self.last_frame_idx != u32::MAX
            && self.cached_grad_ftle_frame != self.last_frame_idx;
        (true, held, self.cached_grad_ftle_frame)
    }

    #[inline(always)]
    pub fn ftle_product_status(&self) -> (bool, bool, u32) {
        let valid = (self.diagnostic_validity_mask & VALIDITY_BIT_FTLE) != 0
            && self.cached_ftle_frame != u32::MAX;
        if !valid {
            return (false, false, u32::MAX);
        }
        let held = self.last_frame_idx != u32::MAX && self.cached_ftle_frame != self.last_frame_idx;
        (true, held, self.cached_ftle_frame)
    }

    #[inline(always)]
    pub fn modal_product_status(&self) -> (bool, bool, u32) {
        let valid = (self.diagnostic_validity_mask & VALIDITY_BIT_MODAL) != 0
            && self.cached_modal_frame != u32::MAX
            && self.cached_modal_cells > 0;
        if !valid {
            return (false, false, u32::MAX);
        }
        let held =
            self.last_frame_idx != u32::MAX && self.cached_modal_frame != self.last_frame_idx;
        (true, held, self.cached_modal_frame)
    }
}

#[inline(always)]
fn uf_find(parent: &mut [i16; MAX_MICRO], mut x: usize) -> usize {
    while parent[x] >= 0 {
        let p = (parent[x] as usize).min(MAX_MICRO - 1);
        if parent[p] >= 0 {
            parent[x] = parent[p];
        }
        x = p;
    }
    x
}

#[inline(always)]
fn uf_union(parent: &mut [i16; MAX_MICRO], a: usize, b: usize) -> i16 {
    let mut ra = uf_find(parent, a);
    let mut rb = uf_find(parent, b);
    if ra == rb {
        return parent[ra].saturating_neg();
    }
    if parent[ra] > parent[rb] {
        // More negative means larger component size.
        core::mem::swap(&mut ra, &mut rb);
    }
    let combined = parent[ra].saturating_add(parent[rb]);
    parent[ra] = combined;
    parent[rb] = ra as i16;
    combined.saturating_neg()
}

#[inline(never)]
fn reset_slot_modal(ms: &mut MicroSwarmState, cs: &mut ContinuumState, i: usize, reseed_eq: bool) {
    cs.modal_axis_xx[i] = 0.0;
    cs.modal_axis_xy[i] = 0.0;
    cs.modal_axis_yy[i] = 0.0;
    cs.modal_u_dc[i] = 0.0;
    cs.modal_err_var[i] = 0.0;
    cs.modal_cross_age_s[i] = 0.0;
    cs.modal_ema_period_s[i] = 0.0;
    cs.modal_tent_period_01ms[i] = 0;
    cs.modal_prev_ax_q13[i] = i16::MIN;
    cs.modal_zc_state[i] = 0;
    cs.modal_cycles[i] = 0;

    ms.tracks.bp_i[i] = 0.0;
    ms.tracks.bp_q[i] = 0.0;
    ms.tracks.bp_prev_y[i] = 0.0;
    ms.tracks.disp_y[i] = 0.0;
    ms.tracks.modal_phase[i] = 0.0;
    ms.tracks.modal_freq_hz[i] = 0.0;
    ms.tracks.modal_amp[i] = 0.0;
    if reseed_eq && ms.tracks.state[i] != MICRO_STATE_INACTIVE {
        ms.tracks.eq_x[i] = ms.tracks.pos_x[i];
        ms.tracks.eq_y[i] = ms.tracks.pos_y[i];
    }
}

#[inline(never)]
fn reset_slot_knn(cs: &mut ContinuumState, i: usize) {
    cs.knn[i] = KnnSlot::EMPTY;
    cs.slot_dtheta_a[i] = 0.0;
    cs.slot_dtheta_b[i] = 0.0;
    cs.prev_slot_body_rank[i] = -1;
    cs.slot_dtheta_valid[i] = 0;
}

#[inline(never)]
fn reset_track_ftle_sentinels(ms: &mut MicroSwarmState) {
    if ms.swarm_active == 0 {
        ms.tracks.ftle_val.fill(0.0);
        return;
    }
    let mut i = 0usize;
    while i < MAX_MICRO {
        ms.tracks.ftle_val[i] = if ms.tracks.state[i] == MICRO_STATE_INACTIVE {
            0.0
        } else {
            -999.0
        };
        i += 1;
    }
}

/// Cross-state continuity invalidation hook for Wave-2 `lib.rs` on scene cuts, gaps, or seeks.
/// Clears SoA continuity/anchors and per-track physics together with `ContinuumState`.
#[inline(never)]
pub fn invalidate_continuum_continuity(ms: &mut MicroSwarmState, cs: &mut ContinuumState) {
    ms.invalidate_continuity();
    ms.tracks.body_id.fill(0);
    reset_track_ftle_sentinels(ms);
    let mut i = 0usize;
    while i < MAX_MICRO {
        if ms.tracks.state[i] != MICRO_STATE_INACTIVE {
            ms.tracks.eq_x[i] = ms.tracks.pos_x[i];
            ms.tracks.eq_y[i] = ms.tracks.pos_y[i];
        }
        i += 1;
    }
    cs.reset();
    cs.initialized = 0;
    if ms.swarm_active != 0 {
        apply_active_export_sentinels(cs);
    }
    sync_micro_physics_exports(ms, cs);
}

#[inline(never)]
fn reset_on_frame_discontinuity(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    is_backward_or_wrap: bool,
) {
    ms.tracks.body_id.fill(0);
    reset_track_ftle_sentinels(ms);
    let mut i = 0usize;
    while i < MAX_MICRO {
        if is_backward_or_wrap || ms.tracks.ftle_a_frame[i] != frame_idx {
            ms.tracks.ftle_a_frame[i] = FTLE_INVALID_FRAME;
        }
        if is_backward_or_wrap || ms.tracks.ftle_b_frame[i] != frame_idx {
            ms.tracks.ftle_b_frame[i] = FTLE_INVALID_FRAME;
        }
        ms.tracks.bp_i[i] = 0.0;
        ms.tracks.bp_q[i] = 0.0;
        ms.tracks.bp_prev_y[i] = 0.0;
        ms.tracks.disp_y[i] = 0.0;
        ms.tracks.modal_phase[i] = 0.0;
        ms.tracks.modal_freq_hz[i] = 0.0;
        ms.tracks.modal_amp[i] = 0.0;
        if ms.tracks.state[i] != MICRO_STATE_INACTIVE {
            ms.tracks.eq_x[i] = ms.tracks.pos_x[i];
            ms.tracks.eq_y[i] = ms.tracks.pos_y[i];
        }
        i += 1;
    }
    cs.reset();
    ms.invalidate_metrology_temporal();
    if ms.swarm_active != 0 {
        apply_active_export_sentinels(cs);
    }
    cs.clean_epoch_frame = frame_idx;
}

const KNN_EMPTY: u64 = ((0x7F80_0000u64) << 16) | 0xFFFFu64;

#[inline(always)]
fn knn_pack(d2: f32, j: i16) -> u64 {
    ((d2.to_bits() as u64) << 16) | ((j as u16) as u64)
}

#[derive(Copy, Clone)]
struct KnnTop6 {
    p0: u64,
    p1: u64,
    p2: u64,
    p3: u64,
    p4: u64,
    p5: u64,
    cnt: u8,
    lim: f32,
}

impl KnnTop6 {
    #[inline(always)]
    fn new(r_max2: f32) -> Self {
        Self {
            p0: KNN_EMPTY,
            p1: KNN_EMPTY,
            p2: KNN_EMPTY,
            p3: KNN_EMPTY,
            p4: KNN_EMPTY,
            p5: KNN_EMPTY,
            cnt: 0,
            lim: r_max2,
        }
    }

    #[inline(always)]
    fn sort6(&mut self) {
        macro_rules! cswap_kv {
            ($a:expr, $b:expr) => {{
                let sw = $a > $b;
                let t = if sw { $a } else { $b };
                $a = if sw { $b } else { $a };
                $b = t;
            }};
        }
        cswap_kv!(self.p1, self.p2);
        cswap_kv!(self.p0, self.p2);
        cswap_kv!(self.p0, self.p1);
        cswap_kv!(self.p4, self.p5);
        cswap_kv!(self.p3, self.p5);
        cswap_kv!(self.p3, self.p4);
        cswap_kv!(self.p0, self.p3);
        cswap_kv!(self.p1, self.p4);
        cswap_kv!(self.p2, self.p5);
        cswap_kv!(self.p2, self.p4);
        cswap_kv!(self.p1, self.p3);
        cswap_kv!(self.p2, self.p3);
        self.lim = f32::from_bits((self.p5 >> 16) as u32);
    }

    #[inline(always)]
    fn push_lt6(&mut self, d2: f32, j: i16) {
        let pk = knn_pack(d2, j);
        let c = self.cnt;
        match c {
            0 => self.p0 = pk,
            1 => self.p1 = pk,
            2 => self.p2 = pk,
            3 => self.p3 = pk,
            4 => self.p4 = pk,
            _ => self.p5 = pk,
        }
        self.cnt = c + 1;
        if self.cnt == 6 {
            self.sort6();
        }
    }

    #[inline(always)]
    fn push_full(&mut self, d2: f32, j: i16) {
        let pk = knn_pack(d2, j);
        let c4 = pk < self.p4;
        self.p5 = if c4 { self.p4 } else { pk };
        let c3 = pk < self.p3;
        self.p4 = if c3 { self.p3 } else if c4 { pk } else { self.p4 };
        let c2 = pk < self.p2;
        self.p3 = if c2 { self.p2 } else if c3 { pk } else { self.p3 };
        let c1 = pk < self.p1;
        self.p2 = if c1 { self.p1 } else if c2 { pk } else { self.p2 };
        let c0 = pk < self.p0;
        self.p1 = if c0 { self.p0 } else if c1 { pk } else { self.p1 };
        self.p0 = if c0 { pk } else { self.p0 };
        self.lim = f32::from_bits((self.p5 >> 16) as u32);
    }
}

#[inline(always)]
fn clamp_pos0(v: f32) -> f32 {
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

#[inline(always)]
fn row_mask_slice(
    cs: &ContinuumState,
    row_occ: &[u32; MICRO_CELL_H],
    ry: usize,
    mask3: u32,
) -> (usize, usize) {
    let rb = row_occ[ry] & mask3;
    if rb == 0 {
        return (0, 0);
    }
    let row_off = ry * MICRO_CELL_W;
    let ca = rb.trailing_zeros() as usize;
    let cb = (31 - rb.leading_zeros()) as usize;
    let e = (cs.field_cell_start[(row_off + cb + 1).min(MICRO_CELL_W * MICRO_CELL_H)] as usize)
        .min(MAX_MICRO);
    let s = (cs.field_cell_start[(row_off + ca).min(MICRO_CELL_W * MICRO_CELL_H)] as usize).min(e);
    (s, e)
}

#[inline(always)]
fn visit_knn_slice_reg(
    cs: &ContinuumState,
    s: usize,
    e: usize,
    i: usize,
    xi: f32,
    yi: f32,
    mut top: KnnTop6,
) -> KnnTop6 {
    let e = e.min(MAX_MICRO);
    let s = s.min(e);
    let n = e - s;
    let xs = &cs.field_grid_x[s..e];
    let ys = &cs.field_grid_y[s..e];
    let tis = &cs.field_grid_ti[s..e];
    let mut p = 0usize;
    while p < n && top.cnt < 6 {
        let dx = xs[p] - xi;
        let dy = ys[p] - yi;
        let d2 = dx * dx + dy * dy;
        if d2 <= top.lim {
            let j = tis[p] as usize;
            if j != i {
                top.push_lt6(d2, j as i16);
            }
        }
        p += 1;
    }
    let rem_s = p.min(n);
    if n - rem_s >= 4 {
        {
            use crate::simd::*;
            let vxi = f32x4_splat(xi);
            let vyi = f32x4_splat(yi);
            let mut vlim = f32x4_splat(top.lim);
            for ((xc, yc), tc) in xs[rem_s..]
                .chunks_exact(4)
                .zip(ys[rem_s..].chunks_exact(4))
                .zip(tis[rem_s..].chunks_exact(4))
            {
                if let (&[x0, x1, x2, x3], &[y0, y1, y2, y3], &[t0, t1, t2, t3]) = (xc, yc, tc) {
                    let vdx = f32x4_sub(f32x4(x0, x1, x2, x3), vxi);
                    let vdy = f32x4_sub(f32x4(y0, y1, y2, y3), vyi);
                    let vd2 = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
                    let cmp = f32x4_lt(vd2, vlim);
                    if v128_any_true(cmp) {
                        let mut mask = i32x4_bitmask(cmp) as u32;
                        while mask != 0 {
                            let lane = mask.trailing_zeros();
                            mask &= mask - 1;
                            let (d_lane, t_lane) = match lane {
                                0 => (f32x4_extract_lane::<0>(vd2), t0),
                                1 => (f32x4_extract_lane::<1>(vd2), t1),
                                2 => (f32x4_extract_lane::<2>(vd2), t2),
                                _ => (f32x4_extract_lane::<3>(vd2), t3),
                            };
                            if d_lane < top.lim {
                                let j = t_lane as usize;
                                if j != i {
                                    top.push_full(d_lane, j as i16);
                                }
                            }
                        }
                        vlim = f32x4_splat(top.lim);
                    }
                }
            }
            p = rem_s + ((n - rem_s) & !3);
        }
    }
    while p < n {
        let dx = xs[p] - xi;
        let dy = ys[p] - yi;
        let d2 = dx * dx + dy * dy;
        if d2 < top.lim {
            let j = tis[p] as usize;
            if j != i {
                top.push_full(d2, j as i16);
            }
        }
        p += 1;
    }
    top
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn knn_outer_row_chord_search(
    cs: &ContinuumState,
    row_occ: &[u32; MICRO_CELL_H],
    cy0: usize,
    x3_lo: usize,
    x3_hi: usize,
    dt0: f32,
    db0: f32,
    i: usize,
    xi: f32,
    yi: f32,
    inv_mw: f32,
    m_ch: f32,
    dir_mask: u8,
    top: &mut KnnTop6,
) {
    let left_mask = if (dir_mask & 1) != 0 {
        (1u32 << x3_lo) - 1
    } else {
        0
    };
    let right_mask = if (dir_mask & 2) != 0 {
        (!0u64 << (x3_hi + 1)) as u32
    } else {
        0
    };
    let wing_mask = left_mask | right_mask;
    let mut phase = if wing_mask != 0 { 0usize } else { 6usize };
    let mut d = 2usize;
    let mut up_alive = (dir_mask & 4) != 0 && cy0 >= 2;
    let mut dn_alive = (dir_mask & 8) != 0 && cy0 + 2 < MICRO_CELL_H;
    let mut dn_turn = !up_alive;
    while phase < 6 || up_alive || dn_alive {
        let ry;
        let dy;
        let x_mask;
        if phase < 6 {
            let side = phase & 1;
            let row_sel = phase >> 1;
            phase += 1;
            x_mask = if side == 0 { left_mask } else { right_mask };
            if x_mask == 0 {
                continue;
            }
            if row_sel == 0 {
                ry = cy0;
                dy = 0.0;
            } else if row_sel == 1 {
                if cy0 == 0 {
                    continue;
                }
                ry = cy0 - 1;
                dy = dt0;
            } else {
                if cy0 + 1 >= MICRO_CELL_H {
                    continue;
                }
                ry = cy0 + 1;
                dy = db0;
            }
        } else {
            x_mask = !0u32;
            if !dn_turn && up_alive {
                ry = cy0 - d;
                dy = (yi - ((ry + 1) as f32) * m_ch - 1e-2).max(0.0);
                if dn_alive {
                    dn_turn = true;
                } else {
                    d += 1;
                }
                if dy * dy > top.lim {
                    up_alive = false;
                    continue;
                }
                up_alive = ry > 0;
            } else if dn_alive {
                ry = cy0 + d;
                dy = ((ry as f32) * m_ch - yi - 1e-2).max(0.0);
                dn_turn = false;
                d += 1;
                if dy * dy > top.lim {
                    dn_alive = false;
                    continue;
                }
                dn_alive = ry + 1 < MICRO_CELL_H;
            } else {
                break;
            }
        }
        let dy2 = dy * dy;
        if dy2 > top.lim {
            continue;
        }
        let masked = row_occ[ry.min(MICRO_CELL_H - 1)] & x_mask;
        if masked == 0 {
            continue;
        }
        let rem = wasm_sqrtf((top.lim - dy2).max(0.0));
        let xb_f = (xi + rem) * inv_mw + 1e-4;
        if xb_f < 0.0 {
            continue;
        }
        let xa_f = (xi - rem) * inv_mw - 1e-4;
        let cx_a = if xa_f <= 0.0 {
            0
        } else {
            (xa_f as usize).min(MICRO_CELL_W - 1)
        };
        let cx_b = (xb_f.min((MICRO_CELL_W - 1) as f32)) as usize;
        if cx_a > cx_b {
            continue;
        }
        let sub_bits = masked & (((2u64 << cx_b) - (1u64 << cx_a)) as u32);
        if sub_bits != 0 {
            let row_off = ry * MICRO_CELL_W;
            let ca = sub_bits.trailing_zeros() as usize;
            let cb = (31 - sub_bits.leading_zeros()) as usize;
            let e = (cs.field_cell_start[(row_off + cb + 1).min(MICRO_CELL_W * MICRO_CELL_H)]
                as usize)
                .min(MAX_MICRO);
            let s = (cs.field_cell_start[(row_off + ca).min(MICRO_CELL_W * MICRO_CELL_H)] as usize)
                .min(e);
            if e > s {
                *top = visit_knn_slice_reg(cs, s, e, i, xi, yi, *top);
            }
        }
    }
}

#[inline(never)]
fn sync_knn_and_build_csr(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    inv_mw: f32,
    inv_mh: f32,
    row_occ: &mut [u32; MICRO_CELL_H],
) -> usize {
    const MICRO_CELLS: usize = MICRO_CELL_W * MICRO_CELL_H; // 576
    cs.field_cell_start[..MICRO_CELLS + 1].fill(0);
    let min_age = min_continuum_age(ms);
    let mut clean_n = 0usize;
    let mut conf_n = 0usize;
    let mut active_max = 0usize;

    let mut i = 0usize;
    while i < MAX_MICRO {
        let st = ms.tracks.state[i];
        let tid = ms.tracks.track_id[i];
        if st != MICRO_STATE_INACTIVE {
            active_max = i + 1;
        }
        if st == MICRO_STATE_INACTIVE || tid == 0 {
            if st != MICRO_STATE_INACTIVE {
                ms.tracks.ftle_val[i] = -999.0;
            }
            if cs.prev_slot_tid[i] != 0 || cs.slot_dtheta_valid[i] != 0 {
                reset_slot_knn(cs, i);
                reset_slot_modal(ms, cs, i, false);
                cs.prev_slot_tid[i] = 0;
                cs.slot_clean_since_frame[i] = u32::MAX;
            } else if cs.modal_prev_ax_q13[i] != i16::MIN
                || cs.modal_cycles[i] != 0
                || cs.modal_zc_state[i] != 0
                || cs.modal_cross_age_s[i] != 0.0
            {
                reset_slot_modal(ms, cs, i, false);
            }
        } else {
            if tid != cs.prev_slot_tid[i] {
                reset_slot_knn(cs, i);
                reset_slot_modal(ms, cs, i, true);
                cs.prev_slot_tid[i] = tid;
                cs.slot_clean_since_frame[i] =
                    ms.tracks.birth_frame[i].max(cs.clean_epoch_frame);
            }
            let conf_eligible = st == MICRO_STATE_CONFIRMED && ms.tracks.age[i] >= min_age;
            if conf_eligible {
                let cf_idx = conf_n.min(MAX_MICRO - 1);
                cs.conf_slots[cf_idx] = i as u16;
                conf_n += 1;
            }
            if ms.tracks.coasting[i] != 0
                || ms.tracks.reacq_flag[i] != 0
                || ms.tracks.ambiguous_flag[i] != 0
            {
                cs.slot_clean_since_frame[i] = frame_idx.saturating_add(1);
                ms.tracks.ftle_val[i] = -999.0;
                if cs.knn[i].count != 0 || cs.slot_dtheta_valid[i] != 0 {
                    reset_slot_knn(cs, i);
                }
                reset_slot_modal(ms, cs, i, true);
            } else if conf_eligible {
                if frame_idx < 8 {
                    ms.tracks.ftle_val[i] = -999.0;
                }
                let clean_since = cs.slot_clean_since_frame[i];
                let mut dtv_n = cs.slot_dtheta_valid[i];
                let af = ms.tracks.ftle_a_frame[i];
                if af == frame_idx {
                    cs.slot_dtheta_a[i] = 0.0;
                    dtv_n = (dtv_n | 1u8) & !8u8;
                } else if (dtv_n & 1u8) != 0 {
                    if af == FTLE_INVALID_FRAME || af < clean_since || af > frame_idx {
                        cs.slot_dtheta_a[i] = 0.0;
                        dtv_n &= !(1u8 | 8u8);
                    } else {
                        dtv_n |= 8u8;
                    }
                } else {
                    dtv_n &= !8u8;
                }
                let bf = ms.tracks.ftle_b_frame[i];
                if bf == frame_idx {
                    cs.slot_dtheta_b[i] = 0.0;
                    dtv_n = (dtv_n | 2u8) & !16u8;
                } else if (dtv_n & 2u8) != 0 {
                    if bf == FTLE_INVALID_FRAME || bf < clean_since || bf > frame_idx {
                        cs.slot_dtheta_b[i] = 0.0;
                        dtv_n &= !(2u8 | 16u8);
                    } else {
                        dtv_n |= 16u8;
                    }
                } else {
                    dtv_n &= !16u8;
                }
                cs.slot_dtheta_valid[i] = dtv_n | 32u8;
                let cx = clamped_grid_coord(ms.tracks.pos_x[i], inv_mw, MICRO_CELL_W - 1);
                let cy = clamped_grid_coord(ms.tracks.pos_y[i], inv_mh, MICRO_CELL_H - 1);
                let cell = cy * MICRO_CELL_W + cx;
                let cn_idx = clean_n.min(MAX_MICRO - 1);
                cs.clean_slots[cn_idx] = i as u16;
                cs.field_cell_start[(cell + 1).min(MICRO_CELLS)] += 1;
                cs.uf_parent[cn_idx] = cell as i16;
                cs.comp_next[cn_idx] = i as i16;
                clean_n += 1;
            } else {
                ms.tracks.ftle_val[i] = -999.0;
                if cs.knn[i].count != 0 || cs.slot_dtheta_valid[i] != 0 {
                    reset_slot_knn(cs, i);
                }
                if cs.modal_prev_ax_q13[i] != i16::MIN
                    || cs.modal_cycles[i] != 0
                    || cs.modal_zc_state[i] != 0
                    || cs.modal_cross_age_s[i] != 0.0
                {
                    reset_slot_modal(ms, cs, i, true);
                }
            }
        }
        i += 1;
    }
    let clean_n = clean_n.min(MAX_MICRO);
    cs.clean_slots_len = clean_n as u16;
    cs.conf_slots_len = conf_n.min(MAX_MICRO) as u16;
    cs.active_max_slot = active_max.min(MAX_MICRO) as u16;
    cs.lists_frame_idx = frame_idx;

    let mut acc = 0u16;
    let mut cy = 0usize;
    while cy < MICRO_CELL_H {
        let mut r_mask = 0u32;
        let row_base = cy * MICRO_CELL_W;
        let mut cx = 0usize;
        while cx < MICRO_CELL_W {
            let cell = row_base + cx;
            let cnt = cs.field_cell_start[cell + 1];
            cs.comp_head[cell] = acc as i16;
            acc += cnt;
            cs.field_cell_start[cell + 1] = acc;
            if cnt > 0 {
                r_mask |= 1u32 << cx;
            }
            cx += 1;
        }
        row_occ[cy] = r_mask;
        cy += 1;
    }
    let mut kc = 0usize;
    while kc < clean_n {
        let cell = (cs.uf_parent[kc] as usize).min(MICRO_CELLS - 1);
        let dst = (cs.comp_head[cell] as usize).min(MAX_MICRO - 1);
        cs.comp_head[cell] = (dst + 1) as i16;
        let tk = (cs.comp_next[kc] as usize).min(MAX_MICRO - 1);
        cs.field_grid_ti[dst] = tk as u16;
        cs.field_grid_x[dst] = ms.tracks.pos_x[tk];
        cs.field_grid_y[dst] = ms.tracks.pos_y[tk];
        kc += 1;
    }
    clean_n
}

#[inline(always)]
fn visit_knn_slice_full_noskip(
    cs: &ContinuumState,
    s: usize,
    e: usize,
    xi: f32,
    yi: f32,
    mut top: KnnTop6,
) -> KnnTop6 {
    let e = e.min(MAX_MICRO);
    let mut p = s.min(e);
    let xs = &cs.field_grid_x[..MAX_MICRO];
    let ys = &cs.field_grid_y[..MAX_MICRO];
    let tis = &cs.field_grid_ti[..MAX_MICRO];
    if e - p >= 4 {
        use crate::simd::*;
        let xs_p = xs.as_ptr();
        let ys_p = ys.as_ptr();
        let tis_p = tis.as_ptr();
        let vxi = f32x4_splat(xi);
        let vyi = f32x4_splat(yi);
        let mut vlim = f32x4_splat(top.lim);
        while p + 4 <= e {
            let vx = unsafe { core::ptr::read_unaligned(xs_p.add(p) as *const v128) };
            let vy = unsafe { core::ptr::read_unaligned(ys_p.add(p) as *const v128) };
            let vdx = f32x4_sub(vx, vxi);
            let vdy = f32x4_sub(vy, vyi);
            let vd2 = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
            let cmp = f32x4_lt(vd2, vlim);
            if v128_any_true(cmp) {
                let mask = (i32x4_bitmask(cmp) as u32) & 0xF;
                if (mask & 1) != 0 {
                    let d = f32x4_extract_lane::<0>(vd2);
                    if d < top.lim {
                        top.push_full(d, unsafe { *tis_p.add(p) } as i16);
                    }
                }
                if (mask & 2) != 0 {
                    let d = f32x4_extract_lane::<1>(vd2);
                    if d < top.lim {
                        top.push_full(d, unsafe { *tis_p.add(p + 1) } as i16);
                    }
                }
                if (mask & 4) != 0 {
                    let d = f32x4_extract_lane::<2>(vd2);
                    if d < top.lim {
                        top.push_full(d, unsafe { *tis_p.add(p + 2) } as i16);
                    }
                }
                if (mask & 8) != 0 {
                    let d = f32x4_extract_lane::<3>(vd2);
                    if d < top.lim {
                        top.push_full(d, unsafe { *tis_p.add(p + 3) } as i16);
                    }
                }
                vlim = f32x4_splat(top.lim);
            }
            p += 4;
        }
    }
    while p < e {
        let dx = xs[p] - xi;
        let dy = ys[p] - yi;
        let d2 = dx * dx + dy * dy;
        if d2 < top.lim {
            top.push_full(d2, tis[p] as i16);
        }
        p += 1;
    }
    top
}

#[inline(always)]
fn visit_knn_slice_noskip(
    cs: &ContinuumState,
    s: usize,
    e: usize,
    xi: f32,
    yi: f32,
    mut top: KnnTop6,
) -> KnnTop6 {
    let e = e.min(MAX_MICRO);
    let mut p = s.min(e);
    if top.cnt < MICRO_KNN_K as u8 {
        let xs = &cs.field_grid_x[..MAX_MICRO];
        let ys = &cs.field_grid_y[..MAX_MICRO];
        let tis = &cs.field_grid_ti[..MAX_MICRO];
        while p < e && top.cnt < MICRO_KNN_K as u8 {
            let dx = xs[p] - xi;
            let dy = ys[p] - yi;
            let d2 = dx * dx + dy * dy;
            if d2 <= top.lim {
                top.push_lt6(d2, tis[p] as i16);
            }
            p += 1;
        }
    }
    if p < e {
        return visit_knn_slice_full_noskip(cs, p, e, xi, yi, top);
    }
    top
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn search_knn_dense_cell(
    cs: &ContinuumState,
    cy0: usize,
    cx0: usize,
    x3_lo: usize,
    x3_hi: usize,
    s0: usize,
    e0: usize,
    dst_i: usize,
    s_l: usize,
    e_l: usize,
    s_r: usize,
    e_r: usize,
    s_top: usize,
    e_top: usize,
    s_bot: usize,
    e_bot: usize,
    xi: f32,
    yi: f32,
    dl2: f32,
    dr2: f32,
    dt2: f32,
    db2: f32,
    min_b0_sq: f32,
    top: &mut KnnTop6,
) -> bool {
    const MICRO_CELLS: usize = MICRO_CELL_W * MICRO_CELL_H;
    let mut t = *top;
    let c0_left = dst_i.max(s0).min(e0);
    let c0_right = (dst_i + 1).max(s0).min(e0);
    let side_swap = dr2 < dl2;
    let row_swap = db2 < dt2;
    let mut stage = 0usize;
    while stage < 6 {
        let (sq, eq) = match stage {
            0 => (s0, c0_left),
            1 => (c0_right, e0),
            2 | 3 => {
                if stage == 2 && t.cnt == MICRO_KNN_K as u8 && t.lim <= min_b0_sq {
                    *top = t;
                    return true;
                }
                let is_left = (stage == 2) ^ side_swap;
                if is_left {
                    if e_l > s_l && dl2 <= t.lim {
                        (s_l, e_l)
                    } else {
                        (0, 0)
                    }
                } else if e_r > s_r && dr2 <= t.lim {
                    (s_r, e_r)
                } else {
                    (0, 0)
                }
            }
            _ => {
                let is_top = (stage == 4) ^ row_swap;
                let (sb, eb, dtb) = if is_top {
                    (s_top, e_top, dt2)
                } else {
                    (s_bot, e_bot, db2)
                };
                if eb <= sb || dtb > t.lim {
                    (0, 0)
                } else if dtb + dl2 > t.lim || dtb + dr2 > t.lim {
                    let lim_now = t.lim;
                    let row_base_y = (if is_top { cy0 - 1 } else { cy0 + 1 }) * MICRO_CELL_W;
                    let xl = if dtb + dl2 > lim_now { cx0 } else { x3_lo };
                    let xr = if dtb + dr2 > lim_now { cx0 } else { x3_hi };
                    (
                        (cs.field_cell_start[(row_base_y + xl).min(MICRO_CELLS)] as usize)
                            .min(MAX_MICRO),
                        (cs.field_cell_start[(row_base_y + xr + 1).min(MICRO_CELLS)] as usize)
                            .min(MAX_MICRO),
                    )
                } else {
                    (sb, eb)
                }
            }
        };
        stage += 1;
        if eq > sq {
            t = visit_knn_slice_noskip(cs, sq, eq, xi, yi, t);
        }
    }
    *top = t;
    false
}

/// NaN-free fast float minimum when inputs are proven non-NaN (e.g. `clamp_pos0` outputs).
#[inline(always)]
fn fast_min_f32(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

/// NaN-free fast float maximum when inputs are proven non-NaN.
#[inline(always)]
fn fast_max_f32(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// Phase 1: Maintains continuity flags, resets interrupted/ambiguous slots across both states,
/// accumulates per-slot local rotation increments for unwrapped `SE(2)` (`H5`), builds contiguous
/// `32x18` confirmed CSR hash, and updates exact sparse `k`-NN (`k = 6`, `r_max = 192*q`).
#[inline(never)]
pub fn build_sparse_knn_graph(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    frame_w: usize,
    frame_h: usize,
    q: f32,
) {
    if ms.swarm_active == 0 {
        cs.reset();
        cs.initialized = 0;
        return;
    }
    if cs.initialized == 0 {
        cs.reset();
    }
    const MICRO_CELLS: usize = MICRO_CELL_W * MICRO_CELL_H; // 576
    let inv_mw = (MICRO_CELL_W as f32) / (frame_w as f32);
    let inv_mh = (MICRO_CELL_H as f32) / (frame_h as f32);
    let m_cw = (frame_w as f32) / (MICRO_CELL_W as f32);
    let m_ch = (frame_h as f32) / (MICRO_CELL_H as f32);

    let mut row_occ = [0u32; MICRO_CELL_H];
    let _clean_n = sync_knn_and_build_csr(ms, cs, frame_idx, inv_mw, inv_mh, &mut row_occ);

    let dense_knn = ms.confirmed_count >= 960;
    if dense_knn && frame_idx < 7 {
        cs.rigid_seed_n = 0;
        cs.initialized &= !2u8;
        cs.knn_edge_count = 0;
        cs.mean_strain = 0.0;
        cs.cached_knn_frame = u32::MAX;
        return;
    }
    let reuse_knn_cached = dense_knn
        && frame_idx >= 8
        && (frame_idx & 1) == 0
        && cs.cached_knn_frame == frame_idx - 1;

    // Step 1b: Exact 3x3 fused + center-out row-chord k-NN search over 32x18 CSR
    let r_max = if dense_knn {
        44.0 * q
    } else {
        192.0 * q
    };
    let r_max2 = r_max * r_max;
    let min_ref_dist = 8.0 * q;

    let mut total_edges = 0u32;
    let mut observed_strain_edges = 0u32;
    let mut strain_sum = 0.0f64;
    let mut rigid_seed_n = 0usize;

    let mut cy0 = 0usize;
    while cy0 < MICRO_CELL_H {
        let mut row_m = row_occ[cy0];
        if row_m == 0 {
            cy0 += 1;
            continue;
        }
        let has_top = cy0 > 0;
        let has_bot = cy0 + 1 < MICRO_CELL_H;
        let wall_t_safe = if has_top { (cy0 as f32) * m_ch + 1e-2 } else { -1e9 };
        let wall_b_safe = if has_bot { ((cy0 + 1) as f32) * m_ch - 1e-2 } else { 1e9 };
        let y3_lo = cy0.saturating_sub(1);
        let y3_hi = (cy0 + 1).min(MICRO_CELL_H - 1);
        let wall3_t_safe = if y3_lo > 0 { (y3_lo as f32) * m_ch + 1e-2 } else { -1e9 };
        let wall3_b_safe = if y3_hi + 1 < MICRO_CELL_H {
            ((y3_hi + 1) as f32) * m_ch - 1e-2
        } else {
            1e9
        };
        let row_base = cy0 * MICRO_CELL_W;

        while row_m != 0 {
            let cx0 = row_m.trailing_zeros() as usize;
            row_m &= row_m - 1;
            let c0 = row_base + cx0;
            let e0 = (cs.field_cell_start[(c0 + 1).min(MICRO_CELLS)] as usize).min(MAX_MICRO);
            let s0 = (cs.field_cell_start[c0.min(MICRO_CELLS)] as usize).min(e0);
            let x3_lo = cx0.saturating_sub(1);
            let x3_hi = (cx0 + 1).min(MICRO_CELL_W - 1);
            let wall_l_safe = if cx0 > 0 { (cx0 as f32) * m_cw + 1e-2 } else { -1e9 };
            let wall_r_safe = if cx0 + 1 < MICRO_CELL_W {
                ((cx0 + 1) as f32) * m_cw - 1e-2
            } else {
                1e9
            };
            let wall3_l_safe = if x3_lo > 0 { (x3_lo as f32) * m_cw + 1e-2 } else { -1e9 };
            let wall3_r_safe = if x3_hi + 1 < MICRO_CELL_W {
                ((x3_hi + 1) as f32) * m_cw - 1e-2
            } else {
                1e9
            };
            let mask3 = (!0u32 >> (31 - x3_hi)) & (!0u32 << x3_lo);
            let (s_top, e_top) = if has_top {
                row_mask_slice(cs, &row_occ, cy0 - 1, mask3)
            } else {
                (0, 0)
            };
            let (s_bot, e_bot) = if has_bot {
                row_mask_slice(cs, &row_occ, cy0 + 1, mask3)
            } else {
                (0, 0)
            };
            let dense_c0 = (e0 - s0) >= 7;
            let (s_mid, e_mid) = if !dense_c0 {
                row_mask_slice(cs, &row_occ, cy0, mask3)
            } else {
                (s0, e0)
            };
            let split_mid = dense_c0;
            let (s_l, e_l) = if split_mid && cx0 > 0 && (row_occ[cy0] & (1u32 << (cx0 - 1))) != 0
            {
                let e_n = s0;
                let s_n = (cs.field_cell_start[(c0 - 1).min(MICRO_CELLS)] as usize).min(e_n);
                (s_n, e_n)
            } else {
                (0, 0)
            };
            let (s_r, e_r) = if split_mid
                && cx0 + 1 < MICRO_CELL_W
                && (row_occ[cy0] & (1u32 << (cx0 + 1))) != 0
            {
                let e_n = (cs.field_cell_start[(c0 + 2).min(MICRO_CELLS)] as usize).min(MAX_MICRO);
                let s_n = e0.min(e_n);
                (s_n, e_n)
            } else {
                (0, 0)
            };

            let mut dst_i = s0;
            while dst_i < e0 {
                let i = (cs.field_grid_ti[dst_i] as usize).min(MAX_MICRO - 1);
                let xi = cs.field_grid_x[dst_i];
                let yi = cs.field_grid_y[dst_i];

                if reuse_knn_cached && cs.knn[i].count == MICRO_KNN_K as u8 {
                    let idx6 = cs.knn[i].idx;
                    let tid6 = cs.knn[i].tid;
                    if (idx6[0] | idx6[1] | idx6[2] | idx6[3] | idx6[4] | idx6[5]) >= 0 {
                        let j0 = (idx6[0] as usize).min(MAX_MICRO - 1);
                        let j1 = (idx6[1] as usize).min(MAX_MICRO - 1);
                        let j2 = (idx6[2] as usize).min(MAX_MICRO - 1);
                        let j3 = (idx6[3] as usize).min(MAX_MICRO - 1);
                        let j4 = (idx6[4] as usize).min(MAX_MICRO - 1);
                        let j5 = (idx6[5] as usize).min(MAX_MICRO - 1);
                        let dtv_all = cs.slot_dtheta_valid[j0]
                            & cs.slot_dtheta_valid[j1]
                            & cs.slot_dtheta_valid[j2]
                            & cs.slot_dtheta_valid[j3]
                            & cs.slot_dtheta_valid[j4]
                            & cs.slot_dtheta_valid[j5];
                        if (dtv_all & 32u8) != 0
                            && ((ms.tracks.track_id[j0] ^ tid6[0])
                                | (ms.tracks.track_id[j1] ^ tid6[1])
                                | (ms.tracks.track_id[j2] ^ tid6[2])
                                | (ms.tracks.track_id[j3] ^ tid6[3])
                                | (ms.tracks.track_id[j4] ^ tid6[4])
                                | (ms.tracks.track_id[j5] ^ tid6[5]))
                                == 0
                        {
                            let dx0 = ms.tracks.pos_x[j0] - xi;
                            let dy0 = ms.tracks.pos_y[j0] - yi;
                            let d2_0 = dx0 * dx0 + dy0 * dy0;
                            let dx1 = ms.tracks.pos_x[j1] - xi;
                            let dy1 = ms.tracks.pos_y[j1] - yi;
                            let d2_1 = dx1 * dx1 + dy1 * dy1;
                            let dx2 = ms.tracks.pos_x[j2] - xi;
                            let dy2 = ms.tracks.pos_y[j2] - yi;
                            let d2_2 = dx2 * dx2 + dy2 * dy2;
                            let dx3 = ms.tracks.pos_x[j3] - xi;
                            let dy3 = ms.tracks.pos_y[j3] - yi;
                            let d2_3 = dx3 * dx3 + dy3 * dy3;
                            let dx4 = ms.tracks.pos_x[j4] - xi;
                            let dy4 = ms.tracks.pos_y[j4] - yi;
                            let d2_4 = dx4 * dx4 + dy4 * dy4;
                            let dx5 = ms.tracks.pos_x[j5] - xi;
                            let dy5 = ms.tracks.pos_y[j5] - yi;
                            let d2_5 = dx5 * dx5 + dy5 * dy5;
                            if ((d2_0 <= r_max2) as u8
                                & (d2_1 <= r_max2) as u8
                                & (d2_2 <= r_max2) as u8
                                & (d2_3 <= r_max2) as u8
                                & (d2_4 <= r_max2) as u8
                                & (d2_5 <= r_max2) as u8)
                                != 0
                            {
                                let (obs_inc, strain_inc, persist_fwd) =
                                    update_knn_slot_edges_same6_wasm(
                                        ms,
                                        cs,
                                        i,
                                        xi,
                                        yi,
                                        j0,
                                        j1,
                                        j2,
                                        j3,
                                        j4,
                                        j5,
                                        d2_0,
                                        d2_1,
                                        d2_2,
                                        d2_3,
                                        d2_4,
                                        d2_5,
                                        q,
                                        min_ref_dist,
                                    );
                                if persist_fwd {
                                    let sidx = rigid_seed_n.min(MAX_MICRO - 1);
                                    cs.rigid_seeds[sidx] = i as u16;
                                    rigid_seed_n += 1;
                                }
                                observed_strain_edges += obs_inc;
                                strain_sum += strain_inc;
                                total_edges += MICRO_KNN_K as u32;
                                dst_i += 1;
                                continue;
                            }
                        }
                    }
                }

                let dt0 = clamp_pos0(yi - wall_t_safe);
                let db0 = clamp_pos0(wall_b_safe - yi);
                let dt2 = dt0 * dt0;
                let db2 = db0 * db0;
                let dl0 = clamp_pos0(xi - wall_l_safe);
                let dr0 = clamp_pos0(wall_r_safe - xi);
                let dl2 = dl0 * dl0;
                let dr2 = dr0 * dr0;
                let mut top = KnnTop6::new(r_max2);
                let early_c0 = if !split_mid {
                    let left_end = dst_i;
                    let right_start = dst_i + 1;
                    // Skip empty center-cell sub-slices to avoid extra loop/br_table dispatches in solo/edge cells.
                    let mut stage = if e_mid <= s_mid + 1 {
                        2usize
                    } else if left_end <= s_mid {
                        1usize
                    } else {
                        0usize
                    };
                    while stage < 4 {
                        if stage == 2
                            && top.cnt == MICRO_KNN_K as u8
                            && top.lim <= fast_min_f32(dt2, db2)
                        {
                            break;
                        }
                        let (sq, eq) = if stage == 0 {
                            (s_mid, left_end)
                        } else if stage == 1 {
                            (right_start, e_mid)
                        } else {
                            let is_top = (stage == 2) ^ (db2 < dt2);
                            let (sb, eb, dtb) = if is_top {
                                (s_top, e_top, dt2)
                            } else {
                                (s_bot, e_bot, db2)
                            };
                            if eb <= sb || (top.cnt == MICRO_KNN_K as u8 && dtb > top.lim) {
                                (0, 0)
                            } else if top.cnt == MICRO_KNN_K as u8
                                && (dtb + dl2 > top.lim || dtb + dr2 > top.lim)
                            {
                                let lim_now = top.lim;
                                let row_tb = (if is_top { cy0 - 1 } else { cy0 + 1 })
                                    .min(MICRO_CELL_H - 1);
                                let xl_0 = if dtb + dl2 > lim_now { cx0 } else { x3_lo };
                                let xr_0 = if dtb + dr2 > lim_now { cx0 } else { x3_hi };
                                let mask_tb = (!0u32 >> (31 - xr_0)) & (!0u32 << xl_0);
                                let occ_tb = row_occ[row_tb] & mask_tb;
                                if occ_tb == 0 {
                                    (0, 0)
                                } else {
                                    let row_base_y = row_tb * MICRO_CELL_W;
                                    let xl = occ_tb.trailing_zeros() as usize;
                                    let xr = (31 - occ_tb.leading_zeros()) as usize;
                                    (
                                        (cs.field_cell_start[(row_base_y + xl).min(MICRO_CELLS)]
                                            as usize)
                                            .min(MAX_MICRO),
                                        (cs.field_cell_start
                                            [(row_base_y + xr + 1).min(MICRO_CELLS)]
                                            as usize)
                                            .min(MAX_MICRO),
                                    )
                                }
                            } else {
                                (sb, eb)
                            }
                        };
                        stage += 1;
                        if eq > sq {
                            top = visit_knn_slice_noskip(cs, sq, eq, xi, yi, top);
                        }
                    }
                    false
                } else {
                    let min_b0 = fast_min_f32(fast_min_f32(dl0, dr0), fast_min_f32(dt0, db0));
                    let mut top_mem = top;
                    let early = search_knn_dense_cell(
                        cs,
                        cy0,
                        cx0,
                        x3_lo,
                        x3_hi,
                        s0,
                        e0,
                        dst_i,
                        s_l,
                        e_l,
                        s_r,
                        e_r,
                        s_top,
                        e_top,
                        s_bot,
                        e_bot,
                        xi,
                        yi,
                        dl2,
                        dr2,
                        dt2,
                        db2,
                        min_b0 * min_b0,
                        &mut top_mem,
                    );
                    top = top_mem;
                    early && dense_c0
                };

                if !early_c0 && !(dense_knn && top.cnt >= 3) {
                    let dl3 = clamp_pos0(xi - wall3_l_safe);
                    let dr3 = clamp_pos0(wall3_r_safe - xi);
                    let dt3 = clamp_pos0(yi - wall3_t_safe);
                    let db3 = clamp_pos0(wall3_b_safe - yi);
                    let lim = top.lim;
                    let min3 = fast_min_f32(fast_min_f32(dl3, dr3), fast_min_f32(dt3, db3));
                    if !(top.cnt == MICRO_KNN_K as u8 && lim <= min3 * min3) {
                        let dir_mask = ((dl3 * dl3 <= lim) as u8)
                            | (((dr3 * dr3 <= lim) as u8) << 1)
                            | (((dt3 * dt3 <= lim) as u8) << 2)
                            | (((db3 * db3 <= lim) as u8) << 3);
                        if dir_mask != 0 {
                            let mut top_mem = top;
                            knn_outer_row_chord_search(
                                cs,
                                &row_occ,
                                cy0,
                                x3_lo,
                                x3_hi,
                                dt0,
                                db0,
                                i,
                                xi,
                                yi,
                                inv_mw,
                                m_ch,
                                dir_mask,
                                &mut top_mem,
                            );
                            top = top_mem;
                        }
                    }
                }
                let best_cnt = top.cnt as usize;

                let (obs_inc, strain_inc, persist_fwd) = update_knn_slot_edges(
                    ms,
                    cs,
                    i,
                    xi,
                    yi,
                    top.p0,
                    top.p1,
                    top.p2,
                    top.p3,
                    top.p4,
                    top.p5,
                    best_cnt,
                    frame_idx,
                    q,
                    min_ref_dist,
                );
                if persist_fwd {
                    let sidx = rigid_seed_n.min(MAX_MICRO - 1);
                    cs.rigid_seeds[sidx] = i as u16;
                    rigid_seed_n += 1;
                }
                observed_strain_edges += obs_inc;
                strain_sum += strain_inc;
                total_edges += best_cnt as u32;

                dst_i += 1;
            }
        }
        cy0 += 1;
    }

    cs.rigid_seed_n = rigid_seed_n.min(MAX_MICRO) as u16;
    if rigid_seed_n > 0 {
        cs.initialized |= 2u8;
    } else {
        cs.initialized &= !2u8;
    }
    cs.knn_edge_count = total_edges;
    cs.mean_strain = if observed_strain_edges > 0 {
        finite_or_zero((strain_sum / (observed_strain_edges as f64)) as f32)
    } else {
        0.0
    };
    if !reuse_knn_cached {
        cs.cached_knn_frame = frame_idx;
    }
}

#[inline(always)]
fn find_matched_knn_slot(old_tid: &[u32; MICRO_KNN_K], old_cnt: usize, j_tid: u32) -> usize {
    let mut m = 0usize;
    while m < old_cnt {
        if old_tid[m] == j_tid {
            return m;
        }
        m += 1;
    }
    old_cnt
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn compute_knn_star_dtheta(
    ms: &MicroSwarmState,
    best_idx: &[i16; MICRO_KNN_K],
    star_mask: u8,
    pxi: f32,
    pyi: f32,
    xi: f32,
    yi: f32,
    q: f32,
) -> (bool, f32) {
    let mut sx0 = 0.0f32;
    let mut sy0 = 0.0f32;
    let mut sxt = 0.0f32;
    let mut syt = 0.0f32;
    let mut sum_uu = 0.0f32;
    let mut sum_vv = 0.0f32;
    let mut sum_uv_cross = 0.0f32;
    let mut sum_uv_dot = 0.0f32;
    let mut sc_n = 1.0f32;
    let mut m = star_mask;
    while m != 0 {
        let slot_k = (m.trailing_zeros() as usize).min(MICRO_KNN_K - 1);
        m &= m - 1;
        let j = (best_idx[slot_k] as usize).min(MAX_MICRO - 1);
        let ux = ms.tracks.prev_x[j] - pxi;
        let uy = ms.tracks.prev_y[j] - pyi;
        let vx = ms.tracks.pos_x[j] - xi;
        let vy = ms.tracks.pos_y[j] - yi;
        sx0 += ux;
        sy0 += uy;
        sxt += vx;
        syt += vy;
        sum_uu += ux * ux + uy * uy;
        sum_vv += vx * vx + vy * vy;
        sum_uv_cross += ux * vy - uy * vx;
        sum_uv_dot += ux * vx + uy * vy;
        sc_n += 1.0;
    }
    let inv_s = 1.0 / sc_n;
    let sum_r2 = sum_uu - (sx0 * sx0 + sy0 * sy0) * inv_s;
    let sq_t = sum_vv - (sxt * sxt + syt * syt) * inv_s;
    let s_cross = sum_uv_cross - (sx0 * syt - sy0 * sxt) * inv_s;
    let c_dot = sum_uv_dot - (sx0 * sxt + sy0 * syt) * inv_s;
    let norm = wasm_sqrtf(s_cross * s_cross + c_dot * c_dot);
    if sum_r2 >= 25.0 * q * q && norm > 1e-4 {
        let sq_e = (sq_t + sum_r2 - 2.0 * norm).max(0.0);
        if (sq_e * inv_s) <= 0.80 * q * 0.80 * q {
            let dt_cand = fast_atan2f(s_cross, c_dot);
            if dt_cand.is_finite() {
                return (true, dt_cand);
            }
        }
    }
    (false, 0.0)
}

#[inline(always)]
fn sqrt_top6_d2(best_d2: &[f32; MICRO_KNN_K]) -> [f32; MICRO_KNN_K] {
    {
        use crate::simd::{f32x4, f32x4_extract_lane, f32x4_max, f32x4_splat, f32x4_sqrt};
        let z = f32x4_splat(0.0);
        let v0 = f32x4_sqrt(f32x4_max(
            f32x4(best_d2[0], best_d2[1], best_d2[2], best_d2[3]),
            z,
        ));
        let v1 = f32x4_sqrt(f32x4_max(f32x4(best_d2[4], best_d2[5], 0.0, 0.0), z));
        [
            f32x4_extract_lane::<0>(v0),
            f32x4_extract_lane::<1>(v0),
            f32x4_extract_lane::<2>(v0),
            f32x4_extract_lane::<3>(v0),
            f32x4_extract_lane::<0>(v1),
            f32x4_extract_lane::<1>(v1),
        ]
    }
}

#[inline(never)]
fn update_knn_slot_init(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    i: usize,
    best_idx: &[i16; MICRO_KNN_K],
    best_d2: &[f32; MICRO_KNN_K],
    best_cnt: usize,
    dtv: u8,
) {
    let dists = sqrt_top6_d2(best_d2);
    if (dtv & (8u8 | 16u8)) != 0 {
        if (dtv & 8u8) != 0 {
            cs.slot_dtheta_a[i] = 0.0;
        }
        if (dtv & 16u8) != 0 {
            cs.slot_dtheta_b[i] = 0.0;
        }
        cs.slot_dtheta_valid[i] = dtv & !(1u8 | 2u8 | 4u8 | 8u8 | 16u8);
    } else {
        cs.slot_dtheta_valid[i] = dtv & !4u8;
    }
    let slot = &mut cs.knn[i];
    slot.idx = *best_idx;
    slot.strain_q15 = [0; MICRO_KNN_K];
    slot.rigid_streak = [0; MICRO_KNN_K];
    slot.count = best_cnt as u8;
    slot.persist_mask = 0;
    let mut k = 0usize;
    while k < best_cnt {
        let j = (best_idx[k] as usize).min(MAX_MICRO - 1);
        slot.tid[k] = ms.tracks.track_id[j];
        slot.ref_dist[k] = dists[k];
        k += 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn finalize_knn_slot_dtheta(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    i: usize,
    xi: f32,
    yi: f32,
    q: f32,
    dtv: u8,
    need_a: bool,
    need_b: bool,
    star_mask: u8,
    pxi: f32,
    pyi: f32,
    has_persist_fwd: bool,
) {
    let (star_ok, dtheta_loc) = if (need_a || need_b) && star_mask.count_ones() >= 2 {
        compute_knn_star_dtheta(ms, &cs.knn[i].idx, star_mask, pxi, pyi, xi, yi, q)
    } else {
        (false, 0.0)
    };
    let mut dtv_out = dtv;
    if need_a {
        if star_ok {
            cs.slot_dtheta_a[i] += dtheta_loc;
        } else {
            cs.slot_dtheta_a[i] = 0.0;
            dtv_out &= !(1u8 | 8u8);
        }
    }
    if need_b {
        if star_ok {
            cs.slot_dtheta_b[i] += dtheta_loc;
        } else {
            cs.slot_dtheta_b[i] = 0.0;
            dtv_out &= !(2u8 | 16u8);
        }
    }
    cs.slot_dtheta_valid[i] = if has_persist_fwd {
        dtv_out | 4u8
    } else {
        dtv_out & !4u8
    };
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn eval_knn_edges_6_wasm(
    ms: &MicroSwarmState,
    dst: &mut KnnSlot,
    i: usize,
    d2_0: f32,
    d2_1: f32,
    d2_2: f32,
    d2_3: f32,
    d2_4: f32,
    d2_5: f32,
    j0: usize,
    j1: usize,
    j2: usize,
    j3: usize,
    j4: usize,
    j5: usize,
    min_ref_dist: f32,
    q: f32,
    need_dtheta: bool,
    pxi: f32,
    pyi: f32,
    matched_mask: u8,
    vref0_old: crate::simd::v128,
    vref1_old: crate::simd::v128,
    prev_strain: [u16; MICRO_KNN_K],
    prev_streak: [u8; MICRO_KNN_K],
) -> (f32, u8, u8, bool) {
    use crate::simd::*;
    let vz = f32x4_splat(0.0);
    let v0 = f32x4_sqrt(f32x4_max(f32x4(d2_0, d2_1, d2_2, d2_3), vz));
    let v1 = f32x4_sqrt(f32x4_max(f32x4(d2_4, d2_5, 0.0, 0.0), vz));
    let (vref0, vref1) = if matched_mask != 0x3Fu8 {
        if (matched_mask & 1u8) == 0 {
            dst.ref_dist[0] = f32x4_extract_lane::<0>(v0);
        }
        if (matched_mask & 2u8) == 0 {
            dst.ref_dist[1] = f32x4_extract_lane::<1>(v0);
        }
        if (matched_mask & 4u8) == 0 {
            dst.ref_dist[2] = f32x4_extract_lane::<2>(v0);
        }
        if (matched_mask & 8u8) == 0 {
            dst.ref_dist[3] = f32x4_extract_lane::<3>(v0);
        }
        if (matched_mask & 16u8) == 0 {
            dst.ref_dist[4] = f32x4_extract_lane::<0>(v1);
        }
        if (matched_mask & 32u8) == 0 {
            dst.ref_dist[5] = f32x4_extract_lane::<1>(v1);
        }
        let vmm = i32x4_splat(matched_mask as i32);
        let izero = i32x4_splat(0);
        let m0 = i32x4_ne(v128_and(vmm, i32x4(1, 2, 4, 8)), izero);
        let m1 = i32x4_ne(
            v128_or(v128_and(vmm, i32x4(16, 32, 0, 0)), i32x4(0, 0, 1, 1)),
            izero,
        );
        (
            v128_bitselect(vref0_old, v0, m0),
            v128_bitselect(vref1_old, v1, m1),
        )
    } else {
        (vref0_old, vref1_old)
    };
    let vmin_ref = f32x4_splat(min_ref_dist);
    let vinst0 = f32x4_div(
        f32x4_abs(f32x4_sub(v0, vref0)),
        f32x4_max(vref0, vmin_ref),
    );
    let vinst1 = f32x4_div(
        f32x4_abs(f32x4_sub(v1, vref1)),
        f32x4_max(vref1, vmin_ref),
    );
    let vinv_sc = f32x4_splat(INV_KNN_STRAIN_SCALE);
    let vprev0 = f32x4_mul(
        f32x4(
            prev_strain[0] as f32,
            prev_strain[1] as f32,
            prev_strain[2] as f32,
            prev_strain[3] as f32,
        ),
        vinv_sc,
    );
    let vprev1 = f32x4_mul(
        f32x4(prev_strain[4] as f32, prev_strain[5] as f32, 0.0, 0.0),
        vinv_sc,
    );
    let vf030 = f32x4_splat(0.30);
    let vema0 = f32x4_add(vprev0, f32x4_mul(vf030, f32x4_sub(vinst0, vprev0)));
    let vema1 = f32x4_add(vprev1, f32x4_mul(vf030, f32x4_sub(vinst1, vprev1)));
    let slot_strain = hsum_f32x4(vema0)
        + f32x4_extract_lane::<0>(vema1)
        + f32x4_extract_lane::<1>(vema1);

    let v2 = f32x4_splat(2.0);
    let vscale = f32x4_splat(KNN_STRAIN_SCALE);
    let v05 = f32x4_splat(0.5);
    let vq0 = f32x4_add(f32x4_mul(f32x4_min(f32x4_max(vema0, vz), v2), vscale), v05);
    let vq1 = f32x4_add(f32x4_mul(f32x4_min(f32x4_max(vema1, vz), v2), vscale), v05);
    let vthresh = f32x4_splat(KNN_STRAIN_THRESH);
    let v045 = f32x4_splat(0.045);
    let ok_s0 = i32x4_bitmask(v128_and(
        f32x4_lt(vema0, vthresh),
        f32x4_lt(vinst0, v045),
    )) as u8;
    let ok_s1 = (i32x4_bitmask(v128_and(
        f32x4_lt(vema1, vthresh),
        f32x4_lt(vinst1, v045),
    )) as u8)
        & 3u8;
    let ok_streak = (ok_s0 | (ok_s1 << 4)) & matched_mask;

    let mut persist_mask = 0u8;
    let mut has_persist_fwd = false;
    macro_rules! update_edge_lane {
        ($k:expr, $vq_vec:expr, $lane:literal, $j:expr) => {{
            let q15_new = f32x4_extract_lane::<$lane>($vq_vec) as u16;
            let str_new = if ((ok_streak >> $k) & 1u8) != 0 {
                prev_streak[$k].saturating_add(1)
            } else {
                0
            };
            dst.strain_q15[$k] = q15_new;
            dst.rigid_streak[$k] = str_new;
            if str_new >= RIGID_PERSIST_STREAK && q15_new < KNN_STRAIN_THRESH_Q15 {
                persist_mask |= 1u8 << $k;
                if ($j as isize) > (i as isize) {
                    has_persist_fwd = true;
                }
            }
        }};
    }
    update_edge_lane!(0, vq0, 0, j0);
    update_edge_lane!(1, vq0, 1, j1);
    update_edge_lane!(2, vq0, 2, j2);
    update_edge_lane!(3, vq0, 3, j3);
    update_edge_lane!(4, vq1, 0, j4);
    update_edge_lane!(5, vq1, 1, j5);

    let mut star_mask = 0u8;
    if need_dtheta && ok_streak != 0 {
        let vq = f32x4_splat(q);
        let prev_x = &ms.tracks.prev_x;
        let prev_y = &ms.tracks.prev_y;
        if (ok_streak & 0x0Fu8) != 0 {
            let vdx = f32x4_sub(
                f32x4(prev_x[j0], prev_x[j1], prev_x[j2], prev_x[j3]),
                f32x4_splat(pxi),
            );
            let vdy = f32x4_sub(
                f32x4(prev_y[j0], prev_y[j1], prev_y[j2], prev_y[j3]),
                f32x4_splat(pyi),
            );
            let vdp2 = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
            let vlo = f32x4_max(f32x4_sub(v0, vq), vz);
            let vhi = f32x4_add(v0, vq);
            let vok = v128_and(
                f32x4_ge(vdp2, f32x4_mul(vlo, vlo)),
                f32x4_le(vdp2, f32x4_mul(vhi, vhi)),
            );
            star_mask |= i32x4_bitmask(vok) as u8;
        }
        if (ok_streak & 0x30u8) != 0 {
            let vdx = f32x4_sub(
                f32x4(prev_x[j4], prev_x[j5], 0.0, 0.0),
                f32x4_splat(pxi),
            );
            let vdy = f32x4_sub(
                f32x4(prev_y[j4], prev_y[j5], 0.0, 0.0),
                f32x4_splat(pyi),
            );
            let vdp2 = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
            let vlo = f32x4_max(f32x4_sub(v1, vq), vz);
            let vhi = f32x4_add(v1, vq);
            let vok = v128_and(
                f32x4_ge(vdp2, f32x4_mul(vlo, vlo)),
                f32x4_le(vdp2, f32x4_mul(vhi, vhi)),
            );
            star_mask |= ((i32x4_bitmask(vok) as u8) & 3u8) << 4;
        }
        star_mask &= ok_streak;
    }
    (slot_strain, star_mask, persist_mask, has_persist_fwd)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn update_knn_slot_reordered_slow(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    i: usize,
    xi: f32,
    yi: f32,
    best_idx: &[i16; MICRO_KNN_K],
    best_d2: &[f32; MICRO_KNN_K],
    best_cnt: usize,
    old_cnt: usize,
    dtv: u8,
    q: f32,
    min_ref_dist: f32,
) -> (u32, f64, bool) {
    let old_tid = cs.knn[i].tid;
    let old_ref = cs.knn[i].ref_dist;
    let old_strain = cs.knn[i].strain_q15;
    let old_streak = cs.knn[i].rigid_streak;
    let need_a = (dtv & 8u8) != 0;
    let need_b = (dtv & 16u8) != 0;
    let need_dtheta = need_a || need_b;
    let (pxi, pyi) = if need_dtheta {
        (ms.tracks.prev_x[i], ms.tracks.prev_y[i])
    } else {
        (0.0, 0.0)
    };

    let dst = &mut cs.knn[i];
    dst.idx = *best_idx;
    dst.count = best_cnt as u8;
    let mut observed_strain_edges = 0u32;
    let mut slot_strain = 0.0f32;
    let mut star_mask = 0u8;
    let mut persist_mask = 0u8;
    let mut has_persist_fwd = false;

    let dists = sqrt_top6_d2(best_d2);
    let mut k = 0usize;
    while k < best_cnt {
        let j = (best_idx[k] as usize).min(MAX_MICRO - 1);
        let j_tid = ms.tracks.track_id[j];
        let dist = dists[k];
        dst.tid[k] = j_tid;

        let mut matched_m = old_cnt;
        if j_tid != 0 {
            if k < old_cnt && old_tid[k] == j_tid {
                matched_m = k;
            } else {
                matched_m = find_matched_knn_slot(&old_tid, old_cnt, j_tid);
            }
        }

        if matched_m < old_cnt {
            let mm = matched_m.min(MICRO_KNN_K - 1);
            let ref_d = old_ref[mm];
            let denom = ref_d.max(min_ref_dist);
            let inst_strain = (dist - ref_d).abs() / denom;
            let prev_ema = unpack_strain_q15(old_strain[mm]);
            let ema_strain = prev_ema + 0.30 * (inst_strain - prev_ema);
            let q15_new = pack_strain_q15(ema_strain);
            let str_new = if ema_strain < KNN_STRAIN_THRESH && inst_strain < 0.045 {
                old_streak[mm].saturating_add(1)
            } else {
                0
            };
            dst.ref_dist[k] = ref_d;
            dst.strain_q15[k] = q15_new;
            dst.rigid_streak[k] = str_new;
            if str_new >= RIGID_PERSIST_STREAK && q15_new < KNN_STRAIN_THRESH_Q15 {
                persist_mask |= 1u8 << k;
                if (j as isize) > (i as isize) {
                    has_persist_fwd = true;
                }
            }
            observed_strain_edges += 1;
            slot_strain += ema_strain;

            if need_dtheta && str_new > 0 {
                let dxp = ms.tracks.prev_x[j] - pxi;
                let dyp = ms.tracks.prev_y[j] - pyi;
                let dp2 = dxp * dxp + dyp * dyp;
                let lo = (dist - q).max(0.0);
                let hi = dist + q;
                if dp2 >= lo * lo && dp2 <= hi * hi {
                    star_mask |= 1u8 << k;
                }
            }
        } else {
            dst.ref_dist[k] = dist;
            dst.strain_q15[k] = 0;
            dst.rigid_streak[k] = 0;
        }
        k += 1;
    }
    dst.persist_mask = persist_mask;

    finalize_knn_slot_dtheta(
        ms,
        cs,
        i,
        xi,
        yi,
        q,
        dtv,
        need_a,
        need_b,
        star_mask,
        pxi,
        pyi,
        has_persist_fwd,
    );
    (observed_strain_edges, slot_strain as f64, has_persist_fwd)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "wasm32", inline(always))]
#[cfg_attr(not(target_arch = "wasm32"), inline(never))]
fn update_knn_slot_reordered(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    i: usize,
    xi: f32,
    yi: f32,
    best_idx: &[i16; MICRO_KNN_K],
    best_d2: &[f32; MICRO_KNN_K],
    best_cnt: usize,
    old_cnt: usize,
    dtv: u8,
    q: f32,
    min_ref_dist: f32,
) -> (u32, f64, bool) {
    if best_cnt == MICRO_KNN_K && old_cnt == MICRO_KNN_K {
        use crate::simd::*;
        let old_tid = cs.knn[i].tid;
        let old_ref = cs.knn[i].ref_dist;
        let old_strain = cs.knn[i].strain_q15;
        let old_streak = cs.knn[i].rigid_streak;
        let vt03 = i32x4(
            old_tid[0] as i32,
            old_tid[1] as i32,
            old_tid[2] as i32,
            old_tid[3] as i32,
        );
        let ot4 = old_tid[4];
        let ot5 = old_tid[5];

        let need_a = (dtv & 8u8) != 0;
        let need_b = (dtv & 16u8) != 0;
        let need_dtheta = need_a || need_b;
        let (pxi, pyi) = if need_dtheta {
            (ms.tracks.prev_x[i], ms.tracks.prev_y[i])
        } else {
            (0.0, 0.0)
        };

        let j0 = (best_idx[0] as usize).min(MAX_MICRO - 1);
        let j1 = (best_idx[1] as usize).min(MAX_MICRO - 1);
        let j2 = (best_idx[2] as usize).min(MAX_MICRO - 1);
        let j3 = (best_idx[3] as usize).min(MAX_MICRO - 1);
        let j4 = (best_idx[4] as usize).min(MAX_MICRO - 1);
        let j5 = (best_idx[5] as usize).min(MAX_MICRO - 1);
        let tids = &ms.tracks.track_id;
        let jt0 = tids[j0];
        let jt1 = tids[j1];
        let jt2 = tids[j2];
        let jt3 = tids[j3];
        let jt4 = tids[j4];
        let jt5 = tids[j5];

        let dst = &mut cs.knn[i];
        dst.idx = *best_idx;
        dst.count = MICRO_KNN_K as u8;
        dst.tid = [jt0, jt1, jt2, jt3, jt4, jt5];

        let mut observed_strain_edges = 0u32;
        let mut matched_mask = 0u8;
        let mut m_ref = [0.0f32; MICRO_KNN_K];
        let mut m_strain = [0u16; MICRO_KNN_K];
        let mut m_streak = [0u8; MICRO_KNN_K];
        macro_rules! match_slot {
            ($k:expr, $jt:expr) => {{
                let jt = $jt;
                if jt != 0 {
                    let mm = if old_tid[$k] == jt {
                        $k
                    } else {
                        let bm = i32x4_bitmask(i32x4_eq(vt03, i32x4_splat(jt as i32))) as u32;
                        if bm != 0 {
                            (bm.trailing_zeros() as usize).min(3)
                        } else if ot4 == jt {
                            4
                        } else if ot5 == jt {
                            5
                        } else {
                            6
                        }
                    };
                    if mm < MICRO_KNN_K {
                        matched_mask |= 1u8 << $k;
                        let r = old_ref[mm];
                        dst.ref_dist[$k] = r;
                        m_ref[$k] = r;
                        m_strain[$k] = old_strain[mm];
                        m_streak[$k] = old_streak[mm];
                        observed_strain_edges += 1;
                    }
                }
            }};
        }
        match_slot!(0, jt0);
        match_slot!(1, jt1);
        match_slot!(2, jt2);
        match_slot!(3, jt3);
        match_slot!(4, jt4);
        match_slot!(5, jt5);

        let vref0_m = f32x4(m_ref[0], m_ref[1], m_ref[2], m_ref[3]);
        let vref1_m = f32x4(m_ref[4], m_ref[5], 1.0, 1.0);
        let (slot_strain, star_mask, persist_mask, has_persist_fwd) = eval_knn_edges_6_wasm(
            ms,
            dst,
            i,
            best_d2[0],
            best_d2[1],
            best_d2[2],
            best_d2[3],
            best_d2[4],
            best_d2[5],
            j0,
            j1,
            j2,
            j3,
            j4,
            j5,
            min_ref_dist,
            q,
            need_dtheta,
            pxi,
            pyi,
            matched_mask,
            vref0_m,
            vref1_m,
            m_strain,
            m_streak,
        );
        dst.persist_mask = persist_mask;

        finalize_knn_slot_dtheta(
            ms,
            cs,
            i,
            xi,
            yi,
            q,
            dtv,
            need_a,
            need_b,
            star_mask,
            pxi,
            pyi,
            has_persist_fwd,
        );
        return (observed_strain_edges, slot_strain as f64, has_persist_fwd);
    }

    update_knn_slot_reordered_slow(
        ms,
        cs,
        i,
        xi,
        yi,
        best_idx,
        best_d2,
        best_cnt,
        old_cnt,
        dtv,
        q,
        min_ref_dist,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn update_knn_slot_slow(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    i: usize,
    xi: f32,
    yi: f32,
    p0: u64,
    p1: u64,
    p2: u64,
    p3: u64,
    p4: u64,
    p5: u64,
    best_cnt: usize,
    old_cnt: usize,
    dtv: u8,
    q: f32,
    min_ref_dist: f32,
    init_only: bool,
) -> (u32, f64, bool) {
    let best_idx = [
        (p0 as u16) as i16,
        (p1 as u16) as i16,
        (p2 as u16) as i16,
        (p3 as u16) as i16,
        (p4 as u16) as i16,
        (p5 as u16) as i16,
    ];
    let best_d2 = [
        f32::from_bits((p0 >> 16) as u32),
        f32::from_bits((p1 >> 16) as u32),
        f32::from_bits((p2 >> 16) as u32),
        f32::from_bits((p3 >> 16) as u32),
        f32::from_bits((p4 >> 16) as u32),
        f32::from_bits((p5 >> 16) as u32),
    ];
    if init_only {
        update_knn_slot_init(ms, cs, i, &best_idx, &best_d2, best_cnt, dtv);
        return (0, 0.0, false);
    }
    if old_cnt != best_cnt {
        return update_knn_slot_reordered(
            ms,
            cs,
            i,
            xi,
            yi,
            &best_idx,
            &best_d2,
            best_cnt,
            old_cnt,
            dtv,
            q,
            min_ref_dist,
        );
    }
    {
        let old_tid = &cs.knn[i].tid;
        let mut all_same = true;
        let mut k = 0usize;
        while k < best_cnt {
            let j = (best_idx[k] as usize).min(MAX_MICRO - 1);
            let j_tid = ms.tracks.track_id[j];
            if j_tid == 0 || j_tid != old_tid[k] {
                all_same = false;
                break;
            }
            k += 1;
        }
        if !all_same {
            return update_knn_slot_reordered(
                ms,
                cs,
                i,
                xi,
                yi,
                &best_idx,
                &best_d2,
                best_cnt,
                old_cnt,
                dtv,
                q,
                min_ref_dist,
            );
        }
    }

    let need_a = (dtv & 8u8) != 0;
    let need_b = (dtv & 16u8) != 0;
    let need_dtheta = need_a || need_b;
    let (pxi, pyi) = if need_dtheta {
        (ms.tracks.prev_x[i], ms.tracks.prev_y[i])
    } else {
        (0.0, 0.0)
    };

    let mut slot_strain = 0.0f32;
    let mut star_mask = 0u8;
    let mut persist_mask = 0u8;
    let mut has_persist_fwd = false;
    {
        let dst = &mut cs.knn[i];
        dst.idx = best_idx;
        let dists = sqrt_top6_d2(&best_d2);
        let mut k = 0usize;
        while k < best_cnt {
            let dist = dists[k];
            let ref_d = dst.ref_dist[k];
            let denom = ref_d.max(min_ref_dist);
            let inst_strain = (dist - ref_d).abs() / denom;
            let prev_ema = unpack_strain_q15(dst.strain_q15[k]);
            let ema_strain = prev_ema + 0.30 * (inst_strain - prev_ema);
            let q15_new = pack_strain_q15(ema_strain);
            let str_new = if ema_strain < KNN_STRAIN_THRESH && inst_strain < 0.045 {
                dst.rigid_streak[k].saturating_add(1)
            } else {
                0
            };
            dst.strain_q15[k] = q15_new;
            dst.rigid_streak[k] = str_new;
            let j = (best_idx[k] as usize).min(MAX_MICRO - 1);
            if str_new >= RIGID_PERSIST_STREAK && q15_new < KNN_STRAIN_THRESH_Q15 {
                persist_mask |= 1u8 << k;
                if (j as isize) > (i as isize) {
                    has_persist_fwd = true;
                }
            }
            slot_strain += ema_strain;
            if need_dtheta && str_new > 0 {
                let dxp = ms.tracks.prev_x[j] - pxi;
                let dyp = ms.tracks.prev_y[j] - pyi;
                let dp2 = dxp * dxp + dyp * dyp;
                let lo = (dist - q).max(0.0);
                let hi = dist + q;
                if dp2 >= lo * lo && dp2 <= hi * hi {
                    star_mask |= 1u8 << k;
                }
            }
            k += 1;
        }
        dst.persist_mask = persist_mask;
    }

    finalize_knn_slot_dtheta(
        ms,
        cs,
        i,
        xi,
        yi,
        q,
        dtv,
        need_a,
        need_b,
        star_mask,
        pxi,
        pyi,
        has_persist_fwd,
    );
    (best_cnt as u32, slot_strain as f64, has_persist_fwd)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn update_knn_slot_edges_same6_wasm(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    i: usize,
    xi: f32,
    yi: f32,
    j0: usize,
    j1: usize,
    j2: usize,
    j3: usize,
    j4: usize,
    j5: usize,
    d2_0: f32,
    d2_1: f32,
    d2_2: f32,
    d2_3: f32,
    d2_4: f32,
    d2_5: f32,
    q: f32,
    min_ref_dist: f32,
) -> (u32, f64, bool) {
    let dtv = cs.slot_dtheta_valid[i];
    let need_a = (dtv & 8u8) != 0;
    let need_b = (dtv & 16u8) != 0;
    let need_dtheta = need_a || need_b;
    let (pxi, pyi) = if need_dtheta {
        (ms.tracks.prev_x[i], ms.tracks.prev_y[i])
    } else {
        (0.0, 0.0)
    };
    let (slot_strain, star_mask, has_persist_fwd) = {
        use crate::simd::f32x4;
        let dst = &mut cs.knn[i];
        let vref0_cur =
            f32x4(dst.ref_dist[0], dst.ref_dist[1], dst.ref_dist[2], dst.ref_dist[3]);
        let vref1_cur = f32x4(dst.ref_dist[4], dst.ref_dist[5], 1.0, 1.0);
        let prev_strain = dst.strain_q15;
        let prev_streak = dst.rigid_streak;
        let (ss, sm, pm, hpf) = eval_knn_edges_6_wasm(
            ms,
            dst,
            i,
            d2_0,
            d2_1,
            d2_2,
            d2_3,
            d2_4,
            d2_5,
            j0,
            j1,
            j2,
            j3,
            j4,
            j5,
            min_ref_dist,
            q,
            need_dtheta,
            pxi,
            pyi,
            0x3Fu8,
            vref0_cur,
            vref1_cur,
            prev_strain,
            prev_streak,
        );
        dst.persist_mask = pm;
        (ss, sm, hpf)
    };
    finalize_knn_slot_dtheta(
        ms,
        cs,
        i,
        xi,
        yi,
        q,
        dtv,
        need_a,
        need_b,
        star_mask,
        pxi,
        pyi,
        has_persist_fwd,
    );
    (MICRO_KNN_K as u32, slot_strain as f64, has_persist_fwd)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn update_knn_slot_edges(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    i: usize,
    xi: f32,
    yi: f32,
    p0: u64,
    p1: u64,
    p2: u64,
    p3: u64,
    p4: u64,
    p5: u64,
    best_cnt: usize,
    frame_idx: u32,
    q: f32,
    min_ref_dist: f32,
) -> (u32, f64, bool) {
    let _ = frame_idx;
    let best_cnt = best_cnt.min(MICRO_KNN_K);
    let old_cnt = (cs.knn[i].count as usize).min(MICRO_KNN_K);
    let dtv = cs.slot_dtheta_valid[i];
    if old_cnt == 0 {
        let best_idx = [
            (p0 as u16) as i16,
            (p1 as u16) as i16,
            (p2 as u16) as i16,
            (p3 as u16) as i16,
            (p4 as u16) as i16,
            (p5 as u16) as i16,
        ];
        let best_d2 = [
            f32::from_bits((p0 >> 16) as u32),
            f32::from_bits((p1 >> 16) as u32),
            f32::from_bits((p2 >> 16) as u32),
            f32::from_bits((p3 >> 16) as u32),
            f32::from_bits((p4 >> 16) as u32),
            f32::from_bits((p5 >> 16) as u32),
        ];
        update_knn_slot_init(ms, cs, i, &best_idx, &best_d2, best_cnt, dtv);
        return (0, 0.0, false);
    }

    if old_cnt == MICRO_KNN_K && best_cnt == MICRO_KNN_K {
        let j0 = ((p0 as u16) as usize).min(MAX_MICRO - 1);
        let j1 = ((p1 as u16) as usize).min(MAX_MICRO - 1);
        let j2 = ((p2 as u16) as usize).min(MAX_MICRO - 1);
        let j3 = ((p3 as u16) as usize).min(MAX_MICRO - 1);
        let j4 = ((p4 as u16) as usize).min(MAX_MICRO - 1);
        let j5 = ((p5 as u16) as usize).min(MAX_MICRO - 1);
        let all_same = {
            let old_tid = &cs.knn[i].tid;
            let t0 = ms.tracks.track_id[j0];
            t0 != 0
                && t0 == old_tid[0]
                && {
                    let t1 = ms.tracks.track_id[j1];
                    t1 != 0 && t1 == old_tid[1]
                }
                && {
                    let t2 = ms.tracks.track_id[j2];
                    t2 != 0 && t2 == old_tid[2]
                }
                && {
                    let t3 = ms.tracks.track_id[j3];
                    t3 != 0 && t3 == old_tid[3]
                }
                && {
                    let t4 = ms.tracks.track_id[j4];
                    t4 != 0 && t4 == old_tid[4]
                }
                && {
                    let t5 = ms.tracks.track_id[j5];
                    t5 != 0 && t5 == old_tid[5]
                }
        };
        if all_same {
            let need_a = (dtv & 8u8) != 0;
            let need_b = (dtv & 16u8) != 0;
            let need_dtheta = need_a || need_b;
            let (pxi, pyi) = if need_dtheta {
                (ms.tracks.prev_x[i], ms.tracks.prev_y[i])
            } else {
                (0.0, 0.0)
            };
            let (slot_strain, star_mask, has_persist_fwd) = {
                use crate::simd::f32x4;
                let dst = &mut cs.knn[i];
                dst.idx = [
                    (p0 as u16) as i16,
                    (p1 as u16) as i16,
                    (p2 as u16) as i16,
                    (p3 as u16) as i16,
                    (p4 as u16) as i16,
                    (p5 as u16) as i16,
                ];
                let vref0_cur =
                    f32x4(dst.ref_dist[0], dst.ref_dist[1], dst.ref_dist[2], dst.ref_dist[3]);
                let vref1_cur = f32x4(dst.ref_dist[4], dst.ref_dist[5], 1.0, 1.0);
                let prev_strain = dst.strain_q15;
                let prev_streak = dst.rigid_streak;
                let (ss, sm, pm, hpf) = eval_knn_edges_6_wasm(
                    ms,
                    dst,
                    i,
                    f32::from_bits((p0 >> 16) as u32),
                    f32::from_bits((p1 >> 16) as u32),
                    f32::from_bits((p2 >> 16) as u32),
                    f32::from_bits((p3 >> 16) as u32),
                    f32::from_bits((p4 >> 16) as u32),
                    f32::from_bits((p5 >> 16) as u32),
                    j0,
                    j1,
                    j2,
                    j3,
                    j4,
                    j5,
                    min_ref_dist,
                    q,
                    need_dtheta,
                    pxi,
                    pyi,
                    0x3Fu8,
                    vref0_cur,
                    vref1_cur,
                    prev_strain,
                    prev_streak,
                );
                dst.persist_mask = pm;
                (ss, sm, hpf)
            };
            finalize_knn_slot_dtheta(
                ms,
                cs,
                i,
                xi,
                yi,
                q,
                dtv,
                need_a,
                need_b,
                star_mask,
                pxi,
                pyi,
                has_persist_fwd,
            );
            return (MICRO_KNN_K as u32, slot_strain as f64, has_persist_fwd);
        }
        let best_idx = [
            (p0 as u16) as i16,
            (p1 as u16) as i16,
            (p2 as u16) as i16,
            (p3 as u16) as i16,
            (p4 as u16) as i16,
            (p5 as u16) as i16,
        ];
        let best_d2 = [
            f32::from_bits((p0 >> 16) as u32),
            f32::from_bits((p1 >> 16) as u32),
            f32::from_bits((p2 >> 16) as u32),
            f32::from_bits((p3 >> 16) as u32),
            f32::from_bits((p4 >> 16) as u32),
            f32::from_bits((p5 >> 16) as u32),
        ];
        return update_knn_slot_reordered(
            ms,
            cs,
            i,
            xi,
            yi,
            &best_idx,
            &best_d2,
            MICRO_KNN_K,
            MICRO_KNN_K,
            dtv,
            q,
            min_ref_dist,
        );
    }

    update_knn_slot_slow(
        ms, cs, i, xi, yi, p0, p1, p2, p3, p4, p5, best_cnt, old_cnt, dtv, q, min_ref_dist,
        false,
    )
}

/// Solves proper `det = +1` 2D Procrustes (`SO(2)` rotation + translation) over masked members.
/// Uses rotational spread / RMS radius (`sum_r2 >= 180*q^2`, `trace_cov >= 36*q^2`) so observable
/// slender bars pass while tiny collapsed clumps fail (`M2`). Returns `(ok, ..., rmse, cnt, sum_r2)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn finalize_kabsch_2d(
    cnt: usize,
    x0_r: f32,
    y0_r: f32,
    xt_r: f32,
    yt_r: f32,
    sx0: f64,
    sy0: f64,
    sxt: f64,
    syt: f64,
    sx02: f64,
    sx0y0: f64,
    sy02: f64,
    st2: f64,
    spu: f64,
    sqv: f64,
    squ: f64,
    spv: f64,
    q: f32,
) -> (bool, f32, f32, f32, f32, f32, f32, f32, usize, f32) {
    if cnt < MIN_RIGID_MEMBERS {
        return (false, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, cnt, 0.0);
    }

    let n_f64 = cnt as f64;
    let inv_n = 1.0 / n_f64;
    let mx0 = sx0 * inv_n;
    let my0 = sy0 * inv_n;
    let mxt = sxt * inv_n;
    let myt = syt * inv_n;
    let x0_bar = x0_r + (mx0 as f32);
    let y0_bar = y0_r + (my0 as f32);
    let xt_bar = xt_r + (mxt as f32);
    let yt_bar = yt_r + (myt as f32);

    let gxx = sx02 - sx0 * mx0;
    let gxy = sx0y0 - sx0 * my0;
    let gyy = sy02 - sy0 * my0;
    let cov_xx = (gxx * inv_n) as f32;
    let cov_xy = (gxy * inv_n) as f32;
    let cov_yy = (gyy * inv_n) as f32;
    let trace_cov = cov_xx + cov_yy;
    let sum_r2_0 = gxx + gyy;
    let sum_r2 = sum_r2_0 as f32;
    let (lmin, lmax) = sym2x2_eigenvalues(cov_xx, cov_xy, cov_yy);
    if !lmin.is_finite()
        || !lmax.is_finite()
        || !sum_r2.is_finite()
        || lmax <= 1e-6
        || trace_cov < 36.0 * q * q
        || sum_r2 < 180.0 * q * q
    {
        return (false, x0_bar, y0_bar, xt_bar, yt_bar, 1.0, 0.0, 0.0, cnt, sum_r2);
    }

    let s_cross = (squ - syt * mx0) - (spv - sxt * my0);
    let c_dot = (spu - sxt * mx0) + (sqv - syt * my0);
    let norm = wasm_sqrt_f64(s_cross * s_cross + c_dot * c_dot);
    if norm < 1e-6 {
        return (false, x0_bar, y0_bar, xt_bar, yt_bar, 1.0, 0.0, 0.0, cnt, sum_r2);
    }
    let cos_t = (c_dot / norm) as f32;
    let sin_t = (s_cross / norm) as f32;

    let sum_r2_t = st2 - (sxt * mxt + syt * myt);
    let sq_err = (sum_r2_t + sum_r2_0 - 2.0 * norm).max(0.0);
    let rmse = wasm_sqrtf((sq_err * inv_n) as f32);
    (
        true, x0_bar, y0_bar, xt_bar, yt_bar, cos_t, sin_t, rmse, cnt, sum_r2,
    )
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn solve_kabsch_2d_chains(
    x0_arr: &[f32; MAX_MICRO],
    y0_arr: &[f32; MAX_MICRO],
    xt_arr: &[f32; MAX_MICRO],
    yt_arr: &[f32; MAX_MICRO],
    comp_next: &[i16; MAX_MICRO],
    mask: &[u8; MAX_MICRO],
    head_a: i16,
    mask_a: u8,
    head_b: i16,
    mask_b: u8,
    q: f32,
) -> (bool, f32, f32, f32, f32, f32, f32, f32, usize, f32) {
    let mut cnt = 0usize;
    let mut x0_r = 0.0f32;
    let mut y0_r = 0.0f32;
    let mut xt_r = 0.0f32;
    let mut yt_r = 0.0f32;
    let mut sx0 = 0.0f64;
    let mut sy0 = 0.0f64;
    let mut sxt = 0.0f64;
    let mut syt = 0.0f64;
    let mut sx02 = 0.0f64;
    let mut sx0y0 = 0.0f64;
    let mut sy02 = 0.0f64;
    let mut st2 = 0.0f64;
    let mut spu = 0.0f64;
    let mut sqv = 0.0f64;
    let mut squ = 0.0f64;
    let mut spv = 0.0f64;

    let chains = [(head_a, mask_a), (head_b, mask_b)];
    let mut ci = 0usize;
    while ci < 2 {
        let (mut cur, target_mask) = chains[ci];
        ci += 1;
        while cur >= 0 {
            let ti = (cur as usize).min(MAX_MICRO - 1);
            if mask[ti] == target_mask {
                let x0 = x0_arr[ti];
                let y0 = y0_arr[ti];
                let xt = xt_arr[ti];
                let yt = yt_arr[ti];
                if cnt == 0 {
                    x0_r = x0;
                    y0_r = y0;
                    xt_r = xt;
                    yt_r = yt;
                }
                let u = (x0 - x0_r) as f64;
                let v = (y0 - y0_r) as f64;
                let px = (xt - xt_r) as f64;
                let qy = (yt - yt_r) as f64;
                sx0 += u;
                sy0 += v;
                sxt += px;
                syt += qy;
                sx02 += u * u;
                sx0y0 += u * v;
                sy02 += v * v;
                st2 += px * px + qy * qy;
                spu += px * u;
                sqv += qy * v;
                squ += qy * u;
                spv += px * v;
                cnt += 1;
            }
            cur = comp_next[ti];
        }
    }
    finalize_kabsch_2d(
        cnt, x0_r, y0_r, xt_r, yt_r, sx0, sy0, sxt, syt, sx02, sx0y0, sy02, st2, spu, sqv, squ,
        spv, q,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn solve_kabsch_2d_masked(
    x0_arr: &[f32; MAX_MICRO],
    y0_arr: &[f32; MAX_MICRO],
    xt_arr: &[f32; MAX_MICRO],
    yt_arr: &[f32; MAX_MICRO],
    comp_next: &[i16; MAX_MICRO],
    mask: &[u8; MAX_MICRO],
    head: i16,
    target_mask: u8,
    q: f32,
) -> (bool, f32, f32, f32, f32, f32, f32, f32, usize, f32) {
    solve_kabsch_2d_chains(
        x0_arr,
        y0_arr,
        xt_arr,
        yt_arr,
        comp_next,
        mask,
        head,
        target_mask,
        -1,
        0,
        q,
    )
}

fn solve_kabsch_2d_slice(
    x0_arr: &[f32; MAX_MICRO],
    y0_arr: &[f32; MAX_MICRO],
    xt_arr: &[f32; MAX_MICRO],
    yt_arr: &[f32; MAX_MICRO],
    idxs: &[u16],
    q: f32,
) -> (bool, f32, f32, f32, f32, f32, f32, f32, usize, f32) {
    let cnt = idxs.len();
    if cnt < MIN_RIGID_MEMBERS {
        return (false, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, cnt, 0.0);
    }
    let ti0 = (idxs[0] as usize).min(MAX_MICRO - 1);
    let x0_r = x0_arr[ti0];
    let y0_r = y0_arr[ti0];
    let xt_r = xt_arr[ti0];
    let yt_r = yt_arr[ti0];
    let mut sx0 = 0.0f64;
    let mut sy0 = 0.0f64;
    let mut sxt = 0.0f64;
    let mut syt = 0.0f64;
    let mut sx02 = 0.0f64;
    let mut sx0y0 = 0.0f64;
    let mut sy02 = 0.0f64;
    let mut st2 = 0.0f64;
    let mut spu = 0.0f64;
    let mut sqv = 0.0f64;
    let mut squ = 0.0f64;
    let mut spv = 0.0f64;

    let mut i = 1usize;
    while i < cnt {
        let ti = (idxs[i] as usize).min(MAX_MICRO - 1);
        let u = (x0_arr[ti] - x0_r) as f64;
        let v = (y0_arr[ti] - y0_r) as f64;
        let px = (xt_arr[ti] - xt_r) as f64;
        let qy = (yt_arr[ti] - yt_r) as f64;
        sx0 += u;
        sy0 += v;
        sxt += px;
        syt += qy;
        sx02 += u * u;
        sx0y0 += u * v;
        sy02 += v * v;
        st2 += px * px + qy * qy;
        spu += px * u;
        sqv += qy * v;
        squ += qy * u;
        spv += px * v;
        i += 1;
    }
    finalize_kabsch_2d(
        cnt, x0_r, y0_r, xt_r, yt_r, sx0, sy0, sxt, syt, sx02, sx0y0, sy02, st2, spu, sqv, squ,
        spv, q,
    )
}

#[inline(always)]
fn has_reciprocal_persistent_edge(ms: &MicroSwarmState, cs: &ContinuumState, i: usize, j: usize, q: f32) -> bool {
    let slot_j = &cs.knn[j];
    let mut pm = slot_j.persist_mask;
    while pm != 0 {
        let k = (pm.trailing_zeros() as usize).min(MICRO_KNN_K - 1);
        pm &= pm - 1;
        if slot_j.idx[k] == i as i16 {
            let dx_t = ms.tracks.pos_x[j] - ms.tracks.pos_x[i];
            let dy_t = ms.tracks.pos_y[j] - ms.tracks.pos_y[i];
            let dx_p = ms.tracks.prev_x[j] - ms.tracks.prev_x[i];
            let dy_p = ms.tracks.prev_y[j] - ms.tracks.prev_y[i];
            let d_t2 = dx_t * dx_t + dy_t * dy_t;
            let d_p2 = dx_p * dx_p + dy_p * dy_p;
            let eps = 1.25 * q;
            let s = d_t2 + d_p2 - eps * eps;
            return s <= 0.0 || s * s <= 4.0 * d_t2 * d_p2;
        }
    }
    false
}

#[inline(always)]
fn has_rigid_edge_support(cs: &ContinuumState, i: usize, j: usize) -> bool {
    let slot_j = &cs.knn[j];
    let mut pm_j = slot_j.persist_mask;
    if pm_j.count_ones() < 3 {
        return true;
    }
    let slot_i = &cs.knn[i];
    let mut pm_i = slot_i.persist_mask;
    if pm_i.count_ones() < 3 {
        return true;
    }
    let mut j_rig = [0i16; MICRO_KNN_K];
    let mut deg_j = 0usize;
    while pm_j != 0 {
        let b = (pm_j.trailing_zeros() as usize).min(MICRO_KNN_K - 1);
        pm_j &= pm_j - 1;
        let nb = slot_j.idx[b];
        if nb >= 0 && deg_j < MICRO_KNN_K {
            j_rig[deg_j] = nb;
            deg_j += 1;
        }
    }
    if deg_j < 3 {
        return true;
    }
    let mut deg_i = 0usize;
    while pm_i != 0 {
        let a = (pm_i.trailing_zeros() as usize).min(MICRO_KNN_K - 1);
        pm_i &= pm_i - 1;
        let nb = slot_i.idx[a];
        if nb >= 0 {
            deg_i += 1;
            let mut c = 0usize;
            while c < deg_j.min(MICRO_KNN_K) {
                if j_rig[c] == nb {
                    return true;
                }
                c += 1;
            }
        }
    }
    deg_i < 3
}

#[derive(Copy, Clone)]
struct MultiFrameFit {
    ok: bool,
    omega: f32,
    bank: u8,
    af: u32,
    ax0: f32,
    ay0: f32,
    axt: f32,
    ayt: f32,
    cos_m: f32,
    sin_m: f32,
}

impl MultiFrameFit {
    #[inline(always)]
    fn empty() -> Self {
        Self {
            ok: false,
            omega: 0.0,
            bank: 0,
            af: FTLE_INVALID_FRAME,
            ax0: 0.0,
            ay0: 0.0,
            axt: 0.0,
            ayt: 0.0,
            cos_m: 1.0,
            sin_m: 0.0,
        }
    }
}

#[inline(always)]
fn matches_multi_frame_fit(
    ms: &MicroSwarmState,
    cs: &ContinuumState,
    ti: usize,
    fit: &MultiFrameFit,
    px: f32,
    py: f32,
    lim2: f32,
) -> bool {
    if !fit.ok {
        return false;
    }
    let (af, ax0, ay0) = if fit.bank == 1 {
        (
            ms.tracks.ftle_a_frame[ti],
            ms.tracks.ftle_a_x0[ti],
            ms.tracks.ftle_a_y0[ti],
        )
    } else {
        (
            ms.tracks.ftle_b_frame[ti],
            ms.tracks.ftle_b_x0[ti],
            ms.tracks.ftle_b_y0[ti],
        )
    };
    if af != fit.af || af < cs.slot_clean_since_frame[ti] {
        return false;
    }
    let ux = ax0 - fit.ax0;
    let uy = ay0 - fit.ay0;
    let pred_x = fit.axt + (fit.cos_m * ux - fit.sin_m * uy);
    let pred_y = fit.ayt + (fit.sin_m * ux + fit.cos_m * uy);
    let rx = px - pred_x;
    let ry = py - pred_y;
    rx * rx + ry * ry <= lim2
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn eval_multi_frame_se2_chains(
    ms: &MicroSwarmState,
    cs: &ContinuumState,
    head_a: i16,
    mask_a: u8,
    head_b: i16,
    mask_b: u8,
    kept_slice: Option<&[u16]>,
    frame_idx: u32,
    q: f32,
    need_alt_bank: bool,
) -> (MultiFrameFit, MultiFrameFit) {
    let mut primary = MultiFrameFit::empty();
    let mut secondary = MultiFrameFit::empty();
    let max_multi_rmse = 1.65 * q;

    let mut total_members = 0usize;
    let mut tested_a_afs = [u32::MAX; 4];
    let mut tested_a_cnts = [0u16; 4];
    let mut tested_a_ages = [0u32; 4];
    let mut tested_a_n = 0usize;
    let mut tested_b_afs = [u32::MAX; 4];
    let mut tested_b_cnts = [0u16; 4];
    let mut tested_b_ages = [0u32; 4];
    let mut tested_b_n = 0usize;

    #[inline(always)]
    fn tally_anchor(
        af: u32,
        cs_f: u32,
        frame_idx: u32,
        afs: &mut [u32; 4],
        cnts: &mut [u16; 4],
        ages: &mut [u32; 4],
        n: &mut usize,
    ) {
        if af != FTLE_INVALID_FRAME && af >= cs_f && frame_idx > af {
            let age = frame_idx - af;
            if (8..=20).contains(&age) {
                let tn = (*n).min(4);
                let mut z = 0usize;
                while z < tn {
                    if afs[z] == af {
                        cnts[z] = cnts[z].saturating_add(1);
                        return;
                    }
                    z += 1;
                }
                if tn < 4 {
                    afs[tn] = af;
                    cnts[tn] = 1;
                    ages[tn] = age;
                    *n = tn + 1;
                }
            }
        }
    }

    if let Some(idxs) = kept_slice {
        total_members = idxs.len();
        if total_members < MIN_RIGID_MEMBERS {
            return (primary, secondary);
        }
        let mut ki = 0usize;
        while ki < idxs.len() {
            let ti = (idxs[ki] as usize).min(MAX_MICRO - 1);
            let cs_f = cs.slot_clean_since_frame[ti];
            tally_anchor(
                ms.tracks.ftle_a_frame[ti],
                cs_f,
                frame_idx,
                &mut tested_a_afs,
                &mut tested_a_cnts,
                &mut tested_a_ages,
                &mut tested_a_n,
            );
            tally_anchor(
                ms.tracks.ftle_b_frame[ti],
                cs_f,
                frame_idx,
                &mut tested_b_afs,
                &mut tested_b_cnts,
                &mut tested_b_ages,
                &mut tested_b_n,
            );
            ki += 1;
        }
    } else {
        let chains = [(head_a, mask_a), (head_b, mask_b)];
        let mut c_i = 0usize;
        while c_i < 2 {
            let (mut cur, tm) = chains[c_i];
            c_i += 1;
            while cur >= 0 {
                let ti = (cur as usize).min(MAX_MICRO - 1);
                if cs.member_mask[ti] == tm {
                    total_members += 1;
                    let cs_f = cs.slot_clean_since_frame[ti];
                    tally_anchor(
                        ms.tracks.ftle_a_frame[ti],
                        cs_f,
                        frame_idx,
                        &mut tested_a_afs,
                        &mut tested_a_cnts,
                        &mut tested_a_ages,
                        &mut tested_a_n,
                    );
                    tally_anchor(
                        ms.tracks.ftle_b_frame[ti],
                        cs_f,
                        frame_idx,
                        &mut tested_b_afs,
                        &mut tested_b_cnts,
                        &mut tested_b_ages,
                        &mut tested_b_n,
                    );
                }
                cur = cs.comp_next[ti];
            }
        }
        if total_members < MIN_RIGID_MEMBERS {
            return (primary, secondary);
        }
    }

    let mut cand_banks = [0u8; 4];
    let mut cand_afs = [u32::MAX; 4];
    let mut cand_cnts = [0usize; 4];
    let mut cand_scores = [0usize; 4];
    let mut n_cand = 0usize;

    let mut pass_bank = 1u8;
    while pass_bank <= 2 {
        let (tested_n, tested_cnts, tested_ages, tested_afs) = if pass_bank == 1 {
            (tested_a_n, &tested_a_cnts, &tested_a_ages, &tested_a_afs)
        } else {
            (tested_b_n, &tested_b_cnts, &tested_b_ages, &tested_b_afs)
        };
        let tn = tested_n.min(4);
        let mut z = 0usize;
        while z < tn {
            let zi = z & 3;
            let match_cnt = tested_cnts[zi] as usize;
            if match_cnt >= MIN_RIGID_MEMBERS && match_cnt * 10 >= total_members * 7 {
                let age = tested_ages[zi];
                let af = tested_afs[zi];
                let score = match_cnt * 2 + usize::from(age <= 15);
                let mut p = if n_cand < 4 {
                    n_cand
                } else if score <= cand_scores[3] {
                    z += 1;
                    continue;
                } else {
                    3
                };
                while p > 0 {
                    let prev = (p - 1) & 3;
                    if score <= cand_scores[prev] {
                        break;
                    }
                    let cur_p = p & 3;
                    cand_banks[cur_p] = cand_banks[prev];
                    cand_afs[cur_p] = cand_afs[prev];
                    cand_cnts[cur_p] = cand_cnts[prev];
                    cand_scores[cur_p] = cand_scores[prev];
                    p -= 1;
                }
                let ins_p = p & 3;
                cand_banks[ins_p] = pass_bank;
                cand_afs[ins_p] = af;
                cand_cnts[ins_p] = match_cnt;
                cand_scores[ins_p] = score;
                if n_cand < 4 {
                    n_cand += 1;
                }
            }
            z += 1;
        }
        pass_bank += 1;
    }

    let mut ci = 0usize;
    let n_cand_lim = n_cand.min(4);
    while ci < n_cand_lim {
        let ci_i = ci & 3;
        let bank_id = cand_banks[ci_i];
        let af_sel = cand_afs[ci_i];
        let match_cnt = cand_cnts[ci_i];
        ci += 1;

        if primary.ok && (!need_alt_bank || bank_id == primary.bank || secondary.ok) {
            continue;
        }

        let (ok_t, t_multi) = cs.elapsed_since_anchor(af_sel, frame_idx);
        if !ok_t {
            continue;
        }

        let mut ref_set = false;
        let mut ax0_r = 0.0f32;
        let mut ay0_r = 0.0f32;
        let mut axt_r = 0.0f32;
        let mut ayt_r = 0.0f32;
        let mut sx0 = 0.0f64;
        let mut sy0 = 0.0f64;
        let mut sxt = 0.0f64;
        let mut syt = 0.0f64;
        let mut sx02_y02 = 0.0f64;
        let mut st2 = 0.0f64;
        let mut spu = 0.0f64;
        let mut sqv = 0.0f64;
        let mut squ = 0.0f64;
        let mut spv = 0.0f64;
        let mut pred_dtheta_sum = 0.0f64;
        let mut pred_dtheta_cnt = 0usize;
        let bank_bit = if bank_id == 1 { 1u8 } else { 2u8 };
        let (af_s, ax0_s, ay0_s, dth_s) = if bank_id == 1 {
            (
                &ms.tracks.ftle_a_frame,
                &ms.tracks.ftle_a_x0,
                &ms.tracks.ftle_a_y0,
                &cs.slot_dtheta_a,
            )
        } else {
            (
                &ms.tracks.ftle_b_frame,
                &ms.tracks.ftle_b_x0,
                &ms.tracks.ftle_b_y0,
                &cs.slot_dtheta_b,
            )
        };

        if let Some(idxs) = kept_slice {
            let mut ki = 0usize;
            while ki < idxs.len() {
                let ti = (idxs[ki] as usize).min(MAX_MICRO - 1);
                let af = af_s[ti];
                if af == af_sel && af >= cs.slot_clean_since_frame[ti] {
                    let ax0 = ax0_s[ti];
                    let ay0 = ay0_s[ti];
                    let dth = dth_s[ti];
                    let xt = ms.tracks.pos_x[ti];
                    let yt = ms.tracks.pos_y[ti];
                    if !ref_set {
                        ax0_r = ax0;
                        ay0_r = ay0;
                        axt_r = xt;
                        ayt_r = yt;
                        ref_set = true;
                    }
                    let u = (ax0 - ax0_r) as f64;
                    let v = (ay0 - ay0_r) as f64;
                    let px = (xt - axt_r) as f64;
                    let qy = (yt - ayt_r) as f64;
                    sx0 += u;
                    sy0 += v;
                    sxt += px;
                    syt += qy;
                    sx02_y02 += u * u + v * v;
                    st2 += px * px + qy * qy;
                    spu += px * u;
                    sqv += qy * v;
                    squ += qy * u;
                    spv += px * v;
                    if (cs.slot_dtheta_valid[ti] & bank_bit) != 0 && dth.is_finite() {
                        pred_dtheta_sum += dth as f64;
                        pred_dtheta_cnt += 1;
                    }
                }
                ki += 1;
            }
        } else {
            let chains = [(head_a, mask_a), (head_b, mask_b)];
            let mut c_i = 0usize;
            while c_i < 2 {
                let (mut cur, tm) = chains[c_i];
                c_i += 1;
                while cur >= 0 {
                    let ti = (cur as usize).min(MAX_MICRO - 1);
                    if cs.member_mask[ti] == tm {
                        let af = af_s[ti];
                        if af == af_sel && af >= cs.slot_clean_since_frame[ti] {
                            let ax0 = ax0_s[ti];
                            let ay0 = ay0_s[ti];
                            let dth = dth_s[ti];
                            let xt = ms.tracks.pos_x[ti];
                            let yt = ms.tracks.pos_y[ti];
                            if !ref_set {
                                ax0_r = ax0;
                                ay0_r = ay0;
                                axt_r = xt;
                                ayt_r = yt;
                                ref_set = true;
                            }
                            let u = (ax0 - ax0_r) as f64;
                            let v = (ay0 - ay0_r) as f64;
                            let px = (xt - axt_r) as f64;
                            let qy = (yt - ayt_r) as f64;
                            sx0 += u;
                            sy0 += v;
                            sxt += px;
                            syt += qy;
                            sx02_y02 += u * u + v * v;
                            st2 += px * px + qy * qy;
                            spu += px * u;
                            sqv += qy * v;
                            squ += qy * u;
                            spv += px * v;
                            if (cs.slot_dtheta_valid[ti] & bank_bit) != 0 && dth.is_finite() {
                                pred_dtheta_sum += dth as f64;
                                pred_dtheta_cnt += 1;
                            }
                        }
                    }
                    cur = cs.comp_next[ti];
                }
            }
        }
        if !ref_set || pred_dtheta_cnt < MIN_RIGID_MEMBERS || pred_dtheta_cnt * 10 < match_cnt * 7 {
            continue;
        }

        let inv_m = 1.0 / (match_cnt as f64);
        let mx0 = sx0 * inv_m;
        let my0 = sy0 * inv_m;
        let mxt = sxt * inv_m;
        let myt = syt * inv_m;
        let sum_r2_a = sx02_y02 - (sx0 * mx0 + sy0 * my0);
        let trace_cov_a = (sum_r2_a * inv_m) as f32;
        if sum_r2_a < (180.0 * q * q) as f64 || trace_cov_a < 36.0 * q * q {
            continue;
        }
        let sigma_omega =
            (1.4142135 * 0.25 * q) / (wasm_sqrtf(sum_r2_a as f32) * t_multi).max(1e-6);
        if !sigma_omega.is_finite() || sigma_omega > 0.50 {
            continue;
        }
        let sc = (squ - syt * mx0) - (spv - sxt * my0);
        let cd = (spu - sxt * mx0) + (sqv - syt * my0);
        let norm_m = wasm_sqrt_f64(sc * sc + cd * cd);
        if norm_m <= 1e-6 {
            continue;
        }

        let sum_r2_t = st2 - (sxt * mxt + syt * myt);
        let err_m2 = (sum_r2_t + sum_r2_a - 2.0 * norm_m).max(0.0);
        let rmse_multi = wasm_sqrtf((err_m2 * inv_m) as f32);
        if rmse_multi.is_finite() && rmse_multi <= max_multi_rmse {
            let cos_m = (cd / norm_m) as f32;
            let sin_m = (sc / norm_m) as f32;
            let theta_multi = fast_atan2f(sin_m, cos_m);
            let theta_pred = (pred_dtheta_sum / (pred_dtheta_cnt as f64)) as f32;
            let theta_unwrapped = theta_pred + wrap_pi(theta_multi - theta_pred);
            let omega_out = theta_unwrapped / t_multi;
            if omega_out.is_finite() {
                let fit = MultiFrameFit {
                    ok: true,
                    omega: omega_out,
                    bank: bank_id,
                    af: af_sel,
                    ax0: ax0_r + mx0 as f32,
                    ay0: ay0_r + my0 as f32,
                    axt: axt_r + mxt as f32,
                    ayt: ayt_r + myt as f32,
                    cos_m,
                    sin_m,
                };
                if !primary.ok {
                    primary = fit;
                    if !need_alt_bank {
                        break;
                    }
                } else if bank_id != primary.bank && !secondary.ok {
                    secondary = fit;
                    break;
                }
            }
        }
    }

    (primary, secondary)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn finish_validate_and_rank_core(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    root: usize,
    target_mask: u8,
    head_for_eval: i16,
    kept_slice: Option<&[u16]>,
    frame_idx: u32,
    q: f32,
    dt_sec: f32,
    ok: bool,
    x0_bar: f32,
    y0_bar: f32,
    xt_bar: f32,
    yt_bar: f32,
    cos_t: f32,
    sin_t: f32,
    rmse_1f: f32,
    kept: usize,
) -> bool {
    let max_1f_rmse = 0.75 * q;
    if !ok || kept < MIN_RIGID_MEMBERS || !rmse_1f.is_finite() || rmse_1f > max_1f_rmse {
        return false;
    }

    // M3 + H5 fix: mandatory multi-frame validation across synchronized anchor bank (8..=20 frames),
    // retrying eligible banks/anchors if one fails timestamp/conditioning/RMSE, unwrapping against
    // observed accumulated member rotations rather than 1-frame omega extrapolation.
    let dtheta_1f = fast_atan2f(sin_t, cos_t);
    let (mfit, _) = eval_multi_frame_se2_chains(
        ms,
        cs,
        head_for_eval,
        target_mask,
        -1,
        0,
        kept_slice,
        frame_idx,
        q,
        false,
    );
    if !mfit.ok {
        return false;
    }
    let omega_out = mfit.omega;

    let final_kept = kept;
    if final_kept < MIN_RIGID_MEMBERS {
        return false;
    }

    let vx = (xt_bar - x0_bar) / dt_sec;
    let vy = (yt_bar - y0_bar) / dt_sec;
    let cnt_u16 = final_kept as u16;
    let mut pos = cs.body_rank_len;
    while pos > 0
        && (cnt_u16 > cs.body_cnt_rank[pos - 1]
            || (cnt_u16 == cs.body_cnt_rank[pos - 1] && rmse_1f < cs.body_rmse_rank[pos - 1]))
    {
        if pos < MAX_RIGID_BODIES {
            cs.body_root_rank[pos] = cs.body_root_rank[pos - 1];
            cs.body_cnt_rank[pos] = cs.body_cnt_rank[pos - 1];
            cs.body_cx_rank[pos] = cs.body_cx_rank[pos - 1];
            cs.body_cy_rank[pos] = cs.body_cy_rank[pos - 1];
            cs.body_vx_rank[pos] = cs.body_vx_rank[pos - 1];
            cs.body_vy_rank[pos] = cs.body_vy_rank[pos - 1];
            cs.body_omega_rank[pos] = cs.body_omega_rank[pos - 1];
            cs.body_rmse_rank[pos] = cs.body_rmse_rank[pos - 1];
            cs.body_dtheta1f_rank[pos] = cs.body_dtheta1f_rank[pos - 1];
            cs.body_cos1f_rank[pos] = cs.body_cos1f_rank[pos - 1];
            cs.body_sin1f_rank[pos] = cs.body_sin1f_rank[pos - 1];
        }
        pos -= 1;
    }
    if pos < MAX_RIGID_BODIES {
        cs.body_root_rank[pos] = root as u16;
        cs.body_cnt_rank[pos] = cnt_u16;
        cs.body_cx_rank[pos] = xt_bar;
        cs.body_cy_rank[pos] = yt_bar;
        cs.body_vx_rank[pos] = vx;
        cs.body_vy_rank[pos] = vy;
        cs.body_omega_rank[pos] = omega_out;
        cs.body_rmse_rank[pos] = rmse_1f;
        cs.body_dtheta1f_rank[pos] = dtheta_1f;
        cs.body_cos1f_rank[pos] = cos_t;
        cs.body_sin1f_rank[pos] = sin_t;
        if cs.body_rank_len < MAX_RIGID_BODIES {
            cs.body_rank_len += 1;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_validate_and_rank_slice(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    root: usize,
    cand_idxs: &[u16],
    pre_fit: Option<(f32, f32, f32, f32, f32, f32, f32)>,
    frame_idx: u32,
    q: f32,
    dt_sec: f32,
) -> bool {
    let peel_thresh = 1.15 * q;
    let peel_thresh2 = peel_thresh * peel_thresh;
    let (ok1, x0_b, y0_b, xt_b, yt_b, cos1, sin1, rmse1) =
        if let Some((x0, y0, xt, yt, c, s, r)) = pre_fit {
            (true, x0, y0, xt, yt, c, s, r)
        } else {
            let (ok, x0, y0, xt, yt, c, s, r, cnt, _) = solve_kabsch_2d_slice(
                &ms.tracks.prev_x,
                &ms.tracks.prev_y,
                &ms.tracks.pos_x,
                &ms.tracks.pos_y,
                cand_idxs,
                q,
            );
            if !ok || cnt < MIN_RIGID_MEMBERS {
                return false;
            }
            (ok, x0, y0, xt, yt, c, s, r)
        };
    if !ok1 || cand_idxs.len() < MIN_RIGID_MEMBERS {
        return false;
    }

    let mut out_kept = [0u16; 128];
    let mut peeled = 0usize;
    let mut kept_cnt = 0usize;
    let mut ci = 0usize;
    while ci < cand_idxs.len() {
        let ti_u16 = cand_idxs[ci];
        let ti = (ti_u16 as usize).min(MAX_MICRO - 1);
        let dx0 = ms.tracks.prev_x[ti] - x0_b;
        let dy0 = ms.tracks.prev_y[ti] - y0_b;
        let dxt = ms.tracks.pos_x[ti] - xt_b;
        let dyt = ms.tracks.pos_y[ti] - yt_b;
        let rx = dxt - (cos1 * dx0 - sin1 * dy0);
        let ry = dyt - (sin1 * dx0 + cos1 * dy0);
        if rx * rx + ry * ry > peel_thresh2 {
            peeled += 1;
        } else if kept_cnt < out_kept.len() {
            out_kept[kept_cnt] = ti_u16;
            kept_cnt += 1;
        }
        ci += 1;
    }
    if kept_cnt < MIN_RIGID_MEMBERS {
        return false;
    }

    let (ok, x0_bar, y0_bar, xt_bar, yt_bar, cos_t, sin_t, rmse_1f, kept) = if peeled > 0 {
        let (ok, x0, y0, xt, yt, c, s, r, cnt, _) = solve_kabsch_2d_slice(
            &ms.tracks.prev_x,
            &ms.tracks.prev_y,
            &ms.tracks.pos_x,
            &ms.tracks.pos_y,
            &out_kept[..kept_cnt],
            q,
        );
        (ok, x0, y0, xt, yt, c, s, r, cnt)
    } else {
        (ok1, x0_b, y0_b, xt_b, yt_b, cos1, sin1, rmse1, kept_cnt)
    };

    if !finish_validate_and_rank_core(
        ms,
        cs,
        root,
        3,
        -1,
        Some(&out_kept[..kept_cnt]),
        frame_idx,
        q,
        dt_sec,
        ok,
        x0_bar,
        y0_bar,
        xt_bar,
        yt_bar,
        cos_t,
        sin_t,
        rmse_1f,
        kept,
    ) {
        return false;
    }

    // Accepted: clear candidate marks (so peeled nodes become 0), then link kept into comp_head[root]
    ci = 0;
    while ci < cand_idxs.len() {
        let ti = (cand_idxs[ci] as usize).min(MAX_MICRO - 1);
        cs.member_mask[ti] = 0;
        cs.comp_next[ti] = -1;
        ci += 1;
    }
    let mut sub_head: i16 = -1;
    let mut ki = 0usize;
    while ki < kept_cnt {
        let ti = (out_kept[ki] as usize).min(MAX_MICRO - 1);
        cs.member_mask[ti] = 3;
        cs.comp_next[ti] = sub_head;
        sub_head = ti as i16;
        ki += 1;
    }
    cs.comp_head[root] = sub_head;
    true
}

#[inline(never)]
fn try_validate_and_rank_core(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    root: usize,
    target_mask: u8,
    frame_idx: u32,
    q: f32,
    dt_sec: f32,
) -> bool {
    let peel_thresh = 1.15 * q;

    let (ok1, x0_b, y0_b, xt_b, yt_b, cos1, sin1, rmse1, cnt1, _sumr1) = solve_kabsch_2d_masked(
        &ms.tracks.prev_x,
        &ms.tracks.prev_y,
        &ms.tracks.pos_x,
        &ms.tracks.pos_y,
        &cs.comp_next,
        &cs.member_mask,
        cs.comp_head[root],
        target_mask,
        q,
    );
    if !ok1 || cnt1 < MIN_RIGID_MEMBERS {
        return false;
    }

    let mut peeled = 0usize;
    let mut kept_idxs = [0u16; 128];
    let mut kept_cnt = 0usize;
    let mut kept_overflow = false;
    let mut cur = cs.comp_head[root];
    while cur >= 0 {
        let ti = cur as usize;
        if cs.member_mask[ti] == target_mask {
            let dx0 = ms.tracks.prev_x[ti] - x0_b;
            let dy0 = ms.tracks.prev_y[ti] - y0_b;
            let dxt = ms.tracks.pos_x[ti] - xt_b;
            let dyt = ms.tracks.pos_y[ti] - yt_b;
            let rx = dxt - (cos1 * dx0 - sin1 * dy0);
            let ry = dyt - (sin1 * dx0 + cos1 * dy0);
            if rx * rx + ry * ry > peel_thresh * peel_thresh {
                cs.member_mask[ti] = 0;
                peeled += 1;
            } else if kept_cnt < kept_idxs.len() {
                kept_idxs[kept_cnt] = ti as u16;
                kept_cnt += 1;
            } else {
                kept_overflow = true;
            }
        }
        cur = cs.comp_next[ti];
    }

    // H3 fix: reject immediately if peeling drops below MIN_RIGID_MEMBERS (never fall back to pre-peel fit)
    if cnt1.saturating_sub(peeled) < MIN_RIGID_MEMBERS {
        return false;
    }

    let (ok, x0_bar, y0_bar, xt_bar, yt_bar, cos_t, sin_t, rmse_1f, kept, _sumr) = if peeled > 0 {
        if !kept_overflow {
            solve_kabsch_2d_slice(
                &ms.tracks.prev_x,
                &ms.tracks.prev_y,
                &ms.tracks.pos_x,
                &ms.tracks.pos_y,
                &kept_idxs[..kept_cnt],
                q,
            )
        } else {
            solve_kabsch_2d_masked(
                &ms.tracks.prev_x,
                &ms.tracks.prev_y,
                &ms.tracks.pos_x,
                &ms.tracks.pos_y,
                &cs.comp_next,
                &cs.member_mask,
                cs.comp_head[root],
                target_mask,
                q,
            )
        }
    } else {
        (ok1, x0_b, y0_b, xt_b, yt_b, cos1, sin1, rmse1, cnt1, 0.0)
    };

    let kept_slice = if !kept_overflow {
        Some(&kept_idxs[..kept_cnt])
    } else {
        None
    };
    let head_root = cs.comp_head[root];
    finish_validate_and_rank_core(
        ms,
        cs,
        root,
        target_mask,
        head_root,
        kept_slice,
        frame_idx,
        q,
        dt_sec,
        ok,
        x0_bar,
        y0_bar,
        xt_bar,
        yt_bar,
        cos_t,
        sin_t,
        rmse_1f,
        kept,
    )
}

#[inline(never)]
fn merge_co_moving_bodies(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    n_bodies: usize,
    frame_idx: u32,
    q: f32,
    dt_sec: f32,
) -> u32 {
    if n_bodies <= 1 {
        return 0;
    }
    let mut merged_mask = 0u32;
    let dt_safe = dt_sec.max(1e-3);
    let max_v_diff = (2.5 * q / dt_safe).max(6.0 * q);
    let max_v_diff2 = max_v_diff * max_v_diff;
    let max_merge_dist2 = (128.0 * q) * (128.0 * q);

    let mut b_hi = 0usize;
    while b_hi < n_bodies {
        if cs.body_cnt_rank[b_hi] == 0 {
            b_hi += 1;
            continue;
        }
        let root_hi = (cs.body_root_rank[b_hi] as usize).min(MAX_MICRO - 1);
        let tag_hi = (16 + b_hi) as u8;
        let mut b_lo = b_hi + 1;
        while b_lo < n_bodies {
            if cs.body_cnt_rank[b_lo] == 0 {
                b_lo += 1;
                continue;
            }
            let dx = cs.body_cx_rank[b_lo] - cs.body_cx_rank[b_hi];
            let dy = cs.body_cy_rank[b_lo] - cs.body_cy_rank[b_hi];
            if dx * dx + dy * dy <= max_merge_dist2 {
                let om_hi = cs.body_omega_rank[b_hi];
                if (om_hi - cs.body_omega_rank[b_lo]).abs() <= 0.25 {
                    let vpx = cs.body_vx_rank[b_hi] - om_hi * dy;
                    let vpy = cs.body_vy_rank[b_hi] + om_hi * dx;
                    let dvx = vpx - cs.body_vx_rank[b_lo];
                    let dvy = vpy - cs.body_vy_rank[b_lo];
                    if dvx * dvx + dvy * dvy <= max_v_diff2 {
                        let root_lo = (cs.body_root_rank[b_lo] as usize).min(MAX_MICRO - 1);
                        let tag_lo = (16 + b_lo) as u8;
                        let (ok1, x0_m, y0_m, xt_m, yt_m, cos_m, sin_m, rmse_m, cnt_m, _) =
                            solve_kabsch_2d_chains(
                                &ms.tracks.prev_x,
                                &ms.tracks.prev_y,
                                &ms.tracks.pos_x,
                                &ms.tracks.pos_y,
                                &cs.comp_next,
                                &cs.member_mask,
                                cs.comp_head[root_hi],
                                tag_hi,
                                cs.comp_head[root_lo],
                                tag_lo,
                                q,
                            );
                        if ok1 && rmse_m.is_finite() && rmse_m <= 0.75 * q {
                            let (mfit, _) = eval_multi_frame_se2_chains(
                                ms,
                                cs,
                                cs.comp_head[root_hi],
                                tag_hi,
                                cs.comp_head[root_lo],
                                tag_lo,
                                None,
                                frame_idx,
                                q,
                                false,
                            );
                            if mfit.ok {
                                let mut cur = cs.comp_head[root_lo];
                                while cur >= 0 {
                                    let ti = (cur as usize).min(MAX_MICRO - 1);
                                    let nxt = cs.comp_next[ti];
                                    if cs.member_mask[ti] == tag_lo {
                                        cs.member_mask[ti] = tag_hi;
                                        cs.comp_next[ti] = cs.comp_head[root_hi];
                                        cs.comp_head[root_hi] = ti as i16;
                                    } else {
                                        cs.comp_next[ti] = -1;
                                    }
                                    cur = nxt;
                                }
                                cs.comp_head[root_lo] = -1;
                                cs.body_cnt_rank[b_lo] = 0;
                                cs.body_cnt_rank[b_hi] = cnt_m as u16;
                                cs.body_cx_rank[b_hi] = xt_m;
                                cs.body_cy_rank[b_hi] = yt_m;
                                cs.body_vx_rank[b_hi] = (xt_m - x0_m) / dt_sec;
                                cs.body_vy_rank[b_hi] = (yt_m - y0_m) / dt_sec;
                                cs.body_omega_rank[b_hi] = mfit.omega;
                                cs.body_rmse_rank[b_hi] = rmse_m;
                                cs.body_dtheta1f_rank[b_hi] = fast_atan2f(sin_m, cos_m);
                                cs.body_cos1f_rank[b_hi] = cos_m;
                                cs.body_sin1f_rank[b_hi] = sin_m;
                                merged_mask |= 1u32 << b_hi;
                                b_lo = b_hi + 1;
                                continue;
                            }
                        }
                    }
                }
            }
            b_lo += 1;
        }
        b_hi += 1;
    }
    merged_mask
}

#[inline(never)]
fn reconcile_and_complete_rigid_bodies(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    frame_w: usize,
    frame_h: usize,
    q: f32,
    dt_sec: f32,
) {
    let n_bodies = cs.body_rank_len.min(MAX_RIGID_BODIES);
    if n_bodies == 0 {
        return;
    }

    // 1. Strip peeled / evicted members and tag ranked bodies exclusively with `16 + b`.
    let mut dirty_mask = 0u32;
    let mut b = 0usize;
    while b < n_bodies {
        let prev_cnt = cs.body_cnt_rank[b];
        let root = (cs.body_root_rank[b] as usize).min(MAX_MICRO - 1);
        let tag = (16 + b) as u8;
        let mut clean_head: i16 = -1;
        let mut cnt = 0usize;
        let mut cur = cs.comp_head[root];
        while cur >= 0 {
            let ti = (cur as usize).min(MAX_MICRO - 1);
            let nxt = cs.comp_next[ti];
            if cs.member_mask[ti] != 0 {
                cs.member_mask[ti] = tag;
                cs.comp_next[ti] = clean_head;
                clean_head = ti as i16;
                cnt += 1;
            } else {
                cs.comp_next[ti] = -1;
            }
            cur = nxt;
        }
        cs.comp_head[root] = clean_head;
        cs.body_cnt_rank[b] = cnt as u16;
        if (cnt as u16) != prev_cnt {
            dirty_mask |= 1u32 << b;
        }
        b += 1;
    }
    let mut i = 0usize;
    while i < MAX_MICRO {
        if cs.member_mask[i] < 16 {
            cs.member_mask[i] = 0;
            cs.comp_next[i] = -1;
        }
        i += 1;
    }

    // 2. Merge co-moving split fragments belonging to the same rigid SE(2) body.
    dirty_mask |= merge_co_moving_bodies(ms, cs, n_bodies, frame_idx, q, dt_sec);

    // 3. Rescue disconnected rigid members whose k-NN edges were disrupted by crossing clutter,
    // prioritizing bodies with tightest 1f SE(2) fit first and requiring both 1f + multi-frame lock.
    let mut order = [0u8; MAX_RIGID_BODIES];
    let mut ord_len = 0usize;
    b = 0;
    while b < n_bodies {
        if cs.body_cnt_rank[b] >= MIN_RIGID_MEMBERS as u16 {
            let rmse_b = cs.body_rmse_rank[b];
            let mut pos = ord_len.min(MAX_RIGID_BODIES - 1);
            while pos > 0 && rmse_b < cs.body_rmse_rank[(order[pos - 1] as usize).min(MAX_RIGID_BODIES - 1)] {
                order[pos] = order[pos - 1];
                pos -= 1;
            }
            order[pos] = b as u8;
            if ord_len < MAX_RIGID_BODIES {
                ord_len += 1;
            }
        }
        b += 1;
    }

    let m_cw = (frame_w as f32) / (MICRO_CELL_W as f32);
    let m_ch = (frame_h as f32) / (MICRO_CELL_H as f32);
    let inv_mw = 1.0 / m_cw.max(1e-4);
    let inv_mh = 1.0 / m_ch.max(1e-4);
    let e1_lim2 = (0.65 * q) * (0.65 * q);
    let em_lim2 = (1.20 * q) * (1.20 * q);
    let near_mem_r2 = (52.0 * q) * (52.0 * q);
    let max_body_r2 = (96.0 * q) * (96.0 * q);

    let mut any_added = false;
    let mut oi = 0usize;
    let ord_lim = ord_len.min(MAX_RIGID_BODIES);
    while oi < ord_lim {
        let b_idx = (order[oi] as usize).min(MAX_RIGID_BODIES - 1);
        oi += 1;
        if cs.body_cnt_rank[b_idx] == 0 {
            continue;
        }
        let root_b = (cs.body_root_rank[b_idx] as usize).min(MAX_MICRO - 1);
        let tag_b = (16 + b_idx) as u8;

        let mut rpass = 0usize;
        while rpass < 2 {
            let (ok1, x0_b, y0_b, xt_b, yt_b, cos_1f, sin_1f) =
                if rpass == 0 && (dirty_mask & (1u32 << b_idx)) == 0 {
                    let xt = cs.body_cx_rank[b_idx];
                    let yt = cs.body_cy_rank[b_idx];
                    (
                        true,
                        xt - cs.body_vx_rank[b_idx] * dt_sec,
                        yt - cs.body_vy_rank[b_idx] * dt_sec,
                        xt,
                        yt,
                        cs.body_cos1f_rank[b_idx],
                        cs.body_sin1f_rank[b_idx],
                    )
                } else {
                    let (ok_k, x0_k, y0_k, xt_k, yt_k, cos_k, sin_k, _rm1, _cnt_b, _) =
                        solve_kabsch_2d_chains(
                            &ms.tracks.prev_x,
                            &ms.tracks.prev_y,
                            &ms.tracks.pos_x,
                            &ms.tracks.pos_y,
                            &cs.comp_next,
                            &cs.member_mask,
                            cs.comp_head[root_b],
                            tag_b,
                            -1,
                            0,
                            q,
                        );
                    (ok_k, x0_k, y0_k, xt_k, yt_k, cos_k, sin_k)
                };
            if !ok1 {
                break;
            }
            let mut mfit_state = 0u8;
            let mut mfit0 = MultiFrameFit::empty();
            let mut mfit1 = MultiFrameFit::empty();

            let mut max_r2 = 0.0f32;
            let mut mem_n = 0usize;
            let mut mem_overflow = false;
            let mut cur = cs.comp_head[root_b];
            while cur >= 0 {
                let ti = (cur as usize).min(MAX_MICRO - 1);
                if cs.member_mask[ti] == tag_b {
                    let px = ms.tracks.pos_x[ti];
                    let py = ms.tracks.pos_y[ti];
                    let dx = px - xt_b;
                    let dy = py - yt_b;
                    let d2 = dx * dx + dy * dy;
                    if d2 > max_r2 {
                        max_r2 = d2;
                    }
                    if mem_n < CAND_BUF_CAP {
                        cs.cand_buf_dx[mem_n] = px;
                        cs.cand_buf_dy[mem_n] = py;
                        mem_n += 1;
                    } else {
                        mem_overflow = true;
                    }
                }
                cur = cs.comp_next[ti];
            }
            if max_r2 > max_body_r2 || mem_n < MIN_RIGID_MEMBERS {
                break;
            }

            let search_r = (wasm_sqrtf(max_r2) + 56.0 * q).min(116.0 * q);
            let search_r2 = search_r * search_r;
            let cx_lo = clamped_grid_coord(xt_b - search_r, inv_mw, MICRO_CELL_W - 1);
            let cx_hi = clamped_grid_coord(xt_b + search_r, inv_mw, MICRO_CELL_W - 1);
            let cy_lo = clamped_grid_coord(yt_b - search_r, inv_mh, MICRO_CELL_H - 1);
            let cy_hi = clamped_grid_coord(yt_b + search_r, inv_mh, MICRO_CELL_H - 1);

            let mut added = 0usize;
            let mut cy = cy_lo;
            while cy <= cy_hi {
                let row_off = cy.min(MICRO_CELL_H - 1) * MICRO_CELL_W;
                let s = (cs.field_cell_start[(row_off + cx_lo).min(MICRO_CELL_W * MICRO_CELL_H)]
                    as usize)
                    .min(MAX_MICRO);
                let e = (cs.field_cell_start
                    [(row_off + cx_hi + 1).min(MICRO_CELL_W * MICRO_CELL_H)]
                    as usize)
                    .min(MAX_MICRO)
                    .max(s);
                let mut p = s;
                while p < e {
                    let ti = (cs.field_grid_ti[p] as usize).min(MAX_MICRO - 1);
                    if cs.member_mask[ti] == 0 {
                        let px = cs.field_grid_x[p];
                        let py = cs.field_grid_y[p];
                        let cdx = px - xt_b;
                        let cdy = py - yt_b;
                        if cdx * cdx + cdy * cdy <= search_r2 {
                            let ux0 = ms.tracks.prev_x[ti] - x0_b;
                            let uy0 = ms.tracks.prev_y[ti] - y0_b;
                            let rx1 = cdx - (cos_1f * ux0 - sin_1f * uy0);
                            let ry1 = cdy - (sin_1f * ux0 + cos_1f * uy0);
                            if rx1 * rx1 + ry1 * ry1 <= e1_lim2 {
                                let mut near_ok = mem_overflow;
                                if !near_ok {
                                    let mx_s = &cs.cand_buf_dx[..mem_n.min(CAND_BUF_CAP)];
                                    let my_s = &cs.cand_buf_dy[..mem_n.min(CAND_BUF_CAP)];
                                    let mut k = {
                                        use crate::simd::*;
                                        let vpx = f32x4_splat(px);
                                        let vpy = f32x4_splat(py);
                                        let vr2 = f32x4_splat(near_mem_r2);
                                        for (xc, yc) in mx_s.chunks_exact(4).zip(my_s.chunks_exact(4)) {
                                            if let (&[x0, x1, x2, x3], &[y0, y1, y2, y3]) = (xc, yc) {
                                                let vdx = f32x4_sub(vpx, f32x4(x0, x1, x2, x3));
                                                let vdy = f32x4_sub(vpy, f32x4(y0, y1, y2, y3));
                                                let vd2 = f32x4_add(
                                                    f32x4_mul(vdx, vdx),
                                                    f32x4_mul(vdy, vdy),
                                                );
                                                if i32x4_bitmask(f32x4_le(vd2, vr2)) != 0 {
                                                    near_ok = true;
                                                    break;
                                                }
                                            }
                                        }
                                        mx_s.len() & !3
                                    };
                                    if !near_ok {
                                        while k < mx_s.len() {
                                            let mdx = px - mx_s[k];
                                            let mdy = py - my_s[k];
                                            if mdx * mdx + mdy * mdy <= near_mem_r2 {
                                                near_ok = true;
                                                break;
                                            }
                                            k += 1;
                                        }
                                    }
                                }
                                if near_ok {
                                    if mfit_state == 0 {
                                        let (f0, f1) = eval_multi_frame_se2_chains(
                                            ms,
                                            cs,
                                            cs.comp_head[root_b],
                                            tag_b,
                                            -1,
                                            0,
                                            None,
                                            frame_idx,
                                            q,
                                            true,
                                        );
                                        mfit0 = f0;
                                        mfit1 = f1;
                                        mfit_state = if mfit0.ok { 1 } else { 2 };
                                    }
                                    if mfit_state == 2 {
                                        added = 0;
                                        cy = cy_hi + 1;
                                        break;
                                    }
                                    if added < CAND_BUF_CAP
                                        && (matches_multi_frame_fit(
                                            ms, cs, ti, &mfit0, px, py, em_lim2,
                                        ) || matches_multi_frame_fit(
                                            ms, cs, ti, &mfit1, px, py, em_lim2,
                                        ))
                                    {
                                        cs.cand_buf_ti[added] = ti as u16;
                                        added += 1;
                                    }
                                }
                            }
                        }
                    }
                    p += 1;
                }
                cy += 1;
            }
            if mfit_state == 2 || added == 0 {
                break;
            }

            let saved_head = cs.comp_head[root_b];
            let mut k = 0usize;
            while k < added.min(CAND_BUF_CAP) {
                let ti = (cs.cand_buf_ti[k] as usize).min(MAX_MICRO - 1);
                cs.member_mask[ti] = tag_b;
                cs.comp_next[ti] = cs.comp_head[root_b];
                cs.comp_head[root_b] = ti as i16;
                k += 1;
            }
            let (ok_n, x0_n, y0_n, xt_n, yt_n, cos_n, sin_n, rmse_n, cnt_n, _) =
                solve_kabsch_2d_chains(
                    &ms.tracks.prev_x,
                    &ms.tracks.prev_y,
                    &ms.tracks.pos_x,
                    &ms.tracks.pos_y,
                    &cs.comp_next,
                    &cs.member_mask,
                    cs.comp_head[root_b],
                    tag_b,
                    -1,
                    0,
                    q,
                );
            let mut commit_ok = false;
            if ok_n && rmse_n.is_finite() && rmse_n <= 0.75 * q {
                let (mfit_n, _) = eval_multi_frame_se2_chains(
                    ms,
                    cs,
                    cs.comp_head[root_b],
                    tag_b,
                    -1,
                    0,
                    None,
                    frame_idx,
                    q,
                    false,
                );
                if mfit_n.ok {
                    cs.body_cnt_rank[b_idx] = cnt_n as u16;
                    cs.body_cx_rank[b_idx] = xt_n;
                    cs.body_cy_rank[b_idx] = yt_n;
                    cs.body_vx_rank[b_idx] = (xt_n - x0_n) / dt_sec;
                    cs.body_vy_rank[b_idx] = (yt_n - y0_n) / dt_sec;
                    cs.body_omega_rank[b_idx] = mfit_n.omega;
                    cs.body_rmse_rank[b_idx] = rmse_n;
                    cs.body_dtheta1f_rank[b_idx] = fast_atan2f(sin_n, cos_n);
                    cs.body_cos1f_rank[b_idx] = cos_n;
                    cs.body_sin1f_rank[b_idx] = sin_n;
                    commit_ok = true;
                    any_added = true;
                }
            }
            if !commit_ok {
                let mut k = 0usize;
                let a_lim = added.min(CAND_BUF_CAP);
                while k < a_lim {
                    let ti = (cs.cand_buf_ti[k] as usize).min(MAX_MICRO - 1);
                    cs.member_mask[ti] = 0;
                    cs.comp_next[ti] = -1;
                    k += 1;
                }
                cs.comp_head[root_b] = saved_head;
                break;
            }
            rpass += 1;
        }
    }

    // 4. Final merge pass and descending compaction by `(member_count desc, rmse asc)`.
    if any_added {
        merge_co_moving_bodies(ms, cs, n_bodies, frame_idx, q, dt_sec);
    }

    let mut new_len = 0usize;
    b = 0;
    while b < n_bodies {
        let cnt = cs.body_cnt_rank[b];
        if cnt >= MIN_RIGID_MEMBERS as u16 {
            let root = cs.body_root_rank[b];
            let cx = cs.body_cx_rank[b];
            let cy = cs.body_cy_rank[b];
            let vx = cs.body_vx_rank[b];
            let vy = cs.body_vy_rank[b];
            let om = cs.body_omega_rank[b];
            let rm = cs.body_rmse_rank[b];
            let dth = cs.body_dtheta1f_rank[b];
            let c1f = cs.body_cos1f_rank[b];
            let s1f = cs.body_sin1f_rank[b];
            let mut pos = new_len.min(MAX_RIGID_BODIES - 1);
            while pos > 0
                && (cnt > cs.body_cnt_rank[pos - 1]
                    || (cnt == cs.body_cnt_rank[pos - 1] && rm < cs.body_rmse_rank[pos - 1]))
            {
                cs.body_root_rank[pos] = cs.body_root_rank[pos - 1];
                cs.body_cnt_rank[pos] = cs.body_cnt_rank[pos - 1];
                cs.body_cx_rank[pos] = cs.body_cx_rank[pos - 1];
                cs.body_cy_rank[pos] = cs.body_cy_rank[pos - 1];
                cs.body_vx_rank[pos] = cs.body_vx_rank[pos - 1];
                cs.body_vy_rank[pos] = cs.body_vy_rank[pos - 1];
                cs.body_omega_rank[pos] = cs.body_omega_rank[pos - 1];
                cs.body_rmse_rank[pos] = cs.body_rmse_rank[pos - 1];
                cs.body_dtheta1f_rank[pos] = cs.body_dtheta1f_rank[pos - 1];
                cs.body_cos1f_rank[pos] = cs.body_cos1f_rank[pos - 1];
                cs.body_sin1f_rank[pos] = cs.body_sin1f_rank[pos - 1];
                pos -= 1;
            }
            if new_len < MAX_RIGID_BODIES {
                cs.body_root_rank[pos] = root;
                cs.body_cnt_rank[pos] = cnt;
                cs.body_cx_rank[pos] = cx;
                cs.body_cy_rank[pos] = cy;
                cs.body_vx_rank[pos] = vx;
                cs.body_vy_rank[pos] = vy;
                cs.body_omega_rank[pos] = om;
                cs.body_rmse_rank[pos] = rm;
                cs.body_dtheta1f_rank[pos] = dth;
                cs.body_cos1f_rank[pos] = c1f;
                cs.body_sin1f_rank[pos] = s1f;
                new_len += 1;
            }
        }
        b += 1;
    }
    cs.body_rank_len = new_len;
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn recover_h4_subcores(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    root: usize,
    frame_idx: u32,
    q: f32,
    dt_sec: f32,
    max_1f_rmse: f32,
    grow_tol: f32,
) {
    // H4 recovery: split merged or contaminated component into valid >= 6 sub-cores
    let mut rem_head = cs.comp_head[root];
    cs.comp_head[root] = -1;
    let mut rem_len = 0usize;
    let mut cur = rem_head;
    while cur >= 0 {
        let ti = (cur as usize).min(MAX_MICRO - 1);
        cs.member_mask[ti] = 2;
        if rem_len < MAX_MICRO {
            cs.rigid_seeds[rem_len] = ti as u16;
            rem_len += 1;
        }
        cur = cs.comp_next[ti];
    }
    let mut rem_len = rem_len.min(MAX_MICRO);

    let mut seed_checks = 0usize;
    let mut seed_idx = 0usize;
    let mut found_in_root = false;
    let mut cand_idxs = [0u16; 128];
    while seed_idx < rem_len && seed_checks < 24 && rem_len >= MIN_RIGID_MEMBERS {
        if !found_in_root && seed_checks >= 10 {
            break;
        }
        let seed_ti = (cs.rigid_seeds[seed_idx] as usize).min(MAX_MICRO - 1);
        seed_idx += 1;
        if cs.member_mask[seed_ti] != 2 {
            continue;
        }
        seed_checks += 1;

        // Gather unassigned mutual persistent star around seed_ti
        cand_idxs[0] = seed_ti as u16;
        cs.member_mask[seed_ti] = 3;
        let mut star_cnt = 1usize;
        let mut sub_overflow = false;
        let mut pm = cs.knn[seed_ti].persist_mask;
        while pm != 0 {
            let k = (pm.trailing_zeros() as usize).min(MICRO_KNN_K - 1);
            pm &= pm - 1;
            let nb_i16 = cs.knn[seed_ti].idx[k];
            if nb_i16 >= 0 {
                let nb = (nb_i16 as usize).min(MAX_MICRO - 1);
                if cs.member_mask[nb] == 2 && has_reciprocal_persistent_edge(ms, cs, seed_ti, nb, q)
                {
                    cs.member_mask[nb] = 3;
                    if star_cnt < cand_idxs.len() {
                        cand_idxs[star_cnt] = nb as u16;
                    } else {
                        sub_overflow = true;
                    }
                    star_cnt += 1;
                }
            }
        }

        // If star is slightly smaller than 6, also pull 2-hop mutual neighbors in rem pool
        if star_cnt < MIN_RIGID_MEMBERS {
            let star1_cnt = star_cnt.min(cand_idxs.len());
            let mut s_i = 1usize;
            while s_i < star1_cnt {
                let ti = (cand_idxs[s_i] as usize).min(MAX_MICRO - 1);
                let mut pm2 = cs.knn[ti].persist_mask;
                while pm2 != 0 {
                    let kk = (pm2.trailing_zeros() as usize).min(MICRO_KNN_K - 1);
                    pm2 &= pm2 - 1;
                    let nb2_i16 = cs.knn[ti].idx[kk];
                    if nb2_i16 >= 0 {
                        let nb2 = (nb2_i16 as usize).min(MAX_MICRO - 1);
                        if cs.member_mask[nb2] == 2
                            && has_reciprocal_persistent_edge(ms, cs, ti, nb2, q)
                        {
                            cs.member_mask[nb2] = 4; // mark pending 2-hop
                            if star_cnt < cand_idxs.len() {
                                cand_idxs[star_cnt] = nb2 as u16;
                            } else {
                                sub_overflow = true;
                            }
                            star_cnt += 1;
                        }
                    }
                }
                s_i += 1;
            }
            if !sub_overflow {
                let mut u = star1_cnt;
                let u_lim = star_cnt.min(cand_idxs.len());
                while u < u_lim {
                    cs.member_mask[(cand_idxs[u] as usize).min(MAX_MICRO - 1)] = 3;
                    u += 1;
                }
            } else {
                let mut ri = 0usize;
                while ri < rem_len {
                    let ti = (cs.rigid_seeds[ri] as usize).min(MAX_MICRO - 1);
                    if cs.member_mask[ti] == 4 {
                        cs.member_mask[ti] = 3;
                    }
                    ri += 1;
                }
            }
        }

        let mut sub_cnt = star_cnt.min(cand_idxs.len());
        let mut accepted_sub = false;
        if star_cnt >= MIN_RIGID_MEMBERS {
            if sub_overflow {
                rem_head = -1;
                let mut ri = rem_len;
                while ri > 0 {
                    ri -= 1;
                    let ti = (cs.rigid_seeds[ri] as usize).min(MAX_MICRO - 1);
                    cs.comp_next[ti] = rem_head;
                    rem_head = ti as i16;
                }
            }
            let (ok_s, x0s, y0s, xts, yts, coss, sins, rmses, _, _) = if !sub_overflow {
                solve_kabsch_2d_slice(
                    &ms.tracks.prev_x,
                    &ms.tracks.prev_y,
                    &ms.tracks.pos_x,
                    &ms.tracks.pos_y,
                    &cand_idxs[..sub_cnt],
                    q,
                )
            } else {
                solve_kabsch_2d_masked(
                    &ms.tracks.prev_x,
                    &ms.tracks.prev_y,
                    &ms.tracks.pos_x,
                    &ms.tracks.pos_y,
                    &cs.comp_next,
                    &cs.member_mask,
                    rem_head,
                    3,
                    q,
                )
            };
            if ok_s && rmses <= max_1f_rmse * 1.35 {
                // Grow consensus over remaining unassigned tracks in rigid_seeds, then refit once
                // so small edge stars do not shear across wide polygons (Q2 body split).
                let mut grow_pass = 0usize;
                let mut cur_x0 = x0s;
                let mut cur_y0 = y0s;
                let mut cur_xt = xts;
                let mut cur_yt = yts;
                let mut cur_cos = coss;
                let mut cur_sin = sins;
                let mut cur_rmse = rmses;
                let mut fit_dirty = false;
                while grow_pass < 2 {
                    let mut added = 0usize;
                    let mut ri = 0usize;
                    while ri < rem_len {
                        let ti = (cs.rigid_seeds[ri] as usize).min(MAX_MICRO - 1);
                        if cs.member_mask[ti] == 2 {
                            let dx0 = ms.tracks.prev_x[ti] - cur_x0;
                            let dy0 = ms.tracks.prev_y[ti] - cur_y0;
                            let dxt = ms.tracks.pos_x[ti] - cur_xt;
                            let dyt = ms.tracks.pos_y[ti] - cur_yt;
                            let rx = dxt - (cur_cos * dx0 - cur_sin * dy0);
                            let ry = dyt - (cur_sin * dx0 + cur_cos * dy0);
                            if rx * rx + ry * ry <= grow_tol * grow_tol {
                                cs.member_mask[ti] = 3;
                                if !sub_overflow && sub_cnt < cand_idxs.len() {
                                    cand_idxs[sub_cnt] = ti as u16;
                                    sub_cnt += 1;
                                } else {
                                    sub_overflow = true;
                                }
                                added += 1;
                                fit_dirty = true;
                            }
                        }
                        ri += 1;
                    }
                    if grow_pass == 0 && added >= 3 {
                        if sub_overflow {
                            rem_head = -1;
                            let mut rj = rem_len;
                            while rj > 0 {
                                rj -= 1;
                                let ti = (cs.rigid_seeds[rj] as usize).min(MAX_MICRO - 1);
                                cs.comp_next[ti] = rem_head;
                                rem_head = ti as i16;
                            }
                        }
                        let (ok_r, x0r, y0r, xtr, ytr, cosr, sinr, rmser, _, _) = if !sub_overflow {
                            solve_kabsch_2d_slice(
                                &ms.tracks.prev_x,
                                &ms.tracks.prev_y,
                                &ms.tracks.pos_x,
                                &ms.tracks.pos_y,
                                &cand_idxs[..sub_cnt.min(cand_idxs.len())],
                                q,
                            )
                        } else {
                            solve_kabsch_2d_masked(
                                &ms.tracks.prev_x,
                                &ms.tracks.prev_y,
                                &ms.tracks.pos_x,
                                &ms.tracks.pos_y,
                                &cs.comp_next,
                                &cs.member_mask,
                                rem_head,
                                3,
                                q,
                            )
                        };
                        if ok_r && rmser <= max_1f_rmse * 1.35 {
                            cur_x0 = x0r;
                            cur_y0 = y0r;
                            cur_xt = xtr;
                            cur_yt = ytr;
                            cur_cos = cosr;
                            cur_sin = sinr;
                            cur_rmse = rmser;
                            fit_dirty = false;
                            grow_pass += 1;
                            continue;
                        }
                    }
                    break;
                }

                if !sub_overflow {
                    let sr = (cand_idxs[0] as usize).min(MAX_MICRO - 1);
                    let pre_fit = if !fit_dirty {
                        Some((cur_x0, cur_y0, cur_xt, cur_yt, cur_cos, cur_sin, cur_rmse))
                    } else {
                        None
                    };
                    if try_validate_and_rank_slice(
                        ms,
                        cs,
                        sr,
                        &cand_idxs[..sub_cnt.min(cand_idxs.len())],
                        pre_fit,
                        frame_idx,
                        q,
                        dt_sec,
                    ) {
                        accepted_sub = true;
                    }
                } else {
                    let mut sub_root: Option<usize> = None;
                    let mut sub_head: i16 = -1;
                    let mut new_rem_head: i16 = -1;
                    let mut p = rem_head;
                    while p >= 0 {
                        let ti = (p as usize).min(MAX_MICRO - 1);
                        let nxt = cs.comp_next[ti];
                        if cs.member_mask[ti] == 3 {
                            if sub_root.is_none() {
                                sub_root = Some(ti);
                            }
                            cs.comp_next[ti] = sub_head;
                            sub_head = ti as i16;
                        } else {
                            cs.comp_next[ti] = new_rem_head;
                            new_rem_head = ti as i16;
                        }
                        p = nxt;
                    }
                    if let Some(sr) = sub_root {
                        cs.comp_head[sr] = sub_head;
                        if try_validate_and_rank_core(ms, cs, sr, 3, frame_idx, q, dt_sec) {
                            accepted_sub = true;
                        } else {
                            let mut rcur = sub_head;
                            while rcur >= 0 {
                                let ti = (rcur as usize).min(MAX_MICRO - 1);
                                let nxt = cs.comp_next[ti];
                                cs.member_mask[ti] = 2;
                                cs.comp_next[ti] = -1;
                                rcur = nxt;
                            }
                            cs.comp_head[sr] = -1;
                        }
                    }
                }
            }
        }

        if accepted_sub {
            let mut w = 0usize;
            let mut ri = 0usize;
            while ri < rem_len {
                let ti_u16 = cs.rigid_seeds[ri];
                let ti = (ti_u16 as usize).min(MAX_MICRO - 1);
                if cs.member_mask[ti] == 2 {
                    cs.rigid_seeds[w] = ti_u16;
                    w += 1;
                }
                ri += 1;
            }
            rem_len = w;
            found_in_root = true;
            seed_idx = 0;
        } else if !sub_overflow {
            let mut k = 0usize;
            let k_lim = sub_cnt.min(cand_idxs.len());
            while k < k_lim {
                let ti = (cand_idxs[k] as usize).min(MAX_MICRO - 1);
                if cs.member_mask[ti] == 3 {
                    cs.member_mask[ti] = 2;
                }
                k += 1;
            }
        } else {
            let mut ri = 0usize;
            while ri < rem_len {
                let ti = (cs.rigid_seeds[ri] as usize).min(MAX_MICRO - 1);
                if cs.member_mask[ti] == 3 {
                    cs.member_mask[ti] = 2;
                }
                ri += 1;
            }
        }
    }

    // Clear remaining unassigned markers
    let mut ri = 0usize;
    while ri < rem_len {
        let ti = (cs.rigid_seeds[ri] as usize).min(MAX_MICRO - 1);
        cs.member_mask[ti] = 0;
        cs.comp_next[ti] = -1;
        ri += 1;
    }
}

/// Phase 2: Persistent Union-Find rigid-motion clustering + exact `SO(2)` SE(2) fit with
/// reciprocal persistent edges, failed-merge core recovery (`H4`), mandatory multi-frame
/// validation (`M3`), unwrapped rotation (`H5`), and ID-tracked cumulative angle (`M4`).
#[inline(never)]
#[allow(dead_code)]
pub fn solve_rigid_bodies_se2(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    q: f32,
    dt_sec: f32,
) {
    let est_w = ((640.0 * q) as usize).max(32);
    let est_h = ((360.0 * q) as usize).max(32);
    solve_rigid_bodies_se2_impl(ms, cs, frame_idx, est_w, est_h, q, dt_sec);
}

#[inline(never)]
fn solve_rigid_bodies_se2_impl(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    frame_w: usize,
    frame_h: usize,
    q: f32,
    dt_sec: f32,
) {
    if ms.swarm_active == 0 {
        cs.reset();
        cs.initialized = 0;
        return;
    }
    if cs.initialized == 0 {
        cs.reset();
    }
    cs.diagnostic_validity_mask &= !VALIDITY_BIT_RIGID;
    if cs.prev_body_cnt > 0 {
        let prev_n = (cs.prev_body_cnt as usize).min(MAX_RIGID_BODIES);
        cs.export_bodies[..prev_n * BODY_EXPORT_STRIDE].fill(0.0);
        ms.tracks.body_id.fill(0);
    }
    cs.rigid_body_count = 0;
    cs.top_rigid_omega_rad_s = 0.0;
    cs.top_rigid_members = 0;
    cs.top_rigid_se2_rmse_px = 0.0;
    cs.top_rigid_theta_rad = 0.0;
    cs.body_rank_len = 0;

    // Union only reciprocal persistent low-strain edges with multi-edge/1f compatibility (H4).
    // Step 1a guarantees only `cs.field_grid_ti[..clean_n]` hold confirmed non-coasting slots (`knn_count > 0`).
    let clean_n = (cs.field_cell_start[MICRO_CELL_W * MICRO_CELL_H] as usize).min(MAX_MICRO);
    let seed_n = (cs.rigid_seed_n as usize).min(MAX_MICRO);
    let mut any_union = false;
    if seed_n > 0 {
        cs.uf_parent.fill(-1);
        let mut s_idx = 0usize;
        while s_idx < seed_n {
            let i = (cs.rigid_seeds[s_idx] as usize).min(MAX_MICRO - 1);
            if (cs.slot_dtheta_valid[i] & 4u8) != 0 {
                let mut pm = cs.knn[i].persist_mask;
                if pm != 0 {
                    let mut ri = uf_find(&mut cs.uf_parent, i);
                    while pm != 0 {
                        let k = (pm.trailing_zeros() as usize).min(MICRO_KNN_K - 1);
                        pm &= pm - 1;
                        let j_i16 = cs.knn[i].idx[k];
                        if j_i16 > (i as i16) {
                            let j = (j_i16 as usize).min(MAX_MICRO - 1);
                            if ri != uf_find(&mut cs.uf_parent, j)
                                && has_reciprocal_persistent_edge(ms, cs, i, j, q)
                                && has_rigid_edge_support(cs, i, j)
                            {
                                if uf_union(&mut cs.uf_parent, i, j) >= MIN_RIGID_MEMBERS as i16 {
                                    any_union = true;
                                }
                                ri = uf_find(&mut cs.uf_parent, ri);
                            }
                        }
                    }
                }
            }
            s_idx += 1;
        }
    }

    // Group components into linked lists
    let mut cand_n = 0usize;
    if any_union {
        cs.member_mask.fill(0);
        cs.comp_head.fill(-1);
        cs.comp_next.fill(-1);
        let mut p_idx = 0usize;
        while p_idx < clean_n {
            let i = (cs.field_grid_ti[p_idx] as usize).min(MAX_MICRO - 1);
            if cs.uf_parent[i] != -1 {
                let r = uf_find(&mut cs.uf_parent, i);
                if ((-cs.uf_parent[r]) as isize) >= MIN_RIGID_MEMBERS as isize {
                    if cs.comp_head[r] < 0 && cand_n < MAX_RIGID_CORES {
                        cs.cand_roots[cand_n] = r as u16;
                        cand_n += 1;
                    }
                    if cs.comp_head[r] >= 0 || cand_n <= MAX_RIGID_CORES {
                        cs.comp_next[i] = cs.comp_head[r];
                        cs.comp_head[r] = i as i16;
                    }
                }
            }
            p_idx += 1;
        }
    }

    let max_1f_rmse = 0.75 * q;
    let grow_tol = 0.90 * q;

    let mut c_idx = 0usize;
    while c_idx < cand_n.min(MAX_RIGID_CORES) {
        let root = (cs.cand_roots[c_idx] as usize).min(MAX_MICRO - 1);
        c_idx += 1;

        let mut root_cnt = 0usize;
        let mut cur = cs.comp_head[root];
        while cur >= 0 {
            let ti = (cur as usize).min(MAX_MICRO - 1);
            cs.member_mask[ti] = 1;
            root_cnt += 1;
            cur = cs.comp_next[ti];
        }

        if !try_validate_and_rank_core(ms, cs, root, 1, frame_idx, q, dt_sec) {
            if root_cnt <= MIN_RIGID_MEMBERS || root_cnt > 384 {
                let mut p = cs.comp_head[root];
                cs.comp_head[root] = -1;
                while p >= 0 {
                    let ti = (p as usize).min(MAX_MICRO - 1);
                    let nxt = cs.comp_next[ti];
                    cs.member_mask[ti] = 0;
                    cs.comp_next[ti] = -1;
                    p = nxt;
                }
                continue;
            }
            recover_h4_subcores(
                ms,
                cs,
                root,
                frame_idx,
                q,
                dt_sec,
                max_1f_rmse,
                grow_tol,
            );
        }
    }

    if cs.body_rank_len > 0 {
        reconcile_and_complete_rigid_bodies(ms, cs, frame_idx, frame_w, frame_h, q, dt_sec);
    }

    // M4 + persistent body_id: Match ranked bodies frame-to-frame via two-pass slot overlap
    // against persistent IDs (1..=32) so transient flock size rank changes never flip body_id.
    let n_bodies = cs.body_rank_len.min(MAX_RIGID_BODIES);
    let mut used_id_mask = 0u32;
    let mut assigned_id = [0u8; MAX_RIGID_BODIES];
    cs.body_overlap_cnt.fill(0);

    // Pass A: Match components with >= 50% slot overlap onto their previous persistent body_id
    let mut b = 0usize;
    while b < n_bodies {
        let root = (cs.body_root_rank[b] as usize).min(MAX_MICRO - 1);
        let mut seen_mask = 0u32;
        let mut cur = cs.comp_head[root];
        while cur >= 0 {
            let ti = (cur as usize).min(MAX_MICRO - 1);
            if cs.member_mask[ti] != 0 {
                let pr = cs.prev_slot_body_rank[ti];
                if pr >= 0 {
                    let pr_u = pr as usize;
                    if pr_u < MAX_RIGID_BODIES && (used_id_mask & (1u32 << pr_u)) == 0 {
                        cs.body_overlap_cnt[pr_u] = cs.body_overlap_cnt[pr_u].saturating_add(1);
                        seen_mask |= 1u32 << pr_u;
                    }
                }
            }
            cur = cs.comp_next[ti];
        }

        let mut best_pr = usize::MAX;
        let mut best_ov = 0u16;
        let mut m = seen_mask;
        while m != 0 {
            let pr_u = (m.trailing_zeros() as usize).min(MAX_RIGID_BODIES - 1);
            m &= m - 1;
            let ov = cs.body_overlap_cnt[pr_u];
            cs.body_overlap_cnt[pr_u] = 0;
            if ov > best_ov {
                best_ov = ov;
                best_pr = pr_u;
            }
        }
        let dth = cs.body_dtheta1f_rank[b];
        if best_pr < MAX_RIGID_BODIES
            && best_ov >= 3
            && (best_ov as usize) * 2 >= (cs.body_cnt_rank[b] as usize)
        {
            used_id_mask |= 1u32 << best_pr;
            assigned_id[b] = (best_pr + 1) as u8;
            cs.body_cum_theta_tmp[b] = cs.prev_body_cum_theta[best_pr] + dth;
        } else {
            cs.body_cum_theta_tmp[b] = dth;
        }
        b += 1;
    }

    // Pass B: Assign lowest free persistent ID (1..=32) to newly detected bodies
    b = 0;
    while b < n_bodies {
        if assigned_id[b] == 0 {
            let id_idx = ((!used_id_mask).trailing_zeros() as usize).min(MAX_RIGID_BODIES - 1);
            used_id_mask |= 1u32 << id_idx;
            assigned_id[b] = (id_idx + 1) as u8;
        }
        b += 1;
    }

    if cs.prev_body_cnt > 0 || n_bodies > 0 {
        cs.prev_slot_body_rank.fill(-1);
        cs.prev_body_cum_theta.fill(0.0);
    }
    b = 0;
    while b < n_bodies {
        let body_id = assigned_id[b];
        let id_idx = ((body_id.saturating_sub(1)) as usize).min(MAX_RIGID_BODIES - 1);
        let root = (cs.body_root_rank[b] as usize).min(MAX_MICRO - 1);
        let mut cur = cs.comp_head[root];
        while cur >= 0 {
            let ti = (cur as usize).min(MAX_MICRO - 1);
            if cs.member_mask[ti] != 0 {
                ms.tracks.body_id[ti] = body_id;
                cs.prev_slot_body_rank[ti] = id_idx as i8;
                cs.member_mask[ti] = 0;
            }
            cur = cs.comp_next[ti];
        }
        cs.prev_body_cum_theta[id_idx] = cs.body_cum_theta_tmp[b];

        let off = (b * BODY_EXPORT_STRIDE).min(MAX_RIGID_BODIES * BODY_EXPORT_STRIDE - BODY_EXPORT_STRIDE);
        cs.export_bodies[off] = body_id as f32;
        cs.export_bodies[off + 1] = cs.body_cnt_rank[b] as f32;
        cs.export_bodies[off + 2] = finite_or_zero(cs.body_cx_rank[b]);
        cs.export_bodies[off + 3] = finite_or_zero(cs.body_cy_rank[b]);
        cs.export_bodies[off + 4] = finite_or_zero(cs.body_vx_rank[b]);
        cs.export_bodies[off + 5] = finite_or_zero(cs.body_vy_rank[b]);
        cs.export_bodies[off + 6] = finite_or_zero(cs.body_omega_rank[b]);
        cs.export_bodies[off + 7] = finite_or_zero(cs.body_rmse_rank[b]);
        b += 1;
    }
    cs.prev_body_cnt = n_bodies as u8;
    if any_union {
        cs.member_mask.fill(0);
    }

    cs.rigid_body_count = n_bodies as u32;
    if n_bodies > 0 {
        cs.top_body_cum_theta = cs.body_cum_theta_tmp[0];
        cs.top_rigid_omega_rad_s = finite_or_zero(cs.export_bodies[6]);
        cs.top_rigid_members = cs.body_cnt_rank[0] as u32;
        cs.top_rigid_se2_rmse_px = finite_or_zero(cs.export_bodies[7]);
        cs.top_rigid_theta_rad = finite_or_zero(cs.top_body_cum_theta);
        cs.diagnostic_validity_mask |= VALIDITY_BIT_RIGID;
    } else {
        cs.top_body_cum_theta = 0.0;
    }
}

#[inline(always)]
fn partition_cell_cands_mode(
    c_ti: &mut [u16],
    c_d2: &mut [f32],
    lim: usize,
    r0_2: f32,
    r1_2: f32,
    part_mode: u8,
) -> (usize, usize) {
    if part_mode == 0 || c_ti.len() < FIELD_CAND_CAP || c_d2.len() < FIELD_CAND_CAP {
        return (0, 0);
    }
    let lim = lim.min(FIELD_CAND_CAP);
    let c_ti_s = &mut c_ti[..FIELD_CAND_CAP];
    let c_d2_s = &mut c_d2[..FIELD_CAND_CAP];
    if part_mode == 1 {
        let mut lo = 0usize;
        let mut k = 0usize;
        while k < lim {
            let ks = k.min(FIELD_CAND_CAP - 1);
            let d2 = c_d2_s[ks];
            if d2 <= r0_2 {
                if lo != k {
                    let ls = lo.min(FIELD_CAND_CAP - 1);
                    let ti_k = c_ti_s[ks];
                    c_d2_s[ks] = c_d2_s[ls];
                    c_ti_s[ks] = c_ti_s[ls];
                    c_d2_s[ls] = d2;
                    c_ti_s[ls] = ti_k;
                }
                lo += 1;
            }
            k += 1;
        }
        return (lo, lim);
    }
    let mut lo = 0usize;
    let mut mid = 0usize;
    let mut hi = lim;
    while mid < hi {
        let ms_i = mid.min(FIELD_CAND_CAP - 1);
        let d2 = c_d2_s[ms_i];
        if d2 <= r0_2 {
            if lo != mid {
                let ls = lo.min(FIELD_CAND_CAP - 1);
                let ti_m = c_ti_s[ms_i];
                c_d2_s[ms_i] = c_d2_s[ls];
                c_ti_s[ms_i] = c_ti_s[ls];
                c_d2_s[ls] = d2;
                c_ti_s[ls] = ti_m;
            }
            lo += 1;
            mid += 1;
        } else if d2 > r1_2 {
            hi -= 1;
            if mid != hi {
                let hs = hi.min(FIELD_CAND_CAP - 1);
                let ti_m = c_ti_s[ms_i];
                c_d2_s[ms_i] = c_d2_s[hs];
                c_ti_s[ms_i] = c_ti_s[hs];
                c_d2_s[hs] = d2;
                c_ti_s[hs] = ti_m;
            }
        } else {
            mid += 1;
        }
    }
    (lo, mid)
}

#[inline(never)]
fn select_diverse_from_cand_buf(cs: &mut ContinuumState, buf_n: usize) -> usize {
    let n = buf_n.min(CAND_BUF_CAP);
    if n == 0 {
        return 0;
    }
    let b_ti = &cs.cand_buf_ti[..CAND_BUF_CAP];
    let b_d2 = &cs.cand_buf_d2[..CAND_BUF_CAP];
    let b_dx = &cs.cand_buf_dx[..CAND_BUF_CAP];
    let b_dy = &cs.cand_buf_dy[..CAND_BUF_CAP];

    let c_ti = &mut cs.cell_cands_ti[..FIELD_CAND_CAP];
    let c_d2 = &mut cs.cell_cands_d2[..FIELD_CAND_CAP];

    let mut picked0 = 0u64;
    let mut picked1 = 0u64;
    let mut min_d2 = [f32::INFINITY; 16];
    let mut inner_idx = [255u8; 16];
    let mut max_d2 = [-1.0f32; 8];
    let mut outer_idx = [255u8; 8];
    let mut cand_n = 0usize;

    let mut i = 0usize;
    while i < n {
        let ib = i.min(CAND_BUF_CAP - 1);
        let d2 = b_d2[ib];
        if d2 <= 1e-4 {
            if ib < 64 {
                picked0 |= 1u64 << ib;
            } else {
                picked1 |= 1u64 << (ib - 64);
            }
            if cand_n < FIELD_CAND_CAP {
                c_ti[cand_n] = b_ti[ib];
                c_d2[cand_n] = d2;
                cand_n += 1;
            }
        } else {
            let dx = b_dx[ib];
            let dy = b_dy[ib];
            let ax = dx.abs();
            let ay = dy.abs();
            let steep = (ax < ay) as usize;
            let (maj, min_a) = if steep != 0 { (ay, ax) } else { (ax, ay) };
            let sub = (min_a > 0.41421356f32 * maj) as usize;
            let qbits = (((dx < 0.0) as usize) << 1) | ((dy < 0.0) as usize);
            let oct = ((steep << 2) | qbits) & 7;
            let sec = ((oct << 1) | sub) & 15;
            if d2 < min_d2[sec] {
                min_d2[sec] = d2;
                inner_idx[sec] = ib as u8;
            }
            if d2 > max_d2[oct] {
                max_d2[oct] = d2;
                outer_idx[oct] = ib as u8;
            }
        }
        i += 1;
    }

    let mut s = 0usize;
    while s < 16 && cand_n < FIELD_CAND_CAP {
        let idx = inner_idx[s] as usize;
        if idx < n {
            let idxs = idx.min(CAND_BUF_CAP - 1);
            let is_picked = if idxs < 64 {
                (picked0 >> idxs) & 1
            } else {
                (picked1 >> (idxs - 64)) & 1
            };
            if is_picked == 0 {
                if idxs < 64 {
                    picked0 |= 1u64 << idxs;
                } else {
                    picked1 |= 1u64 << (idxs - 64);
                }
                c_ti[cand_n] = b_ti[idxs];
                c_d2[cand_n] = b_d2[idxs];
                cand_n += 1;
            }
        }
        s += 1;
    }

    let mut oct = 0usize;
    while oct < 8 && cand_n < FIELD_CAND_CAP {
        let idx = outer_idx[oct] as usize;
        if idx < n {
            let idxs = idx.min(CAND_BUF_CAP - 1);
            let is_picked = if idxs < 64 {
                (picked0 >> idxs) & 1
            } else {
                (picked1 >> (idxs - 64)) & 1
            };
            if is_picked == 0 {
                if idxs < 64 {
                    picked0 |= 1u64 << idxs;
                } else {
                    picked1 |= 1u64 << (idxs - 64);
                }
                c_ti[cand_n] = b_ti[idxs];
                c_d2[cand_n] = b_d2[idxs];
                cand_n += 1;
            }
        }
        oct += 1;
    }

    if cand_n < FIELD_CAND_CAP {
        let spill_start = cand_n;
        let spill_cap = FIELD_CAND_CAP - spill_start;
        let mut spill_n = 0usize;
        let mut spill_max = -1.0f32;
        let mut max_pos = 0usize;
        i = 0;
        while i < n {
            let ib = i.min(CAND_BUF_CAP - 1);
            let is_picked = if ib < 64 {
                (picked0 >> ib) & 1
            } else {
                (picked1 >> (ib - 64)) & 1
            };
            if is_picked == 0 {
                let d2 = b_d2[ib];
                if spill_n < spill_cap {
                    if d2 > spill_max {
                        spill_max = d2;
                        max_pos = spill_n;
                    }
                    let dst = (spill_start + spill_n).min(FIELD_CAND_CAP - 1);
                    c_d2[dst] = d2;
                    c_ti[dst] = b_ti[ib];
                    spill_n += 1;
                } else if d2 < spill_max {
                    let dst = (spill_start + max_pos).min(FIELD_CAND_CAP - 1);
                    c_d2[dst] = d2;
                    c_ti[dst] = b_ti[ib];
                    let mut m_d2 = c_d2[spill_start.min(FIELD_CAND_CAP - 1)];
                    let mut m_pos = 0usize;
                    let mut j = 1usize;
                    while j < spill_cap {
                        let js = (spill_start + j).min(FIELD_CAND_CAP - 1);
                        if c_d2[js] > m_d2 {
                            m_d2 = c_d2[js];
                            m_pos = j;
                        }
                        j += 1;
                    }
                    spill_max = m_d2;
                    max_pos = m_pos;
                }
            }
            i += 1;
        }
        cand_n += spill_n;
    }

    cand_n.min(FIELD_CAND_CAP)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn gather_cell_candidates_overflow(
    cs: &mut ContinuumState,
    row: usize,
    col: usize,
    xc: f32,
    yc: f32,
    r_max2: f32,
    dy_cells: isize,
    row_dx: &[i8; FIELD_GRID_H],
) -> usize {
    let gx = &cs.field_grid_x[..MAX_MICRO];
    let gy = &cs.field_grid_y[..MAX_MICRO];
    let gti = &cs.field_grid_ti[..MAX_MICRO];
    let row_start = &cs.field_cell_start;
    let c_ti = &mut cs.cell_cands_ti[..FIELD_CAND_CAP];
    let c_d2 = &mut cs.cell_cands_d2[..FIELD_CAND_CAP];
    let o_ti = &mut cs.ftle_cands_ti[..16];
    let o_d2 = &mut cs.ftle_cands_d2[..16];
    let o_dx = &mut cs.ftle_cands_w[..16];
    let o_dy = &mut cs.ftle_cands_res[..16];

    let mut oct_cnt = [0u8; 8];
    let mut oct_best_num = [0.0f32; 8];
    let mut oct_best_d2 = [1.0f32; 8];
    let mut total_oct = 0usize;
    let mut spill_cap = FIELD_CAND_CAP;
    let mut spill_n = 0usize;
    let mut spill_max_d2 = f32::INFINITY;

    let row_s = row as isize;
    let d_max = dy_cells.min((FIELD_GRID_H - 1) as isize);
    let ny_lo = (row_s - d_max).max(0) as usize;
    let ny_hi = (row_s + d_max).min((FIELD_GRID_H - 1) as isize) as usize;
    let mut ny = ny_lo;
    while ny <= ny_hi {
        let d = (ny as isize - row_s).unsigned_abs().min(FIELD_GRID_H - 1);
        let dx_c = row_dx[d] as isize;
        if dx_c < 0 {
            ny += 1;
            continue;
        }
        let cx_lo = (col as isize - dx_c).max(0) as usize;
        let cx_hi = (col as isize + dx_c).min((FIELD_GRID_W - 1) as isize) as usize;
        let row_off = ny * FIELD_GRID_W;
        let idx_a = (row_off + cx_lo).min(FIELD_CELLS);
        let idx_b = (row_off + cx_hi + 1).min(FIELD_CELLS);
        let e = (row_start[idx_b] as usize).min(MAX_MICRO);
        let s = (row_start[idx_a] as usize).min(e);
        let mut p = s;
        while p < e {
            let ps = p.min(MAX_MICRO - 1);
            let dx = gx[ps] - xc;
            let dy = gy[ps] - yc;
            let d2 = dx * dx + dy * dy;
            if d2 <= r_max2 {
                let mut evict_valid = false;
                let mut ev_ti = gti[ps];
                let mut ev_d2 = d2;
                if d2 <= 1e-4 {
                    evict_valid = true;
                } else {
                    let steep = (dx.abs() < dy.abs()) as usize;
                    let sdx = (dx < 0.0) as usize;
                    let sdy = (dy < 0.0) as usize;
                    let bit0 = if steep != 0 { sdy } else { sdx };
                    let bit1 = if steep != 0 { sdx } else { sdy };
                    let oct = ((steep << 2) | (bit1 << 1) | bit0) & 7;
                    let b = (oct << 1) & 14;
                    if oct_cnt[oct] == 0 {
                        o_ti[b] = ev_ti;
                        o_d2[b] = d2;
                        o_dx[b] = dx;
                        o_dy[b] = dy;
                        oct_cnt[oct] = 1;
                        total_oct += 1;
                        spill_cap = FIELD_CAND_CAP.saturating_sub(total_oct).max(8);
                        if spill_n > spill_cap {
                            spill_n = spill_cap;
                        }
                        if spill_n == spill_cap {
                            spill_max_d2 = c_d2[spill_cap - 1];
                        }
                    } else if oct_cnt[oct] == 1 {
                        if d2 < o_d2[b] {
                            o_ti[b + 1] = o_ti[b];
                            o_d2[b + 1] = o_d2[b];
                            o_dx[b + 1] = o_dx[b];
                            o_dy[b + 1] = o_dy[b];
                            o_ti[b] = ev_ti;
                            o_d2[b] = d2;
                            o_dx[b] = dx;
                            o_dy[b] = dy;
                        } else {
                            o_ti[b + 1] = ev_ti;
                            o_d2[b + 1] = d2;
                            o_dx[b + 1] = dx;
                            o_dy[b + 1] = dy;
                        }
                        let cross = o_dx[b] * o_dy[b + 1] - o_dy[b] * o_dx[b + 1];
                        oct_best_num[oct] = cross * cross;
                        oct_best_d2[oct] = o_d2[b + 1];
                        oct_cnt[oct] = 2;
                        total_oct += 1;
                        spill_cap = FIELD_CAND_CAP.saturating_sub(total_oct).max(8);
                        if spill_n > spill_cap {
                            spill_n = spill_cap;
                        }
                        if spill_n == spill_cap {
                            spill_max_d2 = c_d2[spill_cap - 1];
                        }
                    } else {
                        let mut cand_dx = dx;
                        let mut cand_dy = dy;
                        if ev_d2 < o_d2[b] {
                            core::mem::swap(&mut ev_ti, &mut o_ti[b]);
                            core::mem::swap(&mut ev_d2, &mut o_d2[b]);
                            core::mem::swap(&mut cand_dx, &mut o_dx[b]);
                            core::mem::swap(&mut cand_dy, &mut o_dy[b]);
                            let cross0 = o_dx[b] * o_dy[b + 1] - o_dy[b] * o_dx[b + 1];
                            oct_best_num[oct] = cross0 * cross0;
                            oct_best_d2[oct] = o_d2[b + 1];
                        }
                        let dx0 = o_dx[b];
                        let dy0 = o_dy[b];
                        let d2_0 = o_d2[b];
                        let d2_1 = oct_best_d2[oct];
                        let cross_n = dx0 * cand_dy - dy0 * cand_dx;
                        let num_new = cross_n * cross_n;
                        let lhs = num_new * d2_1;
                        let rhs = oct_best_num[oct] * ev_d2;
                        let eps = 1e-4 * d2_0 * ev_d2 * d2_1;
                        let new_beats =
                            lhs > rhs + eps || ((lhs - rhs).abs() <= eps && ev_d2 < d2_1);
                        if new_beats {
                            let prev_ti = o_ti[b + 1];
                            let prev_d2 = o_d2[b + 1];
                            o_ti[b + 1] = ev_ti;
                            o_d2[b + 1] = ev_d2;
                            o_dx[b + 1] = cand_dx;
                            o_dy[b + 1] = cand_dy;
                            oct_best_num[oct] = num_new;
                            oct_best_d2[oct] = ev_d2;
                            ev_ti = prev_ti;
                            ev_d2 = prev_d2;
                        }
                        evict_valid = true;
                    }
                }
                if evict_valid && (spill_n < spill_cap || ev_d2 < spill_max_d2) {
                    let mut pos = spill_n.min(spill_cap - 1);
                    while pos > 0 && ev_d2 < c_d2[pos - 1] {
                        c_d2[pos] = c_d2[pos - 1];
                        c_ti[pos] = c_ti[pos - 1];
                        pos -= 1;
                    }
                    c_d2[pos] = ev_d2;
                    c_ti[pos] = ev_ti;
                    if spill_n < spill_cap {
                        spill_n += 1;
                    }
                    if spill_n == spill_cap {
                        spill_max_d2 = c_d2[spill_cap - 1];
                    }
                }
            }
            p += 1;
        }
        ny += 1;
    }

    let mut cand_n = FIELD_CAND_CAP.saturating_sub(total_oct).min(spill_n);
    let mut oct = 0usize;
    while oct < 8 {
        let b = (oct << 1) & 14;
        let cnt = (oct_cnt[oct] as usize).min(2);
        let mut s = 0usize;
        while s < cnt && cand_n < FIELD_CAND_CAP {
            c_ti[cand_n] = o_ti[b + s];
            c_d2[cand_n] = o_d2[b + s];
            cand_n += 1;
            s += 1;
        }
        oct += 1;
    }
    cand_n.min(FIELD_CAND_CAP)
}

/// Precomputes integer column span `row_dx[|ny - row|]` per vertical cell offset for circular `r_max`.
#[inline(always)]
fn compute_row_dx_spans(r_max: f32, cell_w: f32, cell_h: f32) -> (isize, [i8; FIELD_GRID_H]) {
    let r_max2 = r_max * r_max;
    let dy_cells = (wasm_floorf(r_max / cell_h.max(1e-3) + 0.5 + 1e-4) as isize)
        .max(1)
        .min((FIELD_GRID_H - 1) as isize);
    let mut row_dx = [-1i8; FIELD_GRID_H];
    let inv_cw = 1.0 / cell_w.max(1e-3);
    let mut dy = 0usize;
    while dy <= dy_cells as usize {
        let dy_lb = if dy == 0 {
            0.0
        } else {
            ((dy as f32) - 0.5) * cell_h
        };
        let rem2 = r_max2 - dy_lb * dy_lb;
        if rem2 >= 0.0 {
            let dx_c = (wasm_floorf(wasm_sqrtf(rem2) * inv_cw + 0.5 + 1e-4) as isize)
                .min(FIELD_GRID_W as isize);
            row_dx[dy] = dx_c as i8;
        }
        dy += 1;
    }
    (dy_cells, row_dx)
}

/// Gathers up to `FIELD_CAND_CAP` (`24`) spatially diverse candidates around `(xc, yc)` within
/// `r_max` directly from contiguous CSR row slices (`field_cell_start`, `field_grid_*`),
/// protecting up to 2 orthogonal tracks per octant when `inside > 24`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn gather_cell_candidates_diverse(
    cs: &mut ContinuumState,
    row: usize,
    col: usize,
    xc: f32,
    yc: f32,
    r0_2: f32,
    r1_2: f32,
    r_max2: f32,
    dy_cells: isize,
    row_dx: &[i8; FIELD_GRID_H],
    min_needed: usize,
    part_mode: u8,
) -> (usize, usize, usize) {
    let row_s = row as isize;
    let d_max = dy_cells.min((FIELD_GRID_H - 1) as isize);
    let mut ny = (row_s - d_max).max(0) as usize;
    let ny_hi = (row_s + d_max).min((FIELD_GRID_H - 1) as isize) as usize;

    let mut buf_n = 0usize;
    let mut overflow = false;
    {
        let row_start = &cs.field_cell_start;
        let gx = &cs.field_grid_x[..MAX_MICRO];
        let gy = &cs.field_grid_y[..MAX_MICRO];
        let gti = &cs.field_grid_ti[..MAX_MICRO];
        let b_ti = &mut cs.cand_buf_ti[..CAND_BUF_CAP];
        let b_d2 = &mut cs.cand_buf_d2[..CAND_BUF_CAP];
        let b_dx = &mut cs.cand_buf_dx[..CAND_BUF_CAP];
        let b_dy = &mut cs.cand_buf_dy[..CAND_BUF_CAP];
        let mut row_off = ny * FIELD_GRID_W;
        while ny <= ny_hi && !overflow {
            let d = (ny as isize - row_s).unsigned_abs();
            let dx_c = row_dx[d] as isize;
            if dx_c >= 0 {
                let cx_lo = (col as isize - dx_c).max(0) as usize;
                let cx_hi = (col as isize + dx_c).min((FIELD_GRID_W - 1) as isize) as usize;
                let idx_a = (row_off + cx_lo).min(FIELD_CELLS);
                let idx_b = (row_off + cx_hi + 1).min(FIELD_CELLS);
                let e = (row_start[idx_b] as usize).min(MAX_MICRO);
                let s = (row_start[idx_a] as usize).min(e);
                let len = e - s;
                if len > 0 {
                    let xs = &gx[s..e];
                    let ys = &gy[s..e];
                    let tis = &gti[s..e];
                    let mut p = 0usize;
                    {
                        use crate::simd::*;
                        if len >= 4 {
                            let vxc = f32x4_splat(xc);
                            let vyc = f32x4_splat(yc);
                            let vr2 = f32x4_splat(r_max2);
                            for ((xc_s, yc_s), tc_s) in xs
                                .chunks_exact(4)
                                .zip(ys.chunks_exact(4))
                                .zip(tis.chunks_exact(4))
                            {
                                if let (&[x0, x1, x2, x3], &[y0, y1, y2, y3], &[t0, t1, t2, t3]) =
                                    (xc_s, yc_s, tc_s)
                                {
                                    let vdx = f32x4_sub(f32x4(x0, x1, x2, x3), vxc);
                                    let vdy = f32x4_sub(f32x4(y0, y1, y2, y3), vyc);
                                    let vd2 = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
                                    let mask = i32x4_bitmask(f32x4_le(vd2, vr2));
                                    if mask == 0xF && buf_n <= CAND_BUF_CAP - 4 {
                                        let bn = buf_n;
                                        b_ti[bn] = t0;
                                        b_ti[bn + 1] = t1;
                                        b_ti[bn + 2] = t2;
                                        b_ti[bn + 3] = t3;
                                        b_d2[bn] = f32x4_extract_lane::<0>(vd2);
                                        b_d2[bn + 1] = f32x4_extract_lane::<1>(vd2);
                                        b_d2[bn + 2] = f32x4_extract_lane::<2>(vd2);
                                        b_d2[bn + 3] = f32x4_extract_lane::<3>(vd2);
                                        b_dx[bn] = f32x4_extract_lane::<0>(vdx);
                                        b_dx[bn + 1] = f32x4_extract_lane::<1>(vdx);
                                        b_dx[bn + 2] = f32x4_extract_lane::<2>(vdx);
                                        b_dx[bn + 3] = f32x4_extract_lane::<3>(vdx);
                                        b_dy[bn] = f32x4_extract_lane::<0>(vdy);
                                        b_dy[bn + 1] = f32x4_extract_lane::<1>(vdy);
                                        b_dy[bn + 2] = f32x4_extract_lane::<2>(vdy);
                                        b_dy[bn + 3] = f32x4_extract_lane::<3>(vdy);
                                        buf_n = bn + 4;
                                    } else if mask != 0 {
                                        macro_rules! push_lane {
                                            ($t_val:expr, $lane:literal) => {
                                                if buf_n < CAND_BUF_CAP {
                                                    b_ti[buf_n] = $t_val;
                                                    b_d2[buf_n] = f32x4_extract_lane::<$lane>(vd2);
                                                    b_dx[buf_n] = f32x4_extract_lane::<$lane>(vdx);
                                                    b_dy[buf_n] = f32x4_extract_lane::<$lane>(vdy);
                                                    buf_n += 1;
                                                } else {
                                                    overflow = true;
                                                }
                                            };
                                        }
                                        if (mask & 1) != 0 {
                                            push_lane!(t0, 0);
                                        }
                                        if !overflow && (mask & 2) != 0 {
                                            push_lane!(t1, 1);
                                        }
                                        if !overflow && (mask & 4) != 0 {
                                            push_lane!(t2, 2);
                                        }
                                        if !overflow && (mask & 8) != 0 {
                                            push_lane!(t3, 3);
                                        }
                                        if overflow {
                                            break;
                                        }
                                    }
                                }
                            }
                            p = if overflow { len } else { len & !3 };
                        }
                    }
                    while p < len {
                        let dx = xs[p] - xc;
                        let dy = ys[p] - yc;
                        let d2 = dx * dx + dy * dy;
                        if d2 <= r_max2 {
                            if buf_n < CAND_BUF_CAP {
                                b_ti[buf_n] = tis[p];
                                b_d2[buf_n] = d2;
                                b_dx[buf_n] = dx;
                                b_dy[buf_n] = dy;
                                buf_n += 1;
                            } else {
                                overflow = true;
                                break;
                            }
                        }
                        p += 1;
                    }
                }
            }
            row_off += FIELD_GRID_W;
            ny += 1;
        }
    }
    let lim = if overflow {
        gather_cell_candidates_overflow(cs, row, col, xc, yc, r_max2, dy_cells, row_dx)
    } else if buf_n < min_needed {
        return (0, 0, 0);
    } else if buf_n <= FIELD_CAND_CAP {
        cs.cell_cands_ti[..buf_n].copy_from_slice(&cs.cand_buf_ti[..buf_n]);
        cs.cell_cands_d2[..buf_n].copy_from_slice(&cs.cand_buf_d2[..buf_n]);
        buf_n
    } else {
        select_diverse_from_cand_buf(cs, buf_n)
    };
    let lim = lim.min(FIELD_CAND_CAP);
    if lim < min_needed {
        return (0, 0, 0);
    }
    if part_mode == 0 {
        return (0, 0, lim);
    }
    let (end0, end1) = partition_cell_cands_mode(
        &mut cs.cell_cands_ti,
        &mut cs.cell_cands_d2,
        lim,
        r0_2,
        r1_2,
        part_mode,
    );
    (end0, end1, lim)
}

#[inline(always)]
fn loo_eval_top_k_below_cutoff(
    dxs: &[f32; FIELD_CAND_CAP],
    dys: &[f32; FIELD_CAND_CAP],
    dus: &[f32; FIELD_CAND_CAP],
    dvs: &[f32; FIELD_CAND_CAP],
    m: usize,
    drop: usize,
    c1: f32,
    f11: f32,
    f12: f32,
    c2: f32,
    f21: f32,
    f22: f32,
    top_k: usize,
    cutoff_r2: f32,
) -> (bool, f32) {
    let m = m.min(FIELD_CAND_CAP);
    let mut ge_cnt = 0usize;
    let mut u = 0usize;
    {
        use crate::simd::*;
        let n4 = m & !3;
        if n4 > 0 {
            let vc1 = f32x4_splat(c1);
            let vc2 = f32x4_splat(c2);
            let vf11 = f32x4_splat(f11);
            let vf12 = f32x4_splat(f12);
            let vf21 = f32x4_splat(f21);
            let vf22 = f32x4_splat(f22);
            let vcut = f32x4_splat(cutoff_r2);
            while u < n4 {
                let k = u.min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(
                    vdu,
                    f32x4_add(vc1, f32x4_add(f32x4_mul(vf11, vdx), f32x4_mul(vf12, vdy))),
                );
                let vry = f32x4_sub(
                    vdv,
                    f32x4_add(vc2, f32x4_add(f32x4_mul(vf21, vdx), f32x4_mul(vf22, vdy))),
                );
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                let mut bm = i32x4_bitmask(f32x4_ge(vr2, vcut)) as u32;
                if drop.wrapping_sub(u) < 4 {
                    bm &= !(1u32 << (drop - u));
                }
                ge_cnt += bm.count_ones() as usize;
                if ge_cnt >= top_k {
                    return (false, cutoff_r2);
                }
                u += 4;
            }
        }
    }
    while u < m {
        if u != drop {
            let k = u.min(FIELD_CAND_CAP - 1);
            let dx = dxs[k];
            let dy = dys[k];
            let rx = dus[k] - (c1 + f11 * dx + f12 * dy);
            let ry = dvs[k] - (c2 + f21 * dx + f22 * dy);
            if rx * rx + ry * ry >= cutoff_r2 {
                ge_cnt += 1;
                if ge_cnt >= top_k {
                    return (false, cutoff_r2);
                }
            }
        }
        u += 1;
    }
    // Qualifying outlier drop: extract exact top_k-th largest residual to tighten cutoff_r2.
    let mut top = [0.0f32; 4];
    let mut filled = 0usize;
    u = 0;
    while u < m {
        if u != drop {
            let k = u.min(FIELD_CAND_CAP - 1);
            let dx = dxs[k];
            let dy = dys[k];
            let rx = dus[k] - (c1 + f11 * dx + f12 * dy);
            let ry = dvs[k] - (c2 + f21 * dx + f22 * dy);
            let r2_u = rx * rx + ry * ry;
            if filled < top_k {
                let mut p = filled;
                while p > 0 && top[(p - 1) & 3] < r2_u {
                    top[p & 3] = top[(p - 1) & 3];
                    p -= 1;
                }
                top[p & 3] = r2_u;
                filled += 1;
            } else if r2_u > top[(top_k - 1) & 3] {
                let mut p = top_k - 1;
                while p > 0 && top[(p - 1) & 3] < r2_u {
                    top[p & 3] = top[(p - 1) & 3];
                    p -= 1;
                }
                top[p & 3] = r2_u;
            }
        }
        u += 1;
    }
    (true, top[(top_k - 1) & 3])
}

/// Branchless SIMD bitonic / Bose-Nelson sorting network for `n <= FIELD_CAND_CAP` residuals in `buf`.
#[inline(never)]
fn median_and_q80_r2(
    src: &[f32; FIELD_CAND_CAP],
    n: usize,
    want_q80: bool,
) -> (f32, f32) {
    let n = n.min(FIELD_CAND_CAP);
    if n == 0 {
        return (0.0, 0.0);
    }
    let mid = n >> 1;
    let odd = (n & 1) == 1;
    let idx_q80 = if want_q80 && n >= 2 {
        ((n * 4) / 5).min(n - 2)
    } else {
        0
    };
    let (lo_max, hi_v, q80_r2) = {
        use crate::simd::*;
        let vinf = f32x4_splat(f32::INFINITY);
        let vn = i32x4_splat(n as i32);
        #[inline(always)]
        fn load4_pad(src: &[f32; FIELD_CAND_CAP], base: usize, vn: v128, vinf: v128) -> v128 {
            let b = base.min(FIELD_CAND_CAP - 4);
            let raw = f32x4(src[b], src[b + 1], src[b + 2], src[b + 3]);
            let iv = i32x4(b as i32, (b + 1) as i32, (b + 2) as i32, (b + 3) as i32);
            v128_bitselect(raw, vinf, i32x4_lt(iv, vn))
        }
        #[inline(always)]
        fn sort4(mut v: v128) -> v128 {
            let s1 = i32x4_shuffle::<1, 0, 3, 2>(v, v);
            v = i32x4_shuffle::<0, 5, 2, 7>(f32x4_pmin(v, s1), f32x4_pmax(v, s1));
            let s2 = i32x4_shuffle::<2, 3, 0, 1>(v, v);
            v = i32x4_shuffle::<0, 1, 6, 7>(f32x4_pmin(v, s2), f32x4_pmax(v, s2));
            let s3 = i32x4_shuffle::<0, 2, 1, 3>(v, v);
            i32x4_shuffle::<0, 1, 6, 3>(f32x4_pmin(v, s3), f32x4_pmax(v, s3))
        }
        #[inline(always)]
        fn bitonic4(mut v: v128) -> v128 {
            let s2 = i32x4_shuffle::<2, 3, 0, 1>(v, v);
            v = i32x4_shuffle::<0, 1, 6, 7>(f32x4_pmin(v, s2), f32x4_pmax(v, s2));
            let s1 = i32x4_shuffle::<1, 0, 3, 2>(v, v);
            i32x4_shuffle::<0, 5, 2, 7>(f32x4_pmin(v, s1), f32x4_pmax(v, s1))
        }
        #[inline(always)]
        fn merge4_4(a: v128, b: v128) -> (v128, v128) {
            let br = i32x4_shuffle::<3, 2, 1, 0>(b, b);
            (bitonic4(f32x4_pmin(a, br)), bitonic4(f32x4_pmax(a, br)))
        }
        #[inline(always)]
        fn bitonic8(v0: v128, v1: v128) -> (v128, v128) {
            (bitonic4(f32x4_pmin(v0, v1)), bitonic4(f32x4_pmax(v0, v1)))
        }
        #[inline(always)]
        fn merge8_8(a0: v128, a1: v128, b0: v128, b1: v128) -> (v128, v128, v128, v128) {
            let b1r = i32x4_shuffle::<3, 2, 1, 0>(b1, b1);
            let b0r = i32x4_shuffle::<3, 2, 1, 0>(b0, b0);
            let lo0 = f32x4_pmin(a0, b1r);
            let hi1 = f32x4_pmax(a0, b1r);
            let lo1 = f32x4_pmin(a1, b0r);
            let hi0 = f32x4_pmax(a1, b0r);
            let (r0, r1) = bitonic8(lo0, lo1);
            let (r2, r3) = bitonic8(hi0, hi1);
            (r0, r1, r2, r3)
        }
        #[inline(always)]
        fn pick4(v: v128, sub: usize) -> f32 {
            match sub {
                0 => f32x4_extract_lane::<0>(v),
                1 => f32x4_extract_lane::<1>(v),
                2 => f32x4_extract_lane::<2>(v),
                _ => f32x4_extract_lane::<3>(v),
            }
        }

        if n <= 8 {
            let (r0, r1) = merge4_4(
                sort4(load4_pad(src, 0, vn, vinf)),
                sort4(load4_pad(src, 4, vn, vinf)),
            );
            let pick = |idx: usize| -> f32 {
                if idx < 4 { pick4(r0, idx) } else { pick4(r1, idx - 4) }
            };
            let hi = pick(mid);
            let lo = if odd { 0.0 } else { pick(mid - 1) };
            let q = if want_q80 && n >= 2 { pick(idx_q80) } else { 0.0 };
            (lo, hi, q)
        } else if n <= 16 {
            let (a0, a1) = merge4_4(
                sort4(load4_pad(src, 0, vn, vinf)),
                sort4(load4_pad(src, 4, vn, vinf)),
            );
            let (b0, b1) = merge4_4(
                sort4(load4_pad(src, 8, vn, vinf)),
                sort4(load4_pad(src, 12, vn, vinf)),
            );
            let (r0, r1, r2, r3) = merge8_8(a0, a1, b0, b1);
            let pick = |idx: usize| -> f32 {
                match idx >> 2 {
                    0 => pick4(r0, idx & 3),
                    1 => pick4(r1, idx & 3),
                    2 => pick4(r2, idx & 3),
                    _ => pick4(r3, idx & 3),
                }
            };
            let hi = pick(mid);
            let lo = if odd { 0.0 } else { pick(mid - 1) };
            let q = if want_q80 { pick(idx_q80) } else { 0.0 };
            (lo, hi, q)
        } else {
            let (p0, p1) = merge4_4(
                sort4(load4_pad(src, 0, vn, vinf)),
                sort4(load4_pad(src, 4, vn, vinf)),
            );
            let (p2, p3) = merge4_4(
                sort4(load4_pad(src, 8, vn, vinf)),
                sort4(load4_pad(src, 12, vn, vinf)),
            );
            let (a0, a1, a2, a3) = merge8_8(p0, p1, p2, p3);
            let (b0, b1) = merge4_4(
                sort4(load4_pad(src, 16, vn, vinf)),
                sort4(load4_pad(src, 20, vn, vinf)),
            );
            let b1r = i32x4_shuffle::<3, 2, 1, 0>(b1, b1);
            let b0r = i32x4_shuffle::<3, 2, 1, 0>(b0, b0);
            let lo2 = f32x4_pmin(a2, b1r);
            let hi1 = f32x4_pmax(a2, b1r);
            let lo3 = f32x4_pmin(a3, b0r);
            let hi0 = f32x4_pmax(a3, b0r);
            let (r0, r1) = bitonic8(f32x4_pmin(a0, lo2), f32x4_pmin(a1, lo3));
            let (r2, r3) = bitonic8(f32x4_pmax(a0, lo2), f32x4_pmax(a1, lo3));
            let (r4, r5) = bitonic8(hi0, hi1);
            let pick = |idx: usize| -> f32 {
                match idx >> 2 {
                    0 => pick4(r0, idx & 3),
                    1 => pick4(r1, idx & 3),
                    2 => pick4(r2, idx & 3),
                    3 => pick4(r3, idx & 3),
                    4 => pick4(r4, idx & 3),
                    _ => pick4(r5, idx & 3),
                }
            };
            let hi = pick(mid);
            let lo = if odd { 0.0 } else { pick(mid - 1) };
            let q = if want_q80 { pick(idx_q80) } else { 0.0 };
            (lo, hi, q)
        }
    };
    let med_r2 = if odd {
        hi_v
    } else {
        0.25 * (lo_max + hi_v + 2.0 * wasm_sqrtf((lo_max * hi_v).max(0.0)))
    };
    (med_r2, q80_r2)
}

#[inline(always)]
fn hsum_f32x4(v: crate::simd::v128) -> f32 {
    use crate::simd::*;
    let s1 = f32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    let s2 = f32x4_add(s1, i32x4_shuffle::<1, 0, 3, 2>(s1, s1));
    f32x4_extract_lane::<0>(s2)
}

#[inline(always)]
fn hmax_f32x4(v: crate::simd::v128) -> f32 {
    use crate::simd::*;
    let m1 = f32x4_pmax(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    let m2 = f32x4_pmax(m1, i32x4_shuffle::<1, 0, 3, 2>(m1, m1));
    f32x4_extract_lane::<0>(m2)
}

#[inline(always)]
fn hmin_f32x4(v: crate::simd::v128) -> f32 {
    use crate::simd::*;
    let m1 = f32x4_pmin(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    let m2 = f32x4_pmin(m1, i32x4_shuffle::<1, 0, 3, 2>(m1, m1));
    f32x4_extract_lane::<0>(m2)
}

#[inline(never)]
fn fit_cand_grad_weighted_2x2(
    cs: &ContinuumState,
    m: usize,
    use_active: bool,
    min_eig_px2: f32,
) -> (bool, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    let m = m.min(FIELD_CAND_CAP);
    let ws = &cs.ftle_cands_w[..m];
    let dxs = &cs.cand_dx[..m];
    let dys = &cs.cand_dy[..m];
    let dus = &cs.cand_du[..m];
    let dvs = &cs.cand_dv[..m];

    let mut cnt = 0usize;
    let (w_sum, sx, sy, su, sv, sxx, sxy, syy, sxu, syu, sxv, syv) = {
        {
            use crate::simd::*;
            let mut v_lin = f32x4_splat(0.0);
            let mut v_x = f32x4_splat(0.0);
            let mut v_y = f32x4_splat(0.0);
            let mut k = 0usize;
            while k < m {
                let w = ws[k];
                if !use_active || w > 0.0 {
                    let dx = dxs[k];
                    let dy = dys[k];
                    let du = dus[k];
                    let dv = dvs[k];
                    let v_pts = f32x4(dx, dy, du, dv);
                    v_lin = f32x4_add(v_lin, f32x4_mul(f32x4_splat(w), v_pts));
                    v_x = f32x4_add(v_x, f32x4_mul(f32x4_splat(w * dx), v_pts));
                    let vy_terms = f32x4_replace_lane::<0>(
                        f32x4_mul(f32x4_splat(w * dy), v_pts),
                        w,
                    );
                    v_y = f32x4_add(v_y, vy_terms);
                    cnt += 1;
                }
                k += 1;
            }
            (
                f32x4_extract_lane::<0>(v_y),
                f32x4_extract_lane::<0>(v_lin),
                f32x4_extract_lane::<1>(v_lin),
                f32x4_extract_lane::<2>(v_lin),
                f32x4_extract_lane::<3>(v_lin),
                f32x4_extract_lane::<0>(v_x),
                f32x4_extract_lane::<1>(v_x),
                f32x4_extract_lane::<1>(v_y),
                f32x4_extract_lane::<2>(v_x),
                f32x4_extract_lane::<2>(v_y),
                f32x4_extract_lane::<3>(v_x),
                f32x4_extract_lane::<3>(v_y),
            )
        }
    };
    if cnt < 4 || w_sum <= 1e-6 {
        return (false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let inv_w = 1.0 / w_sum;
    let mx = sx * inv_w;
    let my = sy * inv_w;
    let mu = su * inv_w;
    let mv = sv * inv_w;
    let gxx = sxx - sx * mx;
    let gxy = sxy - sx * my;
    let gyy = syy - sy * my;
    if !is_well_conditioned_cov2_f32(gxx, gxy, gyy, w_sum * min_eig_px2) {
        return (false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let det = gxx * gyy - gxy * gxy;
    if det <= 1e-10 {
        return (false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let bxu = sxu - sx * mu;
    let byu = syu - sy * mu;
    let bxv = sxv - sx * mv;
    let byv = syv - sy * mv;
    let inv_det = 1.0 / det;
    let du_dx = (gyy * bxu - gxy * byu) * inv_det;
    let du_dy = (-gxy * bxu + gxx * byu) * inv_det;
    let dv_dx = (gyy * bxv - gxy * byv) * inv_det;
    let dv_dy = (-gxy * bxv + gxx * byv) * inv_det;
    if !du_dx.is_finite() || !du_dy.is_finite() || !dv_dx.is_finite() || !dv_dy.is_finite() {
        return (false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    (true, mu, mv, mx, my, du_dx, du_dy, dv_dx, dv_dy, w_sum)
}

struct GradPrefixMoments {
    m: usize,
    u_ref: f32,
    v_ref: f32,
    sx: f32,
    sy: f32,
    su: f32,
    sv: f32,
    sxx: f32,
    sxy: f32,
    syy: f32,
    sxu: f32,
    syu: f32,
    sxv: f32,
    syv: f32,
}

impl GradPrefixMoments {
    #[inline(always)]
    fn new(u_ref: f32, v_ref: f32) -> Self {
        Self {
            m: 0,
            u_ref,
            v_ref,
            sx: 0.0,
            sy: 0.0,
            su: 0.0,
            sv: 0.0,
            sxx: 0.0,
            sxy: 0.0,
            syy: 0.0,
            sxu: 0.0,
            syu: 0.0,
            sxv: 0.0,
            syv: 0.0,
        }
    }
}

#[inline(always)]
fn fill_grad_prefix_moments(
    cs: &mut ContinuumState,
    pm: &mut GradPrefixMoments,
    cand_n: usize,
    xc: f32,
    yc: f32,
) -> usize {
    let cand_n = cand_n.min(FIELD_CAND_CAP);
    let mut m = pm.m.min(cand_n);
    if m < cand_n {
        let u_ref = pm.u_ref;
        let v_ref = pm.v_ref;
        let cd2 = &cs.cell_cands_d2[..FIELD_CAND_CAP];
        let cti = &cs.cell_cands_ti[..FIELD_CAND_CAP];
        let gx = &cs.field_grid_x[..MAX_MICRO];
        let gy = &cs.field_grid_y[..MAX_MICRO];
        let gaux = &cs.field_grid_aux[..MAX_MICRO];
        let f_d2 = &mut cs.ftle_cands_d2[..FIELD_CAND_CAP];
        let dxs = &mut cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &mut cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &mut cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &mut cs.cand_dv[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let mut v_lin = f32x4(pm.sx, pm.sy, pm.su, pm.sv);
            let mut v_dx_row = f32x4(pm.sxx, pm.sxy, pm.sxu, pm.sxv);
            let mut v_dy_row = f32x4(0.0, pm.syy, pm.syu, pm.syv);
            while m < cand_n {
                let ms_idx = m.min(FIELD_CAND_CAP - 1);
                let d2 = cd2[ms_idx];
                let p = (cti[ms_idx] as usize).min(MAX_MICRO - 1);
                let dx = gx[p] - xc;
                let dy = gy[p] - yc;
                let uv = gaux[p];
                let du = uv[0] - u_ref;
                let dv = uv[1] - v_ref;
                f_d2[ms_idx] = d2;
                dxs[ms_idx] = dx;
                dys[ms_idx] = dy;
                dus[ms_idx] = du;
                dvs[ms_idx] = dv;
                let v_xyuv = f32x4(dx, dy, du, dv);
                v_lin = f32x4_add(v_lin, v_xyuv);
                v_dx_row = f32x4_add(v_dx_row, f32x4_mul(f32x4_splat(dx), v_xyuv));
                v_dy_row = f32x4_add(v_dy_row, f32x4_mul(f32x4_splat(dy), v_xyuv));
                m += 1;
            }
            pm.sx = f32x4_extract_lane::<0>(v_lin);
            pm.sy = f32x4_extract_lane::<1>(v_lin);
            pm.su = f32x4_extract_lane::<2>(v_lin);
            pm.sv = f32x4_extract_lane::<3>(v_lin);
            pm.sxx = f32x4_extract_lane::<0>(v_dx_row);
            pm.sxy = f32x4_extract_lane::<1>(v_dx_row);
            pm.sxu = f32x4_extract_lane::<2>(v_dx_row);
            pm.sxv = f32x4_extract_lane::<3>(v_dx_row);
            pm.syy = f32x4_extract_lane::<1>(v_dy_row);
            pm.syu = f32x4_extract_lane::<2>(v_dy_row);
            pm.syv = f32x4_extract_lane::<3>(v_dy_row);
            pm.m = m;
        }
    }
    m
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn has_dominant_press_outlier(
    cs: &ContinuumState,
    m: usize,
    mx: f32,
    my: f32,
    gxx: f32,
    gxy: f32,
    gyy: f32,
    inv_det0: f32,
    inv_m: f32,
    frac: f32,
) -> bool {
    let m = m.min(FIELD_CAND_CAP);
    let g00_i = gyy * inv_det0;
    let two_g01_i = -2.0 * gxy * inv_det0;
    let g11_i = gxx * inv_det0;
    let dxs = &cs.cand_dx[..m];
    let dys = &cs.cand_dy[..m];
    let ress_0 = &cs.ftle_cands_res[..m];
    let mut ssr_0 = 0.0f32;
    let mut best_num = -1.0f32;
    let mut best_den = 1.0f32;
    let mut drop = m;
    while drop > 0 {
        drop -= 1;
        let r_k = ress_0[drop];
        ssr_0 += r_k;
        let ux = dxs[drop] - mx;
        let uy = dys[drop] - my;
        let h_i = inv_m + ux * (g00_i * ux + two_g01_i * uy) + g11_i * uy * uy;
        if h_i < 1.0 - 1e-6 {
            let omh = 1.0 - h_i;
            if r_k * best_den > best_num * omh {
                best_num = r_k;
                best_den = omh;
            }
        }
    }
    best_num >= frac * ssr_0 * best_den
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn find_cook_press_outlier_indices(
    cs: &ContinuumState,
    m: usize,
    inv_m: f32,
    mx: f32,
    my: f32,
    g00_i: f32,
    two_g01_i: f32,
    g11_i: f32,
    cook_rhs: f32,
    press_frac: f32,
) -> (usize, usize) {
    let m = m.min(FIELD_CAND_CAP);
    let m_pad = ((m + 3) & !3).min(FIELD_CAND_CAP);
    let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
    let dys = &cs.cand_dy[..FIELD_CAND_CAP];
    let ress_0 = &cs.ftle_cands_res[..FIELD_CAND_CAP];

    let mut cook_idx = m;
    let mut cook_best_num = -1.0f32;
    let mut cook_best_den = 1.0f32;
    let mut press_idx = m;
    let mut press_best_num = -1.0f32;
    let mut press_best_den = 1.0f32;
    let ssr_0: f32;

    {
        use crate::simd::*;
        let vinv_m = f32x4_splat(inv_m);
        let vmx = f32x4_splat(mx);
        let vmy = f32x4_splat(my);
        let vg00 = f32x4_splat(g00_i);
        let v2g01 = f32x4_splat(two_g01_i);
        let vg11 = f32x4_splat(g11_i);
        let v1_eps = f32x4_splat(1.0 - 1e-6);
        let vone = f32x4_splat(1.0);
        let vcook_rhs = f32x4_splat(cook_rhs);
        let vm = f32x4_splat(m as f32);
        let vfour = f32x4_splat(4.0);
        let mut vidx = f32x4_add(
            f32x4_splat((m_pad - 4) as f32),
            f32x4(0.0, 1.0, 2.0, 3.0),
        );
        let mut vssr = f32x4_splat(0.0);
        let mut k = m_pad;
        while k > 0 {
            k -= 4;
            let kk = k.min(FIELD_CAND_CAP - 4);
            let vrk = f32x4(ress_0[kk], ress_0[kk + 1], ress_0[kk + 2], ress_0[kk + 3]);
            let vin = f32x4_lt(vidx, vm);
            vssr = f32x4_add(vssr, v128_and(vrk, vin));
            let vux = f32x4_sub(f32x4(dxs[kk], dxs[kk + 1], dxs[kk + 2], dxs[kk + 3]), vmx);
            let vuy = f32x4_sub(f32x4(dys[kk], dys[kk + 1], dys[kk + 2], dys[kk + 3]), vmy);
            let vh = f32x4_add(
                vinv_m,
                f32x4_add(
                    f32x4_mul(vux, f32x4_add(f32x4_mul(vg00, vux), f32x4_mul(v2g01, vuy))),
                    f32x4_mul(vg11, f32x4_mul(vuy, vuy)),
                ),
            );
            let vomh = f32x4_sub(vone, vh);
            let vc_num = f32x4_mul(vrk, vh);
            let vc_den = f32x4_mul(vomh, vomh);
            let vpass = v128_and(
                vin,
                v128_and(
                    f32x4_lt(vh, v1_eps),
                    f32x4_ge(vc_num, f32x4_mul(vcook_rhs, vc_den)),
                ),
            );
            let bm = i32x4_bitmask(vpass) as u32;
            if bm != 0 {
                macro_rules! check_lane {
                    ($bit:expr, $lane:literal) => {
                        if (bm & $bit) != 0 {
                            let idx = k + $lane;
                            let r_k = f32x4_extract_lane::<$lane>(vrk);
                            let omh = f32x4_extract_lane::<$lane>(vomh);
                            let c_num = f32x4_extract_lane::<$lane>(vc_num);
                            let c_den = f32x4_extract_lane::<$lane>(vc_den);
                            if c_num * cook_best_den > cook_best_num * c_den {
                                cook_best_num = c_num;
                                cook_best_den = c_den;
                                cook_idx = idx;
                            }
                            if r_k * press_best_den > press_best_num * omh {
                                press_best_num = r_k;
                                press_best_den = omh;
                                press_idx = idx;
                            }
                        }
                    };
                }
                check_lane!(8, 3);
                check_lane!(4, 2);
                check_lane!(2, 1);
                check_lane!(1, 0);
            }
            vidx = f32x4_sub(vidx, vfour);
        }
        ssr_0 = hsum_f32x4(vssr);
    }
    if press_idx >= m || press_best_num < press_frac * ssr_0 * press_best_den {
        (m, m)
    } else {
        (cook_idx, press_idx)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_grad_v_loo_drop(
    cs: &mut ContinuumState,
    m: usize,
    mf: f32,
    inv_m: f32,
    mx: f32,
    my: f32,
    mu: f32,
    mv: f32,
    gxx: f32,
    gxy: f32,
    gyy: f32,
    inv_det0: f32,
    bxu: f32,
    byu: f32,
    bxv: f32,
    byv: f32,
    min_eig_px2: f32,
    med0_r2_in: f32,
    base_q80_r2: f32,
    r2_thresh_0: f32,
) -> f32 {
    let idx_q80 = ((m * 4) / 5).min(m - 2);
    let top_k = (m - 1) - idx_q80;
    let mut cutoff_r2 = base_q80_r2 * 0.3025;
    let mut min_drop = m;

    let s_down = mf / ((m - 1) as f32);
    let inv_m1 = 1.0 / ((m - 1) as f32);
    let req_eig1 = ((m - 1) as f32) * min_eig_px2;

    let g00_i = gyy * inv_det0;
    let two_g01_i = -2.0 * gxy * inv_det0;
    let g11_i = gxx * inv_det0;
    let cook_rhs = 0.19f32 * base_q80_r2;

    let (cook_idx, press_idx) = find_cook_press_outlier_indices(
        cs, m, inv_m, mx, my, g00_i, two_g01_i, g11_i, cook_rhs, 0.50,
    );
    if press_idx >= m {
        return med0_r2_in;
    }

    let dxs = &cs.cand_dx;
    let dys = &cs.cand_dy;
    let dus = &cs.cand_du;
    let dvs = &cs.cand_dv;

    let mut best_cu = 0.0f32;
    let mut best_cv = 0.0f32;
    let mut best_dudx = 0.0f32;
    let mut best_dudy = 0.0f32;
    let mut best_dvdx = 0.0f32;
    let mut best_dvdy = 0.0f32;

    let raw_drops = [cook_idx, press_idx];
    let mut di = 0usize;
    while di < 2 {
        let cand_drop = raw_drops[di];
        if cand_drop < m && (di == 0 || cand_drop != raw_drops[0]) {
            let cd = cand_drop.min(FIELD_CAND_CAP - 1);
            let ux = dxs[cd] - mx;
            let uy = dys[cd] - my;
            let uu = dus[cd] - mu;
            let uv = dvs[cd] - mv;
            let sux = s_down * ux;
            let suy = s_down * uy;
            let lgxx = gxx - sux * ux;
            let lgxy = gxy - sux * uy;
            let lgyy = gyy - suy * uy;
            if is_well_conditioned_cov2_f32(lgxx, lgxy, lgyy, req_eig1) {
                let ldet = lgxx * lgyy - lgxy * lgxy;
                if ldet > 1e-10 {
                    let lbxu = bxu - sux * uu;
                    let lbyu = byu - suy * uu;
                    let lbxv = bxv - sux * uv;
                    let lbyv = byv - suy * uv;
                    let linv = 1.0 / ldet;
                    let ldudx = (lgyy * lbxu - lgxy * lbyu) * linv;
                    let ldudy = (-lgxy * lbxu + lgxx * lbyu) * linv;
                    let ldvdx = (lgyy * lbxv - lgxy * lbyv) * linv;
                    let ldvdy = (-lgxy * lbxv + lgxx * lbyv) * linv;
                    if ldudx.is_finite()
                        && ldudy.is_finite()
                        && ldvdx.is_finite()
                        && ldvdy.is_finite()
                    {
                        let lxb = mx - ux * inv_m1;
                        let lyb = my - uy * inv_m1;
                        let lcu = (mu - uu * inv_m1) - ldudx * lxb - ldudy * lyb;
                        let lcv = (mv - uv * inv_m1) - ldvdx * lxb - ldvdy * lyb;
                        let (ok_loo, new_cut) = loo_eval_top_k_below_cutoff(
                            dxs, dys, dus, dvs, m, cand_drop, lcu, ldudx, ldudy, lcv, ldvdx,
                            ldvdy, top_k, cutoff_r2,
                        );
                        if ok_loo && new_cut < cutoff_r2 {
                            cutoff_r2 = new_cut;
                            min_drop = cand_drop;
                            best_cu = lcu;
                            best_cv = lcv;
                            best_dudx = ldudx;
                            best_dudy = ldudy;
                            best_dvdx = ldvdx;
                            best_dvdy = ldvdy;
                            if cutoff_r2 <= r2_thresh_0 {
                                break;
                            }
                        }
                    }
                }
            }
        }
        di += 1;
    }

    if min_drop < m {
        let m = m.min(FIELD_CAND_CAP);
        let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
        let ress = &mut cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vcu = f32x4_splat(best_cu);
            let vcv = f32x4_splat(best_cv);
            let vdudx = f32x4_splat(best_dudx);
            let vdudy = f32x4_splat(best_dudy);
            let vdvdx = f32x4_splat(best_dvdx);
            let vdvdy = f32x4_splat(best_dvdy);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(vdu, f32x4_add(vcu, f32x4_add(f32x4_mul(vdudx, vdx), f32x4_mul(vdudy, vdy))));
                let vry = f32x4_sub(vdv, f32x4_add(vcv, f32x4_add(f32x4_mul(vdvdx, vdx), f32x4_mul(vdvdy, vdy))));
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                ress[k] = f32x4_extract_lane::<0>(vr2);
                ress[k + 1] = f32x4_extract_lane::<1>(vr2);
                ress[k + 2] = f32x4_extract_lane::<2>(vr2);
                ress[k + 3] = f32x4_extract_lane::<3>(vr2);
                bi += 1;
            }
        }
        return median_and_q80_r2(&cs.ftle_cands_res, m, false).0;
    }
    med0_r2_in
}

#[inline(never)]
fn try_solve_cell_grad_v(
    cs: &mut ContinuumState,
    pm: &mut GradPrefixMoments,
    cand_n: usize,
    xc: f32,
    yc: f32,
    radius_px: f32,
    min_eig_px2: f32,
    q: f32,
    dt_sec: f32,
) -> (bool, u8, f32, f32, f32, f32, f32) {
    let cand_n = cand_n.min(FIELD_CAND_CAP);
    if cand_n < 4 {
        return (false, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let r2 = radius_px * radius_px;
    let inv_r2 = 1.0 / r2.max(1e-6);
    let u_ref = pm.u_ref;
    let v_ref = pm.v_ref;

    let m = fill_grad_prefix_moments(cs, pm, cand_n, xc, yc);
    if m < 4 {
        return (false, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let sx = pm.sx;
    let sy = pm.sy;
    let su = pm.su;
    let sv = pm.sv;
    let mf = m as f32;
    let inv_m = 1.0 / mf;
    let mx = sx * inv_m;
    let my = sy * inv_m;
    let mu = su * inv_m;
    let mv = sv * inv_m;
    let gxx = pm.sxx - sx * mx;
    let gxy = pm.sxy - sx * my;
    let gyy = pm.syy - sy * my;
    if !is_well_conditioned_cov2_f32(gxx, gxy, gyy, mf * min_eig_px2) {
        return (false, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let det0 = gxx * gyy - gxy * gxy;
    if det0 <= 1e-10 {
        return (false, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let bxu = pm.sxu - sx * mu;
    let byu = pm.syu - sy * mu;
    let bxv = pm.sxv - sx * mv;
    let byv = pm.syv - sy * mv;
    let inv_det0 = 1.0 / det0;
    let dudx0 = (gyy * bxu - gxy * byu) * inv_det0;
    let dudy0 = (-gxy * bxu + gxx * byu) * inv_det0;
    let dvdx0 = (gyy * bxv - gxy * byv) * inv_det0;
    let dvdy0 = (-gxy * bxv + gxx * byv) * inv_det0;
    if !dudx0.is_finite() || !dudy0.is_finite() || !dvdx0.is_finite() || !dvdy0.is_finite() {
        return (false, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let xb0 = mx;
    let yb0 = my;
    let ub0 = mu;
    let vb0 = mv;
    let cu0 = ub0 - dudx0 * xb0 - dudy0 * yb0;
    let cv0 = vb0 - dvdx0 * xb0 - dvdy0 * yb0;

    let max_r2_0: f32;
    {
        let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
        let ress = &mut cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vcu0 = f32x4_splat(cu0);
            let vcv0 = f32x4_splat(cv0);
            let vdudx0 = f32x4_splat(dudx0);
            let vdudy0 = f32x4_splat(dudy0);
            let vdvdx0 = f32x4_splat(dvdx0);
            let vdvdy0 = f32x4_splat(dvdy0);
            let vm = f32x4_splat(m as f32);
            let vfour = f32x4_splat(4.0);
            let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
            let mut vmax_r2 = f32x4_splat(0.0);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(
                    vdu,
                    f32x4_add(vcu0, f32x4_add(f32x4_mul(vdudx0, vdx), f32x4_mul(vdudy0, vdy))),
                );
                let vry = f32x4_sub(
                    vdv,
                    f32x4_add(vcv0, f32x4_add(f32x4_mul(vdvdx0, vdx), f32x4_mul(vdvdy0, vdy))),
                );
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                let vin = f32x4_lt(vidx, vm);
                vmax_r2 = f32x4_pmax(vmax_r2, v128_and(vr2, vin));
                ress[k] = f32x4_extract_lane::<0>(vr2);
                ress[k + 1] = f32x4_extract_lane::<1>(vr2);
                ress[k + 2] = f32x4_extract_lane::<2>(vr2);
                ress[k + 3] = f32x4_extract_lane::<3>(vr2);
                vidx = f32x4_add(vidx, vfour);
                bi += 1;
            }
            max_r2_0 = hmax_f32x4(vmax_r2);
        }
    }
    if max_r2_0 <= (0.75 * q) * (0.75 * q) {
        let u_c = u_ref + cu0;
        let v_c = v_ref + cv0;
        return (
            true,
            GRAD_FLAG_PASS2,
            u_c,
            v_c,
            dvdx0 - dudy0,
            dudx0 + dvdy0,
            m as f32,
        );
    }
    if m == 4 {
        let dt_safe = dt_sec.max(1e-3);
        let mut ss_res0 = 0.0f32;
        let mut ss_tot0 = 0.0f32;
        let mut k = 0usize;
        while k < 4 {
            ss_res0 += cs.ftle_cands_res[k];
            let du = cs.cand_du[k] - mu;
            let dv = cs.cand_dv[k] - mv;
            ss_tot0 += du * du + dv * dv;
            k += 1;
        }
        let rmse_v = wasm_sqrtf(ss_res0 * 0.25);
        if !(rmse_v <= 2.0 * q || (rmse_v <= (0.65 * q / dt_safe) && ss_res0 <= 0.12 * ss_tot0)) {
            return (false, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        k = 0;
        while k < 4 {
            let t = 1.0 - cs.ftle_cands_d2[k] * inv_r2;
            cs.ftle_cands_w[k] = (t * t).max(1e-4);
            k += 1;
        }
        let (ok_f, ub, vb, xb, yb, dudx, dudy, dvdx, dvdy, w_sum) =
            fit_cand_grad_weighted_2x2(cs, m, false, min_eig_px2);
        let (u_b, v_b, x_b, y_b, ux, uy, vx, vy, ws) = if ok_f {
            (ub, vb, xb, yb, dudx, dudy, dvdx, dvdy, w_sum)
        } else {
            (ub0, vb0, xb0, yb0, dudx0, dudy0, dvdx0, dvdy0, 4.0)
        };
        let u_c = u_ref + u_b - ux * x_b - uy * y_b;
        let v_c = v_ref + v_b - vx * x_b - vy * y_b;
        return (true, GRAD_FLAG_PASS2, u_c, v_c, vx - uy, ux + vy, ws);
    }

    // Clean-patch fast path: all OLS residuals <= 1.75*q (< 2.0*q hard trim floor) -> skip LOO & IRLS
    if max_r2_0 <= (1.75 * q) * (1.75 * q) {
        let d2s = &cs.ftle_cands_d2[..FIELD_CAND_CAP];
        let ws = &mut cs.ftle_cands_w[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vone = f32x4_splat(1.0);
            let vhalf_inv_r2 = f32x4_splat(0.5 * inv_r2);
            let vmin_w = f32x4_splat(0.25);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vd2 = f32x4(d2s[k], d2s[k + 1], d2s[k + 2], d2s[k + 3]);
                let vw = f32x4_max(f32x4_sub(vone, f32x4_mul(vd2, vhalf_inv_r2)), vmin_w);
                ws[k] = f32x4_extract_lane::<0>(vw);
                ws[k + 1] = f32x4_extract_lane::<1>(vw);
                ws[k + 2] = f32x4_extract_lane::<2>(vw);
                ws[k + 3] = f32x4_extract_lane::<3>(vw);
                bi += 1;
            }
        }
        let (ok_f, ub, vb, xb, yb, dudx, dudy, dvdx, dvdy, w_sum) =
            fit_cand_grad_weighted_2x2(cs, m, false, min_eig_px2);
        if !ok_f {
            return (false, GRAD_FLAG_PASS2, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let cu_f = ub - dudx * xb - dudy * yb;
        let cv_f = vb - dvdx * xb - dvdy * yb;
        return (
            true,
            GRAD_FLAG_PASS2,
            u_ref + cu_f,
            v_ref + cv_f,
            dvdx - dudy,
            dudx + dvdy,
            w_sum,
        );
    }

    try_solve_cell_grad_v_fallback(
        cs,
        m,
        u_ref,
        v_ref,
        mx,
        my,
        mu,
        mv,
        gxx,
        gxy,
        gyy,
        inv_det0,
        bxu,
        byu,
        bxv,
        byv,
        inv_r2,
        min_eig_px2,
        q,
        dt_sec,
        max_r2_0,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_solve_cell_grad_v_fallback(
    cs: &mut ContinuumState,
    m: usize,
    u_ref: f32,
    v_ref: f32,
    mx: f32,
    my: f32,
    mu: f32,
    mv: f32,
    gxx: f32,
    gxy: f32,
    gyy: f32,
    inv_det0: f32,
    bxu: f32,
    byu: f32,
    bxv: f32,
    byv: f32,
    inv_r2: f32,
    min_eig_px2: f32,
    q: f32,
    dt_sec: f32,
    max_r2_0: f32,
) -> (bool, u8, f32, f32, f32, f32, f32) {
    let mf = m as f32;
    let inv_m = 1.0 / mf;
    let dt_safe = dt_sec.max(1e-3);
    let vmin = 2.0 * q;
    let vmin2 = vmin * vmin;
    let vmax = (1.25 * q / dt_safe).max(2.5 * q);
    let vmax2 = vmax * vmax;
    let med_lim = (1.0 * q / dt_safe).max(2.5 * q);
    let med_lim2 = med_lim * med_lim;
    let r2_thresh_0 = (2.5 * q) * (2.5 * q);
    let s0_grad2 = (q * q) / 2.25;
    let two_vmax2 = 2.0 * vmax2;
    let mut cnt0_le_t = 0usize;
    let mut cnt0_le_s0 = 0usize;
    let mut cnt0_le_medlim = 0usize;
    let mut cnt0_le_vmax2 = 0usize;
    let mut cnt0_le_2vmax2 = 0usize;
    {
        let ress = &cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vt = f32x4_splat(r2_thresh_0);
            let vs0 = f32x4_splat(s0_grad2);
            let v2vmax2 = f32x4_splat(two_vmax2);
            let vvmax2 = f32x4_splat(vmax2);
            let vmed_lim2 = f32x4_splat(med_lim2);
            let vm = f32x4_splat(m as f32);
            let vfour = f32x4_splat(4.0);
            let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vr2 = f32x4(ress[k], ress[k + 1], ress[k + 2], ress[k + 3]);
                let valid = i32x4_bitmask(f32x4_lt(vidx, vm)) as u32;
                cnt0_le_t +=
                    ((i32x4_bitmask(f32x4_le(vr2, vt)) as u32) & valid).count_ones() as usize;
                cnt0_le_s0 +=
                    ((i32x4_bitmask(f32x4_le(vr2, vs0)) as u32) & valid).count_ones() as usize;
                cnt0_le_2vmax2 += ((i32x4_bitmask(f32x4_le(vr2, v2vmax2)) as u32) & valid)
                    .count_ones() as usize;
                cnt0_le_vmax2 += ((i32x4_bitmask(f32x4_le(vr2, vvmax2)) as u32) & valid)
                    .count_ones() as usize;
                cnt0_le_medlim += ((i32x4_bitmask(f32x4_le(vr2, vmed_lim2)) as u32) & valid)
                    .count_ones() as usize;
                vidx = f32x4_add(vidx, vfour);
                bi += 1;
            }
        }
    }
    let mut k;

    if (m >= 5 && cnt0_le_2vmax2 * 2 <= m) || (m >= 6 && cnt0_le_vmax2 * 2 <= m) {
        return (false, GRAD_FLAG_INCOHERENT_CORE, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let idx_q80 = ((m * 4) / 5).min(m - 2);
    let can_loo = cnt0_le_t <= (m >> 1)
        && max_r2_0 > r2_thresh_0
        && cnt0_le_t <= idx_q80;
    let inv_s2 = if !can_loo && cnt0_le_s0 > (m >> 1) {
        1.0 / (q * q)
    } else {
        if m >= 6 && (cnt0_le_medlim <= ((m - 1) >> 1) || cnt0_le_vmax2 * 5 < m * 4) {
            let has_press = can_loo
                && has_dominant_press_outlier(
                    cs, m, mx, my, gxx, gxy, gyy, inv_det0, inv_m, 0.50,
                );
            if !has_press {
                return (false, GRAD_FLAG_INCOHERENT_CORE, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
        }
        let (mut med0_r2, base_q80_r2) = median_and_q80_r2(&cs.ftle_cands_res, m, can_loo);
        if can_loo && base_q80_r2 > r2_thresh_0 {
            med0_r2 = try_grad_v_loo_drop(
                cs,
                m,
                mf,
                inv_m,
                mx,
                my,
                mu,
                mv,
                gxx,
                gxy,
                gyy,
                inv_det0,
                bxu,
                byu,
                bxv,
                byv,
                min_eig_px2,
                med0_r2,
                base_q80_r2,
                r2_thresh_0,
            );
        }
        if m >= 6 && med0_r2 > med_lim2 {
            return (false, GRAD_FLAG_INCOHERENT_CORE, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        1.0 / (2.25 * med0_r2).max(q * q)
    };

    // IRLS Cauchy reweighting + >= 80% inlier hard trim
    {
        let ress = &cs.ftle_cands_res[..FIELD_CAND_CAP];
        let ws = &mut cs.ftle_cands_w[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vone = f32x4_splat(1.0);
            let vinv_s2 = f32x4_splat(inv_s2);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vr2 = f32x4(ress[k], ress[k + 1], ress[k + 2], ress[k + 3]);
                let vw = f32x4_div(vone, f32x4_add(vone, f32x4_mul(vr2, vinv_s2)));
                ws[k] = f32x4_extract_lane::<0>(vw);
                ws[k + 1] = f32x4_extract_lane::<1>(vw);
                ws[k + 2] = f32x4_extract_lane::<2>(vw);
                ws[k + 3] = f32x4_extract_lane::<3>(vw);
                bi += 1;
            }
        }
    }

    let (ok1, ub1, vb1, xb1, yb1, dudx1, dudy1, dvdx1, dvdy1, _) =
        fit_cand_grad_weighted_2x2(cs, m, false, min_eig_px2);
    if !ok1 {
        return (false, 0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let cu1 = ub1 - dudx1 * xb1 - dudy1 * yb1;
    let cv1 = vb1 - dvdx1 * xb1 - dvdy1 * yb1;
    const K_SIGMA2: f32 = (3.5 * 1.4826) * (3.5 * 1.4826);
    let v_lo2 = vmin2 / K_SIGMA2;
    let v_hi2 = vmax2 / K_SIGMA2;
    let max_r2_1: f32;
    let mut cnt_le_vmin = 0usize;
    let mut cnt_le_vmax = 0usize;
    let mut cnt1_le_vlo = 0usize;
    let mut cnt1_lt_vhi = 0usize;
    {
        let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
        let ress = &mut cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vcu1 = f32x4_splat(cu1);
            let vcv1 = f32x4_splat(cv1);
            let vdudx1 = f32x4_splat(dudx1);
            let vdudy1 = f32x4_splat(dudy1);
            let vdvdx1 = f32x4_splat(dvdx1);
            let vdvdy1 = f32x4_splat(dvdy1);
            let vhi2 = f32x4_splat(v_hi2);
            let vlo2 = f32x4_splat(v_lo2);
            let vvmax2 = f32x4_splat(vmax2);
            let vvmin2 = f32x4_splat(vmin2);
            let vm = f32x4_splat(m as f32);
            let vfour = f32x4_splat(4.0);
            let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
            let mut vmax_r2 = f32x4_splat(0.0);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(
                    vdu,
                    f32x4_add(vcu1, f32x4_add(f32x4_mul(vdudx1, vdx), f32x4_mul(vdudy1, vdy))),
                );
                let vry = f32x4_sub(
                    vdv,
                    f32x4_add(vcv1, f32x4_add(f32x4_mul(vdvdx1, vdx), f32x4_mul(vdvdy1, vdy))),
                );
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                let vin = f32x4_lt(vidx, vm);
                let valid = i32x4_bitmask(vin) as u32;
                vmax_r2 = f32x4_pmax(vmax_r2, v128_and(vr2, vin));
                ress[k] = f32x4_extract_lane::<0>(vr2);
                ress[k + 1] = f32x4_extract_lane::<1>(vr2);
                ress[k + 2] = f32x4_extract_lane::<2>(vr2);
                ress[k + 3] = f32x4_extract_lane::<3>(vr2);
                cnt1_lt_vhi +=
                    ((i32x4_bitmask(f32x4_lt(vr2, vhi2)) as u32) & valid).count_ones() as usize;
                cnt1_le_vlo +=
                    ((i32x4_bitmask(f32x4_le(vr2, vlo2)) as u32) & valid).count_ones() as usize;
                cnt_le_vmax += ((i32x4_bitmask(f32x4_le(vr2, vvmax2)) as u32) & valid)
                    .count_ones() as usize;
                cnt_le_vmin += ((i32x4_bitmask(f32x4_le(vr2, vvmin2)) as u32) & valid)
                    .count_ones() as usize;
                vidx = f32x4_add(vidx, vfour);
                bi += 1;
            }
            max_r2_1 = hmax_f32x4(vmax_r2);
        }
    }
    if cnt_le_vmax < 4 || cnt_le_vmax * 5 < m * 4 {
        let flag = if m >= 6 || (m >= 5 && (cnt_le_vmax * 5 < m * 3 || cnt_le_vmax < 4)) {
            GRAD_FLAG_INCOHERENT_CORE
        } else {
            0
        };
        return (false, flag, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let thresh_v2 = if max_r2_1 <= vmin2 || cnt_le_vmax == cnt_le_vmin || cnt1_le_vlo > (m >> 1) {
        vmin2
    } else if cnt1_lt_vhi <= ((m - 1) >> 1) {
        vmax2
    } else {
        let med1_r2 = median_and_q80_r2(&cs.ftle_cands_res, m, false).0;
        (vmin2.max(K_SIGMA2 * med1_r2)).min(vmax2)
    };

    let mut inliers = 0usize;
    {
        let ress = &cs.ftle_cands_res[..FIELD_CAND_CAP];
        let d2s = &cs.ftle_cands_d2[..FIELD_CAND_CAP];
        let ws = &mut cs.ftle_cands_w[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vone = f32x4_splat(1.0);
            let vhalf_inv_r2 = f32x4_splat(0.5 * inv_r2);
            let vmin_w = f32x4_splat(0.25);
            if max_r2_1 <= thresh_v2 {
                let mut bi = 0usize;
                while bi < blk {
                    let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                    let vd2 = f32x4(d2s[k], d2s[k + 1], d2s[k + 2], d2s[k + 3]);
                    let vw = f32x4_max(f32x4_sub(vone, f32x4_mul(vd2, vhalf_inv_r2)), vmin_w);
                    ws[k] = f32x4_extract_lane::<0>(vw);
                    ws[k + 1] = f32x4_extract_lane::<1>(vw);
                    ws[k + 2] = f32x4_extract_lane::<2>(vw);
                    ws[k + 3] = f32x4_extract_lane::<3>(vw);
                    bi += 1;
                }
                inliers = m;
            } else {
                let vth2 = f32x4_splat(thresh_v2);
                let vm = f32x4_splat(m as f32);
                let vfour = f32x4_splat(4.0);
                let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
                let mut bi = 0usize;
                while bi < blk {
                    let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                    let vr2 = f32x4(ress[k], ress[k + 1], ress[k + 2], ress[k + 3]);
                    let vd2 = f32x4(d2s[k], d2s[k + 1], d2s[k + 2], d2s[k + 3]);
                    let vw_d = f32x4_max(f32x4_sub(vone, f32x4_mul(vd2, vhalf_inv_r2)), vmin_w);
                    let keep = v128_and(f32x4_le(vr2, vth2), f32x4_lt(vidx, vm));
                    let vw = v128_and(vw_d, keep);
                    ws[k] = f32x4_extract_lane::<0>(vw);
                    ws[k + 1] = f32x4_extract_lane::<1>(vw);
                    ws[k + 2] = f32x4_extract_lane::<2>(vw);
                    ws[k + 3] = f32x4_extract_lane::<3>(vw);
                    inliers += (i32x4_bitmask(keep) as u32).count_ones() as usize;
                    vidx = f32x4_add(vidx, vfour);
                    bi += 1;
                }
            }
        }
    }
    if inliers < 4 || inliers * 5 < m * 4 {
        let flag = if m >= 6 { GRAD_FLAG_INCOHERENT_CORE } else { 0 };
        return (false, flag, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let (ok_f, ub, vb, xb, yb, dudx, dudy, dvdx, dvdy, w_sum) =
        fit_cand_grad_weighted_2x2(cs, m, true, min_eig_px2);
    if !ok_f {
        return (false, GRAD_FLAG_PASS2, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let cu_f = ub - dudx * xb - dudy * yb;
    let cv_f = vb - dvdx * xb - dvdy * yb;
    let in_ss_res: f32;
    {
        let ws = &cs.ftle_cands_w[..FIELD_CAND_CAP];
        let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vcu = f32x4_splat(cu_f);
            let vcv = f32x4_splat(cv_f);
            let vdudx = f32x4_splat(dudx);
            let vdudy = f32x4_splat(dudy);
            let vdvdx = f32x4_splat(dvdx);
            let vdvdy = f32x4_splat(dvdy);
            let vz = f32x4_splat(0.0);
            let vm = f32x4_splat(m as f32);
            let vfour = f32x4_splat(4.0);
            let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
            let mut vss = f32x4_splat(0.0);
            let all_in = inliers == m;
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(
                    vdu,
                    f32x4_add(vcu, f32x4_add(f32x4_mul(vdudx, vdx), f32x4_mul(vdudy, vdy))),
                );
                let vry = f32x4_sub(
                    vdv,
                    f32x4_add(vcv, f32x4_add(f32x4_mul(vdvdx, vdx), f32x4_mul(vdvdy, vdy))),
                );
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                let vin = f32x4_lt(vidx, vm);
                let vact = if all_in {
                    vin
                } else {
                    let vw = f32x4(ws[k], ws[k + 1], ws[k + 2], ws[k + 3]);
                    v128_and(f32x4_gt(vw, vz), vin)
                };
                vss = f32x4_add(vss, v128_and(vr2, vact));
                vidx = f32x4_add(vidx, vfour);
                bi += 1;
            }
            in_ss_res = hsum_f32x4(vss);
        }
    }
    let inv_in = 1.0 / (inliers as f32);
    let max_rmse1 = 2.5 * q;
    if in_ss_res * inv_in > max_rmse1 * max_rmse1 {
        let max_rmse2 = 0.85 * q / dt_safe;
        let inc_flag = if m >= 6 {
            GRAD_FLAG_PASS2 | GRAD_FLAG_INCOHERENT_CORE
        } else {
            GRAD_FLAG_PASS2
        };
        if in_ss_res * inv_in > max_rmse2 * max_rmse2 {
            return (false, inc_flag, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let mut in_u_sum = 0.0f32;
        let mut in_v_sum = 0.0f32;
        let ws = &cs.ftle_cands_w[..m];
        let dus = &cs.cand_du[..m];
        let dvs = &cs.cand_dv[..m];
        k = 0;
        while k < m {
            if ws[k] > 0.0 {
                in_u_sum += dus[k];
                in_v_sum += dvs[k];
            }
            k += 1;
        }
        let mu_u = in_u_sum * inv_in;
        let mu_v = in_v_sum * inv_in;
        let mut in_ss_tot = 0.0f32;
        k = 0;
        while k < m {
            if ws[k] > 0.0 {
                let du = dus[k] - mu_u;
                let dv = dvs[k] - mu_v;
                in_ss_tot += du * du + dv * dv;
            }
            k += 1;
        }
        if in_ss_res > 0.18 * in_ss_tot {
            return (false, inc_flag, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }

    let u_c = u_ref + cu_f;
    let v_c = v_ref + cv_f;
    let vort = dvdx - dudy;
    let div = dudx + dvdy;
    (true, GRAD_FLAG_PASS2, u_c, v_c, vort, div, w_sum)
}

#[inline(never)]
fn fit_cand_ftle_weighted_2x2(
    cs: &ContinuumState,
    m: usize,
    use_active: bool,
    min_eig_px2: f32,
) -> (bool, f32, f32, f32, f32, f32, f32, f32, f32) {
    let m = m.min(FIELD_CAND_CAP);
    let ws = &cs.ftle_cands_w[..m];
    let dxs = &cs.cand_dx[..m];
    let dys = &cs.cand_dy[..m];
    let dus = &cs.cand_du[..m];
    let dvs = &cs.cand_dv[..m];

    let mut cnt = 0usize;

    let (
        w_sum,
        sx0,
        sy0,
        sxt,
        syt,
        sx0x0,
        sx0y0,
        sy0y0,
        sxtxt,
        sxtyt,
        sytyt,
        sb11,
        sb12,
        sb21,
        sb22,
    ) = {
        {
            use crate::simd::*;
            let mut v_lin = f32x4_splat(0.0);
            let mut v_x0 = f32x4_splat(0.0);
            let mut v_y0 = f32x4_splat(0.0);
            let mut v_tt = f32x4_splat(0.0);
            let mut k = 0usize;
            while k < m {
                let w = ws[k];
                if !use_active || w > 0.0 {
                    let dx0 = dxs[k];
                    let dy0 = dys[k];
                    let dxt = dus[k];
                    let dyt = dvs[k];
                    let v_pts = f32x4(dx0, dy0, dxt, dyt);
                    v_lin = f32x4_add(v_lin, f32x4_mul(f32x4_splat(w), v_pts));
                    v_x0 = f32x4_add(v_x0, f32x4_mul(f32x4_splat(w * dx0), v_pts));
                    let vy0_terms = f32x4_replace_lane::<0>(
                        f32x4_mul(f32x4_splat(w * dy0), v_pts),
                        w,
                    );
                    v_y0 = f32x4_add(v_y0, vy0_terms);
                    v_tt = f32x4_add(
                        v_tt,
                        f32x4_mul(f32x4_splat(w), f32x4(dxt * dxt, dxt * dyt, dyt * dyt, 0.0)),
                    );
                    cnt += 1;
                }
                k += 1;
            }
            (
                f32x4_extract_lane::<0>(v_y0),
                f32x4_extract_lane::<0>(v_lin),
                f32x4_extract_lane::<1>(v_lin),
                f32x4_extract_lane::<2>(v_lin),
                f32x4_extract_lane::<3>(v_lin),
                f32x4_extract_lane::<0>(v_x0),
                f32x4_extract_lane::<1>(v_x0),
                f32x4_extract_lane::<1>(v_y0),
                f32x4_extract_lane::<0>(v_tt),
                f32x4_extract_lane::<1>(v_tt),
                f32x4_extract_lane::<2>(v_tt),
                f32x4_extract_lane::<2>(v_x0),
                f32x4_extract_lane::<2>(v_y0),
                f32x4_extract_lane::<3>(v_x0),
                f32x4_extract_lane::<3>(v_y0),
            )
        }
    };
    if cnt < MIN_RIGID_MEMBERS || w_sum <= 1e-6 {
        return (false, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    }

    let inv_w = 1.0 / w_sum;
    let mx0 = sx0 * inv_w;
    let my0 = sy0 * inv_w;
    let mxt = sxt * inv_w;
    let myt = syt * inv_w;

    let gxx = sx0x0 - sx0 * mx0;
    let gxy = sx0y0 - sx0 * my0;
    let gyy = sy0y0 - sy0 * my0;
    let txx = sxtxt - sxt * mxt;
    let txy = sxtyt - sxt * myt;
    let tyy = sytyt - syt * myt;
    let req_eig = w_sum * min_eig_px2;
    if !is_well_conditioned_cov2_f32(gxx, gxy, gyy, req_eig)
        || !is_well_conditioned_cov2_f32(txx, txy, tyy, req_eig)
    {
        return (false, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    }

    let det0 = gxx * gyy - gxy * gxy;
    if det0 <= 1e-10 {
        return (false, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    }
    let b11 = sb11 - sxt * mx0;
    let b12 = sb12 - sxt * my0;
    let b21 = sb21 - syt * mx0;
    let b22 = sb22 - syt * my0;
    let inv_det = 1.0 / det0;
    let f11 = (b11 * gyy - b12 * gxy) * inv_det;
    let f12 = (-b11 * gxy + b12 * gxx) * inv_det;
    let f21 = (b21 * gyy - b22 * gxy) * inv_det;
    let f22 = (-b21 * gxy + b22 * gxx) * inv_det;
    if !f11.is_finite() || !f12.is_finite() || !f21.is_finite() || !f22.is_finite() {
        return (false, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    }
    (true, f11, f12, f21, f22, mx0, my0, mxt, myt)
}

/// Fits robust 2-pass IRLS/trimmed deformation gradient `F = dX_t / dX_0` over synchronized
/// anchors in one anchor-space cell neighborhood for a specified `bank` (`1` or `2`).
/// Uses packed `age_idx = packed >> 11` carried from the bank CSR grid, eliminating scattered
/// lifecycle/anchor-frame lookups in the hot inner loop.
#[derive(Clone, Copy)]
struct FtlePrefixMoments {
    prev_k: usize,
    m: usize,
    x0_ref: f32,
    y0_ref: f32,
    xt_ref: f32,
    yt_ref: f32,
    sx0: f32,
    sy0: f32,
    sxt: f32,
    syt: f32,
    sx0x0: f32,
    sx0y0: f32,
    sy0y0: f32,
    sxtxt: f32,
    sxtyt: f32,
    sytyt: f32,
    sb11: f32,
    sb12: f32,
    sb21: f32,
    sb22: f32,
}

impl FtlePrefixMoments {
    #[inline(always)]
    fn zeroed() -> Self {
        Self {
            prev_k: 0,
            m: 0,
            x0_ref: 0.0,
            y0_ref: 0.0,
            xt_ref: 0.0,
            yt_ref: 0.0,
            sx0: 0.0,
            sy0: 0.0,
            sxt: 0.0,
            syt: 0.0,
            sx0x0: 0.0,
            sx0y0: 0.0,
            sy0y0: 0.0,
            sxtxt: 0.0,
            sxtyt: 0.0,
            sytyt: 0.0,
            sb11: 0.0,
            sb12: 0.0,
            sb21: 0.0,
            sb22: 0.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_ftle_loo_drop(
    cs: &mut ContinuumState,
    m: usize,
    mf: f32,
    inv_m: f32,
    mx0: f32,
    my0: f32,
    mxt: f32,
    myt: f32,
    gxx: f32,
    gxy: f32,
    gyy: f32,
    txx: f32,
    txy: f32,
    tyy: f32,
    b11: f32,
    b12: f32,
    b21: f32,
    b22: f32,
    inv_det0: f32,
    min_eig_px2: f32,
    idx_q80: usize,
    med0_r2_in: f32,
    base_q80_r2: f32,
    floor2: f32,
) -> (bool, f32) {
    let top_k = (m - 1) - idx_q80;
    let mut cutoff_r2 = base_q80_r2 * 0.3025;
    let mut min_drop = m;

    let s_down = mf / ((m - 1) as f32);
    let inv_m1 = 1.0 / ((m - 1) as f32);
    let req_eig1 = ((m - 1) as f32) * min_eig_px2;
    let g00_i32 = gyy * inv_det0;
    let two_g01_i32 = -2.0 * gxy * inv_det0;
    let g11_i32 = gxx * inv_det0;
    let cook_rhs32 = 0.19f32 * base_q80_r2;

    let (cook_idx, press_idx) = find_cook_press_outlier_indices(
        cs, m, inv_m, mx0, my0, g00_i32, two_g01_i32, g11_i32, cook_rhs32, 0.45,
    );

    let dxs = &cs.cand_dx;
    let dys = &cs.cand_dy;
    let dus = &cs.cand_du;
    let dvs = &cs.cand_dv;

    let mut best_c1 = 0.0f32;
    let mut best_c2 = 0.0f32;
    let mut best_f11 = 0.0f32;
    let mut best_f12 = 0.0f32;
    let mut best_f21 = 0.0f32;
    let mut best_f22 = 0.0f32;

    if press_idx < m {
        let raw_drops = [cook_idx, press_idx];
        let mut di = 0usize;
        while di < 2 {
            let cand_drop = raw_drops[di];
            if cand_drop < m && (di == 0 || cand_drop != raw_drops[0]) {
                let cd = cand_drop.min(FIELD_CAND_CAP - 1);
                let ux = dxs[cd] - mx0;
                let uy = dys[cd] - my0;
                let uu = dus[cd] - mxt;
                let uv = dvs[cd] - myt;
                let sux = s_down * ux;
                let suy = s_down * uy;
                let suu = s_down * uu;
                let suv = s_down * uv;
                let lgxx = gxx - sux * ux;
                let lgxy = gxy - sux * uy;
                let lgyy = gyy - suy * uy;
                let ltxx = txx - suu * uu;
                let ltxy = txy - suu * uv;
                let ltyy = tyy - suv * uv;
                if is_well_conditioned_cov2_f32(lgxx, lgxy, lgyy, req_eig1)
                    && is_well_conditioned_cov2_f32(ltxx, ltxy, ltyy, req_eig1)
                {
                    let ldet0 = lgxx * lgyy - lgxy * lgxy;
                    if ldet0 > 1e-10 {
                        let lb11 = b11 - suu * ux;
                        let lb12 = b12 - suu * uy;
                        let lb21 = b21 - suv * ux;
                        let lb22 = b22 - suv * uy;
                        let linv = 1.0 / ldet0;
                        let lf11 = (lb11 * lgyy - lb12 * lgxy) * linv;
                        let lf12 = (-lb11 * lgxy + lb12 * lgxx) * linv;
                        let lf21 = (lb21 * lgyy - lb22 * lgxy) * linv;
                        let lf22 = (-lb21 * lgxy + lb22 * lgxx) * linv;
                        if lf11.is_finite()
                            && lf12.is_finite()
                            && lf21.is_finite()
                            && lf22.is_finite()
                        {
                            let lx0 = mx0 - ux * inv_m1;
                            let ly0 = my0 - uy * inv_m1;
                            let lxt = mxt - uu * inv_m1;
                            let lyt = myt - uv * inv_m1;
                            let lc1 = lxt - lf11 * lx0 - lf12 * ly0;
                            let lc2 = lyt - lf21 * lx0 - lf22 * ly0;
                            let (ok_loo, new_cut) = loo_eval_top_k_below_cutoff(
                                dxs, dys, dus, dvs, m, cand_drop, lc1, lf11, lf12, lc2, lf21, lf22,
                                top_k, cutoff_r2,
                            );
                            if ok_loo && new_cut < cutoff_r2 {
                                cutoff_r2 = new_cut;
                                min_drop = cand_drop;
                                best_c1 = lc1;
                                best_c2 = lc2;
                                best_f11 = lf11;
                                best_f12 = lf12;
                                best_f21 = lf21;
                                best_f22 = lf22;
                                if cutoff_r2 <= floor2 {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            di += 1;
        }
    }

    if min_drop < m {
        let m = m.min(FIELD_CAND_CAP);
        let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
        let ress = &mut cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vc1 = f32x4_splat(best_c1);
            let vc2 = f32x4_splat(best_c2);
            let vf11 = f32x4_splat(best_f11);
            let vf12 = f32x4_splat(best_f12);
            let vf21 = f32x4_splat(best_f21);
            let vf22 = f32x4_splat(best_f22);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(vdu, f32x4_add(vc1, f32x4_add(f32x4_mul(vf11, vdx), f32x4_mul(vf12, vdy))));
                let vry = f32x4_sub(vdv, f32x4_add(vc2, f32x4_add(f32x4_mul(vf21, vdx), f32x4_mul(vf22, vdy))));
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                ress[k] = f32x4_extract_lane::<0>(vr2);
                ress[k + 1] = f32x4_extract_lane::<1>(vr2);
                ress[k + 2] = f32x4_extract_lane::<2>(vr2);
                ress[k + 3] = f32x4_extract_lane::<3>(vr2);
                bi += 1;
            }
        }
        return (true, median_and_q80_r2(&cs.ftle_cands_res, m, false).0);
    }
    (false, med0_r2_in)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_solve_cell_ftle(
    cs: &mut ContinuumState,
    fm: &mut FtlePrefixMoments,
    cand_end: usize,
    best_a: usize,
    all_same_age: bool,
    t_sec: f32,
    q: f32,
    min_eig_px2: f32,
    radius_px: f32,
    save_fm: bool,
) -> (bool, bool, bool, f32, usize, bool) {
    let cand_end = cand_end.min(FIELD_CAND_CAP);
    if cand_end < MIN_RIGID_MEMBERS {
        return (false, false, false, 0.0, 0, false);
    }
    let r2 = radius_px * radius_px;
    let inv_r2 = 1.0 / r2.max(1e-6);

    let mut m = fm.m.min(FIELD_CAND_CAP);
    let mut k = 0usize;
    let mut x0_ref = 0.0f32;
    let mut y0_ref = 0.0f32;
    let mut xt_ref = 0.0f32;
    let mut yt_ref = 0.0f32;
    let mut sx0 = 0.0f32;
    let mut sy0 = 0.0f32;
    let mut sxt = 0.0f32;
    let mut syt = 0.0f32;
    let mut sx0x0 = 0.0f32;
    let mut sx0y0 = 0.0f32;
    let mut sy0y0 = 0.0f32;
    let mut sxtxt = 0.0f32;
    let mut sxtyt = 0.0f32;
    let mut sytyt = 0.0f32;
    let mut sb11 = 0.0f32;
    let mut sb12 = 0.0f32;
    let mut sb21 = 0.0f32;
    let mut sb22 = 0.0f32;
    if m > 0 {
        k = fm.prev_k.min(cand_end);
        x0_ref = fm.x0_ref;
        y0_ref = fm.y0_ref;
        xt_ref = fm.xt_ref;
        yt_ref = fm.yt_ref;
        sx0 = fm.sx0;
        sy0 = fm.sy0;
        sxt = fm.sxt;
        syt = fm.syt;
        sx0x0 = fm.sx0x0;
        sx0y0 = fm.sx0y0;
        sy0y0 = fm.sy0y0;
        sxtxt = fm.sxtxt;
        sxtyt = fm.sxtyt;
        sytyt = fm.sytyt;
        sb11 = fm.sb11;
        sb12 = fm.sb12;
        sb21 = fm.sb21;
        sb22 = fm.sb22;
    }
    if k < cand_end && m < FIELD_CAND_CAP {
        let cd2 = &cs.cell_cands_d2[..FIELD_CAND_CAP];
        let cti = &cs.cell_cands_ti[..FIELD_CAND_CAP];
        let gx0 = &cs.field_grid_x[..MAX_MICRO];
        let gy0 = &cs.field_grid_y[..MAX_MICRO];
        let gaux = &cs.field_grid_aux[..MAX_MICRO];
        let f_ti = &mut cs.ftle_cands_ti[..FIELD_CAND_CAP];
        let f_d2 = &mut cs.ftle_cands_d2[..FIELD_CAND_CAP];
        let dxs = &mut cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &mut cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &mut cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &mut cs.cand_dv[..FIELD_CAND_CAP];

        let (mut v_lin, mut v_dx_row, mut v_dy_row, mut v_tt) = {
            use crate::simd::f32x4;
            (
                f32x4(sx0, sy0, sxt, syt),
                f32x4(sx0x0, sx0y0, sb11, sb21),
                f32x4(0.0, sy0y0, sb12, sb22),
                f32x4(sxtxt, sxtyt, sytyt, 0.0),
            )
        };
        macro_rules! push_ftle_cand {
            ($ks:expr) => {{
                let ks = $ks;
                let p = ((cti[ks] & 0x07FF) as usize).min(MAX_MICRO - 1);
                let dx = gx0[p] - x0_ref;
                let dy = gy0[p] - y0_ref;
                let xy_t = gaux[p];
                let du = xy_t[0] - xt_ref;
                let dv = xy_t[1] - yt_ref;
                let ms_idx = m.min(FIELD_CAND_CAP - 1);
                f_ti[ms_idx] = p as u16;
                f_d2[ms_idx] = cd2[ks];
                dxs[ms_idx] = dx;
                dys[ms_idx] = dy;
                dus[ms_idx] = du;
                dvs[ms_idx] = dv;
                {
                    use crate::simd::*;
                    let v_xyuv = f32x4(dx, dy, du, dv);
                    v_lin = f32x4_add(v_lin, v_xyuv);
                    v_dx_row = f32x4_add(v_dx_row, f32x4_mul(f32x4_splat(dx), v_xyuv));
                    v_dy_row = f32x4_add(v_dy_row, f32x4_mul(f32x4_splat(dy), v_xyuv));
                    v_tt = f32x4_add(
                        v_tt,
                        f32x4_mul(f32x4(du, du, dv, 0.0), f32x4(du, dv, dv, 0.0)),
                    );
                }
                m += 1;
            }};
        }

        if all_same_age {
            if m == 0 {
                let p = ((cti[k.min(FIELD_CAND_CAP - 1)] & 0x07FF) as usize).min(MAX_MICRO - 1);
                x0_ref = gx0[p];
                y0_ref = gy0[p];
                let xy_t = gaux[p];
                xt_ref = xy_t[0];
                yt_ref = xy_t[1];
                f_ti[0] = p as u16;
                f_d2[0] = cd2[k.min(FIELD_CAND_CAP - 1)];
                dxs[0] = 0.0;
                dys[0] = 0.0;
                dus[0] = 0.0;
                dvs[0] = 0.0;
                m = 1;
                k += 1;
            }
            let take = (cand_end - k).min(FIELD_CAND_CAP - m);
            let k_lim = k + take;
            while k < k_lim {
                push_ftle_cand!(k.min(FIELD_CAND_CAP - 1));
                k += 1;
            }
        } else {
            if m == 0 {
                while k < cand_end {
                    let ks = k.min(FIELD_CAND_CAP - 1);
                    let packed = cti[ks];
                    if ((packed >> 11) as usize) == best_a {
                        let p = ((packed & 0x07FF) as usize).min(MAX_MICRO - 1);
                        x0_ref = gx0[p];
                        y0_ref = gy0[p];
                        let xy_t = gaux[p];
                        xt_ref = xy_t[0];
                        yt_ref = xy_t[1];
                        f_ti[0] = p as u16;
                        f_d2[0] = cd2[ks];
                        dxs[0] = 0.0;
                        dys[0] = 0.0;
                        dus[0] = 0.0;
                        dvs[0] = 0.0;
                        m = 1;
                        k += 1;
                        break;
                    }
                    k += 1;
                }
            }
            while k < cand_end && m < FIELD_CAND_CAP {
                let ks = k.min(FIELD_CAND_CAP - 1);
                if ((cti[ks] >> 11) as usize) == best_a {
                    push_ftle_cand!(ks);
                }
                k += 1;
            }
        }
        {
            use crate::simd::f32x4_extract_lane;
            sx0 = f32x4_extract_lane::<0>(v_lin);
            sy0 = f32x4_extract_lane::<1>(v_lin);
            sxt = f32x4_extract_lane::<2>(v_lin);
            syt = f32x4_extract_lane::<3>(v_lin);
            sx0x0 = f32x4_extract_lane::<0>(v_dx_row);
            sx0y0 = f32x4_extract_lane::<1>(v_dx_row);
            sb11 = f32x4_extract_lane::<2>(v_dx_row);
            sb21 = f32x4_extract_lane::<3>(v_dx_row);
            sy0y0 = f32x4_extract_lane::<1>(v_dy_row);
            sb12 = f32x4_extract_lane::<2>(v_dy_row);
            sb22 = f32x4_extract_lane::<3>(v_dy_row);
            sxtxt = f32x4_extract_lane::<0>(v_tt);
            sxtyt = f32x4_extract_lane::<1>(v_tt);
            sytyt = f32x4_extract_lane::<2>(v_tt);
        }
        if save_fm {
            fm.x0_ref = x0_ref;
            fm.y0_ref = y0_ref;
            fm.xt_ref = xt_ref;
            fm.yt_ref = yt_ref;
            fm.sx0 = sx0;
            fm.sy0 = sy0;
            fm.sxt = sxt;
            fm.syt = syt;
            fm.sx0x0 = sx0x0;
            fm.sx0y0 = sx0y0;
            fm.sy0y0 = sy0y0;
            fm.sxtxt = sxtxt;
            fm.sxtyt = sxtyt;
            fm.sytyt = sytyt;
            fm.sb11 = sb11;
            fm.sb12 = sb12;
            fm.sb21 = sb21;
            fm.sb22 = sb22;
        }
    }
    if save_fm {
        fm.prev_k = cand_end;
        fm.m = m;
    }
    if m < MIN_RIGID_MEMBERS {
        return (false, false, false, 0.0, 0, false);
    }

    let mf = m as f32;
    let inv_m = 1.0 / mf;
    let mx0 = sx0 * inv_m;
    let my0 = sy0 * inv_m;
    let mxt = sxt * inv_m;
    let myt = syt * inv_m;
    let gxx = sx0x0 - sx0 * mx0;
    let gxy = sx0y0 - sx0 * my0;
    let gyy = sy0y0 - sy0 * my0;
    let txx = sxtxt - sxt * mxt;
    let txy = sxtyt - sxt * myt;
    let tyy = sytyt - syt * myt;
    let req_eig0 = mf * min_eig_px2;
    if !is_well_conditioned_cov2_f32(gxx, gxy, gyy, req_eig0)
        || !is_well_conditioned_cov2_f32(txx, txy, tyy, req_eig0)
    {
        return (false, false, false, 0.0, 0, false);
    }
    let det0 = gxx * gyy - gxy * gxy;
    if det0 <= 1e-10 {
        return (false, false, false, 0.0, 0, false);
    }
    let b11 = sb11 - sxt * mx0;
    let b12 = sb12 - sxt * my0;
    let b21 = sb21 - syt * mx0;
    let b22 = sb22 - syt * my0;
    let inv_det0 = 1.0 / det0;
    let f11_0 = (b11 * gyy - b12 * gxy) * inv_det0;
    let f12_0 = (-b11 * gxy + b12 * gxx) * inv_det0;
    let f21_0 = (b21 * gyy - b22 * gxy) * inv_det0;
    let f22_0 = (-b21 * gxy + b22 * gxx) * inv_det0;
    if !f11_0.is_finite() || !f12_0.is_finite() || !f21_0.is_finite() || !f22_0.is_finite() {
        return (false, false, false, 0.0, 0, false);
    }

    let x0_b0 = mx0;
    let y0_b0 = my0;
    let xt_b0 = mxt;
    let yt_b0 = myt;
    let c1_0 = xt_b0 - f11_0 * x0_b0 - f12_0 * y0_b0;
    let c2_0 = yt_b0 - f21_0 * x0_b0 - f22_0 * y0_b0;

    let m = m.min(FIELD_CAND_CAP);
    let floor2 = (0.75 * q) * (0.75 * q);
    let s0_clamp = (0.35 / 1.5) * q;
    let s0_clamp2 = s0_clamp * s0_clamp;
    let max_r2_0: f32;
    let mut cnt0_le_floor2 = 0usize;
    let mut cnt0_le_s0 = 0usize;
    {
        let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
        let ress = &mut cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vc1 = f32x4_splat(c1_0);
            let vc2 = f32x4_splat(c2_0);
            let vf11 = f32x4_splat(f11_0);
            let vf12 = f32x4_splat(f12_0);
            let vf21 = f32x4_splat(f21_0);
            let vf22 = f32x4_splat(f22_0);
            let vfloor2 = f32x4_splat(floor2);
            let vs02 = f32x4_splat(s0_clamp2);
            let vm = f32x4_splat(m as f32);
            let vfour = f32x4_splat(4.0);
            let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
            let mut vmax_r2 = f32x4_splat(0.0);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(
                    vdu,
                    f32x4_add(vc1, f32x4_add(f32x4_mul(vf11, vdx), f32x4_mul(vf12, vdy))),
                );
                let vry = f32x4_sub(
                    vdv,
                    f32x4_add(vc2, f32x4_add(f32x4_mul(vf21, vdx), f32x4_mul(vf22, vdy))),
                );
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                let vin = f32x4_lt(vidx, vm);
                let valid = i32x4_bitmask(vin) as u32;
                vmax_r2 = f32x4_pmax(vmax_r2, v128_and(vr2, vin));
                ress[k] = f32x4_extract_lane::<0>(vr2);
                ress[k + 1] = f32x4_extract_lane::<1>(vr2);
                ress[k + 2] = f32x4_extract_lane::<2>(vr2);
                ress[k + 3] = f32x4_extract_lane::<3>(vr2);
                cnt0_le_floor2 += ((i32x4_bitmask(f32x4_le(vr2, vfloor2)) as u32) & valid)
                    .count_ones() as usize;
                cnt0_le_s0 +=
                    ((i32x4_bitmask(f32x4_le(vr2, vs02)) as u32) & valid).count_ones() as usize;
                vidx = f32x4_add(vidx, vfour);
                bi += 1;
            }
            max_r2_0 = hmax_f32x4(vmax_r2);
        }
    }

    // Exact-affine shortcut: sub-0.75px residual -> OLS F_0 satisfies conditioning & inlier checks
    // without distance weighting, IRLS, or Pass 2 residual re-verification.
    if max_r2_0 <= floor2 {
        let det_f = f11_0 * f22_0 - f12_0 * f21_0;
        if !det_f.is_finite() || det_f <= 0.0 {
            return (false, true, false, 0.0, 0, false);
        }
        let c11 = f11_0 * f11_0 + f21_0 * f21_0;
        let c12 = f11_0 * f12_0 + f21_0 * f22_0;
        let c22 = f12_0 * f12_0 + f22_0 * f22_0;
        let (_, lambda_max) = sym2x2_eigenvalues(c11, c12, c22);
        if !lambda_max.is_finite() {
            return (false, true, false, 0.0, 0, false);
        }
        let ftle = 0.5 * fast_lnf(lambda_max.max(1e-12)) / t_sec;
        if !ftle.is_finite() {
            return (false, true, false, 0.0, 0, false);
        }
        return (true, true, false, ftle, m, true);
    }

    let clean_patch = max_r2_0 <= (0.85 * q) * (0.85 * q) && cnt0_le_s0 > (m >> 1);
    if max_r2_0 <= q * q
        || (max_r2_0 <= (1.18 * q) * (1.18 * q) && cnt0_le_floor2 * 2 >= m)
    {
        // Curved clean-patch fast path: all points already near <= 1.18*q (< 1.25*q trim cap)
        // with median <= 0.75*q, so skip LOO & Cauchy IRLS and run Pass 2 + verification.
        let d2s = &cs.ftle_cands_d2[..FIELD_CAND_CAP];
        let ws = &mut cs.ftle_cands_w[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vone = f32x4_splat(1.0);
            let vhalf_inv_r2 = f32x4_splat(0.5 * inv_r2);
            let vmin_w = f32x4_splat(0.25);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vd2 = f32x4(d2s[k], d2s[k + 1], d2s[k + 2], d2s[k + 3]);
                let vw = f32x4_max(f32x4_sub(vone, f32x4_mul(vd2, vhalf_inv_r2)), vmin_w);
                ws[k] = f32x4_extract_lane::<0>(vw);
                ws[k + 1] = f32x4_extract_lane::<1>(vw);
                ws[k + 2] = f32x4_extract_lane::<2>(vw);
                ws[k + 3] = f32x4_extract_lane::<3>(vw);
                bi += 1;
            }
        }
        return finish_ftle_pass2(cs, m, m, clean_patch, min_eig_px2, q, t_sec);
    }

    try_solve_cell_ftle_fallback(
        cs,
        m,
        mx0,
        my0,
        mxt,
        myt,
        gxx,
        gxy,
        gyy,
        txx,
        txy,
        tyy,
        b11,
        b12,
        b21,
        b22,
        inv_det0,
        inv_r2,
        t_sec,
        q,
        min_eig_px2,
        max_r2_0,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn try_solve_cell_ftle_fallback(
    cs: &mut ContinuumState,
    m: usize,
    mx0: f32,
    my0: f32,
    mxt: f32,
    myt: f32,
    gxx: f32,
    gxy: f32,
    gyy: f32,
    txx: f32,
    txy: f32,
    tyy: f32,
    b11: f32,
    b12: f32,
    b21: f32,
    b22: f32,
    inv_det0: f32,
    inv_r2: f32,
    t_sec: f32,
    q: f32,
    min_eig_px2: f32,
    max_r2_0: f32,
) -> (bool, bool, bool, f32, usize, bool) {
    let mf = m as f32;
    let inv_m = 1.0 / mf;
    let four_q2 = (4.0 * q) * (4.0 * q);
    let med_bail2 = (1.45 * q) * (1.45 * q);
    let floor2 = (0.75 * q) * (0.75 * q);
    let s0_clamp = (0.35 / 1.5) * q;
    let s0_clamp2 = s0_clamp * s0_clamp;
    let mut cnt0_le_4q2 = 0usize;
    let mut cnt0_le_med = 0usize;
    let mut cnt0_le_floor2 = 0usize;
    let mut cnt0_le_s0 = 0usize;
    {
        let ress = &cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let v4q2 = f32x4_splat(four_q2);
            let vmed2 = f32x4_splat(med_bail2);
            let vfloor2 = f32x4_splat(floor2);
            let vs02 = f32x4_splat(s0_clamp2);
            let vm = f32x4_splat(m as f32);
            let vfour = f32x4_splat(4.0);
            let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vr2 = f32x4(ress[k], ress[k + 1], ress[k + 2], ress[k + 3]);
                let valid = i32x4_bitmask(f32x4_lt(vidx, vm)) as u32;
                cnt0_le_4q2 +=
                    ((i32x4_bitmask(f32x4_le(vr2, v4q2)) as u32) & valid).count_ones() as usize;
                cnt0_le_med += ((i32x4_bitmask(f32x4_le(vr2, vmed2)) as u32) & valid)
                    .count_ones() as usize;
                cnt0_le_floor2 += ((i32x4_bitmask(f32x4_le(vr2, vfloor2)) as u32) & valid)
                    .count_ones() as usize;
                cnt0_le_s0 +=
                    ((i32x4_bitmask(f32x4_le(vr2, vs02)) as u32) & valid).count_ones() as usize;
                vidx = f32x4_add(vidx, vfour);
                bi += 1;
            }
        }
    }

    if (m >= 7 && cnt0_le_4q2 * 2 <= m) || (m == 6 && cnt0_le_4q2 < 5) {
        return (false, false, true, 0.0, 0, false);
    }
    let idx_q80 = ((m * 4) / 5).min(m - 2);
    let can_loo = m >= 7 && max_r2_0 > q * q && cnt0_le_floor2 <= idx_q80;
    let scale0 = if !can_loo && cnt0_le_s0 > (m >> 1) {
        0.35 * q
    } else {
        if m >= 7 && (cnt0_le_med <= ((m - 1) >> 1) || cnt0_le_4q2 * 5 < m * 4) {
            let has_press = can_loo
                && has_dominant_press_outlier(
                    cs, m, mx0, my0, gxx, gxy, gyy, inv_det0, inv_m, 0.45,
                );
            if !has_press {
                return (false, false, true, 0.0, 0, false);
            }
        }
        let (med0_r2, base_q80_r2) = median_and_q80_r2(&cs.ftle_cands_res, m, can_loo);
        let mut med0 = wasm_sqrtf(med0_r2);

        if can_loo && base_q80_r2 > floor2 {
            let (did_drop, loo_med) = try_ftle_loo_drop(
                cs,
                m,
                mf,
                inv_m,
                mx0,
                my0,
                mxt,
                myt,
                gxx,
                gxy,
                gyy,
                txx,
                txy,
                tyy,
                b11,
                b12,
                b21,
                b22,
                inv_det0,
                min_eig_px2,
                idx_q80,
                med0_r2,
                base_q80_r2,
                floor2,
            );
            if did_drop {
                med0 = wasm_sqrtf(loo_med);
            } else if med0_r2 > med_bail2 {
                return (false, false, true, 0.0, 0, false);
            }
        } else if m >= 7 && med0_r2 > med_bail2 {
            return (false, false, true, 0.0, 0, false);
        }
        (1.5 * med0).max(0.35 * q)
    };

    // IRLS Cauchy reweighting step to suppress outliers before hard trimming
    let inv_s2 = 1.0 / (scale0 * scale0);
    {
        let ress = &cs.ftle_cands_res[..FIELD_CAND_CAP];
        let ws = &mut cs.ftle_cands_w[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vone = f32x4_splat(1.0);
            let vinv_s2 = f32x4_splat(inv_s2);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vr2 = f32x4(ress[k], ress[k + 1], ress[k + 2], ress[k + 3]);
                let vw = f32x4_div(vone, f32x4_add(vone, f32x4_mul(vr2, vinv_s2)));
                ws[k] = f32x4_extract_lane::<0>(vw);
                ws[k + 1] = f32x4_extract_lane::<1>(vw);
                ws[k + 2] = f32x4_extract_lane::<2>(vw);
                ws[k + 3] = f32x4_extract_lane::<3>(vw);
                bi += 1;
            }
        }
    }

    // Pass 1: IRLS refit -> bounded residual threshold -> hard-trim >=80% inliers (H7)
    let (ok1, f11_1, f12_1, f21_1, f22_1, x0_b1, y0_b1, xt_b1, yt_b1) =
        fit_cand_ftle_weighted_2x2(cs, m, false, min_eig_px2);
    if !ok1 {
        return (false, false, false, 0.0, 0, false);
    }

    let c1_1 = xt_b1 - f11_1 * x0_b1 - f12_1 * y0_b1;
    let c2_1 = yt_b1 - f21_1 * x0_b1 - f22_1 * y0_b1;
    let cap2 = (1.25 * q) * (1.25 * q);
    let lo1_c = (0.75 / (3.0 * 1.4826)) * q;
    let lo1_2 = lo1_c * lo1_c;
    let hi1_c = (1.25 / (3.0 * 1.4826)) * q;
    let hi1_2 = hi1_c * hi1_c;
    let mut cnt_le_cap = 0usize;
    let mut cnt1_le_lo = 0usize;
    let mut cnt1_lt_hi = 0usize;
    let max_r2_1: f32;
    {
        let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
        let dys = &cs.cand_dy[..FIELD_CAND_CAP];
        let dus = &cs.cand_du[..FIELD_CAND_CAP];
        let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
        let ress = &mut cs.ftle_cands_res[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vc1 = f32x4_splat(c1_1);
            let vc2 = f32x4_splat(c2_1);
            let vf11 = f32x4_splat(f11_1);
            let vf12 = f32x4_splat(f12_1);
            let vf21 = f32x4_splat(f21_1);
            let vf22 = f32x4_splat(f22_1);
            let vcap2 = f32x4_splat(cap2);
            let vhi2 = f32x4_splat(hi1_2);
            let vlo2 = f32x4_splat(lo1_2);
            let vm = f32x4_splat(m as f32);
            let vfour = f32x4_splat(4.0);
            let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
            let mut vmax_r2 = f32x4_splat(0.0);
            let mut bi = 0usize;
            while bi < blk {
                let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                let vrx = f32x4_sub(
                    vdu,
                    f32x4_add(vc1, f32x4_add(f32x4_mul(vf11, vdx), f32x4_mul(vf12, vdy))),
                );
                let vry = f32x4_sub(
                    vdv,
                    f32x4_add(vc2, f32x4_add(f32x4_mul(vf21, vdx), f32x4_mul(vf22, vdy))),
                );
                let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                let vin = f32x4_lt(vidx, vm);
                let valid = i32x4_bitmask(vin) as u32;
                vmax_r2 = f32x4_pmax(vmax_r2, v128_and(vr2, vin));
                ress[k] = f32x4_extract_lane::<0>(vr2);
                ress[k + 1] = f32x4_extract_lane::<1>(vr2);
                ress[k + 2] = f32x4_extract_lane::<2>(vr2);
                ress[k + 3] = f32x4_extract_lane::<3>(vr2);
                cnt_le_cap +=
                    ((i32x4_bitmask(f32x4_le(vr2, vcap2)) as u32) & valid).count_ones() as usize;
                cnt1_lt_hi +=
                    ((i32x4_bitmask(f32x4_lt(vr2, vhi2)) as u32) & valid).count_ones() as usize;
                cnt1_le_lo +=
                    ((i32x4_bitmask(f32x4_le(vr2, vlo2)) as u32) & valid).count_ones() as usize;
                vidx = f32x4_add(vidx, vfour);
                bi += 1;
            }
            max_r2_1 = hmax_f32x4(vmax_r2);
        }
    }
    if cnt_le_cap < MIN_RIGID_MEMBERS || cnt_le_cap * 5 < m * 4 {
        return (false, false, true, 0.0, 0, false);
    }

    let thresh2 = if max_r2_1 <= floor2 || cnt1_le_lo > (m >> 1) {
        floor2
    } else if cnt1_lt_hi <= ((m - 1) >> 1) {
        cap2
    } else {
        let med1 = wasm_sqrtf(median_and_q80_r2(&cs.ftle_cands_res, m, false).0);
        let sigma_pair = 1.4826 * med1;
        let thresh = ((0.75 * q).max(3.0 * sigma_pair)).min(1.25 * q);
        thresh * thresh
    };

    let mut inliers = 0usize;
    {
        let ress = &cs.ftle_cands_res[..FIELD_CAND_CAP];
        let d2s = &cs.ftle_cands_d2[..FIELD_CAND_CAP];
        let ws = &mut cs.ftle_cands_w[..FIELD_CAND_CAP];
        {
            use crate::simd::*;
            let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
            let vone = f32x4_splat(1.0);
            let vhalf_inv_r2 = f32x4_splat(0.5 * inv_r2);
            let vmin_w = f32x4_splat(0.25);
            if max_r2_1 <= thresh2 {
                let mut bi = 0usize;
                while bi < blk {
                    let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                    let vd2 = f32x4(d2s[k], d2s[k + 1], d2s[k + 2], d2s[k + 3]);
                    let vw = f32x4_max(f32x4_sub(vone, f32x4_mul(vd2, vhalf_inv_r2)), vmin_w);
                    ws[k] = f32x4_extract_lane::<0>(vw);
                    ws[k + 1] = f32x4_extract_lane::<1>(vw);
                    ws[k + 2] = f32x4_extract_lane::<2>(vw);
                    ws[k + 3] = f32x4_extract_lane::<3>(vw);
                    bi += 1;
                }
                inliers = m;
            } else {
                let vth2 = f32x4_splat(thresh2);
                let vm = f32x4_splat(m as f32);
                let vfour = f32x4_splat(4.0);
                let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
                let mut bi = 0usize;
                while bi < blk {
                    let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                    let vr2 = f32x4(ress[k], ress[k + 1], ress[k + 2], ress[k + 3]);
                    let vd2 = f32x4(d2s[k], d2s[k + 1], d2s[k + 2], d2s[k + 3]);
                    let vw_d = f32x4_max(f32x4_sub(vone, f32x4_mul(vd2, vhalf_inv_r2)), vmin_w);
                    let keep = v128_and(f32x4_le(vr2, vth2), f32x4_lt(vidx, vm));
                    let vw = v128_and(vw_d, keep);
                    ws[k] = f32x4_extract_lane::<0>(vw);
                    ws[k + 1] = f32x4_extract_lane::<1>(vw);
                    ws[k + 2] = f32x4_extract_lane::<2>(vw);
                    ws[k + 3] = f32x4_extract_lane::<3>(vw);
                    inliers += (i32x4_bitmask(keep) as u32).count_ones() as usize;
                    vidx = f32x4_add(vidx, vfour);
                    bi += 1;
                }
            }
        }
    }
    if inliers < MIN_RIGID_MEMBERS || inliers * 5 < m * 4 {
        return (false, false, false, 0.0, 0, false);
    }

    finish_ftle_pass2(cs, m, inliers, false, min_eig_px2, q, t_sec)
}

#[inline(always)]
fn finish_ftle_pass2(
    cs: &ContinuumState,
    m: usize,
    inliers: usize,
    clean_patch: bool,
    min_eig_px2: f32,
    q: f32,
    t_sec: f32,
) -> (bool, bool, bool, f32, usize, bool) {
    // Pass 2: final clean fit on hard-trimmed inliers + final inlier consistency check (H7)
    let (ok_f, f11, f12, f21, f22, x0_bf, y0_bf, xt_bf, yt_bf) =
        fit_cand_ftle_weighted_2x2(cs, m, inliers < m, min_eig_px2);
    if !ok_f {
        return (false, true, false, 0.0, 0, false);
    }

    if !clean_patch {
        let c1_f = xt_bf - f11 * x0_bf - f12 * y0_bf;
        let c2_f = yt_bf - f21 * x0_bf - f22 * y0_bf;
        let lim65_2 = (0.65 * q) * (0.65 * q);
        let ss_in: f32;
        let mut in_cnt = 0usize;
        let mut in_le_65 = 0usize;
        let max_le_65: f32;
        let min_gt_65: f32;
        {
            let ws = &cs.ftle_cands_w[..FIELD_CAND_CAP];
            let dxs = &cs.cand_dx[..FIELD_CAND_CAP];
            let dys = &cs.cand_dy[..FIELD_CAND_CAP];
            let dus = &cs.cand_du[..FIELD_CAND_CAP];
            let dvs = &cs.cand_dv[..FIELD_CAND_CAP];
            {
                use crate::simd::*;
                let blk = ((m + 3) >> 2).min(FIELD_CAND_CAP >> 2);
                let vc1 = f32x4_splat(c1_f);
                let vc2 = f32x4_splat(c2_f);
                let vf11 = f32x4_splat(f11);
                let vf12 = f32x4_splat(f12);
                let vf21 = f32x4_splat(f21);
                let vf22 = f32x4_splat(f22);
                let vlim = f32x4_splat(lim65_2);
                let vz = f32x4_splat(0.0);
                let vinf = f32x4_splat(f32::INFINITY);
                let vm = f32x4_splat(m as f32);
                let vfour = f32x4_splat(4.0);
                let mut vidx = f32x4(0.0, 1.0, 2.0, 3.0);
                let mut vss = f32x4_splat(0.0);
                let mut vmax = f32x4_splat(0.0);
                let mut vmin = vinf;
                let all_in = inliers == m;
                let mut bi = 0usize;
                while bi < blk {
                    let k = (bi << 2).min(FIELD_CAND_CAP - 4);
                    let vdx = f32x4(dxs[k], dxs[k + 1], dxs[k + 2], dxs[k + 3]);
                    let vdy = f32x4(dys[k], dys[k + 1], dys[k + 2], dys[k + 3]);
                    let vdu = f32x4(dus[k], dus[k + 1], dus[k + 2], dus[k + 3]);
                    let vdv = f32x4(dvs[k], dvs[k + 1], dvs[k + 2], dvs[k + 3]);
                    let vrx = f32x4_sub(
                        vdu,
                        f32x4_add(vc1, f32x4_add(f32x4_mul(vf11, vdx), f32x4_mul(vf12, vdy))),
                    );
                    let vry = f32x4_sub(
                        vdv,
                        f32x4_add(vc2, f32x4_add(f32x4_mul(vf21, vdx), f32x4_mul(vf22, vdy))),
                    );
                    let vr2 = f32x4_add(f32x4_mul(vrx, vrx), f32x4_mul(vry, vry));
                    let vin = f32x4_lt(vidx, vm);
                    let vact = if all_in {
                        vin
                    } else {
                        let vw = f32x4(ws[k], ws[k + 1], ws[k + 2], ws[k + 3]);
                        v128_and(f32x4_gt(vw, vz), vin)
                    };
                    let vle = v128_and(vact, f32x4_le(vr2, vlim));
                    let vgt = v128_and(vact, v128_not(vle));
                    vss = f32x4_add(vss, v128_and(vr2, vact));
                    in_cnt += (i32x4_bitmask(vact) as u32).count_ones() as usize;
                    in_le_65 += (i32x4_bitmask(vle) as u32).count_ones() as usize;
                    vmax = f32x4_pmax(vmax, v128_and(vr2, vle));
                    vmin = f32x4_pmin(vmin, v128_bitselect(vr2, vinf, vgt));
                    vidx = f32x4_add(vidx, vfour);
                    bi += 1;
                }
                ss_in = hsum_f32x4(vss);
                max_le_65 = hmax_f32x4(vmax);
                min_gt_65 = hmin_f32x4(vmin);
            }
        }
        if in_cnt == 0 || !ss_in.is_finite() {
            return (false, true, false, 0.0, 0, false);
        }
        let rmse_lim = 0.85 * q;
        if ss_in > (in_cnt as f32) * rmse_lim * rmse_lim {
            return (false, true, false, 0.0, 0, false);
        }
        if min_gt_65 < f32::INFINITY {
            let mid = in_cnt >> 1;
            let pass_med = if in_le_65 > mid {
                true
            } else if (in_cnt & 1) == 1 || in_le_65 < mid {
                false
            } else {
                let med_r2 = 0.25
                    * (max_le_65
                        + min_gt_65
                        + 2.0 * wasm_sqrtf((max_le_65 * min_gt_65).max(0.0)));
                med_r2.is_finite() && med_r2 <= lim65_2
            };
            if !pass_med {
                return (false, true, false, 0.0, 0, false);
            }
        }
    }

    let det_f = f11 * f22 - f12 * f21;
    if !det_f.is_finite() || det_f <= 0.0 {
        return (false, true, false, 0.0, 0, false);
    }

    // Right Cauchy-Green tensor C = F^T F
    let c11 = f11 * f11 + f21 * f21;
    let c12 = f11 * f12 + f21 * f22;
    let c22 = f12 * f12 + f22 * f22;
    let (_, lambda_max) = sym2x2_eigenvalues(c11, c12, c22);
    if !lambda_max.is_finite() {
        return (false, true, false, 0.0, 0, false);
    }

    let ftle = 0.5 * fast_lnf(lambda_max.max(1e-12)) / t_sec;
    if !ftle.is_finite() {
        return (false, true, false, 0.0, 0, false);
    }
    (true, true, false, ftle, m, inliers == m)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn gather_3x3_split_le24(
    cs: &mut ContinuumState,
    s0: usize,
    e0: usize,
    s1: usize,
    e1: usize,
    s2: usize,
    e2: usize,
    xc: f32,
    yc: f32,
    r0_2: f32,
    r_max2: f32,
    min_needed: usize,
) -> u32 {
    let mut lo = 0usize;
    let mut hi = FIELD_CAND_CAP;
    {
        let gx = &cs.field_grid_x[..MAX_MICRO];
        let gy = &cs.field_grid_y[..MAX_MICRO];
        let gti = &cs.field_grid_ti[..MAX_MICRO];
        let c_ti = &mut cs.cell_cands_ti[..FIELD_CAND_CAP];
        let c_d2 = &mut cs.cell_cands_d2[..FIELD_CAND_CAP];
        let mut si = 0usize;
        while si < 3 && lo < hi {
            let (sa, ea) = match si {
                0 => (s0, e0),
                1 => (s1, e1),
                _ => (s2, e2),
            };
            let e = ea.min(MAX_MICRO);
            let s = sa.min(e);
            if s < e {
                let mut p = s;
                let e_lim = s + (e - s).min(hi - lo);
                while p < e_lim {
                    let dx = gx[p] - xc;
                    let dy = gy[p] - yc;
                    let d2 = dx * dx + dy * dy;
                    let m_all = (d2 <= r_max2) as usize;
                    let m0 = (d2 <= r0_2) as usize;
                    let m1 = m_all & (m0 ^ 1);
                    let idx = (if m0 != 0 { lo } else { hi - 1 }).min(FIELD_CAND_CAP - 1);
                    c_ti[idx] = gti[p];
                    c_d2[idx] = d2;
                    lo += m0;
                    hi -= m1;
                    p += 1;
                }
            }
            si += 1;
        }
    }
    let end0 = lo;
    let n1 = FIELD_CAND_CAP - hi;
    let cand_n = end0 + n1;
    if cand_n < min_needed {
        return 0;
    }
    if n1 > 0 && end0 < hi && hi < FIELD_CAND_CAP {
        let mut dst_k = end0;
        let mut src_k = hi;
        while src_k < FIELD_CAND_CAP && dst_k < FIELD_CAND_CAP {
            cs.cell_cands_ti[dst_k] = cs.cell_cands_ti[src_k];
            cs.cell_cands_d2[dst_k] = cs.cell_cands_d2[src_k];
            dst_k += 1;
            src_k += 1;
        }
    }
    (end0 as u32) | ((cand_n as u32) << 8) | ((cand_n as u32) << 16)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn gather_3x3_flat_le24(
    cs: &mut ContinuumState,
    s0: usize,
    e0: usize,
    s1: usize,
    e1: usize,
    s2: usize,
    e2: usize,
    xc: f32,
    yc: f32,
    r_max2: f32,
    min_needed: usize,
) -> u32 {
    let mut cand_n = 0usize;
    let gx = &cs.field_grid_x[..MAX_MICRO];
    let gy = &cs.field_grid_y[..MAX_MICRO];
    let gti = &cs.field_grid_ti[..MAX_MICRO];
    let c_ti = &mut cs.cell_cands_ti[..FIELD_CAND_CAP];
    let c_d2 = &mut cs.cell_cands_d2[..FIELD_CAND_CAP];
    let (vxc, vyc, vrmax2) = {
        use crate::simd::f32x4_splat;
        (f32x4_splat(xc), f32x4_splat(yc), f32x4_splat(r_max2))
    };
    let mut si = 0usize;
    while si < 3 && cand_n < FIELD_CAND_CAP {
        let (sa, ea) = match si {
            0 => (s0, e0),
            1 => (s1, e1),
            _ => (s2, e2),
        };
        let e = ea.min(MAX_MICRO);
        let s = sa.min(e);
        if s < e {
            let mut p = s;
            let e_lim = s + (e - s).min(FIELD_CAND_CAP - cand_n);
            {
                use crate::simd::*;
                while p + 4 <= e_lim {
                    let vx = f32x4(gx[p], gx[p + 1], gx[p + 2], gx[p + 3]);
                    let vy = f32x4(gy[p], gy[p + 1], gy[p + 2], gy[p + 3]);
                    let vdx = f32x4_sub(vx, vxc);
                    let vdy = f32x4_sub(vy, vyc);
                    let vd2 = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
                    let m_all = i32x4_bitmask(f32x4_le(vd2, vrmax2)) as u32;
                    if m_all != 0 {
                        let d0 = f32x4_extract_lane::<0>(vd2);
                        let d1 = f32x4_extract_lane::<1>(vd2);
                        let d2v = f32x4_extract_lane::<2>(vd2);
                        let d3 = f32x4_extract_lane::<3>(vd2);
                        if m_all == 15 {
                            let idx = cand_n.min(FIELD_CAND_CAP - 4);
                            c_ti[idx] = gti[p];
                            c_ti[idx + 1] = gti[p + 1];
                            c_ti[idx + 2] = gti[p + 2];
                            c_ti[idx + 3] = gti[p + 3];
                            c_d2[idx] = d0;
                            c_d2[idx + 1] = d1;
                            c_d2[idx + 2] = d2v;
                            c_d2[idx + 3] = d3;
                            cand_n += 4;
                        } else {
                            let idx0 = cand_n.min(FIELD_CAND_CAP - 1);
                            c_ti[idx0] = gti[p];
                            c_d2[idx0] = d0;
                            cand_n += (m_all & 1) as usize;
                            let idx1 = cand_n.min(FIELD_CAND_CAP - 1);
                            c_ti[idx1] = gti[p + 1];
                            c_d2[idx1] = d1;
                            cand_n += ((m_all >> 1) & 1) as usize;
                            let idx2 = cand_n.min(FIELD_CAND_CAP - 1);
                            c_ti[idx2] = gti[p + 2];
                            c_d2[idx2] = d2v;
                            cand_n += ((m_all >> 2) & 1) as usize;
                            let idx3 = cand_n.min(FIELD_CAND_CAP - 1);
                            c_ti[idx3] = gti[p + 3];
                            c_d2[idx3] = d3;
                            cand_n += ((m_all >> 3) & 1) as usize;
                        }
                    }
                    p += 4;
                }
            }
            while p < e_lim {
                let dx = gx[p] - xc;
                let dy = gy[p] - yc;
                let d2 = dx * dx + dy * dy;
                let idx = cand_n.min(FIELD_CAND_CAP - 1);
                c_ti[idx] = gti[p];
                c_d2[idx] = d2;
                cand_n += (d2 <= r_max2) as usize;
                p += 1;
            }
        }
        si += 1;
    }
    if cand_n < min_needed {
        0
    } else {
        (cand_n as u32) << 16
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn gather_wide_le24(
    cs: &mut ContinuumState,
    row: usize,
    col: usize,
    xc: f32,
    yc: f32,
    r0_2: f32,
    r1_2: f32,
    r_max2: f32,
    dy_cells: isize,
    row_dx: &[i8; FIELD_GRID_H],
    min_needed: usize,
    part_mode: u8,
) -> u32 {
    let row_s = row as isize;
    let d_max = dy_cells.min((FIELD_GRID_H - 1) as isize);
    let mut ny = (row_s - d_max).max(0) as usize;
    let ny_hi = (row_s + d_max).min((FIELD_GRID_H - 1) as isize) as usize;
    let mut row_off = ny * FIELD_GRID_W;

    let mut cand_n = 0usize;
    {
        let row_start = &cs.field_cell_start;
        let gx = &cs.field_grid_x[..MAX_MICRO];
        let gy = &cs.field_grid_y[..MAX_MICRO];
        let gti = &cs.field_grid_ti[..MAX_MICRO];
        let c_ti = &mut cs.cell_cands_ti[..FIELD_CAND_CAP];
        let c_d2 = &mut cs.cell_cands_d2[..FIELD_CAND_CAP];

        let vxc = crate::simd::f32x4_splat(xc);

        let vyc = crate::simd::f32x4_splat(yc);

        let vrmax2 = crate::simd::f32x4_splat(r_max2);
        while ny <= ny_hi && cand_n < FIELD_CAND_CAP {
            let dx_c = row_dx[(ny as isize - row_s).unsigned_abs().min(FIELD_GRID_H - 1)] as isize;
            if dx_c >= 0 {
                let cx_lo = (col as isize - dx_c).max(0) as usize;
                let cx_hi = (col as isize + dx_c).min((FIELD_GRID_W - 1) as isize) as usize;
                let idx_a = (row_off + cx_lo).min(FIELD_CELLS);
                let idx_b = (row_off + cx_hi + 1).min(FIELD_CELLS);
                let e = (row_start[idx_b] as usize).min(MAX_MICRO);
                let s = (row_start[idx_a] as usize).min(e);
                let mut p = s;
                let e_lim = s + (e - s).min(FIELD_CAND_CAP - cand_n);
                while p + 4 <= e_lim {
                    use crate::simd::*;
                    let ps = p.min(MAX_MICRO - 4);
                    let vdx = f32x4_sub(f32x4(gx[ps], gx[ps + 1], gx[ps + 2], gx[ps + 3]), vxc);
                    let vdy = f32x4_sub(f32x4(gy[ps], gy[ps + 1], gy[ps + 2], gy[ps + 3]), vyc);
                    let vd2 = f32x4_add(f32x4_mul(vdx, vdx), f32x4_mul(vdy, vdy));
                    let mut mask = i32x4_bitmask(f32x4_le(vd2, vrmax2)) as u32;
                    while mask != 0 {
                        let lane = mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        let d_lane = match lane {
                            0 => f32x4_extract_lane::<0>(vd2),
                            1 => f32x4_extract_lane::<1>(vd2),
                            2 => f32x4_extract_lane::<2>(vd2),
                            _ => f32x4_extract_lane::<3>(vd2),
                        };
                        let idx = cand_n.min(FIELD_CAND_CAP - 1);
                        c_ti[idx] = gti[(ps + lane).min(MAX_MICRO - 1)];
                        c_d2[idx] = d_lane;
                        cand_n += 1;
                    }
                    p += 4;
                }
                while p < e_lim {
                    let ps = p.min(MAX_MICRO - 1);
                    let dx = gx[ps] - xc;
                    let dy = gy[ps] - yc;
                    let d2 = dx * dx + dy * dy;
                    let idx = cand_n.min(FIELD_CAND_CAP - 1);
                    c_ti[idx] = gti[ps];
                    c_d2[idx] = d2;
                    cand_n += (d2 <= r_max2) as usize;
                    p += 1;
                }
            }
            row_off += FIELD_GRID_W;
            ny += 1;
        }
    }
    if cand_n < min_needed {
        return 0;
    }
    if part_mode == 0 {
        return (cand_n as u32) << 16;
    }
    let (end0, end1) = partition_cell_cands_mode(
        &mut cs.cell_cands_ti,
        &mut cs.cell_cands_d2,
        cand_n,
        r0_2,
        r1_2,
        part_mode,
    );
    (end0 as u32) | ((end1 as u32) << 8) | ((cand_n as u32) << 16)
}

#[inline(always)]
fn band3_and_band5_summary(
    row_cnt: &[u16; FIELD_GRID_H],
    row_occ: &[u64; FIELD_GRID_H],
    row: usize,
    b5_lo: usize,
    b5_hi: usize,
) -> (usize, usize, usize, u64, usize, u64) {
    let b3_lo = row.saturating_sub(1);
    let b3_hi = (row + 1).min(FIELD_GRID_H - 1);
    let mut band3_tot = row_cnt[row] as usize;
    let mut band3_occ = row_occ[row];
    if row > 0 {
        band3_tot += row_cnt[row - 1] as usize;
        band3_occ |= row_occ[row - 1];
    }
    if row + 1 < FIELD_GRID_H {
        band3_tot += row_cnt[row + 1] as usize;
        band3_occ |= row_occ[row + 1];
    }
    let mut band5_tot = band3_tot;
    let mut band5_occ = band3_occ;
    let mut ny = b5_lo;
    while ny < b3_lo {
        band5_tot += row_cnt[ny] as usize;
        band5_occ |= row_occ[ny];
        ny += 1;
    }
    ny = b3_hi + 1;
    while ny <= b5_hi {
        let ny_c = ny.min(FIELD_GRID_H - 1);
        band5_tot += row_cnt[ny_c] as usize;
        band5_occ |= row_occ[ny_c];
        ny += 1;
    }
    (b3_lo, b3_hi, band3_tot, band3_occ, band5_tot, band5_occ)
}

#[inline(always)]
fn count_box5_tracks(
    cs: &ContinuumState,
    row_occ: &[u64; FIELD_GRID_H],
    b5_lo: usize,
    b5_hi: usize,
    col: usize,
    dx5_max: usize,
) -> usize {
    let c5_lo = col.saturating_sub(dx5_max);
    let c5_hi = (col + dx5_max).min(FIELD_GRID_W - 1);
    let col5_mask = (!0u64 >> (63 - c5_hi)) & (!0u64 << c5_lo);
    let mut b5c = 0usize;
    let mut ny = b5_lo;
    while ny <= b5_hi {
        let ny_c = ny.min(FIELD_GRID_H - 1);
        if (row_occ[ny_c] & col5_mask) != 0 {
            let rb = ny_c * FIELD_GRID_W;
            b5c += cs.field_cell_start[(rb + c5_hi + 1).min(FIELD_CELLS)]
                .saturating_sub(cs.field_cell_start[(rb + c5_lo).min(FIELD_CELLS)])
                as usize;
        }
        ny += 1;
    }
    b5c
}

#[inline(never)]
fn is_split_rigid_boundary(
    cands_ti: &[u16; FIELD_CAND_CAP],
    src_ti: &[u16; MAX_MICRO],
    body_id: &[u8; MAX_MICRO],
    lim: usize,
    filter_age: bool,
    target_age: usize,
) -> bool {
    let lim = lim.min(FIELD_CAND_CAP);
    let mut bcnt = [0u8; 17];
    let mut cnt = 0usize;
    let mut k = 0usize;
    while k < lim {
        let raw = cands_ti[k];
        if !filter_age || ((raw >> 11) as usize) == target_age {
            let p = ((raw & 0x07FF) as usize).min(MAX_MICRO - 1);
            let tk = (src_ti[p] as usize).min(MAX_MICRO - 1);
            let bid = (body_id[tk] as usize).min(16);
            bcnt[bid] = bcnt[bid].saturating_add(1);
            cnt += 1;
        }
        k += 1;
    }
    if cnt < 6 {
        return false;
    }
    let zero_cnt = bcnt[0] as usize;
    if cnt - zero_cnt < 5 {
        return false;
    }
    let mut max_b = 0u8;
    let mut b = 1usize;
    while b <= 16 {
        if bcnt[b] > max_b {
            max_b = bcnt[b];
        }
        b += 1;
    }
    (zero_cnt + (max_b as usize)) * 4 < cnt * 3
}

#[inline(never)]
fn has_multi_body_boundary(
    cands_ti: &[u16; FIELD_CAND_CAP],
    src_ti: &[u16; MAX_MICRO],
    body_id: &[u8; MAX_MICRO],
    lim: usize,
    filter_age: bool,
    target_age: usize,
) -> bool {
    let lim = lim.min(FIELD_CAND_CAP);
    let mut first_bid = u8::MAX;
    let mut saw_diff = false;
    let mut cnt = 0usize;
    let mut k = 0usize;
    while k < lim {
        let raw = cands_ti[k];
        if !filter_age || ((raw >> 11) as usize) == target_age {
            let p = ((raw & 0x07FF) as usize).min(MAX_MICRO - 1);
            let tk = (src_ti[p] as usize).min(MAX_MICRO - 1);
            let bid = body_id[tk];
            if first_bid == u8::MAX {
                first_bid = bid;
            } else if bid != first_bid {
                saw_diff = true;
            }
            cnt += 1;
            if saw_diff && cnt >= 8 {
                return true;
            }
        }
        k += 1;
    }
    false
}

#[inline(never)]
fn run_ftle_bank_pass(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    frame_w: usize,
    frame_h: usize,
    q: f32,
    bank: u8,
    clean_packed_n: usize,
    is_secondary: bool,
    t_sec_by_age: &[(bool, f32); 13],
    row_occ: &mut [u64; FIELD_GRID_H],
    ftle_valid_cnt: &mut u32,
    max_ftle: &mut f32,
) {
    cs.field_cell_start.fill(0);

    let inv_fw = (FIELD_GRID_W as f32) / (frame_w as f32);
    let inv_fh = (FIELD_GRID_H as f32) / (frame_h as f32);
    let cell_w = (frame_w as f32) / (FIELD_GRID_W as f32);
    let cell_h = (frame_h as f32) / (FIELD_GRID_H as f32);
    let r_max = 32.0 * q;
    let r_max2 = r_max * r_max;
    let min_eig_px2 = (2.0 * q) * (2.0 * q);
    let radii = [16.0 * q, 24.0 * q, r_max];
    let r_thresh = [radii[0] * radii[0], radii[1] * radii[1], r_max2];
    let (dy_cells, row_dx) = compute_row_dx_spans(r_max, cell_w, cell_h);
    let r1_in_3x3 = 1.5 * cell_w >= radii[1] && 1.5 * cell_h >= radii[1];
    let mut row_dx_3 = [-1i8; FIELD_GRID_H];
    row_dx_3[0] = 1;
    row_dx_3[1] = 1;

    // H6 + M10: Contiguous CSR over mature bank-eligible anchor coordinates (x0, y0)
    // Packs (p | ((age - 8) << 11)) into field_grid_ti so try_solve_cell_ftle avoids SoA probes.
    let (af_slice, x0_slice, y0_slice) = if bank == 1 {
        (&ms.tracks.ftle_a_frame, &ms.tracks.ftle_a_x0, &ms.tracks.ftle_a_y0)
    } else {
        (&ms.tracks.ftle_b_frame, &ms.tracks.ftle_b_x0, &ms.tracks.ftle_b_y0)
    };
    let mut inserted = 0usize;
    let mut seen_ages_mask = 0u16;
    let mut single_age = 0usize;
    let p_lim = clean_packed_n.min(MAX_MICRO);
    let mut kp = 0usize;
    while kp < p_lim {
        let ti = (cs.rigid_seeds[kp] as usize).min(MAX_MICRO - 1);
        let af = af_slice[ti];
        if af != FTLE_INVALID_FRAME && frame_idx >= af {
            let age = frame_idx - af;
            if (8..=20).contains(&age) {
                let age_idx = (age - 8) as usize;
                if t_sec_by_age[age_idx].0 && af >= cs.slot_clean_since_frame[ti] {
                    let x0 = x0_slice[ti];
                    let y0 = y0_slice[ti];
                    if x0.is_finite() && y0.is_finite() {
                        let gx = clamped_grid_coord(x0, inv_fw, FIELD_GRID_W - 1);
                        let gy = clamped_grid_coord(y0, inv_fh, FIELD_GRID_H - 1);
                        let cell = gy * FIELD_GRID_W + gx;
                        let ins = inserted.min(MAX_MICRO - 1);
                        cs.field_cell_start[(cell + 1).min(FIELD_CELLS)] += 1;
                        cs.uf_parent[ins] = cell as i16;
                        cs.comp_next[ins] = (ti as i16) | ((age_idx as i16) << 11);
                        seen_ages_mask |= 1u16 << age_idx;
                        single_age = age_idx;
                        inserted += 1;
                    }
                }
            }
        }
        kp += 1;
    }
    if inserted < MIN_RIGID_MEMBERS {
        return;
    }
    let uniform_age = seen_ages_mask.count_ones() == 1;

    let mut row_cnt = [0u16; FIELD_GRID_H];
    let mut acc = 0u16;
    let mut rr = 0usize;
    while rr < FIELD_GRID_H {
        let row_acc0 = acc;
        let base_c = rr * FIELD_GRID_W;
        let mut r_mask = 0u64;
        let mut cc = 0usize;
        while cc < FIELD_GRID_W {
            let c = base_c + cc;
            let cnt = cs.field_cell_start[c + 1];
            cs.comp_head[c] = acc as i16;
            acc += cnt;
            cs.field_cell_start[c + 1] = acc;
            if cnt != 0 {
                r_mask |= 1u64 << cc;
            }
            cc += 1;
        }
        row_occ[rr] = r_mask;
        row_cnt[rr] = acc.saturating_sub(row_acc0);
        rr += 1;
    }
    let ins_n = inserted.min(MAX_MICRO);
    let mut k_ins = 0usize;
    while k_ins < ins_n {
        let cell = (cs.uf_parent[k_ins] as usize).min(FIELD_CELLS - 1);
        let dst = (cs.comp_head[cell] as usize).min(MAX_MICRO - 1);
        cs.comp_head[cell] = (dst + 1) as i16;
        let packed = cs.comp_next[k_ins] as u16;
        let tk = ((packed & 0x07FF) as usize).min(MAX_MICRO - 1);
        cs.field_grid_ti[dst] = (dst as u16) | (packed & 0xF800);
        cs.field_grid_src_ti[dst] = tk as u16;
        cs.field_grid_x[dst] = x0_slice[tk];
        cs.field_grid_y[dst] = y0_slice[tk];
        cs.field_grid_aux[dst] = [ms.tracks.pos_x[tk], ms.tracks.pos_y[tk]];
        k_ins += 1;
    }

    const VALID_COLS_MASK: u64 = (1u64 << FIELD_GRID_W) - 1;
    let mut row = 0usize;
    while row < FIELD_GRID_H {
        let row_cell_base = row * FIELD_GRID_W;
        let mut row_todo: u64 = if !is_secondary {
            VALID_COLS_MASK
        } else {
            let mut m = 0u64;
            let row_valid = &cs.cell_valid_mask[row_cell_base..row_cell_base + FIELD_GRID_W];
            let mut c = 0usize;
            while c < FIELD_GRID_W {
                if (row_valid[c] & CELL_VALID_FTLE) == 0 {
                    m |= 1u64 << c;
                }
                c += 1;
            }
            if m == 0 {
                row += 1;
                continue;
            }
            m
        };
        let b5_lo = (row as isize - dy_cells).max(0) as usize;
        let b5_hi = (row as isize + dy_cells).min((FIELD_GRID_H - 1) as isize) as usize;
        let (b3_lo, b3_hi, band3_tot, band3_occ, band5_tot, band5_occ) =
            band3_and_band5_summary(&row_cnt, row_occ, row, b5_lo, b5_hi);
        if band5_tot < MIN_RIGID_MEMBERS {
            row += 1;
            continue;
        }

        let r_lo_px = if row == 0 {
            f32::NEG_INFINITY
        } else {
            (row as f32) * cell_h
        };
        let r_hi_px = if row + 1 == FIELD_GRID_H {
            f32::INFINITY
        } else {
            ((row + 1) as f32) * cell_h
        };
        let dx5_max = row_dx[0].max(0) as usize;
        let mut has_5x5 = band5_occ;
        let mut sh = 1usize;
        while sh <= dx5_max {
            has_5x5 |= (band5_occ << sh) | (band5_occ >> sh);
            sh += 1;
        }
        row_todo &= has_5x5 & VALID_COLS_MASK;
        let yc = (row as f32 + 0.5) * cell_h;
        while row_todo != 0 {
            let col = (row_todo.trailing_zeros() as usize).min(FIELD_GRID_W - 1);
            row_todo &= row_todo - 1;
            let cell = row_cell_base + col;
            let c3_lo = col.saturating_sub(1);
            let c3_hi = (col + 1).min(FIELD_GRID_W - 1);
            let col3_mask = (7u64 << col) >> 1;
            let (s0, e0, s1, e1, s2, e2, box3_cnt) = if r1_in_3x3
                && band3_tot >= MIN_RIGID_MEMBERS
                && (band3_occ & col3_mask) != 0
            {
                let rb0 = b3_lo * FIELD_GRID_W;
                let a0 = cs.field_cell_start[(rb0 + c3_lo).min(FIELD_CELLS)] as usize;
                let b0 = cs.field_cell_start[(rb0 + c3_hi + 1).min(FIELD_CELLS)] as usize;
                let (a1, b1) = if b3_lo < b3_hi {
                    let rb1 = (b3_lo + 1) * FIELD_GRID_W;
                    (
                        cs.field_cell_start[(rb1 + c3_lo).min(FIELD_CELLS)] as usize,
                        cs.field_cell_start[(rb1 + c3_hi + 1).min(FIELD_CELLS)] as usize,
                    )
                } else {
                    (0, 0)
                };
                let (a2, b2) = if b3_lo + 2 <= b3_hi {
                    let rb2 = b3_hi * FIELD_GRID_W;
                    (
                        cs.field_cell_start[(rb2 + c3_lo).min(FIELD_CELLS)] as usize,
                        cs.field_cell_start[(rb2 + c3_hi + 1).min(FIELD_CELLS)] as usize,
                    )
                } else {
                    (0, 0)
                };
                let cnt = b0.saturating_sub(a0) + b1.saturating_sub(a1) + b2.saturating_sub(a2);
                (a0, b0, a1, b1, a2, b2, cnt)
            } else {
                (0, 0, 0, 0, 0, 0, 0)
            };
            let mut pass = if r1_in_3x3 && box3_cnt >= MIN_RIGID_MEMBERS {
                0usize
            } else {
                1usize
            };
            let mut box5_cnt = usize::MAX;
            if pass == 1 {
                let b5c = count_box5_tracks(cs, row_occ, b5_lo, b5_hi, col, dx5_max);
                if b5c < MIN_RIGID_MEMBERS {
                    continue;
                }
                box5_cnt = b5c;
            }
            let c_lo_px = if col == 0 {
                f32::NEG_INFINITY
            } else {
                (col as f32) * cell_w
            };
            let c_hi_px = if col + 1 == FIELD_GRID_W {
                f32::INFINITY
            } else {
                ((col + 1) as f32) * cell_w
            };
            let xc = (col as f32 + 0.5) * cell_w;
            let mut solved = false;
            let mut prev_a = 13usize;
            let mut prev_cnt = 0usize;
            let mut prev_pass2 = false;
            let mut skip_pass1 = false;
            while !solved && pass < 2 {
                let (dy_p, rdx_p, r_lim2, ri_start, ri_end, part_mode, box_cnt) =
                    if pass == 0 && r1_in_3x3 {
                        let (rs0, pm0) = if box3_cnt < 10 {
                            (1usize, 0u8)
                        } else {
                            (0usize, 1u8)
                        };
                        (1isize, &row_dx_3, r_thresh[1], rs0, 2usize, pm0, box3_cnt)
                    } else {
                        if box5_cnt == usize::MAX {
                            let b5c = count_box5_tracks(cs, row_occ, b5_lo, b5_hi, col, dx5_max);
                            if b5c < MIN_RIGID_MEMBERS {
                                break;
                            }
                            box5_cnt = b5c;
                        }
                        let (rs, pm) = if r1_in_3x3 && pass == 1 {
                            (2usize, 0u8)
                        } else {
                            (0usize, 2u8)
                        };
                        pass = 1;
                        (dy_cells, &row_dx, r_max2, rs, 3usize, pm, box5_cnt)
                    };
                let (end0, end1, cand_n) = if box_cnt <= FIELD_CAND_CAP {
                    let pk = if dy_p == 1 {
                        if part_mode == 1 {
                            gather_3x3_split_le24(
                                cs,
                                s0,
                                e0,
                                s1,
                                e1,
                                s2,
                                e2,
                                xc,
                                yc,
                                r_thresh[0],
                                r_lim2,
                                MIN_RIGID_MEMBERS,
                            )
                        } else {
                            gather_3x3_flat_le24(
                                cs,
                                s0,
                                e0,
                                s1,
                                e1,
                                s2,
                                e2,
                                xc,
                                yc,
                                r_lim2,
                                MIN_RIGID_MEMBERS,
                            )
                        }
                    } else {
                        gather_wide_le24(
                            cs,
                            row,
                            col,
                            xc,
                            yc,
                            r_thresh[0],
                            r_thresh[1],
                            r_lim2,
                            dy_p,
                            rdx_p,
                            MIN_RIGID_MEMBERS,
                            part_mode,
                        )
                    };
                    (
                        (pk & 0xFF) as usize,
                        ((pk >> 8) & 0xFF) as usize,
                        ((pk >> 16) & 0xFF) as usize,
                    )
                } else {
                    gather_cell_candidates_diverse(
                        cs,
                        row,
                        col,
                        xc,
                        yc,
                        r_thresh[0],
                        r_thresh[1],
                        r_lim2,
                        dy_p,
                        rdx_p,
                        MIN_RIGID_MEMBERS,
                        part_mode,
                    )
                };
                if cand_n >= MIN_RIGID_MEMBERS {
                    let lim = cand_n.min(FIELD_CAND_CAP);
                    let end_ri = [end0.min(lim), end1.min(lim), lim];
                    let mut best_a_ri = [13usize; 3];
                    let mut best_cnt_ri = [0usize; 3];
                    let mut all_same_age = uniform_age;
                    if uniform_age {
                        best_a_ri = [single_age; 3];
                        best_cnt_ri = end_ri;
                    } else if lim > 0 {
                        let cti = &cs.cell_cands_ti[..FIELD_CAND_CAP];
                        let mut age_cnt = [0u8; 13];
                        let mut k = 0usize;
                        let mut cur_ba = 13usize;
                        let mut cur_bc = 0usize;
                        let mut ri_b = 0usize;
                        while ri_b < ri_end {
                            let eb = end_ri[ri_b].min(FIELD_CAND_CAP);
                            while k < eb {
                                let a = (cti[k.min(FIELD_CAND_CAP - 1)] >> 11) as usize;
                                if a < 13 {
                                    let c_new = age_cnt[a].saturating_add(1);
                                    age_cnt[a] = c_new;
                                    let cu = c_new as usize;
                                    if cu > cur_bc
                                        || (cu == cur_bc && (cur_ba == 13 || a > cur_ba))
                                    {
                                        cur_bc = cu;
                                        cur_ba = a;
                                    }
                                }
                                k += 1;
                            }
                            if eb >= MIN_RIGID_MEMBERS && ri_b >= ri_start {
                                best_a_ri[ri_b] = cur_ba;
                                best_cnt_ri[ri_b] = cur_bc;
                            }
                            ri_b += 1;
                        }
                        all_same_age = cur_bc == lim;
                    }

                    let max_cnt = best_cnt_ri[ri_end.saturating_sub(1).min(2)];
                    if max_cnt >= MIN_RIGID_MEMBERS {
                        let mut fm = FtlePrefixMoments::zeroed();
                        let mut ri = ri_start;
                        while ri < ri_end {
                            let ba = best_a_ri[ri];
                            let bc = best_cnt_ri[ri];
                            if ba < 13 && bc >= MIN_RIGID_MEMBERS && t_sec_by_age[ba].0 {
                                let same_unsolved = (ri > ri_start || lim < FIELD_CAND_CAP)
                                    && ba == prev_a
                                    && bc == prev_cnt
                                    && !prev_pass2;
                                if !same_unsolved {
                                    if cs.rigid_body_count >= 2
                                        && is_split_rigid_boundary(
                                            &cs.cell_cands_ti,
                                            &cs.field_grid_src_ti,
                                            &ms.tracks.body_id,
                                            end_ri[ri],
                                            !all_same_age,
                                            ba,
                                        )
                                    {
                                        prev_a = ba;
                                        prev_cnt = bc;
                                        prev_pass2 = false;
                                        if pass == 0
                                            && (ri >= 1
                                                || end_ri[1] == lim
                                                || end_ri[1] == end_ri[ri])
                                        {
                                            skip_pass1 = true;
                                            break;
                                        }
                                        ri += 1;
                                        continue;
                                    }
                                    if ba != prev_a || ri == ri_start {
                                        fm.m = 0;
                                    }
                                    let t_sec = t_sec_by_age[ba].1;
                                    let (ok_f, reached_pass2, incoherent, ftle, cand_m, all_inliers) =
                                        try_solve_cell_ftle(
                                            cs,
                                            &mut fm,
                                            end_ri[ri],
                                            ba,
                                            all_same_age,
                                            t_sec,
                                            q,
                                            min_eig_px2,
                                            radii[ri],
                                            ri + 1 < ri_end,
                                        );
                                    prev_a = ba;
                                    prev_cnt = bc;
                                    prev_pass2 = reached_pass2;
                                    if ok_f {
                                        let cell_c = cell.min(FIELD_CELLS - 1);
                                        let base = cell_c * FIELD_EXPORT_STRIDE;
                                        cs.export_field[base + 4] = ftle;
                                        cs.cell_valid_mask[cell_c] |= CELL_VALID_FTLE;
                                        if ftle > *max_ftle {
                                            *max_ftle = ftle;
                                        }
                                        *ftle_valid_cnt += 1;

                                        // M7: Restrict Lagrangian track FTLE strictly to accepted IRLS inliers,
                                        // prioritizing each track's home anchor cell when overlapped by neighbors.
                                        let cand_lim = cand_m.min(FIELD_CAND_CAP);
                                        let x0_ref = fm.x0_ref;
                                        let y0_ref = fm.y0_ref;
                                        let fti = &cs.ftle_cands_ti[..FIELD_CAND_CAP];
                                        let fw = &cs.ftle_cands_w[..FIELD_CAND_CAP];
                                        let cdx = &cs.cand_dx[..FIELD_CAND_CAP];
                                        let cdy = &cs.cand_dy[..FIELD_CAND_CAP];
                                        let src_ti = &cs.field_grid_src_ti[..MAX_MICRO];
                                        let mm = &mut cs.member_mask[..MAX_MICRO];
                                        let ftle_val = &mut ms.tracks.ftle_val[..MAX_MICRO];
                                        let mut k = 0usize;
                                        while k < cand_lim {
                                            let ks = k.min(FIELD_CAND_CAP - 1);
                                            if all_inliers || fw[ks] > 0.0 {
                                                let p = (fti[ks] as usize).min(MAX_MICRO - 1);
                                                let tk = (src_ti[p] as usize).min(MAX_MICRO - 1);
                                                let m_cur = mm[tk];
                                                if m_cur != 2 {
                                                    let x0 = x0_ref + cdx[ks];
                                                    let y0 = y0_ref + cdy[ks];
                                                    if x0 >= c_lo_px
                                                        && x0 < c_hi_px
                                                        && y0 >= r_lo_px
                                                        && y0 < r_hi_px
                                                    {
                                                        ftle_val[tk] = ftle;
                                                        mm[tk] = 2;
                                                    } else if m_cur == 0 {
                                                        ftle_val[tk] = ftle;
                                                        mm[tk] = 1;
                                                    }
                                                }
                                            }
                                            k += 1;
                                        }
                                        solved = true;
                                        break;
                                    }
                                    if pass == 0
                                        && (((ri == 1 || end_ri[1] == end_ri[ri]) && bc >= 7)
                                            || (incoherent
                                                && ((ri == 0 && bc >= 8)
                                                    || ((uniform_age || best_a_ri[1] == ba)
                                                        && bc >= 7)
                                                    || end_ri[1] < MIN_RIGID_MEMBERS))
                                            || (cs.rigid_body_count >= 2
                                                && bc >= 8
                                                && (ri == 1
                                                    || ri + 1 == ri_end
                                                    || end_ri[1] == end_ri[ri])
                                                && has_multi_body_boundary(
                                                    &cs.cell_cands_ti,
                                                    &cs.field_grid_src_ti,
                                                    &ms.tracks.body_id,
                                                    bc,
                                                    !all_same_age,
                                                    ba,
                                                )))
                                    {
                                        skip_pass1 = true;
                                        break;
                                    }
                                }
                            }
                            ri += 1;
                        }
                    }
                }
                if pass == 0 && skip_pass1 {
                    break;
                }
                if pass == 0 && !solved && box3_cnt <= FIELD_CAND_CAP && !prev_pass2 {
                    let b5c = count_box5_tracks(cs, row_occ, b5_lo, b5_hi, col, dx5_max);
                    if b5c < MIN_RIGID_MEMBERS || b5c == cand_n {
                        break;
                    }
                    box5_cnt = b5c;
                }
                pass += 1;
            }
        }
        row += 1;
    }
}

#[inline(always)]
fn slot_ftle_window_eligible(
    ms: &MicroSwarmState,
    cs: &ContinuumState,
    ti: usize,
    frame_idx: u32,
    t_sec_by_age: &[(bool, f32); 13],
) -> (bool, bool) {
    let clean_since = cs.slot_clean_since_frame[ti];
    let af = ms.tracks.ftle_a_frame[ti];
    let bf = ms.tracks.ftle_b_frame[ti];
    let a_ok = af != FTLE_INVALID_FRAME
        && af >= clean_since
        && frame_idx >= af
        && {
            let age = frame_idx - af;
            (8..=20).contains(&age) && t_sec_by_age[(age - 8) as usize].0
        };
    let b_ok = bf != FTLE_INVALID_FRAME
        && bf >= clean_since
        && frame_idx >= bf
        && {
            let age = frame_idx - bf;
            (8..=20).contains(&age) && t_sec_by_age[(age - 8) as usize].0
        };
    (a_ok, b_ok)
}

/// Phase 3: Adaptive `40x22` Eulerian velocity gradient (`16 -> 24 -> 32 HD px`, `M5`, `M6`) &
/// anchor-space robust staggered Cauchy-Green FTLE solver (`H6`, `H7`, `M7`, `M10`, `#96`).
#[inline(never)]
pub fn solve_eulerian_grad_and_ftle(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    frame_w: usize,
    frame_h: usize,
    dt_sec: f32,
) {
    if ms.swarm_active == 0 {
        cs.reset();
        cs.initialized = 0;
        return;
    }
    if cs.initialized == 0 {
        cs.reset();
    }

    let mut t_sec_by_age = [(false, 0.0f32); 13];
    let mut a = 0usize;
    while a < 13 {
        let age = (a as u32) + 8;
        if frame_idx >= age {
            t_sec_by_age[a] = cs.elapsed_since_anchor(frame_idx - age, frame_idx);
        }
        a += 1;
    }

    let min_age = min_continuum_age(ms);
    let use_cached = cs.lists_frame_idx != u32::MAX && cs.lists_frame_idx == frame_idx;
    let dense_split = ms.confirmed_count >= 960;
    if dense_split && frame_idx < 6 {
        if cs.export_field_dirty != 0 {
            let mut c = 0usize;
            while c < FIELD_CELLS {
                let base = c * FIELD_EXPORT_STRIDE;
                cs.export_field[base] = 0.0;
                cs.export_field[base + 1] = 0.0;
                cs.export_field[base + 2] = 0.0;
                cs.export_field[base + 3] = 0.0;
                cs.export_field[base + 4] = -999.0;
                cs.export_field[base + 5] = 0.0;
                cs.cell_valid_mask[c] &= !(CELL_VALID_GRAD | CELL_VALID_FTLE);
                c += 1;
            }
            reset_track_ftle_sentinels(ms);
            cs.export_field_dirty = 0;
        }
        cs.cached_grad_ftle_frame = u32::MAX;
        cs.cached_ftle_frame = u32::MAX;
        return;
    }

    let hold_grad = dense_split
        && frame_idx >= 7
        && (frame_idx & 1) == 1
        && cs.cached_grad_ftle_frame == frame_idx - 1;
    let skip_ftle_solve = dense_split && (!hold_grad || frame_idx < 17);
    let mut hold_ftle = dense_split
        && !hold_grad
        && frame_idx >= 18
        && cs.cached_ftle_frame == frame_idx - 1;

    if hold_grad && skip_ftle_solve {
        let mut mature_cnt = 0u32;
        let loop_lim = if use_cached {
            (cs.clean_slots_len as usize).min(MAX_MICRO)
        } else {
            MAX_MICRO
        };
        let mut ci = 0usize;
        while ci < loop_lim {
            let ti = if use_cached {
                (cs.clean_slots[ci] as usize).min(MAX_MICRO - 1)
            } else {
                ci
            };
            ci += 1;
            if !use_cached && !(is_clean_confirmed(ms, ti) && ms.tracks.age[ti] >= min_age) {
                if ms.tracks.state[ti] != MICRO_STATE_INACTIVE {
                    ms.tracks.ftle_val[ti] = -999.0;
                }
                continue;
            }
            let (a_ok, b_ok) = slot_ftle_window_eligible(ms, cs, ti, frame_idx, &t_sec_by_age);
            if a_ok || b_ok {
                mature_cnt += 1;
            } else {
                ms.tracks.ftle_val[ti] = -999.0;
            }
        }
        if !((cs.cached_grad_ftle_mask & VALIDITY_BIT_FTLE) != 0 && mature_cnt == 0) {
            cs.valid_field_cells = cs.cached_valid_field_cells;
            cs.max_abs_vorticity_s1 = cs.cached_max_abs_vorticity_s1;
            cs.rms_vorticity_s1 = cs.cached_rms_vorticity_s1;
            cs.rms_divergence_s1 = cs.cached_rms_divergence_s1;
            cs.max_ftle_s1 = cs.cached_max_ftle_s1;
            cs.ftle_cells = cs.cached_ftle_cells;
            cs.diagnostic_validity_mask = (cs.diagnostic_validity_mask
                & !(VALIDITY_BIT_GRAD | VALIDITY_BIT_FTLE))
                | (cs.cached_grad_ftle_mask & (VALIDITY_BIT_GRAD | VALIDITY_BIT_FTLE));
            cs.ftle_mature_tracks = mature_cnt;
            return;
        }
    }

    cs.diagnostic_validity_mask &= !(VALIDITY_BIT_GRAD | VALIDITY_BIT_FTLE);
    let q = ((frame_w as f32) / 640.0).max(0.25);
    let mut row_occ = [0u64; FIELD_GRID_H];
    let mut mature_cnt = 0u32;
    let mut bank_a_eligible = 0u32;
    let mut bank_b_eligible = 0u32;
    let mut clean_n = 0usize;

    if !hold_grad {
        cs.cached_grad_ftle_frame = u32::MAX;
        let mut c = 0usize;
        if hold_ftle {
            while c < FIELD_CELLS {
                let base = c * FIELD_EXPORT_STRIDE;
                cs.export_field[base] = 0.0;
                cs.export_field[base + 1] = 0.0;
                cs.export_field[base + 2] = 0.0;
                cs.export_field[base + 3] = 0.0;
                cs.export_field[base + 5] = 0.0;
                cs.cell_valid_mask[c] &= !CELL_VALID_GRAD;
                c += 1;
            }
        } else {
            while c < FIELD_CELLS {
                let base = c * FIELD_EXPORT_STRIDE;
                cs.export_field[base] = 0.0;
                cs.export_field[base + 1] = 0.0;
                cs.export_field[base + 2] = 0.0;
                cs.export_field[base + 3] = 0.0;
                cs.export_field[base + 4] = -999.0; // #96 invalid FTLE sentinel
                cs.export_field[base + 5] = 0.0;
                cs.cell_valid_mask[c] &= !(CELL_VALID_GRAD | CELL_VALID_FTLE);
                c += 1;
            }
            cs.max_ftle_s1 = 0.0;
            cs.ftle_cells = 0;
            cs.export_field_dirty = 0;
            reset_track_ftle_sentinels(ms);
        }
        cs.valid_field_cells = 0;
        cs.max_abs_vorticity_s1 = 0.0;
        cs.rms_vorticity_s1 = 0.0;
        cs.rms_divergence_s1 = 0.0;
        cs.ftle_mature_tracks = 0;

        let cell_w = (frame_w as f32) / (FIELD_GRID_W as f32);
        let cell_h = (frame_h as f32) / (FIELD_GRID_H as f32);
        let r_max = 32.0 * q;
        let r_max2 = r_max * r_max;
        let min_eig_px2 = (2.0 * q) * (2.0 * q);
        let radii = [16.0 * q, 24.0 * q, r_max];
        let (dy_cells, row_dx) = compute_row_dx_spans(r_max, cell_w, cell_h);

        // Phase 3A: Fused 40x22 CSR grid count + mature staggered FTLE window census
        cs.field_cell_start.fill(0);
        let inv_fw = (FIELD_GRID_W as f32) / (frame_w as f32);
        let inv_fh = (FIELD_GRID_H as f32) / (frame_h as f32);
        let mut packed_n = 0usize;
        let loop_lim = if use_cached {
            (cs.clean_slots_len as usize).min(MAX_MICRO)
        } else {
            MAX_MICRO
        };
        let mut ci = 0usize;
        while ci < loop_lim {
            let ti = if use_cached {
                (cs.clean_slots[ci] as usize).min(MAX_MICRO - 1)
            } else {
                ci
            };
            ci += 1;
            if !use_cached && !(is_clean_confirmed(ms, ti) && ms.tracks.age[ti] >= min_age) {
                if hold_ftle && ms.tracks.state[ti] != MICRO_STATE_INACTIVE {
                    ms.tracks.ftle_val[ti] = -999.0;
                }
                continue;
            }
            cs.rigid_seeds[clean_n.min(MAX_MICRO - 1)] = ti as u16;
            clean_n += 1;
            let px = ms.tracks.pos_x[ti];
            let py = ms.tracks.pos_y[ti];
            if px.is_finite() && py.is_finite() {
                let gx = clamped_grid_coord(px, inv_fw, FIELD_GRID_W - 1);
                let gy = clamped_grid_coord(py, inv_fh, FIELD_GRID_H - 1);
                let cell = gy * FIELD_GRID_W + gx;
                cs.field_cell_start[cell + 1] += 1;
                let pn = packed_n.min(MAX_MICRO - 1);
                cs.uf_parent[pn] = cell as i16;
                cs.comp_next[pn] = ti as i16;
                packed_n += 1;
            }
            let (a_ok, b_ok) = slot_ftle_window_eligible(ms, cs, ti, frame_idx, &t_sec_by_age);
            if a_ok {
                bank_a_eligible += 1;
            }
            if b_ok {
                bank_b_eligible += 1;
            }
            if a_ok || b_ok {
                mature_cnt += 1;
            } else if hold_ftle {
                ms.tracks.ftle_val[ti] = -999.0;
            }
        }
        cs.ftle_mature_tracks = mature_cnt;
        if hold_ftle {
            if (cs.cached_grad_ftle_mask & VALIDITY_BIT_FTLE) != 0 && mature_cnt == 0 {
                hold_ftle = false;
                let mut c = 0usize;
                while c < FIELD_CELLS {
                    let base = c * FIELD_EXPORT_STRIDE;
                    cs.export_field[base + 4] = -999.0;
                    cs.cell_valid_mask[c] &= !CELL_VALID_FTLE;
                    c += 1;
                }
                reset_track_ftle_sentinels(ms);
                cs.max_ftle_s1 = 0.0;
                cs.ftle_cells = 0;
                cs.cached_ftle_frame = u32::MAX;
                cs.cached_grad_ftle_mask &= !VALIDITY_BIT_FTLE;
                cs.cached_max_ftle_s1 = 0.0;
                cs.cached_ftle_cells = 0;
            } else {
                cs.max_ftle_s1 = cs.cached_max_ftle_s1;
                cs.ftle_cells = cs.cached_ftle_cells;
                cs.diagnostic_validity_mask |= cs.cached_grad_ftle_mask & VALIDITY_BIT_FTLE;
            }
        }
    let mut row_cnt = [0u16; FIELD_GRID_H];
    let mut acc = 0u16;
    let mut rr = 0usize;
    while rr < FIELD_GRID_H {
        let row_acc0 = acc;
        let base_c = rr * FIELD_GRID_W;
        let mut r_mask = 0u64;
        let mut cc = 0usize;
        while cc < FIELD_GRID_W {
            let c = base_c + cc;
            let cnt = cs.field_cell_start[c + 1];
            cs.comp_head[c] = acc as i16;
            acc += cnt;
            cs.field_cell_start[c + 1] = acc;
            if cnt != 0 {
                r_mask |= 1u64 << cc;
            }
            cc += 1;
        }
        row_occ[rr] = r_mask;
        row_cnt[rr] = acc.saturating_sub(row_acc0);
        rr += 1;
    }
    let kp_lim = packed_n.min(MAX_MICRO);
    let mut kp = 0usize;
    while kp < kp_lim {
        let cell = (cs.uf_parent[kp] as usize).min(FIELD_CELLS - 1);
        let dst = (cs.comp_head[cell] as usize).min(MAX_MICRO - 1);
        cs.comp_head[cell] = (dst + 1) as i16;
        let tk = (cs.comp_next[kp] as usize).min(MAX_MICRO - 1);
        cs.field_grid_ti[dst] = dst as u16;
        cs.field_grid_src_ti[dst] = tk as u16;
        cs.field_grid_x[dst] = ms.tracks.pos_x[tk];
        cs.field_grid_y[dst] = ms.tracks.pos_y[tk];
        cs.field_grid_aux[dst] = [ms.tracks.vel_x[tk], ms.tracks.vel_y[tk]];
        kp += 1;
    }

    let mut grad_valid_cnt = 0u32;
    let mut sum_vort2 = 0.0f64;
    let mut sum_div2 = 0.0f64;
    let mut max_abs_vort = 0.0f32;

    if packed_n >= 4 {
        const VALID_COLS_MASK: u64 = (1u64 << FIELD_GRID_W) - 1;
        let r0_2 = radii[0] * radii[0];
        let r1_2 = radii[1] * radii[1];
        let r1_in_3x3 = 1.5 * cell_w >= radii[1] && 1.5 * cell_h >= radii[1];
        let mut row_dx_3 = [-1i8; FIELD_GRID_H];
        row_dx_3[0] = 1;
        row_dx_3[1] = 1;
        let mut row = 0usize;
        while row < FIELD_GRID_H {
            let b5_lo = (row as isize - dy_cells).max(0) as usize;
            let b5_hi = (row as isize + dy_cells).min((FIELD_GRID_H - 1) as isize) as usize;
            let (b3_lo, b3_hi, band3_tot, band3_occ, band5_tot, band5_occ) =
                band3_and_band5_summary(&row_cnt, &row_occ, row, b5_lo, b5_hi);
            if band5_tot < 4 {
                row += 1;
                continue;
            }
            let dx5_max = row_dx[0].max(0) as usize;
            let mut has_5x5 = band5_occ;
            let mut sh = 1usize;
            while sh <= dx5_max {
                has_5x5 |= (band5_occ << sh) | (band5_occ >> sh);
                sh += 1;
            }
            let mut row_todo = has_5x5 & VALID_COLS_MASK;
            let yc = (row as f32 + 0.5) * cell_h;
            while row_todo != 0 {
                let col = (row_todo.trailing_zeros() as usize).min(FIELD_GRID_W - 1);
                row_todo &= row_todo - 1;
                let c3_lo = col.saturating_sub(1);
                let c3_hi = (col + 1).min(FIELD_GRID_W - 1);
                let col3_mask = (7u64 << col) >> 1;
                let (s0, e0, s1, e1, s2, e2, box3_cnt) = if r1_in_3x3
                    && band3_tot >= 4
                    && (band3_occ & col3_mask) != 0
                {
                    let rb0 = b3_lo * FIELD_GRID_W;
                    let a0 = cs.field_cell_start[(rb0 + c3_lo).min(FIELD_CELLS)] as usize;
                    let b0 = cs.field_cell_start[(rb0 + c3_hi + 1).min(FIELD_CELLS)] as usize;
                    let (a1, b1) = if b3_lo < b3_hi {
                        let rb1 = (b3_lo + 1) * FIELD_GRID_W;
                        (
                            cs.field_cell_start[(rb1 + c3_lo).min(FIELD_CELLS)] as usize,
                            cs.field_cell_start[(rb1 + c3_hi + 1).min(FIELD_CELLS)] as usize,
                        )
                    } else {
                        (0, 0)
                    };
                    let (a2, b2) = if b3_lo + 2 <= b3_hi {
                        let rb2 = b3_hi * FIELD_GRID_W;
                        (
                            cs.field_cell_start[(rb2 + c3_lo).min(FIELD_CELLS)] as usize,
                            cs.field_cell_start[(rb2 + c3_hi + 1).min(FIELD_CELLS)] as usize,
                        )
                    } else {
                        (0, 0)
                    };
                    let cnt =
                        b0.saturating_sub(a0) + b1.saturating_sub(a1) + b2.saturating_sub(a2);
                    (a0, b0, a1, b1, a2, b2, cnt)
                } else {
                    (0, 0, 0, 0, 0, 0, 0)
                };
                let mut pass = if r1_in_3x3 && box3_cnt >= 4 {
                    0usize
                } else {
                    1usize
                };
                let mut box5_cnt = usize::MAX;
                if pass == 1 {
                    let b5c = count_box5_tracks(cs, &row_occ, b5_lo, b5_hi, col, dx5_max);
                    if b5c < 4 {
                        continue;
                    }
                    box5_cnt = b5c;
                }
                let xc = (col as f32 + 0.5) * cell_w;
                let cell = row * FIELD_GRID_W + col;
                let base = cell * FIELD_EXPORT_STRIDE;
                let mut solved = false;
                let mut prev_m = 0usize;
                let mut prev_pass2 = false;
                let mut skip_pass1 = false;
                while !solved && pass < 2 {
                    let (dy_p, rdx_p, r_lim2, ri_start, ri_end, part_mode, box_cnt) =
                        if pass == 0 && r1_in_3x3 {
                            let (rs0, pm0) = if box3_cnt < 6 {
                                (1usize, 0u8)
                            } else {
                                (0usize, 1u8)
                            };
                            (1isize, &row_dx_3, r1_2, rs0, 2usize, pm0, box3_cnt)
                        } else {
                            if box5_cnt == usize::MAX {
                                let b5c =
                                    count_box5_tracks(cs, &row_occ, b5_lo, b5_hi, col, dx5_max);
                                if b5c < 4 {
                                    break;
                                }
                                box5_cnt = b5c;
                            }
                            let (rs, pm) = if r1_in_3x3 && pass == 1 {
                                (2usize, 0u8)
                            } else {
                                (0usize, 2u8)
                            };
                            pass = 1;
                            (dy_cells, &row_dx, r_max2, rs, 3usize, pm, box5_cnt)
                        };
                    let (end0, end1, cand_n) = if box_cnt <= FIELD_CAND_CAP {
                        let pk = if dy_p == 1 {
                            if part_mode == 1 {
                                gather_3x3_split_le24(
                                    cs, s0, e0, s1, e1, s2, e2, xc, yc, r0_2, r_lim2, 4,
                                )
                            } else {
                                gather_3x3_flat_le24(
                                    cs, s0, e0, s1, e1, s2, e2, xc, yc, r_lim2, 4,
                                )
                            }
                        } else {
                            gather_wide_le24(
                                cs,
                                row,
                                col,
                                xc,
                                yc,
                                r0_2,
                                r1_2,
                                r_lim2,
                                dy_p,
                                rdx_p,
                                4,
                                part_mode,
                            )
                        };
                        (
                            (pk & 0xFF) as usize,
                            ((pk >> 8) & 0xFF) as usize,
                            ((pk >> 16) & 0xFF) as usize,
                        )
                    } else {
                        gather_cell_candidates_diverse(
                            cs,
                            row,
                            col,
                            xc,
                            yc,
                            r0_2,
                            r1_2,
                            r_lim2,
                            dy_p,
                            rdx_p,
                            4,
                            part_mode,
                        )
                    };
                    if cand_n >= 4 {
                        let end_ri = [end0.min(cand_n), end1.min(cand_n), cand_n];
                        let p0 = (cs.cell_cands_ti[0] as usize).min(MAX_MICRO - 1);
                        let uv0 = cs.field_grid_aux[p0];
                        let mut pm = GradPrefixMoments::new(uv0[0], uv0[1]);
                        let mut ri = ri_start;
                        while ri < ri_end {
                            let cand_end = end_ri[ri];
                            let same_unsolved = prev_m > 0
                                && !prev_pass2
                                && cand_end == prev_m
                                && (ri > ri_start || cand_end < FIELD_CAND_CAP);
                            if cand_end < 4 || same_unsolved {
                                ri += 1;
                                continue;
                            }
                            if cs.rigid_body_count >= 2
                                && is_split_rigid_boundary(
                                    &cs.cell_cands_ti,
                                    &cs.field_grid_src_ti,
                                    &ms.tracks.body_id,
                                    cand_end,
                                    false,
                                    0,
                                )
                            {
                                prev_m = cand_end;
                                prev_pass2 = false;
                                if pass == 0
                                    && (ri >= 1 || end_ri[1] == cand_n || end_ri[1] == cand_end)
                                {
                                    skip_pass1 = true;
                                    break;
                                }
                                ri += 1;
                                continue;
                            }
                            let (ok, grad_flags, u_c, v_c, vort, div, w_sum) =
                                try_solve_cell_grad_v(
                                    cs,
                                    &mut pm,
                                    cand_end,
                                    xc,
                                    yc,
                                    radii[ri],
                                    min_eig_px2,
                                    q,
                                    dt_sec,
                                );
                            prev_m = pm.m;
                            prev_pass2 = (grad_flags & GRAD_FLAG_PASS2) != 0;
                            if ok
                                && u_c.is_finite()
                                && v_c.is_finite()
                                && vort.is_finite()
                                && div.is_finite()
                            {
                                cs.export_field[base] = u_c;
                                cs.export_field[base + 1] = v_c;
                                cs.export_field[base + 2] = vort;
                                cs.export_field[base + 3] = div;
                                cs.export_field[base + 5] = finite_or_zero(w_sum).max(1e-4);
                                cs.cell_valid_mask[cell] |= CELL_VALID_GRAD;

                                let av = vort.abs();
                                if av > max_abs_vort {
                                    max_abs_vort = av;
                                }
                                sum_vort2 += (vort as f64) * (vort as f64);
                                sum_div2 += (div as f64) * (div as f64);
                                grad_valid_cnt += 1;
                                solved = true;
                                break;
                            }
                            if pass == 0
                                && (((ri == 1 || end_ri[1] == pm.m) && pm.m >= 6)
                                    || ((grad_flags & GRAD_FLAG_INCOHERENT_CORE) != 0
                                        && (ri == 1
                                            || pm.m >= 5
                                            || end_ri[1] < 4
                                            || end_ri[1] == pm.m))
                                    || (cs.rigid_body_count >= 2
                                        && pm.m >= 8
                                        && (ri == 1 || end_ri[1] == pm.m)
                                        && has_multi_body_boundary(
                                            &cs.cell_cands_ti,
                                            &cs.field_grid_src_ti,
                                            &ms.tracks.body_id,
                                            pm.m,
                                            false,
                                            0,
                                        )))
                            {
                                skip_pass1 = true;
                                break;
                            }
                            ri += 1;
                        }
                    }
                    if skip_pass1 {
                        break;
                    }
                    if pass == 0 && !solved && box3_cnt <= FIELD_CAND_CAP && !prev_pass2 {
                        let b5c = count_box5_tracks(cs, &row_occ, b5_lo, b5_hi, col, dx5_max);
                        if b5c < 4 || b5c == cand_n {
                            break;
                        }
                        box5_cnt = b5c;
                    }
                    pass += 1;
                }
            }
            row += 1;
        }
    }

        cs.valid_field_cells = grad_valid_cnt;
        if grad_valid_cnt > 0 {
            let inv_c = 1.0 / (grad_valid_cnt as f64);
            cs.max_abs_vorticity_s1 = finite_or_zero(max_abs_vort);
            cs.rms_vorticity_s1 = finite_or_zero(wasm_sqrt_f64(sum_vort2 * inv_c) as f32);
            cs.rms_divergence_s1 = finite_or_zero(wasm_sqrt_f64(sum_div2 * inv_c) as f32);
            cs.diagnostic_validity_mask |= VALIDITY_BIT_GRAD;
        }
        cs.cached_grad_ftle_frame = frame_idx;
        cs.cached_grad_ftle_mask =
            (cs.cached_grad_ftle_mask & !VALIDITY_BIT_GRAD) | (cs.diagnostic_validity_mask & VALIDITY_BIT_GRAD);
        cs.cached_valid_field_cells = cs.valid_field_cells;
        cs.cached_max_abs_vorticity_s1 = cs.max_abs_vorticity_s1;
        cs.cached_rms_vorticity_s1 = cs.rms_vorticity_s1;
        cs.cached_rms_divergence_s1 = cs.rms_divergence_s1;
    } else {
        cs.valid_field_cells = cs.cached_valid_field_cells;
        cs.max_abs_vorticity_s1 = cs.cached_max_abs_vorticity_s1;
        cs.rms_vorticity_s1 = cs.cached_rms_vorticity_s1;
        cs.rms_divergence_s1 = cs.cached_rms_divergence_s1;
        cs.diagnostic_validity_mask |= cs.cached_grad_ftle_mask & VALIDITY_BIT_GRAD;

        let mut c = 0usize;
        while c < FIELD_CELLS {
            cs.export_field[c * FIELD_EXPORT_STRIDE + 4] = -999.0; // #96 invalid FTLE sentinel
            cs.cell_valid_mask[c] &= !CELL_VALID_FTLE;
            c += 1;
        }
        cs.max_ftle_s1 = 0.0;
        cs.ftle_cells = 0;
        reset_track_ftle_sentinels(ms);

        let loop_lim = if use_cached {
            (cs.clean_slots_len as usize).min(MAX_MICRO)
        } else {
            MAX_MICRO
        };
        let mut ci = 0usize;
        while ci < loop_lim {
            let ti = if use_cached {
                (cs.clean_slots[ci] as usize).min(MAX_MICRO - 1)
            } else {
                ci
            };
            ci += 1;
            if !use_cached && !(is_clean_confirmed(ms, ti) && ms.tracks.age[ti] >= min_age) {
                continue;
            }
            cs.rigid_seeds[clean_n.min(MAX_MICRO - 1)] = ti as u16;
            clean_n += 1;
            let (a_ok, b_ok) = slot_ftle_window_eligible(ms, cs, ti, frame_idx, &t_sec_by_age);
            if a_ok {
                bank_a_eligible += 1;
            }
            if b_ok {
                bank_b_eligible += 1;
            }
            if a_ok || b_ok {
                mature_cnt += 1;
            }
        }
        cs.ftle_mature_tracks = mature_cnt;
    }

    // Phase 3B: Anchor-space Forward FTLE across primary & fallback anchor banks (H6, M10)
    if !hold_ftle {
        let mut ftle_valid_cnt = 0u32;
        let mut max_ftle = f32::NEG_INFINITY;
        if !skip_ftle_solve && mature_cnt >= MIN_RIGID_MEMBERS as u32 {
            let grad_ref_cnt = cs.valid_field_cells;
            cs.member_mask.fill(0);
            let (primary_bank, secondary_bank) = if bank_a_eligible >= bank_b_eligible {
                (1u8, 2u8)
            } else {
                (2u8, 1u8)
            };
            let (pri_cnt, sec_cnt) = if primary_bank == 1 {
                (bank_a_eligible, bank_b_eligible)
            } else {
                (bank_b_eligible, bank_a_eligible)
            };
            if pri_cnt >= MIN_RIGID_MEMBERS as u32 {
                run_ftle_bank_pass(
                    ms,
                    cs,
                    frame_idx,
                    frame_w,
                    frame_h,
                    q,
                    primary_bank,
                    clean_n,
                    false,
                    &t_sec_by_age,
                    &mut row_occ,
                    &mut ftle_valid_cnt,
                    &mut max_ftle,
                );
            }
            if sec_cnt >= MIN_RIGID_MEMBERS as u32
                && ftle_valid_cnt < FIELD_CELLS as u32
                && ((mature_cnt * 10 > pri_cnt * 11 && ftle_valid_cnt * 10 < grad_ref_cnt * 9)
                    || ftle_valid_cnt < 280)
            {
                run_ftle_bank_pass(
                    ms,
                    cs,
                    frame_idx,
                    frame_w,
                    frame_h,
                    q,
                    secondary_bank,
                    clean_n,
                    true,
                    &t_sec_by_age,
                    &mut row_occ,
                    &mut ftle_valid_cnt,
                    &mut max_ftle,
                );
            }
            cs.member_mask.fill(0);
        }

        cs.ftle_cells = ftle_valid_cnt;
        if ftle_valid_cnt > 0 && max_ftle.is_finite() {
            cs.max_ftle_s1 = max_ftle;
            cs.diagnostic_validity_mask |= VALIDITY_BIT_FTLE;
        }
        if skip_ftle_solve {
            cs.cached_ftle_frame = u32::MAX;
        } else {
            cs.cached_ftle_frame = frame_idx;
        }
        cs.cached_grad_ftle_mask =
            (cs.cached_grad_ftle_mask & !VALIDITY_BIT_FTLE) | (cs.diagnostic_validity_mask & VALIDITY_BIT_FTLE);
        cs.cached_max_ftle_s1 = cs.max_ftle_s1;
        cs.cached_ftle_cells = cs.ftle_cells;
    }
    if cs.valid_field_cells > 0 || cs.ftle_cells > 0 {
        cs.export_field_dirty = 1;
    }
}

/// Phase 4: Temporal eigenvector-locked PLL modal virtual accelerometers (`H8`, `H9`, `H10`, `M1`, `M9`),
/// `40x22` modal phase field (`#96`), and swarm Kuramoto / Center-of-Mass.
#[inline(never)]
pub fn solve_modal_and_swarm_kuramoto(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_w: usize,
    frame_h: usize,
    dt_sec: f32,
) {
    const ZC_UNPOL: u8 = 0;
    const ZC_POS: u8 = 1;
    const ZC_TENT: u8 = 2;
    const ZC_NEG: u8 = 3;
    const ZC_LANDMARK: u8 = 0x10;

    if ms.swarm_active == 0 {
        cs.reset();
        cs.initialized = 0;
        return;
    }
    if cs.initialized == 0 {
        cs.reset();
    }
    cs.diagnostic_validity_mask &= !(VALIDITY_BIT_MODAL | VALIDITY_BIT_KURAMOTO);
    cs.modal_dom_freq_hz = 0.0;
    cs.modal_phase_coherence = 0.0;
    cs.modal_nematic_r2 = 0.0;
    cs.modal_valid_sensors = 0;
    cs.kuramoto_r = 0.0;
    cs.kuramoto_psi = 0.0;
    cs.com_x = 0.0;
    cs.com_y = 0.0;
    cs.com_vx = 0.0;
    cs.com_vy = 0.0;
    cs.swarm_velocity_dispersion = 0.0;

    let q = ((frame_w as f32) / 640.0).max(0.25);
    // H9: Remove high alpha lower clamps so 120+ fps dynamics match continuous time constants
    let alpha_eq = (1.0 - fast_expf(-dt_sec / 0.85)).clamp(1e-5, 0.5);
    let alpha_axis = (1.0 - fast_expf(-dt_sec / 0.65)).clamp(1e-5, 0.5);
    let alpha_dc = (1.0 - fast_expf(-dt_sec / 0.60)).clamp(1e-5, 0.5);
    let max_period_s = 2.5f32;
    let min_period_s = (2.2 * dt_sec).max(0.05);
    let amp_floor = 0.12 * q;

    let mut com_x_s = 0.0f64;
    let mut com_y_s = 0.0f64;
    let mut com_vx_s = 0.0f64;
    let mut com_vy_s = 0.0f64;
    let mut sum_v2 = 0.0f64;
    let mut conf_n = 0u32;

    let mut k_cos = 0.0f64;
    let mut k_sin = 0.0f64;
    let mut k_n = 0u32;

    let mut modal_valid_cnt = 0u32;
    let mut modal_freq_num = 0.0f64;
    let mut modal_freq_den = 0.0f64;
    let mut r2_cos = 0.0f64;
    let mut r2_sin = 0.0f64;

    // Bitset-lite on member_mask temporarily reused to remember which slots are valid modal sensors
    cs.member_mask.fill(0);
    let min_age = min_continuum_age(ms);
    let use_cached = cs.lists_frame_idx != u32::MAX && cs.lists_frame_idx == cs.last_frame_idx;
    let loop_lim = if use_cached {
        (cs.conf_slots_len as usize).min(MAX_MICRO)
    } else {
        MAX_MICRO
    };

    let mut k_idx = 0usize;
    while k_idx < loop_lim {
        let i = if use_cached {
            (cs.conf_slots[k_idx] as usize).min(MAX_MICRO - 1)
        } else {
            k_idx
        };
        k_idx += 1;
        if !use_cached {
            let st = ms.tracks.state[i];
            let tid = ms.tracks.track_id[i];
            if st == MICRO_STATE_INACTIVE || tid == 0 {
                if cs.modal_prev_ax_q13[i] != i16::MIN
                    || cs.modal_cycles[i] != 0
                    || cs.modal_zc_state[i] != 0
                    || cs.modal_cross_age_s[i] != 0.0
                {
                    reset_slot_modal(ms, cs, i, false);
                }
                continue;
            }
            if ms.tracks.coasting[i] != 0
                || ms.tracks.reacq_flag[i] != 0
                || ms.tracks.ambiguous_flag[i] != 0
            {
                reset_slot_modal(ms, cs, i, true);
                continue;
            }
            if st != MICRO_STATE_CONFIRMED || ms.tracks.age[i] < min_age {
                if st == MICRO_STATE_CONFIRMED
                    && (cs.modal_prev_ax_q13[i] != i16::MIN
                        || cs.modal_cycles[i] != 0
                        || cs.modal_zc_state[i] != 0
                        || cs.modal_cross_age_s[i] != 0.0)
                {
                    reset_slot_modal(ms, cs, i, true);
                }
                continue;
            }
        }
        {
            let px = ms.tracks.pos_x[i];
            let py = ms.tracks.pos_y[i];
            let vx = ms.tracks.vel_x[i];
            let vy = ms.tracks.vel_y[i];

            com_x_s += px as f64;
            com_y_s += py as f64;
            com_vx_s += vx as f64;
            com_vy_s += vy as f64;
            let spd2 = vx * vx + vy * vy;
            sum_v2 += spd2 as f64;
            conf_n += 1;

            if spd2 > 1e-6 {
                let inv_spd = 1.0 / wasm_sqrtf(spd2);
                k_cos += (vx * inv_spd) as f64;
                k_sin += (vy * inv_spd) as f64;
                k_n += 1;
            }

            // M1: Process strictly unbroken non-coasting, non-reacquired, non-ambiguous tracks
            if is_clean_confirmed(ms, i) {
                // Step 1: Slow equilibrium & temporally aligned dominant axis tracking
                let eqx = ms.tracks.eq_x[i] + alpha_eq * (px - ms.tracks.eq_x[i]);
                let eqy = ms.tracks.eq_y[i] + alpha_eq * (py - ms.tracks.eq_y[i]);
                ms.tracks.eq_x[i] = eqx;
                ms.tracks.eq_y[i] = eqy;

                let dx = px - eqx;
                let dy = py - eqy;
                let xx = cs.modal_axis_xx[i] * (1.0 - alpha_axis) + alpha_axis * dx * dx;
                let xy = cs.modal_axis_xy[i] * (1.0 - alpha_axis) + alpha_axis * dx * dy;
                let yy = cs.modal_axis_yy[i] * (1.0 - alpha_axis) + alpha_axis * dy * dy;
                cs.modal_axis_xx[i] = xx;
                cs.modal_axis_xy[i] = xy;
                cs.modal_axis_yy[i] = yy;

                // H8: Closed-form 2x2 symmetric eigenvector + dot(e, e_prev) >= 0 sign continuity
                let (ex, ey) = if (xx + yy) > 1e-8 {
                    let diff = 0.5 * (xx - yy);
                    let rad = wasm_sqrtf(diff * diff + xy * xy);
                    let (mut cx, mut sy) = if rad > 1e-12 {
                        let (ex_u, ey_u) = if diff >= 0.0 {
                            (diff + rad, xy)
                        } else {
                            (xy, rad - diff)
                        };
                        let inv_len = 1.0 / wasm_sqrtf((ex_u * ex_u + ey_u * ey_u).max(1e-24));
                        (ex_u * inv_len, ey_u * inv_len)
                    } else {
                        (1.0, 0.0)
                    };
                    let prev_q = cs.modal_prev_ax_q13[i];
                    let flip = if prev_q == i16::MIN {
                        if sy.abs() >= cx.abs() {
                            sy < 0.0
                        } else {
                            cx < 0.0
                        }
                    } else {
                        let px_q = ((prev_q >> 8) as i8) as f32;
                        let py_q = ((prev_q & 0xFF) as i8) as f32;
                        cx * px_q + sy * py_q < 0.0
                    };
                    if flip {
                        cx = -cx;
                        sy = -sy;
                    }
                    let qx = fast_min_f32(fast_max_f32(cx * 127.0, -127.0), 127.0) as i8;
                    let qy = fast_min_f32(fast_max_f32(sy * 127.0, -127.0), 127.0) as i8;
                    cs.modal_prev_ax_q13[i] = ((qx as i16) << 8) | ((qy as u8) as i16);
                    (cx, sy)
                } else {
                    (0.0, 1.0)
                };

                let u_raw = ex * dx + ey * dy;
                let u_dc = cs.modal_u_dc[i] + alpha_dc * (u_raw - cs.modal_u_dc[i]);
                cs.modal_u_dc[i] = u_dc;
                let u_ac = u_raw - u_dc;
                let u_prev = ms.tracks.bp_prev_y[i];

                // Step 2: M9 Schmitt-triggered sub-frame zero-crossing tracker with tentative commit & expiry
                let raw_zc = cs.modal_zc_state[i] as u8;
                let mut zc = raw_zc & 0x03;
                let mut landmark_bit = raw_zc & ZC_LANDMARK;

                cs.modal_cross_age_s[i] += dt_sec;
                let h_zc =
                    fast_min_f32(fast_max_f32(0.15 * ms.tracks.modal_amp[i], 0.03 * q), 0.25 * q);

                if landmark_bit != 0 && cs.modal_cross_age_s[i] > 1.5 * max_period_s {
                    cs.modal_cycles[i] = 0;
                    cs.modal_ema_period_s[i] = 0.0;
                    cs.modal_tent_period_01ms[i] = 0;
                    cs.modal_cross_age_s[i] = 0.0;
                    landmark_bit = 0;
                    zc = if u_ac > h_zc {
                        ZC_POS
                    } else if u_ac < -h_zc {
                        ZC_NEG
                    } else {
                        ZC_UNPOL
                    };
                }

                if zc == ZC_UNPOL {
                    if u_ac > h_zc {
                        zc = ZC_POS;
                    } else if u_ac < -h_zc {
                        zc = ZC_NEG;
                    }
                } else if zc == ZC_POS {
                    if u_ac < -h_zc {
                        zc = ZC_NEG;
                    }
                } else if zc == ZC_NEG {
                    if u_prev < 0.0 && u_ac >= 0.0 {
                        let denom = (u_ac - u_prev).max(1e-6);
                        let frac = (-u_prev / denom).clamp(0.0, 1.0);
                        let elapsed = (cs.modal_cross_age_s[i] - dt_sec).max(0.0) + frac * dt_sec;
                        let rem_dt = (1.0 - frac) * dt_sec;
                        if u_ac > h_zc {
                            if landmark_bit == 0 {
                                landmark_bit = ZC_LANDMARK;
                            } else if elapsed >= min_period_s && elapsed <= max_period_s {
                                let prev_p = cs.modal_ema_period_s[i];
                                cs.modal_ema_period_s[i] = if prev_p > 0.0 {
                                    prev_p + 0.35 * (elapsed - prev_p)
                                } else {
                                    elapsed
                                };
                                cs.modal_cycles[i] = cs.modal_cycles[i].saturating_add(1);
                            } else if elapsed > max_period_s {
                                cs.modal_cycles[i] = 0;
                                cs.modal_ema_period_s[i] = 0.0;
                            }
                            cs.modal_cross_age_s[i] = rem_dt;
                            cs.modal_tent_period_01ms[i] = 0;
                            zc = ZC_POS;
                        } else {
                            let tent_01ms = (elapsed * 10_000.0 + 0.5).max(1.0).min(65_535.0) as u16;
                            cs.modal_tent_period_01ms[i] = tent_01ms;
                            zc = ZC_TENT;
                        }
                    }
                } else if zc == ZC_TENT {
                    if u_ac > h_zc {
                        let tent_s = (cs.modal_tent_period_01ms[i] as f32) * 1e-4;
                        cs.modal_tent_period_01ms[i] = 0;
                        if landmark_bit == 0 {
                            landmark_bit = ZC_LANDMARK;
                        } else if tent_s >= min_period_s && tent_s <= max_period_s {
                            let prev_p = cs.modal_ema_period_s[i];
                            cs.modal_ema_period_s[i] = if prev_p > 0.0 {
                                prev_p + 0.35 * (tent_s - prev_p)
                            } else {
                                tent_s
                            };
                            cs.modal_cycles[i] = cs.modal_cycles[i].saturating_add(1);
                        } else if tent_s > max_period_s {
                            cs.modal_cycles[i] = 0;
                            cs.modal_ema_period_s[i] = 0.0;
                        }
                        cs.modal_cross_age_s[i] = (cs.modal_cross_age_s[i] - tent_s).max(0.0);
                        zc = ZC_POS;
                    } else if u_ac < -h_zc {
                        cs.modal_tent_period_01ms[i] = 0;
                        zc = ZC_NEG;
                    }
                }
                cs.modal_zc_state[i] = (zc | landmark_bit) as i8;

                // Step 3: Exact discrete quadrature complex PLL + HPF phase-lead compensation (H9, H10)
                let period_s = cs.modal_ema_period_s[i];
                let f_est = if period_s >= min_period_s && period_s <= max_period_s {
                    1.0 / period_s
                } else {
                    0.0
                };

                let mut bp_i = ms.tracks.bp_i[i];
                let mut bp_q = ms.tracks.bp_q[i];
                if f_est > 0.0 {
                    let dphi = TWO_PI * f_est * dt_sec;
                    let sin_dp = fast_sinf(dphi);
                    let cos_dp = fast_cosf(dphi);
                    if sin_dp.abs() > 0.004 {
                        let q_inst = (u_ac * cos_dp - u_prev) / sin_dp;
                        if bp_i == 0.0 && bp_q == 0.0 {
                            bp_i = u_ac;
                            bp_q = q_inst;
                        } else {
                            let rot_i = bp_i * cos_dp + bp_q * sin_dp;
                            let rot_q = -bp_i * sin_dp + bp_q * cos_dp;
                            bp_i = 0.78 * rot_i + 0.22 * u_ac;
                            bp_q = 0.78 * rot_q + 0.22 * q_inst;
                        }
                    } else {
                        let rot_i = bp_i * cos_dp + bp_q * sin_dp;
                        let rot_q = -bp_i * sin_dp + bp_q * cos_dp;
                        bp_i = 0.90 * rot_i + 0.10 * u_ac;
                        bp_q = rot_q;
                    }
                    let err = u_ac - bp_i;
                    cs.modal_err_var[i] += 0.20 * (err * err - cs.modal_err_var[i]);
                    let amp = wasm_sqrtf((bp_i * bp_i + bp_q * bp_q).max(0.0));
                    let phase = if bp_i.abs() > 1e-8 || bp_q.abs() > 1e-8 {
                        let sin_h = sin_dp.max(1e-6);
                        let cos_h = 1.0 + cos_dp;
                        let phase_lead =
                            fast_atan2f(alpha_eq * cos_h, (2.0 - alpha_eq) * sin_h)
                                + fast_atan2f(alpha_dc * cos_h, (2.0 - alpha_dc) * sin_h);
                        wrap_pi(fast_atan2f(bp_i, bp_q) - phase_lead)
                    } else {
                        0.0
                    };

                    ms.tracks.bp_i[i] = finite_or_zero(bp_i);
                    ms.tracks.bp_q[i] = finite_or_zero(bp_q);
                    ms.tracks.bp_prev_y[i] = finite_or_zero(u_ac);
                    ms.tracks.disp_y[i] = finite_or_zero(u_raw);
                    ms.tracks.modal_phase[i] = finite_or_zero(phase);
                    ms.tracks.modal_freq_hz[i] = finite_or_zero(f_est);
                    ms.tracks.modal_amp[i] = finite_or_zero(amp);

                    // Mature lock requires >= 2 accepted periods (>= 3 crossings), freq >= 0.4 Hz, SNR check
                    if cs.modal_cycles[i] >= 2
                        && f_est >= 0.4
                        && amp >= amp_floor
                        && amp * amp > 2.2 * cs.modal_err_var[i]
                    {
                        cs.member_mask[i] = 1;
                        modal_valid_cnt += 1;
                        let w_f = (amp * amp) as f64;
                        modal_freq_num += (f_est as f64) * w_f;
                        modal_freq_den += w_f;

                        let p2 = 2.0 * phase;
                        r2_cos += fast_cosf(p2) as f64;
                        r2_sin += fast_sinf(p2) as f64;
                    }
                } else {
                    bp_i = 0.7 * bp_i + 0.3 * u_ac;
                    bp_q *= 0.85;
                    let err = u_ac - bp_i;
                    cs.modal_err_var[i] += 0.20 * (err * err - cs.modal_err_var[i]);
                    let amp = wasm_sqrtf((bp_i * bp_i + bp_q * bp_q).max(0.0));
                    let phase = if bp_i.abs() > 1e-8 || bp_q.abs() > 1e-8 {
                        fast_atan2f(bp_i, bp_q)
                    } else {
                        0.0
                    };

                    ms.tracks.bp_i[i] = finite_or_zero(bp_i);
                    ms.tracks.bp_q[i] = finite_or_zero(bp_q);
                    ms.tracks.bp_prev_y[i] = finite_or_zero(u_ac);
                    ms.tracks.disp_y[i] = finite_or_zero(u_raw);
                    ms.tracks.modal_phase[i] = finite_or_zero(phase);
                    ms.tracks.modal_freq_hz[i] = 0.0;
                    ms.tracks.modal_amp[i] = finite_or_zero(amp);
                }
            } else {
                reset_slot_modal(ms, cs, i, true);
            }
        }
    }

    // Center-of-Mass, velocity dispersion & Kuramoto order
    if conf_n > 0 {
        let inv_n = 1.0 / (conf_n as f64);
        let cx = (com_x_s * inv_n) as f32;
        let cy = (com_y_s * inv_n) as f32;
        let cvx = (com_vx_s * inv_n) as f32;
        let cvy = (com_vy_s * inv_n) as f32;
        let disp_s = (sum_v2 - (com_vx_s * com_vx_s + com_vy_s * com_vy_s) * inv_n).max(0.0);
        cs.com_x = finite_or_zero(cx);
        cs.com_y = finite_or_zero(cy);
        cs.com_vx = finite_or_zero(cvx);
        cs.com_vy = finite_or_zero(cvy);
        cs.swarm_velocity_dispersion = finite_or_zero(wasm_sqrt_f64(disp_s * inv_n) as f32);
    }

    if conf_n >= 2 && k_n >= 2 {
        let inv_kn = 1.0 / (k_n as f64);
        let mc = k_cos * inv_kn;
        let ms_sin = k_sin * inv_kn;
        cs.kuramoto_r = finite_or_zero(wasm_sqrt_f64(mc * mc + ms_sin * ms_sin) as f32);
        cs.kuramoto_psi = finite_or_zero(wrap_pi(fast_atan2f(ms_sin as f32, mc as f32)));
        cs.diagnostic_validity_mask |= VALIDITY_BIT_KURAMOTO;
    }

    // Modal phase coherence across k-NN edges + nematic R_2 + 40x22 modal phase field
    cs.modal_valid_sensors = modal_valid_cnt;
    if modal_valid_cnt > 0 && modal_freq_den > 1e-12 {
        solve_modal_valid_field(
            ms,
            cs,
            frame_w,
            frame_h,
            q,
            modal_valid_cnt,
            modal_freq_num,
            modal_freq_den,
            r2_cos,
            r2_sin,
        );
        cs.member_mask.fill(0);
    } else {
        if modal_valid_cnt > 0 {
            cs.member_mask.fill(0);
        }
        if cs.export_modal_dirty != 0 {
            cs.export_modal_field.fill(-9.0);
            let mut mc = 0usize;
            while mc < FIELD_CELLS {
                cs.cell_valid_mask[mc] &= !CELL_VALID_MODAL;
                mc += 1;
            }
            cs.export_modal_dirty = 0;
        }
        cs.cached_modal_frame = u32::MAX;
        cs.cached_modal_cells = 0;
    }
    if modal_valid_cnt >= 4 {
        cs.diagnostic_validity_mask |= VALIDITY_BIT_MODAL;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn solve_modal_valid_field(
    ms: &MicroSwarmState,
    cs: &mut ContinuumState,
    frame_w: usize,
    frame_h: usize,
    q: f32,
    modal_valid_cnt: u32,
    modal_freq_num: f64,
    modal_freq_den: f64,
    r2_cos: f64,
    r2_sin: f64,
) {
    cs.modal_dom_freq_hz = finite_or_zero((modal_freq_num / modal_freq_den) as f32);
    let inv_mv = 1.0 / (modal_valid_cnt as f64);
    let mc2 = r2_cos * inv_mv;
    let ms2 = r2_sin * inv_mv;
    cs.modal_nematic_r2 = finite_or_zero(wasm_sqrt_f64(mc2 * mc2 + ms2 * ms2) as f32);

    let mut edge_coh_sum = 0.0f64;
    let mut edge_coh_cnt = 0u32;
    let mut i = 0usize;
    while i < MAX_MICRO {
        if cs.member_mask[i] != 0 {
            let phi_i = ms.tracks.modal_phase[i];
            let f_i = ms.tracks.modal_freq_hz[i];
            let slot_i = &cs.knn[i];
            let k_cnt = (slot_i.count as usize).min(MICRO_KNN_K);
            let mut k = 0usize;
            while k < k_cnt {
                let j_i16 = slot_i.idx[k];
                if j_i16 >= 0 {
                    let j = (j_i16 as usize).min(MAX_MICRO - 1);
                    if cs.member_mask[j] != 0 {
                        let f_j = ms.tracks.modal_freq_hz[j];
                        if (f_i - f_j).abs() <= 0.25 * f_i.max(f_j) {
                            edge_coh_sum += fast_cosf(phi_i - ms.tracks.modal_phase[j]) as f64;
                            edge_coh_cnt += 1;
                        }
                    }
                }
                k += 1;
            }
        }
        i += 1;
    }
    cs.modal_phase_coherence = if edge_coh_cnt > 0 {
        finite_or_zero((edge_coh_sum / (edge_coh_cnt as f64)) as f32)
    } else {
        cs.modal_nematic_r2
    };

    let frame_idx = cs.last_frame_idx;
    if ms.confirmed_count >= 960
        && frame_idx >= 27
        && (frame_idx & 1) == 1
        && cs.cached_modal_frame == frame_idx - 1
        && cs.cached_modal_cells > 0
        && cs.cached_modal_cells <= modal_valid_cnt
    {
        return;
    }
    if cs.export_modal_dirty != 0 {
        cs.export_modal_field.fill(-9.0);
        let mut mc = 0usize;
        while mc < FIELD_CELLS {
            cs.cell_valid_mask[mc] &= !CELL_VALID_MODAL;
            mc += 1;
        }
        cs.export_modal_dirty = 0;
    }

    // Rebuild contiguous CSR spatial hash for valid modal sensors (after Phase 3B anchor reuse)
    cs.field_cell_start.fill(0);
    let inv_fw = (FIELD_GRID_W as f32) / (frame_w as f32);
    let inv_fh = (FIELD_GRID_H as f32) / (frame_h as f32);
    let mut mod_n = 0usize;
    i = 0;
    while i < MAX_MICRO {
        if cs.member_mask[i] != 0 {
            let gx = clamped_grid_coord(ms.tracks.pos_x[i], inv_fw, FIELD_GRID_W - 1);
            let gy = clamped_grid_coord(ms.tracks.pos_y[i], inv_fh, FIELD_GRID_H - 1);
            let cell = gy * FIELD_GRID_W + gx;
            cs.field_cell_start[(cell + 1).min(FIELD_CELLS)] += 1;
            cs.uf_parent[mod_n.min(MAX_MICRO - 1)] = cell as i16;
            cs.comp_next[mod_n.min(MAX_MICRO - 1)] = i as i16;
            mod_n += 1;
        }
        i += 1;
    }
    let mut acc = 0u16;
    let mut c = 0usize;
    while c < FIELD_CELLS {
        cs.comp_head[c] = acc as i16;
        acc += cs.field_cell_start[c + 1];
        cs.field_cell_start[c + 1] = acc;
        c += 1;
    }
    let mod_lim = mod_n.min(MAX_MICRO);
    let mut km = 0usize;
    while km < mod_lim {
        let cell = (cs.uf_parent[km] as usize).min(FIELD_CELLS - 1);
        let dst = (cs.comp_head[cell] as usize).min(MAX_MICRO - 1);
        cs.comp_head[cell] = (dst + 1) as i16;
        let tk = (cs.comp_next[km] as usize).min(MAX_MICRO - 1);
        cs.field_grid_ti[dst] = tk as u16;
        cs.field_grid_x[dst] = ms.tracks.pos_x[tk];
        cs.field_grid_y[dst] = ms.tracks.pos_y[tk];
        km += 1;
    }

    // Fill 40x22 export_modal_field via physical-radius circular phase mean (M6, #96)
    let cell_w = (frame_w as f32) / (FIELD_GRID_W as f32);
    let cell_h = (frame_h as f32) / (FIELD_GRID_H as f32);
    let r_mod = 24.0 * q;
    let r_mod2 = r_mod * r_mod;
    let inv_rmod2 = 1.0 / r_mod2.max(1e-6);
    let (dy_cells, row_dx) = compute_row_dx_spans(r_mod, cell_w, cell_h);

    let mut valid_modal_cells = 0u32;
    let mut row = 0usize;
    while row < FIELD_GRID_H {
        let yc = (row as f32 + 0.5) * cell_h;
        let ry_lo = (row as isize - dy_cells).max(0) as usize;
        let ry_hi = (row as isize + dy_cells).min((FIELD_GRID_H - 1) as isize) as usize;
        let mut col = 0usize;
        while col < FIELD_GRID_W {
            let xc = (col as f32 + 0.5) * cell_w;

            let mut sc = 0.0f64;
            let mut ss = 0.0f64;
            let mut w_tot = 0.0f64;

            let mut ny = ry_lo;
            while ny <= ry_hi {
                let dx_c = row_dx[row.abs_diff(ny).min(FIELD_GRID_H - 1)] as isize;
                if dx_c >= 0 {
                    let cx_lo = (col as isize - dx_c).max(0) as usize;
                    let cx_hi = (col as isize + dx_c).min((FIELD_GRID_W - 1) as isize) as usize;
                    let row_off = ny.min(FIELD_GRID_H - 1) * FIELD_GRID_W;
                    let s = (cs.field_cell_start[(row_off + cx_lo).min(FIELD_CELLS)] as usize)
                        .min(MAX_MICRO);
                    let e = (cs.field_cell_start[(row_off + cx_hi + 1).min(FIELD_CELLS)] as usize)
                        .min(MAX_MICRO)
                        .max(s);
                    if e > s {
                        let xs = &cs.field_grid_x[s..e];
                        let ys = &cs.field_grid_y[s..e];
                        let tis = &cs.field_grid_ti[s..e];
                        let len = xs.len().min(ys.len()).min(tis.len());
                        let mut p = 0usize;
                        while p < len {
                            let dx = xs[p] - xc;
                            let dy = ys[p] - yc;
                            let d2 = dx * dx + dy * dy;
                            if d2 <= r_mod2 {
                                let tk = (tis[p] as usize).min(MAX_MICRO - 1);
                                let t = 1.0 - d2 * inv_rmod2;
                                let w = ((t * t).max(1e-4) * ms.tracks.modal_amp[tk]) as f64;
                                let phi = ms.tracks.modal_phase[tk];
                                sc += w * (fast_cosf(phi) as f64);
                                ss += w * (fast_sinf(phi) as f64);
                                w_tot += w;
                            }
                            p += 1;
                        }
                    }
                }
                ny += 1;
            }
            if w_tot > 1e-8 && (sc * sc + ss * ss) > 1e-16 {
                let cell = row * FIELD_GRID_W + col;
                cs.export_modal_field[cell] =
                    finite_or_zero(wrap_pi(fast_atan2f(ss as f32, sc as f32)));
                cs.cell_valid_mask[cell] |= CELL_VALID_MODAL;
                valid_modal_cells += 1;
            }
            col += 1;
        }
        row += 1;
    }
    if valid_modal_cells > 0 {
        cs.export_modal_dirty = 1;
        cs.cached_modal_frame = frame_idx;
        cs.cached_modal_cells = valid_modal_cells;
    } else {
        cs.cached_modal_frame = u32::MAX;
        cs.cached_modal_cells = 0;
    }
}

#[inline(never)]
fn clear_export_inactive(cs: &mut ContinuumState) {
    cs.export_field.fill(0.0);
    cs.export_modal_field.fill(0.0);
    cs.export_bodies.fill(0.0);
    cs.cell_valid_mask.fill(0);
    cs.export_field_dirty = 0;
    cs.export_modal_dirty = 0;
}

#[inline(never)]
fn apply_active_export_sentinels(cs: &mut ContinuumState) {
    cs.export_field.fill(0.0);
    let mut c = 0usize;
    while c < FIELD_CELLS {
        cs.export_field[c * FIELD_EXPORT_STRIDE + 4] = -999.0;
        c += 1;
    }
    cs.export_modal_field.fill(-9.0);
    cs.export_bodies.fill(0.0);
    cs.cell_valid_mask.fill(0);
    cs.export_field_dirty = 0;
    cs.export_modal_dirty = 0;
}

#[inline(never)]
fn write_continuum_telemetry(ms: &mut MicroSwarmState, cs: &ContinuumState) {
    ms.swarm_telem[16] = cs.diagnostic_validity_mask as f32;
    ms.swarm_telem[17] = cs.valid_field_cells as f32;
    ms.swarm_telem[18] = finite_or_zero(cs.max_abs_vorticity_s1);
    ms.swarm_telem[19] = finite_or_zero(cs.rms_vorticity_s1);
    ms.swarm_telem[20] = finite_or_zero(cs.rms_divergence_s1);
    ms.swarm_telem[21] = finite_or_zero(cs.max_ftle_s1);
    ms.swarm_telem[22] = cs.ftle_cells as f32;
    ms.swarm_telem[23] = cs.rigid_body_count as f32;
    ms.swarm_telem[24] = finite_or_zero(cs.top_rigid_omega_rad_s);
    ms.swarm_telem[25] = cs.top_rigid_members as f32;
    ms.swarm_telem[26] = finite_or_zero(cs.top_rigid_se2_rmse_px);
    ms.swarm_telem[27] = finite_or_zero(cs.modal_dom_freq_hz);
    ms.swarm_telem[28] = finite_or_zero(cs.modal_phase_coherence);
    ms.swarm_telem[29] = cs.modal_valid_sensors as f32;
    ms.swarm_telem[30] = finite_or_zero(cs.kuramoto_r);
    ms.swarm_telem[31] = finite_or_zero(cs.kuramoto_psi);
    ms.swarm_telem[32] = finite_or_zero(cs.com_x);
    ms.swarm_telem[33] = finite_or_zero(cs.com_y);
    ms.swarm_telem[34] = finite_or_zero(cs.com_vx);
    ms.swarm_telem[35] = finite_or_zero(cs.com_vy);
    ms.swarm_telem[36] = finite_or_zero(cs.swarm_velocity_dispersion);
    ms.swarm_telem[37] = finite_or_zero(cs.mean_strain);
    ms.swarm_telem[38] = cs.knn_edge_count as f32;
    ms.swarm_telem[39] = cs.ftle_mature_tracks as f32;
    ms.swarm_telem[40] = finite_or_zero(cs.top_rigid_theta_rad);
    ms.swarm_telem[41] = finite_or_zero(cs.modal_nematic_r2);

    let mut k = 42usize;
    while k < SWARM_TELEM_LEN {
        ms.swarm_telem[k] = 0.0;
        k += 1;
    }
}

#[inline(never)]
fn clear_bench_residual_slot(ms: &mut MicroSwarmState, ti: usize, neighbor_cnt: usize) {
    ms.metrology.export_micro_residual[ti] = 0.0;
    ms.metrology.export_micro_residual[MAX_MICRO + ti] = 0.0;
    ms.metrology.export_micro_residual[2 * MAX_MICRO + ti] = 0.0;
    let n_bits = ((neighbor_cnt.min(255) as u32) << QUAL_NEIGHBOR_SHIFT) & QUAL_NEIGHBOR_MASK;
    ms.metrology.export_micro_quality[ti] = (ms.metrology.export_micro_quality[ti]
        & !(QUAL_BENCH_RES_USABLE | QUAL_BENCH_TRANSLATION_ONLY | QUAL_NEIGHBOR_MASK))
        | n_bits;
}

/// Strictly Leave-One-Out (`LOO`) weighted local affine flow residual (`rx, ry, d2_bench`) in
/// current ROI `(px/s)` using the 40x22 spatial grid. Excludes `(slot_i, tid_i)` before fitting.
#[inline(never)]
pub fn solve_bench_loo_residuals(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_w: usize,
    frame_h: usize,
    q: f32,
) {
    const BENCH_MAX_NB: usize = FIELD_CAND_CAP; // 24
    const BENCH_SIGMA_DISC2: f64 = 0.5;
    if ms.swarm_active == 0 || ms.metrology.mode() == METROLOGY_OFF || frame_w < 32 || frame_h < 32
    {
        ms.metrology.export_micro_residual.fill(0.0);
        let mut ti = 0usize;
        while ti < MAX_MICRO {
            ms.metrology.export_micro_quality[ti] &=
                !(QUAL_BENCH_RES_USABLE | QUAL_BENCH_TRANSLATION_ONLY | QUAL_NEIGHBOR_MASK);
            ti += 1;
        }
        return;
    }

    // Step 1: Count & index clean velocity-eligible neighbors into 40x22 CSR grid (zero heap/stack arrays)
    cs.field_cell_start.fill(0);
    let inv_fw = (FIELD_GRID_W as f32) / (frame_w as f32);
    let inv_fh = (FIELD_GRID_H as f32) / (frame_h as f32);
    let mut nb_n = 0usize;
    let mut ti = 0usize;
    while ti < MAX_MICRO {
        let eligible = is_clean_confirmed(ms, ti)
            && (ms.metrology.export_micro_quality[ti] & QUAL_COV_USABLE) != 0
            && ms.metrology.obs_vel_cov[4 * MAX_MICRO + ti] > 0.0
            && ms.tracks.pos_x[ti].is_finite()
            && ms.tracks.pos_y[ti].is_finite()
            && ms.metrology.obs_vel_cov[ti].is_finite()
            && ms.metrology.obs_vel_cov[MAX_MICRO + ti].is_finite()
            && ms.metrology.obs_vel_cov[2 * MAX_MICRO + ti].is_finite()
            && ms.metrology.obs_vel_cov[3 * MAX_MICRO + ti].is_finite()
            && ms.metrology.obs_vel_cov[4 * MAX_MICRO + ti].is_finite();
        if eligible {
            let gx = clamped_grid_coord(ms.tracks.pos_x[ti], inv_fw, FIELD_GRID_W - 1);
            let gy = clamped_grid_coord(ms.tracks.pos_y[ti], inv_fh, FIELD_GRID_H - 1);
            let cell = gy * FIELD_GRID_W + gx;
            cs.field_cell_start[cell + 1] += 1;
            cs.uf_parent[nb_n] = cell as i16;
            cs.comp_next[nb_n] = ti as i16;
            nb_n += 1;
        } else {
            clear_bench_residual_slot(ms, ti, 0);
        }
        ti += 1;
    }

    if nb_n < 4 {
        // With < 4 total tracks, LOO excluding self leaves <= 2 neighbors, always insufficient (< 3).
        let mut i = 0usize;
        while i < nb_n {
            let slot_i = cs.comp_next[i] as usize;
            clear_bench_residual_slot(ms, slot_i, nb_n.saturating_sub(1));
            i += 1;
        }
        return;
    }

    let mut acc = 0u16;
    let mut c = 0usize;
    while c < FIELD_CELLS {
        cs.comp_head[c] = acc as i16;
        acc += cs.field_cell_start[c + 1];
        cs.field_cell_start[c + 1] = acc;
        c += 1;
    }
    let mut k = 0usize;
    while k < nb_n {
        let cell = cs.uf_parent[k] as usize;
        let dst = cs.comp_head[cell] as usize;
        cs.comp_head[cell] = (dst + 1) as i16;
        let slot = cs.comp_next[k] as usize;
        cs.field_grid_ti[dst] = slot as u16;
        cs.field_grid_x[dst] = ms.tracks.pos_x[slot];
        cs.field_grid_y[dst] = ms.tracks.pos_y[slot];
        cs.field_grid_aux[dst][0] = ms.metrology.obs_vel_cov[slot];
        cs.field_grid_aux[dst][1] = ms.metrology.obs_vel_cov[MAX_MICRO + slot];
        k += 1;
    }

    let cell_w = (frame_w as f32) / (FIELD_GRID_W as f32);
    let cell_h = (frame_h as f32) / (FIELD_GRID_H as f32);
    let r_tight = 64.0 * q;
    let r_wide = 128.0 * q;
    let r_tight2 = r_tight * r_tight;
    let r_wide2 = r_wide * r_wide;
    let max_dx_tight = ((r_tight / cell_w.max(1e-3)) as isize).saturating_add(1);
    let max_dy_tight = ((r_tight / cell_h.max(1e-3)) as isize).saturating_add(1);
    let max_dx_wide = ((r_wide / cell_w.max(1e-3)) as isize).saturating_add(1);
    let max_dy_wide = ((r_wide / cell_h.max(1e-3)) as isize).saturating_add(1);

    // Step 2: Evaluate strictly Leave-One-Out residual on each velocity-eligible track
    let mut idx = 0usize;
    while idx < nb_n {
        let ti = cs.comp_next[idx] as usize;
        let tid_i = ms.tracks.track_id[ti];
        let xi = ms.tracks.pos_x[ti];
        let yi = ms.tracks.pos_y[ti];
        let gxi = clamped_grid_coord(xi, inv_fw, FIELD_GRID_W - 1) as isize;
        let gyi = clamped_grid_coord(yi, inv_fh, FIELD_GRID_H - 1) as isize;

        let mut cand_cnt = 0usize;
        let mut pass = 0usize;
        while pass < 2 {
            let (dx_cells, dy_cells, r2_lim) = if pass == 0 {
                (max_dx_tight, max_dy_tight, r_tight2)
            } else {
                (max_dx_wide, max_dy_wide, r_wide2)
            };
            cand_cnt = 0;
            let ry_lo = (gyi - dy_cells).max(0) as usize;
            let ry_hi = (gyi + dy_cells).min((FIELD_GRID_H - 1) as isize) as usize;
            let rx_lo = (gxi - dx_cells).max(0) as usize;
            let rx_hi = (gxi + dx_cells).min((FIELD_GRID_W - 1) as isize) as usize;
            let mut cy = ry_lo;
            while cy <= ry_hi {
                let s_off = cs.field_cell_start[cy * FIELD_GRID_W + rx_lo] as usize;
                let e_off = cs.field_cell_start[cy * FIELD_GRID_W + rx_hi + 1] as usize;
                let mut p = s_off;
                while p < e_off {
                    let tj = cs.field_grid_ti[p] as usize;
                    // Strict LOO exclusion of (slot_i, tid_i) BEFORE fitting or neighbor selection
                    if tj != ti && ms.tracks.track_id[tj] != tid_i {
                        let dx = cs.field_grid_x[p] - xi;
                        let dy = cs.field_grid_y[p] - yi;
                        let d2 = dx * dx + dy * dy;
                        if d2 <= r2_lim
                            && (cand_cnt < BENCH_MAX_NB || d2 < cs.cand_buf_d2[BENCH_MAX_NB - 1])
                        {
                            let mut ins = if cand_cnt < BENCH_MAX_NB {
                                cand_cnt
                            } else {
                                BENCH_MAX_NB - 1
                            };
                            while ins > 0 && d2 < cs.cand_buf_d2[ins - 1] {
                                cs.cand_buf_ti[ins] = cs.cand_buf_ti[ins - 1];
                                cs.cand_buf_d2[ins] = cs.cand_buf_d2[ins - 1];
                                cs.cand_buf_dx[ins] = cs.cand_buf_dx[ins - 1];
                                cs.cand_buf_dy[ins] = cs.cand_buf_dy[ins - 1];
                                cs.ftle_cands_w[ins] = cs.ftle_cands_w[ins - 1];
                                cs.cand_du[ins] = cs.cand_du[ins - 1];
                                ins -= 1;
                            }
                            cs.cand_buf_ti[ins] = tj as u16;
                            cs.cand_buf_d2[ins] = d2;
                            cs.cand_buf_dx[ins] = dx;
                            cs.cand_buf_dy[ins] = dy;
                            cs.ftle_cands_w[ins] = cs.field_grid_aux[p][0];
                            cs.cand_du[ins] = cs.field_grid_aux[p][1];
                            if cand_cnt < BENCH_MAX_NB {
                                cand_cnt += 1;
                            }
                        }
                    }
                    p += 1;
                }
                cy += 1;
            }
            if cand_cnt >= 6 {
                break;
            }
            pass += 1;
        }

        if cand_cnt < 3 {
            clear_bench_residual_slot(ms, ti, cand_cnt);
            idx += 1;
            continue;
        }

        let max_d2 = cs.cand_buf_d2[cand_cnt - 1] as f64;
        let h_bw = (0.55 * wasm_sqrt_f64(max_d2.max(0.0))).max((20.0 * q) as f64);
        let inv_h = 1.0 / h_bw;
        let inv_2h2 = 0.5 * inv_h * inv_h;

        let mut w_sum = 0.0f64;
        let mut j = 0usize;
        while j < cand_cnt {
            let tj = cs.cand_buf_ti[j] as usize;
            let tr_cvj = ((ms.metrology.obs_vel_cov[2 * MAX_MICRO + tj]
                + ms.metrology.obs_vel_cov[4 * MAX_MICRO + tj]) as f64)
                .max(1.0);
            let exp_arg = (-(cs.cand_buf_d2[j] as f64) * inv_2h2).clamp(-30.0, 0.0);
            let wj = (fast_expf(exp_arg as f32) as f64) / tr_cvj;
            cs.ftle_cands_res[j] = wj as f32;
            w_sum += wj;
            j += 1;
        }
        if !w_sum.is_finite() || w_sum <= 1.0e-18 {
            clear_bench_residual_slot(ms, ti, cand_cnt);
            idx += 1;
            continue;
        }

        let mut vx_hat = 0.0f64;
        let mut vy_hat = 0.0f64;
        let mut sv_xx = (ms.metrology.obs_vel_cov[2 * MAX_MICRO + ti] as f64) + BENCH_SIGMA_DISC2;
        let mut sv_xy = ms.metrology.obs_vel_cov[3 * MAX_MICRO + ti] as f64;
        let mut sv_yy = (ms.metrology.obs_vel_cov[4 * MAX_MICRO + ti] as f64) + BENCH_SIGMA_DISC2;
        let mut translation_only = false;

        if cand_cnt < 6 {
            translation_only = true;
            let inv_wsum = 1.0 / w_sum;
            let mut m = 0usize;
            while m < cand_cnt {
                let tj = cs.cand_buf_ti[m] as usize;
                let a = (cs.ftle_cands_res[m] as f64) * inv_wsum;
                let a2 = a * a;
                vx_hat += a * (cs.ftle_cands_w[m] as f64);
                vy_hat += a * (cs.cand_du[m] as f64);
                sv_xx += a2 * (ms.metrology.obs_vel_cov[2 * MAX_MICRO + tj] as f64);
                sv_xy += a2 * (ms.metrology.obs_vel_cov[3 * MAX_MICRO + tj] as f64);
                sv_yy += a2 * (ms.metrology.obs_vel_cov[4 * MAX_MICRO + tj] as f64);
                m += 1;
            }
        } else {
            let mut m00 = 0.0f64;
            let mut m01 = 0.0f64;
            let mut m02 = 0.0f64;
            let mut m11 = 0.0f64;
            let mut m12 = 0.0f64;
            let mut m22 = 0.0f64;
            let mut m = 0usize;
            while m < cand_cnt {
                let wj = cs.ftle_cands_res[m] as f64;
                let ux = (cs.cand_buf_dx[m] as f64) * inv_h;
                let uy = (cs.cand_buf_dy[m] as f64) * inv_h;
                m00 += wj;
                m01 += wj * ux;
                m02 += wj * uy;
                m11 += wj * ux * ux;
                m12 += wj * ux * uy;
                m22 += wj * uy * uy;
                m += 1;
            }
            if !(m00.is_finite()
                && m11.is_finite()
                && m22.is_finite()
                && m01.is_finite()
                && m02.is_finite()
                && m12.is_finite()
                && m00 > 1.0e-18)
            {
                clear_bench_residual_slot(ms, ti, cand_cnt);
                idx += 1;
                continue;
            }
            // Scale-normalize `N = M / m00` (`N_00 = 1`) and explicitly bound `lambda_min(N)`
            // and `kappa(N) = lambda_max / lambda_min` (`tr(N^-1) = cof_tr / det_n`).
            let inv_m00 = 1.0 / m00;
            let n01 = m01 * inv_m00;
            let n02 = m02 * inv_m00;
            let n11 = m11 * inv_m00;
            let n12 = m12 * inv_m00;
            let n22 = m22 * inv_m00;
            let nc00 = n11 * n22 - n12 * n12;
            let nc01 = -(n01 * n22 - n02 * n12);
            let nc02 = n01 * n12 - n02 * n11;
            let nc11 = n22 - n02 * n02;
            let nc22 = n11 - n01 * n01;
            let det_n = nc00 + n01 * nc01 + n02 * nc02;
            let cof_tr = nc00 + nc11 + nc22;
            let frob2 = 1.0
                + n11 * n11
                + n22 * n22
                + 2.0 * (n01 * n01 + n02 * n02 + n12 * n12);
            let frob = wasm_sqrt_f64(frob2.max(0.0));
            if !(det_n.is_finite()
                && det_n > 1.0e-12
                && n11 >= 1.0e-4
                && n22 >= 1.0e-4
                && nc00 > 1.0e-12
                && nc11 > 1.0e-12
                && nc22 > 1.0e-12
                && cof_tr.is_finite()
                && cof_tr > 0.0
                && frob.is_finite()
                && (det_n / cof_tr) >= 1.0e-4
                && frob * (cof_tr / det_n) <= 500.0)
            {
                clear_bench_residual_slot(ms, ti, cand_cnt);
                idx += 1;
                continue;
            }
            let inv_det = 1.0 / det_n;
            let inv00 = nc00 * inv_det * inv_m00;
            let inv01 = nc01 * inv_det * inv_m00;
            let inv02 = nc02 * inv_det * inv_m00;
            let mut sum_a = 0.0f64;
            let mut sum_a2 = 0.0f64;
            m = 0;
            while m < cand_cnt {
                let tj = cs.cand_buf_ti[m] as usize;
                let wj = cs.ftle_cands_res[m] as f64;
                let ux = (cs.cand_buf_dx[m] as f64) * inv_h;
                let uy = (cs.cand_buf_dy[m] as f64) * inv_h;
                let a = wj * (inv00 + inv01 * ux + inv02 * uy);
                let a2 = a * a;
                sum_a += a;
                sum_a2 += a2;
                vx_hat += a * (cs.ftle_cands_w[m] as f64);
                vy_hat += a * (cs.cand_du[m] as f64);
                sv_xx += a2 * (ms.metrology.obs_vel_cov[2 * MAX_MICRO + tj] as f64);
                sv_xy += a2 * (ms.metrology.obs_vel_cov[3 * MAX_MICRO + tj] as f64);
                sv_yy += a2 * (ms.metrology.obs_vel_cov[4 * MAX_MICRO + tj] as f64);
                m += 1;
            }
            if !sum_a.is_finite() || !sum_a2.is_finite() || (sum_a - 1.0).abs() > 1.0e-3 || sum_a2 > 4.0
            {
                clear_bench_residual_slot(ms, ti, cand_cnt);
                idx += 1;
                continue;
            }
        }

        let rx = (ms.metrology.obs_vel_cov[ti] as f64) - vx_hat;
        let ry = (ms.metrology.obs_vel_cov[MAX_MICRO + ti] as f64) - vy_hat;
        match inv_2x2_quad_f64(sv_xx, sv_xy, sv_yy, rx, ry) {
            Some(d2) if rx.is_finite() && ry.is_finite() => {
                ms.metrology.export_micro_residual[ti] = rx as f32;
                ms.metrology.export_micro_residual[MAX_MICRO + ti] = ry as f32;
                ms.metrology.export_micro_residual[2 * MAX_MICRO + ti] = d2 as f32;
                let n_bits =
                    ((cand_cnt.min(255) as u32) << QUAL_NEIGHBOR_SHIFT) & QUAL_NEIGHBOR_MASK;
                ms.metrology.export_micro_quality[ti] = (ms.metrology.export_micro_quality[ti]
                    & !(QUAL_BENCH_RES_USABLE
                        | QUAL_BENCH_TRANSLATION_ONLY
                        | QUAL_NEIGHBOR_MASK))
                    | QUAL_BENCH_RES_USABLE
                    | (if translation_only {
                        QUAL_BENCH_TRANSLATION_ONLY
                    } else {
                        0
                    })
                    | n_bits;
            }
            _ => {
                clear_bench_residual_slot(ms, ti, cand_cnt);
            }
        }
        idx += 1;
    }
}

/// Refreshes strictly per-track continuum export columns `6` (`body_id`) and `7` (`ftle_val`)
/// without altering `export_micro[..+0..+6]` or tracker-owned telemetry `swarm_telem[0..=15]` (`F5`).
#[inline(never)]
fn sync_micro_physics_exports(ms: &mut MicroSwarmState, cs: &mut ContinuumState) {
    let use_cached = ms.swarm_active != 0
        && cs.lists_frame_idx != u32::MAX
        && cs.lists_frame_idx == cs.last_frame_idx;
    let lim = if use_cached {
        (cs.active_max_slot.max(cs.prev_active_max_slot) as usize).min(MAX_MICRO)
    } else {
        MAX_MICRO
    };
    let mut i = 0usize;
    while i < lim {
        let base = i * MICRO_EXPORT_STRIDE;
        if ms.swarm_active == 0 || ms.tracks.state[i] == MICRO_STATE_INACTIVE {
            ms.export_micro[base + 6] = 0.0;
            ms.export_micro[base + 7] = 0.0;
        } else {
            let ftle = if ms.tracks.ftle_val[i].is_finite() {
                ms.tracks.ftle_val[i]
            } else {
                ms.tracks.ftle_val[i] = -999.0;
                -999.0
            };
            ms.export_micro[base + 6] = ms.tracks.body_id[i] as f32;
            ms.export_micro[base + 7] = ftle;
        }
        i += 1;
    }
    if use_cached {
        cs.prev_active_max_slot = cs.active_max_slot;
    }
}

/// Top-level Pillar 3 continuum physics entry point. Runs after `soa_step(...) == true` and
/// updates `body_id`, `ftle_val`, `export_micro` columns `6..=7`, and `swarm_telem[16..64]` cleanly.
#[inline(never)]
pub fn continuum_step(
    ms: &mut MicroSwarmState,
    cs: &mut ContinuumState,
    frame_idx: u32,
    frame_w: usize,
    frame_h: usize,
    dt_sec: f32,
) {
    if ms.swarm_active == 0
        || !dt_sec.is_finite()
        || dt_sec <= 1e-4
        || frame_w < 32
        || frame_h < 32
    {
        invalidate_continuum_continuity(ms, cs);
        sync_micro_physics_exports(ms, cs);
        write_continuum_telemetry(ms, cs);
        return;
    }

    if cs.initialized == 0 {
        cs.reset();
        apply_active_export_sentinels(cs);
        cs.clean_epoch_frame = frame_idx;
    } else if cs.last_frame_idx == u32::MAX {
        apply_active_export_sentinels(cs);
        cs.clean_epoch_frame = frame_idx;
    } else if frame_idx == cs.last_frame_idx {
        sync_micro_physics_exports(ms, cs);
        write_continuum_telemetry(ms, cs);
        return;
    } else if frame_idx < cs.last_frame_idx || frame_idx > cs.last_frame_idx.saturating_add(1) {
        let is_back_or_wrap = frame_idx < cs.last_frame_idx;
        reset_on_frame_discontinuity(ms, cs, frame_idx, is_back_or_wrap);
    }

    cs.clear_diagnostics();
    cs.sim_time_s += dt_sec as f64;
    let slot = (frame_idx as usize) & (FTLE_TIME_RING - 1);
    cs.time_ring_s[slot] = cs.sim_time_s;
    cs.time_ring_frame[slot] = frame_idx;
    cs.last_frame_idx = frame_idx;

    let q = ((frame_w as f32) / 640.0).max(0.25);

    cs.lists_frame_idx = u32::MAX;
    build_sparse_knn_graph(ms, cs, frame_idx, frame_w, frame_h, q);
    if (cs.initialized & 2u8) != 0 || cs.prev_body_cnt != 0 {
        solve_rigid_bodies_se2_impl(ms, cs, frame_idx, frame_w, frame_h, q, dt_sec);
    } else {
        cs.top_body_cum_theta = 0.0;
        cs.body_rank_len = 0;
    }
    solve_eulerian_grad_and_ftle(ms, cs, frame_idx, frame_w, frame_h, dt_sec);
    solve_modal_and_swarm_kuramoto(ms, cs, frame_w, frame_h, dt_sec);
    if ms.get_micro_metrology_mode() != 0 {
        solve_bench_loo_residuals(ms, cs, frame_w, frame_h, q);
        update_hero_candidates_hysteresis(ms);
    }
    sync_micro_physics_exports(ms, cs);
    write_continuum_telemetry(ms, cs);
}
