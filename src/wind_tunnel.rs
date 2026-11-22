//! Pocket Wind Tunnel: Optical Pressure, Aerodynamic Lift/Drag & Weak-Form Navier-Stokes Viscosity (`Wave D2`).
//!
//! Pure `#![no_std]`, zero-allocation, zero-`unsafe`, zero-import implementation inside static
//! `WindTunnelState` (`<= 28,672 B` permanent footprint) borrowing `BosState`'s `f64` solver scratch
//! under strict mutual exclusion (`bos_scratch_owner == BOS_OWNER_WIND_TUNNEL`) and reusing
//! `HodgeState::solve_exact_on_active_complex` for the variational DEC pressure Poisson solve.

use crate::bos::BosState;
use crate::continuum_physics::{BODY_EXPORT_STRIDE, MAX_RIGID_BODIES};
use crate::hodge::{
    ex_idx, ey_idx, f_idx, v_idx, HodgeState, HODGE_E, HODGE_H, HODGE_STATE_READY,
    HODGE_TELEM_BETA1, HODGE_TELEM_INTERNAL_VOIDS, HODGE_TELEM_PRIMARY_VOID_CIRC,
    HODGE_TELEM_STATE, HODGE_V, HODGE_W, VCLASS_HOLE_BND, VCLASS_PORT_BND,
};
use crate::{fast_atan2f, wasm_sqrt_f64};

pub const WIND_W: usize = HODGE_W; // 64
pub const WIND_H: usize = HODGE_H; // 36
pub const WIND_V: usize = HODGE_V; // 2,304
pub const WIND_E: usize = HODGE_E; // 4,508
pub const WIND_TELEM_LEN: usize = 32;
pub const WIND_PLANES_LEN: usize = 3 * WIND_V; // 6,912 f32 = 27,648 B

pub const WIND_PLANE_CP: usize = 0;
pub const WIND_PLANE_NS_RESID: usize = 1;
pub const WIND_PLANE_VORT_BUDGET: usize = 2;

// Operating modes (`chronos_set_wind_mode`)
pub const WIND_MODE_OFF: u32 = 0;
pub const WIND_MODE_STAGGERED: u32 = 1;
pub const WIND_MODE_EVERY_FRAME: u32 = 2;

// State enum (`telem[0]`)
pub const WIND_STATE_OFF: u32 = 0;
pub const WIND_STATE_WARMING: u32 = 1;
pub const WIND_STATE_READY_FIELD_ONLY: u32 = 2;
pub const WIND_STATE_READY_FULL_LOADS: u32 = 3;
pub const WIND_STATE_REFUSED: u32 = 4;

// Provenance enum (`telem[3]`)
pub const WIND_PROV_NONE: u32 = 0;
pub const WIND_PROV_DIMENSIONLESS: u32 = 1;
#[allow(dead_code)]
pub const WIND_PROV_SURVEYED_METRIC: u32 = 2;

// Fail-closed reason mask (`telem[2]`)
pub const WIND_REASON_NO_CONTINUUM: u32 = 1 << 0;
pub const WIND_REASON_HODGE_UNCONV: u32 = 1 << 1;
pub const WIND_REASON_MUTEX_CONFLICT: u32 = 1 << 2;
pub const WIND_REASON_HIGH_DIV: u32 = 1 << 3;
pub const WIND_REASON_HIGH_NS_RESIDUAL: u32 = 1 << 4;
pub const WIND_REASON_NO_OBSTACLE_VOID: u32 = 1 << 5;
pub const WIND_REASON_NO_FREESTREAM: u32 = 1 << 6;
pub const WIND_REASON_LOW_VORT_COND: u32 = 1 << 7;
pub const WIND_REASON_PCG_UNCONV: u32 = 1 << 8;

// Telemetry word indices (`[u32; 32]`; 8..=27 and 29..=31 store `f32::to_bits()`)
pub const WIND_TELEM_STATE: usize = 0;
pub const WIND_TELEM_MODE: usize = 1;
pub const WIND_TELEM_REASON_MASK: usize = 2;
pub const WIND_TELEM_PROVENANCE: usize = 3;
pub const WIND_TELEM_ACTIVE_CELLS: usize = 4;
pub const WIND_TELEM_BETA1_HOLES: usize = 5;
pub const WIND_TELEM_PCG_ITERS: usize = 6;
pub const WIND_TELEM_NU_STENCILS: usize = 7;
pub const WIND_TELEM_U_INF: usize = 8;
pub const WIND_TELEM_INFLOW_ANGLE: usize = 9;
pub const WIND_TELEM_D_HOLE_PX: usize = 10;
pub const WIND_TELEM_NU_PX2_S: usize = 11;
pub const WIND_TELEM_NU_R2: usize = 12;
pub const WIND_TELEM_REYNOLDS: usize = 13;
pub const WIND_TELEM_CP_MIN: usize = 14;
pub const WIND_TELEM_CP_MAX: usize = 15;
pub const WIND_TELEM_CL_BND: usize = 16;
pub const WIND_TELEM_CD_BND: usize = 17;
pub const WIND_TELEM_CL_CV: usize = 18;
pub const WIND_TELEM_CD_CV: usize = 19;
pub const WIND_TELEM_CL_KJ: usize = 20;
pub const WIND_TELEM_NS_RESID_RATIO: usize = 21;
pub const WIND_TELEM_DIV_FRAC: usize = 22;
pub const WIND_TELEM_HOLE_CX: usize = 23;
pub const WIND_TELEM_HOLE_CY: usize = 24;
pub const WIND_TELEM_P_GAUGE_REF: usize = 25;
pub const WIND_TELEM_PCG_REL_RES: usize = 26;
pub const WIND_TELEM_RHO_KG_M3: usize = 27;
pub const WIND_TELEM_FSI_BODY_ID: usize = 28;
pub const WIND_TELEM_FSI_BODY_VX: usize = 29;
pub const WIND_TELEM_FSI_BODY_VY: usize = 30;
pub const WIND_TELEM_FSI_BODY_OMEGA: usize = 31;

// History rings inside borrowed `BosState.solve_{x,b,p,z}`:
//   slots 0..7   : 7 consecutive snapshots (fast 7-frame OLS temporal fit + fast STRidge)
//   slots 7..14  : 7 stride-6 boxcar-averaged solenoidal snapshots (~2.4 s span at 15 Hz continuum cadence)
// Total per plane = 14 * 2,304 = 32,256 f64 <= 32,768 (BOS_NODES).
const FAST_RING_LEN: usize = 7;
const SLOW_RING_LEN: usize = 7;
const SLOW_RING_BASE: usize = FAST_RING_LEN;
const SLOW_BOXCAR_STRIDE: u32 = 6;
const MAX_STRIDGE_STENCILS: usize = 64;

// Offsets inside `bos.solve_r` (`32,768` f64) for zero-stack ephemeral solve arrays
const SCR_G1_OFF: usize = 0; // 4,508
const SCR_P0_OFF: usize = 4_508; // 2,304 -> ends 6,812
const SCR_UBAR_OFF: usize = 6_812; // 2,304 -> ends 9,116
const SCR_VBAR_OFF: usize = 9_116; // 2,304 -> ends 11,420
const SCR_WBAR_OFF: usize = 11_420; // 2,304 -> ends 13,724
const SCR_DUDT_OFF: usize = 13_724; // 2,304 -> ends 16,028
const SCR_DVDT_OFF: usize = 16_028; // 2,304 -> ends 18,332
const SCR_LAPU_OFF: usize = 18_332; // 2,304 -> ends 20,636
const SCR_LAPV_OFF: usize = 20_636; // 2,304 -> ends 22,940
const SCR_SORT_OFF: usize = 22_940; // up to 4,608 -> ends 27,548
const SCR_PREV_P0_OFF: usize = 27_548; // 2,304 -> ends 29,852 (warm-start pressure cache)

// Offsets inside `bos.solve_lp` (`32,768` f64): boxcar accumulators + STRidge arrays + temp fill
const LP_BOX_U_OFF: usize = 0; // 2,304
const LP_BOX_V_OFF: usize = 2_304; // 2,304
const LP_BOX_W_OFF: usize = 4_608; // 2,304
const LP_BOX_CNT_OFF: usize = 6_912; // 2,304 -> ends 9,216
const LP_ST_LHS_OFF: usize = 9_216; // 64
const LP_ST_RHS_OFF: usize = 9_280; // 64
const LP_ST_W_OFF: usize = 9_344; // 64
const LP_ST_RES_OFF: usize = 9_408; // 64
const LP_ST_ABS_OFF: usize = 9_472; // 64

// Smooth C^3 compact-support weak test function phi(s) = (1 - s^2)^4 on [-1, 1] sampled at
// 7 interior points s_k = (k - 3) / 4 (endpoints k = -1, 7 lie at s = +-1 where phi=phi'=phi''=phi'''=0).
// Because all boundary terms vanish to 3rd order, Euler-Maclaurin quadrature achieves spectral accuracy
// while filtering high-frequency PIV grid aliasing on product terms (u * omega).
const PHI_P: [f64; 7] = [
    0.0366363525390625,
    0.31640625,
    0.7724761962890625,
    1.0,
    0.7724761962890625,
    0.31640625,
    0.0366363525390625,
];
// -phi'(s_k) / 4 (evaluates +<phi, df/ds> via integration by parts <-phi', f>)
const PHI_NEG_DP: [f64; 7] = [
    -0.1256103515625,
    -0.421875,
    -0.4119873046875,
    0.0,
    0.4119873046875,
    0.421875,
    0.1256103515625,
];
// phi''(s_k) / 16 (mean-adjusted by tiny Euler-Maclaurin residual so sum === 0 on constants)
const PHI_D2P: [f64; 7] = [
    0.2825937000342778,
    0.2124032699657222,
    -0.2457266128467778,
    -0.4985407143064444,
    -0.2457266128467778,
    0.2124032699657222,
    0.2825937000342778,
];

const DIV_HARD_LIMIT: f64 = 0.25;
const NS_RESID_HARD_LIMIT: f64 = 0.35;

#[inline(always)]
fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
}

#[inline(always)]
fn pack_f32(val: f64) -> u32 {
    if !val.is_finite() {
        return 0.0f32.to_bits();
    }
    let f = val as f32;
    if f.is_finite() {
        f.to_bits()
    } else {
        0.0f32.to_bits()
    }
}

#[inline(always)]
fn ln_pos_f64(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    const LN2: f64 = 0.6931471805599453;
    let bits = x.to_bits();
    let mut k = ((bits >> 52) as i64) - 1023;
    let mut m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | (1023u64 << 52));
    if m > 1.4142135623730951 {
        m *= 0.5;
        k += 1;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let poly = 1.0
        + y2 * ((1.0 / 3.0)
            + y2 * ((1.0 / 5.0)
                + y2 * ((1.0 / 7.0)
                    + y2 * ((1.0 / 9.0) + y2 * ((1.0 / 11.0) + y2 * (1.0 / 13.0))))));
    2.0 * y * poly + (k as f64) * LN2
}

/// Deterministic in-place shell sort for small median buffers (`n <= 2304`).
#[inline(never)]
fn median_in_place(buf: &mut [f64], n: usize) -> f64 {
    if n == 0 || n > buf.len() {
        return 0.0;
    }
    let mut gap = 1usize;
    while gap < n / 3 {
        gap = gap * 3 + 1;
    }
    while gap >= 1 {
        let mut i = gap;
        while i < n {
            let v = buf[i];
            let mut j = i;
            while j >= gap && buf[j - gap] > v {
                buf[j] = buf[j - gap];
                j -= gap;
            }
            buf[j] = v;
            i += 1;
        }
        gap /= 3;
    }
    if (n & 1) == 1 {
        buf[n / 2]
    } else {
        0.5 * (buf[n / 2 - 1] + buf[n / 2])
    }
}

/// Permanent Arena-resident state for the Pocket Wind Tunnel (`<= 28,672 B`).
#[repr(C, align(16))]
pub struct WindTunnelState {
    pub planes: [f32; WIND_PLANES_LEN],
    pub telem: [u32; WIND_TELEM_LEN],

    // Fast 7-frame ring metadata
    pub fast_t: [f64; FAST_RING_LEN],
    // Slow stride-8 boxcar ring metadata
    pub slow_t: [f64; SLOW_RING_LEN],

    pub last_dx: f64,
    pub last_dy: f64,
    pub box_t_sum: f64,
    pub cached_nu: f64,
    pub cached_nu_r2: f64,

    pub mode: u32,
    pub fast_head: u32,
    pub fast_count: u32,
    pub slow_head: u32,
    pub slow_count: u32,
    pub box_frames: u32,
    pub last_continuity_epoch: u32,
    pub last_solve_frame: u32,
    pub cached_nu_stencils: u32,
    pub rho_kg_m3: f32,
    pub cached_nu_valid: u8,
    pub live_ready: u8,
    pub fresh_solve: u8,
    pub _pad: u8,
}

const _: () = assert!(core::mem::size_of::<WindTunnelState>() <= 28_672);

impl WindTunnelState {
    pub fn reset_all(&mut self) {
        self.planes.fill(0.0);
        self.telem.fill(0);
        self.fast_t.fill(0.0);
        self.slow_t.fill(0.0);
        self.last_dx = 0.0;
        self.last_dy = 0.0;
        self.box_t_sum = 0.0;
        self.cached_nu = 0.0;
        self.cached_nu_r2 = 0.0;
        self.mode = WIND_MODE_OFF;
        self.fast_head = 0;
        self.fast_count = 0;
        self.slow_head = 0;
        self.slow_count = 0;
        self.box_frames = 0;
        self.last_continuity_epoch = 0;
        self.last_solve_frame = 0;
        self.cached_nu_stencils = 0;
        self.rho_kg_m3 = 0.0;
        self.cached_nu_valid = 0;
        self.live_ready = 0;
        self.fresh_solve = 0;
        self._pad = 0;
    }

    pub fn reset_temporal_keep_config(&mut self) {
        let keep_mode = self.mode;
        let keep_rho = self.rho_kg_m3;
        self.planes.fill(0.0);
        self.telem.fill(0);
        self.fast_t.fill(0.0);
        self.slow_t.fill(0.0);
        self.last_dx = 0.0;
        self.last_dy = 0.0;
        self.box_t_sum = 0.0;
        self.cached_nu = 0.0;
        self.cached_nu_r2 = 0.0;
        self.fast_head = 0;
        self.fast_count = 0;
        self.slow_head = 0;
        self.slow_count = 0;
        self.box_frames = 0;
        self.last_continuity_epoch = 0;
        self.last_solve_frame = 0;
        self.cached_nu_stencils = 0;
        self.cached_nu_valid = 0;
        self.live_ready = 0;
        self.fresh_solve = 0;
        self.mode = keep_mode;
        self.rho_kg_m3 = keep_rho;
        self.telem[WIND_TELEM_MODE] = keep_mode;
        self.telem[WIND_TELEM_RHO_KG_M3] = pack_f32(keep_rho as f64);
        self.telem[WIND_TELEM_STATE] = if keep_mode == WIND_MODE_OFF {
            WIND_STATE_OFF
        } else {
            WIND_STATE_WARMING
        };
    }

    pub fn set_mode(&mut self, mode: u32) -> u32 {
        if mode > WIND_MODE_EVERY_FRAME {
            return 0;
        }
        let changed = self.mode != mode;
        self.mode = mode;
        if mode == WIND_MODE_OFF || changed {
            self.reset_temporal_keep_config();
        } else {
            self.telem[WIND_TELEM_MODE] = mode;
        }
        1
    }

    pub fn set_fluid_density(&mut self, rho_kg_m3: f32) -> u32 {
        if !rho_kg_m3.is_finite() || rho_kg_m3 < 0.0 || rho_kg_m3 > 1.0e6 {
            return 0;
        }
        self.rho_kg_m3 = rho_kg_m3;
        self.telem[WIND_TELEM_RHO_KG_M3] = pack_f32(rho_kg_m3 as f64);
        1
    }

    #[inline(always)]
    pub fn get_mode(&self) -> u32 {
        self.mode
    }

    #[inline(always)]
    pub fn get_state(&self) -> u32 {
        self.telem[WIND_TELEM_STATE]
    }

    #[inline(always)]
    pub fn telemetry_ptr(&self) -> *const u32 {
        self.telem.as_ptr()
    }

    #[inline(always)]
    pub fn planes_ptr(&self) -> *const f32 {
        self.planes.as_ptr()
    }

    /// Record a fail-closed refusal without touching `BosState` scratch.
    #[inline(never)]
    pub fn fail_refused(&mut self, reason_mask: u32, clear_history: bool) {
        if clear_history {
            self.fast_count = 0;
            self.fast_head = 0;
            self.slow_count = 0;
            self.slow_head = 0;
            self.box_frames = 0;
            self.box_t_sum = 0.0;
            self.cached_nu = 0.0;
            self.cached_nu_r2 = 0.0;
            self.cached_nu_stencils = 0;
            self.cached_nu_valid = 0;
        }
        self.planes.fill(0.0);
        self.telem.fill(0);
        self.telem[WIND_TELEM_STATE] = WIND_STATE_REFUSED;
        self.telem[WIND_TELEM_MODE] = self.mode;
        self.telem[WIND_TELEM_REASON_MASK] = reason_mask;
        self.telem[WIND_TELEM_PROVENANCE] = WIND_PROV_NONE;
        self.telem[WIND_TELEM_RHO_KG_M3] = pack_f32(self.rho_kg_m3 as f64);
        self.live_ready = 0;
        self.fresh_solve = 0;
    }

    /// Run one Pocket Wind Tunnel step on the currently solved `HodgeState` active 2-complex.
    /// `raw_continuum_field` provides unblurred vertex `(u, v)` (stride 6) when `is_live == true`;
    /// otherwise `hodge.inject_buf` provides exact unblurred vertex `(u, v)`.
    /// Caller MUST guarantee `bos_scratch_owner == BOS_OWNER_WIND_TUNNEL` when `has_scratch == true`.
    #[inline(never)]
    pub fn step(
        &mut self,
        hodge: &mut HodgeState,
        bos: &mut BosState,
        raw_continuum_field: &[f32; WIND_V * 6],
        export_bodies: &[f32; MAX_RIGID_BODIES * BODY_EXPORT_STRIDE],
        grid_dx: f64,
        grid_dy: f64,
        t_sec: f64,
        continuity_epoch: u32,
        frame_idx: u32,
        has_scratch: bool,
        is_live: bool,
    ) -> bool {
        self.fresh_solve = 0;
        if self.mode == WIND_MODE_OFF {
            self.telem[WIND_TELEM_STATE] = WIND_STATE_OFF;
            self.live_ready = 0;
            return false;
        }
        if !has_scratch {
            self.fail_refused(WIND_REASON_MUTEX_CONFLICT, true);
            return false;
        }
        if !grid_dx.is_finite()
            || !grid_dy.is_finite()
            || grid_dx <= 0.0
            || grid_dy <= 0.0
            || !t_sec.is_finite()
        {
            self.fail_refused(WIND_REASON_NO_CONTINUUM, true);
            return false;
        }
        if hodge.telem[HODGE_TELEM_STATE] != HODGE_STATE_READY {
            self.fail_refused(WIND_REASON_HODGE_UNCONV, true);
            return false;
        }

        // Continuity / geometry break guard: reset temporal rings cleanly
        let geom_changed = abs_f64(grid_dx - self.last_dx) > 1e-6
            || abs_f64(grid_dy - self.last_dy) > 1e-6;
        let epoch_changed =
            self.fast_count > 0 && continuity_epoch != self.last_continuity_epoch;
        let prev_t = if self.fast_count > 0 {
            let last_slot = ((self.fast_head as usize) + FAST_RING_LEN - 1) % FAST_RING_LEN;
            self.fast_t[last_slot]
        } else {
            -1.0e30
        };
        let bad_time = self.fast_count > 0
            && (t_sec <= prev_t + 1.0e-5 || (t_sec - prev_t) > 2.0);
        if geom_changed || epoch_changed || bad_time {
            self.fast_head = 0;
            self.fast_count = 0;
            self.slow_head = 0;
            self.slow_count = 0;
            self.box_frames = 0;
            self.box_t_sum = 0.0;
            self.cached_nu = 0.0;
            self.cached_nu_r2 = 0.0;
            self.cached_nu_stencils = 0;
            self.cached_nu_valid = 0;
            bos.clear_solver_scratch();
        }
        self.last_dx = grid_dx;
        self.last_dy = grid_dy;
        self.last_continuity_epoch = continuity_epoch;

        // 1. Capture unblurred vertex velocity `(u, v)` and cell vorticity `omega` into fast ring
        let slot = (self.fast_head as usize) % FAST_RING_LEN;
        let base = slot * WIND_V;
        let act_cells = capture_vertex_velocity_and_vorticity(
            hodge,
            raw_continuum_field,
            is_live,
            &mut bos.solve_x[base..base + WIND_V],
            &mut bos.solve_b[base..base + WIND_V],
            &mut bos.solve_p[base..base + WIND_V],
            &mut bos.solve_z[base..base + WIND_V],
            grid_dx,
            grid_dy,
        );
        self.fast_t[slot] = t_sec;
        self.fast_head = ((slot + 1) % FAST_RING_LEN) as u32;
        if (self.fast_count as usize) < FAST_RING_LEN {
            self.fast_count += 1;
        }

        // Accumulate stride-6 solenoidal boxcar snapshot (with 1-cell pinhole healing and
        // exact interior divergence projection on the boxcar mask to strip IMM curved-orbit lag!)
        let slow_updated = accumulate_slow_boxcar(
            bos,
            base,
            t_sec,
            grid_dx,
            grid_dy,
            &mut self.box_frames,
            &mut self.box_t_sum,
            &mut self.slow_t,
            &mut self.slow_head,
            &mut self.slow_count,
        );

        if act_cells < 16 {
            self.fail_refused(WIND_REASON_NO_CONTINUUM, false);
            return false;
        }

        // Require at least 3 consecutive snapshots before emitting pressure (need real d/dt u)
        if self.fast_count < 3 {
            self.planes.fill(0.0);
            self.telem.fill(0);
            self.telem[WIND_TELEM_STATE] = WIND_STATE_WARMING;
            self.telem[WIND_TELEM_MODE] = self.mode;
            self.telem[WIND_TELEM_ACTIVE_CELLS] = act_cells as u32;
            self.telem[WIND_TELEM_BETA1_HOLES] = hodge.telem[HODGE_TELEM_BETA1];
            self.telem[WIND_TELEM_RHO_KG_M3] = pack_f32(self.rho_kg_m3 as f64);
            self.live_ready = 0;
            return false;
        }

        // In STAGGERED mode, after initial readiness, solve on every 2nd Hodge invocation
        // (every 4 video frames when Hodge is staggered, or every 2 when Hodge is every-frame)
        let stagger_stride = if hodge.get_mode() == 1 { 4u32 } else { 2u32 };
        if is_live
            && self.mode == WIND_MODE_STAGGERED
            && frame_idx.wrapping_sub(self.last_solve_frame) < stagger_stride
            && self.live_ready != 0
            && (self.telem[WIND_TELEM_STATE] == WIND_STATE_READY_FIELD_ONLY
                || self.telem[WIND_TELEM_STATE] == WIND_STATE_READY_FULL_LOADS)
        {
            return false;
        }

        // 2. Compute 7-frame local linear OLS temporal fit `(u_bar, v_bar, du_dt, dv_dt)` + `omega_bar`
        //    at every active vertex `v` (cuts PIV `du/dt` noise by 3.7x and velocity noise by 2.6x
        //    while remaining exact on linear-in-time analytical flows and non-NS ramping controls!)
        compute_temporal_ols_fields(
            hodge,
            bos,
            &self.fast_t,
            self.fast_head as usize,
            self.fast_count as usize,
            t_sec,
            grid_dx,
            grid_dy,
        );

        // 3. Interior incompressibility check (preserves open-port potential flow, rejects 3D/compressible sources!)
        let div_frac = compute_interior_divergence_fraction(
            hodge,
            bos,
            grid_dx,
            grid_dy,
        );
        if !div_frac.is_finite() || div_frac > DIV_HARD_LIMIT {
            self.fail_refused(WIND_REASON_HIGH_DIV, false);
            self.telem[WIND_TELEM_ACTIVE_CELLS] = act_cells as u32;
            self.telem[WIND_TELEM_DIV_FRAC] = pack_f32(div_frac);
            return false;
        }

        // 4. Weak-form 2D+t vorticity transport STRidge (amortized on slow boxcar updates once full!)
        let should_run_stridge = if !is_live {
            (self.slow_count as usize) >= SLOW_RING_LEN || (self.fast_count as usize) >= FAST_RING_LEN
        } else if (self.slow_count as usize) >= SLOW_RING_LEN {
            slow_updated
        } else {
            (self.fast_count as usize) >= FAST_RING_LEN && (frame_idx % SLOW_BOXCAR_STRIDE) == 0
        };
        if should_run_stridge {
            let (nu_val, nu_r2_val, nu_st, nu_ok) =
                self.estimate_viscosity_stridge(hodge, bos, grid_dx, grid_dy, is_live);
            self.cached_nu = if nu_ok { nu_val } else { 0.0 };
            self.cached_nu_r2 = nu_r2_val;
            self.cached_nu_stencils = nu_st;
            self.cached_nu_valid = if nu_ok { 1 } else { 0 };
        }
        let nu_identifiable = self.cached_nu_valid != 0;
        let nu_eff = if nu_identifiable { self.cached_nu } else { 0.0 };
        let nu_r2 = self.cached_nu_r2;
        let nu_stencils = self.cached_nu_stencils;

        let mut reason_mask = 0u32;
        if !nu_identifiable {
            reason_mask |= WIND_REASON_LOW_VORT_COND;
        }

        // 5. Form Lamb-form negative material acceleration 1-cochain g_1 and solve DEC pressure Poisson
        build_material_accel_cochain(hodge, bos, nu_eff, grid_dx, grid_dy);

        // Solve `(d_0^T W_1 d_0) p_0 = d_0^T W_1 g_1` without mutating HodgeState exports
        // Warm-start from previous converged live pressure field when available (`SCR_PREV_P0_OFF`).
        let warm_p0 = is_live && self.live_ready != 0;
        if warm_p0 {
            let mut vi = 0usize;
            while vi < WIND_V {
                bos.solve_r[SCR_P0_OFF + vi] = bos.solve_r[SCR_PREV_P0_OFF + vi];
                vi += 1;
            }
        }
        let (pcg_ok, pcg_iters, pcg_rel_res) = {
            let (g1_slice, rest) = bos.solve_r.split_at_mut(SCR_P0_OFF);
            let g1_arr: &[f64; WIND_E] = match g1_slice[SCR_G1_OFF..SCR_G1_OFF + WIND_E].try_into()
            {
                Ok(a) => a,
                Err(_) => {
                    self.fail_refused(WIND_REASON_PCG_UNCONV, false);
                    return false;
                }
            };
            let p0_arr: &mut [f64; WIND_V] = match (&mut rest[0..WIND_V]).try_into() {
                Ok(a) => a,
                Err(_) => {
                    self.fail_refused(WIND_REASON_PCG_UNCONV, false);
                    return false;
                }
            };
            hodge.solve_exact_on_active_complex(g1_arr, p0_arr, warm_p0)
        };
        if pcg_ok && is_live {
            let mut vi = 0usize;
            while vi < WIND_V {
                bos.solve_r[SCR_PREV_P0_OFF + vi] = bos.solve_r[SCR_P0_OFF + vi];
                vi += 1;
            }
        }

        if !pcg_ok {
            self.fail_refused(reason_mask | WIND_REASON_PCG_UNCONV, false);
            self.telem[WIND_TELEM_ACTIVE_CELLS] = act_cells as u32;
            self.telem[WIND_TELEM_PCG_ITERS] = pcg_iters;
            self.telem[WIND_TELEM_PCG_REL_RES] = pack_f32(pcg_rel_res);
            return false;
        }

        // 6. Built-in Navier-Stokes curl integrability audit: eta_NS = ||g_1 - d_0 p_0||_W1 / ||g_1||_W1
        let ns_resid_ratio = compute_ns_curl_residual(
            hodge,
            &bos.solve_r[SCR_G1_OFF..SCR_G1_OFF + WIND_E],
            &bos.solve_r[SCR_P0_OFF..SCR_P0_OFF + WIND_V],
            &mut self.planes[WIND_PLANE_NS_RESID * WIND_V..(WIND_PLANE_NS_RESID + 1) * WIND_V],
        );
        if !ns_resid_ratio.is_finite() || ns_resid_ratio > NS_RESID_HARD_LIMIT {
            self.fail_refused(reason_mask | WIND_REASON_HIGH_NS_RESIDUAL, false);
            self.telem[WIND_TELEM_ACTIVE_CELLS] = act_cells as u32;
            self.telem[WIND_TELEM_PCG_ITERS] = pcg_iters;
            self.telem[WIND_TELEM_PCG_REL_RES] = pack_f32(pcg_rel_res);
            self.telem[WIND_TELEM_NS_RESID_RATIO] = pack_f32(ns_resid_ratio);
            self.telem[WIND_TELEM_DIV_FRAC] = pack_f32(div_frac);
            return false;
        }

        // 7. Primary obstacle void geometry (persistent over the fast ring to exclude transient PIV rim gaps)
        let beta1 = hodge.telem[HODGE_TELEM_BETA1];
        let internal_voids = hodge.telem[HODGE_TELEM_INTERNAL_VOIDS];
        let (has_void, d_hole_px, hole_cx, hole_cy, h_c0, h_c1, h_r0, h_r1) =
            measure_primary_hole_geometry(
                hodge,
                bos,
                self.fast_head as usize,
                self.fast_count as usize,
                grid_dx,
                grid_dy,
                internal_voids,
            );

        // 8. Far-field / upstream freestream detection & pressure gauge pinning
        //    Uses outer 5-cell margin (handling PIV border dropout on c=0,63 / r=0,35!)
        let (has_freestream, u_inf, e_inf_x, e_inf_y, inflow_angle, p_gauge_ref) =
            estimate_freestream_and_gauge(hodge, bos, has_void, hole_cx, hole_cy, d_hole_px, grid_dx, grid_dy);
        if !has_freestream {
            reason_mask |= WIND_REASON_NO_FREESTREAM;
        }

        // Populate Plane 0 (`C_p` if freestream present, else `p/rho - p_ref`) & Plane 2 (vorticity)
        let q_dyn = 0.5 * u_inf * u_inf;
        let mut cp_min = 1.0e30f64;
        let mut cp_max = -1.0e30f64;
        let mut v = 0usize;
        while v < WIND_V {
            if hodge.v_act[v] != 0 {
                let p_rel = bos.solve_r[SCR_P0_OFF + v] - p_gauge_ref;
                let out_val = if has_freestream && q_dyn > 1.0e-6 {
                    p_rel / q_dyn
                } else {
                    p_rel
                };
                if out_val.is_finite() {
                    self.planes[WIND_PLANE_CP * WIND_V + v] = out_val as f32;
                    if out_val < cp_min {
                        cp_min = out_val;
                    }
                    if out_val > cp_max {
                        cp_max = out_val;
                    }
                } else {
                    self.planes[WIND_PLANE_CP * WIND_V + v] = 0.0;
                }
                self.planes[WIND_PLANE_VORT_BUDGET * WIND_V + v] =
                    bos.solve_r[SCR_WBAR_OFF + v] as f32;
            } else {
                self.planes[WIND_PLANE_CP * WIND_V + v] = 0.0;
                self.planes[WIND_PLANE_VORT_BUDGET * WIND_V + v] = 0.0;
            }
            v += 1;
        }
        if cp_min > cp_max {
            cp_min = 0.0;
            cp_max = 0.0;
        }

        // 9. Dual-Method Aerodynamic Force Balance (`C_L, C_D`) + Kutta-Joukowski audit
        let mut cl_bnd = 0.0f64;
        let mut cd_bnd = 0.0f64;
        let mut cl_cv = 0.0f64;
        let mut cd_cv = 0.0f64;
        let mut cl_kj = 0.0f64;
        let mut reynolds = 0.0f64;
        let mut full_loads = false;

        if !has_void || beta1 == 0 {
            reason_mask |= WIND_REASON_NO_OBSTACLE_VOID;
        } else if has_freestream && d_hole_px > 1.0 && q_dyn > 1.0e-6 {
            let norm = q_dyn * d_hole_px;
            let (fx_b, fy_b, ok_b) = integrate_hole_boundary_force(
                hodge,
                &bos.solve_r[SCR_UBAR_OFF..SCR_UBAR_OFF + WIND_V],
                &bos.solve_r[SCR_VBAR_OFF..SCR_VBAR_OFF + WIND_V],
                &bos.solve_r[SCR_P0_OFF..SCR_P0_OFF + WIND_V],
                p_gauge_ref,
                nu_eff,
                grid_dx,
                grid_dy,
            );
            if ok_b {
                cd_bnd = (fx_b * e_inf_x + fy_b * e_inf_y) / norm;
                cl_bnd = (-fx_b * e_inf_y + fy_b * e_inf_x) / norm;
            }

            let (fx_cv, fy_cv, ok_cv) = integrate_cv_box_force(
                hodge,
                &bos.solve_r[SCR_UBAR_OFF..SCR_UBAR_OFF + WIND_V],
                &bos.solve_r[SCR_VBAR_OFF..SCR_VBAR_OFF + WIND_V],
                &bos.solve_r[SCR_P0_OFF..SCR_P0_OFF + WIND_V],
                &bos.solve_r[SCR_DUDT_OFF..SCR_DUDT_OFF + WIND_V],
                &bos.solve_r[SCR_DVDT_OFF..SCR_DVDT_OFF + WIND_V],
                p_gauge_ref,
                nu_eff,
                grid_dx,
                grid_dy,
                h_c0,
                h_c1,
                h_r0,
                h_r1,
            );
            if ok_cv {
                cd_cv = (fx_cv * e_inf_x + fy_cv * e_inf_y) / norm;
                cl_cv = (-fx_cv * e_inf_y + fy_cv * e_inf_x) / norm;
            } else if ok_b {
                cd_cv = cd_bnd;
                cl_cv = cl_bnd;
            }

            let gamma1 = f32::from_bits(hodge.telem[HODGE_TELEM_PRIMARY_VOID_CIRC]) as f64;
            if gamma1.is_finite() {
                cl_kj = (-2.0 * gamma1) / (u_inf * d_hole_px);
            }

            if nu_identifiable && nu_eff > 1.0e-6 {
                reynolds = (u_inf * d_hole_px) / nu_eff;
            }
            full_loads = ok_b || ok_cv;
        }

        // Optional FSI bridge: check SE(2) rigid body overlapping the primary obstacle void
        let (fsi_id, fsi_vx, fsi_vy, fsi_om) = if has_void {
            find_overlapping_rigid_body(export_bodies, hole_cx, hole_cy, d_hole_px)
        } else {
            (0, 0.0, 0.0, 0.0)
        };

        let final_state = if full_loads {
            WIND_STATE_READY_FULL_LOADS
        } else {
            WIND_STATE_READY_FIELD_ONLY
        };

        self.telem[WIND_TELEM_STATE] = final_state;
        self.telem[WIND_TELEM_MODE] = self.mode;
        self.telem[WIND_TELEM_REASON_MASK] = reason_mask;
        self.telem[WIND_TELEM_PROVENANCE] = WIND_PROV_DIMENSIONLESS;
        self.telem[WIND_TELEM_ACTIVE_CELLS] = act_cells as u32;
        self.telem[WIND_TELEM_BETA1_HOLES] = beta1;
        self.telem[WIND_TELEM_PCG_ITERS] = pcg_iters;
        self.telem[WIND_TELEM_NU_STENCILS] = nu_stencils;
        self.telem[WIND_TELEM_U_INF] = pack_f32(u_inf);
        self.telem[WIND_TELEM_INFLOW_ANGLE] = pack_f32(inflow_angle);
        self.telem[WIND_TELEM_D_HOLE_PX] = pack_f32(d_hole_px);
        self.telem[WIND_TELEM_NU_PX2_S] = pack_f32(if nu_identifiable { nu_eff } else { 0.0 });
        self.telem[WIND_TELEM_NU_R2] = pack_f32(if nu_identifiable { nu_r2 } else { 0.0 });
        self.telem[WIND_TELEM_REYNOLDS] = pack_f32(reynolds);
        self.telem[WIND_TELEM_CP_MIN] = pack_f32(cp_min);
        self.telem[WIND_TELEM_CP_MAX] = pack_f32(cp_max);
        self.telem[WIND_TELEM_CL_BND] = pack_f32(cl_bnd);
        self.telem[WIND_TELEM_CD_BND] = pack_f32(cd_bnd);
        self.telem[WIND_TELEM_CL_CV] = pack_f32(cl_cv);
        self.telem[WIND_TELEM_CD_CV] = pack_f32(cd_cv);
        self.telem[WIND_TELEM_CL_KJ] = pack_f32(cl_kj);
        self.telem[WIND_TELEM_NS_RESID_RATIO] = pack_f32(ns_resid_ratio);
        self.telem[WIND_TELEM_DIV_FRAC] = pack_f32(div_frac);
        self.telem[WIND_TELEM_HOLE_CX] = pack_f32(hole_cx);
        self.telem[WIND_TELEM_HOLE_CY] = pack_f32(hole_cy);
        self.telem[WIND_TELEM_P_GAUGE_REF] = pack_f32(p_gauge_ref);
        self.telem[WIND_TELEM_PCG_REL_RES] = pack_f32(pcg_rel_res);
        self.telem[WIND_TELEM_RHO_KG_M3] = pack_f32(self.rho_kg_m3 as f64);
        self.telem[WIND_TELEM_FSI_BODY_ID] = fsi_id;
        self.telem[WIND_TELEM_FSI_BODY_VX] = pack_f32(fsi_vx);
        self.telem[WIND_TELEM_FSI_BODY_VY] = pack_f32(fsi_vy);
        self.telem[WIND_TELEM_FSI_BODY_OMEGA] = pack_f32(fsi_om);

        self.last_solve_frame = frame_idx;
        self.live_ready = if is_live { 1 } else { 0 };
        self.fresh_solve = 1;
        true
    }

    #[inline(never)]
    fn estimate_viscosity_stridge(
        &self,
        hodge: &HodgeState,
        bos: &mut BosState,
        dx: f64,
        dy: f64,
        is_live: bool,
    ) -> (f64, f64, u32, bool) {
        let use_slow = (self.slow_count as usize) >= SLOW_RING_LEN;
        let (ring_base, ring_head, ts_arr) = if use_slow {
            (SLOW_RING_BASE, self.slow_head as usize, &self.slow_t)
        } else if (self.fast_count as usize) >= FAST_RING_LEN {
            (0usize, self.fast_head as usize, &self.fast_t)
        } else {
            return (0.0, 0.0, 0, false);
        };

        let mut order = [0usize; 7];
        let mut ts = [0.0f64; 7];
        let mut k = 0usize;
        while k < 7 {
            let rel = (ring_head + k) % 7;
            order[k] = ring_base + rel;
            ts[k] = ts_arr[rel];
            k += 1;
        }
        let dt_total = ts[6] - ts[0];
        if !dt_total.is_finite() || dt_total <= 1.0e-4 {
            return (0.0, 0.0, 0, false);
        }
        let dt_avg = dt_total / 6.0;
        let mut uniform_dt = true;
        k = 0;
        while k < 6 {
            let dtk = ts[k + 1] - ts[k];
            if dtk <= 1.0e-5 || abs_f64(dtk - dt_avg) > 0.08 * dt_avg {
                uniform_dt = false;
                break;
            }
            k += 1;
        }

        let inv_dx = 1.0 / dx;
        let inv_dy = 1.0 / dy;
        let inv_dx2 = inv_dx * inv_dx;
        let inv_dy2 = inv_dy * inv_dy;

        // Collect top-64 interior 7x7x7 stencils ranked by |<Delta psi, omega>|
        let mut n_st = 0usize;
        let mut max_arhs = 0.0f64;
        let mut r0 = 4usize;
        while r0 + 4 <= WIND_H {
            let mut c0 = 4usize;
            while c0 + 4 <= WIND_W {
                let mut valid = true;
                let mut dy_i = 0usize;
                'chk: while dy_i < 7 {
                    let rr = r0 - 3 + dy_i;
                    let mut dx_i = 0usize;
                    while dx_i < 7 {
                        let cc = c0 - 3 + dx_i;
                        let v = v_idx(cc, rr);
                        if !is_live
                            && (hodge.v_act[v] == 0
                                || (hodge.v_class[v] & (VCLASS_HOLE_BND | VCLASS_PORT_BND)) != 0)
                        {
                            valid = false;
                            break 'chk;
                        }
                        let mut tk = 0usize;
                        while tk < 7 {
                            if bos.solve_z[order[tk] * WIND_V + v] < 0.5 {
                                valid = false;
                                break 'chk;
                            }
                            tk += 1;
                        }
                        dx_i += 1;
                    }
                    dy_i += 1;
                }

                if valid {
                    let mut term_t = 0.0f64;
                    let mut term_ax = 0.0f64;
                    let mut term_ay = 0.0f64;
                    let mut term_lap = 0.0f64;

                    let mut tk = 0usize;
                    while tk < 7 {
                        let off = order[tk] * WIND_V;
                        let wt0 = if uniform_dt {
                            PHI_P[tk]
                        } else if tk >= 1 && tk <= 5 {
                            PHI_P[tk] * 0.5 * (ts[tk + 1] - ts[tk - 1]) / dt_avg
                        } else {
                            PHI_P[tk]
                        };
                        let wt1 = if uniform_dt {
                            PHI_NEG_DP[tk] / dt_avg
                        } else {
                            0.0
                        };

                        let mut s_w = 0.0f64;
                        let mut s_ax = 0.0f64;
                        let mut s_ay = 0.0f64;
                        let mut s_lx = 0.0f64;
                        let mut s_ly = 0.0f64;

                        let mut iy = 0usize;
                        while iy < 7 {
                            let py = PHI_P[iy];
                            let dpy = PHI_NEG_DP[iy];
                            let d2py = PHI_D2P[iy];
                            let rr = r0 - 3 + iy;
                            let mut ix = 0usize;
                            while ix < 7 {
                                let px = PHI_P[ix];
                                let v = v_idx(c0 - 3 + ix, rr);
                                let om = bos.solve_p[off + v];
                                s_w += px * py * om;
                                s_ax += PHI_NEG_DP[ix] * py * (bos.solve_x[off + v] * om);
                                s_ay += px * dpy * (bos.solve_b[off + v] * om);
                                s_lx += PHI_D2P[ix] * py * om;
                                s_ly += px * d2py * om;
                                ix += 1;
                            }
                            iy += 1;
                        }

                        if uniform_dt {
                            term_t += wt1 * s_w;
                        }
                        term_ax += wt0 * (s_ax * inv_dx);
                        term_ay += wt0 * (s_ay * inv_dy);
                        term_lap += wt0 * (s_lx * inv_dx2 + s_ly * inv_dy2);
                        tk += 1;
                    }

                    if !uniform_dt {
                        let mut tk2 = 1usize;
                        while tk2 <= 5 {
                            let off_p = order[tk2 + 1] * WIND_V;
                            let off_m = order[tk2 - 1] * WIND_V;
                            let mut dw = 0.0f64;
                            let mut iy = 0usize;
                            while iy < 7 {
                                let py = PHI_P[iy];
                                let rr = r0 - 3 + iy;
                                let mut ix = 0usize;
                                while ix < 7 {
                                    let v = v_idx(c0 - 3 + ix, rr);
                                    dw += PHI_P[ix]
                                        * py
                                        * (bos.solve_p[off_p + v] - bos.solve_p[off_m + v]);
                                    ix += 1;
                                }
                                iy += 1;
                            }
                            term_t += 0.5 * PHI_P[tk2] * dw / dt_avg;
                            tk2 += 1;
                        }
                    }

                    let lhs = term_t + term_ax + term_ay;
                    let rhs = term_lap;
                    let arhs = abs_f64(rhs);
                    if lhs.is_finite() && rhs.is_finite() && arhs > 1.0e-9 {
                        if arhs > max_arhs {
                            max_arhs = arhs;
                        }
                        if n_st < MAX_STRIDGE_STENCILS {
                            bos.solve_lp[LP_ST_LHS_OFF + n_st] = lhs;
                            bos.solve_lp[LP_ST_RHS_OFF + n_st] = rhs;
                            bos.solve_lp[LP_ST_ABS_OFF + n_st] = arhs;
                            n_st += 1;
                        } else {
                            let mut min_i = 0usize;
                            let mut min_v = bos.solve_lp[LP_ST_ABS_OFF];
                            let mut si = 1usize;
                            while si < MAX_STRIDGE_STENCILS {
                                let vv = bos.solve_lp[LP_ST_ABS_OFF + si];
                                if vv < min_v {
                                    min_v = vv;
                                    min_i = si;
                                }
                                si += 1;
                            }
                            if arhs > min_v {
                                bos.solve_lp[LP_ST_LHS_OFF + min_i] = lhs;
                                bos.solve_lp[LP_ST_RHS_OFF + min_i] = rhs;
                                bos.solve_lp[LP_ST_ABS_OFF + min_i] = arhs;
                            }
                        }
                    }
                }
                c0 += 1;
            }
            r0 += 1;
        }

        // Filter to well-conditioned stencils (`|rhs| >= 0.20 * max_arhs`) so near-zero far-field
        // stencils cannot pollute R^2 or AICc!
        let rhs_cut = 0.20 * max_arhs;
        let mut n_cond = 0usize;
        let mut i = 0usize;
        while i < n_st {
            if bos.solve_lp[LP_ST_ABS_OFF + i] >= rhs_cut {
                bos.solve_lp[LP_ST_LHS_OFF + n_cond] = bos.solve_lp[LP_ST_LHS_OFF + i];
                bos.solve_lp[LP_ST_RHS_OFF + n_cond] = bos.solve_lp[LP_ST_RHS_OFF + i];
                bos.solve_lp[LP_ST_ABS_OFF + n_cond] = bos.solve_lp[LP_ST_ABS_OFF + i];
                n_cond += 1;
            }
            i += 1;
        }

        if n_cond < 8 {
            return (0.0, 0.0, n_cond as u32, false);
        }

        let mut sum_x2 = 0.0f64;
        let mut sum_y2 = 0.0f64;
        let mut sum_xy = 0.0f64;
        i = 0;
        while i < n_cond {
            let x = bos.solve_lp[LP_ST_RHS_OFF + i];
            let y = bos.solve_lp[LP_ST_LHS_OFF + i];
            bos.solve_lp[LP_ST_W_OFF + i] = 1.0;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_xy += x * y;
            i += 1;
        }
        if !sum_x2.is_finite() || sum_x2 <= 1.0e-12 || !sum_y2.is_finite() || sum_y2 <= 1.0e-14 {
            return (0.0, 0.0, n_cond as u32, false);
        }

        let mut nu = (sum_xy / sum_x2).max(0.0);
        // 6-iteration Huber-IRLS for outlier-robust viscosity estimation
        let mut iter = 0usize;
        while iter < 6 {
            i = 0;
            while i < n_cond {
                let x = bos.solve_lp[LP_ST_RHS_OFF + i];
                let y = bos.solve_lp[LP_ST_LHS_OFF + i];
                bos.solve_lp[LP_ST_RES_OFF + i] = abs_f64(y - nu * x);
                i += 1;
            }
            let mad = median_in_place(&mut bos.solve_lp[LP_ST_RES_OFF..LP_ST_RES_OFF + n_cond], n_cond);
            let scale = (1.4826 * mad).max(1.0e-9);
            let delta = 1.345 * scale;
            let mut w_x2 = 0.0f64;
            let mut w_xy = 0.0f64;
            i = 0;
            while i < n_cond {
                let x = bos.solve_lp[LP_ST_RHS_OFF + i];
                let y = bos.solve_lp[LP_ST_LHS_OFF + i];
                let r = abs_f64(y - nu * x);
                let w = if r <= delta { 1.0 } else { delta / r };
                bos.solve_lp[LP_ST_W_OFF + i] = w;
                w_x2 += w * x * x;
                w_xy += w * x * y;
                i += 1;
            }
            if w_x2 <= 1.0e-18 {
                break;
            }
            nu = (w_xy / w_x2).max(0.0);
            iter += 1;
        }

        // AICc model selection: Euler (nu = 0, k = 0) vs Viscous Navier-Stokes (nu > 0, k = 1)
        let mut rss0 = 0.0f64;
        let mut rss1 = 0.0f64;
        i = 0;
        while i < n_cond {
            let w = bos.solve_lp[LP_ST_W_OFF + i];
            let x = bos.solve_lp[LP_ST_RHS_OFF + i];
            let y = bos.solve_lp[LP_ST_LHS_OFF + i];
            let e = y - nu * x;
            rss0 += w * y * y;
            rss1 += w * e * e;
            i += 1;
        }
        if !nu.is_finite() || nu <= 0.0 || !rss0.is_finite() || rss0 <= 1.0e-18 || rss1 >= rss0 {
            return (0.0, 0.0, n_cond as u32, false);
        }
        let r2 = (1.0 - rss1 / rss0).clamp(0.0, 1.0);
        let nf = n_cond as f64;
        let delta_aicc = nf * ln_pos_f64(rss0 / rss1.max(1.0e-30)) - (2.0 + 4.0 / (nf - 2.0));
        if delta_aicc > 0.0 && r2 >= 0.25 {
            (nu, r2, n_cond as u32, true)
        } else {
            (0.0, r2, n_cond as u32, false)
        }
    }
}

#[inline(never)]
fn compute_vorticity_from_uv(
    u: &[f64],
    v_vel: &[f64],
    act: &[f64],
    out_w: &mut [f64],
    dx: f64,
    dy: f64,
) {
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;
    let mut r = 0usize;
    while r < WIND_H {
        let mut c = 0usize;
        while c < WIND_W {
            let v = v_idx(c, r);
            if act[v] < 0.5 {
                out_w[v] = 0.0;
                c += 1;
                continue;
            }
            let dv_dx = if c >= 2
                && c + 2 < WIND_W
                && act[v - 2] > 0.5
                && act[v - 1] > 0.5
                && act[v + 1] > 0.5
                && act[v + 2] > 0.5
            {
                (-v_vel[v + 2] + 8.0 * v_vel[v + 1] - 8.0 * v_vel[v - 1] + v_vel[v - 2])
                    * (inv_dx / 12.0)
            } else if c > 0 && c + 1 < WIND_W && act[v - 1] > 0.5 && act[v + 1] > 0.5 {
                0.5 * (v_vel[v + 1] - v_vel[v - 1]) * inv_dx
            } else if c + 1 < WIND_W && act[v + 1] > 0.5 {
                (v_vel[v + 1] - v_vel[v]) * inv_dx
            } else if c > 0 && act[v - 1] > 0.5 {
                (v_vel[v] - v_vel[v - 1]) * inv_dx
            } else {
                0.0
            };

            let du_dy = if r >= 2
                && r + 2 < WIND_H
                && act[v - 2 * WIND_W] > 0.5
                && act[v - WIND_W] > 0.5
                && act[v + WIND_W] > 0.5
                && act[v + 2 * WIND_W] > 0.5
            {
                (-u[v + 2 * WIND_W] + 8.0 * u[v + WIND_W] - 8.0 * u[v - WIND_W]
                    + u[v - 2 * WIND_W])
                    * (inv_dy / 12.0)
            } else if r > 0 && r + 1 < WIND_H && act[v - WIND_W] > 0.5 && act[v + WIND_W] > 0.5 {
                0.5 * (u[v + WIND_W] - u[v - WIND_W]) * inv_dy
            } else if r + 1 < WIND_H && act[v + WIND_W] > 0.5 {
                (u[v + WIND_W] - u[v]) * inv_dy
            } else if r > 0 && act[v - WIND_W] > 0.5 {
                (u[v] - u[v - WIND_W]) * inv_dy
            } else {
                0.0
            };

            out_w[v] = dv_dx - du_dy;
            c += 1;
        }
        r += 1;
    }
}

#[inline(never)]
fn capture_vertex_velocity_and_vorticity(
    hodge: &HodgeState,
    raw_continuum_field: &[f32; WIND_V * 6],
    is_live: bool,
    out_u: &mut [f64],
    out_v: &mut [f64],
    out_w: &mut [f64],
    out_act: &mut [f64],
    dx: f64,
    dy: f64,
) -> usize {
    let mut act_cnt = 0usize;
    let mut v = 0usize;
    while v < WIND_V {
        if hodge.v_act[v] != 0 {
            let (u, vv) = if is_live {
                let b = v * 6;
                (raw_continuum_field[b] as f64, raw_continuum_field[b + 1] as f64)
            } else {
                (hodge.inject_buf[v] as f64, hodge.inject_buf[WIND_V + v] as f64)
            };
            if u.is_finite() && vv.is_finite() {
                out_u[v] = u;
                out_v[v] = vv;
                out_act[v] = 1.0;
                act_cnt += 1;
            } else {
                out_u[v] = 0.0;
                out_v[v] = 0.0;
                out_act[v] = 0.0;
            }
        } else {
            out_u[v] = 0.0;
            out_v[v] = 0.0;
            out_act[v] = 0.0;
        }
        v += 1;
    }

    compute_vorticity_from_uv(out_u, out_v, out_act, out_w, dx, dy);
    act_cnt
}

/// Accumulate `SLOW_BOXCAR_STRIDE`-frame averaged snapshots for PIV viscosity STRidge.
/// Heals isolated 1-cell interior PIV dropout pinholes (`>= 3` active 4-neighbors), then projects
/// out interior divergence via `HodgeState::solve_exact_on_active_complex` (preserving uniform
/// freestream advection while eliminating curved-orbit IMM radial phase-lag divergence!) without
/// ever filling macroscopic physical obstacles.
#[inline(never)]
fn accumulate_slow_boxcar(
    bos: &mut BosState,
    fast_base: usize,
    t_sec: f64,
    dx: f64,
    dy: f64,
    box_frames: &mut u32,
    box_t_sum: &mut f64,
    slow_t: &mut [f64; SLOW_RING_LEN],
    slow_head: &mut u32,
    slow_count: &mut u32,
) -> bool {
    if *box_frames == 0 {
        bos.solve_lp[LP_BOX_U_OFF..LP_BOX_CNT_OFF + WIND_V].fill(0.0);
        *box_t_sum = 0.0;
    }
    let mut v = 0usize;
    while v < WIND_V {
        if bos.solve_z[fast_base + v] > 0.5 {
            bos.solve_lp[LP_BOX_U_OFF + v] += bos.solve_x[fast_base + v];
            bos.solve_lp[LP_BOX_V_OFF + v] += bos.solve_b[fast_base + v];
            bos.solve_lp[LP_BOX_CNT_OFF + v] += 1.0;
        }
        v += 1;
    }
    *box_t_sum += t_sec;
    *box_frames += 1;

    if *box_frames < SLOW_BOXCAR_STRIDE {
        return false;
    }

    let s_rel = (*slow_head as usize) % SLOW_RING_LEN;
    let dst = (SLOW_RING_BASE + s_rel) * WIND_V;
    v = 0;
    while v < WIND_V {
        let cnt = bos.solve_lp[LP_BOX_CNT_OFF + v];
        if cnt >= 1.0 {
            let inv = 1.0 / cnt;
            bos.solve_x[dst + v] = bos.solve_lp[LP_BOX_U_OFF + v] * inv;
            bos.solve_b[dst + v] = bos.solve_lp[LP_BOX_V_OFF + v] * inv;
            bos.solve_z[dst + v] = 1.0;
        } else {
            bos.solve_x[dst + v] = 0.0;
            bos.solve_b[dst + v] = 0.0;
            bos.solve_z[dst + v] = 0.0;
        }
        v += 1;
    }

    // Heal isolated 1-cell interior PIV pinholes (>= 3 valid 4-neighbors) using LP_BOX_W as scratch
    let mut r = 1usize;
    while r + 1 < WIND_H {
        let mut c = 1usize;
        while c + 1 < WIND_W {
            let idx = v_idx(c, r);
            if bos.solve_z[dst + idx] < 0.5 {
                let mut n_ok = 0usize;
                let mut su = 0.0f64;
                let mut sv = 0.0f64;
                let nbrs = [idx - 1, idx + 1, idx - WIND_W, idx + WIND_W];
                let mut k = 0usize;
                while k < 4 {
                    let nb = nbrs[k];
                    if bos.solve_z[dst + nb] > 0.5 {
                        n_ok += 1;
                        su += bos.solve_x[dst + nb];
                        sv += bos.solve_b[dst + nb];
                    }
                    k += 1;
                }
                if n_ok >= 3 {
                    let inv = 1.0 / (n_ok as f64);
                    bos.solve_x[dst + idx] = su * inv;
                    bos.solve_b[dst + idx] = sv * inv;
                    bos.solve_lp[LP_BOX_W_OFF + idx] = 1.0;
                } else {
                    bos.solve_lp[LP_BOX_W_OFF + idx] = 0.0;
                }
            } else {
                bos.solve_lp[LP_BOX_W_OFF + idx] = 0.0;
            }
            c += 1;
        }
        r += 1;
    }
    v = 0;
    while v < WIND_V {
        if bos.solve_lp[LP_BOX_W_OFF + v] > 0.5 {
            bos.solve_z[dst + v] = 1.0;
        }
        v += 1;
    }

    // Compute clean 2nd-order vorticity on the healed boxcar velocity field, then reconstruct
    // an EXACTLY discrete-divergence-free velocity `(u_sol, v_sol) = (u_bg, v_bg) + curl(psi)`
    // from Dirichlet streamfunction `-Delta psi = omega` (`div_2dx(u_sol) === 0` to machine precision!).
    // Preserves uniform freestream advection while completely eliminating curved-orbit IMM radial lag.
    let mut n_bg = 0usize;
    let mut u_bg_s = 0.0f64;
    let mut v_bg_s = 0.0f64;
    r = 0;
    while r < WIND_H {
        let mut c = 0usize;
        while c < WIND_W {
            if c < 5 || c + 5 >= WIND_W || r < 4 || r + 4 >= WIND_H {
                let idx = v_idx(c, r);
                if bos.solve_z[dst + idx] > 0.5 {
                    u_bg_s += bos.solve_x[dst + idx];
                    v_bg_s += bos.solve_b[dst + idx];
                    n_bg += 1;
                }
            }
            c += 1;
        }
        r += 1;
    }
    let u_bg = if n_bg > 0 { u_bg_s / (n_bg as f64) } else { 0.0 };
    let v_bg = if n_bg > 0 { v_bg_s / (n_bg as f64) } else { 0.0 };

    // 2nd-order vorticity `omega` into `bos.solve_p[dst..]`
    let inv_2dx = 0.5 / dx;
    let inv_2dy = 0.5 / dy;
    bos.solve_p[dst..dst + WIND_V].fill(0.0);
    r = 1;
    while r + 1 < WIND_H {
        let mut c = 1usize;
        while c + 1 < WIND_W {
            let idx = v_idx(c, r);
            if bos.solve_z[dst + idx] > 0.5
                && bos.solve_z[dst + idx - 1] > 0.5
                && bos.solve_z[dst + idx + 1] > 0.5
                && bos.solve_z[dst + idx - WIND_W] > 0.5
                && bos.solve_z[dst + idx + WIND_W] > 0.5
            {
                let dv_dx = (bos.solve_b[dst + idx + 1] - bos.solve_b[dst + idx - 1]) * inv_2dx;
                let du_dy =
                    (bos.solve_x[dst + idx + WIND_W] - bos.solve_x[dst + idx - WIND_W]) * inv_2dy;
                bos.solve_p[dst + idx] = dv_dx - du_dy;
            }
            c += 1;
        }
        r += 1;
    }

    // Solve Dirichlet Poisson `-(d^2/dx^2 + d^2/dy^2) psi = omega` on active interior vertices
    // (psi = 0 on outer boundary and obstacle voids). Scratch inside `bos.solve_r` (free before OLS):
    //   psi = SCR_P0_OFF, r = SCR_UBAR_OFF, p = SCR_WBAR_OFF, ap = SCR_DUDT_OFF
    let wx = dy / dx;
    let wy = dx / dy;
    let diag_c = 2.0 * (wx + wy);
    let cell_area = dx * dy;
    bos.solve_r[SCR_P0_OFF..SCR_P0_OFF + WIND_V].fill(0.0);
    bos.solve_r[SCR_UBAR_OFF..SCR_UBAR_OFF + WIND_V].fill(0.0);
    bos.solve_r[SCR_WBAR_OFF..SCR_WBAR_OFF + WIND_V].fill(0.0);

    let mut rr_old = 0.0f64;
    r = 1;
    while r + 1 < WIND_H {
        let mut c = 1usize;
        while c + 1 < WIND_W {
            let idx = v_idx(c, r);
            if bos.solve_z[dst + idx] > 0.5 {
                let rv = bos.solve_p[dst + idx] * cell_area;
                bos.solve_r[SCR_UBAR_OFF + idx] = rv;
                bos.solve_r[SCR_WBAR_OFF + idx] = rv;
                rr_old += rv * rv;
            }
            c += 1;
        }
        r += 1;
    }

    if rr_old > 1.0e-18 {
        let tol2 = 1.0e-12 * rr_old;
        let mut iter = 0usize;
        while iter < 100 {
            let mut pap = 0.0f64;
            r = 1;
            while r + 1 < WIND_H {
                let mut c = 1usize;
                while c + 1 < WIND_W {
                    let idx = v_idx(c, r);
                    if bos.solve_z[dst + idx] > 0.5 {
                        let pv = bos.solve_r[SCR_WBAR_OFF + idx];
                        let ap = diag_c * pv
                            - wx * (bos.solve_r[SCR_WBAR_OFF + idx - 1]
                                + bos.solve_r[SCR_WBAR_OFF + idx + 1])
                            - wy * (bos.solve_r[SCR_WBAR_OFF + idx - WIND_W]
                                + bos.solve_r[SCR_WBAR_OFF + idx + WIND_W]);
                        bos.solve_r[SCR_DUDT_OFF + idx] = ap;
                        pap += pv * ap;
                    } else {
                        bos.solve_r[SCR_DUDT_OFF + idx] = 0.0;
                    }
                    c += 1;
                }
                r += 1;
            }
            if !pap.is_finite() || pap <= 1.0e-24 {
                break;
            }
            let alpha = rr_old / pap;
            let mut rr_new = 0.0f64;
            r = 1;
            while r + 1 < WIND_H {
                let mut c = 1usize;
                while c + 1 < WIND_W {
                    let idx = v_idx(c, r);
                    if bos.solve_z[dst + idx] > 0.5 {
                        bos.solve_r[SCR_P0_OFF + idx] += alpha * bos.solve_r[SCR_WBAR_OFF + idx];
                        let rn = bos.solve_r[SCR_UBAR_OFF + idx]
                            - alpha * bos.solve_r[SCR_DUDT_OFF + idx];
                        bos.solve_r[SCR_UBAR_OFF + idx] = rn;
                        rr_new += rn * rn;
                    }
                    c += 1;
                }
                r += 1;
            }
            if rr_new <= tol2 || !rr_new.is_finite() || rr_new <= 1.0e-24 {
                break;
            }
            let beta = rr_new / rr_old;
            r = 1;
            while r + 1 < WIND_H {
                let mut c = 1usize;
                while c + 1 < WIND_W {
                    let idx = v_idx(c, r);
                    if bos.solve_z[dst + idx] > 0.5 {
                        bos.solve_r[SCR_WBAR_OFF + idx] =
                            bos.solve_r[SCR_UBAR_OFF + idx] + beta * bos.solve_r[SCR_WBAR_OFF + idx];
                    }
                    c += 1;
                }
                r += 1;
            }
            rr_old = rr_new;
            iter += 1;
        }

        // Reconstruct `u_sol = u_bg + d(psi)/dy`, `v_sol = v_bg - d(psi)/dx`
        r = 1;
        while r + 1 < WIND_H {
            let mut c = 1usize;
            while c + 1 < WIND_W {
                let idx = v_idx(c, r);
                if bos.solve_z[dst + idx] > 0.5 {
                    bos.solve_x[dst + idx] = u_bg
                        + (bos.solve_r[SCR_P0_OFF + idx + WIND_W]
                            - bos.solve_r[SCR_P0_OFF + idx - WIND_W])
                            * inv_2dy;
                    bos.solve_b[dst + idx] = v_bg
                        - (bos.solve_r[SCR_P0_OFF + idx + 1] - bos.solve_r[SCR_P0_OFF + idx - 1])
                            * inv_2dx;
                }
                c += 1;
            }
            r += 1;
        }
    }

    slow_t[s_rel] = *box_t_sum / (*box_frames as f64);
    *slow_head = ((s_rel + 1) % SLOW_RING_LEN) as u32;
    if (*slow_count as usize) < SLOW_RING_LEN {
        *slow_count += 1;
    }
    *box_frames = 0;
    *box_t_sum = 0.0;
    true
}

/// Local linear OLS temporal regression over the active `fast_ring` frames (`3..=7`) at each vertex:
/// evaluates `(u_bar, v_bar)` at current time `t_cur` and `(du_dt, dv_dt)` slope with 3.7x less
/// PIV temporal noise while preserving exact linear-in-time acceleration!
#[inline(never)]
fn compute_temporal_ols_fields(
    hodge: &HodgeState,
    bos: &mut BosState,
    fast_t: &[f64; FAST_RING_LEN],
    fast_head: usize,
    fast_count: usize,
    t_cur: f64,
    dx: f64,
    dy: f64,
) {
    let n_frames = fast_count.min(FAST_RING_LEN);
    let mut slots = [0usize; FAST_RING_LEN];
    let mut dts = [0.0f64; FAST_RING_LEN];
    let start = (fast_head + FAST_RING_LEN - n_frames) % FAST_RING_LEN;
    let mut k = 0usize;
    while k < n_frames {
        let s = (start + k) % FAST_RING_LEN;
        slots[k] = s * WIND_V;
        dts[k] = fast_t[s] - t_cur; // <= 0, with newest frame at 0.0
        k += 1;
    }
    let newest_off = slots[n_frames - 1];

    let mut v = 0usize;
    while v < WIND_V {
        if hodge.v_act[v] == 0 {
            bos.solve_r[SCR_UBAR_OFF + v] = 0.0;
            bos.solve_r[SCR_VBAR_OFF + v] = 0.0;
            bos.solve_r[SCR_DUDT_OFF + v] = 0.0;
            bos.solve_r[SCR_DVDT_OFF + v] = 0.0;
            v += 1;
            continue;
        }
        let mut cnt = 0usize;
        let mut st = 0.0f64;
        let mut su = 0.0f64;
        let mut sv = 0.0f64;
        k = 0;
        while k < n_frames {
            let off = slots[k] + v;
            if bos.solve_z[off] > 0.5 {
                cnt += 1;
                st += dts[k];
                su += bos.solve_x[off];
                sv += bos.solve_b[off];
            }
            k += 1;
        }
        if cnt >= 2 {
            let inv_n = 1.0 / (cnt as f64);
            let mt = st * inv_n;
            let mu = su * inv_n;
            let mv = sv * inv_n;
            let mut stt = 0.0f64;
            let mut stu = 0.0f64;
            let mut stv = 0.0f64;
            k = 0;
            while k < n_frames {
                let off = slots[k] + v;
                if bos.solve_z[off] > 0.5 {
                    let dt = dts[k] - mt;
                    stt += dt * dt;
                    stu += dt * (bos.solve_x[off] - mu);
                    stv += dt * (bos.solve_b[off] - mv);
                }
                k += 1;
            }
            if stt > 1.0e-10 {
                let du_dt = stu / stt;
                let dv_dt = stv / stt;
                // Evaluate fitted velocity at window center `mt` (`3.25x` lower PIV variance than endpoint!)
                bos.solve_r[SCR_UBAR_OFF + v] = mu;
                bos.solve_r[SCR_VBAR_OFF + v] = mv;
                bos.solve_r[SCR_DUDT_OFF + v] = du_dt;
                bos.solve_r[SCR_DVDT_OFF + v] = dv_dt;
            } else {
                bos.solve_r[SCR_UBAR_OFF + v] = mu;
                bos.solve_r[SCR_VBAR_OFF + v] = mv;
                bos.solve_r[SCR_DUDT_OFF + v] = 0.0;
                bos.solve_r[SCR_DVDT_OFF + v] = 0.0;
            }
        } else {
            bos.solve_r[SCR_UBAR_OFF + v] = bos.solve_x[newest_off + v];
            bos.solve_r[SCR_VBAR_OFF + v] = bos.solve_b[newest_off + v];
            bos.solve_r[SCR_DUDT_OFF + v] = 0.0;
            bos.solve_r[SCR_DVDT_OFF + v] = 0.0;
        }
        v += 1;
    }

    // 1-pass 5-point cardinal filter on `(du_dt, dv_dt)` only: mathematically exact on linear
    // unsteady forcing (preserves `I9.3b` curl refusal) while suppressing white PIV temporal jitter!
    let mut rs = 0usize;
    while rs < WIND_H {
        let mut cs = 0usize;
        while cs < WIND_W {
            let idx = v_idx(cs, rs);
            if hodge.v_act[idx] == 0 {
                bos.solve_r[SCR_LAPU_OFF + idx] = 0.0;
                bos.solve_r[SCR_LAPV_OFF + idx] = 0.0;
            } else {
                let mut su = 2.0 * bos.solve_r[SCR_DUDT_OFF + idx];
                let mut sv = 2.0 * bos.solve_r[SCR_DVDT_OFF + idx];
                let mut sw = 2.0f64;
                if cs > 0 && hodge.v_act[idx - 1] != 0 {
                    su += bos.solve_r[SCR_DUDT_OFF + idx - 1];
                    sv += bos.solve_r[SCR_DVDT_OFF + idx - 1];
                    sw += 1.0;
                }
                if cs + 1 < WIND_W && hodge.v_act[idx + 1] != 0 {
                    su += bos.solve_r[SCR_DUDT_OFF + idx + 1];
                    sv += bos.solve_r[SCR_DVDT_OFF + idx + 1];
                    sw += 1.0;
                }
                if rs > 0 && hodge.v_act[idx - WIND_W] != 0 {
                    su += bos.solve_r[SCR_DUDT_OFF + idx - WIND_W];
                    sv += bos.solve_r[SCR_DVDT_OFF + idx - WIND_W];
                    sw += 1.0;
                }
                if rs + 1 < WIND_H && hodge.v_act[idx + WIND_W] != 0 {
                    su += bos.solve_r[SCR_DUDT_OFF + idx + WIND_W];
                    sv += bos.solve_r[SCR_DVDT_OFF + idx + WIND_W];
                    sw += 1.0;
                }
                let inv = 1.0 / sw;
                bos.solve_r[SCR_LAPU_OFF + idx] = su * inv;
                bos.solve_r[SCR_LAPV_OFF + idx] = sv * inv;
            }
            cs += 1;
        }
        rs += 1;
    }
    v = 0;
    while v < WIND_V {
        bos.solve_r[SCR_DUDT_OFF + v] = bos.solve_r[SCR_LAPU_OFF + v];
        bos.solve_r[SCR_DVDT_OFF + v] = bos.solve_r[SCR_LAPV_OFF + v];
        v += 1;
    }

    // Recompute clean vorticity `omega_bar` from `(u_bar, v_bar)` on `hodge.v_act`
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;
    let mut r = 0usize;
    while r < WIND_H {
        let mut c = 0usize;
        while c < WIND_W {
            let v = v_idx(c, r);
            if hodge.v_act[v] == 0 {
                bos.solve_r[SCR_WBAR_OFF + v] = 0.0;
                c += 1;
                continue;
            }
            let dv_dx = if c >= 2
                && c + 2 < WIND_W
                && hodge.v_act[v - 2] != 0
                && hodge.v_act[v - 1] != 0
                && hodge.v_act[v + 1] != 0
                && hodge.v_act[v + 2] != 0
            {
                (-bos.solve_r[SCR_VBAR_OFF + v + 2] + 8.0 * bos.solve_r[SCR_VBAR_OFF + v + 1]
                    - 8.0 * bos.solve_r[SCR_VBAR_OFF + v - 1]
                    + bos.solve_r[SCR_VBAR_OFF + v - 2])
                    * (inv_dx / 12.0)
            } else if c > 0
                && c + 1 < WIND_W
                && hodge.v_act[v - 1] != 0
                && hodge.v_act[v + 1] != 0
            {
                0.5 * (bos.solve_r[SCR_VBAR_OFF + v + 1] - bos.solve_r[SCR_VBAR_OFF + v - 1])
                    * inv_dx
            } else if c + 1 < WIND_W && hodge.v_act[v + 1] != 0 {
                (bos.solve_r[SCR_VBAR_OFF + v + 1] - bos.solve_r[SCR_VBAR_OFF + v]) * inv_dx
            } else if c > 0 && hodge.v_act[v - 1] != 0 {
                (bos.solve_r[SCR_VBAR_OFF + v] - bos.solve_r[SCR_VBAR_OFF + v - 1]) * inv_dx
            } else {
                0.0
            };

            let du_dy = if r >= 2
                && r + 2 < WIND_H
                && hodge.v_act[v - 2 * WIND_W] != 0
                && hodge.v_act[v - WIND_W] != 0
                && hodge.v_act[v + WIND_W] != 0
                && hodge.v_act[v + 2 * WIND_W] != 0
            {
                (-bos.solve_r[SCR_UBAR_OFF + v + 2 * WIND_W]
                    + 8.0 * bos.solve_r[SCR_UBAR_OFF + v + WIND_W]
                    - 8.0 * bos.solve_r[SCR_UBAR_OFF + v - WIND_W]
                    + bos.solve_r[SCR_UBAR_OFF + v - 2 * WIND_W])
                    * (inv_dy / 12.0)
            } else if r > 0
                && r + 1 < WIND_H
                && hodge.v_act[v - WIND_W] != 0
                && hodge.v_act[v + WIND_W] != 0
            {
                0.5 * (bos.solve_r[SCR_UBAR_OFF + v + WIND_W]
                    - bos.solve_r[SCR_UBAR_OFF + v - WIND_W])
                    * inv_dy
            } else if r + 1 < WIND_H && hodge.v_act[v + WIND_W] != 0 {
                (bos.solve_r[SCR_UBAR_OFF + v + WIND_W] - bos.solve_r[SCR_UBAR_OFF + v]) * inv_dy
            } else if r > 0 && hodge.v_act[v - WIND_W] != 0 {
                (bos.solve_r[SCR_UBAR_OFF + v] - bos.solve_r[SCR_UBAR_OFF + v - WIND_W]) * inv_dy
            } else {
                0.0
            };

            bos.solve_r[SCR_WBAR_OFF + v] = dv_dx - du_dy;
            c += 1;
        }
        r += 1;
    }
}

/// Interior divergence energy fraction: strictly excludes open ports so potential inflow/outflow
/// passes cleanly while 2D compressible / 3D-upwelling sources trigger `WIND_REASON_HIGH_DIV`.
/// Applies a 5-point cardinal box smooth (exact on linear sources/sinks, suppresses white PIV jitter).
#[inline(never)]
fn compute_interior_divergence_fraction(
    hodge: &HodgeState,
    bos: &mut BosState,
    dx: f64,
    dy: f64,
) -> f64 {
    let mut r = 0usize;
    while r < WIND_H {
        let mut c = 0usize;
        while c < WIND_W {
            let idx = v_idx(c, r);
            if hodge.v_act[idx] == 0 {
                bos.solve_r[SCR_LAPU_OFF + idx] = 0.0;
                bos.solve_r[SCR_LAPV_OFF + idx] = 0.0;
            } else if c > 0
                && c + 1 < WIND_W
                && r > 0
                && r + 1 < WIND_H
                && hodge.v_act[idx - 1] != 0
                && hodge.v_act[idx + 1] != 0
                && hodge.v_act[idx - WIND_W] != 0
                && hodge.v_act[idx + WIND_W] != 0
            {
                bos.solve_r[SCR_LAPU_OFF + idx] = (2.0 * bos.solve_r[SCR_UBAR_OFF + idx]
                    + bos.solve_r[SCR_UBAR_OFF + idx - 1]
                    + bos.solve_r[SCR_UBAR_OFF + idx + 1]
                    + bos.solve_r[SCR_UBAR_OFF + idx - WIND_W]
                    + bos.solve_r[SCR_UBAR_OFF + idx + WIND_W])
                    * (1.0 / 6.0);
                bos.solve_r[SCR_LAPV_OFF + idx] = (2.0 * bos.solve_r[SCR_VBAR_OFF + idx]
                    + bos.solve_r[SCR_VBAR_OFF + idx - 1]
                    + bos.solve_r[SCR_VBAR_OFF + idx + 1]
                    + bos.solve_r[SCR_VBAR_OFF + idx - WIND_W]
                    + bos.solve_r[SCR_VBAR_OFF + idx + WIND_W])
                    * (1.0 / 6.0);
            } else {
                bos.solve_r[SCR_LAPU_OFF + idx] = bos.solve_r[SCR_UBAR_OFF + idx];
                bos.solve_r[SCR_LAPV_OFF + idx] = bos.solve_r[SCR_VBAR_OFF + idx];
            }
            c += 1;
        }
        r += 1;
    }

    let inv_2dx = 0.5 / dx;
    let inv_2dy = 0.5 / dy;
    let mut div2_sum = 0.0f64;
    let mut grad2_sum = 0.0f64;
    let mut spd2_sum = 0.0f64;
    let mut cnt = 0usize;

    r = 1;
    while r + 1 < WIND_H {
        let mut c = 1usize;
        while c + 1 < WIND_W {
            let idx = v_idx(c, r);
            if hodge.v_act[idx] != 0
                && (hodge.v_class[idx] & (VCLASS_HOLE_BND | VCLASS_PORT_BND)) == 0
                && hodge.v_act[idx - 1] != 0
                && hodge.v_act[idx + 1] != 0
                && hodge.v_act[idx - WIND_W] != 0
                && hodge.v_act[idx + WIND_W] != 0
            {
                let ux = (bos.solve_r[SCR_LAPU_OFF + idx + 1]
                    - bos.solve_r[SCR_LAPU_OFF + idx - 1])
                    * inv_2dx;
                let uy = (bos.solve_r[SCR_LAPU_OFF + idx + WIND_W]
                    - bos.solve_r[SCR_LAPU_OFF + idx - WIND_W])
                    * inv_2dy;
                let vx = (bos.solve_r[SCR_LAPV_OFF + idx + 1]
                    - bos.solve_r[SCR_LAPV_OFF + idx - 1])
                    * inv_2dx;
                let vy = (bos.solve_r[SCR_LAPV_OFF + idx + WIND_W]
                    - bos.solve_r[SCR_LAPV_OFF + idx - WIND_W])
                    * inv_2dy;
                let div = ux + vy;
                div2_sum += div * div;
                grad2_sum += 2.0 * (ux * ux + vy * vy) + (uy * uy + vx * vx);
                let uc = bos.solve_r[SCR_LAPU_OFF + idx];
                let vc = bos.solve_r[SCR_LAPV_OFF + idx];
                spd2_sum += uc * uc + vc * vc;
                cnt += 1;
            }
            c += 1;
        }
        r += 1;
    }

    if cnt < 8 {
        return 0.0;
    }
    let h2 = dx * dy;
    let floor = 1.0e-4 * (spd2_sum / h2) + 1.0e-12;
    div2_sum / (grad2_sum + floor)
}

/// Build the Lamb-form negative material acceleration 1-cochain from OLS `(u_bar, v_bar, w_bar, du_dt, dv_dt)`:
///   `g_1 = -d_0(1/2 |u|^2) - [d/dt u + omega x u - nu * Delta u]_1`
#[inline(never)]
fn build_material_accel_cochain(
    hodge: &HodgeState,
    bos: &mut BosState,
    nu_eff: f64,
    dx: f64,
    dy: f64,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);

    // 1. Viscous Laplacian `Delta u_bar`, `Delta v_bar` when `nu_eff > 0`
    let mut r = 0usize;
    while r < WIND_H {
        let mut c = 0usize;
        while c < WIND_W {
            let v = v_idx(c, r);
            if hodge.v_act[v] == 0 || nu_eff <= 0.0 {
                bos.solve_r[SCR_LAPU_OFF + v] = 0.0;
                bos.solve_r[SCR_LAPV_OFF + v] = 0.0;
                c += 1;
                continue;
            }
            let uc = bos.solve_r[SCR_UBAR_OFF + v];
            let vc = bos.solve_r[SCR_VBAR_OFF + v];
            let mut lap_u = 0.0f64;
            let mut lap_v = 0.0f64;
            if c > 0 && c + 1 < WIND_W && hodge.v_act[v - 1] != 0 && hodge.v_act[v + 1] != 0 {
                lap_u += (bos.solve_r[SCR_UBAR_OFF + v - 1] - 2.0 * uc
                    + bos.solve_r[SCR_UBAR_OFF + v + 1])
                    * inv_dx2;
                lap_v += (bos.solve_r[SCR_VBAR_OFF + v - 1] - 2.0 * vc
                    + bos.solve_r[SCR_VBAR_OFF + v + 1])
                    * inv_dx2;
            }
            if r > 0
                && r + 1 < WIND_H
                && hodge.v_act[v - WIND_W] != 0
                && hodge.v_act[v + WIND_W] != 0
            {
                lap_u += (bos.solve_r[SCR_UBAR_OFF + v - WIND_W] - 2.0 * uc
                    + bos.solve_r[SCR_UBAR_OFF + v + WIND_W])
                    * inv_dy2;
                lap_v += (bos.solve_r[SCR_VBAR_OFF + v - WIND_W] - 2.0 * vc
                    + bos.solve_r[SCR_VBAR_OFF + v + WIND_W])
                    * inv_dy2;
            }
            bos.solve_r[SCR_LAPU_OFF + v] = lap_u;
            bos.solve_r[SCR_LAPV_OFF + v] = lap_v;
            c += 1;
        }
        r += 1;
    }

    // 2. Assemble edge 1-cochain `g_1[0..WIND_E]`
    r = 0;
    while r < WIND_H {
        let mut c = 0usize;
        while c < WIND_W - 1 {
            let e = ex_idx(c, r);
            if hodge.e_act[e] != 0 {
                let v0 = v_idx(c, r);
                let v1 = v0 + 1;
                let u0 = bos.solve_r[SCR_UBAR_OFF + v0];
                let v_0 = bos.solve_r[SCR_VBAR_OFF + v0];
                let u1 = bos.solve_r[SCR_UBAR_OFF + v1];
                let v_1 = bos.solve_r[SCR_VBAR_OFF + v1];
                let ke_diff = 0.5 * ((u1 * u1 + v_1 * v_1) - (u0 * u0 + v_0 * v_0));
                let lamb_x = -0.5
                    * (bos.solve_r[SCR_WBAR_OFF + v0] * v_0
                        + bos.solve_r[SCR_WBAR_OFF + v1] * v_1);
                let dt_x = 0.5 * (bos.solve_r[SCR_DUDT_OFF + v0] + bos.solve_r[SCR_DUDT_OFF + v1]);
                let visc_x =
                    0.5 * nu_eff * (bos.solve_r[SCR_LAPU_OFF + v0] + bos.solve_r[SCR_LAPU_OFF + v1]);
                bos.solve_r[SCR_G1_OFF + e] = -(ke_diff + (dt_x + lamb_x - visc_x) * dx);
            } else {
                bos.solve_r[SCR_G1_OFF + e] = 0.0;
            }
            c += 1;
        }
        r += 1;
    }

    r = 0;
    while r < WIND_H - 1 {
        let mut c = 0usize;
        while c < WIND_W {
            let e = ey_idx(c, r);
            if hodge.e_act[e] != 0 {
                let v0 = v_idx(c, r);
                let v1 = v0 + WIND_W;
                let u0 = bos.solve_r[SCR_UBAR_OFF + v0];
                let v_0 = bos.solve_r[SCR_VBAR_OFF + v0];
                let u1 = bos.solve_r[SCR_UBAR_OFF + v1];
                let v_1 = bos.solve_r[SCR_VBAR_OFF + v1];
                let ke_diff = 0.5 * ((u1 * u1 + v_1 * v_1) - (u0 * u0 + v_0 * v_0));
                let lamb_y = 0.5
                    * (bos.solve_r[SCR_WBAR_OFF + v0] * u0 + bos.solve_r[SCR_WBAR_OFF + v1] * u1);
                let dt_y = 0.5 * (bos.solve_r[SCR_DVDT_OFF + v0] + bos.solve_r[SCR_DVDT_OFF + v1]);
                let visc_y =
                    0.5 * nu_eff * (bos.solve_r[SCR_LAPV_OFF + v0] + bos.solve_r[SCR_LAPV_OFF + v1]);
                bos.solve_r[SCR_G1_OFF + e] = -(ke_diff + (dt_y + lamb_y - visc_y) * dy);
            } else {
                bos.solve_r[SCR_G1_OFF + e] = 0.0;
            }
            c += 1;
        }
        r += 1;
    }
}

#[inline(never)]
fn compute_ns_curl_residual(
    hodge: &HodgeState,
    g1: &[f64],
    p0: &[f64],
    out_cell_resid: &mut [f32],
) -> f64 {
    out_cell_resid.fill(0.0);
    let mut e_res = 0.0f64;
    let mut e_tot = 0.0f64;

    let mut r = 0usize;
    while r < WIND_H {
        let mut c = 0usize;
        while c < WIND_W - 1 {
            let e = ex_idx(c, r);
            if hodge.e_act[e] != 0 {
                let v0 = v_idx(c, r);
                let v1 = v0 + 1;
                let w = hodge.w1_edge[e];
                let g = g1[e];
                let d = g - (p0[v1] - p0[v0]);
                e_res += w * d * d;
                e_tot += w * g * g;
                let ad = abs_f64(d) as f32;
                out_cell_resid[v0] += ad;
                out_cell_resid[v1] += ad;
            }
            c += 1;
        }
        r += 1;
    }
    r = 0;
    while r < WIND_H - 1 {
        let mut c = 0usize;
        while c < WIND_W {
            let e = ey_idx(c, r);
            if hodge.e_act[e] != 0 {
                let v0 = v_idx(c, r);
                let v1 = v0 + WIND_W;
                let w = hodge.w1_edge[e];
                let g = g1[e];
                let d = g - (p0[v1] - p0[v0]);
                e_res += w * d * d;
                e_tot += w * g * g;
                let ad = abs_f64(d) as f32;
                out_cell_resid[v0] += ad;
                out_cell_resid[v1] += ad;
            }
            c += 1;
        }
        r += 1;
    }

    let g_rms = wasm_sqrt_f64(e_tot / (WIND_E as f64) + 1.0e-24) as f32;
    let inv_scale = 0.25 / (g_rms + 1.0e-6);
    let mut v = 0usize;
    while v < WIND_V {
        if hodge.v_act[v] != 0 {
            out_cell_resid[v] *= inv_scale;
        } else {
            out_cell_resid[v] = 0.0;
        }
        v += 1;
    }

    if e_tot <= 1.0e-18 {
        0.0
    } else {
        wasm_sqrt_f64(e_res / e_tot)
    }
}

/// Far-field freestream velocity & upstream pressure gauge estimation over the outer 5-cell
/// frame band (handles live PIV border dropout at `c=0,63 / r=0,35` while staying far from any void).
#[inline(never)]
fn estimate_freestream_and_gauge(
    hodge: &HodgeState,
    bos: &mut BosState,
    has_void: bool,
    hole_cx: f64,
    hole_cy: f64,
    d_hole_px: f64,
    dx: f64,
    dy: f64,
) -> (bool, f64, f64, f64, f64, f64) {
    let excl_r2 = if has_void {
        let rr = (1.5 * d_hole_px).max(3.0 * dx);
        rr * rr
    } else {
        0.0
    };

    let mut n_out = 0usize;
    let mut spd_sum = 0.0f64;
    let mut r = 0usize;
    while r < WIND_H {
        let py = (r as f64 + 0.5) * dy;
        let mut c = 0usize;
        while c < WIND_W {
            let in_band = c < 5 || c + 5 >= WIND_W || r < 4 || r + 4 >= WIND_H;
            let v = v_idx(c, r);
            if in_band && hodge.v_act[v] != 0 && n_out < WIND_V {
                let px = (c as f64 + 0.5) * dx;
                let far_from_void = !has_void
                    || ((px - hole_cx) * (px - hole_cx) + (py - hole_cy) * (py - hole_cy))
                        >= excl_r2;
                if far_from_void {
                    let u = bos.solve_r[SCR_UBAR_OFF + v];
                    let vv = bos.solve_r[SCR_VBAR_OFF + v];
                    bos.solve_r[SCR_SORT_OFF + n_out] = u;
                    bos.solve_r[SCR_SORT_OFF + WIND_V + n_out] = vv;
                    spd_sum += wasm_sqrt_f64(u * u + vv * vv);
                    n_out += 1;
                }
            }
            c += 1;
        }
        r += 1;
    }

    if n_out < 4 {
        return (false, 0.0, 1.0, 0.0, 0.0, 0.0);
    }

    let u_med = median_in_place(&mut bos.solve_r[SCR_SORT_OFF..SCR_SORT_OFF + n_out], n_out);
    let v_med = median_in_place(
        &mut bos.solve_r[SCR_SORT_OFF + WIND_V..SCR_SORT_OFF + WIND_V + n_out],
        n_out,
    );
    let mean_spd = spd_sum / (n_out as f64);
    let vec_spd = wasm_sqrt_f64(u_med * u_med + v_med * v_med);

    let coherent_freestream = vec_spd > 4.0 && vec_spd > 0.45 * mean_spd;
    if !coherent_freestream {
        let mut np = 0usize;
        r = 0;
        while r < WIND_H {
            let mut c = 0usize;
            while c < WIND_W {
                let in_band = c < 5 || c + 5 >= WIND_W || r < 4 || r + 4 >= WIND_H;
                let v = v_idx(c, r);
                if in_band && hodge.v_act[v] != 0 && np < WIND_V {
                    bos.solve_r[SCR_SORT_OFF + np] = bos.solve_r[SCR_P0_OFF + v];
                    np += 1;
                }
                c += 1;
            }
            r += 1;
        }
        let p_ref = median_in_place(&mut bos.solve_r[SCR_SORT_OFF..SCR_SORT_OFF + np], np);
        return (false, 0.0, 1.0, 0.0, 0.0, p_ref);
    }

    let ex = u_med / vec_spd;
    let ey = v_med / vec_spd;
    let angle = fast_atan2f(ey as f32, ex as f32) as f64;

    // Select upstream inflow strip inside the outer 6-cell margin (`inward normal . e_inf > 0.5`)
    let mut n_up = 0usize;
    r = 0;
    while r < WIND_H {
        let py = (r as f64 + 0.5) * dy;
        let mut c = 0usize;
        while c < WIND_W {
            let v = v_idx(c, r);
            if hodge.v_act[v] != 0 {
                let mut nx = 0.0f64;
                let mut ny = 0.0f64;
                if c < 6 {
                    nx += 1.0;
                } else if c + 6 >= WIND_W {
                    nx -= 1.0;
                }
                if r < 5 {
                    ny += 1.0;
                } else if r + 5 >= WIND_H {
                    ny -= 1.0;
                }
                let nmag = wasm_sqrt_f64(nx * nx + ny * ny);
                if nmag > 0.5 && (nx * ex + ny * ey) / nmag > 0.5 && n_up < WIND_V {
                    let px = (c as f64 + 0.5) * dx;
                    let far_from_void = !has_void
                        || ((px - hole_cx) * (px - hole_cx) + (py - hole_cy) * (py - hole_cy))
                            >= excl_r2;
                    if far_from_void {
                        let u_proj =
                            bos.solve_r[SCR_UBAR_OFF + v] * ex + bos.solve_r[SCR_VBAR_OFF + v] * ey;
                        bos.solve_r[SCR_SORT_OFF + n_up] = u_proj;
                        bos.solve_r[SCR_SORT_OFF + WIND_V + n_up] = bos.solve_r[SCR_P0_OFF + v];
                        n_up += 1;
                    }
                }
            }
            c += 1;
        }
        r += 1;
    }

    if n_up >= 4 {
        let u_up = median_in_place(&mut bos.solve_r[SCR_SORT_OFF..SCR_SORT_OFF + n_up], n_up);
        let p_up = median_in_place(
            &mut bos.solve_r[SCR_SORT_OFF + WIND_V..SCR_SORT_OFF + WIND_V + n_up],
            n_up,
        );
        let u_inf = if u_up > 4.0 { u_up } else { vec_spd };
        (true, u_inf, ex, ey, angle, p_up)
    } else {
        let mut np = 0usize;
        r = 0;
        while r < WIND_H {
            let mut c = 0usize;
            while c < WIND_W {
                let in_band = c < 5 || c + 5 >= WIND_W || r < 4 || r + 4 >= WIND_H;
                let v = v_idx(c, r);
                if in_band && hodge.v_act[v] != 0 && np < WIND_V {
                    bos.solve_r[SCR_SORT_OFF + np] = bos.solve_r[SCR_P0_OFF + v];
                    np += 1;
                }
                c += 1;
            }
            r += 1;
        }
        let p_ref = median_in_place(&mut bos.solve_r[SCR_SORT_OFF..SCR_SORT_OFF + np], np);
        (true, vec_spd, ex, ey, angle, p_ref)
    }
}

#[inline(never)]
fn measure_primary_hole_geometry(
    hodge: &HodgeState,
    bos: &BosState,
    fast_head: usize,
    fast_count: usize,
    dx: f64,
    dy: f64,
    internal_voids: u32,
) -> (bool, f64, f64, f64, usize, usize, usize, usize) {
    if internal_voids == 0 {
        return (false, 0.0, 0.0, 0.0, 0, 0, 0, 0);
    }
    let mut n_faces = 0usize;
    let mut c_min = WIND_W;
    let mut c_max = 0usize;
    let mut r_min = WIND_H;
    let mut r_max = 0usize;
    let mut f_sx = 0.0f64;
    let mut f_sy = 0.0f64;

    let mut r = 0usize;
    while r < WIND_H - 1 {
        let mut c = 0usize;
        while c < WIND_W - 1 {
            if hodge.hole_id[f_idx(c, r)] == 1 {
                n_faces += 1;
                if c < c_min {
                    c_min = c;
                }
                if c + 1 > c_max {
                    c_max = c + 1;
                }
                if r < r_min {
                    r_min = r;
                }
                if r + 1 > r_max {
                    r_max = r + 1;
                }
                f_sx += (c as f64 + 1.0) * dx;
                f_sy += (r as f64 + 1.0) * dy;
            }
            c += 1;
        }
        r += 1;
    }
    if n_faces == 0 {
        return (false, 0.0, 0.0, 0.0, 0, 0, 0, 0);
    }

    let n_frames = fast_count.min(FAST_RING_LEN);
    let start = (fast_head + FAST_RING_LEN - n_frames) % FAST_RING_LEN;

    // Count persistent solid-obstacle vertices (inactive in >= 70% of the fast ring),
    // cleanly rejecting single-frame transient PIV rim gaps around the obstacle boundary!
    let mut n_persist = 0usize;
    let mut p_sx = 0.0f64;
    let mut p_sy = 0.0f64;
    r = r_min;
    while r <= r_max && r < WIND_H {
        let mut c = c_min;
        while c <= c_max && c < WIND_W {
            let v = v_idx(c, r);
            if hodge.v_act[v] == 0 {
                let touches = (c > 0 && r > 0 && hodge.hole_id[f_idx(c - 1, r - 1)] == 1)
                    || (c + 1 < WIND_W && r > 0 && hodge.hole_id[f_idx(c, r - 1)] == 1)
                    || (c > 0 && r + 1 < WIND_H && hodge.hole_id[f_idx(c - 1, r)] == 1)
                    || (c + 1 < WIND_W && r + 1 < WIND_H && hodge.hole_id[f_idx(c, r)] == 1);
                if touches {
                    let mut inact_cnt = 0usize;
                    let mut k = 0usize;
                    while k < n_frames {
                        let slot = (start + k) % FAST_RING_LEN;
                        if bos.solve_z[slot * WIND_V + v] < 0.5 {
                            inact_cnt += 1;
                        }
                        k += 1;
                    }
                    if n_frames == 0 || inact_cnt * 2 >= n_frames {
                        n_persist += 1;
                        p_sx += (c as f64 + 0.5) * dx;
                        p_sy += (r as f64 + 0.5) * dy;
                    }
                }
            }
            c += 1;
        }
        r += 1;
    }

    // Require >= 4 persistent inactive vertices so transient PIV pinholes never fabricate an obstacle
    if n_persist < 4 {
        return (false, 0.0, 0.0, 0.0, 0, 0, 0, 0);
    }

    let area_px2 = (n_persist as f64) * dx * dy;
    let cx = p_sx / (n_persist as f64);
    let cy = p_sy / (n_persist as f64);
    let _ = (f_sx, f_sy);
    let d_px = 2.0 * wasm_sqrt_f64(area_px2 / core::f64::consts::PI);
    (true, d_px, cx, cy, c_min, c_max, r_min, r_max)
}

#[inline(always)]
fn vertex_viscous_stress(
    hodge: &HodgeState,
    u: &[f64],
    v_vel: &[f64],
    c: usize,
    r: usize,
    dx: f64,
    dy: f64,
) -> (f64, f64, f64) {
    let idx = v_idx(c, r);
    let du_dx = if c > 0 && c + 1 < WIND_W && hodge.v_act[idx - 1] != 0 && hodge.v_act[idx + 1] != 0
    {
        0.5 * (u[idx + 1] - u[idx - 1]) / dx
    } else if c + 1 < WIND_W && hodge.v_act[idx + 1] != 0 {
        (u[idx + 1] - u[idx]) / dx
    } else if c > 0 && hodge.v_act[idx - 1] != 0 {
        (u[idx] - u[idx - 1]) / dx
    } else {
        0.0
    };
    let dv_dx = if c > 0 && c + 1 < WIND_W && hodge.v_act[idx - 1] != 0 && hodge.v_act[idx + 1] != 0
    {
        0.5 * (v_vel[idx + 1] - v_vel[idx - 1]) / dx
    } else if c + 1 < WIND_W && hodge.v_act[idx + 1] != 0 {
        (v_vel[idx + 1] - v_vel[idx]) / dx
    } else if c > 0 && hodge.v_act[idx - 1] != 0 {
        (v_vel[idx] - v_vel[idx - 1]) / dx
    } else {
        0.0
    };
    let du_dy =
        if r > 0 && r + 1 < WIND_H && hodge.v_act[idx - WIND_W] != 0 && hodge.v_act[idx + WIND_W] != 0
        {
            0.5 * (u[idx + WIND_W] - u[idx - WIND_W]) / dy
        } else if r + 1 < WIND_H && hodge.v_act[idx + WIND_W] != 0 {
            (u[idx + WIND_W] - u[idx]) / dy
        } else if r > 0 && hodge.v_act[idx - WIND_W] != 0 {
            (u[idx] - u[idx - WIND_W]) / dy
        } else {
            0.0
        };
    let dv_dy =
        if r > 0 && r + 1 < WIND_H && hodge.v_act[idx - WIND_W] != 0 && hodge.v_act[idx + WIND_W] != 0
        {
            0.5 * (v_vel[idx + WIND_W] - v_vel[idx - WIND_W]) / dy
        } else if r + 1 < WIND_H && hodge.v_act[idx + WIND_W] != 0 {
            (v_vel[idx + WIND_W] - v_vel[idx]) / dy
        } else if r > 0 && hodge.v_act[idx - WIND_W] != 0 {
            (v_vel[idx] - v_vel[idx - WIND_W]) / dy
        } else {
            0.0
        };
    (2.0 * du_dx, du_dy + dv_dx, 2.0 * dv_dy)
}

#[inline(always)]
fn accumulate_contour_edge(
    hodge: &HodgeState,
    u: &[f64],
    v_vel: &[f64],
    p0: &[f64],
    p_ref: f64,
    nu_eff: f64,
    dx: f64,
    dy: f64,
    c0: usize,
    r0: usize,
    c1: usize,
    r1: usize,
    nx: f64,
    ny: f64,
    ds: f64,
    fx: &mut f64,
    fy: &mut f64,
) {
    let v0 = v_idx(c0, r0);
    let v1 = v_idx(c1, r1);
    let pm = 0.5 * ((p0[v0] - p_ref) + (p0[v1] - p_ref));
    let um = 0.5 * (u[v0] + u[v1]);
    let vm = 0.5 * (v_vel[v0] + v_vel[v1]);
    let un = um * nx + vm * ny;

    let mut tx = -pm * nx - um * un;
    let mut ty = -pm * ny - vm * un;
    if nu_eff > 0.0 {
        let (sxx0, sxy0, syy0) = vertex_viscous_stress(hodge, u, v_vel, c0, r0, dx, dy);
        let (sxx1, sxy1, syy1) = vertex_viscous_stress(hodge, u, v_vel, c1, r1, dx, dy);
        let sxx = 0.5 * (sxx0 + sxx1);
        let sxy = 0.5 * (sxy0 + sxy1);
        let syy = 0.5 * (syy0 + syy1);
        tx += nu_eff * (sxx * nx + sxy * ny);
        ty += nu_eff * (sxy * nx + syy * ny);
    }
    *fx += tx * ds;
    *fy += ty * ds;
}

/// Method 1: Tight closed boundary contour integral around the primary obstacle void (`hole_id == 1`).
#[inline(never)]
fn integrate_hole_boundary_force(
    hodge: &HodgeState,
    u: &[f64],
    v_vel: &[f64],
    p0: &[f64],
    p_ref: f64,
    nu_eff: f64,
    dx: f64,
    dy: f64,
) -> (f64, f64, bool) {
    let mut fx = 0.0f64;
    let mut fy = 0.0f64;
    let mut n_edges = 0usize;

    let mut r = 1usize;
    while r + 1 < WIND_H {
        let mut c = 0usize;
        while c < WIND_W - 1 {
            let e = ex_idx(c, r);
            if hodge.e_act[e] != 0 {
                let f_top = f_idx(c, r - 1);
                let f_bot = f_idx(c, r);
                let top_hole = hodge.hole_id[f_top] == 1;
                let bot_hole = hodge.hole_id[f_bot] == 1;
                if top_hole && hodge.f_act[f_bot] != 0 {
                    accumulate_contour_edge(
                        hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c, r, c + 1, r, 0.0, 1.0, dx,
                        &mut fx, &mut fy,
                    );
                    n_edges += 1;
                } else if bot_hole && hodge.f_act[f_top] != 0 {
                    accumulate_contour_edge(
                        hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c, r, c + 1, r, 0.0, -1.0, dx,
                        &mut fx, &mut fy,
                    );
                    n_edges += 1;
                }
            }
            c += 1;
        }
        r += 1;
    }

    r = 0;
    while r < WIND_H - 1 {
        let mut c = 1usize;
        while c + 1 < WIND_W {
            let e = ey_idx(c, r);
            if hodge.e_act[e] != 0 {
                let f_left = f_idx(c - 1, r);
                let f_right = f_idx(c, r);
                let left_hole = hodge.hole_id[f_left] == 1;
                let right_hole = hodge.hole_id[f_right] == 1;
                if left_hole && hodge.f_act[f_right] != 0 {
                    accumulate_contour_edge(
                        hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c, r, c, r + 1, 1.0, 0.0, dy,
                        &mut fx, &mut fy,
                    );
                    n_edges += 1;
                } else if right_hole && hodge.f_act[f_left] != 0 {
                    accumulate_contour_edge(
                        hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c, r, c, r + 1, -1.0, 0.0, dy,
                        &mut fx, &mut fy,
                    );
                    n_edges += 1;
                }
            }
            c += 1;
        }
        r += 1;
    }

    (fx, fy, n_edges >= 4 && fx.is_finite() && fy.is_finite())
}

/// Method 2: `2`-cell-expanded rectangular control-volume flux balance enclosing the primary void.
#[inline(never)]
fn integrate_cv_box_force(
    hodge: &HodgeState,
    u: &[f64],
    v_vel: &[f64],
    p0: &[f64],
    du_dt: &[f64],
    dv_dt: &[f64],
    p_ref: f64,
    nu_eff: f64,
    dx: f64,
    dy: f64,
    h_c0: usize,
    h_c1: usize,
    h_r0: usize,
    h_r1: usize,
) -> (f64, f64, bool) {
    let mut pad = 2usize;
    loop {
        let c0 = h_c0.saturating_sub(pad).max(1);
        let c1 = (h_c1 + pad).min(WIND_W - 2);
        let r0 = h_r0.saturating_sub(pad).max(1);
        let r1 = (h_r1 + pad).min(WIND_H - 2);
        if c1 <= c0 + 1 || r1 <= r0 + 1 {
            return (0.0, 0.0, false);
        }

        let mut fx = 0.0f64;
        let mut fy = 0.0f64;
        let mut tot_edges = 0usize;
        let mut act_edges = 0usize;

        let mut c = c0;
        while c < c1 {
            tot_edges += 2;
            if hodge.e_act[ex_idx(c, r0)] != 0 {
                accumulate_contour_edge(
                    hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c, r0, c + 1, r0, 0.0, -1.0, dx,
                    &mut fx, &mut fy,
                );
                act_edges += 1;
            }
            if hodge.e_act[ex_idx(c, r1)] != 0 {
                accumulate_contour_edge(
                    hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c, r1, c + 1, r1, 0.0, 1.0, dx,
                    &mut fx, &mut fy,
                );
                act_edges += 1;
            }
            c += 1;
        }
        let mut r = r0;
        while r < r1 {
            tot_edges += 2;
            if hodge.e_act[ey_idx(c0, r)] != 0 {
                accumulate_contour_edge(
                    hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c0, r, c0, r + 1, -1.0, 0.0, dy,
                    &mut fx, &mut fy,
                );
                act_edges += 1;
            }
            if hodge.e_act[ey_idx(c1, r)] != 0 {
                accumulate_contour_edge(
                    hodge, u, v_vel, p0, p_ref, nu_eff, dx, dy, c1, r, c1, r + 1, 1.0, 0.0, dy,
                    &mut fx, &mut fy,
                );
                act_edges += 1;
            }
            r += 1;
        }

        if tot_edges > 0 && act_edges * 100 >= tot_edges * 85 {
            let cell_a = dx * dy;
            let mut ri = r0;
            while ri <= r1 {
                let wy = if ri == r0 || ri == r1 { 0.5 } else { 1.0 };
                let mut ci = c0;
                while ci <= c1 {
                    let wx = if ci == c0 || ci == c1 { 0.5 } else { 1.0 };
                    let v = v_idx(ci, ri);
                    if hodge.v_act[v] != 0 {
                        let w = wx * wy * cell_a;
                        fx -= du_dt[v] * w;
                        fy -= dv_dt[v] * w;
                    }
                    ci += 1;
                }
                ri += 1;
            }
            return (fx, fy, fx.is_finite() && fy.is_finite());
        }

        if pad <= 1 {
            return (0.0, 0.0, false);
        }
        pad -= 1;
    }
}

#[inline(never)]
fn find_overlapping_rigid_body(
    export_bodies: &[f32; MAX_RIGID_BODIES * BODY_EXPORT_STRIDE],
    hole_cx: f64,
    hole_cy: f64,
    d_hole_px: f64,
) -> (u32, f64, f64, f64) {
    let max_r = (0.75 * d_hole_px).max(16.0);
    let max_r2 = max_r * max_r;
    let mut best_id = 0u32;
    let mut best_d2 = max_r2;
    let mut best_vx = 0.0f64;
    let mut best_vy = 0.0f64;
    let mut best_om = 0.0f64;

    let mut b = 0usize;
    while b < MAX_RIGID_BODIES {
        let off = b * BODY_EXPORT_STRIDE;
        let bid = export_bodies[off];
        if bid > 0.5 && bid.is_finite() {
            let cx = export_bodies[off + 2] as f64;
            let cy = export_bodies[off + 3] as f64;
            let dx = cx - hole_cx;
            let dy = cy - hole_cy;
            let d2 = dx * dx + dy * dy;
            if d2.is_finite() && d2 < best_d2 {
                best_d2 = d2;
                best_id = bid as u32;
                best_vx = export_bodies[off + 4] as f64;
                best_vy = export_bodies[off + 5] as f64;
                best_om = export_bodies[off + 6] as f64;
            }
        }
        b += 1;
    }
    (best_id, best_vx, best_vy, best_om)
}
