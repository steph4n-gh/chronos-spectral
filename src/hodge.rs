//! Bounded `40x22` Eulerian Geometric `2`-Complex Discrete Hodge Decomposition (`Organ 3` / `Gate I3`).
//!
//! Pure `#![no_std]`, zero host imports, zero heap allocation inside static `HodgeState`.
//!
//! Mathematical structure on the `40x22` active vertex subcomplex (`V=880, E=1698, F=819`):
//! - Single consistent positive diagonal `W_1` edge metric used identically in both projections
//!   and all edge-level orthogonality / energy inner products.
//! - Exact component `omega_exact = d_0 phi_0` via `f64` Jacobi-PCG on `(d_0^T W_1 d_0)` with
//!   per-component zero-mean projection and free open-port boundary potentials.
//! - Coexact component `omega_coexact = W_1^{-1} d_1^T psi_2` via `f64` Jacobi-PCG on
//!   `(d_1 W_1^{-1} d_1^T)` over active plaquettes `f_act`.
//! - Harmonic residual `omega_harm = omega - omega_exact - omega_coexact` and signed circulation
//!   `Gamma_1` around the primary enclosed hole boundary.
//! - Complementary plaquette dual flood-fill distinguishing enclosed internal holes (`beta_1`
//!   generators) from exterior-connected open-port cutouts.
//! - Downstream wake / occupancy-gap hypotheses compare reconstructed downstream speed against
//!   upstream inflow speed; a deficit hypothesis is an observational cue, not proof of a solid
//!   obstacle.

pub const HODGE_W: usize = 40;
pub const HODGE_H: usize = 22;
pub const HODGE_V: usize = HODGE_W * HODGE_H; // 880
pub const HODGE_EX: usize = (HODGE_W - 1) * HODGE_H; // 858
pub const HODGE_EY: usize = HODGE_W * (HODGE_H - 1); // 840
pub const HODGE_E: usize = HODGE_EX + HODGE_EY; // 1,698
pub const HODGE_F: usize = (HODGE_W - 1) * (HODGE_H - 1); // 819
pub const HODGE_TELEM_LEN: usize = 32;
pub const HODGE_PLANES_LEN: usize = 7 * HODGE_V; // 6,160
pub const HODGE_INJECT_LEN: usize = 3 * HODGE_V; // 2,640

pub const HODGE_PLANE_EXACT_U: usize = 0;
pub const HODGE_PLANE_EXACT_V: usize = 1;
pub const HODGE_PLANE_COEXACT_U: usize = 2;
pub const HODGE_PLANE_COEXACT_V: usize = 3;
pub const HODGE_PLANE_HARM_U: usize = 4;
pub const HODGE_PLANE_HARM_V: usize = 5;
pub const HODGE_PLANE_PHI: usize = 6;

pub const HODGE_MODE_OFF: u32 = 0;
pub const HODGE_MODE_STAGGERED: u32 = 1;
pub const HODGE_MODE_EVERY_FRAME: u32 = 2;

pub const HODGE_STATE_OFF: u32 = 0;
pub const HODGE_STATE_STANDBY: u32 = 1;
pub const HODGE_STATE_READY: u32 = 2;
pub const HODGE_STATE_INVALID_UNCONVERGED: u32 = 3;

// Telemetry word indices (`[u32; 32]`; float entries stored as `f32::to_bits()`)
pub const HODGE_TELEM_STATE: usize = 0;
pub const HODGE_TELEM_MODE: usize = 1;
pub const HODGE_TELEM_CONVERGED: usize = 2;
pub const HODGE_TELEM_V_ACT: usize = 3;
pub const HODGE_TELEM_E_ACT: usize = 4;
pub const HODGE_TELEM_F_ACT: usize = 5;
pub const HODGE_TELEM_COMP_CNT: usize = 6;
pub const HODGE_TELEM_BETA1: usize = 7;
pub const HODGE_TELEM_INTERNAL_VOIDS: usize = 8;
pub const HODGE_TELEM_OPEN_PORTS: usize = 9;
pub const HODGE_TELEM_WAKE_HYPS: usize = 10;
pub const HODGE_TELEM_TOTAL_ENERGY: usize = 11;
pub const HODGE_TELEM_EXACT_FRAC: usize = 12;
pub const HODGE_TELEM_COEXACT_FRAC: usize = 13;
pub const HODGE_TELEM_HARM_FRAC: usize = 14;
pub const HODGE_TELEM_ORTH_EC_REL: usize = 15;
pub const HODGE_TELEM_ORTH_EH_REL: usize = 16;
pub const HODGE_TELEM_ORTH_CH_REL: usize = 17;
pub const HODGE_TELEM_ENERGY_PART_ERR: usize = 18;
pub const HODGE_TELEM_EXACT_CG_RES: usize = 19;
pub const HODGE_TELEM_COEXACT_CG_RES: usize = 20;
pub const HODGE_TELEM_PRIMARY_VOID_CIRC: usize = 21;
pub const HODGE_TELEM_PRIMARY_WAKE_DEFICIT: usize = 22;
pub const HODGE_TELEM_VOID_CX: usize = 23;
pub const HODGE_TELEM_VOID_CY: usize = 24;
pub const HODGE_TELEM_WAKE_X0: usize = 25;
pub const HODGE_TELEM_WAKE_Y0: usize = 26;
pub const HODGE_TELEM_WAKE_X1: usize = 27;
pub const HODGE_TELEM_WAKE_Y1: usize = 28;
pub const HODGE_TELEM_EXACT_ITERS: usize = 29;
pub const HODGE_TELEM_COEXACT_ITERS: usize = 30;
pub const HODGE_TELEM_RESERVED31: usize = 31;

const VCLASS_INACTIVE: u8 = 0;
const VCLASS_ACTIVE: u8 = 1;
const VCLASS_HOLE_BND: u8 = 2;
const VCLASS_PORT_BND: u8 = 4;

const EPORT_NONE: u8 = 0;
const EPORT_HOLE_BND: u8 = 1;
const EPORT_PORT_BND: u8 = 2;

const HODGE_PCG_MAX_ITERS: usize = 160;
const HODGE_PCG_TOL: f64 = 1e-8;
const HODGE_PCG_HARD_FAIL: f64 = 1e-5;

#[repr(C, align(16))]
pub struct HodgeState {
    // C-ABI flat export & injection buffers
    pub planes: [f32; HODGE_PLANES_LEN],
    pub inject_buf: [f32; HODGE_INJECT_LEN],
    pub telem: [u32; HODGE_TELEM_LEN],

    // 1-cochain and dual weight vectors (f64)
    pub omega: [f64; HODGE_E],
    pub omega_exact: [f64; HODGE_E],
    pub omega_coexact: [f64; HODGE_E],
    pub omega_harm: [f64; HODGE_E],
    pub w1_edge: [f64; HODGE_E],

    // 0-cochain & 2-cochain potentials + f64 PCG scratch (all <= 880 nodes)
    pub phi_0: [f64; HODGE_V],
    pub psi_2: [f64; HODGE_F],
    pub cg_rhs: [f64; HODGE_V],
    pub cg_r: [f64; HODGE_V],
    pub cg_p: [f64; HODGE_V],
    pub cg_z: [f64; HODGE_V],
    pub cg_ap: [f64; HODGE_V],
    pub cg_inv_diag: [f64; HODGE_V],
    pub comp_sum: [f64; HODGE_V],

    // Topology masks & BFS queue
    pub comp_id: [u16; HODGE_V],
    pub comp_count: [u16; HODGE_V],
    pub bfs_queue: [u16; HODGE_V],
    pub v_act: [u8; HODGE_V],
    pub v_class: [u8; HODGE_V],
    pub e_act: [u8; HODGE_E],
    pub e_port: [u8; HODGE_E],
    pub f_act: [u8; HODGE_F],
    pub hole_id: [u16; HODGE_V],

    // Scalar config
    pub mode: u32,
    pub last_solve_frame: u32,
    pub live_ready: u8,
    pub _pad: [u8; 3],
}

// Budget: export buffers (35.3 KiB) + unaliased f64 2-complex PCG solver (<= 256 KiB total inside 448-page Arena).
const _: () = assert!(core::mem::size_of::<HodgeState>() <= 262_144);

#[inline(always)]
const fn v_idx(c: usize, r: usize) -> usize {
    r * HODGE_W + c
}

#[inline(always)]
const fn ex_idx(c: usize, r: usize) -> usize {
    r * (HODGE_W - 1) + c
}

#[inline(always)]
const fn ey_idx(c: usize, r: usize) -> usize {
    HODGE_EX + r * HODGE_W + c
}

#[inline(always)]
const fn f_idx(c: usize, r: usize) -> usize {
    r * (HODGE_W - 1) + c
}

#[inline(always)]
fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
}

#[inline(always)]
fn pack_f32_checked(val: f64) -> Option<u32> {
    if !val.is_finite() {
        return None;
    }
    let f = val as f32;
    if !f.is_finite() {
        return None;
    }
    Some(f.to_bits())
}

fn project_zero_mean(
    vec: &mut [f64; HODGE_V],
    v_act: &[u8; HODGE_V],
    comp_id: &[u16; HODGE_V],
    comp_count: &[u16; HODGE_V],
    comp_sum: &mut [f64; HODGE_V],
    comp_cnt: usize,
) {
    let mut cid = 0usize;
    while cid < comp_cnt {
        comp_sum[cid] = 0.0;
        cid += 1;
    }
    let mut v = 0usize;
    while v < HODGE_V {
        if v_act[v] != 0 {
            let c = comp_id[v] as usize;
            if c < comp_cnt {
                comp_sum[c] += vec[v];
            }
        }
        v += 1;
    }
    cid = 0;
    while cid < comp_cnt {
        let cnt = comp_count[cid] as usize;
        if cnt > 0 {
            comp_sum[cid] /= cnt as f64;
        }
        cid += 1;
    }
    v = 0;
    while v < HODGE_V {
        if v_act[v] != 0 {
            let c = comp_id[v] as usize;
            if c < comp_cnt {
                vec[v] -= comp_sum[c];
            }
        } else {
            vec[v] = 0.0;
        }
        v += 1;
    }
}

fn apply_l0(
    x: &[f64; HODGE_V],
    y: &mut [f64; HODGE_V],
    e_act: &[u8; HODGE_E],
    w1_edge: &[f64; HODGE_E],
) {
    y.fill(0.0);
    let mut r = 0usize;
    while r < HODGE_H {
        let mut c = 0usize;
        while c < HODGE_W - 1 {
            let e = ex_idx(c, r);
            if e_act[e] != 0 {
                let v0 = v_idx(c, r);
                let v1 = v0 + 1;
                let flux = w1_edge[e] * (x[v1] - x[v0]);
                y[v0] -= flux;
                y[v1] += flux;
            }
            c += 1;
        }
        r += 1;
    }
    r = 0;
    while r < HODGE_H - 1 {
        let mut c = 0usize;
        while c < HODGE_W {
            let e = ey_idx(c, r);
            if e_act[e] != 0 {
                let v0 = v_idx(c, r);
                let v1 = v0 + HODGE_W;
                let flux = w1_edge[e] * (x[v1] - x[v0]);
                y[v0] -= flux;
                y[v1] += flux;
            }
            c += 1;
        }
        r += 1;
    }
}

/// Evaluates `out = W_1^{-1} d_1^T psi` on active edges using CCW plaquette orientation.
fn apply_w1_inv_d1_t(
    psi: &[f64; HODGE_F],
    f_act: &[u8; HODGE_F],
    e_act: &[u8; HODGE_E],
    w1_edge: &[f64; HODGE_E],
    out: &mut [f64; HODGE_E],
) {
    out.fill(0.0);
    let mut r = 0usize;
    while r < HODGE_H - 1 {
        let mut c = 0usize;
        while c < HODGE_W - 1 {
            let f = f_idx(c, r);
            if f_act[f] != 0 {
                let p = psi[f];
                out[ex_idx(c, r)] += p;
                out[ey_idx(c + 1, r)] += p;
                out[ex_idx(c, r + 1)] -= p;
                out[ey_idx(c, r)] -= p;
            }
            c += 1;
        }
        r += 1;
    }
    let mut e = 0usize;
    while e < HODGE_E {
        if e_act[e] != 0 && w1_edge[e] > 0.0 {
            out[e] /= w1_edge[e];
        } else {
            out[e] = 0.0;
        }
        e += 1;
    }
}

fn apply_d1(omega_in: &[f64; HODGE_E], f_act: &[u8; HODGE_F], out: &mut [f64]) {
    let mut r = 0usize;
    while r < HODGE_H - 1 {
        let mut c = 0usize;
        while c < HODGE_W - 1 {
            let f = f_idx(c, r);
            if f_act[f] != 0 {
                out[f] = omega_in[ex_idx(c, r)]
                    + omega_in[ey_idx(c + 1, r)]
                    - omega_in[ex_idx(c, r + 1)]
                    - omega_in[ey_idx(c, r)];
            } else {
                out[f] = 0.0;
            }
            c += 1;
        }
        r += 1;
    }
}

impl HodgeState {
    pub fn reset(&mut self) {
        let ptr = self as *mut Self as *mut u8;
        unsafe {
            core::ptr::write_bytes(ptr, 0, core::mem::size_of::<Self>());
        }
    }

    pub fn reset_keep_mode(&mut self) {
        let keep_mode = self.mode;
        self.reset();
        self.mode = keep_mode;
        self.live_ready = 0;
        self.telem[HODGE_TELEM_MODE] = keep_mode;
        self.telem[HODGE_TELEM_STATE] = if keep_mode == HODGE_MODE_OFF {
            HODGE_STATE_OFF
        } else {
            HODGE_STATE_STANDBY
        };
    }

    pub fn set_mode(&mut self, mode: u32) -> u32 {
        let sanitized = mode.min(HODGE_MODE_EVERY_FRAME);
        self.mode = sanitized;
        if sanitized == HODGE_MODE_OFF {
            self.reset_keep_mode();
        } else {
            self.telem[HODGE_TELEM_MODE] = sanitized;
            if self.telem[HODGE_TELEM_STATE] == HODGE_STATE_OFF {
                self.telem[HODGE_TELEM_STATE] = HODGE_STATE_STANDBY;
            }
        }
        1
    }

    #[inline(always)]
    pub fn get_mode(&self) -> u32 {
        self.mode
    }

    #[inline(always)]
    pub fn get_state(&self) -> u32 {
        self.telem[HODGE_TELEM_STATE]
    }

    #[inline(always)]
    pub fn telemetry_ptr(&self) -> *const u32 {
        self.telem.as_ptr()
    }

    #[inline(always)]
    pub fn planes_ptr(&self) -> *const f32 {
        self.planes.as_ptr()
    }

    #[inline(always)]
    pub fn inject_mut_ptr(&mut self) -> *mut f32 {
        self.inject_buf.as_mut_ptr()
    }

    #[inline(never)]
    pub fn solve_from_continuum(
        &mut self,
        export_field: &[f32; HODGE_V * 6],
        cell_valid_mask: &[u8; HODGE_V],
        grid_dx: f32,
        grid_dy: f32,
        frame_idx: u32,
    ) -> bool {
        if self.mode == HODGE_MODE_OFF {
            self.telem[HODGE_TELEM_STATE] = HODGE_STATE_OFF;
            self.live_ready = 0;
            return false;
        }
        if self.mode == HODGE_MODE_STAGGERED
            && (frame_idx & 1) != 0
            && self.live_ready != 0
            && self.telem[HODGE_TELEM_STATE] == HODGE_STATE_READY
        {
            return false;
        }
        if !grid_dx.is_finite() || !grid_dy.is_finite() || grid_dx <= 0.0 || grid_dy <= 0.0 {
            self.fail_unconverged(false);
            return false;
        }

        let mut v = 0usize;
        while v < HODGE_V {
            if (cell_valid_mask[v] & 1) != 0 {
                let base = v * 6;
                let u = export_field[base];
                let vy = export_field[base + 1];
                if !u.is_finite() || !vy.is_finite() {
                    self.fail_unconverged(false);
                    return false;
                }
                self.v_act[v] = 1;
                self.cg_r[v] = u as f64;
                self.cg_z[v] = vy as f64;
            } else {
                self.v_act[v] = 0;
                self.cg_r[v] = 0.0;
                self.cg_z[v] = 0.0;
            }
            v += 1;
        }

        if self.solve_core(grid_dx as f64, grid_dy as f64) {
            self.last_solve_frame = frame_idx;
            self.live_ready = 1;
            true
        } else {
            self.live_ready = 0;
            false
        }
    }

    #[inline(never)]
    pub fn solve_injected(&mut self, grid_dx: f32, grid_dy: f32) -> u32 {
        self.live_ready = 0;
        if !grid_dx.is_finite() || !grid_dy.is_finite() || grid_dx <= 0.0 || grid_dy <= 0.0 {
            self.fail_unconverged(false);
            self.live_ready = 0;
            return 0;
        }

        let mut v = 0usize;
        while v < HODGE_V {
            let m = self.inject_buf[2 * HODGE_V + v];
            if !m.is_finite() {
                self.fail_unconverged(false);
                self.live_ready = 0;
                return 0;
            }
            if m > 0.5 {
                let u = self.inject_buf[v];
                let vy = self.inject_buf[HODGE_V + v];
                if !u.is_finite() || !vy.is_finite() {
                    self.fail_unconverged(false);
                    self.live_ready = 0;
                    return 0;
                }
                self.v_act[v] = 1;
                self.cg_r[v] = u as f64;
                self.cg_z[v] = vy as f64;
            } else {
                self.v_act[v] = 0;
                self.cg_r[v] = 0.0;
                self.cg_z[v] = 0.0;
            }
            v += 1;
        }

        let ok = self.solve_core(grid_dx as f64, grid_dy as f64);
        self.live_ready = 0;
        if ok {
            1
        } else {
            0
        }
    }

    #[inline(never)]
    fn fail_unconverged(&mut self, keep_topo: bool) {
        self.live_ready = 0;
        self.planes.fill(0.0);
        self.phi_0.fill(0.0);
        self.psi_2.fill(0.0);
        self.omega_exact.fill(0.0);
        self.omega_coexact.fill(0.0);
        self.omega_harm.fill(0.0);
        if !keep_topo {
            self.telem.fill(0);
        } else {
            let mut k = HODGE_TELEM_WAKE_HYPS;
            while k < HODGE_TELEM_LEN {
                self.telem[k] = 0;
                k += 1;
            }
        }
        self.telem[HODGE_TELEM_STATE] = HODGE_STATE_INVALID_UNCONVERGED;
        self.telem[HODGE_TELEM_MODE] = self.mode;
        self.telem[HODGE_TELEM_CONVERGED] = 0;
        self.telem[HODGE_TELEM_RESERVED31] = 0;
    }

    #[inline(never)]
    fn solve_core(&mut self, dx: f64, dy: f64) -> bool {
        self.telem.fill(0);
        self.telem[HODGE_TELEM_MODE] = self.mode;

        let (topo_ok, v_act_cnt, e_act_cnt, f_act_cnt, comp_cnt, beta1, internal_void_cnt, open_port_cnt) =
            self.build_topology_and_cochain(dx, dy);

        self.telem[HODGE_TELEM_V_ACT] = v_act_cnt as u32;
        self.telem[HODGE_TELEM_E_ACT] = e_act_cnt as u32;
        self.telem[HODGE_TELEM_F_ACT] = f_act_cnt as u32;
        self.telem[HODGE_TELEM_COMP_CNT] = comp_cnt as u32;
        self.telem[HODGE_TELEM_BETA1] = beta1;
        self.telem[HODGE_TELEM_INTERNAL_VOIDS] = internal_void_cnt as u32;
        self.telem[HODGE_TELEM_OPEN_PORTS] = open_port_cnt as u32;

        if !topo_ok || v_act_cnt < 4 || e_act_cnt == 0 {
            self.fail_unconverged(topo_ok);
            return false;
        }

        let mut omega_scale = 0.0f64;
        let mut e = 0usize;
        while e < HODGE_E {
            if self.e_act[e] != 0 {
                let wmag = abs_f64(self.omega[e]);
                if !wmag.is_finite() {
                    self.fail_unconverged(true);
                    return false;
                }
                if wmag > omega_scale {
                    omega_scale = wmag;
                }
            } else {
                self.omega[e] = 0.0;
            }
            e += 1;
        }

        let (
            exact_iters,
            coexact_iters,
            exact_res_bits,
            coexact_res_bits,
            e_tot,
            exact_frac,
            coexact_frac,
            harm_frac,
            orth_ec_rel,
            orth_eh_rel,
            orth_ch_rel,
            energy_part_err,
        ) = if omega_scale <= 0.0 {
            self.phi_0.fill(0.0);
            self.psi_2.fill(0.0);
            self.omega_exact.fill(0.0);
            self.omega_coexact.fill(0.0);
            self.omega_harm.fill(0.0);
            (0u32, 0u32, 0u32, 0u32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            let inv_scale = 1.0 / omega_scale;
            if !inv_scale.is_finite() {
                self.fail_unconverged(true);
                return false;
            }
            e = 0;
            while e < HODGE_E {
                if self.e_act[e] != 0 {
                    self.omega[e] *= inv_scale;
                }
                e += 1;
            }

            let (exact_ok, exact_iters, exact_res) = self.solve_exact_pcg(comp_cnt);
            let (coexact_ok, coexact_iters, coexact_res) = self.solve_coexact_pcg();

            if !exact_ok
                || !coexact_ok
                || !exact_res.is_finite()
                || !coexact_res.is_finite()
                || exact_res > HODGE_PCG_HARD_FAIL
                || coexact_res > HODGE_PCG_HARD_FAIL
            {
                self.fail_unconverged(true);
                return false;
            }

            let Some(exact_res_bits) = pack_f32_checked(exact_res) else {
                self.fail_unconverged(true);
                return false;
            };
            let Some(coexact_res_bits) = pack_f32_checked(coexact_res) else {
                self.fail_unconverged(true);
                return false;
            };

            let mut e_tot_u = 0.0f64;
            let mut e_ex_u = 0.0f64;
            let mut e_co_u = 0.0f64;
            let mut e_h_u = 0.0f64;
            let mut dot_ec_u = 0.0f64;
            let mut dot_eh_u = 0.0f64;
            let mut dot_ch_u = 0.0f64;

            e = 0;
            while e < HODGE_E {
                if self.e_act[e] != 0 {
                    let oe = self.omega_exact[e];
                    let oc = self.omega_coexact[e];
                    let oh = self.omega[e] - oe - oc;
                    if !oh.is_finite() {
                        self.fail_unconverged(true);
                        return false;
                    }
                    self.omega_harm[e] = oh;
                    let w = self.w1_edge[e];
                    let om = self.omega[e];
                    e_tot_u += w * om * om;
                    e_ex_u += w * oe * oe;
                    e_co_u += w * oc * oc;
                    e_h_u += w * oh * oh;
                    dot_ec_u += w * oe * oc;
                    dot_eh_u += w * oe * oh;
                    dot_ch_u += w * oc * oh;
                } else {
                    self.omega_harm[e] = 0.0;
                }
                e += 1;
            }

            let (ef, cf, hf, oec, oeh, och, epe) = if e_tot_u <= 1.0e-300 {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            } else {
                let denom = e_tot_u;
                (
                    e_ex_u / denom,
                    e_co_u / denom,
                    e_h_u / denom,
                    abs_f64(dot_ec_u) / denom,
                    abs_f64(dot_eh_u) / denom,
                    abs_f64(dot_ch_u) / denom,
                    abs_f64((e_ex_u + e_co_u + e_h_u) - e_tot_u) / denom,
                )
            };

            let mut v = 0usize;
            while v < HODGE_V {
                if self.v_act[v] != 0 {
                    self.phi_0[v] *= omega_scale;
                } else {
                    self.phi_0[v] = 0.0;
                }
                v += 1;
            }
            e = 0;
            while e < HODGE_E {
                if self.e_act[e] != 0 {
                    self.omega[e] *= omega_scale;
                    self.omega_exact[e] *= omega_scale;
                    self.omega_coexact[e] *= omega_scale;
                    self.omega_harm[e] *= omega_scale;
                }
                e += 1;
            }
            let e_tot_phys = e_tot_u * omega_scale * omega_scale;
            (
                exact_iters,
                coexact_iters,
                exact_res_bits,
                coexact_res_bits,
                e_tot_phys,
                ef,
                cf,
                hf,
                oec,
                oeh,
                och,
                epe,
            )
        };

        // Signed circulation Gamma_1 around primary enclosed hole (hole_id == 1)
        let mut primary_circ = 0.0f64;
        if internal_void_cnt >= 1 {
            let mut r = 0usize;
            while r < HODGE_H - 1 {
                let mut c = 0usize;
                while c < HODGE_W - 1 {
                    let f = f_idx(c, r);
                    if self.hole_id[f] == 1 {
                        primary_circ += self.omega_harm[ex_idx(c, r)]
                            + self.omega_harm[ey_idx(c + 1, r)]
                            - self.omega_harm[ex_idx(c, r + 1)]
                            - self.omega_harm[ey_idx(c, r)];
                    }
                    c += 1;
                }
                r += 1;
            }
        }

        let (recon_ok, wake_hyp_cnt, primary_deficit, void_cx, void_cy, wake_bbox) =
            self.reconstruct_and_detect_wakes(dx, dy, internal_void_cnt, open_port_cnt);
        if !recon_ok {
            self.fail_unconverged(true);
            return false;
        }

        let Some(e_tot_b) = pack_f32_checked(e_tot) else { self.fail_unconverged(true); return false; };
        let Some(ef_b) = pack_f32_checked(exact_frac) else { self.fail_unconverged(true); return false; };
        let Some(cf_b) = pack_f32_checked(coexact_frac) else { self.fail_unconverged(true); return false; };
        let Some(hf_b) = pack_f32_checked(harm_frac) else { self.fail_unconverged(true); return false; };
        let Some(oec_b) = pack_f32_checked(orth_ec_rel) else { self.fail_unconverged(true); return false; };
        let Some(oeh_b) = pack_f32_checked(orth_eh_rel) else { self.fail_unconverged(true); return false; };
        let Some(och_b) = pack_f32_checked(orth_ch_rel) else { self.fail_unconverged(true); return false; };
        let Some(epe_b) = pack_f32_checked(energy_part_err) else { self.fail_unconverged(true); return false; };
        let Some(circ_b) = pack_f32_checked(primary_circ) else { self.fail_unconverged(true); return false; };
        let Some(def_b) = pack_f32_checked(primary_deficit) else { self.fail_unconverged(true); return false; };
        let Some(vcx_b) = pack_f32_checked(void_cx) else { self.fail_unconverged(true); return false; };
        let Some(vcy_b) = pack_f32_checked(void_cy) else { self.fail_unconverged(true); return false; };
        let Some(wx0_b) = pack_f32_checked(wake_bbox[0]) else { self.fail_unconverged(true); return false; };
        let Some(wy0_b) = pack_f32_checked(wake_bbox[1]) else { self.fail_unconverged(true); return false; };
        let Some(wx1_b) = pack_f32_checked(wake_bbox[2]) else { self.fail_unconverged(true); return false; };
        let Some(wy1_b) = pack_f32_checked(wake_bbox[3]) else { self.fail_unconverged(true); return false; };

        self.telem[HODGE_TELEM_STATE] = HODGE_STATE_READY;
        self.telem[HODGE_TELEM_MODE] = self.mode;
        self.telem[HODGE_TELEM_CONVERGED] = 1;
        self.telem[HODGE_TELEM_WAKE_HYPS] = wake_hyp_cnt;
        self.telem[HODGE_TELEM_TOTAL_ENERGY] = e_tot_b;
        self.telem[HODGE_TELEM_EXACT_FRAC] = ef_b;
        self.telem[HODGE_TELEM_COEXACT_FRAC] = cf_b;
        self.telem[HODGE_TELEM_HARM_FRAC] = hf_b;
        self.telem[HODGE_TELEM_ORTH_EC_REL] = oec_b;
        self.telem[HODGE_TELEM_ORTH_EH_REL] = oeh_b;
        self.telem[HODGE_TELEM_ORTH_CH_REL] = och_b;
        self.telem[HODGE_TELEM_ENERGY_PART_ERR] = epe_b;
        self.telem[HODGE_TELEM_EXACT_CG_RES] = exact_res_bits;
        self.telem[HODGE_TELEM_COEXACT_CG_RES] = coexact_res_bits;
        self.telem[HODGE_TELEM_PRIMARY_VOID_CIRC] = circ_b;
        self.telem[HODGE_TELEM_PRIMARY_WAKE_DEFICIT] = def_b;
        self.telem[HODGE_TELEM_VOID_CX] = vcx_b;
        self.telem[HODGE_TELEM_VOID_CY] = vcy_b;
        self.telem[HODGE_TELEM_WAKE_X0] = wx0_b;
        self.telem[HODGE_TELEM_WAKE_Y0] = wy0_b;
        self.telem[HODGE_TELEM_WAKE_X1] = wx1_b;
        self.telem[HODGE_TELEM_WAKE_Y1] = wy1_b;
        self.telem[HODGE_TELEM_EXACT_ITERS] = exact_iters;
        self.telem[HODGE_TELEM_COEXACT_ITERS] = coexact_iters;
        self.telem[HODGE_TELEM_RESERVED31] = 0;
        true
    }

    #[inline(never)]
    fn build_topology_and_cochain(
        &mut self,
        dx: f64,
        dy: f64,
    ) -> (bool, usize, usize, usize, usize, u32, usize, usize) {
        let mut v_act_cnt = 0usize;
        let mut v = 0usize;
        while v < HODGE_V {
            if self.v_act[v] != 0 {
                v_act_cnt += 1;
                self.v_class[v] = VCLASS_ACTIVE;
            } else {
                self.v_class[v] = VCLASS_INACTIVE;
            }
            v += 1;
        }

        let mut e_act_cnt = 0usize;
        let dy_over_dx = dy / dx;
        let dx_over_dy = dx / dy;

        let mut r = 0usize;
        while r < HODGE_H {
            let weight = dy_over_dx * if r == 0 || r == HODGE_H - 1 { 0.5 } else { 1.0 };
            let mut c = 0usize;
            while c < HODGE_W - 1 {
                let e = ex_idx(c, r);
                let v0 = v_idx(c, r);
                let v1 = v0 + 1;
                self.w1_edge[e] = weight;
                self.e_port[e] = EPORT_NONE;
                if self.v_act[v0] != 0 && self.v_act[v1] != 0 {
                    let line = 0.5 * (self.cg_r[v0] + self.cg_r[v1]) * dx;
                    if !line.is_finite() {
                        return (false, v_act_cnt, e_act_cnt, 0, 0, 0, 0, 0);
                    }
                    self.e_act[e] = 1;
                    self.omega[e] = line;
                    e_act_cnt += 1;
                } else {
                    self.e_act[e] = 0;
                    self.omega[e] = 0.0;
                }
                c += 1;
            }
            r += 1;
        }

        r = 0;
        while r < HODGE_H - 1 {
            let mut c = 0usize;
            while c < HODGE_W {
                let e = ey_idx(c, r);
                let v0 = v_idx(c, r);
                let v1 = v0 + HODGE_W;
                let weight = dx_over_dy * if c == 0 || c == HODGE_W - 1 { 0.5 } else { 1.0 };
                self.w1_edge[e] = weight;
                self.e_port[e] = EPORT_NONE;
                if self.v_act[v0] != 0 && self.v_act[v1] != 0 {
                    let line = 0.5 * (self.cg_z[v0] + self.cg_z[v1]) * dy;
                    if !line.is_finite() {
                        return (false, v_act_cnt, e_act_cnt, 0, 0, 0, 0, 0);
                    }
                    self.e_act[e] = 1;
                    self.omega[e] = line;
                    e_act_cnt += 1;
                } else {
                    self.e_act[e] = 0;
                    self.omega[e] = 0.0;
                }
                c += 1;
            }
            r += 1;
        }

        let mut f_act_cnt = 0usize;
        r = 0;
        while r < HODGE_H - 1 {
            let mut c = 0usize;
            while c < HODGE_W - 1 {
                let f = f_idx(c, r);
                let et = ex_idx(c, r);
                let er = ey_idx(c + 1, r);
                let eb = ex_idx(c, r + 1);
                let el = ey_idx(c, r);
                if self.e_act[et] != 0
                    && self.e_act[er] != 0
                    && self.e_act[eb] != 0
                    && self.e_act[el] != 0
                {
                    self.f_act[f] = 1;
                    self.hole_id[f] = 0;
                    f_act_cnt += 1;
                } else {
                    self.f_act[f] = 0;
                    self.hole_id[f] = u16::MAX;
                }
                c += 1;
            }
            r += 1;
        }
        let mut extra = HODGE_F;
        while extra < HODGE_V {
            self.hole_id[extra] = 0;
            extra += 1;
        }

        // Connected active vertex components via BFS
        self.comp_id.fill(u16::MAX);
        self.comp_count.fill(0);
        let mut comp_cnt = 0usize;
        let mut v0 = 0usize;
        while v0 < HODGE_V {
            if self.v_act[v0] == 0 || self.comp_id[v0] != u16::MAX {
                v0 += 1;
                continue;
            }
            let cid = comp_cnt as u16;
            self.comp_id[v0] = cid;
            self.bfs_queue[0] = v0 as u16;
            let mut head = 0usize;
            let mut tail = 1usize;
            while head < tail {
                let u = self.bfs_queue[head] as usize;
                head += 1;
                let ur = u / HODGE_W;
                let uc = u - ur * HODGE_W;
                if uc > 0 && self.e_act[ex_idx(uc - 1, ur)] != 0 && self.comp_id[u - 1] == u16::MAX
                {
                    self.comp_id[u - 1] = cid;
                    self.bfs_queue[tail] = (u - 1) as u16;
                    tail += 1;
                }
                if uc + 1 < HODGE_W
                    && self.e_act[ex_idx(uc, ur)] != 0
                    && self.comp_id[u + 1] == u16::MAX
                {
                    self.comp_id[u + 1] = cid;
                    self.bfs_queue[tail] = (u + 1) as u16;
                    tail += 1;
                }
                if ur > 0 && self.e_act[ey_idx(uc, ur - 1)] != 0 && self.comp_id[u - HODGE_W] == u16::MAX
                {
                    self.comp_id[u - HODGE_W] = cid;
                    self.bfs_queue[tail] = (u - HODGE_W) as u16;
                    tail += 1;
                }
                if ur + 1 < HODGE_H
                    && self.e_act[ey_idx(uc, ur)] != 0
                    && self.comp_id[u + HODGE_W] == u16::MAX
                {
                    self.comp_id[u + HODGE_W] = cid;
                    self.bfs_queue[tail] = (u + HODGE_W) as u16;
                    tail += 1;
                }
            }
            self.comp_count[comp_cnt] = tail as u16;
            comp_cnt += 1;
            v0 += 1;
        }

        let euler_b1 =
            (e_act_cnt as i32) - (v_act_cnt as i32) - (f_act_cnt as i32) + (comp_cnt as i32);
        let beta1 = if euler_b1 > 0 { euler_b1 as u32 } else { 0 };

        // Complementary plaquette flood-fill (`!f_act` across absent primal edges `!e_act`)
        let mut internal_void_cnt = 0usize;
        let mut open_port_cnt = 0usize;
        let mut primary_hole_size = 0usize;

        let mut f0 = 0usize;
        while f0 < HODGE_F {
            if self.hole_id[f0] != u16::MAX {
                f0 += 1;
                continue;
            }
            self.hole_id[f0] = 0xFFFE;
            self.bfs_queue[0] = f0 as u16;
            let mut head = 0usize;
            let mut tail = 1usize;
            let mut reaches_ext = false;

            while head < tail {
                let f = self.bfs_queue[head] as usize;
                head += 1;
                let fr = f / (HODGE_W - 1);
                let fc = f - fr * (HODGE_W - 1);

                // Check exterior virtual frame contact across absent domain-border edges
                if (fr == 0 && self.e_act[ex_idx(fc, 0)] == 0)
                    || (fr == HODGE_H - 2 && self.e_act[ex_idx(fc, HODGE_H - 1)] == 0)
                    || (fc == 0 && self.e_act[ey_idx(0, fr)] == 0)
                    || (fc == HODGE_W - 2 && self.e_act[ey_idx(HODGE_W - 1, fr)] == 0)
                {
                    reaches_ext = true;
                }

                // Traverse neighbor inactive plaquettes across absent interior primal edges
                if fr > 0 && self.e_act[ex_idx(fc, fr)] == 0 {
                    let nf = f - (HODGE_W - 1);
                    if self.hole_id[nf] == u16::MAX {
                        self.hole_id[nf] = 0xFFFE;
                        self.bfs_queue[tail] = nf as u16;
                        tail += 1;
                    }
                }
                if fr + 1 < HODGE_H - 1 && self.e_act[ex_idx(fc, fr + 1)] == 0 {
                    let nf = f + (HODGE_W - 1);
                    if self.hole_id[nf] == u16::MAX {
                        self.hole_id[nf] = 0xFFFE;
                        self.bfs_queue[tail] = nf as u16;
                        tail += 1;
                    }
                }
                if fc > 0 && self.e_act[ey_idx(fc, fr)] == 0 {
                    let nf = f - 1;
                    if self.hole_id[nf] == u16::MAX {
                        self.hole_id[nf] = 0xFFFE;
                        self.bfs_queue[tail] = nf as u16;
                        tail += 1;
                    }
                }
                if fc + 1 < HODGE_W - 1 && self.e_act[ey_idx(fc + 1, fr)] == 0 {
                    let nf = f + 1;
                    if self.hole_id[nf] == u16::MAX {
                        self.hole_id[nf] = 0xFFFE;
                        self.bfs_queue[tail] = nf as u16;
                        tail += 1;
                    }
                }
            }

            let tag = if reaches_ext {
                open_port_cnt += 1;
                0x8000u16 | (open_port_cnt as u16)
            } else {
                internal_void_cnt += 1;
                internal_void_cnt as u16
            };

            let mut qi = 0usize;
            while qi < tail {
                let ff = self.bfs_queue[qi] as usize;
                self.hole_id[ff] = tag;
                qi += 1;
            }

            if !reaches_ext {
                if internal_void_cnt == 1 {
                    primary_hole_size = tail;
                } else if tail > primary_hole_size {
                    primary_hole_size = tail;
                    let mut fi = 0usize;
                    while fi < HODGE_F {
                        if self.hole_id[fi] == 1 {
                            self.hole_id[fi] = tag;
                        } else if self.hole_id[fi] == tag {
                            self.hole_id[fi] = 1;
                        }
                        fi += 1;
                    }
                }
            }

            f0 += 1;
        }

        // Annotate boundary edges and vertices touching enclosed holes vs open ports
        r = 0;
        while r < HODGE_H - 1 {
            let mut c = 0usize;
            while c < HODGE_W - 1 {
                let f = f_idx(c, r);
                let hid = self.hole_id[f];
                if hid != 0 {
                    let is_port = (hid & 0x8000) != 0;
                    let ep = if is_port { EPORT_PORT_BND } else { EPORT_HOLE_BND };
                    let vc = if is_port { VCLASS_PORT_BND } else { VCLASS_HOLE_BND };
                    let edges = [
                        ex_idx(c, r),
                        ey_idx(c + 1, r),
                        ex_idx(c, r + 1),
                        ey_idx(c, r),
                    ];
                    let verts = [
                        v_idx(c, r),
                        v_idx(c + 1, r),
                        v_idx(c + 1, r + 1),
                        v_idx(c, r + 1),
                    ];
                    let mut k = 0usize;
                    while k < 4 {
                        let e = edges[k];
                        if self.e_act[e] != 0 && self.e_port[e] != EPORT_PORT_BND {
                            self.e_port[e] = ep;
                        }
                        let vk = verts[k];
                        if self.v_act[vk] != 0 {
                            self.v_class[vk] |= vc;
                        }
                        k += 1;
                    }
                }
                c += 1;
            }
            r += 1;
        }

        (
            true,
            v_act_cnt,
            e_act_cnt,
            f_act_cnt,
            comp_cnt,
            beta1,
            internal_void_cnt,
            open_port_cnt,
        )
    }

    #[inline(never)]
    fn solve_exact_pcg(&mut self, comp_cnt: usize) -> (bool, u32, f64) {
        self.phi_0.fill(0.0);
        self.cg_rhs.fill(0.0);
        self.cg_inv_diag.fill(0.0);

        let mut r = 0usize;
        while r < HODGE_H {
            let mut c = 0usize;
            while c < HODGE_W - 1 {
                let e = ex_idx(c, r);
                if self.e_act[e] != 0 {
                    let v0 = v_idx(c, r);
                    let v1 = v0 + 1;
                    let w = self.w1_edge[e];
                    let wo = w * self.omega[e];
                    self.cg_rhs[v0] -= wo;
                    self.cg_rhs[v1] += wo;
                    self.cg_inv_diag[v0] += w;
                    self.cg_inv_diag[v1] += w;
                }
                c += 1;
            }
            r += 1;
        }
        r = 0;
        while r < HODGE_H - 1 {
            let mut c = 0usize;
            while c < HODGE_W {
                let e = ey_idx(c, r);
                if self.e_act[e] != 0 {
                    let v0 = v_idx(c, r);
                    let v1 = v0 + HODGE_W;
                    let w = self.w1_edge[e];
                    let wo = w * self.omega[e];
                    self.cg_rhs[v0] -= wo;
                    self.cg_rhs[v1] += wo;
                    self.cg_inv_diag[v0] += w;
                    self.cg_inv_diag[v1] += w;
                }
                c += 1;
            }
            r += 1;
        }

        let mut v = 0usize;
        while v < HODGE_V {
            let d = self.cg_inv_diag[v];
            if self.v_act[v] != 0 && d > 1e-18 {
                self.cg_inv_diag[v] = 1.0 / d;
            } else {
                self.cg_inv_diag[v] = 0.0;
            }
            v += 1;
        }

        project_zero_mean(
            &mut self.cg_rhs,
            &self.v_act,
            &self.comp_id,
            &self.comp_count,
            &mut self.comp_sum,
            comp_cnt,
        );

        let mut b2 = 0.0f64;
        v = 0;
        while v < HODGE_V {
            let b = self.cg_rhs[v];
            b2 += b * b;
            self.cg_r[v] = b;
            self.cg_z[v] = self.cg_inv_diag[v] * b;
            v += 1;
        }
        let b_norm = crate::wasm_sqrt_f64(b2);
        if !b_norm.is_finite() {
            return (false, 0, f64::NAN);
        }
        if b_norm <= 1e-18 {
            self.omega_exact.fill(0.0);
            return (true, 0, 0.0);
        }

        project_zero_mean(
            &mut self.cg_z,
            &self.v_act,
            &self.comp_id,
            &self.comp_count,
            &mut self.comp_sum,
            comp_cnt,
        );

        let mut rz = 0.0f64;
        v = 0;
        while v < HODGE_V {
            self.cg_p[v] = self.cg_z[v];
            rz += self.cg_r[v] * self.cg_z[v];
            v += 1;
        }

        let mut iters = 0u32;
        let mut iter = 0usize;
        while iter < HODGE_PCG_MAX_ITERS {
            apply_l0(&self.cg_p, &mut self.cg_ap, &self.e_act, &self.w1_edge);
            let mut pap = 0.0f64;
            v = 0;
            while v < HODGE_V {
                if self.v_act[v] != 0 {
                    pap += self.cg_p[v] * self.cg_ap[v];
                }
                v += 1;
            }
            if !pap.is_finite() || pap <= 1e-30 * b2 {
                break;
            }
            let alpha = rz / pap;
            if !alpha.is_finite() {
                return (false, iters, f64::NAN);
            }
            v = 0;
            while v < HODGE_V {
                if self.v_act[v] != 0 {
                    self.phi_0[v] += alpha * self.cg_p[v];
                    self.cg_r[v] -= alpha * self.cg_ap[v];
                }
                v += 1;
            }
            project_zero_mean(
                &mut self.phi_0,
                &self.v_act,
                &self.comp_id,
                &self.comp_count,
                &mut self.comp_sum,
                comp_cnt,
            );
            project_zero_mean(
                &mut self.cg_r,
                &self.v_act,
                &self.comp_id,
                &self.comp_count,
                &mut self.comp_sum,
                comp_cnt,
            );

            iters = (iter + 1) as u32;
            let mut r2 = 0.0f64;
            v = 0;
            while v < HODGE_V {
                let rv = self.cg_r[v];
                r2 += rv * rv;
                self.cg_z[v] = self.cg_inv_diag[v] * rv;
                v += 1;
            }
            let rel = crate::wasm_sqrt_f64(r2) / b_norm;
            if rel <= HODGE_PCG_TOL {
                break;
            }

            project_zero_mean(
                &mut self.cg_z,
                &self.v_act,
                &self.comp_id,
                &self.comp_count,
                &mut self.comp_sum,
                comp_cnt,
            );

            let mut rz_new = 0.0f64;
            v = 0;
            while v < HODGE_V {
                rz_new += self.cg_r[v] * self.cg_z[v];
                v += 1;
            }
            if !rz_new.is_finite() || abs_f64(rz) <= 1e-30 * b2 {
                break;
            }
            let beta = rz_new / rz;
            v = 0;
            while v < HODGE_V {
                self.cg_p[v] = self.cg_z[v] + beta * self.cg_p[v];
                v += 1;
            }
            rz = rz_new;
            iter += 1;
        }

        // Certified true residual ||P(b - L_0 phi_0)||_2 / ||b||_2
        apply_l0(&self.phi_0, &mut self.cg_ap, &self.e_act, &self.w1_edge);
        v = 0;
        while v < HODGE_V {
            self.cg_r[v] = if self.v_act[v] != 0 {
                self.cg_rhs[v] - self.cg_ap[v]
            } else {
                0.0
            };
            v += 1;
        }
        project_zero_mean(
            &mut self.cg_r,
            &self.v_act,
            &self.comp_id,
            &self.comp_count,
            &mut self.comp_sum,
            comp_cnt,
        );
        let mut r_true2 = 0.0f64;
        v = 0;
        while v < HODGE_V {
            let rv = self.cg_r[v];
            r_true2 += rv * rv;
            v += 1;
        }
        let rel_true = crate::wasm_sqrt_f64(r_true2) / b_norm;

        // omega_exact = d_0 phi_0
        self.omega_exact.fill(0.0);
        r = 0;
        while r < HODGE_H {
            let mut c = 0usize;
            while c < HODGE_W - 1 {
                let e = ex_idx(c, r);
                if self.e_act[e] != 0 {
                    let v0 = v_idx(c, r);
                    self.omega_exact[e] = self.phi_0[v0 + 1] - self.phi_0[v0];
                }
                c += 1;
            }
            r += 1;
        }
        r = 0;
        while r < HODGE_H - 1 {
            let mut c = 0usize;
            while c < HODGE_W {
                let e = ey_idx(c, r);
                if self.e_act[e] != 0 {
                    let v0 = v_idx(c, r);
                    self.omega_exact[e] = self.phi_0[v0 + HODGE_W] - self.phi_0[v0];
                }
                c += 1;
            }
            r += 1;
        }

        (rel_true.is_finite(), iters, rel_true)
    }

    #[inline(never)]
    fn solve_coexact_pcg(&mut self) -> (bool, u32, f64) {
        self.psi_2.fill(0.0);
        self.cg_rhs.fill(0.0);
        self.cg_inv_diag.fill(0.0);

        apply_d1(&self.omega, &self.f_act, &mut self.cg_rhs[..HODGE_F]);

        let mut b2 = 0.0f64;
        let mut r = 0usize;
        while r < HODGE_H - 1 {
            let mut c = 0usize;
            while c < HODGE_W - 1 {
                let f = f_idx(c, r);
                if self.f_act[f] != 0 {
                    let diag = (1.0 / self.w1_edge[ex_idx(c, r)])
                        + (1.0 / self.w1_edge[ey_idx(c + 1, r)])
                        + (1.0 / self.w1_edge[ex_idx(c, r + 1)])
                        + (1.0 / self.w1_edge[ey_idx(c, r)]);
                    self.cg_inv_diag[f] = if diag > 1e-18 { 1.0 / diag } else { 0.0 };
                    let b = self.cg_rhs[f];
                    b2 += b * b;
                    self.cg_r[f] = b;
                    self.cg_z[f] = self.cg_inv_diag[f] * b;
                    self.cg_p[f] = self.cg_z[f];
                } else {
                    self.cg_r[f] = 0.0;
                    self.cg_z[f] = 0.0;
                    self.cg_p[f] = 0.0;
                }
                c += 1;
            }
            r += 1;
        }

        let b_norm = crate::wasm_sqrt_f64(b2);
        if !b_norm.is_finite() {
            return (false, 0, f64::NAN);
        }
        if b_norm <= 1e-18 {
            self.omega_coexact.fill(0.0);
            return (true, 0, 0.0);
        }

        let mut rz = 0.0f64;
        let mut f = 0usize;
        while f < HODGE_F {
            rz += self.cg_r[f] * self.cg_z[f];
            f += 1;
        }

        let mut iters = 0u32;
        let mut iter = 0usize;
        while iter < HODGE_PCG_MAX_ITERS {
            // Reuse cg_ap[0..HODGE_F] as p copy? No, copy cg_p[..HODGE_F] into psi_2 temporarily?
            // Keep psi_2 intact: overload psi_2 or copy cg_p into cg_ap? Wait: apply_w1_inv_d1_t takes &[f64; HODGE_F].
            // Use cg_ap[0..HODGE_F] to hold L_2 * p while passing cg_p[0..HODGE_F] directly!
            let p_arr: &[f64; HODGE_F] = match self.cg_p[..HODGE_F].try_into() {
                Ok(a) => a,
                Err(_) => return (false, iters, f64::NAN),
            };
            apply_w1_inv_d1_t(
                p_arr,
                &self.f_act,
                &self.e_act,
                &self.w1_edge,
                &mut self.omega_coexact,
            );
            apply_d1(&self.omega_coexact, &self.f_act, &mut self.cg_ap[..HODGE_F]);

            let mut pap = 0.0f64;
            f = 0;
            while f < HODGE_F {
                if self.f_act[f] != 0 {
                    pap += self.cg_p[f] * self.cg_ap[f];
                }
                f += 1;
            }
            if !pap.is_finite() || pap <= 1e-30 * b2 {
                break;
            }
            let alpha = rz / pap;
            if !alpha.is_finite() {
                return (false, iters, f64::NAN);
            }
            let mut r2 = 0.0f64;
            f = 0;
            while f < HODGE_F {
                if self.f_act[f] != 0 {
                    self.psi_2[f] += alpha * self.cg_p[f];
                    let rv = self.cg_r[f] - alpha * self.cg_ap[f];
                    self.cg_r[f] = rv;
                    r2 += rv * rv;
                }
                f += 1;
            }
            iters = (iter + 1) as u32;
            let rel = crate::wasm_sqrt_f64(r2) / b_norm;
            if rel <= HODGE_PCG_TOL {
                break;
            }

            let mut rz_new = 0.0f64;
            f = 0;
            while f < HODGE_F {
                if self.f_act[f] != 0 {
                    let zv = self.cg_inv_diag[f] * self.cg_r[f];
                    self.cg_z[f] = zv;
                    rz_new += self.cg_r[f] * zv;
                }
                f += 1;
            }
            if !rz_new.is_finite() || abs_f64(rz) <= 1e-30 * b2 {
                break;
            }
            let beta = rz_new / rz;
            f = 0;
            while f < HODGE_F {
                if self.f_act[f] != 0 {
                    self.cg_p[f] = self.cg_z[f] + beta * self.cg_p[f];
                }
                f += 1;
            }
            rz = rz_new;
            iter += 1;
        }

        // Compute omega_coexact = W_1^{-1} d_1^T psi_2 and certified true residual
        apply_w1_inv_d1_t(
            &self.psi_2,
            &self.f_act,
            &self.e_act,
            &self.w1_edge,
            &mut self.omega_coexact,
        );
        apply_d1(&self.omega_coexact, &self.f_act, &mut self.cg_ap[..HODGE_F]);

        let mut r_true2 = 0.0f64;
        f = 0;
        while f < HODGE_F {
            if self.f_act[f] != 0 {
                let d = self.cg_rhs[f] - self.cg_ap[f];
                r_true2 += d * d;
            }
            f += 1;
        }
        let rel_true = crate::wasm_sqrt_f64(r_true2) / b_norm;
        (rel_true.is_finite(), iters, rel_true)
    }

    #[inline(never)]
    fn reconstruct_and_detect_wakes(
        &mut self,
        dx: f64,
        dy: f64,
        internal_void_cnt: usize,
        open_port_cnt: usize,
    ) -> (bool, u32, f64, f64, f64, [f64; 4]) {
        self.planes.fill(0.0);

        let inv_dx = 1.0 / dx;
        let inv_dy = 1.0 / dy;
        let mut sum_ex_u = 0.0f64;
        let mut sum_ex_v = 0.0f64;
        let mut act_v_cnt = 0usize;

        let mut r = 0usize;
        while r < HODGE_H {
            let mut c = 0usize;
            while c < HODGE_W {
                let v = v_idx(c, r);
                if self.v_act[v] == 0 {
                    c += 1;
                    continue;
                }
                act_v_cnt += 1;

                let mut nx = 0usize;
                let mut ex_u_s = 0.0f64;
                let mut co_u_s = 0.0f64;
                let mut ha_u_s = 0.0f64;
                if c > 0 {
                    let e = ex_idx(c - 1, r);
                    if self.e_act[e] != 0 {
                        nx += 1;
                        ex_u_s += self.omega_exact[e];
                        co_u_s += self.omega_coexact[e];
                        ha_u_s += self.omega_harm[e];
                    }
                }
                if c + 1 < HODGE_W {
                    let e = ex_idx(c, r);
                    if self.e_act[e] != 0 {
                        nx += 1;
                        ex_u_s += self.omega_exact[e];
                        co_u_s += self.omega_coexact[e];
                        ha_u_s += self.omega_harm[e];
                    }
                }
                let (ex_u, co_u, ha_u) = if nx > 0 {
                    let sc = inv_dx / (nx as f64);
                    (ex_u_s * sc, co_u_s * sc, ha_u_s * sc)
                } else {
                    (0.0, 0.0, 0.0)
                };

                let mut ny = 0usize;
                let mut ex_v_s = 0.0f64;
                let mut co_v_s = 0.0f64;
                let mut ha_v_s = 0.0f64;
                if r > 0 {
                    let e = ey_idx(c, r - 1);
                    if self.e_act[e] != 0 {
                        ny += 1;
                        ex_v_s += self.omega_exact[e];
                        co_v_s += self.omega_coexact[e];
                        ha_v_s += self.omega_harm[e];
                    }
                }
                if r + 1 < HODGE_H {
                    let e = ey_idx(c, r);
                    if self.e_act[e] != 0 {
                        ny += 1;
                        ex_v_s += self.omega_exact[e];
                        co_v_s += self.omega_coexact[e];
                        ha_v_s += self.omega_harm[e];
                    }
                }
                let (ex_v, co_v, ha_v) = if ny > 0 {
                    let sc = inv_dy / (ny as f64);
                    (ex_v_s * sc, co_v_s * sc, ha_v_s * sc)
                } else {
                    (0.0, 0.0, 0.0)
                };

                let phi_val = self.phi_0[v];
                let p_ex_u = ex_u as f32;
                let p_ex_v = ex_v as f32;
                let p_co_u = co_u as f32;
                let p_co_v = co_v as f32;
                let p_ha_u = ha_u as f32;
                let p_ha_v = ha_v as f32;
                let p_phi = phi_val as f32;

                if !p_ex_u.is_finite()
                    || !p_ex_v.is_finite()
                    || !p_co_u.is_finite()
                    || !p_co_v.is_finite()
                    || !p_ha_u.is_finite()
                    || !p_ha_v.is_finite()
                    || !p_phi.is_finite()
                {
                    return (false, 0, 0.0, 0.0, 0.0, [0.0; 4]);
                }

                self.planes[HODGE_PLANE_EXACT_U * HODGE_V + v] = p_ex_u;
                self.planes[HODGE_PLANE_EXACT_V * HODGE_V + v] = p_ex_v;
                self.planes[HODGE_PLANE_COEXACT_U * HODGE_V + v] = p_co_u;
                self.planes[HODGE_PLANE_COEXACT_V * HODGE_V + v] = p_co_v;
                self.planes[HODGE_PLANE_HARM_U * HODGE_V + v] = p_ha_u;
                self.planes[HODGE_PLANE_HARM_V * HODGE_V + v] = p_ha_v;
                self.planes[HODGE_PLANE_PHI * HODGE_V + v] = p_phi;

                sum_ex_u += ex_u;
                sum_ex_v += ex_v;
                c += 1;
            }
            r += 1;
        }

        // Compute primary enclosed hole centroid when present
        let mut void_cx = 0.0f64;
        let mut void_cy = 0.0f64;
        if internal_void_cnt >= 1 {
            let mut cnt = 0usize;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            r = 0;
            while r < HODGE_H - 1 {
                let mut c = 0usize;
                while c < HODGE_W - 1 {
                    let f = f_idx(c, r);
                    if self.hole_id[f] == 1 {
                        sx += (c as f64 + 1.0) * dx;
                        sy += (r as f64 + 1.0) * dy;
                        cnt += 1;
                    }
                    c += 1;
                }
                r += 1;
            }
            if cnt > 0 {
                void_cx = sx / (cnt as f64);
                void_cy = sy / (cnt as f64);
            }
        }

        // Downstream wake / occupancy-gap hypotheses (observational velocity-deficit cue only)
        let mut wake_hyp_cnt = 0u32;
        let mut best_deficit = 0.0f64;
        let mut best_tag = 0u16;
        let mut best_bbox = [0.0f64; 4];

        if act_v_cnt > 0 && (internal_void_cnt > 0 || open_port_cnt > 0) {
            let mean_ex_u = sum_ex_u / (act_v_cnt as f64);
            let mean_ex_v = sum_ex_v / (act_v_cnt as f64);
            let inflow_mag =
                crate::wasm_sqrt_f64(mean_ex_u * mean_ex_u + mean_ex_v * mean_ex_v);

            if inflow_mag > 1e-5 {
                let dir_x = mean_ex_u / inflow_mag;
                let dir_y = mean_ex_v / inflow_mag;
                let h_step = 0.5 * (dx + dy);

                let use_holes = internal_void_cnt > 0;
                let n_tags = if use_holes {
                    internal_void_cnt
                } else {
                    open_port_cnt
                };

                let mut ti = 1usize;
                while ti <= n_tags {
                    let tag = if use_holes {
                        ti as u16
                    } else {
                        0x8000u16 | (ti as u16)
                    };
                    let mut n_faces = 0usize;
                    let mut sx = 0.0f64;
                    let mut sy = 0.0f64;
                    r = 0;
                    while r < HODGE_H - 1 {
                        let mut c = 0usize;
                        while c < HODGE_W - 1 {
                            if self.hole_id[f_idx(c, r)] == tag {
                                sx += (c as f64 + 1.0) * dx;
                                sy += (r as f64 + 1.0) * dy;
                                n_faces += 1;
                            }
                            c += 1;
                        }
                        r += 1;
                    }
                    if n_faces == 0 || (!use_holes && n_faces >= HODGE_F / 4) {
                        ti += 1;
                        continue;
                    }
                    let cx = sx / (n_faces as f64);
                    let cy = sy / (n_faces as f64);

                    let mut r_long = 0.5 * h_step;
                    let mut r_perp = 0.5 * h_step;
                    r = 0;
                    while r < HODGE_H - 1 {
                        let mut c = 0usize;
                        while c < HODGE_W - 1 {
                            if self.hole_id[f_idx(c, r)] == tag {
                                let px = (c as f64 + 1.0) * dx - cx;
                                let py = (r as f64 + 1.0) * dy - cy;
                                let al = abs_f64(px * dir_x + py * dir_y) + 0.5 * h_step;
                                let ap = abs_f64(-px * dir_y + py * dir_x) + 0.5 * h_step;
                                if al > r_long {
                                    r_long = al;
                                }
                                if ap > r_perp {
                                    r_perp = ap;
                                }
                            }
                            c += 1;
                        }
                        r += 1;
                    }

                    let down_reach = (6.0 * h_step).max(2.5 * r_long);
                    let mut up_sum = 0.0f64;
                    let mut up_n = 0usize;
                    let mut down_sum = 0.0f64;
                    let mut down_n = 0usize;

                    let mut vr = 0usize;
                    while vr < HODGE_H {
                        let mut vc = 0usize;
                        while vc < HODGE_W {
                            let v = v_idx(vc, vr);
                            if self.v_act[v] != 0 {
                                let vx = (vc as f64 + 0.5) * dx;
                                let vy = (vr as f64 + 0.5) * dy;
                                let s_l = (vx - cx) * dir_x + (vy - cy) * dir_y;
                                let s_p = abs_f64(-(vx - cx) * dir_y + (vy - cy) * dir_x);

                                let tu = (self.planes[HODGE_PLANE_EXACT_U * HODGE_V + v] as f64)
                                    + (self.planes[HODGE_PLANE_COEXACT_U * HODGE_V + v] as f64)
                                    + (self.planes[HODGE_PLANE_HARM_U * HODGE_V + v] as f64);
                                let tv = (self.planes[HODGE_PLANE_EXACT_V * HODGE_V + v] as f64)
                                    + (self.planes[HODGE_PLANE_COEXACT_V * HODGE_V + v] as f64)
                                    + (self.planes[HODGE_PLANE_HARM_V * HODGE_V + v] as f64);
                                let spd = crate::wasm_sqrt_f64(tu * tu + tv * tv);

                                if s_l >= -(r_long + 6.0 * h_step)
                                    && s_l <= -(r_long + 0.5 * h_step)
                                    && s_p <= 1.2 * r_perp
                                {
                                    up_sum += spd;
                                    up_n += 1;
                                } else if s_l > r_long
                                    && s_l <= r_long + down_reach
                                    && s_p <= r_perp
                                {
                                    down_sum += spd;
                                    down_n += 1;
                                }
                            }
                            vc += 1;
                        }
                        vr += 1;
                    }

                    if down_n >= 2 {
                        let u_up = if up_n > 0 {
                            up_sum / (up_n as f64)
                        } else {
                            inflow_mag
                        };
                        let u_down = down_sum / (down_n as f64);
                        let u_ref = if u_up > inflow_mag { u_up } else { inflow_mag };
                        let deficit = if u_ref > 1e-12 {
                            ((u_ref - u_down) / u_ref).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let fore_aft_asym = up_n < 2 || u_down < 0.88 * u_up;

                        if deficit >= 0.20 && fore_aft_asym {
                            wake_hyp_cnt += 1;
                            let mut bx0 = f64::INFINITY;
                            let mut by0 = f64::INFINITY;
                            let mut bx1 = f64::NEG_INFINITY;
                            let mut by1 = f64::NEG_INFINITY;
                            let mut any_def = false;

                            vr = 0;
                            while vr < HODGE_H {
                                let mut vc = 0usize;
                                while vc < HODGE_W {
                                    let v = v_idx(vc, vr);
                                    if self.v_act[v] != 0 {
                                        let vx = (vc as f64 + 0.5) * dx;
                                        let vy = (vr as f64 + 0.5) * dy;
                                        let s_l = (vx - cx) * dir_x + (vy - cy) * dir_y;
                                        let s_p = abs_f64(-(vx - cx) * dir_y + (vy - cy) * dir_x);
                                        if s_l > r_long && s_l <= r_long + down_reach && s_p <= r_perp
                                        {
                                            let tu = (self.planes[HODGE_PLANE_EXACT_U * HODGE_V + v]
                                                as f64)
                                                + (self.planes
                                                    [HODGE_PLANE_COEXACT_U * HODGE_V + v]
                                                    as f64)
                                                + (self.planes[HODGE_PLANE_HARM_U * HODGE_V + v]
                                                    as f64);
                                            let tv = (self.planes[HODGE_PLANE_EXACT_V * HODGE_V + v]
                                                as f64)
                                                + (self.planes
                                                    [HODGE_PLANE_COEXACT_V * HODGE_V + v]
                                                    as f64)
                                                + (self.planes[HODGE_PLANE_HARM_V * HODGE_V + v]
                                                    as f64);
                                            let spd = crate::wasm_sqrt_f64(tu * tu + tv * tv);
                                            if spd < 0.80 * u_ref {
                                                any_def = true;
                                                if vx < bx0 { bx0 = vx; }
                                                if vy < by0 { by0 = vy; }
                                                if vx > bx1 { bx1 = vx; }
                                                if vy > by1 { by1 = vy; }
                                            }
                                        }
                                    }
                                    vc += 1;
                                }
                                vr += 1;
                            }

                            if !any_def {
                                bx0 = cx;
                                by0 = cy;
                                bx1 = cx;
                                by1 = cy;
                            }

                            if tag == 1 || (best_tag != 1 && deficit > best_deficit) {
                                best_tag = tag;
                                best_deficit = deficit;
                                best_bbox = [bx0, by0, bx1, by1];
                                if internal_void_cnt == 0 {
                                    void_cx = cx;
                                    void_cy = cy;
                                }
                            }
                        }
                    }

                    ti += 1;
                }
            }
        }

        (true, wake_hyp_cnt, best_deficit, void_cx, void_cy, best_bbox)
    }
}
