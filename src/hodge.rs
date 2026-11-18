//! Bounded `40x22` Eulerian Geometric `2`-Complex Discrete Hodge Decomposition (`Organ 3` / `Gate I3`).
//!
//! Pure `#![no_std]`, zero host imports, zero heap allocation inside static `HodgeState`.

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
        self.telem[1] = keep_mode;
        self.telem[0] = if keep_mode == HODGE_MODE_OFF {
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
            self.telem[1] = sanitized;
            if self.telem[0] == HODGE_STATE_OFF {
                self.telem[0] = HODGE_STATE_STANDBY;
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
        self.telem[0]
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
        _export_field: &[f32; HODGE_V * 6],
        _cell_valid_mask: &[u8; HODGE_V],
        _grid_dx: f32,
        _grid_dy: f32,
        frame_idx: u32,
    ) -> bool {
        if self.mode == HODGE_MODE_OFF {
            self.telem[0] = HODGE_STATE_OFF;
            return false;
        }
        self.last_solve_frame = frame_idx;
        self.telem[0] = HODGE_STATE_STANDBY;
        false
    }

    #[inline(never)]
    pub fn solve_injected(&mut self, _grid_dx: f32, _grid_dy: f32) -> u32 {
        if self.mode == HODGE_MODE_OFF {
            self.telem[0] = HODGE_STATE_OFF;
            return 0;
        }
        self.telem[0] = HODGE_STATE_STANDBY;
        0
    }
}
