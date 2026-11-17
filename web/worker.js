// ============================================================================
// CHRONOS-SPECTRAL Dedicated V8 Web Worker Isolate
// Manages Zero-Allocation Wasm SIMD128 Arena Views, Pre-Assignment Shifted
// Lanczos Fiedler Eigensolver, Multi-Scenario Blind Pixel Testbench, and
// OffscreenCanvas Non-Causal Proof HUD & Split Raw/Eulerian PiP.
// ============================================================================

const MAX_FRAME_W = 640;
const MAX_FRAME_H = 360;
const MAX_FRAME_BYTES = MAX_FRAME_W * MAX_FRAME_H * 4;
const RING_LEN = 12;
const MAX_TARGETS = 16;
const GRAPH_NODES = 32;
const TELEM_PEAKS = 928;
const TELEM_EXCUR = 992;
const TELEM_KINETIC = 1120;
const TELEM_GLOBAL = 1248;
const TELEM_PERSIST = 1264;
const TELEM_PHYSICS_START = 1280;
const TELEM_PHYSICS_STRIDE = 16;
const TELEM_ENSEMBLE_START = 1536;
const TELEMETRY_LEN = 1568;
const MAX_PIP_BYTES = 64 * 64 * 4;

let resMode = 0; // 0: 320x180 SD (32x32 patch/PiP), 1: 640x360 HD (64x64 foveated patch/PiP)
let frameW = 320;
let frameH = 180;
let framePixels = frameW * frameH;
let frameBytes = framePixels * 4;
let pipDim = 32;
let resScale = 1;

let wasmExports = null;
let wasmMemory = null;

// Pre-allocated permanent views into 320-page (20 MB, up from 288-page Wave-1 baseline) static Wasm linear memory
const ringSlotU8 = new Array(RING_LEN);
const ringSlotImgData = new Array(RING_LEN);
const ecaOverlayBackingBuf = new Uint8ClampedArray(MAX_FRAME_BYTES);
let ecaResView = null;
let pipImgData = null;
let laplacianView = null;
let fiedlerView = null;
let telemView = null;
let slotMetaView = null;
const slotPipImgData = new Array(MAX_TARGETS).fill(null);

// CHRONOS-1000 Zero-ML Continuum & Micro-Swarm C-ABI Constants (parameterized grid)
const MAX_MICRO = 1536;
const MICRO_STRIDE = 8; // [track_id, pos_x, pos_y, vel_x, vel_y, state, body_id, ftle_val]
const MAX_RIGID_BODIES = 32;
const BODY_STRIDE = 8; // [body_id, member_count, cx, cy, vx, vy, omega_rad_s, se2_rmse_px]
const FIELD_GRID_W = 40;
const FIELD_GRID_H = 22;
const FIELD_CELLS = FIELD_GRID_W * FIELD_GRID_H; // 880
const FIELD_STRIDE = 6; // [u_px_s, v_px_s, vorticity_s1, divergence_s1, ftle_s1, valid_weight]
const SWARM_TELEM_LEN = 64;

let microTracksView = null;
let continuumFieldView = null;
let rigidBodiesView = null;
let modalFieldView = null;
let swarmTelemView = null;

let userSwarmMode = 0; // Persistent user selection (default Off per Astra-1 gate: 0=off, 1=auto, 2=forced)
let swarmMode = 0; // Active mode applied to Wasm (includes dense scenario/sample overrides)
let denseSwarmOverrideActive = false;
let chronos1000AbiValid = false;
let fieldOverlayMode = 0; // 0=off (tracks+hulls only), 1=velocity, 2=vorticity, 3=ftle, 4=modal_phase
let inspectedMicroTid = 0;
let lastInspectStatus = 'idle';
let lastInspectReqSeq = 0;
let lastInspectResSeq = 0;
let lastInspectReqX = null;
let lastInspectReqY = null;
let lastInspectReqNormX = null;
let lastInspectReqNormY = null;

function clearActiveInspectionSelection() {
  inspectedMicroTid = 0;
  lastInspectStatus = 'idle';
  lastInspectReqX = null;
  lastInspectReqY = null;
  lastInspectReqNormX = null;
  lastInspectReqNormY = null;
}
let lastFieldRenderMs = 0.0;
let lastSwarmCopyMs = 0.0;
let lastFieldBackend = 'none';
let lastDtSecUsed = 1.0 / 30.0;
let ingestElapsedMs = 0.0;
let latestCausalTimeMs = 0.0;
let latestCausalPtsMs = -1.0;

function hasChronos1000Exports() {
  return Boolean(
    chronos1000AbiValid &&
    wasmExports &&
    typeof wasmExports.chronos_get_micro_tracks_ptr === 'function' &&
    typeof wasmExports.chronos_get_continuum_field_ptr === 'function' &&
    typeof wasmExports.chronos_get_modal_field_ptr === 'function' &&
    typeof wasmExports.chronos_get_rigid_bodies_ptr === 'function' &&
    typeof wasmExports.chronos_get_swarm_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_set_swarm_mode === 'function' &&
    typeof wasmExports.chronos_get_swarm_mode === 'function' &&
    typeof wasmExports.chronos_inspect_field_point === 'function' &&
    typeof wasmExports.chronos_set_focus_target === 'function'
  );
}

function clearChronos1000Views() {
  chronos1000AbiValid = false;
  microTracksView = null;
  continuumFieldView = null;
  rigidBodiesView = null;
  modalFieldView = null;
  swarmTelemView = null;
}

function refreshChronos1000Views() {
  if (
    !wasmMemory ||
    !wasmExports ||
    typeof wasmExports.chronos_get_micro_tracks_ptr !== 'function' ||
    typeof wasmExports.chronos_get_continuum_field_ptr !== 'function' ||
    typeof wasmExports.chronos_get_modal_field_ptr !== 'function' ||
    typeof wasmExports.chronos_get_rigid_bodies_ptr !== 'function' ||
    typeof wasmExports.chronos_get_swarm_telemetry_ptr !== 'function' ||
    typeof wasmExports.chronos_set_swarm_mode !== 'function' ||
    typeof wasmExports.chronos_get_swarm_mode !== 'function' ||
    typeof wasmExports.chronos_inspect_field_point !== 'function' ||
    typeof wasmExports.chronos_set_focus_target !== 'function'
  ) {
    clearChronos1000Views();
    return false;
  }
  const buf = wasmMemory.buffer;
  const specs = [
    { ptr: wasmExports.chronos_get_micro_tracks_ptr(), len: MAX_MICRO * MICRO_STRIDE },
    { ptr: wasmExports.chronos_get_continuum_field_ptr(), len: FIELD_CELLS * FIELD_STRIDE },
    { ptr: wasmExports.chronos_get_modal_field_ptr(), len: FIELD_CELLS },
    { ptr: wasmExports.chronos_get_rigid_bodies_ptr(), len: MAX_RIGID_BODIES * BODY_STRIDE },
    { ptr: wasmExports.chronos_get_swarm_telemetry_ptr(), len: SWARM_TELEM_LEN },
  ];
  for (let i = 0; i < specs.length; i++) {
    const p = specs[i].ptr;
    const byteLen = specs[i].len * 4;
    if (!Number.isInteger(p) || p <= 0 || (p & 3) !== 0 || p + byteLen > buf.byteLength) {
      clearChronos1000Views();
      return false;
    }
    for (let j = i + 1; j < specs.length; j++) {
      const q = specs[j].ptr;
      const qByteLen = specs[j].len * 4;
      if (p < q + qByteLen && q < p + byteLen) {
        clearChronos1000Views();
        return false;
      }
    }
  }
  microTracksView = new Float32Array(buf, specs[0].ptr, specs[0].len);
  continuumFieldView = new Float32Array(buf, specs[1].ptr, specs[1].len);
  modalFieldView = new Float32Array(buf, specs[2].ptr, specs[2].len);
  rigidBodiesView = new Float32Array(buf, specs[3].ptr, specs[3].len);
  swarmTelemView = new Float32Array(buf, specs[4].ptr, specs[4].len);
  chronos1000AbiValid = true;
  wasmExports.chronos_set_swarm_mode(swarmMode);
  return true;
}

function isChronos1000Scenario(sc) {
  return sc === 'vortex_swarm_1200' || sc === 'murmuration_1200' || sc === 'modal_beam_1152';
}

// Pre-allocated ring-aligned snapshot buffers so t-effK HUD renders stay locked to dispSlot
const swarmRingValid = new Uint8Array(RING_LEN);
const ringIngestFrameId = new Int32Array(RING_LEN).fill(-1);
const ringIngestTimeMs = new Float64Array(RING_LEN).fill(-1.0);
const ringIngestPtsMs = new Float64Array(RING_LEN).fill(-1.0);
const swarmRingMicro = Array.from({ length: RING_LEN }, () => new Float32Array(MAX_MICRO * MICRO_STRIDE));
const swarmRingField = Array.from({ length: RING_LEN }, () => new Float32Array(FIELD_CELLS * FIELD_STRIDE));
const swarmRingBodies = Array.from({ length: RING_LEN }, () => new Float32Array(MAX_RIGID_BODIES * BODY_STRIDE));
const swarmRingModal = Array.from({ length: RING_LEN }, () => new Float32Array(FIELD_CELLS));
const swarmRingTelem = Array.from({ length: RING_LEN }, () => new Float32Array(SWARM_TELEM_LEN));

// OffscreenCanvases & 2D contexts (pre-allocated once at boot)
let hudCanvas = null;
let hudCtx = null;
let pipCanvas = null;
let pipCtx = null;
let hilbertCanvas = null;
let hilbertCtx = null;
let spectralCanvas = null;
let spectralCtx = null;
let persistenceCanvas = null;
let persistenceCtx = null;
let physicsPhaseCanvas = null;
let physicsPhaseCtx = null;
const slotPipCanvases = new Array(MAX_TARGETS).fill(null);
const slotPipCtxs = new Array(MAX_TARGETS).fill(null);

// Pre-allocated Per-Slot Sticky Hold-on-Loss & EMA-Smoothed Roster State (zero runtime GC)
const SLOT_HOLD_FRAMES = 48; // ~1.5s graceful hold when a target exits or briefly drops
const PINNED_HOLD_FRAMES = 180; // Match Wasm 180-frame user-pin retention window
const slotHoldLeft = new Int16Array(MAX_TARGETS);
const slotHoldTid = new Int32Array(MAX_TARGETS);
const slotIdleDrawn = new Uint8Array(MAX_TARGETS);
const slotHoldVeilDrawn = new Uint8Array(MAX_TARGETS);
const slotSpeedPillOn = new Uint8Array(MAX_TARGETS);
const slotSmoothedSpeed = new Float32Array(MAX_TARGETS);
const slotSmoothedHeading = new Float32Array(MAX_TARGETS);
const slotSmoothedOmega = new Float32Array(MAX_TARGETS);
const slotSmoothedEca = new Float32Array(MAX_TARGETS);
const slotLastAgeFrames = new Int32Array(MAX_TARGETS);
const slotLastCoast = new Int32Array(MAX_TARGETS);
const slotLastReacq = new Int32Array(MAX_TARGETS);
const slotLastVal = new Int32Array(MAX_TARGETS);
const slotLastSigHi = new Int32Array(MAX_TARGETS);
const slotLastSigLo = new Int32Array(MAX_TARGETS);
const slotLastFreqHz = new Float32Array(MAX_TARGETS);
const slotLastMetricValid = new Float32Array(MAX_TARGETS);
const slotLastCalSegId = new Int32Array(MAX_TARGETS).fill(-1);
const slotLastPivotValid = new Uint8Array(MAX_TARGETS);
const slotLastPendSwing = new Float32Array(MAX_TARGETS);
const slotLastPendRad = new Float32Array(MAX_TARGETS);
let camPanStreakWorker = 0;

// Internal staging & simulation OffscreenCanvases
let dispStageCanvas = null;
let dispStageCtx = null;
let pipStageCanvas = null;
let pipStageCtx = null;
let simCanvas = null;
let simCtx = null;
let ecaOverlayImgData = null;
let ecaOverlayBuf = null;
let ecaStageCanvas = null;
let ecaStageCtx = null;

// Runtime state
let running = false;
let sourceMode = 'testbench'; // 'testbench' | 'webcam' | 'video'
let benchScenario = 'pillar_crossing'; // 'pillar_crossing' | 'short_gap_bridge' | 'ballistic_cascade' | 'harmonic_tremor' | 'percussion_ricochet'
let showEcaOverlay = false;
let showKalmanBaseline = true;
let showHeroTargets = true;
let showAllOverlays = false;
let lastOverlaySyncSeq = 0;
let strobeEnabled = true;
const cropNorm = new Float64Array([0.0, 0.0, 1.0, 1.0]); // [x, y, w, h] in [0..1]

let lookaheadK = 8;
let ecaAdaptRate = 0.035;
let fistaLambda = 1.45;
let eulerianGain = 25.0;
let phaseOffsetRad = -Math.PI * 0.5;
let labMode = 0; // 0: Mode A (Ballistic/Bridge), 1: Mode B (Tremor/Reflex), 2: Mode C (Impulses)
let audioOutLatencySec = 0.0;

// Timing & Rolling Benchmark Telemetry (pre-allocated ring + median sort buffer)
const TIME_HIST_LEN = 120;
const wasmTimeHist = new Float32Array(TIME_HIST_LEN);
const wasmTimeSortBuf = new Float32Array(TIME_HIST_LEN);
let timeHistIdx = 0;
let lastRafTs = 0;
let smoothFps = 60.0;
let sourceFps = 60.0;
let tickCounter = 0;
let strobeFlashAlpha = 0.0;
const padFlashAlpha = new Float32Array([0, 0, 0, 0]);
let totalBridgedEvents = 0;
let totalStrikeEvents = 0;

// Pre-allocated Mode C Localized Impact Shockwave Pool (synchronized to t - K presentation)
const SHOCK_POOL = 24;
const shockTriggerFrame = new Int32Array(SHOCK_POOL).fill(-1);
const shockX = new Float32Array(SHOCK_POOL);
const shockY = new Float32Array(SHOCK_POOL);
const shockZone = new Uint8Array(SHOCK_POOL);
const shockVel = new Float32Array(SHOCK_POOL);
const shockInDeg = new Float32Array(SHOCK_POOL);
const shockDeflectDeg = new Float32Array(SHOCK_POOL);
const shockRestitution = new Float32Array(SHOCK_POOL);
const shockAge = new Int16Array(SHOCK_POOL).fill(-1);
let shockWriteHead = 0;
let lastStrikeDeflectDeg = 0.0;
let lastStrikeRestitution = 0.0;
let lastStrikeTid = 0;
let lastStrikeFrame = -1;
let lastStrikeValid = false;
let lastScatterDeg = 0.0;
let lastScatterFrame = -1;
const slotLastStrikeX = new Float32Array(MAX_TARGETS);
const slotLastStrikeY = new Float32Array(MAX_TARGETS);
const prevRectHeading = new Float32Array(MAX_TARGETS);
const prevRectTid = new Int32Array(MAX_TARGETS);
const prevRectFrame = new Int32Array(MAX_TARGETS).fill(-1);
const slotRectAngVel = new Float32Array(MAX_TARGETS);

// Pre-allocated Mode A Frozen Parabolic Apex Reticle & Live Accuracy Score Buffers
const apexLockTid = new Int32Array(MAX_TARGETS);
const apexLockPredX = new Float32Array(MAX_TARGETS);
const apexLockPredY = new Float32Array(MAX_TARGETS);
const apexLockMinY = new Float32Array(MAX_TARGETS);
const apexLockAtMinX = new Float32Array(MAX_TARGETS);
const apexLockErrPx = new Float32Array(MAX_TARGETS).fill(-1.0);
const apexLockHold = new Int16Array(MAX_TARGETS);
let lastScoredApexErrPx = -1.0;

// Hilbert & Spectral Scope circular histories (pre-allocated, including track ID continuity)
const SCOPE_HIST = 128;
const histTremorR = new Float32Array(SCOPE_HIST);
const histTremorH = new Float32Array(SCOPE_HIST);
const histPhaseValid = new Uint8Array(SCOPE_HIST);
const histStrobe = new Uint8Array(SCOPE_HIST);
const histPipTid = new Int32Array(SCOPE_HIST).fill(-1);
const histLambda2 = new Float32Array(SCOPE_HIST);
let scopeWriteIdx = 0;

// Pre-allocated Zero-GC Phase-Space Portrait & Conservation Scope Buffers (15 Hz OffscreenCanvas)
const PHYS_REGIME_NAMES = [
  'INERTIAL CV',
  'BALLISTIC',
  'BALLISTIC + AERO DRAG',
  'POWERED BOOST',
  'PARACHUTE / TERMINAL',
  'PENDULUM / HARMONIC',
  'MICRO-TREMOR',
  'IMPULSIVE COLLISIONAL',
  'CURVED CTRV',
];
const JUMP_CODE_NAMES = [
  'NONE',
  'GALILEO PEG JUMP',
  'BOOST BURNOUT',
  'PARACHUTE DEPLOY',
  'COLLISION IMPULSE',
];
const PHASE_HIST = 140;
const phaseHistX = new Float32Array(PHASE_HIST);
const phaseHistY = new Float32Array(PHASE_HIST);
const phaseHistVx = new Float32Array(PHASE_HIST);
const phaseHistVy = new Float32Array(PHASE_HIST);
const phaseHistE = new Float32Array(PHASE_HIST);
const phaseHistEnsK = new Float32Array(PHASE_HIST);
const phaseHistKurR = new Float32Array(PHASE_HIST);
const phaseHistKurN = new Uint8Array(PHASE_HIST);
const phaseHistRegime = new Uint8Array(PHASE_HIST);
const phaseHistJump = new Uint8Array(PHASE_HIST);
const phaseHistTid = new Int32Array(PHASE_HIST).fill(-1);
let phaseHistWrite = 0;
let phaseHistCount = 0;
let lastPhysicsPhaseRenderMs = 0.0;

// Pre-allocated HUD label collision stagger buffers (zero per-frame allocation)
const MAX_HUD_LABELS = 24;
const hudLabelX = new Float32Array(MAX_HUD_LABELS);
const hudLabelY = new Float32Array(MAX_HUD_LABELS);
const hudLabelW = new Float32Array(MAX_HUD_LABELS);

// Pre-allocated 44-frame (~1.5s) persistent per-track comet trail buffers (zero runtime GC)
const TRAIL_SLOTS = 64;
const TRAIL_LEN = 44;
const trailOwnerTid = new Int32Array(TRAIL_SLOTS).fill(-1);
const trailHead = new Uint8Array(TRAIL_SLOTS);
const trailCount = new Uint8Array(TRAIL_SLOTS);
const trailLastFrame = new Int32Array(TRAIL_SLOTS).fill(-1);
const trailX = new Float32Array(TRAIL_SLOTS * TRAIL_LEN);
const trailY = new Float32Array(TRAIL_SLOTS * TRAIL_LEN);

// Pre-allocated Complete Shifted Lanczos + Invariant-Subspace Restart Workspaces (up to 31 = N - 1)
const MAX_KRYLOV = 31;
const activeNodeIdx = new Int32Array(GRAPH_NODES);
const subL = new Float64Array(GRAPH_NODES * GRAPH_NODES);
const lanczosQ = new Float64Array(MAX_KRYLOV * GRAPH_NODES);
const lanczosAlpha = new Float64Array(MAX_KRYLOV);
const lanczosBeta = new Float64Array(MAX_KRYLOV);
const dDiag = new Float64Array(MAX_KRYLOV);
const eSub = new Float64Array(GRAPH_NODES);
const lanczosW = new Float64Array(GRAPH_NODES);
const zVec = new Float64Array(MAX_KRYLOV * MAX_KRYLOV);
const prevFiedlerTrack = new Float64Array(MAX_TARGETS);
const eigVecBuf = new Float64Array(GRAPH_NODES);

// Pre-allocated Zero-GC Frame-Accurate Telemetry Session Recorder (18,000 frames = 5 min @ 60 Hz, 56 cols)
const REC_COLS = 56;
const REC_MAX_FRAMES = 18000;
const REC_COLUMNS = [
  'frame_idx', 'elapsed_sec', 'frame_w', 'frame_h', 'lookahead_k', 'lab_mode',
  'wasm_tick_ms', 'active_targets', 'clutter_db', 'clutter_eval_ratio',
  'padic_swaps', 'oracle_swaps', 'kalman_swaps', 'in_pillar_collisions',
  'fiedler_lambda2', 'spectral_gap', 'fista_duality_gap',
  'primary_track_id', 'primary_sig_u32', 'primary_padic_val',
  'disp_raw_x', 'disp_raw_y', 'disp_fista_x', 'disp_fista_y',
  'disp_kalman_x', 'disp_kalman_y', 'disp_vis', 'disp_bridged',
  'tremor_r', 'tremor_h', 'tremor_phase_deg', 'tremor_freq_hz', 'tremor_amp',
  'phase_valid', 'strobe_fired', 'perc_mask', 'apex_ms', 'saccade_latency_ms',
  'heading_deg', 'speed_phys', 'ang_vel_deg_s', 'pend_swing_deg',
  'strike_deflect_deg', 'strike_speed_ratio', 'metric_x_m', 'metric_y_m',
  'cal_segment_id', 'metric_valid',
  'phys_regime', 'phys_regime_conf', 'ode_r2', 'aero_drag_beta',
  'thrust_accel', 'damping_zeta', 'kuramoto_order_r', 'lyapunov_div_rate',
];
const recSlab = new Float64Array(REC_MAX_FRAMES * REC_COLS);
let recActive = false;
let recHead = 0;
let recCount = 0;
let recTotalWritten = 0;
let recElapsedSec = 0.0;
let calSegmentId = 0;
let recCalSegments = [];

// Pre-allocated Zero-GC Multi-Minute Tracking Continuity & Custody Timeline State
const CONT_MAX_TRACKS = 128;
const CONT_MAX_SEGS = 24;
const CONT_HIST_BINS = 240;
const contTid = new Int32Array(CONT_MAX_TRACKS);
const contSlot = new Int16Array(CONT_MAX_TRACKS).fill(-1);
const contFirstSeq = new Int32Array(CONT_MAX_TRACKS);
const contLastSeq = new Int32Array(CONT_MAX_TRACKS);
const contTotalFrames = new Int32Array(CONT_MAX_TRACKS);
const contTotalSec = new Float32Array(CONT_MAX_TRACKS);
const contRelocks = new Int32Array(CONT_MAX_TRACKS);
const contSegCount = new Uint8Array(CONT_MAX_TRACKS);
const contSegStart = new Int32Array(CONT_MAX_TRACKS * CONT_MAX_SEGS);
const contSegEnd = new Int32Array(CONT_MAX_TRACKS * CONT_MAX_SEGS);
const contWasActive = new Uint8Array(CONT_MAX_TRACKS);
const contBinMaxActive = new Uint8Array(CONT_HIST_BINS);
const contBinFrames = new Uint32Array(CONT_HIST_BINS);
const contRankIdx = new Int32Array(CONT_MAX_TRACKS);
let contCount = 0;
let persistSampleSeq = 0;
let sessionElapsedSec = 0.0;
let lastPersistRenderMs = 0.0;
let contBinStrideFrames = 15;
let contBinHead = 0;

// ============================================================================
// Pre-Allocated Metric Ruler, Auto-g Scale, & 4-Point Plane Homography Engine
// Stored in normalized [0..1] viewport coordinates so SD <-> HD is automatic!
// ============================================================================
const GRAVITY_M_S2 = 9.80665;
const HOMOG_MIN_W = 0.14;
const AUTO_G_LOCK_CONF = 0.50;
const CORNER_TAGS = ['TL', 'TR', 'BR', 'BL'];
let calMode = 'none';       // 'none' | 'ruler' | 'auto_g' | 'homography'
let calProvenance = 'none'; // 'none' | 'assumed' | 'surveyed'
let calArmed = 'none';      // 'none' | 'ruler' | 'homography'
let calPtCount = 0;
let calDraftCount = 0;
let lastCalSyncSeq = 0;
let calRulerMeters = 1.0;
let calPlaneW = 2.84;
let calPlaneH = 1.42;
let calDraftRulerMeters = 1.0;
let calDraftPlaneW = 2.84;
let calDraftPlaneH = 1.42;
let calPresetName = '';
let calAutoGNormH = 0.0;    // scene_gravity_px_s2 / frameH (resolution-invariant)
let calAutoGLocked = false; // true only when >= 2 consistent free-fall arcs lock
let lastLambda2 = 0.0;
const calPtsNormX = new Float64Array(4);
const calPtsNormY = new Float64Array(4);
const calDraftPtsNormX = new Float64Array(4);
const calDraftPtsNormY = new Float64Array(4);
const calHomography = new Float64Array(9);    // H: (nx, ny, 1)^T -> ~ (X_m, Y_m, 1)^T
const calHomographyInv = new Float64Array(9); // H^-1: (X_m, Y_m, 1)^T -> ~ (nx, ny, 1)^T
let calHomographyValid = false;

// Pre-allocated solver & output scratchpad (zero runtime heap allocations)
const homMatA = new Float64Array(64);
const homVecB = new Float64Array(8);
const homSol = new Float64Array(8);
const homTargetX = new Float64Array(4);
const homTargetY = new Float64Array(4);
const homScratchMat = new Float64Array(9);
const homScratchInv = new Float64Array(9);
const calOutPt = new Float64Array(3);      // [Xm, Ym, metricValid: 1=m, 0=px, -1=out_of_plane]
const calOutNorm = new Float64Array(2);
const calOutVel = new Float64Array(3);     // [speedPhys, rectifiedHeadingDeg, metricValid]
const calOutStrike = new Float64Array(3);  // [rectInDeg, rectDeflectDeg, rectSpeedRatio]
const calOutScatter = new Float64Array(3); // [rectTh1Deg, rectTh2Deg, rectScatterDeg]
const calOutPend = new Float64Array(3);    // [swingDeg, radiusPhys, metricValid]

function wrapDeg180(deg) {
  let d = ((deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
  if (d === -180.0) d = 180.0;
  return d;
}

function invert3x3(m, out) {
  const a00 = m[0], a01 = m[1], a02 = m[2];
  const a10 = m[3], a11 = m[4], a12 = m[5];
  const a20 = m[6], a21 = m[7], a22 = m[8];

  const b01 = a22 * a11 - a12 * a21;
  const b11 = -a22 * a10 + a12 * a20;
  const b21 = a21 * a10 - a11 * a20;
  const det = a00 * b01 + a01 * b11 + a02 * b21;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-11) return false;
  const invDet = 1.0 / det;

  out[0] = b01 * invDet;
  out[1] = (-a22 * a01 + a02 * a21) * invDet;
  out[2] = (a12 * a01 - a02 * a11) * invDet;
  out[3] = b11 * invDet;
  out[4] = (a22 * a00 - a02 * a20) * invDet;
  out[5] = (-a12 * a00 + a02 * a10) * invDet;
  out[6] = b21 * invDet;
  out[7] = (-a21 * a00 + a01 * a20) * invDet;
  out[8] = (a11 * a00 - a01 * a10) * invDet;
  return true;
}

function solveHomography4Pt(planeW, planeH, ptsX = calPtsNormX, ptsY = calPtsNormY, commit = true) {
  const Wp = Math.max(0.01, planeW);
  const Hp = Math.max(0.01, planeH);

  for (let i = 0; i < 4; i++) {
    if (!Number.isFinite(ptsX[i]) || !Number.isFinite(ptsY[i])) return false;
  }

  // Enforce strictly convex clockwise TL -> TR -> BR -> BL screen-space quad (+x right, +y down)
  // and minimum normalized viewport area (>= 0.1% of frame, rejecting 1px/collinear/mirrored/bowtie quads).
  let signedArea2 = 0.0;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) & 3;
    const k = (i + 2) & 3;
    const ux = ptsX[j] - ptsX[i];
    const uy = ptsY[j] - ptsY[i];
    const vx = ptsX[k] - ptsX[j];
    const vy = ptsY[k] - ptsY[j];
    const cr = ux * vy - uy * vx;
    if (!Number.isFinite(cr) || cr <= 1e-5) return false;
    signedArea2 += ptsX[i] * ptsY[j] - ptsX[j] * ptsY[i];
  }
  if (!Number.isFinite(signedArea2) || signedArea2 < 2e-3) return false;

  // Closed-form projective rectangle basis solve (zero h33 = 1 singularity even when the
  // vanishing horizon passes directly through image (0, 0)):
  // Map unit square (u, v) = (X / Wp, Y / Hp) corners:
  //   BL (0,0) -> p3,  BR (1,0) -> lam2 * p2,  TL (0,1) -> lam0 * p0,  TR (1,1) -> lam1 * p1
  // where lam2 * (p2 - p1) + lam0 * (p0 - p1) = p3 - p1 and lam1 = lam2 + lam0 - 1.
  const dx2 = ptsX[2] - ptsX[1];
  const dy2 = ptsY[2] - ptsY[1];
  const dx0 = ptsX[0] - ptsX[1];
  const dy0 = ptsY[0] - ptsY[1];
  const sx = ptsX[3] - ptsX[1];
  const sy = ptsY[3] - ptsY[1];
  const det = dx2 * dy0 - dx0 * dy2;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-11) return false;

  const lam2 = (sx * dy0 - dx0 * sy) / det;
  const lam0 = (dx2 * sy - sx * dy2) / det;
  const lam1 = lam2 + lam0 - 1.0;
  if (!(lam0 > 1e-4 && lam1 > 1e-4 && lam2 > 1e-4)) return false;

  // Build H_inv (metric [X, Y, 1]^T -> w_px * [x, y, 1]^T) directly from [c0/Wp, c1/Hp, c2]:
  const invWp = 1.0 / Wp;
  const invHp = 1.0 / Hp;
  homScratchInv[0] = (lam2 * ptsX[2] - ptsX[3]) * invWp;
  homScratchInv[1] = (lam0 * ptsX[0] - ptsX[3]) * invHp;
  homScratchInv[2] = ptsX[3];
  homScratchInv[3] = (lam2 * ptsY[2] - ptsY[3]) * invWp;
  homScratchInv[4] = (lam0 * ptsY[0] - ptsY[3]) * invHp;
  homScratchInv[5] = ptsY[3];
  homScratchInv[6] = (lam2 - 1.0) * invWp;
  homScratchInv[7] = (lam0 - 1.0) * invHp;
  homScratchInv[8] = 1.0;

  if (!invert3x3(homScratchInv, homScratchMat)) return false;

  // Projective depth at corner i is proportional to 1 / lam_i > 0.
  // Align sign so all 4 corners have w_i > 0, then scale H so min(w_corners) = 1.0
  // (guarantees w >= 1.0 > HOMOG_MIN_W throughout the entire convex ground quad while
  // bounding extreme horizon-grazing foreshortening ratios <= 250x).
  let posCount = 0;
  let negCount = 0;
  for (let i = 0; i < 4; i++) {
    const w = homScratchMat[6] * ptsX[i] + homScratchMat[7] * ptsY[i] + homScratchMat[8];
    if (!Number.isFinite(w)) return false;
    if (w > 1e-8) posCount++;
    else if (w < -1e-8) negCount++;
  }
  if (negCount === 4) {
    for (let k = 0; k < 9; k++) homScratchMat[k] = -homScratchMat[k];
  } else if (posCount !== 4) {
    return false;
  }

  let minW = Infinity;
  let maxW = 0.0;
  for (let i = 0; i < 4; i++) {
    const w = homScratchMat[6] * ptsX[i] + homScratchMat[7] * ptsY[i] + homScratchMat[8];
    if (w < minW) minW = w;
    if (w > maxW) maxW = w;
  }
  if (!(minW > 1e-8) || maxW / minW > 250.0) return false;

  const invMinW = 1.0 / minW;
  for (let k = 0; k < 9; k++) homScratchMat[k] *= invMinW;
  if (!invert3x3(homScratchMat, homScratchInv)) return false;

  for (let i = 0; i < 4; i++) {
    const w = homScratchMat[6] * ptsX[i] + homScratchMat[7] * ptsY[i] + homScratchMat[8];
    if (w <= HOMOG_MIN_W) return false;
  }

  if (commit) {
    calHomography.set(homScratchMat);
    calHomographyInv.set(homScratchInv);
  }
  return true;
}

function isCameraMotionSuspendingCal() {
  return !!(telemView && sourceMode !== 'testbench' && Math.round(telemView[TELEM_GLOBAL + 9]) > 0);
}

function getEffectivePxPerMeter() {
  if (isCameraMotionSuspendingCal()) return 0.0;
  if (calMode === 'ruler' && calPtCount >= 2) {
    const dx = (calPtsNormX[1] - calPtsNormX[0]) * frameW;
    const dy = (calPtsNormY[1] - calPtsNormY[0]) * frameH;
    const spanPx = Math.hypot(dx, dy);
    return spanPx > 1.0 ? spanPx / Math.max(0.01, calRulerMeters) : 0.0;
  }
  if (calMode === 'auto_g') {
    if (calAutoGLocked && calAutoGNormH > 0.05) {
      return (calAutoGNormH * frameH) / GRAVITY_M_S2;
    }
  }
  return 0.0;
}

function isMetricCalibrated() {
  if (isCameraMotionSuspendingCal()) return false;
  if (calMode === 'homography') return calHomographyValid;
  if (calMode === 'ruler' || calMode === 'auto_g') return getEffectivePxPerMeter() > 0.5;
  return false;
}

function safeFinite(v, fallback = 0.0) {
  return Number.isFinite(v) ? v : fallback;
}

function computeLocalHomographyPpm(px, py) {
  if (isCameraMotionSuspendingCal()) return -1.0;
  if (calMode !== 'homography' || !calHomographyValid) return 0.0;
  const nx = px / frameW;
  const ny = py / frameH;
  const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
  if (!(w > HOMOG_MIN_W)) return -1.0;
  const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
  const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;
  const j00 = ((calHomography[0] - X * calHomography[6]) / w) / frameW;
  const j01 = ((calHomography[1] - X * calHomography[7]) / w) / frameH;
  const j10 = ((calHomography[3] - Y * calHomography[6]) / w) / frameW;
  const j11 = ((calHomography[4] - Y * calHomography[7]) / w) / frameH;
  const det = Math.abs(j00 * j11 - j01 * j10);
  if (!Number.isFinite(det) || det < 1e-12) return 0.0;
  const ppm = 1.0 / Math.sqrt(det);
  return Number.isFinite(ppm) ? ppm : 0.0;
}

function formatCompactPhysicsVal(v, digits = 2) {
  if (!Number.isFinite(v) || Math.abs(v) < 1e-9) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e4) return v.toExponential(2);
  return v.toFixed(digits);
}

function formatSymbolicOdeString(p) {
  if (!p.hasPrimary) {
    if (p.kuramotoOscCount >= 2 && p.kuramotoOrderR > 0.05) {
      return `Σθ̇ᵢ = ω + (K/N)Σsin(θⱼ-θᵢ) [R=${p.kuramotoOrderR.toFixed(2)}, N=${p.kuramotoOscCount}]`;
    }
    return 'Awaiting confirmed target dynamics...';
  }
  const betaStr = `${formatCompactPhysicsVal(p.betaDisp, 3)} ${p.betaUnit}`;
  const thrustStr = `${formatCompactPhysicsVal(p.thrustDisp, 2)} ${p.thrustUnit}`;
  const kappaStr = `${formatCompactPhysicsVal(p.curvatureDisp, 3)} ${p.curvatureUnit}`;
  const zetaStr = p.dampingZeta.toFixed(3);
  const qStr = p.qualityFactorQ >= 998.5 ? '>=999' : (p.qualityFactorQ > 0 ? p.qualityFactorQ.toFixed(1) : '--');
  const tStr = p.ellipticPeriodS > 1e-3 ? `${p.ellipticPeriodS.toFixed(3)}s` : '--';

  switch (p.physRegime) {
    case 0:
      return `dr/dt = v (${p.speedStr}), d²r/dt² ≈ 0 [R²=${p.odeR2.toFixed(2)}]`;
    case 1:
      return `d²y/dt² = +g (${p.gStr}) [R²=${p.odeR2.toFixed(2)}]`;
    case 2:
      return `d²r/dt² = g - β‖v‖v (β=${betaStr}, g=${p.gStr})`;
    case 3:
      return `d²r/dt² = a_th + g - β‖v‖v (a_th=${thrustStr}, β=${betaStr})`;
    case 4:
      return `dv_y/dt = g - βv_y² (β=${betaStr}, g=${p.gStr})`;
    case 5:
      return `θ̈ + (g/L)sinθ + 2ζω₀θ̇ = 0 (T=${tStr}, ζ=${zetaStr}, Q=${qStr})`;
    case 6: {
      const fTag = p.tremorFreqHz > 0.05 ? `f=${p.tremorFreqHz.toFixed(2)}Hz, ` : '';
      return `q̈ + 2ζω₀q̇ + ω₀²q + αq³ = 0 (${fTag}ζ=${zetaStr}, Q=${qStr})`;
    }
    case 7: {
      const eStr = p.normalRestitutionE > 0.01 ? p.normalRestitutionE.toFixed(3) : '--';
      const mTag = p.lastPairMassRatio > 0 && p.lastPairFrame > 0 ? `, m₂/m₁=${p.lastPairMassRatio.toFixed(2)}` : '';
      return `Δvₙ⁺ = -eₙΔvₙ⁻ (eₙ=${eStr}${mTag})`;
    }
    case 8:
      return `ψ̇ = ω (${p.angVelDegS.toFixed(1)}°/s), d²r/dt² = ω×v (κ=${kappaStr})`;
    default:
      return `dr/dt = v (${p.speedStr})`;
  }
}

function mapPixelToMetric(px, py) {
  if (isCameraMotionSuspendingCal()) {
    calOutPt[0] = 0.0;
    calOutPt[1] = 0.0;
    calOutPt[2] = -1.0;
    return calOutPt;
  }
  if (calMode === 'homography' && calHomographyValid) {
    const nx = px / frameW;
    const ny = py / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      calOutPt[0] = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      calOutPt[1] = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;
      calOutPt[2] = 1.0;
      return calOutPt;
    }
    calOutPt[0] = 0.0;
    calOutPt[1] = 0.0;
    calOutPt[2] = -1.0;
    return calOutPt;
  }
  const ppm = getEffectivePxPerMeter();
  if (ppm > 0.5) {
    if (calMode === 'ruler' && calPtCount >= 2) {
      const ox = calPtsNormX[0] * frameW;
      const oy = calPtsNormY[0] * frameH;
      calOutPt[0] = (px - ox) / ppm;
      calOutPt[1] = (oy - py) / ppm;
    } else {
      calOutPt[0] = px / ppm;
      calOutPt[1] = (frameH - py) / ppm;
    }
    calOutPt[2] = 1.0;
    return calOutPt;
  }
  calOutPt[0] = 0.0;
  calOutPt[1] = 0.0;
  calOutPt[2] = 0.0;
  return calOutPt;
}

function mapMetricToNorm(Xm, Ym) {
  const w = calHomographyInv[6] * Xm + calHomographyInv[7] * Ym + calHomographyInv[8];
  const safeW = Math.abs(w) < 1e-5 ? 1e-5 : w;
  calOutNorm[0] = (calHomographyInv[0] * Xm + calHomographyInv[1] * Ym + calHomographyInv[2]) / safeW;
  calOutNorm[1] = (calHomographyInv[3] * Xm + calHomographyInv[4] * Ym + calHomographyInv[5]) / safeW;
  return calOutNorm;
}

function mapVelocityToPhysical(px, py, speedPxS, headingDeg) {
  if (isCameraMotionSuspendingCal()) {
    calOutVel[0] = 0.0;
    calOutVel[1] = headingDeg;
    calOutVel[2] = -1.0;
    return calOutVel;
  }
  if (calMode === 'homography' && calHomographyValid) {
    const nx = px / frameW;
    const ny = py / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;
      const rad = headingDeg * (Math.PI / 180.0);
      const dnx = (speedPxS * Math.cos(rad)) / frameW;
      const dny = (-speedPxS * Math.sin(rad)) / frameH;
      const dw = calHomography[6] * dnx + calHomography[7] * dny;
      const dX = ((calHomography[0] * dnx + calHomography[1] * dny) - X * dw) / w;
      const dY = ((calHomography[3] * dnx + calHomography[4] * dny) - Y * dw) / w;
      const spdM = Math.hypot(dX, dY);
      calOutVel[0] = spdM;
      calOutVel[1] = spdM > 1e-4 ? Math.atan2(dY, dX) * (180.0 / Math.PI) : headingDeg;
      calOutVel[2] = 1.0;
      return calOutVel;
    }
    // Never fall back to raw px/s disguised as m/s above horizon / outside valid frustum!
    calOutVel[0] = 0.0;
    calOutVel[1] = headingDeg;
    calOutVel[2] = -1.0;
    return calOutVel;
  }
  const ppm = getEffectivePxPerMeter();
  if (ppm > 0.5) {
    calOutVel[0] = speedPxS / ppm;
    calOutVel[1] = headingDeg;
    calOutVel[2] = 1.0;
  } else {
    calOutVel[0] = speedPxS;
    calOutVel[1] = headingDeg;
    calOutVel[2] = 0.0;
  }
  return calOutVel;
}

function rectifyStrikeKinematics(sxPx, syPx, inDeg, deflectDeg, restitution) {
  if (calMode === 'homography' && calHomographyValid && !isCameraMotionSuspendingCal() && restitution > 0.01) {
    const nx = sxPx / frameW;
    const ny = syPx / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;

      const inRad = inDeg * (Math.PI / 180.0);
      const inDnx = Math.cos(inRad) / frameW;
      const inDny = -Math.sin(inRad) / frameH;
      const inDw = calHomography[6] * inDnx + calHomography[7] * inDny;
      const inDX = ((calHomography[0] * inDnx + calHomography[1] * inDny) - X * inDw) / w;
      const inDY = ((calHomography[3] * inDnx + calHomography[4] * inDny) - Y * inDw) / w;

      const outDeg = inDeg + deflectDeg;
      const outRad = outDeg * (Math.PI / 180.0);
      const outDnx = (restitution * Math.cos(outRad)) / frameW;
      const outDny = (-restitution * Math.sin(outRad)) / frameH;
      const outDw = calHomography[6] * outDnx + calHomography[7] * outDny;
      const outDX = ((calHomography[0] * outDnx + calHomography[1] * outDny) - X * outDw) / w;
      const outDY = ((calHomography[3] * outDnx + calHomography[4] * outDny) - Y * outDw) / w;

      const inNorm = Math.hypot(inDX, inDY);
      const outNorm = Math.hypot(outDX, outDY);
      if (inNorm > 1e-6 && outNorm > 1e-6) {
        const rInDeg = Math.atan2(inDY, inDX) * (180.0 / Math.PI);
        const rOutDeg = Math.atan2(outDY, outDX) * (180.0 / Math.PI);
        calOutStrike[0] = rInDeg;
        calOutStrike[1] = wrapDeg180(rOutDeg - rInDeg);
        calOutStrike[2] = Math.min(2.5, outNorm / inNorm);
        return calOutStrike;
      }
    }
  }
  calOutStrike[0] = inDeg;
  calOutStrike[1] = deflectDeg;
  calOutStrike[2] = restitution;
  return calOutStrike;
}

function rectifyScatterKinematics(scxPx, scyPx, th1Deg, th2Deg, rawScatterDeg) {
  if (calMode === 'homography' && calHomographyValid && !isCameraMotionSuspendingCal()) {
    const nx = scxPx / frameW;
    const ny = scyPx / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;

      const r1 = th1Deg * (Math.PI / 180.0);
      const dnx1 = Math.cos(r1) / frameW;
      const dny1 = -Math.sin(r1) / frameH;
      const dw1 = calHomography[6] * dnx1 + calHomography[7] * dny1;
      const dX1 = ((calHomography[0] * dnx1 + calHomography[1] * dny1) - X * dw1) / w;
      const dY1 = ((calHomography[3] * dnx1 + calHomography[4] * dny1) - Y * dw1) / w;

      const r2 = th2Deg * (Math.PI / 180.0);
      const dnx2 = Math.cos(r2) / frameW;
      const dny2 = -Math.sin(r2) / frameH;
      const dw2 = calHomography[6] * dnx2 + calHomography[7] * dny2;
      const dX2 = ((calHomography[0] * dnx2 + calHomography[1] * dny2) - X * dw2) / w;
      const dY2 = ((calHomography[3] * dnx2 + calHomography[4] * dny2) - Y * dw2) / w;

      if (Math.hypot(dX1, dY1) > 1e-6 && Math.hypot(dX2, dY2) > 1e-6) {
        const a1 = Math.atan2(dY1, dX1) * (180.0 / Math.PI);
        const a2 = Math.atan2(dY2, dX2) * (180.0 / Math.PI);
        calOutScatter[0] = a1;
        calOutScatter[1] = a2;
        calOutScatter[2] = Math.abs(wrapDeg180(a1 - a2));
        return calOutScatter;
      }
    }
  }
  calOutScatter[0] = th1Deg;
  calOutScatter[1] = th2Deg;
  calOutScatter[2] = rawScatterDeg;
  return calOutScatter;
}

function computePendulumKinematics(fx, fy, pivX, pivY) {
  const dx = fx - pivX;
  const dyAbs = Math.max(1e-3, Math.abs(fy - pivY));
  const swingDeg = Math.atan2(dx, dyAbs) * (180.0 / Math.PI);
  const rPx = Math.hypot(dx, fy - pivY);
  if (calMode === 'homography' && calHomographyValid) {
    const p0 = mapPixelToMetric(pivX, pivY);
    const p0x = p0[0], p0y = p0[1], p0v = p0[2];
    const p1 = mapPixelToMetric(fx, fy);
    calOutPend[0] = swingDeg;
    if (p0v > 0.5 && p1[2] > 0.5) {
      calOutPend[1] = Math.hypot(p1[0] - p0x, p1[1] - p0y);
      calOutPend[2] = 1.0;
    } else {
      calOutPend[1] = 0.0;
      calOutPend[2] = -1.0;
    }
    return calOutPend;
  }
  const ppm = getEffectivePxPerMeter();
  calOutPend[0] = swingDeg;
  if (ppm > 0.5) {
    calOutPend[1] = rPx / ppm;
    calOutPend[2] = 1.0;
  } else {
    calOutPend[1] = rPx;
    calOutPend[2] = 0.0;
  }
  return calOutPend;
}

function formatCalibrationSummary(overrideNotice) {
  if (overrideNotice) return overrideNotice;
  if (calArmed === 'ruler') {
    return `ARMED 2-PT RULER (${calDraftCount}/2 CLICKS · L=${calDraftRulerMeters.toFixed(2)} m · ESC PRESERVES PRIOR)`;
  }
  if (calArmed === 'homography') {
    return `ARMED 4-PT PLANE (${calDraftCount}/4 [${CORNER_TAGS[Math.min(3, calDraftCount)]}] · ${calDraftPlaneW.toFixed(2)}×${calDraftPlaneH.toFixed(2)} m · ESC PRESERVES)`;
  }
  if (isCameraMotionSuspendingCal() && calMode !== 'none') {
    return 'CAMERA MOTION — IMAGE-PLANE CALIBRATION SUSPENDED';
  }
  const provTag = calProvenance === 'assumed' ? 'ASSUMED SCALE' : 'SURVEYED';
  if (calMode === 'ruler') {
    const ppm = getEffectivePxPerMeter();
    return `2-PT RULER [${provTag}]: ${calRulerMeters.toFixed(2)} m (${Math.round(ppm)} px/m)`;
  }
  if (calMode === 'auto_g') {
    const ppm = getEffectivePxPerMeter();
    const arcs = telemView ? Math.round(telemView[TELEM_GLOBAL + 11]) : 0;
    if (calAutoGLocked && calAutoGNormH > 0.05) {
      const gPx = Math.round(calAutoGNormH * frameH);
      return `AUTO-g LOCKED (${Math.max(2, arcs)} ARCS): ĝ=${gPx} px/s² → ${Math.round(ppm)} px/m`;
    }
    if (telemView && telemView[TELEM_GLOBAL + 0] > 50.0) {
      return `AUTO-g PROVISIONAL (${Math.max(1, arcs)}/2 ARCS · ĝ=${Math.round(telemView[TELEM_GLOBAL + 0])} px/s² · AWAITING 2ND ARC)`;
    }
    return 'AUTO-g ARMED · AWAITING >= 2 PARABOLIC FLIGHT ARCS (g = 9.81 m/s²)';
  }
  if (calMode === 'homography' && calHomographyValid) {
    return `4-PT HOMOGRAPHY [${provTag}]: ${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m [RECTIFIED]`;
  }
  return 'UNCALIBRATED · PIXEL UNITS (px · px/s)';
}

function getCalibrationSnapshot() {
  return {
    segmentId: calSegmentId,
    mode: calMode,
    provenance: calProvenance,
    armed: calArmed,
    ptCount: calPtCount,
    rulerMeters: calRulerMeters,
    planeW: calPlaneW,
    planeH: calPlaneH,
    presetName: calPresetName,
    autoGLocked: calAutoGLocked,
    pxPerMeter: Number(getEffectivePxPerMeter().toFixed(4)),
    ptsNorm: calPtCount > 0
      ? Array.from({ length: calPtCount }, (_, i) => [
          Number(calPtsNormX[i].toFixed(5)),
          Number(calPtsNormY[i].toFixed(5)),
        ])
      : [],
    homographyValid: calHomographyValid,
    homography: calHomographyValid ? Array.from(calHomography) : null,
  };
}

function recordCalibrationSegmentTransition(reason) {
  calSegmentId++;
  if (recActive) {
    recCalSegments.push({
      segmentId: calSegmentId,
      absStartFrame: recTotalWritten,
      startRow: recCount,
      elapsedSec: Number(recElapsedSec.toFixed(4)),
      reason,
      sourceMode,
      calibration: getCalibrationSnapshot(),
    });
  }
}

function emitCalibrationState(overrideNotice) {
  self.postMessage({
    type: 'cal_state',
    calSegmentId,
    calMode,
    calProvenance,
    calArmed,
    calPtCount,
    calDraftCount,
    calSyncSeq: lastCalSyncSeq,
    rulerMeters: calArmed === 'ruler' ? calDraftRulerMeters : calRulerMeters,
    planeW: calArmed === 'homography' ? calDraftPlaneW : calPlaneW,
    planeH: calArmed === 'homography' ? calDraftPlaneH : calPlaneH,
    presetName: calPresetName,
    autoGLocked: calAutoGLocked,
    calibrated: isMetricCalibrated(),
    ptsNorm: calPtCount > 0
      ? Array.from({ length: calPtCount }, (_, i) => [calPtsNormX[i], calPtsNormY[i]])
      : [],
    summary: formatCalibrationSummary(overrideNotice),
  });
}

function clearCalibrationState() {
  calMode = 'none';
  calProvenance = 'none';
  calArmed = 'none';
  calPtCount = 0;
  calDraftCount = 0;
  calPresetName = '';
  calAutoGNormH = 0.0;
  calAutoGLocked = false;
  calHomographyValid = false;
  calHomography.fill(0.0);
  calHomographyInv.fill(0.0);
}

function applySceneCalibrationPreset(presetKey) {
  clearCalibrationState();
  calPresetName = presetKey || '';
  if (presetKey === 'carom_billiards_impulses') {
    // Assumed illustrative Carom table inner plane (2.84 m x 1.42 m)
    calPlaneW = 2.84;
    calPlaneH = 1.42;
    calPtsNormX[0] = 0.18; calPtsNormY[0] = 0.21;
    calPtsNormX[1] = 0.82; calPtsNormY[1] = 0.21;
    calPtsNormX[2] = 0.94; calPtsNormY[2] = 0.87;
    calPtsNormX[3] = 0.06; calPtsNormY[3] = 0.87;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
    calProvenance = calHomographyValid ? 'assumed' : 'none';
  } else if (presetKey === 'roundabout_occlusion') {
    // Assumed illustrative multi-lane highway deck (20.0 m x 32.0 m)
    calPlaneW = 20.00;
    calPlaneH = 32.00;
    calPtsNormX[0] = 0.20; calPtsNormY[0] = 0.17;
    calPtsNormX[1] = 0.78; calPtsNormY[1] = 0.17;
    calPtsNormX[2] = 0.93; calPtsNormY[2] = 0.90;
    calPtsNormX[3] = 0.07; calPtsNormY[3] = 0.90;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
    calProvenance = calHomographyValid ? 'assumed' : 'none';
  } else if (presetKey === 'juggling_ballistic') {
    // Newtonian Auto-g (9.80665 m/s^2) — requires >= 2 independent free-fall arcs before locking!
    calMode = 'auto_g';
    calProvenance = 'assumed';
    if (telemView && telemView[TELEM_GLOBAL + 1] >= AUTO_G_LOCK_CONF && telemView[TELEM_GLOBAL + 0] > 50.0) {
      calAutoGNormH = telemView[TELEM_GLOBAL + 0] / frameH;
      calAutoGLocked = true;
    }
  } else if (presetKey === 'galileo_pendulum_peg') {
    // Assumed Galileo pendulum stand chord (0.65 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.65;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.14;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.78;
    calPtCount = 2;
  } else if (presetKey === 'harmonic_metronome_phase') {
    // Assumed Nikko mechanical metronome tower body (0.20 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.20;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.18;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.84;
    calPtCount = 2;
  } else if (presetKey === 'cologne_motorway_occlusion') {
    // Assumed illustrative multi-lane A4 motorway deck (28.0 m x 48.0 m)
    calPlaneW = 28.00;
    calPlaneH = 48.00;
    calPtsNormX[0] = 0.22; calPtsNormY[0] = 0.16;
    calPtsNormX[1] = 0.78; calPtsNormY[1] = 0.16;
    calPtsNormX[2] = 0.94; calPtsNormY[2] = 0.90;
    calPtsNormX[3] = 0.06; calPtsNormY[3] = 0.90;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
    calProvenance = calHomographyValid ? 'assumed' : 'none';
  } else if (presetKey === 'chaotic_magnetic_pendulum') {
    // Assumed magnetic pendulum suspension chord (0.35 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.35;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.12;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.80;
    calPtCount = 2;
  } else if (presetKey === 'coupled_metronomes_sync') {
    // Assumed coupled metronome platform span (0.45 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.45;
    calPtsNormX[0] = 0.22; calPtsNormY[0] = 0.58;
    calPtsNormX[1] = 0.78; calPtsNormY[1] = 0.58;
    calPtCount = 2;
  } else if (presetKey === 'newtons_cradle_impulses') {
    // Assumed 5-sphere Newton's cradle span (0.18 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.18;
    calPtsNormX[0] = 0.28; calPtsNormY[0] = 0.62;
    calPtsNormX[1] = 0.72; calPtsNormY[1] = 0.62;
    calPtCount = 2;
  } else if (
    presetKey === 'model_rocket_full_flights' ||
    presetKey === 'rocket_vtvl_full_flight' ||
    presetKey === 'water_rocket_parachute_flights'
  ) {
    // Assumed full-flight vertical ascent/descent span for real-camera rocketry showcases
    const rocketRulerPresets = {
      model_rocket_full_flights: [15.00, 0.88, 0.12],
      rocket_vtvl_full_flight: [10.00, 0.92, 0.16],
      water_rocket_parachute_flights: [12.00, 0.90, 0.12],
    };
    const [meters, y0, y1] = rocketRulerPresets[presetKey];
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = meters;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = y0;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = y1;
    calPtCount = 2;
  } else if (presetKey === 'testbench') {
    // Synthetic testbench 1.00 m optical bench reference
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 1.00;
    calPtsNormX[0] = 0.20; calPtsNormY[0] = 0.50;
    calPtsNormX[1] = 0.80; calPtsNormY[1] = 0.50;
    calPtCount = 2;
  }
  // Custom uploaded videos, webcams, and live feeds remain UNCALIBRATED (pixel units) until calibrated!
  recordCalibrationSegmentTransition(`preset:${calPresetName || 'uncalibrated'}`);
  emitCalibrationState();
}

// ============================================================================
// Blind Pixel Pillar Testbench Scenarios & 1-to-1 Ground-Truth Swap Oracle
// ============================================================================

const PILLARS_WIDE = [
  { x0: 88, x1: 136, y0: 14, y1: 166 },
  { x0: 184, x1: 232, y0: 14, y1: 166 },
];
const PILLARS_SHORT_GAP = [
  { x0: 102, x1: 118, y0: 14, y1: 166 },
  { x0: 202, x1: 218, y0: 14, y1: 166 },
];
const PILLARS_SWARM = [
  { x0: 108, x1: 124, y0: 14, y1: 166 },
  { x0: 196, x1: 212, y0: 14, y1: 166 },
];
const PILLARS_NONE = [];

function getActivePillars() {
  if (benchScenario === 'pillar_crossing') return PILLARS_WIDE;
  if (benchScenario === 'short_gap_bridge') return PILLARS_SHORT_GAP;
  if (benchScenario === 'multi_target_swarm') return PILLARS_SWARM;
  return PILLARS_NONE;
}

function getActiveSimDiscCount() {
  return benchScenario === 'multi_target_swarm' ? 12 : 4;
}

// ============================================================================
// CHRONOS-1000 High-Density Blind Scene Generators (Matched to benchmark_1000.mjs Q1-Q3)
// ============================================================================

const C1K_REF_W = 640.0;
const C1K_REF_H = 360.0;
const C1K_BG_U32 = (new Uint8Array(new Uint32Array([0x11223344]).buffer)[0] === 0x44)
  ? 0xFF160E0A
  : 0x0A0E16FF;

let c1kPixelBuf = new Uint8ClampedArray(640 * 360 * 4);
let c1kPixelU32 = new Uint32Array(c1kPixelBuf.buffer);
let c1kPixelImgData = new ImageData(c1kPixelBuf, 640, 360);

const C1K_MAX_PTS = 1208;
const c1kX = new Float32Array(C1K_MAX_PTS);
const c1kY = new Float32Array(C1K_MAX_PTS);
const c1kVx = new Float32Array(C1K_MAX_PTS);
const c1kVy = new Float32Array(C1K_MAX_PTS);
const c1kBaseX = new Float32Array(C1K_MAX_PTS);
const c1kBaseY = new Float32Array(C1K_MAX_PTS);
const c1kModeShape = new Float32Array(C1K_MAX_PTS);
const c1kPeak = new Float32Array(C1K_MAX_PTS);
const c1kTintR = new Float32Array(C1K_MAX_PTS);
const c1kTintG = new Float32Array(C1K_MAX_PTS);
const c1kTintB = new Float32Array(C1K_MAX_PTS);
const c1kBoidCellHead = new Int16Array(60 * 35);
const c1kBoidCellNext = new Int16Array(1100);
const c1kVortexList = [];
const c1kRigidDefs = [
  { cx: 140.0, cy: 135.0, vx: 16.0, vy: 8.0, omega: 0.90, theta: 0.0 },
  { cx: 330.0, cy: 235.0, vx: 8.0, vy: -10.0, omega: -0.70, theta: 0.0 },
  { cx: 505.0, cy: 135.0, vx: -14.0, vy: 12.0, omega: 0.60, theta: 0.0 },
];
const c1kPolyOffsetsX = new Float32Array(36);
const c1kPolyOffsetsY = new Float32Array(36);
let c1kSimCount = 0;
let c1kTimeSec = 0.0;

(function initC1kPolyOffsets() {
  for (let k = 0; k < 24; k++) {
    const a = (2.0 * Math.PI * k) / 24.0;
    const r = 42.0 + ((k & 1) === 0 ? 2.0 : -2.0);
    c1kPolyOffsetsX[k] = r * Math.cos(a);
    c1kPolyOffsetsY[k] = r * Math.sin(a);
  }
  for (let k = 0; k < 12; k++) {
    const a = (2.0 * Math.PI * k) / 12.0 + Math.PI / 12.0;
    c1kPolyOffsetsX[24 + k] = 22.0 * Math.cos(a);
    c1kPolyOffsetsY[24 + k] = 22.0 * Math.sin(a);
  }
})();

function seedC1kParticleVisual(slot, seedId) {
  const h1 = Math.abs(Math.sin(seedId * 12.9898 + 0.5) * 43758.5453) % 1.0;
  const h2 = Math.abs(Math.sin(seedId * 78.233 + 1.9) * 43758.5453) % 1.0;
  c1kPeak[slot] = 212.0 + h1 * 38.0;
  c1kTintR[slot] = 0.88 + 0.22 * h2;
  c1kTintG[slot] = 0.94 + 0.12 * (1.0 - h1);
  c1kTintB[slot] = 1.05 - 0.18 * h2;
}

function ensureC1kPixelSurface(w, h) {
  if (c1kPixelImgData.width === w && c1kPixelImgData.height === h) return;
  c1kPixelBuf = new Uint8ClampedArray(w * h * 4);
  c1kPixelU32 = new Uint32Array(c1kPixelBuf.buffer);
  c1kPixelImgData = new ImageData(c1kPixelBuf, w, h);
}

function c1kVortexVelocity(x, y, out) {
  let u = 0.0;
  let v = 0.0;
  const invRc2 = 1.0 / (48.0 * 48.0);
  for (let i = 0; i < c1kVortexList.length; i++) {
    const vo = c1kVortexList[i];
    const dx = x - vo.cx;
    const dy = y - vo.cy;
    const r2 = dx * dx + dy * dy;
    if (r2 > 1e-6) {
      const f = (vo.k / r2) * (1.0 - Math.exp(-r2 * invRc2));
      u += -dy * f;
      v += dx * f;
    }
  }
  out[0] = u;
  out[1] = v;
}

const _c1kK1 = new Float32Array(2);
const _c1kK2 = new Float32Array(2);
const _c1kK3 = new Float32Array(2);
const _c1kK4 = new Float32Array(2);

function resetChronos1000Scenario() {
  c1kTimeSec = 0.0;
  c1kSimCount = 0;
  if (benchScenario === 'vortex_swarm_1200') {
    c1kVortexList.length = 0;
    const K = 4600.0;
    const bases = [
      [210.0, 180.0, K],
      [430.0, 180.0, -K],
    ];
    for (const mx of [-1, 0, 1]) {
      for (const my of [-1, 0, 1]) {
        const sign = (mx !== 0 ? -1 : 1) * (my !== 0 ? -1 : 1);
        for (const [bx, by, bk] of bases) {
          const vx = mx === 0 ? bx : mx === -1 ? 2.0 * 14.0 - bx : 2.0 * 626.0 - bx;
          const vy = my === 0 ? by : my === -1 ? 2.0 * 14.0 - by : 2.0 * 346.0 - by;
          c1kVortexList.push({ cx: vx, cy: vy, k: bk * sign });
        }
      }
    }
    const nx = 48;
    const ny = 25;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const h1 = Math.sin(idx * 12.9898 + 1.23) * 43758.5453;
        const h2 = Math.sin(idx * 78.233 + 4.56) * 43758.5453;
        c1kX[idx] = 22.0 + ((i + 0.5) * (618.0 - 22.0)) / nx + (h1 - Math.floor(h1) - 0.5) * 3.0;
        c1kY[idx] = 20.0 + ((j + 0.5) * (340.0 - 20.0)) / ny + (h2 - Math.floor(h2) - 0.5) * 3.0;
        seedC1kParticleVisual(idx, idx + 1);
      }
    }
    c1kSimCount = 1200;
  } else if (benchScenario === 'murmuration_1200') {
    c1kRigidDefs[0] = { cx: 140.0, cy: 135.0, vx: 16.0, vy: 8.0, omega: 0.90, theta: 0.0 };
    c1kRigidDefs[1] = { cx: 330.0, cy: 235.0, vx: 8.0, vy: -10.0, omega: -0.70, theta: 0.0 };
    c1kRigidDefs[2] = { cx: 505.0, cy: 135.0, vx: -14.0, vy: 12.0, omega: 0.60, theta: 0.0 };
    let bSeed = 0;
    for (let b = 0; b < 3; b++) {
      for (let k = 0; k < 36; k++) {
        seedC1kParticleVisual(b * 36 + k, 1 + b * 40 + k);
        bSeed++;
      }
    }
    const candX = [];
    const candY = [];
    const nx = 50;
    const ny = 28;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const h1 = Math.sin(idx * 17.13 + 0.7) * 43758.5453;
        const h2 = Math.sin(idx * 91.7 + 2.1) * 43758.5453;
        const px = 22.0 + ((i + 0.5) * 596.0) / nx + (h1 - Math.floor(h1) - 0.5) * 3.0;
        const py = 20.0 + ((j + 0.5) * 320.0) / ny + (h2 - Math.floor(h2) - 0.5) * 3.0;
        let ok = true;
        for (let b = 0; b < 3; b++) {
          if (Math.hypot(px - c1kRigidDefs[b].cx, py - c1kRigidDefs[b].cy) < 58.0) {
            ok = false;
            break;
          }
        }
        if (ok) {
          candX.push(px);
          candY.push(py);
        }
      }
    }
    const step = candX.length / 1100.0;
    for (let m = 0; m < 1100; m++) {
      const si = Math.floor(m * step);
      const px = candX[si];
      c1kX[108 + m] = px;
      c1kY[108 + m] = candY[si];
      c1kVx[108 + m] = 32.0;
      c1kVy[108 + m] = 6.0 * Math.sin(0.02 * px);
      seedC1kParticleVisual(108 + m, 200 + m);
    }
    c1kSimCount = 1208;
  } else if (benchScenario === 'modal_beam_1152') {
    const nx = 48;
    const ny = 24;
    for (let j = 0; j < ny; j++) {
      const y0 = 72.0 + (j * (288.0 - 72.0)) / (ny - 1);
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const x0 = 44.0 + (i * (596.0 - 44.0)) / (nx - 1);
        const u = (x0 - 320.0) / 276.0;
        c1kBaseX[idx] = x0;
        c1kBaseY[idx] = y0;
        c1kModeShape[idx] = Math.sin(0.5 * Math.PI * u);
        c1kX[idx] = x0;
        c1kY[idx] = y0;
        seedC1kParticleVisual(idx, idx + 1);
      }
    }
    c1kSimCount = 1152;
  }
}

function stepAndRenderChronos1000Scene(dtSec) {
  const dt = Math.min(0.05, Math.max(0.005, dtSec));
  const sx = frameW / C1K_REF_W;
  const sy = frameH / C1K_REF_H;
  ensureC1kPixelSurface(frameW, frameH);
  c1kPixelU32.fill(C1K_BG_U32);

  if (benchScenario === 'vortex_swarm_1200') {
    for (let i = 0; i < 1200; i++) {
      const x = c1kX[i];
      const y = c1kY[i];
      c1kVortexVelocity(x, y, _c1kK1);
      c1kVortexVelocity(x + 0.5 * dt * _c1kK1[0], y + 0.5 * dt * _c1kK1[1], _c1kK2);
      c1kVortexVelocity(x + 0.5 * dt * _c1kK2[0], y + 0.5 * dt * _c1kK2[1], _c1kK3);
      c1kVortexVelocity(x + dt * _c1kK3[0], y + dt * _c1kK3[1], _c1kK4);
      let nx = x + ((_c1kK1[0] + 2.0 * _c1kK2[0] + 2.0 * _c1kK3[0] + _c1kK4[0]) / 6.0) * dt;
      let ny = y + ((_c1kK1[1] + 2.0 * _c1kK2[1] + 2.0 * _c1kK3[1] + _c1kK4[1]) / 6.0) * dt;
      if (nx < 14.0) nx = 14.0;
      else if (nx > 626.0) nx = 626.0;
      if (ny < 14.0) ny = 14.0;
      else if (ny > 346.0) ny = 346.0;
      c1kX[i] = nx;
      c1kY[i] = ny;
    }
  } else if (benchScenario === 'murmuration_1200') {
    const t = c1kTimeSec;
    for (let b = 0; b < 3; b++) {
      const bd = c1kRigidDefs[b];
      bd.cx += bd.vx * dt;
      bd.cy += bd.vy * dt;
      bd.theta += bd.omega * dt;
      if (bd.cx < 62.0) { bd.cx = 62.0; bd.vx = Math.abs(bd.vx); }
      else if (bd.cx > 578.0) { bd.cx = 578.0; bd.vx = -Math.abs(bd.vx); }
      if (bd.cy < 62.0) { bd.cy = 62.0; bd.vy = Math.abs(bd.vy); }
      else if (bd.cy > 298.0) { bd.cy = 298.0; bd.vy = -Math.abs(bd.vy); }
      const ct = Math.cos(bd.theta);
      const st = Math.sin(bd.theta);
      for (let k = 0; k < 36; k++) {
        const lx = c1kPolyOffsetsX[k];
        const ly = c1kPolyOffsetsY[k];
        c1kX[b * 36 + k] = bd.cx + lx * ct - ly * st;
        c1kY[b * 36 + k] = bd.cy + lx * st + ly * ct;
      }
    }
    c1kBoidCellHead.fill(-1);
    for (let m = 0; m < 1100; m++) {
      const idx = 108 + m;
      const gx = Math.max(0, Math.min(59, Math.floor(c1kX[idx] / 12.0)));
      const gy = Math.max(0, Math.min(34, Math.floor(c1kY[idx] / 12.0)));
      const cIdx = gy * 60 + gx;
      c1kBoidCellNext[m] = c1kBoidCellHead[cIdx];
      c1kBoidCellHead[cIdx] = m;
    }
    for (let m = 0; m < 1100; m++) {
      const idx = 108 + m;
      const bx = c1kX[idx];
      const by = c1kY[idx];
      let ux = 36.0 * Math.cos(0.45 * t) + 18.0 * Math.sin(0.012 * by + 1.2 * t);
      let uy = 22.0 * Math.sin(0.65 * t) + 14.0 * Math.cos(0.009 * bx - 0.9 * t);
      if (bx < 40.0) ux += (40.0 - bx) * 2.5;
      if (bx > 600.0) ux -= (bx - 600.0) * 2.5;
      if (by < 36.0) uy += (36.0 - by) * 2.5;
      if (by > 324.0) uy -= (by - 324.0) * 2.5;
      for (let b = 0; b < 3; b++) {
        const dx = bx - c1kRigidDefs[b].cx;
        const dy = by - c1kRigidDefs[b].cy;
        const d = Math.hypot(dx, dy);
        if (d < 58.0 && d > 1e-3) {
          const push = (58.0 - d) * 4.5;
          ux += (dx / d) * push;
          uy += (dy / d) * push;
        }
      }
      const gx = Math.max(0, Math.min(59, Math.floor(bx / 12.0)));
      const gy = Math.max(0, Math.min(34, Math.floor(by / 12.0)));
      for (let oy = -1; oy <= 1; oy++) {
        const yy = gy + oy;
        if (yy < 0 || yy >= 35) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const xx = gx + ox;
          if (xx < 0 || xx >= 60) continue;
          let other = c1kBoidCellHead[yy * 60 + xx];
          while (other !== -1) {
            if (other !== m) {
              const jIdx = 108 + other;
              const dx = bx - c1kX[jIdx];
              const dy = by - c1kY[jIdx];
              const d2 = dx * dx + dy * dy;
              if (d2 < 90.25 && d2 > 1e-4) {
                const d = Math.sqrt(d2);
                const push = (9.5 - d) * 6.0;
                ux += (dx / d) * push;
                uy += (dy / d) * push;
              }
            }
            other = c1kBoidCellNext[other];
          }
        }
      }
      let nvx = c1kVx[idx] * 0.65 + ux * 0.35;
      let nvy = c1kVy[idx] * 0.65 + uy * 0.35;
      const spd = Math.hypot(nvx, nvy);
      if (spd > 75.0) {
        nvx *= 75.0 / spd;
        nvy *= 75.0 / spd;
      }
      c1kVx[idx] = nvx;
      c1kVy[idx] = nvy;
    }
    for (let m = 0; m < 1100; m++) {
      const idx = 108 + m;
      let nbx = Math.min(624.0, Math.max(16.0, c1kX[idx] + c1kVx[idx] * dt));
      let nby = Math.min(344.0, Math.max(16.0, c1kY[idx] + c1kVy[idx] * dt));
      for (let b = 0; b < 3; b++) {
        const dx = nbx - c1kRigidDefs[b].cx;
        const dy = nby - c1kRigidDefs[b].cy;
        const d = Math.hypot(dx, dy);
        if (d < 50.0 && d > 1e-3) {
          nbx = c1kRigidDefs[b].cx + (dx / d) * 50.0;
          nby = c1kRigidDefs[b].cy + (dy / d) * 50.0;
        }
      }
      c1kX[idx] = nbx;
      c1kY[idx] = nby;
    }
  } else if (benchScenario === 'modal_beam_1152') {
    const phase = Math.sin(2.0 * Math.PI * 3.00 * c1kTimeSec);
    for (let i = 0; i < 1152; i++) {
      c1kX[i] = c1kBaseX[i];
      c1kY[i] = c1kBaseY[i] + 1.25 * c1kModeShape[i] * phase;
    }
  }

  c1kTimeSec += dt;

  // Stamp textured Gaussian sub-pixel blobs matching benchmark_1000.mjs appearance
  const invTwoSig2 = 1.0 / (2.0 * (0.85 * sx) * (0.85 * sx));
  const r2Max = 4.84 * sx * sx;
  const radPx = Math.max(1, Math.ceil(2.2 * sx));
  for (let i = 0; i < c1kSimCount; i++) {
    const cx = c1kX[i] * sx;
    const cy = c1kY[i] * sy;
    const x0 = Math.max(0, Math.floor(cx) - radPx);
    const x1 = Math.min(frameW - 1, Math.floor(cx) + radPx);
    const y0 = Math.max(0, Math.floor(cy) - radPx);
    const y1 = Math.min(frameH - 1, Math.floor(cy) + radPx);
    const pk = c1kPeak[i];
    const tR = c1kTintR[i];
    const tG = c1kTintG[i];
    const tB = c1kTintB[i];
    for (let py = y0; py <= y1; py++) {
      const dy = py - cy;
      for (let px = x0; px <= x1; px++) {
        const dx = px - cx;
        const r2 = dx * dx + dy * dy;
        if (r2 > r2Max) continue;
        const g = Math.exp(-r2 * invTwoSig2);
        const off = (py * frameW + px) * 4;
        const r = Math.min(255, Math.round(10 + pk * tR * g));
        const gr = Math.min(255, Math.round(14 + pk * tG * g));
        const b = Math.min(255, Math.round(22 + pk * tB * g));
        if (r > c1kPixelBuf[off]) c1kPixelBuf[off] = r;
        if (gr > c1kPixelBuf[off + 1]) c1kPixelBuf[off + 1] = gr;
        if (b > c1kPixelBuf[off + 2]) c1kPixelBuf[off + 2] = b;
      }
    }
  }

  simCtx.putImageData(c1kPixelImgData, 0, 0);
  return c1kPixelBuf;
}

const simDiscs = [
  { id: 0,  x: 52,  y: 56,  baseX: 66,  baseY: 56,  vx: 3.0,  vy: 0.0, ballistic: false, r: '#f83a44', core: '#ffe66e' },
  { id: 1,  x: 172, y: 56,  baseX: 254, baseY: 56,  vx: -3.0, vy: 0.0, ballistic: false, r: '#1ee1fa', core: '#0f2d55' },
  { id: 2,  x: 148, y: 124, baseX: 66,  baseY: 124, vx: 3.0,  vy: 0.0, ballistic: false, r: '#32f270', core: '#ebffeb' },
  { id: 3,  x: 268, y: 124, baseX: 254, baseY: 124, vx: -3.0, vy: 0.0, ballistic: false, r: '#e840f8', core: '#5af5ff' },
  { id: 4,  x: 54,  y: 78,  baseX: 54,  baseY: 78,  vx: 2.9,  vy: 0.0, ballistic: false, r: '#ff922b', core: '#fff3bf' },
  { id: 5,  x: 266, y: 78,  baseX: 266, baseY: 78,  vx: -2.9, vy: 0.0, ballistic: false, r: '#38d9a9', core: '#0b4f4a' },
  { id: 6,  x: 88,  y: 104, baseX: 88,  baseY: 104, vx: 3.0,  vy: 0.0, ballistic: false, r: '#a9e34b', core: '#1e3a10' },
  { id: 7,  x: 232, y: 104, baseX: 232, baseY: 104, vx: -3.0, vy: 0.0, ballistic: false, r: '#748ffc', core: '#eef2ff' },
  { id: 8,  x: 44,  y: 130, baseX: 44,  baseY: 130, vx: 3.1,  vy: 0.0, ballistic: false, r: '#f06595', core: '#fff0f6' },
  { id: 9,  x: 276, y: 130, baseX: 276, baseY: 130, vx: -3.1, vy: 0.0, ballistic: false, r: '#ffd43b', core: '#3b2400' },
  { id: 10, x: 68,  y: 154, baseX: 68,  baseY: 154, vx: 3.2,  vy: 0.0, ballistic: false, r: '#66d9e8', core: '#0c3547' },
  { id: 11, x: 252, y: 154, baseX: 252, baseY: 154, vx: -3.2, vy: 0.0, ballistic: false, r: '#c0eb75', core: '#26380a' },
];

// Pre-allocated model rocket multi-stage buffers (0: boost, 1: coast/apogee, 2: drogue recovery)
const rocketPhase = new Uint8Array(4);
const rocketBoostTicks = new Uint16Array(4);
const SMOKE_PUFFS = 40;
const smokeX = new Float32Array(SMOKE_PUFFS);
const smokeY = new Float32Array(SMOKE_PUFFS);
const smokeR = new Float32Array(SMOKE_PUFFS);
const smokeAlpha = new Float32Array(SMOKE_PUFFS);
let smokeWriteIdx = 0;

// Pre-allocated 65,536-ID ownership table (eliminates & 255 ID wrap aliasing)
const MAX_ORACLE_IDS = 65536;
const oracleTrackOwner = new Int16Array(MAX_ORACLE_IDS).fill(-1);
const oracleCandTrackSlot = new Int32Array(MAX_TARGETS * MAX_TARGETS);
const oracleCandDiscIdx = new Int32Array(MAX_TARGETS * MAX_TARGETS);
const oracleCandDist = new Float32Array(MAX_TARGETS * MAX_TARGETS);
let oraclePadicSwaps = 0;
let inPillarCollisions = 0;
let simTimeSec = 0.0;
let pendingSaccadeFrames = 0;
let pendingAirSnapFrames = 0;

const TARGET_COLORS = [
  '#00f0ff', '#ff4d6d', '#38f27a', '#ffb703',
  '#c77dff', '#4cc9f0', '#f72585', '#aacc00',
  '#ff7b00', '#06d6a0', '#e0aaff', '#ffd166',
  '#72efdd', '#ff8fab', '#9ef01a', '#80ffdb',
];

const ZONE_COLORS = ['#00f0ff', '#ffb703', '#38f27a', '#ff4d6d'];

function clearVisualQueues() {
  swarmRingValid.fill(0);
  ringIngestFrameId.fill(-1);
  ringIngestTimeMs.fill(-1.0);
  ringIngestPtsMs.fill(-1.0);
  clearActiveInspectionSelection();
  lastSwarmCopyMs = 0.0;
  lastFieldRenderMs = 0.0;
  ingestElapsedMs = 0.0;
  latestCausalTimeMs = 0.0;
  latestCausalPtsMs = -1.0;
  shockTriggerFrame.fill(-1);
  shockAge.fill(-1);
  shockInDeg.fill(0.0);
  shockDeflectDeg.fill(0.0);
  shockRestitution.fill(0.0);
  lastStrikeDeflectDeg = 0.0;
  lastStrikeRestitution = 0.0;
  lastStrikeTid = 0;
  lastStrikeFrame = -1;
  lastStrikeValid = false;
  lastScatterDeg = 0.0;
  lastScatterFrame = -1;
  slotLastStrikeX.fill(0.0);
  slotLastStrikeY.fill(0.0);
  prevRectHeading.fill(0.0);
  prevRectTid.fill(0);
  prevRectFrame.fill(-1);
  slotRectAngVel.fill(0.0);
  padFlashAlpha.fill(0.0);
  apexLockTid.fill(0);
  apexLockErrPx.fill(-1.0);
  apexLockHold.fill(0);
  lastScoredApexErrPx = -1.0;
  histTremorR.fill(0.0);
  histTremorH.fill(0.0);
  histPhaseValid.fill(0);
  histStrobe.fill(0);
  histPipTid.fill(-1);
  histLambda2.fill(0.0);
  scopeWriteIdx = 0;
  strobeFlashAlpha = 0.0;
  totalBridgedEvents = 0;
  totalStrikeEvents = 0;
  trailOwnerTid.fill(-1);
  trailHead.fill(0);
  trailCount.fill(0);
  trailLastFrame.fill(-1);
  slotHoldLeft.fill(0);
  slotHoldTid.fill(0);
  slotIdleDrawn.fill(0);
  slotHoldVeilDrawn.fill(0);
  slotSpeedPillOn.fill(0);
  slotSmoothedSpeed.fill(0.0);
  slotSmoothedHeading.fill(0.0);
  slotSmoothedOmega.fill(0.0);
  slotSmoothedEca.fill(0.0);
  slotLastAgeFrames.fill(0);
  slotLastCoast.fill(0);
  slotLastReacq.fill(0);
  slotLastVal.fill(0);
  slotLastSigHi.fill(0);
  slotLastSigLo.fill(0);
  slotLastFreqHz.fill(0.0);
  slotLastMetricValid.fill(0.0);
  slotLastPivotValid.fill(0);
  slotLastPendSwing.fill(0.0);
  slotLastPendRad.fill(0.0);
  phaseHistWrite = 0;
  phaseHistCount = 0;
  phaseHistTid.fill(-1);
  phaseHistX.fill(0.0);
  phaseHistY.fill(0.0);
  phaseHistVx.fill(0.0);
  phaseHistVy.fill(0.0);
  phaseHistE.fill(0.0);
  phaseHistEnsK.fill(0.0);
  phaseHistKurR.fill(0.0);
  phaseHistKurN.fill(0);
  phaseHistRegime.fill(0);
  phaseHistJump.fill(0);
  lastPhysicsPhaseRenderMs = 0.0;
}

function resetPersistenceState() {
  contTid.fill(0);
  contSlot.fill(-1);
  contFirstSeq.fill(0);
  contLastSeq.fill(0);
  contTotalFrames.fill(0);
  contTotalSec.fill(0.0);
  contRelocks.fill(0);
  contSegCount.fill(0);
  contSegStart.fill(0);
  contSegEnd.fill(0);
  contWasActive.fill(0);
  contBinMaxActive.fill(0);
  contBinFrames.fill(0);
  contCount = 0;
  persistSampleSeq = 0;
  sessionElapsedSec = 0.0;
  lastPersistRenderMs = 0.0;
  contBinStrideFrames = 15;
  contBinHead = 0;
}

function updatePersistenceState(fpsDt) {
  if (!telemView) return;
  persistSampleSeq++;
  sessionElapsedSec += fpsDt;
  const seq = persistSampleSeq;
  const activeNow = Math.max(0, Math.min(MAX_TARGETS, Math.round(telemView[3])));

  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] <= 0.5) continue;
    const tid = Math.round(telemView[b + 1]);
    if (tid <= 0) continue;

    let idx = -1;
    for (let t = 0; t < contCount; t++) {
      if (contTid[t] === tid) {
        idx = t;
        break;
      }
    }
    if (idx < 0) {
      if (contCount < CONT_MAX_TRACKS) {
        idx = contCount++;
      } else {
        let evictIdx = -1;
        let evictFrames = 1e9;
        for (let t = 0; t < CONT_MAX_TRACKS; t++) {
          if (contLastSeq[t] < seq - 2 && contTotalFrames[t] < evictFrames) {
            evictFrames = contTotalFrames[t];
            evictIdx = t;
          }
        }
        if (evictIdx < 0) {
          for (let t = 0; t < CONT_MAX_TRACKS; t++) {
            if (contTotalFrames[t] < evictFrames) {
              evictFrames = contTotalFrames[t];
              evictIdx = t;
            }
          }
        }
        idx = evictIdx >= 0 ? evictIdx : 0;
      }
      contTid[idx] = tid;
      contSlot[idx] = m;
      contFirstSeq[idx] = seq;
      contLastSeq[idx] = seq;
      contTotalFrames[idx] = 1;
      contTotalSec[idx] = fpsDt;
      contRelocks[idx] = 0;
      contSegCount[idx] = 1;
      contSegStart[idx * CONT_MAX_SEGS] = seq;
      contSegEnd[idx * CONT_MAX_SEGS] = seq;
      contWasActive[idx] = 1;
    } else {
      const gap = seq - contLastSeq[idx];
      contSlot[idx] = m;
      contLastSeq[idx] = seq;
      contTotalFrames[idx]++;
      contTotalSec[idx] += fpsDt;
      if (!contWasActive[idx] && gap > 6) {
        contRelocks[idx]++;
        const sc = contSegCount[idx];
        if (sc < CONT_MAX_SEGS) {
          contSegStart[idx * CONT_MAX_SEGS + sc] = seq;
          contSegEnd[idx * CONT_MAX_SEGS + sc] = seq;
          contSegCount[idx] = sc + 1;
        } else {
          contSegEnd[idx * CONT_MAX_SEGS + CONT_MAX_SEGS - 1] = seq;
        }
      } else {
        const sc = contSegCount[idx];
        if (sc > 0) {
          contSegEnd[idx * CONT_MAX_SEGS + (sc - 1)] = seq;
        }
      }
      contWasActive[idx] = 1;
    }
  }

  for (let t = 0; t < contCount; t++) {
    if (contLastSeq[t] !== seq) contWasActive[t] = 0;
  }

  let binIdx = Math.floor(seq / contBinStrideFrames);
  if (binIdx >= CONT_HIST_BINS) {
    const half = CONT_HIST_BINS >> 1;
    for (let k = 0; k < half; k++) {
      contBinMaxActive[k] = Math.max(contBinMaxActive[k * 2], contBinMaxActive[k * 2 + 1]);
      contBinFrames[k] = contBinFrames[k * 2];
    }
    contBinMaxActive.fill(0, half);
    contBinFrames.fill(0, half);
    contBinStrideFrames *= 2;
    contBinHead = half;
    binIdx = Math.min(CONT_HIST_BINS - 1, Math.floor(seq / contBinStrideFrames));
  }
  if (binIdx >= contBinHead) contBinHead = binIdx + 1;
  contBinFrames[binIdx]++;
  if (activeNow > contBinMaxActive[binIdx]) contBinMaxActive[binIdx] = activeNow;
}

function formatPersistDurationSec(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return m > 0 ? `${m}m ${String(rem).padStart(2, '0')}s` : `${sec.toFixed(sec < 10 ? 1 : 0)}s`;
}

function renderPersistenceTimeline(force = false) {
  if (!persistenceCtx || !persistenceCanvas) return;
  const nowMs = performance.now();
  if (!force && nowMs - lastPersistRenderMs < 500.0) return;
  lastPersistRenderMs = nowMs;

  const W = persistenceCanvas.width;
  const H = persistenceCanvas.height;
  const ctx = persistenceCtx;

  ctx.fillStyle = '#070b13';
  ctx.fillRect(0, 0, W, H);

  const sparkX0 = 10;
  const sparkY0 = 20;
  const sparkW = W - 20;
  const sparkH = 44;

  ctx.fillStyle = '#9db4d4';
  ctx.font = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  const binsUsed = Math.max(1, contBinHead);
  const spanTxt = formatPersistDurationSec(sessionElapsedSec);
  ctx.fillText(`CONCURRENCY TIMELINE (0..${MAX_TARGETS} LOCKS | SPAN ${spanTxt})`, sparkX0, 13);

  ctx.fillStyle = '#0b1220';
  ctx.fillRect(sparkX0, sparkY0, sparkW, sparkH);
  ctx.strokeStyle = '#192438';
  ctx.lineWidth = 1;
  ctx.strokeRect(sparkX0 + 0.5, sparkY0 + 0.5, sparkW - 1, sparkH - 1);

  const midY = sparkY0 + sparkH * 0.5;
  ctx.strokeStyle = 'rgba(100, 128, 168, 0.25)';
  ctx.beginPath();
  ctx.moveTo(sparkX0, midY);
  ctx.lineTo(sparkX0 + sparkW, midY);
  ctx.stroke();

  if (persistSampleSeq > 0) {
    const barW = Math.max(1.5, sparkW / Math.max(16, binsUsed));
    for (let b = 0; b < binsUsed; b++) {
      const val = contBinMaxActive[b];
      if (val <= 0) continue;
      const frac = Math.min(1.0, val / MAX_TARGETS);
      const bh = Math.max(2, Math.round(frac * (sparkH - 4)));
      const bx = sparkX0 + (b / Math.max(1, binsUsed)) * sparkW;
      ctx.fillStyle = val >= 9 ? '#38f27a' : '#00f0ff';
      ctx.fillRect(bx, sparkY0 + sparkH - 2 - bh, Math.max(1.5, barW - 0.5), bh);
    }
  }

  const laneHeaderY = sparkY0 + sparkH + 15;
  ctx.fillStyle = '#9db4d4';
  ctx.fillText('TOP-10 TARGET CUSTODY SWIMLANES (LOCKED vs RE-ENGAGED)', sparkX0, laneHeaderY);

  const swimX0 = 56;
  const swimW = W - swimX0 - 48;
  const swimTop = laneHeaderY + 6;
  const rowH = 11.5;

  if (contCount === 0 || persistSampleSeq <= 0) {
    ctx.fillStyle = '#5c7394';
    ctx.fillText('AWAITING MULTI-SECOND TARGET CUSTODY...', swimX0, swimTop + 42);
    return;
  }

  for (let i = 0; i < contCount; i++) contRankIdx[i] = i;
  const topN = Math.min(10, contCount);
  for (let i = 0; i < topN; i++) {
    let best = i;
    for (let j = i + 1; j < contCount; j++) {
      const a = contRankIdx[j];
      const b = contRankIdx[best];
      if (
        contTotalFrames[a] > contTotalFrames[b] ||
        (contTotalFrames[a] === contTotalFrames[b] && contTid[a] < contTid[b])
      ) {
        best = j;
      }
    }
    if (best !== i) {
      const tmp = contRankIdx[i];
      contRankIdx[i] = contRankIdx[best];
      contRankIdx[best] = tmp;
    }
  }

  const maxSeq = Math.max(1, persistSampleSeq);
  for (let r = 0; r < topN; r++) {
    const ti = contRankIdx[r];
    const tid = contTid[ti];
    const col = TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length];
    const ry = swimTop + r * rowH;

    ctx.fillStyle = contWasActive[ti] ? col : '#7f95b5';
    ctx.font = '600 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(`T${tid}`, sparkX0, ry + 8);

    ctx.fillStyle = '#0d1524';
    ctx.fillRect(swimX0, ry + 2, swimW, 6);

    const sc = Math.min(CONT_MAX_SEGS, contSegCount[ti]);
    if (sc > 1) {
      ctx.strokeStyle = '#ffd166';
      ctx.lineWidth = 1;
      for (let sIdx = 1; sIdx < sc; sIdx++) {
        const gStart = swimX0 + (contSegEnd[ti * CONT_MAX_SEGS + sIdx - 1] / maxSeq) * swimW;
        const gEnd = swimX0 + (contSegStart[ti * CONT_MAX_SEGS + sIdx] / maxSeq) * swimW;
        ctx.beginPath();
        ctx.moveTo(gStart, ry + 5);
        ctx.lineTo(Math.max(gStart + 1, gEnd), ry + 5);
        ctx.stroke();
      }
    }
    ctx.fillStyle = col;
    for (let sIdx = 0; sIdx < sc; sIdx++) {
      const sx0 = swimX0 + (contSegStart[ti * CONT_MAX_SEGS + sIdx] / maxSeq) * swimW;
      const sx1 = swimX0 + (contSegEnd[ti * CONT_MAX_SEGS + sIdx] / maxSeq) * swimW;
      ctx.fillRect(sx0, ry + 2, Math.max(2.0, sx1 - sx0), 6);
    }

    ctx.fillStyle = contWasActive[ti] ? '#d9e6f7' : '#7f95b5';
    ctx.fillText(formatPersistDurationSec(contTotalSec[ti]), swimX0 + swimW + 5, ry + 8);
  }
}

function resetTestbenchState() {
  const s = resScale;
  if (isChronos1000Scenario(benchScenario)) {
    resetChronos1000Scenario();
  } else if (benchScenario === 'ballistic_cascade') {
    simDiscs[0].x = 42 * s;  simDiscs[0].y = 146 * s; simDiscs[0].baseY = 146 * s; simDiscs[0].vx = 2.6 * s;  simDiscs[0].vy = -6.8 * s; simDiscs[0].ballistic = true;
    simDiscs[1].x = 278 * s; simDiscs[1].y = 138 * s; simDiscs[1].baseY = 146 * s; simDiscs[1].vx = -2.8 * s; simDiscs[1].vy = -5.6 * s; simDiscs[1].ballistic = true;
    simDiscs[2].x = 112 * s; simDiscs[2].y = 148 * s; simDiscs[2].baseY = 148 * s; simDiscs[2].vx = 2.2 * s;  simDiscs[2].vy = -6.2 * s; simDiscs[2].ballistic = true;
    simDiscs[3].x = 210 * s; simDiscs[3].y = 144 * s; simDiscs[3].baseY = 148 * s; simDiscs[3].vx = -2.4 * s; simDiscs[3].vy = -7.0 * s; simDiscs[3].ballistic = true;
  } else if (benchScenario === 'model_rocket_flight') {
    const laneX = [56, 124, 196, 264];
    const initY = [140, 74, 42, 92];
    const initVy = [-2.4, -3.8, -0.8, 1.9];
    const initPhase = [0, 1, 1, 2];
    const initTicks = [4, 16, 28, 40];
    for (let i = 0; i < 4; i++) {
      simDiscs[i].baseX = laneX[i] * s;
      simDiscs[i].x = laneX[i] * s;
      simDiscs[i].y = initY[i] * s;
      simDiscs[i].baseY = 152 * s;
      simDiscs[i].vx = (i % 2 === 0 ? 0.35 : -0.35) * s;
      simDiscs[i].vy = initVy[i] * s;
      simDiscs[i].ballistic = true;
      rocketPhase[i] = initPhase[i];
      rocketBoostTicks[i] = initTicks[i];
    }
    smokeAlpha.fill(0.0);
    smokeWriteIdx = 0;
  } else if (benchScenario === 'harmonic_tremor') {
    simDiscs[0].x = 76 * s;  simDiscs[0].y = 56 * s;  simDiscs[0].baseX = 76 * s;  simDiscs[0].baseY = 56 * s;  simDiscs[0].vx = 0; simDiscs[0].vy = 0; simDiscs[0].ballistic = false;
    simDiscs[1].x = 244 * s; simDiscs[1].y = 56 * s;  simDiscs[1].baseX = 244 * s; simDiscs[1].baseY = 56 * s;  simDiscs[1].vx = 0; simDiscs[1].vy = 0; simDiscs[1].ballistic = false;
    simDiscs[2].x = 76 * s;  simDiscs[2].y = 124 * s; simDiscs[2].baseX = 76 * s;  simDiscs[2].baseY = 124 * s; simDiscs[2].vx = 0; simDiscs[2].vy = 0; simDiscs[2].ballistic = false;
    simDiscs[3].x = 244 * s; simDiscs[3].y = 124 * s; simDiscs[3].baseX = 244 * s; simDiscs[3].baseY = 124 * s; simDiscs[3].vx = 0; simDiscs[3].vy = 0; simDiscs[3].ballistic = false;
  } else if (benchScenario === 'percussion_ricochet') {
    simDiscs[0].x = 58 * s;  simDiscs[0].y = 44 * s;  simDiscs[0].baseY = 44 * s;  simDiscs[0].vx = 4.6 * s;  simDiscs[0].vy = 3.4 * s;  simDiscs[0].ballistic = false;
    simDiscs[1].x = 262 * s; simDiscs[1].y = 52 * s;  simDiscs[1].baseY = 52 * s;  simDiscs[1].vx = -4.4 * s; simDiscs[1].vy = 3.7 * s;  simDiscs[1].ballistic = false;
    simDiscs[2].x = 74 * s;  simDiscs[2].y = 136 * s; simDiscs[2].baseY = 136 * s; simDiscs[2].vx = 4.2 * s;  simDiscs[2].vy = -3.8 * s; simDiscs[2].ballistic = false;
    simDiscs[3].x = 246 * s; simDiscs[3].y = 128 * s; simDiscs[3].baseY = 128 * s; simDiscs[3].vx = -4.5 * s; simDiscs[3].vy = -3.3 * s; simDiscs[3].ballistic = false;
  } else if (benchScenario === 'multi_target_swarm') {
    const laneY = [26, 52, 78, 104, 130, 154];
    const xPairs = [
      [38, 282],
      [76, 244],
      [54, 266],
      [88, 232],
      [44, 276],
      [68, 252],
    ];
    for (let lane = 0; lane < 6; lane++) {
      const spd = (2.7 + 0.1 * lane) * s;
      const ly = laneY[lane] * s;
      const a = simDiscs[lane * 2];
      const b = simDiscs[lane * 2 + 1];
      a.x = xPairs[lane][0] * s; a.y = ly; a.baseY = ly; a.vx = spd;  a.vy = 0; a.ballistic = false;
      b.x = xPairs[lane][1] * s; b.y = ly; b.baseY = ly; b.vx = -spd; b.vy = 0; b.ballistic = false;
    }
  } else {
    const spd = benchScenario === 'short_gap_bridge' ? 3.2 * s : 3.0 * s;
    simDiscs[0].x = 52 * s;  simDiscs[0].y = 56 * s;  simDiscs[0].baseY = 56 * s;  simDiscs[0].vx = spd;  simDiscs[0].vy = 0; simDiscs[0].ballistic = false;
    simDiscs[1].x = 172 * s; simDiscs[1].y = 56 * s;  simDiscs[1].baseY = 56 * s;  simDiscs[1].vx = -spd; simDiscs[1].vy = 0; simDiscs[1].ballistic = false;
    simDiscs[2].x = 148 * s; simDiscs[2].y = 124 * s; simDiscs[2].baseY = 124 * s; simDiscs[2].vx = spd;  simDiscs[2].vy = 0; simDiscs[2].ballistic = false;
    simDiscs[3].x = 268 * s; simDiscs[3].y = 124 * s; simDiscs[3].baseY = 124 * s; simDiscs[3].vx = -spd; simDiscs[3].vy = 0; simDiscs[3].ballistic = false;
  }

  oracleTrackOwner.fill(-1);
  prevFiedlerTrack.fill(0.0);
  clearVisualQueues();
  resetPersistenceState();
  oraclePadicSwaps = 0;
  inPillarCollisions = 0;
  simTimeSec = 0.0;
  pendingSaccadeFrames = 0;
  pendingAirSnapFrames = 0;
  renderAllSlotPips(false);
  renderPersistenceTimeline(true);
  renderPhysicsPhaseCanvas(true);
}

function stepAndRenderBlindPillarScene(dtSec) {
  if (isChronos1000Scenario(benchScenario)) {
    return stepAndRenderChronos1000Scene(dtSec);
  }
  const ctx = simCtx;
  const s = resScale;
  const stepScale = Math.min(2.0, Math.max(0.25, dtSec * 60.0));
  simTimeSec += dtSec;

  // Deterministic Mode B / Mode C testbench stimuli (zero setTimeout jitter)
  if (pendingSaccadeFrames > 0) {
    pendingSaccadeFrames--;
    if (pendingSaccadeFrames === 0) {
      const d0 = simDiscs[0];
      if (benchScenario === 'harmonic_tremor') {
        d0.baseX = d0.baseX < 160 * s ? d0.baseX + 32 * s : d0.baseX - 32 * s;
      } else {
        d0.x = d0.x < 160 * s ? Math.min(84 * s, d0.x + 30 * s) : Math.max(145 * s, d0.x - 30 * s);
      }
    }
  }
  if (pendingAirSnapFrames > 0) {
    pendingAirSnapFrames--;
    if (pendingAirSnapFrames === 0 && benchScenario !== 'percussion_ricochet' && benchScenario !== 'harmonic_tremor' && benchScenario !== 'model_rocket_flight') {
      const nDiscsSnap = getActiveSimDiscCount();
      for (let i = 0; i < nDiscsSnap; i++) {
        simDiscs[i].vx = Math.sign(simDiscs[i].vx || 1) * 3.0 * s;
      }
    }
  }

  // 1. Static structured aerospace grid clutter (tests uncensored >60 dB ECA cancellation)
  ctx.fillStyle = '#121722';
  ctx.fillRect(0, 0, frameW, frameH);

  const gridStep = 24 * s;
  ctx.strokeStyle = '#1d2536';
  ctx.lineWidth = s;
  ctx.beginPath();
  for (let x = gridStep; x < frameW; x += gridStep) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, frameH);
  }
  for (let y = gridStep; y < frameH; y += gridStep) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(frameW, y + 0.5);
  }
  ctx.stroke();

  if (benchScenario === 'model_rocket_flight') {
    // Launch pads, guide rails, and dissipating exhaust smoke puffs behind active boosters
    ctx.fillStyle = '#1b2434';
    ctx.fillRect(0, 156 * s, frameW, frameH - 156 * s);
    ctx.strokeStyle = 'rgba(130, 165, 210, 0.35)';
    ctx.lineWidth = Math.max(1, s);
    const laneX = [56, 124, 196, 264];
    for (let i = 0; i < laneX.length; i++) {
      ctx.beginPath();
      ctx.moveTo(laneX[i] * s, 156 * s);
      ctx.lineTo(laneX[i] * s, 126 * s);
      ctx.stroke();
    }
    for (let p = 0; p < SMOKE_PUFFS; p++) {
      const a = smokeAlpha[p];
      if (a <= 0.015) continue;
      smokeY[p] += 0.14 * s * stepScale;
      smokeR[p] += 0.05 * s * stepScale;
      smokeAlpha[p] = Math.max(0.0, a - 0.022 * stepScale);
      ctx.fillStyle = `rgba(112, 130, 154, ${smokeAlpha[p].toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(smokeX[p], smokeY[p], smokeR[p], 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 2. Advance disc kinematics & collisions
  const tSec = simTimeSec;
  const wallMargin = 18 * s;
  const pillars = getActivePillars();
  const nActiveDiscs = getActiveSimDiscCount();

  for (let i = 0; i < nActiveDiscs; i++) {
    const d = simDiscs[i];
    if (benchScenario === 'model_rocket_flight') {
      if (rocketPhase[i] === 0) {
        // Powered boost phase (ay < 0 upward thrust)
        const thrust = (0.34 + 0.03 * (i % 3)) * s * stepScale;
        d.vy -= thrust;
        d.y += d.vy * stepScale;
        rocketBoostTicks[i]++;
        if ((rocketBoostTicks[i] & 1) === 0) {
          smokeX[smokeWriteIdx] = d.x + (Math.sin(tSec * 9.0 + i) * 1.1 * s);
          smokeY[smokeWriteIdx] = d.y + 7.5 * s;
          smokeR[smokeWriteIdx] = 1.8 * s;
          smokeAlpha[smokeWriteIdx] = 0.08;
          smokeWriteIdx = (smokeWriteIdx + 1) % SMOKE_PUFFS;
        }
        if (rocketBoostTicks[i] >= 12 || d.vy <= -5.2 * s) {
          rocketPhase[i] = 1;
        }
      } else if (rocketPhase[i] === 1) {
        // Newtonian free-fall coast + apogee rollover (ay = +g > 0)
        const vyPrev = d.vy;
        d.vy += 0.20 * s * stepScale;
        d.y += 0.5 * (vyPrev + d.vy) * stepScale;
        if (d.vy >= 2.2 * s) {
          rocketPhase[i] = 2;
        }
      } else {
        // Drogue parachute recovery descent toward pad
        d.vy = Math.min(d.vy + 0.04 * s * stepScale, (2.15 + 0.15 * (i % 2)) * s);
        d.y += d.vy * stepScale;
      }

      // Gentle in-lane windward arc (preserves non-degenerate 2D motion without crossing lanes)
      d.x += d.vx * stepScale;
      const laneOffset = d.x - d.baseX;
      if (laneOffset < -16 * s) {
        d.x = d.baseX - 16 * s;
        d.vx = Math.abs(d.vx);
      } else if (laneOffset > 16 * s) {
        d.x = d.baseX + 16 * s;
        d.vx = -Math.abs(d.vx);
      }

      if (d.y < 16 * s) {
        d.y = 16 * s;
        d.vy = Math.abs(d.vy) * 0.35;
        rocketPhase[i] = 1;
      }
      if (d.y >= 152 * s) {
        d.y = 152 * s;
        d.vy = -(1.25 + 0.15 * (i % 3)) * s;
        rocketPhase[i] = 0;
        rocketBoostTicks[i] = 0;
      }
      continue;
    }

    if (benchScenario === 'harmonic_tremor' && !d.ballistic) {
      const phase = 2.0 * Math.PI * 8.5 * tSec + i * 0.9;
      const amp = 4.2 * s;
      d.x = d.baseX + amp * Math.cos(phase);
      d.y = d.baseY - amp * Math.sin(phase);
      continue;
    }

    d.x += d.vx * stepScale;

    if (d.ballistic) {
      const vyPrev = d.vy;
      d.vy += 0.22 * s * stepScale;
      d.y += 0.5 * (vyPrev + d.vy) * stepScale;
      if (d.y >= 152 * s) {
        d.y = 152 * s;
        if (benchScenario === 'ballistic_cascade') {
          d.vy = -(5.8 + (i % 3) * 0.55) * s;
        } else {
          d.vy = -Math.abs(d.vy) * 0.86;
          if (Math.abs(d.vy) < 1.2 * s) {
            d.ballistic = false;
            d.vy = 0;
            d.y = d.baseY;
          }
        }
      }
      if (d.y < 16 * s) {
        d.y = 16 * s;
        d.vy = Math.abs(d.vy) * 0.5;
      }
    } else if (benchScenario === 'percussion_ricochet') {
      d.y += d.vy * stepScale;
      if (d.y < wallMargin) {
        d.y = wallMargin;
        d.vy = Math.abs(d.vy);
      } else if (d.y > frameH - wallMargin) {
        d.y = frameH - wallMargin;
        d.vy = -Math.abs(d.vy);
      }
    } else {
      const tremor = 0.95 * s * Math.sin(2.0 * Math.PI * 9.5 * tSec + i * 1.3);
      d.y = d.baseY + tremor;
    }

    if (d.x < wallMargin) {
      d.x = wallMargin;
      d.vx = Math.abs(d.vx);
    } else if (d.x > frameW - wallMargin) {
      d.x = frameW - wallMargin;
      d.vx = -Math.abs(d.vx);
    }
  }

  // Pairwise elastic collisions
  const hitThresh = 14.5 * s;
  if (benchScenario === 'percussion_ricochet') {
    for (let i = 0; i < nActiveDiscs; i++) {
      for (let j = i + 1; j < nActiveDiscs; j++) {
        const a = simDiscs[i];
        const b = simDiscs[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.hypot(dx, dy);
        if (dist > 1e-3 && dist < 16.0 * s) {
          const nx = dx / dist;
          const ny = dy / dist;
          const dvx = a.vx - b.vx;
          const dvy = a.vy - b.vy;
          const relN = dvx * nx + dvy * ny;
          if (relN > 0) {
            a.vx -= relN * nx;
            a.vy -= relN * ny;
            b.vx += relN * nx;
            b.vy += relN * ny;
          }
        }
      }
    }
  } else if (benchScenario !== 'ballistic_cascade' && benchScenario !== 'harmonic_tremor' && benchScenario !== 'model_rocket_flight') {
    const nLanes = benchScenario === 'multi_target_swarm' ? 6 : 2;
    for (let lane = 0; lane < nLanes; lane++) {
      const a = simDiscs[lane * 2];
      const b = simDiscs[lane * 2 + 1];
      if (a.ballistic || b.ballistic) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      if (Math.abs(dx) < hitThresh && Math.abs(dy) < hitThresh && (a.vx - b.vx) * dx > 0) {
        const tmp = a.vx;
        a.vx = b.vx;
        b.vx = tmp;
        const midX = 0.5 * (a.x + b.x);
        for (let p = 0; p < pillars.length; p++) {
          if (midX >= pillars[p].x0 * s && midX <= pillars[p].x1 * s) {
            inPillarCollisions++;
            break;
          }
        }
      }
    }
  }

  // 3. Draw physical textured targets
  if (benchScenario === 'model_rocket_flight') {
    for (let i = 0; i < nActiveDiscs; i++) {
      const d = simDiscs[i];
      const ang = Math.atan2(d.vy, d.vx || 0.001) + Math.PI * 0.5;
      ctx.save();
      ctx.translate(d.x, d.y);
      ctx.rotate(rocketPhase[i] === 2 ? Math.PI * 0.04 * Math.sin(tSec * 4.0 + i) : ang * 0.35);

      if (rocketPhase[i] === 2) {
        // Deployed drogue chute above nosecone (compact so centroid stays on fuselage)
        ctx.fillStyle = 'rgba(255, 107, 53, 0.72)';
        ctx.beginPath();
        ctx.arc(0, -7.8 * s, 4.4 * s, Math.PI, 0);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = 'rgba(215, 228, 245, 0.45)';
        ctx.lineWidth = Math.max(1, 0.7 * s);
        ctx.beginPath();
        ctx.moveTo(-3.6 * s, -7.8 * s);
        ctx.lineTo(0, -3.5 * s);
        ctx.lineTo(3.6 * s, -7.8 * s);
        ctx.stroke();
      }

      // High-contrast fuselage body + bright beacon core (matches PD32 16x16 / 32x32 aperture)
      ctx.fillStyle = '#f4f8ff';
      ctx.fillRect(-3.0 * s, -3.5 * s, 6.0 * s, 7.5 * s);

      // Painted nosecone
      ctx.fillStyle = d.r;
      ctx.beginPath();
      ctx.moveTo(0, -7.5 * s);
      ctx.lineTo(3.2 * s, -3.5 * s);
      ctx.lineTo(-3.2 * s, -3.5 * s);
      ctx.closePath();
      ctx.fill();

      // Tail fins
      ctx.fillStyle = d.r;
      ctx.beginPath();
      ctx.moveTo(-3.0 * s, 1.5 * s);
      ctx.lineTo(-5.0 * s, 4.8 * s);
      ctx.lineTo(-3.0 * s, 4.0 * s);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(3.0 * s, 1.5 * s);
      ctx.lineTo(5.0 * s, 4.8 * s);
      ctx.lineTo(3.0 * s, 4.0 * s);
      ctx.closePath();
      ctx.fill();

      // High-contrast avionics core at center of mass
      ctx.fillStyle = d.core;
      ctx.beginPath();
      ctx.arc(0, 0, 2.2 * s, 0, Math.PI * 2);
      ctx.fill();

      if (rocketPhase[i] === 0) {
        // Compact thrust exhaust plume during boost
        ctx.fillStyle = '#ffb703';
        ctx.beginPath();
        ctx.moveTo(-2.0 * s, 4.0 * s);
        ctx.lineTo(0, 8.8 * s + 1.4 * s * Math.sin(tSec * 42.0 + i));
        ctx.lineTo(2.0 * s, 4.0 * s);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }
  } else {
    for (let i = 0; i < nActiveDiscs; i++) {
      const d = simDiscs[i];
      ctx.fillStyle = d.r;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 7.6 * s, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = d.core;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 3.4 * s, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 4. Draw 100% OPAQUE occlusion pillars ON TOP (wipes out underlying disc pixels completely)
  for (let p = 0; p < pillars.length; p++) {
    const pil = pillars[p];
    const px0 = pil.x0 * s;
    const py0 = pil.y0 * s;
    const pw = (pil.x1 - pil.x0) * s;
    const ph = (pil.y1 - pil.y0) * s;
    ctx.fillStyle = '#222a3a';
    ctx.fillRect(px0, py0, pw, ph);
    ctx.strokeStyle = '#36435c';
    ctx.lineWidth = s;
    ctx.strokeRect(px0 + 0.5 * s, py0 + 0.5 * s, pw - s, ph - s);
  }

  return ctx.getImageData(0, 0, frameW, frameH).data;
}

/// Strict 1-to-1 Nearest-Match Ground-Truth Identity Oracle (no & 255 wrap, no multi-disc aliasing).
function auditOracleSwaps() {
  const s = resScale;
  const pillars = getActivePillars();
  const nActiveDiscs = getActiveSimDiscCount();
  let nPairs = 0;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;
    const latestVis = telemView[base + 44 + 11];
    if (latestVis < 0.5) continue;

    const ix = telemView[base + 8];
    const iy = telemView[base + 9];

    for (let i = 0; i < nActiveDiscs; i++) {
      const d = simDiscs[i];
      let occluded = false;
      for (let p = 0; p < pillars.length; p++) {
        if (d.x > (pillars[p].x0 - 6) * s && d.x < (pillars[p].x1 + 6) * s) {
          occluded = true;
          break;
        }
      }
      if (occluded) continue;

      const dist = Math.hypot(ix - d.x, iy - d.y);
      if (dist < 16.0 * s) {
        oracleCandTrackSlot[nPairs] = m;
        oracleCandDiscIdx[nPairs] = i;
        oracleCandDist[nPairs] = dist;
        nPairs++;
      }
    }
  }

  for (let a = 0; a < nPairs; a++) {
    for (let b = a + 1; b < nPairs; b++) {
      if (oracleCandDist[b] < oracleCandDist[a]) {
        const td = oracleCandDist[a]; oracleCandDist[a] = oracleCandDist[b]; oracleCandDist[b] = td;
        const tm = oracleCandTrackSlot[a]; oracleCandTrackSlot[a] = oracleCandTrackSlot[b]; oracleCandTrackSlot[b] = tm;
        const ti = oracleCandDiscIdx[a]; oracleCandDiscIdx[a] = oracleCandDiscIdx[b]; oracleCandDiscIdx[b] = ti;
      }
    }
  }

  let usedTracks = 0;
  let usedDiscs = 0;
  for (let k = 0; k < nPairs; k++) {
    const m = oracleCandTrackSlot[k];
    const di = oracleCandDiscIdx[k];
    if ((usedTracks & (1 << m)) !== 0 || (usedDiscs & (1 << di)) !== 0) continue;
    usedTracks |= 1 << m;
    usedDiscs |= 1 << di;

    const base = 32 + m * 56;
    const tid = Math.round(telemView[base + 1]);
    if (tid >= 0 && tid < MAX_ORACLE_IDS) {
      const discId = simDiscs[di].id;
      const prevOwner = oracleTrackOwner[tid];
      if (prevOwner === -1) {
        oracleTrackOwner[tid] = discId;
      } else if (prevOwner !== discId) {
        oraclePadicSwaps++;
        oracleTrackOwner[tid] = discId;
      }
    }
  }
}

// ============================================================================
// Pre-allocated Shifted Lanczos Fiedler Eigensolver with Invariant-Subspace
// Orthogonal Completion & Certified Symmetric Jacobi Tridiagonal Solver
// ============================================================================

function findOrthogonalBasisSeed(n, numExisting, outOffset) {
  for (let cand = 0; cand < n; cand++) {
    const invN = 1.0 / n;
    for (let a = 0; a < n; a++) {
      lanczosW[a] = (a === cand ? 1.0 : 0.0) - invN;
    }
    for (let pass = 0; pass < 2; pass++) {
      for (let k = 0; k < numExisting; k++) {
        const kOff = k * GRAPH_NODES;
        let dot = 0.0;
        for (let a = 0; a < n; a++) dot += lanczosQ[kOff + a] * lanczosW[a];
        for (let a = 0; a < n; a++) lanczosW[a] -= dot * lanczosQ[kOff + a];
      }
      let mean = 0.0;
      for (let a = 0; a < n; a++) mean += lanczosW[a];
      mean /= n;
      for (let a = 0; a < n; a++) lanczosW[a] -= mean;
    }

    let norm2 = 0.0;
    for (let a = 0; a < n; a++) norm2 += lanczosW[a] * lanczosW[a];
    if (norm2 > 1e-12) {
      const invNorm = 1.0 / Math.sqrt(norm2);
      for (let a = 0; a < n; a++) {
        lanczosQ[outOffset + a] = lanczosW[a] * invNorm;
      }
      return true;
    }
  }
  return false;
}

let tqlBestCol = 0;
let tqlSecCol = -1;
let tqlThetaMax = -1e30;
let tqlThetaSecond = -1e30;

function solveTridiagTql2(mDim) {
  for (let i = 0; i < mDim; i++) {
    dDiag[i] = lanczosAlpha[i];
    eSub[i] = i + 1 < mDim ? lanczosBeta[i] : 0.0;
  }
  zVec.fill(0.0, 0, mDim * mDim);
  for (let i = 0; i < mDim; i++) zVec[i * mDim + i] = 1.0;

  for (let l = 0; l < mDim; l++) {
    let iter = 0;
    while (iter++ < 40) {
      let m = l;
      for (; m < mDim - 1; m++) {
        if (Math.abs(eSub[m]) <= 1e-14 * (Math.abs(dDiag[m]) + Math.abs(dDiag[m + 1])) + 1e-30) {
          break;
        }
      }
      if (m === l) break;
      let g = (dDiag[l + 1] - dDiag[l]) / (2.0 * eSub[l]);
      const ag = Math.abs(g);
      let r = ag < 1e100 ? Math.sqrt(g * g + 1.0) : ag;
      g = dDiag[m] - dDiag[l] + eSub[l] / (g + (g >= 0.0 ? r : -r));
      let s = 1.0;
      let c = 1.0;
      let p = 0.0;
      for (let i = m - 1; i >= l; i--) {
        const f = s * eSub[i];
        const b = c * eSub[i];
        const r2 = f * f + g * g;
        r = (r2 > 1e-200 && r2 < 1e200) ? Math.sqrt(r2) : Math.hypot(f, g);
        eSub[i + 1] = r;
        if (r < 1e-30) {
          dDiag[i + 1] -= p;
          eSub[m] = 0.0;
          break;
        }
        s = f / r;
        c = g / r;
        g = dDiag[i + 1] - p;
        r = (dDiag[i] - g) * s + 2.0 * c * b;
        p = s * r;
        dDiag[i + 1] = g + p;
        g = c * r - b;
        const rowI = i * mDim;
        const rowI1 = rowI + mDim;
        for (let k = 0; k < mDim; k++) {
          const fz = zVec[rowI1 + k];
          const zki = zVec[rowI + k];
          zVec[rowI1 + k] = s * zki + c * fz;
          zVec[rowI + k] = c * zki - s * fz;
        }
      }
      if (r < 1e-30 && eSub[m] === 0.0) continue;
      dDiag[l] -= p;
      eSub[l] = g;
      eSub[m] = 0.0;
    }
  }

  let bestCol = 0;
  let secCol = -1;
  let thetaMax = -1e30;
  let thetaSecond = -1e30;
  for (let c = 0; c < mDim; c++) {
    const val = dDiag[c];
    if (val > thetaMax) {
      thetaSecond = thetaMax;
      secCol = bestCol;
      thetaMax = val;
      bestCol = c;
    } else if (val > thetaSecond) {
      thetaSecond = val;
      secCol = c;
    }
  }
  tqlBestCol = bestCol;
  tqlSecCol = secCol;
  tqlThetaMax = thetaMax;
  tqlThetaSecond = thetaSecond;
}

function solveShiftedLanczosFiedler() {
  if (wasmExports && typeof wasmExports.chronos_solve_fiedler_lanczos === 'function') {
    return wasmExports.chronos_solve_fiedler_lanczos();
  }
  let n = 0;
  for (let i = 0; i < GRAPH_NODES; i++) {
    if (laplacianView[i * GRAPH_NODES + i] > 1e-6) {
      activeNodeIdx[n++] = i;
    }
  }

  fiedlerView.fill(0.0);
  if (n < 2) {
    wasmExports.chronos_set_fiedler_spectrum(0.0, 0.0);
    return 0.0;
  }

  for (let a = 0; a < n; a++) {
    const ia = activeNodeIdx[a] * GRAPH_NODES;
    const rOff = a * n;
    subL[rOff + a] = laplacianView[ia + activeNodeIdx[a]];
    for (let b = 0; b < a; b++) {
      const v = laplacianView[ia + activeNodeIdx[b]];
      subL[rOff + b] = v;
      subL[b * n + a] = v;
    }
  }

  let mu = 0.0;
  for (let a = 0; a < n; a++) {
    const rOff = a * n;
    let rowSum = subL[rOff + a];
    for (let b = 0; b < n; b++) {
      if (a !== b) {
        rowSum += Math.abs(subL[rOff + b]);
      }
    }
    if (rowSum > mu) mu = rowSum;
  }
  mu += 0.1;

  const mSteps = Math.min(n - 1, MAX_KRYLOV);
  const invN = 1.0 / n;

  let mean0 = 0.0;
  for (let a = 0; a < n; a++) {
    const val = (a + 1) - 0.5 * (n + 1);
    lanczosQ[a] = val;
    mean0 += val;
  }
  mean0 *= invN;
  let norm0 = 0.0;
  for (let a = 0; a < n; a++) {
    lanczosQ[a] -= mean0;
    norm0 += lanczosQ[a] * lanczosQ[a];
  }
  const invNorm0 = 1.0 / Math.sqrt(Math.max(norm0, 1e-18));
  for (let a = 0; a < n; a++) {
    lanczosQ[a] *= invNorm0;
  }

  let actualM = 0;
  let convergedEarly = false;
  for (let j = 0; j < mSteps; j++) {
    const qOff = j * GRAPH_NODES;

    for (let a = 0; a < n; a++) {
      const rOff = a * n;
      let sumL = 0.0;
      for (let b = 0; b < n; b++) {
        sumL += subL[rOff + b] * lanczosQ[qOff + b];
      }
      lanczosW[a] = mu * lanczosQ[qOff + a] - sumL;
    }

    let wMean = 0.0;
    for (let a = 0; a < n; a++) wMean += lanczosW[a];
    wMean *= invN;
    if (j > 0 && lanczosBeta[j - 1] > 0.0) {
      const prevOff = (j - 1) * GRAPH_NODES;
      const betaPrev = lanczosBeta[j - 1];
      for (let a = 0; a < n; a++) {
        lanczosW[a] = (lanczosW[a] - wMean) - betaPrev * lanczosQ[prevOff + a];
      }
    } else {
      for (let a = 0; a < n; a++) lanczosW[a] -= wMean;
    }

    let alpha = 0.0;
    for (let a = 0; a < n; a++) {
      alpha += lanczosQ[qOff + a] * lanczosW[a];
    }
    lanczosAlpha[j] = alpha;

    let preNorm2 = 0.0;
    for (let a = 0; a < n; a++) {
      const nv = lanczosW[a] - alpha * lanczosQ[qOff + a];
      lanczosW[a] = nv;
      preNorm2 += nv * nv;
    }

    for (let pass = 0; pass < 2; pass++) {
      for (let k = 0; k <= j; k++) {
        const kOff = k * GRAPH_NODES;
        let dot = 0.0;
        for (let a = 0; a < n; a++) dot += lanczosQ[kOff + a] * lanczosW[a];
        for (let a = 0; a < n; a++) lanczosW[a] -= dot * lanczosQ[kOff + a];
      }
      let wm = 0.0;
      for (let a = 0; a < n; a++) wm += lanczosW[a];
      wm *= invN;
      let post2 = 0.0;
      for (let a = 0; a < n; a++) {
        const nv = lanczosW[a] - wm;
        lanczosW[a] = nv;
        post2 += nv * nv;
      }
      if (pass === 0 && post2 > 0.5 * preNorm2) break;
    }

    let beta2 = 0.0;
    for (let a = 0; a < n; a++) {
      beta2 += lanczosW[a] * lanczosW[a];
    }

    const beta = Math.sqrt(Math.max(0.0, beta2));
    actualM = j + 1;

    if (j + 1 >= mSteps) {
      lanczosBeta[j] = 0.0;
      break;
    }

    const nextOff = (j + 1) * GRAPH_NODES;
    if (beta < 1e-8) {
      lanczosBeta[j] = 0.0;
      if (!findOrthogonalBasisSeed(n, j + 1, nextOff)) {
        break;
      }
    } else {
      lanczosBeta[j] = beta;
      if (n > 12 && (j === 9 || j === 15)) {
        solveTridiagTql2(actualM);
        const ritzRes1 = beta * Math.abs(zVec[tqlBestCol * actualM + (actualM - 1)]);
        const ritzRes2 = tqlSecCol >= 0 ? beta * Math.abs(zVec[tqlSecCol * actualM + (actualM - 1)]) : 0.0;
        if ((mu - tqlThetaMax) > 1e-4 && ritzRes1 < 1e-8 && ritzRes2 < 1e-7) {
          convergedEarly = true;
          break;
        }
      }
      const invBeta = 1.0 / beta;
      for (let a = 0; a < n; a++) {
        lanczosQ[nextOff + a] = lanczosW[a] * invBeta;
      }
    }
  }

  if (!convergedEarly) solveTridiagTql2(actualM);
  const bestCol = tqlBestCol;
  const thetaMax = tqlThetaMax;
  const thetaSecond = tqlThetaSecond;

  const lambda2 = Math.max(0.0, mu - thetaMax);
  const lambda3 = actualM >= 2 ? Math.max(lambda2, mu - thetaSecond) : lambda2;
  const spectralGap = Math.max(0.0, lambda3 - lambda2);

  let dotPrev = 0.0;
  const bestZOff = bestCol * actualM;
  for (let a = 0; a < n; a++) {
    let comp = 0.0;
    for (let j = 0; j < actualM; j++) {
      comp += lanczosQ[j * GRAPH_NODES + a] * zVec[bestZOff + j];
    }
    eigVecBuf[a] = comp;
    const nodeIdx = activeNodeIdx[a];
    if (nodeIdx < MAX_TARGETS) {
      dotPrev += comp * prevFiedlerTrack[nodeIdx];
    }
  }

  const signFix = (dotPrev < -1e-4 || (Math.abs(dotPrev) <= 1e-4 && eigVecBuf[0] < 0.0)) ? -1.0 : 1.0;
  for (let a = 0; a < n; a++) {
    const nodeIdx = activeNodeIdx[a];
    const val = eigVecBuf[a] * signFix;
    fiedlerView[nodeIdx] = val;
    if (nodeIdx < MAX_TARGETS) {
      prevFiedlerTrack[nodeIdx] = val;
    }
  }

  wasmExports.chronos_set_fiedler_spectrum(lambda2, spectralGap);
  return lambda2;
}

// ============================================================================
// OffscreenCanvas Aerospace HUD & Telemetry Scope Renderers
// ============================================================================

function toHex8(sigHi, sigLo) {
  const u32 = (((sigHi | 0) << 16) | (sigLo | 0)) >>> 0;
  return '0x' + u32.toString(16).toUpperCase().padStart(8, '0');
}

function resolvePrimarySlot() {
  if (!telemView) return -1;
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  if (focusTid > 0) {
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      if (telemView[base] > 0.5 && Math.round(telemView[base + 1]) === focusTid) {
        return m;
      }
    }
  }
  return Math.round(telemView[8]);
}

function updateFrozenApexLocks(uDisp) {
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) {
      apexLockTid[m] = 0;
      apexLockHold[m] = 0;
      apexLockErrPx[m] = -1.0;
      continue;
    }
    const tid = Math.round(telemView[base + 1]);
    const apexFrames = telemView[base + 17];
    const apexX = telemView[base + 15];
    const apexY = telemView[base + 16];
    const curX = telemView[base + 20 + 11];
    const curY = telemView[base + 32 + 11];
    const dispY = telemView[base + 32 + uDisp];

    if (apexFrames >= 0.75) {
      if (apexLockTid[m] !== tid || apexLockHold[m] <= 0 || apexLockErrPx[m] >= 0.0) {
        // Lock new ballistic throw prediction while ascending!
        apexLockTid[m] = tid;
        apexLockPredX[m] = apexX;
        apexLockPredY[m] = apexY;
        apexLockMinY[m] = curY;
        apexLockAtMinX[m] = curX;
        apexLockErrPx[m] = -1.0;
        apexLockHold[m] = Math.max(16, lookaheadK + 12);
      } else if (apexFrames >= 1.25) {
        // Early ascent wide-chord refinement; freezes completely before peak!
        apexLockPredX[m] = 0.65 * apexLockPredX[m] + 0.35 * apexX;
        apexLockPredY[m] = 0.65 * apexLockPredY[m] + 0.35 * apexY;
        apexLockHold[m] = Math.max(apexLockHold[m], lookaheadK + 12);
      }
    }

    if (apexLockTid[m] === tid && apexLockHold[m] > 0) {
      if (curY < apexLockMinY[m]) {
        apexLockMinY[m] = curY;
        apexLockAtMinX[m] = curX;
      }
      // Score full Euclidean peak accuracy once projectile turns downward near predicted peak!
      if (apexLockErrPx[m] < 0.0 && apexFrames <= 0.05 && curY > apexLockMinY[m] + 2.5 * resScale) {
        const err = Math.hypot(
          apexLockAtMinX[m] - apexLockPredX[m],
          apexLockMinY[m] - apexLockPredY[m]
        );
        if (err <= 9.5 * resScale) {
          apexLockErrPx[m] = err;
          lastScoredApexErrPx = err;
        } else {
          apexLockTid[m] = 0;
          apexLockHold[m] = 0;
          apexLockErrPx[m] = -1.0;
          continue;
        }
      }
      if (apexLockErrPx[m] >= 0.0 && dispY > apexLockMinY[m] + 3.0 * resScale) {
        apexLockHold[m]--;
      } else if (apexFrames <= 0.05 && curY <= apexLockMinY[m] + 2.5 * resScale) {
        apexLockHold[m]--;
      }
    } else if (apexLockTid[m] !== tid) {
      apexLockTid[m] = 0;
      apexLockHold[m] = 0;
      apexLockErrPx[m] = -1.0;
    }
  }
}

function drawHudArrow(ctx, x0, y0, x1, y1, headLen) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 2.0) return;
  const ang = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1 - headLen * Math.cos(ang - 0.46), y1 - headLen * Math.sin(ang - 0.46));
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - headLen * Math.cos(ang + 0.46), y1 - headLen * Math.sin(ang + 0.46));
  ctx.stroke();
}

function reserveHudLabelSlot(W, H, prefX, prefY, boxW, placedCount) {
  let lx = Math.max(8, Math.min(W - boxW - 8, prefX));
  let ly = Math.max(38, Math.min(H - 54, prefY));
  for (let pass = 0; pass < 4; pass++) {
    let hit = false;
    for (let k = 0; k < placedCount; k++) {
      const overlapX = lx < hudLabelX[k] + hudLabelW[k] + 4 && lx + boxW + 4 > hudLabelX[k];
      const overlapY = Math.abs(ly - hudLabelY[k]) < 21;
      if (overlapX && overlapY) {
        hit = true;
        break;
      }
    }
    if (!hit) break;
    ly = ly + 22 < H - 54 ? ly + 22 : Math.max(38, ly - 22);
  }
  if (placedCount < MAX_HUD_LABELS) {
    hudLabelX[placedCount] = lx;
    hudLabelY[placedCount] = ly;
    hudLabelW[placedCount] = boxW;
  }
  calOutPt[0] = lx;
  calOutPt[1] = ly;
  return calOutPt;
}

function renderCalibrationOverlay(W, H) {
  if (calMode === 'none' && calArmed === 'none') return;
  hudCtx.save();

  if (calMode === 'ruler' && calPtCount >= 2 && calArmed === 'none') {
    const x0 = calPtsNormX[0] * W;
    const y0 = calPtsNormY[0] * H;
    const x1 = calPtsNormX[1] * W;
    const y1 = calPtsNormY[1] * H;
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.max(1.0, Math.hypot(dx, dy));
    const nx = -dy / len;
    const ny = dx / len;

    hudCtx.strokeStyle = 'rgba(255, 209, 102, 0.85)';
    hudCtx.lineWidth = 1.8;
    hudCtx.setLineDash([6, 4]);
    hudCtx.beginPath();
    hudCtx.moveTo(x0, y0);
    hudCtx.lineTo(x1, y1);
    hudCtx.stroke();
    hudCtx.setLineDash([]);

    hudCtx.beginPath();
    hudCtx.moveTo(x0 - nx * 8, y0 - ny * 8);
    hudCtx.lineTo(x0 + nx * 8, y0 + ny * 8);
    hudCtx.moveTo(x1 - nx * 8, y1 - ny * 8);
    hudCtx.lineTo(x1 + nx * 8, y1 + ny * 8);
    hudCtx.stroke();

    const mx = 0.5 * (x0 + x1);
    const my = 0.5 * (y0 + y1);
    const rTxt = `📏 ${calRulerMeters.toFixed(2)} m`;
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
    hudCtx.fillRect(mx - 36, my - 10, 72, 18);
    hudCtx.strokeStyle = 'rgba(255, 209, 102, 0.65)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(mx - 36, my - 10, 72, 18);
    hudCtx.fillStyle = '#ffd166';
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(rTxt, mx - 30, my + 3);
  } else if (calMode === 'homography' && calHomographyValid && calArmed === 'none') {
    // Perspective plane boundary + 4x4 iso-metric grid lines mapped back through H^-1
    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.22)';
    hudCtx.lineWidth = 1.0;
    hudCtx.setLineDash([4, 4]);
    for (let g = 1; g <= 3; g++) {
      const u = (g / 4.0) * calPlaneW;
      const pA = mapMetricToNorm(u, 0.0);
      const ax = pA[0] * W, ay = pA[1] * H;
      const pB = mapMetricToNorm(u, calPlaneH);
      const bx = pB[0] * W, by = pB[1] * H;
      hudCtx.beginPath();
      hudCtx.moveTo(ax, ay);
      hudCtx.lineTo(bx, by);
      hudCtx.stroke();

      const v = (g / 4.0) * calPlaneH;
      const pC = mapMetricToNorm(0.0, v);
      const cx = pC[0] * W, cy = pC[1] * H;
      const pD = mapMetricToNorm(calPlaneW, v);
      const dx = pD[0] * W, dy = pD[1] * H;
      hudCtx.beginPath();
      hudCtx.moveTo(cx, cy);
      hudCtx.lineTo(dx, dy);
      hudCtx.stroke();
    }
    hudCtx.setLineDash([]);

    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.58)';
    hudCtx.lineWidth = 1.8;
    hudCtx.beginPath();
    for (let i = 0; i < 4; i++) {
      const qx = calPtsNormX[i] * W;
      const qy = calPtsNormY[i] * H;
      if (i === 0) hudCtx.moveTo(qx, qy);
      else hudCtx.lineTo(qx, qy);
    }
    hudCtx.closePath();
    hudCtx.stroke();

    hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (let i = 0; i < 4; i++) {
      const qx = calPtsNormX[i] * W;
      const qy = calPtsNormY[i] * H;
      hudCtx.fillStyle = '#00e5ff';
      hudCtx.beginPath();
      hudCtx.arc(qx, qy, 3.6, 0, Math.PI * 2);
      hudCtx.fill();
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.82)';
      hudCtx.fillRect(qx + 5, qy - 12, 20, 13);
      hudCtx.fillStyle = '#8ef9f3';
      hudCtx.fillText(CORNER_TAGS[i], qx + 7, qy - 2);
    }
  }

  // Interactive click-capture crosshairs & guidance banner when armed (draft points keep prior cal intact on Esc!)
  if (calArmed !== 'none') {
    for (let i = 0; i < calDraftCount; i++) {
      const px = calDraftPtsNormX[i] * W;
      const py = calDraftPtsNormY[i] * H;
      hudCtx.strokeStyle = '#ffd166';
      hudCtx.lineWidth = 2.0;
      hudCtx.beginPath();
      hudCtx.arc(px, py, 7, 0, Math.PI * 2);
      hudCtx.moveTo(px - 12, py); hudCtx.lineTo(px + 12, py);
      hudCtx.moveTo(px, py - 12); hudCtx.lineTo(px, py + 12);
      hudCtx.stroke();
      if (i > 0) {
        hudCtx.setLineDash([4, 3]);
        hudCtx.beginPath();
        hudCtx.moveTo(calDraftPtsNormX[i - 1] * W, calDraftPtsNormY[i - 1] * H);
        hudCtx.lineTo(px, py);
        hudCtx.stroke();
        hudCtx.setLineDash([]);
      }
    }
    const prompt = calArmed === 'ruler'
      ? `📏 CLICK RULER ENDPOINT ${calDraftCount + 1}/2 (${calRulerMeters.toFixed(2)} m · ESC PRESERVES PRIOR)`
      : `⬠ CLICK PLANE CORNER ${calDraftCount + 1}/4 [${CORNER_TAGS[Math.min(3, calDraftCount)]}] (${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m · ESC PRESERVES)`;
    const bw = 504;
    const bx = Math.round((W - bw) * 0.5);
    hudCtx.fillStyle = 'rgba(8, 13, 24, 0.92)';
    hudCtx.fillRect(bx, 40, bw, 24);
    hudCtx.strokeStyle = '#ffd166';
    hudCtx.lineWidth = 1.5;
    hudCtx.strokeRect(bx, 40, bw, 24);
    hudCtx.fillStyle = '#ffd166';
    hudCtx.font = '700 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(prompt, bx + 12, 56);
  }

  hudCtx.restore();
}

// ============================================================================
// CHRONOS-1000 Field Layer Compositor (WebGL2 Instanced + Batched 2D Fallback)
// ============================================================================

const fieldTexRgba = new Uint8Array(FIELD_CELLS * 4);
const fieldTexGlPremul = new Uint8Array(FIELD_CELLS * 4);
const microInstBuf = new Float32Array(MAX_MICRO * 6);
const bodyMemHead = new Int16Array(MAX_RIGID_BODIES + 1);
const bodyMemNext = new Int16Array(MAX_MICRO);
const hullPtsX = new Float32Array(MAX_MICRO);
const hullPtsY = new Float32Array(MAX_MICRO);
const hullIdx = new Int32Array(MAX_MICRO);
const hullPoly = new Int32Array(MAX_MICRO * 2);

let fieldGlCanvas = null;
let fieldGlCtx = null;
let fieldGlBroken = false;
let fieldGlContextLost = false;
let fieldGlQuadProg = null;
let fieldGlQuadVao = null;
let fieldGlTex = null;
let fieldGlStemProg = null;
let fieldGlStemVao = null;
let fieldGlStemUInv = null;
let fieldGlPipProg = null;
let fieldGlPipVao = null;
let fieldGlPipUInv = null;
let fieldGlPipUHalf = null;
let fieldGlInstVbo = null;

function resetFieldWebGlResources() {
  fieldGlCtx = null;
  fieldGlQuadProg = null;
  fieldGlQuadVao = null;
  fieldGlTex = null;
  fieldGlStemProg = null;
  fieldGlStemVao = null;
  fieldGlStemUInv = null;
  fieldGlPipProg = null;
  fieldGlPipVao = null;
  fieldGlPipUInv = null;
  fieldGlPipUHalf = null;
  fieldGlInstVbo = null;
}

let field2dCanvas = null;
let field2dCtx = null;
let field2dImgData = null;

function compileFieldShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(sh) || 'shader compile error';
    gl.deleteShader(sh);
    throw new Error(info);
  }
  return sh;
}

function linkFieldProgram(gl, vsSrc, fsSrc) {
  const vs = compileFieldShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileFieldShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) || 'program link error';
    gl.deleteProgram(prog);
    throw new Error(info);
  }
  return prog;
}

function ensureFieldWebGl2(W, H) {
  if (fieldGlBroken || fieldGlContextLost) return false;
  if (fieldGlCtx) {
    if (fieldGlCtx.isContextLost()) {
      fieldGlContextLost = true;
      resetFieldWebGlResources();
      return false;
    }
    if (fieldGlCanvas.width !== W || fieldGlCanvas.height !== H) {
      fieldGlCanvas.width = W;
      fieldGlCanvas.height = H;
    }
    return true;
  }
  try {
    if (!fieldGlCanvas) {
      fieldGlCanvas = new OffscreenCanvas(W, H);
      fieldGlCanvas.addEventListener('webglcontextlost', (ev) => {
        ev.preventDefault();
        fieldGlContextLost = true;
        resetFieldWebGlResources();
      });
      fieldGlCanvas.addEventListener('webglcontextrestored', () => {
        fieldGlContextLost = false;
        resetFieldWebGlResources();
      });
    } else if (fieldGlCanvas.width !== W || fieldGlCanvas.height !== H) {
      fieldGlCanvas.width = W;
      fieldGlCanvas.height = H;
    }
    const gl = fieldGlCanvas.getContext('webgl2', {
      alpha: true,
      antialias: true,
      depth: false,
      stencil: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
    });
    if (!gl || gl.isContextLost()) {
      if (!gl) fieldGlBroken = true;
      return false;
    }

    const quadVs = `#version 300 es
      out vec2 vUv;
      void main() {
        vec2 p = vec2(
          (gl_VertexID == 1 || gl_VertexID == 3 || gl_VertexID == 4) ? 1.0 : -1.0,
          (gl_VertexID == 2 || gl_VertexID == 4 || gl_VertexID == 5) ? 1.0 : -1.0
        );
        vUv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
        gl_Position = vec4(p, 0.0, 1.0);
      }`;
    const quadFs = `#version 300 es
      precision mediump float;
      uniform sampler2D uFieldTex;
      in vec2 vUv;
      out vec4 fragColor;
      void main() {
        fragColor = texture(uFieldTex, vUv);
      }`;

    const stemVs = `#version 300 es
      layout(location = 0) in vec2 aPos;
      layout(location = 1) in vec2 aVel;
      layout(location = 2) in vec2 aMeta;
      uniform vec2 uInvFrame;
      out vec4 vColor;
      void main() {
        vec2 p = aPos * uInvFrame;
        float spd = length(aVel);
        vec2 stepPx = spd > 1e-3 ? aVel * min(0.16, 22.0 / spd) : vec2(0.0);
        vec2 uv = p + (gl_VertexID == 1 ? stepPx * uInvFrame : vec2(0.0));
        gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
        float code = aMeta.x;
        float insp = aMeta.y;
        vec3 col = insp > 0.5 ? vec3(0.99, 0.88, 0.28)
          : (code > 2.5 ? vec3(1.0, 0.62, 0.18)
          : (code > 1.5 ? vec3(0.96, 0.62, 0.04) : vec3(0.0, 0.90, 1.0)));
        float alpha = (code > 1.5 && code < 2.5) ? 0.36 : 0.72;
        vColor = vec4(col * alpha, alpha);
      }`;
    const stemFs = `#version 300 es
      precision mediump float;
      in vec4 vColor;
      out vec4 fragColor;
      void main() {
        fragColor = vColor;
      }`;

    const pipVs = `#version 300 es
      layout(location = 0) in vec2 aPos;
      layout(location = 1) in vec2 aVel;
      layout(location = 2) in vec2 aMeta;
      uniform vec2 uInvFrame;
      uniform vec2 uHalfNdc;
      out vec2 vLocal;
      out vec2 vMeta;
      void main() {
        vec2 uv = aPos * uInvFrame;
        vec2 q = vec2(
          (gl_VertexID & 1) == 0 ? -1.0 : 1.0,
          (gl_VertexID & 2) == 0 ? -1.0 : 1.0
        );
        float scale = aMeta.y > 0.5 ? 1.65 : (aMeta.x > 2.5 ? 1.15 : 1.0);
        vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) + q * uHalfNdc * scale;
        gl_Position = vec4(ndc, 0.0, 1.0);
        vLocal = q;
        vMeta = aMeta;
      }`;
    const pipFs = `#version 300 es
      precision mediump float;
      in vec2 vLocal;
      in vec2 vMeta;
      out vec4 fragColor;
      void main() {
        float d = abs(vLocal.x) + abs(vLocal.y);
        float cov = 1.0 - smoothstep(0.68, 1.02, d);
        if (cov <= 0.01) discard;
        float code = vMeta.x;
        float insp = vMeta.y;
        vec3 col = insp > 0.5 ? vec3(0.99, 0.88, 0.28)
          : (code > 2.5 ? vec3(1.0, 0.62, 0.18)
          : (code > 1.5 ? vec3(0.96, 0.62, 0.04) : vec3(0.0, 0.90, 1.0)));
        float alpha = ((code > 1.5 && code < 2.5) ? 0.55 : 0.92) * cov;
        fragColor = vec4(col * alpha, alpha);
      }`;

    fieldGlQuadProg = linkFieldProgram(gl, quadVs, quadFs);
    fieldGlStemProg = linkFieldProgram(gl, stemVs, stemFs);
    fieldGlPipProg = linkFieldProgram(gl, pipVs, pipFs);

    fieldGlStemUInv = gl.getUniformLocation(fieldGlStemProg, 'uInvFrame');
    fieldGlPipUInv = gl.getUniformLocation(fieldGlPipProg, 'uInvFrame');
    fieldGlPipUHalf = gl.getUniformLocation(fieldGlPipProg, 'uHalfNdc');

    fieldGlQuadVao = gl.createVertexArray();
    gl.bindVertexArray(fieldGlQuadVao);
    gl.useProgram(fieldGlQuadProg);
    gl.uniform1i(gl.getUniformLocation(fieldGlQuadProg, 'uFieldTex'), 0);

    fieldGlTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fieldGlTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA8, FIELD_GRID_W, FIELD_GRID_H, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, fieldTexRgba
    );

    fieldGlInstVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, fieldGlInstVbo);
    gl.bufferData(gl.ARRAY_BUFFER, microInstBuf.byteLength, gl.DYNAMIC_DRAW);

    const setupInstVao = (vao) => {
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, fieldGlInstVbo);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
      gl.vertexAttribDivisor(0, 1);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 24, 8);
      gl.vertexAttribDivisor(1, 1);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 24, 16);
      gl.vertexAttribDivisor(2, 1);
    };

    fieldGlStemVao = gl.createVertexArray();
    setupInstVao(fieldGlStemVao);
    fieldGlPipVao = gl.createVertexArray();
    setupInstVao(fieldGlPipVao);
    gl.bindVertexArray(null);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    fieldGlCtx = gl;
    return true;
  } catch (_err) {
    fieldGlBroken = true;
    fieldGlCtx = null;
    return false;
  }
}

function packFieldOverlayTexture(dispField, dispModal, validityMask, dispSwarmActive = true) {
  const mask = validityMask | 0;
  const active = Boolean(dispSwarmActive);
  const modeAllowed =
    !active ? false
      : fieldOverlayMode === 1 ? true
      : fieldOverlayMode === 2 ? (mask & 1) !== 0
      : fieldOverlayMode === 3 ? (mask & 2) !== 0
      : fieldOverlayMode === 4 ? Boolean(dispModal && (mask & 8) !== 0)
      : false;

  if (!modeAllowed) {
    fieldTexRgba.fill(0);
    fieldTexGlPremul.fill(0);
    return 1.0;
  }

  let maxMag = 0.0;
  for (let c = 0; c < FIELD_CELLS; c++) {
    const b = c * FIELD_STRIDE;
    const w = dispField[b + 5];
    let m = 0.0;
    if (fieldOverlayMode === 1 || fieldOverlayMode === 2) {
      if (!Number.isFinite(w) || w <= 0.0) continue;
      if (fieldOverlayMode === 1) {
        const u = dispField[b + 0];
        const v = dispField[b + 1];
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
        m = Math.hypot(u, v);
      } else {
        const vort = dispField[b + 2];
        if (!Number.isFinite(vort)) continue;
        m = Math.abs(vort);
      }
    } else if (fieldOverlayMode === 3) {
      const ftle = dispField[b + 4];
      if (!Number.isFinite(ftle) || ftle <= -900.0) continue;
      m = Math.max(0.0, ftle);
    } else {
      continue;
    }
    if (Number.isFinite(m) && m > maxMag) maxMag = m;
  }

  const normMax = fieldOverlayMode === 1
    ? Math.max(24.0, maxMag)
    : (fieldOverlayMode === 2 ? Math.max(0.5, maxMag) : Math.max(0.25, maxMag));

  for (let c = 0; c < FIELD_CELLS; c++) {
    const b = c * FIELD_STRIDE;
    const p = c * 4;
    const w = dispField[b + 5];
    let r = 0;
    let g = 0;
    let bl = 0;
    let a = 0;

    if (fieldOverlayMode === 4) {
      const phi = dispModal[c];
      if (Number.isFinite(phi) && phi >= -3.15 && phi <= 3.15) {
        r = Math.round(127.5 + 118.0 * Math.cos(phi));
        g = Math.round(127.5 + 118.0 * Math.cos(phi - 2.0943951));
        bl = Math.round(127.5 + 118.0 * Math.cos(phi + 2.0943951));
        a = 145;
      }
    } else if (fieldOverlayMode === 3) {
      const ftle = dispField[b + 4];
      if (Number.isFinite(ftle) && ftle > -900.0) {
        const wGain = Number.isFinite(w) && w > 0.0 ? Math.min(1.0, Math.max(0.25, w)) : 0.85;
        const t = Math.min(1.0, Math.max(0.0, ftle) / normMax);
        r = Math.round(70 + t * 185);
        g = Math.round(24 + t * t * 205);
        bl = Math.round(135 + (t < 0.6 ? t * 160 : (1.0 - t) * 240));
        a = Math.round((t < 0.08 ? 4 : 26 + t * 154) * wGain);
      }
    } else if (Number.isFinite(w) && w > 0.0) {
      const wGain = Math.min(1.0, Math.max(0.25, w));
      if (fieldOverlayMode === 1) {
        const u = dispField[b + 0];
        const v = dispField[b + 1];
        if (Number.isFinite(u) && Number.isFinite(v)) {
          const t = Math.min(1.0, Math.hypot(u, v) / normMax);
          r = Math.round(t < 0.55 ? 8 + t * 40 : 30 + (t - 0.55) * 490);
          g = Math.round(35 + t * 205);
          bl = Math.round(90 + (t < 0.65 ? t * 240 : (1.0 - t) * 420));
          a = Math.round((32 + t * 138) * wGain);
        }
      } else if (fieldOverlayMode === 2) {
        const vort = dispField[b + 2];
        if (Number.isFinite(vort)) {
          const s = Math.max(-1.0, Math.min(1.0, vort / normMax));
          const mag = Math.abs(s);
          if (s >= 0.0) {
            r = Math.round(180 + mag * 75);
            g = Math.round(70 + mag * 110);
            bl = 36;
          } else {
            r = 44;
            g = Math.round(115 + mag * 115);
            bl = 255;
          }
          a = Math.round((mag < 0.06 ? 6 : 24 + mag * 150) * wGain);
        }
      }
    }

    fieldTexRgba[p] = r;
    fieldTexRgba[p + 1] = g;
    fieldTexRgba[p + 2] = bl;
    fieldTexRgba[p + 3] = a;
    if (a <= 0) {
      fieldTexGlPremul[p] = 0;
      fieldTexGlPremul[p + 1] = 0;
      fieldTexGlPremul[p + 2] = 0;
      fieldTexGlPremul[p + 3] = 0;
    } else {
      const aF = a / 255.0;
      fieldTexGlPremul[p] = Math.round(r * aF);
      fieldTexGlPremul[p + 1] = Math.round(g * aF);
      fieldTexGlPremul[p + 2] = Math.round(bl * aF);
      fieldTexGlPremul[p + 3] = a;
    }
  }
  return normMax;
}

function computeConvexHullCount(n) {
  if (n < 3) return 0;
  for (let i = 0; i < n; i++) hullIdx[i] = i;
  for (let i = 1; i < n; i++) {
    const key = hullIdx[i];
    const kx = hullPtsX[key];
    const ky = hullPtsY[key];
    let j = i - 1;
    while (j >= 0) {
      const cj = hullIdx[j];
      const cx = hullPtsX[cj];
      if (cx < kx || (cx === kx && hullPtsY[cj] <= ky)) break;
      hullIdx[j + 1] = cj;
      j--;
    }
    hullIdx[j + 1] = key;
  }
  let k = 0;
  for (let i = 0; i < n; i++) {
    const idx = hullIdx[i];
    while (k >= 2) {
      const a = hullPoly[k - 2];
      const b = hullPoly[k - 1];
      const cross =
        (hullPtsX[b] - hullPtsX[a]) * (hullPtsY[idx] - hullPtsY[a]) -
        (hullPtsY[b] - hullPtsY[a]) * (hullPtsX[idx] - hullPtsX[a]);
      if (cross > 0.0) break;
      k--;
    }
    hullPoly[k++] = idx;
  }
  const lowerK = k + 1;
  for (let i = n - 2; i >= 0; i--) {
    const idx = hullIdx[i];
    while (k >= lowerK) {
      const a = hullPoly[k - 2];
      const b = hullPoly[k - 1];
      const cross =
        (hullPtsX[b] - hullPtsX[a]) * (hullPtsY[idx] - hullPtsY[a]) -
        (hullPtsY[b] - hullPtsY[a]) * (hullPtsX[idx] - hullPtsX[a]);
      if (cross > 0.0) break;
      k--;
    }
    hullPoly[k++] = idx;
  }
  return Math.max(0, k - 1);
}

function renderFieldFallback2D(W, H, visCount, sx, sy) {
  if (fieldOverlayMode > 0) {
    if (!field2dCanvas || field2dCanvas.width !== FIELD_GRID_W || field2dCanvas.height !== FIELD_GRID_H) {
      field2dCanvas = new OffscreenCanvas(FIELD_GRID_W, FIELD_GRID_H);
      field2dCtx = field2dCanvas.getContext('2d');
      field2dImgData = new ImageData(FIELD_GRID_W, FIELD_GRID_H);
    }
    field2dImgData.data.set(fieldTexRgba);
    field2dCtx.putImageData(field2dImgData, 0, 0);
    hudCtx.save();
    hudCtx.imageSmoothingEnabled = fieldOverlayMode !== 4;
    if (fieldOverlayMode !== 4) hudCtx.imageSmoothingQuality = 'low';
    hudCtx.drawImage(field2dCanvas, 0, 0, W, H);
    hudCtx.restore();
  }

  if (visCount > 0) {
    hudCtx.save();
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.55)';
    hudCtx.beginPath();
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      const px = microInstBuf[b] * sx;
      const py = microInstBuf[b + 1] * sy;
      const vx = microInstBuf[b + 2];
      const vy = microInstBuf[b + 3];
      const spd = Math.hypot(vx, vy);
      if (spd > 2.0) {
        const gain = Math.min(0.16, 22.0 / spd);
        hudCtx.moveTo(px, py);
        hudCtx.lineTo(px + vx * gain * sx, py + vy * gain * sy);
      }
    }
    hudCtx.stroke();

    hudCtx.fillStyle = 'rgba(0, 229, 255, 0.90)';
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      if (microInstBuf[b + 4] === 1.0) {
        hudCtx.fillRect(microInstBuf[b] * sx - 1.5, microInstBuf[b + 1] * sy - 1.5, 3, 3);
      }
    }
    hudCtx.fillStyle = 'rgba(255, 158, 44, 0.95)';
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      if (microInstBuf[b + 4] === 3.0) {
        hudCtx.fillRect(microInstBuf[b] * sx - 1.8, microInstBuf[b + 1] * sy - 1.8, 3.6, 3.6);
      }
    }
    hudCtx.fillStyle = 'rgba(245, 158, 11, 0.55)';
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      if (microInstBuf[b + 4] === 2.0) {
        hudCtx.fillRect(microInstBuf[b] * sx - 1.0, microInstBuf[b + 1] * sy - 1.0, 2, 2);
      }
    }
    hudCtx.restore();
  }
}

function renderChronos1000FieldLayer(W, H, dispSlot, effK, sx, sy) {
  const ringOk = dispSlot >= 0 && dispSlot < RING_LEN && swarmRingValid[dispSlot] === 1;
  let dispMicro = null;
  let dispField = null;
  let dispBodies = null;
  let dispModal = null;
  let dispSwarm = null;

  if (ringOk) {
    dispMicro = swarmRingMicro[dispSlot];
    dispField = swarmRingField[dispSlot];
    dispBodies = swarmRingBodies[dispSlot];
    dispModal = swarmRingModal[dispSlot];
    dispSwarm = swarmRingTelem[dispSlot];
  } else if (hasChronos1000Exports() && effK === 0 && swarmTelemView && swarmTelemView[1] > 0.5) {
    dispMicro = microTracksView;
    dispField = continuumFieldView;
    dispBodies = rigidBodiesView;
    dispModal = modalFieldView;
    dispSwarm = swarmTelemView;
  } else {
    lastFieldRenderMs = 0.0;
    return;
  }

  if (!dispSwarm || (dispSwarm[1] <= 0.5 && fieldOverlayMode === 0)) {
    lastFieldRenderMs = 0.0;
    return;
  }

  const t0 = performance.now();
  bodyMemHead.fill(-1);
  let visCount = 0;
  let inspFound = false;
  let inspX = 0.0;
  let inspY = 0.0;
  let inspVx = 0.0;
  let inspVy = 0.0;
  let inspBody = 0;
  let inspFtle = 0.0;

  for (let i = 0; i < MAX_MICRO; i++) {
    const base = i * MICRO_STRIDE;
    const st = intOrNull(dispMicro[base + 5], 1, 2);
    if (st === null) continue;
    const px = dispMicro[base + 1];
    const py = dispMicro[base + 2];
    const vx = dispMicro[base + 3];
    const vy = dispMicro[base + 4];
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(vx) || !Number.isFinite(vy)) {
      continue;
    }
    const tid = intOrNull(dispMicro[base + 0], 1, 16777215) ?? 0;
    const bId = intOrNull(dispMicro[base + 6], 0, MAX_RIGID_BODIES) ?? 0;
    const isInsp = tid > 0 && tid === inspectedMicroTid;
    const mOff = visCount * 6;
    microInstBuf[mOff] = px;
    microInstBuf[mOff + 1] = py;
    microInstBuf[mOff + 2] = vx;
    microInstBuf[mOff + 3] = vy;
    microInstBuf[mOff + 4] = st === 1 ? (bId > 0 ? 3.0 : 1.0) : 2.0;
    microInstBuf[mOff + 5] = isInsp ? 1.0 : 0.0;
    visCount++;

    if (st === 1 && bId >= 1 && bId <= MAX_RIGID_BODIES) {
      bodyMemNext[i] = bodyMemHead[bId];
      bodyMemHead[bId] = i;
    }
    if (isInsp) {
      const rawFtle = dispMicro[base + 7];
      inspFound = true;
      inspX = px;
      inspY = py;
      inspVx = vx;
      inspVy = vy;
      inspBody = bId;
      inspFtle = Number.isFinite(rawFtle) && rawFtle > -900.0 ? rawFtle : null;
    }
  }

  const dispSwarmActive = Boolean(dispSwarm && dispSwarm[1] > 0.5);
  const maskVal = intOrNull(dispSwarm[16], 0, 31) ?? 0;
  let fieldNormMax = 1.0;
  if (fieldOverlayMode > 0 && dispField) {
    fieldNormMax = packFieldOverlayTexture(dispField, dispModal, maskVal, dispSwarmActive);
  }

  if (ensureFieldWebGl2(W, H)) {
    const gl = fieldGlCtx;
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (fieldOverlayMode > 0 && dispField) {
      gl.useProgram(fieldGlQuadProg);
      gl.bindVertexArray(fieldGlQuadVao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fieldGlTex);
      const texFilter = fieldOverlayMode === 4 ? gl.NEAREST : gl.LINEAR;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, texFilter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, texFilter);
      gl.texSubImage2D(
        gl.TEXTURE_2D, 0, 0, 0, FIELD_GRID_W, FIELD_GRID_H,
        gl.RGBA, gl.UNSIGNED_BYTE, fieldTexGlPremul
      );
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    if (visCount > 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, fieldGlInstVbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, microInstBuf, 0, visCount * 6);

      gl.useProgram(fieldGlStemProg);
      gl.uniform2f(fieldGlStemUInv, 1.0 / frameW, 1.0 / frameH);
      gl.bindVertexArray(fieldGlStemVao);
      gl.drawArraysInstanced(gl.LINES, 0, 2, visCount);

      gl.useProgram(fieldGlPipProg);
      gl.uniform2f(fieldGlPipUInv, 1.0 / frameW, 1.0 / frameH);
      gl.uniform2f(fieldGlPipUHalf, 4.2 / W, 4.2 / H);
      gl.bindVertexArray(fieldGlPipVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, visCount);
    }

    gl.bindVertexArray(null);
    hudCtx.drawImage(fieldGlCanvas, 0, 0, W, H);
    lastFieldBackend = 'webgl2';
  } else {
    renderFieldFallback2D(W, H, visCount, sx, sy);
    lastFieldBackend = 'canvas2d';
  }

  // Rigid-motion SE(2) convex hulls & angular velocity labels (gated strictly by validity bit 2)
  if (dispBodies && (maskVal & 4) !== 0) {
    hudCtx.save();
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (let b = 0; b < MAX_RIGID_BODIES; b++) {
      const rb = parseRigidBodyDescriptor(dispBodies, b * BODY_STRIDE);
      if (!rb) {
        continue;
      }
      const bId = rb.bodyId;
      const memCount = rb.memberCount;
      const omega = rb.omegaRadS;
      const cx = rb.cx * sx;
      const cy = rb.cy * sy;

      let nPts = 0;
      let mIdx = bodyMemHead[bId];
      let minY = cy - 22.0;
      while (mIdx !== -1 && nPts < MAX_MICRO) {
        const px = dispMicro[mIdx * MICRO_STRIDE + 1] * sx;
        const py = dispMicro[mIdx * MICRO_STRIDE + 2] * sy;
        hullPtsX[nPts] = px;
        hullPtsY[nPts] = py;
        if (py < minY) minY = py;
        nPts++;
        mIdx = bodyMemNext[mIdx];
      }

      const hLen = computeConvexHullCount(nPts);
      if (hLen >= 3) {
        hudCtx.beginPath();
        hudCtx.moveTo(hullPtsX[hullPoly[0]], hullPtsY[hullPoly[0]]);
        for (let k = 1; k < hLen; k++) {
          hudCtx.lineTo(hullPtsX[hullPoly[k]], hullPtsY[hullPoly[k]]);
        }
        hudCtx.closePath();
        hudCtx.fillStyle = 'rgba(255, 158, 44, 0.10)';
        hudCtx.fill();
        hudCtx.strokeStyle = 'rgba(255, 178, 72, 0.88)';
        hudCtx.lineWidth = 1.6;
        hudCtx.stroke();
      }

      // Rotational spin arc at body centroid
      hudCtx.strokeStyle = '#ff9e2c';
      hudCtx.lineWidth = 1.6;
      hudCtx.beginPath();
      const arcR = 11.0;
      if (omega >= 0) {
        hudCtx.arc(cx, cy, arcR, -Math.PI * 0.75, Math.PI * 0.55, false);
      } else {
        hudCtx.arc(cx, cy, arcR, Math.PI * 0.75, -Math.PI * 0.55, true);
      }
      hudCtx.stroke();

      const lbl = `B${bId} · ω=${omega >= 0 ? '+' : ''}${omega.toFixed(2)} rad/s · N=${memCount}`;
      const lw = hudCtx.measureText(lbl).width + 10;
      const lx = Math.max(8, Math.min(W - lw - 8, cx - lw * 0.5));
      const ly = Math.max(42, Math.min(H - 26, minY - 18));
      hudCtx.fillStyle = 'rgba(8, 16, 30, 0.88)';
      hudCtx.fillRect(lx, ly, lw, 16);
      hudCtx.strokeStyle = 'rgba(255, 158, 44, 0.75)';
      hudCtx.lineWidth = 1.0;
      hudCtx.strokeRect(lx, ly, lw, 16);
      hudCtx.fillStyle = '#ffd18a';
      hudCtx.fillText(lbl, lx + 5, ly + 11);
    }
    hudCtx.restore();
  }

  // Inspected point/track diamond reticle + callout
  if (inspFound) {
    const ix = inspX * sx;
    const iy = inspY * sy;
    const spd = Math.hypot(inspVx, inspVy);
    hudCtx.save();
    hudCtx.strokeStyle = '#fde047';
    hudCtx.lineWidth = 1.8;
    hudCtx.beginPath();
    hudCtx.moveTo(ix, iy - 10);
    hudCtx.lineTo(ix + 10, iy);
    hudCtx.lineTo(ix, iy + 10);
    hudCtx.lineTo(ix - 10, iy);
    hudCtx.closePath();
    hudCtx.stroke();
    const tag = `μT${inspectedMicroTid}${inspBody > 0 ? ` · B${inspBody}` : ''} · ${spd.toFixed(1)} px/s | FTLE ${inspFtle !== null ? inspFtle.toFixed(2) : '--'}`;
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const tw = hudCtx.measureText(tag).width + 10;
    const tx = Math.max(8, Math.min(W - tw - 8, ix + 14));
    const ty = Math.max(44, Math.min(H - 26, iy - 16));
    hudCtx.fillStyle = 'rgba(10, 18, 32, 0.90)';
    hudCtx.fillRect(tx, ty, tw, 16);
    hudCtx.strokeStyle = '#fde047';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(tx, ty, tw, 16);
    hudCtx.fillStyle = '#fef08a';
    hudCtx.fillText(tag, tx + 5, ty + 11);
    hudCtx.restore();
  }

  // Non-color Field Overlay legend caption (top-left clear of header strip)
  if (fieldOverlayMode > 0) {
    let cap = '';
    if (fieldOverlayMode === 1) cap = `FIELD: VELOCITY |v| (0..${fieldNormMax.toFixed(0)} px/s)`;
    else if (fieldOverlayMode === 2) cap = `FIELD: VORTICITY CURL v (±${fieldNormMax.toFixed(2)} s⁻¹)`;
    else if (fieldOverlayMode === 3) cap = `FIELD: FTLE 16F RIDGES (0..${fieldNormMax.toFixed(2)} s⁻¹)`;
    else cap = 'FIELD: MODAL PHASE ([-π..+π] rad)';
    if (!dispSwarmActive) cap += ' · STANDBY';
    hudCtx.save();
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const cw = hudCtx.measureText(cap).width + 12;
    hudCtx.fillStyle = 'rgba(7, 13, 24, 0.84)';
    hudCtx.fillRect(10, 44, cw, 17);
    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
    hudCtx.lineWidth = 1;
    hudCtx.strokeRect(10, 44, cw, 17);
    hudCtx.fillStyle = '#bae6fd';
    hudCtx.fillText(cap, 16, 56);
    hudCtx.restore();
  }

  lastFieldRenderMs = performance.now() - t0;
}

function renderMainHud(dispSlot) {
  if (!hudCtx) return;
  const W = hudCanvas.width;   // 960
  const H = hudCanvas.height;  // 540
  const sx = W / frameW;       // 3.0 in SD, 1.5 in HD
  const sy = H / frameH;       // 3.0 in SD, 1.5 in HD
  const fIdx = Math.round(telemView[0]);
  const effK = Math.round(telemView[1]);
  const dispFrame = fIdx - effK;
  const metricOn = isMetricCalibrated();
  const spdUnit = metricOn ? 'm/s' : 'px/s';

  // Activate any Mode C localized shockwaves whose scheduled presentation frame has arrived at t - K
  for (let q = 0; q < SHOCK_POOL; q++) {
    if (shockTriggerFrame[q] >= 0 && fIdx >= shockTriggerFrame[q]) {
      shockAge[q] = 0;
      padFlashAlpha[shockZone[q] & 3] = 1.0;
      shockTriggerFrame[q] = -1;
    }
  }

  // 1. Blit delayed display frame `t - K` from static Wasm ring buffer
  dispStageCtx.putImageData(ringSlotImgData[dispSlot], 0, 0);
  hudCtx.imageSmoothingEnabled = false;
  hudCtx.drawImage(dispStageCanvas, 0, 0, W, H);

  // 2. Optional Live ECA Motion-Residual Overlay (explicitly labeled as frame `t`)
  if (showEcaOverlay && ecaResView) {
    for (let i = 0; i < framePixels; i++) {
      const res = ecaResView[i];
      const off = i * 4;
      if (res > 0.025) {
        const v = Math.min(1.0, res * 3.2);
        ecaOverlayBuf[off] = (v * 30) | 0;
        ecaOverlayBuf[off + 1] = (v * 235) | 0;
        ecaOverlayBuf[off + 2] = (v * 255) | 0;
        ecaOverlayBuf[off + 3] = (v * 185) | 0;
      } else {
        ecaOverlayBuf[off + 3] = 0;
      }
    }
    ecaStageCtx.putImageData(ecaOverlayImgData, 0, 0);
    hudCtx.drawImage(ecaStageCanvas, 0, 0, W, H);
  }

  // 3. In Testbench mode, outline active opaque occlusion zones cleanly on HUD
  if (sourceMode === 'testbench') {
    const pillars = getActivePillars();
    if (pillars.length > 0) {
      hudCtx.save();
      hudCtx.strokeStyle = 'rgba(116, 152, 210, 0.50)';
      hudCtx.setLineDash([6, 4]);
      hudCtx.lineWidth = 1.5;
      for (let p = 0; p < pillars.length; p++) {
        const pil = pillars[p];
        const px = pil.x0 * resScale * sx;
        const py = pil.y0 * resScale * sy;
        const pw = (pil.x1 - pil.x0) * resScale * sx;
        const ph = (pil.y1 - pil.y0) * resScale * sy;
        hudCtx.strokeRect(px, py, pw, ph);
        hudCtx.fillStyle = 'rgba(140, 175, 235, 0.78)';
        hudCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText(`OCCLUDER P${p + 1}`, px + 6, py + 15);
      }
      hudCtx.restore();
    }
  }

  // 3a. CHRONOS-1000 Continuum Heatmap, 1,000+ Micro-Tracks & Rigid-Motion Hulls
  renderChronos1000FieldLayer(W, H, dispSlot, effK, sx, sy);

  // 3b. Metric Ruler or 4-Point Perspective Homography Floor Grid Overlay
  renderCalibrationOverlay(W, H);

  let placedLabels = 0;

  // 4. Mode C: Quadrant Voice Grid + Localized Impact Shockwaves + CAD Deflection & Scatter Protractors
  if (labMode === 2) {
    const labels = ['Q0: SUB-KICK 808', 'Q1: CRISP SNARE', 'Q2: METALLIC FM HAT', 'Q3: HARMONIC TOM'];
    hudCtx.save();
    hudCtx.strokeStyle = 'rgba(0, 240, 255, 0.16)';
    hudCtx.lineWidth = 1;
    hudCtx.beginPath();
    hudCtx.moveTo(W * 0.5, 0); hudCtx.lineTo(W * 0.5, H);
    hudCtx.moveTo(0, H * 0.5); hudCtx.lineTo(W, H * 0.5);
    hudCtx.stroke();

    for (let z = 0; z < 4; z++) {
      const qx = (z & 1) ? W * 0.5 : 0;
      const qy = (z & 2) ? H * 0.5 : 0;
      if (padFlashAlpha[z] > 0.01) {
        hudCtx.strokeStyle = ZONE_COLORS[z];
        hudCtx.globalAlpha = padFlashAlpha[z] * 0.65;
        hudCtx.lineWidth = 2.5;
        hudCtx.strokeRect(qx + 3, qy + 3, W * 0.5 - 6, H * 0.5 - 6);
        hudCtx.globalAlpha = 1.0;
        padFlashAlpha[z] *= 0.82;
      }
      const badgeY = z >= 2 ? H * 0.5 - 52 : 40;
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.72)';
      hudCtx.fillRect(qx + 10, qy + badgeY, 138, 18);
      hudCtx.fillStyle = ZONE_COLORS[z];
      hudCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(labels[z], qx + 15, qy + badgeY + 13);
    }

    // Localized concentric shockwave rings + CAD deflection & restitution protractors!
    for (let q = 0; q < SHOCK_POOL; q++) {
      if (shockAge[q] >= 0) {
        const age = shockAge[q];
        const cx = shockX[q] * sx;
        const cy = shockY[q] * sy;
        const zCol = ZONE_COLORS[shockZone[q] & 3];

        hudCtx.save();
        if (age <= 15) {
          const prog = age / 15.0;
          const alpha = Math.max(0.0, 1.0 - prog);
          const rOuter = 10.0 + prog * 54.0 * (0.65 + 0.5 * shockVel[q]);
          const rInner = 4.0 + prog * 28.0;

          hudCtx.globalAlpha = alpha;
          hudCtx.strokeStyle = zCol;
          hudCtx.lineWidth = 2.6 * (1.0 - 0.5 * prog);
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, rOuter, 0, Math.PI * 2);
          hudCtx.stroke();

          hudCtx.strokeStyle = '#ffffff';
          hudCtx.lineWidth = 1.4;
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, rInner, 0, Math.PI * 2);
          hudCtx.moveTo(cx - 8, cy); hudCtx.lineTo(cx + 8, cy);
          hudCtx.moveTo(cx, cy - 8); hudCtx.lineTo(cx, cy + 8);
          hudCtx.stroke();
        }

        // CAD Deflection Protractor & Restitution readout (held 26 frames for clear reading)
        if (shockRestitution[q] > 0.04 && Math.abs(shockDeflectDeg[q]) >= 2.5) {
          const fade = age <= 18 ? 0.96 : Math.max(0.0, 1.0 - (age - 18) / 9.0);
          hudCtx.globalAlpha = fade;

          // Canvas rays follow actual 2D screen geometry; numeric angle & restitution rectify via J_H!
          const rawInCanvas = -shockInDeg[q] * (Math.PI / 180.0);
          const rawOutCanvas = -(shockInDeg[q] + shockDeflectDeg[q]) * (Math.PI / 180.0);
          const rectSt = rectifyStrikeKinematics(
            shockX[q],
            shockY[q],
            shockInDeg[q],
            shockDeflectDeg[q],
            shockRestitution[q]
          );
          const dispDefl = rectSt[1];
          const dispRest = rectSt[2];

          // Incident continuation dashed reference ray
          hudCtx.strokeStyle = 'rgba(226, 238, 255, 0.68)';
          hudCtx.lineWidth = 1.5;
          hudCtx.setLineDash([4, 3]);
          hudCtx.beginPath();
          hudCtx.moveTo(cx, cy);
          hudCtx.lineTo(cx + Math.cos(rawInCanvas) * 34.0, cy + Math.sin(rawInCanvas) * 34.0);
          hudCtx.stroke();
          hudCtx.setLineDash([]);

          // Outgoing deflected velocity vector
          hudCtx.strokeStyle = zCol;
          hudCtx.lineWidth = 2.2;
          drawHudArrow(
            hudCtx,
            cx,
            cy,
            cx + Math.cos(rawOutCanvas) * 42.0,
            cy + Math.sin(rawOutCanvas) * 42.0,
            7.0
          );

          // Signed CAD deflection arc
          const dArc = Math.atan2(Math.sin(rawOutCanvas - rawInCanvas), Math.cos(rawOutCanvas - rawInCanvas));
          hudCtx.fillStyle = 'rgba(255, 209, 102, 0.20)';
          hudCtx.beginPath();
          hudCtx.moveTo(cx, cy);
          hudCtx.arc(cx, cy, 22.0, rawInCanvas, rawInCanvas + dArc, dArc < 0);
          hudCtx.closePath();
          hudCtx.fill();
          hudCtx.strokeStyle = '#ffd166';
          hudCtx.lineWidth = 1.6;
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, 22.0, rawInCanvas, rawInCanvas + dArc, dArc < 0);
          hudCtx.stroke();

          const signStr = dispDefl >= 0 ? '+' : '';
          const cadTxt = `⚡ Δθ=${signStr}${dispDefl.toFixed(1)}° | v⁺/v⁻=${dispRest.toFixed(2)}`;
          const bw = 176;
          const pos = reserveHudLabelSlot(W, H, cx + 14, cy - 26, bw, placedLabels);
          placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
          hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
          hudCtx.fillRect(pos[0], pos[1], bw, 18);
          hudCtx.strokeStyle = zCol;
          hudCtx.lineWidth = 1.0;
          hudCtx.strokeRect(pos[0], pos[1], bw, 18);
          hudCtx.fillStyle = '#ffd166';
          hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
          hudCtx.fillText(cadTxt, pos[0] + 5, pos[1] + 12);
        } else if (age <= 10) {
          const bw = 110;
          const pos = reserveHudLabelSlot(W, H, cx + 12, cy - 22, bw, placedLabels);
          placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
          hudCtx.fillStyle = 'rgba(6, 10, 18, 0.84)';
          hudCtx.fillRect(pos[0], pos[1], bw, 16);
          hudCtx.fillStyle = zCol;
          hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
          hudCtx.fillText(`⚡ L1 IMPACT Q${shockZone[q]}`, pos[0] + 4, pos[1] + 11);
        }
        hudCtx.restore();

        shockAge[q]++;
        if (shockAge[q] > 26) shockAge[q] = -1;
      }
    }

    // Two-Body Elastic Scatter Wedge (90-deg carom billiard / disc pair collision proof!)
    const rawScDeg = telemView[TELEM_GLOBAL + 2];
    const scEventFrame = Math.round(telemView[TELEM_GLOBAL + 7]);
    if (rawScDeg > 2.0 && scEventFrame > 0 && Math.abs(dispFrame - scEventFrame) <= 24) {
      const scx = telemView[TELEM_GLOBAL + 3] * sx;
      const scy = telemView[TELEM_GLOBAL + 4] * sy;
      const th1Canvas = -telemView[TELEM_GLOBAL + 5] * (Math.PI / 180.0);
      const th2Canvas = -telemView[TELEM_GLOBAL + 6] * (Math.PI / 180.0);
      const rectSc = rectifyScatterKinematics(
        telemView[TELEM_GLOBAL + 3],
        telemView[TELEM_GLOBAL + 4],
        telemView[TELEM_GLOBAL + 5],
        telemView[TELEM_GLOBAL + 6],
        rawScDeg
      );
      const dispScDeg = rectSc[2];
      const dSc = Math.atan2(Math.sin(th2Canvas - th1Canvas), Math.cos(th2Canvas - th1Canvas));

      hudCtx.save();
      hudCtx.fillStyle = 'rgba(0, 240, 255, 0.22)';
      hudCtx.beginPath();
      hudCtx.moveTo(scx, scy);
      hudCtx.arc(scx, scy, 34.0, th1Canvas, th1Canvas + dSc, dSc < 0);
      hudCtx.closePath();
      hudCtx.fill();

      hudCtx.strokeStyle = '#00f0ff';
      hudCtx.lineWidth = 2.0;
      drawHudArrow(hudCtx, scx, scy, scx + Math.cos(th1Canvas) * 44.0, scy + Math.sin(th1Canvas) * 44.0, 6.5);
      drawHudArrow(hudCtx, scx, scy, scx + Math.cos(th2Canvas) * 44.0, scy + Math.sin(th2Canvas) * 44.0, 6.5);

      const scTxt = `⟁ CONTACT SCATTER ${dispScDeg.toFixed(1)}°`;
      const bw = 166;
      const pos = reserveHudLabelSlot(W, H, scx - 62, scy + 18, bw, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.90)';
      hudCtx.fillRect(pos[0], pos[1], bw, 18);
      hudCtx.strokeStyle = '#00f0ff';
      hudCtx.lineWidth = 1.2;
      hudCtx.strokeRect(pos[0], pos[1], bw, 18);
      hudCtx.fillStyle = '#8ef9f3';
      hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(scTxt, pos[0] + 6, pos[1] + 12);
      hudCtx.restore();
    }

    hudCtx.restore();
  }

  const uDisp = (RING_LEN - 1) - effK;
  const effFps = Math.max(8.0, sourceMode === 'testbench' ? 60.0 : sourceFps);
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  const primarySlot = resolvePrimarySlot();
  if (Math.round(telemView[TELEM_GLOBAL + 9]) === 2) {
    trailOwnerTid.fill(-1);
    trailCount.fill(0);
    trailLastFrame.fill(-1);
  }

  updateFrozenApexLocks(uDisp);

  // 5. Render Target Trajectories, Hermite Spline Bridges, Frozen Apex Reticles, Protractors & Kalman Baselines
  if (showHeroTargets) {
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;

    const kbase = TELEM_KINETIC + m * 8;
    const tid = Math.round(telemView[base + 1]);
    const isPrimary = m === primarySlot;
    const isPinned = focusTid !== 0 && tid === focusTid;
    const color = TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length];
    const fxPx = telemView[base + 2];
    const fyPx = telemView[base + 3];
    const fx = fxPx * sx;
    const fy = fyPx * sy;
    const rawDispX = telemView[base + 4] * sx;
    const rawDispY = telemView[base + 5] * sy;
    const kx = telemView[base + 6] * sx;
    const ky = telemView[base + 7] * sy;
    const dispVis = telemView[base + 10] > 0.5;
    const isBridged = telemView[base + 11] > 0.5;
    const sigHex = toHex8(telemView[base + 12], telemView[base + 13]);
    const padicVal = Math.round(telemView[base + 14]);
    const apexFrames = telemView[base + 17];
    const fiedlerCoord = telemView[base + 19];
    const headX = telemView[base + 20 + 11] * sx;
    const headY = telemView[base + 32 + 11] * sy;

    const rawHeadingDeg = telemView[kbase + 0];
    const rawSpeedPxS = telemView[kbase + 1];
    const pivXPx = telemView[kbase + 6];
    const pivYPx = telemView[kbase + 7];
    const hasPivot = pivXPx > -9.0e5 && pivYPx > -9.0e5;

    // Suppress static background scale wake clutter in Mode B real-camera clips unless Show All is active
    if (
      labMode === 1 &&
      sourceMode !== 'testbench' &&
      !showAllOverlays &&
      !isPrimary &&
      !isPinned &&
      !hasPivot &&
      rawSpeedPxS < 20.0 * resScale
    ) {
      continue;
    }

    // Update & render 44-frame (~1.5s) persistent comet trail for smooth visual trajectory history
    const ts = tid & (TRAIL_SLOTS - 1);
    const tBase = ts * TRAIL_LEN;
    if (dispVis || isBridged) {
      if (
        trailOwnerTid[ts] !== tid ||
        trailCount[ts] === 0 ||
        Math.abs(dispFrame - trailLastFrame[ts]) > 14
      ) {
        trailOwnerTid[ts] = tid;
        trailHead[ts] = 1;
        trailCount[ts] = 1;
        trailLastFrame[ts] = dispFrame;
        trailX[tBase] = fxPx;
        trailY[tBase] = fyPx;
      } else if (dispFrame !== trailLastFrame[ts]) {
        const prevIdx = (trailHead[ts] + TRAIL_LEN - 1) % TRAIL_LEN;
        const jumpPx = Math.hypot(fxPx - trailX[tBase + prevIdx], fyPx - trailY[tBase + prevIdx]);
        if (jumpPx > 52.0 * resScale) {
          trailHead[ts] = 1;
          trailCount[ts] = 1;
          trailX[tBase] = fxPx;
          trailY[tBase] = fyPx;
        } else {
          const wr = trailHead[ts];
          trailX[tBase + wr] = fxPx;
          trailY[tBase + wr] = fyPx;
          trailHead[ts] = (wr + 1) % TRAIL_LEN;
          if (trailCount[ts] < TRAIL_LEN) trailCount[ts] += 1;
        }
        trailLastFrame[ts] = dispFrame;
      }
    }

    hudCtx.save();

    const nTrail = trailOwnerTid[ts] === tid ? trailCount[ts] : 0;
    if (nTrail >= 3) {
      const oldest = (trailHead[ts] + TRAIL_LEN - nTrail) % TRAIL_LEN;
      const split1 = Math.max(1, Math.floor(nTrail * 0.42));
      const split2 = Math.max(split1 + 1, Math.floor(nTrail * 0.74));
      hudCtx.strokeStyle = color;
      hudCtx.lineCap = 'round';
      hudCtx.lineJoin = 'round';

      for (let band = 0; band < 3; band++) {
        const i0 = band === 0 ? 0 : (band === 1 ? split1 - 1 : split2 - 1);
        const i1 = band === 0 ? split1 : (band === 1 ? split2 : nTrail - 1);
        if (i1 <= i0) continue;
        hudCtx.globalAlpha = (isPrimary ? 1.0 : 0.78) * (band === 0 ? 0.18 : (band === 1 ? 0.36 : 0.60));
        hudCtx.lineWidth = (isPrimary ? 1.15 : 0.95) * (band === 0 ? 1.2 : (band === 1 ? 1.8 : 2.4));
        hudCtx.beginPath();
        for (let k = i0; k <= i1; k++) {
          const idx = tBase + ((oldest + k) % TRAIL_LEN);
          const tx = trailX[idx] * sx;
          const ty = trailY[idx] * sy;
          if (k === i0) hudCtx.moveTo(tx, ty);
          else hudCtx.lineTo(tx, ty);
        }
        hudCtx.stroke();
      }
      hudCtx.globalAlpha = 1.0;
    }

    // 5a. Virtual Pendulum / Metronome Pivot, Plumb Reference, Tether & Swing Wedge
    if (hasPivot && (isPrimary || isPinned || labMode === 1)) {
      const pivX = pivXPx * sx;
      const pivY = pivYPx * sy;
      const pendKin = computePendulumKinematics(fxPx, fyPx, pivXPx, pivYPx);
      const swingDeg = pendKin[0];
      const radPhys = pendKin[1];
      const plumbAng = fy >= pivY ? Math.PI * 0.5 : -Math.PI * 0.5;
      const tethAng = Math.atan2(fy - pivY, fx - pivX);
      const tethHudLen = Math.hypot(fx - pivX, fy - pivY);
      const wedgeR = Math.min(54.0, Math.max(22.0, tethHudLen * 0.36));

      // Plumb vertical reference dashed ray
      hudCtx.strokeStyle = 'rgba(168, 192, 223, 0.45)';
      hudCtx.lineWidth = 1.3;
      hudCtx.setLineDash([4, 4]);
      hudCtx.beginPath();
      hudCtx.moveTo(pivX, Math.max(38, Math.min(H - 38, pivY)));
      hudCtx.lineTo(pivX, fy >= pivY ? Math.min(H - 36, pivY + tethHudLen * 0.95) : Math.max(38, pivY - tethHudLen * 0.95));
      hudCtx.stroke();

      // Rigid tether ray from virtual pivot to FISTA bob/weight
      hudCtx.strokeStyle = '#38f27a';
      hudCtx.lineWidth = 1.8;
      hudCtx.setLineDash([6, 4]);
      hudCtx.beginPath();
      hudCtx.moveTo(pivX, pivY);
      hudCtx.lineTo(fx, fy);
      hudCtx.stroke();
      hudCtx.setLineDash([]);

      // Swing angle sector wedge (when pivot is near/inside viewport)
      if (pivY >= -30 && pivY <= H + 30 && Math.abs(swingDeg) >= 0.8) {
        const dSw = Math.atan2(Math.sin(tethAng - plumbAng), Math.cos(tethAng - plumbAng));
        hudCtx.fillStyle = 'rgba(56, 242, 122, 0.18)';
        hudCtx.beginPath();
        hudCtx.moveTo(pivX, pivY);
        hudCtx.arc(pivX, pivY, wedgeR, plumbAng, plumbAng + dSw, dSw < 0);
        hudCtx.closePath();
        hudCtx.fill();
        hudCtx.strokeStyle = '#38f27a';
        hudCtx.lineWidth = 1.6;
        hudCtx.beginPath();
        hudCtx.arc(pivX, pivY, wedgeR, plumbAng, plumbAng + dSw, dSw < 0);
        hudCtx.stroke();
      }

      // Pivot bullseye ring (or top-edge directional marker if ceiling mount is above camera frame)
      if (pivY >= 40 && pivY <= H - 36 && pivX >= 8 && pivX <= W - 8) {
        hudCtx.strokeStyle = '#38f27a';
        hudCtx.lineWidth = 2.0;
        hudCtx.beginPath();
        hudCtx.arc(pivX, pivY, 6.5, 0, Math.PI * 2);
        hudCtx.moveTo(pivX - 11, pivY); hudCtx.lineTo(pivX + 11, pivY);
        hudCtx.moveTo(pivX, pivY - 11); hudCtx.lineTo(pivX, pivY + 11);
        hudCtx.stroke();
        hudCtx.fillStyle = 'rgba(6, 10, 18, 0.84)';
        hudCtx.fillRect(pivX + 9, pivY - 8, 56, 15);
        hudCtx.fillStyle = '#38f27a';
        hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText('◎ PIVOT', pivX + 12, pivY + 3);
      } else if (pivY < 40) {
        const tClip = Math.abs(fy - pivY) > 1e-3 ? (44.0 - pivY) / (fy - pivY) : 0.0;
        const edgeX = Math.max(24, Math.min(W - 96, pivX + tClip * (fx - pivX)));
        hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
        hudCtx.fillRect(edgeX - 8, 38, 86, 15);
        hudCtx.fillStyle = '#38f27a';
        hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText('▲ VIRT PIVOT', edgeX - 4, 49);
      }

      const swSign = swingDeg >= 0 ? '+' : '';
      const rStr = metricOn ? `${radPhys.toFixed(2)}m` : `${Math.round(radPhys)}px`;
      const swTxt = `θ=${swSign}${swingDeg.toFixed(1)}° | R=${rStr}`;
      const bw = 138;
      const pos = reserveHudLabelSlot(W, H, fx + 30, fy + 16, bw, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
      hudCtx.fillRect(pos[0], pos[1], bw, 17);
      hudCtx.strokeStyle = 'rgba(56, 242, 122, 0.65)';
      hudCtx.lineWidth = 1.0;
      hudCtx.strokeRect(pos[0], pos[1], bw, 17);
      hudCtx.fillStyle = '#38f27a';
      hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(swTxt, pos[0] + 5, pos[1] + 12);
    }

    // 5b. Past Dual-FISTA Ribbon [0 .. uDisp]
    hudCtx.lineWidth = isPrimary ? 2.8 : 2.2;
    hudCtx.strokeStyle = color;
    hudCtx.beginPath();
    for (let u = 0; u <= uDisp; u++) {
      const px = telemView[base + 20 + u] * sx;
      const py = telemView[base + 32 + u] * sy;
      if (u === 0) hudCtx.moveTo(px, py);
      else hudCtx.lineTo(px, py);
    }
    hudCtx.stroke();

    // 5c. Future Non-Causal Lookahead / Hermite Bridge Ribbon [uDisp .. 11]
    if (uDisp < RING_LEN - 1) {
      hudCtx.lineWidth = isBridged ? 3.0 : 2.0;
      hudCtx.strokeStyle = isBridged ? '#ffd166' : color;
      hudCtx.setLineDash([5, 4]);
      hudCtx.beginPath();
      for (let u = uDisp; u < RING_LEN; u++) {
        const px = telemView[base + 20 + u] * sx;
        const py = telemView[base + 32 + u] * sy;
        if (u === uDisp) hudCtx.moveTo(px, py);
        else hudCtx.lineTo(px, py);
      }
      hudCtx.stroke();
      hudCtx.setLineDash([]);

      hudCtx.fillStyle = isBridged ? '#ffd166' : color;
      hudCtx.beginPath();
      hudCtx.arc(headX, headY, isBridged ? 4.2 : 3.2, 0, Math.PI * 2);
      hudCtx.fill();

      // Explicit Non-Causal Bridge Proof: show unrevised causal coasting ghost vs t NOW re-emergence anchor!
      if (isBridged) {
        const ghostDist = Math.hypot(rawDispX - fx, rawDispY - fy);
        if (ghostDist > 3.5) {
          hudCtx.strokeStyle = 'rgba(255, 107, 107, 0.72)';
          hudCtx.lineWidth = 1.3;
          hudCtx.setLineDash([2, 3]);
          hudCtx.strokeRect(rawDispX - 10, rawDispY - 10, 20, 20);
          hudCtx.setLineDash([]);
        }
        hudCtx.fillStyle = '#ffd166';
        hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText('t NOW', headX + 6, headY - 5);
      }
    }

    // 5d. Tangent Velocity Vector & Flight-Angle Protractor (θ(t-K) & speed in m/s or px/s)
    if (rawSpeedPxS >= 14.0 * resScale) {
      const velCanvasAng = -rawHeadingDeg * (Math.PI / 180.0);
      const arrowLen = Math.min(54.0, Math.max(26.0, (rawSpeedPxS / resScale) * 0.22));
      const tipX = fx + Math.cos(velCanvasAng) * arrowLen;
      const tipY = fy + Math.sin(velCanvasAng) * arrowLen;
      const showFullProtractor = (isPrimary || isPinned || (labMode === 0 && apexLockHold[m] > 0)) && !hasPivot;

      if (showFullProtractor) {
        // Horizontal 0-deg datum reference tick
        const refDir = Math.cos(velCanvasAng) >= 0 ? 1.0 : -1.0;
        const refAng = refDir > 0 ? 0.0 : Math.PI;
        hudCtx.strokeStyle = 'rgba(168, 192, 223, 0.55)';
        hudCtx.lineWidth = 1.2;
        hudCtx.setLineDash([3, 3]);
        hudCtx.beginPath();
        hudCtx.moveTo(fx, fy);
        hudCtx.lineTo(fx + refDir * 32.0, fy);
        hudCtx.stroke();
        hudCtx.setLineDash([]);

        const dVelArc = Math.atan2(Math.sin(velCanvasAng - refAng), Math.cos(velCanvasAng - refAng));
        if (Math.abs(dVelArc) > 0.04) {
          hudCtx.fillStyle = 'rgba(0, 229, 255, 0.16)';
          hudCtx.beginPath();
          hudCtx.moveTo(fx, fy);
          hudCtx.arc(fx, fy, 20.0, refAng, refAng + dVelArc, dVelArc < 0);
          hudCtx.closePath();
          hudCtx.fill();
          hudCtx.strokeStyle = '#00e5ff';
          hudCtx.lineWidth = 1.4;
          hudCtx.beginPath();
          hudCtx.arc(fx, fy, 20.0, refAng, refAng + dVelArc, dVelArc < 0);
          hudCtx.stroke();
        }

        hudCtx.strokeStyle = '#00e5ff';
        hudCtx.lineWidth = 2.1;
        drawHudArrow(hudCtx, fx, fy, tipX, tipY, 6.5);

        const physVel = mapVelocityToPhysical(fxPx, fyPx, rawSpeedPxS, rawHeadingDeg);
        const dispSpd = physVel[0];
        const dispHeading = physVel[1];
        const vValid = physVel[2];
        const thSign = dispHeading >= 0 ? '+' : '';
        const kinTxt = vValid < -0.5
          ? `∠${thSign}${dispHeading.toFixed(1)}° · OUT-OF-PLANE`
          : `∠${thSign}${dispHeading.toFixed(1)}° · ${vValid > 0.5 ? dispSpd.toFixed(2) + 'm/s' : Math.round(dispSpd) + 'px/s'}`;
        const bw = vValid < -0.5 ? 168 : 148;
        const pos = reserveHudLabelSlot(W, H, fx + 30, fy + 12, bw, placedLabels);
        placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
        hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
        hudCtx.fillRect(pos[0], pos[1], bw, 17);
        hudCtx.strokeStyle = vValid < -0.5 ? 'rgba(255, 183, 3, 0.72)' : 'rgba(0, 229, 255, 0.58)';
        hudCtx.lineWidth = 1.0;
        hudCtx.strokeRect(pos[0], pos[1], bw, 17);
        hudCtx.fillStyle = vValid < -0.5 ? '#ffd166' : '#8ef9f3';
        hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText(kinTxt, pos[0] + 5, pos[1] + 12);
      } else if (labMode !== 1) {
        hudCtx.strokeStyle = color;
        hudCtx.globalAlpha = 0.78;
        hudCtx.lineWidth = 1.6;
        drawHudArrow(hudCtx, fx, fy, tipX, tipY, 5.5);
        hudCtx.globalAlpha = 1.0;
      }
    }

    // 5e. Kinematic-Only 2x2 Kalman Baseline Marker (shown always on testbench; on real feeds when diverging)
    const kalDist = Math.hypot(kx - fx, ky - fy);
    const minKalShow = sourceMode === 'testbench' ? 0.0 : 11.0;
    if (showKalmanBaseline && kalDist >= minKalShow && kalDist <= 165.0) {
      hudCtx.strokeStyle = 'rgba(255, 89, 94, 0.85)';
      hudCtx.lineWidth = 1.4;
      hudCtx.setLineDash([3, 3]);
      hudCtx.strokeRect(kx - 13, ky - 13, 26, 26);
      if (kalDist > 6.0) {
        hudCtx.beginPath();
        hudCtx.moveTo(fx, fy);
        hudCtx.lineTo(kx, ky);
        hudCtx.stroke();
      }
      hudCtx.setLineDash([]);
      if (kalDist > 16.0) {
        hudCtx.fillStyle = '#ff7b82';
        hudCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText(`KAL-${tid}`, kx - 14, ky + 23);
      }
    }

    // 5f. Mode A: Frozen Parabolic Apex Reticle & Live Peak Accuracy Score (APEX Δ = X.X px)
    if (labMode === 0 && apexLockTid[m] === tid && apexLockHold[m] > 0) {
      const predX = apexLockPredX[m] * sx;
      const predY = apexLockPredY[m] * sy;
      const scored = apexLockErrPx[m] >= 0.0;
      const arcCol = scored ? '#38f27a' : '#ffd166';

      if (!scored || fy > predY + 6.0) {
        hudCtx.strokeStyle = arcCol;
        hudCtx.lineWidth = 1.8;
        hudCtx.setLineDash([4, 4]);
        hudCtx.beginPath();
        hudCtx.moveTo(fx, fy);
        hudCtx.quadraticCurveTo(0.5 * (fx + predX), predY, predX, predY);
        hudCtx.stroke();
        hudCtx.setLineDash([]);
      }

      hudCtx.strokeStyle = arcCol;
      hudCtx.lineWidth = 2.0;
      hudCtx.beginPath();
      hudCtx.arc(predX, predY, 10, 0, Math.PI * 2);
      hudCtx.moveTo(predX - 15, predY); hudCtx.lineTo(predX + 15, predY);
      hudCtx.moveTo(predX, predY - 15); hudCtx.lineTo(predX, predY + 15);
      hudCtx.stroke();

      const badgeText = scored
        ? `APEX Δ=${apexLockErrPx[m].toFixed(1)}px`
        : `APEX ${Math.max(8, apexFrames * (1000.0 / effFps)).toFixed(0)}ms`;
      const bw = 114;
      const pos = reserveHudLabelSlot(W, H, predX + 13, predY - 18, bw, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
      hudCtx.fillRect(pos[0], pos[1], bw, 17);
      hudCtx.fillStyle = arcCol;
      hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(badgeText, pos[0] + 4, pos[1] + 12);
    }

    // 5g. Primary 2-Adic + Fiedler + FISTA Corner-Bracket Reticle at `t - K`
    const boxR = isPrimary ? 30 : 26;
    const bLen = isPrimary ? 9 : 8;
    hudCtx.strokeStyle = isBridged ? '#ffd166' : color;
    hudCtx.lineWidth = isPrimary ? 2.6 : 2.0;
    hudCtx.beginPath();
    hudCtx.moveTo(fx - boxR, fy - boxR + bLen); hudCtx.lineTo(fx - boxR, fy - boxR); hudCtx.lineTo(fx - boxR + bLen, fy - boxR);
    hudCtx.moveTo(fx + boxR - bLen, fy - boxR); hudCtx.lineTo(fx + boxR, fy - boxR); hudCtx.lineTo(fx + boxR, fy - boxR + bLen);
    hudCtx.moveTo(fx - boxR, fy + boxR - bLen); hudCtx.lineTo(fx - boxR, fy + boxR); hudCtx.lineTo(fx - boxR + bLen, fy + boxR);
    hudCtx.moveTo(fx + boxR - bLen, fy + boxR); hudCtx.lineTo(fx + boxR, fy + boxR); hudCtx.lineTo(fx + boxR, fy + boxR - bLen);
    hudCtx.stroke();

    if (isPinned) {
      hudCtx.strokeStyle = '#ffffff';
      hudCtx.lineWidth = 1.2;
      hudCtx.strokeRect(fx - boxR - 4, fy - boxR - 4, (boxR + 4) * 2, (boxR + 4) * 2);
    }

    const coastNow = slotMetaView ? Math.round(slotMetaView[m * 8 + 1]) : 0;
    const showFullCallout =
      showAllOverlays ||
      isPrimary ||
      isPinned ||
      sourceMode === 'testbench' ||
      (isBridged && coastNow >= 5) ||
      (labMode === 0 && apexLockHold[m] > 0);
    if (showFullCallout) {
      const fSign = fiedlerCoord >= 0 ? 'F+' : 'F-';
      const stateTag = isPinned
        ? 'PINNED'
        : isBridged
        ? 'HERMITE-BRIDGE'
        : dispVis
        ? (isPrimary ? 'FOCUS' : 'LOCK')
        : 'COAST';
      const labelTop = `T${tid} [${stateTag}] v2:${padicVal} ${fSign}`;
      const posTop = reserveHudLabelSlot(W, H, fx - 26, fy > 68 ? fy - 46 : fy + 28, 162, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);

      hudCtx.fillStyle = 'rgba(7, 11, 18, 0.85)';
      hudCtx.fillRect(posTop[0], posTop[1], 162, 19);
      hudCtx.fillStyle = isBridged ? '#ffd166' : color;
      hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(labelTop, posTop[0] + 4, posTop[1] + 13);

      hudCtx.fillStyle = 'rgba(205, 224, 245, 0.82)';
      hudCtx.font = '500 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(sigHex, fx + 29, fy - 2);
    } else {
      if (rawSpeedPxS >= 22.0 * resScale) slotSpeedPillOn[m] = 1;
      else if (rawSpeedPxS < 14.0 * resScale) slotSpeedPillOn[m] = 0;

      let compactTxt = isBridged ? `T${tid}·BRG` : `T${tid}`;
      let compactW = isBridged ? 54 : 34;
      if (slotSpeedPillOn[m]) {
        const secVel = mapVelocityToPhysical(fxPx, fyPx, rawSpeedPxS, rawHeadingDeg);
        const smSpd = slotSmoothedSpeed[m] > 0.01 ? slotSmoothedSpeed[m] : secVel[0];
        if (secVel[2] > 0.5) {
          compactTxt = isBridged
            ? `T${tid}·BRG·${smSpd.toFixed(1)}m/s`
            : `T${tid} · ${smSpd.toFixed(1)}m/s`;
          compactW = isBridged ? 102 : 78;
        } else if (secVel[2] > -0.5) {
          compactTxt = isBridged
            ? `T${tid}·BRG·${Math.round(smSpd)}px/s`
            : `T${tid} · ${Math.round(smSpd)}px/s`;
          compactW = isBridged ? 106 : 82;
        }
      }
      const posSec = reserveHudLabelSlot(W, H, fx - boxR, fy - boxR - 17, compactW, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(7, 11, 18, 0.84)';
      hudCtx.fillRect(posSec[0], posSec[1], compactW, 14);
      hudCtx.fillStyle = isBridged ? '#ffd166' : color;
      hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(compactTxt, posSec[0] + 4, posSec[1] + 10);
    }
    hudCtx.restore();
  }
  }

  // 6. Present-Time Phase-Locked Strobe Border Pulse (Mode B Biophysical Entrainment)
  if (labMode === 1 && strobeEnabled && strobeFlashAlpha > 0.02) {
    const isPosLead = phaseOffsetRad > 0;
    const rgb = isPosLead ? '255, 183, 3' : '0, 240, 255';
    hudCtx.save();
    hudCtx.strokeStyle = `rgba(${rgb}, ${strobeFlashAlpha.toFixed(3)})`;
    hudCtx.lineWidth = 10;
    hudCtx.strokeRect(5, 5, W - 10, H - 10);
    hudCtx.fillStyle = `rgba(${rgb}, ${(strobeFlashAlpha * 0.9).toFixed(3)})`;
    hudCtx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(
      isPosLead ? '⚡ STROBE CUE [+π/2 QUADRATURE LEAD]' : '◆ STROBE CUE [−π/2 QUADRATURE LAG]',
      18,
      H - 18
    );
    hudCtx.restore();
  }
  if (strobeFlashAlpha > 0.001) {
    strobeFlashAlpha *= 0.78;
  }

  // 7. Top-left HUD Status Strip + Top-right Focus Hint + Camera Motion Guard + Bottom Calibration Strip
  const clutterReady = telemView[22] > 0.5;
  const bgMaskPct = Math.round(telemView[23] * 100);
  const clutterStr = clutterReady ? `${telemView[4].toFixed(1)}dB (${bgMaskPct}%bg)` : 'WARMING';
  const ecaTag = showEcaOverlay ? ' | ECA@t' : '';
  const shownSwaps = sourceMode === 'testbench' ? oraclePadicSwaps : Math.round(telemView[5]);

  hudCtx.fillStyle = 'rgba(6, 10, 18, 0.82)';
  hudCtx.fillRect(10, 10, 598, 24);
  hudCtx.fillStyle = '#8ef9f3';
  hudCtx.font = '600 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  hudCtx.fillText(
    `VIEW [t-${effK}] ${frameW}x${frameH} | ECA ${clutterStr} | IDSW:${shownSwaps} | KAL:${Math.round(telemView[6])}${ecaTag}`,
    16,
    26
  );

  const camState = Math.round(telemView[TELEM_GLOBAL + 9]);
  if (camState > 0 && sourceMode !== 'testbench') {
    const camMsg = camState === 2
      ? '⚡ SCENE CUT DETECTED · ECA SUBSPACE & TRACKS AUTO-RESET'
      : `⚠ CAMERA PTZ PAN (${Math.round(telemView[TELEM_GLOBAL + 10])} px/s) · TRACKING & SCALE HELD`;
    const bw = 468;
    const bx = Math.round((W - bw) * 0.5);
    hudCtx.fillStyle = 'rgba(26, 12, 16, 0.92)';
    hudCtx.fillRect(bx, 40, bw, 22);
    hudCtx.strokeStyle = camState === 2 ? '#00f0ff' : '#ffb703';
    hudCtx.lineWidth = 1.4;
    hudCtx.strokeRect(bx, 40, bw, 22);
    hudCtx.fillStyle = camState === 2 ? '#8ef9f3' : '#ffd166';
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(camMsg, bx + 12, 55);
  }

  const pTid = primarySlot >= 0 ? Math.round(telemView[32 + primarySlot * 56 + 1]) : 0;
  const focusStr = focusTid !== 0
    ? `◎ FOCUS: T${focusTid} [PINNED · CLICK TO UNPIN]`
    : pTid > 0
    ? `◎ FOCUS: T${pTid} [AUTO · CLICK PATCH TO PIN]`
    : '◎ AWAITING MOTION PATCH';
  hudCtx.fillStyle = 'rgba(6, 10, 18, 0.82)';
  hudCtx.fillRect(W - 266, 10, 256, 24);
  hudCtx.fillStyle = focusTid !== 0 ? '#ffd166' : '#a8c0df';
  hudCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  hudCtx.fillText(focusStr, W - 258, 26);

  // Bottom-left Scene Gravity Prior badge (Provisional on 1 arc vs Locked >= 2 arcs)
  const sceneGPx = telemView[TELEM_GLOBAL + 0];
  const sceneGConf = telemView[TELEM_GLOBAL + 1];
  const sceneGArcs = Math.round(telemView[TELEM_GLOBAL + 11]);
  if (labMode === 0 && sceneGConf >= 0.20 && sceneGPx > 50.0) {
    const locked = sceneGConf >= AUTO_G_LOCK_CONF;
    const gStatus = locked ? `LOCKED · ${Math.max(2, sceneGArcs)} ARCS` : `PROV · ${Math.max(1, sceneGArcs)}/2 ARCS`;
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.84)';
    hudCtx.fillRect(10, H - 32, 264, 22);
    hudCtx.strokeStyle = locked ? 'rgba(56, 242, 122, 0.58)' : 'rgba(255, 209, 102, 0.48)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(10, H - 32, 264, 22);
    hudCtx.fillStyle = locked ? '#38f27a' : '#ffd166';
    hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(`ĝ = ${Math.round(sceneGPx)} px/s² (${gStatus})`, 16, H - 17);
  }

  // Bottom-right Metric / Homography Calibration badge with explicit Provenance tag
  if (calMode !== 'none') {
    const pBadge = calProvenance === 'assumed' ? 'ASSUMED' : 'SURVEYED';
    const calShort = isCameraMotionSuspendingCal()
      ? '⚠ CALIBRATION HELD (CAM MOTION)'
      : calMode === 'homography' && calHomographyValid
      ? `⬠ PLANE H [${pBadge}]: ${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)}m`
      : calMode === 'auto_g'
      ? (calAutoGLocked
          ? `⚡ AUTO-g [LOCKED]: ${Math.round(getEffectivePxPerMeter())} px/m`
          : `⚡ AUTO-g [PROVISIONAL · px/s]`)
      : `📏 RULER [${pBadge}]: ${calRulerMeters.toFixed(2)}m (${Math.round(getEffectivePxPerMeter())}px/m)`;
    const cw = 324;
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
    hudCtx.fillRect(W - cw - 10, H - 32, cw, 22);
    hudCtx.strokeStyle = calProvenance === 'assumed' ? 'rgba(255, 209, 102, 0.55)' : 'rgba(0, 229, 255, 0.55)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(W - cw - 10, H - 32, cw, 22);
    hudCtx.fillStyle = calProvenance === 'assumed' ? '#ffd166' : '#00e5ff';
    hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(calShort, W - cw - 2, H - 17);
  }
}

/// Side-by-Side Split PiP: Left = 1x Raw Stabilized Crop from `rgba_ring[dispSlot]`,
/// Right = 25x Eulerian Temporal-Contrast Magnified Residual (`chronos_get_pip_rgba_ptr()`).
/// Zero extra Wasm memory bytes and zero per-frame allocations.
function renderEulerianPip() {
  if (!pipCtx || !pipImgData) return;
  const W = pipCanvas.width;   // 248
  const H = pipCanvas.height;  // 144

  pipCtx.fillStyle = '#070b13';
  pipCtx.fillRect(0, 0, W, H);

  const pSlot = resolvePrimarySlot();
  const boxSize = Math.min(H - 42, Math.floor((W - 20) * 0.5));
  const leftX = 6;
  const rightX = W - boxSize - 6;
  const topY = 19;

  pipCtx.fillStyle = '#9db4d4';
  pipCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
  pipCtx.fillText('1x RAW [t−K]', leftX + 2, 13);
  pipCtx.fillStyle = '#00f0ff';
  pipCtx.fillText(`${eulerianGain.toFixed(0)}x TEMPORAL Δ`, rightX + 2, 13);

  if (pSlot >= 0 && telemView[32 + pSlot * 56] > 0.5) {
    const b = 32 + pSlot * 56;
    const fx = telemView[b + 2];
    const fy = telemView[b + 3];
    const halfP = pipDim * 0.5;
    const cropX = Math.max(0, Math.min(frameW - pipDim, Math.round(fx - halfP)));
    const cropY = Math.max(0, Math.min(frameH - pipDim, Math.round(fy - halfP)));

    pipCtx.imageSmoothingEnabled = false;
    // Left viewport: 1x Raw stabilized crop directly from `dispStageCanvas` (`rgba_ring[dispSlot]`)
    pipCtx.drawImage(dispStageCanvas, cropX, cropY, pipDim, pipDim, leftX, topY, boxSize, boxSize);

    // Right viewport: 25x Eulerian temporal-contrast residual from Wasm SIMD128 PiP buffer
    pipStageCtx.putImageData(pipImgData, 0, 0);
    pipCtx.drawImage(pipStageCanvas, 0, 0, pipDim, pipDim, rightX, topY, boxSize, boxSize);

    pipCtx.strokeStyle = 'rgba(168, 192, 223, 0.55)';
    pipCtx.lineWidth = 1;
    pipCtx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);

    pipCtx.strokeStyle = 'rgba(0, 240, 255, 0.70)';
    pipCtx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
    pipCtx.beginPath();
    pipCtx.moveTo(rightX + boxSize * 0.5, topY);
    pipCtx.lineTo(rightX + boxSize * 0.5, topY + boxSize);
    pipCtx.moveTo(rightX, topY + boxSize * 0.5);
    pipCtx.lineTo(rightX + boxSize, topY + boxSize * 0.5);
    pipCtx.stroke();

    const pTid = Math.round(telemView[b + 1]);
    const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
    const modeTag = focusTid === pTid ? 'PINNED' : 'AUTO';
    pipCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
    pipCtx.fillRect(4, H - 19, W - 8, 15);
    pipCtx.fillStyle = focusTid === pTid ? '#ffd166' : '#00f0ff';
    pipCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    pipCtx.fillText(`T${pTid} [${modeTag}] | ${pipDim}x${pipDim} FISTA STABILIZED`, 8, H - 8);
  } else {
    pipCtx.strokeStyle = '#1d293d';
    pipCtx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
    pipCtx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
    pipCtx.fillStyle = '#9db4d4';
    pipCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    pipCtx.fillText('AWAITING PATCH LOCK', 12, H - 8);
  }
}

/// Updates EMA-smoothed per-slot kinematics & sticky hold snapshot before HUD and Roster rendering.
function updateSlotRosterEma() {
  if (!telemView) return;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] <= 0.5) continue;

    const kb = TELEM_KINETIC + m * 8;
    const tid = Math.round(telemView[b + 1]);
    const fxPx = telemView[b + 2];
    const fyPx = telemView[b + 3];
    const physVel = mapVelocityToPhysical(fxPx, fyPx, telemView[kb + 1], telemView[kb + 0]);
    const rawSpd = physVel[0];
    const rawHdg = physVel[1];
    const rawOmega = (calMode === 'homography' && calHomographyValid && physVel[2] > 0.5)
      ? slotRectAngVel[m]
      : telemView[kb + 2];
    const rawEca = slotMetaView ? Math.min(100.0, Math.max(0.0, slotMetaView[m * 8 + 2] * 100.0)) : 0.0;

    const sameTid = slotHoldTid[m] === tid;
    const unitChanged = Math.round(slotLastMetricValid[m]) !== Math.round(physVel[2]);
    const segChanged = slotLastCalSegId[m] !== calSegmentId;
    if (!sameTid || unitChanged || segChanged) {
      slotSmoothedSpeed[m] = rawSpd;
      slotSmoothedHeading[m] = rawHdg;
      slotSmoothedOmega[m] = rawOmega;
      slotSmoothedEca[m] = rawEca;
    } else {
      slotSmoothedSpeed[m] = 0.78 * slotSmoothedSpeed[m] + 0.22 * rawSpd;
      slotSmoothedHeading[m] = wrapDeg180(
        slotSmoothedHeading[m] + 0.24 * wrapDeg180(rawHdg - slotSmoothedHeading[m])
      );
      slotSmoothedOmega[m] = 0.78 * slotSmoothedOmega[m] + 0.22 * rawOmega;
      slotSmoothedEca[m] = 0.80 * slotSmoothedEca[m] + 0.20 * rawEca;
    }

    const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
    slotHoldTid[m] = tid;
    slotHoldLeft[m] = focusTid !== 0 && tid === focusTid ? PINNED_HOLD_FRAMES : SLOT_HOLD_FRAMES;
    slotHoldVeilDrawn[m] = 0;
    slotLastVal[m] = Math.round(telemView[b + 14]);
    slotLastSigHi[m] = Math.round(telemView[b + 12]);
    slotLastSigLo[m] = Math.round(telemView[b + 13]);
    slotLastMetricValid[m] = physVel[2];
    slotLastCalSegId[m] = calSegmentId;

    if (slotMetaView) {
      slotLastAgeFrames[m] = Math.round(slotMetaView[m * 8 + 0]);
      slotLastCoast[m] = Math.round(slotMetaView[m * 8 + 1]);
      slotLastReacq[m] = Math.round(slotMetaView[m * 8 + 3]);
      slotLastFreqHz[m] = slotMetaView[m * 8 + 4];
    }

    if (telemView[kb + 6] > -9.0e5 && telemView[kb + 7] > -9.0e5) {
      const pend = computePendulumKinematics(fxPx, fyPx, telemView[kb + 6], telemView[kb + 7]);
      slotLastPivotValid[m] = 1;
      slotLastPendSwing[m] = sameTid && !segChanged
        ? 0.76 * slotLastPendSwing[m] + 0.24 * pend[0]
        : pend[0];
      slotLastPendRad[m] = sameTid && !segChanged
        ? 0.82 * slotLastPendRad[m] + 0.18 * pend[1]
        : pend[1];
    } else {
      slotLastPivotValid[m] = 0;
      slotLastPendSwing[m] = 0.0;
      slotLastPendRad[m] = 0.0;
    }
  }
}

/// Renders all 16 per-slot Side-by-Side Split `1x RAW [t-K] | 25x TEMPORAL Δ` OffscreenCanvases
/// with exact integer scaling (`128x128` per half: `4x` for `32x32` SD, `2x` for `64x64` HD)
/// and a calm 48-frame (~1.5s) translucent hold-on-loss veil so slots never blink on exit.
function renderAllSlotPips(advanceHold = true) {
  const hasTelem = Boolean(telemView);
  const primarySlot = hasTelem ? resolvePrimarySlot() : -1;
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  const gainTxt = `${eulerianGain.toFixed(0)}x Δ`;

  for (let m = 0; m < MAX_TARGETS; m++) {
    const active = hasTelem && telemView[32 + m * 56] > 0.5;
    const ctx = slotPipCtxs[m];
    const imgData = slotPipImgData[m];

    if (!active && slotHoldLeft[m] > 0) {
      if (ctx && !slotHoldVeilDrawn[m]) {
        slotHoldVeilDrawn[m] = 1;
        const W = slotPipCanvases[m].width;
        const H = slotPipCanvases[m].height;
        const boxSize = Math.min(H - 20, Math.floor((W - 14) * 0.5));
        const leftX = 4;
        const topY = 14;
        ctx.fillStyle = 'rgba(6, 10, 18, 0.48)';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = 'rgba(7, 11, 18, 0.90)';
        ctx.fillRect(leftX + 3, topY + boxSize - 17, W - leftX * 2 - 6, 15);
        ctx.fillStyle = '#9db4d4';
        ctx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillText(`HOLD T${slotHoldTid[m]} · LAST LOCK`, leftX + 7, topY + boxSize - 6);
      }
      if (advanceHold) {
        slotHoldLeft[m]--;
      }
      continue;
    }
    slotHoldVeilDrawn[m] = 0;

    if (!ctx) continue;
    const W = slotPipCanvases[m].width;   // 276
    const H = slotPipCanvases[m].height;  // 148
    const boxSize = Math.min(H - 20, Math.floor((W - 14) * 0.5)); // 128 = 32*4 = 64*2
    const leftX = 4;
    const rightX = W - boxSize - 4;       // 144
    const topY = 14;

    if (active && imgData) {
      const b = 32 + m * 56;
      const tid = Math.round(telemView[b + 1]);
      const fx = telemView[b + 2];
      const fy = telemView[b + 3];
      const dispVis = telemView[b + 10] > 0.5;
      const isBridged = telemView[b + 11] > 0.5;
      const isPinned = focusTid !== 0 && tid === focusTid;
      const isPrimary = m === primarySlot;
      const color = TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length];

      slotIdleDrawn[m] = 0;

      ctx.fillStyle = '#070b13';
      ctx.fillRect(0, 0, W, H);

      ctx.fillStyle = '#9db4d4';
      ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText('1x RAW [t−K]', leftX + 2, 10);
      ctx.fillStyle = '#00f0ff';
      ctx.fillText(gainTxt, rightX + 2, 10);

      const halfP = pipDim * 0.5;
      const cropX = Math.max(0, Math.min(frameW - pipDim, Math.round(fx - halfP)));
      const cropY = Math.max(0, Math.min(frameH - pipDim, Math.round(fy - halfP)));

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(dispStageCanvas, cropX, cropY, pipDim, pipDim, leftX, topY, boxSize, boxSize);

      pipStageCtx.putImageData(imgData, 0, 0);
      ctx.drawImage(pipStageCanvas, 0, 0, pipDim, pipDim, rightX, topY, boxSize, boxSize);

      ctx.lineWidth = 1;
      ctx.strokeStyle = isBridged ? '#ffd166' : (!dispVis ? 'rgba(255, 209, 102, 0.78)' : color);
      ctx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      if (isPinned || isPrimary) {
        ctx.strokeStyle = isPinned ? '#ffffff' : 'rgba(0, 240, 255, 0.85)';
        ctx.strokeRect(leftX - 1.5, topY - 1.5, boxSize + 3, boxSize + 3);
      }

      ctx.strokeStyle = 'rgba(0, 240, 255, 0.72)';
      ctx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      ctx.beginPath();
      ctx.moveTo(rightX + boxSize * 0.5, topY);
      ctx.lineTo(rightX + boxSize * 0.5, topY + boxSize);
      ctx.moveTo(rightX, topY + boxSize * 0.5);
      ctx.lineTo(rightX + boxSize, topY + boxSize * 0.5);
      ctx.stroke();
    } else if (!slotIdleDrawn[m]) {
      slotIdleDrawn[m] = 1;
      ctx.fillStyle = '#060a12';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = '#162033';
      ctx.lineWidth = 1;
      ctx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      ctx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      ctx.fillStyle = '#5c7394';
      ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText('1x RAW', leftX + 3, 10);
      ctx.fillText('Δ STANDBY', rightX + 3, 10);
      ctx.fillStyle = '#4c6180';
      ctx.fillText(`SLOT ${String(m + 1).padStart(2, '0')} IDLE`, W * 0.5 - 32, topY + boxSize * 0.5 + 3);
    }
  }
}

function renderHilbertScope() {
  if (!hilbertCtx) return;
  const W = hilbertCanvas.width;
  const H = hilbertCanvas.height;

  hilbertCtx.fillStyle = '#080c14';
  hilbertCtx.fillRect(0, 0, W, H);

  const pSize = Math.min(H - 28, 140);
  const cx = 14 + pSize * 0.5;
  const cy = H * 0.5 + 6;
  const rad = pSize * 0.43;

  // Auto-range scale across clean physical amplitude tiers (handles ±0.8px micro-tremor to ±28px metronomes!)
  let peakAbs = 0.0;
  for (let k = 0; k < 64; k++) {
    const idx = (scopeWriteIdx + SCOPE_HIST - 64 + k) % SCOPE_HIST;
    if (histPhaseValid[idx]) {
      const ar = Math.abs(histTremorR[idx]);
      const ah = Math.abs(histTremorH[idx]);
      if (ar > peakAbs) peakAbs = ar;
      if (ah > peakAbs) peakAbs = ah;
    }
  }
  const peakNorm = peakAbs / resScale;
  let rangeBasePx = 1.5;
  if (peakNorm > 10.0) rangeBasePx = 28.0;
  else if (peakNorm > 3.6) rangeBasePx = 12.0;
  else if (peakNorm > 1.35) rangeBasePx = 4.0;
  const fullScalePx = rangeBasePx * resScale;
  const polarScale = rad / fullScalePx;
  const invWaveScale = 1.0 / fullScalePx;

  hilbertCtx.strokeStyle = '#1d293d';
  hilbertCtx.lineWidth = 1;
  hilbertCtx.beginPath();
  hilbertCtx.arc(cx, cy, rad, 0, Math.PI * 2);
  hilbertCtx.arc(cx, cy, rad * 0.5, 0, Math.PI * 2);
  hilbertCtx.moveTo(cx - rad, cy); hilbertCtx.lineTo(cx + rad, cy);
  hilbertCtx.moveTo(cx, cy - rad); hilbertCtx.lineTo(cx, cy + rad);
  hilbertCtx.stroke();

  hilbertCtx.fillStyle = '#738db5';
  hilbertCtx.font = '600 8px ui-monospace, SFMono-Regular, Menlo, monospace';
  hilbertCtx.fillText(`±${fullScalePx.toFixed(fullScalePx < 4 ? 1 : 0)}px`, cx - rad + 2, cy + rad + 10);

  const tx = cx + Math.cos(phaseOffsetRad) * rad;
  const ty = cy - Math.sin(phaseOffsetRad) * rad;
  hilbertCtx.strokeStyle = phaseOffsetRad > 0 ? '#ffb703' : '#00f0ff';
  hilbertCtx.setLineDash([3, 3]);
  hilbertCtx.beginPath();
  hilbertCtx.moveTo(cx, cy);
  hilbertCtx.lineTo(tx, ty);
  hilbertCtx.stroke();
  hilbertCtx.setLineDash([]);

  hilbertCtx.strokeStyle = '#38f27a';
  hilbertCtx.lineWidth = 1.6;
  hilbertCtx.beginPath();
  let polarPenDown = false;
  let prevTid = -1;
  for (let k = 0; k < 48; k++) {
    const idx = (scopeWriteIdx + SCOPE_HIST - 48 + k) % SCOPE_HIST;
    const tidHere = histPipTid[idx];
    if (!histPhaseValid[idx] || tidHere < 0) {
      polarPenDown = false;
      prevTid = -1;
      continue;
    }
    const rx = cx + Math.max(-rad, Math.min(rad, histTremorR[idx] * polarScale));
    const hy = cy - Math.max(-rad, Math.min(rad, histTremorH[idx] * polarScale));
    if (!polarPenDown || tidHere !== prevTid) {
      hilbertCtx.moveTo(rx, hy);
      polarPenDown = true;
    } else {
      hilbertCtx.lineTo(rx, hy);
    }
    prevTid = tidHere;
  }
  hilbertCtx.stroke();

  const curIdx = (scopeWriteIdx + SCOPE_HIST - 1) % SCOPE_HIST;
  const headX = cx + Math.max(-rad, Math.min(rad, histTremorR[curIdx] * polarScale));
  const headY = cy - Math.max(-rad, Math.min(rad, histTremorH[curIdx] * polarScale));
  if (histPhaseValid[curIdx]) {
    hilbertCtx.fillStyle = '#ffffff';
    hilbertCtx.beginPath();
    hilbertCtx.arc(headX, headY, 3.2, 0, Math.PI * 2);
    hilbertCtx.fill();
  }

  const waveX0 = 26 + pSize;
  const waveW = W - waveX0 - 10;
  const waveMidY = H * 0.5 + 6;
  const waveAmp = (H - 40) * 0.42;

  hilbertCtx.strokeStyle = '#182234';
  hilbertCtx.beginPath();
  hilbertCtx.moveTo(waveX0, waveMidY);
  hilbertCtx.lineTo(waveX0 + waveW, waveMidY);
  hilbertCtx.stroke();

  hilbertCtx.strokeStyle = 'rgba(255, 183, 3, 0.38)';
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    if (histStrobe[idx]) {
      const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
      hilbertCtx.beginPath();
      hilbertCtx.moveTo(wx, 22);
      hilbertCtx.lineTo(wx, H - 10);
      hilbertCtx.stroke();
    }
  }

  // R(t) trace (breaks cleanly across target switches)
  hilbertCtx.strokeStyle = '#00f0ff';
  hilbertCtx.lineWidth = 1.5;
  hilbertCtx.beginPath();
  let rPenDown = false;
  prevTid = -1;
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const tidHere = histPipTid[idx];
    if (tidHere < 0) {
      rPenDown = false;
      prevTid = -1;
      continue;
    }
    const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
    const wy = waveMidY - Math.max(-1, Math.min(1, histTremorR[idx] * invWaveScale)) * waveAmp;
    if (!rPenDown || tidHere !== prevTid) {
      hilbertCtx.moveTo(wx, wy);
      rPenDown = true;
    } else {
      hilbertCtx.lineTo(wx, wy);
    }
    prevTid = tidHere;
  }
  hilbertCtx.stroke();

  // H{R}(t) quadrature trace
  hilbertCtx.strokeStyle = '#e840f8';
  hilbertCtx.lineWidth = 1.3;
  hilbertCtx.beginPath();
  let hPenDown = false;
  prevTid = -1;
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const tidHere = histPipTid[idx];
    if (!histPhaseValid[idx] || tidHere < 0) {
      hPenDown = false;
      prevTid = -1;
      continue;
    }
    const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
    const wy = waveMidY - Math.max(-1, Math.min(1, histTremorH[idx] * invWaveScale)) * waveAmp;
    if (!hPenDown || tidHere !== prevTid) {
      hilbertCtx.moveTo(wx, wy);
      hPenDown = true;
    } else {
      hilbertCtx.lineTo(wx, wy);
    }
    prevTid = tidHere;
  }
  hilbertCtx.stroke();

  const pSlot = Math.round(telemView[8]);
  const cohPct = pSlot >= 0 ? Math.round(telemView[TELEM_EXCUR + pSlot * 8 + 6] * 100) : 0;
  hilbertCtx.fillStyle = '#9db4d4';
  hilbertCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const phaseValid = telemView[26] > 0.5;
  const freqValid = telemView[TELEM_GLOBAL + 12] > 0.5 && telemView[13] > 0.02;
  if (phaseValid) {
    const phiDeg = (telemView[11] * 180 / Math.PI).toFixed(0);
    const fTxt = freqValid ? `f=${telemView[13].toFixed(2)}Hz` : 'APERIODIC';
    hilbertCtx.fillText(
      `HILBERT φ=${phiDeg}° | ${fTxt} | COH ${cohPct}% | R CYAN · H{R} MAGENTA`,
      10,
      14
    );
  } else {
    const effK = Math.round(telemView[1]);
    hilbertCtx.fillText(
      effK < 7 ? 'HILBERT QUADRATURE GATED (REQUIRES K >= 7 FUTURE SUPPORT)' : 'HILBERT WINDOW WARMING / AWAITING OSCILLATION',
      10,
      14
    );
  }
}

function renderSpectralAndRingScope() {
  if (!spectralCtx) return;
  const W = spectralCanvas.width;
  const H = spectralCanvas.height;

  spectralCtx.fillStyle = '#080c14';
  spectralCtx.fillRect(0, 0, W, H);

  const midH = Math.floor(H * 0.52);
  spectralCtx.fillStyle = '#9db4d4';
  spectralCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  spectralCtx.fillText(
    `PRE-ASSIGNMENT FIEDLER λ₂=${telemView[7].toFixed(4)} (Δλ=${telemView[24].toFixed(3)})`,
    10,
    14
  );

  const barBaseY = Math.floor(midH * 0.58);
  spectralCtx.strokeStyle = '#1d293d';
  spectralCtx.beginPath();
  spectralCtx.moveTo(10, barBaseY);
  spectralCtx.lineTo(W * 0.56, barBaseY);
  spectralCtx.stroke();

  const barAreaW = W * 0.54;
  const barSlotW = barAreaW / MAX_TARGETS;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    const bx = 12 + m * barSlotW;
    if (telemView[base] < 0.5) {
      spectralCtx.fillStyle = '#151e2e';
      spectralCtx.fillRect(bx, barBaseY - 2, barSlotW - 4, 4);
      continue;
    }
    const v2 = fiedlerView[m];
    const bh = Math.max(-28, Math.min(28, v2 * 36));
    spectralCtx.fillStyle = v2 >= 0 ? '#00f0ff' : '#ff4d6d';
    if (bh >= 0) spectralCtx.fillRect(bx, barBaseY - bh, barSlotW - 4, Math.max(2, bh));
    else spectralCtx.fillRect(bx, barBaseY, barSlotW - 4, Math.max(2, -bh));
  }

  const spX0 = W * 0.60;
  const spW = W - spX0 - 10;
  spectralCtx.strokeStyle = '#ffd166';
  spectralCtx.lineWidth = 1.4;
  spectralCtx.beginPath();
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const x = spX0 + (i / (SCOPE_HIST - 1)) * spW;
    const y = midH - 8 - Math.min(1.0, histLambda2[idx] * 0.35) * (midH - 28);
    if (i === 0) spectralCtx.moveTo(x, y);
    else spectralCtx.lineTo(x, y);
  }
  spectralCtx.stroke();

  // Chronological 12-Slot Non-Causal Timeline: u = 0 (t-11) .. uDisp [t-K VIEW] .. 11 [t NOW]
  const ringTop = midH + 20;
  const effK = Math.round(telemView[1]);
  const uDisp = (RING_LEN - 1) - effK;
  spectralCtx.fillStyle = '#9db4d4';
  spectralCtx.fillText(`NON-CAUSAL TIMELINE: PAST [t−11] → VIEW [t−${effK}] → FUTURE [t NOW]`, 10, midH + 12);

  const fIdx = Math.round(telemView[0]);
  const slotOldest = (fIdx + 1) % RING_LEN;
  const cellW = (W - 20) / RING_LEN;
  const cellH = H - ringTop - 8;
  const pIdx = Math.round(telemView[8]);
  const pBridged = pIdx >= 0 && telemView[32 + pIdx * 56 + 11] > 0.5;

  for (let u = 0; u < RING_LEN; u++) {
    const cx = 10 + u * cellW;
    const isNow = u === RING_LEN - 1;
    const isDisp = u === uDisp;
    const isFuture = u > uDisp;
    const relLag = u - (RING_LEN - 1); // -11 .. 0
    const ringSlot = (slotOldest + u) % RING_LEN;

    let visState = 1;
    if (pIdx >= 0) {
      visState = telemView[32 + pIdx * 56 + 44 + u] > 0.5 ? 1 : 0;
    }

    spectralCtx.fillStyle = isDisp
      ? 'rgba(0, 240, 255, 0.30)'
      : isNow
      ? 'rgba(56, 242, 122, 0.28)'
      : visState === 0
      ? 'rgba(255, 209, 102, 0.24)'
      : isFuture
      ? 'rgba(0, 180, 216, 0.12)'
      : '#101826';
    spectralCtx.fillRect(cx + 1, ringTop, cellW - 3, cellH);

    spectralCtx.strokeStyle = isDisp
      ? '#00f0ff'
      : isNow
      ? '#38f27a'
      : visState === 0
      ? '#ffd166'
      : isFuture
      ? '#1d4e68'
      : '#26344d';
    spectralCtx.lineWidth = isDisp || isNow ? 1.8 : 1.0;
    spectralCtx.strokeRect(cx + 1.5, ringTop + 0.5, cellW - 4, cellH - 1);

    spectralCtx.fillStyle = isDisp ? '#ffffff' : '#c6d8f0';
    spectralCtx.font = '600 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    const topLbl = relLag === 0 ? 't' : `t${relLag}`;
    spectralCtx.fillText(topLbl, cx + 3, ringTop + 12);

    spectralCtx.fillStyle = '#5c7599';
    spectralCtx.font = '500 7.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    spectralCtx.fillText(`s${ringSlot}`, cx + 4, ringTop + 23);

    spectralCtx.font = '700 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    if (isDisp) {
      spectralCtx.fillStyle = '#00f0ff';
      spectralCtx.fillText('VIEW', cx + 3, ringTop + cellH - 5);
    } else if (isNow) {
      spectralCtx.fillStyle = '#38f27a';
      spectralCtx.fillText('NOW', cx + 4, ringTop + cellH - 5);
    } else if (visState === 0) {
      spectralCtx.fillStyle = '#ffd166';
      spectralCtx.fillText(pBridged ? 'BRG' : 'OCC', cx + 4, ringTop + cellH - 5);
    } else if (isFuture) {
      spectralCtx.fillStyle = '#6ec8e8';
      spectralCtx.fillText('+LK', cx + 4, ringTop + cellH - 5);
    }
  }
}

function pushPhysicsPhaseSample() {
  if (!telemView) return;
  const pIdx = resolvePrimarySlot();
  const w = phaseHistWrite;
  if (pIdx >= 0 && telemView[32 + pIdx * 56] > 0.5) {
    const b = 32 + pIdx * 56;
    const kb = TELEM_KINETIC + pIdx * 8;
    const pb = TELEM_PHYSICS_START + pIdx * TELEM_PHYSICS_STRIDE;
    const spd = safeFinite(telemView[kb + 1]);
    const hRad = safeFinite(telemView[kb + 0]) * (Math.PI / 180.0);
    phaseHistTid[w] = Math.round(safeFinite(telemView[b + 1]));
    phaseHistX[w] = safeFinite(telemView[b + 2]);
    phaseHistY[w] = safeFinite(telemView[b + 3]);
    phaseHistVx[w] = spd * Math.cos(hRad);
    phaseHistVy[w] = spd * Math.sin(hRad);
    phaseHistE[w] = safeFinite(telemView[pb + 12]);
    phaseHistRegime[w] = Math.max(0, Math.min(8, Math.round(safeFinite(telemView[pb + 4]))));
    phaseHistJump[w] = Math.max(0, Math.min(4, Math.round(safeFinite(telemView[pb + 15]))));
  } else {
    phaseHistTid[w] = -1;
    phaseHistX[w] = 0.0;
    phaseHistY[w] = 0.0;
    phaseHistVx[w] = 0.0;
    phaseHistVy[w] = 0.0;
    phaseHistE[w] = 0.0;
    phaseHistRegime[w] = 0;
    phaseHistJump[w] = 0;
  }
  const eb = TELEM_ENSEMBLE_START;
  phaseHistEnsK[w] = safeFinite(telemView[eb + 3]);
  phaseHistKurR[w] = safeFinite(telemView[eb + 0]);
  phaseHistKurN[w] = Math.max(0, Math.min(255, Math.round(safeFinite(telemView[eb + 2]))));
  phaseHistWrite = (phaseHistWrite + 1) % PHASE_HIST;
  if (phaseHistCount < PHASE_HIST) phaseHistCount++;
}

function renderPhysicsPhaseCanvas(force = false) {
  if (!physicsPhaseCtx || !physicsPhaseCanvas) return;
  const nowMs = performance.now();
  if (!force && nowMs - lastPhysicsPhaseRenderMs < 66.0) return;
  lastPhysicsPhaseRenderMs = nowMs;

  const W = physicsPhaseCanvas.width;
  const H = physicsPhaseCanvas.height;
  const ctx = physicsPhaseCtx;

  ctx.fillStyle = '#070b13';
  ctx.fillRect(0, 0, W, H);

  const lx = 6;
  const ly = 6;
  const lw = Math.max(96, Math.floor(W * 0.49) - 8);
  const lh = Math.max(60, H - 12);
  const rx = lx + lw + 6;
  const ry = 6;
  const rw = Math.max(96, W - rx - 6);
  const rh = lh;

  ctx.fillStyle = '#0b1220';
  ctx.fillRect(lx, ly, lw, lh);
  ctx.fillRect(rx, ry, rw, rh);
  ctx.strokeStyle = '#192438';
  ctx.lineWidth = 1;
  ctx.strokeRect(lx + 0.5, ly + 0.5, lw - 1, lh - 1);
  ctx.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rh - 1);

  const px0 = lx + 6;
  const py0 = ly + 20;
  const pw = Math.max(20, lw - 12);
  const ph = Math.max(20, lh - 26);

  const ex0 = rx + 6;
  const ey0 = ry + 20;
  const ew = Math.max(20, rw - 12);
  const eh = Math.max(24, Math.floor((rh - 44) * 0.62));
  const kx0 = rx + 6;
  const ky0 = ey0 + eh + 18;
  const kw = ew;
  const kh = Math.max(18, ry + rh - ky0 - 6);

  ctx.font = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  if (phaseHistCount <= 0 || !telemView) {
    ctx.fillStyle = '#9db4d4';
    ctx.fillText('PHASE PORTRAIT (q, qdot)', lx + 6, ly + 13);
    ctx.fillText('CONSERVATION E(t) & R_sync', rx + 6, ry + 13);
    ctx.fillStyle = '#5c7394';
    ctx.fillText('AWAITING TRACK DYNAMICS...', px0 + 8, py0 + ph * 0.5);
    ctx.fillText('AWAITING ENERGY & R_sync...', ex0 + 8, ey0 + eh * 0.55);
    return;
  }

  const lastIdx = (phaseHistWrite + PHASE_HIST - 1) % PHASE_HIST;
  const curTid = phaseHistTid[lastIdx];
  const curRegime = phaseHistRegime[lastIdx];
  const curR = phaseHistKurR[lastIdx];
  const curN = phaseHistKurN[lastIdx];

  // Left pane: contiguous phase portrait for current primary track
  let segLen = 0;
  if (curTid > 0) {
    for (let k = 0; k < phaseHistCount; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - 1 - k) % PHASE_HIST;
      if (phaseHistTid[idx] !== curTid) break;
      segLen++;
    }
  }

  ctx.strokeStyle = '#182438';
  ctx.lineWidth = 1;
  ctx.strokeRect(px0 + 0.5, py0 + 0.5, pw - 1, ph - 1);
  const pcx = px0 + pw * 0.5;
  const pcy = py0 + ph * 0.5;
  ctx.strokeStyle = 'rgba(100, 128, 168, 0.25)';
  ctx.beginPath();
  ctx.moveTo(px0, pcy);
  ctx.lineTo(px0 + pw, pcy);
  ctx.moveTo(pcx, py0);
  ctx.lineTo(pcx, py0 + ph);
  ctx.stroke();

  if (segLen >= 2) {
    let sumX = 0.0;
    let sumY = 0.0;
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      sumX += phaseHistX[idx];
      sumY += phaseHistY[idx];
    }
    const meanX = sumX / segLen;
    const meanY = sumY / segLen;

    let varX = 0.0;
    let varY = 0.0;
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      const dx = phaseHistX[idx] - meanX;
      const dy = phaseHistY[idx] - meanY;
      varX += dx * dx;
      varY += dy * dy;
    }
    const useY = varY > varX * 1.15;

    let maxQ = 4.0;
    let maxV = 15.0;
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      const q = useY ? -(phaseHistY[idx] - meanY) : (phaseHistX[idx] - meanX);
      const v = useY ? phaseHistVy[idx] : phaseHistVx[idx];
      if (Math.abs(q) > maxQ) maxQ = Math.abs(q);
      if (Math.abs(v) > maxV) maxV = Math.abs(v);
    }
    maxQ *= 1.15;
    maxV *= 1.15;

    const col = TARGET_COLORS[Math.max(0, curTid - 1) % TARGET_COLORS.length];
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      const q = useY ? -(phaseHistY[idx] - meanY) : (phaseHistX[idx] - meanX);
      const v = useY ? phaseHistVy[idx] : phaseHistVx[idx];
      const gx = pcx + (q / maxQ) * (pw * 0.45);
      const gy = pcy - (v / maxV) * (ph * 0.45);
      if (k === 0) ctx.moveTo(gx, gy);
      else ctx.lineTo(gx, gy);
    }
    ctx.stroke();

    // Jump markers & live state dot
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      if (phaseHistJump[idx] > 0) {
        const q = useY ? -(phaseHistY[idx] - meanY) : (phaseHistX[idx] - meanX);
        const v = useY ? phaseHistVy[idx] : phaseHistVx[idx];
        const gx = pcx + (q / maxQ) * (pw * 0.45);
        const gy = pcy - (v / maxV) * (ph * 0.45);
        ctx.fillStyle = '#ff4d6d';
        ctx.fillRect(gx - 2.5, gy - 2.5, 5, 5);
      }
    }
    const qNow = useY ? -(phaseHistY[lastIdx] - meanY) : (phaseHistX[lastIdx] - meanX);
    const vNow = useY ? phaseHistVy[lastIdx] : phaseHistVx[lastIdx];
    const gxNow = pcx + (qNow / maxQ) * (pw * 0.45);
    const gyNow = pcy - (vNow / maxV) * (ph * 0.45);
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(gxNow, gyNow, 2.8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#9db4d4';
    ctx.fillText(`PHASE (${useY ? 'y,vy' : 'x,vx'}) T${curTid}`, lx + 6, ly + 13);
    ctx.fillStyle = col;
    ctx.fillText(PHYS_REGIME_NAMES[curRegime].slice(0, 12), lx + lw - 80, ly + 13);
  } else {
    ctx.fillStyle = '#9db4d4';
    ctx.fillText('PHASE PORTRAIT (q, qdot)', lx + 6, ly + 13);
    ctx.fillStyle = '#5c7394';
    ctx.fillText(curTid > 0 ? 'WARMING TRACE...' : 'NO PRIMARY TARGET', px0 + 8, pcy + 14);
  }

  // Right pane top: Energy Conservation E(t) & Ensemble K_ens(t)
  ctx.fillStyle = '#9db4d4';
  ctx.fillText('E(t) CYAN · K_ens AMBER', rx + 6, ry + 13);
  ctx.strokeStyle = '#182438';
  ctx.strokeRect(ex0 + 0.5, ey0 + 0.5, ew - 1, eh - 1);

  let maxE = 1.0;
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    if (phaseHistE[idx] > maxE) maxE = phaseHistE[idx];
    if (phaseHistEnsK[idx] > maxE) maxE = phaseHistEnsK[idx];
  }
  maxE *= 1.12;

  const denom = Math.max(1, phaseHistCount - 1);
  // Jump vertical markers
  ctx.fillStyle = 'rgba(255, 77, 109, 0.55)';
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    if (phaseHistJump[idx] > 0) {
      const gx = ex0 + 2 + (i / denom) * (ew - 4);
      ctx.fillRect(gx, ey0 + 1, 1.5, eh - 2);
    }
  }

  // K_ens curve
  ctx.strokeStyle = '#ffb703';
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    const gx = ex0 + 2 + (i / denom) * (ew - 4);
    const gy = ey0 + eh - 3 - Math.min(1.0, phaseHistEnsK[idx] / maxE) * (eh - 6);
    if (i === 0) ctx.moveTo(gx, gy);
    else ctx.lineTo(gx, gy);
  }
  ctx.stroke();

  // Specific Energy E(t) curve
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    const gx = ex0 + 2 + (i / denom) * (ew - 4);
    const gy = ey0 + eh - 3 - Math.min(1.0, phaseHistE[idx] / maxE) * (eh - 6);
    if (i === 0) ctx.moveTo(gx, gy);
    else ctx.lineTo(gx, gy);
  }
  ctx.stroke();

  // Right pane bottom: Kuramoto R_sync [0..1]
  ctx.fillStyle = '#9db4d4';
  ctx.fillText(`KURAMOTO R=${curR.toFixed(2)} (N=${curN})`, rx + 6, ky0 - 5);
  ctx.strokeStyle = '#182438';
  ctx.strokeRect(kx0 + 0.5, ky0 + 0.5, kw - 1, kh - 1);
  ctx.strokeStyle = 'rgba(100, 128, 168, 0.25)';
  ctx.beginPath();
  ctx.moveTo(kx0, ky0 + kh * 0.5);
  ctx.lineTo(kx0 + kw, ky0 + kh * 0.5);
  ctx.stroke();

  ctx.strokeStyle = '#38f27a';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    const gx = kx0 + 2 + (i / denom) * (kw - 4);
    const rClamp = Math.max(0.0, Math.min(1.0, phaseHistKurR[idx]));
    const gy = ky0 + kh - 2 - rClamp * (kh - 4);
    if (i === 0) ctx.moveTo(gx, gy);
    else ctx.lineTo(gx, gy);
  }
  ctx.stroke();
}

function buildPhysicsIdSnapshot(pIdx, hasPrimary, fxPx, fyPx, speedPhys, speedUnit, pMetricValid, angVelDegS) {
  const pb = hasPrimary ? (TELEM_PHYSICS_START + pIdx * TELEM_PHYSICS_STRIDE) : -1;
  const eb = TELEM_ENSEMBLE_START;

  const immWcv = hasPrimary ? safeFinite(telemView[pb + 0]) : 0.0;
  const immWbal = hasPrimary ? safeFinite(telemView[pb + 1]) : 0.0;
  const immWturnOsc = hasPrimary ? safeFinite(telemView[pb + 2]) : 0.0;
  const fistaSharpKnots = hasPrimary ? Math.max(0, Math.min(10, Math.round(safeFinite(telemView[pb + 3])))) : 0;
  const physRegime = hasPrimary ? Math.max(0, Math.min(8, Math.round(safeFinite(telemView[pb + 4])))) : 0;
  const physRegimeConf = hasPrimary ? safeFinite(telemView[pb + 5]) : 0.0;
  const odeR2 = hasPrimary ? safeFinite(telemView[pb + 6]) : 0.0;
  const aeroDragBeta = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 7])) : 0.0;
  const thrustAccel = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 8])) : 0.0;
  const ellipticPeriodS = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 9])) : 0.0;
  const dampingZeta = hasPrimary ? safeFinite(telemView[pb + 10]) : 0.0;
  const qualityFactorQ = hasPrimary ? safeFinite(telemView[pb + 11]) : 0.0;
  const specificEnergy = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 12])) : 0.0;
  const specificAngMom = hasPrimary ? safeFinite(telemView[pb + 13]) : 0.0;
  const normalRestitutionE = hasPrimary ? safeFinite(telemView[pb + 14]) : 0.0;
  const constraintJumpCode = hasPrimary ? Math.max(0, Math.min(4, Math.round(safeFinite(telemView[pb + 15])))) : 0;

  const kuramotoOrderR = safeFinite(telemView[eb + 0]);
  const kuramotoMeanPsi = safeFinite(telemView[eb + 1]);
  const kuramotoOscCount = Math.max(0, Math.round(safeFinite(telemView[eb + 2])));
  const ensembleKinetic = safeFinite(telemView[eb + 3]);
  const ensembleMechEnergy = safeFinite(telemView[eb + 4]);
  const lastPairMassRatio = safeFinite(telemView[eb + 5]);
  const lastPairMomResid = safeFinite(telemView[eb + 6]);
  const lastPairEnergyRetention = safeFinite(telemView[eb + 7]);
  const lastPairTidA = Math.round(safeFinite(telemView[eb + 8]));
  const lastPairTidB = Math.round(safeFinite(telemView[eb + 9]));
  const lastPairFrame = Math.round(safeFinite(telemView[eb + 10]));
  const lyapunovDivRate = Math.max(0.0, safeFinite(telemView[eb + 11]));
  const taubinVsKasaDeltaPx = Math.max(0.0, safeFinite(telemView[eb + 12]));
  const taubinCurvatureInvPx = Math.max(0.0, safeFinite(telemView[eb + 13]));

  const camSusp = isCameraMotionSuspendingCal();
  let unitMode = 'pixel';
  let pxPerMeter = 0.0;
  let localPxPerMeter = 0.0;
  if (camSusp) {
    unitMode = 'out_of_plane';
  } else if (calMode === 'homography' && calHomographyValid) {
    const hX = hasPrimary ? fxPx : frameW * 0.5;
    const hY = hasPrimary ? fyPx : frameH * 0.5;
    const lPpm = computeLocalHomographyPpm(hX, hY);
    if (lPpm > 0.5 && pMetricValid >= 0) {
      unitMode = 'metric';
      localPxPerMeter = lPpm;
    } else if (lPpm < 0 || pMetricValid < -0.5) {
      unitMode = 'out_of_plane';
    }
  } else {
    const ppm = getEffectivePxPerMeter();
    pxPerMeter = ppm;
    if (ppm > 0.5) {
      unitMode = 'metric';
      localPxPerMeter = ppm;
    }
  }

  const metricOk = unitMode === 'metric' && localPxPerMeter > 0.5;
  const ppm = metricOk ? localPxPerMeter : 1.0;
  const ppm2 = ppm * ppm;

  const betaDisp = metricOk ? aeroDragBeta * ppm : aeroDragBeta;
  const betaUnit = metricOk ? 'm^-1' : 'px^-1';
  const thrustDisp = metricOk ? thrustAccel / ppm : thrustAccel;
  const thrustUnit = metricOk ? 'm/s^2' : 'px/s^2';
  const energyDisp = metricOk ? specificEnergy / ppm2 : specificEnergy;
  const energyUnit = metricOk ? 'J/kg' : 'px^2/s^2';
  const angMomDisp = metricOk ? specificAngMom / ppm2 : specificAngMom;
  const angMomUnit = metricOk ? 'm^2/s' : 'px^2/s';
  const ensKineticDisp = metricOk ? ensembleKinetic / ppm2 : ensembleKinetic;
  const ensMechEnergyDisp = metricOk ? ensembleMechEnergy / ppm2 : ensembleMechEnergy;
  const taubinDeltaDisp = metricOk ? taubinVsKasaDeltaPx / ppm : taubinVsKasaDeltaPx;
  const taubinDeltaUnit = metricOk ? 'm' : 'px';
  const curvatureDisp = metricOk ? taubinCurvatureInvPx * ppm : taubinCurvatureInvPx;
  const curvatureUnit = metricOk ? 'm^-1' : 'px^-1';

  const sceneGravPx = safeFinite(telemView[TELEM_GLOBAL + 0]);
  const gStr = metricOk
    ? `${(calMode === 'auto_g' ? GRAVITY_M_S2 : (sceneGravPx > 50.0 ? sceneGravPx / ppm : GRAVITY_M_S2)).toFixed(2)} m/s^2`
    : (sceneGravPx > 10.0 ? `${sceneGravPx.toFixed(0)} px/s^2` : 'g');
  const speedStr = speedUnit === 'OUT OF PLANE'
    ? 'OOP'
    : (metricOk ? `${speedPhys.toFixed(2)} m/s` : `${speedPhys.toFixed(1)} px/s`);

  const symbolicOde = formatSymbolicOdeString({
    hasPrimary,
    physRegime,
    odeR2,
    betaDisp,
    betaUnit,
    thrustDisp,
    thrustUnit,
    curvatureDisp,
    curvatureUnit,
    dampingZeta,
    qualityFactorQ,
    ellipticPeriodS,
    normalRestitutionE,
    lastPairMassRatio,
    lastPairFrame,
    kuramotoOrderR,
    kuramotoOscCount,
    tremorFreqHz: safeFinite(telemView[13]),
    angVelDegS,
    speedStr,
    gStr,
  });

  return {
    hasPrimary,
    physRegime,
    physRegimeName: PHYS_REGIME_NAMES[physRegime] || 'INERTIAL CV',
    physRegimeConf,
    odeR2,
    symbolicOde,
    immWcv,
    immWbal,
    immWturnOsc,
    fistaSharpKnots,
    aeroDragBeta,
    thrustAccel,
    ellipticPeriodS,
    dampingZeta,
    qualityFactorQ,
    specificEnergy,
    specificAngMom,
    normalRestitutionE,
    constraintJumpCode,
    constraintJumpName: JUMP_CODE_NAMES[constraintJumpCode] || 'NONE',
    kuramotoOrderR,
    kuramotoMeanPsi,
    kuramotoOscCount,
    ensembleKinetic,
    ensembleMechEnergy,
    lastPairMassRatio,
    lastPairMomResid,
    lastPairEnergyRetention,
    lastPairTidA,
    lastPairTidB,
    lastPairFrame,
    lyapunovDivRate,
    taubinVsKasaDeltaPx,
    taubinCurvatureInvPx,
    pxPerMeter,
    localPxPerMeter,
    unitMode,
    display: {
      beta: betaDisp,
      betaUnit,
      thrust: thrustDisp,
      thrustUnit,
      energy: energyDisp,
      energyUnit,
      angMom: angMomDisp,
      angMomUnit,
      ensKinetic: ensKineticDisp,
      ensMechEnergy: ensMechEnergyDisp,
      taubinDelta: taubinDeltaDisp,
      taubinDeltaUnit,
      curvature: curvatureDisp,
      curvatureUnit,
    },
  };
}

// ============================================================================
// Core Per-Frame Execution Pipeline (Pre-Tick -> Lanczos Fiedler -> Post-Tick)
// ============================================================================

function computeMedianTickMs(count) {
  if (count <= 0) return 0.0;
  for (let i = 0; i < count; i++) {
    wasmTimeSortBuf[i] = wasmTimeHist[i];
  }
  for (let i = 1; i < count; i++) {
    const v = wasmTimeSortBuf[i];
    let j = i - 1;
    while (j >= 0 && wasmTimeSortBuf[j] > v) {
      wasmTimeSortBuf[j + 1] = wasmTimeSortBuf[j];
      j--;
    }
    wasmTimeSortBuf[j + 1] = v;
  }
  const mid = count >> 1;
  return (count & 1) ? wasmTimeSortBuf[mid] : 0.5 * (wasmTimeSortBuf[mid - 1] + wasmTimeSortBuf[mid]);
}

function drawCroppedLetterboxedFrame(bmp) {
  const bw = Math.max(1, bmp.width || frameW);
  const bh = Math.max(1, bmp.height || frameH);
  const cx = Math.max(0.0, Math.min(0.98, cropNorm[0]));
  const cy = Math.max(0.0, Math.min(0.98, cropNorm[1]));
  const cw = Math.max(0.02, Math.min(1.0 - cx, cropNorm[2]));
  const ch = Math.max(0.02, Math.min(1.0 - cy, cropNorm[3]));
  const sx = Math.max(0, Math.min(bw - 1, Math.round(cx * bw)));
  const sy = Math.max(0, Math.min(bh - 1, Math.round(cy * bh)));
  const sw = Math.max(1, Math.min(bw - sx, Math.round(cw * bw)));
  const sh = Math.max(1, Math.min(bh - sy, Math.round(ch * bh)));

  const srcAspect = sw / sh;
  const dstAspect = frameW / frameH;
  const fullFrame = sx === 0 && sy === 0 && sw === bw && sh === bh;
  if (fullFrame && Math.abs(srcAspect - dstAspect) / dstAspect <= 0.02) {
    simCtx.drawImage(bmp, 0, 0, frameW, frameH);
    return;
  }

  simCtx.fillStyle = '#000000';
  simCtx.fillRect(0, 0, frameW, frameH);
  let dw = frameW;
  let dh = frameH;
  let dx = 0;
  let dy = 0;
  if (srcAspect > dstAspect) {
    dh = Math.max(1, Math.round(frameW / srcAspect));
    dy = (frameH - dh) >> 1;
  } else {
    dw = Math.max(1, Math.round(frameH * srcAspect));
    dx = (frameW - dw) >> 1;
  }
  simCtx.drawImage(bmp, sx, sy, sw, sh, dx, dy, dw, dh);
}

function finiteOrNull(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function intOrNull(v, min = 0, max = 16777215) {
  if (typeof v !== 'number' || !Number.isFinite(v) || !Number.isInteger(v)) return null;
  return v >= min && v <= max ? v : null;
}

function parseRigidBodyDescriptor(buf, bo) {
  if (!buf) return null;
  const bodyId = intOrNull(buf[bo], 1, MAX_RIGID_BODIES);
  const memberCount = intOrNull(buf[bo + 1], 1, MAX_MICRO);
  const cx = finiteOrNull(buf[bo + 2]);
  const cy = finiteOrNull(buf[bo + 3]);
  const vx = finiteOrNull(buf[bo + 4]);
  const vy = finiteOrNull(buf[bo + 5]);
  const omegaRadS = finiteOrNull(buf[bo + 6]);
  if (
    bodyId === null ||
    memberCount === null ||
    cx === null ||
    cy === null ||
    vx === null ||
    vy === null ||
    omegaRadS === null
  ) {
    return null;
  }
  return {
    bodyId,
    memberCount,
    cx,
    cy,
    vx,
    vy,
    omegaRadS,
    se2RmsePx: finiteOrNull(buf[bo + 7]),
  };
}

function computeLookaheadTiming(effK, dispSlot, fIdxNow) {
  const k = intOrNull(effK, 0, RING_LEN - 1) ?? (Number.isFinite(effK) ? Math.max(0, Math.round(effK)) : 0);
  if (k === 0) {
    return {
      lookaheadWindowMs: 0.0,
      lookaheadSource: 'zero_lookahead',
      displayedFrameIdx: fIdxNow,
      causalFrameIdx: fIdxNow,
      renderedDelayFrames: 0,
    };
  }
  const targetDispFrame = fIdxNow - k;
  const ringMatch = dispSlot >= 0 && dispSlot < RING_LEN && ringIngestFrameId[dispSlot] === targetDispFrame;
  if (
    ringMatch &&
    latestCausalPtsMs >= 0.0 &&
    ringIngestPtsMs[dispSlot] >= 0.0 &&
    latestCausalPtsMs >= ringIngestPtsMs[dispSlot]
  ) {
    return {
      lookaheadWindowMs: Math.max(0.0, latestCausalPtsMs - ringIngestPtsMs[dispSlot]),
      lookaheadSource: 'source_pts',
      displayedFrameIdx: targetDispFrame,
      causalFrameIdx: fIdxNow,
      renderedDelayFrames: k,
    };
  }
  if (
    ringMatch &&
    ringIngestTimeMs[dispSlot] >= 0.0 &&
    latestCausalTimeMs >= ringIngestTimeMs[dispSlot]
  ) {
    return {
      lookaheadWindowMs: Math.max(0.0, latestCausalTimeMs - ringIngestTimeMs[dispSlot]),
      lookaheadSource: 'accumulated_dt',
      displayedFrameIdx: targetDispFrame,
      causalFrameIdx: fIdxNow,
      renderedDelayFrames: k,
    };
  }
  const dtSec = lastDtSecUsed > 0.001 && lastDtSecUsed < 0.25 ? lastDtSecUsed : (1.0 / 30.0);
  return {
    lookaheadWindowMs: k * dtSec * 1000.0,
    lookaheadSource: 'dt_estimate',
    displayedFrameIdx: Math.max(0, targetDispFrame),
    causalFrameIdx: fIdxNow,
    renderedDelayFrames: k,
  };
}

function buildChronos1000Snapshot(medianMs, effK, fIdxNow = 0, dispSlot = -1) {
  const dtSec = lastDtSecUsed > 0.001 && lastDtSecUsed < 0.25 ? lastDtSecUsed : (1.0 / 30.0);
  const timing = computeLookaheadTiming(effK, dispSlot, fIdxNow);
  const lookaheadWindowMs = timing.lookaheadWindowMs;
  const partialPipelineLatencyMs = lookaheadWindowMs + medianMs + lastSwarmCopyMs + lastFieldRenderMs;
  const st = swarmTelemView;
  const avail = Boolean(st && hasChronos1000Exports());
  const stInspTid = avail ? intOrNull(st[11], 0, 16777215) : null;
  const inspectExplicitClear =
    lastInspectStatus === 'expired' ||
    lastInspectStatus === 'miss' ||
    lastInspectStatus === 'unavailable';
  const activeInspTid = inspectExplicitClear
    ? 0
    : (inspectedMicroTid > 0 ? inspectedMicroTid : (stInspTid !== null && stInspTid > 0 ? stInspTid : 0));
  let inspectedTrack = null;
  if (avail && activeInspTid > 0 && microTracksView) {
    for (let i = 0; i < MAX_MICRO; i++) {
      const b = i * MICRO_STRIDE;
      const trackSt = intOrNull(microTracksView[b + 5], 1, 2);
      const trackTid = intOrNull(microTracksView[b], 1, 16777215);
      if (trackSt !== null && trackTid === activeInspTid) {
        const rawTrackFtle = finiteOrNull(microTracksView[b + 7]);
        inspectedTrack = {
          tid: activeInspTid,
          x: finiteOrNull(microTracksView[b + 1]),
          y: finiteOrNull(microTracksView[b + 2]),
          vx: finiteOrNull(microTracksView[b + 3]),
          vy: finiteOrNull(microTracksView[b + 4]),
          state: trackSt,
          bodyId: intOrNull(microTracksView[b + 6], 0, MAX_RIGID_BODIES),
          ftle: rawTrackFtle !== null && rawTrackFtle > -900.0 ? rawTrackFtle : null,
        };
        break;
      }
    }
  }

  const swarmActiveFlag = avail ? intOrNull(st[1], 0, 1) : null;
  const validityMaskVal = avail ? intOrNull(st[16], 0, 31) : null;
  const rigidBodyCountVal = avail ? intOrNull(st[23], 0, MAX_RIGID_BODIES) : 0;
  const topRigidMembersVal = avail ? intOrNull(st[25], 0, MAX_MICRO) : 0;
  const topRigidOmegaVal = avail ? finiteOrNull(st[24]) : 0.0;
  let validRigidBodyCount = 0;
  if (avail && rigidBodiesView && validityMaskVal !== null && (validityMaskVal & 4) !== 0) {
    for (let b = 0; b < MAX_RIGID_BODIES; b++) {
      if (parseRigidBodyDescriptor(rigidBodiesView, b * BODY_STRIDE)) {
        validRigidBodyCount++;
      }
    }
  }
  const topRigidBodyValid = Boolean(
    avail &&
    validityMaskVal !== null &&
    (validityMaskVal & 4) !== 0 &&
    validRigidBodyCount > 0 &&
    rigidBodyCountVal !== null &&
    rigidBodyCountVal > 0 &&
    topRigidMembersVal !== null &&
    topRigidMembersVal >= 1 &&
    topRigidOmegaVal !== null
  );

  const out = {
    available: avail,
    diagnosticsValid: false,
    delayFrames: 0,
    causalFrameIdx: timing.causalFrameIdx,
    displayedFrameIdx: timing.displayedFrameIdx,
    renderedDelayFrames: timing.renderedDelayFrames,
    inspectStatus: lastInspectStatus,
    inspectSeq: lastInspectResSeq,
    inspectReqSeq: lastInspectReqSeq,
    inspectResSeq: lastInspectResSeq,
    inspectReqPxX: lastInspectReqX,
    inspectReqPxY: lastInspectReqY,
    inspectReqX: lastInspectReqX,
    inspectReqY: lastInspectReqY,
    inspectReqNormX: lastInspectReqNormX,
    inspectReqNormY: lastInspectReqNormY,
    swarmMode: avail ? intOrNull(st[0], 0, 2) : swarmMode,
    userSwarmMode,
    denseSwarmOverrideActive,
    swarmActive: avail ? swarmActiveFlag === 1 : false,
    validRigidBodyCount,
    topRigidBodyValid,
    fieldOverlayMode,
    fieldBackend: lastFieldBackend,
    fieldRenderMs: lastFieldRenderMs,
    snapshotCopyMs: lastSwarmCopyMs,
    peakCount: avail ? intOrNull(st[2], 0, 2048) : 0,
    activeCount: avail ? intOrNull(st[3], 0, MAX_MICRO) : 0,
    confirmedCount: avail ? intOrNull(st[4], 0, MAX_MICRO) : 0,
    peakSimultaneous: avail ? intOrNull(st[5], 0, MAX_MICRO) : 0,
    uniqueTracks: avail ? intOrNull(st[6], 0, 16777215) : 0,
    coastingCount: avail ? intOrNull(st[7], 0, MAX_MICRO) : 0,
    auctionSteps: avail ? intOrNull(st[8], 0, 16777215) : 0,
    reacqEvents: avail ? intOrNull(st[9], 0, 16777215) : 0,
    meanIcLkRes: avail ? finiteOrNull(st[10]) : 0.0,
    inspectedTid: activeInspTid,
    inspectedBodyId: avail ? intOrNull(st[12], 0, MAX_RIGID_BODIES) : 0,
    births: avail ? intOrNull(st[13], 0, MAX_MICRO) : 0,
    tentativeCount: avail ? intOrNull(st[14], 0, MAX_MICRO) : 0,
    ambiguousCount: avail ? intOrNull(st[15], 0, MAX_MICRO) : 0,
    validityMask: avail ? intOrNull(st[16], 0, 31) : 0,
    validCells: avail ? intOrNull(st[17], 0, FIELD_CELLS) : 0,
    maxVorticityS1: avail ? finiteOrNull(st[18]) : 0.0,
    rmsVorticityS1: avail ? finiteOrNull(st[19]) : 0.0,
    rmsDivergenceS1: avail ? finiteOrNull(st[20]) : 0.0,
    maxFtleS1: avail ? finiteOrNull(st[21]) : 0.0,
    ftleCells: avail ? intOrNull(st[22], 0, FIELD_CELLS) : 0,
    rigidBodyCount: avail ? intOrNull(st[23], 0, MAX_RIGID_BODIES) : 0,
    topRigidOmegaRadS: avail ? finiteOrNull(st[24]) : 0.0,
    topRigidMembers: avail ? intOrNull(st[25], 0, MAX_MICRO) : 0,
    topRigidSe2RmsePx: avail ? finiteOrNull(st[26]) : 0.0,
    modalDomFreqHz: avail ? finiteOrNull(st[27]) : 0.0,
    modalCoherence: avail ? finiteOrNull(st[28]) : 0.0,
    modalValidSensors: avail ? intOrNull(st[29], 0, MAX_MICRO) : 0,
    kuramotoR: avail ? finiteOrNull(st[30]) : 0.0,
    kuramotoPsi: avail ? finiteOrNull(st[31]) : 0.0,
    comX: avail ? finiteOrNull(st[32]) : 0.0,
    comY: avail ? finiteOrNull(st[33]) : 0.0,
    comVx: avail ? finiteOrNull(st[34]) : 0.0,
    comVy: avail ? finiteOrNull(st[35]) : 0.0,
    velDispersion: avail ? finiteOrNull(st[36]) : 0.0,
    meanStrain: avail ? finiteOrNull(st[37]) : 0.0,
    knnEdgeCount: avail ? intOrNull(st[38], 0, MAX_MICRO * 6) : 0,
    ftleMatureTracks: avail ? intOrNull(st[39], 0, MAX_MICRO) : 0,
    topRigidThetaRad: avail ? finiteOrNull(st[40]) : 0.0,
    modalR2: avail ? finiteOrNull(st[41]) : 0.0,
    kernelComputeMedianMs: medianMs,
    lookaheadWindowMs,
    lookaheadSource: timing.lookaheadSource,
    estimatedPartialPipelineLatencyMs: partialPipelineLatencyMs,
    partialPipelineLatencyMs,
    dtSec,
    inspectedTrack,
  };

  out.diagnosticsValid = Boolean(
    avail &&
    out.swarmMode !== null &&
    swarmActiveFlag !== null &&
    out.peakCount !== null &&
    out.activeCount !== null &&
    out.confirmedCount !== null &&
    out.peakSimultaneous !== null &&
    out.uniqueTracks !== null &&
    out.coastingCount !== null &&
    out.auctionSteps !== null &&
    out.reacqEvents !== null &&
    out.meanIcLkRes !== null &&
    stInspTid !== null &&
    out.inspectedBodyId !== null &&
    out.births !== null &&
    out.tentativeCount !== null &&
    out.ambiguousCount !== null &&
    out.validityMask !== null &&
    out.validCells !== null &&
    out.maxVorticityS1 !== null &&
    out.rmsVorticityS1 !== null &&
    out.rmsDivergenceS1 !== null &&
    out.maxFtleS1 !== null &&
    out.ftleCells !== null &&
    out.rigidBodyCount !== null &&
    out.topRigidOmegaRadS !== null &&
    out.topRigidMembers !== null &&
    out.topRigidSe2RmsePx !== null &&
    out.modalDomFreqHz !== null &&
    out.modalCoherence !== null &&
    out.modalValidSensors !== null &&
    out.kuramotoR !== null &&
    out.kuramotoPsi !== null &&
    out.comX !== null &&
    out.comY !== null &&
    out.comVx !== null &&
    out.comVy !== null &&
    out.velDispersion !== null &&
    out.meanStrain !== null &&
    out.knnEdgeCount !== null &&
    out.ftleMatureTracks !== null &&
    out.topRigidThetaRad !== null &&
    out.modalR2 !== null
  );
  return out;
}

function emitTelemetrySnapshot() {
  if (!wasmExports || !telemView) return;
  let sumMs = 0.0;
  let maxMs = 0.0;
  const count = Math.min(Math.max(1, tickCounter), TIME_HIST_LEN);
  for (let i = 0; i < count; i++) {
    sumMs += wasmTimeHist[i];
    if (wasmTimeHist[i] > maxMs) maxMs = wasmTimeHist[i];
  }
  const meanMs = sumMs / Math.max(1, count);
  const medianMs = computeMedianTickMs(count);
  const effSampleRate = sourceMode === 'testbench' ? 60.0 : Math.max(8.0, sourceFps);
  const fIdxNow = Math.round(telemView[0]);

  const pIdx = resolvePrimarySlot();
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  const metricOn = isMetricCalibrated();
  let pTid = 0;
  let pSigHex = '0x00000000';
  let pVal = 0;
  let apexMs = 0;
  let primaryCoherence = 0;
  let headingDeg = 0.0;
  let speedPhys = 0.0;
  let pMetricValid = metricOn ? 1.0 : 0.0;
  let angVelDegS = 0.0;
  let pivotValid = false;
  let pendSwingDeg = 0.0;
  let pendRadiusPhys = 0.0;
  let pFxPx = frameW * 0.5;
  let pFyPx = frameH * 0.5;
  const hasPrimary = pIdx >= 0 && telemView[32 + pIdx * 56] > 0.5;
  if (hasPrimary) {
    const b = 32 + pIdx * 56;
    const kb = TELEM_KINETIC + pIdx * 8;
    const fxPx = telemView[b + 2];
    const fyPx = telemView[b + 3];
    pFxPx = fxPx;
    pFyPx = fyPx;
    pTid = Math.round(telemView[b + 1]);
    pSigHex = toHex8(telemView[b + 12], telemView[b + 13]);
    pVal = Math.round(telemView[b + 14]);
    apexMs = telemView[b + 17] * (1000.0 / effSampleRate);
    primaryCoherence = telemView[TELEM_EXCUR + pIdx * 8 + 6];
    const physVel = mapVelocityToPhysical(fxPx, fyPx, telemView[kb + 1], telemView[kb + 0]);
    speedPhys = physVel[0];
    headingDeg = physVel[1];
    pMetricValid = physVel[2];
    angVelDegS = (calMode === 'homography' && calHomographyValid && physVel[2] > 0.5)
      ? slotRectAngVel[pIdx]
      : telemView[kb + 2];
    if (telemView[kb + 6] > -9.0e5 && telemView[kb + 7] > -9.0e5) {
      pivotValid = true;
      const pend = computePendulumKinematics(fxPx, fyPx, telemView[kb + 6], telemView[kb + 7]);
      pendSwingDeg = pend[0];
      pendRadiusPhys = pend[1];
    }
  }
  if (apexMs <= 0.0) {
    for (let m = 0; m < MAX_TARGETS; m++) {
      const mb = 32 + m * 56;
      if (telemView[mb] > 0.5 && telemView[mb + 17] > 0.35) {
        const candApexMs = telemView[mb + 17] * (1000.0 / effSampleRate);
        if (candApexMs > apexMs) apexMs = candApexMs;
      }
    }
  }

  const speedUnit = pMetricValid < -0.5
    ? 'OUT OF PLANE'
    : (pMetricValid > 0.5 ? 'm/s' : 'px/s');
  const distUnit = pMetricValid < -0.5
    ? 'OUT OF PLANE'
    : (metricOn ? 'm' : 'px');
  const lastStrikeAgeFrames = lastStrikeFrame >= 0 ? Math.max(0, fIdxNow - lastStrikeFrame) : -1;
  const scatterFrameWasm = Math.round(telemView[TELEM_GLOBAL + 7]);
  if (scatterFrameWasm > 0) lastScatterFrame = scatterFrameWasm;
  const scatterAgeFrames = lastScatterFrame > 0 ? Math.max(0, fIdxNow - lastScatterFrame) : -1;

  const slotRoster = new Array(MAX_TARGETS);
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    const active = telemView[b] > 0.5;
    if (active) {
      const tid = Math.round(telemView[b + 1]);
      const dispVis = telemView[b + 10] > 0.5;
      const isBridged = telemView[b + 11] > 0.5;
      const isPinned = focusTid !== 0 && tid === focusTid;
      const isPrimary = m === pIdx;
      const mv = slotLastMetricValid[m];
      const isPiv = slotLastPivotValid[m] > 0;
      const state = isBridged
        ? 'BRIDGED'
        : !dispVis
        ? 'COAST'
        : isPinned
        ? 'PINNED'
        : isPrimary
        ? 'FOCUS'
        : 'LOCKED';
      slotRoster[m] = {
        slot: m,
        state,
        tid,
        color: TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length],
        sigHex: toHex8(slotLastSigHi[m], slotLastSigLo[m]),
        padicVal: slotLastVal[m],
        reacq: slotLastReacq[m],
        coastFrames: slotLastCoast[m],
        ageSec: slotLastAgeFrames[m] / effSampleRate,
        speed: slotSmoothedSpeed[m],
        speedUnit: mv < -0.5 ? 'OOP' : (mv > 0.5 ? 'm/s' : 'px/s'),
        isPivot: isPiv,
        angleDeg: isPiv ? slotLastPendSwing[m] : slotSmoothedHeading[m],
        rateOrRad: isPiv ? slotLastPendRad[m] : slotSmoothedOmega[m],
        rateUnit: isPiv ? (metricOn ? 'm' : 'px') : '°/s',
        freqHz: slotLastFreqHz[m],
        ecaPct: slotSmoothedEca[m],
        isPinned,
        isPrimary,
        physRegime: Math.max(0, Math.min(8, Math.round(safeFinite(telemView[TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE + 4])))),
        odeR2: safeFinite(telemView[TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE + 6]),
        jumpCode: Math.max(0, Math.min(4, Math.round(safeFinite(telemView[TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE + 15])))),
      };
    } else if (slotHoldLeft[m] > 0 && slotHoldTid[m] > 0) {
      const tid = slotHoldTid[m];
      const mv = slotLastMetricValid[m];
      const isPiv = slotLastPivotValid[m] > 0;
      slotRoster[m] = {
        slot: m,
        state: 'HOLD',
        tid,
        color: TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length],
        sigHex: toHex8(slotLastSigHi[m], slotLastSigLo[m]),
        padicVal: slotLastVal[m],
        reacq: slotLastReacq[m],
        coastFrames: slotLastCoast[m],
        ageSec: slotLastAgeFrames[m] / effSampleRate,
        speed: slotSmoothedSpeed[m],
        speedUnit: mv < -0.5 ? 'OOP' : (mv > 0.5 ? 'm/s' : 'px/s'),
        isPivot: isPiv,
        angleDeg: isPiv ? slotLastPendSwing[m] : slotSmoothedHeading[m],
        rateOrRad: isPiv ? slotLastPendRad[m] : slotSmoothedOmega[m],
        rateUnit: isPiv ? (metricOn ? 'm' : 'px') : '°/s',
        freqHz: slotLastFreqHz[m],
        ecaPct: slotSmoothedEca[m],
        isPinned: focusTid !== 0 && tid === focusTid,
        isPrimary: false,
        physRegime: 0,
        odeR2: 0.0,
        jumpCode: 0,
      };
    } else {
      slotRoster[m] = {
        slot: m,
        state: 'IDLE',
        tid: 0,
        color: '#64748b',
        sigHex: '0x00000000',
        padicVal: 0,
        reacq: 0,
        coastFrames: 0,
        ageSec: 0.0,
        speed: 0.0,
        speedUnit: metricOn ? 'm/s' : 'px/s',
        isPivot: false,
        angleDeg: 0.0,
        rateOrRad: 0.0,
        rateUnit: '°/s',
        freqHz: 0.0,
        ecaPct: 0.0,
        isPinned: false,
        isPrimary: false,
        physRegime: 0,
        odeR2: 0.0,
        jumpCode: 0,
      };
    }
  }

  const dispSlotNow = Math.round(telemView[2]);
  const effKNow = Math.round(telemView[1]);
  const c1kSnap = buildChronos1000Snapshot(medianMs, effKNow, fIdxNow, dispSlotNow);
  const oracleValid = sourceMode === 'testbench' && !isChronos1000Scenario(benchScenario);

  self.postMessage({
    type: 'telemetry',
    frameIdx: fIdxNow,
    lookaheadK: effKNow,
    fps: effSampleRate,
    wasmMeanMs: meanMs,
    wasmMedianMs: medianMs,
    wasmMaxMs: maxMs,
    lookaheadWindowMs: c1kSnap.lookaheadWindowMs,
    lookaheadSource: c1kSnap.lookaheadSource,
    partialPipelineLatencyMs: c1kSnap.partialPipelineLatencyMs,
    estimatedPartialPipelineLatencyMs: c1kSnap.estimatedPartialPipelineLatencyMs,
    snapshotCopyMs: lastSwarmCopyMs,
    fieldRenderMs: lastFieldRenderMs,
    activeTargets: Math.round(telemView[3]),
    clutterDb: telemView[4],
    clutterDbValid: telemView[22] > 0.5,
    clutterEvalRatio: telemView[23],
    padicSwaps: Math.round(telemView[5]),
    oracleSwaps: oracleValid ? oraclePadicSwaps : null,
    oracleSwapsAvailable: oracleValid,
    sourceMode,
    benchScenario,
    kalmanSwaps: Math.round(telemView[6]),
    inPillarCollisions,
    bridgedFrames: totalBridgedEvents,
    strikeEvents: totalStrikeEvents,
    lambda2: lastLambda2,
    spectralGap: telemView[24],
    fistaDualityGap: telemView[25],
    phaseValid: telemView[26] > 0.5,
    freqValid: telemView[TELEM_GLOBAL + 12] > 0.5,
    tremorPhaseDeg: telemView[11] * (180.0 / Math.PI),
    tremorFreqHz: telemView[13],
    tremorAmp: telemView[12],
    tremorCoherence: primaryCoherence,
    saccadeLatencyMs: telemView[19],
    stepSettleMs: telemView[28],
    stimulusArmed: telemView[29] > 0.5,
    primaryTrackId: pTid,
    focusTrackId: focusTid,
    primarySigHex: pSigHex,
    primaryPadicVal: pVal,
    apexMs,
    apexErrPx: lastScoredApexErrPx,
    headingDeg,
    speedPhys,
    metricValid: pMetricValid,
    speedUnit,
    distUnit,
    angVelDegS,
    pivotValid,
    pendSwingDeg,
    pendRadiusPhys,
    sceneGravityPxS2: telemView[TELEM_GLOBAL + 0],
    sceneGravityConf: telemView[TELEM_GLOBAL + 1],
    sceneGravityArcs: Math.round(telemView[TELEM_GLOBAL + 11]),
    dispLagMs: telemView[TELEM_GLOBAL + 8],
    camMotionState: Math.round(telemView[TELEM_GLOBAL + 9]),
    camPanSpeedPxS: telemView[TELEM_GLOBAL + 10],
    discontCount: Math.round(telemView[TELEM_GLOBAL + 15]),
    lastDeflectDeg: lastStrikeDeflectDeg,
    lastRestitution: lastStrikeRestitution,
    lastStrikeTid,
    lastStrikeAgeFrames,
    lastStrikeValid,
    scatterDeg: lastScatterDeg,
    scatterAgeFrames,
    calSegmentId,
    calMode,
    calProvenance,
    calArmed,
    calibrated: metricOn,
    showHeroTargets,
    showAllOverlays,
    overlaySyncSeq: lastOverlaySyncSeq,
    swarmMode,
    cropNorm: [cropNorm[0], cropNorm[1], cropNorm[2], cropNorm[3]],
    arenaBytes: Math.round(telemView[20]),
    memoryPages: wasmMemory.buffer.byteLength / 65536,
    resMode,
    frameW,
    frameH,
    pipDim,
    recActive,
    recFrames: recCount,
    recDurationSec: recElapsedSec,
    sessionPersistence: {
      uniqueIds: Math.round(telemView[TELEM_PERSIST + 0]),
      coastReacqEvents: Math.round(telemView[TELEM_PERSIST + 1]),
      dormantReidEvents: Math.round(telemView[TELEM_PERSIST + 2]),
      totalRelocks: Math.round(telemView[TELEM_PERSIST + 1]) + Math.round(telemView[TELEM_PERSIST + 2]),
      peakSimultaneous: Math.round(telemView[TELEM_PERSIST + 3]),
      longestTenureFrames: Math.round(telemView[TELEM_PERSIST + 4]),
      longestTenureSec: Math.round(telemView[TELEM_PERSIST + 4]) / effSampleRate,
      longestTenureTid: Math.round(telemView[TELEM_PERSIST + 5]),
      sustained10s: Math.round(telemView[TELEM_PERSIST + 6]),
      sustained60s: Math.round(telemView[TELEM_PERSIST + 7]),
      dormantActive: Math.round(telemView[TELEM_PERSIST + 8]),
      lastReidGapFrames: Math.round(telemView[TELEM_PERSIST + 9]),
      sessionElapsedSec,
    },
    physicsId: buildPhysicsIdSnapshot(pIdx, hasPrimary, pFxPx, pFyPx, speedPhys, speedUnit, pMetricValid, angVelDegS),
    slotRoster,
    chronos1000: c1kSnap,
  });
}

function processIngestedFrame(rawPixels, dtSec, mediaTimeSec) {
  const rawDt = Math.max(0.001, Number.isFinite(dtSec) && dtSec > 0 ? dtSec : 1.0 / 60.0);
  const fpsDt = Math.min(0.25, Math.max(0.002, rawDt));
  const instSourceFps = 1.0 / fpsDt;
  sourceFps = 0.90 * sourceFps + 0.10 * instSourceFps;

  const ptr = wasmExports.chronos_get_ingest_rgba_ptr();
  let slotIdx = 0;
  for (let s = 0; s < RING_LEN; s++) {
    if (ringSlotU8[s].byteOffset === ptr) {
      slotIdx = s;
      break;
    }
  }
  ringSlotU8[slotIdx].set(rawPixels);

  if (
    typeof wasmExports.chronos_set_ingest_pts === 'function' &&
    Number.isFinite(mediaTimeSec) &&
    mediaTimeSec >= 0.0
  ) {
    wasmExports.chronos_set_ingest_pts(mediaTimeSec);
  }

  // Measure pure kernel compute path: Wasm Pre-Tick + Shifted Lanczos Fiedler + Wasm Post-Tick
  const t0 = performance.now();
  wasmExports.chronos_pre_tick(rawDt);
  const lambda2 = solveShiftedLanczosFiedler();
  const dispSlot = wasmExports.chronos_post_tick(rawDt);
  const wasmMs = performance.now() - t0;

  lastDtSecUsed = rawDt;
  ingestElapsedMs += rawDt * 1000.0;
  latestCausalTimeMs = ingestElapsedMs;
  const causalFrameNow = Math.round(telemView[0]);
  const hasPtsNow = Number.isFinite(mediaTimeSec) && mediaTimeSec >= 0.0;
  if (hasPtsNow) {
    const ptsMsNow = mediaTimeSec * 1000.0;
    if (latestCausalPtsMs >= 0.0 && ptsMsNow < latestCausalPtsMs) {
      ringIngestPtsMs.fill(-1.0);
    }
    latestCausalPtsMs = ptsMsNow;
    if (slotIdx >= 0 && slotIdx < RING_LEN) {
      ringIngestPtsMs[slotIdx] = ptsMsNow;
    }
  } else {
    latestCausalPtsMs = -1.0;
    ringIngestPtsMs.fill(-1.0);
  }
  if (slotIdx >= 0 && slotIdx < RING_LEN) {
    ringIngestFrameId[slotIdx] = causalFrameNow;
    ringIngestTimeMs[slotIdx] = ingestElapsedMs;
  }

  const tCopy0 = performance.now();
  if (hasChronos1000Exports() && slotIdx >= 0 && slotIdx < RING_LEN) {
    if (swarmTelemView && (intOrNull(swarmTelemView[1], 0, 1) === 1 || swarmMode === 2)) {
      swarmRingMicro[slotIdx].set(microTracksView);
      swarmRingField[slotIdx].set(continuumFieldView);
      swarmRingBodies[slotIdx].set(rigidBodiesView);
      swarmRingModal[slotIdx].set(modalFieldView);
      swarmRingTelem[slotIdx].set(swarmTelemView);
      swarmRingValid[slotIdx] = 1;
    } else {
      swarmRingValid[slotIdx] = 0;
    }
  }
  lastSwarmCopyMs = performance.now() - tCopy0;
  lastLambda2 = lambda2;

  wasmTimeHist[timeHistIdx] = wasmMs;
  timeHistIdx = (timeHistIdx + 1) % TIME_HIST_LEN;

  const camMotionState = Math.round(telemView[TELEM_GLOBAL + 9]);
  if (camMotionState === 2) {
    camPanStreakWorker = 0;
    clearVisualQueues();
    if (sourceMode !== 'testbench' && calMode !== 'none') {
      clearCalibrationState();
      recordCalibrationSegmentTransition('scene_cut_reset');
      emitCalibrationState('SCENE CUT · ECA & IMAGE-PLANE CALIBRATION RESET');
    }
  } else if (camMotionState === 1 && sourceMode !== 'testbench') {
    camPanStreakWorker++;
    if (
      camPanStreakWorker >= 4 &&
      (calMode === 'homography' || calMode === 'ruler' || calArmed !== 'none')
    ) {
      clearCalibrationState();
      recordCalibrationSegmentTransition('camera_ptz_pan_reset');
      emitCalibrationState('CAMERA PTZ PAN · IMAGE-PLANE CALIBRATION INVALIDATED (PX UNITS)');
    }
  } else {
    camPanStreakWorker = 0;
  }

  if (sourceMode === 'testbench' && !isChronos1000Scenario(benchScenario)) {
    auditOracleSwaps();
  }

  const fIdx = Math.round(telemView[0]);
  let frameHasBridge = false;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] > 0.5) {
      if (telemView[b + 11] > 0.5) frameHasBridge = true;
      if (calMode === 'homography' && calHomographyValid) {
        const kb = TELEM_KINETIC + m * 8;
        const tidM = Math.round(telemView[b + 1]);
        const pv = mapVelocityToPhysical(telemView[b + 2], telemView[b + 3], telemView[kb + 1], telemView[kb + 0]);
        if (pv[2] > 0.5 && pv[0] > 0.08) {
          if (prevRectTid[m] === tidM && prevRectFrame[m] >= 0 && fIdx > prevRectFrame[m] && (fIdx - prevRectFrame[m]) <= 4) {
            const dth = wrapDeg180(pv[1] - prevRectHeading[m]);
            const spanDt = Math.max(0.002, (fIdx - prevRectFrame[m]) * fpsDt);
            const instW = Math.max(-720.0, Math.min(720.0, dth / spanDt));
            slotRectAngVel[m] = 0.62 * slotRectAngVel[m] + 0.38 * instW;
          } else {
            slotRectAngVel[m] = 0.0;
          }
          prevRectHeading[m] = pv[1];
          prevRectTid[m] = tidM;
          prevRectFrame[m] = fIdx;
        } else {
          slotRectAngVel[m] *= 0.78;
        }
      }
    } else {
      prevRectTid[m] = 0;
      prevRectFrame[m] = -1;
      slotRectAngVel[m] = 0.0;
    }
  }
  if (frameHasBridge) totalBridgedEvents++;
  updateSlotRosterEma();

  // Record scope traces + primary track ID for seamless discontinuity breaks
  const pIdxNow = Math.round(telemView[8]);
  const curPipTid = pIdxNow >= 0 && telemView[32 + pIdxNow * 56] > 0.5
    ? Math.round(telemView[32 + pIdxNow * 56 + 1])
    : -1;
  histTremorR[scopeWriteIdx] = telemView[9];
  histTremorH[scopeWriteIdx] = telemView[10];
  histPhaseValid[scopeWriteIdx] = telemView[26] > 0.5 ? 1 : 0;
  histStrobe[scopeWriteIdx] = telemView[14] > 0.5 ? 1 : 0;
  histPipTid[scopeWriteIdx] = curPipTid;
  histLambda2[scopeWriteIdx] = lambda2;
  scopeWriteIdx = (scopeWriteIdx + 1) % SCOPE_HIST;

  // Present-time phase strobe cue
  if (telemView[14] > 0.5 && strobeEnabled) {
    strobeFlashAlpha = 1.0;
    self.postMessage({
      type: 'audio_strobe',
      resonant: phaseOffsetRad > 0,
      freqHz: telemView[13],
      amp: telemView[12],
    });
  }

  // Multi-arc consensus scene gravity lock (>= 2 consistent free-fall arcs required)
  const gConf = telemView[TELEM_GLOBAL + 1];
  const gPxS2 = telemView[TELEM_GLOBAL + 0];
  if (gConf >= AUTO_G_LOCK_CONF && gPxS2 > 50.0) {
    const hadLock = calAutoGLocked;
    calAutoGNormH = gPxS2 / frameH;
    calAutoGLocked = true;
    if (calMode === 'auto_g' && !hadLock) {
      recordCalibrationSegmentTransition('auto_g_locked');
      emitCalibrationState();
    }
  } else if (calAutoGLocked && gConf < 0.35) {
    calAutoGLocked = false;
    if (calMode === 'auto_g') {
      recordCalibrationSegmentTransition('auto_g_provisional');
      emitCalibrationState();
    }
  } else if (calMode === 'auto_g' && !calAutoGLocked && gPxS2 > 50.0 && Math.round(telemView[TELEM_GLOBAL + 11]) === 1 && calAutoGNormH === 0.0) {
    calAutoGNormH = gPxS2 / frameH;
    emitCalibrationState();
  }

  // Emit per-target deduplicated Mode C strikes and schedule synchronized t - K shockwave rings!
  const percMask = Math.round(telemView[15]);
  if (percMask !== 0) {
    const effSampleRate = Math.max(8.0, sourceMode === 'testbench' ? smoothFps : sourceFps);
    const nowEpochMs = performance.timeOrigin + performance.now();
    for (let m = 0; m < MAX_TARGETS; m++) {
      const ebase = TELEM_EXCUR + m * 8;
      if (telemView[ebase] > 0.5) {
        totalStrikeEvents++;
        const kbase = TELEM_KINETIC + m * 8;
        const zone = Math.round(telemView[ebase + 1]) & 3;
        const leadFrames = telemView[ebase + 2];
        const vel = telemView[ebase + 3];
        const sxPos = telemView[ebase + 4];
        const syPos = telemView[ebase + 5];
        const rawDefl = telemView[kbase + 3];
        const rawRest = telemView[kbase + 4];
        const rawIn = telemView[kbase + 5];
        slotLastStrikeX[m] = sxPos;
        slotLastStrikeY[m] = syPos;

        const maxValidRest = sourceMode === 'testbench' ? 1.65 : 1.18;
        if (rawRest >= 0.06 && rawRest <= maxValidRest && Math.abs(rawDefl) >= 3.0) {
          const rectSt = rectifyStrikeKinematics(sxPos, syPos, rawIn, rawDefl, rawRest);
          if (rectSt[2] <= maxValidRest) {
            lastStrikeDeflectDeg = rectSt[1];
            lastStrikeRestitution = rectSt[2];
            lastStrikeTid = Math.round(telemView[32 + m * 56 + 1]);
            lastStrikeFrame = fIdx + Math.round(leadFrames);
            lastStrikeValid = true;
          }
        }

        if (labMode === 2) {
          shockTriggerFrame[shockWriteHead] = fIdx + Math.round(leadFrames);
          shockX[shockWriteHead] = sxPos;
          shockY[shockWriteHead] = syPos;
          shockZone[shockWriteHead] = zone;
          shockVel[shockWriteHead] = vel;
          shockInDeg[shockWriteHead] = rawIn;
          shockDeflectDeg[shockWriteHead] = rawDefl;
          shockRestitution[shockWriteHead] = rawRest;
          shockAge[shockWriteHead] = -1;
          shockWriteHead = (shockWriteHead + 1) % SHOCK_POOL;

          self.postMessage({
            type: 'audio_percussion',
            zone,
            leadSec: leadFrames / effSampleRate,
            velocity: vel,
            sentEpochMs: nowEpochMs,
          });
        }
      }
    }
  }

  if (telemView[TELEM_GLOBAL + 2] > 2.0) {
    const rectSc = rectifyScatterKinematics(
      telemView[TELEM_GLOBAL + 3],
      telemView[TELEM_GLOBAL + 4],
      telemView[TELEM_GLOBAL + 5],
      telemView[TELEM_GLOBAL + 6],
      telemView[TELEM_GLOBAL + 2]
    );
    lastScatterDeg = rectSc[2];
    lastScatterFrame = Math.round(telemView[TELEM_GLOBAL + 7]);
  }

  updatePersistenceState(fpsDt);
  pushPhysicsPhaseSample();

  // Render all OffscreenCanvases inside Worker isolate
  renderMainHud(dispSlot);
  renderEulerianPip();
  renderAllSlotPips(true);
  renderHilbertScope();
  renderSpectralAndRingScope();
  renderPersistenceTimeline(false);
  renderPhysicsPhaseCanvas(false);

  // Zero-allocation per-frame scientific telemetry recording (1:1 frame fidelity, 56 columns)
  if (recActive) {
    const effRateRec = sourceMode === 'testbench' ? 60.0 : Math.max(8.0, sourceFps);
    const pSlot = Math.round(telemView[8]);
    const ro = recHead * REC_COLS;
    recElapsedSec += Math.min(1.0, rawDt);

    recSlab[ro + 0] = fIdx;
    recSlab[ro + 1] = recElapsedSec;
    recSlab[ro + 2] = frameW;
    recSlab[ro + 3] = frameH;
    recSlab[ro + 4] = Math.round(telemView[1]);
    recSlab[ro + 5] = labMode;
    recSlab[ro + 6] = wasmMs;
    recSlab[ro + 7] = Math.round(telemView[3]);
    recSlab[ro + 8] = telemView[4];
    recSlab[ro + 9] = telemView[23];
    const oracleAvailRec = sourceMode === 'testbench' && !isChronos1000Scenario(benchScenario);
    recSlab[ro + 10] = Math.round(telemView[5]);
    recSlab[ro + 11] = oracleAvailRec ? oraclePadicSwaps : Number.NaN;
    recSlab[ro + 12] = Math.round(telemView[6]);
    recSlab[ro + 13] = inPillarCollisions;
    recSlab[ro + 14] = lambda2;
    recSlab[ro + 15] = telemView[24];
    recSlab[ro + 16] = telemView[25];

    if (pSlot >= 0 && telemView[32 + pSlot * 56] > 0.5) {
      const b = 32 + pSlot * 56;
      const kb = TELEM_KINETIC + pSlot * 8;
      const fxPx = telemView[b + 2];
      const fyPx = telemView[b + 3];
      const hi = Math.round(telemView[b + 12]) & 0xffff;
      const lo = Math.round(telemView[b + 13]) & 0xffff;
      recSlab[ro + 17] = Math.round(telemView[b + 1]);
      recSlab[ro + 18] = hi * 65536.0 + lo;
      recSlab[ro + 19] = Math.round(telemView[b + 14]);
      recSlab[ro + 20] = telemView[b + 4];
      recSlab[ro + 21] = telemView[b + 5];
      recSlab[ro + 22] = fxPx;
      recSlab[ro + 23] = fyPx;
      recSlab[ro + 24] = telemView[b + 6];
      recSlab[ro + 25] = telemView[b + 7];
      recSlab[ro + 26] = telemView[b + 10];
      recSlab[ro + 27] = telemView[b + 11];
      recSlab[ro + 36] = telemView[b + 17] * (1000.0 / effRateRec);

      const physVel = mapVelocityToPhysical(fxPx, fyPx, telemView[kb + 1], telemView[kb + 0]);
      recSlab[ro + 38] = physVel[1];
      recSlab[ro + 39] = physVel[0];
      recSlab[ro + 40] = (calMode === 'homography' && calHomographyValid && physVel[2] > 0.5)
        ? slotRectAngVel[pSlot]
        : telemView[kb + 2];
      if (telemView[kb + 6] > -9.0e5 && telemView[kb + 7] > -9.0e5) {
        const pend = computePendulumKinematics(fxPx, fyPx, telemView[kb + 6], telemView[kb + 7]);
        recSlab[ro + 41] = pend[0];
      } else {
        recSlab[ro + 41] = 0.0;
      }
      const sxRec = slotLastStrikeX[pSlot] > 0.0 ? slotLastStrikeX[pSlot] : fxPx;
      const syRec = slotLastStrikeY[pSlot] > 0.0 ? slotLastStrikeY[pSlot] : fyPx;
      const stRect = rectifyStrikeKinematics(sxRec, syRec, telemView[kb + 5], telemView[kb + 3], telemView[kb + 4]);
      recSlab[ro + 42] = stRect[1];
      recSlab[ro + 43] = stRect[2];
      const mPt = mapPixelToMetric(fxPx, fyPx);
      recSlab[ro + 44] = mPt[0];
      recSlab[ro + 45] = mPt[1];
      recSlab[ro + 46] = calSegmentId;
      recSlab[ro + 47] = physVel[2];
      const pb = TELEM_PHYSICS_START + pSlot * TELEM_PHYSICS_STRIDE;
      recSlab[ro + 48] = Math.max(0, Math.min(8, Math.round(safeFinite(telemView[pb + 4]))));
      recSlab[ro + 49] = safeFinite(telemView[pb + 5]);
      recSlab[ro + 50] = safeFinite(telemView[pb + 6]);
      recSlab[ro + 51] = safeFinite(telemView[pb + 7]);
      recSlab[ro + 52] = safeFinite(telemView[pb + 8]);
      recSlab[ro + 53] = safeFinite(telemView[pb + 10]);
    } else {
      for (let c = 17; c <= 27; c++) recSlab[ro + c] = 0.0;
      recSlab[ro + 36] = 0.0;
      for (let c = 38; c <= 45; c++) recSlab[ro + c] = 0.0;
      recSlab[ro + 46] = calSegmentId;
      recSlab[ro + 47] = isMetricCalibrated() ? 1.0 : 0.0;
      for (let c = 48; c <= 53; c++) recSlab[ro + c] = 0.0;
    }

    recSlab[ro + 54] = safeFinite(telemView[TELEM_ENSEMBLE_START + 0]);
    recSlab[ro + 55] = safeFinite(telemView[TELEM_ENSEMBLE_START + 11]);
    recSlab[ro + 28] = telemView[9];
    recSlab[ro + 29] = telemView[10];
    recSlab[ro + 30] = telemView[11] * (180.0 / Math.PI);
    recSlab[ro + 31] = telemView[13];
    recSlab[ro + 32] = telemView[12];
    recSlab[ro + 33] = telemView[26] > 0.5 ? 1 : 0;
    recSlab[ro + 34] = telemView[14] > 0.5 ? 1 : 0;
    recSlab[ro + 35] = Math.round(telemView[15]);
    recSlab[ro + 37] = telemView[19];

    recHead = (recHead + 1) % REC_MAX_FRAMES;
    if (recCount < REC_MAX_FRAMES) recCount++;
    recTotalWritten++;
  }

  tickCounter++;
  if (tickCounter % 3 === 0 || sourceMode !== 'testbench') {
    emitTelemetrySnapshot();
  }
}

function configureResolutionViews(mode) {
  refreshChronos1000Views();
  const targetMode = mode ? 1 : 0;
  wasmExports.chronos_set_resolution(targetMode);
  resMode = wasmExports.chronos_get_resolution();
  frameW = wasmExports.chronos_get_frame_w();
  frameH = wasmExports.chronos_get_frame_h();
  pipDim = wasmExports.chronos_get_pip_dim();
  resScale = resMode === 1 ? 2 : 1;
  framePixels = frameW * frameH;
  frameBytes = framePixels * 4;

  dispStageCanvas.width = frameW;
  dispStageCanvas.height = frameH;
  pipStageCanvas.width = pipDim;
  pipStageCanvas.height = pipDim;
  simCanvas.width = frameW;
  simCanvas.height = frameH;
  ecaStageCanvas.width = frameW;
  ecaStageCanvas.height = frameH;

  ecaOverlayBuf = ecaOverlayBackingBuf.subarray(0, frameBytes);
  ecaOverlayImgData = new ImageData(ecaOverlayBuf, frameW, frameH);

  for (let s = 0; s < RING_LEN; s++) {
    const byteOff = wasmExports.chronos_get_ring_slot_ptr(s);
    ringSlotU8[s] = new Uint8Array(wasmMemory.buffer, byteOff, frameBytes);
    ringSlotImgData[s] = new ImageData(
      new Uint8ClampedArray(wasmMemory.buffer, byteOff, frameBytes),
      frameW,
      frameH
    );
  }

  ecaResView = new Float32Array(wasmMemory.buffer, wasmExports.chronos_get_eca_res_ptr(), framePixels);
  const pipBytes = pipDim * pipDim * 4;
  pipImgData = new ImageData(
    new Uint8ClampedArray(wasmMemory.buffer, wasmExports.chronos_get_pip_rgba_ptr(), pipBytes),
    pipDim,
    pipDim
  );
  if (typeof wasmExports.chronos_get_slot_meta_ptr === 'function') {
    slotMetaView = new Float32Array(
      wasmMemory.buffer,
      wasmExports.chronos_get_slot_meta_ptr(),
      MAX_TARGETS * 8
    );
  }
  if (typeof wasmExports.chronos_get_all_pip_rgba_ptr === 'function') {
    const allPipBase = wasmExports.chronos_get_all_pip_rgba_ptr();
    for (let m = 0; m < MAX_TARGETS; m++) {
      slotPipImgData[m] = new ImageData(
        new Uint8ClampedArray(wasmMemory.buffer, allPipBase + m * MAX_PIP_BYTES, pipBytes),
        pipDim,
        pipDim
      );
    }
  }
  slotIdleDrawn.fill(0);

  wasmTimeHist.fill(0.0);
  timeHistIdx = 0;
  tickCounter = 0;
  clearVisualQueues();
  resetTestbenchState();
}

function redrawPausedHudIfReady() {
  if (wasmExports && telemView && Math.round(telemView[0]) > 0) {
    const dispSlot = Math.round(telemView[2]);
    renderMainHud(dispSlot);
    renderEulerianPip();
    renderAllSlotPips(false);
    renderHilbertScope();
    renderSpectralAndRingScope();
    renderPersistenceTimeline(true);
    renderPhysicsPhaseCanvas(true);
    emitTelemetrySnapshot();
  }
}

function commitSurveyedCalibrationFromDraft(targetMode, reason) {
  if (targetMode === 'ruler') {
    const spanPx = Math.hypot(
      (calDraftPtsNormX[1] - calDraftPtsNormX[0]) * frameW,
      (calDraftPtsNormY[1] - calDraftPtsNormY[0]) * frameH
    );
    if (spanPx < 4.0) {
      calArmed = 'none';
      calDraftCount = 0;
      emitCalibrationState('RULER REJECTED (< 4 px SPAN) · PRIOR CALIBRATION PRESERVED');
      return false;
    }
    calRulerMeters = calDraftRulerMeters;
    calPtsNormX[0] = calDraftPtsNormX[0];
    calPtsNormY[0] = calDraftPtsNormY[0];
    calPtsNormX[1] = calDraftPtsNormX[1];
    calPtsNormY[1] = calDraftPtsNormY[1];
    calPtCount = 2;
    calDraftCount = 0;
    calArmed = 'none';
    calMode = 'ruler';
    calProvenance = 'surveyed';
    calPresetName = '';
    calHomographyValid = false;
    recordCalibrationSegmentTransition(reason || 'surveyed_ruler');
    emitCalibrationState();
    return true;
  }

  if (targetMode === 'homography') {
    const solved = solveHomography4Pt(calDraftPlaneW, calDraftPlaneH, calDraftPtsNormX, calDraftPtsNormY, true);
    if (!solved) {
      calArmed = 'none';
      calDraftCount = 0;
      emitCalibrationState('INVALID QUAD (NON-CONVEX / HORIZON) · PRIOR CALIBRATION PRESERVED');
      return false;
    }
    calPlaneW = calDraftPlaneW;
    calPlaneH = calDraftPlaneH;
    for (let i = 0; i < 4; i++) {
      calPtsNormX[i] = calDraftPtsNormX[i];
      calPtsNormY[i] = calDraftPtsNormY[i];
    }
    calPtCount = 4;
    calDraftCount = 0;
    calArmed = 'none';
    calHomographyValid = true;
    calMode = 'homography';
    calProvenance = 'surveyed';
    calPresetName = '';
    recordCalibrationSegmentTransition(reason || 'surveyed_homography');
    emitCalibrationState();
    return true;
  }
  return false;
}

function syncSwarmConfigToMain(sc) {
  self.postMessage({
    type: 'swarm_config_sync',
    scenario: sc,
    resMode,
    frameW,
    frameH,
    pipDim,
    swarmMode,
    userSwarmMode,
    denseSwarmOverrideActive,
    swarmActive: Boolean(hasChronos1000Exports() && swarmTelemView && swarmTelemView[1] > 0.5),
    fieldOverlayMode,
    showHeroTargets,
    overlaySyncSeq: lastOverlaySyncSeq,
  });
}

function applyScenarioResolutionAndSwarm(sc) {
  if (sourceMode === 'testbench' && isChronos1000Scenario(sc)) {
    if (resMode !== 1 && wasmExports) {
      configureResolutionViews(1);
      self.postMessage({
        type: 'resolution_changed',
        resMode,
        frameW,
        frameH,
        pipDim,
      });
      emitCalibrationState();
    }
    denseSwarmOverrideActive = true;
    if (swarmMode !== 2) {
      swarmMode = 2;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(2);
      }
    }
    syncSwarmConfigToMain(sc);
    return;
  }

  if (denseSwarmOverrideActive) {
    denseSwarmOverrideActive = false;
    if (swarmMode !== userSwarmMode) {
      swarmMode = userSwarmMode;
      swarmRingValid.fill(0);
      clearActiveInspectionSelection();
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
    }
    syncSwarmConfigToMain(sc);
  }
}

function applyTestbenchScenarioSwitch(sc) {
  benchScenario = sc || 'pillar_crossing';
  wasmTimeHist.fill(0.0);
  timeHistIdx = 0;
  tickCounter = 0;
  clearVisualQueues();
  applyScenarioResolutionAndSwarm(benchScenario);
  if (wasmExports) wasmExports.chronos_reset_tracks();
  resetTestbenchState();
  redrawPausedHudIfReady();
  emitTelemetrySnapshot();
}

function findLiveMicroTrackIndexByTid(tid, confirmedOnly = false) {
  if (!microTracksView || !Number.isInteger(tid) || tid < 1 || tid > 16777215) return -1;
  const maxState = confirmedOnly ? 1 : 2;
  for (let i = 0; i < MAX_MICRO; i++) {
    const b = i * MICRO_STRIDE;
    const st = intOrNull(microTracksView[b + 5], 1, maxState);
    const curTid = intOrNull(microTracksView[b], 1, 16777215);
    if (st !== null && curTid === tid) {
      return i;
    }
  }
  return -1;
}

function inspectFieldPointAt(rawX, rawY, reqSeq = null, rawNormX = null, rawNormY = null) {
  const maxX = Math.max(0.0, frameW - 1e-3);
  const maxY = Math.max(0.0, frameH - 1e-3);
  const x = Math.max(0.0, Math.min(maxX, Number.isFinite(rawX) ? rawX : frameW * 0.5));
  const y = Math.max(0.0, Math.min(maxY, Number.isFinite(rawY) ? rawY : frameH * 0.5));
  const abiOk = hasChronos1000Exports();

  lastInspectResSeq = reqSeq !== null ? Math.max(lastInspectResSeq + 1, reqSeq) : lastInspectResSeq + 1;
  lastInspectReqSeq = reqSeq !== null ? reqSeq : lastInspectResSeq;
  lastInspectReqX = x;
  lastInspectReqY = y;
  lastInspectReqNormX = frameW > 0 ? x / frameW : (Number.isFinite(rawNormX) ? Math.max(0.0, Math.min(1.0, rawNormX)) : 0.0);
  lastInspectReqNormY = frameH > 0 ? y / frameH : (Number.isFinite(rawNormY) ? Math.max(0.0, Math.min(1.0, rawNormY)) : 0.0);

  // Map click from displayed ring slot (t-effK) onto exact live confirmed track identity
  let bestHistTid = 0;
  const qScale = Math.max(0.25, frameW / 640.0);
  const dispSlot = telemView ? Math.round(telemView[2]) : -1;
  if (dispSlot >= 0 && dispSlot < RING_LEN && swarmRingValid[dispSlot] === 1) {
    const dispMicro = swarmRingMicro[dispSlot];
    let bestD2 = (16.0 * qScale) * (16.0 * qScale);
    for (let i = 0; i < MAX_MICRO; i++) {
      const b = i * MICRO_STRIDE;
      const st = intOrNull(dispMicro[b + 5], 1, 1);
      const candTid = intOrNull(dispMicro[b], 1, 16777215);
      if (st === null || candTid === null) continue;
      const px = dispMicro[b + 1];
      const py = dispMicro[b + 2];
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      const dx = px - x;
      const dy = py - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestHistTid = candTid;
      }
    }
  }

  if (!abiOk) {
    inspectedMicroTid = 0;
    lastInspectStatus = 'unavailable';
    redrawPausedHudIfReady();
    emitTelemetrySnapshot();
    return;
  }

  if (bestHistTid > 0) {
    const liveIdx = findLiveMicroTrackIndexByTid(bestHistTid, true);
    let pinnedOk = false;
    if (liveIdx >= 0) {
      let ret = 0;
      if (typeof wasmExports.chronos_inspect_micro_track === 'function') {
        ret = wasmExports.chronos_inspect_micro_track(bestHistTid);
      } else {
        const lb = liveIdx * MICRO_STRIDE;
        const lx = Math.max(0.0, Math.min(maxX, microTracksView[lb + 1]));
        const ly = Math.max(0.0, Math.min(maxY, microTracksView[lb + 2]));
        ret = wasmExports.chronos_inspect_field_point(lx, ly);
      }
      const retTid = intOrNull(ret, 1, 16777215);
      if (retTid === bestHistTid) {
        inspectedMicroTid = bestHistTid;
        lastInspectStatus = 'pinned';
        wasmExports.chronos_set_focus_target(bestHistTid);
        pinnedOk = true;
      }
    }
    if (!pinnedOk) {
      inspectedMicroTid = 0;
      lastInspectStatus = 'expired';
      wasmExports.chronos_set_focus_target(0);
    }
  } else {
    const ret = wasmExports.chronos_inspect_field_point(x, y);
    const retTid = intOrNull(ret, 1, 16777215);
    const candTid = retTid ?? 0;
    if (candTid > 0 && findLiveMicroTrackIndexByTid(candTid, true) >= 0) {
      inspectedMicroTid = candTid;
      lastInspectStatus = 'live';
      wasmExports.chronos_set_focus_target(candTid);
    } else {
      inspectedMicroTid = 0;
      lastInspectStatus = 'miss';
      wasmExports.chronos_set_focus_target(0);
    }
  }

  redrawPausedHudIfReady();
  emitTelemetrySnapshot();
}

function handleHudCanvasClick(normX, normY, intent = '', expectedCalArmed = '') {
  if (!wasmExports || !telemView) return;

  if (calArmed !== 'none' && expectedCalArmed && expectedCalArmed !== calArmed) {
    return;
  }

  // Stage interactive 2-Pt Ruler & 4-Pt Plane Homography clicks in draft buffers so Esc/invalid preserves prior state!
  if (calArmed === 'ruler') {
    calDraftPtsNormX[calDraftCount] = normX;
    calDraftPtsNormY[calDraftCount] = normY;
    calDraftCount++;
    if (calDraftCount >= 2) {
      commitSurveyedCalibrationFromDraft('ruler', 'canvas_ruler');
    } else {
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
    return;
  }

  if (calArmed === 'homography') {
    calDraftPtsNormX[calDraftCount] = normX;
    calDraftPtsNormY[calDraftCount] = normY;
    calDraftCount++;
    if (calDraftCount >= 4) {
      commitSurveyedCalibrationFromDraft('homography', 'canvas_homography');
    } else {
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
    return;
  }

  // Never let a stale queued calibration click fall through into Hero target focus pinning.
  if (intent === 'calibration') {
    return;
  }

  const cx = normX * frameW;
  const cy = normY * frameH;
  const hitRadius = 28.0 * resScale;
  let bestDist = hitRadius;
  let bestTid = 0;

  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;
    const fx = telemView[base + 2];
    const fy = telemView[base + 3];
    const d = Math.hypot(cx - fx, cy - fy);
    if (d <= bestDist) {
      bestDist = d;
      bestTid = Math.round(telemView[base + 1]);
    }
  }

  const curFocus = wasmExports.chronos_get_focus_target();
  if (bestTid > 0 && bestTid !== curFocus) {
    wasmExports.chronos_set_focus_target(bestTid);
  } else {
    wasmExports.chronos_set_focus_target(0);
  }

  // Re-render canvases & KPI snapshot immediately if video is paused
  redrawPausedHudIfReady();
}

function cycleFocusTarget() {
  if (!wasmExports || !telemView) return;
  const activeIds = [];
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] > 0.5) {
      activeIds.push(Math.round(telemView[base + 1]));
    }
  }
  activeIds.sort((a, b) => a - b);
  const curFocus = wasmExports.chronos_get_focus_target();
  if (activeIds.length === 0) {
    wasmExports.chronos_set_focus_target(0);
    return;
  }
  if (curFocus === 0) {
    wasmExports.chronos_set_focus_target(activeIds[0]);
  } else {
    const idx = activeIds.indexOf(curFocus);
    if (idx < 0 || idx + 1 >= activeIds.length) {
      wasmExports.chronos_set_focus_target(0);
    } else {
      wasmExports.chronos_set_focus_target(activeIds[idx + 1]);
    }
  }
  redrawPausedHudIfReady();
}

let rafHandle = 0;
let rafWatchdogId = 0;

function cancelWorkerTickLoop() {
  clearTimeout(rafWatchdogId);
  rafWatchdogId = 0;
  if (rafHandle && typeof self.cancelAnimationFrame === 'function') {
    self.cancelAnimationFrame(rafHandle);
  }
  rafHandle = 0;
}

function scheduleNextWorkerTick() {
  if (!running || sourceMode !== 'testbench') return;
  if (typeof self.requestAnimationFrame === 'function') {
    clearTimeout(rafWatchdogId);
    const prevTick = tickCounter;
    rafHandle = self.requestAnimationFrame(workerRafLoop);
    rafWatchdogId = setTimeout(() => {
      if (running && sourceMode === 'testbench' && tickCounter === prevTick) {
        if (typeof self.cancelAnimationFrame === 'function') {
          self.cancelAnimationFrame(rafHandle);
        }
        workerRafLoop(performance.now());
      }
    }, 18);
  } else {
    rafWatchdogId = setTimeout(() => workerRafLoop(performance.now()), 16);
  }
}

function workerRafLoop(now) {
  if (!running || sourceMode !== 'testbench') return;
  clearTimeout(rafWatchdogId);
  const dtMs = lastRafTs > 0 ? Math.min(50.0, Math.max(4.0, now - lastRafTs)) : 16.667;
  lastRafTs = now;
  const instFps = 1000.0 / dtMs;
  smoothFps = 0.92 * smoothFps + 0.08 * instFps;

  try {
    if (wasmExports) {
      const simDt = 1.0 / 60.0;
      const rawPixels = stepAndRenderBlindPillarScene(simDt);
      processIngestedFrame(rawPixels, simDt, -1.0);
    }
  } finally {
    scheduleNextWorkerTick();
  }
}

// ============================================================================
// Worker Message Control Plane
// ============================================================================

self.onmessage = async (ev) => {
  const msg = ev.data;

  if (msg.type === 'init') {
    hudCanvas = msg.hudCanvas;
    pipCanvas = msg.pipCanvas;
    hilbertCanvas = msg.hilbertCanvas;
    spectralCanvas = msg.spectralCanvas;
    persistenceCanvas = msg.persistenceCanvas || null;
    physicsPhaseCanvas = msg.physicsPhaseCanvas || null;

    hudCtx = hudCanvas.getContext('2d', { alpha: false });
    pipCtx = pipCanvas.getContext('2d', { alpha: false });
    hilbertCtx = hilbertCanvas.getContext('2d', { alpha: false });
    spectralCtx = spectralCanvas.getContext('2d', { alpha: false });
    persistenceCtx = persistenceCanvas ? persistenceCanvas.getContext('2d', { alpha: false }) : null;
    physicsPhaseCtx = physicsPhaseCanvas ? physicsPhaseCanvas.getContext('2d', { alpha: false }) : null;

    if (Array.isArray(msg.slotPipCanvases)) {
      for (let m = 0; m < Math.min(MAX_TARGETS, msg.slotPipCanvases.length); m++) {
        const c = msg.slotPipCanvases[m];
        if (c && typeof c.getContext === 'function') {
          slotPipCanvases[m] = c;
          slotPipCtxs[m] = c.getContext('2d', { alpha: false });
        }
      }
    }

    dispStageCanvas = new OffscreenCanvas(frameW, frameH);
    dispStageCtx = dispStageCanvas.getContext('2d');
    pipStageCanvas = new OffscreenCanvas(pipDim, pipDim);
    pipStageCtx = pipStageCanvas.getContext('2d');
    simCanvas = new OffscreenCanvas(frameW, frameH);
    simCtx = simCanvas.getContext('2d', { willReadFrequently: true });

    ecaStageCanvas = new OffscreenCanvas(frameW, frameH);
    ecaStageCtx = ecaStageCanvas.getContext('2d');

    try {
      const resp = await fetch(msg.wasmUrl || './chronos_spectral.wasm?v=20261115c1k');
      if (!resp.ok) {
        throw new Error(`Wasm fetch failed (HTTP ${resp.status})`);
      }
      const wasmBytes = await resp.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(wasmBytes, {});

      wasmExports = instance.exports;
      wasmMemory = wasmExports.memory;
      wasmExports.chronos_init();
      wasmExports.chronos_set_params(
        lookaheadK,
        ecaAdaptRate,
        fistaLambda,
        eulerianGain,
        phaseOffsetRad,
        labMode
      );

      laplacianView = new Float32Array(
        wasmMemory.buffer,
        wasmExports.chronos_get_laplacian_ptr(),
        GRAPH_NODES * GRAPH_NODES
      );
      fiedlerView = new Float32Array(
        wasmMemory.buffer,
        wasmExports.chronos_get_fiedler_ptr(),
        GRAPH_NODES
      );
      telemView = new Float32Array(
        wasmMemory.buffer,
        wasmExports.chronos_get_telemetry_ptr(),
        TELEMETRY_LEN
      );

      configureResolutionViews(msg.resMode || 0);
      applySceneCalibrationPreset('testbench');
      running = true;
      self.postMessage({
        type: 'ready',
        memoryBytes: wasmMemory.buffer.byteLength,
        pages: wasmMemory.buffer.byteLength / 65536,
        resMode,
        frameW,
        frameH,
        pipDim,
        showHeroTargets,
        overlaySyncSeq: lastOverlaySyncSeq,
      });
      scheduleNextWorkerTick();
    } catch (initErr) {
      running = false;
      self.postMessage({
        type: 'init_error',
        message: initErr && initErr.message ? initErr.message : String(initErr),
      });
    }
  } else if (msg.type === 'set_resolution') {
    if (wasmExports) {
      configureResolutionViews(msg.resMode ? 1 : 0);
      self.postMessage({
        type: 'resolution_changed',
        resMode,
        frameW,
        frameH,
        pipDim,
      });
      emitCalibrationState();
    }
  } else if (msg.type === 'set_scenario') {
    applyTestbenchScenarioSwitch(msg.scenario);
  } else if (msg.type === 'set_crop') {
    if (Array.isArray(msg.cropNorm) && msg.cropNorm.length === 4) {
      const x = Math.max(0.0, Math.min(0.95, Number(msg.cropNorm[0]) || 0.0));
      const y = Math.max(0.0, Math.min(0.95, Number(msg.cropNorm[1]) || 0.0));
      const w = Math.max(0.05, Math.min(1.0 - x, Number(msg.cropNorm[2]) || 1.0));
      const h = Math.max(0.05, Math.min(1.0 - y, Number(msg.cropNorm[3]) || 1.0));
      const cropChanged =
        Math.abs(cropNorm[0] - x) > 1e-6 ||
        Math.abs(cropNorm[1] - y) > 1e-6 ||
        Math.abs(cropNorm[2] - w) > 1e-6 ||
        Math.abs(cropNorm[3] - h) > 1e-6;
      cropNorm[0] = x;
      cropNorm[1] = y;
      cropNorm[2] = w;
      cropNorm[3] = h;
      if (cropChanged && (calMode === 'ruler' || calMode === 'homography' || calArmed !== 'none')) {
        clearCalibrationState();
        recordCalibrationSegmentTransition('crop_changed');
        emitCalibrationState('ROI CROP CHANGED · IMAGE-PLANE CALIBRATION RESET TO PX UNITS');
        redrawPausedHudIfReady();
      }
    }
  } else if (msg.type === 'cal_control') {
    if (typeof msg.syncSeq === 'number' && Number.isFinite(msg.syncSeq) && msg.syncSeq > lastCalSyncSeq) {
      lastCalSyncSeq = Math.trunc(msg.syncSeq);
    }
    const inRulerM = typeof msg.rulerMeters === 'number' && msg.rulerMeters >= 0.01 ? msg.rulerMeters : null;
    const inPlaneW = typeof msg.planeW === 'number' && msg.planeW >= 0.01 ? msg.planeW : null;
    const inPlaneH = typeof msg.planeH === 'number' && msg.planeH >= 0.01 ? msg.planeH : null;

    if (msg.action === 'arm_ruler') {
      if (calArmed === 'ruler') {
        calArmed = 'none';
        calDraftCount = 0;
      } else {
        calDraftRulerMeters = inRulerM ?? calRulerMeters;
        calArmed = 'ruler';
        calDraftCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'arm_homography') {
      if (calArmed === 'homography') {
        calArmed = 'none';
        calDraftCount = 0;
      } else {
        calDraftPlaneW = inPlaneW ?? calPlaneW;
        calDraftPlaneH = inPlaneH ?? calPlaneH;
        calArmed = 'homography';
        calDraftCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'cancel_arm') {
      if (calArmed !== 'none') {
        calArmed = 'none';
        calDraftCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'stage_dims') {
      if (inRulerM !== null) calDraftRulerMeters = inRulerM;
      if (inPlaneW !== null) calDraftPlaneW = inPlaneW;
      if (inPlaneH !== null) calDraftPlaneH = inPlaneH;
    } else if (msg.action === 'set_coords' && Array.isArray(msg.pts)) {
      const targetMode = msg.coordMode === 'homography' ? 'homography' : 'ruler';
      const reqN = targetMode === 'homography' ? 4 : 2;
      if (msg.pts.length >= reqN) {
        const tmpX = new Float64Array(4);
        const tmpY = new Float64Array(4);
        let validNums = true;
        for (let i = 0; i < reqN; i++) {
          const px = Number(msg.pts[i] && msg.pts[i][0]);
          const py = Number(msg.pts[i] && msg.pts[i][1]);
          if (!Number.isFinite(px) || !Number.isFinite(py) || px < 0.0 || px > 1.0 || py < 0.0 || py > 1.0) {
            validNums = false;
            break;
          }
          tmpX[i] = px;
          tmpY[i] = py;
        }
        if (validNums) {
          calDraftRulerMeters = inRulerM ?? calRulerMeters;
          calDraftPlaneW = inPlaneW ?? calPlaneW;
          calDraftPlaneH = inPlaneH ?? calPlaneH;
          for (let i = 0; i < reqN; i++) {
            calDraftPtsNormX[i] = tmpX[i];
            calDraftPtsNormY[i] = tmpY[i];
          }
          calDraftCount = reqN;
          commitSurveyedCalibrationFromDraft(targetMode, `keyboard_${targetMode}`);
        } else {
          emitCalibrationState('COORDINATES MUST BE NORMALIZED IN [0.00, 1.00] · PRIOR CALIBRATION PRESERVED');
        }
      }
    } else if (msg.action === 'set_auto_g') {
      calArmed = 'none';
      calDraftCount = 0;
      calPtCount = 0;
      calPresetName = '';
      calHomographyValid = false;
      calMode = 'auto_g';
      calProvenance = 'assumed';
      calAutoGLocked = false;
      calAutoGNormH = 0.0;
      if (telemView && telemView[TELEM_GLOBAL + 1] >= AUTO_G_LOCK_CONF && telemView[TELEM_GLOBAL + 0] > 50.0) {
        calAutoGNormH = telemView[TELEM_GLOBAL + 0] / frameH;
        calAutoGLocked = true;
      } else if (telemView && telemView[TELEM_GLOBAL + 0] > 50.0) {
        calAutoGNormH = telemView[TELEM_GLOBAL + 0] / frameH;
      }
      recordCalibrationSegmentTransition('set_auto_g');
      emitCalibrationState();
    } else if (msg.action === 'apply_preset') {
      applySceneCalibrationPreset(msg.preset || 'testbench');
    } else if (msg.action === 'update_dims') {
      if (inRulerM !== null) calDraftRulerMeters = inRulerM;
      if (inPlaneW !== null) calDraftPlaneW = inPlaneW;
      if (inPlaneH !== null) calDraftPlaneH = inPlaneH;
      if (calArmed !== 'none') {
        emitCalibrationState();
      } else if (calMode === 'homography' && calPtCount >= 4) {
        const nextW = inPlaneW ?? calPlaneW;
        const nextH = inPlaneH ?? calPlaneH;
        const changed = Math.abs(nextW - calPlaneW) > 1e-6 || Math.abs(nextH - calPlaneH) > 1e-6;
        if (inRulerM !== null) calRulerMeters = inRulerM;
        if (changed) {
          const ok = solveHomography4Pt(nextW, nextH, calPtsNormX, calPtsNormY, true);
          if (ok) {
            calPlaneW = nextW;
            calPlaneH = nextH;
            calHomographyValid = true;
            recordCalibrationSegmentTransition('update_dims');
          }
        }
        emitCalibrationState();
      } else {
        const rulerChanged = inRulerM !== null && Math.abs(inRulerM - calRulerMeters) > 1e-6;
        if (inRulerM !== null) calRulerMeters = inRulerM;
        if (inPlaneW !== null) calPlaneW = inPlaneW;
        if (inPlaneH !== null) calPlaneH = inPlaneH;
        if (calMode === 'ruler' && calPtCount >= 2 && rulerChanged) {
          recordCalibrationSegmentTransition('update_dims');
        }
        emitCalibrationState();
      }
    } else if (msg.action === 'clear') {
      clearCalibrationState();
      recordCalibrationSegmentTransition('clear');
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
  } else if (msg.type === 'canvas_click') {
    if (typeof msg.normX === 'number' && typeof msg.normY === 'number') {
      const intent = typeof msg.intent === 'string' ? msg.intent : '';
      const expectedCalArmed = typeof msg.expectedCalArmed === 'string' ? msg.expectedCalArmed : '';
      handleHudCanvasClick(msg.normX, msg.normY, intent, expectedCalArmed);
    }
  } else if (msg.type === 'cycle_focus') {
    cycleFocusTarget();
  } else if (msg.type === 'pin_target') {
    if (wasmExports) {
      const reqTid = Math.max(0, Math.round(Number(msg.trackId) || 0));
      const curFocus = wasmExports.chronos_get_focus_target();
      wasmExports.chronos_set_focus_target(reqTid > 0 && reqTid !== curFocus ? reqTid : 0);
      redrawPausedHudIfReady();
    }
  } else if (msg.type === 'params') {
    if (typeof msg.overlaySyncSeq === 'number' && msg.overlaySyncSeq > lastOverlaySyncSeq) {
      lastOverlaySyncSeq = msg.overlaySyncSeq;
    }
    let visualFlagChanged = false;
    if (typeof msg.lookaheadK === 'number') lookaheadK = msg.lookaheadK;
    if (typeof msg.ecaAdaptRate === 'number') ecaAdaptRate = msg.ecaAdaptRate;
    if (typeof msg.fistaLambda === 'number') fistaLambda = msg.fistaLambda;
    if (typeof msg.eulerianGain === 'number') eulerianGain = msg.eulerianGain;
    if (typeof msg.phaseOffsetRad === 'number') phaseOffsetRad = msg.phaseOffsetRad;
    if (typeof msg.labMode === 'number') labMode = msg.labMode;
    if (typeof msg.showEcaOverlay === 'boolean') {
      visualFlagChanged = visualFlagChanged || showEcaOverlay !== msg.showEcaOverlay;
      showEcaOverlay = msg.showEcaOverlay;
    }
    if (typeof msg.showKalmanBaseline === 'boolean') {
      visualFlagChanged = visualFlagChanged || showKalmanBaseline !== msg.showKalmanBaseline;
      showKalmanBaseline = msg.showKalmanBaseline;
    }
    if (typeof msg.showHeroTargets === 'boolean') {
      visualFlagChanged = visualFlagChanged || showHeroTargets !== msg.showHeroTargets;
      showHeroTargets = msg.showHeroTargets;
    }
    if (typeof msg.showAllOverlays === 'boolean') {
      visualFlagChanged = visualFlagChanged || showAllOverlays !== msg.showAllOverlays;
      showAllOverlays = msg.showAllOverlays;
    }
    if (typeof msg.strobeEnabled === 'boolean') {
      visualFlagChanged = visualFlagChanged || strobeEnabled !== msg.strobeEnabled;
      strobeEnabled = msg.strobeEnabled;
    }
    if (typeof msg.audioOutLatencySec === 'number') {
      audioOutLatencySec = msg.audioOutLatencySec;
      if (wasmExports) wasmExports.chronos_set_audio_latency(audioOutLatencySec);
    }

    if (wasmExports) {
      wasmExports.chronos_set_params(
        lookaheadK,
        ecaAdaptRate,
        fistaLambda,
        eulerianGain,
        phaseOffsetRad,
        labMode
      );
      if (sourceMode !== 'testbench' || visualFlagChanged) {
        redrawPausedHudIfReady();
        if (visualFlagChanged && (!telemView || Math.round(telemView[0]) <= 0)) {
          emitTelemetrySnapshot();
        }
      }
    }
  } else if (msg.type === 'set_swarm_mode') {
    const rawM = Number(msg.swarmMode);
    const nextMode = Number.isFinite(rawM) ? Math.max(0, Math.min(2, Math.round(rawM))) : 0;
    userSwarmMode = nextMode;
    denseSwarmOverrideActive = false;
    swarmMode = nextMode;
    if (hasChronos1000Exports()) {
      wasmExports.chronos_set_swarm_mode(nextMode);
    }
    if (nextMode === 0) {
      swarmRingValid.fill(0);
      clearActiveInspectionSelection();
    }
    syncSwarmConfigToMain(benchScenario);
    redrawPausedHudIfReady();
    if (!telemView || Math.round(telemView[0]) <= 0) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_field_overlay_mode') {
    if (typeof msg.overlaySyncSeq === 'number' && msg.overlaySyncSeq > lastOverlaySyncSeq) {
      lastOverlaySyncSeq = msg.overlaySyncSeq;
    }
    const rawO = Number(msg.fieldOverlayMode);
    fieldOverlayMode = Number.isFinite(rawO) ? Math.max(0, Math.min(4, Math.round(rawO))) : 0;
    syncSwarmConfigToMain(benchScenario);
    redrawPausedHudIfReady();
    if (!telemView || Math.round(telemView[0]) <= 0) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'inspect_point') {
    if (calArmed === 'none') {
      const parseNorm = (val, fallback = 0.5) => {
        if (val === undefined || val === null || val === '') return fallback;
        const n = Number(val);
        return Number.isFinite(n) ? Math.max(0.0, Math.min(1.0, n)) : fallback;
      };
      const hasAbsX = typeof msg.x === 'number' && Number.isFinite(msg.x);
      const hasAbsY = typeof msg.y === 'number' && Number.isFinite(msg.y);
      const reqNormX = hasAbsX ? (frameW > 0 ? Math.max(0.0, Math.min(1.0, msg.x / frameW)) : 0.0) : parseNorm(msg.normX, 0.5);
      const reqNormY = hasAbsY ? (frameH > 0 ? Math.max(0.0, Math.min(1.0, msg.y / frameH)) : 0.0) : parseNorm(msg.normY, 0.5);
      const qx = hasAbsX ? msg.x : reqNormX * frameW;
      const qy = hasAbsY ? msg.y : reqNormY * frameH;
      const reqSeq = typeof msg.inspectSeq === 'number' && Number.isFinite(msg.inspectSeq)
        ? Math.max(1, Math.trunc(msg.inspectSeq))
        : null;
      inspectFieldPointAt(qx, qy, reqSeq, reqNormX, reqNormY);
    }
  } else if (msg.type === 'set_bench_scenario') {
    applyTestbenchScenarioSwitch(msg.scenario);
  } else if (msg.type === 'set_source') {
    if (typeof msg.scenario === 'string' && msg.scenario) {
      benchScenario = msg.scenario;
    }
    sourceMode = msg.sourceMode;
    let syncSwarmUi = false;
    if (typeof msg.swarmMode === 'number' && Number.isFinite(msg.swarmMode)) {
      const explicitMode = Math.max(0, Math.min(2, Math.round(msg.swarmMode)));
      userSwarmMode = explicitMode;
      denseSwarmOverrideActive = false;
      swarmMode = explicitMode;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
      syncSwarmUi = true;
    } else if (typeof msg.swarmOverride === 'number' && Number.isFinite(msg.swarmOverride)) {
      const ovMode = Math.max(0, Math.min(2, Math.round(msg.swarmOverride)));
      denseSwarmOverrideActive = true;
      swarmMode = ovMode;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
      syncSwarmUi = true;
    } else if (denseSwarmOverrideActive) {
      denseSwarmOverrideActive = false;
      swarmMode = userSwarmMode;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
      syncSwarmUi = true;
    }

    if (sourceMode === 'testbench') {
      applyScenarioResolutionAndSwarm(benchScenario);
    }
    if (syncSwarmUi && !(sourceMode === 'testbench' && isChronos1000Scenario(benchScenario))) {
      syncSwarmConfigToMain(benchScenario);
    }

    clearVisualQueues();
    if (typeof msg.fpsHint === 'number' && msg.fpsHint >= 8.0) {
      sourceFps = msg.fpsHint;
    } else if (sourceMode !== 'testbench') {
      sourceFps = 30.0;
    }
    cropNorm[0] = 0.0;
    cropNorm[1] = 0.0;
    cropNorm[2] = 1.0;
    cropNorm[3] = 1.0;
    wasmTimeHist.fill(0.0);
    timeHistIdx = 0;
    tickCounter = 0;
    lastRafTs = 0;
    clearCalibrationState();
    recordCalibrationSegmentTransition(`source:${sourceMode}`);
    emitCalibrationState();
    if (wasmExports) {
      wasmExports.chronos_reset_tracks();
    }
    resetTestbenchState();
    if (sourceMode === 'testbench') {
      applySceneCalibrationPreset('testbench');
      scheduleNextWorkerTick();
    } else {
      cancelWorkerTickLoop();
    }
  } else if (msg.type === 'video_frame') {
    const bmp = msg.bitmap;
    try {
      if (wasmExports && sourceMode !== 'testbench' && bmp) {
        drawCroppedLetterboxedFrame(bmp);
        const rawPixels = simCtx.getImageData(0, 0, frameW, frameH).data;
        processIngestedFrame(rawPixels, msg.dtSec || 1.0 / 30.0, msg.mediaTimeSec);
      }
    } finally {
      if (bmp && typeof bmp.close === 'function') bmp.close();
      self.postMessage({
        type: 'frame_ack',
        epoch: msg.epoch || 0,
        seq: msg.seq || 0,
      });
    }
  } else if (msg.type === 'video_rgba') {
    if (wasmExports && sourceMode !== 'testbench' && msg.buffer) {
      const rawPixels = new Uint8Array(msg.buffer);
      if (rawPixels.byteLength === frameBytes) {
        processIngestedFrame(rawPixels, msg.dtSec || 1.0 / 30.0, msg.mediaTimeSec);
      }
    }
  } else if (msg.type === 'lab_action') {
    if (msg.action === 'saccade_step') {
      if (wasmExports) wasmExports.chronos_trigger_stimulus();
      if (sourceMode === 'testbench') {
        pendingSaccadeFrames = 9;
      }
    } else if (msg.action === 'ballistic_toss' && sourceMode === 'testbench') {
      if (benchScenario === 'model_rocket_flight') {
        simDiscs[0].x = simDiscs[0].baseX;
        simDiscs[0].y = 152 * resScale;
        simDiscs[0].vx = 0.35 * resScale;
        simDiscs[0].vy = -1.6 * resScale;
        simDiscs[0].ballistic = true;
        rocketPhase[0] = 0;
        rocketBoostTicks[0] = 0;
      } else {
        simDiscs[0].x = 24 * resScale;
        simDiscs[0].y = 148 * resScale;
        simDiscs[0].vx = 3.8 * resScale;
        simDiscs[0].vy = -6.8 * resScale;
        simDiscs[0].ballistic = true;
      }
    } else if (msg.action === 'air_strike' && sourceMode === 'testbench') {
      const nActiveStrike = getActiveSimDiscCount();
      for (let i = 0; i < nActiveStrike; i++) {
        simDiscs[i].vx = -simDiscs[i].vx * 1.45;
      }
      pendingAirSnapFrames = 7;
    } else if (msg.action === 'reset') {
      if (wasmExports) wasmExports.chronos_reset_tracks();
      resetTestbenchState();
    }
  } else if (msg.type === 'rec_control') {
    if (msg.action === 'start') {
      if (msg.clear || recCalSegments.length === 0) {
        recHead = 0;
        recCount = 0;
        recTotalWritten = 0;
        recElapsedSec = 0.0;
        recCalSegments = [
          {
            segmentId: calSegmentId,
            absStartFrame: 0,
            startRow: 0,
            elapsedSec: 0.0,
            reason: 'rec_start',
            sourceMode,
            calibration: getCalibrationSnapshot(),
          },
        ];
      } else {
        const lastSeg = recCalSegments[recCalSegments.length - 1];
        if (!lastSeg || lastSeg.segmentId !== calSegmentId || lastSeg.sourceMode !== sourceMode) {
          recCalSegments.push({
            segmentId: calSegmentId,
            absStartFrame: recTotalWritten,
            startRow: recCount,
            elapsedSec: Number(recElapsedSec.toFixed(4)),
            reason: 'rec_resume_sync',
            sourceMode,
            calibration: getCalibrationSnapshot(),
          });
        }
      }
      recActive = true;
    } else if (msg.action === 'stop') {
      recActive = false;
    } else if (msg.action === 'clear') {
      recActive = false;
      recHead = 0;
      recCount = 0;
      recTotalWritten = 0;
      recElapsedSec = 0.0;
      recCalSegments = [];
    }
    self.postMessage({
      type: 'rec_state',
      recActive,
      recFrames: recCount,
      recDurationSec: recElapsedSec,
    });
  } else if (msg.type === 'export_telemetry') {
    const rows = recCount;
    const ordered = new Float64Array(rows * REC_COLS);
    if (rows > 0) {
      const oldest = rows < REC_MAX_FRAMES ? 0 : recHead;
      for (let i = 0; i < rows; i++) {
        const srcOff = ((oldest + i) % REC_MAX_FRAMES) * REC_COLS;
        ordered.set(recSlab.subarray(srcOff, srcOff + REC_COLS), i * REC_COLS);
      }
    }
    const winStartAbs = Math.max(0, recTotalWritten - rows);
    const exportedSegments = [];
    for (let i = 0; i < recCalSegments.length; i++) {
      const seg = recCalSegments[i];
      const nextAbs = i + 1 < recCalSegments.length ? recCalSegments[i + 1].absStartFrame : recTotalWritten;
      if (rows > 0 && nextAbs <= winStartAbs && i + 1 < recCalSegments.length) {
        continue;
      }
      const absStart = typeof seg.absStartFrame === 'number' ? seg.absStartFrame : seg.startRow;
      const startRow = Math.min(rows, Math.max(0, absStart - winStartAbs));
      exportedSegments.push({
        segmentId: seg.segmentId,
        startRow,
        elapsedSec: seg.elapsedSec,
        reason: seg.reason,
        sourceMode: seg.sourceMode,
        calibration: seg.calibration,
      });
    }
    self.postMessage(
      {
        type: 'telemetry_export_data',
        format: msg.format || 'csv',
        rows,
        cols: REC_COLS,
        columns: REC_COLUMNS,
        buffer: ordered.buffer,
        meta: {
          schemaVersion: '1.5.0',
          sourceMode,
          benchScenario,
          resMode,
          frameW,
          frameH,
          pipDim,
          lookaheadK,
          ecaAdaptRate,
          fistaLambda,
          eulerianGain,
          phaseOffsetRad,
          labMode,
          calibration: getCalibrationSnapshot(),
          calibrationSegments: exportedSegments,
          chronos1000: (() => {
            const medCount = Math.min(Math.max(1, tickCounter), TIME_HIST_LEN);
            const medMs = tickCounter > 0 ? computeMedianTickMs(medCount) : 0.0;
            const effK = telemView ? intOrNull(telemView[1], 0, RING_LEN - 1) ?? lookaheadK : lookaheadK;
            const fIdxExp = telemView ? intOrNull(telemView[0], 0) ?? 0 : 0;
            const dispSlotExp = telemView ? intOrNull(telemView[2], 0, RING_LEN - 1) ?? -1 : -1;
            const snap = buildChronos1000Snapshot(medMs, effK, fIdxExp, dispSlotExp);
            const bodies = [];
            const validMaskExp = Number.isInteger(snap.validityMask) ? snap.validityMask : 0;
            if (rigidBodiesView && hasChronos1000Exports() && (validMaskExp & 4) !== 0) {
              for (let b = 0; b < MAX_RIGID_BODIES; b++) {
                const rb = parseRigidBodyDescriptor(rigidBodiesView, b * BODY_STRIDE);
                if (rb) {
                  bodies.push(rb);
                }
              }
            }
            return {
              ...snap,
              rigidBodies: bodies,
            };
          })(),
        },
      },
      [ordered.buffer]
    );
  }
};
