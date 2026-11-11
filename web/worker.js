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
const MAX_TARGETS = 8;
const GRAPH_NODES = 16;
const TELEMETRY_LEN = 648;

let resMode = 0; // 0: 320x180 SD (16x16 patch/PiP), 1: 640x360 HD (32x32 foveated patch/PiP)
let frameW = 320;
let frameH = 180;
let framePixels = frameW * frameH;
let frameBytes = framePixels * 4;
let pipDim = 16;
let resScale = 1;

let wasmExports = null;
let wasmMemory = null;

// Pre-allocated permanent views into 256-page (16 MB) static Wasm linear memory
const ringSlotU8 = new Array(RING_LEN);
const ringSlotImgData = new Array(RING_LEN);
const ecaOverlayBackingBuf = new Uint8ClampedArray(MAX_FRAME_BYTES);
let ecaResView = null;
let pipImgData = null;
let laplacianView = null;
let fiedlerView = null;
let telemView = null;

// OffscreenCanvases & 2D contexts (pre-allocated once at boot)
let hudCanvas = null;
let hudCtx = null;
let pipCanvas = null;
let pipCtx = null;
let hilbertCanvas = null;
let hilbertCtx = null;
let spectralCanvas = null;
let spectralCtx = null;

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
let strobeEnabled = true;

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
let lastStrikeValid = false;
let lastScatterDeg = 0.0;

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

// Pre-allocated HUD label collision stagger buffers (zero per-frame allocation)
const MAX_HUD_LABELS = 24;
const hudLabelX = new Float32Array(MAX_HUD_LABELS);
const hudLabelY = new Float32Array(MAX_HUD_LABELS);
const hudLabelW = new Float32Array(MAX_HUD_LABELS);

// Pre-allocated Complete Shifted Lanczos + Invariant-Subspace Restart Workspaces (up to 15 = N - 1)
const MAX_KRYLOV = 15;
const activeNodeIdx = new Int32Array(GRAPH_NODES);
const lanczosQ = new Float64Array(MAX_KRYLOV * GRAPH_NODES);
const lanczosAlpha = new Float64Array(MAX_KRYLOV);
const lanczosBeta = new Float64Array(MAX_KRYLOV);
const lanczosW = new Float64Array(GRAPH_NODES);
const jacobiA = new Float64Array(MAX_KRYLOV * MAX_KRYLOV);
const jacobiV = new Float64Array(MAX_KRYLOV * MAX_KRYLOV);
const prevFiedlerTrack = new Float64Array(MAX_TARGETS);
const eigVecBuf = new Float64Array(GRAPH_NODES);

// Pre-allocated Zero-GC Frame-Accurate Telemetry Session Recorder (18,000 frames = 5 min @ 60 Hz, 46 cols)
const REC_COLS = 46;
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
  'strike_deflect_deg', 'strike_restitution', 'metric_x_m', 'metric_y_m',
];
const recSlab = new Float64Array(REC_MAX_FRAMES * REC_COLS);
let recActive = false;
let recHead = 0;
let recCount = 0;
let recElapsedSec = 0.0;

// ============================================================================
// Pre-Allocated Metric Ruler, Auto-g Scale, & 4-Point Plane Homography Engine
// Stored in normalized [0..1] viewport coordinates so SD <-> HD is automatic!
// ============================================================================
const GRAVITY_M_S2 = 9.80665;
const CORNER_TAGS = ['TL', 'TR', 'BR', 'BL'];
let calMode = 'none';  // 'none' | 'ruler' | 'auto_g' | 'homography'
let calArmed = 'none'; // 'none' | 'ruler' | 'homography'
let calPtCount = 0;
let calRulerMeters = 1.0;
let calPlaneW = 2.84;
let calPlaneH = 1.42;
let calPresetName = '';
let calAutoGNormH = 0.0;         // scene_gravity_px_s2 / frameH (resolution-invariant)
let calAutoGFallbackNormW = 0.0; // optional fallback span in normalized frameW units before 1st toss
const calPtsNormX = new Float64Array(4);
const calPtsNormY = new Float64Array(4);
const calHomography = new Float64Array(9);    // H: (nx, ny, 1)^T -> ~ (X_m, Y_m, 1)^T
const calHomographyInv = new Float64Array(9); // H^-1: (X_m, Y_m, 1)^T -> ~ (nx, ny, 1)^T
let calHomographyValid = false;

// Pre-allocated solver & output scratchpad (zero runtime heap allocations)
const homMatA = new Float64Array(64);
const homVecB = new Float64Array(8);
const homSol = new Float64Array(8);
const homTargetX = new Float64Array(4);
const homTargetY = new Float64Array(4);
const calOutPt = new Float64Array(2);
const calOutNorm = new Float64Array(2);
const calOutVel = new Float64Array(2);     // [speedPhys, rectifiedHeadingDeg]
const calOutStrike = new Float64Array(3);  // [rectInDeg, rectDeflectDeg, rectRestitution]
const calOutScatter = new Float64Array(3); // [rectTh1Deg, rectTh2Deg, rectScatterDeg]
const calOutPend = new Float64Array(2);    // [swingDeg, radiusPhys]

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

function solveHomography4Pt(planeW, planeH) {
  const Wp = Math.max(0.01, planeW);
  const Hp = Math.max(0.01, planeH);
  // Corner order TL -> TR -> BR -> BL mapped to right-handed plane [0..Wp] x [0..Hp] (+X right, +Y forward/up)
  homTargetX[0] = 0.0; homTargetY[0] = Hp;
  homTargetX[1] = Wp;  homTargetY[1] = Hp;
  homTargetX[2] = Wp;  homTargetY[2] = 0.0;
  homTargetX[3] = 0.0; homTargetY[3] = 0.0;

  homMatA.fill(0.0);
  for (let i = 0; i < 4; i++) {
    const x = calPtsNormX[i];
    const y = calPtsNormY[i];
    const X = homTargetX[i];
    const Y = homTargetY[i];
    const r0 = (i * 2) * 8;
    const r1 = (i * 2 + 1) * 8;

    homMatA[r0 + 0] = x;
    homMatA[r0 + 1] = y;
    homMatA[r0 + 2] = 1.0;
    homMatA[r0 + 6] = -x * X;
    homMatA[r0 + 7] = -y * X;
    homVecB[i * 2] = X;

    homMatA[r1 + 3] = x;
    homMatA[r1 + 4] = y;
    homMatA[r1 + 5] = 1.0;
    homMatA[r1 + 6] = -x * Y;
    homMatA[r1 + 7] = -y * Y;
    homVecB[i * 2 + 1] = Y;
  }

  // 8x8 Gaussian elimination with partial pivoting
  for (let col = 0; col < 8; col++) {
    let pivRow = col;
    let maxAbs = Math.abs(homMatA[col * 8 + col]);
    for (let r = col + 1; r < 8; r++) {
      const v = Math.abs(homMatA[r * 8 + col]);
      if (v > maxAbs) {
        maxAbs = v;
        pivRow = r;
      }
    }
    if (maxAbs < 1e-10) return false;
    if (pivRow !== col) {
      for (let c = col; c < 8; c++) {
        const tmp = homMatA[col * 8 + c];
        homMatA[col * 8 + c] = homMatA[pivRow * 8 + c];
        homMatA[pivRow * 8 + c] = tmp;
      }
      const tb = homVecB[col];
      homVecB[col] = homVecB[pivRow];
      homVecB[pivRow] = tb;
    }
    const diag = homMatA[col * 8 + col];
    for (let r = col + 1; r < 8; r++) {
      const f = homMatA[r * 8 + col] / diag;
      if (f === 0.0) continue;
      homMatA[r * 8 + col] = 0.0;
      for (let c = col + 1; c < 8; c++) {
        homMatA[r * 8 + c] -= f * homMatA[col * 8 + c];
      }
      homVecB[r] -= f * homVecB[col];
    }
  }

  for (let r = 7; r >= 0; r--) {
    let sum = homVecB[r];
    for (let c = r + 1; c < 8; c++) {
      sum -= homMatA[r * 8 + c] * homSol[c];
    }
    homSol[r] = sum / homMatA[r * 8 + r];
    if (!Number.isFinite(homSol[r])) return false;
  }

  for (let k = 0; k < 8; k++) calHomography[k] = homSol[k];
  calHomography[8] = 1.0;

  // Verify strictly positive projective depth across all 4 corners (reject self-intersecting bowties)
  for (let i = 0; i < 4; i++) {
    const w = calHomography[6] * calPtsNormX[i] + calHomography[7] * calPtsNormY[i] + 1.0;
    if (w <= 0.02) return false;
  }
  return invert3x3(calHomography, calHomographyInv);
}

function getEffectivePxPerMeter() {
  if (calMode === 'ruler' && calPtCount >= 2) {
    const dx = (calPtsNormX[1] - calPtsNormX[0]) * frameW;
    const dy = (calPtsNormY[1] - calPtsNormY[0]) * frameH;
    const spanPx = Math.hypot(dx, dy);
    return spanPx > 1.0 ? spanPx / Math.max(0.01, calRulerMeters) : 0.0;
  }
  if (calMode === 'auto_g') {
    if (calAutoGNormH > 0.05) {
      return (calAutoGNormH * frameH) / GRAVITY_M_S2;
    }
    if (calAutoGFallbackNormW > 0.05) {
      return (calAutoGFallbackNormW * frameW) / Math.max(0.01, calRulerMeters);
    }
  }
  return 0.0;
}

function isMetricCalibrated() {
  if (calMode === 'homography') return calHomographyValid;
  if (calMode === 'ruler' || calMode === 'auto_g') return getEffectivePxPerMeter() > 0.5;
  return false;
}

function mapPixelToMetric(px, py) {
  if (calMode === 'homography' && calHomographyValid) {
    const nx = px / frameW;
    const ny = py / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > 0.28) {
      calOutPt[0] = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      calOutPt[1] = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;
      return calOutPt;
    }
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
    return calOutPt;
  }
  calOutPt[0] = 0.0;
  calOutPt[1] = 0.0;
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
  if (calMode === 'homography' && calHomographyValid) {
    const nx = px / frameW;
    const ny = py / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > 0.28) {
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
      return calOutVel;
    }
  }
  const ppm = getEffectivePxPerMeter();
  calOutVel[0] = ppm > 0.5 ? speedPxS / ppm : speedPxS;
  calOutVel[1] = headingDeg;
  return calOutVel;
}

function rectifyStrikeKinematics(sxPx, syPx, inDeg, deflectDeg, restitution) {
  if (calMode === 'homography' && calHomographyValid && restitution > 0.01) {
    const nx = sxPx / frameW;
    const ny = syPx / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (Math.abs(w) > 1e-4) {
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
  if (calMode === 'homography' && calHomographyValid) {
    const nx = scxPx / frameW;
    const ny = scyPx / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (Math.abs(w) > 1e-4) {
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
    const p0x = p0[0], p0y = p0[1];
    const p1 = mapPixelToMetric(fx, fy);
    calOutPend[0] = swingDeg;
    calOutPend[1] = Math.hypot(p1[0] - p0x, p1[1] - p0y);
    return calOutPend;
  }
  const ppm = getEffectivePxPerMeter();
  calOutPend[0] = swingDeg;
  calOutPend[1] = ppm > 0.5 ? rPx / ppm : rPx;
  return calOutPend;
}

function formatCalibrationSummary(overrideNotice) {
  if (overrideNotice) return overrideNotice;
  if (calArmed === 'ruler') {
    return `ARMED 2-PT RULER (${calPtCount}/2 CLICKS · L=${calRulerMeters.toFixed(2)} m · ESC CANCEL)`;
  }
  if (calArmed === 'homography') {
    return `ARMED 4-PT PLANE (${calPtCount}/4 [${CORNER_TAGS[Math.min(3, calPtCount)]}] · ${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m)`;
  }
  if (calMode === 'ruler') {
    const ppm = getEffectivePxPerMeter();
    return `2-PT RULER: ${calRulerMeters.toFixed(2)} m (${Math.round(ppm)} px/m · METRIC m/s)`;
  }
  if (calMode === 'auto_g') {
    const ppm = getEffectivePxPerMeter();
    if (calAutoGNormH > 0.05) {
      const gPx = Math.round(calAutoGNormH * frameH);
      return `AUTO-g LOCKED: ĝ=${gPx} px/s² → ${Math.round(ppm)} px/m (9.81 m/s²)`;
    }
    if (ppm > 0.5) {
      return `AUTO-g ARMED (INITIAL ${Math.round(ppm)} px/m · REFINING ON BALLISTIC ARC)`;
    }
    return 'AUTO-g ARMED · AWAITING PARABOLIC FLIGHT ARC (g = 9.81 m/s²)';
  }
  if (calMode === 'homography' && calHomographyValid) {
    return `4-PT HOMOGRAPHY: ${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m PLANE [PERSPECTIVE RECTIFIED]`;
  }
  return 'UNCALIBRATED · PIXEL UNITS (px · px/s)';
}

function getCalibrationSnapshot() {
  return {
    mode: calMode,
    armed: calArmed,
    ptCount: calPtCount,
    rulerMeters: calRulerMeters,
    planeW: calPlaneW,
    planeH: calPlaneH,
    presetName: calPresetName,
    pxPerMeter: Number(getEffectivePxPerMeter().toFixed(4)),
    homographyValid: calHomographyValid,
    homography: calHomographyValid ? Array.from(calHomography) : null,
  };
}

function emitCalibrationState(overrideNotice) {
  self.postMessage({
    type: 'cal_state',
    calMode,
    calArmed,
    calPtCount,
    rulerMeters: calRulerMeters,
    planeW: calPlaneW,
    planeH: calPlaneH,
    presetName: calPresetName,
    calibrated: isMetricCalibrated(),
    summary: formatCalibrationSummary(overrideNotice),
  });
}

function clearCalibrationState() {
  calMode = 'none';
  calArmed = 'none';
  calPtCount = 0;
  calPresetName = '';
  calAutoGNormH = 0.0;
  calAutoGFallbackNormW = 0.0;
  calHomographyValid = false;
  calHomography.fill(0.0);
  calHomographyInv.fill(0.0);
}

function applySceneCalibrationPreset(presetKey) {
  clearCalibrationState();
  calPresetName = presetKey || 'testbench';
  if (presetKey === 'carom_billiards_impulses') {
    // UMB Regulation Match Carom Table playing surface (2.84 m x 1.42 m)
    calPlaneW = 2.84;
    calPlaneH = 1.42;
    calPtsNormX[0] = 0.18; calPtsNormY[0] = 0.21;
    calPtsNormX[1] = 0.82; calPtsNormY[1] = 0.21;
    calPtsNormX[2] = 0.94; calPtsNormY[2] = 0.87;
    calPtsNormX[3] = 0.06; calPtsNormY[3] = 0.87;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
  } else if (presetKey === 'roundabout_occlusion') {
    // Multi-lane highway interchange deck (20.0 m width x 32.0 m longitudinal span)
    calPlaneW = 20.00;
    calPlaneH = 32.00;
    calPtsNormX[0] = 0.20; calPtsNormY[0] = 0.17;
    calPtsNormX[1] = 0.78; calPtsNormY[1] = 0.17;
    calPtsNormX[2] = 0.93; calPtsNormY[2] = 0.90;
    calPtsNormX[3] = 0.07; calPtsNormY[3] = 0.90;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
  } else if (presetKey === 'juggling_ballistic') {
    // Newtonian Auto-g (9.80665 m/s^2) with 0.90 m shoulder-span seed prior to 1st apex
    calMode = 'auto_g';
    calRulerMeters = 0.90;
    calAutoGFallbackNormW = 0.42;
    if (telemView && telemView[641] >= 0.25 && telemView[640] > 50.0) {
      calAutoGNormH = telemView[640] / frameH;
    }
  } else if (presetKey === 'galileo_pendulum_peg') {
    // Vertical Galileo pendulum stand chord (0.65 m)
    calMode = 'ruler';
    calRulerMeters = 0.65;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.14;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.78;
    calPtCount = 2;
  } else if (presetKey === 'harmonic_metronome_phase') {
    // Seiko mechanical metronome tower body (0.20 m)
    calMode = 'ruler';
    calRulerMeters = 0.20;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.18;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.84;
    calPtCount = 2;
  } else {
    // Default synthetic testbench / generic video 1.00 m horizontal optical bench span
    calMode = 'ruler';
    calRulerMeters = 1.00;
    calPtsNormX[0] = 0.20; calPtsNormY[0] = 0.50;
    calPtsNormX[1] = 0.80; calPtsNormY[1] = 0.50;
    calPtCount = 2;
  }
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
const PILLARS_NONE = [];

function getActivePillars() {
  if (benchScenario === 'pillar_crossing') return PILLARS_WIDE;
  if (benchScenario === 'short_gap_bridge') return PILLARS_SHORT_GAP;
  return PILLARS_NONE;
}

const simDiscs = [
  { id: 0, x: 52,  y: 56,  baseX: 66,  baseY: 56,  vx: 3.0,  vy: 0.0, ballistic: false, r: '#f83a44', core: '#ffe66e' },
  { id: 1, x: 172, y: 56,  baseX: 254, baseY: 56,  vx: -3.0, vy: 0.0, ballistic: false, r: '#1ee1fa', core: '#0f2d55' },
  { id: 2, x: 148, y: 124, baseX: 66,  baseY: 124, vx: 3.0,  vy: 0.0, ballistic: false, r: '#32f270', core: '#ebffeb' },
  { id: 3, x: 268, y: 124, baseX: 254, baseY: 124, vx: -3.0, vy: 0.0, ballistic: false, r: '#e840f8', core: '#5af5ff' },
];

// Pre-allocated 65,536-ID ownership table (eliminates & 255 ID wrap aliasing)
const MAX_ORACLE_IDS = 65536;
const oracleTrackOwner = new Int16Array(MAX_ORACLE_IDS).fill(-1);
const oracleCandTrackSlot = new Int32Array(MAX_TARGETS * 4);
const oracleCandDiscIdx = new Int32Array(MAX_TARGETS * 4);
const oracleCandDist = new Float32Array(MAX_TARGETS * 4);
let oraclePadicSwaps = 0;
let inPillarCollisions = 0;
let simTimeSec = 0.0;
let pendingSaccadeFrames = 0;
let pendingAirSnapFrames = 0;

const TARGET_COLORS = [
  '#00f0ff', '#ff4d6d', '#38f27a', '#ffb703',
  '#c77dff', '#4cc9f0', '#f72585', '#aacc00',
];

const ZONE_COLORS = ['#00f0ff', '#ffb703', '#38f27a', '#ff4d6d'];

function clearVisualQueues() {
  shockTriggerFrame.fill(-1);
  shockAge.fill(-1);
  shockInDeg.fill(0.0);
  shockDeflectDeg.fill(0.0);
  shockRestitution.fill(0.0);
  lastStrikeDeflectDeg = 0.0;
  lastStrikeRestitution = 0.0;
  lastStrikeValid = false;
  lastScatterDeg = 0.0;
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
}

function resetTestbenchState() {
  const s = resScale;
  if (benchScenario === 'ballistic_cascade') {
    simDiscs[0].x = 42 * s;  simDiscs[0].y = 146 * s; simDiscs[0].baseY = 146 * s; simDiscs[0].vx = 2.6 * s;  simDiscs[0].vy = -6.8 * s; simDiscs[0].ballistic = true;
    simDiscs[1].x = 278 * s; simDiscs[1].y = 138 * s; simDiscs[1].baseY = 146 * s; simDiscs[1].vx = -2.8 * s; simDiscs[1].vy = -5.6 * s; simDiscs[1].ballistic = true;
    simDiscs[2].x = 112 * s; simDiscs[2].y = 148 * s; simDiscs[2].baseY = 148 * s; simDiscs[2].vx = 2.2 * s;  simDiscs[2].vy = -6.2 * s; simDiscs[2].ballistic = true;
    simDiscs[3].x = 210 * s; simDiscs[3].y = 144 * s; simDiscs[3].baseY = 148 * s; simDiscs[3].vx = -2.4 * s; simDiscs[3].vy = -7.0 * s; simDiscs[3].ballistic = true;
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
  oraclePadicSwaps = 0;
  inPillarCollisions = 0;
  simTimeSec = 0.0;
  pendingSaccadeFrames = 0;
  pendingAirSnapFrames = 0;
}

function stepAndRenderBlindPillarScene(dtSec) {
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
    if (pendingAirSnapFrames === 0 && benchScenario !== 'percussion_ricochet' && benchScenario !== 'harmonic_tremor') {
      for (let i = 0; i < simDiscs.length; i++) {
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

  // 2. Advance disc kinematics & collisions
  const tSec = simTimeSec;
  const wallMargin = 18 * s;
  const pillars = getActivePillars();

  for (let i = 0; i < simDiscs.length; i++) {
    const d = simDiscs[i];
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
    for (let i = 0; i < simDiscs.length; i++) {
      for (let j = i + 1; j < simDiscs.length; j++) {
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
  } else if (benchScenario !== 'ballistic_cascade' && benchScenario !== 'harmonic_tremor') {
    for (let lane = 0; lane < 2; lane++) {
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

  // 3. Draw physical textured discs
  for (let i = 0; i < simDiscs.length; i++) {
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
  let nPairs = 0;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;
    const latestVis = telemView[base + 44 + 11];
    if (latestVis < 0.5) continue;

    const ix = telemView[base + 8];
    const iy = telemView[base + 9];

    for (let i = 0; i < simDiscs.length; i++) {
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

function solveShiftedLanczosFiedler() {
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

  let mu = 0.0;
  for (let a = 0; a < n; a++) {
    const ia = activeNodeIdx[a];
    let rowSum = laplacianView[ia * GRAPH_NODES + ia];
    for (let b = 0; b < n; b++) {
      if (a !== b) {
        rowSum += Math.abs(laplacianView[ia * GRAPH_NODES + activeNodeIdx[b]]);
      }
    }
    if (rowSum > mu) mu = rowSum;
  }
  mu += 0.1;

  const mSteps = Math.min(n - 1, MAX_KRYLOV);

  let mean0 = 0.0;
  for (let a = 0; a < n; a++) {
    const val = (a + 1) - 0.5 * (n + 1);
    lanczosQ[a] = val;
    mean0 += val;
  }
  mean0 /= n;
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
  for (let j = 0; j < mSteps; j++) {
    const qOff = j * GRAPH_NODES;

    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a];
      let sumL = 0.0;
      for (let b = 0; b < n; b++) {
        sumL += laplacianView[ia * GRAPH_NODES + activeNodeIdx[b]] * lanczosQ[qOff + b];
      }
      lanczosW[a] = mu * lanczosQ[qOff + a] - sumL;
    }

    let wMean = 0.0;
    for (let a = 0; a < n; a++) wMean += lanczosW[a];
    wMean /= n;
    for (let a = 0; a < n; a++) lanczosW[a] -= wMean;

    if (j > 0 && lanczosBeta[j - 1] > 0.0) {
      const prevOff = (j - 1) * GRAPH_NODES;
      const betaPrev = lanczosBeta[j - 1];
      for (let a = 0; a < n; a++) {
        lanczosW[a] -= betaPrev * lanczosQ[prevOff + a];
      }
    }

    let alpha = 0.0;
    for (let a = 0; a < n; a++) {
      alpha += lanczosQ[qOff + a] * lanczosW[a];
    }
    lanczosAlpha[j] = alpha;

    for (let a = 0; a < n; a++) {
      lanczosW[a] -= alpha * lanczosQ[qOff + a];
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
      wm /= n;
      for (let a = 0; a < n; a++) lanczosW[a] -= wm;
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
      const invBeta = 1.0 / beta;
      for (let a = 0; a < n; a++) {
        lanczosQ[nextOff + a] = lanczosW[a] * invBeta;
      }
    }
  }

  jacobiA.fill(0.0);
  jacobiV.fill(0.0);
  for (let r = 0; r < actualM; r++) {
    jacobiV[r * actualM + r] = 1.0;
    jacobiA[r * actualM + r] = lanczosAlpha[r];
    if (r + 1 < actualM) {
      const b = lanczosBeta[r];
      jacobiA[r * actualM + (r + 1)] = b;
      jacobiA[(r + 1) * actualM + r] = b;
    }
  }

  for (let sweep = 0; sweep < 20; sweep++) {
    let maxOff = 0.0;
    for (let p = 0; p < actualM; p++) {
      for (let q = p + 1; q < actualM; q++) {
        const apq = jacobiA[p * actualM + q];
        const absApq = Math.abs(apq);
        if (absApq > maxOff) maxOff = absApq;
        if (absApq < 1e-12) continue;

        const app = jacobiA[p * actualM + p];
        const aqq = jacobiA[q * actualM + q];
        const tau = (aqq - app) / (2.0 * apq);
        const t = tau >= 0.0
          ? 1.0 / (tau + Math.sqrt(1.0 + tau * tau))
          : -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;

        jacobiA[p * actualM + p] = app - t * apq;
        jacobiA[q * actualM + q] = aqq + t * apq;
        jacobiA[p * actualM + q] = 0.0;
        jacobiA[q * actualM + p] = 0.0;

        for (let k = 0; k < actualM; k++) {
          if (k !== p && k !== q) {
            const akp = jacobiA[k * actualM + p];
            const akq = jacobiA[k * actualM + q];
            const nkp = c * akp - s * akq;
            const nkq = s * akp + c * akq;
            jacobiA[k * actualM + p] = nkp;
            jacobiA[p * actualM + k] = nkp;
            jacobiA[k * actualM + q] = nkq;
            jacobiA[q * actualM + k] = nkq;
          }
          const vkp = jacobiV[k * actualM + p];
          const vkq = jacobiV[k * actualM + q];
          jacobiV[k * actualM + p] = c * vkp - s * vkq;
          jacobiV[k * actualM + q] = s * vkp + c * vkq;
        }
      }
    }
    if (maxOff < 1e-11) break;
  }

  let bestCol = 0;
  let thetaMax = -1e30;
  let thetaSecond = -1e30;
  for (let c = 0; c < actualM; c++) {
    const val = jacobiA[c * actualM + c];
    if (val > thetaMax) {
      thetaSecond = thetaMax;
      thetaMax = val;
      bestCol = c;
    } else if (val > thetaSecond) {
      thetaSecond = val;
    }
  }

  const lambda2 = Math.max(0.0, mu - thetaMax);
  const lambda3 = actualM >= 2 ? Math.max(lambda2, mu - thetaSecond) : lambda2;
  const spectralGap = Math.max(0.0, lambda3 - lambda2);

  let dotPrev = 0.0;
  for (let a = 0; a < n; a++) {
    let comp = 0.0;
    for (let j = 0; j < actualM; j++) {
      comp += lanczosQ[j * GRAPH_NODES + a] * jacobiV[j * actualM + bestCol];
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

    if (apexFrames >= 0.35) {
      if (apexLockTid[m] !== tid || apexLockHold[m] <= 0 || apexLockErrPx[m] >= 0.0) {
        // Lock new ballistic throw prediction while ascending!
        apexLockTid[m] = tid;
        apexLockPredX[m] = apexX;
        apexLockPredY[m] = apexY;
        apexLockMinY[m] = curY;
        apexLockAtMinX[m] = curX;
        apexLockErrPx[m] = -1.0;
        apexLockHold[m] = Math.max(16, lookaheadK + 12);
      } else if (apexFrames >= 1.15) {
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
      // Score peak accuracy once ball turns downward after reaching peak!
      if (apexLockErrPx[m] < 0.0 && apexFrames <= 0.05 && curY > apexLockMinY[m] + 2.5 * resScale) {
        const err = Math.hypot(
          (apexLockAtMinX[m] - apexLockPredX[m]) * 0.5,
          apexLockMinY[m] - apexLockPredY[m]
        );
        apexLockErrPx[m] = err;
        lastScoredApexErrPx = err;
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

  // Interactive click-capture crosshairs & guidance banner when armed
  if (calArmed !== 'none') {
    for (let i = 0; i < calPtCount; i++) {
      const px = calPtsNormX[i] * W;
      const py = calPtsNormY[i] * H;
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
        hudCtx.moveTo(calPtsNormX[i - 1] * W, calPtsNormY[i - 1] * H);
        hudCtx.lineTo(px, py);
        hudCtx.stroke();
        hudCtx.setLineDash([]);
      }
    }
    const prompt = calArmed === 'ruler'
      ? `📏 CLICK RULER ENDPOINT ${calPtCount + 1}/2 (${calRulerMeters.toFixed(2)} m · ESC TO CANCEL)`
      : `⬠ CLICK PLANE CORNER ${calPtCount + 1}/4 [${CORNER_TAGS[Math.min(3, calPtCount)]}] (${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m · ESC CANCEL)`;
    const bw = 472;
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
          const cadTxt = `⚡ Δθ=${signStr}${dispDefl.toFixed(1)}° | e=${dispRest.toFixed(2)}`;
          const bw = 152;
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
    const rawScDeg = telemView[642];
    const scEventFrame = Math.round(telemView[647]);
    if (rawScDeg > 2.0 && scEventFrame > 0 && Math.abs(dispFrame - scEventFrame) <= 24) {
      const scx = telemView[643] * sx;
      const scy = telemView[644] * sy;
      const th1Canvas = -telemView[645] * (Math.PI / 180.0);
      const th2Canvas = -telemView[646] * (Math.PI / 180.0);
      const rectSc = rectifyScatterKinematics(
        telemView[643],
        telemView[644],
        telemView[645],
        telemView[646],
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

      const scTxt = `⟁ SCATTER ${dispScDeg.toFixed(1)}°`;
      const bw = 126;
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
  const effFps = Math.max(8.0, sourceMode === 'testbench' ? smoothFps : sourceFps);
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  const primarySlot = Math.round(telemView[8]);

  updateFrozenApexLocks(uDisp);

  // 5. Render Target Trajectories, Hermite Spline Bridges, Frozen Apex Reticles, Protractors & Kalman Baselines
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;

    const kbase = 576 + m * 8;
    const tid = Math.round(telemView[base + 1]);
    const isPrimary = m === primarySlot;
    const isPinned = focusTid !== 0 && tid === focusTid;
    const color = TARGET_COLORS[(tid - 1) % TARGET_COLORS.length];
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

    hudCtx.save();

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
        const thSign = dispHeading >= 0 ? '+' : '';
        const spdFmt = metricOn ? dispSpd.toFixed(2) : Math.round(dispSpd).toString();
        const kinTxt = `∠${thSign}${dispHeading.toFixed(1)}° · ${spdFmt}${spdUnit}`;
        const bw = 148;
        const pos = reserveHudLabelSlot(W, H, fx + 30, fy + 12, bw, placedLabels);
        placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
        hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
        hudCtx.fillRect(pos[0], pos[1], bw, 17);
        hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.58)';
        hudCtx.lineWidth = 1.0;
        hudCtx.strokeRect(pos[0], pos[1], bw, 17);
        hudCtx.fillStyle = '#8ef9f3';
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

    // 5e. Kinematic-Only 2x2 Kalman Baseline Marker
    const kalDist = Math.hypot(kx - fx, ky - fy);
    if (showKalmanBaseline && kalDist <= 165.0) {
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
      if (kalDist > 20.0) {
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
    const boxR = isPrimary ? 26 : 23;
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
    hudCtx.restore();
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

  // 7. Top-left HUD Status Strip + Top-right Focus Hint + Bottom Calibration / Gravity Strip
  const clutterReady = telemView[22] > 0.5;
  const clutterStr = clutterReady ? `${telemView[4].toFixed(1)} dB` : 'WARMING';
  const ecaTag = showEcaOverlay ? ' | LIVE ECA@t' : '';
  const shownSwaps = sourceMode === 'testbench' ? oraclePadicSwaps : Math.round(telemView[5]);

  hudCtx.fillStyle = 'rgba(6, 10, 18, 0.82)';
  hudCtx.fillRect(10, 10, 566, 24);
  hudCtx.fillStyle = '#8ef9f3';
  hudCtx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
  hudCtx.fillText(
    `VIEW [t - ${effK}] ${frameW}x${frameH} | ECA ${clutterStr} | ID SWITCHES: ${shownSwaps} | KALMAN: ${Math.round(telemView[6])}${ecaTag}`,
    18,
    26
  );

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

  // Bottom-left Scene Gravity Prior badge when parabolic gravity is locked
  const sceneGPx = telemView[640];
  const sceneGConf = telemView[641];
  if (labMode === 0 && sceneGConf >= 0.25 && sceneGPx > 50.0) {
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.84)';
    hudCtx.fillRect(10, H - 32, 232, 22);
    hudCtx.strokeStyle = 'rgba(255, 209, 102, 0.48)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(10, H - 32, 232, 22);
    hudCtx.fillStyle = '#ffd166';
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(`ĝ = ${Math.round(sceneGPx)} px/s² (CONF ${Math.round(sceneGConf * 100)}%)`, 17, H - 17);
  }

  // Bottom-right Metric / Homography Calibration badge
  if (calMode !== 'none') {
    const calShort = calMode === 'homography' && calHomographyValid
      ? `⬠ PLANE H: ${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m [RECTIFIED m/s]`
      : calMode === 'auto_g'
      ? `⚡ AUTO-g: ${Math.round(getEffectivePxPerMeter())} px/m (9.81 m/s²)`
      : `📏 RULER: ${calRulerMeters.toFixed(2)} m (${Math.round(getEffectivePxPerMeter())} px/m)`;
    const cw = 304;
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
    hudCtx.fillRect(W - cw - 10, H - 32, cw, 22);
    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.55)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(W - cw - 10, H - 32, cw, 22);
    hudCtx.fillStyle = '#00e5ff';
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
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

  const pSlot = Math.round(telemView[8]);
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
  const cohPct = pSlot >= 0 ? Math.round(telemView[512 + pSlot * 8 + 6] * 100) : 0;
  hilbertCtx.fillStyle = '#9db4d4';
  hilbertCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const phaseValid = telemView[26] > 0.5;
  if (phaseValid) {
    const phiDeg = (telemView[11] * 180 / Math.PI).toFixed(0);
    hilbertCtx.fillText(
      `HILBERT φ=${phiDeg}° | f=${telemView[13].toFixed(1)}Hz | COH ${cohPct}% | R CYAN · H{R} MAGENTA`,
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

function processIngestedFrame(rawPixels, dtSec) {
  const safeDt = Math.min(0.1, Math.max(0.002, dtSec || 1.0 / 60.0));
  const instSourceFps = 1.0 / safeDt;
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

  // Measure full production compute path: Wasm Pre-Tick + Shifted Lanczos Fiedler + Wasm Post-Tick
  const t0 = performance.now();
  wasmExports.chronos_pre_tick(safeDt);
  const lambda2 = solveShiftedLanczosFiedler();
  const dispSlot = wasmExports.chronos_post_tick(safeDt);
  const wasmMs = performance.now() - t0;

  wasmTimeHist[timeHistIdx] = wasmMs;
  timeHistIdx = (timeHistIdx + 1) % TIME_HIST_LEN;

  if (sourceMode === 'testbench') {
    auditOracleSwaps();
  }

  let frameHasBridge = false;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] > 0.5 && telemView[b + 11] > 0.5) {
      frameHasBridge = true;
      break;
    }
  }
  if (frameHasBridge) totalBridgedEvents++;

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

  // Update resolution-invariant scene gravity lock when high-confidence parabolic arc is observed
  if (telemView[641] >= 0.25 && telemView[640] > 50.0) {
    const hadLock = calAutoGNormH > 0.05;
    calAutoGNormH = telemView[640] / frameH;
    if (calMode === 'auto_g' && !hadLock) {
      emitCalibrationState();
    }
  }

  // Emit per-target deduplicated Mode C strikes and schedule synchronized t - K shockwave rings!
  const percMask = Math.round(telemView[15]);
  if (percMask !== 0) {
    const fIdx = Math.round(telemView[0]);
    const effSampleRate = Math.max(8.0, sourceMode === 'testbench' ? smoothFps : sourceFps);
    const nowEpochMs = performance.timeOrigin + performance.now();
    for (let m = 0; m < MAX_TARGETS; m++) {
      const ebase = 512 + m * 8;
      if (telemView[ebase] > 0.5) {
        totalStrikeEvents++;
        const kbase = 576 + m * 8;
        const zone = Math.round(telemView[ebase + 1]) & 3;
        const leadFrames = telemView[ebase + 2];
        const vel = telemView[ebase + 3];
        const sxPos = telemView[ebase + 4];
        const syPos = telemView[ebase + 5];
        const rawDefl = telemView[kbase + 3];
        const rawRest = telemView[kbase + 4];
        const rawIn = telemView[kbase + 5];

        const maxValidRest = sourceMode === 'testbench' ? 1.65 : 1.18;
        if (rawRest >= 0.06 && rawRest <= maxValidRest && Math.abs(rawDefl) >= 3.0) {
          const rectSt = rectifyStrikeKinematics(sxPos, syPos, rawIn, rawDefl, rawRest);
          if (rectSt[2] <= maxValidRest) {
            lastStrikeDeflectDeg = rectSt[1];
            lastStrikeRestitution = rectSt[2];
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

  if (telemView[642] > 2.0) {
    const rectSc = rectifyScatterKinematics(
      telemView[643],
      telemView[644],
      telemView[645],
      telemView[646],
      telemView[642]
    );
    lastScatterDeg = rectSc[2];
  }

  // Render all 4 OffscreenCanvases inside Worker isolate
  renderMainHud(dispSlot);
  renderEulerianPip();
  renderHilbertScope();
  renderSpectralAndRingScope();

  // Zero-allocation per-frame scientific telemetry recording (1:1 frame fidelity, 46 columns)
  if (recActive) {
    const effRateRec = Math.max(8.0, sourceMode === 'testbench' ? smoothFps : sourceFps);
    const pSlot = Math.round(telemView[8]);
    const ro = recHead * REC_COLS;
    recElapsedSec += safeDt;

    recSlab[ro + 0] = Math.round(telemView[0]);
    recSlab[ro + 1] = recElapsedSec;
    recSlab[ro + 2] = frameW;
    recSlab[ro + 3] = frameH;
    recSlab[ro + 4] = Math.round(telemView[1]);
    recSlab[ro + 5] = labMode;
    recSlab[ro + 6] = wasmMs;
    recSlab[ro + 7] = Math.round(telemView[3]);
    recSlab[ro + 8] = telemView[4];
    recSlab[ro + 9] = telemView[23];
    recSlab[ro + 10] = Math.round(telemView[5]);
    recSlab[ro + 11] = oraclePadicSwaps;
    recSlab[ro + 12] = Math.round(telemView[6]);
    recSlab[ro + 13] = inPillarCollisions;
    recSlab[ro + 14] = lambda2;
    recSlab[ro + 15] = telemView[24];
    recSlab[ro + 16] = telemView[25];

    if (pSlot >= 0 && telemView[32 + pSlot * 56] > 0.5) {
      const b = 32 + pSlot * 56;
      const kb = 576 + pSlot * 8;
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
      recSlab[ro + 40] = telemView[kb + 2];
      if (telemView[kb + 6] > -9.0e5 && telemView[kb + 7] > -9.0e5) {
        const pend = computePendulumKinematics(fxPx, fyPx, telemView[kb + 6], telemView[kb + 7]);
        recSlab[ro + 41] = pend[0];
      } else {
        recSlab[ro + 41] = 0.0;
      }
      const stRect = rectifyStrikeKinematics(fxPx, fyPx, telemView[kb + 5], telemView[kb + 3], telemView[kb + 4]);
      recSlab[ro + 42] = stRect[1];
      recSlab[ro + 43] = stRect[2];
      const mPt = mapPixelToMetric(fxPx, fyPx);
      recSlab[ro + 44] = mPt[0];
      recSlab[ro + 45] = mPt[1];
    } else {
      for (let c = 17; c <= 27; c++) recSlab[ro + c] = 0.0;
      recSlab[ro + 36] = 0.0;
      for (let c = 38; c <= 45; c++) recSlab[ro + c] = 0.0;
    }

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
  }

  tickCounter++;
  if (tickCounter % 3 === 0 || sourceMode !== 'testbench') {
    let sumMs = 0.0;
    let maxMs = 0.0;
    const count = Math.min(tickCounter, TIME_HIST_LEN);
    for (let i = 0; i < count; i++) {
      sumMs += wasmTimeHist[i];
      if (wasmTimeHist[i] > maxMs) maxMs = wasmTimeHist[i];
    }
    const meanMs = sumMs / Math.max(1, count);
    const medianMs = computeMedianTickMs(count);
    const effSampleRate = Math.max(8.0, sourceMode === 'testbench' ? smoothFps : sourceFps);

    const pIdx = Math.round(telemView[8]);
    const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
    const metricOn = isMetricCalibrated();
    let pTid = 0;
    let pSigHex = '0x00000000';
    let pVal = 0;
    let apexMs = 0;
    let primaryCoherence = 0;
    let headingDeg = 0.0;
    let speedPhys = 0.0;
    let angVelDegS = 0.0;
    let pivotValid = false;
    let pendSwingDeg = 0.0;
    let pendRadiusPhys = 0.0;
    if (pIdx >= 0 && telemView[32 + pIdx * 56] > 0.5) {
      const b = 32 + pIdx * 56;
      const kb = 576 + pIdx * 8;
      const fxPx = telemView[b + 2];
      const fyPx = telemView[b + 3];
      pTid = Math.round(telemView[b + 1]);
      pSigHex = toHex8(telemView[b + 12], telemView[b + 13]);
      pVal = Math.round(telemView[b + 14]);
      apexMs = telemView[b + 17] * (1000.0 / effSampleRate);
      primaryCoherence = telemView[512 + pIdx * 8 + 6];
      const physVel = mapVelocityToPhysical(fxPx, fyPx, telemView[kb + 1], telemView[kb + 0]);
      speedPhys = physVel[0];
      headingDeg = physVel[1];
      angVelDegS = telemView[kb + 2];
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

    self.postMessage({
      type: 'telemetry',
      frameIdx: Math.round(telemView[0]),
      lookaheadK: Math.round(telemView[1]),
      fps: effSampleRate,
      wasmMeanMs: meanMs,
      wasmMedianMs: medianMs,
      wasmMaxMs: maxMs,
      activeTargets: Math.round(telemView[3]),
      clutterDb: telemView[4],
      clutterDbValid: telemView[22] > 0.5,
      clutterEvalRatio: telemView[23],
      padicSwaps: Math.round(telemView[5]),
      oracleSwaps: oraclePadicSwaps,
      sourceMode,
      benchScenario,
      kalmanSwaps: Math.round(telemView[6]),
      inPillarCollisions,
      bridgedFrames: totalBridgedEvents,
      strikeEvents: totalStrikeEvents,
      lambda2,
      spectralGap: telemView[24],
      fistaDualityGap: telemView[25],
      phaseValid: telemView[26] > 0.5,
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
      speedUnit: metricOn ? 'm/s' : 'px/s',
      distUnit: metricOn ? 'm' : 'px',
      angVelDegS,
      pivotValid,
      pendSwingDeg,
      pendRadiusPhys,
      sceneGravityPxS2: telemView[640],
      sceneGravityConf: telemView[641],
      lastDeflectDeg: lastStrikeDeflectDeg,
      lastRestitution: lastStrikeRestitution,
      lastStrikeValid,
      scatterDeg: lastScatterDeg,
      calMode,
      calArmed,
      calibrated: metricOn,
      arenaBytes: Math.round(telemView[20]),
      memoryPages: wasmMemory.buffer.byteLength / 65536,
      resMode,
      frameW,
      frameH,
      pipDim,
      recActive,
      recFrames: recCount,
      recDurationSec: recElapsedSec,
    });
  }
}

function configureResolutionViews(mode) {
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

  wasmTimeHist.fill(0.0);
  timeHistIdx = 0;
  tickCounter = 0;
  resetTestbenchState();
}

function redrawPausedHudIfReady() {
  if (wasmExports && telemView && Math.round(telemView[0]) > 0) {
    const dispSlot = Math.round(telemView[2]);
    renderMainHud(dispSlot);
    renderEulerianPip();
  }
}

function handleHudCanvasClick(normX, normY) {
  if (!wasmExports || !telemView) return;

  // Intercept interactive 2-Pt Ruler & 4-Pt Plane Homography calibration clicks first
  if (calArmed === 'ruler') {
    calPtsNormX[calPtCount] = normX;
    calPtsNormY[calPtCount] = normY;
    calPtCount++;
    if (calPtCount >= 2) {
      const spanPx = Math.hypot(
        (calPtsNormX[1] - calPtsNormX[0]) * frameW,
        (calPtsNormY[1] - calPtsNormY[0]) * frameH
      );
      if (spanPx < 4.0) {
        calPtCount = 0;
        calArmed = 'none';
        emitCalibrationState('RULER CANCELLED · ENDPOINTS TOO CLOSE (< 4 px)');
      } else {
        calMode = 'ruler';
        calArmed = 'none';
        calPresetName = '';
        emitCalibrationState();
      }
    } else {
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
    return;
  }

  if (calArmed === 'homography') {
    calPtsNormX[calPtCount] = normX;
    calPtsNormY[calPtCount] = normY;
    calPtCount++;
    if (calPtCount >= 4) {
      calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
      if (calHomographyValid) {
        calMode = 'homography';
        calArmed = 'none';
        calPresetName = '';
        emitCalibrationState();
      } else {
        calMode = 'none';
        calArmed = 'none';
        calPtCount = 0;
        emitCalibrationState('INVALID QUAD · CLICK CONVEX CORNERS TL → TR → BR → BL');
      }
    } else {
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
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

  // Re-render canvases immediately if video is paused
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
  const dispSlot = Math.round(telemView[2]);
  renderMainHud(dispSlot);
  renderEulerianPip();
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

  if (wasmExports) {
    const simDt = 1.0 / 60.0;
    const rawPixels = stepAndRenderBlindPillarScene(simDt);
    processIngestedFrame(rawPixels, simDt);
  }

  scheduleNextWorkerTick();
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

    hudCtx = hudCanvas.getContext('2d', { alpha: false });
    pipCtx = pipCanvas.getContext('2d', { alpha: false });
    hilbertCtx = hilbertCanvas.getContext('2d', { alpha: false });
    spectralCtx = spectralCanvas.getContext('2d', { alpha: false });

    dispStageCanvas = new OffscreenCanvas(frameW, frameH);
    dispStageCtx = dispStageCanvas.getContext('2d');
    pipStageCanvas = new OffscreenCanvas(pipDim, pipDim);
    pipStageCtx = pipStageCanvas.getContext('2d');
    simCanvas = new OffscreenCanvas(frameW, frameH);
    simCtx = simCanvas.getContext('2d', { willReadFrequently: true });

    ecaStageCanvas = new OffscreenCanvas(frameW, frameH);
    ecaStageCtx = ecaStageCanvas.getContext('2d');

    const resp = await fetch(msg.wasmUrl || './chronos_spectral.wasm');
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
    running = true;
    self.postMessage({
      type: 'ready',
      memoryBytes: wasmMemory.buffer.byteLength,
      pages: wasmMemory.buffer.byteLength / 65536,
      resMode,
      frameW,
      frameH,
      pipDim,
    });
    scheduleNextWorkerTick();
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
    benchScenario = msg.scenario || 'pillar_crossing';
    wasmTimeHist.fill(0.0);
    timeHistIdx = 0;
    tickCounter = 0;
    if (wasmExports) wasmExports.chronos_reset_tracks();
    resetTestbenchState();
  } else if (msg.type === 'cal_control') {
    if (typeof msg.rulerMeters === 'number' && msg.rulerMeters >= 0.01) {
      calRulerMeters = msg.rulerMeters;
    }
    if (typeof msg.planeW === 'number' && msg.planeW >= 0.01) {
      calPlaneW = msg.planeW;
    }
    if (typeof msg.planeH === 'number' && msg.planeH >= 0.01) {
      calPlaneH = msg.planeH;
    }

    if (msg.action === 'arm_ruler') {
      if (calArmed === 'ruler') {
        calArmed = 'none';
        calPtCount = calMode === 'ruler' ? 2 : (calMode === 'homography' ? 4 : 0);
      } else {
        calArmed = 'ruler';
        calPtCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'arm_homography') {
      if (calArmed === 'homography') {
        calArmed = 'none';
        calPtCount = calMode === 'homography' ? 4 : (calMode === 'ruler' ? 2 : 0);
      } else {
        calArmed = 'homography';
        calPtCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'cancel_arm') {
      if (calArmed !== 'none') {
        calArmed = 'none';
        calPtCount = calMode === 'homography' ? 4 : (calMode === 'ruler' ? 2 : 0);
        emitCalibrationState();
      }
    } else if (msg.action === 'set_auto_g') {
      calArmed = 'none';
      calPtCount = 0;
      calPresetName = '';
      calHomographyValid = false;
      calMode = 'auto_g';
      calAutoGFallbackNormW = 0.0;
      if (telemView && telemView[641] >= 0.25 && telemView[640] > 50.0) {
        calAutoGNormH = telemView[640] / frameH;
      }
      emitCalibrationState();
    } else if (msg.action === 'apply_preset') {
      applySceneCalibrationPreset(msg.preset || 'testbench');
    } else if (msg.action === 'update_dims') {
      if (calMode === 'homography' && calPtCount >= 4) {
        calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
      }
      emitCalibrationState();
    } else if (msg.action === 'clear') {
      clearCalibrationState();
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
  } else if (msg.type === 'canvas_click') {
    if (typeof msg.normX === 'number' && typeof msg.normY === 'number') {
      handleHudCanvasClick(msg.normX, msg.normY);
    }
  } else if (msg.type === 'cycle_focus') {
    cycleFocusTarget();
  } else if (msg.type === 'params') {
    if (typeof msg.lookaheadK === 'number') lookaheadK = msg.lookaheadK;
    if (typeof msg.ecaAdaptRate === 'number') ecaAdaptRate = msg.ecaAdaptRate;
    if (typeof msg.fistaLambda === 'number') fistaLambda = msg.fistaLambda;
    if (typeof msg.eulerianGain === 'number') eulerianGain = msg.eulerianGain;
    if (typeof msg.phaseOffsetRad === 'number') phaseOffsetRad = msg.phaseOffsetRad;
    if (typeof msg.labMode === 'number') labMode = msg.labMode;
    if (typeof msg.showEcaOverlay === 'boolean') showEcaOverlay = msg.showEcaOverlay;
    if (typeof msg.showKalmanBaseline === 'boolean') showKalmanBaseline = msg.showKalmanBaseline;
    if (typeof msg.strobeEnabled === 'boolean') strobeEnabled = msg.strobeEnabled;
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
      if (sourceMode !== 'testbench' && telemView && Math.round(telemView[0]) > 0) {
        const dispSlot = Math.round(telemView[2]);
        renderMainHud(dispSlot);
        renderEulerianPip();
        renderHilbertScope();
        renderSpectralAndRingScope();
      }
    }
  } else if (msg.type === 'set_source') {
    sourceMode = msg.sourceMode;
    if (typeof msg.fpsHint === 'number' && msg.fpsHint >= 8.0) {
      sourceFps = msg.fpsHint;
    } else if (sourceMode !== 'testbench') {
      sourceFps = 30.0;
    }
    wasmTimeHist.fill(0.0);
    timeHistIdx = 0;
    tickCounter = 0;
    lastRafTs = 0;
    clearCalibrationState();
    emitCalibrationState();
    if (wasmExports) {
      wasmExports.chronos_reset_tracks();
    }
    resetTestbenchState();
    if (sourceMode === 'testbench') {
      scheduleNextWorkerTick();
    } else {
      cancelWorkerTickLoop();
    }
  } else if (msg.type === 'video_frame') {
    const bmp = msg.bitmap;
    if (wasmExports && sourceMode !== 'testbench') {
      simCtx.drawImage(bmp, 0, 0, frameW, frameH);
      const rawPixels = simCtx.getImageData(0, 0, frameW, frameH).data;
      processIngestedFrame(rawPixels, msg.dtSec || 1.0 / 30.0);
    }
    bmp.close();
  } else if (msg.type === 'video_rgba') {
    if (wasmExports && sourceMode !== 'testbench' && msg.buffer) {
      const rawPixels = new Uint8Array(msg.buffer);
      if (rawPixels.byteLength === frameBytes) {
        processIngestedFrame(rawPixels, msg.dtSec || 1.0 / 30.0);
      }
    }
  } else if (msg.type === 'lab_action') {
    if (msg.action === 'saccade_step') {
      if (wasmExports) wasmExports.chronos_trigger_stimulus();
      if (sourceMode === 'testbench') {
        pendingSaccadeFrames = 9;
      }
    } else if (msg.action === 'ballistic_toss' && sourceMode === 'testbench') {
      simDiscs[0].x = 24 * resScale;
      simDiscs[0].y = 148 * resScale;
      simDiscs[0].vx = 3.8 * resScale;
      simDiscs[0].vy = -6.8 * resScale;
      simDiscs[0].ballistic = true;
    } else if (msg.action === 'air_strike' && sourceMode === 'testbench') {
      for (let i = 0; i < simDiscs.length; i++) {
        simDiscs[i].vx = -simDiscs[i].vx * 1.45;
      }
      pendingAirSnapFrames = 7;
    } else if (msg.action === 'reset') {
      if (wasmExports) wasmExports.chronos_reset_tracks();
      resetTestbenchState();
    }
  } else if (msg.type === 'rec_control') {
    if (msg.action === 'start') {
      if (msg.clear) {
        recHead = 0;
        recCount = 0;
        recElapsedSec = 0.0;
      }
      recActive = true;
    } else if (msg.action === 'stop') {
      recActive = false;
    } else if (msg.action === 'clear') {
      recActive = false;
      recHead = 0;
      recCount = 0;
      recElapsedSec = 0.0;
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
    self.postMessage(
      {
        type: 'telemetry_export_data',
        format: msg.format || 'csv',
        rows,
        cols: REC_COLS,
        columns: REC_COLUMNS,
        buffer: ordered.buffer,
        meta: {
          schemaVersion: '1.3.0',
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
        },
      },
      [ordered.buffer]
    );
  }
};
