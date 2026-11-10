// ============================================================================
// CHRONOS-SPECTRAL Dedicated V8 Web Worker Isolate
// Manages Zero-Allocation Wasm SIMD128 Arena Views, Pre-Assignment Shifted
// Lanczos Fiedler Eigensolver, Blind Pixel Pillar Testbench, and OffscreenCanvas HUD.
// ============================================================================

const FRAME_W = 320;
const FRAME_H = 180;
const FRAME_PIXELS = FRAME_W * FRAME_H;
const FRAME_BYTES = FRAME_PIXELS * 4;
const RING_LEN = 12;
const MAX_TARGETS = 8;
const GRAPH_NODES = 16;
const TELEMETRY_LEN = 576;

let wasmExports = null;
let wasmMemory = null;

// Pre-allocated permanent views into 64-page (4 MB) static Wasm linear memory
const ringSlotU8 = new Array(RING_LEN);
const ringSlotImgData = new Array(RING_LEN);
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

// Internal 320x180 staging & simulation OffscreenCanvases
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
let showEcaOverlay = false;
let showKalmanBaseline = true;
let strobeEnabled = true;

let lookaheadK = 8;
let ecaAdaptRate = 0.035;
let fistaLambda = 1.45;
let eulerianGain = 25.0;
let phaseOffsetRad = -Math.PI * 0.5;
let labMode = 0; // 0: Mode A (Ballistic/Sleight), 1: Mode B (Tremor/Saccade), 2: Mode C (Air-Percussion)
let audioOutLatencySec = 0.0;

// Timing & Rolling Benchmark Telemetry (pre-allocated ring buffers)
const TIME_HIST_LEN = 120;
const wasmTimeHist = new Float32Array(TIME_HIST_LEN);
let timeHistIdx = 0;
let lastRafTs = 0;
let smoothFps = 60.0;
let sourceFps = 60.0;
let tickCounter = 0;
let strobeFlashAlpha = 0.0;
const padFlashAlpha = new Float32Array([0, 0, 0, 0]);

// Pre-allocated delayed Mode C pad-flash schedule (synchronizes HUD flash with t - K presentation)
const FLASH_QUEUE_LEN = 32;
const flashQueueFrame = new Int32Array(FLASH_QUEUE_LEN).fill(-1);
const flashQueueZone = new Uint8Array(FLASH_QUEUE_LEN);
let flashQueueHead = 0;

// Hilbert & Spectral Scope circular histories (pre-allocated)
const SCOPE_HIST = 128;
const histTremorR = new Float32Array(SCOPE_HIST);
const histTremorH = new Float32Array(SCOPE_HIST);
const histStrobe = new Uint8Array(SCOPE_HIST);
const histLambda2 = new Float32Array(SCOPE_HIST);
let scopeWriteIdx = 0;

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

// ============================================================================
// Blind Pixel Pillar Testbench Physics & 1-to-1 Ground-Truth Swap Oracle
// ============================================================================

const PILLARS = [
  { x0: 88, x1: 136, y0: 14, y1: 166 },
  { x0: 184, x1: 232, y0: 14, y1: 166 },
];

const simDiscs = [
  { id: 0, x: 52,  y: 56,  baseY: 56,  vx: 3.0,  vy: 0.0, ballistic: false, r: '#f83a44', core: '#ffe66e' },
  { id: 1, x: 172, y: 56,  baseY: 56,  vx: -3.0, vy: 0.0, ballistic: false, r: '#1ee1fa', core: '#0f2d55' },
  { id: 2, x: 148, y: 124, baseY: 124, vx: 3.0,  vy: 0.0, ballistic: false, r: '#32f270', core: '#ebffeb' },
  { id: 3, x: 268, y: 124, baseY: 124, vx: -3.0, vy: 0.0, ballistic: false, r: '#e840f8', core: '#5af5ff' },
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

function resetTestbenchState() {
  simDiscs[0].x = 52;  simDiscs[0].y = 56;  simDiscs[0].baseY = 56;  simDiscs[0].vx = 3.0;  simDiscs[0].vy = 0; simDiscs[0].ballistic = false;
  simDiscs[1].x = 172; simDiscs[1].y = 56;  simDiscs[1].baseY = 56;  simDiscs[1].vx = -3.0; simDiscs[1].vy = 0; simDiscs[1].ballistic = false;
  simDiscs[2].x = 148; simDiscs[2].y = 124; simDiscs[2].baseY = 124; simDiscs[2].vx = 3.0;  simDiscs[2].vy = 0; simDiscs[2].ballistic = false;
  simDiscs[3].x = 268; simDiscs[3].y = 124; simDiscs[3].baseY = 124; simDiscs[3].vx = -3.0; simDiscs[3].vy = 0; simDiscs[3].ballistic = false;
  oracleTrackOwner.fill(-1);
  prevFiedlerTrack.fill(0.0);
  flashQueueFrame.fill(-1);
  padFlashAlpha.fill(0.0);
  strobeFlashAlpha = 0.0;
  oraclePadicSwaps = 0;
  inPillarCollisions = 0;
  simTimeSec = 0.0;
  pendingSaccadeFrames = 0;
  pendingAirSnapFrames = 0;
}

function stepAndRenderBlindPillarScene(dtSec) {
  const ctx = simCtx;
  const stepScale = Math.min(2.0, Math.max(0.25, dtSec * 60.0));
  simTimeSec += dtSec;

  // Deterministic Mode B / Mode C testbench stimuli (zero setTimeout jitter)
  if (pendingSaccadeFrames > 0) {
    pendingSaccadeFrames--;
    if (pendingSaccadeFrames === 0) {
      const d0 = simDiscs[0];
      d0.x = d0.x < 160 ? Math.min(84, d0.x + 30) : Math.max(145, d0.x - 30);
    }
  }
  if (pendingAirSnapFrames > 0) {
    pendingAirSnapFrames--;
    if (pendingAirSnapFrames === 0) {
      for (let i = 0; i < simDiscs.length; i++) {
        simDiscs[i].vx = Math.sign(simDiscs[i].vx || 1) * 3.0;
      }
    }
  }

  // 1. Static structured aerospace grid clutter (tests uncensored >60 dB ECA cancellation)
  ctx.fillStyle = '#121722';
  ctx.fillRect(0, 0, FRAME_W, FRAME_H);

  ctx.strokeStyle = '#1d2536';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 24; x < FRAME_W; x += 24) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, FRAME_H);
  }
  for (let y = 24; y < FRAME_H; y += 24) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(FRAME_W, y + 0.5);
  }
  ctx.stroke();

  // 2. Advance disc kinematics & collisions (frame-rate independent for 60Hz/120Hz ProMotion)
  const tSec = simTimeSec;
  for (let i = 0; i < simDiscs.length; i++) {
    const d = simDiscs[i];
    d.x += d.vx * stepScale;

    if (d.ballistic) {
      // Physical downward gravity = 792 px/s^2 (0.22 px/frame^2 at 60Hz)
      const vyPrev = d.vy;
      d.vy += 0.22 * stepScale;
      d.y += 0.5 * (vyPrev + d.vy) * stepScale;
      if (d.y >= 152) {
        d.y = 152;
        d.vy = -Math.abs(d.vy) * 0.86;
        if (Math.abs(d.vy) < 1.2) {
          d.ballistic = false;
          d.vy = 0;
          d.y = d.baseY;
        }
      }
      if (d.y < 16) {
        d.y = 16;
        d.vy = Math.abs(d.vy) * 0.5;
      }
    } else {
      const tremor = 0.95 * Math.sin(2.0 * Math.PI * 9.5 * tSec + i * 1.3);
      d.y = d.baseY + tremor;
    }

    if (d.x < 18) {
      d.x = 18;
      d.vx = Math.abs(d.vx);
    } else if (d.x > FRAME_W - 18) {
      d.x = FRAME_W - 18;
      d.vx = -Math.abs(d.vx);
    }
  }

  // Pairwise elastic collisions on upper lane (0,1) and lower lane (2,3)
  for (let lane = 0; lane < 2; lane++) {
    const a = simDiscs[lane * 2];
    const b = simDiscs[lane * 2 + 1];
    if (a.ballistic || b.ballistic) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    if (Math.abs(dx) < 14.5 && Math.abs(dy) < 14.5 && (a.vx - b.vx) * dx > 0) {
      const tmp = a.vx;
      a.vx = b.vx;
      b.vx = tmp;
      const midX = 0.5 * (a.x + b.x);
      if (
        (midX >= PILLARS[0].x0 && midX <= PILLARS[0].x1) ||
        (midX >= PILLARS[1].x0 && midX <= PILLARS[1].x1)
      ) {
        inPillarCollisions++;
      }
    }
  }

  // 3. Draw physical textured discs
  for (let i = 0; i < simDiscs.length; i++) {
    const d = simDiscs[i];
    ctx.fillStyle = d.r;
    ctx.beginPath();
    ctx.arc(d.x, d.y, 7.6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = d.core;
    ctx.beginPath();
    ctx.arc(d.x, d.y, 3.4, 0, Math.PI * 2);
    ctx.fill();
  }

  // 4. Draw 100% OPAQUE occlusion pillars ON TOP (wipes out underlying disc pixels completely)
  for (let p = 0; p < PILLARS.length; p++) {
    const pil = PILLARS[p];
    const pw = pil.x1 - pil.x0;
    const ph = pil.y1 - pil.y0;
    ctx.fillStyle = '#222a3a';
    ctx.fillRect(pil.x0, pil.y0, pw, ph);
    ctx.strokeStyle = '#36435c';
    ctx.lineWidth = 1;
    ctx.strokeRect(pil.x0 + 0.5, pil.y0 + 0.5, pw - 1, ph - 1);
  }

  return ctx.getImageData(0, 0, FRAME_W, FRAME_H).data;
}

/// Strict 1-to-1 Nearest-Match Ground-Truth Identity Oracle (no & 255 wrap, no multi-disc aliasing).
function auditOracleSwaps() {
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
      const occluded =
        (d.x > PILLARS[0].x0 - 6 && d.x < PILLARS[0].x1 + 6) ||
        (d.x > PILLARS[1].x0 - 6 && d.x < PILLARS[1].x1 + 6);
      if (occluded) continue;

      const dist = Math.hypot(ix - d.x, iy - d.y);
      if (dist < 16.0) {
        oracleCandTrackSlot[nPairs] = m;
        oracleCandDiscIdx[nPairs] = i;
        oracleCandDist[nPairs] = dist;
        nPairs++;
      }
    }
  }

  // Sort candidate (track, disc) pairs ascending by Euclidean distance
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
    // Two-pass Modified Gram-Schmidt against 1_n and q_0 .. q_{numExisting - 1}
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

  // 1. Gershgorin bound mu > lambda_max(L)
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

  // 2. Initial seed orthogonal to 1_n (completed via orthogonal restarts if symmetry-blind)
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

    // Matrix-vector multiply w = (mu * I - L) q_j
    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a];
      let sumL = 0.0;
      for (let b = 0; b < n; b++) {
        sumL += laplacianView[ia * GRAPH_NODES + activeNodeIdx[b]] * lanczosQ[qOff + b];
      }
      lanczosW[a] = mu * lanczosQ[qOff + a] - sumL;
    }

    // Deflate nullspace vector 1_n
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

    // Two-pass Modified Gram-Schmidt re-orthogonalization against q_0..q_j and 1_n
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
      // Invariant Krylov subspace reached early: decouple tridiagonal block and complete 1_n^perp!
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

  // 3. Clear full workspace upfront so lower-diagonal beta entries are NEVER overwritten!
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

  // Largest Ritz value thetaMax -> lambda_2 = mu - thetaMax; second largest -> lambda_3
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

function renderMainHud(dispSlot) {
  if (!hudCtx) return;
  const W = hudCanvas.width;   // 960
  const H = hudCanvas.height;  // 540
  const sx = W / FRAME_W;      // 3.0
  const sy = H / FRAME_H;      // 3.0
  const fIdx = Math.round(telemView[0]);

  // Activate any Mode C pad flashes whose scheduled presentation frame has arrived at t - K
  for (let q = 0; q < FLASH_QUEUE_LEN; q++) {
    if (flashQueueFrame[q] >= 0 && fIdx >= flashQueueFrame[q]) {
      padFlashAlpha[flashQueueZone[q] & 3] = 1.0;
      flashQueueFrame[q] = -1;
    }
  }

  // 1. Blit delayed display frame `t - K` from static Wasm ring buffer
  dispStageCtx.putImageData(ringSlotImgData[dispSlot], 0, 0);
  hudCtx.imageSmoothingEnabled = false;
  hudCtx.drawImage(dispStageCanvas, 0, 0, W, H);

  // 2. Optional Live ECA Motion-Residual Overlay (explicitly labeled as frame `t`)
  if (showEcaOverlay && ecaResView) {
    for (let i = 0; i < FRAME_PIXELS; i++) {
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

  // 3. In Testbench mode, outline the two opaque occlusion zones cleanly on HUD
  if (sourceMode === 'testbench') {
    hudCtx.save();
    hudCtx.strokeStyle = 'rgba(116, 152, 210, 0.45)';
    hudCtx.setLineDash([6, 4]);
    hudCtx.lineWidth = 1.5;
    for (let p = 0; p < PILLARS.length; p++) {
      const pil = PILLARS[p];
      const px = pil.x0 * sx;
      const py = pil.y0 * sy;
      const pw = (pil.x1 - pil.x0) * sx;
      const ph = (pil.y1 - pil.y0) * sy;
      hudCtx.strokeRect(px, py, pw, ph);
      hudCtx.fillStyle = 'rgba(140, 175, 235, 0.72)';
      hudCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(`OCCLUDER P${p + 1}`, px + 8, py + 15);
    }
    hudCtx.restore();
  }

  // 4. Mode C: Air-Percussion Quadrant Zone Overlay (synchronized to delayed presentation)
  if (labMode === 2) {
    const labels = ['Q0: SUB-KICK 808', 'Q1: CRISP SNARE', 'Q2: METALLIC FM HAT', 'Q3: HARMONIC TOM'];
    hudCtx.save();
    hudCtx.strokeStyle = 'rgba(0, 240, 255, 0.22)';
    hudCtx.lineWidth = 1;
    hudCtx.beginPath();
    hudCtx.moveTo(W * 0.5, 0); hudCtx.lineTo(W * 0.5, H);
    hudCtx.moveTo(0, H * 0.5); hudCtx.lineTo(W, H * 0.5);
    hudCtx.stroke();

    for (let z = 0; z < 4; z++) {
      const qx = (z & 1) ? W * 0.5 : 0;
      const qy = (z & 2) ? H * 0.5 : 0;
      if (padFlashAlpha[z] > 0.01) {
        hudCtx.fillStyle = `rgba(0, 240, 255, ${(padFlashAlpha[z] * 0.24).toFixed(3)})`;
        hudCtx.fillRect(qx + 2, qy + 2, W * 0.5 - 4, H * 0.5 - 4);
        padFlashAlpha[z] *= 0.84;
      }
      hudCtx.fillStyle = 'rgba(165, 235, 255, 0.78)';
      hudCtx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(labels[z], qx + 14, qy + 22);
    }
    hudCtx.restore();
  }

  const effK = Math.round(telemView[1]);
  const uDisp = (RING_LEN - 1) - effK;
  const effFps = sourceMode === 'testbench' ? smoothFps : sourceFps;

  // 5. Render Target Trajectories, Hermite Spline Bridges, Reticles & Kinematic Kalman Baselines
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;

    const tid = Math.round(telemView[base + 1]);
    const color = TARGET_COLORS[(tid - 1) % TARGET_COLORS.length];
    const fx = telemView[base + 2] * sx;
    const fy = telemView[base + 3] * sy;
    const kx = telemView[base + 6] * sx;
    const ky = telemView[base + 7] * sy;
    const dispVis = telemView[base + 10] > 0.5;
    const isBridged = telemView[base + 11] > 0.5;
    const sigHex = toHex8(telemView[base + 12], telemView[base + 13]);
    const padicVal = Math.round(telemView[base + 14]);
    const apexX = telemView[base + 15] * sx;
    const apexY = telemView[base + 16] * sy;
    const apexFrames = telemView[base + 17];
    const fiedlerCoord = telemView[base + 19];
    const headX = telemView[base + 20 + 11] * sx;
    const headY = telemView[base + 32 + 11] * sy;

    // 5a. Past Dual-FISTA Ribbon [0 .. uDisp]
    hudCtx.save();
    hudCtx.lineWidth = 2.5;
    hudCtx.strokeStyle = color;
    hudCtx.beginPath();
    for (let u = 0; u <= uDisp; u++) {
      const px = telemView[base + 20 + u] * sx;
      const py = telemView[base + 32 + u] * sy;
      if (u === 0) hudCtx.moveTo(px, py);
      else hudCtx.lineTo(px, py);
    }
    hudCtx.stroke();

    // 5b. Future Non-Causal Lookahead / Hermite Bridge Ribbon [uDisp .. 11]
    if (uDisp < RING_LEN - 1) {
      hudCtx.lineWidth = isBridged ? 2.8 : 2.0;
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
      hudCtx.arc(headX, headY, 3.5, 0, Math.PI * 2);
      hudCtx.fill();
    }

    // 5c. Kinematic-Only 2x2 Kalman Baseline Marker
    if (showKalmanBaseline) {
      hudCtx.strokeStyle = '#ff595e';
      hudCtx.lineWidth = 1.5;
      hudCtx.setLineDash([3, 3]);
      hudCtx.strokeRect(kx - 14, ky - 14, 28, 28);
      hudCtx.beginPath();
      hudCtx.moveTo(fx, fy);
      hudCtx.lineTo(kx, ky);
      hudCtx.stroke();
      hudCtx.setLineDash([]);
      hudCtx.fillStyle = '#ff7b82';
      hudCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(`KAL-${tid}`, kx - 14, ky + 24);
    }

    // 5d. Mode A: Exact Newtonian Parabolic Ballistic Arc (quadratic Bezier control at (0.5*(x0+xA), yA))
    if (labMode === 0 && apexFrames > 1.5) {
      hudCtx.strokeStyle = '#ffd166';
      hudCtx.lineWidth = 1.6;
      hudCtx.setLineDash([4, 4]);
      hudCtx.beginPath();
      hudCtx.moveTo(headX, headY);
      hudCtx.quadraticCurveTo(0.5 * (headX + apexX), apexY, apexX, apexY);
      hudCtx.stroke();
      hudCtx.setLineDash([]);

      hudCtx.strokeStyle = '#ffd166';
      hudCtx.beginPath();
      hudCtx.arc(apexX, apexY, 9, 0, Math.PI * 2);
      hudCtx.moveTo(apexX - 13, apexY); hudCtx.lineTo(apexX + 13, apexY);
      hudCtx.moveTo(apexX, apexY - 13); hudCtx.lineTo(apexX, apexY + 13);
      hudCtx.stroke();
      hudCtx.fillStyle = '#ffd166';
      hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(`APEX ${(apexFrames * (1000 / Math.max(effFps, 24))).toFixed(0)}ms`, apexX + 12, apexY - 6);
    }

    // 5e. Primary 2-Adic + Fiedler + FISTA Corner-Bracket Reticle at `t - K`
    const boxR = 24;
    const bLen = 8;
    hudCtx.strokeStyle = isBridged ? '#ffd166' : color;
    hudCtx.lineWidth = 2.2;
    hudCtx.beginPath();
    hudCtx.moveTo(fx - boxR, fy - boxR + bLen); hudCtx.lineTo(fx - boxR, fy - boxR); hudCtx.lineTo(fx - boxR + bLen, fy - boxR);
    hudCtx.moveTo(fx + boxR - bLen, fy - boxR); hudCtx.lineTo(fx + boxR, fy - boxR); hudCtx.lineTo(fx + boxR, fy - boxR + bLen);
    hudCtx.moveTo(fx - boxR, fy + boxR - bLen); hudCtx.lineTo(fx - boxR, fy + boxR); hudCtx.lineTo(fx - boxR + bLen, fy + boxR);
    hudCtx.moveTo(fx + boxR - bLen, fy + boxR); hudCtx.lineTo(fx + boxR, fy + boxR); hudCtx.lineTo(fx + boxR, fy + boxR - bLen);
    hudCtx.stroke();

    const fSign = fiedlerCoord >= 0 ? 'F+' : 'F-';
    const stateTag = isBridged ? 'HERMITE-BRIDGE' : (dispVis ? 'LOCK' : 'COAST');
    const labelTop = `T${tid} [${stateTag}] v2:${padicVal} ${fSign}`;
    hudCtx.fillStyle = 'rgba(7, 11, 18, 0.82)';
    hudCtx.fillRect(fx - 26, fy - 46, 156, 19);
    hudCtx.fillStyle = isBridged ? '#ffd166' : color;
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(labelTop, fx - 22, fy - 33);

    hudCtx.fillStyle = 'rgba(205, 224, 245, 0.82)';
    hudCtx.font = '500 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(sigHex, fx + 28, fy + 4);
    hudCtx.restore();
  }

  // 6. Present-Time Phase-Locked Strobe Border Pulse
  if (strobeEnabled && strobeFlashAlpha > 0.02) {
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
    strobeFlashAlpha *= 0.78;
  }

  // 7. Top-left HUD Status Strip
  const clutterReady = telemView[22] > 0.5;
  const clutterStr = clutterReady ? `${telemView[4].toFixed(1)} dB` : 'WARMING';
  const ecaTag = showEcaOverlay ? ' | LIVE ECA@t' : '';
  const shownSwaps = sourceMode === 'testbench' ? oraclePadicSwaps : Math.round(telemView[5]);

  hudCtx.fillStyle = 'rgba(6, 10, 18, 0.80)';
  hudCtx.fillRect(10, 10, 490, 24);
  hudCtx.fillStyle = '#8ef9f3';
  hudCtx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
  hudCtx.fillText(
    `VIEW [t - ${effK}] | ECA ${clutterStr} | ID SWITCHES: ${shownSwaps} | KALMAN: ${Math.round(telemView[6])}${ecaTag}`,
    18,
    26
  );
}

function renderEulerianPip() {
  if (!pipCtx || !pipImgData) return;
  const W = pipCanvas.width;
  const H = pipCanvas.height;

  pipStageCtx.putImageData(pipImgData, 0, 0);
  pipCtx.imageSmoothingEnabled = false;
  pipCtx.drawImage(pipStageCanvas, 0, 0, W, H);

  pipCtx.strokeStyle = 'rgba(0, 240, 255, 0.45)';
  pipCtx.lineWidth = 1;
  pipCtx.beginPath();
  pipCtx.moveTo(W * 0.5, 0); pipCtx.lineTo(W * 0.5, H);
  pipCtx.moveTo(0, H * 0.5); pipCtx.lineTo(W, H * 0.5);
  pipCtx.stroke();
  pipCtx.strokeRect(W * 0.25, H * 0.25, W * 0.5, H * 0.5);

  pipCtx.fillStyle = 'rgba(6, 10, 18, 0.80)';
  pipCtx.fillRect(4, H - 22, W - 8, 18);
  pipCtx.fillStyle = '#00f0ff';
  pipCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const pid = Math.round(telemView[8]);
  pipCtx.fillText(
    pid >= 0 ? `X_FISTA CO-REG | ${eulerianGain.toFixed(0)}x CONTRAST` : 'AWAITING TARGET LOCK',
    10,
    H - 9
  );
}

function renderHilbertScope() {
  if (!hilbertCtx) return;
  const W = hilbertCanvas.width;
  const H = hilbertCanvas.height;

  hilbertCtx.fillStyle = '#080c14';
  hilbertCtx.fillRect(0, 0, W, H);

  const pSize = Math.min(H - 24, 144);
  const cx = 14 + pSize * 0.5;
  const cy = H * 0.5 + 4;
  const rad = pSize * 0.44;

  hilbertCtx.strokeStyle = '#1d293d';
  hilbertCtx.lineWidth = 1;
  hilbertCtx.beginPath();
  hilbertCtx.arc(cx, cy, rad, 0, Math.PI * 2);
  hilbertCtx.arc(cx, cy, rad * 0.5, 0, Math.PI * 2);
  hilbertCtx.moveTo(cx - rad, cy); hilbertCtx.lineTo(cx + rad, cy);
  hilbertCtx.moveTo(cx, cy - rad); hilbertCtx.lineTo(cx, cy + rad);
  hilbertCtx.stroke();

  const tx = cx + Math.cos(phaseOffsetRad) * rad;
  const ty = cy - Math.sin(phaseOffsetRad) * rad;
  hilbertCtx.strokeStyle = phaseOffsetRad > 0 ? '#ffb703' : '#00f0ff';
  hilbertCtx.setLineDash([3, 3]);
  hilbertCtx.beginPath();
  hilbertCtx.moveTo(cx, cy);
  hilbertCtx.lineTo(tx, ty);
  hilbertCtx.stroke();
  hilbertCtx.setLineDash([]);

  const scale = rad / 0.85;
  hilbertCtx.strokeStyle = '#38f27a';
  hilbertCtx.lineWidth = 1.6;
  hilbertCtx.beginPath();
  for (let k = 0; k < 48; k++) {
    const idx = (scopeWriteIdx + SCOPE_HIST - 48 + k) % SCOPE_HIST;
    const rx = cx + Math.max(-rad, Math.min(rad, histTremorR[idx] * scale));
    const hy = cy - Math.max(-rad, Math.min(rad, histTremorH[idx] * scale));
    if (k === 0) hilbertCtx.moveTo(rx, hy);
    else hilbertCtx.lineTo(rx, hy);
  }
  hilbertCtx.stroke();

  const curIdx = (scopeWriteIdx + SCOPE_HIST - 1) % SCOPE_HIST;
  const headX = cx + Math.max(-rad, Math.min(rad, histTremorR[curIdx] * scale));
  const headY = cy - Math.max(-rad, Math.min(rad, histTremorH[curIdx] * scale));
  hilbertCtx.fillStyle = '#ffffff';
  hilbertCtx.beginPath();
  hilbertCtx.arc(headX, headY, 3.2, 0, Math.PI * 2);
  hilbertCtx.fill();

  const waveX0 = 26 + pSize;
  const waveW = W - waveX0 - 10;
  const waveMidY = H * 0.5 + 4;
  const waveAmp = (H - 38) * 0.42;

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

  hilbertCtx.strokeStyle = '#00f0ff';
  hilbertCtx.lineWidth = 1.5;
  hilbertCtx.beginPath();
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
    const wy = waveMidY - Math.max(-1, Math.min(1, histTremorR[idx] * 1.3)) * waveAmp;
    if (i === 0) hilbertCtx.moveTo(wx, wy);
    else hilbertCtx.lineTo(wx, wy);
  }
  hilbertCtx.stroke();

  hilbertCtx.strokeStyle = '#e840f8';
  hilbertCtx.lineWidth = 1.3;
  hilbertCtx.beginPath();
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
    const wy = waveMidY - Math.max(-1, Math.min(1, histTremorH[idx] * 1.3)) * waveAmp;
    if (i === 0) hilbertCtx.moveTo(wx, wy);
    else hilbertCtx.lineTo(wx, wy);
  }
  hilbertCtx.stroke();

  hilbertCtx.fillStyle = '#9db4d4';
  hilbertCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const phaseValid = telemView[26] > 0.5;
  if (phaseValid) {
    const phiDeg = (telemView[11] * 180 / Math.PI).toFixed(0);
    hilbertCtx.fillText(
      `EQUALIZED HILBERT φ=${phiDeg}° | f=${telemView[13].toFixed(1)}Hz | R(t) CYAN  H{R} MAGENTA`,
      10,
      14
    );
  } else {
    const effK = Math.round(telemView[1]);
    hilbertCtx.fillText(
      effK < 7 ? 'HILBERT QUADRATURE GATED (REQUIRES K >= 7 FUTURE SUPPORT)' : 'HILBERT WINDOW WARMING / OCCLUDED',
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

  const midH = Math.floor(H * 0.54);
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
    const bh = Math.max(-30, Math.min(30, v2 * 38));
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

  const ringTop = midH + 18;
  spectralCtx.fillStyle = '#9db4d4';
  spectralCtx.fillText('12-SLOT NON-CAUSAL LOOKAHEAD ARENA (INGEST t vs DISPLAY t-K)', 10, midH + 10);

  // Exact ingest slot for just-processed `f_idx` (`telemView[0]` is pre-increment f_idx)
  const fIdx = Math.round(telemView[0]);
  const slotIn = fIdx % RING_LEN;
  const slotDisp = Math.round(telemView[2]);
  const cellW = (W - 20) / RING_LEN;
  const cellH = H - ringTop - 10;

  const pIdx = Math.round(telemView[8]);
  for (let s = 0; s < RING_LEN; s++) {
    const cx = 10 + s * cellW;
    const isIn = s === slotIn;
    const isDisp = s === slotDisp;

    let visState = 1;
    if (pIdx >= 0) {
      const u = (s + RING_LEN - ((slotIn + 1) % RING_LEN)) % RING_LEN;
      visState = telemView[32 + pIdx * 56 + 44 + u] > 0.5 ? 1 : 0;
    }

    spectralCtx.fillStyle = isDisp
      ? 'rgba(0, 240, 255, 0.28)'
      : isIn
      ? 'rgba(56, 242, 122, 0.26)'
      : visState === 0
      ? 'rgba(255, 209, 102, 0.22)'
      : '#101826';
    spectralCtx.fillRect(cx + 1, ringTop, cellW - 3, cellH);

    spectralCtx.strokeStyle = isDisp ? '#00f0ff' : isIn ? '#38f27a' : '#26344d';
    spectralCtx.lineWidth = isDisp || isIn ? 1.8 : 1.0;
    spectralCtx.strokeRect(cx + 1.5, ringTop + 0.5, cellW - 4, cellH - 1);

    spectralCtx.fillStyle = '#dbe7f8';
    spectralCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    spectralCtx.fillText(`S${s}`, cx + 5, ringTop + 13);

    if (isDisp) {
      spectralCtx.fillStyle = '#00f0ff';
      spectralCtx.fillText('t-K', cx + 4, ringTop + cellH - 5);
    } else if (isIn) {
      spectralCtx.fillStyle = '#38f27a';
      spectralCtx.fillText('IN', cx + 6, ringTop + cellH - 5);
    } else if (visState === 0) {
      spectralCtx.fillStyle = '#ffd166';
      spectralCtx.fillText('OCC', cx + 3, ringTop + cellH - 5);
    }
  }
}

// ============================================================================
// Core Per-Frame Execution Pipeline (Pre-Tick -> Lanczos Fiedler -> Post-Tick)
// ============================================================================

function processIngestedFrame(rawPixels, dtSec) {
  const safeDt = Math.min(0.1, Math.max(0.002, dtSec || 1.0 / 60.0));
  const instSourceFps = 1.0 / safeDt;
  sourceFps = 0.90 * sourceFps + 0.10 * instSourceFps;

  const ptr = wasmExports.chronos_get_ingest_rgba_ptr();
  const slotIdx = ((ptr - ringSlotU8[0].byteOffset) / FRAME_BYTES) | 0;
  ringSlotU8[slotIdx >= 0 && slotIdx < RING_LEN ? slotIdx : 0].set(rawPixels);

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

  // Record scope traces
  histTremorR[scopeWriteIdx] = telemView[9];
  histTremorH[scopeWriteIdx] = telemView[10];
  histStrobe[scopeWriteIdx] = telemView[14] > 0.5 ? 1 : 0;
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

  // Emit per-target deduplicated Mode C strikes and schedule synchronized t - K pad flashes
  const percMask = Math.round(telemView[15]);
  if (percMask !== 0 && labMode === 2) {
    const fIdx = Math.round(telemView[0]);
    const effSampleRate = Math.max(24.0, sourceMode === 'testbench' ? smoothFps : sourceFps);
    for (let m = 0; m < MAX_TARGETS; m++) {
      const ebase = 512 + m * 8;
      if (telemView[ebase] > 0.5) {
        const zone = Math.round(telemView[ebase + 1]) & 3;
        const leadFrames = telemView[ebase + 2];
        const vel = telemView[ebase + 3];

        flashQueueFrame[flashQueueHead] = fIdx + Math.round(leadFrames);
        flashQueueZone[flashQueueHead] = zone;
        flashQueueHead = (flashQueueHead + 1) % FLASH_QUEUE_LEN;

        self.postMessage({
          type: 'audio_percussion',
          zone,
          leadSec: leadFrames / effSampleRate,
          velocity: vel,
          sentPerfMs: performance.now(),
        });
      }
    }
  }

  // Render all 4 OffscreenCanvases inside Worker isolate
  renderMainHud(dispSlot);
  renderEulerianPip();
  renderHilbertScope();
  renderSpectralAndRingScope();

  tickCounter++;
  if (tickCounter % 3 === 0) {
    let sumMs = 0.0;
    let maxMs = 0.0;
    const count = Math.min(tickCounter, TIME_HIST_LEN);
    for (let i = 0; i < count; i++) {
      sumMs += wasmTimeHist[i];
      if (wasmTimeHist[i] > maxMs) maxMs = wasmTimeHist[i];
    }
    const meanMs = sumMs / Math.max(1, count);
    const effSampleRate = Math.max(24.0, sourceMode === 'testbench' ? smoothFps : sourceFps);

    const pIdx = Math.round(telemView[8]);
    let pSigHex = '0x00000000';
    let pVal = 0;
    let apexMs = 0;
    if (pIdx >= 0) {
      const b = 32 + pIdx * 56;
      pSigHex = toHex8(telemView[b + 12], telemView[b + 13]);
      pVal = Math.round(telemView[b + 14]);
      apexMs = telemView[b + 17] * (1000.0 / effSampleRate);
    }

    self.postMessage({
      type: 'telemetry',
      frameIdx: Math.round(telemView[0]),
      lookaheadK: Math.round(telemView[1]),
      fps: effSampleRate,
      wasmMeanMs: meanMs,
      wasmMaxMs: maxMs,
      activeTargets: Math.round(telemView[3]),
      clutterDb: telemView[4],
      clutterDbValid: telemView[22] > 0.5,
      clutterEvalRatio: telemView[23],
      padicSwaps: Math.round(telemView[5]),
      oracleSwaps: oraclePadicSwaps,
      sourceMode,
      kalmanSwaps: Math.round(telemView[6]),
      inPillarCollisions,
      lambda2,
      spectralGap: telemView[24],
      fistaDualityGap: telemView[25],
      phaseValid: telemView[26] > 0.5,
      tremorPhaseDeg: telemView[11] * (180.0 / Math.PI),
      tremorFreqHz: telemView[13],
      tremorAmp: telemView[12],
      saccadeLatencyMs: telemView[19],
      stepSettleMs: telemView[28],
      stimulusArmed: telemView[29] > 0.5,
      primarySigHex: pSigHex,
      primaryPadicVal: pVal,
      apexMs,
      arenaBytes: Math.round(telemView[20]),
      memoryPages: wasmMemory.buffer.byteLength / 65536,
    });
  }
}

let rafHandle = 0;
let rafWatchdogId = 0;

function scheduleNextWorkerTick() {
  if (!running) return;
  if (typeof self.requestAnimationFrame === 'function') {
    clearTimeout(rafWatchdogId);
    const prevTick = tickCounter;
    rafHandle = self.requestAnimationFrame(workerRafLoop);
    rafWatchdogId = setTimeout(() => {
      if (running && tickCounter === prevTick) {
        if (typeof self.cancelAnimationFrame === 'function') {
          self.cancelAnimationFrame(rafHandle);
        }
        workerRafLoop(performance.now());
      }
    }, 18);
  } else {
    setTimeout(() => workerRafLoop(performance.now()), 16);
  }
}

function workerRafLoop(now) {
  if (!running) return;
  clearTimeout(rafWatchdogId);
  const dtMs = lastRafTs > 0 ? Math.min(50.0, Math.max(4.0, now - lastRafTs)) : 16.667;
  lastRafTs = now;
  const instFps = 1000.0 / dtMs;
  smoothFps = 0.92 * smoothFps + 0.08 * instFps;

  if (sourceMode === 'testbench' && wasmExports) {
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

    dispStageCanvas = new OffscreenCanvas(FRAME_W, FRAME_H);
    dispStageCtx = dispStageCanvas.getContext('2d');
    pipStageCanvas = new OffscreenCanvas(16, 16);
    pipStageCtx = pipStageCanvas.getContext('2d');
    simCanvas = new OffscreenCanvas(FRAME_W, FRAME_H);
    simCtx = simCanvas.getContext('2d', { willReadFrequently: true });

    ecaStageCanvas = new OffscreenCanvas(FRAME_W, FRAME_H);
    ecaStageCtx = ecaStageCanvas.getContext('2d');
    ecaOverlayBuf = new Uint8ClampedArray(FRAME_BYTES);
    ecaOverlayImgData = new ImageData(ecaOverlayBuf, FRAME_W, FRAME_H);

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

    const slot0Ptr = wasmExports.chronos_get_ingest_rgba_ptr();
    for (let s = 0; s < RING_LEN; s++) {
      const byteOff = slot0Ptr + s * FRAME_BYTES;
      ringSlotU8[s] = new Uint8Array(wasmMemory.buffer, byteOff, FRAME_BYTES);
      ringSlotImgData[s] = new ImageData(
        new Uint8ClampedArray(wasmMemory.buffer, byteOff, FRAME_BYTES),
        FRAME_W,
        FRAME_H
      );
    }

    ecaResView = new Float32Array(wasmMemory.buffer, wasmExports.chronos_get_eca_res_ptr(), FRAME_PIXELS);
    pipImgData = new ImageData(
      new Uint8ClampedArray(wasmMemory.buffer, wasmExports.chronos_get_pip_rgba_ptr(), 1024),
      16,
      16
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

    resetTestbenchState();
    running = true;
    self.postMessage({
      type: 'ready',
      memoryBytes: wasmMemory.buffer.byteLength,
      pages: wasmMemory.buffer.byteLength / 65536,
    });
    scheduleNextWorkerTick();
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
    }
  } else if (msg.type === 'set_source') {
    sourceMode = msg.sourceMode;
    if (wasmExports) {
      wasmExports.chronos_reset_tracks();
    }
    resetTestbenchState();
  } else if (msg.type === 'video_frame') {
    const bmp = msg.bitmap;
    if (wasmExports && sourceMode !== 'testbench') {
      simCtx.drawImage(bmp, 0, 0, FRAME_W, FRAME_H);
      const rawPixels = simCtx.getImageData(0, 0, FRAME_W, FRAME_H).data;
      processIngestedFrame(rawPixels, msg.dtSec || 1.0 / 60.0);
    }
    bmp.close();
  } else if (msg.type === 'lab_action') {
    if (msg.action === 'saccade_step') {
      if (wasmExports) wasmExports.chronos_trigger_stimulus();
      if (sourceMode === 'testbench') {
        pendingSaccadeFrames = 9; // ~150 ms deterministic simulated stimulus-to-onset latency
      }
    } else if (msg.action === 'ballistic_toss' && sourceMode === 'testbench') {
      simDiscs[0].x = 24;
      simDiscs[0].y = 148;
      simDiscs[0].vx = 3.8;
      simDiscs[0].vy = -6.8;
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
  }
};
