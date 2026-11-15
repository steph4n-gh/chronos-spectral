#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Streaming Real-Camera MP4 Calibration & Diagnostic Suite
// Streams raw RGBA frames one-by-one from ffmpeg (O(1) single-frame memory,
// zero whole-video buffering) through the exact production Wasm + 20-sweep
// Jacobi Lanczos solver, verifying 1568-float finiteness, reserved-tail zeros,
// eigen-residuals, ECA clutter suppression, CLEAR-MOT recall & precision,
// multi-arc Auto-g 2x SD/HD consistency, Kåsa pivot gating, harmonic phase,
// and Wave-2 in-situ ODE/Kuramoto/Lyapunov/Collision physics invariants.
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const WASM_PATH = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');
const GT_PATH = path.join(ROOT, 'scripts', 'gt_mp4_reference.json');
const gtDataset = fs.existsSync(GT_PATH) ? JSON.parse(fs.readFileSync(GT_PATH, 'utf8')) : null;

const GRAPH_NODES = 32;
const MAX_TARGETS = 16;
const MAX_KRYLOV = 31;
const TELEMETRY_LEN = 1568;
const TELEM_PHYSICS_START = 1280;
const TELEM_PHYSICS_STRIDE = 16;
const TELEM_ENSEMBLE_START = 1536;
const TELEM_RESERVED_TAIL_START = 1550;

const CORE_CLIPS = [
  {
    file: 'juggling_ballistic.mp4',
    fps: 25,
    labMode: 0,
    hasGt: true,
    minLockPct: 90.0,
    minRecallPct: 70.0,
    minPrecisionPct: 14.0,
    minClutterDb: 38.0,
    minPersistentAge: 10,
    minPersistent: 4,
    minApexSd: 8,
    minApexHd: 8,
    minGravArcs: 2,
    minGravConf: 0.50,
    checkDualResGravity: true,
    minBridges: 0,
    minPhase: 0,
    minPerc: 0,
    desc: '25fps Chalkboard Newtonian Juggling (Parabolic Ballistic Free-Flight & Multi-Arc Auto-g)',
  },
  {
    file: 'roundabout_occlusion.mp4',
    fps: 24,
    labMode: 0,
    hasGt: true,
    minLockPct: 92.0,
    minRecallPct: 88.0,
    minPrecisionPct: 28.0,
    minClutterDb: 41.0,
    minPersistentAge: 15,
    minPersistent: 4,
    minApexSd: 0,
    minApexHd: 0,
    maxPivotFrames: 0,
    maxFreqValidFrames: 0,
    minBridges: 6,
    minPhase: 0,
    minPerc: 0,
    physicsNegativeGate: true,
    desc: '24fps Highway Gantry Sign Occlusion (Short-Gap Hermite Bridge, 0 False Pivots/Hz/Physics)',
  },
  {
    file: 'galileo_pendulum_peg.mp4',
    fps: 30,
    labMode: 0,
    hasGt: true,
    minLockPct: 94.0,
    minRecallPct: 95.0,
    minPrecisionPct: 28.0,
    minClutterDb: 35.0,
    minPersistentAge: 15,
    minPersistent: 1,
    minApexSd: 2,
    minApexHd: 4,
    minPivotFrames: 40,
    minBridges: 0,
    minPhase: 0,
    minPerc: 0,
    requirePendulumPhysics: true,
    desc: '30fps Galileo Interrupted Pendulum & Peg (Kåsa/Taubin Pivot + Peg Jump + Elliptic T/zeta/Q)',
  },
  {
    file: 'harmonic_metronome_phase.mp4',
    fps: 30,
    labMode: 1,
    hasGt: true,
    minLockPct: 94.0,
    minRecallPct: 80.0,
    minPrecisionPct: 11.0,
    minClutterDb: 42.0,
    minPersistentAge: 15,
    minPersistent: 1,
    minApexSd: 0,
    minApexHd: 0,
    minFreqValidFrames: 50,
    freqRangeHz: [0.68, 0.92],
    minBridges: 0,
    minPhase: 150,
    minPerc: 0,
    desc: '30fps Nikko Mechanical Metronome @ 96 BPM / 0.80 Hz (Hilbert Quadrature & Phase Strobe)',
  },
  {
    file: 'carom_billiards_impulses.mp4',
    fps: 30,
    labMode: 2,
    hasGt: true,
    minLockPct: 94.0,
    minRecallPct: 82.0,
    minPrecisionPct: 30.0,
    minClutterDb: 42.0,
    minPersistentAge: 15,
    minPersistent: 2,
    minApexSd: 0,
    minApexHd: 0,
    maxPivotFrames: 0,
    maxFreqValidFrames: 0,
    minBridges: 0,
    minPhase: 0,
    minPerc: 8,
    requireImpulsivePhysics: true,
    desc: '30fps 3-Cushion Carom Billiards (Single-Ball Rail Cushion L1 Impulses, 0 False Pivots/Hz)',
  },
];

const EXTENDED_CLIPS = [
  {
    file: 'model_rocket_full_flights.mp4',
    fps: 25,
    labMode: 0,
    hasGt: false,
    minLockPct: 45.0,
    minClutterDb: 25.0,
    minPersistentAge: 12,
    minPersistent: 1,
    maxStaticPins: 6,
    minBoostFrames: 1,
    minBurnoutJumpFrames: 1,
    minParachuteFrames: 1,
    minParachuteJumpFrames: 1,
    desc: '25fps Twin-Stage Model Rockets (Boost Thrust, Burnout Jump & Parachute Recovery)',
  },
  {
    file: 'rocket_vtvl_full_flight.mp4',
    fps: 30,
    labMode: 0,
    hasGt: false,
    minLockPct: 45.0,
    minClutterDb: 25.0,
    minPersistentAge: 15,
    minPersistent: 1,
    maxStaticPins: 8,
    minBoostFrames: 1,
    desc: '30fps VTVL Pad-to-Pad Rocket Free-Flight (Powered Boost)',
  },
  {
    file: 'water_rocket_parachute_flights.mp4',
    fps: 25,
    labMode: 0,
    hasGt: false,
    minLockPct: 45.0,
    minClutterDb: 25.0,
    minPersistentAge: 12,
    minPersistent: 1,
    maxStaticPins: 6,
    minChuteOrDragFrames: 1,
    desc: '25fps Water Rockets & Parachute Recovery (Aero Drag & Recovery)',
  },
  {
    file: 'cologne_motorway_occlusion.mp4',
    fps: 30,
    labMode: 0,
    hasGt: false,
    minLockPct: 80.0,
    minClutterDb: 35.0,
    minPersistentAge: 15,
    minPersistent: 4,
    maxStaticPins: 8,
    physicsNegativeGate: true,
    desc: '30fps Cologne Motorway & Overpass Occlusion (0 False Boost/Pendulum/Collision/Kuramoto/Lyap)',
  },
  {
    file: 'rally_ego_pan_tracking.mp4',
    fps: 30,
    labMode: 0,
    hasGt: false,
    minLockPct: 45.0,
    minClutterDb: 18.0,
    minPersistentAge: 12,
    minPersistent: 1,
    maxStaticPins: 6,
    physicsNegativeGate: true,
    desc: '30fps Panning-Camera Rally Ego-Motion Tracking (Camera Motion Gated, 0 False Physics)',
  },
  {
    file: 'chaotic_magnetic_pendulum.mp4',
    fps: 25,
    labMode: 0,
    hasGt: false,
    minLockPct: 80.0,
    minClutterDb: 32.0,
    minPersistentAge: 15,
    minPersistent: 1,
    maxStaticPins: 4,
    minLyapunovMax: 0.15,
    desc: '25fps Chaotic Magnetic Pendulum (Positive Lyapunov Divergence lambda_max > 0.15 s^-1)',
  },
  {
    file: 'coupled_metronomes_sync.mp4',
    fps: 30,
    labMode: 1,
    hasGt: false,
    minLockPct: 80.0,
    minClutterDb: 32.0,
    minPersistentAge: 15,
    minPersistent: 2,
    maxStaticPins: 4,
    minFreqValidFrames: 30,
    minKuramotoOscCount: 2,
    minKuramotoOrderR: 0.45,
    desc: '30fps Coupled Metronomes Phase Sync (Kuramoto N >= 2 & R_sync >= 0.45)',
  },
  {
    file: 'newtons_cradle_impulses.mp4',
    fps: 30,
    labMode: 2,
    hasGt: false,
    minLockPct: 80.0,
    minClutterDb: 32.0,
    minPersistentAge: 15,
    minPersistent: 2,
    maxStaticPins: 8,
    minPerc: 4,
    requireCollisionPhysics: true,
    desc: '30fps Slow-Motion 5-Sphere Newton Cradle Collisions (Two-Body Impulses, Mass Ratio & e_n)',
  },
];

function streamMp4Frames(mp4Path, w, h, onFrame) {
  const frameBytes = w * h * 4;
  const frameScratch = new Uint8Array(frameBytes);
  let carry = 0;

  return new Promise((resolve, reject) => {
    const proc = spawn(
      'ffmpeg',
      ['-v', 'error', '-i', mp4Path, '-vf', `scale=${w}:${h}`, '-pix_fmt', 'rgba', '-f', 'rawvideo', 'pipe:1'],
      { stdio: ['ignore', 'pipe', 'pipe'] }
    );
    let stderrText = '';
    let frameCount = 0;

    proc.stderr.on('data', (chunk) => {
      stderrText += chunk.toString('utf8');
    });

    proc.stdout.on('data', (chunk) => {
      let offset = 0;
      while (offset < chunk.length) {
        const needed = frameBytes - carry;
        const take = Math.min(needed, chunk.length - offset);
        frameScratch.set(chunk.subarray(offset, offset + take), carry);
        carry += take;
        offset += take;
        if (carry === frameBytes) {
          onFrame(frameScratch, frameCount++);
          carry = 0;
        }
      }
    });

    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}: ${stderrText}`));
      } else {
        resolve({ numFrames: frameCount, frameBytes });
      }
    });
  });
}

/// Exact production Shifted Lanczos + Invariant-Subspace Restart + 20-Sweep Symmetric Jacobi solver (matches web/worker.js).
function createLanczosSolver(exp, laplacian, fiedler) {
  const activeNodeIdx = new Int32Array(GRAPH_NODES);
  const lanczosQ = new Float64Array(MAX_KRYLOV * GRAPH_NODES);
  const lanczosAlpha = new Float64Array(MAX_KRYLOV);
  const lanczosBeta = new Float64Array(MAX_KRYLOV);
  const lanczosW = new Float64Array(GRAPH_NODES);
  const jacobiA = new Float64Array(MAX_KRYLOV * MAX_KRYLOV);
  const jacobiV = new Float64Array(MAX_KRYLOV * MAX_KRYLOV);
  const prevFiedlerTrack = new Float64Array(MAX_TARGETS);
  const eigVecBuf = new Float64Array(GRAPH_NODES);

  function findOrthogonalBasisSeed(n, numExisting, outOffset) {
    for (let cand = 0; cand < n; cand++) {
      const invN = 1.0 / n;
      for (let a = 0; a < n; a++) lanczosW[a] = (a === cand ? 1.0 : 0.0) - invN;
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
        for (let a = 0; a < n; a++) lanczosQ[outOffset + a] = lanczosW[a] * invNorm;
        return true;
      }
    }
    return false;
  }

  return function solveShiftedLanczosFiedler() {
    let n = 0;
    for (let i = 0; i < GRAPH_NODES; i++) {
      if (laplacian[i * GRAPH_NODES + i] > 1e-6) activeNodeIdx[n++] = i;
    }
    fiedler.fill(0.0);
    if (n < 2) {
      exp.chronos_set_fiedler_spectrum(0.0, 0.0);
      return { lambda2: 0.0, resNorm: 0.0 };
    }
    let mu = 0.0;
    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a];
      let rowSum = laplacian[ia * GRAPH_NODES + ia];
      for (let b = 0; b < n; b++) {
        if (a !== b) rowSum += Math.abs(laplacian[ia * GRAPH_NODES + activeNodeIdx[b]]);
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
    for (let a = 0; a < n; a++) lanczosQ[a] *= invNorm0;

    let actualM = 0;
    for (let j = 0; j < mSteps; j++) {
      const qOff = j * GRAPH_NODES;
      for (let a = 0; a < n; a++) {
        const ia = activeNodeIdx[a];
        let sumL = 0.0;
        for (let b = 0; b < n; b++) sumL += laplacian[ia * GRAPH_NODES + activeNodeIdx[b]] * lanczosQ[qOff + b];
        lanczosW[a] = mu * lanczosQ[qOff + a] - sumL;
      }
      let wMean = 0.0;
      for (let a = 0; a < n; a++) wMean += lanczosW[a];
      wMean /= n;
      for (let a = 0; a < n; a++) lanczosW[a] -= wMean;

      if (j > 0 && lanczosBeta[j - 1] > 0.0) {
        const prevOff = (j - 1) * GRAPH_NODES;
        const betaPrev = lanczosBeta[j - 1];
        for (let a = 0; a < n; a++) lanczosW[a] -= betaPrev * lanczosQ[prevOff + a];
      }
      let alpha = 0.0;
      for (let a = 0; a < n; a++) alpha += lanczosQ[qOff + a] * lanczosW[a];
      lanczosAlpha[j] = alpha;
      for (let a = 0; a < n; a++) lanczosW[a] -= alpha * lanczosQ[qOff + a];

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
      for (let a = 0; a < n; a++) beta2 += lanczosW[a] * lanczosW[a];
      const beta = Math.sqrt(Math.max(0.0, beta2));
      actualM = j + 1;

      if (j + 1 >= mSteps) {
        lanczosBeta[j] = 0.0;
        break;
      }
      const nextOff = (j + 1) * GRAPH_NODES;
      if (beta < 1e-8) {
        lanczosBeta[j] = 0.0;
        if (!findOrthogonalBasisSeed(n, j + 1, nextOff)) break;
      } else {
        lanczosBeta[j] = beta;
        const invBeta = 1.0 / beta;
        for (let a = 0; a < n; a++) lanczosQ[nextOff + a] = lanczosW[a] * invBeta;
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
      if (nodeIdx < MAX_TARGETS) dotPrev += comp * prevFiedlerTrack[nodeIdx];
    }

    const signFix = (dotPrev < -1e-4 || (Math.abs(dotPrev) <= 1e-4 && eigVecBuf[0] < 0.0)) ? -1.0 : 1.0;
    for (let a = 0; a < n; a++) {
      const nodeIdx = activeNodeIdx[a];
      const val = eigVecBuf[a] * signFix;
      fiedler[nodeIdx] = val;
      if (nodeIdx < MAX_TARGETS) prevFiedlerTrack[nodeIdx] = val;
    }

    let res2 = 0.0;
    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a];
      let lv = 0.0;
      for (let b = 0; b < n; b++) lv += laplacian[ia * GRAPH_NODES + activeNodeIdx[b]] * eigVecBuf[b];
      const diff = lv - lambda2 * eigVecBuf[a];
      res2 += diff * diff;
    }

    exp.chronos_set_fiedler_spectrum(lambda2, spectralGap);
    return { lambda2, resNorm: Math.sqrt(res2) };
  };
}

const wasmBytes = fs.readFileSync(WASM_PATH);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const exp = instance.exports;
const mem = exp.memory;

const strictGate = process.argv.includes('--gate');
const hdMode = process.argv.includes('--hd') && typeof exp.chronos_set_resolution === 'function';
const useExtended = process.argv.includes('--extended');
const useAll = process.argv.includes('--all');
const noSpeedGate = process.argv.includes('--no-speed-gate');
const clipArg = process.argv.find((a) => a.startsWith('--clip='));
const clipFilter = clipArg ? clipArg.slice('--clip='.length) : '';
const budgetScale = Math.max(1.0, Number(process.env.CHRONOS_CI_BUDGET_SCALE || '1.0'));
const speedBudgetMs = (hdMode ? 1.00 : 0.35) * budgetScale;

let CLIPS = useAll
  ? [...CORE_CLIPS, ...EXTENDED_CLIPS]
  : useExtended
  ? [...EXTENDED_CLIPS]
  : [...CORE_CLIPS];
if (clipFilter) {
  CLIPS = CLIPS.filter((c) => c.file.includes(clipFilter));
  if (CLIPS.length === 0) {
    throw new Error(`No clips matched --clip=${clipFilter}`);
  }
}

const needsGtDataset = CLIPS.some((c) => c.hasGt !== false);
if (strictGate && needsGtDataset && !gtDataset) {
  throw new Error(`--gate requires ground-truth reference file: ${GT_PATH}`);
}

async function measureLockedSceneGravity(mp4Path, fps, labMode, resMode) {
  exp.chronos_init();
  if (typeof exp.chronos_set_resolution === 'function') {
    exp.chronos_set_resolution(resMode);
  }
  const w = typeof exp.chronos_get_frame_w === 'function' ? exp.chronos_get_frame_w() : 320;
  const h = typeof exp.chronos_get_frame_h === 'function' ? exp.chronos_get_frame_h() : 180;
  const frameBytes = w * h * 4;
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, labMode);
  const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
  const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 1024);
  const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 32);
  const ringViews = Array.from({ length: 12 }, (_, slot) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), frameBytes));
  const solveLanczos = createLanczosSolver(exp, laplacian, fiedler);
  const dtSec = 1.0 / fps;
  let lastGrav = 0.0;
  let maxConf = 0.0;
  let maxArcs = 0;
  await streamMp4Frames(mp4Path, w, h, (frameRgba, f) => {
    ringViews[f % 12].set(frameRgba);
    if (typeof exp.chronos_set_ingest_pts === 'function') {
      exp.chronos_set_ingest_pts(f * dtSec);
    }
    exp.chronos_pre_tick(dtSec);
    solveLanczos();
    exp.chronos_post_tick(dtSec);
    if (telem[1248] > 0.0) lastGrav = telem[1248];
    if (telem[1249] > maxConf) maxConf = telem[1249];
    if (Math.round(telem[1259]) > maxArcs) maxArcs = Math.round(telem[1259]);
  });
  return { grav: lastGrav, conf: maxConf, arcs: maxArcs };
}

console.log(`========================================================================`);
console.log(` CHRONOS-SPECTRAL // Real-Camera MP4 Calibration Suite (${hdMode ? '640x360 HD' : '320x180 SD'})`);
console.log(`========================================================================`);

let allPassed = true;

for (const clip of CLIPS) {
  const mp4Path = path.join(SAMPLES_DIR, clip.file);
  if (!fs.existsSync(mp4Path)) {
    throw new Error(`Missing sample video: ${mp4Path}`);
  }
  if (strictGate && clip.hasGt !== false && (!gtDataset || !gtDataset[clip.file])) {
    throw new Error(`Missing ground-truth annotations for ${clip.file} in ${GT_PATH}`);
  }

  exp.chronos_init();
  if (typeof exp.chronos_set_resolution === 'function') {
    exp.chronos_set_resolution(hdMode ? 1 : 0);
  }
  const w = typeof exp.chronos_get_frame_w === 'function' ? exp.chronos_get_frame_w() : 320;
  const h = typeof exp.chronos_get_frame_h === 'function' ? exp.chronos_get_frame_h() : 180;
  const frameBytes = w * h * 4;

  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, clip.labMode);

  const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
  const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 1024);
  const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 32);
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), frameBytes));
  const solveLanczos = createLanczosSolver(exp, laplacian, fiedler);
  const dtSec = 1.0 / clip.fps;
  const s = w / 320.0;
  const gtScale = w / 640.0;

  const gtByFrame = new Map();
  if (gtDataset && gtDataset[clip.file]) {
    for (const tr of gtDataset[clip.file].tracks) {
      if (tr.stationary) continue;
      for (const pt of tr.frames) {
        if (!gtByFrame.has(pt.f)) gtByFrame.set(pt.f, []);
        gtByFrame.get(pt.f).push({
          gtId: tr.id,
          x: pt.x * gtScale,
          y: pt.y * gtScale,
          rad: tr.matchRadiusHd * gtScale,
        });
      }
    }
  }

  const trackAges = new Map();
  const trackLife = new Map();
  const lastTidForGt = new Map();
  const lastGtForTid = new Map();
  let gtTotal = 0;
  let gtMatched = 0;
  let gtPredTotal = 0;
  let gtUnmatchedFp = 0;
  let gtSumSqErr = 0.0;
  let gtIdSwaps = 0;
  let gtHijacks = 0;
  let teleportJumps = 0;
  let sumTickMs = 0.0;
  let warmSumTickMs = 0.0;
  const tickSamples = [];
  let maxClutterDb = 0.0;
  let lastClutterDb = 0.0;
  let lastEvalRatio = 0.0;
  let framesWithTargets = 0;
  let sumActiveTargets = 0;
  let maxActiveTargets = 0;
  let bridgeFrames = 0;
  let apexFrames = 0;
  let phaseValidFrames = 0;
  let freqValidFrames = 0;
  const validFreqSamples = [];
  let percTriggers = 0;
  let pivotFrames = 0;
  let maxDeflectDeg = 0.0;
  let lastSceneGrav = 0.0;
  let maxGravConf = 0.0;
  let maxGravArcs = 0;
  let lastScatterDeg = 0.0;
  let maxEigRes = 0.0;
  let nanDetected = false;
  let tailNonZeroCount = 0;
  let inactiveNonZeroCount = 0;
  const regimeFrames = new Int32Array(9);
  const jumpFrames = new Int32Array(5);
  let invalidRegimeCount = 0;
  let invalidJumpCount = 0;
  let posEllipticFrames = 0;
  let posQualQFrames = 0;
  let validRestitutionFrames = 0;
  let invalidRestitutionFrames = 0;
  let maxKuramotoR = 0.0;
  let maxKuramotoCount = 0;
  let maxLyapunov = 0.0;
  let maxLastPairFrame = 0;
  let maxTaubinDelta = 0.0;
  let pairEventCount = 0;
  let invalidPairEvents = 0;
  let minPairMomResid = Infinity;
  let maxPairMassRatio = 0.0;
  let lastSeenPairFrame = 0;
  let lastSeenPairTidA = 0;
  let lastSeenPairTidB = 0;
  const teleportStepMax = 42.0 * s * Math.max(1.0, (60.0 / clip.fps) * 0.78);

  const { numFrames } = await streamMp4Frames(mp4Path, w, h, (frameRgba, f) => {
    ringViews[f % 12].set(frameRgba);
    if (typeof exp.chronos_set_ingest_pts === 'function') {
      exp.chronos_set_ingest_pts(f * dtSec);
    }

    const t0 = performance.now();
    exp.chronos_pre_tick(dtSec);
    const { resNorm } = solveLanczos();
    exp.chronos_post_tick(dtSec);
    const tickDt = performance.now() - t0;
    sumTickMs += tickDt;
    if (f > 0) warmSumTickMs += tickDt;
    tickSamples.push(tickDt);

    if (resNorm > maxEigRes) maxEigRes = resNorm;
    for (let i = 0; i < TELEMETRY_LEN; i++) {
      if (!Number.isFinite(telem[i])) nanDetected = true;
    }
    for (let i = TELEM_RESERVED_TAIL_START; i < TELEMETRY_LEN; i++) {
      if (telem[i] !== 0.0) tailNonZeroCount++;
    }

    const eb = TELEM_ENSEMBLE_START;
    const kurR = telem[eb + 0];
    const kurCount = Math.round(telem[eb + 2]);
    const pairMassRatio = telem[eb + 5];
    const pairMomResid = telem[eb + 6];
    const pairEnergyRet = telem[eb + 7];
    const pairTidA = Math.round(telem[eb + 8]);
    const pairTidB = Math.round(telem[eb + 9]);
    const pairFrame = Math.round(telem[eb + 10]);
    const lyapDiv = telem[eb + 11];
    const taubinDelta = telem[eb + 12];
    if (kurR > maxKuramotoR) maxKuramotoR = kurR;
    if (kurCount > maxKuramotoCount) maxKuramotoCount = kurCount;
    if (lyapDiv > maxLyapunov) maxLyapunov = lyapDiv;
    if (pairFrame > maxLastPairFrame) maxLastPairFrame = pairFrame;
    if (taubinDelta > maxTaubinDelta) maxTaubinDelta = taubinDelta;
    if (
      pairFrame > 0 &&
      (pairFrame !== lastSeenPairFrame || pairTidA !== lastSeenPairTidA || pairTidB !== lastSeenPairTidB)
    ) {
      lastSeenPairFrame = pairFrame;
      lastSeenPairTidA = pairTidA;
      lastSeenPairTidB = pairTidB;
      pairEventCount++;
      const validPair =
        pairMassRatio > 0.0 &&
        pairMassRatio <= 20.0 &&
        pairMomResid >= 0.0 &&
        pairMomResid <= 0.45 &&
        pairEnergyRet > 0.0 &&
        pairEnergyRet <= 1.10;
      if (!validPair) invalidPairEvents++;
      if (pairMomResid < minPairMomResid) minPairMomResid = pairMomResid;
      if (pairMassRatio > maxPairMassRatio) maxPairMassRatio = pairMassRatio;
    }

    const active = Math.round(telem[3]);
    if (active > 0) framesWithTargets++;
    sumActiveTargets += active;
    if (active > maxActiveTargets) maxActiveTargets = active;

    if (telem[22] > 0.5) {
      lastClutterDb = telem[4];
      lastEvalRatio = telem[23];
      if (lastClutterDb > maxClutterDb) maxClutterDb = lastClutterDb;
    }
    if (telem[26] > 0.5) phaseValidFrames++;
    if (telem[1260] > 0.5 && telem[13] > 0.0) {
      freqValidFrames++;
      validFreqSamples.push(telem[13]);
    }
    if (Math.round(telem[15]) !== 0) percTriggers++;
    if (telem[1248] > 0.0) lastSceneGrav = telem[1248];
    if (telem[1249] > maxGravConf) maxGravConf = telem[1249];
    if (Math.round(telem[1259]) > maxGravArcs) maxGravArcs = Math.round(telem[1259]);
    if (telem[1250] > 0.0) lastScatterDeg = telem[1250];

    const activeTracks = [];
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      const kbase = 1120 + m * 8;
      const pbase = TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE;
      if (telem[base] > 0.5) {
        const tid = Math.round(telem[base + 1]);
        const fx = telem[base + 2];
        const fy = telem[base + 3];
        const ix = telem[base + 8];
        const iy = telem[base + 9];
        const latestVis = telem[base + 44 + 11] > 0.5;
        trackAges.set(tid, (trackAges.get(tid) || 0) + 1);
        if (telem[base + 11] > 0.5) bridgeFrames++;
        if (telem[base + 17] > 0.35) apexFrames++;
        if (telem[kbase + 6] > -9e5) pivotFrames++;
        if (Math.abs(telem[kbase + 3]) > maxDeflectDeg) maxDeflectDeg = Math.abs(telem[kbase + 3]);

        const physRegime = Math.round(telem[pbase + 4]);
        if (physRegime >= 0 && physRegime <= 8) regimeFrames[physRegime]++;
        else invalidRegimeCount++;
        const jumpCode = Math.round(telem[pbase + 15]);
        if (jumpCode >= 1 && jumpCode <= 4) jumpFrames[jumpCode]++;
        else if (jumpCode !== 0) invalidJumpCount++;
        if (telem[pbase + 9] > 0.0) posEllipticFrames++;
        if (telem[pbase + 11] > 0.0) posQualQFrames++;
        const restE = telem[pbase + 14];
        if (restE > 0.0 && restE <= 1.0) validRestitutionFrames++;
        else if (restE < 0.0 || restE > 1.0) invalidRestitutionFrames++;

        activeTracks.push({ m, tid, fx, fy });

        let tl = trackLife.get(tid);
        if (!tl) {
          tl = { n: 0, minX: ix, maxX: ix, minY: iy, maxY: iy, prevX: ix, prevY: iy, prevVis: latestVis };
          trackLife.set(tid, tl);
        } else {
          if (latestVis && tl.prevVis && Math.hypot(ix - tl.prevX, iy - tl.prevY) > teleportStepMax) {
            teleportJumps++;
          }
          tl.minX = Math.min(tl.minX, ix);
          tl.maxX = Math.max(tl.maxX, ix);
          tl.minY = Math.min(tl.minY, iy);
          tl.maxY = Math.max(tl.maxY, iy);
          tl.prevX = ix;
          tl.prevY = iy;
          tl.prevVis = latestVis;
        }
        tl.n++;
      } else {
        for (let k = 0; k < TELEM_PHYSICS_STRIDE; k++) {
          if (telem[pbase + k] !== 0.0) inactiveNonZeroCount++;
        }
      }
    }

    const effK = Math.round(telem[1]);
    const fDisp = f - effK;
    if (fDisp >= 14 && gtByFrame.size > 0) {
      const gts = gtByFrame.get(fDisp) || [];
      gtTotal += gts.length;
      gtPredTotal += activeTracks.length;
      let matchedThisFrame = 0;
      if (gts.length > 0 && activeTracks.length > 0) {
        const pairs = [];
        for (let gi = 0; gi < gts.length; gi++) {
          for (let ti = 0; ti < activeTracks.length; ti++) {
            const d = Math.hypot(activeTracks[ti].fx - gts[gi].x, activeTracks[ti].fy - gts[gi].y);
            if (d <= gts[gi].rad) pairs.push({ gi, ti, d });
          }
        }
        pairs.sort((a, b) => a.d - b.d);
        const usedG = new Set();
        const usedT = new Set();
        for (const p of pairs) {
          if (usedG.has(p.gi) || usedT.has(p.ti)) continue;
          usedG.add(p.gi);
          usedT.add(p.ti);
          gtMatched++;
          matchedThisFrame++;
          gtSumSqErr += p.d * p.d;
          const gtId = gts[p.gi].gtId;
          const tid = activeTracks[p.ti].tid;
          if (lastTidForGt.has(gtId) && lastTidForGt.get(gtId) !== tid) gtIdSwaps++;
          lastTidForGt.set(gtId, tid);
          if (lastGtForTid.has(tid) && lastGtForTid.get(tid) !== gtId) gtHijacks++;
          lastGtForTid.set(tid, gtId);
        }
      }
      gtUnmatchedFp += activeTracks.length - matchedThisFrame;
    }
  });

  let staticPins = 0;
  for (const tl of trackLife.values()) {
    if (tl.n >= 30 && Math.hypot(tl.maxX - tl.minX, tl.maxY - tl.minY) < 4.5 * s) staticPins++;
  }
  const gtRecallPct = gtTotal > 0 ? (100.0 * gtMatched) / gtTotal : 100.0;
  const gtPrecisionPct = gtPredTotal > 0 ? (100.0 * gtMatched) / gtPredTotal : 100.0;
  const gtRmsePx = gtMatched > 0 ? Math.sqrt(gtSumSqErr / gtMatched) : 0.0;

  validFreqSamples.sort((a, b) => a - b);
  const medianFreqHz =
    validFreqSamples.length === 0
      ? 0.0
      : validFreqSamples.length % 2 === 1
      ? validFreqSamples[validFreqSamples.length >> 1]
      : 0.5 * (validFreqSamples[(validFreqSamples.length >> 1) - 1] + validFreqSamples[validFreqSamples.length >> 1]);

  const totalSpawned = trackAges.size;
  const minAge = clip.minPersistentAge || 15;
  const persistentTracks = [...trackAges.values()].filter((c) => c >= minAge).length;
  const maxTrackLife = totalSpawned > 0 ? Math.max(...trackAges.values()) : 0;
  const meanTickMs = sumTickMs / Math.max(1, numFrames);
  const warmMeanTickMs = numFrames > 1 ? warmSumTickMs / (numFrames - 1) : meanTickMs;
  tickSamples.sort((a, b) => a - b);
  const medianTickMs =
    tickSamples.length === 0
      ? 0.0
      : tickSamples.length % 2 === 1
      ? tickSamples[tickSamples.length >> 1]
      : 0.5 * (tickSamples[(tickSamples.length >> 1) - 1] + tickSamples[tickSamples.length >> 1]);
  const lockPct = (100.0 * framesWithTargets) / Math.max(1, numFrames);
  const padicSwaps = Math.round(telem[5]);
  const kalmanSwaps = Math.round(telem[6]);

  let dualResGravErr = 0.0;
  let dualResSummary = '';
  if (clip.checkDualResGravity) {
    const otherRes = await measureLockedSceneGravity(mp4Path, clip.fps, clip.labMode, hdMode ? 0 : 1);
    const gSd = hdMode ? otherRes.grav : lastSceneGrav;
    const gHd = hdMode ? lastSceneGrav : otherRes.grav;
    dualResGravErr = gSd > 1e-6 ? Math.abs(gHd / (2.0 * gSd) - 1.0) : 1.0;
    dualResSummary = ` | SD/HD 2x err=${(dualResGravErr * 100).toFixed(1)}% (g_sd=${gSd.toFixed(1)}, g_hd=${gHd.toFixed(1)})`;
  }

  const minMomResidStr = Number.isFinite(minPairMomResid) ? minPairMomResid.toFixed(3) : '--';
  console.log(`\n[${clip.file}] (${numFrames} frames @ ${clip.fps}fps) — ${clip.desc}`);
  console.log(`  Compute Tick  : warmMean=${warmMeanTickMs.toFixed(3)} ms (<${speedBudgetMs.toFixed(2)} ms) | p50=${medianTickMs.toFixed(3)} ms (rawMean=${meanTickMs.toFixed(3)} ms) | NaNs: ${nanDetected ? 'FAIL' : '0'} | tailZeros: ${tailNonZeroCount === 0 ? 'OK' : 'FAIL'} | max ||Lv-lv||: ${maxEigRes.toExponential(2)}`);
  console.log(`  ECA Clutter   : ${lastClutterDb.toFixed(2)} dB (peak ${maxClutterDb.toFixed(2)} dB, bg mask ${(lastEvalRatio * 100).toFixed(1)}%)`);
  console.log(`  Target Locks  : ${lockPct.toFixed(1)}% frames locked | avg ${(sumActiveTargets / numFrames).toFixed(2)} (max ${maxActiveTargets}) | spawned=${totalSpawned} (persistent>=${minAge}f: ${persistentTracks}, longest=${maxTrackLife}f)`);
  if (gtTotal > 0) {
    console.log(`  Ground Truth  : Recall=${gtRecallPct.toFixed(1)}% (${gtMatched}/${gtTotal}) | Prec=${gtPrecisionPct.toFixed(1)}% (FP=${gtUnmatchedFp}) | RMSE=${gtRmsePx.toFixed(2)} px | IDSW=${gtIdSwaps} | Hijacks=${gtHijacks} | Teleports=${teleportJumps} | StaticPins=${staticPins}`);
  }
  console.log(`  Features      : bridges=${bridgeFrames}f | apexFrames=${apexFrames}f | hilbertValid=${phaseValidFrames}f | freqValid=${freqValidFrames}f (med=${medianFreqHz.toFixed(2)} Hz) | percStrikes=${percTriggers} | 2adicAudit=${padicSwaps} vs kalmanSwaps=${kalmanSwaps}`);
  console.log(`  Kinematics    : g_hat=${lastSceneGrav.toFixed(1)} px/s² (conf=${maxGravConf.toFixed(2)}, arcs=${maxGravArcs})${dualResSummary} | virtualPivot=${pivotFrames}f | maxDeflect=${maxDeflectDeg.toFixed(1)}° | scatter=${lastScatterDeg.toFixed(1)}°`);
  console.log(`  Physics ODE   : regimes=[CV:${regimeFrames[0]}, BAL:${regimeFrames[1]}, DRAG:${regimeFrames[2]}, BST:${regimeFrames[3]}, CHT:${regimeFrames[4]}, PND:${regimeFrames[5]}, TRM:${regimeFrames[6]}, IMP:${regimeFrames[7]}, CTRV:${regimeFrames[8]}] | jumps=[peg:${jumpFrames[1]}, burn:${jumpFrames[2]}, chute:${jumpFrames[3]}, col:${jumpFrames[4]}] | ellipT=${posEllipticFrames}f Qpos=${posQualQFrames}f e_n=${validRestitutionFrames}f`);
  console.log(`  Ensemble      : kuramotoR=${maxKuramotoR.toFixed(3)} (N=${maxKuramotoCount}) | lyap=${maxLyapunov.toFixed(3)}/s | pairEvents=${pairEventCount} (bad=${invalidPairEvents}, minMomRes=${minMomResidStr}, maxMR=${maxPairMassRatio.toFixed(2)}) | taubinDelta=${maxTaubinDelta.toFixed(2)}px | inactiveNonZero=${inactiveNonZeroCount}`);

  if (strictGate) {
    const reasons = [];
    const minPersistent = clip.minPersistent ?? 1;
    const minLockPct = clip.minLockPct ?? 90.0;
    const minRecallPct = clip.minRecallPct ?? 78.0;
    const minPrecisionPct = clip.minPrecisionPct ?? 12.0;
    const minClutterDb = clip.minClutterDb ?? 35.0;
    const maxStaticPins = clip.maxStaticPins ?? 1;
    const minApex = hdMode ? (clip.minApexHd || 0) : (clip.minApexSd || 0);
    if (telem.length !== 1568) reasons.push(`telemLen=${telem.length}!==1568`);
    if (nanDetected) reasons.push('NaN/Inf in telemetry');
    if (tailNonZeroCount > 0) reasons.push(`reservedTailNonZero=${tailNonZeroCount}`);
    if (inactiveNonZeroCount > 0) reasons.push(`inactivePhysicsNonZero=${inactiveNonZeroCount}`);
    if (invalidRegimeCount > 0) reasons.push(`invalidRegime=${invalidRegimeCount}`);
    if (invalidJumpCount > 0) reasons.push(`invalidJump=${invalidJumpCount}`);
    if (maxEigRes > 1e-5) reasons.push(`maxEigRes=${maxEigRes.toExponential(2)}`);
    if (!noSpeedGate && warmMeanTickMs >= speedBudgetMs) {
      reasons.push(`warmMeanTickMs=${warmMeanTickMs.toFixed(3)}>=${speedBudgetMs.toFixed(2)}`);
    }
    if (persistentTracks < minPersistent) reasons.push(`persistentTracks=${persistentTracks}<${minPersistent}`);
    if (lockPct < minLockPct) reasons.push(`lockPct=${lockPct.toFixed(1)}<${minLockPct}`);
    if (lastClutterDb < minClutterDb) reasons.push(`clutterDb=${lastClutterDb.toFixed(2)}<${minClutterDb}`);
    if (kalmanSwaps > 0 ? padicSwaps >= kalmanSwaps : padicSwaps > 0) reasons.push(`swapOrdering(${padicSwaps} vs ${kalmanSwaps})`);
    if (apexFrames < minApex) reasons.push(`apexFrames=${apexFrames}<${minApex}`);
    if (bridgeFrames < (clip.minBridges || 0)) reasons.push(`bridgeFrames=${bridgeFrames}<${clip.minBridges}`);
    if (phaseValidFrames < (clip.minPhase || 0)) reasons.push(`phaseValidFrames=${phaseValidFrames}<${clip.minPhase}`);
    if (percTriggers < (clip.minPerc || 0)) reasons.push(`percTriggers=${percTriggers}<${clip.minPerc}`);
    if (clip.minPivotFrames !== undefined && pivotFrames < clip.minPivotFrames) {
      reasons.push(`pivotFrames=${pivotFrames}<${clip.minPivotFrames}`);
    }
    if (clip.maxPivotFrames !== undefined && pivotFrames > clip.maxPivotFrames) {
      reasons.push(`falsePivotFrames=${pivotFrames}>${clip.maxPivotFrames}`);
    }
    if (clip.minFreqValidFrames !== undefined && freqValidFrames < clip.minFreqValidFrames) {
      reasons.push(`freqValidFrames=${freqValidFrames}<${clip.minFreqValidFrames}`);
    }
    if (clip.maxFreqValidFrames !== undefined && freqValidFrames > clip.maxFreqValidFrames) {
      reasons.push(`falseFreqValidFrames=${freqValidFrames}>${clip.maxFreqValidFrames}`);
    }
    if (clip.freqRangeHz) {
      const [loHz, hiHz] = clip.freqRangeHz;
      if (medianFreqHz < loHz || medianFreqHz > hiHz) {
        reasons.push(`medianFreqHz=${medianFreqHz.toFixed(3)} outside [${loHz}, ${hiHz}]`);
      }
    }
    if (clip.minGravArcs !== undefined && maxGravArcs < clip.minGravArcs) {
      reasons.push(`gravArcs=${maxGravArcs}<${clip.minGravArcs}`);
    }
    if (clip.minGravConf !== undefined && maxGravConf < clip.minGravConf) {
      reasons.push(`gravConf=${maxGravConf.toFixed(2)}<${clip.minGravConf}`);
    }
    if (clip.checkDualResGravity && dualResGravErr > 0.16) {
      reasons.push(`dualResGravErr=${(dualResGravErr * 100).toFixed(1)}%>16%`);
    }
    if (clip.physicsNegativeGate) {
      if (regimeFrames[3] > 0) reasons.push(`falseBoost=${regimeFrames[3]}`);
      if (regimeFrames[4] > 0) reasons.push(`falseParachute=${regimeFrames[4]}`);
      if (regimeFrames[5] > 0) reasons.push(`falsePendulum=${regimeFrames[5]}`);
      if (jumpFrames[1] > 0) reasons.push(`falsePegJump=${jumpFrames[1]}`);
      if (jumpFrames[4] > 0 || maxLastPairFrame > 0) reasons.push(`falseCollision(j4=${jumpFrames[4]},pairF=${maxLastPairFrame})`);
      if (maxKuramotoR > 0.0) reasons.push(`falseKuramotoR=${maxKuramotoR.toFixed(3)}`);
      if (maxLyapunov > 0.0) reasons.push(`falseLyapunov=${maxLyapunov.toFixed(3)}`);
    }
    if (clip.requirePendulumPhysics) {
      if (regimeFrames[5] < 1) reasons.push('missingPendulumRegime5');
      if (jumpFrames[1] < 1) reasons.push('missingGalileoPegJump1');
      if (posEllipticFrames < 1) reasons.push('missingEllipticPeriod');
      if (posQualQFrames < 1) reasons.push('missingQualityFactorQ');
    }
    if (clip.requireCollisionPhysics) {
      if (jumpFrames[4] < 1) reasons.push('missingCollisionJump4');
      if (pairEventCount < 1) reasons.push('missingCollisionPairEvent');
      if (validRestitutionFrames < 1) reasons.push('missingValidRestitutionE');
      if (invalidRestitutionFrames > 0) reasons.push(`invalidRestitutionE=${invalidRestitutionFrames}`);
      if (invalidPairEvents > 0) reasons.push(`invalidPairEvents=${invalidPairEvents}`);
    }
    if (clip.requireImpulsivePhysics) {
      if (regimeFrames[7] < 1) reasons.push('missingImpulsiveRegime7');
      if (regimeFrames[3] > 0) reasons.push(`falseBoost=${regimeFrames[3]}`);
      if (regimeFrames[4] > 0) reasons.push(`falseParachute=${regimeFrames[4]}`);
      if (regimeFrames[5] > 0) reasons.push(`falsePendulum=${regimeFrames[5]}`);
      if (maxKuramotoR > 0.0) reasons.push(`falseKuramotoR=${maxKuramotoR.toFixed(3)}`);
      if (maxLyapunov > 0.0) reasons.push(`falseLyapunov=${maxLyapunov.toFixed(3)}`);
    }
    if (clip.minBoostFrames !== undefined && regimeFrames[3] < clip.minBoostFrames) {
      reasons.push(`boostFrames=${regimeFrames[3]}<${clip.minBoostFrames}`);
    }
    if (clip.minBurnoutJumpFrames !== undefined && jumpFrames[2] < clip.minBurnoutJumpFrames) {
      reasons.push(`burnoutJump2Frames=${jumpFrames[2]}<${clip.minBurnoutJumpFrames}`);
    }
    if (clip.minParachuteFrames !== undefined && regimeFrames[4] < clip.minParachuteFrames) {
      reasons.push(`parachuteFrames=${regimeFrames[4]}<${clip.minParachuteFrames}`);
    }
    if (clip.minParachuteJumpFrames !== undefined && jumpFrames[3] < clip.minParachuteJumpFrames) {
      reasons.push(`parachuteJump3Frames=${jumpFrames[3]}<${clip.minParachuteJumpFrames}`);
    }
    if (clip.minChuteOrDragFrames !== undefined) {
      const chuteOrDrag = regimeFrames[4] + regimeFrames[2] + jumpFrames[3];
      if (chuteOrDrag < clip.minChuteOrDragFrames) {
        reasons.push(`chuteOrDragFrames=${chuteOrDrag}<${clip.minChuteOrDragFrames}`);
      }
    }
    if (clip.minLyapunovMax !== undefined && maxLyapunov <= clip.minLyapunovMax) {
      reasons.push(`lyapunovMax=${maxLyapunov.toFixed(3)}<=${clip.minLyapunovMax}`);
    }
    if (clip.minKuramotoOscCount !== undefined && maxKuramotoCount < clip.minKuramotoOscCount) {
      reasons.push(`kuramotoCount=${maxKuramotoCount}<${clip.minKuramotoOscCount}`);
    }
    if (clip.minKuramotoOrderR !== undefined && maxKuramotoR < clip.minKuramotoOrderR) {
      reasons.push(`kuramotoOrderR=${maxKuramotoR.toFixed(3)}<${clip.minKuramotoOrderR}`);
    }
    if (clip.hasGt !== false) {
      if (gtTotal <= 0) reasons.push('gtTotal===0');
      if (gtRecallPct < minRecallPct) reasons.push(`gtRecall=${gtRecallPct.toFixed(1)}%<${minRecallPct}%`);
      if (gtPrecisionPct < minPrecisionPct) reasons.push(`gtPrecision=${gtPrecisionPct.toFixed(1)}%<${minPrecisionPct}%`);
      if (gtHijacks > 16) reasons.push(`gtHijacks=${gtHijacks}`);
    }
    if (teleportJumps > 0) reasons.push(`teleportJumps=${teleportJumps}`);
    if (staticPins > maxStaticPins) reasons.push(`staticPins=${staticPins}>${maxStaticPins}`);

    if (reasons.length > 0) {
      allPassed = false;
      console.error(`  [FAIL] Calibration invariant missed on ${clip.file}: ${reasons.join(', ')}`);
    } else {
      console.log(`  [PASS] All calibration & physics-proof invariants verified on ${clip.file}`);
    }
  }
}

if (strictGate && !allPassed) {
  process.exit(1);
}
