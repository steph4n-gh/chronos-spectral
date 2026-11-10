#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Streaming Real-Camera MP4 Calibration & Diagnostic Suite
// Streams raw RGBA frames one-by-one from ffmpeg (O(1) single-frame memory,
// zero whole-video buffering) through the exact production Wasm + 20-sweep
// Jacobi Lanczos solver, verifying 576-float finiteness, eigen-residuals,
// ECA clutter suppression, persistent lock stability, and swap ordering.
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

const GRAPH_NODES = 16;
const MAX_TARGETS = 8;
const MAX_KRYLOV = 15;
const TELEMETRY_LEN = 576;

const CLIPS = [
  { file: 'juggling_ballistic.mp4', fps: 60, labMode: 0, desc: '60fps Chalkboard Juggling (Ballistic + Color)' },
  { file: 'pets_pedestrian_occlusion.mp4', fps: 10, labMode: 0, desc: 'PETS2009 Outdoor Crossing & Sign Occlusion' },
  { file: 'newton_cradle_impulses.mp4', fps: 60, labMode: 2, desc: '60fps Steel Newton Cradle (L1 Impulses)' },
  { file: 'cup_hand_tremor.mp4', fps: 26.78, labMode: 1, desc: 'Handheld Cup (Postural Tremor + Wall AE)' },
  { file: 'webcam_book_occlusion.mp4', fps: 25, labMode: 0, desc: 'Grayscale Webcam Face + Book Occlusion' },
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

console.log(`========================================================================`);
console.log(` CHRONOS-SPECTRAL // Real-Camera MP4 Calibration Suite (${hdMode ? '640x360 HD' : '320x180 SD'})`);
console.log(`========================================================================`);

let allPassed = true;

for (const clip of CLIPS) {
  const mp4Path = path.join(SAMPLES_DIR, clip.file);
  if (!fs.existsSync(mp4Path)) {
    throw new Error(`Missing sample video: ${mp4Path}`);
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
  const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 256);
  const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 16);
  const ringViews = Array.from({ length: 12 }, (_, s) => new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(s), frameBytes));
  const solveLanczos = createLanczosSolver(exp, laplacian, fiedler);
  const dtSec = 1.0 / clip.fps;
  const s = w / 320.0;
  const gtScale = w / 640.0;

  const gtByFrame = new Map();
  if (gtDataset && gtDataset[clip.file]) {
    for (const tr of gtDataset[clip.file].tracks) {
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
  let gtSumSqErr = 0.0;
  let gtIdSwaps = 0;
  let gtHijacks = 0;
  let teleportJumps = 0;
  let sumTickMs = 0.0;
  let maxClutterDb = 0.0;
  let lastClutterDb = 0.0;
  let lastEvalRatio = 0.0;
  let framesWithTargets = 0;
  let sumActiveTargets = 0;
  let maxActiveTargets = 0;
  let bridgeFrames = 0;
  let apexFrames = 0;
  let phaseValidFrames = 0;
  let percTriggers = 0;
  let maxEigRes = 0.0;
  let nanDetected = false;

  const { numFrames } = await streamMp4Frames(mp4Path, w, h, (frameRgba, f) => {
    ringViews[f % 12].set(frameRgba);

    const t0 = performance.now();
    exp.chronos_pre_tick(dtSec);
    const { resNorm } = solveLanczos();
    exp.chronos_post_tick(dtSec);
    sumTickMs += performance.now() - t0;

    if (resNorm > maxEigRes) maxEigRes = resNorm;
    for (let i = 0; i < TELEMETRY_LEN; i++) {
      if (!Number.isFinite(telem[i])) nanDetected = true;
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
    if (Math.round(telem[15]) !== 0) percTriggers++;

    const activeTracks = [];
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      if (telem[base] > 0.5) {
        const tid = Math.round(telem[base + 1]);
        const fx = telem[base + 2];
        const fy = telem[base + 3];
        const ix = telem[base + 8];
        const iy = telem[base + 9];
        const latestVis = telem[base + 44 + 11] > 0.5;
        trackAges.set(tid, (trackAges.get(tid) || 0) + 1);
        if (telem[base + 11] > 0.5) bridgeFrames++;
        if (telem[base + 17] > 1.0) apexFrames++;
        activeTracks.push({ m, tid, fx, fy });

        let tl = trackLife.get(tid);
        if (!tl) {
          tl = { n: 0, minX: ix, maxX: ix, minY: iy, maxY: iy, prevX: ix, prevY: iy, prevVis: latestVis };
          trackLife.set(tid, tl);
        } else {
          if (latestVis && tl.prevVis && Math.hypot(ix - tl.prevX, iy - tl.prevY) > 42.0 * s) {
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
      }
    }

    const effK = Math.round(telem[1]);
    const fDisp = f - effK;
    if (fDisp >= 14 && gtByFrame.size > 0) {
      const gts = gtByFrame.get(fDisp) || [];
      gtTotal += gts.length;
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
          gtSumSqErr += p.d * p.d;
          const gtId = gts[p.gi].gtId;
          const tid = activeTracks[p.ti].tid;
          if (lastTidForGt.has(gtId) && lastTidForGt.get(gtId) !== tid) gtIdSwaps++;
          lastTidForGt.set(gtId, tid);
          if (lastGtForTid.has(tid) && lastGtForTid.get(tid) !== gtId) gtHijacks++;
          lastGtForTid.set(tid, gtId);
        }
      }
    }
  });

  let staticPins = 0;
  for (const tl of trackLife.values()) {
    if (tl.n >= 30 && Math.hypot(tl.maxX - tl.minX, tl.maxY - tl.minY) < 4.5 * s) staticPins++;
  }
  const gtRecallPct = gtTotal > 0 ? (100.0 * gtMatched) / gtTotal : 100.0;
  const gtRmsePx = gtMatched > 0 ? Math.sqrt(gtSumSqErr / gtMatched) : 0.0;

  const totalSpawned = trackAges.size;
  const persistentTracks = [...trackAges.values()].filter((c) => c >= 15).length;
  const maxTrackLife = totalSpawned > 0 ? Math.max(...trackAges.values()) : 0;
  const meanTickMs = sumTickMs / Math.max(1, numFrames);
  const lockPct = (100.0 * framesWithTargets) / Math.max(1, numFrames);
  const padicSwaps = Math.round(telem[5]);
  const kalmanSwaps = Math.round(telem[6]);

  console.log(`\n[${clip.file}] (${numFrames} frames @ ${clip.fps}fps) — ${clip.desc}`);
  console.log(`  Compute Tick  : ${meanTickMs.toFixed(3)} ms/frame | NaNs: ${nanDetected ? 'FAIL' : '0'} | max ||Lv-lv||: ${maxEigRes.toExponential(2)}`);
  console.log(`  ECA Clutter   : ${lastClutterDb.toFixed(2)} dB (peak ${maxClutterDb.toFixed(2)} dB, bg mask ${(lastEvalRatio * 100).toFixed(1)}%)`);
  console.log(`  Target Locks  : ${lockPct.toFixed(1)}% frames locked | avg ${(sumActiveTargets / numFrames).toFixed(2)} (max ${maxActiveTargets}) | spawned=${totalSpawned} (persistent>=15f: ${persistentTracks}, longest=${maxTrackLife}f)`);
  if (gtTotal > 0) {
    console.log(`  Ground Truth  : Recall=${gtRecallPct.toFixed(1)}% (${gtMatched}/${gtTotal}) | RMSE=${gtRmsePx.toFixed(2)} px | IDSW=${gtIdSwaps} | Hijacks=${gtHijacks} | Teleports=${teleportJumps} | StaticPins=${staticPins}`);
  }
  console.log(`  Features      : bridges=${bridgeFrames}f | apexFrames=${apexFrames}f | hilbertValid=${phaseValidFrames}f | percStrikes=${percTriggers} | 2adicAudit=${padicSwaps} vs kalmanSwaps=${kalmanSwaps}`);

  if (strictGate) {
    const minPersistent =
      gtDataset && gtDataset[clip.file]
        ? Math.min(3, gtDataset[clip.file].tracks.length)
        : 3;
    const swapFailed = kalmanSwaps > 0 ? padicSwaps >= kalmanSwaps : padicSwaps > 0;
    const gtFailed =
      gtTotal > 0 &&
      (gtRecallPct < 78.0 || teleportJumps > 0 || staticPins > 1 || gtHijacks > 16);
    if (
      nanDetected ||
      maxEigRes > 1e-5 ||
      persistentTracks < minPersistent ||
      lockPct < 90.0 ||
      lastClutterDb < 44.0 ||
      swapFailed ||
      gtFailed
    ) {
      allPassed = false;
      console.error(`  [FAIL] Calibration invariant missed on ${clip.file}`);
    }
  }
}

if (strictGate && !allPassed) {
  process.exit(1);
}
