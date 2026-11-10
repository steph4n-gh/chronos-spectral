#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const WASM_PATH = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');

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

function decodeMp4Frames(mp4Path, w, h) {
  const raw = execFileSync(
    'ffmpeg',
    ['-v', 'error', '-i', mp4Path, '-vf', `scale=${w}:${h}`, '-pix_fmt', 'rgba', '-f', 'rawvideo', 'pipe:1'],
    { maxBuffer: 1024 * 1024 * 1024 }
  );
  const frameBytes = w * h * 4;
  const numFrames = Math.floor(raw.length / frameBytes);
  return { raw, frameBytes, numFrames };
}

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
      return 0.0;
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
      let beta = 0.0;
      for (let a = 0; a < n; a++) beta += lanczosW[a] * lanczosW[a];
      beta = Math.sqrt(beta);
      actualM = j + 1;

      if (j + 1 < mSteps) {
        const nextOff = (j + 1) * GRAPH_NODES;
        if (beta < 1e-7) {
          lanczosBeta[j] = 0.0;
          if (!findOrthogonalBasisSeed(n, actualM, nextOff)) break;
        } else {
          lanczosBeta[j] = beta;
          const invB = 1.0 / beta;
          for (let a = 0; a < n; a++) lanczosQ[nextOff + a] = lanczosW[a] * invB;
        }
      } else {
        lanczosBeta[j] = 0.0;
      }
    }

    const m = actualM;
    jacobiA.fill(0.0);
    jacobiV.fill(0.0);
    for (let r = 0; r < m; r++) {
      jacobiV[r * MAX_KRYLOV + r] = 1.0;
      jacobiA[r * MAX_KRYLOV + r] = lanczosAlpha[r];
      if (r + 1 < m) {
        jacobiA[r * MAX_KRYLOV + (r + 1)] = lanczosBeta[r];
        jacobiA[(r + 1) * MAX_KRYLOV + r] = lanczosBeta[r];
      }
    }
    for (let sweep = 0; sweep < 24; sweep++) {
      let maxOff = 0.0;
      let p = 0, q = 1;
      for (let r = 0; r < m; r++) {
        for (let c = r + 1; c < m; c++) {
          const v = Math.abs(jacobiA[r * MAX_KRYLOV + c]);
          if (v > maxOff) { maxOff = v; p = r; q = c; }
        }
      }
      if (maxOff < 1e-11) break;
      const app = jacobiA[p * MAX_KRYLOV + p];
      const aqq = jacobiA[q * MAX_KRYLOV + q];
      const apq = jacobiA[p * MAX_KRYLOV + q];
      const tau = (aqq - app) / (2.0 * apq);
      const t = tau >= 0.0 ? 1.0 / (tau + Math.hypot(1.0, tau)) : -1.0 / (-tau + Math.hypot(1.0, tau));
      const cs = 1.0 / Math.hypot(1.0, t);
      const sn = t * cs;
      for (let k = 0; k < m; k++) {
        if (k !== p && k !== q) {
          const akp = jacobiA[k * MAX_KRYLOV + p];
          const akq = jacobiA[k * MAX_KRYLOV + q];
          jacobiA[k * MAX_KRYLOV + p] = cs * akp - sn * akq;
          jacobiA[p * MAX_KRYLOV + k] = jacobiA[k * MAX_KRYLOV + p];
          jacobiA[k * MAX_KRYLOV + q] = sn * akp + cs * akq;
          jacobiA[q * MAX_KRYLOV + k] = jacobiA[k * MAX_KRYLOV + q];
        }
      }
      jacobiA[p * MAX_KRYLOV + p] = cs * cs * app - 2.0 * sn * cs * apq + sn * sn * aqq;
      jacobiA[q * MAX_KRYLOV + q] = sn * sn * app + 2.0 * sn * cs * apq + cs * cs * aqq;
      jacobiA[p * MAX_KRYLOV + q] = 0.0;
      jacobiA[q * MAX_KRYLOV + p] = 0.0;
      for (let k = 0; k < m; k++) {
        const vkp = jacobiV[k * MAX_KRYLOV + p];
        const vkq = jacobiV[k * MAX_KRYLOV + q];
        jacobiV[k * MAX_KRYLOV + p] = cs * vkp - sn * vkq;
        jacobiV[k * MAX_KRYLOV + q] = sn * vkp + cs * vkq;
      }
    }

    let bestIdx = 0;
    let bestTheta = -1e30;
    let secondTheta = -1e30;
    for (let r = 0; r < m; r++) {
      const th = jacobiA[r * MAX_KRYLOV + r];
      if (th > bestTheta) { secondTheta = bestTheta; bestTheta = th; bestIdx = r; }
      else if (th > secondTheta) { secondTheta = th; }
    }
    const lambda2 = Math.max(0.0, mu - bestTheta);
    const lambda3 = secondTheta > -1e20 ? Math.max(lambda2, mu - secondTheta) : lambda2;
    const spectralGap = Math.max(0.0, lambda3 - lambda2);

    let vNorm2 = 0.0;
    for (let a = 0; a < n; a++) {
      let s = 0.0;
      for (let j = 0; j < m; j++) s += lanczosQ[j * GRAPH_NODES + a] * jacobiV[j * MAX_KRYLOV + bestIdx];
      eigVecBuf[a] = s;
      vNorm2 += s * s;
    }
    const invV = 1.0 / Math.sqrt(Math.max(vNorm2, 1e-18));
    let temporalDot = 0.0;
    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a];
      const val = eigVecBuf[a] * invV;
      eigVecBuf[a] = val;
      if (ia < MAX_TARGETS) temporalDot += val * prevFiedlerTrack[ia];
    }
    const sign = temporalDot < -1e-6 ? -1.0 : 1.0;
    for (let i = 0; i < MAX_TARGETS; i++) prevFiedlerTrack[i] = 0.0;
    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a];
      const val = sign * eigVecBuf[a];
      fiedler[ia] = val;
      if (ia < MAX_TARGETS) prevFiedlerTrack[ia] = val;
    }
    exp.chronos_set_fiedler_spectrum(lambda2, spectralGap);
    return lambda2;
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

  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, clip.labMode);

  const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
  const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 256);
  const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 16);
  const solveLanczos = createLanczosSolver(exp, laplacian, fiedler);

  const { raw, frameBytes, numFrames } = decodeMp4Frames(mp4Path, w, h);
  const dtSec = 1.0 / clip.fps;

  const trackAges = new Map();
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
  let nanDetected = false;

  for (let f = 0; f < numFrames; f++) {
    const ingestPtr = exp.chronos_get_ingest_rgba_ptr();
    const dst = new Uint8Array(mem.buffer, ingestPtr, frameBytes);
    dst.set(raw.subarray(f * frameBytes, (f + 1) * frameBytes));

    const t0 = performance.now();
    exp.chronos_pre_tick(dtSec);
    solveLanczos();
    exp.chronos_post_tick(dtSec);
    sumTickMs += performance.now() - t0;

    for (let i = 0; i < 32; i++) {
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

    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      if (telem[base] > 0.5) {
        const tid = Math.round(telem[base + 1]);
        trackAges.set(tid, (trackAges.get(tid) || 0) + 1);
        if (telem[base + 11] > 0.5) bridgeFrames++;
        if (telem[base + 17] > 1.0) apexFrames++;
      }
    }
  }

  const totalSpawned = trackAges.size;
  const persistentTracks = [...trackAges.values()].filter((c) => c >= 15).length;
  const maxTrackLife = totalSpawned > 0 ? Math.max(...trackAges.values()) : 0;
  const meanTickMs = sumTickMs / numFrames;
  const lockPct = (100.0 * framesWithTargets) / numFrames;
  const padicSwaps = Math.round(telem[5]);
  const kalmanSwaps = Math.round(telem[6]);

  console.log(`\n[${clip.file}] (${numFrames} frames @ ${clip.fps}fps) — ${clip.desc}`);
  console.log(`  Compute Tick  : ${meanTickMs.toFixed(3)} ms/frame | NaNs: ${nanDetected ? 'FAIL' : '0'}`);
  console.log(`  ECA Clutter   : ${lastClutterDb.toFixed(2)} dB (peak ${maxClutterDb.toFixed(2)} dB, bg mask ${(lastEvalRatio * 100).toFixed(1)}%)`);
  console.log(`  Target Locks  : ${lockPct.toFixed(1)}% frames locked | avg ${(sumActiveTargets / numFrames).toFixed(2)} (max ${maxActiveTargets}) | spawned=${totalSpawned} (persistent>=15f: ${persistentTracks}, longest=${maxTrackLife}f)`);
  console.log(`  Features      : bridges=${bridgeFrames}f | apexFrames=${apexFrames}f | hilbertValid=${phaseValidFrames}f | percStrikes=${percTriggers} | 2adicSwaps=${padicSwaps} vs kalmanSwaps=${kalmanSwaps}`);

  if (strictGate) {
    if (nanDetected || persistentTracks < 1 || lockPct < 50.0) {
      allPassed = false;
      console.error(`  [FAIL] Calibration invariant missed on ${clip.file}`);
    }
  }
}

if (strictGate && !allPassed) {
  process.exit(1);
}
