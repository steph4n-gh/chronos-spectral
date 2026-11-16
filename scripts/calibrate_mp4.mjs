#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Streaming Real-Camera MP4 Calibration & Diagnostic Suite
// Streams raw RGBA frames one-by-one from ffmpeg (O(1) single-frame memory,
// zero whole-video frames[] buffering even across multi-trial dual-mode checks)
// through the exact production Wasm + 20-sweep Jacobi Lanczos solver, verifying
// 1568-float finiteness, reserved-tail zeros, eigen-residuals, ECA clutter
// suppression, CLEAR-MOT recall & precision, multi-arc Auto-g 2x SD/HD consistency,
// Kåsa pivot gating, harmonic phase, and Wave-2 physics/continuum invariants.
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import v8 from 'node:v8';
import vm from 'node:vm';

const runExplicitGc = (() => {
  try {
    v8.setFlagsFromString('--expose_gc');
    const fn = vm.runInNewContext('gc');
    return typeof fn === 'function' ? fn : () => {};
  } catch {
    return () => {};
  }
})();

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

// Dense 1,000+ target sample kept cleanly separate from the 13-clip legacy regression denominator.
const SWARM_CLIPS = [
  {
    file: 'starling_murmuration_1000.mp4',
    fps: 30,
    labMode: 0,
    hasGt: false,
    expectDenseSwarm: true,
    minLockPct: 85.0,
    minClutterDb: 35.0,
    minPersistentAge: 15,
    minPersistent: 4,
    maxStaticPins: 8,
    maxTickMsSd: 0.80,
    maxTickMsHd: 2.00,
    desc: '30fps Real Dense Starling Murmuration (1,000+ Moving Targets, Continuum Flow & Swarm Tracking)',
  },
];

function solveHungarianMinCost(cost, n) {
  const INF = 1e18;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  const minv = new Float64Array(n + 1);
  const used = new Uint8Array(n + 1);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    minv.fill(INF);
    used.fill(0);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      const rowOff = (i0 - 1) * n;
      for (let j = 1; j <= n; j++) {
        if (!used[j]) {
          const cur = cost[rowOff + (j - 1)] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const rowToCol = new Int32Array(n).fill(-1);
  for (let j = 1; j <= n; j++) {
    if (p[j] >= 1 && p[j] <= n) {
      rowToCol[p[j] - 1] = j - 1;
    }
  }
  return rowToCol;
}

function matchClearMotContinuity(gts, activeTracks, prevFrameGtToTid) {
  const matches = [];
  if (gts.length === 0 || activeTracks.length === 0) return matches;

  const usedG = new Uint8Array(gts.length);
  const usedT = new Uint8Array(activeTracks.length);
  const tidToTi = new Map();
  for (let ti = 0; ti < activeTracks.length; ti++) {
    if (!tidToTi.has(activeTracks[ti].tid)) tidToTi.set(activeTracks[ti].tid, ti);
  }

  // (a) Carry over previous-frame (gtId, tid) matches if still within gts[gi].rad
  for (let gi = 0; gi < gts.length; gi++) {
    const prevTid = prevFrameGtToTid.get(gts[gi].gtId);
    if (prevTid === undefined) continue;
    const ti = tidToTi.get(prevTid);
    if (ti === undefined || usedT[ti]) continue;
    const d = Math.hypot(activeTracks[ti].fx - gts[gi].x, activeTracks[ti].fy - gts[gi].y);
    if (d <= gts[gi].rad) {
      usedG[gi] = 1;
      usedT[ti] = 1;
      matches.push({ gi, ti, d });
    }
  }

  // (b) Exact optimal minimum-cost bipartite matching over remaining unmatched gts x activeTracks
  const remG = [];
  const remT = [];
  let maxRad = 1.0;
  for (let gi = 0; gi < gts.length; gi++) {
    if (!usedG[gi]) {
      remG.push(gi);
      if (gts[gi].rad > maxRad) maxRad = gts[gi].rad;
    }
  }
  for (let ti = 0; ti < activeTracks.length; ti++) {
    if (!usedT[ti]) remT.push(ti);
  }

  const nG = remG.length;
  const nT = remT.length;
  if (nG > 0 && nT > 0) {
    const n = Math.max(nG, nT);
    const sentinelCost = Math.max(1e6, (Math.min(nG, nT) + 1) * (maxRad + 1.0) * 100.0);
    const cost = new Float64Array(n * n);
    cost.fill(sentinelCost);
    for (let i = 0; i < nG; i++) {
      const g = gts[remG[i]];
      for (let j = 0; j < nT; j++) {
        const t = activeTracks[remT[j]];
        const d = Math.hypot(t.fx - g.x, t.fy - g.y);
        if (d <= g.rad) cost[i * n + j] = d;
      }
    }
    const rowToCol = solveHungarianMinCost(cost, n);
    for (let i = 0; i < nG; i++) {
      const j = rowToCol[i];
      if (j >= 0 && j < nT) {
        const d = cost[i * n + j];
        if (d < sentinelCost * 0.5 && d <= gts[remG[i]].rad) {
          matches.push({ gi: remG[i], ti: remT[j], d });
        }
      }
    }
  }

  return matches;
}

/**
 * Reference-conformant 19-threshold TrackEval HOTA (Luiten et al., IJCV 2021)
 * using normalized global similarity IoU accumulation and global_alignment * sim assignment.
 * Point localization similarity: S(g, t) = max(0, 1 - dist / (2 * g.rad))
 * (so S >= 0.5 corresponds to dist <= g.rad).
 */
function computeTrackEvalHota(evalFrames) {
  let totalGt = 0;
  let totalPred = 0;
  const gtTotalFrames = new Map();
  const trTotalFrames = new Map();

  for (let f = 0; f < evalFrames.length; f++) {
    const { gts, tracks } = evalFrames[f];
    totalGt += gts.length;
    totalPred += tracks.length;
    for (let gi = 0; gi < gts.length; gi++) {
      gtTotalFrames.set(gts[gi].gtId, (gtTotalFrames.get(gts[gi].gtId) || 0) + 1);
    }
    for (let ti = 0; ti < tracks.length; ti++) {
      trTotalFrames.set(tracks[ti].tid, (trTotalFrames.get(tracks[ti].tid) || 0) + 1);
    }
  }
  if (totalGt === 0) return null;
  if (totalPred === 0) return { hota: 0.0, detA: 0.0, assA: 0.0, locA: 0.0 };

  // Stage 1: Accumulate normalized potential matches across all frames (TrackEval sim_iou)
  const potentialMatches = new Map();
  const frameSims = new Array(evalFrames.length);
  for (let f = 0; f < evalFrames.length; f++) {
    const { gts, tracks } = evalFrames[f];
    const nG = gts.length;
    const nT = tracks.length;
    if (nG === 0 || nT === 0) {
      frameSims[f] = null;
      continue;
    }
    const rSum = new Float64Array(nG);
    const cSum = new Float64Array(nT);
    const gEdges = Array.from({ length: nG }, () => []);
    for (let gi = 0; gi < nG; gi++) {
      const g = gts[gi];
      const invMaxDist = 1.0 / Math.max(1e-6, 2.0 * g.rad);
      for (let ti = 0; ti < nT; ti++) {
        const t = tracks[ti];
        const tx = t.fx ?? t.x;
        const ty = t.fy ?? t.y;
        const d = Math.hypot(tx - g.x, ty - g.y);
        const sim = Math.max(0.0, 1.0 - d * invMaxDist);
        if (sim > 0.0) {
          gEdges[gi].push({ gi, ti, sim });
          rSum[gi] += sim;
          cSum[ti] += sim;
        }
      }
    }
    for (let gi = 0; gi < nG; gi++) {
      const gid = gts[gi].gtId;
      for (let k = 0; k < gEdges[gi].length; k++) {
        const e = gEdges[gi][k];
        const denom = rSum[gi] + cSum[e.ti] - e.sim;
        const simIou = denom > 0.0 ? e.sim / denom : 0.0;
        if (simIou > 0.0) {
          const key = `${gid}:${tracks[e.ti].tid}`;
          potentialMatches.set(key, (potentialMatches.get(key) || 0.0) + simIou);
        }
      }
    }
    frameSims[f] = gEdges;
  }

  const globalAlign = new Map();
  for (const [key, pot] of potentialMatches.entries()) {
    const colon = key.indexOf(':');
    const gid = Number(key.slice(0, colon));
    const tid = Number(key.slice(colon + 1));
    const gTot = gtTotalFrames.get(gid) || 0;
    const tTot = trTotalFrames.get(tid) || 0;
    const denom = gTot + tTot - pot;
    globalAlign.set(key, denom > 0.0 ? pot / denom : 0.0);
  }

  // Stage 2: Thresholded assignment per alpha maximizing sum(global_alignment * similarity)
  let sumHota = 0.0;
  let sumDetA = 0.0;
  let sumAssA = 0.0;
  let sumLocA = 0.0;
  let locCount = 0;
  const numAlphas = 19;

  for (let ai = 1; ai <= numAlphas; ai++) {
    const alpha = ai * 0.05;
    let tp = 0;
    let simSum = 0.0;
    const tpaMap = new Map();
    const matchedPairs = [];

    for (let f = 0; f < evalFrames.length; f++) {
      const gEdges = frameSims[f];
      if (!gEdges) continue;
      const { gts, tracks } = evalFrames[f];
      const gUsedIdx = [];
      const gMap = new Map();
      const tUsedIdx = [];
      const tMap = new Map();
      for (let gi = 0; gi < gEdges.length; gi++) {
        const edges = gEdges[gi];
        for (let k = 0; k < edges.length; k++) {
          if (edges[k].sim < alpha - 1e-12) continue;
          if (!gMap.has(gi)) {
            gMap.set(gi, gUsedIdx.length);
            gUsedIdx.push(gi);
          }
          const ti = edges[k].ti;
          if (!tMap.has(ti)) {
            tMap.set(ti, tUsedIdx.length);
            tUsedIdx.push(ti);
          }
        }
      }
      const nG = gUsedIdx.length;
      const nT = tUsedIdx.length;
      if (nG === 0 || nT === 0) continue;
      const n = Math.max(nG, nT);
      const BASE_UNMATCHED = 2.0;
      const cost = new Float64Array(n * n).fill(BASE_UNMATCHED);
      const edgeLookup = new Map();
      for (let r = 0; r < nG; r++) {
        const gi = gUsedIdx[r];
        const gtId = gts[gi].gtId;
        for (const e of gEdges[gi]) {
          if (e.sim < alpha - 1e-12) continue;
          const c = tMap.get(e.ti);
          const key = `${gtId}:${tracks[e.ti].tid}`;
          const align = globalAlign.get(key) || 0.0;
          const score = align * e.sim;
          if (score > 0.0) {
            cost[r * n + c] = BASE_UNMATCHED - score;
            edgeLookup.set(r * n + c, { gtId, tid: tracks[e.ti].tid, sim: e.sim, key });
          }
        }
      }
      const rowToCol = solveHungarianMinCost(cost, n);
      for (let r = 0; r < nG; r++) {
        const c = rowToCol[r];
        if (c >= 0 && c < nT) {
          const item = edgeLookup.get(r * n + c);
          if (item) {
            tp++;
            simSum += item.sim;
            tpaMap.set(item.key, (tpaMap.get(item.key) || 0) + 1);
            matchedPairs.push(item);
          }
        }
      }
    }

    if (tp > 0) {
      const fn = totalGt - tp;
      const fp = totalPred - tp;
      const detA = tp / Math.max(1, tp + fp + fn);
      let assASum = 0.0;
      for (let i = 0; i < matchedPairs.length; i++) {
        const { gtId, tid, key } = matchedPairs[i];
        const tpa = tpaMap.get(key) || 1;
        const gTot = gtTotalFrames.get(gtId) || tpa;
        const tTot = trTotalFrames.get(tid) || tpa;
        assASum += tpa / Math.max(1, gTot + tTot - tpa);
      }
      const assA = assASum / tp;
      const locA = simSum / tp;
      sumDetA += detA;
      sumAssA += assA;
      sumLocA += locA;
      locCount++;
      sumHota += Math.sqrt(detA * assA);
    }
  }

  return {
    hota: sumHota / numAlphas,
    detA: sumDetA / numAlphas,
    assA: sumAssA / numAlphas,
    locA: locCount > 0 ? sumLocA / locCount : 0.0,
  };
}

/** Validates Wasm binary isolation, pointer 4-byte alignment, bounds, and disjointness. */
function validateWasmModuleAndPointers(wasmModule, exp, mem, frameBytes, checkSwarm) {
  const errs = [];
  if (WebAssembly.Module.imports(wasmModule).length !== 0) {
    errs.push(`wasmImports=${WebAssembly.Module.imports(wasmModule).length}!==0`);
  }
  const memBytes = mem.buffer.byteLength;
  if (checkSwarm) {
    const expectedWave2Bytes = 320 * 65536;
    if (memBytes !== expectedWave2Bytes) {
      errs.push(`wave2MemoryBytes=${memBytes}!==${expectedWave2Bytes}`);
    }
    let grew = false;
    try {
      mem.grow(1);
      grew = true;
    } catch (_) {
      // Expected RangeError on non-growable 320-page Wasm memory
    }
    if (grew) errs.push('wasmMemoryGrowSucceeded');
  }
  const ranges = [];
  const checkPtr = (name, ptr, bytes) => {
    if (!Number.isInteger(ptr) || ptr <= 0) {
      errs.push(`${name}:invalidPtr(${ptr})`);
      return;
    }
    if ((ptr & 3) !== 0) errs.push(`${name}:unalignedPtr(${ptr})`);
    if (ptr + bytes > memBytes) errs.push(`${name}:oob(${ptr}+${bytes}>${memBytes})`);
    ranges.push({ name, start: ptr, end: ptr + bytes });
  };

  checkPtr('telemetry', exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN * 4);
  checkPtr('laplacian', exp.chronos_get_laplacian_ptr(), 1024 * 4);
  checkPtr('fiedler', exp.chronos_get_fiedler_ptr(), 32 * 4);
  for (let s = 0; s < 12; s++) {
    checkPtr(`ring[${s}]`, exp.chronos_get_ring_slot_ptr(s), frameBytes);
  }
  if (checkSwarm) {
    checkPtr('swarmTelemetry', exp.chronos_get_swarm_telemetry_ptr(), 64 * 4);
    checkPtr('microTracks', exp.chronos_get_micro_tracks_ptr(), 1536 * 8 * 4);
    checkPtr('continuumField', exp.chronos_get_continuum_field_ptr(), 880 * 6 * 4);
    checkPtr('rigidBodies', exp.chronos_get_rigid_bodies_ptr(), 32 * 8 * 4);
    checkPtr('modalField', exp.chronos_get_modal_field_ptr(), 880 * 4);
  }

  ranges.sort((a, b) => a.start - b.start);
  for (let i = 1; i < ranges.length; i++) {
    if (ranges[i].start < ranges[i - 1].end) {
      errs.push(`pointerOverlap(${ranges[i - 1].name}[..${ranges[i - 1].end}) vs ${ranges[i].name}[${ranges[i].start}..))`);
    }
  }
  return errs;
}

function streamMp4Frames(mp4Path, w, h, onFrame, maxFrames = null) {
  const frameBytes = w * h * 4;
  const frameScratch = new Uint8Array(frameBytes);
  let carry = 0;

  return new Promise((resolve, reject) => {
    const proc = spawn(
      'ffmpeg',
      ['-v', 'error', '-threads', '1', '-i', mp4Path, '-filter_threads', '1', '-threads', '1', '-vf', `scale=${w}:${h}`, '-pix_fmt', 'rgba', '-f', 'rawvideo', 'pipe:1'],
      { stdio: ['ignore', 'pipe', 'pipe'] }
    );
    let stderrText = '';
    let frameCount = 0;
    let stoppedEarly = false;

    proc.stderr.on('data', (chunk) => {
      stderrText += chunk.toString('utf8');
    });

    proc.stdout.on('data', (chunk) => {
      if (stoppedEarly) return;
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
          if (maxFrames !== null && frameCount >= maxFrames) {
            stoppedEarly = true;
            proc.stdout.destroy();
            proc.kill('SIGTERM');
            break;
          }
        }
      }
    });

    proc.on('error', reject);
    proc.on('close', (code) => {
      if (stoppedEarly) {
        resolve({ numFrames: frameCount, frameBytes });
      } else if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}: ${stderrText}`));
      } else if (carry !== 0) {
        reject(new Error(`Truncated RGBA stream from ${mp4Path}: trailing ${carry}/${frameBytes} bytes`));
      } else {
        resolve({ numFrames: frameCount, frameBytes });
      }
    });
  });
}

/// Exact production EISPACK-tql2 + DGKS Lanczos solver (matches web/worker.js & scripts/benchmark_1000.mjs).
function createLanczosSolver(exp, laplacian, fiedler) {
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
  let tqlBestCol = 0;
  let tqlSecCol = -1;
  let tqlThetaMax = -1e30;
  let tqlThetaSecond = -1e30;
  let lastN = 0;
  let lastLambda2 = 0.0;

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
          if (Math.abs(eSub[m]) <= 1e-14 * (Math.abs(dDiag[m]) + Math.abs(dDiag[m + 1])) + 1e-30) break;
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

  const solver = function solveShiftedLanczosFiedler() {
    if (typeof exp.chronos_solve_fiedler_lanczos === 'function') {
      return exp.chronos_solve_fiedler_lanczos();
    }
    let n = 0;
    for (let i = 0; i < GRAPH_NODES; i++) {
      if (laplacian[i * GRAPH_NODES + i] > 1e-6) activeNodeIdx[n++] = i;
    }
    fiedler.fill(0.0);
    if (n < 2) {
      lastN = 0;
      lastLambda2 = 0.0;
      exp.chronos_set_fiedler_spectrum(0.0, 0.0);
      return 0.0;
    }
    for (let a = 0; a < n; a++) {
      const ia = activeNodeIdx[a] * GRAPH_NODES;
      const rOff = a * n;
      subL[rOff + a] = laplacian[ia + activeNodeIdx[a]];
      for (let b = 0; b < a; b++) {
        const v = laplacian[ia + activeNodeIdx[b]];
        subL[rOff + b] = v;
        subL[b * n + a] = v;
      }
    }
    let mu = 0.0;
    for (let a = 0; a < n; a++) {
      const rOff = a * n;
      let rowSum = subL[rOff + a];
      for (let b = 0; b < n; b++) {
        if (a !== b) rowSum += Math.abs(subL[rOff + b]);
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
    for (let a = 0; a < n; a++) lanczosQ[a] *= invNorm0;

    let actualM = 0;
    let convergedEarly = false;
    for (let j = 0; j < mSteps; j++) {
      const qOff = j * GRAPH_NODES;
      for (let a = 0; a < n; a++) {
        const rOff = a * n;
        let sumL = 0.0;
        for (let b = 0; b < n; b++) sumL += subL[rOff + b] * lanczosQ[qOff + b];
        lanczosW[a] = mu * lanczosQ[qOff + a] - sumL;
      }
      let wMean = 0.0;
      for (let a = 0; a < n; a++) wMean += lanczosW[a];
      wMean *= invN;
      if (j > 0 && lanczosBeta[j - 1] > 0.0) {
        const prevOff = (j - 1) * GRAPH_NODES;
        const betaPrev = lanczosBeta[j - 1];
        for (let a = 0; a < n; a++) lanczosW[a] = (lanczosW[a] - wMean) - betaPrev * lanczosQ[prevOff + a];
      } else {
        for (let a = 0; a < n; a++) lanczosW[a] -= wMean;
      }
      let alpha = 0.0;
      for (let a = 0; a < n; a++) alpha += lanczosQ[qOff + a] * lanczosW[a];
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
        for (let a = 0; a < n; a++) lanczosQ[nextOff + a] = lanczosW[a] * invBeta;
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
      if (nodeIdx < MAX_TARGETS) dotPrev += comp * prevFiedlerTrack[nodeIdx];
    }

    const signFix = (dotPrev < -1e-4 || (Math.abs(dotPrev) <= 1e-4 && eigVecBuf[0] < 0.0)) ? -1.0 : 1.0;
    for (let a = 0; a < n; a++) {
      const nodeIdx = activeNodeIdx[a];
      const val = eigVecBuf[a] * signFix;
      fiedler[nodeIdx] = val;
      if (nodeIdx < MAX_TARGETS) prevFiedlerTrack[nodeIdx] = val;
    }

    lastN = n;
    lastLambda2 = lambda2;
    exp.chronos_set_fiedler_spectrum(lambda2, spectralGap);
    return lambda2;
  };
  solver.computeResNorm = () => {
    if (typeof exp.chronos_get_fiedler_res_norm === 'function') {
      return exp.chronos_get_fiedler_res_norm();
    }
    if (lastN < 2) return 0.0;
    let res2 = 0.0;
    for (let a = 0; a < lastN; a++) {
      const rOff = a * lastN;
      let lv = 0.0;
      for (let b = 0; b < lastN; b++) lv += subL[rOff + b] * eigVecBuf[b];
      const diff = lv - lastLambda2 * eigVecBuf[a];
      res2 += diff * diff;
    }
    return Math.sqrt(res2);
  };
  solver.reset = () => {
    prevFiedlerTrack.fill(0.0);
    lastN = 0;
    lastLambda2 = 0.0;
  };
  return solver;
}

const ALLOWED_FLAGS = new Set([
  '--gate',
  '--hd',
  '--extended',
  '--all',
  '--starling',
  '--dual-res',
  '--dual-mode',
  '--no-speed-gate',
  '--help',
  '-h',
]);

const cliArgs = process.argv.slice(2);
if (cliArgs.includes('--help') || cliArgs.includes('-h')) {
  console.log(
    'Usage: node scripts/calibrate_mp4.mjs [--gate] [--hd] [--dual-res] [--extended|--all|--starling] [--dual-mode] [--swarm-mode=off|auto|always] [--clip=substr]'
  );
  process.exit(0);
}

for (const arg of cliArgs) {
  if (arg.startsWith('--clip=') || arg.startsWith('--swarm-mode=')) continue;
  if (!ALLOWED_FLAGS.has(arg)) {
    console.error(`[FAIL] Unknown CLI flag: ${arg}`);
    process.exit(1);
  }
}

const strictGate = cliArgs.includes('--gate');
const hdFlag = cliArgs.includes('--hd');
const dualResFlag = cliArgs.includes('--dual-res');
const useExtended = cliArgs.includes('--extended');
const useAll = cliArgs.includes('--all');
const useStarling = cliArgs.includes('--starling');
const dualModeFlag = cliArgs.includes('--dual-mode');
const noSpeedGate = cliArgs.includes('--no-speed-gate');
const clipArg = cliArgs.find((a) => a.startsWith('--clip='));
const clipFilter = clipArg ? clipArg.slice('--clip='.length) : '';
if (clipArg && clipFilter.length === 0) {
  console.error('[FAIL] Empty --clip= filter is not allowed.');
  process.exit(1);
}

const rawBudgetScale = process.env.CHRONOS_CI_BUDGET_SCALE;
const budgetScale = rawBudgetScale === undefined ? 1.0 : Number(rawBudgetScale);
if (!Number.isFinite(budgetScale) || budgetScale < 1.0) {
  console.error(`[FAIL] Invalid CHRONOS_CI_BUDGET_SCALE=${rawBudgetScale} (must be finite >= 1.0).`);
  process.exit(1);
}
if (strictGate && noSpeedGate) {
  console.error('[FAIL] --gate cannot be combined with --no-speed-gate.');
  process.exit(1);
}

let explicitSwarmMode = null;
const swarmArg = cliArgs.find((a) => a.startsWith('--swarm-mode='));
if (swarmArg) {
  const v = swarmArg.slice('--swarm-mode='.length).toLowerCase();
  if (v === '0' || v === 'off') explicitSwarmMode = 0;
  else if (v === '1' || v === 'auto') explicitSwarmMode = 1;
  else if (v === '2' || v === 'always') explicitSwarmMode = 2;
  else {
    console.error(`[FAIL] Invalid --swarm-mode=${v} (expected off|auto|always).`);
    process.exit(1);
  }
}
if (dualModeFlag && explicitSwarmMode !== null) {
  console.error('[FAIL] --dual-mode cannot be combined with --swarm-mode=.');
  process.exit(1);
}
if (dualModeFlag && useStarling) {
  console.error('[FAIL] --dual-mode is strictly for the 13 sparse legacy clips; run --starling separately.');
  process.exit(1);
}

const REQUIRED_WAVE2_EXPORTS = [
  'chronos_set_swarm_mode',
  'chronos_get_swarm_mode',
  'chronos_get_micro_tracks_ptr',
  'chronos_get_continuum_field_ptr',
  'chronos_get_rigid_bodies_ptr',
  'chronos_get_swarm_telemetry_ptr',
  'chronos_get_modal_field_ptr',
  'chronos_inspect_field_point',
  'chronos_set_ingest_pts',
];

const wasmBytes = fs.readFileSync(WASM_PATH);
const wasmModule = new WebAssembly.Module(wasmBytes);
const baseInstance = new WebAssembly.Instance(wasmModule, {});
const baseExp = baseInstance.exports;
const hasResSwitch = typeof baseExp.chronos_set_resolution === 'function';
const missingWave2Exports = REQUIRED_WAVE2_EXPORTS.filter((name) => typeof baseExp[name] !== 'function');
const hasSwarmControls = missingWave2Exports.length === 0;

if (dualModeFlag && !hasSwarmControls) {
  console.error(`[FAIL] --dual-mode requires all 9 Wave-2 swarm exports (missing: ${missingWave2Exports.join(', ')}).`);
  process.exit(1);
}
if (explicitSwarmMode !== null && !hasSwarmControls) {
  console.error(`[FAIL] --swarm-mode requires all 9 Wave-2 swarm exports (missing: ${missingWave2Exports.join(', ')}).`);
  process.exit(1);
}
if (useStarling && !hasSwarmControls) {
  console.error(`[FAIL] --starling requires all 9 Wave-2 swarm exports (missing: ${missingWave2Exports.join(', ')}).`);
  process.exit(1);
}

let CLIPS;
if (useStarling) {
  CLIPS = [...SWARM_CLIPS];
} else if (useAll) {
  CLIPS = [...CORE_CLIPS, ...EXTENDED_CLIPS];
} else if (useExtended) {
  CLIPS = [...EXTENDED_CLIPS];
} else {
  CLIPS = [...CORE_CLIPS];
}

if (clipFilter) {
  CLIPS = CLIPS.filter((c) => c.file.includes(clipFilter));
  if (CLIPS.length === 0) {
    console.error(`[FAIL] No clips matched --clip=${clipFilter}`);
    process.exit(1);
  }
}

const resModes = dualResFlag ? (hasResSwitch ? [0, 1] : [0]) : [hdFlag && hasResSwitch ? 1 : 0];

const needsGtDataset = CLIPS.some((c) => c.hasGt !== false);
if (strictGate && needsGtDataset && !gtDataset) {
  throw new Error(`--gate requires ground-truth reference file: ${GT_PATH}`);
}

function createWasmSession(resMode, swarmModeInt, labMode) {
  const inst = new WebAssembly.Instance(wasmModule, {});
  const exp = inst.exports;
  const mem = exp.memory;
  exp.chronos_init();
  if (typeof exp.chronos_set_resolution === 'function') {
    exp.chronos_set_resolution(resMode);
  }
  const w = typeof exp.chronos_get_frame_w === 'function' ? exp.chronos_get_frame_w() : 320;
  const h = typeof exp.chronos_get_frame_h === 'function' ? exp.chronos_get_frame_h() : 180;
  const frameBytes = w * h * 4;
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, labMode);
  let swarmModeVerifyErr = null;
  if (swarmModeInt !== null && hasSwarmControls) {
    exp.chronos_set_swarm_mode(swarmModeInt);
    const readBack = exp.chronos_get_swarm_mode();
    if (readBack !== swarmModeInt) {
      swarmModeVerifyErr = `swarmModeReadBack=${readBack}!==${swarmModeInt}`;
    }
  }
  const ptrErrors = validateWasmModuleAndPointers(wasmModule, exp, mem, frameBytes, hasSwarmControls);
  if (swarmModeVerifyErr) ptrErrors.push(swarmModeVerifyErr);

  const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
  const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 1024);
  const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 32);
  const ringViews = Array.from({ length: 12 }, (_, slot) =>
    new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), frameBytes)
  );
  const swarmTelem = hasSwarmControls
    ? new Float32Array(mem.buffer, exp.chronos_get_swarm_telemetry_ptr(), 64)
    : null;
  const microTracks = hasSwarmControls
    ? new Float32Array(mem.buffer, exp.chronos_get_micro_tracks_ptr(), 1536 * 8)
    : null;
  const continuumField = hasSwarmControls
    ? new Float32Array(mem.buffer, exp.chronos_get_continuum_field_ptr(), 880 * 6)
    : null;
  const rigidBodies = hasSwarmControls
    ? new Float32Array(mem.buffer, exp.chronos_get_rigid_bodies_ptr(), 32 * 8)
    : null;
  const modalField = hasSwarmControls
    ? new Float32Array(mem.buffer, exp.chronos_get_modal_field_ptr(), 880)
    : null;
  const arenaBytes = Math.round(telem[20]);
  if (hasSwarmControls && (!Number.isFinite(telem[20]) || arenaBytes < 17_000_000 || arenaBytes > 20_800_000)) {
    ptrErrors.push(`invalidWave2ArenaBytes(${telem[20]})`);
  }
  const solveLanczos = createLanczosSolver(exp, laplacian, fiedler);

  return {
    exp,
    mem,
    w,
    h,
    frameBytes,
    arenaBytes,
    telem,
    ringViews,
    swarmTelem,
    microTracks,
    continuumField,
    rigidBodies,
    modalField,
    solveLanczos,
    ptrErrors,
  };
}

const POOL_SESSIONS = [null, null, null];

function acquirePooledWasmSession(poolIdx, resMode, swarmModeInt, labMode) {
  let sess = POOL_SESSIONS[poolIdx];
  if (!sess) {
    const inst = new WebAssembly.Instance(wasmModule, {});
    const exp = inst.exports;
    const mem = exp.memory;
    const initDataPrefix = new Uint8Array(mem.buffer.slice(0, 65536));
    const memU8 = new Uint8Array(mem.buffer);
    memU8.fill(0, 65536);
    const telem = new Float32Array(mem.buffer, exp.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
    const laplacian = new Float32Array(mem.buffer, exp.chronos_get_laplacian_ptr(), 1024);
    const fiedler = new Float32Array(mem.buffer, exp.chronos_get_fiedler_ptr(), 32);
    const swarmTelem = hasSwarmControls
      ? new Float32Array(mem.buffer, exp.chronos_get_swarm_telemetry_ptr(), 64)
      : null;
    const microTracks = hasSwarmControls
      ? new Float32Array(mem.buffer, exp.chronos_get_micro_tracks_ptr(), 1536 * 8)
      : null;
    const continuumField = hasSwarmControls
      ? new Float32Array(mem.buffer, exp.chronos_get_continuum_field_ptr(), 880 * 6)
      : null;
    const rigidBodies = hasSwarmControls
      ? new Float32Array(mem.buffer, exp.chronos_get_rigid_bodies_ptr(), 32 * 8)
      : null;
    const modalField = hasSwarmControls
      ? new Float32Array(mem.buffer, exp.chronos_get_modal_field_ptr(), 880)
      : null;
    const solveLanczos = createLanczosSolver(exp, laplacian, fiedler);
    sess = {
      exp,
      mem,
      memU8,
      initDataPrefix,
      w: 0,
      h: 0,
      frameBytes: 0,
      arenaBytes: 0,
      telem,
      ringViews: null,
      swarmTelem,
      microTracks,
      continuumField,
      rigidBodies,
      modalField,
      solveLanczos,
      basePtrErrors: null,
      ptrErrors: [],
    };
    POOL_SESSIONS[poolIdx] = sess;
  }

  sess.memU8.set(sess.initDataPrefix, 0);
  sess.memU8.fill(0, 65536);
  sess.solveLanczos.reset();

  const { exp, mem, telem } = sess;
  exp.chronos_init();
  if (typeof exp.chronos_set_resolution === 'function') {
    exp.chronos_set_resolution(resMode);
  }
  const w = typeof exp.chronos_get_frame_w === 'function' ? exp.chronos_get_frame_w() : 320;
  const h = typeof exp.chronos_get_frame_h === 'function' ? exp.chronos_get_frame_h() : 180;
  const frameBytes = w * h * 4;
  if (frameBytes !== sess.frameBytes || !sess.ringViews) {
    sess.w = w;
    sess.h = h;
    sess.frameBytes = frameBytes;
    sess.ringViews = Array.from({ length: 12 }, (_, slot) =>
      new Uint8Array(mem.buffer, exp.chronos_get_ring_slot_ptr(slot), frameBytes)
    );
    sess.basePtrErrors = validateWasmModuleAndPointers(wasmModule, exp, mem, frameBytes, hasSwarmControls);
  }
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, labMode);
  const ptrErrors = [...sess.basePtrErrors];
  if (swarmModeInt !== null && hasSwarmControls) {
    exp.chronos_set_swarm_mode(swarmModeInt);
    const readBack = exp.chronos_get_swarm_mode();
    if (readBack !== swarmModeInt) {
      ptrErrors.push(`swarmModeReadBack=${readBack}!==${swarmModeInt}`);
    }
  }
  const arenaBytes = Math.round(telem[20]);
  if (hasSwarmControls && (!Number.isFinite(telem[20]) || arenaBytes < 17_000_000 || arenaBytes > 20_800_000)) {
    ptrErrors.push(`invalidWave2ArenaBytes(${telem[20]})`);
  }
  sess.arenaBytes = arenaBytes;
  sess.ptrErrors = ptrErrors;
  return sess;
}

async function measureLockedSceneGravity(mp4Path, fps, labMode, resMode, swarmModeInt) {
  const sess = acquirePooledWasmSession(2, resMode, swarmModeInt, labMode);
  const dtSec = 1.0 / fps;
  let lastGrav = 0.0;
  let maxConf = 0.0;
  let maxArcs = 0;
  await streamMp4Frames(mp4Path, sess.w, sess.h, (frameRgba, f) => {
    sess.ringViews[f % 12].set(frameRgba);
    if (typeof sess.exp.chronos_set_ingest_pts === 'function') {
      sess.exp.chronos_set_ingest_pts(f * dtSec);
    }
    sess.exp.chronos_pre_tick(dtSec);
    sess.solveLanczos();
    sess.exp.chronos_post_tick(dtSec);
    if (sess.telem[1248] > 0.0) lastGrav = sess.telem[1248];
    if (sess.telem[1249] > maxConf) maxConf = sess.telem[1249];
    if (Math.round(sess.telem[1259]) > maxArcs) maxArcs = Math.round(sess.telem[1259]);
  });
  return { grav: lastGrav, conf: maxConf, arcs: maxArcs };
}

function createMotState() {
  return {
    prevFrameGtToTid: new Map(),
    lastTidForGt: new Map(),
    lastGtForTid: new Map(),
    gtSeenTracked: new Set(),
    gtCurrentlyLost: new Set(),
    evalFrames: [],
    gtTotal: 0,
    gtMatched: 0,
    gtPredTotal: 0,
    gtUnmatchedFp: 0,
    gtSumSqErr: 0.0,
    gtSumDist: 0.0,
    gtIdSwaps: 0,
    gtHijacks: 0,
    gtFrags: 0,
  };
}

function recordMotFrame(mot, gts, tracks) {
  mot.evalFrames.push({ gts, tracks });
  mot.gtTotal += gts.length;
  mot.gtPredTotal += tracks.length;

  const matches = matchClearMotContinuity(gts, tracks, mot.prevFrameGtToTid);
  const matchedThisFrame = matches.length;
  const matchedGi = new Uint8Array(gts.length);
  mot.prevFrameGtToTid.clear();

  for (let k = 0; k < matches.length; k++) {
    const m = matches[k];
    matchedGi[m.gi] = 1;
    mot.gtMatched++;
    mot.gtSumSqErr += m.d * m.d;
    mot.gtSumDist += m.d;

    const gtId = gts[m.gi].gtId;
    const tid = tracks[m.ti].tid;
    mot.prevFrameGtToTid.set(gtId, tid);

    if (mot.lastTidForGt.has(gtId) && mot.lastTidForGt.get(gtId) !== tid) mot.gtIdSwaps++;
    mot.lastTidForGt.set(gtId, tid);
    if (mot.lastGtForTid.has(tid) && mot.lastGtForTid.get(tid) !== gtId) mot.gtHijacks++;
    mot.lastGtForTid.set(tid, gtId);

    if (mot.gtCurrentlyLost.has(gtId)) {
      mot.gtFrags++;
      mot.gtCurrentlyLost.delete(gtId);
    }
    mot.gtSeenTracked.add(gtId);
  }

  for (let gi = 0; gi < gts.length; gi++) {
    if (!matchedGi[gi]) {
      const gtId = gts[gi].gtId;
      if (mot.gtSeenTracked.has(gtId)) mot.gtCurrentlyLost.add(gtId);
    }
  }
  mot.gtUnmatchedFp += tracks.length - matchedThisFrame;
}

function finalizeMotStats(mot) {
  const fn = mot.gtTotal - mot.gtMatched;
  return {
    gtTotal: mot.gtTotal,
    gtMatched: mot.gtMatched,
    gtPredTotal: mot.gtPredTotal,
    gtUnmatchedFp: mot.gtUnmatchedFp,
    gtIdSwaps: mot.gtIdSwaps,
    gtHijacks: mot.gtHijacks,
    gtFrags: mot.gtFrags,
    recallPct: mot.gtTotal > 0 ? (100.0 * mot.gtMatched) / mot.gtTotal : null,
    precisionPct: mot.gtPredTotal > 0 ? (100.0 * mot.gtMatched) / mot.gtPredTotal : null,
    motaPct: mot.gtTotal > 0 ? (1.0 - (fn + mot.gtUnmatchedFp + mot.gtIdSwaps) / mot.gtTotal) * 100.0 : null,
    rmsePx: mot.gtMatched > 0 ? Math.sqrt(mot.gtSumSqErr / mot.gtMatched) : null,
    motpPx: mot.gtMatched > 0 ? mot.gtSumDist / mot.gtMatched : null,
    hota: mot.gtTotal > 0 ? computeTrackEvalHota(mot.evalFrames) : null,
  };
}

const SWARM_RESERVED_TAIL_START = 42;

function computePercentileCal(values, pct) {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.floor((pct / 100.0) * (s.length - 1))));
  return s[idx];
}

function sampleBicubicLumaCal(buf, w, h, fx, fy) {
  const ix = Math.floor(fx);
  const iy = Math.floor(fy);
  const tx = fx - ix;
  const ty = fy - iy;
  if (tx < 1e-6 && ty < 1e-6) {
    const cx = ix < 0 ? 0 : ix >= w ? w - 1 : ix;
    const cy = iy < 0 ? 0 : iy >= h ? h - 1 : iy;
    return buf[cy * w + cx];
  }
  const w0x = -0.5 * tx * (1 - tx) * (1 - tx);
  const w1x = 1 - 2.5 * tx * tx + 1.5 * tx * tx * tx;
  const w2x = 0.5 * tx * (1 + 4 * tx - 3 * tx * tx);
  const w3x = -0.5 * tx * tx * (1 - tx);
  const w0y = -0.5 * ty * (1 - ty) * (1 - ty);
  const w1y = 1 - 2.5 * ty * ty + 1.5 * ty * ty * ty;
  const w2y = 0.5 * ty * (1 + 4 * ty - 3 * ty * ty);
  const w3y = -0.5 * ty * ty * (1 - ty);
  const clampX = (x) => (x < 0 ? 0 : x >= w ? w - 1 : x);
  const clampY = (y) => (y < 0 ? 0 : y >= h ? h - 1 : y);
  const x0 = clampX(ix - 1);
  const x1 = clampX(ix);
  const x2 = clampX(ix + 1);
  const x3 = clampX(ix + 2);
  const y0 = clampY(iy - 1) * w;
  const y1 = clampY(iy) * w;
  const y2 = clampY(iy + 1) * w;
  const y3 = clampY(iy + 2) * w;
  const r0 = w0x * buf[y0 + x0] + w1x * buf[y0 + x1] + w2x * buf[y0 + x2] + w3x * buf[y0 + x3];
  const r1 = w0x * buf[y1 + x0] + w1x * buf[y1 + x1] + w2x * buf[y1 + x2] + w3x * buf[y1 + x3];
  const r2 = w0x * buf[y2 + x0] + w1x * buf[y2 + x1] + w2x * buf[y2 + x2] + w3x * buf[y2 + x3];
  const r3 = w0x * buf[y3 + x0] + w1x * buf[y3 + x1] + w2x * buf[y3 + x2] + w3x * buf[y3 + x3];
  const val = w0y * r0 + w1y * r1 + w2y * r2 + w3y * r3;
  return val < 0.0 ? 0.0 : val > 255.0 ? 255.0 : val;
}

function measureStarlingDarkExtremaPrecisionCal(frameRgba, prevRgba, w, h, observedTracks) {
  if (observedTracks.length === 0) return 0.0;
  const N = w * h;
  const luma = new Float32Array(N);
  const prevLuma = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    luma[i] = 0.299 * frameRgba[o] + 0.587 * frameRgba[o + 1] + 0.114 * frameRgba[o + 2];
    prevLuma[i] = 0.299 * prevRgba[o] + 0.587 * prevRgba[o + 1] + 0.114 * prevRgba[o + 2];
  }

  // Gradient-stratified + jittered background sampling over bounded pan [-maxPan..maxPan] + wide-scene guard + Catmull-Rom refinement
  const maxPan = Math.max(8, Math.round((8 * w) / 640));
  const margin = 3;
  const samplesX = [];
  const samplesY = [];
  const sampleAdded = new Uint8Array(N);
  const prevTexX = [];
  const prevTexY = [];
  const currTexX = [];
  const currTexY = [];
  let minGx = w;
  let maxGx = -1;
  let minGy = h;
  let maxGy = -1;
  for (let y = margin; y < h - margin; y++) {
    const rowOff = y * w;
    for (let x = margin; x < w - margin; x++) {
      const idx = rowOff + x;
      const pGx = Math.abs(prevLuma[idx + 1] - prevLuma[idx - 1]);
      const pGy = Math.abs(prevLuma[idx + w] - prevLuma[idx - w]);
      const pAvg4 = 0.25 * (prevLuma[idx + 1] + prevLuma[idx - 1] + prevLuma[idx + w] + prevLuma[idx - w]);
      if (pGx > 4.0 || pGy > 4.0 || Math.abs(prevLuma[idx] - pAvg4) > 4.0) {
        prevTexX.push(x);
        prevTexY.push(y);
        if (x < minGx) minGx = x;
        if (x > maxGx) maxGx = x;
        if (y < minGy) minGy = y;
        if (y > maxGy) maxGy = y;
      }
      const cGx = Math.abs(luma[idx + 1] - luma[idx - 1]);
      const cGy = Math.abs(luma[idx + w] - luma[idx - w]);
      const cAvg4 = 0.25 * (luma[idx + 1] + luma[idx - 1] + luma[idx + w] + luma[idx - w]);
      if (cGx > 4.0 || cGy > 4.0 || Math.abs(luma[idx] - cAvg4) > 4.0) {
        currTexX.push(x);
        currTexY.push(y);
        if (x < minGx) minGx = x;
        if (x > maxGx) maxGx = x;
        if (y < minGy) minGy = y;
        if (y > maxGy) maxGy = y;
      }
    }
  }
  const prevStep = Math.max(1, Math.ceil(prevTexX.length / 900));
  for (let i = 0; i < prevTexX.length; i += prevStep) {
    const x = prevTexX[i];
    const y = prevTexY[i];
    const idx = y * w + x;
    if (!sampleAdded[idx]) {
      sampleAdded[idx] = 1;
      samplesX.push(x);
      samplesY.push(y);
    }
  }
  const currStep = Math.max(1, Math.ceil(currTexX.length / 900));
  for (let i = 0; i < currTexX.length; i += currStep) {
    const x = currTexX[i];
    const y = currTexY[i];
    const idx = y * w + x;
    if (!sampleAdded[idx]) {
      sampleAdded[idx] = 1;
      samplesX.push(x);
      samplesY.push(y);
    }
  }
  const stepY = Math.max(4, Math.floor(h / 48));
  const stepX = Math.max(5, Math.floor(w / 64));
  for (let row = 0, y = margin; y < h - margin; y += stepY, row++) {
    for (let col = 0, x = margin; x < w - margin; x += stepX, col++) {
      const jx = Math.min(w - margin - 1, x + ((row * 13 + col * 29) % stepX));
      const jy = Math.min(h - margin - 1, y + ((row * 17 + col * 11) % stepY));
      const idx = jy * w + jx;
      if (!sampleAdded[idx]) {
        sampleAdded[idx] = 1;
        samplesX.push(jx);
        samplesY.push(jy);
      }
    }
  }
  const S = samplesX.length;
  let texTileCnt = 0;
  {
    const tSeen = new Uint8Array(48);
    const addT = (xs, ys) => {
      for (let i = 0; i < xs.length; i++) {
        const tx = Math.min(7, Math.floor((xs[i] * 8) / w));
        const ty = Math.min(5, Math.floor((ys[i] * 6) / h));
        const ti = ty * 8 + tx;
        if (!tSeen[ti]) { tSeen[ti] = 1; texTileCnt++; }
      }
    };
    addT(prevTexX, prevTexY);
    addT(currTexX, currTexY);
  }
  const hasWideSceneTexture =
    texTileCnt >= 4 && prevTexX.length + currTexX.length >= 64 && (maxGx - minGx >= w * 0.25 || maxGy - minGy >= h * 0.25);

  const evalShiftOn = (xs, ys, cnt, dx, dy) => {
    if (cnt <= 0) {
      return {
        dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, corr: 0.0, informative: false,
        invGain: -1.0, invOffset: 0.0, invMse: Infinity, invInformative: false,
      };
    }
    const isInt = Math.abs(dx - Math.round(dx)) < 1e-6 && Math.abs(dy - Math.round(dy)) < 1e-6;
    const idxShift = Math.round(dx);
    const idyShift = Math.round(dy);
    let sumC = 0.0;
    let sumP = 0.0;
    let sumCC = 0.0;
    let sumPP = 0.0;
    let sumCP = 0.0;
    let validCnt = 0;
    for (let i = 0; i < cnt; i++) {
      const x = xs[i];
      const y = ys[i];
      let p;
      if (isInt) {
        const px = x - idxShift;
        const py = y - idyShift;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;
        p = prevLuma[py * w + px];
      } else {
        const sx = x - dx;
        const sy = y - dy;
        if (sx < 1.0 || sx > w - 2.0 || sy < 1.0 || sy > h - 2.0) continue;
        p = sampleBicubicLumaCal(prevLuma, w, h, sx, sy);
      }
      const c = luma[y * w + x];
      sumC += c;
      sumP += p;
      sumCC += c * c;
      sumPP += p * p;
      sumCP += c * p;
      validCnt++;
    }
    if (validCnt < Math.max(8, Math.floor(cnt * 0.5))) {
      return {
        dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, informative: false,
        invGain: -1.0, invOffset: 0.0, invMse: Infinity, invInformative: false,
      };
    }
    const meanC = sumC / validCnt;
    const meanP = sumP / validCnt;
    const varC = Math.max(0.0, sumCC / validCnt - meanC * meanC);
    const varP = Math.max(0.0, sumPP / validCnt - meanP * meanP);
    const cov = sumCP / validCnt - meanC * meanP;
    const rawGain = varP > 1e-4 ? cov / varP : 1.0;
    const gain = Math.max(0.25, Math.min(4.0, rawGain));
    const offset = meanC - gain * meanP;
    const mse = Math.max(0.0, varC + gain * gain * varP - 2.0 * gain * cov);
    const varBalanced = varC > 0.25 && varP > 0.25 && varP >= varC * 0.25 && varC >= varP * 0.25;
    const informative = cov > 0.0 && varBalanced;
    const corr = varBalanced ? cov / Math.sqrt(varC * varP) : 0.0;
    const invGain = Math.max(-4.0, Math.min(-0.25, varP > 1e-4 ? cov / varP : -1.0));
    const invOffset = meanC - invGain * meanP;
    const invMse = Math.max(0.0, varC + invGain * invGain * varP - 2.0 * invGain * cov);
    const invInformative = cov < 0.0 && varBalanced && corr <= -0.75;
    return { dx, dy, mse, gain, offset, corr, informative, invGain, invOffset, invMse, invInformative };
  };
  const evalShift = (dx, dy) => evalShiftOn(samplesX, samplesY, S, dx, dy);
  const compareFit = (a, b) => {
    const mseDiff = a.mse - b.mse;
    if (Math.abs(mseDiff) > 1e-4) return mseDiff;
    const affA = Math.abs(a.gain - 1.0) + 0.01 * Math.abs(a.offset);
    const affB = Math.abs(b.gain - 1.0) + 0.01 * Math.abs(b.offset);
    if (Math.abs(affA - affB) > 1e-4) return affA - affB;
    return Math.abs(a.dx) + Math.abs(a.dy) - (Math.abs(b.dx) + Math.abs(b.dy));
  };

  const zeroFit = evalShift(0, 0);
  let best = zeroFit;
  let rerankX = samplesX;
  let rerankY = samplesY;
  let R = S;
  let zeroRerank = zeroFit;
  if (hasWideSceneTexture && S > 0) {
    const intFits = [];
    for (let dy = -maxPan; dy <= maxPan; dy++) {
      for (let dx = -maxPan; dx <= maxPan; dx++) {
        if (dx === 0 && dy === 0) continue;
        const fit = evalShift(dx, dy);
        if (fit.informative) intFits.push(fit);
      }
    }
    if (intFits.length > 0) {
      intFits.sort(compareFit);
      const minIntMse = intFits[0].mse;
      const candCutoff = minIntMse * 1.35 + 0.25;
      const cands = [{ dx: 0, dy: 0 }];
      for (let i = 0; i < intFits.length && cands.length < 16; i++) {
        if (intFits[i].mse <= candCutoff) cands.push({ dx: intFits[i].dx, dy: intFits[i].dy });
      }
      const rAdded = new Uint8Array(N);
      rerankX = [];
      rerankY = [];
      const addUnion = (xs, ys, maxKeep) => {
        const st = Math.max(1, Math.ceil(xs.length / maxKeep));
        for (let i = 0; i < xs.length; i += st) {
          const x = xs[i];
          const y = ys[i];
          const idx = y * w + x;
          if (!rAdded[idx]) {
            rAdded[idx] = 1;
            rerankX.push(x);
            rerankY.push(y);
          }
        }
      };
      addUnion(prevTexX, prevTexY, 2000);
      addUnion(currTexX, currTexY, 2000);
      const bgStep = w <= 360 ? 3 : 4;
      for (let y = margin; y < h - margin; y += bgStep) {
        for (let x = margin; x < w - margin; x += bgStep) {
          const idx = y * w + x;
          if (!rAdded[idx]) {
            rAdded[idx] = 1;
            rerankX.push(x);
            rerankY.push(y);
          }
        }
      }
      R = rerankX.length;
      const reranked = [];
      for (const c of cands) {
        const rf = evalShiftOn(rerankX, rerankY, R, c.dx, c.dy);
        if (c.dx === 0 && c.dy === 0) zeroRerank = rf;
        if (rf.informative) reranked.push(rf);
      }
      if (reranked.length > 0) {
        reranked.sort(compareFit);
        best = reranked[0];
      }
    }
    const centers =
      best.dx === 0 && best.dy === 0
        ? [{ dx: 0, dy: 0 }]
        : [{ dx: best.dx, dy: best.dy }, { dx: 0, dy: 0 }];
    for (const seed of centers) {
      let cx = seed.dx;
      let cy = seed.dy;
      let localBest = evalShift(cx, cy);
      if (localBest.informative && (!best.informative || compareFit(localBest, best) < 0)) best = localBest;
      for (const step of [0.5, 0.2, 0.1, 0.05]) {
        const baseX = cx;
        const baseY = cy;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            if (kx === 0 && ky === 0) continue;
            const tx = baseX + kx * step;
            const ty = baseY + ky * step;
            const fit = evalShift(tx, ty);
            if (fit.informative && (!localBest.informative || compareFit(fit, localBest) < 0)) {
              localBest = fit;
              cx = tx;
              cy = ty;
              if (!best.informative || compareFit(fit, best) < 0) best = fit;
            }
          }
        }
      }
    }
    if (best.informative && R > 0 && (best.dx !== 0 || best.dy !== 0)) {
      const polX = [], polY = [];
      const isSubB = Math.abs(best.dx - Math.round(best.dx)) > 1e-4 || Math.abs(best.dy - Math.round(best.dy)) > 1e-4;
      for (let i = 0; i < R; i++) {
        const x = rerankX[i], y = rerankY[i];
        const px = x - best.dx, py = y - best.dy;
        if (px < 1 || px >= w - 2 || py < 1 || py >= h - 2) continue;
        const p = isSubB ? sampleBicubicLumaCal(prevLuma, w, h, px, py) : prevLuma[Math.round(py) * w + Math.round(px)];
        if (Math.abs(luma[y * w + x] - (best.gain * p + best.offset)) <= 8.0) { polX.push(x); polY.push(y); }
      }
      if (polX.length >= 24) {
        let cxP = best.dx, cyP = best.dy;
        let bP = evalShiftOn(polX, polY, polX.length, cxP, cyP);
        for (const stP of [0.2, 0.1, 0.05]) {
          const bx0 = cxP, by0 = cyP;
          for (let ky = -1; ky <= 1; ky++) for (let kx = -1; kx <= 1; kx++) {
            if (kx === 0 && ky === 0) continue;
            const fP = evalShiftOn(polX, polY, polX.length, bx0 + kx * stP, by0 + ky * stP);
            if (fP.informative && (!bP.informative || compareFit(fP, bP) < 0)) { bP = fP; cxP = bx0 + kx * stP; cyP = by0 + ky * stP; }
          }
        }
        if (bP.informative) best = { ...bP };
      }
      best = evalShiftOn(rerankX, rerankY, R, best.dx, best.dy);
      if (zeroRerank === zeroFit) zeroRerank = evalShiftOn(rerankX, rerankY, R, 0, 0);
    }
  }

  const useStaticInvertedZero =
    zeroRerank.invInformative && zeroRerank.invMse <= 6.0 && (!zeroRerank.informative || zeroRerank.invMse < zeroRerank.mse);
  const zeroCompareMse = useStaticInvertedZero ? zeroRerank.invMse : zeroRerank.mse;

  let useDx = 0.0;
  let useDy = 0.0;
  let allowNegGain = false;
  let useGain = (!hasWideSceneTexture && (!zeroRerank.informative || zeroRerank.corr < 0.65)) ? 1.0 : zeroRerank.gain;
  let useOffset = (!hasWideSceneTexture && (!zeroRerank.informative || zeroRerank.corr < 0.65)) ? 0.0 : zeroRerank.offset;
  if (
    hasWideSceneTexture &&
    S > 0 &&
    best.informative &&
    zeroCompareMse > 1e-4 &&
    best.mse + 1e-5 < zeroCompareMse * 0.65 &&
    (Math.abs(best.dx) > 1e-4 || Math.abs(best.dy) > 1e-4)
  ) {
    useDx = best.dx;
    useDy = best.dy;
    useGain = best.gain;
    useOffset = best.offset;
  } else if (useStaticInvertedZero) {
    allowNegGain = true;
    useGain = zeroRerank.invGain;
    useOffset = zeroRerank.invOffset;
  }
  let stage1AltShifts = [];
  let inStage1Sub = false;
  let stage1GridX = null;
  const geomCovOk = (dx, dy) => (Math.max(0.0, w - Math.abs(dx)) * Math.max(0.0, h - Math.abs(dy))) / (w * h) >= 0.58;
  const maxPanX = Math.max(64, Math.round((64 * w) / 640));
  const maxPanY = Math.max(48, Math.round((48 * h) / 360));
  const evalBilinearOnPts = (xs, ys, cnt, dx, dy) => {
    if (cnt <= 0) return { dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
    const isInt = Math.abs(dx - Math.round(dx)) < 1e-6 && Math.abs(dy - Math.round(dy)) < 1e-6;
    const idxS = Math.round(dx);
    const idyS = Math.round(dy);
    const sampBil = (buf, fx, fy) => {
      const x0 = Math.floor(fx), y0 = Math.floor(fy);
      const x1 = x0 + 1 < w ? x0 + 1 : x0;
      const y1 = y0 + 1 < h ? y0 + 1 : y0;
      const tx = fx - x0, ty = fy - y0;
      const r0 = y0 * w, r1 = y1 * w;
      const v00 = buf[r0 + x0], v10 = buf[r0 + x1], v01 = buf[r1 + x0], v11 = buf[r1 + x1];
      return (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + (v01 * (1.0 - tx) + v11 * tx) * ty;
    };
    const cs = [], ps = [];
    let sc = 0.0, sp = 0.0, scc = 0.0, spp = 0.0, scp = 0.0;
    for (let i = 0; i < cnt; i++) {
      const x = xs[i], y = ys[i];
      let p;
      if (isInt) {
        const px = x - idxS, py = y - idyS;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;
        p = prevLuma[py * w + px];
      } else {
        const sx = x - dx, sy = y - dy;
        if (sx < 0.0 || sx > w - 1.0 || sy < 0.0 || sy > h - 1.0) continue;
        p = sampBil(prevLuma, sx, sy);
      }
      const c = luma[y * w + x];
      cs.push(c); ps.push(p);
      sc += c; sp += p; scc += c * c; spp += p * p; scp += c * p;
    }
    const n = cs.length;
    if (n < Math.max(12, Math.floor(cnt * 0.5))) {
      return { dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
    }
    const mc = sc / n, mp = sp / n;
    const vc = Math.max(0.0, scc / n - mc * mc);
    const vp = Math.max(0.0, spp / n - mp * mp);
    const cov = scp / n - mc * mp;
    if (cov <= 0.0 || vc < 2.0 || vp < 2.0) {
      return { dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
    }
    const gain = Math.max(0.25, Math.min(4.0, cov / vp));
    const offset = mc - gain * mp;
    const errs = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const d = cs[i] - (gain * ps[i] + offset);
      errs[i] = d * d;
    }
    errs.sort();
    const keepN = Math.max(8, Math.floor(n * 0.65));
    let trimSum = 0.0;
    for (let i = 0; i < keepN; i++) trimSum += errs[i];
    // Refit on lowest 65% inliers for pure background immunity to moving dark clouds
    const thresh = errs[keepN - 1] + 1e-6;
    let isc = 0.0, isp = 0.0, iscc = 0.0, ispp = 0.0, iscp = 0.0, inn = 0;
    for (let i = 0; i < n; i++) {
      const d = cs[i] - (gain * ps[i] + offset);
      if (d * d > thresh) continue;
      const c = cs[i], p = ps[i];
      isc += c; isp += p; iscc += c * c; ispp += p * p; iscp += c * p;
      inn++;
    }
    let trimMse = trimSum / keepN, inCorr = cov / Math.sqrt(vc * vp), inGain = gain, inOffset = offset;
    if (inn >= 8) {
      const imc = isc / inn, imp = isp / inn;
      const ivc = Math.max(0.0, iscc / inn - imc * imc);
      const ivp = Math.max(0.0, ispp / inn - imp * imp);
      const icov = iscp / inn - imc * imp;
      if (ivc > 1.0 && ivp > 1.0 && icov > 0.0) {
        inGain = Math.max(0.25, Math.min(4.0, icov / ivp));
        inOffset = imc - inGain * imp;
        trimMse = Math.max(0.0, ivc + inGain * inGain * ivp - 2.0 * inGain * icov);
        inCorr = icov / Math.sqrt(ivc * ivp);
      }
    }
    return { dx, dy, mse: trimMse, gain: inGain, offset: inOffset, corr: inCorr, informative: true };
  };

  // Uniform grid background points (excluding dark pit pixels) for static-textured-background consensus
  const bgGridX = [], bgGridY = [];
  {
    const gsx = Math.max(6, Math.floor((w - 10) / 24));
    const gsy = Math.max(6, Math.floor((h - 10) / 16));
    let meanAll = 0.0, nAll = 0;
    for (let gy = 5; gy <= h - 6; gy += gsy) {
      for (let gx = 5; gx <= w - 6; gx += gsx) {
        meanAll += luma[gy * w + gx];
        nAll++;
      }
    }
    meanAll = nAll > 0 ? meanAll / nAll : 128.0;
    for (let gy = 5; gy <= h - 6; gy += gsy) {
      for (let gx = 5; gx <= w - 6; gx += gsx) {
        const c = luma[gy * w + gx];
        let maxNb = c;
        for (let oy = -3; oy <= 3; oy += 3) {
          for (let ox = -3; ox <= 3; ox += 3) {
            const v = luma[(gy + oy) * w + (gx + ox)];
            if (v > maxNb) maxNb = v;
          }
        }
        if (maxNb - c >= 25.0 || meanAll - c >= 35.0) continue;
        const pc = prevLuma[gy * w + gx];
        let pMaxNb = pc;
        let pSumNb = 0.0;
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const pv = prevLuma[(gy + oy) * w + (gx + ox)];
            if (pv > pMaxNb) pMaxNb = pv;
            pSumNb += pv;
          }
        }
        if (pMaxNb - pc >= 18.0 || pSumNb / 9.0 - pc >= 12.0 || maxNb - c >= 18.0) continue;
        bgGridX.push(gx);
        bgGridY.push(gy);
      }
    }
  }
  const zeroBgFit = evalBilinearOnPts(bgGridX, bgGridY, bgGridX.length, 0, 0);
  const hasTexturedBackground = zeroBgFit.informative || (bgGridX.length >= 16 && (() => {
    let s = 0, s2 = 0;
    for (let i = 0; i < bgGridX.length; i++) { const v = luma[bgGridY[i] * w + bgGridX[i]]; s += v; s2 += v * v; }
    const m = s / bgGridX.length;
    return s2 / bgGridX.length - m * m >= 3.0;
  })());
  const acceptsBgConsensus = (candDx, candDy) => {
    if (!hasTexturedBackground || (Math.abs(candDx) <= 1e-4 && Math.abs(candDy) <= 1e-4)) return true;
    const candBg = evalBilinearOnPts(bgGridX, bgGridY, bgGridX.length, candDx, candDy);
    if (zeroBgFit.informative && zeroBgFit.corr >= 0.95 && zeroBgFit.mse <= 2.0) {
      const candBetterBg = candBg.informative && candBg.corr >= Math.max(0.995, zeroBgFit.corr) && candBg.mse < zeroBgFit.mse && best.informative && best.corr >= 0.985 && zeroRerank.mse > Math.max(8.0, best.mse * 8.0) && zeroRerank.corr < best.corr - 0.03;
      if (!candBetterBg && (!candBg.informative || candBg.corr < zeroBgFit.corr - 1e-4 || candBg.mse > zeroBgFit.mse * 0.70 + 1e-4)) return false;
    }
    if (zeroBgFit.informative && zeroBgFit.corr >= 0.90 && zeroBgFit.mse <= 4.0) {
      if (!candBg.informative || candBg.corr < 0.85 || candBg.mse > Math.max(4.0, zeroBgFit.mse * 2.5)) return false;
    }
    if (candBg.informative && zeroBgFit.informative && candBg.mse > zeroBgFit.mse * 1.10 + 0.5 && candBg.corr < zeroBgFit.corr) {
      return false;
    }
    return true;
  };
  if ((Math.abs(useDx) > 1e-4 || Math.abs(useDy) > 1e-4) && !acceptsBgConsensus(useDx, useDy)) {
    useDx = 0.0;
    useDy = 0.0;
    useGain = zeroBgFit.informative ? zeroBgFit.gain : zeroRerank.gain;
    useOffset = zeroBgFit.informative ? zeroBgFit.offset : zeroRerank.offset;
  }

  let blurCurr = null;
  let blurPrev = null;
  let blur3Curr = null;
  let blur3Prev = null;
  let sepScratch = null;
  const runSepPass = (src, outBuf) => {
    const k0 = 0.0044318484, k1 = 0.0539909665, k2 = 0.2419707245, k3 = 0.3989422804;
    const norm = 1.0 / (k3 + 2.0 * (k0 + k1 + k2));
    const w0 = k0 * norm, w1 = k1 * norm, w2 = k2 * norm, w3 = k3 * norm;
    if (!sepScratch) sepScratch = new Float32Array(N);
    const tmp = sepScratch;
    const out = outBuf || new Float32Array(N);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const xm3 = x >= 3 ? x - 3 : 0, xm2 = x >= 2 ? x - 2 : 0, xm1 = x >= 1 ? x - 1 : 0;
        const xp1 = x + 1 < w ? x + 1 : w - 1, xp2 = x + 2 < w ? x + 2 : w - 1, xp3 = x + 3 < w ? x + 3 : w - 1;
        tmp[row + x] =
          w3 * src[row + x] +
          w2 * (src[row + xm1] + src[row + xp1]) +
          w1 * (src[row + xm2] + src[row + xp2]) +
          w0 * (src[row + xm3] + src[row + xp3]);
      }
    }
    for (let y = 0; y < h; y++) {
      const ym3 = (y >= 3 ? y - 3 : 0) * w, ym2 = (y >= 2 ? y - 2 : 0) * w, ym1 = (y >= 1 ? y - 1 : 0) * w;
      const y0 = y * w;
      const yp1 = (y + 1 < h ? y + 1 : h - 1) * w, yp2 = (y + 2 < h ? y + 2 : h - 1) * w, yp3 = (y + 3 < h ? y + 3 : h - 1) * w;
      for (let x = 0; x < w; x++) {
        out[y0 + x] =
          w3 * tmp[y0 + x] +
          w2 * (tmp[ym1 + x] + tmp[yp1 + x]) +
          w1 * (tmp[ym2 + x] + tmp[yp2 + x]) +
          w0 * (tmp[ym3 + x] + tmp[yp3 + x]);
      }
    }
    return out;
  };
  const ensureSepBlur = () => {
    if (blurCurr) return;
    blurCurr = runSepPass(luma);
    blurPrev = runSepPass(prevLuma);
  };
  const ensureBlur3 = () => {
    if (blur3Curr) return;
    ensureSepBlur();
    const t = runSepPass(blurCurr);
    blur3Curr = runSepPass(t, t);
    const tP = runSepPass(blurPrev);
    blur3Prev = runSepPass(tP, tP);
  };

  if (hasWideSceneTexture && !allowNegGain && (useDx !== 0.0 || useDy !== 0.0) && best.informative && best.mse >= 0.05 && best.mse <= 6.0 && R > 0) {
    ensureSepBlur();
    const b1PtsX = [], b1PtsY = [];
    const isSub0 = Math.abs(useDx - Math.round(useDx)) > 1e-4 || Math.abs(useDy - Math.round(useDy)) > 1e-4;
    for (let i = 0; i < R; i++) {
      const x = rerankX[i], y = rerankY[i];
      const px = x - useDx, py = y - useDy;
      if (px < 1.0 || px >= w - 2.0 || py < 1.0 || py >= h - 2.0) continue;
      const pv = isSub0 ? sampleBicubicLumaCal(prevLuma, w, h, px, py) : prevLuma[Math.round(py) * w + Math.round(px)];
      if (Math.abs(luma[y * w + x] - (useGain * pv + useOffset)) <= 6.0) { b1PtsX.push(x); b1PtsY.push(y); }
    }
    if (b1PtsX.length >= 24) {
      const evalB1 = (dx, dy) => {
        let sc=0,sp=0,scc=0,spp=0,scp=0,n=0;
        for (let i = 0; i < b1PtsX.length; i++) {
          const x=b1PtsX[i], y=b1PtsY[i], sx=x-dx, sy=y-dy;
          if (sx<1||sx>w-2||sy<1||sy>h-2) continue;
          const c=blurCurr[y*w+x], p=sampleBicubicLumaCal(blurPrev,w,h,sx,sy);
          sc+=c; sp+=p; scc+=c*c; spp+=p*p; scp+=c*p; n++;
        }
        if (n < 16) return { dx, dy, mse: Infinity, informative: false };
        const mc=sc/n, mp=sp/n, vc=Math.max(0,scc/n-mc*mc), vp=Math.max(0,spp/n-mp*mp), cov=scp/n-mc*mp;
        if (cov<=0 || vc<=0.25 || vp<=0.25) return { dx, dy, mse: Infinity, informative: false };
        const g=Math.max(0.25,Math.min(4.0,cov/vp));
        return { dx, dy, mse: Math.max(0, vc+g*g*vp-2*g*cov), informative: true };
      };
      let cxB=useDx, cyB=useDy, bBest1=evalB1(cxB, cyB);
      for (const st of [0.1, 0.05]) {
        const bx0=cxB, by0=cyB;
        for (let ky=-1;ky<=1;ky++) for (let kx=-1;kx<=1;kx++) {
          if (kx===0&&ky===0) continue;
          const f=evalB1(bx0+kx*st, by0+ky*st);
          if (f.informative && (!bBest1.informative || f.mse < bBest1.mse)) { bBest1=f; cxB=bx0+kx*st; cyB=by0+ky*st; }
        }
      }
      if (bBest1.informative && (Math.abs(cxB-useDx)>1e-4 || Math.abs(cyB-useDy)>1e-4)) {
        const rfRaw = evalShiftOn(b1PtsX, b1PtsY, b1PtsX.length, cxB, cyB);
        if (rfRaw.informative && rfRaw.mse <= best.mse * 1.15) {
          useDx = cxB; useDy = cyB; useGain = rfRaw.gain; useOffset = rfRaw.offset; best = rfRaw;
        }
      }
    }
  }
  if (hasWideSceneTexture && !allowNegGain && (useDx !== 0.0 || useDy !== 0.0) && best.informative && best.mse <= 6.0) {
    ensureBlur3();
    const gbx = [], gby = [];
    const gStX = Math.max(4, Math.floor(w / 18)), gStY = Math.max(4, Math.floor(h / 12));
    for (let y = gStY; y < h - gStY; y += gStY) for (let x = gStX; x < w - gStX; x += gStX) { gbx.push(x); gby.push(y); }
    let gMean = 0.0, gVar = 0.0, gGradSq = 0.0, gGN = 0;
    if (gbx.length > 0) {
      for (let i = 0; i < gbx.length; i++) gMean += blur3Curr[gby[i] * w + gbx[i]];
      gMean /= gbx.length;
      for (let i = 0; i < gbx.length; i++) {
        const x = gbx[i], y = gby[i], d = blur3Curr[y * w + x] - gMean;
        gVar += d * d;
        if (x >= 2 && x < w - 2 && y >= 2 && y < h - 2) {
          const gx = blur3Curr[y * w + (x + 2)] - blur3Curr[y * w + (x - 2)];
          const gy = blur3Curr[(y + 2) * w + x] - blur3Curr[(y - 2) * w + x];
          gGradSq += gx * gx + gy * gy; gGN++;
        }
      }
      gVar /= gbx.length;
      gGradSq = gGN > 0 ? gGradSq / gGN : 0.0;
    }
    if (gVar >= 25.0 && gGradSq >= 16.0 && gbx.length >= 24) {
      const evU3 = (dx, dy, xs, ys) => {
        let sc=0,sp=0,scc=0,spp=0,scp=0,n=0;
        for (let i=0;i<xs.length;i++) {
          const x=xs[i],y=ys[i],sx=x-dx,sy=y-dy;
          if(sx<2||sx>w-3||sy<2||sy>h-3)continue;
          const c=blur3Curr[y*w+x], p=sampleBicubicLumaCal(blur3Prev,w,h,sx,sy);
          sc+=c;sp+=p;scc+=c*c;spp+=p*p;scp+=c*p;n++;
        }
        if (n<16) return null;
        const mc=sc/n,mp=sp/n,vc=Math.max(0,scc/n-mc*mc),vp=Math.max(0,spp/n-mp*mp),cov=scp/n-mc*mp;
        if (cov<=0||vc<=0.5||vp<=0.5) return null;
        const g=Math.max(0.25,Math.min(4.0,cov/vp));
        return { mse: Math.max(0,vc+g*g*vp-2*g*cov), g, o: mc - g*mp };
      };
      const f0 = evU3(useDx, useDy, gbx, gby);
      if (f0) {
        const inX = [], inY = [];
        for (let i=0;i<gbx.length;i++) {
          const x=gbx[i],y=gby[i],sx=x-useDx,sy=y-useDy;
          if(sx<2||sx>w-3||sy<2||sy>h-3)continue;
          if (Math.abs(blur3Curr[y*w+x] - (f0.g * sampleBicubicLumaCal(blur3Prev,w,h,sx,sy) + f0.o)) <= 6.0) { inX.push(x); inY.push(y); }
        }
        if (inX.length >= 16) {
          let cx3 = useDx, cy3 = useDy, b3 = evU3(cx3, cy3, inX, inY);
          if (b3) {
            for (const st of [0.2, 0.1, 0.05, 0.025, 0.01]) {
              const bx0 = cx3, by0 = cy3;
              for (let ky=-1;ky<=1;ky++) for (let kx=-1;kx<=1;kx++) {
                if (kx===0&&ky===0) continue;
                const f = evU3(bx0+kx*st, by0+ky*st, inX, inY);
                if (f && f.mse < b3.mse) { b3 = f; cx3 = bx0+kx*st; cy3 = by0+ky*st; }
              }
            }
            if ((Math.abs(cx3-useDx)>1e-4 || Math.abs(cy3-useDy)>1e-4) && R > 0) {
              const rfRaw = evalShiftOn(rerankX, rerankY, R, cx3, cy3);
              if (rfRaw.informative && rfRaw.mse <= Math.max(1.5, best.mse * 1.15)) {
                if (rfRaw.mse < best.mse) { useDx = cx3; useDy = cy3; useGain = rfRaw.gain; useOffset = rfRaw.offset; best = rfRaw; }
              }
            }
          }
        }
      }
    }
  }
  // Detect sparse outliers (e.g. sp=20 periodic grid where 1 margin column differs, or pan > 12)
  let curOutlierCount = 0;
  let curCheckedTex = 0;
  {
    const sampPAtCur = (fx, fy) => {
      if (Math.abs(useDx - Math.round(useDx)) < 1e-6 && Math.abs(useDy - Math.round(useDy)) < 1e-6) {
        const px = fx - Math.round(useDx), py = fy - Math.round(useDy);
        if (px < 0 || px >= w || py < 0 || py >= h) return NaN;
        return prevLuma[py * w + px];
      }
      const sx = fx - useDx, sy = fy - useDy;
      if (sx < 1.0 || sx > w - 2.0 || sy < 1.0 || sy > h - 2.0) return NaN;
      return sampleBicubicLumaCal(prevLuma, w, h, sx, sy);
    };
    const stC = Math.max(1, Math.floor(currTexX.length / 80));
    for (let i = 0; i < currTexX.length; i += stC) {
      const x = currTexX[i], y = currTexY[i];
      const pv = sampPAtCur(x, y);
      if (!Number.isFinite(pv)) continue;
      curCheckedTex++;
      if (Math.abs(luma[y * w + x] - (useGain * pv + useOffset)) >= 8.0) curOutlierCount++;
    }
    const stP = Math.max(1, Math.floor(prevTexX.length / 80));
    for (let i = 0; i < prevTexX.length; i += stP) {
      const px = prevTexX[i], py = prevTexY[i];
      const cx = Math.round(px + useDx), cy = Math.round(py + useDy);
      if (cx < 0 || cx >= w || cy < 0 || cy >= h) {
        if (Math.abs(luma[py * w + px] - (useGain * prevLuma[py * w + px] + useOffset)) >= 8.0) {
          curOutlierCount++; curCheckedTex++;
        }
        continue;
      }
      const pv = sampPAtCur(cx, cy);
      if (!Number.isFinite(pv)) continue;
      curCheckedTex++;
      if (Math.abs(luma[cy * w + cx] - (useGain * pv + useOffset)) >= 8.0) curOutlierCount++;
    }
  }
  const hasCurOutliers = curCheckedTex > 0 && (curOutlierCount >= 4 || curOutlierCount / curCheckedTex >= 0.02);

  if (
    hasWideSceneTexture &&
    S > 0 &&
    (zeroCompareMse > 1e-4 || hasCurOutliers) &&
    !useStaticInvertedZero &&
    (!best.informative ||
      (best.corr !== undefined ? best.corr < 0.994 : false) ||
      best.mse > 3.0 ||
      ((Math.abs(useDx) >= maxPan - 1.5 || Math.abs(useDy) >= maxPan - 1.5) && best.mse > 0.25) ||
      (useDx === 0.0 && useDy === 0.0 && zeroRerank.mse > 3.0) ||
      hasCurOutliers)
  ) {
    ensureBlur3();
    const evalOnBufs = (cBuf, pBuf, xs, ys, cnt, dx, dy) => {
      if (cnt <= 0) return { dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
      const isInt = Math.abs(dx - Math.round(dx)) < 1e-6 && Math.abs(dy - Math.round(dy)) < 1e-6;
      const idxS = Math.round(dx), idyS = Math.round(dy);
      let sc = 0.0, sp = 0.0, scc = 0.0, spp = 0.0, scp = 0.0, n = 0;
      if (isInt) {
        for (let i = 0; i < cnt; i++) {
          const x = xs[i], y = ys[i];
          const px = x - idxS, py = y - idyS;
          if (px < 0 || px >= w || py < 0 || py >= h) continue;
          const c = cBuf[y * w + x], p = pBuf[py * w + px];
          sc += c; sp += p; scc += c * c; spp += p * p; scp += c * p;
          n++;
        }
      } else {
        for (let i = 0; i < cnt; i++) {
          const x = xs[i], y = ys[i];
          const sx = x - dx, sy = y - dy;
          if (sx < 1.0 || sx > w - 2.0 || sy < 1.0 || sy > h - 2.0) continue;
          const c = cBuf[y * w + x], p = sampleBicubicLumaCal(pBuf, w, h, sx, sy);
          sc += c; sp += p; scc += c * c; spp += p * p; scp += c * p;
          n++;
        }
      }
      if (n < Math.max(8, Math.floor(cnt * (inStage1Sub && xs !== stage1GridX && geomCovOk(dx, dy) ? 0.35 : 0.55)))) {
        return { dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
      }
      const mc = sc / n, mp = sp / n;
      const vc = Math.max(0.0, scc / n - mc * mc);
      const vp = Math.max(0.0, spp / n - mp * mp);
      const cov = scp / n - mc * mp;
      const varBal = vc > 0.10 && vp > 0.10 && vp >= vc * 0.25 && vc >= vp * 0.25;
      if (cov <= 0.0 || !varBal) {
        return { dx, dy, mse: Infinity, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
      }
      const gain = Math.max(0.25, Math.min(4.0, cov / vp));
      const offset = mc - gain * mp;
      const mse = Math.max(0.0, vc + gain * gain * vp - 2.0 * gain * cov);
      const corr = cov / Math.sqrt(vc * vp);
      return { dx, dy, mse, gain, offset, corr, informative: true };
    };
    const evalTrimmedShift = (cBuf, pBuf, xs, ys, cnt, dx, dy) => {
      if (cnt <= 0) return { dx, dy, mse: Infinity, trimMse: Infinity, rankCost: Infinity, coverage: 0.0, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
      const isInt = Math.abs(dx - Math.round(dx)) < 1e-6 && Math.abs(dy - Math.round(dy)) < 1e-6;
      const idxS = Math.round(dx), idyS = Math.round(dy);
      const cs = [], ps = [];
      let sc = 0.0, sp = 0.0, scc = 0.0, spp = 0.0, scp = 0.0;
      for (let i = 0; i < cnt; i++) {
        const x = xs[i], y = ys[i];
        let p;
        if (isInt) {
          const px = x - idxS, py = y - idyS;
          if (px < 0 || px >= w || py < 0 || py >= h) continue;
          p = pBuf[py * w + px];
        } else {
          const sx = x - dx, sy = y - dy;
          if (sx < 1.0 || sx > w - 2.0 || sy < 1.0 || sy > h - 2.0) continue;
          p = sampleBicubicLumaCal(pBuf, w, h, sx, sy);
        }
        const c = cBuf[y * w + x];
        cs.push(c); ps.push(p);
        sc += c; sp += p; scc += c * c; spp += p * p; scp += c * p;
      }
      const n = cs.length;
      if (n < Math.max(8, Math.floor(cnt * (inStage1Sub && xs !== stage1GridX && geomCovOk(dx, dy) ? 0.35 : 0.55)))) {
        return { dx, dy, mse: Infinity, trimMse: Infinity, rankCost: Infinity, coverage: 0.0, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
      }
      const mc = sc / n, mp = sp / n;
      const vc = Math.max(0.0, scc / n - mc * mc);
      const vp = Math.max(0.0, spp / n - mp * mp);
      const cov = scp / n - mc * mp;
      const varBal = vc > 0.10 && vp > 0.10 && vp >= vc * 0.25 && vc >= vp * 0.25;
      if (cov <= 0.0 || !varBal) {
        return { dx, dy, mse: Infinity, trimMse: Infinity, rankCost: Infinity, coverage: 0.0, gain: 1.0, offset: 0.0, corr: 0.0, informative: false };
      }
      const gain = Math.max(0.25, Math.min(4.0, cov / vp));
      const offset = mc - gain * mp;
      const errs = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const d = cs[i] - (gain * ps[i] + offset);
        errs[i] = d * d;
      }
      errs.sort();
      const keepN = Math.max(8, Math.floor(n * 0.78));
      let tSum = 0.0;
      for (let i = 0; i < keepN; i++) tSum += errs[i];
      const thresh = errs[keepN - 1] + 1e-6;
      let isc = 0.0, isp = 0.0, iscc = 0.0, ispp = 0.0, iscp = 0.0, inn = 0;
      for (let i = 0; i < n; i++) {
        const d = cs[i] - (gain * ps[i] + offset);
        if (d * d > thresh) continue;
        const c = cs[i], p = ps[i];
        isc += c; isp += p; iscc += c * c; ispp += p * p; iscp += c * p;
        inn++;
      }
      let trimMse = tSum / keepN, inCorr = cov / Math.sqrt(vc * vp), inGain = gain, inOffset = offset;
      if (inn >= 8) {
        const imc = isc / inn, imp = isp / inn;
        const ivc = Math.max(0.0, iscc / inn - imc * imc);
        const ivp = Math.max(0.0, ispp / inn - imp * imp);
        const icov = iscp / inn - imc * imp;
        if (ivc > 0.10 && ivp > 0.10 && icov > 0.0) {
          inGain = Math.max(0.25, Math.min(4.0, icov / ivp));
          inOffset = imc - inGain * imp;
          trimMse = Math.max(0.0, ivc + inGain * inGain * ivp - 2.0 * inGain * icov);
          inCorr = icov / Math.sqrt(ivc * ivp);
        }
      }
      const fullMse = Math.max(0.0, vc + gain * gain * vp - 2.0 * gain * cov);
      const coverage = (Math.max(0.0, w - Math.abs(dx)) * Math.max(0.0, h - Math.abs(dy))) / (w * h);
      const rankCost = trimMse + 2.0 * (1.0 - coverage) + 1e-4 * Math.hypot(dx, dy);
      return { dx, dy, mse: fullMse, trimMse, rankCost, coverage, gain: inGain, offset: inOffset, corr: inCorr, informative: true };
    };
    const evalBlurShiftOn = (xs, ys, cnt, dx, dy) => evalOnBufs(blur3Curr, blur3Prev, xs, ys, cnt, dx, dy);
    const evalBlurTrimmedOn = (xs, ys, cnt, dx, dy) => evalTrimmedShift(blur3Curr, blur3Prev, xs, ys, cnt, dx, dy);
    const compareTrimmed = (a, b) => a.rankCost - b.rankCost;
    const bx = [], by = [], bUsed = new Uint8Array(N);
    const bTexX = [], bTexY = [];
    const bGridStepX = Math.max(4, Math.floor((w - 8) / 24));
    const bGridStepY = Math.max(4, Math.floor((h - 8) / 16));
    for (let gy = 4; gy <= h - 5; gy += bGridStepY) {
      for (let gx = 4; gx <= w - 5; gx += bGridStepX) {
        const idx = gy * w + gx;
        if (!bUsed[idx]) { bUsed[idx] = 1; bx.push(gx); by.push(gy); }
      }
    }
    const texCap = 90;
    const addPts = (pxArr, pyArr, limit) => {
      if (!pxArr.length) return;
      const st = Math.max(1, Math.ceil(pxArr.length / limit));
      for (let i = 0; i < pxArr.length; i += st) {
        const px = pxArr[i], py = pyArr[i], idx = py * w + px;
        if (!bUsed[idx]) { bUsed[idx] = 1; bx.push(px); by.push(py); bTexX.push(px); bTexY.push(py); }
      }
    };
    addPts(currTexX, currTexY, texCap);
    addPts(prevTexX, prevTexY, texCap);
    // Also add stratified mismatch points at current (useDx, useDy) so sparse margin shifts (sp=20) cannot hide!
    const diffPtsX = [], diffPtsY = [];
    for (let y = 4; y <= h - 5 && diffPtsX.length < 180; y += 3) {
      for (let x = 4; x <= w - 5 && diffPtsX.length < 180; x += 3) {
        if (Math.abs(luma[y * w + x] - prevLuma[y * w + x]) >= 8.0) {
          diffPtsX.push(x); diffPtsY.push(y);
        }
      }
    }
    addPts(diffPtsX, diffPtsY, 60);
    // Prefer unweighted textured points when textured points exist (avoids flat-sky dilution), plus full grid+tex
    const bC = bx.length;
    inStage1Sub = true;
    stage1GridX = bx;
    const cStep = w >= 960 ? 3 : 2;
    const coarsePtsX = [], coarsePtsY = [];
    {
      const srcX = bTexX.length >= 24 ? bTexX : bx;
      const srcY = bTexY.length >= 24 ? bTexY : by;
      const st = Math.max(1, Math.ceil(srcX.length / 96));
      for (let i = 0; i < srcX.length; i += st) { coarsePtsX.push(srcX[i]); coarsePtsY.push(srcY[i]); }
    }
    // Precompute magnitude weights on coarsePts so sign-inverted features (e.g. partial3_pan) still surface in coarse search
    let mcC = 0.0, mpC = 0.0;
    for (let i = 0; i < coarsePtsX.length; i++) {
      const idx = coarsePtsY[i] * w + coarsePtsX[i];
      mcC += blur3Curr[idx]; mpC += blur3Prev[idx];
    }
    mcC /= Math.max(1, coarsePtsX.length); mpC /= Math.max(1, coarsePtsX.length);
    const coarseAbsC = new Float32Array(coarsePtsX.length);
    for (let i = 0; i < coarsePtsX.length; i++) {
      coarseAbsC[i] = Math.abs(blur3Curr[coarsePtsY[i] * w + coarsePtsX[i]] - mcC);
    }
    const evalCoarseMag = (dx, dy) => {
      let s = 0.0, n = 0;
      const cLen = coarsePtsX.length;
      for (let i = 0; i < cLen; i++) {
        const px = coarsePtsX[i] - dx, py = coarsePtsY[i] - dy;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;
        s += coarseAbsC[i] * Math.abs(blur3Prev[py * w + px] - mpC);
        n++;
      }
      return n >= Math.max(8, Math.floor(cLen * 0.55)) ? s / cLen : 0.0;
    };
    const intCand = [];
    const magCand = [];
    for (let dy = -maxPanY; dy <= maxPanY; dy += cStep) {
      for (let dx = -maxPanX; dx <= maxPanX; dx += cStep) {
        const f = evalBlurShiftOn(coarsePtsX, coarsePtsY, coarsePtsX.length, dx, dy);
        if (f.informative) intCand.push(f);
        const ms = evalCoarseMag(dx, dy);
        if (ms > 0.0) magCand.push({ dx, dy, ms });
      }
    }
    // Also check dense inner [-8..8] in case sharp low-sigma features exist
    for (let dy = -8; dy <= 8; dy++) {
      for (let dx = -8; dx <= 8; dx++) {
        if (Math.abs(dx) % cStep === 0 && Math.abs(dy) % cStep === 0) continue;
        const f = evalBlurShiftOn(coarsePtsX, coarsePtsY, coarsePtsX.length, dx, dy);
        if (f.informative) intCand.push(f);
        const ms = evalCoarseMag(dx, dy);
        if (ms > 0.0) magCand.push({ dx, dy, ms });
      }
    }
    if (intCand.length > 0 || magCand.length > 0) {
      intCand.sort(compareFit);
      magCand.sort((a, b) => b.ms - a.ms || Math.hypot(a.dx, a.dy) - Math.hypot(b.dx, b.dy));
      const coarseSeeds = [];
      for (let i = 0; i < intCand.length && i < 16; i++) coarseSeeds.push(intCand[i]);
      for (let i = 0; i < magCand.length && i < 6; i++) coarseSeeds.push(magCand[i]);
      const seenInt = new Set();
      const refinedInt = [];
      for (let i = 0; i < coarseSeeds.length; i++) {
        for (let ky = -cStep; ky <= cStep; ky++) {
          for (let kx = -cStep; kx <= cStep; kx++) {
            const dx = coarseSeeds[i].dx + kx, dy = coarseSeeds[i].dy + ky;
            if (Math.abs(dx) > maxPanX || Math.abs(dy) > maxPanY) continue;
            const key = dx + "," + dy;
            if (seenInt.has(key)) continue;
            seenInt.add(key);
            const f = evalBlurTrimmedOn(bx, by, bC, dx, dy);
            if (f.informative) refinedInt.push(f);
          }
        }
      }
      refinedInt.sort(compareTrimmed);
      const seeds = [];
      for (let i = 0; i < refinedInt.length && i < 16; i++) seeds.push({ dx: refinedInt[i].dx, dy: refinedInt[i].dy });
      seeds.push({ dx: 0, dy: 0 });
      if (best.informative && (Math.abs(best.dx) > 1e-4 || Math.abs(best.dy) > 1e-4)) seeds.push({ dx: best.dx, dy: best.dy });
      if (useDx !== 0.0 || useDy !== 0.0) seeds.push({ dx: useDx, dy: useDy });
      if (refinedInt.length > 0) {
        const r0 = refinedInt[0];
        if (r0.dx !== 0) seeds.push({ dx: 0, dy: r0.dy });
        if (r0.dy !== 0) seeds.push({ dx: r0.dx, dy: 0 });
      }
      const evalPtsX = bTexX.length >= 24 ? bTexX : bx;
      const evalPtsY = bTexY.length >= 24 ? bTexY : by;
      const evalCnt = evalPtsX.length;
      const bCands = [];
      let bBest = null;
      for (const sd of seeds) {
        let cx = sd.dx, cy = sd.dy;
        const fSeed = evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, cx, cy);
        const useTrimSub = true;
        const evalSub = (dx, dy) => useTrimSub
          ? evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, dx, dy)
          : evalBlurShiftOn(evalPtsX, evalPtsY, evalCnt, dx, dy);
        const cmpSub = (a, b) => useTrimSub ? compareTrimmed(a, b) : compareFit(a, b);
        let loc = evalSub(cx, cy);
        for (const step of [0.5, 0.2, 0.1, 0.05]) {
          const baseX = cx, baseY = cy;
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              if (kx === 0 && ky === 0) continue;
              const tx = baseX + kx * step, ty = baseY + ky * step;
              const f = evalSub(tx, ty);
              if (f.informative && (!loc.informative || cmpSub(f, loc) < 0)) {
                loc = f; cx = tx; cy = ty;
              }
            }
          }
        }
        if (loc.informative) {
          const locFull = evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, cx, cy);
          const cand = { ...loc, trimMse: locFull.trimMse, rankCost: locFull.rankCost, coverage: locFull.coverage };
          const takeCand = !bBest || !bBest.informative || cand.rankCost < bBest.rankCost;
          bCands.push(cand);
          if (takeCand) bBest = cand;
        }
      }
      if (!bBest) bBest = evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, 0, 0);
      const bZeroFull = evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, 0, 0);
      const bCurFull = (useDx === 0.0 && useDy === 0.0) ? bZeroFull : evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, useDx, useDy);
      stage1AltShifts = refinedInt.filter((z) => z.trimMse <= 0.05 && z.corr >= 0.99).slice(0, 4);
      // Check how textured the uniform bx grid is (prevents pointGrid flat-sky wash-out)
      let bxMean = 0.0, bxVar = 0.0, bxGradSq = 0.0, bxGN = 0;
      if (bC > 0) {
        for (let i = 0; i < bC; i++) bxMean += blur3Curr[by[i] * w + bx[i]];
        bxMean /= bC;
        for (let i = 0; i < bC; i++) {
          const x = bx[i], y = by[i], d = blur3Curr[y * w + x] - bxMean;
          bxVar += d * d;
          if (x >= 2 && x < w - 2 && y >= 2 && y < h - 2) {
            const gx = blur3Curr[y * w + (x + 2)] - blur3Curr[y * w + (x - 2)];
            const gy = blur3Curr[(y + 2) * w + x] - blur3Curr[(y - 2) * w + x];
            bxGradSq += gx * gx + gy * gy; bxGN++;
          }
        }
        bxVar /= bC;
        bxGradSq = bxGN > 0 ? bxGradSq / bxGN : 0.0;
      }
      const polishOnBxBlur3 = (sx0, sy0) => {
        if (bxVar < 25.0 || bxGradSq < 16.0 || bC < 16) return null;
        const fT = evalBlurTrimmedOn(bx, by, bC, sx0, sy0);
        if (!fT.informative) return null;
        const inX = [], inY = [];
        for (let i = 0; i < bC; i++) {
          const x = bx[i], y = by[i], px = x - sx0, py = y - sy0;
          if (px < 2.0 || px > w - 3.0 || py < 2.0 || py > h - 3.0) continue;
          const pv = sampleBicubicLumaCal(blur3Prev, w, h, px, py);
          if (Math.abs(blur3Curr[y * w + x] - (fT.gain * pv + fT.offset)) <= 6.0) { inX.push(x); inY.push(y); }
        }
        if (inX.length < 16) return null;
        let cxB = sx0, cyB = sy0, locB = evalOnBufs(blur3Curr, blur3Prev, inX, inY, inX.length, cxB, cyB);
        if (!locB.informative) return null;
        for (const st of [0.5, 0.2, 0.1, 0.05, 0.025, 0.01]) {
          const bx0 = cxB, by0 = cyB;
          for (let ky = -1; ky <= 1; ky++) for (let kx = -1; kx <= 1; kx++) {
            if (kx === 0 && ky === 0) continue;
            const f = evalOnBufs(blur3Curr, blur3Prev, inX, inY, inX.length, bx0 + kx * st, by0 + ky * st);
            if (f.informative && f.mse < locB.mse) { locB = f; cxB = bx0 + kx * st; cyB = by0 + ky * st; }
          }
        }
        return { dx: cxB, dy: cyB, mse: locB.mse };
      };
      if (bxVar >= 25.0 && bxGradSq >= 16.0 && bCands.length > 0) {
        for (let ci = 0; ci < bCands.length; ci++) {
          const c = bCands[ci];
          if (!c.informative || c.trimMse > 0.25) continue;
          const pB = polishOnBxBlur3(c.dx, c.dy);
          if (!pB) continue;
          const chkT = evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, pB.dx, pB.dy);
          if (chkT.informative && chkT.trimMse <= Math.max(0.05, c.trimMse * 1.5)) {
            bCands[ci] = chkT;
            if (Math.abs(c.dx - bBest.dx) < 1e-4 && Math.abs(c.dy - bBest.dy) < 1e-4) bBest = chkT;
          }
        }
      }
      if (bBest && bBest.informative && bBest.trimMse <= 0.15 && bCands.length > 1) {
        for (const c of bCands) {
          if (!c.informative || c.trimMse > 0.15 || c.coverage < bBest.coverage - 0.12) continue;
          const fc = evalBlurTrimmedOn(bx, by, bC, c.dx, c.dy);
          const ft = evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, c.dx, c.dy);
          const fb = evalBlurTrimmedOn(bx, by, bC, bBest.dx, bBest.dy);
          const fbt = evalBlurTrimmedOn(evalPtsX, evalPtsY, evalCnt, bBest.dx, bBest.dy);
          if (bBest.trimMse > 0.005 && fc.informative && ft.informative && fb.informative && fbt.informative &&
              fc.coverage >= fb.coverage - 0.12 &&
              fb.trimMse > 0.005 && fc.mse + ft.mse < (fb.mse + fbt.mse) * 0.45 && fc.mse <= 0.25 && ft.mse <= 1.0) {
            bBest = c;
          }
        }
        const dkX = [], dkY = [];
        for (let y = 4; y < h - 4 && dkX.length < 64; y++) {
          for (let x = 4; x < w - 4 && dkX.length < 64; x++) {
            const idx = y * w + x, c = luma[idx];
            if (c <= luma[idx - 1] && c <= luma[idx + 1] && c <= luma[idx - w] && c <= luma[idx + w] &&
                c + 8.0 <= luma[idx - 2] && c + 8.0 <= luma[idx + 2] && c + 8.0 <= luma[idx - 2 * w] && c + 8.0 <= luma[idx + 2 * w]) {
              dkX.push(x); dkY.push(y);
            }
          }
        }
        if (dkX.length > 0 && bBest.trimMse <= 0.10) {
          ensureSepBlur();
          const spotB1Mse = (cand, gain, off) => {
            let se = 0.0, n = 0;
            const isS = Math.abs(cand.dx - Math.round(cand.dx)) > 1e-4 || Math.abs(cand.dy - Math.round(cand.dy)) > 1e-4;
            const rx = Math.round(cand.dx), ry = Math.round(cand.dy);
            for (let i = 0; i < dkX.length; i++) {
              const x = dkX[i], y = dkY[i], px = isS ? x - cand.dx : x - rx, py = isS ? y - cand.dy : y - ry;
              if (px < 1.0 || px >= w - 2.0 || py < 1.0 || py >= h - 2.0) continue;
              const p = isS ? sampleBicubicLumaCal(blurPrev, w, h, px, py) : blurPrev[py * w + px];
              const d = blurCurr[y * w + x] - (gain * p + off);
              se += d * d; n++;
            }
            return n > 0 ? se / n : Infinity;
          };
          const fb0 = evalBlurTrimmedOn(bx, by, bC, bBest.dx, bBest.dy);
          if (fb0.informative) {
            let smB = spotB1Mse(bBest, fb0.gain, fb0.offset);
            for (const c of bCands) {
              if (!c.informative || c.trimMse > 0.10 || Math.hypot(c.dx - bBest.dx, c.dy - bBest.dy) < 4.0) continue;
              const fc0 = evalBlurTrimmedOn(bx, by, bC, c.dx, c.dy);
              if (!fc0.informative || fc0.mse > 0.15) continue;
              const smC = spotB1Mse(c, fc0.gain, fc0.offset);
              if (smC <= 16.0 && smC < smB * 0.25) { bBest = c; smB = smC; }
            }
          }
        }
      }
      if (bBest && bBest.informative && bBest.trimMse <= 0.10 && refinedInt.length > 0) {
        const ckX = bx.slice(), ckY = by.slice();
        const ckSeen = new Uint8Array(N);
        for (let i = 0; i < ckX.length; i++) ckSeen[ckY[i] * w + ckX[i]] = 1;
        let dAdded = 0;
        for (let y = 3; y < h - 3 && dAdded < 48; y += 2) {
          for (let x = 3; x < w - 3 && dAdded < 48; x += 2) {
            const idx = y * w + x;
            if (ckSeen[idx]) continue;
            const c = luma[idx];
            if (c + 8.0 <= luma[idx - 1] && c + 8.0 <= luma[idx + 1] && c + 8.0 <= luma[idx - w] && c + 8.0 <= luma[idx + w]) {
              ckSeen[idx] = 1; ckX.push(x); ckY.push(y); dAdded++;
            }
          }
        }
        const fbCurR = evalBlurTrimmedOn(bx, by, bC, bBest.dx, bBest.dy);
        const rawCur = evalTrimmedShift(luma, prevLuma, ckX, ckY, ckX.length, bBest.dx, bBest.dy);
        if (!rawCur.informative || rawCur.mse > 0.05) {
          let rawBest = null;
          for (let ri = 0; ri < refinedInt.length; ri++) {
            const rf = refinedInt[ri];
            if (rf.trimMse > 0.10) continue;
            const fbRf = evalBlurTrimmedOn(bx, by, bC, rf.dx, rf.dy);
            if (fbCurR.informative && (!fbRf.informative || fbRf.mse > Math.max(0.02, fbCurR.mse * 2.0))) continue;
            const rc = evalTrimmedShift(luma, prevLuma, ckX, ckY, ckX.length, rf.dx, rf.dy);
            if (!rc.informative || rc.mse > 0.05) continue;
            if (!rawBest || rc.mse + 0.02 * (1.0 - rc.coverage) + 1e-4 * Math.hypot(rc.dx, rc.dy) < rawBest.score) {
              rawBest = { rc, rf, score: rc.mse + 0.02 * (1.0 - rc.coverage) + 1e-4 * Math.hypot(rc.dx, rc.dy) };
            }
          }
          if (rawBest && (!rawCur.informative || rawBest.rc.mse + 0.05 < rawCur.mse)) {
            bBest = rawBest.rf;
          }
        }
      }
      const bBestFull = bBest;
      const candAccepted = acceptsBgConsensus(bBestFull.dx, bBestFull.dy);
      if (
        candAccepted &&
        bBestFull.informative && bBestFull.trimMse <= 6.0 &&
        bBestFull.corr >= 0.85 &&
        bBestFull.trimMse <= 25.0 &&
        (Math.abs(bBestFull.dx - useDx) > 1e-4 || Math.abs(bBestFull.dy - useDy) > 1e-4) &&
        (!bZeroFull.informative || bBestFull.trimMse < bZeroFull.trimMse * 0.45) &&
        (() => {
          if (useDx !== 0.0 || useDy !== 0.0 || bC < 16) return true;
          const ez = [], ec = [];
          const isS = Math.abs(bBestFull.dx - Math.round(bBestFull.dx)) > 1e-4 || Math.abs(bBestFull.dy - Math.round(bBestFull.dy)) > 1e-4;
          const iDx = Math.round(bBestFull.dx), iDy = Math.round(bBestFull.dy);
          for (let i = 0; i < bC; i++) {
            const x = bx[i], y = by[i], c = luma[y * w + x];
            ez.push(Math.abs(c - (useGain * prevLuma[y * w + x] + useOffset)));
            const sx = x - bBestFull.dx, sy = y - bBestFull.dy;
            const ok = isS ? (sx >= 1.0 && sx <= w - 2.0 && sy >= 1.0 && sy <= h - 2.0) : (x - iDx >= 0 && x - iDx < w && y - iDy >= 0 && y - iDy < h);
            if (ok) {
              const p = isS ? sampleBicubicLumaCal(prevLuma, w, h, sx, sy) : prevLuma[(y - iDy) * w + (x - iDx)];
              ec.push(Math.abs(c - (bBestFull.gain * p + bBestFull.offset)));
            }
          }
          ez.sort((a, b) => a - b); ec.sort((a, b) => a - b);
          const medZ = ez[Math.floor(ez.length * 0.50)] || 0;
          const medC = ec.length >= Math.floor(bC * 0.50) ? ec[Math.floor(ec.length * 0.50)] : 999;
          return medZ > 0.50 || medC <= medZ + 1.5;
        })() &&
        (!bCurFull.informative || bBestFull.trimMse < bCurFull.trimMse * 0.65 || (() => {
          if (useDx === 0.0 && useDy === 0.0) return false;
          if (bBestFull.trimMse > 0.05) return false;
          const b1Best = evalTrimmedShift(blurCurr, blurPrev, bx, by, bC, bBestFull.dx, bBestFull.dy);
          const b1Cur = evalTrimmedShift(blurCurr, blurPrev, bx, by, bC, useDx, useDy);
          if (b1Best.informative && b1Cur.informative && b1Cur.trimMse > 0.005 && b1Best.trimMse < b1Cur.trimMse * 0.60 && bBestFull.trimMse <= Math.max(0.03, bCurFull.trimMse * 1.15)) return true;
          const rBest = evalTrimmedShift(luma, prevLuma, bx, by, bC, bBestFull.dx, bBestFull.dy);
          const rCur = evalTrimmedShift(luma, prevLuma, bx, by, bC, useDx, useDy);
          return rBest.informative && rBest.mse <= 0.05 && (!rCur.informative || rCur.mse > 1.0);
        })())
      ) {
        useDx = bBestFull.dx;
        useDy = bBestFull.dy;
        // Refine gain/offset on raw luma when shift is exact integer or smooth, else keep blur3 affine
        const rawCand = evalTrimmedShift(luma, prevLuma, evalPtsX, evalPtsY, evalCnt, useDx, useDy);
        if (rawCand.informative && rawCand.corr >= 0.99 && rawCand.trimMse <= 2.0) {
          useGain = rawCand.gain;
          useOffset = rawCand.offset;
        } else {
          useGain = bBestFull.gain;
          useOffset = bBestFull.offset;
        }
      }
    }
  }
  if (globalThis.__p11Dbg) {
    console.log("  PAN_INFO:", {
      best: { dx: best.dx, dy: best.dy, mse: +best.mse.toFixed(2) },
      useDx, useDy, useGain: +useGain.toFixed(3), useOffset: +useOffset.toFixed(2),
      zeroBgFit: { inf: zeroBgFit.informative, mse: +zeroBgFit.mse.toFixed(3), corr: +zeroBgFit.corr.toFixed(4) },
      bgLen: bgGridX.length, hasTexturedBackground, hasCurOutliers,
    });
  }
  const hasCredibleAltPan =
    hasWideSceneTexture &&
    S > 0 &&
    best.informative &&
    (Math.abs(best.dx - useDx) > 1e-4 || Math.abs(best.dy - useDy) > 1e-4) &&
    best.mse <= zeroCompareMse * 1.05;

  const isSubPan = Math.abs(useDx - Math.round(useDx)) > 1e-4 || Math.abs(useDy - Math.round(useDy)) > 1e-4;
  const roundDx = Math.round(useDx);
  const roundDy = Math.round(useDy);
  const hasPrevSupport = (px, py) =>
    isSubPan ? px >= 1.0 && px < w - 2.0 && py >= 1.0 && py < h - 2.0 : px >= 0 && px < w && py >= 0 && py < h;

  // Refine affine photometry (gain, offset) over balanced grid + textured spots using robust TLS
  {
    const csArr = [], psArr = [];
    const phUsed = new Uint8Array(N);
    const accPh = (x, y) => {
      const idx = y * w + x;
      if (phUsed[idx]) return;
      const py = isSubPan ? y - useDy : y - roundDy;
      const px = isSubPan ? x - useDx : x - roundDx;
      if (!hasPrevSupport(px, py)) return;
      phUsed[idx] = 1;
      csArr.push(luma[idx]);
      psArr.push(isSubPan ? sampleBicubicLumaCal(prevLuma, w, h, px, py) : prevLuma[py * w + px]);
    };
    const y0 = Math.max(2, Math.ceil(useDy) + 2), y1 = Math.min(h - 2, h + Math.floor(useDy) - 2);
    const x0 = Math.max(2, Math.ceil(useDx) + 2), x1 = Math.min(w - 2, w + Math.floor(useDx) - 2);
    const stY = Math.max(2, Math.floor((y1 - y0) / 20));
    const stX = Math.max(2, Math.floor((x1 - x0) / 24));
    for (let y = y0; y < y1; y += stY) for (let x = x0; x < x1; x += stX) accPh(x, y);
    const stT = Math.max(1, Math.ceil(currTexX.length / 160));
    for (let i = 0; i < currTexX.length; i += stT) accPh(currTexX[i], currTexY[i]);
    const fitStats = (cList, pList) => {
      const n = cList.length;
      if (n === 0) return null;
      let sc = 0, sp = 0, scc = 0, spp = 0, scp = 0;
      for (let i = 0; i < n; i++) { const c = cList[i], p = pList[i]; sc += c; sp += p; scc += c * c; spp += p * p; scp += c * p; }
      const mc = sc / n, mp = sp / n, vc = Math.max(0, scc / n - mc * mc), vp = Math.max(0, spp / n - mp * mp), cov = scp / n - mc * mp;
      const corr = vc > 0.25 && vp > 0.25 ? cov / Math.sqrt(vc * vp) : 0.0;
      return { n, mc, mp, vc, vp, cov, corr };
    };
    const st0 = fitStats(csArr, psArr);
    if (st0) {
      let stF = st0;
      if (!allowNegGain && csArr.length >= 40 && st0.vp > 0.25) {
        const diffs = new Float64Array(csArr.length);
        for (let i = 0; i < csArr.length; i++) diffs[i] = csArr[i] - psArr[i];
        const sD = diffs.slice().sort();
        const medOff = sD[Math.floor(csArr.length * 0.5)];
        const eU = new Float64Array(csArr.length);
        for (let i = 0; i < csArr.length; i++) eU[i] = Math.abs(diffs[i] - medOff);
        const sEU = eU.slice().sort();
        const qU = sEU[Math.floor(csArr.length * 0.40)];
        let bestG = 1.0, bestO = medOff, bestQ = qU;
        if (st0.corr >= 0.50) {
          const g0 = Math.max(0.25, Math.min(4.0, st0.cov / st0.vp)), o0 = st0.mc - g0 * st0.mp;
          const eA = new Float64Array(csArr.length);
          for (let i = 0; i < csArr.length; i++) eA[i] = Math.abs(csArr[i] - (g0 * psArr[i] + o0));
          const sEA = eA.slice().sort();
          const qA = sEA[Math.floor(csArr.length * 0.40)];
          if (qA + 0.25 < bestQ) { bestG = g0; bestO = o0; bestQ = qA; }
        }
        const thr = Math.max(3.0, bestQ + 1.0);
        const cIn = [], pIn = [];
        for (let i = 0; i < csArr.length; i++) if (Math.abs(csArr[i] - (bestG * psArr[i] + bestO)) <= thr) { cIn.push(csArr[i]); pIn.push(psArr[i]); }
        if (cIn.length >= 24) { const st1 = fitStats(cIn, pIn); if (st1 && st1.vp > 0.25 && st1.corr >= 0.70) stF = st1; }
      }
      if (stF.vp > 0.25 && (allowNegGain || stF.corr >= 0.65)) {
        const diffV = stF.vc - stF.vp;
        const rawTls = Math.abs(stF.cov) > 1e-6 ? (diffV + Math.sign(stF.cov) * Math.sqrt(diffV * diffV + 4.0 * stF.cov * stF.cov)) / (2.0 * stF.cov) : stF.cov / stF.vp;
        const rawRefine = (!allowNegGain && stF.corr >= 0.82 && stF.vc >= stF.vp * 0.25 && stF.vp >= stF.vc * 0.25) ? rawTls : stF.cov / stF.vp;
        useGain = allowNegGain
          ? Math.max(-4.0, Math.min(-0.25, rawRefine))
          : Math.max(0.25, Math.min(4.0, rawRefine));
        useOffset = stF.mc - useGain * stF.mp;
      } else if (!allowNegGain && !hasWideSceneTexture) {
        useGain = 1.0;
        useOffset = stF.mc - stF.mp;
      }
    }
  }

  let blur3Affine = null;
  let blur1Affine = null;
  const fitBlurAffine = (cBuf, pBuf) => {
    const pxPts = [], pyPts = [];
    const bgStepX = Math.max(4, Math.floor((w - 8) / 20));
    const bgStepY = Math.max(4, Math.floor((h - 8) / 14));
    for (let gy = 4; gy <= h - 5; gy += bgStepY) {
      for (let gx = 4; gx <= w - 5; gx += bgStepX) {
        if (gx - useDx >= 1.0 && gx - useDx < w - 2.0 && gy - useDy >= 1.0 && gy - useDy < h - 2.0) {
          pxPts.push(gx); pyPts.push(gy);
        }
      }
    }
    const addT = (xs, ys, lim) => {
      if (!xs.length) return;
      const st = Math.max(1, Math.ceil(xs.length / lim));
      for (let i = 0; i < xs.length; i += st) {
        const gx = xs[i], gy = ys[i];
        if (gx - useDx >= 1.0 && gx - useDx < w - 2.0 && gy - useDy >= 1.0 && gy - useDy < h - 2.0) {
          pxPts.push(gx); pyPts.push(gy);
        }
      }
    };
    addT(currTexX, currTexY, 150);
    addT(prevTexX, prevTexY, 150);
    let sc = 0.0, sp = 0.0, spp = 0.0, scp = 0.0, n = 0;
    for (let k = 0; k < pxPts.length; k++) {
      const gx = pxPts[k], gy = pyPts[k];
      const bc = cBuf[gy * w + gx];
      const bp = sampleBicubicLumaCal(pBuf, w, h, gx - useDx, gy - useDy);
      sc += bc; sp += bp; spp += bp * bp; scp += bc * bp; n++;
    }
    if (n >= 8) {
      const mc = sc / n, mp = sp / n;
      const vp = Math.max(0.0, spp / n - mp * mp);
      const cov = scp / n - mc * mp;
      if (vp > 0.25 && cov > 0.0) {
        const gain = Math.max(0.25, Math.min(4.0, cov / vp));
        return { gain, offset: mc - gain * mp };
      }
    }
    return { gain: useGain, offset: useOffset };
  };
  const getBlur3Affine = () => {
    if (!blur3Affine) {
      ensureBlur3();
      blur3Affine = fitBlurAffine(blur3Curr, blur3Prev);
    }
    return blur3Affine;
  };
  const getBlur1Affine = () => {
    if (!blur1Affine) {
      ensureSepBlur();
      blur1Affine = fitBlurAffine(blurCurr, blurPrev);
    }
    return blur1Affine;
  };
  const measurePatchFlow = (cBuf, pBuf, ba, cx, cy) => {
    let e0 = 0.0, sGx2 = 0.0, sGy2 = 0.0, sGxGy = 0.0, sGxd = 0.0, sGyd = 0.0, n = 0;
    for (let oy = -2; oy <= 2; oy++) {
      const y = cy + oy;
      if (y < 1 || y >= h - 1) continue;
      const py0 = y - useDy;
      if (py0 < 1.0 || py0 >= h - 2.0) continue;
      for (let ox = -2; ox <= 2; ox++) {
        const x = cx + ox;
        if (x < 1 || x >= w - 1) continue;
        const px0 = x - useDx;
        if (px0 < 1.0 || px0 >= w - 2.0) continue;
        const d = cBuf[y * w + x] - (ba.gain * sampleBicubicLumaCal(pBuf, w, h, px0, py0) + ba.offset);
        e0 += d * d;
        const gx = 0.5 * (cBuf[y * w + x + 1] - cBuf[y * w + x - 1]);
        const gy = 0.5 * (cBuf[(y + 1) * w + x] - cBuf[(y - 1) * w + x]);
        sGx2 += gx * gx; sGy2 += gy * gy; sGxGy += gx * gy;
        sGxd += gx * d; sGyd += gy * d;
        n++;
      }
    }
    if (n < 9) return { n, rms: Infinity, disp: 0.0, corrFrac: 0.0 };
    const rms = Math.sqrt(e0 / n);
    const det = sGx2 * sGy2 - sGxGy * sGxGy;
    let sx = det > 1e-6 ? (sGxd * sGy2 - sGyd * sGxGy) / det : 0.0;
    let sy = det > 1e-6 ? (sGyd * sGx2 - sGxd * sGxGy) / det : 0.0;
    const disp = Math.hypot(sx, sy);
    if (disp > 0.6) { sx *= 0.6 / disp; sy *= 0.6 / disp; }
    let eCorr = 0.0, nCorr = 0;
    for (let oy = -2; oy <= 2; oy++) {
      const y = cy + oy;
      if (y < 1 || y >= h - 1) continue;
      const pyS = y - useDy + sy;
      if (pyS < 1.0 || pyS >= h - 2.0) continue;
      for (let ox = -2; ox <= 2; ox++) {
        const x = cx + ox;
        if (x < 1 || x >= w - 1) continue;
        const pxS = x - useDx + sx;
        if (pxS < 1.0 || pxS >= w - 2.0) continue;
        const dc = cBuf[y * w + x] - (ba.gain * sampleBicubicLumaCal(pBuf, w, h, pxS, pyS) + ba.offset);
        eCorr += dc * dc;
        nCorr++;
      }
    }
    const corrFrac = nCorr >= 9 && e0 > 1e-9 ? 1.0 - (eCorr / nCorr) / (e0 / n) : 0.0;
    return { n, rms, disp, corrFrac };
  };
  const sampBilLuma = (buf, fx, fy) => {
    const x0 = Math.max(0, Math.min(w - 1, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(h - 1, Math.floor(fy)));
    const x1 = x0 + 1 < w ? x0 + 1 : x0;
    const y1 = y0 + 1 < h ? y0 + 1 : y0;
    const tx = fx - Math.floor(fx), ty = fy - Math.floor(fy);
    const r0 = y0 * w, r1 = y1 * w;
    return (buf[r0 + x0] * (1.0 - tx) + buf[r0 + x1] * tx) * (1.0 - ty) + (buf[r1 + x0] * (1.0 - tx) + buf[r1 + x1] * tx) * ty;
  };
  const checkLocalStaticSubpixel = (cx, cy) => {
    // 1. Check exact forward bilinear match (panShift stationary) and reverse bilinear match (stationary 2x2 -> 1x1)
    let eFB = 0.0, maxFB = 0.0, ringEB = 0.0, ringNB = 0, coreMaxFB = 0.0, nFB = 0;
    let eRB = 0.0, maxRB = 0.0, ringERB = 0.0, ringNRB = 0, coreMaxRB = 0.0, nRB = 0;
    for (let oy = -2; oy <= 2; oy++) {
      const y = cy + oy;
      if (y < 0 || y >= h) continue;
      const pyF = y - useDy;
      for (let ox = -2; ox <= 2; ox++) {
        const x = cx + ox;
        if (x < 0 || x >= w) continue;
        const pxF = x - useDx;
        const isRing = Math.abs(ox) === 2 || Math.abs(oy) === 2;
        if (pxF >= 0.0 && pxF <= w - 1.0 && pyF >= 0.0 && pyF <= h - 1.0) {
          const pred = Math.max(0.0, Math.min(255.0, useGain * sampBilLuma(prevLuma, pxF, pyF) + useOffset));
          const df = Math.abs(luma[y * w + x] - pred);
          eFB += df * df;
          if (df > maxFB) maxFB = df;
          if (isRing) { ringEB += df * df; ringNB++; }
          else if (df > coreMaxFB) coreMaxFB = df;
          nFB++;
        }
        const px0 = Math.round(pxF), py0 = Math.round(pyF);
        if (px0 >= 0 && px0 < w && py0 >= 0 && py0 < h) {
          const qx = px0 + useDx, qy = py0 + useDy;
          if (qx >= 0.0 && qx <= w - 1.0 && qy >= 0.0 && qy <= h - 1.0) {
            const invPred = (sampBilLuma(luma, qx, qy) - useOffset) / Math.max(0.25, useGain);
            const dr = Math.abs(prevLuma[py0 * w + px0] - invPred);
            eRB += dr * dr;
            if (dr > maxRB) maxRB = dr;
            if (isRing) { ringERB += dr * dr; ringNRB++; }
            else if (dr > coreMaxRB) coreMaxRB = dr;
            nRB++;
          }
        }
      }
    }
    const rmsFB = nFB >= 9 ? Math.sqrt(eFB / nFB) : Infinity;
    const rmsRB = nRB >= 9 ? Math.sqrt(eRB / nRB) : Infinity;
    const ringRmsFB = ringNB >= 4 ? Math.sqrt(ringEB / ringNB) : 1.0;
    const ringRmsRB = ringNRB >= 4 ? Math.sqrt(ringERB / ringNRB) : 1.0;
    const cRmsFB = Math.sqrt((eFB - ringEB) / Math.max(1, nFB - ringNB));
    const fScale = Math.max(1.0, Math.abs(useGain));
    const rScale = Math.max(1.0, 1.0 / Math.max(0.25, Math.abs(useGain)));
    const fNoiseLim = ringRmsFB <= 0.55 * fScale ? 1.25 * fScale : 2.75 * fScale;
    const rNoiseLim = ringRmsRB <= 0.55 * rScale ? 1.25 * rScale : 2.75 * rScale;
    const fLim = ringRmsFB > 0.55 * fScale ? 1.62 * fScale : 1.18 * fScale;
    const rLim = ringRmsRB > 0.55 * rScale ? 1.62 * rScale : 1.18 * rScale;
    if ((rmsFB <= fLim && coreMaxFB <= fNoiseLim && maxFB <= 2.75 * fScale) || (rmsRB <= rLim && coreMaxRB <= rNoiseLim && maxRB <= 2.75 * rScale) || (!isSubPan && ringRmsFB >= 1.10 * fScale && cRmsFB <= ringRmsFB + 0.15 * fScale && rmsFB <= fLim && rmsRB <= rLim && maxFB <= 2.90 * fScale)) {
      return { isStatic: true, blurDiff: 0.0 };
    }
    {
      ensureBlur3();
      const ba3 = getBlur3Affine();
      const m3 = measurePatchFlow(blur3Curr, blur3Prev, ba3, cx, cy);
      const m1 = measurePatchFlow(blurCurr, blurPrev, getBlur1Affine(), cx, cy);
      const rbBlock = (rmsRB > 10.0 && rmsFB <= 2.20 * fScale) && !(m3.n >= 9 && ((m3.disp < 0.09 && rmsFB <= 1.18 * fScale) || (m3.rms <= 0.25 && rmsFB <= 1.48 * fScale && m1.n >= 9 && m1.rms <= 0.65)) && m1.n >= 9 && m1.corrFrac < 0.25);
      const smallStationaryM3 = m3.rms <= 0.62 && ((m1.n >= 9 && m1.corrFrac < 0.10) || (m3.corrFrac < 0.68 && m1.n >= 9 && m1.corrFrac < 0.20 && m1.disp < 0.12) || (m3.corrFrac < 0.55 && m1.n >= 9 && m1.corrFrac < 0.35) || (m3.disp < 0.12 && m3.corrFrac < 0.60 && m1.n >= 9 && m1.corrFrac < 0.48 && m1.disp < 0.12) || (m3.corrFrac < 0.48 && rmsFB <= Math.max(ringRmsFB * 1.35, 0.35 * fScale) && m1.n >= 9 && m1.corrFrac < 0.50));
      if (!rbBlock && m3.n >= 9 && (smallStationaryM3 || (m3.rms <= 0.78 && m3.disp < 0.075) || (m3.rms <= 1.35 && m3.disp < 0.075 && m3.corrFrac < 0.15 && m1.n >= 9 && m1.corrFrac < 0.15) || (m3.rms <= 1.0 && m3.disp < 0.12 && m3.corrFrac <= 0.40 && m1.n >= 9 && m1.corrFrac < 0.10 && m1.disp <= 0.05) || (m3.rms <= 0.85 && m3.corrFrac < 0.15 && m1.n >= 9 && m1.corrFrac < 0.10))) {
        if ((m3.corrFrac >= 0.62 && m3.disp >= 0.22 && m1.n >= 9 && m1.rms >= 1.20 && m1.disp >= 0.22 && m1.corrFrac >= 0.32 && rmsFB >= 3.5 * fScale) || (m3.rms >= 0.12 && m3.corrFrac >= 0.72 && m3.disp >= 0.042 && m1.n >= 9 && m1.corrFrac >= (isSubPan && ringRmsFB > 0.55 * fScale ? 0.50 : 0.42) && m1.disp >= 0.042 && m3.rms <= 0.78) || (m1.n >= 9 && ((m1.corrFrac >= 0.62 && m1.disp >= 0.085) || (m1.corrFrac >= 0.56 && m1.disp >= 0.092)))) {
          return { isStatic: false, blurDiff: 999.0 };
        }
        return { isStatic: true, blurDiff: 0.0 };
      }
    }
    return { isStatic: false, blurDiff: 999.0 };
  };

  const isStaticCloudShift = (cx, cy) => {
    let sd = 0.0, sox = 0.0, soy = 0.0, n = 0;
    const ds = [];
    for (let oy = -2; oy <= 2; oy++) {
      const y = cy + oy; if (y < 0 || y >= h) continue;
      const pyF = y - useDy;
      for (let ox = -2; ox <= 2; ox++) {
        const x = cx + ox; if (x < 0 || x >= w) continue;
        const pxF = x - useDx;
        if (pxF < 0.0 || pxF > w - 1.0 || pyF < 0.0 || pyF > h - 1.0) continue;
        const pred = Math.max(0.0, Math.min(255.0, useGain * sampBilLuma(prevLuma, pxF, pyF) + useOffset));
        const d = luma[y * w + x] - pred;
        ds.push({ ox, oy, d }); sd += d; sox += d * ox; soy += d * oy; n++;
      }
    }
    if (n < 9) return false;
    const md = sd / n, gx = sox / 50.0, gy = soy / 50.0;
    let ae = 0.0, pe = 0.0, pMax = 0.0;
    for (let i = 0; i < n; i++) {
      const q = ds[i];
      const ar = q.d - md; ae += ar * ar;
      const pr = Math.abs(q.d - (md + gx * q.ox + gy * q.oy));
      pe += pr * pr; if (pr > pMax) pMax = pr;
    }
    const acRms = Math.sqrt(ae / n), planeRms = Math.sqrt(pe / n);
    const fScale = Math.max(1.0, Math.abs(useGain));
    return Math.abs(md) >= 1.5 && Math.abs(md) >= 1.5 * acRms && acRms <= 1.15 * fScale && planeRms <= 0.65 * fScale && pMax <= Math.max(1.85 * fScale, 0.75 * Math.abs(md));
  };
  const evalWindowInvStats = (cx, cy, sDx, sDy, subShift) => {
    const rDx = Math.round(sDx), rDy = Math.round(sDy);
    let sc = 0.0, sp = 0.0, scc = 0.0, spp = 0.0, scp = 0.0, n = 0;
    for (let dy = -3; dy <= 3; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= h) continue;
      const py = subShift ? y - sDy : y - rDy;
      for (let dx = -3; dx <= 3; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= w) continue;
        const px = subShift ? x - sDx : x - rDx;
        const ok = subShift ? (px >= 1.0 && px < w - 2.0 && py >= 1.0 && py < h - 2.0) : (px >= 0 && px < w && py >= 0 && py < h);
        if (!ok) continue;
        const c = luma[y * w + x];
        const p = subShift ? sampleBicubicLumaCal(prevLuma, w, h, px, py) : prevLuma[py * w + px];
        sc += c; sp += p; scc += c * c; spp += p * p; scp += c * p; n++;
      }
    }
    if (n < 25) return null;
    const mc = sc / n, mp = sp / n;
    const vc = Math.max(0.0, scc / n - mc * mc);
    const vp = Math.max(0.0, spp / n - mp * mp);
    const cov = scp / n - mc * mp;
    const corr = vc > 0.0 && vp > 0.0 ? cov / Math.sqrt(vc * vp) : 0.0;
    const g = vp > 0.0 ? cov / vp : 0.0;
    const invMse = Math.max(0.0, vc + g * g * vp - 2.0 * g * cov);
    return { vc, vp, cov, corr, g, invMse };
  };
  const dtDxBuf = new Int8Array(49), dtDyBuf = new Int8Array(49);
  const dtCBuf = new Float32Array(49), dtPBuf = new Float32Array(49);
  const evalWindowInvStatsDt = (cx, cy, sDx, sDy, subShift) => {
        const rDx = Math.round(sDx), rDy = Math.round(sDy);
        let rC0=0,rCx=0,rCy=0,rP0=0,rPx=0,rPy=0,rN=0,rXX=0,rYY=0,pCnt=0;
        for (let dy = -3; dy <= 3; dy++) {
          const y = cy + dy; if (y < 0 || y >= h) continue;
          const py = subShift ? y - sDy : y - rDy;
          for (let dx = -3; dx <= 3; dx++) {
            const x = cx + dx; if (x < 0 || x >= w) continue;
            const px = subShift ? x - sDx : x - rDx;
            const ok = subShift ? (px >= 1.0 && px < w - 2.0 && py >= 1.0 && py < h - 2.0) : (px >= 0 && px < w && py >= 0 && py < h);
            if (!ok) continue;
            const c = luma[y * w + x], p = subShift ? sampleBicubicLumaCal(prevLuma, w, h, px, py) : prevLuma[py * w + px];
            dtDxBuf[pCnt] = dx; dtDyBuf[pCnt] = dy; dtCBuf[pCnt] = c; dtPBuf[pCnt] = p; pCnt++;
            if (Math.abs(dx) === 3 || Math.abs(dy) === 3) {
              rC0 += c; rCx += c * dx; rCy += c * dy;
              rP0 += p; rPx += p * dx; rPy += p * dy;
              rXX += dx * dx; rYY += dy * dy; rN++;
            }
          }
        }
        if (rN < 12 || rXX < 1 || rYY < 1 || pCnt < 25) return null;
        const mc0=rC0/rN, gxC=rCx/rXX, gyC=rCy/rYY, mp0=rP0/rN, gxP=rPx/rXX, gyP=rPy/rYY;
        let sc=0,sp=0,scc=0,spp=0,scp=0,n=0,maxPosP=0,minNegC=0;
        for (let i = 0; i < pCnt; i++) {
          const qdx = dtDxBuf[i], qdy = dtDyBuf[i];
          if (qdx < -2 || qdx > 2 || qdy < -2 || qdy > 2) continue;
          const dc = dtCBuf[i] - (mc0 + gxC * qdx + gyC * qdy);
          const dp = dtPBuf[i] - (mp0 + gxP * qdx + gyP * qdy);
          if (dp > maxPosP) maxPosP = dp;
          if (dc < minNegC) minNegC = dc;
          sc += dc; sp += dp; scc += dc*dc; spp += dp*dp; scp += dc*dp; n++;
        }
        if (n < 16 || maxPosP < 8.0 || minNegC > -5.0) return null;
        const mc=sc/n, mp=sp/n, vc=Math.max(0,scc/n-mc*mc), vp=Math.max(0,spp/n-mp*mp), cov=scp/n-mc*mp;
        const corr = vc > 0 && vp > 0 ? cov / Math.sqrt(vc * vp) : 0;
        const g = vp > 0 ? cov / vp : 0;
        const invMse = Math.max(0, vc + g * g * vp - 2 * g * cov);
        return { vc, vp, cov, corr, g, invMse, isDt: true };
      };
      const isStrongInvStats = (st) => {
    if (!st) return false;
    const { vc, vp, cov, corr } = st;
    if (cov >= 0.0 || vc <= 0.25 || vp <= 0.25) return false;
    if (vp < vc * 0.0025 || vc < vp * 0.0025) return false;
    if (corr > -0.85) return false;
    const g = Math.max(-20.0, Math.min(-0.02, cov / vp));
    const invMse = Math.max(0.0, vc + g * g * vp - 2.0 * g * cov);
    return invMse <= vc * 0.30;
  };
  const sampleAlignedAt = (sDx, sDy, subShift, ax, ay) => {
    const rDx = Math.round(sDx), rDy = Math.round(sDy);
    const px = subShift ? ax - sDx : ax - rDx;
    const py = subShift ? ay - sDy : ay - rDy;
    const ok = subShift ? (px >= 1.0 && px < w - 2.0 && py >= 1.0 && py < h - 2.0) : (px >= 0 && px < w && py >= 0 && py < h);
    if (!ok) return NaN;
    return subShift ? sampleBicubicLumaCal(prevLuma, w, h, px, py) : prevLuma[py * w + px];
  };
  const invResolutionCache = new Map();
  const resolveInversionAtShift = (sDx, sDy) => {
    const subShift = Math.abs(sDx - Math.round(sDx)) > 1e-4 || Math.abs(sDy - Math.round(sDy)) > 1e-4;
    const key = sDx.toFixed(4) + "," + sDy.toFixed(4);
    let res = invResolutionCache.get(key);
    if (res) return res;
    // Frame-wide NMS peak extraction of departing dark features and arriving dark features
    const depRaw = [];
    const arrRaw = [];
    for (let ry = 2; ry <= h - 3; ry++) {
      for (let rx = 2; rx <= w - 3; rx++) {
        const pr = sampleAlignedAt(sDx, sDy, subShift, rx, ry);
        if (!Number.isFinite(pr)) continue;
        const cr = luma[ry * w + rx];
        const diffDep = cr - pr;
        if (diffDep >= 10.0) {
          const pl1 = sampleAlignedAt(sDx, sDy, subShift, rx - 1, ry);
          const pr1 = sampleAlignedAt(sDx, sDy, subShift, rx + 1, ry);
          const pu1 = sampleAlignedAt(sDx, sDy, subShift, rx, ry - 1);
          const pd1 = sampleAlignedAt(sDx, sDy, subShift, rx, ry + 1);
          if (pr <= Math.min(pl1, pr1, pu1, pd1) + 1e-6) {
            const pl2 = sampleAlignedAt(sDx, sDy, subShift, rx - 2, ry);
            const pr2 = sampleAlignedAt(sDx, sDy, subShift, rx + 2, ry);
            const pu2 = sampleAlignedAt(sDx, sDy, subShift, rx, ry - 2);
            const pd2 = sampleAlignedAt(sDx, sDy, subShift, rx, ry + 2);
            if (Number.isFinite(pl2) && Number.isFinite(pr2) && Number.isFinite(pu2) && Number.isFinite(pd2) &&
                pr < Math.min(pl2, pr2) - 2.0 && pr < Math.min(pu2, pd2) - 2.0) {
              const rs = evalWindowInvStats(rx, ry, sDx, sDy, subShift);
              if (!rs || !(rs.vc > 1.0 && rs.vp > 1.0 && rs.corr <= -0.75)) {
                depRaw.push({ x: rx, y: ry, amp: diffDep });
              }
            }
          }
        }
        const diffArr = pr - cr;
        if (diffArr >= 5.0) {
          const row = ry * w;
          const cl = luma[row + rx - 1], crn = luma[row + rx + 1], cu = luma[row - w + rx], cd = luma[row + w + rx];
          if (cr <= Math.min(cl, crn, cu, cd) + 1e-6) {
            arrRaw.push({ x: rx, y: ry, amp: diffArr });
          }
        }
      }
    }
    const nmsPeaks = (raw, rad = 2) => {
      raw.sort((a, b) => b.amp - a.amp || a.y - b.y || a.x - b.x);
      const mask = new Uint8Array(N);
      const out = [];
      for (let i = 0; i < raw.length; i++) {
        const p = raw[i];
        if (mask[p.y * w + p.x]) continue;
        out.push(p);
        for (let oy = -rad; oy <= rad; oy++) {
          const yy = p.y + oy; if (yy < 0 || yy >= h) continue;
          for (let ox = -rad; ox <= rad; ox++) {
            const xx = p.x + ox; if (xx >= 0 && xx < w) mask[yy * w + xx] = 1;
          }
        }
      }
      return out;
    };
    const depPeaks = nmsPeaks(depRaw, 2);
    const arrPeaksAll = nmsPeaks(arrRaw, 2);
    const nonInvArrs = [];
    const invArrs = [];
    for (let i = 0; i < arrPeaksAll.length; i++) {
      const a = arrPeaksAll[i];
      let st = evalWindowInvStats(a.x, a.y, sDx, sDy, subShift);
      if (!isStrongInvStats(st)) { const sdt = evalWindowInvStatsDt(a.x, a.y, sDx, sDy, subShift); if (sdt && isStrongInvStats(sdt) && sdt.corr <= -0.90) st = sdt; }
      if (isStrongInvStats(st)) {
        a.st = st;
        invArrs.push(a);
      } else {
        nonInvArrs.push(a);
      }
    }
    const patchScore = (rx, ry, ax, ay) => {
      const dx = rx - ax, dy = ry - ay, d2 = dx * dx + dy * dy;
      if (d2 < 4) return -1; // strictly > 1px separation; no upper cap (frame-wide Math.hypot(w, h))
      let uu = 0.0, vv = 0.0, uv = 0.0, uSum = 0.0, vSum = 0.0, maxU = 0.0, maxV = 0.0;
      let up2 = 0.0, vp2 = 0.0, uvp = 0.0, cnt = 0;
      const uArr = [], vArr = [], upArr = [], vpArr = [];
      for (let oy = -2; oy <= 2; oy++) {
        const y1 = ry + oy, y2 = ay + oy;
        if (y1 < 0 || y1 >= h || y2 < 0 || y2 >= h) continue;
        for (let ox = -2; ox <= 2; ox++) {
          const x1 = rx + ox, x2 = ax + ox;
          if (x1 < 0 || x1 >= w || x2 < 0 || x2 >= w) continue;
          const p1 = sampleAlignedAt(sDx, sDy, subShift, x1, y1);
          const p2 = sampleAlignedAt(sDx, sDy, subShift, x2, y2);
          if (!Number.isFinite(p1) || !Number.isFinite(p2)) continue;
          const u = luma[y1 * w + x1] - p1;
          const v = p2 - luma[y2 * w + x2];
          uArr.push(u); vArr.push(v);
          uu += u * u; vv += v * v; uv += u * v; uSum += u; vSum += v;
          const up = u > 0.0 ? u : 0.0, vp = v > 0.0 ? v : 0.0;
          upArr.push(up); vpArr.push(vp);
          up2 += up * up; vp2 += vp * vp; uvp += up * vp;
          if (up > maxU) maxU = up;
          if (vp > maxV) maxV = vp;
          cnt++;
        }
      }
      if (cnt < 9 || maxV <= 1e-6 || maxU <= 1e-6) return -1;
      const peakRatio = maxU / maxV;
      if (peakRatio < 0.40 || peakRatio > 2.25) return -1;
      if (d2 <= 16) {
        // Strict disjoint/positive-core similarity to block 3-4px amp=10 noise leaks
        if (maxU < 12.0 || maxV < 12.0 || vp2 <= 1e-6 || up2 <= 1e-6) return -1;
        const l2PosRatio = Math.sqrt(up2 / vp2);
        const cosPos = uvp / Math.sqrt(up2 * vp2 + 1e-12);
        let muP = 0, mvP = 0;
        for (let k = 0; k < cnt; k++) { muP += upArr[k]; mvP += vpArr[k]; }
        muP /= cnt; mvP /= cnt;
        let cu = 0, cv = 0, cuv = 0;
        for (let k = 0; k < cnt; k++) {
          const du = upArr[k] - muP, dv = vpArr[k] - mvP;
          cu += du * du; cv += dv * dv; cuv += du * dv;
        }
        const nccPos = cu > 1e-6 && cv > 1e-6 ? cuv / Math.sqrt(cu * cv) : 0.0;
        if (l2PosRatio >= 0.65 && l2PosRatio <= 1.55 && cosPos >= 0.78 && nccPos >= 0.65) return nccPos;
        return -1;
      }
      if (vv <= 1e-6 || uu <= 1e-6) return -1;
      const l2Ratio = Math.sqrt(uu / vv);
      const cos = uv / Math.sqrt(uu * vv + 1e-12);
      const mu = uSum / cnt, mv = vSum / cnt;
      let cuu = 0.0, cvv = 0.0, cuv = 0.0;
      for (let k = 0; k < cnt; k++) {
        const du = uArr[k] - mu, dv = vArr[k] - mv;
        cuu += du * du; cvv += dv * dv; cuv += du * dv;
      }
      const ncc = cuu > 1e-6 && cvv > 1e-6 ? cuv / Math.sqrt(cuu * cvv) : 0.0;
      if (l2Ratio >= 0.40 && l2Ratio <= 1.65 && cos >= 0.80 && ncc >= 0.72) return { sc: ncc, l2Ratio, peakRatio };
      return -1;
    };

    const rescuedCenters = [];
    if (invArrs.length > 0 && depPeaks.length > 0) {
      const buildPatchDesc = (px, py, isDep) => {
        if (px < 2 || px >= w - 2 || py < 2 || py >= h - 2) return null;
        const raw = new Float32Array(25);
        let uu = 0.0, uSum = 0.0, maxU = 0.0, k = 0;
        for (let oy = -2; oy <= 2; oy++) {
          const y = py + oy;
          for (let ox = -2; ox <= 2; ox++) {
            const x = px + ox;
            const p = sampleAlignedAt(sDx, sDy, subShift, x, y);
            if (!Number.isFinite(p)) return null;
            const u = isDep ? (luma[y * w + x] - p) : (p - luma[y * w + x]);
            raw[k++] = u;
            uu += u * u;
            uSum += u;
            if (u > maxU) maxU = u;
          }
        }
        if (maxU <= 1e-6 || uu <= 1e-6) return { raw, nu: null, l2: 0.0, maxP: maxU };
        const mu = uSum / 25.0;
        let cuu = 0.0;
        for (let i = 0; i < 25; i++) { const d = raw[i] - mu; cuu += d * d; }
        let nu = null;
        if (cuu > 1e-6) {
          nu = new Float32Array(25);
          const invNorm = 1.0 / Math.sqrt(cuu);
          for (let i = 0; i < 25; i++) nu[i] = (raw[i] - mu) * invNorm;
        }
        return { raw, nu, l2: Math.sqrt(uu), maxP: maxU };
      };
      const fastScore = (d, a, dDesc, aDesc) => {
        const dx = d.x - a.x, dy = d.y - a.y, d2 = dx * dx + dy * dy;
        if (d2 < 4) return -1;
        if (d2 <= 16 || !dDesc || !aDesc) return patchScore(d.x, d.y, a.x, a.y);
        if (dDesc.maxP <= 1e-6 || aDesc.maxP <= 1e-6 || dDesc.l2 <= 0.0 || aDesc.l2 <= 0.0) return -1;
        const peakRatio = dDesc.maxP / aDesc.maxP;
        if (peakRatio < 0.40 || peakRatio > 2.25) return -1;
        const l2Ratio = dDesc.l2 / aDesc.l2;
        if (l2Ratio < 0.40 || l2Ratio > 1.65 || !dDesc.nu || !aDesc.nu) return -1;
        let ncc = 0.0;
        for (let i = 0; i < 25; i++) ncc += dDesc.nu[i] * aDesc.nu[i];
        if (ncc < 0.72) return -1;
        let uv = 0.0;
        for (let i = 0; i < 25; i++) uv += dDesc.raw[i] * aDesc.raw[i];
        const cos = uv / (dDesc.l2 * aDesc.l2 + 1e-12);
        return cos >= 0.80 ? { sc: ncc, l2Ratio, peakRatio } : -1;
      };
      const depDesc = depPeaks.map((d) => buildPatchDesc(d.x, d.y, true));
      const invDesc = invArrs.map((a) => buildPatchDesc(a.x, a.y, false));
      // Early-exit: if zero (invArr, depPeak) pairs are compatible regardless of Pass 1, nothing can be rescued
      let anyInvPossible = false;
      for (let ii = 0; ii < invArrs.length && !anyInvPossible; ii++) {
        const a = invArrs[ii], aD = invDesc[ii];
        for (let di = 0; di < depPeaks.length; di++) {
          const d = depPeaks[di];
          const pr = Math.max(d.amp, 1e-6) / Math.max(a.amp, 1e-6);
          if (pr < 0.40 || pr > 2.25) continue;
          const sc = fastScore(d, a, depDesc[di], aD);
          if ((typeof sc === "object" ? sc.sc : sc) > 0.0) { anyInvPossible = true; break; }
        }
      }
      if (!anyInvPossible) {
        res = { rescuedCenters };
        invResolutionCache.set(key, res);
        return res;
      }

      const depUsed = new Uint8Array(depPeaks.length);
      const arrUsed = new Uint8Array(nonInvArrs.length);
      const nonInvDesc = nonInvArrs.map((a) => buildPatchDesc(a.x, a.y, false));
      // Pass 1a: Nearby high-confidence 1-to-1 claim (dist <= 20px, sc >= 0.88) avoids O(N^2) blowup on dense flocks
      if (nonInvArrs.length > 0) {
        const cellSz = 20;
        const gW = Math.max(1, Math.ceil(w / cellSz)), gH = Math.max(1, Math.ceil(h / cellSz));
        const depGrid = new Map();
        for (let di = 0; di < depPeaks.length; di++) {
          const cx0 = Math.floor(depPeaks[di].x / cellSz), cy0 = Math.floor(depPeaks[di].y / cellSz);
          const ck = cy0 * gW + cx0;
          let b = depGrid.get(ck);
          if (!b) { b = []; depGrid.set(ck, b); }
          b.push(di);
        }
        const nearEdges = [];
        for (let ai = 0; ai < nonInvArrs.length; ai++) {
          const a = nonInvArrs[ai], aD = nonInvDesc[ai];
          const ax0 = Math.floor(a.x / cellSz), ay0 = Math.floor(a.y / cellSz);
          for (let gy = Math.max(0, ay0 - 1); gy <= Math.min(gH - 1, ay0 + 1); gy++) {
            for (let gx = Math.max(0, ax0 - 1); gx <= Math.min(gW - 1, ax0 + 1); gx++) {
              const bucket = depGrid.get(gy * gW + gx);
              if (!bucket) continue;
              for (let bi = 0; bi < bucket.length; bi++) {
                const di = bucket[bi], d = depPeaks[di];
                const dist2 = (d.x - a.x) * (d.x - a.x) + (d.y - a.y) * (d.y - a.y);
                if (dist2 > 400) continue;
                const pr = Math.max(d.amp, 1e-6) / Math.max(a.amp, 1e-6);
                if (pr < 0.40 || pr > 2.25) continue;
                const sc = fastScore(d, a, depDesc[di], aD);
                const scN = typeof sc === "object" ? sc.sc : sc;
                if (scN >= 0.88) nearEdges.push({ ai, di, sc: scN - 1e-4 * Math.sqrt(dist2) });
              }
            }
          }
        }
        nearEdges.sort((p, q) => q.sc - p.sc);
        for (let e = 0; e < nearEdges.length; e++) {
          const { ai, di } = nearEdges[e];
          if (!arrUsed[ai] && !depUsed[di]) { arrUsed[ai] = 1; depUsed[di] = 1; }
        }
      }

      // Pass 1b: Global frame-wide lossless greedy claim over remaining unmatched arrivals & departures
      let remArrList = [], remDepList = [];
      for (let ai = 0; ai < nonInvArrs.length; ai++) if (!arrUsed[ai]) remArrList.push(ai);
      for (let di = 0; di < depPeaks.length; di++) if (!depUsed[di]) remDepList.push(di);
      if (remArrList.length > 0 && remDepList.length > 0) {
        if (remArrList.length * remDepList.length <= 250000) {
          const nonInvEdges = [];
          for (let ia = 0; ia < remArrList.length; ia++) {
            const ai = remArrList[ia], a = nonInvArrs[ai], aD = nonInvDesc[ai];
            for (let id = 0; id < remDepList.length; id++) {
              const di = remDepList[id], d = depPeaks[di];
              const pr = Math.max(d.amp, 1e-6) / Math.max(a.amp, 1e-6);
              if (pr < 0.40 || pr > 2.25) continue;
              const sc = fastScore(d, a, depDesc[di], aD);
              const scN = typeof sc === "object" ? sc.sc : sc;
              if (scN > 0.0) {
                nonInvEdges.push({ ai, di, sc: scN - 1e-4 * Math.hypot(d.x - a.x, d.y - a.y) });
              }
            }
          }
          nonInvEdges.sort((p, q) => q.sc - p.sc);
          for (let e = 0; e < nonInvEdges.length; e++) {
            const { ai, di } = nonInvEdges[e];
            if (!arrUsed[ai] && !depUsed[di]) { arrUsed[ai] = 1; depUsed[di] = 1; }
          }
        } else {
          // Adaptive descending score-threshold peeling [lo, hi): preserves strict global greedy order with bounded memory
          const MAX_BAND_EDGES = 150000;
          const baseBands = [0.99, 0.97, 0.94, 0.90, 0.85, 0.78, 0.70, 0.0];
          const stack = [];
          for (let bi = baseBands.length - 1; bi >= 0; bi--) {
            const lo = baseBands[bi];
            const hi = bi === 0 ? 1.05 : baseBands[bi - 1];
            stack.push([lo, hi]);
          }
          while (stack.length > 0 && remArrList.length > 0 && remDepList.length > 0) {
            const [lo, hi] = stack.pop();
            const bandEdges = [];
            let overflow = false;
            let hitCap = false;
            for (let ia = 0; ia < remArrList.length; ia++) {
              const ai = remArrList[ia], a = nonInvArrs[ai], aD = nonInvDesc[ai];
              for (let id = 0; id < remDepList.length; id++) {
                const di = remDepList[id], d = depPeaks[di];
                const pr = Math.max(d.amp, 1e-6) / Math.max(a.amp, 1e-6);
                if (pr < 0.40 || pr > 2.25) continue;
                const sc = fastScore(d, a, depDesc[di], aD);
                const scN = typeof sc === "object" ? sc.sc : sc;
                if (scN > 0.0) {
                  const adj = scN - 1e-4 * Math.hypot(d.x - a.x, d.y - a.y);
                  if (adj >= lo && adj < hi) {
                    if (bandEdges.length >= MAX_BAND_EDGES) {
                      if (hi - lo > 1e-6) {
                        overflow = true;
                      } else {
                        hitCap = true;
                      }
                      break;
                    }
                    bandEdges.push({ ai, di, sc: adj });
                  }
                }
              }
              if (overflow || hitCap) break;
            }
            if (overflow) {
              const mid = 0.5 * (lo + hi);
              stack.push([lo, mid]);
              stack.push([mid, hi]);
              continue;
            }
            if (bandEdges.length === 0) continue;
            bandEdges.sort((p, q) => q.sc - p.sc);
            let matchedBand = 0;
            for (let e = 0; e < bandEdges.length; e++) {
              const { ai, di } = bandEdges[e];
              if (!arrUsed[ai] && !depUsed[di]) { arrUsed[ai] = 1; depUsed[di] = 1; matchedBand++; }
            }
            if (matchedBand > 0) {
              remArrList = remArrList.filter((ai) => !arrUsed[ai]);
              remDepList = remDepList.filter((di) => !depUsed[di]);
            }
            if (hitCap && matchedBand > 0 && remArrList.length > 0 && remDepList.length > 0) {
              stack.push([lo, hi]);
            }
          }
        }
      }

      // Pass 2: Remaining unmatched departures vs locally inverted arrival candidates
      const remDeps = [];
      const remDepIndices = [];
      for (let di = 0; di < depPeaks.length; di++) if (!depUsed[di]) { remDeps.push(depPeaks[di]); remDepIndices.push(di); }
      const symInvFlags = invArrs.map((a) => a.st.g >= -1.35 && a.st.g <= -0.70 && a.st.corr <= -0.94);
      const invAdj = Array.from({ length: invArrs.length }, () => []);
      const depAdj = Array.from({ length: remDeps.length }, () => []);
      for (let ii = 0; ii < invArrs.length; ii++) {
        const a = invArrs[ii], aD = invDesc[ii];
        for (let ri = 0; ri < remDeps.length; ri++) {
          const d = remDeps[ri], di = remDepIndices[ri];
          const pr = Math.max(d.amp, 1e-6) / Math.max(a.amp, 1e-6);
          if (pr < 0.40 || pr > 2.25) continue;
          const rawSc = fastScore(d, a, depDesc[di], aD);
          const scN = typeof rawSc === "object" ? rawSc.sc : rawSc;
          if (scN > 0.0) {
            const isSatInv = luma[a.y * w + a.x] <= 1 || luma[a.y * w + a.x] >= 254 || prevLuma[d.y * w + d.x] <= 1 || prevLuma[d.y * w + d.x] >= 254;
            if (symInvFlags[ii] && typeof rawSc === "object" && (rawSc.l2Ratio < 0.75 || rawSc.peakRatio < (isSatInv ? 0.50 : 0.72))) continue;
            invAdj[ii].push({ ri, sc: scN });
            depAdj[ri].push(ii);
          }
        }
      }
      // Per-component symmetric cohort guard so unlinked inversions cannot poison unrelated genuine 1-to-1 arrivals
      const visI = new Uint8Array(invArrs.length);
      const visD = new Uint8Array(remDeps.length);
      const blockedInv = new Uint8Array(invArrs.length);
      for (let ii0 = 0; ii0 < invArrs.length; ii0++) {
        if (visI[ii0] || invAdj[ii0].length === 0) continue;
        const compI = [ii0], compD = [];
        visI[ii0] = 1;
        let qi = 0, qd = 0;
        while (qi < compI.length || qd < compD.length) {
          while (qi < compI.length) {
            const u = compI[qi++];
            for (const e of invAdj[u]) if (!visD[e.ri]) { visD[e.ri] = 1; compD.push(e.ri); }
          }
          while (qd < compD.length) {
            const v = compD[qd++];
            for (const u of depAdj[v]) if (!visI[u]) { visI[u] = 1; compI.push(u); }
          }
        }
        let cSym = 0;
        for (const u of compI) if (symInvFlags[u]) cSym++;
        if (cSym > compD.length) {
          for (const u of compI) if (symInvFlags[u]) blockedInv[u] = 1;
        }
      }
      // 1-to-1 greedy weighted assignment on remaining edges
      const invEdges = [];
      for (let ii = 0; ii < invArrs.length; ii++) {
        if (blockedInv[ii]) continue;
        for (const e of invAdj[ii]) {
          const d = remDeps[e.ri], a = invArrs[ii];
          invEdges.push({ ii, ri: e.ri, sc: e.sc - 1e-4 * Math.hypot(d.x - a.x, d.y - a.y) });
        }
      }
      invEdges.sort((p, q) => q.sc - p.sc);
      const invMatched = new Uint8Array(invArrs.length);
      const remUsed = new Uint8Array(remDeps.length);
      for (let e = 0; e < invEdges.length; e++) {
        const { ii, ri } = invEdges[e];
        if (!invMatched[ii] && !remUsed[ri]) {
          invMatched[ii] = 1;
          remUsed[ri] = 1;
          rescuedCenters.push([invArrs[ii].x, invArrs[ii].y]);
        }
      }
    }
    res = { rescuedCenters };
    invResolutionCache.set(key, res);
    return res;
  };
  const hasDepartingDarkCorrespondence = (cx, cy, sDx, sDy) => {
    const { rescuedCenters } = resolveInversionAtShift(sDx, sDy);
    for (let i = 0; i < rescuedCenters.length; i++) {
      const dx = rescuedCenters[i][0] - cx, dy = rescuedCenters[i][1] - cy;
      if (dx * dx + dy * dy <= 6.25) return true;
    }
    return false;
  };

  let altMagPan = undefined;
  const getAltMagPan = () => {
    if (altMagPan !== undefined) return altMagPan;
    altMagPan = null;
    if (!hasWideSceneTexture) return null;
    const step = w >= 480 ? 3 : 2;
    let meanC = 0.0, meanP = 0.0, nAll = 0;
    for (let y = 2; y <= h - 3; y += step) {
      const row = y * w;
      for (let x = 2; x <= w - 3; x += step) {
        meanC += luma[row + x]; meanP += prevLuma[row + x]; nAll++;
      }
    }
    if (nAll === 0) return null;
    meanC /= nAll; meanP /= nAll;
    const GX = 16, GY = 12;
    const tileBuckets = Array.from({ length: GX * GY }, () => []);
    const innerW = Math.max(1, w - 4), innerH = Math.max(1, h - 4);
    for (let y = 2; y <= h - 3; y++) {
      const ty = Math.min(GY - 1, Math.floor(((y - 2) * GY) / innerH));
      const row = y * w;
      for (let x = 2; x <= w - 3; x++) {
        const dc = Math.abs(luma[row + x] - meanC);
        if (dc < 5.0) continue;
        const tx = Math.min(GX - 1, Math.floor(((x - 2) * GX) / innerW));
        const b = tileBuckets[ty * GX + tx];
        if (b.length < 4) b.push([x, y, dc]);
        else {
          let minI = 0;
          for (let q = 1; q < 4; q++) if (b[q][2] < b[minI][2]) minI = q;
          if (dc > b[minI][2]) b[minI] = [x, y, dc];
        }
      }
    }
    const devPts = [];
    for (let t = 0; t < tileBuckets.length; t++) {
      const b = tileBuckets[t];
      for (let q = 0; q < b.length; q++) devPts.push(b[q]);
    }
    if (devPts.length < 12) return null;
    ensureSepBlur();
    const coarsePts = devPts.length <= 96 ? devPts : devPts.filter((_, i) => i % Math.ceil(devPts.length / 96) === 0);
    const evalMagShiftBuf = (cPts, pBuf, meanRef, sdx, sdy) => {
      const isInt = Math.abs(sdx - Math.round(sdx)) < 1e-6 && Math.abs(sdy - Math.round(sdy)) < 1e-6;
      const idxS = Math.round(sdx), idyS = Math.round(sdy);
      let s = 0.0, cnt = 0;
      for (let i = 0; i < cPts.length; i++) {
        const [x, y, dc] = cPts[i];
        let pVal;
        if (isInt) {
          const px = x - idxS, py = y - idyS;
          if (px < 0 || px >= w || py < 0 || py >= h) continue;
          pVal = pBuf[py * w + px];
        } else {
          const sx = x - sdx, sy = y - sdy;
          if (sx < 1.0 || sx >= w - 2.0 || sy < 1.0 || sy >= h - 2.0) continue;
          pVal = sampleBicubicLumaCal(pBuf, w, h, sx, sy);
        }
        s += dc * Math.abs(pVal - meanRef);
        cnt++;
      }
      return cnt >= Math.max(8, Math.floor(cPts.length * 0.5)) ? { meanSc: s / cnt, covSc: s / cPts.length } : null;
    };
    const cStep = w >= 960 ? 3 : 2;
    const coarseCands = [];
    for (let dy = -maxPanY; dy <= maxPanY; dy += cStep) {
      for (let dx = -maxPanX; dx <= maxPanX; dx += cStep) {
        const res = evalMagShiftBuf(coarsePts, blurPrev, meanP, dx, dy);
        if (res && res.covSc > 0.0) coarseCands.push({ dx, dy, sc: res.covSc, meanSc: res.meanSc });
      }
    }
    coarseCands.sort((a, b) => b.sc - a.sc || Math.hypot(a.dx, a.dy) - Math.hypot(b.dx, b.dy));
    const seenInt = new Set();
    const topRaw = [];
    for (let i = 0; i < coarseCands.length && i < 12; i++) {
      for (let ky = -cStep; ky <= cStep; ky++) {
        for (let kx = -cStep; kx <= cStep; kx++) {
          const dx = coarseCands[i].dx + kx, dy = coarseCands[i].dy + ky;
          if (Math.abs(dx) > maxPanX || Math.abs(dy) > maxPanY) continue;
          const k = dx + "," + dy;
          if (seenInt.has(k)) continue;
          seenInt.add(k);
          const res = evalMagShiftBuf(devPts, prevLuma, meanP, dx, dy);
          if (res && res.covSc > 0.0) topRaw.push({ dx, dy, sc: res.covSc, meanSc: res.meanSc });
        }
      }
    }
    topRaw.sort((a, b) => b.sc - a.sc || Math.hypot(a.dx, a.dy) - Math.hypot(b.dx, b.dy));
    const zRes = evalMagShiftBuf(devPts, prevLuma, meanP, 0, 0);
    const zMeanScore = zRes ? zRes.meanSc : 0.0;
    let bScore = zRes ? zRes.covSc : 0.0, bMeanScore = zMeanScore, bDx = 0, bDy = 0;
    if (topRaw.length > 0 && topRaw[0].sc > bScore) {
      let chosen = topRaw[0];
      const topSc = topRaw[0].sc;
      for (let i = 1; i < topRaw.length; i++) {
        if (topRaw[i].sc < topSc * 0.96) break;
        if (Math.hypot(topRaw[i].dx, topRaw[i].dy) < Math.hypot(chosen.dx, chosen.dy)) chosen = topRaw[i];
      }
      bScore = chosen.sc; bMeanScore = chosen.meanSc; bDx = chosen.dx; bDy = chosen.dy;
    }
    if (bScore > 0.0) {
      for (const stepSz of [0.5, 0.2, 0.1, 0.05]) {
        const baseDx = bDx, baseDy = bDy;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            if (kx === 0 && ky === 0) continue;
            const tx = baseDx + kx * stepSz, ty = baseDy + ky * stepSz;
            const res = evalMagShiftBuf(devPts, prevLuma, meanP, tx, ty);
            if (res && res.covSc > bScore) { bScore = res.covSc; bMeanScore = res.meanSc; bDx = tx; bDy = ty; }
          }
        }
      }
    }
    const kept = [];
    const addKeep = (dx, dy) => {
      if (dx === 0 && dy === 0) return;
      for (const q of kept) if (Math.hypot(q.dx - dx, q.dy - dy) < 4.0) return;
      kept.push({ dx, dy });
    };
    if ((bDx !== 0 || bDy !== 0) && bMeanScore > zMeanScore * 1.05 + 1e-6) addKeep(bDx, bDy);
    if (topRaw.length > 0) {
      const topScAll = topRaw[0].sc;
      for (let i = 0; i < topRaw.length && kept.length < 8; i++) {
        const t = topRaw[i];
        if (t.sc < topScAll * 0.90) break;
        if (t.meanSc > zMeanScore * 1.01 + 1e-6 || t.sc > (zRes ? zRes.covSc * 1.005 : 0)) {
          addKeep(t.dx, t.dy);
        }
      }
    }
    const refAnchors = devPts.slice().sort((a, b) => b[2] - a[2]).slice(0, 16);
    const evalSignedPatchLoss = (sdx, sdy) => {
      let totLoss = 0.0, totW = 0.0, used = 0;
      for (let i = 0; i < refAnchors.length; i++) {
        const ax = refAnchors[i][0], ay = refAnchors[i][1];
        if (ax < 2 || ax >= w - 2 || ay < 2 || ay >= h - 2) continue;
        let sc = 0.0, sp = 0.0, scc = 0.0, spp = 0.0, scp = 0.0, n = 0;
        for (let oy = -2; oy <= 2; oy++) {
          const y = ay + oy, sy = y - sdy;
          if (sy < 1.0 || sy >= h - 2.0) continue;
          for (let ox = -2; ox <= 2; ox++) {
            const x = ax + ox, sx = x - sdx;
            if (sx < 1.0 || sx >= w - 2.0) continue;
            const c = luma[y * w + x], p = sampleBicubicLumaCal(prevLuma, w, h, sx, sy);
            sc += c; sp += p; scc += c * c; spp += p * p; scp += c * p; n++;
          }
        }
        if (n < 16) continue;
        const mc = sc / n, mp = sp / n;
        const vc = Math.max(0.0, scc / n - mc * mc), vp = Math.max(0.0, spp / n - mp * mp), cov = scp / n - mc * mp;
        if (vc <= 0.25 || vp <= 0.25) continue;
        const r2 = Math.min(1.0, (cov * cov) / (vc * vp));
        totLoss += vc * (1.0 - r2);
        totW += vc;
        used++;
      }
      return used >= Math.max(1, Math.floor(refAnchors.length * 0.25)) && totW > 1e-6 ? totLoss / totW : Infinity;
    };
    const refinedRaw = [];
    for (let ki = 0; ki < kept.length; ki++) {
      let rdx = kept[ki].dx, rdy = kept[ki].dy;
      let rLoss = evalSignedPatchLoss(rdx, rdy);
      if (Number.isFinite(rLoss)) {
        for (const stepSz of [0.5, 0.2, 0.1, 0.05]) {
          const b0x = rdx, b0y = rdy;
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              if (kx === 0 && ky === 0) continue;
              const tx = b0x + kx * stepSz, ty = b0y + ky * stepSz;
              const ls = evalSignedPatchLoss(tx, ty);
              if (ls < rLoss) { rLoss = ls; rdx = tx; rdy = ty; }
            }
          }
        }
      }
      refinedRaw.push({ dx: rdx, dy: rdy, loss: rLoss });
    }
    refinedRaw.sort((a, b) => a.loss - b.loss);
    const refinedKept = [];
    for (let i = 0; i < refinedRaw.length; i++) {
      const r = refinedRaw[i];
      let dup = false;
      for (const q of refinedKept) if (Math.hypot(q.dx - r.dx, q.dy - r.dy) < 2.0) { dup = true; break; }
      if (!dup) refinedKept.push({ dx: r.dx, dy: r.dy });
    }
    altMagPan = refinedKept;
    return altMagPan;
  };
  const isLocalInvAt = (cx, cy, sDx, sDy) => {
    const subShift = Math.abs(sDx - Math.round(sDx)) > 1e-4 || Math.abs(sDy - Math.round(sDy)) > 1e-4;
    let st = evalWindowInvStats(cx, cy, sDx, sDy, subShift);
    if (!isStrongInvStats(st)) { const sdt = evalWindowInvStatsDt(cx, cy, sDx, sDy, subShift); if (!sdt || !isStrongInvStats(sdt) || sdt.corr > -0.90) return false; st = sdt; }
    return !hasDepartingDarkCorrespondence(cx, cy, sDx, sDy);
  };
  const isLocalInvZero = (cx, cy) => {
    if (isLocalInvAt(cx, cy, useDx, useDy)) return true;
    if (hasCredibleAltPan && isLocalInvAt(cx, cy, best.dx, best.dy)) return true;
    for (let si = 0; si < stage1AltShifts.length; si++) { const s1 = stage1AltShifts[si]; if ((Math.abs(s1.dx - useDx) > 1e-4 || Math.abs(s1.dy - useDy) > 1e-4) && isLocalInvAt(cx, cy, s1.dx, s1.dy)) return true; }
    const aps = getAltMagPan();
    if (aps && aps.length > 0) {
      for (let ai = 0; ai < aps.length; ai++) {
        const ap = aps[ai];
        if ((Math.abs(ap.dx - useDx) > 1e-4 || Math.abs(ap.dy - useDy) > 1e-4) && isLocalInvAt(cx, cy, ap.dx, ap.dy)) return true;
      }
    }
    return false;
  };

  const darkFg = new Uint8Array(N);
  const bgMap = new Float32Array(N);
  const meanOuterMap = new Float32Array(N);
  const meanNbMap = new Float32Array(N);
  const cands = [];
  for (let y = 3; y < h - 3; y++) {
    const prevPy = isSubPan ? y - useDy : y - roundDy;
    for (let x = 3; x < w - 3; x++) {
      const idx = y * w + x;
      const c = luma[idx];
      let minNb = 255.0;
      let sumNb = 0.0;
      let maxOuter = 0.0, sumOuter = 0.0;
      for (let dy = -3; dy <= 3; dy++) {
        const rowOff = (y + dy) * w;
        for (let dx = -3; dx <= 3; dx++) {
          const v = luma[rowOff + x + dx];
          if (Math.abs(dx) === 3 || Math.abs(dy) === 3) {
            if (v > maxOuter) maxOuter = v;
            sumOuter += v;
          } else if (dx !== 0 || dy !== 0) {
            if (v < minNb) minNb = v;
            sumNb += v;
          }
        }
      }
      const meanNb = sumNb / 24.0;
      const meanOuter = sumOuter / 24.0;
      const bgRef = Math.max(meanNb, 0.5 * (maxOuter + meanNb));
      const contrast = meanNb - c;
      bgMap[idx] = bgRef;
      meanOuterMap[idx] = meanOuter;
      meanNbMap[idx] = meanNb;
      const prevPx = isSubPan ? x - useDx : x - roundDx;
      if (!hasPrevSupport(prevPx, prevPy)) continue;
      const prevVal = isSubPan
        ? sampleBicubicLumaCal(prevLuma, w, h, prevPx, prevPy)
        : prevLuma[prevPy * w + prevPx];
      const predC = Math.max(0.0, Math.min(255.0, useGain * prevVal + useOffset));
      const tDiffRaw = Math.abs(c - predC);
      let tDiff = tDiffRaw;
      let tNb = 0.0;
      const nbIdx = idx + 1;
      const prevNbPx = isSubPan ? x + 1 - useDx : x + 1 - roundDx;
      if (hasPrevSupport(prevNbPx, prevPy)) {
        const prevNbVal = isSubPan
          ? sampleBicubicLumaCal(prevLuma, w, h, prevNbPx, prevPy)
          : prevLuma[prevPy * w + prevNbPx];
        const predNb = Math.max(0.0, Math.min(255.0, useGain * prevNbVal + useOffset));
        const tNbRaw = Math.abs(luma[nbIdx] - predNb);
        if (isSubPan) {
          const gxN = x + 2 < w ? 0.5 * (luma[nbIdx + 1] - luma[nbIdx - 1]) : luma[nbIdx] - luma[idx];
          const gyN = 0.5 * (luma[nbIdx + w] - luma[nbIdx - w]);
          tNb = Math.max(0.0, tNbRaw - 0.14 * Math.hypot(gxN, gyN));
        } else {
          tNb = tNbRaw;
        }
      }
      if (isSubPan) {
        const gx = 0.5 * (luma[idx + 1] - luma[idx - 1]);
        const gy = 0.5 * (luma[idx + w] - luma[idx - w]);
        tDiff = Math.max(0.0, tDiffRaw - 0.14 * Math.hypot(gx, gy));
      }
      const pureBrightening = (() => {
        if (c - predC < 8.0) return false;
        let maxDark5 = 0.0;
        for (let dy5 = -2; dy5 <= 2; dy5++) {
          const py5 = (isSubPan ? y - useDy : y - roundDy) + dy5;
          const cy5 = y + dy5;
          if (cy5 < 0 || cy5 >= h) continue;
          for (let dx5 = -2; dx5 <= 2; dx5++) {
            const px5 = (isSubPan ? x - useDx : x - roundDx) + dx5;
            const cx5 = x + dx5;
            if (cx5 < 0 || cx5 >= w || !hasPrevSupport(px5, py5)) continue;
            const pv5 = isSubPan ? sampleBicubicLumaCal(prevLuma, w, h, px5, py5) : prevLuma[py5 * w + px5];
            const pr5 = Math.max(0.0, Math.min(255.0, useGain * pv5 + useOffset));
            const dk5 = pr5 - luma[cy5 * w + cx5];
            if (dk5 > maxDark5) maxDark5 = dk5;
          }
        }
        return maxDark5 <= 2.5 && maxDark5 < 0.15 * (c - predC);
      })();
      const needDarkCheck = bgRef - c >= 6.0 || (c <= minNb && contrast >= 6.0);
      const statMaxLim = 3.5 * Math.max(1.0, Math.abs(useGain));
      if (pureBrightening || (needDarkCheck && (tDiff >= 1.8 || (contrast >= 14.0 && tNb >= 1.5)) && isStaticCloudShift(x, y))) { tDiff = 0.0; tNb = 0.0; }
      if (needDarkCheck && (isSubPan ? (tDiff >= 1.8 || (contrast >= 14.0 && tNb >= 1.5)) : ((tDiff >= 1.8 && tDiff <= statMaxLim) || (contrast >= 14.0 && tNb >= 1.5 && tNb <= statMaxLim)))) {
        const ls = checkLocalStaticSubpixel(x, y);
        if (ls.isStatic) {
          tDiff = 0.0;
          tNb = 0.0;
        }
      }
      const maybeFg = bgRef - c >= 6.0 && tDiff >= 1.8;
      const maybeSeed = c <= minNb && contrast >= 6.0 && (tDiff >= 1.8 || (contrast >= 14.0 && tNb >= 1.5));
      if ((maybeFg || maybeSeed) && !isLocalInvZero(x, y)) {
        if (maybeFg) darkFg[idx] = 1;
        if (maybeSeed) cands.push({ x, y, idx, contrast });
      }
    }
  }

  // Group seeds via unbounded 8-connected dark silhouette components
  const compId = new Int32Array(N).fill(-1);
  const qIdx = new Int32Array(N);
  let numComps = 0;
  cands.sort((a, b) => b.contrast - a.contrast);

  const MAX_BRIDGE_COMP = 512;
  for (let i = 0; i < cands.length; i++) {
    const seed = cands[i];
    if (compId[seed.idx] >= 0) continue;
    const cid = numComps++;
    let head = 0;
    let tail = 0;
    let overflowed = false;
    qIdx[tail++] = seed.idx;
    compId[seed.idx] = cid;

    while (head < tail) {
      const cIdx = qIdx[head++];
      const cx = cIdx % w;
      const cy = (cIdx - cx) / w;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cy + dy;
        if (ny < 3 || ny >= h - 3) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = cx + dx;
          if (nx < 3 || nx >= w - 3) continue;
          const nIdx = ny * w + nx;
          if (compId[nIdx] >= 0) continue;
          const canBridge =
            meanOuterMap[nIdx] - luma[nIdx] >= Math.max(10.0, 0.10 * seed.contrast) &&
            meanOuterMap[cIdx] - luma[cIdx] >= Math.max(10.0, 0.10 * seed.contrast) &&
            meanNbMap[nIdx] >= luma[nIdx] &&
            Math.abs(luma[nIdx] - luma[cIdx]) <= 25.0;
          if (darkFg[nIdx] || canBridge) {
            if (!darkFg[nIdx] && tail >= MAX_BRIDGE_COMP) {
              overflowed = true;
              break;
            }
            compId[nIdx] = cid;
            qIdx[tail++] = nIdx;
          }
        }
        if (overflowed) break;
      }
      if (overflowed) break;
    }
    if (overflowed) {
      // Static wave trough / unbounded background leak: roll back non-darkFg bridge and keep darkFg-only
      for (let k = 0; k < tail; k++) compId[qIdx[k]] = -1;
      head = 0;
      tail = 0;
      qIdx[tail++] = seed.idx;
      compId[seed.idx] = cid;
      while (head < tail) {
        const cIdx = qIdx[head++];
        const cx = cIdx % w;
        const cy = (cIdx - cx) / w;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 3 || ny >= h - 3) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            if (nx < 3 || nx >= w - 3) continue;
            const nIdx = ny * w + nx;
            if (compId[nIdx] < 0 && darkFg[nIdx]) {
              compId[nIdx] = cid;
              qIdx[tail++] = nIdx;
            }
          }
        }
      }
    }
  }

  if (numComps === 0) return 0.0;
  // True 1-to-1 per-component bipartite matching (distance-sorted)
  const radPx = Math.max(3.0, 4.0 * (w / 640.0));
  const radSq = radPx * radPx;
  const rSearch = Math.ceil(radPx);
  const trackPairs = [];
  for (let ti = 0; ti < observedTracks.length; ti++) {
    const tx = observedTracks[ti].x;
    const ty = observedTracks[ti].y;
    const x0 = Math.max(3, Math.floor(tx - rSearch));
    const x1 = Math.min(w - 4, Math.ceil(tx + rSearch));
    const y0 = Math.max(3, Math.floor(ty - rSearch));
    const y1 = Math.min(h - 4, Math.ceil(ty + rSearch));
    let bestD2ByCid = null;
    for (let py = y0; py <= y1; py++) {
      const dy = py - ty;
      const dy2 = dy * dy;
      if (dy2 > radSq) continue;
      for (let px = x0; px <= x1; px++) {
        const cid = compId[py * w + px];
        if (cid < 0) continue;
        const dx = px - tx;
        const d2 = dx * dx + dy2;
        if (d2 <= radSq) {
          if (!bestD2ByCid) bestD2ByCid = new Map();
          const prevBest = bestD2ByCid.get(cid);
          if (prevBest === undefined || d2 < prevBest) bestD2ByCid.set(cid, d2);
        }
      }
    }
    if (bestD2ByCid) {
      for (const [cid, d2] of bestD2ByCid.entries()) {
        trackPairs.push({ ti, cid, d2 });
      }
    }
  }
  trackPairs.sort((a, b) => a.d2 - b.d2);
  const usedT = new Uint8Array(observedTracks.length);
  const usedC = new Uint8Array(numComps);
  let matched = 0;
  for (let p = 0; p < trackPairs.length; p++) {
    const { ti, cid } = trackPairs[p];
    if (!usedT[ti] && !usedC[cid]) {
      usedT[ti] = 1;
      usedC[cid] = 1;
      matched++;
    }
  }
  return matched / observedTracks.length;
}

function createClipAccumulator(clip, w, h, expectedArenaBytes = 0) {
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

  return {
    clip,
    w,
    h,
    s: w / 320.0,
    expectedArenaBytes,
    arenaMismatchErrors: 0,
    gtByFrame,
    trackAges: new Map(),
    trackLife: new Map(),
    dispMot: createMotState(),
    obsBrgMot: createMotState(),
    strictObsMot: createMotState(),
    duplicateTidErrors: 0,
    invalidEncodingErrors: 0,
    coastedOnlySlots: 0,
    bridgedSlots: 0,
    strictObsSlots: 0,
    observedOrBridgedSlots: 0,
    teleportJumps: 0,
    sumTickMs: 0.0,
    warmSumTickMs: 0.0,
    tickSamples: [],
    warmTickSamples: [],
    maxClutterDb: 0.0,
    lastClutterDb: 0.0,
    lastEvalRatio: 0.0,
    framesWithTargets: 0,
    sumActiveTargets: 0,
    maxActiveTargets: 0,
    bridgeFrames: 0,
    apexFrames: 0,
    phaseValidFrames: 0,
    freqValidFrames: 0,
    validFreqSamples: [],
    percTriggers: 0,
    pivotFrames: 0,
    maxDeflectDeg: 0.0,
    lastSceneGrav: 0.0,
    maxGravConf: 0.0,
    maxGravArcs: 0,
    lastScatterDeg: 0.0,
    maxEigRes: 0.0,
    nanDetected: false,
    tailNonZeroCount: 0,
    inactiveNonZeroCount: 0,
    regimeFrames: new Int32Array(9),
    jumpFrames: new Int32Array(5),
    invalidRegimeCount: 0,
    invalidJumpCount: 0,
    posEllipticFrames: 0,
    posQualQFrames: 0,
    validRestitutionFrames: 0,
    invalidRestitutionFrames: 0,
    maxKuramotoR: 0.0,
    maxKuramotoCount: 0,
    maxLyapunov: 0.0,
    maxLastPairFrame: 0,
    maxTaubinDelta: 0.0,
    pairEventCount: 0,
    invalidPairEvents: 0,
    minPairMomResid: Infinity,
    maxPairMassRatio: 0.0,
    lastSeenPairFrame: 0,
    lastSeenPairTidA: 0,
    lastSeenPairTidB: 0,
    swarmProbeOrActiveFrames: 0,
    swarmConfirmedPeak: 0,
    swarmTailNonZeroCount: 0,
    swarmCountParityErrors: 0,
    microDupTidErrors: 0,
    microInvalidEncodingErrors: 0,
    swarmSentinelErrors: 0,
    inspectorErrors: 0,
    inspectorMsgs: [],
    swarmPostWarmupFrames: 0,
    swarmPostWarmupActiveFrames: 0,
    swarmActiveObsCounts: [],
    microStreakMap: new Map(),
    starlingPrevRgba: clip.expectDenseSwarm ? new Uint8Array(w * h * 4) : null,
    starlingOraclePrecisions: [],
  };
}

function verifyPopulatedInspectorCal(exp, w, h, confirmedMicro, swarmTelem) {
  const errors = [];
  if (!confirmedMicro || confirmedMicro.length === 0) return errors;
  const maxR = 24.0 * (w / 640.0);
  const maxRSq = maxR * maxR;
  const target =
    confirmedMicro.find((t) => t.x >= 20 && t.x <= w - 20 && t.y >= 20 && t.y <= h - 20) ||
    confirmedMicro[0];
  const qx = Math.max(0.0, Math.min(w - 0.05, target.x + 0.25));
  const qy = Math.max(0.0, Math.min(h - 0.05, target.y - 0.25));
  const gotHit = exp.chronos_inspect_field_point(qx, qy);
  let nearest = null;
  let bestD2 = maxRSq;
  for (const t of confirmedMicro) {
    const d2 = (t.x - qx) ** 2 + (t.y - qy) ** 2;
    if (d2 < bestD2) {
      bestD2 = d2;
      nearest = t;
    }
  }
  if (!nearest || gotHit !== nearest.tid) {
    errors.push(`calInspectorHitMismatch(got=${gotHit},exp=${nearest?.tid ?? 'NONE'})`);
    return errors;
  }
  if (swarmTelem[11] !== nearest.tid || swarmTelem[12] !== nearest.bodyId) {
    errors.push(`calInspectorTelemMismatch([11..12]=[${swarmTelem[11]},${swarmTelem[12]}],exp=[${nearest.tid},${nearest.bodyId}])`);
  }
  const preOob11 = swarmTelem[11];
  const preOob12 = swarmTelem[12];
  const hasFocusApi =
    typeof exp.chronos_get_focus_target === 'function' && typeof exp.chronos_set_focus_target === 'function';
  const focusAfterHit = hasFocusApi ? exp.chronos_get_focus_target() : null;
  if (hasFocusApi && nearest.tid > 0) {
    if (!Number.isInteger(focusAfterHit) || focusAfterHit <= 0) {
      errors.push(`calInspectorNoHeroPromotion(${nearest.tid})=>focus=${focusAfterHit}`);
    } else {
      exp.chronos_set_focus_target(nearest.tid);
      const pinnedHero = exp.chronos_get_focus_target();
      if (pinnedHero !== focusAfterHit) {
        errors.push(`calInspectorHeroPin(${nearest.tid})=>${pinnedHero}!=${focusAfterHit}`);
      }
    }
  }
  for (const [ox, oy] of [[-12.0, -12.0], [-0.01, h * 0.5], [w, 10.0], [10.0, h], [NaN, 10.0], [10.0, Infinity]]) {
    const ret = exp.chronos_inspect_field_point(ox, oy);
    const focusNow = hasFocusApi ? exp.chronos_get_focus_target() : null;
    if (ret !== 0 || swarmTelem[11] !== preOob11 || swarmTelem[12] !== preOob12 || (focusAfterHit !== null && focusNow !== focusAfterHit)) {
      errors.push(`calInspectorOobMutate(${ox},${oy})`);
      break;
    }
  }
  const cornerRet = exp.chronos_inspect_field_point(0.0, 0.0);
  let cornerNearest = null;
  let cornerBest = maxRSq;
  for (const t of confirmedMicro) {
    const d2 = t.x * t.x + t.y * t.y;
    if (d2 < cornerBest) { cornerBest = d2; cornerNearest = t; }
  }
  if (cornerRet !== (cornerNearest ? cornerNearest.tid : 0)) {
    errors.push(`calInspectorOriginMismatch(${cornerRet})`);
  }
  exp.chronos_inspect_field_point(qx, qy);
  for (const [cx, cy] of [[w - 0.5, 0.5], [0.5, h - 0.5], [4.0, 4.0], [w * 0.5, h - 1.0]]) {
    if (confirmedMicro.reduce((mn, t) => Math.min(mn, Math.hypot(t.x - cx, t.y - cy)), Infinity) >= maxR + 2.0) {
      const missRet = exp.chronos_inspect_field_point(cx, cy);
      if (missRet !== 0 || swarmTelem[11] !== 0.0 || swarmTelem[12] !== 0.0) {
        errors.push(`calInspectorEmptyMiss(${cx},${cy})=>${missRet}`);
      }
      break;
    }
  }
  if (hasFocusApi) exp.chronos_set_focus_target(0);
  return errors;
}

function recordSessionFrame(acc, sess, tickDt, resNorm, f, checkAutoInactivity, frameRgba = null) {
  const { telem, swarmTelem, microTracks, continuumField, rigidBodies, modalField } = sess;
  const clip = acc.clip;
  const teleportStepMax = 42.0 * acc.s * Math.max(1.0, (60.0 / clip.fps) * 0.78);

  acc.sumTickMs += tickDt;
  if (f > 0) {
    acc.warmSumTickMs += tickDt;
    acc.warmTickSamples.push(tickDt);
  }
  acc.tickSamples.push(tickDt);

  if (resNorm > acc.maxEigRes) acc.maxEigRes = resNorm;
  for (let i = 0; i < TELEMETRY_LEN; i++) {
    if (!Number.isFinite(telem[i])) acc.nanDetected = true;
  }
  if (acc.expectedArenaBytes > 0 && Math.round(telem[20]) !== acc.expectedArenaBytes) {
    acc.arenaMismatchErrors++;
  }
  for (let i = TELEM_RESERVED_TAIL_START; i < TELEMETRY_LEN; i++) {
    if (telem[i] !== 0.0) acc.tailNonZeroCount++;
  }

  if (swarmTelem) {
    for (let i = 0; i < 64; i++) {
      if (!Number.isFinite(swarmTelem[i])) acc.nanDetected = true;
    }
    for (let i = SWARM_RESERVED_TAIL_START; i < 64; i++) {
      if (swarmTelem[i] !== 0.0) acc.swarmTailNonZeroCount++;
    }
    acc.swarmConfirmedPeak = Math.max(acc.swarmConfirmedPeak, Math.round(swarmTelem[5] || 0));
    if (checkAutoInactivity) {
      let microActive = 0;
      if (microTracks) {
        for (let i = 0; i < 1536; i++) {
          if (microTracks[i * 8 + 5] !== 0.0) microActive++;
        }
      }
      if (swarmTelem[1] !== 0.0 || swarmTelem[2] !== 0.0 || swarmTelem[3] !== 0.0 || microActive > 0) {
        acc.swarmProbeOrActiveFrames++;
      }
    }
    // Validate #96/#115 continuum, modal, micro FTLE & swarmTelem[39]/[41] contracts on EVERY Wave-2 frame (active and inactive)
    const swarmActive = swarmTelem[1] > 0.5;
    const activeMicroCnt = Math.round(swarmTelem[3]);
    const ftleMatureTracks = swarmTelem[39];
    if (
      !Number.isFinite(ftleMatureTracks) ||
      !Number.isInteger(ftleMatureTracks) ||
      ftleMatureTracks < 0 ||
      ftleMatureTracks > activeMicroCnt ||
      (!swarmActive && ftleMatureTracks !== 0)
    ) {
      acc.swarmSentinelErrors++;
    }
    const r2 = swarmTelem[41];
    if (!Number.isFinite(r2) || r2 < -1e-4 || r2 > 1.0 + 1e-4) {
      acc.swarmSentinelErrors++;
    }
    if (continuumField) {
      for (let c = 0; c < 880; c++) {
        const fb = c * 6;
        for (let k = 0; k < 6; k++) {
          if (!Number.isFinite(continuumField[fb + k])) acc.nanDetected = true;
        }
        const cw = continuumField[fb + 5];
        const ftle = continuumField[fb + 4];
        if (!Number.isFinite(cw) || cw < 0.0) {
          acc.swarmSentinelErrors++;
        } else if (cw === 0.0 && (continuumField[fb] !== 0.0 || continuumField[fb + 1] !== 0.0 || continuumField[fb + 2] !== 0.0 || continuumField[fb + 3] !== 0.0)) {
          acc.swarmSentinelErrors++;
        }
        if (!Number.isFinite(ftle)) {
          acc.swarmSentinelErrors++;
        } else if (swarmActive) {
          if (ftle <= -900.0 && ftle !== -999.0) acc.swarmSentinelErrors++;
        } else if (ftle !== 0.0) {
          acc.swarmSentinelErrors++;
        }
      }
    }
    if (modalField) {
      for (let c = 0; c < 880; c++) {
        const phi = modalField[c];
        if (!Number.isFinite(phi)) {
          acc.nanDetected = true;
        } else if (swarmActive) {
          if (phi !== -9.0 && (phi < -3.15 || phi > 3.15)) acc.swarmSentinelErrors++;
        } else if (phi !== 0.0) {
          acc.swarmSentinelErrors++;
        }
      }
    }
    if (rigidBodies) {
      for (let i = 0; i < rigidBodies.length; i++) if (!Number.isFinite(rigidBodies[i])) acc.nanDetected = true;
    }
    if (microTracks) {
      const microObs = [];
      const seenMicroTids = new Set();
      for (let i = 0; i < 1536; i++) {
        const mb = i * 8;
        for (let k = 0; k < 8; k++) {
          if (!Number.isFinite(microTracks[mb + k])) acc.nanDetected = true;
        }
        const st = microTracks[mb + 5];
        if (!Number.isInteger(st) || st < 0 || st > 2) {
          acc.microInvalidEncodingErrors++;
          continue;
        }
        if (st === 0) continue;
        if (!swarmActive) acc.swarmSentinelErrors++;
        const rawTid = microTracks[mb + 0];
        const mx = microTracks[mb + 1];
        const my = microTracks[mb + 2];
        const bId = microTracks[mb + 6];
        const ft = microTracks[mb + 7];
        if (!Number.isFinite(ft) || (ft <= -900.0 && ft !== -999.0)) {
          acc.microInvalidEncodingErrors++;
        }
        if (
          !Number.isInteger(rawTid) ||
          rawTid <= 0 ||
          !Number.isInteger(bId) ||
          bId < 0 ||
          bId > 32 ||
          !Number.isFinite(mx) ||
          !Number.isFinite(my)
        ) {
          acc.microInvalidEncodingErrors++;
          continue;
        }
        if (seenMicroTids.has(rawTid)) {
          acc.microDupTidErrors++;
          continue;
        }
        seenMicroTids.add(rawTid);
        if (st === 1) microObs.push({ tid: rawTid, x: mx, y: my, bodyId: bId });
      }
      if (!swarmActive || f === 0) {
        acc.microStreakMap.clear();
      }
      if (swarmActive) {
        const maxMicroStep = 48.0 * (acc.w / 640.0);
        const nextStreakMap = new Map();
        let indepMature9 = 0;
        for (let i = 0; i < microObs.length; i++) {
          const t = microObs[i];
          const prev = acc.microStreakMap.get(t.tid);
          const streak =
            prev && Math.hypot(t.x - prev.x, t.y - prev.y) <= maxMicroStep ? prev.streak + 1 : 1;
          nextStreakMap.set(t.tid, { x: t.x, y: t.y, streak });
          if (streak >= 9) indepMature9++;
        }
        acc.microStreakMap = nextStreakMap;
        if (Number.isInteger(ftleMatureTracks) && ftleMatureTracks >= 0 && ftleMatureTracks > indepMature9) {
          acc.swarmSentinelErrors++;
        }
      }
      if (clip.expectDenseSwarm) {
        if (microObs.length !== Math.round(swarmTelem[4])) acc.swarmCountParityErrors++;
        if (f >= 20) {
          acc.swarmPostWarmupFrames++;
          if (swarmTelem[1] > 0.5) {
            acc.swarmPostWarmupActiveFrames++;
            acc.swarmActiveObsCounts.push(microObs.length);
          }
        }
        if (f === 60 && microObs.length > 0 && typeof sess.exp.chronos_inspect_field_point === 'function') {
          const iErrs = verifyPopulatedInspectorCal(sess.exp, acc.w, acc.h, microObs, swarmTelem);
          if (iErrs.length > 0) {
            acc.inspectorErrors += iErrs.length;
            acc.inspectorMsgs.push(...iErrs);
          }
        }
        if (frameRgba && acc.starlingPrevRgba) {
          if (f > 0 && (f === 30 || f === 90 || f === 150 || f === 210 || f === 270 || f === 330)) {
            acc.starlingOraclePrecisions.push(
              measureStarlingDarkExtremaPrecisionCal(frameRgba, acc.starlingPrevRgba, acc.w, acc.h, microObs)
            );
          }
          acc.starlingPrevRgba.set(frameRgba);
        }
      }
    }
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
  if (kurR > acc.maxKuramotoR) acc.maxKuramotoR = kurR;
  if (kurCount > acc.maxKuramotoCount) acc.maxKuramotoCount = kurCount;
  if (lyapDiv > acc.maxLyapunov) acc.maxLyapunov = lyapDiv;
  if (pairFrame > acc.maxLastPairFrame) acc.maxLastPairFrame = pairFrame;
  if (taubinDelta > acc.maxTaubinDelta) acc.maxTaubinDelta = taubinDelta;
  if (
    pairFrame > 0 &&
    (pairFrame !== acc.lastSeenPairFrame || pairTidA !== acc.lastSeenPairTidA || pairTidB !== acc.lastSeenPairTidB)
  ) {
    acc.lastSeenPairFrame = pairFrame;
    acc.lastSeenPairTidA = pairTidA;
    acc.lastSeenPairTidB = pairTidB;
    acc.pairEventCount++;
    const validPair =
      pairMassRatio > 0.0 &&
      pairMassRatio <= 20.0 &&
      pairMomResid >= 0.0 &&
      pairMomResid <= 0.45 &&
      pairEnergyRet > 0.0 &&
      pairEnergyRet <= 1.10;
    if (!validPair) acc.invalidPairEvents++;
    if (pairMomResid < acc.minPairMomResid) acc.minPairMomResid = pairMomResid;
    if (pairMassRatio > acc.maxPairMassRatio) acc.maxPairMassRatio = pairMassRatio;
  }

  const active = Math.round(telem[3]);
  if (active > 0) acc.framesWithTargets++;
  acc.sumActiveTargets += active;
  if (active > acc.maxActiveTargets) acc.maxActiveTargets = active;

  if (telem[22] > 0.5) {
    acc.lastClutterDb = telem[4];
    acc.lastEvalRatio = telem[23];
    if (acc.lastClutterDb > acc.maxClutterDb) acc.maxClutterDb = acc.lastClutterDb;
  }
  if (telem[26] > 0.5) acc.phaseValidFrames++;
  if (telem[1260] > 0.5 && telem[13] > 0.0) {
    acc.freqValidFrames++;
    acc.validFreqSamples.push(telem[13]);
  }
  if (Math.round(telem[15]) !== 0) acc.percTriggers++;
  if (telem[1248] > 0.0) acc.lastSceneGrav = telem[1248];
  if (telem[1249] > acc.maxGravConf) acc.maxGravConf = telem[1249];
  if (Math.round(telem[1259]) > acc.maxGravArcs) acc.maxGravArcs = Math.round(telem[1259]);
  if (telem[1250] > 0.0) acc.lastScatterDeg = telem[1250];

  const activeTracks = [];
  const seenFrameTids = new Set();
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    const kbase = 1120 + m * 8;
    const pbase = TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE;
    const rawSlotActive = telem[base];
    if (rawSlotActive !== 0.0 && rawSlotActive !== 1.0) acc.invalidEncodingErrors++;
    if (rawSlotActive > 0.5) {
      const rawTid = telem[base + 1];
      if (!Number.isInteger(rawTid) || rawTid <= 0) acc.invalidEncodingErrors++;
      const tid = Math.round(rawTid);
      if (seenFrameTids.has(tid)) acc.duplicateTidErrors++;
      seenFrameTids.add(tid);

      const fx = telem[base + 2];
      const fy = telem[base + 3];
      const ix = telem[base + 8];
      const iy = telem[base + 9];
      const dispVis = telem[base + 10] > 0.5;
      const bridgedAtDisp = telem[base + 11] > 0.5;
      if (dispVis) acc.strictObsSlots++;
      else if (bridgedAtDisp) acc.bridgedSlots++;
      if (dispVis || bridgedAtDisp) acc.observedOrBridgedSlots++;
      else acc.coastedOnlySlots++;

      const latestVis = telem[base + 44 + 11] > 0.5;
      acc.trackAges.set(tid, (acc.trackAges.get(tid) || 0) + 1);
      if (bridgedAtDisp) acc.bridgeFrames++;
      if (telem[base + 17] > 0.35) acc.apexFrames++;
      if (telem[kbase + 6] > -9e5) acc.pivotFrames++;
      if (Math.abs(telem[kbase + 3]) > acc.maxDeflectDeg) acc.maxDeflectDeg = Math.abs(telem[kbase + 3]);

      const physRegime = Math.round(telem[pbase + 4]);
      if (physRegime >= 0 && physRegime <= 8) acc.regimeFrames[physRegime]++;
      else acc.invalidRegimeCount++;
      const jumpCode = Math.round(telem[pbase + 15]);
      if (jumpCode >= 1 && jumpCode <= 4) acc.jumpFrames[jumpCode]++;
      else if (jumpCode !== 0) acc.invalidJumpCount++;
      if (telem[pbase + 9] > 0.0) acc.posEllipticFrames++;
      if (telem[pbase + 11] > 0.0) acc.posQualQFrames++;
      const restE = telem[pbase + 14];
      if (restE > 0.0 && restE <= 1.0) acc.validRestitutionFrames++;
      else if (restE < 0.0 || restE > 1.0) acc.invalidRestitutionFrames++;

      activeTracks.push({ m, tid, fx, fy, dispVis, obsOrBridged: dispVis || bridgedAtDisp });

      let tl = acc.trackLife.get(tid);
      if (!tl) {
        tl = { n: 0, minX: ix, maxX: ix, minY: iy, maxY: iy, prevX: ix, prevY: iy, prevVis: latestVis };
        acc.trackLife.set(tid, tl);
      } else {
        if (latestVis && tl.prevVis && Math.hypot(ix - tl.prevX, iy - tl.prevY) > teleportStepMax) {
          acc.teleportJumps++;
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
        if (telem[pbase + k] !== 0.0) acc.inactiveNonZeroCount++;
      }
    }
  }

  const effK = Math.round(telem[1]);
  const fDisp = f - effK;
  if (fDisp >= 14 && acc.gtByFrame.size > 0) {
    const gts = acc.gtByFrame.get(fDisp) || [];
    const obsBrgTracks = activeTracks.filter((t) => t.obsOrBridged);
    const strictObsTracks = activeTracks.filter((t) => t.dispVis);
    recordMotFrame(acc.dispMot, gts, activeTracks);
    recordMotFrame(acc.obsBrgMot, gts, obsBrgTracks);
    recordMotFrame(acc.strictObsMot, gts, strictObsTracks);
  }
}

async function finalizeAndValidateClip(acc, sess, numFrames, resMode, swarmModeInt, modeLabel, extraReasons = [], opts = {}) {
  const clip = acc.clip;
  const hdMode = resMode === 1;
  const quiet = Boolean(opts.quiet);
  let staticPins = 0;
  for (const tl of acc.trackLife.values()) {
    if (tl.n >= 30 && Math.hypot(tl.maxX - tl.minX, tl.maxY - tl.minY) < 4.5 * acc.s) staticPins++;
  }

  const dispMot = finalizeMotStats(acc.dispMot);
  const obsBrgMot = finalizeMotStats(acc.obsBrgMot);
  const strictObsMot = finalizeMotStats(acc.strictObsMot);
  const gtRecallPct = dispMot.recallPct;
  const gtPrecisionPct = dispMot.precisionPct;
  const obsRecallPct = strictObsMot.recallPct;
  const obsPrecisionPct = strictObsMot.precisionPct;

  acc.validFreqSamples.sort((a, b) => a - b);
  const medianFreqHz =
    acc.validFreqSamples.length === 0
      ? 0.0
      : acc.validFreqSamples.length % 2 === 1
      ? acc.validFreqSamples[acc.validFreqSamples.length >> 1]
      : 0.5 *
        (acc.validFreqSamples[(acc.validFreqSamples.length >> 1) - 1] +
          acc.validFreqSamples[acc.validFreqSamples.length >> 1]);

  const totalSpawned = acc.trackAges.size;
  const minAge = clip.minPersistentAge || 15;
  const persistentTracks = [...acc.trackAges.values()].filter((c) => c >= minAge).length;
  const maxTrackLife = totalSpawned > 0 ? Math.max(...acc.trackAges.values()) : 0;
  const meanTickMs = acc.sumTickMs / Math.max(1, numFrames);
  const warmMeanTickMs = numFrames > 1 ? acc.warmSumTickMs / (numFrames - 1) : meanTickMs;
  const sortedWarm = [...acc.warmTickSamples].sort((a, b) => a - b);
  const medianTickMs =
    sortedWarm.length === 0
      ? meanTickMs
      : sortedWarm.length % 2 === 1
      ? sortedWarm[sortedWarm.length >> 1]
      : 0.5 * (sortedWarm[(sortedWarm.length >> 1) - 1] + sortedWarm[sortedWarm.length >> 1]);
  const lockPct = (100.0 * acc.framesWithTargets) / Math.max(1, numFrames);
  const padicSwaps = acc.finalPadicSwaps ?? Math.round(sess.telem[5]);
  const kalmanSwaps = acc.finalKalmanSwaps ?? Math.round(sess.telem[6]);

  let dualResGravErr = 0.0;
  let dualResSummary = '';
  if (clip.checkDualResGravity && hasResSwitch) {
    const mp4Path = path.join(SAMPLES_DIR, clip.file);
    const otherRes = await measureLockedSceneGravity(mp4Path, clip.fps, clip.labMode, hdMode ? 0 : 1, swarmModeInt);
    const gSd = hdMode ? otherRes.grav : acc.lastSceneGrav;
    const gHd = hdMode ? acc.lastSceneGrav : otherRes.grav;
    dualResGravErr = gSd > 1e-6 ? Math.abs(gHd / (2.0 * gSd) - 1.0) : 1.0;
    dualResSummary = ` | SD/HD 2x err=${(dualResGravErr * 100).toFixed(1)}% (g_sd=${gSd.toFixed(1)}, g_hd=${gHd.toFixed(1)})`;
  }

  const minMomResidStr = Number.isFinite(acc.minPairMomResid) ? acc.minPairMomResid.toFixed(3) : '--';
  const clipSpeedBudgetMs = (hdMode ? (clip.maxTickMsHd ?? 1.00) : (clip.maxTickMsSd ?? 0.35)) * budgetScale;
  const resLabel = hdMode ? 'HD' : 'SD';
  const swarmActiveRate =
    acc.swarmPostWarmupFrames > 0 ? acc.swarmPostWarmupActiveFrames / acc.swarmPostWarmupFrames : 0.0;
  const swarmP10Obs = computePercentileCal(acc.swarmActiveObsCounts, 10);
  const meanStarlingOraclePrec =
    acc.starlingOraclePrecisions.length > 0
      ? acc.starlingOraclePrecisions.reduce((a, b) => a + b, 0) / acc.starlingOraclePrecisions.length
      : 0.0;

  if (!quiet) {
    console.log(`\n[${clip.file} | ${resLabel} | ${modeLabel}] (${numFrames} frames @ ${clip.fps}fps) — ${clip.desc}`);
    console.log(
      `  Compute Tick  : warmMean=${warmMeanTickMs.toFixed(3)} ms (<${clipSpeedBudgetMs.toFixed(2)} ms) | warmP50=${medianTickMs.toFixed(3)} ms (rawMean=${meanTickMs.toFixed(3)} ms) | NaNs: ${acc.nanDetected ? 'FAIL' : '0'} | tailZeros: ${acc.tailNonZeroCount === 0 ? 'OK' : 'FAIL'} | max ||Lv-lv||: ${acc.maxEigRes.toExponential(2)}`
    );
    console.log(
      `  ECA Clutter   : ${acc.lastClutterDb.toFixed(2)} dB (peak ${acc.maxClutterDb.toFixed(2)} dB, bg mask ${(acc.lastEvalRatio * 100).toFixed(1)}%)`
    );
    console.log(
      `  Target Locks  : ${lockPct.toFixed(1)}% frames locked | avg ${(acc.sumActiveTargets / numFrames).toFixed(2)} (max ${acc.maxActiveTargets}) | spawned=${totalSpawned} (persistent>=${minAge}f: ${persistentTracks}, longest=${maxTrackLife}f) | obs/brg=${acc.observedOrBridgedSlots}, coastOnly=${acc.coastedOnlySlots}`
    );
    if (dispMot.gtTotal > 0) {
      const fmtMot = (m) => {
        const rStr = m.recallPct !== null ? `${m.recallPct.toFixed(1)}%` : 'N/A';
        const pStr = m.precisionPct !== null ? `${m.precisionPct.toFixed(1)}%` : 'N/A';
        const mStr = m.motaPct !== null ? `${m.motaPct.toFixed(1)}%` : 'N/A';
        const rmseStr = m.rmsePx !== null ? `${m.rmsePx.toFixed(2)} px` : 'N/A';
        const motpStr = m.motpPx !== null ? `${m.motpPx.toFixed(2)} px` : 'N/A';
        const hotaStr = m.hota
          ? `${(m.hota.hota * 100).toFixed(1)}% (DetA=${(m.hota.detA * 100).toFixed(1)}%, AssA=${(m.hota.assA * 100).toFixed(1)}%, LocA=${(m.hota.locA * 100).toFixed(1)}%)`
          : 'N/A';
        return `Recall=${rStr} (${m.gtMatched}/${m.gtTotal}) | Prec=${pStr} (FP=${m.gtUnmatchedFp}) | MOTA=${mStr} | RMSE(MOTP)=${rmseStr} (${motpStr}) | HOTA=${hotaStr} | IDSW=${m.gtIdSwaps} | Frag=${m.gtFrags} | Hijacks=${m.gtHijacks}`;
      };
      console.log(
        `  Display+Coast : ${fmtMot(dispMot)} | Teleports=${acc.teleportJumps} | StaticPins=${staticPins}`
      );
      if (acc.bridgedSlots > 0) {
        console.log(`  Obs+Bridge    : ${fmtMot(obsBrgMot)} | BridgeSlots=${acc.bridgedSlots}`);
      }
      console.log(
        `  Observed Only : ${fmtMot(strictObsMot)} | StrictVisSlots=${acc.strictObsSlots} | CoastSlots=${acc.coastedOnlySlots}`
      );
    }
    if (clip.expectDenseSwarm) {
      console.log(
        `  Wave-2 Swarm  : active=${acc.swarmPostWarmupActiveFrames}/${acc.swarmPostWarmupFrames} (${(swarmActiveRate * 100).toFixed(1)}%) | peakConf=${acc.swarmConfirmedPeak} | p10Obs=${swarmP10Obs} | parityErr=${acc.swarmCountParityErrors} | microDup=${acc.microDupTidErrors} | movingDarkFeat1to1Prec=${(meanStarlingOraclePrec * 100).toFixed(1)}% (${acc.starlingOraclePrecisions.length} gain/pan-norm CC samples; surrogate w/o bird GT)`
      );
    }
    console.log(
      `  Features      : bridges=${acc.bridgeFrames}f | apexFrames=${acc.apexFrames}f | hilbertValid=${acc.phaseValidFrames}f | freqValid=${acc.freqValidFrames}f (med=${medianFreqHz.toFixed(2)} Hz) | percStrikes=${acc.percTriggers} | 2adicAudit=${padicSwaps} vs kalmanSwaps=${kalmanSwaps}`
    );
    console.log(
      `  Kinematics    : g_hat=${acc.lastSceneGrav.toFixed(1)} px/s² (conf=${acc.maxGravConf.toFixed(2)}, arcs=${acc.maxGravArcs})${dualResSummary} | virtualPivot=${acc.pivotFrames}f | maxDeflect=${acc.maxDeflectDeg.toFixed(1)}° | scatter=${acc.lastScatterDeg.toFixed(1)}°`
    );
    console.log(
      `  Physics ODE   : regimes=[CV:${acc.regimeFrames[0]}, BAL:${acc.regimeFrames[1]}, DRAG:${acc.regimeFrames[2]}, BST:${acc.regimeFrames[3]}, CHT:${acc.regimeFrames[4]}, PND:${acc.regimeFrames[5]}, TRM:${acc.regimeFrames[6]}, IMP:${acc.regimeFrames[7]}, CTRV:${acc.regimeFrames[8]}] | jumps=[peg:${acc.jumpFrames[1]}, burn:${acc.jumpFrames[2]}, chute:${acc.jumpFrames[3]}, col:${acc.jumpFrames[4]}] | ellipT=${acc.posEllipticFrames}f Qpos=${acc.posQualQFrames}f e_n=${acc.validRestitutionFrames}f`
    );
    console.log(
      `  Ensemble      : kuramotoR=${acc.maxKuramotoR.toFixed(3)} (N=${acc.maxKuramotoCount}) | lyap=${acc.maxLyapunov.toFixed(3)}/s | pairEvents=${acc.pairEventCount} (bad=${acc.invalidPairEvents}, minMomRes=${minMomResidStr}, maxMR=${acc.maxPairMassRatio.toFixed(2)}) | taubinDelta=${acc.maxTaubinDelta.toFixed(2)}px | inactiveNonZero=${acc.inactiveNonZeroCount}`
    );
  }

  let passed = true;
  if (strictGate) {
    const reasons = [...sess.ptrErrors, ...extraReasons];
    const minPersistent = clip.minPersistent ?? 1;
    const minLockPct = clip.minLockPct ?? 90.0;
    const minRecallPct = clip.minRecallPct ?? 78.0;
    const minPrecisionPct = clip.minPrecisionPct ?? 12.0;
    const minClutterDb = clip.minClutterDb ?? 35.0;
    const maxStaticPins = clip.maxStaticPins ?? 1;
    const minApex = hdMode ? (clip.minApexHd || 0) : (clip.minApexSd || 0);
    if (acc.nanDetected) reasons.push('NaN/Inf in telemetry');
    if (acc.tailNonZeroCount > 0) reasons.push(`reservedTailNonZero=${acc.tailNonZeroCount}`);
    if (acc.inactiveNonZeroCount > 0) reasons.push(`inactivePhysicsNonZero=${acc.inactiveNonZeroCount}`);
    if (acc.invalidEncodingErrors > 0) reasons.push(`invalidSlotEncoding=${acc.invalidEncodingErrors}`);
    if (acc.duplicateTidErrors > 0) reasons.push(`duplicateTidsInFrame=${acc.duplicateTidErrors}`);
    if (acc.invalidRegimeCount > 0) reasons.push(`invalidRegime=${acc.invalidRegimeCount}`);
    if (acc.invalidJumpCount > 0) reasons.push(`invalidJump=${acc.invalidJumpCount}`);
    if (acc.maxEigRes > 1e-5) reasons.push(`maxEigRes=${acc.maxEigRes.toExponential(2)}`);
    if (!noSpeedGate && warmMeanTickMs >= clipSpeedBudgetMs) {
      reasons.push(`warmMeanTickMs=${warmMeanTickMs.toFixed(3)}>=${clipSpeedBudgetMs.toFixed(2)}`);
    }
    if (persistentTracks < minPersistent) reasons.push(`persistentTracks=${persistentTracks}<${minPersistent}`);
    if (lockPct < minLockPct) reasons.push(`lockPct=${lockPct.toFixed(1)}<${minLockPct}`);
    if (acc.lastClutterDb < minClutterDb) reasons.push(`clutterDb=${acc.lastClutterDb.toFixed(2)}<${minClutterDb}`);
    if (kalmanSwaps > 0 ? padicSwaps >= kalmanSwaps : padicSwaps > 0) reasons.push(`swapOrdering(${padicSwaps} vs ${kalmanSwaps})`);
    if (acc.apexFrames < minApex) reasons.push(`apexFrames=${acc.apexFrames}<${minApex}`);
    if (acc.bridgeFrames < (clip.minBridges || 0)) reasons.push(`bridgeFrames=${acc.bridgeFrames}<${clip.minBridges}`);
    if (acc.phaseValidFrames < (clip.minPhase || 0)) reasons.push(`phaseValidFrames=${acc.phaseValidFrames}<${clip.minPhase}`);
    if (acc.percTriggers < (clip.minPerc || 0)) reasons.push(`percTriggers=${acc.percTriggers}<${clip.minPerc}`);
    if (clip.minPivotFrames !== undefined && acc.pivotFrames < clip.minPivotFrames) {
      reasons.push(`pivotFrames=${acc.pivotFrames}<${clip.minPivotFrames}`);
    }
    if (clip.maxPivotFrames !== undefined && acc.pivotFrames > clip.maxPivotFrames) {
      reasons.push(`falsePivotFrames=${acc.pivotFrames}>${clip.maxPivotFrames}`);
    }
    if (clip.minFreqValidFrames !== undefined && acc.freqValidFrames < clip.minFreqValidFrames) {
      reasons.push(`freqValidFrames=${acc.freqValidFrames}<${clip.minFreqValidFrames}`);
    }
    if (clip.maxFreqValidFrames !== undefined && acc.freqValidFrames > clip.maxFreqValidFrames) {
      reasons.push(`falseFreqValidFrames=${acc.freqValidFrames}>${clip.maxFreqValidFrames}`);
    }
    if (clip.freqRangeHz) {
      const [loHz, hiHz] = clip.freqRangeHz;
      if (medianFreqHz < loHz || medianFreqHz > hiHz) {
        reasons.push(`medianFreqHz=${medianFreqHz.toFixed(3)} outside [${loHz}, ${hiHz}]`);
      }
    }
    if (clip.minGravArcs !== undefined && acc.maxGravArcs < clip.minGravArcs) {
      reasons.push(`gravArcs=${acc.maxGravArcs}<${clip.minGravArcs}`);
    }
    if (clip.minGravConf !== undefined && acc.maxGravConf < clip.minGravConf) {
      reasons.push(`gravConf=${acc.maxGravConf.toFixed(2)}<${clip.minGravConf}`);
    }
    if (clip.checkDualResGravity && dualResGravErr > 0.16) {
      reasons.push(`dualResGravErr=${(dualResGravErr * 100).toFixed(1)}%>16%`);
    }
    if (clip.physicsNegativeGate) {
      if (acc.regimeFrames[3] > 0) reasons.push(`falseBoost=${acc.regimeFrames[3]}`);
      if (acc.regimeFrames[4] > 0) reasons.push(`falseParachute=${acc.regimeFrames[4]}`);
      if (acc.regimeFrames[5] > 0) reasons.push(`falsePendulum=${acc.regimeFrames[5]}`);
      if (acc.jumpFrames[1] > 0) reasons.push(`falsePegJump=${acc.jumpFrames[1]}`);
      if (acc.jumpFrames[4] > 0 || acc.maxLastPairFrame > 0) {
        reasons.push(`falseCollision(j4=${acc.jumpFrames[4]},pairF=${acc.maxLastPairFrame})`);
      }
      if (acc.maxKuramotoR > 0.0) reasons.push(`falseKuramotoR=${acc.maxKuramotoR.toFixed(3)}`);
      if (acc.maxLyapunov > 0.0) reasons.push(`falseLyapunov=${acc.maxLyapunov.toFixed(3)}`);
    }
    if (clip.requirePendulumPhysics) {
      if (acc.regimeFrames[5] < 1) reasons.push('missingPendulumRegime5');
      if (acc.jumpFrames[1] < 1) reasons.push('missingGalileoPegJump1');
      if (acc.posEllipticFrames < 1) reasons.push('missingEllipticPeriod');
      if (acc.posQualQFrames < 1) reasons.push('missingQualityFactorQ');
    }
    if (clip.requireCollisionPhysics) {
      if (acc.jumpFrames[4] < 1) reasons.push('missingCollisionJump4');
      if (acc.pairEventCount < 1) reasons.push('missingCollisionPairEvent');
      if (acc.validRestitutionFrames < 1) reasons.push('missingValidRestitutionE');
      if (acc.invalidRestitutionFrames > 0) reasons.push(`invalidRestitutionE=${acc.invalidRestitutionFrames}`);
      if (acc.invalidPairEvents > 0) reasons.push(`invalidPairEvents=${acc.invalidPairEvents}`);
    }
    if (clip.requireImpulsivePhysics) {
      if (acc.regimeFrames[7] < 1) reasons.push('missingImpulsiveRegime7');
      if (acc.regimeFrames[3] > 0) reasons.push(`falseBoost=${acc.regimeFrames[3]}`);
      if (acc.regimeFrames[4] > 0) reasons.push(`falseParachute=${acc.regimeFrames[4]}`);
      if (acc.regimeFrames[5] > 0) reasons.push(`falsePendulum=${acc.regimeFrames[5]}`);
      if (acc.maxKuramotoR > 0.0) reasons.push(`falseKuramotoR=${acc.maxKuramotoR.toFixed(3)}`);
      if (acc.maxLyapunov > 0.0) reasons.push(`falseLyapunov=${acc.maxLyapunov.toFixed(3)}`);
    }
    if (clip.minBoostFrames !== undefined && acc.regimeFrames[3] < clip.minBoostFrames) {
      reasons.push(`boostFrames=${acc.regimeFrames[3]}<${clip.minBoostFrames}`);
    }
    if (clip.minBurnoutJumpFrames !== undefined && acc.jumpFrames[2] < clip.minBurnoutJumpFrames) {
      reasons.push(`burnoutJump2Frames=${acc.jumpFrames[2]}<${clip.minBurnoutJumpFrames}`);
    }
    if (clip.minParachuteFrames !== undefined && acc.regimeFrames[4] < clip.minParachuteFrames) {
      reasons.push(`parachuteFrames=${acc.regimeFrames[4]}<${clip.minParachuteFrames}`);
    }
    if (clip.minParachuteJumpFrames !== undefined && acc.jumpFrames[3] < clip.minParachuteJumpFrames) {
      reasons.push(`parachuteJump3Frames=${acc.jumpFrames[3]}<${clip.minParachuteJumpFrames}`);
    }
    if (clip.minChuteOrDragFrames !== undefined) {
      const chuteOrDrag = acc.regimeFrames[4] + acc.regimeFrames[2] + acc.jumpFrames[3];
      if (chuteOrDrag < clip.minChuteOrDragFrames) {
        reasons.push(`chuteOrDragFrames=${chuteOrDrag}<${clip.minChuteOrDragFrames}`);
      }
    }
    if (clip.minLyapunovMax !== undefined && acc.maxLyapunov <= clip.minLyapunovMax) {
      reasons.push(`lyapunovMax=${acc.maxLyapunov.toFixed(3)}<=${clip.minLyapunovMax}`);
    }
    if (clip.minKuramotoOscCount !== undefined && acc.maxKuramotoCount < clip.minKuramotoOscCount) {
      reasons.push(`kuramotoCount=${acc.maxKuramotoCount}<${clip.minKuramotoOscCount}`);
    }
    if (clip.minKuramotoOrderR !== undefined && acc.maxKuramotoR < clip.minKuramotoOrderR) {
      reasons.push(`kuramotoOrderR=${acc.maxKuramotoR.toFixed(3)}<${clip.minKuramotoOrderR}`);
    }
    if (clip.hasGt !== false) {
      if (dispMot.gtTotal <= 0) reasons.push('gtTotal===0');
      if (gtRecallPct === null || gtRecallPct < minRecallPct) {
        reasons.push(`dispCoastRecall=${gtRecallPct === null ? 'N/A' : `${gtRecallPct.toFixed(1)}%`}<${minRecallPct}%`);
      }
      if (gtPrecisionPct === null || gtPrecisionPct < minPrecisionPct) {
        reasons.push(`dispCoastPrecision=${gtPrecisionPct === null ? 'N/A' : `${gtPrecisionPct.toFixed(1)}%`}<${minPrecisionPct}%`);
      }
      if (dispMot.gtHijacks > 16) reasons.push(`gtHijacks=${dispMot.gtHijacks}`);
      const minObsRecallPct = clip.minObsRecallPct ?? Math.max(25.0, minRecallPct - 35.0);
      const minObsPrecisionPct = clip.minObsPrecisionPct ?? minPrecisionPct;
      if (obsRecallPct === null || obsRecallPct < minObsRecallPct) {
        reasons.push(`strictObsRecall=${obsRecallPct === null ? 'N/A' : `${obsRecallPct.toFixed(1)}%`}<${minObsRecallPct}%`);
      }
      if (obsPrecisionPct === null || obsPrecisionPct < minObsPrecisionPct) {
        reasons.push(`strictObsPrecision=${obsPrecisionPct === null ? 'N/A' : `${obsPrecisionPct.toFixed(1)}%`}<${minObsPrecisionPct}%`);
      }
    }
    if (acc.arenaMismatchErrors > 0) reasons.push(`arenaMismatch=${acc.arenaMismatchErrors}`);
    if (acc.swarmTailNonZeroCount > 0) reasons.push(`swarmTailNonZero=${acc.swarmTailNonZeroCount}`);
    if (acc.swarmSentinelErrors > 0) reasons.push(`swarmSentinelErrors=${acc.swarmSentinelErrors}`);
    if (acc.microInvalidEncodingErrors > 0) reasons.push(`microInvalidEnc=${acc.microInvalidEncodingErrors}`);
    if (acc.microDupTidErrors > 0) reasons.push(`microDupTid=${acc.microDupTidErrors}`);
    if (acc.inspectorErrors > 0) reasons.push(`inspector=${acc.inspectorMsgs.slice(0, 2).join('|') || acc.inspectorErrors}`);
    if (clip.expectDenseSwarm) {
      if (!sess.swarmTelem || !sess.microTracks) reasons.push('missingWave2SwarmBuffers');
      if (acc.swarmCountParityErrors > 0) reasons.push(`swarmObsParity=${acc.swarmCountParityErrors}`);
      if (swarmActiveRate < 0.85) reasons.push(`swarmActiveRate=${(swarmActiveRate * 100).toFixed(1)}%<85%`);
      const minSwarmPeak = hdMode ? 1000 : 400;
      const minSwarmP10 = hdMode ? 1000 : 350;
      // At SD 320x180 half-res ~1,200 birds optically merge into ~460-585 connected silhouettes; HD 640x360 strictly enforces >=65%.
      const minOraclePrec = hdMode ? 0.65 : 0.35;
      if (acc.swarmConfirmedPeak < minSwarmPeak) reasons.push(`swarmPeak=${acc.swarmConfirmedPeak}<${minSwarmPeak}`);
      if (swarmP10Obs < minSwarmP10) reasons.push(`swarmP10Obs=${swarmP10Obs}<${minSwarmP10}`);
      if (acc.starlingOraclePrecisions.length < 4) reasons.push(`oracleSamples=${acc.starlingOraclePrecisions.length}<4`);
      if (meanStarlingOraclePrec < minOraclePrec) {
        reasons.push(`movingDarkFeatPrecision=${(meanStarlingOraclePrec * 100).toFixed(1)}%<${(minOraclePrec * 100).toFixed(0)}%`);
      }
    }
    if (acc.teleportJumps > 0) reasons.push(`teleportJumps=${acc.teleportJumps}`);
    if (staticPins > maxStaticPins) reasons.push(`staticPins=${staticPins}>${maxStaticPins}`);

    if (opts.outReasons) opts.outReasons.push(...reasons);
    const passTag = budgetScale > 1.0 ? '[PARTIAL PASS - SCALED BUDGET]' : '[PASS]';
    if (reasons.length > 0) {
      passed = false;
      if (!quiet) {
        console.error(`  [FAIL] Calibration invariant missed on ${clip.file} (${resLabel}/${modeLabel}): ${reasons.join(', ')}`);
      }
    } else if (!quiet) {
      console.log(`  ${passTag} All calibration & physics-proof invariants verified on ${clip.file} (${resLabel}/${modeLabel})`);
    }
  }

  return passed;
}

function auditInactiveSwarmBuffers(sess) {
  let nanDetected = false;
  let sentinelErrors = 0;
  let microActive = 0;
  const { continuumField, modalField, rigidBodies, microTracks } = sess;
  if (continuumField) {
    for (let c = 0; c < 880; c++) {
      const fb = c * 6;
      for (let k = 0; k < 6; k++) {
        if (!Number.isFinite(continuumField[fb + k])) nanDetected = true;
      }
      if (
        continuumField[fb + 5] !== 0.0 ||
        continuumField[fb + 4] !== 0.0 ||
        continuumField[fb] !== 0.0 ||
        continuumField[fb + 1] !== 0.0 ||
        continuumField[fb + 2] !== 0.0 ||
        continuumField[fb + 3] !== 0.0
      ) {
        sentinelErrors++;
      }
    }
  }
  if (modalField) {
    for (let c = 0; c < 880; c++) {
      const phi = modalField[c];
      if (!Number.isFinite(phi)) nanDetected = true;
      else if (phi !== 0.0) sentinelErrors++;
    }
  }
  if (rigidBodies) {
    for (let i = 0; i < rigidBodies.length; i++) {
      if (!Number.isFinite(rigidBodies[i])) nanDetected = true;
    }
  }
  if (microTracks) {
    for (let i = 0; i < 1536; i++) {
      const mb = i * 8;
      for (let k = 0; k < 8; k++) {
        if (!Number.isFinite(microTracks[mb + k])) nanDetected = true;
      }
      if (microTracks[mb + 5] !== 0.0) microActive++;
    }
  }
  return { nanDetected, sentinelErrors, microActive };
}

async function runSingleClip(clip, resMode, swarmModeInt, modeLabel) {
  const mp4Path = path.join(SAMPLES_DIR, clip.file);
  if (!fs.existsSync(mp4Path)) throw new Error(`Missing sample video: ${mp4Path}`);
  if (strictGate && clip.hasGt !== false && (!gtDataset || !gtDataset[clip.file])) {
    throw new Error(`Missing ground-truth annotations for ${clip.file} in ${GT_PATH}`);
  }
  const effectiveSwarmMode = clip.expectDenseSwarm && swarmModeInt === null ? 1 : swarmModeInt;
  const dtSec = 1.0 / clip.fps;
  const checkAutoInactivity = effectiveSwarmMode === 1 && !clip.expectDenseSwarm;

  // Untimed 96-frame JIT warmup pass to compile all Wasm & Lanczos paths before measurement
  {
    const warmSess = acquirePooledWasmSession(0, resMode, effectiveSwarmMode, clip.labMode);
    await streamMp4Frames(
      mp4Path,
      warmSess.w,
      warmSess.h,
      (frameRgba, f) => {
        warmSess.ringViews[f % 12].set(frameRgba);
        if (typeof warmSess.exp.chronos_set_ingest_pts === 'function') {
          warmSess.exp.chronos_set_ingest_pts(f * dtSec);
        }
        warmSess.exp.chronos_pre_tick(dtSec);
        warmSess.solveLanczos();
        warmSess.exp.chronos_post_tick(dtSec);
      },
      96
    );
  }

  const sess = acquirePooledWasmSession(0, resMode, effectiveSwarmMode, clip.labMode);
  const acc = createClipAccumulator(clip, sess.w, sess.h, sess.arenaBytes);

  const { numFrames } = await streamMp4Frames(mp4Path, sess.w, sess.h, (frameRgba, f) => {
    sess.ringViews[f % 12].set(frameRgba);
    if (typeof sess.exp.chronos_set_ingest_pts === 'function') {
      sess.exp.chronos_set_ingest_pts(f * dtSec);
    }
    const t0 = performance.now();
    sess.exp.chronos_pre_tick(dtSec);
    sess.solveLanczos();
    sess.exp.chronos_post_tick(dtSec);
    const tickDt = performance.now() - t0;
    const resNorm = sess.solveLanczos.computeResNorm();
    recordSessionFrame(acc, sess, tickDt, resNorm, f, checkAutoInactivity, frameRgba);
  });

  const extraReasons = [];
  if (checkAutoInactivity && acc.swarmProbeOrActiveFrames > 0) {
    extraReasons.push(`falseSwarmProbeOrActiveFrames=${acc.swarmProbeOrActiveFrames}`);
  }
  const ok = await finalizeAndValidateClip(acc, sess, numFrames, resMode, effectiveSwarmMode, modeLabel, extraReasons);
  return { passed: ok, totalResults: 1, passedResults: ok ? 1 : 0 };
}

/**
 * Paired within-frame Off(0) vs Auto(1)-inactive calibration & overhead measurement:
 *   1. Pass 0 streams the full clip on pre-faulted pooled Wasm sessions, running full per-frame
 *      recordSessionFrame correctness + CLEAR-MOT/HOTA + sentinel auditing while completing TurboFan JIT warmup.
 *   2. Executes a preselected even ABBA-balanced trial count (20..32 trials determined solely from
 *      clip frame count, zero optional stopping) on the same pre-faulted pooled sessions without
 *      sweeping cold inactive swarm buffers inside the timed frame loop, checking per-frame 1,568-float
 *      parity, scalar inactivity, and post-trial inactive buffer sentinels.
 */
async function runPairedDualModeClip(clip, resMode) {
  const mp4Path = path.join(SAMPLES_DIR, clip.file);
  if (!fs.existsSync(mp4Path)) throw new Error(`Missing sample video: ${mp4Path}`);
  if (strictGate && clip.hasGt !== false && (!gtDataset || !gtDataset[clip.file])) {
    throw new Error(`Missing ground-truth annotations for ${clip.file} in ${GT_PATH}`);
  }

  const dtSec = 1.0 / clip.fps;
  const tickOne = (sess) => {
    const t0 = performance.now();
    sess.exp.chronos_pre_tick(dtSec);
    sess.solveLanczos();
    sess.exp.chronos_post_tick(dtSec);
    return performance.now() - t0;
  };

  let numFrames = 0;
  let outputMismatchFrames = 0;
  let totalAutoProbeOrActive = 0;
  let accOff = null;
  let accAuto = null;

  // Pass 0: Full-clip correctness + per-frame sentinel audit + full-pipeline TurboFan warmup
  {
    const pass0Off = acquirePooledWasmSession(0, resMode, 0, clip.labMode);
    const pass0Auto = acquirePooledWasmSession(1, resMode, 1, clip.labMode);
    accOff = createClipAccumulator(clip, pass0Off.w, pass0Off.h, pass0Off.arenaBytes);
    accAuto = createClipAccumulator(clip, pass0Auto.w, pass0Auto.h, pass0Auto.arenaBytes);

    const pass0Meta = await streamMp4Frames(mp4Path, pass0Off.w, pass0Off.h, (frameRgba, f) => {
      const slot = f % 12;
      pass0Off.ringViews[slot].set(frameRgba);
      pass0Auto.ringViews[slot].set(frameRgba);
      pass0Off.exp.chronos_set_ingest_pts(f * dtSec);
      pass0Auto.exp.chronos_set_ingest_pts(f * dtSec);

      let dtOff;
      let dtAuto;
      let resOff;
      let resAuto;
      if ((f & 1) === 0) {
        dtOff = tickOne(pass0Off);
        resOff = pass0Off.solveLanczos.computeResNorm();
        dtAuto = tickOne(pass0Auto);
        resAuto = pass0Auto.solveLanczos.computeResNorm();
      } else {
        dtAuto = tickOne(pass0Auto);
        resAuto = pass0Auto.solveLanczos.computeResNorm();
        dtOff = tickOne(pass0Off);
        resOff = pass0Off.solveLanczos.computeResNorm();
      }
      for (let i = 0; i < TELEMETRY_LEN; i++) {
        if (pass0Off.telem[i] !== pass0Auto.telem[i]) {
          outputMismatchFrames++;
          break;
        }
      }
      recordSessionFrame(accOff, pass0Off, dtOff, resOff, f, false, frameRgba);
      recordSessionFrame(accAuto, pass0Auto, dtAuto, resAuto, f, true, frameRgba);
    });

    numFrames = pass0Meta.numFrames;
    accOff.finalPadicSwaps = Math.round(pass0Off.telem[5]);
    accOff.finalKalmanSwaps = Math.round(pass0Off.telem[6]);
    accAuto.finalPadicSwaps = Math.round(pass0Auto.telem[5]);
    accAuto.finalKalmanSwaps = Math.round(pass0Auto.telem[6]);
    totalAutoProbeOrActive += accAuto.swarmProbeOrActiveFrames;

    const warmCount = Math.max(0, numFrames - 1);
    accOff.sumTickMs = 0.0;
    accOff.warmSumTickMs = 0.0;
    accOff.tickSamples = [];
    accOff.warmTickSamples = new Array(warmCount);
    accAuto.sumTickMs = 0.0;
    accAuto.warmSumTickMs = 0.0;
    accAuto.tickSamples = [];
    accAuto.warmTickSamples = new Array(warmCount);
  }

  const warmFramesPerTrial = Math.max(1, numFrames - 1);
  const rawPreselectedTrials = Math.max(20, Math.min(32, Math.ceil(5400 / warmFramesPerTrial)));
  const DUAL_TRIALS = rawPreselectedTrials + (rawPreselectedTrials & 1);
  const T_975_BY_DF = {
    1: 12.706205,
    3: 3.182446,
    5: 2.570582,
    7: 2.364624,
    9: 2.262157,
    11: 2.200985,
    13: 2.160369,
    15: 2.131450,
    17: 2.109816,
    19: 2.093024,
    21: 2.079614,
    23: 2.068658,
    25: 2.059539,
    27: 2.051831,
    29: 2.045230,
    31: 2.039513,
  };
  const clipMaxOverheadMs = 0.005 * budgetScale;
  const clipSpeedBudgetMs = (resMode === 1 ? (clip.maxTickMsHd ?? 1.00) : (clip.maxTickMsSd ?? 0.35)) * budgetScale;
  let sessOff = null;
  let sessAuto = null;
  const offTrialExtras = [];
  const autoTrialExtras = [];
  const offTrialTiming = [];
  const autoTrialTiming = [];
  const pairedWarmDiffs = new Float64Array(DUAL_TRIALS * Math.max(1, numFrames - 1));
  const sortScratch = new Float64Array(pairedWarmDiffs.length);
  let diffCount = 0;
  const trialMeans = [];

  function computeDualStats() {
    let sumDiff = 0.0;
    for (let i = 0; i < diffCount; i++) sumDiff += pairedWarmDiffs[i];
    const meanDiff = diffCount > 0 ? sumDiff / diffCount : 0.0;
    let varSum = 0.0;
    for (let i = 0; i < diffCount; i++) {
      const d = pairedWarmDiffs[i] - meanDiff;
      varSum += d * d;
    }
    const stdDiff = diffCount > 1 ? Math.sqrt(varSum / (diffCount - 1)) : 0.0;
    const seDiff = diffCount > 0 ? stdDiff / Math.sqrt(diffCount) : 0.0;
    const ci95Diff = 1.96 * seDiff;
    let medianDiff = 0.0;
    if (diffCount > 0) {
      const slice = sortScratch.subarray(0, diffCount);
      slice.set(pairedWarmDiffs.subarray(0, diffCount));
      slice.sort();
      medianDiff = slice[diffCount >> 1];
    }

    const nTrials = trialMeans.length;
    const gTrim = nTrials >= 6 ? Math.floor(nTrials * 0.2) : 0;
    const hTrim = Math.max(1, nTrials - 2 * gTrim);
    let trialMean = 0.0;
    let seTrial = 0.0;
    let df = Math.max(1, nTrials - 1);
    if (nTrials > 0 && gTrim === 0) {
      trialMean = trialMeans.reduce((a, b) => a + b, 0) / nTrials;
      let trialVar = 0.0;
      for (let i = 0; i < nTrials; i++) {
        const dt = trialMeans[i] - trialMean;
        trialVar += dt * dt;
      }
      seTrial = nTrials > 1 ? Math.sqrt(trialVar / (nTrials - 1) / nTrials) : 0.0;
    } else if (nTrials > 0) {
      const sortedT = [...trialMeans].sort((a, b) => a - b);
      let sumT = 0.0;
      for (let i = gTrim; i < nTrials - gTrim; i++) sumT += sortedT[i];
      trialMean = sumT / hTrim;
      const loW = sortedT[gTrim];
      const hiW = sortedT[nTrials - 1 - gTrim];
      let sumW = 0.0;
      for (let i = 0; i < nTrials; i++) {
        sumW += i < gTrim ? loW : i >= nTrials - gTrim ? hiW : sortedT[i];
      }
      const meanW = sumW / nTrials;
      let ssdW = 0.0;
      for (let i = 0; i < nTrials; i++) {
        const vw = i < gTrim ? loW : i >= nTrials - gTrim ? hiW : sortedT[i];
        const dw = vw - meanW;
        ssdW += dw * dw;
      }
      seTrial = hTrim > 1 ? Math.sqrt(ssdW / (hTrim * (hTrim - 1))) : 0.0;
      df = Math.max(1, hTrim - 1);
    }
    const tCrit = T_975_BY_DF[df] ?? 2.039513;
    const ci95Trial = tCrit * seTrial;
    const upper95Trial = trialMean + ci95Trial;
    return { nDiff: diffCount, meanDiff, ci95Diff, medianDiff, nTrials, gTrim, trialMean, ci95Trial, upper95Trial };
  }

  for (let trial = 0; trial < DUAL_TRIALS; trial++) {
    runExplicitGc();
    const poolOff = trial & 1;
    const poolAuto = 1 - poolOff;
    const sOff = acquirePooledWasmSession(poolOff, resMode, 0, clip.labMode);
    const sAuto = acquirePooledWasmSession(poolAuto, resMode, 1, clip.labMode);
    const tOff = sOff.telem;
    const tAuto = sAuto.telem;
    const stAuto = sAuto.swarmTelem;
    const rvOff = sOff.ringViews;
    const rvAuto = sAuto.ringViews;
    const expOff = sOff.exp;
    const expAuto = sAuto.exp;
    const expectedArena = sOff.arenaBytes;

    let trialDiffSum = 0.0;
    let trialDiffCnt = 0;
    let sumOff = 0.0;
    let warmSumOff = 0.0;
    let sumAuto = 0.0;
    let warmSumAuto = 0.0;
    let warmCnt = 0;
    let maxEigOff = 0.0;
    let maxEigAuto = 0.0;
    let nanOff = false;
    let nanAuto = false;
    let tailOff = 0;
    let tailAuto = 0;
    let arenaErrOff = 0;
    let arenaErrAuto = 0;
    let swarmTailErr = 0;
    let trialAutoProbe = 0;
    let trialMismatch = 0;

    await streamMp4Frames(mp4Path, sOff.w, sOff.h, (frameRgba, f) => {
      const slot = f % 12;
      let dtOff;
      let dtAuto;
      const autoFirst = ((f ^ trial) & 1) === 1;
      if (!autoFirst) {
        rvOff[slot].set(frameRgba);
        expOff.chronos_set_ingest_pts(f * dtSec);
        rvAuto[slot].set(frameRgba);
        expAuto.chronos_set_ingest_pts(f * dtSec);
        dtOff = tickOne(sOff);
        dtAuto = tickOne(sAuto);
      } else {
        rvAuto[slot].set(frameRgba);
        expAuto.chronos_set_ingest_pts(f * dtSec);
        rvOff[slot].set(frameRgba);
        expOff.chronos_set_ingest_pts(f * dtSec);
        dtAuto = tickOne(sAuto);
        dtOff = tickOne(sOff);
      }
      const resOff = sOff.solveLanczos.computeResNorm();
      const resAuto = sAuto.solveLanczos.computeResNorm();
      sumOff += dtOff;
      sumAuto += dtAuto;
      if (resOff > maxEigOff) maxEigOff = resOff;
      if (resAuto > maxEigAuto) maxEigAuto = resAuto;

      if (f > 0) {
        const diff = dtAuto - dtOff;
        pairedWarmDiffs[diffCount++] = diff;
        trialDiffSum += diff;
        trialDiffCnt++;
        warmSumOff += dtOff;
        warmSumAuto += dtAuto;
        warmCnt++;
        if (trial === 0) {
          accOff.warmTickSamples[f - 1] = dtOff;
          accAuto.warmTickSamples[f - 1] = dtAuto;
        }
      }

      let mismatch = false;
      for (let i = 0; i < TELEMETRY_LEN; i++) {
        const vo = tOff[i];
        const va = tAuto[i];
        if (vo - vo !== 0) nanOff = true;
        if (va - va !== 0) nanAuto = true;
        if (vo !== va) mismatch = true;
        if (i >= TELEM_RESERVED_TAIL_START) {
          if (vo !== 0.0) tailOff++;
          if (va !== 0.0) tailAuto++;
        }
      }
      if (mismatch) trialMismatch++;
      if (expectedArena > 0) {
        if (Math.round(tOff[20]) !== expectedArena) arenaErrOff++;
        if (Math.round(tAuto[20]) !== expectedArena) arenaErrAuto++;
      }
      if (stAuto) {
        for (let i = 0; i < 64; i++) {
          const sv = stAuto[i];
          if (sv - sv !== 0) nanAuto = true;
          if (i >= SWARM_RESERVED_TAIL_START && sv !== 0.0) swarmTailErr++;
        }
        if (stAuto[1] !== 0.0 || stAuto[2] !== 0.0 || stAuto[3] !== 0.0 || stAuto[39] !== 0.0) {
          trialAutoProbe++;
        }
      }
    });

    const bufAuditOff = auditInactiveSwarmBuffers(sOff);
    const bufAuditAuto = auditInactiveSwarmBuffers(sAuto);
    if (bufAuditOff.nanDetected) nanOff = true;
    if (bufAuditAuto.nanDetected) nanAuto = true;
    if (bufAuditAuto.microActive > 0) trialAutoProbe++;

    trialMeans.push(trialDiffCnt > 0 ? trialDiffSum / trialDiffCnt : 0.0);
    outputMismatchFrames += trialMismatch;
    totalAutoProbeOrActive += trialAutoProbe;
    offTrialTiming.push({ sum: sumOff, warmSum: warmSumOff });
    autoTrialTiming.push({ sum: sumAuto, warmSum: warmSumAuto });

    sessOff = { ...sOff, ptrErrors: [...sOff.ptrErrors] };
    sessAuto = { ...sAuto, ptrErrors: [...sAuto.ptrErrors] };
    if (trial === 0) {
      if (nanOff) accOff.nanDetected = true;
      if (nanAuto) accAuto.nanDetected = true;
      accOff.tailNonZeroCount += tailOff;
      accAuto.tailNonZeroCount += tailAuto;
      accOff.swarmTailNonZeroCount += swarmTailErr;
      accAuto.swarmTailNonZeroCount += swarmTailErr;
      accOff.arenaMismatchErrors += arenaErrOff;
      accAuto.arenaMismatchErrors += arenaErrAuto;
      accOff.swarmSentinelErrors += bufAuditOff.sentinelErrors;
      accAuto.swarmSentinelErrors += bufAuditAuto.sentinelErrors;
      if (maxEigOff > accOff.maxEigRes) accOff.maxEigRes = maxEigOff;
      if (maxEigAuto > accAuto.maxEigRes) accAuto.maxEigRes = maxEigAuto;
      if (trialAutoProbe > 0) accAuto.swarmProbeOrActiveFrames += trialAutoProbe;
    } else {
      const rOffT = [...sOff.ptrErrors];
      const rAutoT = [...sAuto.ptrErrors];
      if (nanOff) rOffT.push('NaN/Inf in telemetry');
      if (nanAuto) rAutoT.push('NaN/Inf in telemetry');
      if (tailOff > 0) rOffT.push(`reservedTailNonZero=${tailOff}`);
      if (tailAuto > 0) rAutoT.push(`reservedTailNonZero=${tailAuto}`);
      if (swarmTailErr > 0) rAutoT.push(`swarmTailNonZero=${swarmTailErr}`);
      if (maxEigOff > 1e-5) rOffT.push(`maxEigRes=${maxEigOff.toExponential(2)}`);
      if (maxEigAuto > 1e-5) rAutoT.push(`maxEigRes=${maxEigAuto.toExponential(2)}`);
      if (arenaErrOff > 0) rOffT.push(`arenaMismatch=${arenaErrOff}`);
      if (arenaErrAuto > 0) rAutoT.push(`arenaMismatch=${arenaErrAuto}`);
      if (bufAuditOff.sentinelErrors > 0) rOffT.push(`swarmSentinelErrors=${bufAuditOff.sentinelErrors}`);
      if (bufAuditAuto.sentinelErrors > 0) rAutoT.push(`swarmSentinelErrors=${bufAuditAuto.sentinelErrors}`);
      if (rOffT.length > 0) offTrialExtras.push(`trial${trial}:[${rOffT.join(';')}]`);
      if (rAutoT.length > 0) autoTrialExtras.push(`trial${trial}:[${rAutoT.join(';')}]`);
      if (trialAutoProbe > 0) autoTrialExtras.push(`trial${trial}_autoProbe=${trialAutoProbe}`);
      if (trialMismatch > 0) autoTrialExtras.push(`trial${trial}_mismatch=${trialMismatch}`);
    }
  }

  const {
    nDiff,
    meanDiff,
    ci95Diff,
    medianDiff,
    nTrials,
    trialMean,
    ci95Trial,
    upper95Trial,
  } = computeDualStats();

  const resLabel = resMode === 1 ? 'HD' : 'SD';
  console.log(
    `\n  [Dual-Mode Paired | ${clip.file} | ${resLabel}] trials=${nTrials} (n=${nDiff}) | autoProbeOrActive=${totalAutoProbeOrActive}f | outputMismatch=${outputMismatchFrames}f | sample=${meanDiff.toFixed(5)}±${ci95Diff.toFixed(5)} ms (p50=${medianDiff.toFixed(5)} ms) | trialCI95=${trialMean.toFixed(5)}±${ci95Trial.toFixed(5)} ms (upper95=${upper95Trial.toFixed(5)} ms | trials=[${trialMeans.map((v) => v.toFixed(5)).join(',')}])`
  );

  offTrialTiming.sort((a, b) => a.warmSum - b.warmSum);
  autoTrialTiming.sort((a, b) => a.warmSum - b.warmSum);
  const medOff = offTrialTiming[offTrialTiming.length >> 1];
  const medAuto = autoTrialTiming[autoTrialTiming.length >> 1];
  accOff.sumTickMs = medOff.sum;
  accOff.warmSumTickMs = medOff.warmSum;
  accAuto.sumTickMs = medAuto.sum;
  accAuto.warmSumTickMs = medAuto.warmSum;

  const okOff = await finalizeAndValidateClip(accOff, sessOff, numFrames, resMode, 0, 'Off(0)', offTrialExtras);
  const autoExtras = [...autoTrialExtras];
  if (accAuto.swarmProbeOrActiveFrames > 0) {
    autoExtras.push(`falseSwarmProbeOrActiveFrames=${accAuto.swarmProbeOrActiveFrames}`);
  }
  if (outputMismatchFrames > 0) {
    autoExtras.push(`legacyOutputMismatchFrames=${outputMismatchFrames}`);
  }
  if (!noSpeedGate && (trialMean >= clipMaxOverheadMs || upper95Trial >= clipMaxOverheadMs || medianDiff >= clipMaxOverheadMs)) {
    autoExtras.push(
      `perClipAutoOverhead(trialMean=${trialMean.toFixed(5)}ms,trialUpper95=${upper95Trial.toFixed(5)}ms,p50=${medianDiff.toFixed(5)}ms)>=${clipMaxOverheadMs.toFixed(5)}ms`
    );
  }
  const okAuto = await finalizeAndValidateClip(accAuto, sessAuto, numFrames, resMode, 1, 'Auto(1)', autoExtras);

  return {
    passed: okOff && okAuto,
    totalResults: 2,
    passedResults: (okOff ? 1 : 0) + (okAuto ? 1 : 0),
    pairedWarmDiffs: pairedWarmDiffs.subarray(0, diffCount),
    trialMeans,
    meanDiff,
    ci95Diff,
    ci95Trial,
    upper95Trial,
  };
}

const modeTitle = dualModeFlag
  ? 'Paired Off(0)+Auto(1) Matrix'
  : explicitSwarmMode !== null
  ? `swarmMode=${explicitSwarmMode}`
  : useStarling
  ? 'Auto(1) Starling Swarm'
  : 'Legacy Default';
const resTitle = resModes.map((r) => (r === 1 ? '640x360 HD' : '320x180 SD')).join(' + ');

console.log('========================================================================');
console.log(` CHRONOS-SPECTRAL // Real-Camera MP4 Calibration Suite (${resTitle} | ${modeTitle})`);
console.log('========================================================================');

let allPassed = true;
let totalResults = 0;
let passedResults = 0;
const pooledPairedDiffs = [];
const pooledTrialMeans = [];

for (const resMode of resModes) {
  for (const clip of CLIPS) {
    if (dualModeFlag) {
      const res = await runPairedDualModeClip(clip, resMode);
      totalResults += res.totalResults;
      passedResults += res.passedResults;
      if (!res.passed) allPassed = false;
      for (const d of res.pairedWarmDiffs) pooledPairedDiffs.push(d);
      for (const tm of res.trialMeans) pooledTrialMeans.push(tm);
    } else {
      const mLabel = explicitSwarmMode !== null ? `Swarm(${explicitSwarmMode})` : useStarling ? 'Auto(1)' : 'Default';
      const res = await runSingleClip(clip, resMode, explicitSwarmMode, mLabel);
      totalResults += res.totalResults;
      passedResults += res.passedResults;
      if (!res.passed) allPassed = false;
    }
  }
}

if (dualModeFlag && pooledPairedDiffs.length > 0) {
  const nPool = pooledPairedDiffs.length;
  const pooledMeanDiff = pooledPairedDiffs.reduce((a, b) => a + b, 0) / nPool;
  let varSum = 0.0;
  for (let i = 0; i < nPool; i++) {
    const d = pooledPairedDiffs[i] - pooledMeanDiff;
    varSum += d * d;
  }
  const pooledCi95 = 1.96 * Math.sqrt(varSum / Math.max(1, nPool - 1) / nPool);
  const nT = pooledTrialMeans.length;
  const pooledTMean = nT > 0 ? pooledTrialMeans.reduce((a, b) => a + b, 0) / nT : pooledMeanDiff;
  let tVar = 0.0;
  for (let i = 0; i < nT; i++) {
    const d = pooledTrialMeans[i] - pooledTMean;
    tVar += d * d;
  }
  const pooledTrialCi95 = 2.0 * Math.sqrt(tVar / Math.max(1, nT - 1) / Math.max(1, nT));
  const pooledTrialUpper95 = pooledTMean + pooledTrialCi95;
  const pooledBudget = 0.005 * budgetScale;
  console.log(`\n------------------------------------------------------------------------`);
  console.log(
    `  [Dual-Mode Pooled Overhead] samples=${nPool} warm frames (trials=${nT}) | sample=${pooledMeanDiff.toFixed(5)}±${pooledCi95.toFixed(5)} ms | trial=${pooledTMean.toFixed(5)}±${pooledTrialCi95.toFixed(5)} ms (upper95=${pooledTrialUpper95.toFixed(5)} ms < ${pooledBudget.toFixed(5)} ms)`
  );
  if (strictGate && !noSpeedGate && (pooledMeanDiff >= pooledBudget || pooledTrialUpper95 >= pooledBudget)) {
    allPassed = false;
    console.error(
      `  [FAIL] Pooled Auto-inactive overhead (mean=${pooledMeanDiff.toFixed(5)} ms, trialUpper95=${pooledTrialUpper95.toFixed(5)} ms) >= ${pooledBudget.toFixed(5)} ms`
    );
  }
}

if (dualModeFlag && totalResults !== 52) {
  console.error(
    `  [NOTICE] Dual-mode matrix incomplete: got ${totalResults}/52 results (expected 13 legacy clips x 2 resolutions x 2 modes; pass --all --dual-res).`
  );
  if (strictGate) allPassed = false;
}

const isPartialRun = Boolean(clipFilter) || budgetScale > 1.0 || noSpeedGate || (dualModeFlag && totalResults !== 52);
const manifestTag = !allPassed && strictGate
  ? 'FAIL'
  : noSpeedGate
  ? 'UNCERTIFIED_NO_SPEED_GATE'
  : isPartialRun
  ? 'PARTIAL'
  : strictGate
  ? 'PASS'
  : 'DIAGNOSTIC';
console.log(
  `\n[Calibration Manifest] status=${manifestTag} | results=${passedResults}/${totalResults} | clips=${CLIPS.length} (${useStarling ? 'starling' : 'legacy'}) | res=[${resModes.map((r) => (r === 1 ? 'HD' : 'SD')).join(',')}]`
);

if (strictGate && !allPassed) {
  process.exit(1);
}
