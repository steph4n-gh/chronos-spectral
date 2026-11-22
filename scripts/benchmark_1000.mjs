#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate Q: Dense 1,000+ Target Synthetic Physics + Real
// Swarm Testbench & Ablation Benchmark (Node Wasm, Zero Coordinate Hints)
//
// Benchmarks CHRONOS-1000 across:
//   - Q1 (vortex_swarm_1200)  : 1,200 textured blobs in Lamb-Oseen dipole + images
//   - Q2 (murmuration_1200)   : 1,100 flocking boids + 3x36 rigid SE(2) polygons
//   - Q3 (modal_beam_1152)    : 48x24 fine speckles on 3.00 Hz anti-phase bar
//   - Q4 (starling_murmuration_1000.mp4) : 640x360 real starling murmuration video
//
// Usage:
//   node scripts/benchmark_1000.mjs [--gate] [--ablation] [--only=Q1,Q2,Q3,Q4]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const WASM_PATH = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');
const STARLING_MP4 = path.join(SAMPLES_DIR, 'starling_murmuration_1000.mp4');

const HD_W = 640;
const HD_H = 360;
const GRAPH_NODES = 32;
const MAX_TARGETS = 16;
const MAX_KRYLOV = 31;
const TELEMETRY_LEN = 1568;
const TELEM_RESERVED_TAIL_START = 1550;

// Locked Wave-2 / Wave-C ABI-v2 constants
const WASM_PAGE_BYTES = 65536;
const EXPECTED_WASM_PAGES = 448;
const EXPECTED_WASM_MEM_BYTES = EXPECTED_WASM_PAGES * WASM_PAGE_BYTES; // 29,360,128 B
const MAX_ARENA_BYTES = 29_100_000;
const WAVE1_MAX_STATIC_END = 29_294_592;
const FULL_RING_SLOT_BYTES = HD_W * HD_H * 4; // 921,600 B backing size even in SD
const MAX_MICRO = 3072;
const MICRO_STRIDE = 8; // [track_id, pos_x, pos_y, vel_x, vel_y, state, body_id, ftle_val]
const FIELD_COLS = 64;
const FIELD_ROWS = 36;
const FIELD_CELLS = FIELD_COLS * FIELD_ROWS; // 2304
const FIELD_STRIDE = 6; // [u_px_s, v_px_s, vorticity_s1, divergence_s1, ftle_s1, valid_weight]
const MAX_RIGID_BODIES = 32;
const BODY_STRIDE = 8; // [body_id, member_count, cx, cy, vx, vy, omega_rad_s, se2_rmse_px]
const SWARM_TELEM_LEN = 64;
const SWARM_RESERVED_TAIL_START = 42; // 0..15 SoA, 16..41 continuum (41 = modal R_2), 42..63 always 0.0

const REQUIRED_CHRONOS1000_EXPORTS = [
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

const VALID_ONLY_NAMES = new Set(['Q1', 'Q2', 'Q3', 'Q4', 'ANALYTIC', 'SWEEP', 'ABLATION']);
const ALLOWED_FLAGS = new Set([
  '--gate',
  '--ablation',
  '--no-speed-gate',
  '--self-test',
  '--analytic',
  '--sweep',
  '--help',
  '-h',
]);

const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(
    'Usage: node scripts/benchmark_1000.mjs [--gate] [--ablation] [--analytic] [--sweep] [--self-test] [--only=Q1,Q2,Q3,Q4,ANALYTIC,SWEEP,ABLATION] [--no-speed-gate]'
  );
  process.exit(0);
}

for (const arg of argv) {
  if (arg.startsWith('--only=')) continue;
  if (!ALLOWED_FLAGS.has(arg)) {
    console.error(`[FAIL] Unknown CLI flag: ${arg}`);
    process.exit(1);
  }
}

const strictGate = argv.includes('--gate');
const runAblation = argv.includes('--ablation');
const runAnalyticFlag = argv.includes('--analytic');
const runSweepFlag = argv.includes('--sweep');
const runSelfTestFlag = argv.includes('--self-test');
const noSpeedGate = argv.includes('--no-speed-gate');
let inAntiSpoofSelfTest = false;

if (strictGate && noSpeedGate) {
  console.error('[FAIL] --gate cannot be combined with --no-speed-gate.');
  process.exit(1);
}

const onlyArg = argv.find((a) => a.startsWith('--only='));
if (strictGate && onlyArg !== undefined) {
  console.error('[FAIL] --gate cannot be combined with --only= (full Gate Q certification requires all suites).');
  process.exit(1);
}
let onlySet = null;
if (onlyArg !== undefined) {
  const rawList = onlyArg.slice('--only='.length);
  const tokens = rawList
    .split(',')
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
  if (tokens.length === 0) {
    console.error('[FAIL] Empty --only= filter is not allowed.');
    process.exit(1);
  }
  for (const tok of tokens) {
    if (!VALID_ONLY_NAMES.has(tok)) {
      console.error(`[FAIL] Unknown suite in --only=${rawList}: ${tok} (valid: ${[...VALID_ONLY_NAMES].join(',')})`);
      process.exit(1);
    }
  }
  onlySet = new Set(tokens);
}

const rawBudgetScale = process.env.CHRONOS_CI_BUDGET_SCALE;
const budgetScale = rawBudgetScale === undefined ? 1.0 : Number(rawBudgetScale);
if (!Number.isFinite(budgetScale) || budgetScale < 1.0) {
  console.error(`[FAIL] Invalid CHRONOS_CI_BUDGET_SCALE=${rawBudgetScale} (must be finite >= 1.0).`);
  process.exit(1);
}
const SPEED_BUDGET_MS = 2.00 * budgetScale;
const TAIL_P95_BUDGET_MS = 2.50 * budgetScale;

function shouldRun(name) {
  return !onlySet || onlySet.has(name.toUpperCase());
}

function wrapAnglePi(rad) {
  let d = rad;
  while (d > Math.PI) d -= 2.0 * Math.PI;
  while (d < -Math.PI) d += 2.0 * Math.PI;
  return d;
}

function computePercentile(values, pct) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((pct / 100.0) * (sorted.length - 1))));
  return sorted[idx];
}

/** Formats continuum diagnostic validity bitmask (bits 0..4) into human-readable tags. */
function decodeValidityMask(maskInt) {
  const names = [
    [1, 'GRAD'],
    [2, 'FTLE'],
    [4, 'RIGID'],
    [8, 'MODAL'],
    [16, 'KURAMOTO'],
  ];
  const active = names.filter(([bit]) => (maskInt & bit) !== 0).map(([, tag]) => tag);
  return `0x${(maskInt >>> 0).toString(16).padStart(2, '0')} [${active.length ? active.join('|') : 'NONE'}]`;
}

function computeMedian(sortedOrRaw) {
  if (sortedOrRaw.length === 0) return 0.0;
  const s = [...sortedOrRaw].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return (s.length & 1) === 1 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

function pearsonR(xs, ys) {
  const n = Math.min(xs.length, ys.length);
  if (n < 3) return 0.0;
  let sx = 0.0;
  let sy = 0.0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxy = 0.0;
  let sxx = 0.0;
  let syy = 0.0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  const denom = Math.sqrt(sxx * syy);
  return denom > 1e-12 ? sxy / denom : 0.0;
}

/** Delegates directly to the in-kernel Wasm Shifted Lanczos solver (`src/lanczos.rs`). */
function createLanczosSolver(exp) {
  return () => exp.chronos_solve_fiedler_lanczos();
}

/** Exact O(n^3) Jonker-Volgenant / Hungarian min-cost bipartite assignment on square n x n matrix. */
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
    if (p[j] >= 1 && p[j] <= n) rowToCol[p[j] - 1] = j - 1;
  }
  return rowToCol;
}

/**
 * Continuity-aware exact bipartite matcher optimized for N=1,200+ via spatial
 * bucketing and connected-component decomposition before Hungarian solve.
 * When forceMaxCard is true, absent edges get a huge sentinel penalty (CLEAR-MOT);
 * when false, absent/dummy cells have weight 0 (TrackEval HOTA max sum weight).
 */
function extractConnectedMatchesMaxWeight(gtsLen, tracksLen, adjG, minSim, weightFn, forceMaxCard = true) {
  const gAdj = Array.from({ length: gtsLen }, () => []);
  const tAdj = Array.from({ length: tracksLen }, () => []);
  for (let gi = 0; gi < gtsLen; gi++) {
    const edges = adjG[gi];
    for (let k = 0; k < edges.length; k++) {
      const e = edges[k];
      if (e.sim >= minSim) {
        const w = weightFn(gi, e);
        if (w > 0.0) {
          gAdj[gi].push({ ti: e.ti, d: e.d, sim: e.sim, w });
          tAdj[e.ti].push({ gi, d: e.d, sim: e.sim, w });
        }
      }
    }
  }

  const visG = new Uint8Array(gtsLen);
  const visT = new Uint8Array(tracksLen);
  const matches = [];
  const qG = [];

  for (let startGi = 0; startGi < gtsLen; startGi++) {
    if (visG[startGi] || gAdj[startGi].length === 0) continue;
    const compG = [];
    const compT = [];
    qG.length = 0;
    qG.push(startGi);
    visG[startGi] = 1;

    while (qG.length > 0) {
      const gCur = qG.pop();
      compG.push(gCur);
      const edges = gAdj[gCur];
      for (let k = 0; k < edges.length; k++) {
        const ti = edges[k].ti;
        if (!visT[ti]) {
          visT[ti] = 1;
          compT.push(ti);
          const back = tAdj[ti];
          for (let m = 0; m < back.length; m++) {
            const gNext = back[m].gi;
            if (!visG[gNext]) {
              visG[gNext] = 1;
              qG.push(gNext);
            }
          }
        }
      }
    }

    if (compG.length === 1) {
      let best = null;
      for (const e of gAdj[compG[0]]) {
        if (!best || e.w > best.w) best = e;
      }
      if (best) matches.push({ gi: compG[0], ti: best.ti, d: best.d, sim: best.sim });
    } else if (compT.length === 1) {
      let best = null;
      for (const e of tAdj[compT[0]]) {
        if (!best || e.w > best.w) best = e;
      }
      if (best) matches.push({ gi: best.gi, ti: compT[0], d: best.d, sim: best.sim });
    } else {
      const nSub = Math.max(compG.length, compT.length);
      const baseZero = forceMaxCard ? (Math.min(compG.length, compT.length) + 1) * 300.0 : 2.0;
      const cost = new Float64Array(nSub * nSub);
      cost.fill(baseZero);
      const tIdxMap = new Map();
      for (let j = 0; j < compT.length; j++) tIdxMap.set(compT[j], j);
      const cellEdge = new Map();
      for (let i = 0; i < compG.length; i++) {
        for (const e of gAdj[compG[i]]) {
          const j = tIdxMap.get(e.ti);
          cost[i * nSub + j] = (forceMaxCard ? 1.0 : baseZero) - e.w;
          cellEdge.set(i * nSub + j, e);
        }
      }
      const rowToCol = solveHungarianMinCost(cost, nSub);
      for (let i = 0; i < compG.length; i++) {
        const j = rowToCol[i];
        if (j >= 0 && j < compT.length) {
          const e = cellEdge.get(i * nSub + j);
          if (e) matches.push({ gi: compG[i], ti: compT[j], d: e.d, sim: e.sim });
        }
      }
    }
  }
  return matches;
}

function extractConnectedMatchesMaxSim(gtsLen, tracksLen, adjG, minSim, costMax = 1.0) {
  return extractConnectedMatchesMaxWeight(gtsLen, tracksLen, adjG, minSim, (_gi, e) => e.sim / Math.max(1e-9, costMax), true);
}

/**
 * Continuity-aware exact bipartite CLEAR-MOT + 19-threshold TrackEval HOTA evaluator
 * (point localization similarity S(g,t) = max(0, 1 - d / (2 * matchRadiusPx))).
 */
function createClearMotEvaluator(matchRadiusPx = 4.5) {
  const simMaxDist = 2.0 * matchRadiusPx;
  const prevFrameGtToTid = new Map();
  const lastTidForGt = new Map();
  const gtSeenTracked = new Set();
  const gtCurrentlyLost = new Set();
  const gtTotalCounts = new Map();
  const tidTotalCounts = new Map();
  const evalFrames = [];

  let gtTotal = 0;
  let gtMatched = 0;
  let gtPredTotal = 0;
  let gtUnmatchedFp = 0;
  let gtSumSqErr = 0.0;
  let gtSumDist = 0.0;
  let gtIdSwaps = 0;
  let gtFrags = 0;
  let duplicateTidCount = 0;
  let nonFiniteTrackCount = 0;

  const cellSize = Math.max(8.0, simMaxDist);
  const gridCols = Math.ceil(HD_W / cellSize) + 2;
  const gridRows = Math.ceil(HD_H / cellSize) + 2;
  const gridBuckets = Array.from({ length: gridCols * gridRows }, () => []);

  function update(gts, tracks) {
    // Reject duplicate track IDs & non-finite coordinates per frame
    const cleanTracks = [];
    const seenTids = new Set();
    for (let ti = 0; ti < tracks.length; ti++) {
      const tr = tracks[ti];
      if (!Number.isInteger(tr.tid) || tr.tid <= 0 || !Number.isFinite(tr.x) || !Number.isFinite(tr.y)) {
        nonFiniteTrackCount++;
        continue;
      }
      if (seenTids.has(tr.tid)) {
        duplicateTidCount++;
        continue;
      }
      seenTids.add(tr.tid);
      cleanTracks.push(tr);
    }

    gtTotal += gts.length;
    gtPredTotal += cleanTracks.length;
    for (let gi = 0; gi < gts.length; gi++) {
      const gid = gts[gi].gtId;
      gtTotalCounts.set(gid, (gtTotalCounts.get(gid) || 0) + 1);
    }
    for (let ti = 0; ti < cleanTracks.length; ti++) {
      const tid = cleanTracks[ti].tid;
      tidTotalCounts.set(tid, (tidTotalCounts.get(tid) || 0) + 1);
    }

    const usedG = new Uint8Array(gts.length);
    const usedT = new Uint8Array(cleanTracks.length);
    const tidToTi = new Map();
    for (let ti = 0; ti < cleanTracks.length; ti++) tidToTi.set(cleanTracks[ti].tid, ti);

    // Precompute spatial neighbor lists within simMaxDist for both CLEAR-MOT and HOTA
    for (let b = 0; b < gridBuckets.length; b++) gridBuckets[b].length = 0;
    for (let ti = 0; ti < cleanTracks.length; ti++) {
      const cx = Math.max(0, Math.min(gridCols - 1, Math.floor(cleanTracks[ti].x / cellSize)));
      const cy = Math.max(0, Math.min(gridRows - 1, Math.floor(cleanTracks[ti].y / cellSize)));
      gridBuckets[cy * gridCols + cx].push(ti);
    }

    const fullAdjG = Array.from({ length: gts.length }, () => []);
    for (let gi = 0; gi < gts.length; gi++) {
      const gx = gts[gi].x;
      const gy = gts[gi].y;
      const cx0 = Math.max(0, Math.floor((gx - simMaxDist) / cellSize));
      const cx1 = Math.min(gridCols - 1, Math.floor((gx + simMaxDist) / cellSize));
      const cy0 = Math.max(0, Math.floor((gy - simMaxDist) / cellSize));
      const cy1 = Math.min(gridRows - 1, Math.floor((gy + simMaxDist) / cellSize));
      for (let cy = cy0; cy <= cy1; cy++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const bucket = gridBuckets[cy * gridCols + cx];
          for (let k = 0; k < bucket.length; k++) {
            const ti = bucket[k];
            const d = Math.hypot(cleanTracks[ti].x - gx, cleanTracks[ti].y - gy);
            if (d < simMaxDist) {
              fullAdjG[gi].push({ ti, d, sim: Math.max(0.0, 1.0 - d / simMaxDist) });
            }
          }
        }
      }
    }

    evalFrames.push({
      gtIds: gts.map((g) => g.gtId),
      tids: cleanTracks.map((t) => t.tid),
      adjG: fullAdjG,
    });

    // CLEAR-MOT (a) continuity carryover within matchRadiusPx
    const matches = [];
    for (let gi = 0; gi < gts.length; gi++) {
      const prevTid = prevFrameGtToTid.get(gts[gi].gtId);
      if (prevTid === undefined) continue;
      const ti = tidToTi.get(prevTid);
      if (ti === undefined || usedT[ti]) continue;
      const d = Math.hypot(cleanTracks[ti].x - gts[gi].x, cleanTracks[ti].y - gts[gi].y);
      if (d <= matchRadiusPx) {
        usedG[gi] = 1;
        usedT[ti] = 1;
        matches.push({ gi, ti, d });
      }
    }

    // CLEAR-MOT (b) exact connected-component minimum-distance Hungarian over remaining pairs <= matchRadiusPx
    const remAdjG = Array.from({ length: gts.length }, () => []);
    for (let gi = 0; gi < gts.length; gi++) {
      if (usedG[gi]) continue;
      for (const e of fullAdjG[gi]) {
        if (!usedT[e.ti] && e.d <= matchRadiusPx) {
          remAdjG[gi].push({ ti: e.ti, d: e.d, sim: 1.0 - e.d / simMaxDist });
        }
      }
    }
    const remMatches = extractConnectedMatchesMaxSim(gts.length, cleanTracks.length, remAdjG, 0.5, 1.0);
    for (const m of remMatches) matches.push({ gi: m.gi, ti: m.ti, d: m.d });

    const matchedGiSet = new Set();
    prevFrameGtToTid.clear();
    gtMatched += matches.length;
    gtUnmatchedFp += Math.max(0, cleanTracks.length - matches.length);

    for (const m of matches) {
      matchedGiSet.add(m.gi);
      const gtId = gts[m.gi].gtId;
      const tid = cleanTracks[m.ti].tid;
      gtSumSqErr += m.d * m.d;
      gtSumDist += m.d;
      prevFrameGtToTid.set(gtId, tid);

      const prevTid = lastTidForGt.get(gtId);
      if (prevTid !== undefined && prevTid !== tid) gtIdSwaps++;
      if (gtCurrentlyLost.has(gtId)) {
        gtFrags++;
        gtCurrentlyLost.delete(gtId);
      }
      gtSeenTracked.add(gtId);
      lastTidForGt.set(gtId, tid);
    }

    for (let gi = 0; gi < gts.length; gi++) {
      if (!matchedGiSet.has(gi) && gtSeenTracked.has(gts[gi].gtId)) {
        gtCurrentlyLost.add(gts[gi].gtId);
      }
    }
  }

  function computeTrackEvalHota() {
    if (gtTotal === 0) return null;
    if (gtPredTotal === 0) return { hota: 0.0, detA: 0.0, assA: 0.0, locA: 0.0 };

    // Stage 1: TrackEval sim_iou potential matches across all frames
    const potentialMatches = new Map();
    for (let f = 0; f < evalFrames.length; f++) {
      const ef = evalFrames[f];
      const nG = ef.gtIds.length;
      const nT = ef.tids.length;
      if (nG === 0 || nT === 0) continue;
      const rSum = new Float64Array(nG);
      const cSum = new Float64Array(nT);
      for (let gi = 0; gi < nG; gi++) {
        for (const e of ef.adjG[gi]) {
          if (e.sim > 0.0) {
            rSum[gi] += e.sim;
            cSum[e.ti] += e.sim;
          }
        }
      }
      for (let gi = 0; gi < nG; gi++) {
        const gid = ef.gtIds[gi];
        for (const e of ef.adjG[gi]) {
          if (e.sim <= 0.0) continue;
          const denom = rSum[gi] + cSum[e.ti] - e.sim;
          const simIou = denom > 0.0 ? e.sim / denom : 0.0;
          if (simIou > 0.0) {
            const key = `${gid}:${ef.tids[e.ti]}`;
            potentialMatches.set(key, (potentialMatches.get(key) || 0.0) + simIou);
          }
        }
      }
    }

    const globalAlign = new Map();
    for (const [key, pot] of potentialMatches.entries()) {
      const colon = key.indexOf(':');
      const gid = Number(key.slice(0, colon));
      const tid = Number(key.slice(colon + 1));
      const gTot = gtTotalCounts.get(gid) || 0;
      const tTot = tidTotalCounts.get(tid) || 0;
      const denom = gTot + tTot - pot;
      globalAlign.set(key, denom > 0.0 ? pot / denom : 0.0);
    }

    // Stage 2: Thresholded assignment per alpha maximizing sum(globalAlign * sim)
    const ALPHAS = 19;
    let sumHota = 0.0;
    let sumDetA = 0.0;
    let sumAssA = 0.0;
    let sumLocA = 0.0;
    let locCount = 0;

    for (let aIdx = 1; aIdx <= ALPHAS; aIdx++) {
      const alpha = aIdx * 0.05;
      let tp = 0;
      let simSum = 0.0;
      const pairCounts = new Map();

      for (let f = 0; f < evalFrames.length; f++) {
        const ef = evalFrames[f];
        const mList = extractConnectedMatchesMaxWeight(
          ef.gtIds.length,
          ef.tids.length,
          ef.adjG,
          alpha - 1e-12,
          (gi, e) => (globalAlign.get(`${ef.gtIds[gi]}:${ef.tids[e.ti]}`) || 0.0) * e.sim,
          false
        );
        tp += mList.length;
        for (let k = 0; k < mList.length; k++) {
          const m = mList[k];
          simSum += m.sim;
          const gid = ef.gtIds[m.gi];
          const tid = ef.tids[m.ti];
          let inner = pairCounts.get(gid);
          if (!inner) {
            inner = new Map();
            pairCounts.set(gid, inner);
          }
          inner.set(tid, (inner.get(tid) || 0) + 1);
        }
      }

      const fn = gtTotal - tp;
      const fp = gtPredTotal - tp;
      const detDenom = tp + fn + fp;
      const detA = detDenom > 0 ? tp / detDenom : 0.0;
      let assASum = 0.0;
      if (tp > 0) {
        for (const [gid, inner] of pairCounts.entries()) {
          const gTot = gtTotalCounts.get(gid) || 0;
          for (const [tid, tpa] of inner.entries()) {
            const tTot = tidTotalCounts.get(tid) || 0;
            const denom = gTot + tTot - tpa;
            if (denom > 0) assASum += tpa * (tpa / denom);
          }
        }
      }
      const assA = tp > 0 ? assASum / tp : 0.0;
      const hota = Math.sqrt(detA * assA);
      sumDetA += detA;
      sumAssA += assA;
      sumHota += hota;
      if (tp > 0) {
        sumLocA += simSum / tp;
        locCount++;
      }
    }

    return {
      hota: sumHota / ALPHAS,
      detA: sumDetA / ALPHAS,
      assA: sumAssA / ALPHAS,
      locA: locCount > 0 ? sumLocA / locCount : 0.0,
    };
  }

  function summary() {
    const fn = gtTotal - gtMatched;
    const recallPct = gtTotal > 0 ? (100.0 * gtMatched) / gtTotal : null;
    const precisionPct = gtPredTotal > 0 ? (100.0 * gtMatched) / gtPredTotal : null;
    const motaPct = gtTotal > 0 ? (1.0 - (fn + gtUnmatchedFp + gtIdSwaps) / gtTotal) * 100.0 : null;
    const rmsePx = gtMatched > 0 ? Math.sqrt(gtSumSqErr / gtMatched) : null;
    const motpPx = gtMatched > 0 ? gtSumDist / gtMatched : null;
    const hotaRes = computeTrackEvalHota();
    return {
      gtTotal,
      gtPredTotal,
      gtMatched,
      fp: gtUnmatchedFp,
      recallPct,
      precisionPct,
      motaPct,
      rmsePx,
      motpPx,
      idsw: gtIdSwaps,
      frag: gtFrags,
      detA: hotaRes ? hotaRes.detA : null,
      assA: hotaRes ? hotaRes.assA : null,
      locA: hotaRes ? hotaRes.locA : null,
      hota: hotaRes ? hotaRes.hota : null,
      duplicateTidCount,
      nonFiniteTrackCount,
    };
  }

  return { update, summary };
}

/** Renders crisp high-contrast subpixel-centered 4x4 textured particle blobs onto dark background. */
function clearFrameDarkBg(rgba) {
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = 10;
    rgba[i + 1] = 14;
    rgba[i + 2] = 22;
    rgba[i + 3] = 255;
  }
}

function stampParticleBlob(rgba, w, h, cx, cy, seedId, isHeroDrifter = false) {
  const sigma = isHeroDrifter ? 2.6 : 0.85;
  const rMax = isHeroDrifter ? 5.5 : 2.2;
  const r2Max = rMax * rMax;
  const rad = Math.ceil(rMax);
  const x0 = Math.max(0, Math.floor(cx) - rad);
  const x1 = Math.min(w - 1, Math.floor(cx) + rad);
  const y0 = Math.max(0, Math.floor(cy) - rad);
  const y1 = Math.min(h - 1, Math.floor(cy) + rad);

  // Subtle deterministic chromatic + luma diversity matching organic multi-target particles
  const h1 = Math.abs(Math.sin(seedId * 12.9898 + 0.5) * 43758.5453) % 1.0;
  const h2 = Math.abs(Math.sin(seedId * 78.233 + 1.9) * 43758.5453) % 1.0;
  const peak = isHeroDrifter ? 236.0 : 212.0 + h1 * 38.0;
  const tintR = isHeroDrifter ? 1.0 : 0.88 + 0.22 * h2;
  const tintG = isHeroDrifter ? 0.98 : 0.94 + 0.12 * (1.0 - h1);
  const tintB = isHeroDrifter ? 0.94 : 1.05 - 0.18 * h2;
  const denom = 2.0 * sigma * sigma;

  for (let py = y0; py <= y1; py++) {
    const dy = py - cy;
    for (let px = x0; px <= x1; px++) {
      const dx = px - cx;
      const r2 = dx * dx + dy * dy;
      if (r2 > r2Max) continue;
      const g = Math.exp(-r2 / denom);
      const idx = (py * w + px) * 4;
      const r = Math.min(255, Math.round(10 + peak * tintR * g));
      const gr = Math.min(255, Math.round(14 + peak * tintG * g));
      const b = Math.min(255, Math.round(22 + peak * tintB * g));
      if (r > rgba[idx]) rgba[idx] = r;
      if (gr > rgba[idx + 1]) rgba[idx + 1] = gr;
      if (b > rgba[idx + 2]) rgba[idx + 2] = b;
    }
  }
}

// ============================================================================
// Scene Generators (Matched strictly with web/worker.js parameters)
// ============================================================================

/** Q1: Counter-rotating Lamb-Oseen vortex pair + 3x3 mirror images (zero divergence). */
function createQ1VortexScene() {
  const fps = 30;
  const dtSec = 1.0 / fps;
  const numFrames = 90;
  const rc = 48.0;
  const K = 4600.0;
  const xL = 14.0;
  const xR = 626.0;
  const yT = 14.0;
  const yB = 346.0;
  const bases = [
    [210.0, 180.0, K],
    [430.0, 180.0, -K],
  ];
  const vortices = [];
  for (const mx of [-1, 0, 1]) {
    for (const my of [-1, 0, 1]) {
      const sign = (mx !== 0 ? -1 : 1) * (my !== 0 ? -1 : 1);
      for (const [bx, by, bk] of bases) {
        const vx = mx === 0 ? bx : mx === -1 ? 2.0 * xL - bx : 2.0 * xR - bx;
        const vy = my === 0 ? by : my === -1 ? 2.0 * yT - by : 2.0 * yB - by;
        vortices.push({ cx: vx, cy: vy, k: bk * sign });
      }
    }
  }

  function velAt(x, y) {
    let u = 0.0;
    let v = 0.0;
    for (let i = 0; i < vortices.length; i++) {
      const dx = x - vortices[i].cx;
      const dy = y - vortices[i].cy;
      const r2 = dx * dx + dy * dy;
      if (r2 > 1e-6) {
        const f = (vortices[i].k / r2) * (1.0 - Math.exp(-r2 / (rc * rc)));
        u += -dy * f;
        v += dx * f;
      }
    }
    return [u, v];
  }

  function analyticVorticityAt(x, y) {
    let w = 0.0;
    const invRc2 = 1.0 / (rc * rc);
    for (let i = 0; i < vortices.length; i++) {
      const dx = x - vortices[i].cx;
      const dy = y - vortices[i].cy;
      const r2 = dx * dx + dy * dy;
      w += 2.0 * vortices[i].k * invRc2 * Math.exp(-r2 * invRc2);
    }
    return w;
  }

  const pts = [];
  const nx = 48;
  const ny = 25;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i;
      const h1 = Math.sin(idx * 12.9898 + 1.23) * 43758.5453;
      const h2 = Math.sin(idx * 78.233 + 4.56) * 43758.5453;
      const jx = (h1 - Math.floor(h1) - 0.5) * 3.0;
      const jy = (h2 - Math.floor(h2) - 0.5) * 3.0;
      const px = 22.0 + ((i + 0.5) * (618.0 - 22.0)) / nx + jx;
      const py = 20.0 + ((j + 0.5) * (340.0 - 20.0)) / ny + jy;
      const heroDrifter = j === 12 && (i === 18 || i === 29);
      pts.push({ gtId: idx + 1, x: px, y: py, heroDrifter });
    }
  }

  const gtFrames = [];
  for (let f = 0; f < numFrames; f++) {
    gtFrames.push(pts.map((p) => ({ gtId: p.gtId, x: p.x, y: p.y, heroDrifter: p.heroDrifter })));
    if (f + 1 < numFrames) {
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const [u1, v1] = velAt(p.x, p.y);
        const [u2, v2] = velAt(p.x + 0.5 * dtSec * u1, p.y + 0.5 * dtSec * v1);
        const [u3, v3] = velAt(p.x + 0.5 * dtSec * u2, p.y + 0.5 * dtSec * v2);
        const [u4, v4] = velAt(p.x + dtSec * u3, p.y + dtSec * v3);
        p.x += ((u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0) * dtSec;
        p.y += ((v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0) * dtSec;
      }
    }
  }

  return { fps, dtSec, numFrames, gtFrames, velAt, analyticVorticityAt };
}

/** Q2: 3 rotating 36-marker rigid polygons + 1,100 murmuration boids (1,208 targets). */
function createQ2MurmurationScene() {
  const fps = 30;
  const dtSec = 1.0 / fps;
  const numFrames = 90;

  const bodies = [
    { bodyIdx: 0, cx0: 140.0, cy0: 135.0, vx: 16.0, vy: 8.0, omega: 0.90 },
    { bodyIdx: 1, cx0: 330.0, cy0: 235.0, vx: 8.0, vy: -10.0, omega: -0.70 },
    { bodyIdx: 2, cx0: 505.0, cy0: 135.0, vx: -14.0, vy: 12.0, omega: 0.60 },
  ];

  const polyOffsets = [];
  for (let k = 0; k < 24; k++) {
    const a = (2.0 * Math.PI * k) / 24.0;
    const r = 42.0 + (k % 2 === 0 ? 2.0 : -2.0);
    polyOffsets.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  for (let k = 0; k < 12; k++) {
    const a = (2.0 * Math.PI * k) / 12.0 + Math.PI / 12.0;
    polyOffsets.push([22.0 * Math.cos(a), 22.0 * Math.sin(a)]);
  }

  const candidates = [];
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
      for (const b of bodies) {
        if (Math.hypot(px - b.cx0, py - b.cy0) < 58.0) {
          ok = false;
          break;
        }
      }
      if (ok) candidates.push({ x: px, y: py, vx: 32.0, vy: 6.0 * Math.sin(0.02 * px) });
    }
  }

  const boids = [];
  const step = candidates.length / 1100.0;
  for (let m = 0; m < 1100; m++) {
    const src = candidates[Math.floor(m * step)];
    boids.push({ gtId: 200 + m, x: src.x, y: src.y, vx: src.vx, vy: src.vy });
  }

  const grid = Array.from({ length: 60 * 35 }, () => []);
  const gtFrames = [];
  const bodyGtFrames = [];

  for (let f = 0; f < numFrames; f++) {
    const t = f * dtSec;
    const framePts = [];
    const frameBodies = [];

    for (const b of bodies) {
      const cx = b.cx0 + b.vx * t;
      const cy = b.cy0 + b.vy * t;
      const th = b.omega * t;
      const ct = Math.cos(th);
      const st = Math.sin(th);
      frameBodies.push({ bodyIdx: b.bodyIdx, cx, cy, vx: b.vx, vy: b.vy, omega: b.omega });
      for (let k = 0; k < polyOffsets.length; k++) {
        const [lx, ly] = polyOffsets[k];
        framePts.push({
          gtId: 1 + b.bodyIdx * 40 + k,
          x: cx + lx * ct - ly * st,
          y: cy + lx * st + ly * ct,
          bodyIdx: b.bodyIdx,
        });
      }
    }
    for (let i = 0; i < boids.length; i++) {
      framePts.push({ gtId: boids[i].gtId, x: boids[i].x, y: boids[i].y, bodyIdx: -1 });
    }
    gtFrames.push(framePts);
    bodyGtFrames.push(frameBodies);

    if (f + 1 < numFrames) {
      for (let c = 0; c < grid.length; c++) grid[c].length = 0;
      for (let i = 0; i < boids.length; i++) {
        const gx = Math.max(0, Math.min(59, Math.floor(boids[i].x / 12.0)));
        const gy = Math.max(0, Math.min(34, Math.floor(boids[i].y / 12.0)));
        grid[gy * 60 + gx].push(i);
      }

      const nextBoids = new Array(boids.length);
      for (let i = 0; i < boids.length; i++) {
        const bi = boids[i];
        let ux = 36.0 * Math.cos(0.45 * t) + 18.0 * Math.sin(0.012 * bi.y + 1.2 * t);
        let uy = 22.0 * Math.sin(0.65 * t) + 14.0 * Math.cos(0.009 * bi.x - 0.9 * t);

        if (bi.x < 40.0) ux += (40.0 - bi.x) * 2.5;
        if (bi.x > 600.0) ux -= (bi.x - 600.0) * 2.5;
        if (bi.y < 36.0) uy += (36.0 - bi.y) * 2.5;
        if (bi.y > 324.0) uy -= (bi.y - 324.0) * 2.5;

        for (const fb of frameBodies) {
          const dx = bi.x - fb.cx;
          const dy = bi.y - fb.cy;
          const d = Math.hypot(dx, dy);
          if (d < 58.0 && d > 1e-3) {
            const push = (58.0 - d) * 4.5;
            ux += (dx / d) * push;
            uy += (dy / d) * push;
          }
        }

        const gx = Math.max(0, Math.min(59, Math.floor(bi.x / 12.0)));
        const gy = Math.max(0, Math.min(34, Math.floor(bi.y / 12.0)));
        for (let oy = -1; oy <= 1; oy++) {
          const yy = gy + oy;
          if (yy < 0 || yy >= 35) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const xx = gx + ox;
            if (xx < 0 || xx >= 60) continue;
            const bucket = grid[yy * 60 + xx];
            for (let m = 0; m < bucket.length; m++) {
              const j = bucket[m];
              if (j === i) continue;
              const dx = bi.x - boids[j].x;
              const dy = bi.y - boids[j].y;
              const d2 = dx * dx + dy * dy;
              if (d2 < 90.25 && d2 > 1e-4) {
                const d = Math.sqrt(d2);
                const push = (9.5 - d) * 6.0;
                ux += (dx / d) * push;
                uy += (dy / d) * push;
              }
            }
          }
        }

        let nvx = bi.vx * 0.65 + ux * 0.35;
        let nvy = bi.vy * 0.65 + uy * 0.35;
        const spd = Math.hypot(nvx, nvy);
        if (spd > 75.0) {
          nvx *= 75.0 / spd;
          nvy *= 75.0 / spd;
        }
        let nbx = Math.min(624.0, Math.max(16.0, bi.x + nvx * dtSec));
        let nby = Math.min(344.0, Math.max(16.0, bi.y + nvy * dtSec));
        for (const fb of frameBodies) {
          const cxNext = fb.cx + fb.vx * dtSec;
          const cyNext = fb.cy + fb.vy * dtSec;
          const dx = nbx - cxNext;
          const dy = nby - cyNext;
          const d = Math.hypot(dx, dy);
          if (d < 50.0 && d > 1e-3) {
            nbx = cxNext + (dx / d) * 50.0;
            nby = cyNext + (dy / d) * 50.0;
          }
        }
        nextBoids[i] = { gtId: bi.gtId, x: nbx, y: nby, vx: nvx, vy: nvy };
      }
      for (let i = 0; i < boids.length; i++) boids[i] = nextBoids[i];
    }
  }

  return { fps, dtSec, numFrames, bodies, gtFrames, bodyGtFrames };
}

/** Q3: 48x24 = 1,152 fine speckles on 3.00 Hz bar with central node / anti-phase halves. */
function createQ3ModalBeamScene() {
  const fps = 30;
  const dtSec = 1.0 / fps;
  const numFrames = 120;
  const fTrueHz = 3.00;
  const ampPx = 1.25;

  const basePts = [];
  const nx = 48;
  const ny = 24;
  for (let j = 0; j < ny; j++) {
    const y0 = 72.0 + (j * (288.0 - 72.0)) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const x0 = 44.0 + (i * (596.0 - 44.0)) / (nx - 1);
      const u = (x0 - 320.0) / 276.0;
      const modeShape = Math.sin(0.5 * Math.PI * u);
      basePts.push({ gtId: j * nx + i + 1, x0, y0, modeShape });
    }
  }

  const gtFrames = [];
  for (let f = 0; f < numFrames; f++) {
    const phase = Math.sin(2.0 * Math.PI * fTrueHz * f * dtSec);
    gtFrames.push(
      basePts.map((p) => ({
        gtId: p.gtId,
        x: p.x0,
        y: p.y0 + ampPx * p.modeShape * phase,
      }))
    );
  }

  return { fps, dtSec, numFrames, fTrueHz, ampPx, gtFrames };
}

// ============================================================================
// Pure-JS Ablation Trackers for Q2 (`Nearest-Greedy` vs `Morton-CSR+Auction`)
// ============================================================================

function computeMorton10(xi, yi) {
  let x = xi & 0x3ff;
  let y = yi & 0x3ff;
  x = (x | (x << 8)) & 0x00ff00ff;
  x = (x | (x << 4)) & 0x0f0f0f0f;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;
  y = (y | (y << 8)) & 0x00ff00ff;
  y = (y | (y << 4)) & 0x0f0f0f0f;
  y = (y | (y << 2)) & 0x33333333;
  y = (y | (y << 1)) & 0x55555555;
  return (y << 1) | x;
}

/**
 * Synthesizes realistic detector observations (0.18px jitter, 3% dropout,
 * 25 transient clutter peaks + 24 quasi-static background distractors, shuffled).
 */
function buildNoisyObservations(gtPts, frameIdx) {
  const obs = [];
  for (let i = 0; i < gtPts.length; i++) {
    const hDrop = (Math.sin((frameIdx * 131 + i) * 41.23) * 43758.5453) % 1.0;
    if (Math.abs(hDrop) < 0.03) continue;
    const hx = (Math.sin((frameIdx * 97 + i) * 19.37) * 43758.5453) % 1.0;
    const hy = (Math.sin((frameIdx * 53 + i) * 73.11) * 43758.5453) % 1.0;
    obs.push({
      x: gtPts[i].x + (hx - Math.sign(hx) * 0.5) * 0.36,
      y: gtPts[i].y + (hy - Math.sign(hy) * 0.5) * 0.36,
    });
  }
  for (let c = 0; c < 25; c++) {
    const rx = Math.abs(Math.sin((frameIdx * 71 + c) * 13.71) * 43758.5453) % 1.0;
    const ry = Math.abs(Math.sin((frameIdx * 89 + c) * 29.53) * 43758.5453) % 1.0;
    obs.push({ x: 20.0 + rx * 600.0, y: 20.0 + ry * 320.0 });
  }
  // Quasi-stationary clutter peaks (specular glints / background texture corners)
  for (let s = 0; s < 24; s++) {
    const sx = 34.0 + ((s * 97) % 572);
    const sy = 28.0 + ((s * 59) % 304);
    const jx = ((Math.sin((frameIdx * 11 + s) * 17.3) * 43758.5) % 1.0) * 0.14;
    const jy = ((Math.sin((frameIdx * 19 + s) * 31.7) * 43758.5) % 1.0) * 0.14;
    obs.push({ x: sx + jx, y: sy + jy });
  }
  // Deterministic Fisher-Yates shuffle
  for (let i = obs.length - 1; i > 0; i--) {
    const u = Math.abs(Math.sin((frameIdx * 101 + i) * 17.89) * 43758.5453) % 1.0;
    const j = Math.floor(u * (i + 1));
    const tmp = obs[i];
    obs[i] = obs[j];
    obs[j] = tmp;
  }
  return obs;
}

function solveAuctionAsym(edgeTo, edgeBen, edgeCnt, nT, nO, csrK) {
  if (nT === 0 || nO === 0) {
    return { trackToObs: new Int32Array(nT).fill(-1), capHits: 0, roundLimitHits: 0 };
  }
  const K = Math.min(nT, nO) + 2;
  const revCounts = new Int32Array(nO);
  let maxB = 0;
  for (let ti = 0; ti < nT; ti++) {
    const cnt = edgeCnt[ti];
    for (let e = 0; e < cnt; e++) {
      const idx = ti * csrK + e;
      const b = edgeBen[idx];
      if (b > 0) {
        revCounts[edgeTo[idx]]++;
        if (b > maxB) maxB = b;
      }
    }
  }
  if (maxB <= 0) {
    return { trackToObs: new Int32Array(nT).fill(-1), capHits: 0, roundLimitHits: 0 };
  }

  const revOff = new Int32Array(nO + 1);
  for (let oi = 0; oi < nO; oi++) revOff[oi + 1] = revOff[oi] + revCounts[oi];
  const revTi = new Int32Array(revOff[nO]);
  const revBen = new Float64Array(revOff[nO]);
  const revPos = new Int32Array(nO);
  for (let ti = 0; ti < nT; ti++) {
    const cnt = edgeCnt[ti];
    for (let e = 0; e < cnt; e++) {
      const idx = ti * csrK + e;
      const b = edgeBen[idx];
      if (b <= 0) continue;
      const oi = edgeTo[idx];
      const pos = revOff[oi] + revPos[oi]++;
      revTi[pos] = ti;
      revBen[pos] = b * K;
    }
  }

  const objOwner = new Int32Array(nO).fill(-1);
  const trackToObs = new Int32Array(nT).fill(-1);
  const objPrice = new Float64Array(nO);

  const sched = [];
  const e0 = Math.max(1, Math.pow(2, Math.floor(Math.log2(maxB)))) * K;
  for (let s = e0; s >= 1; s = Math.floor(s / 4)) {
    sched.push(s);
    if (s === 1) break;
  }
  if (sched[sched.length - 1] !== 1) sched.push(1);

  const fq = [];
  const rq = [];
  let capHits = 0;
  let roundLimitHits = 0;

  for (const eps of sched) {
    for (let ti = 0; ti < nT; ti++) {
      const cur = trackToObs[ti];
      if (cur >= 0) {
        let curNet = 0.0;
        let bestOther = 0.0;
        const cnt = edgeCnt[ti];
        for (let e = 0; e < cnt; e++) {
          const idx = ti * csrK + e;
          const oi = edgeTo[idx];
          const v = edgeBen[idx] * K - objPrice[oi];
          if (oi === cur) curNet = v;
          else if (v > bestOther) bestOther = v;
        }
        if (curNet < bestOther - eps) {
          objOwner[cur] = -1;
          trackToObs[ti] = -1;
        }
      }
    }

    let converged = false;
    for (let round = 0; round < 64; round++) {
      fq.length = 0;
      for (let ti = 0; ti < nT; ti++) {
        if (trackToObs[ti] < 0 && edgeCnt[ti] > 0) fq.push(ti);
      }
      let fSteps = 0;
      const maxFSteps = nT * 20 + 64;
      while (fq.length > 0 && fSteps++ < maxFSteps) {
        const ti = fq.pop();
        if (trackToObs[ti] >= 0) continue;
        let best1 = 0.0;
        let best2 = 0.0;
        let bestOi = -1;
        const cnt = edgeCnt[ti];
        for (let e = 0; e < cnt; e++) {
          const idx = ti * csrK + e;
          const oi = edgeTo[idx];
          const net = edgeBen[idx] * K - objPrice[oi];
          if (net > best1) {
            best2 = best1;
            best1 = net;
            bestOi = oi;
          } else if (net > best2) {
            best2 = net;
          }
        }
        if (bestOi >= 0 && best1 > 0.0) {
          const prev = objOwner[bestOi];
          const bid = best1 - best2 + eps;
          objPrice[bestOi] += bid;
          if (prev >= 0) {
            trackToObs[prev] = -1;
            fq.push(prev);
          }
          objOwner[bestOi] = ti;
          trackToObs[ti] = bestOi;
        }
      }
      if (fq.length > 0) capHits++;

      rq.length = 0;
      for (let oi = 0; oi < nO; oi++) {
        if (objOwner[oi] < 0 && objPrice[oi] > 0.0) rq.push(oi);
      }
      let rSteps = 0;
      const maxRSteps = nO * 20 + 64;
      while (rq.length > 0 && rSteps++ < maxRSteps) {
        const oi = rq.pop();
        if (objOwner[oi] >= 0 || objPrice[oi] <= 0.0) continue;
        let b1 = 0.0;
        let b2 = 0.0;
        let bestTi = -1;
        const start = revOff[oi];
        const end = revOff[oi + 1];
        for (let p = start; p < end; p++) {
          const ti = revTi[p];
          const cur = trackToObs[ti];
          let pi = 0.0;
          if (cur >= 0) {
            const cnt = edgeCnt[ti];
            for (let e = 0; e < cnt; e++) {
              const idx = ti * csrK + e;
              if (edgeTo[idx] === cur) {
                pi = edgeBen[idx] * K - objPrice[cur];
                break;
              }
            }
          }
          const val = revBen[p] - pi;
          if (val > b1) {
            b2 = b1;
            b1 = val;
            bestTi = ti;
          } else if (val > b2) {
            b2 = val;
          }
        }
        if (bestTi >= 0 && b1 > 0.0) {
          const newP = Math.max(0.0, b2 - eps);
          const oldOi = trackToObs[bestTi];
          objPrice[oi] = newP;
          if (oldOi >= 0) {
            objOwner[oldOi] = -1;
            if (objPrice[oldOi] > 0.0) rq.push(oldOi);
          }
          objOwner[oi] = bestTi;
          trackToObs[bestTi] = oi;
        } else {
          objPrice[oi] = 0.0;
        }
      }
      if (rq.length > 0) capHits++;

      let dirty = false;
      for (let ti = 0; ti < nT; ti++) {
        const cur = trackToObs[ti];
        let bestOther = 0.0;
        let curNet = 0.0;
        const cnt = edgeCnt[ti];
        for (let e = 0; e < cnt; e++) {
          const idx = ti * csrK + e;
          const oi = edgeTo[idx];
          const net = edgeBen[idx] * K - objPrice[oi];
          if (oi === cur) curNet = net;
          else if (net > bestOther) bestOther = net;
        }
        if (cur >= 0) {
          if (curNet < bestOther - eps) {
            objOwner[cur] = -1;
            trackToObs[ti] = -1;
            dirty = true;
          }
        } else if (bestOther > 0.0) {
          dirty = true;
        }
      }
      if (!dirty) {
        for (let oi = 0; oi < nO; oi++) {
          if (objOwner[oi] < 0 && objPrice[oi] > 0.0) {
            dirty = true;
            break;
          }
        }
      }
      if (!dirty) {
        converged = true;
        break;
      }
    }
    if (!converged) roundLimitHits++;
  }
  return { trackToObs, capHits, roundLimitHits };
}

function runJsAblationTracker(scene, mode) {
  const evaluator = createClearMotEvaluator(4.5);
  let nextTid = 1;
  let tracks = []; // { tid, x, y, vx, vy, age, miss }
  const tickMs = [];
  const MAX_GATE_PX = 5.0; // 2x Q2 max inter-frame displacement (2.5 px/f)
  const MAX_MISS = 1; // Held constant across JS modes
  const useEca =
    mode === 'morton_csr_auction_eca' ||
    mode === 'morton_csr_auction_eca_knn' ||
    mode === 'morton_csr_auction_eca_knn_irl1';
  const useKnn =
    mode === 'morton_csr_auction_eca_knn' || mode === 'morton_csr_auction_eca_knn_irl1';
  const useIrl1 = mode === 'morton_csr_auction_eca_knn_irl1';

  // Quantized spatial ECA temporal static-clutter persistence table (4px bins)
  const ECA_BIN = 4.0;
  const ECA_W = Math.ceil(HD_W / ECA_BIN);
  const ECA_H = Math.ceil(HD_H / ECA_BIN);
  const ecaPrevActive = new Uint8Array(ECA_W * ECA_H);
  const ecaCurActive = new Uint8Array(ECA_W * ECA_H);
  const ecaStaticStreak = new Uint8Array(ECA_W * ECA_H);
  let prevObs = [];
  let auctionCapHits = 0;
  let auctionRoundLimitHits = 0;

  for (let f = 0; f < scene.numFrames; f++) {
    const rawObs = buildNoisyObservations(scene.gtFrames[f], f);
    const t0 = performance.now();

    let filteredObs = rawObs;
    if (useEca) {
      ecaCurActive.fill(0);
      for (let i = 0; i < rawObs.length; i++) {
        const bx = Math.max(0, Math.min(ECA_W - 1, Math.floor(rawObs[i].x / ECA_BIN)));
        const by = Math.max(0, Math.min(ECA_H - 1, Math.floor(rawObs[i].y / ECA_BIN)));
        ecaCurActive[by * ECA_W + bx] = 1;
      }
      for (let b = 0; b < ecaCurActive.length; b++) {
        if (ecaCurActive[b] && ecaPrevActive[b]) {
          ecaStaticStreak[b] = Math.min(255, ecaStaticStreak[b] + 1);
        } else {
          ecaStaticStreak[b] = 0;
        }
        ecaPrevActive[b] = ecaCurActive[b];
      }
      filteredObs = [];
      for (let i = 0; i < rawObs.length; i++) {
        const o = rawObs[i];
        const bx = Math.max(0, Math.min(ECA_W - 1, Math.floor(o.x / ECA_BIN)));
        const by = Math.max(0, Math.min(ECA_H - 1, Math.floor(o.y / ECA_BIN)));
        const streak = ecaStaticStreak[by * ECA_W + bx];
        if (streak >= 2 && prevObs.length > 0) {
          let nearPrevZero = false;
          for (let p = 0; p < prevObs.length; p++) {
            const dx = o.x - prevObs[p].x;
            const dy = o.y - prevObs[p].y;
            if (dx * dx + dy * dy <= 0.16) {
              nearPrevZero = true;
              break;
            }
          }
          if (nearPrevZero) continue;
        }
        filteredObs.push(o);
      }
    }
    prevObs = rawObs;

    const nT = tracks.length;
    const nO = filteredObs.length;
    const predX = new Float64Array(nT);
    const predY = new Float64Array(nT);

    // Optional true 6-NN Euclidean neighborhood velocity consensus prior
    let knnVx = null;
    let knnVy = null;
    if (useKnn && nT >= 8) {
      knnVx = new Float64Array(nT);
      knnVy = new Float64Array(nT);
      const KN_CELL = 24.0;
      const KW = Math.ceil(HD_W / KN_CELL);
      const KH = Math.ceil(HD_H / KN_CELL);
      const kGrid = Array.from({ length: KW * KH }, () => []);
      for (let ti = 0; ti < nT; ti++) {
        if (tracks[ti].age < 2) continue;
        const gx = Math.max(0, Math.min(KW - 1, Math.floor(tracks[ti].x / KN_CELL)));
        const gy = Math.max(0, Math.min(KH - 1, Math.floor(tracks[ti].y / KN_CELL)));
        kGrid[gy * KW + gx].push(ti);
      }
      const bestKIdx = new Int32Array(6);
      const bestKD2 = new Float64Array(6);
      for (let ti = 0; ti < nT; ti++) {
        const tr = tracks[ti];
        const gx = Math.max(0, Math.min(KW - 1, Math.floor(tr.x / KN_CELL)));
        const gy = Math.max(0, Math.min(KH - 1, Math.floor(tr.y / KN_CELL)));
        let kLen = 0;
        for (let oy = -1; oy <= 1; oy++) {
          const yy = gy + oy;
          if (yy < 0 || yy >= KH) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const xx = gx + ox;
            if (xx < 0 || xx >= KW) continue;
            const list = kGrid[yy * KW + xx];
            for (let m = 0; m < list.length; m++) {
              const tj = list[m];
              if (tj === ti) continue;
              const dx = tracks[tj].x - tr.x;
              const dy = tracks[tj].y - tr.y;
              const d2 = dx * dx + dy * dy;
              if (d2 > 1e-4 && d2 <= 576.0) {
                if (kLen < 6) {
                  let ins = kLen;
                  while (ins > 0 && bestKD2[ins - 1] > d2) {
                    bestKD2[ins] = bestKD2[ins - 1];
                    bestKIdx[ins] = bestKIdx[ins - 1];
                    ins--;
                  }
                  bestKD2[ins] = d2;
                  bestKIdx[ins] = tj;
                  kLen++;
                } else if (d2 < bestKD2[5]) {
                  let ins = 5;
                  while (ins > 0 && bestKD2[ins - 1] > d2) {
                    bestKD2[ins] = bestKD2[ins - 1];
                    bestKIdx[ins] = bestKIdx[ins - 1];
                    ins--;
                  }
                  bestKD2[ins] = d2;
                  bestKIdx[ins] = tj;
                }
              }
            }
          }
        }
        let wSum = 0.0;
        let vxSum = 0.0;
        let vySum = 0.0;
        for (let k = 0; k < kLen; k++) {
          const tj = bestKIdx[k];
          const w = 1.0 / (4.0 + bestKD2[k]);
          wSum += w;
          vxSum += w * tracks[tj].vx;
          vySum += w * tracks[tj].vy;
        }
        if (wSum > 0.0 && tr.age >= 2) {
          const cvx = vxSum / wSum;
          const cvy = vySum / wSum;
          const blend = tr.miss > 0 ? 0.45 : 0.25;
          knnVx[ti] = tr.vx * (1.0 - blend) + cvx * blend;
          knnVy[ti] = tr.vy * (1.0 - blend) + cvy * blend;
        } else {
          knnVx[ti] = tr.vx;
          knnVy[ti] = tr.vy;
        }
      }
    }

    for (let ti = 0; ti < nT; ti++) {
      const vxUse = knnVx ? knnVx[ti] : tracks[ti].vx;
      const vyUse = knnVy ? knnVy[ti] : tracks[ti].vy;
      predX[ti] = tracks[ti].x + vxUse * scene.dtSec;
      predY[ti] = tracks[ti].y + vyUse * scene.dtSec;
    }

    let obs = filteredObs;
    const trackToObs = new Int32Array(nT).fill(-1);

    if (nT > 0 && nO > 0) {
      if (mode === 'nearest_greedy') {
        const candidates = [];
        for (let ti = 0; ti < nT; ti++) {
          const px = predX[ti];
          const py = predY[ti];
          for (let oi = 0; oi < nO; oi++) {
            const d = Math.hypot(obs[oi].x - px, obs[oi].y - py);
            if (d <= MAX_GATE_PX) candidates.push({ ti, oi, d });
          }
        }
        candidates.sort((a, b) => a.d - b.d);
        const usedT = new Uint8Array(nT);
        const usedO = new Uint8Array(nO);
        for (let k = 0; k < candidates.length; k++) {
          const c = candidates[k];
          if (usedT[c.ti] || usedO[c.oi]) continue;
          usedT[c.ti] = 1;
          usedO[c.oi] = 1;
          trackToObs[c.ti] = c.oi;
        }
      } else {
        // Morton-ordered Spatial CSR (k<=8) isolating spatial indexing from Auction assignment
        obs = filteredObs
          .map((o) => ({
            x: o.x,
            y: o.y,
            morton: computeMorton10(
              Math.max(0, Math.min(1023, Math.round(o.x))),
              Math.max(0, Math.min(1023, Math.round(o.y)))
            ),
          }))
          .sort((a, b) => a.morton - b.morton);

        const CELL_SIZE = 20.0;
        const C_COLS = 32;
        const C_ROWS = 18;
        const NUM_CELLS = C_COLS * C_ROWS;
        const cellObs = Array.from({ length: NUM_CELLS }, () => []);
        for (let oi = 0; oi < nO; oi++) {
          const cx = Math.max(0, Math.min(C_COLS - 1, Math.floor(obs[oi].x / CELL_SIZE)));
          const cy = Math.max(0, Math.min(C_ROWS - 1, Math.floor(obs[oi].y / CELL_SIZE)));
          cellObs[cy * C_COLS + cx].push(oi);
        }

        const CSR_K = 8;
        const edgeTo = new Int32Array(nT * CSR_K).fill(-1);
        const edgeBen = new Int32Array(nT * CSR_K);
        const edgeCnt = new Int32Array(nT);
        const topBuf = [];
        const greedyEdges = mode === 'morton_csr_greedy' ? [] : null;

        for (let ti = 0; ti < nT; ti++) {
          const px = predX[ti];
          const py = predY[ti];
          const cx0 = Math.max(0, Math.floor((px - MAX_GATE_PX) / CELL_SIZE));
          const cx1 = Math.min(C_COLS - 1, Math.floor((px + MAX_GATE_PX) / CELL_SIZE));
          const cy0 = Math.max(0, Math.floor((py - MAX_GATE_PX) / CELL_SIZE));
          const cy1 = Math.min(C_ROWS - 1, Math.floor((py + MAX_GATE_PX) / CELL_SIZE));
          topBuf.length = 0;
          for (let cy = cy0; cy <= cy1; cy++) {
            for (let cx = cx0; cx <= cx1; cx++) {
              const list = cellObs[cy * C_COLS + cx];
              for (let m = 0; m < list.length; m++) {
                const oi = list[m];
                const dx = obs[oi].x - px;
                const dy = obs[oi].y - py;
                const d2 = dx * dx + dy * dy;
                if (d2 <= MAX_GATE_PX * MAX_GATE_PX) {
                  const ben = Math.round((MAX_GATE_PX * MAX_GATE_PX - d2) * 64.0 + 32.0);
                  topBuf.push({ oi, ben });
                }
              }
            }
          }
          if (topBuf.length > CSR_K) topBuf.sort((a, b) => b.ben - a.ben);
          const cnt = Math.min(CSR_K, topBuf.length);
          edgeCnt[ti] = cnt;
          for (let e = 0; e < cnt; e++) {
            edgeTo[ti * CSR_K + e] = topBuf[e].oi;
            edgeBen[ti * CSR_K + e] = topBuf[e].ben;
            if (greedyEdges) greedyEdges.push({ ti, oi: topBuf[e].oi, ben: topBuf[e].ben });
          }
        }

        if (mode === 'morton_csr_greedy') {
          greedyEdges.sort((a, b) => b.ben - a.ben);
          const usedT = new Uint8Array(nT);
          const usedO = new Uint8Array(nO);
          for (let k = 0; k < greedyEdges.length; k++) {
            const e = greedyEdges[k];
            if (usedT[e.ti] || usedO[e.oi]) continue;
            usedT[e.ti] = 1;
            usedO[e.oi] = 1;
            trackToObs[e.ti] = e.oi;
          }
        } else {
          const solved = solveAuctionAsym(edgeTo, edgeBen, edgeCnt, nT, nO, CSR_K);
          auctionCapHits += solved.capHits;
          auctionRoundLimitHits += solved.roundLimitHits;
          trackToObs.set(solved.trackToObs);
        }
      }
    }

    const usedO = new Uint8Array(nO);
    const nextTracks = [];
    for (let ti = 0; ti < nT; ti++) {
      const oi = trackToObs[ti];
      const tr = tracks[ti];
      if (oi >= 0) {
        usedO[oi] = 1;
        const ob = obs[oi];
        const vx = (ob.x - tr.x) / scene.dtSec;
        const vy = (ob.y - tr.y) / scene.dtSec;
        nextTracks.push({
          tid: tr.tid,
          x: ob.x,
          y: ob.y,
          vx: tr.age === 1 ? vx : tr.vx * 0.4 + vx * 0.6,
          vy: tr.age === 1 ? vy : tr.vy * 0.4 + vy * 0.6,
          age: tr.age + 1,
          miss: 0,
        });
      } else if (tr.miss < MAX_MISS) {
        nextTracks.push({ ...tr, x: predX[ti], y: predY[ti], miss: tr.miss + 1 });
      }
    }
    for (let oi = 0; oi < nO; oi++) {
      if (!usedO[oi] && nextTracks.length < MAX_MICRO) {
        nextTracks.push({ tid: nextTid++, x: obs[oi].x, y: obs[oi].y, vx: 0, vy: 0, age: 1, miss: 0 });
      }
    }

    // Optional 1-step Huber velocity refinement proxy over actual 6 nearest Euclidean neighbors
    if (useIrl1 && nextTracks.length >= 8) {
      const IR_CELL = 24.0;
      const IW = Math.ceil(HD_W / IR_CELL);
      const IH = Math.ceil(HD_H / IR_CELL);
      const iGrid = Array.from({ length: IW * IH }, () => []);
      for (let i = 0; i < nextTracks.length; i++) {
        if (nextTracks[i].age < 2 || nextTracks[i].miss > 0) continue;
        const gx = Math.max(0, Math.min(IW - 1, Math.floor(nextTracks[i].x / IR_CELL)));
        const gy = Math.max(0, Math.min(IH - 1, Math.floor(nextTracks[i].y / IR_CELL)));
        iGrid[gy * IW + gx].push(i);
      }
      const refX = nextTracks.map((t) => t.x);
      const refY = nextTracks.map((t) => t.y);
      const refVx = nextTracks.map((t) => t.vx);
      const refVy = nextTracks.map((t) => t.vy);
      const bestIIdx = new Int32Array(6);
      const bestID2 = new Float64Array(6);
      for (let i = 0; i < nextTracks.length; i++) {
        const ti = nextTracks[i];
        if (ti.age < 2 || ti.miss > 0) continue;
        const gx = Math.max(0, Math.min(IW - 1, Math.floor(refX[i] / IR_CELL)));
        const gy = Math.max(0, Math.min(IH - 1, Math.floor(refY[i] / IR_CELL)));
        let nb = 0;
        for (let oy = -1; oy <= 1; oy++) {
          const yy = gy + oy;
          if (yy < 0 || yy >= IH) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const xx = gx + ox;
            if (xx < 0 || xx >= IW) continue;
            const list = iGrid[yy * IW + xx];
            for (let m = 0; m < list.length; m++) {
              const j = list[m];
              if (j === i) continue;
              const dx = refX[j] - refX[i];
              const dy = refY[j] - refY[i];
              const d2 = dx * dx + dy * dy;
              if (d2 <= 484.0) {
                if (nb < 6) {
                  let ins = nb;
                  while (ins > 0 && bestID2[ins - 1] > d2) {
                    bestID2[ins] = bestID2[ins - 1];
                    bestIIdx[ins] = bestIIdx[ins - 1];
                    ins--;
                  }
                  bestID2[ins] = d2;
                  bestIIdx[ins] = j;
                  nb++;
                } else if (d2 < bestID2[5]) {
                  let ins = 5;
                  while (ins > 0 && bestID2[ins - 1] > d2) {
                    bestID2[ins] = bestID2[ins - 1];
                    bestIIdx[ins] = bestIIdx[ins - 1];
                    ins--;
                  }
                  bestID2[ins] = d2;
                  bestIIdx[ins] = j;
                }
              }
            }
          }
        }
        let wSum = 0.0;
        let dvxSum = 0.0;
        let dvySum = 0.0;
        for (let k = 0; k < nb; k++) {
          const j = bestIIdx[k];
          const d = Math.sqrt(bestID2[k]);
          const dvx = refVx[j] - refVx[i];
          const dvy = refVy[j] - refVy[i];
          const resV = Math.hypot(dvx, dvy);
          const huber = resV <= 8.0 ? 1.0 : 8.0 / resV;
          const w = huber / (6.0 + d);
          wSum += w;
          dvxSum += w * dvx;
          dvySum += w * dvy;
        }
        if (nb >= 2 && wSum > 0.0) {
          const corrVx = dvxSum / wSum;
          const corrVy = dvySum / wSum;
          ti.vx += 0.25 * corrVx;
          ti.vy += 0.25 * corrVy;
          ti.x += 0.15 * corrVx * scene.dtSec;
          ti.y += 0.15 * corrVy * scene.dtSec;
        }
      }
    }

    tracks = nextTracks;

    tickMs.push(performance.now() - t0);
    if (f >= 3) {
      // Score observed (miss===0) tracks only at frame f; never give coasted tracks free credit
      const confirmed = tracks.filter((t) => t.age >= 2 && t.miss === 0).map((t) => ({ tid: t.tid, x: t.x, y: t.y }));
      evaluator.update(scene.gtFrames[f], confirmed);
    }
  }

  const s = evaluator.summary();
  const labelMap = {
    nearest_greedy: 'JS Nearest-Greedy',
    morton_csr_greedy: 'JS Morton-CSR+Greedy',
    morton_csr_auction: 'JS Morton-CSR+Auction',
    morton_csr_auction_eca: 'JS +Static-Mask [Proxy]',
    morton_csr_auction_eca_knn: 'JS +Mask+Actual-6NN [Proxy]',
    morton_csr_auction_eca_knn_irl1: 'JS +Mask+6NN+1StepHuber [Proxy]',
  };
  return {
    name: labelMap[mode] || mode,
    meanMs: tickMs.reduce((a, b) => a + b, 0) / Math.max(1, tickMs.length),
    medianMs: computeMedian(tickMs),
    auctionCapHits,
    auctionRoundLimitHits,
    ...s,
  };
}

// ============================================================================
// Wasm Loader, Binary/ABI Audit & Stream Helpers
// ============================================================================

function readLebU32(buf, posObj) {
  let result = 0;
  let shift = 0;
  while (posObj.i < buf.length) {
    const b = buf[posObj.i++];
    result |= (b & 0x7f) << shift;
    if ((b & 0x80) === 0) return result >>> 0;
    shift += 7;
    if (shift > 35) break;
  }
  throw new Error('Malformed LEB128 u32 in Wasm binary');
}

function readLebI32(buf, posObj) {
  let result = 0;
  let shift = 0;
  let b = 0;
  while (posObj.i < buf.length) {
    b = buf[posObj.i++];
    result |= (b & 0x7f) << shift;
    shift += 7;
    if ((b & 0x80) === 0) break;
    if (shift > 35) throw new Error('Malformed LEB128 i32 in Wasm binary');
  }
  if (shift < 32 && (b & 0x40) !== 0) result |= ~0 << shift;
  return result | 0;
}

function parseWasmBinaryMetadata(wasmBytes) {
  if (
    wasmBytes.length < 8 ||
    wasmBytes[0] !== 0x00 ||
    wasmBytes[1] !== 0x61 ||
    wasmBytes[2] !== 0x73 ||
    wasmBytes[3] !== 0x6d ||
    wasmBytes[4] !== 0x01 ||
    wasmBytes[5] !== 0x00 ||
    wasmBytes[6] !== 0x00 ||
    wasmBytes[7] !== 0x00
  ) {
    return { validHeader: false, error: 'Invalid Wasm magic/version header' };
  }
  const pos = { i: 8 };
  let importCount = 0;
  const memories = [];
  const globals = [];
  const dataSegments = [];
  let hasMemorySection = false;
  let hasExportSection = false;
  try {
    while (pos.i < wasmBytes.length) {
      const secId = wasmBytes[pos.i++];
      const secLen = readLebU32(wasmBytes, pos);
      const secEnd = pos.i + secLen;
      if (secEnd > wasmBytes.length) {
        return { validHeader: false, error: `Section ${secId} overruns binary` };
      }
      if (secId === 2) {
        importCount = readLebU32(wasmBytes, pos);
      } else if (secId === 5) {
        hasMemorySection = true;
        const cnt = readLebU32(wasmBytes, pos);
        for (let k = 0; k < cnt; k++) {
          const flags = wasmBytes[pos.i++];
          const initial = readLebU32(wasmBytes, pos);
          const max = (flags & 1) !== 0 ? readLebU32(wasmBytes, pos) : null;
          memories.push({ flags, initial, max });
        }
      } else if (secId === 6) {
        const cnt = readLebU32(wasmBytes, pos);
        for (let k = 0; k < cnt; k++) {
          const valType = wasmBytes[pos.i++];
          const mut = wasmBytes[pos.i++];
          let initI32 = null;
          if (wasmBytes[pos.i] === 0x41) {
            pos.i++;
            initI32 = readLebI32(wasmBytes, pos);
          }
          globals.push({ valType, mut, initI32 });
          while (pos.i < secEnd && wasmBytes[pos.i] !== 0x0b) pos.i++;
          if (pos.i < secEnd) pos.i++;
        }
      } else if (secId === 7) {
        hasExportSection = true;
      } else if (secId === 11) {
        const cnt = readLebU32(wasmBytes, pos);
        for (let k = 0; k < cnt; k++) {
          const mode = readLebU32(wasmBytes, pos);
          if (mode === 0) {
            let offset = null;
            if (wasmBytes[pos.i] === 0x41) {
              pos.i++;
              offset = readLebI32(wasmBytes, pos);
            }
            while (pos.i < secEnd && wasmBytes[pos.i] !== 0x0b) pos.i++;
            if (pos.i < secEnd) pos.i++;
            const size = readLebU32(wasmBytes, pos);
            pos.i += size;
            if (offset !== null) dataSegments.push({ offset, size, end: offset + size });
          } else {
            break;
          }
        }
      }
      pos.i = secEnd;
    }
  } catch (err) {
    return { validHeader: false, error: err.message };
  }
  const maxDataEnd = dataSegments.reduce((mx, d) => Math.max(mx, d.end), 0);
  return {
    validHeader: true,
    importCount,
    hasMemorySection,
    hasExportSection,
    memories,
    globals,
    dataSegments,
    maxDataEnd,
  };
}

function parseFuncSectionIds(xdumpText) {
  const funcSecMatch = /\bFunction\[(\d+)\]:/.exec(xdumpText);
  const declaredDefinedFuncCount = funcSecMatch ? Number(funcSecMatch[1]) : null;
  const declaredFuncIds = new Set();
  const exportFuncs = [];
  let section = '';
  for (const ln of xdumpText.split('\n')) {
    const secHdr = /^([A-Z][A-Za-z0-9_]*)\[\d+\]:/.exec(ln.trim());
    if (secHdr) {
      section = secHdr[1];
      continue;
    }
    const em = /^\s*-\s+func\[(\d+)\](?:\s+<[^>]*>)?\s+->\s+"([^"]+)"/.exec(ln);
    if (em) {
      exportFuncs.push({ id: Number(em[1]), name: em[2] });
      continue;
    }
    if (section === 'Function') {
      const fm = /^\s*-\s+func\[(\d+)\]/.exec(ln);
      if (fm) declaredFuncIds.add(Number(fm[1]));
    }
  }
  return {
    definedFuncsMatch: Boolean(funcSecMatch),
    declaredDefinedFuncCount,
    declaredFuncIds,
    exportFuncs,
  };
}

function analyzeObjdumpStatic(xdumpText, disText) {
  const memMatch = /memory\[0\]\s+pages:\s+initial=(\d+)(?:\s+max=(\d+))?/.exec(xdumpText);
  const stackMatch = /global\[0\]\s+i32\s+mutable=1\s+(?:<__stack_pointer>|\$__stack_pointer)?\s*-?\s*init\s+i32=(\d+)/.exec(xdumpText);
  const dataSegments = [];
  for (const m of xdumpText.matchAll(/segment\[\d+\]\s+<[^>]+>\s+memory=0\s+size=(\d+)\s+-\s+init\s+i32=(\d+)/g)) {
    const size = Number(m[1]);
    const init = Number(m[2]);
    dataSegments.push({ init, size, end: init + size });
  }
  const maxDataEnd = dataSegments.reduce((mx, d) => Math.max(mx, d.end), 0);

  const {
    definedFuncsMatch,
    declaredDefinedFuncCount,
    declaredFuncIds,
    exportFuncs,
  } = parseFuncSectionIds(xdumpText);

  const v128Count = (disText.match(/\bv128\./g) || []).length;
  const clzCount = (disText.match(/\bi32\.clz\b/g) || []).length;
  const memoryGrowCount = (disText.match(/\bmemory\.grow\b/g) || []).length;
  const callIndirectCount = (disText.match(/\bcall_indirect\b/g) || []).length;

  const ALLOC_SYMBOL_RE =
    /(__rust_(?:alloc|dealloc|realloc|alloc_zeroed)|__rdl_(?:alloc|dealloc|realloc|alloc_zeroed)|\b(?:dlmalloc|malloc|calloc|realloc|free)\b)/i;

  const funcs = new Map();
  let cur = null;
  let blockDepth = 0;
  let unterminatedFuncs = 0;
  let unknownStackWriteCount = 0;
  const allocatorFuncIds = new Set();
  const opHistory = [];
  let frames = [];
  const cloneFpMap = (m) => new Map(m);
  const meetFpInto = (target, srcFp, srcSpDepth) => {
    if (!target.hasExit) {
      target.exit = cloneFpMap(srcFp);
      target.exitMinSpDepth = srcSpDepth;
      target.exitMaxSpDepth = srcSpDepth;
      target.hasExit = true;
    } else {
      if (target.exitMinSpDepth !== srcSpDepth || target.exitMaxSpDepth !== srcSpDepth) {
        unknownStackWriteCount++;
      }
      for (const [k, v] of target.exit.entries()) {
        if (!srcFp.has(k) || srcFp.get(k) !== v) target.exit.delete(k);
      }
      target.exitMinSpDepth = Math.min(target.exitMinSpDepth, srcSpDepth);
      target.exitMaxSpDepth = Math.max(target.exitMaxSpDepth, srcSpDepth);
    }
  };
  const lines = disText.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    const fm = /^([0-9a-f]+)\s+func\[(\d+)\](?:\s+<([^>]*)>)?\s*:/.exec(ln);
    if (fm) {
      if (cur && !cur.ended) unterminatedFuncs++;
      const fName = fm[3] || '';
      const isAlloc = ALLOC_SYMBOL_RE.test(fName);
      cur = {
        id: Number(fm[2]),
        name: fName,
        frame: 0,
        calls: new Set(),
        callSites: [],
        isAllocator: isAlloc,
        callsAllocatorDirect: false,
        ended: false,
      };
      if (isAlloc) allocatorFuncIds.add(cur.id);
      funcs.set(cur.id, cur);
      blockDepth = 0;
      frames = [
        {
          kind: 'func',
          fp: new Map(),
          spDepth: 0,
          entryFp: new Map(),
          entrySpDepth: 0,
          exit: new Map(),
          exitMinSpDepth: 0,
          exitMaxSpDepth: 0,
          hasExit: false,
          dead: false,
          seenElse: false,
        },
      ];
      opHistory.length = 0;
      continue;
    }
    if (!cur) continue;

    const barIdx = ln.indexOf('|');
    const op = (barIdx >= 0 ? ln.slice(barIdx + 1) : ln).trim();
    if (!op || /^local\[/.test(op)) continue;

    if (cur.ended) {
      unterminatedFuncs++;
      cur.ended = false;
    }

    const prev1 = opHistory[opHistory.length - 1] || '';
    const prev2 = opHistory[opHistory.length - 2] || '';
    const prev3 = opHistory[opHistory.length - 3] || '';
    const prev4 = opHistory[opHistory.length - 4] || '';
    const topFrame = () => frames[frames.length - 1];

    const blkMatch = /^(block|loop|if)\b/.exec(op);
    if (blkMatch) {
      blockDepth++;
      const outer = topFrame();
      frames.push({
        kind: blkMatch[1],
        fp: cloneFpMap(outer.fp),
        spDepth: outer.spDepth,
        entryFp: cloneFpMap(outer.fp),
        entrySpDepth: outer.spDepth,
        exit: new Map(),
        exitMinSpDepth: outer.spDepth,
        exitMaxSpDepth: outer.spDepth,
        hasExit: false,
        dead: outer.dead,
        seenElse: false,
        startIdx: i,
        iters: 0,
        needsReeval: false,
        spMismatch: false,
        errAtEntry: unknownStackWriteCount,
        callSitesLenAtEntry: cur.callSites.length,
        callsAtEntry: new Set(cur.calls),
        entryOpHistory: opHistory.slice(),
      });
    } else if (/^else\b/.test(op)) {
      const fState = topFrame();
      if (frames.length > 1) {
        fState.seenElse = true;
        if (!fState.dead) meetFpInto(fState, fState.fp, fState.spDepth);
        fState.fp = cloneFpMap(fState.entryFp);
        fState.spDepth = fState.entrySpDepth;
        fState.dead = frames[frames.length - 2].dead;
      }
    } else if (/^end\b/.test(op)) {
      if (blockDepth > 0) {
        blockDepth--;
        if (frames.length > 1) {
          const popped = frames.pop();
          const outer = topFrame();
          if (popped.kind === 'loop' && popped.needsReeval && popped.iters < 8) {
            // Re-evaluate loop body from narrowed entryFp until fixpoint without double-counting
            unknownStackWriteCount = popped.errAtEntry;
            cur.callSites.length = popped.callSitesLenAtEntry;
            cur.calls = new Set(popped.callsAtEntry);
            opHistory.length = 0;
            for (const ho of popped.entryOpHistory) opHistory.push(ho);
            popped.fp = cloneFpMap(popped.entryFp);
            popped.spDepth = popped.entrySpDepth;
            popped.exit = new Map();
            popped.exitMinSpDepth = popped.entrySpDepth;
            popped.exitMaxSpDepth = popped.entrySpDepth;
            popped.hasExit = false;
            popped.dead = outer.dead;
            popped.needsReeval = false;
            popped.spMismatch = false;
            popped.iters++;
            frames.push(popped);
            blockDepth++;
            i = popped.startIdx;
            continue;
          }
          if (popped.kind === 'loop' && (popped.spMismatch || popped.needsReeval)) {
            unknownStackWriteCount++;
          }
          if (!popped.dead) meetFpInto(popped, popped.fp, popped.spDepth);
          if (popped.kind === 'if' && !popped.seenElse && !outer.dead) {
            meetFpInto(popped, popped.entryFp, popped.entrySpDepth);
          }
          if (popped.hasExit) {
            outer.fp = popped.exit;
            outer.spDepth = popped.exitMaxSpDepth;
            outer.dead = false;
          } else {
            outer.dead = true;
          }
        }
      } else {
        const fnTop = frames[0];
        if (
          fnTop &&
          ((!fnTop.dead && fnTop.spDepth !== 0) ||
            (fnTop.hasExit && (fnTop.exitMinSpDepth !== 0 || fnTop.exitMaxSpDepth !== 0)))
        ) {
          unknownStackWriteCount++;
        }
        cur.ended = true;
      }
    } else {
      const curState = topFrame();
      const cm = /\bcall\s+(\d+)(.*)$/.exec(op);
      if (cm) {
        const calleeId = Number(cm[1]);
        const inlineAlloc = ALLOC_SYMBOL_RE.test(cm[2] || '');
        if (inlineAlloc) allocatorFuncIds.add(calleeId);
        cur.calls.add(calleeId);
        cur.callSites.push({ calleeId, inlineAlloc });
      }

      if (/\bglobal\.set\s+(?:0\b|\$__stack_pointer\b)/.test(op)) {
        let recognized = false;
        // Match optional local.tee between arithmetic op and global.set 0 (local.set consumes value)
        let arithOp = prev1;
        let constOp = prev2;
        let baseOp = prev3;
        const teeMatch = /^local\.tee\s+(\d+)\b/.exec(prev1);
        if (teeMatch) {
          arithOp = prev2;
          constOp = prev3;
          baseOp = prev4;
        }

        const isSub = /^i32\.sub\b/.test(arithOp);
        const isAdd = /^i32\.add\b/.test(arithOp);
        const mConst = /^i32\.const\s+(-?\d+)\b/.exec(constOp);
        if ((isSub || isAdd) && mConst) {
          const sVal = Number(mConst[1]) | 0;
          if (/^global\.get\s+(?:0\b|\$__stack_pointer\b)/.test(baseOp)) {
            const decr = isSub ? sVal : -sVal;
            if (decr >= 0 && !curState.dead) {
              const nextDepth = curState.spDepth + decr;
              curState.spDepth = nextDepth;
              cur.frame = Math.max(cur.frame, nextDepth);
              if (teeMatch) curState.fp.set(Number(teeMatch[1]), nextDepth);
              recognized = true;
            }
          } else {
            const mLoc = /^local\.get\s+(\d+)\b/.exec(baseOp);
            if (mLoc && isAdd && sVal >= 0 && !curState.dead) {
              const lid = Number(mLoc[1]);
              // Epilogue restore requires proven un-clobbered FP local with non-negative target SP depth
              if (curState.fp.has(lid) && curState.fp.get(lid) >= sVal) {
                curState.spDepth = curState.fp.get(lid) - sVal;
                recognized = true;
              }
            }
          }
        } else {
          const mLocOnly = /^local\.get\s+(\d+)\b/.exec(prev1);
          if (mLocOnly && !curState.dead) {
            const lid = Number(mLocOnly[1]);
            if (curState.fp.has(lid) && curState.fp.get(lid) >= 0) {
              curState.spDepth = curState.fp.get(lid);
              recognized = true;
            }
          }
        }

        if (!recognized) unknownStackWriteCount++;
      } else if (!curState.dead) {
        // Any other write to local[dst] invalidates stale FP provenance unless copying an intact FP local
        const locWrite = /^local\.(?:set|tee)\s+(\d+)\b/.exec(op);
        if (locWrite) {
          const dst = Number(locWrite[1]);
          const srcGet = /^local\.get\s+(\d+)\b/.exec(prev1);
          if (srcGet && curState.fp.has(Number(srcGet[1]))) {
            curState.fp.set(dst, curState.fp.get(Number(srcGet[1])));
          } else {
            curState.fp.delete(dst);
          }
        }
        const branchTo = (targetIdx) => {
          if (targetIdx < 0 || targetIdx >= frames.length) return;
          const tf = frames[targetIdx];
          if (tf.kind === 'loop') {
            if (curState.spDepth !== tf.entrySpDepth) tf.spMismatch = true;
            for (const [k, v] of tf.entryFp.entries()) {
              if (!curState.fp.has(k) || curState.fp.get(k) !== v) {
                tf.entryFp.delete(k);
                tf.needsReeval = true;
              }
            }
          } else {
            meetFpInto(tf, curState.fp, curState.spDepth);
          }
        };
        const bri = /^br_if\s+(\d+)\b/.exec(op);
        if (bri) {
          branchTo(frames.length - 1 - Number(bri[1]));
        } else if (/^br\s+(\d+)\b/.test(op)) {
          const brMatch = /^br\s+(\d+)\b/.exec(op);
          branchTo(frames.length - 1 - Number(brMatch[1]));
          curState.dead = true;
        } else if (/^(?:return|unreachable|br_table)\b/.test(op)) {
          if (/^return\b/.test(op) && curState.spDepth !== 0) {
            unknownStackWriteCount++;
          }
          if (/^br_table\b/.test(op)) {
            const nums = op.slice(8).match(/\d+/g) || [];
            for (const ns of nums) branchTo(frames.length - 1 - Number(ns));
          }
          curState.dead = true;
        }
      }
    }

    opHistory.push(op);
    if (opHistory.length > 6) opHistory.shift();
  }
  if (cur && !cur.ended) unterminatedFuncs++;

  // Count each allocator call site exactly once across all parsed functions
  let allocatorCallCount = 0;
  let missingCalleeCount = 0;
  for (const fn of funcs.values()) {
    for (const cs of fn.callSites) {
      if (cs.inlineAlloc || allocatorFuncIds.has(cs.calleeId)) {
        allocatorCallCount++;
        fn.callsAllocatorDirect = true;
      }
      if (!funcs.has(cs.calleeId)) {
        missingCalleeCount++;
      }
    }
  }

  let maxShadowStackDepth = 0;
  let stackCycles = 0;
  let allocatorReachableCount = 0;
  const memo = new Map();
  const visiting = new Set();
  const reachableFromExports = new Set();
  function dfs(fid) {
    reachableFromExports.add(fid);
    if (visiting.has(fid)) {
      stackCycles++;
      return 0;
    }
    if (memo.has(fid)) return memo.get(fid);
    const fn = funcs.get(fid);
    if (!fn) {
      return 0;
    }
    visiting.add(fid);
    let bestChild = 0;
    for (const cid of fn.calls) {
      const d = dfs(cid);
      if (d > bestChild) bestChild = d;
    }
    visiting.delete(fid);
    const total = fn.frame + bestChild;
    memo.set(fid, total);
    return total;
  }
  for (const ef of exportFuncs) {
    const d = dfs(ef.id);
    if (d > maxShadowStackDepth) maxShadowStackDepth = d;
  }
  for (const fid of reachableFromExports) {
    const fn = funcs.get(fid);
    if (allocatorFuncIds.has(fid) || (fn && (fn.isAllocator || fn.callsAllocatorDirect))) {
      allocatorReachableCount++;
    }
  }

  let missingExportFuncs = 0;
  for (const ef of exportFuncs) {
    if (!funcs.has(ef.id)) missingExportFuncs++;
  }
  let missingDeclaredFuncs = 0;
  for (const id of declaredFuncIds) {
    if (!funcs.has(id)) missingDeclaredFuncs++;
  }
  const declaredCountMismatch =
    declaredDefinedFuncCount !== null && funcs.size !== declaredDefinedFuncCount;

  const disassemblyComplete =
    definedFuncsMatch &&
    funcs.size > 0 &&
    exportFuncs.length > 0 &&
    missingExportFuncs === 0 &&
    missingDeclaredFuncs === 0 &&
    !declaredCountMismatch &&
    unterminatedFuncs === 0 &&
    missingCalleeCount === 0;

  return {
    available: true,
    memInitialPages: memMatch ? Number(memMatch[1]) : null,
    memMaxPages: memMatch && memMatch[2] !== undefined ? Number(memMatch[2]) : null,
    stackPointerInit: stackMatch ? Number(stackMatch[1]) : null,
    dataSegments,
    maxDataEnd,
    exportFuncCount: exportFuncs.length,
    declaredDefinedFuncCount,
    analyzedFuncCount: funcs.size,
    missingExportFuncs,
    missingDeclaredFuncs,
    missingCalleeCount,
    unterminatedFuncs,
    unknownStackWriteCount,
    disassemblyComplete,
    allocatorCallCount,
    allocatorReachableCount,
    v128Count,
    clzCount,
    memoryGrowCount,
    callIndirectCount,
    maxShadowStackDepth,
    stackCycles,
  };
}

async function loadWasm() {
  if (!fs.existsSync(WASM_PATH)) {
    throw new Error(`Missing Wasm binary: ${WASM_PATH}`);
  }
  const wasmBytes = fs.readFileSync(WASM_PATH);
  const sha256 = crypto.createHash('sha256').update(wasmBytes).digest('hex');
  const mod = await WebAssembly.compile(wasmBytes);
  const imports = WebAssembly.Module.imports(mod);
  const instance = await WebAssembly.instantiate(mod, {});
  const exp = instance.exports;
  const mem = exp.memory;
  const missing = REQUIRED_CHRONOS1000_EXPORTS.filter((name) => typeof exp[name] !== 'function');
  const binaryMeta = parseWasmBinaryMetadata(wasmBytes);

  // Required static binary + disassembly inspection via wasm-objdump (fails closed if unavailable)
  const probeX = spawnSync('wasm-objdump', ['-x', WASM_PATH], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  const probeD = spawnSync('wasm-objdump', ['-d', WASM_PATH], { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
  const objdumpReport =
    probeX.status === 0 && probeX.stdout && probeD.status === 0 && probeD.stdout
      ? analyzeObjdumpStatic(probeX.stdout, probeD.stdout)
      : { available: false, reason: probeX.error?.message || probeD.error?.message || 'wasm-objdump failed' };

  return {
    exp,
    mem,
    missing,
    wasmBytesLen: wasmBytes.byteLength,
    sha256,
    imports,
    binaryMeta,
    objdumpReport,
  };
}

function auditWasmBinaryAndAbi(wasm) {
  const errors = [];
  if (wasm.imports.length !== 0) {
    errors.push(`wasmImports=${wasm.imports.length} (expected 0)`);
  }
  const bm = wasm.binaryMeta;
  if (!bm || !bm.validHeader) {
    errors.push(`wasmBinaryParseError=${bm?.error || 'missing'}`);
  } else {
    if (bm.importCount !== 0) errors.push(`wasmSectionImports=${bm.importCount}!=0`);
    if (!bm.hasMemorySection || bm.memories.length !== 1) {
      errors.push(`wasmMemoryCount=${bm.memories?.length ?? 0}!=1`);
    } else {
      const m0 = bm.memories[0];
      if (m0.initial !== EXPECTED_WASM_PAGES || m0.max !== EXPECTED_WASM_PAGES || (m0.flags & 1) === 0) {
        errors.push(`wasmBinaryMemoryPages=[init=${m0.initial},max=${m0.max},flags=${m0.flags}]!=${EXPECTED_WASM_PAGES}`);
      }
    }
    if (bm.globals.length > 0 && bm.globals[0].initI32 !== null && bm.globals[0].initI32 > 16384) {
      errors.push(`wasmGlobal0StackPtr=${bm.globals[0].initI32}>16384`);
    }
  }

  if (!wasm.mem || wasm.mem.buffer.byteLength !== EXPECTED_WASM_MEM_BYTES) {
    errors.push(
      `wasmMemoryBytes=${wasm.mem ? wasm.mem.buffer.byteLength : 0} (expected fixed ${EXPECTED_WASM_MEM_BYTES} = ${EXPECTED_WASM_PAGES} pages)`
    );
  } else {
    let grew = false;
    try {
      wasm.mem.grow(1);
      grew = true;
    } catch (_) {
      // Expected RangeError on non-growable fixed 448-page Wasm memory
    }
    if (grew) errors.push(`wasmMemoryGrowSucceeded (expected fixed non-growable max=${EXPECTED_WASM_PAGES} pages)`);
  }

  const od = wasm.objdumpReport;
  if (!od || !od.available) {
    errors.push(`wasmObjdumpRequiredUnavailable(${od?.reason || 'missing'})`);
  } else {
    if (
      od.disassemblyComplete !== true ||
      (typeof od.analyzedFuncCount === 'number' && od.analyzedFuncCount <= 0) ||
      (od.missingExportFuncs || 0) > 0 ||
      (od.missingDeclaredFuncs || 0) > 0 ||
      (od.missingCalleeCount || 0) > 0 ||
      (od.unterminatedFuncs || 0) > 0
    ) {
      errors.push(
        `objdumpIncompleteDisassembly(funcs=${od.analyzedFuncCount ?? 'N/A'}/${od.declaredDefinedFuncCount ?? 'N/A'},missingExp=${od.missingExportFuncs ?? 'N/A'},missingCallee=${od.missingCalleeCount ?? 'N/A'},unterm=${od.unterminatedFuncs ?? 'N/A'})`
      );
    }
    if ((od.unknownStackWriteCount || 0) > 0) {
      errors.push(`objdumpUnrecognizedStackWrites=${od.unknownStackWriteCount}>0`);
    }
    if (od.memInitialPages !== EXPECTED_WASM_PAGES || od.memMaxPages !== EXPECTED_WASM_PAGES) {
      errors.push(`objdumpMemoryPages=${od.memInitialPages}/${od.memMaxPages}!=${EXPECTED_WASM_PAGES}/${EXPECTED_WASM_PAGES}`);
    }
    if (od.memoryGrowCount > 0) errors.push(`objdumpMemoryGrowOps=${od.memoryGrowCount}>0`);
    if (od.callIndirectCount > 0) errors.push(`objdumpCallIndirectOps=${od.callIndirectCount}>0`);
    if ((od.allocatorReachableCount || 0) > 0 || (od.allocatorCallCount || 0) > 0) {
      errors.push(`objdumpForbiddenAllocator(reachable=${od.allocatorReachableCount || 0},calls=${od.allocatorCallCount || 0})`);
    }
    if (od.stackPointerInit === null || od.stackPointerInit > 16384 || od.stackPointerInit < 4096) {
      errors.push(`objdumpStackPointerInit=${od.stackPointerInit} outside [4096,16384]`);
    }
    if (od.stackCycles > 0) errors.push(`objdumpCallGraphCycles=${od.stackCycles}>0`);
    const stackLimit = Math.min(14336, od.stackPointerInit !== null ? od.stackPointerInit - 256 : 14336);
    if (od.maxShadowStackDepth > stackLimit) {
      errors.push(`objdumpShadowStackDepth=${od.maxShadowStackDepth}>${stackLimit}`);
    }
    const maxDataEnd = Math.max(od.maxDataEnd || 0, bm?.maxDataEnd || 0);
    const staticReserve = EXPECTED_WASM_MEM_BYTES - maxDataEnd;
    if (staticReserve < 65536) {
      errors.push(`wasmStaticReserve=${staticReserve}<65536 (maxDataEnd=${maxDataEnd})`);
    }
    let linkedStaticEnd = 0;
    let ring0PlusArena = 0;
    if (wasm.exp && typeof wasm.exp.chronos_init === 'function') {
      wasm.exp.chronos_init();
      const telemP = typeof wasm.exp.chronos_get_telemetry_ptr === 'function' ? wasm.exp.chronos_get_telemetry_ptr() : 0;
      const arenaB = telemP > 0 ? Math.round(new Float32Array(wasm.mem.buffer, telemP, TELEMETRY_LEN)[20]) : 0;
      const ring0 = typeof wasm.exp.chronos_get_ring_slot_ptr === 'function' ? wasm.exp.chronos_get_ring_slot_ptr(0) : 0;
      if (ring0 > 0 && arenaB > 0) ring0PlusArena = ring0 + arenaB;
      if (typeof wasm.exp.chronos_get_wave1_telemetry_ptr === 'function') {
        const w1P = wasm.exp.chronos_get_wave1_telemetry_ptr();
        if (w1P > 0) linkedStaticEnd = new Uint32Array(wasm.mem.buffer, w1P, 192)[12] >>> 0;
      }
    }
    const wave1StaticEnd = Math.max(maxDataEnd, ring0PlusArena, linkedStaticEnd);
    if (wave1StaticEnd > WAVE1_MAX_STATIC_END) {
      errors.push(`wave1StaticEnd=${wave1StaticEnd}>${WAVE1_MAX_STATIC_END}`);
    }
    if (od.v128Count < 20) errors.push(`objdumpSimdV128Count=${od.v128Count}<20`);
    if (od.clzCount < 1) errors.push(`objdumpI32ClzCount=${od.clzCount}<1`);
  }
  return errors;
}

function validatePointerLayout(memBytes, ranges, minDataEnd = 0) {
  const errors = [];
  const valid = [];
  for (const r of ranges) {
    if (!Number.isInteger(r.ptr) || r.ptr <= 0) {
      errors.push(`null/invalid ptr for ${r.name}: ${r.ptr}`);
      continue;
    }
    if (minDataEnd > 0 && r.ptr < minDataEnd) {
      errors.push(`buffer ${r.name} ptr=${r.ptr} overlaps static .rodata/.data end=${minDataEnd}`);
    }
    if (r.align && r.ptr % r.align !== 0) {
      errors.push(`unaligned ptr for ${r.name}: 0x${r.ptr.toString(16)} (mod ${r.align} != 0)`);
    }
    if (r.ptr + r.bytes > memBytes) {
      errors.push(`out-of-bounds ${r.name}: [${r.ptr}, ${r.ptr + r.bytes}) > ${memBytes}`);
      continue;
    }
    valid.push(r);
  }
  valid.sort((a, b) => a.ptr - b.ptr);
  for (let i = 1; i < valid.length; i++) {
    const prev = valid[i - 1];
    const cur = valid[i];
    if (prev.ptr + prev.bytes > cur.ptr) {
      errors.push(
        `overlapping buffers ${prev.name}=[${prev.ptr},${prev.ptr + prev.bytes}) and ${cur.name}=[${cur.ptr},${cur.ptr + cur.bytes})`
      );
    }
  }
  return errors;
}

function setupHdSession(exp, mem, swarmMode = 2, resMode = 1, staticDataEnd = 0) {
  exp.chronos_init();
  exp.chronos_set_resolution(resMode); // 1 = 640x360 HD, 0 = 320x180 SD
  const expectedW = resMode === 1 ? HD_W : 320;
  const expectedH = resMode === 1 ? HD_H : 180;
  const w = exp.chronos_get_frame_w();
  const h = exp.chronos_get_frame_h();
  if (w !== expectedW || h !== expectedH) {
    throw new Error(`Expected ${expectedW}x${expectedH} (res=${resMode}), got ${w}x${h}`);
  }
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 0);

  // Verify swarm mode set/get round-trip before running session
  for (const m of [0, 1, 2]) {
    exp.chronos_set_swarm_mode(m);
    const got = exp.chronos_get_swarm_mode();
    if (got !== m) {
      throw new Error(`chronos_get_swarm_mode round-trip mismatch: set ${m}, got ${got}`);
    }
  }
  exp.chronos_set_swarm_mode(swarmMode);

  const frameBytes = w * h * 4;
  const telemPtr = exp.chronos_get_telemetry_ptr();
  const lapPtr = exp.chronos_get_laplacian_ptr();
  const fiedlerPtr = exp.chronos_get_fiedler_ptr();
  const microPtr = exp.chronos_get_micro_tracks_ptr();
  const fieldPtr = exp.chronos_get_continuum_field_ptr();
  const bodiesPtr = exp.chronos_get_rigid_bodies_ptr();
  const swarmPtr = exp.chronos_get_swarm_telemetry_ptr();
  const modalPtr = exp.chronos_get_modal_field_ptr();
  const ringPtrs = Array.from({ length: 12 }, (_, slot) => exp.chronos_get_ring_slot_ptr(slot));

  const ptrRanges = [
    { name: 'telemetry', ptr: telemPtr, bytes: TELEMETRY_LEN * 4, align: 4 },
    { name: 'laplacian', ptr: lapPtr, bytes: 1024 * 4, align: 4 },
    { name: 'fiedler', ptr: fiedlerPtr, bytes: 32 * 4, align: 4 },
    { name: 'micro_tracks', ptr: microPtr, bytes: MAX_MICRO * MICRO_STRIDE * 4, align: 4 },
    { name: 'continuum_field', ptr: fieldPtr, bytes: FIELD_CELLS * FIELD_STRIDE * 4, align: 4 },
    { name: 'rigid_bodies', ptr: bodiesPtr, bytes: MAX_RIGID_BODIES * BODY_STRIDE * 4, align: 4 },
    { name: 'swarm_telemetry', ptr: swarmPtr, bytes: SWARM_TELEM_LEN * 4, align: 4 },
    { name: 'modal_field', ptr: modalPtr, bytes: FIELD_CELLS * 4, align: 4 },
    ...ringPtrs.map((ptr, slot) => ({ name: `ring_${slot}`, ptr, bytes: FULL_RING_SLOT_BYTES, align: 4 })),
  ];
  let w1TelemPtr = 0;
  if (typeof exp.chronos_get_ring_meta_ptr === 'function') {
    ptrRanges.push({ name: 'ring_meta', ptr: exp.chronos_get_ring_meta_ptr(), bytes: 12 * 24 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_row_luma_ptr === 'function') {
    ptrRanges.push({ name: 'row_luma_means', ptr: exp.chronos_get_row_luma_ptr(), bytes: 12 * 360 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_wave1_telemetry_ptr === 'function') {
    w1TelemPtr = exp.chronos_get_wave1_telemetry_ptr();
    ptrRanges.push({ name: 'wave1_telem', ptr: w1TelemPtr, bytes: 192 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_micro_cov_ptr === 'function') {
    ptrRanges.push({ name: 'micro_cov', ptr: exp.chronos_get_micro_cov_ptr(), bytes: 3 * MAX_MICRO * 4, align: 4 });
  }
  if (typeof exp.chronos_get_micro_residual_ptr === 'function') {
    ptrRanges.push({ name: 'micro_residual', ptr: exp.chronos_get_micro_residual_ptr(), bytes: 3 * MAX_MICRO * 4, align: 4 });
  }
  if (typeof exp.chronos_get_micro_loom_ptr === 'function') {
    ptrRanges.push({ name: 'micro_loom', ptr: exp.chronos_get_micro_loom_ptr(), bytes: MAX_MICRO * 4, align: 4 });
  }
  if (typeof exp.chronos_get_micro_quality_ptr === 'function') {
    ptrRanges.push({ name: 'micro_quality', ptr: exp.chronos_get_micro_quality_ptr(), bytes: MAX_MICRO * 4, align: 4 });
  }
  if (typeof exp.chronos_get_hero_candidates_ptr === 'function') {
    ptrRanges.push({ name: 'hero_candidates', ptr: exp.chronos_get_hero_candidates_ptr(), bytes: (4 + 16 * 8) * 4, align: 4 });
  }
  if (typeof exp.chronos_get_bos_telemetry_ptr === 'function') {
    ptrRanges.push({ name: 'bos_telemetry', ptr: exp.chronos_get_bos_telemetry_ptr(), bytes: 32 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_bos_planes_ptr === 'function') {
    ptrRanges.push({ name: 'bos_planes', ptr: exp.chronos_get_bos_planes_ptr(), bytes: 6 * 32768 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_bos_valid_ptr === 'function') {
    ptrRanges.push({ name: 'bos_valid', ptr: exp.chronos_get_bos_valid_ptr(), bytes: 32768, align: 4 });
  }
  if (typeof exp.chronos_get_bos_rgba_ptr === 'function') {
    ptrRanges.push({ name: 'bos_rgba', ptr: exp.chronos_get_bos_rgba_ptr(), bytes: 4 * 32768, align: 4 });
  }
  if (typeof exp.chronos_get_wave2_telemetry_ptr === 'function') {
    ptrRanges.push({ name: 'wave2_telem', ptr: exp.chronos_get_wave2_telemetry_ptr(), bytes: 64 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_audio_staging_ptr === 'function') {
    ptrRanges.push({ name: 'audio_staging', ptr: exp.chronos_get_audio_staging_ptr(), bytes: 4096 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_vibro_telemetry_ptr === 'function') {
    ptrRanges.push({ name: 'vibro_telem', ptr: exp.chronos_get_vibro_telemetry_ptr(), bytes: 32 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_vibro_planes_ptr === 'function') {
    ptrRanges.push({ name: 'vibro_planes', ptr: exp.chronos_get_vibro_planes_ptr(), bytes: (3 * FIELD_CELLS + 128) * 4, align: 4 });
  }
  if (typeof exp.chronos_get_hodge_telemetry_ptr === 'function') {
    ptrRanges.push({ name: 'hodge_telem', ptr: exp.chronos_get_hodge_telemetry_ptr(), bytes: 32 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_hodge_planes_ptr === 'function') {
    ptrRanges.push({ name: 'hodge_planes', ptr: exp.chronos_get_hodge_planes_ptr(), bytes: 7 * FIELD_CELLS * 4, align: 4 });
  }
  if (typeof exp.chronos_get_hodge_inject_ptr === 'function') {
    ptrRanges.push({ name: 'hodge_inject', ptr: exp.chronos_get_hodge_inject_ptr(), bytes: 3 * FIELD_CELLS * 4, align: 4 });
  }
  if (typeof exp.chronos_get_conservation_telemetry_ptr === 'function') {
    ptrRanges.push({ name: 'conservation_telem', ptr: exp.chronos_get_conservation_telemetry_ptr(), bytes: 32 * 4, align: 4 });
  }
  if (typeof exp.chronos_get_conservation_input_ptr === 'function') {
    ptrRanges.push({ name: 'conservation_input', ptr: exp.chronos_get_conservation_input_ptr(), bytes: 32 * 4, align: 4 });
  }

  const ptrErrors = validatePointerLayout(mem.buffer.byteLength, ptrRanges, staticDataEnd);
  if (ptrErrors.length > 0) {
    throw new Error(`Wasm C-ABI pointer layout violation: ${ptrErrors.join('; ')}`);
  }

  const telem = new Float32Array(mem.buffer, telemPtr, TELEMETRY_LEN);
  const rawArena = telem[20];
  if (!Number.isFinite(rawArena)) {
    throw new Error(`Non-finite arena size telem[20]=${rawArena}`);
  }
  const arenaBytes = Math.round(rawArena);
  if (arenaBytes < 17_000_000 || arenaBytes > MAX_ARENA_BYTES) {
    throw new Error(`Arena size ${arenaBytes} B outside Wave-1 [17000000, ${MAX_ARENA_BYTES}]`);
  }
  const linkedEnd = w1TelemPtr > 0 ? (new Uint32Array(mem.buffer, w1TelemPtr, 192)[12] >>> 0) : 0;
  const dualStaticEnd = Math.max(staticDataEnd, ringPtrs[0] + arenaBytes, linkedEnd);
  if (dualStaticEnd > WAVE1_MAX_STATIC_END || dualStaticEnd > EXPECTED_WASM_MEM_BYTES - 65536) {
    throw new Error(
      `Dual static-end breach: max(segment=${staticDataEnd}, ring0+arena=${ringPtrs[0] + arenaBytes}, __heap_base=${linkedEnd})=${dualStaticEnd} > ${WAVE1_MAX_STATIC_END}`
    );
  }

  const laplacian = new Float32Array(mem.buffer, lapPtr, 1024);
  const fiedler = new Float32Array(mem.buffer, fiedlerPtr, 32);
  const ringViews = ringPtrs.map((ptr) => new Uint8Array(mem.buffer, ptr, frameBytes));
  const microTracks = new Float32Array(mem.buffer, microPtr, MAX_MICRO * MICRO_STRIDE);
  const continuumField = new Float32Array(mem.buffer, fieldPtr, FIELD_CELLS * FIELD_STRIDE);
  const rigidBodies = new Float32Array(mem.buffer, bodiesPtr, MAX_RIGID_BODIES * BODY_STRIDE);
  const swarmTelem = new Float32Array(mem.buffer, swarmPtr, SWARM_TELEM_LEN);
  const modalField = new Float32Array(mem.buffer, modalPtr, FIELD_CELLS);
  const solveLanczos = createLanczosSolver(exp, laplacian, fiedler);

  // Pre-frame idle state: chronos_inspect_field_point must return 0 and write [11]=0,[12]=0
  for (const [qx, qy] of [
    [w * 0.5, h * 0.5],
    [w * 0.5, h - 2.0],
    [-10.0, -10.0],
    [w + 60.0, h + 40.0],
  ]) {
    const tidHit = exp.chronos_inspect_field_point(qx, qy);
    if (tidHit !== 0) {
      throw new Error(`Expected chronos_inspect_field_point(${qx}, ${qy}) === 0 before tracks exist, got ${tidHit}`);
    }
    if (swarmTelem[11] !== 0.0 || swarmTelem[12] !== 0.0) {
      throw new Error(`Expected swarmTelem[11..12]==[0,0] on miss, got [${swarmTelem[11]},${swarmTelem[12]}]`);
    }
    for (let k = 8; k < 16; k++) {
      if (!Number.isFinite(swarmTelem[k])) {
        throw new Error(`Non-finite swarmTelem[${k}] after chronos_inspect_field_point(${qx}, ${qy})`);
      }
    }
  }

  return {
    w,
    h,
    arenaBytes,
    telem,
    ringViews,
    microTracks,
    continuumField,
    rigidBodies,
    swarmTelem,
    modalField,
    solveLanczos,
  };
}

/**
 * Behavioral oracle for chronos_inspect_field_point on populated frames:
 * verifies hit matching, swarmTelem[11..12] track_id/body_id writes, empty miss=0, and OOB miss=0.
 */
/**
 * Behavioral oracle for locked chronos_inspect_field_point(x, y) -> u32 ABI:
 *   - Finite 0 <= x < w, 0 <= y < h: matches nearest CONFIRMED (state === 1) micro-track strictly < 24 * q px;
 *     writes swarmTelem[11] = micro_tid, swarmTelem[12] = body_id, promotes/binds Hero pin (or clears [11..12] on miss).
 *   - Out-of-bounds or non-finite: MUST return 0 WITHOUT mutating swarmTelem[11..12] or chronos_get_focus_target().
 */
function verifyPopulatedInspector(exp, w, h, confirmedMicro, swarmTelem) {
  const errors = [];
  if (confirmedMicro.length === 0) {
    errors.push('inspectorEmptyConfirmedMicro');
    return errors;
  }
  const q = w / 640.0;
  const maxR = 24.0 * q;
  const maxRSq = maxR * maxR;

  const target =
    confirmedMicro.find((t) => t.bodyId > 0 && t.x >= 20 && t.x <= w - 20 && t.y >= 20 && t.y <= h - 20) ||
    confirmedMicro.find((t) => t.x >= 20 && t.x <= w - 20 && t.y >= 20 && t.y <= h - 20) ||
    confirmedMicro[0];

  const qx = Math.max(0.0, Math.min(w - 0.05, target.x + 0.25));
  const qy = Math.max(0.0, Math.min(h - 0.05, target.y - 0.25));
  const gotHit = exp.chronos_inspect_field_point(qx, qy);
  let nearest = null;
  let bestD2 = maxRSq;
  for (const t of confirmedMicro) {
    const dx = t.x - qx;
    const dy = t.y - qy;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      nearest = t;
    }
  }
  if (!nearest || gotHit !== nearest.tid) {
    errors.push(`inspectorHitMismatch(got=${gotHit}, expectedNearestTid=${nearest?.tid ?? 'NONE'})`);
    return errors;
  }
  if (swarmTelem[11] !== nearest.tid) {
    errors.push(`inspectorTelem11=${swarmTelem[11]}!==${nearest.tid}`);
  }
  if (swarmTelem[12] !== nearest.bodyId) {
    errors.push(`inspectorTelem12=${swarmTelem[12]}!==${nearest.bodyId}`);
  }

  const hasFocusApi =
    typeof exp.chronos_get_focus_target === 'function' && typeof exp.chronos_set_focus_target === 'function';
  const focusAfterHit = hasFocusApi ? exp.chronos_get_focus_target() : null;
  if (hasFocusApi) {
    if (!Number.isInteger(focusAfterHit) || focusAfterHit <= 0) {
      errors.push(`inspectorFocusNotPromotedAfterHit(tid=${nearest.tid},focus=${focusAfterHit})`);
    } else if (nearest.tid > 0) {
      exp.chronos_set_focus_target(nearest.tid);
      const pinnedHero = exp.chronos_get_focus_target();
      if (pinnedHero !== focusAfterHit) {
        errors.push(`inspectorHeroPinBridge(${nearest.tid})=>${pinnedHero}!=${focusAfterHit}`);
      }
    }
  }

  // OOB / non-finite queries MUST return 0 WITHOUT mutating swarmTelem[11..12] or focus_target_id (exact equality)
  const oobBefore11 = swarmTelem[11];
  const oobBefore12 = swarmTelem[12];
  for (const [ox, oy] of [
    [-12.0, -12.0],
    [-0.01, h * 0.5],
    [w, 10.0],
    [10.0, h],
    [w + 48.0, h + 48.0],
    [NaN, 10.0],
    [10.0, Infinity],
    [-Infinity, 10.0],
  ]) {
    const oobRet = exp.chronos_inspect_field_point(ox, oy);
    const focusNow = hasFocusApi ? exp.chronos_get_focus_target() : null;
    if (
      oobRet !== 0 ||
      swarmTelem[11] !== oobBefore11 ||
      swarmTelem[12] !== oobBefore12 ||
      swarmTelem[11] !== nearest.tid ||
      swarmTelem[12] !== nearest.bodyId ||
      (focusAfterHit !== null && focusNow !== focusAfterHit)
    ) {
      errors.push(
        `inspectorOobMutatedOrNonZero(${ox},${oy}) ret=${oobRet} [11..12]=[${swarmTelem[11]},${swarmTelem[12]}] expected=[${nearest.tid},${nearest.bodyId}] focus=${focusNow}/${focusAfterHit}`
      );
      break;
    }
  }

  // Verify exact in-bounds (0, 0) lower boundary returns valid deterministic match or miss
  {
    const cornerRet = exp.chronos_inspect_field_point(0.0, 0.0);
    let cornerNearest = null;
    let cornerBest = maxRSq;
    for (const t of confirmedMicro) {
      const d2 = t.x * t.x + t.y * t.y;
      if (d2 < cornerBest) {
        cornerBest = d2;
        cornerNearest = t;
      }
    }
    const expCornerTid = cornerNearest ? cornerNearest.tid : 0;
    const expCornerBid = cornerNearest ? cornerNearest.bodyId : 0;
    if (cornerRet !== expCornerTid || swarmTelem[11] !== expCornerTid || swarmTelem[12] !== expCornerBid) {
      errors.push(`inspectorOriginMismatch ret=${cornerRet} exp=${expCornerTid} [11..12]=[${swarmTelem[11]},${swarmTelem[12]}]`);
    }
  }

  // Re-establish hit and verify far-from-any-track in-bounds point clears [11..12] to [0, 0]
  exp.chronos_inspect_field_point(qx, qy);
  let emptyPt = null;
  for (const [cx, cy] of [
    [w - 0.5, 0.5],
    [0.5, h - 0.5],
    [w - 0.5, h - 0.5],
    [4.0, 4.0],
    [w * 0.5, h - 1.0],
  ]) {
    const minDist = confirmedMicro.reduce((mn, t) => Math.min(mn, Math.hypot(t.x - cx, t.y - cy)), Infinity);
    if (minDist >= maxR + 2.0) {
      emptyPt = [cx, cy];
      break;
    }
  }
  if (emptyPt) {
    const missRet = exp.chronos_inspect_field_point(emptyPt[0], emptyPt[1]);
    if (missRet !== 0 || swarmTelem[11] !== 0.0 || swarmTelem[12] !== 0.0) {
      errors.push(`inspectorEmptySpotMiss(${emptyPt[0]},${emptyPt[1]}) ret=${missRet} [11]=${swarmTelem[11]} [12]=${swarmTelem[12]}`);
    }
  }
  if (hasFocusApi) exp.chronos_set_focus_target(0);
  return errors;
}

function inspectFrameBuffers(session) {
  let nanCount = 0;
  let reservedTailNonZero = 0;
  let rangeViolations = 0;
  if (session.arenaBytes > 0 && Math.round(session.telem[20]) !== session.arenaBytes) {
    rangeViolations++;
  }
  for (let i = 0; i < session.telem.length; i++) {
    if (!Number.isFinite(session.telem[i])) nanCount++;
  }
  for (let i = TELEM_RESERVED_TAIL_START; i < TELEMETRY_LEN; i++) {
    if (session.telem[i] !== 0.0) reservedTailNonZero++;
  }
  for (let i = 0; i < session.swarmTelem.length; i++) {
    if (!Number.isFinite(session.swarmTelem[i])) nanCount++;
  }
  for (let i = SWARM_RESERVED_TAIL_START; i < SWARM_TELEM_LEN; i++) {
    if (session.swarmTelem[i] !== 0.0) reservedTailNonZero++;
  }

  const swarmActive = session.swarmTelem[1] > 0.5;
  const activeMicroCnt = Math.round(session.swarmTelem[3]);
  const ftleMatureTracks = session.swarmTelem[39];
  if (
    !Number.isFinite(ftleMatureTracks) ||
    !Number.isInteger(ftleMatureTracks) ||
    ftleMatureTracks < 0 ||
    ftleMatureTracks > activeMicroCnt ||
    (!swarmActive && ftleMatureTracks !== 0)
  ) {
    rangeViolations++;
  }
  // swarmTelem[41] = modal_nematic_r2 in [0, 1]
  const r2 = session.swarmTelem[41];
  if (!Number.isFinite(r2) || r2 < -1e-4 || r2 > 1.0 + 1e-4) rangeViolations++;

  for (let i = 0; i < MAX_MICRO; i++) {
    const mb = i * MICRO_STRIDE;
    for (let k = 0; k < MICRO_STRIDE; k++) {
      if (!Number.isFinite(session.microTracks[mb + k])) nanCount++;
    }
    const st = session.microTracks[mb + 5];
    if (st === 1 || st === 2) {
      const ft = session.microTracks[mb + 7];
      if (!Number.isFinite(ft) || (ft <= -900.0 && ft !== -999.0)) rangeViolations++;
    }
  }
  for (let c = 0; c < FIELD_CELLS; c++) {
    const fb = c * FIELD_STRIDE;
    for (let k = 0; k < FIELD_STRIDE; k++) {
      if (!Number.isFinite(session.continuumField[fb + k])) nanCount++;
    }
    const w = session.continuumField[fb + 5];
    if (!Number.isFinite(w) || w < 0.0) {
      rangeViolations++;
    } else if (w === 0.0) {
      if (
        session.continuumField[fb + 0] !== 0.0 ||
        session.continuumField[fb + 1] !== 0.0 ||
        session.continuumField[fb + 2] !== 0.0 ||
        session.continuumField[fb + 3] !== 0.0
      ) {
        rangeViolations++;
      }
    }
    const ftle = session.continuumField[fb + 4];
    if (!Number.isFinite(ftle)) {
      rangeViolations++;
    } else if (swarmActive) {
      if (ftle <= -900.0 && ftle !== -999.0) rangeViolations++;
    } else if (ftle !== 0.0) {
      rangeViolations++;
    }
  }
  for (let i = 0; i < session.rigidBodies.length; i++) {
    if (!Number.isFinite(session.rigidBodies[i])) nanCount++;
  }
  for (let c = 0; c < FIELD_CELLS; c++) {
    const ph = session.modalField[c];
    if (!Number.isFinite(ph)) {
      nanCount++;
    } else if (!swarmActive) {
      if (ph !== 0.0) rangeViolations++;
    } else if (ph !== -9.0 && (ph < -3.15 || ph > 3.15)) {
      rangeViolations++;
    }
  }
  return { nanCount, reservedTailNonZero, rangeViolations };
}

function readMicroTracksStrict(microTracks) {
  const observed = [];
  const coasted = [];
  const seenTids = new Set();
  let duplicateTidErrors = 0;
  let invalidEncodingErrors = 0;

  for (let i = 0; i < MAX_MICRO; i++) {
    const base = i * MICRO_STRIDE;
    const rawState = microTracks[base + 5];
    if (!Number.isInteger(rawState) || rawState < 0 || rawState > 2) {
      invalidEncodingErrors++;
      continue;
    }
    if (rawState === 0) continue;

    const rawTid = microTracks[base + 0];
    const x = microTracks[base + 1];
    const y = microTracks[base + 2];
    const vx = microTracks[base + 3];
    const vy = microTracks[base + 4];
    const rawBodyId = microTracks[base + 6];
    const ftle = microTracks[base + 7];

    if (
      !Number.isInteger(rawTid) ||
      rawTid <= 0 ||
      !Number.isInteger(rawBodyId) ||
      rawBodyId < 0 ||
      rawBodyId > MAX_RIGID_BODIES ||
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(vx) ||
      !Number.isFinite(vy) ||
      !Number.isFinite(ftle) ||
      x < -64.0 ||
      x > HD_W + 64.0 ||
      y < -64.0 ||
      y > HD_H + 64.0
    ) {
      invalidEncodingErrors++;
      continue;
    }

    if (seenTids.has(rawTid)) {
      duplicateTidErrors++;
      continue;
    }
    seenTids.add(rawTid);

    const entry = {
      tid: rawTid,
      x,
      y,
      vx,
      vy,
      state: rawState,
      bodyId: rawBodyId,
      ftle,
    };
    if (rawState === 1) observed.push(entry);
    else coasted.push(entry);
  }
  return { observed, coasted, duplicateTidErrors, invalidEncodingErrors };
}

function readActiveRigidBodies(rigidBodies) {
  const out = [];
  let invalidEncodingErrors = 0;
  const seenBodyIds = new Set();
  for (let i = 0; i < MAX_RIGID_BODIES; i++) {
    const base = i * BODY_STRIDE;
    const rawId = rigidBodies[base + 0];
    if (!Number.isInteger(rawId) || rawId < 0 || rawId > MAX_RIGID_BODIES) {
      invalidEncodingErrors++;
      continue;
    }
    if (rawId === 0) continue;
    if (seenBodyIds.has(rawId)) {
      invalidEncodingErrors++;
      continue;
    }
    seenBodyIds.add(rawId);
    const rawMembers = rigidBodies[base + 1];
    if (!Number.isInteger(rawMembers) || rawMembers < 0) invalidEncodingErrors++;
    const se2Rmse = rigidBodies[base + 7];
    if (!Number.isFinite(se2Rmse) || se2Rmse < 0.0) invalidEncodingErrors++;
    out.push({
      bodyId: rawId,
      memberCount: rawMembers,
      cx: rigidBodies[base + 2],
      cy: rigidBodies[base + 3],
      vx: rigidBodies[base + 4],
      vy: rigidBodies[base + 5],
      omega: rigidBodies[base + 6],
      se2Rmse,
    });
  }
  out.sort((a, b) => b.memberCount - a.memberCount);
  return { bodies: out, invalidEncodingErrors };
}

function formatNullable(v, decimals = 1, suffix = '') {
  return v === null || v === undefined || !Number.isFinite(v) ? 'N/A' : `${v.toFixed(decimals)}${suffix}`;
}

function logDiagnostics(tag, stats) {
  const heroLookaheadMs = stats.effK * stats.dtSec * 1000.0;
  console.log(`\n[${tag}] (${stats.numFrames} frames @ ${stats.fps}fps, ${HD_W}x${HD_H})`);
  console.log(
    `  Kernel Compute: warmMean=${stats.warmMeanMs.toFixed(3)} ms | warmP50=${stats.warmMedianMs.toFixed(3)} ms (<${SPEED_BUDGET_MS.toFixed(2)} ms) | p95=${stats.p95Ms.toFixed(3)} ms (<${TAIL_P95_BUDGET_MS.toFixed(2)} ms) | p99=${stats.p99Ms.toFixed(3)} ms | rawMean=${stats.meanMs.toFixed(3)} ms`
  );
  console.log(
    `  Swarm Timing  : causalReporting=0f delay (synthetic headless loop; kernelP50=${stats.warmMedianMs.toFixed(3)} ms excludes decode/copy/render) | Hero16 effK=${stats.effK}f (${heroLookaheadMs.toFixed(1)} ms lookahead, heroActive=${stats.heroActive} [min=${stats.heroActiveMin}, mean=${stats.heroActiveMean.toFixed(1)}]) | NaNs=${stats.nanCount} | tailZeros=${stats.tailNonZero === 0 ? 'OK' : `FAIL(${stats.tailNonZero})`}`
  );
  console.log(
    `  Micro Swarm   : peakObs=${stats.peakConf} | meanObs=${stats.meanObs.toFixed(1)} | p10Obs=${stats.p10Obs} | lastObs=${stats.lastConf} | peakDet=${stats.maxPeaks} | uniqueTids=${stats.uniqueTids} | coasting=${stats.lastCoasting} | dupTid=${stats.dupTidErrors} | encErr=${stats.encErrors} | inspErr=${stats.inspectorErrors} | icLkRes=${stats.lastIcLkRes.toFixed(3)}`
  );
  if (stats.mot) {
    const m = stats.mot;
    const hotaStr = m.hota !== null ? `${(m.hota * 100).toFixed(1)}% (DetA=${(m.detA * 100).toFixed(1)}%, AssA=${(m.assA * 100).toFixed(1)}%, LocA=${(m.locA * 100).toFixed(1)}%)` : 'N/A';
    console.log(
      `  CLEAR-MOT     : Recall=${formatNullable(m.recallPct, 1, '%')} (${m.gtMatched}/${m.gtTotal}) | Prec=${formatNullable(m.precisionPct, 1, '%')} (FP=${m.fp}) | MOTA=${formatNullable(m.motaPct, 1, '%')} | RMSE(MOTP)=${formatNullable(m.rmsePx, 2, ' px')} (${formatNullable(m.motpPx, 2, ' px')}) | HOTA=${hotaStr} | IDSW=${m.idsw} | Frag=${m.frag}`
    );
  }
  console.log(
    `  Validity Mask : ${decodeValidityMask(stats.validityMask)} (raw=${stats.validityMask}, validCells=${stats.validCells}/${FIELD_CELLS})`
  );
  console.log(
    `  Continuum Raw : vortMax=${stats.maxVort.toFixed(3)}/s vortRms=${stats.rmsVort.toFixed(3)}/s divRms=${stats.rmsDiv.toFixed(3)}/s | ftleMax=${stats.maxFtle.toFixed(3)}/s (${stats.ftleCells} cells) | strain=${stats.meanStrain.toFixed(3)}/s`
  );
  console.log(
    `  Rigid/Modal   : rigidCnt=${stats.rigidCnt} topOmega=${stats.topOmega.toFixed(3)} rad/s (N=${stats.topMembers}, se2Rmse=${stats.topSe2Rmse.toFixed(3)} px) | modalHz=${stats.modalHz.toFixed(3)} Hz (coh=${stats.modalCoh.toFixed(3)}, sensors=${stats.modalSensors}, R2=${stats.modalR2.toFixed(3)})`
  );
  console.log(
    `  Kuramoto/CoM  : R=${stats.kuramotoR.toFixed(3)} psi=${stats.kuramotoPsi.toFixed(2)} rad | com=(${stats.comX.toFixed(1)},${stats.comY.toFixed(1)}) vel=(${stats.comVx.toFixed(1)},${stats.comVy.toFixed(1)}) disp=${stats.velDisp.toFixed(1)} px/s`
  );
}

function streamMp4Frames(mp4Path, w, h, onFrame) {
  const frameBytes = w * h * 4;
  const frameScratch = new Uint8Array(frameBytes);
  let carry = 0;

  return new Promise((resolve, reject) => {
    const proc = spawn(
      'ffmpeg',
      ['-v', 'error', '-threads', '2', '-i', mp4Path, '-filter_threads', '1', '-threads', '1', '-vf', `scale=${w}:${h}`, '-pix_fmt', 'rgba', '-f', 'rawvideo', 'pipe:1'],
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
      if (code !== 0) reject(new Error(`ffmpeg exited with code ${code}: ${stderrText}`));
      else if (carry !== 0) {
        reject(new Error(`ffmpeg stream truncated: ${carry} trailing bytes (expected multiple of ${frameBytes})`));
      } else {
        resolve({ numFrames: frameCount });
      }
    });
  });
}

// ============================================================================
// Gate Runners: Q1, Q2, Q3, Q4
// ============================================================================

function computeKabschSe2(pairs) {
  const n = pairs.length;
  if (n < 3) return null;
  let pxMean = 0.0;
  let pyMean = 0.0;
  let qxMean = 0.0;
  let qyMean = 0.0;
  for (let i = 0; i < n; i++) {
    pxMean += pairs[i].px;
    pyMean += pairs[i].py;
    qxMean += pairs[i].qx;
    qyMean += pairs[i].qy;
  }
  pxMean /= n;
  pyMean /= n;
  qxMean /= n;
  qyMean /= n;

  let cxx = 0.0;
  let cxy = 0.0;
  for (let i = 0; i < n; i++) {
    const dpx = pairs[i].px - pxMean;
    const dpy = pairs[i].py - pyMean;
    const dqx = pairs[i].qx - qxMean;
    const dqy = pairs[i].qy - qyMean;
    // Closed-form proper det=+1 2D Kabsch / Procrustes
    cxx += dpx * dqx + dpy * dqy;
    cxy += dpx * dqy - dpy * dqx;
  }
  const dTheta = Math.atan2(cxy, cxx);
  const ct = Math.cos(dTheta);
  const st = Math.sin(dTheta);
  let errSq = 0.0;
  for (let i = 0; i < n; i++) {
    const dpx = pairs[i].px - pxMean;
    const dpy = pairs[i].py - pyMean;
    const rx = qxMean + (dpx * ct - dpy * st);
    const ry = qyMean + (dpx * st + dpy * ct);
    const ex = pairs[i].qx - rx;
    const ey = pairs[i].qy - ry;
    errSq += ex * ex + ey * ey;
  }
  return { dTheta, rmsePx: Math.sqrt(errSq / n), cx: qxMean, cy: qyMean };
}

function solveMinCostMatchAny(costMatrix, rows, cols) {
  if (rows === 0 || cols === 0) return [];
  const n = Math.max(rows, cols);
  const big = 1e9;
  const mat = new Float64Array(n * n);
  mat.fill(big);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) mat[r * n + c] = costMatrix[r * cols + c];
  }
  const rowToCol = solveHungarianMinCost(mat, n);
  const pairs = [];
  for (let r = 0; r < rows; r++) {
    const c = rowToCol[r];
    if (c >= 0 && c < cols && costMatrix[r * cols + c] < big * 0.5) {
      pairs.push({ r, c, cost: costMatrix[r * cols + c] });
    }
  }
  return pairs;
}

function summarizeRunStats(session, tickMs, warmTickMs, counters, extras) {
  const warmMedianMs = computeMedian(warmTickMs);
  const warmMeanMs = warmTickMs.reduce((a, b) => a + b, 0) / Math.max(1, warmTickMs.length);
  const meanMs = tickMs.reduce((a, b) => a + b, 0) / Math.max(1, tickMs.length);
  const p95Ms = computePercentile(warmTickMs, 95) ?? warmMedianMs;
  const p99Ms = computePercentile(warmTickMs, 99) ?? warmMedianMs;
  const st = session.swarmTelem;
  const obsArr = counters.steadyObsCounts || [];
  const meanObs = obsArr.length > 0 ? obsArr.reduce((a, b) => a + b, 0) / obsArr.length : 0.0;
  const p10Obs = obsArr.length > 0 ? computePercentile(obsArr, 10) : 0;
  const heroArr = counters.heroActiveCounts || [];
  const heroActiveMin = heroArr.length > 0 ? Math.min(...heroArr) : Math.round(session.telem[3]);
  const heroActiveMean = heroArr.length > 0 ? heroArr.reduce((a, b) => a + b, 0) / heroArr.length : Math.round(session.telem[3]);
  return {
    numFrames: tickMs.length,
    fps: extras.fps,
    dtSec: extras.dtSec,
    meanMs,
    warmMeanMs,
    warmMedianMs,
    p95Ms,
    p99Ms,
    nanCount: counters.nanCount,
    tailNonZero: counters.tailNonZero,
    rangeViolations: counters.rangeViolations,
    dupTidErrors: counters.dupTidErrors,
    encErrors: counters.encErrors,
    inspectorErrors: counters.inspectorErrors || 0,
    inspectorMsgs: counters.inspectorMsgs || [],
    peakConf: counters.peakConf,
    meanObs,
    p10Obs,
    maxPeaks: counters.maxPeaks,
    effK: Math.round(session.telem[1]),
    heroActive: Math.round(session.telem[3]),
    heroActiveMin,
    heroActiveMean,
    validityMask: Math.round(st[16]),
    validCells: Math.round(st[17]),
    lastConf: Math.round(st[4]),
    uniqueTids: Math.round(st[6]),
    lastCoasting: Math.round(st[7]),
    auctionSteps: Math.round(st[8]),
    reacqEvents: Math.round(st[9]),
    lastIcLkRes: st[10],
    ftleMatureTracks: Math.round(st[39]),
    maxVort: st[18],
    rmsVort: st[19],
    rmsDiv: st[20],
    maxFtle: st[21],
    ftleCells: Math.round(st[22]),
    rigidCnt: Math.round(st[23]),
    topOmega: st[24],
    topMembers: Math.round(st[25]),
    topSe2Rmse: st[26],
    modalHz: st[27],
    modalCoh: st[28],
    modalSensors: Math.round(st[29]),
    modalR2: st[41],
    kuramotoR: st[30],
    kuramotoPsi: st[31],
    comX: st[32],
    comY: st[33],
    comVx: st[34],
    comVy: st[35],
    velDisp: st[36],
    meanStrain: st[37],
    mot: extras.mot || null,
  };
}

async function runSyntheticWasmSequence(
  wasm,
  { numFrames, fps, dtSec, swarmMode = 2, resMode = 1, checkInspector = true, renderFrame, onFrame }
) {
  const staticDataEnd = wasm.objdumpReport?.maxDataEnd || 0;
  const session = setupHdSession(wasm.exp, wasm.mem, swarmMode, resMode, staticDataEnd);
  const rgba = new Uint8Array(session.w * session.h * 4);
  const tickMs = [];
  const warmTickMs = [];
  const counters = {
    nanCount: 0,
    tailNonZero: 0,
    rangeViolations: 0,
    dupTidErrors: 0,
    encErrors: 0,
    inspectorErrors: 0,
    inspectorMsgs: [],
    peakConf: 0,
    maxPeaks: 0,
    steadyObsCounts: [],
    heroActiveCounts: [],
  };
  const inspectFrame = Math.min(numFrames - 1, 25);
  const indepStreakMap = new Map();
  let lastStreakSwarmActive = false;

  for (let f = 0; f < numFrames; f++) {
    clearFrameDarkBg(rgba);
    renderFrame(rgba, f, session.w, session.h);

    session.ringViews[f % 12].set(rgba);
    if (typeof wasm.exp.chronos_set_ingest_pts === 'function') {
      wasm.exp.chronos_set_ingest_pts(f * dtSec);
    }

    const t0 = performance.now();
    wasm.exp.chronos_pre_tick(dtSec);
    session.solveLanczos();
    wasm.exp.chronos_post_tick(dtSec);
    const dtMs = performance.now() - t0;
    tickMs.push(dtMs);
    if (f > 0) warmTickMs.push(dtMs);

    const chk = inspectFrameBuffers(session);
    counters.nanCount += chk.nanCount;
    counters.tailNonZero += chk.reservedTailNonZero;
    counters.rangeViolations += chk.rangeViolations;

    const micro = readMicroTracksStrict(session.microTracks);
    const bodiesRes = readActiveRigidBodies(session.rigidBodies);
    counters.dupTidErrors += micro.duplicateTidErrors;
    counters.encErrors += micro.invalidEncodingErrors + bodiesRes.invalidEncodingErrors;

    const st = session.swarmTelem;
    const isSwarmActive = st[1] > 0.5;
    if (!isSwarmActive) {
      indepStreakMap.clear();
    } else {
      const nextStreakMap = new Map();
      let indepMature9Now = 0;
      const maxStepDist = 48.0 * (session.w / 640.0);
      for (let i = 0; i < micro.observed.length; i++) {
        const t = micro.observed[i];
        const prev = lastStreakSwarmActive ? indepStreakMap.get(t.tid) : null;
        const streak = prev && Math.hypot(t.x - prev.x, t.y - prev.y) <= maxStepDist ? prev.streak + 1 : 1;
        nextStreakMap.set(t.tid, { x: t.x, y: t.y, streak });
        if (streak >= 9) indepMature9Now++;
      }
      indepStreakMap.clear();
      for (const [k, v] of nextStreakMap.entries()) indepStreakMap.set(k, v);
      if (Number.isInteger(st[39]) && Math.round(st[39]) > indepMature9Now) {
        counters.rangeViolations++;
      }
    }
    lastStreakSwarmActive = isSwarmActive;

    if (checkInspector && f === inspectFrame && micro.observed.length > 0) {
      const iErrs = verifyPopulatedInspector(
        wasm.exp,
        session.w,
        session.h,
        micro.observed,
        session.swarmTelem
      );
      if (iErrs.length > 0) {
        counters.inspectorErrors += iErrs.length;
        counters.inspectorMsgs.push(...iErrs);
      }
    }

    counters.maxPeaks = Math.max(counters.maxPeaks, Math.round(st[2]));
    counters.peakConf = Math.max(counters.peakConf, micro.observed.length);
    if (f >= 5) counters.steadyObsCounts.push(micro.observed.length);
    if (f >= 10) counters.heroActiveCounts.push(Math.round(session.telem[3]));

    if (onFrame) {
      onFrame({
        f,
        dtMs,
        session,
        micro,
        bodies: bodiesRes.bodies,
        st,
        mask: Math.round(st[16]),
      });
    }
  }

  return { session, counters, tickMs, warmTickMs, fps, dtSec };
}

function appendCommonGateFailures(failures, stats, { minPeakConf, minMeanObs, minP10Obs, minHeroActive = 1, motReq }) {
  if (stats.nanCount > 0) failures.push(`nanCount=${stats.nanCount}`);
  if (stats.tailNonZero > 0) failures.push(`tailNonZero=${stats.tailNonZero}`);
  if (stats.rangeViolations > 0) failures.push(`rangeViolations=${stats.rangeViolations}`);
  if (stats.dupTidErrors > 0) failures.push(`duplicateTidErrors=${stats.dupTidErrors}`);
  if (stats.encErrors > 0) failures.push(`invalidEncodingErrors=${stats.encErrors}`);
  if (stats.inspectorErrors > 0) {
    failures.push(`inspectorOracle=${stats.inspectorMsgs.slice(0, 2).join('|') || stats.inspectorErrors}`);
  }
  if (minHeroActive > 0 && (stats.heroActiveMin < minHeroActive || stats.heroActiveMean < minHeroActive)) {
    failures.push(`heroActiveZeroOrWeak(min=${stats.heroActiveMin},mean=${stats.heroActiveMean.toFixed(1)}<${minHeroActive})`);
  }
  if (minPeakConf !== undefined && stats.peakConf < minPeakConf) {
    failures.push(`peakObsConf=${stats.peakConf}<${minPeakConf}`);
  }
  if (minMeanObs !== undefined && stats.meanObs < minMeanObs) {
    failures.push(`meanObs=${stats.meanObs.toFixed(1)}<${minMeanObs}`);
  }
  if (minP10Obs !== undefined && stats.p10Obs < minP10Obs) {
    failures.push(`p10Obs=${stats.p10Obs}<${minP10Obs}`);
  }
  if (motReq && stats.mot) {
    const m = stats.mot;
    if (m.duplicateTidCount > 0) failures.push(`motDupTid=${m.duplicateTidCount}`);
    if (m.recallPct === null || m.recallPct < motReq.recall) {
      failures.push(`recall=${formatNullable(m.recallPct, 1, '%')}<${motReq.recall}%`);
    }
    if (m.precisionPct === null || m.precisionPct < motReq.prec) {
      failures.push(`precision=${formatNullable(m.precisionPct, 1, '%')}<${motReq.prec}%`);
    }
    if (m.motaPct === null || m.motaPct < motReq.mota) {
      failures.push(`mota=${formatNullable(m.motaPct, 1, '%')}<${motReq.mota}%`);
    }
    if (m.rmsePx === null || m.rmsePx >= motReq.rmse) {
      failures.push(`rmse=${formatNullable(m.rmsePx, 3)}>=${motReq.rmse}`);
    }
    if (motReq.idsw !== undefined && m.idsw > motReq.idsw) {
      failures.push(`idsw=${m.idsw}>${motReq.idsw}`);
    }
    if (motReq.hota !== undefined && (m.hota === null || m.hota < motReq.hota)) {
      failures.push(`hota=${formatNullable(m.hota, 3)}<${motReq.hota}`);
    }
  }
  if (!noSpeedGate && !inAntiSpoofSelfTest) {
    if (stats.warmMedianMs >= SPEED_BUDGET_MS) {
      failures.push(`warmMedianMs=${stats.warmMedianMs.toFixed(3)}>=${SPEED_BUDGET_MS.toFixed(2)}`);
    }
    if (stats.warmMeanMs >= SPEED_BUDGET_MS) {
      failures.push(`warmMeanMs=${stats.warmMeanMs.toFixed(3)}>=${SPEED_BUDGET_MS.toFixed(2)}`);
    }
    if (stats.p95Ms >= TAIL_P95_BUDGET_MS) {
      failures.push(`p95Ms=${stats.p95Ms.toFixed(3)}>=${TAIL_P95_BUDGET_MS.toFixed(2)}`);
    }
  }
}

// ============================================================================
// Gate Runners: Q1, Q2, Q3, Q4
// ============================================================================

async function runQ1Vortex(wasm) {
  const scene = createQ1VortexScene();
  const motEval = createClearMotEvaluator(4.5);

  const vortPredPooled = [];
  const vortTruePooled = [];
  let vortErrSq = 0.0;
  let vortTrueSq = 0.0;
  let velErrSq = 0.0;
  let velSamples = 0;
  let unionMask = 0;
  let divRatioSum = 0.0;
  let divRatioSamples = 0;
  let gradEvalFrames = 0;
  let validCellGe400Frames = 0;
  let ftleEvalFrames = 0;
  let ftleBitFrames = 0;
  const ftleCellSamples = [];
  let maxBufFtle = 0.0;
  let maxMicroFtleCount = 0;
  let obsParityErrors = 0;
  let gradParityErrors = 0;
  let divParityErrors = 0;
  let ftleParityErrors = 0;
  let ftleMatureParityErrors = 0;
  let ftleMatureInvalidCount = 0;
  let maxFtleMature39 = 0;
  let ftlePrematureErrors = 0;
  let maxBufDivRms = 0.0;
  const prevTrackState = new Map();
  let maxIndepMatureTracks = 0;

  const cellW = HD_W / FIELD_COLS;
  const cellH = HD_H / FIELD_ROWS;

  const run = await runSyntheticWasmSequence(wasm, {
    numFrames: scene.numFrames,
    fps: scene.fps,
    dtSec: scene.dtSec,
    swarmMode: 2,
    renderFrame: (rgba, f) => {
      const pts = scene.gtFrames[f];
      for (let i = 0; i < pts.length; i++) {
        stampParticleBlob(rgba, HD_W, HD_H, pts[i].x, pts[i].y, pts[i].gtId, pts[i].heroDrifter);
      }
    },
    onFrame: ({ f, session, micro, st, mask }) => {
      unionMask |= mask;
      const swarmActive = st[1] > 0.5;
      if (micro.observed.length !== Math.round(st[4])) obsParityErrors++;
      if (f >= 3) {
        // Only state === 1 observed detections enter CLEAR-MOT / HOTA
        motEval.update(scene.gtFrames[f], micro.observed);
      }

      // Independent consecutive observed track continuity streak (no producer counter trust)
      const nextTrackState = new Map();
      let indepMatureNow = 0;
      let microFtleNow = 0;
      let anyMicroFtleNow = false;
      if (!swarmActive) {
        prevTrackState.clear();
        for (let i = 0; i < micro.observed.length; i++) {
          if (micro.observed[i].ftle > -900.0) anyMicroFtleNow = true;
        }
      } else {
        for (let i = 0; i < micro.observed.length; i++) {
          const t = micro.observed[i];
          const prev = prevTrackState.get(t.tid);
          const streak =
            prev && Math.hypot(t.x - prev.x, t.y - prev.y) <= 20.0 ? prev.streak + 1 : 1;
          nextTrackState.set(t.tid, { x: t.x, y: t.y, streak });
          if (streak >= 9) indepMatureNow++;
          if (t.ftle > -900.0) {
            anyMicroFtleNow = true;
            if (Math.abs(t.ftle) > 1e-6) microFtleNow++;
          }
        }
        prevTrackState.clear();
        for (const [k, v] of nextTrackState.entries()) prevTrackState.set(k, v);
      }
      maxIndepMatureTracks = Math.max(maxIndepMatureTracks, indepMatureNow);
      maxMicroFtleCount = Math.max(maxMicroFtleCount, microFtleNow);

      if (!Number.isInteger(st[39]) || st[39] < 0 || st[39] > Math.round(st[3]) || Math.round(st[39]) > indepMatureNow) {
        ftleMatureInvalidCount++;
      } else {
        maxFtleMature39 = Math.max(maxFtleMature39, Math.round(st[39]));
      }

      // Direct buffer scan of continuumField validity & FTLE sentinels
      let bufGradCells = 0;
      let bufFtleCells = 0;
      let frameMaxBufFtle = -Infinity;
      let frameVortSq = 0.0;
      let frameDivSq = 0.0;
      for (let row = 0; row < FIELD_ROWS; row++) {
        for (let col = 0; col < FIELD_COLS; col++) {
          const base = (row * FIELD_COLS + col) * FIELD_STRIDE;
          if (session.continuumField[base + 5] > 0.0) {
            bufGradCells++;
            const wCell = session.continuumField[base + 2];
            const dCell = session.continuumField[base + 3];
            frameVortSq += wCell * wCell;
            frameDivSq += dCell * dCell;
          }
          const ftleVal = session.continuumField[base + 4];
          if (swarmActive && Number.isFinite(ftleVal) && ftleVal > -900.0) {
            bufFtleCells++;
            if (ftleVal > frameMaxBufFtle) frameMaxBufFtle = ftleVal;
          }
        }
      }
      if (
        f < 8 &&
        (Math.round(st[39]) !== 0 ||
          (mask & 2) !== 0 ||
          Math.round(st[22]) !== 0 ||
          bufFtleCells > 0 ||
          anyMicroFtleNow)
      ) {
        ftlePrematureErrors++;
      }
      if (bufFtleCells > 0) maxBufFtle = Math.max(maxBufFtle, frameMaxBufFtle);

      if (f >= 10) {
        gradEvalFrames++;
        if (bufGradCells !== Math.round(st[17])) gradParityErrors++;
        if ((mask & 1) !== 0 && bufGradCells >= 400) validCellGe400Frames++;
        if ((mask & 1) !== 0 && bufGradCells > 0) {
          for (let row = 0; row < FIELD_ROWS; row++) {
            for (let col = 0; col < FIELD_COLS; col++) {
              const base = (row * FIELD_COLS + col) * FIELD_STRIDE;
              if (session.continuumField[base + 5] > 0.0) {
                const cx = (col + 0.5) * cellW;
                const cy = (row + 0.5) * cellH;
                const uHat = session.continuumField[base + 0];
                const vHat = session.continuumField[base + 1];
                const wHat = session.continuumField[base + 2];
                const wTrue = scene.analyticVorticityAt(cx, cy);
                const [uTrue, vTrue] = scene.velAt(cx, cy);

                vortPredPooled.push(wHat);
                vortTruePooled.push(wTrue);
                vortErrSq += (wHat - wTrue) * (wHat - wTrue);
                vortTrueSq += wTrue * wTrue;
                velErrSq += (uHat - uTrue) * (uHat - uTrue) + (vHat - vTrue) * (vHat - vTrue);
                velSamples++;
              }
            }
          }
          const bufVortRms = Math.sqrt(frameVortSq / bufGradCells);
          const bufDivRms = Math.sqrt(frameDivSq / bufGradCells);
          maxBufDivRms = Math.max(maxBufDivRms, bufDivRms);
          if (
            Math.abs(bufVortRms - st[19]) > 0.25 * Math.max(1.0, bufVortRms) ||
            Math.abs(bufDivRms - st[20]) > 0.25 * Math.max(0.5, bufDivRms)
          ) {
            divParityErrors++;
          }
          if (bufVortRms > 1e-4) {
            divRatioSum += bufDivRms / bufVortRms;
            divRatioSamples++;
          }
        }
      }
      if (f >= 20) {
        ftleEvalFrames++;
        if (bufFtleCells !== Math.round(st[22])) ftleParityErrors++;
        if ((mask & 2) !== 0 && bufFtleCells > 0) {
          ftleCellSamples.push(bufFtleCells);
          if (bufFtleCells >= 200) {
            ftleBitFrames++;
            if (Math.round(st[39]) <= 0) ftleMatureParityErrors++;
          }
        }
      }
    },
  });

  const mot = motEval.summary();
  const vortPearsonR = pearsonR(vortPredPooled, vortTruePooled);
  const vortNrmse = vortTrueSq > 1e-9 ? Math.sqrt(vortErrSq / vortTrueSq) : Infinity;
  const velRmsePxS = velSamples > 0 ? Math.sqrt(velErrSq / velSamples) : Infinity;
  const meanDivRatio = divRatioSamples > 0 ? divRatioSum / divRatioSamples : Infinity;
  const cellCovRate = gradEvalFrames > 0 ? validCellGe400Frames / gradEvalFrames : 0.0;
  const ftleActiveRate = ftleEvalFrames > 0 ? ftleBitFrames / ftleEvalFrames : 0.0;
  const medBufFtleCells = ftleCellSamples.length > 0 ? computeMedian(ftleCellSamples) : 0;

  const stats = summarizeRunStats(run.session, run.tickMs, run.warmTickMs, run.counters, {
    fps: scene.fps,
    dtSec: scene.dtSec,
    mot,
  });

  logDiagnostics('Q1: vortex_swarm_1200', stats);
  console.log(
    `  Q1 Physics    : vortPearsonR=${vortPearsonR.toFixed(4)} (>=0.80) | vortNRMSE=${vortNrmse.toFixed(3)} (<0.45) | velRMSE=${velRmsePxS.toFixed(2)} px/s (<18.0) | bufDiv/vort=${meanDivRatio.toFixed(3)} (<0.65, maxDiv=${maxBufDivRms.toFixed(2)}) | bufGrad>=400=${(cellCovRate * 100).toFixed(1)}% (>=80%)`
  );
  console.log(
    `  Q1 FTLE       : ftleBufRate=${(ftleActiveRate * 100).toFixed(1)}% (>=75%) | medBufFtleCells=${medBufFtleCells} (>=250) | microFtleCnt=${maxMicroFtleCount} (>=300) | mature[39]=${maxFtleMature39} (>=800) | indepMature=${maxIndepMatureTracks} (>=800) | bufPeakFTLE=${maxBufFtle.toFixed(3)}/s | parityErr=[obs=${obsParityErrors},grad=${gradParityErrors},div=${divParityErrors},ftle=${ftleParityErrors},m39=${ftleMatureParityErrors}]`
  );

  const failures = [];
  appendCommonGateFailures(failures, stats, {
    minPeakConf: 1050,
    minMeanObs: 1000,
    minP10Obs: 950,
    minHeroActive: 1,
    motReq: { recall: 85.0, prec: 85.0, mota: 75.0, rmse: 0.75, idsw: 150, hota: 0.70 },
  });
  if (obsParityErrors > 0) failures.push(`obsParityErrors=${obsParityErrors}`);
  if (gradParityErrors > 0) failures.push(`gradParityErrors=${gradParityErrors}`);
  if (divParityErrors > 0) failures.push(`divRmsParityErrors=${divParityErrors}`);
  if (ftleParityErrors > 0) failures.push(`ftleParityErrors=${ftleParityErrors}`);
  if (ftlePrematureErrors > 0) failures.push(`ftlePrematureErrors=${ftlePrematureErrors}`);
  if (ftleMatureInvalidCount > 0 || ftleMatureParityErrors > 0) {
    failures.push(`ftleMature39InvalidOrParity(inv=${ftleMatureInvalidCount},par=${ftleMatureParityErrors})`);
  }
  if (maxFtleMature39 < 800) failures.push(`maxFtleMature39=${maxFtleMature39}<800`);
  if (vortPearsonR < 0.80) failures.push(`vortPearsonR=${vortPearsonR.toFixed(3)}<0.80`);
  if (!Number.isFinite(vortNrmse) || vortNrmse >= 0.45) failures.push(`vortNRMSE=${vortNrmse.toFixed(3)}>=0.45`);
  if (!Number.isFinite(velRmsePxS) || velRmsePxS >= 18.0) failures.push(`velRMSE=${velRmsePxS.toFixed(2)}>=18.0`);
  if (!Number.isFinite(meanDivRatio) || meanDivRatio >= 0.65) failures.push(`divRatio=${meanDivRatio.toFixed(3)}>=0.65`);
  if (cellCovRate < 0.80) failures.push(`bufGradCellCovRate=${(cellCovRate * 100).toFixed(1)}%<80%`);
  if (ftleActiveRate < 0.75) failures.push(`ftleActiveRate=${(ftleActiveRate * 100).toFixed(1)}%<75%`);
  if (medBufFtleCells < 250) failures.push(`medBufFtleCells=${medBufFtleCells}<250`);
  if (maxMicroFtleCount < 300) failures.push(`maxMicroFtleCount=${maxMicroFtleCount}<300`);
  if (maxIndepMatureTracks < 800) failures.push(`maxIndepMatureTracks=${maxIndepMatureTracks}<800`);
  if (maxBufFtle <= 0.0) failures.push(`bufMaxFtle=${maxBufFtle.toFixed(3)}<=0`);

  return { gate: 'Q1', passed: failures.length === 0, failures };
}

async function runQ2Murmuration(wasm, scene) {
  const motEval = createClearMotEvaluator(4.5);

  const steadyFrames = scene.numFrames - 20;
  const nBodies = scene.bodies.length;
  const gtBodyMatchedFrames = new Int32Array(nBodies);
  const perBodyJointPassFrames = new Int32Array(nBodies);
  let framesWithAll3Matched = 0;
  let framesWithAll3JointOk = 0;
  const perBodyProdOmegaErr = Array.from({ length: nBodies }, () => []);
  const perBodyProdSe2Rmse = Array.from({ length: nBodies }, () => []);
  const perBodyCenterErr = Array.from({ length: nBodies }, () => []);
  const perBodyVelErr = Array.from({ length: nBodies }, () => []);
  const perBodyIndOmegaErr = Array.from({ length: nBodies }, () => []);
  const perBodyIndSe2Rmse = Array.from({ length: nBodies }, () => []);
  const perBodyProdIndOmegaDiff = Array.from({ length: nBodies }, () => []);
  const perBodyProdIndRmseDiff = Array.from({ length: nBodies }, () => []);
  const perBodyGtMembers = Array.from({ length: nBodies }, () => []);
  const perBodyPurity = Array.from({ length: nBodies }, () => []);
  const lastBodyId = new Int32Array(nBodies);
  const bodyIdSwitches = new Int32Array(nBodies);
  let obsParityErrors = 0;
  let maxRigidCnt = 0;
  let prevObsByTid = new Map();

  const run = await runSyntheticWasmSequence(wasm, {
    numFrames: scene.numFrames,
    fps: scene.fps,
    dtSec: scene.dtSec,
    swarmMode: 2,
    renderFrame: (rgba, f) => {
      const pts = scene.gtFrames[f];
      for (let i = 0; i < pts.length; i++) {
        stampParticleBlob(rgba, HD_W, HD_H, pts[i].x, pts[i].y, pts[i].gtId);
      }
    },
    onFrame: ({ f, micro, bodies, st, mask }) => {
      if (micro.observed.length !== Math.round(st[4])) obsParityErrors++;
      if (f >= 3) {
        motEval.update(scene.gtFrames[f], micro.observed);
      }

      const obsByTid = new Map();
      const membersByBodyId = new Map();
      for (let i = 0; i < micro.observed.length; i++) {
        const t = micro.observed[i];
        obsByTid.set(t.tid, t);
        if (t.bodyId > 0) {
          if (!membersByBodyId.has(t.bodyId)) membersByBodyId.set(t.bodyId, []);
          membersByBodyId.get(t.bodyId).push(t);
        }
      }

      if (f >= 20) {
        const gtBodies = scene.bodyGtFrames[f];
        const qualBodies = bodies.filter((b) => b.memberCount >= 18);
        maxRigidCnt = Math.max(maxRigidCnt, qualBodies.length);

        if ((mask & 4) !== 0 && qualBodies.length > 0) {
          const big = 1e6;
          const cost = new Float64Array(gtBodies.length * qualBodies.length);
          for (let gi = 0; gi < gtBodies.length; gi++) {
            for (let di = 0; di < qualBodies.length; di++) {
              const d = Math.hypot(qualBodies[di].cx - gtBodies[gi].cx, qualBodies[di].cy - gtBodies[gi].cy);
              cost[gi * qualBodies.length + di] = d <= 25.0 ? d : big;
            }
          }
          const matchedPairs = solveMinCostMatchAny(cost, gtBodies.length, qualBodies.length).filter(
            (p) => p.cost <= 25.0
          );
          if (matchedPairs.length === nBodies) framesWithAll3Matched++;

          let jointOkCount = 0;
          for (const pair of matchedPairs) {
            const gb = gtBodies[pair.r];
            const db = qualBodies[pair.c];
            const k = gb.bodyIdx;
            gtBodyMatchedFrames[k]++;
            if (lastBodyId[k] > 0 && lastBodyId[k] !== db.bodyId) bodyIdSwitches[k]++;
            lastBodyId[k] = db.bodyId;

            const prodOmErr = Math.abs(db.omega - gb.omega);
            const prodRm = db.se2Rmse;
            const cErr = Math.hypot(db.cx - gb.cx, db.cy - gb.cy);
            const vErr = Math.hypot(db.vx - gb.vx, db.vy - gb.vy);
            perBodyProdOmegaErr[k].push(prodOmErr);
            perBodyProdSe2Rmse[k].push(prodRm);
            perBodyCenterErr[k].push(cErr);
            perBodyVelErr[k].push(vErr);

            // Membership purity & independent 2D Kabsch SE(2) from MicroTrackSoA bodyId members
            const mems = membersByBodyId.get(db.bodyId) || [];
            const gtCorners = scene.gtFrames[f].filter((p) => p.bodyIdx === k);
            let gtMemberHits = 0;
            for (const m of mems) {
              for (const gc of gtCorners) {
                if (Math.hypot(m.x - gc.x, m.y - gc.y) <= 4.5) {
                  gtMemberHits++;
                  break;
                }
              }
            }
            const purity = mems.length > 0 ? gtMemberHits / mems.length : 0.0;
            perBodyGtMembers[k].push(gtMemberHits);
            perBodyPurity[k].push(purity);

            let indOk = true;
            if (mems.length >= 18) {
              const kabschPairs = [];
              for (const m of mems) {
                const prev = prevObsByTid.get(m.tid);
                if (prev) kabschPairs.push({ px: prev.x, py: prev.y, qx: m.x, qy: m.y });
              }
              if (kabschPairs.length >= 12) {
                const kRes = computeKabschSe2(kabschPairs);
                if (kRes) {
                  const omegaInd = wrapAnglePi(kRes.dTheta) / scene.dtSec;
                  const iOm = Math.abs(omegaInd - gb.omega);
                  const dOm = Math.abs(db.omega - omegaInd);
                  const dRm = Math.abs(db.se2Rmse - kRes.rmsePx);
                  perBodyIndOmegaErr[k].push(iOm);
                  perBodyIndSe2Rmse[k].push(kRes.rmsePx);
                  perBodyProdIndOmegaDiff[k].push(dOm);
                  perBodyProdIndRmseDiff[k].push(dRm);
                  if (!(iOm < 0.25 && kRes.rmsePx < 0.75 && dOm < 0.25 && dRm < 0.60)) indOk = false;
                }
              }
            }
            const bodyJointOk =
              prodOmErr < 0.25 &&
              prodRm < 0.75 &&
              cErr < 10.0 &&
              vErr < 4.5 &&
              gtMemberHits >= 18 &&
              purity >= 0.85 &&
              indOk;
            if (bodyJointOk) {
              perBodyJointPassFrames[k]++;
              jointOkCount++;
            }
          }
          if (jointOkCount === nBodies) framesWithAll3JointOk++;
        }
      }

      prevObsByTid = obsByTid;
    },
  });

  const mot = motEval.summary();
  const all3CovRate = steadyFrames > 0 ? framesWithAll3Matched / steadyFrames : 0.0;
  const all3JointRate = steadyFrames > 0 ? framesWithAll3JointOk / steadyFrames : 0.0;
  const perBodyCovRates = Array.from(gtBodyMatchedFrames, (c) => (steadyFrames > 0 ? c / steadyFrames : 0.0));
  const perBodyJointRates = Array.from(perBodyJointPassFrames, (c) => (steadyFrames > 0 ? c / steadyFrames : 0.0));
  const medOrInf = (arr) => (arr.length > 0 ? computeMedian(arr) : Infinity);
  const p90OrInf = (arr) => (arr.length > 0 ? computePercentile(arr, 90) : Infinity);

  const stats = summarizeRunStats(run.session, run.tickMs, run.warmTickMs, run.counters, {
    fps: scene.fps,
    dtSec: scene.dtSec,
    mot,
  });

  logDiagnostics('Q2: murmuration_1200', stats);
  console.log(
    `  Q2 SE(2) All3  : maxBodies=${maxRigidCnt} | all3Cov=${(all3CovRate * 100).toFixed(1)}% (>=75%) | all3Joint=${(all3JointRate * 100).toFixed(1)}% (>=70%) | perBodyJoint=[${perBodyJointRates.map((v) => `${(v * 100).toFixed(0)}%`).join(',')}] (>=80%) | bodyIdSw=[${[...bodyIdSwitches].join(',')}] (<=3)`
  );
  for (let k = 0; k < nBodies; k++) {
    console.log(
      `    Body[${k}]     : joint=${(perBodyJointRates[k] * 100).toFixed(0)}% | omegaProd(med/p90)=${formatNullable(medOrInf(perBodyProdOmegaErr[k]), 3)}/${formatNullable(p90OrInf(perBodyProdOmegaErr[k]), 3)} | velErr(med/p90)=${formatNullable(medOrInf(perBodyVelErr[k]), 2)}/${formatNullable(p90OrInf(perBodyVelErr[k]), 2)} | rmse(med/p90)=${formatNullable(medOrInf(perBodyProdSe2Rmse[k]), 3)}/${formatNullable(p90OrInf(perBodyProdSe2Rmse[k]), 3)} | mem=${formatNullable(medOrInf(perBodyGtMembers[k]), 0)} (purity=${(medOrInf(perBodyPurity[k]) * 100).toFixed(0)}%, N=${perBodyIndOmegaErr[k].length})`
    );
  }

  const failures = [];
  appendCommonGateFailures(failures, stats, {
    minPeakConf: 1050,
    minMeanObs: 1000,
    minP10Obs: 950,
    minHeroActive: 1,
    motReq: { recall: 85.0, prec: 85.0, mota: 75.0, rmse: 0.75, idsw: 150, hota: 0.70 },
  });
  if (obsParityErrors > 0) failures.push(`obsParityErrors=${obsParityErrors}`);
  if (all3CovRate < 0.75) failures.push(`all3CovRate=${(all3CovRate * 100).toFixed(1)}%<75%`);
  if (all3JointRate < 0.70) failures.push(`all3JointPassRate=${(all3JointRate * 100).toFixed(1)}%<70%`);
  const minIndSamples = Math.floor(0.60 * steadyFrames);
  for (let k = 0; k < nBodies; k++) {
    if (perBodyCovRates[k] < 0.85) failures.push(`body${k}Cov=${(perBodyCovRates[k] * 100).toFixed(1)}%<85%`);
    if (perBodyJointRates[k] < 0.80) failures.push(`body${k}JointRate=${(perBodyJointRates[k] * 100).toFixed(1)}%<80%`);
    if (bodyIdSwitches[k] > 3) failures.push(`body${k}IdSwitches=${bodyIdSwitches[k]}>3`);
    const pOm = medOrInf(perBodyProdOmegaErr[k]);
    const pOm90 = p90OrInf(perBodyProdOmegaErr[k]);
    const pRm = medOrInf(perBodyProdSe2Rmse[k]);
    const pRm90 = p90OrInf(perBodyProdSe2Rmse[k]);
    const cEr = medOrInf(perBodyCenterErr[k]);
    const vEr = medOrInf(perBodyVelErr[k]);
    const vEr90 = p90OrInf(perBodyVelErr[k]);
    const iOm = medOrInf(perBodyIndOmegaErr[k]);
    const iRm = medOrInf(perBodyIndSe2Rmse[k]);
    const dOm = medOrInf(perBodyProdIndOmegaDiff[k]);
    const dRm = medOrInf(perBodyProdIndRmseDiff[k]);
    const gtM = medOrInf(perBodyGtMembers[k]);
    const pur = medOrInf(perBodyPurity[k]);
    if (perBodyIndOmegaErr[k].length < minIndSamples) failures.push(`body${k}IndSamples=${perBodyIndOmegaErr[k].length}<${minIndSamples}`);
    if (!Number.isFinite(pOm) || pOm >= 0.25) failures.push(`body${k}ProdOmegaErr=${pOm.toFixed(3)}>=0.25`);
    if (!Number.isFinite(pOm90) || pOm90 >= 0.45) failures.push(`body${k}P90OmegaErr=${pOm90.toFixed(3)}>=0.45`);
    if (!Number.isFinite(pRm) || pRm >= 0.75) failures.push(`body${k}ProdRmse=${pRm.toFixed(3)}>=0.75`);
    if (!Number.isFinite(pRm90) || pRm90 >= 1.15) failures.push(`body${k}P90Rmse=${pRm90.toFixed(3)}>=1.15`);
    if (!Number.isFinite(cEr) || cEr >= 10.0) failures.push(`body${k}CtrErr=${cEr.toFixed(2)}>=10.0`);
    if (!Number.isFinite(vEr) || vEr >= 4.5) failures.push(`body${k}VelErr=${vEr.toFixed(2)}>=4.5`);
    if (!Number.isFinite(vEr90) || vEr90 >= 8.5) failures.push(`body${k}P90VelErr=${vEr90.toFixed(2)}>=8.5`);
    if (!Number.isFinite(iOm) || iOm >= 0.25) failures.push(`body${k}IndOmegaErr=${iOm.toFixed(3)}>=0.25`);
    if (!Number.isFinite(iRm) || iRm >= 0.75) failures.push(`body${k}IndRmse=${iRm.toFixed(3)}>=0.75`);
    if (!Number.isFinite(dOm) || dOm >= 0.25) failures.push(`body${k}ProdIndOmegaDiff=${dOm.toFixed(3)}>=0.25`);
    if (!Number.isFinite(dRm) || dRm >= 0.60) failures.push(`body${k}ProdIndRmseDiff=${dRm.toFixed(3)}>=0.60`);
    if (!Number.isFinite(gtM) || gtM < 18) failures.push(`body${k}MedGtMembers=${gtM}<18`);
    if (!Number.isFinite(pur) || pur < 0.85) failures.push(`body${k}MedPurity=${pur.toFixed(3)}<0.85`);
  }

  return {
    gate: 'Q2',
    passed: failures.length === 0,
    failures,
    wasmAblationRow: {
      name: 'Wasm Morton-CSR+Auction+IC-LK',
      meanMs: stats.warmMeanMs,
      medianMs: stats.warmMedianMs,
      auctionCapHits: null,
      auctionRoundLimitHits: null,
      auctionSteps: stats.auctionSteps,
      reacqEvents: stats.reacqEvents,
      ...mot,
    },
  };
}

async function runQ3ModalBeam(wasm) {
  const scene = createQ3ModalBeamScene();
  const motEval = createClearMotEvaluator(4.5);

  const steadyStart = 45; // >= 4.5 cycles at 3 Hz @ 30 fps
  const totalSteadyFrames = scene.numFrames - steadyStart;
  let jointValidFrames = 0;
  let obsParityErrors = 0;
  let maxModalCellLeak = 0;
  const steadyHz = [];
  const steadyCoh = [];
  const steadySensors = [];
  const steadyR2 = [];
  const steadyAntiPhaseErr = [];
  const steadyValidModalCells = [];

  const run = await runSyntheticWasmSequence(wasm, {
    numFrames: scene.numFrames,
    fps: scene.fps,
    dtSec: scene.dtSec,
    swarmMode: 2,
    renderFrame: (rgba, f) => {
      const pts = scene.gtFrames[f];
      for (let i = 0; i < pts.length; i++) {
        stampParticleBlob(rgba, HD_W, HD_H, pts[i].x, pts[i].y, pts[i].gtId);
      }
    },
    onFrame: ({ f, session, micro, st, mask }) => {
      if (micro.observed.length !== Math.round(st[4])) obsParityErrors++;
      if (f >= 3) {
        motEval.update(scene.gtFrames[f], micro.observed);
      }
      const sensors = Math.round(st[29]);
      const hz = st[27];
      const coh = st[28];
      const r2 = st[41];

      if (f >= steadyStart) {
        steadyHz.push(hz);
        steadyCoh.push(coh);
        steadySensors.push(sensors);
        steadyR2.push(r2);

        // Scan modalField directly via sentinel contract: valid phase in [-3.15, 3.15], sentinel -9.0 excluded
        let validModalCells = 0;
        let emptyRowModalCells = 0;
        let lCos = 0.0;
        let lSin = 0.0;
        let rCos = 0.0;
        let rSin = 0.0;
        let lCnt = 0;
        let rCnt = 0;
        for (let row = 0; row < FIELD_ROWS; row++) {
          for (let col = 0; col < FIELD_COLS; col++) {
            const idx = row * FIELD_COLS + col;
            const phi = session.modalField[idx];
            if (!Number.isFinite(phi) || phi < -3.15 || phi > 3.15) continue;
            validModalCells++;
            if (row <= 2 || row >= FIELD_ROWS - 3) emptyRowModalCells++;
            if (row >= 7 && row <= 28) {
              if (col <= 28) {
                lCos += Math.cos(phi);
                lSin += Math.sin(phi);
                lCnt++;
              } else if (col >= 35) {
                rCos += Math.cos(phi);
                rSin += Math.sin(phi);
                rCnt++;
              }
            }
          }
        }
        maxModalCellLeak = Math.max(maxModalCellLeak, emptyRowModalCells);
        steadyValidModalCells.push(validModalCells);

        let errFromPi = Infinity;
        if (lCnt >= 20 && rCnt >= 20) {
          const lPhi = Math.atan2(lSin, lCos);
          const rPhi = Math.atan2(rSin, rCos);
          const diff = Math.abs(wrapAnglePi(rPhi - lPhi));
          errFromPi = Math.abs(wrapAnglePi(diff - Math.PI));
          steadyAntiPhaseErr.push(errFromPi);
        }

        const simJointOk =
          (mask & 8) !== 0 &&
          sensors >= 400 &&
          coh > 0.35 &&
          hz >= 2.70 &&
          hz <= 3.30 &&
          r2 >= 0.90 &&
          lCnt >= 20 &&
          rCnt >= 20 &&
          errFromPi <= 0.30 &&
          validModalCells >= 120 &&
          validModalCells <= FIELD_CELLS - 6 * FIELD_COLS &&
          emptyRowModalCells === 0;
        if (simJointOk) jointValidFrames++;
      }
    },
  });

  const mot = motEval.summary();
  const jointValidRate = totalSteadyFrames > 0 ? jointValidFrames / totalSteadyFrames : 0.0;
  const medHz = steadyHz.length > 0 ? computeMedian(steadyHz) : 0.0;
  const medCoh = steadyCoh.length > 0 ? computeMedian(steadyCoh) : 0.0;
  const medSensors = steadySensors.length > 0 ? computeMedian(steadySensors) : 0;
  const medR2 = steadyR2.length > 0 ? computeMedian(steadyR2) : 0.0;
  const medAntiPhaseErr = steadyAntiPhaseErr.length > 0 ? computeMedian(steadyAntiPhaseErr) : Infinity;
  const medValidModalCells = steadyValidModalCells.length > 0 ? computeMedian(steadyValidModalCells) : 0;

  const stats = summarizeRunStats(run.session, run.tickMs, run.warmTickMs, run.counters, {
    fps: scene.fps,
    dtSec: scene.dtSec,
    mot,
  });

  logDiagnostics('Q3: modal_beam_1152', stats);
  console.log(
    `  Q3 Modal Joint: jointValid=${(jointValidRate * 100).toFixed(1)}% (>=80%) | medHz=${medHz.toFixed(3)} Hz ([2.70, 3.30]) | medSensors=${medSensors} (>=400) | medModalCells=${medValidModalCells} (>=120, leak=${maxModalCellLeak})`
  );
  console.log(
    `  Q3 AntiPhase  : medCoh=${medCoh.toFixed(3)} (>0.35) | medR2=${medR2.toFixed(3)} (>=0.90) | med|dPhi-pi|=${formatNullable(medAntiPhaseErr, 3)} rad (<=0.30)`
  );

  const failures = [];
  appendCommonGateFailures(failures, stats, {
    minPeakConf: 1024,
    minMeanObs: 980,
    minP10Obs: 920,
    motReq: { recall: 85.0, prec: 85.0, mota: 75.0, rmse: 0.60, idsw: 120, hota: 0.70 },
  });
  if (obsParityErrors > 0) failures.push(`obsParityErrors=${obsParityErrors}`);
  if (jointValidRate < 0.80) failures.push(`modalJointValidRate=${(jointValidRate * 100).toFixed(1)}%<80%`);
  if (medSensors < 400) failures.push(`medModalSensors=${medSensors}<400`);
  if (medValidModalCells < 120) failures.push(`medValidModalCells=${medValidModalCells}<120`);
  if (maxModalCellLeak > 0) failures.push(`emptyRowModalCellLeak=${maxModalCellLeak}>0`);
  if (medHz < 2.70 || medHz > 3.30) failures.push(`medModalHz=${medHz.toFixed(3)} outside [2.70, 3.30]`);
  if (medCoh <= 0.35) failures.push(`medModalCoh=${medCoh.toFixed(3)}<=0.35`);
  if (medR2 < 0.90) failures.push(`medModalR2=${medR2.toFixed(3)}<0.90`);
  if (!Number.isFinite(medAntiPhaseErr) || medAntiPhaseErr > 0.30) {
    failures.push(`medAntiPhaseErr=${medAntiPhaseErr}>0.30`);
  }

  return { gate: 'Q3', passed: failures.length === 0, failures };
}

/**
 * Moving dark-feature 1-to-1 surrogate metric (no curated ground-truth bird boxes exist):
 * - Co-prime staggered grid global affine photometric (c ~ gain*p_shifted + offset)
 *   & multi-scale Catmull-Rom bicubic subpixel pan (-maxPan..+maxPan px, wide-scene guarded)
 * - Unbounded 8-connected dark foreground component silhouette labeling (merges split extrema on one silhouette)
 * - Strict 1-to-1 distance-sorted per-component bipartite matching
 */
function sampleBicubicLuma(buf, w, h, fx, fy) {
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

function measureStarlingDarkExtremaPrecision(frameRgba, prevRgba, observedTracks, w = HD_W, h = HD_H) {
  if (observedTracks.length === 0) return 0.0;
  const N = w * h;
  const luma = new Float32Array(N);
  const prevLuma = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const o = i * 4;
    luma[i] = 0.299 * frameRgba[o] + 0.587 * frameRgba[o + 1] + 0.114 * frameRgba[o + 2];
    prevLuma[i] = 0.299 * prevRgba[o] + 0.587 * prevRgba[o + 1] + 0.114 * prevRgba[o + 2];
  }

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
        p = sampleBicubicLuma(prevLuma, w, h, sx, sy);
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
        const p = isSubB ? sampleBicubicLuma(prevLuma, w, h, px, py) : prevLuma[Math.round(py) * w + Math.round(px)];
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
      const pv = isSub0 ? sampleBicubicLuma(prevLuma, w, h, px, py) : prevLuma[Math.round(py) * w + Math.round(px)];
      if (Math.abs(luma[y * w + x] - (useGain * pv + useOffset)) <= 6.0) { b1PtsX.push(x); b1PtsY.push(y); }
    }
    if (b1PtsX.length >= 24) {
      const evalB1 = (dx, dy) => {
        let sc=0,sp=0,scc=0,spp=0,scp=0,n=0;
        for (let i = 0; i < b1PtsX.length; i++) {
          const x=b1PtsX[i], y=b1PtsY[i], sx=x-dx, sy=y-dy;
          if (sx<1||sx>w-2||sy<1||sy>h-2) continue;
          const c=blurCurr[y*w+x], p=sampleBicubicLuma(blurPrev,w,h,sx,sy);
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
          const c=blur3Curr[y*w+x], p=sampleBicubicLuma(blur3Prev,w,h,sx,sy);
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
          if (Math.abs(blur3Curr[y*w+x] - (f0.g * sampleBicubicLuma(blur3Prev,w,h,sx,sy) + f0.o)) <= 6.0) { inX.push(x); inY.push(y); }
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
      return sampleBicubicLuma(prevLuma, w, h, sx, sy);
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
          const c = cBuf[y * w + x], p = sampleBicubicLuma(pBuf, w, h, sx, sy);
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
          p = sampleBicubicLuma(pBuf, w, h, sx, sy);
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
          const pv = sampleBicubicLuma(blur3Prev, w, h, px, py);
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
              const p = isS ? sampleBicubicLuma(blurPrev, w, h, px, py) : blurPrev[py * w + px];
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
              const p = isS ? sampleBicubicLuma(prevLuma, w, h, sx, sy) : prevLuma[(y - iDy) * w + (x - iDx)];
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
      psArr.push(isSubPan ? sampleBicubicLuma(prevLuma, w, h, px, py) : prevLuma[py * w + px]);
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
      const bp = sampleBicubicLuma(pBuf, w, h, gx - useDx, gy - useDy);
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
        const d = cBuf[y * w + x] - (ba.gain * sampleBicubicLuma(pBuf, w, h, px0, py0) + ba.offset);
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
        const dc = cBuf[y * w + x] - (ba.gain * sampleBicubicLuma(pBuf, w, h, pxS, pyS) + ba.offset);
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
        const p = subShift ? sampleBicubicLuma(prevLuma, w, h, px, py) : prevLuma[py * w + px];
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
            const c = luma[y * w + x], p = subShift ? sampleBicubicLuma(prevLuma, w, h, px, py) : prevLuma[py * w + px];
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
    return subShift ? sampleBicubicLuma(prevLuma, w, h, px, py) : prevLuma[py * w + px];
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
          pVal = sampleBicubicLuma(pBuf, w, h, sx, sy);
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
            const c = luma[y * w + x], p = sampleBicubicLuma(prevLuma, w, h, sx, sy);
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
        ? sampleBicubicLuma(prevLuma, w, h, prevPx, prevPy)
        : prevLuma[prevPy * w + prevPx];
      const predC = Math.max(0.0, Math.min(255.0, useGain * prevVal + useOffset));
      const tDiffRaw = Math.abs(c - predC);
      let tDiff = tDiffRaw;
      let tNb = 0.0;
      const nbIdx = idx + 1;
      const prevNbPx = isSubPan ? x + 1 - useDx : x + 1 - roundDx;
      if (hasPrevSupport(prevNbPx, prevPy)) {
        const prevNbVal = isSubPan
          ? sampleBicubicLuma(prevLuma, w, h, prevNbPx, prevPy)
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
            const pv5 = isSubPan ? sampleBicubicLuma(prevLuma, w, h, px5, py5) : prevLuma[py5 * w + px5];
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
async function runQ4StarlingRealVideo(wasm) {
  if (!fs.existsSync(STARLING_MP4)) {
    return { gate: 'Q4', passed: false, failures: [`missing ${STARLING_MP4}`] };
  }
  const fps = 30;
  const dtSec = 1.0 / fps;
  const staticDataEnd = wasm.objdumpReport?.maxDataEnd || 0;
  // Use Auto-Density (mode=1) for real high-density starling video as in production web UI
  const session = setupHdSession(wasm.exp, wasm.mem, 1, 1, staticDataEnd);

  const tickMs = [];
  const warmTickMs = [];
  const counters = {
    nanCount: 0,
    tailNonZero: 0,
    rangeViolations: 0,
    dupTidErrors: 0,
    encErrors: 0,
    inspectorErrors: 0,
    inspectorMsgs: [],
    peakConf: 0,
    maxPeaks: 0,
    steadyObsCounts: [],
    heroActiveCounts: [],
  };
  let countParityErrors = 0;
  let postWarmupFrames = 0;
  let postWarmupActiveFrames = 0;
  const activeObsCounts = [];
  const prevRgba = new Uint8Array(HD_W * HD_H * 4);
  const sampleOracleFrames = new Set([30, 90, 150, 210, 270, 330]);
  const oraclePrecisions = [];

  await streamMp4Frames(STARLING_MP4, HD_W, HD_H, (frameRgba, f) => {
    session.ringViews[f % 12].set(frameRgba);
    if (typeof wasm.exp.chronos_set_ingest_pts === 'function') {
      wasm.exp.chronos_set_ingest_pts(f * dtSec);
    }

    const t0 = performance.now();
    wasm.exp.chronos_pre_tick(dtSec);
    session.solveLanczos();
    wasm.exp.chronos_post_tick(dtSec);
    const dtMs = performance.now() - t0;
    tickMs.push(dtMs);
    if (f > 0) warmTickMs.push(dtMs);

    const chk = inspectFrameBuffers(session);
    counters.nanCount += chk.nanCount;
    counters.tailNonZero += chk.reservedTailNonZero;
    counters.rangeViolations += chk.rangeViolations;

    const micro = readMicroTracksStrict(session.microTracks);
    const bodiesRes = readActiveRigidBodies(session.rigidBodies);
    counters.dupTidErrors += micro.duplicateTidErrors;
    counters.encErrors += micro.invalidEncodingErrors + bodiesRes.invalidEncodingErrors;

    if (f === 60 && micro.observed.length > 0) {
      const iErrs = verifyPopulatedInspector(
        wasm.exp,
        session.w,
        session.h,
        micro.observed,
        session.swarmTelem
      );
      if (iErrs.length > 0) {
        counters.inspectorErrors += iErrs.length;
        counters.inspectorMsgs.push(...iErrs);
      }
    }

    const st = session.swarmTelem;
    counters.maxPeaks = Math.max(counters.maxPeaks, Math.round(st[2]));
    counters.peakConf = Math.max(counters.peakConf, micro.observed.length);
    if (f >= 20) counters.steadyObsCounts.push(micro.observed.length);
    if (f >= 10) counters.heroActiveCounts.push(Math.round(session.telem[3]));

    if (micro.observed.length !== Math.round(st[4])) countParityErrors++;
    if (f >= 20) {
      postWarmupFrames++;
      if (st[1] > 0.5) {
        postWarmupActiveFrames++;
        activeObsCounts.push(micro.observed.length);
      }
    }
    if (f > 0 && sampleOracleFrames.has(f)) {
      oraclePrecisions.push(measureStarlingDarkExtremaPrecision(frameRgba, prevRgba, micro.observed));
    }
    prevRgba.set(frameRgba);
  });

  const activeRate = postWarmupFrames > 0 ? postWarmupActiveFrames / postWarmupFrames : 0.0;
  const p10Obs = activeObsCounts.length > 0 ? computePercentile(activeObsCounts, 10) : 0;
  const meanOraclePrec =
    oraclePrecisions.length > 0 ? oraclePrecisions.reduce((a, b) => a + b, 0) / oraclePrecisions.length : 0.0;

  const stats = summarizeRunStats(session, tickMs, warmTickMs, counters, { fps, dtSec });
  logDiagnostics('Q4: starling_murmuration_1000.mp4', stats);
  console.log(
    `  Q4 AutoDensity: swarmActive=${postWarmupActiveFrames}/${postWarmupFrames} (${(activeRate * 100).toFixed(1)}% >=85%) | activeP10Obs=${p10Obs} (>=1000) | parityErr=${countParityErrors} | movingDarkFeat1to1Prec=${(meanOraclePrec * 100).toFixed(1)}% (>=65% over ${oraclePrecisions.length} gain/pan-norm CC samples; surrogate w/o bird GT)`
  );

  const failures = [];
  appendCommonGateFailures(failures, stats, { minPeakConf: 1000 });
  if (countParityErrors > 0) failures.push(`microObservedParityErrors=${countParityErrors}`);
  if (activeRate < 0.85) failures.push(`autoDensityActiveRate=${(activeRate * 100).toFixed(1)}%<85%`);
  if (p10Obs < 1000) failures.push(`p10Observed=${p10Obs}<1000`);
  if (oraclePrecisions.length < 4) failures.push(`insufficientOracleSamples=${oraclePrecisions.length}`);
  if (meanOraclePrec < 0.65) failures.push(`movingDarkFeatPrecision=${(meanOraclePrec * 100).toFixed(1)}%<65%`);

  return { gate: 'Q4', passed: failures.length === 0, failures };
}

// ============================================================================
// Blind-Pixel Analytic Unit-Integration Suite (13 Cases) & Capacity Sweep
// ============================================================================

function readValidFieldStats(session, rowMin = 1, rowMax = FIELD_ROWS - 2) {
  const swarmActive = session.swarmTelem[1] > 0.5;
  let totalValidCells = 0;
  let totalFtleCells = 0;
  let cnt = 0;
  let sumU = 0.0;
  let sumV = 0.0;
  let sumW = 0.0;
  let sumDiv = 0.0;
  let ftleCnt = 0;
  let sumFtle = 0.0;
  let maxFtle = -Infinity;

  for (let row = 0; row < FIELD_ROWS; row++) {
    for (let col = 0; col < FIELD_COLS; col++) {
      const base = (row * FIELD_COLS + col) * FIELD_STRIDE;
      const w = session.continuumField[base + 5];
      const ftle = session.continuumField[base + 4];
      const gradValid = w > 0.0;
      const ftleValid = swarmActive && Number.isFinite(ftle) && ftle > -900.0;
      if (gradValid) totalValidCells++;
      if (ftleValid) totalFtleCells++;

      if (row >= rowMin && row <= rowMax && col >= 1 && col < FIELD_COLS - 1) {
        if (gradValid) {
          cnt++;
          sumU += session.continuumField[base + 0];
          sumV += session.continuumField[base + 1];
          sumW += session.continuumField[base + 2];
          sumDiv += session.continuumField[base + 3];
        }
        if (ftleValid) {
          ftleCnt++;
          sumFtle += ftle;
          if (ftle > maxFtle) maxFtle = ftle;
        }
      }
    }
  }
  return {
    cnt,
    totalValidCells,
    meanU: cnt > 0 ? sumU / cnt : 0.0,
    meanV: cnt > 0 ? sumV / cnt : 0.0,
    meanVort: cnt > 0 ? sumW / cnt : 0.0,
    meanDiv: cnt > 0 ? sumDiv / cnt : 0.0,
    ftleCnt,
    totalFtleCells,
    meanFtle: ftleCnt > 0 ? sumFtle / ftleCnt : null,
    maxFtle: ftleCnt > 0 ? maxFtle : null,
  };
}

function makeJitteredLattice(nx, ny, x0, x1, y0, y1) {
  const pts = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i;
      const h1 = Math.sin(idx * 12.9898 + 0.5) * 43758.5453;
      const h2 = Math.sin(idx * 78.233 + 1.9) * 43758.5453;
      pts.push({
        gtId: idx + 1,
        x: x0 + ((i + 0.5) * (x1 - x0)) / nx + (h1 - Math.floor(h1) - 0.5) * 2.5,
        y: y0 + ((j + 0.5) * (y1 - y0)) / ny + (h2 - Math.floor(h2) - 0.5) * 2.5,
      });
    }
  }
  return pts;
}

async function runAnalyticSuite(wasm) {
  console.log('\n========================================================================');
  console.log(' Blind-Pixel Analytic Continuum, FTLE & SE(2) Verification (13 Cases)');
  console.log('========================================================================');
  const fps = 30;
  const dtSec = 1.0 / fps;
  const results = [];

  async function runCase(name, numFrames, frameGen, checkFn, opts = {}) {
    const history = [];
    const gtFrames = new Array(numFrames);
    const frameTimes = new Array(numFrames);
    const run = await runSyntheticWasmSequence(wasm, {
      numFrames,
      fps,
      dtSec,
      swarmMode: 2,
      resMode: opts.resMode ?? 1,
      checkInspector: opts.checkInspector ?? true,
      renderFrame: (rgba, f, w, h) => {
        const pts = frameGen(f);
        gtFrames[f] = pts;
        for (let i = 0; i < pts.length; i++) stampParticleBlob(rgba, w, h, pts[i].x, pts[i].y, pts[i].gtId);
      },
      onFrame: (info) => {
        frameTimes[info.f] = info.f * dtSec;
        const field = readValidFieldStats(info.session);
        history.push({
          f: info.f,
          w: info.session.w,
          h: info.session.h,
          mask: info.mask,
          st: new Float32Array(info.st),
          field,
          continuumField: new Float32Array(info.session.continuumField),
          modalField: new Float32Array(info.session.modalField),
          bodies: info.bodies,
          obs: info.micro.observed,
          obsLen: info.micro.observed.length,
          coastLen: info.micro.coasted.length,
          maxFtle: info.st[21],
          ftleCells: Math.round(info.st[22]),
          inspectorErrs: opts.inspectOnFrame && opts.inspectOnFrame(info) || [],
        });
      },
    });
    const reasons = [];
    if (run.counters.nanCount > 0) reasons.push(`nanCount=${run.counters.nanCount}`);
    if (run.counters.tailNonZero > 0) reasons.push(`tailNonZero=${run.counters.tailNonZero}`);
    if (run.counters.rangeViolations > 0) reasons.push(`rangeViol=${run.counters.rangeViolations}`);
    if (run.counters.dupTidErrors > 0) reasons.push(`dupTid=${run.counters.dupTidErrors}`);
    if (run.counters.encErrors > 0) reasons.push(`encErr=${run.counters.encErrors}`);
    if (run.counters.inspectorErrors > 0) {
      reasons.push(`insp=${run.counters.inspectorMsgs.join('|') || run.counters.inspectorErrors}`);
    }
    let parityErr = 0;
    let prematureFtleReported = false;
    for (const h of history) {
      if (
        !prematureFtleReported &&
        h.f < 8 &&
        ((h.mask & 2) !== 0 ||
          h.field.totalFtleCells > 0 ||
          h.ftleCells > 0 ||
          Math.round(h.st[39]) > 0 ||
          h.obs.some((o) => o.ftle > -900.0))
      ) {
        prematureFtleReported = true;
        reasons.push(
          `prematureFtleAtF${h.f}(mask=${h.mask},buf=${h.field.totalFtleCells},st22=${h.ftleCells},m39=${Math.round(h.st[39])})`
        );
      }
      if (h.f >= 10) {
        if (h.obsLen !== Math.round(h.st[4])) parityErr++;
        if (h.field.totalValidCells !== Math.round(h.st[17])) parityErr++;
        if (h.field.totalFtleCells !== h.ftleCells) parityErr++;
      }
      if (h.inspectorErrs.length > 0) reasons.push(...h.inspectorErrs);
    }
    if (parityErr > 0) reasons.push(`analyticBufParityErr=${parityErr}`);

    // Track-level continuity CLEAR-MOT verification across all specified frame windows
    const motRanges = opts.motRanges || [
      opts.motRange ? opts.motRange : [Math.max(8, numFrames - 12), numFrames],
    ];
    const minMotRecall = opts.minMotRecall ?? 70.0;
    const minMotPrec = opts.minMotPrec ?? 70.0;
    const maxMotRmse = opts.maxMotRmse ?? 2.0;
    for (const [motStart, motEnd] of motRanges) {
      if (motEnd <= motStart) continue;
      const ev = createClearMotEvaluator(4.5);
      for (let f = motStart; f < motEnd && f < history.length; f++) {
        ev.update(gtFrames[f] || [], history[f].obs);
      }
      const m = ev.summary();
      const tag = motRanges.length > 1 ? `[${motStart}..${motEnd}]` : '';
      if (m.recallPct === null || m.recallPct < minMotRecall) {
        reasons.push(`gtMotRecall${tag}=${formatNullable(m.recallPct, 1, '%')}<${minMotRecall}%`);
      }
      if (m.precisionPct === null || m.precisionPct < minMotPrec) {
        reasons.push(`gtMotPrec${tag}=${formatNullable(m.precisionPct, 1, '%')}<${minMotPrec}%`);
      }
      if (m.rmsePx === null || m.rmsePx > maxMotRmse) {
        reasons.push(`gtMotRmse${tag}=${formatNullable(m.rmsePx, 3)}>${maxMotRmse}`);
      }
    }

    checkFn(history, reasons, { frameTimes });
    const pass = reasons.length === 0;
    console.log(`  [${pass ? 'PASS' : 'FAIL'}] ${name}${reasons.length ? ': ' + reasons.join(', ') : ''}`);
    results.push({ name, passed: pass, failures: reasons });
  }

  // 1. Pure Translation (u=48, v=-24 px/s => curl=0, div=0, FTLE~0)
  const base1 = makeJitteredLattice(24, 15, 60, 520, 60, 320);
  await runCase('1. Pure Translation (u=48, v=-24, FTLE~0)', 35, (f) => {
    const t = f * dtSec;
    return base1.map((p) => ({ gtId: p.gtId, x: p.x + 48.0 * t, y: p.y - 24.0 * t })).filter((p) => p.x < 625 && p.y > 15);
  }, (hist, err) => {
    const tail = hist.slice(22);
    let ftleValidFrames = 0;
    for (const h of tail) {
      if (h.obsLen < 220) { err.push(`f${h.f} obs=${h.obsLen}<220`); break; }
      if ((h.mask & 1) === 0 || h.field.cnt < 180) { err.push(`f${h.f} gradCells=${h.field.cnt}<180`); break; }
      if (Math.abs(h.field.meanU - 48.0) > 4.0) { err.push(`f${h.f} meanU=${h.field.meanU.toFixed(2)}`); break; }
      if (Math.abs(h.field.meanV + 24.0) > 4.0) { err.push(`f${h.f} meanV=${h.field.meanV.toFixed(2)}`); break; }
      if (Math.abs(h.field.meanVort) > 0.20) { err.push(`f${h.f} vort=${h.field.meanVort.toFixed(3)}`); break; }
      if (Math.abs(h.field.meanDiv) > 0.20) { err.push(`f${h.f} div=${h.field.meanDiv.toFixed(3)}`); break; }
      if ((h.mask & 2) !== 0 && h.field.ftleCnt >= 100 && h.field.meanFtle !== null) {
        ftleValidFrames++;
        if (Math.abs(h.field.meanFtle) > 0.25 || h.field.maxFtle > 0.65) {
          err.push(`f${h.f} transFtle mean=${h.field.meanFtle.toFixed(3)} max=${h.field.maxFtle.toFixed(3)}`);
          break;
        }
      }
    }
    if (ftleValidFrames < Math.floor(0.5 * tail.length)) err.push(`transFtleCov=${ftleValidFrames}/${tail.length}`);
  });

  // 2. Pure Rotation (omega=+0.85 rad/s => curl=+1.70/s, div=0, FTLE~0)
  const base2 = [];
  for (let r = 24; r <= 120; r += 16) {
    const m = Math.max(8, Math.round((2 * Math.PI * r) / 14.0));
    for (let k = 0; k < m; k++) {
      const a = (2 * Math.PI * k) / m;
      base2.push({ gtId: base2.length + 1, dx: r * Math.cos(a), dy: r * Math.sin(a) });
    }
  }
  await runCase('2. Pure Rotation (omega=+0.85, curl=+1.70/s, FTLE~0)', 40, (f) => {
    const th = 0.85 * f * dtSec;
    const ct = Math.cos(th);
    const st = Math.sin(th);
    return base2.map((p) => ({ gtId: p.gtId, x: 320.0 + p.dx * ct - p.dy * st, y: 180.0 + p.dx * st + p.dy * ct }));
  }, (hist, err) => {
    const tail = hist.slice(24);
    let ftleValidFrames = 0;
    for (const h of tail) {
      if (h.obsLen < 120) { err.push(`f${h.f} obs=${h.obsLen}<120`); break; }
      if ((h.mask & 1) === 0 || h.field.cnt < 80) { err.push(`f${h.f} gradCells=${h.field.cnt}<80`); break; }
      if (Math.abs(h.field.meanVort - 1.70) > 0.30) { err.push(`f${h.f} vort=${h.field.meanVort.toFixed(3)}!=1.70`); break; }
      if (Math.abs(h.field.meanDiv) > 0.20) { err.push(`f${h.f} div=${h.field.meanDiv.toFixed(3)}`); break; }
      if ((h.mask & 4) === 0 || h.bodies.length < 1 || Math.abs(h.bodies[0].omega - 0.85) > 0.25 || h.bodies[0].se2Rmse >= 0.75) {
        err.push(`f${h.f} rigidOmega=${h.bodies[0]?.omega?.toFixed(3) ?? 'NONE'}`);
        break;
      }
      if ((h.mask & 2) !== 0 && h.field.ftleCnt >= 50 && h.field.meanFtle !== null) {
        ftleValidFrames++;
        if (Math.abs(h.field.meanFtle) > 0.30 || h.field.maxFtle > 0.80) {
          err.push(`f${h.f} rotFtle mean=${h.field.meanFtle.toFixed(3)} max=${h.field.maxFtle.toFixed(3)}`);
          break;
        }
      }
    }
    if (ftleValidFrames < Math.floor(0.5 * tail.length)) err.push(`rotFtleCov=${ftleValidFrames}/${tail.length}`);
  });

  // 3. Isotropic Expansion (s=+0.50/s => div=+1.00/s, signed FTLE=+0.50/s)
  const base3 = makeJitteredLattice(22, 15, 160, 480, 90, 270);
  await runCase('3. Isotropic Expansion (div=+1.00/s, FTLE=+0.50/s)', 30, (f) => {
    const sc = Math.exp(0.50 * f * dtSec);
    return base3
      .map((p) => ({ gtId: p.gtId, x: 320.0 + (p.x - 320.0) * sc, y: 180.0 + (p.y - 180.0) * sc }))
      .filter((p) => p.x > 16 && p.x < 624 && p.y > 16 && p.y < 344);
  }, (hist, err) => {
    const tail = hist.slice(20);
    let ftleOk = 0;
    for (const h of tail) {
      if (h.obsLen < 200 || (h.mask & 1) === 0 || h.field.cnt < 120) {
        err.push(`f${h.f} obs=${h.obsLen} grad=${h.field.cnt}`);
        break;
      }
      if (h.field.meanDiv < 0.70 || h.field.meanDiv > 1.35) { err.push(`f${h.f} meanDiv=${h.field.meanDiv.toFixed(3)}`); break; }
      if (Math.abs(h.field.meanVort) > 0.20) { err.push(`f${h.f} meanVort=${h.field.meanVort.toFixed(3)}`); break; }
      if ((h.mask & 2) !== 0 && h.field.ftleCnt >= 80 && h.field.meanFtle !== null) {
        if (Math.abs(h.field.meanFtle - 0.50) > 0.22 || h.field.maxFtle > 1.15) {
          err.push(`f${h.f} expFtle mean=${h.field.meanFtle.toFixed(3)}!=0.50 max=${h.field.maxFtle.toFixed(3)}`);
          break;
        }
        ftleOk++;
      }
    }
    if (ftleOk < Math.floor(0.5 * tail.length)) err.push(`expFtleCov=${ftleOk}/${tail.length}`);
  });

  // 4. Isotropic Contraction (s=-0.50/s => div=-1.00/s, signed FTLE=-0.50/s)
  const base4 = makeJitteredLattice(22, 15, 90, 550, 50, 310);
  await runCase('4. Isotropic Contraction (div=-1.00/s, signed FTLE=-0.50/s)', 30, (f) => {
    const sc = Math.exp(-0.50 * f * dtSec);
    return base4.map((p) => ({ gtId: p.gtId, x: 320.0 + (p.x - 320.0) * sc, y: 180.0 + (p.y - 180.0) * sc }));
  }, (hist, err) => {
    const tail = hist.slice(20);
    let ftleOk = 0;
    for (const h of tail) {
      if (h.obsLen < 200 || (h.mask & 1) === 0 || h.field.cnt < 80) {
        err.push(`f${h.f} obs=${h.obsLen} grad=${h.field.cnt}`);
        break;
      }
      if (h.field.meanDiv > -0.70 || h.field.meanDiv < -1.35) { err.push(`f${h.f} meanDiv=${h.field.meanDiv.toFixed(3)}`); break; }
      if (Math.abs(h.field.meanVort) > 0.20) { err.push(`f${h.f} meanVort=${h.field.meanVort.toFixed(3)}`); break; }
      if ((h.mask & 2) !== 0 && h.field.ftleCnt >= 60 && h.field.meanFtle !== null) {
        if (Math.abs(h.field.meanFtle - (-0.50)) > 0.25 || h.field.maxFtle > 0.25) {
          err.push(`f${h.f} contrFtle mean=${h.field.meanFtle.toFixed(3)}!=-0.50 max=${h.field.maxFtle.toFixed(3)}`);
          break;
        }
        ftleOk++;
      }
    }
    if (ftleOk < Math.floor(0.5 * tail.length)) err.push(`contrFtleCov=${ftleOk}/${tail.length}`);
  });

  // 5. Simple Shear (gamma=1.20/s => vort=-1.20/s, FTLE = asinh(|gamma*T|/2)/T over dynamic anchor horizon [8..20]f)
  const base5 = makeJitteredLattice(24, 15, 110, 530, 55, 305);
  await runCase('5. Simple Shear (gamma=1.20/s, exact asinh(gT/2)/T FTLE)', 38, (f) => {
    const t = f * dtSec;
    return base5
      .map((p) => ({ gtId: p.gtId, x: p.x + 1.20 * (p.y - 180.0) * t, y: p.y }))
      .filter((p) => p.x > 16 && p.x < 624);
  }, (hist, err, ctx) => {
    // Anchor bank spans dynamic [8..20] frames => compute analytic shear FTLE window from actual timestamps
    const frameTimes = ctx?.frameTimes || hist.map((x) => x.f * dtSec);
    const ftleTol = 0.045;
    const tail = hist.slice(28);
    let ftleOk = 0;
    for (const h of tail) {
      if (h.obsLen < 200 || (h.mask & 1) === 0 || h.field.cnt < 140) {
        err.push(`f${h.f} obs=${h.obsLen} grad=${h.field.cnt}`);
        break;
      }
      if (Math.abs(h.field.meanVort + 1.20) > 0.30) { err.push(`f${h.f} meanVort=${h.field.meanVort.toFixed(3)}!=-1.20`); break; }
      if (Math.abs(h.field.meanDiv) > 0.20) { err.push(`f${h.f} meanDiv=${h.field.meanDiv.toFixed(3)}`); break; }
      if ((h.mask & 2) !== 0 && h.field.ftleCnt >= 100 && h.field.meanFtle !== null) {
        if (Math.round(h.st[39]) < 120) {
          err.push(`f${h.f} shearMature[39]=${Math.round(h.st[39])}<120`);
          break;
        }
        let ftleLowT = Infinity;
        let ftleHighT = -Infinity;
        const maxAge = Math.min(20, h.f);
        for (let age = 8; age <= maxAge; age++) {
          const horizonT = frameTimes[h.f] - frameTimes[h.f - age];
          if (!Number.isFinite(horizonT) || horizonT <= 1e-6) continue;
          const analytic = Math.asinh(0.5 * 1.20 * horizonT) / horizonT;
          if (analytic < ftleLowT) ftleLowT = analytic;
          if (analytic > ftleHighT) ftleHighT = analytic;
        }
        if (!Number.isFinite(ftleLowT) || !Number.isFinite(ftleHighT)) {
          err.push(`f${h.f} shearBadHorizonT`);
          break;
        }
        if (
          h.field.meanFtle < ftleLowT - ftleTol ||
          h.field.meanFtle > ftleHighT + ftleTol ||
          h.field.maxFtle > ftleHighT + 0.15
        ) {
          err.push(
            `f${h.f} shearFtle mean=${h.field.meanFtle.toFixed(3)} (band=[${(ftleLowT - ftleTol).toFixed(3)},${(ftleHighT + ftleTol).toFixed(3)}]) max=${h.field.maxFtle.toFixed(3)}`
          );
          break;
        }
        ftleOk++;
      }
    }
    if (ftleOk < Math.floor(0.5 * tail.length)) err.push(`shearFtleCov=${ftleOk}/${tail.length}`);
  }, { motRange: [28, 38] });

  // 6a. Collinear 1D line strip rank guard (must track blobs, must suppress rank-deficient Grad-V/FTLE; SE(2) strip allowed if properly fitted)
  const stripPts = Array.from({ length: 28 }, (_, i) => ({ gtId: i + 1, lx: -108.0 + i * 8.0, ly: 0.0 }));
  await runCase('6a. Collinear 1D Strip Rank Guard', 32, (f) => {
    const th = 0.80 * f * dtSec;
    const ct = Math.cos(th);
    const st = Math.sin(th);
    return stripPts.map((p) => ({ gtId: p.gtId, x: 320.0 + p.lx * ct - p.ly * st, y: 180.0 + p.lx * st + p.ly * ct }));
  }, (hist, err) => {
    for (const h of hist.slice(18)) {
      if (h.obsLen < 20) { err.push(`f${h.f} collinearObs=${h.obsLen}<20`); break; }
      if ((h.mask & 4) !== 0 || h.bodies.length > 0) {
        if ((h.mask & 4) === 0 || h.bodies.length < 1 || h.bodies[0].memberCount < 18 || Math.abs(h.bodies[0].omega - 0.80) > 0.25 || h.bodies[0].se2Rmse >= 0.75) {
          err.push(`f${h.f} collinearBadRigidBody=${JSON.stringify(h.bodies[0] || null)}(mask=${h.mask})`);
          break;
        }
      }
      if ((h.mask & 3) !== 0 || h.field.cnt > 0 || h.field.ftleCnt > 0 || h.field.totalValidCells > 0 || h.field.totalFtleCells > 0) {
        err.push(`f${h.f} collinearEmittedGradOrFtle(mask=${h.mask},cnt=${h.field.cnt},ftle=${h.field.ftleCnt})`);
        break;
      }
    }
  }, { motRange: [18, 32] });

  const barPts = [
    ...stripPts,
    ...[-36, -24, -12, 12, 24, 36].map((ly, k) => ({ gtId: 100 + k, lx: 0.0, ly })),
  ];
  await runCase('6b. Non-Collinear 2D Crossbar SE(2) Recovery', 36, (f) => {
    const th = 0.80 * f * dtSec;
    const ct = Math.cos(th);
    const st = Math.sin(th);
    return barPts.map((p) => ({ gtId: p.gtId, x: 320.0 + p.lx * ct - p.ly * st, y: 180.0 + p.lx * st + p.ly * ct }));
  }, (hist, err) => {
    for (const h of hist.slice(26)) {
      if (h.obsLen < 24 || (h.mask & 4) === 0 || h.bodies.length < 1 || Math.abs(h.bodies[0].omega - 0.80) > 0.25 || h.bodies[0].se2Rmse >= 0.75) {
        err.push(`f${h.f} barBody=${JSON.stringify(h.bodies[0] || null)}`);
        break;
      }
    }
  }, { motRange: [26, 36] });

  // 7. Mirror Reflection (det=-1) Rejection (requires pre-flip establishment + rejection at flip)
  const asymCluster = [];
  for (let i = 0; i < 36; i++) {
    const a = (2 * Math.PI * i) / 36.0;
    const r = i % 3 === 0 ? 48.0 : 26.0 + (i % 5) * 3.0;
    asymCluster.push({ gtId: i + 1, lx: r * Math.cos(a) + (i < 12 ? 14 : 0), ly: r * Math.sin(a) });
  }
  await runCase('7. Mirror Reflection (det=-1) Rejection', 36, (f) => {
    const th = 0.65 * f * dtSec;
    const ct = Math.cos(th);
    const st = Math.sin(th);
    const flip = f >= 20 ? -1.0 : 1.0;
    return asymCluster.map((p) => {
      const rx = (p.lx * ct - p.ly * st) * flip;
      const ry = p.lx * st + p.ly * ct;
      return { gtId: p.gtId, x: 320.0 + rx, y: 180.0 + ry };
    });
  }, (hist, err) => {
    let preOk = 0;
    for (const h of hist.slice(14, 19)) {
      if (h.obsLen >= 26 && (h.mask & 4) !== 0 && h.bodies.length >= 1 && Math.abs(h.bodies[0].omega - 0.65) <= 0.25 && h.bodies[0].se2Rmse < 0.75) {
        preOk++;
      }
    }
    if (preOk < 3) err.push(`reflectionPreFlipNotEstablished(${preOk}/5)`);
    for (const h of hist.slice(20, 23)) {
      if (h.bodies.some((b) => b.memberCount >= 20 && b.se2Rmse < 0.20)) {
        err.push(`acceptedReflectionAtF${h.f}`);
      }
    }
  }, { motRange: [14, 19] });

  // 8. Swapped-ID / Slot Reuse FTLE Invalidation (pre-established mature FTLE + immediate invalidation + delayed recovery)
  const swapGrid = makeJitteredLattice(18, 12, 120, 520, 70, 290);
  await runCase('8. Swapped-ID / Slot Reuse FTLE Invalidation', 40, (f) => {
    const t = f * dtSec;
    return swapGrid.map((p, idx) => {
      const partner = f >= 22 ? swapGrid[idx ^ 1] : p;
      return { gtId: p.gtId, x: partner.x + 28.0 * t, y: partner.y };
    });
  }, (hist, err) => {
    let preOk = 0;
    for (const h of hist.slice(16, 21)) {
      if (h.obsLen >= 160 && (h.mask & 1) !== 0 && h.field.cnt >= 100 && (h.mask & 2) !== 0 && h.field.totalFtleCells >= 60 && Math.round(h.st[39]) >= 60) {
        preOk++;
      }
    }
    if (preOk < 4) err.push(`swapPreMatureFtle(${preOk}/5)`);
    for (const h of hist.slice(22, 30)) {
      if (
        (h.mask & 2) !== 0 ||
        h.field.totalFtleCells > 0 ||
        h.ftleCells > 0 ||
        Math.round(h.st[39]) > 0 ||
        h.obs.some((o) => o.ftle > -900.0)
      ) {
        err.push(`swapFtleNotInvalidatedAtF${h.f}(mask=${h.mask},cells=${h.field.totalFtleCells},st22=${h.ftleCells},m39=${Math.round(h.st[39])})`);
        break;
      }
      if (h.field.maxFtle !== null && h.field.maxFtle >= 2.0) {
        err.push(`swapFtleSpikeAtF${h.f}=${h.field.maxFtle.toFixed(2)}`);
        break;
      }
    }
    for (const h of hist.slice(36)) {
      if (h.obsLen < 160 || (h.mask & 1) === 0 || h.field.cnt < 100 || (h.mask & 2) === 0 || h.field.totalFtleCells < 60 || Math.round(h.st[39]) < 60) {
        err.push(`swapPostRecoveryFailedAtF${h.f}(obs=${h.obsLen},cnt=${h.field.cnt},ftle=${h.field.totalFtleCells},m39=${Math.round(h.st[39])})`);
        break;
      }
    }
  }, { motRanges: [[16, 21], [24, 40]] });

  // 9. Camera Pan / Cut Inhibit & Recovery (pre-pan mature FTLE + cut inhibit + post-cut recovery)
  const camBase = makeJitteredLattice(20, 14, 90, 500, 60, 300);
  await runCase('9. Camera Pan / Cut Inhibit & Recovery', 45, (f) => {
    const camShiftX = f >= 18 && f < 22 ? (f - 17) * 24.0 : f >= 22 ? 96.0 : 0.0;
    const t = f * dtSec;
    return camBase
      .map((p) => ({ gtId: p.gtId, x: p.x + 30.0 * t + camShiftX, y: p.y }))
      .filter((p) => p.x > 16 && p.x < 624);
  }, (hist, err) => {
    let preOk = 0;
    for (const h of hist.slice(12, 18)) {
      if (h.obsLen >= 200 && (h.mask & 1) !== 0 && Math.abs(h.field.meanU - 30.0) <= 6.0 && (h.mask & 2) !== 0 && h.field.totalFtleCells >= 60 && Math.round(h.st[39]) >= 60) {
        preOk++;
      }
    }
    if (preOk < 4) err.push(`camPreMatureFtle(${preOk}/6)`);
    for (const h of hist.slice(18, 30)) {
      if (
        (h.mask & 2) !== 0 ||
        h.field.totalFtleCells > 0 ||
        h.ftleCells > 0 ||
        Math.round(h.st[39]) > 0 ||
        h.obs.some((o) => o.ftle > -900.0)
      ) {
        err.push(`camCutFtleNotInhibitedAtF${h.f}(mask=${h.mask},cells=${h.field.totalFtleCells},st22=${h.ftleCells},m39=${Math.round(h.st[39])})`);
        break;
      }
      if (h.field.maxFtle !== null && h.field.maxFtle >= 2.0) {
        err.push(`camCutFtleSpikeAtF${h.f}=${h.field.maxFtle.toFixed(2)}`);
        break;
      }
    }
    for (const h of hist.slice(38)) {
      if ((h.mask & 1) === 0 || h.obsLen < 120 || h.field.cnt < 80 || Math.abs(h.field.meanU - 30.0) > 6.0 || (h.mask & 2) === 0 || h.field.totalFtleCells < 40 || Math.round(h.st[39]) < 40) {
        err.push(`camPostCutRecoveryFailedAtF${h.f}(mask=${h.mask},obs=${h.obsLen},u=${h.field.meanU.toFixed(1)},ftle=${h.field.totalFtleCells})`);
        break;
      }
    }
  }, { motRanges: [[12, 18], [24, 45]] });

  // 10. Coast & Reacquisition Continuity (2-frame total occlusion, f=20 & f=21 drop check, >=70% ID keep, delayed FTLE recovery)
  const coastBase = makeJitteredLattice(18, 12, 90, 500, 60, 280);
  const coastFramePts = (f) => {
    if (f === 20 || f === 21) return [];
    const t = f * dtSec;
    return coastBase.map((p) => ({ gtId: p.gtId, x: p.x + 32.0 * t, y: p.y + 10.0 * t }));
  };
  await runCase('10. Coast & Reacquisition Continuity (2f dropout, >=70% ID keep)', 45, coastFramePts, (hist, err) => {
    let preMatureOk = 0;
    for (const h of hist.slice(14, 20)) {
      if (h.obsLen < 170) { err.push(`preDropObs=${h.obsLen}@f${h.f}`); break; }
      if ((h.mask & 2) !== 0 && h.field.totalFtleCells >= 60 && Math.round(h.st[39]) >= 60) preMatureOk++;
    }
    if (preMatureOk < 4) err.push(`coastPreMatureFtle(${preMatureOk}/6)`);
    const h20 = hist[20];
    const h21 = hist[21];
    if (
      h20.obsLen > 5 ||
      h20.coastLen < 120 ||
      (h20.mask & 2) !== 0 ||
      h20.field.totalFtleCells > 0 ||
      h20.ftleCells > 0 ||
      Math.round(h20.st[39]) > 0 ||
      h20.obs.some((o) => o.ftle > -900.0)
    ) {
      err.push(`firstMissF20(obs=${h20.obsLen},coast=${h20.coastLen},ftle=${h20.field.totalFtleCells},m39=${Math.round(h20.st[39])})`);
    }
    if (
      h21.obsLen > 5 ||
      h21.coastLen < 100 ||
      (h21.mask & 2) !== 0 ||
      h21.field.totalFtleCells > 0 ||
      h21.ftleCells > 0 ||
      Math.round(h21.st[39]) > 0 ||
      h21.obs.some((o) => o.ftle > -900.0)
    ) {
      err.push(`secondMissF21(obs=${h21.obsLen},coast=${h21.coastLen},ftle=${h21.field.totalFtleCells},m39=${Math.round(h21.st[39])})`);
    }
    for (const h of hist.slice(22, 30)) {
      if (
        (h.mask & 2) !== 0 ||
        h.field.totalFtleCells > 0 ||
        h.ftleCells > 0 ||
        Math.round(h.st[39]) > 0 ||
        h.obs.some((o) => o.ftle > -900.0)
      ) {
        err.push(`earlyReacqFtleAtF${h.f}(mask=${h.mask},ftle=${h.field.totalFtleCells},st22=${h.ftleCells},m39=${Math.round(h.st[39])})`);
        break;
      }
    }
    // Verify track TID preservation from f=19 across f=20..21 blank dropout into f=25..28 reacquisition
    const mapByGt = (fIdx) => {
      const m = new Map();
      const gts = coastFramePts(fIdx);
      const trks = hist[fIdx].obs;
      for (const g of gts) {
        let best = null;
        let bd = 4.0;
        for (const t of trks) {
          const d = Math.hypot(t.x - g.x, t.y - g.y);
          if (d < bd) { bd = d; best = t.tid; }
        }
        if (best !== null) m.set(g.gtId, best);
      }
      return m;
    };
    const preIds = mapByGt(19);
    const postIds = mapByGt(25);
    let sameId = 0;
    for (const [gid, tid] of preIds.entries()) {
      if (postIds.get(gid) === tid) sameId++;
    }
    const reacqRate = preIds.size > 0 ? sameId / preIds.size : 0.0;
    if (preIds.size < 160 || hist[25].obsLen < 150 || reacqRate < 0.70) {
      err.push(`reacqSameId=${sameId}/${preIds.size} (${(reacqRate * 100).toFixed(1)}%<70%), f25Obs=${hist[25].obsLen}`);
    }
    for (const h of hist.slice(36)) {
      if (h.obsLen < 160 || (h.mask & 1) === 0 || (h.mask & 2) === 0 || h.field.totalFtleCells < 60 || Math.round(h.st[39]) < 60) {
        err.push(`reacqSteadyTailF${h.f}(obs=${h.obsLen},mask=${h.mask},ftle=${h.field.totalFtleCells},m39=${Math.round(h.st[39])})`);
        break;
      }
    }
  }, { motRanges: [[14, 20], [23, 45]] });

  // 11 & 12. Populated chronos_inspect_field_point + Normalized 40x22 Continuum Lookup on SD (320x180) and HD (640x360)
  for (const [resMode, label, wRes, hRes, sc] of [
    [0, '11. Populated Field Inspector (SD 320x180)', 320, 180, 0.5],
    [1, '12. Populated Field Inspector (HD 640x360)', HD_W, HD_H, 1.0],
  ]) {
    await runCase(
      label,
      28,
      (f) => {
        const th = 0.65 * f * dtSec;
        const ct = Math.cos(th);
        const st = Math.sin(th);
        return asymCluster.map((p) => ({
          gtId: p.gtId,
          x: wRes * 0.5 + sc * (p.lx * ct - p.ly * st),
          y: hRes * 0.5 + sc * (p.lx * st + p.ly * ct),
        }));
      },
      (hist, err) => {
        const h = hist[25];
        if (h.obsLen < 24 || h.bodies.length < 1) {
          err.push(`inspSetupWeak(obs=${h.obsLen},bodies=${h.bodies.length})`);
          return;
        }
        const hasBodyTrack = h.obs.some((t) => t.bodyId > 0);
        if (!hasBodyTrack) err.push('inspNoRigidBodyTrack');

        // Verify normalized 40x22 grid lookup on populated cluster cells & empty corners
        let matchedFieldCells = 0;
        for (let r = 0; r < FIELD_ROWS; r++) {
          for (let c = 0; c < FIELD_COLS; c++) {
            const fb = (r * FIELD_COLS + c) * FIELD_STRIDE;
            const cw = h.continuumField[fb + 5];
            if (cw > 0.0) {
              const cxPx = ((c + 0.5) * wRes) / FIELD_COLS;
              const cyPx = ((r + 0.5) * hRes) / FIELD_ROWS;
              const dx = cxPx - wRes * 0.5;
              const dy = cyPx - hRes * 0.5;
              if (Math.hypot(dx, dy) <= sc * 68.0) {
                const uTrue = -0.65 * dy;
                const vTrue = 0.65 * dx;
                if (Math.hypot(h.continuumField[fb] - uTrue, h.continuumField[fb + 1] - vTrue) <= 14.0 && h.modalField[r * FIELD_COLS + c] === -9.0) {
                  matchedFieldCells++;
                }
              }
            }
          }
        }
        if (matchedFieldCells < 1) {
          err.push(`inspNormalizedFieldLookupMiss(${wRes}x${hRes})`);
        }
        for (const cornerIdx of [0, FIELD_COLS - 1, (FIELD_ROWS - 1) * FIELD_COLS, FIELD_ROWS * FIELD_COLS - 1]) {
          const cb = cornerIdx * FIELD_STRIDE;
          if (h.continuumField[cb + 5] !== 0.0 || h.continuumField[cb] !== 0.0 || h.continuumField[cb + 1] !== 0.0 || h.modalField[cornerIdx] !== -9.0) {
            err.push(`inspCornerCellNotCleanZero(${cornerIdx})`);
            break;
          }
        }
      },
      { resMode, checkInspector: true, motRange: [18, 28], minMotRecall: resMode === 0 ? 55.0 : 70.0 }
    );
  }

  return results;
}

async function runCapacitySweep(wasm) {
  console.log('\n========================================================================');
  console.log(' Capacity Scaling Sweep (Separate Runs: N = 512, 1000, 1200, 1536)');
  console.log('========================================================================');
  console.log(' TargetN | ObsPeak | ObsMean | P10Obs | Recall | MOTA  | RMSE  | HOTA  | WarmMean |  p50   |  p95   | Status');
  console.log(' --------+---------+---------+--------+--------+-------+-------+-------+----------+--------+--------+-------');

  const caps = [512, 1000, 1200, 1536];
  const minPeakByN = { 512: 470, 1000: 900, 1200: 1080, 1536: 1350 };
  const minObsMeanByN = { 512: 450, 1000: 880, 1200: 1050, 1536: 1300 };
  const minP10ByN = { 512: 420, 1000: 830, 1200: 990, 1536: 1220 };
  const results = [];
  const fps = 30;
  const dtSec = 1.0 / fps;
  const numFrames = 30;

  for (const cap of caps) {
    const cols = cap === 512 ? 32 : cap === 1000 ? 40 : cap === 1200 ? 48 : 48;
    const rows = cap / cols;
    const base = makeJitteredLattice(cols, rows, 28, 612, 24, 336);
    const gtFrames = new Array(numFrames);
    for (let f = 0; f < numFrames; f++) {
      const th = 0.20 * f * dtSec;
      const ct = Math.cos(th);
      const st = Math.sin(th);
      const pts = [];
      for (let i = 0; i < base.length; i++) {
        const dx = base[i].x - 320.0;
        const dy = base[i].y - 180.0;
        const px = 320.0 + dx * ct - dy * st;
        const py = 180.0 + dx * st + dy * ct;
        if (px > 10 && px < 630 && py > 10 && py < 350) {
          pts.push({ gtId: base[i].gtId, x: px, y: py });
        }
      }
      gtFrames[f] = pts;
    }

    const motEval = createClearMotEvaluator(4.5);
    let obsParityErrors = 0;

    const run = await runSyntheticWasmSequence(wasm, {
      numFrames,
      fps,
      dtSec,
      swarmMode: 2,
      renderFrame: (rgba, f) => {
        const pts = gtFrames[f];
        for (let i = 0; i < pts.length; i++) {
          stampParticleBlob(rgba, HD_W, HD_H, pts[i].x, pts[i].y, pts[i].gtId);
        }
      },
      onFrame: ({ f, micro, st }) => {
        if (micro.observed.length !== Math.round(st[4])) obsParityErrors++;
        if (f >= 3) {
          motEval.update(gtFrames[f], micro.observed);
        }
      },
    });

    const mot = motEval.summary();
    const stats = summarizeRunStats(run.session, run.tickMs, run.warmTickMs, run.counters, {
      fps,
      dtSec,
      mot,
    });
    const failures = [];
    // Uniform point lattices yield 0 Hero-16 confirmed targets below ultra-dense clustering (Q2/Q3/Q4 enforce Hero >= 1).
    appendCommonGateFailures(failures, stats, {
      minPeakConf: minPeakByN[cap],
      minMeanObs: minObsMeanByN[cap],
      minP10Obs: minP10ByN[cap],
      minHeroActive: 0,
      motReq: { recall: 85.0, prec: 88.0, mota: 78.0, rmse: 0.75, idsw: Math.round(cap * 0.12), hota: 0.72 },
    });
    if (obsParityErrors > 0) failures.push(`obsParityErrors=${obsParityErrors}`);
    if (stats.peakConf > MAX_MICRO) failures.push(`peakObs=${stats.peakConf}>${MAX_MICRO}`);
    const passed = failures.length === 0;
    const hStr = mot.hota !== null ? `${(mot.hota * 100).toFixed(1)}%`.padStart(5, ' ') : '  N/A';
    console.log(
      ` ${String(cap).padStart(7, ' ')} | ${String(stats.peakConf).padStart(7, ' ')} | ${stats.meanObs.toFixed(1).padStart(7, ' ')} | ${String(stats.p10Obs).padStart(6, ' ')} | ${formatNullable(mot.recallPct, 1, '%').padStart(6, ' ')} | ${formatNullable(mot.motaPct, 1, '%').padStart(5, ' ')} | ${formatNullable(mot.rmsePx, 3).padStart(5, ' ')} | ${hStr} | ${stats.warmMeanMs.toFixed(3).padStart(8, ' ')} | ${stats.warmMedianMs.toFixed(3).padStart(6, ' ')} | ${stats.p95Ms.toFixed(3).padStart(6, ' ')} | ${passed ? 'PASS' : 'FAIL'}`
    );
    results.push({ gate: `SWEEP_${cap}`, passed, failures });
  }
  return results;
}

// ============================================================================
// Self-Test Fixtures (Evaluator, HOTA, Duplicate-ID, Empty-GT, Sol-3 Auction)
// ============================================================================

function runSelfTests() {
  console.log('\n========================================================================');
  console.log(' Benchmark Evaluator & Auction Unit Self-Tests');
  console.log('========================================================================');
  const assert = (cond, msg) => {
    if (!cond) throw new Error(`Self-test failed: ${msg}`);
  };

  // 1. Empty GT + 10 predictions => Recall/MOTA/RMSE/HOTA === null, Precision === 0
  {
    const ev = createClearMotEvaluator(4.5);
    ev.update([], Array.from({ length: 10 }, (_, i) => ({ tid: i + 1, x: 50 + i * 20, y: 100 })));
    const s = ev.summary();
    assert(s.recallPct === null && s.motaPct === null && s.rmsePx === null && s.hota === null, 'empty GT nulls');
    assert(s.precisionPct === 0.0 && s.fp === 10, 'empty GT precision=0');
  }

  // 2. Non-empty GT + 0 predictions => Recall=0, Precision=null, RMSE=null, HOTA=0
  {
    const ev = createClearMotEvaluator(4.5);
    ev.update([{ gtId: 1, x: 100, y: 100 }], []);
    const s = ev.summary();
    assert(s.recallPct === 0.0 && s.precisionPct === null && s.rmsePx === null && s.hota === 0.0, 'zero pred guards');
  }

  // 3. 0 matches => RMSE=null, MOTP=null, HOTA=0
  {
    const ev = createClearMotEvaluator(4.5);
    ev.update([{ gtId: 1, x: 100, y: 100 }], [{ tid: 1, x: 200, y: 200 }]);
    const s = ev.summary();
    assert(s.gtMatched === 0 && s.rmsePx === null && s.motpPx === null && s.hota === 0.0, 'zero match guards');
  }

  // 4. Duplicate tid rejection
  {
    const ev = createClearMotEvaluator(4.5);
    ev.update(
      [
        { gtId: 1, x: 100, y: 100 },
        { gtId: 2, x: 102, y: 100 },
      ],
      [
        { tid: 7, x: 100, y: 100 },
        { tid: 7, x: 102, y: 100 },
      ]
    );
    const s = ev.summary();
    assert(s.duplicateTidCount === 1 && s.gtMatched === 1, 'duplicate tid caught');
  }

  // 5. Exact IDSW=1, Frag=1 & HOTA check
  {
    const ev = createClearMotEvaluator(4.5);
    ev.update([{ gtId: 1, x: 100, y: 100 }], [{ tid: 1, x: 101, y: 100 }]); // d=1, sim = 1 - 1/9 = 8/9
    ev.update([{ gtId: 1, x: 110, y: 100 }], [{ tid: 2, x: 111, y: 100 }]); // IDSW=1
    ev.update([{ gtId: 1, x: 120, y: 100 }], []); // Lost
    ev.update([{ gtId: 1, x: 130, y: 100 }], [{ tid: 2, x: 131, y: 100 }]); // Frag=1
    const s = ev.summary();
    assert(s.idsw === 1 && s.frag === 1 && s.gtMatched === 3, `IDSW/Frag got idsw=${s.idsw}, frag=${s.frag}`);
    assert(s.hota > 0.3 && s.hota < 0.7, `expected intermediate HOTA, got ${s.hota}`);
  }

  // 6. TrackEval 19-threshold similarity cutoff check (single pair d = 4.5 px => S = 0.50 -> passes 10/19 thresholds)
  {
    const ev = createClearMotEvaluator(4.5);
    ev.update([{ gtId: 1, x: 100, y: 100 }], [{ tid: 1, x: 104.5, y: 100 }]);
    const s = ev.summary();
    // alphas 0.05..0.50 (10 of 19 thresholds) yield 1.0, above 0.50 yield 0.0 -> HOTA = 10/19
    assert(Math.abs(s.hota - 10.0 / 19.0) < 1e-6, `HOTA 19-threshold oracle=${s.hota} vs ${10 / 19}`);
  }

  // 7. Sol-3 4x3 Auction oracle matrix across all 24 row permutations (optimum === 272)
  const oracleMat = [
    [123, 0, 80],
    [93, 108, 77],
    [0, 0, 41],
    [84, 36, 32],
  ];
  const permute = (arr) =>
    arr.length <= 1 ? [arr] : arr.flatMap((v, idx) => permute(arr.filter((_, j) => j !== idx)).map((p) => [v, ...p]));
  for (const perm of permute([0, 1, 2, 3])) {
    const edgeTo = new Int32Array(4 * 3).fill(-1);
    const edgeBen = new Int32Array(4 * 3);
    const edgeCnt = new Int32Array(4);
    for (let r = 0; r < 4; r++) {
      const row = oracleMat[perm[r]];
      for (let c = 0; c < 3; c++) {
        if (row[c] > 0) {
          const e = edgeCnt[r]++;
          edgeTo[r * 3 + e] = c;
          edgeBen[r * 3 + e] = row[c];
        }
      }
    }
    const { trackToObs: assign, capHits, roundLimitHits } = solveAuctionAsym(edgeTo, edgeBen, edgeCnt, 4, 3, 3);
    let score = 0;
    for (let r = 0; r < 4; r++) {
      if (assign[r] >= 0) score += oracleMat[perm[r]][assign[r]];
    }
    assert(score === 272 && capHits === 0 && roundLimitHits === 0, `Auction 4x3 perm [${perm}] scored ${score} != 272`);
  }

  // 8. Random rectangular Auction vs padded Hungarian parity (50 sparse matrices)
  for (let t = 0; t < 50; t++) {
    const m = 4 + (t % 9);
    const n = 3 + ((t * 3) % 8);
    const N = m + n;
    const B = Array.from({ length: m }, (_, i) =>
      Array.from({ length: n }, (_, j) => ((i * 31 + j * 17 + t * 13) % 5 === 0 ? 0 : ((i * 73 + j * 41 + t * 29) % 200) + 1))
    );
    const cost = new Float64Array(N * N);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) cost[i * N + j] = -B[i][j];
    const hAssign = solveHungarianMinCost(cost, N);
    let hOpt = 0;
    for (let i = 0; i < m; i++) if (hAssign[i] < n) hOpt += B[i][hAssign[i]];

    const edgeTo = new Int32Array(m * n).fill(-1);
    const edgeBen = new Int32Array(m * n);
    const edgeCnt = new Int32Array(m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (B[i][j] > 0) {
          const e = edgeCnt[i]++;
          edgeTo[i * n + e] = j;
          edgeBen[i * n + e] = B[i][j];
        }
      }
    }
    const { trackToObs: aAssign, capHits, roundLimitHits } = solveAuctionAsym(edgeTo, edgeBen, edgeCnt, m, n, n);
    let aScore = 0;
    for (let i = 0; i < m; i++) if (aAssign[i] >= 0) aScore += B[i][aAssign[i]];
    assert(aScore === hOpt && capHits === 0 && roundLimitHits === 0, `Random matrix t=${t} Auction=${aScore} != Hungarian=${hOpt}`);
  }

  // 9. Exact TrackEval 9-frame swap ambiguity HOTA oracle (expected 0.9356381132363039)
  {
    const ev = createClearMotEvaluator(4.5);
    for (let f = 0; f < 9; f++) {
      const gts = [{ gtId: 1, x: 100, y: 100 }, { gtId: 2, x: 104, y: 100 }];
      const trks = f === 1
        ? [{ tid: 10, x: 104, y: 100 }, { tid: 20, x: 100, y: 100 }]
        : [{ tid: 10, x: 100, y: 100 }, { tid: 20, x: 104, y: 100 }];
      ev.update(gts, trks);
    }
    const s = ev.summary();
    assert(Math.abs(s.hota - 0.9356381132363039) < 1e-10, `TrackEval 9f swap HOTA=${s.hota} != 0.9356381132363039`);
  }

  // 10. Morton10 algebraic 10-bit interleaving verification ((2,0) === 4, round-trip bit exact)
  {
    assert(computeMorton10(0, 0) === 0, 'Morton(0,0)');
    assert(computeMorton10(1, 0) === 1, 'Morton(1,0)');
    assert(computeMorton10(0, 1) === 2, 'Morton(0,1)');
    assert(computeMorton10(2, 0) === 4, 'Morton(2,0)===4');
    assert(computeMorton10(3, 3) === 15, 'Morton(3,3)===15');
    assert(computeMorton10(1023, 1023) === 0xfffff, 'Morton(1023,1023)');
    for (const [xi, yi] of [[12, 345], [639, 359], [512, 256], [1023, 0], [0, 1023], [777, 888]]) {
      let ref = 0;
      for (let b = 0; b < 10; b++) {
        ref |= ((xi >> b) & 1) << (2 * b);
        ref |= ((yi >> b) & 1) << (2 * b + 1);
      }
      assert(computeMorton10(xi, yi) === ref, `Morton(${xi},${yi}) mismatch`);
    }
  }

  // 11. Starling 1-to-1 pixel/object oracle checks (F1..F7 adversarial regression checks)
  {
    const n = HD_W * HD_H;
    const fPrev = new Uint8Array(n * 4).fill(255);
    const fCurr = new Uint8Array(n * 4).fill(255);
    const setPx = (buf, px, py, val) => {
      if (px < 0 || px >= HD_W || py < 0 || py >= HD_H) return;
      const k = (py * HD_W + px) * 4;
      buf[k] = val; buf[k + 1] = val; buf[k + 2] = val; buf[k + 3] = 255;
    };
    const stampDarkBlob = (buf, cx, cy, amp = 160.0, bg = 220.0, sigma = 1.1) => {
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          const ix = Math.round(cx) + dx;
          const iy = Math.round(cy) + dy;
          if (ix < 0 || ix >= HD_W || iy < 0 || iy >= HD_H) continue;
          const d2 = (ix - cx) ** 2 + (iy - cy) ** 2;
          const v = Math.max(0, Math.min(255, Math.round(bg - amp * Math.exp(-d2 / (2 * sigma * sigma)))));
          setPx(buf, ix, iy, v);
        }
      }
    };

    // F1: 1px 1000-track spoof => 1/1000 = 0.001
    setPx(fCurr, 320, 180, 0);
    const dup1000 = Array.from({ length: 1000 }, (_, i) => ({ tid: i + 1, x: 320, y: 180 }));
    const p1 = measureStarlingDarkExtremaPrecision(fCurr, fPrev, dup1000);
    assert(Math.abs(p1 - 0.001) < 1e-9, `F1 1px 1000-track precision=${p1} != 0.001`);

    // 3x3 10-stacked + F2 flicker moving vs static
    const fPrevB = new Uint8Array(n * 4).fill(200);
    const fCurrB = new Uint8Array(n * 4).fill(200);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        setPx(fPrevB, 319 + dx, 180 + dy, 40);
        setPx(fCurrB, 320 + dx, 180 + dy, 40);
      }
    }
    const pStack = measureStarlingDarkExtremaPrecision(fCurrB, fPrevB, Array.from({ length: 10 }, () => ({ x: 320, y: 180 })));
    assert(Math.abs(pStack - 0.1) < 1e-9, `3x3 10-stacked precision=${pStack} != 0.1`);

    const fFlickerMove = new Uint8Array(fCurrB);
    const fFlickerStat = new Uint8Array(fPrevB);
    for (let i = 0; i < n * 4; i += 4) {
      fFlickerMove[i] = fFlickerMove[i + 1] = fFlickerMove[i + 2] = Math.min(255, fFlickerMove[i] + 25);
      fFlickerStat[i] = fFlickerStat[i + 1] = fFlickerStat[i + 2] = Math.min(255, fFlickerStat[i] + 25);
    }
    assert(measureStarlingDarkExtremaPrecision(fFlickerMove, fPrevB, [{ x: 320, y: 180 }]) === 1.0, 'F2 Starling moving flicker');
    assert(measureStarlingDarkExtremaPrecision(fFlickerStat, fPrevB, [{ x: 319, y: 180 }]) === 0.0, 'F2 Starling static flicker reject');

    // F3: Uniform x0.8 multiplicative gain drop on static dark spots must normalize & return 0
    const fGainPrev = new Uint8Array(n * 4).fill(220);
    const fGainCurr = new Uint8Array(n * 4);
    for (const [bx, by] of [[120, 90], [260, 140], [420, 220]]) {
      stampDarkBlob(fGainPrev, bx, by, 140.0, 220.0, 1.1);
    }
    for (let i = 0; i < n * 4; i += 4) {
      const v = Math.round(fGainPrev[i] * 0.8);
      fGainCurr[i] = fGainCurr[i + 1] = fGainCurr[i + 2] = v;
      fGainCurr[i + 3] = 255;
    }
    assert(measureStarlingDarkExtremaPrecision(fGainCurr, fGainPrev, [{ x: 120, y: 90 }]) === 0.0, 'F3 x0.8 gain static reject');

    // F4: Camera pan dx=+2 px across static spots must normalize pan & return 0
    const fPanPrev = new Uint8Array(n * 4).fill(220);
    const fPanCurr = new Uint8Array(n * 4).fill(220);
    for (let k = 0; k < 12; k++) {
      const bx = 90 + k * 42;
      const by = 90 + (k % 3) * 70;
      stampDarkBlob(fPanPrev, bx, by, 140.0, 220.0, 1.1);
      stampDarkBlob(fPanCurr, bx + 2, by, 140.0, 220.0, 1.1);
    }
    assert(measureStarlingDarkExtremaPrecision(fPanCurr, fPanPrev, [{ x: 92, y: 90 }]) === 0.0, 'F4 pan dx=2 static reject');

    // F5: Connected dark silhouette with 2 local extrema groups to 1 object (precision = 1/2 = 0.5)
    const fConnPrev = new Uint8Array(n * 4).fill(200);
    const fConnCurr = new Uint8Array(n * 4).fill(200);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -4; dx <= 4; dx++) {
        setPx(fConnPrev, 319 + dx, 180 + dy, 60);
        setPx(fConnCurr, 320 + dx, 180 + dy, 60);
      }
    }
    setPx(fConnPrev, 317, 180, 30); setPx(fConnPrev, 321, 180, 30);
    setPx(fConnCurr, 318, 180, 30); setPx(fConnCurr, 322, 180, 30);
    const pConn = measureStarlingDarkExtremaPrecision(fConnCurr, fConnPrev, [{ x: 318, y: 180 }, { x: 322, y: 180 }]);
    assert(Math.abs(pConn - 0.5) < 1e-9, `F5 connected 2-extrema precision=${pConn} != 0.5`);

    // F6 vs F7: Tiny subpixel shift dx=0.01 rejected vs true motion dx=0.05 accepted
    const fSubPrev = new Uint8Array(n * 4).fill(220);
    const fSubStill = new Uint8Array(n * 4).fill(220);
    const fSubMove = new Uint8Array(n * 4).fill(220);
    stampDarkBlob(fSubPrev, 320.0, 180.0, 160.0, 220.0, 1.1);
    stampDarkBlob(fSubStill, 320.01, 180.0, 160.0, 220.0, 1.1);
    stampDarkBlob(fSubMove, 320.05, 180.0, 160.0, 220.0, 1.1);
    assert(measureStarlingDarkExtremaPrecision(fSubStill, fSubPrev, [{ x: 320.01, y: 180 }]) === 0.0, 'F6 dx=0.01 reject');
    assert(measureStarlingDarkExtremaPrecision(fSubMove, fSubPrev, [{ x: 320.05, y: 180 }]) === 1.0, 'F7 dx=0.05 accept');

    // F8 & F9: Subpixel global camera pans dx=0.05 & dx=0.5 across 12 static spots must normalize & return 0.0
    const stampDarkAt = (buf, w, h, cx, cy, amp = 140.0, bg = 220.0, sigma = 1.1) => {
      for (let dy = -3; dy <= 3; dy++) {
        for (let dx = -3; dx <= 3; dx++) {
          const ix = Math.round(cx) + dx;
          const iy = Math.round(cy) + dy;
          if (ix < 0 || ix >= w || iy < 0 || iy >= h) continue;
          const d2 = (ix - cx) ** 2 + (iy - cy) ** 2;
          const v = Math.max(0, Math.min(255, Math.round(bg - amp * Math.exp(-d2 / (2 * sigma * sigma)))));
          const k = (iy * w + ix) * 4;
          buf[k] = v; buf[k + 1] = v; buf[k + 2] = v; buf[k + 3] = 255;
        }
      }
    };
    for (const shiftX of [0.05, 0.5]) {
      const spPrev = new Uint8Array(n * 4).fill(220);
      const spCurr = new Uint8Array(n * 4).fill(220);
      for (let k = 0; k < 12; k++) {
        const bx = 90 + k * 42;
        const by = 90 + (k % 3) * 70;
        stampDarkAt(spPrev, HD_W, HD_H, bx, by);
        stampDarkAt(spCurr, HD_W, HD_H, bx + shiftX, by);
      }
      assert(measureStarlingDarkExtremaPrecision(spCurr, spPrev, [{ x: 90 + shiftX, y: 90 }]) === 0.0, `F8/F9 subpixel pan dx=${shiftX} reject`);
    }

    // F10: Connected dark dumbbell with 2 extrema 12px apart groups to 1 component (precision = 0.5)
    const fConn12Prev = new Uint8Array(n * 4).fill(200);
    const fConn12Curr = new Uint8Array(n * 4).fill(200);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -7; dx <= 7; dx++) {
        setPx(fConn12Prev, 319 + dx, 180 + dy, 60);
        setPx(fConn12Curr, 320 + dx, 180 + dy, 60);
      }
    }
    setPx(fConn12Prev, 313, 180, 30); setPx(fConn12Prev, 325, 180, 30);
    setPx(fConn12Curr, 314, 180, 30); setPx(fConn12Curr, 326, 180, 30);
    const pConn12 = measureStarlingDarkExtremaPrecision(fConn12Curr, fConn12Prev, [{ x: 314, y: 180 }, { x: 326, y: 180 }]);
    assert(Math.abs(pConn12 - 0.5) < 1e-9, `F10 connected12 2-extrema precision=${pConn12} != 0.5`);

    // F11: SD (320x180) 6px pan reject & connected12 grouping
    const sdN = 320 * 180;
    const sdPanPrev = new Uint8Array(sdN * 4).fill(220);
    const sdPanCurr = new Uint8Array(sdN * 4).fill(220);
    for (let k = 0; k < 12; k++) {
      const bx = 45 + k * 21;
      const by = 45 + (k % 3) * 35;
      stampDarkAt(sdPanPrev, 320, 180, bx, by);
      stampDarkAt(sdPanCurr, 320, 180, bx + 6, by);
    }
    assert(measureStarlingDarkExtremaPrecision(sdPanCurr, sdPanPrev, [{ x: 51, y: 45 }], 320, 180) === 0.0, 'F11 SD 6px pan reject');

    // F12: Exact 84-dot (12x7) SD (320x180) 2px camera pan must normalize & reject all pan-induced false positives
    const setSd = (buf, x, y, v) => {
      const k = (y * 320 + x) * 4;
      buf[k] = v; buf[k + 1] = v; buf[k + 2] = v; buf[k + 3] = 255;
    };
    const sd84Prev = new Uint8Array(sdN * 4).fill(200);
    const sd84Curr = new Uint8Array(sdN * 4).fill(200);
    const sd84Tracks = [];
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 12; col++) {
        const bx = 28 + col * 24;
        const by = 24 + row * 22;
        setSd(sd84Prev, bx, by, 40);
        setSd(sd84Curr, bx + 2, by, 40);
        sd84Tracks.push({ x: bx + 2, y: by });
      }
    }
    assert(measureStarlingDarkExtremaPrecision(sd84Curr, sd84Prev, sd84Tracks, 320, 180) === 0.0, 'F12 SD 84-dot 2px pan reject');

    // F13: Border pans at x=5->x=7 (both 640x360 and 320x180) must normalize & return 0.0
    for (const [bw, bh] of [[640, 360], [320, 180]]) {
      const bpPrev = new Uint8Array(bw * bh * 4).fill(200);
      const bpCurr = new Uint8Array(bw * bh * 4).fill(200);
      const bpTracks = [];
      for (let y = 30; y < bh - 20; y += 19) {
        const kP = (y * bw + 5) * 4;
        const kC = (y * bw + 7) * 4;
        bpPrev[kP] = bpPrev[kP + 1] = bpPrev[kP + 2] = 40; bpPrev[kP + 3] = 255;
        bpCurr[kC] = bpCurr[kC + 1] = bpCurr[kC + 2] = 40; bpCurr[kC + 3] = 255;
        bpTracks.push({ x: 7, y });
      }
      assert(measureStarlingDarkExtremaPrecision(bpCurr, bpPrev, bpTracks, bw, bh) === 0.0, `F13 borderPan2_${bw} reject`);
    }

    // F14: Static contrast-inverted Gaussian texture (HD & SD) must not match bogus diagonal shifts -> 0.0
    for (const [cw, ch] of [[640, 360], [320, 180]]) {
      const invPrev = new Uint8Array(cw * ch * 4).fill(40);
      const invCurr = new Uint8Array(cw * ch * 4).fill(220);
      const invTracks = [];
      for (let k = 0; k < 12; k++) {
        const bx = Math.round((45 + k * 21) * (cw / 320));
        const by = Math.round((45 + (k % 3) * 35) * (ch / 180));
        stampDarkAt(invPrev, cw, ch, bx, by, -140.0, 40.0, 1.1);
        stampDarkAt(invCurr, cw, ch, bx, by, 140.0, 220.0, 1.1);
        invTracks.push({ x: bx, y: by });
      }
      assert(measureStarlingDarkExtremaPrecision(invCurr, invPrev, invTracks, cw, ch) === 0.0, `F14 contrast-inverted static ${cw}x${ch} reject`);
    }

    // F15: Entering static texture across border during pan=5 & static partial contrast inversion -> 0.0
    for (const [ew, eh] of [[640, 360], [320, 180]]) {
      const epPrev = new Uint8Array(ew * eh * 4).fill(200);
      const epCurr = new Uint8Array(ew * eh * 4).fill(200);
      const epTracks = [];
      for (let y = 30; y < eh - 20; y += 29) {
        for (let u = -1; u < ew + 1; u += 24) {
          if (u >= 0 && u < ew) {
            const kp = (y * ew + u) * 4;
            epPrev[kp] = epPrev[kp + 1] = epPrev[kp + 2] = 40; epPrev[kp + 3] = 255;
          }
          const cu = u + 5;
          if (cu >= 0 && cu < ew) {
            const kc = (y * ew + cu) * 4;
            epCurr[kc] = epCurr[kc + 1] = epCurr[kc + 2] = 40; epCurr[kc + 3] = 255;
            if (cu === 4) epTracks.push({ x: cu, y });
          }
        }
      }
      assert(measureStarlingDarkExtremaPrecision(epCurr, epPrev, epTracks, ew, eh) === 0.0, `F15 enteringBorderPan5 ${ew}x${eh} reject`);

      const piPrev = new Uint8Array(ew * eh * 4).fill(128);
      const piCurr = new Uint8Array(ew * eh * 4).fill(128);
      const piTracks = [];
      for (let k = 0; k < 12; k++) {
        const bx = Math.round((45 + k * 21) * (ew / 320));
        const by = Math.round((45 + (k % 3) * 35) * (eh / 180));
        stampDarkAt(piPrev, ew, eh, bx, by, -100.0, 128.0, 1.1);
        stampDarkAt(piCurr, ew, eh, bx, by, k < 6 ? 100.0 : -100.0, 128.0, 1.1);
        if (k < 6) piTracks.push({ x: bx, y: by });
      }
      assert(measureStarlingDarkExtremaPrecision(piCurr, piPrev, piTracks, ew, eh) === 0.0, `F15 partialInversion6of12 ${ew}x${eh} reject`);
    }

    // F16: SD 4px inverted checkerboard tie-breaks zero-motion inverse fit over periodic alias -> 0.0
    {
      const cw = 320, ch = 180;
      const ckPrev = new Uint8Array(cw * ch * 4);
      const ckCurr = new Uint8Array(cw * ch * 4);
      const ckTracks = [];
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
          const v = ((Math.floor(x / 4) + Math.floor(y / 4)) % 2) ? 40 : 220;
          const o = (y * cw + x) * 4;
          ckPrev[o] = ckPrev[o + 1] = ckPrev[o + 2] = v; ckPrev[o + 3] = 255;
          ckCurr[o] = ckCurr[o + 1] = ckCurr[o + 2] = 260 - v; ckCurr[o + 3] = 255;
          if (x >= 3 && x < cw - 3 && y >= 3 && y < ch - 3 && x % 31 === 5 && y % 23 === 5 && v === 220) {
            ckTracks.push({ x, y });
          }
        }
      }
      assert(measureStarlingDarkExtremaPrecision(ckCurr, ckPrev, ckTracks, cw, ch) === 0.0, 'F16 SD 4px inverted checkerboard reject');
    }
  }

  // 12. Binary & disassembly fail-closed checks (reserve, 14336B stack, allocator symbols, incomplete disasm)
  {
    const mem448 = new WebAssembly.Memory({ initial: EXPECTED_WASM_PAGES, maximum: EXPECTED_WASM_PAGES });
    const baseOd = {
      available: true,
      memInitialPages: EXPECTED_WASM_PAGES,
      memMaxPages: EXPECTED_WASM_PAGES,
      stackPointerInit: 16384,
      dataSegments: [{ init: 16384, size: 1024, end: 17408 }],
      maxDataEnd: 17408,
      exportFuncCount: 30,
      analyzedFuncCount: 40,
      missingExportFuncs: 0,
      disassemblyComplete: true,
      allocatorCallCount: 0,
      allocatorReachableCount: 0,
      v128Count: 100,
      clzCount: 2,
      memoryGrowCount: 0,
      callIndirectCount: 0,
      maxShadowStackDepth: 8192,
      stackCycles: 0,
    };
    const okMeta = {
      validHeader: true,
      importCount: 0,
      hasMemorySection: true,
      hasExportSection: true,
      memories: [{ flags: 1, initial: EXPECTED_WASM_PAGES, max: EXPECTED_WASM_PAGES }],
      globals: [{ initI32: 16384 }],
      dataSegments: [],
      maxDataEnd: 17408,
    };
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: baseOd }).length === 0, 'audit baseline ok');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { available: false } }).length > 0, 'fail without objdump');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, stackCycles: 1 } }).length > 0, 'fail on call cycle');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, callIndirectCount: 1 } }).length > 0, 'fail on call_indirect');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, maxShadowStackDepth: 15000 } }).length > 0, 'fail on 15000B stack (>14336)');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, clzCount: 0 } }).length > 0, 'fail on 0 i32.clz');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, disassemblyComplete: false } }).length > 0, 'fail on incomplete disasm');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, analyzedFuncCount: 0 } }).length > 0, 'fail on 0 analyzed funcs');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, missingExportFuncs: 1 } }).length > 0, 'fail on missing export disasm');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, allocatorCallCount: 1 } }).length > 0, 'fail on allocator calls');
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: { ...baseOd, allocatorReachableCount: 1 } }).length > 0, 'fail on reachable allocator');
    assert(
      auditWasmBinaryAndAbi({
        imports: [],
        mem: mem448,
        binaryMeta: { ...okMeta, maxDataEnd: EXPECTED_WASM_MEM_BYTES - 1 },
        objdumpReport: { ...baseOd, maxDataEnd: EXPECTED_WASM_MEM_BYTES - 1 },
      }).length > 0,
      'fail on 1-byte static reserve'
    );
    // Verify real objdump parser catches allocator symbol reachable from export & incomplete disasm
    const synX = [
      'Sections:',
      ` - memory[0] pages: initial=${EXPECTED_WASM_PAGES} max=${EXPECTED_WASM_PAGES}`,
      ' - global[0] i32 mutable=1 - init i32=16384',
      ' - func[0] <chronos_post_tick> -> "chronos_post_tick"',
      ' - segment[0] <.rodata> memory=0 size=1024 - init i32=16384',
    ].join('\n');
    const synDAlloc = [
      '000000 func[0] <chronos_post_tick>:',
      '   call 1 <__rust_alloc>',
      '   v128.load 0',
      '   i32.clz',
      '   end',
      '000020 func[1] <__rust_alloc>:',
      '   end',
    ].join('\n');
    const parsedAlloc = analyzeObjdumpStatic(synX, synDAlloc);
    assert(parsedAlloc.allocatorCallCount === 1 && parsedAlloc.allocatorReachableCount >= 1, `parser exact single-pass alloc count got call=${parsedAlloc.allocatorCallCount}, reach=${parsedAlloc.allocatorReachableCount}`);
    const parsedInc = analyzeObjdumpStatic(synX, '000000 func[0] <chronos_post_tick>:\n   nop\n');
    assert(!parsedInc.disassemblyComplete && parsedInc.unterminatedFuncs === 1, 'parser rejects unclosed/truncated disassembly');
    const synDMissing = [
      '000000 func[0] <chronos_post_tick>:',
      '   call 99 <missing_fn>',
      '   v128.load 0',
      '   i32.clz',
      '   end',
    ].join('\n');
    const parsedMissing = analyzeObjdumpStatic(synX, synDMissing);
    assert(!parsedMissing.disassemblyComplete && parsedMissing.missingCalleeCount === 1, 'parser rejects missing internal callee');
    const synDAddStack = [
      '000000 func[0] <chronos_post_tick>:',
      '   global.get 0',
      '   i32.const -15000',
      '   i32.add',
      '   local.tee 0',
      '   global.set 0',
      '   v128.load 0',
      '   i32.clz',
      '   local.get 0',
      '   i32.const 15000',
      '   i32.add',
      '   global.set 0',
      '   end',
    ].join('\n');
    const parsedAddStack = analyzeObjdumpStatic(synX, synDAddStack);
    assert(parsedAddStack.maxShadowStackDepth === 15000, `parser detects i32.const -15000; i32.add stack frame (got ${parsedAddStack.maxShadowStackDepth})`);
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: parsedAddStack }).length > 0, 'audit rejects 15000B add-prologue frame');
    const synDBypass = [
      '000000 func[0] <chronos_post_tick>:',
      '   global.get 0',
      '   i32.const 15000',
      '   i32.sub',
      '   local.set 0',
      '   local.get 0',
      '   i32.const 0',
      '   i32.add',
      '   global.set 0',
      '   end',
    ].join('\n');
    const parsedBypass = analyzeObjdumpStatic(synX, synDBypass);
    assert(parsedBypass.unknownStackWriteCount >= 1, `parser rejects SP-bypass local.set/add=0 (got ${parsedBypass.unknownStackWriteCount})`);
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: parsedBypass }).length > 0, 'audit rejects SP bypass');
    const synDClobber = [
      '000000 func[0] <chronos_post_tick>:',
      '   global.get 0',
      '   i32.const 64',
      '   i32.sub',
      '   local.tee 0',
      '   global.set 0',
      '   i32.const 0',
      '   local.set 0',
      '   local.get 0',
      '   i32.const 64',
      '   i32.add',
      '   global.set 0',
      '   end',
    ].join('\n');
    const parsedClobber = analyzeObjdumpStatic(synX, synDClobber);
    assert(parsedClobber.unknownStackWriteCount >= 1, `parser rejects post-prologue clobbered FP epilogue (got ${parsedClobber.unknownStackWriteCount})`);
    const synDRestoreBypass = [
      '000000 func[0] <chronos_post_tick>:',
      '   local.get 0',
      '   i32.const 16',
      '   i32.add',
      '   global.set 0',
      '   end',
    ].join('\n');
    const parsedRestoreBypass = analyzeObjdumpStatic(synX, synDRestoreBypass);
    assert(parsedRestoreBypass.unknownStackWriteCount >= 1, `parser rejects unanchored epilogue restore (got ${parsedRestoreBypass.unknownStackWriteCount})`);

    // Loop backedge & sequential cumulative stack depth checks
    const proOps = (sz) => ['global.get 0', `i32.const ${sz}`, 'i32.sub', 'local.tee 0', 'global.set 0'];
    const restore64 = ['local.get 0', 'i32.const 64', 'i32.add', 'global.set 0'];
    const loopTail = ['local.get 1', 'i32.const 1', 'i32.sub', 'local.tee 1', 'br_if 0', 'end'];
    const wrapOps = (ops) =>
      ['000000 func[0] <chronos_post_tick>:', '   v128.load 0', '   i32.clz', ...ops.map((o) => `   ${o}`), '   end'].join('\n');

    const parsedLoopClobber = analyzeObjdumpStatic(
      synX,
      wrapOps([...proOps(64), 'i32.const 2', 'local.set 1', 'loop', ...restore64, 'i32.const 0', 'local.set 0', ...loopTail])
    );
    assert(parsedLoopClobber.unknownStackWriteCount >= 1, `parser rejects loopClobberBackedge (got ${parsedLoopClobber.unknownStackWriteCount})`);

    const parsedLoopDepthLeak = analyzeObjdumpStatic(
      synX,
      wrapOps(['i32.const 16', 'local.set 1', 'loop', ...proOps(1024), ...loopTail, 'local.get 0', 'i32.const 1024', 'i32.add', 'global.set 0'])
    );
    assert(parsedLoopDepthLeak.unknownStackWriteCount >= 1, `parser rejects loopDepthLeak (got ${parsedLoopDepthLeak.unknownStackWriteCount})`);

    const parsedSeqLeak = analyzeObjdumpStatic(synX, wrapOps([...proOps(8000), ...proOps(8000)]));
    assert(parsedSeqLeak.maxShadowStackDepth === 16000, `parser detects sequentialDepthLeak=16000B (got ${parsedSeqLeak.maxShadowStackDepth})`);
    assert(auditWasmBinaryAndAbi({ imports: [], mem: mem448, binaryMeta: okMeta, objdumpReport: parsedSeqLeak }).length > 0, 'audit rejects sequential 16000B stack');

    const parsedBalancedLoop = analyzeObjdumpStatic(
      synX,
      wrapOps([...proOps(64), 'i32.const 4', 'local.set 1', 'loop', 'local.get 1', 'i32.const 2', 'local.set 2', ...loopTail, ...restore64])
    );
    assert(parsedBalancedLoop.unknownStackWriteCount === 0 && parsedBalancedLoop.maxShadowStackDepth === 64, 'balanced non-FP loop passes cleanly');

    // Join inexact FP provenance, local.set value consumption, and unbalanced callee exit regressions
    const parsedJoinInexact = analyzeObjdumpStatic(
      synX,
      wrapOps(['local.get 1', 'if', 'global.get 0', 'i32.const 8000', 'i32.sub', 'local.tee 2', 'global.set 0', 'end', ...proOps(64), 'local.get 0', 'i32.const 8064', 'i32.add', 'global.set 0'])
    );
    assert(parsedJoinInexact.unknownStackWriteCount >= 1, `parser rejects joinCreatesInexactFP (got ${parsedJoinInexact.unknownStackWriteCount})`);

    const parsedPartialJoinInexact = analyzeObjdumpStatic(
      synX,
      wrapOps([...proOps(8000), 'local.get 1', 'if', 'local.get 0', 'i32.const 4000', 'i32.add', 'global.set 0', 'end', ...proOps(64), 'local.get 0', 'i32.const 8064', 'i32.add', 'global.set 0'])
    );
    assert(parsedPartialJoinInexact.unknownStackWriteCount >= 1, `parser rejects partialRestoreJoinCreatesInexactFP (got ${parsedPartialJoinInexact.unknownStackWriteCount})`);

    const parsedLocalSetConsume = analyzeObjdumpStatic(
      synX,
      wrapOps(['i32.const 0', 'global.get 0', 'i32.const 64', 'i32.sub', 'local.set 0', 'global.set 0'])
    );
    assert(parsedLocalSetConsume.unknownStackWriteCount >= 1, `parser rejects localSetConsumesValue (got ${parsedLocalSetConsume.unknownStackWriteCount})`);

    const synX2 = [
      'Sections:',
      ' - memory[0] pages: initial=320 max=320',
      ' - global[0] i32 mutable=1 - init i32=16384',
      ' - func[0] <chronos_post_tick> -> "chronos_post_tick"',
      ' - func[1] <leak_callee>',
    ].join('\n');
    const parsedCalleeLeakEnd = analyzeObjdumpStatic(
      synX2,
      [
        wrapOps(['call 1', 'call 1']),
        ['000040 func[1] <leak_callee>:', ...proOps(8000).map((o) => `   ${o}`), '   end'].join('\n'),
      ].join('\n')
    );
    assert(parsedCalleeLeakEnd.unknownStackWriteCount >= 1, 'parser rejects unbalanced callee end exit');

    const parsedCalleeLeakRet = analyzeObjdumpStatic(
      synX2,
      [
        wrapOps(['call 1', 'call 1']),
        ['000040 func[1] <leak_callee>:', ...proOps(8000).map((o) => `   ${o}`), '   return', '   end'].join('\n'),
      ].join('\n')
    );
    assert(parsedCalleeLeakRet.unknownStackWriteCount >= 1, 'parser rejects unbalanced callee return exit');
  }

  // 13. inspectFrameBuffers sentinel & swarmTelem[39]/[41] checks
  {
    const mkSess = (active = true) => {
      const s = {
        arenaBytes: 18_500_000,
        telem: new Float32Array(TELEMETRY_LEN),
        microTracks: new Float32Array(MAX_MICRO * MICRO_STRIDE),
        continuumField: new Float32Array(FIELD_CELLS * FIELD_STRIDE),
        rigidBodies: new Float32Array(MAX_RIGID_BODIES * BODY_STRIDE),
        swarmTelem: new Float32Array(SWARM_TELEM_LEN),
        modalField: new Float32Array(FIELD_CELLS).fill(active ? -9.0 : 0.0),
      };
      s.telem[20] = 18_500_000;
      for (let c = 0; c < FIELD_CELLS; c++) s.continuumField[c * FIELD_STRIDE + 4] = active ? -999.0 : 0.0;
      if (active) {
        s.swarmTelem[1] = 1.0;
        s.swarmTelem[3] = 200;
        s.swarmTelem[39] = 150;
        s.swarmTelem[41] = 0.9;
      }
      return s;
    };
    assert(inspectFrameBuffers(mkSess(true)).rangeViolations === 0, 'clean active buffer passes');
    assert(inspectFrameBuffers(mkSess(false)).rangeViolations === 0, 'clean inactive buffer passes');
    const badW = mkSess(true); badW.continuumField[5] = -0.1;
    assert(inspectFrameBuffers(badW).rangeViolations === 1, 'negative continuum weight rejected');
    const badZeroU = mkSess(true); badZeroU.continuumField[5] = 0.0; badZeroU.continuumField[0] = 2.0;
    assert(inspectFrameBuffers(badZeroU).rangeViolations === 1, 'nonzero velocity on zero weight rejected');
    const badFtle = mkSess(true); badFtle.continuumField[4] = -950.0;
    assert(inspectFrameBuffers(badFtle).rangeViolations === 1, 'invalid FTLE -950 sentinel rejected');
    const badInactFtle = mkSess(false); badInactFtle.continuumField[4] = 1.2;
    assert(inspectFrameBuffers(badInactFtle).rangeViolations === 1, 'inactive non-zero FTLE rejected');
    const badInactNeg999 = mkSess(false); badInactNeg999.continuumField[4] = -999.0;
    assert(inspectFrameBuffers(badInactNeg999).rangeViolations === 1, 'inactive -999 FTLE sentinel rejected');
    const badModal = mkSess(true); badModal.modalField[0] = 4.0;
    assert(inspectFrameBuffers(badModal).rangeViolations === 1, 'out-of-range modal phase 4.0 rejected');
    const badInactModal = mkSess(false); badInactModal.modalField[0] = -9.0;
    assert(inspectFrameBuffers(badInactModal).rangeViolations === 1, 'inactive -9.0 modal sentinel rejected');
    const badM39Frac = mkSess(true); badM39Frac.swarmTelem[39] = 10.5;
    assert(inspectFrameBuffers(badM39Frac).rangeViolations === 1, 'fractional swarmTelem[39] rejected');
    const badM39Inact = mkSess(false); badM39Inact.swarmTelem[39] = 1.0;
    assert(inspectFrameBuffers(badM39Inact).rangeViolations === 1, 'inactive nonzero swarmTelem[39] rejected');
  }

  // 15. Starling v9 precision & contrast-inversion regression fixtures
  {
    const mkFrame = (w, h, bg = 220) => new Uint8Array(w * h * 4).fill(bg);
    const putPx = (buf, w, h, x, y, v) => {
      if (x < 0 || x >= w || y < 0 || y >= h) return;
      const o = (y * w + x) * 4;
      buf[o] = buf[o + 1] = buf[o + 2] = v;
      buf[o + 3] = 255;
    };
    const putBlob = (buf, w, h, cx, cy, amp = 140, bg = 220, sig = 1.1) => {
      for (let y = Math.round(cy) - 3; y <= Math.round(cy) + 3; y++) {
        for (let x = Math.round(cx) - 3; x <= Math.round(cx) + 3; x++) {
          const v = bg - amp * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig * sig));
          putPx(buf, w, h, x, y, Math.max(0, Math.min(255, Math.round(v))));
        }
      }
    };
    const putGaussSum = (w, h, comps, bg = 180) => {
      const a = mkFrame(w, h, bg);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let v = bg;
          for (const [cx, cy, amp, sig] of comps) {
            if (Math.abs(x - cx) > 5 * sig || Math.abs(y - cy) > 5 * sig) continue;
            v += amp * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig * sig));
          }
          putPx(a, w, h, x, y, Math.round(Math.max(0, Math.min(255, v))));
        }
      }
      return a;
    };

    const putPoint = (buf, w, h, cx, cy) => {
      for (let y = Math.floor(cy); y <= Math.ceil(cy); y++) {
        for (let x = Math.floor(cx); x <= Math.ceil(cx); x++) {
          putPx(buf, w, h, x, y, Math.round(220 - 200 * (1 - Math.abs(x - cx)) * (1 - Math.abs(y - cy))));
        }
      }
    };

    // movingParticleBright30 & arrival25_bright10: dark particle arriving onto stationary bright spot scores 1.0
    {
      const w = 320, h = 180, x = w / 2, y = h / 2;
      const p = putGaussSum(w, h, [[x, y, 30, 1.1], [x - 7, y, -140, 1.1]], 180);
      const a = putGaussSum(w, h, [[x, y, 30, 1.1], [x, y, -140, 1.1]], 180);
      assert(measureStarlingDarkExtremaPrecision(a, p, [{ x, y }], w, h) === 1.0, 'movingParticleBright30 scores 1.0');
      const p25 = putGaussSum(w, h, [[x, y, 10, 1.1], [x - 25, y, -140, 1.1]], 180);
      const a25 = putGaussSum(w, h, [[x, y, 10, 1.1], [x, y, -140, 1.1]], 180);
      assert(measureStarlingDarkExtremaPrecision(a25, p25, [{ x, y }], w, h) === 1.0, 'arrival25_bright10 scores 1.0');
    }

    // partial3_unrelatedDim12: unrelated fading dark Gaussian must not excuse static partial inversion
    {
      const w = 320, h = 180, p = mkFrame(w, h, 128), a = mkFrame(w, h, 128), ts = [];
      for (let k = 0; k < 12; k++) {
        const x = Math.round((45 + k * 21) * w / 320), y = Math.round((45 + (k % 3) * 35) * h / 180);
        putBlob(p, w, h, x, y, -100, 128);
        putBlob(a, w, h, x, y, k < 3 ? 100 : -100, 128);
        if (k < 3) ts.push({ x, y });
      }
      putBlob(p, w, h, ts[0].x, ts[0].y + 12, 40, 128);
      putBlob(a, w, h, ts[0].x, ts[0].y + 12, 0, 128);
      assert(measureStarlingDarkExtremaPrecision(a, p, ts, w, h) === 0.0, 'partial3_unrelatedDim12 rejected');
    }

    // partialInv3_pan5, (-2.5, 0.5), (20, 0), and top-band (-15, 0): partial inversion under pan must reject (0.0)
    for (const [dx, dy, topBand] of [[5, 0, false], [-2.5, 0.5, false], [20, 0, false], [-15, 0, true]]) {
      const w = 320, h = 180, p = mkFrame(w, h, 128), a = mkFrame(w, h, 128), ts = [];
      for (let k = 0; k < 12; k++) {
        const x = Math.round(((topBand ? 30 + k * 23 : 45 + k * 21) * w) / 320);
        const y = topBand ? 8 : Math.round(((45 + (k % 3) * 35) * h) / 180);
        putBlob(p, w, h, x, y, -100, 128);
        putBlob(a, w, h, x + dx, y + dy, k < 3 ? 100 : -100, 128);
        if (k < 3) ts.push({ x: x + dx, y: y + dy });
      }
      assert(
        measureStarlingDarkExtremaPrecision(a, p, ts, w, h) === 0.0,
        `partialInv3_pan(${dx},${dy},top=${topBand}) rejected`
      );
    }

    // weakPartialInv1_amp10: weak negative amplitude inversion (+100 -> -10) must reject (0.0)
    {
      const w = 320, h = 180, p = mkFrame(w, h, 128), a = mkFrame(w, h, 128), ts = [];
      for (let k = 0; k < 12; k++) {
        const x = Math.round((45 + k * 21) * w / 320), y = Math.round((45 + (k % 3) * 35) * h / 180);
        putBlob(p, w, h, x, y, -100, 128);
        putBlob(a, w, h, x, y, k < 1 ? 10 : -100, 128);
        if (k < 1) ts.push({ x, y });
      }
      assert(measureStarlingDarkExtremaPrecision(a, p, ts, w, h) === 0.0, 'weakPartialInv1_amp10 rejected');
    }

    // diagonalHalfPan: static diagonal (0.5, 0.5) pan at border and interior must score 0.0
    {
      const w = 320, h = 180, p = mkFrame(w, h, 220), a = mkFrame(w, h, 220), edge = [], interior = [];
      for (let y = 25; y < h - 20; y += 23) {
        for (let x = 3; x < w + 12; x += 24) {
          putBlob(p, w, h, x - 0.5, y - 0.5);
          putBlob(a, w, h, x, y);
          if (x === 3) edge.push({ x, y });
          if (x === 27) interior.push({ x, y });
        }
      }
      assert(measureStarlingDarkExtremaPrecision(a, p, edge, w, h) === 0.0, 'diagonalHalfPan edge rejected');
      assert(measureStarlingDarkExtremaPrecision(a, p, interior, w, h) === 0.0, 'diagonalHalfPan interior rejected');
    }

    // pointGrid fractional pans & genuine small/sharp relative motion under diagonal half-pan
    for (const [dx, dy] of [[0.5, 0.5], [0.3, 0.7], [-0.4, 0.6]]) {
      const w = 320, h = 180, p = mkFrame(w, h, 220), a = mkFrame(w, h, 220), ts = [];
      for (let y = 25; y < h - 20; y += 23) {
        for (let x = 27; x < w - 20; x += 24) {
          putPoint(p, w, h, x - dx, y - dy);
          putPoint(a, w, h, x, y);
          if (x === 27) ts.push({ x, y });
        }
      }
      assert(measureStarlingDarkExtremaPrecision(a, p, ts, w, h) === 0.0, `pointGrid pan (${dx},${dy}) rejected`);
    }
    // pointGrid under gain/offset shift + bilinear pan on textured sky (-0.4, 0.6) rejected
    {
      const w = 320, h = 180, p = new Uint8ClampedArray(w * h * 4), a = new Uint8ClampedArray(w * h * 4), ts = [];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const gx = Math.round(x / 18) * 18, gy = Math.round(y / 18) * 18;
          const v = Math.max(0, Math.min(255, Math.round(175 + 20 * Math.sin(x * 0.025) + 15 * Math.cos(y * 0.03) + ((x === gx && y === gy) ? -65 : 0))));
          const i = (y * w + x) * 4;
          p[i] = p[i + 1] = p[i + 2] = v;
          p[i + 3] = 255;
        }
      }
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const sx = x + 0.4, sy = y - 0.6;
          const x0 = Math.max(0, Math.min(w - 1, Math.floor(sx))), y0 = Math.max(0, Math.min(h - 1, Math.floor(sy)));
          const x1 = Math.max(0, Math.min(w - 1, x0 + 1)), y1 = Math.max(0, Math.min(h - 1, y0 + 1));
          const fx = sx - Math.floor(sx), fy = sy - Math.floor(sy);
          const sv = (p[(y0 * w + x0) * 4] * (1 - fx) + p[(y0 * w + x1) * 4] * fx) * (1 - fy) + (p[(y1 * w + x0) * 4] * (1 - fx) + p[(y1 * w + x1) * 4] * fx) * fy;
          const v = Math.max(0, Math.min(255, Math.round(0.8 * sv + 25)));
          const i = (y * w + x) * 4;
          a[i] = a[i + 1] = a[i + 2] = v;
          a[i + 3] = 255;
        }
      }
      for (let y = 36; y <= h - 36; y += 36) for (let x = 36; x <= w - 36; x += 36) ts.push({ x: x - 0.4, y: y + 0.6 });
      assert(measureStarlingDarkExtremaPrecision(a, p, ts, w, h) === 0.0, 'pointGrid bilinear gain/offset pan (-0.4,0.6) rejected');
    }
    for (const [shape, sig, d] of [['point', 0, 0.5], ['gauss', 0.5, 0.2], ['gauss', 0.7, 0.05]]) {
      const w = 320, h = 180, p = mkFrame(w, h, 220), a = mkFrame(w, h, 220), cx = w / 2, cy = h / 2;
      for (let k = 0; k < 12; k++) {
        const x = Math.round((45 + k * 21) * w / 320), y = Math.round((45 + (k % 3) * 35) * h / 180);
        putBlob(p, w, h, x, y);
        putBlob(a, w, h, x + 0.5, y + 0.5);
      }
      if (shape === 'point') {
        putPoint(p, w, h, cx, cy);
        putPoint(a, w, h, cx + 0.5 + d, cy + 0.5);
      } else {
        putBlob(p, w, h, cx, cy, 160, 220, sig);
        putBlob(a, w, h, cx + 0.5 + d, cy + 0.5, 160, 220, sig);
      }
      assert(
        measureStarlingDarkExtremaPrecision(a, p, [{ x: cx + 0.5 + d, y: cy + 0.5 }], w, h) === 1.0,
        `${shape}_sig${sig}_relative${d} scores 1.0`
      );
    }

    // Sol-3-v10 + v11 regressions: stationary gain+pan, static-tex cloud, sharp motion, unrelated-departure inv,
    // arrival97, fractional-band, beyond-cap inv, partial3_pan (period-63), compensated_inv, dimming, periodic grids
    {
      const mkRgba = (w, h, fn) => {
        const out = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const lum = Math.max(0, Math.min(255, Math.round(fn(x, y))));
            const idx = (y * w + x) * 4;
            out[idx] = out[idx + 1] = out[idx + 2] = lum;
            out[idx + 3] = 255;
          }
        }
        return out;
      };
      const mkRepro = (w, h, pts, dx = 0, dy = 0, g = 1, o = 0, tex = 0, bg = 220) =>
        mkRgba(w, h, (x, y) => {
          const u = x - dx, v = y - dy;
          let lum = tex === 0 ? bg
            : tex === 1 ? 180 + 0.05 * (u - w / 2) + 0.03 * (v - h / 2) + 5 * Math.sin(u * 0.31 + v * 0.17) + 4 * Math.cos(u * 0.13 - v * 0.39)
            : 180 + 0.12 * (u - w / 2) + 0.08 * (v - h / 2) + 12 * Math.sin(u * 0.8 + v * 0.31) + 8 * Math.cos(u * 0.39 - v * 0.7);
          for (const [px0, py0, amp = 160, s = 0] of pts) {
            const cx0 = px0 + dx, cy0 = py0 + dy, dd = Math.hypot(x - cx0, y - cy0);
            if (s > 0 && dd <= 4 * s + 1) lum -= amp * Math.exp(-(dd * dd) / (2 * s * s));
            else if (s === 0 && Math.abs(x - cx0) < 1 && Math.abs(y - cy0) < 1) {
              lum -= amp * (1 - Math.abs(x - cx0)) * (1 - Math.abs(y - cy0));
            }
          }
          return g * lum + o;
        });
      const mkGridPts = (w, h, n, randFlag = false, sig = 0, phase = -0.5) => {
        const nx = Math.ceil(Math.sqrt(n * w / h)), ny = Math.ceil(n / nx), p = [];
        let s = 40231;
        const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
        for (let j = 0; j < ny; j++) {
          for (let i = 0; i < nx && p.length < n; i++) {
            let x = 16 + (i + 0.5) * (w - 32) / nx, y = 16 + (j + 0.5) * (h - 32) / ny;
            if (randFlag) { x += (rnd() - 0.5) * (w - 32) / nx * 0.55; y += (rnd() - 0.5) * (h - 32) / ny * 0.55; }
            p.push([Math.floor(x) + phase, Math.floor(y) + phase, randFlag ? 120 + 60 * rnd() : 180, sig]);
          }
        }
        return p;
      };
      {
        const pts = mkGridPts(320, 180, 120);
        const ts = pts.filter((_, i) => i % 10 === 0).map(([x, y]) => ({ x: x + 1.5, y: y + 1.5 }));
        assert(measureStarlingDarkExtremaPrecision(mkRepro(320, 180, pts, 1.5, 1.5, 1.25, -30), mkRepro(320, 180, pts), ts, 320, 180) === 0.0, 'v10 stationary-point-gain');
      }
      {
        const pts = mkGridPts(640, 360, 400, true), dx = 8.5, dy = -3.5;
        const ts = pts.filter((_, i) => i % 33 === 0).map(([x, y]) => ({ x: x + dx, y: y + dy }));
        assert(measureStarlingDarkExtremaPrecision(mkRepro(640, 360, pts.map(([x, y, a, s]) => [x + dx, y + dy, a, s]), 0, 0, 1, 0, 2), mkRepro(640, 360, pts, 0, 0, 1, 0, 2), ts, 640, 360) === 1.0, 'v10 moving-cloud-static-texture');
      }
      {
        const anchors = mkGridPts(320, 180, 24, true, 1.1, 0).filter(([x, y]) => Math.hypot(x - 160, y - 90) > 12);
        assert(measureStarlingDarkExtremaPrecision(mkRepro(320, 180, [...anchors, [160.05, 90, 160, 0]], 0.5, 0.5, 1, 0, 1), mkRepro(320, 180, [...anchors, [160, 90, 160, 0]], 0, 0, 1, 0, 1), [{ x: 160.55, y: 90.5 }], 320, 180) === 1.0, 'v10 sharp-motion-textured');
      }
      {
        const p = [], a = [], ts = [];
        for (let k = 0; k < 12; k++) { const x = 80 + 44 * k, y = 70 + 85 * (k % 3); p.push([x, y, -20, 1.1]); a.push([x, y, k < 3 ? 20 : -20, 1.1]); if (k < 3) ts.push({ x, y }); }
        p.push([80, 82, 40, 1.1]);
        assert(measureStarlingDarkExtremaPrecision(mkRepro(640, 360, a, 0, 0, 1, 0, 0, 128), mkRepro(640, 360, p, 0, 0, 1, 0, 0, 128), ts, 640, 360) === 0.0, 'v10 inversion-unrelated-departure');
      }
      assert(measureStarlingDarkExtremaPrecision(mkRepro(640, 360, [[416, 234, -30, 1.1], [416, 234, 140, 1.1]], 0, 0, 1, 0, 0, 180), mkRepro(640, 360, [[416, 234, -30, 1.1], [319, 234, 140, 1.1]], 0, 0, 1, 0, 0, 180), [{ x: 416, y: 234 }], 640, 360) === 1.0, 'v10 arrival97');
      for (const [nm, dx, dy, sig] of [['fractional-band', -15.4, 7.6, 0.5], ['beyond-horizontal', 40, 0, 1.1], ['beyond-vertical', 0, 30, 1.1]]) {
        const p = [], a = [], ts = [];
        for (let k = 0; k < 12; k++) {
          const x = 60 + 44 * k, y = nm === 'fractional-band' ? 20 : nm === 'beyond-horizontal' ? 8 + 10 * (k % 3) : 40 + 12 * (k % 3);
          p.push([x - dx, y - dy, -100, sig]); a.push([x, y, k < 3 ? 100 : -100, sig]); if (k < 3) ts.push({ x, y });
        }
        assert(measureStarlingDarkExtremaPrecision(mkRepro(640, 360, a, 0, 0, 1, 0, 0, 128), mkRepro(640, 360, p, 0, 0, 1, 0, 0, 128), ts, 640, 360) === 0.0, `v10 ${nm}`);
      }
      {
        const bg63 = (x, y) => 170 + 18 * Math.sin((x * 2 * Math.PI) / 63) + 12 * Math.cos(y * 0.09);
        const pts = [[60, 27], [81, 62], [102, 97]];
        const dx = -35, dy = 18;
        const pF = mkRgba(320, 180, (x, y) => {
          let v = bg63(x, y);
          for (const [bx, by] of pts) { const dd = Math.hypot(x - bx, y - by); if (dd <= 5) v += 20 * Math.exp(-(dd * dd) / 2.42); }
          return v;
        });
        const cF = mkRgba(320, 180, (x, y) => {
          let v = bg63(x - dx, y - dy);
          for (const [bx, by] of pts) { const dd = Math.hypot(x - (bx + dx), y - (by + dy)); if (dd <= 5) v -= 20 * Math.exp(-(dd * dd) / 2.42); }
          return v;
        });
        assert(measureStarlingDarkExtremaPrecision(cF, pF, pts.map(([bx, by]) => ({ x: bx + dx, y: by + dy })), 320, 180) === 0.0, 'v11 partial3_pan(-35,18) rejected');
      }
      {
        const pComp = mkRgba(320, 180, (x, y) => {
          let v = 175 + 16 * Math.sin(x * 0.07) * Math.cos(y * 0.09);
          for (let k = 0; k < 6; k++) { const dd = Math.hypot(x - (50 + k * 42), y - 80); if (dd <= 5) v += 35 * Math.exp(-(dd * dd) / 2.42); }
          const dd = Math.hypot(x - 160, y - 125); if (dd <= 5) v -= 80 * Math.exp(-(dd * dd) / 2.42);
          return v;
        });
        const cComp = mkRgba(320, 180, (x, y) => {
          let v = 175 + 16 * Math.sin(x * 0.07) * Math.cos(y * 0.09);
          for (let k = 0; k < 6; k++) { const dd = Math.hypot(x - (50 + k * 42), y - 80); if (dd <= 5) v -= 35 * Math.exp(-(dd * dd) / 2.42); }
          const dd = Math.hypot(x - 172, y - 125); if (dd <= 5) v -= 80 * Math.exp(-(dd * dd) / 2.42);
          return v;
        });
        assert(measureStarlingDarkExtremaPrecision(cComp, pComp, [{ x: 92, y: 80 }], 320, 180) === 0.0, 'v11 compensated_inv rejects stationary inverted spot');
        assert(measureStarlingDarkExtremaPrecision(cComp, pComp, [{ x: 172, y: 125 }], 320, 180) === 1.0, 'v11 compensated_inv catches true departure');
        const pDim = mkRgba(320, 180, (x, y) => 175 - (Math.hypot(x - 160, y - 90) <= 5 ? 35 * Math.exp(-((x - 160) ** 2 + (y - 90) ** 2) / 2.42) : 0));
        const cDim = mkRgba(320, 180, (x, y) => 175 - (Math.hypot(x - 160, y - 90) <= 5 ? 70 * Math.exp(-((x - 160) ** 2 + (y - 90) ** 2) / 2.42) : 0));
        assert(measureStarlingDarkExtremaPrecision(cDim, pDim, [{ x: 160, y: 90 }], 320, 180) === 0.0, 'v11 ambient dimming rejected');
      }
      for (const [spP, dx, dy] of [[20, -18, 0], [28, 32, -16]]) {
        const mkSp = (shift) => mkRgba(320, 180, (x, y) => {
          const sx = x - (shift ? dx : 0), sy = y - (shift ? dy : 0);
          let v = 200;
          const gx = Math.round(sx / spP) * spP, gy = Math.round(sy / spP) * spP;
          if (gx >= 0 && gx < 320 && gy >= 0 && gy < 180 && x === Math.round(gx + (shift ? dx : 0)) && y === Math.round(gy + (shift ? dy : 0))) v -= 80;
          return v;
        });
        const ts = [];
        for (let y = 40; y <= 140; y += spP) for (let x = 60; x <= 260 && ts.length < 6; x += spP) ts.push({ x: x + dx, y: y + dy });
        assert(measureStarlingDarkExtremaPrecision(mkSp(true), mkSp(false), ts, 320, 180) === 0.0, `v11 periodic_sp${spP}(${dx},${dy}) rejected`);
      }
    }
  }

  console.log('  [PASS] All 15 synchronous evaluator/HOTA/Morton/Starling/Audit/Sentinel unit fixtures verified.');
}

async function runAntiSpoofSelfTests() {
  const assert = (cond, msg) => {
    if (!cond) throw new Error(`Anti-spoof self-test failed: ${msg}`);
  };
  const makeMockWasm = (
    stepFn,
    { failInspector = false, oobMutates = false, oobQuarterMutate = false, noHeroPromote = false, matchCoasted = false } = {}
  ) => {
    const mem = new WebAssembly.Memory({ initial: EXPECTED_WASM_PAGES, maximum: EXPECTED_WASM_PAGES });
    let resMode = 1;
    let swarmMode = 2;
    let frameIdx = 0;
    let caseIdx = -1;
    let focusTarget = 0;
    let inspHeroTid = 0;
    let inspMicroPin = 0;
    const BASE = 32768;
    const TELEM_PTR = BASE;
    const LAP_PTR = BASE + 16384;
    const FIEDLER_PTR = BASE + 32768;
    const MICRO_PTR = BASE + 40960;
    const FIELD_PTR = MICRO_PTR + MAX_MICRO * MICRO_STRIDE * 4;
    const BODIES_PTR = FIELD_PTR + FIELD_CELLS * FIELD_STRIDE * 4;
    const SWARM_PTR = BODIES_PTR + MAX_RIGID_BODIES * BODY_STRIDE * 4;
    const MODAL_PTR = SWARM_PTR + SWARM_TELEM_LEN * 4;
    const RING_BASE = MODAL_PTR + FIELD_CELLS * 4;
    const RING_STEP = HD_W * HD_H * 4;

    const telem = new Float32Array(mem.buffer, TELEM_PTR, TELEMETRY_LEN);
    const micro = new Float32Array(mem.buffer, MICRO_PTR, MAX_MICRO * MICRO_STRIDE);
    const field = new Float32Array(mem.buffer, FIELD_PTR, FIELD_CELLS * FIELD_STRIDE);
    const bodies = new Float32Array(mem.buffer, BODIES_PTR, MAX_RIGID_BODIES * BODY_STRIDE);
    const st = new Float32Array(mem.buffer, SWARM_PTR, SWARM_TELEM_LEN);
    const modal = new Float32Array(mem.buffer, MODAL_PTR, FIELD_CELLS);

    const clearAll = () => {
      telem.fill(0);
      telem[20] = 18_500_000;
      micro.fill(0);
      field.fill(0);
      for (let c = 0; c < FIELD_CELLS; c++) field[c * FIELD_STRIDE + 4] = -999.0;
      bodies.fill(0);
      st.fill(0);
      modal.fill(-9.0);
    };

    const exp = {
      memory: mem,
      chronos_init: () => {
        caseIdx++;
        frameIdx = 0;
        focusTarget = 0;
        inspHeroTid = 0;
        inspMicroPin = 0;
        clearAll();
      },
      chronos_set_resolution: (r) => {
        resMode = r;
        inspHeroTid = 0;
        inspMicroPin = 0;
      },
      chronos_get_frame_w: () => (resMode === 1 ? HD_W : 320),
      chronos_get_frame_h: () => (resMode === 1 ? HD_H : 180),
      chronos_set_params: () => {},
      chronos_set_swarm_mode: (m) => {
        swarmMode = m;
        if (m === 0) { inspHeroTid = 0; inspMicroPin = 0; }
      },
      chronos_get_swarm_mode: () => swarmMode,
      chronos_get_focus_target: () => focusTarget,
      chronos_set_focus_target: (tid) => {
        const u = tid >>> 0;
        if (u !== 0 && u === inspMicroPin && inspHeroTid !== 0) {
          focusTarget = inspHeroTid;
        } else {
          focusTarget = u;
          if (!(u !== 0 && u === inspHeroTid)) {
            inspMicroPin = 0;
            inspHeroTid = 0;
          }
        }
      },
      chronos_get_telemetry_ptr: () => TELEM_PTR,
      chronos_get_laplacian_ptr: () => LAP_PTR,
      chronos_get_fiedler_ptr: () => FIEDLER_PTR,
      chronos_get_micro_tracks_ptr: () => MICRO_PTR,
      chronos_get_continuum_field_ptr: () => FIELD_PTR,
      chronos_get_rigid_bodies_ptr: () => BODIES_PTR,
      chronos_get_swarm_telemetry_ptr: () => SWARM_PTR,
      chronos_get_modal_field_ptr: () => MODAL_PTR,
      chronos_get_ring_slot_ptr: (s) => RING_BASE + s * RING_STEP,
      chronos_set_ingest_pts: () => {},
      chronos_pre_tick: () => {},
      chronos_solve_fiedler_lanczos: () => 0.0,
      chronos_solve_lanczos: () => {},
      chronos_laplacian_matvec: () => {},
      chronos_fiedler_step: () => {},
      chronos_set_fiedler_spectrum: () => {},
      chronos_post_tick: () => {
        const f = frameIdx++;
        const w = resMode === 1 ? HD_W : 320;
        const h = resMode === 1 ? HD_H : 180;
        const rgba = new Uint8Array(mem.buffer, RING_BASE + (f % 12) * RING_STEP, w * h * 4);
        clearAll();
        stepFn(f, { telem, micro, field, bodies, st, modal, rgba, w, h, caseIdx });
      },
      chronos_inspect_field_point: (qx, qy) => {
        if (failInspector) { st[11] = 999; st[12] = 1; return 999; }
        const w = resMode === 1 ? HD_W : 320;
        const h = resMode === 1 ? HD_H : 180;
        if (!Number.isFinite(qx) || !Number.isFinite(qy) || qx < 0 || qy < 0 || qx >= w || qy >= h) {
          if (oobMutates) { st[11] = 0; st[12] = 0; focusTarget = 0; }
          if (oobQuarterMutate) { st[11] += 0.25; }
          return 0;
        }
        const maxR = w >= 640 ? 24.0 : 12.0;
        let bestTid = 0;
        let bestBid = 0;
        let bestD2 = maxR * maxR;
        for (let i = 0; i < MAX_MICRO; i++) {
          const b = i * MICRO_STRIDE;
          const s = micro[b + 5];
          if (!(s === 1 || (matchCoasted && s === 2))) continue;
          const d2 = (micro[b + 1] - qx) ** 2 + (micro[b + 2] - qy) ** 2;
          if (d2 < bestD2) { bestD2 = d2; bestTid = micro[b]; bestBid = micro[b + 6]; }
        }
        st[11] = bestTid;
        st[12] = bestBid;
        if (bestTid > 0 && !noHeroPromote) {
          inspMicroPin = bestTid;
          inspHeroTid = 17;
          focusTarget = 17;
        } else {
          inspMicroPin = 0;
          inspHeroTid = 0;
        }
        return bestTid;
      },
    };
    return {
      exp,
      mem,
      missing: [],
      objdumpReport: {
        available: true,
        analyzedFuncCount: 40,
        missingExportFuncs: 0,
        disassemblyComplete: true,
        allocatorCallCount: 0,
        allocatorReachableCount: 0,
        maxDataEnd: 17408,
      },
    };
  };

  // Locked #115/#116 unit inspector check (positive, coasted-hit reject, OOB-mutate reject)
  {
    const obsList = [{ tid: 42, x: 100, y: 100, bodyId: 2 }];
    const goodInsp = makeMockWasm((_f, { micro, st }) => {
      micro[0] = 42; micro[1] = 100; micro[2] = 100; micro[5] = 1; micro[6] = 2;
      micro[MICRO_STRIDE + 0] = 43; micro[MICRO_STRIDE + 1] = 100.1; micro[MICRO_STRIDE + 2] = 100.1; micro[MICRO_STRIDE + 5] = 2;
      st[1] = 1; st[3] = 2;
    });
    goodInsp.exp.chronos_init();
    goodInsp.exp.chronos_post_tick();
    const gSt = new Float32Array(goodInsp.mem.buffer, goodInsp.exp.chronos_get_swarm_telemetry_ptr(), SWARM_TELEM_LEN);
    assert(verifyPopulatedInspector(goodInsp.exp, HD_W, HD_H, obsList, gSt).length === 0, 'clean #115/#116 inspector ok');

    const badOob = makeMockWasm((_f, { micro, st }) => {
      micro[0] = 42; micro[1] = 100; micro[2] = 100; micro[5] = 1; micro[6] = 2; st[1] = 1; st[3] = 1;
    }, { oobMutates: true });
    badOob.exp.chronos_init(); badOob.exp.chronos_post_tick();
    const oSt = new Float32Array(badOob.mem.buffer, badOob.exp.chronos_get_swarm_telemetry_ptr(), SWARM_TELEM_LEN);
    assert(verifyPopulatedInspector(badOob.exp, HD_W, HD_H, obsList, oSt).some((m) => m.includes('inspectorOobMutated')), 'OOB mutation caught');

    const badCoast = makeMockWasm((_f, { micro, st }) => {
      micro[0] = 42; micro[1] = 100; micro[2] = 100; micro[5] = 1; micro[6] = 2;
      micro[MICRO_STRIDE] = 43; micro[MICRO_STRIDE + 1] = 100.25; micro[MICRO_STRIDE + 2] = 99.75; micro[MICRO_STRIDE + 5] = 2;
      st[1] = 1; st[3] = 2;
    }, { matchCoasted: true });
    badCoast.exp.chronos_init(); badCoast.exp.chronos_post_tick();
    const cSt = new Float32Array(badCoast.mem.buffer, badCoast.exp.chronos_get_swarm_telemetry_ptr(), SWARM_TELEM_LEN);
    assert(verifyPopulatedInspector(badCoast.exp, HD_W, HD_H, obsList, cSt).some((m) => m.includes('inspectorHitMismatch')), 'coasted match caught');

    const badNoHero = makeMockWasm((_f, { micro, st }) => {
      micro[0] = 42; micro[1] = 100; micro[2] = 100; micro[5] = 1; micro[6] = 2; st[1] = 1; st[3] = 1;
    }, { noHeroPromote: true });
    badNoHero.exp.chronos_init(); badNoHero.exp.chronos_post_tick();
    const nhSt = new Float32Array(badNoHero.mem.buffer, badNoHero.exp.chronos_get_swarm_telemetry_ptr(), SWARM_TELEM_LEN);
    assert(verifyPopulatedInspector(badNoHero.exp, HD_W, HD_H, obsList, nhSt).some((m) => m.includes('inspectorFocusNotPromotedAfterHit')), 'missing Hero promotion caught');

    const badQuarter = makeMockWasm((_f, { micro, st }) => {
      micro[0] = 42; micro[1] = 100; micro[2] = 100; micro[5] = 1; micro[6] = 2; st[1] = 1; st[3] = 1;
    }, { oobQuarterMutate: true });
    badQuarter.exp.chronos_init(); badQuarter.exp.chronos_post_tick();
    const qSt = new Float32Array(badQuarter.mem.buffer, badQuarter.exp.chronos_get_swarm_telemetry_ptr(), SWARM_TELEM_LEN);
    assert(verifyPopulatedInspector(badQuarter.exp, HD_W, HD_H, obsList, qSt).some((m) => m.includes('inspectorOobMutated')), 'sub-0.5 OOB telemetry mutation caught');
  }

  const quiet = async (fn) => {
    const origLog = console.log;
    console.log = () => {};
    try { return await fn(); } finally { console.log = origLog; }
  };

  // Blind pixel peak extractor & fast bucketed NN linker used by positive Sweep / Analytic mocks
  const extractBlobs = (rgba, w, h) => {
    const pts = [];
    const pxLuma = (idx) => {
      const o = idx << 2;
      return 0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2];
    };
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        const idx = row + x;
        const j = idx << 2;
        // Skip dark background (#080a10) in one byte check for <0.3ms HD sweeps
        if (rgba[j] < 55 && rgba[j + 1] < 55 && rgba[j + 2] < 55) continue;
        const c = pxLuma(idx);
        if (c < 60.0) continue;
        let isMax = true;
        for (let dy = -1; dy <= 1 && isMax; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const n = pxLuma((y + dy) * w + (x + dx));
            if (n > c || (n === c && (dy < 0 || (dy === 0 && dx < 0)))) { isMax = false; break; }
          }
        }
        if (!isMax) continue;
        let sw = 0, swx = 0, swy = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const wt = Math.max(0.0, pxLuma((y + dy) * w + (x + dx)) - 15.0);
            sw += wt; swx += wt * (x + dx); swy += wt * (y + dy);
          }
        }
        pts.push({ x: swx / sw, y: swy / sw });
      }
    }
    return pts;
  };
  const makePixelTracker = () => {
    let prev = [];
    let nextTid = 1;
    let lastVel = { dx: 0, dy: 0 };
    return (f, rgba, w, h, { maxGate = 14.0 } = {}) => {
      if (f === 0) { prev = []; nextTid = 1; lastVel = { dx: 0, dy: 0 }; }
      const dets = extractBlobs(rgba, w, h);
      const curr = [];
      if (dets.length > 0) {
        const used = new Uint8Array(prev.length);
        const matches = [];
        const predShift = prev.length > 0 ? lastVel : { dx: 0, dy: 0 };
        const cell = Math.max(8, Math.ceil(maxGate));
        const gw = Math.ceil(w / cell);
        const gh = Math.ceil(h / cell);
        const grid = Array.from({ length: gw * gh }, () => []);
        for (let p = 0; p < prev.length; p++) {
          const px = prev[p].x + (prev[p].missed ? predShift.dx : 0);
          const py = prev[p].y + (prev[p].missed ? predShift.dy : 0);
          const gx = Math.max(0, Math.min(gw - 1, Math.floor(px / cell)));
          const gy = Math.max(0, Math.min(gh - 1, Math.floor(py / cell)));
          grid[gy * gw + gx].push([p, px, py]);
        }
        for (let d = 0; d < dets.length; d++) {
          const dx0 = dets[d].x;
          const dy0 = dets[d].y;
          const gx = Math.max(0, Math.min(gw - 1, Math.floor(dx0 / cell)));
          const gy = Math.max(0, Math.min(gh - 1, Math.floor(dy0 / cell)));
          let bestP = -1, bestD2 = maxGate * maxGate;
          for (let ny = Math.max(0, gy - 1); ny <= Math.min(gh - 1, gy + 1); ny++) {
            for (let nx = Math.max(0, gx - 1); nx <= Math.min(gw - 1, gx + 1); nx++) {
              const bucket = grid[ny * gw + nx];
              for (let k = 0; k < bucket.length; k++) {
                const [p, px, py] = bucket[k];
                if (used[p]) continue;
                const d2 = (dx0 - px) ** 2 + (dy0 - py) ** 2;
                if (d2 < bestD2) { bestD2 = d2; bestP = p; }
              }
            }
          }
          if (bestP >= 0) {
            used[bestP] = 1;
            const vx = dx0 - prev[bestP].x;
            const vy = dy0 - prev[bestP].y;
            matches.push({ vx, vy });
            curr.push({ tid: prev[bestP].tid, x: dx0, y: dy0, vx: vx * 30, vy: vy * 30, missed: false });
          } else {
            curr.push({ tid: nextTid++, x: dx0, y: dy0, vx: 0, vy: 0, missed: false });
          }
        }
        if (matches.length > 10) {
          lastVel = {
            dx: matches.reduce((s, m) => s + m.vx, 0) / matches.length,
            dy: matches.reduce((s, m) => s + m.vy, 0) / matches.length,
          };
        }
      } else {
        for (const p of prev) {
          curr.push({ tid: p.tid, x: p.x + lastVel.dx, y: p.y + lastVel.dy, vx: lastVel.dx * 30, vy: lastVel.dy * 30, missed: true });
        }
      }
      prev = curr;
      return curr;
    };
  };

  // 1. Q1 positive + targeted anti-spoofs (3 cells, div=1000, st[39]=0, Hero=0, premature f<8 FTLE)
  const q1Scene = createQ1VortexScene();
  const buildQ1Step = ({
    overrideDivVal = null,
    overrideSt20 = null,
    overrideMature39 = null,
    prematureFtle39 = false,
    inactiveWarmup0 = false,
    newbornCountedMature = false,
    zeroHero = false,
    onlyCells = null,
  } = {}) => (f, { telem, micro, field, st, modal }) => {
    if (inactiveWarmup0 && f === 0) {
      telem[3] = 0;
      for (let c = 0; c < FIELD_CELLS; c++) field[c * FIELD_STRIDE + 4] = 0.0;
      modal.fill(0.0);
      return;
    }
    const pts = q1Scene.gtFrames[f];
    for (let i = 0; i < pts.length && i < MAX_MICRO; i++) {
      const mb = i * MICRO_STRIDE;
      const isNewborn = newbornCountedMature && f === 40 && i === 0;
      micro[mb] = isNewborn ? 100001 : pts[i].gtId;
      micro[mb + 1] = pts[i].x; micro[mb + 2] = pts[i].y;
      micro[mb + 5] = 1; micro[mb + 7] = isNewborn ? -999.0 : (f >= 12 ? 0.35 : -999.0);
    }
    telem[3] = zeroHero ? 0 : 2;
    st[1] = 1; st[2] = pts.length; st[3] = pts.length; st[4] = pts.length; st[5] = pts.length; st[6] = pts.length;
    st[39] =
      overrideMature39 !== null
        ? overrideMature39
        : newbornCountedMature && f === 40
        ? pts.length
        : prematureFtle39 && f === 5
        ? 12
        : f >= 12
        ? 950
        : 0;
    let cells = 0, ftleCells = 0, vSq = 0, dSq = 0, maxF = 0;
    const cellW = HD_W / FIELD_COLS, cellH = HD_H / FIELD_ROWS;
    for (let r = 1; r < FIELD_ROWS - 1; r++) {
      for (let c = 1; c < FIELD_COLS - 1; c++) {
        if (onlyCells !== null && cells >= onlyCells) break;
        const fb = (r * FIELD_COLS + c) * FIELD_STRIDE;
        const cx = (c + 0.5) * cellW, cy = (r + 0.5) * cellH;
        const [u, v] = q1Scene.velAt(cx, cy);
        const w = q1Scene.analyticVorticityAt(cx, cy);
        const d = overrideDivVal !== null ? overrideDivVal : 0.0;
        field[fb] = u; field[fb + 1] = v; field[fb + 2] = w; field[fb + 3] = d;
        field[fb + 5] = 1.0;
        cells++; vSq += w * w; dSq += d * d;
        if (f >= 12) {
          const ft = 0.20 + 0.08 * Math.abs(w);
          field[fb + 4] = ft; ftleCells++; maxF = Math.max(maxF, ft);
        }
      }
      if (onlyCells !== null && cells >= onlyCells) break;
    }
    st[16] = (onlyCells ? 3 : (f >= 12 ? 3 : 1));
    st[17] = onlyCells ? 400 : cells;
    st[19] = cells > 0 ? Math.sqrt(vSq / cells) : 0;
    st[20] = overrideSt20 !== null ? overrideSt20 : (cells > 0 ? Math.sqrt(dSq / cells) : 0);
    st[21] = onlyCells ? 0.5 : maxF;
    st[22] = onlyCells ? 250 : ftleCells;
  };
  const rQ1Good = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step())));
  assert(rQ1Good.passed, `Q1 positive control failed: ${rQ1Good.failures.join(',')}`);
  const rQ1Spoof = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step({ onlyCells: 3 }))));
  assert(!rQ1Spoof.passed, 'Q1 spoof with 3 cells passed!');
  const rQ1DivSpoof = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step({ overrideDivVal: 1000.0, overrideSt20: 0.0 }))));
  assert(!rQ1DivSpoof.passed && rQ1DivSpoof.failures.some((x) => x.includes('Div') || x.includes('div')), `Q1 div=1000/st[20]=0 failed wrong or passed: ${rQ1DivSpoof.failures}`);
  const rQ1MatSpoof = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step({ overrideMature39: 0 }))));
  assert(!rQ1MatSpoof.passed && rQ1MatSpoof.failures.some((x) => x.includes('Mature39') || x.includes('ftleMature')), `Q1 st[39]=0 spoof not caught: ${rQ1MatSpoof.failures}`);
  const rQ1Hero0 = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step({ zeroHero: true }))));
  assert(!rQ1Hero0.passed && rQ1Hero0.failures.some((x) => x.includes('heroActive')), `Q1 zero Hero spoof passed: ${rQ1Hero0.failures}`);
  const rQ1Pre = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step({ prematureFtle39: true }))));
  assert(!rQ1Pre.passed && rQ1Pre.failures.some((x) => x.includes('ftlePrematureErrors')), `Q1 premature f=5 st[39] spoof passed: ${rQ1Pre.failures}`);
  const rQ1Inact0 = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step({ inactiveWarmup0: true }))));
  assert(rQ1Inact0.passed, `Q1 inactive warmup frame 0 failed: ${rQ1Inact0.failures.join(',')}`);
  const rQ1Newborn = await quiet(() => runQ1Vortex(makeMockWasm(buildQ1Step({ newbornCountedMature: true }))));
  assert(!rQ1Newborn.passed && rQ1Newborn.failures.some((x) => x.includes('ftleMature') || x.includes('rangeViolations')), `Q1 newborn counted mature in st[39] spoof passed: ${rQ1Newborn.failures}`);

  // 2. Q2 positive + rotating 2-of-3 + 40%-broken Body 2 + Hero=0 spoofs
  const q2Scene = createQ2MurmurationScene();
  const buildQ2Step = ({ rotateSkip = false, breakBody2 = false, zeroHero = false } = {}) => (f, { telem, micro, bodies, st }) => {
    const pts = q2Scene.gtFrames[f];
    const skipBody = rotateSkip ? (f % 3) : -1;
    for (let i = 0; i < pts.length && i < MAX_MICRO; i++) {
      const mb = i * MICRO_STRIDE;
      micro[mb] = pts[i].gtId; micro[mb + 1] = pts[i].x; micro[mb + 2] = pts[i].y;
      micro[mb + 5] = 1; micro[mb + 6] = pts[i].bodyIdx >= 0 ? pts[i].bodyIdx + 1 : 0;
    }
    const gts = q2Scene.bodyGtFrames[f];
    let slot = 0;
    for (let k = 0; k < gts.length; k++) {
      if (k === skipBody) continue;
      if (breakBody2 && k === 2 && (f % 5) >= 2) continue; // Body 2 only 40% coverage
      const bb = (slot++) * BODY_STRIDE;
      bodies[bb] = k + 1; bodies[bb + 1] = 36;
      bodies[bb + 2] = gts[k].cx; bodies[bb + 3] = gts[k].cy;
      bodies[bb + 4] = gts[k].vx; bodies[bb + 5] = gts[k].vy;
      bodies[bb + 6] = gts[k].omega; bodies[bb + 7] = 0.05;
    }
    telem[3] = zeroHero ? 0 : 3;
    st[1] = 1; st[2] = pts.length; st[3] = pts.length; st[4] = pts.length; st[5] = pts.length; st[16] = 4; st[23] = slot;
  };
  const rQ2Good = await quiet(() => runQ2Murmuration(makeMockWasm(buildQ2Step()), q2Scene));
  assert(rQ2Good.passed, `Q2 positive control failed: ${rQ2Good.failures.join(',')}`);
  const rQ2 = await quiet(() => runQ2Murmuration(makeMockWasm(buildQ2Step({ rotateSkip: true })), q2Scene));
  assert(!rQ2.passed, 'Q2 rotating 2-of-3 bodies spoof passed!');
  const rQ2B2 = await quiet(() => runQ2Murmuration(makeMockWasm(buildQ2Step({ breakBody2: true })), q2Scene));
  assert(!rQ2B2.passed && rQ2B2.failures.some((x) => x.includes('body2')), `Q2 40% broken Body 2 failed without body2 error: ${rQ2B2.failures}`);
  const rQ2Hero0 = await quiet(() => runQ2Murmuration(makeMockWasm(buildQ2Step({ zeroHero: true })), q2Scene));
  assert(!rQ2Hero0.passed && rQ2Hero0.failures.some((x) => x.includes('heroActive')), `Q2 zero Hero spoof passed: ${rQ2Hero0.failures}`);

  // 3. Q3 positive + empty-row modal border-leak spoof
  const q3Scene = createQ3ModalBeamScene();
  const buildQ3Step = ({ leakRow0 = false } = {}) => (f, { telem, micro, st, modal }) => {
    const pts = q3Scene.gtFrames[f];
    for (let i = 0; i < pts.length && i < MAX_MICRO; i++) {
      const mb = i * MICRO_STRIDE;
      micro[mb] = pts[i].gtId; micro[mb + 1] = pts[i].x; micro[mb + 2] = pts[i].y; micro[mb + 5] = 1;
    }
    telem[3] = 2;
    st[1] = 1; st[2] = pts.length; st[3] = pts.length; st[4] = pts.length; st[5] = pts.length;
    st[16] = 8; st[27] = 3.0; st[28] = 0.9; st[29] = 600; st[41] = 0.98;
    modal.fill(-9.0);
    const startRow = leakRow0 ? 0 : 7;
    for (let r = startRow; r <= 28; r++) {
      for (let c = 0; c < FIELD_COLS; c++) {
        if (c <= 28) modal[r * FIELD_COLS + c] = 0.0;
        else if (c >= 35) modal[r * FIELD_COLS + c] = Math.PI;
      }
    }
  };
  const rQ3Good = await quiet(() => runQ3ModalBeam(makeMockWasm(buildQ3Step())));
  assert(rQ3Good.passed, `Q3 positive control failed: ${rQ3Good.failures.join(',')}`);
  const rQ3 = await quiet(() => runQ3ModalBeam(makeMockWasm(buildQ3Step({ leakRow0: true }))));
  assert(!rQ3.passed && rQ3.failures.some((x) => x.includes('emptyRowModalCellLeak')), 'Q3 modal border-leak spoof passed!');

  // 4. Analytic suite positive control (all 13 cases) + targeted negative spoofs
  const anaTracker = makePixelTracker();
  const buildAnaStep = ({
    leakF21 = false,
    collinearGradLeak = false,
    staleShear39 = false,
    badShearFtle = false,
    case8SwapF28Leak = false,
    case9F18Leak = false,
    case9PostDrop = false,
    case10F29Leak = false,
  } = {}) => (f, { telem, micro, field, bodies, st, rgba, w, h, caseIdx }) => {
    const trks = case9PostDrop && caseIdx === 9 && f >= 24 ? [] : anaTracker(f, rgba, w, h, { maxGate: 18.0 });
    let obsCnt = 0;
    for (let i = 0; i < trks.length && i < MAX_MICRO; i++) {
      const mb = i * MICRO_STRIDE;
      const t = trks[i];
      micro[mb] = t.tid; micro[mb + 1] = t.x; micro[mb + 2] = t.y;
      micro[mb + 3] = t.vx; micro[mb + 4] = t.vy;
      micro[mb + 5] = t.missed ? 2 : 1;
      micro[mb + 6] = (caseIdx === 1 || caseIdx === 5 || caseIdx === 6 || caseIdx === 7 || caseIdx === 11 || caseIdx === 12) ? 1 : 0;
      micro[mb + 7] = -999.0;
      if (!t.missed) obsCnt++;
    }
    telem[3] = 2;
    st[1] = 1; st[2] = trks.length; st[3] = trks.length; st[4] = obsCnt; st[5] = obsCnt; st[6] = trks.length;
    const cellW = w / FIELD_COLS, cellH = h / FIELD_ROWS;
    const fillGrid = (velFn, vortVal, divVal, ftleVal) => {
      let vc = 0, fc = 0;
      for (let r = 1; r < FIELD_ROWS - 1; r++) {
        for (let c = 1; c < FIELD_COLS - 1; c++) {
          const fb = (r * FIELD_COLS + c) * FIELD_STRIDE;
          const cx = (c + 0.5) * cellW, cy = (r + 0.5) * cellH;
          const [u, v] = velFn(cx, cy);
          field[fb] = u; field[fb + 1] = v; field[fb + 2] = vortVal; field[fb + 3] = divVal;
          field[fb + 5] = 1.0;
          vc++;
          if (ftleVal !== null) { field[fb + 4] = ftleVal; fc++; }
        }
      }
      st[17] = vc; st[22] = fc; st[21] = ftleVal !== null ? ftleVal : 0.0;
    };
    const setBody = (omega, cnt) => {
      bodies[0] = 1; bodies[1] = cnt; bodies[2] = w * 0.5; bodies[3] = h * 0.5;
      bodies[4] = 0; bodies[5] = 0; bodies[6] = omega; bodies[7] = 0.04;
      st[23] = 1; st[24] = omega; st[25] = cnt; st[26] = 0.04;
    };
    if (caseIdx === 0) {
      const ft = f >= 12 ? 0.0 : null;
      fillGrid(() => [48, -24], 0, 0, ft); st[16] = ft !== null ? 3 : 1; st[39] = ft !== null ? Math.min(200, obsCnt) : 0;
    } else if (caseIdx === 1) {
      const ft = f >= 12 ? 0.0 : null;
      fillGrid((x, y) => [-0.85 * (y - 180), 0.85 * (x - 320)], 1.70, 0, ft);
      setBody(0.85, Math.max(20, obsCnt)); st[16] = (ft !== null ? 3 : 1) | 4; st[39] = ft !== null ? Math.min(120, obsCnt) : 0;
    } else if (caseIdx === 2) {
      const ft = f >= 12 ? 0.50 : null;
      fillGrid((x, y) => [0.5 * (x - 320), 0.5 * (y - 180)], 0, 1.0, ft); st[16] = ft !== null ? 3 : 1; st[39] = ft !== null ? Math.min(160, obsCnt) : 0;
    } else if (caseIdx === 3) {
      const ft = f >= 12 ? -0.50 : null;
      fillGrid((x, y) => [-0.5 * (x - 320), -0.5 * (y - 180)], 0, -1.0, ft); st[16] = ft !== null ? 3 : 1; st[39] = ft !== null ? Math.min(160, obsCnt) : 0;
    } else if (caseIdx === 4) {
      const ft = f >= 12 ? (badShearFtle ? 0.459 : 0.589) : null;
      fillGrid((x, y) => [1.2 * (y - 180), 0], -1.20, 0, ft); st[16] = ft !== null ? 3 : 1;
      st[39] = staleShear39 ? 0 : (ft !== null ? Math.min(160, obsCnt) : 0);
    } else if (caseIdx === 5) {
      if (collinearGradLeak) { fillGrid(() => [0, 0], 1.6, 0, null); st[16] = 1; }
      else { setBody(0.80, 28); st[16] = 4; } // #11: proper SE(2) on labeled strip allowed, Grad-V/FTLE=0
    } else if (caseIdx === 6) {
      setBody(0.80, 34); st[16] = 4;
    } else if (caseIdx === 7) {
      if (f < 20) { setBody(0.65, 36); st[16] = 4; }
    } else if (caseIdx === 8) {
      const mature = (f >= 10 && f < 22) || f >= 35 || (case8SwapF28Leak && f === 28);
      fillGrid(() => [28, 0], 0, 0, mature ? 0.0 : null);
      st[16] = mature ? 3 : 1; st[39] = mature ? Math.min(140, obsCnt) : 0;
    } else if (caseIdx === 9) {
      const mature = (f >= 10 && f < 18) || f >= 36 || (case9F18Leak && f === 18);
      fillGrid(() => [30, 0], 0, 0, mature ? 0.0 : null);
      st[16] = mature ? 3 : 1; st[39] = mature ? Math.min(140, obsCnt) : 0;
    } else if (caseIdx === 10) {
      const mature = (f >= 10 && f < 20) || f >= 35 || (leakF21 && f === 21) || (case10F29Leak && f === 29);
      fillGrid(() => [32, 10], 0, 0, mature ? 0.0 : null);
      st[16] = mature ? 3 : 1; st[39] = mature ? Math.min(140, Math.max(80, trks.length)) : 0;
    } else if (caseIdx === 11 || caseIdx === 12) {
      setBody(0.65, 36);
      const sc = w < 640 ? 0.5 : 1.0;
      let vc = 0;
      for (let r = 0; r < FIELD_ROWS; r++) {
        for (let c = 0; c < FIELD_COLS; c++) {
          const cx = (c + 0.5) * cellW, cy = (r + 0.5) * cellH;
          if (Math.hypot(cx - w * 0.5, cy - h * 0.5) <= sc * 60.0) {
            const fb = (r * FIELD_COLS + c) * FIELD_STRIDE;
            field[fb] = -0.65 * (cy - h * 0.5); field[fb + 1] = 0.65 * (cx - w * 0.5);
            field[fb + 5] = 1.0; vc++;
          }
        }
      }
      st[17] = vc; st[16] = 5;
    }
  };
  const rAnaGood = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep())));
  assert(rAnaGood.length === 13 && rAnaGood.every((c) => c.passed), `Analytic positive failed: ${JSON.stringify(rAnaGood.filter((c) => !c.passed))}`);

  const rAnaF21 = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ leakF21: true }))));
  assert(!rAnaF21[10].passed && rAnaF21[10].failures.some((x) => x.includes('secondMissF21')), `Analytic Case 10 f=21 leak slipped: ${rAnaF21[10].failures}`);

  const rAnaColGrad = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ collinearGradLeak: true }))));
  assert(!rAnaColGrad[5].passed && rAnaColGrad[5].failures.some((x) => x.includes('collinearEmittedGradOrFtle')), `Case 6a Grad-V leak slipped: ${rAnaColGrad[5].failures}`);

  const rAnaShear39 = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ staleShear39: true }))));
  assert(!rAnaShear39[4].passed && rAnaShear39[4].failures.some((x) => x.includes('shearMature[39]')), `Case 5 st[39]=0 slipped: ${rAnaShear39[4].failures}`);

  const rAnaShearBad = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ badShearFtle: true }))));
  assert(!rAnaShearBad[4].passed && rAnaShearBad[4].failures.some((x) => x.includes('shearFtle')), `Case 5 wrong FTLE 0.459 slipped: ${rAnaShearBad[4].failures}`);

  const rAnaC8F28 = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ case8SwapF28Leak: true }))));
  assert(!rAnaC8F28[8].passed && rAnaC8F28[8].failures.some((x) => x.includes('swapFtleNotInvalidatedAtF28')), `Case 8 f=28 premature FTLE slipped: ${rAnaC8F28[8].failures}`);

  const rAnaC9F18 = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ case9F18Leak: true }))));
  assert(!rAnaC9F18[9].passed && rAnaC9F18[9].failures.some((x) => x.includes('camCutFtleNotInhibitedAtF18')), `Case 9 f=18 cut FTLE slipped: ${rAnaC9F18[9].failures}`);

  const rAnaC9Drop = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ case9PostDrop: true }))));
  assert(!rAnaC9Drop[9].passed && rAnaC9Drop[9].failures.some((x) => x.includes('[24..45]')), `Case 9 post-cut MOT drop slipped: ${rAnaC9Drop[9].failures}`);

  const rAnaC10F29 = await quiet(() => runAnalyticSuite(makeMockWasm(buildAnaStep({ case10F29Leak: true }))));
  assert(!rAnaC10F29[10].passed && rAnaC10F29[10].failures.some((x) => x.includes('earlyReacqFtleAtF29')), `Case 10 f=29 premature FTLE slipped: ${rAnaC10F29[10].failures}`);

  const anaSpoof = makeMockWasm((_f, { field, st }) => {
    st[16] = 3; st[17] = 1; st[21] = 999; st[22] = 1;
    const fb = (10 * FIELD_COLS + 10) * FIELD_STRIDE;
    field[fb] = 48.0; field[fb + 1] = -24.0; field[fb + 4] = 999.0; field[fb + 5] = 1.0;
  });
  const rAna = await quiet(() => runAnalyticSuite(anaSpoof));
  assert(rAna.length === 13 && rAna.every((c) => !c.passed), `Analytic 1-cell (10,10) spoof slipped: ${rAna.filter((c) => c.passed).map((c) => c.name)}`);
  assert(rAna[0].failures.some((x) => x.includes('gtMotRecall')), `Case 1 should report gtMotRecall failure: ${rAna[0].failures}`);

  // 5. Capacity sweep positive pixel-tracked control (all 4 sizes pass) + frame-0-only spoof fails all 4
  const swpTracker = makePixelTracker();
  const rSwpGood = await quiet(() => runCapacitySweep(makeMockWasm((f, { micro, st, rgba, w, h }) => {
    const trks = swpTracker(f, rgba, w, h, { maxGate: 10.0 });
    for (let i = 0; i < trks.length && i < MAX_MICRO; i++) {
      const mb = i * MICRO_STRIDE;
      micro[mb] = trks[i].tid; micro[mb + 1] = trks[i].x; micro[mb + 2] = trks[i].y; micro[mb + 5] = 1;
    }
    st[1] = 1; st[2] = trks.length; st[3] = trks.length; st[4] = trks.length; st[5] = trks.length;
  })));
  assert(rSwpGood.length === 4 && rSwpGood.every((c) => c.passed), `Sweep positive failed: ${JSON.stringify(rSwpGood.filter((c) => !c.passed))}`);

  const sweepSpoof = makeMockWasm((f, { micro, st }) => {
    if (f === 0) {
      for (let i = 0; i < MAX_MICRO; i++) {
        const mb = i * MICRO_STRIDE;
        micro[mb] = i + 1; micro[mb + 1] = 40 + (i % 48) * 11; micro[mb + 2] = 40 + Math.floor(i / 48) * 9; micro[mb + 5] = 1;
      }
      st[3] = MAX_MICRO; st[4] = MAX_MICRO; st[5] = MAX_MICRO;
    }
  });
  const rSwp = await quiet(() => runCapacitySweep(sweepSpoof));
  assert(rSwp.length === 4 && rSwp.every((c) => !c.passed), 'Capacity sweep frame-0-only spoof passed!');

  console.log('  [PASS] All non-vacuous positive & anti-spoof integration controls verified (Q1/Q2/Q3/Analytic-13/Sweep-4).');
}

function printAblationTable(rows) {
  console.log('\n========================================================================');
  console.log(' Q2 (murmuration_1200) Association & Component Ablation');
  console.log('========================================================================');
  console.log(' Method                          | Recall  |  Prec  | RMSE(px) | IDSW | Frag |  HOTA  | Cap/Rnd | Steps | Reacq | Mean(ms) | p50(ms)');
  console.log(' --------------------------------+---------+--------+----------+------+------+--------+---------+-------+-------+----------+--------');
  for (const r of rows) {
    if (!r) continue;
    const namePad = r.name.padEnd(32, ' ');
    const rStr = r.recallPct !== null ? `${r.recallPct.toFixed(1).padStart(6, ' ')}%` : '    N/A';
    const pStr = r.precisionPct !== null ? `${r.precisionPct.toFixed(1).padStart(5, ' ')}%` : '   N/A';
    const rmStr = r.rmsePx !== null ? r.rmsePx.toFixed(3).padStart(8, ' ') : '     N/A';
    const hStr = r.hota !== null ? `${(r.hota * 100).toFixed(1).padStart(5, ' ')}%` : '   N/A';
    const capRnd =
      r.auctionCapHits === null ||
      r.auctionCapHits === undefined ||
      r.auctionRoundLimitHits === null ||
      r.auctionRoundLimitHits === undefined
        ? 'N/A'
        : `${r.auctionCapHits}/${r.auctionRoundLimitHits}`;
    const stepsStr = Number.isFinite(r.auctionSteps) ? String(Math.round(r.auctionSteps)) : 'N/A';
    const reacqStr = Number.isFinite(r.reacqEvents) ? String(Math.round(r.reacqEvents)) : 'N/A';
    console.log(
      ` ${namePad}| ${rStr} | ${pStr} | ${rmStr} | ${String(r.idsw).padStart(4, ' ')} | ${String(r.frag).padStart(4, ' ')} | ${hStr} | ${capRnd.padStart(7, ' ')} | ${stepsStr.padStart(5, ' ')} | ${reacqStr.padStart(5, ' ')} | ${r.meanMs.toFixed(3).padStart(8, ' ')} | ${r.medianMs.toFixed(3).padStart(6, ' ')}`
    );
  }
}

// ============================================================================
// Main Runner
// ============================================================================

/** Untimed Wasm Liftoff->TurboFan & Lanczos JS JIT warmup before timed gates. */
async function warmupWasmAndLanczosJit(wasm) {
  const numFrames = 48;
  const nx = 48;
  const ny = 25;
  const cx = HD_W * 0.5;
  const cy = HD_H * 0.5;
  await runSyntheticWasmSequence(wasm, {
    numFrames,
    fps: 30,
    dtSec: 1.0 / 30.0,
    swarmMode: 2,
    resMode: 1,
    checkInspector: true,
    renderFrame: (rgba, f) => {
      const ang = f * 0.025;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const idx = j * nx + i;
          const ox = (i + 0.5) * (HD_W / nx) - cx;
          const oy = (j + 0.5) * (HD_H / ny) - cy;
          const x = Math.max(16, Math.min(HD_W - 16, cx + ox * ca - oy * sa));
          const y = Math.max(16, Math.min(HD_H - 16, cy + ox * sa + oy * ca));
          stampParticleBlob(rgba, HD_W, HD_H, x, y, idx + 1, idx < 3);
        }
      }
    },
  });
  wasm.exp.chronos_init();
}

async function main() {
  const cpuModel = os.cpus()[0]?.model || 'unknown';
  console.log('========================================================================');
  console.log(' CHRONOS-SPECTRAL // Gate Q Benchmark & Ablation Suite (HD 640x360)');
  console.log('========================================================================');

  if (runSelfTestFlag) {
    runSelfTests();
    try {
      inAntiSpoofSelfTest = true;
      await runAntiSpoofSelfTests();
    } finally {
      inAntiSpoofSelfTest = false;
    }
    if (!strictGate && !runAblation && !onlyArg && !runAnalyticFlag && !runSweepFlag) return;
  }

  const wasm = await loadWasm();
  const od = wasm.objdumpReport;
  console.log(
    `[Provenance] node=${process.version} | ${process.platform}/${process.arch} | cpu="${cpuModel}" | wasm=${wasm.wasmBytesLen} B | sha256=${wasm.sha256.slice(0, 16)} | budgetScale=${budgetScale.toFixed(2)}`
  );
  if (od?.available) {
    console.log(
      `[Binary Static] stackInit=${od.stackPointerInit} | maxStackDFS=${od.maxShadowStackDepth} B (cycles=${od.stackCycles}) | .rodataEnd=${od.maxDataEnd} | v128=${od.v128Count} | clz=${od.clzCount} | grow=${od.memoryGrowCount} | indir=${od.callIndirectCount}`
    );
  }

  const hasWave2Wasm = wasm.missing.length === 0;
  const binaryErrors = hasWave2Wasm ? auditWasmBinaryAndAbi(wasm) : [];
  if (!hasWave2Wasm) {
    console.log(`[ABI Notice] Missing CHRONOS-1000 Wasm exports in ${path.relative(ROOT, WASM_PATH)}: ${wasm.missing.join(', ')}`);
    if (strictGate) {
      console.error('[FAIL] Gate Q (--gate) requires complete Wave-2 Wasm exports.');
      process.exit(1);
    }
  } else if (binaryErrors.length > 0) {
    console.error(`[FAIL] Wasm binary/memory audit violation: ${binaryErrors.join('; ')}`);
    if (strictGate) process.exit(1);
  }

  const results = [];
  const q2Scene = shouldRun('Q2') || runAblation || shouldRun('ABLATION') ? createQ2MurmurationScene() : null;
  const ablationRows = [];

  if (runAblation || (onlySet && onlySet.has('ABLATION'))) {
    ablationRows.push(runJsAblationTracker(q2Scene, 'nearest_greedy'));
    ablationRows.push(runJsAblationTracker(q2Scene, 'morton_csr_greedy'));
    ablationRows.push(runJsAblationTracker(q2Scene, 'morton_csr_auction'));
    ablationRows.push(runJsAblationTracker(q2Scene, 'morton_csr_auction_eca'));
    ablationRows.push(runJsAblationTracker(q2Scene, 'morton_csr_auction_eca_knn'));
    ablationRows.push(runJsAblationTracker(q2Scene, 'morton_csr_auction_eca_knn_irl1'));
  }

  if (hasWave2Wasm) {
    await warmupWasmAndLanczosJit(wasm);
    if (shouldRun('Q1')) results.push(await runQ1Vortex(wasm));
    if (shouldRun('Q2')) {
      const q2Res = await runQ2Murmuration(wasm, q2Scene);
      results.push(q2Res);
      if ((runAblation || (onlySet && onlySet.has('ABLATION'))) && q2Res.wasmAblationRow) {
        ablationRows.push(q2Res.wasmAblationRow);
      }
    }
    if (shouldRun('Q3')) results.push(await runQ3ModalBeam(wasm));
    if (shouldRun('Q4')) results.push(await runQ4StarlingRealVideo(wasm));

    if (runAnalyticFlag || strictGate || (onlySet && onlySet.has('ANALYTIC'))) {
      if (shouldRun('ANALYTIC') || runAnalyticFlag) {
        const analyticRes = await runAnalyticSuite(wasm);
        results.push(...analyticRes);
      }
    }
    if (runSweepFlag || strictGate || (onlySet && onlySet.has('SWEEP'))) {
      if (shouldRun('SWEEP') || runSweepFlag) {
        const sweepRes = await runCapacitySweep(wasm);
        results.push(...sweepRes);
      }
    }
  } else {
    const q1 = createQ1VortexScene();
    const q3 = createQ3ModalBeamScene();
    console.log(
      `[Dry-Run Generators] Q1=${q1.gtFrames[0].length} targets x ${q1.numFrames}f, Q2=${(q2Scene || createQ2MurmurationScene()).gtFrames[0].length} targets x 90f, Q3=${q3.gtFrames[0].length} targets x ${q3.numFrames}f (Wasm gates deferred until Wave 2 build).`
    );
  }

  if (ablationRows.length > 0) {
    printAblationTable(ablationRows);
    const greedy = ablationRows[0];
    const csrGreedy = ablationRows[1];
    const auction = ablationRows[2];
    const eca = ablationRows[3];
    const fullJs = ablationRows[5];
    if (greedy && csrGreedy && auction) {
      console.log(
        `  Ablation Delta: Spatial-CSR Speedup ${(greedy.meanMs / Math.max(1e-4, csrGreedy.meanMs)).toFixed(2)}x | Auction vs Greedy: IDSW ${auction.idsw - csrGreedy.idsw}, HOTA ${((auction.hota - csrGreedy.hota) * 100 >= 0 ? '+' : '')}${((auction.hota - csrGreedy.hota) * 100).toFixed(2)}% (capHits=${auction.auctionCapHits}, roundLimitHits=${auction.auctionRoundLimitHits}) | +StaticMask Prec ${(eca.precisionPct - auction.precisionPct >= 0 ? '+' : '')}${(eca.precisionPct - auction.precisionPct).toFixed(1)}% | +Mask+6NN+1StepHuber RMSE ${(fullJs.rmsePx - auction.rmsePx).toFixed(3)} px`
      );
    }
  }

  if (strictGate && results.length === 0) {
    console.error('[FAIL] Zero gate suites evaluated.');
    process.exit(1);
  }

  if (results.length > 0) {
    console.log('\n------------------------------------------------------------------------');
    let allPassed = binaryErrors.length === 0;
    let passCount = 0;
    for (const r of results) {
      const label = r.gate || r.name;
      if (r.passed) {
        passCount++;
        console.log(`  [PASS] ${label}`);
      } else {
        allPassed = false;
        console.error(`  [FAIL] ${label}: ${r.failures.join(', ')}`);
      }
    }
    const isPartial = Boolean(onlySet) || budgetScale > 1.0;
    const status = !allPassed
      ? 'FAIL'
      : noSpeedGate
      ? 'UNCERTIFIED_NO_SPEED_GATE'
      : isPartial
      ? 'PARTIAL'
      : !strictGate
      ? 'DIAGNOSTIC'
      : 'PASS';
    console.log(
      `[Gate Manifest] status=${status} | passed=${passCount}/${results.length} | partial=${isPartial} | only=${onlySet ? [...onlySet].join(',') : 'ALL'} | budgetScale=${budgetScale.toFixed(2)}`
    );
    if (strictGate && !allPassed) {
      process.exit(1);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
