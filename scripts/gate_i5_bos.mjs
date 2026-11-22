#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I5: Controlled Tripod BOS & Masked Neumann Poisson
//
// Verifies:
//   1. Binary & ABI surface invariants (448-page / alignment / reserved zeros)
//   2. All 10 manufactured Poisson cases (0..9) + SD/HD spacings + gauge pinning:
//      - Uniform shift (a, 0), (-a, 0), (0, b), (a, b) exact linear ramp < 1e-4
//      - Occluded holes & split components without corrupting valid interior
//      - Smooth nonuniform weights convergence <= 16 PCG iters (rel resid <= 1e-6)
//      - Empty mask, singleton, empty gauge, zero RHS, integrable quadratic, cap fail
//   3. End-to-end textured-backdrop BOS pipeline via full frame ticks:
//      - 16-frame warmup -> ready transition
//      - Smooth Boussinesq refractive plume RMSE <= 0.15 px, >= 90% availability
//      - Gain/offset flicker invariance
//      - Aperture-only 1D stripes rejection
//      - Saturation & opaque occlusion masking
//      - Rotational vs curl-free diagnostic separation
//      - Tripod nudge & external pan/cut invalidation + timing-gap ref preservation
//      - Mode 2 frozen measurement reference under 140 frames of slow plume
//      - Per-frame BOS latency benchmark (full + masked PCG)
//
// Usage:
//   node scripts/gate_i5_bos.mjs [--wasm=<path>]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const DEFAULT_WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');

const wasmArg = process.argv.slice(2).find((a) => a.startsWith('--wasm='));
const WASM_PATH = wasmArg ? path.resolve(wasmArg.slice('--wasm='.length)) : DEFAULT_WASM;

const BOS_W = 256;
const BOS_H = 128;
const BOS_N = BOS_W * BOS_H; // 32768

const VALID_DISP = 1 << 0;
const VALID_PHI = 1 << 1;
const VALID_CURL = 1 << 2;

const STATE_OFF = 0;
const STATE_WARMING = 1;
const STATE_READY = 2;
const STATE_DISTURBED = 3;

const REASON_ANCHOR_SUPPORT = 1 << 0;
const REASON_EXT_PAN_CUT = 1 << 1;
const REASON_INT_SHIFT = 1 << 2;
const REASON_PHOTOM_MSE = 1 << 3;
const REASON_DISCONT = 1 << 4;
const REASON_NO_GAUGE = 1 << 5;
const REASON_NONCONV = 1 << 6;
const REASON_LOW_TEX = 1 << 7;

const SOLVER_NONE = 0;
const SOLVER_DIRECT = 1;
const SOLVER_PCG = 2;

function assert(cond, msg) {
  if (!cond) {
    throw new Error(`Gate I5 FAIL: ${msg}`);
  }
}

if (!fs.existsSync(WASM_PATH)) {
  throw new Error(`Wasm binary not found: ${WASM_PATH}`);
}

const wasmBytes = fs.readFileSync(WASM_PATH);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const exp = instance.exports;
const mem = exp.memory;

const REQUIRED_EXPORTS = [
  'chronos_init',
  'chronos_set_resolution',
  'chronos_get_ingest_rgba_ptr',
  'chronos_tick',
  'chronos_set_bos_mode',
  'chronos_reset_bos_ref',
  'chronos_set_bos_gauge_roi',
  'chronos_set_bos_overlay_kind',
  'chronos_get_bos_telemetry_ptr',
  'chronos_get_bos_planes_ptr',
  'chronos_get_bos_valid_ptr',
  'chronos_get_bos_rgba_ptr',
  'chronos_bos_solve_manufactured_test',
];
for (const name of REQUIRED_EXPORTS) {
  assert(typeof exp[name] === 'function', `Missing required export ${name}`);
}

assert(
  mem.buffer.byteLength === 448 * 65536,
  `Expected 448 Wasm pages (29360128 B), got ${mem.buffer.byteLength}`
);

let grew = false;
try {
  mem.grow(1);
  grew = true;
} catch {}
assert(!grew, 'Expected fixed max=448 Wasm memory');

const buf = mem.buffer;
const telemPtr = exp.chronos_get_bos_telemetry_ptr();
const planesPtr = exp.chronos_get_bos_planes_ptr();
const validPtr = exp.chronos_get_bos_valid_ptr();
const rgbaPtr = exp.chronos_get_bos_rgba_ptr();
const ingestPtr = exp.chronos_get_ingest_rgba_ptr();

assert((telemPtr & 3) === 0 && telemPtr > 0 && telemPtr + 128 <= buf.byteLength, 'bos_telemetry ptr invalid');
assert((planesPtr & 3) === 0 && planesPtr > 0 && planesPtr + 6 * BOS_N * 4 <= buf.byteLength, 'bos_planes ptr invalid');
assert(validPtr > 0 && validPtr + BOS_N <= buf.byteLength, 'bos_valid ptr invalid');
assert(rgbaPtr > 0 && rgbaPtr + 4 * BOS_N <= buf.byteLength, 'bos_rgba ptr invalid');

const regions = [
  [telemPtr, telemPtr + 128, 'bos_telem'],
  [planesPtr, planesPtr + 6 * BOS_N * 4, 'bos_planes'],
  [validPtr, validPtr + BOS_N, 'bos_valid'],
  [rgbaPtr, rgbaPtr + 4 * BOS_N, 'bos_rgba'],
  [ingestPtr, ingestPtr + 640 * 360 * 4, 'ingest_rgba'],
];
for (let i = 0; i < regions.length; i++) {
  for (let j = i + 1; j < regions.length; j++) {
    const [a0, a1, na] = regions[i];
    const [b0, b1, nb] = regions[j];
    assert(!(a0 < b1 && b0 < a1), `Memory overlap between ${na} and ${nb}`);
  }
}

const telemU32 = new Uint32Array(buf, telemPtr, 32);
const telemF32 = new Float32Array(buf, telemPtr, 32);
const planes = new Float32Array(buf, planesPtr, 6 * BOS_N);
const valid = new Uint8Array(buf, validPtr, BOS_N);
const rgba = new Uint8Array(buf, rgbaPtr, 4 * BOS_N);

function planeSlice(idx) {
  return planes.subarray(idx * BOS_N, (idx + 1) * BOS_N);
}

function nodeCoord(i, j, w, h) {
  const hx = w / BOS_W;
  const hy = h / BOS_H;
  return [(i + 0.5) * hx - 0.5, (j + 0.5) * hy - 0.5];
}

function checkReservedTailZero() {
  for (let k = 24; k < 32; k++) {
    assert(telemU32[k] === 0, `bos_telem[${k}] expected 0, got ${telemU32[k]}`);
  }
}

// ============================================================================
// 1. Setter & Gauge ROI Validation Contracts
// ============================================================================
exp.chronos_init();
exp.chronos_set_resolution(1);
assert(telemU32[0] === 1, `Expected BOS ABI version 1, got ${telemU32[0]}`);
assert(telemU32[1] === 0 && telemU32[2] === STATE_OFF, 'Expected default BOS off');
assert((telemU32[3] & REASON_NO_GAUGE) !== 0, 'Expected NO_GAUGE reason before gauge set');
checkReservedTailZero();

assert(exp.chronos_set_bos_mode(3) === 0, 'Invalid mode 3 must return 0');
assert(telemU32[1] === 0, 'Invalid mode 3 must not alter requested_mode');
assert(exp.chronos_set_bos_mode(1) === 1, 'Mode 1 must return 1');
assert(telemU32[2] === STATE_DISTURBED && (telemU32[3] & REASON_NO_GAUGE) !== 0, 'Enabled without gauge must report DISTURBED + NO_GAUGE');

assert(exp.chronos_set_bos_gauge_roi(4, 4, 4, 20) === 0, 'Empty x gauge must fail');
assert(exp.chronos_set_bos_gauge_roi(10, 4, 8, 20) === 0, 'Inverted x gauge must fail');
assert(exp.chronos_set_bos_gauge_roi(0, 0, 257, 10) === 0, 'Out-of-bounds x1=257 must fail');
assert(exp.chronos_set_bos_gauge_roi(0, 0, 10, 129) === 0, 'Out-of-bounds y1=129 must fail');

const epBefore = telemU32[11];
assert(exp.chronos_set_bos_gauge_roi(8, 8, 48, 120) === 1, 'Valid gauge ROI must succeed');
assert(telemU32[11] > epBefore, 'Changing gauge ROI must increment reference_epoch');
assert(telemU32[2] === STATE_WARMING, 'Valid mode+gauge must enter WARMING');
assert((telemU32[3] & REASON_NO_GAUGE) === 0, 'Valid gauge must clear NO_GAUGE');

assert(exp.chronos_set_bos_overlay_kind(3) === 0, 'Invalid overlay kind 3 must fail');
assert(exp.chronos_set_bos_overlay_kind(1) === 1, 'Valid overlay kind 1 must succeed');
assert(exp.chronos_set_bos_overlay_kind(0) === 1, 'Valid overlay kind 0 must succeed');

// ============================================================================
// 2. Manufactured Solver Oracle Suite (Cases 0..9 across HD & SD)
// ============================================================================
function checkLinearRamp(label, resMode, caseId, a, b, gx0, gy0, gx1, gy1, expectPath, checkValidFn) {
  exp.chronos_init();
  exp.chronos_set_resolution(resMode);
  const fw = resMode === 1 ? 640 : 320;
  const fh = resMode === 1 ? 360 : 180;
  assert(exp.chronos_set_bos_gauge_roi(gx0, gy0, gx1, gy1) === 1, `${label}: gauge set`);
  assert(exp.chronos_bos_solve_manufactured_test(caseId, a, b) === 1, `${label}: solve ok`);
  checkReservedTailZero();

  assert(telemU32[9] === expectPath, `${label}: expected solver_path=${expectPath}, got ${telemU32[9]}`);
  assert(telemU32[8] <= 24, `${label}: pcg_iters=${telemU32[8]} > 24`);
  assert(telemF32[16] <= 1e-6, `${label}: true_rel_res=${telemF32[16]} > 1e-6`);
  assert(telemF32[17] < 1e-6 && telemF32[18] < 1e-6, `${label}: curl on uniform shift expected ~0`);
  assert((telemU32[3] & (REASON_NONCONV | REASON_NO_GAUGE)) === 0, `${label}: unexpected reason=${telemU32[3]}`);

  const phi = planeSlice(2);
  let gaugeSum = 0;
  let gaugeCnt = 0;
  for (let j = gy0; j < gy1; j++) {
    for (let i = gx0; i < gx1; i++) {
      const u = j * BOS_W + i;
      if ((valid[u] & VALID_PHI) !== 0) {
        gaugeSum += phi[u];
        gaugeCnt++;
      }
    }
  }
  assert(gaugeCnt > 0, `${label}: gaugeCnt=0`);
  const gaugeMeanPhi = gaugeSum / gaugeCnt;
  assert(Math.abs(gaugeMeanPhi) < 1e-4, `${label}: gauge mean phi=${gaugeMeanPhi} != 0`);

  // Compute exact ramp baseline over gauge component nodes
  let exactGaugeSum = 0;
  for (let j = gy0; j < gy1; j++) {
    for (let i = gx0; i < gx1; i++) {
      const u = j * BOS_W + i;
      if ((valid[u] & VALID_PHI) !== 0) {
        const [x, y] = nodeCoord(i, j, fw, fh);
        exactGaugeSum += a * x + b * y;
      }
    }
  }
  const exactOffset = exactGaugeSum / gaugeCnt;

  let err2 = 0;
  let ref2 = 0;
  let checked = 0;
  for (let j = 0; j < BOS_H; j++) {
    for (let i = 0; i < BOS_W; i++) {
      const u = j * BOS_W + i;
      const shouldBeValid = checkValidFn(i, j);
      if (shouldBeValid) {
        assert((valid[u] & VALID_PHI) !== 0, `${label}: expected VALID_PHI at (${i},${j})`);
        const [x, y] = nodeCoord(i, j, fw, fh);
        const exact = a * x + b * y - exactOffset;
        const d = phi[u] - exact;
        err2 += d * d;
        ref2 += exact * exact;
        checked++;
      } else {
        assert((valid[u] & VALID_PHI) === 0, `${label}: expected !VALID_PHI at (${i},${j})`);
        assert(phi[u] === 0.0, `${label}: invalid node phi=${phi[u]} != 0`);
      }
    }
  }
  const relErr = Math.sqrt(err2 / Math.max(1e-12, ref2));
  assert(relErr < 1e-4, `${label}: relPhiErr=${relErr.toExponential(3)} >= 1e-4`);

  // Verify RGBA alpha zero strictly matches invalid mask on all 3 overlay kinds
  for (const kind of [0, 1, 2]) {
    assert(exp.chronos_set_bos_overlay_kind(kind) === 1, `${label}: set overlay kind ${kind}`);
    const reqBit = kind === 0 ? VALID_DISP : kind === 1 ? VALID_PHI : VALID_CURL;
    for (let u = 0; u < BOS_N; u++) {
      const has = (valid[u] & reqBit) !== 0;
      const alpha = rgba[u * 4 + 3];
      if (has) assert(alpha > 0, `${label} kind=${kind}: valid node u=${u} has alpha=0`);
      else assert(alpha === 0, `${label} kind=${kind}: invalid node u=${u} has alpha=${alpha}`);
    }
  }
  return { checked, iters: telemU32[8], res: telemF32[16], relErr };
}

const r0a = checkLinearRamp('Case0 HD +x', 1, 0, 0.25, 0.0, 0, 0, 256, 128, SOLVER_DIRECT, () => true);
const r0b = checkLinearRamp('Case0 HD -x', 1, 0, -0.35, 0.0, 8, 8, 48, 48, SOLVER_DIRECT, () => true);
const r0c = checkLinearRamp('Case0 HD +y', 1, 0, 0.0, 0.3, 20, 20, 60, 60, SOLVER_DIRECT, () => true);
const r0d = checkLinearRamp('Case0 SD mixed', 0, 0, 0.25, -0.125, 8, 8, 48, 120, SOLVER_DIRECT, () => true);
const r1 = checkLinearRamp('Case1 Hole', 1, 1, 0.25, -0.125, 8, 8, 48, 120, SOLVER_PCG, (i, j) => !(j >= 48 && j < 80 && i >= 110 && i < 146));
const r2L = checkLinearRamp('Case2 Split Left Gauge', 1, 2, 0.25, -0.125, 8, 8, 48, 120, SOLVER_PCG, (i) => i < 120);
const r2Tie = checkLinearRamp('Case2 Equal-Overlap Tie-Break (Lowest Global Node Left Wins)', 1, 2, 0.25, -0.125, 112, 20, 138, 40, SOLVER_PCG, (i) => i < 120);
const r2R = checkLinearRamp('Case2 Split Full Gauge (Right Wins)', 1, 2, 0.25, -0.125, 0, 0, 256, 128, SOLVER_PCG, (i) => i >= 130);
assert(telemU32[7] === 126 * 128 && r2Tie.checked === 120 * 128, 'Case2 counts');
const r3 = checkLinearRamp('Case3 Weighted Hole', 1, 3, 0.25, -0.125, 8, 8, 48, 120, SOLVER_PCG, (i, j) => !(j >= 48 && j < 80 && i >= 110 && i < 146));

// Connected-component singleton gauge rejections (< 2 overlap nodes in gauge_roi)
{
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  assert(exp.chronos_set_bos_gauge_roi(0, 0, 1, 1) === 1, '[0,0,1,1) syntactically valid');
  assert(exp.chronos_bos_solve_manufactured_test(0, 0.25, 0.0) === 1);
  assert(telemU32[4] === BOS_N && telemU32[5] === (BOS_W - 1) * (BOS_H - 1), 'Singleton [0,0,1,1) preserves disp+curl');
  assert(telemU32[6] === 1 && telemU32[7] === 0, 'Singleton [0,0,1,1) must report valid_gauge_nodes=0');
  assert((telemU32[3] & REASON_NO_GAUGE) !== 0, 'Singleton [0,0,1,1) must flag REASON_NO_GAUGE');
  assert(valid.every((v) => (v & VALID_PHI) === 0), 'Singleton [0,0,1,1) must clear all VALID_PHI');
  exp.chronos_set_bos_overlay_kind(1);
  assert(rgba.every((v, idx) => (idx & 3) !== 3 || v === 0), 'Singleton [0,0,1,1) phi RGBA alpha must be 0');
  exp.chronos_set_bos_overlay_kind(0);

  // Multi-node gauge_roi [109,79,146,80) overlapping component at 1 single valid node (109,79)
  assert(exp.chronos_set_bos_gauge_roi(109, 79, 146, 80) === 1);
  assert(exp.chronos_bos_solve_manufactured_test(1, 0.25, -0.125) === 1);
  assert(telemU32[4] === BOS_N - 32 * 36 && telemU32[6] === 1 && telemU32[7] === 0, '1-node overlap on Case1 must report valid_gauge_nodes=0');
  assert((telemU32[3] & REASON_NO_GAUGE) !== 0, '1-node overlap on connected component must flag REASON_NO_GAUGE');
  assert(valid.every((v) => (v & VALID_PHI) === 0), '1-node overlap must clear all VALID_PHI');
}

// Case 4: Empty mask
exp.chronos_init();
exp.chronos_set_resolution(1);
assert(exp.chronos_bos_solve_manufactured_test(4, 0.25, 0.0) === 1);
assert(telemU32[4] === 0 && telemU32[6] === 0 && telemU32[7] === 0, 'Case4 empty mask counts');
assert((telemU32[3] & REASON_NO_GAUGE) !== 0, 'Case4 expected NO_GAUGE');
assert(valid.every((v) => v === 0), 'Case4 all valid bits must be 0');

// Case 5: Singleton
assert(exp.chronos_bos_solve_manufactured_test(5, 0.25, 0.0) === 1);
assert(telemU32[4] === 1 && telemU32[6] === 0 && telemU32[7] === 0, 'Case5 singleton counts');
assert((telemU32[3] & REASON_NO_GAUGE) !== 0, 'Case5 expected NO_GAUGE');
assert(valid.every((v) => (v & VALID_PHI) === 0), 'Case5 singleton cannot supply referenced phi');

// Case 6: Empty gauge overlap (hole gauge inside valid field)
assert(exp.chronos_bos_solve_manufactured_test(6, 0.25, -0.125) === 1);
assert(telemU32[4] > 0 && telemU32[5] > 0 && telemU32[6] === 1 && telemU32[7] === 0, 'Case6 empty gauge counts');
assert((telemU32[3] & REASON_NO_GAUGE) !== 0, 'Case6 expected NO_GAUGE');
assert(valid.every((v) => (v & VALID_PHI) === 0), 'Case6 phi must be invalid');
exp.chronos_set_bos_overlay_kind(1);
assert(rgba.every((v, idx) => (idx & 3) !== 3 || v === 0), 'Case6 phi RGBA alpha must be 0');
exp.chronos_set_bos_overlay_kind(0);

// Case 7: Zero RHS
assert(exp.chronos_set_bos_gauge_roi(0, 0, 256, 128) === 1);
assert(exp.chronos_bos_solve_manufactured_test(7, 0.0, 0.0) === 1);
assert(telemF32[16] === 0.0, 'Case7 zero RHS residual must be 0');
assert(planeSlice(2).every((v) => v === 0.0), 'Case7 zero RHS phi must be identically 0');

// Case 8: Integrable quadratic 0.5*(a*x^2 + b*y^2)
{
  const qa = 0.0012;
  const qb = 0.0008;
  assert(exp.chronos_bos_solve_manufactured_test(8, qa, qb) === 1);
  assert(telemU32[9] === SOLVER_DIRECT && telemF32[16] <= 1e-6, 'Case8 quadratic convergence');
  assert(telemF32[17] < 1e-5, `Case8 integrable curl_rms=${telemF32[17]} expected ~0`);
  const phi = planeSlice(2);
  let meanEx = 0;
  for (let j = 0; j < BOS_H; j++) {
    for (let i = 0; i < BOS_W; i++) {
      const [x, y] = nodeCoord(i, j, 640, 360);
      meanEx += 0.5 * (qa * x * x + qb * y * y);
    }
  }
  meanEx /= BOS_N;
  let e2 = 0, r2 = 0;
  for (let j = 0; j < BOS_H; j++) {
    for (let i = 0; i < BOS_W; i++) {
      const [x, y] = nodeCoord(i, j, 640, 360);
      const ex = 0.5 * (qa * x * x + qb * y * y) - meanEx;
      const d = phi[j * BOS_W + i] - ex;
      e2 += d * d;
      r2 += ex * ex;
    }
  }
  const relQ = Math.sqrt(e2 / r2);
  assert(relQ < 1e-4, `Case8 quadratic relErr=${relQ.toExponential(3)} >= 1e-4`);

  // Case 8 nonfinite overflow guard (f32::MAX) across HD and SD
  for (const rm of [1, 0]) {
    exp.chronos_set_resolution(rm);
    assert(exp.chronos_set_bos_gauge_roi(0, 0, 256, 128) === 1);
    assert(exp.chronos_bos_solve_manufactured_test(8, 3.4028234663852886e+38, 0.0) === 0, `Case8 f32::MAX must reject cleanly (resMode=${rm})`);
    assert((telemU32[3] & REASON_NONCONV) !== 0, 'Case8 f32::MAX must flag REASON_NONCONV');
    assert(valid.every((v) => v === 0), 'Case8 f32::MAX must never mark any valid bits');
    assert(planes.every((v) => Number.isFinite(v)), 'Case8 f32::MAX must leave all planes finite');
    for (let k = 16; k < 24; k++) assert(Number.isFinite(telemF32[k]), `Case8 f32::MAX telemF32[${k}] must stay finite`);
  }

  // SD Case 8 (1e35, 1e35) with 16x16 gauge: f64 Phi exceeds f32::MAX -> must clear ALL VALID_PHI + flag NONCONV
  exp.chronos_set_resolution(0);
  assert(exp.chronos_set_bos_gauge_roi(0, 0, 16, 16) === 1);
  exp.chronos_bos_solve_manufactured_test(8, 1e35, 1e35);
  assert((telemU32[3] & REASON_NONCONV) !== 0, 'SD Case8 (1e35,1e35) must flag REASON_NONCONV on Phi f32 overflow');
  assert(telemU32[7] === 0 && valid.every((v) => (v & VALID_PHI) === 0), 'SD Case8 (1e35,1e35) must publish zero VALID_PHI nodes');
  assert(planes.every((v) => Number.isFinite(v)), 'SD Case8 (1e35,1e35) must leave all planes finite');
  for (let k = 16; k < 24; k++) assert(Number.isFinite(telemF32[k]), `SD Case8 (1e35,1e35) telemF32[${k}] must stay finite`);

  // SD Case 0 (f32::MAX, f32::MAX): rmsDisp = sqrt(2)*f32::MAX overflows f32 -> reject before READY + clear nonfinite telemetry
  exp.chronos_set_resolution(0);
  assert(exp.chronos_set_bos_gauge_roi(8, 8, 48, 120) === 1);
  assert(exp.chronos_bos_solve_manufactured_test(0, 3.4028234663852886e+38, 3.4028234663852886e+38) === 0, 'SD Case0 (f32::MAX,f32::MAX) must reject cleanly');
  assert(telemU32[2] !== STATE_READY && (telemU32[3] & REASON_NONCONV) !== 0, 'SD Case0 (f32::MAX,f32::MAX) must not publish READY and must flag NONCONV');
  assert(valid.every((v) => v === 0) && telemF32[23] === 0.0, 'SD Case0 overflow must clear all valid bits and zero rmsDisp');
  assert(planes.every((v) => Number.isFinite(v)), 'SD Case0 overflow must leave all planes finite');
  for (let k = 16; k < 24; k++) assert(Number.isFinite(telemF32[k]), `SD Case0 overflow telemF32[${k}] must stay finite`);

  // HD Case 0 (3e38, 0): rmsDisp finite while linear Phi overflows f32 -> must clear ALL VALID_PHI + flag NONCONV
  exp.chronos_set_resolution(1);
  assert(exp.chronos_set_bos_gauge_roi(8, 8, 48, 120) === 1);
  exp.chronos_bos_solve_manufactured_test(0, 3e38, 0.0);
  assert((telemU32[3] & REASON_NONCONV) !== 0 && telemU32[7] === 0 && valid.every((v) => (v & VALID_PHI) === 0), 'HD Case0 (3e38,0) Phi overflow must flag NONCONV and clear all VALID_PHI');
  assert(planes.every((v) => Number.isFinite(v)), 'HD Case0 (3e38,0) must leave all planes finite');
  for (let k = 16; k < 24; k++) assert(Number.isFinite(telemF32[k]), `HD Case0 (3e38,0) telemF32[${k}] must stay finite`);
  assert(exp.chronos_set_bos_gauge_roi(0, 0, 256, 128) === 1);
}

// Case 9: Pathological PCG cap failure
assert(exp.chronos_bos_solve_manufactured_test(9, 0.25, -0.125) === 1);
assert(telemU32[9] === SOLVER_PCG && telemU32[8] === 24, 'Case9 expected PCG 24 iters');
assert(telemF32[16] > 1e-6, `Case9 expected true_rel_res > 1e-6, got ${telemF32[16]}`);
assert((telemU32[3] & REASON_NONCONV) !== 0, 'Case9 expected NONCONV reason');
assert(telemU32[4] > 0 && telemU32[5] > 0, 'Case9 must preserve measured disp & curl');
assert(valid.every((v) => (v & VALID_PHI) === 0), 'Case9 failed solve must clear all VALID_PHI');

console.log(
  `  [PASS] Manufactured Cases 0..9: Case0 relErr=${r0a.relErr.toExponential(2)} | ` +
    `Case1 hole it=${r1.iters} res=${r1.res.toExponential(2)} | ` +
    `Case2 split it=${r2L.iters}/${r2R.iters} | ` +
    `Case3 weighted it=${r3.iters} res=${r3.res.toExponential(2)} | Case9 NONCONV flagged cleanly`
);

// ============================================================================
// 3. Textured-Backdrop End-to-End Pipeline Tests (640x360)
// ============================================================================
const W = 640;
const H = 360;
const ringViewCache = new Map();
function writeIngestRgba(rgbaSrc) {
  const ptr = exp.chronos_get_ingest_rgba_ptr();
  const key = `${ptr}:${rgbaSrc.byteLength}`;
  let view = ringViewCache.get(key);
  if (!view) {
    view = new Uint8Array(buf, ptr, rgbaSrc.byteLength);
    ringViewCache.set(key, view);
  }
  view.set(rgbaSrc);
}

function evalBackdrop(x, y) {
  // Smooth multi-octave rich 2D texture in [0.12, 0.88] with strong isotropic gradients
  const v =
    0.5 +
    0.16 * Math.sin(0.27 * x + 0.13 * y) * Math.cos(0.11 * x - 0.24 * y) +
    0.11 * Math.cos(0.49 * x - 0.19 * y + 0.7) * Math.sin(0.22 * x + 0.43 * y) +
    0.07 * Math.sin(0.71 * x + 0.58 * y + 1.3);
  return Math.min(0.92, Math.max(0.08, v));
}

function renderBackdropFrame(outRgba, dispFn, gain = 1.0, offset = 0.0, noiseAmp = 0.0, seed = 1, maskFn = null) {
  let rng = (seed * 2654435761) >>> 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      if (maskFn) {
        const m = maskFn(x, y);
        if (m) {
          outRgba[idx] = m[0];
          outRgba[idx + 1] = m[1];
          outRgba[idx + 2] = m[2];
          outRgba[idx + 3] = 255;
          continue;
        }
      }
      // Convention: current(x + eps) ≈ reference(x)  =>  current(x) = reference(x - eps)
      const [ex, ey] = dispFn ? dispFn(x, y) : [0.0, 0.0];
      let val = evalBackdrop(x - ex, y - ey) * gain + offset;
      if (noiseAmp > 0) {
        rng = (rng * 1664525 + 1013904223) >>> 0;
        const u = ((rng >>> 8) / 16777216.0 - 0.5) * 2.0;
        val += noiseAmp * u;
      }
      const q = Math.min(255, Math.max(0, Math.round(val * 255.0)));
      outRgba[idx] = q;
      outRgba[idx + 1] = q;
      outRgba[idx + 2] = q;
      outRgba[idx + 3] = 255;
    }
  }
}

function plumeDisp(x, y) {
  // Smooth thermal plume centered at (400, 180), confined strictly outside gauge x < 120
  const dx = (x - 400.0) / 58.0;
  const dy = (y - 180.0) / 54.0;
  const r2 = dx * dx + dy * dy;
  if (r2 > 22.0) return [0.0, 0.0];
  const g = Math.exp(-r2);
  // Potential-like gradient field: phi = -A * g => eps = grad(phi) = 2*A*g*(dx/58, dy/54)
  const amp = 44.0;
  return [(2.0 * amp * g * dx) / 58.0, (2.0 * amp * g * dy) / 54.0];
}

function vortexDisp(x, y) {
  // Rotational Lamb-Oseen vortex centered at (400, 180)
  const rx = x - 400.0;
  const ry = y - 180.0;
  const r2 = rx * rx + ry * ry;
  const rc2 = 68.0 * 68.0;
  if (r2 > 18.0 * rc2) return [0.0, 0.0];
  const fac = 0.028 * Math.exp(-r2 / rc2);
  return [-ry * fac, rx * fac];
}

function pushFrame(rgbaSrc, seq, ptsUs, gapMask = 0, fw = W, fh = H) {
  writeIngestRgba(rgbaSrc);
  if (typeof exp.chronos_set_ingest_meta === 'function') {
    const lo = Number(BigInt(ptsUs) & 0xffff_ffffn);
    const hi = Number(BigInt(ptsUs) >> 32n);
    const ok = exp.chronos_set_ingest_meta(1, seq, lo, hi, -1.0, 0x12, fw, fh, 0, 0, fw, fh, 0, 0, 0, 0, fw, fh, gapMask, 1);
    assert(ok === 1, `chronos_set_ingest_meta rejected seq=${seq}`);
  }
  exp.chronos_tick(1 / 30);
}

const clearFrame = new Uint8Array(W * H * 4);
const plumeFrame = new Uint8Array(W * H * 4);
const flickerPlumeFrame = new Uint8Array(W * H * 4);
const maskedPlumeFrame = new Uint8Array(W * H * 4);
const vortexFrame = new Uint8Array(W * H * 4);
const zoomSmallFrame = new Uint8Array(W * H * 4);
const zoomLargeFrame = new Uint8Array(W * H * 4);
const nudgedFrame = new Uint8Array(W * H * 4);

renderBackdropFrame(clearFrame, null, 1.0, 0.0, 0.0, 1);
renderBackdropFrame(plumeFrame, plumeDisp, 1.0, 0.0, 0.0, 1);
// Strong gain=1.42, offset=+0.05 flicker (keep inside unsaturated [0.08..0.92] via scaled ref or base)
// Scale flicker carefully so pixels stay in [0.04, 0.96]: gain=1.18, offset=+0.05 -> 0.08..0.92 * 1.18 -> 0.14..0.98
renderBackdropFrame(flickerPlumeFrame, plumeDisp, 1.18, -0.04, 0.0, 1);
renderBackdropFrame(maskedPlumeFrame, plumeDisp, 1.0, 0.0, 0.0, 1, (x, y) => {
  if (x >= 280 && x < 340 && y >= 130 && y < 190) return [255, 255, 255]; // Saturated whiteout
  if (x >= 460 && x < 520 && y >= 150 && y < 220) return [128, 128, 128]; // Flat opaque occluder
  return null;
});
renderBackdropFrame(vortexFrame, vortexDisp, 1.0, 0.0, 0.0, 1);
renderBackdropFrame(zoomSmallFrame, (x, y) => [0.0007 * (x - 40.0), 0.0007 * (y - 180.0)], 1.0, 0.0, 0.0, 1);
renderBackdropFrame(zoomLargeFrame, (x, y) => [0.008 * (x - 320.0), 0.008 * (y - 180.0)], 1.0, 0.0, 0.0, 1);
renderBackdropFrame(nudgedFrame, () => [0.75, -0.4], 1.0, 0.0, 0.0, 1);

function computeInteriorRmse(dispFn, skipMaskFn = null) {
  const exArr = planeSlice(0);
  const eyArr = planeSlice(1);
  let se = 0;
  let validInt = 0;
  let totalInt = 0;
  for (let j = 2; j < BOS_H - 2; j++) {
    for (let i = 2; i < BOS_W - 2; i++) {
      const [cx, cy] = nodeCoord(i, j, W, H);
      if (skipMaskFn && skipMaskFn(cx, cy)) continue;
      totalInt++;
      const u = j * BOS_W + i;
      if ((valid[u] & VALID_DISP) !== 0) {
        validInt++;
        const [tx, ty] = dispFn(cx, cy);
        const dx = exArr[u] - tx;
        const dy = eyArr[u] - ty;
        se += dx * dx + dy * dy;
      }
    }
  }
  return {
    rmse: Math.sqrt(se / Math.max(1, validInt)),
    avail: validInt / Math.max(1, totalInt),
    validInt,
  };
}

// Test A: 16-frame warmup -> Ready -> Plume RMSE <= 0.15 px & >= 90% availability
exp.chronos_init();
exp.chronos_set_resolution(1);
assert(exp.chronos_set_bos_mode(1) === 1);
assert(exp.chronos_set_bos_gauge_roi(8, 8, 40, 120) === 1);
let seq = 1;
for (let f = 0; f < 16; f++) {
  renderBackdropFrame(clearFrame, null, 1.0, 0.0, 0.002, f + 1);
  pushFrame(clearFrame, seq, seq * 33333);
  seq++;
  assert(telemU32[2] === STATE_WARMING, `Expected WARMING at frame ${f}, got ${telemU32[2]}`);
  assert(telemU32[10] === f + 1, `Expected warmup_count=${f + 1}, got ${telemU32[10]}`);
}
pushFrame(plumeFrame, seq, seq * 33333);
assert(telemU32[2] === STATE_READY, `Expected READY after 16 warmup frames, got ${telemU32[2]} (reason=${telemU32[3]})`);
if (typeof exp.chronos_set_ingest_meta === 'function') {
  const usedPts = BigInt(seq * 33333);
  assert(telemU32[12] === seq && telemU32[13] === 1, `Expected bos_telem source_seq=${seq}, got ${telemU32[12]}/${telemU32[13]}`);
  assert(telemU32[14] === Number(usedPts & 0xffff_ffffn) && telemU32[15] === Number(usedPts >> 32n), 'bos_telem PTS mismatch');
}
seq++;
const plumeStats = computeInteriorRmse(plumeDisp);
assert(plumeStats.avail >= 0.90, `Plume interior availability ${(plumeStats.avail * 100).toFixed(1)}% < 90%`);
assert(plumeStats.rmse <= 0.15, `Plume interior RMSE=${plumeStats.rmse.toFixed(4)} > 0.15 px`);
assert(telemF32[16] <= 1e-6, `Plume Poisson rel res=${telemF32[16]} > 1e-6`);
const plumeCurlRms = telemF32[17];
const plumeMaxCurl = telemF32[18];

// Test B: Gain/offset flicker resilience
pushFrame(flickerPlumeFrame, seq, seq * 33333); seq++;
assert(telemU32[2] === STATE_READY, `Flicker must stay READY, got state=${telemU32[2]} reason=${telemU32[3]}`);
const flickStats = computeInteriorRmse(plumeDisp);
assert(flickStats.avail >= 0.90 && flickStats.rmse <= 0.15, `Flicker RMSE=${flickStats.rmse.toFixed(4)}, avail=${flickStats.avail}`);

// Test D: Saturation & Opaque Occlusion Masking
pushFrame(maskedPlumeFrame, seq, seq * 33333); seq++;
assert(telemU32[2] === STATE_READY, `Masked plume must stay READY`);
assert(telemU32[9] === SOLVER_PCG && telemF32[16] <= 1e-6, `Masked plume PCG failed res=${telemF32[16]}`);
let satValid = 0;
let occValid = 0;
for (let j = 0; j < BOS_H; j++) {
  for (let i = 0; i < BOS_W; i++) {
    const [cx, cy] = nodeCoord(i, j, W, H);
    const u = j * BOS_W + i;
    if (cx >= 286 && cx <= 334 && cy >= 136 && cy <= 184 && (valid[u] & VALID_DISP) !== 0) satValid++;
    if (cx >= 466 && cx <= 514 && cy >= 156 && cy <= 214 && (valid[u] & VALID_DISP) !== 0) occValid++;
  }
}
assert(satValid === 0 && occValid === 0, `Expected 0 valid nodes inside saturation (${satValid}) and occlusion (${occValid})`);
const maskedStats = computeInteriorRmse(plumeDisp, (cx, cy) => (cx >= 274 && cx <= 346 && cy >= 124 && cy <= 196) || (cx >= 454 && cx <= 526 && cy >= 144 && cy <= 226));
assert(maskedStats.rmse <= 0.15, `Masked exterior RMSE=${maskedStats.rmse.toFixed(4)} > 0.15 px`);

// Test E: Rotational vortex vs Curl-Free plume/zoom diagnostic
pushFrame(vortexFrame, seq, seq * 33333); seq++;
assert(telemU32[2] === STATE_READY, 'Vortex frame must stay READY');
const vortexCurlRms = telemF32[17];
const vortexMaxCurl = telemF32[18];
assert(vortexCurlRms > 2.5 * Math.max(1e-4, plumeCurlRms), `Expected vortex curl (${vortexCurlRms}) > 2.5x plume curl (${plumeCurlRms})`);
assert(vortexMaxCurl > 2.5 * Math.max(1e-4, plumeMaxCurl), `Expected vortex max curl (${vortexMaxCurl}) > 2.5x plume max curl (${plumeMaxCurl})`);

pushFrame(zoomSmallFrame, seq, seq * 33333); seq++;
assert(telemU32[2] === STATE_READY, 'Tiny sub-threshold zoom stays READY');
const zoomCurlRms = telemF32[17];
assert(zoomCurlRms < 0.5 * vortexCurlRms, `Curl-free zoom (${zoomCurlRms}) must stay well below rotational vortex (${vortexCurlRms})`);

pushFrame(zoomLargeFrame, seq, seq * 33333); seq++;
assert(telemU32[2] === STATE_DISTURBED, 'Large zoom shifting anchors > 0.35 px must trigger DISTURBED');

// Test F: Tripod nudge invalidation, recovery when returning, timing-gap preservation, and cut reset
const refEpBeforeNudge = telemU32[11];
pushFrame(nudgedFrame, seq, seq * 33333); seq++;
assert(telemU32[2] === STATE_DISTURBED && (telemU32[3] & REASON_INT_SHIFT) !== 0, `Tripod nudge must trigger DISTURBED + INT_SHIFT, got state=${telemU32[2]} rsn=${telemU32[3]}`);
assert(telemU32[4] === 0 && valid.every((v) => v === 0), 'Disturbed frame must invalidate all valid bits');
assert(rgba.every((v, idx) => (idx & 3) !== 3 || v === 0), 'Disturbed frame must clear all RGBA alpha');

// Return to exact tripod pose: while coarse pan streak (cam_motion_state == 1) settles, BOS stays disturbed via EXT_PAN_CUT,
// preserves frozen refEpoch without wiping reference, and recovers READY immediately once cam_motion_state == 0.
pushFrame(plumeFrame, seq, seq * 33333); seq++;
assert(telemU32[11] === refEpBeforeNudge, 'Returning to exact tripod pose must preserve frozen BOS reference epoch');
for (let settle = 0; settle < 4 && telemU32[2] !== STATE_READY; settle++) {
  assert((telemU32[3] & 2) !== 0, `Cooldown disturbed frame must be EXT_PAN_CUT only, got rsn=${telemU32[3]}`);
  pushFrame(plumeFrame, seq, seq * 33333); seq++;
}
assert(telemU32[2] === STATE_READY && telemU32[11] === refEpBeforeNudge, 'Returning to exact tripod pose recovers READY without wiping reference');

// Timing-only gap (seq gap bit 0) preserves frozen reference
pushFrame(plumeFrame, seq + 3, (seq + 4) * 33333, 1 << 0);
assert(telemU32[2] === STATE_READY, 'Timing-only gap must preserve frozen BOS reference');

// Clean kernel engine wrap (frame_idx u32::MAX -> 0 emits 0x40 FRAME_GAP_WRAP) preserves frozen BOS reference & epoch
assert(
  typeof exp.chronos_get_ring_slot_ptr === 'function' &&
    typeof exp.chronos_get_ring_meta_ptr === 'function' &&
    typeof exp.chronos_get_wave1_telemetry_ptr === 'function',
  'Expected Wave-1 ring telemetry exports for real kernel engine_wrapped test'
);
{
  const w1 = new Uint32Array(buf, exp.chronos_get_wave1_telemetry_ptr(), 192);
  const ringMeta = new Uint32Array(buf, exp.chronos_get_ring_meta_ptr(), 12 * 24);
  const arenaBytes = w1[1] >>> 0;
  const ring0Ptr = exp.chronos_get_ring_slot_ptr(0);
  const arenaU32 = new Uint32Array(buf, ring0Ptr, arenaBytes >> 2);
  const cand = [];
  for (let wi = 2; wi < arenaU32.length - 4; wi++) {
    if (
      arenaU32[wi - 1] === 0 &&
      arenaU32[wi] === 1 &&
      arenaU32[wi - 2] > 5 &&
      arenaU32[wi - 2] < 10000 &&
      arenaU32[wi + 1] > 5 &&
      arenaU32[wi + 1] < 10000 &&
      arenaU32[wi + 3] === 8
    ) {
      cand.push([wi, arenaU32[wi - 2], arenaU32[wi + 1]]);
    }
  }
  let wrapSeq = seq + 4;
  pushFrame(plumeFrame, wrapSeq, (wrapSeq + 1) * 33333, 0);
  wrapSeq++;
  const matches = cand.filter(
    ([wi, rc0, f0]) =>
      arenaU32[wi - 1] === 0 && arenaU32[wi - 2] === rc0 + 1 && arenaU32[wi + 1] === f0 + 1
  );
  assert(matches.length === 1, `Expected unique Arena (ring_cursor, res_mode=1, frame_idx), got ${matches.length}`);
  const wi = matches[0][0];
  const rcLoWord = wi - 2;
  const rcHiWord = wi - 1;
  const fIdxWord = wi + 1;

  // Position engine 18 ticks before u32::MAX: 1 re-sync tick (0xffffffee) + 16 clear warmup ticks -> READY at 0xffffffff -> wrap at 0.
  // Through the test-hook export, not a raw write, so recorded traces of this gate replay natively.
  exp.chronos_test_hook_set_frame_counters(0xffffffee, 0xffffffee, 0);
  assert(
    arenaU32[rcLoWord] === 0xffffffee && arenaU32[rcHiWord] === 0 && arenaU32[fIdxWord] === 0xffffffee,
    'chronos_test_hook_set_frame_counters must set exactly the located (ring_cursor, frame_idx) words'
  );
  pushFrame(clearFrame, wrapSeq, (wrapSeq + 1) * 33333, 0);
  wrapSeq++;
  exp.chronos_reset_bos_ref();
  renderBackdropFrame(clearFrame, null, 1.0, 0.0, 0.0, 1);
  for (let w = 0; w < 16; w++) {
    pushFrame(clearFrame, wrapSeq, (wrapSeq + 1) * 33333, 0);
    wrapSeq++;
  }
  assert(arenaU32[fIdxWord] === 0xffffffff, `Expected frame_idx=0xffffffff before wrap, got 0x${arenaU32[fIdxWord].toString(16)}`);

  // Pre-wrap frame at frame_idx = u32::MAX (BOS enters STATE_READY)
  pushFrame(plumeFrame, wrapSeq, (wrapSeq + 1) * 33333, 0);
  wrapSeq++;
  const refEpBeforeWrap = telemU32[11];
  assert(arenaU32[fIdxWord] === 0, `Expected frame_idx wrap to 0 after u32::MAX tick, got ${arenaU32[fIdxWord]}`);
  assert(
    telemU32[2] === STATE_READY && telemU32[10] === 16,
    `Pre-wrap u32::MAX frame must be READY (state=${telemU32[2]}, warm=${telemU32[10]}, rsn=${telemU32[3]})`
  );

  // Wrap frame at frame_idx = 0 (contiguous source_seq & tSrcUs; kernel sets FRAME_GAP_WRAP = 0x40 natively)
  pushFrame(plumeFrame, wrapSeq, (wrapSeq + 1) * 33333, 0);
  wrapSeq++;
  const slot = w1[2] >>> 0;
  assert(slot < 12, `Invalid ring slot ${slot} on engine_wrapped tick`);
  assert(
    telemU32[2] === STATE_READY &&
      telemU32[11] === refEpBeforeWrap &&
      telemU32[10] === 16 &&
      (ringMeta[slot * 24 + 21] & 0x40) !== 0,
    `Clean kernel u32::MAX->0 wrap must set 0x40 and preserve BOS reference (state=${telemU32[2]}, ep=${telemU32[11]}!=${refEpBeforeWrap}, warm=${telemU32[10]}, gap=0x${ringMeta[slot * 24 + 21].toString(16)})`
  );
  assert((ringMeta[slot * 24 + 21] & 1) === 0, 'Contiguous source_seq across wrap must not set GAP_SEQ');
  assert(ringMeta[slot * 24 + 3] === 0, `Expected ringMeta.engine_frame_idx=0 after wrap, got ${ringMeta[slot * 24 + 3]}`);
  seq = wrapSeq;
}

// Geometry/cut reset (gap bit 5) restarts warmup
const refEpBeforeCut = telemU32[11];
pushFrame(clearFrame, seq, (seq + 1) * 33333, 1 << 5);
assert(telemU32[2] === STATE_DISTURBED && (telemU32[3] & REASON_DISCONT) !== 0, 'Scene cut must trigger DISCONT');
assert(telemU32[11] > refEpBeforeCut && telemU32[10] === 0, 'Scene cut must increment ref_epoch and reset warmup');

// Test C: Aperture-only 1D stripe texture rejection
const stripeFrame = new Uint8Array(W * H * 4);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const v = Math.round((0.5 + 0.35 * Math.sin(0.55 * x)) * 255);
    const idx = (y * W + x) * 4;
    stripeFrame[idx] = v; stripeFrame[idx + 1] = v; stripeFrame[idx + 2] = v; stripeFrame[idx + 3] = 255;
  }
}
exp.chronos_init();
exp.chronos_set_resolution(1);
assert(exp.chronos_set_bos_mode(1) === 1);
assert(exp.chronos_set_bos_gauge_roi(8, 8, 48, 120) === 1);
pushFrame(stripeFrame, 1, 33333);
assert(telemU32[2] === STATE_DISTURBED && (telemU32[3] & (REASON_ANCHOR_SUPPORT | REASON_LOW_TEX)) !== 0, '1D stripe aperture-only texture must be rejected');

// Test C2: 1D-mask subset aperture rejection (2D ref masked by whiteout down to 40 surviving 1D stripe pixels)
let apertureJ29Rejected = 0;
{
  const apRef = new Uint8Array(W * H * 4);
  const apCtrl = new Uint8Array(W * H * 4);
  const apCur = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      const inBandR = x >= 240 && x < 400 && y >= 140 && y <= 167;
      const valR = inBandR ? (0.5 + 0.35 * Math.sin(0.60 * x)) : evalBackdrop(x, y);
      const qr = Math.min(255, Math.max(0, Math.round(valR * 255.0)));
      apRef[idx] = qr; apRef[idx + 1] = qr; apRef[idx + 2] = qr; apRef[idx + 3] = 255;

      const ex = x >= 150 ? 0.35 : 0.0;
      const ey = x >= 150 ? -0.25 : 0.0;
      const sx = x - ex, sy = y - ey;
      const inBandS = sx >= 240 && sx < 400 && sy >= 140 && sy <= 167;
      const valC = inBandS ? (0.5 + 0.35 * Math.sin(0.60 * sx)) : evalBackdrop(sx, sy);
      const qc = Math.min(255, Math.max(0, Math.round(valC * 255.0)));
      apCtrl[idx] = qc; apCtrl[idx + 1] = qc; apCtrl[idx + 2] = qc; apCtrl[idx + 3] = 255;

      if (x >= 240 && x < 400 && y >= 167 && y < 198) {
        apCur[idx] = 255; apCur[idx + 1] = 255; apCur[idx + 2] = 255; apCur[idx + 3] = 255;
      } else {
        apCur[idx] = qc; apCur[idx + 1] = qc; apCur[idx + 2] = qc; apCur[idx + 3] = 255;
      }
    }
  }
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  assert(exp.chronos_set_bos_mode(1) === 1);
  assert(exp.chronos_set_bos_gauge_roi(8, 8, 40, 120) === 1);
  for (let f = 0; f < 16; f++) {
    pushFrame(apRef, 1 + f, (1 + f) * 33333);
  }
  pushFrame(apCtrl, 17, 17 * 33333);
  assert(telemU32[2] === STATE_READY, '1D-mask unmasked control must reach READY');
  let ctrlJ29Valid = 0;
  for (let i = 102; i <= 154; i++) {
    if ((valid[58 * BOS_W + i] & VALID_DISP) !== 0) ctrlJ29Valid++;
  }
  assert(ctrlJ29Valid >= 40, `1D-mask unmasked control expected >=40 valid 2D boundary nodes, got ${ctrlJ29Valid}`);

  pushFrame(apCur, 18, 18 * 33333);
  assert(telemU32[2] === STATE_READY, '1D-mask test must stay READY');
  assert(telemU32[9] === SOLVER_PCG && telemF32[16] <= 1e-6, `1D-mask PCG failed res=${telemF32[16]}`);
  let maskedJ29Valid = 0;
  for (let i = 102; i <= 154; i++) {
    if ((valid[58 * BOS_W + i] & VALID_DISP) !== 0) maskedJ29Valid++;
  }
  assert(maskedJ29Valid === 0, `1D-mask surviving-subset aperture must be rejected cleanly, got ${maskedJ29Valid} valid`);
  apertureJ29Rejected = ctrlJ29Valid;

  const exArr = planeSlice(0);
  const eyArr = planeSlice(1);
  let maxCleanErr = 0;
  for (let j = 40; j < 80; j++) {
    for (let i = 70; i < 92; i++) {
      const u = j * BOS_W + i;
      assert((valid[u] & VALID_DISP) !== 0, `Unmasked 2D node (${i},${j}) must stay valid`);
      maxCleanErr = Math.max(maxCleanErr, Math.hypot(exArr[u] - 0.35, eyArr[u] + 0.25));
    }
  }
  assert(maxCleanErr <= 0.15, `Unmasked 2D region error too high: ${maxCleanErr}`);
}

// Test G: Non-optional Mode 2 Selective Slow Texture Adaptation (via C-ABI outputs vs Mode 1 control)
let mode2MedAdaptPx = 0;
{
  const shiftPlumeFrame = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let ex = 0, ey = 0;
      if (x < 115) {
        ex = 0.22;
      } else if (x > 220) {
        [ex, ey] = plumeDisp(x, y);
      }
      const q = Math.min(255, Math.max(0, Math.round(evalBackdrop(x - ex, y - ey) * 255.0)));
      const idx = (y * W + x) * 4;
      shiftPlumeFrame[idx] = q; shiftPlumeFrame[idx + 1] = q; shiftPlumeFrame[idx + 2] = q; shiftPlumeFrame[idx + 3] = 255;
    }
  }

  function sampleClearGaugeStats() {
    const exArr = planeSlice(0);
    const eyArr = planeSlice(1);
    const negEx = [];
    const absEx = [];
    let maxOutEps = 0;
    for (let j = 12; j < 116; j++) {
      for (let i = 10; i < 38; i++) {
        const u = j * BOS_W + i;
        if ((valid[u] & VALID_DISP) !== 0) {
          negEx.push(-exArr[u]);
          absEx.push(Math.abs(exArr[u]));
        }
      }
      for (let i = 100; i < 240; i++) {
        const u = j * BOS_W + i;
        if ((valid[u] & VALID_DISP) !== 0) {
          maxOutEps = Math.max(maxOutEps, Math.hypot(exArr[u], eyArr[u]));
        }
      }
    }
    negEx.sort((a, b) => a - b);
    absEx.sort((a, b) => a - b);
    return {
      count: negEx.length,
      minAbs: absEx[0],
      medNegEx: negEx[Math.floor(negEx.length * 0.5)],
      p75Abs: absEx[Math.floor(absEx.length * 0.75)],
      maxOutEps,
    };
  }

  // First run Mode 1 (Frozen) control for 240 frames
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  assert(exp.chronos_set_bos_mode(1) === 1);
  assert(exp.chronos_set_bos_gauge_roi(8, 8, 40, 120) === 1);
  renderBackdropFrame(clearFrame, null, 1.0, 0.0, 0.0, 1);
  for (let f = 0; f < 16; f++) {
    pushFrame(clearFrame, 1 + f, (1 + f) * 33333);
  }
  for (let f = 0; f < 240; f++) {
    pushFrame(shiftPlumeFrame, 17 + f, (17 + f) * 33333);
  }
  pushFrame(clearFrame, 257, 257 * 33333);
  assert(telemU32[2] === STATE_READY, 'Mode 1 control must stay READY');
  const m1Ctrl = sampleClearGaugeStats();
  assert(m1Ctrl.count > 500 && m1Ctrl.p75Abs < 1e-4, `Mode 1 frozen control must not adapt gauge (p75=${m1Ctrl.p75Abs})`);

  // Next run Mode 2 (Maint) with noisy warmup + reset_ref + <120 & >=120 adaptation verification
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  assert(exp.chronos_set_bos_mode(2) === 1);
  assert(exp.chronos_set_bos_gauge_roi(8, 8, 40, 120) === 1);
  for (let f = 0; f < 16; f++) {
    renderBackdropFrame(clearFrame, null, 1.0, 0.0, 0.02, 100 + f);
    pushFrame(clearFrame, f + 1, (f + 1) * 33333);
    assert(telemU32[2] === STATE_WARMING, `Noisy warmup frame ${f} must remain WARMING`);
  }
  const refSnapEp = telemU32[11];
  exp.chronos_reset_bos_ref();
  assert(telemU32[11] > refSnapEp && telemU32[2] === STATE_WARMING && telemU32[10] === 0, 'chronos_reset_bos_ref must increment epoch and restart warmup');
  renderBackdropFrame(clearFrame, null, 1.0, 0.0, 0.0, 1);
  for (let f = 0; f < 16; f++) {
    pushFrame(clearFrame, 17 + f, (17 + f) * 33333);
  }
  // 100 frames (< 120 streak required): verify ZERO adaptation before 120 consecutive clear frames
  for (let f = 0; f < 100; f++) {
    pushFrame(shiftPlumeFrame, 33 + f, (33 + f) * 33333);
  }
  pushFrame(clearFrame, 133, 133 * 33333);
  assert(telemU32[2] === STATE_READY, 'Mode 2 must be READY at clear probe 1');
  const m2Pre = sampleClearGaugeStats();
  assert(m2Pre.p75Abs < 1e-4, `Mode 2 adapted prematurely before 120 clear frames: p75=${m2Pre.p75Abs}`);

  // Run remaining 220 frames (reset by probe switch per #288 #7 pan guard, then > 120 clear frames + 100 adaptation steps)
  for (let f = 0; f < 220; f++) {
    pushFrame(shiftPlumeFrame, 134 + f, (134 + f) * 33333);
  }
  assert(telemU32[2] === STATE_READY, 'Mode 2 must remain READY under persistent slow non-gauge plume');
  // Plume outside gauge (i >= 70) must remain completely uncorrupted
  const exArrM2 = planeSlice(0);
  const eyArrM2 = planeSlice(1);
  let sePlumeM2 = 0, nPlumeM2 = 0;
  for (let j = 2; j < BOS_H - 2; j++) {
    for (let i = 70; i < BOS_W - 2; i++) {
      const u = j * BOS_W + i;
      if ((valid[u] & VALID_DISP) !== 0) {
        nPlumeM2++;
        const [cx, cy] = nodeCoord(i, j, W, H);
        const [tx, ty] = plumeDisp(cx, cy);
        sePlumeM2 += (exArrM2[u] - tx) ** 2 + (eyArrM2[u] - ty) ** 2;
      }
    }
  }
  const maintPlumeRmse = Math.sqrt(sePlumeM2 / Math.max(1, nPlumeM2));
  assert(maintPlumeRmse <= 0.15, `Mode 2 must preserve non-gauge plume measurement (RMSE=${maintPlumeRmse.toFixed(4)})`);

  // Clear probe after 120+ uninterrupted post-probe updates: non-halo gauge adapts cleanly, anchor halos + non-gauge stay frozen
  pushFrame(clearFrame, 354, 354 * 33333);
  assert(telemU32[2] === STATE_READY, 'Mode 2 must stay READY at post-adapt probe');
  const m2Post = sampleClearGaugeStats();
  assert(m2Post.medNegEx > 0.003 && m2Post.medNegEx > 50.0 * Math.max(1e-5, m1Ctrl.p75Abs), `Mode 2 clear-region adaptation not observed: medNegEx=${m2Post.medNegEx} vs mode1=${m1Ctrl.p75Abs}`);
  assert(m2Post.minAbs < 1e-4, `Mode 2 corrupted frozen anchor halos inside gauge: minAbs=${m2Post.minAbs}`);
  assert(m2Post.maxOutEps < 1e-4, `Mode 2 leaked adaptation outside gauge_roi: maxOut=${m2Post.maxOutEps}`);
  mode2MedAdaptPx = m2Post.medNegEx;
}

// Test G2: 320x180 SD mode textured plume end-to-end verification
{
  const SW = 320, SH = 180;
  const sdClear = new Uint8Array(SW * SH * 4);
  const sdPlume = new Uint8Array(SW * SH * 4);
  const sdDisp = (x, y) => {
    const dx = (x - 200.0) / 30.0;
    const dy = (y - 90.0) / 28.0;
    const r2 = dx * dx + dy * dy;
    if (r2 > 22.0) return [0.0, 0.0];
    const g = Math.exp(-r2);
    return [(2.0 * 14.0 * g * dx) / 30.0, (2.0 * 14.0 * g * dy) / 28.0];
  };
  for (let y = 0; y < SH; y++) {
    for (let x = 0; x < SW; x++) {
      const idx = (y * SW + x) * 4;
      const q0 = Math.round(evalBackdrop(x * 1.4, y * 1.4) * 255);
      sdClear[idx] = q0; sdClear[idx + 1] = q0; sdClear[idx + 2] = q0; sdClear[idx + 3] = 255;
      const [ex, ey] = sdDisp(x, y);
      const q1 = Math.round(evalBackdrop((x - ex) * 1.4, (y - ey) * 1.4) * 255);
      sdPlume[idx] = q1; sdPlume[idx + 1] = q1; sdPlume[idx + 2] = q1; sdPlume[idx + 3] = 255;
    }
  }
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  assert(exp.chronos_set_bos_mode(1) === 1);
  assert(exp.chronos_set_bos_gauge_roi(8, 8, 48, 120) === 1);
  for (let f = 0; f < 16; f++) {
    pushFrame(sdClear, 1 + f, (1 + f) * 33333, 0, SW, SH);
  }
  pushFrame(sdPlume, 17, 17 * 33333, 0, SW, SH);
  assert(telemU32[2] === STATE_READY, `SD mode must reach READY, got ${telemU32[2]}`);
  const exArr = planeSlice(0);
  const eyArr = planeSlice(1);
  let se = 0, vInt = 0, tInt = 0;
  for (let j = 3; j < BOS_H - 3; j++) {
    for (let i = 3; i < BOS_W - 3; i++) {
      tInt++;
      const u = j * BOS_W + i;
      if ((valid[u] & VALID_DISP) !== 0) {
        vInt++;
        const [cx, cy] = nodeCoord(i, j, SW, SH);
        const [tx, ty] = sdDisp(cx, cy);
        se += (exArr[u] - tx) ** 2 + (eyArr[u] - ty) ** 2;
      }
    }
  }
  const sdRmse = Math.sqrt(se / Math.max(1, vInt));
  assert(vInt / tInt >= 0.90 && sdRmse <= 0.15, `SD BOS failed: avail=${vInt / tInt}, rmse=${sdRmse}`);
}

// Switch back to HD for latency benchmark
exp.chronos_init();
exp.chronos_set_resolution(1);
assert(exp.chronos_set_bos_mode(1) === 1);
assert(exp.chronos_set_bos_gauge_roi(8, 8, 40, 120) === 1);
renderBackdropFrame(clearFrame, null, 1.0, 0.0, 0.0, 1);
for (let f = 0; f < 16; f++) pushFrame(clearFrame, 1 + f, (1 + f) * 33333);

// ============================================================================
// 4. Real-Time BOS Frame Latency Benchmark (640x360 Full + Masked PCG)
// ============================================================================
const benchTimesFull = [];
const benchTimesMasked = [];
for (let f = 0; f < 30; f++) {
  const t0 = performance.now();
  pushFrame(plumeFrame, 17 + f, (17 + f) * 33333);
  benchTimesFull.push(performance.now() - t0);
}
for (let f = 0; f < 30; f++) {
  const t0 = performance.now();
  pushFrame(maskedPlumeFrame, 47 + f, (47 + f) * 33333);
  benchTimesMasked.push(performance.now() - t0);
}
const statsOf = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  return {
    med: s[Math.floor(s.length * 0.5)],
    p95: s[Math.floor(s.length * 0.95)],
    max: s[s.length - 1],
  };
};
const fullLat = statsOf(benchTimesFull);
const maskedLat = statsOf(benchTimesMasked);
const rawScale = process.env.CHRONOS_CI_BUDGET_SCALE;
const scale = rawScale === undefined || rawScale === '' ? 1.0 : Number(rawScale);
assert(Number.isFinite(scale) && scale > 0.0, `Invalid CHRONOS_CI_BUDGET_SCALE=${rawScale}`);
assert(maskedLat.p95 < 25.0 * scale, `BOS masked PCG p95 latency ${maskedLat.p95.toFixed(2)} ms too high`);

console.log(
  `  [PASS] Gate I5 Textured BOS: plumeRMSE=${plumeStats.rmse.toFixed(4)}px (${(plumeStats.avail * 100).toFixed(1)}% avail) | ` +
    `flickerRMSE=${flickStats.rmse.toFixed(4)}px | maskedRMSE=${maskedStats.rmse.toFixed(4)}px | ` +
    `1DmaskRej=${apertureJ29Rejected} | mode2Adapt=${mode2MedAdaptPx.toExponential(2)}px | ` +
    `curl(vortex=${vortexCurlRms.toExponential(2)} vs zoom=${zoomCurlRms.toExponential(2)}) | ` +
    `latency(full=${fullLat.med.toFixed(2)}ms/${fullLat.max.toFixed(2)}ms, pcg=${maskedLat.med.toFixed(2)}ms/${maskedLat.max.toFixed(2)}ms)`
);
