#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I5b: Organ 5 Occluded Two-Body Conservation &
// Four-Branch Hypothesis Scoring (H0_keep, H0_swap, H1_keep, H1_swap)
//
// Usage:
//   node scripts/gate_i5b_conservation.mjs [--wasm=<path>]
// ============================================================================

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const wasmArg = process.argv.slice(2).find((a) => a.startsWith('--wasm='));
const WASM_PATH = wasmArg ? path.resolve(wasmArg.slice('--wasm='.length)) : DEFAULT_WASM;

const OCCL_STATUS_NONE = 0;
const OCCL_STATUS_UNIQUE = 1;
const OCCL_STATUS_AMBIGUOUS = 2;
const OCCL_STATUS_VIOLATED = 3;

const HYP_PASSTHROUGH_KEEP = 1 << 0;
const HYP_PASSTHROUGH_SWAP = 1 << 1;
const HYP_COLLISION_KEEP = 1 << 2;
const HYP_COLLISION_SWAP = 1 << 3;

const STAMP_VALID = 1 << 0;
const STAMP_HELD = 1 << 1;

const WAVE2_TELEM_LEN = 64;
const PROD_W2_CONSERVATION_WORD = 32; // WAVE2_PRODUCT_STAMPS_START (16) + 2 * STAMP_STRIDE (8)
const CONSERVATION_LEN = 32;

const W = 320;
const H = 180;
const BYTES = W * H * 4;

function assertNear(actual, expected, tol, label) {
  assert(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tol,
    `${label}: expected ${expected} +/- ${tol}, got ${actual}`
  );
}

// ----------------------------------------------------------------------------
// 0. Source & Script Hygiene Audit (zero em dashes, zero banned codenames)
// ----------------------------------------------------------------------------
{
  const selfText = fs.readFileSync(fileURLToPath(import.meta.url), 'utf8');
  const physText = fs.readFileSync(path.join(ROOT, 'src', 'physics_id.rs'), 'utf8');
  const bannedRe = new RegExp(
    `(${['CHRONOS', 'IMPOSSIBLE'].join('-')}|${['CHRONOS', '1000'].join('-')})`
  );
  for (const [label, txt] of [
    ['gate_i5b_conservation.mjs', selfText],
    ['src/physics_id.rs', physText],
  ]) {
    assert(!txt.includes('\u2014'), `${label}: contains forbidden em dash`);
    assert(!bannedRe.test(txt), `${label}: contains forbidden internal codename`);
  }
}

assert(fs.existsSync(WASM_PATH), `Wasm binary not found: ${WASM_PATH}`);
const wasmBytes = fs.readFileSync(WASM_PATH);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const exp = instance.exports;
const mem = exp.memory;

const REQUIRED_EXPORTS = [
  'chronos_init',
  'chronos_set_resolution',
  'chronos_set_params',
  'chronos_get_ingest_rgba_ptr',
  'chronos_set_ingest_meta',
  'chronos_tick',
  'chronos_get_telemetry_ptr',
  'chronos_get_slot_meta_ptr',
  'chronos_get_wave2_telemetry_ptr',
  'chronos_get_conservation_telemetry_ptr',
  'chronos_get_conservation_input_ptr',
  'chronos_eval_occluded_pair',
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
const consTelemPtr = exp.chronos_get_conservation_telemetry_ptr();
const consInputPtr = exp.chronos_get_conservation_input_ptr();
const wave2Ptr = exp.chronos_get_wave2_telemetry_ptr();
const telemPtr = exp.chronos_get_telemetry_ptr();
const slotMetaPtr = exp.chronos_get_slot_meta_ptr();
const ingestPtr = exp.chronos_get_ingest_rgba_ptr();

const regions = [
  ['conservation_telem', consTelemPtr, CONSERVATION_LEN * 4],
  ['conservation_input', consInputPtr, CONSERVATION_LEN * 4],
  ['wave2_telem', wave2Ptr, WAVE2_TELEM_LEN * 4],
  ['telemetry', telemPtr, 1568 * 4],
  ['slot_meta', slotMetaPtr, 16 * 8 * 4],
  ['ingest_rgba', ingestPtr, 640 * 360 * 4],
];
for (const [name, ptr, bytes] of regions) {
  assert((ptr & 3) === 0 && ptr > 0 && ptr + bytes <= buf.byteLength, `Invalid region ${name}`);
}
regions.sort((a, b) => a[1] - b[1]);
for (let i = 1; i < regions.length; i++) {
  assert(
    regions[i - 1][1] + regions[i - 1][2] <= regions[i][1],
    `Overlap between ${regions[i - 1][0]} and ${regions[i][0]}`
  );
}

const consTelem = new Float32Array(buf, consTelemPtr, CONSERVATION_LEN);
const consInput = new Float32Array(buf, consInputPtr, CONSERVATION_LEN);
const wave2Telem = new Uint32Array(buf, wave2Ptr, WAVE2_TELEM_LEN);
const slotMeta = new Float32Array(buf, slotMetaPtr, 16 * 8);
const mainTelem = new Float32Array(buf, telemPtr, 1568);
const ingestViewCache = new Map();
function writeIngestRgba(frameBuf) {
  const ptr = exp.chronos_get_ingest_rgba_ptr();
  let view = ingestViewCache.get(ptr);
  if (!view) {
    view = new Uint8Array(buf, ptr, frameBuf.byteLength);
    ingestViewCache.set(ptr, view);
  }
  view.set(frameBuf);
}

function stageEval(vals) {
  consInput.fill(0.0);
  for (let i = 0; i < vals.length; i++) {
    consInput[i] = vals[i];
  }
  const st = exp.chronos_eval_occluded_pair();
  assert(
    Math.round(consTelem[0]) === st,
    `Returned status ${st} mismatches conservation_telem[0]=${consTelem[0]}`
  );
  for (let k = 24; k < 32; k++) {
    assert(consTelem[k] === 0.0, `Expected reserved conservation_telem[${k}] == 0, got ${consTelem[k]}`);
  }
  assert(
    (wave2Telem[PROD_W2_CONSERVATION_WORD] & STAMP_VALID) === 0,
    'chronos_eval_occluded_pair() must invalidate PROD_W2_CONSERVATION live stamp'
  );
  return {
    status: st,
    mask: Math.round(consTelem[1]),
    count: Math.round(consTelem[2]),
    netImpulse: consTelem[3],
    envBound: consTelem[4],
    momResid: consTelem[5],
    angMomResid: consTelem[6],
    deltaKeRatio: consTelem[7],
    h0KeepChi2: consTelem[8],
    h0SwapChi2: consTelem[9],
    h1KeepChi2: consTelem[10],
    h1SwapChi2: consTelem[11],
    tau: consTelem[12],
    e: consTelem[13],
    tidA: Math.round(consTelem[14]),
    tidB: Math.round(consTelem[15]),
    dt: consTelem[16],
    eventFrame: Math.round(consTelem[17]),
    h0KeepImp: consTelem[18],
    h0SwapImp: consTelem[19],
    h1KeepImp: consTelem[20],
    h1SwapImp: consTelem[21],
    gy: consTelem[22],
    massRatio: consTelem[23],
  };
}

function makeInput({
  p1,
  p2,
  a,
  b,
  dt,
  gy = 0,
  dragMax = 0,
  spinMax = 0,
  perspMax = 0,
  sigmaPos = 0,
  sigmaVel = 0,
  tidA = 1,
  tidB = 2,
  eventFrame = 50,
  massKnown = 1,
}) {
  return [
    p1[0], p1[1], p1[2], p1[3], p1[4], p1[5],
    p2[0], p2[1], p2[2], p2[3], p2[4], p2[5],
    a[0], a[1], a[2], a[3],
    b[0], b[1], b[2], b[3],
    dt, gy, dragMax, spinMax, perspMax, sigmaPos, sigmaVel,
    tidA, tidB, eventFrame, massKnown, 0,
  ];
}

// ----------------------------------------------------------------------------
// 0b. Fail-Closed Input Validation & Non-Circular Unit-Mass Prior Check
// ----------------------------------------------------------------------------
exp.chronos_init();

for (const bad of [
  makeInput({ p1: [10, 10, 50, 0, 1, 5], p2: [50, 10, -50, 0, 1, 5], a: [10, 10, -50, 0], b: [50, 10, 50, 0], dt: 0 }),
  makeInput({ p1: [10, 10, 50, 0, 1, 5], p2: [50, 10, -50, 0, 1, 5], a: [10, 10, -50, 0], b: [50, 10, 50, 0], dt: -0.2 }),
  makeInput({ p1: [NaN, 10, 50, 0, 1, 5], p2: [50, 10, -50, 0, 1, 5], a: [10, 10, -50, 0], b: [50, 10, 50, 0], dt: 0.2 }),
  makeInput({ p1: [10, 10, 50, 0, 0, 5], p2: [50, 10, -50, 0, 1, 5], a: [10, 10, -50, 0], b: [50, 10, 50, 0], dt: 0.2, massKnown: 1 }),
  makeInput({ p1: [10, 10, 50, 0, 1, 5], p2: [50, 10, -50, 0, 1, 5], a: [10, 10, -50, 0], b: [50, 10, 50, 0], dt: 0.2, sigmaVel: -1 }),
]) {
  const res = stageEval(bad);
  assert.equal(res.status, OCCL_STATUS_NONE, 'Invalid input must yield OCCL_STATUS_NONE');
  for (let k = 0; k < 32; k++) {
    assert.equal(consTelem[k], 0.0, `Expected zeroed telemetry[${k}] on OCCL_STATUS_NONE`);
  }
}

// Non-circular mass_known_flag=0 test: velocity jumps matching m2/m1=3 MUST fail under declared 1.0:1.0 prior
{
  const nonCirc = stageEval(
    makeInput({
      p1: [100, 90, 150, 0, 1.0, 6.0],
      p2: [160, 90, -50, 0, 3.0, 6.0],
      a: [100, 90, -150, 0],
      b: [160, 90, 50, 0],
      dt: 0.30,
      sigmaPos: 1.0,
      sigmaVel: 3.0,
      massKnown: 0.0,
    })
  );
  assert.equal(nonCirc.status, OCCL_STATUS_VIOLATED, 'mass_known_flag=0 must not back-solve m2/m1=3 from dv');
  assertNear(nonCirc.massRatio, 1.0, 1e-6, 'mass_known_flag=0 declared unit mass ratio');
  assert(nonCirc.netImpulse > nonCirc.envBound, 'mass_known_flag=0 unequal-dv impulse must exceed unit-mass bound');
}

// ----------------------------------------------------------------------------
// Case 1: Equal-Mass Head-On Elastic Exchange Behind Occluder (Ambiguous)
// ----------------------------------------------------------------------------
const case1Point = stageEval(
  makeInput({
    p1: [100, 90, 150, 0, 1.0, 0.0],
    p2: [160, 90, -150, 0, 1.0, 0.0],
    a: [100, 90, -150, 0],
    b: [160, 90, 150, 0],
    dt: 0.40,
    dragMax: 0.02,
    spinMax: 1.0,
    perspMax: 1.0,
    sigmaPos: 1.0,
    sigmaVel: 3.0,
    tidA: 11,
    tidB: 12,
    eventFrame: 42,
    massKnown: 1.0,
  })
);
assert.equal(case1Point.status, OCCL_STATUS_AMBIGUOUS, 'Case 1 (point exchange) must be OCCL_STATUS_AMBIGUOUS');
assert(case1Point.count >= 2, `Case 1 expected admissible_count >= 2, got ${case1Point.count}`);
assert(
  (case1Point.mask & HYP_PASSTHROUGH_SWAP) !== 0 && (case1Point.mask & HYP_COLLISION_KEEP) !== 0,
  `Case 1 expected BOTH HYP_PASSTHROUGH_SWAP and HYP_COLLISION_KEEP in mask=${case1Point.mask}`
);
assertNear(case1Point.netImpulse, 0.0, 1e-3, 'Case 1 netImpulse');
assertNear(case1Point.momResid, 0.0, 1e-4, 'Case 1 momResid');
assertNear(case1Point.angMomResid, 0.0, 1e-4, 'Case 1 angMomResid');
assertNear(case1Point.deltaKeRatio, 0.0, 1e-4, 'Case 1 deltaKeRatio');
assertNear(case1Point.e, 1.0, 1e-3, 'Case 1 restitution');
assertNear(case1Point.tau, 0.20, 1e-3, 'Case 1 tau');

// H1 rejections: severe interpenetration at fitted tau and closing post-impact (e = -0.01) must fail H1
{
  const interpen = stageEval(
    makeInput({
      p1: [-10, 0, 10, 0, 1.0, 6.0],
      p2: [10, 0, -10, 0, 1.0, 6.0],
      a: [-5, 0, -5, 0],
      b: [5, 0, 5, 0],
      dt: 2.0,
      massKnown: 1.0,
    })
  );
  assert.equal(interpen.status, OCCL_STATUS_VIOLATED, 'Deep interpenetration with r=6 must be VIOLATED');
  assert.equal(interpen.mask & (HYP_COLLISION_KEEP | HYP_COLLISION_SWAP), 0, 'Deep interpenetration must reject H1');

  const stillClosing = stageEval(
    makeInput({
      p1: [-30, 0, 30, 0, 1.0, 0.0],
      p2: [10, 0, -10, 0, 3.0, 0.0],
      a: [0.3, 0, 0.3, 0],
      b: [-0.1, 0, -0.1, 0],
      dt: 2.0,
      massKnown: 1.0,
    })
  );
  assert.equal(stillClosing.status, OCCL_STATUS_VIOLATED, 'Negative restitution e=-0.01 must be VIOLATED');
  assert.equal(stillClosing.mask & (HYP_COLLISION_KEEP | HYP_COLLISION_SWAP), 0, 'Still-closing exit must reject H1');

  const extremeMass = stageEval(
    makeInput({
      p1: [10, 10, 0, 0, 1e-37, 0.0],
      p2: [50, 10, 0, 0, 1e37, 0.0],
      a: [10, 10, 0, 0],
      b: [50, 10, 0, 0],
      dt: 1.0,
      massKnown: 1.0,
    })
  );
  assert.equal(extremeMass.status, OCCL_STATUS_NONE, 'Unrepresentable 1e37/1e-37 mass ratio must fail closed to NONE');
  for (let k = 0; k < 32; k++) {
    assert.equal(consTelem[k], 0.0, `Expected zeroed telemetry[${k}] on extreme mass overflow`);
  }
}

// Hard-sphere surface-contact bounce within position uncertainty also keeps both branches admissible
const case1Disc = stageEval(
  makeInput({
    p1: [100, 90, 150, 0, 1.0, 6.0],
    p2: [160, 90, -150, 0, 1.0, 6.0],
    a: [88, 90, -150, 0],
    b: [172, 90, 150, 0],
    dt: 0.40,
    sigmaPos: 3.0,
    sigmaVel: 15.0,
    tidA: 11,
    tidB: 12,
    eventFrame: 43,
    massKnown: 1.0,
  })
);
assert.equal(case1Disc.status, OCCL_STATUS_AMBIGUOUS, 'Case 1 (disc contact w/ sigma) must be OCCL_STATUS_AMBIGUOUS');
assert(
  (case1Disc.mask & HYP_PASSTHROUGH_SWAP) !== 0 && (case1Disc.mask & HYP_COLLISION_KEEP) !== 0,
  `Case 1 disc expected BOTH HYP_PASSTHROUGH_SWAP and HYP_COLLISION_KEEP, got mask=${case1Disc.mask}`
);

// ----------------------------------------------------------------------------
// Case 2: Distinct-Mass (m2/m1 = 3.0) Inelastic (e_n = 0.70) Oblique Collision
// ----------------------------------------------------------------------------
let case2;
{
  const m1 = 1.0, r1 = 6.0;
  const m2 = 3.0, r2 = 8.0;
  const eTrue = 0.70;
  const gy = 240.0;
  const dt = 0.30;
  const tau = 0.14;
  const dtRem = dt - tau;

  const theta = 0.35;
  const nx = -Math.cos(theta);
  const ny = -Math.sin(theta);

  const x1c = 148.0, y1c = 92.0;
  const x2c = x1c - (r1 + r2) * nx;
  const y2c = y1c - (r1 + r2) * ny;

  const vx1In = 140.0, vy1In = 22.0;
  const vx2In = -45.0, vy2In = -28.0;

  const x1Pre = x1c - vx1In * tau;
  const y1Pre = y1c - vy1In * tau + 0.5 * gy * tau * tau;
  const vx1Pre = vx1In;
  const vy1Pre = vy1In - gy * tau;

  const x2Pre = x2c - vx2In * tau;
  const y2Pre = y2c - vy2In * tau + 0.5 * gy * tau * tau;
  const vx2Pre = vx2In;
  const vy2Pre = vy2In - gy * tau;

  const vrelInN = (vx1In - vx2In) * nx + (vy1In - vy2In) * ny;
  assert(vrelInN < 0, 'Case 2 setup must be approaching');
  const jImp = (-(1.0 + eTrue) * vrelInN) / (1.0 / m1 + 1.0 / m2);

  const vx1OutTau = vx1In + (jImp / m1) * nx;
  const vy1OutTau = vy1In + (jImp / m1) * ny;
  const vx2OutTau = vx2In - (jImp / m2) * nx;
  const vy2OutTau = vy2In - (jImp / m2) * ny;

  const x1Post = x1c + vx1OutTau * dtRem;
  const y1Post = y1c + vy1OutTau * dtRem + 0.5 * gy * dtRem * dtRem;
  const vx1Post = vx1OutTau;
  const vy1Post = vy1OutTau + gy * dtRem;

  const x2Post = x2c + vx2OutTau * dtRem;
  const y2Post = y2c + vy2OutTau * dtRem + 0.5 * gy * dtRem * dtRem;
  const vx2Post = vx2OutTau;
  const vy2Post = vy2OutTau + gy * dtRem;

  case2 = stageEval(
    makeInput({
      p1: [x1Pre, y1Pre, vx1Pre, vy1Pre, m1, r1],
      p2: [x2Pre, y2Pre, vx2Pre, vy2Pre, m2, r2],
      a: [x1Post, y1Post, vx1Post, vy1Post],
      b: [x2Post, y2Post, vx2Post, vy2Post],
      dt,
      gy,
      dragMax: 0.02,
      spinMax: 1.0,
      perspMax: 1.0,
      sigmaPos: 0.5,
      sigmaVel: 2.0,
      tidA: 21,
      tidB: 22,
      eventFrame: 68,
      massKnown: 1.0,
    })
  );
  assert.equal(case2.status, OCCL_STATUS_UNIQUE, 'Case 2 must be OCCL_STATUS_UNIQUE');
  assert.equal(case2.count, 1, `Case 2 expected admissible_count=1, got ${case2.count}`);
  assert.equal(case2.mask, HYP_COLLISION_KEEP, `Case 2 expected mask=${HYP_COLLISION_KEEP}, got ${case2.mask}`);
  assertNear(case2.tau, tau, 2e-3, 'Case 2 est_tau_sec');
  assertNear(case2.e, eTrue, 2e-3, 'Case 2 est_restitution_e');
  assertNear(case2.massRatio, 3.0, 1e-5, 'Case 2 mass_ratio');
  assert(case2.deltaKeRatio < -0.05, `Case 2 expected dissipative deltaKeRatio < -0.05, got ${case2.deltaKeRatio}`);
  assertNear(case2.momResid, 0.0, 1e-4, 'Case 2 momResid');
  assertNear(case2.angMomResid, 0.0, 1e-4, 'Case 2 angMomResid');
  assert(
    case2.h0KeepChi2 > 9.0 && case2.h0SwapChi2 > 9.0 && case2.h1SwapChi2 > 9.0,
    `Case 2 expected rejected branches chi2 > 9, got h0K=${case2.h0KeepChi2}, h0S=${case2.h0SwapChi2}, h1S=${case2.h1SwapChi2}`
  );
}

// ----------------------------------------------------------------------------
// Case 3: Non-Conservative External Impulse + Positive Energy Gain Violation
// ----------------------------------------------------------------------------
const case3 = stageEval(
  makeInput({
    p1: [110, 85, 110, -10, 1.0, 6.0],
    p2: [185, 95, -90, 5, 2.0, 7.0],
    a: [165, 65, 230, -70],
    b: [148, 102, -40, 20],
    dt: 0.30,
    gy: 180.0,
    dragMax: 0.04,
    spinMax: 2.0,
    perspMax: 2.0,
    sigmaPos: 1.0,
    sigmaVel: 4.0,
    tidA: 31,
    tidB: 32,
    eventFrame: 88,
    massKnown: 1.0,
  })
);
assert.equal(case3.status, OCCL_STATUS_VIOLATED, 'Case 3 must be OCCL_STATUS_VIOLATED');
assert.equal(case3.count, 0, `Case 3 expected admissible_count=0, got ${case3.count}`);
assert.equal(case3.mask, 0, `Case 3 expected ambiguity_mask=0, got ${case3.mask}`);
assert(
  case3.netImpulse > case3.envBound,
  `Case 3 expected netImpulse (${case3.netImpulse}) > envBound (${case3.envBound})`
);
assert(case3.deltaKeRatio > 0.05, `Case 3 expected positive deltaKeRatio, got ${case3.deltaKeRatio}`);

// Symmetric zero-net-impulse super-elastic energy gain (e_n > 1, dK > eps_K) must also violate
const case3SymGain = stageEval(
  makeInput({
    p1: [100, 90, 120, 0, 1.0, 6.0],
    p2: [160, 90, -120, 0, 1.0, 6.0],
    a: [85, 90, -210, 0],
    b: [175, 90, 210, 0],
    dt: 0.35,
    dragMax: 0.02,
    spinMax: 1.0,
    perspMax: 1.0,
    sigmaPos: 1.0,
    sigmaVel: 3.0,
    massKnown: 1.0,
  })
);
assert.equal(case3SymGain.status, OCCL_STATUS_VIOLATED, 'Case 3b symmetric energy gain must be OCCL_STATUS_VIOLATED');
assert.equal(case3SymGain.count, 0, 'Case 3b expected admissible_count=0');

// ----------------------------------------------------------------------------
// Case 4: Ballistic Crossing Under Gravity + Bounded Drag Without Contact
// ----------------------------------------------------------------------------
let case4;
{
  const gy = 320.0;
  const beta = 0.18;
  const dragMax = 0.25;
  const dt = 0.30;
  const steps = 600;
  const h = dt / steps;

  function integrateDrag(x0, y0, vx0, vy0) {
    let x = x0, y = y0, vx = vx0, vy = vy0;
    const deriv = (vxk, vyk) => [vxk, vyk, -beta * vxk, gy - beta * vyk];
    for (let s = 0; s < steps; s++) {
      const [dx1, dy1, dvx1, dvy1] = deriv(vx, vy);
      const [dx2, dy2, dvx2, dvy2] = deriv(vx + 0.5 * h * dvx1, vy + 0.5 * h * dvy1);
      const [dx3, dy3, dvx3, dvy3] = deriv(vx + 0.5 * h * dvx2, vy + 0.5 * h * dvy2);
      const [dx4, dy4, dvx4, dvy4] = deriv(vx + h * dvx3, vy + h * dvy3);
      x += (h / 6) * (dx1 + 2 * dx2 + 2 * dx3 + dx4);
      y += (h / 6) * (dy1 + 2 * dy2 + 2 * dy3 + dy4);
      vx += (h / 6) * (dvx1 + 2 * dvx2 + 2 * dvx3 + dvx4);
      vy += (h / 6) * (dvy1 + 2 * dvy2 + 2 * dvy3 + dvy4);
    }
    return [x, y, vx, vy];
  }

  const p1 = [80.0, 70.0, 200.0, -35.0, 1.2, 5.0];
  const p2 = [195.0, 102.0, -190.0, -25.0, 0.8, 5.0];
  const post1 = integrateDrag(p1[0], p1[1], p1[2], p1[3]);
  const post2 = integrateDrag(p2[0], p2[1], p2[2], p2[3]);

  case4 = stageEval(
    makeInput({
      p1,
      p2,
      a: post1,
      b: post2,
      dt,
      gy,
      dragMax,
      spinMax: 1.0,
      perspMax: 1.0,
      sigmaPos: 0.5,
      sigmaVel: 2.0,
      tidA: 41,
      tidB: 42,
      eventFrame: 105,
      massKnown: 1.0,
    })
  );
  assert.equal(case4.status, OCCL_STATUS_UNIQUE, 'Case 4 must be OCCL_STATUS_UNIQUE');
  assert.equal(case4.count, 1, `Case 4 expected admissible_count=1, got ${case4.count}`);
  assert.equal(case4.mask, HYP_PASSTHROUGH_KEEP, `Case 4 expected HYP_PASSTHROUGH_KEEP, got mask=${case4.mask}`);
  assert(
    case4.netImpulse <= case4.envBound,
    `Case 4 expected netImpulse (${case4.netImpulse}) <= envBound (${case4.envBound})`
  );
  assert(case4.deltaKeRatio <= 0.0, `Case 4 expected dissipative drag deltaKeRatio <= 0, got ${case4.deltaKeRatio}`);
}

// ----------------------------------------------------------------------------
// Case 5: Live Wasm Frame-Loop Co-Occlusion Test (lab_mode = 2)
// ----------------------------------------------------------------------------
function buildBgWithCentralBar(barY0 = 14, barY1 = 166) {
  const out = new Uint8Array(BYTES);
  const gStep = 24;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      const inBar = x >= 136 && x < 184 && y >= barY0 && y < barY1;
      if (inBar) {
        const v = (x === 136 || x === 183) ? 58 : 34 + ((y >> 3) & 3);
        out[idx] = v;
        out[idx + 1] = v + 5;
        out[idx + 2] = v + 12;
        out[idx + 3] = 255;
      } else {
        const grid = (x % gStep === 0 || y % gStep === 0) ? 14 : 0;
        const base = 18 + ((x * 7 + y * 11) >> 6) + grid;
        out[idx] = base;
        out[idx + 1] = base + 4;
        out[idx + 2] = base + 10;
        out[idx + 3] = 255;
      }
    }
  }
  return out;
}

function paintBarOverlay(bufRgba, barY0 = 14, barY1 = 166) {
  for (let y = barY0; y < barY1; y++) {
    for (let x = 136; x < 184; x++) {
      const idx = (y * W + x) * 4;
      const v = (x === 136 || x === 183) ? 58 : 34 + ((y >> 3) & 3);
      bufRgba[idx] = v;
      bufRgba[idx + 1] = v + 5;
      bufRgba[idx + 2] = v + 12;
      bufRgba[idx + 3] = 255;
    }
  }
}

function stampDisc(bufRgba, cx, cy, rad, r, g, b, cr, cg, cb) {
  const outerR2 = rad * rad;
  const innerR2 = (0.48 * rad) * (0.48 * rad);
  const x0 = Math.max(0, Math.floor(cx - rad - 1));
  const x1 = Math.min(W - 1, Math.ceil(cx + rad + 1));
  const y0 = Math.max(0, Math.floor(cy - rad - 1));
  const y1 = Math.min(H - 1, Math.ceil(cy + rad + 1));
  for (let py = y0; py <= y1; py++) {
    for (let px = x0; px <= x1; px++) {
      const r2 = (px - cx) * (px - cx) + (py - cy) * (py - cy);
      if (r2 <= outerR2) {
        const idx = (py * W + px) * 4;
        const core = r2 <= innerR2;
        bufRgba[idx] = core ? cr : r;
        bufRgba[idx + 1] = core ? cg : g;
        bufRgba[idx + 2] = core ? cb : b;
        bufRgba[idx + 3] = 255;
      }
    }
  }
}

function pushLiveFrame(frameBuf, seq, ptsUs) {
  writeIngestRgba(frameBuf);
  const lo = Number(BigInt(ptsUs) & 0xffff_ffffn);
  const hi = Number(BigInt(ptsUs) >> 32n);
  const ok = exp.chronos_set_ingest_meta(
    1, seq, lo, hi, -1.0, 0x12, W, H, 0, 0, W, H, 0, 0, 0, 0, W, H, 0, 1
  );
  assert.equal(ok, 1, `chronos_set_ingest_meta rejected seq=${seq}`);
  exp.chronos_tick(1 / 60);
}

function runLiveCoOcclusionScene({
  boostA = 1.0,
  boostB = 1.0,
  barY0 = 14,
  barY1 = 166,
  y0 = 90.0,
  y1 = 90.0,
  noBounce = false,
  skipSeqAt50 = false,
  expectNoValidStamp = false,
}) {
  const bg = buildBgWithCentralBar(barY0, barY1);
  const frameBuf = new Uint8Array(BYTES);

  exp.chronos_init();
  exp.chronos_set_resolution(0);
  exp.chronos_set_params(8, 0.035, 1.45, 25.0, -Math.PI * 0.5, 2);

  let seq = 1;
  const dtUs = 16667n;
  let ptsUs = 1_000_000n;

  // 12-frame static warmup so background ECA locks the central occluding bar cleanly
  for (let w = 0; w < 12; w++) {
    pushLiveFrame(bg, seq++, ptsUs);
    ptsUs += dtUs;
  }

  let x0 = 40.0;
  let x1 = 280.0;
  let vx0 = 2.6;
  let vx1 = -2.6;
  let bounced = false;

  let maxCoastActive = 0;
  let sawBothCoasting = false;
  let stampBeforeEmergenceValid = false;
  let liveStampFrame = -1;
  let liveStampSnapshot = null;
  let sawHeldAfterFresh = false;

  for (let f = 0; f < 90; f++) {
    x0 += vx0;
    x1 += vx1;
    if (!noBounce && !bounced && x1 - x0 <= 11.0) {
      bounced = true;
      const tmp = vx0;
      vx0 = vx1 * boostA;
      vx1 = tmp * boostB;
    }

    frameBuf.set(bg);
    stampDisc(frameBuf, x0, y0, 6.2, 248, 58, 68, 255, 230, 110);
    stampDisc(frameBuf, x1, y1, 6.2, 30, 225, 250, 15, 45, 85);
    paintBarOverlay(frameBuf, barY0, barY1);

    if (skipSeqAt50 && f === 50) {
      seq += 1;
    }
    pushLiveFrame(frameBuf, seq++, ptsUs);
    ptsUs += dtUs;

    let coastingCount = 0;
    let activeConfirmed = 0;
    for (let m = 0; m < 16; m++) {
      const tb = 32 + m * 56;
      if (mainTelem[tb] > 0.5 && Math.round(mainTelem[tb + 1]) > 0) {
        activeConfirmed++;
        const coast = Math.round(slotMeta[m * 8 + 1]);
        if (coast > 0) {
          coastingCount++;
          if (coast > maxCoastActive) maxCoastActive = coast;
        }
      }
    }
    if (coastingCount >= 2) {
      sawBothCoasting = true;
    }

    const w2Flags = wave2Telem[PROD_W2_CONSERVATION_WORD];
    const w2Valid = (w2Flags & STAMP_VALID) !== 0;
    const w2Held = (w2Flags & STAMP_HELD) !== 0;

    if ((!sawBothCoasting || coastingCount > 0) && w2Valid) {
      stampBeforeEmergenceValid = true;
    }

    if (liveStampFrame < 0 && w2Valid) {
      assert(!w2Held, 'First live PROD_W2_CONSERVATION emission must be fresh (STAMP_HELD == 0)');
      liveStampFrame = f;
      liveStampSnapshot = {
        status: Math.round(consTelem[0]),
        mask: Math.round(consTelem[1]),
        count: Math.round(consTelem[2]),
        netImpulse: consTelem[3],
        envBound: consTelem[4],
        deltaKeRatio: consTelem[7],
        tidA: Math.round(consTelem[14]),
        tidB: Math.round(consTelem[15]),
        dt: consTelem[16],
        massRatio: consTelem[23],
      };
    } else if (liveStampFrame >= 0 && f === liveStampFrame + 1) {
      assert(w2Valid && w2Held, 'Frame following live conservation event must hold STAMP_VALID | STAMP_HELD');
      sawHeldAfterFresh = true;
    }

    if (!expectNoValidStamp && activeConfirmed >= 2 && liveStampFrame >= 0 && sawHeldAfterFresh && f >= liveStampFrame + 3) {
      break;
    }
  }

  if (expectNoValidStamp) {
    assert.equal(liveStampFrame, -1, 'Expected zero PROD_W2_CONSERVATION emissions under non-co-occluding / epoch-reset scenario');
    return { maxCoastActive, liveStampFrame };
  }

  assert(!stampBeforeEmergenceValid, 'PROD_W2_CONSERVATION must not become valid before co-occlusion occurs or while coasting');
  assert(sawBothCoasting && maxCoastActive >= 3, `Expected both discs coasting behind bar (maxCoast=${maxCoastActive})`);
  assert(liveStampSnapshot !== null, 'Live co-occlusion never triggered PROD_W2_CONSERVATION');
  assert(sawHeldAfterFresh, 'Live PROD_W2_CONSERVATION did not transition to STAMP_HELD on next tick');

  // Verify chronos_eval_occluded_pair clears live stamp and does not spoof live status on subsequent idle ticks
  exp.chronos_eval_occluded_pair();
  assert.equal(
    wave2Telem[PROD_W2_CONSERVATION_WORD] & STAMP_VALID,
    0,
    'chronos_eval_occluded_pair must clear PROD_W2_CONSERVATION live stamp'
  );
  for (let k = 0; k < 2; k++) {
    pushLiveFrame(bg, seq++, ptsUs);
    ptsUs += dtUs;
    assert.equal(
      wave2Telem[PROD_W2_CONSERVATION_WORD] & STAMP_VALID,
      0,
      'PROD_W2_CONSERVATION must stay invalid after staged override until a new live co-occlusion occurs'
    );
  }

  return { maxCoastActive, liveStampFrame, ...liveStampSnapshot };
}

const liveElastic = runLiveCoOcclusionScene({ boostA: 1.0, boostB: 1.0 });
assert(
  liveElastic.status === OCCL_STATUS_AMBIGUOUS || liveElastic.status === OCCL_STATUS_UNIQUE,
  `Live elastic co-occlusion expected UNIQUE/AMBIGUOUS, got status=${liveElastic.status}`
);
assert(liveElastic.count >= 1, `Live elastic co-occlusion expected admissible_count >= 1, got ${liveElastic.count}`);
assert(
  liveElastic.netImpulse <= liveElastic.envBound,
  `Live elastic co-occlusion expected netImpulse (${liveElastic.netImpulse}) <= envBound (${liveElastic.envBound})`
);
assertNear(liveElastic.massRatio, 1.0, 1e-6, 'Live co-occlusion declared unit mass ratio');

const liveViolated = runLiveCoOcclusionScene({ boostA: 1.75, boostB: 0.80 });
assert.equal(
  liveViolated.status,
  OCCL_STATUS_VIOLATED,
  `Live boosted co-occlusion expected OCCL_STATUS_VIOLATED, got status=${liveViolated.status} (dK=${liveViolated.deltaKeRatio})`
);
assert.equal(liveViolated.count, 0, `Live boosted co-occlusion expected admissible_count=0, got ${liveViolated.count}`);
assert(
  liveViolated.netImpulse > liveViolated.envBound,
  `Live boosted co-occlusion expected netImpulse (${liveViolated.netImpulse}) > envBound (${liveViolated.envBound})`
);

// Single-coasting disc (disc 1 at y=111 outside [78..102] bar, no bounce) must NOT trigger co-occlusion
runLiveCoOcclusionScene({
  barY0: 78,
  barY1: 102,
  y0: 90.0,
  y1: 111.0,
  noBounce: true,
  expectNoValidStamp: true,
});

// Continuity-epoch skip during occlusion (f=50) must reset pending conservation tracking
runLiveCoOcclusionScene({
  boostA: 1.0,
  boostB: 1.0,
  skipSeqAt50: true,
  expectNoValidStamp: true,
});

console.log(
  `[PASS] Gate I5b (Case1 mask=${case1Point.mask} status=${case1Point.status} e=${case1Point.e.toFixed(3)} | ` +
    `Case2 status=${case2.status} mask=${case2.mask} tau=${case2.tau.toFixed(3)}s e=${case2.e.toFixed(3)} m2/m1=${case2.massRatio.toFixed(1)} | ` +
    `Case3 status=${case3.status} J=${case3.netImpulse.toFixed(1)}>${case3.envBound.toFixed(1)} | ` +
    `Case4 status=${case4.status} mask=${case4.mask} dK=${case4.deltaKeRatio.toFixed(4)} | ` +
    `Case5 Live coast=${liveElastic.maxCoastActive}f st=${liveElastic.status}/mask=${liveElastic.mask} vs Viol=${liveViolated.status})`
);
