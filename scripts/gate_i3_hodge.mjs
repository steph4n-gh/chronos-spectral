#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I3: 40x22 Geometric 2-Complex Discrete Hodge
//
// Verifies:
//   1. Source hygiene (zero em dashes, zero banned codenames) & C-ABI memory layout
//   2. Full 40x22 topology: (V=880, E=1698, F=819, C=1, beta_1=0, holes=0, ports=0)
//   3. Punctured cylinder vs slit-to-border topology & dual plaquette flood-fill
//   4. Exact potential / irrotational source fields on full & punctured grids
//   5. Simply-connected contained vortex (beta_1=0, coexact > 0.98, harm < 1e-6)
//   6. Analytic 2D potential flow around cylinder (Gamma=0, beta_1=1, harm < 0.02)
//   7. Nonzero circulation (+Gamma / -Gamma) around cylinder (harm > 0.80, signed Gamma_1)
//   8. Downstream wake-deficit hypothesis vs symmetric flow + fail-closed invalidation
//   9. Live MicroSwarm continuum integration & Wave-2 stamp anti-spoofing
//
// Usage:
//   node scripts/gate_i3_hodge.mjs [--wasm=<path>]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const wasmArg = process.argv.slice(2).find((a) => a.startsWith('--wasm='));
const WASM_PATH = wasmArg ? path.resolve(wasmArg.slice('--wasm='.length)) : DEFAULT_WASM;

const HODGE_W = 40;
const HODGE_H = 22;
const HODGE_V = HODGE_W * HODGE_H; // 880
const HODGE_EX = (HODGE_W - 1) * HODGE_H; // 858
const HODGE_EY = HODGE_W * (HODGE_H - 1); // 840
const HODGE_E = HODGE_EX + HODGE_EY; // 1,698
const HODGE_F = (HODGE_W - 1) * (HODGE_H - 1); // 819

const HODGE_MODE_OFF = 0;
const HODGE_MODE_STAGGERED = 1;
const HODGE_MODE_EVERY_FRAME = 2;

const HODGE_STATE_OFF = 0;
const HODGE_STATE_STANDBY = 1;
const HODGE_STATE_READY = 2;
const HODGE_STATE_INVALID_UNCONVERGED = 3;

const STAMP_VALID = 1 << 0;
const STAMP_HELD = 1 << 1;

const DX = 16.0;
const DY = 360.0 / 22.0;

function assert(cond, msg) {
  if (!cond) {
    throw new Error(`Gate I3 FAIL: ${msg}`);
  }
}

// ----------------------------------------------------------------------------
// 1. Source Hygiene & Wasm Instantiation
// ----------------------------------------------------------------------------
{
  const bannedCodenameRe = new RegExp(
    `(${['CHRONOS', 'IMPOSSIBLE'].join('-')}|${['CHRONOS', '1000'].join('-')})`
  );
  const hodgeSrc = fs.readFileSync(path.join(ROOT, 'src', 'hodge.rs'), 'utf8');
  const selfSrc = fs.readFileSync(fileURLToPath(import.meta.url), 'utf8');
  for (const [label, txt] of [
    ['src/hodge.rs', hodgeSrc],
    ['scripts/gate_i3_hodge.mjs', selfSrc],
  ]) {
    assert(!txt.includes('\u2014'), `${label}: contains forbidden em dash`);
    assert(!bannedCodenameRe.test(txt), `${label}: contains forbidden internal codename`);
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
  'chronos_set_swarm_mode',
  'chronos_set_ingest_meta',
  'chronos_get_ingest_rgba_ptr',
  'chronos_tick',
  'chronos_get_wave1_telemetry_ptr',
  'chronos_get_wave2_telemetry_ptr',
  'chronos_get_continuum_field_ptr',
  'chronos_get_swarm_telemetry_ptr',
  'chronos_set_hodge_mode',
  'chronos_get_hodge_telemetry_ptr',
  'chronos_get_hodge_planes_ptr',
  'chronos_get_hodge_inject_ptr',
  'chronos_hodge_solve_injected',
];
for (const name of REQUIRED_EXPORTS) {
  assert(typeof exp[name] === 'function', `Missing required export ${name}`);
}

assert(mem.buffer.byteLength === 448 * 65536, `Expected 448 Wasm pages, got ${mem.buffer.byteLength}`);
let grew = false;
try {
  mem.grow(1);
  grew = true;
} catch {}
assert(!grew, 'Expected fixed max=448 Wasm memory');

const buf = mem.buffer;
const hTelemPtr = exp.chronos_get_hodge_telemetry_ptr();
const hPlanesPtr = exp.chronos_get_hodge_planes_ptr();
const hInjectPtr = exp.chronos_get_hodge_inject_ptr();
const w2TelemPtr = exp.chronos_get_wave2_telemetry_ptr();
const contFieldPtr = exp.chronos_get_continuum_field_ptr();
const swarmTelemPtr = exp.chronos_get_swarm_telemetry_ptr();

assert((hTelemPtr & 3) === 0 && hTelemPtr + 32 * 4 <= buf.byteLength, 'hodge_telem ptr invalid');
assert((hPlanesPtr & 3) === 0 && hPlanesPtr + 7 * HODGE_V * 4 <= buf.byteLength, 'hodge_planes ptr invalid');
assert((hInjectPtr & 3) === 0 && hInjectPtr + 3 * HODGE_V * 4 <= buf.byteLength, 'hodge_inject ptr invalid');

const regions = [
  [hTelemPtr, hTelemPtr + 32 * 4, 'hodge_telem'],
  [hPlanesPtr, hPlanesPtr + 7 * HODGE_V * 4, 'hodge_planes'],
  [hInjectPtr, hInjectPtr + 3 * HODGE_V * 4, 'hodge_inject'],
  [w2TelemPtr, w2TelemPtr + 64 * 4, 'wave2_telem'],
  [contFieldPtr, contFieldPtr + HODGE_V * 6 * 4, 'continuum_field'],
];
for (let i = 0; i < regions.length; i++) {
  for (let j = i + 1; j < regions.length; j++) {
    const [a0, a1, na] = regions[i];
    const [b0, b1, nb] = regions[j];
    assert(!(a0 < b1 && b0 < a1), `Memory overlap between ${na} and ${nb}`);
  }
}

const hTelemU32 = new Uint32Array(buf, hTelemPtr, 32);
const hTelemF32 = new Float32Array(buf, hTelemPtr, 32);
const hPlanes = new Float32Array(buf, hPlanesPtr, 7 * HODGE_V);
const hInject = new Float32Array(buf, hInjectPtr, 3 * HODGE_V);
const w2Telem = new Uint32Array(buf, w2TelemPtr, 64);
const swarmTelem = new Float32Array(buf, swarmTelemPtr, 64);

function readTelem() {
  return {
    state: hTelemU32[0],
    mode: hTelemU32[1],
    converged: hTelemU32[2],
    vAct: hTelemU32[3],
    eAct: hTelemU32[4],
    fAct: hTelemU32[5],
    compCnt: hTelemU32[6],
    beta1: hTelemU32[7],
    internalVoids: hTelemU32[8],
    openPorts: hTelemU32[9],
    wakeHyps: hTelemU32[10],
    totalEnergy: hTelemF32[11],
    exactFrac: hTelemF32[12],
    coexactFrac: hTelemF32[13],
    harmFrac: hTelemF32[14],
    orthEcRel: hTelemF32[15],
    orthEhRel: hTelemF32[16],
    orthChRel: hTelemF32[17],
    energyPartErr: hTelemF32[18],
    exactCgRes: hTelemF32[19],
    coexactCgRes: hTelemF32[20],
    primaryVoidCirc: hTelemF32[21],
    primaryWakeDeficit: hTelemF32[22],
    voidCx: hTelemF32[23],
    voidCy: hTelemF32[24],
    wakeX0: hTelemF32[25],
    wakeY0: hTelemF32[26],
    wakeX1: hTelemF32[27],
    wakeY1: hTelemF32[28],
    exactIters: hTelemU32[29],
    coexactIters: hTelemU32[30],
    reserved31: hTelemU32[31],
  };
}

function checkFiniteTelemAndPlanes(label) {
  assert(hTelemU32[31] === 0, `${label}: telem[31] must stay 0, got ${hTelemU32[31]}`);
  for (let k = 11; k <= 28; k++) {
    assert(Number.isFinite(hTelemF32[k]), `${label}: non-finite telemF32[${k}]=${hTelemF32[k]}`);
  }
  for (let i = 0; i < hPlanes.length; i++) {
    assert(Number.isFinite(hPlanes[i]), `${label}: non-finite plane[${i}]=${hPlanes[i]}`);
  }
}

function stageField(dx, dy, fn) {
  for (let r = 0; r < HODGE_H; r++) {
    for (let c = 0; c < HODGE_W; c++) {
      const v = r * HODGE_W + c;
      const x = (c + 0.5) * dx;
      const y = (r + 0.5) * dy;
      const [u, vy, mask] = fn(c, r, x, y);
      hInject[v] = u;
      hInject[HODGE_V + v] = vy;
      hInject[2 * HODGE_V + v] = mask;
    }
  }
}

function assertOrthogonal(t, label) {
  assert(t.state === HODGE_STATE_READY && t.converged === 1, `${label}: expected READY+converged`);
  assert(t.exactCgRes <= 1e-5 && t.coexactCgRes <= 1e-5, `${label}: CG residual too high (${t.exactCgRes}, ${t.coexactCgRes})`);
  assert(t.orthEcRel < 1e-7, `${label}: orthEcRel=${t.orthEcRel} >= 1e-7`);
  assert(t.orthEhRel < 1e-7, `${label}: orthEhRel=${t.orthEhRel} >= 1e-7`);
  assert(t.orthChRel < 1e-7, `${label}: orthChRel=${t.orthChRel} >= 1e-7`);
  assert(t.energyPartErr < 1e-6, `${label}: energyPartErr=${t.energyPartErr} >= 1e-6`);
}

// Mode clamping & initial state check
exp.chronos_init();
assert(hTelemU32[0] === HODGE_STATE_OFF && hTelemU32[1] === HODGE_MODE_OFF, 'Expected init OFF');
assert(exp.chronos_set_hodge_mode(99) === 1, 'set_hodge_mode must clamp and return 1');
assert(hTelemU32[1] === HODGE_MODE_EVERY_FRAME && hTelemU32[0] === HODGE_STATE_STANDBY, 'Mode 99 must clamp to 2 and STANDBY');
assert((w2Telem[2] & (1 << 1)) !== 0, 'wave2_telem[2] bit 1 must reflect active Hodge mode');
assert(exp.chronos_set_hodge_mode(HODGE_MODE_OFF) === 1 && hTelemU32[0] === HODGE_STATE_OFF, 'Mode 0 must return to OFF');

// ----------------------------------------------------------------------------
// 2. Full Grid Topology & Uniform Inflow
// ----------------------------------------------------------------------------
exp.chronos_init();
stageField(DX, DY, () => [12.0, -5.0, 1.0]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Full uniform solve failed');
const tFull = readTelem();
checkFiniteTelemAndPlanes('FullUniform');
assert(
  tFull.vAct === HODGE_V &&
    tFull.eAct === HODGE_E &&
    tFull.fAct === HODGE_F &&
    tFull.compCnt === 1 &&
    tFull.beta1 === 0 &&
    tFull.internalVoids === 0 &&
    tFull.openPorts === 0,
  `Full topology mismatch: ${JSON.stringify(tFull)}`
);
assertOrthogonal(tFull, 'FullUniform');
assert(tFull.exactFrac > 0.99999 && tFull.coexactFrac < 1e-7 && tFull.harmFrac < 1e-7, 'Uniform flow must be purely exact');

// ----------------------------------------------------------------------------
// 3. Punctured Cylinder vs Slit-to-Border vs Corner / Two-Hole Topology
// ----------------------------------------------------------------------------
const CYL_CX_CELL = 19.5;
const CYL_CY_CELL = 10.5;
const CYL_R_CELL = 4.0;
const CYL_MASK = (c, r) => (Math.hypot(c - CYL_CX_CELL, r - CYL_CY_CELL) > CYL_R_CELL ? 1.0 : 0.0);

// 3a. Enclosed disk hole -> beta1 = 1, holes = 1, ports = 0
stageField(DX, DY, (c, r) => [1.0, 0.0, CYL_MASK(c, r)]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Cyl mask solve failed');
const tCylTopo = readTelem();
checkFiniteTelemAndPlanes('CylTopo');
assert(
  tCylTopo.beta1 === 1 && tCylTopo.internalVoids === 1 && tCylTopo.openPorts === 0,
  `Cyl topology expected beta1=1,holes=1,ports=0; got beta1=${tCylTopo.beta1},holes=${tCylTopo.internalVoids},ports=${tCylTopo.openPorts}`
);

// 3b. Open 1-cell slit from disk to left border -> flood enters -> beta1 = 0, holes = 0, ports = 1
stageField(DX, DY, (c, r) => [1.0, 0.0, CYL_MASK(c, r) > 0.5 && !(r === 10 && c <= 19) ? 1.0 : 0.0]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Slit mask solve failed');
const tSlitTopo = readTelem();
checkFiniteTelemAndPlanes('SlitTopo');
assert(
  tSlitTopo.beta1 === 0 && tSlitTopo.internalVoids === 0 && tSlitTopo.openPorts === 1,
  `Slit topology expected beta1=0,holes=0,ports=1; got beta1=${tSlitTopo.beta1},holes=${tSlitTopo.internalVoids},ports=${tSlitTopo.openPorts}`
);

// 3c. Single interior vertex (1, 1) removed -> beta1 = 1, holes = 1
stageField(DX, DY, (c, r) => [1.0, 0.0, c === 1 && r === 1 ? 0.0 : 1.0]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Interior point mask solve failed');
const tPtIn = readTelem();
assert(tPtIn.beta1 === 1 && tPtIn.internalVoids === 1 && tPtIn.openPorts === 0, 'Interior single-vertex puncture must yield beta1=1,holes=1');

// 3d. Corner vertex (0, 0) removed -> open to exterior -> beta1 = 0, holes = 0, ports = 1
stageField(DX, DY, (c, r) => [1.0, 0.0, c === 0 && r === 0 ? 0.0 : 1.0]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Corner point mask solve failed');
const tCorner = readTelem();
assert(tCorner.beta1 === 0 && tCorner.internalVoids === 0 && tCorner.openPorts === 1, 'Corner puncture must yield beta1=0,holes=0,ports=1');

// 3e. Two separated enclosed holes -> beta1 = 2, holes = 2
stageField(DX, DY, (c, r) => [
  1.0,
  0.0,
  Math.hypot(c - 12.5, r - 10.5) > 3.0 && Math.hypot(c - 26.5, r - 10.5) > 3.0 ? 1.0 : 0.0,
]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Two-hole mask solve failed');
const tTwo = readTelem();
assert(tTwo.beta1 === 2 && tTwo.internalVoids === 2 && tTwo.openPorts === 0, 'Two enclosed holes must yield beta1=2,holes=2');

// ----------------------------------------------------------------------------
// 4. Pure Irrotational Source / Expansion & Discrete Gradient Exactness
// ----------------------------------------------------------------------------
const CX_PHYS = (CYL_CX_CELL + 0.5) * DX;
const CY_PHYS = (CYL_CY_CELL + 0.5) * DY;

stageField(DX, DY, (c, r, x, y) => [x - CX_PHYS, y - CY_PHYS, 1.0]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Full expansion solve failed');
const tExpFull = readTelem();
checkFiniteTelemAndPlanes('ExpFull');
assertOrthogonal(tExpFull, 'ExpFull');
assert(
  tExpFull.exactFrac > 0.99 && tExpFull.coexactFrac < 1e-3 && tExpFull.harmFrac < 1e-6,
  `Full expansion fractions out of range: ex=${tExpFull.exactFrac}, co=${tExpFull.coexactFrac}, h=${tExpFull.harmFrac}`
);

// Punctured grid with radial source (zero circulation -> pure exact gradient on punctured domain)
stageField(DX, DY, (c, r, x, y) => {
  const rx = x - CX_PHYS;
  const ry = y - CY_PHYS;
  const r2 = Math.max(rx * rx + ry * ry, 1.0);
  return [(1500.0 * rx) / r2, (1500.0 * ry) / r2, CYL_MASK(c, r)];
});
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Punctured source solve failed');
const tExpCyl = readTelem();
checkFiniteTelemAndPlanes('ExpCyl');
assertOrthogonal(tExpCyl, 'ExpCyl');
assert(
  tExpCyl.exactFrac > 0.99 && tExpCyl.coexactFrac < 1e-3 && tExpCyl.harmFrac < 1e-6,
  `Punctured source fractions out of range: ex=${tExpCyl.exactFrac}, co=${tExpCyl.coexactFrac}, h=${tExpCyl.harmFrac}`
);

// ----------------------------------------------------------------------------
// 5. Simply-Connected Contained Vortex (beta_1 = 0 -> zero harmonic leakage)
// ----------------------------------------------------------------------------
const LX = HODGE_W * DX;
const LY = HODGE_H * DY;
stageField(DX, DY, (c, r, x, y) => {
  const amp = 80.0;
  const u = amp * (Math.PI / LY) * Math.sin((Math.PI * x) / LX) * Math.cos((Math.PI * y) / LY);
  const v = -amp * (Math.PI / LX) * Math.cos((Math.PI * x) / LX) * Math.sin((Math.PI * y) / LY);
  return [u, v, 1.0];
});
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Simply-connected vortex solve failed');
const tVort = readTelem();
checkFiniteTelemAndPlanes('SimpleVortex');
assertOrthogonal(tVort, 'SimpleVortex');
assert(tVort.beta1 === 0, `SimpleVortex expected beta1=0, got ${tVort.beta1}`);
assert(
  tVort.coexactFrac > 0.98 && tVort.harmFrac < 1e-6,
  `SimpleVortex fractions out of range: co=${tVort.coexactFrac}, harm=${tVort.harmFrac}`
);

// ----------------------------------------------------------------------------
// 6. Analytic 2D Potential Flow Around Circular Cylinder (Gamma = 0)
// ----------------------------------------------------------------------------
const RC_PHYS = CYL_R_CELL * DX;
const U_INF = 20.0;
stageField(DX, DY, (c, r, x, y) => {
  const m = CYL_MASK(c, r);
  if (m < 0.5) return [0.0, 0.0, 0.0];
  const rx = x - CX_PHYS;
  const ry = y - CY_PHYS;
  const r2 = rx * rx + ry * ry;
  const r4 = r2 * r2;
  const u = U_INF * (1.0 + (RC_PHYS * RC_PHYS * (ry * ry - rx * rx)) / r4);
  const v = U_INF * ((-2.0 * RC_PHYS * RC_PHYS * rx * ry) / r4);
  return [u, v, 1.0];
});
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Potential cylinder flow solve failed');
const tPotCyl = readTelem();
checkFiniteTelemAndPlanes('PotCyl');
assertOrthogonal(tPotCyl, 'PotCyl');
assert(
  tPotCyl.beta1 === 1 &&
    tPotCyl.internalVoids === 1 &&
    tPotCyl.exactFrac > 0.98 &&
    tPotCyl.harmFrac < 0.02 &&
    tPotCyl.wakeHyps === 0,
  `PotCyl invariant failed: ${JSON.stringify(tPotCyl)}`
);

// ----------------------------------------------------------------------------
// 7. Nonzero Circulation (+Gamma and -Gamma) Around Enclosed Disk Mask
// ----------------------------------------------------------------------------
let tCircPos = null;
for (const gammaTarget of [1200.0, -1200.0]) {
  stageField(DX, DY, (c, r, x, y) => {
    const m = CYL_MASK(c, r);
    if (m < 0.5) return [0.0, 0.0, 0.0];
    const rx = x - CX_PHYS;
    const ry = y - CY_PHYS;
    const r2 = rx * rx + ry * ry;
    const scale = gammaTarget / (2.0 * Math.PI * r2);
    return [-scale * ry, scale * rx, 1.0];
  });
  assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, `Circ(${gammaTarget}) solve failed`);
  const tCirc = readTelem();
  if (gammaTarget > 0) tCircPos = tCirc;
  checkFiniteTelemAndPlanes(`Circ(${gammaTarget})`);
  assertOrthogonal(tCirc, `Circ(${gammaTarget})`);
  const relErr = Math.abs(tCirc.primaryVoidCirc - gammaTarget) / Math.abs(gammaTarget);
  assert(
    tCirc.beta1 === 1 &&
      tCirc.internalVoids === 1 &&
      tCirc.harmFrac > 0.80 &&
      Math.sign(tCirc.primaryVoidCirc) === Math.sign(gammaTarget) &&
      relErr < 0.02,
    `Circ(${gammaTarget}) failed: harmFrac=${tCirc.harmFrac}, circ=${tCirc.primaryVoidCirc}, relErr=${relErr}`
  );
}

// ----------------------------------------------------------------------------
// 8. Downstream Wake-Deficit Hypothesis & Degenerate Fail-Closed Invalidation
// ----------------------------------------------------------------------------
stageField(DX, DY, (c, r, x, y) => {
  const m = CYL_MASK(c, r);
  if (m < 0.5) return [0.0, 0.0, 0.0];
  const rx = x - CX_PHYS;
  const ry = y - CY_PHYS;
  const r2 = rx * rx + ry * ry;
  const r4 = r2 * r2;
  let u = 40.0 * (1.0 + (RC_PHYS * RC_PHYS * (ry * ry - rx * rx)) / r4);
  let v = 40.0 * ((-2.0 * RC_PHYS * RC_PHYS * rx * ry) / r4);
  if (rx > 0.0 && Math.abs(ry) < 1.1 * RC_PHYS) {
    const dDown = rx / RC_PHYS;
    const atten = 1.0 - 0.78 * Math.exp(-0.22 * Math.max(0.0, dDown - 0.8));
    u *= atten;
    v *= atten;
  }
  return [u, v, 1.0];
});
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Wake cylinder solve failed');
const tWake = readTelem();
checkFiniteTelemAndPlanes('WakeCyl');
assertOrthogonal(tWake, 'WakeCyl');
assert(
  tWake.wakeHyps >= 1 &&
    tWake.primaryWakeDeficit > 0.25 &&
    Math.abs(tWake.voidCx - CX_PHYS) < 2.0 * DX &&
    Math.abs(tWake.voidCy - CY_PHYS) < 2.0 * DY &&
    tWake.wakeX0 > tWake.voidCx &&
    tWake.wakeX1 >= tWake.wakeX0 &&
    tWake.wakeY1 >= tWake.wakeY0,
  `Wake hypothesis failed: ${JSON.stringify(tWake)}`
);

function assertFailClosed(label, dxVal = DX, dyVal = DY) {
  const rc = exp.chronos_hodge_solve_injected(dxVal, dyVal);
  const t = readTelem();
  assert(rc === 0, `${label}: expected return 0, got ${rc}`);
  assert(t.state === HODGE_STATE_INVALID_UNCONVERGED, `${label}: expected state 3, got ${t.state}`);
  assert(t.converged === 0, `${label}: expected converged=0, got ${t.converged}`);
  assert((w2Telem[24] & STAMP_VALID) === 0, `${label}: live stamp must stay invalid`);
  checkFiniteTelemAndPlanes(label);
  for (let i = 0; i < hPlanes.length; i++) {
    assert(hPlanes[i] === 0.0, `${label}: plane[${i}]=${hPlanes[i]} must be zeroed`);
  }
}

// 8a. < 4 active vertices (only 3 active)
stageField(DX, DY, (c, r) => [1.0, 0.0, r === 5 && c >= 5 && c < 8 ? 1.0 : 0.0]);
assertFailClosed('TooFewVertices');

// 8b. 4 isolated active vertices (0 active edges)
stageField(DX, DY, (c, r) => [1.0, 0.0, (c === 2 || c === 6) && (r === 2 || r === 6) ? 1.0 : 0.0]);
assertFailClosed('ZeroEdges');

// 8c. NaN active velocity
stageField(DX, DY, (c, r) => [c === 10 && r === 10 ? Number.NaN : 1.0, 0.0, 1.0]);
assertFailClosed('NanVelocity');

// 8d. NaN mask entry
stageField(DX, DY, (c, r) => [1.0, 0.0, c === 10 && r === 10 ? Number.NaN : 1.0]);
assertFailClosed('NanMask');

// 8e. Invalid grid spacing (dx <= 0)
stageField(DX, DY, () => [1.0, 0.0, 1.0]);
assertFailClosed('InvalidDx', 0.0, DY);

// 8f. f32 overflow input (3.4028234e38 -> energy overflows f32)
stageField(DX, DY, () => [3.4028234e38, 0.0, 1.0]);
assertFailClosed('F32Overflow');

// ----------------------------------------------------------------------------
// 9. Live MicroSwarm Continuum Integration & Wave-2 Stamp Anti-Spoofing
// ----------------------------------------------------------------------------
function buildVortexSwarmScene(numFrames = 26) {
  const rc = 48.0;
  const K = 4600.0;
  const xL = 14.0;
  const xR = 626.0;
  const yT = 14.0;
  const yB = 346.0;
  const vortices = [];
  for (const mx of [-1, 0, 1]) {
    for (const my of [-1, 0, 1]) {
      const sign = (mx !== 0 ? -1 : 1) * (my !== 0 ? -1 : 1);
      for (const [bx, by, bk] of [
        [210.0, 180.0, K],
        [430.0, 180.0, -K],
      ]) {
        vortices.push({
          cx: mx === 0 ? bx : mx === -1 ? 2.0 * xL - bx : 2.0 * xR - bx,
          cy: my === 0 ? by : my === -1 ? 2.0 * yT - by : 2.0 * yB - by,
          k: bk * sign,
        });
      }
    }
  }
  const velAt = (x, y) => {
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
  };
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
      pts.push({
        id: idx + 1,
        x: 22.0 + ((i + 0.5) * 596.0) / nx + jx,
        y: 20.0 + ((j + 0.5) * 320.0) / ny + jy,
      });
    }
  }
  const dt = 1.0 / 30.0;
  const frames = [];
  const w = 640;
  const h = 360;
  for (let f = 0; f < numFrames; f++) {
    const rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < rgba.length; i += 4) {
      rgba[i] = 10;
      rgba[i + 1] = 14;
      rgba[i + 2] = 22;
      rgba[i + 3] = 255;
    }
    for (let p = 0; p < pts.length; p++) {
      const pt = pts[p];
      const cx = pt.x;
      const cy = pt.y;
      const x0 = Math.max(0, Math.floor(cx) - 3);
      const x1 = Math.min(w - 1, Math.floor(cx) + 3);
      const y0 = Math.max(0, Math.floor(cy) - 3);
      const y1 = Math.min(h - 1, Math.floor(cy) + 3);
      for (let py = y0; py <= y1; py++) {
        const dy = py - cy;
        for (let px = x0; px <= x1; px++) {
          const dx = px - cx;
          const r2 = dx * dx + dy * dy;
          if (r2 > 4.84) continue;
          const g = Math.exp(-r2 / 1.445);
          const idx = (py * w + px) * 4;
          const val = Math.min(255, Math.round(14 + 225 * g));
          if (val > rgba[idx]) rgba[idx] = val;
          if (val > rgba[idx + 1]) rgba[idx + 1] = val;
          if (val > rgba[idx + 2]) rgba[idx + 2] = val;
        }
      }
      const [u1, v1] = velAt(pt.x, pt.y);
      const [u2, v2] = velAt(pt.x + 0.5 * dt * u1, pt.y + 0.5 * dt * v1);
      const [u3, v3] = velAt(pt.x + 0.5 * dt * u2, pt.y + 0.5 * dt * v2);
      const [u4, v4] = velAt(pt.x + dt * u3, pt.y + dt * v3);
      pt.x += ((u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0) * dt;
      pt.y += ((v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0) * dt;
    }
    frames.push(rgba);
  }
  return frames;
}

const liveFrames = buildVortexSwarmScene(26);
exp.chronos_init();
exp.chronos_set_resolution(1);
exp.chronos_set_swarm_mode(2);
exp.chronos_set_hodge_mode(HODGE_MODE_EVERY_FRAME);

const dtSec = 1.0 / 30.0;
const flagsBase = 2 | 16; // PTS known + Geom known
let liveFreshFrames = 0;
let liveHeldFrames = 0;
let lastLiveTelem = null;

for (let f = 0; f < liveFrames.length; f++) {
  const ptr = exp.chronos_get_ingest_rgba_ptr();
  new Uint8Array(buf, ptr, 640 * 360 * 4).set(liveFrames[f]);
  const tUs = BigInt(Math.round(f * dtSec * 1_000_000));
  const ok = exp.chronos_set_ingest_meta(
    1,
    f + 1,
    Number(tUs & 0xffffffffn),
    Number((tUs >> 32n) & 0xffffffffn),
    -1.0,
    flagsBase,
    640,
    360,
    0,
    0,
    640,
    360,
    0,
    0xffffffff,
    0,
    0,
    640,
    360,
    0,
    1
  );
  assert(ok === 1, `Live frame ${f}: chronos_set_ingest_meta failed`);
  exp.chronos_tick(dtSec);

  if (f >= 20) {
    const st = w2Telem[24];
    const t = readTelem();
    checkFiniteTelemAndPlanes(`LiveFrame${f}`);
    if ((st & STAMP_VALID) !== 0) {
      assert(t.state === HODGE_STATE_READY && t.converged === 1, `LiveFrame${f}: stamp valid without READY`);
      if ((st & STAMP_HELD) !== 0) liveHeldFrames++;
      else liveFreshFrames++;
      lastLiveTelem = t;
    }
  }
}

assert(liveFreshFrames >= 2, `Expected >= 2 fresh live Hodge frames, got ${liveFreshFrames}`);
assert(liveHeldFrames >= 2, `Expected >= 2 held live Hodge frames under dense split, got ${liveHeldFrames}`);
assert(
  lastLiveTelem &&
    lastLiveTelem.vAct >= 200 &&
    lastLiveTelem.coexactFrac > 0.30 &&
    lastLiveTelem.orthEcRel < 1e-7 &&
    lastLiveTelem.orthEhRel < 1e-7 &&
    lastLiveTelem.orthChRel < 1e-7,
  `Live vortex swarm Hodge telemetry invalid: ${JSON.stringify(lastLiveTelem)}`
);

// Anti-spoof: injected solve MUST clear Wave-2 live stamp immediately (and never claim live readiness)
stageField(DX, DY, () => [5.0, 0.0, 1.0]);
assert(exp.chronos_hodge_solve_injected(DX, DY) === 1, 'Injected solve after live run must succeed');
assert((w2Telem[24] & STAMP_VALID) === 0, 'chronos_hodge_solve_injected must invalidate Wave-2 Hodge live stamp');

// Verify staggered mode (mode = 1) runs cleanly on subsequent live frames
exp.chronos_set_hodge_mode(HODGE_MODE_STAGGERED);
assert(hTelemU32[1] === HODGE_MODE_STAGGERED, 'Expected staggered mode 1');
for (let f = 22; f < 26; f++) {
  const ptr = exp.chronos_get_ingest_rgba_ptr();
  new Uint8Array(buf, ptr, 640 * 360 * 4).set(liveFrames[f]);
  const tUs = BigInt(Math.round((f + 10) * dtSec * 1_000_000));
  assert(
    exp.chronos_set_ingest_meta(
      1,
      f + 10,
      Number(tUs & 0xffffffffn),
      Number((tUs >> 32n) & 0xffffffffn),
      -1.0,
      flagsBase,
      640,
      360,
      0,
      0,
      640,
      360,
      0,
      0xffffffff,
      0,
      0,
      640,
      360,
      0,
      1
    ) === 1
  );
  exp.chronos_tick(dtSec);
}
assert(
  hTelemU32[0] === HODGE_STATE_READY && hTelemU32[2] === 1 && (w2Telem[24] & STAMP_VALID) !== 0,
  'Live tick after staggered mode switch must re-establish live READY and Wave-2 stamp'
);
checkFiniteTelemAndPlanes('LiveStaggered');

// Disabling swarm while paused MUST invalidate live Hodge stamp, clear planes, and return to STANDBY
exp.chronos_set_swarm_mode(0);
assert(
  (w2Telem[24] & STAMP_VALID) === 0 &&
    hTelemU32[0] === HODGE_STATE_STANDBY &&
    hTelemU32[2] === 0 &&
    hPlanes.every((v) => v === 0.0),
  'chronos_set_swarm_mode(0) must invalidate live Hodge stamp, clear planes, and leave READY for STANDBY'
);

// Small finite exact fields (u = 1e-11 and u = 1e-20, v = 0, unit spacing) must keep normalized exactFrac ~= 1.0
for (const tinyU of [1e-11, 1e-20]) {
  stageField(1.0, 1.0, () => [tinyU, 0.0, 1.0]);
  assert(exp.chronos_hodge_solve_injected(1.0, 1.0) === 1, `Small finite injected field u=${tinyU} solve must succeed`);
  const tTiny = readTelem();
  assert(
    tTiny.state === HODGE_STATE_READY &&
      tTiny.converged === 1 &&
      Math.abs(tTiny.exactFrac - 1.0) < 1e-5 &&
      tTiny.harmFrac < 1e-10 &&
      tTiny.energyPartErr < 1e-6,
    `Expected exactFrac~=1.0 on u=${tinyU} field, got ${JSON.stringify(tTiny)}`
  );
  checkFiniteTelemAndPlanes(`TinyExact_${tinyU}`);
}
stageField(1.0, 1.0, () => [0.0, 0.0, 1.0]);
assert(exp.chronos_hodge_solve_injected(1.0, 1.0) === 1, 'Exact zero field solve must succeed');
{
  const tZero = readTelem();
  assert(
    tZero.state === HODGE_STATE_READY &&
      tZero.converged === 1 &&
      tZero.exactFrac === 0.0 &&
      tZero.coexactFrac === 0.0 &&
      tZero.harmFrac === 0.0,
    `Zero field expected 0/0/0 fractions, got ${JSON.stringify(tZero)}`
  );
  checkFiniteTelemAndPlanes('ZeroField');
}

// Live SD 320x180 Hodge must use 40x22 spacing (voidCx <= 320, voidCy <= 180)
{
  exp.chronos_init();
  exp.chronos_set_resolution(0);
  exp.chronos_set_swarm_mode(2);
  exp.chronos_set_hodge_mode(HODGE_MODE_EVERY_FRAME);
  const sdBuf = new Uint8Array(320 * 180 * 4);
  for (let f = 0; f < 14; f++) {
    const hd = liveFrames[f];
    for (let y = 0; y < 180; y++) {
      for (let x = 0; x < 320; x++) {
        const src = ((y * 2) * 640 + (x * 2)) * 4;
        const dst = (y * 320 + x) * 4;
        sdBuf[dst] = hd[src];
        sdBuf[dst + 1] = hd[src + 1];
        sdBuf[dst + 2] = hd[src + 2];
        sdBuf[dst + 3] = 255;
      }
    }
    new Uint8Array(buf, exp.chronos_get_ingest_rgba_ptr(), 320 * 180 * 4).set(sdBuf);
    const tUs = BigInt(Math.round(f * dtSec * 1_000_000));
    assert(
      exp.chronos_set_ingest_meta(
        1, f + 1, Number(tUs & 0xffffffffn), Number((tUs >> 32n) & 0xffffffffn), -1.0,
        flagsBase, 320, 180, 0, 0, 320, 180, 0, 0xffffffff, 0, 0, 320, 180, 0, 1
      ) === 1
    );
    exp.chronos_tick(dtSec);
  }
  const tSd = readTelem();
  assert(
    tSd.state === HODGE_STATE_READY &&
      tSd.converged === 1 &&
      (w2Telem[24] & STAMP_VALID) !== 0 &&
      tSd.voidCx >= 0.0 &&
      tSd.voidCx <= 320.0 &&
      tSd.voidCy >= 0.0 &&
      tSd.voidCy <= 180.0 &&
      tSd.wakeX1 <= 320.0 &&
      tSd.wakeY1 <= 180.0,
    `Live SD 320x180 Hodge coordinates out of frame bounds: ${JSON.stringify(tSd)}`
  );
  checkFiniteTelemAndPlanes('LiveSD320x180');
}

console.log(
  `[PASS] Gate I3 (Hodge 40x22): Full=(V=${tFull.vAct},E=${tFull.eAct},F=${tFull.fAct},b1=${tFull.beta1}) | ` +
    `Cyl=(b1=${tCylTopo.beta1},holes=${tCylTopo.internalVoids}) vs Slit=(b1=${tSlitTopo.beta1},ports=${tSlitTopo.openPorts}) | ` +
    `PotCyl(ex=${tPotCyl.exactFrac.toFixed(4)},h=${tPotCyl.harmFrac.toExponential(2)}) | ` +
    `Circ+1200(h=${tCircPos.harmFrac.toFixed(3)},G1=${tCircPos.primaryVoidCirc.toFixed(1)}) | Wake(hyps=${tWake.wakeHyps},def=${tWake.primaryWakeDeficit.toFixed(3)}) | ` +
    `Live(v=${lastLiveTelem.vAct},co=${lastLiveTelem.coexactFrac.toFixed(3)},fresh=${liveFreshFrames},held=${liveHeldFrames},orth=${lastLiveTelem.orthEcRel.toExponential(1)}) | ` +
    `SwarmTelem[16]=${swarmTelem[16]}`
);
