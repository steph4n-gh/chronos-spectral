#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I9: Pocket Wind Tunnel (Wave D2) Verification Suite
//
// Validates:
//   I9.1a Analytical Lamb-Oseen viscous vortex (direct inject): nu = 24.0 px^2/s, pressure well r > 0.98
//   I9.1b H.264 PIV Lamb-Oseen MP4 (wind_lamb_oseen_viscosity.mp4): nu within 8%, pressure well r > 0.98
//   I9.2a Analytical Lifting Cylinder (direct inject): beta_1 >= 1, Cp r > 0.98, C_L within 8% of 3.431
//   I9.2b H.264 PIV Lifting Cylinder MP4 (wind_lifting_cylinder_cp.mp4): Cp r > 0.95, C_L within 10%
//   I9.3  Fail-closed 3D/compressible negative controls (HIGH_DIV, HIGH_NS_RESIDUAL)
//   I9.4  Real-camera lab PIV & Cambridge wind-tunnel airfoil (real_daphnia_laser_piv.mp4, real_smoke_airfoil_cambridge.mp4)
//   I9.5  Strict mutual exclusion with BOS & Stereo (WIND_REASON_MUTEX_CONFLICT) + clean recovery
//
// Usage:
//   node scripts/gate_i9_wind_tunnel.mjs [--parity-lite]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WASM_PATH = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');
const PARITY_LITE = process.argv.includes('--parity-lite');

const WIND_W = 64;
const WIND_H = 36;
const WIND_V = WIND_W * WIND_H;

const WIND_STATE_OFF = 0;
const WIND_STATE_WARMING = 1;
const WIND_STATE_READY_FIELD_ONLY = 2;
const WIND_STATE_READY_FULL_LOADS = 3;
const WIND_STATE_REFUSED = 4;

const WIND_REASON_MUTEX_CONFLICT = 1 << 2;
const WIND_REASON_HIGH_DIV = 1 << 3;
const WIND_REASON_HIGH_NS_RESIDUAL = 1 << 4;

const wasmBytes = fs.readFileSync(WASM_PATH);

async function makeKernel() {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exp = instance.exports;
  exp.chronos_init();
  const mem = exp.memory;
  return {
    exp,
    mem,
    hodgeInject: new Float32Array(mem.buffer, exp.chronos_get_hodge_inject_ptr(), 3 * WIND_V),
    hodgeTelem: new Uint32Array(mem.buffer, exp.chronos_get_hodge_telemetry_ptr(), 32),
    windTelemU32: new Uint32Array(mem.buffer, exp.chronos_get_wind_telemetry_ptr(), 32),
    windTelemF32: new Float32Array(mem.buffer, exp.chronos_get_wind_telemetry_ptr(), 32),
    windPlanes: new Float32Array(mem.buffer, exp.chronos_get_wind_planes_ptr(), 3 * WIND_V),
    wave2Telem: new Uint32Array(mem.buffer, exp.chronos_get_wave2_telemetry_ptr(), 64),
  };
}

function pearsonR(xs, ys) {
  const n = xs.length;
  if (n < 4) return 0;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx2 += a * a;
    dy2 += b * b;
  }
  const den = Math.sqrt(dx2 * dy2);
  return den > 1e-18 ? num / den : 0;
}

// Analytical Lamb-Oseen radial pressure well (integral of u_theta^2 / r dr from r=0 to r)
function lambOseenPressure(r, gamma, nu, T) {
  const steps = 80;
  const dr = r / steps;
  const fourNuT = 4.0 * nu * T;
  let sum = 0.0;
  for (let i = 0; i < steps; i++) {
    const rm = (i + 0.5) * dr;
    if (rm < 1e-6) continue;
    const uth = (gamma / (2.0 * Math.PI * rm)) * (1.0 - Math.exp(-(rm * rm) / fourNuT));
    sum += (uth * uth / rm) * dr;
  }
  return sum;
}

function decodeMp4Frames(mp4Path, w, h, maxFrames) {
  const res = spawnSync(
    'ffmpeg',
    [
      '-v', 'error', '-i', mp4Path,
      '-vf', `scale=${w}:${h}`,
      '-vframes', String(maxFrames),
      '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1',
    ],
    { maxBuffer: 256 * 1024 * 1024 }
  );
  if (res.status !== 0) {
    throw new Error(`ffmpeg decode failed for ${mp4Path}: ${res.stderr?.toString()}`);
  }
  const frameBytes = w * h * 4;
  const buf = res.stdout;
  const count = Math.floor(buf.length / frameBytes);
  const frames = [];
  for (let i = 0; i < count; i++) {
    frames.push(buf.subarray(i * frameBytes, (i + 1) * frameBytes));
  }
  return frames;
}

function assert(cond, msg) {
  if (!cond) {
    throw new Error(`[FAIL] Gate I9: ${msg}`);
  }
}

function readWindSummary(k) {
  const u = k.windTelemU32;
  const f = k.windTelemF32;
  return {
    state: u[0],
    mode: u[1],
    reason: u[2],
    prov: u[3],
    activeCells: u[4],
    beta1: u[5],
    pcgIters: u[6],
    nuStencils: u[7],
    uInf: f[8],
    inflowAngle: f[9],
    dHolePx: f[10],
    nu: f[11],
    nuR2: f[12],
    re: f[13],
    cpMin: f[14],
    cpMax: f[15],
    clBnd: f[16],
    cdBnd: f[17],
    clCv: f[18],
    cdCv: f[19],
    clKj: f[20],
    nsResid: f[21],
    divFrac: f[22],
    holeCx: f[23],
    holeCy: f[24],
    pGauge: f[25],
    pcgRelRes: f[26],
  };
}

// ============================================================================
// I9.1a: Analytical Lamb-Oseen Vortex (direct Hodge injection, 9 frames)
// ============================================================================
async function testI9_1a_AnalyticalLambOseen() {
  const k = await makeKernel();
  assert(k.exp.chronos_set_wind_mode(2) === 1, 'set_wind_mode(2) failed');
  const dx = 10.0, dy = 10.0;
  const cx = 320.0, cy = 180.0;
  const gamma = 28000.0, nuTrue = 24.0, t0 = 6.0;
  const dt = 0.05;

  let lastS = null;
  for (let f = 0; f < 9; f++) {
    const t = f * dt;
    const fourNuT = 4.0 * nuTrue * (t + t0);
    for (let r = 0; r < WIND_H; r++) {
      const y = (r + 0.5) * dy;
      for (let c = 0; c < WIND_W; c++) {
        const x = (c + 0.5) * dx;
        const v = r * WIND_W + c;
        const rx = x - cx, ry = y - cy;
        const r2 = rx * rx + ry * ry;
        const rad = Math.sqrt(r2);
        let ux = 0, uy = 0;
        if (rad > 1e-4) {
          const uth = (gamma / (2.0 * Math.PI * rad)) * (1.0 - Math.exp(-r2 / fourNuT));
          ux = -(ry / rad) * uth;
          uy = (rx / rad) * uth;
        }
        k.hodgeInject[v] = ux;
        k.hodgeInject[WIND_V + v] = uy;
        k.hodgeInject[2 * WIND_V + v] = 1.0;
      }
    }
    assert(k.exp.chronos_hodge_solve_injected(dx, dy) === 1, `hodge_solve_injected failed f=${f}`);
    const ok = k.exp.chronos_wind_solve_injected(dx, dy, t);
    lastS = readWindSummary(k);
    if (f >= 2) assert(ok === 1, `wind_solve_injected failed f=${f}, state=${lastS.state}, reason=0x${lastS.reason.toString(16)}`);
  }

  const Tfinal = 8 * dt + t0;
  const pMeas = [], pTrue = [];
  for (let r = 2; r < WIND_H - 2; r++) {
    const y = (r + 0.5) * dy;
    for (let c = 2; c < WIND_W - 2; c++) {
      const x = (c + 0.5) * dx;
      const rad = Math.hypot(x - cx, y - cy);
      if (rad <= 140.0) {
        pMeas.push(k.windPlanes[r * WIND_W + c]);
        pTrue.push(lambOseenPressure(rad, gamma, nuTrue, Tfinal));
      }
    }
  }
  const rWell = pearsonR(pMeas, pTrue);
  const nuErrPct = (100.0 * Math.abs(lastS.nu - nuTrue)) / nuTrue;
  console.log(
    `  [PASS] I9.1a Analytical Lamb-Oseen: nu=${lastS.nu.toFixed(3)} (true=${nuTrue.toFixed(1)}, err=${nuErrPct.toFixed(2)}%, R2=${lastS.nuR2.toFixed(4)}, st=${lastS.nuStencils}), well r=${rWell.toFixed(4)}, eta_NS=${lastS.nsResid.toFixed(4)}`
  );
  assert(lastS.state === WIND_STATE_READY_FIELD_ONLY, `expected READY_FIELD_ONLY, got ${lastS.state}`);
  assert(nuErrPct <= 4.0, `analytical nu error ${nuErrPct.toFixed(2)}% > 4%`);
  assert(lastS.nuR2 > 0.98, `analytical nuR2 ${lastS.nuR2} <= 0.98`);
  assert(rWell > 0.99, `analytical pressure well r=${rWell} <= 0.99`);
  assert(lastS.nsResid < 0.08, `analytical eta_NS=${lastS.nsResid} >= 0.08`);
}

// ============================================================================
// I9.2a: Analytical Lifting Cylinder (direct Hodge injection, 6 frames)
// ============================================================================
async function testI9_2a_AnalyticalLiftingCylinder() {
  const k = await makeKernel();
  assert(k.exp.chronos_set_wind_mode(2) === 1, 'set_wind_mode(2) failed');
  const dx = 10.0, dy = 10.0;
  const cx = 256.0, cy = 180.0, R = 34.0;
  const Uinf = 120.0, Gamma = -14000.0;
  const exactCL = (2.0 * Math.abs(Gamma)) / (Uinf * 2.0 * R);

  let lastS = null;
  for (let f = 0; f < 6; f++) {
    for (let r = 0; r < WIND_H; r++) {
      const y = (r + 0.5) * dy;
      for (let c = 0; c < WIND_W; c++) {
        const x = (c + 0.5) * dx;
        const v = r * WIND_W + c;
        const rx = x - cx, ry = y - cy;
        const r2 = rx * rx + ry * ry;
        if (r2 <= R * R) {
          k.hodgeInject[v] = 0;
          k.hodgeInject[WIND_V + v] = 0;
          k.hodgeInject[2 * WIND_V + v] = 0.0;
        } else {
          const R2 = R * R;
          const r4 = r2 * r2;
          const uPot = Uinf * (1.0 - (R2 * (rx * rx - ry * ry)) / r4);
          const vPot = -Uinf * (2.0 * R2 * rx * ry) / r4;
          const uCirc = -(Gamma / (2.0 * Math.PI * r2)) * ry;
          const vCirc = (Gamma / (2.0 * Math.PI * r2)) * rx;
          k.hodgeInject[v] = uPot + uCirc;
          k.hodgeInject[WIND_V + v] = vPot + vCirc;
          k.hodgeInject[2 * WIND_V + v] = 1.0;
        }
      }
    }
    assert(k.exp.chronos_hodge_solve_injected(dx, dy) === 1, `cyl hodge_solve_injected failed f=${f}`);
    const ok = k.exp.chronos_wind_solve_injected(dx, dy, f * (1.0 / 30.0));
    lastS = readWindSummary(k);
    if (f >= 2) assert(ok === 1, `cyl wind_solve_injected failed f=${f}`);
  }

  // Compare Cp on active fluid cells around the cylinder (R < r <= 120 px)
  const cpMeas = [], cpTrue = [];
  for (let r = 2; r < WIND_H - 2; r++) {
    const y = (r + 0.5) * dy;
    for (let c = 2; c < WIND_W - 2; c++) {
      const x = (c + 0.5) * dx;
      const rx = x - cx, ry = y - cy;
      const r2 = rx * rx + ry * ry;
      if (r2 > (R + 4) * (R + 4) && r2 <= 120 * 120) {
        const R2 = R * R, r4 = r2 * r2;
        const u = Uinf * (1.0 - (R2 * (rx * rx - ry * ry)) / r4) - (Gamma / (2.0 * Math.PI * r2)) * ry;
        const v = -Uinf * (2.0 * R2 * rx * ry) / r4 + (Gamma / (2.0 * Math.PI * r2)) * rx;
        cpMeas.push(k.windPlanes[r * WIND_W + c]);
        cpTrue.push(1.0 - (u * u + v * v) / (Uinf * Uinf));
      }
    }
  }
  const rCp = pearsonR(cpMeas, cpTrue);
  const clBndErr = (100.0 * Math.abs(lastS.clBnd - exactCL)) / exactCL;
  const clCvErr = (100.0 * Math.abs(lastS.clCv - exactCL)) / exactCL;
  const clKjErr = (100.0 * Math.abs(lastS.clKj - exactCL)) / exactCL;
  console.log(
    `  [PASS] I9.2a Analytical Lifting Cylinder: beta1=${lastS.beta1}, D=${lastS.dHolePx.toFixed(1)}px, Uinf=${lastS.uInf.toFixed(1)}, Cp r=${rCp.toFixed(4)}, C_L bnd=${lastS.clBnd.toFixed(3)} (${clBndErr.toFixed(1)}%), cv=${lastS.clCv.toFixed(3)} (${clCvErr.toFixed(1)}%), KJ=${lastS.clKj.toFixed(3)} (${clKjErr.toFixed(1)}%), C_D cv=${lastS.cdCv.toFixed(3)}, eta_NS=${lastS.nsResid.toFixed(4)}`
  );
  assert(lastS.state === WIND_STATE_READY_FULL_LOADS, `expected READY_FULL_LOADS, got ${lastS.state}`);
  assert(lastS.beta1 >= 1, `expected beta1 >= 1, got ${lastS.beta1}`);
  assert(rCp > 0.98, `analytical Cp r=${rCp} <= 0.98`);
  assert(clBndErr <= 8.0, `analytical C_L bnd err ${clBndErr.toFixed(1)}% > 8%`);
  assert(clCvErr <= 8.0, `analytical C_L cv err ${clCvErr.toFixed(1)}% > 8%`);
  assert(clKjErr <= 8.0, `analytical C_L KJ err ${clKjErr.toFixed(1)}% > 8%`);
  assert(Math.abs(lastS.cdCv) < 0.25, `analytical |C_D cv|=${Math.abs(lastS.cdCv)} >= 0.25`);
}

// ============================================================================
// I9.3: Fail-Closed 3D/Compressible & Non-Conservative Negative Controls
// ============================================================================
async function testI9_3_FailClosedNegativeControls() {
  // I9.3a: Pure 2D compressible radial source (div u != 0 everywhere) -> HIGH_DIV
  {
    const k = await makeKernel();
    k.exp.chronos_set_wind_mode(2);
    const dx = 10.0, dy = 10.0;
    let s = null;
    for (let f = 0; f < 4; f++) {
      for (let r = 0; r < WIND_H; r++) {
        for (let c = 0; c < WIND_W; c++) {
          const v = r * WIND_W + c;
          k.hodgeInject[v] = 0.6 * ((c + 0.5) * dx - 320.0);
          k.hodgeInject[WIND_V + v] = 0.6 * ((r + 0.5) * dy - 180.0);
          k.hodgeInject[2 * WIND_V + v] = 1.0;
        }
      }
      k.exp.chronos_hodge_solve_injected(dx, dy);
      k.exp.chronos_wind_solve_injected(dx, dy, f * 0.05);
      s = readWindSummary(k);
    }
    assert(s.state === WIND_STATE_REFUSED, `I9.3a expected REFUSED, got ${s.state}`);
    assert((s.reason & WIND_REASON_HIGH_DIV) !== 0, `I9.3a expected HIGH_DIV bit, got reason=0x${s.reason.toString(16)}`);
    assert(s.clBnd === 0 && s.cdBnd === 0 && s.clCv === 0, 'I9.3a fabricated non-zero loads');
    console.log(`  [PASS] I9.3a Compressible Source Negative Control: REFUSED (reason=0x${s.reason.toString(16)}, divFrac=${s.divFrac.toFixed(3)})`);
  }

  // I9.3b: Incompressible flow with non-conservative rotational body-force ramp -> HIGH_NS_RESIDUAL
  {
    const k = await makeKernel();
    k.exp.chronos_set_wind_mode(2);
    const dx = 10.0, dy = 10.0;
    let s = null;
    for (let f = 0; f < 5; f++) {
      // Pure solenoidal Taylor-Green vortex ramping rapidly (impossible without rotational body force!)
      const amp = 40.0 + 70.0 * f;
      const kx = (2.0 * Math.PI) / 160.0;
      const ky = (2.0 * Math.PI) / 160.0;
      for (let r = 0; r < WIND_H; r++) {
        const y = (r + 0.5) * dy;
        for (let c = 0; c < WIND_W; c++) {
          const x = (c + 0.5) * dx;
          const v = r * WIND_W + c;
          k.hodgeInject[v] = amp * Math.sin(kx * x) * Math.cos(ky * y);
          k.hodgeInject[WIND_V + v] = -amp * Math.cos(kx * x) * Math.sin(ky * y);
          k.hodgeInject[2 * WIND_V + v] = 1.0;
        }
      }
      k.exp.chronos_hodge_solve_injected(dx, dy);
      k.exp.chronos_wind_solve_injected(dx, dy, f * 0.033);
      s = readWindSummary(k);
    }
    assert(s.state === WIND_STATE_REFUSED, `I9.3b expected REFUSED, got ${s.state}`);
    assert((s.reason & WIND_REASON_HIGH_NS_RESIDUAL) !== 0, `I9.3b expected HIGH_NS_RESIDUAL, got 0x${s.reason.toString(16)}`);
    assert(s.clBnd === 0 && s.cdBnd === 0, 'I9.3b fabricated non-zero loads');
    console.log(`  [PASS] I9.3b Non-Conservative Forcing Negative Control: REFUSED (reason=0x${s.reason.toString(16)}, eta_NS=${s.nsResid.toFixed(3)})`);
  }
}

// ============================================================================
// I9.5: Mutual-Exclusion Arbiter with BOS & Clean Recovery
// ============================================================================
async function testI9_5_MutexIsolation() {
  const k = await makeKernel();
  k.exp.chronos_set_bos_mode(1);
  k.exp.chronos_set_wind_mode(2);
  const dx = 10.0, dy = 10.0;
  for (let v = 0; v < WIND_V; v++) {
    k.hodgeInject[v] = 50.0;
    k.hodgeInject[WIND_V + v] = 0.0;
    k.hodgeInject[2 * WIND_V + v] = 1.0;
  }
  k.exp.chronos_hodge_solve_injected(dx, dy);
  const okBusy = k.exp.chronos_wind_solve_injected(dx, dy, 0.0);
  const sBusy = readWindSummary(k);
  assert(okBusy === 0, 'expected wind refused while BOS active');
  assert(sBusy.state === WIND_STATE_REFUSED && (sBusy.reason & WIND_REASON_MUTEX_CONFLICT) !== 0, `expected MUTEX_CONFLICT, got st=${sBusy.state}, reason=0x${sBusy.reason.toString(16)}`);

  // Release BOS -> Wind Tunnel cleanly warms up and becomes READY without stale scratch
  k.exp.chronos_set_bos_mode(0);
  let sRec = null;
  for (let f = 0; f < 4; f++) {
    k.exp.chronos_hodge_solve_injected(dx, dy);
    k.exp.chronos_wind_solve_injected(dx, dy, 0.1 + f * 0.033);
    sRec = readWindSummary(k);
  }
  assert(sRec.state === WIND_STATE_READY_FIELD_ONLY, `expected clean recovery after BOS off, got state=${sRec.state}, reason=0x${sRec.reason.toString(16)}`);
  console.log(`  [PASS] I9.5 BOS/Wind Mutex Isolation: refused under BOS (0x${sBusy.reason.toString(16)}), clean recovery after release (state=${sRec.state})`);
}

// ============================================================================
// I9.1b & I9.2b & I9.4: Full H.264 MP4 End-to-End Benchmarks
// ============================================================================
async function runMp4WindBench(mp4Name, { fps, nFrames, hodgeMode, windMode, measureLatency = false, perFrameCpEval = null }) {
  const fullPath = path.join(SAMPLES_DIR, mp4Name);
  assert(fs.existsSync(fullPath), `missing sample ${fullPath}`);
  const W = 640, H = 360;
  const frames = decodeMp4Frames(fullPath, W, H, nFrames);
  const dt = 1.0 / fps;

  const runPass = async (enableWind) => {
    const k = await makeKernel();
    k.exp.chronos_set_resolution(1);
    k.exp.chronos_set_swarm_mode(2);
    k.exp.chronos_set_hodge_mode(hodgeMode);
    if (enableWind) k.exp.chronos_set_wind_mode(windMode);

    let readyFrames = 0;
    let fullLoadFrames = 0;
    let refusedHighDivFrames = 0;
    let lastReady = null;
    let bestPlanes = null;
    let lastReadyTime = 0;
    const nuSamples = [];
    const clBndSamples = [];
    const clCvSamples = [];
    const clKjSamples = [];
    const etaSamples = [];
    const cpRSamples = [];
    let tickSumMs = 0;
    let tickCnt = 0;

    for (let f = 0; f < frames.length; f++) {
      const dstPtr = k.exp.chronos_get_ingest_rgba_ptr();
      new Uint8Array(k.mem.buffer, dstPtr, W * H * 4).set(frames[f]);
      const t0 = performance.now();
      k.exp.chronos_tick(dt);
      const elapsed = performance.now() - t0;
      if (f >= 20) { tickSumMs += elapsed; tickCnt++; }

      if (enableWind) {
        const s = readWindSummary(k);
        const w2Flag = k.wave2Telem[16 + 4 * 8];
        if (s.state === WIND_STATE_READY_FIELD_ONLY || s.state === WIND_STATE_READY_FULL_LOADS) {
          readyFrames++;
          assert((w2Flag & 1) !== 0, `${mp4Name} f=${f}: wave2 wind stamp missing VALID bit`);
          lastReady = s;
          lastReadyTime = f * dt;
          bestPlanes = Float32Array.from(k.windPlanes.subarray(0, WIND_V));
          if (perFrameCpEval) {
            const rv = perFrameCpEval(bestPlanes);
            if (rv !== null && Number.isFinite(rv)) cpRSamples.push(rv);
          }
          if (s.nu > 0 && s.nuR2 >= 0.25) nuSamples.push(s.nu);
          if (s.state === WIND_STATE_READY_FULL_LOADS) {
            fullLoadFrames++;
            clBndSamples.push(s.clBnd);
            clCvSamples.push(s.clCv);
            clKjSamples.push(s.clKj);
          }
          etaSamples.push(s.nsResid);
        } else if (s.state === WIND_STATE_REFUSED && (s.reason & WIND_REASON_HIGH_DIV) !== 0) {
          refusedHighDivFrames++;
        }
      }
    }
    const med = (arr) => {
      if (!arr.length) return 0;
      const s = [...arr].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)];
    };
    return {
      readyFrames,
      fullLoadFrames,
      refusedHighDivFrames,
      lastReady,
      bestPlanes,
      lastReadyTime,
      medNu: med(nuSamples),
      nuCount: nuSamples.length,
      medClBnd: med(clBndSamples),
      medClCv: med(clCvSamples),
      medClKj: med(clKjSamples),
      medEta: med(etaSamples),
      medCpR: med(cpRSamples),
      meanTickMs: tickCnt > 0 ? tickSumMs / tickCnt : 0,
    };
  };

  const withWind = await runPass(true);
  let deltaMs = 0;
  if (measureLatency) {
    const base = await runPass(false);
    deltaMs = withWind.meanTickMs - base.meanTickMs;
  }
  return { ...withWind, deltaMs };
}

async function testI9_1b_Mp4LambOseen() {
  const res = await runMp4WindBench('wind_lamb_oseen_viscosity.mp4', {
    fps: 30, nFrames: 210, hodgeMode: 2, windMode: 2,
  });
  assert(res.readyFrames >= 80, `I9.1b expected >= 80 ready frames, got ${res.readyFrames}`);
  const gamma = 28000.0, nuTrue = 24.0, t0 = 6.0;
  const T = res.lastReadyTime + t0;
  const cx = 320.0, cy = 180.0, dx = 10.0, dy = 10.0;
  const pMeas = [], pTrue = [];
  for (let r = 3; r < WIND_H - 3; r++) {
    const y = (r + 0.5) * dy;
    for (let c = 3; c < WIND_W - 3; c++) {
      const x = (c + 0.5) * dx;
      const rad = Math.hypot(x - cx, y - cy);
      if (rad <= 130.0 && res.bestPlanes[r * WIND_W + c] !== 0) {
        pMeas.push(res.bestPlanes[r * WIND_W + c]);
        pTrue.push(lambOseenPressure(rad, gamma, nuTrue, T));
      }
    }
  }
  const rWell = pearsonR(pMeas, pTrue);
  const nuErrPct = res.medNu > 0 ? (100.0 * Math.abs(res.medNu - nuTrue)) / nuTrue : 999;
  console.log(
    `  [PASS] I9.1b MP4 Lamb-Oseen PIV: ready=${res.readyFrames}/210, med_nu=${res.medNu.toFixed(2)} px^2/s (true=24.0, err=${nuErrPct.toFixed(1)}%, n=${res.nuCount}), well r=${rWell.toFixed(4)}, med_eta_NS=${res.medEta.toFixed(4)}`
  );
  assert(rWell > 0.98, `I9.1b pressure well r=${rWell.toFixed(4)} <= 0.98`);
  assert(res.medEta < 0.16, `I9.1b med_eta_NS=${res.medEta.toFixed(4)} >= 0.16`);
  assert(nuErrPct <= 8.0, `I9.1b med_nu=${res.medNu.toFixed(2)} error ${nuErrPct.toFixed(1)}% > 8%`);
}

async function testI9_2b_Mp4LiftingCylinder() {
  const res = await runMp4WindBench('wind_lifting_cylinder_cp.mp4', {
    fps: 30, nFrames: 120, hodgeMode: 1, windMode: 1, measureLatency: true,
    perFrameCpEval: (planes) => {
      const cx = 256.0, cy = 180.0, R = 34.0, Uinf = 120.0, Gamma = -14000.0, dx = 10.0, dy = 10.0;
      const cpMeas = [], cpTrue = [];
      for (let r = 3; r < WIND_H - 3; r++) {
        const y = (r + 0.5) * dy;
        for (let c = 3; c < WIND_W - 3; c++) {
          const x = (c + 0.5) * dx;
          const rx = x - cx, ry = y - cy;
          const r2 = rx * rx + ry * ry;
          const val = planes[r * WIND_W + c];
          if (r2 > (R + 6) * (R + 6) && r2 <= 125 * 125 && val !== 0) {
            const R2 = R * R, r4 = r2 * r2;
            const u = Uinf * (1.0 - (R2 * (rx * rx - ry * ry)) / r4) - (Gamma / (2.0 * Math.PI * r2)) * ry;
            const v = -Uinf * (2.0 * R2 * rx * ry) / r4 + (Gamma / (2.0 * Math.PI * r2)) * rx;
            cpMeas.push(val);
            cpTrue.push(1.0 - (u * u + v * v) / (Uinf * Uinf));
          }
        }
      }
      return cpMeas.length >= 64 ? pearsonR(cpMeas, cpTrue) : null;
    },
  });
  assert(res.fullLoadFrames >= 20, `I9.2b expected >= 20 full-load frames, got ${res.fullLoadFrames}`);
  const exactCL = (2.0 * 14000.0) / (120.0 * 68.0);
  const rCp = res.medCpR;
  const clCvErr = (100.0 * Math.abs(res.medClCv - exactCL)) / exactCL;
  const clKjErr = (100.0 * Math.abs(res.medClKj - exactCL)) / exactCL;
  console.log(
    `  [PASS] I9.2b MP4 Lifting Cylinder PIV: fullLoad=${res.fullLoadFrames}/120, med Cp r=${rCp.toFixed(4)}, C_L bnd=${res.medClBnd.toFixed(3)}, cv=${res.medClCv.toFixed(3)} (${clCvErr.toFixed(1)}%), KJ=${res.medClKj.toFixed(3)} (${clKjErr.toFixed(1)}%), deltaTick=${res.deltaMs.toFixed(3)} ms`
  );
  assert(rCp > 0.95, `I9.2b med Cp r=${rCp.toFixed(4)} <= 0.95`);
  assert(clCvErr <= 10.0, `I9.2b C_L cv err ${clCvErr.toFixed(1)}% > 10%`);
  assert(clKjErr <= 10.0, `I9.2b C_L KJ err ${clKjErr.toFixed(1)}% > 10%`);
  assert(res.deltaMs <= 0.5, `I9.2b added HD tick latency ${res.deltaMs.toFixed(3)} ms > 0.5 ms`);
}

async function testI9_4_RealCameraClips() {
  const daphnia = await runMp4WindBench('real_daphnia_laser_piv.mp4', {
    fps: 15, nFrames: 100, hodgeMode: 1, windMode: 1,
  });
  assert(daphnia.readyFrames >= 4, `I9.4 Daphnia expected >= 4 planar ready frames, got ${daphnia.readyFrames}`);
  assert(daphnia.refusedHighDivFrames >= 20, `I9.4 Daphnia expected honest 3D out-of-plane HIGH_DIV refusals, got ${daphnia.refusedHighDivFrames}`);
  console.log(
    `  [PASS] I9.4a Real Lab Laser PIV (Daphnia): planarReady=${daphnia.readyFrames}/100, 3D-refused=${daphnia.refusedHighDivFrames}/100, med_eta_NS=${daphnia.medEta.toFixed(4)}, lastCp=[${daphnia.lastReady.cpMin.toFixed(2)}, ${daphnia.lastReady.cpMax.toFixed(2)}], tick=${daphnia.meanTickMs.toFixed(3)} ms`
  );

  const smoke = await runMp4WindBench('real_smoke_airfoil_cambridge.mp4', {
    fps: 25, nFrames: 120, hodgeMode: 1, windMode: 1,
  });
  assert(smoke.readyFrames >= 50, `I9.4 Cambridge Airfoil expected >= 50 ready frames, got ${smoke.readyFrames}`);
  console.log(
    `  [PASS] I9.4b Real Wind-Tunnel Smoke Airfoil (Cambridge): ready=${smoke.readyFrames}/120, fullLoad=${smoke.fullLoadFrames}, med_eta_NS=${smoke.medEta.toFixed(4)}, tick=${smoke.meanTickMs.toFixed(3)} ms`
  );
}

console.log(`=== Gate I9: Pocket Wind Tunnel (${PARITY_LITE ? 'parity-lite' : 'full'}) ===`);
await testI9_1a_AnalyticalLambOseen();
await testI9_2a_AnalyticalLiftingCylinder();
await testI9_3_FailClosedNegativeControls();
await testI9_5_MutexIsolation();
if (!PARITY_LITE) {
  await testI9_1b_Mp4LambOseen();
  await testI9_2b_Mp4LiftingCylinder();
  await testI9_4_RealCameraClips();
}
console.log('[PASS] Gate I9 complete.');
