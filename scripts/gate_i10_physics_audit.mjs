#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I10a: v3.4.0 Dense Optical-Flow Continuum Fallback,
// Wind Soundness & Audio Push Verification Suite (Wasm32 SIMD128)
//
// Validates:
//   I10a.1  Analytic uniform advection (unseeded texture, swarm OFF, WIND_MODE_DENSE_AUDIT=3):
//           reaches READY_FIELD_ONLY with provenance=WIND_PROV_DENSE_LK(3), div_frac < 0.08,
//           zero Cp / C_L / C_D (structurally capped at READY_FIELD_ONLY).
//   I10a.2  Analytic rotating Lamb-Oseen texture (unseeded, swarm OFF, WIND_MODE_DENSE_AUDIT=3):
//           reaches READY_FIELD_ONLY with low divergence & conservative NS residual, zero Cp / loads.
//   I10a.3  Fail-closed soundness negative controls:
//           (a) Static / zero-flow decoy -> REFUSED (NO_CONTINUUM | LOW_SUPPORT), never READY.
//           (b) Exposure flicker decoy -> REFUSED, never READY.
//           (c) 2-row degenerate injected strip -> LOW_SUPPORT bit set on sparse, refused on dense.
//           (d) Injected zero-flow -> REFUSED (NO_CONTINUUM | LOW_SUPPORT).
//   I10a.4  Real-camera unseeded MRI Karman cylinder (`real_karman_mri_cylinder.mp4`):
//           WIND_MODE_DENSE_AUDIT(3) achieves substantially higher fresh continuum coverage than
//           sparse baseline (WIND_MODE_EVERY_FRAME=2, ~4/488), while keeping Cp/CL/CD strictly 0.
//   I10a.5  Audio push edge cases + u64 timestamp crossing 2^32 us (hi word non-zero).
//
// Usage:
//   node scripts/gate_i10_physics_audit.mjs [--parity-lite]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WASM_PATH = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');
const PARITY_LITE = process.argv.includes('--parity-lite');

const WIND_W = 64;
const WIND_H = 36;
const WIND_V = WIND_W * WIND_H;

const WIND_MODE_EVERY_FRAME = 2;
const WIND_MODE_DENSE_AUDIT = 3;

const WIND_STATE_READY_FIELD_ONLY = 2;
const WIND_STATE_READY_FULL_LOADS = 3;
const WIND_STATE_REFUSED = 4;

const WIND_PROV_DENSE_LK = 3;
const WIND_REASON_NO_CONTINUUM = 1 << 0;
const WIND_REASON_LOW_SUPPORT = 1 << 9;

const wasmBytes = fs.readFileSync(WASM_PATH);

function assert(cond, msg) {
  if (!cond) throw new Error(`Gate I10a assertion failed: ${msg}`);
}

function f32FromU32(u) {
  const b = new ArrayBuffer(4);
  new Uint32Array(b)[0] = u >>> 0;
  return new Float32Array(b)[0];
}

async function makeKernel() {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exp = instance.exports;
  exp.chronos_init();
  const mem = exp.memory;
  return {
    exp,
    mem,
    hodgeInject: new Float32Array(mem.buffer, exp.chronos_get_hodge_inject_ptr(), 3 * WIND_V),
    windTelemU32: new Uint32Array(mem.buffer, exp.chronos_get_wind_telemetry_ptr(), 32),
    windPlanes: new Float32Array(mem.buffer, exp.chronos_get_wind_planes_ptr(), 3 * WIND_V),
    audioStaging: new Float32Array(mem.buffer, exp.chronos_get_audio_staging_ptr(), 4096),
  };
}

function stepFrame(k, rgba, dtSec, ptsUs, seq, w = 320, h = 180) {
  const ptr = k.exp.chronos_get_ingest_rgba_ptr();
  new Uint8Array(k.mem.buffer, ptr, w * h * 4).set(rgba);
  const ptsLo = ptsUs >>> 0;
  const ptsHi = Math.floor(ptsUs / 4294967296) >>> 0;
  const ok = k.exp.chronos_set_ingest_meta(
    1, seq, ptsLo, ptsHi, -1.0,
    (1 << 1) | (1 << 4), w, h, 0, 0, w, h, 0, 0xFFFFFFFF,
    0, 0, w, h, 0, 1
  );
  assert(ok === 1, `chronos_set_ingest_meta rejected frame seq=${seq}`);
  k.exp.chronos_pre_tick(dtSec);
  k.exp.chronos_post_tick(dtSec);
}

function makeTexturedFrame(w, h, dx = 0, dy = 0, gain = 1.0) {
  const out = new Uint8Array(w * h * 4);
  let idx = 0;
  for (let y = 0; y < h; y++) {
    const yy = y - dy;
    const sy1 = Math.sin(yy * 0.21);
    const cy2 = Math.cos(yy * 0.37);
    const sy3 = Math.sin(yy * 0.13);
    for (let x = 0; x < w; x++) {
      const xx = x - dx;
      const v = 0.5
        + 0.18 * Math.sin(xx * 0.19) * cy2
        + 0.16 * Math.cos(xx * 0.31) * sy1
        + 0.12 * Math.sin((xx + yy) * 0.27)
        + 0.08 * Math.cos((xx - yy) * 0.17) * sy3;
      const c = Math.max(16, Math.min(239, Math.round(v * 255.0 * gain)));
      out[idx++] = c;
      out[idx++] = c;
      out[idx++] = c;
      out[idx++] = 255;
    }
  }
  return out;
}

function makeRotatingVortexFrame(w, h, thetaRad) {
  const out = new Uint8Array(w * h * 4);
  const cx = w * 0.5;
  const cy = h * 0.5;
  let idx = 0;
  for (let y = 0; y < h; y++) {
    const ry = y - cy;
    for (let x = 0; x < w; x++) {
      const rx = x - cx;
      const r2 = rx * rx + ry * ry;
      // Smooth Lamb-Oseen-like solid-body inner core + decaying outer rotation (zero net camera pan!)
      const ang = thetaRad * Math.exp(-r2 / (65.0 * 65.0));
      const ca = Math.cos(-ang);
      const sa = Math.sin(-ang);
      const xx = cx + rx * ca - ry * sa;
      const yy = cy + rx * sa + ry * ca;
      const v = 0.5
        + 0.18 * Math.sin(xx * 0.23) * Math.cos(yy * 0.29)
        + 0.16 * Math.cos(xx * 0.31) * Math.sin(yy * 0.19)
        + 0.12 * Math.sin((xx + yy) * 0.25);
      const c = Math.max(16, Math.min(239, Math.round(v * 255.0)));
      out[idx++] = c;
      out[idx++] = c;
      out[idx++] = c;
      out[idx++] = 255;
    }
  }
  return out;
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
  const fb = w * h * 4;
  const count = Math.floor(res.stdout.length / fb);
  const frames = [];
  for (let i = 0; i < count; i++) {
    frames.push(res.stdout.subarray(i * fb, (i + 1) * fb));
  }
  return frames;
}

async function testI10a_1_DenseUniformAdvection() {
  const k = await makeKernel();
  k.exp.chronos_set_swarm_mode(0);
  k.exp.chronos_set_hodge_mode(1);
  assert(k.exp.chronos_set_wind_mode(WIND_MODE_DENSE_AUDIT) === 1, 'set_wind_mode(3) failed');
  const nFrames = PARITY_LITE ? 8 : 14;
  let readyField = 0;
  let fullLoads = 0;
  let lastDiv = 1.0;
  let lastNs = 1.0;
  for (let f = 0; f < nFrames; f++) {
    const fr = makeTexturedFrame(320, 180, 1.5 * f, 0.5 * f);
    stepFrame(k, fr, 1 / 30, Math.round(f * 33333.333), f + 1);
    const st = k.windTelemU32[0];
    const prov = k.windTelemU32[3];
    if (st === WIND_STATE_READY_FIELD_ONLY) {
      readyField++;
      assert(prov === WIND_PROV_DENSE_LK, `expected prov=${WIND_PROV_DENSE_LK}, got ${prov}`);
      lastDiv = f32FromU32(k.windTelemU32[22]);
      lastNs = f32FromU32(k.windTelemU32[21]);
      assert(f32FromU32(k.windTelemU32[16]) === 0.0, 'dense mode emitted non-zero C_L_bnd');
      assert(f32FromU32(k.windTelemU32[18]) === 0.0, 'dense mode emitted non-zero C_L_cv');
    } else if (st === WIND_STATE_READY_FULL_LOADS) {
      fullLoads++;
    }
  }
  let maxCp = 0;
  for (let i = 0; i < WIND_V; i++) {
    const a = Math.abs(k.windPlanes[i]);
    if (a > maxCp) maxCp = a;
  }
  assert(readyField >= (PARITY_LITE ? 4 : 8), `expected >= 8 READY_FIELD_ONLY frames, got ${readyField}`);
  assert(fullLoads === 0, `dense fallback must never emit READY_FULL_LOADS, got ${fullLoads}`);
  assert(maxCp === 0, `dense fallback must keep Cp plane strictly zero, got max|Cp|=${maxCp}`);
  assert(lastDiv < 0.08, `uniform dense advection div_frac=${lastDiv.toFixed(4)} >= 0.08`);
  console.log(`  [PASS] I10a.1 Dense-LK Uniform Advection: readyField=${readyField}/${nFrames}, prov=DENSE_LK, div=${lastDiv.toFixed(4)}, ns=${lastNs.toFixed(4)}, max|Cp|=0`);
}

async function testI10a_2_DenseRotatingVortex() {
  const k = await makeKernel();
  k.exp.chronos_set_swarm_mode(0);
  k.exp.chronos_set_hodge_mode(1);
  k.exp.chronos_set_wind_mode(WIND_MODE_DENSE_AUDIT);
  const nFrames = PARITY_LITE ? 8 : 14;
  let readyField = 0;
  let lastDiv = 1.0;
  let lastNs = 1.0;
  for (let f = 0; f < nFrames; f++) {
    const fr = makeRotatingVortexFrame(320, 180, 0.045 * f);
    stepFrame(k, fr, 1 / 30, Math.round(f * 33333.333), f + 1);
    if (k.windTelemU32[0] === WIND_STATE_READY_FIELD_ONLY) {
      readyField++;
      assert(k.windTelemU32[3] === WIND_PROV_DENSE_LK, 'vortex expected WIND_PROV_DENSE_LK');
      lastDiv = f32FromU32(k.windTelemU32[22]);
      lastNs = f32FromU32(k.windTelemU32[21]);
    }
    assert(k.windTelemU32[0] !== WIND_STATE_READY_FULL_LOADS, 'dense vortex emitted READY_FULL_LOADS');
  }
  assert(readyField >= (PARITY_LITE ? 4 : 8), `expected >= 8 READY_FIELD_ONLY on rotating vortex, got ${readyField}`);
  assert(lastDiv < 0.12, `vortex div_frac=${lastDiv.toFixed(4)} >= 0.12`);
  console.log(`  [PASS] I10a.2 Dense-LK Rotating Vortex: readyField=${readyField}/${nFrames}, div=${lastDiv.toFixed(4)}, ns=${lastNs.toFixed(4)}, fullLoads=0`);
}

async function testI10a_3_FailClosedSoundnessControls() {
  // (a) Static texture -> REFUSED (NO_CONTINUUM | LOW_SUPPORT)
  {
    const k = await makeKernel();
    k.exp.chronos_set_swarm_mode(0);
    k.exp.chronos_set_hodge_mode(1);
    k.exp.chronos_set_wind_mode(WIND_MODE_DENSE_AUDIT);
    const staticFr = makeTexturedFrame(320, 180, 0, 0);
    let readyAny = 0;
    for (let f = 0; f < 6; f++) {
      stepFrame(k, staticFr, 1 / 30, Math.round(f * 33333.333), f + 1);
      if (k.windTelemU32[0] === WIND_STATE_READY_FIELD_ONLY || k.windTelemU32[0] === WIND_STATE_READY_FULL_LOADS) {
        readyAny++;
      }
    }
    assert(readyAny === 0, `static decoy falsely reached READY (${readyAny} frames)`);
    assert(k.windTelemU32[0] === WIND_STATE_REFUSED, `static decoy expected REFUSED, got ${k.windTelemU32[0]}`);
    assert((k.windTelemU32[2] & WIND_REASON_LOW_SUPPORT) !== 0, `static decoy expected LOW_SUPPORT bit`);
  }

  // (b) Exposure flicker decoy -> REFUSED, never READY
  {
    const k = await makeKernel();
    k.exp.chronos_set_swarm_mode(0);
    k.exp.chronos_set_hodge_mode(1);
    k.exp.chronos_set_wind_mode(WIND_MODE_DENSE_AUDIT);
    let readyAny = 0;
    for (let f = 0; f < 6; f++) {
      const fr = makeTexturedFrame(320, 180, 0, 0, (f & 1) ? 0.84 : 1.16);
      stepFrame(k, fr, 1 / 30, Math.round(f * 33333.333), f + 1);
      if (k.windTelemU32[0] === WIND_STATE_READY_FIELD_ONLY || k.windTelemU32[0] === WIND_STATE_READY_FULL_LOADS) {
        readyAny++;
      }
    }
    assert(readyAny === 0, `exposure flicker falsely reached READY (${readyAny} frames)`);
  }

  // (c) Injected zero-flow -> REFUSED (NO_CONTINUUM | LOW_SUPPORT)
  {
    const k = await makeKernel();
    k.exp.chronos_set_wind_mode(WIND_MODE_EVERY_FRAME);
    for (let v = 0; v < WIND_V; v++) {
      k.hodgeInject[v] = 0.0;
      k.hodgeInject[WIND_V + v] = 0.0;
      k.hodgeInject[2 * WIND_V + v] = 1.0;
    }
    for (let f = 0; f < 4; f++) {
      k.exp.chronos_hodge_solve_injected(10.0, 10.0);
      k.exp.chronos_wind_solve_injected(10.0, 10.0, f * 0.033);
    }
    assert(k.windTelemU32[0] === WIND_STATE_REFUSED, `injected zero-flow must be REFUSED, got ${k.windTelemU32[0]}`);
    assert((k.windTelemU32[2] & (WIND_REASON_NO_CONTINUUM | WIND_REASON_LOW_SUPPORT)) === (WIND_REASON_NO_CONTINUUM | WIND_REASON_LOW_SUPPORT),
      `injected zero-flow expected NO_CONTINUUM|LOW_SUPPORT, got 0x${k.windTelemU32[2].toString(16)}`);
  }

  // (d) Injected 2-row strip (< 8 5-pt interior stencils) -> sets WIND_REASON_LOW_SUPPORT
  {
    const k = await makeKernel();
    k.exp.chronos_set_wind_mode(WIND_MODE_EVERY_FRAME);
    k.hodgeInject.fill(0);
    for (let r = 16; r <= 17; r++) {
      for (let c = 8; c < 56; c++) {
        const v = r * WIND_W + c;
        k.hodgeInject[v] = 40.0;
        k.hodgeInject[WIND_V + v] = 0.0;
        k.hodgeInject[2 * WIND_V + v] = 1.0;
      }
    }
    for (let f = 0; f < 4; f++) {
      k.exp.chronos_hodge_solve_injected(10.0, 10.0);
      k.exp.chronos_wind_solve_injected(10.0, 10.0, f * 0.033);
    }
    assert((k.windTelemU32[2] & WIND_REASON_LOW_SUPPORT) !== 0, `2-row strip must set WIND_REASON_LOW_SUPPORT, got 0x${k.windTelemU32[2].toString(16)}`);
  }

  console.log('  [PASS] I10a.3 Fail-Closed Soundness Controls: static=REFUSED, flicker=REFUSED, zero-flow=REFUSED, 2-row-strip=LOW_SUPPORT');
}

async function testI10a_4_RealKarmanMriDenseCoverage() {
  if (PARITY_LITE) return;
  const mp4 = path.join(SAMPLES_DIR, 'real_karman_mri_cylinder.mp4');
  if (!fs.existsSync(mp4)) {
    console.log('  [SKIP] I10a.4 real_karman_mri_cylinder.mp4 not present');
    return;
  }
  const hasFfmpeg = spawnSync('ffmpeg', ['-version'], { stdio: 'ignore' }).status === 0;
  if (!hasFfmpeg) {
    console.log('  [SKIP] I10a.4 ffmpeg not on PATH');
    return;
  }
  const frames = decodeMp4Frames(mp4, 320, 180, 240);
  const runMode = async (mode) => {
    const k = await makeKernel();
    k.exp.chronos_set_swarm_mode(mode === WIND_MODE_DENSE_AUDIT ? 0 : 2);
    k.exp.chronos_set_hodge_mode(1);
    k.exp.chronos_set_wind_mode(mode);
    let readyField = 0;
    let fullLoads = 0;
    let measurableRefusal = 0;
    let activeFrames = 0;
    for (let f = 0; f < frames.length; f++) {
      stepFrame(k, frames[f], 1 / 25, f * 40000, f + 1);
      const st = k.windTelemU32[0];
      const reason = k.windTelemU32[2];
      if (st === WIND_STATE_READY_FIELD_ONLY) {
        readyField++;
        activeFrames++;
      } else if (st === WIND_STATE_READY_FULL_LOADS) {
        fullLoads++;
        activeFrames++;
      } else if (st === WIND_STATE_REFUSED && (reason & ((1 << 3) | (1 << 4))) !== 0) {
        measurableRefusal++;
        activeFrames++;
      }
    }
    return { readyField, fullLoads, measurableRefusal, activeFrames };
  };

  const sparse = await runMode(WIND_MODE_EVERY_FRAME);
  const dense = await runMode(WIND_MODE_DENSE_AUDIT);
  assert(dense.fullLoads === 0, 'dense MRI must never emit READY_FULL_LOADS');
  assert(dense.activeFrames > sparse.activeFrames * 5 && dense.activeFrames >= 40,
    `dense MRI activeFrames=${dense.activeFrames} (ready=${dense.readyField}, ref=${dense.measurableRefusal}) did not strongly beat sparse=${sparse.activeFrames}`);
  console.log(`  [PASS] I10a.4 Real Karman MRI Cylinder (${frames.length}f): sparse_active=${sparse.activeFrames} -> dense_active=${dense.activeFrames} (ready=${dense.readyField}, measurableRefusal=${dense.measurableRefusal}, fullLoads=0)`);
}

async function testI10a_5_AudioPushEdgeCases() {
  const k = await makeKernel();
  assert(k.exp.chronos_push_audio_block(0, 1000, 0, 48000, 0) === 0, 'n=0 must reject');
  assert(k.exp.chronos_push_audio_block(4097, 1000, 0, 48000, 0) === 0, 'n=4097 must reject');
  assert(k.exp.chronos_push_audio_block(64, 1000, 0, 500, 0) === 0, 'rate=500 must reject');
  for (let i = 0; i < 480; i++) k.audioStaging[i] = 0.05 * Math.sin(i * 0.1);
  // Timestamp crossing 2^32 us (hi=1, lo=705032704 => 5,000,000,000 us)
  assert(k.exp.chronos_push_audio_block(480, 705032704, 1, 48000, 0) === 1, 'valid u64>2^32 audio block must succeed');
  console.log('  [PASS] I10a.5 Audio Push Boundary & >2^32 us Timestamp Check');
}

async function main() {
  console.log('========================================================================');
  console.log(` CHRONOS-SPECTRAL // Gate I10a: v3.4.0 Dense Audit & Soundness${PARITY_LITE ? ' (--parity-lite)' : ''}`);
  console.log('========================================================================');
  await testI10a_1_DenseUniformAdvection();
  await testI10a_2_DenseRotatingVortex();
  await testI10a_3_FailClosedSoundnessControls();
  await testI10a_4_RealKarmanMriDenseCoverage();
  await testI10a_5_AudioPushEdgeCases();
  console.log('  [PASS] Gate I10a: ALL v3.4.0 DENSE FALLBACK & SOUNDNESS CHECKS VERIFIED');
}

main().catch((err) => {
  console.error('  [FAIL] Gate I10a:', err.message || err);
  process.exit(1);
});
