#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I2: Calibrated Rolling-Shutter Vibrometry & Welch Coherence
//
// Verifies:
//   1. Source & binary hygiene (zero em dash, zero banned tags, 448-page arena,
//      zero overlap, wave2_telem[0..3] & vibro_telem[22..32] ABI invariants)
//   2. Exposure sinc H(f) precision & null detection (H(0)=1, 880 Hz @ 1ms ~ 0.1329,
//      880 Hz @ 1/880s null -> RS_REASON_EXPOSURE_NULL + bounded amplitude)
//   3. Sub-frame rolling-shutter vibrometry (240 Hz & 421.875 Hz @ 30 fps across
//      inter-frame deadtime gaps, signed displacement flip, D4 + scan_dir + crop)
//   4. K = 1 vs K >= 8 Welch coherence trap (telem[9] > 0.95 vs telem[6] < gamma2_crit)
//   5. Localized acoustic-optical coupling (bands 4..8 coupled vs quiet margins)
//   6. Fail-closed matrix & continuity resets (bad calib, scan_dir, native_1to1,
//      missing PTS, axis transposed, clock skew, backward/gapped audio, scene cut)
//
// Usage:
//   node scripts/gate_i2_rs_welch.mjs [--wasm=<path>]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const wasmArg = process.argv.slice(2).find((a) => a.startsWith('--wasm='));
const WASM_PATH = wasmArg ? path.resolve(wasmArg.slice('--wasm='.length)) : DEFAULT_WASM;

const FRAME_W = 640;
const FRAME_H = 360;
const FRAME_PIXELS = FRAME_W * FRAME_H;
const FRAME_BYTES = FRAME_PIXELS * 4;

const RS_STATE_OFF = 0;
const RS_STATE_UNCALIBRATED = 1;
const RS_STATE_WARMING = 2;
const RS_STATE_READY = 3;
const RS_STATE_DISTURBED = 4;

const RS_MODE_OFF = 0;
const RS_MODE_RS_ONLY = 1;
const RS_MODE_RS_WELCH = 2;

const RS_REASON_BAD_CALIB = 1 << 0;
const RS_REASON_NO_SCAN_DIR = 1 << 1;
const RS_REASON_NOT_NATIVE_1TO1 = 1 << 2;
const RS_REASON_FRAME_GAP = 1 << 3;
const RS_REASON_CAMERA_MOTION = 1 << 4;
const RS_REASON_CLOCK_SKEW = 1 << 5;
const RS_REASON_EXPOSURE_NULL = 1 << 6;
const RS_REASON_LOW_GRAD = 1 << 7;
const RS_REASON_AUDIO_UNDERRUN = 1 << 8;
const RS_REASON_AXIS_TRANSPOSED = 1 << 9;
const RS_REASON_NO_PTS = 1 << 10;

const VIBRO_CELL_W = 64;
const VIBRO_CELL_H = 36;
const VIBRO_CELLS = VIBRO_CELL_W * VIBRO_CELL_H; // 2304
const VIBRO_SPEC_BINS = 128;
const VIBRO_PLANES_LEN = 3 * VIBRO_CELLS + VIBRO_SPEC_BINS; // 7040
const VIBRO_PLANE_AMP = 0;
const VIBRO_PLANE_COHERENCE = VIBRO_CELLS;
const VIBRO_PLANE_PHASE = 2 * VIBRO_CELLS;
const VIBRO_PLANE_SPECTRUM = 3 * VIBRO_CELLS;

const INGEST_FLAG_PTS_KNOWN = 1 << 1;
const INGEST_FLAG_GEOM_KNOWN = 1 << 4;

function assert(cond, msg) {
  if (!cond) {
    throw new Error(`Gate I2 FAIL: ${msg}`);
  }
}

// ============================================================================
// 1. Source Hygiene & Wasm Binary / ABI Audit
// ============================================================================
{
  const bannedRe = new RegExp(
    `(${['CHRONOS', 'IMPOSSIBLE'].join('-')}|${['CHRONOS', '1000'].join('-')})`
  );
  for (const rel of ['src/vibrometry.rs', 'scripts/gate_i2_rs_welch.mjs']) {
    const txt = fs.readFileSync(path.join(ROOT, rel), 'utf8');
    assert(!txt.includes('\u2014'), `${rel}: contains forbidden em dash`);
    assert(!bannedRe.test(txt), `${rel}: contains forbidden internal tag`);
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
  'chronos_pre_tick',
  'chronos_post_tick',
  'chronos_get_wave2_telemetry_ptr',
  'chronos_set_rs_calib',
  'chronos_get_audio_staging_ptr',
  'chronos_push_audio_block',
  'chronos_get_vibro_telemetry_ptr',
  'chronos_get_vibro_planes_ptr',
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
const wave2Ptr = exp.chronos_get_wave2_telemetry_ptr();
const vibroTelemPtr = exp.chronos_get_vibro_telemetry_ptr();
const vibroPlanesPtr = exp.chronos_get_vibro_planes_ptr();
const audioStagingPtr = exp.chronos_get_audio_staging_ptr();

assert((wave2Ptr & 3) === 0 && wave2Ptr > 0 && wave2Ptr + 64 * 4 <= buf.byteLength, 'wave2Ptr invalid');
assert((vibroTelemPtr & 3) === 0 && vibroTelemPtr > 0 && vibroTelemPtr + 32 * 4 <= buf.byteLength, 'vibroTelemPtr invalid');
assert((vibroPlanesPtr & 3) === 0 && vibroPlanesPtr > 0 && vibroPlanesPtr + VIBRO_PLANES_LEN * 4 <= buf.byteLength, 'vibroPlanesPtr invalid');
assert((audioStagingPtr & 3) === 0 && audioStagingPtr > 0 && audioStagingPtr + 4096 * 4 <= buf.byteLength, 'audioStagingPtr invalid');

const regions = [
  [wave2Ptr, wave2Ptr + 64 * 4, 'wave2_telem'],
  [vibroTelemPtr, vibroTelemPtr + 32 * 4, 'vibro_telem'],
  [vibroPlanesPtr, vibroPlanesPtr + VIBRO_PLANES_LEN * 4, 'vibro_planes'],
  [audioStagingPtr, audioStagingPtr + 4096 * 4, 'audio_staging'],
];
for (let i = 0; i < regions.length; i++) {
  for (let j = i + 1; j < regions.length; j++) {
    const [a0, a1, na] = regions[i];
    const [b0, b1, nb] = regions[j];
    assert(!(a0 < b1 && b0 < a1), `Memory overlap between ${na} and ${nb}`);
  }
}

const wave2U32 = new Uint32Array(buf, wave2Ptr, 64);
const telemU32 = new Uint32Array(buf, vibroTelemPtr, 32);
const telemF32 = new Float32Array(buf, vibroTelemPtr, 32);
const planesF32 = new Float32Array(buf, vibroPlanesPtr, VIBRO_PLANES_LEN);
const audioStaging = new Float32Array(buf, audioStagingPtr, 4096);

function checkReservedTailZero() {
  for (let k = 22; k < 32; k++) {
    assert(telemU32[k] === 0, `vibro_telem[${k}] expected 0, got ${telemU32[k]}`);
  }
}

function readTelem() {
  checkReservedTailZero();
  for (let k = 5; k <= 17; k++) {
    assert(Number.isFinite(telemF32[k]), `Non-finite vibro_telem[${k}]=${telemF32[k]}`);
  }
  return {
    state: telemU32[0] >>> 0,
    mode: telemU32[1] >>> 0,
    reason: telemU32[2] >>> 0,
    kSegs: telemU32[3] >>> 0,
    validCells: telemU32[4] >>> 0,
    domFreqHz: telemF32[5],
    maxCoh: telemF32[6],
    gamma2Crit: telemF32[7],
    sincH: telemF32[8],
    k1Coh: telemF32[9],
    skewUs: telemF32[10],
    domPhaseRad: telemF32[11],
    domAmpPx: telemF32[12],
    dfHz: telemF32[13],
    dtRowUs: telemF32[14],
    exposureUs: telemF32[15],
    calibAudioRateHz: telemF32[16],
    maxSkewUs: telemF32[17],
    audioTotalSamples: telemU32[18] >>> 0,
    activeFrames: telemU32[19] >>> 0,
    domBand: telemU32[20] >>> 0,
    coupledCells: telemU32[21] >>> 0,
    stampValid: (wave2U32[16] & 1) !== 0,
  };
}

function wrapPi(a) {
  while (a > Math.PI) a -= 2.0 * Math.PI;
  while (a < -Math.PI) a += 2.0 * Math.PI;
  return a;
}

function sampleTexLuma(xs, ys) {
  const v =
    128.0 +
    38.0 * Math.sin(0.27 * xs + 0.13 * ys) +
    34.0 * Math.cos(0.19 * xs - 0.31 * ys) +
    22.0 * Math.sin(0.37 * xs + 0.29 * ys);
  return Math.max(8, Math.min(247, v));
}

function engineToSensorXY(ex, ey, ow, cw, ch, cropX0, cropY0, orientation) {
  const o = orientation & 7;
  const rx = (o & 4) !== 0 ? ow - 1 - ex : ex;
  const ry = ey;
  let xc = 0;
  let yc = 0;
  const rotK = o & 3;
  if (rotK === 0) {
    xc = rx;
    yc = ry;
  } else if (rotK === 1) {
    xc = ry;
    yc = ch - 1 - rx;
  } else if (rotK === 2) {
    xc = cw - 1 - rx;
    yc = ch - 1 - ry;
  } else {
    xc = cw - 1 - ry;
    yc = rx;
  }
  return [cropX0 + xc, cropY0 + yc];
}

const scratchRgba = new Uint8Array(FRAME_BYTES);

function initSession({ mode, dtRowUs, exposureUs, audioRateHz, maxSkewUs }) {
  exp.chronos_init();
  exp.chronos_set_resolution(1); // 640x360 HD
  exp.chronos_set_swarm_mode(0);
  const ok = exp.chronos_set_rs_calib(mode, dtRowUs, exposureUs, audioRateHz, maxSkewUs);
  assert(ok === 1, `chronos_set_rs_calib failed for mode=${mode}`);
  assert((wave2U32[2] & 1) !== 0, 'wave2_telem[2] bit0 (vibro enabled) must be set');
  assert(wave2U32[1] <= 29_100_000, `Arena bytes ${wave2U32[1]} exceeds 29100000`);
}

function splitU64(usBig) {
  const lo = Number(BigInt.asUintN(32, usBig));
  const hi = Number(BigInt.asUintN(32, usBig >> 32n));
  return [lo, hi];
}

function pushFrame({
  epoch = 1,
  seq,
  tSrcUs,
  dtSec = 1 / 30,
  geom = {
    srcW: 640,
    srcH: 360,
    cropX0: 0,
    cropY0: 0,
    cropW: 640,
    cropH: 360,
    orientation: 0,
    scanDir: 0,
    native1to1: 1,
  },
  flags = INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_GEOM_KNOWN,
  gapMask = 0,
  dtRowUs = 25.0,
  exposureUs = 200.0,
  boxcarSamples = 1,
  dispFn = () => [0.0, 0.0],
  flatLuma = null,
}) {
  const tBig = BigInt(Math.round(tSrcUs));
  const [tLo, tHi] = splitU64(tBig);
  if (flatLuma !== null) {
    for (let i = 0; i < FRAME_BYTES; i += 4) {
      scratchRgba[i] = flatLuma;
      scratchRgba[i + 1] = flatLuma;
      scratchRgba[i + 2] = flatLuma;
      scratchRgba[i + 3] = 255;
    }
  } else {
    const expSec = exposureUs * 1e-6;
    const invBox = 1.0 / boxcarSamples;
    for (let ey = 0; ey < FRAME_H; ey++) {
      const rowBase = ey * FRAME_W * 4;
      for (let ex = 0; ex < FRAME_W; ex++) {
        const [sx, sy] = engineToSensorXY(
          ex,
          ey,
          FRAME_W,
          geom.cropW,
          geom.cropH,
          geom.cropX0,
          geom.cropY0,
          geom.orientation
        );
        const coord = geom.scanDir <= 1 ? sy : sx;
        const extent = geom.scanDir <= 1 ? geom.srcH : geom.srcW;
        const q = geom.scanDir === 0 || geom.scanDir === 2 ? coord : extent - 1 - coord;
        const tMidSec = (tSrcUs + q * dtRowUs + 0.5 * exposureUs) * 1e-6;
        let lum = 0.0;
        if (boxcarSamples <= 1) {
          const [ux, uy] = dispFn(sx, sy, ex, ey, tMidSec);
          lum = sampleTexLuma(sx - ux, sy - uy);
        } else {
          for (let s = 0; s < boxcarSamples; s++) {
            const frac = (s + 0.5) * invBox - 0.5;
            const tS = tMidSec + frac * expSec;
            const [ux, uy] = dispFn(sx, sy, ex, ey, tS);
            lum += sampleTexLuma(sx - ux, sy - uy);
          }
          lum *= invBox;
        }
        const v = Math.max(0, Math.min(255, Math.round(lum)));
        const o = rowBase + ex * 4;
        scratchRgba[o] = v;
        scratchRgba[o + 1] = v;
        scratchRgba[o + 2] = v;
        scratchRgba[o + 3] = 255;
      }
    }
  }

  const ok = exp.chronos_set_ingest_meta(
    epoch,
    seq,
    tLo,
    tHi,
    -1.0,
    flags,
    geom.srcW,
    geom.srcH,
    geom.cropX0,
    geom.cropY0,
    geom.cropW,
    geom.cropH,
    geom.orientation,
    geom.scanDir,
    0,
    0,
    FRAME_W,
    FRAME_H,
    gapMask,
    geom.native1to1
  );
  assert(ok === 1, `chronos_set_ingest_meta rejected frame seq=${seq}`);
  const dstPtr = exp.chronos_get_ingest_rgba_ptr();
  new Uint8Array(mem.buffer, dstPtr, FRAME_BYTES).set(scratchRgba);
  exp.chronos_pre_tick(dtSec);
  exp.chronos_post_tick(dtSec);
}

function pushAudioSamples({
  startSampleIdx,
  numSamples,
  t0Us,
  sampleRateHz,
  estSkewUs = 0.0,
  audioFn,
}) {
  for (let i = 0; i < numSamples; i++) {
    const tSec = t0Us * 1e-6 + (startSampleIdx + i) / sampleRateHz;
    audioStaging[i] = audioFn(tSec);
  }
  const tEndUs = BigInt(t0Us) + BigInt(Math.round(((startSampleIdx + numSamples) * 1e6) / sampleRateHz));
  const [tLo, tHi] = splitU64(tEndUs);
  return exp.chronos_push_audio_block(numSamples, tLo, tHi, sampleRateHz, estSkewUs);
}

function runCoupledWelchSession({
  t0Us = 1_000_000,
  fps = 30,
  totalFrames = 30,
  dtRowUs = 25.0,
  exposureUs = 200.0,
  audioRateHz = 48000.0,
  maxSkewUs = 500.0,
  geom = {
    srcW: 640,
    srcH: 360,
    cropX0: 0,
    cropY0: 0,
    cropW: 640,
    cropH: 360,
    orientation: 0,
    scanDir: 0,
    native1to1: 1,
  },
  boxcarSamples = 1,
  dispFn,
  audioFn,
  onFrame = null,
}) {
  initSession({
    mode: RS_MODE_RS_WELCH,
    dtRowUs,
    exposureUs,
    audioRateHz,
    maxSkewUs,
  });
  let sampleCursor = 0;
  for (let f = 0; f < totalFrames; f++) {
    const tSrcUs = t0Us + Math.round((f * 1e6) / fps);
    const targetEndSample = Math.round((((f + 1) / fps) * audioRateHz));
    const blockN = targetEndSample - sampleCursor;
    const pushed = pushAudioSamples({
      startSampleIdx: sampleCursor,
      numSamples: blockN,
      t0Us,
      sampleRateHz: audioRateHz,
      estSkewUs: 0.0,
      audioFn,
    });
    assert(pushed === 1, `pushAudioSamples failed at frame ${f}`);
    sampleCursor = targetEndSample;

    pushFrame({
      epoch: 1,
      seq: f,
      tSrcUs,
      dtSec: 1 / fps,
      geom,
      dtRowUs,
      exposureUs,
      boxcarSamples,
      dispFn,
    });
    if (onFrame) onFrame(f, readTelem());
  }
  return readTelem();
}

function runRsOnlySession({
  t0Us = 1_000_000,
  fps = 30,
  totalFrames = 26,
  dtRowUs = 25.0,
  exposureUs = 200.0,
  geom = {
    srcW: 640,
    srcH: 360,
    cropX0: 0,
    cropY0: 0,
    cropW: 640,
    cropH: 360,
    orientation: 0,
    scanDir: 0,
    native1to1: 1,
  },
  boxcarSamples = 1,
  dispFn,
}) {
  initSession({
    mode: RS_MODE_RS_ONLY,
    dtRowUs,
    exposureUs,
    audioRateHz: 0.0,
    maxSkewUs: 500.0,
  });
  for (let f = 0; f < totalFrames; f++) {
    const tSrcUs = t0Us + Math.round((f * 1e6) / fps);
    pushFrame({
      epoch: 1,
      seq: f,
      tSrcUs,
      dtSec: 1 / fps,
      geom,
      dtRowUs,
      exposureUs,
      boxcarSamples,
      dispFn,
    });
  }
  return readTelem();
}

// ============================================================================
// Suite 1: Exposure Sinc H(f) Precision & Null Detection
// ============================================================================
console.log('=== Gate I2: Calibrated RS Vibrometry & Multi-Segment Welch Coherence ===');
{
  // 1a. H(0) = 1.0 on static textured scene
  const staticT = runRsOnlySession({
    totalFrames: 26,
    exposureUs: 1000.0,
    dispFn: () => [0.0, 0.0],
  });
  assert(staticT.state === RS_STATE_READY && staticT.stampValid, 'Static RS_ONLY must reach READY');
  assert(Math.abs(staticT.sincH - 1.0) < 1e-5, `Expected H(0)=1.0, got ${staticT.sincH}`);
  assert((staticT.reason & RS_REASON_EXPOSURE_NULL) === 0, 'Static scene must not set EXPOSURE_NULL');
  assert(staticT.domAmpPx < 1e-3, `Static amp expected ~0, got ${staticT.domAmpPx}`);

  // 1b. 880 Hz at T_e = 1 ms (H(880) ~ 0.1329) with true 11-tap boxcar exposure blur
  const fTone = 880.0;
  const trueAmp = 0.85;
  const sinc1msRs = runRsOnlySession({
    totalFrames: 26,
    exposureUs: 1000.0,
    boxcarSamples: 11,
    dispFn: (_sx, _sy, _ex, _ey, tSec) => [trueAmp * Math.cos(2 * Math.PI * fTone * tSec + 0.4), 0.0],
  });
  assert(sinc1msRs.state === RS_STATE_READY, '880Hz 1ms RS_ONLY must reach READY');
  assert(Math.abs(sinc1msRs.domFreqHz - 880.0) < 2.0, `RS_ONLY 880Hz expected ~880, got ${sinc1msRs.domFreqHz}`);
  assert(Math.abs(sinc1msRs.sincH - 0.1329) < 1e-2, `RS_ONLY H(880,1ms) expected ~0.1329, got ${sinc1msRs.sincH}`);
  assert((sinc1msRs.reason & RS_REASON_EXPOSURE_NULL) === 0, '1ms at 880Hz must not flag EXPOSURE_NULL');
  assert(
    Math.abs(sinc1msRs.domAmpPx - trueAmp) < 0.18,
    `1/H(f) recovery from boxcar-blurred 880Hz failed: got ${sinc1msRs.domAmpPx} vs ${trueAmp}`
  );

  // Welch mode at fs=45056 Hz (bin 20 = exact 880.0 Hz)
  const sinc1msWelch = runCoupledWelchSession({
    totalFrames: 30,
    exposureUs: 1000.0,
    audioRateHz: 45056.0,
    boxcarSamples: 11,
    dispFn: (_sx, _sy, _ex, _ey, tSec) => [trueAmp * Math.cos(2 * Math.PI * fTone * tSec + 0.4), 0.0],
    audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * fTone * tSec + 0.1),
  });
  assert(sinc1msWelch.state === RS_STATE_READY && sinc1msWelch.kSegs >= 8, '880Hz 1ms Welch must reach READY');
  assert(Math.abs(sinc1msWelch.domFreqHz - 880.0) < 1e-3, `Welch 880Hz expected 880.0, got ${sinc1msWelch.domFreqHz}`);
  assert(Math.abs(sinc1msWelch.sincH - 0.1329) < 1e-3, `Welch H(880,1ms) expected 0.1329, got ${sinc1msWelch.sincH}`);
  assert(Math.abs(sinc1msWelch.domAmpPx - trueAmp) < 0.18, `Welch 1/H amplitude recovery got ${sinc1msWelch.domAmpPx}`);

  // 1c. 880 Hz at T_e = 1/880 s = 1136.3636 us -> |H| < 0.05 + RS_REASON_EXPOSURE_NULL + bounded amp
  const teNullUs = 1e6 / 880.0;
  const nullWelchBox = runCoupledWelchSession({
    totalFrames: 30,
    exposureUs: teNullUs,
    audioRateHz: 48000.0,
    boxcarSamples: 11,
    dispFn: (_sx, _sy, _ex, _ey, tSec) => [trueAmp * Math.cos(2 * Math.PI * fTone * tSec + 0.4), 0.0],
    audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * fTone * tSec + 0.1),
  });
  assert(nullWelchBox.state === RS_STATE_READY, 'Null Welch boxcar must report READY');
  assert(Math.abs(nullWelchBox.sincH) < 0.05, `Expected |H| < 0.05 at exposure null, got ${nullWelchBox.sincH}`);
  assert((nullWelchBox.reason & RS_REASON_EXPOSURE_NULL) !== 0, 'Expected RS_REASON_EXPOSURE_NULL on boxcar null');
  assert(Number.isFinite(nullWelchBox.domAmpPx) && nullWelchBox.domAmpPx < 0.10, `Expected bounded raw amp under null, got ${nullWelchBox.domAmpPx}`);

  const nullRsUnblurred = runRsOnlySession({
    totalFrames: 26,
    exposureUs: teNullUs,
    boxcarSamples: 1,
    dispFn: (_sx, _sy, _ex, _ey, tSec) => [0.45 * Math.cos(2 * Math.PI * fTone * tSec + 0.4), 0.0],
  });
  assert((nullRsUnblurred.reason & RS_REASON_EXPOSURE_NULL) !== 0, 'RS_ONLY at T_e=1/880 must flag EXPOSURE_NULL');
  assert(Math.abs(nullRsUnblurred.sincH) < 0.05 && nullRsUnblurred.domAmpPx < 0.6, 'RS_ONLY null must suppress 1/H blowup');

  console.log(
    `  [PASS] Suite 1 (Exposure Sinc & Null): H(0)=${staticT.sincH.toFixed(4)}, ` +
      `H(880,1ms)=${sinc1msWelch.sincH.toFixed(4)} (amp=${sinc1msWelch.domAmpPx.toFixed(3)} vs ${trueAmp}), ` +
      `null |H|=${Math.abs(nullWelchBox.sincH).toFixed(4)} reason=0x${nullWelchBox.reason.toString(16)}`
  );
}

// ============================================================================
// Suite 2: Sub-Frame RS Vibrometry (240 Hz & 421.875 Hz @ 30 fps, Signed, D4)
// ============================================================================
{
  const amp240 = 0.45;
  // Note: 240 Hz is an exact 8x harmonic of 30.000 fps, so frames 0..15 establish the
  // static reference baseline before acoustic/shaker onset (or use non-stroboscopic 25 fps).
  const rs240 = runRsOnlySession({
    totalFrames: 26,
    dispFn: (_sx, _sy, _ex, _ey, tSec) =>
      tSec >= 1_000_000 * 1e-6 + 15.5 / 30.0
        ? [amp240 * Math.cos(2 * Math.PI * 240.0 * tSec + 0.5), 0.0]
        : [0.0, 0.0],
  });
  assert(rs240.state === RS_STATE_READY, '240Hz RS_ONLY must be READY');
  assert(Math.abs(rs240.domFreqHz - 240.0) < 3.0, `RS_ONLY 240Hz got ${rs240.domFreqHz}`);
  assert(Math.abs(rs240.domAmpPx - amp240) < 0.08, `RS_ONLY 240Hz amp got ${rs240.domAmpPx}`);

  // Also test continuous 240 Hz from frame 0 at 25 fps (non-integer subharmonic 240/25 = 9.6)
  // on a vibrating target inside a stationary tripod perimeter (avoids full-frame camera-pan alias).
  const rs240Cont25 = runRsOnlySession({
    fps: 25,
    totalFrames: 26,
    dispFn: (_sx, _sy, ex, _ey, tSec) =>
      ex >= 80 && ex < 560
        ? [amp240 * Math.cos(2 * Math.PI * 240.0 * tSec + 0.5), 0.0]
        : [0.0, 0.0],
  });
  assert(rs240Cont25.state === RS_STATE_READY, '240Hz @ 25fps continuous RS_ONLY must be READY');
  assert(Math.abs(rs240Cont25.domFreqHz - 240.0) < 3.0, `RS_ONLY 240Hz @ 25fps got ${rs240Cont25.domFreqHz}`);
  assert(Math.abs(rs240Cont25.domAmpPx - amp240) < 0.08, `RS_ONLY 240Hz @ 25fps amp got ${rs240Cont25.domAmpPx}`);

  const w240 = runCoupledWelchSession({
    totalFrames: 30,
    dispFn: (_sx, _sy, _ex, _ey, tSec) =>
      tSec >= 1_000_000 * 1e-6 + 15.5 / 30.0
        ? [amp240 * Math.cos(2 * Math.PI * 240.0 * tSec + 0.5), 0.0]
        : [0.0, 0.0],
    audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * 240.0 * tSec + 0.1),
  });
  assert(w240.state === RS_STATE_READY && w240.kSegs >= 8, '240Hz Welch must be READY');
  assert(Math.abs(w240.domFreqHz - 240.0) <= 24.0, `Welch 240Hz bin tolerance exceeded: ${w240.domFreqHz}`);
  assert(w240.maxCoh > 0.90, `240Hz Welch coherence expected > 0.90, got ${w240.maxCoh}`);

  // Signed displacement (+0.45 vs -0.45) phase flip pi +/- 0.20 rad at 421.875 Hz
  const pos422 = runCoupledWelchSession({
    totalFrames: 30,
    dispFn: (_sx, _sy, _ex, _ey, tSec) => [0.45 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.6), 0.0],
    audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.2),
  });
  const neg422 = runCoupledWelchSession({
    totalFrames: 30,
    dispFn: (_sx, _sy, _ex, _ey, tSec) => [-0.45 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.6), 0.0],
    audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.2),
  });
  const phaseDiff = Math.abs(wrapPi(pos422.domPhaseRad - neg422.domPhaseRad));
  assert(Math.abs(pos422.domFreqHz - 421.875) < 0.1, `421.875Hz freq got ${pos422.domFreqHz}`);
  assert(Math.abs(pos422.domAmpPx - 0.45) < 0.08, `421.875Hz amp got ${pos422.domAmpPx}`);
  assert(Math.abs(phaseDiff - Math.PI) < 0.20, `Signed flip phase delta expected pi, got ${phaseDiff}`);

  // D4 + scan_dir + crop consistency across 4 sensor/crop configurations
  const d4Configs = [
    { srcW: 1920, srcH: 1080, cropX0: 200, cropY0: 120, cropW: 640, cropH: 360, orientation: 0, scanDir: 0, native1to1: 1 },
    { srcW: 1920, srcH: 1080, cropX0: 200, cropY0: 120, cropW: 640, cropH: 360, orientation: 2, scanDir: 0, native1to1: 1 },
    { srcW: 1920, srcH: 1080, cropX0: 200, cropY0: 120, cropW: 640, cropH: 360, orientation: 4, scanDir: 1, native1to1: 1 },
    { srcW: 1080, srcH: 1920, cropX0: 80, cropY0: 100, cropW: 360, cropH: 640, orientation: 1, scanDir: 2, native1to1: 1 },
  ];
  for (const cfg of d4Configs) {
    const res = runCoupledWelchSession({
      totalFrames: 30,
      geom: cfg,
      dispFn: (_sx, _sy, _ex, _ey, tSec) => [0.45 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.6), 0.0],
      audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.2),
    });
    assert(res.state === RS_STATE_READY && res.stampValid, `D4 config o=${cfg.orientation} scan=${cfg.scanDir} failed READY`);
    assert(Math.abs(res.domFreqHz - 421.875) < 0.1, `D4 o=${cfg.orientation} freq=${res.domFreqHz}`);
    assert(res.maxCoh > 0.90, `D4 o=${cfg.orientation} scan=${cfg.scanDir} coh=${res.maxCoh}`);
    assert(Math.abs(res.domAmpPx - 0.45) < 0.08, `D4 o=${cfg.orientation} amp=${res.domAmpPx}`);
    const pDiff = Math.abs(wrapPi(res.domPhaseRad - pos422.domPhaseRad));
    assert(pDiff < 0.25, `D4 o=${cfg.orientation} scan=${cfg.scanDir} phase mismatch ${pDiff} rad vs baseline`);
  }

  console.log(
    `  [PASS] Suite 2 (Sub-Frame RS & D4): RS_ONLY 240Hz=${rs240.domFreqHz.toFixed(2)}Hz (${rs240.domAmpPx.toFixed(3)}px), ` +
      `Welch 421.875Hz coh=${pos422.maxCoh.toFixed(4)}, sign-flip dPhi=${phaseDiff.toFixed(3)}rad, 4x D4/crop/scanDir aligned`
  );
}

// ============================================================================
// Suite 3: K = 1 vs K >= 8 Welch Coherence Trap
// ============================================================================
{
  let sawWarmSuppression = false;
  const trap = runCoupledWelchSession({
    totalFrames: 30,
    dispFn: (_sx, _sy, _ex, _ey, tSec) => [0.45 * Math.cos(2 * Math.PI * 750.0 * tSec + 0.3), 0.0],
    audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.1),
    onFrame: (_f, t) => {
      if (t.activeFrames >= 17 && t.kSegs < 8) {
        assert(t.state === RS_STATE_WARMING, `Expected WARMING while K=${t.kSegs} < 8`);
        assert(t.gamma2Crit === 1.0, `Expected gamma2_crit=1.0 while K < 8, got ${t.gamma2Crit}`);
        assert(t.validCells === 0, 'Expected validCells=0 while K < 8');
        sawWarmSuppression = true;
      }
    },
  });
  assert(sawWarmSuppression, 'Expected to observe K < 8 warming gate');
  assert(trap.state === RS_STATE_READY && trap.kSegs >= 8, `Trap final state=${trap.state}, K=${trap.kSegs}`);
  assert(trap.k1Coh > 0.95, `Expected K=1 raw coherence > 0.95, got ${trap.k1Coh}`);
  const expectedBonfCrit = 1.0 - Math.pow(0.01 / (32 * 127), 1.0 / (0.95 * trap.kSegs - 1.0));
  assert(
    Math.abs(trap.gamma2Crit - expectedBonfCrit) < 1e-5,
    `Expected Bonferroni gamma2Crit=${expectedBonfCrit.toFixed(6)} for 32x127 tests, got ${trap.gamma2Crit}`
  );
  assert(trap.maxCoh < trap.gamma2Crit, `Expected averaged coherence ${trap.maxCoh} < gamma2_crit ${trap.gamma2Crit}`);
  assert(trap.validCells === 0 && trap.coupledCells === 0, `Expected 0 coupled/valid cells, got ${trap.validCells}/${trap.coupledCells}`);
  console.log(
    `  [PASS] Suite 3 (K=1 vs K>=8 Trap): K=${trap.kSegs}, K=1 raw coh=${trap.k1Coh.toFixed(4)} (>0.95) vs ` +
      `averaged max_coh=${trap.maxCoh.toFixed(4)} < crit=${trap.gamma2Crit.toFixed(4)}, validCells=0`
  );
}

// ============================================================================
// Suite 4: Localized Acoustic-Optical Coupling (Bands 4..8 Only)
// ============================================================================
{
  const loc = runCoupledWelchSession({
    totalFrames: 30,
    dispFn: (_sx, _sy, ex, _ey, tSec) => {
      if (ex >= 160 && ex < 320) {
        return [0.45 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.5), 0.0];
      }
      return [0.0, 0.0];
    },
    audioFn: (tSec) => 0.5 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.2),
  });
  assert(loc.state === RS_STATE_READY && loc.stampValid, 'Localized coupling must reach READY with valid wave2 stamp');
  assert(loc.maxCoh > 0.85, `Expected maxCoh > 0.85, got ${loc.maxCoh}`);
  assert(loc.domBand >= 8 && loc.domBand < 16, `Expected domBand in 8..15, got ${loc.domBand}`);
  assert(loc.validCells === 576 && loc.coupledCells === 576, `Expected 576 active cells, got valid=${loc.validCells}, coupled=${loc.coupledCells}`);

  let activeSum = 0.0;
  let activeCount = 0;
  let quietMax = 0.0;
  for (let cy = 0; cy < VIBRO_CELL_H; cy++) {
    for (let cx = 0; cx < VIBRO_CELL_W; cx++) {
      const idx = cy * VIBRO_CELL_W + cx;
      const coh = planesF32[VIBRO_PLANE_COHERENCE + idx];
      const amp = planesF32[VIBRO_PLANE_AMP + idx];
      if (cx >= 16 && cx < 32) {
        activeSum += coh;
        activeCount++;
        assert(amp > 0.30, `Active cell (${cx},${cy}) expected amp > 0.30, got ${amp}`);
      } else {
        if (coh > quietMax) quietMax = coh;
        assert(amp === 0.0, `Quiet cell (${cx},${cy}) expected amp=0.0, got ${amp}`);
      }
    }
  }
  const activeMean = activeSum / activeCount;
  assert(activeMean > 0.85, `Active zone mean coherence expected > 0.85, got ${activeMean}`);
  assert(quietMax < 0.10, `Quiet zone max coherence expected < 0.10, got ${quietMax}`);
  console.log(
    `  [PASS] Suite 4 (Localized 64x36 Coupling): domBand=${loc.domBand}, activeMeanCoh=${activeMean.toFixed(4)}, ` +
      `quietMaxCoh=${quietMax.toFixed(4)}, validCells=${loc.validCells}/${VIBRO_CELLS}`
  );
}

// ============================================================================
// Suite 5: Fail-Closed Matrix & Continuity Resets
// ============================================================================
{
  // 5a. Bad calibration rejected (including non-finite audio_rate_hz in RS_ONLY mode)
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  assert(exp.chronos_set_rs_calib(RS_MODE_RS_WELCH, 0.0, 200.0, 48000.0, 500.0) === 0, 'dt_row=0 must fail');
  let t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_BAD_CALIB) !== 0, 'Bad calib state');
  assert(exp.chronos_set_rs_calib(RS_MODE_RS_ONLY, 25.0, 200.0, Infinity, 500.0) === 0, 'Infinity audio_rate_hz in RS_ONLY must fail');
  t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_BAD_CALIB) !== 0 && t.calibAudioRateHz === 0.0, 'Non-finite audio rate must fail cleanly with zeroed word 16');

  // 5b. scan_dir = 0xffffffff -> RS_REASON_NO_SCAN_DIR
  initSession({ mode: RS_MODE_RS_ONLY, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 0.0, maxSkewUs: 500.0 });
  pushFrame({
    seq: 0,
    tSrcUs: 1_000_000,
    geom: { srcW: 640, srcH: 360, cropX0: 0, cropY0: 0, cropW: 640, cropH: 360, orientation: 0, scanDir: 0xffffffff, native1to1: 1 },
  });
  t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_NO_SCAN_DIR) !== 0, 'Expected NO_SCAN_DIR');

  // 5c. native_1to1 = 0 -> RS_REASON_NOT_NATIVE_1TO1
  initSession({ mode: RS_MODE_RS_ONLY, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 0.0, maxSkewUs: 500.0 });
  pushFrame({
    seq: 0,
    tSrcUs: 1_000_000,
    geom: { srcW: 640, srcH: 360, cropX0: 0, cropY0: 0, cropW: 640, cropH: 360, orientation: 0, scanDir: 0, native1to1: 0 },
  });
  t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_NOT_NATIVE_1TO1) !== 0, 'Expected NOT_NATIVE_1TO1');

  // 5d. Missing PTS (flags & 2 == 0) -> RS_REASON_NO_PTS
  initSession({ mode: RS_MODE_RS_ONLY, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 0.0, maxSkewUs: 500.0 });
  pushFrame({
    seq: 0,
    tSrcUs: 0,
    flags: INGEST_FLAG_GEOM_KNOWN,
  });
  t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_NO_PTS) !== 0, 'Expected NO_PTS');

  // 5e. Transposed axis (o=0, scan_dir=2 and o=1, scan_dir=0) -> RS_REASON_AXIS_TRANSPOSED
  initSession({ mode: RS_MODE_RS_ONLY, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 0.0, maxSkewUs: 500.0 });
  pushFrame({
    seq: 0,
    tSrcUs: 1_000_000,
    geom: { srcW: 640, srcH: 360, cropX0: 0, cropY0: 0, cropW: 640, cropH: 360, orientation: 0, scanDir: 2, native1to1: 1 },
  });
  t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_AXIS_TRANSPOSED) !== 0, 'Expected AXIS_TRANSPOSED o=0,sd=2');

  initSession({ mode: RS_MODE_RS_ONLY, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 0.0, maxSkewUs: 500.0 });
  pushFrame({
    seq: 0,
    tSrcUs: 1_000_000,
    geom: { srcW: 360, srcH: 640, cropX0: 0, cropY0: 0, cropW: 360, cropH: 640, orientation: 1, scanDir: 0, native1to1: 1 },
  });
  t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_AXIS_TRANSPOSED) !== 0, 'Expected AXIS_TRANSPOSED o=1,sd=0');

  // 5f. Excess clock skew, backward audio, gapped audio, and scene cut continuity reset + recovery
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
  const t0 = 2_000_000;
  let sampleCursor = 0;
  const sigOpt = (_sx, _sy, _ex, _ey, tSec) => [0.45 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.5), 0.0];
  const sigAud = (tSec) => 0.5 * Math.cos(2 * Math.PI * 421.875 * tSec + 0.2);

  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    assert(
      pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud }) === 1,
      `Setup audio failed at ${f}`
    );
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY, 'Expected READY before fault injection');

  // Excess skew
  const badSkew = pushAudioSamples({
    startSampleIdx: sampleCursor,
    numSamples: 1600,
    t0Us: t0,
    sampleRateHz: 48000.0,
    estSkewUs: 2500.0,
    audioFn: sigAud,
  });
  assert(badSkew === 0, 'Excess clock skew must return 0');
  t = readTelem();
  assert(t.state === RS_STATE_DISTURBED && (t.reason & RS_REASON_CLOCK_SKEW) !== 0, 'Expected DISTURBED + CLOCK_SKEW');
  assert(!t.stampValid && t.kSegs === 0, 'Excess skew must immediately invalidate stamp and clear Welch segments');

  // Re-establish READY then test backward audio timestamp
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
  sampleCursor = 0;
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().kSegs >= 8, 'Must have K>=8 before backward audio check');
  const [bLo, bHi] = splitU64(BigInt(t0 + 500_000));
  assert(exp.chronos_push_audio_block(1600, bLo, bHi, 48000.0, 0.0) === 0, 'Backward audio block must return 0');
  assert(readTelem().kSegs === 0 && !readTelem().stampValid && (readTelem().reason & RS_REASON_AUDIO_UNDERRUN) !== 0, 'Backward audio must reset Welch count and stamp');

  // Single dropped 1,600-sample block gap resets Welch interval & invalidates stamp
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + 1_000_000 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    const prevEnd = f === 0 ? 0 : Math.round((f / 30) * 48000);
    const blkN = nextEnd - prevEnd;
    pushAudioSamples({
      startSampleIdx: prevEnd,
      numSamples: blkN,
      t0Us: t0 + 1_000_000,
      sampleRateHz: 48000.0,
      audioFn: sigAud,
    });
    pushFrame({ seq: 30 + f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY && readTelem().stampValid, 'Expected recovery before single-block gap');
  const postDropCursor = Math.round(48000) + 1600; // skip one 1,600-sample block (~33.3 ms)
  assert(
    pushAudioSamples({
      startSampleIdx: postDropCursor,
      numSamples: 1600,
      t0Us: t0 + 1_000_000,
      sampleRateHz: 48000.0,
      audioFn: sigAud,
    }) === 1,
    'Post-gap block starts fresh interval'
  );
  assert(
    readTelem().kSegs === 0 && !readTelem().stampValid && (readTelem().reason & RS_REASON_AUDIO_UNDERRUN) !== 0,
    'Single 1,600-sample audio gap must reset Welch count and invalidate stamp'
  );

  // Short 36-sample (750 us) PCM gap must also immediately invalidate READY stamp & trigger AUDIO_UNDERRUN
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
  sampleCursor = 0;
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY && readTelem().kSegs >= 8 && readTelem().stampValid, 'Expected READY before 36-sample short gap check');
  assert(
    pushAudioSamples({
      startSampleIdx: sampleCursor + 36,
      numSamples: 1600,
      t0Us: t0,
      sampleRateHz: 48000.0,
      audioFn: sigAud,
    }) === 1,
    'Post short-gap block starts fresh interval'
  );
  t = readTelem();
  assert(
    t.kSegs === 0 && t.state === RS_STATE_WARMING && (t.reason & RS_REASON_AUDIO_UNDERRUN) !== 0 && !t.stampValid,
    `Short 36-sample (750 us) PCM gap must immediately invalidate READY stamp and reset Welch interval, got state=${t.state} k=${t.kSegs} reason=${t.reason} valid=${t.stampValid}`
  );
  pushFrame({ seq: 30, tSrcUs: t0 + 1_000_000, dispFn: sigOpt });
  t = readTelem();
  assert(
    t.state === RS_STATE_WARMING && !t.stampValid,
    `Next video frame after 36-sample PCM gap must remain WARMING with invalid stamp, got state=${t.state} valid=${t.stampValid}`
  );

  // 1-sample gap at 192 kHz (4.333 us < previous 4.604 us tolerance) must immediately invalidate READY stamp
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 20.0, audioRateHz: 192000.0, maxSkewUs: 500.0 });
  const t192 = 2_000_000;
  let c192 = 0;
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t192 + Math.round((f * 1e6) / 60);
    const nextEnd = Math.round(((f + 1) / 60) * 192000);
    while (c192 < nextEnd) {
      const nChunk = Math.min(4096, nextEnd - c192);
      pushAudioSamples({ startSampleIdx: c192, numSamples: nChunk, t0Us: t192, sampleRateHz: 192000.0, audioFn: sigAud });
      c192 += nChunk;
    }
    pushFrame({ seq: f, tSrcUs, dtSec: 1 / 60, exposureUs: 20.0, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY && readTelem().stampValid, '192 kHz must reach READY before 1-sample gap');
  // Re-anchor exact 2933339 us block end then skip 1 sample and end next 128-sample block at 2934010 us (measured diff = 4.333 us)
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 20.0, audioRateHz: 192000.0, maxSkewUs: 500.0 });
  c192 = 0;
  const sigOpt5625 = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * 562.5 * tSec + 0.3), 0.0];
  const sigAud5625 = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * 562.5 * tSec + 0.1);
  for (let f = 0; f < 28; f++) {
    const tSrcUs = t192 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 192000);
    while (c192 < nextEnd) {
      const nChunk = Math.min(4096, nextEnd - c192);
      pushAudioSamples({ startSampleIdx: c192, numSamples: nChunk, t0Us: t192, sampleRateHz: 192000.0, audioFn: sigAud5625 });
      c192 += nChunk;
    }
    pushFrame({ seq: f, tSrcUs, dtSec: 1 / 30, exposureUs: 20.0, dispFn: sigOpt5625 });
  }
  assert(readTelem().state === RS_STATE_READY && readTelem().kSegs >= 8 && readTelem().stampValid, '192 kHz @ 30fps must be READY before 2933339->2934010 check');
  for (let i = 0; i < 128; i++) audioStaging[i] = sigAud5625(i / 192000.0);
  const [b1Lo, b1Hi] = splitU64(2_933_339n);
  assert(exp.chronos_push_audio_block(128, b1Lo, b1Hi, 192000.0, 0.0) === 1, '192 kHz pre-gap block push failed');
  const [b2Lo, b2Hi] = splitU64(2_934_010n);
  assert(exp.chronos_push_audio_block(128, b2Lo, b2Hi, 192000.0, 0.0) === 1, '192 kHz post-gap block push failed');
  t = readTelem();
  assert(
    t.state === RS_STATE_WARMING && t.kSegs === 0 && (t.reason & RS_REASON_AUDIO_UNDERRUN) !== 0 && !t.stampValid,
    `1-sample gap at 192 kHz (4.333 us) must reset Welch and invalidate stamp, got state=${t.state} k=${t.kSegs} reason=${t.reason} valid=${t.stampValid}`
  );

  // RS_MODE_RS_ONLY must be completely isolated from audio gaps and audio faults
  initSession({ mode: RS_MODE_RS_ONLY, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 0.0, maxSkewUs: 500.0 });
  for (let f = 0; f < 26; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    pushFrame({
      seq: f,
      tSrcUs,
      dispFn: (_sx, _sy, _ex, _ey, tSec) =>
        tSec >= t0 * 1e-6 + 15.5 / 30.0 ? [0.45 * Math.cos(2.0 * Math.PI * 240.0 * tSec + 0.5), 0.0] : [0.0, 0.0],
    });
  }
  const rsOnlyBefore = readTelem();
  const rsOnlyAmpCopy = Float32Array.from(planesF32.subarray(VIBRO_PLANE_AMP, VIBRO_PLANE_AMP + VIBRO_CELLS));
  assert(rsOnlyBefore.state === RS_STATE_READY && rsOnlyBefore.validCells === VIBRO_CELLS && rsOnlyBefore.stampValid, 'Expected READY RS_ONLY before audio stress');
  pushAudioSamples({ startSampleIdx: 0, numSamples: 1600, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
  pushAudioSamples({ startSampleIdx: 1601, numSamples: 1600, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
  pushAudioSamples({ startSampleIdx: 3201, numSamples: 1600, t0Us: t0, sampleRateHz: 44100.0, audioFn: sigAud });
  exp.chronos_push_audio_block(1600, 1_500_000, 0, 48000.0, NaN);
  const rsOnlyAfter = readTelem();
  assert(
    rsOnlyAfter.state === RS_STATE_READY &&
      rsOnlyAfter.validCells === VIBRO_CELLS &&
      rsOnlyAfter.stampValid &&
      Math.abs(rsOnlyAfter.domFreqHz - rsOnlyBefore.domFreqHz) < 1e-3,
    `RS_ONLY must stay READY with ${VIBRO_CELLS} validCells & valid stamp across audio gap/faults, got ${JSON.stringify(rsOnlyAfter)}`
  );
  for (let i = 0; i < VIBRO_CELLS; i++) {
    assert(planesF32[VIBRO_PLANE_AMP + i] === rsOnlyAmpCopy[i], `RS_ONLY amplitude cell ${i} mutated by audio push`);
  }

  // Sample-rate mismatch and non-finite PCM sample rejection keep DISTURBED across video gap + clean frames
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
  sampleCursor = 0;
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY && readTelem().stampValid, 'Expected READY before 44.1 kHz mismatch check');
  audioStaging[0] = Infinity;
  const [infLo, infHi] = splitU64(BigInt(t0 + 1_020_000));
  assert(exp.chronos_push_audio_block(1, infLo, infHi, 48000.0, 0.0) === 0, 'Non-finite PCM sample in audio staging must fail closed');
  t = readTelem();
  assert(t.state === RS_STATE_DISTURBED && (t.reason & RS_REASON_BAD_CALIB) !== 0 && !t.stampValid, 'Non-finite PCM sample must trigger DISTURBED|BAD_CALIB');

  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
  sampleCursor = 0;
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dispFn: sigOpt });
  }
  const badRateRes = pushAudioSamples({
    startSampleIdx: sampleCursor,
    numSamples: 1600,
    t0Us: t0,
    sampleRateHz: 44100.0,
    audioFn: sigAud,
  });
  assert(badRateRes === 0, 'Mismatched 44.1 kHz audio block must return 0');
  t = readTelem();
  assert(
    t.state === RS_STATE_DISTURBED && (t.reason & RS_REASON_BAD_CALIB) !== 0 && !t.stampValid && t.kSegs === 0,
    'Sample-rate mismatch must immediately clear Welch interval, enter DISTURBED|BAD_CALIB, and invalidate stamp'
  );
  pushFrame({ seq: 35, tSrcUs: t0 + 1_000_000, gapMask: 1, dispFn: sigOpt });
  t = readTelem();
  assert(
    t.state === RS_STATE_DISTURBED && (t.reason & RS_REASON_BAD_CALIB) !== 0 && !t.stampValid,
    'Video continuity gap after rate rejection must retain DISTURBED|BAD_CALIB until valid audio resumes'
  );
  pushFrame({ seq: 36, tSrcUs: t0 + 1_033_333, dispFn: sigOpt });
  t = readTelem();
  assert(
    t.state === RS_STATE_DISTURBED && (t.reason & RS_REASON_BAD_CALIB) !== 0 && !t.stampValid,
    'Subsequent clean video frame after rate rejection+gap must still stay DISTURBED|BAD_CALIB'
  );

  // Stopping audio holds original stamp only while covered, then expires IMMEDIATELY on first uncovered frame
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
  sampleCursor = 0;
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY && readTelem().stampValid && (wave2U32[16] & 2) === 0, 'Expected fresh READY before audio stop');
  const seqAtAudioStop = wave2U32[19] >>> 0;

  // Sub-hop frame whose scan start is still inside audio_last_t_end_us (t0 + 1_000_000) holds original stamp
  pushFrame({ seq: 30, tSrcUs: t0 + 972_000, dispFn: sigOpt });
  {
    const vh = readTelem();
    assert(
      vh.state === RS_STATE_READY && (wave2U32[16] & 3) === 3 && (wave2U32[19] >>> 0) === seqAtAudioStop,
      `Covered no-new-segment frame should hold original Welch seq=${seqAtAudioStop}, got state=${vh.state} flags=${wave2U32[16]} seq=${wave2U32[19]}`
    );
  }
  // Very first uncovered frame (opt_cur_min_us = t0 + 1_000_100 > audio_last_t_end_us) expires immediately
  for (let f = 30; f < 66; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    pushFrame({ seq: f + 1, tSrcUs, dispFn: sigOpt });
    const v = readTelem();
    assert(
      v.state === RS_STATE_WARMING && (v.reason & RS_REASON_AUDIO_UNDERRUN) !== 0 && !v.stampValid,
      `Uncovered frame f=${f} must immediately expire to WARMING|AUDIO_UNDERRUN without stamp, got state=${v.state} reason=${v.reason} valid=${v.stampValid}`
    );
  }

  // Optical alias guard: 4375 Hz optical motion vs 5625 Hz audio tone @ dtRow=100us (10 kHz row lattice) must not couple
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 100.0, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
  sampleCursor = 0;
  const optAlias = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * 4375.0 * tSec + 0.3), 0.0];
  const audAlias = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * 5625.0 * tSec + 0.1);
  for (let f = 0; f < 29; f++) {
    const tSrcUs = t0 + f * 41_000;
    const nextEnd = Math.round((f + 1) * 0.041 * 96000.0);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 96000.0, audioFn: audAlias });
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dtSec: 0.041, dtRowUs: 100.0, exposureUs: 20.0, dispFn: optAlias });
  }
  t = readTelem();
  const expectedAliasCrit = 1.0 - Math.pow(0.01 / (32 * 102), 1.0 / (0.95 * t.kSegs - 1.0));
  assert(
    t.coupledCells === 0 && t.maxCoh < t.gamma2Crit && Math.abs(t.gamma2Crit - expectedAliasCrit) < 1e-5,
    `Aliased 5625 Hz audio must be excluded above optical Nyquist with 32*102 Bonferroni crit, got coupled=${t.coupledCells} g2=${t.maxCoh} crit=${t.gamma2Crit}`
  );

  // Off-bin out-of-band spectral leakage guard: 4718.75 Hz optical vs 5281.25 / 4875 / 4820 Hz audio @ 96 kHz
  for (const [fOptLeak, fAudLeak] of [[4718.75, 5281.25], [4718.75, 4875.0], [4718.75, 4820.0]]) {
    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 100.0, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    sampleCursor = 0;
    const optLeakFn = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * fOptLeak * tSec + 0.3), 0.0];
    const audLeakFn = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * fAudLeak * tSec + 0.1);
    for (let f = 0; f < 29; f++) {
      const tSrcUs = t0 + f * 41_000;
      const nextEnd = Math.round((f + 1) * 0.041 * 96000.0);
      pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 96000.0, audioFn: audLeakFn });
      sampleCursor = nextEnd;
      pushFrame({ seq: f, tSrcUs, dtSec: 0.041, dtRowUs: 100.0, exposureUs: 20.0, dispFn: optLeakFn });
    }
    const tl = readTelem();
    assert(
      tl.state === RS_STATE_READY && tl.coupledCells === 0 && tl.maxCoh < tl.gamma2Crit,
      `Out-of-band ${fAudLeak} Hz skirt vs ${fOptLeak} Hz optical must stay READY without coupling, got state=${tl.state} coupled=${tl.coupledCells} g2=${tl.maxCoh} crit=${tl.gamma2Crit}`
    );
  }

  // Exact-cutoff tone (4800.0 Hz == inclusive f_opt_max=4800.0 Hz @ dtRow=100us, 96 kHz)
  {
    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 100.0, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    let exactCursor = 0;
    const opt4800 = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * 4800.0 * tSec + 0.2), 0.0];
    const aud4800 = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * 4800.0 * tSec + 0.4);
    for (let f = 0; f < 29; f++) {
      const tSrcUs = t0 + f * 41_000;
      const nextEnd = Math.round((f + 1) * 0.041 * 96000.0);
      pushAudioSamples({ startSampleIdx: exactCursor, numSamples: nextEnd - exactCursor, t0Us: t0, sampleRateHz: 96000.0, audioFn: aud4800 });
      exactCursor = nextEnd;
      pushFrame({ seq: f, tSrcUs, dtSec: 0.041, dtRowUs: 100.0, exposureUs: 20.0, dispFn: opt4800 });
    }
    const tsExact = readTelem();
    assert(
      tsExact.state === RS_STATE_READY && tsExact.coupledCells === VIBRO_CELLS && tsExact.maxCoh > 0.90,
      `Exact-cutoff 4800.0 Hz tone (<= 4800.0 Hz) must couple all 880 cells, got state=${tsExact.state} coupled=${tsExact.coupledCells} g2=${tsExact.maxCoh}`
    );
  }

  // Just-below-bin cutoff (dtRowUs=102.4f32 @ 96 kHz -> f_opt_max=4687.49993 Hz < bin 50 = 4687.5 Hz):
  // f_opt_max + 2.0 Hz audio must be rejected, while matched f_opt_max must couple all 880 cells.
  {
    const fCutoffJustBelow = 0.48 * 1_000_000.0 / Math.fround(102.4);
    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 102.4, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    let subCursor = 0;
    const optCut = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * fCutoffJustBelow * tSec + 0.3), 0.0];
    const audAboveCut = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * (fCutoffJustBelow + 2.0) * tSec + 0.1);
    for (let f = 0; f < 25; f++) {
      const tSrcUs = t0 + f * 42_000;
      const nextEnd = Math.round((f + 1) * 0.042 * 96000.0);
      pushAudioSamples({ startSampleIdx: subCursor, numSamples: nextEnd - subCursor, t0Us: t0, sampleRateHz: 96000.0, audioFn: audAboveCut });
      subCursor = nextEnd;
      pushFrame({ seq: f, tSrcUs, dtSec: 0.042, dtRowUs: 102.4, exposureUs: 20.0, dispFn: optCut });
    }
    const tsRejAbove = readTelem();
    assert(
      tsRejAbove.state === RS_STATE_READY && tsRejAbove.coupledCells === 0,
      `Out-of-band +2 Hz audio above 4687.49993 Hz cutoff must not couple, got state=${tsRejAbove.state} coupled=${tsRejAbove.coupledCells} g2=${tsRejAbove.maxCoh}`
    );

    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 102.4, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    subCursor = 0;
    const audAtCut = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * fCutoffJustBelow * tSec + 0.1);
    for (let f = 0; f < 25; f++) {
      const tSrcUs = t0 + f * 42_000;
      const nextEnd = Math.round((f + 1) * 0.042 * 96000.0);
      pushAudioSamples({ startSampleIdx: subCursor, numSamples: nextEnd - subCursor, t0Us: t0, sampleRateHz: 96000.0, audioFn: audAtCut });
      subCursor = nextEnd;
      pushFrame({ seq: f, tSrcUs, dtSec: 0.042, dtRowUs: 102.4, exposureUs: 20.0, dispFn: optCut });
    }
    const tsAccCut = readTelem();
    assert(
      tsAccCut.state === RS_STATE_READY && tsAccCut.coupledCells === VIBRO_CELLS && tsAccCut.maxCoh > tsAccCut.gamma2Crit && tsAccCut.maxCoh > 0.75,
      `Matched 4687.49993 Hz exact cutoff tone must couple all 880 cells, got state=${tsAccCut.state} coupled=${tsAccCut.coupledCells} g2=${tsAccCut.maxCoh}`
    );

    // Sub-bin OOB leakage through k_end - 2 when f_opt_max aligns with (k_end - 1) * df_hz:
    // 1) 96 kHz, dtRowUs = 182.857131958 -> f_opt_max = 2625 Hz (bin 28), 2627 Hz audio must not couple at bin 27 (2531.25 Hz)
    const dt96 = 182.857131958;
    const fCut96 = 0.48 * 1_000_000.0 / Math.fround(dt96);
    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: dt96, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    subCursor = 0;
    const opt96 = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * fCut96 * tSec + 0.3), 0.0];
    const aud96Oob = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * (fCut96 + 2.0) * tSec + 0.1);
    for (let f = 0; f < 34; f++) {
      const tSrcUs = t0 + f * 69_829;
      const nextEnd = Math.round((f + 1) * 0.069829 * 96000.0);
      while (subCursor < nextEnd) {
        const nBlk = Math.min(4096, nextEnd - subCursor);
        pushAudioSamples({ startSampleIdx: subCursor, numSamples: nBlk, t0Us: t0, sampleRateHz: 96000.0, audioFn: aud96Oob });
        subCursor += nBlk;
      }
      pushFrame({ seq: f, tSrcUs, dtSec: 0.069829, dtRowUs: dt96, exposureUs: 20.0, dispFn: opt96 });
    }
    const tsRej96 = readTelem();
    assert(
      tsRej96.state === RS_STATE_READY && tsRej96.coupledCells === 0,
      `96 kHz 2627 Hz OOB tone must not couple at 2531.25 Hz, got state=${tsRej96.state} coupled=${tsRej96.coupledCells} f=${tsRej96.domFreqHz} g2=${tsRej96.maxCoh}`
    );

    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: dt96, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    subCursor = 0;
    const aud96Match = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * fCut96 * tSec + 0.1);
    for (let f = 0; f < 34; f++) {
      const tSrcUs = t0 + f * 69_829;
      const nextEnd = Math.round((f + 1) * 0.069829 * 96000.0);
      while (subCursor < nextEnd) {
        const nBlk = Math.min(4096, nextEnd - subCursor);
        pushAudioSamples({ startSampleIdx: subCursor, numSamples: nBlk, t0Us: t0, sampleRateHz: 96000.0, audioFn: aud96Match });
        subCursor += nBlk;
      }
      pushFrame({ seq: f, tSrcUs, dtSec: 0.069829, dtRowUs: dt96, exposureUs: 20.0, dispFn: opt96 });
    }
    const tsAcc96 = readTelem();
    assert(
      tsAcc96.state === RS_STATE_READY && tsAcc96.coupledCells === VIBRO_CELLS && tsAcc96.maxCoh > 0.90,
      `96 kHz matched 2625 Hz cutoff tone must couple all 880 cells, got state=${tsAcc96.state} coupled=${tsAcc96.coupledCells} g2=${tsAcc96.maxCoh}`
    );

    // 2) 192 kHz, dtRowUs = 64.0 -> f_opt_max = 7500 Hz (bin 40), 7502 Hz audio must not couple at bin 39 (7312.5 Hz)
    const fCut192 = 0.48 * 1_000_000.0 / 64.0;
    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 64.0, exposureUs: 20.0, audioRateHz: 192000.0, maxSkewUs: 500.0 });
    subCursor = 0;
    const opt192 = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * fCut192 * tSec + 0.3), 0.0];
    const aud192Oob = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * (fCut192 + 2.0) * tSec + 0.1);
    for (let f = 0; f < 34; f++) {
      const tSrcUs = t0 + f * 27_040;
      const nextEnd = Math.round((f + 1) * 0.02704 * 192000.0);
      while (subCursor < nextEnd) {
        const nBlk = Math.min(4096, nextEnd - subCursor);
        pushAudioSamples({ startSampleIdx: subCursor, numSamples: nBlk, t0Us: t0, sampleRateHz: 192000.0, audioFn: aud192Oob });
        subCursor += nBlk;
      }
      pushFrame({ seq: f, tSrcUs, dtSec: 0.02704, dtRowUs: 64.0, exposureUs: 20.0, dispFn: opt192 });
    }
    const tsRej192 = readTelem();
    assert(
      tsRej192.state === RS_STATE_READY && tsRej192.coupledCells === 0,
      `192 kHz 7502 Hz OOB tone must not couple at 7312.5 Hz, got state=${tsRej192.state} coupled=${tsRej192.coupledCells} f=${tsRej192.domFreqHz} g2=${tsRej192.maxCoh}`
    );

    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 64.0, exposureUs: 20.0, audioRateHz: 192000.0, maxSkewUs: 500.0 });
    subCursor = 0;
    const aud192Match = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * fCut192 * tSec + 0.1);
    for (let f = 0; f < 34; f++) {
      const tSrcUs = t0 + f * 27_040;
      const nextEnd = Math.round((f + 1) * 0.02704 * 192000.0);
      while (subCursor < nextEnd) {
        const nBlk = Math.min(4096, nextEnd - subCursor);
        pushAudioSamples({ startSampleIdx: subCursor, numSamples: nBlk, t0Us: t0, sampleRateHz: 192000.0, audioFn: aud192Match });
        subCursor += nBlk;
      }
      pushFrame({ seq: f, tSrcUs, dtSec: 0.02704, dtRowUs: 64.0, exposureUs: 20.0, dispFn: opt192 });
    }
    const tsAcc192 = readTelem();
    assert(
      tsAcc192.state === RS_STATE_READY && tsAcc192.coupledCells === VIBRO_CELLS && tsAcc192.maxCoh > 0.90,
      `192 kHz matched 7500 Hz cutoff tone must couple all 880 cells, got state=${tsAcc192.state} coupled=${tsAcc192.coupledCells} g2=${tsAcc192.maxCoh}`
    );
  }

  // Valid near-cutoff straddle (4850 Hz <= f_opt_max=4873.10 Hz @ dtRow=98.5us) & weak in-band + strong distant OOB (7500 Hz)
  {
    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 98.5, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    sampleCursor = 0;
    const opt4850 = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * 4850.0 * tSec + 0.3), 0.0];
    const aud4850 = (tSec) => 0.5 * Math.cos(2.0 * Math.PI * 4850.0 * tSec + 0.1);
    for (let f = 0; f < 29; f++) {
      const tSrcUs = t0 + f * 41_000;
      const nextEnd = Math.round((f + 1) * 0.041 * 96000.0);
      pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 96000.0, audioFn: aud4850 });
      sampleCursor = nextEnd;
      pushFrame({ seq: f, tSrcUs, dtSec: 0.041, dtRowUs: 98.5, exposureUs: 20.0, dispFn: opt4850 });
    }
    const ts = readTelem();
    assert(
      ts.state === RS_STATE_READY && ts.coupledCells === VIBRO_CELLS && ts.maxCoh > 0.90,
      `Valid 4850 Hz near-cutoff straddle (< 4873.1 Hz) must couple all 880 cells, got state=${ts.state} coupled=${ts.coupledCells} g2=${ts.maxCoh}`
    );

    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 100.0, exposureUs: 20.0, audioRateHz: 96000.0, maxSkewUs: 500.0 });
    sampleCursor = 0;
    const optWeak = (_sx, _sy, _ex, _ey, tSec) => [0.15 * Math.cos(2.0 * Math.PI * 4687.5 * tSec + 0.3), 0.0];
    const audTwoTone = (tSec) =>
      0.025 * Math.cos(2.0 * Math.PI * 4687.5 * tSec + 0.1) +
      0.5 * Math.cos(2.0 * Math.PI * 7500.0 * tSec + 0.4);
    for (let f = 0; f < 29; f++) {
      const tSrcUs = t0 + f * 41_000;
      const nextEnd = Math.round((f + 1) * 0.041 * 96000.0);
      pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 96000.0, audioFn: audTwoTone });
      sampleCursor = nextEnd;
      pushFrame({ seq: f, tSrcUs, dtSec: 0.041, dtRowUs: 100.0, exposureUs: 20.0, dispFn: optWeak });
    }
    const tw = readTelem();
    assert(
      tw.state === RS_STATE_READY && tw.coupledCells === VIBRO_CELLS && tw.maxCoh > 0.90,
      `Weak in-band 4687.5 Hz + strong 7500 Hz OOB tone must couple all 880 cells, got state=${tw.state} coupled=${tw.coupledCells} g2=${tw.maxCoh}`
    );
  }

  // High-epoch sub-microsecond expiry guard (near 2^53 us): exposureUs=201 -> T + 0.5 us must expire immediately
  {
    const tAudioStopBig = (1n << 53n) - 100_000n;
    const t0Big = tAudioStopBig - 1_000_000n;
    initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 201.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
    let curBig = 0;
    const pushFrameBig = (seq, tSrcBig) => {
      for (let ey = 0; ey < FRAME_H; ey++) {
        const tMidSec = (Number(tSrcBig - t0Big) + ey * 25.0 + 100.5) * 1e-6;
        const [ux] = sigOpt(0, ey, 0, ey, tMidSec);
        const rowBase = ey * FRAME_W * 4;
        for (let ex = 0; ex < FRAME_W; ex++) {
          const v = Math.max(0, Math.min(255, Math.round(sampleTexLuma(ex - ux, ey))));
          const o = rowBase + ex * 4;
          scratchRgba[o] = v; scratchRgba[o + 1] = v; scratchRgba[o + 2] = v; scratchRgba[o + 3] = 255;
        }
      }
      const [lo, hi] = splitU64(tSrcBig);
      assert(
        exp.chronos_set_ingest_meta(
          1, seq, lo, hi, -1.0, INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_GEOM_KNOWN,
          640, 360, 0, 0, 640, 360, 0, 0, 0, 0, 640, 360, 0, 1
        ) === 1,
        `High-epoch frame reject seq=${seq}`
      );
      new Uint8Array(mem.buffer, exp.chronos_get_ingest_rgba_ptr(), FRAME_BYTES).set(scratchRgba);
      exp.chronos_pre_tick(1 / 30);
      exp.chronos_post_tick(1 / 30);
    };
    for (let f = 0; f < 30; f++) {
      const tSrcBig = t0Big + BigInt(Math.round((f * 1e6) / 30));
      const nextEnd = Math.round(((f + 1) / 30) * 48000);
      const nBlk = nextEnd - curBig;
      for (let i = 0; i < nBlk; i++) {
        audioStaging[i] = sigAud((curBig + i) / 48000.0);
      }
      const tEndBig = t0Big + BigInt(Math.round((nextEnd * 1e6) / 48000));
      const [aLo, aHi] = splitU64(tEndBig);
      assert(exp.chronos_push_audio_block(nBlk, aLo, aHi, 48000.0, 0.0) === 1, `High-epoch audio f=${f}`);
      curBig = nextEnd;
      pushFrameBig(f, tSrcBig);
    }
    assert(readTelem().state === RS_STATE_READY && readTelem().stampValid, 'High-epoch session must reach READY');
    pushFrameBig(30, t0Big + 972_000n);
    assert(readTelem().state === RS_STATE_READY && (wave2U32[16] & 3) === 3, 'High-epoch covered sub-hop frame must hold stamp');
    pushFrameBig(31, tAudioStopBig - 100n);
    const vExp = readTelem();
    assert(
      vExp.state === RS_STATE_WARMING && (vExp.reason & RS_REASON_AUDIO_UNDERRUN) !== 0 && !vExp.stampValid,
      `High-epoch T+0.5us uncovered frame must expire immediately, got state=${vExp.state} reason=${vExp.reason} valid=${vExp.stampValid}`
    );
  }

  // SD 320x180 transposed column scan (o=0, scanDir=2) fits 320 <= 360 and reaches READY without AXIS_TRANSPOSED
  {
    const sdW = 320, sdH = 180, sdBytes = sdW * sdH * 4;
    const sdBuf = new Uint8Array(sdBytes);
    exp.chronos_init();
    exp.chronos_set_resolution(0);
    exp.chronos_set_swarm_mode(0);
    assert(exp.chronos_set_rs_calib(RS_MODE_RS_ONLY, 25.0, 200.0, 0.0, 500.0) === 1);
    for (let f = 0; f < 24; f++) {
      const tSrcUs = t0 + Math.round((f * 1e6) / 30);
      for (let ey = 0; ey < sdH; ey++) {
        for (let ex = 0; ex < sdW; ex++) {
          const tMidSec = (tSrcUs + ex * 25.0 + 100.0) * 1e-6;
          const ux = 0.45 * Math.cos(2.0 * Math.PI * 421.875 * tMidSec + 0.5);
          const v = Math.max(0, Math.min(255, Math.round(sampleTexLuma(ex - ux, ey))));
          const o = (ey * sdW + ex) * 4;
          sdBuf[o] = v; sdBuf[o + 1] = v; sdBuf[o + 2] = v; sdBuf[o + 3] = 255;
        }
      }
      const [tLo, tHi] = splitU64(BigInt(tSrcUs));
      assert(
        exp.chronos_set_ingest_meta(
          1, f + 1, tLo, tHi, -1.0, INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_GEOM_KNOWN,
          sdW, sdH, 0, 0, sdW, sdH, 0, 2, 0, 0, sdW, sdH, 0, 1
        ) === 1,
        `SD o=0,scanDir=2 frame ${f} ingest failed`
      );
      new Uint8Array(mem.buffer, exp.chronos_get_ingest_rgba_ptr(), sdBytes).set(sdBuf);
      exp.chronos_pre_tick(1 / 30);
      exp.chronos_post_tick(1 / 30);
    }
    const tsd = readTelem();
    assert(
      (tsd.reason & RS_REASON_AXIS_TRANSPOSED) === 0 &&
        tsd.state === RS_STATE_READY &&
        tsd.stampValid &&
        Math.abs(tsd.domFreqHz - 421.875) < 35.0,
      `SD 320x180 o=0,scanDir=2 expected READY ~421.875 Hz, got ${JSON.stringify(tsd)}`
    );
    exp.chronos_set_resolution(1);
  }

  // Sequence gap (gapMask = 1) resets continuity then recovers after re-warming
  initSession({ mode: RS_MODE_RS_WELCH, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 48000.0, maxSkewUs: 500.0 });
  sampleCursor = 0;
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY && readTelem().stampValid, 'Expected READY before gap');

  // Gap frame
  {
    const f = 30;
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f + 5, tSrcUs, gapMask: 1, dispFn: sigOpt });
    t = readTelem();
    assert(t.state === RS_STATE_DISTURBED && (t.reason & RS_REASON_FRAME_GAP) !== 0, 'Gap frame must trigger DISTURBED + FRAME_GAP');
    assert(!t.stampValid && t.kSegs === 0, 'Gap frame must invalidate wave2 stamp and Welch slots');
  }
  // Recover over 30 clean frames
  for (let f = 31; f < 61; f++) {
    const tSrcUs = t0 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    pushAudioSamples({ startSampleIdx: sampleCursor, numSamples: nextEnd - sampleCursor, t0Us: t0, sampleRateHz: 48000.0, audioFn: sigAud });
    sampleCursor = nextEnd;
    pushFrame({ seq: f + 5, tSrcUs, dispFn: sigOpt });
  }
  t = readTelem();
  assert(t.state === RS_STATE_READY && t.stampValid && t.kSegs >= 8, 'Expected clean recovery to READY after re-warming');

  // Flat texture -> RS_REASON_LOW_GRAD
  initSession({ mode: RS_MODE_RS_ONLY, dtRowUs: 25.0, exposureUs: 200.0, audioRateHz: 0.0, maxSkewUs: 500.0 });
  for (let f = 0; f < 17; f++) {
    pushFrame({ seq: f, tSrcUs: t0 + Math.round((f * 1e6) / 30), flatLuma: 128 });
  }
  t = readTelem();
  assert(t.state === RS_STATE_DISTURBED && (t.reason & RS_REASON_LOW_GRAD) !== 0, 'Flat texture must trigger LOW_GRAD');

  console.log('  [PASS] Suite 5 (Fail-Closed & Continuity Resets): bad_calib, no_scan_dir, not_native_1to1, no_pts, axis_transposed, clock_skew, audio_gap, frame_gap+recovery, low_grad');
}

console.log('ALL GATE I2 CHECKS PASSED.');
