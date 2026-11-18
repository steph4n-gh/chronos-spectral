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

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
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

const VIBRO_CELL_W = 40;
const VIBRO_CELL_H = 22;
const VIBRO_CELLS = VIBRO_CELL_W * VIBRO_CELL_H; // 880
const VIBRO_SPEC_BINS = 64;
const VIBRO_PLANES_LEN = 3 * VIBRO_CELLS + VIBRO_SPEC_BINS; // 2704
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
  assert(wave2U32[1] <= 24_117_248, `Arena bytes ${wave2U32[1]} exceeds 24117248`);
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
  assert(loc.domBand >= 4 && loc.domBand < 8, `Expected domBand in 4..7, got ${loc.domBand}`);
  assert(loc.validCells === 220 && loc.coupledCells === 220, `Expected 220 active cells, got valid=${loc.validCells}, coupled=${loc.coupledCells}`);

  let activeSum = 0.0;
  let activeCount = 0;
  let quietMax = 0.0;
  for (let cy = 0; cy < VIBRO_CELL_H; cy++) {
    for (let cx = 0; cx < VIBRO_CELL_W; cx++) {
      const idx = cy * VIBRO_CELL_W + cx;
      const coh = planesF32[VIBRO_PLANE_COHERENCE + idx];
      const amp = planesF32[VIBRO_PLANE_AMP + idx];
      if (cx >= 10 && cx < 20) {
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
    `  [PASS] Suite 4 (Localized 40x22 Coupling): domBand=${loc.domBand}, activeMeanCoh=${activeMean.toFixed(4)}, ` +
      `quietMaxCoh=${quietMax.toFixed(4)}, validCells=${loc.validCells}/880`
  );
}

// ============================================================================
// Suite 5: Fail-Closed Matrix & Continuity Resets
// ============================================================================
{
  // 5a. Bad calibration rejected
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  assert(exp.chronos_set_rs_calib(RS_MODE_RS_WELCH, 0.0, 200.0, 48000.0, 500.0) === 0, 'dt_row=0 must fail');
  let t = readTelem();
  assert(t.state === RS_STATE_UNCALIBRATED && (t.reason & RS_REASON_BAD_CALIB) !== 0, 'Bad calib state');

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
  assert(readTelem().kSegs === 0 && (readTelem().reason & RS_REASON_AUDIO_UNDERRUN) !== 0, 'Backward audio must reset Welch count');

  // Forward audio gap (> 2.5 * dt_audio) resets Welch interval cleanly
  for (let f = 0; f < 30; f++) {
    const tSrcUs = t0 + 1_000_000 + Math.round((f * 1e6) / 30);
    const nextEnd = Math.round(((f + 1) / 30) * 48000);
    const blkN = nextEnd - (f === 0 ? 0 : Math.round((f / 30) * 48000));
    pushAudioSamples({
      startSampleIdx: f === 0 ? 0 : Math.round((f / 30) * 48000),
      numSamples: blkN,
      t0Us: t0 + 1_000_000,
      sampleRateHz: 48000.0,
      audioFn: sigAud,
    });
    pushFrame({ seq: 30 + f, tSrcUs, dispFn: sigOpt });
  }
  assert(readTelem().state === RS_STATE_READY, 'Expected recovery after forward audio interval jump');
  const tGapJump = t0 + 5_000_000;
  const [gLo, gHi] = splitU64(BigInt(tGapJump));
  assert(exp.chronos_push_audio_block(1600, gLo, gHi, 48000.0, 0.0) === 1, 'Forward gap starts fresh interval');
  assert(readTelem().kSegs === 0 && (readTelem().reason & RS_REASON_AUDIO_UNDERRUN) !== 0, 'Forward audio gap must reset Welch count');

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
