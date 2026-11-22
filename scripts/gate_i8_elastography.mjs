#!/usr/bin/env node
// ============================================================================
// CHRONOS-SPECTRAL // Gate I8: kHz Optical Elastography, Structural Power Flow
//                      & Lock-In Acoustic BOS (Wave D1)
//
// Verifies:
//   I8.0  Wasm ABI, 448-page arena, zero em dash, zero buffer overlap
//   I8.1  elasto_hidden_crack_panel.mp4 (380+860 Hz Kirchhoff plate: alpha~0.5,
//         stiff vertical rib contrast, source & crack energy sink localization)
//   I8.2  elasto_guitar_soundboard.mp4 (247.5+495 Hz anisotropic spruce:
//         beta1>=1 soundhole, c_x/c_y anisotropy, bridge source)
//   I8.3  elasto_gelatin_phantom.mp4 (110+220 Hz non-dispersive shear wave:
//         alpha~1.0 MEMBRANE_SHEAR, buried circular stiff inclusion contrast)
//   I8.4  acoustic_bos_standing_wave.mp4 (1,200 Hz Mode 4 Lock-In Acoustic BOS:
//         256x128 |Phi| recovery vs exact ground truth r > 0.90, low curl)
//   I8.5  Negative controls (rigid-body oscillation -> WAVELENGTH_TOO_LONG,
//         exposure null -> EXPOSURE_NULL, BOS mutex conflict + clean recovery),
//         Mode 2 self-locked check, and real-camera Mode 3 checks.
//
// Usage:
//   node scripts/gate_i8_elastography.mjs [--wasm=<path>] [--parity-lite]
// ============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_WASM = path.join(ROOT, 'web', 'chronos_spectral.wasm');
const SAMPLES_DIR = path.join(ROOT, 'web', 'samples');
const args = process.argv.slice(2);
const wasmArg = args.find((a) => a.startsWith('--wasm='));
const WASM_PATH = wasmArg ? path.resolve(wasmArg.slice('--wasm='.length)) : DEFAULT_WASM;
const PARITY_LITE = args.includes('--parity-lite');

const FRAME_W = 640;
const FRAME_H = 360;
const FRAME_BYTES = FRAME_W * FRAME_H * 4;
const ELASTO_W = 64;
const ELASTO_H = 36;
const ELASTO_V = ELASTO_W * ELASTO_H; // 2,304
const BOS_W = 256;
const BOS_H = 128;
const BOS_V = BOS_W * BOS_H; // 32,768

const ELASTO_STATE_OFF = 0;
const ELASTO_STATE_UNCALIBRATED = 1;
const ELASTO_STATE_WARMING = 2;
const ELASTO_STATE_READY = 3;
const ELASTO_STATE_REFUSED = 4;

const ELASTO_CLASS_KIRCHHOFF_PLATE = 1;
const ELASTO_CLASS_MEMBRANE_SHEAR = 2;

const RS_REASON_EXPOSURE_NULL = 1 << 6;
const ELASTO_REASON_WAVELENGTH_TOO_LONG = 1 << 17;
const ELASTO_REASON_MUTEX_CONFLICT = 1 << 19;

const INGEST_FLAG_PTS_KNOWN = 1 << 1;
const INGEST_FLAG_GEOM_KNOWN = 1 << 4;

function assert(cond, msg) {
  if (!cond) throw new Error(`Gate I8 FAIL: ${msg}`);
}

function splitU64(vBig) {
  return [Number(BigInt.asUintN(32, vBig)), Number(BigInt.asUintN(32, vBig >> 32n))];
}

function pearsonR(a, b) {
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) continue;
    sa += a[i];
    sb += b[i];
    n++;
  }
  if (n < 8) return 0;
  const ma = sa / n;
  const mb = sb / n;
  let num = 0;
  let da2 = 0;
  let db2 = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) continue;
    const da = a[i] - ma;
    const db = b[i] - mb;
    num += da * db;
    da2 += da * da;
    db2 += db * db;
  }
  return da2 > 1e-18 && db2 > 1e-18 ? num / Math.sqrt(da2 * db2) : 0;
}

function median(arr) {
  if (!arr.length) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  return s[s.length >> 1];
}

// ============================================================================
// I8.0 Source Hygiene & Wasm ABI Audit
// ============================================================================
for (const rel of ['src/elastography.rs', 'scripts/gate_i8_elastography.mjs']) {
  const txt = fs.readFileSync(path.join(ROOT, rel), 'utf8');
  assert(!txt.includes('\u2014'), `${rel}: contains forbidden em dash`);
}

assert(fs.existsSync(WASM_PATH), `Wasm binary not found: ${WASM_PATH}`);
const wasmBytes = fs.readFileSync(WASM_PATH);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const exp = instance.exports;
const mem = exp.memory;

for (const fn of [
  'chronos_init',
  'chronos_set_resolution',
  'chronos_set_swarm_mode',
  'chronos_set_bos_mode',
  'chronos_set_rs_calib',
  'chronos_get_audio_staging_ptr',
  'chronos_push_audio_block',
  'chronos_set_ingest_meta',
  'chronos_get_ingest_rgba_ptr',
  'chronos_pre_tick',
  'chronos_post_tick',
  'chronos_get_wave2_telemetry_ptr',
  'chronos_set_elasto_mode',
  'chronos_set_elasto_lock_freqs',
  'chronos_get_elasto_telemetry_ptr',
  'chronos_get_elasto_planes_ptr',
  'chronos_get_elasto_valid_ptr',
  'chronos_get_bos_planes_ptr',
  'chronos_get_bos_valid_ptr',
]) {
  assert(typeof exp[fn] === 'function', `Missing required export ${fn}`);
}

assert(mem.buffer.byteLength === 448 * 65536, `Expected 448 pages, got ${mem.buffer.byteLength}`);
let grew = false;
try {
  mem.grow(1);
  grew = true;
} catch {}
assert(!grew, 'Expected fixed max=448 Wasm memory');

const buf = mem.buffer;
const wave2Ptr = exp.chronos_get_wave2_telemetry_ptr();
const elastoTelemPtr = exp.chronos_get_elasto_telemetry_ptr();
const elastoPlanesPtr = exp.chronos_get_elasto_planes_ptr();
const elastoValidPtr = exp.chronos_get_elasto_valid_ptr();
const bosPlanesPtr = exp.chronos_get_bos_planes_ptr();
const bosValidPtr = exp.chronos_get_bos_valid_ptr();
const audioStagingPtr = exp.chronos_get_audio_staging_ptr();

assert(elastoPlanesPtr === bosPlanesPtr, 'elasto_planes must alias bos_planes');
assert(elastoValidPtr === bosValidPtr, 'elasto_valid must alias bos_valid');
assert((elastoTelemPtr & 3) === 0 && elastoTelemPtr + 128 <= buf.byteLength, 'elastoTelemPtr invalid');
assert(
  !(wave2Ptr < elastoTelemPtr + 128 && elastoTelemPtr < wave2Ptr + 256),
  'wave2_telem and elasto_telem overlap'
);

const wave2U32 = new Uint32Array(buf, wave2Ptr, 64);
const telemU32 = new Uint32Array(buf, elastoTelemPtr, 32);
const telemF32 = new Float32Array(buf, elastoTelemPtr, 32);
const surfPlanesF32 = new Float32Array(buf, elastoPlanesPtr, 6 * ELASTO_V);
const surfValidU8 = new Uint8Array(buf, elastoValidPtr, ELASTO_V);
const abosPlanesF32 = new Float32Array(buf, elastoPlanesPtr, 6 * BOS_V);
const abosValidU8 = new Uint8Array(buf, elastoValidPtr, BOS_V);
const audioStaging = new Float32Array(buf, audioStagingPtr, 4096);

function readElastoTelem() {
  for (let k = 8; k <= 27; k++) {
    assert(Number.isFinite(telemF32[k]), `Non-finite elasto_telem[${k}]=${telemF32[k]}`);
  }
  return {
    state: telemU32[0] >>> 0,
    mode: telemU32[1] >>> 0,
    reason: telemU32[2] >>> 0,
    dispClass: telemU32[3] >>> 0,
    validCells: telemU32[4] >>> 0,
    lockedFreqs: telemU32[5] >>> 0,
    welchSegs: telemU32[6] >>> 0,
    beta1: telemU32[7] >>> 0,
    f0: telemF32[8],
    f1: telemF32[9],
    f2: telemF32[10],
    f3: telemF32[11],
    medC: telemF32[12],
    alpha: telemF32[13],
    anisotropy: telemF32[14],
    meanCoh: telemF32[15],
    srcX: telemF32[16],
    srcY: telemF32[17],
    sinkX: telemF32[18],
    sinkY: telemF32[19],
    coexactFrac: telemF32[20],
    exactFrac: telemF32[21],
    medK: telemF32[22],
    sincH0: telemF32[23],
    abosCurlRms: telemF32[24],
    abosPhiRms: telemF32[25],
    cP10: telemF32[26],
    cP90: telemF32[27],
    gridW: telemU32[28] >>> 0,
    gridH: telemU32[29] >>> 0,
    provenance: telemU32[30] >>> 0,
    stampValid: (wave2U32[56] & 1) !== 0,
  };
}

function decodeMp4VideoRgba(mp4Path, maxFrames) {
  const raw = execFileSync(
    'ffmpeg',
    [
      '-v', 'error',
      '-i', mp4Path,
      '-vf', `scale=${FRAME_W}:${FRAME_H}:flags=lanczos`,
      '-frames:v', String(maxFrames),
      '-f', 'rawvideo',
      '-pix_fmt', 'rgba',
      '-',
    ],
    { maxBuffer: 512 * 1024 * 1024 }
  );
  const nFrames = Math.floor(raw.length / FRAME_BYTES);
  return { raw, nFrames };
}

function decodeMp4AudioF32(mp4Path) {
  const raw = execFileSync(
    'ffmpeg',
    ['-v', 'error', '-i', mp4Path, '-vn', '-f', 'f32le', '-ac', '1', '-ar', '48000', '-'],
    { maxBuffer: 64 * 1024 * 1024 }
  );
  return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength >> 2);
}

function probeFps(mp4Path) {
  const txt = execFileSync(
    'ffprobe',
    ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=nw=1:nk=1', mp4Path],
    { encoding: 'utf8' }
  ).trim();
  const [n, d] = txt.split('/').map(Number);
  return d ? n / d : n || 30.0;
}

function pushVideoFrame(rgbaSub, seq, tSrcUs, dtSec, scanDir = 0, native1to1 = 1) {
  const [tLo, tHi] = splitU64(BigInt(Math.round(tSrcUs)));
  const ok = exp.chronos_set_ingest_meta(
    1,
    seq,
    tLo,
    tHi,
    -1.0,
    INGEST_FLAG_PTS_KNOWN | INGEST_FLAG_GEOM_KNOWN,
    FRAME_W,
    FRAME_H,
    0,
    0,
    FRAME_W,
    FRAME_H,
    0,
    scanDir,
    0,
    0,
    FRAME_W,
    FRAME_H,
    0,
    native1to1
  );
  assert(ok === 1, `chronos_set_ingest_meta rejected seq=${seq}`);
  const dstPtr = exp.chronos_get_ingest_rgba_ptr();
  new Uint8Array(buf, dstPtr, FRAME_BYTES).set(rgbaSub);
  exp.chronos_pre_tick(dtSec);
  exp.chronos_post_tick(dtSec);
}

function pushAudioSlice(samples, startIdx, count, t0Us, sampleRateHz = 48000.0) {
  const n = Math.min(count, samples.length - startIdx);
  if (n <= 0) return 1;
  let pos = 0;
  while (pos < n) {
    const chunk = Math.min(2048, n - pos);
    audioStaging.set(samples.subarray(startIdx + pos, startIdx + pos + chunk), 0);
    const endSample = startIdx + pos + chunk;
    const tEndUs = BigInt(t0Us) + BigInt(Math.round((endSample * 1_000_000) / sampleRateHz));
    const [tLo, tHi] = splitU64(tEndUs);
    const ok = exp.chronos_push_audio_block(chunk, tLo, tHi, sampleRateHz, 0.0);
    assert(ok === 1, `chronos_push_audio_block failed at sample ${endSample}`);
    pos += chunk;
  }
  return 1;
}

function runMp4Elasto({
  file,
  mode,
  dtRowUs = 35.0,
  exposureUs = 200.0,
  lockFreqs = [],
  maxFrames = 140,
  hasAudio = true,
  sampleRateHz = 48000.0,
}) {
  const mp4Path = path.join(SAMPLES_DIR, file);
  assert(fs.existsSync(mp4Path), `Missing sample clip ${mp4Path}`);
  const fps = probeFps(mp4Path);
  const dtSec = 1.0 / fps;
  const { raw, nFrames } = decodeMp4VideoRgba(mp4Path, maxFrames);
  const audio = hasAudio ? decodeMp4AudioF32(mp4Path) : null;

  exp.chronos_init();
  exp.chronos_set_resolution(1);
  exp.chronos_set_swarm_mode(0);
  if (mode === 1 || mode === 2 || mode === 4) {
    const rsMode = hasAudio ? 2 : 1;
    assert(
      exp.chronos_set_rs_calib(rsMode, dtRowUs, exposureUs, sampleRateHz, 5000.0) === 1,
      'chronos_set_rs_calib failed'
    );
  }
  assert(exp.chronos_set_elasto_mode(mode) === 1, `chronos_set_elasto_mode(${mode}) failed`);
  if (lockFreqs.length > 0) {
    const f = [lockFreqs[0] || 0, lockFreqs[1] || 0, lockFreqs[2] || 0, lockFreqs[3] || 0];
    assert(exp.chronos_set_elasto_lock_freqs(lockFreqs.length, f[0], f[1], f[2], f[3]) === 1, 'set_lock_freqs failed');
  } else {
    assert(exp.chronos_set_elasto_lock_freqs(0, 0, 0, 0, 0) === 1, 'clear lock_freqs failed');
  }

  const t0Us = 1_000_000;
  let audioCursor = 0;
  let readyFrames = 0;
  let lastReady = null;
  let tickMsSum = 0;

  // Pre-roll 60 ms of audio past frame 0 so rolling-shutter scanline window is always covered
  for (let f = 0; f < nFrames; f++) {
    const tSrcUs = t0Us + Math.round((f * 1_000_000) / fps);
    if (audio) {
      // Ensure audio ring covers through tSrcUs + 360 * dtRowUs + 15 ms margin
      const targetSec = (f + 1) / fps + 0.025;
      const targetSample = Math.min(audio.length, Math.round(targetSec * sampleRateHz));
      if (targetSample > audioCursor) {
        pushAudioSlice(audio, audioCursor, targetSample - audioCursor, t0Us, sampleRateHz);
        audioCursor = targetSample;
      }
    }
    const frameView = raw.subarray(f * FRAME_BYTES, (f + 1) * FRAME_BYTES);
    const tStart = performance.now();
    pushVideoFrame(frameView, f + 1, tSrcUs, dtSec, 0, 1);
    tickMsSum += performance.now() - tStart;
    const t = readElastoTelem();
    if (t.state === ELASTO_STATE_READY) {
      readyFrames++;
      lastReady = t;
    }
  }
  return {
    nFrames,
    readyFrames,
    last: readElastoTelem(),
    lastReady,
    msPerFrame: tickMsSum / Math.max(1, nFrames),
  };
}

function sampleRegionMedianC(x0, x1, y0, y1) {
  const vals = [];
  for (let r = 0; r < ELASTO_H; r++) {
    const py = (r + 0.5) * (FRAME_H / ELASTO_H);
    if (py < y0 || py > y1) continue;
    for (let c = 0; c < ELASTO_W; c++) {
      const px = (c + 0.5) * (FRAME_W / ELASTO_W);
      if (px < x0 || px > x1) continue;
      const v = r * ELASTO_W + c;
      if (surfValidU8[v] && surfPlanesF32[v] > 0) {
        vals.push(surfPlanesF32[v]);
      }
    }
  }
  return median(vals);
}

// ============================================================================
// I8.5a Quick Synthetic Negative Controls & Mutex (runs in BOTH parity-lite & full)
// ============================================================================
{
  const synthRgba = new Uint8Array(FRAME_BYTES);
  const sampleTex = (x, y) => {
    const v =
      128 +
      42 * Math.sin(0.29 * x + 0.17 * y) +
      36 * Math.cos(0.21 * x - 0.33 * y) +
      24 * Math.sin(0.41 * x + 0.27 * y);
    return Math.max(8, Math.min(247, Math.round(v)));
  };

  // 1) Mutex conflict with Thermal BOS + clean recovery
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  exp.chronos_set_rs_calib(2, 35.0, 200.0, 48000.0, 5000.0);
  exp.chronos_set_elasto_mode(1);
  exp.chronos_set_bos_mode(1);
  for (let i = 0; i < FRAME_BYTES; i += 4) {
    const p = i >> 2;
    const v = sampleTex(p % FRAME_W, Math.floor(p / FRAME_W));
    synthRgba[i] = v;
    synthRgba[i + 1] = v;
    synthRgba[i + 2] = v;
    synthRgba[i + 3] = 255;
  }
  pushVideoFrame(synthRgba, 1, 1_000_000, 1 / 30);
  const tBusy = readElastoTelem();
  assert(tBusy.state === ELASTO_STATE_REFUSED, `Expected REFUSED under BOS mutex, got state=${tBusy.state}`);
  assert((tBusy.reason & ELASTO_REASON_MUTEX_CONFLICT) !== 0, `Expected MUTEX_CONFLICT reason bit, got 0x${tBusy.reason.toString(16)}`);
  exp.chronos_set_bos_mode(0);
  pushVideoFrame(synthRgba, 2, 1_033_333, 1 / 30);
  const tRec = readElastoTelem();
  assert((tRec.reason & ELASTO_REASON_MUTEX_CONFLICT) === 0, 'Mutex conflict bit did not clear after BOS release');
  console.log('  [PASS] I8.5a BOS/Elasto mutex conflict refused (0x80000) & recovered cleanly');

  // 2) Exposure null refusal (f = 500 Hz, T_e = 2000 us -> sinc(pi * 1.0) = 0)
  exp.chronos_init();
  exp.chronos_set_resolution(1);
  exp.chronos_set_rs_calib(2, 35.0, 2000.0, 48000.0, 5000.0);
  exp.chronos_set_elasto_mode(1);
  exp.chronos_set_elasto_lock_freqs(1, 500.0, 0, 0, 0);
  const sr = 48000.0;
  const t0Us = 1_000_000;
  for (let f = 0; f < 22; f++) {
    const nAud = 1600;
    for (let i = 0; i < nAud; i++) {
      const tSec = (f * nAud + i) / sr;
      audioStaging[i] = 0.7 * Math.sin(2 * Math.PI * 500.0 * tSec);
    }
    const [aLo, aHi] = splitU64(BigInt(t0Us) + BigInt(Math.round(((f + 1) * nAud * 1e6) / sr)));
    exp.chronos_push_audio_block(nAud, aLo, aHi, sr, 0.0);
    pushVideoFrame(synthRgba, f + 1, t0Us + Math.round((f * 1e6) / 30), 1 / 30);
  }
  const tNull = readElastoTelem();
  assert(tNull.state === ELASTO_STATE_REFUSED, `Expected REFUSED on exposure null, got state=${tNull.state}`);
  assert((tNull.reason & RS_REASON_EXPOSURE_NULL) !== 0, `Expected RS_REASON_EXPOSURE_NULL, got 0x${tNull.reason.toString(16)}`);
  console.log(`  [PASS] I8.5b Exposure null (500 Hz @ 2000 us) refused cleanly (reason=0x${tNull.reason.toString(16)})`);

  // 3) Rigid-body zero-phase-gradient oscillation (k = 0 -> WAVELENGTH_TOO_LONG refusal)
  if (!PARITY_LITE) {
    exp.chronos_init();
    exp.chronos_set_resolution(1);
    exp.chronos_set_rs_calib(2, 35.0, 200.0, 48000.0, 5000.0);
    exp.chronos_set_elasto_mode(1);
    exp.chronos_set_elasto_lock_freqs(1, 380.0, 0, 0, 0);
    for (let f = 0; f < 64; f++) {
      const nAud = 1600;
      for (let i = 0; i < nAud; i++) {
        const tSec = (f * nAud + i) / sr;
        audioStaging[i] = 0.7 * Math.sin(2 * Math.PI * 380.0 * tSec);
      }
      const [aLo, aHi] = splitU64(BigInt(t0Us) + BigInt(Math.round(((f + 1) * nAud * 1e6) / sr)));
      exp.chronos_push_audio_block(nAud, aLo, aHi, sr, 0.0);
      const tFrame = f / 30.0;
      for (let y = 0; y < FRAME_H; y++) {
        const u = f >= 18 ? 0.25 * Math.sin(2 * Math.PI * 380.0 * (tFrame + y * 35e-6)) : 0.0;
        const row = y * FRAME_W * 4;
        for (let x = 0; x < FRAME_W; x++) {
          const v = sampleTex(x - u, y);
          const o = row + x * 4;
          synthRgba[o] = v;
          synthRgba[o + 1] = v;
          synthRgba[o + 2] = v;
          synthRgba[o + 3] = 255;
        }
      }
      pushVideoFrame(synthRgba, f + 1, t0Us + Math.round((f * 1e6) / 30), 1 / 30);
    }
    const tRigid = readElastoTelem();
    assert(tRigid.state === ELASTO_STATE_REFUSED, `Expected REFUSED on rigid-body k=0 oscillation, got state=${tRigid.state} medC=${tRigid.medC}`);
    assert(
      (tRigid.reason & ELASTO_REASON_WAVELENGTH_TOO_LONG) !== 0,
      `Expected WAVELENGTH_TOO_LONG on rigid-body oscillation, got reason=0x${tRigid.reason.toString(16)}`
    );
    console.log(`  [PASS] I8.5c Rigid-body oscillation (k=0) refused cleanly (reason=0x${tRigid.reason.toString(16)})`);
  }
}

// ============================================================================
// I8.1 Hidden Crack Panel (Kirchhoff Plate: 380 + 860 Hz, alpha = 0.5)
// ============================================================================
{
  const res = runMp4Elasto({
    file: 'elasto_hidden_crack_panel.mp4',
    mode: 1,
    dtRowUs: 35.0,
    exposureUs: 200.0,
    lockFreqs: [], // auto-detect 380 + 860 Hz from 48 kHz muxed audio!
    maxFrames: PARITY_LITE ? 55 : 150,
    hasAudio: true,
  });
  const t = res.lastReady || res.last;
  const cBgLeft = sampleRegionMedianC(150, 260, 80, 280);
  const cRib = sampleRegionMedianC(300, 350, 80, 280);
  const ribRatio = cBgLeft > 0 ? cRib / cBgLeft : 0;
  const srcDist = Math.hypot(t.srcX - 92, t.srcY - 180);
  const sinkDist = Math.hypot(t.sinkX - 495, t.sinkY - 195);
  console.log(
    `  [INFO] I8.1 Crack Panel: ready=${res.readyFrames}/${res.nFrames} state=${t.state} reason=0x${t.reason.toString(16)} ` +
    `freqs=[${t.f0.toFixed(1)},${t.f1.toFixed(1)}] valid=${t.validCells} coh=${t.meanCoh.toFixed(3)} ` +
    `medC=${t.medC.toFixed(0)} px/s alpha=${t.alpha.toFixed(3)} class=${t.dispClass} ` +
    `ribRatio=${ribRatio.toFixed(2)} (cBg=${cBgLeft.toFixed(0)}, cRib=${cRib.toFixed(0)}) ` +
    `src=(${t.srcX.toFixed(0)},${t.srcY.toFixed(0)}) [d=${srcDist.toFixed(0)}] ` +
    `sink=(${t.sinkX.toFixed(0)},${t.sinkY.toFixed(0)}) [d=${sinkDist.toFixed(0)}] ` +
    `ms/f=${res.msPerFrame.toFixed(2)}`
  );
  assert(res.readyFrames > 0, `I8.1 never reached READY (last state=${t.state} reason=0x${t.reason.toString(16)})`);
  assert(t.stampValid, 'I8.1 wave2_telem[56] freshness stamp invalid');
  assert(Math.abs(t.f0 - 380) < 25 && Math.abs(t.f1 - 860) < 25, `I8.1 auto-detected wrong freqs [${t.f0}, ${t.f1}]`);
  assert(t.dispClass === ELASTO_CLASS_KIRCHHOFF_PLATE, `I8.1 expected KIRCHHOFF_PLATE (1), got ${t.dispClass} (alpha=${t.alpha})`);
  assert(t.alpha >= 0.35 && t.alpha <= 0.65, `I8.1 alpha=${t.alpha.toFixed(3)} out of [0.35, 0.65]`);
  if (!PARITY_LITE) {
    assert(ribRatio >= 1.18, `I8.1 stiff rib ratio ${ribRatio.toFixed(2)} < 1.18`);
    assert(srcDist <= 110, `I8.1 source (${t.srcX.toFixed(0)},${t.srcY.toFixed(0)}) too far from (92,180)`);
    assert(sinkDist <= 110, `I8.1 crack sink (${t.sinkX.toFixed(0)},${t.sinkY.toFixed(0)}) too far from (495,195)`);
  }
  console.log('  [PASS] I8.1 Hidden Crack Panel (Kirchhoff plate + rib + source/sink)');
}

// ============================================================================
// I8.4 Lock-In Acoustic BOS (1,200 Hz standing wave in air, 256x128)
// ============================================================================
{
  const res = runMp4Elasto({
    file: 'acoustic_bos_standing_wave.mp4',
    mode: 4,
    dtRowUs: 35.0,
    exposureUs: 180.0,
    lockFreqs: [1200.0],
    maxFrames: PARITY_LITE ? 55 : 160,
    hasAudio: true,
  });
  const t = res.lastReady || res.last;

  // Build exact ground-truth |Phi(x,y)| on 256x128 from generate_v33_demos.mjs
  const kx = (2 * Math.PI) / 140.0;
  const gtMag = new Float64Array(BOS_V);
  const recMag = new Float64Array(BOS_V);
  for (let j = 0; j < BOS_H; j++) {
    const y = 3.5 + (j * (FRAME_H - 8.0)) / (BOS_H - 1.0);
    const dy = (y - 180.0) / 88.0;
    const envY = Math.exp(-dy * dy);
    for (let i = 0; i < BOS_W; i++) {
      const x = 3.5 + (i * (FRAME_W - 8.0)) / (BOS_W - 1.0);
      const envX = x >= 70 && x <= 560 ? Math.sqrt(Math.sin((Math.PI * (x - 70)) / 490.0)) : 0.0;
      const idx = j * BOS_W + i;
      gtMag[idx] = Math.abs(4.2 * envY * envX * Math.cos(kx * (x - 560.0)));
      recMag[idx] = abosValidU8[idx] ? abosPlanesF32[idx] : NaN;
    }
  }
  // Also evaluate inside the active beam interior [100..530] x [60..300]
  const gtBeam = [];
  const recBeam = [];
  for (let j = 0; j < BOS_H; j++) {
    const y = 3.5 + (j * (FRAME_H - 8.0)) / (BOS_H - 1.0);
    if (y < 60 || y > 300) continue;
    for (let i = 0; i < BOS_W; i++) {
      const x = 3.5 + (i * (FRAME_W - 8.0)) / (BOS_W - 1.0);
      if (x < 100 || x > 530) continue;
      const idx = j * BOS_W + i;
      if (abosValidU8[idx]) {
        gtBeam.push(gtMag[idx]);
        recBeam.push(abosPlanesF32[idx]);
      }
    }
  }
  const rFull = pearsonR(gtMag, recMag);
  const rBeam = pearsonR(gtBeam, recBeam);
  const curlRatio = t.abosPhiRms > 1e-9 ? t.abosCurlRms / t.abosPhiRms : 99;
  console.log(
    `  [INFO] I8.4 Acoustic BOS 1200 Hz: ready=${res.readyFrames}/${res.nFrames} state=${t.state} ` +
    `grid=${t.gridW}x${t.gridH} valid=${t.validCells} phiRms=${t.abosPhiRms.toFixed(3)} ` +
    `curlRms=${t.abosCurlRms.toFixed(4)} rFull=${rFull.toFixed(4)} rBeam=${rBeam.toFixed(4)} ms/f=${res.msPerFrame.toFixed(2)}`
  );
  assert(res.readyFrames > 0, `I8.4 never reached READY (state=${t.state} reason=0x${t.reason.toString(16)})`);
  assert(t.gridW === 256 && t.gridH === 128, `I8.4 expected 256x128 grid, got ${t.gridW}x${t.gridH}`);
  assert(t.stampValid, 'I8.4 wave2_telem[56] freshness stamp invalid');
  assert(rFull >= (PARITY_LITE ? 0.80 : 0.90), `I8.4 |Phi| correlation rFull=${rFull.toFixed(4)} below threshold`);
  assert(curlRatio < 0.35, `I8.4 curlRatio=${curlRatio.toFixed(3)} too high`);
  console.log('  [PASS] I8.4 Lock-In Acoustic BOS 1,200 Hz standing wavefield');
}

if (!PARITY_LITE) {
  // ============================================================================
  // I8.2 Acoustic Guitar Soundboard (Anisotropic Spruce + beta1 Soundhole)
  // ============================================================================
  {
    const res = runMp4Elasto({
      file: 'elasto_guitar_soundboard.mp4',
      mode: 1,
      dtRowUs: 35.0,
      exposureUs: 200.0,
      lockFreqs: [247.5, 495.0],
      maxFrames: 150,
      hasAudio: true,
    });
    const t = res.lastReady || res.last;
    const bridgeDist = Math.hypot(t.srcX - 392, t.srcY - 180);
    console.log(
      `  [INFO] I8.2 Guitar Soundboard: ready=${res.readyFrames}/${res.nFrames} state=${t.state} ` +
      `beta1=${t.beta1} valid=${t.validCells} alpha=${t.alpha.toFixed(3)} class=${t.dispClass} ` +
      `aniso=${t.anisotropy.toFixed(2)} bridge=(${t.srcX.toFixed(0)},${t.srcY.toFixed(0)}) [d=${bridgeDist.toFixed(0)}] ` +
      `medC=${t.medC.toFixed(0)} ms/f=${res.msPerFrame.toFixed(2)}`
    );
    assert(res.readyFrames > 0, `I8.2 never reached READY (state=${t.state} reason=0x${t.reason.toString(16)})`);
    assert(t.beta1 >= 1, `I8.2 expected beta1 >= 1 soundhole, got ${t.beta1}`);
    assert(t.anisotropy >= 1.25, `I8.2 expected anisotropy >= 1.25, got ${t.anisotropy.toFixed(2)}`);
    assert(t.dispClass === ELASTO_CLASS_KIRCHHOFF_PLATE, `I8.2 expected KIRCHHOFF_PLATE, got ${t.dispClass}`);
    assert(bridgeDist <= 115, `I8.2 bridge source (${t.srcX.toFixed(0)},${t.srcY.toFixed(0)}) too far from (392,180)`);
    console.log('  [PASS] I8.2 Acoustic Guitar Soundboard (beta1 soundhole + anisotropy + bridge source)');
  }

  // ============================================================================
  // I8.3 Gelatin Shear-Wave Phantom (Non-Dispersive alpha=1.0 + Stiff Marble)
  // ============================================================================
  {
    const res = runMp4Elasto({
      file: 'elasto_gelatin_phantom.mp4',
      mode: 1,
      dtRowUs: 35.0,
      exposureUs: 250.0,
      lockFreqs: [110.0, 220.0],
      maxFrames: 150,
      hasAudio: true,
    });
    const t = res.lastReady || res.last;
    const cIn = sampleRegionMedianC(325, 395, 145, 215);
    const cOut = sampleRegionMedianC(120, 250, 90, 270);
    const ratio = cOut > 0 ? cIn / cOut : 0;
    console.log(
      `  [INFO] I8.3 Gelatin Phantom: ready=${res.readyFrames}/${res.nFrames} state=${t.state} ` +
      `valid=${t.validCells} alpha=${t.alpha.toFixed(3)} class=${t.dispClass} ` +
      `cIn=${cIn.toFixed(0)} cOut=${cOut.toFixed(0)} ratio=${ratio.toFixed(2)} ms/f=${res.msPerFrame.toFixed(2)}`
    );
    assert(res.readyFrames > 0, `I8.3 never reached READY (state=${t.state} reason=0x${t.reason.toString(16)})`);
    assert(t.dispClass === ELASTO_CLASS_MEMBRANE_SHEAR, `I8.3 expected MEMBRANE_SHEAR (2), got ${t.dispClass} (alpha=${t.alpha})`);
    assert(t.alpha >= 0.80 && t.alpha <= 1.20, `I8.3 alpha=${t.alpha.toFixed(3)} out of [0.80, 1.20]`);
    assert(ratio >= 1.50, `I8.3 stiff marble ratio ${ratio.toFixed(2)} < 1.50`);
    console.log('  [PASS] I8.3 Gelatin Shear-Wave Phantom (non-dispersive alpha=1.0 + stiff marble)');
  }

  // ============================================================================
  // I8.5d Mode 2 Self-Locked RS Surface Elastography (no audio!)
  // ============================================================================
  {
    const res = runMp4Elasto({
      file: 'elasto_hidden_crack_panel.mp4',
      mode: 2,
      dtRowUs: 35.0,
      exposureUs: 200.0,
      lockFreqs: [380.0, 860.0],
      maxFrames: 95,
      hasAudio: false,
    });
    const t = res.lastReady || res.last;
    console.log(
      `  [INFO] I8.5d Mode 2 Self-Locked: ready=${res.readyFrames}/${res.nFrames} state=${t.state} ` +
      `valid=${t.validCells} medC=${t.medC.toFixed(0)} alpha=${t.alpha.toFixed(3)} coh=${t.meanCoh.toFixed(3)}`
    );
    assert(res.readyFrames > 0, `I8.5d Mode 2 never reached READY (state=${t.state} reason=0x${t.reason.toString(16)})`);
    assert(t.alpha >= 0.35 && t.alpha <= 0.65, `I8.5d Mode 2 alpha=${t.alpha.toFixed(3)} out of [0.35, 0.65]`);
    console.log('  [PASS] I8.5d Mode 2 Self-Locked RS Surface Elastography');
  }

  // ============================================================================
  // I8.5e Real-Camera Footage Mode 3 Smoke Checks
  // ============================================================================
  for (const clip of [
    'real_tuning_fork_ripples.mp4',
    'real_glass_faraday_waves.mp4',
    'real_chladni_plate_audio.mp4',
    'real_shear_wave_oce.mp4',
  ]) {
    if (!fs.existsSync(path.join(SAMPLES_DIR, clip))) continue;
    const res = runMp4Elasto({
      file: clip,
      mode: 3,
      maxFrames: 75,
      hasAudio: false,
    });
    const t = res.lastReady || res.last;
    console.log(
      `  [INFO] I8.5e Real-Camera ${clip}: ready=${res.readyFrames}/${res.nFrames} state=${t.state} ` +
      `reason=0x${t.reason.toString(16)} f0=${t.f0.toFixed(2)}Hz valid=${t.validCells} medC=${t.medC.toFixed(1)} ` +
      `coh=${t.meanCoh.toFixed(3)} ms/f=${res.msPerFrame.toFixed(2)}`
    );
    assert(
      t.state === ELASTO_STATE_READY || t.state === ELASTO_STATE_WARMING || t.state === ELASTO_STATE_REFUSED,
      `I8.5e unexpected state ${t.state} on ${clip}`
    );
  }
  console.log('  [PASS] I8.5e Real-Camera Mode 3 checks clean & finite');
}

console.log(`\nGate I8 ALL PASS (${PARITY_LITE ? 'parity-lite' : 'full'})`);
